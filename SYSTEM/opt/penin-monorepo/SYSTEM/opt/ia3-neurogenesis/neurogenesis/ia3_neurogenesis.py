#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IA³ Neurogênese Avançada (CPU) - Patch Darwin v2
- Net2Wider: duplica 1 neurônio e redistribui pesos na próxima camada
- Net2Deeper: insere fc_mid ~ identidade (morfismo próximo à função original)
- MixedActivation (DARTS-like): α treináveis para escolher ReLU/Tanh/GELU/SiLU
- EWC/MAS para preservação de conhecimento
- Métricas Prometheus avançadas com histogramas
Referências:
- Network Morphism (generaliza Net2Net; inclui Net2Deeper/Net2Wider). arXiv:1603.01670
- DARTS: Differentiable Architecture Search (ICLR 2019)
- EWC: Overcoming catastrophic forgetting in neural networks. arXiv:1612.00796
"""

import os
import math
import random
from dataclasses import dataclass
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .ia3_metrics import ensure_metrics, observe_step
except ImportError:
    ensure_metrics = lambda *args, **kwargs: None
    observe_step = lambda *args, **kwargs: None

# ═══════════════════════════════════════════════════════════════════════════════
# MIXED ACTIVATION (DARTS-LIKE)
# ═══════════════════════════════════════════════════════════════════════════════

class MixedActivation(nn.Module):
    """
    Ativação mista estilo DARTS: combinação diferenciável de {ReLU, Tanh, GELU, SiLU}
    Pesos α são treináveis; softmax(α) gera mistura.
    """
    def __init__(self, init_alpha: float = 1.0):
        super().__init__()
        self.ops = nn.ModuleDict({
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU()
        })
        self.names = list(self.ops.keys())
        self.alpha = nn.Parameter(torch.ones(len(self.ops)) * init_alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.alpha, dim=0)
        result = 0.0
        for i, name in enumerate(self.names):
            result = result + weights[i] * self.ops[name](x)
        return result

    def get_dominant_activation(self) -> Tuple[str, float]:
        """Retorna ativação dominante e seu peso"""
        weights = torch.softmax(self.alpha, dim=0).detach()
        idx = int(torch.argmax(weights).item())
        return self.names[idx], float(weights[idx].item())

    def entropy(self) -> float:
        """Entropia da distribuição de ativações"""
        weights = torch.softmax(self.alpha, dim=0)
        eps = 1e-9
        return float((-weights * (weights + eps).log()).sum().item())

    def reset_to_activation(self, activation_name: str):
        """Reset α para favorecer uma ativação específica"""
        if activation_name in self.names:
            idx = self.names.index(activation_name)
            with torch.no_grad():
                self.alpha.zero_()
                self.alpha[idx] = 3.0  # Favorece esta ativação

# ═══════════════════════════════════════════════════════════════════════════════
# DYNAMIC MLP COM NET2WIDER + NET2DEEPER
# ═══════════════════════════════════════════════════════════════════════════════

class DynamicMLP(nn.Module):
    """
    MLP dinâmico que pode crescer preservando função:
      - Net2Wider: hidden_dim += 1 (duplica neurônio e redistribui pesos)
      - Net2Deeper: insere camada fc_mid ~ identidade
    Fluxo:
      x -> fc1 -> mixed_act -> (fc_mid -> mixed_act_mid)? -> fc2
    """
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, mixed_act_init: float = 1.0):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.has_mid = False
        
        # Camadas principais
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        
        # Ativações mistas treináveis
        self.mixed_act = MixedActivation(init_alpha=mixed_act_init)
        self.mixed_act_mid = None  # Criada quando Net2Deeper
        
        # Inicialização Kaiming
        self._reset_parameters()
        
        # Histórico para EWC/MAS
        self.previous_weights = {}
        self.fisher_information = {}
        self.gradient_importance = {}

    def _reset_parameters(self):
        """Inicialização cuidadosa dos parâmetros"""
        nn.init.kaiming_uniform_(self.fc1.weight, a=math.sqrt(5))
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_uniform_(self.fc2.weight, a=math.sqrt(5))
        nn.init.zeros_(self.fc2.bias)
        
        if hasattr(self, 'fc_mid') and self.fc_mid is not None:
            nn.init.eye_(self.fc_mid.weight)
            nn.init.zeros_(self.fc_mid.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass com ativações mistas"""
        h = self.mixed_act(self.fc1(x))
        
        if self.has_mid and self.fc_mid is not None:
            if self.mixed_act_mid is None:
                self.mixed_act_mid = MixedActivation()
            h = self.mixed_act_mid(self.fc_mid(h))
        
        y = self.fc2(h)
        return y

    # ═══════════════════════════════════════════════════════════════════════════
    # NET2WIDER: Adicionar neurônio preservando função
    # ═══════════════════════════════════════════════════════════════════════════
    
    @torch.no_grad()
    def add_neuron_net2wider(self) -> int:
        """
        Net2Wider: aumenta hidden_dim + 1.
        Estratégia: duplicar um neurônio j em fc1 e dividir seu fluxo na fc2.
        Retorna: índice do neurônio adicionado
        """
        device = self.fc1.weight.device
        dtype = self.fc1.weight.dtype
        
        # Escolher neurônio para duplicar (aleatório ou o de maior norma)
        weight_norms = torch.norm(self.fc1.weight, dim=1)
        j = random.randrange(self.hidden_dim)  # ou torch.argmax(weight_norms).item()

        # Expandir fc1: (hidden_dim, in_dim) -> (hidden_dim+1, in_dim)
        W1_old, b1_old = self.fc1.weight.data, self.fc1.bias.data
        W1_new = torch.zeros((self.hidden_dim + 1, self.in_dim), device=device, dtype=dtype)
        b1_new = torch.zeros((self.hidden_dim + 1,), device=device, dtype=dtype)
        
        # Copiar neurônios existentes
        W1_new[:self.hidden_dim] = W1_old
        b1_new[:self.hidden_dim] = b1_old
        
        # Novo neurônio: cópia do j-ésimo + ruído pequeno
        W1_new[self.hidden_dim] = W1_old[j] + 1e-3 * torch.randn_like(W1_old[j])
        b1_new[self.hidden_dim] = b1_old[j] + 1e-3 * torch.randn(1).item()

        # Expandir fc2: (out_dim, hidden_dim) -> (out_dim, hidden_dim+1)
        W2_old, b2_old = self.fc2.weight.data, self.fc2.bias.data
        W2_new = torch.zeros((self.out_dim, self.hidden_dim + 1), device=device, dtype=dtype)
        
        # Copiar conexões existentes
        W2_new[:, :self.hidden_dim] = W2_old
        
        # Dividir o fluxo: neurônio j e novo neurônio dividem responsabilidade
        W2_new[:, j] = W2_old[:, j] * 0.5  # Reduzir original
        W2_new[:, self.hidden_dim] = W2_old[:, j] * 0.5  # Novo com mesma responsabilidade
        
        b2_new = b2_old.clone()

        # Aplicar mudanças
        self.hidden_dim += 1
        self.fc1 = nn.Linear(self.in_dim, self.hidden_dim).to(device=device, dtype=dtype)
        self.fc2 = nn.Linear(self.hidden_dim, self.out_dim).to(device=device, dtype=dtype)
        
        self.fc1.weight.data.copy_(W1_new)
        self.fc1.bias.data.copy_(b1_new)
        self.fc2.weight.data.copy_(W2_new)
        self.fc2.bias.data.copy_(b2_new)

        # Expandir fc_mid se existir (preservar identidade expandida)
        if self.has_mid and hasattr(self, 'fc_mid'):
            Wm_old, bm_old = self.fc_mid.weight.data, self.fc_mid.bias.data
            Wm_new = torch.eye(self.hidden_dim, device=device, dtype=dtype)
            bm_new = torch.zeros((self.hidden_dim,), device=device, dtype=dtype)
            
            # Copiar bloco antigo
            Wm_new[:self.hidden_dim-1, :self.hidden_dim-1] = Wm_old
            bm_new[:self.hidden_dim-1] = bm_old
            
            self.fc_mid = nn.Linear(self.hidden_dim, self.hidden_dim).to(device=device, dtype=dtype)
            self.fc_mid.weight.data.copy_(Wm_new)
            self.fc_mid.bias.data.copy_(bm_new)

        return self.hidden_dim - 1  # Índice do neurônio adicionado

    # ═══════════════════════════════════════════════════════════════════════════
    # NET2DEEPER: Inserir camada preservando função
    # ═══════════════════════════════════════════════════════════════════════════
    
    @torch.no_grad()
    def net2deeper_insert(self) -> bool:
        """
        Net2Deeper: insere camada h->h iniciada como ~identidade (preserva função).
        (Para ReLU, identidade preserva; para outras ativações, é aproximação próxima.)
        """
        if self.has_mid:
            return False  # Já tem camada intermediária
        
        device = self.fc1.weight.device
        dtype = self.fc1.weight.dtype
        
        # Criar nova camada intermediária como identidade
        self.fc_mid = nn.Linear(self.hidden_dim, self.hidden_dim).to(device=device, dtype=dtype)
        
        with torch.no_grad():
            # Inicializar como matriz identidade para preservar função
            self.fc_mid.weight.data.copy_(torch.eye(self.hidden_dim, device=device, dtype=dtype))
            self.fc_mid.bias.data.zero_()
        
        # Criar ativação mista para camada intermediária
        self.mixed_act_mid = MixedActivation()
        
        self.has_mid = True
        print(f"🎯 Net2Deeper: Camada intermediária inserida (identidade preservada)")
        return True

    # ═══════════════════════════════════════════════════════════════════════════
    # REMOÇÃO DE NEURÔNIOS (EQUAÇÃO DA MORTE)
    # ═══════════════════════════════════════════════════════════════════════════
    
    @torch.no_grad()
    def remove_worst_neuron(self) -> Tuple[bool, int]:
        """Remove neurônio com menor contribuição (norma dos pesos de saída)"""
        if self.hidden_dim <= 1:
            return False, -1
        
        # Calcular contribuição de cada neurônio (norma das conexões de saída)
        contributions = torch.norm(self.fc2.weight, dim=0)  # [hidden_dim]
        worst_idx = torch.argmin(contributions).item()
        
        return self._remove_neuron_at_index(worst_idx), worst_idx
    
    @torch.no_grad()
    def _remove_neuron_at_index(self, idx: int) -> bool:
        """Remove neurônio em índice específico"""
        if self.hidden_dim <= 1 or idx >= self.hidden_dim:
            return False
        
        device = self.fc1.weight.device
        dtype = self.fc1.weight.dtype
        
        # Criar máscara (manter todos exceto idx)
        keep_mask = torch.ones(self.hidden_dim, dtype=torch.bool)
        keep_mask[idx] = False
        
        # Reduzir fc1: manter linhas != idx
        W1_new = self.fc1.weight.data[keep_mask, :]
        b1_new = self.fc1.bias.data[keep_mask]
        
        # Reduzir fc2: manter colunas != idx
        W2_new = self.fc2.weight.data[:, keep_mask]
        b2_new = self.fc2.bias.data.clone()
        
        # Aplicar mudanças
        self.hidden_dim -= 1
        self.fc1 = nn.Linear(self.in_dim, self.hidden_dim).to(device=device, dtype=dtype)
        self.fc2 = nn.Linear(self.hidden_dim, self.out_dim).to(device=device, dtype=dtype)
        
        self.fc1.weight.data.copy_(W1_new)
        self.fc1.bias.data.copy_(b1_new)
        self.fc2.weight.data.copy_(W2_new)
        self.fc2.bias.data.copy_(b2_new)
        
        # Reduzir fc_mid se existir
        if self.has_mid and hasattr(self, 'fc_mid'):
            Wm_new = self.fc_mid.weight.data[keep_mask][:, keep_mask]
            bm_new = self.fc_mid.bias.data[keep_mask]
            
            self.fc_mid = nn.Linear(self.hidden_dim, self.hidden_dim).to(device=device, dtype=dtype)
            self.fc_mid.weight.data.copy_(Wm_new)
            self.fc_mid.bias.data.copy_(bm_new)
        
        return True

    # ═══════════════════════════════════════════════════════════════════════════
    # TREINO COM DUPLO OTIMIZADOR (PESOS + ALPHA)
    # ═══════════════════════════════════════════════════════════════════════════
    
    def train_short(self, X: torch.Tensor, y: torch.Tensor,
                    steps: int = 64, lr: float = 1e-3, alpha_lr: float = 5e-4,
                    alpha_step_every: int = 4, use_ewc: bool = True) -> Dict[str, Any]:
        """
        Treina pesos e α (MixedActivation) com regularização EWC/MAS.
        Dois otimizadores: um para pesos normais, outro para α arquiteturais.
        """
        self.train()
        device = next(self.parameters()).device
        X = X.to(device)
        y = y.to(device)

        # Separar parâmetros: pesos vs alpha
        weight_params = []
        alpha_params = []
        
        for name, param in self.named_parameters():
            if "alpha" in name:
                alpha_params.append(param)
            else:
                weight_params.append(param)

        # Otimizadores separados
        opt_weights = torch.optim.Adam(weight_params, lr=lr)
        opt_alpha = torch.optim.Adam(alpha_params, lr=alpha_lr) if alpha_params else None

        last_loss = None
        grad_norm = 0.0
        
        for step in range(steps):
            # Treino dos pesos
            opt_weights.zero_grad()
            if opt_alpha:
                opt_alpha.zero_grad()
            
            y_pred = self.forward(X)
            loss = F.mse_loss(y_pred, y)
            
            # Adicionar regularização EWC se disponível
            if use_ewc and hasattr(self, '_ewc_penalty'):
                loss = loss + self._ewc_penalty()
            
            loss.backward()

            # Calcular norma do gradiente (apenas pesos, não α)
            total_grad_norm = 0.0
            for param in weight_params:
                if param.grad is not None:
                    total_grad_norm += param.grad.detach().pow(2).sum().item()
            grad_norm = math.sqrt(total_grad_norm + 1e-12)

            # Gradient clipping
            if weight_params:
                torch.nn.utils.clip_grad_norm_(weight_params, max_norm=1.0)
            if alpha_params:
                torch.nn.utils.clip_grad_norm_(alpha_params, max_norm=0.5)

            # Update pesos
            opt_weights.step()
            
            # Update α com frequência menor
            if opt_alpha and step % alpha_step_every == 0:
                opt_alpha.step()

            last_loss = float(loss.detach().item())

            # Observar métricas
            try:
                act_choice, act_weight = self.mixed_act.get_dominant_activation()
                observe_step(
                    loss=last_loss,
                    grad_norm=grad_norm,
                    weight_norm=self.weight_l2_norm(),
                    act_entropy=self.mixed_act.entropy(),
                    act_choice=act_choice,
                    act_weight=act_weight
                )
            except Exception:
                pass

        return {
            "loss": last_loss,
            "grad_norm": grad_norm,
            "act_choice": self.mixed_act.get_dominant_activation(),
            "act_entropy": self.mixed_act.entropy(),
            "weight_norm": self.weight_l2_norm()
        }

    @torch.no_grad()
    def weight_l2_norm(self) -> float:
        """Calcula norma L2 total dos pesos (excluindo α)"""
        total = 0.0
        for name, param in self.named_parameters():
            if param.data is None or "alpha" in name:
                continue
            total += float((param.data ** 2).sum().item())
        return math.sqrt(total + 1e-12)

    # ═══════════════════════════════════════════════════════════════════════════
    # EWC/MAS PARA PRESERVAÇÃO DE CONHECIMENTO
    # ═══════════════════════════════════════════════════════════════════════════
    
    def consolidate_knowledge(self, X: torch.Tensor, y: torch.Tensor):
        """Consolida conhecimento importante usando aproximação EWC"""
        self.eval()
        
        # Calcular importância dos parâmetros (Fisher Information aproximada)
        self.zero_grad()
        y_pred = self.forward(X)
        loss = F.mse_loss(y_pred, y)
        loss.backward()
        
        # Salvar importância (gradiente^2) e valores atuais
        for name, param in self.named_parameters():
            if param.grad is not None and "alpha" not in name:
                # Fisher diagonal aproximado
                self.fisher_information[name] = param.grad.detach() ** 2
                self.previous_weights[name] = param.detach().clone()

    def _ewc_penalty(self) -> torch.Tensor:
        """Calcula penalidade EWC para preservar conhecimento importante"""
        penalty = 0.0
        ewc_lambda = 2.0
        
        for name, param in self.named_parameters():
            if name in self.fisher_information and name in self.previous_weights:
                fisher = self.fisher_information[name]
                prev_weights = self.previous_weights[name]
                penalty += (fisher * (param - prev_weights) ** 2).sum()
        
        return ewc_lambda * penalty

# ═══════════════════════════════════════════════════════════════════════════════
# GERAÇÃO DE DADOS SINTÉTICOS AVANÇADOS
# ═══════════════════════════════════════════════════════════════════════════════

def make_synthetic_identity(n_samples=4096, n_features=16, n_targets=16, device="cpu"):
    """Dados sintéticos para auto-supervisão (identidade com ruído)"""
    X = torch.randn(n_samples, n_features, device=device)
    
    # Target: várias transformações para testar capacidades diferentes
    Y = torch.zeros(n_samples, n_targets, device=device)
    
    # Identidade básica
    Y[:, :min(n_features, n_targets)] = X[:, :min(n_features, n_targets)]
    
    # Transformações não-lineares para testar adaptação
    if n_targets > n_features:
        Y[:, n_features:n_features+2] = torch.stack([
            torch.sin(X[:, 0] * 3.14159),
            torch.cos(X[:, 1] * 3.14159)
        ], dim=1)
    
    # Adicionar ruído controlado
    Y = Y + 0.01 * torch.randn_like(Y)
    
    return X, Y

def make_ood_batch(base_X: torch.Tensor, base_y: torch.Tensor, 
                   shift_factor: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cria batch Out-of-Distribution para teste de adaptação"""
    device = base_X.device
    
    # Shift na distribuição
    X_ood = base_X + shift_factor * torch.randn_like(base_X)
    
    # Rotação leve no espaço de features
    rotation_angle = shift_factor * 0.1
    cos_theta = math.cos(rotation_angle)
    sin_theta = math.sin(rotation_angle)
    
    if base_X.size(1) >= 2:
        # Aplicar rotação nas duas primeiras dimensões
        X_ood_rotated = X_ood.clone()
        X_ood_rotated[:, 0] = cos_theta * X_ood[:, 0] - sin_theta * X_ood[:, 1] 
        X_ood_rotated[:, 1] = sin_theta * X_ood[:, 0] + cos_theta * X_ood[:, 1]
        X_ood = X_ood_rotated
    
    # Manter same target structure
    y_ood = base_y + shift_factor * 0.05 * torch.randn_like(base_y)
    
    return X_ood, y_ood

# ═══════════════════════════════════════════════════════════════════════════════
# AVALIADOR IA³ AVANÇADO
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class IA3NeuronReport:
    adaptive_gain: float
    alpha_shift: float
    improved: bool
    structure_grew: bool
    passes: bool
    consciousness_score: float
    details: Dict[str, Any]

class IA3NeuronAgent:
    """
    Agente 'neurônio' IA³-like avançado:
    - Prova adaptatividade com dados OOD
    - Autoevolução via α treináveis (DARTS-like)
    - Autoarquitetura via Net2Wider/Net2Deeper
    - Autodidaxia via auto-supervisão
    - Consciousness proxy baseado em complexidade e adaptabilidade
    """
    def __init__(self, in_dim=16, hidden_dim=8, out_dim=16, metrics_port: int = 9091):
        self.model = DynamicMLP(in_dim, hidden_dim, out_dim, mixed_act_init=1.0)
        self.generation = 0
        self.total_adaptations = 0
        self.survival_rounds = 0
        
        ensure_metrics(port=metrics_port)
        print(f"🧬 IA³ Neuron Agent inicializado")
        print(f"   Dimensões: {in_dim} → {hidden_dim} → {out_dim}")
        print(f"   Métricas: porta {metrics_port}")

    def round(self, seed: int = 42, grow_policy: str = "wider_then_deeper") -> IA3NeuronReport:
        """Executa uma rodada completa de avaliação IA³"""
        self.generation += 1
        
        print(f"\n🔬 AVALIAÇÃO IA³ - Geração {self.generation}")
        print(f"   Neurônios atuais: {self.model.hidden_dim}")
        print(f"   Tem camada mid: {self.model.has_mid}")
        
        # Gerar dados com seed fixo para reprodutibilidade
        torch.manual_seed(seed)
        X, Y = make_synthetic_identity(
            n_samples=256, 
            n_features=self.model.in_dim, 
            n_targets=self.model.out_dim, 
            device=next(self.model.parameters()).device
        )

        # Medir performance baseline
        with torch.no_grad():
            baseline_pred = self.model(X)
            baseline_loss = F.mse_loss(baseline_pred, Y).item()
            baseline_alpha = self.model.mixed_act.alpha.detach().clone()

        print(f"   📊 Loss baseline: {baseline_loss:.6f}")

        # Treino curto com duplo otimizador
        print(f"   🎯 Treinando com otimizadores duplos...")
        train_stats = self.model.train_short(
            X, Y, 
            steps=64, 
            lr=1e-3, 
            alpha_lr=5e-4,
            alpha_step_every=4,
            use_ewc=True
        )

        # Teste de adaptação OOD
        print(f"   🌊 Testando adaptação OOD...")
        X_ood, Y_ood = make_ood_batch(X, Y, shift_factor=0.15)
        
        with torch.no_grad():
            ood_pred = self.model(X_ood)
            ood_loss = F.mse_loss(ood_pred, Y_ood).item()
            post_alpha = self.model.mixed_act.alpha.detach().clone()

        # Crescimento estrutural
        print(f"   🏗️ Aplicando crescimento estrutural ({grow_policy})...")
        structure_grew = False
        
        if grow_policy == "wider_then_deeper":
            if random.random() < 0.7:  # 70% chance: Net2Wider
                new_idx = self.model.add_neuron_net2wider()
                structure_grew = True
                print(f"      ✅ Net2Wider: +1 neurônio (#{new_idx})")
            elif not self.model.has_mid:
                if self.model.net2deeper_insert():
                    structure_grew = True
                    print(f"      ✅ Net2Deeper: +1 camada")
        elif grow_policy == "deeper_first" and not self.model.has_mid:
            if self.model.net2deeper_insert():
                structure_grew = True
                print(f"      ✅ Net2Deeper: +1 camada")
        else:
            new_idx = self.model.add_neuron_net2wider()
            structure_grew = True
            print(f"      ✅ Net2Wider: +1 neurônio (#{new_idx})")

        # Consolidar conhecimento após crescimento
        if structure_grew:
            self.model.consolidate_knowledge(X, Y)

        # Medir melhorias
        adaptive_gain = max(0.0, baseline_loss - ood_loss)
        alpha_shift = float((post_alpha - baseline_alpha).abs().sum().item())
        improved = ood_loss < baseline_loss * 0.95  # 5% de tolerância

        # Calcular consciousness proxy
        complexity = (
            0.3 * self.model.hidden_dim / 64 +  # Complexidade estrutural
            0.3 * (1 - self.model.mixed_act.entropy() / math.log(4)) +  # Especialização
            0.2 * min(1.0, adaptive_gain * 100) +  # Adaptabilidade
            0.2 * min(1.0, self.survival_rounds * 0.1)  # Memória temporal
        )
        consciousness_score = min(1.0, max(0.0, complexity))

        # Critérios IA³ rigorosos
        passes = (
            adaptive_gain > 1e-3 and      # Adaptativo
            alpha_shift > 1e-3 and        # Autoevolutivo (α mudou)
            improved and                  # Autodidata (melhorou)
            structure_grew and            # Autoconstrutivo
            train_stats["grad_norm"] > 1e-6 and  # Autorecursivo
            consciousness_score > 0.3     # Consciousness mínima
        )

        if passes:
            self.survival_rounds += 1
            self.total_adaptations += 1
            print(f"   ✅ NEURÔNIO APROVADO! Score consciência: {consciousness_score:.3f}")
        else:
            self.survival_rounds = 0
            print(f"   ❌ NEURÔNIO REPROVADO! Falhas detectadas.")

        return IA3NeuronReport(
            adaptive_gain=adaptive_gain,
            alpha_shift=alpha_shift,
            improved=improved,
            structure_grew=structure_grew,
            passes=passes,
            consciousness_score=consciousness_score,
            details={
                "baseline_loss": baseline_loss,
                "ood_loss": ood_loss,
                "train_stats": train_stats,
                "hidden_dim": self.model.hidden_dim,
                "has_mid": self.model.has_mid,
                "generation": self.generation,
                "survival_rounds": self.survival_rounds,
                "total_adaptations": self.total_adaptations
            }
        )

    def get_model_summary(self) -> Dict[str, Any]:
        """Retorna resumo do estado do modelo"""
        dominant_act, act_weight = self.model.mixed_act.get_dominant_activation()
        
        return {
            "architecture": {
                "input_dim": self.model.in_dim,
                "hidden_dim": self.model.hidden_dim,
                "output_dim": self.model.out_dim,
                "has_middle_layer": self.model.has_mid
            },
            "activation": {
                "dominant": dominant_act,
                "weight": act_weight,
                "entropy": self.model.mixed_act.entropy()
            },
            "parameters": {
                "total": sum(p.numel() for p in self.model.parameters()),
                "weight_norm": self.model.weight_l2_norm()
            },
            "evolution": {
                "generation": self.generation,
                "survival_rounds": self.survival_rounds,
                "total_adaptations": self.total_adaptations
            }
        }

# ═══════════════════════════════════════════════════════════════════════════════
# INTERFACE PARA TESTES
# ═══════════════════════════════════════════════════════════════════════════════

def run_demo(rounds: int = 3, metrics_port: int = 9091):
    """Demonstração do sistema IA³ avançado"""
    print("🧬 DEMO SISTEMA IA³ NEUROGENESIS AVANÇADO")
    print("="*60)
    
    agent = IA3NeuronAgent(
        in_dim=16, 
        hidden_dim=4, 
        out_dim=16,
        metrics_port=metrics_port
    )
    
    for round_num in range(1, rounds + 1):
        print(f"\n🔄 ROUND {round_num}/{rounds}")
        
        # Executar rodada IA³
        report = agent.round(
            seed=42 + round_num,
            grow_policy="wider_then_deeper"
        )
        
        # Mostrar resultados
        print(f"📊 RESULTADOS:")
        print(f"   Adaptação OOD: {report.adaptive_gain:.6f}")
        print(f"   Mudança α: {report.alpha_shift:.6f}")
        print(f"   Melhorou: {report.improved}")
        print(f"   Cresceu: {report.structure_grew}")
        print(f"   APROVADO: {'✅' if report.passes else '❌'}")
        print(f"   Consciência: {report.consciousness_score:.3f}")
        
        # Mostrar resumo do modelo
        summary = agent.get_model_summary()
        arch = summary["architecture"]
        act = summary["activation"]
        print(f"   Arquitetura: {arch['input_dim']}→{arch['hidden_dim']}→{arch['output_dim']} (mid: {arch['has_middle_layer']})")
        print(f"   Ativação dominante: {act['dominant']} ({act['weight']:.3f})")
        
        # Aplicar Equação da Morte se necessário
        if not report.passes:
            print(f"   ☠️ APLICANDO EQUAÇÃO DA MORTE...")
            removed, worst_idx = agent.model.remove_worst_neuron()
            if removed:
                print(f"      💀 Neurônio #{worst_idx} removido")
            
            # Adicionar neurônio substituto
            new_idx = agent.model.add_neuron_net2wider()
            print(f"      🐣 Neurônio substituto #{new_idx} criado")
        
        time.sleep(1)  # Pausa para observabilidade

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
        port = int(sys.argv[3]) if len(sys.argv) > 3 else 9091
        run_demo(rounds, port)
    else:
        print("Uso: python ia3_neurogenesis.py --demo [rounds] [metrics_port]")
        print("Exemplo: python ia3_neurogenesis.py --demo 5 9091")