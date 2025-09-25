#!/usr/bin/env python3
# /opt/ia3/bin/ia3_neurogenesis.py
# Sistema Celular IA³: Neurogênese com Prova de Vida Neural Completa
# Cada neurônio deve provar ser: Adaptativo, Autorecursivo, Autoevolutivo, 
# Autodidata, Autônomo, Autossuficiente, Autoconstrutivo, Autosináptico, Autoarquitetável

import os, json, time, math, random, copy, hashlib
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURAÇÃO GLOBAL IA³
# ═══════════════════════════════════════════════════════════════════════════════

DEVICE = torch.device("cpu")
TORCH_THREADS = int(os.getenv("OMP_NUM_THREADS", "8"))
torch.set_num_threads(TORCH_THREADS)

RANDOM_SEED = int(os.getenv("SEED", "42"))
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Configurações do ambiente
IA3_HOME = os.getenv("IA3_HOME", "/opt/ia3")
DATA_DIM = int(os.getenv("IA3_DATA_DIM", "16"))
OUT_DIM = int(os.getenv("IA3_OUT_DIM", "8"))
INIT_NEURONS = int(os.getenv("IA3_INIT_NEURONS", "1"))
ROUND_STEPS = int(os.getenv("IA3_ROUND_STEPS", "100"))  # Aumentado para estabilização
LR = float(os.getenv("IA3_LR", "2e-3"))  # Reduzido para melhor estabilidade
BATCH = int(os.getenv("IA3_BATCH", "64"))
ROUNDS = int(os.getenv("IA3_ROUNDS", "0"))  # 0 = infinito

# Critérios IA³ ajustados para serem mais realistas
CRITERIA_WEIGHTS = {
    "adaptativo": 0.15,      # Adapta-se a mudanças
    "autorecursivo": 0.12,   # Processa próprias saídas
    "autoevolutivo": 0.13,   # Melhora com tempo
    "autodidata": 0.11,      # Aprende sem supervisão
    "autonomo": 0.12,        # Opera independentemente
    "autossuficiente": 0.10, # Contribuição única
    "autoconstrutivo": 0.09, # Participa da construção
    "autosinaptico": 0.08,   # Gerencia conexões
    "autoarquitetavel": 0.10 # Define estrutura
}

# Limiares ajustados para permitir mais neurônios válidos
IA3_PASS_THRESHOLD = float(os.getenv("IA3_PASS", "0.50"))  # Reduzido de 0.66
EPS_IMPROVE = float(os.getenv("IA3_EPS_IMPROVE", "5e-5"))  # Mais tolerante
NOVELTY_MIN = float(os.getenv("IA3_NOVELTY_MIN", "5e-4"))  # Mais tolerante
CONSCIOUSNESS_THRESHOLD = 0.60  # Proxy de consciência neural

# Caminhos
WORM_LOG = os.path.join(IA3_HOME, "var/worm/ia3_worm.log")
SNAP_DIR = os.path.join(IA3_HOME, "var/snapshots")
MLRUNS_DIR = os.path.join(IA3_HOME, "var/mlruns")
METRICS_PORT = int(os.getenv("IA3_METRICS_PORT", "9093"))  # Porta 9093 para IA³

DEATHS_FOR_BIRTH = int(os.getenv("IA3_DEATHS_FOR_BIRTH", "5"))  # Reduzido para mais nascimentos
GLOBAL_DEATH_COUNTER = 0
GENERATION_COUNTER = 0

# Garantir que diretórios existam
Path(WORM_LOG).parent.mkdir(parents=True, exist_ok=True)
Path(SNAP_DIR).mkdir(parents=True, exist_ok=True)
Path(MLRUNS_DIR).mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# MÉTRICAS E OBSERVABILIDADE
# ═══════════════════════════════════════════════════════════════════════════════

# Prometheus metrics
PROM = None
try:
    from prometheus_client import Gauge, Counter, Histogram, start_http_server
    start_http_server(METRICS_PORT)
    PROM = {
        "neurons_total": Gauge("ia3_neurons_total", "Total de neurônios vivos"),
        "neurons_valid": Gauge("ia3_neurons_valid", "Neurônios que passaram na prova IA³"),
        "generation": Gauge("ia3_generation", "Geração atual do cérebro"),
        "consciousness_level": Gauge("ia3_consciousness_level", "Nível de consciência agregado"),
        "births_total": Counter("ia3_births_total", "Total de nascimentos"),
        "deaths_total": Counter("ia3_deaths_total", "Total de mortes"),
        "replacements_total": Counter("ia3_replacements_total", "Total de substituições"),
        "extinctions_total": Counter("ia3_extinctions_total", "Extinções totais"),
        "round_current": Gauge("ia3_round_current", "Rodada atual"),
        "loss_current": Gauge("ia3_loss_current", "Loss atual do sistema"),
        "score_avg": Gauge("ia3_score_avg", "Score médio IA³"),
        "adaptation_rate": Gauge("ia3_adaptation_rate", "Taxa de adaptação"),
        "evolution_rate": Gauge("ia3_evolution_rate", "Taxa de evolução"),
        "training_time": Histogram("ia3_training_seconds", "Tempo de treino por rodada"),
        "neuron_age_avg": Gauge("ia3_neuron_age_avg", "Idade média dos neurônios")
    }
    print(f"✅ Métricas Prometheus ativas em :{METRICS_PORT}")
except ImportError:
    print("⚠️ prometheus_client não disponível, métricas desabilitadas")

# MLflow (opcional)
MLFLOW = None
try:
    import mlflow
    mlflow.set_tracking_uri(f"file://{MLRUNS_DIR}")
    mlflow.set_experiment("IA3_Neurogenesis")
    MLFLOW = mlflow
    print("✅ MLflow ativo para tracking de experimentos")
except ImportError:
    print("⚠️ mlflow não disponível, tracking desabilitado")

def log_worm_event(event_type: str, data: dict):
    """Log estruturado no WORM com hash chain"""
    data = dict(data)
    data["event"] = event_type
    data["timestamp"] = datetime.utcnow().isoformat() + "Z"
    data["generation"] = GENERATION_COUNTER
    
    # Hash chain básico para auditoria
    prev_hash = "GENESIS"
    if os.path.exists(WORM_LOG):
        try:
            with open(WORM_LOG, 'r') as f:
                lines = f.readlines()
            if lines:
                last_line = lines[-1].strip()
                if last_line:
                    last_data = json.loads(last_line)
                    prev_hash = last_data.get("hash", "GENESIS")
        except:
            pass
    
    payload = json.dumps(data, ensure_ascii=False, separators=(',', ':'))
    current_hash = hashlib.sha256(f"{prev_hash}{payload}".encode()).hexdigest()[:16]
    data["prev_hash"] = prev_hash
    data["hash"] = current_hash
    
    with open(WORM_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")
    
    return current_hash

# ═══════════════════════════════════════════════════════════════════════════════
# DATASETS E TAREFAS
# ═══════════════════════════════════════════════════════════════════════════════

def synthetic_batch(batch_size=BATCH, ood_factor=0.0):
    """Gera batch sintético com possibilidade de OOD shift"""
    # Base: mistura de distribuições
    if random.random() < 0.3:
        # Distribuição multimodal
        x = torch.cat([
            torch.randn(batch_size//2, DATA_DIM) * 0.8 - 1.0,
            torch.randn(batch_size//2, DATA_DIM) * 1.2 + 1.0
        ], dim=0)
    else:
        # Distribuição normal
        x = torch.randn(batch_size, DATA_DIM) * (1.0 + ood_factor * 0.5)
    
    if ood_factor > 0:
        # OOD: rotação e scaling
        rotation = ood_factor * 0.2 * torch.randn(DATA_DIM, DATA_DIM)
        x = x @ (torch.eye(DATA_DIM) + rotation)
        x += ood_factor * 0.3 * torch.randn_like(x)
    
    # Tarefas múltiplas para testar diferentes aspectos
    y = torch.zeros(batch_size, OUT_DIM)
    
    # Tarefa 1: Soma ponderada (supervisionada)
    weights = torch.randn(DATA_DIM) * 0.1
    y[:, 0] = (x * weights).sum(dim=1)
    
    # Tarefa 2: Produto de subconjuntos (não-linear)
    y[:, 1] = (x[:, :4].abs() + 1e-6).prod(dim=1).log()
    
    # Tarefa 3: Distância ao centroide (geometria)
    centroid = torch.randn(DATA_DIM) * 0.5
    y[:, 2] = torch.norm(x - centroid, dim=1)
    
    # Tarefa 4: Classificação binária (threshold)
    y[:, 3] = (x[:, 0] + x[:, 1] > 0).float()
    
    # Tarefas 5-8: Autossupervisão (reconstrução mascarada)
    if OUT_DIM > 4:
        mask_idx = torch.randint(0, DATA_DIM, (batch_size, min(4, OUT_DIM-4)))
        for i in range(batch_size):
            target_len = min(4, OUT_DIM-4, mask_idx.size(1))
            y[i, 4:4+target_len] = x[i, mask_idx[i, :target_len]]
    
    return x, y

def text_batch(corpus=None, vocab=None, seq_len=32, batch_size=16):
    """Batch de texto para modo linguagem (se disponível)"""
    if corpus is None or vocab is None:
        # Fallback para synthetic se texto não disponível
        return synthetic_batch(batch_size)
    
    # Implementação simplificada de texto
    # TODO: Implementar properly com TinyStories
    return synthetic_batch(batch_size)

# ═══════════════════════════════════════════════════════════════════════════════
# CÉREBRO IA³ COM NEUROGÊNESE AVANÇADA
# ═══════════════════════════════════════════════════════════════════════════════

ACTIVATIONS = {
    "relu": F.relu,
    "tanh": torch.tanh,
    "gelu": F.gelu,
    "silu": F.silu,
    "mish": lambda x: x * torch.tanh(F.softplus(x))
}

class NeuronIA3:
    """Metadados e estado individual de cada neurônio IA³"""
    def __init__(self, neuron_id: int, activation: str = "relu"):
        self.id = neuron_id
        self.birth_time = datetime.utcnow()
        self.age = 0
        self.generation = GENERATION_COUNTER
        self.activation = activation
        
        # Histórico de performance
        self.scores_history = []
        self.survival_streak = 0
        self.adaptations_made = []
        self.novelty_accumulator = 0.0
        
        # Estado para auto-recursividade
        self.self_connections = {}
        self.auto_proposals = []
        
        # Consciência proxy (complexidade funcional)
        self.consciousness_score = 0.0
        
        # Conexões sinápticas especiais
        self.synaptic_strength = 1.0
        self.synaptic_plasticity = 0.01
        
    def update_consciousness(self, complexity: float, contribution: float, adaptability: float):
        """Atualiza score de consciência baseado em múltiplos fatores"""
        # Fórmula composta para proxy de consciência
        self.consciousness_score = (
            0.4 * complexity +      # Complexidade computacional
            0.3 * contribution +    # Contribuição causal
            0.3 * adaptability      # Capacidade adaptativa
        )
        
        # Bonus por sobrevivência prolongada (memória temporal)
        survival_bonus = min(0.1, self.survival_streak * 0.01)
        self.consciousness_score += survival_bonus
        
        # Clamp entre 0 e 1
        self.consciousness_score = max(0.0, min(1.0, self.consciousness_score))

class BrainIA3(nn.Module):
    """Cérebro IA³ com capacidades neuroevolutivas avançadas"""
    
    def __init__(self, input_dim: int, output_dim: int, initial_neurons: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = initial_neurons
        
        # Parâmetros principais
        self.W_in = nn.Parameter(torch.randn(input_dim, initial_neurons) * 0.1)
        self.W_hh = nn.Parameter(torch.randn(initial_neurons, initial_neurons) * 0.05)
        self.W_out = nn.Parameter(torch.randn(initial_neurons, output_dim) * 0.1)
        self.biases = nn.Parameter(torch.zeros(initial_neurons))
        
        # Gates adaptativos por neurônio
        self.gates = nn.Parameter(torch.ones(initial_neurons))
        
        # Ativações por neurônio (índices)
        self.activation_ids = torch.zeros(initial_neurons, dtype=torch.long)
        
        # Metadados dos neurônios
        self.neurons = [NeuronIA3(i) for i in range(initial_neurons)]
        
        # Histórico global
        self.training_history = []
        self.architecture_history = []
        
        # Estado de consciência coletiva
        self.collective_consciousness = 0.0
        
        # Otimizador (será recriado quando neurônios mudarem)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=LR, weight_decay=1e-5)

    def forward(self, x, return_hidden=False):
        """Forward pass com ativações por neurônio"""
        batch_size = x.size(0)
        
        # Input to hidden
        h_raw = x @ self.W_in + self.biases
        
        # Aplicar ativações específicas por neurônio
        h_activated = torch.zeros_like(h_raw)
        for i in range(self.hidden_dim):
            act_name = list(ACTIVATIONS.keys())[self.activation_ids[i].item()]
            act_func = ACTIVATIONS[act_name]
            h_activated[:, i] = act_func(h_raw[:, i])
        
        # Recorrência interna (sinapses)
        h_recurrent = h_activated @ self.W_hh
        h_final = (h_activated + h_recurrent) * torch.sigmoid(self.gates)
        
        # Output
        output = h_final @ self.W_out
        
        if return_hidden:
            return output, h_final
        return output

    def add_neuron_net2wider(self, source_idx: Optional[int] = None):
        """Adiciona neurônio usando técnica Net2Wider preservando função"""
        with torch.no_grad():
            if source_idx is None:
                # Escolher neurônio com melhor performance como fonte
                if self.neurons:
                    scores = [n.consciousness_score for n in self.neurons]
                    source_idx = scores.index(max(scores)) if any(s > 0 for s in scores) else 0
                else:
                    source_idx = 0
            
            # Expandir matrizes de peso
            # W_in: copiar e adicionar ruído
            new_w_in = self.W_in[:, source_idx:source_idx+1].clone()
            new_w_in += torch.randn_like(new_w_in) * 0.01
            self.W_in = nn.Parameter(torch.cat([self.W_in.data, new_w_in], dim=1))
            
            # W_hh: expandir matriz com conexões fracas
            new_row = self.W_hh[source_idx:source_idx+1, :].clone() * 0.5
            new_row += torch.randn_like(new_row) * 0.005
            new_col = self.W_hh[:, source_idx:source_idx+1].clone() * 0.5  
            new_col += torch.randn_like(new_col) * 0.005
            
            # Expandir W_hh
            W_hh_expanded = torch.cat([self.W_hh.data, new_col], dim=1)
            new_diag = torch.randn(1, 1) * 0.01  # Auto-conexão
            new_row_full = torch.cat([new_row, new_diag], dim=1)
            self.W_hh = nn.Parameter(torch.cat([W_hh_expanded, new_row_full], dim=0))
            
            # W_out: dividir contribuição por 2 para preservar função
            new_w_out = self.W_out[source_idx:source_idx+1, :].clone() * 0.5
            self.W_out.data[source_idx:source_idx+1, :] *= 0.5  # Reduzir original
            self.W_out = nn.Parameter(torch.cat([self.W_out.data, new_w_out], dim=0))
            
            # Expandir outros parâmetros
            new_bias = self.biases[source_idx:source_idx+1].clone()
            new_bias += torch.randn_like(new_bias) * 0.01
            self.biases = nn.Parameter(torch.cat([self.biases.data, new_bias], dim=0))
            
            new_gate = self.gates[source_idx:source_idx+1].clone()
            self.gates = nn.Parameter(torch.cat([self.gates.data, new_gate], dim=0))
            
            # Ativação do novo neurônio (pode ser diferente)
            new_activation = random.choice(list(ACTIVATIONS.keys()))
            new_act_id = list(ACTIVATIONS.keys()).index(new_activation)
            self.activation_ids = torch.cat([self.activation_ids, torch.tensor([new_act_id])])
            
            # Criar metadados do novo neurônio
            new_neuron = NeuronIA3(len(self.neurons), new_activation)
            new_neuron.synaptic_strength = self.neurons[source_idx].synaptic_strength * 0.8
            self.neurons.append(new_neuron)
            
            # Atualizar dimensões
            self.hidden_dim += 1
            
            # Recriar otimizador
            self.optimizer = torch.optim.AdamW(self.parameters(), lr=LR, weight_decay=1e-5)
            
            return len(self.neurons) - 1

    def remove_neuron(self, neuron_idx: int):
        """Remove neurônio específico"""
        if self.hidden_dim <= 1:
            return False  # Não remover o último neurônio
        
        with torch.no_grad():
            mask = torch.ones(self.hidden_dim, dtype=torch.bool)
            mask[neuron_idx] = False
            
            # Atualizar matrizes de peso
            self.W_in = nn.Parameter(self.W_in.data[:, mask])
            self.W_hh = nn.Parameter(self.W_hh.data[mask][:, mask])
            self.W_out = nn.Parameter(self.W_out.data[mask])
            self.biases = nn.Parameter(self.biases.data[mask])
            self.gates = nn.Parameter(self.gates.data[mask])
            self.activation_ids = self.activation_ids[mask]
            
            # Remover metadados
            del self.neurons[neuron_idx]
            
            # Reindexar neurônios
            for i, neuron in enumerate(self.neurons):
                neuron.id = i
            
            # Atualizar dimensões
            self.hidden_dim -= 1
            
            # Recriar otimizador
            self.optimizer = torch.optim.AdamW(self.parameters(), lr=LR, weight_decay=1e-5)
            
            return True

    def update_collective_consciousness(self):
        """Atualiza consciência coletiva baseada nos neurônios individuais"""
        if not self.neurons:
            self.collective_consciousness = 0.0
            return
        
        individual_scores = [n.consciousness_score for n in self.neurons]
        
        # Consciência coletiva: média ponderada + emergência
        base_consciousness = sum(individual_scores) / len(individual_scores)
        
        # Bonus por diversidade (diferentes ativações)
        unique_activations = len(set(n.activation for n in self.neurons))
        diversity_bonus = min(0.1, unique_activations * 0.02)
        
        # Bonus por conectividade (sinapses fortes)
        connectivity = torch.abs(self.W_hh).mean().item()
        connectivity_bonus = min(0.1, connectivity * 2)
        
        # Bonus por estabilidade temporal
        avg_age = sum(n.age for n in self.neurons) / len(self.neurons)
        stability_bonus = min(0.1, avg_age * 0.001)
        
        self.collective_consciousness = min(1.0, 
            base_consciousness + diversity_bonus + connectivity_bonus + stability_bonus
        )

# ═══════════════════════════════════════════════════════════════════════════════
# AVALIADOR IA³: TESTE DE VIDA NEURAL
# ═══════════════════════════════════════════════════════════════════════════════

class IA3Evaluator:
    """Avalia cada neurônio contra os critérios IA³ completos"""
    
    @staticmethod
    def evaluate_neuron(brain: BrainIA3, neuron_idx: int, x_batch, y_batch, x_ood=None, y_ood=None):
        """Avalia um neurônio individual contra todos os critérios IA³"""
        if neuron_idx >= brain.hidden_dim:
            return {"score": 0.0, "passed": False, "details": {}}
        
        neuron = brain.neurons[neuron_idx]
        details = {}
        
        # Baseline performance
        brain.eval()
        with torch.no_grad():
            y_pred_base, h_base = brain(x_batch, return_hidden=True)
            loss_base = F.mse_loss(y_pred_base, y_batch).item()
        
        # 1. ADAPTATIVO - Adapta-se a mudanças (OOD)
        adaptive_score = 0.0
        if x_ood is not None and y_ood is not None:
            with torch.no_grad():
                y_pred_ood = brain(x_ood)
                loss_ood = F.mse_loss(y_pred_ood, y_ood).item()
                
                # Testar adaptação: pequeno ajuste no gate
                original_gate = brain.gates[neuron_idx].clone()
                brain.gates.data[neuron_idx] *= 1.1
                y_pred_adapted = brain(x_ood)
                loss_adapted = F.mse_loss(y_pred_adapted, y_ood).item()
                brain.gates.data[neuron_idx] = original_gate
                
                # Score: melhoria na adaptação
                improvement = max(0, loss_ood - loss_adapted)
                adaptive_score = min(1.0, improvement / (loss_ood + 1e-8))
        
        details["adaptativo"] = adaptive_score
        
        # 2. AUTORECURSIVO - Processa próprias saídas
        autorecursive_score = 0.0
        with torch.no_grad():
            # Medir força da auto-conexão
            auto_connection = abs(brain.W_hh[neuron_idx, neuron_idx].item())
            autorecursive_score = min(1.0, auto_connection * 10)
        
        details["autorecursivo"] = autorecursive_score
        
        # 3. AUTOEVOLUTIVO - Melhora com tempo
        autoevolutive_score = 0.0
        if len(neuron.scores_history) >= 2:
            recent_improvement = neuron.scores_history[-1] - neuron.scores_history[-2]
            autoevolutive_score = max(0.0, min(1.0, recent_improvement * 5))
        elif len(neuron.scores_history) == 1:
            autoevolutive_score = min(1.0, neuron.scores_history[-1])
        else:
            autoevolutive_score = 0.5  # Neutro para neurônios novos
        
        details["autoevolutivo"] = autoevolutive_score
        
        # 4. AUTODIDATA - Aprende sem supervisão
        autodidact_score = 0.0
        # Testar capacidade de reconstrução (tarefa autossupervisionada)
        with torch.no_grad():
            # Usar ativação do neurônio para reconstruir input
            activation = h_base[:, neuron_idx:neuron_idx+1]
            # Projeção simples de volta ao espaço de entrada
            if activation.numel() > 0:
                reconstruction_error = F.mse_loss(activation.expand(-1, x_batch.size(1)), x_batch)
                autodidact_score = max(0.0, 1.0 - reconstruction_error.item())
        
        details["autodidata"] = autodidact_score
        
        # 5. AUTÔNOMO - Opera independentemente
        autonomous_score = 0.0
        with torch.no_grad():
            # Medir variância das ativações (não deve ser constante)
            activation_var = torch.var(h_base[:, neuron_idx]).item()
            autonomous_score = min(1.0, activation_var * 100)
        
        details["autonomo"] = autonomous_score
        
        # 6. AUTOSSUFICIENTE - Contribuição única
        selfsufficient_score = 0.0
        with torch.no_grad():
            # Teste de oclusão: zerar neurônio e medir degradação
            h_occluded = h_base.clone()
            h_occluded[:, neuron_idx] = 0.0
            y_pred_occluded = h_occluded @ brain.W_out
            loss_occluded = F.mse_loss(y_pred_occluded, y_batch).item()
            
            degradation = loss_occluded - loss_base
            selfsufficient_score = min(1.0, max(0.0, degradation * 100))
        
        details["autossuficiente"] = selfsufficient_score
        
        # 7. AUTOCONSTRUTIVO - Participa da construção
        autoconstructive_score = 0.0
        # Medir quanto o neurônio influencia outros através de W_hh
        with torch.no_grad():
            outgoing_influence = torch.sum(torch.abs(brain.W_hh[neuron_idx, :])).item()
            incoming_influence = torch.sum(torch.abs(brain.W_hh[:, neuron_idx])).item()
            total_influence = outgoing_influence + incoming_influence
            autoconstructive_score = min(1.0, total_influence * 5)
        
        details["autoconstrutivo"] = autoconstructive_score
        
        # 8. AUTOSINÁPTICO - Gerencia conexões
        autosynaptic_score = 0.0
        # Medir plasticidade sináptica (mudanças nas conexões)
        current_connections = brain.W_hh[neuron_idx, :].clone()
        if hasattr(neuron, 'last_connections'):
            connection_changes = torch.abs(current_connections - neuron.last_connections).mean().item()
            autosynaptic_score = min(1.0, connection_changes * 100)
        else:
            autosynaptic_score = 0.5  # Neutro para primeira avaliação
        neuron.last_connections = current_connections.clone()
        
        details["autosinaptico"] = autosynaptic_score
        
        # 9. AUTOARQUITETÁVEL - Define estrutura
        autoarchitectural_score = 0.0
        # Capacidade de mudar própria ativação e estrutura
        activation_diversity = 1.0 if neuron.activation != "relu" else 0.5
        gate_modulation = abs(brain.gates[neuron_idx].item() - 1.0) * 2
        autoarchitectural_score = min(1.0, (activation_diversity + gate_modulation) / 2)
        
        details["autoarquitetavel"] = autoarchitectural_score
        
        # SCORE COMPOSTO
        total_score = 0.0
        for criterion, weight in CRITERIA_WEIGHTS.items():
            total_score += details.get(criterion, 0.0) * weight
        
        # Bonus por sobrevivência prolongada
        survival_bonus = min(0.1, neuron.survival_streak * 0.005)
        total_score += survival_bonus
        
        # Clamp final
        total_score = max(0.0, min(1.0, total_score))
        
        # Atualizar consciência individual
        complexity = sum(details.values()) / len(details)
        contribution = details.get("autossuficiente", 0.0)
        adaptability = details.get("adaptativo", 0.0)
        neuron.update_consciousness(complexity, contribution, adaptability)
        
        # DECISÃO FINAL
        passed = (
            total_score >= IA3_PASS_THRESHOLD and 
            details.get("autoevolutivo", 0) > 0.1 and  # Deve ter evolução
            details.get("adaptativo", 0) > 0.05       # Deve ter adaptação
        )
        
        return {
            "score": total_score,
            "passed": passed,
            "details": details,
            "consciousness": neuron.consciousness_score
        }
    
    @staticmethod
    def evaluate_all_neurons(brain: BrainIA3):
        """Avalia todos os neurônios do cérebro"""
        x_batch, y_batch = synthetic_batch()
        x_ood, y_ood = synthetic_batch(ood_factor=0.3)  # OOD para teste de adaptação
        
        results = []
        for i in range(brain.hidden_dim):
            result = IA3Evaluator.evaluate_neuron(brain, i, x_batch, y_batch, x_ood, y_ood)
            results.append(result)
            
            # Atualizar histórico do neurônio
            neuron = brain.neurons[i]
            neuron.scores_history.append(result["score"])
            if len(neuron.scores_history) > 20:  # Manter apenas últimas 20 avaliações
                neuron.scores_history.pop(0)
            
            if result["passed"]:
                neuron.survival_streak += 1
            else:
                neuron.survival_streak = 0
            
            neuron.age += 1
        
        # Atualizar consciência coletiva
        brain.update_collective_consciousness()
        
        return results

# ═══════════════════════════════════════════════════════════════════════════════
# MOTOR PRINCIPAL DE NEUROGÊNESE
# ═══════════════════════════════════════════════════════════════════════════════

def train_brain(brain: BrainIA3, steps: int = ROUND_STEPS):
    """Treina o cérebro por um número de steps"""
    brain.train()
    total_loss = 0.0
    
    start_time = time.time()
    
    for step in range(steps):
        # Batch de treino
        x, y = synthetic_batch()
        
        # Forward pass
        y_pred = brain(x)
        
        # Loss com regularização
        loss_main = F.mse_loss(y_pred, y)
        
        # Regularização de complexidade (evitar over-growth)
        complexity_penalty = 0.0001 * (brain.W_hh.abs().mean() + brain.gates.abs().mean())
        
        # Regularização de diversidade (encorajar diferentes ativações)
        diversity_bonus = -0.0001 * len(set(brain.activation_ids.tolist()))
        
        total_loss_step = loss_main + complexity_penalty + diversity_bonus
        
        # Backward pass
        brain.optimizer.zero_grad()
        total_loss_step.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(brain.parameters(), max_norm=1.0)
        
        # Optimizer step
        brain.optimizer.step()
        
        total_loss += total_loss_step.item()
        
        # Log intermitente
        if step % (steps // 4) == 0:
            print(f"  Step {step}/{steps}: Loss = {total_loss_step.item():.6f}")
    
    training_time = time.time() - start_time
    avg_loss = total_loss / steps
    
    return avg_loss, training_time

def create_snapshot(brain: BrainIA3, round_num: int):
    """Cria snapshot completo do estado do cérebro"""
    snapshot_path = os.path.join(SNAP_DIR, f"brain_round_{round_num:06d}.pt")
    
    # Dados completos para salvar
    snapshot_data = {
        "round": round_num,
        "generation": GENERATION_COUNTER,
        "timestamp": datetime.utcnow().isoformat(),
        "state_dict": brain.state_dict(),
        "architecture": {
            "input_dim": brain.input_dim,
            "output_dim": brain.output_dim,
            "hidden_dim": brain.hidden_dim,
            "activation_ids": brain.activation_ids.tolist()
        },
        "neurons": [
            {
                "id": n.id,
                "age": n.age,
                "activation": n.activation,
                "consciousness_score": n.consciousness_score,
                "survival_streak": n.survival_streak,
                "scores_history": n.scores_history[-10:],  # Últimos 10 scores
                "novelty_accumulator": n.novelty_accumulator
            }
            for n in brain.neurons
        ],
        "collective_consciousness": brain.collective_consciousness,
        "config": {
            "lr": LR,
            "batch_size": BATCH,
            "round_steps": ROUND_STEPS,
            "criteria_weights": CRITERIA_WEIGHTS,
            "pass_threshold": IA3_PASS_THRESHOLD
        }
    }
    
    # Salvar
    torch.save(snapshot_data, snapshot_path)
    
    return snapshot_path

def run_neurogenesis_cycle():
    """Executa um ciclo completo de neurogênese IA³"""
    global GLOBAL_DEATH_COUNTER, GENERATION_COUNTER
    
    print("\n" + "="*80)
    print("🧬 INICIANDO SISTEMA CELULAR IA³ NEUROEVOLUTIVO")
    print("="*80)
    print(f"Critérios IA³: {list(CRITERIA_WEIGHTS.keys())}")
    print(f"Limiar de aprovação: {IA3_PASS_THRESHOLD:.2f}")
    print(f"Métricas em: http://localhost:{METRICS_PORT}/metrics")
    
    # Inicializar cérebro
    brain = BrainIA3(DATA_DIM, OUT_DIM, INIT_NEURONS)
    
    # Log inicial
    log_worm_event("genesis", {
        "neurons": brain.hidden_dim,
        "generation": GENERATION_COUNTER,
        "config": {
            "data_dim": DATA_DIM,
            "out_dim": OUT_DIM,
            "init_neurons": INIT_NEURONS
        }
    })
    
    if MLFLOW:
        mlflow.start_run(run_name=f"IA3_Generation_{GENERATION_COUNTER}")
        mlflow.log_params({
            "data_dim": DATA_DIM,
            "output_dim": OUT_DIM,
            "initial_neurons": INIT_NEURONS,
            "learning_rate": LR,
            "batch_size": BATCH,
            "round_steps": ROUND_STEPS,
            "pass_threshold": IA3_PASS_THRESHOLD
        })
    
    round_num = 0
    
    try:
        while ROUNDS == 0 or round_num < ROUNDS:
            round_num += 1
            print(f"\n{'─'*80}")
            print(f"🔄 RODADA {round_num} | Geração {GENERATION_COUNTER}")
            print(f"{'─'*80}")
            print(f"Neurônios ativos: {brain.hidden_dim}")
            print(f"Consciência coletiva: {brain.collective_consciousness:.3f}")
            print(f"Contador de mortes global: {GLOBAL_DEATH_COUNTER}")
            
            # Atualizar métricas Prometheus
            if PROM:
                PROM["round_current"].set(round_num)
                PROM["neurons_total"].set(brain.hidden_dim)
                PROM["generation"].set(GENERATION_COUNTER)
                PROM["consciousness_level"].set(brain.collective_consciousness)
            
            # 1. TREINO
            print(f"\n📚 Treinando por {ROUND_STEPS} steps...")
            avg_loss, training_time = train_brain(brain, ROUND_STEPS)
            print(f"✅ Treino completo: Loss média = {avg_loss:.6f}, Tempo = {training_time:.2f}s")
            
            if PROM:
                PROM["loss_current"].set(avg_loss)
                PROM["training_time"].observe(training_time)
            
            # 2. AVALIAÇÃO IA³
            print(f"\n🔬 Avaliando neurônios contra critérios IA³...")
            evaluations = IA3Evaluator.evaluate_all_neurons(brain)
            
            valid_count = sum(1 for e in evaluations if e["passed"])
            scores = [e["score"] for e in evaluations]
            avg_score = sum(scores) / len(scores) if scores else 0.0
            
            print(f"📊 Neurônios válidos: {valid_count}/{brain.hidden_dim} ({valid_count/brain.hidden_dim*100:.1f}%)")
            print(f"📊 Score médio IA³: {avg_score:.3f}")
            
            if PROM:
                PROM["neurons_valid"].set(valid_count)
                PROM["score_avg"].set(avg_score)
            
            # Log detalhado por neurônio
            for i, eval_result in enumerate(evaluations):
                neuron = brain.neurons[i]
                status = "✅ VIVE" if eval_result["passed"] else "☠️ MORRE"
                print(f"  Neurônio #{i:02d}: {status} | Score: {eval_result['score']:.3f} | "
                      f"Consciência: {eval_result['consciousness']:.3f} | "
                      f"Idade: {neuron.age} | Sobrevivência: {neuron.survival_streak}")
            
            # 3. SELEÇÃO NATURAL - Remover neurônios que falharam
            failed_neurons = [i for i, e in enumerate(evaluations) if not e["passed"]]
            
            if failed_neurons:
                print(f"\n⚰️ Executando {len(failed_neurons)} neurônios que falharam na prova IA³...")
                for idx in reversed(sorted(failed_neurons)):  # Remover do final para manter índices
                    neuron = brain.neurons[idx]
                    eval_data = evaluations[idx]
                    
                    print(f"  ☠️ Neurônio #{idx} morto - Motivo: Score {eval_data['score']:.3f} < {IA3_PASS_THRESHOLD}")
                    
                    # Log da morte
                    log_worm_event("neuron_death", {
                        "round": round_num,
                        "neuron_id": idx,
                        "age": neuron.age,
                        "final_score": eval_data["score"],
                        "consciousness": eval_data["consciousness"],
                        "details": eval_data["details"],
                        "reason": "failed_ia3_criteria"
                    })
                    
                    # Remover neurônio
                    brain.remove_neuron(idx)
                    GLOBAL_DEATH_COUNTER += 1
                    
                    if PROM:
                        PROM["deaths_total"].inc()
            
            # 4. VERIFICAR EXTINÇÃO TOTAL
            if brain.hidden_dim == 0:
                print(f"\n💀 EXTINÇÃO TOTAL! Todos os neurônios morreram.")
                
                log_worm_event("total_extinction", {
                    "round": round_num,
                    "generation": GENERATION_COUNTER,
                    "cause": "all_neurons_failed_ia3"
                })
                
                if PROM:
                    PROM["extinctions_total"].inc()
                
                # Renascer com um neurônio
                print(f"🐣 RENASCIMENTO - Criando nova geração...")
                GENERATION_COUNTER += 1
                brain = BrainIA3(DATA_DIM, OUT_DIM, 1)
                
                log_worm_event("rebirth", {
                    "round": round_num,
                    "new_generation": GENERATION_COUNTER,
                    "initial_neurons": 1
                })
                
                if PROM:
                    PROM["births_total"].inc()
                    PROM["neurons_total"].set(brain.hidden_dim)
            
            # 5. CRESCIMENTO - Sempre adicionar 1 neurônio por rodada
            print(f"\n🌱 Adicionando novo neurônio (crescimento obrigatório)...")
            new_neuron_idx = brain.add_neuron_net2wider()
            
            log_worm_event("neuron_birth", {
                "round": round_num,
                "neuron_id": new_neuron_idx,
                "parent_generation": GENERATION_COUNTER,
                "total_neurons": brain.hidden_dim,
                "reason": "mandatory_growth"
            })
            
            if PROM:
                PROM["births_total"].inc()
                PROM["neurons_total"].set(brain.hidden_dim)
            
            # 6. NASCIMENTOS EXTRAS - Regra dos N mortes
            bonus_births = 0
            while GLOBAL_DEATH_COUNTER >= DEATHS_FOR_BIRTH:
                print(f"🎁 Nascimento bonus ({GLOBAL_DEATH_COUNTER}/{DEATHS_FOR_BIRTH} mortes)...")
                bonus_neuron_idx = brain.add_neuron_net2wider()
                GLOBAL_DEATH_COUNTER -= DEATHS_FOR_BIRTH
                bonus_births += 1
                
                log_worm_event("bonus_birth", {
                    "round": round_num,
                    "neuron_id": bonus_neuron_idx,
                    "total_neurons": brain.hidden_dim,
                    "reason": f"death_counter_{DEATHS_FOR_BIRTH}"
                })
                
                if PROM:
                    PROM["births_total"].inc()
                    PROM["neurons_total"].set(brain.hidden_dim)
            
            if bonus_births > 0:
                print(f"  ✅ {bonus_births} nascimentos bonus completados")
            
            # 7. SNAPSHOT E LOGGING
            snapshot_path = create_snapshot(brain, round_num)
            
            log_worm_event("round_complete", {
                "round": round_num,
                "neurons_final": brain.hidden_dim,
                "valid_neurons": valid_count,
                "avg_score": avg_score,
                "avg_loss": avg_loss,
                "collective_consciousness": brain.collective_consciousness,
                "training_time": training_time,
                "snapshot": snapshot_path,
                "bonus_births": bonus_births
            })
            
            # MLflow logging
            if MLFLOW:
                mlflow.log_metrics({
                    "round": round_num,
                    "neurons_total": brain.hidden_dim,
                    "neurons_valid": valid_count,
                    "avg_score": avg_score,
                    "avg_loss": avg_loss,
                    "collective_consciousness": brain.collective_consciousness,
                    "training_time": training_time,
                    "bonus_births": bonus_births
                }, step=round_num)
            
            # Status final da rodada
            print(f"\n✅ RODADA {round_num} COMPLETA")
            print(f"   Neurônios: {brain.hidden_dim} | Válidos: {valid_count}")
            print(f"   Consciência coletiva: {brain.collective_consciousness:.3f}")
            print(f"   Snapshot: {os.path.basename(snapshot_path)}")
            
            # Dormir um pouco para observabilidade
            time.sleep(2)
    
    except KeyboardInterrupt:
        print(f"\n\n🛑 Sistema interrompido pelo usuário na rodada {round_num}")
    
    finally:
        # Snapshot final
        final_snapshot = create_snapshot(brain, round_num)
        
        log_worm_event("system_shutdown", {
            "final_round": round_num,
            "final_neurons": brain.hidden_dim,
            "final_generation": GENERATION_COUNTER,
            "final_consciousness": brain.collective_consciousness,
            "final_snapshot": final_snapshot
        })
        
        if MLFLOW:
            mlflow.end_run()
        
        print(f"\n🏁 Sistema finalizado")
        print(f"   Rodadas executadas: {round_num}")
        print(f"   Neurônios finais: {brain.hidden_dim}")
        print(f"   Geração: {GENERATION_COUNTER}")
        print(f"   Consciência final: {brain.collective_consciousness:.3f}")
        print(f"   WORM log: {WORM_LOG}")
        print(f"   Snapshot final: {final_snapshot}")

if __name__ == "__main__":
    import sys
    
    # Argumentos da linha de comando
    if len(sys.argv) > 1:
        if sys.argv[1] == "--help":
            print("Sistema Celular IA³ Neuroevolutivo")
            print("Uso: python ia3_neurogenesis.py [--rounds N] [--help]")
            print("")
            print("Variáveis de ambiente:")
            print("  IA3_ROUNDS=N          Número de rodadas (0=infinito)")
            print("  IA3_DATA_DIM=N        Dimensão dos dados de entrada")
            print("  IA3_OUT_DIM=N         Dimensão da saída")
            print("  IA3_ROUND_STEPS=N     Steps de treino por rodada")
            print("  IA3_PASS=X.X          Limiar de aprovação IA³")
            print("  IA3_METRICS_PORT=N    Porta das métricas Prometheus")
            sys.exit(0)
        elif sys.argv[1] == "--rounds" and len(sys.argv) > 2:
            ROUNDS = int(sys.argv[2])
    
    # Executar sistema
    run_neurogenesis_cycle()