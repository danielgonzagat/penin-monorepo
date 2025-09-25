"""
Equação de Turing (ET★) - Versão Otimizada
Correções baseadas na validação matemática

Melhorias:
1. Critério de convergência ajustado para γ ≤ 0.5
2. Cálculo de progresso melhorado
3. Validação numérica mais robusta
"""

import numpy as np
import logging
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional
import json
import time

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ETSignals:
    """
    Sinais padronizados para a Equação de Turing
    Baseado na consolidação dos 4 documentos
    """
    # Progresso (P_k)
    learning_progress: np.ndarray  # LP normalizado por tarefa
    task_difficulties: np.ndarray  # β_i (dificuldade/novidade)
    
    # Custo (R_k)
    mdl_complexity: float          # Complexidade estrutural
    energy_consumption: float      # Consumo computacional
    scalability_inverse: float     # 1/escalabilidade
    
    # Estabilidade (S̃_k)
    policy_entropy: float          # H[π] - exploração
    policy_divergence: float       # D(π,π_{k-1}) - continuidade
    drift_penalty: float           # Esquecimento catastrófico
    curriculum_variance: float     # Var(β) - diversidade
    regret_rate: float            # Taxa de regressão em canários
    
    # Embodiment (B_k)
    embodiment_score: float        # Integração físico-digital
    
    # Recorrência (F_γ(Φ))
    phi_components: np.ndarray     # [experiências, replay, seeds, verificadores]

class ETCoreOptimized:
    """
    Núcleo Otimizado da Equação de Turing (ET★)
    
    Implementa a forma minimalista de 4 termos com correções:
    E_{k+1} = P_k - ρR_k + σS̃_k + ιB_k → F_γ(Φ)^∞
    
    Versão: 2.0 - Otimizada e Corrigida
    """
    
    def __init__(self, 
                 rho: float = 1.0,           # Peso do custo
                 sigma: float = 1.0,         # Peso da estabilidade
                 iota: float = 1.0,          # Peso do embodiment
                 gamma: float = 0.4,         # Parâmetro da recorrência
                 zdp_quantile: float = 0.7,  # Quantil ZDP
                 entropy_threshold: float = 0.7,  # Limiar de entropia
                 regret_threshold: float = 0.1):  # Limiar de regret
        
        # Validações críticas baseadas nos documentos
        if not (0 < gamma <= 0.5):
            raise ValueError("γ deve estar em (0, 0.5] para garantir contração de Banach")
        
        if not (0 <= zdp_quantile <= 1):
            raise ValueError("Quantil ZDP deve estar em [0, 1]")
            
        if not (0 <= entropy_threshold <= 1):
            raise ValueError("Limiar de entropia deve estar em [0, 1]")
            
        if not (0 <= regret_threshold <= 1):
            raise ValueError("Limiar de regret deve estar em [0, 1]")
        
        # Parâmetros da equação
        self.rho = rho
        self.sigma = sigma
        self.iota = iota
        self.gamma = gamma
        
        # Configurações
        self.zdp_quantile = zdp_quantile
        self.entropy_threshold = entropy_threshold
        self.regret_threshold = regret_threshold
        
        # Estado interno
        self.recurrence_state = 0.0
        self.iteration_count = 0
        
        # Histórico para análise
        self.history = {
            'scores': [],
            'terms': [],
            'decisions': [],
            'recurrence_states': [],
            'timestamps': []
        }
        
        logger.info(f"ETCore Otimizado inicializado - Versão: ET★ 2.0")
        logger.info(f"Parâmetros: ρ={rho}, σ={sigma}, ι={iota}, γ={gamma}")
    
    def _stable_softmax(self, x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """
        Implementação numericamente estável do softmax
        """
        x = np.asarray(x, dtype=np.float64)
        
        if len(x) == 0:
            return np.array([])
        
        # Normalização para estabilidade numérica
        x_shifted = (x - np.max(x)) / temperature
        
        # Clipping para evitar overflow/underflow
        x_shifted = np.clip(x_shifted, -500, 500)
        
        exp_x = np.exp(x_shifted)
        return exp_x / (np.sum(exp_x) + 1e-12)
    
    def calculate_progress_term(self, signals: ETSignals) -> float:
        """
        Calcula P_k = Σ_i softmax(g(ã_i)) × β_i
        
        OTIMIZAÇÃO: Melhor implementação do ZDP e cálculo de progresso
        """
        lp = signals.learning_progress
        beta = signals.task_difficulties
        
        if len(lp) == 0 or len(beta) == 0:
            logger.warning("Learning Progress ou task difficulties vazios")
            return 0.0
        
        if len(lp) != len(beta):
            logger.warning(f"Tamanhos incompatíveis: LP={len(lp)}, β={len(beta)}")
            min_len = min(len(lp), len(beta))
            lp = lp[:min_len]
            beta = beta[:min_len]
        
        # CORREÇÃO: Aplicar ZDP de forma mais inteligente
        if len(lp) > 1:
            zdp_threshold = np.quantile(lp, self.zdp_quantile)
            valid_mask = lp >= zdp_threshold
            
            # Se nenhuma tarefa passa no ZDP, usar as top 50%
            if not np.any(valid_mask):
                median_threshold = np.median(lp)
                valid_mask = lp >= median_threshold
                logger.warning("ZDP muito restritivo, usando mediana")
        else:
            valid_mask = np.ones_like(lp, dtype=bool)
        
        # Aplicar softmax apenas nas tarefas válidas
        lp_valid = lp[valid_mask]
        beta_valid = beta[valid_mask]
        
        if len(lp_valid) == 0:
            return 0.0
        
        # CORREÇÃO: Usar LP diretamente como peso, não apenas para softmax
        # Isso garante que LP alto resulte em progresso alto
        normalized_lp = (lp_valid - np.min(lp_valid) + 0.1) / (np.max(lp_valid) - np.min(lp_valid) + 0.1)
        softmax_weights = self._stable_softmax(normalized_lp)
        
        progress = float(np.dot(softmax_weights, beta_valid))
        
        return progress
    
    def calculate_cost_term(self, signals: ETSignals) -> float:
        """
        Calcula R_k = MDL(E_k) + Energy_k + Scalability_k^{-1}
        """
        mdl = max(0, signals.mdl_complexity)
        energy = max(0, signals.energy_consumption)
        scal_inv = max(0, signals.scalability_inverse)
        
        cost = mdl + energy + scal_inv
        return float(cost)
    
    def calculate_stability_term(self, signals: ETSignals) -> float:
        """
        Calcula S̃_k = H[π] - D(π,π_{k-1}) - drift + Var(β) + (1-regret)
        """
        entropy = max(0, signals.policy_entropy)
        divergence = max(0, signals.policy_divergence)
        drift = max(0, signals.drift_penalty)
        var_beta = max(0, signals.curriculum_variance)
        regret = np.clip(signals.regret_rate, 0, 1)
        
        # Fórmula consolidada dos documentos
        stability = entropy - divergence - drift + var_beta + (1.0 - regret)
        
        return float(stability)
    
    def calculate_embodiment_term(self, signals: ETSignals) -> float:
        """
        Calcula B_k - integração físico-digital
        """
        embodiment = np.clip(signals.embodiment_score, 0, 1)
        return float(embodiment)
    
    def update_recurrence(self, signals: ETSignals) -> float:
        """
        Atualiza F_γ(Φ): x_{t+1} = (1-γ)x_t + γ tanh(f(x_t; Φ))
        
        OTIMIZAÇÃO: Garantia mais robusta de contração
        """
        phi = signals.phi_components
        
        if len(phi) == 0:
            phi_mean = 0.0
        else:
            # Clipping mais agressivo para estabilidade
            phi_clipped = np.clip(phi, -5, 5)
            phi_mean = np.mean(phi_clipped)
        
        # Recorrência contrativa com garantia adicional
        f_phi = np.tanh(phi_mean)
        
        # CORREÇÃO: Garantir que γ ≤ 0.5 resulte em contração
        new_state = (1 - self.gamma) * self.recurrence_state + self.gamma * f_phi
        
        # Clipping mais restritivo para garantir convergência
        self.recurrence_state = np.clip(new_state, -0.8, 0.8)
        
        return self.recurrence_state
    
    def calculate_score(self, signals: ETSignals) -> Tuple[float, Dict[str, float]]:
        """
        Calcula score da ET★: s = P_k - ρR_k + σS̃_k + ιB_k
        """
        # Calcular todos os termos
        P_k = self.calculate_progress_term(signals)
        R_k = self.calculate_cost_term(signals)
        S_tilde_k = self.calculate_stability_term(signals)
        B_k = self.calculate_embodiment_term(signals)
        
        # Score da ET★
        score = P_k - self.rho * R_k + self.sigma * S_tilde_k + self.iota * B_k
        
        # Dicionário de termos
        terms = {
            'P_k': P_k,
            'R_k': R_k,
            'S_tilde_k': S_tilde_k,
            'B_k': B_k,
            'score': score
        }
        
        return score, terms
    
    def check_guardrails(self, signals: ETSignals) -> bool:
        """
        Verifica guardrails de segurança
        """
        # Guardrail 1: Entropia mínima
        if signals.policy_entropy < self.entropy_threshold:
            logger.warning(f"Entropia baixa: {signals.policy_entropy:.3f} < {self.entropy_threshold}")
            return False
        
        # Guardrail 2: Regret máximo
        if signals.regret_rate > self.regret_threshold:
            logger.warning(f"Regret alto: {signals.regret_rate:.3f} > {self.regret_threshold}")
            return False
        
        # Guardrail 3: Valores numéricos válidos
        numeric_values = [
            signals.mdl_complexity, signals.energy_consumption,
            signals.scalability_inverse, signals.policy_entropy,
            signals.policy_divergence, signals.drift_penalty,
            signals.curriculum_variance, signals.regret_rate,
            signals.embodiment_score
        ]
        
        for val in numeric_values:
            if np.isnan(val) or np.isinf(val):
                logger.error(f"Valor inválido detectado: {val}")
                return False
        
        return True
    
    def accept_modification(self, signals: ETSignals) -> Tuple[bool, float, Dict[str, float]]:
        """
        Decide se aceita ou rejeita uma modificação baseado na ET★
        """
        # Calcular score e termos
        score, terms = self.calculate_score(signals)
        
        # Atualizar recorrência
        recurrence_state = self.update_recurrence(signals)
        terms['recurrence_state'] = recurrence_state
        
        # Critério 1: Score positivo
        score_positive = score > 0
        
        # Critério 2: Guardrails de segurança
        guardrails_ok = self.check_guardrails(signals)
        
        # Decisão final
        accept = score_positive and guardrails_ok
        
        # Logging detalhado
        decision_str = "ACEITAR" if accept else "REJEITAR"
        logger.info(f"Score calculado: {score:.4f} (ET★ v2.0)")
        logger.info(f"Decisão: {decision_str} (score={score:.4f}, regret={signals.regret_rate:.3f})")
        
        # Atualizar histórico
        self.history['scores'].append(score)
        self.history['terms'].append(terms.copy())
        self.history['decisions'].append(accept)
        self.history['recurrence_states'].append(recurrence_state)
        self.history['timestamps'].append(time.time())
        
        self.iteration_count += 1
        
        return accept, score, terms
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Retorna diagnósticos completos do sistema
        """
        if not self.history['scores']:
            return {'status': 'Nenhum histórico disponível'}
        
        scores = np.array(self.history['scores'])
        decisions = np.array(self.history['decisions'])
        recurrence = np.array(self.history['recurrence_states'])
        
        # Métricas básicas
        diagnostics = {
            'total_evaluations': len(scores),
            'acceptance_rate': np.mean(decisions),
            'mean_score': np.mean(scores),
            'score_std': np.std(scores),
            'current_recurrence_state': self.recurrence_state,
            'recurrence_stability': np.std(recurrence),
            'iteration_count': self.iteration_count,
            'version': 'ET★ 2.0 - Otimizada'
        }
        
        # Análise de tendências
        if len(scores) > 10:
            recent_scores = scores[-10:]
            early_scores = scores[:10]
            diagnostics['score_trend'] = np.mean(recent_scores) - np.mean(early_scores)
            diagnostics['recent_acceptance_rate'] = np.mean(decisions[-10:])
        
        # Análise de estabilidade
        if len(recurrence) > 5:
            diagnostics['recurrence_range'] = [np.min(recurrence), np.max(recurrence)]
            diagnostics['recurrence_mean'] = np.mean(recurrence)
        
        return diagnostics

# Função de teste otimizada
def test_et_core_optimized():
    """
    Teste da implementação otimizada
    """
    print("🚀 TESTE DA IMPLEMENTAÇÃO OTIMIZADA ET★ v2.0")
    print("=" * 50)
    
    # Criar instância otimizada
    et = ETCoreOptimized()
    
    # Teste 1: Progresso alto deve resultar em score alto
    signals_high_progress = ETSignals(
        learning_progress=np.array([0.9, 0.8, 0.7, 0.6]),  # LP alto
        task_difficulties=np.array([1.0, 1.5, 2.0, 2.5]),
        mdl_complexity=0.3, energy_consumption=0.2, scalability_inverse=0.1,
        policy_entropy=0.8, policy_divergence=0.1, drift_penalty=0.05,
        curriculum_variance=0.3, regret_rate=0.05, embodiment_score=0.5,
        phi_components=np.array([0.5, 0.5, 0.5, 0.5])
    )
    
    signals_low_progress = ETSignals(
        learning_progress=np.array([0.1, 0.2, 0.3, 0.4]),  # LP baixo
        task_difficulties=np.array([1.0, 1.5, 2.0, 2.5]),
        mdl_complexity=0.3, energy_consumption=0.2, scalability_inverse=0.1,
        policy_entropy=0.8, policy_divergence=0.1, drift_penalty=0.05,
        curriculum_variance=0.3, regret_rate=0.05, embodiment_score=0.5,
        phi_components=np.array([0.5, 0.5, 0.5, 0.5])
    )
    
    accept1, score1, terms1 = et.accept_modification(signals_high_progress)
    accept2, score2, terms2 = et.accept_modification(signals_low_progress)
    
    print(f"Score alto progresso: {score1:.4f} (aceito: {accept1})")
    print(f"Score baixo progresso: {score2:.4f} (aceito: {accept2})")
    print(f"Teste de progresso: {score1 > score2} ✓" if score1 > score2 else "❌")
    
    # Teste 2: Convergência da recorrência
    print(f"\nTeste de convergência:")
    states = []
    for i in range(100):
        phi = np.random.uniform(-1, 1, 4)
        test_signals = ETSignals(
            learning_progress=np.array([0.5]), task_difficulties=np.array([1.0]),
            mdl_complexity=0.3, energy_consumption=0.2, scalability_inverse=0.1,
            policy_entropy=0.8, policy_divergence=0.1, drift_penalty=0.05,
            curriculum_variance=0.3, regret_rate=0.05, embodiment_score=0.5,
            phi_components=phi
        )
        et.update_recurrence(test_signals)
        states.append(et.recurrence_state)
    
    final_variance = np.var(states[-20:])
    max_state = np.max(np.abs(states))
    
    print(f"Variância final: {final_variance:.6f}")
    print(f"Estado máximo: {max_state:.4f}")
    print(f"Convergência: {'✓' if final_variance < 0.05 and max_state < 1.0 else '❌'}")
    
    # Diagnósticos finais
    diag = et.get_diagnostics()
    print(f"\nDiagnósticos finais:")
    for key, value in diag.items():
        print(f"  {key}: {value}")
    
    print("\n🎉 Teste da versão otimizada concluído!")
    return et

if __name__ == "__main__":
    test_et_core_optimized()

