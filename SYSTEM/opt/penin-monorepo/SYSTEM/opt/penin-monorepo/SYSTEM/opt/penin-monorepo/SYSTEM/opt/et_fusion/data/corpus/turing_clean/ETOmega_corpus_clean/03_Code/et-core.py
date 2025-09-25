"""
Equação de Turing (ET) - Núcleo de Implementação
Versões ET★ (4 termos) e ET† (5 termos)

Baseado na análise consolidada dos três documentos fornecidos.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import warnings

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ETSignals:
    """Estrutura para armazenar todos os sinais necessários para a ET"""
    # Progresso
    learning_progress: np.ndarray  # LP normalizado por tarefa
    task_difficulties: np.ndarray  # β_i (dificuldade × novidade)
    
    # Recursos
    mdl_complexity: float  # Complexidade MDL
    energy_consumption: float  # Consumo energético
    scalability_inverse: float  # 1/escalabilidade
    
    # Estabilidade
    policy_entropy: float  # H[π]
    policy_divergence: float  # D(π, π_{k-1})
    drift_penalty: float  # Penalidade por esquecimento
    curriculum_variance: float  # Var(β)
    regret_rate: float  # Taxa de falhas em canários
    
    # Embodiment
    embodiment_score: float  # Sucesso em tarefas físicas
    
    # Recorrência
    phi_components: np.ndarray  # [novas, replay, seeds, verificadores]

class ETCore:
    """
    Núcleo da Equação de Turing - Implementação completa
    
    Suporta ambas as versões:
    - ET★ (4 termos): E_{k+1} = P_k - ρR_k + σS̃_k + ιB_k → F_γ(Φ)^∞
    - ET† (5 termos): E_{k+1} = P_k - ρR_k + σS_k + υV_k + ιB_k → F_γ(Φ)^∞
    """
    
    def __init__(self, 
                 rho: float = 1.0,      # Peso do custo
                 sigma: float = 1.0,    # Peso da estabilidade
                 iota: float = 1.0,     # Peso do embodiment
                 upsilon: float = 1.0,  # Peso da validação (apenas ET†)
                 gamma: float = 0.4,    # Parâmetro da recorrência
                 use_five_terms: bool = False,  # ET† vs ET★
                 zdp_quantile: float = 0.7,     # Quantil ZDP
                 entropy_min: float = 0.7,      # Entropia mínima
                 energy_threshold: float = 0.3): # Limiar de energia
        
        # Validações
        if not (0 < gamma <= 0.5):
            raise ValueError("gamma deve estar em (0, 0.5] para garantir contração de Banach")
        
        if not (0 <= zdp_quantile <= 1):
            raise ValueError("zdp_quantile deve estar em [0, 1]")
            
        # Parâmetros da equação
        self.rho = rho
        self.sigma = sigma
        self.iota = iota
        self.upsilon = upsilon
        self.gamma = gamma
        self.use_five_terms = use_five_terms
        
        # Parâmetros de controle
        self.zdp_quantile = zdp_quantile
        self.entropy_min = entropy_min
        self.energy_threshold = energy_threshold
        
        # Estado interno da recorrência
        self.recurrence_state = 0.0
        
        # Histórico para análise
        self.history = {
            'scores': [],
            'terms': [],
            'decisions': [],
            'recurrence_states': []
        }
        
        logger.info(f"ETCore inicializado - Versão: {'ET† (5 termos)' if use_five_terms else 'ET★ (4 termos)'}")
        logger.info(f"Parâmetros: ρ={rho}, σ={sigma}, ι={iota}, υ={upsilon}, γ={gamma}")
    
    def softmax(self, x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """
        Função softmax estável numericamente
        
        Args:
            x: Array de entrada
            temperature: Parâmetro de temperatura (default=1.0)
        
        Returns:
            Array normalizado via softmax
        """
        x = np.asarray(x, dtype=np.float64)
        
        # Evitar overflow subtraindo o máximo
        x_shifted = (x - np.max(x)) / temperature
        exp_x = np.exp(x_shifted)
        
        # Evitar divisão por zero
        return exp_x / (np.sum(exp_x) + 1e-12)
    
    def calculate_progress_term(self, signals: ETSignals) -> float:
        """
        Calcula o termo de Progresso P_k
        
        P_k = Σ_i softmax(g(ã_i)) * β_i
        
        Implementa ZDP: apenas tarefas com LP ≥ quantil são consideradas
        """
        lp = signals.learning_progress
        beta = signals.task_difficulties
        
        if len(lp) == 0 or len(beta) == 0:
            return 0.0
        
        # Aplicar ZDP - filtrar por quantil
        if len(lp) > 1:
            zdp_threshold = np.quantile(lp, self.zdp_quantile)
            valid_mask = lp >= zdp_threshold
            
            if not np.any(valid_mask):
                # Se nenhuma tarefa passa no ZDP, usar todas
                valid_mask = np.ones_like(lp, dtype=bool)
                warnings.warn("Nenhuma tarefa passou no ZDP, usando todas")
        else:
            valid_mask = np.ones_like(lp, dtype=bool)
        
        # Aplicar softmax apenas nas tarefas válidas
        lp_valid = lp[valid_mask]
        beta_valid = beta[valid_mask]
        
        if len(lp_valid) == 0:
            return 0.0
        
        # Calcular softmax e progresso ponderado
        softmax_weights = self.softmax(lp_valid)
        progress = float(np.dot(softmax_weights, beta_valid))
        
        logger.debug(f"Progresso P_k = {progress:.4f} (tarefas válidas: {len(lp_valid)}/{len(lp)})")
        return progress
    
    def calculate_cost_term(self, signals: ETSignals) -> float:
        """
        Calcula o termo de Custo/Recursos R_k
        
        R_k = MDL(E_k) + Energy_k + Scalability_k^{-1}
        """
        cost = (signals.mdl_complexity + 
                signals.energy_consumption + 
                signals.scalability_inverse)
        
        logger.debug(f"Custo R_k = {cost:.4f} (MDL={signals.mdl_complexity:.3f}, "
                    f"Energy={signals.energy_consumption:.3f}, Scal^-1={signals.scalability_inverse:.3f})")
        return cost
    
    def calculate_stability_term(self, signals: ETSignals) -> float:
        """
        Calcula o termo de Estabilidade S_k ou S̃_k
        
        ET★ (4 termos): S̃_k = H[π] - D(π,π_{k-1}) - drift + Var(β) + (1-regret)
        ET† (5 termos): S_k = H[π] - D(π,π_{k-1}) - drift + Var(β)
        """
        stability = (signals.policy_entropy - 
                    signals.policy_divergence - 
                    signals.drift_penalty + 
                    signals.curriculum_variance)
        
        # Na versão de 4 termos, incluir validação
        if not self.use_five_terms:
            stability += (1.0 - signals.regret_rate)
            term_name = "S̃_k"
        else:
            term_name = "S_k"
        
        logger.debug(f"Estabilidade {term_name} = {stability:.4f}")
        return stability
    
    def calculate_validation_term(self, signals: ETSignals) -> float:
        """
        Calcula o termo de Validação V_k (apenas para ET† de 5 termos)
        
        V_k = 1 - regret
        """
        if not self.use_five_terms:
            return 0.0
        
        validation = 1.0 - signals.regret_rate
        logger.debug(f"Validação V_k = {validation:.4f}")
        return validation
    
    def calculate_embodiment_term(self, signals: ETSignals) -> float:
        """
        Calcula o termo de Embodiment B_k
        
        B_k = sucesso em tarefas físicas/sensoriais
        """
        embodiment = signals.embodiment_score
        logger.debug(f"Embodiment B_k = {embodiment:.4f}")
        return embodiment
    
    def calculate_score(self, signals: ETSignals) -> Tuple[float, Dict[str, float]]:
        """
        Calcula o score da Equação de Turing
        
        ET★: s = P_k - ρR_k + σS̃_k + ιB_k
        ET†: s = P_k - ρR_k + σS_k + υV_k + ιB_k
        
        Returns:
            Tuple[score, terms_dict]
        """
        # Calcular todos os termos
        P_k = self.calculate_progress_term(signals)
        R_k = self.calculate_cost_term(signals)
        S_k = self.calculate_stability_term(signals)
        V_k = self.calculate_validation_term(signals)
        B_k = self.calculate_embodiment_term(signals)
        
        # Calcular score
        if self.use_five_terms:
            score = P_k - self.rho * R_k + self.sigma * S_k + self.upsilon * V_k + self.iota * B_k
        else:
            score = P_k - self.rho * R_k + self.sigma * S_k + self.iota * B_k
        
        # Dicionário com todos os termos
        terms = {
            'P_k': P_k,
            'R_k': R_k,
            'S_k': S_k,
            'V_k': V_k,
            'B_k': B_k,
            'score': score
        }
        
        logger.info(f"Score calculado: {score:.4f} ({'ET†' if self.use_five_terms else 'ET★'})")
        return score, terms
    
    def update_recurrence(self, signals: ETSignals) -> float:
        """
        Atualiza o estado da recorrência contrativa F_γ(Φ)
        
        x_{t+1} = (1-γ)x_t + γ tanh(f(x_t; Φ))
        
        onde Φ = [novas, replay, seeds, verificadores]
        """
        # Agregar componentes de Φ
        phi_mean = np.mean(signals.phi_components) if len(signals.phi_components) > 0 else 0.0
        
        # Aplicar função f (média simples) e tanh para saturação
        f_phi = np.tanh(phi_mean)
        
        # Atualizar estado com contração
        new_state = (1 - self.gamma) * self.recurrence_state + self.gamma * f_phi
        
        logger.debug(f"Recorrência: {self.recurrence_state:.4f} → {new_state:.4f} (Φ={phi_mean:.4f})")
        
        self.recurrence_state = new_state
        return new_state
    
    def accept_modification(self, signals: ETSignals) -> Tuple[bool, float, Dict[str, float]]:
        """
        Decide se aceita ou rejeita uma modificação
        
        Critérios:
        1. score > 0
        2. regret não aumentou (para ET★) ou validação não diminuiu (para ET†)
        3. Guardrails de segurança
        
        Returns:
            Tuple[accept, score, terms]
        """
        # Calcular score e termos
        score, terms = self.calculate_score(signals)
        
        # Atualizar recorrência
        recurrence_state = self.update_recurrence(signals)
        
        # Critério 1: Score positivo
        score_positive = score > 0
        
        # Critério 2: Validação não piorou
        validation_ok = signals.regret_rate <= 0.1  # Limiar de regressão aceitável
        
        # Critério 3: Guardrails de segurança
        entropy_ok = signals.policy_entropy >= self.entropy_min
        energy_ok = signals.energy_consumption <= self.energy_threshold
        stability_ok = not (np.isnan(score) or np.isinf(score))
        
        # Decisão final
        accept = (score_positive and validation_ok and 
                 entropy_ok and energy_ok and stability_ok)
        
        # Log da decisão
        decision_info = {
            'accept': accept,
            'score_positive': score_positive,
            'validation_ok': validation_ok,
            'entropy_ok': entropy_ok,
            'energy_ok': energy_ok,
            'stability_ok': stability_ok
        }
        
        logger.info(f"Decisão: {'ACEITAR' if accept else 'REJEITAR'} "
                   f"(score={score:.4f}, regret={signals.regret_rate:.3f})")
        
        # Armazenar no histórico
        self.history['scores'].append(score)
        self.history['terms'].append(terms)
        self.history['decisions'].append(accept)
        self.history['recurrence_states'].append(recurrence_state)
        
        return accept, score, terms
    
    def get_diagnostics(self) -> Dict:
        """
        Retorna diagnósticos do sistema
        """
        if not self.history['scores']:
            return {'status': 'Nenhum histórico disponível'}
        
        scores = np.array(self.history['scores'])
        decisions = np.array(self.history['decisions'])
        recurrence = np.array(self.history['recurrence_states'])
        
        return {
            'total_evaluations': len(scores),
            'acceptance_rate': np.mean(decisions),
            'mean_score': np.mean(scores),
            'score_std': np.std(scores),
            'score_trend': np.polyfit(range(len(scores)), scores, 1)[0] if len(scores) > 1 else 0,
            'recurrence_stability': np.std(recurrence),
            'current_recurrence_state': self.recurrence_state,
            'version': 'ET† (5 termos)' if self.use_five_terms else 'ET★ (4 termos)'
        }
    
    def reset_history(self):
        """Limpa o histórico de avaliações"""
        self.history = {
            'scores': [],
            'terms': [],
            'decisions': [],
            'recurrence_states': []
        }
        logger.info("Histórico resetado")
    
    def adjust_parameters(self, 
                         rho: Optional[float] = None,
                         sigma: Optional[float] = None,
                         iota: Optional[float] = None,
                         upsilon: Optional[float] = None):
        """
        Ajusta parâmetros da equação (meta-aprendizado)
        """
        if rho is not None:
            self.rho = rho
        if sigma is not None:
            self.sigma = sigma
        if iota is not None:
            self.iota = iota
        if upsilon is not None:
            self.upsilon = upsilon
            
        logger.info(f"Parâmetros ajustados: ρ={self.rho}, σ={self.sigma}, ι={self.iota}, υ={self.upsilon}")


def create_test_signals(seed: int = 42) -> ETSignals:
    """
    Cria sinais de teste para validação
    """
    np.random.seed(seed)
    
    return ETSignals(
        learning_progress=np.random.uniform(0, 1, 5),
        task_difficulties=np.random.uniform(0.5, 2.0, 5),
        mdl_complexity=np.random.uniform(0.1, 0.5),
        energy_consumption=np.random.uniform(0.0, 0.3),
        scalability_inverse=np.random.uniform(0.1, 0.4),
        policy_entropy=np.random.uniform(0.5, 1.0),
        policy_divergence=np.random.uniform(0.0, 0.3),
        drift_penalty=np.random.uniform(0.0, 0.2),
        curriculum_variance=np.random.uniform(0.1, 0.5),
        regret_rate=np.random.uniform(0.0, 0.1),
        embodiment_score=np.random.uniform(0.0, 1.0),
        phi_components=np.random.uniform(-0.5, 0.5, 4)
    )


if __name__ == "__main__":
    # Teste básico do sistema
    print("=== Teste da Equação de Turing ===")
    
    # Testar ambas as versões
    for use_five_terms in [False, True]:
        version = "ET† (5 termos)" if use_five_terms else "ET★ (4 termos)"
        print(f"\n--- Testando {version} ---")
        
        et = ETCore(use_five_terms=use_five_terms)
        
        # Executar várias iterações
        for i in range(5):
            signals = create_test_signals(seed=42+i)
            accept, score, terms = et.accept_modification(signals)
            
            print(f"Iteração {i+1}: {'✓' if accept else '✗'} Score: {score:.4f}")
        
        # Mostrar diagnósticos
        diag = et.get_diagnostics()
        print(f"Taxa de aceitação: {diag['acceptance_rate']:.2%}")
        print(f"Score médio: {diag['mean_score']:.4f} ± {diag['score_std']:.4f}")
        print(f"Estabilidade da recorrência: {diag['recurrence_stability']:.4f}")

