"""
Equação de Turing Aperfeiçoada (ET★★) - Versão 6.0
Incorporando melhorias identificadas na análise teórica

Principais Aperfeiçoamentos:
1. Adaptação dinâmica de parâmetros por domínio
2. Calibração automática de guardrails
3. Robustez aprimorada para LLMs
4. Métricas de diagnóstico avançadas
5. Implementação da versão ETΩ com Expected Improvement
"""

import numpy as np
import logging
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional, List
import json
import time
from enum import Enum

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DomainType(Enum):
    """Tipos de domínio suportados"""
    REINFORCEMENT_LEARNING = "rl"
    LARGE_LANGUAGE_MODEL = "llm"
    ROBOTICS = "robotics"
    SCIENTIFIC_DISCOVERY = "science"
    GENERAL = "general"

@dataclass
class ETSignals:
    """Sinais padronizados para a Equação de Turing"""
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

class ETCoreAperfeicoado:
    """
    Núcleo Aperfeiçoado da Equação de Turing (ET★★)
    
    Versão: 6.0 - Aperfeiçoada com adaptação dinâmica
    Suporta tanto ET★ quanto ETΩ
    """
    
    def __init__(self, 
                 domain: DomainType = DomainType.GENERAL,
                 use_omega: bool = True,           # Usar versão ETΩ
                 auto_calibrate: bool = True,      # Calibração automática
                 rho: Optional[float] = None,      # Peso do custo
                 sigma: Optional[float] = None,    # Peso da estabilidade
                 iota: Optional[float] = None,     # Peso do embodiment
                 gamma: float = 0.4,               # Parâmetro da recorrência
                 tau_ei: float = 1.0,              # Temperatura para EI
                 zdp_quantile: float = 0.7):       # Quantil ZDP
        
        self.domain = domain
        self.use_omega = use_omega
        self.auto_calibrate = auto_calibrate
        self.tau_ei = tau_ei
        self.zdp_quantile = zdp_quantile
        
        # Validações críticas
        if not (0 < gamma <= 0.5):
            raise ValueError("γ deve estar em (0, 0.5] para garantir contração de Banach")
        self.gamma = gamma
        
        # Configurar parâmetros por domínio
        self._configure_domain_parameters(rho, sigma, iota)
        
        # Estado interno
        self.recurrence_state = 0.0
        self.iteration_count = 0
        self.adaptation_history = []
        
        # Histórico para análise
        self.history = {
            'scores': [],
            'terms': [],
            'decisions': [],
            'recurrence_states': [],
            'parameters': [],
            'timestamps': []
        }
        
        # Estatísticas para calibração automática
        self.calibration_stats = {
            'lp_mean': 0.5,
            'lp_std': 0.2,
            'acceptance_rate': 0.5,
            'score_trend': 0.0
        }
        
        logger.info(f"ETCore Aperfeiçoado inicializado - Versão: ET★★ 6.0")
        logger.info(f"Domínio: {domain.value}, ETΩ: {use_omega}, Auto-calibração: {auto_calibrate}")
        logger.info(f"Parâmetros: ρ={self.rho:.2f}, σ={self.sigma:.2f}, ι={self.iota:.2f}, γ={gamma}")
    
    def _configure_domain_parameters(self, rho: Optional[float], 
                                   sigma: Optional[float], 
                                   iota: Optional[float]):
        """Configura parâmetros específicos por domínio"""
        
        # Configurações otimizadas por domínio
        domain_configs = {
            DomainType.REINFORCEMENT_LEARNING: {
                'rho': 1.0, 'sigma': 1.2, 'iota': 0.3,
                'entropy_threshold': 0.7, 'regret_threshold': 0.1,
                'divergence_threshold': 0.2, 'drift_threshold': 0.1,
                'cost_threshold': 2.0, 'var_min': 0.1
            },
            DomainType.LARGE_LANGUAGE_MODEL: {
                'rho': 0.8, 'sigma': 1.0, 'iota': 0.1,  # Reduzido ρ para LLMs
                'entropy_threshold': 0.5, 'regret_threshold': 0.15,  # Relaxado
                'divergence_threshold': 0.3, 'drift_threshold': 0.15,
                'cost_threshold': 4.0, 'var_min': 0.05  # Maior tolerância a custo
            },
            DomainType.ROBOTICS: {
                'rho': 0.8, 'sigma': 1.5, 'iota': 2.0,
                'entropy_threshold': 0.6, 'regret_threshold': 0.08,
                'divergence_threshold': 0.15, 'drift_threshold': 0.08,
                'cost_threshold': 2.5, 'var_min': 0.15
            },
            DomainType.SCIENTIFIC_DISCOVERY: {
                'rho': 1.2, 'sigma': 2.0, 'iota': 1.8,
                'entropy_threshold': 0.8, 'regret_threshold': 0.05,
                'divergence_threshold': 0.18, 'drift_threshold': 0.1,
                'cost_threshold': 3.0, 'var_min': 0.2
            },
            DomainType.GENERAL: {
                'rho': 1.0, 'sigma': 1.0, 'iota': 1.0,
                'entropy_threshold': 0.7, 'regret_threshold': 0.1,
                'divergence_threshold': 0.2, 'drift_threshold': 0.1,
                'cost_threshold': 2.0, 'var_min': 0.1
            }
        }
        
        config = domain_configs[self.domain]
        
        # Usar parâmetros fornecidos ou padrões do domínio
        self.rho = rho if rho is not None else config['rho']
        self.sigma = sigma if sigma is not None else config['sigma']
        self.iota = iota if iota is not None else config['iota']
        
        # Configurar guardrails
        self.entropy_threshold = config['entropy_threshold']
        self.regret_threshold = config['regret_threshold']
        self.divergence_threshold = config['divergence_threshold']
        self.drift_threshold = config['drift_threshold']
        self.cost_threshold = config['cost_threshold']
        self.var_min = config['var_min']
    
    def calculate_progress_term(self, signals: ETSignals) -> float:
        """
        Calcula P_k usando ETΩ (Expected Improvement) ou ET★ (Learning Progress)
        """
        lp = signals.learning_progress
        beta = signals.task_difficulties
        
        if len(lp) == 0 or len(beta) == 0:
            return 0.0
        
        if len(lp) != len(beta):
            min_len = min(len(lp), len(beta))
            lp = lp[:min_len]
            beta = beta[:min_len]
        
        if self.use_omega:
            # ETΩ: Expected Improvement com softmax
            return self._calculate_progress_omega(lp, beta)
        else:
            # ET★: Learning Progress original
            return self._calculate_progress_star(lp, beta)
    
    def _calculate_progress_omega(self, lp: np.ndarray, beta: np.ndarray) -> float:
        """Calcula progresso usando Expected Improvement (ETΩ)"""
        # Atualizar estatísticas de calibração
        self.calibration_stats['lp_mean'] = 0.9 * self.calibration_stats['lp_mean'] + 0.1 * np.mean(lp)
        self.calibration_stats['lp_std'] = 0.9 * self.calibration_stats['lp_std'] + 0.1 * np.std(lp)
        
        # Calcular Expected Improvement
        mu_lp = self.calibration_stats['lp_mean']
        sigma_lp = max(self.calibration_stats['lp_std'], 0.01)  # Evitar divisão por zero
        
        # Z-score truncado
        ei = np.maximum(0, (lp - mu_lp) / sigma_lp)
        
        if np.sum(ei) == 0:
            return 0.0
        
        # Softmax com temperatura
        weights = np.exp(ei / self.tau_ei)
        weights = weights / np.sum(weights)
        
        # Progresso ponderado
        progress = np.sum(weights * beta)
        
        return float(progress)
    
    def _calculate_progress_star(self, lp: np.ndarray, beta: np.ndarray) -> float:
        """Calcula progresso usando Learning Progress original (ET★)"""
        # Aplicar ZDP - filtrar por quantil
        if len(lp) > 1:
            zdp_threshold = np.quantile(lp, self.zdp_quantile)
            valid_mask = lp >= zdp_threshold
            
            if not np.any(valid_mask):
                # Fallback: usar as melhores tarefas
                sorted_indices = np.argsort(lp)[::-1]
                n_keep = max(1, len(lp) // 2)
                valid_mask = np.zeros_like(lp, dtype=bool)
                valid_mask[sorted_indices[:n_keep]] = True
        else:
            valid_mask = np.ones_like(lp, dtype=bool)
        
        # Filtrar tarefas válidas
        lp_valid = lp[valid_mask]
        beta_valid = beta[valid_mask]
        
        if len(lp_valid) == 0:
            return 0.0
        
        # Fórmula ET★: LP_médio × β_médio × fator_qualidade
        lp_mean = np.mean(lp_valid)
        beta_mean = np.mean(beta_valid)
        quality_factor = np.sum(valid_mask) / len(lp)
        
        progress = lp_mean * beta_mean * (1 + quality_factor)
        
        return float(progress)
    
    def calculate_cost_term(self, signals: ETSignals) -> float:
        """Calcula R_k = MDL(E_k) + Energy_k + Scalability_k^{-1}"""
        mdl = max(0, signals.mdl_complexity)
        energy = max(0, signals.energy_consumption)
        scal_inv = max(0, signals.scalability_inverse)
        
        cost = mdl + energy + scal_inv
        return float(cost)
    
    def calculate_stability_term(self, signals: ETSignals) -> float:
        """Calcula S̃_k = H[π] - D(π,π_{k-1}) - drift + Var(β) + (1-regret)"""
        entropy = max(0, signals.policy_entropy)
        divergence = max(0, signals.policy_divergence)
        drift = max(0, signals.drift_penalty)
        var_beta = max(0, signals.curriculum_variance)
        regret = np.clip(signals.regret_rate, 0, 1)
        
        stability = entropy - divergence - drift + var_beta + (1.0 - regret)
        return float(stability)
    
    def calculate_embodiment_term(self, signals: ETSignals) -> float:
        """Calcula B_k - integração físico-digital"""
        embodiment = np.clip(signals.embodiment_score, 0, 1)
        return float(embodiment)
    
    def update_recurrence(self, signals: ETSignals) -> float:
        """Atualiza F_γ(Φ): x_{t+1} = (1-γ)x_t + γ tanh(f(x_t; Φ))"""
        phi = signals.phi_components
        
        if len(phi) == 0:
            phi_mean = 0.0
        else:
            phi_clipped = np.clip(phi, -5, 5)
            phi_mean = np.mean(phi_clipped)
        
        # Recorrência contrativa
        f_phi = np.tanh(phi_mean)
        new_state = (1 - self.gamma) * self.recurrence_state + self.gamma * f_phi
        
        # Garantir estabilidade
        self.recurrence_state = np.clip(new_state, -1, 1)
        return self.recurrence_state
    
    def check_guardrails(self, signals: ETSignals) -> Tuple[bool, List[str]]:
        """Verifica guardrails de segurança com diagnósticos detalhados"""
        violations = []
        
        # Guardrail 1: Entropia mínima
        if signals.policy_entropy < self.entropy_threshold:
            violations.append(f"Entropia baixa: {signals.policy_entropy:.3f} < {self.entropy_threshold}")
        
        # Guardrail 2: Regret máximo
        if signals.regret_rate > self.regret_threshold:
            violations.append(f"Regret alto: {signals.regret_rate:.3f} > {self.regret_threshold}")
        
        # Guardrail 3: Divergência limitada (ETΩ)
        if self.use_omega and signals.policy_divergence > self.divergence_threshold:
            violations.append(f"Divergência alta: {signals.policy_divergence:.3f} > {self.divergence_threshold}")
        
        # Guardrail 4: Drift controlado (ETΩ)
        if self.use_omega and signals.drift_penalty > self.drift_threshold:
            violations.append(f"Drift alto: {signals.drift_penalty:.3f} > {self.drift_threshold}")
        
        # Guardrail 5: Orçamento de custo (ETΩ)
        if self.use_omega:
            cost = self.calculate_cost_term(signals)
            if cost > self.cost_threshold:
                violations.append(f"Custo alto: {cost:.3f} > {self.cost_threshold}")
        
        # Guardrail 6: Variância mínima (ETΩ)
        if self.use_omega and signals.curriculum_variance < self.var_min:
            violations.append(f"Variância baixa: {signals.curriculum_variance:.3f} < {self.var_min}")
        
        # Guardrail 7: Valores numéricos válidos
        numeric_values = [
            signals.mdl_complexity, signals.energy_consumption,
            signals.scalability_inverse, signals.policy_entropy,
            signals.policy_divergence, signals.drift_penalty,
            signals.curriculum_variance, signals.regret_rate,
            signals.embodiment_score
        ]
        
        for i, val in enumerate(numeric_values):
            if np.isnan(val) or np.isinf(val):
                violations.append(f"Valor inválido detectado na posição {i}: {val}")
        
        return len(violations) == 0, violations
    
    def adapt_parameters(self, recent_performance: Dict[str, float]):
        """Adaptação dinâmica de parâmetros baseada na performance"""
        if not self.auto_calibrate:
            return
        
        acceptance_rate = recent_performance.get('acceptance_rate', 0.5)
        score_trend = recent_performance.get('score_trend', 0.0)
        
        # Atualizar estatísticas
        self.calibration_stats['acceptance_rate'] = acceptance_rate
        self.calibration_stats['score_trend'] = score_trend
        
        # Adaptação conservadora
        adaptation_rate = 0.05
        
        # Se taxa de aceitação muito baixa, relaxar restrições
        if acceptance_rate < 0.2:
            self.entropy_threshold *= (1 - adaptation_rate)
            self.regret_threshold *= (1 + adaptation_rate)
            if self.use_omega:
                self.cost_threshold *= (1 + adaptation_rate)
                self.divergence_threshold *= (1 + adaptation_rate)
        
        # Se taxa de aceitação muito alta, apertar restrições
        elif acceptance_rate > 0.8:
            self.entropy_threshold *= (1 + adaptation_rate)
            self.regret_threshold *= (1 - adaptation_rate)
            if self.use_omega:
                self.cost_threshold *= (1 - adaptation_rate)
                self.divergence_threshold *= (1 - adaptation_rate)
        
        # Registrar adaptação
        self.adaptation_history.append({
            'iteration': self.iteration_count,
            'acceptance_rate': acceptance_rate,
            'entropy_threshold': self.entropy_threshold,
            'regret_threshold': self.regret_threshold,
            'timestamp': time.time()
        })
    
    def calculate_score(self, signals: ETSignals) -> Tuple[float, Dict[str, float]]:
        """Calcula score da ET: s = P_k - ρR_k + σS̃_k + ιB_k"""
        # Calcular todos os termos
        P_k = self.calculate_progress_term(signals)
        R_k = self.calculate_cost_term(signals)
        S_tilde_k = self.calculate_stability_term(signals)
        B_k = self.calculate_embodiment_term(signals)
        
        # Score da ET
        score = P_k - self.rho * R_k + self.sigma * S_tilde_k + self.iota * B_k
        
        # Dicionário de termos
        terms = {
            'P_k': P_k,
            'R_k': R_k,
            'S_tilde_k': S_tilde_k,
            'B_k': B_k,
            'score': score,
            'version': 'ETΩ' if self.use_omega else 'ET★'
        }
        
        return score, terms
    
    def accept_modification(self, signals: ETSignals) -> Tuple[bool, float, Dict[str, Any]]:
        """Decide se aceita ou rejeita uma modificação baseado na ET"""
        # Calcular score e termos
        score, terms = self.calculate_score(signals)
        
        # Atualizar recorrência
        recurrence_state = self.update_recurrence(signals)
        terms['recurrence_state'] = recurrence_state
        
        # Verificar guardrails
        guardrails_ok, violations = self.check_guardrails(signals)
        terms['guardrail_violations'] = violations
        
        # Critérios de aceitação
        score_positive = score > 0
        
        # Decisão final
        accept = score_positive and guardrails_ok
        
        # Logging detalhado
        decision_str = "ACEITAR" if accept else "REJEITAR"
        if violations:
            logger.warning(f"Violações: {'; '.join(violations)}")
        logger.info(f"Score: {score:.4f} | Decisão: {decision_str} | P_k: {terms['P_k']:.4f} | Domínio: {self.domain.value}")
        
        # Atualizar histórico
        self.history['scores'].append(score)
        self.history['terms'].append(terms.copy())
        self.history['decisions'].append(accept)
        self.history['recurrence_states'].append(recurrence_state)
        self.history['parameters'].append({
            'rho': self.rho, 'sigma': self.sigma, 'iota': self.iota,
            'entropy_threshold': self.entropy_threshold,
            'regret_threshold': self.regret_threshold
        })
        self.history['timestamps'].append(time.time())
        
        self.iteration_count += 1
        
        # Adaptação automática a cada 10 iterações
        if self.auto_calibrate and self.iteration_count % 10 == 0:
            recent_decisions = self.history['decisions'][-10:]
            recent_scores = self.history['scores'][-10:]
            
            performance = {
                'acceptance_rate': np.mean(recent_decisions),
                'score_trend': np.mean(recent_scores[-5:]) - np.mean(recent_scores[:5]) if len(recent_scores) >= 5 else 0
            }
            self.adapt_parameters(performance)
        
        return accept, score, terms
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Retorna diagnósticos completos do sistema"""
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
            'version': 'ET★★ 6.0 - Aperfeiçoada',
            'domain': self.domain.value,
            'use_omega': self.use_omega,
            'auto_calibrate': self.auto_calibrate
        }
        
        # Parâmetros atuais
        diagnostics['current_parameters'] = {
            'rho': self.rho,
            'sigma': self.sigma,
            'iota': self.iota,
            'gamma': self.gamma,
            'entropy_threshold': self.entropy_threshold,
            'regret_threshold': self.regret_threshold
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
        
        # Estatísticas de violações
        all_violations = []
        for terms in self.history['terms']:
            if 'guardrail_violations' in terms:
                all_violations.extend(terms['guardrail_violations'])
        
        if all_violations:
            from collections import Counter
            violation_counts = Counter(all_violations)
            diagnostics['most_common_violations'] = violation_counts.most_common(3)
        
        # Histórico de adaptação
        if self.adaptation_history:
            diagnostics['adaptation_count'] = len(self.adaptation_history)
            diagnostics['latest_adaptation'] = self.adaptation_history[-1]
        
        return diagnostics

# Função de teste aperfeiçoada
def test_et_aperfeicoado():
    """Teste da ET★★ 6.0 Aperfeiçoada"""
    print("🚀 TESTE DA ET★★ 6.0 APERFEIÇOADA")
    print("=" * 60)
    
    # Teste para cada domínio
    domains = [
        DomainType.REINFORCEMENT_LEARNING,
        DomainType.LARGE_LANGUAGE_MODEL,
        DomainType.ROBOTICS,
        DomainType.SCIENTIFIC_DISCOVERY
    ]
    
    results = {}
    
    for domain in domains:
        print(f"\n🔬 TESTANDO DOMÍNIO: {domain.value.upper()}")
        print("-" * 50)
        
        # Testar ambas as versões
        for use_omega in [False, True]:
            version_name = "ETΩ" if use_omega else "ET★"
            print(f"\n  Versão: {version_name}")
            
            et = ETCoreAperfeicoado(
                domain=domain,
                use_omega=use_omega,
                auto_calibrate=True
            )
            
            # Gerar sinais de teste apropriados para o domínio
            test_signals = generate_domain_signals(domain, "moderate")
            
            # Executar teste
            accept, score, terms = et.accept_modification(test_signals)
            
            print(f"    Score: {score:.4f}")
            print(f"    Decisão: {'ACEITAR' if accept else 'REJEITAR'}")
            print(f"    P_k: {terms['P_k']:.4f}")
            
            # Armazenar resultado
            key = f"{domain.value}_{version_name}"
            results[key] = {
                'score': score,
                'accept': accept,
                'terms': terms
            }
    
    print(f"\n📊 RESUMO DOS RESULTADOS")
    print("=" * 60)
    for key, result in results.items():
        domain, version = key.split('_')
        status = "✅ ACEITO" if result['accept'] else "❌ REJEITADO"
        print(f"{domain:>20} {version:>5}: {result['score']:>8.3f} {status}")
    
    return results

def generate_domain_signals(domain: DomainType, scenario: str) -> ETSignals:
    """Gera sinais de teste apropriados para cada domínio"""
    
    if scenario == "high_performance":
        lp_range = (0.7, 0.9)
        regret_range = (0.02, 0.06)
        entropy_range = (0.75, 0.9)
    elif scenario == "challenging":
        lp_range = (0.1, 0.4)
        regret_range = (0.08, 0.15)
        entropy_range = (0.4, 0.6)
    else:  # moderate
        lp_range = (0.4, 0.7)
        regret_range = (0.04, 0.08)
        entropy_range = (0.7, 0.85)
    
    # Ajustes específicos por domínio
    if domain == DomainType.LARGE_LANGUAGE_MODEL:
        # LLMs têm naturalmente alta complexidade e baixo embodiment
        mdl_range = (1.0, 3.0)
        energy_range = (0.5, 0.9)
        embodiment_range = (0.0, 0.2)
        n_tasks = 3
    elif domain == DomainType.ROBOTICS:
        # Robótica tem alto embodiment
        mdl_range = (0.3, 1.0)
        energy_range = (0.4, 0.8)
        embodiment_range = (0.6, 0.9)
        n_tasks = 5
    elif domain == DomainType.SCIENTIFIC_DISCOVERY:
        # Descoberta científica tem alto embodiment e complexidade
        mdl_range = (0.5, 1.5)
        energy_range = (0.3, 0.7)
        embodiment_range = (0.7, 0.95)
        n_tasks = 4
    else:  # RL ou GENERAL
        mdl_range = (0.2, 0.8)
        energy_range = (0.3, 0.7)
        embodiment_range = (0.1, 0.4)
        n_tasks = 4
    
    return ETSignals(
        learning_progress=np.random.uniform(*lp_range, n_tasks),
        task_difficulties=np.random.uniform(1.0, 2.0, n_tasks),
        mdl_complexity=np.random.uniform(*mdl_range),
        energy_consumption=np.random.uniform(*energy_range),
        scalability_inverse=np.random.uniform(0.1, 0.3),
        policy_entropy=np.random.uniform(*entropy_range),
        policy_divergence=np.random.uniform(0.05, 0.15),
        drift_penalty=np.random.uniform(0.02, 0.08),
        curriculum_variance=np.random.uniform(0.2, 0.5),
        regret_rate=np.random.uniform(*regret_range),
        embodiment_score=np.random.uniform(*embodiment_range),
        phi_components=np.random.uniform(-1, 1, 4)
    )

if __name__ == "__main__":
    test_et_aperfeicoado()

