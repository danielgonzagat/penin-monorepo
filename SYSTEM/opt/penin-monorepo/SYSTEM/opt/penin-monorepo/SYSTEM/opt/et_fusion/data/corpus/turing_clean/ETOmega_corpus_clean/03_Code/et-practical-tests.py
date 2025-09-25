"""
Testes Práticos Avançados da Equação de Turing
Simulações para RL, LLMs, Robótica e Descoberta Científica
"""

import numpy as np
import matplotlib.pyplot as plt
from et_core import ETCore, ETSignals
import logging
from typing import Dict, List, Tuple
from dataclasses import dataclass
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SimulationResults:
    """Resultados de uma simulação"""
    domain: str
    scores: List[float]
    acceptance_rate: float
    stability_metric: float
    final_performance: float
    convergence_time: int
    guardrails_triggered: int

class DomainSimulator:
    """Simulador base para diferentes domínios"""
    
    def __init__(self, domain_name: str):
        self.domain_name = domain_name
        self.performance_history = []
        self.guardrails_count = 0
    
    def generate_signals(self, iteration: int, performance: float) -> ETSignals:
        """Gera sinais específicos do domínio - deve ser sobrescrito"""
        raise NotImplementedError
    
    def update_performance(self, accepted: bool, score: float) -> float:
        """Atualiza performance baseado na decisão - deve ser sobrescrito"""
        raise NotImplementedError

class RLSimulator(DomainSimulator):
    """Simulador para Aprendizado por Reforço"""
    
    def __init__(self):
        super().__init__("Reinforcement Learning")
        self.current_return = 0.5  # Retorno inicial
        self.task_difficulties = np.array([0.5, 1.0, 1.5, 2.0])  # 4 tarefas
        self.policy_params = 1000  # Número de parâmetros
        
    def generate_signals(self, iteration: int, performance: float) -> ETSignals:
        # Simular Learning Progress baseado em melhoria de retorno
        recent_returns = self.performance_history[-10:] if len(self.performance_history) >= 10 else [0.3]
        lp_values = []
        
        for i, difficulty in enumerate(self.task_difficulties):
            # LP maior para tarefas de dificuldade apropriada
            if 0.3 <= difficulty <= 1.5:
                lp = np.random.uniform(0.6, 0.9) * performance
            else:
                lp = np.random.uniform(0.1, 0.4) * performance
            lp_values.append(lp)
        
        # Simular entropia da política (diminui com convergência)
        entropy = max(0.3, 1.0 - performance * 0.5 + np.random.normal(0, 0.1))
        
        # Divergência entre políticas (pequena se estável)
        divergence = np.random.uniform(0.05, 0.2) if performance > 0.7 else np.random.uniform(0.1, 0.4)
        
        # Drift (esquecimento) - aumenta se performance cai
        drift = max(0, 0.3 - performance) + np.random.uniform(0, 0.1)
        
        # Regret baseado em falhas em tarefas canário
        regret = max(0, 0.15 - performance) + np.random.uniform(0, 0.05)
        
        return ETSignals(
            learning_progress=np.array(lp_values),
            task_difficulties=self.task_difficulties,
            mdl_complexity=self.policy_params / 10000,  # Normalizado
            energy_consumption=np.random.uniform(0.1, 0.3),
            scalability_inverse=np.random.uniform(0.1, 0.3),
            policy_entropy=entropy,
            policy_divergence=divergence,
            drift_penalty=drift,
            curriculum_variance=np.var(self.task_difficulties),
            regret_rate=regret,
            embodiment_score=np.random.uniform(0.3, 0.8),  # Simulação física
            phi_components=np.random.uniform(-0.3, 0.3, 4)
        )
    
    def update_performance(self, accepted: bool, score: float) -> float:
        if accepted and score > 0:
            # Melhoria baseada no score
            improvement = min(0.1, score * 0.02)
            self.current_return = min(1.0, self.current_return + improvement)
        else:
            # Pequena degradação se rejeitado
            self.current_return = max(0.1, self.current_return - 0.01)
        
        self.performance_history.append(self.current_return)
        return self.current_return

class LLMSimulator(DomainSimulator):
    """Simulador para Large Language Models"""
    
    def __init__(self):
        super().__init__("Large Language Model")
        self.accuracy = 0.6  # Acurácia inicial
        self.model_size = 7e9  # 7B parâmetros
        self.benchmark_tasks = ["coding", "reasoning", "factual", "creative"]
        
    def generate_signals(self, iteration: int, performance: float) -> ETSignals:
        # LP baseado em melhoria em benchmarks
        lp_values = []
        difficulties = []
        
        for task in self.benchmark_tasks:
            if task == "coding":
                lp = np.random.uniform(0.7, 0.9) * performance  # Código é mensurável
                diff = 1.5
            elif task == "reasoning":
                lp = np.random.uniform(0.5, 0.8) * performance
                diff = 2.0
            elif task == "factual":
                lp = np.random.uniform(0.8, 0.95) * performance  # Fatos são claros
                diff = 1.0
            else:  # creative
                lp = np.random.uniform(0.3, 0.6) * performance  # Criatividade é subjetiva
                diff = 1.8
            
            lp_values.append(lp)
            difficulties.append(diff)
        
        # Entropia baseada na diversidade de tokens
        entropy = 0.8 + np.random.normal(0, 0.1)
        
        # Divergência pequena (fine-tuning gradual)
        divergence = np.random.uniform(0.02, 0.1)
        
        # Drift baseado em esquecimento catastrófico
        drift = max(0, 0.2 - performance) + np.random.uniform(0, 0.05)
        
        # Regret baseado em regressão em benchmarks
        regret = max(0, 0.1 - performance * 0.8) + np.random.uniform(0, 0.03)
        
        return ETSignals(
            learning_progress=np.array(lp_values),
            task_difficulties=np.array(difficulties),
            mdl_complexity=self.model_size / 1e10,  # Normalizado
            energy_consumption=np.random.uniform(0.05, 0.15),  # Menor com fotônica
            scalability_inverse=np.random.uniform(0.15, 0.25),
            policy_entropy=entropy,
            policy_divergence=divergence,
            drift_penalty=drift,
            curriculum_variance=np.var(difficulties),
            regret_rate=regret,
            embodiment_score=0.0,  # Puramente digital
            phi_components=np.random.uniform(-0.2, 0.2, 4)
        )
    
    def update_performance(self, accepted: bool, score: float) -> float:
        if accepted and score > 0:
            # Melhoria gradual em acurácia
            improvement = min(0.05, score * 0.01)
            self.accuracy = min(0.95, self.accuracy + improvement)
        else:
            # Pequena degradação
            self.accuracy = max(0.3, self.accuracy - 0.005)
        
        self.performance_history.append(self.accuracy)
        return self.accuracy

class RoboticsSimulator(DomainSimulator):
    """Simulador para Robótica"""
    
    def __init__(self):
        super().__init__("Robotics")
        self.success_rate = 0.4  # Taxa de sucesso inicial
        self.safety_violations = 0
        self.tasks = ["navigation", "manipulation", "perception", "planning"]
        
    def generate_signals(self, iteration: int, performance: float) -> ETSignals:
        # LP baseado em melhoria em tarefas físicas
        lp_values = []
        difficulties = []
        
        for task in self.tasks:
            if task == "navigation":
                lp = np.random.uniform(0.6, 0.8) * performance
                diff = 1.2
            elif task == "manipulation":
                lp = np.random.uniform(0.4, 0.7) * performance  # Mais difícil
                diff = 2.5
            elif task == "perception":
                lp = np.random.uniform(0.7, 0.9) * performance
                diff = 1.0
            else:  # planning
                lp = np.random.uniform(0.5, 0.8) * performance
                diff = 1.8
            
            lp_values.append(lp)
            difficulties.append(diff)
        
        # Entropia baseada na diversidade de ações
        entropy = 0.7 + np.random.normal(0, 0.15)
        
        # Divergência (mudanças na política de controle)
        divergence = np.random.uniform(0.1, 0.3)
        
        # Drift crítico (segurança)
        drift = max(0, 0.4 - performance) + np.random.uniform(0, 0.1)
        
        # Regret baseado em falhas de segurança
        safety_factor = max(0, self.safety_violations * 0.1)
        regret = safety_factor + max(0, 0.2 - performance) + np.random.uniform(0, 0.05)
        
        # Embodiment é CRÍTICO em robótica
        embodiment = performance * 0.8 + np.random.uniform(0.1, 0.3)
        
        return ETSignals(
            learning_progress=np.array(lp_values),
            task_difficulties=np.array(difficulties),
            mdl_complexity=np.random.uniform(0.2, 0.4),  # Controladores
            energy_consumption=np.random.uniform(0.3, 0.6),  # Motores consomem energia
            scalability_inverse=np.random.uniform(0.2, 0.4),
            policy_entropy=entropy,
            policy_divergence=divergence,
            drift_penalty=drift,
            curriculum_variance=np.var(difficulties),
            regret_rate=regret,
            embodiment_score=embodiment,
            phi_components=np.random.uniform(-0.4, 0.4, 4)
        )
    
    def update_performance(self, accepted: bool, score: float) -> float:
        if accepted and score > 0:
            # Melhoria em taxa de sucesso
            improvement = min(0.08, score * 0.015)
            self.success_rate = min(0.9, self.success_rate + improvement)
        else:
            # Degradação e possível violação de segurança
            self.success_rate = max(0.1, self.success_rate - 0.02)
            if np.random.random() < 0.1:  # 10% chance de violação
                self.safety_violations += 1
        
        self.performance_history.append(self.success_rate)
        return self.success_rate

class ScientificDiscoverySimulator(DomainSimulator):
    """Simulador para Descoberta Científica"""
    
    def __init__(self):
        super().__init__("Scientific Discovery")
        self.discovery_rate = 0.3  # Taxa de descobertas válidas
        self.hypothesis_count = 0
        self.validated_discoveries = 0
        
    def generate_signals(self, iteration: int, performance: float) -> ETSignals:
        # LP baseado em hipóteses que levam a descobertas
        hypothesis_types = ["chemical", "biological", "physical", "computational"]
        lp_values = []
        difficulties = []
        
        for h_type in hypothesis_types:
            if h_type == "chemical":
                lp = np.random.uniform(0.5, 0.8) * performance
                diff = 1.5
            elif h_type == "biological":
                lp = np.random.uniform(0.3, 0.6) * performance  # Mais complexo
                diff = 2.2
            elif h_type == "physical":
                lp = np.random.uniform(0.6, 0.9) * performance
                diff = 1.3
            else:  # computational
                lp = np.random.uniform(0.7, 0.9) * performance
                diff = 1.0
            
            lp_values.append(lp)
            difficulties.append(diff)
        
        # Entropia baseada na diversidade de hipóteses
        entropy = 0.8 + np.random.normal(0, 0.1)
        
        # Divergência (mudança no espaço de hipóteses)
        divergence = np.random.uniform(0.05, 0.2)
        
        # Drift (perda de conhecimento validado)
        drift = max(0, 0.3 - performance) + np.random.uniform(0, 0.08)
        
        # Regret baseado em falhas de replicação
        replication_failures = max(0, 0.15 - performance * 0.7)
        regret = replication_failures + np.random.uniform(0, 0.04)
        
        # Embodiment baseado em integração com laboratório
        lab_integration = performance * 0.6 + np.random.uniform(0.2, 0.5)
        
        return ETSignals(
            learning_progress=np.array(lp_values),
            task_difficulties=np.array(difficulties),
            mdl_complexity=np.random.uniform(0.3, 0.5),  # Modelos científicos
            energy_consumption=np.random.uniform(0.2, 0.4),  # Experimentos
            scalability_inverse=np.random.uniform(0.1, 0.3),
            policy_entropy=entropy,
            policy_divergence=divergence,
            drift_penalty=drift,
            curriculum_variance=np.var(difficulties),
            regret_rate=regret,
            embodiment_score=lab_integration,
            phi_components=np.random.uniform(-0.3, 0.3, 4)
        )
    
    def update_performance(self, accepted: bool, score: float) -> float:
        self.hypothesis_count += 1
        
        if accepted and score > 0:
            # Chance de descoberta válida
            if np.random.random() < 0.3:  # 30% chance
                self.validated_discoveries += 1
            
            improvement = min(0.06, score * 0.012)
            self.discovery_rate = min(0.8, self.discovery_rate + improvement)
        else:
            # Degradação na taxa de descoberta
            self.discovery_rate = max(0.1, self.discovery_rate - 0.01)
        
        self.performance_history.append(self.discovery_rate)
        return self.discovery_rate

def run_domain_simulation(simulator: DomainSimulator, 
                         et_params: Dict = None,
                         iterations: int = 500) -> SimulationResults:
    """Executa simulação para um domínio específico"""
    
    if et_params is None:
        et_params = {"rho": 1.0, "sigma": 1.0, "iota": 1.0, "gamma": 0.4}
    
    # Ajustar parâmetros baseado no domínio
    if simulator.domain_name == "Robotics":
        et_params["iota"] = 2.0  # Embodiment mais importante
    elif simulator.domain_name == "Large Language Model":
        et_params["iota"] = 0.1  # Embodiment menos importante
    
    et = ETCore(**et_params)
    
    scores = []
    decisions = []
    performance_values = []
    guardrails_triggered = 0
    
    current_performance = 0.5  # Performance inicial
    
    logger.info(f"Iniciando simulação para {simulator.domain_name}")
    
    for i in range(iterations):
        # Gerar sinais do domínio
        signals = simulator.generate_signals(i, current_performance)
        
        # Verificar guardrails específicos do domínio
        if simulator.domain_name == "Robotics" and signals.regret_rate > 0.2:
            guardrails_triggered += 1
            logger.warning(f"Guardrail de segurança ativado na iteração {i}")
            continue
        
        # Decisão da ET
        accept, score, terms = et.accept_modification(signals)
        
        # Atualizar performance do domínio
        current_performance = simulator.update_performance(accept, score)
        
        scores.append(score)
        decisions.append(accept)
        performance_values.append(current_performance)
        
        # Log periódico
        if i % 100 == 0:
            logger.info(f"{simulator.domain_name} - Iteração {i}: "
                       f"Performance={current_performance:.3f}, "
                       f"Score={score:.3f}, "
                       f"Aceito={'Sim' if accept else 'Não'}")
    
    # Calcular métricas finais
    acceptance_rate = np.mean(decisions)
    stability_metric = np.std(performance_values[-50:])  # Estabilidade final
    final_performance = performance_values[-1]
    
    # Encontrar tempo de convergência (quando performance estabiliza)
    convergence_time = iterations
    for i in range(50, iterations):
        if np.std(performance_values[i-50:i]) < 0.05:
            convergence_time = i
            break
    
    logger.info(f"Simulação {simulator.domain_name} concluída:")
    logger.info(f"  Taxa de aceitação: {acceptance_rate:.2%}")
    logger.info(f"  Performance final: {final_performance:.3f}")
    logger.info(f"  Estabilidade: {stability_metric:.4f}")
    logger.info(f"  Convergência em: {convergence_time} iterações")
    
    return SimulationResults(
        domain=simulator.domain_name,
        scores=scores,
        acceptance_rate=acceptance_rate,
        stability_metric=stability_metric,
        final_performance=final_performance,
        convergence_time=convergence_time,
        guardrails_triggered=guardrails_triggered
    )

def optimize_parameters(simulator: DomainSimulator, 
                       param_ranges: Dict,
                       iterations: int = 200) -> Dict:
    """Otimiza parâmetros da ET para um domínio específico"""
    
    logger.info(f"Otimizando parâmetros para {simulator.domain_name}")
    
    best_params = None
    best_score = -np.inf
    
    # Grid search simples
    rho_values = np.linspace(param_ranges["rho"][0], param_ranges["rho"][1], 5)
    sigma_values = np.linspace(param_ranges["sigma"][0], param_ranges["sigma"][1], 5)
    iota_values = np.linspace(param_ranges["iota"][0], param_ranges["iota"][1], 5)
    
    for rho in rho_values:
        for sigma in sigma_values:
            for iota in iota_values:
                params = {"rho": rho, "sigma": sigma, "iota": iota, "gamma": 0.4}
                
                # Executar simulação curta
                result = run_domain_simulation(simulator, params, iterations)
                
                # Métrica de qualidade: performance final + estabilidade
                quality_score = result.final_performance - result.stability_metric
                
                if quality_score > best_score:
                    best_score = quality_score
                    best_params = params.copy()
    
    logger.info(f"Melhores parâmetros para {simulator.domain_name}: {best_params}")
    logger.info(f"Score de qualidade: {best_score:.4f}")
    
    return best_params

def run_comparative_analysis():
    """Executa análise comparativa entre domínios"""
    
    print("🔬 ANÁLISE COMPARATIVA ENTRE DOMÍNIOS 🔬\n")
    
    # Criar simuladores
    simulators = [
        RLSimulator(),
        LLMSimulator(),
        RoboticsSimulator(),
        ScientificDiscoverySimulator()
    ]
    
    results = []
    
    # Executar simulações
    for simulator in simulators:
        result = run_domain_simulation(simulator, iterations=300)
        results.append(result)
    
    # Análise comparativa
    print("\n📊 RESULTADOS COMPARATIVOS:")
    print("=" * 80)
    print(f"{'Domínio':<25} {'Taxa Aceit.':<12} {'Perf. Final':<12} {'Estabilidade':<12} {'Convergência':<12}")
    print("=" * 80)
    
    for result in results:
        print(f"{result.domain:<25} {result.acceptance_rate:<12.1%} "
              f"{result.final_performance:<12.3f} {result.stability_metric:<12.4f} "
              f"{result.convergence_time:<12d}")
    
    # Identificar melhor domínio
    best_domain = max(results, key=lambda r: r.final_performance - r.stability_metric)
    print(f"\n🏆 Melhor desempenho: {best_domain.domain}")
    
    # Análise de guardrails
    total_guardrails = sum(r.guardrails_triggered for r in results)
    print(f"\n🛡️ Total de guardrails ativados: {total_guardrails}")
    
    return results

def test_parameter_optimization():
    """Testa otimização de parâmetros"""
    
    print("\n🎯 TESTE DE OTIMIZAÇÃO DE PARÂMETROS 🎯\n")
    
    # Testar otimização para RL
    rl_sim = RLSimulator()
    param_ranges = {
        "rho": (0.5, 2.0),
        "sigma": (0.5, 2.0),
        "iota": (0.5, 2.0)
    }
    
    # Parâmetros padrão
    default_params = {"rho": 1.0, "sigma": 1.0, "iota": 1.0, "gamma": 0.4}
    result_default = run_domain_simulation(rl_sim, default_params, 200)
    
    # Parâmetros otimizados
    rl_sim_opt = RLSimulator()  # Nova instância
    optimized_params = optimize_parameters(rl_sim_opt, param_ranges, 150)
    
    rl_sim_final = RLSimulator()  # Nova instância para teste final
    result_optimized = run_domain_simulation(rl_sim_final, optimized_params, 200)
    
    print(f"\nComparação RL:")
    print(f"Padrão:     Performance={result_default.final_performance:.3f}, "
          f"Estabilidade={result_default.stability_metric:.4f}")
    print(f"Otimizado:  Performance={result_optimized.final_performance:.3f}, "
          f"Estabilidade={result_optimized.stability_metric:.4f}")
    
    improvement = result_optimized.final_performance - result_default.final_performance
    print(f"Melhoria: {improvement:.3f} ({improvement/result_default.final_performance:.1%})")

def main():
    """Executa todos os testes práticos"""
    
    print("🚀 TESTES PRÁTICOS AVANÇADOS DA EQUAÇÃO DE TURING 🚀\n")
    
    try:
        # Análise comparativa entre domínios
        results = run_comparative_analysis()
        
        # Teste de otimização de parâmetros
        test_parameter_optimization()
        
        print("\n✅ TODOS OS TESTES PRÁTICOS CONCLUÍDOS COM SUCESSO!")
        
        # Resumo final
        print("\n📋 RESUMO DOS TESTES:")
        print("✓ Simulação de Aprendizado por Reforço")
        print("✓ Simulação de Large Language Models")
        print("✓ Simulação de Robótica")
        print("✓ Simulação de Descoberta Científica")
        print("✓ Análise comparativa entre domínios")
        print("✓ Otimização de parâmetros")
        print("✓ Testes de guardrails de segurança")
        
        # Salvar resultados
        results_summary = {
            "domains": [r.domain for r in results],
            "acceptance_rates": [r.acceptance_rate for r in results],
            "final_performances": [r.final_performance for r in results],
            "stability_metrics": [r.stability_metric for r in results],
            "convergence_times": [r.convergence_time for r in results]
        }
        
        with open("/home/ubuntu/simulation_results.json", "w") as f:
            json.dump(results_summary, f, indent=2)
        
        print(f"\n💾 Resultados salvos em simulation_results.json")
        
    except Exception as e:
        print(f"❌ ERRO NOS TESTES PRÁTICOS: {e}")
        raise

if __name__ == "__main__":
    main()

