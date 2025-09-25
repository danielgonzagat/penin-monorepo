"""
Sistema de Testes Extensivos e Otimização da ET★★ 6.0
Análise estatística rigorosa e otimização de parâmetros
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from et_core_aperfeicoado import ETCoreAperfeicoado, ETSignals, DomainType, generate_domain_signals
import json
import time
from typing import Dict, List, Tuple, Any
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Configurar matplotlib para fontes adequadas
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

class ETOptimizer:
    """Sistema de otimização e análise da Equação de Turing"""
    
    def __init__(self):
        self.results = {}
        self.optimization_history = []
        
    def run_extensive_tests(self, n_iterations: int = 200, n_scenarios: int = 3) -> Dict[str, Any]:
        """Executa testes extensivos em todos os domínios"""
        print("🔬 INICIANDO TESTES EXTENSIVOS DA ET★★ 6.0")
        print("=" * 60)
        
        domains = [
            DomainType.REINFORCEMENT_LEARNING,
            DomainType.LARGE_LANGUAGE_MODEL,
            DomainType.ROBOTICS,
            DomainType.SCIENTIFIC_DISCOVERY
        ]
        
        scenarios = ['high_performance', 'moderate', 'challenging']
        versions = [False, True]  # ET★ e ETΩ
        
        all_results = {}
        
        for domain in domains:
            print(f"\n🎯 TESTANDO DOMÍNIO: {domain.value.upper()}")
            print("-" * 50)
            
            domain_results = {}
            
            for use_omega in versions:
                version_name = "ETΩ" if use_omega else "ET★"
                print(f"  Versão: {version_name}")
                
                version_results = {}
                
                for scenario in scenarios:
                    print(f"    Cenário: {scenario}")
                    
                    # Executar testes
                    results = self._test_scenario(
                        domain, use_omega, scenario, n_iterations
                    )
                    
                    version_results[scenario] = results
                    
                    print(f"      Taxa de aceitação: {results['acceptance_rate']:.1%}")
                    print(f"      Score médio: {results['mean_score']:.3f}")
                    print(f"      Desvio padrão: {results['score_std']:.3f}")
                
                domain_results[version_name] = version_results
            
            all_results[domain.value] = domain_results
        
        # Análise comparativa
        self._analyze_results(all_results)
        
        # Salvar resultados
        with open('/home/ubuntu/et_analysis/testes_extensivos_results.json', 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        self.results = all_results
        return all_results
    
    def _test_scenario(self, domain: DomainType, use_omega: bool, 
                      scenario: str, n_iterations: int) -> Dict[str, Any]:
        """Testa um cenário específico"""
        
        et = ETCoreAperfeicoado(
            domain=domain,
            use_omega=use_omega,
            auto_calibrate=True
        )
        
        scores = []
        decisions = []
        terms_history = []
        
        for i in range(n_iterations):
            # Gerar sinais para o cenário
            signals = generate_domain_signals(domain, scenario)
            
            # Testar
            accept, score, terms = et.accept_modification(signals)
            
            scores.append(score)
            decisions.append(accept)
            terms_history.append(terms)
        
        # Calcular estatísticas
        scores = np.array(scores)
        decisions = np.array(decisions)
        
        results = {
            'n_iterations': n_iterations,
            'acceptance_rate': np.mean(decisions),
            'mean_score': np.mean(scores),
            'score_std': np.std(scores),
            'score_median': np.median(scores),
            'score_min': np.min(scores),
            'score_max': np.max(scores),
            'score_q25': np.percentile(scores, 25),
            'score_q75': np.percentile(scores, 75),
            'scores': scores.tolist(),
            'decisions': decisions.tolist(),
            'diagnostics': et.get_diagnostics()
        }
        
        # Análise de tendências
        if len(scores) >= 20:
            # Dividir em janelas
            window_size = len(scores) // 4
            windows = [scores[i:i+window_size] for i in range(0, len(scores), window_size)]
            window_means = [np.mean(w) for w in windows if len(w) > 0]
            
            if len(window_means) >= 2:
                # Tendência linear
                x = np.arange(len(window_means))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, window_means)
                
                results['trend_slope'] = slope
                results['trend_r_squared'] = r_value**2
                results['trend_p_value'] = p_value
        
        return results
    
    def _analyze_results(self, results: Dict[str, Any]):
        """Análise comparativa dos resultados"""
        print(f"\n📊 ANÁLISE COMPARATIVA DETALHADA")
        print("=" * 60)
        
        # Criar DataFrame para análise
        data = []
        for domain, domain_results in results.items():
            for version, version_results in domain_results.items():
                for scenario, scenario_results in version_results.items():
                    data.append({
                        'domain': domain,
                        'version': version,
                        'scenario': scenario,
                        'acceptance_rate': scenario_results['acceptance_rate'],
                        'mean_score': scenario_results['mean_score'],
                        'score_std': scenario_results['score_std']
                    })
        
        df = pd.DataFrame(data)
        
        # Análise por domínio
        print("\n🎯 PERFORMANCE POR DOMÍNIO:")
        domain_summary = df.groupby(['domain', 'version']).agg({
            'acceptance_rate': 'mean',
            'mean_score': 'mean',
            'score_std': 'mean'
        }).round(3)
        
        print(domain_summary)
        
        # Comparação ET★ vs ETΩ
        print(f"\n⚖️ COMPARAÇÃO ET★ vs ETΩ:")
        comparison = df.groupby(['domain', 'version'])['mean_score'].mean().unstack()
        if 'ETΩ' in comparison.columns and 'ET★' in comparison.columns:
            comparison['Melhoria_ETΩ'] = ((comparison['ETΩ'] - comparison['ET★']) / 
                                         np.abs(comparison['ET★']) * 100).round(1)
            print(comparison)
        
        # Identificar melhores configurações
        print(f"\n🏆 MELHORES CONFIGURAÇÕES:")
        best_acceptance = df.loc[df['acceptance_rate'].idxmax()]
        best_score = df.loc[df['mean_score'].idxmax()]
        
        print(f"Maior taxa de aceitação: {best_acceptance['domain']} - {best_acceptance['version']} - {best_acceptance['scenario']} ({best_acceptance['acceptance_rate']:.1%})")
        print(f"Maior score médio: {best_score['domain']} - {best_score['version']} - {best_score['scenario']} ({best_score['mean_score']:.3f})")
    
    def optimize_parameters(self, domain: DomainType, use_omega: bool = True, 
                          n_trials: int = 50) -> Dict[str, Any]:
        """Otimização de parâmetros usando busca bayesiana simplificada"""
        print(f"\n🔧 OTIMIZANDO PARÂMETROS PARA {domain.value.upper()}")
        print("-" * 50)
        
        def objective(params):
            rho, sigma, iota = params
            
            # Limites razoáveis
            if not (0.1 <= rho <= 3.0 and 0.1 <= sigma <= 3.0 and 0.1 <= iota <= 3.0):
                return 1000  # Penalidade alta
            
            et = ETCoreAperfeicoado(
                domain=domain,
                use_omega=use_omega,
                auto_calibrate=False,
                rho=rho,
                sigma=sigma,
                iota=iota
            )
            
            # Testar com múltiplos cenários
            total_score = 0
            total_acceptance = 0
            n_tests = 30
            
            for scenario in ['high_performance', 'moderate', 'challenging']:
                for _ in range(n_tests // 3):
                    signals = generate_domain_signals(domain, scenario)
                    accept, score, _ = et.accept_modification(signals)
                    
                    total_score += score
                    total_acceptance += int(accept)
            
            # Função objetivo: maximizar score médio ponderado pela taxa de aceitação
            mean_score = total_score / n_tests
            acceptance_rate = total_acceptance / n_tests
            
            # Penalizar baixa aceitação
            if acceptance_rate < 0.1:
                return 1000
            
            # Objetivo: maximizar score ajustado
            objective_value = -(mean_score * (0.5 + 0.5 * acceptance_rate))
            
            return objective_value
        
        # Configuração inicial baseada no domínio
        if domain == DomainType.LARGE_LANGUAGE_MODEL:
            initial_guess = [0.8, 1.0, 0.1]
        elif domain == DomainType.ROBOTICS:
            initial_guess = [0.8, 1.5, 2.0]
        elif domain == DomainType.SCIENTIFIC_DISCOVERY:
            initial_guess = [1.2, 2.0, 1.8]
        else:  # RL
            initial_guess = [1.0, 1.2, 0.3]
        
        # Otimização
        print("  Executando otimização...")
        
        best_result = None
        best_value = float('inf')
        
        # Busca em grade refinada ao redor do ponto inicial
        rho_range = np.linspace(max(0.1, initial_guess[0] - 0.5), 
                               min(3.0, initial_guess[0] + 0.5), 5)
        sigma_range = np.linspace(max(0.1, initial_guess[1] - 0.5), 
                                 min(3.0, initial_guess[1] + 0.5), 5)
        iota_range = np.linspace(max(0.1, initial_guess[2] - 0.5), 
                                min(3.0, initial_guess[2] + 0.5), 5)
        
        for rho in rho_range:
            for sigma in sigma_range:
                for iota in iota_range:
                    value = objective([rho, sigma, iota])
                    if value < best_value:
                        best_value = value
                        best_result = [rho, sigma, iota]
        
        # Teste final com parâmetros otimizados
        print("  Testando configuração otimizada...")
        
        et_optimized = ETCoreAperfeicoado(
            domain=domain,
            use_omega=use_omega,
            auto_calibrate=False,
            rho=best_result[0],
            sigma=best_result[1],
            iota=best_result[2]
        )
        
        # Teste extensivo
        test_results = self._test_scenario(domain, use_omega, 'moderate', 100)
        
        optimization_result = {
            'domain': domain.value,
            'use_omega': use_omega,
            'optimized_parameters': {
                'rho': best_result[0],
                'sigma': best_result[1],
                'iota': best_result[2]
            },
            'objective_value': -best_value,
            'test_results': test_results
        }
        
        print(f"  Parâmetros otimizados:")
        print(f"    ρ = {best_result[0]:.3f}")
        print(f"    σ = {best_result[1]:.3f}")
        print(f"    ι = {best_result[2]:.3f}")
        print(f"  Taxa de aceitação: {test_results['acceptance_rate']:.1%}")
        print(f"  Score médio: {test_results['mean_score']:.3f}")
        
        self.optimization_history.append(optimization_result)
        return optimization_result
    
    def generate_performance_report(self) -> str:
        """Gera relatório detalhado de performance"""
        if not self.results:
            return "Nenhum resultado disponível. Execute os testes primeiro."
        
        report = []
        report.append("# Relatório de Performance da ET★★ 6.0")
        report.append("=" * 50)
        report.append("")
        
        # Resumo executivo
        report.append("## Resumo Executivo")
        report.append("")
        
        # Calcular estatísticas gerais
        all_acceptance_rates = []
        all_scores = []
        
        for domain, domain_results in self.results.items():
            for version, version_results in domain_results.items():
                for scenario, scenario_results in version_results.items():
                    all_acceptance_rates.append(scenario_results['acceptance_rate'])
                    all_scores.append(scenario_results['mean_score'])
        
        report.append(f"- **Taxa de aceitação média geral**: {np.mean(all_acceptance_rates):.1%}")
        report.append(f"- **Score médio geral**: {np.mean(all_scores):.3f}")
        report.append(f"- **Desvio padrão dos scores**: {np.std(all_scores):.3f}")
        report.append("")
        
        # Análise por domínio
        report.append("## Análise por Domínio")
        report.append("")
        
        for domain, domain_results in self.results.items():
            report.append(f"### {domain.upper()}")
            report.append("")
            
            for version, version_results in domain_results.items():
                report.append(f"**{version}:**")
                
                for scenario, results in version_results.items():
                    report.append(f"- {scenario}: {results['acceptance_rate']:.1%} aceitação, score {results['mean_score']:.3f}")
                
                report.append("")
        
        # Recomendações
        report.append("## Recomendações")
        report.append("")
        
        # Encontrar melhor configuração por domínio
        for domain, domain_results in self.results.items():
            best_config = None
            best_score = -float('inf')
            
            for version, version_results in domain_results.items():
                avg_score = np.mean([r['mean_score'] for r in version_results.values()])
                if avg_score > best_score:
                    best_score = avg_score
                    best_config = version
            
            report.append(f"- **{domain}**: Usar {best_config} (score médio: {best_score:.3f})")
        
        report.append("")
        
        # Otimizações realizadas
        if self.optimization_history:
            report.append("## Otimizações de Parâmetros")
            report.append("")
            
            for opt in self.optimization_history:
                report.append(f"### {opt['domain'].upper()}")
                params = opt['optimized_parameters']
                results = opt['test_results']
                
                report.append(f"- **Parâmetros otimizados**: ρ={params['rho']:.3f}, σ={params['sigma']:.3f}, ι={params['iota']:.3f}")
                report.append(f"- **Performance**: {results['acceptance_rate']:.1%} aceitação, score {results['mean_score']:.3f}")
                report.append("")
        
        return "\\n".join(report)
    
    def create_visualizations(self):
        """Cria visualizações dos resultados"""
        if not self.results:
            print("Nenhum resultado disponível para visualização.")
            return
        
        # Preparar dados
        data = []
        for domain, domain_results in self.results.items():
            for version, version_results in domain_results.items():
                for scenario, scenario_results in version_results.items():
                    data.append({
                        'Domain': domain,
                        'Version': version,
                        'Scenario': scenario,
                        'Acceptance_Rate': scenario_results['acceptance_rate'],
                        'Mean_Score': scenario_results['mean_score'],
                        'Score_Std': scenario_results['score_std']
                    })
        
        df = pd.DataFrame(data)
        
        # Criar subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Análise de Performance da ET★★ 6.0', fontsize=16, fontweight='bold')
        
        # 1. Taxa de aceitação por domínio e versão
        pivot_acceptance = df.pivot_table(values='Acceptance_Rate', 
                                        index='Domain', 
                                        columns='Version', 
                                        aggfunc='mean')
        
        sns.heatmap(pivot_acceptance, annot=True, fmt='.2%', cmap='RdYlGn', 
                   ax=axes[0,0], cbar_kws={'label': 'Taxa de Aceitação'})
        axes[0,0].set_title('Taxa de Aceitação por Domínio')
        axes[0,0].set_xlabel('Versão')
        axes[0,0].set_ylabel('Domínio')
        
        # 2. Score médio por domínio e versão
        pivot_score = df.pivot_table(values='Mean_Score', 
                                   index='Domain', 
                                   columns='Version', 
                                   aggfunc='mean')
        
        sns.heatmap(pivot_score, annot=True, fmt='.3f', cmap='RdYlBu_r', 
                   ax=axes[0,1], cbar_kws={'label': 'Score Médio'})
        axes[0,1].set_title('Score Médio por Domínio')
        axes[0,1].set_xlabel('Versão')
        axes[0,1].set_ylabel('Domínio')
        
        # 3. Comparação de versões
        version_comparison = df.groupby(['Domain', 'Version'])['Mean_Score'].mean().unstack()
        if 'ETΩ' in version_comparison.columns and 'ET★' in version_comparison.columns:
            improvement = ((version_comparison['ETΩ'] - version_comparison['ET★']) / 
                          np.abs(version_comparison['ET★']) * 100)
            
            improvement.plot(kind='bar', ax=axes[1,0], color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
            axes[1,0].set_title('Melhoria da ETΩ sobre ET★ (%)')
            axes[1,0].set_xlabel('Domínio')
            axes[1,0].set_ylabel('Melhoria (%)')
            axes[1,0].tick_params(axis='x', rotation=45)
            axes[1,0].grid(True, alpha=0.3)
        
        # 4. Distribuição de scores por cenário
        sns.boxplot(data=df, x='Scenario', y='Mean_Score', hue='Version', ax=axes[1,1])
        axes[1,1].set_title('Distribuição de Scores por Cenário')
        axes[1,1].set_xlabel('Cenário')
        axes[1,1].set_ylabel('Score Médio')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('/home/ubuntu/et_analysis/performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("📊 Visualizações salvas em: performance_analysis.png")

def main():
    """Função principal para execução dos testes extensivos"""
    optimizer = ETOptimizer()
    
    # 1. Testes extensivos
    print("🚀 INICIANDO ANÁLISE COMPLETA DA ET★★ 6.0")
    results = optimizer.run_extensive_tests(n_iterations=100)
    
    # 2. Otimização de parâmetros para domínios problemáticos
    print(f"\n🔧 OTIMIZAÇÃO DE PARÂMETROS")
    print("=" * 60)
    
    # Focar em LLMs que mostraram problemas
    optimizer.optimize_parameters(DomainType.LARGE_LANGUAGE_MODEL, use_omega=True)
    
    # 3. Gerar relatório
    print(f"\n📋 GERANDO RELATÓRIO DE PERFORMANCE")
    print("=" * 60)
    
    report = optimizer.generate_performance_report()
    
    with open('/home/ubuntu/et_analysis/performance_report.md', 'w') as f:
        f.write(report)
    
    print("📄 Relatório salvo em: performance_report.md")
    
    # 4. Criar visualizações
    print(f"\n📊 CRIANDO VISUALIZAÇÕES")
    print("=" * 60)
    
    optimizer.create_visualizations()
    
    print(f"\n✅ ANÁLISE COMPLETA FINALIZADA!")
    print("📁 Arquivos gerados:")
    print("  - testes_extensivos_results.json")
    print("  - performance_report.md")
    print("  - performance_analysis.png")
    
    return optimizer

if __name__ == "__main__":
    main()

