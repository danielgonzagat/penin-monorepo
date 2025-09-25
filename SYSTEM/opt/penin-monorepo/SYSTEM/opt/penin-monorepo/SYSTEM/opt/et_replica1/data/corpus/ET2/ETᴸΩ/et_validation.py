"""
Validação Matemática Detalhada da Equação de Turing
Testes específicos para verificar correção dos cálculos e estabilidade
"""

import numpy as np
import matplotlib.pyplot as plt
from et_core import ETCore, ETSignals, create_test_signals
import logging

# Configurar logging para mostrar detalhes
logging.basicConfig(level=logging.DEBUG)

def test_softmax_stability():
    """Testa estabilidade numérica do softmax"""
    print("=== Teste de Estabilidade do Softmax ===")
    
    et = ETCore()
    
    # Teste com valores extremos
    test_cases = [
        np.array([1, 2, 3]),
        np.array([100, 200, 300]),  # Valores grandes
        np.array([-100, -200, -300]),  # Valores negativos grandes
        np.array([1e-10, 1e-9, 1e-8]),  # Valores muito pequenos
        np.array([0, 0, 0]),  # Zeros
    ]
    
    for i, x in enumerate(test_cases):
        result = et.softmax(x)
        print(f"Caso {i+1}: {x} → {result} (soma: {np.sum(result):.6f})")
        assert np.abs(np.sum(result) - 1.0) < 1e-10, f"Softmax não normalizado no caso {i+1}"
    
    print("✓ Softmax estável numericamente\n")

def test_recurrence_contraction():
    """Testa se a recorrência é realmente contrativa"""
    print("=== Teste de Contração da Recorrência ===")
    
    # Testar diferentes valores de gamma
    gamma_values = [0.1, 0.3, 0.5, 0.6, 0.8]  # Incluindo valores inválidos
    
    for gamma in gamma_values:
        print(f"\nTestando γ = {gamma}")
        
        try:
            et = ETCore(gamma=gamma)
            
            # Executar muitas iterações para testar estabilidade
            states = [et.recurrence_state]
            
            for i in range(100):
                signals = create_test_signals(seed=i)
                new_state = et.update_recurrence(signals)
                states.append(new_state)
            
            states = np.array(states)
            
            # Verificar se converge (variância diminui)
            early_var = np.var(states[:20])
            late_var = np.var(states[-20:])
            
            print(f"  Variância inicial: {early_var:.6f}")
            print(f"  Variância final: {late_var:.6f}")
            print(f"  Estado final: {states[-1]:.6f}")
            print(f"  Range: [{np.min(states):.3f}, {np.max(states):.3f}]")
            
            if gamma <= 0.5:
                assert late_var < early_var or late_var < 0.01, f"Não convergiu para γ={gamma}"
                print(f"  ✓ Convergiu (γ ≤ 0.5)")
            else:
                print(f"  ⚠ γ > 0.5 pode não garantir contração")
                
        except ValueError as e:
            print(f"  ✗ Erro esperado: {e}")
    
    print("\n✓ Teste de contração concluído\n")

def test_zdp_mechanism():
    """Testa o mecanismo de Zona de Desenvolvimento Proximal"""
    print("=== Teste do Mecanismo ZDP ===")
    
    et = ETCore(zdp_quantile=0.7)
    
    # Criar sinais com LP variados
    lp_values = np.array([0.1, 0.3, 0.5, 0.7, 0.9])  # 5 tarefas
    beta_values = np.ones(5)  # Dificuldades iguais
    
    signals = ETSignals(
        learning_progress=lp_values,
        task_difficulties=beta_values,
        mdl_complexity=0.1,
        energy_consumption=0.1,
        scalability_inverse=0.1,
        policy_entropy=0.8,
        policy_divergence=0.1,
        drift_penalty=0.05,
        curriculum_variance=0.3,
        regret_rate=0.05,
        embodiment_score=0.5,
        phi_components=np.array([0.1, 0.2, 0.3, 0.4])
    )
    
    # Calcular progresso
    progress = et.calculate_progress_term(signals)
    
    # Com quantil 0.7, apenas tarefas com LP ≥ 0.7 devem ser consideradas
    # Isso são as tarefas com LP = [0.7, 0.9]
    expected_lp = np.array([0.7, 0.9])
    expected_beta = np.array([1.0, 1.0])
    expected_progress = np.dot(et.softmax(expected_lp), expected_beta)
    
    print(f"LP original: {lp_values}")
    print(f"Progresso calculado: {progress:.4f}")
    print(f"Progresso esperado (ZDP): {expected_progress:.4f}")
    
    # Testar com quantil diferente
    et2 = ETCore(zdp_quantile=0.3)
    progress2 = et2.calculate_progress_term(signals)
    print(f"Progresso com quantil 0.3: {progress2:.4f}")
    
    assert progress2 > progress, "Quantil menor deveria incluir mais tarefas"
    print("✓ Mecanismo ZDP funcionando corretamente\n")

def test_score_components():
    """Testa cada componente do score separadamente"""
    print("=== Teste dos Componentes do Score ===")
    
    # Criar sinais controlados
    signals = ETSignals(
        learning_progress=np.array([0.8, 0.9]),
        task_difficulties=np.array([1.0, 1.5]),
        mdl_complexity=0.2,
        energy_consumption=0.1,
        scalability_inverse=0.15,
        policy_entropy=0.8,
        policy_divergence=0.1,
        drift_penalty=0.05,
        curriculum_variance=0.3,
        regret_rate=0.05,
        embodiment_score=0.6,
        phi_components=np.array([0.1, 0.2, 0.3, 0.4])
    )
    
    et = ETCore(rho=1.0, sigma=1.0, iota=1.0)
    
    # Calcular cada termo
    P_k = et.calculate_progress_term(signals)
    R_k = et.calculate_cost_term(signals)
    S_k = et.calculate_stability_term(signals)
    B_k = et.calculate_embodiment_term(signals)
    
    print(f"Progresso P_k: {P_k:.4f}")
    print(f"Custo R_k: {R_k:.4f}")
    print(f"Estabilidade S_k: {S_k:.4f}")
    print(f"Embodiment B_k: {B_k:.4f}")
    
    # Verificar cálculo manual do custo
    expected_R_k = 0.2 + 0.1 + 0.15  # MDL + Energy + Scalability^-1
    assert np.abs(R_k - expected_R_k) < 1e-10, f"Custo incorreto: {R_k} vs {expected_R_k}"
    
    # Verificar embodiment
    assert np.abs(B_k - 0.6) < 1e-10, f"Embodiment incorreto: {B_k} vs 0.6"
    
    # Score final
    score, terms = et.calculate_score(signals)
    expected_score = P_k - R_k + S_k + B_k
    
    print(f"Score calculado: {score:.4f}")
    print(f"Score esperado: {expected_score:.4f}")
    
    assert np.abs(score - expected_score) < 1e-10, "Score não bate com cálculo manual"
    print("✓ Todos os componentes calculados corretamente\n")

def test_decision_logic():
    """Testa a lógica de decisão de aceitar/rejeitar"""
    print("=== Teste da Lógica de Decisão ===")
    
    et = ETCore()
    
    # Caso 1: Score positivo, baixo regret → ACEITAR
    signals_good = ETSignals(
        learning_progress=np.array([0.9, 0.8]),
        task_difficulties=np.array([1.0, 1.0]),
        mdl_complexity=0.1,
        energy_consumption=0.05,
        scalability_inverse=0.1,
        policy_entropy=0.8,
        policy_divergence=0.1,
        drift_penalty=0.02,
        curriculum_variance=0.3,
        regret_rate=0.02,  # Baixo regret
        embodiment_score=0.7,
        phi_components=np.array([0.1, 0.2, 0.3, 0.4])
    )
    
    accept1, score1, _ = et.accept_modification(signals_good)
    print(f"Caso bom: Score={score1:.4f}, Aceito={accept1}")
    
    # Caso 2: Score positivo, alto regret → REJEITAR
    signals_bad_regret = signals_good
    signals_bad_regret.regret_rate = 0.15  # Alto regret
    
    accept2, score2, _ = et.accept_modification(signals_bad_regret)
    print(f"Caso alto regret: Score={score2:.4f}, Aceito={accept2}")
    
    # Caso 3: Score negativo → REJEITAR
    signals_negative = signals_good
    signals_negative.mdl_complexity = 10.0  # Custo muito alto
    
    accept3, score3, _ = et.accept_modification(signals_negative)
    print(f"Caso score negativo: Score={score3:.4f}, Aceito={accept3}")
    
    assert accept1 == True, "Deveria aceitar caso bom"
    assert accept2 == False, "Deveria rejeitar alto regret"
    assert accept3 == False, "Deveria rejeitar score negativo"
    
    print("✓ Lógica de decisão funcionando corretamente\n")

def test_parameter_sensitivity():
    """Testa sensibilidade aos parâmetros ρ, σ, ι"""
    print("=== Teste de Sensibilidade aos Parâmetros ===")
    
    signals = create_test_signals(seed=42)
    
    # Testar diferentes valores de rho (peso do custo)
    rho_values = [0.5, 1.0, 2.0]
    scores_rho = []
    
    for rho in rho_values:
        et = ETCore(rho=rho, sigma=1.0, iota=1.0)
        score, _ = et.calculate_score(signals)
        scores_rho.append(score)
        print(f"ρ={rho}: Score={score:.4f}")
    
    # Score deve diminuir com aumento de rho (mais penalização de custo)
    assert scores_rho[0] > scores_rho[1] > scores_rho[2], "Score deveria diminuir com aumento de ρ"
    
    print("✓ Sensibilidade aos parâmetros correta\n")

def run_stability_simulation():
    """Executa simulação longa para testar estabilidade"""
    print("=== Simulação de Estabilidade (1000 iterações) ===")
    
    et = ETCore(gamma=0.4)
    
    scores = []
    recurrence_states = []
    acceptance_rate = []
    
    for i in range(1000):
        signals = create_test_signals(seed=i)
        accept, score, _ = et.accept_modification(signals)
        
        scores.append(score)
        recurrence_states.append(et.recurrence_state)
        
        # Taxa de aceitação em janela móvel
        if i >= 99:
            recent_decisions = et.history['decisions'][-100:]
            acceptance_rate.append(np.mean(recent_decisions))
    
    scores = np.array(scores)
    recurrence_states = np.array(recurrence_states)
    acceptance_rate = np.array(acceptance_rate)
    
    print(f"Score médio: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    print(f"Range de scores: [{np.min(scores):.3f}, {np.max(scores):.3f}]")
    print(f"Estado de recorrência final: {recurrence_states[-1]:.6f}")
    print(f"Estabilidade da recorrência: {np.std(recurrence_states):.6f}")
    print(f"Taxa de aceitação média: {np.mean(acceptance_rate):.2%}")
    
    # Verificar se não há explosões numéricas
    assert not np.any(np.isnan(scores)), "Encontrados NaN nos scores"
    assert not np.any(np.isinf(scores)), "Encontrados Inf nos scores"
    assert np.abs(recurrence_states[-1]) < 1.0, "Estado de recorrência explodiu"
    
    print("✓ Sistema estável após 1000 iterações\n")
    
    return scores, recurrence_states, acceptance_rate

def main():
    """Executa todos os testes de validação"""
    print("🧮 VALIDAÇÃO MATEMÁTICA DA EQUAÇÃO DE TURING 🧮\n")
    
    try:
        test_softmax_stability()
        test_recurrence_contraction()
        test_zdp_mechanism()
        test_score_components()
        test_decision_logic()
        test_parameter_sensitivity()
        scores, recurrence, acceptance = run_stability_simulation()
        
        print("🎉 TODOS OS TESTES PASSARAM! 🎉")
        print("\n📊 Resumo da Validação:")
        print(f"✓ Softmax numericamente estável")
        print(f"✓ Recorrência contrativa (γ ≤ 0.5)")
        print(f"✓ ZDP funcionando corretamente")
        print(f"✓ Componentes do score corretos")
        print(f"✓ Lógica de decisão implementada")
        print(f"✓ Sensibilidade aos parâmetros adequada")
        print(f"✓ Sistema estável em simulação longa")
        
        # Diagnósticos finais
        print(f"\n📈 Estatísticas da Simulação:")
        print(f"   Score médio: {np.mean(scores):.4f}")
        print(f"   Desvio padrão: {np.std(scores):.4f}")
        print(f"   Taxa de aceitação: {np.mean(acceptance):.1%}")
        print(f"   Estabilidade recorrência: {np.std(recurrence):.6f}")
        
    except Exception as e:
        print(f"❌ ERRO NA VALIDAÇÃO: {e}")
        raise

if __name__ == "__main__":
    main()

