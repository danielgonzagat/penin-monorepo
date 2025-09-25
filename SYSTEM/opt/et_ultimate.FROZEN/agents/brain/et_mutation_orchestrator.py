import time
import random
from datetime import datetime
from pathlib import Path

from agents.brain.et_llm_bridge import requisitar_mutacoes
from agents.brain.et_fusionator import fundir_mutacoes
from agents.brain.et_evaluator import avaliar_mutacoes, selecionar_dominante
from agents.brain.et_mutacao_writer import registrar_mutacao
from agents.brain.et_autocritico import gerar_autocritica
from agents.brain.et_estrategico import avaliar_estrategia
from agents.brain.et_watchdog import loop_guard, enforce_novelty

HIST_DIR = Path("/opt/et_ultimate/history")
HIST_DIR.mkdir(parents=True, exist_ok=True)

random.seed(20250813)

def _log(msg):
    ts = datetime.utcnow().isoformat()
    print(f"[MUTATION_ORCHESTRATOR] {msg}")
    with open(HIST_DIR / "mutation_orchestrator.log", "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")

def executar_ciclo_mutacoes(objetivo, topic_obj):
    """
    Executa um ciclo completo de mutações com múltiplas IAs, fusão, avaliação e seleção do dominante.
    """
    _log(f"🎯 Iniciando ciclo de mutações para objetivo: {objetivo}")

    # 1) Geração multi-IA
    mutacoes = requisitar_mutacoes(objetivo)

    # 2) Fusão de mutações
    mutacao_fundida = fundir_mutacoes(mutacoes)
    if mutacao_fundida:
        mutacoes.append(mutacao_fundida)

    # 3) Enriquecimento e verificação de loops
    for mut in mutacoes:
        mut["autocritica"] = gerar_autocritica(mut.get("eq", ""))
        mut["estrategia"] = avaliar_estrategia(mut.get("eq", ""), mut["autocritica"])
        lg = loop_guard(mut.get("eq", ""))
        mut["loop_stuck"] = lg["stuck"]
        mut["sim_local"] = lg["sim_max"]

    # 4) Portão de novidade
    mutacoes = enforce_novelty(mutacoes, sim_threshold=0.86, min_keep=2)

    # 5) Avaliação e seleção do dominante
    avaliacoes = avaliar_mutacoes(mutacoes)
    dominante = selecionar_dominante(avaliacoes)

    if dominante:
        _log(f"🏆 Dominante: {dominante['ia']} | Score={dominante['score']:.2f} | novelty={dominante.get('novelty', 0.0):.2f}")
        dominante["objetivo"] = objetivo
        dominante["topic"] = topic_obj
        dominante["ts"] = time.time()

        # Registrar dominante
        registrar_mutacao(dominante)
        return dominante
    else:
        _log("⚠️ Nenhuma mutação dominante encontrada.")
        return None

def ciclo_infinito_mutacoes(intervalo_min=10, intervalo_max=20):
    """
    Loop infinito de execução de ciclos de mutações.
    """
    while True:
        try:
            objetivo = "Evoluir continuamente a Equação de Turing e a própria IA"
            topic_obj = "evolution_core"

            dominante = executar_ciclo_mutacoes(objetivo, topic_obj)

            if dominante:
                _log(f"✅ Ciclo concluído. Dominante: {dominante['ia']} | Score={dominante['score']:.2f}")
            else:
                _log("⚠️ Ciclo concluído sem dominante.")

            time.sleep(random.randint(intervalo_min, intervalo_max))
        except Exception as e:
            _log(f"❌ Erro no ciclo: {e}")
            time.sleep(5)
def orquestrar_mutacoes(*args, **kwargs):
    """
    Alias de compatibilidade para a função de orquestração de mutações.
    Mantém compatibilidade com módulos que usam o nome antigo.
    """
    if 'executar_orquestracao' in globals():
        return executar_orquestracao(*args, **kwargs)
    if 'main_orquestracao' in globals():
        return main_orquestracao(*args, **kwargs)
    raise NotImplementedError("Função de orquestração de mutações não implementada.")
