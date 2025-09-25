# /opt/et_ultimate/agents/brain/et_brain_operacional.py
# ETΩ Brain Operacional — imports absolutas corrigidas para o pacote agents.brain

import os
import time
import json
import traceback
from pathlib import Path

# ===== Imports internos (usar caminho absoluto do pacote) =====
from agents.brain.et_goal_planner import definir_objetivo, registrar_feedback
from agents.brain.et_watchdog import iniciar_watchdog, marcar_progresso, relatar_falha
from agents.brain.et_evaluator import avaliar_mutacoes, selecionar_dominante
from agents.brain.et_liga_copilotos import solicitar_mutacoes_liga
from agents.brain.et_llm_bridge import requisitar_mutacoes
from agents.brain.et_paper_ingestor import estudar_topicos_com_ias, fundir_conhecimento_chatgpt
from agents.brain.et_ensemble_fusion import fundir_ensemble
from agents.brain.et_autocritico import ciclo_autocritico
from agents.brain.et_memory_sync import sync_memorias
from agents.brain.et_global_benchmark import benchmark_global, registrar_benchmark
from agents.brain.et_mutation_orchestrator import orquestrar_mutacoes
from agents.brain.et_snapshot_manager import criar_snapshot, restaurar_ultimo_snapshot

# ===== Caminhos =====
BASE_DIR = Path("/opt/et_ultimate")
LOGS = BASE_DIR / "logs"
LOGS.mkdir(parents=True, exist_ok=True)
BRAIN_LOG = LOGS / "brain.log"
BEST_EQ = BASE_DIR / "history" / "BEST_ETΩ.txt"
BEST_EQ.parent.mkdir(parents=True, exist_ok=True)

# ===== Util =====
def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        with open(BRAIN_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass

def bootstrap_best_equation():
    if not BEST_EQ.exists():
        BEST_EQ.write_text("x^2 + y^2", encoding="utf-8")
    return BEST_EQ.read_text(encoding="utf-8").strip()

# ===== Loop principal =====
def rodada_evolucao():
    objetivo, topicos = definir_objetivo()
    log(f"🎯 Objetivo: {objetivo} | 🧭 Tópicos: {', '.join(topicos) if topicos else '—'}")

    # 1) Estudo orientado via IAs (papers/conceitos sintetizados via APIs, sem webcrawl externo)
    try:
        conhecimentos = estudar_topicos_com_ias(topicos=topicos, tentativas=3)
        fusao_conhecimento = fundir_conhecimento_chatgpt(conhecimentos)
        log(f"📚 Conceitos assimilados: {len(fusao_conhecimento.get('conceitos', []))} itens")
    except Exception as e:
        log(f"⚠️ Estudo/assimilação falhou: {e}")
        fusao_conhecimento = {"conceitos": [], "sintese": ""}

    # 2) Geração de mutações (liga de copilotos + bridge unificado)
    base = bootstrap_best_equation()
    mutacoes = []
    try:
        mutacoes += solicitar_mutacoes_liga(base, objetivo, fusao_conhecimento)
    except Exception as e:
        log(f"⚠️ Liga de copilotos falhou: {e}")
    try:
        mutacoes += requisitar_mutacoes(objetivo=objetivo, contexto=fusao_conhecimento.get("sintese", ""))
    except Exception as e:
        log(f"⚠️ Bridge LLM falhou: {e}")
    try:
        mutacoes += orquestrar_mutacoes(base, objetivo, fusao_conhecimento)
    except Exception as e:
        log(f"⚠️ Orquestrador de mutações falhou: {e}")

    if not mutacoes:
        raise RuntimeError("Nenhuma mutação gerada.")

    # 3) Fusão de respostas concorrentes (ensemble)
    try:
        mutacao_fundida = fundir_ensemble(mutacoes, objetivo=objetivo, conhecimento=fusao_conhecimento)
    except Exception as e:
        log(f"⚠️ Fusão de ensemble falhou: {e}")
        mutacao_fundida = max(mutacoes, key=lambda m: m.get("score", 0.0))

    # 4) Avaliação determinística e seleção dominante
    try:
        avaliadas = avaliar_mutacoes([mutacao_fundida], contexto=fusao_conhecimento)
        dominante = selecionar_dominante(avaliadas, baseline=base)
        log(f"🏁 Candidata dominante: score={dominante.get('score')} fonte={dominante.get('source','ensemble')}")
    except Exception as e:
        log(f"⚠️ Avaliação/seleção falhou: {e}")
        dominante = mutacao_fundida

    # 5) Ciclo autocritico com aplicação e verificação
    try:
        resultado_auto = ciclo_autocritico(dominante, conhecimento=fusao_conhecimento)
        if resultado_auto.get("aplicado"):
            BEST_EQ.write_text(resultado_auto.get("equacao_final", base), encoding="utf-8")
            log("✅ Autocrítica aplicada e validada.")
        else:
            log("ℹ️ Autocrítica não aplicada; mantendo estado atual.")
    except Exception as e:
        log(f"⚠️ Autocrítica falhou: {e}")

    # 6) Benchmark & histórico
    try:
        bres = benchmark_global()
        registrar_benchmark(bres)
        log(f"📈 Benchmark atualizado: {json.dumps(bres)[:300]}...")
    except Exception as e:
        log(f"⚠️ Benchmark falhou: {e}")

    # 7) Sincronização de memória
    try:
        sync_memorias()
    except Exception as e:
        log(f"⚠️ Sync memórias falhou: {e}")

    # 8) Feedback do objetivo (fechamento da rodada)
    try:
        registrar_feedback(objetivo, sucesso=True)
    except Exception as e:
        log(f"⚠️ Feedback falhou: {e}")

def main():
    log("🧠 Boot: ETΩ Brain Operacional (imports corrigidos)")
    iniciar_watchdog()
    criar_snapshot(tag="boot")
    try:
        while True:
            try:
                marcar_progresso("rodada:start")
                rodada_evolucao()
                marcar_progresso("rodada:end")
            except KeyboardInterrupt:
                log("🛑 Interrompido pelo usuário.")
                break
            except Exception as e:
                relatar_falha(f"Erro na rodada: {e}\n{traceback.format_exc()}")
                criar_snapshot(tag="falha")
                time.sleep(2)
            time.sleep(1)
    finally:
        criar_snapshot(tag="shutdown")
        log("👋 Encerrando.")

if __name__ == "__main__":
    main()
