# et_brain_operacional.py
# 🧠 Cérebro Operacional da ETΩ

import time, json, random
from et_goal_planner import definir_objetivo
from et_llm_bridge import requisitar_mutacoes
from et_fusionator import fundir_mutacoes
from et_evaluator import avaliar_mutacoes, selecionar_dominante
from et_autocritico import gerar_autocritica
from et_estrategico import avaliar_estrategia
from pathlib import Path

HIST = Path("/opt/et_ultimate/history/etomega_scores.jsonl")
SNAPSHOT = Path("/opt/et_ultimate/history/snapshot_ETΩ.json")

print("🧠 Iniciando Cérebro Operacional da ETΩ")

while True:
    try:
        print("\n⏳ Nova rodada de evolução iniciada")

        # 1. Planejar objetivo
        objetivo = definir_objetivo()
        print(f"🎯 Objetivo definido: {objetivo}")

        # 2. Obter mutações das IAs
        mutacoes = requisitar_mutacoes(objetivo)

        # 3. Fundir mutações em uma versão híbrida
        mutacao_fundida = fundir_mutacoes(mutacoes)
        mutacoes.append(mutacao_fundida)

        # 4. Avaliação simbólica estratégica
        for mut in mutacoes:
            mut["autocritica"] = gerar_autocritica(mut["eq"])
            mut["estrategia"] = avaliar_estrategia(mut["eq"], mut["autocritica"])

        # 5. Avaliar e selecionar dominante
        avaliacoes = avaliar_mutacoes(mutacoes)
        dominante = selecionar_dominante(avaliacoes)

        # 6. Atualizar snapshot e registrar
        if dominante:
            print(f"🏆 Mutação dominante: {dominante['ia']} → Score {dominante['score']:.2f}")
            SNAPSHOT.write_text(json.dumps({
                "equation": dominante["eq"],
                "autor": dominante["ia"],
                "score": dominante["score"],
                "autocritica": dominante.get("autocritica", ""),
                "estrategia": dominante.get("estrategia", "")
            }, indent=2), encoding="utf-8")
            with open(HIST, "a", encoding="utf-8") as f:
                f.write(json.dumps(dominante) + "\n")

        time.sleep(15)  # Aguarda 15s antes da próxima rodada

    except Exception as e:
        print(f"⚠️ Erro inesperado: {e}")
        time.sleep(10)
