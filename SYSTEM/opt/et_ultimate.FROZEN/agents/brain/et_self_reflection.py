# et_self_reflection.py
# 📒 Autoquestionamento + Autocrítica Evolutiva (persistente)

import json, time
from pathlib import Path

REFLECT_LOG = Path("/opt/et_ultimate/history/self_reflection.jsonl")
REFLECT_SUM = Path("/opt/et_ultimate/history/self_reflection_last.json")

QUESTOES = [
    "O que ainda não aprendi?",
    "Quais são meus maiores gargalos agora?",
    "Como devo redefinir minha prioridade nesta rodada?"
]

def _append_jl(p: Path, obj: dict):
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def autoquestionar(contexto: dict) -> dict:
    """
    Gera perguntas com placeholders de resposta que podem ser preenchidos por módulos futuros.
    """
    q = [{"q": pergunta, "resposta": ""} for pergunta in QUESTOES]
    return {"ts": time.time(), "perguntas": q, "contexto": contexto}

def autocritica_estruturada(metrics: dict) -> dict:
    """
    Cria uma crítica curta com base nas métricas recebidas.
    """
    texto = []
    if metrics.get("novelty_mean", 0) < 0.5:
        texto.append("Repetição detectada → aumentar diversidade de mutações.")
    if metrics.get("acuracia", 1) < 0.9:
        texto.append("Baixa precisão em tarefas simbólicas → reforçar manipulação simbólica e inferência lógica.")
    if metrics.get("cost", 0) > 0.5:
        texto.append("Custo alto → reduzir temperatura/complexidade ou limitar tokens.")
    critica = "; ".join(texto) or "Desempenho estável; manter exploração moderada."
    return {
        "task": metrics.get("task", ""),
        "resultado": metrics,
        "autocritica": critica
    }

def consolidar(contexto: dict, metrics: dict) -> dict:
    """
    Consolida autoquestionamento + autocrítica e salva em arquivos de histórico.
    """
    ask = autoquestionar(contexto)
    crit = autocritica_estruturada(metrics)
    out = {"ts": time.time(), "autoquestionamento": ask, "autocritica": crit}
    REFLECT_SUM.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    _append_jl(REFLECT_LOG, out)
    return out
