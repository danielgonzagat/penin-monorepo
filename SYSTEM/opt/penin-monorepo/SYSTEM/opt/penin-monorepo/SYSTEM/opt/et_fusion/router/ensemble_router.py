#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ensemble Router com RAG local e professores (GPT‑5 via proxy + modelos locais).
"""

import os
import sys
import time
import json
import asyncio
import traceback
from pathlib import Path
from typing import List, Dict, Any

import aiohttp
from fastapi import FastAPI
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel

# -----------------------------------------------------------------------------
# Configurações de logs e diretórios
# -----------------------------------------------------------------------------
DATA_DIR = "/opt/et_fusion/data"
LOG_DIR = "/opt/et_fusion/logs"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def log(*args: Any) -> None:
    """Grava mensagens de log no console e em arquivo."""
    msg = "[router] " + " ".join(str(x) for x in args)
    print(msg, flush=True)
    try:
        with open(f"{LOG_DIR}/router.out", "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except Exception:
        pass

# -----------------------------------------------------------------------------
# Configuração de RAG local (ETKB)
# -----------------------------------------------------------------------------
ROUTER_DIR = "/opt/et_fusion/router"
if ROUTER_DIR not in sys.path:
    sys.path.append(ROUTER_DIR)

# Tentativa de importar o módulo etkb; se falhar, usa stub
try:
    import etkb  # provê etkb.search(text, k)
except Exception:
    class _ETKBStub:
        def search(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
            # Retorna dicionário vazio, evitando erros ao acessar 'context' ou 'hits'
            return {"context": "", "hits": []}
    etkb = _ETKBStub()

# -----------------------------------------------------------------------------
# Configuração do modelo remoto (GPT‑5 via proxy LiteLLM)
# -----------------------------------------------------------------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL   = os.environ.get("OPENAI_MODEL", "gpt-5")
USE_RESPONSES  = os.environ.get("OPENAI_USE_RESPONSES", "0") == "1"

# -----------------------------------------------------------------------------
# Definição dos professores (GPT‑5 + três modelos locais)
# -----------------------------------------------------------------------------
TEACHERS: List[Dict[str, str]] = [
    {
        "name": "gpt5",
        "kind": "openai",
        "url": "http://127.0.0.1:8003/v1"
    },
    {
        "name": "qwen2.5-7b",
        "kind": "local",
        "url": "http://127.0.0.1:8002/v1/chat/completions"
    },
    {
        "name": "deepseek-r1-qwen7b",
        "kind": "local",
        "url": "http://127.0.0.1:8004/v1/chat/completions"
    },
    {
        "name": "llama-3.1-8b",
        "kind": "local",
        "url": "http://127.0.0.1:8006/v1/chat/completions"
    }
]

# Permite desativar professores específicos via variável de ambiente:
DISABLED: set = set(os.getenv("DISABLE_PROFS", "").split(",")) if os.getenv("DISABLE_PROFS") else set()
if DISABLED:
    original_teachers = list(TEACHERS)
    TEACHERS = [t for t in original_teachers if t["name"] not in DISABLED]

# -----------------------------------------------------------------------------
# Funções utilitárias
# -----------------------------------------------------------------------------
def score_answer(txt: str) -> float:
    """
    Atribui uma pontuação à resposta de um professor.
    Penaliza respostas vazias ou curtas demais e dá bônus para detalhes e passos.
    """
    if not txt or not txt.strip():
        return -1e6
    t = txt.lower()
    s = 0.0
    if "```" in txt:
        s += 0.8  # valoriza blocos de código ou formatação
    if any(x in t for x in ["passo", "step", "1.", "2.", "3."]):
        s += 0.6  # valoriza estrutura passo a passo
    if len(txt) > 400:
        s += 0.5  # valoriza respostas mais extensas
    if any(x in t for x in ["desculp", "não posso", "cannot", "i cannot"]):
        s -= 0.6  # penaliza pedidos de desculpas ou incapacidade
    return s

def build_system_preamble(rag_context: str) -> str:
    """
    Monta a mensagem de sistema com base no contexto recuperado pelo RAG.
    Limita o contexto a 3000 caracteres para não estourar a janela de contexto do LLM.
    """
    rag_context = (rag_context or "")[:3000]
    return (
        "Você é um PROFESSOR ETΩ. Responda com base APENAS no contexto a seguir.\n"
        "Se algo não estiver no contexto, diga que não está e evite inventar.\n"
        "=== CONTEXTO ETΩ INÍCIO ===\n"
        + rag_context +
        "\n=== CONTEXTO ETΩ FIM ==="
    )

# -----------------------------------------------------------------------------
# Funções de consulta aos professores
# -----------------------------------------------------------------------------
async def query_local(session: aiohttp.ClientSession, url: str,
                      sys_txt: str, msgs: List[Dict[str, str]],
                      temperature: float, max_tokens: int) -> str:
    """
    Faz chamada POST a um endpoint de modelo local compatível com OpenAI (local).
    Evita parâmetros desnecessários que possam não ser suportados.
    """
    payload = {
        "model": "local",
        "messages": [{"role": "system", "content": sys_txt}] + msgs
    }
    try:
        async with session.post(url, json=payload, timeout=180) as response:
            j = await response.json()
            try:
                return j["choices"][0]["message"]["content"]
            except Exception:
                return json.dumps(j)[:1500]
    except Exception as e:
        return f"[Erro local] {e}"

async def query_openai(session: aiohttp.ClientSession, base: str,
                       sys_txt: str, msgs: List[Dict[str, str]],
                       temperature: float, max_tokens: int) -> str:
    """
    Faz chamada ao GPT‑5 via proxy LiteLLM (base é http://127.0.0.1:8003/v1).
    Força o uso de temperature=1 quando o modelo é GPT‑5.
    """
    if not OPENAI_API_KEY:
        return ""

    temp = 1 if OPENAI_MODEL.startswith("gpt-5") else float(temperature)

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": OPENAI_MODEL,
        "messages": [{"role": "system", "content": sys_txt}] + msgs,
        "temperature": float(temp),
        "max_tokens": int(max_tokens)
    }

    try:
        async with session.post(f"{base}/chat/completions",
                                headers=headers, json=payload, timeout=180) as resp:
            j = await resp.json()
            try:
                return j["choices"][0]["message"]["content"]
            except Exception:
                return json.dumps(j)[:1500]
    except Exception as e:
        return f"[Erro openai] {e}"

# -----------------------------------------------------------------------------
# Definição da API FastAPI
# -----------------------------------------------------------------------------
class ChatRequest(BaseModel):
    model: str = "ensemble-5x"
    messages: List[Dict[str, str]]
    temperature: float = 1.0
    max_tokens: int = 768
    top_k_ctx: int = 6

app = FastAPI()

@app.get("/healthz")
async def healthz():
    """Rota de saúde para verificação rápida."""
    return PlainTextResponse("ok", status_code=200)

@app.post("/v1/chat/completions")
async def chat(req: ChatRequest):
    """
    Endpoint principal do ensemble. Recupera contexto via RAG (ETKB),
    consulta os professores em paralelo, pontua as respostas,
    escolhe a melhor e retorna junto com fontes.
    """
    ts = int(time.time())
    try:
        # Recupera texto do usuário para o RAG
        user_text = " ".join(
            [m.get("content", "") for m in req.messages if m.get("role") == "user"]
        )[-2000:]

        rag = etkb.search(user_text, k=int(req.top_k_ctx))
        # Se o RAG retornar lista vazia (stub simples), cria estrutura vazia
        if isinstance(rag, list):
            rag = {"context": "", "hits": []}

        sys_txt = build_system_preamble(rag.get("context", ""))

        # Chamar professores em paralelo
        async with aiohttp.ClientSession() as session:
            tasks = []
            for teacher in TEACHERS:
                if teacher["kind"] == "local":
                    tasks.append(
                        query_local(session, teacher["url"], sys_txt,
                                    req.messages, req.temperature, req.max_tokens)
                    )
                else:
                    tasks.append(
                        query_openai(session, teacher["url"], sys_txt,
                                     req.messages, req.temperature, req.max_tokens)
                    )
            outs = await asyncio.gather(*tasks, return_exceptions=True)

        # Monta lista de candidatos com score
        candidates: List[Dict[str, Any]] = []
        for teacher, out in zip(TEACHERS, outs):
            if isinstance(out, Exception):
                log("ERRO professor", teacher["name"], f"{type(out).__name__}: {out}")
                candidates.append(
                    {"name": teacher["name"], "error": str(out), "score": -1e9}
                )
            else:
                text = (out or "").strip()
                candidates.append(
                    {"name": teacher["name"], "content": text, "score": score_answer(text)}
                )

        # Seleciona o melhor
        best = max(candidates, key=lambda x: x["score"])
        # Se nenhum professor foi útil, faz fallback para RAG
        if best["score"] <= -1e5:
            bullets = [
                f"- {Path(h['file']).name} (chunk {h['chunk']})"
                for h in rag.get("hits", [])[:6]
            ]
            best = {
                "name": "fallback-rag",
                "content": "Sem saída útil dos professores.\nFontes RAG:\n"
                + "\n".join(bullets),
                "score": 0.0,
            }

        # Armazena histórico para depuração
        rec = {
            "ts": ts,
            "messages": req.messages,
            "candidates": candidates,
            "best_name": best["name"],
            "best_text": best.get("content", ""),
            "sources": rag.get("hits", []),
        }
        with open(f"{DATA_DIR}/distill_{ts}.json", "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False)

        # Retorna resposta formatada
        return JSONResponse(
            {
                "id": f"chatcmpl-{ts}",
                "object": "chat.completion",
                "created": ts,
                "model": req.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": best.get("content", "")},
                        "finish_reason": "stop",
                    }
                ],
                "sources": rag.get("hits", [])[:5],
            }
        )
    except Exception as e:
        # Em caso de falha inesperada, loga e retorna erro
        log("FATAL chat:", repr(e), "\n", traceback.format_exc())
        return JSONResponse({"error": str(e)}, status_code=500)

# -----------------------------------------------------------------------------
# Execução do servidor (apenas se chamado diretamente)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    log("Iniciando ensemble_router com RAG e professores...")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8800, log_level="info")

