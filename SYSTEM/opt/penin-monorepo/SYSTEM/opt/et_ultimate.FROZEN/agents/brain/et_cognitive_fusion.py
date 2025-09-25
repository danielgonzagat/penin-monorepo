# -*- coding: utf-8 -*-
"""
Fusão cognitiva: une respostas das copilotas preservando 100% do conteúdo,
elimina redundâncias e produz uma versão superior (ChatGPT como "árbitro").
"""

from __future__ import annotations
import os, requests
from typing import Dict, List

OPENAI_FUSE_MODEL = os.getenv("OPENAI_FUSE_MODEL", "gpt-4o-2024-08-06")
OPENAI_BASE = os.getenv("OPENAI_BASE", "https://api.openai.com/v1/chat/completions")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

SYSTEM_FUSE = (
    "Você é um agente de FUSÃO COGNITIVA. Receberá N respostas sobre o MESMO assunto, "
    "deve: (1) preservar TODOS os fatos e detalhes úteis, (2) remover repetições, "
    "(3) resolver conflitos explicitando critérios, (4) organizar em seções claras, "
    "(5) produzir uma versão FINAL melhor que qualquer individual, "
    "(6) não inventar nada fora das respostas originais."
)

def fuse_responses(step_name: str, prompt: str, per_model: Dict[str, str], temperature: float = 0.2, timeout: int = 600) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY ausente para fusão")
    # Monta contexto com TODAS as respostas
    blocks = []
    for prov, txt in per_model.items():
        blocks.append(f"### {prov}\n{txt}")
    joined = "\n\n".join(blocks)

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_FUSE},
        {"role": "user", "content": f"Etapa: {step_name}\nPrompt original:\n{prompt}\n\nRespostas das copilotas:\n{joined}\n\nProduza a síntese final."}
    ]
    payload = {"model": OPENAI_FUSE_MODEL, "messages": messages, "temperature": temperature}
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {OPENAI_API_KEY}"}
    r = requests.post(OPENAI_BASE, headers=headers, json=payload, timeout=timeout)
    r.raise_for_status()
    j = r.json()
    return j["choices"][0]["message"]["content"].strip()
