#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETΩ Liga de Copilotos
- Orquestra chamadas às IAs integradas (ChatGPT, DeepSeek, Mistral)
- Gera mutações/ideias a partir de um objetivo textual
- Expõe a função solicitar_mutacoes_liga(objetivo, timeout_s=600, retries=3)
  — que é exatamente o que o et_brain_operacional importa.

Política de tempo:
- timeout_s default = 600s (10 min) por IA; se responder antes, segue o fluxo.
- retries=3 com backoff exponencial (1s, 2s, 4s).
- Se uma IA falhar, seguimos com as outras.
"""

from __future__ import annotations
import os
import time
import json
import math
import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

import requests

# ===== Config =====
MODEL_OPENAI = os.environ.get("ET_OPENAI_MODEL", "gpt-4o-mini-2024-07-18")
MODEL_DEEPSEEK = os.environ.get("ET_DEEPSEEK_MODEL", "deepseek-reasoner")
MODEL_MISTRAL = os.environ.get("ET_MISTRAL_MODEL", "mistral-large-latest")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "")

LOG_DIR = Path(os.environ.get("ET_LOG_DIR", "/opt/et_ultimate/logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
LIGA_LOG = LOG_DIR / "liga_copilotos.log"

# IA registry ativa
IAS = [
    {"name": "chatgpt", "type": "openai", "api_key": OPENAI_API_KEY, "model": MODEL_OPENAI},
    {"name": "deepseek", "type": "deepseek", "api_key": DEEPSEEK_API_KEY, "model": MODEL_DEEPSEEK},
    {"name": "mistral", "type": "mistral", "api_key": MISTRAL_API_KEY, "model": MODEL_MISTRAL},
]

SYSTEM_MUTATOR = (
    "Você é um agente de mutação da Equação de Turing (ETΩ). "
    "Dado um objetivo de evolução, gere uma ou mais propostas concisas que possam: "
    "1) aprimorar a equação ETΩ, 2) melhorar arquitetura/loop cognitivo, 3) aumentar autonomia, "
    "4) reduzir latência e 5) elevar a robustez. "
    "Retorne em JSON com campos: equation | rationale | action_plan (curto)."
)

def _log(event: str, data: Optional[Dict[str, Any]] = None) -> None:
    payload = {"ts": time.time(), "event": event}
    if data:
        payload.update(data)
    with LIGA_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")

# ===== Low-level callers =====
def _call_openai(prompt: str, model: str, api_key: str, timeout_s: int) -> str:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_MUTATOR},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
    r.raise_for_status()
    j = r.json()
    return j["choices"][0]["message"]["content"]

def _call_deepseek(prompt: str, model: str, api_key: str, timeout_s: int) -> str:
    url = "https://api.deepseek.com/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_MUTATOR},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
    r.raise_for_status()
    j = r.json()
    return j["choices"][0]["message"]["content"]

def _call_mistral(prompt: str, model: str, api_key: str, timeout_s: int) -> str:
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_MUTATOR},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
    r.raise_for_status()
    j = r.json()
    # Mistral pode usar "choices"[0]["message"]["content"] igualmente
    return j["choices"][0]["message"]["content"]

def _safe_json_parse(txt: str) -> Dict[str, Any]:
    """
    Aceita tanto JSON puro quanto um bloco contendo JSON.
    Tenta heurísticas simples para localizar o primeiro objeto JSON.
    """
    txt = txt.strip()
    # Caminho feliz: já é JSON
    try:
        return json.loads(txt)
    except Exception:
        pass
    # Heurística: localizar primeiro '{' e último '}'
    first = txt.find("{")
    last = txt.rfind("}")
    if first != -1 and last != -1 and last > first:
        try:
            return json.loads(txt[first:last+1])
        except Exception:
            pass
    # fallback: retornar como texto em rationale
    return {"equation": "", "rationale": txt, "action_plan": ""}

# ===== High-level per-IA =====
def _request_one_ia(ia: Dict[str, str], prompt: str, timeout_s: int, retries: int) -> Dict[str, Any]:
    name = ia["name"]
    api_type = ia["type"]
    api_key = ia.get("api_key", "")
    model = ia.get("model", "")
    backoff = 1.0

    if not api_key:
        _log("ia-skip-no-key", {"ia": name})
        return {"source": name, "equation": "", "rationale": "[skip:missing_api_key]", "action_plan": ""}

    for attempt in range(1, retries + 1):
        try:
            _log("ia-request", {"ia": name, "attempt": attempt})
            if api_type == "openai":
                raw = _call_openai(prompt, model, api_key, timeout_s)
            elif api_type == "deepseek":
                raw = _call_deepseek(prompt, model, api_key, timeout_s)
            elif api_type == "mistral":
                raw = _call_mistral(prompt, model, api_key, timeout_s)
            else:
                return {"source": name, "equation": "", "rationale": f"[skip:unknown_type:{api_type}]", "action_plan": ""}

            parsed = _safe_json_parse(raw)
            parsed["source"] = name
            # sane defaults
            parsed.setdefault("equation", "")
            parsed.setdefault("rationale", "")
            parsed.setdefault("action_plan", "")
            _log("ia-success", {"ia": name, "len": len(raw)})
            return parsed
        except requests.Timeout:
            _log("ia-timeout", {"ia": name, "attempt": attempt})
        except Exception as e:
            _log("ia-error", {"ia": name, "attempt": attempt, "error": repr(e)})

        time.sleep(backoff)
        backoff *= 2.0  # 1s, 2s, 4s

    return {"source": name, "equation": "", "rationale": f"[fail:timeout_or_error]", "action_plan": ""}

# ===== API PRINCIPAL =====
def solicitar_mutacoes_liga(objetivo: str, timeout_s: int = 600, retries: int = 3) -> List[Dict[str, Any]]:
    """
    Retorna uma lista de mutações/ideias propostas pelas IAs ativas.
    Cada item: {source, equation, rationale, action_plan}
    """
    _log("liga-start", {"objetivo": objetivo, "timeout_s": timeout_s, "retries": retries})

    # Prompt orientado ao seu processo
    prompt = (
        "OBJETIVO PRINCIPAL: Evoluir continuamente a Equação de Turing (ETΩ) e a própria IA (autonomia, velocidade, "
        "inteligência, robustez), SEM nunca regredir nem perder funções. "
        "Respeite integralmente a ETΩ. Gere propostas concretas de mutação/ação.\n\n"
        f"OBJETIVO DA RODADA: {objetivo}\n\n"
        "FORMATO DE SAÍDA (JSON): {\"equation\": \"...\", \"rationale\": \"...\", \"action_plan\": \"...\"}\n"
        "Se não puder propor equação, deixe \"equation\" vazio e foque em mudanças de arquitetura/loop."
    )

    results: List[Dict[str, Any]] = []
    for ia in IAS:
        res = _request_one_ia(ia, prompt, timeout_s=timeout_s, retries=retries)
        results.append(res)

    _log("liga-end", {"count": len(results)})
    return results

# ===== CLI rápido =====
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--objetivo", required=True, help="Objetivo da rodada (texto curto)")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--retries", type=int, default=3)
    args = ap.parse_args()

    out = solicitar_mutacoes_liga(args.objetivo, timeout_s=args.timeout, retries=args.retries)
    print(json.dumps(out, ensure_ascii=False, indent=2))
