# -*- coding: utf-8 -*-
"""
Registry de provedores LLM + chamada concorrente com timeout longo.
Habilitados: openai (ChatGPT), deepseek, mistral. Grok/Gemini desabilitados.
"""

from __future__ import annotations
import os, time, json, concurrent.futures as cf
from typing import Dict, List, Any, Tuple

import requests

DEFAULT_TIMEOUT = int(os.getenv("ET_TIMEOUT_SECONDS", "600"))  # 10 min
USER_AGENT = "ETOmega/brain-multiapi/1.0"

# Modelos padrão por provedor
MODELS = {
    "openai": os.getenv("OPENAI_MODEL", "gpt-4o-mini-2024-07-18"),
    "deepseek": os.getenv("DEEPSEEK_MODEL", "deepseek-reasoner"),
    "mistral": os.getenv("MISTRAL_MODEL", "mistral-large-latest"),
}

# Habilitados (Grok/Gemini OFF)
ENABLED = [m.strip() for m in os.getenv("ET_PROVIDERS", "openai,deepseek,mistral").split(",") if m.strip()]

# Endpoints + headers (sem SDK, via HTTP)
def _headers_json(api_key: str) -> Dict[str, str]:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "User-Agent": USER_AGENT,
    }

def _call_openai(messages: List[Dict[str, str]], timeout: int) -> str:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY ausente")
    url = os.getenv("OPENAI_BASE", "https://api.openai.com/v1/chat/completions")
    payload = {"model": MODELS["openai"], "messages": messages, "temperature": 0.4}
    r = requests.post(url, headers=_headers_json(api_key), json=payload, timeout=timeout)
    r.raise_for_status()
    j = r.json()
    return j["choices"][0]["message"]["content"].strip()

def _call_deepseek(messages: List[Dict[str, str]], timeout: int) -> str:
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY ausente")
    url = os.getenv("DEEPSEEK_BASE", "https://api.deepseek.com/chat/completions")
    payload = {"model": MODELS["deepseek"], "messages": messages, "temperature": 0.4}
    r = requests.post(url, headers=_headers_json(api_key), json=payload, timeout=timeout)
    r.raise_for_status()
    j = r.json()
    return j["choices"][0]["message"]["content"].strip()

def _call_mistral(messages: List[Dict[str, str]], timeout: int) -> str:
    api_key = os.getenv("MISTRAL_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY ausente")
    url = os.getenv("MISTRAL_BASE", "https://api.mistral.ai/v1/chat/completions")
    payload = {"model": MODELS["mistral"], "messages": messages, "temperature": 0.4}
    r = requests.post(url, headers=_headers_json(api_key), json=payload, timeout=timeout)
    r.raise_for_status()
    j = r.json()
    # Alguns endpoints usam formato alternativo; padronizamos
    if "choices" in j and j["choices"] and "message" in j["choices"][0]:
        return j["choices"][0]["message"]["content"].strip()
    if "output" in j:
        return str(j["output"]).strip()
    return json.dumps(j)

CALLERS = {
    "openai": _call_openai,
    "deepseek": _call_deepseek,
    "mistral": _call_mistral,
    # "grok": _call_xai,    # desabilitado
    # "gemini": _call_gemini,  # desabilitado
}

def enabled_providers() -> List[str]:
    return [p for p in ENABLED if p in CALLERS]

def call_one(provider: str, messages: List[Dict[str, str]], timeout: int = DEFAULT_TIMEOUT) -> Tuple[str, str]:
    """Retorna (provider, texto_ou_erro). Nunca levanta exceção para cima."""
    try:
        txt = CALLERS[provider](messages, timeout)
        return provider, txt
    except Exception as e:
        return provider, f"[erro:{provider}] {type(e).__name__}: {e}"

def call_all(messages: List[Dict[str, str]], timeout: int = DEFAULT_TIMEOUT, max_workers: int = 3) -> Dict[str, str]:
    """
    Chama todos os provedores habilitados em paralelo.
    Retorna dict {provider: texto_ou_erro_str}.
    """
    provs = enabled_providers()
    out: Dict[str, str] = {}
    if not provs:
        return out
    with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(call_one, p, messages, timeout): p for p in provs}
        for fut in cf.as_completed(futs):
            p = futs[fut]
            try:
                _, txt = fut.result()
                out[p] = txt
            except Exception as e:
                out[p] = f"[erro:{p}] {type(e).__name__}: {e}"
    return out
