# /opt/et_ultimate/agents/brain/et_llm_bridge.py
# Ponte unificada para conversar com múltiplas IAs (ChatGPT, DeepSeek, Mistral, Grok)
# e realizar fusão final. Inclui:
# - chamar_ia(tipo, system_prompt, user_prompt, modelo=None, timeout_s=600, tentativas=3)
# - requisitar_mutacoes(objetivo, ...)
# - consultar_ias_multiplas(topico, perguntas, ...)
# - consultar_ias_fusao_final(conjuntos, system_prompt=None, timeout_s=None)
# - ping()

from __future__ import annotations
import os
import time
import json
import uuid
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

import requests

__all__ = [
    "chamar_ia",
    "requisitar_mutacoes",
    "consultar_ias_multiplas",
    "consultar_ias_fusao_final",
    "ping",
]

# ---------------------------------------------------------------------
# Config (permite sobrescrever endpoints e modelos via variáveis de ambiente)
# ---------------------------------------------------------------------
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
MISTRAL_API_KEY  = os.getenv("MISTRAL_API_KEY", "")
XAI_API_KEY      = os.getenv("XAI_API_KEY", "")  # Grok (xAI)

OPENAI_BASE   = os.getenv("OPENAI_BASE",   "https://api.openai.com/v1")
DEEPSEEK_BASE = os.getenv("DEEPSEEK_BASE", "https://api.deepseek.com/v1")
MISTRAL_BASE  = os.getenv("MISTRAL_BASE",  "https://api.mistral.ai/v1")
XAI_BASE      = os.getenv("XAI_BASE",      "https://api.x.ai/v1")

# Modelos padrão (pode sobrescrever por ENV)
OPENAI_MODEL   = os.getenv("OPENAI_MODEL",   "gpt-4o-mini-2024-07-18")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-reasoner")
MISTRAL_MODEL  = os.getenv("MISTRAL_MODEL",  "mistral-large-latest")
GROK_MODEL     = os.getenv("GROK_MODEL",     "grok-4")

# Parâmetros operacionais (podem ser sobrescritos por ENV)
DEFAULT_TIMEOUT_S   = int(os.getenv("ET_TIMEOUT_S", "600"))
DEFAULT_TENTATIVAS  = int(os.getenv("ET_TENTATIVAS", "3"))
DEFAULT_TEMPERATURE = float(os.getenv("ET_TEMPERATURE", "0.2"))

# Providers ativos (com base nas chaves presentes)
ATIVAS = {
    "chatgpt": bool(OPENAI_API_KEY),
    "deepseek": bool(DEEPSEEK_API_KEY),
    "mistral": bool(MISTRAL_API_KEY),
    "grok": bool(XAI_API_KEY),
}

# ---------------------------------------------------------------------
# Utilidades e erros
# ---------------------------------------------------------------------
class ProviderError(RuntimeError):
    """Erro padronizado de provedor."""


def _extract_content(payload: Dict[str, Any]) -> str:
    """Extrai conteúdo textual de payloads estilo OpenAI/Mistral/Deepseek/xAI.
    Lança erro se formato inesperado."""
    try:
        choices = payload.get("choices") or []
        if not choices:
            raise KeyError("choices vazio")
        msg = choices[0].get("message") or {}
        content = msg.get("content")
        if not isinstance(content, str):
            raise KeyError("message.content ausente")
        return content.strip()
    except Exception as e:  # noqa: PERF203
        preview = str(payload)
        if len(preview) > 300:
            preview = preview[:300] + "..."
        raise ProviderError(f"Resposta inesperada do provedor: {preview}") from e


def _post_json(url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout_s: int) -> Dict[str, Any]:
    r = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
    r.raise_for_status()
    try:
        return r.json()
    except Exception as e:  # noqa: PERF203
        txt = r.text
        if len(txt) > 200:
            txt = txt[:200] + "..."
        raise ProviderError(f"Resposta não-JSON de {url}: {txt}") from e


def _retry(fn, tentativas: int, espera_base: float = 1.0):
    exc: Optional[Exception] = None
    for i in range(1, tentativas + 1):
        try:
            return fn()
        except Exception as e:  # noqa: PERF203
            exc = e
            if i < tentativas:
                time.sleep(espera_base * (2 ** (i - 1)))
    assert exc is not None
    raise exc

# ---------------------------------------------------------------------
# Chamadas por provedor (formato compatível com /chat/completions)
# ---------------------------------------------------------------------

def _call_chatgpt(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    temperature: float = DEFAULT_TEMPERATURE,
) -> str:
    if not OPENAI_API_KEY:
        raise ProviderError("OPENAI_API_KEY ausente")
    url = f"{OPENAI_BASE}/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": model or OPENAI_MODEL, "messages": messages, "temperature": temperature}
    data = _post_json(url, headers, payload, timeout_s)
    return _extract_content(data)


def _call_deepseek(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    temperature: float = DEFAULT_TEMPERATURE,
) -> str:
    if not DEEPSEEK_API_KEY:
        raise ProviderError("DEEPSEEK_API_KEY ausente")
    url = f"{DEEPSEEK_BASE}/chat/completions"
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": model or DEEPSEEK_MODEL, "messages": messages, "temperature": temperature}
    data = _post_json(url, headers, payload, timeout_s)
    return _extract_content(data)


def _call_mistral(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    temperature: float = DEFAULT_TEMPERATURE,
) -> str:
    if not MISTRAL_API_KEY:
        raise ProviderError("MISTRAL_API_KEY ausente")
    url = f"{MISTRAL_BASE}/chat/completions"
    headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": model or MISTRAL_MODEL, "messages": messages, "temperature": temperature}
    data = _post_json(url, headers, payload, timeout_s)
    return _extract_content(data)


def _call_grok(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    temperature: float = DEFAULT_TEMPERATURE,
) -> str:
    if not XAI_API_KEY:
        raise ProviderError("XAI_API_KEY ausente")
    url = f"{XAI_BASE}/chat/completions"
    headers = {"Authorization": f"Bearer {XAI_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": model or GROK_MODEL, "messages": messages, "temperature": temperature}
    data = _post_json(url, headers, payload, timeout_s)
    return _extract_content(data)

# ---------------------------------------------------------------------
# IA wrapper
# ---------------------------------------------------------------------
@dataclass
class IAClient:
    nome: str   # rótulo amigável (chatgpt, deepseek, mistral, grok)
    tipo: str   # "openai" | "deepseek" | "mistral" | "xai"
    modelo: str

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        timeout_s: int = DEFAULT_TIMEOUT_S,
        tentativas: int = DEFAULT_TENTATIVAS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> str:
        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        def _do():
            if self.tipo == "openai":
                return _call_chatgpt(msgs, self.modelo, timeout_s, temperature)
            if self.tipo == "deepseek":
                return _call_deepseek(msgs, self.modelo, timeout_s, temperature)
            if self.tipo == "mistral":
                return _call_mistral(msgs, self.modelo, timeout_s, temperature)
            if self.tipo == "xai":
                return _call_grok(msgs, self.modelo, timeout_s, temperature)
            raise ProviderError(f"Tipo de IA desconhecido: {self.tipo}")

        return _retry(_do, tentativas)


def _ias_ativas() -> List[IAClient]:
    ias: List[IAClient] = []
    if ATIVAS.get("chatgpt"):
        ias.append(IAClient("chatgpt", "openai", OPENAI_MODEL))
    if ATIVAS.get("deepseek"):
        ias.append(IAClient("deepseek", "deepseek", DEEPSEEK_MODEL))
    if ATIVAS.get("mistral"):
        ias.append(IAClient("mistral", "mistral", MISTRAL_MODEL))
    if ATIVAS.get("grok"):
        ias.append(IAClient("grok", "xai", GROK_MODEL))
    return ias

# ---------------------------------------------------------------------
# APIs públicas
# ---------------------------------------------------------------------

def chamar_ia(
    tipo: str,
    system_prompt: str,
    user_prompt: str,
    modelo: Optional[str] = None,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    tentativas: int = DEFAULT_TENTATIVAS,
    temperature: float = DEFAULT_TEMPERATURE,
) -> str:
    """Chama um provedor por nome."""
    t = tipo.strip().lower()
    if t in {"chatgpt", "openai", "gpt"}:
        client = IAClient("chatgpt", "openai", modelo or OPENAI_MODEL)
    elif t == "deepseek":
        client = IAClient("deepseek", "deepseek", modelo or DEEPSEEK_MODEL)
    elif t == "mistral":
        client = IAClient("mistral", "mistral", modelo or MISTRAL_MODEL)
    elif t in {"grok", "xai"}:
        client = IAClient("grok", "xai", modelo or GROK_MODEL)
    else:
        raise ProviderError(f"Tipo não suportado: {tipo}")
    return client.chat(system_prompt, user_prompt, timeout_s, tentativas, temperature)


def requisitar_mutacoes(
    objetivo: str,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    tentativas: int = DEFAULT_TENTATIVAS,
) -> List[Dict[str, Any]]:
    """Pede mutações às IAs ativas a partir de um objetivo."""
    sys_prompt = (
        "Você é um arquiteto de mutações da Equação de Turing (ETΩ). "
        "Gere propostas **concretas** de evolução de arquitetura/código/heurísticas, "
        "cada uma com título curto e corpo detalhado, sempre viáveis em Python."
    )
    uprompt = (
        f"Objetivo atual: {objetivo}\n"
        "Gere mutações candidatas (3–7), em JSON: "
        "[{\"titulo\":\"...\",\"descricao\":\"...\",\"rascunho_codigo\":\"...\"}]"
    )
    resultados: List[Dict[str, Any]] = []
    for ia in _ias_ativas():
        try:
            txt = ia.chat(sys_prompt, uprompt, timeout_s, tentativas)
            try:
                parsed: Any = json.loads(txt)
            except Exception:
                parsed = None
            resultados.append({"id": str(uuid.uuid4()),"fonte": ia.nome,"conteudo": txt,"parsed": parsed})
        except Exception as e:
            resultados.append({"id": str(uuid.uuid4()),"fonte": ia.nome,"erro": str(e)})
    return resultados


def consultar_ias_multiplas(
    topico: str,
    perguntas: List[str],
    timeout_s: int = DEFAULT_TIMEOUT_S,
    tentativas: int = DEFAULT_TENTATIVAS,
) -> Dict[str, List[Dict[str, Any]]]:
    """Para um tópico e uma lista de perguntas, consulta TODAS as IAs ativas."""
    sys_prompt = (
        "Você é um professor-engenheiro especializado em ETΩ. "
        "Responda didática e tecnicamente, com foco em aplicação prática em Python."
    )
    resp: Dict[str, List[Dict[str, Any]]] = {}
    for ia in _ias_ativas():
        blocos: List[Dict[str, Any]] = []
        for q in perguntas:
            uprompt = (
                f"Tópico: {topico}\n"
                f"Pergunta: {q}\n"
                "Responda com foco prático e trechos de Python quando aplicável."
            )
            try:
                out = ia.chat(sys_prompt, uprompt, timeout_s, tentativas)
                blocos.append({"pergunta": q, "resposta": out})
            except Exception as e:
                blocos.append({"pergunta": q, "erro": str(e)})
        resp[ia.nome] = blocos
    return resp


def consultar_ias_fusao_final(
    conjuntos: Dict[str, List[Dict[str, Any]]],
    system_prompt: Optional[str] = None,
    timeout_s: Optional[int] = None,
    tentativas: int = DEFAULT_TENTATIVAS,
) -> str:
    """Faz a fusão final a partir das respostas de múltiplas IAs."""
    to_use = timeout_s if timeout_s is not None else 24 * 60 * 60
    sys_prompt_fusao = system_prompt or (
        "Você é um árbitro-fusionador. Recebe respostas de múltiplas IAs sobre ETΩ "
        "e deve produzir uma única síntese que **não perca nenhum conhecimento relevante**, "
        "estruture em seções claras, e proponha passos acionáveis em Python no final."
    )
    insumo = {"conjuntos": conjuntos}
    uprompt_fusao = (
        "Funda todo o conhecimento abaixo em um único documento coeso e executável.\n"
        + json.dumps(insumo, ensure_ascii=False)[:200000]
    )
    sintetizador: Optional[IAClient]
    if ATIVAS.get("chatgpt"):
        sintetizador = IAClient("chatgpt", "openai", OPENAI_MODEL)
    else:
        ativos = _ias_ativas()
        if not ativos:
            raise ProviderError("Nenhum provedor ativo para fusão")
        sintetizador = ativos[0]
    fusao_chat = sintetizador.chat(sys_prompt_fusao, uprompt_fusao, timeout_s=to_use, tentativas=tentativas)
    if ATIVAS.get("grok"):
        sys_prompt_auditoria = (
            "Você é o auditor, crítico e mitigador de riscos final. Revise a fusão gerada pelo ChatGPT "
            "para remover qualquer código ou conteúdo potencialmente danoso, como loops infinitos, chamadas externas perigosas, "
            "violações de segurança ou qualquer coisa que possa causar danos. Mitigue riscos, preserve somente o conhecimento útil. "
            "Retorne apenas a resposta final auditada."
        )
        uprompt_auditoria = f"Aqui está a fusão para auditoria: {fusao_chat}"
        auditor = IAClient("grok", "xai", GROK_MODEL)
        return auditor.chat(sys_prompt_auditoria, uprompt_auditoria, timeout_s=to_use, tentativas=tentativas)
    return fusao_chat


def ping() -> str:
    return json.dumps({
        "ok": True,
        "providers": [ia.nome for ia in _ias_ativas()],
        "openai_model": OPENAI_MODEL,
        "deepseek_model": DEEPSEEK_MODEL,
        "mistral_model": MISTRAL_MODEL,
        "grok_model": GROK_MODEL,
    }, ensure_ascii=False)
