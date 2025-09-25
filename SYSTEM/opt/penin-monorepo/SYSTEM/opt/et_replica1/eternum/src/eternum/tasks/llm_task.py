from __future__ import annotations
import os, re, time, json, random
from typing import Tuple

import requests

TAG_OUT_RE = re.compile(r"<OUTPUT>(.*?)</OUTPUT>", re.S | re.I)
TAG_SCORE_RE = re.compile(r"<(?:SCORE|EISCORE)>\s*([0-9]+(?:[.,][0-9]+)?)\s*</(?:SCORE|EISCORE)>", re.I)

def _clamp01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x

def _parse_response(txt: str) -> Tuple[str, float]:
    """
    Extrai texto e score do conteúdo do modelo.
    Aceita:
      - blocos: <OUTPUT>...</OUTPUT> e <SCORE>0.xx</SCORE> ou <EISCORE>...</EISCORE>
      - JSON com {"output": "...", "score": 0.xx}
      - fallback: conteúdo bruto, score=0.0
    """
    if not isinstance(txt, str):
        txt = str(txt)

    # 1) tags
    m_out = TAG_OUT_RE.search(txt)
    m_sc  = TAG_SCORE_RE.search(txt)
    if m_out and m_sc:
        out = m_out.group(1).strip()
        sc  = float(m_sc.group(1).replace(",", "."))
        return out, _clamp01(sc)

    # 2) JSON
    try:
        j = json.loads(txt)
        out = j.get("output") or j.get("text") or ""
        sc  = float(j.get("score", 0.0))
        return (out or txt).strip(), _clamp01(sc)
    except Exception:
        pass

    # 3) fallback
    return txt.strip(), 0.0

class LLMTask:
    """
    Adapta um endpoint compatível com /chat/completions.
    - propose(): chama o endpoint pedindo micro-atualização segura
    - evaluate(): usa o SCORE do próprio modelo como progresso
    - commit(): atualiza estado e baseline do score
    """
    def __init__(self):
        self.text = "seed"
        self.prev_score = 0.0        # baseline p/ EI do orquestrador
        self.last_latency = 0.0
        self.last_raw = ""
        # endpoint / auth
        self.et_url   = os.environ.get("ET_URL", "http://127.0.0.1:8000").rstrip("/")
        self.et_key   = os.environ.get("ET_KEY", "")
        self.et_model = os.environ.get("ET_MODEL", "et-default")

    # — interface esperada pelo orquestrador —
    def export_state(self) -> str:
        return self.text

    def _call_model(self, system: str, user: str, temperature: float = 0.4, max_tokens: int = 512) -> str:
        url = f"{self.et_url}/chat/completions"
        headers = {"Content-Type":"application/json"}
        if self.et_key:
            headers["Authorization"] = f"Bearer {self.et_key}"
        payload = {
            "model": self.et_model,
            "messages": [
                {"role":"system","content": system},
                {"role":"user","content": user}
            ],
            "temperature": float(temperature),
            "max_tokens": int(max_tokens)
        }
        t0 = time.time()
        r = requests.post(url, headers=headers, json=payload, timeout=120)
        self.last_latency = time.time() - t0
        r.raise_for_status()
        data = r.json()
        content = data["choices"][0]["message"]["content"]
        return content

    def propose(self, ctx: str, beta: float) -> str:
        """
        Gera uma pequena modificação (drift controlado). O 'beta' regula ambição.
        """
        # temperatura derivada de beta (mais exploração em fases avançadas)
        temp = 0.2 + 0.8 * max(0.0, min(1.0, beta))
        system = (
            "Você é o otimizador ETΩ. Faça MICRO-atualizações seguras no estado atual, "
            "maximizando o progresso previsto (Expected Improvement) e respeitando guardrails: "
            "entropia mínima, divergência e drift limitados, custo sob orçamento."
            "\nSaída OBRIGATÓRIA neste formato:"
            "\n<OUTPUT>novo_conteudo</OUTPUT>\n<SCORE>0.xx</SCORE>"
        )
        user = (
            f"ESTADO ATUAL:\n{self.text}\n\n"
            f"CONTEXTO (pode ignorar se irrelevante):\n{ctx}\n\n"
            "Regras:\n"
            "- Mude pouco; preferir melhorias incrementais.\n"
            "- SCORE em [0,1], onde 1 é muito melhor que o estado atual.\n"
            "- Não escreva fora dos blocos exigidos."
        )
        try:
            self.last_raw = self._call_model(system, user, temperature=temp, max_tokens=800)
            out, score = _parse_response(self.last_raw)
            # se vier vazio, faz mutação mínima local como fallback
            if not out:
                out = (self.text + " " + "".join(random.choice("abcxyz ") for _ in range(2+int(4*beta)))).strip()
            # guarda provisório do score da proposta (avaliado depois)
            self._proposed_score = float(score)
            return out
        except Exception:
            # falha no endpoint → fallback local leve
            self.last_raw = ""
            self.last_latency = 0.0
            self._proposed_score = 0.0
            return (self.text + " " + "".join(random.choice("abcxyz ") for _ in range(2+int(4*beta)))).strip()

    def evaluate(self, old_text: str, new_text: str):
        """
        Usa o SCORE do modelo como 'progress'. Custo ~ latência + tamanho.
        Retorna: (progress, cost, success)
        """
        score = getattr(self, "_proposed_score", 0.0)
        # se a resposta bruta existir mas sem tag, tenta parsear novamente
        if self.last_raw and score == 0.0:
            _, score = _parse_response(self.last_raw)

        progress = _clamp01(float(score))               # <- EI será calculado sobre isso
        # custo simples: latência + tamanho (aprox. tokens/char)
        cost = float(self.last_latency) + (len(new_text) / 1000.0)
        # sucesso para currículo: melhorou um pouco?
        success = 1.0 if (progress - float(self.prev_score)) >= 0.01 else 0.0
        return progress, cost, success

    def commit(self, new_text: str):
        # confirma a mudança e atualiza baseline de score
        self.text = new_text
        self.prev_score = getattr(self, "_proposed_score", self.prev_score)
