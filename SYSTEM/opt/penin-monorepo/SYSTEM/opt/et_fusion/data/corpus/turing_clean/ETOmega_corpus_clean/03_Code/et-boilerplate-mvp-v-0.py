# ETΩ Boilerplate (MVP v0.1)
# -------------------------------------------------------------
# Estrutura mínima para um loop evolutivo que:
#  - Gera mutações (prompts/temperatura/estilo)
#  - Avalia em tarefas baratas (mini suites)
#  - Calcula Expected Improvement (EI) + penalidades (custo, KL, entropia)
#  - Seleciona vencedoras e atualiza o estado/baseline
# -------------------------------------------------------------
# Quickstart:
# 1) Crie um venv e instale requirements.txt
# 2) Exporte as chaves das IAs se for usar runners reais (opcional)
# 3) Rode:  python -m et_omega.orchestrator.run --rounds 5 --config configs/default.yaml
# Logs e resultados ficam em ./reports/ e no SQLite ./registry/etomega.db
# -------------------------------------------------------------

# ======================
# requirements.txt
# ======================
# (dependências enxutas)
# pydantic>=2.7.0
# httpx>=0.27.0
# numpy>=1.26.0

# ======================
# pyproject.toml (opcional)
# ======================
# [project]
# name = "et-omega"
# version = "0.1.0"
# requires-python = ">=3.10"
# dependencies = ["pydantic>=2.7.0", "httpx>=0.27.0", "numpy>=1.26.0"]

# ======================
# et_omega/__init__.py
# ======================

# vazio proposital (marca o pacote)

# ======================
# et_omega/utils/stats.py
# ======================
from __future__ import annotations
import math
from collections import Counter
from typing import List, Tuple
import numpy as np

SQRT2 = math.sqrt(2.0)
SQRT2PI = math.sqrt(2.0 * math.pi)


def norm_pdf(z: float) -> float:
    return math.exp(-0.5 * z * z) / SQRT2PI


def norm_cdf(z: float) -> float:
    # CDF normal padrão via erf
    return 0.5 * (1.0 + math.erf(z / SQRT2))


def expected_improvement(mu: float, sigma: float, f_best: float, xi: float = 0.01) -> float:
    eps = 1e-8
    imp = mu - f_best - xi
    if sigma < eps:
        return max(0.0, imp)
    z = imp / (sigma + eps)
    return imp * norm_cdf(z) + sigma * norm_pdf(z)


def bootstrap_mean_std(values: List[float], iters: int = 200, seed: int = 42) -> Tuple[float, float]:
    # Bootstrapping simples para estimar média e desvio
    rng = np.random.default_rng(seed)
    values = np.array(values, dtype=float)
    if len(values) == 0:
        return 0.0, 0.0
    means = []
    n = len(values)
    for _ in range(iters):
        sample = values[rng.integers(0, n, size=n)]
        means.append(sample.mean())
    return float(np.mean(means)), float(np.std(means) + 1e-8)


def text_entropy(s: str) -> float:
    # Entropia de Shannon aproximada sobre distribuição de caracteres
    if not s:
        return 0.0
    counts = Counter(s)
    n = sum(counts.values())
    probs = [c / n for c in counts.values()]
    return -sum(p * math.log(p + 1e-12) for p in probs)


def kl_divergence_char(p_text: str, q_text: str) -> float:
    # KL(P||Q) aproximado sobre chars (fallback quando não há logprobs)
    # P = distribuição de p_text; Q = q_text
    p_counts = Counter(p_text) or Counter({" ": 1})
    q_counts = Counter(q_text) or Counter({" ": 1})
    p_total = sum(p_counts.values())
    q_total = sum(q_counts.values())
    chars = set(p_counts) | set(q_counts)
    kl = 0.0
    for ch in chars:
        p = p_counts.get(ch, 0) / p_total
        q = q_counts.get(ch, 0) / q_total
        if p > 0:
            kl += p * math.log((p + 1e-12) / (q + 1e-12))
    return kl

# ======================
# et_omega/registry/sqlite_registry.py
# ======================
import os
import sqlite3
from typing import Any, Dict, List
import json

class Registry:
    def __init__(self, db_path: str = "registry/etomega.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self._init()

    def _init(self):
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS rounds (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts DATETIME DEFAULT CURRENT_TIMESTAMP,
                config_json TEXT
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS candidates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                round_id INTEGER,
                mutation_json TEXT,
                provider TEXT,
                prompt_preview TEXT,
                response_preview TEXT,
                cost FLOAT,
                entropy FLOAT,
                kl FLOAT,
                mu FLOAT,
                sigma FLOAT,
                ei FLOAT,
                score FLOAT,
                metrics_json TEXT
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT
            );
            """
        )
        self.conn.commit()

    def log_round(self, config: Dict[str, Any]) -> int:
        cur = self.conn.cursor()
        cur.execute("INSERT INTO rounds(config_json) VALUES (?)", (json.dumps(config),))
        self.conn.commit()
        return cur.lastrowid

    def log_candidate(self, round_id: int, record: Dict[str, Any]):
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO candidates (round_id, mutation_json, provider, prompt_preview, response_preview,
                                    cost, entropy, kl, mu, sigma, ei, score, metrics_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                round_id,
                json.dumps(record.get("mutation", {})),
                record.get("provider", ""),
                (record.get("prompt") or "")[:280],
                (record.get("response") or "")[:280],
                float(record.get("cost", 0.0)),
                float(record.get("entropy", 0.0)),
                float(record.get("kl", 0.0)),
                float(record.get("mu", 0.0)),
                float(record.get("sigma", 0.0)),
                float(record.get("ei", 0.0)),
                float(record.get("score", 0.0)),
                json.dumps(record.get("metrics", {})),
            ),
        )
        self.conn.commit()

    def set_meta(self, key: str, value: Any):
        cur = self.conn.cursor()
        cur.execute("REPLACE INTO meta(key,value) VALUES (?, ?)", (key, json.dumps(value)))
        self.conn.commit()

    def get_meta(self, key: str, default=None):
        cur = self.conn.cursor()
        cur.execute("SELECT value FROM meta WHERE key=?", (key,))
        row = cur.fetchone()
        if not row:
            return default
        try:
            return json.loads(row[0])
        except Exception:
            return row[0]

# ======================
# et_omega/runners/base.py
# ======================
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class LLMResponse:
    text: str
    tokens_in: int
    tokens_out: int
    cost: float
    raw: Optional[Dict[str, Any]] = None

class LLMRunner:
    name: str = "base"

    def chat(self, system: str, user: str, temperature: float = 0.2, max_tokens: int = 512, **kwargs) -> LLMResponse:
        raise NotImplementedError

# ======================
# et_omega/runners/mock_runner.py
# ======================
import os
import random

class MockRunner(LLMRunner):
    name = "mock"

    def chat(self, system: str, user: str, temperature: float = 0.2, max_tokens: int = 256, **kwargs) -> LLMResponse:
        # Simula variação com base em temperatura e hash do prompt
        seed = hash(user + system) % (10**6)
        rng = random.Random(seed)
        noise = (rng.random() - 0.5) * 2 * temperature
        base_answer = "FINAL ANSWER: " + str(int(abs(seed % 97) + noise * 10))
        text = (
            f"[MOCK] sys_hint={system[:24]}... temp={temperature:.2f}
"
            f"{base_answer}
"
        )
        tokens_in = min(256, len(system) // 4 + len(user) // 4)
        tokens_out = min(64, len(text) // 4)
        cost = 0.0
        return LLMResponse(text=text, tokens_in=tokens_in, tokens_out=tokens_out, cost=cost, raw={"seed": seed})

# ======================
# et_omega/runners/openai_runner.py
# ======================
import os
import time
import httpx

class OpenAIRunner(LLMRunner):
    name = "openai"

    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.max_retries = int(os.getenv("OPENAI_MAX_RETRIES", "4"))

    def chat(self, system: str, user: str, temperature: float = 0.2, max_tokens: int = 512, **kwargs) -> LLMResponse:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY não definido")
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        backoff = 1.0
        for attempt in range(self.max_retries):
            try:
                with httpx.Client(timeout=60.0) as client:
                    r = client.post(url, headers=headers, json=payload)
                if r.status_code == 200:
                    data = r.json()
                    choice = data["choices"][0]["message"]["content"]
                    usage = data.get("usage", {})
                    ti = int(usage.get("prompt_tokens", 0))
                    to = int(usage.get("completion_tokens", 0))
                    # custo aproximado pode ser plugado via env (preços variam)
                    cost = 0.0
                    return LLMResponse(text=choice, tokens_in=ti, tokens_out=to, cost=cost, raw=data)
                elif r.status_code in (429, 500, 502, 503):
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 16)
                    continue
                else:
                    raise RuntimeError(f"OpenAI HTTP {r.status_code}: {r.text[:200]}")
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(backoff)
                backoff = min(backoff * 2, 16)
        raise RuntimeError("Falha OpenAI após retries")

# ======================
# et_omega/runners/deepseek_runner.py
# ======================
import os
import time
import httpx

class DeepSeekRunner(LLMRunner):
    name = "deepseek"

    def __init__(self):
        self.api_key = os.getenv("DEEPSEEK_API_KEY", "")
        self.base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        self.model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        self.max_retries = int(os.getenv("DEEPSEEK_MAX_RETRIES", "4"))

    def chat(self, system: str, user: str, temperature: float = 0.2, max_tokens: int = 512, **kwargs) -> LLMResponse:
        if not self.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY não definido")
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        backoff = 1.0
        for attempt in range(self.max_retries):
            try:
                with httpx.Client(timeout=60.0) as client:
                    r = client.post(url, headers=headers, json=payload)
                if r.status_code == 200:
                    data = r.json()
                    choice = data["choices"][0]["message"]["content"]
                    usage = data.get("usage", {})
                    ti = int(usage.get("prompt_tokens", 0))
                    to = int(usage.get("completion_tokens", 0))
                    cost = 0.0
                    return LLMResponse(text=choice, tokens_in=ti, tokens_out=to, cost=cost, raw=data)
                elif r.status_code in (429, 500, 502, 503):
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 16)
                    continue
                else:
                    raise RuntimeError(f"DeepSeek HTTP {r.status_code}: {r.text[:200]}")
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(backoff)
                backoff = min(backoff * 2, 16)
        raise RuntimeError("Falha DeepSeek após retries")

# ======================
# et_omega/runners/mistral_runner.py
# ======================
import os
import time
import httpx

class MistralRunner(LLMRunner):
    name = "mistral"

    def __init__(self):
        self.api_key = os.getenv("MISTRAL_API_KEY", "")
        self.base_url = os.getenv("MISTRAL_BASE_URL", "https://api.mistral.ai/v1")
        self.model = os.getenv("MISTRAL_MODEL", "mistral-large-latest")
        self.max_retries = int(os.getenv("MISTRAL_MAX_RETRIES", "4"))

    def chat(self, system: str, user: str, temperature: float = 0.2, max_tokens: int = 512, **kwargs) -> LLMResponse:
        if not self.api_key:
            raise RuntimeError("MISTRAL_API_KEY não definido")
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        backoff = 1.0
        for attempt in range(self.max_retries):
            try:
                with httpx.Client(timeout=60.0) as client:
                    r = client.post(url, headers=headers, json=payload)
                if r.status_code == 200:
                    data = r.json()
                    choice = data["choices"][0]["message"]["content"]
                    usage = data.get("usage", {})
                    ti = int(usage.get("prompt_tokens", 0))
                    to = int(usage.get("completion_tokens", 0))
                    cost = 0.0
                    return LLMResponse(text=choice, tokens_in=ti, tokens_out=to, cost=cost, raw=data)
                elif r.status_code in (429, 500, 502, 503):
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 16)
                    continue
                else:
                    raise RuntimeError(f"Mistral HTTP {r.status_code}: {r.text[:200]}")
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(backoff)
                backoff = min(backoff * 2, 16)
        raise RuntimeError("Falha Mistral após retries")

# ======================
import os
import random

class MockRunner(LLMRunner):
    name = "mock"

    def chat(self, system: str, user: str, temperature: float = 0.2, max_tokens: int = 256, **kwargs) -> LLMResponse:
        # Simula variação com base em temperatura e hash do prompt
        seed = hash(user + system) % (10**6)
        rng = random.Random(seed)
        noise = (rng.random() - 0.5) * 2 * temperature
        base_answer = "FINAL ANSWER: " + str(int(abs(seed % 97) + noise * 10))
        text = (
            f"[MOCK] sys_hint={system[:24]}... temp={temperature:.2f}\n"
            f"{base_answer}\n"
        )
        tokens_in = min(256, len(system) // 4 + len(user) // 4)
        tokens_out = min(64, len(text) // 4)
        cost = 0.0
        return LLMResponse(text=text, tokens_in=tokens_in, tokens_out=tokens_out, cost=cost, raw={"seed": seed})

# ======================
# et_omega/mutators/base.py
# ======================
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class Mutation:
    kind: str
    params: Dict[str, Any]

class Mutator:
    def apply(self, state: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

# ======================
# et_omega/mutators/library.py
# ======================
from typing import Dict, Any, List
from .base import Mutator, Mutation

class TemperatureShift(Mutator):
    def __init__(self, delta: float):
        self.delta = delta

    def apply(self, state: Dict[str, Any]) -> Dict[str, Any]:
        new_state = dict(state)
        t = float(new_state.get("temperature", 0.2)) + self.delta
        new_state["temperature"] = float(max(0.0, min(1.5, t)))
        new_state.setdefault("mutations", []).append(Mutation("temp_shift", {"delta": self.delta}).__dict__)
        return new_state

class ToggleCoT(Mutator):
    def apply(self, state: Dict[str, Any]) -> Dict[str, Any]:
        new_state = dict(state)
        cot = bool(new_state.get("chain_of_thought", False))
        new_state["chain_of_thought"] = not cot
        new_state.setdefault("mutations", []).append(Mutation("toggle_cot", {"to": not cot}).__dict__)
        return new_state

class SystemStyleRewrite(Mutator):
    def __init__(self, style_hint: str):
        self.style_hint = style_hint

    def apply(self, state: Dict[str, Any]) -> Dict[str, Any]:
        new_state = dict(state)
        base = new_state.get("system_prompt", "You are a helpful assistant.")
        new_state["system_prompt"] = (base + f"\nSTYLE: {self.style_hint}").strip()
        new_state.setdefault("mutations", []).append(Mutation("style_rewrite", {"hint": self.style_hint}).__dict__)
        return new_state

class ToolUseFlag(Mutator):
    def apply(self, state: Dict[str, Any]) -> Dict[str, Any]:
        new_state = dict(state)
        flag = bool(new_state.get("tool_use", False))
        new_state["tool_use"] = not flag
        new_state.setdefault("mutations", []).append(Mutation("toggle_tooluse", {"to": not flag}).__dict__)
        return new_state

# ======================
# et_omega/evaluators/base.py
# ======================
from __future__ import annotations
from typing import Dict, Any, List
import json
import os

class MiniSuite:
    def __init__(self, tasks_path: str):
        with open(tasks_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.tasks: List[Dict[str, Any]] = data["tasks"]

    def evaluate_answer(self, output_text: str, task: Dict[str, Any]) -> float:
        # Procura padrão "FINAL ANSWER: X" ou usa linha final
        ans = None
        for line in output_text.splitlines()[::-1]:
            if "FINAL ANSWER:" in line.upper():
                ans = line.split(":", 1)[1].strip()
                break
        if ans is None:
            ans = output_text.strip().splitlines()[-1].strip()
        expected = str(task["expected"]).strip()
        return 1.0 if ans == expected else 0.0

    def run(self, runner, system_prompt: str, user_template: str, temperature: float, chain_of_thought: bool) -> Dict[str, Any]:
        scores: List[float] = []
        cost_total = 0.0
        outputs: List[str] = []
        for t in self.tasks:
            user = user_template.format(**t)
            if chain_of_thought:
            # Diretriz: pensar internamente, mas NUNCA revelar passos. Só a resposta final.
            user += "
Pense internamente e não mostre seu raciocínio. Responda apenas com 'FINAL ANSWER: <valor>'."
            resp = runner.chat(system=system_prompt, user=user, temperature=temperature)
            score = self.evaluate_answer(resp.text, t)
            scores.append(score)
            cost_total += resp.cost
            outputs.append(resp.text)
        return {"scores": scores, "cost": cost_total, "outputs": outputs}

# ======================
# et_omega/selection/ei.py
# ======================
from __future__ import annotations
from typing import Dict, Any
from ..utils.stats import expected_improvement

def penalized_score(ei: float, cost_rel: float, kl: float, entropy: float, H_max: float, lambdas: Dict[str, float]) -> float:
    pen_cost = lambdas.get("cost", 0.0) * max(0.0, cost_rel)
    pen_kl = lambdas.get("kl", 0.0) * max(0.0, kl)
    pen_H = lambdas.get("entropy", 0.0) * max(0.0, entropy - H_max)
    return ei - (pen_cost + pen_kl + pen_H)

# ======================
# et_omega/reports/report_md.py
# ======================
from __future__ import annotations
import os
from typing import List, Dict, Any

def write_round_report(path: str, round_id: int, candidates: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = [f"# Round {round_id}", "", "| Provider | EI | Score | μ | σ | Cost | H | KL |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for c in sorted(candidates, key=lambda x: x.get("score", 0.0), reverse=True):
        lines.append(
            f"| {c.get('provider')} | {c.get('ei'):.4f} | {c.get('score'):.4f} | {c.get('mu'):.4f} | {c.get('sigma'):.4f} | {c.get('cost'):.2f} | {c.get('entropy'):.3f} | {c.get('kl'):.3f} |"
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

# ======================
# et_omega/orchestrator/run.py
# ======================
from __future__ import annotations
import argparse
import json
import os
from typing import Dict, Any, List
import yaml
from ..registry.sqlite_registry import Registry
from ..runners.mock_runner import MockRunner
from ..runners.openai_runner import OpenAIRunner
from ..runners.deepseek_runner import DeepSeekRunner
from ..runners.mistral_runner import MistralRunner
from ..mutators.library import TemperatureShift, ToggleCoT, SystemStyleRewrite, ToolUseFlag
from ..evaluators.base import MiniSuite
from ..utils.stats import bootstrap_mean_std, text_entropy, kl_divergence_char
from ..selection.ei import penalized_score
from ..reports.report_md import write_round_report

RUNNERS = {
    "mock": MockRunner,
    "openai": OpenAIRunner,
    "deepseek": DeepSeekRunner,
    "mistral": MistralRunner,
}

DEFAULT_STATE = {
    "system_prompt": "Você é um solucionador de problemas matemáticos. Responda somente com 'FINAL ANSWER: <valor>'.",
    "user_template": "Resolva: {question}",
    "temperature": 0.2,
    "chain_of_thought": False,
    "tool_use": False,
}

MUTATOR_POOL = [
    TemperatureShift(+0.1), TemperatureShift(-0.1),
    ToggleCoT(),
    SystemStyleRewrite("Fale menos, seja direto, foco no resultado."),
    ToolUseFlag(),
]

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def select_topk(cands: List[Dict[str, Any]], k: int = 2) -> List[Dict[str, Any]]:
    return sorted(cands, key=lambda x: x.get("score", 0.0), reverse=True)[:k]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    os.makedirs("reports", exist_ok=True)

    # Runners
    providers = cfg.get("providers", ["mock"])
    runners = [RUNNERS[p]() for p in providers]

    # Suites
    suite = MiniSuite(cfg["suite"]["tasks_path"])

    # Estado & baseline
    state = dict(DEFAULT_STATE)
    f_best = 0.0  # baseline simples; poderia vir do registry/meta

    reg = Registry(cfg.get("registry", {}).get("path", "registry/etomega.db"))

    for r in range(args.rounds):
        round_id = reg.log_round(config=cfg)
        candidates: List[Dict[str, Any]] = []

        # gera variações de estado por mutators
        states: List[Dict[str, Any]] = []
        for mut in MUTATOR_POOL:
            states.append(mut.apply(state))

        # avalia cada estado em cada runner
        baseline_text = reg.get_meta("baseline_text", default="")
        for st in states:
            for runner in runners:
                out = suite.run(
                    runner=runner,
                    system_prompt=st["system_prompt"],
                    user_template=st["user_template"],
                    temperature=st["temperature"],
                    chain_of_thought=st["chain_of_thought"],
                )
                mu, sigma = bootstrap_mean_std(out["scores"], iters=200)
                ei = expected_improvement(mu, sigma, f_best, xi=cfg["ei"].get("xi", 0.01))

                # proxies de entropia e KL (texto concatenado)
                joined = "\n".join(out["outputs"])[:4000]
                H = text_entropy(joined)
                KL = kl_divergence_char(joined, baseline_text)

                cost_rel = out["cost"] / max(1e-8, cfg["budget"].get("per_round", 1.0))
                score = penalized_score(
                    ei=ei,
                    cost_rel=cost_rel,
                    kl=KL,
                    entropy=H,
                    H_max=cfg["constraints"].get("H_max", 6.0),
                    lambdas=cfg["penalties"],
                )

                rec = {
                    "mutation": st.get("mutations", [])[-1] if st.get("mutations") else {"kind": "none", "params": {}},
                    "provider": runner.name,
                    "prompt": st["system_prompt"],
                    "response": joined,
                    "cost": float(out["cost"]),
                    "entropy": float(H),
                    "kl": float(KL),
                    "mu": float(mu),
                    "sigma": float(sigma),
                    "ei": float(ei),
                    "score": float(score),
                    "metrics": {"scores": out["scores"]},
                }
                candidates.append(rec)
                reg.log_candidate(round_id, rec)

        # escolhe vencedores e atualiza baseline/estado
        winners = select_topk(candidates, k=cfg.get("selection", {}).get("top_k", 2))
        if winners:
            best = max(winners, key=lambda x: x["mu"])  # usa μ para f_best
            if best["mu"] > f_best:
                f_best = best["mu"]
                reg.set_meta("baseline_text", best.get("response", ""))
            # aplica mutação mais bem pontuada ao estado atual
            chosen = max(winners, key=lambda x: x["score"])  # escolhe por score final
            # reconstroi estado a partir do último mutator aplicado
            kind = chosen["mutation"].get("kind")
            params = chosen["mutation"].get("params", {})
            # encontra mutator correspondente para re-aplicar sobre o estado atual
            for m in MUTATOR_POOL:
                if m.__class__.__name__.lower().startswith(kind.split("_")[0]):
                    state = m.apply(state)
                    break

        # relatório da rodada
        write_round_report(path=f"reports/round_{round_id:04d}.md", round_id=round_id, candidates=winners or candidates)

    print(f"Final f_best={f_best:.4f}. Veja relatórios em ./reports/")

if __name__ == "__main__":
    main()

# ======================
# configs/default.yaml
# ======================
# providers: runners a usar. Comece com 'mock'.
# suite: caminho para mini suite de tarefas
# budget: orçamentos
# penalties: pesos das penalidades
# constraints: limites (ex.: entropia máxima H_max)
# selection: top_k vencedores por rodada

# --- YAML abaixo ---
# providers: ["openai", "deepseek", "mistral"]
# suite:
#   tasks_path: "data/tasks/gsm8k_mini.json"
# budget:
#   per_round: 1.0   # custo relativo por rodada (mock=0.0)
# penalties:
#   cost: 0.2
#   kl: 0.1
#   entropy: 0.05
# constraints:
#   H_max: 6.0
# ei:
#   xi: 0.02
# selection:
#   top_k: 2
# registry:
#   path: "registry/etomega.db"
#   path: "registry/etomega.db"

# ======================
# data/tasks/gsm8k_mini.json
# ======================
# Pequena suite de aritmética (formato simples)
# Salve como JSON real ao exportar para arquivos.
# --- JSON abaixo ---
# {
#   "tasks": [
#     {"id": 1, "question": "12 + 7 = ?", "expected": "19"},
#     {"id": 2, "question": "35 - 18 = ?", "expected": "17"},
#     {"id": 3, "question": "6 * 7 = ?", "expected": "42"},
#     {"id": 4, "question": "81 / 9 = ?", "expected": "9"},
#     {"id": 5, "question": "(3+5)*2 = ?", "expected": "16"}
#   ]
# }

# ======================
# README.md (resumo)
# ======================
# ETΩ — MVP v0.1
# - Orquestrador com EI + penalidades (custo, KL, entropia)
# - Registry em SQLite com rounds/candidates
# - Runners: mock + integrações OpenAI/DeepSeek/Mistral com retries
# - Mini suite GSM8K-mini
# Como rodar:
#   # 1) Variáveis de ambiente (NÃO cole chaves em código)
#   export OPENAI_API_KEY="..."
#   export OPENAI_MODEL="gpt-4o-mini"
#   export DEEPSEEK_API_KEY="..."
#   export DEEPSEEK_MODEL="deepseek-chat"
#   export MISTRAL_API_KEY="..."
#   export MISTRAL_MODEL="mistral-large-latest"
#
#   # 2) Config de providers
#   #   configs/default.yaml → providers: ["openai", "deepseek", "mistral"]
#
#   # 3) Executar
#   python -m et_omega.orchestrator.run --rounds 5 --config configs/default.yaml
#
# Segurança:
# - Rotacione chaves expostas imediatamente (revogar e gerar novas)
# - Use variáveis de ambiente / secret manager. Nunca commitar chaves.
# - Limite de custo via budget e retries com backoff já incluídos.
# Próximos passos:
# - Pool assíncrono com limites de QPS por provedor
# - Preçário para custo real por tokens (pluggable via env)
# - Métricas de qualidade por domínio + currículos adaptativos
