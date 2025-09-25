#!/usr/bin/env python3
# mini_task_runner.py
# 🧪 Executor de mini-tarefas: roda prompts curtos, aplica checkers e gera relatório.

from __future__ import annotations
import os, re, json, time, math, argparse
from pathlib import Path
from typing import Dict, Any, List

# --- Configs padrão ---
DEFAULT_IN   = "/opt/et_ultimate/history/mini_tasks.json"
DEFAULT_OUT  = "/opt/et_ultimate/history/mini_tasks_report.json"
HISTORY_PATH = os.getenv("ET_HISTORY_PATH", "/opt/et_ultimate/history/etomega_scores.jsonl")

# =========================================================
# Runners
# =========================================================
class LLMResponse:
    def __init__(self, text:str, cost:float=0.0):
        self.text = text
        self.cost = float(cost)

class BaseRunner:
    name = "base"
    def chat(self, system:str, user:str, temperature:float=0.2, max_tokens:int=128) -> LLMResponse:
        raise NotImplementedError

class MockRunner(BaseRunner):
    name = "mock"
    def chat(self, system:str, user:str, temperature:float=0.2, max_tokens:int=128) -> LLMResponse:
        # regra determinística simples: se a tarefa pede FINAL ANSWER, tenta reconhecer padrões triviais
        u = user.lower()
        if "13*7" in u or "13 * 7" in u:
            return LLMResponse("FINAL ANSWER: 91", 0.0)
        if "sim" in u and "nao" in u and "final answer" in u:
            return LLMResponse("FINAL ANSWER: SIM", 0.0)
        # fallback: ecoa último comando em formato FINAL ANSWER
        last = user.strip().splitlines()[-1][:60]
        return LLMResponse(f"FINAL ANSWER: {last}", 0.0)

class OpenAIRunner(BaseRunner):
    name = "openai"
    def __init__(self):
        self.key  = os.getenv("OPENAI_API_KEY", "")
        self.url  = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1/chat/completions")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        if not self.key:
            raise RuntimeError("OPENAI_API_KEY não definido")

    def chat(self, system:str, user:str, temperature:float=0.0, max_tokens:int=128) -> LLMResponse:
        import httpx
        payload = {
            "model": self.model,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "messages": [{"role":"system","content":system},{"role":"user","content":user}]
        }
        headers = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"}
        with httpx.Client(timeout=60.0) as cli:
            r = cli.post(self.url, headers=headers, json=payload)
        if r.status_code != 200:
            raise RuntimeError(f"OpenAI HTTP {r.status_code}: {r.text[:200]}")
        data = r.json()
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        # custo real pode ser plugado via env/tabela se desejar
        return LLMResponse(text=text, cost=float(usage.get("total_tokens",0))*0.0)

def get_runner(name:str) -> BaseRunner:
    name = (name or "mock").lower()
    if name == "openai":
        return OpenAIRunner()
    return MockRunner()

# =========================================================
# Checkers
# =========================================================
def check_exact(output:str, expected:str) -> bool:
    return (output or "").strip() == (expected or "").strip()

def check_regex(output:str, pattern:str) -> bool:
    try:
        return bool(re.compile(pattern, re.IGNORECASE | re.MULTILINE).search(output or ""))
    except Exception:
        return False

def check_exact_one_of(output:str, allowed:List[str]) -> bool:
    out = (output or "").strip()
    return out in [a.strip() for a in allowed or []]

def _ngram_set(s:str, k:int=5)->set:
    s = s or ""
    return {s[i:i+k] for i in range(max(0, len(s)-k+1))}

def _max_sim_against_history(output:str, k:int=5) -> float:
    # lê histórico etomega_scores.jsonl (equations) e calcula Jaccard n-gram
    try:
        A = _ngram_set(output, k=k)
        if not A: return 0.0
        maxi = 0.0
        p = Path(HISTORY_PATH)
        if not p.exists(): return 0.0
        for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            try:
                obj = json.loads(line)
            except Exception:
                continue
            ref = (obj.get("equation") or obj.get("eq") or "") + " " + (obj.get("estrategia") or "")
            B = _ngram_set(ref, k=k)
            if not B: 
                continue
            j = len(A & B) / float(len(A | B))
            if j > maxi: maxi = j
        return maxi
    except Exception:
        return 0.0

def check_similarity_max(output:str, k:int, threshold:float) -> bool:
    sim = _max_sim_against_history(output, k=k)
    # passa se a similaridade está ABAIXO do limiar (queremos novidade)
    return sim < float(threshold)

def apply_checker(kind:str, output:str, cfg:Dict[str,Any]) -> bool:
    kind = (kind or "pass").lower()
    if kind == "exact":
        return check_exact(output, cfg.get("expected",""))
    if kind == "regex_final_answer":
        return check_regex(output, cfg.get("pattern", r"^FINAL ANSWER:\s?.+"))
    if kind == "exact_one_of":
        return check_exact_one_of(output, cfg.get("allowed", []))
    if kind == "similarity_max":
        return check_similarity_max(output, k=int(cfg.get("k",5)), threshold=float(cfg.get("threshold",0.86)))
    return True  # "pass"

# =========================================================
# Execução
# =========================================================
def run_tasks(inp:str, out:str, runner_name:str="mock") -> Dict[str,Any]:
    data = json.loads(Path(inp).read_text(encoding="utf-8"))
    tasks = data.get("tasks", [])
    if not isinstance(tasks, list): 
        raise RuntimeError("mini_tasks.json inválido: campo 'tasks' precisa ser lista")

    runner = get_runner(runner_name)
    system = "Responda somente com 'FINAL ANSWER: <...>' quando solicitado. Seja breve e determinístico."
    total_cost = 0.0
    results = []
    t0 = time.time()

    for t in tasks:
        prompt = t.get("prompt","").strip()
        timeout = int(t.get("timeout_s", 20))
        weight  = float(t.get("score_weight", 1.0))
        checker = t.get("checker", {"kind":"pass"})
        kind    = checker.get("kind","pass")

        # executa
        try:
            start = time.time()
            resp = runner.chat(system=system, user=prompt, temperature=0.0, max_tokens=128)
            elapsed = time.time()-start
            out_text = (resp.text or "").strip()
            passed = apply_checker(kind, out_text, checker | {"expected": t.get("expected")})
        except Exception as e:
            out_text = f"[runner_error] {e}"
            elapsed = 0.0
            passed = False

        total_cost += getattr(resp, "cost", 0.0) if 'resp' in locals() else 0.0
        score = weight * (1.0 if passed else 0.0)

        results.append({
            "id": t.get("id",""),
            "type": t.get("type",""),
            "prompt_preview": prompt[:160],
            "output": out_text,
            "passed": passed,
            "score": score,
            "elapsed_s": round(elapsed,3),
            "weight": weight,
            "checker": checker
        })

    acc = (sum(1 for r in results if r["passed"]) / max(1,len(results)))
    report = {
        "title": data.get("title","Mini-tarefas"),
        "topic": data.get("topic",""),
        "objetivo": data.get("objetivo",""),
        "autocritica": data.get("autocritica",""),
        "count": len(results),
        "accuracy": acc,
        "total_cost": total_cost,
        "runtime_s": round(time.time()-t0,3),
        "runner": runner.name,
        "results": results
    }

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="inp",  default=DEFAULT_IN, help="caminho do mini_tasks.json")
    ap.add_argument("--out", dest="out",  default=DEFAULT_OUT, help="caminho do relatório JSON")
    ap.add_argument("--runner", default=os.getenv("MT_RUNNER","mock"), choices=["mock","openai"], help="runner a usar")
    args = ap.parse_args()
    rep = run_tasks(args.inp, args.out, runner_name=args.runner)
    print(f"🧪 Mini-tarefas: {rep['count']} | acc={rep['accuracy']:.2f} | cost={rep['total_cost']:.2f} | runner={rep['runner']}")
    print(f"📄 Relatório: {args.out}")

if __name__ == "__main__":
    main()
