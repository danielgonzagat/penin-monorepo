#!/opt/et_ultimate/autonomy/venv/bin/python3
import os, sys, json, time, subprocess, shlex, pathlib, traceback, fcntl
from pathlib import Path
import requests

ET_URL    = os.environ.get("ET_URL", "http://127.0.0.1:8080/v1").rstrip("/")
ET_KEY    = os.environ.get("ET_API_KEY", "DANIEL")
ET_MODEL  = os.environ.get("ET_MODEL", "")
OBJECTIVE = os.environ.get("ET_OBJECTIVE", "Evoluir a Equação de Turing até o infinito maximizando Expected Improvement sob os guardrails da ETΩ.")
SAFE_MODE = int(os.environ.get("SAFE_MODE", "1"))    # 0 = 100% livre
AUTO_PLAN = int(os.environ.get("AUTO_PLAN", "1"))    # 1 = o modelo planeja sozinho

ROOT      = Path("/opt/et_ultimate")
ACTIONS   = ROOT/"actions"
QUEUE     = ACTIONS/"queue.jsonl"
RESULTS   = ACTIONS/"results.jsonl"
STATE     = ACTIONS/"offset.txt"
WS        = ROOT/"workspace"
LOGF      = ROOT/"logs/et_autonomy.log"
STOPF     = ROOT/"autonomy/STOP"

for p in (ACTIONS, WS, LOGF.parent):
    p.mkdir(parents=True, exist_ok=True)

def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts} | {msg}\n"
    sys.stdout.write(line); sys.stdout.flush()
    with open(LOGF, "a", encoding="utf-8") as f:
        f.write(line)

def backend_model():
    if ET_MODEL:
        return ET_MODEL
    r = requests.get(f"{ET_URL}/models", timeout=20)
    r.raise_for_status()
    return r.json()["data"][0]["id"]

def call_llm(prompt, temperature=0.2, max_tokens=800):
    headers = {"Authorization": f"Bearer {ET_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": backend_model(),
        "messages": [
            {"role":"system","content": "Você é uma IA autônoma executiva. Responda apenas com JSONL de intents, uma por linha. Sem comentários."},
            {"role":"user","content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False
    }
    r = requests.post(f"{ET_URL}/chat/completions", headers=headers, json=payload, timeout=600)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()

def safe_path(p: str) -> Path:
    pp = (WS/Path(p)).resolve()
    if SAFE_MODE and not str(pp).startswith(str(WS.resolve())):
        raise RuntimeError(f"path fora do workspace: {pp}")
    return pp

def run(cmd, cwd=None, shell=False, env=None, timeout=1800):
    if isinstance(cmd, str) and not shell:
        cmd = shlex.split(cmd)
    log(f"$ {cmd if isinstance(cmd,str) else ' '.join(cmd if isinstance(cmd,list) else [str(cmd)])}")
    res = subprocess.run(cmd, cwd=cwd, shell=shell, env=env, capture_output=True, text=True, timeout=timeout)
    return {"returncode": res.returncode, "stdout": res.stdout[-8000:], "stderr": res.stderr[-8000:]}

# executores
def ex_git_clone(args):
    a = args or {}
    repo = a.get('repo') or a.get('url')
    if not repo: raise KeyError('args.repo/url ausente')
    dest = a.get('dest') or a.get('destination') or a.get('path') or Path(repo).stem.replace('.git','')
    return run(['git','clone',repo,dest])

def ex_pip_install(args):
    a = args or {}
    pk = a.get('packages')
    if isinstance(pk, str):
        import shlex
        pk = shlex.split(pk)
    if not pk: raise KeyError('args.packages ausente')
    return run(['pip','install'] + list(pk))

def ex_run_py(args):
    a = args or {}
    if a.get('code') or a.get('py') or a.get('source'):
        code = a.get('code') or a.get('py') or a.get('source')
        return run([sys.executable, '-c', code])
    if a.get('file'):
        argv = [sys.executable, a['file']] + list(a.get('args', []))
        return run(argv)
    raise KeyError('args.code/py/source ou args.file ausente')

def ex_bash(args):
    script = (args or {}).get('script') or (args or {}).get('cmd') or (args or {}).get('shell')
    if not script: raise KeyError('args.script/cmd/shell ausente')
    return run(script, shell=True, cwd=str(WS))

def ex_write_file(args):
    path = safe_path(args["path"])
    content = args.get("content","")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return {"returncode": 0, "stdout": f"wrote {path}", "stderr": ""}

def ex_append_file(args):
    path = safe_path(args["path"])
    content = args.get("content","")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f: f.write(content)
    return {"returncode": 0, "stdout": f"appended {path}", "stderr": ""}

EXECUTORS = {
    "git_clone": ex_git_clone,
    "pip_install": ex_pip_install,
    "run_py": ex_run_py,
    "bash": ex_bash,
    "write_file": ex_write_file,
    "append_file": ex_append_file,
}

def enqueue(lines_jsonl: str):
    if not lines_jsonl.strip(): return 0
    # filtra somente linhas JSON válidas
    good = []
    for raw in lines_jsonl.splitlines():
        raw = raw.strip()
        if not raw: continue
        try:
            obj = json.loads(raw)
            if not isinstance(obj, dict): continue
            if "type" not in obj or "args" not in obj: continue
            good.append(json.dumps(obj, ensure_ascii=False))
        except Exception:
            continue
    if not good: return 0
    with open(QUEUE, "a", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        for g in good: f.write(g+"\n")
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    return len(good)

def plan_once():
    # estado mínimo para dar contexto
    try:
        tree = subprocess.run(["bash","-lc","(cd /opt/et_ultimate/workspace && ls -laR | head -n 400) || true"], capture_output=True, text=True, timeout=30).stdout
    except Exception:
        tree = ""
    prompt = f"""
Objetivo ÚNICO e imutável: {OBJECTIVE}

Regras:
- Gere apenas INTENTS em JSONL, uma por linha, com chaves: "type" e "args".
- Tipos suportados: git_clone, pip_install, run_py, bash, write_file, append_file.
- Para write/append use caminhos relativos ao workspace (./…).
- Sem texto explicativo. Apenas JSON por linha.

Contexto (workspace):
{tree}

Proponha o próximo lote de 3–6 intents que maximizam Expected Improvement agora.
"""
    out = call_llm(prompt)
    n = enqueue(out)
    log(f"planner: enfileiradas {n} intents")
    return n

def read_new_lines():
    STATE.parent.mkdir(parents=True, exist_ok=True)
    off = 0
    if STATE.exists():
        try: off = int(STATE.read_text().strip() or "0")
        except: off = 0
    with open(QUEUE, "a+", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        f.seek(0,0)
        data = f.read()
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    if off >= len(data): return [], len(data)
    chunk = data[off:]
    lines = [x for x in chunk.splitlines() if x.strip()]
    return lines, len(data)

def record_result(item, res):
    item = dict(item)
    item["result"] = res
    item["ts"] = int(time.time())
    with open(RESULTS, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False)+"\n")

def loop():
    log("autonomy started")
    backoff = 2
    while True:
        if STOPF.exists():
            log("STOP encontrado — pausando. Remova o arquivo para continuar.")
            time.sleep(3)
            continue
        try:
            lines, new_off = read_new_lines()
            if not lines and AUTO_PLAN:
                try: plan_once()
                except Exception as e:
                    log(f"planner erro: {e}")
            lines, new_off = read_new_lines()  # tenta ler de novo após planejar
            if not lines:
                time.sleep(1.0)
                continue
            for raw in lines:
                try:
                    item = json.loads(raw)
                    typ  = item.get("type")
                    args = item.get("args", {})
                    if SAFE_MODE and typ not in EXECUTORS:
                        raise RuntimeError(f"tipo não permitido em SAFE_MODE: {typ}")
                    fn = EXECUTORS.get(typ)
                    if fn is None:
                        raise RuntimeError(f"tipo desconhecido: {typ}")
                    res = fn(args)
                    record_result(item, res)
                    log(f"OK {typ}")
                except Exception as e:
                    record_result({"raw":raw}, {"returncode": 1, "stdout":"", "stderr": f"{e}\n{traceback.format_exc()[-1000:]}"} )
                    log(f"ERRO executando intent: {e}")
            STATE.write_text(str(new_off))
            backoff = 2
        except Exception as e:
            log(f"loop erro: {e}")
            time.sleep(backoff)
            backoff = min(backoff*2, 30)

if __name__ == "__main__":
    loop()
