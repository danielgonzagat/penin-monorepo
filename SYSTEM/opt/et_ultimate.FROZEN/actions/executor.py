#!/usr/bin/env python3
import json, os, sys, time, subprocess, shlex, pathlib, signal
QUEUE = "/opt/et_ultimate/actions/queue.jsonl"
LOG   = "/opt/et_ultimate/logs/executor.log"
ALLOW_CMDS = {
  "pip_install": "pip install {packages}",
  "git_clone":   "git clone {repo} {dest}",
  "run_py":      "python -c {code}",
  "bash":        "bash -lc {script}"
}
DOCKER_IMAGE = "et-sandbox:latest"
MOUNTS = [
  "-v", "/opt/et_ultimate/sandbox:/workspace:rw",
  "-v", "/opt/et_ultimate/logs:/logs:rw"
]
LIMITS = [
  "--network=none",          # sem rede por padrão (abrimos exceção depois, se quiser)
  "--cpus=2", "--memory=4g", # limites de recursos
  "--pids-limit=512",
  "--read-only"              # rootfs só leitura
]
def log(msg): open(LOG,"a").write(f"{time.strftime('%F %T')} | {msg}\n")
def run_in_container(cmd):
    base = ["docker","run","--rm"]+LIMITS+MOUNTS, 
    # como rootfs é read-only, usamos /tmp e /workspace para escrita
    docker_cmd = ["docker","run","--rm"] + LIMITS + MOUNTS + \
                 ["-w","/workspace","-u","runner", DOCKER_IMAGE, "bash","-lc", cmd]
    return subprocess.run(docker_cmd, capture_output=True, text=True, timeout=600)

def handle(action):
    t = action.get("type")
    args = action.get("args", {})
    if t not in ALLOW_CMDS:
        return False, f"acao_nao_permitida:{t}"
    # renderiza comando
    try:
        cmd = ALLOW_CMDS[t].format(**args)
    except KeyError as e:
        return False, f"arg_faltando:{e}"
    r = run_in_container(cmd)
    ok = (r.returncode == 0)
    out = (r.stdout or "") + (r.stderr or "")
    return ok, out[:10000]  # limita log

def main():
    pathlib.Path(QUEUE).touch(exist_ok=True)
    log("executor_start")
    with open(QUEUE,"r") as f:
        # processa entradas já existentes uma vez
        pass
    # acompanha o arquivo em tempo real
    with open(QUEUE,"r") as f:
        f.seek(0, os.SEEK_END)
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.5); continue
            try:
                action = json.loads(line)
            except Exception as e:
                log(f"json_invalido:{e} line={line[:200]}")
                continue
            ok, out = handle(action)
            log(f"done type={action.get('type')} ok={ok} out={out.replace(chr(10),' | ')[:500]}")
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("executor_stop")
