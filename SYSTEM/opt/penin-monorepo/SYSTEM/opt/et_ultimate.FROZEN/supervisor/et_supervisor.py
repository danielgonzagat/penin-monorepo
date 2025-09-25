#!/opt/et_ultimate/venv/bin/python3
import json, time, subprocess, threading
from pathlib import Path
import requests

Q = Path("/opt/et_ultimate/queue/actions.jsonl")
LOG = Path("/var/log/et/supervisor.log")
PORTAL = "http://127.0.0.1:9876"

def log(msg):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text((LOG.read_text() if LOG.exists() else "") + time.strftime("[%F %T] ") + msg + "\n")

def post(path, payload):
    r = requests.post(PORTAL+path, json=payload, timeout=180)
    return r.json()

def handle(line):
    obj = json.loads(line)
    tool = obj.get("tool")
    if tool == "shell.run":
        return post("/v1/shell.run", {"cmd": obj.get("cmd","")})
    if tool == "python.eval":
        code = obj.get("code","")
        p = subprocess.run(["/opt/et_ultimate/venv/bin/python3","-c",code],
                           capture_output=True, text=True, timeout=180)
        return {"ok": True, "data": {"code": p.returncode, "out": p.stdout[-8000:], "err": p.stderr[-8000:]}}
    if tool == "systemd":
        return post("/v1/systemd", {"cmd": obj.get("cmd","")})
    return {"ok": False, "error": "tool desconhecida"}

def loop():
    log("supervisor iniciado")
    while True:
        if not Q.exists():
            time.sleep(2); continue
        lines = Q.read_text().strip().splitlines()
        if not lines:
            time.sleep(2); continue
        Q.write_text("")  # consume de maneira simples
        for ln in lines:
            try:
                res = handle(ln)
                log(f"exec: {ln} => {res}")
            except Exception as e:
                log(f"erro: {ln} => {e}")
        time.sleep(1)

if __name__ == "__main__":
    loop()
