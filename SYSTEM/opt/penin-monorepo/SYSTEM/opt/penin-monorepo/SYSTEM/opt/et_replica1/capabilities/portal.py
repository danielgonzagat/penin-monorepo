#!/opt/et_replica1/venv/bin/python3
import json, os, subprocess, time, shutil, hashlib, traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import yaml

ROOT = Path("/opt/et_replica1")
POLICY = yaml.safe_load(open(ROOT/"policy.yaml"))
LOGDIR = Path("/var/log/et"); LOGDIR.mkdir(parents=True, exist_ok=True)

def ok(data): return 200, {"ok": True, "data": data}
def err(msg): return 400, {"ok": False, "error": msg}

def backup(p: Path) -> Path:
    ts = time.strftime("%Y%m%d-%H%M%S")
    b = p.with_suffix(p.suffix+f".bak.{ts}")
    if p.exists(): shutil.copy2(p, b)
    return b

def run_shell(cmd: str, timeout: int):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return ok({"code": r.returncode, "out": r.stdout[-8000:], "err": r.stderr[-8000:]})
    except subprocess.TimeoutExpired:
        return err("timeout shell")

def fs_patch(target: str, before_sha256: str, after: str):
    if not POLICY.get("unsafe_full_write"): return err("full write desabilitado na policy")
    p = Path(target)
    cur = p.read_text(encoding="utf-8", errors="ignore") if p.exists() else ""
    if cur and hashlib.sha256(cur.encode()).hexdigest() != before_sha256:
        return err("hash não confere (race)")
    backup(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(after, encoding="utf-8")
    return ok({"sha256": hashlib.sha256(after.encode()).hexdigest()})

class H(BaseHTTPRequestHandler):
    def _json(self, code, body):
        self.send_response(code)
        self.send_header("Content-Type","application/json")
        self.end_headers()
        self.wfile.write(json.dumps(body, ensure_ascii=False).encode("utf-8"))

    def do_POST(self):
        try:
            length = int(self.headers.get('Content-Length','0'))
            raw = self.rfile.read(length) if length else b"{}"
            data = json.loads(raw or b"{}")
            if self.path == "/v1/health.check":
                self._json(*ok({"status":200,"ts":time.time()})); return
            if self.path == "/v1/shell.run":
                timeout = int(POLICY.get("budgets",{}).get("action_timeouts",{}).get("shell_seconds",120))
                self._json(*run_shell(data.get("cmd",""), timeout)); return
            if self.path == "/v1/fs.patch":
                self._json(*fs_patch(data["target"], data["before_sha256"], data["after"])); return
            if self.path == "/v1/systemd":
                svc_cmd = data.get("cmd","")
                if svc_cmd.startswith(("start ","stop ","restart ","status ")):
                    self._json(*run_shell("systemctl "+svc_cmd, 30)); return
                self._json(*err("cmd inválido")); return
            self._json(*err("rota desconhecida"))
        except Exception as e:
            self._json(500, {"ok":False,"error":str(e),"trace":traceback.format_exc()})

if __name__ == "__main__":
    HTTPServer(("127.0.0.1", 9876), H).serve_forever()
