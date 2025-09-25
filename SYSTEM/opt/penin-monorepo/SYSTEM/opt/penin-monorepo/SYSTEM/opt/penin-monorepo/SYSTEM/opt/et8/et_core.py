import os, time, json, pathlib, http.server, socketserver, threading
BASE = "/workspace"
OUT  = "/changes_out"
HOST_RO = "/host-ro"  # montagem read-only do host
pathlib.Path(OUT).mkdir(parents=True, exist_ok=True)

def write_changeset(title:str, commands:list[str]):
    ts = time.strftime("%Y%m%d-%H%M%S")
    safe = "".join(c for c in title if c.isalnum() or c in "-_")[:64] or "change"
    fn = f"{OUT}/{ts}-{safe}.sh"
    with open(fn,"w") as f:
        f.write("#!/usr/bin/env bash\nset -euo pipefail\n")
        for c in commands: f.write(c + "\n")
    os.chmod(fn, 0o755)
    return fn

# HTTP para debug/health e pedido de amostra (porta 7008)
class H(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200); self.end_headers(); self.wfile.write(b"ok")
        elif self.path == "/sample-proposal":
            fn = write_changeset("exemplo-ajuste-nginx",
               ["echo \"# exemplo: alteracao revisavel\"",
                "sed -n 1,5p /etc/nginx/nginx.conf # (sera bloqueado na aplicacao se tentar escrever)"])
            self.send_response(200); self.end_headers(); self.wfile.write(fn.encode())
        else:
            self.send_response(404); self.end_headers()

def serve():
    with socketserver.TCPServer(("0.0.0.0",7008), H) as httpd:
        httpd.serve_forever()

if __name__ == "__main__":
    threading.Thread(target=serve, daemon=True).start()
    # Loop do agente (placeholder): varre /host-ro (somente leitura)
    while True:
        # Exemplo de leitura segura do host:
        for p in ["/host-ro/etc/os-release","/host-ro/proc/cpuinfo"]:
            try:
                with open(p, "rb") as f: _ = f.read(512)
            except Exception: pass
        time.sleep(5)
