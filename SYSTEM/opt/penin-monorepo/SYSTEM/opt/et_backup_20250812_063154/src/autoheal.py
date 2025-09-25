import os, time, subprocess, json
import requests

TIMEOUT = 5
BACKENDS = [("127.0.0.1",8090), ("127.0.0.1",8091)]
API = ("127.0.0.1",8080)

def ok_models(host,port):
    try:
        r = requests.get(f"http://{host}:{port}/v1/models", timeout=TIMEOUT)
        return r.status_code==200 and "models" in r.text
    except Exception:
        return False

def restart(unit):
    subprocess.run(["systemctl","restart",unit], check=False)

def ensure_nginx_site():
    # só garante que nginx carrega o site; conteúdo já foi configurado antes
    subprocess.run(["nginx","-t"], check=True)
    subprocess.run(["systemctl","reload","nginx"], check=False)

def heal():
    # backends
    any_bad=False
    for i,(h,p) in enumerate(BACKENDS):
        if not ok_models(h,p):
            any_bad=True
            restart(f"llama-s{i}.service")
            time.sleep(2)
    # nginx
    if not ok_models(*API):
        ensure_nginx_site()
        time.sleep(1)
    # revalida
    return ok_models(*API)

if __name__=="__main__":
    ok = heal()
    print(json.dumps({"api_healthy": ok}))
