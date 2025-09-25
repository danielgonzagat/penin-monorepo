from fastapi import FastAPI
import subprocess, json, time
import requests

app = FastAPI()

def curl_api():
    try:
        r = requests.get("http://127.0.0.1:8080/v1/models", timeout=5)
        return (r.status_code, r.text[:400])
    except Exception as e:
        return (0, str(e))

@app.get("/health")
def health():
    code, body = curl_api()
    return {"ok": code==200 and "models" in body, "status": code}

@app.post("/fix")
def fix():
    # roda o auto-heal e retorna estado
    p = subprocess.run(["/opt/et/venv/bin/python","/opt/et/autoheal.py"], capture_output=True, text=True)
    try:
        data = json.loads(p.stdout.strip() or "{}")
    except Exception:
        data = {"raw": p.stdout}
    return {"healed": data.get("api_healthy", False), "details": data}
