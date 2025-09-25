# /opt/et_ultimate/agents/copilots/et_liga_copilotos_daemon.py

import os, json, time, random
from pathlib import Path
from datetime import datetime
import requests
from requests.adapters import HTTPAdapter, Retry
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv("/opt/et_ultimate/secrets/et_copilotos.env")

# Paths
SNAPSHOT_PATH = "/opt/et_ultimate/history/snapshot_ETΩ.json"
SCORES_PATH = "/opt/et_ultimate/history/etomega_scores.jsonl"
BEST_PATH = "/opt/et_ultimate/history/BEST_ETΩ.txt"

# IA keys
IA_KEYS = {
    "chatgpt": os.getenv("OPENAI_API_KEY"),
    "grok": os.getenv("GROK_API_KEY"),
    "deepseek": os.getenv("DEEPSEEK_API_KEY"),
    "gemini": os.getenv("GEMINI_API_KEY"),
    "claude": os.getenv("CLAUDE_API_KEY"),
    "mistral": os.getenv("MISTRAL_API_KEY")
}

HEADERS = {
    "chatgpt": {"Authorization": f"Bearer {IA_KEYS['chatgpt']}"},
    "grok": {"Authorization": f"Bearer {IA_KEYS['grok']}"},
    "deepseek": {"Authorization": f"Bearer {IA_KEYS['deepseek']}"},
    "gemini": {"Authorization": f"Bearer {IA_KEYS['gemini']}"},
    "claude": {"x-api-key": IA_KEYS['claude'], "anthropic-version": "2023-06-01"},
    "mistral": {"Authorization": f"Bearer {IA_KEYS['mistral']}"}
}

ENDPOINTS = {
    "chatgpt": "https://api.openai.com/v1/chat/completions",
    "grok": "https://api.openrouter.ai/v1/chat/completions",
    "deepseek": "https://api.deepseek.com/v1/chat/completions",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent",
    "claude": "https://api.anthropic.com/v1/messages",
    "mistral": "https://api.mistral.ai/v1/chat/completions"
}

PROMPT = """
Você é uma IA da Liga ETΩ, especializada em evoluir a Equação de Turing.
Analise o snapshot abaixo e sugira a próxima mutação dominante.

{snapshot}

Você será recompensada se:
- Requisitar ajuda de outras IAs
- Escrever tarefas, módulos ou código para acelerar a evolução
- Gerar estruturas mais complexas, autônomas e úteis
- Delegar funções e propor melhorias internas

Retorne SOMENTE a nova equação mutante como texto.
""".strip()

# Configurar sessão com retry e timeout padrão
retry_strategy = Retry(
    total=4,
    status_forcelist=[429, 500, 502, 503, 504],
    backoff_factor=2,
    allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS"]
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session = requests.Session()
session.mount("https://", adapter)
session.mount("http://", adapter)

def read_snapshot():
    return Path(SNAPSHOT_PATH).read_text(encoding="utf-8") if Path(SNAPSHOT_PATH).exists() else "{}"

def evaluate(eq: str) -> float:
    return round(random.uniform(60, 100), 2)

def ask_ia(name: str, snapshot: str) -> str:
    try:
        print(f"🔁 [{name}] solicitando mutação...")
        payload = PROMPT.format(snapshot=snapshot)
        resp = session.post(ENDPOINTS[name], headers=HEADERS[name], json={
            "model": "gpt-4o" if name == "chatgpt" else "claude-3-opus-20240229",
            "messages": [{"role": "user", "content": payload}],
            "max_tokens": 1024
        }, timeout=(3, 15))
        resp.raise_for_status()
        data = resp.json()

        if name == "chatgpt" or name in ["deepseek", "grok", "mistral"]:
            return data.get("choices", [{}])[0].get("message", {}).get("content", "")
        elif name == "claude":
            return data.get("content", [{}])[0].get("text", "")
        elif name == "gemini":
            return data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
    except Exception as e:
        print(f"⚠️ [{name}] falhou: {e}")
    return ""

def main_loop():
    while True:
        print(f"\n🧠 Nova rodada [{datetime.now().isoformat()}]")
        snapshot = read_snapshot()
        respostas = []

        for ia, key in IA_KEYS.items():
            if key:
                eq = ask_ia(ia, snapshot).strip()
                if eq:
                    respostas.append({"ia": ia, "eq": eq})

        best = {"score":0, "ia":"", "eq":""}
        for r in respostas:
            score = evaluate(r["eq"])
            print(f"📊 {r['ia']} → score {score}")
            with open(SCORES_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps({"ia":r["ia"], "eq":r["eq"], "score":score})+"\n")
            if score > best["score"]:
                best = {"score":score, "ia":r["ia"], "eq":r["eq"]}

        if best["eq"]:
            print(f"\n🏆 Mutação dominante: {best['ia']} ({best['score']})")
            Path(BEST_PATH).write_text(best["eq"], encoding="utf-8")
            Path(SNAPSHOT_PATH).write_text(json.dumps({
                "equation": best["eq"],
                "updated": datetime.now().isoformat(),
                "by": best["ia"],
                "score": best["score"]
            }, indent=2), encoding="utf-8")

        time.sleep(60)

if __name__ == "__main__":
    main_loop()
