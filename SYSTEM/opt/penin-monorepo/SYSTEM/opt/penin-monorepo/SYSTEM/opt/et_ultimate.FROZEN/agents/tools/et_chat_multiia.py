import os, json, requests, time
from datetime import datetime
from pathlib import Path

SNAPSHOT_PATH = "/opt/et_ultimate/history/snapshot_ETΩ.json"
HISTORY_PATH = "/opt/et_ultimate/history/etomega_chat.log"

snapshot = Path(SNAPSHOT_PATH).read_text(encoding="utf-8") if Path(SNAPSHOT_PATH).exists() else "{}"

IA_KEYS = {
    "chatgpt": os.getenv("OPENAI_API_KEY"),
    "deepseek": os.getenv("DEEPSEEK_API_KEY"),
    "claude": os.getenv("CLAUDE_API_KEY"),
    "gemini": os.getenv("GEMINI_API_KEY"),
    "mistral": os.getenv("MISTRAL_API_KEY"),
}

ENDPOINTS = {
    "chatgpt": "https://api.openai.com/v1/chat/completions",
    "deepseek": "https://api.deepseek.com/v1/chat/completions",
    "claude": "https://api.anthropic.com/v1/messages",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent",
    "mistral": "https://api.mistral.ai/v1/chat/completions",
}

HEADERS = {
    "chatgpt": {"Authorization": f"Bearer {IA_KEYS['chatgpt']}", "Content-Type": "application/json"},
    "deepseek": {"Authorization": f"Bearer {IA_KEYS['deepseek']}", "Content-Type": "application/json"},
    "claude": {
        "x-api-key": IA_KEYS['claude'],
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01"
    },
    "gemini": {"Content-Type": "application/json"},
    "mistral": {"Authorization": f"Bearer {IA_KEYS['mistral']}", "Content-Type": "application/json"},
}

def ask_all(prompt):
    respostas = {}
    for ia in ENDPOINTS:
        try:
            if not IA_KEYS.get(ia): continue
            print(f"🔁 [{ia}] solicitando resposta...", flush=True)
            payload = {}

            if ia in ["chatgpt", "deepseek", "mistral"]:
                payload = {
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False
                }
                res = requests.post(ENDPOINTS[ia], headers=HEADERS[ia], json=payload, timeout=30)
                content = res.json().get("choices", [{}])[0].get("message", {}).get("content", "")

            elif ia == "claude":
                payload = {
                    "model": "claude-3-opus-20240229",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 1024
                }
                res = requests.post(ENDPOINTS[ia], headers=HEADERS[ia], json=payload, timeout=30)
                content = res.json().get("content", [{}])[0].get("text", "")

            elif ia == "gemini":
                payload = {
                    "contents": [{"parts": [{"text": prompt}]}]
                }
                res = requests.post(ENDPOINTS[ia] + f"?key={IA_KEYS['gemini']}", json=payload, timeout=30)
                content = res.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")

            respostas[ia] = content.strip()

        except Exception as e:
            print(f"⚠️ [{ia}] falhou: {e}", flush=True)
    return respostas

def choose_best(respostas):
    return max(respostas.items(), key=lambda x: len(x[1].split()))  # simples: mais conteúdo = melhor

print("🧠 Modo REPL iniciado. Digite sua pergunta para a ETΩ.\nDigite 'exit' para sair.\n", flush=True)

while True:
    user_input = input("💬 Você: ")
    if user_input.lower() in ['exit', 'quit']: break

    contexto = f"""
    Snapshot atual da ETΩ:
    {snapshot}

    Considere isso como base para sua resposta:

    Pergunta: {user_input}
    """

    respostas = ask_all(contexto)
    ia, resposta = choose_best(respostas)

    print(f"\n✅ Resposta da IA dominante ({ia}):\n{resposta}\n", flush=True)

    with open(HISTORY_PATH, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat()} | {ia} | {user_input} -> {resposta}\n")
