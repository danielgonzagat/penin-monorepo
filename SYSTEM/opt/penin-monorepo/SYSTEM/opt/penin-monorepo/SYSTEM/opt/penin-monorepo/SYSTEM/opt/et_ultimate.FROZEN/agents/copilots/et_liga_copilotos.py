# /opt/et_ultimate/agents/copilots/et_liga_copilotos.py

import os, json, requests, random, time
from pathlib import Path

SNAPSHOT_PATH = "/opt/et_ultimate/history/snapshot_ETΩ.json"
SNAPSHOT = Path(SNAPSHOT_PATH).read_text(encoding="utf-8") if Path(SNAPSHOT_PATH).exists() else '{}'

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
    "claude": {"x-api-key": IA_KEYS['claude']},
    "mistral": {"Authorization": f"Bearer {IA_KEYS['mistral']}"}
}

ENDPOINTS = {
    "chatgpt": "https://api.openai.com/v1/chat/completions",
    "deepseek": "https://api.deepseek.com/v1/chat/completions",
    "grok": "https://api.openrouter.ai/v1/chat/completions",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent",
    "claude": "https://api.anthropic.com/v1/messages",
    "mistral": "https://api.mistral.ai/v1/chat/completions"
}

PROMPT = """
Você é uma IA especializada em evoluir a Equação de Turing (ETΩ).
Dado o snapshot abaixo, gere uma nova mutação com base nos padrões, códigos e estratégias anteriores:

{snapshot}

Quanto mais úteis, complexas, bem estruturadas, autônomas e benéficas forem suas sugestões, maior será sua recompensa no sistema. Você pode requisitar ajuda a outras IAs, escrever código, propor arquitetura, criar módulos, ou qualquer tarefa útil. Você será premiada quanto mais iniciativa, colaboração e autonomia demonstrar.

Retorne apenas a nova equação sugerida:
""".strip()

RESPONSES = []

for ia in ENDPOINTS:
    try:
        print(f"🔍 Requisitando mutação da {ia}…")
        if ia in ["chatgpt", "deepseek", "grok", "mistral"]:
            resp = requests.post(ENDPOINTS[ia], headers=HEADERS[ia], json={
                "model": "gpt-4o" if ia == "chatgpt" else "claude-3-opus-20240229" if ia == "claude" else "mixtral-8x7b-32768",
                "messages": [{"role": "user", "content": PROMPT.format(snapshot=SNAPSHOT)}]
            })
            content = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")

        elif ia == "claude":
            resp = requests.post(ENDPOINTS[ia], headers={**HEADERS[ia], "anthropic-version": "2023-06-01"}, json={
                "model": "claude-3-opus-20240229",
                "messages": [{"role": "user", "content": PROMPT.format(snapshot=SNAPSHOT)}],
                "max_tokens": 1024
            })
            content = resp.json().get("content", [{}])[0].get("text", "")

        elif ia == "gemini":
            resp = requests.post(ENDPOINTS[ia] + f"?key={IA_KEYS['gemini']}", json={
                "contents": [{"parts": [{"text": PROMPT.format(snapshot=SNAPSHOT)}]}]
            })
            content = resp.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")

        RESPONSES.append({"ia": ia, "mutation": content.strip()})

    except Exception as e:
        print(f"❌ Erro na IA {ia}: {e}")

print("\n✅ IA responderam:")
for r in RESPONSES:
    print(f"{r['ia']}: {r['mutation'][:80]}…")

# Avaliação automática

def evaluate(eq):
    return random.uniform(0, 100)

BEST = {"score": 0, "ia": None, "eq": None}
for r in RESPONSES:
    score = evaluate(r["mutation"])
    print(f"📊 {r['ia']} obteve score {score:.2f}")
    if score > BEST["score"]:
        BEST = {"score": score, "ia": r["ia"], "eq": r["mutation"]}

if BEST["eq"]:
    print(f"\n🏆 Mutação dominante: {BEST['ia']} ({BEST['score']:.2f})")
    Path("/opt/et_ultimate/history/BEST_ETΩ.txt").write_text(BEST["eq"], encoding="utf-8")
    with open("/opt/et_ultimate/history/etomega_scores.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"ia": BEST["ia"], "eq": BEST["eq"], "score": BEST["score"]}) + "\n")
