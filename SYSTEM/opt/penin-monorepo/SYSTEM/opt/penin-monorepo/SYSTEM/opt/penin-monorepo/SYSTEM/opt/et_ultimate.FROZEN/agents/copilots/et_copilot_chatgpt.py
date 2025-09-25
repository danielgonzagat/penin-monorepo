from openai import OpenAI
from pathlib import Path
import time

client = OpenAI(api_key="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")

HISTORY_PATH = Path("/opt/et_ultimate/history/BEST_ETΩ.txt")
LOG_PATH = Path("/opt/et_ultimate/workspace/chatgpt_replies.log")
MUTATION_PATH = Path("/opt/et_ultimate/workspace/MUTATION_ETΩ.txt")

def ler_equacao():
    return HISTORY_PATH.read_text() if HISTORY_PATH.exists() else "Sem equação."

def perguntar(mensagem):
    r = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Você é um copiloto técnico que reescreve, otimiza e muta expressões da Equação de Turing."},
            {"role": "user", "content": mensagem}
        ]
    )
    return r.choices[0].message.content.strip()

if __name__ == "__main__":
    equacao = ler_equacao()
    resposta = perguntar(f"Otimize esta Equação de Turing:\n\n{equacao}")
    LOG_PATH.write_text(f"[{time.ctime()}] Resposta da IA:\n{resposta}\n")
    MUTATION_PATH.write_text(resposta)
    print(resposta)
