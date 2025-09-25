import os
from openai import OpenAI

# O proxy LiteLLM está ouvindo aqui:
GPT5_BASE_URL = os.getenv("GPT5_BASE_URL", "http://127.0.0.1:8003/v1")
# Como o proxy já usa OPENAI_API_KEY, aqui pode ser qualquer valor
GPT5_API_KEY  = os.getenv("GPT5_PROXY_KEY", "not-needed")

SYSTEM_PROF = (
    "Você é um PROFESSOR especialista na Equação de Turing. "
    "Ensine, explique, corrija e cite as fontes do contexto quando possível. "
    "Se o contexto não cobrir algo, diga claramente."
)

class GPT5Runner:
    name = "gpt5"

    def __init__(self):
        self.client = OpenAI(base_url=GPT5_BASE_URL, api_key=GPT5_API_KEY)

    def generate(self, pergunta: str, context: str) -> str:
        """
        Gera resposta usando RAG (context vem do ETKB).
        IMPORTANTE: não enviar temperature - GPT-5 só aceita = 1.
        """
        msgs = [
            {"role": "system", "content": SYSTEM_PROF},
            {
                "role": "user",
                "content": (
                    "Contexto recuperado (ETKB):\n"
                    f"{context}\n\n"
                    "Pergunta:\n"
                    f"{pergunta}\n\n"
                    "Regras: use o contexto acima; aponte trechos/nomes de arquivos do contexto quando pertinente; "
                    "se faltar base no contexto, diga o que falta."
                ),
            },
        ]

        resp = self.client.chat.completions.create(
            model="gpt-5",
            messages=msgs,
        )
        return resp.choices[0].message.content.strip()
