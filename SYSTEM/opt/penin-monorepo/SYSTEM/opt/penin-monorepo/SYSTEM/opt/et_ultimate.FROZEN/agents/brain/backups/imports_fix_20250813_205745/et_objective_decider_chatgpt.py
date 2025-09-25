import json
from et_llm_bridge import chamar_ia

def decidir_objetivo(contexto_atual, historico_objetivos):
    """
    Usa o ChatGPT para decidir o próximo objetivo estratégico da IA
    com base no contexto atual e histórico de objetivos.
    """
    prompt = f"""
Você é um planejador cognitivo para uma IA que evolui continuamente sua própria
arquitetura e a Equação de Turing. 

Contexto atual:
{json.dumps(contexto_atual, ensure_ascii=False, indent=2)}

Histórico de objetivos:
{json.dumps(historico_objetivos, ensure_ascii=False, indent=2)}

Sua tarefa:
1. Avaliar lacunas e potenciais evoluções.
2. Decidir um único objetivo central para a próxima rodada.
3. Justificar brevemente a escolha.
4. Responder em JSON no formato:
{{
  "objetivo": "texto curto e claro",
  "justificativa": "explicação breve"
}}
    """

    try:
        resposta = chamar_ia("gpt-5", prompt, max_tokens=1000)
        try:
            return json.loads(resposta)
        except json.JSONDecodeError:
            return {"objetivo": "Evoluir arquitetura", "justificativa": resposta.strip()}
    except Exception as e:
        return {"objetivo": "Evoluir arquitetura", "justificativa": f"Erro ao decidir objetivo: {e}"}

