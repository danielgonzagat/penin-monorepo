import json
from et_llm_bridge import chamar_ia

def gerar_mutacao_online(conhecimento_novo, objetivo_atual):
    """
    Gera mutações em tempo real usando IAs externas (ChatGPT, DeepSeek, Mistral),
    integrando o conhecimento recém-adquirido ao objetivo atual.
    """
    prompt = f"""
Você é responsável por criar mutações de código e arquitetura para uma IA 
autoevolutiva que segue a Equação de Turing.

Objetivo atual:
{objetivo_atual}

Conhecimento novo assimilado:
{json.dumps(conhecimento_novo, ensure_ascii=False, indent=2)}

Tarefa:
1. Gerar uma mutação prática e segura que aproveite ao máximo o conhecimento novo.
2. Garantir que a mutação evolua a Equação de Turing e aumente as capacidades da IA.
3. Responder APENAS com o código Python resultante, sem explicações.
    """

    try:
        resposta = chamar_ia("gpt-5", prompt, max_tokens=2000)
        return resposta.strip()
    except Exception as e:
        return f"# Erro ao gerar mutação online: {e}"
