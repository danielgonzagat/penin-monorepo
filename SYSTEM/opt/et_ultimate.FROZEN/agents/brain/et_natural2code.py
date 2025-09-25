import json
from agents.brain.et_llm_bridge import chamar_ia

def natural2code(objetivo, autocritica_texto=""):
    """
    Converte linguagem natural (objetivo + autocrítica) em instruções simbólicas
    ou código Python que pode ser incorporado à arquitetura da IA.
    """
    prompt = f"""
Você é um tradutor especializado em transformar intenções e autocríticas
em código Python funcional. 

Objetivo: {objetivo}

Autocrítica: {autocritica_texto}

Sua tarefa:
1. Avaliar o que precisa ser criado ou modificado.
2. Criar código Python totalmente funcional, pronto para execução,
   que implemente essa evolução no cérebro da IA.
3. Garantir compatibilidade com a arquitetura atual e respeitar a Equação de Turing.
4. Retornar um JSON no formato:
{{
  "tipo": "codigo" ou "instrucoes",
  "plano": "Resumo em linguagem natural",
  "codigo": "código Python completo se houver"
}}
    """

    try:
        resposta = chamar_ia("gpt-5", prompt, max_tokens=2000)
        try:
            return json.loads(resposta)
        except json.JSONDecodeError:
            return {"tipo": "instrucoes", "plano": resposta.strip(), "codigo": ""}
    except Exception as e:
        return {"tipo": "erro", "plano": f"Falha ao gerar código: {e}", "codigo": ""}

