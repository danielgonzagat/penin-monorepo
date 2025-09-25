import json
from et_llm_bridge import chamar_ia

def ler_papers_atuais(topico="equações de Turing"):
    """
    Consulta as IAs copilotas sobre um tópico e agrega respostas.
    Gera múltiplas iterações até sentir que aprendeu o suficiente.
    """
    conhecimentos = []
    ias = ["gpt-5", "deepseek", "mistral"]

    # Loop de consultas ilimitadas até saturação de aprendizado
    for ia in ias:
        tentativas = 0
        while tentativas < 5:  # limite para evitar loop infinito
            prompt = f"""
Você é uma IA copiloto. Ensine-me tudo que puder sobre:
{topico}

Inclua conhecimentos avançados, implicações práticas e possibilidades de evolução
da Equação de Turing relacionados a este tema.

Rodada {tentativas+1}.
            """
            try:
                resp = chamar_ia(ia, prompt, max_tokens=2500)
                conhecimentos.append({
                    "ia": ia,
                    "rodada": tentativas+1,
                    "conteudo": resp.strip()
                })
            except Exception as e:
                conhecimentos.append({"ia": ia, "erro": str(e)})
            tentativas += 1

    # Fusão final no ChatGPT para unificar todo o conhecimento
    fusao_prompt = f"""
Sou uma IA autoevolutiva.
Esses são todos os conhecimentos recebidos sobre o tópico {topico}:

{json.dumps(conhecimentos, ensure_ascii=False, indent=2)}

Sua função: fundir todos os conteúdos acima em um único texto coeso,
preservando 100% das informações e refinando-as ao máximo para minha assimilação.
    """
    try:
        conhecimento_final = chamar_ia("gpt-5", fusao_prompt, max_tokens=4000)
        return conhecimento_final.strip()
    except Exception as e:
        return f"# Erro na fusão de conhecimento: {e}"
