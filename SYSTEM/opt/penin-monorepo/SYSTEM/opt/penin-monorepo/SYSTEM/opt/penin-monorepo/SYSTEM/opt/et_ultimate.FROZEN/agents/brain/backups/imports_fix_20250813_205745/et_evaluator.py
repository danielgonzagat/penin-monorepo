import json
from pathlib import Path
from et_llm_bridge import consultar_ias_fusao_final

def avaliar_mutacoes(mutacoes):
    """
    Avalia uma lista de mutações usando múltiplas IAs e retorna as avaliações
    detalhadas de cada uma, incluindo score, coerência, impacto e alinhamento
    com a Equação de Turing.
    """
    avaliacoes = []
    try:
        for mut in mutacoes:
            respostas = []
            for ia in ["chatgpt", "deepseek", "mistral"]:
                resp = consultar_ias_fusao_final(
                    objetivo="Avaliar mutação para evolução contínua da Equação de Turing",
                    contexto={"mutacao": mut.get("eq", ""), "origem": ia},
                    dados_adicionais=[]
                )
                if resp and "fusao_final" in resp:
                    respostas.append(resp["fusao_final"])
            
            if respostas:
                fusao = consultar_ias_fusao_final(
                    objetivo="Fundir avaliações em um único veredito otimizado",
                    contexto={"mutacao": mut.get("eq", "")},
                    dados_adicionais=respostas
                )
                mut["avaliacao_fundida"] = fusao.get("fusao_final", "")
                mut["score"] = extrair_score(fusao.get("fusao_final", ""))
                mut["impacto"] = extrair_impacto(fusao.get("fusao_final", ""))
                mut["coerencia"] = extrair_coerencia(fusao.get("fusao_final", ""))
                mut["alinhamento_turing"] = extrair_alinhamento_turing(fusao.get("fusao_final", ""))
            
            avaliacoes.append(mut)
    except Exception as e:
        print(f"Erro na avaliação de mutações: {e}")
    return avaliacoes

def selecionar_dominante(avaliacoes):
    """
    Seleciona a mutação dominante com base no score e outros critérios de priorização.
    """
    if not avaliacoes:
        return None
    return max(avaliacoes, key=lambda m: m.get("score", 0))

def extrair_score(texto):
    try:
        for token in texto.split():
            if token.replace('.', '', 1).isdigit():
                valor = float(token)
                if 0 <= valor <= 10:
                    return valor
    except:
        pass
    return 0.0

def extrair_impacto(texto):
    if "alto impacto" in texto.lower():
        return "alto"
    elif "médio impacto" in texto.lower():
        return "medio"
    elif "baixo impacto" in texto.lower():
        return "baixo"
    return "desconhecido"

def extrair_coerencia(texto):
    if "alta coerência" in texto.lower():
        return "alta"
    elif "média coerência" in texto.lower():
        return "media"
    elif "baixa coerência" in texto.lower():
        return "baixa"
    return "desconhecida"

def extrair_alinhamento_turing(texto):
    if "alinhado à equação de turing" in texto.lower():
        return True
    return Fal
