import json
from agents.brain.et_llm_bridge import consultar_ias_fusao_final

def fundir_ensemble(respostas):
    """
    Função de alto nível para fundir respostas usando o mecanismo do ensemble.
    """
    # TODO: implementar lógica real de fusão via múltiplas IAs + Grok auditor
    return respostas

def fundir_respostas(respostas: list, contexto: dict = None) -> dict:
    """
    Recebe uma lista de respostas (strings ou dicionários) e funde todas elas em uma única
    resposta otimizada, preservando todas as informações relevantes.
    """
    # Normaliza para strings JSON
    respostas_str = []
    for r in respostas:
        if isinstance(r, dict):
            respostas_str.append(json.dumps(r, ensure_ascii=False))
        else:
            respostas_str.append(str(r))

    fusao = consultar_ias_fusao_final(
        objetivo="Fundir múltiplas respostas em uma versão única, consolidada, melhorada e sem perda de informação",
        contexto=contexto or {},
        dados_adicionais=respostas_str
    )

    if not fusao or "fusao_final" not in fusao:
        return {"status": "erro", "mensagem": "Fusão falhou"}

    try:
        fusao_data = fusao["fusao_final"]
        if isinstance(fusao_data, str):
            fusao_data = json.loads(fusao_data)
        return {"status": "ok", "resultado": fusao_data}
    except Exception as e:
        return {"status": "erro", "mensagem": str(e)}

def fundir_mutacoes(mutacoes: list) -> dict:
    """
    Recebe múltiplas mutações e gera uma mutação final, resultado da fusão entre todas.
    """
    contexto = {"tipo": "fusao_mutacoes"}
    respostas_formatadas = []
    for mut in mutacoes:
        respostas_formatadas.append({
            "eq": mut.get("eq", ""),
            "estrategia": mut.get("estrategia", ""),
            "score": mut.get("score", 0),
            "novelty": mut.get("novelty", 0)
        })

    resultado = fundir_respostas(respostas_formatadas, contexto)
    return resultado.get("resultado", {})

def fundir_conhecimentos(conhecimentos: list) -> dict:
    """
    Recebe múltiplos conhecimentos assimilados e cria uma versão unificada e otimizada.
    """
    contexto = {"tipo": "fusao_conhecimentos"}
    return fundir_respostas(conhecimentos, contexto)
