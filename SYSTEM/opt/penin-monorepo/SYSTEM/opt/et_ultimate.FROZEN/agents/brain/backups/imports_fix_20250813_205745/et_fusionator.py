from et_llm_bridge import consultar_ias_fusao_final

def fundir_mutacoes(mutacoes):
    """
    Realiza a fusão de múltiplas mutações usando todas as IAs disponíveis
    e gera uma única mutação otimizada que preserve todas as evoluções.
    """
    try:
        if not mutacoes:
            return {}

        # Coleta apenas o código/ideias de cada mutação
        dados_mutacoes = [m.get("eq", "") for m in mutacoes if m.get("eq")]

        # Pede para as IAs fundirem todas as mutações mantendo 100% do conhecimento
        fusao = consultar_ias_fusao_final(
            objetivo="Fundir todas as mutações em uma única versão otimizada e evoluída",
            contexto={"descricao": "Combinar mutações preservando todo o conhecimento e melhorias"},
            dados_adicionais=dados_mutacoes
        )

        mutacao_fundida = {
            "eq": fusao.get("fusao_final", ""),
            "origem": "fusao_total_inteligencias",
            "score": 0.0,
            "novelty": 0.0,
            "sim_max": 0.0
        }

        return mutacao_fundida
    except Exception as e:
        print(f"Erro ao fundir mutações: {e}")
        return {}
