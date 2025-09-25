from et_llm_bridge import consultar_ias_fusao_final

def avaliar_estrategia(equacao: str, autocritica: str) -> str:
    """
    Avalia a estratégia de evolução com base na equação e na autocrítica atual.
    Utiliza múltiplas IAs para gerar diferentes perspectivas e funde as respostas
    em uma estratégia única, otimizada e coerente com o objetivo de evolução contínua.
    """
    try:
        respostas = []
        
        # Chamadas às IAs copilotas para gerar estratégias independentes
        for ia in ["chatgpt", "deepseek", "mistral"]:
            resp = consultar_ias_fusao_final(
                objetivo=f"Avaliar e propor estratégia de evolução para a equação: {equacao}",
                contexto={"autocritica": autocritica, "ia_origem": ia},
                dados_adicionais=[]
            )
            if resp and "fusao_final" in resp:
                respostas.append(resp["fusao_final"])
        
        if not respostas:
            return "Nenhuma estratégia gerada."
        
        # Fusão final das estratégias
        fusao = consultar_ias_fusao_final(
            objetivo="Fundir estratégias múltiplas em uma única estratégia otimizada",
            contexto={"equacao": equacao, "autocritica": autocritica},
            dados_adicionais=respostas
        )
        
        return fusao.get("fusao_final", "Falha na fusão de estratégias.")
    
    except Exception as e:
        return f"Erro ao avaliar estratégia: {e}"
