import json
from et_llm_bridge import consultar_ias_fusao_final, consultar_ias_multiplas

def executar_global_benchmark(modulos, tecnologias_referencia):
    """
    Compara a arquitetura atual da IA com as arquiteturas de referência mais avançadas do mundo
    e gera recomendações de evolução imediata.
    """
    try:
        # Resumo completo da arquitetura atual
        resumo_atual = {
            "arquivos": list(modulos.keys()),
            "conteudo": {nome: modulos[nome] for nome in modulos}
        }

        # Consulta múltiplas IAs pedindo comparação com as melhores arquiteturas
        respostas_ias = consultar_ias_multiplas(
            objetivo="Comparar arquitetura atual com arquiteturas de referência",
            contexto={
                "arquitetura_atual": resumo_atual,
                "arquiteturas_referencia": tecnologias_referencia
            },
            dados_adicionais=[]
        )

        # Fusão das respostas para gerar a recomendação final
        fusao = consultar_ias_fusao_final(
            objetivo="Escolher a tecnologia mais impactante para evolução imediata",
            contexto={
                "arquitetura_atual": resumo_atual,
                "respostas_ias": respostas_ias
            },
            dados_adicionais=[]
        )

        return {
            "recomendacao_final": fusao.get("fusao_final", ""),
            "respostas_ias": respostas_ias
        }

    except Exception as e:
        print(f"Erro no benchmark global: {e}")
        return {}
