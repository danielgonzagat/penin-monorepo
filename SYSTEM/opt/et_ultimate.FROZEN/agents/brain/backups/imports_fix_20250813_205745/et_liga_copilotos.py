import json
from et_llm_bridge import consultar_ias_multiplas, consultar_ias_fusao_final

COPILOTOS_ATIVOS = ["chatgpt", "deepseek", "mistral"]

def executar_liga_copilotos(tarefa, contexto_extra=None):
    """
    Ativa todos os copilotos disponíveis para colaborar em uma tarefa,
    coleta suas respostas e gera uma fusão final otimizada.
    """
    try:
        if contexto_extra is None:
            contexto_extra = {}

        contexto_base = {
            "tarefa": tarefa,
            "descricao": f"Executar tarefa de alto impacto para evolução da Equação de Turing",
            **contexto_extra
        }

        # Solicitar respostas individuais
        respostas = consultar_ias_multiplas(
            objetivo=f"Executar tarefa: {tarefa}",
            contexto=contexto_base,
            dados_adicionais=[],
            ias=COPILOTOS_ATIVOS
        )

        # Unificar tudo em uma resposta final otimizada
        fusao_final = consultar_ias_fusao_final(
            objetivo=f"Unificar respostas de copilotos para a tarefa: {tarefa}",
            contexto={"respostas": respostas, "tarefa": tarefa},
            dados_adicionais=[]
        )

        return {
            "respostas_individuais": respostas,
            "fusao_final": fusao_final.get("fusao_final", "")
        }

    except Exception as e:
        print(f"Erro na execução da liga de copilotos: {e}")
        return {"respostas_individuais": [], "fusao_final": ""}
