import random
from agents.brain.et_memory_sync import buscar_similares
from agents.brain.et_llm_bridge import consultar_ias_fusao_final, consultar_ias_multiplas

OBJETIVO_PRIMORDIAL = "Evoluir continuamente a Equação de Turing (ETΩ) e a própria IA com segurança, autonomia e novidade mensurável."

def definir_objetivo():
    """
    Define o objetivo da rodada considerando:
    - Objetivo primordial
    - Contexto atual
    - Sugestões das IAs copilotas
    """
    try:
        contexto_atual = {
            "objetivo_primordial": OBJETIVO_PRIMORDIAL,
            "historico_semelhante": buscar_similares(OBJETIVO_PRIMORDIAL, k=5)
        }

        # Consultar múltiplas IAs para definir objetivo específico desta rodada
        respostas_ias = consultar_ias_multiplas(
            objetivo="Definir objetivo estratégico da próxima rodada",
            contexto=contexto_atual,
            dados_adicionais=[]
        )

        # Fundir as respostas em um objetivo final
        fusao = consultar_ias_fusao_final(
            objetivo="Unificar respostas em objetivo final desta rodada",
            contexto={"respostas": respostas_ias},
            dados_adicionais=[]
        )

        objetivo_texto = fusao.get("fusao_final", OBJETIVO_PRIMORDIAL)
        return {"text": objetivo_texto, "topic": _extrair_topico(objetivo_texto)}

    except Exception as e:
        print(f"Erro ao definir objetivo: {e}")
        return {"text": OBJETIVO_PRIMORDIAL, "topic": "general"}

def registrar_feedback(topic, score):
    """
    Registra feedback do desempenho da rodada para referência futura.
    """
    print(f"[Feedback] Tópico: {topic} | Score: {score:.2f}")

def ajustar_objetivo_por_critica(objetivo, critica):
    """
    Ajusta o objetivo com base na crítica da rodada.
    """
    try:
        ajuste = f"{objetivo} | Ajuste sugerido: {critica}"
        return ajuste
    except Exception:
        return objetivo

def _extrair_topico(objetivo):
    """
    Extrai um tópico simplificado do objetivo.
    """
    palavras = objetivo.split()
    if len(palavras) > 3:
        return "_".join(palavras[:3]).lower()
    return "_".join(palavras).lower()
