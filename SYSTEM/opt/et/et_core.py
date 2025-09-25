# -*- coding: utf-8 -*-
# ET★ núcleo inicial — a própria ET★ poderá reescrever este arquivo.
import math

# Pesos do score (ref.: LLM em CPU, baixa latência e estabilidade alta)
RHO   = 0.6   # penalização de custo/latência
SIGMA = 0.7   # recompensa por estabilidade/validação
IOTA  = 0.0   # embodiment (0 em puro software)
GAMMA = 0.5   # contração da recorrência

def calcular_score(P: float, R: float, S: float, B: float = 0.0) -> float:
    """
    P: progresso (0..1)    -> ex.: taxa de acerto nos canários
    R: custo   (segundos)  -> latência média por consulta (quanto menor, melhor)
    S: estabilidade (0..1) -> ex.: consistência/ausência de regressão
    B: embodiment          -> aqui 0 (sem robótica)
    """
    return float(P - RHO*R + SIGMA*S + IOTA*B)

def recorrencia_contrativa(estado: float, phi: list[float]) -> float:
    """ Atualiza estado mantendo estabilidade no longo prazo. """
    if not phi:
        return estado
    m = sum(phi)/len(phi)
    return (1.0 - GAMMA)*estado + GAMMA*math.tanh(m)

def guardrails(decisao: bool, score: float, termos: dict) -> tuple[bool, str]:
    """
    Rejeita mudanças perigosas/regressivas mesmo se o score parecer bom.
    - Latência explosiva
    - Queda forte de progresso
    - Instabilidade (S baixo)
    """
    R = float(termos.get("R", 9.99))
    P = float(termos.get("P", 0.0))
    S = float(termos.get("S", 0.0))

    if R > 6.0:
        return False, "reprovado: latência média acima de 6s"
    if P < 0.50:
        return False, "reprovado: progresso abaixo de 0.50"
    if S < 0.50:
        return False, "reprovado: estabilidade abaixo de 0.50"
    return bool(decisao), "ok"
