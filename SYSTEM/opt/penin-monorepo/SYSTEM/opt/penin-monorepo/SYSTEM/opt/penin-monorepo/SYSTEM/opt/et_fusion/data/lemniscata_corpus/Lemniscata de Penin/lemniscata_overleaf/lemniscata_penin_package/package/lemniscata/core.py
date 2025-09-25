# -*- coding: utf-8 -*-
from typing import Dict, Tuple, Optional
from .audit import AuditLogger

def clamp01(x: float) -> float:
    if x < 0.0: 
        return 0.0
    if x > 1.0: 
        return 1.0
    return x

def lemniscata(E: float, N: float, I: float, 
               mode: str = "partial", 
               audit: Optional[AuditLogger] = None,
               reason: Optional[str] = None) -> Tuple[float, Dict]:
    E = max(0.0, E)
    N = max(0.0, N)
    I = clamp01(I)

    iN = (1.0 - I) * N
    if mode == "hard":
        if I < 1.0:
            P = E
            rejected = True
        else:
            P = E + N
            rejected = False
    else:  # partial
        P = E + I * N
        rejected = (I == 0.0)

    P = max(E, min(E + N, P))

    if audit is not None:
        audit.log(E=E, N=N, I=I, iN=iN, P=P, mode=mode, reason=reason)

    info = {"iN": iN, "rejected": rejected, "mode": mode}
    return P, info


class InfinityOperator:
    def __init__(self, mode: str = "partial", audit: Optional[AuditLogger] = None):
        assert mode in ("partial", "hard")
        self.mode = mode
        self.audit = audit

    def apply(self, E: float, N: float, I: float, reason: Optional[str] = None) -> Tuple[float, Dict]:
        return lemniscata(E=E, N=N, I=I, mode=self.mode, audit=self.audit, reason=reason)


def invariants_hold(E: float, N: float, I: float, mode: str = "partial") -> bool:
    P, info = lemniscata(E, N, I, mode=mode)
    ok = True
    ok &= (P >= E - 1e-12)
    ok &= (P <= E + N + 1e-12)
    eps = 1e-6
    if mode == "partial":
        P_I_plus, _ = lemniscata(E, N, min(1.0, I + eps), mode=mode)
        ok &= (P_I_plus + 1e-9 >= P)
        P_N_plus, _ = lemniscata(E, N + eps, I, mode=mode)
        ok &= (P_N_plus + 1e-9 >= P)
    P1, _ = lemniscata(E, N, I, mode=mode)
    P2, _ = lemniscata(P1, 0.0, 1.0, mode=mode)
    ok &= (abs(P2 - P1) <= 1e-9)
    return bool(ok)
