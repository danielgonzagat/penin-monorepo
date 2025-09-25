# API — Lemniscata de Penin (infty-bar)

Função mínima (Python):
```python
from lemniscata.core import lemniscata
P, info = lemniscata(E, N, I, mode="partial", audit=logger, reason="bias")
```
Entradas:
- E (float >= 0, normalizado), N (float >= 0, normalizado), I (float in [0,1]).
- mode: "partial" (P=E+I*N) ou "hard" (I<1 => P=E; I=1 => P=E+N).
- audit: instância AuditLogger (opcional).
- reason: string com o motivo do iN>0 (opcional).

Saída:
- P (float), info = {"iN": float, "rejected": bool, "mode": str}.

Auditoria:
- CSV append-only: timestamp,E,N,I,iN,P,mode,reason.
- Recomenda-se cálculo de hash ao final de cada execução para selar o arquivo.

Boas práticas:
- Normalizar E e N (scaling.MinMax).
- Calcular I via I_from_risk ou agregadores I_hard_min / I_product / I_weighted.
