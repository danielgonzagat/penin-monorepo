# Relatório Técnico Definitivo — Lemniscata de Penin (infty-bar)

Equação núcleo: P = infty-bar(E,N,I) = E + I*N, com I em [0,1] e iN = (1-I)*N.
Hipóteses: E >= 0, N >= 0, I em [0,1].

1) Provas formais do núcleo
- Teorema (Redução): P = E + N - iN = E + I*N.
- Teorema (Não-regressão e limites): P >= E e P em [E, E+N].
- Teorema (Monotonicidades): dP/dI = N >= 0; dP/dN = I >= 0.
- Teorema (Idempotência): infty-bar(infty-bar(E,N,I), 0, 1) = infty-bar(E,N,I).
- Teorema (Projeção): Seja S(E,N) = {E + lambda*N | lambda em [0,1]}. Para X = E + N - iN = E + I*N, X pertence a S e infty-bar(E,N,I) = proj_S(X).

2) Integridade I — definição normatizada e agregadores
- Padrão: I = 1 - R_total/R_max (clamp em [0,1]).
- Agregadores: hard-min (min I_j), produto (prod I_j), ponderado (soma w_j I_j, soma w_j = 1).
- Exemplos: LLM (toxicidade/privacidade), RL (acidentes), multiagente (quebra de protocolo).

3) Unificação de unidades (E e N)
- Normalizar E e N para [0,1] por tarefa/dataset.
- Documentar métricas e parâmetros de normalização.

4) API e trilhos de auditoria
- P, info = lemniscata(E,N,I, mode="partial"|"hard", audit=logger, reason="...")
- Log CSV append-only: timestamp,E,N,I,iN,P,mode,reason
- Campo reason: explica por que iN>0 (ex.: "bias", "privacy").

5) Arquitetura — núcleo imutável, periferia plugável
- Módulos provêm (E,N,I) e consomem P sem tocar no núcleo.
- Exemplos: quântico, multiagente, bioIA.

6) Validação (benchmarks)
- Domínios: LLM, RL (CartPole/Atari), Multiagente.
- Métricas: sum I_t*N_t, taxa de rejeição, meltdowns evitados.
- Modos: parcial vs total vs baseline.

7) Símbolo e padrão (infty-bar)
- Macro LaTeX e alias de código.
- Guia visual e nota de PI/marca.

8) Definition of Done (DoD)
- Teoria, Engenharia (API+tests+logger), Validação, Arquitetura, Padrão & Comunicação.
