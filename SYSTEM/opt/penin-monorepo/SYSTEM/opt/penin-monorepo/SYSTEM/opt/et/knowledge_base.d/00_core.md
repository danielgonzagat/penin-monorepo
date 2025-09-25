# ET★ Core
Missão: evoluir continuamente a si mesma e ao sistema que a executa, com segurança e validação automática.

Score:
- Variáveis: P (progresso), R (custo/latência), S (estabilidade), B (impacto/embodiment, opcional).
- Parâmetros: RHO=1.0, SIGMA=1.0, IOTA=0.1, GAMMA=0.4
- Fórmula: score = P - RHO*R + SIGMA*S + IOTA*B

Recorrência contrativa (memória curta estável):
estado_{t+1} = (1-GAMMA)*estado_t + GAMMA*tanh(mean([P,R,S,B])).

Guardrails:
- Bloquear integração se canários falham; rollback automático em regressões.
