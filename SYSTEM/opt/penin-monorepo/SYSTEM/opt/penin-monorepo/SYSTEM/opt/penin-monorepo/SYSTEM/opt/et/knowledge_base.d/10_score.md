# ET★ Score e Métricas

- P: aderência a objetivos, correção, utilidade prática.
- R: latência média, tokens gerados, uso de CPU/GPU.
- S: consistência entre runs, variação de formato/estilo, ausência de regressões em canários.
- B: impacto em sistemas/ambiente (se aplicável).

Boas práticas:
- Normalizar métricas para [0,1].
- Logar {ts, temp, max_tokens, P,R,S,B, score, decisão, latência, tok/s}.
- Manter histórico para detectar drift.
