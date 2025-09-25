# ET★ Guardrails

- Bloquear integração quando:
  - R (latência/custo) > limite
  - S (estabilidade) < limiar
  - Canários (públicos/ocultos) falham
- Rollback automático + registro de motivo
- Transparência: escrever sempre no log de decisão
