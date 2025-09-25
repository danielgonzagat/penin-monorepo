#!/usr/bin/env bash
set -euo pipefail
exec 9>/var/lock/lemni-autopilot.lock || exit 0
flock -n 9 || exit 0

cd /opt/lemniscata
LOG=/var/log/lemniscata/autopilot.log
TS="$(date -u +%Y%m%dT%H%M%SZ)"

# (a) Garante serviços essenciais (tentativa suave; sem despejar logs)
docker compose up -d router api mock-llm >/dev/null 2>&1 || true

# (b) Saúde do Router (até 20s)
ROUTER_OK=0
for i in {1..40}; do
  code="$(curl -s -o /dev/null -w "%{http_code}" http://localhost:18009/health || true)"
  [ "$code" = "200" ] && ROUTER_OK=1 && break || sleep 0.5
done

# (c) Monta a instrução (do arquivo)
INSTR="$(cat _autopilot/instructions.md 2>/dev/null || echo 'Aplicar comentário dgm-note em services/api/app.py')"

# (d) Dispara DGM guiado (via Router se saudável)
if [ "$ROUTER_OK" = "1" ]; then
  /usr/local/bin/lemni-dgm-say "$INSTR" >/dev/null 2>&1 || true
else
  # fallback: executa outer-loop simples (sem instrução) e promove-condicional
  /usr/local/bin/lemni-outer-loop >/dev/null 2>&1 || true
  /usr/local/bin/lemni-promote-ifok >/dev/null 2>&1 || true
fi

# (e) Eval e promote-ifok (caso o runner não tenha gerado eval.json)
LAST="$(ls -1dt _artifacts/dgm/* 2>/dev/null | head -n1 || true)"
if [ -n "$LAST" ] && [ ! -s "$LAST/eval.json" ] && command -v /usr/local/bin/lemni-eval-batch >/dev/null 2>&1; then
  /usr/local/bin/lemni-eval-batch "$LAST" >/dev/null 2>&1 || true
fi
command -v /usr/local/bin/lemni-promote-ifok >/dev/null 2>&1 && /usr/local/bin/lemni-promote-ifok >/dev/null 2>&1 || true

# (f) Resumo curto no log
LEDG_TAIL="$(tail -n 2 .ledger 2>/dev/null || true)"
printf "[%s] autopilot done | router_ok=%s | last=%s\n%s\n" "$TS" "$ROUTER_OK" "${LAST:-n/a}" "$LEDG_TAIL" >> "$LOG"
