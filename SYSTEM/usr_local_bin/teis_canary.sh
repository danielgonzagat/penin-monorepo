#!/usr/bin/env bash
set -euo pipefail

# --- Config carregada do /etc/default/teis-canary ---
: "${TEIS_WORM_PATH:=/root/teis_worm.log}"
: "${TEIS_CHAMPION_TAG:=teis-champion-001}"
: "${TEIS_CANDIDATE_TAG:=teis-candidate-002}"
: "${TEIS_PY:=python3}"
: "${TEIS_WORKDIR:=/root}"

LOG_DIR="/var/log/teis"
mkdir -p "$LOG_DIR"

ts="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_JSON="$LOG_DIR/canary_${ts}.json"

cd "$TEIS_WORKDIR"

# Executa canário → grava resultado JSON e deixa tudo no WORM
$TEIS_PY canary_runner.py \
  --champion  "$TEIS_CHAMPION_TAG" \
  --candidate "$TEIS_CANDIDATE_TAG" \
  --worm      "$TEIS_WORM_PATH" \
  --promote-script /usr/local/bin/promote_on_allow.sh \
  --checkpoint /root/teis_checkpoints/final_20250921_174031.json \
  --model-dir /root/models \
  --symlink   /root/current_model \
  ${TEIS_DRY_RUN:+--dry-run} \
  | tee "$OUT_JSON"

# (opcional) compactar JSONs antigos para higiene
find "$LOG_DIR" -name 'canary_*.json' -mtime +7 -print0 | xargs -0 -r gzip -f