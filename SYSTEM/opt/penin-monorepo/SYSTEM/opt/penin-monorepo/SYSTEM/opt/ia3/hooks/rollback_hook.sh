#!/bin/bash
# /opt/ia3/hooks/rollback_hook.sh
# Hook executado quando Darwin decide fazer rollback

set -euo pipefail

WORM_LOG="/opt/ia3/var/worm/ia3_worm.log"
TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)

# Log do evento de rollback no WORM
echo "{\"event\":\"darwin_rollback_trigger\",\"timestamp\":\"$TIMESTAMP\",\"source\":\"darwin\",\"action\":\"requested_death\"}" >> "$WORM_LOG"

# O sistema IA³ detectará este evento e poderá matar neurônios
echo "✅ Rollback requisitado para o sistema IA³"