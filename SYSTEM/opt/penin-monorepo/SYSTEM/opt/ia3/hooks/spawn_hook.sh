#!/bin/bash
# /opt/ia3/hooks/spawn_hook.sh
# Hook executado quando Darwin decide que um agente deve nascer

set -euo pipefail

WORM_LOG="/opt/ia3/var/worm/ia3_worm.log"
TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)

# Log do evento de spawn no WORM
echo "{\"event\":\"darwin_spawn_trigger\",\"timestamp\":\"$TIMESTAMP\",\"source\":\"darwin\",\"action\":\"requested_birth\"}" >> "$WORM_LOG"

# O sistema IA³ detectará este evento e criará um neurônio
echo "✅ Spawn requisitado para o sistema IA³"