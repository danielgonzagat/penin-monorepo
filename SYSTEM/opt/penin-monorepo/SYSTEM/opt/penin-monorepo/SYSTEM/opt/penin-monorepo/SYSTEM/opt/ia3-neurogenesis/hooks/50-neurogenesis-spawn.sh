#!/usr/bin/env bash
set -euo pipefail
# Hook disparado ao final de cada rodada Darwin (ou quando promoção==allow)
# Usar variável $WORM_LINE se fornecida (linha JSON upstream)

ROOT="/opt/ia3-neurogenesis"

echo "🔗 Hook Darwin → IA³ (SPAWN) ativado"
echo "   WORM_LINE: ${WORM_LINE:-<vazio>}"

"$ROOT/bin/ia3-neurogenesis" --event cycle --worm-line "${WORM_LINE:-}"

echo "✅ Ciclo IA³ executado via hook Darwin"