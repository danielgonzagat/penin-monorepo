#!/usr/bin/env bash
set -euo pipefail
# Hook disparado quando há rollback do modelo campeão→candidato

ROOT="/opt/ia3-neurogenesis"

echo "🔗 Hook Darwin → IA³ (ROLLBACK) ativado"

"$ROOT/bin/ia3-neurogenesis" --event rollback

echo "✅ Rollback IA³ executado via hook Darwin"