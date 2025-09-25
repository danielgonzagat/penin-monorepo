#!/usr/bin/env bash
set -euo pipefail
echo "[post_spawn] id=$NEURON_ID reason=$REASON ts=$TIMESTAMP" >> /root/ia3_darwin_hooks.log
echo "✅ Neurônio $NEURON_ID logado com sucesso"
