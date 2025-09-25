#!/usr/bin/env bash
set -euo pipefail

MODEL="/root/models/qwen2.5-7b-instruct-gguf/Qwen2.5-7B-Instruct-Q5_K_M.gguf"
THREADS="$(nproc)"
CTX=8192
PORT=8080
API_KEY="DANIEL"

# Evita oversubscription quando usar BLAS/OpenMP
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS="${THREADS}"
export GGML_N_THREADS="${THREADS}"
export GGML_NUMA=1

ulimit -n 65535 || true

CMD=(/root/llama.cpp/build/bin/llama-server
  -m "$MODEL" -t "$THREADS" -c "$CTX"
  --host 0.0.0.0 --port "$PORT" --api-key "$API_KEY"
)

# Se numactl existir, interleave entre nós de memória (melhora latência)
if command -v numactl >/dev/null 2>&1; then
  exec numactl --interleave=all "${CMD[@]}"
else
  exec "${CMD[@]}"
fi
