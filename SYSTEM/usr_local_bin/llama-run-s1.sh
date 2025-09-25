#!/usr/bin/env bash
set -euo pipefail
BIN="/root/llama.cpp/build/bin/llama-server"
MODEL="/root/models/qwen2.5-7b-instruct-gguf/Qwen2.5-7B-Instruct-Q5_K_M.gguf"
THREADS="24"
CTX="8192"
PORT="8091"
API_KEY="DANIEL"
ulimit -n 65535 || true
exec numactl --cpunodebind=1 --membind=1 \
  "$BIN" -m "$MODEL" -t "$THREADS" -c "$CTX" \
  --host 127.0.0.1 --port "$PORT" 
