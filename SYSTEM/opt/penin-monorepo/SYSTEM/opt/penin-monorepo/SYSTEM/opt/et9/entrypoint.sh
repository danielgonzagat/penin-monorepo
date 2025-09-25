#!/usr/bin/env bash
set -euo pipefail
redis-server --daemonize yes
ray start --head --dashboard-host=0.0.0.0 --dashboard-port=8266 >/tmp/ray.log 2>&1 || true
exec python /app/et_core.py
