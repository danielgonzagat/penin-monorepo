#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=/opt/et_ultimate
exec /opt/et_ultimate/venv/bin/python3 -m agents.brain.et_brain_operacional "$@"
