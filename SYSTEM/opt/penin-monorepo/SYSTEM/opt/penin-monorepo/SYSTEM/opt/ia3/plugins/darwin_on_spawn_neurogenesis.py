#!/usr/bin/env python3
"""
Hook de spawn Darwin → Neurogênese
Quando Darwin decide um nascimento, este hook cria 1 neurônio real
"""
import os, json, subprocess, sys
from pathlib import Path
from datetime import datetime

IA3_HOME = Path(os.getenv("IA3_HOME","/opt/ia3"))
NEUROGEN = IA3_HOME / "bin" / "neurongen.py"
LOGS_DIR = IA3_HOME / "logs"
WORM_LOG = LOGS_DIR / "worm.log"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

def append_worm(event):
    event["timestamp"] = datetime.utcnow().isoformat() + "Z"
    with open(WORM_LOG, "a") as f:
        f.write("EVENT:" + json.dumps(event, ensure_ascii=False) + "\n")

def main():
    # O darwinctl passa caminho do JSON do "newborn"
    newborn_json = sys.argv[1] if len(sys.argv) > 1 else None
    agent_id = sys.argv[2] if len(sys.argv) > 2 else f"neuron_{int(datetime.utcnow().timestamp())}"

    # Log início
    append_worm({
        "event": "spawn_hook_start",
        "newborn_json": newborn_json,
        "agent_id": agent_id
    })

    # Adiciona 1 neurônio e treina
    cmd = [str(NEUROGEN), "--add-one-neuron", "--steps", "50", "--lr", "0.001"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    
    if r.returncode == 0:
        print(r.stdout.strip())
        # Registrar sucesso
        append_worm({
            "event": "spawn_hook_success",
            "agent_id": agent_id,
            "output": r.stdout.strip()
        })
    else:
        print(f"❌ Erro na neurogênese: {r.stderr}", file=sys.stderr)
        append_worm({
            "event": "spawn_hook_error",
            "agent_id": agent_id,
            "error": r.stderr
        })
        sys.exit(1)

if __name__ == "__main__":
    main()