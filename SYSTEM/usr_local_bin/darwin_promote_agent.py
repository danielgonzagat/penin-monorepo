#!/usr/bin/env python3
import json, sys, os
from datetime import datetime
def now(): return datetime.utcnow().isoformat() + "Z"
args = sys.argv[1:]
agent = None; worm="/root/darwin_worm.log"
for i,a in enumerate(args):
    if a == "--agent" and i+1 < len(args): agent = args[i+1]
    if a == "--worm" and i+1 < len(args): worm = args[i+1]
if not agent:
    print("uso: darwin_promote_agent.py --agent AGENT_ID [--worm /root/darwin_worm.log]")
    sys.exit(2)
evt = {"ts": now(), "event": "darwin_promote_manual", "agent": agent}
open(worm,"a").write("EVENT:"+json.dumps(evt)+"\n")
open(worm,"a").write("HASH:manual\n")
print(f"Promoted {agent}")
