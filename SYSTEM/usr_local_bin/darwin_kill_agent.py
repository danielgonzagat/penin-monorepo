#!/usr/bin/env python3
import json, sys, os, time
from datetime import datetime
def now(): return datetime.utcnow().isoformat() + "Z"
def usage():
    print("uso: darwin_kill_agent.py --agent AGENT_ID --manifest /root/agents_active.json --worm /root/darwin_worm.log")
    sys.exit(2)
args = sys.argv[1:]
agent_id = None; manifest = "/root/agents_active.json"; worm="/root/darwin_worm.log"
for i,a in enumerate(args):
    if a == "--agent" and i+1 < len(args): agent_id = args[i+1]
    if a == "--manifest" and i+1 < len(args): manifest = args[i+1]
    if a == "--worm" and i+1 < len(args): worm = args[i+1]
if not agent_id: usage()
data = json.load(open(manifest)) if os.path.exists(manifest) else {"agents":[],"next_id":0,"death_counter":0}
before = len(data["agents"])
data["agents"] = [a for a in data["agents"] if a.get("id") != agent_id]
after = len(data["agents"])
json.dump(data, open(manifest,"w"), indent=2)
evt = {"ts": now(), "event": "darwin_kill_manual", "agent": agent_id, "removed": (before-after)}
open(worm,"a").write("EVENT:"+json.dumps(evt)+"\n")
open(worm,"a").write("HASH:manual\n")
print(f"Killed {agent_id} (removed {before-after})")
