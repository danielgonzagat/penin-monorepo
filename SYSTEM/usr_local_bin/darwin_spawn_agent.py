#!/usr/bin/env python3
import json, sys, os, time
from datetime import datetime
def now(): return datetime.utcnow().isoformat() + "Z"
args = sys.argv[1:]
manifest = "/root/agents_active.json"; worm="/root/darwin_worm.log"; note="manual_spawn"
for i,a in enumerate(args):
    if a == "--manifest" and i+1 < len(args): manifest = args[i+1]
    if a == "--worm" and i+1 < len(args): worm = args[i+1]
    if a == "--note" and i+1 < len(args): note = args[i+1]
data = json.load(open(manifest)) if os.path.exists(manifest) else {"agents":[],"next_id":0,"death_counter":0}
aid = f"agent_{data['next_id']}"; data["next_id"] += 1
agent = {"id": aid, "born_ts": time.time(), "meta": {"origin": "manual_spawn", "note": note}}
data["agents"].append(agent)
json.dump(data, open(manifest,"w"), indent=2)
evt = {"ts": now(), "event": "darwin_spawn_manual", "agent": aid, "note": note}
open(worm,"a").write("EVENT:"+json.dumps(evt)+"\n")
open(worm,"a").write("HASH:manual\n")
print(aid)
