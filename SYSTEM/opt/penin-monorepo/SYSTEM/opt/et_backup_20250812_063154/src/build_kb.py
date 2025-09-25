#!/usr/bin/env python3
import os, json, glob
SRC="/opt/et/knowledge_base.d"
OUT="/opt/et/knowledge_baked.json"
parts=[]
for p in sorted(glob.glob(os.path.join(SRC,"*.md"))):
    parts.append(open(p,"r",encoding="utf-8").read().strip())
full="\n\n".join(parts)
data={"full":full, "short":{"core":full}}
open(OUT,"w",encoding="utf-8").write(json.dumps(data,ensure_ascii=False))
print("KB refeito:", OUT, "len=", len(full))
