import json, time
from pathlib import Path

path = Path("/opt/et_ultimate/actions/results.jsonl")
if path.exists():
    lines = [json.loads(l) for l in open(path) if "score" in l]
    best = max(lines, key=lambda x: x["score"])
    print(f"🏆 Melhor Equação: {best.get('output', '')[:300]}")
    with open("/opt/et_ultimate/workspace/BEST_ETΩ.txt", "w") as f:
        f.write(best.get("output", ""))
