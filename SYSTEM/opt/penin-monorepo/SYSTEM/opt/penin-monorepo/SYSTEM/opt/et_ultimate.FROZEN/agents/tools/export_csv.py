import csv, json
lines = [json.loads(l) for l in open("/opt/et_ultimate/actions/results.jsonl") if "score" in l]
with open("/opt/et_ultimate/workspace/et_history.csv", "w") as f:
    writer = csv.DictWriter(f, fieldnames=["ts","score","output"])
    writer.writeheader()
    for l in lines:
        writer.writerow({"ts": l["ts"], "score": l["score"], "output": l["output"][:300]})
