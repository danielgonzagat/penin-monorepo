import os, json, hashlib, time

def _last_hash(path: str):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            lines = f.readlines()
            if not lines:
                return None
            last = lines[-1]
        rec = json.loads(last.decode("utf-8"))
        return rec.get("hash")
    except Exception:
        return None

def worm_append_event(path: str, event: str, payload: dict):
    prev = _last_hash(path)
    rec = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "event": event,
        "payload": payload,
        "prev": prev or "GENESIS"
    }
    raw = json.dumps(rec, sort_keys=True).encode("utf-8")
    h = hashlib.sha256(raw).hexdigest()
    rec["hash"] = h
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    with open(path, "ab") as f:
        f.write((json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8"))