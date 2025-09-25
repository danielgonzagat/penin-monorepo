#!/usr/bin/env python3
import json, os, functools, time, datetime, csv
import requests

BAKED = "/opt/et/knowledge_baked.json"
CSV   = "/var/log/et/telemetry.csv"
JSONL = "/var/log/et/telemetry.jsonl"
DEFAULT_FALLBACK = "Missão: evoluir continuamente (ET★). Use score=P-RHO*R+SIGMA*S+IOTA*B; canários como gate; guardrails e rollback."

def _load_baked():
    try:
        with open(BAKED, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"full": DEFAULT_FALLBACK, "short":{"core":DEFAULT_FALLBACK}}

BAKED_KB = _load_baked()

def _select_prompt(context_kind: str | None):
    short = BAKED_KB.get("short", {})
    if not context_kind:
        return short.get("core") or BAKED_KB.get("full", DEFAULT_FALLBACK)
    ck = context_kind.lower()
    if "score" in ck:      return short.get("score")     or BAKED_KB.get("full", DEFAULT_FALLBACK)
    if "canar" in ck:      return short.get("canarios")  or BAKED_KB.get("full", DEFAULT_FALLBACK)
    if "guard" in ck or "segur" in ck:
                            return short.get("guardrails")or BAKED_KB.get("full", DEFAULT_FALLBACK)
    return short.get("core") or BAKED_KB.get("full", DEFAULT_FALLBACK)

def _append_csv(row: dict):
    exists = os.path.exists(CSV) and os.path.getsize(CSV) > 0
    with open(CSV, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ts","model","temp","max_tokens","lat_ms","tokens","tps","status"])
        if not exists:
            w.writeheader()
        w.writerow(row)

def _append_jsonl(obj: dict):
    with open(JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def install_requests_hook(context_kind: str | None = None):
    kb_text = _select_prompt(context_kind)
    orig_request = requests.sessions.Session.request

    @functools.wraps(orig_request)
    def wrapped(self, method, url, **kwargs):
        js = kwargs.get("json")
        temp = None
        max_tokens = None
        model = None
        try:
            if isinstance(js, dict):
                model = js.get("model")
                temp = js.get("temperature")
                max_tokens = js.get("max_tokens")
                if "messages" in js and isinstance(js["messages"], list):
                    msgs = js["messages"]
                    if not (len(msgs) and isinstance(msgs[0], dict) and msgs[0].get("role") == "system"):
                        sys_msg = {"role":"system","content": kb_text}
                        js = dict(js)
                        js["messages"] = [sys_msg] + msgs
                        kwargs["json"] = js
        except Exception:
            pass

        t0 = time.time()
        resp = orig_request(self, method, url, **kwargs)
        dt_ms = (time.time() - t0) * 1000.0

        tokens = None
        tps = None
        status = resp.status_code

        try:
            data = resp.json()
            # OpenAI-like usage
            usage = data.get("usage") or {}
            completion = usage.get("completion_tokens")
            prompt = usage.get("prompt_tokens")
            total = usage.get("total_tokens")
            tokens = completion or total
            # llama.cpp às vezes retorna "timings"
            timings = None
            if "timings" in data:
                timings = data["timings"]
                pred_n = timings.get("predicted_n")
                pred_ms = timings.get("predicted_ms")
                if (not tokens) and pred_n:
                    tokens = pred_n
                if pred_n and pred_ms and pred_ms > 0:
                    tps = (pred_n * 1000.0) / float(pred_ms)
            if (not tps) and tokens and dt_ms > 0:
                tps = (float(tokens) * 1000.0) / dt_ms
        except Exception:
            pass

        ts = datetime.datetime.utcnow().isoformat() + "Z"
        row = {
            "ts": ts, "model": model, "temp": temp, "max_tokens": max_tokens,
            "lat_ms": round(dt_ms,2), "tokens": tokens, "tps": (round(tps,2) if tps else None),
            "status": status
        }
        try:
            _append_csv(row)
            _append_jsonl({"meta": row})
        except Exception:
            pass

        return resp

    requests.sessions.Session.request = wrapped
