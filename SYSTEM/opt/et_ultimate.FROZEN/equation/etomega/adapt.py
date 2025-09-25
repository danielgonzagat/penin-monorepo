import yaml, json, math
from pathlib import Path

CFG = Path("/opt/et_ultimate/equation/etomega.yaml")
STATE_DIR = Path("/opt/et_ultimate/state"); STATE_DIR.mkdir(parents=True, exist_ok=True)

def _load_cfg():
    return yaml.safe_load(CFG.read_text(encoding="utf-8"))

def _dump_cfg(cfg):
    CFG.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")

def adapt_cfg(metrics: dict, cfg_file: str = str(CFG)):
    # Objetivo: manter H >= entropy_min+margin, sem relaxar guardrails duros
    cfg = _load_cfg()
    h_min = float(cfg.get("entropy_min", 0.5))
    tau   = float(cfg.get("tau_ei", 0.7))
    margin_low, margin_high = 0.05, 0.30

    h = None
    try:
        h = float(metrics.get("omega", {}).get("entropy", None))
    except Exception:
        pass

    changed = False
    if h is not None:
        if h < h_min + margin_low:
            tau = min(1.4, tau + 0.05)  # ↑exploração
            changed = True
        elif h > h_min + margin_high:
            tau = max(0.4, tau - 0.05)  # ↓exploração
            changed = True

    if changed:
        cfg["tau_ei"] = float(f"{tau:.3f}")
        _dump_cfg(cfg)
    return {"tau_ei": tau, "changed": changed}
