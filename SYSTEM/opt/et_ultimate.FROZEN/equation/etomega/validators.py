import numpy as np

def entropy(p):
    p = np.asarray(p, dtype=float)
    p = p / np.clip(p.sum(), 1e-12, None)
    with np.errstate(divide='ignore', invalid='ignore'):
        h = -(p * np.log2(np.clip(p, 1e-12, 1.0))).sum()
    return float(h)

def kl_divergence(p, q):
    p = np.asarray(p, dtype=float); p /= np.clip(p.sum(), 1e-12, None)
    q = np.asarray(q, dtype=float); q /= np.clip(q.sum(), 1e-12, None)
    with np.errstate(divide='ignore', invalid='ignore'):
        return float((p * (np.log(np.clip(p,1e-12,1.0)) - np.log(np.clip(q,1e-12,1.0)))).sum())

def guardrails_check(metrics: dict, cfg: dict) -> dict:
    out = {"passed": True, "violations": []}

    # Entropia mínima (π_new)
    if "policy_new" in metrics:
        h = entropy(metrics["policy_new"])
        out["entropy"] = h
        if h < cfg.get("entropy_min", 0.5):
            out["passed"] = False; out["violations"].append("entropy_min")

    # Divergência limitada KL(π_new || π_prev)
    if "policy_prev" in metrics and "policy_new" in metrics:
        dkl = kl_divergence(metrics["policy_new"], metrics["policy_prev"])
        out["kl"] = dkl
        if dkl > cfg.get("kl_max", 0.075):
            out["passed"] = False; out["violations"].append("kl_max")

    # Drift controlado (queda em tarefas já dominadas)
    if "dominated_drop" in metrics:
        drift = float(metrics["dominated_drop"])
        out["drift"] = drift
        if drift > cfg.get("drift_max", 0.02):
            out["passed"] = False; out["violations"].append("drift_max")

    # Orçamento de custo relativo
    if "cost_ratio" in metrics:
        cr = float(metrics["cost_ratio"])
        out["cost_ratio"] = cr
        if cr > cfg.get("cost_max", 1.0):
            out["passed"] = False; out["violations"].append("cost_max")

    # Variância mínima do currículo
    if "curriculum_beta" in metrics:
        v = float(np.var(np.asarray(metrics["curriculum_beta"], dtype=float)))
        out["var_beta"] = v
        if v < cfg.get("var_min", 0.15):
            out["passed"] = False; out["violations"].append("var_min")

    return out
