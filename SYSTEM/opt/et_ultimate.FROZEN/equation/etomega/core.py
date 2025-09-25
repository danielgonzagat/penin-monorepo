import yaml, numpy as np
from .signals import expected_improvement as _expected_improvement, softmax_t as _softmax_t
from .validators import guardrails_check as _gr_check, entropy as _entropy

def _load_cfg(cfg_path="/opt/et_ultimate/equation/etomega.yaml"):
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def entropy(p): return _entropy(p)
def softmax_t(x, tau=0.7, axis=-1): return _softmax_t(x, tau=tau, axis=axis)

def expected_improvement(scores_new, scores_hist_mean, scores_hist_std, tau: float=0.7):
    return _expected_improvement(scores_new, scores_hist_mean, scores_hist_std, tau=tau)

def acceptance_score(P_hat, R, S, B, rho=1.0, sigma=1.0, iota=1.0):
    return float(P_hat - rho*R + sigma*S + iota*B)

def guardrails_check(metrics: dict, cfg_file="/opt/et_ultimate/equation/etomega.yaml"):
    cfg = _load_cfg(cfg_file)
    out = _gr_check(metrics, cfg)
    return {"passed": bool(out["passed"]), **out, "cfg": cfg}

def compute_ei_bundle(scores_new, scores_hist_mean, scores_hist_std, cfg_file="/opt/et_ultimate/equation/etomega.yaml"):
    cfg = _load_cfg(cfg_file)
    P_hat, ei_vec, weights = expected_improvement(scores_new, scores_hist_mean, scores_hist_std, tau=cfg.get("tau_ei", 0.7))
    return {"P_hat": float(P_hat), "ei": ei_vec.tolist(), "weights": weights.tolist(), "tau_ei": cfg.get("tau_ei", 0.7)}
