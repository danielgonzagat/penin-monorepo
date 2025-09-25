import numpy as np

def softmax_t(x, tau: float=0.7, axis=-1):
    x = np.asarray(x, dtype=float)
    x = x / max(tau, 1e-8)
    x -= x.max(axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.clip(ex.sum(axis=axis, keepdims=True), 1e-12, None)

def expected_improvement(scores_new, scores_hist_mean, scores_hist_std, tau: float=0.7):
    s_new = np.asarray(scores_new, dtype=float)
    mu = np.asarray(scores_hist_mean, dtype=float)
    sd = np.asarray(scores_hist_std, dtype=float)
    z = (s_new - mu) / (np.abs(sd) + 1e-8)
    ei = np.maximum(0.0, z)
    w = softmax_t(ei, tau=tau)
    return float((w * ei).sum()), ei, w
