import numpy as np
from scipy.stats import norm
def expected_improvement(x, x_samples, y_samples, acq_func='EI', kappa=1.96):
    if acq_func == 'EI':
        mean, std = model.predict(x)
        z = (mean - np.max(y_samples) - kappa) / std
        ei = (mean - np.max(y_samples)) * norm.cdf(z) + std * norm.pdf(z)
        return ei