"""TwoNN Intrinsic Dimensionality estimator (Facco et al. 2017)."""
import numpy as np
from scipy.spatial.distance import cdist

def twonn_id(X, max_n=2000) -> float:
    if len(X) > max_n:
        X = X[np.random.choice(len(X), max_n, replace=False)]
    D = cdist(X, X)
    np.fill_diagonal(D, np.inf)
    n1 = np.partition(D, 1, axis=1)[:, 1]
    n2 = np.partition(D, 2, axis=1)[:, 2]
    mu = np.sort(n2 / (n1 + 1e-12))
    mu = mu[mu > 1 + 1e-8]
    if len(mu) < 10: return 1.0
    F = np.arange(1, len(mu) + 1) / len(mu)
    v = F < 0.9
    if v.sum() < 5: return 1.0
    return max(1.0, float(-np.polyfit(np.log(mu[v]), np.log(1 - F[v] + 1e-12), 1)[0]))
