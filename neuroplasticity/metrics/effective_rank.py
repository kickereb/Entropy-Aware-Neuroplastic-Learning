"""Effective Rank — SVD spectral entropy of weight matrices."""
import numpy as np

def effective_rank(model) -> float:
    """Mean effective rank across all weight matrices."""
    ranks = []
    for W in model.weight_matrices():
        s = np.linalg.svd(W, compute_uv=False)
        s = s[s > 1e-10]
        if len(s) == 0:
            ranks.append(1.0); continue
        p = s / s.sum()
        ranks.append(float(np.exp(-np.sum(p * np.log(p + 1e-12)))))
    return float(np.mean(ranks))
