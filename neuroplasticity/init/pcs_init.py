"""PCsInit: Initialize first layer with top PCs of training data (arXiv 2501.19114)."""
import numpy as np

def pcs_init(X_train, width):
    """Returns a [feat_dim, width] weight matrix from top PCs of X_train."""
    Xc = X_train - X_train.mean(0)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    n_pcs = min(width, Vt.shape[0])
    W = np.zeros((X_train.shape[1], width), np.float32)
    for i in range(n_pcs):
        W[:, i] = Vt[i] * (S[i] / (S[0] + 1e-8)) * np.sqrt(2.0 / X_train.shape[1])
    if width > n_pcs:
        W[:, n_pcs:] = np.random.randn(X_train.shape[1], width - n_pcs).astype(np.float32) * 0.01
    return W
