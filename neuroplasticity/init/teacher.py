"""
neuroplasticity.init.teacher
==============================
Train a small reference model with regularisation, then project its
learned weights via SVD into a larger architecture.
"""
from __future__ import annotations
import numpy as np


def _relu(x):  return np.maximum(0, x)
def _drelu(x): return (x > 0).astype(np.float32)
def _softmax(z):
    e = np.exp(z - z.max(1, keepdims=True))
    return e / (e.sum(1, keepdims=True) + 1e-12)


def train_reference_model(
    X_tr, y_tr, X_te, y_te,
    width: int = 48,
    epochs: int = 50,
    lr: float = 0.005,
    wd: float = 3e-4,
    batch_size: int = 256,
    verbose: bool = True,
) -> dict:
    """
    Train a small 2-hidden-layer MLP with weight decay.

    Returns dict with keys W1, b1, W2, b2, W3, b3.
    """
    d_in = X_tr.shape[1]
    n_classes = int(y_tr.max()) + 1
    sc = lambda f: np.sqrt(2.0 / f)

    W1 = np.random.randn(d_in, width).astype(np.float32) * sc(d_in)
    b1 = np.zeros(width, np.float32)
    W2 = np.random.randn(width, width).astype(np.float32) * sc(width)
    b2 = np.zeros(width, np.float32)
    W3 = np.random.randn(width, n_classes).astype(np.float32) * sc(width) * 0.5
    b3 = np.zeros(n_classes, np.float32)

    for ep in range(epochs):
        idx = np.random.permutation(len(X_tr))
        for s in range(0, len(idx), batch_size):
            bi = idx[s:s + batch_size]
            x, y = X_tr[bi], y_tr[bi]
            n = len(y)

            z1 = x @ W1 + b1;  h1 = _relu(z1)
            z2 = h1 @ W2 + b2; h2 = _relu(z2)
            lo = h2 @ W3 + b3;  pr = _softmax(lo)

            dl = pr.copy(); dl[np.arange(n), y] -= 1.0; dl /= n

            gW3 = h2.T @ dl + wd * W3; gb3 = dl.sum(0)
            dh2 = dl @ W3.T * _drelu(z2)
            gW2 = h1.T @ dh2 + wd * W2; gb2 = dh2.sum(0)
            dh1 = dh2 @ W2.T * _drelu(z1)
            gW1 = x.T @ dh1 + wd * W1; gb1 = dh1.sum(0)

            W1 -= lr * gW1; b1 -= lr * gb1
            W2 -= lr * gW2; b2 -= lr * gb2
            W3 -= lr * gW3; b3 -= lr * gb3

        if verbose and (ep + 1) % 10 == 0:
            te_pred = _softmax(_relu(_relu(X_te @ W1 + b1) @ W2 + b2) @ W3 + b3).argmax(1)
            tr_pred = _softmax(_relu(_relu(X_tr @ W1 + b1) @ W2 + b2) @ W3 + b3).argmax(1)
            print(f"    RefModel ep {ep+1}: train={float((tr_pred==y_tr).mean()):.3f} "
                  f"test={float((te_pred==y_te).mean()):.3f}")

    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3': W3, 'b3': b3}


def project_weights_svd(W_small, target_rows, target_cols):
    """
    Project a small weight matrix into a larger one using SVD.
    Preserves the learned representation's principal directions.
    """
    U, S, Vt = np.linalg.svd(W_small, full_matrices=False)
    r, c = W_small.shape
    rank = min(len(S), r, c, target_rows, target_cols)

    W_large = np.zeros((target_rows, target_cols), dtype=np.float32)
    min_r = min(r, target_rows)
    min_c = min(c, target_cols)
    k = min(rank, min_r, min_c)

    W_large[:min_r, :min_c] = (
        U[:min_r, :k] @ np.diag(S[:k]).astype(np.float32) @ Vt[:k, :min_c]
    )
    return W_large
