"""Mutual Information I(T;Y) via binning first PC of penultimate activations."""
import numpy as np

def mutual_info_ib(model, X, y, n_bins=28) -> float:
    X, y = X[:2000], y[:2000]
    reps = model.represent(X, max_n=2000)
    rc = (reps - reps.mean(0)).astype(np.float64)
    try:
        _, _, Vt = np.linalg.svd(rc, full_matrices=False)
        t = (rc @ Vt[0]).astype(np.float32)
    except Exception:
        t = rc[:, 0].astype(np.float32)
    lo, hi = float(t.min()) - 1e-6, float(t.max()) + 1e-6
    edges = np.linspace(lo, hi, n_bins + 1)
    Tb = np.clip(np.digitize(t, edges) - 1, 0, n_bins - 1)
    n_classes = int(y.max()) + 1
    pT = np.bincount(Tb, minlength=n_bins).astype(np.float64) / len(y)
    HT = -np.sum(pT * np.log(pT + 1e-12))
    HTY = 0.0
    for c in range(n_classes):
        m = (y == c)
        if not m.any(): continue
        pc = m.mean()
        pTc = np.bincount(Tb[m], minlength=n_bins).astype(np.float64)
        pTc /= pTc.sum() + 1e-12
        HTY += pc * (-np.sum(pTc * np.log(pTc + 1e-12)))
    return float(max(0, HT - HTY))
