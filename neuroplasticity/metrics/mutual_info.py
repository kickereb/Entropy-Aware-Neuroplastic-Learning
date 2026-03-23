"""
neuroplasticity.metrics.mutual_info
=====================================
Mutual Information I(T ; Y) estimator for the Information Bottleneck.

Estimates I(T ; Y) = H(T) − H(T | Y) by:
  1. Computing the penultimate-layer activations T for a fixed subset
  2. Projecting T to 1D via its first principal component
  3. Binning the 1D projection and computing H(T) and H(T|Y) from
     empirical frequency tables

Interpretation
--------------
Under the Information Bottleneck principle (Tishby & Schwartz-Ziv 2017),
a good representation maximises I(T ; Y) (label compression) while
compressing I(X ; T) (input compression). We track only I(T ; Y) here.

  Near-zero I(T ; Y):
    Representations carry no class-discriminative structure.
    The network is fitting noise or has collapsed. Definitive
    underfitting signature.

  Rising I(T ; Y):
    Representations are becoming increasingly class-discriminative.
    Capacity is being converted into useful computation.

Limitations
-----------
The 1D-projection binning estimator underestimates the true I(T ; Y) in
high-dimensional representation spaces. It provides a consistent relative
indicator (rising = better) but not an absolute value. For precise
estimates, use a matrix-based estimator (e.g. HSIC or KSG).

Reference
---------
Tishby, N. & Schwartz-Ziv, R. (2017). Opening the Black Box of Deep
Neural Networks via Information. ITW.
"""

from __future__ import annotations

import numpy as np

from neuroplasticity.models.growing_mlp import GrowingMLP


def mutual_info_ib(
    model:  GrowingMLP,
    X:      np.ndarray,
    y:      np.ndarray,
    n_bins: int = 28,
) -> float:
    """
    Estimate I(T ; Y) for the penultimate activations T of `model`.

    Parameters
    ----------
    model  : GrowingMLP
    X      : float32 array [N, feat_dim]  — fixed analysis subset
    y      : int32   array [N]            — labels for the subset
    n_bins : number of histogram bins for 1D projection (default 28)

    Returns
    -------
    float — estimated I(T ; Y) in nats, clipped to [0, ∞)
    """
    n_classes = int(y.max()) + 1
    reps  = model.represent(X, max_n=len(X))      # [N, hidden_dim], float32
    rc    = (reps - reps.mean(0)).astype(np.float64)

    # Project to 1D via first principal component (cheapest MI estimator)
    try:
        _, _, Vt = np.linalg.svd(rc, full_matrices=False)
        t = (rc @ Vt[0]).astype(np.float32)
    except np.linalg.LinAlgError:
        t = rc[:, 0].astype(np.float32)

    # Bin the 1D projection
    lo, hi = float(t.min()) - 1e-6, float(t.max()) + 1e-6
    edges  = np.linspace(lo, hi, n_bins + 1)
    T_bin  = np.clip(np.digitize(t, edges) - 1, 0, n_bins - 1)

    # H(T)
    pT  = np.bincount(T_bin, minlength=n_bins).astype(np.float64) / len(y)
    H_T = -np.sum(pT * np.log(pT + 1e-12))

    # H(T | Y) = Σ_c p(c) H(T | Y = c)
    H_TY = 0.0
    for c in range(n_classes):
        mask = (y == c)
        if not mask.any():
            continue
        p_c  = mask.mean()
        pTc  = np.bincount(T_bin[mask], minlength=n_bins).astype(np.float64)
        pTc /= pTc.sum() + 1e-12
        H_TY += p_c * (-np.sum(pTc * np.log(pTc + 1e-12)))

    return float(max(0.0, H_T - H_TY))
