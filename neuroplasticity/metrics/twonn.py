"""
neuroplasticity.metrics.twonn
===============================
TwoNN intrinsic dimensionality estimator (Facco et al. 2017).

For a set of points X, compute:
    μᵢ = d(2nd-nearest-neighbour of i) / d(1st-nearest-neighbour of i)

The intrinsic dimensionality is then:
    ID = 1 / E[ ln μᵢ ]

This is a model-free estimator that requires no assumptions about the
data distribution beyond the existence of a smooth local manifold structure.

Application in this experiment
-------------------------------
twonn_id(X_train) → Data ID
    The intrinsic dimensionality of the raw feature space.
    Fixed across all epochs — the "complexity ceiling" the model must match.

rep_id(model, X_subset) → Rep ID
    The intrinsic dimensionality of the penultimate-layer activations.
    Rises as the model gains capacity to organise its representations
    along geometrically richer dimensions.

Complexity gap = Data ID − Rep ID
    The central neuroplasticity signal. A large gap means the model cannot
    yet represent the data's structural complexity. Each growth event
    produces a measurable reduction in this gap.

Memory note
-----------
cdist is O(n²) in memory. We subsample to n_sub=150 points by default,
keeping the distance matrix at ~180 KB (float64). Never pass large arrays.

Reference
---------
Facco, E. et al. (2017). Estimating the intrinsic dimension of datasets by
a minimal neighborhood information. Scientific Reports, 7, 12140.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

from neuroplasticity.models.growing_mlp import GrowingMLP


def twonn_id(X: np.ndarray, n_sub: int = 150) -> float:
    """
    Estimate intrinsic dimensionality of point cloud X.

    Parameters
    ----------
    X     : float array [N, D] — point cloud (raw features or activations)
    n_sub : subsample size for cdist (default 150 keeps memory at ~180 KB)

    Returns
    -------
    float — estimated intrinsic dimensionality, clipped to [1, D]

    Notes
    -----
    Always use a FIXED random subsample (same indices every epoch) when
    tracking Rep ID over time, otherwise epoch-to-epoch variance from
    different subsamples dominates the signal.
    """
    if X.shape[0] > n_sub:
        idx = np.random.choice(X.shape[0], n_sub, replace=False)
        X   = X[idx]

    X64 = X.astype(np.float64)
    D   = cdist(X64, X64)
    np.fill_diagonal(D, np.inf)

    ds  = np.sort(D, axis=1)
    mu  = (ds[:, 1] + 1e-12) / (ds[:, 0] + 1e-12)
    mu  = mu[mu > 1.0]

    if len(mu) < 5:
        return 1.0

    return float(np.clip(1.0 / np.mean(np.log(mu)), 1.0, X64.shape[1]))


def rep_id(model: GrowingMLP, X: np.ndarray, n_sub: int = 150) -> float:
    """
    Intrinsic dimensionality of `model`'s penultimate-layer activations.

    Parameters
    ----------
    model : GrowingMLP
    X     : float32 array [N, feat_dim] — input samples (fixed subset)
    n_sub : passed to twonn_id (default 150)

    Returns
    -------
    float — Rep ID
    """
    reps = model.represent(X, max_n=n_sub)
    return twonn_id(reps, n_sub=n_sub)
