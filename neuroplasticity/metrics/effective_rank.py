"""
neuroplasticity.metrics.effective_rank
========================================
Effective rank of a weight matrix (Roy & Vetterli 2007).

    eff_rank(W) = exp( H(σ̃) )

where H(σ̃) = −Σ σ̃ᵢ log σ̃ᵢ is the Shannon entropy of the normalised
singular values σ̃ᵢ = σᵢ / Σσⱼ.

Range: [1, min(rows, cols)]
  1         → all singular values equal (rank-1 in information terms)
  min(r, c) → all singular values equal (fully diverse, maximally used)

Interpretation
--------------
Low effective rank: most neurons compute nearly the same features.
  The layer is redundant — wasted capacity.
High effective rank: each neuron contributes an independent feature direction.
  The layer efficiently uses all of its representational degrees of freedom.

Growth signature: sharp upward jumps at each growth event as new neurons
introduce fresh, uncorrelated feature directions before they begin to
converge with the existing learned representations.

Reference
---------
Roy, O. & Vetterli, M. (2007). The effective rank: A measure of effective
dimensionality. EUSIPCO.
"""

from __future__ import annotations

import numpy as np

from neuroplasticity.models.growing_mlp import GrowingMLP


def effective_rank(W: np.ndarray) -> float:
    """
    Compute the effective rank of weight matrix W.

    Parameters
    ----------
    W : 2-D array of any dtype (converted to float64 internally for SVD)

    Returns
    -------
    float in [1, min(W.shape)]
    """
    W64  = W.astype(np.float64)
    _, sv, _ = np.linalg.svd(W64, full_matrices=False)
    sv = sv[sv > 1e-9]
    if len(sv) == 0:
        return 1.0
    p = sv / sv.sum()
    return float(np.exp(-np.sum(p * np.log(p + 1e-12))))


def mean_eff_rank(model: GrowingMLP) -> float:
    """
    Mean effective rank averaged over all weight matrices in `model`.

    Parameters
    ----------
    model : GrowingMLP instance

    Returns
    -------
    float — mean eff_rank across all layers
    """
    return float(np.mean([effective_rank(L.W) for L in model.layers]))
