"""
neuroplasticity.metrics.fisher
================================
Fisher Information Matrix (FIM) trace estimator.

    FIM_trace ≈ Σ_θ  E_x[ (∂ log p(y|x;θ) / ∂θ)² ]

Approximated by the sum of squared gradients over a single fixed mini-batch
(empirical Fisher, not the full expectation).

Interpretation
--------------
High trace → every parameter is actively shaping the loss.
             The model uses its full capacity efficiently.
Low trace  → many parameters are idle; wasted capacity.

Neuroplasticity signature
-------------------------
The Fisher trace displays a characteristic dip-then-rise pattern at each
growth event:

  1. Before growth: trace rises steeply as all neurons push against the
     capacity ceiling (every parameter straining).

  2. Immediately after growth: trace drops sharply. New neurons have
     near-zero weights → near-zero gradients → they contribute minimally
     to the total gradient magnitude. The average parameter sensitivity falls.

  3. Over the following epochs: new neurons specialise, their gradients grow,
     and the trace recovers and surpasses its previous peak.

This dip-then-rise is the Fisher Information signature of long-term
potentiation: synapses that begin weak and strengthen through use.
"""

from __future__ import annotations

import numpy as np

from neuroplasticity.models.growing_mlp import GrowingMLP


def fisher_trace(
    model:   GrowingMLP,
    X_batch: np.ndarray,
    y_batch: np.ndarray,
) -> float:
    """
    Estimate the FIM trace via one forward-backward pass on a fixed batch.

    Parameters
    ----------
    model   : GrowingMLP
    X_batch : float32 array [N, feat_dim]  — fixed analysis subset
    y_batch : int32   array [N]

    Returns
    -------
    float — FIM trace (sum of squared gradient components)

    Notes
    -----
    The batch should be a FIXED subset (same samples every epoch) so that
    the metric is comparable across epochs without batch-variance noise.
    """
    acts = model._forward(X_batch)
    prob = acts[-1]
    n    = len(y_batch)

    # Combined softmax + cross-entropy gradient
    g = prob.copy()
    g[np.arange(n), y_batch] -= 1
    g /= n

    trace = 0.0
    for i in reversed(range(len(model.layers))):
        L  = model.layers[i]
        gW = acts[i].T @ g           # [in_dim, out_dim]
        gb = g.sum(axis=0)           # [out_dim]
        trace += float(np.sum(gW * gW) + np.sum(gb * gb))
        if i > 0:
            g = ((g @ L.W.T) * (acts[i] > 0)).astype(np.float32)

    return trace
