"""
neuroplasticity.growth.operators
=================================
Net2Net-inspired function-preserving architecture expansion.

Two operations are supported:

grow_width(model, delta)
    Add `delta` neurons to every hidden layer.
    Existing weights are copied exactly (function-preserving for old neurons).
    New output neurons: He-init × 0.1  — start quiet, won't disrupt signal.
    New input connections from expanded previous layer: near-zero (1e-3 scale).
    Adam state transferred for old neurons — momentum continuity.

    Biological analogy: dendritic branching within an existing cortical layer.
    New synapses form weak and strengthen through long-term potentiation.

grow_depth(model)
    Insert one new hidden layer (half the width of the previous last hidden)
    before the output layer.
    New layer: near-identity (eye × 0.1) — passes signal cleanly at birth,
    then specialises under gradient pressure.
    Output layer: bias preserved, weights re-initialised for new fan-in.

    Biological analogy: formation of a new functional cortical area.

References
----------
Chen et al. (2016). Net2Net: Accelerating Learning via Knowledge Transfer.
"""

from __future__ import annotations

import numpy as np

from neuroplasticity.models.growing_mlp import GrowingMLP, Layer


# ── Adam state transfer helper ────────────────────────────────────────────────

def _transfer_adam(
    src:     Layer,
    dst:     Layer,
    old_in:  int,
    old_out: int,
) -> None:
    """
    Copy Adam moment buffers for the preserved (old_in × old_out) weight block
    from src into dst. Preserves momentum continuity for existing neurons.
    """
    dst.mW[:old_in, :old_out] = src.mW
    dst.vW[:old_in, :old_out] = src.vW
    dst.mb[:old_out]          = src.mb
    dst.vb[:old_out]          = src.vb
    dst.t                     = src.t


# ── Width growth ──────────────────────────────────────────────────────────────

def grow_width(model: GrowingMLP, delta: int = 12) -> GrowingMLP:
    """
    Add `delta` neurons to every hidden layer of `model`.

    Parameters
    ----------
    model : GrowingMLP to expand
    delta : number of neurons to add per hidden layer

    Returns
    -------
    new GrowingMLP with expanded architecture.
    The original model is not mutated.

    Weight copying scheme
    ---------------------
    Layer i has old weight matrix W[old_in × old_out]:

      dst.W[:old_in, :old_out]  ← src.W          (exact copy, function-preserving)
      dst.W[:old_in, old_out:]  ← He × 0.1       (new output neurons, start small)
      dst.W[old_in:, :old_out]  ← N(0, 1e-3)     (new inputs from expanded prev layer)
      dst.W[old_in:, old_out:]  ← N(0, 1e-3)     (new inputs to new output neurons)
    """
    old      = model.sizes
    new_sz   = [old[0]] + [s + delta for s in old[1:-1]] + [old[-1]]
    new_model = GrowingMLP(new_sz)

    for i, (src, dst) in enumerate(zip(model.layers, new_model.layers)):
        old_in  = old[i]
        old_out = old[i + 1]

        # ── Preserve existing computation ──────────────────────────────────
        dst.W[:old_in, :old_out] = src.W.copy()
        dst.b[:old_out]          = src.b.copy()

        # ── New output neurons (rows old_out:) ─────────────────────────────
        if new_sz[i + 1] > old_out:
            extra = new_sz[i + 1] - old_out
            std   = np.sqrt(2.0 / old_in) * 0.1   # He × 0.1: start quiet
            dst.W[:old_in, old_out:] = (
                np.random.randn(old_in, extra) * std
            ).astype(np.float32)

        # ── New input connections from expanded previous layer ──────────────
        if i > 0 and new_sz[i] > old_in:
            extra_in = new_sz[i] - old_in
            dst.W[old_in:, :old_out] = (
                np.random.randn(extra_in, old_out) * 1e-3
            ).astype(np.float32)
            if new_sz[i + 1] > old_out:
                dst.W[old_in:, old_out:] = (
                    np.random.randn(extra_in, new_sz[i + 1] - old_out) * 1e-3
                ).astype(np.float32)

        # ── Transfer Adam state for preserved neurons ──────────────────────
        _transfer_adam(src, dst, old_in, old_out)

    return new_model


# ── Depth growth ──────────────────────────────────────────────────────────────

def grow_depth(model: GrowingMLP) -> GrowingMLP:
    """
    Insert a new hidden layer before the output layer.

    The new layer has width = max(8, last_hidden_width // 2).
    It is initialised as a near-identity mapping (eye × 0.1) so it
    initially acts as a pass-through and then specialises under gradient
    pressure — analogous to a new cortical layer that relays signal at birth.

    Parameters
    ----------
    model : GrowingMLP to deepen

    Returns
    -------
    new GrowingMLP with one additional hidden layer.
    The original model is not mutated.
    """
    old    = model.sizes
    new_h  = max(8, old[-2] // 2)
    new_sz = old[:-1] + [new_h, old[-1]]
    new_model = GrowingMLP(new_sz)

    # ── Copy all layers except the output ──────────────────────────────────
    for i in range(len(model.layers) - 1):
        s, d = model.layers[i], new_model.layers[i]
        d.W[:] = s.W.copy()
        d.b[:] = s.b.copy()
        d.mW[:] = s.mW;  d.vW[:] = s.vW
        d.mb[:] = s.mb;  d.vb[:] = s.vb
        d.t     = s.t

    # ── New hidden layer: near-identity ────────────────────────────────────
    nd = new_model.layers[-2]
    mn = min(nd.W.shape)
    nd.W[:mn, :mn] = (np.eye(mn) * 0.1).astype(np.float32)
    # remainder of nd.W already near-zero from GrowingMLP.__init__

    # ── Output layer: preserve bias, re-init weights for new fan-in ────────
    new_model.layers[-1].b[:] = model.layers[-1].b.copy()
    new_model.layers[-1].W[:] = (
        np.random.randn(*new_model.layers[-1].W.shape) * 0.01
    ).astype(np.float32)

    return new_model
