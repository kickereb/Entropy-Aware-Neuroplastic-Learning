"""
neuroplasticity.growth.operators
==================================
Net2Net-style function-preserving expansion operators.
Supports both GrowingMLP (plain) and SkipMLP (residual).
"""
from __future__ import annotations
import numpy as np


# ── Plain MLP operators ────────────────────────────────────────────

def grow_width(model, delta: int = 12):
    """Grow all hidden layers of a GrowingMLP by `delta` neurons."""
    for i in range(len(model.layers) - 1):
        layer = model.layers[i]
        old_out = layer.W.shape[1]
        new_out = old_out + delta

        # Expand current layer output
        new_W = np.zeros((layer.W.shape[0], new_out), dtype=np.float32)
        new_W[:, :old_out] = layer.W
        new_W[:, old_out:] = np.random.randn(layer.W.shape[0], delta).astype(np.float32) * 0.01
        layer.W = new_W
        layer.b = np.concatenate([layer.b, np.zeros(delta, np.float32)])
        # Reset Adam state
        layer.mW = np.zeros_like(layer.W); layer.vW = np.zeros_like(layer.W)
        layer.mb = np.zeros_like(layer.b); layer.vb = np.zeros_like(layer.b)

        # Expand next layer input
        nxt = model.layers[i + 1]
        old_in = nxt.W.shape[0]
        new_next = np.zeros((old_in + delta, nxt.W.shape[1]), np.float32)
        new_next[:old_in] = nxt.W
        new_next[old_in:] = np.random.randn(delta, nxt.W.shape[1]).astype(np.float32) * 0.001
        nxt.W = new_next
        nxt.mW = np.zeros_like(nxt.W); nxt.vW = np.zeros_like(nxt.W)

    # Update sizes
    model.sizes = [model.layers[0].W.shape[0]]
    for l in model.layers:
        model.sizes.append(l.W.shape[1])


def grow_depth(model):
    """Insert a new hidden layer (near-identity) before the output."""
    from .operators import Layer  # avoid circular
    w = model.layers[-2].W.shape[1]
    from neuroplasticity.models.growing_mlp import Layer
    new_layer = Layer(w, w, is_output=False)
    new_layer.W = np.eye(w, dtype=np.float32) * 0.1 + np.random.randn(w, w).astype(np.float32) * 0.01
    model.layers.insert(-1, new_layer)
    model.sizes.insert(-1, w)


# ── SkipMLP operators ──────────────────────────────────────────────

def grow_width_skip(model, delta: int = 16):
    """Grow all layers of a SkipMLP by `delta` neurons."""
    old_w = model.width
    new_w = old_w + delta

    # Input projection
    new_Win = np.zeros((model.d_in, new_w), np.float32)
    new_Win[:, :old_w] = model.Win
    new_Win[:, old_w:] = np.random.randn(model.d_in, delta).astype(np.float32) * 0.01
    model.Win = new_Win
    model.bin = np.concatenate([model.bin, np.zeros(delta, np.float32)])

    # Each block
    for i in range(model.n_blocks):
        for attr in ['Wa', 'Wb']:
            W_old = getattr(model, attr)[i]
            W_new = np.zeros((new_w, new_w), np.float32)
            W_new[:old_w, :old_w] = W_old
            W_new[old_w:, :old_w] = np.random.randn(delta, old_w).astype(np.float32) * 0.001
            W_new[:old_w, old_w:] = np.random.randn(old_w, delta).astype(np.float32) * 0.01
            getattr(model, attr)[i] = W_new
        for attr in ['ba', 'bb']:
            getattr(model, attr)[i] = np.concatenate(
                [getattr(model, attr)[i], np.zeros(delta, np.float32)])

    # Output
    new_Wout = np.zeros((new_w, model.d_out), np.float32)
    new_Wout[:old_w] = model.Wout
    new_Wout[old_w:] = np.random.randn(delta, model.d_out).astype(np.float32) * 0.001
    model.Wout = new_Wout


def grow_depth_skip(model):
    """Add a new residual block (near-identity) to a SkipMLP."""
    w = model.width
    model.Wa.append(np.eye(w, dtype=np.float32) * np.sqrt(2.0 / w))
    model.ba.append(np.zeros(w, np.float32))
    model.Wb.append(np.random.randn(w, w).astype(np.float32) * 0.01)
    model.bb.append(np.zeros(w, np.float32))
