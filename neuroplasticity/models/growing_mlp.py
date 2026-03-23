"""
neuroplasticity.models.growing_mlp
====================================
Dynamically growing fully-connected network.

  • Pure NumPy implementation — no deep learning framework required
  • float32 throughout (RAM-efficient; only tiny local buffers touch float64)
  • Adam optimiser with per-layer momentum state
  • grow_width() and grow_depth() are implemented in neuroplasticity.growth

Architecture
------------
  sizes = [in_dim, h1, h2, ..., n_classes]
  Activations: ReLU for hidden layers, softmax for output
  Loss: cross-entropy

Example
-------
  model = GrowingMLP([96, 4, 4, 10])
  loss, acc = model.fit_epoch(X_tr, y_tr, lr=2.5e-3, l2=5e-4)
  te_loss, te_acc = model.evaluate(X_te, y_te)
  reps = model.represent(X_te)          # penultimate activations
"""

from __future__ import annotations

import numpy as np


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x, dtype=np.float32)


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    ex = np.exp(z)
    return ex / (ex.sum(axis=1, keepdims=True) + 1e-7)


class Layer:
    """
    A single linear layer with float32 Adam moment buffers.

    Attributes
    ----------
    W  : weight matrix [in_dim, out_dim]
    b  : bias vector   [out_dim]
    mW, vW : Adam first and second moments for W
    mb, vb : Adam first and second moments for b
    t  : Adam step counter
    """

    __slots__ = ("W", "b", "mW", "vW", "mb", "vb", "t")

    def __init__(self, in_dim: int, out_dim: int) -> None:
        scale   = np.sqrt(2.0 / in_dim)
        self.W  = (np.random.randn(in_dim, out_dim) * scale).astype(np.float32)
        self.b  = np.zeros(out_dim, dtype=np.float32)
        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b)
        self.vb = np.zeros_like(self.b)
        self.t  = 0


class GrowingMLP:
    """
    Fully-connected network whose architecture expands at runtime.

    Parameters
    ----------
    sizes : list of ints, e.g. [96, 4, 4, 10]
            First element = input dim, last = number of classes.

    Notes
    -----
    Weights live in float32. The only float64 operations are SVD (for
    effective rank) and TwoNN cdist (both done on small subsets in the
    metrics module). All Adam buffers are float32.
    """

    def __init__(self, sizes: list[int]) -> None:
        self.sizes  = list(sizes)
        self.layers = [Layer(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]

    # ── Forward pass ─────────────────────────────────────────────────────────

    def _forward(self, X: np.ndarray) -> list[np.ndarray]:
        """
        Return list of activations [a0=X, a1, a2, ..., a_L=softmax].
        All arrays are float32.
        """
        acts = [X]
        for i, L in enumerate(self.layers):
            z = acts[-1] @ L.W + L.b
            if i < len(self.layers) - 1:
                acts.append(_relu(z))
            else:
                acts.append(_softmax(z))
        return acts

    # ── Single mini-batch Adam update ─────────────────────────────────────────

    def _step(
        self,
        X:   np.ndarray,
        y:   np.ndarray,
        lr:  float,
        l2:  float,
        b1:  float = 0.9,
        b2:  float = 0.999,
        eps: float = 1e-7,
    ) -> tuple[float, float]:
        n    = len(y)
        acts = self._forward(X)
        prob = acts[-1]

        loss = -np.log(prob[np.arange(n), y] + 1e-7).mean()
        acc  = (prob.argmax(1) == y).mean()

        # Softmax + cross-entropy combined gradient
        g = prob.copy()
        g[np.arange(n), y] -= 1
        g /= n

        for i in reversed(range(len(self.layers))):
            L  = self.layers[i]
            gW = (acts[i].T @ g + l2 * L.W).astype(np.float32)
            gb = g.sum(axis=0).astype(np.float32)

            L.t += 1
            bc1 = 1.0 - b1 ** L.t
            bc2 = 1.0 - b2 ** L.t

            L.mW = b1 * L.mW + (1 - b1) * gW
            L.vW = b2 * L.vW + (1 - b2) * gW * gW
            L.mb = b1 * L.mb + (1 - b1) * gb
            L.vb = b2 * L.vb + (1 - b2) * gb * gb

            L.W -= (lr / bc1) * L.mW / (np.sqrt(L.vW / bc2) + eps)
            L.b -= (lr / bc1) * L.mb / (np.sqrt(L.vb / bc2) + eps)

            if i > 0:
                g = ((g @ L.W.T) * (acts[i] > 0)).astype(np.float32)

        return float(loss), float(acc)

    # ── Full-epoch training ───────────────────────────────────────────────────

    def fit_epoch(
        self,
        X:     np.ndarray,
        y:     np.ndarray,
        lr:    float,
        l2:    float,
        batch: int = 256,
    ) -> tuple[float, float]:
        """
        Train for one epoch with random mini-batch shuffling.

        Returns
        -------
        (mean_loss, mean_accuracy)  both averaged over all samples
        """
        idx    = np.random.permutation(len(y))
        tl, tc = 0.0, 0
        for s in range(0, len(y), batch):
            bi  = idx[s:s + batch]
            l, a = self._step(X[bi], y[bi], lr, l2)
            tl  += l * len(bi)
            tc  += int(a * len(bi))
        return tl / len(y), tc / len(y)

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        X:     np.ndarray,
        y:     np.ndarray,
        batch: int = 512,
    ) -> tuple[float, float]:
        """Return (loss, accuracy) on the full dataset X, y."""
        tl, tc = 0.0, 0
        for s in range(0, len(y), batch):
            xb, yb = X[s:s + batch], y[s:s + batch]
            prob   = self._forward(xb)[-1]
            tl    += -np.log(prob[np.arange(len(yb)), yb] + 1e-7).mean() * len(yb)
            tc    += (prob.argmax(1) == yb).sum()
        return tl / len(y), tc / len(y)

    # ── Penultimate representations ───────────────────────────────────────────

    def represent(
        self,
        X:     np.ndarray,
        max_n: int = 300,
        batch: int = 256,
    ) -> np.ndarray:
        """
        Return penultimate-layer activations (before the final linear).

        Parameters
        ----------
        X     : input array, first max_n rows used
        max_n : maximum number of samples to process (keeps memory bounded)

        Returns
        -------
        float32 array of shape (min(len(X), max_n), hidden_dim)
        """
        X    = X[:max_n]
        reps = []
        for s in range(0, len(X), batch):
            a = X[s:s + batch]
            for L in self.layers[:-1]:
                a = _relu(a @ L.W + L.b)
            reps.append(a)
        return np.vstack(reps).astype(np.float32)

    # ── Utilities ─────────────────────────────────────────────────────────────

    def n_params(self) -> int:
        """Total number of trainable parameters."""
        return sum(L.W.size + L.b.size for L in self.layers)

    def __repr__(self) -> str:
        return f"GrowingMLP(sizes={self.sizes}, params={self.n_params():,})"
