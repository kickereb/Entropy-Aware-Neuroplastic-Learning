"""
neuroplasticity.models.growing_mlp
====================================
Pure-NumPy MLP with Adam optimiser. Supports arbitrary depth/width
and exposes representations + weight matrices for IT metric computation.
"""
from __future__ import annotations
import numpy as np


class Layer:
    """Single linear layer with Adam state."""
    def __init__(self, fan_in: int, fan_out: int, is_output: bool = False):
        scale = np.sqrt(1.0 / fan_in) if is_output else np.sqrt(2.0 / fan_in)
        self.W = np.random.randn(fan_in, fan_out).astype(np.float32) * scale
        self.b = np.zeros(fan_out, dtype=np.float32)
        # Adam state
        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b)
        self.vb = np.zeros_like(self.b)
        self.t  = 0

    def adam_step(self, gW, gb, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1
        self.mW = beta1 * self.mW + (1 - beta1) * gW
        self.vW = beta2 * self.vW + (1 - beta2) * gW ** 2
        mW_hat = self.mW / (1 - beta1 ** self.t)
        vW_hat = self.vW / (1 - beta2 ** self.t)
        self.W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)

        self.mb = beta1 * self.mb + (1 - beta1) * gb
        self.vb = beta2 * self.vb + (1 - beta2) * gb ** 2
        mb_hat = self.mb / (1 - beta1 ** self.t)
        vb_hat = self.vb / (1 - beta2 ** self.t)
        self.b -= lr * mb_hat / (np.sqrt(vb_hat) + eps)


class GrowingMLP:
    """
    Multi-layer perceptron that can be grown wider (Net2Net) or deeper.

    Parameters
    ----------
    sizes : list[int]  e.g. [96, 4, 4, 10] for 2 hidden layers of width 4
    """
    def __init__(self, sizes: list[int]):
        self.sizes = list(sizes)
        self.layers = []
        for i in range(len(sizes) - 1):
            self.layers.append(Layer(sizes[i], sizes[i+1], is_output=(i == len(sizes) - 2)))

    @property
    def n_hidden(self): return len(self.layers) - 1
    @property
    def hidden_widths(self): return [l.W.shape[1] for l in self.layers[:-1]]
    @property
    def n_params(self): return sum(l.W.size + l.b.size for l in self.layers)

    def forward(self, X: np.ndarray):
        """Returns (probs, activations_list)."""
        acts = [X]
        h = X
        for i, layer in enumerate(self.layers):
            z = h @ layer.W + layer.b
            if i < len(self.layers) - 1:
                h = np.maximum(0, z)  # ReLU
            else:
                e = np.exp(z - z.max(axis=1, keepdims=True))
                h = e / (e.sum(axis=1, keepdims=True) + 1e-12)
            acts.append(h)
        return h, acts

    def train_step(self, X, y, lr, l2=5e-4):
        probs, acts = self.forward(X)
        n = len(y)
        loss = -np.log(probs[np.arange(n), y] + 1e-12).mean()

        dl = probs.copy()
        dl[np.arange(n), y] -= 1.0
        dl /= n

        for i in reversed(range(len(self.layers))):
            h_in = acts[i]
            gW = h_in.T @ dl + l2 * self.layers[i].W
            gb = dl.sum(axis=0)

            if i > 0:
                z_prev = acts[i - 1] @ self.layers[i - 1].W + self.layers[i - 1].b
                # This is wrong — we need pre-activation of current layer's input
                dl = dl @ self.layers[i].W.T
                dl *= (acts[i] > 0).astype(np.float32) if i < len(self.layers) - 1 else 1.0

            self.layers[i].adam_step(gW, gb, lr)
        return loss

    def predict(self, X, batch_size=1024):
        preds = []
        for s in range(0, len(X), batch_size):
            p, _ = self.forward(X[s:s+batch_size])
            preds.append(p.argmax(axis=1))
        return np.concatenate(preds)

    def represent(self, X, max_n=2000):
        """Penultimate activations."""
        X = X[:max_n]
        h = X
        for i in range(len(self.layers) - 1):
            h = np.maximum(0, h @ self.layers[i].W + self.layers[i].b)
        return h

    def weight_matrices(self):
        return [l.W for l in self.layers]

    def __repr__(self):
        return f"GrowingMLP({self.sizes}, params={self.n_params:,})"
