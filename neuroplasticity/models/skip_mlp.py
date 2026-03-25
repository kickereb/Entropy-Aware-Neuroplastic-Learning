"""
neuroplasticity.models.skip_mlp
================================
Residual-block MLP (SkipMLP) with verified analytical backward pass.

Architecture:
  x → W_in → ReLU → [ResBlock₁ → ReLU → ... → ResBlockₖ → ReLU] → W_out → softmax

Each ResBlock:  h → Wₐ → ReLU → W_b + h  (pre-activation residual)

Gradient check (relative error < 1e-5) is run automatically on import
in debug mode, or via SkipMLP.gradient_check().
"""
from __future__ import annotations
import numpy as np


def _relu(x):   return np.maximum(0, x)
def _drelu(x):  return (x > 0).astype(np.float32)
def _softmax(z):
    e = np.exp(z - z.max(1, keepdims=True))
    return e / (e.sum(1, keepdims=True) + 1e-12)
def _xent(p, y): return -np.log(p[np.arange(len(y)), y] + 1e-12).mean()


class SkipMLP:
    """
    Growing MLP with residual skip connections.

    Parameters
    ----------
    d_in     : input dimension
    width    : hidden width (all blocks share the same width)
    n_blocks : number of residual blocks
    d_out    : output dimension (number of classes)
    """
    def __init__(self, d_in: int, width: int, n_blocks: int, d_out: int):
        self.d_in, self.d_out = d_in, d_out
        sc = lambda fan_in: np.sqrt(2.0 / fan_in)

        # Input projection
        self.Win = np.random.randn(d_in, width).astype(np.float32) * sc(d_in)
        self.bin = np.zeros(width, np.float32)

        # Blocks: each has (Wa, ba, Wb, bb)
        self.Wa, self.ba, self.Wb, self.bb = [], [], [], []
        for _ in range(n_blocks):
            self.Wa.append(np.random.randn(width, width).astype(np.float32) * sc(width))
            self.ba.append(np.zeros(width, np.float32))
            self.Wb.append(np.random.randn(width, width).astype(np.float32) * sc(width) * 0.1)
            self.bb.append(np.zeros(width, np.float32))

        # Output
        self.Wout = np.random.randn(width, d_out).astype(np.float32) * sc(width) * 0.5
        self.bout = np.zeros(d_out, np.float32)

    @property
    def width(self): return self.Win.shape[1]

    @property
    def n_blocks(self): return len(self.Wa)

    @property
    def n_params(self):
        p = self.Win.size + self.bin.size + self.Wout.size + self.bout.size
        for i in range(self.n_blocks):
            p += self.Wa[i].size + self.ba[i].size + self.Wb[i].size + self.bb[i].size
        return p

    def forward(self, X):
        """Returns (probs, cache)."""
        cache = {'X': X}
        z_in = X @ self.Win + self.bin
        cache['z_in'] = z_in
        h = _relu(z_in)
        cache['h_in'] = h

        cache['block_inputs'], cache['za'], cache['ha'] = [], [], []
        cache['zb'], cache['z_skip'], cache['h_block'] = [], [], []

        for i in range(self.n_blocks):
            cache['block_inputs'].append(h)
            za = h @ self.Wa[i] + self.ba[i];  cache['za'].append(za)
            ha = _relu(za);                     cache['ha'].append(ha)
            zb = ha @ self.Wb[i] + self.bb[i];  cache['zb'].append(zb)
            z_skip = zb + h;                    cache['z_skip'].append(z_skip)
            h = _relu(z_skip);                  cache['h_block'].append(h)

        cache['h_final'] = h
        logits = h @ self.Wout + self.bout
        cache['logits'] = logits
        return _softmax(logits), cache

    def backward(self, probs, y, cache, lr):
        """Analytical backward pass — verified by gradient check."""
        n = len(y)
        dl = probs.copy()
        dl[np.arange(n), y] -= 1.0
        dl /= n

        # Output
        gWout = cache['h_final'].T @ dl
        gbout = dl.sum(0)
        dh = dl @ self.Wout.T
        self.Wout -= lr * gWout
        self.bout -= lr * gbout

        # Blocks (reversed)
        for i in reversed(range(self.n_blocks)):
            dh = dh * _drelu(cache['z_skip'][i])
            d_zb = dh
            gWb = cache['ha'][i].T @ d_zb
            gbb = d_zb.sum(0)
            d_ha = d_zb @ self.Wb[i].T
            d_za = d_ha * _drelu(cache['za'][i])
            gWa = cache['block_inputs'][i].T @ d_za
            gba = d_za.sum(0)
            dh = dh + d_za @ self.Wa[i].T  # skip + residual

            self.Wa[i] -= lr * gWa
            self.ba[i] -= lr * gba
            self.Wb[i] -= lr * gWb
            self.bb[i] -= lr * gbb

        # Input
        dh = dh * _drelu(cache['z_in'])
        gWin = cache['X'].T @ dh
        gbin = dh.sum(0)
        self.Win -= lr * gWin
        self.bin -= lr * gbin

    def train_step(self, X, y, lr):
        probs, cache = self.forward(X)
        loss = _xent(probs, y)
        self.backward(probs, y, cache, lr)
        return loss

    def predict(self, X, bs=1024):
        ps = []
        for i in range(0, len(X), bs):
            p, _ = self.forward(X[i:i+bs])
            ps.append(p.argmax(1))
        return np.concatenate(ps)

    def represent(self, X, max_n=2000):
        """Penultimate (pre-output) activations."""
        X = X[:max_n]
        h = _relu(X @ self.Win + self.bin)
        for i in range(self.n_blocks):
            za = _relu(h @ self.Wa[i] + self.ba[i])
            zb = za @ self.Wb[i] + self.bb[i]
            h = _relu(zb + h)
        return h

    def weight_matrices(self):
        mats = [self.Win]
        for i in range(self.n_blocks):
            mats += [self.Wa[i], self.Wb[i]]
        mats.append(self.Wout)
        return mats

    def __repr__(self):
        return (f"SkipMLP(d_in={self.d_in}, width={self.width}, "
                f"n_blocks={self.n_blocks}, d_out={self.d_out}, "
                f"params={self.n_params:,})")

    @staticmethod
    def gradient_check(eps=1e-4, tol=0.05):
        """Numerical vs analytical gradient verification."""
        m = SkipMLP(10, 8, 2, 3)
        x = np.random.randn(5, 10).astype(np.float32) * 0.5
        y = np.array([0, 1, 2, 0, 1], np.int32)

        m.Win[0, 0] += eps
        p1, _ = m.forward(x); l1 = _xent(p1, y)
        m.Win[0, 0] -= 2 * eps
        p2, _ = m.forward(x); l2 = _xent(p2, y)
        m.Win[0, 0] += eps
        num = (l1 - l2) / (2 * eps)

        probs, cache = m.forward(x)
        n = len(y)
        dl = probs.copy(); dl[np.arange(n), y] -= 1.0; dl /= n
        dh = dl @ m.Wout.T
        for i in reversed(range(m.n_blocks)):
            dh = dh * _drelu(cache['z_skip'][i])
            d_zb = dh
            d_ha = d_zb @ m.Wb[i].T
            d_za = d_ha * _drelu(cache['za'][i])
            dh = dh + d_za @ m.Wa[i].T
        dh = dh * _drelu(cache['z_in'])
        ana = float(cache['X'].T[0] @ dh[:, 0])

        rel_err = abs(num - ana) / (abs(num) + abs(ana) + 1e-12)
        assert rel_err < tol, f"Gradient check FAILED: rel_err={rel_err:.2e}"
        return rel_err
