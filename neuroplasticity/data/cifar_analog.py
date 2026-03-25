"""
neuroplasticity.data.cifar_analog
==================================
Synthetic 96-dim, 10-class dataset with quadratic (XOR) label structure.
Designed to mirror CIFAR-10's statistical complexity with known intrinsic
dimensionality and controllable difficulty.
"""
from __future__ import annotations
import numpy as np

N_CLASSES  = 10
FEAT_DIM   = 96
LATENT_DIM = 8
PAIRS      = [(0,1),(2,3),(4,5),(6,7),(0,2),(1,3),(2,4),(3,5)]

# Fixed label weights and projection matrix (shared across all splits)
_W_LBL  = np.random.RandomState(0).randn(len(PAIRS), N_CLASSES).astype(np.float32) * 0.8
_P_PROJ = np.random.RandomState(1).randn(LATENT_DIM, FEAT_DIM).astype(np.float32)
_P_PROJ /= np.linalg.norm(_P_PROJ, axis=1, keepdims=True)


def make_dataset(
    n: int,
    seed: int = 42,
    noise_sigma: float = 0.22,
    balanced: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a balanced CIFAR-10 analog dataset.

    Returns (X, y) where X is [n, 96] float32 and y is [n] int32.
    """
    rng = np.random.RandomState(seed)
    per_class = n // N_CLASSES

    if balanced:
        Xs, ys = [], []
        counts = np.zeros(N_CLASSES, dtype=int)
        for _ in range(300):
            if all(counts >= per_class):
                break
            Z = rng.randn(2000, LATENT_DIM).astype(np.float32)
            sc = np.zeros((2000, N_CLASSES), np.float32)
            for i, (a, b) in enumerate(PAIRS):
                sc += (Z[:, a] * Z[:, b]).reshape(-1, 1) * _W_LBL[i:i+1]
            yb = sc.argmax(1)
            for j in range(2000):
                c = yb[j]
                if counts[c] < per_class:
                    Xs.append(Z[j])
                    ys.append(c)
                    counts[c] += 1
        Z = np.array(Xs[:n], np.float32)
        y = np.array(ys[:n], np.int32)
    else:
        Z = rng.randn(n, LATENT_DIM).astype(np.float32)
        sc = np.zeros((n, N_CLASSES), np.float32)
        for i, (a, b) in enumerate(PAIRS):
            sc += (Z[:, a] * Z[:, b]).reshape(-1, 1) * _W_LBL[i:i+1]
        y = sc.argmax(1).astype(np.int32)

    perm = rng.permutation(len(Z))
    Z, y = Z[perm], y[perm]

    X = Z @ _P_PROJ + rng.randn(len(Z), FEAT_DIM).astype(np.float32) * noise_sigma
    mu, sd = X.mean(0), X.std(0) + 1e-8
    X = (X - mu) / sd
    return X, y


class CIFARAnalog:
    """Container holding train/test splits."""
    feat_dim  = FEAT_DIM
    n_classes = N_CLASSES

    def __init__(self, n_train=12_000, n_test=4_000, noise_sigma=0.22):
        self.X_tr, self.y_tr = make_dataset(n_train, seed=1, noise_sigma=noise_sigma)
        self.X_te, self.y_te = make_dataset(n_test,  seed=2, noise_sigma=noise_sigma)
        self.n_train = n_train
        self.n_test  = n_test

    def __repr__(self):
        return (f"CIFARAnalog(n_train={self.n_train:,}, n_test={self.n_test:,}, "
                f"feat_dim={self.feat_dim}, n_classes={self.n_classes})")
