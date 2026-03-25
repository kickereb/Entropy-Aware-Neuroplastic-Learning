"""
neuroplasticity.init.pipeline
===============================
Full dataset-informed initialisation pipeline for SkipMLP:
  1. PCsInit for first layer
  2. Train reference model → project weights into first block
  3. LSUV calibration
"""
from __future__ import annotations
import numpy as np
from .pcs_init import pcs_init
from .teacher import train_reference_model, project_weights_svd
from .lsuv import lsuv_calibrate


def dataset_informed_init(model, X_tr, y_tr, X_te, y_te, ref_width=48, verbose=True):
    """
    Apply the full three-stage init to a SkipMLP.

    Parameters
    ----------
    model     : SkipMLP instance
    X_tr, y_tr: training data
    X_te, y_te: test data (for reference model evaluation)
    ref_width : width of the small reference model
    verbose   : print progress
    """
    if verbose:
        print("\n=== Dataset-Informed Initialisation ===")

    # 1. PCsInit
    model.Win = pcs_init(X_tr, model.width)
    if verbose:
        print(f"  [PCsInit] {min(model.width, X_tr.shape[1])} PCs → {model.width}-wide layer")

    # 2. Reference model
    ref = train_reference_model(X_tr, y_tr, X_te, y_te, width=ref_width, verbose=verbose)

    # 3. Project reference weights into first block
    if model.n_blocks > 0:
        w = model.width
        U, S, Vt = np.linalg.svd(ref['W2'], full_matrices=False)
        k = min(len(S), w)
        proj = np.zeros((w, w), np.float32)
        proj[:k, :k] = np.diag(S[:k])
        if k < w:
            proj[k:, k:] = np.eye(w - k, dtype=np.float32) * 0.01
        model.Wa[0] = proj
        if verbose:
            print(f"  [WeightProj] Projected ref weights into block 0")

    # 4. LSUV
    si = np.random.choice(len(X_tr), min(800, len(X_tr)), replace=False)
    lsuv_calibrate(model, X_tr[si])
    if verbose:
        print("=== Init Complete ===\n")
