"""LSUV (Layer-Sequential Unit-Variance) calibration (Mishkin & Matas 2015)."""
import numpy as np

def lsuv_calibrate(model, X_sample, target_var=1.0, max_iter=10, tol=0.1):
    """Normalise each layer's output variance to target_var using a forward pass."""
    _relu = lambda x: np.maximum(0, x)

    # Input layer
    for _ in range(max_iter):
        h = _relu(X_sample @ model.Win + model.bin)
        v = np.var(h)
        if v < 1e-8 or abs(v - target_var) / target_var < tol:
            break
        model.Win *= np.sqrt(target_var / (v + 1e-8))

    # Re-trace through all blocks
    h = _relu(X_sample @ model.Win + model.bin)
    for i in range(model.n_blocks):
        for _ in range(max_iter):
            za = _relu(h @ model.Wa[i] + model.ba[i])
            zb = za @ model.Wb[i] + model.bb[i]
            out = _relu(zb + h)
            v = np.var(out)
            if v < 1e-8 or abs(v - target_var) / target_var < tol:
                break
            model.Wb[i] *= np.sqrt(target_var / (v + 1e-8))
        # Advance h
        za = _relu(h @ model.Wa[i] + model.ba[i])
        zb = za @ model.Wb[i] + model.bb[i]
        h = _relu(zb + h)
