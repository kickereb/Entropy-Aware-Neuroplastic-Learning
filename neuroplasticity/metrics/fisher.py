"""Fisher Information Trace — E[(∂L/∂θ)²] proxy via output-layer gradient norm."""
import numpy as np

def fisher_trace(model, X_batch, y_batch) -> float:
    """Approximate Fisher trace via output-layer gradient norm."""
    probs, cache = model.forward(X_batch)
    n = len(y_batch)
    dl = probs.copy()
    dl[np.arange(n), y_batch] -= 1.0
    dl /= n
    h_final = cache.get('h_final', cache[-1] if isinstance(cache, list) else None)
    if h_final is None:
        # For GrowingMLP, use last hidden activations
        h_final = model.represent(X_batch, max_n=len(X_batch))
    gW = h_final.T @ dl
    return float(np.sum(gW ** 2))
