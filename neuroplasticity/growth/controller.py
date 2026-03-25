"""
neuroplasticity.growth.controller
===================================
NeuroplasticityController — detects when a model is struggling and
decides whether to grow width or depth.
"""
from __future__ import annotations


class NeuroplasticityController:
    """
    Monitors training and triggers growth when the model stalls.

    Parameters
    ----------
    cooldown     : minimum epochs between growth events
    delta_acc    : per-epoch improvement threshold below which = stalled
    acc_thresh   : don't grow if accuracy exceeds this
    max_params   : hard parameter cap
    window       : lookback window for improvement rate
    """
    def __init__(
        self,
        cooldown:   int   = 8,
        delta_acc:  float = 0.005,
        acc_thresh: float = 0.93,
        max_params: int   = 300_000,
        window:     int   = 5,
    ):
        self.cooldown   = cooldown
        self.delta_acc  = delta_acc
        self.acc_thresh = acc_thresh
        self.max_params = max_params
        self.window     = window
        self.last_grow  = -cooldown
        self.width_gains: list[float] = []

    def should_grow(self, epoch: int, accs: list[float], total_params: int):
        """
        Returns (should_grow: bool, reason: str).
        """
        if epoch - self.last_grow < self.cooldown:
            return False, "cooldown"
        if total_params >= self.max_params:
            return False, "max_params"
        if len(accs) < 3:
            return False, "too_early"
        if accs[-1] > self.acc_thresh:
            return False, "above_threshold"

        w = min(self.window, len(accs) - 1)
        improvement = (accs[-1] - accs[-1 - w]) / w
        if improvement < self.delta_acc:
            return True, f"stalled (Δ={improvement:.4f}/ep)"
        return False, f"improving ({improvement:.4f}/ep)"

    def decide_growth_type(self):
        """Returns 'width' or 'depth'."""
        if len(self.width_gains) >= 3 and all(g < 0.008 for g in self.width_gains[-3:]):
            self.width_gains.clear()
            return "depth"
        return "width"

    def record_growth(self, epoch: int, kind: str):
        self.last_grow = epoch
        if kind == "width":
            self.width_gains.append(0.0)

    def update_width_gain(self, gain: float):
        if self.width_gains:
            self.width_gains[-1] = max(0, gain)
