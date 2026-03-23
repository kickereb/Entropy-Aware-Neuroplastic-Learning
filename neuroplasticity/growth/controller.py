"""
neuroplasticity.growth.controller
===================================
NeuroplasticityController — the "struggle detector".

Growth fires when ALL three conditions hold simultaneously:

  (1) train_acc < acc_thresh
        The model is genuinely struggling.

  (2) |train_acc − prev_train_acc| < delta_acc
        Accuracy improvement in the last epoch is below the plateau threshold.
        The gradient signal is no longer moving the needle.

  (3) epoch − last_growth_epoch >= cooldown
        Enough time has elapsed since the last growth event.
        New neurons need epochs to specialise before another trigger is valid.

Width → Depth switch:
  After >= 3 width events, if the last 2 each produced < 1.5% accuracy gain,
  width expansion has hit diminishing returns. Switch to depth growth
  (a new representational layer) instead.

Typical parameters
------------------
  acc_thresh  = 0.88    grow until within 12 pp of target
  delta_acc   = 0.006   plateau = < 0.6% gain per epoch
  cooldown    = 7       epochs between events

Example
-------
  ctrl = NeuroplasticityController()
  for ep in range(MAX_EPOCHS):
      tr_loss, tr_acc = model.fit_epoch(...)
      if ctrl.should_grow(ep, tr_acc):
          if ctrl.prefer_depth():
              model = grow_depth(model)
              ctrl.record('depth', ep, tr_acc)
          else:
              model = grow_width(model, delta=12)
              ctrl.record('width', ep, tr_acc)
"""

from __future__ import annotations


class NeuroplasticityController:
    """
    Detects when a model is genuinely struggling and needs to grow.

    Parameters
    ----------
    acc_thresh  : growth fires while train_acc < this threshold (default 0.88)
    delta_acc   : plateau threshold — fire if improvement < this per epoch
                  (default 0.006 = 0.6%)
    cooldown    : minimum epochs between consecutive growth events (default 7)
    depth_gain_thresh : width→depth switch threshold — if last 2 width events
                  each gained less than this, prefer depth (default 0.015)
    """

    def __init__(
        self,
        acc_thresh:        float = 0.88,
        delta_acc:         float = 0.006,
        cooldown:          int   = 7,
        depth_gain_thresh: float = 0.015,
    ) -> None:
        self.acc_thresh        = acc_thresh
        self.delta_acc         = delta_acc
        self.cooldown          = cooldown
        self.depth_gain_thresh = depth_gain_thresh

        self._prev_acc:     float = 0.0
        self._last_growth:  int   = -9999
        self.growth_accs:   list  = []   # train_acc at each growth event
        self.growth_kinds:  list  = []   # 'width' | 'depth'
        self.width_count:   int   = 0

    def should_grow(self, epoch: int, train_acc: float) -> bool:
        """
        Return True if growth conditions are met.

        Side effect: updates `_prev_acc` — call once per epoch.

        Parameters
        ----------
        epoch     : current epoch number (1-indexed)
        train_acc : training accuracy in [0, 1] this epoch
        """
        low    = train_acc  < self.acc_thresh
        stall  = abs(train_acc - self._prev_acc) < self.delta_acc
        cooled = (epoch - self._last_growth) >= self.cooldown

        self._prev_acc = train_acc   # always update
        return bool(low and stall and cooled)

    def prefer_depth(self) -> bool:
        """
        Return True if the controller recommends a depth growth instead
        of the default width growth.

        Fires when the last two width events each produced less than
        `depth_gain_thresh` accuracy gain (diminishing returns from width).
        """
        if self.width_count < 3 or len(self.growth_accs) < 3:
            return False
        gains = [
            self.growth_accs[i] - self.growth_accs[i - 1]
            for i in range(1, len(self.growth_accs))
        ]
        return len(gains) >= 2 and all(
            g < self.depth_gain_thresh for g in gains[-2:]
        )

    def record(self, kind: str, epoch: int, train_acc: float) -> None:
        """
        Register a completed growth event.

        Parameters
        ----------
        kind      : 'width' or 'depth'
        epoch     : epoch at which growth occurred
        train_acc : training accuracy at the time of growth
        """
        self.growth_accs.append(train_acc)
        self.growth_kinds.append(kind)
        self._last_growth = epoch
        if kind == "width":
            self.width_count += 1

    @property
    def n_events(self) -> int:
        """Total number of growth events recorded."""
        return len(self.growth_accs)

    def __repr__(self) -> str:
        return (
            f"NeuroplasticityController("
            f"events={self.n_events}, "
            f"width={self.width_count}, "
            f"depth={self.n_events - self.width_count})"
        )
