"""
neuroplasticity.training.trainer
==================================
Orchestrates the neuroplasticity training loop for both GrowingMLP and SkipMLP.
Logs four IT signals every epoch and triggers autonomous growth.
"""
from __future__ import annotations
import time, gc
import numpy as np
from neuroplasticity.metrics import effective_rank, fisher_trace, mutual_info_ib, twonn_id
from neuroplasticity.growth import NeuroplasticityController


class History:
    """Stores per-epoch metrics and growth events."""
    def __init__(self):
        self.epoch     = []
        self.tr_acc    = []
        self.te_acc    = []
        self.loss      = []
        self.n_params  = []
        self.width     = []
        self.n_blocks  = []
        self.eff_rank  = []
        self.fisher    = []
        self.mi        = []
        self.rep_id    = []
        self.data_id   = 0.0
        self.growths   = []  # (epoch, kind, old_params, new_params, reason)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  NEUROPLASTICITY EXPERIMENT SUMMARY",
            "=" * 60,
            f"  Epochs    : {len(self.epoch)}",
            f"  Growths   : {len(self.growths)}  "
            f"{[(e, k) for e, k, _, _, _ in self.growths]}",
            f"  Params    : {self.n_params[0]:,} → {self.n_params[-1]:,}  "
            f"({self.n_params[-1] / max(self.n_params[0], 1):.0f}×)",
            f"  Train acc : {self.tr_acc[0]*100:.1f}% → {self.tr_acc[-1]*100:.1f}%",
            f"  Test  acc : {self.te_acc[0]*100:.1f}% → {self.te_acc[-1]*100:.1f}%",
            f"  Eff Rank  : {self.eff_rank[0]:.2f} → {self.eff_rank[-1]:.2f}",
            f"  I(T;Y)    : {self.mi[0]:.3f} → {self.mi[-1]:.3f} nats",
            f"  Rep ID    : {self.rep_id[0]:.2f} → {self.rep_id[-1]:.2f}"
            f"  (Data ID = {self.data_id:.2f})",
            "=" * 60,
        ]
        return "\n".join(lines)


class Trainer:
    """
    Trains a model (GrowingMLP or SkipMLP) with neuroplasticity growth.

    Parameters
    ----------
    model        : GrowingMLP or SkipMLP
    controller   : NeuroplasticityController
    dataset      : object with X_tr, y_tr, X_te, y_te
    grow_fn      : callable(model) for width growth
    depth_fn     : callable(model) for depth growth (optional)
    max_epochs   : total training epochs
    lr           : initial learning rate
    lr_decay     : multiply LR by this after each growth event
    l2           : weight decay coefficient
    batch        : mini-batch size
    metric_every : compute IT metrics every N epochs
    verbose      : print per-epoch log
    """
    def __init__(
        self,
        model,
        controller: NeuroplasticityController,
        dataset,
        grow_fn,
        depth_fn=None,
        max_epochs: int = 90,
        lr: float = 5e-3,
        lr_decay: float = 0.95,
        lr_min: float = 1e-3,
        l2: float = 1e-4,
        batch: int = 256,
        metric_every: int = 3,
        verbose: bool = True,
    ):
        self.model = model
        self.ctrl = controller
        self.ds = dataset
        self.grow_fn = grow_fn
        self.depth_fn = depth_fn
        self.max_epochs = max_epochs
        self.lr = lr
        self.lr_decay = lr_decay
        self.lr_min = lr_min
        self.l2 = l2
        self.batch = batch
        self.metric_every = metric_every
        self.verbose = verbose

    def _get_width(self):
        if hasattr(self.model, 'width'):
            return self.model.width
        elif hasattr(self.model, 'hidden_widths'):
            return self.model.hidden_widths[0] if self.model.hidden_widths else 0
        return 0

    def _get_nblocks(self):
        if hasattr(self.model, 'n_blocks'):
            return self.model.n_blocks
        return self.model.n_hidden if hasattr(self.model, 'n_hidden') else 0

    def _get_nparams(self):
        return self.model.n_params if hasattr(self.model, 'n_params') else self.model.total_params()

    def run(self) -> History:
        hist = History()
        X_tr, y_tr = self.ds.X_tr, self.ds.y_tr
        X_te, y_te = self.ds.X_te, self.ds.y_te
        N = len(y_tr)

        # Fixed analysis subset
        an_idx = np.random.RandomState(7).choice(N, min(2000, N), replace=False)
        X_an, y_an = X_tr[an_idx], y_tr[an_idx]

        # Data ID
        hist.data_id = twonn_id(X_an)
        if self.verbose:
            print(f"Data ID = {hist.data_id:.2f}")
            print(f"\nTraining {self.max_epochs} epochs, target ~{self.ctrl.max_params:,} params")
            print("-" * 75)

        t0 = time.time()
        for ep in range(1, self.max_epochs + 1):
            idx = np.random.permutation(N)
            eloss, nb = 0.0, 0
            for s in range(0, N, self.batch):
                bi = idx[s:s + self.batch]
                # Weight decay
                for W in self.model.weight_matrices():
                    W *= (1 - self.l2 * self.lr)
                eloss += self.model.train_step(X_tr[bi], y_tr[bi], self.lr)
                nb += 1
            eloss /= nb

            tr_a = float((self.model.predict(X_tr) == y_tr).mean())
            te_a = float((self.model.predict(X_te) == y_te).mean())

            # IT metrics
            if ep % self.metric_every == 0 or ep <= 3 or ep == self.max_epochs:
                er = effective_rank(self.model)
                fi = fisher_trace(self.model, X_an[:512], y_an[:512])
                mi = mutual_info_ib(self.model, X_an, y_an)
                rid = twonn_id(self.model.represent(X_an))
            else:
                er = hist.eff_rank[-1] if hist.eff_rank else 1.0
                fi = hist.fisher[-1] if hist.fisher else 0.0
                mi = hist.mi[-1] if hist.mi else 0.0
                rid = hist.rep_id[-1] if hist.rep_id else 1.0

            hist.epoch.append(ep)
            hist.tr_acc.append(tr_a)
            hist.te_acc.append(te_a)
            hist.loss.append(eloss)
            hist.n_params.append(self._get_nparams())
            hist.width.append(self._get_width())
            hist.n_blocks.append(self._get_nblocks())
            hist.eff_rank.append(er)
            hist.fisher.append(fi)
            hist.mi.append(mi)
            hist.rep_id.append(rid)

            # Growth check
            do_g, reason = self.ctrl.should_grow(ep, hist.tr_acc, self._get_nparams())
            status = ""
            if do_g:
                old_p = self._get_nparams()
                kind = self.ctrl.decide_growth_type()
                if kind == "depth" and self.depth_fn:
                    self.depth_fn(self.model)
                    status = " ★ DEPTH ↕"
                else:
                    self.grow_fn(self.model)
                    kind = "width"
                    status = " ★ WIDTH ↔"

                new_p = self._get_nparams()
                hist.growths.append((ep, kind, old_p, new_p, reason))
                self.ctrl.record_growth(ep, kind)
                self.lr = max(self.lr * self.lr_decay, self.lr_min)

            elif self.ctrl.width_gains and ep == self.ctrl.last_grow + 1:
                if len(hist.tr_acc) >= 2:
                    self.ctrl.update_width_gain(hist.tr_acc[-1] - hist.tr_acc[-2])

            if self.verbose and (ep % 5 == 0 or ep <= 3 or do_g or ep == self.max_epochs):
                gap = hist.data_id - rid
                print(f"  Ep {ep:3d} | tr {tr_a:.3f} te {te_a:.3f} | "
                      f"loss {eloss:.3f} | p={self._get_nparams():>7,} "
                      f"w={self._get_width():3d} b={self._get_nblocks()} | "
                      f"ER={er:.1f} MI={mi:.2f} RID={rid:.1f} gap={gap:.1f} | "
                      f"{time.time()-t0:.0f}s{status}")

            if ep % 20 == 0:
                gc.collect()

        if self.verbose:
            print("-" * 75)
            print(f"Done in {time.time()-t0:.0f}s")
            print(hist.summary())

        return hist
