"""
neuroplasticity.training.trainer
==================================
Main training loop that wires together model, growth controller,
IT metrics, and history logging into a single Trainer object.

Example
-------
    from neuroplasticity.data      import CIFARAnalog
    from neuroplasticity.models    import GrowingMLP
    from neuroplasticity.growth    import NeuroplasticityController
    from neuroplasticity.training  import Trainer
    from neuroplasticity.utils     import plot_results

    ds      = CIFARAnalog(n_train=10_000, n_test=3_000)
    model   = GrowingMLP([96, 4, 4, 10])
    ctrl    = NeuroplasticityController()
    trainer = Trainer(model, ctrl, ds, max_epochs=65)
    history = trainer.run()

    plot_results(history, out_path="outputs/results.png")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from neuroplasticity.data.cifar_analog   import CIFARAnalog
from neuroplasticity.growth.controller   import NeuroplasticityController
from neuroplasticity.growth.operators    import grow_depth, grow_width
from neuroplasticity.metrics.effective_rank import mean_eff_rank
from neuroplasticity.metrics.fisher      import fisher_trace
from neuroplasticity.metrics.mutual_info import mutual_info_ib
from neuroplasticity.metrics.twonn       import rep_id, twonn_id
from neuroplasticity.models.growing_mlp  import GrowingMLP


# ── History container ─────────────────────────────────────────────────────────

@dataclass
class History:
    """
    All per-epoch metrics and growth events recorded during a training run.

    Attributes
    ----------
    ep         : epoch numbers (1-indexed)
    tr_loss    : training cross-entropy loss
    tr_acc     : training accuracy
    te_loss    : test cross-entropy loss
    te_acc     : test accuracy
    n_params   : model parameter count
    sizes      : model architecture sizes list
    eff_rank   : mean effective rank across all weight matrices
    fisher     : log10 of Fisher Information trace
    mi         : I(T;Y) mutual information in nats
    rep_id     : TwoNN intrinsic dimensionality of penultimate activations
    data_id    : TwoNN intrinsic dimensionality of raw data (fixed reference)
    growths    : list of (epoch, 'width'|'depth', old_repr, new_repr)
    elapsed_s  : total training time in seconds
    """

    ep:        List[int]        = field(default_factory=list)
    tr_loss:   List[float]      = field(default_factory=list)
    tr_acc:    List[float]      = field(default_factory=list)
    te_loss:   List[float]      = field(default_factory=list)
    te_acc:    List[float]      = field(default_factory=list)
    n_params:  List[int]        = field(default_factory=list)
    sizes:     List[list]       = field(default_factory=list)
    eff_rank:  List[float]      = field(default_factory=list)
    fisher:    List[float]      = field(default_factory=list)
    mi:        List[float]      = field(default_factory=list)
    rep_id:    List[float]      = field(default_factory=list)
    data_id:   float            = 0.0
    growths:   List[Tuple]      = field(default_factory=list)
    elapsed_s: float            = 0.0

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  TRAINING SUMMARY",
            "=" * 60,
            f"  Epochs     : {len(self.ep)}",
            f"  Elapsed    : {self.elapsed_s:.1f}s",
            f"  Growths    : {[(e, k) for e, k, _, _ in self.growths]}",
            f"  Params     : {self.n_params[0]:,} → {self.n_params[-1]:,}  "
            f"({self.n_params[-1] / max(self.n_params[0], 1):.0f}×)",
            f"  Train acc  : {self.tr_acc[0]*100:.1f}% → {self.tr_acc[-1]*100:.1f}%",
            f"  Test  acc  : {self.te_acc[0]*100:.1f}% → {self.te_acc[-1]*100:.1f}%",
            f"  Eff Rank   : {self.eff_rank[0]:.2f} → {self.eff_rank[-1]:.2f}",
            f"  I(T;Y)     : {self.mi[0]:.3f} → {self.mi[-1]:.3f} nats",
            f"  Rep ID     : {self.rep_id[0]:.2f} → {self.rep_id[-1]:.2f}"
            f"  (Data ID = {self.data_id:.2f})",
            f"  ID gap     : {self.data_id - self.rep_id[0]:.2f} →"
            f" {self.data_id - self.rep_id[-1]:.2f}",
            "=" * 60,
        ]
        return "\n".join(lines)


# ── Trainer ───────────────────────────────────────────────────────────────────

class Trainer:
    """
    Orchestrates the neuroplasticity training loop.

    Parameters
    ----------
    model       : initial GrowingMLP
    controller  : NeuroplasticityController
    dataset     : CIFARAnalog (or any object with X_tr, y_tr, X_te, y_te)
    max_epochs  : total training epochs (default 65)
    lr_init     : initial learning rate (default 2.5e-3)
    lr_post     : learning rate after growth (default 1.8e-3)
    l2          : L2 weight decay (default 5e-4)
    batch       : mini-batch size (default 256)
    width_delta : neurons added per hidden layer per width growth (default 12)
    it_n_sub    : fixed subset size for IT metrics per epoch (default 300)
    fi_n_sub    : fixed subset size for Fisher trace (default 256)
    seed        : random seed for fixed analysis subsets (default 77)
    verbose     : print per-epoch table if True (default True)
    """

    def __init__(
        self,
        model:       GrowingMLP,
        controller:  NeuroplasticityController,
        dataset:     CIFARAnalog,
        max_epochs:  int   = 65,
        lr_init:     float = 2.5e-3,
        lr_post:     float = 1.8e-3,
        l2:          float = 5e-4,
        batch:       int   = 256,
        width_delta: int   = 12,
        it_n_sub:    int   = 300,
        fi_n_sub:    int   = 256,
        seed:        int   = 77,
        verbose:     bool  = True,
    ) -> None:
        self.model       = model
        self.ctrl        = controller
        self.ds          = dataset
        self.max_epochs  = max_epochs
        self.lr          = lr_init
        self.lr_post     = lr_post
        self.l2          = l2
        self.batch       = batch
        self.width_delta = width_delta
        self.it_n_sub    = it_n_sub
        self.fi_n_sub    = fi_n_sub
        self.verbose     = verbose

        # Fixed analysis subsets — same samples every epoch for comparability
        rng      = np.random.RandomState(seed)
        it_n_sub = min(it_n_sub, len(dataset.y_te))
        fi_n_sub = min(fi_n_sub, len(dataset.y_tr))
        it_idx   = rng.choice(len(dataset.y_te), it_n_sub, replace=False)
        fi_idx   = rng.choice(len(dataset.y_tr), fi_n_sub, replace=False)
        self._X_it = dataset.X_te[it_idx]
        self._y_it = dataset.y_te[it_idx]
        self._X_fi = dataset.X_tr[fi_idx]
        self._y_fi = dataset.y_tr[fi_idx]

    def run(self) -> History:
        """
        Execute the full training run.

        Returns
        -------
        History object with all per-epoch metrics and growth events.
        """
        hist = History()

        # Fixed Data ID — computed once before training
        hist.data_id = twonn_id(self.ds.X_tr, n_sub=150)

        if self.verbose:
            print(f"\n{repr(self.model)}")
            print(f"\nData ID = {hist.data_id:.2f}  "
                  f"(complexity ceiling)\n")
            print("─" * 100)
            print(
                f"{'Ep':>3}  {'TrLoss':>8} {'TrAcc':>7}  "
                f"{'TeLoss':>8} {'TeAcc':>7}  "
                f"{'EffRank':>8} {'FIM(log)':>9} {'I(T;Y)':>7} {'RepID':>6}   Event"
            )
            print("─" * 100)

        t0 = time.time()

        for ep in range(1, self.max_epochs + 1):

            # ── Train ─────────────────────────────────────────────────────
            tr_loss, tr_acc = self.model.fit_epoch(
                self.ds.X_tr, self.ds.y_tr,
                lr=self.lr, l2=self.l2, batch=self.batch,
            )
            te_loss, te_acc = self.model.evaluate(self.ds.X_te, self.ds.y_te)

            # ── IT metrics ────────────────────────────────────────────────
            er  = mean_eff_rank(self.model)
            fi  = fisher_trace(self.model, self._X_fi, self._y_fi)
            mi  = mutual_info_ib(self.model, self._X_it, self._y_it)
            rid = rep_id(self.model, self._X_it)

            # ── Record ────────────────────────────────────────────────────
            hist.ep.append(ep)
            hist.tr_loss.append(tr_loss);  hist.tr_acc.append(tr_acc)
            hist.te_loss.append(te_loss);  hist.te_acc.append(te_acc)
            hist.n_params.append(self.model.n_params())
            hist.sizes.append(self.model.sizes.copy())
            hist.eff_rank.append(er)
            hist.fisher.append(np.log10(fi + 1e-20))
            hist.mi.append(mi)
            hist.rep_id.append(rid)

            # ── Growth check ──────────────────────────────────────────────
            event_str = ""
            if self.ctrl.should_grow(ep, tr_acc):
                old_repr = repr(self.model)

                if self.ctrl.prefer_depth():
                    kind        = "depth"
                    self.model  = grow_depth(self.model)
                    tag         = "↕ DEPTH"
                else:
                    kind        = "width"
                    self.model  = grow_width(self.model, delta=self.width_delta)
                    tag         = "↔ WIDTH"

                self.lr = self.lr_post
                self.ctrl.record(kind, ep, tr_acc)
                hist.growths.append((ep, kind, old_repr, repr(self.model)))
                event_str = f"  [{tag}] → {repr(self.model)}"

            if self.verbose:
                print(
                    f"{ep:3d}  {tr_loss:8.4f} {tr_acc*100:6.2f}%  "
                    f"{te_loss:8.4f} {te_acc*100:6.2f}%  "
                    f"{er:8.3f} {np.log10(fi+1e-20):9.3f} "
                    f"{mi:7.3f} {rid:6.2f}  {event_str}"
                )

        hist.elapsed_s = time.time() - t0

        if self.verbose:
            print("─" * 100)
            print(f"\n{hist.summary()}")

        return hist
