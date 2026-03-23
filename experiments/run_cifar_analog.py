#!/usr/bin/env python3
"""
experiments/run_cifar_analog.py
=================================
End-to-end experiment: train a growing MLP on the CIFAR-10 analog dataset,
log four information-theoretic signals every epoch, and save the results figure.

Usage
-----
    python experiments/run_cifar_analog.py                  # default config
    python experiments/run_cifar_analog.py --epochs 80 --width-delta 16
    python experiments/run_cifar_analog.py --help

Output
------
    outputs/neuroplasticity_results.png   — 9-panel results figure
    outputs/neuroplasticity_history.json  — all metrics as JSON (for further analysis)
"""

import argparse
import json
import os
import sys

import numpy as np

# Allow running from repo root without install
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuroplasticity.data     import CIFARAnalog
from neuroplasticity.growth   import NeuroplasticityController
from neuroplasticity.models   import GrowingMLP
from neuroplasticity.training import Trainer
from neuroplasticity.utils    import plot_results


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Neuroplasticity Growing Neural Network — CIFAR-10 Analog",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--n-train",     type=int,   default=10_000, help="Training set size")
    p.add_argument("--n-test",      type=int,   default=3_000,  help="Test set size")
    p.add_argument("--epochs",      type=int,   default=65,     help="Total training epochs")
    p.add_argument("--batch",       type=int,   default=256,    help="Mini-batch size")
    p.add_argument("--lr",          type=float, default=2.5e-3, help="Initial learning rate")
    p.add_argument("--lr-post",     type=float, default=1.8e-3, help="LR after growth events")
    p.add_argument("--l2",          type=float, default=5e-4,   help="L2 weight decay")
    p.add_argument("--width-delta", type=int,   default=12,     help="Neurons added per width growth")
    p.add_argument("--acc-thresh",  type=float, default=0.88,   help="Growth accuracy threshold")
    p.add_argument("--delta-acc",   type=float, default=0.006,  help="Plateau threshold (per epoch)")
    p.add_argument("--cooldown",    type=int,   default=7,      help="Min epochs between growth events")
    p.add_argument("--init-hidden", type=int,   default=4,      help="Initial neurons per hidden layer")
    p.add_argument("--n-hidden",    type=int,   default=2,      help="Number of hidden layers")
    p.add_argument("--seed",        type=int,   default=42,     help="Global random seed")
    p.add_argument("--out-dir",     type=str,   default="outputs", help="Output directory")
    p.add_argument("--no-plot",     action="store_true",           help="Skip figure generation")
    p.add_argument("--quiet",       action="store_true",           help="Suppress per-epoch output")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    np.random.seed(args.seed)

    # ── Dataset ───────────────────────────────────────────────────────────────
    print(f"\n[Data]  Generating {args.n_train:,} train / {args.n_test:,} test samples ...")
    ds = CIFARAnalog(n_train=args.n_train, n_test=args.n_test)
    print(f"        {ds}")
    print(f"        Class balance: {np.bincount(ds.y_tr)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    sizes = (
        [ds.feat_dim]
        + [args.init_hidden] * args.n_hidden
        + [ds.n_classes]
    )
    model = GrowingMLP(sizes)
    print(f"\n[Model] {repr(model)}")
    print(f"        Intentionally tiny — guaranteed to underfit at initialisation")

    # ── Controller ────────────────────────────────────────────────────────────
    ctrl = NeuroplasticityController(
        acc_thresh  = args.acc_thresh,
        delta_acc   = args.delta_acc,
        cooldown    = args.cooldown,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer = Trainer(
        model       = model,
        controller  = ctrl,
        dataset     = ds,
        max_epochs  = args.epochs,
        lr_init     = args.lr,
        lr_post     = args.lr_post,
        l2          = args.l2,
        batch       = args.batch,
        width_delta = args.width_delta,
        verbose     = not args.quiet,
    )
    history = trainer.run()

    # ── Save results ──────────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)

    # JSON history
    json_path = os.path.join(args.out_dir, "neuroplasticity_history.json")
    serialisable = {
        k: (
            [float(x) if not isinstance(x, list) else x for x in v]
            if isinstance(v, list) else float(v)
        )
        for k, v in history.__dict__.items()
        if k != "growths"
    }
    serialisable["growths"] = [
        {"epoch": e, "kind": k, "before": o, "after": n}
        for e, k, o, n in history.growths
    ]
    with open(json_path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"\n[Save]  History → {json_path}")

    # Figure
    if not args.no_plot:
        fig_path = os.path.join(args.out_dir, "neuroplasticity_results.png")
        try:
            saved = plot_results(history, out_path=fig_path, width_delta=args.width_delta)
            print(f"[Save]  Figure  → {saved}")
        except ImportError as e:
            print(f"[Warn]  Figure skipped — {e}")

    print(f"\n{history.summary()}")


if __name__ == "__main__":
    main()
