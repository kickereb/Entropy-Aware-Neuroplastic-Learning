#!/usr/bin/env python3
"""
experiments/run_phase3_skip.py
================================
Phase 3 experiment: SkipMLP with dataset-informed initialisation.

Usage:
    python experiments/run_phase3_skip.py
    python experiments/run_phase3_skip.py --epochs 120 --width-delta 16
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuroplasticity.data import CIFARAnalog
from neuroplasticity.models import SkipMLP
from neuroplasticity.growth import NeuroplasticityController, grow_width_skip, grow_depth_skip
from neuroplasticity.training import Trainer
from neuroplasticity.init import dataset_informed_init
from neuroplasticity.utils import plot_results


def parse_args():
    p = argparse.ArgumentParser(description="Phase 3: SkipMLP + Dataset-Informed Init")
    p.add_argument("--n-train",     type=int,   default=12_000)
    p.add_argument("--n-test",      type=int,   default=4_000)
    p.add_argument("--epochs",      type=int,   default=90)
    p.add_argument("--init-width",  type=int,   default=16)
    p.add_argument("--init-blocks", type=int,   default=2)
    p.add_argument("--width-delta", type=int,   default=16)
    p.add_argument("--max-params",  type=int,   default=280_000)
    p.add_argument("--lr",          type=float, default=5e-3)
    p.add_argument("--ref-width",   type=int,   default=48)
    p.add_argument("--out-dir",     type=str,   default="outputs")
    p.add_argument("--seed",        type=int,   default=42)
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    # Dataset
    print(f"\n[Data] Generating {args.n_train:,} train / {args.n_test:,} test ...")
    ds = CIFARAnalog(n_train=args.n_train, n_test=args.n_test)
    print(f"       {ds}")

    # Model
    model = SkipMLP(ds.feat_dim, args.init_width, args.init_blocks, ds.n_classes)
    print(f"\n[Model] {model}")

    # Dataset-informed init
    dataset_informed_init(model, ds.X_tr, ds.y_tr, ds.X_te, ds.y_te,
                          ref_width=args.ref_width)

    # Gradient check
    rel_err = SkipMLP.gradient_check()
    print(f"[GradCheck] Relative error: {rel_err:.2e} ✓")

    # Controller
    ctrl = NeuroplasticityController(max_params=args.max_params)

    # Growth functions
    grow_w = lambda m: grow_width_skip(m, delta=args.width_delta)
    grow_d = lambda m: grow_depth_skip(m)

    # Train
    trainer = Trainer(model, ctrl, ds, grow_fn=grow_w, depth_fn=grow_d,
                      max_epochs=args.epochs, lr=args.lr)
    hist = trainer.run()

    # Save
    os.makedirs(args.out_dir, exist_ok=True)

    json_path = os.path.join(args.out_dir, "phase3_history.json")
    serialisable = {
        k: [float(x) for x in v] if isinstance(v, list) else float(v)
        for k, v in hist.__dict__.items() if k != "growths"
    }
    serialisable["growths"] = [
        {"epoch": e, "kind": k, "before": o, "after": n, "reason": r}
        for e, k, o, n, r in hist.growths
    ]
    with open(json_path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"\n[Save] History → {json_path}")

    try:
        fig_path = os.path.join(args.out_dir, "phase3_results.png")
        plot_results(hist, out_path=fig_path, title_prefix="Phase 3: SkipMLP")
    except ImportError as e:
        print(f"[Warn] Figure skipped — {e}")


if __name__ == "__main__":
    main()
