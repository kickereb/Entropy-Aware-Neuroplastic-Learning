#!/usr/bin/env python3
"""
experiments/run_baseline_mlp.py
=================================
Baseline experiment: Plain GrowingMLP (no skip connections).
Reproduces Phase 1 results.

Usage:
    python experiments/run_baseline_mlp.py
    python experiments/run_baseline_mlp.py --epochs 65 --init-hidden 4
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuroplasticity.data import CIFARAnalog
from neuroplasticity.models import GrowingMLP
from neuroplasticity.growth import NeuroplasticityController, grow_width
from neuroplasticity.training import Trainer
from neuroplasticity.utils import plot_results


def parse_args():
    p = argparse.ArgumentParser(description="Baseline: Plain Growing MLP")
    p.add_argument("--n-train",     type=int,   default=12_000)
    p.add_argument("--n-test",      type=int,   default=4_000)
    p.add_argument("--epochs",      type=int,   default=65)
    p.add_argument("--init-hidden", type=int,   default=4)
    p.add_argument("--n-hidden",    type=int,   default=2)
    p.add_argument("--width-delta", type=int,   default=12)
    p.add_argument("--max-params",  type=int,   default=300_000)
    p.add_argument("--lr",          type=float, default=5e-3)
    p.add_argument("--out-dir",     type=str,   default="outputs")
    p.add_argument("--seed",        type=int,   default=42)
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    ds = CIFARAnalog(n_train=args.n_train, n_test=args.n_test)
    print(f"\n[Data] {ds}")

    sizes = [ds.feat_dim] + [args.init_hidden] * args.n_hidden + [ds.n_classes]
    model = GrowingMLP(sizes)
    print(f"[Model] {model}")

    ctrl = NeuroplasticityController(max_params=args.max_params)
    grow_fn = lambda m: grow_width(m, delta=args.width_delta)

    trainer = Trainer(model, ctrl, ds, grow_fn=grow_fn,
                      max_epochs=args.epochs, lr=args.lr)
    hist = trainer.run()

    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, "baseline_history.json")
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

    try:
        plot_results(hist, out_path=os.path.join(args.out_dir, "baseline_results.png"),
                     title_prefix="Baseline: Plain MLP")
    except ImportError as e:
        print(f"[Warn] Figure skipped — {e}")


if __name__ == "__main__":
    main()
