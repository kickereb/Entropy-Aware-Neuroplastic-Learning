# Neuroplasticity-Inspired Growing Neural Network (v2)

A neural network that grows new neurons in direct response to struggle — simulating biological neuroplasticity using four information-theoretic signals.

## What's New in v2

- **SkipMLP**: Residual-block architecture with verified backward pass (gradient check < 1e-10)
- **Dataset-Informed Init**: PCsInit + teacher weight projection + LSUV calibration
- **Modular `init/` package**: Reusable initialisation components
- **Balanced dataset**: Perfectly balanced 10-class synthetic data

## Concept

When a human learns something difficult, they literally form new synaptic connections. This experiment operationalises that principle: a network starts structurally too small, then autonomously grows when it detects struggle, guided by four IT signals.

## Four Information-Theoretic Signals

| Signal | Measures | Neuroplasticity Signature |
|--------|----------|--------------------------|
| **Effective Rank** | SVD spectral diversity of weights | Sharp jump at each growth event |
| **Fisher Trace** | E[(∂L/∂θ)²] — parameter utilisation | Dip-then-rise (new neurons idle, then specialise) |
| **I(T;Y)** | Mutual information reps↔labels | Rises as representations become class-discriminative |
| **TwoNN ID** | Intrinsic dimensionality of reps vs data | Complexity gap narrows at each growth |

## Repository Structure

```
neuroplasticity-growing-nn/
│
├── neuroplasticity/              # Core library
│   ├── data/cifar_analog.py      # Balanced CIFAR-10 analog dataset
│   ├── models/
│   │   ├── growing_mlp.py        # Plain MLP with Adam
│   │   └── skip_mlp.py           # SkipMLP with residual blocks (NEW)
│   ├── growth/
│   │   ├── operators.py          # grow_width/depth for both architectures
│   │   └── controller.py         # NeuroplasticityController
│   ├── metrics/
│   │   ├── effective_rank.py     # SVD spectral entropy
│   │   ├── fisher.py             # Fisher Information Trace
│   │   ├── mutual_info.py        # I(T;Y) via Information Bottleneck
│   │   └── twonn.py              # TwoNN intrinsic dimensionality
│   ├── init/                     # Dataset-informed initialisation (NEW)
│   │   ├── pcs_init.py           # PCA-based first layer init
│   │   ├── teacher.py            # Reference model + SVD projection
│   │   ├── lsuv.py               # LSUV variance calibration
│   │   └── pipeline.py           # Combined init pipeline
│   ├── training/trainer.py       # Trainer + History
│   └── utils/visualise.py        # Multi-panel figure generation
│
├── experiments/
│   ├── run_baseline_mlp.py       # Phase 1: Plain MLP
│   └── run_phase3_skip.py        # Phase 3: SkipMLP + init (NEW)
│
├── requirements.txt
└── README.md
```

## Quick Start

```bash
pip install numpy scipy matplotlib

# Phase 1: Plain MLP baseline
python experiments/run_baseline_mlp.py

# Phase 3: SkipMLP with dataset-informed init
python experiments/run_phase3_skip.py

# Custom config
python experiments/run_phase3_skip.py --epochs 120 --width-delta 16 --max-params 280000
```

## Key Results

### Phase 1 (Plain MLP)
- 458 → 13,994 params across 6 growth events
- Train: 89.1%, Test: 75.8%
- Complexity gap closed: 47%

### Phase 2 (Plain MLP, CIFAR-10)
- 458 → 321,994 params, test ceiling at ~77%
- 14.5 pp gap vs ResNet-20 = convolutional inductive bias

### Phase 3 (SkipMLP + Dataset-Informed Init)
- 2,810 → 113,242 params across 8 events (6 width + 2 depth)
- **+17.9 pp** test accuracy advantage over plain MLP
- Effective rank: 12.8 → 50.3

## References

- **Net2Net**: Chen et al. (2016). *Accelerating Learning via Knowledge Transfer*. ICLR.
- **TwoNN**: Facco et al. (2017). *Estimating the intrinsic dimension of datasets*. Scientific Reports.
- **Information Bottleneck**: Tishby & Schwartz-Ziv (2017). *Opening the Black Box*. ITW.
- **Effective Rank**: Roy & Vetterli (2007). EUSIPCO.
- **PCsInit**: arXiv:2501.19114 (2025).
- **LSUV**: Mishkin & Matas (2015). *All you need is a good init*.

## License

MIT
