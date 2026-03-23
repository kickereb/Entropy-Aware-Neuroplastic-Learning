# Neuroplasticity-Inspired Growing Neural Network

A neural network that grows new neurons in direct response to struggle — simulating biological neuroplasticity using four information-theoretic signals.

## Concept

When a human learns something difficult, they literally form new synaptic connections. This experiment operationalises that principle: a network starts structurally too small to solve the task, then autonomously grows its architecture when it detects it is struggling, guided by four information-theoretic signals measuring the mismatch between model capacity and data complexity.

```
Epoch 1:  MLP[96,  4,  4, 10]  —   458 params  — Train: 14.5%
Epoch 16: MLP[96, 16, 16, 10]  — 1,994 params  — GROWTH ↔ (plateau detected)
Epoch 27: MLP[96, 28, 28, 10]  — 3,818 params  — GROWTH ↔
Epoch 35: MLP[96, 40, 40, 10]  — 5,930 params  — GROWTH ↔
Epoch 42: MLP[96, 52, 52, 10]  — 8,330 params  — GROWTH ↔
Epoch 53: MLP[96, 64, 64, 10]  — 11,018 params — GROWTH ↔
Epoch 60: MLP[96, 64, 64, 10]  — 13,994 params — GROWTH ↔
Epoch 65:                       — 13,994 params — Train: 89.1% / Test: 75.8%
```

## Four Information-Theoretic Signals

| Signal | What it measures | Neuroplasticity signature |
|--------|-----------------|--------------------------|
| **Effective Rank** | Spectral diversity of weight matrices — how many independent feature directions are used | Sharp jump at each growth event |
| **Fisher Information Trace** | E[(∂L/∂θ)²] — how hard every parameter is working | Dip-then-rise at each growth (new neurons idle, then specialise) |
| **I(T ; Y)** | Mutual information between representations and labels (Information Bottleneck) | Rises as representations become class-discriminative |
| **TwoNN ID** | Intrinsic dimensionality of learned representations vs. data manifold | Gap (Data ID − Rep ID) closes with each growth event |

## Repository Structure

```
neuroplasticity-growing-nn/
│
├── neuroplasticity/           # Core library
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── cifar_analog.py    # CIFAR-10 analog dataset generator
│   ├── models/
│   │   ├── __init__.py
│   │   └── growing_mlp.py     # GrowingMLP — pure NumPy, float32
│   ├── growth/
│   │   ├── __init__.py
│   │   ├── operators.py       # grow_width(), grow_depth() — Net2Net style
│   │   └── controller.py      # NeuroplasticityController — struggle detector
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── effective_rank.py  # SVD spectral entropy
│   │   ├── fisher.py          # Fisher Information Matrix trace
│   │   ├── mutual_info.py     # I(T;Y) via Information Bottleneck
│   │   └── twonn.py           # TwoNN intrinsic dimensionality estimator
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py         # Main training loop + history logging
│   └── utils/
│       ├── __init__.py
│       └── visualise.py       # 9-panel results figure (3 memory-safe strips)
│
├── experiments/
│   └── run_cifar_analog.py    # Full end-to-end experiment script
│
├── notebooks/
│   └── neuroplasticity_colab.py  # PyTorch version ready for Google Colab
│
├── outputs/                   # Generated figures and results (git-ignored)
│
├── requirements.txt
├── requirements_torch.txt     # For the Colab/PyTorch version
└── README.md
```

## Quick Start

### NumPy version (no GPU required, runs anywhere)
```bash
pip install -r requirements.txt
python experiments/run_cifar_analog.py
```

### PyTorch version (recommended for real CIFAR-10, Google Colab)
```bash
pip install -r requirements_torch.txt
# Upload notebooks/neuroplasticity_colab.py to Colab
# Runtime → Run all
```

## Dataset

The NumPy version uses a synthetic CIFAR-10 analog:
- **3,072-dim** ambient space (32×32×3, same as CIFAR-10)
- **96-dim** CNN-style patch features (MLP input)
- **10 classes** with quadratic cross-product label functions — a network with 4 hidden neurons structurally cannot solve this
- **10,000 train / 3,000 test** samples

The PyTorch Colab notebook uses **real CIFAR-10** downloaded via `torchvision`.

## Growth Protocol

**Width growth** (primary): Add `WIDTH_DELTA=12` neurons to every hidden layer.  
Fires when: `train_acc < ACC_THRESH` AND `|Δacc| < DELTA_ACC` AND `epochs_since_last_growth ≥ COOLDOWN`

**Depth growth** (secondary): Insert a new hidden layer (half the width of the last hidden).  
Fires instead of width when: ≥3 width events each produced <1.5% accuracy gain.

Both use **Net2Net-style weight copying** — existing weights preserved exactly, new neurons initialised small.

## Key Results

| Metric | Epoch 1 | Epoch 65 | Change |
|--------|---------|----------|--------|
| Architecture | [96,4,4,10] | [96,64,64,10] | — |
| Parameters | 458 | 13,994 | **31×** |
| Train accuracy | 14.5% | 89.1% | +74.6 pp |
| Test accuracy | 18.6% | 75.8% | +57.2 pp |
| Effective rank | 3.57 | 31.45 | **8.8×** |
| Rep ID (TwoNN) | 1.72 | 9.01 | Gap 47% closed |

## References

- **Net2Net**: Chen et al. (2016). *Net2Net: Accelerating Learning via Knowledge Transfer*. ICLR.
- **TwoNN**: Facco et al. (2017). *Estimating the intrinsic dimension of datasets by a minimal neighborhood information*. Scientific Reports.
- **Information Bottleneck**: Tishby & Schwartz-Ziv (2017). *Opening the Black Box of Deep Neural Networks via Information*. ITW.
- **Effective Rank**: Roy & Vetterli (2007). *The effective rank: A measure of effective dimensionality*. EUSIPCO.

## License

MIT
