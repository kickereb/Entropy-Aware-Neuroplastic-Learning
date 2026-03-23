"""
neuroplasticity
===============
A neural network that grows new neurons in direct response to struggle,
guided by four information-theoretic signals.

Subpackages
-----------
data        — CIFAR-10 analog dataset generator
models      — GrowingMLP (pure NumPy, float32)
growth      — Net2Net-style grow_width / grow_depth + NeuroplasticityController
metrics     — Effective Rank, Fisher Trace, I(T;Y), TwoNN ID
training    — Main training loop and history logger
utils       — Visualisation (9-panel figure rendered in memory-safe strips)
"""

__version__ = "1.0.0"
__author__  = "Neuroplasticity Experiment"
