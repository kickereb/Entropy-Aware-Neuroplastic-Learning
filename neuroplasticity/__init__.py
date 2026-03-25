"""
neuroplasticity
===============
A neural network that grows new neurons in direct response to struggle,
guided by four information-theoretic signals.

Subpackages
-----------
data        — CIFAR-10 analog dataset generator
models      — GrowingMLP (plain) + SkipMLP (residual blocks)
growth      — Net2Net-style grow_width / grow_depth + NeuroplasticityController
metrics     — Effective Rank, Fisher Trace, I(T;Y), TwoNN ID
training    — Main training loop and history logger
init        — Dataset-informed initialisation (PCsInit, LSUV, teacher projection)
utils       — Visualisation (multi-panel figure)
"""

__version__ = "2.0.0"
__author__  = "Neuroplasticity Experiment"
