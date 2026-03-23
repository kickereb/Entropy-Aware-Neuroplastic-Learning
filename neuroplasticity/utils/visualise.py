"""
neuroplasticity.utils.visualise
=================================
9-panel results figure rendered in three memory-safe horizontal strips
that are stitched into a single PNG using Pillow.

Why strips instead of a single figure?
---------------------------------------
A single 21×16 matplotlib figure with 9 filled panels at 150 dpi can
exceed 4 GB of RAM during the rasterisation pass on resource-constrained
machines. Building each row separately caps peak usage at ~400 MB per strip,
making the visualisation safe on Colab free-tier and CI runners.

Usage
-----
    from neuroplasticity.training import History
    from neuroplasticity.utils    import plot_results

    history = trainer.run()
    plot_results(history, out_path="outputs/results.png")

Output
------
A single PNG at the given path containing all 9 panels arranged as:

    Row 1  │ Accuracy │ Loss │ Parameter Count        │
    Row 2  │ Eff Rank │ Fisher Trace │ I(T;Y)          │
    Row 3  │ Neuroplasticity Gap (wide)   │ Architecture │
"""

from __future__ import annotations

import os
import tempfile
from typing import TYPE_CHECKING

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

if TYPE_CHECKING:
    from neuroplasticity.training.trainer import History


# ── Colour palette ────────────────────────────────────────────────────────────
_C = {
    "train"  : "#4fc3f7",
    "test"   : "#ef9a9a",
    "width"  : "#69f0ae",
    "depth"  : "#ce93d8",
    "data"   : "#90caf9",
    "param"  : "#80deea",
    "rank"   : "#ffcc80",
    "fisher" : "#ff8a65",
    "mi"     : "#b39ddb",
    "id"     : "#4dd0e1",
}
_BG  = "#0d0d1c"
_PBG = "#131330"
_TC  = dict(color="white", fontsize=10, fontweight="bold")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sty(ax):
    ax.set_facecolor(_PBG)
    ax.tick_params(colors="#ccd", labelsize=8)
    for sp in ax.spines.values():
        sp.set_color("#252545")
    return ax


def _vlines(ax, growths):
    """Annotate growth events on an axes object."""
    yl = ax.get_ylim()
    for ep_g, kind, *_ in growths:
        col = _C["width"] if kind == "width" else _C["depth"]
        ax.axvline(ep_g, color=col, lw=1.4, ls="--", alpha=0.82, zorder=5)
        ax.text(
            ep_g, yl[0] + (yl[1] - yl[0]) * 0.94,
            "W↔" if kind == "width" else "D↕",
            color=col, ha="center", va="top", fontsize=7.5, fontweight="bold",
            transform=ax.get_xaxis_transform(),
        )


# ── Strip builders ────────────────────────────────────────────────────────────

def _strip1(history: "History", tmp_path: str) -> str:
    """Row 1: Accuracy · Loss · Parameter Count"""
    eps = history.ep
    fig, axes = plt.subplots(1, 3, figsize=(21, 5.2), facecolor=_BG)
    fig.subplots_adjust(wspace=0.35, left=0.06, right=0.97, top=0.88, bottom=0.13)

    # Accuracy
    ax = _sty(axes[0])
    ax.plot(eps, [a * 100 for a in history.tr_acc], _C["train"], lw=2, label="Train Acc")
    ax.plot(eps, [a * 100 for a in history.te_acc], _C["test"],  lw=2, ls="--", label="Test Acc")
    ax.axhline(88, color="#ffeb3b", lw=1.2, ls=":", alpha=0.75, label="Growth threshold (88%)")
    _vlines(ax, history.growths)
    ax.set_title("Classification Accuracy", **_TC)
    ax.set_xlabel("Epoch", color="#aac", fontsize=9)
    ax.set_ylabel("Accuracy (%)", color="#aac", fontsize=9)
    ax.legend(fontsize=8, facecolor=_PBG, labelcolor="white", framealpha=0.65)

    # Loss
    ax = _sty(axes[1])
    ax.plot(eps, history.tr_loss, _C["train"], lw=2, label="Train Loss")
    ax.plot(eps, history.te_loss, _C["test"],  lw=2, ls="--", label="Test Loss")
    _vlines(ax, history.growths)
    ax.set_title("Cross-Entropy Loss", **_TC)
    ax.set_xlabel("Epoch", color="#aac", fontsize=9)
    ax.set_ylabel("Loss", color="#aac", fontsize=9)
    ax.legend(fontsize=8, facecolor=_PBG, labelcolor="white", framealpha=0.65)

    # Parameter count
    ax = _sty(axes[2])
    ax.step(eps, history.n_params, _C["param"], lw=2.3, where="post")
    ax.fill_between(eps, 0, history.n_params, step="post", color=_C["param"], alpha=0.14)
    prev_p = -1
    for ep_g, kind, *_ in history.growths:
        i   = history.ep.index(ep_g)
        p   = history.n_params[i]
        col = _C["width"] if kind == "width" else _C["depth"]
        if p - prev_p > 200:
            ax.annotate(
                f"{int(p):,}",
                xy=(ep_g, p), xytext=(ep_g + 1, p * 1.11),
                color=col, fontsize=7,
                arrowprops=dict(arrowstyle="->", color=col, lw=0.8),
            )
            prev_p = p
    _vlines(ax, history.growths)
    ax.set_title("Model Capacity (Parameters)", **_TC)
    ax.set_xlabel("Epoch", color="#aac", fontsize=9)
    ax.set_ylabel("Parameters", color="#aac", fontsize=9)

    n_w = sum(1 for _, k, *_ in history.growths if k == "width")
    n_d = sum(1 for _, k, *_ in history.growths if k == "depth")
    fig.suptitle(
        f"Neuroplasticity Growing Neural Network  ·  CIFAR-10 Analog  ·  "
        f"Width growth ×{n_w}  ·  Depth growth ×{n_d}",
        fontsize=11, fontweight="bold", color="white", y=0.99,
    )
    path = os.path.join(tmp_path, "strip1.png")
    plt.savefig(path, dpi=135, bbox_inches="tight", facecolor=_BG)
    plt.close()
    return path


def _strip2(history: "History", tmp_path: str) -> str:
    """Row 2: Effective Rank · Fisher Trace · I(T;Y)"""
    eps = history.ep
    fig, axes = plt.subplots(1, 3, figsize=(21, 5.2), facecolor=_BG)
    fig.subplots_adjust(wspace=0.35, left=0.06, right=0.97, top=0.85, bottom=0.13)

    # Effective rank
    ax = _sty(axes[0])
    ax.plot(eps, history.eff_rank, _C["rank"], lw=2, marker="o", ms=3)
    _vlines(ax, history.growths)
    if history.growths:
        i0 = history.ep.index(history.growths[0][0])
        ax.annotate(
            f"{history.eff_rank[i0]:.1f}→{history.eff_rank[i0+1]:.1f}",
            xy=(history.growths[0][0], history.eff_rank[i0 + 1]),
            xytext=(history.growths[0][0] + 7, history.eff_rank[i0 + 1] * 0.82),
            color=_C["rank"], fontsize=8,
            arrowprops=dict(arrowstyle="->", color=_C["rank"], lw=0.9),
        )
    ax.set_title("Mean Effective Rank\nexp(H(σ̃)) — independent feature directions", **_TC)
    ax.set_xlabel("Epoch", color="#aac", fontsize=9)
    ax.set_ylabel("Eff. Rank", color="#aac", fontsize=9)

    # Fisher trace
    ax = _sty(axes[1])
    ax.plot(eps, history.fisher, _C["fisher"], lw=2, marker="^", ms=3)
    _vlines(ax, history.growths)
    if history.growths and history.ep.index(history.growths[0][0]) + 2 < len(history.fisher):
        i0 = history.ep.index(history.growths[0][0])
        ax.annotate(
            "← idle neurons\n   then specialise →",
            xy=(history.growths[0][0] + 1, history.fisher[i0 + 1]),
            xytext=(history.growths[0][0] + 9, history.fisher[i0 + 1] - 0.3),
            color=_C["fisher"], fontsize=7.5,
            arrowprops=dict(arrowstyle="->", color=_C["fisher"], lw=0.9),
        )
    ax.set_title("Fisher Information Trace (log₁₀)\nE[(∂L/∂θ)²] — parameter sensitivity", **_TC)
    ax.set_xlabel("Epoch", color="#aac", fontsize=9)
    ax.set_ylabel("log₁₀ FIM", color="#aac", fontsize=9)

    # Mutual information
    ax = _sty(axes[2])
    ax.plot(eps, history.mi, _C["mi"], lw=2, marker="s", ms=3)
    _vlines(ax, history.growths)
    ax.set_title("Mutual Information I(T;Y) [nats]\nInformation Bottleneck — label compression", **_TC)
    ax.set_xlabel("Epoch", color="#aac", fontsize=9)
    ax.set_ylabel("MI (nats)", color="#aac", fontsize=9)

    path = os.path.join(tmp_path, "strip2.png")
    plt.savefig(path, dpi=135, bbox_inches="tight", facecolor=_BG)
    plt.close()
    return path


def _strip3_gap(history: "History", tmp_path: str) -> str:
    """Row 3, left: Neuroplasticity complexity-gap panel (wide)"""
    eps     = history.ep
    rid     = history.rep_id
    data_id = history.data_id
    dl      = [data_id] * len(eps)

    fig, ax = plt.subplots(1, 1, figsize=(14, 5.8), facecolor=_BG)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.80, bottom=0.13)
    ax.set_facecolor(_PBG)
    ax.tick_params(colors="#ccd", labelsize=8)
    for sp in ax.spines.values():
        sp.set_color("#252545")

    ax.fill_between(eps, rid, dl,
        where=[r < d for r, d in zip(rid, dl)],
        color="#f44336", alpha=0.18, label="Complexity gap  (model < data)")
    ax.fill_between(eps, rid, dl,
        where=[r >= d for r, d in zip(rid, dl)],
        color="#4caf50", alpha=0.22, label="Model complexity ≥ data")
    ax.plot(eps, rid, _C["id"], lw=2.8, marker="o", ms=4.5,
            label="Rep ID  (TwoNN on hidden activations)")
    ax.axhline(data_id, color=_C["data"], lw=2, ls="--",
               label=f"Data ID ≈ {data_id:.1f}  (fixed reference)")

    for ep_g, kind, *_ in history.growths:
        col = _C["width"] if kind == "width" else _C["depth"]
        ax.axvline(ep_g, color=col, lw=1.5, ls="--", alpha=0.85, zorder=5)
        ax.text(ep_g, 1.0, "W↔" if kind == "width" else "D↕",
                color=col, ha="center", va="top", fontsize=8, fontweight="bold",
                transform=ax.get_xaxis_transform())

    if rid:
        gap0 = data_id - rid[0]
        gapN = data_id - rid[-1]
        ax.annotate(
            f"Gap: {gap0:.1f}",
            xy=(eps[0], rid[0]), xytext=(eps[min(3, len(eps)-1)], rid[0] - 0.4),
            color="#ef9a9a", fontsize=8,
            arrowprops=dict(arrowstyle="->", color="#ef9a9a", lw=0.9),
        )
        pct = (1 - gapN / gap0) * 100 if gap0 > 0 else 0
        ax.annotate(
            f"Gap: {gapN:.1f}  ({pct:.0f}% closed)",
            xy=(eps[-1], rid[-1]), xytext=(eps[max(-12, -len(eps))], rid[-1] + 0.6),
            color="#69f0ae", fontsize=8,
            arrowprops=dict(arrowstyle="->", color="#69f0ae", lw=0.9),
        )

    ax.set_title(
        "🧠  NEUROPLASTICITY SIGNAL — Complexity Gap Closure\n"
        "TwoNN Intrinsic Dimensionality: Learned Representations vs. Data Manifold\n"
        "  Red = model cannot represent data structure  |  "
        "Shrinking gap = growth working",
        color="white", fontsize=10, fontweight="bold",
    )
    ax.set_xlabel("Epoch", color="#aac", fontsize=9)
    ax.set_ylabel("Intrinsic Dimensionality", color="#aac", fontsize=9)
    ax.legend(fontsize=8.5, facecolor=_PBG, labelcolor="white",
              framealpha=0.65, loc="lower right")

    path = os.path.join(tmp_path, "strip3_gap.png")
    plt.savefig(path, dpi=135, bbox_inches="tight", facecolor=_BG)
    plt.close()
    return path


def _strip3_arch(history: "History", width_delta: int, tmp_path: str) -> str:
    """Row 3, right: Architecture evolution stacked bar"""
    eps = history.ep
    fig, ax = plt.subplots(1, 1, figsize=(7, 5.8), facecolor=_BG)
    fig.subplots_adjust(left=0.12, right=0.97, top=0.85, bottom=0.13)
    ax.set_facecolor(_PBG)
    ax.tick_params(colors="#ccd", labelsize=8)
    for sp in ax.spines.values():
        sp.set_color("#252545")

    # Reconstruct sizes per epoch from history
    sizes_per_ep = []
    cur = list(history.sizes[0]) if history.sizes else [96, 4, 4, 10]
    gi  = 0
    for ep in eps:
        sizes_per_ep.append(cur[1:-1])
        if gi < len(history.growths) and history.growths[gi][0] == ep:
            kind = history.growths[gi][1]
            if kind == "width":
                cur = [cur[0]] + [s + width_delta for s in cur[1:-1]] + [cur[-1]]
            else:
                nh  = max(8, cur[-2] // 2)
                cur = cur[:-1] + [nh, cur[-1]]
            gi += 1

    max_layers = max(len(h) for h in sizes_per_ep)
    pal = plt.cm.plasma(np.linspace(0.15, 0.92, max_layers))
    bot = np.zeros(len(eps))
    for l in range(max_layers):
        vals = np.array([h[l] if l < len(h) else 0 for h in sizes_per_ep], float)
        ax.bar(eps, vals, bottom=bot, color=pal[l], label=f"H{l+1}", width=0.9)
        bot += vals

    for ep_g, kind, *_ in history.growths:
        col = _C["width"] if kind == "width" else _C["depth"]
        ax.axvline(ep_g, color=col, lw=1.4, ls="--", alpha=0.85)

    ax.set_title("Architecture Evolution\nCumulative hidden neurons per epoch",
                 color="white", fontsize=10, fontweight="bold")
    ax.set_xlabel("Epoch", color="#aac", fontsize=9)
    ax.set_ylabel("Total hidden neurons", color="#aac", fontsize=9)
    ax.legend(fontsize=7, facecolor=_PBG, labelcolor="white", ncol=2, framealpha=0.65)

    path = os.path.join(tmp_path, "strip3_arch.png")
    plt.savefig(path, dpi=135, bbox_inches="tight", facecolor=_BG)
    plt.close()
    return path


# ── Public interface ──────────────────────────────────────────────────────────

def plot_results(
    history:     "History",
    out_path:    str  = "outputs/neuroplasticity_results.png",
    width_delta: int  = 12,
) -> str:
    """
    Build and save the full 9-panel results figure.

    Renders three horizontal strips separately (memory-safe), then
    stitches them into a single PNG using Pillow.

    Parameters
    ----------
    history     : History object returned by Trainer.run()
    out_path    : destination path for the final PNG
    width_delta : neurons added per growth event (for architecture panel)

    Returns
    -------
    str — absolute path to the saved PNG

    Raises
    ------
    ImportError if Pillow is not installed (pip install Pillow)
    """
    if not _PIL_AVAILABLE:
        raise ImportError(
            "Pillow is required for plot_results().  "
            "Install with:  pip install Pillow"
        )

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        p1    = _strip1(history, tmp)
        p2    = _strip2(history, tmp)
        p_gap = _strip3_gap(history, tmp)
        p_arc = _strip3_arch(history, width_delta, tmp)

        # Combine gap + arch side-by-side at matching height
        s_gap = Image.open(p_gap)
        s_arc = Image.open(p_arc)
        s_arc_r = s_arc.resize(
            (int(s_arc.width * s_gap.height / s_arc.height), s_gap.height),
            Image.LANCZOS,
        )
        row3 = Image.new("RGB", (s_gap.width + s_arc_r.width, s_gap.height), (13, 13, 28))
        row3.paste(s_gap, (0, 0))
        row3.paste(s_arc_r, (s_gap.width, 0))

        # Scale all rows to the same width
        strips = [Image.open(p1), Image.open(p2), row3]
        W      = max(s.width for s in strips)
        scaled = [
            s.resize((W, int(s.height * W / s.width)), Image.LANCZOS)
            for s in strips
        ]

        # Stack vertically
        H_total = sum(s.height for s in scaled)
        canvas  = Image.new("RGB", (W, H_total), (13, 13, 28))
        y = 0
        for s in scaled:
            canvas.paste(s, (0, y))
            y += s.height

        canvas.save(out_path, quality=95)

    return os.path.abspath(out_path)
