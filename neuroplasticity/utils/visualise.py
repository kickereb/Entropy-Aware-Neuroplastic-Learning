"""
neuroplasticity.utils.visualise
=================================
Plot 3-row results figure for neuroplasticity experiments.
"""
from __future__ import annotations
import numpy as np

BG  = '#0D1117'
PBG = '#161B22'
C   = {'tr': '#58A6FF', 'te': '#3FB950', 'loss': '#F0883E', 'par': '#BC8CFF',
       'w': '#FF7B72', 'd': '#FFA657', 'er': '#79C0FF', 'fi': '#D2A8FF',
       'mi': '#7EE787', 'ri': '#58A6FF', 'di': '#F85149', 'gap': '#F8514933',
       'bl': '#8B949E'}


def _style_ax(ax):
    ax.set_facecolor(PBG)
    ax.tick_params(colors='#8B949E', labelsize=8)
    for s in ax.spines.values():
        s.set_color('#30363D')
    ax.grid(True, alpha=0.15, color='#484F58')


def plot_results(hist, out_path='neuroplasticity_results.png', baseline_hist=None,
                 title_prefix='Neuroplasticity'):
    """
    Generate a 3×3 figure from a History object.

    Parameters
    ----------
    hist          : History object with epoch, tr_acc, te_acc, etc.
    out_path      : save path for the PNG
    baseline_hist : optional second History for comparison
    title_prefix  : prefix for figure titles
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    eps = hist.epoch
    growths = hist.growths

    # Row 1: Accuracy, Loss, Params
    fig1, ax1 = plt.subplots(1, 3, figsize=(16, 4.5), facecolor=BG)
    fig1.subplots_adjust(wspace=0.35, top=0.85, bottom=0.18)
    fig1.suptitle(f'{title_prefix}: Accuracy, Loss & Capacity',
                  color='white', fontsize=14, fontweight='bold')

    a = ax1[0]; _style_ax(a)
    a.plot(eps, hist.tr_acc, color=C['tr'], lw=2, label='Train')
    a.plot(eps, hist.te_acc, color=C['te'], lw=2, label='Test')
    if baseline_hist:
        a.plot(baseline_hist.epoch, baseline_hist.tr_acc, color=C['tr'],
               lw=1.2, ls='--', alpha=0.5, label='Train (baseline)')
        a.plot(baseline_hist.epoch, baseline_hist.te_acc, color=C['te'],
               lw=1.2, ls='--', alpha=0.5, label='Test (baseline)')
    for g in growths:
        a.axvline(g[0], color=C['w'] if g[1] == 'width' else C['d'],
                  lw=1.3, ls='--', alpha=0.7)
    a.set_title('Accuracy', color='white', fontsize=11, fontweight='bold')
    a.set_xlabel('Epoch', color='#C9D1D9', fontsize=9)
    a.legend(fontsize=7, facecolor=PBG, labelcolor='white', framealpha=0.7)

    a = ax1[1]; _style_ax(a)
    a.plot(eps, hist.loss, color=C['loss'], lw=2)
    for g in growths:
        a.axvline(g[0], color=C['w'] if g[1] == 'width' else C['d'],
                  lw=1.3, ls='--', alpha=0.7)
    a.set_title('Training Loss', color='white', fontsize=11, fontweight='bold')
    a.set_xlabel('Epoch', color='#C9D1D9', fontsize=9)

    a = ax1[2]; _style_ax(a)
    a.plot(eps, [p / 1000 for p in hist.n_params], color=C['par'], lw=2, label='Model')
    if baseline_hist:
        a.plot(baseline_hist.epoch, [p / 1000 for p in baseline_hist.n_params],
               color=C['bl'], lw=1.5, ls='--', label='Baseline')
    a.axhline(272, color='#F85149', ls=':', lw=1.5, alpha=0.6, label='ResNet-20 (272K)')
    for g in growths:
        a.axvline(g[0], color=C['w'] if g[1] == 'width' else C['d'],
                  lw=1.3, ls='--', alpha=0.7)
    a.set_title('Parameters (K)', color='white', fontsize=11, fontweight='bold')
    a.set_xlabel('Epoch', color='#C9D1D9', fontsize=9)
    a.legend(fontsize=7, facecolor=PBG, labelcolor='white', framealpha=0.7)

    base = out_path.rsplit('.', 1)[0]
    fig1.savefig(f'{base}_row1.png', dpi=140, bbox_inches='tight', facecolor=BG)
    plt.close(fig1)

    # Row 2: IT signals
    fig2, ax2 = plt.subplots(1, 3, figsize=(16, 4.5), facecolor=BG)
    fig2.subplots_adjust(wspace=0.35, top=0.85, bottom=0.18)
    fig2.suptitle(f'{title_prefix}: IT Signals',
                  color='white', fontsize=14, fontweight='bold')

    for i, (k, label, c) in enumerate([
        ('eff_rank', 'Effective Rank', C['er']),
        ('fisher', 'Fisher Trace', C['fi']),
        ('mi', 'I(T;Y)', C['mi']),
    ]):
        a = ax2[i]; _style_ax(a)
        a.plot(eps, getattr(hist, k), color=c, lw=2)
        for g in growths:
            a.axvline(g[0], color=C['w'] if g[1] == 'width' else C['d'],
                      lw=1.3, ls='--', alpha=0.7)
        a.set_title(label, color='white', fontsize=11, fontweight='bold')
        a.set_xlabel('Epoch', color='#C9D1D9', fontsize=9)

    fig2.savefig(f'{base}_row2.png', dpi=140, bbox_inches='tight', facecolor=BG)
    plt.close(fig2)

    # Row 3: Complexity gap + Architecture
    fig3, ax3 = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG,
                              gridspec_kw={'width_ratios': [2, 1]})
    fig3.subplots_adjust(wspace=0.3, top=0.85, bottom=0.18)
    fig3.suptitle(f'{title_prefix}: Complexity Gap & Architecture',
                  color='white', fontsize=14, fontweight='bold')

    a = ax3[0]; _style_ax(a)
    a.plot(eps, hist.rep_id, color=C['ri'], lw=2.5, label='Rep ID')
    a.axhline(hist.data_id, color=C['di'], ls='--', lw=2,
              label=f'Data ID={hist.data_id:.1f}')
    a.fill_between(eps, hist.rep_id, hist.data_id,
                   where=[r < hist.data_id for r in hist.rep_id],
                   color=C['gap'], alpha=0.3, label='Gap')
    for g in growths:
        a.axvline(g[0], color=C['w'] if g[1] == 'width' else C['d'],
                  lw=1.3, ls='--', alpha=0.7)
    a.set_title('Neuroplasticity Signal', color='white', fontsize=11, fontweight='bold')
    a.set_xlabel('Epoch', color='#C9D1D9', fontsize=9)
    a.legend(fontsize=8, facecolor=PBG, labelcolor='white', framealpha=0.7)

    a = ax3[1]; _style_ax(a)
    a.plot(eps, hist.width, color=C['w'], lw=2, label='Width')
    a2 = a.twinx()
    a2.plot(eps, hist.n_blocks, color=C['d'], lw=2, ls='--', label='Blocks')
    a2.tick_params(colors='#8B949E', labelsize=8)
    a.set_title('Architecture', color='white', fontsize=11, fontweight='bold')
    a.set_xlabel('Epoch', color='#C9D1D9', fontsize=9)
    a.set_ylabel('Width', color='#C9D1D9', fontsize=9)
    a2.set_ylabel('Blocks', color=C['d'], fontsize=9)
    lines1, labels1 = a.get_legend_handles_labels()
    lines2, labels2 = a2.get_legend_handles_labels()
    a.legend(lines1 + lines2, labels1 + labels2, fontsize=8,
             facecolor=PBG, labelcolor='white', framealpha=0.7)

    fig3.savefig(f'{base}_row3.png', dpi=140, bbox_inches='tight', facecolor=BG)
    plt.close(fig3)

    print(f"[✓] Figures saved: {base}_row{{1,2,3}}.png")
    return out_path
