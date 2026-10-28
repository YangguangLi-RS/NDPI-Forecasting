"""Refine the main L12 seasonal tracking figure without changing its data."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

import redraw_phenology as base

STEM = 'Fig9_phenology_dynamic_tracking_refined'


def plot_refined(table, obs, stem=STEM):
    font_manager.findfont('Times New Roman', fallback_to_default=False)
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 18,
                         'axes.titleweight': 'normal', 'axes.labelweight': 'normal',
                         'mathtext.fontset': 'stix', 'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.4), sharex=True, sharey='row')
    for idx, (ax, metric) in enumerate(zip(axes.flat, base.METRICS)):
        center = obs[metric].mean()
        ax.axvspan(center-30, center+30, color='#EEEEEE', lw=0, zorder=0)
        ax.axvline(center, color='#A4A4A4', ls=':', lw=1, zorder=1)
        for j, model in enumerate(base.MODELS):
            curve = table.loc[table.metric.eq(metric) & table.model.eq(model)].sort_values('obs_count')
            ax.plot(curve.cutoff_doy, curve.rmse, color=base.COLORS[j], ls=base.STYLES[j],
                    lw=1.75 if j < 4 else 2.05, zorder=3+j*.1)
            if j >= 4:
                # Marker positions alternate; every original update remains in the lines.
                positions = np.arange(3 if j == 4 else 6, len(curve), 6)
                ax.plot(curve.cutoff_doy.iloc[positions], curve.rmse.iloc[positions],
                        linestyle='None', marker='o' if j == 4 else 'D',
                        markersize=5.1, markerfacecolor='white',
                        markeredgecolor=base.COLORS[j], markeredgewidth=1.25, zorder=5)
        ax.set_title(f'({chr(97+idx)})  {metric.replace("_", "")}', fontsize=22, loc='left', pad=12)
        ax.set_xlim(0, 361)
        ax.set_xticks(np.arange(0, 361, 60))
        ax.tick_params(axis='both', labelsize=18, length=4, pad=6)
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_linewidth(.85)
        ax.grid(False)
        if idx % 3 == 0:
            ax.set_ylabel(f'{metric[:3]} RMSE (days)', fontsize=21, labelpad=8)
        if idx >= 3:
            ax.set_xlabel('Last observation date (DOY)', fontsize=19, labelpad=9)
    for row, phase in enumerate(('SOS', 'EOS')):
        maximum = table.loc[table.metric.str.startswith(phase), 'rmse'].max()
        main_limit = 16.5 if row == 0 else 13
        if maximum < main_limit:
            limit = main_limit
            ticks = [0, 4, 8, 12, 16] if row == 0 else [0, 3, 6, 9, 12]
        else:
            ticks = MaxNLocator(nbins=5, integer=True).tick_values(0, maximum * 1.06)
            ticks = ticks[ticks >= 0]
            limit = ticks[-1]
        axes[row, 0].set_ylim(0, limit)
        axes[row, 0].set_yticks(ticks)
        assert maximum < limit
    handles = []
    for j, (model, color, style) in enumerate(zip(base.MODELS, base.COLORS, base.STYLES)):
        handles.append(Line2D([0], [0], color=color, ls=style, lw=2,
                              marker=None if j < 4 else ('o' if j == 4 else 'D'),
                              markersize=5.1, markerfacecolor='white', markeredgewidth=1.2,
                              label=model))
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.535, .012),
               ncol=3, frameon=False, fontsize=18, handlelength=3,
               columnspacing=2.0, labelspacing=.5)
    fig.subplots_adjust(left=.075, right=.987, top=.955, bottom=.195,
                        wspace=.15, hspace=.30)
    base.save_figure(fig, stem)


def main():
    obs, predictions, doys = base.load_data()
    table = base.dynamic_table(obs, predictions, doys)
    previous = pd.read_csv(base.OUT / 'dynamic_tracking_rmse.csv')
    keys = ['metric', 'model', 'obs_count']
    actual = table.set_index(keys).sort_index()
    expected = previous.set_index(keys).sort_index()
    assert actual.index.equals(expected.index)
    assert np.allclose(actual.rmse, expected.rmse, atol=1e-12, rtol=0)
    assert table.n.eq(136).all()
    assert table.groupby(['metric', 'model']).size().eq(46).all()
    plot_refined(table, obs)
    table.to_csv(base.OUT / f'{STEM}_rmse.csv', index=False, encoding='utf-8-sig')
    print('Verified: 1,656 RMSE values unchanged, 136 stations, 46 updates per curve, no clipped values.')


if __name__ == '__main__':
    main()
