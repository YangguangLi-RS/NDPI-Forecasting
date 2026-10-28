"""Grouped MAE dot plots at the same pre-event times for SOS and EOS."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

import redraw_phenology as base
from plot_pre_event_accuracy import calculate

LEADS = [60, 30, 8]
STEM = 'phenology_pre_event_mae_dotplot'
COLORS = ['#0072B2', '#D55E00', '#009E73']
MARKERS = ['o', 's', '^']


def plot(summary):
    font_manager.findfont('Times New Roman', fallback_to_default=False)
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 19,
                         'axes.titleweight': 'normal', 'axes.labelweight': 'normal',
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharex=True, sharey=True)
    positions = np.arange(len(base.MODELS))
    offsets = [-.20, 0, .20]
    upper = max(10, 2*np.ceil(summary.mae.max()/2))
    for idx, (ax, metric) in enumerate(zip(axes.flat, base.METRICS)):
        for lead, color, marker, offset in zip(LEADS, COLORS, MARKERS, offsets):
            data = summary.loc[summary.metric.eq(metric) & summary.nominal_lead_days.eq(lead)]
            values = data.set_index('model').loc[list(base.MODELS), 'mae'].to_numpy()
            ax.scatter(values, positions+offset, s=75, marker=marker,
                       facecolors=color, edgecolors='white', linewidths=.5, zorder=3)
        ax.set_title(f'({chr(97+idx)})  {metric.replace("_", "")}',
                     fontsize=23, loc='left', pad=14)
        ax.set_yticks(positions, list(base.MODELS))
        ax.set_ylim(len(base.MODELS)-.5, -.5)
        ax.set_xlim(0, upper)
        ax.set_xticks(np.arange(0, upper+.1, 2))
        ax.tick_params(axis='both', labelsize=19, length=4, pad=7)
        ax.tick_params(axis='y', length=0, labelleft=idx % 3 == 0)
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_linewidth(.8)
        ax.grid(False)
        if idx >= 3:
            ax.set_xlabel('Phenology MAE (days)', fontsize=21, labelpad=9)
    handles = [Line2D([0], [0], marker=m, color=c, linestyle='None', markersize=9,
                      label=f'{lead} days before')
               for lead, c, m in zip(LEADS, COLORS, MARKERS)]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.59, .015),
               ncol=3, frameon=False, fontsize=20, handletextpad=.45, columnspacing=2.5)
    fig.subplots_adjust(left=.205, right=.985, top=.94, bottom=.15,
                        wspace=.16, hspace=.28)
    base.save_figure(fig, STEM)


def main():
    detail, summary, diagnostics = calculate({'SOS': LEADS, 'EOS': LEADS})
    # Shared evaluation times must reproduce the earlier five-time calculation.
    previous = pd.read_csv(base.OUT / 'pre_event_mae_summary.csv')
    keys = ['metric', 'model', 'nominal_lead_days']
    reference = previous.loc[previous.nominal_lead_days.isin(LEADS)].set_index(keys).sort_index()
    current = summary.set_index(keys).sort_index()
    assert reference.index.equals(current.index)
    assert np.allclose(reference.mae, current.mae, rtol=0, atol=1e-12)
    assert np.array_equal(reference.n, current.n)
    for table, name in [(detail, 'pre_event_dotplot_station_errors'),
                        (summary, 'pre_event_dotplot_summary'),
                        (diagnostics, 'pre_event_dotplot_sample_diagnostics')]:
        table.to_csv(base.OUT / f'{name}.csv', index=False, encoding='utf-8-sig')
    plot(summary)
    print(diagnostics.to_string(index=False))
    print('Validated: 108 MAE values match prior results; fixed samples and pre-event cutoffs.')


if __name__ == '__main__':
    main()
