"""Replot verified Fig. 10 values; rings denote within-lead minima, not significance."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from PIL import Image


BASE = Path(__file__).resolve().parents[1] / 'data' / '补充实验'
SOURCE = BASE / '重绘/pre_event_dotplot_summary.csv'
CHECK = BASE / '老师批注补充_20260928/tables/Fig10_pre_event_MAE.csv'
ERRORS = BASE / '老师批注补充_20260928/tables/Fig10_station_errors.csv'
OUT = Path(__file__).resolve().parents[1] / 'outputs'
OUT.mkdir(exist_ok=True)
STEM = 'Fig_10_minima'
METRICS = ['SOS_20', 'SOS_50', 'SOS_80', 'EOS_20', 'EOS_50', 'EOS_80']
LABELS = ['S20', 'S50', 'S80', 'F20', 'F50', 'F80']
MODELS = ['CNN', 'LSTM', 'MLP', 'Transformer', 'Dynamic ensemble', 'Equal-weight ensemble']
LEADS = [60, 30, 8]
COLORS = ['#0072B2', '#D55E00', '#009E73']
MARKERS = ['o', 's', '^']
OFFSETS = [-0.28, 0, 0.28]
KEYS = ['metric', 'model', 'nominal_lead_days']


def verify_data():
    source = pd.read_csv(SOURCE)
    prior = pd.read_csv(CHECK)
    errors = pd.read_csv(ERRORS)
    original = source.set_index(KEYS).sort_index()
    checked = prior.set_index(KEYS).sort_index()
    assert len(original) == 108 and original.index.equals(checked.index)
    assert not source.duplicated(KEYS).any()
    for col in ['mae', 'n', 'bias', 'actual_lead_min', 'actual_lead_max']:
        np.testing.assert_allclose(original[col], checked[col], rtol=0, atol=1e-12)
    recalculated = errors.groupby(KEYS).absolute_error_days.agg(['mean', 'count']).sort_index()
    assert original.index.equals(recalculated.index)
    np.testing.assert_allclose(original.mae, recalculated['mean'], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(original.n, recalculated['count'])
    for _, block in errors.groupby('metric'):
        station_sets = [set(g.Station) for _, g in block.groupby(['model', 'nominal_lead_days'])]
        assert all(s == station_sets[0] for s in station_sets)
    minima = source.groupby(['metric', 'nominal_lead_days']).mae.transform('min')
    source['is_minimum'] = np.isclose(source.mae, minima, rtol=0, atol=1e-12)
    source['display_metric'] = source.metric.map(dict(zip(METRICS, LABELS)))
    return source


def main():
    table = verify_data()
    font = font_manager.findfont('Times New Roman', fallback_to_default=False)
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 8.5,
                         'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'savefig.facecolor': 'white'})
    fig, axes = plt.subplots(2, 3, figsize=(180 / 25.4, 122 / 25.4), sharex=True, sharey=True)
    positions = np.arange(6)
    for idx, (ax, metric, label) in enumerate(zip(axes.flat, METRICS, LABELS)):
        for lead, color, marker, offset in zip(LEADS, COLORS, MARKERS, OFFSETS):
            block = table.loc[table.metric.eq(metric) & table.nominal_lead_days.eq(lead)]
            block = block.set_index('model').loc[MODELS]
            x, y = block.mae.to_numpy(), positions + offset
            ax.scatter(x, y, s=20, marker=marker, facecolors=color,
                       edgecolors='white', linewidths=0.22, zorder=3)
            best = block.is_minimum.to_numpy()
            ax.scatter(x[best], y[best], s=49, marker='o', facecolors='none',
                       edgecolors='#444444', linewidths=0.65, zorder=4)
        ax.set_title(f'({chr(97 + idx)})  {label}', loc='left', fontsize=10.5, pad=6)
        ax.set_yticks(positions, MODELS)
        ax.set_ylim(5.6, -0.6)
        ax.set_xlim(0, 10)
        ax.set_xticks(np.arange(0, 11, 2))
        ax.tick_params(axis='both', labelsize=8.5, length=2.2, width=0.45, pad=3)
        ax.tick_params(axis='y', length=0, labelleft=idx % 3 == 0)
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_linewidth(0.45)
        if idx >= 3:
            ax.set_xlabel('Phenology MAE (days)', fontsize=9.2, labelpad=4)
    handles = [Line2D([], [], marker=m, color=c, linestyle='None', markersize=5,
                      label=f'{lead} days before')
               for lead, c, m in zip(LEADS, COLORS, MARKERS)]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.59, 0.035),
               ncol=3, frameon=False, fontsize=10.2, handletextpad=0.45, columnspacing=1.8)
    fig.subplots_adjust(left=0.205, right=0.985, top=0.94, bottom=0.175,
                        wspace=0.16, hspace=0.29)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for txt in fig.findobj(matplotlib.text.Text):
        if txt.get_visible() and txt.get_text():
            bounds = txt.get_window_extent(renderer)
            assert bounds.x0 >= -1 and bounds.y0 >= -1
            assert bounds.x1 <= fig.bbox.width + 1 and bounds.y1 <= fig.bbox.height + 1
    table.to_csv(OUT / f'{STEM}_values.csv', index=False, encoding='utf-8-sig')
    winners = table.loc[table.is_minimum, ['display_metric', 'nominal_lead_days', 'model', 'mae', 'n']]
    winners.to_csv(OUT / f'{STEM}_marked_minima.csv', index=False, encoding='utf-8-sig')
    provenance = {'sources': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in [SOURCE, CHECK, ERRORS]},
                  'font': font, 'groups_verified': 108,
                  'minima_marked': len(winners), 'ties': 'unrounded MAE; absolute tolerance 1e-12, relative tolerance 0',
                  'scope': 'minimum among six methods within each event and lead time',
                  'uncertainty': 'not plotted; rings do not indicate significance',
                  'palette': COLORS, 'markers': MARKERS, 'vertical_offsets': OFFSETS,
                  'journal_profile': 'none; preserve existing figure, no journal compliance claim',
                  'tiff_dpi': 1000, 'png_dpi': 425, 'tiff_compression': 'tiff_lzw'}
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'{STEM}.{ext}', dpi=425, facecolor='white')
    (OUT / f'{STEM}_provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf8')
    fig.savefig(OUT / f'{STEM}.tiff', dpi=1000, facecolor='white',
                pil_kwargs={'compression': 'tiff_lzw'})
    with Image.open(OUT / f'{STEM}.tiff') as raster:
        rgb = raster.convert('RGB')
    rgb.save(OUT / f'{STEM}.tiff', dpi=(1000, 1000), compression='tiff_lzw')
    rgb.close()
    fig.savefig(OUT / f'{STEM}_preview.png', dpi=180, facecolor='white')
    plt.close(fig)
    report = {}
    for ext in ['png', 'tiff']:
        path = OUT / f'{STEM}.{ext}'
        with Image.open(path) as im:
            im.load()
            report[ext] = {'pixels': im.size, 'dpi': tuple(float(v) for v in im.info['dpi']),
                           'mode': im.mode, 'compression': im.info.get('compression'),
                           'bytes': path.stat().st_size}
            assert abs(im.width / im.info['dpi'][0] * 25.4 - 180) < 0.1
    (OUT / f'{STEM}_verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    caption = ('Open circles indicate the lowest MAE across the six methods for each phenological '
               'event and nominal lead time. Tied minima are all marked; these markers do not '
               'imply statistically significant differences.\n')
    (OUT / f'{STEM}_caption.txt').write_text(caption, encoding='utf-8')
    print(winners.to_string(index=False))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
