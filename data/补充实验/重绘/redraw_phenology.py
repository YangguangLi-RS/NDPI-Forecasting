"""Redraw station phenology figures from existing predictions; no model training."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from matplotlib import font_manager

ROOT = Path('data')
BASE = ROOT / '预测结果' / '单点尺度预测'
PHENO = BASE / '物候提取结果'
OUT = Path(__file__).resolve().parent
METRICS = ['SOS_20', 'SOS_50', 'SOS_80', 'EOS_20', 'EOS_50', 'EOS_80']
MODELS = {
    'CNN': 'CNN12_带季节性_滚动预测_phenology.csv',
    'LSTM': 'L12_带季节性_滚动预测_phenology.csv',
    'MLP': 'MLP12_带季节性_滚动预测_phenology.csv',
    'Transformer': 'T12_带季节性_滚动预测_phenology.csv',
    'Dynamic ensemble': 'DynamicWeight_g01_T46_CNN12_滚动预测_phenology.csv',
    'Equal-weight ensemble': 'EqualWeight_T1st_T46_CNN12_滚动预测_phenology.csv',
}
COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#222222', '#7A6C47']
STYLES = ['-', '-', '-', '-', '--', '-.']
LEADS = [60, 30, 15]


def load_data():
    ts = pd.read_csv(ROOT / '站点训练数据_CVmax_2.csv', parse_dates=['Date'])
    ts = ts.loc[ts.Date.dt.year.eq(2018)].sort_values('Date').reset_index(drop=True)
    assert len(ts) == 46 and ts.Date.is_unique
    stations = []
    for station in range(1, 322):
        if 1 <= station <= 41 or 207 <= station <= 258:
            continue
        values = ts[f'NDPI_{station}']
        if values.mean() > .3 and ts.loc[values.idxmax(), 'Date'].month in [5, 6, 7, 8]:
            stations.append(station)
    previous = pd.read_csv(BASE / '物候趋势图_最终版_剔除混交_独立图例' / 'Filtered_Obs_2018_phenology.csv')
    assert set(stations) == set(previous.Station), 'Station filtering differs from original figure'
    obs = pd.read_csv(PHENO / 'Obs_2018_phenology.csv').set_index('Station').loc[stations]
    assert obs.index.is_unique
    predictions = {}
    for model, filename in MODELS.items():
        df = pd.read_csv(PHENO / filename)
        assert not df.duplicated(['Station', 'obs_count']).any()
        assert set(df.obs_count) == set(range(46))
        predictions[model] = df.set_index(['Station', 'obs_count'])
    # With k observations, the last available observation is row k-1.
    # k=0 is an initialization state, displayed at x=0, not a calendar date.
    doys = np.r_[0, ts.Date.dt.dayofyear.to_numpy()[:45]]
    return obs, predictions, doys


def dynamic_table(obs, predictions, doys):
    rows = []
    for metric in METRICS:
        for count, doy in enumerate(doys):
            values = pd.DataFrame({'reference': obs[metric]})
            for model, df in predictions.items():
                values[model] = df.xs(count, level='obs_count')[metric]
            values = values.replace([np.inf, -np.inf], np.nan).dropna()
            assert len(values) > 0
            for model in MODELS:
                err = values[model] - values.reference
                rows.append(dict(metric=metric, model=model, obs_count=count,
                                 cutoff_doy=int(doy), n=len(values),
                                 reference_mean_doy=values.reference.mean(),
                                 rmse=np.sqrt(np.mean(err**2))))
    return pd.DataFrame(rows)


def event_table(obs, predictions, doys):
    rows = []
    diagnostics = []
    for metric in ['SOS_50', 'EOS_50']:
        retained = 0
        for station, reference in obs[metric].items():
            if not np.isfinite(reference):
                continue
            station_rows = []
            valid = True
            for lead in LEADS:
                eligible = np.flatnonzero((doys <= reference - lead) & (np.arange(len(doys)) > 0))
                if not len(eligible):
                    valid = False
                    break
                count = int(eligible[-1])
                for model, df in predictions.items():
                    value = df.loc[(station, count), metric]
                    if not np.isfinite(value):
                        valid = False
                    station_rows.append(dict(metric=metric, Station=int(station), model=model,
                                             nominal_lead_days=lead, obs_count=count,
                                             cutoff_doy=int(doys[count]), reference_doy=reference,
                                             actual_lead_days=reference-doys[count],
                                             predicted_doy=value, error_days=value-reference))
            if valid:
                retained += 1
                rows.extend(station_rows)
        diagnostics.append(dict(metric=metric, filtered_stations=len(obs),
                                complete_case_stations=retained, excluded=len(obs)-retained))
    table = pd.DataFrame(rows)
    assert (table.actual_lead_days >= table.nominal_lead_days).all()
    assert (table.actual_lead_days < table.nominal_lead_days + 8).all()
    for metric, group in table.groupby('metric'):
        sets = [set(g.Station) for _, g in group.groupby(['model', 'nominal_lead_days'])]
        assert all(s == sets[0] for s in sets)
    return table, diagnostics


def style_axes(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_linewidth(.9)
    ax.tick_params(axis='both', labelsize=17, length=4, pad=5)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))
    ax.grid(False)


def save_figure(fig, stem):
    for ext in ['png', 'pdf']:
        fig.savefig(OUT / f'{stem}.{ext}', dpi=400, facecolor='white', bbox_inches='tight')
    fig.savefig(OUT / f'{stem}_preview.png', dpi=110, facecolor='white', bbox_inches='tight')
    plt.close(fig)


def plot_dynamic(table, obs, stem='Fig9_phenology_dynamic_tracking', title=None):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5), sharex=True, sharey='row')
    if title is not None:
        fig.suptitle(title, fontsize=22, fontweight='normal', y=1.015)
    for idx, (ax, metric) in enumerate(zip(axes.flat, METRICS)):
        data = table.loc[table.metric.eq(metric)]
        center = obs[metric].mean()
        ax.axvspan(center-30, center+30, color='#EDEBE6', alpha=.65, lw=0, zorder=0)
        ax.axvline(center, color='#9B968C', ls=':', lw=1.1, zorder=1)
        for j, model in enumerate(MODELS):
            curve = data.loc[data.model.eq(model)].sort_values('obs_count')
            ax.plot(curve.cutoff_doy, curve.rmse, color=COLORS[j], ls=STYLES[j],
                    lw=1.8 if j < 4 else 2.2, zorder=3+j/10)
        ax.set_title(f'({chr(97+idx)})  {metric.replace("_", "")}', fontsize=21, loc='left', pad=12)
        ax.set_xlim(0, 361)
        ax.set_xticks([0, 90, 180, 270, 360])
        if idx % 3 == 0:
            ax.set_ylabel('Phenology RMSE (days)', fontsize=20, labelpad=8)
        if idx >= 3:
            ax.set_xlabel('Last observation date (DOY)', fontsize=19, labelpad=8)
        style_axes(ax)
    for row in range(2):
        maximum = table.loc[table.metric.isin(METRICS[row*3:row*3+3]), 'rmse'].max()
        axes[row, 0].set_ylim(0, maximum*1.10)
    handles = [Line2D([0], [0], color=c, ls=s, lw=2.2, label=m)
               for m, c, s in zip(MODELS, COLORS, STYLES)]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.52, .005),
               ncol=3, frameon=False, fontsize=18, handlelength=2.5,
               columnspacing=2, labelspacing=.55)
    fig.subplots_adjust(left=.08, right=.99, top=.95, bottom=.20, wspace=.15, hspace=.30)
    save_figure(fig, stem)


def plot_events(table):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharey='row')
    labels = ['CNN', 'LSTM', 'MLP', 'Trans.', 'Dynamic', 'Equal']
    for row, metric in enumerate(['SOS_50', 'EOS_50']):
        metric_data = table.loc[table.metric.eq(metric)]
        lo, hi = metric_data.error_days.min(), metric_data.error_days.max()
        pad = max((hi-lo)*.08, 2)
        for col, lead in enumerate(LEADS):
            ax = axes[row, col]
            data = metric_data.loc[metric_data.nominal_lead_days.eq(lead)]
            arrays = [data.loc[data.model.eq(model), 'error_days'].to_numpy() for model in MODELS]
            boxes = ax.boxplot(arrays, positions=np.arange(1, 7), widths=.58,
                               patch_artist=True, showfliers=True,
                               medianprops=dict(color='#222222', lw=1.6),
                               whiskerprops=dict(color='#555555', lw=1.0),
                               capprops=dict(color='#555555', lw=1.0),
                               flierprops=dict(marker='o', markersize=2.8, alpha=.55))
            for j, box in enumerate(boxes['boxes']):
                box.set(facecolor=COLORS[j], edgecolor=COLORS[j], alpha=.55, linewidth=1.2)
                boxes['fliers'][j].set(markeredgecolor=COLORS[j], markerfacecolor='none')
            ax.axhline(0, color='#777777', ls='--', lw=1, zorder=0)
            idx = row*3+col
            ax.set_title(f'({chr(97+idx)})  {metric.replace("_", "")} | ~{lead} days ahead',
                         fontsize=20, loc='left', pad=14)
            ax.set_xticks(np.arange(1, 7), labels, rotation=35, ha='right', rotation_mode='anchor')
            ax.set_ylim(min(lo-pad, -2), max(hi+pad, 2))
            if col == 0:
                ax.set_ylabel('Date error (days)', fontsize=20, labelpad=8)
            style_axes(ax)
            ax.tick_params(axis='x', labelsize=16)
    fig.subplots_adjust(left=.08, right=.99, top=.95, bottom=.13, wspace=.15, hspace=.48)
    save_figure(fig, 'phenology_pre_event_error_boxplots')


def main():
    font_manager.findfont('Times New Roman', fallback_to_default=False)
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 18,
                         'axes.titleweight': 'normal', 'axes.labelweight': 'normal',
                         'mathtext.fontset': 'stix', 'pdf.fonttype': 42, 'ps.fonttype': 42})
    obs, predictions, doys = load_data()
    dynamic = dynamic_table(obs, predictions, doys)
    events, diagnostics = event_table(obs, predictions, doys)
    dynamic.to_csv(OUT / 'dynamic_tracking_rmse.csv', index=False, encoding='utf-8-sig')
    events.to_csv(OUT / 'pre_event_station_errors.csv', index=False, encoding='utf-8-sig')
    stats = []
    for (metric, lead, model), data in events.groupby(['metric', 'nominal_lead_days', 'model'], sort=False):
        error = data.error_days
        stats.append(dict(metric=metric, nominal_lead_days=lead, model=model, n=len(data),
                          actual_lead_min=data.actual_lead_days.min(), actual_lead_max=data.actual_lead_days.max(),
                          bias=error.mean(), median=error.median(), mae=error.abs().mean(),
                          rmse=np.sqrt(np.mean(error**2)), q25=error.quantile(.25), q75=error.quantile(.75)))
    pd.DataFrame(stats).to_csv(OUT / 'pre_event_summary.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(diagnostics).to_csv(OUT / 'sample_diagnostics.csv', index=False, encoding='utf-8-sig')
    plot_dynamic(dynamic, obs)
    plot_events(events)
    print(json.dumps({'output': str(OUT), 'filtered_stations': len(obs),
                      'dynamic_n_range': [int(dynamic.n.min()), int(dynamic.n.max())],
                      'event_samples': diagnostics, 'checks': 'passed'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
