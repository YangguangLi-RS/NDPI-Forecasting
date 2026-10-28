"""Six-threshold pre-event MAE curves using matched stations at every lead."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

import redraw_phenology as base

LEADS = [60, 45, 30, 15, 8]
STEM = 'phenology_pre_event_mae_curves'


def calculate(leads_by_phase=None):
    obs, predictions, doys = base.load_data()
    records, diagnostics = [], []
    for metric in base.METRICS:
        metric_leads = LEADS if leads_by_phase is None else leads_by_phase[metric[:3]]
        retained = []
        for station, reference in obs[metric].items():
            if not np.isfinite(reference):
                continue
            station_records = []
            complete = True
            for lead in metric_leads:
                available = np.flatnonzero((doys <= reference - lead) & (np.arange(len(doys)) > 0))
                if not len(available):
                    complete = False
                    break
                count = int(available[-1])
                for model, prediction in predictions.items():
                    value = prediction.loc[(station, count), metric]
                    if not np.isfinite(value):
                        complete = False
                    station_records.append(dict(
                        metric=metric, Station=int(station), model=model,
                        nominal_lead_days=lead, actual_lead_days=reference-doys[count],
                        obs_count=count, cutoff_doy=int(doys[count]), reference_doy=reference,
                        predicted_doy=value, error_days=value-reference,
                        absolute_error_days=abs(value-reference)))
            if complete:
                retained.append(station)
                records.extend(station_records)
        if not retained:
            raise ValueError(f'No complete stations for {metric}')
        diagnostics.append(dict(metric=metric, initial_stations=len(obs), retained=len(retained),
                                excluded=len(obs)-len(retained),
                                excluded_station_ids=','.join(map(str, sorted(set(obs.index)-set(retained))))))
    detail = pd.DataFrame(records)
    assert (detail.actual_lead_days >= detail.nominal_lead_days).all()
    assert (detail.actual_lead_days < detail.nominal_lead_days + 8).all()
    assert (detail.cutoff_doy < detail.reference_doy).all()
    assert not detail.duplicated(['metric', 'Station', 'model', 'nominal_lead_days']).any()
    for metric, group in detail.groupby('metric'):
        metric_leads = LEADS if leads_by_phase is None else leads_by_phase[metric[:3]]
        sets = [set(g.Station) for _, g in group.groupby(['model', 'nominal_lead_days'])]
        assert len(sets) == len(metric_leads)*len(base.MODELS)
        assert all(s == sets[0] for s in sets)
    summary = detail.groupby(['metric', 'model', 'nominal_lead_days'], sort=False).agg(
        n=('Station', 'size'), mae=('absolute_error_days', 'mean'),
        bias=('error_days', 'mean'), actual_lead_min=('actual_lead_days', 'min'),
        actual_lead_max=('actual_lead_days', 'max')).reset_index()
    expected = sum(len(LEADS if leads_by_phase is None else leads_by_phase[m[:3]])
                   for m in base.METRICS) * len(base.MODELS)
    assert len(summary) == expected
    return detail, summary, pd.DataFrame(diagnostics)


def plot(summary):
    font_manager.findfont('Times New Roman', fallback_to_default=False)
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 18,
                         'axes.titleweight': 'normal', 'axes.labelweight': 'normal',
                         'mathtext.fontset': 'stix', 'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5), sharex=True, sharey=True)
    markers = ['o', 's', '^', 'D', 'P', 'X']
    maximum = summary.mae.max()
    for idx, (ax, metric) in enumerate(zip(axes.flat, base.METRICS)):
        for j, model in enumerate(base.MODELS):
            data = summary.loc[summary.metric.eq(metric) & summary.model.eq(model)]
            data = data.set_index('nominal_lead_days').loc[LEADS]
            ax.plot(LEADS, data.mae, color=base.COLORS[j], ls=base.STYLES[j],
                    marker=markers[j], ms=6.2 if j < 4 else 7,
                    markerfacecolor='white' if j < 4 else base.COLORS[j],
                    markeredgewidth=1.2, lw=1.8 if j < 4 else 2.1, zorder=3+j*.1)
        ax.set_title(f'({chr(97+idx)})  {metric.replace("_", "")}',
                     fontsize=22, loc='left', pad=13)
        ax.set_xlim(63, 5)
        ax.set_xticks(LEADS)
        ax.set_ylim(0, np.ceil(maximum)+.3)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_linewidth(.9)
        ax.tick_params(axis='both', labelsize=18, length=4, pad=6)
        ax.grid(False)
        if idx % 3 == 0:
            ax.set_ylabel('Phenology MAE (days)', fontsize=21, labelpad=8)
        if idx >= 3:
            ax.set_xlabel('Lead time before event (days)', fontsize=19, labelpad=9)
    handles = [Line2D([0], [0], color=c, ls=s, marker=markers[j], ms=6,
                      markerfacecolor='white' if j < 4 else c, lw=2, label=model)
               for j, (model, c, s) in enumerate(zip(base.MODELS, base.COLORS, base.STYLES))]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.52, .012),
               ncol=3, frameon=False, fontsize=18, handlelength=2.7,
               columnspacing=2, labelspacing=.55)
    fig.subplots_adjust(left=.08, right=.99, top=.95, bottom=.20, wspace=.15, hspace=.30)
    base.save_figure(fig, STEM)


def main():
    detail, summary, diagnostics = calculate()
    for table, name in [(detail, 'pre_event_mae_station_errors'),
                        (summary, 'pre_event_mae_summary'),
                        (diagnostics, 'pre_event_mae_sample_diagnostics')]:
        table.to_csv(base.OUT / f'{name}.csv', index=False, encoding='utf-8-sig')
    plot(summary)
    print(diagnostics.to_string(index=False))
    print('Checks passed: paired samples, actual leads, finite results, 180 summary points.')


if __name__ == '__main__':
    main()
