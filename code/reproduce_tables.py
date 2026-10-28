"""Recalculate the included phenology tables and spatial summary means.

Phenology functions below are copied from the original analysis scripts.
This command does not retrain models or reconstruct complete regional maps.
"""
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / 'data'
BASE = ROOT / '预测结果' / '单点尺度预测'
PHENO = BASE / '物候提取结果'
LEADS = [60, 30, 8]

METRICS = ['SOS_20', 'SOS_50', 'SOS_80', 'EOS_20', 'EOS_50', 'EOS_80']

MODELS = {
    'CNN': 'CNN12_带季节性_滚动预测_phenology.csv',
    'LSTM': 'L12_带季节性_滚动预测_phenology.csv',
    'MLP': 'MLP12_带季节性_滚动预测_phenology.csv',
    'Transformer': 'T12_带季节性_滚动预测_phenology.csv',
    'Dynamic ensemble': 'DynamicWeight_g01_T46_CNN12_滚动预测_phenology.csv',
    'Equal-weight ensemble': 'EqualWeight_T1st_T46_CNN12_滚动预测_phenology.csv',
}

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
base = SimpleNamespace(load_data=load_data, METRICS=METRICS, MODELS=MODELS)

def compare(actual, expected, keys, values):
    a = actual.set_index(keys).sort_index()
    e = expected.set_index(keys).sort_index()
    assert a.index.equals(e.index), "Result keys differ"
    np.testing.assert_allclose(a[values].to_numpy(dtype=float),
                               e[values].to_numpy(dtype=float), rtol=0, atol=1e-10,
                               equal_nan=True)
    return len(a)

def main():
    output = PROJECT / 'outputs'
    output.mkdir(exist_ok=True)
    reference = ROOT / '补充实验/老师批注补充_20260928/tables'
    obs, predictions, doys = load_data()
    dynamic = dynamic_table(obs, predictions, doys)
    detail, events, diagnostics = calculate()
    report = {}
    report['Fig9_rows_verified'] = compare(dynamic,
        pd.read_csv(reference/'Fig9_dynamic_tracking.csv'),
        ['metric','model','obs_count'], ['cutoff_doy','n','reference_mean_doy','rmse'])
    report['Fig10_rows_verified'] = compare(events,
        pd.read_csv(reference/'Fig10_pre_event_MAE.csv'),
        ['metric','model','nominal_lead_days'],
        ['n','mae','bias','actual_lead_min','actual_lead_max'])
    spatial = ROOT/'补充实验/空间格局量化实验'
    maps = pd.read_csv(spatial/'spatial_metrics_per_map.csv')
    expected = pd.read_csv(spatial/'spatial_metrics_summary.csv')
    rows = []
    for _, row in expected.iterrows():
        g = maps.loc[maps.model_id.eq(row.model_id) & maps.step.eq(row.step), row.metric]
        v = g.to_numpy(dtype=float); v = v[np.isfinite(v)]
        mean = np.tanh(np.mean(np.arctanh(np.clip(v,-.999999,.999999)))) if row.metric in ['r_H','r_G'] else np.mean(v)
        rows.append(dict(model_id=row.model_id, step=row.step, metric=row.metric,
                         mean=mean, n_maps=len(v)))
    means=pd.DataFrame(rows)
    report['spatial_means_verified'] = compare(means,expected,
        ['model_id','step','metric'],['mean','n_maps'])
    dynamic.to_csv(output/'Fig9_recalculated.csv',index=False)
    events.to_csv(output/'Fig10_recalculated.csv',index=False)
    means.to_csv(output/'spatial_means_recalculated.csv',index=False)
    report['scope'] = 'Existing phenology estimates and per-map metrics; no model retraining; spatial confidence intervals not recalculated.'
    (output/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report,indent=2))

if __name__ == '__main__':
    main()
