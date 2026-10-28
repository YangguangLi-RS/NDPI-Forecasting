"""Three supplementary configuration figures matching the main Fig. 9 style."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager

import redraw_phenology as base
from plot_main_phenology_refined import plot_refined

CONFIGURATIONS = {
    'L12_no_seasonality': {
        'title': '12-step input without seasonal encoding',
        'models': {
            'CNN': 'CNN12_无季节性_滚动预测_phenology.csv',
            'LSTM': 'L12_无季节性_滚动预测_phenology.csv',
            'MLP': 'MLP12_无季节性_滚动预测_phenology.csv',
            'Transformer': 'T12_不带季节性_滚动预测_phenology.csv',
        },
    },
    'L46_seasonality': {
        'title': '46-step input with seasonal encoding',
        'models': {
            'CNN': 'CNN46_带季节性_滚动预测_phenology.csv',
            'LSTM': 'L46_带季节性_滚动预测_phenology.csv',
            'MLP': 'MLP46_带季节性_滚动预测_phenology.csv',
            'Transformer': 'T46_带季节性_滚动预测_phenology.csv',
        },
    },
    'L46_no_seasonality': {
        'title': '46-step input without seasonal encoding',
        'models': {
            'CNN': 'CNN46_无季节性_滚动预测_phenology.csv',
            'LSTM': 'L46_无季节性_滚动预测_phenology.csv',
            'MLP': 'MLP46_无季节性_滚动预测_phenology.csv',
            'Transformer': 'T46_不带季节性_滚动预测_phenology.csv',
        },
    },
}

# Retain the initialization state (obs_count=0), then show actual observation
# cutoffs at DOY 1, 17, ..., 353. Adjacent displayed observations are 16 days apart.
DISPLAY_COUNTS = [0] + list(range(1, 46, 2))


def main():
    font_manager.findfont('Times New Roman', fallback_to_default=False)
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 18,
                         'axes.titleweight': 'normal', 'axes.labelweight': 'normal',
                         'mathtext.fontset': 'stix', 'pdf.fonttype': 42, 'ps.fonttype': 42})
    obs, main_predictions, doys = base.load_data()
    main_table = base.dynamic_table(obs, main_predictions, doys)
    previous = pd.read_csv(base.OUT / 'dynamic_tracking_rmse.csv')
    keys = ['metric', 'model', 'obs_count']
    assert np.allclose(main_table.set_index(keys).sort_index().rmse,
                       previous.set_index(keys).sort_index().rmse, rtol=0, atol=1e-12)
    expected_index = pd.MultiIndex.from_product([obs.index, range(46)], names=['Station', 'obs_count'])
    tables, diagnostics, manifest, availability = [], [], [], []
    for tag, config in CONFIGURATIONS.items():
        predictions = {}
        for model, filename in config['models'].items():
            path = base.PHENO / filename
            frame = pd.read_csv(path)
            assert not frame.duplicated(['Station', 'obs_count']).any(), filename
            assert set(frame.obs_count) == set(range(46)), filename
            predictions[model] = frame.set_index(['Station', 'obs_count'])
            manifest.append(dict(configuration=tag, model=model, path=str(path)))
        # Identical ensembles are comparison references, not retrained configurations.
        for model in ['Dynamic ensemble', 'Equal-weight ensemble']:
            predictions[model] = main_predictions[model]
            manifest.append(dict(configuration=tag, model=model, path=str(base.PHENO / base.MODELS[model])))
        for model, prediction in predictions.items():
            aligned = prediction.reindex(expected_index)[base.METRICS]
            assert not np.isinf(aligned.to_numpy()).any()
            for metric in base.METRICS:
                for count in range(46):
                    valid = aligned.xs(count, level='obs_count')[metric].notna().sum()
                    availability.append(dict(configuration=tag, model=model, metric=metric,
                                             obs_count=count, valid_stations=int(valid),
                                             missing_stations=len(obs)-int(valid)))
        table = base.dynamic_table(obs, predictions, doys)
        assert len(table) == 6*6*46
        assert table.groupby(['metric', 'obs_count']).n.nunique().eq(1).all()
        assert table.n.between(1, len(obs)).all()
        for model in ['Dynamic ensemble', 'Equal-weight ensemble']:
            actual = table.loc[table.model.eq(model) & table.n.eq(len(obs))].set_index(keys).sort_index().rmse
            reference = main_table.set_index(keys).loc[actual.index, 'rmse']
            assert np.allclose(actual, reference, rtol=0, atol=1e-12)
        stem = f'supp_phenology_dynamic_{tag}'
        plot_refined(table, obs, stem=stem)
        table.to_csv(base.OUT / f'{stem}_rmse.csv', index=False, encoding='utf-8-sig')
        display_table = table.loc[table.obs_count.isin(DISPLAY_COUNTS)].copy()
        assert display_table.obs_count.nunique() == len(DISPLAY_COUNTS)
        assert display_table.groupby(['metric', 'model']).size().eq(len(DISPLAY_COUNTS)).all()
        display_stem = f'{stem}_16day'
        display_table.to_csv(base.OUT / f'{display_stem}_rmse.csv', index=False, encoding='utf-8-sig')
        base.plot_dynamic(display_table, obs, stem=display_stem)
        table['configuration'] = tag
        tables.append(table)
        diagnostics.append(dict(configuration=tag, stations=len(obs),
                                common_n_min=int(table.n.min()), common_n_max=int(table.n.max()),
                                sos_max_rmse=table.loc[table.metric.str.startswith('SOS'), 'rmse'].max(),
                                eos_max_rmse=table.loc[table.metric.str.startswith('EOS'), 'rmse'].max(),
                                calculated_observations_per_curve=46,
                                displayed_points_per_curve=len(DISPLAY_COUNTS), curves=36))
    pd.concat(tables, ignore_index=True).to_csv(
        base.OUT / 'supp_dynamic_all_configurations_rmse.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(diagnostics).to_csv(base.OUT / 'supp_dynamic_diagnostics.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(manifest).to_csv(base.OUT / 'supp_dynamic_input_files.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(availability).to_csv(base.OUT / 'supp_dynamic_availability.csv', index=False, encoding='utf-8-sig')
    print(pd.DataFrame(diagnostics).round(3).to_string(index=False))
    print('Verified: main results unchanged; matched methods per cutoff; full 8-day figures and 16-day previews; missingness recorded.')


if __name__ == '__main__':
    main()
