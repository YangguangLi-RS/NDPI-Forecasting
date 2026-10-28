# NDPI-Forecasting

Code and selected data accompanying **A multidimensional comparison for long-horizon recursive vegetation index forecasting: error accumulation, phenological retention, and spatial-pattern preservation**.

## Contents

- `code/reproduce_tables.py`: recompute the included Figure 9/10 tables and spatial-summary means.
- `code/plot_fig10_minima.py`: updated Figure 10 plot with within-lead minimum markers (requires plotting dependencies).
- `code/legacy/`: original regional model-training, point/regional forecasting, ensemble and phenology scripts, with local absolute paths replaced by project-relative paths. Original filenames are retained.
- `data/`: processed point series, point phenology estimates, validation/ensemble summaries, and the later phenology and spatial-pattern analyses.
- `data/examples/`: 100-pixel examples from each of the two regional cases. These are demonstration subsets, not the full regions used to calculate published spatial statistics.
- `MANIFEST.csv`: included files, sizes, and SHA-256 checksums.

## Quick verification

Run from the repository root with Python 3.12 (the release verification environment), NumPy and pandas:

```bash
python -m pip install -r requirements.txt
python code/reproduce_tables.py
```

The command writes results to `outputs/` and checks them against the included reference tables. It recalculates point phenology statistics from the saved phenological estimates and spatial-summary means from saved per-map metrics. It does not rerun neural models or regenerate the complete spatial maps.

## Data notes

- `data/站点训练数据_CVmax_2.csv`: Date plus NDPI_1–NDPI_321; use the 2008–2018 study period. The archived source table also contains 2019 observations, which are not used in the included 2018 verification.
- `data/预测结果/单点尺度预测/物候提取结果/`: full-year reference and prediction-derived phenological dates for the point experiments.
- `data/补充实验/老师批注补充_20260928/tables/`: Figure 9/10 values and station-level errors, including the revised S/F display labels.
- `data/补充实验/空间格局量化实验/`: per-map metrics and summaries for the complete original regional evaluation; mean correlations use the Fisher-z transformation.
- `data/补充实验/重绘/`: additional point phenology results and their original analysis/plot scripts.
- `data/精度/最新/` and `data/模型多步训练结果/`: validation and ensemble comparison tables.

S20/S50/S80 denote spring thresholds; F20/F50/F80 denote fall thresholds. Original CSV/code labels SOS_20/SOS_50/SOS_80 and EOS_20/EOS_50/EOS_80 map to these respective display names. Phenology values and errors are in days. `obs_count` is the number of available observations, whereas `step` is a recursive forecast step. Point ensembles combine the 46-step Transformer and 12-step CNN; regional ensembles use the 46-step components. See each script for its configuration; archived alternative strategies are not all the final manuscript configuration.

## Running the archived scripts

Run scripts from the repository root so their `data/...` paths resolve. The original scripts retain their algorithms, filenames and experiment-specific settings. Additional packages include TensorFlow/Keras for training/inference, SciPy for fitting, and Matplotlib/Seaborn/pyproj for spatial analysis and plotting. Historical supplementary-analysis versions are recorded in `data/补充实验/experiment_manifest.json`; this does not establish the original neural-training environment.

This is a **compact code and selected-data release**, not the complete original experiment archive. Model checkpoints, full regional prediction grids, daily fitted intermediate curves and some original training inputs are omitted. Legacy end-to-end scripts require those additional inputs at their documented relative paths. Their training/inference has not been rerun for this release. The 100-pixel examples cannot reproduce full-region spatial statistics. No complete-reproduction claim is made.

## Original satellite products

- MOD09GA V6.1: https://doi.org/10.5067/MODIS/MOD09GA.061
- MYD09GA V6.1: https://doi.org/10.5067/MODIS/MYD09GA.061
- MCD12Q1 V6.1: https://doi.org/10.5067/MODIS/MCD12Q1.061

## License and archive

Project code is provided under the MIT License. Project-authored data and documentation are provided under CC BY 4.0 to the extent the contributors hold the necessary rights. Third-party data/software retain their original terms; see `DATA_LICENSE.md`.

A Zenodo DOI will be added after this version is actually archived. This repository must not be described as DOI-archived until that step is complete.
