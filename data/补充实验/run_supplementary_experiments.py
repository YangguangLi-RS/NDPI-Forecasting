from __future__ import annotations

import json
import math
import platform
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from pyproj import Transformer
from scipy import ndimage, stats


BASE = Path('data')
OUTPUT_ROOT = BASE / "补充实验"
SPATIAL_OUTPUT = OUTPUT_ROOT / "空间格局量化实验"
TRANSFER_OUTPUT = OUTPUT_ROOT / "误差传递定量实验"

REGIONAL_ROOT = BASE / "区域数据1" / "区域预测csv文件整理"
POINT_ROOT = BASE / "预测结果" / "单点尺度预测"
POINT_PHENO_ROOT = POINT_ROOT / "物候提取结果"
POINT_OBS_PATH = BASE / "站点训练数据_CVmax_2.csv"
POINT_OBS_PHENO_PATH = POINT_PHENO_ROOT / "Obs_2018_phenology.csv"

TARGET_YEAR = 2018
GROWING_MONTHS = set(range(3, 11))
FORECAST_STEPS = (1, 4, 11)
LEAD_DAYS = {1: 8, 4: 32, 11: 88}
PIXEL_SIZE_M = 463.3127165
PHENO_METRICS = ("SOS_20", "SOS_50", "SOS_80", "EOS_80", "EOS_50", "EOS_20")


MODEL_CONFIGS = [
    {
        "model_id": "DynamicWeight",
        "label": "Dynamic ensemble",
        "architecture": "Dynamic ensemble",
        "window": "46",
        "seasonality": "Seasonal",
        "regional_dir": "DynamicWeight",
        "point_dir": "DynamicWeight_g01_T46_CNN12_滚动预测",
    },
    {
        "model_id": "EqualWeight",
        "label": "Equal-weight ensemble",
        "architecture": "Equal-weight ensemble",
        "window": "46",
        "seasonality": "Seasonal",
        "regional_dir": "EqualWeight",
        "point_dir": "EqualWeight_T1st_T46_CNN12_滚动预测",
    },
]

for architecture, point_prefix in [
    ("CNN", "CNN"),
    ("LSTM", "L"),
    ("MLP", "MLP"),
    ("Transformer", "T"),
]:
    for window in (12, 46):
        MODEL_CONFIGS.append(
            {
                "model_id": f"{architecture}_L{window}_Seasonal",
                "label": f"{architecture} L{window} seasonal",
                "architecture": architecture,
                "window": str(window),
                "seasonality": "Seasonal",
                "regional_dir": f"{'Trans' if architecture == 'Transformer' else architecture}_{window}",
                "point_dir": f"{point_prefix}{window}_带季节性_滚动预测",
            }
        )
        MODEL_CONFIGS.append(
            {
                "model_id": f"{architecture}_L{window}_NoSeasonal",
                "label": f"{architecture} L{window} no-season",
                "architecture": architecture,
                "window": str(window),
                "seasonality": "NoSeasonal",
                "regional_dir": f"{'Trans' if architecture == 'Transformer' else architecture}_NoSeasonality_{window}",
                "point_dir": f"{point_prefix}{window}_{'不带' if architecture == 'Transformer' else '无'}季节性_滚动预测",
            }
        )


SITES = {
    "site3": {
        "label": "Deciduous broadleaf forest",
        "obs": BASE / "区域数据1" / "site3" / "NDPI_Site3_SG_filtered.csv",
        "utm_epsg": 32617,
    },
    "site6": {
        "label": "Grassland",
        "obs": BASE / "区域数据1" / "site6" / "NDPI_Site6_SG_filtered.csv",
        "utm_epsg": 32649,
    },
}


def configure_fonts() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    sns.set_theme(style="white", font="Times New Roman")


def ensure_inputs() -> None:
    required = [REGIONAL_ROOT, POINT_ROOT, POINT_PHENO_ROOT, POINT_OBS_PATH, POINT_OBS_PHENO_PATH]
    required.extend(site["obs"] for site in SITES.values())
    missing = [str(path) for path in required if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))


def date_columns(columns, months=None):
    selected = []
    for column in columns:
        parsed = pd.to_datetime(str(column), errors="coerce")
        if pd.isna(parsed) or parsed.year != TARGET_YEAR:
            continue
        if months is None or parsed.month in months:
            selected.append(column)
    return selected


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def r2_score(obs: np.ndarray, pred: np.ndarray) -> float:
    mask = np.isfinite(obs) & np.isfinite(pred)
    obs = obs[mask]
    pred = pred[mask]
    if len(obs) < 2:
        return np.nan
    denominator = np.sum((obs - np.mean(obs)) ** 2)
    if denominator == 0:
        return np.nan
    return float(1.0 - np.sum((obs - pred) ** 2) / denominator)


def resolve_regional_prediction(config, site_name: str, step: int) -> Path:
    model_dir = REGIONAL_ROOT / config["regional_dir"]
    pattern = f"**/Predicted_{site_name}_2018_Step{step}_*.csv"
    candidates = [path for path in model_dir.glob(pattern) if "fitted" not in path.name.lower()]
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one regional prediction for {config['model_id']}, {site_name}, step {step}; "
            f"found {len(candidates)}: {candidates}"
        )
    return candidates[0]


def build_grid_index(obs_df: pd.DataFrame, epsg: int):
    transformer = Transformer.from_crs(4326, epsg, always_xy=True)
    x, y = transformer.transform(
        obs_df["longitude"].to_numpy(dtype=float),
        obs_df["latitude"].to_numpy(dtype=float),
    )
    col = np.rint((x - np.min(x)) / PIXEL_SIZE_M).astype(int)
    row = np.rint((np.max(y) - y) / PIXEL_SIZE_M).astype(int)
    if row.min() != 0 or col.min() != 0 or row.max() != 99 or col.max() != 99:
        raise ValueError(
            f"Unexpected grid extent: rows {row.min()}-{row.max()}, cols {col.min()}-{col.max()}"
        )
    pairs = set(zip(row.tolist(), col.tolist()))
    if len(pairs) != len(obs_df):
        raise ValueError(f"Grid indexing produced {len(obs_df) - len(pairs)} duplicate cells")
    return row, col


def vector_to_grid(values: np.ndarray, row: np.ndarray, col: np.ndarray) -> np.ndarray:
    grid = np.full((100, 100), np.nan, dtype=float)
    grid[row, col] = np.asarray(values, dtype=float)
    return grid


def local_structure_metrics(obs_grid: np.ndarray, pred_grid: np.ndarray):
    common = np.isfinite(obs_grid) & np.isfinite(pred_grid)
    kernel = np.ones((3, 3), dtype=float)
    neighbor_count = ndimage.convolve(common.astype(float), kernel, mode="constant", cval=0.0)
    evaluation_mask = neighbor_count == 9

    obs_fill = np.where(common, obs_grid, 0.0)
    pred_fill = np.where(common, pred_grid, 0.0)

    obs_mean = ndimage.convolve(obs_fill, kernel, mode="constant", cval=0.0) / 9.0
    pred_mean = ndimage.convolve(pred_fill, kernel, mode="constant", cval=0.0) / 9.0
    obs_second = ndimage.convolve(obs_fill**2, kernel, mode="constant", cval=0.0) / 9.0
    pred_second = ndimage.convolve(pred_fill**2, kernel, mode="constant", cval=0.0) / 9.0
    obs_h = np.sqrt(np.maximum(obs_second - obs_mean**2, 0.0))
    pred_h = np.sqrt(np.maximum(pred_second - pred_mean**2, 0.0))

    obs_gx = ndimage.sobel(obs_fill, axis=1, mode="constant", cval=0.0) / 8.0
    obs_gy = ndimage.sobel(obs_fill, axis=0, mode="constant", cval=0.0) / 8.0
    pred_gx = ndimage.sobel(pred_fill, axis=1, mode="constant", cval=0.0) / 8.0
    pred_gy = ndimage.sobel(pred_fill, axis=0, mode="constant", cval=0.0) / 8.0
    obs_g = np.hypot(obs_gx, obs_gy)
    pred_g = np.hypot(pred_gx, pred_gy)

    h_obs = obs_h[evaluation_mask]
    h_pred = pred_h[evaluation_mask]
    g_obs = obs_g[evaluation_mask]
    g_pred = pred_g[evaluation_mask]

    pixel_obs = obs_grid[common]
    pixel_pred = pred_grid[common]
    return {
        "n_common_pixels": int(common.sum()),
        "n_neighborhood_pixels": int(evaluation_mask.sum()),
        "pixel_rmse": float(np.sqrt(np.mean((pixel_pred - pixel_obs) ** 2))),
        "pixel_r2": r2_score(pixel_obs, pixel_pred),
        "pixel_bias": float(np.mean(pixel_pred - pixel_obs)),
        "r_H": pearson_r(h_obs, h_pred),
        "MAE_H": float(np.mean(np.abs(h_pred - h_obs))),
        "r_G": pearson_r(g_obs, g_pred),
        "MAE_G": float(np.mean(np.abs(g_pred - g_obs))),
    }


def summarize_metric(values: pd.Series, correlation: bool):
    arr = values.to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan, 0
    if correlation:
        transformed = np.arctanh(np.clip(arr, -0.999999, 0.999999))
        center_transformed = float(np.mean(transformed))
        center = float(np.tanh(center_transformed))
        sd = float(np.std(arr, ddof=1)) if n > 1 else np.nan
        if n > 1:
            half = stats.t.ppf(0.975, n - 1) * stats.sem(transformed)
            low, high = np.tanh([center_transformed - half, center_transformed + half])
        else:
            low = high = center
    else:
        center = float(np.mean(arr))
        sd = float(np.std(arr, ddof=1)) if n > 1 else np.nan
        if n > 1:
            half = stats.t.ppf(0.975, n - 1) * stats.sem(arr)
            low, high = center - half, center + half
        else:
            low = high = center
    return center, sd, float(low), float(high), n


def grouped_spatial_summary(detail: pd.DataFrame, group_columns):
    rows = []
    metrics = ["pixel_rmse", "pixel_r2", "pixel_bias", "r_H", "MAE_H", "r_G", "MAE_G"]
    for keys, group in detail.groupby(group_columns, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_columns, keys))
        for metric in metrics:
            mean, sd, low, high, n = summarize_metric(group[metric], metric in {"r_H", "r_G"})
            rows.append(
                {
                    **base,
                    "metric": metric,
                    "mean": mean,
                    "sd": sd,
                    "ci95_low": low,
                    "ci95_high": high,
                    "n_maps": n,
                }
            )
    return pd.DataFrame(rows)


def plot_spatial_heatmaps(summary: pd.DataFrame) -> None:
    figure_settings = [
        (
            "spatial_pattern_heterogeneity_heatmap_supplement",
            [
                ("r_H", r"Local heterogeneity correlation ($r_H$)", "viridis", 0.0, 1.0),
                ("MAE_H", r"Local heterogeneity error ($MAE_H$)", "magma_r", 0.0, 0.048),
            ],
        ),
        (
            "spatial_pattern_gradient_heatmap_supplement",
            [
                ("r_G", r"Gradient correlation ($r_G$)", "viridis", 0.0, 1.0),
                ("MAE_G", r"Gradient error ($MAE_G$)", "magma_r", 0.0, 0.048),
            ],
        ),
    ]
    model_order = [config["label"] for config in MODEL_CONFIGS]
    lead_order = [8, 32, 88]
    separators = [2, 6, 10, 14]

    for output_name, metric_settings in figure_settings:
        fig, axes = plt.subplots(1, 2, figsize=(12.8, 10.8))
        for panel_index, (ax, (metric, title, cmap, vmin, vmax)) in enumerate(
            zip(axes, metric_settings)
        ):
            subset = summary[summary["metric"] == metric]
            matrix = subset.pivot(index="model_label", columns="lead_days", values="mean")
            matrix = matrix.reindex(index=model_order, columns=lead_order)
            colorbar_ticks = (
                np.linspace(0.0, 1.0, 6)
                if metric in {"r_H", "r_G"}
                else np.array([0.0, 0.008, 0.016, 0.024, 0.032, 0.040, 0.048])
            )
            sns.heatmap(
                matrix,
                ax=ax,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                annot=True,
                fmt=".3f",
                annot_kws={"fontsize": 10.5, "fontfamily": "Times New Roman"},
                cbar_kws={"shrink": 0.70, "pad": 0.025, "ticks": colorbar_ticks},
                linewidths=0.20,
                linecolor="white",
            )
            normalization = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            color_map = mpl.colormaps[cmap]
            for annotation, value in zip(ax.texts, matrix.to_numpy(dtype=float).ravel()):
                red, green, blue, _ = color_map(normalization(value))
                luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
                annotation.set_color("#202020" if luminance > 0.58 else "white")

            for separator in separators:
                ax.hlines(separator, *ax.get_xlim(), colors="white", linewidth=2.4)
            ax.set_title(title, fontsize=14, pad=13)
            ax.set_xlabel("Forecast lead time (days)", fontsize=12)
            ax.set_ylabel("")
            ax.tick_params(axis="x", labelrotation=0, labelsize=11)
            ax.tick_params(axis="y", labelrotation=0, labelsize=10.5, pad=5)
            if panel_index == 1:
                ax.set_yticklabels([])
                ax.tick_params(axis="y", length=0)
            ax.text(
                -0.13 if panel_index == 0 else -0.08,
                1.025,
                "(a)" if panel_index == 0 else "(b)",
                transform=ax.transAxes,
                fontsize=14,
                fontweight="bold",
                va="bottom",
            )
            colorbar = ax.collections[0].colorbar
            colorbar.ax.tick_params(labelsize=10.5)
            for tick in colorbar.ax.get_yticklabels():
                tick.set_fontname("Times New Roman")

        fig.subplots_adjust(left=0.235, right=0.965, top=0.94, bottom=0.08, wspace=0.22)
        for suffix, dpi in [("png", 600), ("pdf", 300)]:
            fig.savefig(
                SPATIAL_OUTPUT / f"{output_name}.{suffix}",
                dpi=dpi,
                bbox_inches="tight",
            )
        plt.close(fig)


def plot_spatial_factor_summary(summary: pd.DataFrame) -> None:
    """Plot the main-text summary of seasonality, window, and ensemble effects."""
    individual_architectures = {"CNN", "LSTM", "MLP", "Transformer"}
    individual = summary[
        summary["architecture"].isin(individual_architectures)
        & summary["metric"].isin(["r_H", "MAE_H", "r_G", "MAE_G"])
    ].copy()

    factor_rows = []
    for (seasonality, window, lead_days, metric), group in individual.groupby(
        ["seasonality", "window", "lead_days", "metric"], sort=False
    ):
        values = group["mean"].to_numpy(dtype=float)
        if len(values) != 4:
            raise ValueError(
                f"Expected four architectures for {seasonality}, L{window}, "
                f"{lead_days} d, {metric}; found {len(values)}"
            )
        if metric in {"r_H", "r_G"}:
            center = float(
                np.tanh(np.mean(np.arctanh(np.clip(values, -0.999999, 0.999999))))
            )
        else:
            center = float(np.mean(values))
        factor_rows.append(
            {
                "seasonality": seasonality,
                "window": int(window),
                "lead_days": int(lead_days),
                "metric": metric,
                "center": center,
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
            }
        )
    factor_summary = pd.DataFrame(factor_rows)

    line_settings = [
        ("NoSeasonal", 12, "No seasonal encoding, L12", "#D55E00", "--", "o"),
        ("NoSeasonal", 46, "No seasonal encoding, L46", "#D55E00", "-", "s"),
        ("Seasonal", 12, "Seasonal encoding, L12", "#0072B2", "--", "o"),
        ("Seasonal", 46, "Seasonal encoding, L46", "#0072B2", "-", "s"),
    ]
    ensemble_settings = [
        ("DynamicWeight", "Dynamic ensemble", "#111111", "-", "*"),
        ("EqualWeight", "Equal-weight ensemble", "#777777", "-.", "D"),
    ]
    metric_settings = [
        ("r_H", r"Local heterogeneity correlation ($r_H$)", "Correlation", (0.0, 1.0)),
        ("MAE_H", r"Local heterogeneity error ($MAE_H$)", "Mean absolute error", (0.0, 0.048)),
        ("r_G", r"Gradient correlation ($r_G$)", "Correlation", (0.0, 1.0)),
        ("MAE_G", r"Gradient error ($MAE_G$)", "Mean absolute error", (0.0, 0.048)),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 9.3), sharex=True)
    lead_order = np.array([8, 32, 88], dtype=float)
    panel_letters = ["(a)", "(b)", "(c)", "(d)"]
    legend_handles = []
    legend_labels = []

    for panel_index, (ax, (metric, title, ylabel, ylim)) in enumerate(
        zip(axes.flat, metric_settings)
    ):
        for seasonality, window, label, color, linestyle, marker in line_settings:
            data = factor_summary[
                (factor_summary["metric"] == metric)
                & (factor_summary["seasonality"] == seasonality)
                & (factor_summary["window"] == window)
            ].set_index("lead_days").reindex(lead_order.astype(int))
            x = lead_order
            y = data["center"].to_numpy(dtype=float)
            lower = data["minimum"].to_numpy(dtype=float)
            upper = data["maximum"].to_numpy(dtype=float)
            ax.fill_between(x, lower, upper, color=color, alpha=0.075, linewidth=0)
            handle, = ax.plot(
                x,
                y,
                color=color,
                linestyle=linestyle,
                marker=marker,
                markersize=6.0,
                linewidth=1.9,
                markerfacecolor="white" if window == 12 else color,
                markeredgecolor=color,
                markeredgewidth=1.2,
                label=label,
                zorder=3,
            )
            if panel_index == 0:
                legend_handles.append(handle)
                legend_labels.append(label)

        for model_id, label, color, linestyle, marker in ensemble_settings:
            data = summary[
                (summary["metric"] == metric) & (summary["model_id"] == model_id)
            ].set_index("lead_days").reindex(lead_order.astype(int))
            handle, = ax.plot(
                lead_order,
                data["mean"].to_numpy(dtype=float),
                color=color,
                linestyle=linestyle,
                marker=marker,
                markersize=8.0 if marker == "*" else 5.5,
                linewidth=2.15,
                label=label,
                zorder=4,
            )
            if panel_index == 0:
                legend_handles.append(handle)
                legend_labels.append(label)

        ax.set_title(title, fontsize=14, pad=11)
        ax.set_ylabel(ylabel, fontsize=12.5)
        ax.set_ylim(*ylim)
        ax.set_xticks(lead_order)
        if metric in {"MAE_H", "MAE_G"}:
            ax.set_yticks([0.0, 0.008, 0.016, 0.024, 0.032, 0.040, 0.048])
        ax.set_xlabel("Forecast lead time (days)" if panel_index >= 2 else "", fontsize=12.5)
        ax.tick_params(axis="both", labelsize=11.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(
            -0.12,
            1.04,
            panel_letters[panel_index],
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            va="bottom",
        )

    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=3,
        frameon=False,
        fontsize=11.2,
        handlelength=3.0,
        columnspacing=1.9,
    )
    fig.subplots_adjust(left=0.09, right=0.98, top=0.96, bottom=0.15, hspace=0.34, wspace=0.25)
    for suffix, dpi in [("png", 600), ("pdf", 300)]:
        fig.savefig(
            SPATIAL_OUTPUT / f"spatial_pattern_factor_summary_main.{suffix}",
            dpi=dpi,
            bbox_inches="tight",
        )
    plt.close(fig)


def run_spatial_experiment():
    print("[1/2] Spatial-pattern quantification")
    SPATIAL_OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    input_manifest = []
    grid_diagnostics = []

    for site_name, site_info in SITES.items():
        obs_df = pd.read_csv(site_info["obs"])
        row_index, col_index = build_grid_index(obs_df, site_info["utm_epsg"])
        valid_footprint = np.zeros((100, 100), dtype=bool)
        valid_footprint[row_index, col_index] = True
        footprint_neighbors = ndimage.convolve(
            valid_footprint.astype(float), np.ones((3, 3)), mode="constant", cval=0.0
        ) == 9
        grid_diagnostics.append(
            {
                "site": site_name,
                "site_label": site_info["label"],
                "input_pixels": len(obs_df),
                "missing_grid_cells": int(10000 - len(obs_df)),
                "valid_3x3_centers": int(footprint_neighbors.sum()),
                "excluded_for_boundary_or_missing_neighbor": int(len(obs_df) - footprint_neighbors.sum()),
            }
        )
        growing_dates = date_columns(obs_df.columns, GROWING_MONTHS)

        for config in MODEL_CONFIGS:
            for step in FORECAST_STEPS:
                pred_path = resolve_regional_prediction(config, site_name, step)
                pred_df = pd.read_csv(pred_path)
                if len(pred_df) != len(obs_df):
                    raise ValueError(f"Row mismatch: {pred_path} ({len(pred_df)}) vs observation ({len(obs_df)})")
                if not np.allclose(
                    obs_df[["latitude", "longitude"]].to_numpy(dtype=float),
                    pred_df[["latitude", "longitude"]].to_numpy(dtype=float),
                    atol=7e-5,
                    rtol=0,
                ):
                    raise ValueError(f"Coordinate order mismatch: {pred_path}")
                missing_dates = [column for column in growing_dates if column not in pred_df.columns]
                if missing_dates:
                    raise ValueError(f"Missing date columns in {pred_path}: {missing_dates}")
                input_manifest.append(
                    {
                        "experiment": "spatial",
                        "site": site_name,
                        "model_id": config["model_id"],
                        "step": step,
                        "path": str(pred_path),
                    }
                )

                for date_column in growing_dates:
                    obs_values = obs_df[date_column].to_numpy(dtype=float)
                    pred_values = pred_df[date_column].to_numpy(dtype=float)
                    metrics = local_structure_metrics(
                        vector_to_grid(obs_values, row_index, col_index),
                        vector_to_grid(pred_values, row_index, col_index),
                    )
                    date = pd.to_datetime(date_column)
                    rows.append(
                        {
                            "site": site_name,
                            "site_label": site_info["label"],
                            "model_id": config["model_id"],
                            "model_label": config["label"],
                            "architecture": config["architecture"],
                            "window": config["window"],
                            "seasonality": config["seasonality"],
                            "step": step,
                            "lead_days": LEAD_DAYS[step],
                            "date": date.strftime("%Y-%m-%d"),
                            "month": date.month,
                            **metrics,
                        }
                    )

    detail = pd.DataFrame(rows)
    detail.to_csv(SPATIAL_OUTPUT / "spatial_metrics_per_map.csv", index=False, encoding="utf-8-sig")
    diagnostics = pd.DataFrame(grid_diagnostics)
    diagnostics.to_csv(SPATIAL_OUTPUT / "grid_and_mask_diagnostics.csv", index=False, encoding="utf-8-sig")

    group_columns = [
        "model_id",
        "model_label",
        "architecture",
        "window",
        "seasonality",
        "step",
        "lead_days",
    ]
    summary = grouped_spatial_summary(detail, group_columns)
    summary.to_csv(SPATIAL_OUTPUT / "spatial_metrics_summary.csv", index=False, encoding="utf-8-sig")
    site_summary = grouped_spatial_summary(detail, ["site", "site_label"] + group_columns)
    site_summary.to_csv(SPATIAL_OUTPUT / "spatial_metrics_summary_by_site.csv", index=False, encoding="utf-8-sig")
    plot_spatial_heatmaps(summary)
    plot_spatial_factor_summary(summary)

    with (SPATIAL_OUTPUT / "input_files.json").open("w", encoding="utf-8") as handle:
        json.dump(input_manifest, handle, ensure_ascii=False, indent=2)
    print(f"  maps={len(detail)}, output={SPATIAL_OUTPUT}")
    return detail, summary, diagnostics


def valid_phenology_stations(obs_timeseries: pd.DataFrame):
    obs_timeseries = obs_timeseries.copy()
    obs_timeseries["Date"] = pd.to_datetime(obs_timeseries["Date"])
    year_data = obs_timeseries[obs_timeseries["Date"].dt.year == TARGET_YEAR]
    if len(year_data) != 46:
        raise ValueError(f"Expected 46 observations in {TARGET_YEAR}; found {len(year_data)}")
    valid = []
    for station in range(1, 322):
        if 1 <= station <= 41 or 207 <= station <= 258:
            continue
        column = f"NDPI_{station}"
        values = year_data[column]
        peak_date = year_data.loc[values.idxmax(), "Date"]
        if values.mean() > 0.3 and peak_date.month in {5, 6, 7, 8}:
            valid.append(station)
    return year_data, valid


def transfer_stage(obs_count: int) -> str:
    if obs_count <= 15:
        return "Limited (0-15)"
    if obs_count <= 30:
        return "Moderate (16-30)"
    return "Abundant (31-45)"


def safe_correlation(x, y, method="spearman"):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan
    if method == "spearman":
        result = stats.spearmanr(x, y)
    else:
        result = stats.pearsonr(x, y)
    return float(result[0]), float(result[1])


def aggregate_transfer_units(station_detail: pd.DataFrame):
    group_columns = [
        "model_id",
        "model_label",
        "architecture",
        "window",
        "seasonality",
        "obs_count",
        "stage",
        "metric",
    ]
    rows = []
    for keys, group in station_detail.groupby(group_columns, sort=False):
        row = dict(zip(group_columns, keys))
        row.update(
            {
                "n_stations": int(group["station"].nunique()),
                "ndpi_rmse": float(np.sqrt(np.mean(group["ndpi_rmse_station"] ** 2))),
                "ndpi_mae": float(np.mean(group["ndpi_mae_station"])),
                "ndpi_bias": float(np.mean(group["ndpi_bias_station"])),
                "phenology_rmse": float(np.sqrt(np.mean(group["phenology_error"] ** 2))),
                "phenology_mae": float(np.mean(group["phenology_abs_error"])),
                "phenology_bias": float(np.mean(group["phenology_error"])),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def transfer_regression_statistics(unit_summary: pd.DataFrame):
    rows = []
    stage_order = ["All", "Limited (0-15)", "Moderate (16-30)", "Abundant (31-45)"]
    for metric in PHENO_METRICS:
        metric_data = unit_summary[unit_summary["metric"] == metric]
        for stage in stage_order:
            data = metric_data if stage == "All" else metric_data[metric_data["stage"] == stage]
            x = data["ndpi_rmse"].to_numpy(dtype=float)
            y = data["phenology_rmse"].to_numpy(dtype=float)
            finite = np.isfinite(x) & np.isfinite(y)
            x = x[finite]
            y = y[finite]
            rho, rho_p = safe_correlation(x, y, "spearman")
            pearson, pearson_p = safe_correlation(x, y, "pearson")
            if len(x) >= 3 and np.std(x) > 0:
                regression = stats.linregress(x, y)
                slope = float(regression.slope)
                intercept = float(regression.intercept)
                regression_r2 = float(regression.rvalue**2)
                slope_p = float(regression.pvalue)
                slope_se = float(regression.stderr)
            else:
                slope = intercept = regression_r2 = slope_p = slope_se = np.nan
            rows.append(
                {
                    "metric": metric,
                    "stage": stage,
                    "n_model_observation_units": len(x),
                    "spearman_rho": rho,
                    "spearman_p": rho_p,
                    "pearson_r": pearson,
                    "pearson_p": pearson_p,
                    "ols_slope_days_per_ndpi_rmse": slope,
                    "ols_intercept_days": intercept,
                    "ols_r2": regression_r2,
                    "ols_slope_p": slope_p,
                    "ols_slope_se": slope_se,
                }
            )
    return pd.DataFrame(rows)


def station_level_transfer_correlations(station_detail: pd.DataFrame):
    group_columns = ["model_id", "model_label", "obs_count", "stage", "metric"]
    rows = []
    for keys, group in station_detail.groupby(group_columns, sort=False):
        rho, p_value = safe_correlation(
            group["ndpi_rmse_station"], group["phenology_abs_error"], "spearman"
        )
        rows.append(
            {
                **dict(zip(group_columns, keys)),
                "n_stations": int(group["station"].nunique()),
                "spearman_rho_station_level": rho,
                "spearman_p_station_level": p_value,
            }
        )
    return pd.DataFrame(rows)


def within_observation_count_correlations(unit_summary: pd.DataFrame):
    rows = []
    for (metric, obs_count), group in unit_summary.groupby(["metric", "obs_count"], sort=False):
        rho, p_value = safe_correlation(group["ndpi_rmse"], group["phenology_rmse"], "spearman")
        rows.append(
            {
                "metric": metric,
                "obs_count": int(obs_count),
                "stage": transfer_stage(int(obs_count)),
                "n_models": len(group),
                "spearman_rho_across_models": rho,
                "spearman_p_across_models": p_value,
            }
        )
    return pd.DataFrame(rows)


def add_transfer_method_group(unit_summary: pd.DataFrame) -> pd.DataFrame:
    grouped = unit_summary.copy()
    is_ensemble = grouped["architecture"].astype(str).str.contains(
        "ensemble", case=False, na=False
    )
    grouped["method_group"] = np.where(
        is_ensemble,
        "Ensemble",
        np.where(grouped["seasonality"] == "Seasonal", "Seasonal", "No seasonal"),
    )
    return grouped


def grouped_transfer_statistics(unit_summary: pd.DataFrame) -> pd.DataFrame:
    grouped = add_transfer_method_group(unit_summary)
    stage_order = ["All", "Limited (0-15)", "Moderate (16-30)", "Abundant (31-45)"]
    group_order = ["No seasonal", "Seasonal", "Ensemble"]
    rows = []
    for method_group in group_order:
        group_data = grouped[grouped["method_group"] == method_group]
        for metric in PHENO_METRICS:
            metric_data = group_data[group_data["metric"] == metric]
            for stage in stage_order:
                data = metric_data if stage == "All" else metric_data[metric_data["stage"] == stage]
                x = data["ndpi_rmse"].to_numpy(dtype=float)
                y = data["phenology_rmse"].to_numpy(dtype=float)
                finite = np.isfinite(x) & np.isfinite(y)
                x = x[finite]
                y = y[finite]
                rho, rho_p = safe_correlation(x, y, "spearman")
                pearson, pearson_p = safe_correlation(x, y, "pearson")
                if len(x) >= 3 and np.std(x) > 0:
                    regression = stats.linregress(x, y)
                    slope = float(regression.slope)
                    intercept = float(regression.intercept)
                    regression_r2 = float(regression.rvalue**2)
                    slope_p = float(regression.pvalue)
                    slope_se = float(regression.stderr)
                else:
                    slope = intercept = regression_r2 = slope_p = slope_se = np.nan
                rows.append(
                    {
                        "method_group": method_group,
                        "metric": metric,
                        "stage": stage,
                        "n_models": int(data["model_id"].nunique()),
                        "n_model_observation_units": int(len(x)),
                        "spearman_rho": rho,
                        "spearman_p": rho_p,
                        "pearson_r": pearson,
                        "pearson_p": pearson_p,
                        "ols_slope_days_per_ndpi_rmse": slope,
                        "ols_slope_days_per_0.01_ndpi_rmse": slope * 0.01,
                        "ols_intercept_days": intercept,
                        "ols_r2": regression_r2,
                        "ols_slope_p": slope_p,
                        "ols_slope_se": slope_se,
                    }
                )
    return pd.DataFrame(rows)


def grouped_within_observation_correlations(unit_summary: pd.DataFrame) -> pd.DataFrame:
    grouped = add_transfer_method_group(unit_summary)
    grouped = grouped[grouped["method_group"].isin(["No seasonal", "Seasonal"])]
    rows = []
    for (method_group, metric, obs_count), data in grouped.groupby(
        ["method_group", "metric", "obs_count"], sort=False
    ):
        rho, p_value = safe_correlation(data["ndpi_rmse"], data["phenology_rmse"], "spearman")
        rows.append(
            {
                "method_group": method_group,
                "metric": metric,
                "obs_count": int(obs_count),
                "stage": transfer_stage(int(obs_count)),
                "n_models": int(data["model_id"].nunique()),
                "spearman_rho_across_models": rho,
                "spearman_p_across_models": p_value,
            }
        )
    return pd.DataFrame(rows)


def model_level_transfer_slopes(unit_summary: pd.DataFrame) -> pd.DataFrame:
    """Estimate one error-transfer slope for each model and phenological metric."""
    grouped = add_transfer_method_group(unit_summary)
    rows = []
    for (method_group, model_id, metric), data in grouped.groupby(
        ["method_group", "model_id", "metric"], sort=False
    ):
        x = data["ndpi_rmse"].to_numpy(dtype=float)
        y = data["phenology_rmse"].to_numpy(dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        x = x[finite]
        y = y[finite]
        first = data.iloc[0]
        if len(x) >= 3 and np.std(x) > 0:
            regression = stats.linregress(x, y)
            rho, rho_p = safe_correlation(x, y, "spearman")
            slope = float(regression.slope)
            r_squared = float(regression.rvalue**2)
        else:
            slope = r_squared = rho = rho_p = np.nan
        rows.append(
            {
                "method_group": method_group,
                "model_id": model_id,
                "model_label": first["model_label"],
                "architecture": first["architecture"],
                "window": first["window"],
                "seasonality": first["seasonality"],
                "metric": metric,
                "n_observation_counts": int(len(x)),
                "slope_days_per_ndpi_rmse": slope,
                "slope_days_per_0.01_ndpi_rmse": slope * 0.01,
                "spearman_rho": rho,
                "spearman_p": rho_p,
                "ols_r2": r_squared,
            }
        )
    return pd.DataFrame(rows)


def summarize_model_level_slopes(model_slopes: pd.DataFrame) -> pd.DataFrame:
    """Summarize between-model variation using a deterministic model-level bootstrap."""
    rows = []
    group_order = ["No seasonal", "Seasonal", "Ensemble"]
    for group_index, method_group in enumerate(group_order):
        for metric_index, metric in enumerate(PHENO_METRICS):
            values = model_slopes.loc[
                (model_slopes["method_group"] == method_group)
                & (model_slopes["metric"] == metric),
                "slope_days_per_0.01_ndpi_rmse",
            ].dropna().to_numpy(dtype=float)
            ci_low = ci_high = np.nan
            if len(values) >= 3:
                rng = np.random.default_rng(20260921 + group_index * 100 + metric_index)
                bootstrap_means = rng.choice(
                    values, size=(10000, len(values)), replace=True
                ).mean(axis=1)
                ci_low, ci_high = np.quantile(bootstrap_means, [0.025, 0.975])
            rows.append(
                {
                    "method_group": method_group,
                    "metric": metric,
                    "n_models": int(len(values)),
                    "mean_slope_days_per_0.01_ndpi_rmse": float(np.mean(values)),
                    "median_slope_days_per_0.01_ndpi_rmse": float(np.median(values)),
                    "minimum_slope_days_per_0.01_ndpi_rmse": float(np.min(values)),
                    "maximum_slope_days_per_0.01_ndpi_rmse": float(np.max(values)),
                    "bootstrap_95_ci_low": float(ci_low),
                    "bootstrap_95_ci_high": float(ci_high),
                }
            )
    return pd.DataFrame(rows)


def plot_transfer_slope_forest(
    model_slopes: pd.DataFrame,
    slope_summary: pd.DataFrame,
) -> None:
    group_order = ["No seasonal", "Seasonal", "Ensemble"]
    group_colors = {
        "No seasonal": "#D55E00",
        "Seasonal": "#0072B2",
        "Ensemble": "#333333",
    }
    group_titles = {
        "No seasonal": "No seasonal (n = 8)",
        "Seasonal": "Seasonal (n = 8)",
        "Ensemble": "Ensemble (n = 2; no CI)",
    }
    metric_labels = [metric.replace("_", "") for metric in PHENO_METRICS]
    y_positions = np.arange(len(PHENO_METRICS), dtype=float)

    finite_slopes = model_slopes["slope_days_per_0.01_ndpi_rmse"].dropna().to_numpy()
    x_low = math.floor((finite_slopes.min() - 0.6) / 2.0) * 2.0
    x_high = math.ceil((finite_slopes.max() + 0.6) / 2.0) * 2.0
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 6.0), sharex=True, sharey=True)

    for panel_index, (ax, method_group) in enumerate(zip(axes, group_order)):
        color = group_colors[method_group]
        for metric_index, metric in enumerate(PHENO_METRICS):
            data = model_slopes[
                (model_slopes["method_group"] == method_group)
                & (model_slopes["metric"] == metric)
            ].sort_values(["architecture", "window", "model_id"])
            values = data["slope_days_per_0.01_ndpi_rmse"].to_numpy(dtype=float)
            offsets = np.linspace(-0.13, 0.13, len(values)) if len(values) > 1 else np.array([0.0])
            ax.scatter(
                values,
                metric_index + offsets,
                s=30,
                color=color,
                alpha=0.38,
                linewidths=0,
                zorder=2,
            )

            summary = slope_summary[
                (slope_summary["method_group"] == method_group)
                & (slope_summary["metric"] == metric)
            ].iloc[0]
            mean_value = float(summary["mean_slope_days_per_0.01_ndpi_rmse"])
            if int(summary["n_models"]) >= 3:
                ci_low = float(summary["bootstrap_95_ci_low"])
                ci_high = float(summary["bootstrap_95_ci_high"])
                ax.errorbar(
                    mean_value,
                    metric_index,
                    xerr=[[mean_value - ci_low], [ci_high - mean_value]],
                    fmt="D",
                    color=color,
                    markeredgecolor="white",
                    markeredgewidth=0.8,
                    markersize=7.0,
                    capsize=3.5,
                    elinewidth=2.2,
                    capthick=1.6,
                    zorder=4,
                )
            else:
                ax.scatter(
                    mean_value,
                    metric_index,
                    marker="D",
                    s=58,
                    color=color,
                    edgecolor="white",
                    linewidth=0.8,
                    zorder=4,
                )

        ax.axvline(0.0, color="#999999", linewidth=1.0, zorder=0)
        ax.set_xlim(x_low, x_high)
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=7, integer=True))
        ax.set_title(group_titles[method_group], fontsize=13, pad=10)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(metric_labels, fontsize=11)
        ax.tick_params(axis="x", labelsize=10.5)
        ax.tick_params(axis="y", length=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(
            -0.13,
            1.03,
            f"({chr(97 + panel_index)})",
            transform=ax.transAxes,
            fontsize=13.5,
            fontweight="bold",
            va="bottom",
        )

    axes[0].invert_yaxis()
    fig.supxlabel(
        "Model-specific slope (days per 0.01 NDPI RMSE)",
        fontsize=12.5,
        y=0.105,
    )
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#777777",
            markeredgecolor="none",
            alpha=0.5,
            markersize=6,
            label="Individual model",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="#333333",
            markerfacecolor="#333333",
            markersize=6,
            label="Group mean",
        ),
        Line2D([0], [0], color="#333333", linewidth=2.2, label="95% bootstrap CI"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=3,
        frameon=False,
        fontsize=10.5,
    )
    fig.subplots_adjust(left=0.09, right=0.985, top=0.91, bottom=0.22, wspace=0.15)
    for suffix, dpi in [("png", 600), ("pdf", 300)]:
        fig.savefig(
            TRANSFER_OUTPUT / f"error_transfer_model_level_slope_forest.{suffix}",
            dpi=dpi,
            bbox_inches="tight",
        )
    plt.close(fig)


def plot_transfer_correlation_heatmap(
    grouped_statistics: pd.DataFrame,
    grouped_within_obs: pd.DataFrame,
) -> None:
    group_order = ["No seasonal", "Seasonal", "Ensemble"]
    pooled = (
        grouped_statistics[grouped_statistics["stage"] == "All"]
        .pivot(index="method_group", columns="metric", values="spearman_rho")
        .reindex(index=group_order, columns=PHENO_METRICS)
    )
    within = (
        grouped_within_obs.groupby(["method_group", "metric"])[
            "spearman_rho_across_models"
        ]
        .median()
        .unstack("metric")
        .reindex(index=group_order, columns=PHENO_METRICS)
    )
    matrices = [pooled, within]
    titles = [
        "Pooled across observation counts",
        "Median within a fixed observation count",
    ]
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#EEEEEE")
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 3.7))

    for panel_index, (ax, matrix, title) in enumerate(zip(axes, matrices, titles)):
        values = matrix.to_numpy(dtype=float)
        image_artist = ax.imshow(values, cmap=cmap, vmin=-1.0, vmax=1.0, aspect="auto")
        ax.set_xticks(np.arange(len(PHENO_METRICS)))
        ax.set_xticklabels(
            [metric.replace("_", "") for metric in PHENO_METRICS], fontsize=10.5
        )
        ax.set_yticks(np.arange(len(group_order)))
        ax.set_yticklabels(group_order, fontsize=10.5)
        ax.tick_params(axis="both", length=0)
        ax.set_title(title, fontsize=13, pad=10)
        for row_index in range(values.shape[0]):
            for column_index in range(values.shape[1]):
                value = values[row_index, column_index]
                if np.isfinite(value):
                    text_color = "white" if abs(value) >= 0.55 else "#222222"
                    label = f"{value:.2f}"
                else:
                    text_color = "#777777"
                    label = "—"
                ax.text(
                    column_index,
                    row_index,
                    label,
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=10.5,
                )
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.text(
            -0.08,
            1.04,
            f"({chr(97 + panel_index)})",
            transform=ax.transAxes,
            fontsize=13.5,
            fontweight="bold",
            va="bottom",
        )

    fig.subplots_adjust(left=0.105, right=0.90, top=0.87, bottom=0.16, wspace=0.24)
    colorbar_axis = fig.add_axes([0.925, 0.18, 0.014, 0.64])
    colorbar = fig.colorbar(image_artist, cax=colorbar_axis)
    colorbar.set_label("Spearman correlation", fontsize=11)
    colorbar.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    colorbar.ax.tick_params(labelsize=9.5)
    for suffix, dpi in [("png", 600), ("pdf", 300)]:
        fig.savefig(
            TRANSFER_OUTPUT / f"error_transfer_correlation_heatmap_by_seasonality.{suffix}",
            dpi=dpi,
            bbox_inches="tight",
        )
    plt.close(fig)


def plot_transfer_group_trajectories(unit_summary: pd.DataFrame) -> None:
    grouped = add_transfer_method_group(unit_summary)
    titles = {metric: metric.replace("_", "") for metric in PHENO_METRICS}
    group_styles = {
        "No seasonal": {"color": "#D55E00", "linestyle": "--"},
        "Seasonal": {"color": "#0072B2", "linestyle": "-"},
        "Ensemble": {"color": "#333333", "linestyle": "-."},
    }
    selected_counts = [0, 15, 30, 45]
    count_markers = {0: "o", 15: "^", 30: "s", 45: "*"}
    marker_sizes = {0: 42, 15: 46, 30: 46, 45: 82}

    x_max = math.ceil(grouped["ndpi_rmse"].max() * 1.05 / 0.02) * 0.02
    y_max = math.ceil(grouped["phenology_rmse"].max() * 1.05 / 10.0) * 10.0
    fig, axes = plt.subplots(2, 3, figsize=(13.6, 8.7), sharex=True, sharey=True)

    for panel_index, (ax, metric) in enumerate(zip(axes.flat, PHENO_METRICS)):
        metric_data = grouped[grouped["metric"] == metric]
        for method_group, style in group_styles.items():
            data = metric_data[metric_data["method_group"] == method_group]
            trajectory = (
                data.groupby("obs_count")[["ndpi_rmse", "phenology_rmse"]]
                .mean()
                .reindex(range(46))
            )
            ax.plot(
                trajectory["ndpi_rmse"],
                trajectory["phenology_rmse"],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=2.35,
                alpha=0.96,
                zorder=3,
            )

            for obs_count in selected_counts:
                individual = data[data["obs_count"] == obs_count]
                ax.scatter(
                    individual["ndpi_rmse"],
                    individual["phenology_rmse"],
                    color=style["color"],
                    marker=count_markers[obs_count],
                    s=18 if obs_count != 45 else 25,
                    alpha=0.18,
                    linewidths=0,
                    zorder=2,
                )
                center = trajectory.loc[obs_count]
                ax.scatter(
                    center["ndpi_rmse"],
                    center["phenology_rmse"],
                    color=style["color"],
                    edgecolor="white",
                    linewidth=0.8,
                    marker=count_markers[obs_count],
                    s=marker_sizes[obs_count],
                    zorder=5,
                )

            for start, end in [(8, 11), (23, 26), (36, 39)]:
                start_point = trajectory.loc[start]
                end_point = trajectory.loc[end]
                ax.annotate(
                    "",
                    xy=(end_point["ndpi_rmse"], end_point["phenology_rmse"]),
                    xytext=(start_point["ndpi_rmse"], start_point["phenology_rmse"]),
                    arrowprops={
                        "arrowstyle": "-|>",
                        "color": style["color"],
                        "lw": 1.5,
                        "mutation_scale": 10,
                    },
                    zorder=4,
                )

        ax.set_title(titles[metric], fontsize=13.5, pad=8)
        ax.set_xlim(0.0, x_max)
        ax.set_ylim(0.0, y_max)
        ax.tick_params(axis="both", labelsize=10.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(
            -0.13,
            1.03,
            f"({chr(97 + panel_index)})",
            transform=ax.transAxes,
            fontsize=13.5,
            fontweight="bold",
            va="bottom",
        )

    group_handles = [
        Line2D(
            [0],
            [0],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2.5,
            label=method_group,
        )
        for method_group, style in group_styles.items()
    ]
    count_handles = [
        Line2D(
            [0],
            [0],
            color="#444444",
            marker=count_markers[obs_count],
            linestyle="None",
            markersize=7.5 if obs_count != 45 else 10,
            label=str(obs_count),
        )
        for obs_count in selected_counts
    ]
    fig.legend(
        handles=group_handles,
        loc="lower center",
        bbox_to_anchor=(0.27, 0.012),
        ncol=3,
        frameon=False,
        fontsize=10.5,
        title="Method group",
        title_fontsize=10.5,
    )
    fig.legend(
        handles=count_handles,
        loc="lower center",
        bbox_to_anchor=(0.75, 0.012),
        ncol=4,
        frameon=False,
        fontsize=10.5,
        title="Available observations",
        title_fontsize=10.5,
    )
    fig.text(0.5, 0.105, "NDPI forecast RMSE", ha="center", fontsize=12.5)
    fig.text(
        0.012,
        0.54,
        "Phenological-date RMSE (days)",
        va="center",
        rotation="vertical",
        fontsize=12.5,
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.95, bottom=0.19, hspace=0.28, wspace=0.18)
    for suffix, dpi in [("png", 600), ("pdf", 300)]:
        fig.savefig(
            TRANSFER_OUTPUT / f"error_transfer_trajectories_by_seasonality.{suffix}",
            dpi=dpi,
            bbox_inches="tight",
        )
    plt.close(fig)


def plot_transfer_grouped_trajectory_grid(unit_summary: pd.DataFrame) -> None:
    """Plot readable group-mean trajectories with row-specific axis ranges."""
    grouped = add_transfer_method_group(unit_summary)
    display_counts = [8, 19, 31]
    group_order = ["No seasonal", "Seasonal", "Ensemble"]
    group_labels = {
        "No seasonal": "No seasonal",
        "Seasonal": "Seasonal",
        "Ensemble": "Ensemble",
    }
    season_metrics = {
        "Start of season (SOS)": ["SOS_20", "SOS_50", "SOS_80"],
        "End of season (EOS)": ["EOS_20", "EOS_50", "EOS_80"],
    }
    threshold_styles = {
        "20": {"color": "#0072B2", "linestyle": "-"},
        "50": {"color": "#D55E00", "linestyle": "--"},
        "80": {"color": "#009E73", "linestyle": ":"},
    }

    trajectories = {}
    for method_group in group_order:
        for metrics in season_metrics.values():
            for metric in metrics:
                trajectory = (
                    grouped[
                        (grouped["method_group"] == method_group)
                        & (grouped["metric"] == metric)
                    ]
                    .groupby("obs_count")[["ndpi_rmse", "phenology_rmse"]]
                    .mean()
                    .reindex(range(46))
                )
                trajectories[(method_group, metric)] = trajectory

    row_limits = {}
    for method_group in group_order:
        row_data = pd.concat(
            [
                trajectories[(method_group, metric)].loc[display_counts]
                for metric in PHENO_METRICS
            ],
            ignore_index=True,
        )
        x_min_raw = float(row_data["ndpi_rmse"].min())
        x_max_raw = float(row_data["ndpi_rmse"].max())
        y_min_raw = float(row_data["phenology_rmse"].min())
        y_max_raw = float(row_data["phenology_rmse"].max())
        x_padding = max((x_max_raw - x_min_raw) * 0.08, 0.003)
        y_padding = max((y_max_raw - y_min_raw) * 0.08, 0.8)
        x_low = max(0.0, math.floor((x_min_raw - x_padding) / 0.01) * 0.01)
        x_high = math.ceil((x_max_raw + x_padding) / 0.01) * 0.01
        y_span = y_max_raw - y_min_raw
        y_step = 5.0 if y_span > 20 else 2.0 if y_span > 8 else 1.0
        y_low = max(0.0, math.floor((y_min_raw - y_padding) / y_step) * y_step)
        y_high = math.ceil((y_max_raw + y_padding) / y_step) * y_step
        row_limits[method_group] = (x_low, x_high, y_low, y_high)

    fig, axes = plt.subplots(3, 2, figsize=(12.8, 10.0))
    for row_index, method_group in enumerate(group_order):
        x_low, x_high, y_low, y_high = row_limits[method_group]
        for column_index, (season_label, metrics) in enumerate(season_metrics.items()):
            ax = axes[row_index, column_index]
            for metric in metrics:
                threshold = metric.split("_")[1]
                style = threshold_styles[threshold]
                trajectory = trajectories[(method_group, metric)]
                displayed = trajectory.loc[display_counts]
                ax.plot(
                    displayed["ndpi_rmse"],
                    displayed["phenology_rmse"],
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=2.0,
                    alpha=0.92,
                    zorder=2,
                )

                start = trajectory.loc[8]
                middle = trajectory.loc[19]
                endpoint = trajectory.loc[31]
                ax.scatter(
                    start["ndpi_rmse"],
                    start["phenology_rmse"],
                    marker="o",
                    s=48,
                    facecolor="white",
                    edgecolor=style["color"],
                    linewidth=1.5,
                    zorder=4,
                )
                ax.scatter(
                    middle["ndpi_rmse"],
                    middle["phenology_rmse"],
                    marker="s",
                    s=34,
                    color=style["color"],
                    edgecolor="white",
                    linewidth=0.7,
                    zorder=4,
                )
                ax.scatter(
                    endpoint["ndpi_rmse"],
                    endpoint["phenology_rmse"],
                    marker="D",
                    s=43,
                    color=style["color"],
                    edgecolor="white",
                    linewidth=0.7,
                    zorder=4,
                )

                ax.annotate(
                    "",
                    xy=(endpoint["ndpi_rmse"], endpoint["phenology_rmse"]),
                    xytext=(middle["ndpi_rmse"], middle["phenology_rmse"]),
                    arrowprops={
                        "arrowstyle": "-|>",
                        "color": style["color"],
                        "lw": 1.5,
                        "mutation_scale": 10,
                        "shrinkA": 3,
                        "shrinkB": 5,
                    },
                    zorder=3,
                )

            ax.set_xlim(x_low, x_high)
            ax.set_ylim(y_low, y_high)
            ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=6))
            ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5))
            ax.tick_params(axis="both", labelsize=12.5)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if column_index == 1:
                ax.tick_params(axis="y", labelleft=False)
            if row_index == 0:
                ax.set_title(season_label, fontsize=15.5, pad=10)
            panel_index = row_index * 2 + column_index
            ax.text(
                -0.11,
                1.03,
                f"({chr(97 + panel_index)})",
                transform=ax.transAxes,
                fontsize=14.5,
                fontweight="bold",
                va="bottom",
            )

    threshold_handles = [
        Line2D(
            [0],
            [0],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2.2,
            label=f"{threshold}%",
        )
        for threshold, style in threshold_styles.items()
    ]
    cutoff_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="white",
            markeredgecolor="#444444",
            markersize=7,
            label="Feb (8 obs.)",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="None",
            markerfacecolor="#444444",
            markeredgecolor="#444444",
            markersize=6.5,
            label="May (19 obs.)",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="None",
            markerfacecolor="#444444",
            markeredgecolor="#444444",
            markersize=6.5,
            label="Aug (31 obs.)",
        ),
    ]
    fig.subplots_adjust(
        left=0.14,
        right=0.985,
        top=0.94,
        bottom=0.17,
        hspace=0.32,
        wspace=0.15,
    )
    row_centers = [
        (axes[row_index, 0].get_position().y0 + axes[row_index, 0].get_position().y1) / 2
        for row_index in range(len(group_order))
    ]
    for row_center, method_group in zip(row_centers, group_order):
        fig.text(
            0.090,
            row_center,
            group_labels[method_group],
            ha="center",
            va="center",
            rotation="vertical",
            fontsize=14.5,
            fontweight="normal",
        )
    overall_y_center = (
        axes[-1, 0].get_position().y0 + axes[0, 0].get_position().y1
    ) / 2
    fig.text(
        0.035,
        overall_y_center,
        "Phenological-date RMSE (days)",
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=15.0,
    )
    fig.text(0.56, 0.105, "NDPI RMSE over the unobserved period", ha="center", fontsize=15.0)
    fig.legend(
        handles=threshold_handles + cutoff_handles,
        loc="lower center",
        bbox_to_anchor=(0.56, 0.055),
        ncol=6,
        frameon=False,
        fontsize=12.0,
        columnspacing=1.35,
        handletextpad=0.5,
    )
    for suffix, dpi in [("png", 600), ("pdf", 300)]:
        fig.savefig(
            TRANSFER_OUTPUT / f"error_transfer_grouped_trajectory_grid_final.{suffix}",
            dpi=dpi,
            bbox_inches="tight",
        )
    plt.close(fig)


def plot_transfer_grouped_correlation_summary(
    grouped_statistics: pd.DataFrame,
    grouped_within_obs: pd.DataFrame,
) -> None:
    group_styles = {
        "No seasonal": {"color": "#D55E00", "marker": "o"},
        "Seasonal": {"color": "#0072B2", "marker": "s"},
        "Ensemble": {"color": "#333333", "marker": "D"},
    }
    metric_labels = [metric.replace("_", "") for metric in PHENO_METRICS]
    positions = np.arange(len(PHENO_METRICS))
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.9), sharey=True)

    pooled = grouped_statistics[grouped_statistics["stage"] == "All"]
    for method_group, style in group_styles.items():
        data = pooled[pooled["method_group"] == method_group].set_index("metric").reindex(PHENO_METRICS)
        axes[0].plot(
            positions,
            data["spearman_rho"],
            color=style["color"],
            marker=style["marker"],
            linewidth=2.0,
            markersize=6.5,
            label=method_group,
        )

    medians = (
        grouped_within_obs.groupby(["method_group", "metric"])["spearman_rho_across_models"]
        .median()
        .rename("median_rho")
        .reset_index()
    )
    for method_group in ["No seasonal", "Seasonal"]:
        style = group_styles[method_group]
        data = medians[medians["method_group"] == method_group].set_index("metric").reindex(PHENO_METRICS)
        axes[1].plot(
            positions,
            data["median_rho"],
            color=style["color"],
            marker=style["marker"],
            linewidth=2.0,
            markersize=6.5,
            label=method_group,
        )

    titles = [
        "Pooled across observation counts",
        "Within-observation-count median (8 models/group)",
    ]
    for panel_index, (ax, title) in enumerate(zip(axes, titles)):
        ax.axhline(0.0, color="#AAAAAA", linewidth=0.9, zorder=0)
        ax.set_xticks(positions)
        ax.set_xticklabels(metric_labels, fontsize=10.5)
        ax.set_ylim(-0.15, 1.02)
        ax.set_title(title, fontsize=13, pad=9)
        ax.set_xlabel("Phenological metric", fontsize=11.5)
        ax.tick_params(axis="y", labelsize=10.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(
            -0.12,
            1.03,
            "(a)" if panel_index == 0 else "(b)",
            transform=ax.transAxes,
            fontsize=13,
            fontweight="bold",
            va="bottom",
        )
    axes[0].set_ylabel("Spearman correlation", fontsize=11.5)
    for ax, ncol in zip(axes, [3, 2]):
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.19),
            ncol=ncol,
            frameon=False,
            fontsize=10.5,
            handletextpad=0.5,
            columnspacing=1.1,
        )
    fig.subplots_adjust(left=0.08, right=0.985, top=0.90, bottom=0.25, wspace=0.18)
    for suffix, dpi in [("png", 600), ("pdf", 300)]:
        fig.savefig(
            TRANSFER_OUTPUT / f"error_transfer_grouped_correlation_summary.{suffix}",
            dpi=dpi,
            bbox_inches="tight",
        )
    plt.close(fig)


def plot_transfer_scatter(unit_summary: pd.DataFrame, statistics: pd.DataFrame) -> None:
    titles = {
        "SOS_20": "SOS20",
        "SOS_50": "SOS50",
        "SOS_80": "SOS80",
        "EOS_80": "EOS80",
        "EOS_50": "EOS50",
        "EOS_20": "EOS20",
    }
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 8.2), constrained_layout=True)
    scatter_handle = None
    for ax, metric in zip(axes.flat, PHENO_METRICS):
        data = unit_summary[unit_summary["metric"] == metric]
        scatter_handle = ax.scatter(
            data["ndpi_rmse"],
            data["phenology_rmse"],
            c=data["obs_count"],
            cmap="viridis_r",
            vmin=0,
            vmax=45,
            s=16,
            alpha=0.48,
            linewidths=0,
        )
        x = data["ndpi_rmse"].to_numpy(dtype=float)
        y = data["phenology_rmse"].to_numpy(dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        if finite.sum() >= 3 and np.std(x[finite]) > 0:
            regression = stats.linregress(x[finite], y[finite])
            x_line = np.linspace(np.min(x[finite]), np.max(x[finite]), 100)
            ax.plot(x_line, regression.intercept + regression.slope * x_line, color="black", lw=1.3)
        stat_row = statistics[(statistics["metric"] == metric) & (statistics["stage"] == "All")].iloc[0]
        ax.text(
            0.04,
            0.96,
            rf"$\rho$ = {stat_row['spearman_rho']:.3f}" + "\n" + rf"Slope = {stat_row['ols_slope_days_per_ndpi_rmse']:.1f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
        )
        ax.set_title(titles[metric], fontsize=12)
        ax.set_xlabel("NDPI forecast RMSE", fontsize=10)
        ax.set_ylabel("Phenological-date RMSE (days)", fontsize=10)
        ax.tick_params(labelsize=9)
    colorbar = fig.colorbar(scatter_handle, ax=axes, shrink=0.75, pad=0.02)
    colorbar.set_label("Number of available within-year observations", fontsize=10)
    colorbar.ax.tick_params(labelsize=9)
    for suffix, dpi in [("png", 600), ("pdf", 300)]:
        fig.savefig(TRANSFER_OUTPUT / f"error_transfer_by_phenological_metric.{suffix}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_transfer_stage_heatmap(statistics: pd.DataFrame) -> None:
    stage_order = ["Limited (0-15)", "Moderate (16-30)", "Abundant (31-45)", "All"]
    matrix = statistics.pivot(index="metric", columns="stage", values="spearman_rho")
    matrix = matrix.reindex(index=PHENO_METRICS, columns=stage_order)
    matrix.index = [value.replace("_", "") for value in matrix.index]
    fig, ax = plt.subplots(figsize=(8.2, 5.0), constrained_layout=True)
    sns.heatmap(
        matrix,
        ax=ax,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        annot=True,
        fmt=".3f",
        annot_kws={"fontsize": 10, "fontfamily": "Times New Roman"},
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Spearman correlation"},
    )
    ax.set_xlabel("Observation-availability stage", fontsize=11)
    ax.set_ylabel("Phenological metric", fontsize=11)
    ax.tick_params(axis="x", labelrotation=20, labelsize=9)
    ax.tick_params(axis="y", labelrotation=0, labelsize=10)
    for suffix, dpi in [("png", 600), ("pdf", 300)]:
        fig.savefig(TRANSFER_OUTPUT / f"error_transfer_spearman_by_stage.{suffix}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def run_transfer_experiment():
    print("[2/2] Quantitative error-transfer analysis")
    TRANSFER_OUTPUT.mkdir(parents=True, exist_ok=True)
    obs_timeseries = pd.read_csv(POINT_OBS_PATH)
    year_data, valid_stations = valid_phenology_stations(obs_timeseries)
    obs_sequences = {
        station: year_data[f"NDPI_{station}"].to_numpy(dtype=float) for station in valid_stations
    }
    obs_phenology = pd.read_csv(POINT_OBS_PHENO_PATH).set_index("Station")
    rows = []
    diagnostics = []
    input_manifest = []

    for config_index, config in enumerate(MODEL_CONFIGS, start=1):
        prediction_dir = POINT_ROOT / config["point_dir"]
        phenology_path = POINT_PHENO_ROOT / f"{config['point_dir']}_phenology.csv"
        if not prediction_dir.exists() or not phenology_path.exists():
            raise FileNotFoundError(
                f"Missing point-scale input for {config['model_id']}: {prediction_dir}, {phenology_path}"
            )
        pred_phenology = pd.read_csv(phenology_path).set_index(["Station", "obs_count"])
        input_manifest.append(
            {
                "experiment": "error_transfer",
                "model_id": config["model_id"],
                "prediction_dir": str(prediction_dir),
                "phenology_path": str(phenology_path),
            }
        )
        max_observed_prefix_difference = 0.0
        model_row_count = 0

        for station in valid_stations:
            prediction_path = prediction_dir / f"Station_{station}_Predictions.csv"
            if not prediction_path.exists():
                raise FileNotFoundError(prediction_path)
            matrix = np.loadtxt(prediction_path, delimiter=",")
            if matrix.shape != (46, 46):
                raise ValueError(f"Unexpected matrix shape {matrix.shape}: {prediction_path}")
            obs = obs_sequences[station]
            if len(obs) != 46:
                raise ValueError(f"Unexpected observation length for station {station}: {len(obs)}")
            prefix_difference = []
            for obs_count in (1, 12, 30, 45):
                prefix_difference.append(np.max(np.abs(matrix[obs_count, :obs_count] - obs[:obs_count])))
            max_observed_prefix_difference = max(max_observed_prefix_difference, max(prefix_difference))

            for obs_count in range(46):
                residual = matrix[obs_count, obs_count:] - obs[obs_count:]
                ndpi_rmse = float(np.sqrt(np.mean(residual**2)))
                ndpi_mae = float(np.mean(np.abs(residual)))
                ndpi_bias = float(np.mean(residual))
                if (station, obs_count) not in pred_phenology.index:
                    continue
                predicted_pheno = pred_phenology.loc[(station, obs_count)]
                reference_pheno = obs_phenology.loc[station]
                for metric in PHENO_METRICS:
                    predicted_value = float(predicted_pheno[metric])
                    reference_value = float(reference_pheno[metric])
                    if not np.isfinite(predicted_value) or not np.isfinite(reference_value):
                        continue
                    error = predicted_value - reference_value
                    rows.append(
                        {
                            "model_id": config["model_id"],
                            "model_label": config["label"],
                            "architecture": config["architecture"],
                            "window": config["window"],
                            "seasonality": config["seasonality"],
                            "station": station,
                            "obs_count": obs_count,
                            "stage": transfer_stage(obs_count),
                            "metric": metric,
                            "ndpi_rmse_station": ndpi_rmse,
                            "ndpi_mae_station": ndpi_mae,
                            "ndpi_bias_station": ndpi_bias,
                            "phenology_reference_doy": reference_value,
                            "phenology_predicted_doy": predicted_value,
                            "phenology_error": error,
                            "phenology_abs_error": abs(error),
                        }
                    )
                    model_row_count += 1
        diagnostics.append(
            {
                "model_id": config["model_id"],
                "valid_station_count": len(valid_stations),
                "long_table_rows": model_row_count,
                "max_observed_prefix_abs_difference": max_observed_prefix_difference,
            }
        )
        print(f"  {config_index:02d}/{len(MODEL_CONFIGS)} {config['model_id']}: rows={model_row_count}")

    station_detail = pd.DataFrame(rows)
    station_detail.to_csv(
        TRANSFER_OUTPUT / "error_transfer_station_level.csv",
        index=False,
        encoding="utf-8-sig",
    )
    unit_summary = aggregate_transfer_units(station_detail)
    unit_summary.to_csv(
        TRANSFER_OUTPUT / "error_transfer_model_observation_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    regression_stats = transfer_regression_statistics(unit_summary)
    regression_stats.to_csv(
        TRANSFER_OUTPUT / "error_transfer_regression_statistics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    station_correlations = station_level_transfer_correlations(station_detail)
    station_correlations.to_csv(
        TRANSFER_OUTPUT / "error_transfer_station_correlations.csv",
        index=False,
        encoding="utf-8-sig",
    )
    within_obs = within_observation_count_correlations(unit_summary)
    within_obs.to_csv(
        TRANSFER_OUTPUT / "error_transfer_within_observation_count.csv",
        index=False,
        encoding="utf-8-sig",
    )
    grouped_statistics = grouped_transfer_statistics(unit_summary)
    grouped_statistics.to_csv(
        TRANSFER_OUTPUT / "error_transfer_grouped_regression_statistics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    grouped_within_obs = grouped_within_observation_correlations(unit_summary)
    grouped_within_obs.to_csv(
        TRANSFER_OUTPUT / "error_transfer_grouped_within_observation_count.csv",
        index=False,
        encoding="utf-8-sig",
    )
    model_slopes = model_level_transfer_slopes(unit_summary)
    model_slopes.to_csv(
        TRANSFER_OUTPUT / "error_transfer_model_level_slopes.csv",
        index=False,
        encoding="utf-8-sig",
    )
    slope_summary = summarize_model_level_slopes(model_slopes)
    slope_summary.to_csv(
        TRANSFER_OUTPUT / "error_transfer_model_level_slope_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(diagnostics).to_csv(
        TRANSFER_OUTPUT / "input_validation_diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame({"valid_station": valid_stations}).to_csv(
        TRANSFER_OUTPUT / "valid_phenology_stations.csv",
        index=False,
        encoding="utf-8-sig",
    )
    plot_transfer_scatter(unit_summary, regression_stats)
    plot_transfer_stage_heatmap(regression_stats)
    plot_transfer_grouped_trajectory_grid(unit_summary)
    plot_transfer_slope_forest(model_slopes, slope_summary)
    plot_transfer_correlation_heatmap(grouped_statistics, grouped_within_obs)
    with (TRANSFER_OUTPUT / "input_files.json").open("w", encoding="utf-8") as handle:
        json.dump(input_manifest, handle, ensure_ascii=False, indent=2)
    print(f"  station_rows={len(station_detail)}, output={TRANSFER_OUTPUT}")
    return station_detail, unit_summary, regression_stats, station_correlations, within_obs


def best_spatial_results(summary: pd.DataFrame):
    result = []
    for lead_days in (8, 32, 88):
        subset = summary[summary["lead_days"] == lead_days]
        entries = {"lead_days": lead_days}
        for metric, maximize in [("r_H", True), ("MAE_H", False), ("r_G", True), ("MAE_G", False)]:
            metric_subset = subset[subset["metric"] == metric].dropna(subset=["mean"])
            selected_index = metric_subset["mean"].idxmax() if maximize else metric_subset["mean"].idxmin()
            selected = metric_subset.loc[selected_index]
            entries[f"best_{metric}_model"] = selected["model_label"]
            entries[f"best_{metric}_value"] = float(selected["mean"])
        result.append(entries)
    return pd.DataFrame(result)


def dataframe_to_markdown(frame: pd.DataFrame, float_digits: int = 4) -> str:
    """Render a compact Markdown table without pandas' optional tabulate dependency."""
    def render(value):
        if pd.isna(value):
            return "NA"
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.{float_digits}f}"
        return str(value).replace("|", "\\|")

    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(render(value) for value in row) + " |")
    return "\n".join(lines)


def write_report(spatial_detail, spatial_summary, grid_diagnostics, transfer_stats, within_obs):
    best = best_spatial_results(spatial_summary)
    best.to_csv(SPATIAL_OUTPUT / "best_models_by_spatial_metric.csv", index=False, encoding="utf-8-sig")

    overall_transfer = transfer_stats[transfer_stats["stage"] == "All"].copy()
    lead_means = (
        spatial_summary[
            spatial_summary["metric"].isin(["r_H", "MAE_H", "r_G", "MAE_G"])
        ]
        .groupby(["lead_days", "metric"], as_index=False)["mean"]
        .mean()
        .pivot(index="lead_days", columns="metric", values="mean")
    )
    within_obs_summary = (
        within_obs.groupby("metric")["spearman_rho_across_models"]
        .median()
        .reindex(PHENO_METRICS)
    )
    grouped_section = []
    grouped_stats_path = TRANSFER_OUTPUT / "error_transfer_grouped_regression_statistics.csv"
    grouped_within_path = TRANSFER_OUTPUT / "error_transfer_grouped_within_observation_count.csv"
    slope_summary_path = TRANSFER_OUTPUT / "error_transfer_model_level_slope_summary.csv"
    if grouped_stats_path.exists() and grouped_within_path.exists():
        grouped_stats = pd.read_csv(grouped_stats_path)
        grouped_rho = (
            grouped_stats[grouped_stats["stage"] == "All"]
            .pivot(index="metric", columns="method_group", values="spearman_rho")
            .reindex(index=PHENO_METRICS, columns=["No seasonal", "Seasonal", "Ensemble"])
            .reset_index()
        )
        grouped_within = pd.read_csv(grouped_within_path)
        grouped_within_median = (
            grouped_within.groupby(["metric", "method_group"])["spearman_rho_across_models"]
            .median()
            .unstack("method_group")
            .reindex(index=PHENO_METRICS, columns=["No seasonal", "Seasonal"])
            .reset_index()
        )
        slope_lines = []
        if slope_summary_path.exists():
            slope_summary = pd.read_csv(slope_summary_path)
            slope_means = (
                slope_summary.pivot(
                    index="metric",
                    columns="method_group",
                    values="mean_slope_days_per_0.01_ndpi_rmse",
                )
                .reindex(index=PHENO_METRICS, columns=["No seasonal", "Seasonal", "Ensemble"])
                .reset_index()
            )
            slope_lines = [
                "### 模型级误差传播系数",
                "",
                "下表为先在每个模型内部拟合、再跨模型汇总的平均斜率，单位为 NDPI RMSE 每增加 0.01 时物候 RMSE 增加的天数。正文点区间图的区间为模型配置层面的 95% bootstrap 区间，不代表重复训练的不确定性。",
                "",
                dataframe_to_markdown(slope_means),
                "",
            ]
        grouped_section = [
            "### 按季节性配置分组的误差传递",
            "",
            "总体混合结果仅作为基线。下表将单模型分为无季节项和有季节项，并将两种集成方法单独列出。",
            "",
            "正文分组轨迹图按方法组分行、按 SOS/EOS 分列；每条轨迹连接区域实验采用的三个截点（obs_count=8、19、31），数值为同组模型在相同截点下的平均值。单点尺度的完整 obs_count=0–45 结果仍保留在统计表中。各行使用独立坐标范围以展示组内变化，因此跨组绝对误差应依据坐标值而不是轨迹视觉长度比较。",
            "",
            dataframe_to_markdown(grouped_rho),
            "",
            "在固定 obs_count 下分别跨同组模型计算 Spearman rho 后，其中位数如下。集成组仅含两个模型，未计算该项。",
            "",
            dataframe_to_markdown(grouped_within_median),
            "",
            *slope_lines,
        ]

    lines = [
        "# 补充实验结果说明",
        "",
        "## 1. 空间格局量化实验",
        "",
        f"共计算 {len(spatial_detail):,} 个区域-日期-模型-提前期组合。空间指标基于 3 x 3 邻域，边界及邻域内含缺失像元的中心像元不参与计算；未进行插值或补零。",
        "",
        "### 网格与有效像元",
        "",
        dataframe_to_markdown(grid_diagnostics),
        "",
        "### 所有模型的平均变化",
        "",
        dataframe_to_markdown(lead_means.reset_index()),
        "",
        "### 各提前期表现最优的模型",
        "",
        dataframe_to_markdown(best),
        "",
        "注：相关系数采用 Fisher z 变换后求平均；MAE 直接按区域和日期求平均。",
        "",
        "## 2. 误差传递定量实验",
        "",
        "NDPI 误差仅在尚未观测的预测区间内计算，因此不会被已经输入模型的真实观测值稀释。物候误差为预测-提取日期与完整观测序列参考日期之差。",
        "",
        "### 不同物候指标的总体误差传递关系",
        "",
        dataframe_to_markdown(
            overall_transfer[
                [
                    "metric",
                    "n_model_observation_units",
                    "spearman_rho",
                    "ols_slope_days_per_ndpi_rmse",
                    "ols_r2",
                ]
            ]
        ),
        "",
        *grouped_section,
        "### 总体混合的固定观测数敏感性分析（补充）",
        "",
        "下表在固定 obs_count 下跨全部 18 个模型计算 Spearman rho，仅作为总体基线；因其仍混合了季节性配置差异，组内结论以上述分组结果为主。",
        "",
        dataframe_to_markdown(within_obs_summary.rename("median_within_obs_count_rho").reset_index()),
        "",
        "## 3. 文件说明",
        "",
        "- 空间实验明细：`空间格局量化实验/spatial_metrics_per_map.csv`",
        "- 空间实验汇总：`空间格局量化实验/spatial_metrics_summary.csv`",
        "- 空间正文主图：`空间格局量化实验/spatial_pattern_factor_summary_main.png` 和 `.pdf`",
        "- 空间附录图（局部异质性）：`空间格局量化实验/spatial_pattern_heterogeneity_heatmap_supplement.png` 和 `.pdf`",
        "- 空间附录图（梯度）：`空间格局量化实验/spatial_pattern_gradient_heatmap_supplement.png` 和 `.pdf`",
        "- 误差传递站点明细：`误差传递定量实验/error_transfer_station_level.csv`",
        "- 误差传递汇总：`误差传递定量实验/error_transfer_model_observation_summary.csv`",
        "- 回归与相关统计：`误差传递定量实验/error_transfer_regression_statistics.csv`",
        "- 分组误差传递统计：`误差传递定量实验/error_transfer_grouped_regression_statistics.csv`",
        "- 模型级传播系数明细：`误差传递定量实验/error_transfer_model_level_slopes.csv`",
        "- 模型级传播系数汇总：`误差传递定量实验/error_transfer_model_level_slope_summary.csv`",
        "- 分组误差轨迹正文图：`误差传递定量实验/error_transfer_grouped_trajectory_grid_final.png` 和 `.pdf`",
        "- 模型级传播系数分析图：`误差传递定量实验/error_transfer_model_level_slope_forest.png` 和 `.pdf`",
        "- 分组相关性热图（可选附录）：`误差传递定量实验/error_transfer_correlation_heatmap_by_seasonality.png` 和 `.pdf`",
        "- 旧版二维轨迹诊断图：`误差传递定量实验/error_transfer_trajectories_by_seasonality.png` 和 `.pdf`",
        "- 旧版分组相关性折线图：`误差传递定量实验/error_transfer_grouped_correlation_summary.png` 和 `.pdf`",
        "- 总体误差传递附录图：`误差传递定量实验/error_transfer_by_phenological_metric.png` 和 `.pdf`",
        "",
        "所有绘图文字均设置为 Times New Roman。",
    ]
    (OUTPUT_ROOT / "补充实验结果说明.md").write_text("\n".join(lines), encoding="utf-8")


def write_manifest():
    manifest = {
        "base": str(BASE),
        "output_root": str(OUTPUT_ROOT),
        "target_year": TARGET_YEAR,
        "forecast_steps": list(FORECAST_STEPS),
        "lead_days": LEAD_DAYS,
        "growing_months": sorted(GROWING_MONTHS),
        "local_window": "3x3",
        "missing_data_rule": "Exclude borders and centers whose common observed-predicted 3x3 neighborhood is incomplete",
        "font": "Times New Roman",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": __import__("scipy").__version__,
        "matplotlib": mpl.__version__,
        "seaborn": sns.__version__,
    }
    with (OUTPUT_ROOT / "experiment_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    configure_fonts()
    ensure_inputs()
    if "--spatial-plot-only" in sys.argv:
        spatial_summary = pd.read_csv(SPATIAL_OUTPUT / "spatial_metrics_summary.csv")
        plot_spatial_factor_summary(spatial_summary)
        plot_spatial_heatmaps(spatial_summary)
        print(f"Spatial main figure regenerated. Results: {SPATIAL_OUTPUT}")
        return
    if "--trajectory-grid-only" in sys.argv:
        unit_summary = pd.read_csv(
            TRANSFER_OUTPUT / "error_transfer_model_observation_summary.csv"
        )
        plot_transfer_grouped_trajectory_grid(unit_summary)
        print(f"Grouped trajectory figure regenerated. Results: {TRANSFER_OUTPUT}")
        return
    if "--transfer-plot-only" in sys.argv:
        unit_summary = pd.read_csv(
            TRANSFER_OUTPUT / "error_transfer_model_observation_summary.csv"
        )
        grouped_statistics = grouped_transfer_statistics(unit_summary)
        grouped_statistics.to_csv(
            TRANSFER_OUTPUT / "error_transfer_grouped_regression_statistics.csv",
            index=False,
            encoding="utf-8-sig",
        )
        grouped_within_obs = grouped_within_observation_correlations(unit_summary)
        grouped_within_obs.to_csv(
            TRANSFER_OUTPUT / "error_transfer_grouped_within_observation_count.csv",
            index=False,
            encoding="utf-8-sig",
        )
        model_slopes = model_level_transfer_slopes(unit_summary)
        model_slopes.to_csv(
            TRANSFER_OUTPUT / "error_transfer_model_level_slopes.csv",
            index=False,
            encoding="utf-8-sig",
        )
        slope_summary = summarize_model_level_slopes(model_slopes)
        slope_summary.to_csv(
            TRANSFER_OUTPUT / "error_transfer_model_level_slope_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
        plot_transfer_grouped_trajectory_grid(unit_summary)
        plot_transfer_slope_forest(model_slopes, slope_summary)
        plot_transfer_correlation_heatmap(grouped_statistics, grouped_within_obs)
        print(f"Alternative transfer figures regenerated. Results: {TRANSFER_OUTPUT}")
        return
    if "--report-only" in sys.argv:
        spatial_detail = pd.read_csv(SPATIAL_OUTPUT / "spatial_metrics_per_map.csv")
        spatial_summary = pd.read_csv(SPATIAL_OUTPUT / "spatial_metrics_summary.csv")
        grid_diagnostics = pd.read_csv(SPATIAL_OUTPUT / "grid_and_mask_diagnostics.csv")
        transfer_stats = pd.read_csv(TRANSFER_OUTPUT / "error_transfer_regression_statistics.csv")
        within_obs = pd.read_csv(TRANSFER_OUTPUT / "error_transfer_within_observation_count.csv")
        write_report(spatial_detail, spatial_summary, grid_diagnostics, transfer_stats, within_obs)
        write_manifest()
        print(f"Report regenerated. Results: {OUTPUT_ROOT}")
        return
    spatial_detail, spatial_summary, grid_diagnostics = run_spatial_experiment()
    _, _, transfer_stats, _, within_obs = run_transfer_experiment()
    write_report(spatial_detail, spatial_summary, grid_diagnostics, transfer_stats, within_obs)
    write_manifest()
    print(f"Completed. Results: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
