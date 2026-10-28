import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

# ==================== 1. 全局绘图风格 (Nature Style) ====================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

FONT_SIZE_LABEL = 26
FONT_SIZE_TICK = 25
FONT_SIZE_LEGEND = 18
MARKER_SIZE = 9

# ==================== 2. 实验配置 ====================
STEPS_TO_TEST = [1, 2, 3, 4, 5, 6, 12, 23, 34, 46]
X_TICKS_DISPLAY = [1, 3, 6, 12, 23, 34, 46]
TARGET_YEAR = 2018
SITES = list(range(1, 322))

KC = 3.0
DELTA = 1.0

# ==================== 3. 文件路径 ====================
DATA_FILE = 'data/站点训练数据_CVmax_2.csv'
DIR_TRANS = 'data/模型多步训练结果/Transformer_specific_step_evaluation_46步长'
DIR_LT = 'data/模型多步训练结果/CNN_model_specific_step_evaluation_12步_新'
OUTPUT_DIR = 'data/模型多步训练结果/加权策略对比实验_T-CNN'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== 4. 策略与统一样式配置 ====================
STRATEGIES = [
    'Exponential',
    'Inverse_p1', 'Inverse_p2', 'Inverse_p3',
    'SoftShiftPower_g0.1', 'SoftShiftPower_g0.5', 'SoftShiftPower_g0.8'
]

STRATEGY_LABELS = {
    'Exponential': 'Exponential Decay',
    'Inverse_p1': r'Inverse Dist ($1/d$)',
    'Inverse_p2': r'Inverse Sq ($1/d^2$)',
    'Inverse_p3': r'Inverse Cubic ($1/d^3$)',
    'SoftShiftPower_g0.1': r'Soft Power ($\gamma=0.1$)',
    'SoftShiftPower_g0.5': r'Soft Power ($\gamma=0.5$)',
    'SoftShiftPower_g0.8': r'Soft Power ($\gamma=0.8$)'
}

COLORS = {
    'Exponential': '#333333',
    'Inverse_p1': '#009E73',
    'Inverse_p2': '#0072B2',
    'Inverse_p3': '#D55E00',
    'SoftShiftPower_g0.1': '#4B0082',
    'SoftShiftPower_g0.5': '#984EA3',
    'SoftShiftPower_g0.8': '#E377C2'
}

LINESTYLES = {
    'Exponential': '--',
    'Inverse_p1': '-', 'Inverse_p2': '-', 'Inverse_p3': '-',
    'SoftShiftPower_g0.1': '-', 'SoftShiftPower_g0.5': '-.', 'SoftShiftPower_g0.8': ':'
}

LINEWIDTHS = {
    'Exponential': 2.0,
    'Inverse_p1': 2.0, 'Inverse_p2': 2.0, 'Inverse_p3': 2.0,
    'SoftShiftPower_g0.1': 3.5, 'SoftShiftPower_g0.5': 2.5, 'SoftShiftPower_g0.8': 2.5
}


# ==================== 5. 权重策略函数 ====================
def get_weight_strategy(strategy_name, k, k_c):
    k = np.array(k, dtype=float)
    if k_c <= 1: return np.zeros_like(k)

    if strategy_name == 'Exponential':
        lam = np.log(2) / (k_c - 1)
        return np.exp(-lam * (k - 1))
    elif strategy_name.startswith('Inverse_p'):
        try:
            p = int(strategy_name.split('p')[-1])
        except:
            p = 1
        R = np.power(0.5, 1.0 / p)
        b = 1000.0 if abs(1 - R) < 1e-9 else (R * k_c - 1) / (1 - R)
        base = (1 + b) / (k + b)
        return np.power(base, p)
    elif strategy_name.startswith('SoftShiftPower_g'):
        try:
            gamma = float(strategy_name.split('_g')[-1])
        except:
            gamma = 0.1
        delta = DELTA if DELTA < k_c else k_c - 0.1
        term = np.power(np.maximum(0, k - delta) / (k_c - delta), gamma)
        return 1.0 / (1.0 + term)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")


# ==================== 6. 辅助函数 ====================
def load_predictions(base_dir, step, site_ids):
    file_path = os.path.join(base_dir, f"{step}_step_predictions", "predictions.csv")
    if not os.path.exists(file_path): return None
    df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
    target_cols = [f'NDPI_{i}' for i in site_ids]
    existing_cols = [c for c in target_cols if c in df.columns]
    return df[existing_cols]


def calculate_metrics(df_true, df_pred):
    y_true = df_true.values.flatten()
    y_pred = df_pred.values.flatten()
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if not mask.any(): return np.nan, np.nan, np.nan
    y_t, y_p = y_true[mask], y_pred[mask]
    rmse = np.sqrt(mean_squared_error(y_t, y_p))
    bias = np.mean(y_p - y_t)
    mae = mean_absolute_error(y_t, y_p)
    return rmse, bias, mae


# ==================== 7. 主处理循环 ====================
print("Loading Ground Truth Data...")
if os.path.exists(DATA_FILE):
    df_all = pd.read_csv(DATA_FILE, parse_dates=['Date'], index_col='Date')
    df_obs_2018 = df_all[df_all.index.year == TARGET_YEAR]
    valid_cols = [f'NDPI_{i}' for i in SITES if f'NDPI_{i}' in df_obs_2018.columns]
    df_obs_class1 = df_obs_2018[valid_cols]
else:
    print(f"Error: File not found: {DATA_FILE}")
    df_obs_class1 = pd.DataFrame()

all_results = []
print(f"\nStarting Evaluation (Kc={KC})...")

if not df_obs_class1.empty:
    for strategy in STRATEGIES:
        print(f"  Processing: {STRATEGY_LABELS[strategy]}")
        for step in STEPS_TO_TEST:
            w_trans = float(get_weight_strategy(strategy, step, KC))
            w_lt = 1.0 - w_trans

            df_pred_trans = load_predictions(DIR_TRANS, step, SITES)
            df_pred_lt = load_predictions(DIR_LT, step, SITES)
            if df_pred_trans is None or df_pred_lt is None: continue

            common_idx = df_pred_trans.index.intersection(df_pred_lt.index).intersection(df_obs_class1.index)
            common_cols = df_pred_trans.columns.intersection(df_pred_lt.columns)
            if len(common_idx) == 0: continue

            p_t = df_pred_trans.loc[common_idx, common_cols]
            p_l = df_pred_lt.loc[common_idx, common_cols]
            obs = df_obs_class1.loc[common_idx, common_cols]
            p_fused = w_trans * p_t + w_lt * p_l

            rmse_t, bias_t, mae_t = calculate_metrics(obs, p_t)
            rmse_l, bias_l, mae_l = calculate_metrics(obs, p_l)
            rmse_f, bias_f, mae_f = calculate_metrics(obs, p_fused)

            all_results.append({
                'Strategy': strategy, 'Step': step, 'Weight_Trans': w_trans,
                'RMSE_Trans': rmse_t, 'RMSE_LT': rmse_l, 'RMSE_Fused': rmse_f,
                'Bias_Trans': bias_t, 'Bias_LT': bias_l, 'Bias_Fused': bias_f,
                'AbsBias_Trans': abs(bias_t), 'AbsBias_LT': abs(bias_l), 'AbsBias_Fused': abs(bias_f),
                'MAE_Trans': mae_t, 'MAE_LT': mae_l, 'MAE_Fused': mae_f
            })

df_res = pd.DataFrame(all_results)
if not df_res.empty:
    df_res.to_csv(os.path.join(OUTPUT_DIR, "Full_Comparison_Metrics.csv"), index=False)
    print("CSV saved.")


# ==================== 8. 通用绘图函数 ====================
def plot_metric(metric_col, ylabel, filename,
                baseline_trans=None, baseline_lt=None,
                y_fixed=None, add_zero_line=False):
    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)

    if baseline_trans and baseline_lt:
        base = df_res[df_res['Strategy'] == STRATEGIES[0]]
        if not base.empty:
            if add_zero_line:
                ax.axhline(0, color='black', linewidth=1.5, alpha=0.5)
                ax.plot(base['Step'], base[baseline_trans], color='gray', ls='--',
                        marker='^', ms=9, lw=2, alpha=0.6, label='Transformer (Baseline)')
                ax.plot(base['Step'], base[baseline_lt], color='gray', ls='-.',
                        marker='s', ms=9, lw=2, alpha=0.6, label='CNN (Baseline)')
            else:
                ax.plot(base['Step'], base[baseline_trans], color='gray', linestyle=':',
                        linewidth=1.5, alpha=0.6, label='Transformer (Baseline)')
                ax.plot(base['Step'], base[baseline_lt], color='gray', linestyle='-.',
                        linewidth=1.5, alpha=0.6, label='CNN (Baseline)')

    for strat in STRATEGIES:
        sub = df_res[df_res['Strategy'] == strat]
        if sub.empty: continue
        ax.plot(sub['Step'], sub[metric_col],
                label=STRATEGY_LABELS[strat],
                color=COLORS[strat],
                linestyle=LINESTYLES[strat],
                linewidth=LINEWIDTHS[strat],
                marker='o', markersize=MARKER_SIZE, alpha=0.9)

    ax.set_xlabel('Forecast Horizon (Steps)', fontsize=FONT_SIZE_LABEL, fontweight='bold', labelpad=15)
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE_LABEL, fontweight='bold', labelpad=15)
    ax.set_xticks(X_TICKS_DISPLAY)
    ax.set_xticklabels(X_TICKS_DISPLAY, fontsize=FONT_SIZE_TICK)
    ax.tick_params(axis='y', labelsize=FONT_SIZE_TICK)

    if y_fixed:
        ymin, ymax, n = y_fixed
        ticks = np.linspace(ymin, ymax, n + 1)
        ax.set_yticks(ticks)
        pad = (ymax - ymin) / n * 0.4
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    else:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0),
              frameon=False, fontsize=FONT_SIZE_LEGEND, labelspacing=0.6)
    plt.tight_layout()
    for ext in ['.png', '.pdf']:
        plt.savefig(os.path.join(OUTPUT_DIR, filename.replace('.png', ext)),
                    dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


# ==================== 9. 绘制指标图 ====================
if not df_res.empty:
    print("\n--- Generating Metric Plots ---")
    plot_metric('RMSE_Fused', 'RMSE', 'Comparison_RMSE.png',
                baseline_trans='RMSE_Trans', baseline_lt='RMSE_LT',
                y_fixed=(0.038, 0.050, 4))
    plot_metric('Bias_Fused', 'Mean Error (Bias)', 'Comparison_Bias_Signed.png',
                baseline_trans='Bias_Trans', baseline_lt='Bias_LT',
                add_zero_line=True)
    plot_metric('AbsBias_Fused', 'Absolute Bias (|Mean Error|)', 'Comparison_AbsBias.png',
                baseline_trans='AbsBias_Trans', baseline_lt='AbsBias_LT')
    plot_metric('MAE_Fused', 'MAE', 'Comparison_MAE.png',
                baseline_trans='MAE_Trans', baseline_lt='MAE_LT')


# ==================== 10. 权重曲线（marker 与 RMSE 图一致） ====================
print("\n--- Generating Weight Plot ---")
fig_w, ax_w = plt.subplots(figsize=(14, 8), dpi=300)

k_dense = np.linspace(1, 46, 300)
k_steps = np.array(STEPS_TO_TEST, dtype=float)  # marker 标注位置

for strat in STRATEGIES:
    w_dense = get_weight_strategy(strat, k_dense, KC)
    w_steps = get_weight_strategy(strat, k_steps, KC)

    # 平滑曲线（无 marker，无 label）
    ax_w.plot(k_dense, w_dense,
              color=COLORS[strat],
              linestyle=LINESTYLES[strat],
              linewidth=LINEWIDTHS[strat],
              alpha=0.9)

    # 离散步长位置叠加 marker（带 label，用于图例）
    ax_w.plot(k_steps, w_steps,
              label=STRATEGY_LABELS[strat],
              color=COLORS[strat],
              linestyle='none',
              marker='o', markersize=MARKER_SIZE,
              alpha=0.9)

# KC 竖线 + 0.5 水平参考线
ax_w.axvline(x=KC, color='#333333', linestyle=':', linewidth=2, alpha=0.6,
             label=f'Critical Step $k_c={int(KC)}$')
ax_w.axhline(y=0.5, color='#333333', linestyle=':', linewidth=1, alpha=0.3)

ax_w.set_xlabel('Forecast Horizon (Steps)', fontsize=FONT_SIZE_LABEL, fontweight='bold', labelpad=15)
ax_w.set_ylabel('Weight of Short-term Model', fontsize=FONT_SIZE_LABEL, fontweight='bold', labelpad=15)
ax_w.set_xticks(X_TICKS_DISPLAY)
ax_w.set_xticklabels(X_TICKS_DISPLAY, fontsize=FONT_SIZE_TICK)
ax_w.tick_params(axis='y', labelsize=FONT_SIZE_TICK)
ax_w.set_ylim(-0.02, 1.02)

for spine in ax_w.spines.values():
    spine.set_linewidth(1.5)

ax_w.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0),
            frameon=False, fontsize=FONT_SIZE_LEGEND, labelspacing=0.6)
plt.tight_layout()
for ext in ['.png', '.pdf']:
    plt.savefig(os.path.join(OUTPUT_DIR, f'Weight_Curves{ext}'),
                dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: Weight_Curves (.png + .pdf)")

print("\nDone.")