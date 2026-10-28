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

FONT_SIZE_LABEL = 24
FONT_SIZE_TICK = 22
FONT_SIZE_LEGEND = 14
LINE_WIDTH = 2.5
MARKER_SIZE = 9
AXIS_LINE_WIDTH = 2.0

# ==================== 2. 实验配置参数 ====================
STEPS_TO_TEST = [1, 2, 3, 4, 5, 6, 12, 23, 34, 46]
X_TICKS_DISPLAY = [1, 3, 6, 12, 23, 34, 46]
TARGET_YEAR = 2018
SITES = list(range(1, 322))

KC = 3.0
DELTA = 1.0

# ==================== 3. 文件路径配置 ====================
DATA_FILE = 'data/站点训练数据_CVmax_2.csv'
DIR_TRANS = 'data/模型多步训练结果/Transformer_specific_step_evaluation_46步长'
DIR_CNN = 'data/模型多步训练结果/CNN_model_specific_step_evaluation_12步_新'
DIR_LSTM = 'data/模型多步训练结果/LSTM_specific_step_evaluation_321_12步长'

OUTPUT_DIR = 'data/模型多步训练结果/三模型加权融合实验_完整分析_KC3_1'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==================== 4. 权重计算核心逻辑 ====================

def get_base_decay_curve(strategy_name, k, k_c):
    k = np.array(k, dtype=float)
    if k_c <= 1: return np.zeros_like(k)
    target_w = 1.0 / 3.0

    if strategy_name == 'Exponential':
        lam = np.log(1.0 / target_w) / (k_c - 1)
        return np.exp(-lam * (k - 1))

    elif strategy_name.startswith('Inverse_p'):
        try:
            p = int(strategy_name.split('p')[-1])
        except:
            p = 1
        R = np.power(target_w, 1.0 / p)
        b = 1000.0 if abs(1 - R) < 1e-9 else (R * k_c - 1) / (1 - R)
        base = (1 + b) / (k + b)
        return np.power(base, p)

    elif strategy_name.startswith('SoftShiftPower_g'):
        try:
            gamma = float(strategy_name.split('_g')[-1])
        except:
            gamma = 0.1
        delta = DELTA
        if delta >= k_c: delta = k_c - 0.1
        numerator = np.maximum(0, k - delta)
        denominator = k_c - delta
        ratio = numerator / denominator
        term = np.power(ratio, gamma)
        return 1.0 / (1.0 + 2.0 * term)

    else:
        return np.zeros_like(k)


def distribute_weights_3models(strategy_name, k_val, w_curve_val):
    if strategy_name == 'Constant_Equal':
        if k_val == 1:
            return (1.0, 0.0, 0.0)
        else:
            return (1 / 3, 1 / 3, 1 / 3)

    if k_val <= 3:
        w_t = w_curve_val
        w_remain = 1.0 - w_t
        w_c = w_remain / 2.0
        w_l = w_remain / 2.0
    else:
        w_t = w_curve_val
        w_l = w_curve_val
        w_c = 1.0 - 2.0 * w_curve_val

    return (w_t, w_c, w_l)


# ==================== 5. 辅助函数 ====================

def load_predictions(base_dir, step, site_ids):
    paths = [
        os.path.join(base_dir, f"{step}_step_predictions", "predictions.csv"),
        os.path.join(base_dir, f"{step}_step_prediction", "predictions.csv")
    ]
    for p in paths:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p, parse_dates=['Date'], index_col='Date')
                target_cols = [f'NDPI_{i}' for i in site_ids]
                existing_cols = [c for c in target_cols if c in df.columns]
                return df[existing_cols]
            except:
                continue
    print(f"  [警告] 步长 {step} 数据未找到")
    return None


def calculate_metrics(df_true, df_pred):
    y_true = df_true.values.flatten()
    y_pred = df_pred.values.flatten()
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if not mask.any(): return np.nan, np.nan, np.nan
    bias = np.mean(y_pred[mask] - y_true[mask])
    rmse = np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
    mae = mean_absolute_error(y_true[mask], y_pred[mask])
    return rmse, bias, mae


# ==================== 6. 策略与统一样式配置 ====================

STRATEGIES = [
    'Accuracy_Weighted',
    'Constant_Equal',
    'Exponential',
    'Inverse_p1',
    'Inverse_p2',
    'Inverse_p3',
    'SoftShiftPower_g0.1',
    'SoftShiftPower_g0.5',
    'SoftShiftPower_g0.8'
]

STRATEGY_LABELS = {
    'Accuracy_Weighted': 'Accuracy Weighted',
    'Constant_Equal': 'Equal Weight (1/3)',
    'Exponential': 'Exponential',
    'Inverse_p1': r'Inverse ($1/d$)',
    'Inverse_p2': r'Inverse ($1/d^2$)',
    'Inverse_p3': r'Inverse ($1/d^3$)',
    'SoftShiftPower_g0.1': r'SoftPower ($\gamma=0.1$)',
    'SoftShiftPower_g0.5': r'SoftPower ($\gamma=0.5$)',
    'SoftShiftPower_g0.8': r'SoftPower ($\gamma=0.8$)'
}

COLORS = {
    'Accuracy_Weighted': '#d62728',
    'Constant_Equal': '#000000',
    'Exponential': '#333333',
    'Inverse_p1': '#009E73', 'Inverse_p2': '#0072B2', 'Inverse_p3': '#D55E00',
    'SoftShiftPower_g0.1': '#4B0082', 'SoftShiftPower_g0.5': '#984EA3', 'SoftShiftPower_g0.8': '#E377C2'
}

LINESTYLES = {
    'Accuracy_Weighted': '-',
    'Constant_Equal': '-',
    'Exponential': '--',
    'Inverse_p1': '-', 'Inverse_p2': '-', 'Inverse_p3': '-',
    'SoftShiftPower_g0.1': '-', 'SoftShiftPower_g0.5': '-.', 'SoftShiftPower_g0.8': ':'
}

LINEWIDTHS = {
    'Accuracy_Weighted': 3.0,
    'Constant_Equal': 3.0,
    'SoftShiftPower_g0.1': 3.0,
    'Exponential': 2.0,
    'Inverse_p1': 2.0, 'Inverse_p2': 2.0, 'Inverse_p3': 2.0,
    'SoftShiftPower_g0.5': 2.0, 'SoftShiftPower_g0.8': 2.0
}

# --- 与指标图一致的 marker / zorder ---
MARKERS = {
    'Accuracy_Weighted': 'D',
    'Constant_Equal': 's',
    'Exponential': 'o',
    'Inverse_p1': 'o', 'Inverse_p2': 'o', 'Inverse_p3': 'o',
    'SoftShiftPower_g0.1': 'o', 'SoftShiftPower_g0.5': 'o', 'SoftShiftPower_g0.8': 'o'
}

ZORDERS = {
    'Accuracy_Weighted': 10,
    'SoftShiftPower_g0.1': 10,
    'Constant_Equal': 3,
    'Exponential': 3,
    'Inverse_p1': 3, 'Inverse_p2': 3, 'Inverse_p3': 3,
    'SoftShiftPower_g0.5': 3, 'SoftShiftPower_g0.8': 3
}


# ==================== 7. 主程序 ====================

print("正在加载真实值数据...")
if os.path.exists(DATA_FILE):
    df_all = pd.read_csv(DATA_FILE, parse_dates=['Date'], index_col='Date')
    df_obs_2018 = df_all[df_all.index.year == TARGET_YEAR]
    valid_cols = [f'NDPI_{i}' for i in SITES if f'NDPI_{i}' in df_obs_2018.columns]
    df_obs_class1 = df_obs_2018[valid_cols]
else:
    print(f"Error: Data file not found {DATA_FILE}")
    exit()

all_results = []
acc_weighted_trans_weights = {}

print(f"\n开始三模型融合分析 (Kc={KC}, Equal Split 1/3)...")

for strategy in STRATEGIES:
    print(f"  Processing: {STRATEGY_LABELS[strategy]}")

    for step in STEPS_TO_TEST:
        df_t = load_predictions(DIR_TRANS, step, SITES)
        df_c = load_predictions(DIR_CNN, step, SITES)
        df_l = load_predictions(DIR_LSTM, step, SITES)

        if df_t is None or df_c is None or df_l is None: continue

        common_idx = df_t.index.intersection(df_c.index).intersection(df_l.index).intersection(df_obs_class1.index)
        common_cols = df_t.columns.intersection(df_c.columns).intersection(df_l.columns)
        if len(common_idx) == 0: continue

        p_t = df_t.loc[common_idx, common_cols]
        p_c = df_c.loc[common_idx, common_cols]
        p_l = df_l.loc[common_idx, common_cols]
        obs = df_obs_class1.loc[common_idx, common_cols]

        if strategy == 'Accuracy_Weighted':
            rmse_t, _, _ = calculate_metrics(obs, p_t)
            rmse_c, _, _ = calculate_metrics(obs, p_c)
            rmse_l, _, _ = calculate_metrics(obs, p_l)
            inv_t = 1.0 / (rmse_t + 1e-9)
            inv_c = 1.0 / (rmse_c + 1e-9)
            inv_l = 1.0 / (rmse_l + 1e-9)
            total = inv_t + inv_c + inv_l
            w_t, w_c, w_l = inv_t / total, inv_c / total, inv_l / total
            if strategy not in acc_weighted_trans_weights:
                acc_weighted_trans_weights[strategy] = {}
            acc_weighted_trans_weights[strategy][step] = w_t
        else:
            w_val = 0 if strategy == 'Constant_Equal' else float(get_base_decay_curve(strategy, step, KC))
            w_t, w_c, w_l = distribute_weights_3models(strategy, step, w_val)

        p_fused = w_t * p_t + w_c * p_c + w_l * p_l

        rmse_fused, bias_fused, mae_fused = calculate_metrics(obs, p_fused)
        rmse_t, bias_t, mae_t = calculate_metrics(obs, p_t)
        rmse_c, bias_c, mae_c = calculate_metrics(obs, p_c)
        rmse_l, bias_l, mae_l = calculate_metrics(obs, p_l)

        all_results.append({
            'Strategy': strategy, 'Step': step,
            'RMSE': rmse_fused, 'Bias': bias_fused,
            'AbsBias': abs(bias_fused), 'MAE': mae_fused,
            'Weight_Trans': w_t,
            'RMSE_T': rmse_t, 'RMSE_C': rmse_c, 'RMSE_L': rmse_l,
            'Bias_T': bias_t, 'Bias_C': bias_c, 'Bias_L': bias_l,
            'AbsBias_T': abs(bias_t), 'AbsBias_C': abs(bias_c), 'AbsBias_L': abs(bias_l),
            'MAE_T': mae_t, 'MAE_C': mae_c, 'MAE_L': mae_l
        })

# ==================== 8. 绘图与保存 ====================
if not all_results:
    print("没有生成数据。")
else:
    df_res = pd.DataFrame(all_results)
    csv_path = os.path.join(OUTPUT_DIR, "Full_Comparison_3Models_9Strategies.csv")
    df_res.to_csv(csv_path, index=False)
    print(f"CSV Saved: {csv_path}")


    def plot_metric_curve(metric_col, ylabel, filename, add_baseline_mode=None):
        fig, ax = plt.subplots(figsize=(14, 8), dpi=300)

        base_data = df_res[df_res['Strategy'] == STRATEGIES[1]]
        if not base_data.empty and add_baseline_mode:
            if add_baseline_mode == 'Bias':
                ax.axhline(0, color='gray', linewidth=1)
                ax.plot(base_data['Step'], base_data['Bias_T'], color='gray', linestyle=':', marker='^', markersize=6,
                        alpha=0.4, label='Transformer (Baseline)')
                ax.plot(base_data['Step'], base_data['Bias_C'], color='gray', linestyle=':', marker='s', markersize=6,
                        alpha=0.4, label='CNN (Baseline)')
                ax.plot(base_data['Step'], base_data['Bias_L'], color='gray', linestyle=':', marker='d', markersize=6,
                        alpha=0.4, label='LSTM (Baseline)')
            elif add_baseline_mode == 'AbsBias':
                ax.plot(base_data['Step'], base_data['AbsBias_T'], color='gray', linestyle=':', marker='^',
                        markersize=6, alpha=0.4, label='Transformer (Baseline)')
                ax.plot(base_data['Step'], base_data['AbsBias_C'], color='gray', linestyle=':', marker='s',
                        markersize=6, alpha=0.4, label='CNN (Baseline)')
                ax.plot(base_data['Step'], base_data['AbsBias_L'], color='gray', linestyle=':', marker='d',
                        markersize=6, alpha=0.4, label='LSTM (Baseline)')
            elif add_baseline_mode == 'RMSE':
                ax.plot(base_data['Step'], base_data['RMSE_T'], color='gray', linestyle=':', marker='^', markersize=7,
                        alpha=0.6, label='Transformer (Short)')
                ax.plot(base_data['Step'], base_data['RMSE_C'], color='gray', linestyle=':', marker='s', markersize=7,
                        alpha=0.6, label='CNN (Long)')
                ax.plot(base_data['Step'], base_data['RMSE_L'], color='gray', linestyle=':', marker='d', markersize=7,
                        alpha=0.6, label='LSTM (Aux)')
            elif add_baseline_mode == 'MAE':
                ax.plot(base_data['Step'], base_data['MAE_T'], color='gray', linestyle=':', marker='^', markersize=7,
                        alpha=0.6, label='Transformer (Short)')
                ax.plot(base_data['Step'], base_data['MAE_C'], color='gray', linestyle=':', marker='s', markersize=7,
                        alpha=0.6, label='CNN (Long)')
                ax.plot(base_data['Step'], base_data['MAE_L'], color='gray', linestyle=':', marker='d', markersize=7,
                        alpha=0.6, label='LSTM (Aux)')

        for strat in STRATEGIES:
            subset = df_res[df_res['Strategy'] == strat]
            if subset.empty: continue
            ax.plot(subset['Step'], subset[metric_col],
                    label=STRATEGY_LABELS[strat],
                    color=COLORS[strat],
                    linestyle=LINESTYLES[strat],
                    linewidth=LINEWIDTHS[strat],
                    marker=MARKERS[strat], markersize=8,
                    alpha=0.85, zorder=ZORDERS[strat])

        ax.set_xlabel('Forecast Horizon (Steps)', fontsize=FONT_SIZE_LABEL, fontweight='bold', labelpad=15)
        ax.set_ylabel(ylabel, fontsize=FONT_SIZE_LABEL, fontweight='bold', labelpad=15)
        ax.set_xticks(X_TICKS_DISPLAY)
        ax.set_xticklabels(X_TICKS_DISPLAY, fontsize=FONT_SIZE_TICK)

        if add_baseline_mode == 'RMSE':
            TICK_MIN = 0.038
            TICK_MAX = 0.050
            NUM_INTERVALS = 4
            ticks = np.linspace(TICK_MIN, TICK_MAX, NUM_INTERVALS + 1)
            ax.set_yticks(ticks)
            step_size = (TICK_MAX - TICK_MIN) / NUM_INTERVALS
            ax.set_ylim(TICK_MIN - step_size * 0.4, TICK_MAX + step_size * 0.4)
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        if add_baseline_mode in ['Bias', 'AbsBias']:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        if add_baseline_mode == 'MAE':
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

        ax.tick_params(axis='y', labelsize=FONT_SIZE_TICK)
        for spine in ax.spines.values():
            spine.set_linewidth(AXIS_LINE_WIDTH)

        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=FONT_SIZE_LEGEND)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches='tight')
        print(f"Plot Saved: {filename}")


    # 1. RMSE
    plot_metric_curve('RMSE', 'RMSE', 'Comparison_RMSE_3Models.png', add_baseline_mode='RMSE')
    # 2. Bias
    plot_metric_curve('Bias', 'Mean Error (Bias)', 'Comparison_SignedBias_3Models.png', add_baseline_mode='Bias')
    # 3. AbsBias
    plot_metric_curve('AbsBias', 'Absolute Bias (|Mean Error|)', 'Comparison_AbsBias_3Models.png',
                      add_baseline_mode='AbsBias')
    # 4. MAE
    plot_metric_curve('MAE', 'MAE', 'Comparison_MAE_3Models.png', add_baseline_mode='MAE')

    # ==================== 9. 权重图（marker 与指标图一致） ====================
    fig_w, ax_w = plt.subplots(figsize=(14, 8), dpi=300)
    k_dense = np.linspace(1, 46, 300)
    k_steps = np.array(STEPS_TO_TEST, dtype=float)

    for strat in STRATEGIES:
        color = COLORS[strat]
        lw = LINEWIDTHS[strat]
        ls = LINESTYLES[strat]
        mk = MARKERS[strat]
        zo = ZORDERS[strat]
        label = STRATEGY_LABELS[strat]

        if strat == 'Accuracy_Weighted':
            if strat in acc_weighted_trans_weights:
                steps_sorted = sorted(acc_weighted_trans_weights[strat].keys())
                weights_sorted = [acc_weighted_trans_weights[strat][s] for s in steps_sorted]
                ax_w.plot(steps_sorted, weights_sorted,
                          color=color, linestyle=ls, linewidth=lw,
                          marker=mk, markersize=8, label=label,
                          alpha=0.85, zorder=zo)

        elif strat == 'Constant_Equal':
            ax_w.plot(k_dense, np.full_like(k_dense, 1.0 / 3.0),
                      color=color, linestyle=ls, linewidth=lw, alpha=0.85)
            w_steps_const = [1.0 if s == 1 else 1.0 / 3.0 for s in STEPS_TO_TEST]
            ax_w.plot(k_steps, w_steps_const,
                      color=color, linestyle='none',
                      marker=mk, markersize=8, label=label,
                      alpha=0.85, zorder=zo)

        else:
            # 平滑曲线（无 marker、无 label）
            w_dense = get_base_decay_curve(strat, k_dense, KC)
            ax_w.plot(k_dense, w_dense,
                      color=color, linestyle=ls, linewidth=lw, alpha=0.85)

            # 离散步长位置叠加 marker（带 label）
            w_at_steps = get_base_decay_curve(strat, k_steps, KC)
            ax_w.plot(k_steps, w_at_steps,
                      color=color, linestyle='none',
                      marker=mk, markersize=8, label=label,
                      alpha=0.85, zorder=zo)

    ax_w.axvline(x=KC, color='gray', linestyle=':', linewidth=2,
                 label=f'Critical Step $k_c={int(KC)}$')
    ax_w.axhline(y=1.0 / 3.0, color='gray', linestyle=':', linewidth=1, alpha=0.3)

    ax_w.set_xlabel('Forecast Horizon (Steps)', fontsize=FONT_SIZE_LABEL, fontweight='bold', labelpad=15)
    ax_w.set_ylabel('Short-term Model Weight', fontsize=FONT_SIZE_LABEL, fontweight='bold', labelpad=15)
    ax_w.set_xticks(X_TICKS_DISPLAY)
    ax_w.set_xticklabels(X_TICKS_DISPLAY, fontsize=FONT_SIZE_TICK)
    ax_w.tick_params(axis='y', labelsize=FONT_SIZE_TICK)
    ax_w.set_ylim(-0.05, 1.05)

    for spine in ax_w.spines.values():
        spine.set_linewidth(AXIS_LINE_WIDTH)

    ax_w.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=FONT_SIZE_LEGEND)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Weight_Curves_3Models.png"), dpi=300, bbox_inches='tight')
    print("Weight Plot Saved.")

print("Done.")