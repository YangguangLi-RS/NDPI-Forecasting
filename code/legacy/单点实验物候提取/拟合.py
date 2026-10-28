"""
单点尺度双Logistic拟合
输入：321个站点的46×46预测矩阵（无表头，值域[-1,1]）
输出：每个站点一个46×365的拟合矩阵CSV（每行对应一个obs_count的日尺度曲线）
同时对观测数据做拟合，输出321个站点各一条365天曲线。
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, differential_evolution
from joblib import Parallel, delayed
import os
import time
import warnings

warnings.filterwarnings('ignore')
np.seterr(over='ignore')

# ==============================================================================
# 配置
# ==============================================================================

TOTAL_STATIONS = 321
N_JOBS = 12

OBS_PATH = 'data/站点训练数据_CVmax_2.csv'
PRED_BASE = 'data/预测结果/单点尺度预测'

MODEL_FOLDERS = [
    "CNN12_带季节性_滚动预测",
    "CNN12_无季节性_滚动预测",
    "CNN46_带季节性_滚动预测",
    "CNN46_无季节性_滚动预测",
    "L12_带季节性_滚动预测",
    "L12_无季节性_滚动预测",
    "L46_带季节性_滚动预测",
    "L46_无季节性_滚动预测",
    "MLP12_带季节性_滚动预测",
    "MLP12_无季节性_滚动预测",
    "MLP46_带季节性_滚动预测",
    "MLP46_无季节性_滚动预测",
    "T12_带季节性_滚动预测",
    "T12_不带季节性_滚动预测",
    "T46_带季节性_滚动预测",
    "T46_不带季节性_滚动预测",
]

OUTPUT_BASE = os.path.join(PRED_BASE, "拟合结果")
os.makedirs(OUTPUT_BASE, exist_ok=True)


# ==============================================================================
# 双Logistic拟合核心
# ==============================================================================

def dual_logistic(t, L1, k1, t1, L2, k2, t2, baseline):
    rise = L1 / (1 + np.exp(-np.clip(k1 * (t - t1), -500, 500)))
    fall = L2 / (1 + np.exp(-np.clip(k2 * (t - t2), -500, 500)))
    return baseline + rise - fall


def smart_initialize(t, y):
    baseline_init = np.percentile(y, 10)
    y_shifted = y - baseline_init
    grad = np.gradient(y)
    window = max(3, len(y) // 10)
    smooth_grad = np.convolve(grad, np.ones(window) / window, mode='same')
    max_idx = np.argmax(smooth_grad)
    t1_init = t[max_idx]
    min_idx = np.argmin(smooth_grad[max_idx:]) + max_idx
    t2_init = t[min_idx] if min_idx < len(t) else t[-1]
    peak_val = np.max(y_shifted)
    return [np.clip(peak_val*1.2, 0.1, 3.0), 0.05, t1_init,
            np.clip(peak_val*1.0, 0.1, 3.0), 0.05, t2_init, baseline_init]


def fast_fit(t, y):
    p0 = smart_initialize(t, y)
    bl = [0.1, 0.01, t[0],          0.1, 0.01, t[len(t)//3],  max(np.percentile(y,5), -0.5)]
    bu = [3.0, 2.0,  t[len(t)*2//3], 3.0, 2.0,  t[-1],        np.percentile(y,25)]
    p0 = [np.clip(p0[i], bl[i], bu[i]) for i in range(7)]
    params, _ = curve_fit(dual_logistic, t, y, p0=p0, bounds=(bl, bu), maxfev=5000, method='trf')
    return params


def global_fallback_fit(t, y):
    bounds = [(0.1,3.0),(0.01,2.0),(t[0],t[len(t)*2//3]),
              (0.1,3.0),(0.01,2.0),(t[len(t)//3],t[-1]),
              (max(np.percentile(y,5),-0.5), np.percentile(y,25))]
    def loss(p): return np.mean((dual_logistic(t,*p)-y)**2)
    res = differential_evolution(loss, bounds, maxiter=80, tol=1e-6, polish=False)
    try:
        params, _ = curve_fit(dual_logistic, t, y, p0=res.x,
                              bounds=([b[0] for b in bounds],[b[1] for b in bounds]),
                              maxfev=3000, method='trf')
    except:
        params = res.x
    return params


def fit_single_curve(y_46):
    """对46个点做双Logistic拟合，返回365天日尺度曲线"""
    try:
        y = np.array(y_46, dtype=np.float64)
        if np.isnan(y).any() or len(y) != 46:
            return np.full(365, np.nan)

        t_8day = np.arange(46) * 8
        t_daily = np.arange(0, 365)

        try:
            params = fast_fit(t_8day, y)
            y_daily = dual_logistic(t_daily, *params)
            if np.any(np.isnan(y_daily)) or np.any(np.abs(y_daily) > 5):
                raise ValueError
        except:
            params = global_fallback_fit(t_8day, y)
            y_daily = dual_logistic(t_daily, *params)
            if np.any(np.isnan(y_daily)) or np.any(np.abs(y_daily) > 5):
                return np.full(365, np.nan)

        return y_daily
    except:
        return np.full(365, np.nan)


# ==============================================================================
# 观测数据拟合
# ==============================================================================

def fit_obs():
    obs_output_dir = os.path.join(OUTPUT_BASE, "Obs_2018")
    os.makedirs(obs_output_dir, exist_ok=True)

    # 检查是否已全部完成
    existing = len([f for f in os.listdir(obs_output_dir) if f.endswith('.csv')])
    if existing >= TOTAL_STATIONS:
        print(f"[观测] 已全部完成 ({existing} 个文件)，跳过")
        return

    print("[观测] 加载数据...")
    df = pd.read_csv(OBS_PATH, parse_dates=["Date"], index_col=0)
    df_2018 = df[df.index.year == 2018]

    print(f"[观测] 拟合 {TOTAL_STATIONS} 个站点...")
    start_time = time.time()

    tasks = []
    for sid in range(1, TOTAL_STATIONS + 1):
        y = df_2018[f'NDPI_{sid}'].values
        tasks.append((sid, y))

    results = Parallel(n_jobs=N_JOBS, backend='loky')(
        delayed(fit_single_curve)(y) for sid, y in tasks
    )

    for i, (sid, _) in enumerate(tasks):
        output_path = os.path.join(obs_output_dir, f"Station_{sid}_fitted.csv")
        # 观测只有1条曲线，保存为1×365
        np.savetxt(output_path, [results[i]], delimiter=',', fmt='%.6f')

    success = sum(1 for r in results if not np.isnan(r[0]))
    elapsed = time.time() - start_time
    print(f"[观测] 完成: {success}/{TOTAL_STATIONS} 成功, 耗时 {elapsed:.1f}s")


# ==============================================================================
# 预测数据拟合
# ==============================================================================

def fit_single_station_matrix(csv_path):
    """拟合一个站点的46×46矩阵，返回46×365的拟合矩阵"""
    try:
        mat = np.loadtxt(csv_path, delimiter=',')
        if mat.shape != (46, 46):
            return None

        results = []
        for row_idx in range(46):
            y_daily = fit_single_curve(mat[row_idx, :])
            results.append(y_daily)

        return np.array(results)  # (46, 365)
    except:
        return None


def fit_model_folder(model_folder):
    model_dir = os.path.join(PRED_BASE, model_folder)
    if not os.path.isdir(model_dir):
        print(f"  [跳过] 不存在: {model_folder}")
        return

    output_dir = os.path.join(OUTPUT_BASE, model_folder)
    os.makedirs(output_dir, exist_ok=True)

    # 检查已完成数量
    existing = len([f for f in os.listdir(output_dir) if f.endswith('.csv')])
    if existing >= TOTAL_STATIONS:
        print(f"  [跳过] {model_folder} 已全部完成 ({existing} 个)")
        return

    print(f"  处理: {model_folder}")
    start_time = time.time()

    # 收集需要处理的站点
    tasks = []
    for sid in range(1, TOTAL_STATIONS + 1):
        csv_path = os.path.join(model_dir, f"Station_{sid}_Predictions.csv")
        output_path = os.path.join(output_dir, f"Station_{sid}_fitted.csv")

        if os.path.exists(output_path):
            continue  # 跳过已拟合的

        if not os.path.exists(csv_path):
            continue

        tasks.append((sid, csv_path, output_path))

    if not tasks:
        print(f"    全部已完成或无数据")
        return

    print(f"    待处理: {len(tasks)} 个站点")

    # 并行拟合
    fitted_results = Parallel(n_jobs=N_JOBS, backend='loky')(
        delayed(fit_single_station_matrix)(csv_path) for sid, csv_path, _ in tasks
    )

    # 保存
    success = 0
    for i, (sid, _, output_path) in enumerate(tasks):
        if fitted_results[i] is not None:
            np.savetxt(output_path, fitted_results[i], delimiter=',', fmt='%.6f')
            success += 1

    elapsed = time.time() - start_time
    print(f"    完成: {success}/{len(tasks)} 成功, 耗时 {elapsed:.1f}s")


# ==============================================================================
# 主程序
# ==============================================================================

def main():
    print("=" * 60)
    print("单点尺度双Logistic拟合")
    print(f"站点数: {TOTAL_STATIONS}, 并行: {N_JOBS} 核")
    print(f"输出: {OUTPUT_BASE}")
    print("=" * 60)

    # 观测
    fit_obs()

    # 预测
    print(f"\n--- 预测数据拟合 ({len(MODEL_FOLDERS)} 个配置) ---")
    for folder in MODEL_FOLDERS:
        fit_model_folder(folder)

    print("\n" + "=" * 60)
    print("拟合全部完成")


if __name__ == "__main__":
    main()