"""
批量双Logistic拟合：扫描区域滚动预测的输出目录，对所有预测CSV进行拟合。
输入CSV结构：Class | 46列NDPI | latitude | longitude（共49列）
输出：同目录下 fitted/ 子文件夹，文件名加 _fitted 后缀。
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, differential_evolution
from joblib import Parallel, delayed
import os
import glob
import time
import warnings

warnings.filterwarnings('ignore')
np.seterr(over='ignore')

# ======================== 配置 ========================

# 预测文件根目录
INPUT_ROOT = 'data/预测结果/区域滚动预测_物候'

# 并行核心数
N_JOBS = 10

# ======================== 双Logistic模型 ========================

def dual_logistic(t, L1, k1, t1, L2, k2, t2, baseline):
    rise = L1 / (1 + np.exp(-np.clip(k1 * (t - t1), -500, 500)))
    fall = L2 / (1 + np.exp(-np.clip(k2 * (t - t2), -500, 500)))
    return baseline + rise - fall

# ======================== 策略一：高速智能拟合 ========================

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
    L1_init = np.clip(peak_val * 1.2, 0.1, 3.0)
    L2_init = np.clip(peak_val * 1.0, 0.1, 3.0)
    return [L1_init, 0.05, t1_init, L2_init, 0.05, t2_init, baseline_init]


def fast_fit(t, y):
    p0 = smart_initialize(t, y)
    bounds_lower = [0.1, 0.01, t[0],          0.1, 0.01, t[len(t)//3],  max(np.percentile(y, 5), -0.5)]
    bounds_upper = [3.0, 2.0,  t[len(t)*2//3], 3.0, 2.0,  t[-1],        np.percentile(y, 25)]
    p0 = [np.clip(p0[i], bounds_lower[i], bounds_upper[i]) for i in range(7)]
    best_params, _ = curve_fit(
        dual_logistic, t, y, p0=p0,
        bounds=(bounds_lower, bounds_upper), maxfev=5000, method='trf')
    return best_params

# ======================== 策略二：全局稳健兜底 ========================

def global_fallback_fit(t, y):
    bounds = [
        (0.1, 3.0), (0.01, 2.0), (t[0], t[len(t)*2//3]),
        (0.1, 3.0), (0.01, 2.0), (t[len(t)//3], t[-1]),
        (max(np.percentile(y, 5), -0.5), np.percentile(y, 25))
    ]
    def loss(params):
        return np.mean((dual_logistic(t, *params) - y) ** 2)
    result = differential_evolution(loss, bounds, maxiter=80, tol=1e-6, polish=False)
    p0 = result.x
    try:
        best_params, _ = curve_fit(
            dual_logistic, t, y, p0=p0,
            bounds=([b[0] for b in bounds], [b[1] for b in bounds]),
            maxfev=3000, method='trf')
    except RuntimeError:
        best_params = p0
    return best_params

# ======================== 单像元处理 ========================

def process_single_pixel(y_values):
    try:
        y = np.array(y_values, dtype=np.float64)
        if np.isnan(y).any() or len(y) != 46:
            return np.full(365, np.nan)

        t_8day = np.arange(46) * 8
        t_daily = np.arange(0, 365)

        try:
            params = fast_fit(t_8day, y)
            y_daily = dual_logistic(t_daily, *params)
            if np.any(np.isnan(y_daily)) or np.any(np.abs(y_daily) > 5):
                raise ValueError("快速拟合结果异常")
        except Exception:
            params = global_fallback_fit(t_8day, y)
            y_daily = dual_logistic(t_daily, *params)
            if np.any(np.isnan(y_daily)) or np.any(np.abs(y_daily) > 5):
                return np.full(365, np.nan)

        return y_daily
    except Exception:
        return np.full(365, np.nan)

# ======================== 单文件处理 ========================

def process_single_file(input_path, output_path):
    fname = os.path.basename(input_path)
    start_time = time.time()

    try:
        df = pd.read_csv(input_path)

        class_col = df.iloc[:, 0]
        ndpi_data = df.iloc[:, 1:47]   # 46列NDPI
        lat_col = df.iloc[:, -2]       # 倒数第二列
        lon_col = df.iloc[:, -1]       # 最后一列

        n_pixels = len(df)

        results = Parallel(n_jobs=N_JOBS, backend='loky')(
            delayed(process_single_pixel)(ndpi_data.iloc[i].values)
            for i in range(n_pixels)
        )

        daily_columns = [f"DOY_{d}" for d in range(365)]
        df_daily = pd.DataFrame(results, columns=daily_columns)

        df_out = pd.concat([
            class_col.reset_index(drop=True),
            df_daily,
            lat_col.reset_index(drop=True),
            lon_col.reset_index(drop=True)
        ], axis=1)

        df_out.to_csv(output_path, index=False)

        success_count = sum(1 for r in results if not np.isnan(r[0]))
        elapsed = time.time() - start_time
        print(f"    完成: {success_count}/{n_pixels} 成功 "
              f"({success_count/n_pixels*100:.1f}%), 耗时 {elapsed:.1f}s")

    except Exception as e:
        print(f"    处理失败: {e}")

# ======================== 主程序 ========================

def main():
    print(f"扫描目录: {INPUT_ROOT}")
    print(f"并行核心数: {N_JOBS}")
    print("=" * 60)

    # 递归扫描所有 Rolling_*.csv 文件（排除已拟合的）
    all_csvs = glob.glob(os.path.join(INPUT_ROOT, "**", "Rolling_*.csv"), recursive=True)
    all_csvs = [f for f in all_csvs if "_fitted" not in f and "fitted" not in os.path.dirname(f)]

    print(f"找到 {len(all_csvs)} 个预测CSV待拟合\n")

    for i, input_path in enumerate(all_csvs):
        # 输出到同级目录下的 fitted/ 子文件夹
        parent_dir = os.path.dirname(input_path)
        fitted_dir = os.path.join(parent_dir, "fitted")
        os.makedirs(fitted_dir, exist_ok=True)

        base_name = os.path.basename(input_path).replace(".csv", "_fitted.csv")
        output_path = os.path.join(fitted_dir, base_name)

        # 跳过已拟合的
        if os.path.exists(output_path):
            print(f"[{i+1}/{len(all_csvs)}] 已存在，跳过: {base_name}")
            continue

        print(f"[{i+1}/{len(all_csvs)}] {os.path.basename(input_path)}")
        process_single_file(input_path, output_path)

    print("\n" + "=" * 60)
    print("全部拟合完成")


if __name__ == "__main__":
    main()