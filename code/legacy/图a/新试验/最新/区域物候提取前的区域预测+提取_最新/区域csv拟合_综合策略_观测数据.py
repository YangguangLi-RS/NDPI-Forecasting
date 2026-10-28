"""
双Logistic函数批量拟合脚本（针对特定文件 & 特定年份的宽表数据）
功能：
  1. 仅读取指定的 SG 滤波后的观测数据文件。
  2. 自动在表头中筛选出包含 2018 年的 NDPI 时序数据列。
  3. 将NDPI保留4位小数后再拟合（与预测数据精度一致）。
  4. 采用智能混合拟合策略（快速局部优化 + 全局差分进化兜底）。
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

def process_single_file(input_path, output_path, target_year=2018, n_jobs=10):
    file_name = os.path.basename(input_path)
    print(f"\n[{file_name}] 开始处理...")
    start_time = time.time()

    try:
        df = pd.read_csv(input_path)

        class_col = df['Class'] if 'Class' in df.columns else df.iloc[:, 0]

        year_str = str(target_year)
        ndpi_cols = [col for col in df.columns if str(col).startswith(year_str)]

        if len(ndpi_cols) == 0:
            print(f"  -> 错误: 未能在表头中找到以 {target_year} 开头的日期列！")
            return 0, 0

        print(f"  -> 成功提取 {target_year} 年时序: 共找到 {len(ndpi_cols)} 个有效相。")
        ndpi_data = df[ndpi_cols]

        # ===== 关键：保留4位小数，与预测数据精度一致 =====
        ndpi_data = ndpi_data.round(4)

        if 'latitude' in df.columns:
            lat_col = df['latitude']
        elif 'Lat' in df.columns:
            lat_col = df['Lat']
        else:
            lat_col = df.iloc[:, -2]

        if 'longitude' in df.columns:
            lon_col = df['longitude']
        elif 'Lon' in df.columns:
            lon_col = df['Lon']
        else:
            lon_col = df.iloc[:, -1]

        n_pixels = len(df)

        results = Parallel(n_jobs=n_jobs, backend='loky')(
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
        print(f"  -> 拟合完成: 成功 {success_count}/{n_pixels} 像元 "
              f"({success_count/n_pixels*100:.1f}%), 耗时 {elapsed:.1f}s")

        return success_count, n_pixels

    except Exception as e:
        print(f"  -> 处理失败: {e}")
        return 0, 0

# ======================== 主程序 ========================

def main():
    target_files = [
        'data/区域数据1/site3/NDPI_Site3_SG_filtered.csv',
        'data/区域数据1/site6/NDPI_Site6_SG_filtered.csv'
    ]

    root_dir = 'data/预测结果/区域滚动预测_物候'
    output_dir = os.path.join(root_dir, "fitted_results_观测数据")
    os.makedirs(output_dir, exist_ok=True)

    target_year = 2018
    n_jobs = 12

    print(f"=== 双Logistic拟合（观测数据，保留4位小数） ===")
    print(f"目标年份: {target_year}")
    print(f"待处理文件数: {len(target_files)}")
    print(f"输出目录: {output_dir}")
    print("=" * 50)

    total_success = 0
    total_pixels = 0

    for input_path in target_files:
        if not os.path.exists(input_path):
            print(f"\n[错误] 文件不存在: {input_path}")
            continue

        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_2018_fitted_4dp.csv")

        success, total = process_single_file(
            input_path=input_path,
            output_path=output_path,
            target_year=target_year,
            n_jobs=n_jobs
        )
        total_success += success
        total_pixels += total

    print("\n" + "=" * 50)
    print(f"全部完成! 共处理 {total_pixels} 像元, "
          f"成功 {total_success} ({total_success/max(total_pixels,1)*100:.1f}%)")

if __name__ == "__main__":
    main()