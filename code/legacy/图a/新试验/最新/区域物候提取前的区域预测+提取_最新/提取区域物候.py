"""
物候提取脚本（滚动预测版）
功能：基于双Logistic拟合后的日尺度NDPI曲线，采用动态阈值法提取6个物候日期。
输入：拟合后的CSV文件（Class + DOY_0~DOY_364 + latitude + longitude）
输出：每个像元的6个物候日期 + Class + 经纬度

处理范围：
  1. 观测数据（2个文件）
  2. 预测数据（扫描 fitted/ 子文件夹下所有 *_fitted.csv）
"""

import numpy as np
import pandas as pd
import os
import glob
import time


def extract_phenology_single_pixel(vi_values):
    """
    对单个像元提取6个物候日期
    返回: [SOS_20, SOS_50, SOS_80, EOS_80, EOS_50, EOS_20]
    """
    try:
        vi = np.array(vi_values, dtype=np.float64)

        if np.isnan(vi).any() or len(vi) != 365:
            return np.full(6, np.nan)

        baseline = np.mean(np.concatenate([vi[:30], vi[-30:]]))
        peak = np.max(vi)
        amplitude = peak - baseline

        if amplitude < 0.02:
            return np.full(6, np.nan)

        max_idx = np.argmax(vi)

        thresholds = [0.2, 0.5, 0.8]
        target_values = [baseline + amplitude * th for th in thresholds]

        # 春季：从左往右找
        spring_dates = np.zeros(3)
        for j, target in enumerate(target_values):
            found = False
            for k in range(0, max_idx + 1):
                if vi[k] >= target:
                    spring_dates[j] = k
                    found = True
                    break
            if not found:
                diff = np.abs(vi[:max_idx + 1] - target)
                spring_dates[j] = np.argmin(diff)

        # 秋季：从右往左找
        autumn_dates = np.zeros(3)
        for j, target in enumerate(target_values):
            found = False
            for k in range(364, max_idx - 1, -1):
                if vi[k] >= target:
                    autumn_dates[j] = k
                    found = True
                    break
            if not found:
                diff = np.abs(vi[max_idx:] - target)
                autumn_dates[j] = max_idx + np.argmin(diff)

        # DOY从1开始
        return np.array([
            spring_dates[0] + 1, spring_dates[1] + 1, spring_dates[2] + 1,
            autumn_dates[2] + 1, autumn_dates[1] + 1, autumn_dates[0] + 1
        ])

    except Exception:
        return np.full(6, np.nan)


def process_single_file(input_path, output_path):
    file_name = os.path.basename(input_path)
    start_time = time.time()

    try:
        df = pd.read_csv(input_path)

        class_col = df.iloc[:, 0]
        vi_data = df.iloc[:, 1:366]
        lat_col = df.iloc[:, 366]
        lon_col = df.iloc[:, 367]

        n_pixels = len(df)
        results = []

        for i in range(n_pixels):
            phenology = extract_phenology_single_pixel(vi_data.iloc[i].values)
            results.append(phenology)

        phenology_columns = ['SOS_20', 'SOS_50', 'SOS_80', 'EOS_80', 'EOS_50', 'EOS_20']
        df_phenology = pd.DataFrame(results, columns=phenology_columns)

        df_out = pd.concat([
            class_col.reset_index(drop=True),
            df_phenology,
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


def main():
    ROOT = 'data/预测结果/区域滚动预测_物候'

    # ========== 1. 观测数据 ==========
    obs_files = [
        os.path.join(ROOT, "fitted_results_观测数据", "NDPI_Site3_SG_filtered_2018_fitted.csv"),
        os.path.join(ROOT, "fitted_results_观测数据", "NDPI_Site6_SG_filtered_2018_fitted.csv"),
    ]

    obs_output_dir = os.path.join(ROOT, "phenology_观测数据")
    os.makedirs(obs_output_dir, exist_ok=True)

    print("=" * 60)
    print("物候提取 - 动态阈值法 (20%, 50%, 80%)")
    print("=" * 60)

    print(f"\n--- 观测数据 ({len(obs_files)} 个文件) ---")
    for input_path in obs_files:
        if not os.path.exists(input_path):
            print(f"  [跳过] 不存在: {os.path.basename(input_path)}")
            continue
        base_name = os.path.basename(input_path).replace("_fitted.csv", "_phenology.csv")
        output_path = os.path.join(obs_output_dir, base_name)
        print(f"  {os.path.basename(input_path)}")
        process_single_file(input_path, output_path)

    # ========== 2. 预测数据（扫描所有 fitted/ 子文件夹） ==========
    pred_fitted_files = glob.glob(os.path.join(ROOT, "**", "fitted", "*_fitted.csv"), recursive=True)
    print(f"\n--- 预测数据 ({len(pred_fitted_files)} 个文件) ---")

    pred_output_dir = os.path.join(ROOT, "phenology_预测数据")
    os.makedirs(pred_output_dir, exist_ok=True)

    for i, input_path in enumerate(pred_fitted_files):
        base_name = os.path.basename(input_path).replace("_fitted.csv", "_phenology.csv")
        output_path = os.path.join(pred_output_dir, base_name)

        # 跳过已提取的
        if os.path.exists(output_path):
            continue

        print(f"  [{i+1}/{len(pred_fitted_files)}] {base_name}")
        process_single_file(input_path, output_path)

    print("\n" + "=" * 60)
    print("全部物候提取完成")


if __name__ == "__main__":
    main()