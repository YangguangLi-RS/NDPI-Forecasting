"""
单点尺度物候提取
输入：拟合结果（每个站点一个CSV，观测1×365，预测46×365）
输出：每个模型配置一个汇总CSV（Station × obs_count × 6个物候参数）
     观测一个汇总CSV（Station × 6个物候参数）

物候提取方法：动态阈值法（与区域版完全一致）
  1. baseline = 前30天和后30天均值
  2. amplitude = peak - baseline
  3. 春季：从左往右找达到 20%/50%/80% 振幅的DOY
  4. 秋季：从右往左找达到 80%/50%/20% 振幅的DOY
"""

import numpy as np
import pandas as pd
import os
import time

# ==============================================================================
# 配置
# ==============================================================================

TOTAL_STATIONS = 321

PRED_BASE = 'data/预测结果/单点尺度预测'
FIT_BASE = os.path.join(PRED_BASE, "拟合结果")

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

OUTPUT_DIR = os.path.join(PRED_BASE, "物候提取结果")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PHENO_COLS = ['SOS_20', 'SOS_50', 'SOS_80', 'EOS_80', 'EOS_50', 'EOS_20']


# ==============================================================================
# 物候提取核心（与区域版完全一致）
# ==============================================================================

def extract_phenology(vi_daily):
    """
    从365天日尺度NDPI曲线提取6个物候日期。
    返回: [SOS_20, SOS_50, SOS_80, EOS_80, EOS_50, EOS_20]（DOY，从1开始）
    """
    try:
        vi = np.array(vi_daily, dtype=np.float64)
        if np.isnan(vi).any() or len(vi) != 365:
            return np.full(6, np.nan)

        # baseline: 前30天和后30天均值
        baseline = np.mean(np.concatenate([vi[:30], vi[-30:]]))
        peak = np.max(vi)
        amplitude = peak - baseline

        if amplitude < 0.02:
            return np.full(6, np.nan)

        max_idx = np.argmax(vi)
        thresholds = [0.2, 0.5, 0.8]
        target_values = [baseline + amplitude * th for th in thresholds]

        # 春季：从左往右找第一个超过阈值的DOY
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

        # 秋季：从右往左找第一个超过阈值的DOY
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

    except:
        return np.full(6, np.nan)


# ==============================================================================
# 观测物候提取
# ==============================================================================

def extract_obs_phenology():
    output_path = os.path.join(OUTPUT_DIR, "Obs_2018_phenology.csv")
    if os.path.exists(output_path):
        print(f"[观测] 已存在，跳过")
        return

    obs_fit_dir = os.path.join(FIT_BASE, "Obs_2018")
    if not os.path.isdir(obs_fit_dir):
        print(f"[观测] 拟合目录不存在: {obs_fit_dir}")
        return

    print("[观测] 提取物候...")
    rows = []
    for sid in range(1, TOTAL_STATIONS + 1):
        fit_path = os.path.join(obs_fit_dir, f"Station_{sid}_fitted.csv")
        if not os.path.exists(fit_path):
            pheno = np.full(6, np.nan)
        else:
            daily = np.loadtxt(fit_path, delimiter=',')
            if daily.ndim == 2:
                daily = daily[0]  # 观测只有1行
            pheno = extract_phenology(daily)

        row = {'Station': sid}
        for j, col in enumerate(PHENO_COLS):
            row[col] = pheno[j]
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    success = sum(1 for r in rows if not np.isnan(r['SOS_20']))
    print(f"[观测] 完成: {success}/{TOTAL_STATIONS} 成功")


# ==============================================================================
# 预测物候提取
# ==============================================================================

def extract_model_phenology(model_folder):
    output_path = os.path.join(OUTPUT_DIR, f"{model_folder}_phenology.csv")
    if os.path.exists(output_path):
        print(f"  [跳过] 已存在: {os.path.basename(output_path)}")
        return

    fit_dir = os.path.join(FIT_BASE, model_folder)
    if not os.path.isdir(fit_dir):
        print(f"  [跳过] 拟合目录不存在: {model_folder}")
        return

    print(f"  提取: {model_folder}")
    start_time = time.time()

    rows = []
    for sid in range(1, TOTAL_STATIONS + 1):
        fit_path = os.path.join(fit_dir, f"Station_{sid}_fitted.csv")
        if not os.path.exists(fit_path):
            # 填充NaN行
            for obs_count in range(46):
                row = {'Station': sid, 'obs_count': obs_count}
                for col in PHENO_COLS:
                    row[col] = np.nan
                rows.append(row)
            continue

        # 读取46×365拟合矩阵
        daily_mat = np.loadtxt(fit_path, delimiter=',')
        if daily_mat.ndim == 1:
            daily_mat = daily_mat.reshape(1, -1)

        n_rows = daily_mat.shape[0]
        for obs_count in range(n_rows):
            pheno = extract_phenology(daily_mat[obs_count, :])
            row = {'Station': sid, 'obs_count': obs_count}
            for j, col in enumerate(PHENO_COLS):
                row[col] = pheno[j]
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

    total = len(rows)
    success = sum(1 for r in rows if not np.isnan(r.get('SOS_20', np.nan)))
    elapsed = time.time() - start_time
    print(f"    完成: {success}/{total} 成功, 耗时 {elapsed:.1f}s")


# ==============================================================================
# 主程序
# ==============================================================================

def main():
    print("=" * 60)
    print("单点尺度物候提取（动态阈值法）")
    print(f"输入: {FIT_BASE}")
    print(f"输出: {OUTPUT_DIR}")
    print("=" * 60)

    # 观测
    extract_obs_phenology()

    # 预测
    print(f"\n--- 预测物候提取 ({len(MODEL_FOLDERS)} 个配置) ---")
    for folder in MODEL_FOLDERS:
        extract_model_phenology(folder)

    print("\n" + "=" * 60)
    print("物候提取全部完成")


if __name__ == "__main__":
    main()