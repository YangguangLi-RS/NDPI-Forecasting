"""
动态加权融合脚本（纯计算，不调用模型）
读取已有的 CNN_L46_Seasonal 和 Trans_L46_Seasonal 预测CSV，
对预测部分做逐步动态加权融合，输出加权和等权结果。

修改：gamma=0.1，等权融合第一步全给Transformer。
"""

import os
import numpy as np
import pandas as pd

# ==============================================================================
# 配置
# ==============================================================================

ROOT = 'data/预测结果/区域滚动预测_物候'

CUTOFF_POINTS = {
    "Feb": 8,
    "May": 19,
    "Aug": 31,
}

# 动态加权参数
KC_PARAM = 3
GAMMA_PARAM = 0.1   # ← 改为0.1
DELTA_PARAM = 1

SITES = ["site3", "site6"]


# ==============================================================================
# 动态加权函数
# ==============================================================================

def get_dynamic_weight_curve(n_steps, k_c=KC_PARAM, gamma=GAMMA_PARAM, delta=DELTA_PARAM):
    if delta >= k_c:
        delta = k_c - 0.1
    steps = np.arange(1, n_steps + 1, dtype=np.float64)
    numerator = np.maximum(0, steps - delta)
    denominator = k_c - delta
    term = (numerator / denominator) ** gamma
    w_trans = 1.0 / (1.0 + term)
    return w_trans.astype(np.float32)


def get_equal_weight_curve(n_steps):
    """等权融合：第1步全给T（w=1.0），其余步0.5"""
    w = np.full(n_steps, 0.5, dtype=np.float32)
    w[0] = 1.0
    return w


# ==============================================================================
# 主逻辑
# ==============================================================================

def fuse_single_cutoff(cnn_path, trans_path, obs_count, output_dir, site_name, label):
    df_cnn = pd.read_csv(cnn_path)
    df_trans = pd.read_csv(trans_path)

    class_col = df_cnn.iloc[:, 0]
    lat_col = df_cnn.iloc[:, -2]
    lon_col = df_cnn.iloc[:, -1]

    cnn_ndpi = df_cnn.iloc[:, 1:47].values.astype(np.float32)
    trans_ndpi = df_trans.iloc[:, 1:47].values.astype(np.float32)
    col_names = df_cnn.columns[1:47].tolist()

    n_points = cnn_ndpi.shape[1]
    pred_steps = n_points - obs_count

    obs_part = cnn_ndpi[:, :obs_count]
    cnn_pred = cnn_ndpi[:, obs_count:]
    trans_pred = trans_ndpi[:, obs_count:]

    # 动态加权 (gamma=0.1)
    w_dyn = get_dynamic_weight_curve(pred_steps)
    dynamic_pred = w_dyn[np.newaxis, :] * trans_pred + (1 - w_dyn[np.newaxis, :]) * cnn_pred

    # 等权（第1步全给T，其余0.5）
    w_eq = get_equal_weight_curve(pred_steps)
    equal_pred = w_eq[np.newaxis, :] * trans_pred + (1 - w_eq[np.newaxis, :]) * cnn_pred

    dynamic_full = np.hstack([obs_part, dynamic_pred])
    equal_full = np.hstack([obs_part, equal_pred])

    base = f"Rolling_{site_name}_Cutoff{label}_obs{obs_count}"

    for mat, suffix in [(dynamic_full, "DynamicWeight_g01"), (equal_full, "EqualWeight_T1st")]:
        df_pred = pd.DataFrame(mat, columns=col_names)
        df_out = pd.concat([
            class_col.reset_index(drop=True),
            df_pred,
            lat_col.reset_index(drop=True),
            lon_col.reset_index(drop=True),
        ], axis=1)
        fname = f"{base}_{suffix}.csv"
        fpath = os.path.join(output_dir, fname)
        df_out.to_csv(fpath, index=False, float_format='%.4f')
        print(f"  已保存: {fname}")


def main():
    print("动态加权融合 (CNN_L46 + Trans_L46)")
    print(f"动态加权参数: Kc={KC_PARAM}, gamma={GAMMA_PARAM}, delta={DELTA_PARAM}")
    print(f"等权策略: 第1步 w_T=1.0, 其余步 w_T=0.5")

    w = get_dynamic_weight_curve(38)
    print(f"动态权重示例 (38步): step1={w[0]:.3f}, step3={w[2]:.3f}, "
          f"step10={w[9]:.3f}, step38={w[37]:.3f}")

    w_eq = get_equal_weight_curve(38)
    print(f"等权示例 (38步): step1={w_eq[0]:.3f}, step2={w_eq[1]:.3f}, step38={w_eq[37]:.3f}")
    print("=" * 60)

    for site in SITES:
        print(f"\n--- {site} ---")

        cnn_dir = os.path.join(ROOT, site, "CNN_L46_Seasonal")
        trans_dir = os.path.join(ROOT, site, "Trans_L46_Seasonal")
        output_dir = os.path.join(ROOT, site, "DynamicWeight_T46_CNN46_g01")
        os.makedirs(output_dir, exist_ok=True)

        for label, obs_count in CUTOFF_POINTS.items():
            cnn_fname = f"Rolling_{site}_CNN_L46_Seasonal_Cutoff{label}_obs{obs_count}.csv"
            trans_fname = f"Rolling_{site}_Trans_L46_Seasonal_Cutoff{label}_obs{obs_count}.csv"

            cnn_path = os.path.join(cnn_dir, cnn_fname)
            trans_path = os.path.join(trans_dir, trans_fname)

            if not os.path.exists(cnn_path):
                print(f"  [跳过] CNN文件不存在: {cnn_fname}")
                continue
            if not os.path.exists(trans_path):
                print(f"  [跳过] Trans文件不存在: {trans_fname}")
                continue

            print(f"  Cutoff{label} (obs={obs_count}, 递归{46-obs_count}步)")
            fuse_single_cutoff(cnn_path, trans_path, obs_count, output_dir, site, label)

    print("\n" + "=" * 60)
    print("全部完成")


if __name__ == "__main__":
    main()