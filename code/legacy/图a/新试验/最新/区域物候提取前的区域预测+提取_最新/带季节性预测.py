"""
区域滚动起点预测：从不同观测截断点出发，递归预测到年底，输出完整年曲线。
四个模型（LSTM / MLP / CNN / Transformer） × 两种步长（12 / 46） × 带季节性
每个截断点 × 每个模型配置 输出一个CSV文件。

数据结构：Class | 138列时序(2016-2018, 每年46点) | latitude | longitude
历史窗口：2016-2017（92个点）
只处理 site3 和 site6。
"""

import os
import gc
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer, Dense, Dropout, LayerNormalization, MultiHeadAttention
import tensorflow.keras.backend as K
import multiprocessing
from multiprocessing import Pool


# ==============================================================================
# Transformer 自定义层（加载Transformer模型时必需）
# ==============================================================================

class PositionalEncoding(Layer):
    def __init__(self, d_model, max_len=5000, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.d_model = d_model
        self.max_len = max_len

    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model, "max_len": self.max_len})
        return config

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        pos = tf.range(0, seq_len, dtype=tf.float32)[:, tf.newaxis]
        i = tf.range(0, self.d_model, 2, dtype=tf.float32)
        angle_rates = 1 / tf.pow(10000.0, (i / self.d_model))
        angle_rads = pos * angle_rates
        sines = tf.sin(angle_rads)
        cosines = tf.cos(angle_rads)
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return inputs + pos_encoding[:, :seq_len, :]


class TransformerBlock(Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([Dense(dff, activation='relu'), Dense(d_model)])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def call(self, inputs, training=True):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "dropout_rate": self.dropout_rate
        })
        return config


CUSTOM_OBJECTS = {
    "PositionalEncoding": PositionalEncoding,
    "TransformerBlock": TransformerBlock,
}

# ==============================================================================
# 1. 全局配置
# ==============================================================================
TARGET_YEAR = 2018
BATCH_SIZE = 4096
POINTS_PER_YEAR = 46

# --- 截断点配置 ---
# obs_count = 2018年中前N个时间点作为已知观测，其余递归预测
# TODO: 确认精确值后修改
CUTOFF_POINTS = {
    "Feb": 8,    # 观测到2018/2/26，从2018/3/6开始预测
    "May": 19,   # 观测到2018/5/25，从2018/6/2开始预测
    "Aug": 31,   # 观测到2018/8/29，从2018/9/6开始预测
}

# --- 类别 → 模型ID 映射 ---
CLASS_TO_MODEL_MAP = {
    1: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5, 7: 5,
    8: 6, 9: 6, 10: 6
}

# --- 模型配置 ---
# file_pattern: 用于glob匹配，{model_id}会被替换为实际ID
# LSTM/MLP的训练年份因model_id不同而不同，所以用通配符 * 匹配
MODEL_BASE = 'data/区域数据/模型/实际调用的模型'

MODEL_CONFIGS = [
    # ========== CNN ==========
    {
        "model_name": "CNN",
        "time_step": 12,
        "has_seasonal": True,
        "model_dir": os.path.join(MODEL_BASE, "CNN_Random4_2011_2017_12"),
        "file_pattern": "model_CLASS_{model_id}_train_2011_to_2017_Pooling_CNN.keras",
    },
    {
        "model_name": "CNN",
        "time_step": 46,
        "has_seasonal": True,
        "model_dir": os.path.join(MODEL_BASE, "CNN_Random50_2011_2017_46"),
        "file_pattern": "model_CLASS_{model_id}_train_2011_to_2017_Pooling_CNN.keras",
    },
    # ========== LSTM ==========
    {
        "model_name": "LSTM",
        "time_step": 12,
        "has_seasonal": True,
        "model_dir": os.path.join(MODEL_BASE, "LSTM_Random16_WithSeasonality_2011_2017_12"),
        "file_pattern": "model_CLASS_{model_id}_train_*_to_*_LSTM_Pooling.keras",  # 年份不固定，用通配符
    },
    {
        "model_name": "LSTM",
        "time_step": 46,
        "has_seasonal": True,
        "model_dir": os.path.join(MODEL_BASE, "LSTM_Random16_WithSeasonality_2011_2017_46"),
        "file_pattern": "model_CLASS_{model_id}_train_*_to_*_LSTM_Pooling.keras",
    },
    # ========== MLP ==========
    {
        "model_name": "MLP",
        "time_step": 12,
        "has_seasonal": True,
        "model_dir": os.path.join(MODEL_BASE, "MLP_Random4_WithSeasonality_2011_2017_12"),
        "file_pattern": "model_CLASS_{model_id}_train_*_to_*_MLP_WithSeasonality.keras",
    },
    {
        "model_name": "MLP",
        "time_step": 46,
        "has_seasonal": True,
        "model_dir": os.path.join(MODEL_BASE, "MLP_Random50_WithSeasonality_46"),
        "file_pattern": "model_CLASS_{model_id}_train_*_to_*_MLP_WithSeasonality.keras",
    },
    # ========== Transformer ==========
    {
        "model_name": "Trans",
        "time_step": 12,
        "has_seasonal": True,
        "model_dir": os.path.join(MODEL_BASE, "Trans_Random10_2011_2017_12"),
        "file_pattern": "model_class_{model_id}_Trans_VAL_ES_2011_2017.keras",
    },
    {
        "model_name": "Trans",
        "time_step": 46,
        "has_seasonal": True,
        "model_dir": os.path.join(MODEL_BASE, "Trans_Random16_2011_2017_46"),
        "file_pattern": "model_class_{model_id}_Trans_VAL_ES_2011_2017.keras",
    },
]

# --- 站点文件 ---
SITE_FILES = [
    'data/区域数据1/site3/NDPI_Site3_SG_filtered.csv',
    'data/区域数据1/site6/NDPI_Site6_SG_filtered.csv',
]

# 输出根目录
OUTPUT_BASE = 'data/预测结果/区域滚动预测_物候'


# ==============================================================================
# 2. 数据解析
# ==============================================================================

def parse_site_csv(file_path):
    """
    解析区域CSV：Class | 138列时序(2016-2018) | latitude | longitude
    """
    df = pd.read_csv(file_path)

    class_col = df.iloc[:, 0].values.astype(int)
    lat = df['latitude'].values.astype(np.float32)
    lon = df['longitude'].values.astype(np.float32)

    # 中间138列是时序
    date_strs = df.columns[1:-2].tolist()
    dates = pd.to_datetime(date_strs, format='%Y/%m/%d')
    ndpi_raw = df.iloc[:, 1:-2].values.astype(np.float32)

    if np.isnan(ndpi_raw).any():
        ndpi_raw = np.nan_to_num(ndpi_raw, nan=-1.0)

    return class_col, lat, lon, date_strs, dates, ndpi_raw


def prepare_features(ndpi_raw, dates, has_seasonal):
    """
    构造特征矩阵: NDPI归一化 + 可选的sin/cos季节性编码
    """
    N, T = ndpi_raw.shape
    ndpi_norm = ((ndpi_raw + 1) / 2).astype(np.float32)

    if has_seasonal:
        doy = dates.dayofyear.values.astype(np.float32)
        sin_feat = np.sin(2 * np.pi * doy / 366.0)
        cos_feat = np.cos(2 * np.pi * doy / 366.0)
        sin_tiled = np.tile(sin_feat, (N, 1)).astype(np.float32)
        cos_tiled = np.tile(cos_feat, (N, 1)).astype(np.float32)
        X_full = np.stack([ndpi_norm, sin_tiled, cos_tiled], axis=-1)
        n_features = 3
    else:
        X_full = ndpi_norm[:, :, np.newaxis]
        n_features = 1

    return X_full, n_features


# ==============================================================================
# 3. 核心预测逻辑
# ==============================================================================

def rolling_predict_batch(model, history, test_features, obs_count,
                          n_points, time_step, n_features):
    """
    批量滚动预测：对一批像元，从obs_count处截断，递归预测到年底。

    history:       (N, 92, n_features) 2016-2017特征
    test_features: (N, 46, n_features) 2018特征
    obs_count:     已知观测点数（前obs_count个为真实值）

    返回: (N, 46) 完整年曲线，值域 [-1, 1]
    """
    N = history.shape[0]
    results = np.full((N, n_points), np.nan, dtype=np.float32)

    # 填充已知观测（反归一化）
    if obs_count > 0:
        results[:, :obs_count] = test_features[:, :obs_count, 0] * 2 - 1

    if obs_count >= n_points:
        return results

    # 构造初始输入窗口
    if obs_count == 0:
        window = history[:, -time_step:, :].copy()
    elif obs_count < time_step:
        n_from_hist = time_step - obs_count
        hist_part = history[:, -n_from_hist:, :]
        obs_part = test_features[:, :obs_count, :]
        window = np.concatenate([hist_part, obs_part], axis=1)
    else:
        window = test_features[:, obs_count - time_step:obs_count, :].copy()

    # 递归预测
    for i in range(n_points - obs_count):
        input_data = window[:, -time_step:, :]
        pred = model.predict(input_data, batch_size=BATCH_SIZE, verbose=0)

        results[:, obs_count + i] = pred[:, 0] * 2 - 1

        target_idx = obs_count + i
        if n_features > 1 and target_idx < n_points:
            next_row = test_features[:, target_idx, :].copy()
            next_row[:, 0] = pred[:, 0]
        else:
            next_row = pred

        next_row = next_row.reshape(N, 1, n_features)
        window = np.concatenate([window[:, 1:, :], next_row], axis=1)

    return results


# ==============================================================================
# 4. 站点处理
# ==============================================================================

def process_site(file_path):
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)

    site_name = os.path.basename(os.path.dirname(file_path))
    print(f"\n[PID {os.getpid()}] 正在处理: {site_name}")

    try:
        # --- 解析数据 ---
        class_col, lat, lon, date_strs, dates, ndpi_raw = parse_site_csv(file_path)
        N_pixels = len(class_col)

        years = dates.year.values
        mask_hist = (years == 2016) | (years == 2017)
        mask_2018 = years == 2018
        idx_hist = np.where(mask_hist)[0]
        idx_2018 = np.where(mask_2018)[0]
        n_points = len(idx_2018)
        col_names_2018 = [date_strs[i] for i in idx_2018]

        print(f"  像元数={N_pixels}, 历史={len(idx_hist)}点, 2018={n_points}点")
        print(f"  类别: {dict(zip(*np.unique(class_col, return_counts=True)))}")

        # --- 按类别分组 ---
        model_groups = {}
        for cls in np.unique(class_col):
            if cls in CLASS_TO_MODEL_MAP:
                mid = CLASS_TO_MODEL_MAP[cls]
                if mid not in model_groups:
                    model_groups[mid] = []
                model_groups[mid].append(cls)

        print(f"  模型分组: {model_groups}")

        # --- 逐模型配置 ---
        for cfg in MODEL_CONFIGS:
            model_name = cfg["model_name"]
            time_step = cfg["time_step"]
            has_seasonal = cfg["has_seasonal"]
            model_dir = cfg["model_dir"]
            file_pattern = cfg["file_pattern"]
            tag = f"{model_name}_L{time_step}_Seasonal"

            print(f"\n  --- {tag} ---")

            if not os.path.exists(model_dir):
                print(f"  [跳过] 目录不存在: {model_dir}")
                continue

            # 构造特征
            X_full, n_features = prepare_features(ndpi_raw, dates, has_seasonal)
            history = X_full[:, idx_hist, :]
            test_features = X_full[:, idx_2018, :]

            output_dir = os.path.join(OUTPUT_BASE, site_name, tag)
            os.makedirs(output_dir, exist_ok=True)

            cutoff_results = {
                label: np.full((N_pixels, n_points), np.nan, dtype=np.float32)
                for label in CUTOFF_POINTS
            }

            for model_id, class_list in model_groups.items():
                # 用glob匹配模型文件（LSTM/MLP训练年份不固定）
                pattern = os.path.join(model_dir, file_pattern.format(model_id=model_id))
                matches = glob.glob(pattern)

                if not matches:
                    print(f"    [跳过] 未匹配到模型: {pattern}")
                    continue

                model_path = matches[0]  # 取第一个匹配
                if len(matches) > 1:
                    print(f"    [警告] 匹配到多个文件，使用: {os.path.basename(model_path)}")

                if model_name == "Trans":
                    model = load_model(model_path, compile=False,
                                       custom_objects=CUSTOM_OBJECTS)
                else:
                    model = load_model(model_path, compile=False)

                pixel_mask = np.isin(class_col, class_list)
                pixel_indices = np.where(pixel_mask)[0]

                if len(pixel_indices) == 0:
                    del model; K.clear_session()
                    continue

                curr_history = history[pixel_indices]
                curr_test = test_features[pixel_indices]

                print(f"    模型ID={model_id}, 类别={class_list}, "
                      f"像元={len(pixel_indices)}")

                for label, obs_count in CUTOFF_POINTS.items():
                    pred = rolling_predict_batch(
                        model=model,
                        history=curr_history,
                        test_features=curr_test,
                        obs_count=obs_count,
                        n_points=n_points,
                        time_step=time_step,
                        n_features=n_features
                    )
                    cutoff_results[label][pixel_indices] = pred

                del model
                K.clear_session()
                gc.collect()

            # --- 保存 ---
            for label, obs_count in CUTOFF_POINTS.items():
                mat = cutoff_results[label]
                df_pred = pd.DataFrame(mat, columns=col_names_2018)
                df_out = pd.concat([
                    pd.Series(class_col, name='Class'),
                    df_pred,
                    pd.Series(lat, name='latitude'),
                    pd.Series(lon, name='longitude'),
                ], axis=1)

                fname = f"Rolling_{site_name}_{tag}_Cutoff{label}_obs{obs_count}.csv"
                fpath = os.path.join(output_dir, fname)
                df_out.to_csv(fpath, index=False, float_format='%.4f')
                print(f"    已保存: {fname}")

            del cutoff_results, X_full, history, test_features
            gc.collect()

        return f"[{site_name}] 完成"

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"[{site_name}] 异常: {e}"


# ==============================================================================
# 5. 主程序
# ==============================================================================
if __name__ == "__main__":
    multiprocessing.freeze_support()

    NUM_PROCESSES = min(len(SITE_FILES), os.cpu_count())

    print("区域滚动起点预测")
    print(f"截断点: {CUTOFF_POINTS}")
    print(f"模型配置: {len(MODEL_CONFIGS)}个")
    print(f"站点: {[os.path.basename(os.path.dirname(f)) for f in SITE_FILES]}")
    print("=" * 60)

    with Pool(processes=NUM_PROCESSES) as pool:
        results = pool.map(process_site, SITE_FILES)

    print("\n" + "=" * 60)
    for res in results:
        print(res)
    print("全部完成")