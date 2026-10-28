"""
加载已训练的MLP模型，对321个站点执行滚动预测（站点级并行）。
四种模型配置：
  1. MLP12-带季节性特征 (start_2013)
  2. MLP46-带季节性特征 (start_2012)
  3. MLP12-无季节性特征 (start_2011)
  4. MLP46-无季节性特征 (start_2010)

输出：每个站点一个 N×N 的预测矩阵CSV (N=每年时间点数)
  行 = obs_count (已知观测数)
  列 = 目标年各时间点的值（观测+预测拼接，已反归一化）
"""

import os
import gc
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from joblib import Parallel, delayed

# ========================= 配置参数 =========================
TOTAL_STATIONS = 321
TARGET_YEAR = 2018
N_JOBS = 8

DATA_PATH = 'data/站点训练数据_CVmax_2.csv'

# 模型根目录
MODEL_BASE = r"/root/autodl-tmp"

# 输出根目录
OUTPUT_BASE = r"/root/autodl-tmp/预测结果"

# 四种MLP模型配置
CONFIGS = [
    {
        "name": "MLP12_带季节性",
        "time_step": 12,
        "has_seasonal": True,
        "train_start": 2013,
        "model_dir": os.path.join(MODEL_BASE, "saved_models_12步长_带季节性特征_MLP_6class"),
        "output_dir": os.path.join(OUTPUT_BASE, "MLP12_带季节性_滚动预测"),
    },
    {
        "name": "MLP46_带季节性",
        "time_step": 46,
        "has_seasonal": True,
        "train_start": 2012,
        "model_dir": os.path.join(MODEL_BASE, "saved_models_46步长_带季节性特征_MLP_6class"),
        "output_dir": os.path.join(OUTPUT_BASE, "MLP46_带季节性_滚动预测"),
    },
    {
        "name": "MLP12_无季节性",
        "time_step": 12,
        "has_seasonal": False,
        "train_start": 2011,
        "model_dir": os.path.join(MODEL_BASE, "saved_models_12步长_无季节性特征_MLP_6class"),
        "output_dir": os.path.join(OUTPUT_BASE, "MLP12_无季节性_滚动预测"),
    },
    {
        "name": "MLP46_无季节性",
        "time_step": 46,
        "has_seasonal": False,
        "train_start": 2010,
        "model_dir": os.path.join(MODEL_BASE, "saved_models_46步长_无季节性特征_MLP_6class"),
        "output_dir": os.path.join(OUTPUT_BASE, "MLP46_无季节性_滚动预测"),
    },
]


def rolling_predict(model, train_data, test_data, time_step, n_features, n_points):
    """滚动预测，生成 n_points × n_points 的结果矩阵。"""
    results = np.zeros((n_points, n_points))

    for obs_count in range(n_points):
        if obs_count == 0:
            history = train_data[-time_step:, :].copy()
        elif obs_count < time_step:
            history = np.vstack([
                train_data[-(time_step - obs_count):, :],
                test_data[:obs_count, :]
            ])
        else:
            history = test_data[obs_count - time_step:obs_count, :].copy()

        results[obs_count, :obs_count] = (test_data[:obs_count, 0] * 2) - 1

        temp_history = history.copy()
        predictions = []

        for i in range(n_points - obs_count):
            input_data = temp_history[-time_step:, :].reshape(1, time_step, n_features)
            pred = model.predict(input_data, verbose=0)[0, 0]
            predictions.append(pred)

            current_idx = obs_count + i
            if n_features > 1:
                next_features = test_data[current_idx, 1:]
                new_row = np.concatenate(([pred], next_features)).reshape(1, n_features)
            else:
                new_row = np.array([[pred]])

            temp_history = np.vstack([temp_history, new_row])

        results[obs_count, obs_count:] = (np.array(predictions) * 2) - 1

    return results


def process_station(sid, train_data, test_data, time_step, n_features,
                    model_path, output_path):
    """处理单个站点：加载模型 → 滚动预测 → 保存CSV → 释放资源"""
    model = None
    try:
        if not os.path.exists(model_path):
            print(f"  [跳过] 站点 {sid}: 模型文件不存在")
            return

        model = load_model(model_path)
        n_points = len(test_data)

        results = rolling_predict(
            model, train_data, test_data,
            time_step, n_features, n_points
        )

        pd.DataFrame(results).to_csv(output_path, header=False, index=False)
        print(f"  站点 {sid}/{TOTAL_STATIONS} 完成 ({n_points}×{n_points})")

    except Exception as e:
        print(f"  [错误] 站点 {sid}: {e}")
    finally:
        if model:
            del model
        K.clear_session()
        gc.collect()


def main():
    print("加载数据...")
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"], index_col=[0])

    df_features = pd.DataFrame(index=df.index)
    df_features['day_of_year_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 366.0)
    df_features['day_of_year_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 366.0)

    for sid in range(1, TOTAL_STATIONS + 1):
        col = f'NDPI_{sid}'
        df_features[col] = (df[col] + 1) / 2

    for cfg in CONFIGS:
        print(f"\n{'=' * 60}")
        print(f"配置: {cfg['name']}  |  步长={cfg['time_step']}  |  训练起始={cfg['train_start']}")
        print(f"模型目录: {cfg['model_dir']}")
        os.makedirs(cfg['output_dir'], exist_ok=True)

        time_step = cfg['time_step']

        tasks = []
        for sid in range(1, TOTAL_STATIONS + 1):
            station_name = f'NDPI_{sid}'

            if cfg['has_seasonal']:
                feature_cols = [station_name, 'day_of_year_sin', 'day_of_year_cos']
            else:
                feature_cols = [station_name]

            n_features = len(feature_cols)

            train_data = df_features[
                (df_features.index.year >= cfg['train_start']) &
                (df_features.index.year < TARGET_YEAR)
            ][feature_cols].values

            test_data = df_features[
                df_features.index.year == TARGET_YEAR
            ][feature_cols].values

            # MLP模型文件命名: model_site_{sid}_start_{year}.keras
            model_filename = f"model_site_{sid}_start_{cfg['train_start']}.keras"
            model_path = os.path.join(cfg['model_dir'], model_filename)
            output_path = os.path.join(cfg['output_dir'], f"Station_{sid}_Predictions.csv")

            tasks.append((sid, train_data, test_data, time_step,
                          n_features, model_path, output_path))

        Parallel(n_jobs=N_JOBS, backend="loky", max_nbytes="50M")(
            delayed(process_station)(*task) for task in tasks
        )

        print(f"配置 [{cfg['name']}] 全部完成")

    print(f"\n{'=' * 60}")
    print("全部处理完成！")


if __name__ == "__main__":
    main()