import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, Flatten, InputLayer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K
import os
import gc

# ==============================================================================
# 1. 配置参数
# ==============================================================================
# 文件列表 (保持原路径)
FILE_PATHS = [
    None,  # Class 1 (跳过)
    None,  # Class 2 (跳过)
    None,  # Class 3 (跳过)
    None,  # Class 4 (跳过)
    None,  # Class 5 (跳过)
    'data/区域数据/模型训练数据/Final_QC_NDPI_TimeSeries_class6_2009_2017-副本.csv'  # Class 6 (新数据)
]

# [修改] 模型保存路径 (MLP, 带季节性)
MODEL_SAVE_DIR = 'data/区域数据/模型/MLP_Random50_WithSeasonality_2011_2017_1'
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# 站点经纬度保存路径
SITE_COORD_DIR = 'data/区域数据/模型/Site_Locations_MLP_Seasonality_1'
os.makedirs(SITE_COORD_DIR, exist_ok=True)

# 训练设置
TRAIN_START_YEAR = 2013
TRAIN_END_YEAR = 2016
VAL_YEAR = 2017
SAMPLE_SIZE = 50  # 每个类别随机抽取的点数
TIME_STEP = 12  # 输入步长

# [核心修改] 类别参数 (MLP 12步长 带季节性特征 最优参数)
# 依据之前优化的结果替换
CLASS_PARAMS_LIST = [
    # Class 1: 2 layers
    {'n_layers': 2, 'units': [224, 192], 'dropouts': [0.2315, 0.2274],
     'batch_size': 32, 'learning_rate': 0.00078, 'max_epochs': 700},

    # Class 2: 2 layers
    {'n_layers': 2, 'units': [288, 288], 'dropouts': [0.2429, 0.3245],
     'batch_size': 16, 'learning_rate': 0.00025, 'max_epochs': 500},

    # Class 3: 2 layers
    {'n_layers': 2, 'units': [320, 160], 'dropouts': [0.2016, 0.3132],
     'batch_size': 16, 'learning_rate': 0.00011, 'max_epochs': 400},

    # Class 4: 2 layers
    {'n_layers': 2, 'units': [224, 160], 'dropouts': [0.2198, 0.2556],
     'batch_size': 32, 'learning_rate': 0.00041, 'max_epochs': 750},

    # Class 5: 2 layers
    {'n_layers': 2, 'units': [320, 256], 'dropouts': [0.2186, 0.2346],
     'batch_size': 16, 'learning_rate': 0.00108, 'max_epochs': 750},

    # Class 6: 2 layers
    {'n_layers': 2, 'units': [192, 320], 'dropouts': [0.2219, 0.2369],
     'batch_size': 16, 'learning_rate': 0.00035, 'max_epochs': 450}
]


# ==============================================================================
# 2. 核心函数
# ==============================================================================

def create_sliding_window(data_matrix, time_features, time_step):
    """构建滑动窗口数据集 (NDPI + 季节性特征)"""
    X, y = [], []
    L = len(data_matrix)
    for i in range(L - time_step):
        # 取 data_matrix 的一列 (NDPI)，reshape 为 (time_step, 1)
        ndpi_window = data_matrix[i: i + time_step].reshape(-1, 1)

        # [修改] 获取对应的时间特征窗口 (time_step, 2)
        time_window = time_features[i: i + time_step]

        # [修改] 拼接 NDPI 和 时间特征 -> (time_step, 3)
        feature_window = np.concatenate([ndpi_window, time_window], axis=1)

        X.append(feature_window)
        y.append(data_matrix[i + time_step])
    return np.array(X), np.array(y)


def save_site_locations(df_sampled, class_idx):
    """保存抽样站点的经纬度到 TXT"""
    coords = df_sampled.iloc[:, -2:].values

    filename = f"Selected_Sites_Class{class_idx}.txt"
    filepath = os.path.join(SITE_COORD_DIR, filename)

    with open(filepath, 'w') as f:
        f.write("Latitude,Longitude\n")
        for lat, lon in coords:
            f.write(f"{lat},{lon}\n")

    print(f"  📍 站点坐标已保存: {filepath}")


def process_class_file(file_path, class_idx, sample_size):
    """读取CSV，抽样，保存坐标，构建数据集"""
    print(f"正在读取: {os.path.basename(file_path)}")

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"  读取失败: {e}")
        return None, None, None, None

    # 解析日期 (排除最后两列经纬度)
    data_cols = df.columns[:-2]
    try:
        dates = pd.to_datetime(data_cols)
    except:
        dates = pd.to_datetime(data_cols, errors='coerce')

    if dates.isnull().any():
        print("  日期解析有误，请检查表头格式。")
        return None, None, None, None

    # --- 随机抽样 ---
    if len(df) > sample_size:
        df_sampled = df.sample(n=sample_size, random_state=42)
        print(f"  随机抽取 {sample_size} 个站点 (原始 {len(df)} 个)")
    else:
        df_sampled = df
        print(f"  样本不足 {sample_size}，使用全部 {len(df)} 个站点")

    # --- 保存经纬度 ---
    save_site_locations(df_sampled, class_idx)

    # [修改] 恢复季节性特征 (sin/cos)
    doy = dates.dayofyear.values
    sin_time = np.sin(2 * np.pi * doy / 366.0)
    cos_time = np.cos(2 * np.pi * doy / 366.0)
    # 堆叠成 (N, 2) 的矩阵
    time_features = np.stack([sin_time, cos_time], axis=1)

    # 划分时间索引
    train_mask = (dates.year >= TRAIN_START_YEAR) & (dates.year <= TRAIN_END_YEAR)
    val_mask = (dates.year == VAL_YEAR)

    # 提取数值并归一化
    ndpi_values = df_sampled.iloc[:, :-2].values
    ndpi_values = np.nan_to_num(ndpi_values, nan=0.0)
    ndpi_values = (ndpi_values + 1) / 2

    # 构建数据集
    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []

    print("  正在构建滑动窗口序列 (含季节性特征)...")
    for i in range(len(df_sampled)):
        site_data = ndpi_values[i, :]

        # 训练集
        train_data = site_data[train_mask]
        train_time = time_features[train_mask]

        if len(train_data) > TIME_STEP:
            # [修改] 传入 time_features
            X_t, y_t = create_sliding_window(train_data, train_time, TIME_STEP)
            X_train_list.append(X_t)
            y_train_list.append(y_t)

        # 验证集 (需拼接前序数据)
        val_start_idx = np.argmax(val_mask)
        idx_start = max(0, val_start_idx - TIME_STEP)
        idx_end = val_start_idx + np.sum(val_mask)

        val_data_extended = site_data[idx_start: idx_end]
        val_time_extended = time_features[idx_start: idx_end]

        if len(val_data_extended) > TIME_STEP:
            # [修改] 传入 time_features
            X_v, y_v = create_sliding_window(val_data_extended, val_time_extended, TIME_STEP)
            X_val_list.append(X_v)
            y_val_list.append(y_v)

    if X_train_list:
        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)
    else:
        X_train, y_train = None, None

    if X_val_list:
        X_val = np.concatenate(X_val_list, axis=0)
        y_val = np.concatenate(y_val_list, axis=0)
    else:
        X_val, y_val = None, None

    return X_train, y_train, X_val, y_val


# [核心] 构建 MLP 模型
def build_mlp_model(time_step, n_features, params):
    model = Sequential()

    # 输入层 + 展平
    model.add(InputLayer(input_shape=(time_step, n_features)))
    model.add(Flatten())

    # 动态添加隐藏层 (根据 params 中的 list)
    units_list = params['units']
    dropouts_list = params['dropouts']

    for i in range(params['n_layers']):
        model.add(Dense(units_list[i], activation='relu'))
        model.add(Dropout(dropouts_list[i]))

    # 输出层
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='mse', optimizer=Adam(learning_rate=params['learning_rate']))
    return model


# ==============================================================================
# 3. 主流程 (MLP 普通循环版)
# ==============================================================================
if __name__ == "__main__":
    print(f"开始 MLP 模型训练 (每类随机抽样 {SAMPLE_SIZE} 个点)...")
    print(f"训练: {TRAIN_START_YEAR}-{TRAIN_END_YEAR}, 验证: {VAL_YEAR}\n")

    for i, file_path in enumerate(FILE_PATHS):
        # 【关键修复】如果是 None，直接跳过
        if file_path is None:
            continue

        class_id = i + 1
        print(f"{'=' * 60}")
        print(f"处理类别 {class_id} ...")

        if not os.path.exists(file_path):
            print(f"❌ 文件不存在: {file_path}")
            continue

        X_train, y_train, X_val, y_val = process_class_file(file_path, class_id, SAMPLE_SIZE)

        if X_train is None:
            print(f"❌ 类别 {class_id} 数据处理失败，跳过。")
            continue

        print(f"  训练集形状: X={X_train.shape}, y={y_train.shape}")
        print(f"  验证集形状: X={X_val.shape}, y={y_val.shape}")

        params = CLASS_PARAMS_LIST[i]
        n_features = X_train.shape[2]  # 应该是 3 (NDPI, sin, cos)

        # 构建模型
        model = build_mlp_model(TIME_STEP, n_features, params)

        print("  开始训练...")
        early_stopper = EarlyStopping(
            monitor='val_loss',
            patience=50,
            min_delta=0.0005,
            restore_best_weights=True
        )

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=params['max_epochs'],
            batch_size=params['batch_size'],
            callbacks=[early_stopper],
            verbose=0
        )

        model_name = f'model_CLASS_{class_id}_train_{TRAIN_START_YEAR}_to_{TRAIN_END_YEAR}_MLP_WithSeasonality.keras'
        save_path = os.path.join(MODEL_SAVE_DIR, model_name)
        model.save(save_path)

        val_loss = min(history.history['val_loss'])
        print(f"✅ 类别 {class_id} 训练完成，最佳 Val Loss: {val_loss:.6f}")
        print(f"   模型已保存: {save_path}")

        del model, X_train, y_train, X_val, y_val
        K.clear_session()
        gc.collect()

    print(f"\n{'=' * 60}")
    print("🎉 所有类别处理完毕！")