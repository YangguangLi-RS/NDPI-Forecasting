import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K
import os
import gc

# ==============================================================================
# 1. 配置参数
# ==============================================================================
# 文件列表 (请确保路径无误)
FILE_PATHS = [
    'data/区域数据/模型训练数据/Final_QC_NDPI_TimeSeries_class1_2009_2017.csv',
    'data/区域数据/模型训练数据/Final_QC_NDPI_TimeSeries_class2_2009_2017.csv',
    'data/区域数据/模型训练数据/Final_QC_NDPI_TimeSeries_class3_2009_2017.csv',
    'data/区域数据/模型训练数据/Final_QC_NDPI_TimeSeries_class4_2009_2017.csv',
    'data/区域数据/模型训练数据/Final_QC_NDPI_TimeSeries_class5_2009_2017.csv',
    'data/区域数据/模型训练数据/Final_QC_NDPI_TimeSeries_class6_2009_2017-副本.csv'
    # Class 6 (新数据)
]

# 模型保存路径
MODEL_SAVE_DIR = 'data/区域数据/模型/CNN_Random50_2011_2017_46'
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# 站点经纬度保存路径
SITE_COORD_DIR = 'data/区域数据/模型/Site_Locations_46'
os.makedirs(SITE_COORD_DIR, exist_ok=True)

# 训练设置
TRAIN_START_YEAR = 2011
TRAIN_END_YEAR = 2016
VAL_YEAR = 2017
SAMPLE_SIZE = 50  # 每个类别随机抽取的点数
TIME_STEP = 46  # 输入步长

# 类别参数 (CNN) - 已根据上传的 param_search 文件更新
CLASS_PARAMS_LIST = [
    # 类别 1: 3层, Dropout 0.2084, BS 16, LR 0.00203, Epochs 577
    {'n_layers': 3, 'filters': [128, 128, 128], 'kernels': [5, 7, 3], 'dropout_rate': 0.2084, 'batch_size': 16,
     'learning_rate': 0.00203, 'max_epochs': 577},

    # 类别 2: 3层, Dropout 0.3238, BS 16, LR 0.00405, Epochs 585
    {'n_layers': 3, 'filters': [64, 32, 128], 'kernels': [3, 7, 7], 'dropout_rate': 0.3238, 'batch_size': 16,
     'learning_rate': 0.00405, 'max_epochs': 585},

    # 类别 3: 4层, Dropout 0.2166, BS 16, LR 0.00474, Epochs 738
    {'n_layers': 4, 'filters': [128, 32, 64, 64], 'kernels': [3, 7, 5, 3], 'dropout_rate': 0.2166, 'batch_size': 16,
     'learning_rate': 0.00474, 'max_epochs': 738},

    # 类别 4: 4层, Dropout 0.2657, BS 16, LR 0.00102, Epochs 594
    {'n_layers': 4, 'filters': [128, 128, 128, 128], 'kernels': [5, 3, 7, 5], 'dropout_rate': 0.2657, 'batch_size': 16,
     'learning_rate': 0.00102, 'max_epochs': 594},

    # 类别 5: 2层, Dropout 0.2582, BS 16, LR 0.00195, Epochs 524
    {'n_layers': 2, 'filters': [128, 32], 'kernels': [3, 5], 'dropout_rate': 0.2582, 'batch_size': 16,
     'learning_rate': 0.00195, 'max_epochs': 524},

    # 类别 6: 4层, Dropout 0.2061, BS 16, LR 0.00362, Epochs 504
    {'n_layers': 4, 'filters': [32, 128, 32, 128], 'kernels': [5, 3, 3, 5], 'dropout_rate': 0.2061, 'batch_size': 16,
     'learning_rate': 0.00362, 'max_epochs': 504}
]


# ==============================================================================
# 2. 核心函数
# ==============================================================================

def create_sliding_window(data_matrix, time_features, time_step):
    """构建滑动窗口数据集"""
    X, y = [], []
    L = len(data_matrix)
    for i in range(L - time_step):
        ndpi_window = data_matrix[i: i + time_step].reshape(-1, 1)
        time_window = time_features[i: i + time_step]
        feature_window = np.concatenate([ndpi_window, time_window], axis=1)
        X.append(feature_window)
        y.append(data_matrix[i + time_step])
    return np.array(X), np.array(y)


def save_site_locations(df_sampled, class_idx):
    """保存抽样站点的经纬度到 TXT"""
    # 假设倒数第二列是 Latitude, 倒数第一列是 Longitude
    coords = df_sampled.iloc[:, -2:].values

    filename = f"Selected_Sites_Class{class_idx}_CNN.txt"
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

    # 准备时间特征
    doy = dates.dayofyear.values
    sin_time = np.sin(2 * np.pi * doy / 366.0)
    cos_time = np.cos(2 * np.pi * doy / 366.0)
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

    print("  正在构建滑动窗口序列...")
    for i in range(len(df_sampled)):
        site_data = ndpi_values[i, :]

        # 训练集
        train_data = site_data[train_mask]
        train_time = time_features[train_mask]
        if len(train_data) > TIME_STEP:
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


def build_cnn_model(time_step, n_features, params):
    inputs = Input(shape=(time_step, n_features))
    x = inputs
    filters_list = params['filters']
    kernel_sizes = params['kernels']

    # 动态构建卷积层
    for i in range(len(filters_list)):
        x = Conv1D(
            filters=filters_list[i],
            kernel_size=kernel_sizes[i],
            activation='relu',
            padding='causal'
        )(x)

    x = Flatten()(x)
    x = Dropout(params['dropout_rate'])(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='mse', optimizer=Adam(learning_rate=params['learning_rate']))
    return model


# ==============================================================================
# 3. 主流程
# ==============================================================================
if __name__ == "__main__":
    print(f"开始 CNN 模型训练 (每类随机抽样 {SAMPLE_SIZE} 个点)...")
    print(f"训练: {TRAIN_START_YEAR}-{TRAIN_END_YEAR}, 验证: {VAL_YEAR}\n")

    for i, file_path in enumerate(FILE_PATHS):
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
        n_features = X_train.shape[2]

        model = build_cnn_model(TIME_STEP, n_features, params)

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
            verbose=0  # 不显示详细进度条以保持输出整洁
        )

        model_name = f'model_CLASS_{class_id}_train_2011_to_2017_Pooling_CNN.keras'
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