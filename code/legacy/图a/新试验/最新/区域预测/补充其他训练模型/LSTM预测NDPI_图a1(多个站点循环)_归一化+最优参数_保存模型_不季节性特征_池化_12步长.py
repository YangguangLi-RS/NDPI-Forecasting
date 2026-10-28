import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, InputLayer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K
import os
import gc
# [新增] 导入并行库
from joblib import Parallel, delayed

# [建议] 并行CPU训练时，强制禁用GPU以避免多进程显存冲突
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# 设置 TensorFlow 线程数以优化 CPU 并行性能
os.environ['TF_NUM_INTER_OP_THREADS'] = '1'
os.environ['TF_NUM_INTRA_OP_THREADS'] = '1'

# ==============================================================================
# 1. 配置参数
# ==============================================================================
# 文件列表
FILE_PATHS = [
    'data/区域数据/模型训练数据/Final_QC_NDPI_TimeSeries_class1_2009_2017.csv',
    'data/区域数据/模型训练数据/Final_QC_NDPI_TimeSeries_class2_2009_2017.csv',
    'data/区域数据/模型训练数据/Final_QC_NDPI_TimeSeries_class3_2009_2017.csv',
    'data/区域数据/模型训练数据/Final_QC_NDPI_TimeSeries_class4_2009_2017.csv',
    'data/区域数据/模型训练数据/Final_QC_NDPI_TimeSeries_class5_2009_2017.csv',
    'data/区域数据/模型训练数据/Final_QC_NDPI_TimeSeries_class6_2009_2017-副本.csv'  # Class 6 (新数据)
]

# [修改] 模型保存路径 (LSTM, 池化, 无季节性)
MODEL_SAVE_DIR = 'data/区域数据/模型/LSTM_Random50_NoSeasonality_2011_2017_12'
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# 站点经纬度保存路径
SITE_COORD_DIR = 'data/区域数据/模型/Site_Locations_LSTM_NoSeasonality_12'
os.makedirs(SITE_COORD_DIR, exist_ok=True)

# 训练设置
TRAIN_START_YEAR = 2010  # [修改] 根据日志文件显示，训练集为 2010-2016
TRAIN_END_YEAR = 2016
VAL_YEAR = 2017
SAMPLE_SIZE = 50  # 每个类别随机抽取的点数
TIME_STEP = 12  # [修正] 参数文件日志明确显示"输入窗口: 12"

# [核心] 类别参数 (使用你提供的 LSTM 无季节性 最优参数)
CLASS_PARAMS_LIST = [
    # 类别 1
    {
        'neurons_1': 180, 'dropout_1': 0.2158, 'neurons_2': 100, 'dropout_2': 0.2068,
        'batch_size': 16, 'learning_rate': 0.00453, 'max_epochs': 775
    },
    # 类别 2
    {
        'neurons_1': 140, 'dropout_1': 0.2054, 'neurons_2': 160, 'dropout_2': 0.2533,
        'batch_size': 32, 'learning_rate': 0.00452, 'max_epochs': 450
    },
    # 类别 3
    {
        'neurons_1': 150, 'dropout_1': 0.2534, 'neurons_2': 100, 'dropout_2': 0.2856,
        'batch_size': 16, 'learning_rate': 0.00441, 'max_epochs': 550
    },
    # 类别 4
    {
        'neurons_1': 120, 'dropout_1': 0.2363, 'neurons_2': 190, 'dropout_2': 0.2319,
        'batch_size': 16, 'learning_rate': 0.00466, 'max_epochs': 400
    },
    # 类别 5
    {
        'neurons_1': 120, 'dropout_1': 0.2632, 'neurons_2': 130, 'dropout_2': 0.2110,
        'batch_size': 16, 'learning_rate': 0.00438, 'max_epochs': 650
    },
    # 类别 6
    {
        'neurons_1': 100, 'dropout_1': 0.2172, 'neurons_2': 190, 'dropout_2': 0.2511,
        'batch_size': 16, 'learning_rate': 0.00485, 'max_epochs': 775
    }
]


# ==============================================================================
# 2. 核心函数
# ==============================================================================

def create_sliding_window(data_matrix, time_step):
    """
    构建滑动窗口数据集 (LSTM 格式, 仅 NDPI)
    X shape: (Samples, time_step, 1)
    """
    X, y = [], []
    L = len(data_matrix)
    for i in range(L - time_step):
        # 1. NDPI 数据 (time_step, 1)
        ndpi_window = data_matrix[i: i + time_step].reshape(-1, 1)

        # [修改] 不再拼接时间特征
        feature_window = ndpi_window

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
    print(f"[Class {class_idx}] 正在读取: {os.path.basename(file_path)}")

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"  [Class {class_idx}] 读取失败: {e}")
        return None, None, None, None

    # 解析日期 (排除最后两列经纬度)
    data_cols = df.columns[:-2]
    try:
        dates = pd.to_datetime(data_cols)
    except:
        dates = pd.to_datetime(data_cols, errors='coerce')

    if dates.isnull().any():
        print(f"  [Class {class_idx}] 日期解析有误，请检查表头格式。")
        return None, None, None, None

    # --- 随机抽样 ---
    if len(df) > sample_size:
        df_sampled = df.sample(n=sample_size, random_state=42)
        print(f"  [Class {class_idx}] 随机抽取 {sample_size} 个站点 (原始 {len(df)} 个)")
    else:
        df_sampled = df
        print(f"  [Class {class_idx}] 样本不足 {sample_size}，使用全部 {len(df)} 个站点")

    # --- 保存经纬度 ---
    save_site_locations(df_sampled, class_idx)

    # [修改] 不生成季节性特征 (sin/cos)

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

    # print(f"  [Class {class_idx}] 正在构建滑动窗口序列...")
    for i in range(len(df_sampled)):
        site_data = ndpi_values[i, :]  # 一维数组

        # 训练集
        train_data = site_data[train_mask]

        if len(train_data) > TIME_STEP:
            # [修改] 不传入 time_features
            X_t, y_t = create_sliding_window(train_data, TIME_STEP)
            X_train_list.append(X_t)
            y_train_list.append(y_t)

        # 验证集 (需拼接前序数据)
        val_start_idx = np.argmax(val_mask)
        idx_start = max(0, val_start_idx - TIME_STEP)
        idx_end = val_start_idx + np.sum(val_mask)

        val_data_extended = site_data[idx_start: idx_end]

        if len(val_data_extended) > TIME_STEP:
            # [修改] 不传入 time_features
            X_v, y_v = create_sliding_window(val_data_extended, TIME_STEP)
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


# [核心] 构建 LSTM 模型
def build_lstm_model(time_step, n_features, params):
    model = Sequential([
        # 使用 InputLayer 更加规范
        InputLayer(input_shape=(time_step, n_features)),

        LSTM(params['neurons_1'], return_sequences=True),
        Dropout(params['dropout_1']),

        LSTM(params['neurons_2']),
        Dropout(params['dropout_2']),

        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='mse', optimizer=Adam(learning_rate=params['learning_rate']))
    return model


# [新增] 独立的任务函数，用于并行执行
def run_single_class_task(i, file_path):
    class_id = i + 1
    print(f"\n[{os.getpid()}] >>> 开始处理类别 {class_id} ...")

    if not os.path.exists(file_path):
        print(f"❌ [Class {class_id}] 文件不存在: {file_path}")
        return

    # 1. 准备数据
    X_train, y_train, X_val, y_val = process_class_file(file_path, class_id, SAMPLE_SIZE)

    if X_train is None:
        print(f"❌ [Class {class_id}] 数据处理失败，跳过。")
        return

    print(f"  [Class {class_id}] 训练集: {X_train.shape}, 验证集: {X_val.shape}")

    params = CLASS_PARAMS_LIST[i]
    n_features = X_train.shape[2]  # 应该是 1

    # 构建模型
    model = build_lstm_model(TIME_STEP, n_features, params)

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
        verbose=0  # 避免多进程打印混乱
    )

    model_name = f'model_CLASS_{class_id}_train_{TRAIN_START_YEAR}_to_{TRAIN_END_YEAR}_LSTM_Pooling_NoSeasonality.keras'
    save_path = os.path.join(MODEL_SAVE_DIR, model_name)
    model.save(save_path)

    val_loss = min(history.history['val_loss'])
    print(f"✅ [Class {class_id}] 训练完成，最佳 Val Loss: {val_loss:.6f}")
    print(f"   [Class {class_id}] 模型已保存: {save_path}")

    # 显式清理
    del model, X_train, y_train, X_val, y_val
    K.clear_session()
    gc.collect()


# ==============================================================================
# 3. 主流程
# ==============================================================================
if __name__ == "__main__":
    print(f"开始 LSTM 模型池化训练 (并行6个任务, 无季节性特征)...")
    print(f"每类随机抽样: {SAMPLE_SIZE}")
    print(f"训练周期: {TRAIN_START_YEAR}-{TRAIN_END_YEAR}, 验证年份: {VAL_YEAR}\n")

    # 使用 joblib 并行执行 6 个类别的训练任务
    # n_jobs=6 表示同时运行 6 个进程
    Parallel(n_jobs=6)(
        delayed(run_single_class_task)(i, file_path)
        for i, file_path in enumerate(FILE_PATHS)
    )

    print(f"\n{'=' * 60}")
    print("🎉 所有并行任务已分发完毕！")