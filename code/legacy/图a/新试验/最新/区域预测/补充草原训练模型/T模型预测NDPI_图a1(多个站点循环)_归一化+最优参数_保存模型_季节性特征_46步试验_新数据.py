import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, MultiHeadAttention, LayerNormalization, Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K
import os
import gc

# ==============================================================================
# 1. 配置参数
# ==============================================================================
# 文件列表
# 修改说明：前5类设为 None 跳过，只保留第6类的新路径
FILE_PATHS = [
    None,  # Class 1 (跳过)
    None,  # Class 2 (跳过)
    None,  # Class 3 (跳过)
    None,  # Class 4 (跳过)
    None,  # Class 5 (跳过)
    'data/区域数据/模型训练数据/Final_QC_NDPI_TimeSeries_class6_2009_2017-副本.csv'  # Class 6 (新数据)
]

# 模型保存路径 (增加了 _1)
MODEL_SAVE_DIR = 'data/区域数据/模型/Trans_Random50_2011_2017_1'
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# 站点经纬度保存路径 (增加了 _1)
SITE_COORD_DIR = 'data/区域数据/模型/Site_Locations_Trans_1'
os.makedirs(SITE_COORD_DIR, exist_ok=True)

# 训练设置
TRAIN_START_YEAR = 2011
TRAIN_END_YEAR = 2016
VAL_YEAR = 2017
SAMPLE_SIZE = 50
TIME_STEP = 46  # 保持不变

# 类别参数 (Transformer) - 保持原样，代码会自动使用第6个参数
CLASS_PARAMS_LIST = [
    # 类别 1
    {'d_model': 64, 'trans_layers': 2, 'trans_heads': 4, 'trans_ffn': 128, 'dropout_rate': 0.2317, 'batch_size': 16,
     'learning_rate': 0.00184, 'max_epochs': 566},
    # 类别 2
    {'d_model': 16, 'trans_layers': 2, 'trans_heads': 4, 'trans_ffn': 128, 'dropout_rate': 0.2138, 'batch_size': 64,
     'learning_rate': 0.00175, 'max_epochs': 515},
    # 类别 3
    {'d_model': 32, 'trans_layers': 2, 'trans_heads': 2, 'trans_ffn': 256, 'dropout_rate': 0.2077, 'batch_size': 16,
     'learning_rate': 0.00181, 'max_epochs': 578},
    # 类别 4
    {'d_model': 32, 'trans_layers': 1, 'trans_heads': 4, 'trans_ffn': 256, 'dropout_rate': 0.2543, 'batch_size': 32,
     'learning_rate': 0.00274, 'max_epochs': 651},
    # 类别 5
    {'d_model': 16, 'trans_layers': 1, 'trans_heads': 8, 'trans_ffn': 128, 'dropout_rate': 0.2201, 'batch_size': 16,
     'learning_rate': 0.00469, 'max_epochs': 403},
    # 类别 6 (将使用此参数)
    {'d_model': 16, 'trans_layers': 1, 'trans_heads': 8, 'trans_ffn': 256, 'dropout_rate': 0.3370, 'batch_size': 16,
     'learning_rate': 0.00496, 'max_epochs': 612}
]


# ==============================================================================
# 2. 自定义层 (Positional Encoding)
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


# ==============================================================================
# 3. 核心函数
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
    coords = df_sampled.iloc[:, -2:].values
    filename = f"Selected_Sites_Class{class_idx}_Trans.txt"
    filepath = os.path.join(SITE_COORD_DIR, filename)
    with open(filepath, 'w') as f:
        f.write("Latitude,Longitude\n")
        for lat, lon in coords:
            f.write(f"{lat},{lon}\n")
    print(f"  📍 站点坐标已保存: {filepath}")


def process_class_file(file_path, class_idx, sample_size):
    print(f"正在读取: {os.path.basename(file_path)}")
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"  读取失败: {e}")
        return None, None, None, None

    # 解析日期
    data_cols = df.columns[:-2]
    try:
        dates = pd.to_datetime(data_cols)
    except:
        dates = pd.to_datetime(data_cols, errors='coerce')

    if dates.isnull().any():
        return None, None, None, None

    # 随机抽样
    if len(df) > sample_size:
        df_sampled = df.sample(n=sample_size, random_state=42)
        print(f"  随机抽取 {sample_size} 个站点 (原始 {len(df)} 个)")
    else:
        df_sampled = df
        print(f"  样本不足 {sample_size}，使用全部 {len(df)} 个站点")

    # 保存坐标
    save_site_locations(df_sampled, class_idx)

    # 准备时间特征
    doy = dates.dayofyear.values
    sin_time = np.sin(2 * np.pi * doy / 366.0)
    cos_time = np.cos(2 * np.pi * doy / 366.0)
    time_features = np.stack([sin_time, cos_time], axis=1)

    # 划分时间
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

        # 验证集
        val_start_idx = np.argmax(val_mask)
        idx_start = max(0, val_start_idx - TIME_STEP)
        idx_end = val_start_idx + np.sum(val_mask)

        val_data_extended = site_data[idx_start: idx_end]
        val_time_extended = time_features[idx_start: idx_end]

        if len(val_data_extended) > TIME_STEP:
            X_v, y_v = create_sliding_window(val_data_extended, val_time_extended, TIME_STEP)
            X_val_list.append(X_v)
            y_val_list.append(y_v)

    if not X_train_list or not X_val_list:
        print("  ❌ 数据集为空，请检查年份设置")
        return None, None, None, None

    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    X_val = np.concatenate(X_val_list, axis=0)
    y_val = np.concatenate(y_val_list, axis=0)

    return X_train, y_train, X_val, y_val


def build_transformer_model(time_step, n_features, params):
    inputs = Input(shape=(time_step, n_features))

    # 1. 投影层
    x = Dense(params['d_model'])(inputs)

    # 2. 位置编码
    x = PositionalEncoding(params['d_model'])(x)

    # 3. Transformer Blocks
    for _ in range(params['trans_layers']):
        # Attention
        attn_output = MultiHeadAttention(
            num_heads=params['trans_heads'],
            key_dim=params['d_model']
        )(x, x)
        x = LayerNormalization(epsilon=1e-6)(x + attn_output)

        # FFN
        ffn_output = Dense(params['trans_ffn'], activation="relu")(x)
        ffn_output = Dense(params['d_model'])(ffn_output)
        ffn_output = Dropout(params['dropout_rate'])(ffn_output)
        x = LayerNormalization(epsilon=1e-6)(x + ffn_output)

    # 4. 输出层 (取最后一个时间步)
    x = x[:, -1, :]
    x = Dropout(params['dropout_rate'])(x)
    outputs = Dense(1, activation='linear')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=params['learning_rate']), loss='mse')
    return model


# ==============================================================================
# 4. 主流程
# ==============================================================================
if __name__ == "__main__":
    print(f"开始 Transformer 模型训练 (每类随机抽样 {SAMPLE_SIZE} 个点)...")
    print(f"训练: {TRAIN_START_YEAR}-{TRAIN_END_YEAR}, 验证: {VAL_YEAR}\n")

    for i, file_path in enumerate(FILE_PATHS):
        # 增加判断：如果是 None，跳过不处理
        if file_path is None:
            continue

        class_id = i + 1
        print(f"{'=' * 60}")
        print(f"处理类别 {class_id} ...")

        if not os.path.exists(file_path):
            print(f"❌ 文件不存在: {file_path}")
            continue

        # 1. 准备数据
        X_train, y_train, X_val, y_val = process_class_file(file_path, class_id, SAMPLE_SIZE)

        if X_train is None:
            print(f"❌ 类别 {class_id} 数据处理失败，跳过。")
            continue

        print(f"  训练集形状: X={X_train.shape}, y={y_train.shape}")
        print(f"  验证集形状: X={X_val.shape}, y={y_val.shape}")

        # 2. 构建模型
        params = CLASS_PARAMS_LIST[i]
        n_features = X_train.shape[2]  # 3

        model = build_transformer_model(TIME_STEP, n_features, params)

        # 3. 训练模型
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

        # 4. 保存模型
        model_name = f'model_class_{class_id}_Trans_VAL_ES_{TRAIN_START_YEAR}_2017.keras'
        save_path = os.path.join(MODEL_SAVE_DIR, model_name)
        model.save(save_path)

        val_loss = min(history.history['val_loss'])
        print(f"✅ 类别 {class_id} 训练完成，最佳 Val Loss: {val_loss:.6f}")
        print(f"   模型已保存: {save_path}")

        # 清理内存
        del model, X_train, y_train, X_val, y_val
        K.clear_session()
        gc.collect()

    print(f"\n{'=' * 60}")
    print("🎉 所有类别处理完毕！")