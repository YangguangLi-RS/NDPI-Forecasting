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

# [修改] 模型保存路径 (Transformer, 无季节性)
MODEL_SAVE_DIR = 'data/区域数据/模型/Trans_Random50_NoSeasonality_2011_2017_12'
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# 站点经纬度保存路径
SITE_COORD_DIR = 'data/区域数据/模型/Site_Locations_Trans_NoSeasonality_12'
os.makedirs(SITE_COORD_DIR, exist_ok=True)

# 训练设置
TRAIN_START_YEAR = 2010
TRAIN_END_YEAR = 2016
VAL_YEAR = 2017
SAMPLE_SIZE = 50
TIME_STEP = 12

# [核心修改] 类别参数 (Transformer 12步长 无季节性特征 最优参数)
CLASS_PARAMS_LIST = [
    # [cite_start]类别 1 [cite: 27]
    {'d_model': 16, 'trans_layers': 2, 'trans_heads': 4, 'trans_ffn': 128,
     'dropout_rate': 0.3042, 'batch_size': 16, 'learning_rate': 0.00342, 'max_epochs': 558},
    # [cite_start]类别 2 [cite: 28]
    {'d_model': 16, 'trans_layers': 2, 'trans_heads': 2, 'trans_ffn': 128,
     'dropout_rate': 0.2116, 'batch_size': 16, 'learning_rate': 0.00358, 'max_epochs': 619},
    # [cite_start]类别 3 [cite: 28]
    {'d_model': 16, 'trans_layers': 1, 'trans_heads': 4, 'trans_ffn': 64,
     'dropout_rate': 0.3625, 'batch_size': 32, 'learning_rate': 0.00272, 'max_epochs': 697},
    # [cite_start]类别 4 [cite: 29]
    {'d_model': 32, 'trans_layers': 1, 'trans_heads': 2, 'trans_ffn': 128,
     'dropout_rate': 0.2380, 'batch_size': 16, 'learning_rate': 0.00213, 'max_epochs': 583},
    # [cite_start]类别 5 [cite: 30]
    {'d_model': 16, 'trans_layers': 2, 'trans_heads': 8, 'trans_ffn': 256,
     'dropout_rate': 0.3648, 'batch_size': 32, 'learning_rate': 0.00387, 'max_epochs': 726},
    # [cite_start]类别 6 [cite: 30]
    {'d_model': 16, 'trans_layers': 1, 'trans_heads': 8, 'trans_ffn': 64,
     'dropout_rate': 0.2536, 'batch_size': 16, 'learning_rate': 0.00337, 'max_epochs': 733}
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

def create_sliding_window(data_matrix, time_step):
    """
    构建滑动窗口数据集 (仅 NDPI，无季节性特征)
    X shape: (Samples, time_step, 1)
    """
    X, y = [], []
    L = len(data_matrix)
    for i in range(L - time_step):
        # 取 data_matrix 的一列 (NDPI)，reshape 为 (time_step, 1)
        ndpi_window = data_matrix[i: i + time_step].reshape(-1, 1)

        # [修改] 不再拼接时间特征，特征仅为 NDPI
        feature_window = ndpi_window

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
    print(f"  📍 [Class {class_idx}] 站点坐标已保存: {filepath}")


def process_class_file(file_path, class_idx, sample_size):
    print(f"[Class {class_idx}] 正在读取: {os.path.basename(file_path)}")
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"  [Class {class_idx}] 读取失败: {e}")
        return None, None, None, None

    # 解析日期
    data_cols = df.columns[:-2]
    try:
        dates = pd.to_datetime(data_cols)
    except:
        dates = pd.to_datetime(data_cols, errors='coerce')

    if dates.isnull().any():
        print(f"  [Class {class_idx}] 日期解析错误")
        return None, None, None, None

    # 随机抽样
    if len(df) > sample_size:
        df_sampled = df.sample(n=sample_size, random_state=42)
        print(f"  [Class {class_idx}] 随机抽取 {sample_size} 个站点 (原始 {len(df)} 个)")
    else:
        df_sampled = df
        print(f"  [Class {class_idx}] 样本不足 {sample_size}，使用全部 {len(df)} 个站点")

    # 保存坐标
    save_site_locations(df_sampled, class_idx)

    # [修改] 不生成季节性特征 (sin/cos)
    # doy = dates.dayofyear.values ...

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

    # print(f"  [Class {class_idx}] 正在构建滑动窗口序列...")
    for i in range(len(df_sampled)):
        site_data = ndpi_values[i, :]

        # 训练集
        train_data = site_data[train_mask]

        if len(train_data) > TIME_STEP:
            # [修改] 移除 time_features 参数
            X_t, y_t = create_sliding_window(train_data, TIME_STEP)
            X_train_list.append(X_t)
            y_train_list.append(y_t)

        # 验证集
        val_start_idx = np.argmax(val_mask)
        idx_start = max(0, val_start_idx - TIME_STEP)
        idx_end = val_start_idx + np.sum(val_mask)

        val_data_extended = site_data[idx_start: idx_end]

        if len(val_data_extended) > TIME_STEP:
            # [修改] 移除 time_features 参数
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


# ==============================================================================
# 4. Transformer 模型构建
# ==============================================================================

class TransformerBlock(Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate

        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            Dense(dff, activation='relu'),
            Dense(d_model)
        ])
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


def build_transformer_model(time_step, n_features, params):
    inputs = Input(shape=(time_step, n_features))

    # 1. 输入映射与位置编码
    # 将 (batch, 12, 1) 映射到 (batch, 12, d_model)
    x = Dense(params['d_model'])(inputs)
    x = PositionalEncoding(params['d_model'])(x)

    # 2. Transformer 层堆叠
    for _ in range(params['trans_layers']):
        x = TransformerBlock(
            d_model=params['d_model'],
            num_heads=params['trans_heads'],
            dff=params['trans_ffn'],
            dropout_rate=params['dropout_rate']
        )(x)

    # 3. 输出层 (取最后一个时间步)
    last_output = x[:, -1, :]
    outputs = Dense(1, activation='sigmoid')(last_output)

    model = Model(inputs=inputs, outputs=outputs)
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

    # 2. 构建模型
    params = CLASS_PARAMS_LIST[i]
    n_features = X_train.shape[2]  # 应该是 1

    model = build_transformer_model(TIME_STEP, n_features, params)

    # 3. 训练模型
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
        verbose=0  # 不显示详细日志，避免并行打印混乱
    )

    # 4. 保存模型
    model_name = f'model_class_{class_id}_Trans_NoSeasonality_{TRAIN_START_YEAR}_2017.keras'
    save_path = os.path.join(MODEL_SAVE_DIR, model_name)
    model.save(save_path)

    val_loss = min(history.history['val_loss'])
    print(f"✅ [Class {class_id}] 训练完成，最佳 Val Loss: {val_loss:.6f}")
    print(f"   [Class {class_id}] 模型已保存: {save_path}")

    # 清理内存
    del model, X_train, y_train, X_val, y_val
    K.clear_session()
    gc.collect()


# ==============================================================================
# 5. 主流程
# ==============================================================================
if __name__ == "__main__":
    print(f"开始 Transformer 模型池化训练 (并行6个任务, 无季节性特征)...")
    print(f"每类随机抽样: {SAMPLE_SIZE}")
    print(f"训练周期: {TRAIN_START_YEAR}-{TRAIN_END_YEAR}, 验证年份: {VAL_YEAR}\n")

    # 使用 joblib 并行执行 6 个类别的训练任务
    Parallel(n_jobs=6)(
        delayed(run_single_class_task)(i, file_path)
        for i, file_path in enumerate(FILE_PATHS)
    )

    print(f"\n{'=' * 60}")
    print("🎉 所有并行任务已分发完毕！")