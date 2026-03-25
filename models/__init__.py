import os

from .unet3d import UNet3DTimeAsChannel

# 基础路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# 数据相关配置
CONFIG = {
    # 空间网格尺寸（请根据你们插值后的网格修改）
    "nx": 64,
    "ny": 32,
    "nz": 16,

    # 时间序列配置
    "input_steps": 6,   # 过去多少帧作为输入
    "pred_steps": 6,    # 预测未来多少帧

    # 通道配置：温度(1) + 风速(3) + mask(1) = 5
    "in_channels_per_step": 5,
    "out_channels_per_step": 1,  # 只预测温度

    # 训练参数
    "batch_size": 2,
    "num_epochs": 50,
    "learning_rate": 1e-3,
    "num_workers": 4,

    # 设备
    "device": "cuda",  # 如果没有GPU可以改成 "cpu"

    # 数据文件配置（示例：.npz 格式）
    # 你可以把 Fluent 插值后的结果存成类似：
    # data/train_case01.npz, data/train_case02.npz 等
    "train_file_list": os.path.join(DATA_DIR, "train_files.txt"),
    "val_file_list": os.path.join(DATA_DIR, "val_files.txt"),

    # 归一化配置
    "norm_eps": 1e-6,
}

__all__ = ["UNet3DTimeAsChannel", "CONFIG", "CHECKPOINT_DIR"]