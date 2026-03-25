import os

# 基础路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data_pkl")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# 数据相关配置
CONFIG = {
    # 空间网格尺寸（已根据实验室真实比例 3.7m : 8.2m : 3.0m 调整）
    # 比例约为 1.2 : 2.7 : 1 -> 网格设为 32 : 64 : 24
    "nx": 32,
    "ny": 64,
    "nz": 24,

    # 时间序列配置
    "input_steps": 2,   # 过去多少帧作为输入
    "pred_steps": 1,    # 预测未来多少帧,可以改为多步预测
    
    # 每个时间步的通道数：动态(8) + 静态/控制(12) = 20
    # 动态新增: Q_SRC, Global_Avg_T_Channel
    # 静态/控制新增: Cell_Zone_Mask + Current_Time_Norm + Fluid_Avg_T + Solid_Avg_T
    #               + AC_Vel_Set + Fan_Vel_Set + AC_Temp_Set + Fan_Temp_Set
    "in_channels_per_step": 20,

    # 输出通道: T, U, V, W, K, NUT (每个预测时间步共6个)
    "out_channels_per_step": 6,

    # 动态通道: T, U, V, W, K, NUT, Q_SRC, Global_Avg_T_Channel (共8个)
    # 静态/控制通道: Wall_Dist, Inlet_Mask, Z_Coord, Time_Step, Cell_Zone_Mask,
    #               Current_Time_Norm, Fluid_Avg_T, Solid_Avg_T,
    #               AC_Vel_Set, Fan_Vel_Set, AC_Temp_Set, Fan_Temp_Set (共12个)
    # 总输入通道 = (动态 * input_steps) + 静态
    "dynamic_channels": 8,
    "static_channels": 12,

    # 多物理量联合损失加权（T 占 0.7，其余流场量共占 0.3）
    "loss_temp_weight": 0.7,
    "loss_flow_weight": 0.3,
    
    # 默认训练时的空调设置温度 (如果数据中没有标签，则假设为该值)
    "default_ac_temp": 26.0,
    # 空调温度归一化范围 (min, max)
    "ac_temp_range": (16.0, 30.0),

    # 时间步归一化基准 (最大预计步长)
    "max_dt": 2.0,
 
    # 数据增强概率
    "aug_prob": 0.5,

    # 训练超参数
    "batch_size": 2,
    "num_epochs": 20,
    "learning_rate": 1e-3,
    "num_workers": 2,  # Windows 下建议设为 0 或较小值以避免并行错误
    "norm_eps": 1e-6,

    # 设备与加速策略
    "device": "cuda",            # 优先使用 GPU，自动回退到 CPU
    "use_amp": True,               # 自动混合精度以提升显存与吞吐
    "cudnn_benchmark": True,       # 对固定输入尺寸的 3D UNet 收敛更快
    "allow_tf32": True,            # Ampere+ GPU 上允许 TF32 计算

    # UDF 控制逻辑复刻参数（与 Fluent UDF 保持一致）
    "udf_control": {
        "ac_t_low": 297.15,
        "ac_t_mid": 299.15,
        "ac_t_high": 302.15,
        "ac_outlet_temp": 293.15,
        "ac_vel_low": 1.5,
        "ac_vel_mid": 3.0,
        "ac_vel_high": 5.0,
        "fan_t_threshold": 299.15,
        "fan_vel_low": 3.0,
        "fan_vel_high": 5.0,
        # 若为空则默认使用 CELL_ZONE > 0 作为流体区
        "fluid_zone_ids": []
    },

    # 数据预处理插值后端：可用时优先使用 GPU(CuPy)，失败自动回退 CPU(SciPy)
    "enable_gpu_preprocess": True,

    # 数据文件配置
    # 仅保留 2s 步长的数据集
    "train_datasets": [
        {"file_list": os.path.join(DATA_DIR, "train_files.txt"), "dt": 2.0},
    ],
    "val_datasets": [
        {"file_list": os.path.join(DATA_DIR, "val_files.txt"), "dt": 2.0},
    ],

    # AC 空调对应网格区域 (Z_start, Z_end, Y_start, Y_end, X_start, X_end)
    # 请根据实际情况修改这些索引以匹配空调出风口位置
    # 示例: 假设空调在墙壁上方
    "ac_region_indices": [18, 24, 28, 36, 0, 2], 
    
    # 训练/验证文件列表 (保留用于兼容)
    "train_file_list": os.path.join(DATA_DIR, "train_files.txt"),
    "val_file_list": os.path.join(DATA_DIR, "val_files.txt"),

    # CSV 数据列名映射
    "csv_mapping": {
        "x-coordinate": "x-coordinate",
        "y-coordinate": "y-coordinate",
        "z-coordinate": "z-coordinate",
        
        "static-temperature": "temperature",
        "x-velocity": "x-velocity",
        "y-velocity": "y-velocity",
        "z-velocity": "z-velocity",
        "turb-kinetic-energy": "turb-kinetic-energy",
        "turb-viscosity": "viscosity-turb",
        "user-energy-source": "user-energy-source",
        "cell-zone": "cell-zone",
        "cell-volume": "cell-volume"
    },

    # 归一化参数文件
    "stats_file": os.path.join(DATA_DIR, "normalization_stats.json")
}
