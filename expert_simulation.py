import os
import argparse
import json
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, ConcatDataset

from config import CONFIG, CHECKPOINT_DIR, DATA_DIR
from models import UNet3DTimeAsChannel
from data.dataset import LabTemp3DDataset, _read_file_list

# ==========================================
# 专家模块：模型加载与工具函数
# ==========================================

def load_model(device):
    """加载最优模型"""
    import glob
    ckpt_files = glob.glob(os.path.join(CHECKPOINT_DIR, "best_model_epoch*.pt"))
    if not ckpt_files:
        raise FileNotFoundError("没有找到训练好的模型 Checkpoint。")
    
    ckpt_files.sort(key=os.path.getmtime, reverse=True)
    ckpt_path = ckpt_files[0]
    print(f"[Expert] Loading Checkpoint: {ckpt_path}")
    
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config", CONFIG)
    
    # 确保模型结构匹配
    in_channels_total = cfg["input_steps"] * cfg.get("in_channels_per_step", CONFIG["in_channels_per_step"])
    
    model = UNet3DTimeAsChannel(
        in_channels_total=in_channels_total,
        pred_steps=cfg["pred_steps"],
        base_channels=32,
        bilinear=True
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, cfg

def load_stats():
    """加载归一化参数"""
    stats_path = os.path.join(DATA_DIR, "normalization_stats.json")
    if not os.path.exists(stats_path):
        raise FileNotFoundError("找不到 normalization_stats.json，无法进行物理量转换。")
    with open(stats_path, "r") as f:
        stats = json.load(f)
    # 转换为 (mean, std) 格式
    return {k: (v["mean"], v["std"]) for k, v in stats.items()}

def normalize(val, name, stats):
    mean, std = stats[name]
    return (val - mean) / (std + CONFIG["norm_eps"])

def denormalize(val, name, stats):
    mean, std = stats[name]
    return val * (std + CONFIG["norm_eps"]) + mean


def compute_weighted_global_avg_t(t_field, fluid_mask, volume_field):
    weighted_mask = fluid_mask & torch.isfinite(volume_field) & (volume_field > 0)
    if torch.any(weighted_mask):
        vol_sum = volume_field[weighted_mask].sum()
        if vol_sum > 0:
            return (t_field[weighted_mask] * volume_field[weighted_mask]).sum() / vol_sum

    if torch.any(fluid_mask):
        return t_field[fluid_mask].mean()
    return t_field.mean()

# ==========================================
# Part 1: 模型有效性评价
# ==========================================

def evaluate_effectiveness(model, device, stats):
    print("\n" + "="*40)
    print("Part 1: 模型有效性评价 (Model Validation)")
    print("="*40)
    
    # 构建验证集 (只使用 2s 步长配置)
    val_files_path = CONFIG.get("val_file_list")
    if not os.path.exists(val_files_path):
        print(f"验证列表 {val_files_path} 不存在，跳过评价。")
        return

    file_list = _read_file_list(val_files_path)
    if not file_list:
        print("验证列表为空。")
        return
        
    dataset = LabTemp3DDataset(
        file_list,
        input_steps=CONFIG["input_steps"],
        pred_steps=CONFIG["pred_steps"],
        in_channels_per_step=CONFIG["in_channels_per_step"],
        dt=2.0, # 强制评估 2s 模型
        norm_stats=stats
    )
    
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    mses, maes = [], []
    
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.to(device)
            y_pred = model(x).cpu().numpy()
            y_true = y.numpy()
            
            # 反归一化回摄氏度进行误差计算，更直观
            T_pred = denormalize(y_pred, "T", stats)
            T_true = denormalize(y_true, "T", stats)
            
            mse = np.mean((T_pred - T_true)**2)
            mae = np.mean(np.abs(T_pred - T_true))
            
            mses.append(mse)
            maes.append(mae)
            
            if i >= 50: break # 只评估前50个样本节省时间
            
    print(f"评估样本数: {len(mses)}")
    print(f"平均均方误差 (MSE): {np.mean(mses):.4f} (℃^2)")
    print(f"平均绝对误差 (MAE): {np.mean(maes):.4f} (℃)")
    print("评价结论: 模型在 2s 步长下的单步预测精度如上。")

# ==========================================
# Part 2: 未来预测与空调控制
# ==========================================

def apply_ac_boundary(input_tensor, ac_temp_norm):
    """
    在输入 Tensor 中强制设定空调出风口温度。
    input_tensor: (1, Channels, Z, Y, X)
    通道布局按 dataset.py：每帧第 0 通道为 T。
    仅覆盖 Temperature，不干预 U/V/W/K/NUT（由网络自行预测）。
    """
    
    # 读取 Config 中的区域
    # 格式 [z_s, z_e, y_s, y_e, x_s, x_e]
    idx = CONFIG.get("ac_region_indices", [18, 24, 28, 36, 0, 2])
    z_s, z_e, y_s, y_e, x_s, x_e = idx
    
    input_steps = CONFIG["input_steps"]
    ch_per_step = CONFIG["in_channels_per_step"]
    
    # 对每一个时间步的 T 通道 (第0通道) 进行修改
    for t in range(input_steps):
        # T 通道在 channel 维度的索引
        # Layout: [Step0_Ch0...Step0_ChN, Step1_Ch0...Step1_ChN]
        t_channel_idx = t * ch_per_step + 0 
        
        # 强制赋值
        input_tensor[0, t_channel_idx, z_s:z_e, y_s:y_e, x_s:x_e] = ac_temp_norm
        
    return input_tensor

def predict_future_with_ac(model, device, stats, future_seconds, ac_temp_c):
    print("\n" + "="*40)
    print(f"Part 2: 未来预测仿真 (Future Simulation)")
    print(f"目标: {future_seconds} 秒后 | 空调设定: {ac_temp_c} ℃")
    print("="*40)
    
    dt = 2.0 # 模型原生步长
    steps_needed = int(future_seconds / dt)
    print(f"需要迭代预测 {steps_needed} 步 (每步 {dt}s)")
    
    # 1. 获取初始状态 (从验证集取最后一个样本作为当前状态)
    val_files_path = CONFIG.get("val_file_list")
    file_list = _read_file_list(val_files_path)
    # 取其中一个作为初始状态
    start_file_idx = 0 
    
    dataset = LabTemp3DDataset(
        file_list,
        input_steps=CONFIG["input_steps"],
        pred_steps=CONFIG["pred_steps"], # 实际上这里不用pred，只是为了加载input
        in_channels_per_step=CONFIG["in_channels_per_step"],
        dt=dt,
        norm_stats=stats
    )
    
    # 获取 (Input, Target)
    # Input Shape: (C_total, Z, Y, X)
    current_input, _ = dataset[start_file_idx] 
    current_input = current_input.unsqueeze(0).to(device) # (1, C_total, Z, Y, X)

    # 读取固定几何相关场用于体积加权平均温度（VOL 与 fluid mask）
    vol_ref = None
    fluid_mask_ref = None
    try:
        base_frame_path = os.path.join(DATA_DIR, file_list[start_file_idx])
        with open(base_frame_path, "rb") as f:
            base_frame = pickle.load(f)

        if "VOL" in base_frame:
            vol_np = np.asarray(base_frame["VOL"], dtype=np.float32)
            vol_ref = torch.from_numpy(vol_np).to(device).unsqueeze(0)

        if "Cell_Zone_Mask" in base_frame:
            mask_np = (np.asarray(base_frame["Cell_Zone_Mask"], dtype=np.float32) > 0.5)
            fluid_mask_ref = torch.from_numpy(mask_np).to(device).unsqueeze(0)
    except Exception:
        vol_ref = None
        fluid_mask_ref = None
    
    # 归一化目标空调温度
    ac_temp_norm = normalize(ac_temp_c, "T", stats)
    
    pred_field_norm = None
    pred_var_names = ["T", "U", "V", "W", "K", "NUT"]
    
    # 2. 自回归迭代
    model.eval()
    with torch.no_grad():
        for step in range(steps_needed):
            # A. 施加空调边界条件 (修改 Input 中的 T 通道)
            current_input = apply_ac_boundary(current_input, ac_temp_norm)
            
            # B. 预测下一步
            # Output: (B, pred_steps*6, Z, Y, X)
            pred_raw = model(current_input)
            bsz, _, nz, ny, nx = pred_raw.shape
            pred_steps = CONFIG["pred_steps"]
            pred_reshaped = pred_raw.view(bsz, pred_steps, 6, nz, ny, nx)

            # 自回归使用“最近一步”预测（第 0 个 future step）
            next_dyn6 = pred_reshaped[:, 0, ...]  # (B, 6, Z, Y, X)
            pred_field_norm = next_dyn6
            
            # C. 更新 Input Window (滑动窗口)
            # 用预测出的 6 个动态场整体更新下一帧：T/U/V/W/K/NUT
            input_steps = CONFIG["input_steps"]
            ch = CONFIG["in_channels_per_step"]

            # 1) 取当前窗口最后一帧作为模板（保留静态通道与未预测动态通道）
            frame_last_idx = (input_steps - 1) * ch
            frame_last = current_input[:, frame_last_idx : frame_last_idx+ch, ...]

            # 2) 构造新帧并覆盖前 6 个动态通道
            frame_new = frame_last.clone()
            frame_new[:, 0:6, ...] = next_dyn6

            # 3) 依据新预测温度，重算 Global_Avg_T_Channel（第 7 通道）
            # 动态通道约定: [T,U,V,W,K,NUT,Q_SRC,Global_Avg_T]
            # 静态通道最后包含 Cell_Zone_Mask（当前配置为第 12 通道）
            if ch > 7:
                t_pred = frame_new[:, 0, ...]
                if fluid_mask_ref is not None:
                    fluid_mask = fluid_mask_ref
                else:
                    cell_zone_mask = frame_new[:, 12, ...] if ch > 12 else torch.ones_like(t_pred)
                    fluid_mask = cell_zone_mask > 0.5

                if vol_ref is not None:
                    avg_t = compute_weighted_global_avg_t(t_pred, fluid_mask, vol_ref)
                else:
                    if torch.any(fluid_mask):
                        avg_t = t_pred[fluid_mask].mean()
                    else:
                        avg_t = t_pred.mean()
                frame_new[:, 7, ...] = torch.full_like(frame_new[:, 7, ...], avg_t)

            # 4) 拼接滑动窗口
            if input_steps > 1:
                remainder = current_input[:, ch:, ...]
                current_input = torch.cat([remainder, frame_new], dim=1)
            else:
                current_input = frame_new
                
            print(f"Step {step+1}/{steps_needed} 完成...")

    # 3. 保存最后一步的 6 通道结果
    final_pred_norm = pred_field_norm.cpu().numpy()  # (1, 6, Z, Y, X)
    final_pred_denorm = np.empty_like(final_pred_norm)
    for i, name in enumerate(pred_var_names):
        if name in stats:
            mean, std = stats[name]
            final_pred_denorm[:, i, ...] = final_pred_norm[:, i, ...] * (std + CONFIG["norm_eps"]) + mean
        else:
            final_pred_denorm[:, i, ...] = final_pred_norm[:, i, ...]

    final_T = final_pred_denorm[:, 0:1, ...]
    
    # 获取坐标用于保存
    try:
        sample_path = os.path.join(DATA_DIR, file_list[0])
        with open(sample_path, "rb") as f:
            raw_d = pickle.load(f)
            coords = {k: raw_d[k] for k in ["X", "Y", "Z"] if k in raw_d}
    except:
        coords = {}

    out_file = os.path.join(DATA_DIR, "simulation_result.npz")
    np.savez_compressed(
        out_file,
        predictions=final_T,
        predictions_6ch=final_pred_denorm,
        pred_var_names=np.array(pred_var_names, dtype=object),
        **coords,
    )
    print(f"仿真结果已保存至: {out_file}")
    
    # 简单的切片可视化
    if "Z" in coords:
        try:
            mid_z = final_T.shape[2] // 2
            plt.figure(figsize=(10, 6))
            plt.imshow(final_T[0, 0, mid_z, :, :], cmap='jet', origin='lower')
            plt.colorbar(label='Temperature (C)')
            plt.title(f"Simulation Result (+{future_seconds}s, AC={ac_temp_c}C)\nZ-Slice Middle")
            vis_png = os.path.join(DATA_DIR, "sim_vis.png")
            plt.savefig(vis_png)
            print(f"可视化切片已保存至: {vis_png}")
        except Exception as e:
            print(f"Visualization failed: {e}")

def main():
    device = torch.device(CONFIG["device"] if torch.cuda.is_available() else "cpu")
    
    # 1. 加载资源
    try:
        model, _ = load_model(device)
        stats = load_stats()
    except Exception as e:
        print(f"初始化失败: {e}")
        return

    # 2. 交互式输入
    print("\n[Expert System Ready]")
    
    # Step A: Run Evaluation
    evaluate_effectiveness(model, device, stats)
    
    # Step B: User Input for Simulation
    try:
        future_input = input("\n请输入未来预测时间 (秒) [默认 60]: ").strip()
        future_time = float(future_input) if future_input else 60.0
        
        ac_input = input("请输入空调设定温度 (℃) [默认 20]: ").strip()
        ac_temp = float(ac_input) if ac_input else 20.0
    except ValueError:
        print("输入格式错误，使用默认值: 60s, 20℃")
        future_time, ac_temp = 60.0, 20.0
        
    # Step C: Run Simulation
    predict_future_with_ac(model, device, stats, future_time, ac_temp)

if __name__ == "__main__":
    main()
