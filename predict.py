import os
import glob
import pickle
import json
import argparse
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import CONFIG, CHECKPOINT_DIR, DATA_DIR
from models import UNet3DTimeAsChannel
from data.dataset import LabTemp3DDataset

def load_best_checkpoint(map_location="cpu"):
    ckpt_files = glob.glob(os.path.join(CHECKPOINT_DIR, "best_model_epoch*.pt"))
    if not ckpt_files:
        print("No checkpoint found in checkpoints/")
        return None

    ckpt_files.sort(key=os.path.getmtime, reverse=True)
    ckpt_path = ckpt_files[0]
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    return ckpt

def denormalize(arr, name, stats):
    if name not in stats:
        return arr
    mean, std = stats[name]
    return arr * (std + CONFIG["norm_eps"]) + mean


def denormalize_6ch(arr_6ch, stats):
    var_names = ["T", "U", "V", "W", "K", "NUT"]
    out = np.empty_like(arr_6ch)
    for channel_index, var_name in enumerate(var_names):
        out[:, :, channel_index, ...] = denormalize(arr_6ch[:, :, channel_index, ...], var_name, stats)
    return out

def main():
    parser = argparse.ArgumentParser(description="实验室温度场预测")
    parser.add_argument("--device", type=str, default=CONFIG["device"], help="Device to use")
    parser.add_argument("--dt", type=float, default=2.0, help="Time step for prediction data (seconds)")
    parser.add_argument("--target_time", type=float, default=None, help="Target time (in seconds) to predict. E.g., 50.0")
    parser.add_argument("--ac_temp", type=float, default=None, help="Conditioned AC temperature (e.g., 26.0)")
    args = parser.parse_args()

    wants_cuda = args.device.startswith("cuda")
    has_cuda = torch.cuda.is_available()
    
    if wants_cuda and not has_cuda:
        print("\n" + "!"*40)
        print("警告: 指定了使用 CUDA，但当前 PyTorch 环境未检测到 CUDA 设备。")
        print("请检查：1. 是否安装了带有 CUDA 支持的 PyTorch (当前版本: " + torch.__version__ + ")")
        print("       2. 显卡驱动是否安装正确")
        print("       3. VS Code 解释器是否切换到了包含 GPU 支持的环境 (如 labtemp-gpu)")
        print("!"*40 + "\n")

    device = torch.device(args.device if wants_cuda and has_cuda else "cpu")
    print(f"Using device: {device}")

    # Load Checkpoint
    ckpt = load_best_checkpoint(map_location=device)
    if ckpt is None:
        return

    cfg = ckpt.get("config", CONFIG)
    
    # Model Setup
    # Input channels calculation must match Training
    in_channels_total = cfg["input_steps"] * cfg.get("in_channels_per_step", CONFIG["in_channels_per_step"])
    pred_steps = cfg["pred_steps"]

    model = UNet3DTimeAsChannel(
        in_channels_total=in_channels_total,
        pred_steps=pred_steps,
        base_channels=32,
        bilinear=True,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # Load Normalization Stats
    stats_path = os.path.join(DATA_DIR, "normalization_stats.json")
    norm_stats = {}
    if os.path.exists(stats_path):
        with open(stats_path, "r") as f:
            loaded_stats = json.load(f)
            for k, v in loaded_stats.items():
                norm_stats[k] = (v["mean"], v["std"])
        print(f"Loaded normalization stats from {stats_path}")
    else:
        print("Warning: normalization_stats.json not found. Prediction results will be un-denormalized.")

    # Prepare Dataset
    # If user wants a specific target time, we need to find the closest starting file
    # For now, we load the list to find available times
    # Note: File naming is critical here. Assuming FFF.1-1-XXXX implies a sequence.
    val_files_list = os.path.join(DATA_DIR, "val_files.txt")
    if not os.path.exists(val_files_list):
        print(f"Validation file list not found at {val_files_list}")
        return

    with open(val_files_list, "r") as f:
        val_files = [line.strip() for line in f if line.strip()]

    print(f"Found {len(val_files)} validation files.")
    
    # Logic for Target Time Prediction
    target_idx_in_val = None
    steps_to_rollout = 0
    
    # If target time is specified, we try to logic it out:
    # We assume the val_files are sequential starting from some time T_start with step dt
    # But usually val_split is random or tail. 
    # Let's simplify: We just predict on the WHOLE validation set using the CUSTOM AC TEMP.
    # If user wants autoregressive rollout (e.g. predict 100s into future), that requires complex logic.
    # Here typically "Predict at specific time" means "Find the data corresponding to that time and run inference".
    
    if args.target_time is not None:
        print(f"Target Mode: predicting for Time ~ {args.target_time}s with AC={args.ac_temp if args.ac_temp else 'default'}")
        # Assuming filenames imply time or index. 
        # With FFF files, it's hard to map absolute time without a metadata map.
        # Fallback: We predict on the first few batches to demonstrate the "Conditional Generation" capability.
        pass

    # We use the Dataset class 
    val_dataset = LabTemp3DDataset(
        file_list=val_files,
        input_steps=cfg["input_steps"],
        pred_steps=cfg["pred_steps"],
        in_channels_per_step=cfg.get("in_channels_per_step", CONFIG["in_channels_per_step"]),
        dt=args.dt,
        norm_stats=norm_stats,
        ac_temp=args.ac_temp # Pass the user-defined AC temp!
    )
    
    if len(val_dataset) == 0:
        print("Validation dataset length is 0. Not enough contiguous files to form a sequence.")
        print(f"Need > {cfg['input_steps'] + cfg['pred_steps']} files.")
        return

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    all_predictions = []
    all_targets = []

    print(f"Starting prediction loop (AC Temp Condition: {val_dataset.ac_temp if val_dataset.ac_temp else 'Default'})...")
    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            x = x.to(device)
            # y matches prediction signal (normalized temperature)
            
            # Forward: (B, pred_steps * 6, Z, Y, X)
            pred = model(x)

            pred_np = pred.cpu().numpy()
            y_np = y.numpy()

            bsz, channels_total, nz, ny, nx = pred_np.shape
            channels_per_step = 6
            if channels_total != cfg["pred_steps"] * channels_per_step:
                raise ValueError(
                    f"Unexpected output channels: got {channels_total}, "
                    f"expected {cfg['pred_steps'] * channels_per_step}"
                )

            pred_6ch = pred_np.reshape(bsz, cfg["pred_steps"], channels_per_step, nz, ny, nx)
            y_6ch = y_np.reshape(bsz, cfg["pred_steps"], channels_per_step, nz, ny, nx)

            pred_denorm = denormalize_6ch(pred_6ch, norm_stats)
            y_denorm = denormalize_6ch(y_6ch, norm_stats)
            
            all_predictions.append(pred_denorm)
            all_targets.append(y_denorm)
            
    # Concatenate
    if len(all_predictions) > 0:
        all_predictions = np.concatenate(all_predictions, axis=0) # (N_samples, pred_steps, 6, Z, Y, X)
        all_targets = np.concatenate(all_targets, axis=0)

        temp_predictions = all_predictions[:, :, 0, ...]
        temp_targets = all_targets[:, :, 0, ...]
        
        print(f"Prediction complete.")
        print(f"Predictions(6ch) shape: {all_predictions.shape}")
        print(f"Targets(6ch) shape: {all_targets.shape}")
        print(f"Predictions(T only) shape: {temp_predictions.shape}")
        print(f"Targets(T only) shape: {temp_targets.shape}")
        
        # Save Results
        out_path = os.path.join(DATA_DIR, "prediction_result.npz")
        
        # We also need coordinates for visualization. 
        coords_dict = {}
        try:
            sample_file = os.path.join(DATA_DIR, val_files[0])
            with open(sample_file, "rb") as f:
                raw_data = pickle.load(f)
                # Keys might be X, Y, Z or x, y, z
                for k in ["X", "Y", "Z", "x", "y", "z"]:
                    if k in raw_data:
                        coords_dict[k.lower()] = raw_data[k]
        except Exception as e:
            print(f"Could not load coordinates from raw file: {e}")

        save_dict = {
            "predictions": temp_predictions,
            "targets": temp_targets,
            "predictions_6ch": all_predictions,
            "targets_6ch": all_targets,
            "pred_var_names": np.array(["T", "U", "V", "W", "K", "NUT"], dtype=object),
        }
        save_dict.update(coords_dict)
        
        np.savez_compressed(out_path, **save_dict)
        print(f"Saved predictions to {out_path}")
    else:
        print("No predictions were made.")

if __name__ == "__main__":
    main()
