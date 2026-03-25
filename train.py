import csv
import os
import glob
import argparse
import json
from pathlib import Path
from contextlib import nullcontext
from time import time

import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.optim import Adam
from tqdm import tqdm

from config import CONFIG, CHECKPOINT_DIR, DATA_DIR
from data import create_dataloaders
from models import UNet3DTimeAsChannel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "vis_results")
os.makedirs(LOG_DIR, exist_ok=True)
LOSS_LOG_PATH = os.path.join(LOG_DIR, "train_val_loss.csv")


def load_t_mse_scale() -> float:
    stats_path = CONFIG.get("stats_file", "")
    if not stats_path:
        return 1.0
    if not os.path.exists(stats_path):
        return 1.0

    try:
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
        t_item = stats.get("T", None)
        if isinstance(t_item, dict):
            t_std = float(t_item.get("std", 1.0))
        elif isinstance(t_item, (list, tuple)) and len(t_item) >= 2:
            t_std = float(t_item[1])
        else:
            t_std = 1.0
        return (t_std + CONFIG.get("norm_eps", 1e-6)) ** 2
    except Exception:
        return 1.0


def _read_list_file(path: str):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def resolve_active_fold_id(explicit_fold_id=None):
    if explicit_fold_id is not None:
        return int(explicit_fold_id)

    current_val = _read_list_file(os.path.join(DATA_DIR, "val_files.txt"))
    if not current_val:
        return None

    splits_dir = os.path.join(DATA_DIR, "splits")
    if not os.path.isdir(splits_dir):
        return None

    pattern = os.path.join(splits_dir, "control_fold_*_val_files.txt")
    for val_path in glob.glob(pattern):
        candidate = _read_list_file(val_path)
        if candidate == current_val:
            stem = Path(val_path).stem  # control_fold_1_val_files
            parts = stem.split("_")
            if len(parts) >= 3 and parts[2].isdigit():
                return int(parts[2])
    return None


def find_latest_checkpoint() -> str:
    ckpt_files = glob.glob(os.path.join(CHECKPOINT_DIR, "best_model_epoch*.pt"))
    if not ckpt_files:
        return ""
    ckpt_files.sort(key=os.path.getmtime, reverse=True)
    return ckpt_files[0]


def autocast_context(device, amp_enabled: bool):
    if amp_enabled and device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", enabled=True)
    return nullcontext()


def weighted_multichannel_mse_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    pred_steps: int,
    temp_weight: float,
    flow_weight: float,
):
    channels_per_step = 6
    expected_channels = pred_steps * channels_per_step
    if outputs.shape[1] != expected_channels or targets.shape[1] != expected_channels:
        raise ValueError(
            f"Expected channel dim = pred_steps*6 = {expected_channels}, "
            f"but got outputs {outputs.shape[1]} and targets {targets.shape[1]}"
        )

    out_reshaped = outputs.view(outputs.shape[0], pred_steps, channels_per_step, *outputs.shape[2:])
    tgt_reshaped = targets.view(targets.shape[0], pred_steps, channels_per_step, *targets.shape[2:])

    mse_temp = nn.functional.mse_loss(out_reshaped[:, :, 0, ...], tgt_reshaped[:, :, 0, ...])
    mse_flow = nn.functional.mse_loss(out_reshaped[:, :, 1:, ...], tgt_reshaped[:, :, 1:, ...])

    loss = temp_weight * mse_temp + flow_weight * mse_flow
    return loss, mse_temp, mse_flow


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    epoch,
    lambda_smooth: float = 1e-4,
    scaler=None,
    amp_enabled: bool = False,
):
    model.train()
    running_loss = 0.0
    pred_steps = CONFIG["pred_steps"]
    temp_weight = CONFIG.get("loss_temp_weight", 0.7)
    flow_weight = CONFIG.get("loss_flow_weight", 0.3)

    pbar = tqdm(loader, desc=f"Train Epoch {epoch}", ncols=100)
    for batch_idx, (inputs, targets) in enumerate(pbar):
        # inputs: (B, C_in_total, Z, Y, X)
        # targets: (B, pred_steps * 6, Z, Y, X)
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, amp_enabled):
            outputs = model(inputs)  # (B, pred_steps * 6, Z, Y, X)
            loss_data, loss_temp, loss_flow = weighted_multichannel_mse_loss(
                outputs=outputs,
                targets=targets,
                pred_steps=pred_steps,
                temp_weight=temp_weight,
                flow_weight=flow_weight,
            )

            # 可选的平滑正则项：鼓励空间梯度平滑
            dz = outputs[:, :, 1:, :, :] - outputs[:, :, :-1, :, :]
            dy = outputs[:, :, :, 1:, :] - outputs[:, :, :, :-1, :]
            dx = outputs[:, :, :, :, 1:] - outputs[:, :, :, :, :-1]
            smooth_loss = (dz.abs().mean() + dy.abs().mean() + dx.abs().mean())

            loss = loss_data + lambda_smooth * smooth_loss

        if amp_enabled and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix(
            {
                "loss": f"{loss.item():.4e}",
                "data": f"{loss_data.item():.4e}",
                "temp": f"{loss_temp.item():.4e}",
                "flow": f"{loss_flow.item():.4e}",
                "smooth": f"{smooth_loss.item():.4e}",
            }
        )

    return running_loss / len(loader)


@torch.no_grad()
def validate(model, loader, device, epoch, amp_enabled: bool = False):
    model.eval()
    running_loss = 0.0
    running_temp_loss_norm = 0.0
    pred_steps = CONFIG["pred_steps"]
    temp_weight = CONFIG.get("loss_temp_weight", 0.7)
    flow_weight = CONFIG.get("loss_flow_weight", 0.3)

    pbar = tqdm(loader, desc=f"Val Epoch {epoch}", ncols=100)
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with autocast_context(device, amp_enabled):
            outputs = model(inputs)
            loss, loss_temp, loss_flow = weighted_multichannel_mse_loss(
                outputs=outputs,
                targets=targets,
                pred_steps=pred_steps,
                temp_weight=temp_weight,
                flow_weight=flow_weight,
            )
        running_loss += loss.item()
        running_temp_loss_norm += loss_temp.item()
        pbar.set_postfix(
            {
                "loss": f"{loss.item():.4e}",
                "temp": f"{loss_temp.item():.4e}",
                "flow": f"{loss_flow.item():.4e}",
            }
        )

    avg_loss = running_loss / len(loader)
    avg_temp_loss_norm = running_temp_loss_norm / len(loader)
    t_mse_scale = load_t_mse_scale()
    avg_temp_loss_c2 = avg_temp_loss_norm * t_mse_scale
    return avg_loss, avg_temp_loss_norm, avg_temp_loss_c2


def main():
    parser = argparse.ArgumentParser(description="Train 3D UNet surrogate model")
    parser.add_argument("--device", type=str, default=CONFIG["device"], help="Device to use, e.g. cuda or cpu")
    parser.add_argument("--epochs", type=int, default=None, help="Override CONFIG['num_epochs']")
    parser.add_argument("--resume", type=str, default="", help="Checkpoint path to resume model weights from")
    parser.add_argument("--resume_latest", action="store_true", help="Resume from latest best_model checkpoint")
    parser.add_argument("--load_optimizer", action="store_true", help="Also load optimizer state when resuming")
    parser.add_argument("--fold_id", type=int, default=None, help="Optional fold id for per-fold logs")
    args = parser.parse_args()

    wants_cuda = args.device.startswith("cuda")
    has_cuda = torch.cuda.is_available()
    
    if wants_cuda and not has_cuda:
        print("\n" + "!"*40)
        print("警告: 配置文件指定了使用 CUDA，但当前 PyTorch 环境未检测到 CUDA 设备。")
        print("请检查：1. 是否安装了带有 CUDA 支持的 PyTorch (当前版本: " + torch.__version__ + ")")
        print("       2. 显卡驱动是否安装正确")
        print("       3. VS Code 解释器是否切换到了包含 GPU 支持的环境 (如 labtemp-gpu)")
        print("!"*40 + "\n")
    
    device = torch.device(args.device if wants_cuda and has_cuda else "cpu")
    print(f"Using device: {device}")

    if device.type == "cuda":
        if CONFIG.get("cudnn_benchmark", True):
            torch.backends.cudnn.benchmark = True
        if CONFIG.get("allow_tf32", True):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    amp_enabled = device.type == "cuda" and CONFIG.get("use_amp", True)
    scaler = GradScaler('cuda', enabled=amp_enabled) if amp_enabled else None

    train_loader, val_loader = create_dataloaders()

    in_channels_total = CONFIG["input_steps"] * CONFIG["in_channels_per_step"]
    pred_steps = CONFIG["pred_steps"]

    model = UNet3DTimeAsChannel(
        in_channels_total=in_channels_total,
        pred_steps=pred_steps,
        base_channels=32,
        bilinear=True,
    )

    if device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"Using DataParallel across {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    model = model.to(device)

    optimizer = Adam(model.parameters(), lr=CONFIG["learning_rate"])

    resume_path = ""
    if args.resume:
        resume_path = args.resume
    elif args.resume_latest:
        resume_path = find_latest_checkpoint()

    if resume_path:
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

        print(f"Resuming model from: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model_to_load = model.module if isinstance(model, nn.DataParallel) else model
        model_to_load.load_state_dict(ckpt["model_state_dict"], strict=True)

        if args.load_optimizer and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            print("Loaded optimizer state from resume checkpoint.")

    best_val_loss = float("inf")
    start_time = time()
    epochs_to_run = args.epochs if args.epochs is not None else CONFIG["num_epochs"]

    active_fold_id = resolve_active_fold_id(args.fold_id)
    fold_log_path = None
    if active_fold_id is not None:
        fold_dir = os.path.join(LOG_DIR, "folds", f"fold_{active_fold_id}")
        os.makedirs(fold_dir, exist_ok=True)
        fold_log_path = os.path.join(fold_dir, "train_val_loss.csv")
        print(f"Active fold detected: fold_{active_fold_id}")
        print(f"Per-fold loss log: {fold_log_path}")

    with open(LOSS_LOG_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_t_mse_c2"])

    if fold_log_path is not None:
        with open(fold_log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss", "val_t_mse_c2"])

    for epoch in range(1, epochs_to_run + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            scaler=scaler,
            amp_enabled=amp_enabled,
        )
        val_loss, val_t_mse_norm, val_t_mse_c2 = validate(
            model, val_loader, device, epoch, amp_enabled=amp_enabled
        )

        print(
            f"Epoch {epoch}: train_loss={train_loss:.6e}, "
            f"val_loss={val_loss:.6e}, "
            f"val_T_MSE(℃²)={val_t_mse_c2:.6e}"
        )

        with open(LOSS_LOG_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.8e}", f"{val_loss:.8e}", f"{val_t_mse_c2:.8e}"])

        if fold_log_path is not None:
            with open(fold_log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([epoch, f"{train_loss:.8e}", f"{val_loss:.8e}", f"{val_t_mse_c2:.8e}"])

        # 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(
                CHECKPOINT_DIR, f"best_model_epoch{epoch}_valloss{val_loss:.4e}.pt"
            )
            model_to_save = model.module if isinstance(model, nn.DataParallel) else model
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model_to_save.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "config": CONFIG,
                },
                ckpt_path,
            )
            print(f"Saved new best model to {ckpt_path}")

    total_time = time() - start_time
    print(f"Training finished in {total_time/60:.2f} minutes.")


if __name__ == "__main__":
    main()