import csv
import os
import glob
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from config import DATA_DIR

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(BASE_DIR, "vis_results", "train_val_loss.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "vis_results", "train_val_loss.png")


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
            stem = Path(val_path).stem
            parts = stem.split("_")
            if len(parts) >= 3 and parts[2].isdigit():
                return int(parts[2])
    return None


def choose_paths(input_csv=None, output_png=None, fold_id=None):
    if input_csv and output_png:
        return input_csv, output_png

    if input_csv and not output_png:
        base, _ = os.path.splitext(input_csv)
        return input_csv, base + ".png"

    resolved_fold = resolve_active_fold_id(fold_id)
    if resolved_fold is not None:
        fold_dir = os.path.join(BASE_DIR, "vis_results", "folds", f"fold_{resolved_fold}")
        os.makedirs(fold_dir, exist_ok=True)
        fold_log = os.path.join(fold_dir, "train_val_loss.csv")
        fold_png = os.path.join(fold_dir, "train_val_loss.png")
        if os.path.exists(fold_log):
            return fold_log, (output_png if output_png else fold_png)

    return (input_csv if input_csv else LOG_PATH), (output_png if output_png else OUTPUT_PATH)


def main():
    parser = argparse.ArgumentParser(description="Plot training/validation MSE curve")
    parser.add_argument("--file", type=str, default=None, help="Path to loss csv")
    parser.add_argument("--output", type=str, default=None, help="Output png path")
    parser.add_argument("--fold_id", type=int, default=None, help="Optional fold id for fold-specific plotting")
    args = parser.parse_args()

    log_path, output_path = choose_paths(args.file, args.output, args.fold_id)

    if not os.path.exists(log_path):
        print(f"Loss log not found: {log_path}")
        print("Please run training first to generate the log.")
        return

    rows = []

    with open(log_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "epoch": int(row["epoch"]),
                    "train_loss": float(row["train_loss"]),
                    "val_loss": float(row["val_loss"]),
                    "val_t_mse_c2": float(row["val_t_mse_c2"]) if row.get("val_t_mse_c2") not in (None, "") else None,
                }
            )

    if not rows:
        print("Loss log is empty.")
        return

    raw_epochs = [r["epoch"] for r in rows]
    non_monotonic = any(raw_epochs[i] < raw_epochs[i - 1] for i in range(1, len(raw_epochs)))

    # 保留每个 epoch 最后一次记录（适配 resume / 多次训练写入）
    last_by_epoch = {}
    for r in rows:
        last_by_epoch[r["epoch"]] = r

    epochs = sorted(last_by_epoch.keys())
    train_losses = [last_by_epoch[e]["train_loss"] for e in epochs]
    val_losses = [last_by_epoch[e]["val_loss"] for e in epochs]
    has_val_t_curve = all(last_by_epoch[e].get("val_t_mse_c2") is not None for e in epochs)
    val_t_mse_c2 = [last_by_epoch[e]["val_t_mse_c2"] for e in epochs] if has_val_t_curve else []

    if non_monotonic or len(rows) != len(epochs):
        print(
            "Warning: detected unordered or duplicated epochs in train_val_loss.csv; "
            "plot uses sorted epochs with last occurrence per epoch."
        )

    if has_val_t_curve:
        fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

        axes[0].plot(epochs, train_losses, label="Train", linewidth=2)
        axes[0].plot(epochs, val_losses, label="Val", linewidth=2)
        axes[0].set_ylabel("Loss (normalized MSE)")
        axes[0].set_title("Training/Validation Loss")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].plot(epochs, val_t_mse_c2, color="tab:red", label="Val_T_MSE (℃²)", linewidth=2)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Val_T_MSE (℃²)")
        axes[1].set_title("Validation Temperature MSE")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        plt.tight_layout()
    else:
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_losses, label="Train", linewidth=2)
        plt.plot(epochs, val_losses, label="Val", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE)")
        plt.title("Training/Validation Loss")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        print("Warning: val_t_mse_c2 column not found in log; only train/val normalized loss is plotted.")
    plt.savefig(output_path, dpi=150)
    print(f"Saved loss curve to {output_path}")


if __name__ == "__main__":
    main()
