import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import glob
import csv
from pathlib import Path

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data_pkl")
DEFAULT_NPZ_FILE = os.path.join(DATA_DIR, "prediction_result.npz")
OUTPUT_DIR = os.path.join(BASE_DIR, "vis_results")

os.makedirs(OUTPUT_DIR, exist_ok=True)


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


def upsert_fold_t_mse(fold_id: int, t_mse: float):
    summary_csv = os.path.join(OUTPUT_DIR, "fold_t_mse_summary.csv")
    rows = []
    if os.path.exists(summary_csv):
        with open(summary_csv, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

    updated = False
    for row in rows:
        if int(row["fold"]) == int(fold_id):
            row["t_mse"] = f"{t_mse:.8e}"
            updated = True
            break

    if not updated:
        rows.append({"fold": str(int(fold_id)), "t_mse": f"{t_mse:.8e}"})

    rows.sort(key=lambda r: int(r["fold"]))
    with open(summary_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["fold", "t_mse"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def to_celsius_if_kelvin(arr: np.ndarray) -> np.ndarray:
    mean_val = float(np.nanmean(arr))
    if mean_val > 170.0:
        return arr - 273.15
    return arr

def visualize():
    parser = argparse.ArgumentParser(description="Visualize prediction results")
    parser.add_argument("--file", type=str, default=None, help="Path to .npz file")
    parser.add_argument("--fold_id", type=int, default=None, help="Optional fold id for per-fold outputs")
    args = parser.parse_args()

    active_fold_id = resolve_active_fold_id(args.fold_id)
    output_dir = OUTPUT_DIR
    if active_fold_id is not None:
        output_dir = os.path.join(OUTPUT_DIR, "folds", f"fold_{active_fold_id}")
        os.makedirs(output_dir, exist_ok=True)
        print(f"Active fold detected: fold_{active_fold_id}")

    if args.file:
        npz_file = args.file
    else:
        # Check for realtime_prediction.npz first
        realtime_file = os.path.join(DATA_DIR, "realtime_prediction.npz")
        if os.path.exists(realtime_file):
            print(f"Found realtime prediction: {realtime_file}")
            npz_file = realtime_file
        else:
            npz_file = DEFAULT_NPZ_FILE

    if not os.path.exists(npz_file):
        print(f"File not found: {npz_file}")
        return

    print(f"Loading {npz_file}...")
    data = np.load(npz_file)
    
    # New format: predictions (N, T, Z, Y, X), targets (M, T, Z, Y, X)
    predictions_6ch = None
    targets_6ch = None

    if 'predictions' in data:
        predictions = data['predictions']
        targets = data['targets'] if 'targets' in data else None
        start_indices = data['start_indices'] if 'start_indices' in data else None
        if 'predictions_6ch' in data and 'targets_6ch' in data:
            predictions_6ch = data['predictions_6ch']
            targets_6ch = data['targets_6ch']
    elif 'prediction' in data:
        # Realtime prediction format
        predictions = data['prediction']
        # Ensure it has batch dim if missing
        if predictions.ndim == 4: # (T, Z, Y, X)
             predictions = predictions[np.newaxis, ...]
        targets = None
        start_indices = [0]
    else:
        # Fallback to old format
        if 'pred_temps' in data:
            predictions = data['pred_temps'][np.newaxis, ...] # (1, T, Z, Y, X)
            targets = None
            start_indices = [0]
        else:
            print("Unknown data format")
            return

    print(f"Predictions shape: {predictions.shape}")
    if targets is not None:
        print(f"Targets shape: {targets.shape}")
        
        # Align shapes for evaluation (ignore predictions without ground truth)
        n_eval = min(len(predictions), len(targets))
        preds_eval = predictions[:n_eval]
        targets_eval = targets[:n_eval]
        
        # Calculate Metrics
        # MSE
        mse = np.mean((preds_eval - targets_eval) ** 2)
        # MAE
        mae = np.mean(np.abs(preds_eval - targets_eval))
        # RMSE
        rmse = np.sqrt(mse)
        
        print("-" * 30)
        print(f"Evaluation Metrics:")
        print(f"  MSE:  {mse:.6f}")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  MAE:  {mae:.6f}")
        if predictions_6ch is not None and targets_6ch is not None:
            var_names = ["T", "U", "V", "W", "K", "NUT"]
            for i, var_name in enumerate(var_names):
                mse_i = np.mean((predictions_6ch[:n_eval, :, i, ...] - targets_6ch[:n_eval, :, i, ...]) ** 2)
                rmse_i = np.sqrt(mse_i)
                mae_i = np.mean(np.abs(predictions_6ch[:n_eval, :, i, ...] - targets_6ch[:n_eval, :, i, ...]))
                print(f"  {var_name} -> MSE: {mse_i:.6f}, RMSE: {rmse_i:.6f}, MAE: {mae_i:.6f}")
        print("-" * 30)
        
        # Save metrics to file
        with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
            f.write(f"MSE: {mse:.6f}\n")
            f.write(f"RMSE: {rmse:.6f}\n")
            f.write(f"MAE: {mae:.6f}\n")
            if predictions_6ch is not None and targets_6ch is not None:
                var_names = ["T", "U", "V", "W", "K", "NUT"]
                for i, var_name in enumerate(var_names):
                    mse_i = np.mean((predictions_6ch[:n_eval, :, i, ...] - targets_6ch[:n_eval, :, i, ...]) ** 2)
                    rmse_i = np.sqrt(mse_i)
                    mae_i = np.mean(np.abs(predictions_6ch[:n_eval, :, i, ...] - targets_6ch[:n_eval, :, i, ...]))
                    f.write(f"{var_name}_MSE: {mse_i:.6f}\n")
                    f.write(f"{var_name}_RMSE: {rmse_i:.6f}\n")
                    f.write(f"{var_name}_MAE: {mae_i:.6f}\n")

        # 兼容旧路径
        with open(os.path.join(OUTPUT_DIR, "metrics.txt"), "w") as f:
            f.write(f"MSE: {mse:.6f}\n")
            f.write(f"RMSE: {rmse:.6f}\n")
            f.write(f"MAE: {mae:.6f}\n")
            if predictions_6ch is not None and targets_6ch is not None:
                var_names = ["T", "U", "V", "W", "K", "NUT"]
                for i, var_name in enumerate(var_names):
                    mse_i = np.mean((predictions_6ch[:n_eval, :, i, ...] - targets_6ch[:n_eval, :, i, ...]) ** 2)
                    rmse_i = np.sqrt(mse_i)
                    mae_i = np.mean(np.abs(predictions_6ch[:n_eval, :, i, ...] - targets_6ch[:n_eval, :, i, ...]))
                    f.write(f"{var_name}_MSE: {mse_i:.6f}\n")
                    f.write(f"{var_name}_RMSE: {rmse_i:.6f}\n")
                    f.write(f"{var_name}_MAE: {mae_i:.6f}\n")

        if active_fold_id is not None:
            upsert_fold_t_mse(active_fold_id, float(mse))
    
    # Check for AC parameters
    ac_info = ""
    if 'ac_temp' in data and 'ac_speed' in data:
        ac_temp = data['ac_temp']
        ac_speed = data['ac_speed']
        ac_info = f"AC Setting: {ac_temp}°C, {ac_speed} m/s"
        print(f"Found AC info: {ac_info}")

    # Check for coordinates
    extent_xy = None
    extent_xz = None
    extent_yz = None
    xlabel_suffix = " (index)"
    
    if 'x' in data and 'y' in data and 'z' in data:
        print("Found coordinates in file.")
        X = data['x'] # (Z, Y, X)
        Y = data['y']
        Z = data['z']
        
        # Assuming regular grid, get min/max
        x_min, x_max = X.min(), X.max()
        y_min, y_max = Y.min(), Y.max()
        z_min, z_max = Z.min(), Z.max()
        
        # Extent for imshow: [left, right, bottom, top]
        extent_xy = [x_min, x_max, y_min, y_max]
        extent_xz = [x_min, x_max, z_min, z_max]
        extent_yz = [y_min, y_max, z_min, z_max]
        
        xlabel_suffix = " (m)"

    # Visualize the first window
    window_idx = 0
    if len(predictions) == 0:
        print("No predictions found.")
        return

    pred_window = predictions[window_idx] # (T, Z, Y, X)
    target_window = targets[window_idx] if targets is not None and len(targets) > window_idx else None
    
    start_t = start_indices[window_idx] if start_indices is not None else 0
    
    # Visualize the first time step of this window
    t_idx = 0
    pred_vol = pred_window[t_idx]
    
    # Convert to Celsius for visualization
    pred_vol_c = to_celsius_if_kelvin(pred_vol)
    
    nz, ny, nx = pred_vol.shape
    z_slice = nz // 2
    y_slice = ny // 2
    x_slice = nx // 2
    
    # Determine vmin/vmax from Ground Truth if available, else from Prediction
    if target_window is not None:
        target_vol = target_window[t_idx]
        target_vol_c = to_celsius_if_kelvin(target_vol)
        vmin = target_vol_c.min()
        vmax = target_vol_c.max()
    else:
        target_vol_c = None
        vmin = pred_vol_c.min()
        vmax = pred_vol_c.max()
        
    print(f"Visualization Range (Celsius): vmin={vmin:.2f}, vmax={vmax:.2f}")

    # Plotting
    rows = 3 if target_window is not None else 1
    cols = 3
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)
        
    # Prediction Row
    # XY
    im1 = axes[0, 0].imshow(pred_vol_c[z_slice, :, :], origin='lower', aspect='auto', cmap='jet', extent=extent_xy, vmin=vmin, vmax=vmax)
    axes[0, 0].set_title(f'Pred XY (Z={z_slice}) T={start_t + t_idx}\nRange: [{pred_vol_c.min():.1f}, {pred_vol_c.max():.1f}] °C')
    axes[0, 0].set_xlabel(f'X{xlabel_suffix}')
    axes[0, 0].set_ylabel(f'Y{xlabel_suffix}')
    plt.colorbar(im1, ax=axes[0, 0], label='Temp (°C)')
    
    # XZ
    im2 = axes[0, 1].imshow(pred_vol_c[:, y_slice, :], origin='lower', aspect='auto', cmap='jet', extent=extent_xz, vmin=vmin, vmax=vmax)
    axes[0, 1].set_title(f'Pred XZ (Y={y_slice})\nRange: [{pred_vol_c.min():.1f}, {pred_vol_c.max():.1f}] °C')
    axes[0, 1].set_xlabel(f'X{xlabel_suffix}')
    axes[0, 1].set_ylabel(f'Z{xlabel_suffix}')
    plt.colorbar(im2, ax=axes[0, 1], label='Temp (°C)')
    
    # YZ
    im3 = axes[0, 2].imshow(pred_vol_c[:, :, x_slice], origin='lower', aspect='auto', cmap='jet', extent=extent_yz, vmin=vmin, vmax=vmax)
    axes[0, 2].set_title(f'Pred YZ (X={x_slice})\nRange: [{pred_vol_c.min():.1f}, {pred_vol_c.max():.1f}] °C')
    axes[0, 2].set_xlabel(f'Y{xlabel_suffix}')
    axes[0, 2].set_ylabel(f'Z{xlabel_suffix}')
    plt.colorbar(im3, ax=axes[0, 2], label='Temp (°C)')
    
    # Target Row
    if target_window is not None:
        # XY
        im4 = axes[1, 0].imshow(target_vol_c[z_slice, :, :], origin='lower', aspect='auto', cmap='jet', extent=extent_xy, vmin=vmin, vmax=vmax)
        axes[1, 0].set_title(f'GT XY (Z={z_slice})\nRange: [{target_vol_c.min():.1f}, {target_vol_c.max():.1f}] °C')
        axes[1, 0].set_xlabel(f'X{xlabel_suffix}')
        axes[1, 0].set_ylabel(f'Y{xlabel_suffix}')
        plt.colorbar(im4, ax=axes[1, 0], label='Temp (°C)')
        
        # XZ
        im5 = axes[1, 1].imshow(target_vol_c[:, y_slice, :], origin='lower', aspect='auto', cmap='jet', extent=extent_xz, vmin=vmin, vmax=vmax)
        axes[1, 1].set_title(f'GT XZ (Y={y_slice})\nRange: [{target_vol_c.min():.1f}, {target_vol_c.max():.1f}] °C')
        axes[1, 1].set_xlabel(f'X{xlabel_suffix}')
        axes[1, 1].set_ylabel(f'Z{xlabel_suffix}')
        plt.colorbar(im5, ax=axes[1, 1], label='Temp (°C)')
        
        # YZ
        im6 = axes[1, 2].imshow(target_vol_c[:, :, x_slice], origin='lower', aspect='auto', cmap='jet', extent=extent_yz, vmin=vmin, vmax=vmax)
        axes[1, 2].set_title(f'GT YZ (X={x_slice})\nRange: [{target_vol_c.min():.1f}, {target_vol_c.max():.1f}] °C')
        axes[1, 2].set_xlabel(f'Y{xlabel_suffix}')
        axes[1, 2].set_ylabel(f'Z{xlabel_suffix}')
        plt.colorbar(im6, ax=axes[1, 2], label='Temp (°C)')

        # Error Row (absolute error in Celsius)
        error_vol_c = np.abs(pred_vol_c - target_vol_c)
        err_vmin = 0.0
        err_vmax = error_vol_c.max()

        im7 = axes[2, 0].imshow(error_vol_c[z_slice, :, :], origin='lower', aspect='auto', cmap='magma', extent=extent_xy, vmin=err_vmin, vmax=err_vmax)
        axes[2, 0].set_title(f'Error XY (Z={z_slice})\nRange: [0.0, {error_vol_c.max():.1f}] °C')
        axes[2, 0].set_xlabel(f'X{xlabel_suffix}')
        axes[2, 0].set_ylabel(f'Y{xlabel_suffix}')
        plt.colorbar(im7, ax=axes[2, 0], label='Abs Error (°C)')

        im8 = axes[2, 1].imshow(error_vol_c[:, y_slice, :], origin='lower', aspect='auto', cmap='magma', extent=extent_xz, vmin=err_vmin, vmax=err_vmax)
        axes[2, 1].set_title(f'Error XZ (Y={y_slice})\nRange: [0.0, {error_vol_c.max():.1f}] °C')
        axes[2, 1].set_xlabel(f'X{xlabel_suffix}')
        axes[2, 1].set_ylabel(f'Z{xlabel_suffix}')
        plt.colorbar(im8, ax=axes[2, 1], label='Abs Error (°C)')

        im9 = axes[2, 2].imshow(error_vol_c[:, :, x_slice], origin='lower', aspect='auto', cmap='magma', extent=extent_yz, vmin=err_vmin, vmax=err_vmax)
        axes[2, 2].set_title(f'Error YZ (X={x_slice})\nRange: [0.0, {error_vol_c.max():.1f}] °C')
        axes[2, 2].set_xlabel(f'Y{xlabel_suffix}')
        axes[2, 2].set_ylabel(f'Z{xlabel_suffix}')
        plt.colorbar(im9, ax=axes[2, 2], label='Abs Error (°C)')
        
    # Add Suptitle
    title_text = f'Prediction vs Ground Truth'
    if target_window is not None:
        title_text += f' (MSE: {mse:.4f})'
    if ac_info:
        title_text += f'\n{ac_info}'
        
    plt.suptitle(title_text, fontsize=16)
        
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"pred_vs_gt_window{window_idx}_t{t_idx}.png")
    plt.savefig(output_path, dpi=150)
    print(f"Saved visualization to {output_path}")
    save_path = os.path.join(output_dir, "prediction_comparison.png")
    plt.savefig(save_path, dpi=150)
    print(f"Saved visualization to {save_path}")

    # 兼容旧路径
    legacy_save_path = os.path.join(OUTPUT_DIR, "prediction_comparison.png")
    plt.savefig(legacy_save_path, dpi=150)
    plt.close(fig)

if __name__ == "__main__":
    visualize()
