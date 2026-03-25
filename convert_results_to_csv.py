import numpy as np
import pandas as pd
import os
import sys

# 设置路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data_pkl")
NPZ_FILE = os.path.join(DATA_DIR, "prediction_result.npz")
OUTPUT_CSV = os.path.join(BASE_DIR, "prediction_result_export.csv")

def main():
    if not os.path.exists(NPZ_FILE):
        print(f"Error: 找不到预测结果文件: {NPZ_FILE}")
        return

    print(f"Loading {NPZ_FILE}...")
    try:
        data = np.load(NPZ_FILE)
    except Exception as e:
        print(f"Error loading npz file: {e}")
        return

    # 查看文件包含的键
    print("Keys in npz file:", list(data.keys()))

    # 获取坐标信息
    # 优先尝试小写 x, y, z，如果是大写则自动适配
    if 'x' in data:
        X_grid = data['x']
        Y_grid = data['y']
        Z_grid = data['z']
    elif 'X' in data:
        X_grid = data['X']
        Y_grid = data['Y']
        Z_grid = data['Z']
    else:
        print("Error: 找不到坐标数据 (x/y/z)")
        return

    # 获取预测和真实值
    if 'predictions' not in data:
        print("Error: 找不到 predictions 数据")
        # 尝试兼容旧格式
        if 'pred_temps' in data:
            predictions = data['pred_temps']
            print("Using 'pred_temps' key")
        else:
            return
    else:
        predictions = data['predictions'] # Shape: (N_samples, T_pred, Z, Y, X)

    targets = data['targets'] if 'targets' in data else None # Shape: (N_samples, T_pred, Z, Y, X)

    print(f"Grid shape: {X_grid.shape}")
    print(f"Predictions shape: {predictions.shape}")

    # 展平坐标
    # 假设坐标网格是 (Z, Y, X) 或者 (X, Y, Z)，我们需要将其展平以便对应
    # 统一展平顺序
    x_flat = X_grid.flatten()
    y_flat = Y_grid.flatten()
    z_flat = Z_grid.flatten()

    all_rows = []

    # 遍历每个样本和时间步
    # predictions shape 通常是 5 维: (Sample, Time, Z, Y, X)
    # 如果是 4 维 (Sample, Z, Y, X) 或 (Time, Z, Y, X)，需要适配
    
    dims = predictions.ndim
    if dims == 5:
        n_samples, n_times, nz, ny, nx = predictions.shape
    elif dims == 4:
        # 假设是 (Sample, Z, Y, X) 且 Time=1
        n_samples, nz, ny, nx = predictions.shape
        n_times = 1
        predictions = predictions[:, np.newaxis, ...]
        if targets is not None:
             targets = targets[:, np.newaxis, ...]
    else:
        print(f"Error: Unexpected predictions dimension {dims}")
        return

    total_points = len(x_flat)
    if total_points != nz * ny * nx:
        print(f"Warning: Coordinate points count ({total_points}) != Data points count ({nz*ny*nx}). Shape mismatch?")
    
    print("Exporting data to list...")
    
    for s_idx in range(n_samples):
        for t_idx in range(n_times):
            
            pred_vol = predictions[s_idx, t_idx].flatten()
            
            # 创建基础 DataFrame 用于当前帧
            df_frame = pd.DataFrame({
                "Sample_ID": s_idx,
                "Time_Step": t_idx,
                "X": x_flat,
                "Y": y_flat,
                "Z": z_flat,
                "Predicted_T": pred_vol
            })
            
            if targets is not None:
                target_vol = targets[s_idx, t_idx].flatten()
                df_frame["True_T"] = target_vol
                df_frame["Error"] = df_frame["Predicted_T"] - df_frame["True_T"]
            
            all_rows.append(df_frame)

    if not all_rows:
        print("No data to export.")
        return

    print("Concatenating DataFrame...")
    final_df = pd.concat(all_rows, ignore_index=True)

    print(f"Saving to {OUTPUT_CSV}...")
    final_df.to_csv(OUTPUT_CSV, index=False)
    print("Done! View the file at:", OUTPUT_CSV)

if __name__ == "__main__":
    main()
