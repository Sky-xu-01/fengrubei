import os
import pickle
import numpy as np
from config import CONFIG, DATA_DIR

def compute_global_stats():
    train_file_list = CONFIG["train_file_list"]
    
    with open(train_file_list, "r") as f:
        files = [line.strip() for line in f if line.strip()]
        
    print(f"Found {len(files)} training files.")
    
    # Accumulators for new keys
    keys = ["T", "U", "V", "W", "K", "NUT", "Z"]
    sums = {k: 0.0 for k in keys}
    sq_sums = {k: 0.0 for k in keys}
    counts = {k: 0 for k in keys}
    
    for filename in files:
        file_path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found.")
            continue
            
        print(f"Processing {filename}...")
        with open(file_path, "rb") as f:
            data = pickle.load(f)
            
        for key in keys:
            if key in data:
                arr = data[key]
                # Filter out NaNs
                arr_flat = arr.flatten()
                arr_flat = arr_flat[~np.isnan(arr_flat)]
                
                if arr_flat.size > 0:
                    sums[key] += np.sum(arr_flat)
                    sq_sums[key] += np.sum(arr_flat ** 2)
                    counts[key] += arr_flat.size
                
    stats = {}
    for key in sums:
        N = counts[key]
        if N > 0:
            mean = sums[key] / N
            variance = (sq_sums[key] / N) - (mean ** 2)
            std = np.sqrt(max(0, variance))
            stats[key] = {"mean": float(mean), "std": float(std)}
            print(f"{key}: mean={mean:.4f}, std={std:.4f}")
        else:
            print(f"Warning: No valid data for {key}")
            
    # Save stats to a file
    output_path = os.path.join(os.path.dirname(train_file_list), "normalization_stats.json")
    import json
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=4)
    print(f"Saved stats to {output_path}")

if __name__ == "__main__":
    compute_global_stats()
