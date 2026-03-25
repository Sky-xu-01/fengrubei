import numpy as np
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data_pkl")
NPZ_FILE = os.path.join(DATA_DIR, "prediction_result.npz")

if not os.path.exists(NPZ_FILE):
    print(f"File not found: {NPZ_FILE}")
else:
    print(f"Loading {NPZ_FILE}...")
    data = np.load(NPZ_FILE)
    
    if 'targets' in data:
        targets = data['targets']
        print(f"Targets shape: {targets.shape}")
        print(f"Targets min: {targets.min()}")
        print(f"Targets max: {targets.max()}")
        print(f"Targets mean: {targets.mean()}")
        
        # Check a few values
        print("Sample values (flattened):", targets.flatten()[:10])
    else:
        print("No 'targets' found in file.")

    if 'predictions' in data:
        predictions = data['predictions']
        print(f"Predictions shape: {predictions.shape}")
        print(f"Predictions min: {predictions.min()}")
        print(f"Predictions max: {predictions.max()}")
        print(f"Predictions mean: {predictions.mean()}")
