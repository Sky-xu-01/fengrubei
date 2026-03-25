import os
import glob
import importlib
import re
import numpy as np
import pickle
from config import CONFIG

try:
    import torch
except Exception:
    torch = None


def _setup_windows_cuda_dll_paths():
    if os.name != "nt" or torch is None:
        return

    try:
        torch_lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.isdir(torch_lib_dir):
            os.environ["PATH"] = torch_lib_dir + os.pathsep + os.environ.get("PATH", "")
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(torch_lib_dir)
    except Exception:
        pass


_setup_windows_cuda_dll_paths()

try:
    cp = importlib.import_module("cupy")
    cupyx_interp = importlib.import_module("cupyx.scipy.interpolate")
    cupy_linear_nd = getattr(cupyx_interp, "LinearNDInterpolator")
    cupy_nearest_nd = getattr(cupyx_interp, "NearestNDInterpolator")
except Exception as e:
    raise RuntimeError(f"CuPy/CuPyX import failed, cannot run GPU preprocessing: {e}")

if torch is None or not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available in current environment, cannot run convert_data_gpu.py")

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 修改这里改变数据来源路径
INPUT_DATA_DIR = os.path.join(BASE_DIR, "fluent_output_2")

OUTPUT_DIR = os.path.join(BASE_DIR, "data_pkl")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def _extract_time_step_from_filename(filename: str) -> float:
    nums = re.findall(r"\d+", filename)
    if not nums:
        return 0.0
    return float(nums[-1])


def _weighted_avg(temp: np.ndarray, vol: np.ndarray, mask: np.ndarray) -> float:
    valid = mask & np.isfinite(temp) & np.isfinite(vol) & (vol > 0)
    if np.any(valid):
        vol_sum = float(np.sum(vol[valid]))
        if vol_sum > 0:
            return float(np.sum(temp[valid] * vol[valid]) / vol_sum)
    fallback = temp[np.isfinite(temp)]
    if fallback.size > 0:
        return float(np.mean(fallback))
    return 0.0


def _resolve_fluid_mask(cell_zone: np.ndarray) -> np.ndarray:
    udf_cfg = CONFIG.get("udf_control", {})
    fluid_zone_ids = udf_cfg.get("fluid_zone_ids", [])
    if fluid_zone_ids:
        mask = np.zeros_like(cell_zone, dtype=bool)
        for zone_id in fluid_zone_ids:
            mask |= (cell_zone == float(zone_id))
        return mask
    return cell_zone > 0


def _add_udf_control_channels(final_data: dict, step_value: float, max_step_value: float) -> None:
    udf_cfg = CONFIG.get("udf_control", {})

    T = np.asarray(final_data.get("T"), dtype=np.float32)
    if T.size == 0:
        return

    VOL = np.asarray(final_data.get("VOL", np.ones_like(T)), dtype=np.float32)
    CELL_ZONE = np.asarray(final_data.get("CELL_ZONE", np.ones_like(T)), dtype=np.float32)
    fluid_mask = _resolve_fluid_mask(CELL_ZONE)
    valid_mask = np.isfinite(T)
    solid_mask = valid_mask & (~fluid_mask)
    fluid_mask = valid_mask & fluid_mask

    fluid_avg_temp = _weighted_avg(T, VOL, fluid_mask)
    if np.any(solid_mask):
        solid_avg_temp = _weighted_avg(T, VOL, solid_mask)
    else:
        solid_avg_temp = fluid_avg_temp

    ac_t_low = float(udf_cfg.get("ac_t_low", 297.15))
    ac_t_mid = float(udf_cfg.get("ac_t_mid", 299.15))
    ac_t_high = float(udf_cfg.get("ac_t_high", 302.15))
    ac_vel_low = float(udf_cfg.get("ac_vel_low", 1.5))
    ac_vel_mid = float(udf_cfg.get("ac_vel_mid", 3.0))
    ac_vel_high = float(udf_cfg.get("ac_vel_high", 5.0))
    fan_t_threshold = float(udf_cfg.get("fan_t_threshold", 299.15))
    fan_vel_low = float(udf_cfg.get("fan_vel_low", 3.0))
    fan_vel_high = float(udf_cfg.get("fan_vel_high", 5.0))
    ac_outlet_temp = float(udf_cfg.get("ac_outlet_temp", 293.15))

    if fluid_avg_temp >= ac_t_low and fluid_avg_temp < ac_t_mid:
        ac_vel_set = ac_vel_low
    elif fluid_avg_temp >= ac_t_mid and fluid_avg_temp < ac_t_high:
        ac_vel_set = ac_vel_mid
    elif fluid_avg_temp >= ac_t_high:
        ac_vel_set = ac_vel_high
    else:
        ac_vel_set = 0.0

    fan_vel_set = fan_vel_low if fluid_avg_temp <= fan_t_threshold else fan_vel_high
    fan_temp_set = fluid_avg_temp

    time_norm = step_value / max(max_step_value, 1.0)
    base = np.ones_like(T, dtype=np.float32)

    final_data["Current_Time"] = base * np.float32(step_value)
    final_data["Current_Time_Norm"] = base * np.float32(time_norm)
    final_data["Fluid_Avg_T"] = base * np.float32(fluid_avg_temp)
    final_data["Solid_Avg_T"] = base * np.float32(solid_avg_temp)
    final_data["AC_Vel_Set"] = base * np.float32(ac_vel_set)
    final_data["Fan_Vel_Set"] = base * np.float32(fan_vel_set)
    final_data["AC_Temp_Set"] = base * np.float32(ac_outlet_temp)
    final_data["Fan_Temp_Set"] = base * np.float32(fan_temp_set)


def read_fluent_file(file_path):
    """Reads data from Fluent exported file (CSV/Ascii format)."""
    mapping = CONFIG["csv_mapping"]

    q_src_csv_name = mapping.get("user-volumetric-energy-source", mapping.get("user-energy-source"))

    internal_mapping = {
        mapping["static-temperature"]: "T",
        mapping["x-velocity"]: "U",
        mapping["y-velocity"]: "V",
        mapping["z-velocity"]: "W",
        mapping["turb-kinetic-energy"]: "K",
        mapping["turb-viscosity"]: "NUT",
        mapping["cell-volume"]: "VOL",
        mapping["cell-zone"]: "CELL_ZONE"
    }

    if q_src_csv_name is not None:
        internal_mapping[q_src_csv_name] = "Q_SRC"
    else:
        print("  Warning: Q_SRC mapping key not found in CONFIG['csv_mapping']; skipping Q_SRC channel.")

    try:
        import pandas as pd
        df = pd.read_csv(file_path, skipinitialspace=True)
        df.columns = [c.strip() for c in df.columns]

        coords = df[[mapping["x-coordinate"], mapping["y-coordinate"], mapping["z-coordinate"]]].values

        data = {}
        for csv_name, internal_name in internal_mapping.items():
            if csv_name in df.columns:
                data[internal_name] = df[csv_name].values
            else:
                print(f"  Warning: Column {csv_name} not found in {file_path}")

        return coords, data
    except Exception as e:
        print(f"  Error reading {file_path}: {e}")
        return None, None


def interpolate_to_grid(points, values_dict, nx, ny, nz):
    """GPU interpolation: try linear, fallback to nearest on GPU."""
    points_gpu = cp.asarray(points)

    x = points_gpu[:, 0]
    y = points_gpu[:, 1]
    z = points_gpu[:, 2]

    xmin, xmax = float(cp.min(x)), float(cp.max(x))
    ymin, ymax = float(cp.min(y)), float(cp.max(y))
    zmin, zmax = float(cp.min(z)), float(cp.max(z))

    grid_x, grid_y, grid_z = cp.mgrid[
        xmin:xmax:complex(nx),
        ymin:ymax:complex(ny),
        zmin:zmax:complex(nz)
    ]

    interpolated_data = {
        "X": cp.asnumpy(grid_x),
        "Y": cp.asnumpy(grid_y),
        "Z": cp.asnumpy(grid_z),
    }

    grid_points = cp.stack([grid_x, grid_y, grid_z], axis=-1)

    for name, values in values_dict.items():
        print(f"    Interpolating {name}... [GPU]")
        values_gpu = cp.asarray(values)

        try:
            linear_interp = cupy_linear_nd(points_gpu, values_gpu)
            grid_values = linear_interp(grid_points)
            interp_mode = "linear"
        except Exception as e:
            nearest_interp = cupy_nearest_nd(points_gpu, values_gpu)
            grid_values = nearest_interp(grid_points)
            interp_mode = "nearest"
            print(f"      GPU linear unavailable for {name} ({e}); switched to GPU nearest.")

        mask = cp.isnan(grid_values)
        if interp_mode == "linear" and bool(cp.any(mask)):
            nearest_interp = cupy_nearest_nd(points_gpu, values_gpu)
            grid_values_nearest = nearest_interp(grid_points)
            grid_values[mask] = grid_values_nearest[mask]

        interpolated_data[name] = cp.asnumpy(grid_values)

    return interpolated_data


def main():
    data_files = glob.glob(os.path.join(INPUT_DATA_DIR, "FFF*"))
    if not data_files:
        print(f"No FFF* files found in {INPUT_DATA_DIR}")
        return

    data_files.sort(key=lambda x: int(re.findall(r'\d+', os.path.basename(x))[-1]) if re.findall(r'\d+', os.path.basename(x)) else 0)
    file_steps = {os.path.basename(path): _extract_time_step_from_filename(os.path.basename(path)) for path in data_files}
    max_step = max(file_steps.values()) if file_steps else 1.0

    print(f"Found {len(data_files)} files to process.")

    nx, ny, nz = CONFIG['nx'], CONFIG['ny'], CONFIG['nz']
    print(f"Target Grid: {nx}x{ny}x{nz}")
    print("Interpolation backend: GPU (CuPy)")

    for d_file in data_files:
        filename = os.path.basename(d_file)
        print(f"Processing {filename}...")

        try:
            coords, data = read_fluent_file(d_file)
            pkl_name = f"{filename}.pkl"

            if not data:
                print("  No data found. Skipping.")
                continue

            interp_results = interpolate_to_grid(coords, data, nx, ny, nz)

            final_data = {}
            for k, v in interp_results.items():
                final_data[k] = v.transpose(2, 1, 0)

            T_grid = final_data["T"]
            mask = (~np.isnan(T_grid)).astype(np.float32)
            final_data["mask"] = mask

            if "CELL_ZONE" in final_data:
                final_data["Cell_Zone_Mask"] = (final_data["CELL_ZONE"] > 0).astype(np.float32)
            else:
                final_data["Cell_Zone_Mask"] = mask

            final_data["Z_Coord"] = final_data["Z"]
            final_data["Wall_Dist"] = np.zeros_like(mask)
            final_data["Inlet_Mask"] = np.zeros_like(mask)

            step_value = file_steps.get(filename, 0.0)
            _add_udf_control_channels(final_data, step_value=step_value, max_step_value=max_step)

            pkl_path = os.path.join(OUTPUT_DIR, pkl_name)
            with open(pkl_path, 'wb') as f:
                pickle.dump(final_data, f)

            print(f"  Saved to {pkl_name}")

        except Exception as e:
            print(f"  Error processing {filename}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
