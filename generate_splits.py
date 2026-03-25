import os
import glob
import re
import csv
import argparse
import pickle
import numpy as np
from config import CONFIG, DATA_DIR


def natural_sort_key(filename: str) -> int:
    nums = re.findall(r"\d+", filename)
    return int(nums[-1]) if nums else 0


def write_list(file_path: str, file_list):
    with open(file_path, "w", encoding="utf-8") as f:
        for filename in file_list:
            f.write(filename + "\n")


def _resolve_file_path(data_dir: str, rel_or_abs_path: str) -> str:
    return rel_or_abs_path if os.path.isabs(rel_or_abs_path) else os.path.join(data_dir, rel_or_abs_path)


def _safe_scalar_mean(arr) -> float:
    arr_np = np.asarray(arr, dtype=np.float32)
    if arr_np.size == 0:
        return float("nan")
    return float(np.nanmean(arr_np))


def _load_control_signature(data_dir: str, filename: str, precision: int = 3):
    path = _resolve_file_path(data_dir, filename)
    with open(path, "rb") as f:
        frame = pickle.load(f)

    ac_vel = _safe_scalar_mean(frame.get("AC_Vel_Set", np.nan))
    fan_vel = _safe_scalar_mean(frame.get("Fan_Vel_Set", np.nan))
    ac_temp = _safe_scalar_mean(frame.get("AC_Temp_Set", np.nan))

    return (
        round(ac_vel, precision),
        round(fan_vel, precision),
        round(ac_temp, precision),
    )


def build_control_segments(pkl_files, data_dir: str, precision: int = 3):
    signatures = [_load_control_signature(data_dir, f, precision=precision) for f in pkl_files]
    segments = []

    if not pkl_files:
        return segments

    seg_start = 0
    current_sig = signatures[0]
    for idx in range(1, len(pkl_files)):
        if signatures[idx] != current_sig:
            segments.append(
                {
                    "start_idx": seg_start,
                    "end_idx": idx - 1,
                    "start_step": natural_sort_key(pkl_files[seg_start]),
                    "end_step": natural_sort_key(pkl_files[idx - 1]),
                    "signature": current_sig,
                }
            )
            seg_start = idx
            current_sig = signatures[idx]

    segments.append(
        {
            "start_idx": seg_start,
            "end_idx": len(pkl_files) - 1,
            "start_step": natural_sort_key(pkl_files[seg_start]),
            "end_step": natural_sort_key(pkl_files[-1]),
            "signature": current_sig,
        }
    )

    return segments


def _align_val_end_to_segment_boundary(val_end_idx_exclusive: int, segments, n: int) -> int:
    if val_end_idx_exclusive >= n:
        return n

    probe_idx = max(0, val_end_idx_exclusive - 1)
    for seg in segments:
        if seg["start_idx"] <= probe_idx <= seg["end_idx"]:
            return min(n, seg["end_idx"] + 1)

    return min(n, val_end_idx_exclusive)


def _format_signature(sig) -> str:
    return f"AC_Vel={sig[0]},Fan_Vel={sig[1]},AC_Temp={sig[2]}"


def create_control_rolling_splits(
    pkl_files,
    input_steps,
    pred_steps,
    data_dir: str,
    num_folds=4,
    val_ratio=0.2,
    precision: int = 3,
):
    n = len(pkl_files)
    min_required = input_steps + pred_steps
    base_val_len = max(min_required + 1, int(n * val_ratio))

    if n < (base_val_len + min_required + 1):
        raise RuntimeError(
            f"Not enough files ({n}) for control rolling split. "
            f"Need at least {base_val_len + min_required + 1}."
        )

    segments = build_control_segments(pkl_files, data_dir=data_dir, precision=precision)
    if not segments:
        raise RuntimeError("No control segments were detected.")

    earliest_train_end = max(min_required + 1, int(n * 0.3))
    latest_train_end = n - base_val_len
    if latest_train_end <= earliest_train_end:
        earliest_train_end = max(min_required + 1, n - base_val_len - 1)

    boundary_candidates = [
        seg["start_idx"]
        for seg in segments
        if seg["start_idx"] > 0 and earliest_train_end <= seg["start_idx"] <= latest_train_end
    ]

    if not boundary_candidates:
        boundary_candidates = [latest_train_end]

    if num_folds <= 1:
        proposed_starts = [boundary_candidates[0]]
    else:
        if len(boundary_candidates) == 1:
            proposed_starts = [boundary_candidates[0]] * num_folds
        else:
            proposed_starts = []
            for i in range(num_folds):
                ratio = i / (num_folds - 1)
                idx = int(round(ratio * (len(boundary_candidates) - 1)))
                proposed_starts.append(boundary_candidates[idx])

    unique_starts = []
    for s in proposed_starts:
        if s not in unique_starts:
            unique_starts.append(s)

    unique_starts.sort()

    def _next_boundary_at_or_after(idx: int):
        for b in boundary_candidates:
            if b >= idx:
                return b
        return None

    folds = []
    prev_val_end = 0
    for fold_id, planned_start in enumerate(unique_starts):
        val_start = max(planned_start, prev_val_end)
        boundary_start = _next_boundary_at_or_after(val_start)
        if boundary_start is None:
            break
        val_start = boundary_start

        val_end = min(n, val_start + base_val_len)
        val_end = _align_val_end_to_segment_boundary(val_end, segments, n)

        if val_end - val_start < (min_required + 1):
            val_end = min(n, val_start + min_required + 1)

        train_files = pkl_files[:val_start]
        val_files = pkl_files[val_start:val_end]

        if len(train_files) < (min_required + 1) or len(val_files) < (min_required + 1):
            continue

        val_signatures = []
        for seg in segments:
            if seg["end_idx"] < val_start or seg["start_idx"] >= val_end:
                continue
            sig_text = _format_signature(seg["signature"])
            if sig_text not in val_signatures:
                val_signatures.append(sig_text)

        folds.append(
            {
                "fold": fold_id,
                "train_files": train_files,
                "val_files": val_files,
                "train_start_step": natural_sort_key(train_files[0]),
                "train_end_step": natural_sort_key(train_files[-1]),
                "val_start_step": natural_sort_key(val_files[0]),
                "val_end_step": natural_sort_key(val_files[-1]),
                "val_regime_count": len(val_signatures),
                "val_regimes": " | ".join(val_signatures),
            }
        )
        prev_val_end = val_end

    if not folds:
        raise RuntimeError("No valid control rolling folds generated.")

    return folds, segments


def write_control_change_points_report(report_path: str, segments):
    with open(report_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "segment_id",
            "start_step",
            "end_step",
            "ac_vel_set",
            "fan_vel_set",
            "ac_temp_set",
        ])
        for idx, seg in enumerate(segments):
            ac_vel, fan_vel, ac_temp = seg["signature"]
            writer.writerow([
                idx,
                seg["start_step"],
                seg["end_step"],
                ac_vel,
                fan_vel,
                ac_temp,
            ])


def main():
    parser = argparse.ArgumentParser(description="Generate control-aware rolling splits for time-series CFD data")
    parser.add_argument("--mode", choices=["control_rolling"], default="control_rolling", help="Only control_rolling is supported")
    parser.add_argument("--num_folds", type=int, default=4, help="Requested fold count for control_rolling")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation ratio for control_rolling")
    parser.add_argument("--active_fold", type=int, default=0, help="Which fold to copy as train_files.txt/val_files.txt")
    parser.add_argument("--control_precision", type=int, default=3, help="Rounding precision for control signatures")
    args = parser.parse_args()

    data_dir = DATA_DIR
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} does not exist.")
        return

    pkl_files = glob.glob(os.path.join(data_dir, "*.pkl"))
    pkl_files = [os.path.basename(f) for f in pkl_files]
    # 使用自然排序（按数字大小排序），确保时间步顺序正确
    pkl_files.sort(key=natural_sort_key)
    
    if not pkl_files:
        print("No .pkl files found.")
        return

    print(f"Found {len(pkl_files)} files.")

    input_steps = CONFIG.get("input_steps", 2)
    pred_steps = CONFIG.get("pred_steps", 1)

    folds, segments = create_control_rolling_splits(
        pkl_files=pkl_files,
        input_steps=input_steps,
        pred_steps=pred_steps,
        data_dir=data_dir,
        num_folds=args.num_folds,
        val_ratio=args.val_ratio,
        precision=args.control_precision,
    )
    report_name = "control_splits_report.csv"
    fold_prefix = "control_fold"

    splits_dir = os.path.join(data_dir, "splits")
    os.makedirs(splits_dir, exist_ok=True)

    stale_patterns = [
        os.path.join(splits_dir, f"{fold_prefix}_*_train_files.txt"),
        os.path.join(splits_dir, f"{fold_prefix}_*_val_files.txt"),
    ]
    for pattern in stale_patterns:
        for stale_file in glob.glob(pattern):
            try:
                os.remove(stale_file)
            except OSError:
                pass

    report_path = os.path.join(splits_dir, report_name)
    with open(report_path, "w", encoding="utf-8", newline="") as report_file:
        writer = csv.writer(report_file)
        writer.writerow([
            "fold",
            "train_count",
            "val_count",
            "train_start_step",
            "train_end_step",
            "val_start_step",
            "val_end_step",
            "val_regime_count",
            "val_regimes",
        ])

        for fold_info in folds:
            fold_id = fold_info["fold"]
            train_path = os.path.join(splits_dir, f"{fold_prefix}_{fold_id}_train_files.txt")
            val_path = os.path.join(splits_dir, f"{fold_prefix}_{fold_id}_val_files.txt")

            write_list(train_path, fold_info["train_files"])
            write_list(val_path, fold_info["val_files"])

            writer.writerow([
                fold_id,
                len(fold_info["train_files"]),
                len(fold_info["val_files"]),
                fold_info["train_start_step"],
                fold_info["train_end_step"],
                fold_info["val_start_step"],
                fold_info["val_end_step"],
                fold_info["val_regime_count"],
                fold_info["val_regimes"],
            ])

    control_points_report = os.path.join(splits_dir, "control_change_points.csv")
    write_control_change_points_report(control_points_report, segments)

    active_fold = max(0, min(args.active_fold, len(folds) - 1))
    active = folds[active_fold]

    train_list_path = os.path.join(data_dir, "train_files.txt")
    val_list_path = os.path.join(data_dir, "val_files.txt")
    write_list(train_list_path, active["train_files"])
    write_list(val_list_path, active["val_files"])

    mode_tag = "ControlRolling"
    print(f"[{mode_tag}] Generated {len(folds)} folds in {splits_dir}")
    print(f"[{mode_tag}] Report saved to {report_path}")
    print(f"[{mode_tag}] Control change points saved to {os.path.join(splits_dir, 'control_change_points.csv')}")
    print(f"[{mode_tag}] Active fold = {active_fold}")
    print(f"[{mode_tag}] Updated default train/val files: {len(active['train_files'])}/{len(active['val_files'])}")

if __name__ == "__main__":
    main()
