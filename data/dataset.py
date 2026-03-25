import json
import os
import pickle
from typing import List, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from config import CONFIG, DATA_DIR


def _read_file_list(file_list: Union[str, Sequence[str]]) -> List[str]:
    if isinstance(file_list, (list, tuple)):
        return [str(x).strip() for x in file_list if str(x).strip()]

    if isinstance(file_list, str):
        if os.path.isfile(file_list):
            with open(file_list, "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]
        return [file_list.strip()] if file_list.strip() else []

    return []


def _load_norm_stats() -> dict:
    candidate_paths = [
        CONFIG.get("stats_file", ""),
        os.path.join(DATA_DIR, "normalization_stats.json"),
    ]

    for path in candidate_paths:
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            stats = {}
            for key, value in raw.items():
                if isinstance(value, dict):
                    stats[key] = (float(value.get("mean", 0.0)), float(value.get("std", 1.0)))
                elif isinstance(value, (list, tuple)) and len(value) == 2:
                    stats[key] = (float(value[0]), float(value[1]))
            return stats

    return {}


class LabTemp3DDataset(Dataset):
    def __init__(
        self,
        file_list: Union[str, Sequence[str]],
        input_steps: int,
        pred_steps: int,
        in_channels_per_step: int,
        dt: float = 2.0,
        norm_stats: dict = None,
        ac_temp: float = None,
    ):
        self.file_list = _read_file_list(file_list)
        self.input_steps = int(input_steps)
        self.pred_steps = int(pred_steps)
        self.in_channels_per_step = int(in_channels_per_step)
        self.dt = float(dt)
        self.norm_stats = norm_stats if norm_stats is not None else _load_norm_stats()
        self.ac_temp = ac_temp

        self.total_window = self.input_steps + self.pred_steps
        self.sample_starts = list(range(max(0, len(self.file_list) - self.total_window + 1)))

    def __len__(self) -> int:
        return len(self.sample_starts)

    def _resolve_file_path(self, rel_or_abs_path: str) -> str:
        return rel_or_abs_path if os.path.isabs(rel_or_abs_path) else os.path.join(DATA_DIR, rel_or_abs_path)

    def _load_frame(self, file_index: int) -> dict:
        path = self._resolve_file_path(self.file_list[file_index])
        with open(path, "rb") as f:
            return pickle.load(f)

    def _normalize(self, arr: np.ndarray, key: str) -> np.ndarray:
        if key not in self.norm_stats:
            return arr.astype(np.float32)
        mean, std = self.norm_stats[key]
        return ((arr - mean) / (std + CONFIG["norm_eps"])) .astype(np.float32)

    def _get_or_zeros(self, frame: dict, key: str, shape_ref: np.ndarray) -> np.ndarray:
        if key in frame:
            return np.asarray(frame[key], dtype=np.float32)
        return np.zeros_like(shape_ref, dtype=np.float32)

    def _get_cell_zone_mask(self, frame: dict, shape_ref: np.ndarray) -> np.ndarray:
        for key in ["Cell_Zone_Mask", "CELL_ZONE", "cell-zone", "mask"]:
            if key in frame:
                return (np.asarray(frame[key], dtype=np.float32) > 0.5).astype(np.float32)
        return np.ones_like(shape_ref, dtype=np.float32)

    def _get_volume_field(self, frame: dict, shape_ref: np.ndarray) -> np.ndarray:
        for key in ["VOL", "cell-volume", "Cell_Volume"]:
            if key in frame:
                return np.asarray(frame[key], dtype=np.float32)
        return np.ones_like(shape_ref, dtype=np.float32)

    def _build_global_avg_t_channel(
        self,
        temp_field: np.ndarray,
        cell_zone_mask: np.ndarray,
        volume_field: np.ndarray,
    ) -> np.ndarray:
        fluid_mask = cell_zone_mask > 0.5

        if np.any(fluid_mask):
            vol_valid_mask = fluid_mask & np.isfinite(volume_field) & (volume_field > 0)
            if np.any(vol_valid_mask):
                vol_sum = float(np.sum(volume_field[vol_valid_mask]))
                if vol_sum > 0:
                    avg_t = float(np.sum(temp_field[vol_valid_mask] * volume_field[vol_valid_mask]) / vol_sum)
                else:
                    avg_t = float(np.mean(temp_field[fluid_mask]))
            else:
                avg_t = float(np.mean(temp_field[fluid_mask]))
        else:
            avg_t = float(np.mean(temp_field))
        return np.full_like(temp_field, fill_value=avg_t, dtype=np.float32)

    def _apply_ac_temp_if_needed(self, t_channel: np.ndarray, inlet_mask: np.ndarray) -> np.ndarray:
        if self.ac_temp is None:
            return t_channel

        t_modified = t_channel.copy()
        ac_temp_norm = self._normalize(np.asarray(self.ac_temp, dtype=np.float32), "T")

        inlet_binary = inlet_mask > 0.5
        if np.any(inlet_binary):
            t_modified[inlet_binary] = ac_temp_norm
            return t_modified

        z_s, z_e, y_s, y_e, x_s, x_e = CONFIG.get("ac_region_indices", [0, 1, 0, 1, 0, 1])
        t_modified[z_s:z_e, y_s:y_e, x_s:x_e] = ac_temp_norm
        return t_modified

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = self.sample_starts[idx]

        input_frames = [self._load_frame(start + i) for i in range(self.input_steps)]
        target_frames = [self._load_frame(start + self.input_steps + i) for i in range(self.pred_steps)]

        input_channels = []

        for frame in input_frames:
            T = np.asarray(frame["T"], dtype=np.float32)
            U = self._get_or_zeros(frame, "U", T)
            V = self._get_or_zeros(frame, "V", T)
            W = self._get_or_zeros(frame, "W", T)
            K = self._get_or_zeros(frame, "K", T)
            NUT = self._get_or_zeros(frame, "NUT", T)
            Q_SRC = self._get_or_zeros(frame, "Q_SRC", T)

            cell_zone_mask = self._get_cell_zone_mask(frame, T)
            volume_field = self._get_volume_field(frame, T)
            global_avg_t = self._build_global_avg_t_channel(T, cell_zone_mask, volume_field)

            Wall_Dist = self._get_or_zeros(frame, "Wall_Dist", T)
            Inlet_Mask = self._get_or_zeros(frame, "Inlet_Mask", T)
            Z_Coord = np.asarray(frame.get("Z_Coord", frame.get("Z", np.zeros_like(T))), dtype=np.float32)
            Time_Step = np.full_like(T, fill_value=self.dt / max(CONFIG.get("max_dt", 1.0), 1e-6), dtype=np.float32)
            Current_Time_Norm = self._get_or_zeros(frame, "Current_Time_Norm", Time_Step)
            Fluid_Avg_T = self._get_or_zeros(frame, "Fluid_Avg_T", T)
            Solid_Avg_T = self._get_or_zeros(frame, "Solid_Avg_T", T)
            AC_Vel_Set = self._get_or_zeros(frame, "AC_Vel_Set", T)
            Fan_Vel_Set = self._get_or_zeros(frame, "Fan_Vel_Set", T)
            AC_Temp_Set = self._get_or_zeros(frame, "AC_Temp_Set", T)
            Fan_Temp_Set = self._get_or_zeros(frame, "Fan_Temp_Set", T)

            T_norm = self._normalize(T, "T")
            U_norm = self._normalize(U, "U")
            V_norm = self._normalize(V, "V")
            W_norm = self._normalize(W, "W")
            K_norm = self._normalize(K, "K")
            NUT_norm = self._normalize(NUT, "NUT")
            Q_SRC_norm = self._normalize(Q_SRC, "Q_SRC")
            Global_Avg_T_norm = self._normalize(global_avg_t, "T")
            Z_Coord_norm = self._normalize(Z_Coord, "Z")
            Fluid_Avg_T_norm = self._normalize(Fluid_Avg_T, "T")
            Solid_Avg_T_norm = self._normalize(Solid_Avg_T, "T")
            AC_Vel_Set_norm = self._normalize(AC_Vel_Set, "U")
            Fan_Vel_Set_norm = self._normalize(Fan_Vel_Set, "U")
            AC_Temp_Set_norm = self._normalize(AC_Temp_Set, "T")
            Fan_Temp_Set_norm = self._normalize(Fan_Temp_Set, "T")

            T_norm = self._apply_ac_temp_if_needed(T_norm, Inlet_Mask)

            channels_this_step = [
                T_norm,
                U_norm,
                V_norm,
                W_norm,
                K_norm,
                NUT_norm,
                Q_SRC_norm,
                Global_Avg_T_norm,
                Wall_Dist.astype(np.float32),
                Inlet_Mask.astype(np.float32),
                Z_Coord_norm,
                Time_Step,
                cell_zone_mask.astype(np.float32),
                Current_Time_Norm.astype(np.float32),
                Fluid_Avg_T_norm,
                Solid_Avg_T_norm,
                AC_Vel_Set_norm,
                Fan_Vel_Set_norm,
                AC_Temp_Set_norm,
                Fan_Temp_Set_norm,
            ]

            if len(channels_this_step) != self.in_channels_per_step:
                raise ValueError(
                    f"in_channels_per_step mismatch: expected {self.in_channels_per_step}, "
                    f"got {len(channels_this_step)}"
                )

            input_channels.extend(channels_this_step)

        x = np.stack(input_channels, axis=0).astype(np.float32)

        target_channels = []
        for frame in target_frames:
            T = np.asarray(frame["T"], dtype=np.float32)
            U = self._get_or_zeros(frame, "U", T)
            V = self._get_or_zeros(frame, "V", T)
            W = self._get_or_zeros(frame, "W", T)
            K = self._get_or_zeros(frame, "K", T)
            NUT = self._get_or_zeros(frame, "NUT", T)

            target_channels.extend(
                [
                    self._normalize(T, "T"),
                    self._normalize(U, "U"),
                    self._normalize(V, "V"),
                    self._normalize(W, "W"),
                    self._normalize(K, "K"),
                    self._normalize(NUT, "NUT"),
                ]
            )

        y = np.stack(target_channels, axis=0).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)


def create_dataloaders() -> Tuple[DataLoader, DataLoader]:
    train_datasets_cfg = CONFIG.get("train_datasets", [])
    val_datasets_cfg = CONFIG.get("val_datasets", [])

    if not train_datasets_cfg:
        train_datasets_cfg = [{"file_list": CONFIG["train_file_list"], "dt": 2.0}]
    if not val_datasets_cfg:
        val_datasets_cfg = [{"file_list": CONFIG["val_file_list"], "dt": 2.0}]

    norm_stats = _load_norm_stats()

    train_sets = []
    for ds_cfg in train_datasets_cfg:
        ds = LabTemp3DDataset(
            file_list=ds_cfg["file_list"],
            input_steps=CONFIG["input_steps"],
            pred_steps=CONFIG["pred_steps"],
            in_channels_per_step=CONFIG["in_channels_per_step"],
            dt=float(ds_cfg.get("dt", 2.0)),
            norm_stats=norm_stats,
        )
        if len(ds) > 0:
            train_sets.append(ds)

    val_sets = []
    for ds_cfg in val_datasets_cfg:
        ds = LabTemp3DDataset(
            file_list=ds_cfg["file_list"],
            input_steps=CONFIG["input_steps"],
            pred_steps=CONFIG["pred_steps"],
            in_channels_per_step=CONFIG["in_channels_per_step"],
            dt=float(ds_cfg.get("dt", 2.0)),
            norm_stats=norm_stats,
        )
        if len(ds) > 0:
            val_sets.append(ds)

    if not train_sets:
        raise RuntimeError("No valid training samples found. Please check train file list and input/pred steps.")
    if not val_sets:
        raise RuntimeError("No valid validation samples found. Please check val file list and input/pred steps.")

    train_dataset = train_sets[0] if len(train_sets) == 1 else ConcatDataset(train_sets)
    val_dataset = val_sets[0] if len(val_sets) == 1 else ConcatDataset(val_sets)

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader
