from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from iov_config import HeterogeneousIoVConfig


FEATURE_COLUMNS: List[str] = list(HeterogeneousIoVConfig.STATE_FEATURES)

LOG1P_COLUMNS = {
    "data_size_bits",
    "result_size_bits",
    "required_cycles",
    "sinr_linear",
    "downlink_sinr_linear",
    "channel_gain_linear",
    "uplink_rate_shannon_bps",
    "downlink_rate_shannon_bps",
    "cpu_local_hz",
    "cpu_lmec_hz",
    "cpu_rmec_hz",
    "cpu_cloud_hz",
    "workload_local_cycles",
    "workload_lmec_cycles",
    "workload_rmec_cycles",
    "workload_cloud_cycles",
}


@dataclass
class FeaturePreprocessor:
    columns: List[str]
    log1p_columns: List[str]
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, df: pd.DataFrame, columns: Iterable[str] = FEATURE_COLUMNS) -> "FeaturePreprocessor":
        cols = list(columns)
        x = cls._raw_matrix(df, cols)
        log_cols = [c for c in cols if c in LOG1P_COLUMNS]
        for c in log_cols:
            idx = cols.index(c)
            x[:, idx] = np.log1p(np.clip(x[:, idx], a_min=0.0, a_max=None))
        mean = np.nanmean(x, axis=0)
        std = np.nanstd(x, axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        return cls(cols, log_cols, mean.astype(np.float32), std.astype(np.float32))

    @staticmethod
    def _raw_matrix(df: pd.DataFrame, columns: List[str]) -> np.ndarray:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise KeyError(f"Missing feature columns: {missing}")
        x = df[columns].to_numpy(dtype=np.float32, copy=True)
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        x = self._raw_matrix(df, self.columns)
        for c in self.log1p_columns:
            idx = self.columns.index(c)
            x[:, idx] = np.log1p(np.clip(x[:, idx], a_min=0.0, a_max=None))
        x = (x - self.mean) / self.std
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            columns=np.array(self.columns, dtype=object),
            log1p_columns=np.array(self.log1p_columns, dtype=object),
            mean=self.mean,
            std=self.std,
        )

    @classmethod
    def load(cls, path: str | Path) -> "FeaturePreprocessor":
        data = np.load(path, allow_pickle=True)
        return cls(
            columns=[str(x) for x in data["columns"].tolist()],
            log1p_columns=[str(x) for x in data["log1p_columns"].tolist()],
            mean=data["mean"].astype(np.float32),
            std=data["std"].astype(np.float32),
        )


def stratified_split(
    df: pd.DataFrame,
    stratify_cols: List[str],
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []
    strata = df[stratify_cols].astype(str).agg("|".join, axis=1)
    for _, idx_values in strata.groupby(strata).groups.items():
        idx = np.array(list(idx_values), dtype=np.int64)
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        if n >= 3:
            n_train = min(max(1, n_train), n - 2)
            n_val = min(max(1, n_val), n - n_train - 1)
        else:
            n_train = max(1, n - 1)
            n_val = 0
        train_idx.extend(idx[:n_train].tolist())
        val_idx.extend(idx[n_train:n_train + n_val].tolist())
        test_idx.extend(idx[n_train + n_val:].tolist())
    result = {
        "train": np.array(train_idx, dtype=np.int64),
        "val": np.array(val_idx, dtype=np.int64),
        "test": np.array(test_idx, dtype=np.int64),
    }
    for split in result.values():
        rng.shuffle(split)
    return result


def save_split_indices(path: str | Path, splits: Dict[str, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: v.astype(int).tolist() for k, v in splits.items()}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class PriorDataset(Dataset):
    def __init__(self, x: np.ndarray, targets: Dict[str, np.ndarray]):
        self.x = torch.from_numpy(x.astype(np.float32, copy=True))
        self.targets = {k: torch.from_numpy(np.array(v, copy=True)) for k, v in targets.items()}

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int):
        return self.x[idx], {k: v[idx] for k, v in self.targets.items()}


def build_targets(df: pd.DataFrame, task: str) -> Dict[str, np.ndarray]:
    y: Dict[str, np.ndarray] = {
        "action": df["label_action"].to_numpy(dtype=np.int64),
        "cpu_ratio": df["label_cpu_ratio"].to_numpy(dtype=np.float32),
        "delay_sec": df["total_delay_sec"].to_numpy(dtype=np.float32),
        "is_ood": df["is_ood"].to_numpy(dtype=np.int64),
    }
    if task == "urllc":
        y.update({
            "delay_ms": df["total_delay_ms"].to_numpy(dtype=np.float32),
            "margin_ms": df["deadline_margin_ms"].to_numpy(dtype=np.float32),
            "violation": df["deadline_violation"].to_numpy(dtype=np.float32),
            "reliability": df["estimated_reliability"].to_numpy(dtype=np.float32),
        })
    elif task == "embb":
        y.update({
            "status": df["status_id"].to_numpy(dtype=np.int64),
            "utility": df["utility_score"].to_numpy(dtype=np.float32),
            "non_drop": (df["label_action"].to_numpy(dtype=np.int64) != HeterogeneousIoVConfig.ACTION_DROP).astype(np.float32),
        })
    else:
        raise ValueError(f"Unknown task: {task}")
    for k, v in y.items():
        if np.issubdtype(v.dtype, np.floating):
            y[k] = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return y
