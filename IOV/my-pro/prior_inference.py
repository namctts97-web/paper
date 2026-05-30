from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from iov_config import HeterogeneousIoVConfig, OffloadAction, ServiceStatus, TaskType
from iov_scheduler_engine import OffloadingDecisionEngine
from prior_dataset import FEATURE_COLUMNS, FeaturePreprocessor
from prior_models import EMBBNet, URLLCNet


STATUS_ID_TO_NAME = {
    0: "REJECTED",
    1: "SUCCESS",
    2: "DEGRADED",
}


class PriorPolicy:
    """Load and run the supervised prior networks with optional safety shielding."""

    def __init__(
        self,
        task: str,
        model_dir: str | Path = "model",
        device: Optional[str] = None,
    ):
        self.task = task.lower()
        if self.task not in {"urllc", "embb"}:
            raise ValueError("task must be 'urllc' or 'embb'")
        self.model_dir = Path(model_dir)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.preprocessor = FeaturePreprocessor.load(self.model_dir / f"{self.task}_feature_scaler.npz")
        checkpoint = torch.load(self.model_dir / f"{self.task}_prior_best.pth", map_location=self.device, weights_only=False)
        input_dim = int(checkpoint["input_dim"])
        self.model = URLLCNet(input_dim) if self.task == "urllc" else EMBBNet(input_dim)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _masked_urllc_logits(logits: torch.Tensor) -> torch.Tensor:
        masked = logits.clone()
        masked[:, HeterogeneousIoVConfig.ACTION_DROP] = -1e9
        return masked

    def _predict_raw(self, df: pd.DataFrame, batch_size: int = 4096) -> Dict[str, np.ndarray]:
        x = self.preprocessor.transform(df)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(x.astype(np.float32, copy=True))),
            batch_size=batch_size,
            shuffle=False,
            pin_memory=self.device.type == "cuda",
        )
        chunks: Dict[str, List[np.ndarray]] = {}
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self.device, non_blocking=True)
                out = self.model(xb)
                if self.task == "urllc":
                    logits = out["action_logits"]
                    masked_logits = self._masked_urllc_logits(logits)
                    batch = {
                        "prior_action": masked_logits.argmax(dim=-1).cpu().numpy(),
                        "prior_cpu_ratio": out["cpu_ratio"].cpu().numpy(),
                        "prior_delay_ms": out["delay_ms"].cpu().numpy(),
                        "prior_margin_ms": out["margin_ms"].cpu().numpy(),
                        "prior_violation_prob": torch.sigmoid(out["violation_logit"]).cpu().numpy(),
                        "prior_reliability": out["reliability"].cpu().numpy(),
                        "prior_drop_prob": torch.softmax(logits, dim=-1)[:, HeterogeneousIoVConfig.ACTION_DROP].cpu().numpy(),
                    }
                else:
                    status = out["status_logits"].argmax(dim=-1).cpu().numpy()
                    batch = {
                        "prior_action": out["action_logits"].argmax(dim=-1).cpu().numpy(),
                        "prior_cpu_ratio": out["cpu_ratio"].cpu().numpy(),
                        "prior_delay_sec": out["delay_sec"].cpu().numpy(),
                        "prior_status_id": status,
                        "prior_utility": out["utility"].cpu().numpy(),
                    }
                for key, value in batch.items():
                    chunks.setdefault(key, []).append(value)
        return {k: np.concatenate(v, axis=0) for k, v in chunks.items()}

    @staticmethod
    def _state_from_row(row: pd.Series) -> Dict[str, float]:
        return {k: float(row[k]) for k in FEATURE_COLUMNS if k in row.index}

    @staticmethod
    def _evaluate_action(row: pd.Series, action: int, cpu_ratio: float, task_type: str) -> Dict[str, Any]:
        state = PriorPolicy._state_from_row(row)
        return OffloadingDecisionEngine.evaluate_action(state, int(action), float(cpu_ratio), task_type, True)

    @staticmethod
    def _best_urllc_safe_candidate(row: pd.Series) -> Dict[str, Any]:
        state = PriorPolicy._state_from_row(row)
        candidates = []
        for action in [int(OffloadAction.LOCAL), int(OffloadAction.LOCAL_MEC), int(OffloadAction.REMOTE_MEC), int(OffloadAction.CLOUD)]:
            for ratio in HeterogeneousIoVConfig.URLLC_CPU_RATIO_GRID:
                c = OffloadingDecisionEngine.evaluate_action(state, action, ratio, TaskType.URLLC, True)
                if c.get("illegal_action", False):
                    continue
                candidates.append(c)
        feasible = [
            c for c in candidates
            if bool(c.get("feasible", False)) and float(c.get("total_delay_sec", np.inf)) <= HeterogeneousIoVConfig.URLLC_MAX_DELAY_SEC
        ]
        if feasible:
            return min(feasible, key=lambda c: (float(c["total_delay_sec"]), float(c.get("resource_price", 0.0))))
        return min(candidates, key=lambda c: (float(c.get("total_delay_sec", np.inf)), float(c.get("resource_price", 0.0))))

    def predict(
        self,
        df_or_state: pd.DataFrame | Dict[str, Any],
        batch_size: int = 4096,
        safety_shield: bool = False,
        violation_threshold: float = 0.35,
        margin_threshold_ms: float = 0.30,
    ) -> pd.DataFrame:
        if isinstance(df_or_state, dict):
            df = pd.DataFrame([df_or_state])
        else:
            df = df_or_state.reset_index(drop=True).copy()

        raw = self._predict_raw(df, batch_size=batch_size)
        pred = pd.DataFrame(raw)
        pred["final_action"] = pred["prior_action"].astype(int)
        pred["final_cpu_ratio"] = pred["prior_cpu_ratio"].astype(float)
        pred["shield_applied"] = 0

        if self.task == "urllc":
            unsafe = (
                safety_shield
                & (
                    (pred["prior_violation_prob"] > violation_threshold)
                    | (pred["prior_margin_ms"] < margin_threshold_ms)
                    | (pred["prior_action"] == int(OffloadAction.DROP))
                )
            )
            final_delay = []
            final_status = []
            final_reliability = []
            for i, row in df.iterrows():
                if bool(unsafe.iloc[i]):
                    c = self._best_urllc_safe_candidate(row)
                    pred.loc[i, "final_action"] = int(c["action"])
                    pred.loc[i, "final_cpu_ratio"] = float(c.get("cpu_ratio", 1.0))
                    pred.loc[i, "shield_applied"] = 1
                else:
                    c = self._evaluate_action(row, int(pred.loc[i, "final_action"]), float(pred.loc[i, "final_cpu_ratio"]), TaskType.URLLC)
                final_delay.append(float(c.get("total_delay_sec", np.inf)))
                final_status.append(str(c.get("status", "")))
                final_reliability.append(float(c.get("estimated_reliability", np.nan)))
            pred["final_delay_sec"] = final_delay
            pred["final_delay_ms"] = pred["final_delay_sec"] * 1e3
            pred["final_status"] = final_status
            pred["final_reliability"] = final_reliability
            pred["final_violation"] = (pred["final_delay_sec"] > HeterogeneousIoVConfig.URLLC_MAX_DELAY_SEC).astype(int)
        else:
            final_delay = []
            final_status = []
            for i, row in df.iterrows():
                c = self._evaluate_action(row, int(pred.loc[i, "final_action"]), float(pred.loc[i, "final_cpu_ratio"]), TaskType.EMBB)
                final_delay.append(float(c.get("total_delay_sec", np.inf)))
                final_status.append(str(c.get("status", "")))
            pred["prior_status_name"] = pred["prior_status_id"].map(STATUS_ID_TO_NAME)
            pred["final_delay_sec"] = final_delay
            pred["final_status"] = final_status
            pred["final_rejected"] = pred["final_status"].eq(ServiceStatus.REJECTED).astype(int)

        return pred


def load_policy(task: str, model_dir: str | Path = "model", device: Optional[str] = None) -> PriorPolicy:
    return PriorPolicy(task=task, model_dir=model_dir, device=device)
