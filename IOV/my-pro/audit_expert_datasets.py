from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from iov_config import HeterogeneousIoVConfig, OffloadAction


ACTION_NAMES: Dict[int, str] = {
    int(OffloadAction.LOCAL): "LOCAL",
    int(OffloadAction.LOCAL_MEC): "LOCAL_MEC",
    int(OffloadAction.REMOTE_MEC): "REMOTE_MEC",
    int(OffloadAction.CLOUD): "CLOUD",
    int(OffloadAction.DROP): "DROP",
}


def _pct(series: pd.Series) -> Dict[str, float]:
    dist = series.value_counts(normalize=True).sort_index() * 100.0
    return {ACTION_NAMES.get(int(k), str(k)): round(float(v), 3) for k, v in dist.items()}


def _nan_report(df: pd.DataFrame) -> Dict[str, int]:
    na = df.isna().sum()
    return {str(k): int(v) for k, v in na[na > 0].sort_values(ascending=False).items()}


def _inf_count(df: pd.DataFrame) -> int:
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        return 0
    return int(np.isinf(numeric.to_numpy()).sum())


def audit_urllc(path: Path) -> bool:
    df = pd.read_csv(path)
    ok = True
    print(f"\n=== URLLC audit: {path} ===")
    print(f"rows={len(df)} cols={len(df.columns)}")
    print("action_pct=", _pct(df["label_action"]))
    print("ood_pct=", round(float(df["is_ood"].mean() * 100.0), 3))
    print("feasible_pct=", round(float(df["is_feasible_label"].mean() * 100.0), 3))
    print("outage_pct=", round(float(df["urllc_outage"].mean() * 100.0), 3))
    print("delay_ms_pct=", np.round(np.nanpercentile(df["total_delay_sec"] * 1e3, [0, 25, 50, 75, 95, 99, 100]), 6).tolist())
    print("nan=", _nan_report(df))
    print("inf_cells=", _inf_count(df))

    if int(OffloadAction.DROP) in set(df["label_action"].astype(int)):
        print("[FAIL] URLLC contains DROP labels.")
        ok = False
    mismatch = ((df["total_delay_sec"] > HeterogeneousIoVConfig.URLLC_MAX_DELAY_SEC) != df["deadline_violation"].astype(bool)).sum()
    if int(mismatch) != 0:
        print(f"[FAIL] URLLC deadline_violation mismatch rows={int(mismatch)}")
        ok = False
    required = ["upload_delay_sec", "return_delay_sec", "estimated_reliability", "candidate_local_delay_sec", "candidate_local_mec_delay_sec"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[FAIL] URLLC missing columns: {missing}")
        ok = False
    return ok


def audit_embb(path: Path) -> bool:
    df = pd.read_csv(path)
    ok = True
    print(f"\n=== eMBB audit: {path} ===")
    print(f"rows={len(df)} cols={len(df.columns)}")
    print("action_pct=", _pct(df["label_action"]))
    print("status_pct=", {int(k): round(float(v), 3) for k, v in (df["status_id"].value_counts(normalize=True).sort_index() * 100.0).items()})
    print("ood_pct=", round(float(df["is_ood"].mean() * 100.0), 3))
    print("service_pct=", round(float(df["is_feasible_label"].mean() * 100.0), 3))
    served = df[df["label_action"] != int(OffloadAction.DROP)]
    if len(served):
        print("served_delay_s_pct=", np.round(np.nanpercentile(served["total_delay_sec"], [0, 25, 50, 75, 95, 99, 100]), 6).tolist())
    print("nan=", _nan_report(df))
    print("inf_cells=", _inf_count(df))

    success_mismatch = (((df["total_delay_sec"] <= df["deadline_sec"]) & (df["label_action"] != int(OffloadAction.DROP))) != (df["status_id"] == 1)).sum()
    service_mismatch = (((df["total_delay_sec"] <= df["tolerant_deadline_sec"]) & (df["label_action"] != int(OffloadAction.DROP))) != df["is_feasible_label"].astype(bool)).sum()
    if int(success_mismatch) != 0:
        print(f"[FAIL] eMBB SUCCESS mismatch rows={int(success_mismatch)}")
        ok = False
    if int(service_mismatch) != 0:
        print(f"[FAIL] eMBB service/tolerant mismatch rows={int(service_mismatch)}")
        ok = False
    required = ["upload_delay_sec", "return_delay_sec", "estimated_reliability", "candidate_cloud_delay_sec", "candidate_remote_mec_delay_sec"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[FAIL] eMBB missing columns: {missing}")
        ok = False
    return ok


def main() -> None:
    base = Path("data")
    checks = [
        audit_urllc(base / "urllc_expert_dataset.csv"),
        audit_embb(base / "embb_expert_dataset.csv"),
    ]
    if not all(checks):
        raise SystemExit(1)
    print("\nDataset audit passed.")


if __name__ == "__main__":
    main()
