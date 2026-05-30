from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from iov_config import HeterogeneousIoVConfig, OffloadAction
from prior_inference import PriorPolicy
from train_prior import confusion_matrix, macro_f1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate prior inference wrappers and URLLC safety shield.")
    parser.add_argument("--task", choices=["urllc", "embb"], required=True)
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--model-dir", type=str, default="model")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--violation-threshold", type=float, default=0.35)
    parser.add_argument("--margin-threshold-ms", type=float, default=0.30)
    parser.add_argument("--sweep", action="store_true")
    return parser.parse_args()


def load_split(task: str, split: str, data_path: Path, model_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path).reset_index(drop=True)
    splits = json.loads((model_dir / f"{task}_split_indices.json").read_text(encoding="utf-8"))
    idx = np.array(splits[split], dtype=np.int64)
    return df.iloc[idx].reset_index(drop=True)


def summarize_urllc(df: pd.DataFrame, pred: pd.DataFrame) -> Dict[str, float]:
    true_action = df["label_action"].to_numpy(dtype=np.int64)
    final_action = pred["final_action"].to_numpy(dtype=np.int64)
    true_violation = df["deadline_violation"].to_numpy(dtype=np.int64).astype(bool)
    policy_violation = pred["final_violation"].to_numpy(dtype=np.int64).astype(bool)
    expert_feasible = df["is_feasible_label"].to_numpy(dtype=np.int64).astype(bool)
    metrics = {
        "action_accuracy": float((true_action == final_action).mean()),
        "action_macro_f1": macro_f1(final_action, true_action, 5),
        "drop_pred_rate": float((final_action == int(OffloadAction.DROP)).mean()),
        "shield_rate": float(pred["shield_applied"].mean()),
        "policy_violation_rate": float(policy_violation.mean()),
        "policy_violation_on_expert_feasible": float((policy_violation & expert_feasible).sum() / max(1, expert_feasible.sum())),
        "true_violation_recall": float((policy_violation & true_violation).sum() / max(1, true_violation.sum())),
        "true_violation_false_negative_rate": float(((~policy_violation) & true_violation).sum() / max(1, true_violation.sum())),
        "final_delay_mae_ms_vs_expert": float(np.mean(np.abs(pred["final_delay_ms"].to_numpy() - df["total_delay_ms"].to_numpy()))),
    }
    ood_mask = df["is_ood"].to_numpy(dtype=np.int64) == 1
    if ood_mask.any():
        metrics["ood_policy_violation_rate"] = float(policy_violation[ood_mask].mean())
        metrics["ood_action_accuracy"] = float((true_action[ood_mask] == final_action[ood_mask]).mean())
    return metrics


def summarize_embb(df: pd.DataFrame, pred: pd.DataFrame) -> Dict[str, float]:
    true_action = df["label_action"].to_numpy(dtype=np.int64)
    final_action = pred["final_action"].to_numpy(dtype=np.int64)
    metrics = {
        "action_accuracy": float((true_action == final_action).mean()),
        "action_macro_f1": macro_f1(final_action, true_action, 5),
        "drop_pred_rate": float((final_action == int(OffloadAction.DROP)).mean()),
        "over_drop_rate": float(((final_action == int(OffloadAction.DROP)) & (true_action != int(OffloadAction.DROP))).sum() / max(1, (true_action != int(OffloadAction.DROP)).sum())),
        "final_delay_mae_sec_vs_expert": float(np.mean(np.abs(pred["final_delay_sec"].to_numpy() - df["total_delay_sec"].to_numpy()))),
    }
    ood_mask = df["is_ood"].to_numpy(dtype=np.int64) == 1
    if ood_mask.any():
        metrics["ood_action_accuracy"] = float((true_action[ood_mask] == final_action[ood_mask]).mean())
        metrics["ood_macro_f1"] = macro_f1(final_action[ood_mask], true_action[ood_mask], 5)
    return metrics


def print_metrics(title: str, metrics: Dict[str, float]) -> None:
    print(f"\n{title}")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")


def main() -> None:
    args = parse_args()
    task = args.task
    model_dir = Path(args.model_dir)
    data_path = Path(args.data or f"data/{task}_expert_dataset.csv")
    df = load_split(task, args.split, data_path, model_dir)
    policy = PriorPolicy(task, model_dir=model_dir)

    if task == "urllc" and args.sweep:
        thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
        margins = [0.10, 0.20, 0.30, 0.50]
        print(f"URLLC safety sweep on {len(df)} {args.split} samples")
        print("tau_v, tau_m_ms, shield_rate, policy_violation_on_expert_feasible, action_acc, delay_mae_ms")
        for tau in thresholds:
            for margin in margins:
                pred = policy.predict(df, safety_shield=True, violation_threshold=tau, margin_threshold_ms=margin)
                m = summarize_urllc(df, pred)
                print(f"{tau:.2f}, {margin:.2f}, {m['shield_rate']:.6f}, {m['policy_violation_on_expert_feasible']:.6f}, {m['action_accuracy']:.6f}, {m['final_delay_mae_ms_vs_expert']:.6f}")
        return

    if task == "urllc":
        network_only = policy.predict(df, safety_shield=False)
        shielded = policy.predict(
            df,
            safety_shield=True,
            violation_threshold=args.violation_threshold,
            margin_threshold_ms=args.margin_threshold_ms,
        )
        print_metrics("URLLC network-only", summarize_urllc(df, network_only))
        print_metrics("URLLC with safety shield", summarize_urllc(df, shielded))
        print("\nAction confusion with shield rows=true cols=pred")
        print(confusion_matrix(shielded["final_action"].to_numpy(), df["label_action"].to_numpy(), 5))
    else:
        pred = policy.predict(df, safety_shield=False)
        print_metrics("eMBB prior inference", summarize_embb(df, pred))
        print("\nAction confusion rows=true cols=pred")
        print(confusion_matrix(pred["final_action"].to_numpy(), df["label_action"].to_numpy(), 5))


if __name__ == "__main__":
    main()
