from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from iov_config import HeterogeneousIoVConfig
from prior_dataset import (
    FEATURE_COLUMNS,
    FeaturePreprocessor,
    PriorDataset,
    build_targets,
    save_split_indices,
    stratified_split,
)
from prior_models import EMBBNet, URLLCNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train supervised prior networks for IoV expert imitation.")
    parser.add_argument("--task", choices=["urllc", "embb"], required=True)
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default="model")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def class_weights(labels: np.ndarray, num_classes: int, cap: float = 8.0) -> torch.Tensor:
    counts = np.bincount(labels.astype(np.int64), minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = labels.size / (num_classes * counts)
    weights = np.clip(weights, 0.1, cap)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def confusion_matrix(pred: np.ndarray, true: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(true.astype(np.int64), pred.astype(np.int64)):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def macro_f1(pred: np.ndarray, true: np.ndarray, num_classes: int) -> float:
    cm = confusion_matrix(pred, true, num_classes)
    f1s = []
    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        if cm[i, :].sum() == 0:
            continue
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1s.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return float(np.mean(f1s)) if f1s else 0.0


def per_class_f1(pred: np.ndarray, true: np.ndarray, num_classes: int) -> Dict[str, float]:
    cm = confusion_matrix(pred, true, num_classes)
    result: Dict[str, float] = {}
    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        result[f"class_{i}_precision"] = float(precision)
        result[f"class_{i}_recall"] = float(recall)
        result[f"class_{i}_f1"] = float(f1)
    return result


def move_targets(targets: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in targets.items()}


def masked_urllc_logits(logits: torch.Tensor) -> torch.Tensor:
    masked = logits.clone()
    masked[:, HeterogeneousIoVConfig.ACTION_DROP] = -1e9
    return masked


def train_loss_urllc(
    outputs: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    action_ce: nn.Module,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    action_loss = action_ce(masked_urllc_logits(outputs["action_logits"]), targets["action"].long())
    cpu_loss = nn.functional.smooth_l1_loss(outputs["cpu_ratio"], targets["cpu_ratio"].float())
    delay_loss = nn.functional.smooth_l1_loss(outputs["delay_ms"], targets["delay_ms"].float())
    margin_loss = nn.functional.smooth_l1_loss(outputs["margin_ms"], targets["margin_ms"].float())
    violation_loss = nn.functional.binary_cross_entropy_with_logits(outputs["violation_logit"], targets["violation"].float())
    reliability_loss = nn.functional.smooth_l1_loss(outputs["reliability"], targets["reliability"].float())
    drop_prob = torch.softmax(outputs["action_logits"], dim=-1)[:, HeterogeneousIoVConfig.ACTION_DROP].mean()
    loss = (
        action_loss
        + 0.5 * cpu_loss
        + delay_loss
        + margin_loss
        + 4.0 * violation_loss
        + 0.5 * reliability_loss
        + 10.0 * drop_prob
    )
    return loss, {
        "action_loss": float(action_loss.detach().cpu()),
        "violation_loss": float(violation_loss.detach().cpu()),
        "drop_prob": float(drop_prob.detach().cpu()),
    }


def train_loss_embb(
    outputs: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    action_ce: nn.Module,
    status_ce: nn.Module,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    action_loss = action_ce(outputs["action_logits"], targets["action"].long())
    non_drop = targets["non_drop"].float()
    cpu_raw = nn.functional.smooth_l1_loss(outputs["cpu_ratio"], targets["cpu_ratio"].float(), reduction="none")
    cpu_loss = (cpu_raw * non_drop).sum() / non_drop.sum().clamp_min(1.0)
    delay_loss = nn.functional.smooth_l1_loss(outputs["delay_sec"], targets["delay_sec"].float())
    status_loss = status_ce(outputs["status_logits"], targets["status"].long())
    utility_loss = nn.functional.smooth_l1_loss(outputs["utility"], targets["utility"].float())
    loss = action_loss + 0.5 * cpu_loss + delay_loss + status_loss + 0.5 * utility_loss
    return loss, {
        "action_loss": float(action_loss.detach().cpu()),
        "status_loss": float(status_loss.detach().cpu()),
    }


@torch.no_grad()
def collect_predictions(model: nn.Module, loader: DataLoader, task: str, device: torch.device) -> Dict[str, np.ndarray]:
    model.eval()
    chunks: Dict[str, list[np.ndarray]] = {}
    for x, targets in loader:
        x = x.to(device, non_blocking=True)
        outputs = model(x)
        if task == "urllc":
            action_logits = masked_urllc_logits(outputs["action_logits"])
            pred_action = action_logits.argmax(dim=-1)
            pred_violation = (torch.sigmoid(outputs["violation_logit"]) >= 0.5).long()
            batch = {
                "pred_action": pred_action.cpu().numpy(),
                "true_action": targets["action"].numpy(),
                "pred_delay": outputs["delay_ms"].cpu().numpy(),
                "true_delay": targets["delay_ms"].numpy(),
                "pred_violation": pred_violation.cpu().numpy(),
                "true_violation": targets["violation"].numpy().astype(np.int64),
                "is_ood": targets["is_ood"].numpy().astype(np.int64),
                "pred_drop_prob": torch.softmax(outputs["action_logits"], dim=-1)[:, HeterogeneousIoVConfig.ACTION_DROP].cpu().numpy(),
            }
        else:
            pred_action = outputs["action_logits"].argmax(dim=-1)
            pred_status = outputs["status_logits"].argmax(dim=-1)
            batch = {
                "pred_action": pred_action.cpu().numpy(),
                "true_action": targets["action"].numpy(),
                "pred_status": pred_status.cpu().numpy(),
                "true_status": targets["status"].numpy(),
                "pred_delay": outputs["delay_sec"].cpu().numpy(),
                "true_delay": targets["delay_sec"].numpy(),
                "pred_utility": outputs["utility"].cpu().numpy(),
                "true_utility": targets["utility"].numpy(),
                "is_ood": targets["is_ood"].numpy().astype(np.int64),
            }
        for key, value in batch.items():
            chunks.setdefault(key, []).append(value)
    return {k: np.concatenate(v, axis=0) for k, v in chunks.items()}


def prefixed_metrics(metrics: Dict[str, float], prefix: str) -> Dict[str, float]:
    return {f"{prefix}_{k}": v for k, v in metrics.items()}


def metrics_from_predictions(preds: Dict[str, np.ndarray], task: str, suffix: str = "all") -> Dict[str, float]:
    out: Dict[str, float] = {}
    pred_action = preds["pred_action"]
    true_action = preds["true_action"]
    out["action_accuracy"] = float((pred_action == true_action).mean())
    out["action_macro_f1"] = macro_f1(pred_action, true_action, 5)
    out.update(per_class_f1(pred_action, true_action, 5))
    out["delay_mae"] = float(np.mean(np.abs(preds["pred_delay"] - preds["true_delay"])))
    if task == "urllc":
        tv = preds["true_violation"].astype(bool)
        pv = preds["pred_violation"].astype(bool)
        out["violation_recall"] = float(((pv & tv).sum()) / max(1, tv.sum()))
        out["violation_false_negative_rate"] = float(((~pv & tv).sum()) / max(1, tv.sum()))
        out["drop_pred_rate"] = float((pred_action == HeterogeneousIoVConfig.ACTION_DROP).mean())
        out["drop_prob_mean"] = float(preds["pred_drop_prob"].mean())
    else:
        pred_status = preds["pred_status"]
        true_status = preds["true_status"]
        out["status_accuracy"] = float((pred_status == true_status).mean())
        out["status_macro_f1"] = macro_f1(pred_status, true_status, 3)
        out.update({f"status_{k}": v for k, v in per_class_f1(pred_status, true_status, 3).items()})
        out["utility_mae"] = float(np.mean(np.abs(preds["pred_utility"] - preds["true_utility"])))
        non_drop_true = true_action != HeterogeneousIoVConfig.ACTION_DROP
        out["over_drop_rate"] = float(((pred_action == HeterogeneousIoVConfig.ACTION_DROP) & non_drop_true).sum() / max(1, non_drop_true.sum()))
    return prefixed_metrics(out, suffix)


def evaluate(model: nn.Module, loader: DataLoader, task: str, device: torch.device) -> Dict[str, float]:
    preds = collect_predictions(model, loader, task, device)
    metrics = metrics_from_predictions(preds, task, "all")
    for name, flag in [("normal", 0), ("ood", 1)]:
        mask = preds["is_ood"] == flag
        if mask.any():
            sub = {k: v[mask] for k, v in preds.items()}
            metrics.update(metrics_from_predictions(sub, task, name))
    return metrics


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    task: str,
    device: torch.device,
    action_ce: nn.Module,
    status_ce: nn.Module | None,
    scaler: GradScaler,
) -> float:
    model.train()
    total_loss = 0.0
    total = 0
    use_amp = device.type == "cuda"
    for x, targets in loader:
        x = x.to(device, non_blocking=True)
        targets = move_targets(targets, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(x)
            if task == "urllc":
                loss, _ = train_loss_urllc(outputs, targets, action_ce)
            else:
                assert status_ce is not None
                loss, _ = train_loss_embb(outputs, targets, action_ce, status_ce)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()
        batch_size = int(x.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total += batch_size
    return total_loss / max(1, total)


def make_loaders(
    df: pd.DataFrame,
    task: str,
    splits: Dict[str, np.ndarray],
    preprocessor: FeaturePreprocessor,
    batch_size: int,
    num_workers: int,
) -> Dict[str, DataLoader]:
    loaders: Dict[str, DataLoader] = {}
    for split_name, idx in splits.items():
        part = df.iloc[idx].reset_index(drop=True)
        x = preprocessor.transform(part)
        y = build_targets(part, task)
        dataset = PriorDataset(x, y)
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    return loaders


def best_score(metrics: Dict[str, float], task: str) -> float:
    if task == "urllc":
        return (
            metrics.get("all_violation_false_negative_rate", 1.0)
            + 0.20 * (1.0 - metrics.get("all_action_accuracy", 0.0))
            + 0.02 * metrics.get("all_delay_mae", 1e3)
        )
    return (
        1.0 - metrics.get("all_action_macro_f1", 0.0)
        + 0.20 * (1.0 - metrics.get("all_status_macro_f1", 0.0))
        + 0.02 * metrics.get("all_delay_mae", 1e3)
    )


def dump_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    task = args.task
    data_path = Path(args.data or f"data/{'urllc' if task == 'urllc' else 'embb'}_expert_dataset.csv")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Torch CUDA: {torch.version.cuda}")

    df = pd.read_csv(data_path)
    df = df.reset_index(drop=True)
    stratify_cols = ["label_action", "is_ood", "urllc_outage"] if task == "urllc" else ["label_action", "is_ood", "status_id"]
    splits = stratified_split(df, stratify_cols=stratify_cols, seed=args.seed)
    split_sizes = {k: int(len(v)) for k, v in splits.items()}
    print(f"Split sizes: {split_sizes}")
    save_split_indices(out_dir / f"{task}_split_indices.json", splits)

    preprocessor = FeaturePreprocessor.fit(df.iloc[splits["train"]].reset_index(drop=True), FEATURE_COLUMNS)
    scaler_path = out_dir / f"{task}_feature_scaler.npz"
    preprocessor.save(scaler_path)
    loaders = make_loaders(df, task, splits, preprocessor, args.batch_size, args.num_workers)

    input_dim = len(preprocessor.columns)
    model: nn.Module = URLLCNet(input_dim) if task == "urllc" else EMBBNet(input_dim)
    model.to(device)

    train_labels = df.iloc[splits["train"]]["label_action"].to_numpy(dtype=np.int64)
    action_w = class_weights(train_labels, 5).to(device)
    action_ce = nn.CrossEntropyLoss(weight=action_w)
    status_ce = None
    if task == "embb":
        status_labels = df.iloc[splits["train"]]["status_id"].to_numpy(dtype=np.int64)
        status_w = class_weights(status_labels, 3).to(device)
        status_ce = nn.CrossEntropyLoss(weight=status_w)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    best_val_score = math.inf
    best_epoch = -1
    no_improve = 0
    history = []
    ckpt_path = out_dir / f"{task}_prior_best.pth"
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, loaders["train"], optimizer, task, device, action_ce, status_ce, scaler)
        val_metrics = evaluate(model, loaders["val"], task, device)
        score = best_score(val_metrics, task)
        scheduler.step(score)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_score": score,
            "lr": optimizer.param_groups[0]["lr"],
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        print(
            f"epoch {epoch:03d} loss={train_loss:.5f} score={score:.5f} "
            f"acc={val_metrics.get('all_action_accuracy', 0.0):.4f} "
            f"macro_f1={val_metrics.get('all_action_macro_f1', 0.0):.4f} "
            f"delay_mae={val_metrics.get('all_delay_mae', 0.0):.5f}"
        )
        if score < best_val_score:
            best_val_score = score
            best_epoch = epoch
            no_improve = 0
            torch.save(
                {
                    "task": task,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "input_dim": input_dim,
                    "feature_columns": preprocessor.columns,
                    "val_metrics": val_metrics,
                    "val_score": score,
                },
                ckpt_path,
            )
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch}. Best epoch={best_epoch}")
                break

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_metrics = evaluate(model, loaders["val"], task, device)
    test_metrics = evaluate(model, loaders["test"], task, device)
    metrics_payload = {
        "task": task,
        "data_path": str(data_path),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "split_sizes": split_sizes,
        "best_epoch": best_epoch,
        "best_val_score": best_val_score,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "history": history,
        "elapsed_sec": time.time() - start,
    }
    metrics_path = out_dir / f"{task}_prior_metrics.json"
    dump_json(metrics_path, metrics_payload)
    print(f"Saved checkpoint: {ckpt_path}")
    print(f"Saved scaler: {scaler_path}")
    print(f"Saved metrics: {metrics_path}")
    print("Final test metrics:")
    for key, value in test_metrics.items():
        if key.startswith("all_") or key.startswith("ood_"):
            print(f"  {key}: {value:.6f}")


if __name__ == "__main__":
    main()
