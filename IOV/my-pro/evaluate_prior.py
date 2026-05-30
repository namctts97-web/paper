from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from prior_dataset import FeaturePreprocessor, PriorDataset, build_targets
from prior_models import EMBBNet, URLLCNet
from train_prior import evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved prior checkpoints on the held-out test split.")
    parser.add_argument("--task", choices=["urllc", "embb"], required=True)
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--model-dir", type=str, default="model")
    parser.add_argument("--batch-size", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task = args.task
    model_dir = Path(args.model_dir)
    data_path = Path(args.data or f"data/{task}_expert_dataset.csv")
    ckpt_path = model_dir / f"{task}_prior_best.pth"
    scaler_path = model_dir / f"{task}_feature_scaler.npz"
    split_path = model_dir / f"{task}_split_indices.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(data_path).reset_index(drop=True)
    preprocessor = FeaturePreprocessor.load(scaler_path)
    splits = json.loads(split_path.read_text(encoding="utf-8"))
    test_idx = np.array(splits["test"], dtype=np.int64)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    x = preprocessor.transform(test_df)
    y = build_targets(test_df, task)
    loader = DataLoader(PriorDataset(x, y), batch_size=args.batch_size, shuffle=False, pin_memory=torch.cuda.is_available())

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    input_dim = int(checkpoint["input_dim"])
    model = URLLCNet(input_dim) if task == "urllc" else EMBBNet(input_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    metrics = evaluate(model, loader, task, device)
    print(f"Evaluated {task} on {len(test_df)} held-out samples using {device}.")
    for key, value in metrics.items():
        if key.startswith("all_") or key.startswith("ood_"):
            print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
