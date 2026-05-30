from __future__ import annotations

from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F


class MLPEncoder(nn.Module):
    def __init__(self, input_dim: int, widths: tuple[int, ...], dropout: float):
        super().__init__()
        layers = []
        prev = input_dim
        for i, width in enumerate(widths):
            layers.append(nn.Linear(prev, width))
            layers.append(nn.LayerNorm(width))
            layers.append(nn.SiLU())
            if dropout > 0 and i < len(widths) - 1:
                layers.append(nn.Dropout(dropout))
            prev = width
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class URLLCNet(nn.Module):
    def __init__(self, input_dim: int, num_actions: int = 5):
        super().__init__()
        self.encoder = MLPEncoder(input_dim, (256, 256, 128), dropout=0.05)
        self.action_head = nn.Linear(128, num_actions)
        self.cpu_ratio_head = nn.Linear(128, 1)
        self.delay_head = nn.Linear(128, 1)
        self.margin_head = nn.Linear(128, 1)
        self.violation_head = nn.Linear(128, 1)
        self.reliability_head = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.encoder(x)
        return {
            "action_logits": self.action_head(h),
            "cpu_ratio": torch.sigmoid(self.cpu_ratio_head(h)).squeeze(-1),
            "delay_ms": F.softplus(self.delay_head(h)).squeeze(-1),
            "margin_ms": self.margin_head(h).squeeze(-1),
            "violation_logit": self.violation_head(h).squeeze(-1),
            "reliability": torch.sigmoid(self.reliability_head(h)).squeeze(-1),
        }


class EMBBNet(nn.Module):
    def __init__(self, input_dim: int, num_actions: int = 5, num_status: int = 3):
        super().__init__()
        self.encoder = MLPEncoder(input_dim, (384, 256, 128), dropout=0.10)
        self.action_head = nn.Linear(128, num_actions)
        self.cpu_ratio_head = nn.Linear(128, 1)
        self.delay_head = nn.Linear(128, 1)
        self.status_head = nn.Linear(128, num_status)
        self.utility_head = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.encoder(x)
        return {
            "action_logits": self.action_head(h),
            "cpu_ratio": torch.sigmoid(self.cpu_ratio_head(h)).squeeze(-1),
            "delay_sec": F.softplus(self.delay_head(h)).squeeze(-1),
            "status_logits": self.status_head(h),
            "utility": self.utility_head(h).squeeze(-1),
        }
