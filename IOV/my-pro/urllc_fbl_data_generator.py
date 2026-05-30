# ==========================================
# 文件名: urllc_fbl_data_generator.py
# 目的: 生成 URLLC FBL expert dataset
# 规则: URLLC 不允许 DROP；不可行样本标 OUTAGE，但标签仍是可执行动作
# ==========================================
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from iov_config import HeterogeneousIoVConfig, TaskType, OffloadAction
from iov_env_factory import IoVEnvironmentFactory, EnvironmentSnapshot, OODState
from iov_scheduler_engine import OffloadingDecisionEngine


class URLLCExpertDataGenerator:
    def __init__(
        self,
        num_samples: int = HeterogeneousIoVConfig.DEFAULT_NUM_EXPERT_SAMPLES,
        seed: int = HeterogeneousIoVConfig.DEFAULT_SEED,
        scenario: str = HeterogeneousIoVConfig.DEFAULT_SCENARIO,
        load_level: str = "medium",
        include_ood: bool = HeterogeneousIoVConfig.INCLUDE_OOD_IN_EXPERT_DATA,
        ood_ratio: float = HeterogeneousIoVConfig.OOD_SAMPLE_RATIO,
        use_random_queue: bool = True,
    ):
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.scenario = scenario
        self.load_level = load_level
        self.include_ood = bool(include_ood)
        self.ood_ratio = float(np.clip(ood_ratio, 0.0, 1.0))
        self.use_random_queue = bool(use_random_queue)
        self.cfg = HeterogeneousIoVConfig()
        self.cfg.validate()
        self.rng = np.random.default_rng(self.seed)
        self.factory = IoVEnvironmentFactory(seed=self.seed, scenario=self.scenario)

    def _random_ood_type(self) -> str:
        # URLLC 更关心 burst / deep fade / MEC overload。
        choices = [
            self.cfg.OOD_TYPE_CHANNEL_DEEP_FADE,
            self.cfg.OOD_TYPE_URLLC_BURST,
            self.cfg.OOD_TYPE_MEC_OVERLOAD,
            self.cfg.OOD_TYPE_HIGH_MOBILITY,
            self.cfg.OOD_TYPE_MIXED,
        ]
        return str(self.rng.choice(choices, p=[0.22, 0.30, 0.20, 0.13, 0.15]))

    def _build_snapshot(self, idx: int) -> EnvironmentSnapshot:
        t = idx * self.cfg.SLOT_DURATION_SEC
        if self.include_ood and self.rng.random() < self.ood_ratio:
            return self.factory.generate_snapshot(
                TaskType.URLLC,
                task_id=f"urllc_{idx}",
                current_time_sec=t,
                load_level=self.load_level,
                force_ood_type=self._random_ood_type(),
                include_ood=True,
                use_random_queue=self.use_random_queue,
            )
        return self.factory.generate_snapshot(
            TaskType.URLLC,
            task_id=f"urllc_{idx}",
            current_time_sec=t,
            load_level=self.load_level,
            include_ood=False,
            use_random_queue=self.use_random_queue,
        )

    @staticmethod
    def _sf(value: Any, default: float = np.nan) -> float:
        try:
            if value is None:
                return default
            v = float(value)
            return v if np.isfinite(v) else default
        except Exception:
            return default

    def _flatten_sample(self, snapshot: EnvironmentSnapshot, expert: Dict[str, Any], idx: int) -> Dict[str, Any]:
        task = snapshot.task
        label = int(expert.get("label_action", expert.get("action")))
        if label == int(OffloadAction.DROP) or not self.cfg.is_action_allowed(TaskType.URLLC, label):
            raise RuntimeError(f"Illegal URLLC label: {label}")
        total_delay = self._sf(expert.get("total_delay_sec"), np.inf)
        deadline_margin = task.max_delay_sec - total_delay if np.isfinite(total_delay) else -np.inf
        row: Dict[str, Any] = dict(snapshot.state_dict)
        row.update({
            "sample_id": idx,
            "arrival_time_sec": task.arrival_time_sec,
            "deadline_time_sec": task.deadline_time_sec,
            "max_delay_sec": task.max_delay_sec,
            "result_size_bits": task.result_size_bits,
            "position_x_m": snapshot.vehicle.position_x_m,
            "position_y_m": snapshot.vehicle.position_y_m,
            "distance_m": snapshot.channel.distance_m,
            "noise_power_w": snapshot.channel.noise_power_w,
            "interference_w": snapshot.channel.interference_w,
            "ood_deep_fade_loss_db": snapshot.ood.deep_fade_loss_db,
            "ood_backhaul_delay_multiplier": snapshot.ood.backhaul_delay_multiplier,
            "ood_mec_cpu_reduction_ratio": snapshot.ood.mec_cpu_reduction_ratio,
            "ood_speed_multiplier": snapshot.ood.speed_multiplier,
            "ood_queue_multiplier": snapshot.ood.queue_multiplier,
            "label_action": label,
            "label": label,
            "label_cpu_ratio": self._sf(expert.get("label_cpu_ratio"), 1.0),
            "status_id": 1 if bool(expert.get("is_feasible_label", expert.get("feasible", False))) else 0,
            "is_feasible_label": int(bool(expert.get("is_feasible_label", expert.get("feasible", False)))),
            "deadline_violation": int(bool(expert.get("deadline_violation", not expert.get("feasible", False)))),
            "urllc_outage": int(bool(expert.get("urllc_outage", False))),
            "total_delay_sec": total_delay,
            "total_delay_ms": total_delay * 1e3 if np.isfinite(total_delay) else np.inf,
            "deadline_margin_sec": deadline_margin,
            "deadline_margin_ms": deadline_margin * 1e3 if np.isfinite(deadline_margin) else -np.inf,
            "energy_joules": self._sf(expert.get("energy_joules")),
            "tx_energy_joules": self._sf(expert.get("tx_energy_joules")),
            "compute_energy_joules": self._sf(expert.get("compute_energy_joules")),
            "queue_delay_sec": self._sf(expert.get("queue_delay_sec")),
            "transmission_delay_sec": self._sf(expert.get("transmission_delay_sec")),
            "upload_delay_sec": self._sf(expert.get("upload_delay_sec")),
            "return_delay_sec": self._sf(expert.get("return_delay_sec")),
            "mac_delay_eval_sec": self._sf(expert.get("mac_delay_sec")),
            "backhaul_delay_sec": self._sf(expert.get("backhaul_delay_sec")),
            "backhaul_forward_delay_sec": self._sf(expert.get("backhaul_forward_delay_sec")),
            "backhaul_return_delay_sec": self._sf(expert.get("backhaul_return_delay_sec")),
            "compute_delay_sec": self._sf(expert.get("compute_delay_sec")),
            "rate_bps": self._sf(expert.get("rate_bps")),
            "uplink_rate_bps": self._sf(expert.get("uplink_rate_bps")),
            "downlink_rate_bps": self._sf(expert.get("downlink_rate_bps")),
            "estimated_reliability": self._sf(expert.get("estimated_reliability")),
            "estimated_outage_probability": self._sf(expert.get("estimated_outage_probability")),
            "safety_score": self._sf(expert.get("safety_score")),
            "num_candidates": int(expert.get("num_candidates", 0)),
            "num_feasible_candidates": int(expert.get("num_feasible_candidates", 0)),
            "second_best_action": int(expert.get("second_best_action", -1)),
            "expert_margin_to_second": self._sf(expert.get("expert_margin_to_second")),
        })
        for key, value in expert.items():
            if key.startswith("candidate_"):
                row[key] = self._sf(value) if not isinstance(value, (int, np.integer)) else int(value)
        return row

    def generate_dataset(self) -> pd.DataFrame:
        print(f"正在生成 {self.num_samples} 条 URLLC 专家数据...")
        print("Expert: FBL + 3ms safety-first oracle; URLLC 标签不允许 DROP。")
        st = time.time()
        rows: List[Dict[str, Any]] = []
        progress_step = max(10000, self.num_samples // 5) if self.num_samples >= 10000 else 0
        for i in range(self.num_samples):
            snap = self._build_snapshot(i)
            expert = OffloadingDecisionEngine.select_urllc_expert_action(snap.state_dict)
            rows.append(self._flatten_sample(snap, expert, i))
            if progress_step and (i + 1) % progress_step == 0:
                print(f"  progress: {i + 1}/{self.num_samples}", flush=True)
        df = pd.DataFrame(rows)
        print(f"URLLC 数据生成完毕，耗时 {time.time() - st:.2f} 秒。")
        self.print_self_check(df)
        return df

    def print_self_check(self, df: pd.DataFrame) -> None:
        print("\n=== URLLC 专家数据健康检查 ===")
        dist = df["label_action"].value_counts(normalize=True).sort_index() * 100.0
        for a in range(self.cfg.NUM_ACTIONS):
            print(f"动作 {a} ({self.cfg.get_action_name(a):<10}) : {dist.get(a, 0.0):.2f}%")
        if int(OffloadAction.DROP) in set(df["label_action"].unique()):
            raise RuntimeError("URLLC 数据集中出现 DROP 标签")
        illegal = [int(x) for x in df["label_action"].unique() if not self.cfg.is_action_allowed(TaskType.URLLC, int(x))]
        if illegal:
            raise RuntimeError(f"URLLC 数据集中出现非法动作: {illegal}")
        print(f"3ms 可行比例: {df['is_feasible_label'].mean() * 100:.2f}%")
        print(f"URLLC OUTAGE 比例: {df['urllc_outage'].mean() * 100:.2f}%")
        print(f"OOD 样本比例: {df['is_ood'].mean() * 100:.2f}%")
        finite = df["total_delay_sec"].replace([np.inf, -np.inf], np.nan).dropna()
        if len(finite):
            print(f"平均时延: {finite.mean() * 1e3:.6f} ms")
            print(f"P95 时延: {finite.quantile(0.95) * 1e3:.6f} ms")
            print(f"最大时延: {finite.max() * 1e3:.6f} ms")
        normal = df[df["is_ood"] == 0]
        if len(normal):
            print(f"正常场景 3ms 可行比例: {normal['is_feasible_label'].mean() * 100:.2f}%")
        ood = df[df["is_ood"] == 1]
        if len(ood):
            print(f"OOD 场景 3ms 可行比例: {ood['is_feasible_label'].mean() * 100:.2f}%")

    def save_dataset(self, df: pd.DataFrame, output_path: str | Path = "urllc_expert_dataset.csv") -> Path:
        output_path = Path(output_path)
        df.to_csv(output_path, index=False)
        print(f"URLLC 数据集已保存: {output_path}")
        return output_path


def main() -> None:
    gen = URLLCExpertDataGenerator(num_samples=HeterogeneousIoVConfig.DEFAULT_NUM_EXPERT_SAMPLES, seed=42)
    df = gen.generate_dataset()
    gen.save_dataset(df, "urllc_expert_dataset.csv")


if __name__ == "__main__":
    main()
