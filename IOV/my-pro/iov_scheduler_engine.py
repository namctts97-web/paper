# ==========================================
# 文件名: iov_scheduler_engine.py
# 目的: 文献启发式 expert oracle
#   URLLC: safety-first lexicographic oracle, no DROP label
#   eMBB : QoS-aware utility oracle with admission control
# ==========================================
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from iov_config import HeterogeneousIoVConfig, TaskType, OffloadAction, ServiceStatus
from iov_math_physics import IoVMathFormulas


class OffloadingDecisionEngine:
    @staticmethod
    def infer_task_type_from_state(state: Dict[str, float]) -> str:
        return HeterogeneousIoVConfig.ID_TO_TASK_TYPE.get(int(state.get("task_type_id", 0)), TaskType.URLLC)

    @staticmethod
    def is_action_allowed(task_type: str, action: int) -> bool:
        return HeterogeneousIoVConfig.is_action_allowed(task_type, int(action))

    @staticmethod
    def evaluate_action(state: Dict[str, float], action: int, cpu_ratio: float = 1.0, task_type: Optional[str] = None, enforce_action_mask: bool = True) -> Dict[str, Any]:
        if task_type is None:
            task_type = OffloadingDecisionEngine.infer_task_type_from_state(state)
        if enforce_action_mask and not OffloadingDecisionEngine.is_action_allowed(task_type, int(action)):
            return {
                "action": int(action),
                "action_name": HeterogeneousIoVConfig.get_action_name(int(action)),
                "task_type": task_type,
                "cpu_ratio": float(cpu_ratio),
                "feasible": False,
                "illegal_action": True,
                "status": ServiceStatus.ILLEGAL,
                "total_delay_sec": float("inf"),
                "energy_joules": 0.0,
            }
        return IoVMathFormulas.evaluate_action_from_state(state, int(action), cpu_ratio, task_type)

    @staticmethod
    def enumerate_candidates(state: Dict[str, float], task_type: str, actions: List[int], cpu_ratio_grid: Tuple[float, ...]) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        for a in actions:
            if a == int(OffloadAction.DROP):
                candidates.append(OffloadingDecisionEngine.evaluate_action(state, a, 0.0, task_type, True))
                continue
            for r in cpu_ratio_grid:
                cand = OffloadingDecisionEngine.evaluate_action(state, a, float(r), task_type, True)
                if cand.get("illegal_action", False):
                    continue
                candidates.append(cand)
        return candidates

    @staticmethod
    def _safe_float(x: Any, default: float = 0.0) -> float:
        try:
            y = float(x)
            return y if math.isfinite(y) else default
        except Exception:
            return default

    @staticmethod
    def _status_id(status: Any) -> int:
        if status == ServiceStatus.SUCCESS:
            return 1
        if status == ServiceStatus.DEGRADED:
            return 2
        if status == ServiceStatus.OUTAGE:
            return -2
        if status == ServiceStatus.ILLEGAL:
            return -3
        return 0

    @staticmethod
    def _attach_candidate_summaries(best: Dict[str, Any], candidates: List[Dict[str, Any]], score_key: str) -> Dict[str, Any]:
        for action in [int(OffloadAction.LOCAL), int(OffloadAction.LOCAL_MEC), int(OffloadAction.REMOTE_MEC), int(OffloadAction.CLOUD)]:
            action_candidates = [c for c in candidates if int(c.get("action", -1)) == action]
            if not action_candidates:
                continue
            chosen = max(action_candidates, key=lambda c: c.get(score_key, -1e18))
            prefix = f"candidate_{HeterogeneousIoVConfig.get_action_name(action).lower()}"
            best[f"{prefix}_delay_sec"] = OffloadingDecisionEngine._safe_float(chosen.get("total_delay_sec"), np.inf)
            best[f"{prefix}_cpu_ratio"] = OffloadingDecisionEngine._safe_float(chosen.get("cpu_ratio"), 0.0)
            best[f"{prefix}_energy_joules"] = OffloadingDecisionEngine._safe_float(chosen.get("energy_joules"), np.nan)
            best[f"{prefix}_queue_delay_sec"] = OffloadingDecisionEngine._safe_float(chosen.get("queue_delay_sec"), np.nan)
            best[f"{prefix}_upload_delay_sec"] = OffloadingDecisionEngine._safe_float(chosen.get("upload_delay_sec"), np.nan)
            best[f"{prefix}_return_delay_sec"] = OffloadingDecisionEngine._safe_float(chosen.get("return_delay_sec"), np.nan)
            best[f"{prefix}_backhaul_delay_sec"] = OffloadingDecisionEngine._safe_float(chosen.get("backhaul_delay_sec"), np.nan)
            best[f"{prefix}_compute_delay_sec"] = OffloadingDecisionEngine._safe_float(chosen.get("compute_delay_sec"), np.nan)
            best[f"{prefix}_reliability"] = OffloadingDecisionEngine._safe_float(chosen.get("estimated_reliability"), np.nan)
            best[f"{prefix}_outage_probability"] = OffloadingDecisionEngine._safe_float(chosen.get("estimated_outage_probability"), np.nan)
            best[f"{prefix}_score"] = OffloadingDecisionEngine._safe_float(chosen.get(score_key), np.nan)
            best[f"{prefix}_feasible"] = int(bool(chosen.get("feasible", False)))
            best[f"{prefix}_status_id"] = OffloadingDecisionEngine._status_id(chosen.get("status"))
        return best

    @staticmethod
    def _annotate_urllc_candidate(c: Dict[str, Any]) -> Dict[str, Any]:
        # 安全优先：可行性 > deadline margin > 可靠性风险 > 能耗/资源代价。
        delay = OffloadingDecisionEngine._safe_float(c.get("total_delay_sec"), 1e9)
        margin_ms = (HeterogeneousIoVConfig.URLLC_MAX_DELAY_SEC - delay) * 1e3
        energy = OffloadingDecisionEngine._safe_float(c.get("energy_joules"), 0.0)
        price = OffloadingDecisionEngine._safe_float(c.get("resource_price"), 0.0)
        reliability = OffloadingDecisionEngine._safe_float(c.get("estimated_reliability"), 0.0)
        action = int(c.get("action", -1))
        remote_risk = {0: 0.00, 1: 0.04, 2: 0.18, 3: 0.45}.get(action, 1.0)
        # margin 是主导项；代价只用于 margin 接近时破平局。
        c["safety_score"] = margin_ms + 0.25 * reliability - 0.04 * math.log1p(max(0.0, energy)) - 0.06 * price - remote_risk
        c["deadline_margin_ms"] = margin_ms
        return c

    @staticmethod
    def select_urllc_expert_action(state: Dict[str, float], cpu_ratio_grid: Optional[Tuple[float, ...]] = None) -> Dict[str, Any]:
        cfg = HeterogeneousIoVConfig
        if cpu_ratio_grid is None:
            cpu_ratio_grid = cfg.URLLC_CPU_RATIO_GRID
        actions = [int(OffloadAction.LOCAL), int(OffloadAction.LOCAL_MEC), int(OffloadAction.REMOTE_MEC), int(OffloadAction.CLOUD)]
        candidates = [OffloadingDecisionEngine._annotate_urllc_candidate(c) for c in OffloadingDecisionEngine.enumerate_candidates(state, TaskType.URLLC, actions, cpu_ratio_grid)]
        if not candidates:
            raise RuntimeError("No URLLC candidates")
        feasible = [c for c in candidates if bool(c.get("feasible", False)) and c["total_delay_sec"] <= cfg.URLLC_MAX_DELAY_SEC]
        if feasible:
            # 先按 safety_score 选择动作，再在同动作里用较低 ratio 减少资源浪费。
            best = max(feasible, key=lambda c: (c["safety_score"], -c["resource_price"]))
            best["expert_status"] = ServiceStatus.SUCCESS
            best["is_feasible_label"] = True
            best["deadline_violation"] = False
            best["urllc_outage"] = False
        else:
            # URLLC 不能 DROP。若不可行，仍输出最小总时延动作，并标 OUTAGE，供残差 RL/安全层学习。
            best = min(candidates, key=lambda c: (OffloadingDecisionEngine._safe_float(c.get("total_delay_sec"), 1e9), c.get("resource_price", 0.0)))
            best["expert_status"] = ServiceStatus.OUTAGE
            best["status"] = ServiceStatus.OUTAGE
            best["is_feasible_label"] = False
            best["deadline_violation"] = True
            best["urllc_outage"] = True
        best["label_action"] = int(best["action"])
        best["label_cpu_ratio"] = float(best.get("cpu_ratio", 1.0))
        if best["label_action"] == int(OffloadAction.DROP):
            raise RuntimeError("URLLC expert produced DROP")
        best["num_candidates"] = len(candidates)
        best["num_feasible_candidates"] = len(feasible)
        second = sorted(candidates, key=lambda c: c.get("safety_score", -1e9), reverse=True)[1] if len(candidates) > 1 else None
        best["second_best_action"] = int(second["action"]) if second else -1
        best["expert_margin_to_second"] = float(best.get("safety_score", 0.0) - second.get("safety_score", 0.0)) if second else np.nan
        return OffloadingDecisionEngine._attach_candidate_summaries(best, candidates, "safety_score")

    @staticmethod
    def _annotate_embb_candidate(c: Dict[str, Any], state: Dict[str, float]) -> Dict[str, Any]:
        action = int(c.get("action", -1))
        if action == int(OffloadAction.DROP):
            c["utility_score"] = -0.15
            return c
        delay = OffloadingDecisionEngine._safe_float(c.get("total_delay_sec"), 1e9)
        deadline = float(state["deadline_sec"])
        tolerant = float(state.get("tolerant_deadline_sec", deadline))
        bits = float(state["data_size_bits"])
        energy = max(0.0, OffloadingDecisionEngine._safe_float(c.get("energy_joules"), 0.0))
        price = max(0.0, OffloadingDecisionEngine._safe_float(c.get("resource_price"), 0.0))
        delay_ratio = delay / max(1e-9, deadline)
        tolerant_ratio = delay / max(1e-9, tolerant)
        goodput_mbps = (bits / max(delay, 1e-6)) / 1e6 if math.isfinite(delay) else 0.0
        throughput_utility = math.log1p(goodput_mbps) / math.log1p(120.0)
        # deadline 内轻罚；降级区重罚；超过 tolerant 基本无效。
        qos_penalty = 0.15 * min(delay_ratio, 1.0) + 0.85 * max(0.0, delay_ratio - 1.0) + 2.5 * max(0.0, tolerant_ratio - 1.0)
        energy_penalty = 0.045 * math.log1p(energy)
        resource_penalty = 0.22 * price
        remote_risk = {0: 0.00, 1: 0.02, 2: 0.05, 3: 0.09}.get(action, 0.2)
        ood_risk = 0.04 * float(state.get("is_ood", 0.0)) * (1.0 if action in {2, 3} else 0.4)
        c["utility_score"] = throughput_utility - qos_penalty - energy_penalty - resource_penalty - remote_risk - ood_risk
        c["delay_ratio"] = delay_ratio
        c["tolerant_ratio"] = tolerant_ratio
        c["goodput_mbps"] = goodput_mbps
        if delay <= deadline:
            c["status"] = ServiceStatus.SUCCESS
            c["is_feasible_label"] = True
        elif delay <= tolerant:
            c["status"] = ServiceStatus.DEGRADED
            c["is_feasible_label"] = True
        else:
            c["status"] = ServiceStatus.DEADLINE_VIOLATION
            c["is_feasible_label"] = False
        return c

    @staticmethod
    def select_embb_expert_action(
        state: Dict[str, float],
        cpu_ratio_grid: Optional[Tuple[float, ...]] = None,
        objective: str = "utility",
    ) -> Dict[str, Any]:
        cfg = HeterogeneousIoVConfig
        if cpu_ratio_grid is None:
            cpu_ratio_grid = cfg.EMBB_CPU_RATIO_GRID
        actions = [int(OffloadAction.LOCAL), int(OffloadAction.LOCAL_MEC), int(OffloadAction.REMOTE_MEC), int(OffloadAction.CLOUD)]
        candidates = [OffloadingDecisionEngine._annotate_embb_candidate(c, state) for c in OffloadingDecisionEngine.enumerate_candidates(state, TaskType.EMBB, actions, cpu_ratio_grid)]
        useful = [c for c in candidates if bool(c.get("is_feasible_label", False))]
        if useful:
            if objective == "delay":
                best = min(useful, key=lambda c: c["total_delay_sec"])
            elif objective == "energy":
                best = min(useful, key=lambda c: c["energy_joules"] + c.get("resource_price", 0.0))
            else:
                best = max(useful, key=lambda c: c["utility_score"])
            # 如果 utility 过低，说明接入会拖垮资源，DROP。
            if best.get("utility_score", 0.0) < -0.85:
                best = IoVMathFormulas.evaluate_action_from_state(state, int(OffloadAction.DROP), 0.0, TaskType.EMBB)
                best["utility_score"] = -0.15
                best["expert_status"] = ServiceStatus.REJECTED
                best["is_feasible_label"] = False
                best["deadline_violation"] = True
            else:
                best["expert_status"] = best.get("status", ServiceStatus.SUCCESS)
                best["deadline_violation"] = best["total_delay_sec"] > float(state["deadline_sec"])
        else:
            best = IoVMathFormulas.evaluate_action_from_state(state, int(OffloadAction.DROP), 0.0, TaskType.EMBB)
            best["utility_score"] = -0.15
            best["expert_status"] = ServiceStatus.REJECTED
            best["is_feasible_label"] = False
            best["deadline_violation"] = True
        best["label_action"] = int(best["action"])
        best["label_cpu_ratio"] = float(best.get("cpu_ratio", 0.0))
        best["num_candidates"] = len(candidates)
        best["num_feasible_candidates"] = len(useful)
        ranked = sorted(candidates, key=lambda c: c.get("utility_score", -1e9), reverse=True)
        second = ranked[1] if len(ranked) > 1 else None
        best["second_best_action"] = int(second["action"]) if second else -1
        best["expert_margin_to_second"] = float(best.get("utility_score", 0.0) - second.get("utility_score", 0.0)) if second else np.nan
        return OffloadingDecisionEngine._attach_candidate_summaries(best, candidates, "utility_score")

    @staticmethod
    def select_expert_action(state: Dict[str, float], task_type: Optional[str] = None, objective: str = "auto") -> Dict[str, Any]:
        if task_type is None:
            task_type = OffloadingDecisionEngine.infer_task_type_from_state(state)
        if task_type == TaskType.URLLC:
            return OffloadingDecisionEngine.select_urllc_expert_action(state)
        return OffloadingDecisionEngine.select_embb_expert_action(state, objective="utility" if objective == "auto" else objective)


if __name__ == "__main__":
    from iov_env_factory import IoVEnvironmentFactory
    f = IoVEnvironmentFactory(seed=42)
    snap = f.generate_snapshot(TaskType.URLLC, include_ood=False)
    print("iov_scheduler_engine self-check passed", OffloadingDecisionEngine.select_urllc_expert_action(snap.state_dict)["label_action"])
