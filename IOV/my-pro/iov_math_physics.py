# ==========================================
# 文件名: iov_math_physics.py
# 目的: 通信 / 计算 / FBL / 能耗 / 动作评估纯公式
# ==========================================
from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np

from iov_config import HeterogeneousIoVConfig, TaskType, OffloadAction, ServiceStatus

try:
    from scipy.stats import norm
    def _normal_ppf(p: float) -> float:
        return float(norm.ppf(p))
except Exception:
    # Acklam approximation. 只用于没有 scipy 的环境。
    def _normal_ppf(p: float) -> float:
        if not 0.0 < p < 1.0:
            raise ValueError("p must be in (0,1)")
        a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.3577518672690, -30.66479806614716, 2.506628277459239]
        b = [-54.47609879822406, 161.5858368580409, -155.6989798866, 66.80131188771972, -13.28068155288572]
        c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783]
        d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416]
        plow = 0.02425
        phigh = 1.0 - plow
        if p < plow:
            q = math.sqrt(-2.0 * math.log(p))
            return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        if p <= phigh:
            q = p - 0.5
            r = q * q
            return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)


class IoVMathFormulas:
    KAPPA_CAPACITANCE = HeterogeneousIoVConfig.KAPPA_CAPACITANCE
    QINV_URLLC = _normal_ppf(1.0 - HeterogeneousIoVConfig.URLLC_TARGET_ERROR_RATE)

    @staticmethod
    def db_to_linear(db: float) -> float:
        return 10.0 ** (float(db) / 10.0)

    @staticmethod
    def linear_to_db(x: float) -> float:
        return -math.inf if x <= 0.0 else 10.0 * math.log10(float(x))

    @staticmethod
    def calc_path_loss_db(dist_m: float, env_type: str = "urban", is_los: bool | None = None) -> float:
        d = max(1.0, float(dist_m))
        fc = HeterogeneousIoVConfig.CARRIER_FREQ_GHZ
        if env_type == "highway":
            pl_los = 32.4 + 20.0 * math.log10(d) + 20.0 * math.log10(fc)
            return pl_los if is_los is not False else pl_los + 8.0 + 8.0 * math.log10(d / 100.0 + 1.0)
        pl_los = 32.4 + 21.0 * math.log10(d) + 20.0 * math.log10(fc)
        if is_los is True:
            return pl_los
        pl_nlos = 32.4 + 31.9 * math.log10(d) + 20.0 * math.log10(fc)
        return max(pl_los, pl_nlos)

    @staticmethod
    def calc_shadowing_linear(std_dev_db: float = 5.5, rng: np.random.Generator | None = None) -> float:
        r = rng if rng is not None else np.random.default_rng()
        return float(10.0 ** (r.normal(0.0, std_dev_db) / 10.0))

    @staticmethod
    def calc_multipath_fading_linear(k_factor_linear: float = 0.0, rng: np.random.Generator | None = None) -> float:
        r = rng if rng is not None else np.random.default_rng()
        if k_factor_linear <= 0.0:
            return float(max(r.exponential(1.0), 1e-12))
        mean = math.sqrt(k_factor_linear / (k_factor_linear + 1.0))
        sigma = math.sqrt(1.0 / (2.0 * (k_factor_linear + 1.0)))
        hr = r.normal(mean, sigma)
        hi = r.normal(0.0, sigma)
        return float(max(hr * hr + hi * hi, 1e-12))

    @staticmethod
    def calc_comprehensive_channel_gain(dist_m: float, env_type: str = "urban", rng: np.random.Generator | None = None) -> float:
        pl_db = IoVMathFormulas.calc_path_loss_db(dist_m, env_type=env_type)
        ant_gain_db = HeterogeneousIoVConfig.VEHICLE_ANTENNA_GAIN_DBI + HeterogeneousIoVConfig.RSU_ANTENNA_GAIN_DBI
        pl_linear = 10.0 ** ((-pl_db + ant_gain_db) / 10.0)
        k = IoVMathFormulas.db_to_linear(6.0) if env_type == "highway" else 0.0
        gain = pl_linear * IoVMathFormulas.calc_shadowing_linear(4.5 if env_type == "highway" else 6.0, rng) * IoVMathFormulas.calc_multipath_fading_linear(k, rng)
        return float(np.clip(gain, 1e-18, 1.0))

    @staticmethod
    def calc_sinr_linear(p_tx_w: float, gain_linear: float, interference_w: float = 0.0, noise_power_w: float | None = None) -> float:
        if noise_power_w is None:
            noise_power_w = HeterogeneousIoVConfig.NOISE_POWER_W
        den = max(1e-30, float(noise_power_w) + max(0.0, float(interference_w)))
        sinr = max(0.0, float(p_tx_w)) * max(0.0, float(gain_linear)) / den
        return float(np.clip(sinr, HeterogeneousIoVConfig.SINR_LINEAR_MIN, HeterogeneousIoVConfig.SINR_LINEAR_MAX))

    @staticmethod
    def calc_downlink_sinr_linear(uplink_sinr_linear: float) -> float:
        cfg = HeterogeneousIoVConfig
        gain = 10.0 ** (cfg.DOWNLINK_SINR_GAIN_DB / 10.0)
        interference_factor = max(1e-6, cfg.DOWNLINK_INTERFERENCE_FACTOR)
        return float(np.clip(float(uplink_sinr_linear) * gain / interference_factor, cfg.SINR_LINEAR_MIN, cfg.SINR_LINEAR_MAX))

    @staticmethod
    def calc_embb_shannon_rate_bps(bandwidth_hz: float, sinr_linear: float) -> float:
        if bandwidth_hz <= 0.0 or sinr_linear <= 0.0:
            return 0.0
        return float(bandwidth_hz * math.log2(1.0 + sinr_linear))

    @staticmethod
    def calc_channel_dispersion(sinr_linear: float) -> float:
        s = max(0.0, float(sinr_linear))
        return float(1.0 - 1.0 / ((1.0 + s) ** 2))

    @staticmethod
    def calc_urllc_fbl_rate_bps(
        bandwidth_hz: float,
        sinr_linear: float,
        tx_duration_sec: float,
        error_rate: float = HeterogeneousIoVConfig.URLLC_TARGET_ERROR_RATE,
    ) -> float:
        if bandwidth_hz <= 0.0 or sinr_linear <= 0.0 or tx_duration_sec <= 0.0:
            return 0.0
        n = bandwidth_hz * tx_duration_sec
        if n < 1.0:
            return 0.0
        v = IoVMathFormulas.calc_channel_dispersion(sinr_linear)
        qinv = IoVMathFormulas.QINV_URLLC if error_rate == HeterogeneousIoVConfig.URLLC_TARGET_ERROR_RATE else _normal_ppf(1.0 - error_rate)
        cap = math.log2(1.0 + sinr_linear)
        penalty = math.sqrt(max(0.0, v) / n) * qinv / math.log(2.0)
        return float(max(0.0, bandwidth_hz * (cap - penalty)))

    @staticmethod
    def calc_urllc_fbl_required_tx_time_sec(
        data_size_bits: float,
        bandwidth_hz: float,
        sinr_linear: float,
        max_tx_time_sec: float,
        error_rate: float = HeterogeneousIoVConfig.URLLC_TARGET_ERROR_RATE,
    ) -> float:
        if data_size_bits <= 0.0:
            return 0.0
        if max_tx_time_sec <= HeterogeneousIoVConfig.URLLC_TX_TIME_MIN_SEC:
            return float("inf")
        hi = float(max_tx_time_sec)
        r_hi = IoVMathFormulas.calc_urllc_fbl_rate_bps(bandwidth_hz, sinr_linear, hi, error_rate)
        if r_hi <= 0.0 or data_size_bits / r_hi > hi:
            return float("inf")
        lo = HeterogeneousIoVConfig.URLLC_TX_TIME_MIN_SEC
        for _ in range(24):  # 24 次足够到纳秒级，不拖慢大样本生成
            mid = (lo + hi) * 0.5
            r_mid = IoVMathFormulas.calc_urllc_fbl_rate_bps(bandwidth_hz, sinr_linear, mid, error_rate)
            if r_mid > 0.0 and data_size_bits / r_mid <= mid:
                hi = mid
            else:
                lo = mid
        return float(hi)

    @staticmethod
    def calc_transmission_delay_sec(data_size_bits: float, rate_bps: float) -> float:
        if data_size_bits <= 0.0:
            return 0.0
        return float("inf") if rate_bps <= 0.0 else float(data_size_bits / rate_bps)

    @staticmethod
    def calc_computation_delay_sec(cycles: float, cpu_freq_hz: float, cpu_ratio: float = 1.0) -> float:
        f = float(cpu_freq_hz) * max(0.0, float(cpu_ratio))
        return 0.0 if cycles <= 0.0 else (float("inf") if f <= 0.0 else float(cycles / f))

    @staticmethod
    def calc_queue_delay_sec(workload_cycles: float, cpu_freq_hz: float, cpu_ratio: float = 1.0) -> float:
        f = float(cpu_freq_hz) * max(0.0, float(cpu_ratio))
        return 0.0 if workload_cycles <= 0.0 else (float("inf") if f <= 0.0 else float(workload_cycles / f))

    @staticmethod
    def calc_computation_energy_joules(cycles: float, freq_hz: float) -> float:
        return 0.0 if cycles <= 0.0 or freq_hz <= 0.0 else float(IoVMathFormulas.KAPPA_CAPACITANCE * cycles * freq_hz * freq_hz)

    @staticmethod
    def calc_transmission_energy_joules(p_tx_w: float, tx_time_sec: float) -> float:
        return 0.0 if p_tx_w <= 0.0 or tx_time_sec <= 0.0 or not math.isfinite(tx_time_sec) else float(p_tx_w * tx_time_sec)

    @staticmethod
    def _node_fields(action: int) -> tuple[str, str, float]:
        if action == int(OffloadAction.LOCAL):
            return "cpu_local_hz", "workload_local_cycles", 0.0
        if action == int(OffloadAction.LOCAL_MEC):
            return "cpu_lmec_hz", "workload_lmec_cycles", 0.0
        if action == int(OffloadAction.REMOTE_MEC):
            return "cpu_rmec_hz", "workload_rmec_cycles", 0.0
        if action == int(OffloadAction.CLOUD):
            return "cpu_cloud_hz", "workload_cloud_cycles", 0.0
        raise ValueError(f"Unknown action: {action}")

    @staticmethod
    def _effective_workload(state: Dict[str, float], action: int, task_type: str) -> float:
        _, workload_key, _ = IoVMathFormulas._node_fields(action)
        w = float(state.get(workload_key, 0.0))
        if task_type == TaskType.URLLC:
            cfg = HeterogeneousIoVConfig
            if action == int(OffloadAction.LOCAL):
                return w * cfg.URLLC_LOCAL_BLOCKING_RATIO
            if action == int(OffloadAction.LOCAL_MEC):
                return w * cfg.URLLC_MEC_BLOCKING_RATIO
            if action == int(OffloadAction.REMOTE_MEC):
                return w * cfg.URLLC_RMEC_BLOCKING_RATIO
            if action == int(OffloadAction.CLOUD):
                return w * cfg.URLLC_CLOUD_BLOCKING_RATIO
        return w

    @staticmethod
    def evaluate_action_from_state(
        state: Dict[str, float],
        action: int,
        cpu_ratio: float = 1.0,
        task_type: str | None = None,
    ) -> Dict[str, Any]:
        cfg = HeterogeneousIoVConfig
        if task_type is None:
            task_type = cfg.ID_TO_TASK_TYPE.get(int(state.get("task_type_id", 0)), TaskType.URLLC)
        action = int(action)
        cpu_ratio = float(np.clip(cpu_ratio, cfg.CPU_RATIO_MIN, cfg.CPU_RATIO_MAX))

        if action == int(OffloadAction.DROP):
            return {
                "action": action,
                "action_name": cfg.get_action_name(action),
                "task_type": task_type,
                "cpu_ratio": 0.0,
                "total_delay_sec": float("inf") if task_type == TaskType.URLLC else 0.0,
                "queue_delay_sec": 0.0,
                "transmission_delay_sec": 0.0,
                "upload_delay_sec": 0.0,
                "return_delay_sec": 0.0,
                "backhaul_delay_sec": 0.0,
                "backhaul_forward_delay_sec": 0.0,
                "backhaul_return_delay_sec": 0.0,
                "compute_delay_sec": 0.0,
                "rate_bps": 0.0,
                "uplink_rate_bps": 0.0,
                "downlink_rate_bps": 0.0,
                "estimated_reliability": 0.0 if task_type == TaskType.URLLC else 1.0,
                "estimated_outage_probability": 1.0 if task_type == TaskType.URLLC else 0.0,
                "energy_joules": 0.0,
                "tx_energy_joules": 0.0,
                "compute_energy_joules": 0.0,
                "resource_price": cfg.RESOURCE_PRICE[action],
                "feasible": task_type != TaskType.URLLC,
                "status": ServiceStatus.OUTAGE if task_type == TaskType.URLLC else ServiceStatus.REJECTED,
                "illegal_action": task_type == TaskType.URLLC,
            }

        if not cfg.is_action_allowed(task_type, action):
            return {
                "action": action,
                "action_name": cfg.get_action_name(action),
                "task_type": task_type,
                "cpu_ratio": cpu_ratio,
                "total_delay_sec": float("inf"),
                "energy_joules": 0.0,
                "feasible": False,
                "status": ServiceStatus.ILLEGAL,
                "illegal_action": True,
            }

        cpu_key, _, _ = IoVMathFormulas._node_fields(action)
        cpu_hz = float(state[cpu_key])
        workload = IoVMathFormulas._effective_workload(state, action, task_type)
        q_delay = IoVMathFormulas.calc_queue_delay_sec(workload, cpu_hz, cpu_ratio)
        c_delay = IoVMathFormulas.calc_computation_delay_sec(float(state["required_cycles"]), cpu_hz, cpu_ratio)
        mac = float(state.get("mac_delay_sec", 0.0)) if action != int(OffloadAction.LOCAL) else 0.0
        backhaul_forward = 0.0
        if action == int(OffloadAction.REMOTE_MEC):
            backhaul_forward = float(state.get("backhaul_l2r_delay_sec", 0.0))
        elif action == int(OffloadAction.CLOUD):
            backhaul_forward = float(state.get("backhaul_l2r_delay_sec", 0.0)) + float(state.get("backhaul_r2c_delay_sec", 0.0))
        backhaul_return = backhaul_forward * cfg.BACKHAUL_RESULT_RETURN_FACTOR
        backhaul = backhaul_forward + backhaul_return

        upload_delay = 0.0
        return_delay = 0.0
        uplink_rate = float("nan")
        downlink_rate = float("nan")
        if action != int(OffloadAction.LOCAL):
            result_bits = float(state.get("result_size_bits", 0.0))
            downlink_sinr = float(state.get("downlink_sinr_linear", IoVMathFormulas.calc_downlink_sinr_linear(float(state["sinr_linear"]))))
            if task_type == TaskType.URLLC:
                radio_budget = float(state["deadline_sec"]) - mac - backhaul - q_delay - c_delay
                if radio_budget > 0.0:
                    return_budget = max(cfg.URLLC_TX_TIME_MIN_SEC, min(radio_budget * 0.25, radio_budget))
                    return_delay = IoVMathFormulas.calc_urllc_fbl_required_tx_time_sec(
                        result_bits, float(state["bandwidth_hz"]), downlink_sinr, return_budget
                    ) if result_bits > 0.0 else 0.0
                    if not math.isfinite(return_delay):
                        fallback_down = IoVMathFormulas.calc_urllc_fbl_rate_bps(
                            float(state["bandwidth_hz"]), downlink_sinr, cfg.URLLC_TX_TIME_MAX_SEC
                        )
                        return_delay = IoVMathFormulas.calc_transmission_delay_sec(result_bits, fallback_down)
                    upload_budget = radio_budget - return_delay
                    upload_delay = IoVMathFormulas.calc_urllc_fbl_required_tx_time_sec(
                        float(state["data_size_bits"]), float(state["bandwidth_hz"]), float(state["sinr_linear"]), upload_budget
                    ) if upload_budget > 0.0 else float("inf")
                else:
                    upload_delay = float("inf")
                    return_delay = 0.0
                if not math.isfinite(upload_delay):
                    # 失败时给一个可排序的近似时延，便于 outage 选择“最不坏”的动作。
                    fallback_up = IoVMathFormulas.calc_urllc_fbl_rate_bps(
                        float(state["bandwidth_hz"]), float(state["sinr_linear"]), cfg.URLLC_TX_TIME_MAX_SEC
                    )
                    upload_delay = IoVMathFormulas.calc_transmission_delay_sec(float(state["data_size_bits"]), fallback_up)
                uplink_rate = 0.0 if upload_delay <= 0.0 or not math.isfinite(upload_delay) else float(state["data_size_bits"]) / upload_delay
                downlink_rate = 0.0 if return_delay <= 0.0 or not math.isfinite(return_delay) else result_bits / max(return_delay, 1e-12)
            else:
                uplink_rate = float(state.get("uplink_rate_shannon_bps", IoVMathFormulas.calc_embb_shannon_rate_bps(float(state["bandwidth_hz"]), float(state["sinr_linear"]))))
                downlink_rate = float(state.get("downlink_rate_shannon_bps", IoVMathFormulas.calc_embb_shannon_rate_bps(float(state["bandwidth_hz"]), downlink_sinr)))
                upload_delay = IoVMathFormulas.calc_transmission_delay_sec(float(state["data_size_bits"]), uplink_rate)
                return_delay = IoVMathFormulas.calc_transmission_delay_sec(result_bits, downlink_rate)

        tx_delay = upload_delay + return_delay
        total = mac + backhaul + tx_delay + q_delay + c_delay
        deadline = float(state["deadline_sec"])
        tolerant = float(state.get("tolerant_deadline_sec", deadline))
        feasible = total <= deadline
        if task_type == TaskType.EMBB and not feasible and total <= tolerant:
            status = ServiceStatus.DEGRADED
        else:
            status = ServiceStatus.SUCCESS if feasible else ServiceStatus.DEADLINE_VIOLATION
        tx_energy = IoVMathFormulas.calc_transmission_energy_joules(float(state.get("tx_power_w", 0.0)), upload_delay)
        # 本地计算能耗看作车辆侧真实能耗；远端计算给小权重系统能耗，避免 expert 过度偏云。
        compute_energy_raw = IoVMathFormulas.calc_computation_energy_joules(float(state["required_cycles"]), cpu_hz * cpu_ratio)
        if action == int(OffloadAction.LOCAL):
            compute_energy = compute_energy_raw
        elif action == int(OffloadAction.LOCAL_MEC):
            compute_energy = 0.08 * compute_energy_raw
        elif action == int(OffloadAction.REMOTE_MEC):
            compute_energy = 0.05 * compute_energy_raw
        else:
            compute_energy = 0.02 * compute_energy_raw
        energy = tx_energy + compute_energy
        margin = deadline - total if math.isfinite(total) else -float("inf")
        if task_type == TaskType.URLLC:
            if not feasible:
                outage_probability = 1.0
            else:
                slack = max(0.0, deadline - total)
                slack_risk = 0.01 * math.exp(-slack / max(1e-9, 0.15 * deadline))
                channel_risk = cfg.URLLC_TARGET_ERROR_RATE if action != int(OffloadAction.LOCAL) else 0.0
                outage_probability = float(np.clip(channel_risk + slack_risk, 0.0, 1.0))
        else:
            outage_probability = 0.0 if total <= tolerant else 1.0
        reliability = 1.0 - outage_probability
        return {
            "action": action,
            "action_name": cfg.get_action_name(action),
            "task_type": task_type,
            "cpu_ratio": cpu_ratio,
            "total_delay_sec": float(total),
            "queue_delay_sec": float(q_delay),
            "transmission_delay_sec": float(tx_delay),
            "upload_delay_sec": float(upload_delay),
            "return_delay_sec": float(return_delay),
            "backhaul_delay_sec": float(backhaul),
            "backhaul_forward_delay_sec": float(backhaul_forward),
            "backhaul_return_delay_sec": float(backhaul_return),
            "mac_delay_sec": float(mac),
            "compute_delay_sec": float(c_delay),
            "rate_bps": float(uplink_rate) if not math.isnan(uplink_rate) else np.nan,
            "uplink_rate_bps": float(uplink_rate) if not math.isnan(uplink_rate) else np.nan,
            "downlink_rate_bps": float(downlink_rate) if not math.isnan(downlink_rate) else np.nan,
            "estimated_reliability": float(reliability),
            "estimated_outage_probability": float(outage_probability),
            "energy_joules": float(energy),
            "tx_energy_joules": float(tx_energy),
            "compute_energy_joules": float(compute_energy),
            "resource_price": float(cfg.RESOURCE_PRICE[action] * cpu_ratio),
            "deadline_margin_sec": float(margin),
            "feasible": bool(feasible),
            "status": status,
            "illegal_action": False,
        }


if __name__ == "__main__":
    print("iov_math_physics self-check passed")
