# ==========================================
# 文件名: iov_env_factory.py
# 目的: 生成 IoV 环境快照 / 任务 / 队列 / OOD 状态
# ==========================================
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from iov_config import HeterogeneousIoVConfig, TaskType, ScenarioType
from iov_math_physics import IoVMathFormulas


@dataclass
class TaskInstance:
    task_id: str
    task_type: str
    data_size_bits: float
    required_cycles: float
    max_delay_sec: float
    arrival_time_sec: float = 0.0
    deadline_time_sec: Optional[float] = None
    result_size_bits: float = 0.0
    tolerant_delay_sec: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.deadline_time_sec is None:
            self.deadline_time_sec = self.arrival_time_sec + self.max_delay_sec
        if self.tolerant_delay_sec is None:
            self.tolerant_delay_sec = self.max_delay_sec


@dataclass
class NodeComputeInstance:
    node_id: str
    node_type: str
    cpu_freq_ghz: float
    available_cpu_ratio: float = 1.0

    @property
    def cpu_freq_hz(self) -> float:
        return self.cpu_freq_ghz * 1e9

    @property
    def available_cpu_hz(self) -> float:
        return self.cpu_freq_hz * self.available_cpu_ratio


@dataclass
class OODState:
    is_ood: bool = False
    ood_type: str = "none"
    start_time_sec: float = 0.0
    duration_sec: float = 0.0
    deep_fade_loss_db: float = 0.0
    backhaul_delay_multiplier: float = 1.0
    mec_cpu_reduction_ratio: float = 0.0
    speed_multiplier: float = 1.0
    queue_multiplier: float = 1.0


@dataclass
class VehicleState:
    vehicle_id: str
    scenario: str
    speed_mps: float
    acceleration_mps2: float
    distance_to_rsu_m: float
    lane_id: int
    position_x_m: float = 0.0
    position_y_m: float = 0.0


@dataclass
class ChannelState:
    distance_m: float
    bandwidth_hz: float
    tx_power_w: float
    noise_power_w: float
    interference_w: float
    channel_gain_linear: float
    sinr_linear: float
    downlink_sinr_linear: float
    uplink_rate_shannon_bps: float
    downlink_rate_shannon_bps: float
    env_type: str = "urban"


@dataclass
class BackhaulState:
    lmec_to_rmec_delay_sec: float
    rmec_to_cloud_delay_sec: float


@dataclass
class QueueState:
    queue_local_len: int = 0
    queue_lmec_len: int = 0
    queue_rmec_len: int = 0
    queue_cloud_len: int = 0
    workload_local_cycles: float = 0.0
    workload_lmec_cycles: float = 0.0
    workload_rmec_cycles: float = 0.0
    workload_cloud_cycles: float = 0.0


@dataclass
class EnvironmentSnapshot:
    task: TaskInstance
    vehicle: VehicleState
    channel: ChannelState
    compute_nodes: Dict[str, NodeComputeInstance]
    backhaul: BackhaulState
    queue: QueueState
    ood: OODState
    state_dict: Dict[str, float]


class IoVEnvironmentFactory:
    def __init__(self, seed: int = HeterogeneousIoVConfig.DEFAULT_SEED, scenario: str = HeterogeneousIoVConfig.DEFAULT_SCENARIO):
        self.rng = np.random.default_rng(int(seed))
        self.cfg = HeterogeneousIoVConfig()
        if scenario not in self.cfg.SUPPORTED_SCENARIOS:
            raise ValueError(f"Unsupported scenario: {scenario}")
        self.scenario = scenario

    def _u(self, a: float, b: float) -> float:
        return float(self.rng.uniform(a, b))

    def _nclip(self, mu: float, sigma: float, lo: float, hi: float) -> float:
        return float(np.clip(self.rng.normal(mu, sigma), lo, hi))

    def _logu(self, lo: float, hi: float) -> float:
        return float(np.exp(self.rng.uniform(np.log(lo), np.log(hi))))

    def generate_vehicle_cpu(self) -> float:
        return self._u(self.cfg.VEHICLE_CPU_GHZ_MIN, self.cfg.VEHICLE_CPU_GHZ_MAX)

    def generate_local_mec_cpu(self) -> float:
        return self._u(self.cfg.LOCAL_MEC_CPU_GHZ_MIN, self.cfg.LOCAL_MEC_CPU_GHZ_MAX)

    def generate_remote_mec_cpu(self) -> float:
        return self._u(self.cfg.REMOTE_MEC_CPU_GHZ_MIN, self.cfg.REMOTE_MEC_CPU_GHZ_MAX)

    def generate_cloud_cpu(self) -> float:
        return self._u(self.cfg.CLOUD_CPU_GHZ_MIN, self.cfg.CLOUD_CPU_GHZ_MAX)

    def generate_ood_state(self, current_time_sec: float = 0.0, force_type: Optional[str] = None, prob: float | None = None) -> OODState:
        if prob is None:
            prob = self.cfg.OOD_PROB_DEFAULT
        if force_type is None and self.rng.random() >= prob:
            return OODState(False, self.cfg.OOD_TYPE_NONE, current_time_sec, 0.0)
        ood_type = force_type or str(self.rng.choice([x for x in self.cfg.SUPPORTED_OOD_TYPES if x != self.cfg.OOD_TYPE_NONE]))
        return OODState(
            is_ood=True,
            ood_type=ood_type,
            start_time_sec=current_time_sec,
            duration_sec=self._u(self.cfg.OOD_DURATION_SEC_MIN, self.cfg.OOD_DURATION_SEC_MAX),
            deep_fade_loss_db=self._u(self.cfg.OOD_DEEP_FADE_LOSS_DB_MIN, self.cfg.OOD_DEEP_FADE_LOSS_DB_MAX) if ood_type in {self.cfg.OOD_TYPE_CHANNEL_DEEP_FADE, self.cfg.OOD_TYPE_MIXED} else 0.0,
            backhaul_delay_multiplier=self._u(self.cfg.OOD_BACKHAUL_DELAY_MULTIPLIER_MIN, self.cfg.OOD_BACKHAUL_DELAY_MULTIPLIER_MAX) if ood_type in {self.cfg.OOD_TYPE_BACKHAUL_SPIKE, self.cfg.OOD_TYPE_MIXED} else 1.0,
            mec_cpu_reduction_ratio=self._u(self.cfg.OOD_MEC_CPU_REDUCTION_MIN, self.cfg.OOD_MEC_CPU_REDUCTION_MAX) if ood_type in {self.cfg.OOD_TYPE_MEC_OVERLOAD, self.cfg.OOD_TYPE_MIXED} else 0.0,
            speed_multiplier=self._u(self.cfg.OOD_SPEED_MULTIPLIER_MIN, self.cfg.OOD_SPEED_MULTIPLIER_MAX) if ood_type in {self.cfg.OOD_TYPE_HIGH_MOBILITY, self.cfg.OOD_TYPE_MIXED} else 1.0,
            queue_multiplier=self._u(self.cfg.OOD_BURST_QUEUE_MULTIPLIER_MIN, self.cfg.OOD_BURST_QUEUE_MULTIPLIER_MAX) if ood_type in {self.cfg.OOD_TYPE_URLLC_BURST, self.cfg.OOD_TYPE_MIXED} else 1.0,
        )

    def generate_vehicle_state(self, vehicle_id: str = "veh_0", ood_state: Optional[OODState] = None) -> VehicleState:
        if self.scenario == ScenarioType.HIGHWAY:
            speed = self._u(self.cfg.HIGHWAY_SPEED_MPS_MIN, self.cfg.HIGHWAY_SPEED_MPS_MAX)
            lanes = self.cfg.HIGHWAY_NUM_LANES
            env_type = "highway"
        else:
            speed = self._u(self.cfg.URBAN_SPEED_MPS_MIN, self.cfg.URBAN_SPEED_MPS_MAX)
            lanes = self.cfg.URBAN_NUM_LANES
            env_type = "urban"
        if ood_state and ood_state.is_ood:
            speed *= ood_state.speed_multiplier
        lane = int(self.rng.integers(0, lanes))
        # 大多数车辆在中近距离，少数在边缘区，避免数据全是极端坏信道。
        if self.rng.random() < 0.78:
            dist = self._logu(self.cfg.V2I_DISTANCE_MIN_M, 220.0)
        else:
            dist = self._u(220.0, self.cfg.V2I_DISTANCE_MAX_M)
        return VehicleState(vehicle_id, self.scenario, speed, self._u(self.cfg.ACCELERATION_MPS2_MIN, self.cfg.ACCELERATION_MPS2_MAX), dist, lane, 0.0, lane * self.cfg.LANE_WIDTH_M)

    def generate_channel_state(self, vehicle: VehicleState, ood_state: Optional[OODState] = None) -> ChannelState:
        env_type = "highway" if vehicle.scenario == ScenarioType.HIGHWAY else "urban"
        bw = float(self.rng.choice(self.cfg.BANDWIDTH_OPTIONS_HZ, p=[0.28, 0.46, 0.26]))
        ptx = self._u(self.cfg.TX_POWER_W_MIN, self.cfg.TX_POWER_W_MAX)
        gain = IoVMathFormulas.calc_comprehensive_channel_gain(vehicle.distance_to_rsu_m, env_type, self.rng)
        if ood_state and ood_state.is_ood:
            gain *= 10.0 ** (-ood_state.deep_fade_loss_db / 10.0)
        noise = self.cfg.calc_noise_power_watt(bw, "ue")
        # 移动性 OOD 提高干扰波动。
        int_hi = self.cfg.INTERFERENCE_TO_NOISE_RATIO_MAX * (1.8 if ood_state and ood_state.ood_type == self.cfg.OOD_TYPE_HIGH_MOBILITY else 1.0)
        interference = noise * self._u(self.cfg.INTERFERENCE_TO_NOISE_RATIO_MIN, int_hi)
        sinr = IoVMathFormulas.calc_sinr_linear(ptx, gain, interference, noise)
        downlink_sinr = IoVMathFormulas.calc_downlink_sinr_linear(sinr)
        uplink_rate = IoVMathFormulas.calc_embb_shannon_rate_bps(bw, sinr)
        downlink_rate = IoVMathFormulas.calc_embb_shannon_rate_bps(bw, downlink_sinr)
        return ChannelState(vehicle.distance_to_rsu_m, bw, ptx, noise, interference, gain, sinr, downlink_sinr, uplink_rate, downlink_rate, env_type)

    def generate_compute_nodes(self, ood_state: Optional[OODState] = None) -> Dict[str, NodeComputeInstance]:
        nodes = {
            "local": NodeComputeInstance("local", "LOCAL", self.generate_vehicle_cpu()),
            "lmec": NodeComputeInstance("lmec", "LOCAL_MEC", self.generate_local_mec_cpu()),
            "rmec": NodeComputeInstance("rmec", "REMOTE_MEC", self.generate_remote_mec_cpu()),
            "cloud": NodeComputeInstance("cloud", "CLOUD", self.generate_cloud_cpu()),
        }
        if ood_state and ood_state.is_ood and ood_state.mec_cpu_reduction_ratio > 0.0:
            nodes["lmec"].available_cpu_ratio = max(0.08, 1.0 - ood_state.mec_cpu_reduction_ratio)
            nodes["rmec"].available_cpu_ratio = max(0.08, 1.0 - ood_state.mec_cpu_reduction_ratio * 0.75)
        return nodes

    def generate_backhaul_state(self, ood_state: Optional[OODState] = None) -> BackhaulState:
        l2r = self._nclip(self.cfg.DELAY_LMEC_TO_RMEC_MU, self.cfg.DELAY_LMEC_TO_RMEC_SIGMA, self.cfg.DELAY_LMEC_TO_RMEC_MIN, self.cfg.DELAY_LMEC_TO_RMEC_MAX)
        r2c = self._nclip(self.cfg.DELAY_RMEC_TO_CLOUD_MU, self.cfg.DELAY_RMEC_TO_CLOUD_SIGMA, self.cfg.DELAY_RMEC_TO_CLOUD_MIN, self.cfg.DELAY_RMEC_TO_CLOUD_MAX)
        mult = ood_state.backhaul_delay_multiplier if ood_state and ood_state.is_ood else 1.0
        return BackhaulState(l2r * mult, r2c * mult)

    def generate_random_queue_state(self, task_type: str = TaskType.URLLC, load_level: str = "medium", ood_state: Optional[OODState] = None) -> QueueState:
        scale = {"low": 0.55, "medium": 1.0, "high": 1.9, "stress": 3.0}.get(load_level, 1.0)
        if ood_state and ood_state.is_ood:
            scale *= ood_state.queue_multiplier

        if task_type == TaskType.URLLC:
            # 让 LOCAL 与 MEC 的优劣由队列/信道共同决定，而不是永远 LOCAL。
            local_w = self.rng.gamma(1.7, 0.85e6 * scale)
            lmec_w = self.rng.gamma(1.8, 18e6 * scale)
            rmec_w = self.rng.gamma(1.4, 28e6 * scale)
            cloud_w = self.rng.gamma(1.2, 80e6 * scale)
            ql = int(np.clip(local_w / 8e4, 0, self.cfg.LOCAL_QUEUE_MAX_LEN))
            qm = int(np.clip(lmec_w / 2.5e5, 0, self.cfg.LOCAL_MEC_QUEUE_MAX_LEN))
            qr = int(np.clip(rmec_w / 4e5, 0, self.cfg.REMOTE_MEC_QUEUE_MAX_LEN))
            qc = int(np.clip(cloud_w / 1e6, 0, self.cfg.CLOUD_QUEUE_MAX_LEN))
        else:
            local_w = self.rng.gamma(1.7, 0.45e9 * scale)
            lmec_w = self.rng.gamma(1.8, 4.0e9 * scale)
            rmec_w = self.rng.gamma(1.6, 7.0e9 * scale)
            cloud_w = self.rng.gamma(1.4, 28.0e9 * scale)
            ql = int(np.clip(local_w / 3e7, 0, self.cfg.LOCAL_QUEUE_MAX_LEN))
            qm = int(np.clip(lmec_w / 8e7, 0, self.cfg.LOCAL_MEC_QUEUE_MAX_LEN))
            qr = int(np.clip(rmec_w / 1.5e8, 0, self.cfg.REMOTE_MEC_QUEUE_MAX_LEN))
            qc = int(np.clip(cloud_w / 8e8, 0, self.cfg.CLOUD_QUEUE_MAX_LEN))

        return QueueState(ql, qm, qr, qc, float(local_w), float(lmec_w), float(rmec_w), float(cloud_w))

    def generate_empty_queue_state(self) -> QueueState:
        return QueueState()

    def generate_mac_delay(self) -> float:
        return self._nclip(self.cfg.MAC_CONTENTION_DELAY_MU, self.cfg.MAC_CONTENTION_DELAY_SIGMA, self.cfg.MAC_CONTENTION_DELAY_MIN, self.cfg.MAC_CONTENTION_DELAY_MAX)

    def _queue_utilization(self, workload_cycles: float, cpu_hz: float) -> float:
        denom = max(1.0, cpu_hz * self.cfg.QUEUE_UTILIZATION_WINDOW_SEC)
        return float(np.clip(workload_cycles / denom, 0.0, 0.99))

    def instantiate_task(self, task_id: str, task_type: str, arrival_time_sec: float = 0.0, ood_state: Optional[OODState] = None) -> TaskInstance:
        if task_type == TaskType.URLLC:
            if self.rng.random() < 0.70:
                b = self._nclip(self.cfg.URLLC_DATA_BYTES_MU, self.cfg.URLLC_DATA_BYTES_SIGMA, self.cfg.URLLC_DATA_BYTES_MIN, self.cfg.URLLC_DATA_BYTES_MAX)
            else:
                b = self._u(140.0, self.cfg.URLLC_DATA_BYTES_MAX)
            cpb = self._nclip(self.cfg.URLLC_CYCLES_PER_BYTE_MU, self.cfg.URLLC_CYCLES_PER_BYTE_SIGMA, self.cfg.URLLC_CYCLES_PER_BYTE_MIN, self.cfg.URLLC_CYCLES_PER_BYTE_MAX)
            if ood_state and ood_state.ood_type == self.cfg.OOD_TYPE_URLLC_BURST:
                cpb *= self._u(1.05, 1.35)
            bits = b * 8.0
            return TaskInstance(task_id, TaskType.URLLC, bits, b * cpb, self.cfg.URLLC_MAX_DELAY_SEC, arrival_time_sec, result_size_bits=bits * self.cfg.RESULT_SIZE_RATIO_URLLC, tolerant_delay_sec=self.cfg.URLLC_MAX_DELAY_SEC)

        if task_type == TaskType.EMBB:
            # 三峰分布：小视频块 / 普通感知上传 / 大包。
            u = self.rng.random()
            if u < 0.48:
                b = self._logu(2e5, 1.4e6)
            elif u < 0.88:
                b = self._logu(1.4e6, 5.5e6)
            else:
                b = self._logu(5.5e6, self.cfg.EMBB_DATA_BYTES_MAX)
            cpb = self._u(self.cfg.EMBB_CYCLES_PER_BYTE_MIN, self.cfg.EMBB_CYCLES_PER_BYTE_MAX)
            # 大包 deadline 更宽，避免物理上大量必 DROP。
            size_factor = np.clip((b - self.cfg.EMBB_DATA_BYTES_MIN) / (self.cfg.EMBB_DATA_BYTES_MAX - self.cfg.EMBB_DATA_BYTES_MIN), 0.0, 1.0)
            base_deadline = self._u(self.cfg.EMBB_MAX_DELAY_SEC_MIN, self.cfg.EMBB_MAX_DELAY_SEC_MAX)
            d = float(np.clip(base_deadline + 1.8 * size_factor, self.cfg.EMBB_MAX_DELAY_SEC_MIN, 4.2))
            bits = b * 8.0
            return TaskInstance(task_id, TaskType.EMBB, bits, b * cpb, d, arrival_time_sec, result_size_bits=bits * self.cfg.RESULT_SIZE_RATIO_EMBB, tolerant_delay_sec=d * self.cfg.EMBB_TOLERANCE_MULTIPLIER)

        raise ValueError(f"Unknown task_type: {task_type}")

    def build_state_dict(
        self,
        task: TaskInstance,
        vehicle: VehicleState,
        channel: ChannelState,
        compute_nodes: Dict[str, NodeComputeInstance],
        backhaul: BackhaulState,
        queue: QueueState,
        ood: OODState,
    ) -> Dict[str, float]:
        s = {
            "task_type_id": float(self.cfg.TASK_TYPE_TO_ID[task.task_type]),
            "data_size_bits": float(task.data_size_bits),
            "result_size_bits": float(task.result_size_bits),
            "required_cycles": float(task.required_cycles),
            "deadline_sec": float(task.max_delay_sec),
            "tolerant_deadline_sec": float(task.tolerant_delay_sec or task.max_delay_sec),
            "vehicle_speed_mps": float(vehicle.speed_mps),
            "vehicle_acc_mps2": float(vehicle.acceleration_mps2),
            "distance_to_rsu_m": float(vehicle.distance_to_rsu_m),
            "lane_id": float(vehicle.lane_id),
            "sinr_linear": float(channel.sinr_linear),
            "downlink_sinr_linear": float(channel.downlink_sinr_linear),
            "channel_gain_linear": float(channel.channel_gain_linear),
            "tx_power_w": float(channel.tx_power_w),
            "bandwidth_hz": float(channel.bandwidth_hz),
            "uplink_rate_shannon_bps": float(channel.uplink_rate_shannon_bps),
            "downlink_rate_shannon_bps": float(channel.downlink_rate_shannon_bps),
            "mac_delay_sec": float(self.generate_mac_delay()),
            "cpu_local_hz": float(compute_nodes["local"].available_cpu_hz),
            "cpu_lmec_hz": float(compute_nodes["lmec"].available_cpu_hz),
            "cpu_rmec_hz": float(compute_nodes["rmec"].available_cpu_hz),
            "cpu_cloud_hz": float(compute_nodes["cloud"].available_cpu_hz),
            "queue_local_len": float(queue.queue_local_len),
            "queue_lmec_len": float(queue.queue_lmec_len),
            "queue_rmec_len": float(queue.queue_rmec_len),
            "queue_cloud_len": float(queue.queue_cloud_len),
            "workload_local_cycles": float(queue.workload_local_cycles),
            "workload_lmec_cycles": float(queue.workload_lmec_cycles),
            "workload_rmec_cycles": float(queue.workload_rmec_cycles),
            "workload_cloud_cycles": float(queue.workload_cloud_cycles),
            "util_local": self._queue_utilization(queue.workload_local_cycles, compute_nodes["local"].available_cpu_hz),
            "util_lmec": self._queue_utilization(queue.workload_lmec_cycles, compute_nodes["lmec"].available_cpu_hz),
            "util_rmec": self._queue_utilization(queue.workload_rmec_cycles, compute_nodes["rmec"].available_cpu_hz),
            "util_cloud": self._queue_utilization(queue.workload_cloud_cycles, compute_nodes["cloud"].available_cpu_hz),
            "backhaul_l2r_delay_sec": float(backhaul.lmec_to_rmec_delay_sec),
            "backhaul_r2c_delay_sec": float(backhaul.rmec_to_cloud_delay_sec),
            "is_ood": float(ood.is_ood),
            "ood_type_id": float(self.cfg.OOD_TYPE_TO_ID.get(ood.ood_type, 0)),
        }
        return {k: s[k] for k in self.cfg.STATE_FEATURES}

    def generate_snapshot(
        self,
        task_type: str,
        task_id: str = "task_0",
        current_time_sec: float = 0.0,
        load_level: str = "medium",
        force_ood_type: Optional[str] = None,
        include_ood: bool = True,
        ood_prob: Optional[float] = None,
        use_random_queue: bool = True,
    ) -> EnvironmentSnapshot:
        if include_ood:
            ood = self.generate_ood_state(current_time_sec, force_type=force_ood_type, prob=ood_prob)
        else:
            ood = OODState(False, self.cfg.OOD_TYPE_NONE, current_time_sec, 0.0)
        vehicle = self.generate_vehicle_state(f"veh_{task_id}", ood)
        channel = self.generate_channel_state(vehicle, ood)
        nodes = self.generate_compute_nodes(ood)
        backhaul = self.generate_backhaul_state(ood)
        queue = self.generate_random_queue_state(task_type, load_level, ood) if use_random_queue else self.generate_empty_queue_state()
        task = self.instantiate_task(task_id, task_type, current_time_sec, ood)
        state = self.build_state_dict(task, vehicle, channel, nodes, backhaul, queue, ood)
        return EnvironmentSnapshot(task, vehicle, channel, nodes, backhaul, queue, ood, state)


if __name__ == "__main__":
    f = IoVEnvironmentFactory(seed=42)
    snap = f.generate_snapshot(TaskType.URLLC)
    print("iov_env_factory self-check passed", len(snap.state_dict), snap.task.task_type)
