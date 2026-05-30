# ==========================================
# 文件名: iov_config.py
# 目的: IoV 计算卸载 / 资源分配仿真的统一配置
# 设计原则:
#   1) 车辆不做 V2V；车辆可与 LMEC/RMEC/Cloud 通信
#   2) URLLC 以 3ms safety deadline 为硬红线，DROP 不是合法标签
#   3) eMBB 允许降级服务，DROP 是 admission control 结果
# ==========================================
from __future__ import annotations

import math
from enum import IntEnum
from typing import Dict, List, Tuple


class OffloadAction(IntEnum):
    LOCAL = 0
    LOCAL_MEC = 1
    REMOTE_MEC = 2
    CLOUD = 3
    DROP = 4


class TaskType:
    URLLC = "URLLC"
    EMBB = "eMBB"


class ServiceStatus:
    SUCCESS = "SUCCESS"
    DEGRADED = "DEGRADED"
    DEADLINE_VIOLATION = "DEADLINE_VIOLATION"
    OUTAGE = "OUTAGE"
    REJECTED = "REJECTED"
    ILLEGAL = "ILLEGAL"


class ScenarioType:
    URBAN_GRID = "urban_grid"
    HIGHWAY = "highway"


def dbm_to_watt(dbm_value: float) -> float:
    return 10.0 ** ((float(dbm_value) - 30.0) / 10.0)


def thermal_noise_power_watt(
    bandwidth_hz: float,
    noise_density_dbm_per_hz: float = -174.0,
    noise_figure_db: float = 9.0,
) -> float:
    return dbm_to_watt(noise_density_dbm_per_hz + 10.0 * math.log10(float(bandwidth_hz)) + noise_figure_db)


class HeterogeneousIoVConfig:
    # ---------------- 基础仿真 ----------------
    DEFAULT_SEED = 42
    SIM_TIME_SEC = 10.0
    SLOT_DURATION_SEC = 0.001
    DEFAULT_SCENARIO = ScenarioType.URBAN_GRID
    SUPPORTED_SCENARIOS: List[str] = [ScenarioType.URBAN_GRID, ScenarioType.HIGHWAY]

    # ---------------- 动作空间 ----------------
    ACTION_LOCAL = int(OffloadAction.LOCAL)
    ACTION_LOCAL_MEC = int(OffloadAction.LOCAL_MEC)
    ACTION_REMOTE_MEC = int(OffloadAction.REMOTE_MEC)
    ACTION_CLOUD = int(OffloadAction.CLOUD)
    ACTION_DROP = int(OffloadAction.DROP)
    NUM_ACTIONS = 5

    ACTION_NAME: Dict[int, str] = {
        ACTION_LOCAL: "LOCAL",
        ACTION_LOCAL_MEC: "LOCAL_MEC",
        ACTION_REMOTE_MEC: "REMOTE_MEC",
        ACTION_CLOUD: "CLOUD",
        ACTION_DROP: "DROP",
    }

    # URLLC 不允许主动 DROP；远端/云不是禁用，而是由 3ms safety shield 自然筛掉。
    URLLC_ACTION_MASK: Tuple[int, int, int, int, int] = (1, 1, 1, 1, 0)
    # eMBB 允许 admission control / DROP。
    EMBB_ACTION_MASK: Tuple[int, int, int, int, int] = (1, 1, 1, 1, 1)

    # ---------------- 计算资源 ----------------
    VEHICLE_CPU_GHZ_MIN = 0.8
    VEHICLE_CPU_GHZ_MAX = 3.5
    LOCAL_MEC_CPU_GHZ_MIN = 18.0
    LOCAL_MEC_CPU_GHZ_MAX = 80.0
    REMOTE_MEC_CPU_GHZ_MIN = 80.0
    REMOTE_MEC_CPU_GHZ_MAX = 220.0
    CLOUD_CPU_GHZ_MIN = 1000.0
    CLOUD_CPU_GHZ_MAX = 6000.0

    CPU_RATIO_MIN = 0.05
    CPU_RATIO_MAX = 1.0
    URLLC_CPU_RATIO_GRID: Tuple[float, ...] = (0.20, 0.35, 0.50, 0.70, 0.85, 1.00)
    EMBB_CPU_RATIO_GRID: Tuple[float, ...] = (0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 0.85, 1.00)
    KAPPA_CAPACITANCE = 1e-28

    # 简单资源价格：用于 eMBB expert 避免所有样本都跑云。
    RESOURCE_PRICE = {
        ACTION_LOCAL: 0.05,
        ACTION_LOCAL_MEC: 0.18,
        ACTION_REMOTE_MEC: 0.28,
        ACTION_CLOUD: 0.42,
        ACTION_DROP: 1.00,
    }

    # ---------------- 无线通信 ----------------
    CARRIER_FREQ = 5.9e9
    CARRIER_FREQ_GHZ = 5.9
    SYSTEM_BANDWIDTH_HZ = 20e6
    BANDWIDTH_OPTIONS_HZ: Tuple[float, ...] = (10e6, 20e6, 40e6)
    TX_POWER_W_MIN = 0.08
    TX_POWER_W_MAX = 1.20
    DOWNLINK_SINR_GAIN_DB = 3.0
    DOWNLINK_INTERFERENCE_FACTOR = 0.70
    VEHICLE_ANTENNA_GAIN_DBI = 3.0
    RSU_ANTENNA_GAIN_DBI = 8.0
    THERMAL_NOISE_DENSITY_DBM_PER_HZ = -174.0
    UE_NOISE_FIGURE_DB = 9.0
    RSU_NOISE_FIGURE_DB = 5.0
    NOISE_POWER_W = thermal_noise_power_watt(SYSTEM_BANDWIDTH_HZ, THERMAL_NOISE_DENSITY_DBM_PER_HZ, UE_NOISE_FIGURE_DB)
    INTERFERENCE_TO_NOISE_RATIO_MIN = 0.05
    INTERFERENCE_TO_NOISE_RATIO_MAX = 5.0
    SINR_LINEAR_MIN = 1e-6
    SINR_LINEAR_MAX = 1e6

    # ---------------- 信道 / 移动 ----------------
    PATH_LOSS_EXP_URBAN = 3.0
    PATH_LOSS_EXP_HIGHWAY = 2.5
    DISTANCE_MIN_M = 1.0
    DISTANCE_MAX_M = 1000.0
    V2I_DISTANCE_MIN_M = 8.0
    V2I_DISTANCE_MAX_M = 450.0
    URBAN_SPEED_MPS_MIN = 5.56
    URBAN_SPEED_MPS_MAX = 16.67
    HIGHWAY_SPEED_MPS_MIN = 16.67
    HIGHWAY_SPEED_MPS_MAX = 33.33
    ACCELERATION_MPS2_MIN = -3.0
    ACCELERATION_MPS2_MAX = 3.0
    LANE_WIDTH_M = 3.5
    URBAN_NUM_LANES = 4
    HIGHWAY_NUM_LANES = 6

    # ---------------- MAC / FBL / 可靠性 ----------------
    MAC_CONTENTION_DELAY_MU = 0.00022
    MAC_CONTENTION_DELAY_SIGMA = 0.00008
    MAC_CONTENTION_DELAY_MIN = 0.00005
    MAC_CONTENTION_DELAY_MAX = 0.00080
    URLLC_TARGET_ERROR_RATE = 1e-5
    URLLC_TX_TIME_MIN_SEC = 2e-5
    URLLC_TX_TIME_MAX_SEC = 0.003

    # ---------------- 回传 ----------------
    # RMEC 有机会在极少数 URLLC 小包/好信道下可行；Cloud 基本不适合 URLLC。
    DELAY_LMEC_TO_RMEC_MU = 0.0012
    DELAY_LMEC_TO_RMEC_SIGMA = 0.00055
    DELAY_LMEC_TO_RMEC_MIN = 0.00025
    DELAY_LMEC_TO_RMEC_MAX = 0.0080
    DELAY_RMEC_TO_CLOUD_MU = 0.010
    DELAY_RMEC_TO_CLOUD_SIGMA = 0.004
    DELAY_RMEC_TO_CLOUD_MIN = 0.004
    DELAY_RMEC_TO_CLOUD_MAX = 0.050

    # ---------------- 任务模型 ----------------
    URLLC_DATA_BYTES_MU = 96.0
    URLLC_DATA_BYTES_SIGMA = 42.0
    URLLC_DATA_BYTES_MIN = 24.0
    URLLC_DATA_BYTES_MAX = 320.0
    URLLC_CYCLES_PER_BYTE_MU = 1150.0
    URLLC_CYCLES_PER_BYTE_SIGMA = 300.0
    URLLC_CYCLES_PER_BYTE_MIN = 350.0
    URLLC_CYCLES_PER_BYTE_MAX = 2600.0
    URLLC_MAX_DELAY_SEC = 0.003

    # eMBB 用分层采样，不只用一个巨大均值；这些是默认边界。
    EMBB_DATA_BYTES_MIN = 2e5       # 0.2 MB
    EMBB_DATA_BYTES_MAX = 12e6      # 12 MB
    EMBB_CYCLES_PER_BYTE_MIN = 80.0
    EMBB_CYCLES_PER_BYTE_MAX = 900.0
    EMBB_MAX_DELAY_SEC_MIN = 0.45
    EMBB_MAX_DELAY_SEC_MAX = 3.00
    EMBB_TOLERANCE_MULTIPLIER = 2.25

    RESULT_SIZE_RATIO_URLLC = 0.05
    RESULT_SIZE_RATIO_EMBB = 0.10
    BACKHAUL_RESULT_RETURN_FACTOR = 0.20

    # ---------------- 队列 / 业务负载 ----------------
    LOCAL_QUEUE_MAX_LEN = 128
    LOCAL_MEC_QUEUE_MAX_LEN = 2048
    REMOTE_MEC_QUEUE_MAX_LEN = 4096
    CLOUD_QUEUE_MAX_LEN = 20000
    DEFAULT_SCHEDULER_POLICY = "urllc_preemptive_priority"
    # URLLC 可抢占 eMBB，因此只感知一部分共享队列阻塞。
    URLLC_LOCAL_BLOCKING_RATIO = 0.75
    URLLC_MEC_BLOCKING_RATIO = 0.20
    URLLC_RMEC_BLOCKING_RATIO = 0.12
    URLLC_CLOUD_BLOCKING_RATIO = 0.08
    QUEUE_UTILIZATION_WINDOW_SEC = 1.0

    # ---------------- OOD ----------------
    OOD_TYPE_NONE = "none"
    OOD_TYPE_CHANNEL_DEEP_FADE = "channel_deep_fade"
    OOD_TYPE_BACKHAUL_SPIKE = "backhaul_spike"
    OOD_TYPE_URLLC_BURST = "urllc_burst"
    OOD_TYPE_MEC_OVERLOAD = "mec_overload"
    OOD_TYPE_HIGH_MOBILITY = "high_mobility"
    OOD_TYPE_MIXED = "mixed"
    SUPPORTED_OOD_TYPES: List[str] = [
        OOD_TYPE_NONE,
        OOD_TYPE_CHANNEL_DEEP_FADE,
        OOD_TYPE_BACKHAUL_SPIKE,
        OOD_TYPE_URLLC_BURST,
        OOD_TYPE_MEC_OVERLOAD,
        OOD_TYPE_HIGH_MOBILITY,
        OOD_TYPE_MIXED,
    ]
    OOD_PROB_DEFAULT = 0.05
    OOD_DEEP_FADE_LOSS_DB_MIN = 12.0
    OOD_DEEP_FADE_LOSS_DB_MAX = 32.0
    OOD_BACKHAUL_DELAY_MULTIPLIER_MIN = 2.5
    OOD_BACKHAUL_DELAY_MULTIPLIER_MAX = 10.0
    OOD_MEC_CPU_REDUCTION_MIN = 0.35
    OOD_MEC_CPU_REDUCTION_MAX = 0.85
    OOD_SPEED_MULTIPLIER_MIN = 1.4
    OOD_SPEED_MULTIPLIER_MAX = 2.7
    OOD_BURST_QUEUE_MULTIPLIER_MIN = 3.0
    OOD_BURST_QUEUE_MULTIPLIER_MAX = 9.0
    OOD_DURATION_SEC_MIN = 0.005
    OOD_DURATION_SEC_MAX = 1.000
    INCLUDE_OOD_IN_EXPERT_DATA = True
    OOD_SAMPLE_RATIO = 0.20
    DEFAULT_NUM_EXPERT_SAMPLES = 50000

    TASK_TYPE_TO_ID: Dict[str, int] = {TaskType.URLLC: 0, TaskType.EMBB: 1}
    ID_TO_TASK_TYPE: Dict[int, str] = {0: TaskType.URLLC, 1: TaskType.EMBB}
    OOD_TYPE_TO_ID: Dict[str, int] = {
        OOD_TYPE_NONE: 0,
        OOD_TYPE_CHANNEL_DEEP_FADE: 1,
        OOD_TYPE_BACKHAUL_SPIKE: 2,
        OOD_TYPE_URLLC_BURST: 3,
        OOD_TYPE_MEC_OVERLOAD: 4,
        OOD_TYPE_HIGH_MOBILITY: 5,
        OOD_TYPE_MIXED: 6,
    }

    STATE_FEATURES: Tuple[str, ...] = (
        "task_type_id", "data_size_bits", "result_size_bits", "required_cycles", "deadline_sec", "tolerant_deadline_sec",
        "vehicle_speed_mps", "vehicle_acc_mps2", "distance_to_rsu_m", "lane_id",
        "sinr_linear", "downlink_sinr_linear", "channel_gain_linear", "tx_power_w", "bandwidth_hz",
        "uplink_rate_shannon_bps", "downlink_rate_shannon_bps", "mac_delay_sec",
        "cpu_local_hz", "cpu_lmec_hz", "cpu_rmec_hz", "cpu_cloud_hz",
        "queue_local_len", "queue_lmec_len", "queue_rmec_len", "queue_cloud_len",
        "workload_local_cycles", "workload_lmec_cycles", "workload_rmec_cycles", "workload_cloud_cycles",
        "util_local", "util_lmec", "util_rmec", "util_cloud",
        "backhaul_l2r_delay_sec", "backhaul_r2c_delay_sec",
        "is_ood", "ood_type_id",
    )
    NUM_STATE_FEATURES = len(STATE_FEATURES)

    @staticmethod
    def get_action_name(action_id: int) -> str:
        return HeterogeneousIoVConfig.ACTION_NAME.get(int(action_id), "UNKNOWN")

    @staticmethod
    def get_action_mask(task_type: str) -> Tuple[int, int, int, int, int]:
        if task_type == TaskType.URLLC:
            return HeterogeneousIoVConfig.URLLC_ACTION_MASK
        if task_type == TaskType.EMBB:
            return HeterogeneousIoVConfig.EMBB_ACTION_MASK
        raise ValueError(f"Unknown task_type: {task_type}")

    @staticmethod
    def is_action_allowed(task_type: str, action_id: int) -> bool:
        if int(action_id) < 0 or int(action_id) >= HeterogeneousIoVConfig.NUM_ACTIONS:
            return False
        return bool(HeterogeneousIoVConfig.get_action_mask(task_type)[int(action_id)])

    @staticmethod
    def calc_noise_power_watt(bandwidth_hz: float, receiver_type: str = "ue") -> float:
        nf = HeterogeneousIoVConfig.UE_NOISE_FIGURE_DB if receiver_type == "ue" else HeterogeneousIoVConfig.RSU_NOISE_FIGURE_DB
        return thermal_noise_power_watt(bandwidth_hz, HeterogeneousIoVConfig.THERMAL_NOISE_DENSITY_DBM_PER_HZ, nf)

    @staticmethod
    def validate() -> None:
        cfg = HeterogeneousIoVConfig
        assert cfg.URLLC_MAX_DELAY_SEC == 0.003, "URLLC hard deadline must be 3 ms"
        assert cfg.URLLC_ACTION_MASK[cfg.ACTION_DROP] == 0, "URLLC must not allow DROP labels"
        assert cfg.EMBB_ACTION_MASK[cfg.ACTION_DROP] == 1, "eMBB may use DROP as admission control"
        assert cfg.NUM_ACTIONS == 5
        assert cfg.SYSTEM_BANDWIDTH_HZ > 0
        assert cfg.NOISE_POWER_W > 0
        assert set(cfg.SUPPORTED_SCENARIOS) == {ScenarioType.URBAN_GRID, ScenarioType.HIGHWAY}


if __name__ == "__main__":
    HeterogeneousIoVConfig.validate()
    print("iov_config self-check passed")
