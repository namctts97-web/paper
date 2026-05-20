import numpy as np

def get_env_params():
    """
    获取全局仿真参数
    """
    params = {
        'B': 40 * 1e6,               # 带宽 40MHz
        'sigma2': 2 * 1e-13,         # 噪声功率
        'p_ue': 0.5,                 # 车辆发射功率 0.5W
        'f_local': 0.1 * 1e9,        # 局部计算能力 0.1GHz
        'f_mec': 20 * 1e9,           # 本地 MEC 总算力 20GHz
        'f_offsite': 40 * 1e9,       # 远程闲置 MEC 总算力 40GHz
        'f_cloud': 100 * 1e9,        # 云服务器总算力 100GHz
        'r_eo': 10 * 8 * 1e6,        # 车辆到远程 MEC 的传输速率 (示例值)
        'r_ec': 2 * 8 * 1e6,         # 车辆到云端的传输速率 (示例值)
        'k_energy': 1e-27,           # 能耗系数
        'V_max': 3,
        'V_min': -3,
        'lambda': 0.5,               # 时延权重
        'mu': 0.5                    # 能耗权重
    }
    return params

def generate_topology(n_tasks=80, seed=42):
    """
    生成网络拓扑与任务数据
    """
    np.random.seed(seed)
    
    # 1. 生成距离矩阵 (0-1000m)
    d_matrix = np.random.rand(n_tasks) * 1000
    
    # 2. 生成任务输入数据量 U_i (200-700KB) -> 转换为 bits
    U_i = np.random.randint(200, 701, n_tasks) * 8 * 1024
    
    # 3. 生成计算量 D_i (600-1500 Megacycles) -> 转换为 cycles
    # 导师要求：制造极度异构场景 (20%重任务, 80%轻任务)
    D_i = np.zeros(n_tasks)
    heavy_count = int(0.2 * n_tasks)
    indices = np.arange(n_tasks)
    np.random.shuffle(indices)
    heavy_idx = indices[:heavy_count]
    light_idx = indices[heavy_count:]
    
    D_i[heavy_idx] = np.random.randint(4000, 6001, len(heavy_idx)) * 1e6
    D_i[light_idx] = np.random.randint(50, 201, len(light_idx)) * 1e6
    
    return d_matrix, U_i, D_i, heavy_idx, light_idx

def calculate_channel_gain(d_matrix, mode='flawed'):
    """
    信道增益计算 (双版本)
    - flawed: 复刻原论文错误，使用米(m)且直接将 dB 作为线性增益
    - physical: 严谨物理版本，使用千米(km)且执行 10^(-PL/10)
    """
    if mode == 'flawed':
        # 【学术缺陷版本】：直接将 dB 损耗作为线性增益代入 (源码逻辑)
        pl_db = 127 + 30 * np.log10(np.maximum(d_matrix, 1.0))
        return np.abs(pl_db)
    else:
        # 【物理真实版本】：强制单位换算 m -> km，并转换为线性增益 G = 10^(-PL/10)
        # 修正：距离必须为 km (即 d/1000)
        pl_db = 127 + 30 * np.log10(np.maximum(d_matrix / 1000.0, 1e-4))
        return 10**(-pl_db / 10)

def get_uplink_rate(B, n_tasks, p_ue, channel_gain, sigma2):
    """
    根据 Shannon 公式计算上行速率
    """
    # R = (B/N) * log2(1 + (P * G) / sigma2)
    return (B / n_tasks) * np.log2(1 + (p_ue * channel_gain) / sigma2)
