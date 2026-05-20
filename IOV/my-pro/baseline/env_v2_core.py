import numpy as np

def get_env_params_v2():
    """
    V2 高保真物理仿真参数 (完全隔离版)
    对标现代智能网联汽车、边缘计算与云中心算力，并引入真实广域网传播时延
    """
    params = {
        'B': 40 * 1e6,               # 带宽 40MHz
        'sigma2': 2 * 1e-13,         # 噪声功率
        'p_ue': 2.0,                 # 车载发射功率 2.0W (C-V2X 33dBm EIRP上限)
        'f_local': 1.5 * 1e9,        # 局部计算能力 1.5GHz (现代智能座舱基准)
        'f_mec': 28 * 1e9,           # 本地 MEC 总算力 28GHz (典型 OTII 边缘基站)
        'f_offsite': 56 * 1e9,       # 远程闲置 MEC 总算力 56GHz (汇聚机房资源)
        'f_cloud': 200 * 1e9,        # 云服务器总算力 200GHz (AWS EC2 大型实例)
        'r_eo': 500 * 1e6,           # 远程 MEC 间回传带宽 500 Mbps (升级为光纤环网)
        'r_ec': 100 * 1e6,           # 云端核心网回传带宽 100 Mbps (升级为云专线)
        'prop_eo': 0.01,             # 远程 MEC 固定路由延迟 10 ms (城域环网)
        'prop_ec': 0.15,             # 云端固定公网跨地域延迟 150 ms (广域网)
        'k_energy': 1e-27,           # 能耗系数
        'V_max': 3,
        'V_min': -3,
        # 注意：此处移除了全局静态的 lambda 和 mu，将由 generate_topology_v2 动态生成
    }
    return params

def generate_topology_v2(n_tasks=80, seed=42):
    """
    V2 拓扑生成器：包含异构任务的 QoS 权重 (双峰高斯极化模型)
    """
    np.random.seed(seed)
    
    # 1. 距离矩阵 (0-1000m)
    d_matrix = np.random.rand(n_tasks) * 1000
    
    # 2. 任务输入数据量 U_i 和计算量 D_i (与 QoS 强绑定)
    U_i = np.zeros(n_tasks)
    D_i = np.zeros(n_tasks)
    lambda_i = np.zeros(n_tasks)
    
    heavy_idx = []
    light_idx = []
    
    for i in range(n_tasks):
        if np.random.rand() < 0.3:
            # 30% URLLC 任务 (极度敏感于时延，小数据重计算)
            lambda_i[i] = np.clip(np.random.normal(0.85, 0.05), 0.7, 1.0)
            U_i[i] = np.random.uniform(100, 500) * 1024 * 8          # 100KB - 500KB (bits)
            D_i[i] = np.random.uniform(2000, 5000) * 1e6             # 2 - 5 Gcycles
            heavy_idx.append(i)
        else:
            # 70% eMBB 任务 (更敏感于能耗，大数据轻计算)
            lambda_i[i] = np.clip(np.random.normal(0.25, 0.1), 0.0, 0.5)
            U_i[i] = np.random.uniform(5000, 20000) * 1024 * 8       # 5MB - 20MB (bits)
            D_i[i] = np.random.uniform(500, 2000) * 1e6              # 0.5 - 2 Gcycles
            light_idx.append(i)
            
    mu_i = 1.0 - lambda_i
    heavy_idx = np.array(heavy_idx)
    light_idx = np.array(light_idx)
    
    return d_matrix, U_i, D_i, heavy_idx, light_idx, lambda_i, mu_i

def calculate_channel_gain_v2(d_matrix):
    """
    物理严谨版信道增益 (强制 km 单位)
    """
    pl_db = 127 + 30 * np.log10(np.maximum(d_matrix / 1000.0, 1e-4))
    return 10**(-pl_db / 10)

def get_uplink_rate_v2(B, n_tasks, p_ue, channel_gain, sigma2):
    return (B / n_tasks) * np.log2(1 + (p_ue * channel_gain) / sigma2)
