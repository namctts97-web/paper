import numpy as np

def get_env_params_v2():
    """
    V2 高保真物理仿真参数 (绝对契约版)
    对标现代智能网联汽车、边缘计算与云中心算力，并引入真实广域网传播时延
    """
    params = {
        'B': 100 * 1e6,              # 带宽 100MHz (5G Sub-6GHz)
        'K_RBs': 50,                 # 5G NR 资源块数量 (OFDMA 切分)
        'sigma2': 2 * 1e-13,         # 噪声功率
        'p_ue': 2.0,                 # 车载发射功率 2.0W (C-V2X 33dBm EIRP上限)
        'f_cloud': 10000 * 1e9,      # 云服务器总算力 10000GHz (AWS EC2 大型实例)
        'r_eo': 500 * 1e6,           # 远程 MEC 间回传带宽 500 Mbps (光纤环网切片)
        'r_ec': 500 * 1e6,           # 云端核心网回传带宽 500 Mbps (保底切片，确保 eMBB 传输)
        'prop_eo': 0.01,             # 远程 MEC 固定路由延迟 10 ms (城域环网)
        'k_energy': 1e-27,           # 能耗系数
        'V_max': 3,
        'V_min': -3,
        # f_local, f_mec, prop_ec 将在 env 内部动态随机生成
    }
    return params

def calculate_channel_gain_v2(d_matrix):
    """大尺度衰落：Path Loss + Shadowing"""
    # 3GPP UMa/Highway: 128.1 + 37.6 log10(d)
    pl_db = 128.1 + 37.6 * np.log10(np.maximum(d_matrix / 1000.0, 1e-4))
    # Log-Normal Shadowing (std = 8dB)
    shadowing_db = np.random.normal(0, 8, size=pl_db.shape)
    total_loss_db = pl_db + shadowing_db
    return 10**(-total_loss_db / 10)

def get_uplink_rate_v2(B, n_tasks, p_ue, channel_gain, sigma2, K_RBs=50, offload_count=None):
    """
    基于平均场理论 (MFT) 的 OFDMA 干扰速率预估
    用于 Expert 预计算阶段。真实 RL 环境在 step 中使用微观采样。
    """
    if offload_count is None: offload_count = n_tasks
    rho_rb = min(1.0, offload_count / K_RBs)
    avg_channel = np.mean(channel_gain)
    # 平均场同频干扰 (I_mean)
    I_mean = rho_rb * p_ue * avg_channel
    
    sinr = (p_ue * channel_gain) / (sigma2 + I_mean)
    alloc_B = B / max(1, offload_count)
    return alloc_B * np.log2(1 + sinr)
