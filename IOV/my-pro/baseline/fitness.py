import numpy as np

def calculate_fitness(decision, d_matrix, U_i, D_i, R_up, params, mode='average'):
    """
    适应度函数：计算总系统成本 (时延 + 能耗)
    
    参数:
    - decision: 卸载决策向量 (0:本地, 1:本地MEC, 2:远程MEC, 3:云)
    - d_matrix, U_i, D_i, R_up: 环境参数
    - params: 包含各项权重与物理常数的字典
    - mode: 算力分配模式 ('average' 或 'convex')
    """
    n_tasks = len(decision)
    f_assigned = np.zeros(n_tasks)
    
    # ---------------------------------------------------------
    # 核心 1：算力分配逻辑 (双轨制)
    # ---------------------------------------------------------
    # 找到卸载到本地 MEC (1) 和 远程 MEC (2) 的任务索引
    idx_mec = np.where(decision == 1)[0]
    idx_offsite = np.where(decision == 2)[0]
    
    if mode == 'average':
        # 【模式 A：平均分配】—— 复刻原论文源码 logic
        # 不论任务大小，直接按数量均分总算力
        if len(idx_mec) > 0:
            f_assigned[idx_mec] = params['f_mec'] / len(idx_mec)
        if len(idx_offsite) > 0:
            f_assigned[idx_offsite] = params['f_offsite'] / len(idx_offsite)
            
    elif mode == 'convex':
        # 【模式 B：解析解凸优化分配】—— 导师要求的数学修正版本
        # 根据论文公式 (15)-(17)，算力应按计算量 D_i 的平方根比例分配
        if len(idx_mec) > 0:
            sum_sqrt_D = np.sum(np.sqrt(D_i[idx_mec]))
            f_assigned[idx_mec] = params['f_mec'] * (np.sqrt(D_i[idx_mec]) / sum_sqrt_D)
        if len(idx_offsite) > 0:
            sum_sqrt_D = np.sum(np.sqrt(D_i[idx_offsite]))
            f_assigned[idx_offsite] = params['f_offsite'] * (np.sqrt(D_i[idx_offsite]) / sum_sqrt_D)
    
    # ---------------------------------------------------------
    # 核心 2：时延与能耗计算
    # ---------------------------------------------------------
    total_cost = 0
    
    for i in range(n_tasks):
        U = U_i[i]      # 输入数据量 (bits)
        D = D_i[i]      # 计算量 (cycles)
        R = R_up[i]     # 上行速率 (bps)
        
        T_i = 0  # 任务 i 的总时延
        E_i = 0  # 任务 i 的总能耗
        
        if decision[i] == 0:
            # 方案 0: 本地执行
            T_i = D / params['f_local']
            E_i = params['k_energy'] * (params['f_local']**2) * D
            
        elif decision[i] == 1:
            # 方案 1: 本地 MEC 卸载
            T_trans = U / R
            T_exec = D / f_assigned[i]
            T_i = T_trans + T_exec
            E_i = params['p_ue'] * T_trans  # 仅计入车辆侧传输能耗
            
        elif decision[i] == 2:
            # 方案 2: 远程闲置 MEC 卸载 (额外增加 V2I/I2I 传输延迟)
            T_trans = U / R + U / params['r_eo']
            T_exec = D / f_assigned[i]
            T_i = T_trans + T_exec
            E_i = params['p_ue'] * (U / R)
            
        elif decision[i] == 3:
            # 方案 3: 云服务器卸载 (带宽大但往返时延高)
            T_trans = U / R + U / params['r_ec']
            T_exec = D / params['f_cloud']
            T_i = T_trans + T_exec
            E_i = params['p_ue'] * (U / R)
            
        # 计算加权系统成本
        total_cost += params['lambda'] * T_i + params['mu'] * E_i
        
    return total_cost
