import numpy as np

def calculate_fitness_v2(decision, d_matrix, U_i, D_i, R_up, params, lambda_i, mu_i, mode='convex'):
    """
    V2 适应度函数：计算总系统成本 (时延 + 能耗)
    
    新增功能:
    1. 引入了异构 QoS 权重 lambda_i 和 mu_i 进行独立加权
    2. 针对方案 2 和方案 3 引入了级联路由及公网传播时延
    """
    n_tasks = len(decision)
    f_assigned = np.zeros(n_tasks)
    
    # ---------------------------------------------------------
    # 核心 1：算力分配逻辑 (双轨制)
    # ---------------------------------------------------------
    idx_mec = np.where(decision == 1)[0]
    idx_offsite = np.where(decision == 2)[0]
    
    if mode == 'average':
        if len(idx_mec) > 0:
            f_assigned[idx_mec] = params['f_mec'] / len(idx_mec)
        if len(idx_offsite) > 0:
            f_assigned[idx_offsite] = params['f_offsite'] / len(idx_offsite)
            
    elif mode == 'convex':
        if len(idx_mec) > 0:
            sum_sqrt_D = np.sum(np.sqrt(D_i[idx_mec]))
            f_assigned[idx_mec] = params['f_mec'] * (np.sqrt(D_i[idx_mec]) / sum_sqrt_D)
        if len(idx_offsite) > 0:
            sum_sqrt_D = np.sum(np.sqrt(D_i[idx_offsite]))
            f_assigned[idx_offsite] = params['f_offsite'] * (np.sqrt(D_i[idx_offsite]) / sum_sqrt_D)
    
    # ---------------------------------------------------------
    # 核心 2：时延与能耗计算 (包含物理级联延迟)
    # ---------------------------------------------------------
    total_cost = 0
    
    for i in range(n_tasks):
        U = U_i[i]
        D = D_i[i]
        R = R_up[i]
        
        T_i = 0
        E_i = 0
        
        if decision[i] == 0:
            # 方案 0: 本地执行 (无传播延迟)
            T_i = D / params['f_local']
            E_i = params['k_energy'] * (params['f_local']**2) * D
            
        elif decision[i] == 1:
            # 方案 1: 本地 MEC 卸载
            T_trans = U / R
            T_exec = D / f_assigned[i]
            T_i = T_trans + T_exec
            E_i = params['p_ue'] * T_trans
            
        elif decision[i] == 2:
            # 方案 2: 远程 MEC (增加光纤回传传输时延 + 固定路由时延)
            T_trans = U / R + U / params['r_eo'] + params['prop_eo']
            T_exec = D / f_assigned[i]
            T_i = T_trans + T_exec
            E_i = params['p_ue'] * (U / R)
            
        elif decision[i] == 3:
            # 方案 3: 云端服务器 (增加公网带宽瓶颈 + 公网跨地域时延)
            T_trans = U / R + U / params['r_ec'] + params['prop_ec']
            T_exec = D / params['f_cloud']
            T_i = T_trans + T_exec
            E_i = params['p_ue'] * (U / R)
            
        # V2 核心亮点：按每个任务自身的 URLLC / eMBB QoS 偏好进行独立加权！
        total_cost += lambda_i[i] * T_i + mu_i[i] * E_i
        
    return total_cost
