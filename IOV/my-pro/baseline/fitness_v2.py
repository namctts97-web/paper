import numpy as np
import math
import random

def calculate_fitness_v2(decision, d_matrix, U_i, D_i, R_up, params, lambda_i, mu_i, mode='convex'):
    """
    V2 适应度函数：双轨制 Cost 引擎 (对齐强化学习)
    """
    n_tasks = len(decision)
    f_assigned = np.zeros(n_tasks)
    
    idx_mec = np.where(decision == 1)[0]
    idx_offsite = np.where(decision == 2)[0]
    
    f_mec = params.get('f_mec', 40e9)
    f_offsite = params.get('f_offsite', 150e9)
    f_cloud = params.get('f_cloud', 10000e9)
    f_local = params.get('f_local', 3.5e9)
    
    if mode == 'average':
        if len(idx_mec) > 0:
            f_assigned[idx_mec] = f_mec / len(idx_mec)
        if len(idx_offsite) > 0:
            f_assigned[idx_offsite] = f_offsite / len(idx_offsite)
            
    elif mode == 'convex':
        if len(idx_mec) > 0:
            sum_sqrt_D = np.sum(np.sqrt(D_i[idx_mec]))
            f_assigned[idx_mec] = f_mec * (np.sqrt(D_i[idx_mec]) / sum_sqrt_D)
        if len(idx_offsite) > 0:
            sum_sqrt_D = np.sum(np.sqrt(D_i[idx_offsite]))
            f_assigned[idx_offsite] = f_offsite * (np.sqrt(D_i[idx_offsite]) / sum_sqrt_D)
    
    total_cost = 0
    
    # 模拟带宽切片拥塞
    count_offsite = len(idx_offsite)
    count_cloud = np.sum(decision == 3)
    
    r_eo_base = params.get('r_eo', 500e6)
    r_ec_base = params.get('r_ec', 500e6)
    
    shared_r_eo = r_eo_base / count_offsite if count_offsite > 0 else r_eo_base
    shared_r_ec = r_ec_base / count_cloud if count_cloud > 0 else r_ec_base
    
    for i in range(n_tasks):
        U = U_i[i]
        D = D_i[i]
        R = R_up[i]
        
        # 判断任务类型 (双峰高斯)
        task_type = 'URLLC' if lambda_i[i] > 0.6 else 'eMBB'
        
        # 绝对物理下界能耗 (本地执行能耗)
        E_max = params['k_energy'] * (f_local**2) * D
        if E_max <= 0: E_max = 1e-10
        
        T_i = 0
        E_i = 0
        
        if decision[i] == 0:
            T_i = D / f_local
            E_i = params['k_energy'] * (f_local**2) * D
            
        elif decision[i] == 1:
            T_trans = U / R
            T_exec = D / f_assigned[i]
            T_i = T_trans + T_exec
            E_i = params['p_ue'] * T_trans
            
        elif decision[i] == 2:
            T_trans = U / R + U / shared_r_eo + params['prop_eo']
            T_exec = D / f_assigned[i]
            T_i = T_trans + T_exec
            E_i = params['p_ue'] * (U / R)
            
        elif decision[i] == 3:
            if params.get('force_cloud_congestion', False):
                prop_ec_jitter = random.uniform(0.5, 1.0)
            else:
                prop_ec_jitter = 0.0225 # 专家算法取数学期望 (0.015~0.030)
            T_trans = U / R + U / shared_r_ec + prop_ec_jitter
            T_exec = D / f_cloud
            T_i = T_trans + T_exec
            E_i = params['p_ue'] * (U / R)
            
        # 双轨制代价核算 (严格对齐环境的平滑死区与同构流形)
        if task_type == 'URLLC':
            beta = 10.0
            x = (T_i - 0.003) / 0.003
            smooth_max = x if (beta * x) > 50 else (1.0 / beta) * math.log(1.0 + math.exp(beta * x))
            Cost = 10.0 * math.asinh(smooth_max) + 0.01 * math.asinh(E_i / E_max)
        else: # eMBB
            Cost = 1.0 * math.asinh(T_i / 1.5) + 1.0 * math.asinh(E_i / E_max)
            
        total_cost += Cost
        
    return total_cost
