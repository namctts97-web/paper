import numpy as np
from scipy.special import gamma
from fitness import calculate_fitness
from fitness_v2 import calculate_fitness_v2

def logistic_map_init(pop_size, n_tasks, r=3.96):
    """
    使用 Logistic Map 进行混沌初始化
    """
    X = np.zeros((pop_size, n_tasks))
    for i in range(pop_size):
        for k in range(n_tasks):
            state = np.random.rand()
            for _ in range(100):
                state = r * state * (1 - state)
            X[i, k] = np.round(state * 3)
    return X.astype(int)

def get_levy_step(n_tasks, beta=1.5):
    """
    Mantegna 算法生成 Lévy 飞行步长
    """
    sigma_u = (gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
               (gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
    u = np.random.normal(0, sigma_u, n_tasks)
    v = np.random.normal(0, 1, n_tasks)
    step = u / (np.abs(v)**(1 / beta))
    return step

def run_acoras_levy(d_matrix, U_i, D_i, R_up, params, pop_size=50, max_iter=50):
    """
    终极改进版 ACORAS_Levy: Lévy 飞行 + 局部搜索 + 凸优化
    """
    n_tasks = len(d_matrix)
    X = logistic_map_init(pop_size, n_tasks)
    V = np.random.uniform(-3, 3, (pop_size, n_tasks))
    
    pbest_X = X.copy()
    pbest_fit = np.array([calculate_fitness(x, d_matrix, U_i, D_i, R_up, params, mode='convex') for x in X])
    
    gbest_idx = np.argmin(pbest_fit)
    gbest_X = pbest_X[gbest_idx].copy()
    gbest_fit = pbest_fit[gbest_idx]
    
    history = []

    for gen in range(max_iter):
        # --- 核心修复：方差陷阱 (归一化处理) ---
        # 直接对原始成本求方差会导致数值过大而永远无法触发变异。
        # 必须先映射到 [0, 1] 区间后再求方差。
        max_f, min_f = np.max(pbest_fit), np.min(pbest_fit)
        if max_f > min_f:
            norm_fit = (pbest_fit - min_f) / (max_f - min_f)
        else:
            norm_fit = np.zeros_like(pbest_fit)

        # --- 终极改进版：遵循论文原始数学推导 (0.05 阈值) ---
        if np.var(norm_fit) < 0.05: 
            # 执行 Lévy 飞行
            step = get_levy_step(n_tasks)
            # 修正：解除变异死锁，改乘法耦合为加法扰动
            new_gbest = np.clip(np.round(gbest_X + step), 0, 3).astype(int)
            new_fit = calculate_fitness(new_gbest, d_matrix, U_i, D_i, R_up, params, mode='convex')
            if new_fit < gbest_fit:
                gbest_fit = new_fit
                gbest_X = new_gbest.copy()

        # 2. 自适应参数更新 (剥离伪装：移除源码中的 1+ 常数项，回归论文公式 22-23)
        w = 0.6 + (0.8 - 0.6) * (1 - np.exp(-(gen / (max_iter / 2))**3))
        c1 = 2 * np.sin(np.pi / 2 * (1 - gen / max_iter))**2
        c2 = 2 * np.sin(np.pi / 2 * gen / max_iter)**2

        # 3. 粒子群位置更新
        for i in range(pop_size):
            r1, r2 = np.random.rand(), np.random.rand()
            V[i] = w * V[i] + c1 * r1 * (pbest_X[i] - X[i]) + c2 * r2 * (gbest_X - X[i])
            V[i] = np.clip(V[i], -3, 3)
            X[i] = np.round(X[i] + V[i])
            X[i] = np.clip(X[i], 0, 3).astype(int)
            
            # 更新个体最优
            fit = calculate_fitness(X[i], d_matrix, U_i, D_i, R_up, params, mode='convex')
            if fit < pbest_fit[i]:
                pbest_fit[i] = fit
                pbest_X[i] = X[i].copy()
                if fit < gbest_fit:
                    gbest_fit = fit
                    gbest_X = X[i].copy()

        # 4. 贪婪局部搜索 (Local Search)
        for _ in range(5):
            mut_task = np.random.randint(n_tasks)
            alt_node = np.random.randint(0, 4)
            temp_gbest = gbest_X.copy()
            temp_gbest[mut_task] = alt_node
            temp_fit = calculate_fitness(temp_gbest, d_matrix, U_i, D_i, R_up, params, mode='convex')
            if temp_fit < gbest_fit:
                gbest_fit = temp_fit
                gbest_X = temp_gbest.copy()

        # 每 10 代日志记录 (导师要求)
        if (gen + 1) % 10 == 0:
            var_fit = np.var(pbest_fit)
            print(f"[ACORAS_Levy] Gen {gen+1:02d}: Best Cost = {gbest_fit:.4f}, Variance = {var_fit:.6f}")
        
        history.append(gbest_fit)

    return gbest_X, gbest_fit, history

def run_acoras_cauchy(d_matrix, U_i, D_i, R_up, params, pop_size=50, max_iter=50):
    """
    基础版 ACORAS_Cauchy: 柯西变异 + 平均分配 (复刻论文图表)
    """
    n_tasks = len(d_matrix)
    X = logistic_map_init(pop_size, n_tasks)
    V = np.random.uniform(-3, 3, (pop_size, n_tasks))
    
    pbest_X = X.copy()
    pbest_fit = np.array([calculate_fitness(x, d_matrix, U_i, D_i, R_up, params, mode='average') for x in X])
    
    gbest_idx = np.argmin(pbest_fit)
    gbest_X = pbest_X[gbest_idx].copy()
    gbest_fit = pbest_fit[gbest_idx]
    
    history = []

    for gen in range(max_iter):
        # --- 核心修复：方差陷阱 (归一化处理) ---
        max_f, min_f = np.max(pbest_fit), np.min(pbest_fit)
        if max_f > min_f:
            norm_fit = (pbest_fit - min_f) / (max_f - min_f)
        else:
            norm_fit = np.zeros_like(pbest_fit)

        # 变异触发 (使用归一化后的方差)
        if np.var(norm_fit) < 0.04:
            # 柯西变异
            cauchy_step = np.tan(np.pi * (np.random.rand(n_tasks) - 0.5))
            # 修正：解除变异死锁
            new_gbest = np.clip(np.round(gbest_X +  cauchy_step), 0, 3).astype(int)
            new_fit = calculate_fitness(new_gbest, d_matrix, U_i, D_i, R_up, params, mode='average')
            if new_fit < gbest_fit:
                gbest_fit = new_fit
                gbest_X = new_gbest.copy()

        w = 0.6 + (0.8 - 0.6) * (1 - np.exp(-(gen / (max_iter / 2))**3))
        c1 = 1 + 2 * np.sin(np.pi / 2 * (1 - gen / max_iter))**2
        c2 = 1 + 2 * np.sin(np.pi / 2 * gen / max_iter)**2

        for i in range(pop_size):
            r1, r2 = np.random.rand(), np.random.rand()
            V[i] = w * V[i] + c1 * r1 * (pbest_X[i] - X[i]) + c2 * r2 * (gbest_X - X[i])
            V[i] = np.clip(V[i], -3, 3)
            X[i] = np.round(X[i] + V[i])
            X[i] = np.clip(X[i], 0, 3).astype(int)
            
            fit = calculate_fitness(X[i], d_matrix, U_i, D_i, R_up, params, mode='average')
            if fit < pbest_fit[i]:
                pbest_fit[i] = fit
                pbest_X[i] = X[i].copy()
                if fit < gbest_fit:
                    gbest_fit = fit
                    gbest_X = X[i].copy()

        if (gen + 1) % 10 == 0:
            var_fit = np.var(pbest_fit)
            print(f"[ACORAS_Cauchy] Gen {gen+1:02d}: Best Cost = {gbest_fit:.4f}, Variance = {var_fit:.6f}")
        
        history.append(gbest_fit)

    return gbest_X, gbest_fit, history

# ================= 基准算法 =================

def run_dpsao(d_matrix, U_i, D_i, R_up, params, pop_size=50, max_iter=50):
    """
    DPSAO (离散粒子群算法)：标准对照组
    - 初始化：纯随机初始化
    - 参数：固定权重 w=0.8, c1=2.0, c2=2.0
    - 分配：强制 mode='average'
    """
    n_tasks = len(d_matrix)
    # 1. 纯随机初始化 (严禁使用 Logistic 映射)
    X = np.random.randint(0, 4, (pop_size, n_tasks))
    V = np.random.uniform(-3, 3, (pop_size, n_tasks))
    
    pbest_X = X.copy()
    pbest_fit = np.array([calculate_fitness(x, d_matrix, U_i, D_i, R_up, params, mode='average') for x in X])
    
    gbest_idx = np.argmin(pbest_fit)
    gbest_X = pbest_X[gbest_idx].copy()
    gbest_fit = pbest_fit[gbest_idx]
    
    history = []
    
    # 固定参数 (Control Group)
    w, c1, c2 = 0.8, 2.0, 2.0
    
    for gen in range(max_iter):
        for i in range(pop_size):
            r1, r2 = np.random.rand(), np.random.rand()
            # 标准位置更新公式
            V[i] = w * V[i] + c1 * r1 * (pbest_X[i] - X[i]) + c2 * r2 * (gbest_X - X[i])
            V[i] = np.clip(V[i], -3, 3)
            X[i] = np.round(X[i] + V[i])
            X[i] = np.clip(X[i], 0, 3).astype(int)
            
            fit = calculate_fitness(X[i], d_matrix, U_i, D_i, R_up, params, mode='average')
            if fit < pbest_fit[i]:
                pbest_fit[i] = fit
                pbest_X[i] = X[i].copy()
                if fit < gbest_fit:
                    gbest_fit = fit
                    gbest_X = X[i].copy()
                    
        history.append(gbest_fit)
        if (gen + 1) % 10 == 0:
            print(f"[DPSAO] Gen {gen+1:02d}: Best Cost = {gbest_fit:.4f}")
            
    return gbest_X, gbest_fit, history

def run_baseline_alao(d_matrix, U_i, D_i, R_up, params):
    """ALAO: 全本地执行"""
    decision = np.zeros(len(d_matrix), dtype=int)
    return calculate_fitness(decision, d_matrix, U_i, D_i, R_up, params, mode='average')

def run_baseline_almao(d_matrix, U_i, D_i, R_up, params):
    """ALMAO: 全本地 MEC 执行"""
    decision = np.ones(len(d_matrix), dtype=int)
    return calculate_fitness(decision, d_matrix, U_i, D_i, R_up, params, mode='average')

def run_baseline_arao(d_matrix, U_i, D_i, R_up, params):
    """ARAO: 全随机卸载"""
    decision = np.random.randint(0, 4, len(d_matrix))
    return calculate_fitness(decision, d_matrix, U_i, D_i, R_up, params, mode='average')

# ================= 新增代码：支持多模式切换的 v2 版本算法 (不修改原有函数) =================

def run_acoras_levy_v2(d_matrix, U_i, D_i, R_up, params, lambda_i, mu_i, pop_size=50, max_iter=50, mode='convex'):
    """新增 v2 版本：支持 mode 参数，不打印 Gen 日志以加速批量实验"""
    n_tasks = len(d_matrix)
    X = logistic_map_init(pop_size, n_tasks)
    V = np.random.uniform(-3, 3, (pop_size, n_tasks))
    pbest_X = X.copy()
    pbest_fit = np.array([calculate_fitness_v2(x, d_matrix, U_i, D_i, R_up, params, lambda_i, mu_i, mode=mode) for x in X])
    gbest_idx = np.argmin(pbest_fit)
    gbest_X, gbest_fit = pbest_X[gbest_idx].copy(), pbest_fit[gbest_idx]
    history = []
    for gen in range(max_iter):
        max_f, min_f = np.max(pbest_fit), np.min(pbest_fit)
        norm_fit = (pbest_fit - min_f) / (max_f - min_f) if max_f > min_f else np.zeros_like(pbest_fit)
        if np.var(norm_fit) < 0.05: 
            step = get_levy_step(n_tasks)
            # 修正：解除变异死锁
            new_gbest = np.clip(np.round(gbest_X + step), 0, 3).astype(int)
            new_fit = calculate_fitness_v2(new_gbest, d_matrix, U_i, D_i, R_up, params, lambda_i, mu_i, mode=mode)
            if new_fit < gbest_fit: gbest_fit, gbest_X = new_fit, new_gbest.copy()
        w = 0.6 + (0.8 - 0.6) * (1 - np.exp(-(gen / (max_iter / 2))**3))
        c1, c2 = 2 * np.sin(np.pi / 2 * (1 - gen / max_iter))**2, 2 * np.sin(np.pi / 2 * gen / max_iter)**2
        for i in range(pop_size):
            r1, r2 = np.random.rand(), np.random.rand()
            V[i] = np.clip(w * V[i] + c1 * r1 * (pbest_X[i] - X[i]) + c2 * r2 * (gbest_X - X[i]), -3, 3)
            X[i] = np.clip(np.round(X[i] + V[i]), 0, 3).astype(int)
            fit = calculate_fitness_v2(X[i], d_matrix, U_i, D_i, R_up, params, lambda_i, mu_i, mode=mode)
            if fit < pbest_fit[i]:
                pbest_fit[i], pbest_X[i] = fit, X[i].copy()
                if fit < gbest_fit: gbest_fit, gbest_X = fit, X[i].copy()
        for _ in range(5):
            mut_task, alt_node = np.random.randint(n_tasks), np.random.randint(0, 4)
            temp_gbest = gbest_X.copy()
            temp_gbest[mut_task] = alt_node
            temp_fit = calculate_fitness_v2(temp_gbest, d_matrix, U_i, D_i, R_up, params, lambda_i, mu_i, mode=mode)
            if temp_fit < gbest_fit: gbest_fit, gbest_X = temp_fit, temp_gbest.copy()
        history.append(gbest_fit)
    return gbest_X, gbest_fit, history

def run_acoras_cauchy_v2(d_matrix, U_i, D_i, R_up, params, lambda_i, mu_i, pop_size=50, max_iter=50, mode='average'):
    """新增 v2 版本：支持 mode 参数，不打印 Gen 日志"""
    n_tasks = len(d_matrix)
    X = logistic_map_init(pop_size, n_tasks)
    V = np.random.uniform(-3, 3, (pop_size, n_tasks))
    pbest_X = X.copy()
    pbest_fit = np.array([calculate_fitness_v2(x, d_matrix, U_i, D_i, R_up, params, lambda_i, mu_i, mode=mode) for x in X])
    gbest_idx = np.argmin(pbest_fit)
    gbest_X, gbest_fit = pbest_X[gbest_idx].copy(), pbest_fit[gbest_idx]
    history = []
    for gen in range(max_iter):
        max_f, min_f = np.max(pbest_fit), np.min(pbest_fit)
        norm_fit = (pbest_fit - min_f) / (max_f - min_f) if max_f > min_f else np.zeros_like(pbest_fit)
        if np.var(norm_fit) < 0.04:
            cauchy_step = np.tan(np.pi * (np.random.rand(n_tasks) - 0.5))
            # 修正：解除变异死锁
            new_gbest = np.clip(np.round(gbest_X + cauchy_step), 0, 3).astype(int)
            new_fit = calculate_fitness_v2(new_gbest, d_matrix, U_i, D_i, R_up, params, lambda_i, mu_i, mode=mode)
            if new_fit < gbest_fit: gbest_fit, gbest_X = new_fit, new_gbest.copy()
        w = 0.6 + (0.8 - 0.6) * (1 - np.exp(-(gen / (max_iter / 2))**3))
        c1, c2 = 1 + 2 * np.sin(np.pi / 2 * (1 - gen / max_iter))**2, 1 + 2 * np.sin(np.pi / 2 * gen / max_iter)**2
        for i in range(pop_size):
            r1, r2 = np.random.rand(), np.random.rand()
            V[i] = np.clip(w * V[i] + c1 * r1 * (pbest_X[i] - X[i]) + c2 * r2 * (gbest_X - X[i]), -3, 3)
            X[i] = np.clip(np.round(X[i] + V[i]), 0, 3).astype(int)
            fit = calculate_fitness_v2(X[i], d_matrix, U_i, D_i, R_up, params, lambda_i, mu_i, mode=mode)
            if fit < pbest_fit[i]:
                pbest_fit[i], pbest_X[i] = fit, X[i].copy()
                if fit < gbest_fit: gbest_fit, gbest_X = fit, X[i].copy()
        history.append(gbest_fit)
    return gbest_X, gbest_fit, history

def run_dpsao_v2(d_matrix, U_i, D_i, R_up, params, lambda_i, mu_i, pop_size=50, max_iter=50, mode='average'):
    """新增 v2 版本：支持 mode 参数，不打印 Gen 日志"""
    n_tasks = len(d_matrix)
    X = np.random.randint(0, 4, (pop_size, n_tasks))
    V = np.random.uniform(-3, 3, (pop_size, n_tasks))
    pbest_X = X.copy()
    pbest_fit = np.array([calculate_fitness_v2(x, d_matrix, U_i, D_i, R_up, params, lambda_i, mu_i, mode=mode) for x in X])
    gbest_idx = np.argmin(pbest_fit)
    gbest_X, gbest_fit = pbest_X[gbest_idx].copy(), pbest_fit[gbest_idx]
    history = []
    w, c1, c2 = 0.8, 2.0, 2.0
    for gen in range(max_iter):
        for i in range(pop_size):
            r1, r2 = np.random.rand(), np.random.rand()
            V[i] = np.clip(w * V[i] + c1 * r1 * (pbest_X[i] - X[i]) + c2 * r2 * (gbest_X - X[i]), -3, 3)
            X[i] = np.clip(np.round(X[i] + V[i]), 0, 3).astype(int)
            fit = calculate_fitness_v2(X[i], d_matrix, U_i, D_i, R_up, params, lambda_i, mu_i, mode=mode)
            if fit < pbest_fit[i]:
                pbest_fit[i], pbest_X[i] = fit, X[i].copy()
                if fit < gbest_fit: gbest_fit, gbest_X = fit, X[i].copy()
        history.append(gbest_fit)
    return gbest_X, gbest_fit, history

def run_baseline_v2(decision_type, d_matrix, U_i, D_i, R_up, params, lambda_i, mu_i, mode='average'):
    """新增基准算法通用接口，支持 mode 参数"""
    if decision_type == 'alao': decision = np.zeros(len(d_matrix), dtype=int)
    elif decision_type == 'almao': decision = np.ones(len(d_matrix), dtype=int)
    elif decision_type == 'arao': decision = np.random.randint(0, 4, len(d_matrix))
    else: raise ValueError("Unknown decision type")
    return calculate_fitness_v2(decision, d_matrix, U_i, D_i, R_up, params, lambda_i, mu_i, mode=mode)
