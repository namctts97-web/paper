import os
import sys
import numpy as np
import random
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# 确保能正确导入项目模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from iov_env_v3 import IoVEnvV3

def run_heuristic_baseline(env, num_vehicles, max_iter=1000):
    """
    内置的高性能离散遗传算法（Discrete GA），作为 Baseline 预言机。
    利用群体信息交互（交叉、变异）跳出局部最优，全局搜索 4^10 解空间。
    绝对锁定 env.evaluate_actions 进行无状态同构评估。
    """
    pop_size = 40
    # 初始化种群
    population = [np.random.randint(0, 4, size=num_vehicles) for _ in range(pop_size)]
    
    # 评估初始种群
    fitness = [env.evaluate_actions(ind) for ind in population]
    best_idx = np.argmin(fitness)
    best_action = np.copy(population[best_idx])
    best_cost = fitness[best_idx]
    
    for _ in range(max_iter):
        new_population = []
        # 精英保留
        new_population.append(np.copy(best_action))
        
        while len(new_population) < pop_size:
            # 锦标赛选择
            i1, i2 = random.sample(range(pop_size), 2)
            p1 = population[i1] if fitness[i1] < fitness[i2] else population[i2]
            i1, i2 = random.sample(range(pop_size), 2)
            p2 = population[i1] if fitness[i1] < fitness[i2] else population[i2]
            
            # 单点交叉
            pt = random.randint(1, num_vehicles - 1)
            c1 = np.concatenate([p1[:pt], p2[pt:]])
            
            # 突变 (突变率 0.15)
            if random.random() < 0.15:
                mut_pt = random.randint(0, num_vehicles - 1)
                available_actions = [0, 1, 2, 3]
                if c1[mut_pt] in available_actions:
                    available_actions.remove(c1[mut_pt])
                c1[mut_pt] = random.choice(available_actions)
                
            new_population.append(c1)
            
        population = new_population[:pop_size]
        fitness = [env.evaluate_actions(ind) for ind in population]
        
        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < best_cost:
            best_cost = fitness[current_best_idx]
            best_action = np.copy(population[current_best_idx])
            
    return best_action, best_cost

# ==========================================
# 核心修复：定义进程级全局变量，防止内存爆表
# ==========================================
global_env = None

def worker_init():
    """
    子进程初始化函数：每个 CPU 线程在启动时，只实例化一次环境！
    彻底阻断无脑 new 对象导致的内存泄漏。
    """
    global global_env
    global_env = IoVEnvV3(num_vehicles=10)

def worker_task(seed_val):
    """
    独立工作进程
    """
    global global_env
    np.random.seed(seed_val)
    random.seed(int(seed_val))
    
    # 复用该线程专属的沙盘环境
    env = global_env
    
    # 宏观 OOD 灾难随机注入 (完全 IID 采样)
    r_macro = random.random()
    is_ood = False
    if r_macro < 0.2:
        env.trigger_flood()
        is_ood = True
    elif r_macro < 0.4:
        env.trigger_avalanche()
        is_ood = True
    else:
        env.is_compute_avalanche = False
        env.is_traffic_flood = False
        env.is_core_congestion = False
        
    # reset 是安全的，生成独立的 IID 宇宙大爆炸快照
    obs = env.reset()
    
    # ==========================================
    # 核心整改：环境数据增强 (Counterfactual Exploration)
    # 防止网络学习到 eMBB 100% 上云的 Shortcut，强制激活 MEC 与 Offsite
    # ==========================================
    rand_aug = random.random()
    if rand_aug < 0.20:
        # 20% 概率：核心网切片带宽雪崩 (10Mbps - 50Mbps)
        env.r_ec = random.uniform(10.0, 50.0) * 1e6
        # 更新状态矩阵中给智能体看的特征 (core_cong 位于 obs 的 index 2)
        core_cong = env.r_ec_base / env.r_ec
        obs[2] = core_cong
    elif rand_aug >= 0.20 and rand_aug < 0.40:
        # 另外 20% 概率：云端陷入极限拥塞 (传播抖动飙升到 0.2s - 0.5s)
        # 破坏云端对于 eMBB 的吸引力，逼迫其流向 MEC
        for v in env.vehicles:
            v['current_jitter'] = random.uniform(0.2, 0.5)

    # 专家寻优 (使用 GA 基线预言机)
    best_a, best_c = run_heuristic_baseline(env, env.num_vehicles, max_iter=150)
    
    # ==========================================
    # 绝对数据物理海关 (Data Quarantine)
    # ==========================================
    avg_cost = best_c / env.num_vehicles
    NORMAL_COST_THRESHOLD = 15.0
    OOD_COST_THRESHOLD = 1500.0
    
    is_valid_data = False
    if not is_ood and avg_cost <= NORMAL_COST_THRESHOLD:
        is_valid_data = True
    elif is_ood and avg_cost <= OOD_COST_THRESHOLD:
        is_valid_data = True
        
    if is_valid_data:
        return {'state': obs, 'action': best_a}
    else:
        return None

def generate_expert_dataset_multiprocess(num_samples=100000, save_path='data/expert_dataset_v3_clean.npy'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    total_threads = cpu_count()
    use_threads = max(1, total_threads - 2) 
    
    print("="*60)
    print("  启动 9700X 多进程金牌数据大生产 (内存安全 + IID版)  ")
    print(f"[*] 目标数据量: {num_samples} 条")
    print(f"[*] 保存路径: {save_path}")
    print(f"[*] 实际调用线程: {use_threads} (已保留系统线程)")
    print("="*60)
    
    expert_states = []
    expert_actions = []
    
    # 生成安全的随机种子池
    seeds = np.random.randint(0, 2**31 - 1, size=int(num_samples * 1.5)) 
    
    pbar = tqdm(total=num_samples, desc="Generating Expert Data")
    
    # 核心修改：传入 initializer=worker_init，并设置 maxtasksperchild 防止碎片化
    with Pool(processes=use_threads, initializer=worker_init, maxtasksperchild=2000) as pool:
        for result in pool.imap_unordered(worker_task, seeds):
            if result is not None:
                expert_states.append(result['state'])
                expert_actions.append(result['action'])
                pbar.update(1)
                
            if len(expert_states) >= num_samples:
                break
                
    pbar.close()
    
    expert_states = np.array(expert_states)
    expert_actions = np.array(expert_actions)
    
    np.save(save_path, {'states': expert_states, 'actions': expert_actions})
    print(f"\n[OK] {len(expert_states)}条无偏IID专家数据已提炼完毕，安全保存至: {save_path}")

if __name__ == '__main__':
    generate_expert_dataset_multiprocess(num_samples=100000)
