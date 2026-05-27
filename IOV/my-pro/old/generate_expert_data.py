import os
import sys
import numpy as np
import random
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# 将当前目录和 baseline 目录加入路径，防止 Windows 多进程下找不到模块及相对导入失败
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "baseline"))

from my_iov_env import ResidualIoVEnv
from baseline import env_v2_core
from baseline.algorithms import run_acoras_levy_v2

# ==========================================
# 核心修复：定义进程级全局变量，防止内存爆表
# ==========================================
global_env = None

def worker_init():
    """
    子进程初始化函数：每个 CPU 线程在启动时，只实例化一次环境！
    彻底阻断 edge_sim_py 无限膨胀导致的 32GB 内存泄漏。
    """
    global global_env
    global_env = ResidualIoVEnv()

def worker_task(seed_val):
    """
    独立工作进程
    """
    global global_env
    np.random.seed(seed_val)
    random.seed(int(seed_val)) # 对齐 random 引擎的随机性
    
    # 复用该线程专属的沙盘环境
    env = global_env
    params = env_v2_core.get_env_params_v2()
    n_tasks = len(env.vehicles)
    
    # reset 是安全的，它不会无脑新增对象
    obs = env.reset()
    
    # ==========================================
    # 核心整改：环境数据增强 (Counterfactual Exploration)
    # 防止网络学习到 eMBB 100% 上云的 Shortcut (状态-动作坍缩)
    # ==========================================
    rand_aug = np.random.rand()
    if rand_aug < 0.20:
        # 20% 概率：核心网切片带宽雪崩 (10Mbps - 50Mbps)
        env.r_ec = random.uniform(10.0, 50.0) * 1e6
        # 更新状态矩阵中给智能体看的特征
        for i in range(n_tasks): obs[i][5] = env.r_ec / (500.0 * 1e6)
    elif rand_aug < 0.30:
        # 10% 概率：云端陷入极限拥塞 (传播抖动飙升到 0.5s - 1.0s)
        # 注意：这里的 prop_ec_jitter 是环境在 step 和 expert 里动态生成的
        # 为了让 expert 感知到，我们需要把它作为一个硬变量塞入 env，或者塞入 params
        # 为了不破坏现有架构，我们可以直接给环境加一个标记
        env.force_cloud_congestion = True
    else:
        env.force_cloud_congestion = False
        
    # 将 env 内动态生成的算力池注入到 params 字典中，供 expert 使用
    params['f_local'] = env.f_local
    params['f_mec'] = env.f_mec
    params['f_offsite'] = env.f_offsite
    # 由于环境不再有固定的 r_eo / r_ec 随机区间（已锁死 500M），直接读取环境中的值
    params['r_eo'] = env.r_eo
    params['r_ec'] = env.r_ec
    params['force_cloud_congestion'] = getattr(env, 'force_cloud_congestion', False)
    
    # 提取物理参数
    d_matrix = np.zeros(n_tasks)
    U_i = np.zeros(n_tasks)
    D_i = np.zeros(n_tasks)
    lambda_i = np.zeros(n_tasks)
    mu_i = np.zeros(n_tasks)
    for idx, v in enumerate(env.vehicles):
        d_matrix[idx] = np.sqrt(v.coordinates[0]**2 + v.coordinates[1]**2) 
        U_i[idx] = v.U_i
        D_i[idx] = v.D_i
        lambda_i[idx] = v.lambda_i
        mu_i[idx] = v.mu_i
        
    # 信道计算 (True Physical V2)
    channel_gain = env_v2_core.calculate_channel_gain_v2(d_matrix)
    R_up = env_v2_core.get_uplink_rate_v2(params['B'], n_tasks, params['p_ue'], channel_gain, params['sigma2'])
    
    # 专家寻优 (挂载异构 QoS 权重)
    optimal_action, _, _ = run_acoras_levy_v2(
        d_matrix, U_i, D_i, R_up, params, lambda_i, mu_i,
        pop_size=30, max_iter=30, mode='convex'
    )
    
    return {'state': obs, 'action': optimal_action}

def generate_expert_dataset_multiprocess(num_samples=100000, save_path='data/expert_dataset.npy'):
    # 自动检查并创建 data 文件夹
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    total_threads = cpu_count()
    use_threads = max(1, total_threads - 2) 
    
    print("="*50)
    print("  启动 9700X 多进程金牌数据大生产 (内存安全版)  ")
    print(f"[*] 目标数据量: {num_samples} 条")
    print(f"[*] 保存路径: {save_path}")
    print(f"[*] 实际调用线程: {use_threads} (已保留系统线程)")
    print("="*50)
    
    dataset = []
    seeds = np.random.randint(0, 2**31 - 1, size=num_samples) # 修正越界问题
    
    # 核心修改：传入 initializer=worker_init，并设置 maxtasksperchild=2000 达到算力与内存的黄金平衡
    with Pool(processes=use_threads, initializer=worker_init, maxtasksperchild=2000) as pool:
        for result in tqdm(pool.imap_unordered(worker_task, seeds), total=num_samples, desc="Generating"):
            dataset.append(result)
            
    np.save(save_path, dataset)
    print(f"\n[OK] {num_samples}条专家数据已提炼完毕，安全保存至: {save_path}")

if __name__ == "__main__":
    # 执行大生产！
    generate_expert_dataset_multiprocess(num_samples=100000)
