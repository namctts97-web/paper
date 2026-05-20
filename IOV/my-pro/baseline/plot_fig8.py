import matplotlib.pyplot as plt
import numpy as np
import env_config
import algorithms

def reproduce_fig8():
    print("="*50)
    print("Running Fig. 8: ACORAS and DPSAO Convergence Comparison")
    print("="*50)

    # 1. 环境初始化
    n_tasks = 80
    params = env_config.get_env_params()
    d_matrix, U_i, D_i, _, _ = env_config.generate_topology(n_tasks, seed=42)
    
    # 使用 flawed 信道模型对齐原论文
    channel_gain = env_config.calculate_channel_gain(d_matrix, mode='flawed')
    R_up = env_config.get_uplink_rate(params['B'], n_tasks, params['p_ue'], channel_gain, params['sigma2'])

    # 2. 运行算法并记录历史
    max_iter = 50
    pop_size = 50

    print(f"[*] 运行 DPSAO (对照组)...")
    _, _, history_dpsao = algorithms.run_dpsao(
        d_matrix, U_i, D_i, R_up, params, pop_size=pop_size, max_iter=max_iter
    )

    print(f"[*] 运行 ACORAS_Cauchy (论文复刻版)...")
    _, _, history_acoras = algorithms.run_acoras_cauchy(
        d_matrix, U_i, D_i, R_up, params, pop_size=pop_size, max_iter=max_iter
    )

    # 3. 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_iter + 1), history_dpsao, 'r-s', label='DPSAO', markevery=5)
    plt.plot(range(1, max_iter + 1), history_acoras, 'b-o', label='ACORAS (Cauchy)', markevery=5)
    
    plt.xlabel('Number of iterations')
    plt.ylabel('Total system cost')
    plt.title('Fig. 8 Convergence of ACORAS algorithm and DPSAO algorithm')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 保存结果
    import os
    save_dir = os.path.join(os.path.dirname(__file__), "image")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, "fig8_convergence.png")
    plt.savefig(save_path)
    print(f"[*] Image saved to: {save_path}")
    plt.show()

def reproduce_true_fig8():
    """
    使用真实物理公式（正确 Shannon 公式 + 凸优化分配）进行收敛测试
    """
    from scipy.special import gamma
    import os
    print("="*50)
    print("Running Fig. 8: Convergence under True Mathematical Physics")
    print("="*50)

    # 1. 物理参数与信道模型
    def get_true_params():
        return {
            'B': 40 * 1e6, 'sigma2': 2 * 1e-13, 'p_ue': 0.5,
            'f_local': 0.1 * 1e9, 'f_mec': 20 * 1e9, 'f_offsite': 40 * 1e9, 'f_cloud': 100 * 1e9,
            'r_eo': 10 * 8 * 1e6, 'r_ec': 2 * 8 * 1e6,
            'k_energy': 1e-27, 'lambda': 0.5, 'mu': 0.5
        }

    def get_true_channel_rate(d_matrix, B, n_tasks, p_ue, sigma2):
        pl_db = 127 + 30 * np.log10(d_matrix / 1000.0) 
        h_linear = 10 ** (-pl_db / 10)
        return (B / n_tasks) * np.log2(1 + (p_ue * h_linear) / sigma2)

    # 2. 适应度与算法引擎 (内部定义以确保纯粹性)
    from fitness import calculate_fitness
    from algorithms import run_acoras_levy, run_dpsao

    np.random.seed(42)
    params = get_true_params()
    n_tasks = 80
    d_matrix = np.random.rand(n_tasks) * 1000 + 10 
    R_up = get_true_channel_rate(d_matrix, params['B'], n_tasks, params['p_ue'], params['sigma2'])
    
    # 任务异构化
    U_i = np.random.randint(200, 701, n_tasks) * 8 * 1024
    D_i = np.zeros(n_tasks)
    heavy_idx = np.random.choice(n_tasks, int(0.2 * n_tasks), replace=False)
    light_idx = np.setdiff1d(np.arange(n_tasks), heavy_idx)
    D_i[heavy_idx] = np.random.randint(4000, 6001, len(heavy_idx)) * 1e6
    D_i[light_idx] = np.random.randint(50, 201, len(light_idx)) * 1e6

    print("[*] Running DPSAO (True Formulation)...")
    # 使用算法模块中的函数，但传入真实参数和 mode='convex' (DPSAO 默认用 average，此处为对比也切为 convex)
    # 注意：此处直接使用算法模块以保持统一
    _, _, history_dpsao = run_dpsao(d_matrix, U_i, D_i, R_up, params, max_iter=50)

    print("[*] Running ACORAS_Levy (True Formulation)...")
    _, _, history_acoras = run_acoras_levy(d_matrix, U_i, D_i, R_up, params, max_iter=50)

    # 3. 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 51), history_dpsao, color='#D95319', marker='s', linestyle='-', linewidth=2, markersize=6, label='DPSAO (True Formulation)')
    plt.plot(range(1, 51), history_acoras, color='#0072BD', marker='o', linestyle='-', linewidth=2, markersize=6, label='ACORAS-Levy (True Formulation)')
    
    plt.xlabel('Number of iterations', fontsize=12)
    plt.ylabel('Total system cost', fontsize=12)
    plt.title('Fig. 8 Convergence (Under True Mathematical Physics)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    save_dir = os.path.join(os.path.dirname(__file__), "image")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, "true_fig8_convergence.png")
    plt.savefig(save_path, dpi=300)
    print(f"[*] Image saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    reproduce_fig8()
    reproduce_true_fig8()
