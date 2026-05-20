import matplotlib.pyplot as plt
import numpy as np
import os
import env_config
from algorithms import run_acoras_cauchy_v2, run_acoras_levy_v2, run_dpsao_v2, run_baseline_v2

def run_experiment_fig9(channel_mode, fitness_mode, alg_type='Cauchy'):
    """
    运行 Fig 9 实验：车辆规模压力测试 (使用 v2 算法接口)
    """
    N_list = [70, 85, 100, 115]
    params = env_config.get_env_params()
    
    results = {
        'ACORAS': [],
        'DPSAO': [],
        'ALAO': [],
        'ALMAO': [],
        'ARAO': []
    }

    for N in N_list:
        print(f"[*] Testing N = {N} ({channel_mode}/{fitness_mode})...")
        d_matrix, U_i, D_i, _, _ = env_config.generate_topology(N, seed=42)
        
        # 信道增益设置 (已由 env_config 内核处理单位换算)
        gain = env_config.calculate_channel_gain(d_matrix, mode=channel_mode)
        R_up = env_config.get_uplink_rate(params['B'], N, params['p_ue'], gain, params['sigma2'])

        # 基准算法 (使用 v2 接口支持模式切换)
        results['ALAO'].append(run_baseline_v2('alao', d_matrix, U_i, D_i, R_up, params, mode=fitness_mode))
        results['ALMAO'].append(run_baseline_v2('almao', d_matrix, U_i, D_i, R_up, params, mode=fitness_mode))
        results['ARAO'].append(run_baseline_v2('arao', d_matrix, U_i, D_i, R_up, params, mode=fitness_mode))
        
        # 核心算法 (使用 v2 接口)
        if alg_type == 'Cauchy':
            _, fit, _ = run_acoras_cauchy_v2(d_matrix, U_i, D_i, R_up, params, max_iter=30, mode=fitness_mode)
            results['ACORAS'].append(fit)
            _, fit_dpsao, _ = run_dpsao_v2(d_matrix, U_i, D_i, R_up, params, max_iter=30, mode=fitness_mode)
            results['DPSAO'].append(fit_dpsao)
        else: # Levy
            _, fit, _ = run_acoras_levy_v2(d_matrix, U_i, D_i, R_up, params, max_iter=30, mode=fitness_mode)
            results['ACORAS'].append(fit)
            _, fit_dpsao, _ = run_dpsao_v2(d_matrix, U_i, D_i, R_up, params, max_iter=30, mode=fitness_mode)
            results['DPSAO'].append(fit_dpsao)

    return N_list, results

def plot_results(N_list, results, title, filename):
    plt.figure(figsize=(10, 6))
    markers = {'ACORAS': 'o', 'DPSAO': 's', 'ALAO': '^', 'ALMAO': 'p', 'ARAO': 'x'}
    colors = {'ACORAS': '#0072BD', 'DPSAO': '#D95319', 'ALAO': '#EDB120', 'ALMAO': '#77AC30', 'ARAO': '#7E2F8E'}
    
    for alg in results:
        plt.plot(N_list, results[alg], marker=markers[alg], color=colors[alg], label=alg, linewidth=2)
    
    plt.xlabel('Number of vehicles')
    plt.ylabel('Total system cost')
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    save_dir = os.path.join(os.path.dirname(__file__), "image")
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, filename), dpi=300)
    print(f"[*] Saved {filename}")
    plt.close()

def main():
    # 1. 复现伪造现场
    print("\n--- Phase 1: Reproducing Flawed Fig 9 ---")
    N_list, results_flawed = run_experiment_fig9('flawed', 'average', 'Cauchy')
    plot_results(N_list, results_flawed, 'Fig. 9 Impact of number of vehicles (Flawed)', 'fig9_flawed.png')

    # 2. 重建真实物理基准
    print("\n--- Phase 2: Rebuilding True Fig 9 ---")
    N_list, results_true = run_experiment_fig9('physical', 'convex', 'Levy')
    plot_results(N_list, results_true, 'Fig. 9 Impact of number of vehicles (True Physical)', 'fig9_true.png')

if __name__ == "__main__":
    main()
