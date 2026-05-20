import matplotlib.pyplot as plt
import numpy as np
import os
import env_config
from algorithms import run_acoras_cauchy_v2, run_acoras_levy_v2, run_dpsao_v2, run_baseline_v2

def run_experiment_fig11(channel_mode, fitness_mode, alg_type='Cauchy'):
    D_targets = [600, 900, 1200, 1500] # Mc
    params = env_config.get_env_params()
    n_tasks = 80
    
    # 1. 生成异构拓扑基准 (包含 20/80 异构计算量)
    d_matrix, U_i, D_base, _, _ = env_config.generate_topology(n_tasks, seed=42)
    D_base_mean = np.mean(D_base) / 1e6 # 转换为 Mc 均值
    
    results = {'ACORAS': [], 'DPSAO': [], 'ALAO': [], 'ALMAO': [], 'ARAO': []}

    for D_target in D_targets:
        print(f"[*] Testing D_avg = {D_target} Mc ({channel_mode}/{fitness_mode})...")
        
        # 2. 比例缩放：确保凸优化基石（任务差异性）不丢失
        scale_factor = D_target / D_base_mean
        D_i = D_base * scale_factor
        
        # 信道增益设置 (内核处理)
        gain = env_config.calculate_channel_gain(d_matrix, mode=channel_mode)
        R_up = env_config.get_uplink_rate(params['B'], n_tasks, params['p_ue'], gain, params['sigma2'])

        results['ALAO'].append(run_baseline_v2('alao', d_matrix, U_i, D_i, R_up, params, mode=fitness_mode))
        results['ALMAO'].append(run_baseline_v2('almao', d_matrix, U_i, D_i, R_up, params, mode=fitness_mode))
        results['ARAO'].append(run_baseline_v2('arao', d_matrix, U_i, D_i, R_up, params, mode=fitness_mode))
        
        if alg_type == 'Cauchy':
            _, fit, _ = run_acoras_cauchy_v2(d_matrix, U_i, D_i, R_up, params, max_iter=30, mode=fitness_mode)
            results['ACORAS'].append(fit)
            _, fit_dpsao, _ = run_dpsao_v2(d_matrix, U_i, D_i, R_up, params, max_iter=30, mode=fitness_mode)
            results['DPSAO'].append(fit_dpsao)
        else:
            _, fit, _ = run_acoras_levy_v2(d_matrix, U_i, D_i, R_up, params, max_iter=30, mode=fitness_mode)
            results['ACORAS'].append(fit)
            _, fit_dpsao, _ = run_dpsao_v2(d_matrix, U_i, D_i, R_up, params, max_iter=30, mode=fitness_mode)
            results['DPSAO'].append(fit_dpsao)
    return D_targets, results

def plot_results(x_list, results, title, filename, xlabel):
    plt.figure(figsize=(10, 6))
    markers = {'ACORAS': 'o', 'DPSAO': 's', 'ALAO': '^', 'ALMAO': 'p', 'ARAO': 'x'}
    colors = {'ACORAS': '#0072BD', 'DPSAO': '#D95319', 'ALAO': '#EDB120', 'ALMAO': '#77AC30', 'ARAO': '#7E2F8E'}
    for alg in results:
        plt.plot(x_list, results[alg], marker=markers[alg], color=colors[alg], label=alg, linewidth=2)
    plt.xlabel(xlabel)
    plt.ylabel('Total system cost')
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    save_dir = os.path.join(os.path.dirname(__file__), "image")
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, filename), dpi=300)
    plt.close()

if __name__ == "__main__":
    print("\n--- Phase 1: Reproducing Flawed Fig 11 ---")
    D_list, res_f = run_experiment_fig11('flawed', 'average', 'Cauchy')
    plot_results(D_list, res_f, 'Fig. 11 Impact of Task CPU (Flawed)', 'fig11_flawed.png', 'Avg Task workload D (Mc)')
    print("\n--- Phase 2: Rebuilding True Fig 11 ---")
    D_list, res_t = run_experiment_fig11('physical', 'convex', 'Levy')
    plot_results(D_list, res_t, 'Fig. 11 Impact of Task CPU (True)', 'fig11_true.png', 'Avg Task workload D (Mc)')
