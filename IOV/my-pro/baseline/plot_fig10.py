import matplotlib.pyplot as plt
import numpy as np
import os
import env_config
from algorithms import run_acoras_cauchy_v2, run_acoras_levy_v2, run_dpsao_v2, run_baseline_v2

def run_experiment_fig10(channel_mode, fitness_mode, alg_type='Cauchy'):
    U_targets = [200, 300, 400, 500, 600, 700] # KB
    params = env_config.get_env_params()
    n_tasks = 80
    
    # 1. 生成异构拓扑基准
    d_matrix, U_base, D_i, _, _ = env_config.generate_topology(n_tasks, seed=42)
    U_base_mean = np.mean(U_base) / (8 * 1024) # 转换为 KB 均值
    
    results = {'ACORAS': [], 'DPSAO': [], 'ALAO': [], 'ALMAO': [], 'ARAO': []}

    for U_target in U_targets:
        print(f"[*] Testing U_avg = {U_target} KB ({channel_mode}/{fitness_mode})...")
        
        # 2. 比例缩放：保留异构性，不使用 np.full
        scale_factor = U_target / U_base_mean
        U_i = U_base * scale_factor
        
        # 信道增益设置 (内核已处理单位)
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
    return U_targets, results

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
    print("\n--- Phase 1: Reproducing Flawed Fig 10 ---")
    U_list, res_f = run_experiment_fig10('flawed', 'average', 'Cauchy')
    plot_results(U_list, res_f, 'Fig. 10 Impact of Task Data (Flawed)', 'fig10_flawed.png', 'Avg Task data U (KB)')
    print("\n--- Phase 2: Rebuilding True Fig 10 ---")
    U_list, res_t = run_experiment_fig10('physical', 'convex', 'Levy')
    plot_results(U_list, res_t, 'Fig. 10 Impact of Task Data (True)', 'fig10_true.png', 'Avg Task data U (KB)')
