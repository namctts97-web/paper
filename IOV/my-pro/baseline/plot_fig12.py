import matplotlib.pyplot as plt
import numpy as np
import os
import env_config
from algorithms import run_acoras_cauchy_v2, run_acoras_levy_v2, run_dpsao_v2, run_baseline_v2

def run_experiment_fig12(channel_mode, fitness_mode, alg_type='Cauchy'):
    L_list = [0.1, 0.3, 0.5, 0.7, 0.9]
    params = env_config.get_env_params()
    n_tasks = 80
    results = {'ACORAS': [], 'DPSAO': [], 'ALAO': [], 'ALMAO': [], 'ARAO': []}

    for L in L_list:
        print(f"[*] Testing Lambda = {L} ({channel_mode}/{fitness_mode})...")
        d_matrix, U_i, D_i, _, _ = env_config.generate_topology(n_tasks, seed=42)
        params['lambda'] = L
        params['mu'] = 1 - L
        
        params['mu'] = 1 - L
        
        # 信道增益设置 (已由 env_config 内核处理单位换算)
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
    return L_list, results

def plot_bar_results(x_list, results, title, filename, xlabel):
    plt.figure(figsize=(12, 7))
    
    # 柱状图参数
    bar_width = 0.15
    index = np.arange(len(x_list))
    
    algs = ['ACORAS', 'DPSAO', 'ALAO', 'ALMAO', 'ARAO']
    colors = {'ACORAS': '#0072BD', 'DPSAO': '#D95319', 'ALAO': '#EDB120', 'ALMAO': '#77AC30', 'ARAO': '#7E2F8E'}
    
    for i, alg in enumerate(algs):
        plt.bar(index + i * bar_width, results[alg], bar_width, label=alg, color=colors[alg], edgecolor='black', alpha=0.8)
    
    plt.xlabel(xlabel)
    plt.ylabel('Total system cost')
    plt.title(title)
    plt.xticks(index + 2 * bar_width, x_list)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    save_dir = os.path.join(os.path.dirname(__file__), "image")
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, filename), dpi=300)
    plt.close()

if __name__ == "__main__":
    print("\n--- Phase 1: Reproducing Flawed Fig 12 (Bar) ---")
    L_list, res_f = run_experiment_fig12('flawed', 'average', 'Cauchy')
    plot_bar_results(L_list, res_f, 'Fig. 12 Impact of Weight Lambda (Flawed)', 'fig12_flawed.png', 'Latency weight lambda')
    
    print("\n--- Phase 2: Rebuilding True Fig 12 (Bar) ---")
    L_list, res_t = run_experiment_fig12('physical', 'convex', 'Levy')
    plot_bar_results(L_list, res_t, 'Fig. 12 Impact of Weight Lambda (True)', 'fig12_true.png', 'Latency weight lambda')
