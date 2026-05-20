import numpy as np
import env_config
import algorithms

def audit_results():
    n_tasks = 80
    params = env_config.get_env_params()
    d_matrix, U_i, D_i, _, _ = env_config.generate_topology(n_tasks, seed=42)
    channel_gain = env_config.calculate_channel_gain(d_matrix, mode='flawed')
    R_up = env_config.get_uplink_rate(params['B'], n_tasks, params['p_ue'], channel_gain, params['sigma2'])

    max_iter = 50
    pop_size = 50

    # 运行算法
    _, _, history_dpsao = algorithms.run_dpsao(d_matrix, U_i, D_i, R_up, params, pop_size=pop_size, max_iter=max_iter)
    _, _, history_acoras = algorithms.run_acoras_cauchy(d_matrix, U_i, D_i, R_up, params, pop_size=pop_size, max_iter=max_iter)

    print("\n--- 学术审计报告数据提取 ---")
    print(f"Gen 1 - DPSAO (Random): {history_dpsao[0]:.6f}")
    print(f"Gen 1 - ACORAS (Logistic): {history_acoras[0]:.6f}")
    
    # 查找 DPSAO 陷入死锁的代数 (假设连续 5 代变化小于 1e-6)
    deadlock_gen = -1
    for i in range(1, len(history_dpsao)):
        if abs(history_dpsao[i] - history_dpsao[i-1]) < 1e-6:
            if deadlock_gen == -1: deadlock_gen = i + 1
        else:
            deadlock_gen = -1
            
    print(f"DPSAO Deadlock Gen: {deadlock_gen}")

    print("\nACORAS Convergence Detail (Check for steps):")
    for i in range(len(history_acoras)):
        diff = 0 if i == 0 else history_acoras[i] - history_acoras[i-1]
        print(f"Gen {i+1:02d}: {history_acoras[i]:.6f} (Diff: {diff:.6f})")

if __name__ == "__main__":
    audit_results()
