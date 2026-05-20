import numpy as np
import env_config
import fitness
import algorithms

def main():
    print("="*50)
    print("IoV Task Offloading Paper Reproduction Simulation")
    print("="*50)

    # 1. 初始化环境与参数
    params = env_config.get_env_params()
    n_tasks = 80
    d_matrix, U_i, D_i, heavy_idx, light_idx = env_config.generate_topology(n_tasks, seed=42)
    
    # 获取双版本信道增益 (此处默认使用 flawed 以对齐原论文图表)
    # 若需测试真实物理模型，可切换为 'physical'
    channel_gain_flawed = env_config.calculate_channel_gain(d_matrix, mode='flawed')
    R_up = env_config.get_uplink_rate(params['B'], n_tasks, params['p_ue'], channel_gain_flawed, params['sigma2'])

    print(f"[*] 环境配置完成: 车辆数={n_tasks}, 异构任务={len(heavy_idx)}重/{len(light_idx)}轻")
    print(f"[*] 信道模型: flawed (用于图表对齐)")
    print("-" * 50)

    # 2. 执行基准算法
    print("[*] 正在计算基准算法...")
    cost_alao = algorithms.run_baseline_alao(d_matrix, U_i, D_i, R_up, params)
    cost_almao = algorithms.run_baseline_almao(d_matrix, U_i, D_i, R_up, params)
    cost_arao = algorithms.run_baseline_arao(d_matrix, U_i, D_i, R_up, params)
    print(f"    - ALAO (全本地): {cost_alao:.4f}")
    print(f"    - ALMAO (全MEC): {cost_almao:.4f}")
    print(f"    - ARAO (全随机): {cost_arao:.4f}")
    print("-" * 50)

    # 3. 执行核心算法：ACORAS_Cauchy (复刻版)
    print("[*] 启动 ACORAS_Cauchy (基础复刻版)...")
    _, best_fit_cauchy, _ = algorithms.run_acoras_cauchy(d_matrix, U_i, D_i, R_up, params, max_iter=50)
    print(f"    => 最终成本 (Cauchy): {best_fit_cauchy:.4f}")
    print("-" * 50)

    # 4. 执行终极版算法：ACORAS_Levy (降维打击版)
    print("[*] 启动 ACORAS_Levy (终极降维打击版)...")
    _, best_fit_levy, _ = algorithms.run_acoras_levy(d_matrix, U_i, D_i, R_up, params, max_iter=50)
    print(f"    => 最终成本 (Levy+LS): {best_fit_levy:.4f}")
    print("-" * 50)

    # 5. 性能对比总结
    improvement = (best_fit_cauchy - best_fit_levy) / best_fit_cauchy * 100
    print(f"Performance Improvement Summary: Ultimate version optimized by {improvement:.2f}%")
    print("="*50)

if __name__ == "__main__":
    main()
