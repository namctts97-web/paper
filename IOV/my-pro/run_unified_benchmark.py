import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

from my_iov_env import ResidualIoVEnv
from residual_ppo_agent import ResidualPPOAgent

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

def run_evaluation(algo_name, seed, max_episodes=2000):
    set_seed(seed)
    env = ResidualIoVEnv()
    sample_state = env.reset()
    flat_state_dim = sample_state.flatten().shape[0]
    action_dim = len(env.vehicles) * 4
    lr_online = 1e-5
    
    agent = ResidualPPOAgent(state_dim=flat_state_dim, action_dim=action_dim, lr=lr_online, ablation_mode=algo_name)
    
    # 加载模型
    if algo_name == 'ppo':
        agent.load_model('model/ppo_converged.pth')
    elif algo_name == 'ours':
        agent.load_expert('model/prior_dnn_expert.pth')
        agent.load_model('model/ours_converged.pth')
    elif algo_name == 'prior':
        agent.load_expert('model/prior_dnn_expert.pth')
        
        
    steps_per_episode = 200
    hist_cost = []
    hist_gate = []
    hist_urllc = []
    
    for episode in range(1, max_episodes + 1):
        # 严格的灾难时间线
        if episode == 500: env.trigger_capacity_avalanche()
        elif episode == 800: env.recover_from_avalanche()
        elif episode == 1300: env.trigger_traffic_flood()
        elif episode == 1600: env.recover_from_flood()
        
        state = env.reset()
        ep_cost = 0
        ep_urllc = 0
        ep_gate = []
        for t in range(steps_per_episode):
            res = agent.select_action(state)
            action, logprob, bw = res[0], res[1], res[3]
            next_state, reward, done, info = env.step(action)
            agent.store_transition((state, action, logprob, reward, done))
            state = next_state
            ep_cost += info['avg_cost']
            ep_urllc += info.get('urllc_violation_rate', 0.0)
            if algo_name == 'ours' and res[2] is not None:
                ep_gate.append(res[2]) 
            if done: break
        
        avg_cost = ep_cost / steps_per_episode
        hist_cost.append(avg_cost)
        hist_urllc.append(ep_urllc / steps_per_episode)
        if len(ep_gate) > 0:
            hist_gate.append(np.mean(ep_gate))
        else:
            hist_gate.append(0.0)
            
        if episode % 100 == 0:
            print(f"[{algo_name.upper()} | Seed {seed}] Ep {episode} | Cost: {avg_cost:.2f}")

    return hist_cost, hist_gate, hist_urllc

def worker(args):
    return run_evaluation(*args)

def smooth(scalars, weight=0.85):  
    if len(scalars) == 0: return scalars
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)

def main():
    os.makedirs('image', exist_ok=True)
    algs = ['ours', 'ppo', 'prior']
    seeds = [10, 20, 30, 40, 50]
    
    results_cost = {alg: [] for alg in algs}
    results_gate = []
    results_urllc = {alg: [] for alg in algs}
    
    tasks = []
    for alg in algs:
        for seed in seeds:
            tasks.append((alg, seed, 2000))
            
    print("\n[BENCHMARK] Starting multi-seed concurrent evaluation...")
    with ProcessPoolExecutor(max_workers=min(os.cpu_count(), len(tasks))) as executor:
        all_res = list(executor.map(worker, tasks))
        
    idx = 0
    for alg in algs:
        for seed in seeds:
            cost, gate, urllc = all_res[idx]
            results_cost[alg].append(cost)
            results_urllc[alg].append(urllc)
            if alg == 'ours':
                results_gate.append(gate)
            idx += 1
            
    def get_stats(data):
        data = np.array(data)
        return np.mean(data, axis=0), np.std(data, axis=0)

    x = np.arange(1, 2001)

    # =========================================================
    # 图 1: Gate Radar (门控 G(t) 的雷达图，验证异常检测隔离性能)
    # =========================================================
    mean_gate, std_gate = get_stats(results_gate)
    plt.figure(figsize=(10, 4))
    plt.plot(x, mean_gate, color='purple', label='Gate G(t)', linewidth=2.0)
    plt.fill_between(x, np.clip(mean_gate-std_gate, 0, 1), np.clip(mean_gate+std_gate, 0, 1), color='purple', alpha=0.3)
    
    plt.axvspan(500, 800, color='red', alpha=0.1, label='OOD: Avalanche')
    plt.axvspan(1300, 1600, color='blue', alpha=0.1, label='OOD: Flood')
    plt.ylim(-0.1, 1.1)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.title("Gate G(t) Radar Tracker (2000 Episodes, 5 Seeds)")
    plt.xlabel("Episodes")
    plt.ylabel("G(t)")
    plt.tight_layout()
    plt.savefig('image/gate_radar.png', dpi=300)
    print("\n[OK] Generated gate_radar.png")
    
    # =========================================================
    # 图 2: Long term ablation (2000 轮超长周期基线与消融对比图) & The Tribunal
    # =========================================================
    plt.figure(figsize=(14, 8))
    colors = {'ours': 'red', 'ppo': 'gray', 'prior': 'blue'}
    labels = {'ours': 'OURS (Dual-Brain Res-Gating)', 'ppo': 'Context-Aware PPO (Baseline)', 'prior': 'Expert Prior (Static)'}
    
    tribunal_text = "The Tribunal: Resilience Metrics\n--------------------------------------\n"
    tribunal_text += f"{'Algorithm':<25} | {'Δ_max':<8} | {'AUVC'}\n"
    tribunal_text += "--------------------------------------\n"
    
    for alg in algs:
        mean_c, std_c = get_stats(results_cost[alg])
        mean_c_smooth = smooth(mean_c, 0.6)
        std_c_smooth = smooth(std_c, 0.6)
        
        # Calculate Delta_max (Max Degradation)
        # Steady state before 500: avg cost between 400 and 500
        R_steady = np.mean(mean_c[400:500])
        # Avalanche peak: max cost between 500 and 600
        R_peak_avalanche = np.max(mean_c[500:600])
        Delta_max = R_peak_avalanche - R_steady
        
        # Calculate AUVC (Area Under Vulnerability Curve for Avalanche 500-800)
        # AUVC = Integral of (R_t - R_steady) for t in [500, 800]
        # We only integrate the positive vulnerability (when cost > R_steady)
        auvc = np.sum(np.maximum(0, mean_c[500:800] - R_steady))
        
        tribunal_text += f"{labels[alg].split(' ')[0]:<25} | {Delta_max:>8.2f} | {auvc:>8.2f}\n"
        
        plt.plot(x, mean_c_smooth, color=colors[alg], label=labels[alg], linewidth=2.0)
        plt.fill_between(x, mean_c_smooth - std_c_smooth, mean_c_smooth + std_c_smooth, color=colors[alg], alpha=0.2)
        
    plt.axvspan(500, 800, color='red', alpha=0.1, label='OOD: Avalanche')
    plt.axvspan(1300, 1600, color='blue', alpha=0.1, label='OOD: Flood')
    
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.title("2000-Episode Zero-Shot Resilience Benchmark with 95% Confidence Intervals")
    plt.xlabel("Episodes")
    plt.ylabel("System Penalty (Cost)")
    
    # Add Tribunal Text Box
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.02, 0.95, tribunal_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props, family='monospace')
             
    plt.tight_layout()
    plt.savefig('image/long_term_ablation_curve.png', dpi=300)
    print("[OK] Generated long_term_ablation_curve.png")
    
    # =========================================================
    # 图 3: URLLC 10ms 尾部时延违规率
    # =========================================================
    plt.figure(figsize=(12, 6))
    for alg in algs:
        mean_u, std_u = get_stats(results_urllc[alg])
        mean_u_smooth = smooth(mean_u, 0.6)
        std_u_smooth = smooth(std_u, 0.6)
        
        plt.plot(x, mean_u_smooth, color=colors[alg], label=labels[alg], linewidth=2.0)
        plt.fill_between(x, mean_u_smooth - std_u_smooth, mean_u_smooth + std_u_smooth, color=colors[alg], alpha=0.2)
        
    plt.axvspan(500, 800, color='red', alpha=0.1, label='OOD: Avalanche')
    plt.axvspan(1300, 1600, color='blue', alpha=0.1, label='OOD: Flood')
    
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.title("URLLC 10ms Hard-Constraint Violation Rate (99.99th Pct Focus)")
    plt.xlabel("Episodes")
    plt.ylabel("Outage Probability P(Lat > 10ms)")
    plt.tight_layout()
    plt.savefig('image/urllc_violation_rate.png', dpi=300)
    print("[OK] Generated urllc_violation_rate.png")

if __name__ == "__main__":
    main()
