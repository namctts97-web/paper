import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import torch
import matplotlib.pyplot as plt

from my_iov_env import ResidualIoVEnv
from residual_ppo_agent import ResidualPPOAgent
from baseline_gcn_ppo import GCN_PPO_Agent
from baseline_heuristic import Heuristic_Agent

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)

def run_evaluation(algo_name, max_episodes=500):
    print(f"\n==============================================")
    print(f"=== Online Adaptation Phase: {algo_name.upper()} ===")
    print(f"==============================================")
    
    # 强制重置种子，保证每个算法遭遇完全一致的物理环境和灾难序列
    set_seed(42)
    env = ResidualIoVEnv()
    
    sample_state = env.reset()
    flat_state_dim = sample_state.flatten().shape[0]
    action_dim = len(env.vehicles) * 4
    
    # 初始化并加载“完全体”预训练模型，学习率设为极小值 1e-5
    lr_online = 1e-5
    
    if algo_name == 'worst_fit':
        agent = Heuristic_Agent()
    elif algo_name == 'gcn':
        agent = GCN_PPO_Agent(lr=lr_online)
        agent.load_model('model/gcn_converged.pth')
    elif algo_name == 'ppo':
        agent = ResidualPPOAgent(state_dim=flat_state_dim, action_dim=action_dim, lr=lr_online, ablation_mode='ppo')
        agent.load_model('model/ppo_converged.pth')
    elif algo_name == 'prior':
        agent = ResidualPPOAgent(state_dim=flat_state_dim, action_dim=action_dim, lr=lr_online, ablation_mode='prior')
        agent.load_expert('model/prior_dnn_expert.pth')
    elif algo_name == 'ours':
        agent = ResidualPPOAgent(state_dim=flat_state_dim, action_dim=action_dim, lr=lr_online, ablation_mode='ours')
        agent.load_expert('model/prior_dnn_expert.pth')
        agent.load_model('model/ours_converged.pth')
        
    steps_per_episode = 200
    
    hist_latency, hist_energy, hist_success, hist_cost = [], [], [], []
    hist_brain_waves = [] # 仅 ours 记录
    
    for episode in range(1, max_episodes + 1):
        # 统一的灾难时间线
        if episode == 100: env.trigger_capacity_avalanche()
        elif episode == 200: env.recover_from_avalanche()
        elif episode == 300: env.trigger_traffic_flood()
        elif episode == 400: env.recover_from_flood()
            
        state = env.reset()
        ep_latency, ep_energy, ep_success, ep_cost = 0, 0, 0, 0
        ep_brain_wave = []
        
        for t in range(steps_per_episode):
            if algo_name == 'gcn':
                action, logprob, val = agent.select_action(state)
            elif algo_name == 'worst_fit':
                action = agent.select_action(state)
            else:
                action, logprob, val, bw = agent.select_action(state)
                if algo_name == 'ours': ep_brain_wave.append(bw)
                
            next_state, reward, done, info = env.step(action)
            
            if algo_name in ['ours', 'ppo', 'gcn']:
                agent.store_transition((state, action, logprob, reward, done))
                
            state = next_state
            ep_latency += info['avg_latency']
            ep_energy += info['avg_energy']
            ep_success += info['success_rate']
            ep_cost += info['avg_cost']
            
            if done: break
            
        if algo_name in ['ours', 'ppo', 'gcn']:
            agent.update()
            
        avg_latency = ep_latency / steps_per_episode
        avg_energy = ep_energy / steps_per_episode
        avg_success = ep_success / steps_per_episode
        avg_cost = ep_cost / steps_per_episode
        
        hist_latency.append(avg_latency * 1000)
        hist_energy.append(avg_energy)
        hist_success.append(avg_success * 100)
        hist_cost.append(avg_cost)
        if algo_name == 'ours':
            hist_brain_waves.append(np.mean(ep_brain_wave))
            
        if episode % 50 == 0:
            print(f"[{algo_name.upper()}] Ep {episode} | Latency: {avg_latency*1000:.1f}ms | Energy: {avg_energy:.2f}J | Cost: {avg_cost:.2f}")

    return hist_latency, hist_energy, hist_success, hist_cost, hist_brain_waves

def smooth(scalars, weight=0.9):  
    if not scalars: return scalars
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def main():
    os.makedirs('image', exist_ok=True)
    
    # 运行所有算法收集数据
    algs = ['worst_fit', 'gcn', 'ppo', 'prior', 'ours']
    results = {}
    for alg in algs:
        results[alg] = run_evaluation(alg)
        
    # ==========================
    # 绘制 1: Physical Benchmark
    # ==========================
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    metrics = [
        ("System Latency (ms)", 0, axs[0, 0]),
        ("Energy Consumption (Joules)", 1, axs[0, 1]),
        ("Task Success Rate (%)", 2, axs[1, 0]),
        ("Overall System Cost", 3, axs[1, 1])
    ]
    colors_pb = {'worst_fit': 'tab:gray', 'ppo': 'tab:orange', 'gcn': 'tab:blue', 'ours': 'tab:red'}
    labels_pb = {'worst_fit': 'Heuristic (Worst-Fit)', 'ppo': 'Standard DRL (PPO)', 'gcn': 'Topology-Aware (GCN-DRL)', 'ours': 'Dual-Brain ER-CL (Ours)'}
    
    for metric_name, idx, ax in metrics:
        for alg in ['worst_fit', 'ppo', 'gcn', 'ours']:
            smoothed_data = smooth(results[alg][idx], weight=0.8)
            if idx == 0:
                ax.set_yscale('log')
                ax.set_ylabel(metric_name + " (Log Scale)")
            else:
                ax.set_ylabel(metric_name)
            ax.plot(range(1, 501), smoothed_data, color=colors_pb[alg], linewidth=2.5, label=labels_pb[alg])
            
        ax.set_xlabel('Episodes (Testing)')
        ax.axvspan(100, 200, color='red', alpha=0.1, label='OOD: Avalanche')
        ax.axvspan(300, 400, color='orange', alpha=0.1, label='OOD: Flood')
        ax.legend(loc='best')
        ax.grid(True, linestyle='--', alpha=0.5)
        
    fig.suptitle('System-Level Physical Benchmarks under OOD Disasters', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('image/physical_benchmark.png', dpi=300)
    print("\n[OK] Physical benchmark plots saved.")
    
    # ==========================
    # 绘制 2: Ablation Curve
    # ==========================
    plt.figure(figsize=(10, 6))
    colors_ab = {'prior': 'tab:green', 'ppo': 'tab:orange', 'ours': 'tab:red'}
    labels_ab = {'prior': 'Prior Expert Only', 'ppo': 'Residual PPO (No ER-CL)', 'ours': 'Dual-Brain ER-CL (Ours)'}
    
    for alg in ['prior', 'ppo', 'ours']:
        # Ablation 画的是 Cost
        smoothed_data = smooth(results[alg][3], weight=0.85)
        plt.plot(range(1, 501), smoothed_data, color=colors_ab[alg], linewidth=2.5, label=labels_ab[alg])
        
    plt.xlabel('Episodes (Testing)')
    plt.ylabel('Overall System Cost')
    plt.title('Ablation Study: Zero-Shot Recovery vs Catastrophic Forgetting')
    plt.axvspan(100, 200, color='red', alpha=0.1, label='OOD: Avalanche')
    plt.axvspan(300, 400, color='orange', alpha=0.1, label='OOD: Flood')
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('image/ablation_curve.png', dpi=300)
    print("[OK] Ablation curve saved.")
    
    # ==========================
    # 绘制 3: Brain Waves
    # ==========================
    plt.figure(figsize=(10, 4))
    bw_data = smooth(results['ours'][4], weight=0.8)
    plt.plot(range(1, 501), bw_data, color='purple', linewidth=2.0)
    plt.xlabel('Episodes (Testing)')
    plt.ylabel('Mean Magnitude of $|\\Delta$ Logits|')
    plt.title('Brain Waves Analysis: Right Brain Global Intervention Intensity')
    plt.axvspan(100, 200, color='red', alpha=0.1, label='OOD: Avalanche')
    plt.axvspan(300, 400, color='orange', alpha=0.1, label='OOD: Flood')
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('image/brain_waves.png', dpi=300)
    print("[OK] Brain waves plot saved.")

if __name__ == "__main__":
    main()
