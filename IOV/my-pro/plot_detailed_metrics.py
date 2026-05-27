import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import torch
import matplotlib.pyplot as plt
import random

from my_iov_env import ResidualIoVEnv
from residual_ppo_agent import ResidualPPOAgent

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

def smooth(scalars, weight=0.9):  
    if len(scalars) == 0: return scalars
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)

def run_detailed_evaluation():
    os.makedirs('image', exist_ok=True)
    set_seed(42)
    
    print("=== Starting V2.0 Detailed Physics Evaluation ===", flush=True)
    
    env = ResidualIoVEnv()
    sample_state = env.reset()
    flat_state_dim = sample_state.flatten().shape[0]
    action_dim = len(env.vehicles) * 4
    
    # 纯评估模式，学习率极小
    agent = ResidualPPOAgent(state_dim=flat_state_dim, action_dim=action_dim, lr=1e-5, ablation_mode='ours')
    
    agent.load_expert('model/prior_dnn_expert.pth')
    agent.load_model('model/ours_converged.pth')
    
    max_episodes = 2000
    steps_per_episode = 200
    
    history_cost = []
    history_mec_load = []
    history_urllc_lat = []
    history_embb_lat = []
    history_urllc_eng = []
    history_embb_eng = []
    
    history_action_ratios = {0: [], 1: [], 2: [], 3: []}
    history_raw_action_ratios = {0: [], 1: [], 2: [], 3: []}
    
    for episode in range(1, max_episodes + 1):
        if episode == 500: env.trigger_capacity_avalanche()
        elif episode == 800: env.recover_from_avalanche()
        elif episode == 1300: env.trigger_traffic_flood()
        elif episode == 1600: env.recover_from_flood()
            
        state = env.reset()
        
        ep_cost, ep_urllc_lat, ep_embb_lat, ep_urllc_eng, ep_embb_eng, ep_mec_load = 0, 0, 0, 0, 0, 0
        action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        raw_action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        
        for t in range(steps_per_episode):
            res = agent.select_action(state)
            action = res[0]
            
            next_state, reward, done, info = env.step(action)
            
            for a in info['legal_actions']: action_counts[a] += 1
            for a in info['raw_actions']: raw_action_counts[a] += 1
                
            ep_cost += info['avg_cost']
            ep_urllc_lat += info['urllc_latency']
            ep_embb_lat += info['embb_latency']
            ep_urllc_eng += info['urllc_energy']
            ep_embb_eng += info['embb_energy']
            ep_mec_load += info['mec_load']
            
            state = next_state
            if done: break
            
        # 不更新 agent，这是纯推断评估
        
        history_cost.append(ep_cost / steps_per_episode)
        history_urllc_lat.append((ep_urllc_lat / steps_per_episode) * 1000) # ms
        history_embb_lat.append((ep_embb_lat / steps_per_episode) * 1000) # ms
        history_urllc_eng.append(ep_urllc_eng / steps_per_episode)
        history_embb_eng.append(ep_embb_eng / steps_per_episode)
        history_mec_load.append(ep_mec_load / steps_per_episode)
        
        total_actions = sum(action_counts.values()) + 1e-6
        for a in range(4):
            history_action_ratios[a].append(action_counts[a] / total_actions)
            history_raw_action_ratios[a].append(raw_action_counts[a] / total_actions)
            
        if episode % 100 == 0:
            print(f"Ep {episode} | Cost: {history_cost[-1]:.2f} | MEC Load: {history_mec_load[-1]:.1%}")

    print("\n[EVAL] Evaluation Finished. Plotting detailed metrics...")

    x = range(1, max_episodes + 1)
    
    # === 画图 1: 系统总成本与基站负载 ===
    fig, ax1 = plt.subplots(figsize=(10, 6))
    color = 'tab:blue'
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Total System Cost', color=color)
    ax1.plot(x, history_cost, color=color, alpha=0.2)
    ax1.plot(x, smooth(history_cost, 0.95), color=color, linewidth=2, label='Total Cost (Smoothed)')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax1.axvspan(500, 800, color='red', alpha=0.1, label='Avalanche (MEC Drop)')
    ax1.axvspan(1300, 1600, color='blue', alpha=0.1, label='Flood (Traffic Spike)')
    
    ax2 = ax1.twinx()  
    color = 'tab:orange'
    ax2.set_ylabel('MEC Load Ratio', color=color)  
    ax2.plot(x, history_mec_load, color=color, alpha=0.2)
    ax2.plot(x, smooth(history_mec_load, 0.95), color=color, linewidth=2, label='MEC Load (Smoothed)')
    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout() 
    plt.title("V2.0 Inference: Total Cost & MEC Load Response under Disasters")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')
    plt.savefig('image/training_curve.png', dpi=300)

    # === 画图 2: Actual Action Distribution (物理 GKP 投影后) ===
    fig2, ax_act = plt.subplots(figsize=(10, 6))
    y0 = smooth(history_action_ratios[0], 0.95)
    y1 = smooth(history_action_ratios[1], 0.95)
    y2 = smooth(history_action_ratios[2], 0.95)
    y3 = smooth(history_action_ratios[3], 0.95)
    
    ax_act.stackplot(x, y0, y1, y2, y3, 
                     labels=['Local (Action 0)', 'Local MEC (Action 1)', 'Remote MEC (Action 2)', 'Cloud (Action 3)'],
                     colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], alpha=0.8)
    ax_act.set_xlabel('Episodes')
    ax_act.set_ylabel('Execution Probability')
    ax_act.set_title('Post-GKP Physical Execution Profile (The Cold Reality)')
    ax_act.axvspan(500, 800, color='red', alpha=0.1)
    ax_act.axvspan(1300, 1600, color='blue', alpha=0.1)
    ax_act.legend(loc='lower right')
    fig2.tight_layout()
    plt.savefig('image/action_distribution.png', dpi=300)

    # === 画图 3: Raw Action Distribution (神经网络原始意图) ===
    fig3, ax_raw = plt.subplots(figsize=(10, 6))
    r0 = smooth(history_raw_action_ratios[0], 0.95)
    r1 = smooth(history_raw_action_ratios[1], 0.95)
    r2 = smooth(history_raw_action_ratios[2], 0.95)
    r3 = smooth(history_raw_action_ratios[3], 0.95)
    
    ax_raw.stackplot(x, r0, r1, r2, r3, 
                     labels=['Local (Action 0)', 'Local MEC (Action 1)', 'Remote MEC (Action 2)', 'Cloud (Action 3)'],
                     colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], alpha=0.8)
    ax_raw.set_xlabel('Episodes')
    ax_raw.set_ylabel('Network Intent Probability')
    ax_raw.set_title('Pre-GKP Neural Network Intent (Bounded Residual Temperature)')
    ax_raw.axvspan(500, 800, color='red', alpha=0.1)
    ax_raw.axvspan(1300, 1600, color='blue', alpha=0.1)
    ax_raw.legend(loc='lower right')
    fig3.tight_layout()
    plt.savefig('image/raw_action_distribution.png', dpi=300)

    # === 画图 4: 物理指标解耦图 (URLLC vs eMBB) ===
    fig4, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    ax_u_lat, ax_e_lat = axes[0, 0], axes[0, 1]
    ax_u_eng, ax_e_eng = axes[1, 0], axes[1, 1]
    
    # URLLC Latency (Effective Capacity)
    ax_u_lat.plot(x, history_urllc_lat, color='tab:red', alpha=0.2)
    ax_u_lat.plot(x, smooth(history_urllc_lat, 0.95), color='tab:red', linewidth=2)
    ax_u_lat.set_title('URLLC Effective Latency (ms) [LogSumExp Penalized]')
    ax_u_lat.set_ylabel('Milliseconds')
    ax_u_lat.axvspan(500, 800, color='red', alpha=0.1)
    ax_u_lat.axvspan(1300, 1600, color='blue', alpha=0.1)
    
    # eMBB Latency (Ergodic Capacity)
    ax_e_lat.plot(x, history_embb_lat, color='tab:blue', alpha=0.2)
    ax_e_lat.plot(x, smooth(history_embb_lat, 0.95), color='tab:blue', linewidth=2)
    ax_e_lat.set_title('eMBB Ergodic Latency (ms) [Mean Tolerated]')
    ax_e_lat.axvspan(500, 800, color='red', alpha=0.1)
    ax_e_lat.axvspan(1300, 1600, color='blue', alpha=0.1)
    
    # URLLC Energy
    ax_u_eng.plot(x, history_urllc_eng, color='tab:red', alpha=0.2)
    ax_u_eng.plot(x, smooth(history_urllc_eng, 0.95), color='tab:red', linewidth=2)
    ax_u_eng.set_title('URLLC Energy Overhead (J)')
    ax_u_eng.set_ylabel('Joules')
    ax_u_eng.set_xlabel('Episodes')
    ax_u_eng.axvspan(500, 800, color='red', alpha=0.1)
    ax_u_eng.axvspan(1300, 1600, color='blue', alpha=0.1)
    
    # eMBB Energy
    ax_e_eng.plot(x, history_embb_eng, color='tab:blue', alpha=0.2)
    ax_e_eng.plot(x, smooth(history_embb_eng, 0.95), color='tab:blue', linewidth=2)
    ax_e_eng.set_title('eMBB Energy Overhead (J)')
    ax_e_eng.set_xlabel('Episodes')
    ax_e_eng.axvspan(500, 800, color='red', alpha=0.1)
    ax_e_eng.axvspan(1300, 1600, color='blue', alpha=0.1)
    
    fig4.suptitle("V2.0 Physical Decoupling: URLLC Protection vs eMBB Sacrifice", fontsize=16)
    fig4.tight_layout()
    plt.savefig('image/latency_energy_curve.png', dpi=300)
    print("\n[OK] All internal mechanism plots successfully generated to 'image/' folder!")

if __name__ == "__main__":
    run_detailed_evaluation()
