from my_iov_env import ResidualIoVEnv
from residual_ppo_agent_ablation import ResidualPPOAgent
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import shutil

def run_experiment(mode='residual', max_episodes=2000):
    print(f"\n==============================================")
    print(f"=== Starting Training Mode: {mode.upper()} ===")
    print(f"==============================================")
    
    env = ResidualIoVEnv()
    N = len(env.vehicles)
    state_dim = env.observation_space.shape[0]
    action_dim = N * 4
    
    agent = ResidualPPOAgent(
        state_dim=state_dim, 
        action_dim=action_dim,
        lr=3e-4, 
        gamma=0.99,
        K_epochs=4,
        eps_clip=0.2,
        ablation_mode=mode
    )
    
    steps_per_episode = 200
    
    history_rewards = []
    # brain_waves: episode -> [action0_val, action1_val, action2_val, action3_val]
    history_brain_waves = [] 
    
    for episode in range(1, max_episodes + 1):
        frac = 1.0 - (episode - 1.0) / max_episodes
        current_lr = 3e-4 * frac
        for param_group in agent.optimizer.param_groups:
            param_group['lr'] = max(current_lr, 1e-5)
            
        state = env.reset()
        episode_reward = 0
        episode_brain_wave = np.zeros(4)
        
        if episode == 1000:
            env.trigger_capacity_avalanche()
            env.trigger_traffic_flood()
            
        for t in range(steps_per_episode):
            # 获取脑电波
            action, action_logprob, state_val, brain_wave = agent.select_action(state)
            
            next_state, reward, done, info = env.step(action)
            
            is_terminal = (t == steps_per_episode - 1) or done
            agent.store_transition((state, action, action_logprob, reward, is_terminal))
            
            state = next_state
            episode_reward += reward
            episode_brain_wave += brain_wave
            
            if done:
                break
        
        agent.update()
        
        avg_reward = episode_reward / steps_per_episode
        avg_brain_wave = episode_brain_wave / steps_per_episode
        
        if episode % 100 == 0 or (990 <= episode <= 1010):
            print(f"[{mode.upper()}] Ep {episode} \t Avg Cost(Reward): {avg_reward:.4f}", flush=True)
            
        history_rewards.append(avg_reward)
        history_brain_waves.append(avg_brain_wave)

    return history_rewards, history_brain_waves

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
    max_episodes = 2000
    
    # 1. 跑三组实验
    results_rewards = {}
    results_brain_waves = {}
    modes = ['prior', 'ppo', 'residual']
    
    for mode in modes:
        rewards, brain_waves = run_experiment(mode=mode, max_episodes=max_episodes)
        results_rewards[mode] = rewards
        results_brain_waves[mode] = brain_waves

    # 2. 画图一：消融曲线 (Ablation Curve)
    fig1, ax1 = plt.subplots(figsize=(12, 7))
    colors = {'prior': 'tab:gray', 'ppo': 'tab:orange', 'residual': 'tab:red'}
    labels = {'prior': 'Prior-only (Heuristic Expert)', 'ppo': 'PPO-only (Tabula Rasa)', 'residual': 'Residual PPO (Dual-Brain)'}
    
    for mode in modes:
        raw_rewards = results_rewards[mode]
        smoothed_rewards = smooth(raw_rewards, weight=0.95)
        # 画底色真实毛刺 (透明度 0.1)
        ax1.plot(range(1, max_episodes + 1), raw_rewards, color=colors[mode], alpha=0.1)
        # 画平滑主线
        ax1.plot(range(1, max_episodes + 1), smoothed_rewards, color=colors[mode], linewidth=2.5, label=labels[mode])
        
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Average Cost (Negative Reward)')
    ax1.set_title('Ablation Study: Dual-Brain vs Single-Brain under OOD Disaster')
    ax1.axvline(x=1000, color='black', linestyle='--', linewidth=2, label='OOD Disaster Triggered (Ep 1000)')
    ax1.legend(loc='lower right')
    fig1.tight_layout()
    
    path_ablation = os.path.abspath('image/ablation_curve.png')
    plt.savefig(path_ablation, dpi=300)
    print(f"\n[OK] Ablation curve saved to {path_ablation}")
    
    # 3. 画图二：脑电波爆发图 (Brain Waves) 专门针对 Residual 模式的 Action 3
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    
    # 取出 residual 模式的数据，提取 Action 3 的 delta_logits
    res_brain_waves = np.array(results_brain_waves['residual']) # shape: (2000, 4)
    act3_waves = res_brain_waves[:, 3]
    smoothed_act3 = smooth(act3_waves.tolist(), weight=0.8)
    
    ax2.plot(range(1, max_episodes + 1), act3_waves, color='purple', alpha=0.2)
    ax2.plot(range(1, max_episodes + 1), smoothed_act3, color='purple', linewidth=2.5, label='Right Brain Correction Amplitude (Action 3: Cloud)')
    
    ax2.set_xlabel('Episodes')
    ax2.set_ylabel('Magnitude of $\Delta$ Logits (Absolute Value)')
    ax2.set_title('Brain Waves Analysis: Right Brain Intervention under Crisis')
    ax2.axvline(x=1000, color='black', linestyle='--', linewidth=2, label='OOD Disaster Triggered (Ep 1000)')
    
    # 添加一个标注指出突刺
    ax2.annotate('Sudden Spike (Forced Correction)', xy=(1000, max(smoothed_act3)), xytext=(1100, max(smoothed_act3)*0.8),
                 arrowprops=dict(facecolor='black', shrink=0.05))
                 
    ax2.legend(loc='upper right')
    fig2.tight_layout()
    
    path_waves = os.path.abspath('image/brain_waves.png')
    plt.savefig(path_waves, dpi=300)
    print(f"[OK] Brain waves plot saved to {path_waves}")
    
    # 复制到 artifact
    artifact_dir = r"C:\Users\zrd\.gemini\antigravity\brain\798a053b-8f49-4bd5-b613-d2857bc4dfa6"
    shutil.copy(path_ablation, os.path.join(artifact_dir, "ablation_curve.png"))
    shutil.copy(path_waves, os.path.join(artifact_dir, "brain_waves.png"))

if __name__ == "__main__":
    main()
