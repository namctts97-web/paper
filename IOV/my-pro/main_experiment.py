from my_iov_env import ResidualIoVEnv
from residual_ppo_agent import ResidualPPOAgent
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

def train():
    # 1. 初始化环境与智能体
    env = ResidualIoVEnv()
    N = len(env.vehicles)
    state_dim = env.observation_space.shape[0] # 自适应最新遥测状态维度 (26维)
    action_dim = N * 4      # V4 动作空间: 4 个 Logits/车辆
    
    agent = ResidualPPOAgent(
        state_dim=state_dim, 
        action_dim=action_dim,
        lr=3e-4, 
        gamma=0.99,
        K_epochs=4,
        eps_clip=0.2
    )
    
    max_episodes = 2000
    steps_per_episode = 200 # 巨幅提升单回合样本量，让 Critic 眼界大开，Advantage 极度精准
    
    print("=== Starting Residual Meta-RL Training ===", flush=True)
    print(f"Hyperparameters: LR=3e-4, Clip=0.2, Max_Episodes={max_episodes}", flush=True)
    
    # 用于画图的记录列表
    history_rewards = []
    history_mec_load = []
    # 新增：记录四个动作在每轮的使用比例
    history_action_ratios = {0: [], 1: [], 2: [], 3: []}
    
    for episode in range(1, max_episodes + 1):
        # 动态衰减学习率 (Linear LR Decay)
        frac = 1.0 - (episode - 1.0) / max_episodes
        current_lr = 3e-4 * frac
        for param_group in agent.optimizer.param_groups:
            param_group['lr'] = max(current_lr, 1e-5)
            
        state = env.reset()
        episode_reward = 0
        action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        
        # OOD 灾难触发 (在训练一半时触发)
        if episode == 1000:
            print("\n" + "!"*30)
            print("WARNING: OOD Disaster Triggered!")
            env.trigger_capacity_avalanche()
            env.trigger_traffic_flood()
            print("!"*30 + "\n")
            
        for t in range(steps_per_episode):
            # 选择动作
            action, action_logprob, state_val = agent.select_action(state)
            
            # 与环境交互
            next_state, reward, done, info = env.step(action)
            
            # 统计这一步中真正执行的 legal_actions 分布
            for a in info['legal_actions']:
                action_counts[a] += 1
            
            # 核心修正：如果到了最后一步，强制视为 terminal，用于正确计算回报
            is_terminal = (t == steps_per_episode - 1) or done
            
            # 存储 transition
            agent.store_transition((state, action, action_logprob, reward, is_terminal))
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        # ！！！核心修改：每个 Episode 结束后，利用一整条完整轨迹进行 PPO 更新！！！
        agent.update()
        
        # 打印日志
        if episode % 100 == 0 or (990 <= episode <= 1010):
            print(f"Episode {episode} \t Avg Reward: {episode_reward/steps_per_episode:.4f} \t MEC Load: {info['mec_load']:.2%}", flush=True)
            
        # 记录数据
        history_rewards.append(episode_reward/steps_per_episode)
        history_mec_load.append(info['mec_load'])
        
        # 计算该轮各动作的比例
        total_actions_in_ep = sum(action_counts.values())
        for a in range(4):
            history_action_ratios[a].append(action_counts[a] / total_actions_in_ep)

    # === 训练结束，开始画图 ===
    os.makedirs('image', exist_ok=True)
    
    def smooth(scalars, weight=0.9):  
        if not scalars: return scalars
        last = scalars[0]
        smoothed = []
        for point in scalars:
            smoothed_val = last * weight + (1 - weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed

    # 创建画布
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Average Reward (Cost)', color=color)
    
    # 画底色真实毛刺 (透明度 0.2)
    ax1.plot(range(1, max_episodes + 1), history_rewards, color=color, alpha=0.2)
    # 画 EMA 平滑主线
    ax1.plot(range(1, max_episodes + 1), smooth(history_rewards, weight=0.95), color=color, linewidth=2, label='Avg Reward (Smoothed)')
    
    ax1.tick_params(axis='y', labelcolor=color)
    
    # 灾难线
    ax1.axvline(x=1000, color='r', linestyle='--', label='OOD Disaster (Ep 1000)')
    
    # 实例化一个共享 x 轴的第二个 y 轴用于 MEC Load
    ax2 = ax1.twinx()  
    color = 'tab:orange'
    ax2.set_ylabel('MEC Load Ratio', color=color)  
    
    # 画底色真实负载毛刺 (透明度 0.2)
    ax2.plot(range(1, max_episodes + 1), history_mec_load, color=color, alpha=0.2)
    # 画 EMA 平滑负载线
    ax2.plot(range(1, max_episodes + 1), smooth(history_mec_load, weight=0.95), color=color, linewidth=2, label='MEC Load (Smoothed)')
    
    ax2.tick_params(axis='y', labelcolor=color)
    
    # 添加图例和标题
    fig.tight_layout() 
    plt.title("Residual PPO Meta-RL Performance (OOD Disaster Test)")
    
    # 合并图例
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='lower right')
    
    save_path = os.path.abspath('image/training_curve.png')
    plt.savefig(save_path, dpi=300)
    print(f"\n[OK] Plot saved to {save_path}")
    
    # === 画图 2: 动作分布图 ===
    fig2, ax_act = plt.subplots(figsize=(10, 6))
    
    y0 = smooth(history_action_ratios[0], 0.95)
    y1 = smooth(history_action_ratios[1], 0.95)
    y2 = smooth(history_action_ratios[2], 0.95)
    y3 = smooth(history_action_ratios[3], 0.95)
    
    ax_act.stackplot(range(1, max_episodes + 1), y0, y1, y2, y3, 
                     labels=['Local (Action 0)', 'Local MEC (Action 1)', 'Remote MEC (Action 2)', 'Cloud (Action 3)'],
                     colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], alpha=0.8)
                     
    ax_act.set_xlabel('Episodes')
    ax_act.set_ylabel('Action Probability (Smoothed)')
    ax_act.set_title('Action Distribution Over Training (OOD Disaster Test)')
    
    # 灾难线
    ax_act.axvline(x=1000, color='black', linestyle='--', linewidth=2, label='OOD Disaster (Ep 1000)')
    ax_act.legend(loc='lower left')
    
    fig2.tight_layout()
    save_path2 = os.path.abspath('image/action_distribution.png')
    plt.savefig(save_path2, dpi=300)
    print(f"[OK] Action plot saved to {save_path2}")
    
    # 为了让聊天机器人能内嵌展示，把图片复制到 artifact 目录
    import shutil
    artifact_dir = r"C:\Users\zrd\.gemini\antigravity\brain\798a053b-8f49-4bd5-b613-d2857bc4dfa6"
    shutil.copy(save_path, os.path.join(artifact_dir, "training_curve.png"))
    shutil.copy(save_path2, os.path.join(artifact_dir, "action_distribution.png"))

if __name__ == "__main__":
    train()
