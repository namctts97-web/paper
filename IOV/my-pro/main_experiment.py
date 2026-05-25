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
    
    # [终极修复] 唤醒沉睡的左脑专家！
    agent.load_expert('model/prior_dnn_expert.pth')
    
    max_episodes = 2000
    steps_per_episode = 200 # 单回合样本量
    update_timestep = 2000 # 积攒 10 个 Episode (2000 步) 后才进行一次大 Batch 更新，彻底消除震荡
    time_step = 0
    
    print("=== Starting Residual Meta-RL Training ===", flush=True)
    print(f"Hyperparameters: LR=3e-4, Clip=0.2, Max_Episodes={max_episodes}", flush=True)
    
    # 用于画图的记录列表
    history_rewards = []
    history_cost = []
    history_latency = []
    history_energy = []
    history_urllc_lat = []
    history_embb_lat = []
    history_urllc_eng = []
    history_embb_eng = []
    history_mec_load = []
    # 新增：记录四个动作在每轮的使用比例
    history_action_ratios = {0: [], 1: [], 2: [], 3: []}
    history_raw_action_ratios = {0: [], 1: [], 2: [], 3: []}
    
    for episode in range(1, max_episodes + 1):
        # 动态衰减学习率 (Linear LR Decay)
        frac = 1.0 - (episode - 1.0) / max_episodes
        current_lr = 3e-4 * frac
        for param_group in agent.optimizer.param_groups:
            param_group['lr'] = max(current_lr, 1e-5)
            
        state = env.reset()
        episode_reward = 0
        action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        raw_action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        
        # OOD 灾难触发与恢复 (非平稳测试)
        if episode == 500:
            print(f"\n{'!'*30}\nWARNING: OOD 1 (MEC Collapse, Fiber Normal) Triggered at Ep {episode}!\n{'!'*30}")
            env.trigger_capacity_avalanche()
        elif episode == 1300:
            print(f"\n{'!'*30}\nWARNING: OOD 2 (MEC Collapse, Fiber Congested) Triggered at Ep {episode}!\n{'!'*30}")
            env.trigger_capacity_avalanche()
        elif episode == 800 or episode == 1600:
            print(f"\n{'='*30}\nINFO: Environment Recovered at Ep {episode}!\n{'='*30}")
            env.recover_from_avalanche()
            
            # [导师要求] 第一次灾难结束，立即利用 Fisher 信息矩阵固化灾难记忆突触
            if episode == 800:
                agent.update_ewc()
            
        for t in range(steps_per_episode):
            time_step += 1
            
            # 选择动作
            action, action_logprob, state_val, brain_wave = agent.select_action(state)
            
            # 与环境交互
            next_state, reward, done, info = env.step(action)
            
            # 统计这一步中真正执行的 legal_actions 分布
            for a in info['legal_actions']:
                action_counts[a] += 1
            for a in info['raw_actions']:
                raw_action_counts[a] += 1
            
            # 核心修正：如果到了最后一步，强制视为 terminal，用于正确计算回报
            is_terminal = (t == steps_per_episode - 1) or done
            
            # 存储 transition
            agent.store_transition((state, action, action_logprob, reward, is_terminal))
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
                
            # 到达指定步数，执行 PPO 平滑大 Batch 更新
            if time_step % update_timestep == 0:
                agent.update()
        
        # 打印详细日志
        total_acts = sum(action_counts.values()) + 1e-6
        p0 = action_counts[0] / total_acts * 100
        p1 = action_counts[1] / total_acts * 100
        p2 = action_counts[2] / total_acts * 100
        p3 = action_counts[3] / total_acts * 100
        log_line = f"Episode {episode} \t Cost: {info['avg_cost']:.4f} \t Latency: {info['avg_latency']:.4f}s \t Energy: {info['avg_energy']:.4f}J \t Load: {info['mec_load']:.1%} \t Act:[Loc:{p0:.0f}% MEC:{p1:.0f}% Rem:{p2:.0f}% Cld:{p3:.0f}%]"
        # 导师指令：逐条输出日志，包含各个动作比例
        print(log_line, flush=True)
        
        with open("detailed_log.txt", "a") as f:
            f.write(log_line + "\n")
            
        # 记录数据
        history_rewards.append(episode_reward/steps_per_episode)
        history_cost.append(info['avg_cost'])
        history_latency.append(info['avg_latency'])
        history_energy.append(info['avg_energy'])
        history_urllc_lat.append(info['urllc_latency'])
        history_embb_lat.append(info['embb_latency'])
        history_urllc_eng.append(info['urllc_energy'])
        history_embb_eng.append(info['embb_energy'])
        history_mec_load.append(info['mec_load'])
        
        # 计算该轮各动作的比例
        total_actions_in_ep = sum(action_counts.values())
        for a in range(4):
            history_action_ratios[a].append(action_counts[a] / total_actions_in_ep)
            history_raw_action_ratios[a].append(raw_action_counts[a] / total_actions_in_ep)

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
    ax1.set_ylabel('Total System Cost', color=color)
    
    # 画底色真实毛刺 (透明度 0.2)
    ax1.plot(range(1, max_episodes + 1), history_cost, color=color, alpha=0.2)
    # 画 EMA 平滑主线
    ax1.plot(range(1, max_episodes + 1), smooth(history_cost, weight=0.95), color=color, linewidth=2, label='Total Cost (Smoothed)')
    
    ax1.tick_params(axis='y', labelcolor=color)
    
    # 灾难区阴影
    ax1.axvspan(500, 800, color='red', alpha=0.1, label='OOD Disaster')
    ax1.axvspan(1300, 1600, color='red', alpha=0.1)
    
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
    
    # 灾难区阴影
    ax_act.axvspan(500, 800, color='red', alpha=0.1, label='OOD Disaster')
    ax_act.axvspan(1300, 1600, color='red', alpha=0.1)
    ax_act.legend(loc='lower left')
    
    fig2.tight_layout()
    save_path2 = os.path.abspath('image/action_distribution.png')
    plt.savefig(save_path2, dpi=300)
    print(f"[OK] Action plot saved to {save_path2}")
    
    # === 画图 3: Raw 动作分布图 (证明智能体真正意图) ===
    fig3, ax_raw = plt.subplots(figsize=(10, 6))
    r0 = smooth(history_raw_action_ratios[0], 0.95)
    r1 = smooth(history_raw_action_ratios[1], 0.95)
    r2 = smooth(history_raw_action_ratios[2], 0.95)
    r3 = smooth(history_raw_action_ratios[3], 0.95)
    
    ax_raw.stackplot(range(1, max_episodes + 1), r0, r1, r2, r3, 
                     labels=['Local (Action 0)', 'Local MEC (Action 1)', 'Remote MEC (Action 2)', 'Cloud (Action 3)'],
                     colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], alpha=0.8)
    ax_raw.set_xlabel('Episodes')
    ax_raw.set_ylabel('Raw Action Probability (Smoothed)')
    ax_raw.set_title('Neural Network RAW Intent (Proof of Intelligence)')
    # 灾难区阴影
    ax_raw.axvspan(500, 800, color='red', alpha=0.1, label='OOD Disaster')
    ax_raw.axvspan(1300, 1600, color='red', alpha=0.1)
    ax_raw.legend(loc='lower left')
    fig3.tight_layout()
    save_path3 = os.path.abspath('image/raw_action_distribution.png')
    plt.savefig(save_path3, dpi=300)
    print(f"[OK] Raw Action plot saved to {save_path3}")
    
    # === 画图 4: 物理指标解耦图 (URLLC vs eMBB) ===
    fig4, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    
    ax_u_lat = axes[0, 0]
    ax_e_lat = axes[0, 1]
    ax_u_eng = axes[1, 0]
    ax_e_eng = axes[1, 1]
    
    # URLLC Latency (使用它自己专属的低数量级 Y 轴)
    ax_u_lat.plot(range(1, max_episodes + 1), history_urllc_lat, color='tab:red', alpha=0.2)
    ax_u_lat.plot(range(1, max_episodes + 1), smooth(history_urllc_lat, 0.95), color='tab:red', linewidth=2)
    ax_u_lat.set_title('URLLC Emergency Latency (s)')
    ax_u_lat.set_ylabel('Seconds')
    ax_u_lat.axvspan(500, 800, color='red', alpha=0.1)
    ax_u_lat.axvspan(1300, 1600, color='red', alpha=0.1)
    
    # eMBB Latency (使用它专属的高数量级 Y 轴，高达几十秒)
    ax_e_lat.plot(range(1, max_episodes + 1), history_embb_lat, color='tab:blue', alpha=0.2)
    ax_e_lat.plot(range(1, max_episodes + 1), smooth(history_embb_lat, 0.95), color='tab:blue', linewidth=2)
    ax_e_lat.set_title('eMBB Background Latency (s)')
    ax_e_lat.axvspan(500, 800, color='red', alpha=0.1)
    ax_e_lat.axvspan(1300, 1600, color='red', alpha=0.1)
    
    # URLLC Energy
    ax_u_eng.plot(range(1, max_episodes + 1), history_urllc_eng, color='tab:red', alpha=0.2)
    ax_u_eng.plot(range(1, max_episodes + 1), smooth(history_urllc_eng, 0.95), color='tab:red', linewidth=2)
    ax_u_eng.set_title('URLLC Emergency Energy (J)')
    ax_u_eng.set_ylabel('Joules')
    ax_u_eng.set_xlabel('Episodes')
    ax_u_eng.axvspan(500, 800, color='red', alpha=0.1)
    ax_u_eng.axvspan(1300, 1600, color='red', alpha=0.1)
    
    # eMBB Energy
    ax_e_eng.plot(range(1, max_episodes + 1), history_embb_eng, color='tab:blue', alpha=0.2)
    ax_e_eng.plot(range(1, max_episodes + 1), smooth(history_embb_eng, 0.95), color='tab:blue', linewidth=2)
    ax_e_eng.set_title('eMBB Background Energy (J)')
    ax_e_eng.set_xlabel('Episodes')
    ax_e_eng.axvspan(500, 800, color='red', alpha=0.1)
    ax_e_eng.axvspan(1300, 1600, color='red', alpha=0.1)
    
    fig4.suptitle("Disentangled Physical Metrics: URLLC Protection vs eMBB Sacrifice", fontsize=16)
    fig4.tight_layout()
    save_path4 = os.path.abspath('image/latency_energy_curve.png')
    plt.savefig(save_path4, dpi=300)
    print(f"[OK] Physical Metrics plot saved to {save_path4}")
    
    # 为了让聊天机器人能内嵌展示，把图片复制到 artifact 目录
    import shutil
    artifact_dir = r"C:\Users\zrd\.gemini\antigravity\brain\16d936b8-eb04-4bd2-bfb0-d092f15d7b64"
    shutil.copy(save_path, os.path.join(artifact_dir, "training_curve.png"))
    shutil.copy(save_path2, os.path.join(artifact_dir, "action_distribution.png"))
    shutil.copy(save_path3, os.path.join(artifact_dir, "raw_action_distribution.png"))
    shutil.copy(save_path4, os.path.join(artifact_dir, "latency_energy_curve.png"))

if __name__ == "__main__":
    train()
