import torch
import numpy as np
import os
import time
from iov_env_v3 import IoVEnvV3
from residual_ppo_agent_v3 import ResidualPPOAgent

def train_residual():
    print("==================================================")
    print("=== 残差 PPO 在线联合训练 (Residual Joint Training) ===")
    print("==================================================")
    
    env = IoVEnvV3(num_vehicles=10)
    state_dim = env.state_dim
    action_dim = env.num_vehicles * 4
    
    agent = ResidualPPOAgent(state_dim=state_dim, action_dim=action_dim, lr=3e-4, gamma=0.99, K_epochs=4)
    
    # 强制进行 EWC 和门控逻辑的初始化
    prior_path = 'model/prior_dnn_expert.pth'
    if os.path.exists(prior_path):
        agent.load_expert(prior_path)
    else:
        print(f"Error: {prior_path} not found. 必须先完成先验网络的预训练！")
        return
        
    print("\n=== Autoencoder 门控预热 (AE Warmup) ===")
    # 收集纯净的正常状态数据，频繁 reset 确保 S_macro (如 mec_load) 产生合理的方差
    normal_states = []
    env.reset_disasters()
    obs = env.reset()
    for i in range(500):
        normal_states.append(obs)
        if i % 10 == 0 and i > 0:
            obs = env.reset()
        else:
            obs, _, _, _ = env.step(np.zeros(10))
        
    normal_states_ts = torch.FloatTensor(np.array(normal_states)).to(agent.device)
    # [学术修正] 显式更新并冻结 RunningMeanStd，确立正常环境的正态分布锚点
    agent.residual_net.autoencoder.norm.update(normal_states_ts[..., :4])
    
    optimizer_ae = torch.optim.Adam(agent.residual_net.autoencoder.parameters(), lr=1e-2)
    for _ in range(200):
        reconstructed, norm_state = agent.residual_net.autoencoder(normal_states_ts)
        loss = torch.mean((norm_state - reconstructed)**2)
        optimizer_ae.zero_grad()
        loss.backward()
        optimizer_ae.step()
    print(f"AE 预热完成！正常态的基准重构误差已被成功压制到: {loss.item():.6f}")

    num_episodes = 20 # 演示用，进行20个高强度灾难测试 Episode
    steps_per_ep = 200
    update_timestep = 400 # 2个 Episode 更新一次
    time_step = 0
    
    os.makedirs('model', exist_ok=True)
    
    print("\n开始动态灾难注入与门控 $g$ 监控训练...")
    
    for ep in range(1, num_episodes + 1):
        env.reset_disasters()
        state = env.reset()
        
        ep_cost = 0
        gate_history_normal = []
        gate_history_ood = []
        
        for step in range(1, steps_per_ep + 1):
            time_step += 1
            
            # 步骤 3 关键点：第 101 步突发 OOD 灾难
            if step == 101:
                if np.random.rand() > 0.5:
                    env.trigger_flood()
                else:
                    env.trigger_avalanche()
                # 必须调用 reset 才能使新的 surge_mult 和 f_mec 生效于全部车辆参数
                state = env.reset()
            
            # 代理选择动作（此时门控 g 会自动被激活并返回）
            action, action_logprob, value, gate_val = agent.select_action(state)
            
            # 物理引擎步进
            next_state, reward, done, info = env.step(action)
            cost = info['avg_cost']
            
            # 记录经验
            agent.store_transition((state, action, action_logprob, reward, done))
            
            state = next_state
            ep_cost += cost
            
            # 收集并分离门控数据
            if step <= 100:
                gate_history_normal.append(gate_val)
            else:
                gate_history_ood.append(gate_val)
            
            # RL 代理 PPO 更新 & AE/EWC 联合训练
            if time_step % update_timestep == 0:
                loss_mse, loss_entropy = agent.update()
                agent.update_ewc()
                
        # 统计本回合数据的宏观表现
        avg_normal_gate = np.mean(gate_history_normal)
        avg_ood_gate = np.mean(gate_history_ood)
        
        print(f"Ep {ep:02d} | Avg Normal Gate: {avg_normal_gate:.4f} | Avg OOD Gate: {avg_ood_gate:.4f} | Total Cost: {ep_cost:.2f}")
        
    print("\n联合训练大功告成！")
    save_path = 'model/residual_ppo_ours.pth'
    agent.save_model(save_path)
    print(f"网络权重成功固化并保存在: {save_path}")

if __name__ == '__main__':
    train_residual()
