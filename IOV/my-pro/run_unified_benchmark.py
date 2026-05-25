import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import torch
import matplotlib.pyplot as plt

from my_iov_env import ResidualIoVEnv
from residual_ppo_agent import ResidualPPOAgent
from baseline_gcn_ppo import GCN_PPO_Agent
from baseline_heuristic import Heuristic_Agent
import torch.nn as nn
from torch.distributions import Categorical

class Legacy_ActorCritic(nn.Module):
    def __init__(self, feature_dim=7, action_dim=4):
        super(Legacy_ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(42, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward_actor(self, state):
        return self.actor(state)

class LegacyPPOAgent:
    def __init__(self, state_dim=42, action_dim=24):
        self.device = torch.device('cpu')
        self.policy = Legacy_ActorCritic(feature_dim=7, action_dim=4).to(self.device)
    def load_model(self, path):
        self.policy.load_state_dict(torch.load(path, map_location='cpu'))
    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        N = state.shape[0]
        with torch.no_grad():
            x = state.view(-1, 7)
            logits = self.policy.actor(x).view(N, 4)
            val = self.policy.critic(state.view(-1))
            dist = Categorical(logits=logits)
            action = dist.sample()
            action_logprob = dist.log_prob(action).sum()
        return action.cpu().numpy(), action_logprob.item(), val.item(), 0.0
    def store_transition(self, transition):
        pass
    def update(self):
        pass

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
    elif algo_name == 'ablation_hardclip':
        agent = ResidualPPOAgent(state_dim=flat_state_dim, action_dim=action_dim, lr=lr_online, ablation_mode='ablation_hardclip')
        agent.load_expert('model/prior_dnn_expert.pth')
        agent.load_model('model/hardclip_converged.pth')
    elif algo_name == 'ablation_nogate':
        agent = ResidualPPOAgent(state_dim=flat_state_dim, action_dim=action_dim, lr=lr_online, ablation_mode='ablation_nogate')
        agent.load_expert('model/prior_dnn_expert.pth')
        agent.load_model('model/nogate_converged.pth')
        
    steps_per_episode = 200
    
    hist_latency, hist_energy, hist_success, hist_cost = [], [], [], []
    hist_brain_waves = [] # 记录 Gate G(t) 
    hist_mse = []
    hist_entropy = []
    
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
                if algo_name in ['ours', 'ablation_nogate']: ep_brain_wave.append(bw)
                
            next_state, reward, done, info = env.step(action)
            
            if algo_name in ['ours', 'ppo', 'gcn']:
                agent.store_transition((state, action, logprob, reward, done))
                
            state = next_state
            ep_latency += info['avg_latency']
            ep_energy += info['avg_energy']
            ep_success += info['success_rate']
            ep_cost += info['avg_cost']
            
            if done: break
            
        if algo_name in ['ours', 'ppo', 'gcn', 'ablation_hardclip', 'ablation_nogate']:
            if algo_name in ['ours', 'ablation_hardclip', 'ablation_nogate']:
                ep_mse, ep_ent = agent.update()
                hist_mse.append(ep_mse)
                hist_entropy.append(ep_ent)
            else:
                agent.update()
            
        avg_latency = ep_latency / steps_per_episode
        avg_energy = ep_energy / steps_per_episode
        avg_success = ep_success / steps_per_episode
        avg_cost = ep_cost / steps_per_episode
        
        hist_latency.append(avg_latency * 1000)
        hist_energy.append(avg_energy)
        hist_success.append(avg_success * 100)
        hist_cost.append(avg_cost)
        if algo_name in ['ours', 'ablation_nogate']:
            hist_brain_waves.append(np.mean(ep_brain_wave))
            
        print(f"[{algo_name.upper()}] Ep {episode} | Latency: {avg_latency*1000:.1f}ms | Energy: {avg_energy:.2f}J | Cost: {avg_cost:.2f}")

    return hist_latency, hist_energy, hist_success, hist_cost, hist_brain_waves, hist_mse, hist_entropy

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
    algs = ['ours', 'ablation_hardclip', 'ablation_nogate'] # 由于只想重构核心残差并产生消融图，这里只需运行相关变体
    results = {}
    for alg in algs:
        results[alg] = run_evaluation(alg)
        
    # ==========================
    # 绘制 1: Physical Benchmark (简化版，不再生成完整的 pb 图，只生成消融实验所需的图表)
    # 物理性能对比已在之前验证过，本次仅输出 Ablation A 和 Ablation B
    # ==========================
    
    # ==========================
    # 绘制 2: Ablation Study A (Value Starvation & Entropy Collapse)
    # ==========================
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    
    # 子图1: Critic MSE Loss
    smoothed_mse_ours = smooth(results['ours'][5], weight=0.8)
    smoothed_mse_hard = smooth(results['ablation_hardclip'][5], weight=0.8)
    axs[0].plot(range(1, 501), smoothed_mse_ours, color='tab:red', linewidth=2.5, label='OURS (RunningMeanStd + Arcsinh)')
    axs[0].plot(range(1, 501), smoothed_mse_hard, color='tab:gray', linewidth=2.5, label='Ablation-HardClip (LayerNorm + np.clip)')
    axs[0].set_yscale('log')
    axs[0].set_xlabel('Episodes (Testing)')
    axs[0].set_ylabel('Critic MSE Loss (Log Scale)')
    axs[0].set_title('Critic Value Starvation Analysis')
    axs[0].axvspan(100, 200, color='red', alpha=0.1, label='OOD: Avalanche')
    axs[0].axvspan(300, 400, color='orange', alpha=0.1, label='OOD: Flood')
    axs[0].legend(loc='best')
    axs[0].grid(True, linestyle='--', alpha=0.7)
    
    # 子图2: Policy Entropy
    smoothed_ent_ours = smooth(results['ours'][6], weight=0.8)
    smoothed_ent_hard = smooth(results['ablation_hardclip'][6], weight=0.8)
    axs[1].plot(range(1, 501), smoothed_ent_ours, color='tab:red', linewidth=2.5, label='OURS')
    axs[1].plot(range(1, 501), smoothed_ent_hard, color='tab:gray', linewidth=2.5, label='Ablation-HardClip')
    axs[1].set_xlabel('Episodes (Testing)')
    axs[1].set_ylabel('Policy Entropy')
    axs[1].set_title('Policy Entropy Collapse Analysis')
    axs[1].axvspan(100, 200, color='red', alpha=0.1, label='OOD: Avalanche')
    axs[1].axvspan(300, 400, color='orange', alpha=0.1, label='OOD: Flood')
    axs[1].legend(loc='best')
    axs[1].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('image/ablation_A_starvation_entropy.png', dpi=300)
    print("[OK] Ablation A plot saved.")

    # ==========================
    # 绘制 3: Ablation Study B (Dynamic Gate G vs Static Gate)
    # ==========================
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    
    # 子图1: Gate G(t) 变化趋势
    bw_ours = smooth(results['ours'][4], weight=0.8)
    bw_nogate = smooth(results['ablation_nogate'][4], weight=0.8)
    axs[0].plot(range(1, 501), bw_ours, color='purple', linewidth=2.0, label='OURS (Dynamic Gate G)')
    axs[0].plot(range(1, 501), bw_nogate, color='gray', linewidth=2.0, linestyle='--', label='Ablation-NoGate (G=1.0)')
    axs[0].set_xlabel('Episodes (Testing)')
    axs[0].set_ylabel('Gate Value G(t)')
    axs[0].set_title('Dynamic Gate OOD Detection')
    axs[0].axvspan(100, 200, color='red', alpha=0.1, label='OOD: Avalanche')
    axs[0].axvspan(300, 400, color='orange', alpha=0.1, label='OOD: Flood')
    axs[0].legend(loc='best')
    axs[0].grid(True, linestyle='--', alpha=0.5)
    
    # 子图2: 和平期收敛代价 (Overall System Cost)
    cost_ours = smooth(results['ours'][3], weight=0.8)
    cost_nogate = smooth(results['ablation_nogate'][3], weight=0.8)
    axs[1].plot(range(1, 501), cost_ours, color='tab:red', linewidth=2.5, label='OURS (Dynamic Gate G)')
    axs[1].plot(range(1, 501), cost_nogate, color='tab:gray', linewidth=2.5, label='Ablation-NoGate (G=1.0)')
    axs[1].set_xlabel('Episodes (Testing)')
    axs[1].set_ylabel('Overall System Cost')
    axs[1].set_title('Performance Cost Penalty (Normal vs Disaster)')
    axs[1].axvspan(100, 200, color='red', alpha=0.1, label='OOD: Avalanche')
    axs[1].axvspan(300, 400, color='orange', alpha=0.1, label='OOD: Flood')
    axs[1].legend(loc='best')
    axs[1].grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('image/ablation_B_gate_cost.png', dpi=300)
    print("[OK] Ablation B (Brain waves / Gate) plot saved.")

if __name__ == "__main__":
    main()
