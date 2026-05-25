import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import torch
import time

from my_iov_env import ResidualIoVEnv
from residual_ppo_agent import ResidualPPOAgent
from baseline_gcn_ppo import GCN_PPO_Agent

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)

def pretrain_algorithm(algo_name, agent, env, max_episodes=2000, save_path=""):
    print(f"\n[{algo_name.upper()}] Starting Pre-training for {max_episodes} episodes...")
    
    steps_per_episode = 200
    for episode in range(1, max_episodes + 1):
        state = env.reset()
        ep_cost = 0
        
        for t in range(steps_per_episode):
            if algo_name == 'gcn':
                action, logprob, val = agent.select_action(state)
            else:
                action, logprob, val, _ = agent.select_action(state)
                
            next_state, reward, done, info = env.step(action)
            agent.store_transition((state, action, logprob, reward, done))
            state = next_state
            ep_cost += info['avg_cost']
            
            if done: break
            
        agent.update()
        
        avg_cost = ep_cost / steps_per_episode
        if episode % 100 == 0:
            print(f"[{algo_name.upper()}] Ep {episode} | Avg Cost: {avg_cost:.2f}")
            
    if save_path:
        agent.save_model(save_path)
        print(f"[{algo_name.upper()}] Model converged and saved to {save_path}")

def main():
    os.makedirs('model', exist_ok=True)
    set_seed(42)
    
    env = ResidualIoVEnv()
    sample_state = env.reset()
    flat_state_dim = sample_state.flatten().shape[0]
    action_dim = len(env.vehicles) * 4
    
    # 1. Train PPO Baseline (Normal PPO, no expert, no CL)
    # ppo_agent = ResidualPPOAgent(state_dim=flat_state_dim, action_dim=action_dim, lr=3e-4, ablation_mode='ppo')
    # pretrain_algorithm('ppo', ppo_agent, env, max_episodes=2000, save_path='model/ppo_converged.pth')
    
    # 2. Train GCN Baseline
    # gcn_agent = GCN_PPO_Agent(lr=3e-4)
    # pretrain_algorithm('gcn', gcn_agent, env, max_episodes=2000, save_path='model/gcn_converged.pth')
    
    # 3. Train Ours (Dual-Brain)
    ours_agent = ResidualPPOAgent(state_dim=flat_state_dim, action_dim=action_dim, lr=3e-4, ablation_mode='ours')
    ours_agent.load_expert('model/prior_dnn_expert.pth')
    pretrain_algorithm('ours', ours_agent, env, max_episodes=2000, save_path='model/ours_converged.pth')
    
    # 4. Train Ablation: HardClip (No Return-Norm, No Arcsinh)
    env_hardclip = ResidualIoVEnv(ablation_mode='ablation_hardclip')
    hardclip_agent = ResidualPPOAgent(state_dim=flat_state_dim, action_dim=action_dim, lr=3e-4, ablation_mode='ablation_hardclip')
    hardclip_agent.load_expert('model/prior_dnn_expert.pth')
    pretrain_algorithm('ablation_hardclip', hardclip_agent, env_hardclip, max_episodes=2000, save_path='model/hardclip_converged.pth')
    
    # 5. Train Ablation: NoGate (Gate=1.0 constantly)
    env_nogate = ResidualIoVEnv(ablation_mode='ablation_nogate')
    nogate_agent = ResidualPPOAgent(state_dim=flat_state_dim, action_dim=action_dim, lr=3e-4, ablation_mode='ablation_nogate')
    nogate_agent.load_expert('model/prior_dnn_expert.pth')
    pretrain_algorithm('ablation_nogate', nogate_agent, env_nogate, max_episodes=2000, save_path='model/nogate_converged.pth')

if __name__ == "__main__":
    main()
