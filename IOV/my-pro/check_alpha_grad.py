import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from my_iov_env import ResidualIoVEnv
from residual_ppo_agent import ResidualPPOAgent

def run_alpha_grad_test():
    env = ResidualIoVEnv()
    sample_state = env.reset()
    state_dim = sample_state.flatten().shape[0]
    action_dim = len(env.vehicles) * 4
    
    agent = ResidualPPOAgent(state_dim=state_dim, action_dim=action_dim, lr=1e-3, ablation_mode='ours')
    # 不强制加载预训练，因为我们要看“刚切入灾难时的梯度爆发”，即使随机初始化，只要是PPO第一步梯度都会有
    # 但为了真实还原灾难期的决策，我们加载收敛的 prior，以及我们当前实验使用的模型
    agent.load_expert('model/prior_dnn_expert.pth')
    
    # if os.path.exists('model/ours_converged.pth'):
    #     try:
    #         agent.load_model('model/ours_converged.pth')
    #     except Exception as e:
    #         print(f"[Warn] Shape mismatch due to architecture upgrade. Using random residual_net.")
        
    print("=== [Academic Validation] Alpha Gradient Diagnostic ===")
    
    state = env.reset()
    
    # 手动触发算力雪崩
    env.trigger_capacity_avalanche()
    print("\n[EVENT] Avalanche Triggered. Forcing PPO update every step for 10 steps...")
    print(f"{'Step':<5} | {'Alpha_Mean':<12} | {'Alpha_Max':<12} | {'Grad_Norm':<12} | {'R_max*Sig(Max_a)':<20}")
    print("-" * 75)
    
    for t in range(1, 11):
        action, action_logprob, state_val, brain_wave = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        
        agent.store_transition((state, action, action_logprob, reward, done))
        
        # 强制单步更新，获取实时梯度
        agent.update()
        
        alpha = agent.residual_net.alpha
        alpha_mean = alpha.mean().item()
        alpha_max = alpha.max().item()
        
        grad_norm = alpha.grad.norm().item() if alpha.grad is not None else 0.0
        adaptive_scale_max = 5.0 * torch.sigmoid(torch.tensor(alpha_max)).item()
        
        print(f"{t:<5d} | {alpha_mean:<12.4f} | {alpha_max:<12.4f} | {grad_norm:<12.6f} | {adaptive_scale_max:<20.4f}")
        
        state = next_state
        
    print("\n[OK] Diagnostic complete.")

if __name__ == '__main__':
    run_alpha_grad_test()
