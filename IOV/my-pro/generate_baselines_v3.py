import torch
import numpy as np
import random
from iov_env_v3 import IoVEnvV3
from residual_ppo_agent_v3 import ResidualPPOAgent, Prior_DNN
from generate_expert_v3 import simulated_annealing

def run_baselines():
    print("==================================================")
    print("=== 生成顶级期刊对比基线数据 (Phase 3 - Step 4) ===")
    print("==================================================")
    
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    
    env = IoVEnvV3(num_vehicles=10)
    state_dim = env.state_dim
    action_dim = env.num_vehicles * 4
    
    # 1. Ours (Residual PPO)
    agent_ours = ResidualPPOAgent(state_dim=state_dim, action_dim=action_dim, ablation_mode='ours')
    agent_ours.load_expert('model/prior_dnn_expert.pth')
    agent_ours.load_model('model/residual_ppo_ours.pth')
    
    # 2. Prior DNN
    prior_dnn = Prior_DNN(state_dim, action_dim).to(torch.device('cpu'))
    prior_dnn.load_state_dict(torch.load('model/prior_dnn_expert.pth'))
    prior_dnn.eval()
    
    # 3. Vanilla PPO (Untrained, showing cold-start disaster)
    agent_ppo = ResidualPPOAgent(state_dim=state_dim, action_dim=action_dim, ablation_mode='ppo')
    
    costs = {'ours': [], 'prior': [], 'ppo': [], 'oracle': []}
    
    # 我们运行 1 个超长 Episode (100步正常，100步灾难)
    for model_name in ['ours', 'prior', 'ppo', 'oracle']:
        print(f"\n--- Running Baseline: {model_name.upper()} ---")
        
        # 重置相同的随机数种子，确保环境轨迹绝对一致
        np.random.seed(seed)
        random.seed(seed)
        
        env.reset_disasters()
        state = env.reset()
        
        for step in range(1, 201):
            if step == 101:
                env.trigger_flood()
                state = env.reset()
                
            if model_name == 'ours':
                action, _, _, _ = agent_ours.select_action(state)
            elif model_name == 'ppo':
                action, _, _, _ = agent_ppo.select_action(state)
            elif model_name == 'prior':
                obs_tensor = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    logits = prior_dnn(obs_tensor)
                    logits = logits.view(-1, 10, 4)
                    action = torch.argmax(logits, dim=2).squeeze(0).numpy()
            elif model_name == 'oracle':
                # [核心修正] 建立随机数沙盒，隔离 SA 算法对全局 RNG 状态的污染！
                saved_random_state = random.getstate()
                saved_np_state = np.random.get_state()
                
                init_actions = np.array([random.randint(0, 3) for _ in range(env.num_vehicles)])
                best_actions, best_fitness = simulated_annealing(env, init_actions, max_iter=200, T_start=50.0, alpha=0.92)
                action = best_actions
                
                random.setstate(saved_random_state)
                np.random.set_state(saved_np_state)
                
            state, reward, done, info = env.step(action)
            costs[model_name].append(info['avg_cost'])
            
            if step % 20 == 0:
                print(f"  Step {step} | Cost: {info['avg_cost']:.2f}")

    print("\n--- 实验数据收集完成 ---")
    np.save('model/plot_costs.npy', costs)
    print("数据已保存至 model/plot_costs.npy，可用于直接绘制论文折线图。")

if __name__ == '__main__':
    run_baselines()
