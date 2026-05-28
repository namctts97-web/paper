import numpy as np
import random
import math
import os
from iov_env_v3 import IoVEnvV3

def simulated_annealing(env, initial_actions, max_iter=500, T_start=100.0, T_end=0.1, alpha=0.95):
    """
    模拟退火算法，直接调用环境的原生 evaluate_actions 接口进行无状态评估，
    保证找到的适应度与环境最终判定的 Reward 逻辑 100% 同构！
    """
    current_actions = np.copy(initial_actions)
    current_cost = env.evaluate_actions(current_actions)
    
    best_actions = np.copy(current_actions)
    best_cost = current_cost
    
    T = T_start
    num_vehicles = len(current_actions)
    
    for _ in range(max_iter):
        if T < T_end:
            break
            
        # 生成邻居解 (Neighbor generation)
        neighbor_actions = np.copy(current_actions)
        vehicle_idx = random.randint(0, num_vehicles - 1)
        available_actions = [0, 1, 2, 3]
        available_actions.remove(neighbor_actions[vehicle_idx])
        neighbor_actions[vehicle_idx] = random.choice(available_actions)
        
        # 100% MDP 同构的适应度评估！
        neighbor_cost = env.evaluate_actions(neighbor_actions)
        
        if neighbor_cost < current_cost:
            current_actions = np.copy(neighbor_actions)
            current_cost = neighbor_cost
            if current_cost < best_cost:
                best_actions = np.copy(current_actions)
                best_cost = current_cost
        else:
            p = math.exp(-(neighbor_cost - current_cost) / T)
            if random.random() < p:
                current_actions = np.copy(neighbor_actions)
                current_cost = neighbor_cost
                
        T *= alpha
        
    return best_actions, best_cost

def generate_data(num_episodes=10, steps_per_ep=100, save_path="data/expert_data_v3.npy"):
    print("=========================================")
    print("  启动专家数据大生产 (基于严格的 MDP 同构) ")
    print("=========================================")
    env = IoVEnvV3(num_vehicles=10)
    expert_states = []
    expert_actions = []
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    for ep in range(num_episodes):
        print(f"\n>>> Generating Episode {ep+1}/{num_episodes} <<<")
        # 随机触发灾难（OOD），让专家算法学习应对策略
        env.reset_disasters()
        r = random.random()
        if r < 0.2:
            print("  [OOD 注入] Traffic Flood 洪峰来袭！")
            env.trigger_flood()
        elif r < 0.4:
            print("  [OOD 注入] Compute Avalanche 算力雪崩！")
            env.trigger_avalanche()
        else:
            print("  [Normal] 正常行驶状态")
            
        obs = env.reset()
        
        for step in range(steps_per_ep):
            # 初始化随机动作
            init_actions = np.array([random.randint(0, 3) for _ in range(env.num_vehicles)])
            
            # 使用模拟退火寻找最优解
            best_a, best_c = simulated_annealing(env, init_actions, max_iter=800, T_start=50.0, alpha=0.92)
            
            expert_states.append(obs)
            expert_actions.append(best_a)
            
            # 执行动作，环境时钟步进
            obs, reward, done, info = env.step(best_a)
            
            if (step + 1) % 20 == 0:
                print(f"  Step {step+1}: SA Best Cost = {best_c:.4f} | Env Actual Reward = {reward:.4f}")
                
    expert_states = np.array(expert_states)
    expert_actions = np.array(expert_actions)
    
    print(f"\nGenerated {len(expert_states)} samples.")
    print(f"Saving to {save_path}...")
    np.save(save_path, {'states': expert_states, 'actions': expert_actions}, allow_pickle=True)
    print("Generation Complete.")

if __name__ == '__main__':
    generate_data(num_episodes=10, steps_per_ep=100)
