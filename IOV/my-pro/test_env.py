from my_iov_env import ResidualIoVEnv
import numpy as np

def run_test():
    print("=== Initializing ResidualIoVEnv ===")
    env = ResidualIoVEnv()
    
    print("\n--- Phase 1: Normal Operation (Step 1-5) ---")
    for i in range(1, 6):
        # 生成随机动作 (0 或 1)
        actions = np.random.randint(0, 2, size=env.action_space.shape)
        obs, reward, done, info = env.step(actions)
        
        print(f"Step {i}:")
        print(f"  Raw Actions:   {info['raw_actions']}")
        print(f"  Legal Actions: {info['legal_actions']}")
        print(f"  Avg Latency: {info['avg_latency']:.4f} s")
        print(f"  MEC Load: {info['mec_load']:.2%}")
        print(f"  Reward: {reward:.4f}")

    print("\n--- Phase 2: OOD Disaster (Trigger Avalanche) ---")
    env.trigger_capacity_avalanche()
    
    for i in range(6, 9):
        actions = np.random.randint(0, 2, size=env.action_space.shape)
        obs, reward, done, info = env.step(actions)
        print(f"Step {i} (Post-Avalanche):")
        print(f"  Raw Actions:   {info['raw_actions']}")
        print(f"  Legal Actions: {info['legal_actions']}")
        print(f"  Avg Latency: {info['avg_latency']:.4f} s")
        print(f"  MEC Load: {info['mec_load']:.2%}")

    print("\n--- Phase 3: OOD Disaster (Trigger Traffic Flood) ---")
    env.trigger_traffic_flood()
    
    for i in range(9, 12):
        actions = np.random.randint(0, 2, size=env.action_space.shape)
        obs, reward, done, info = env.step(actions)
        print(f"Step {i} (Post-Flood):")
        print(f"  Raw Actions:   {info['raw_actions']}")
        print(f"  Legal Actions: {info['legal_actions']}")
        print(f"  Avg Latency: {info['avg_latency']:.4f} s")
        print(f"  MEC Load: {info['mec_load']:.2%}")

if __name__ == "__main__":
    run_test()
