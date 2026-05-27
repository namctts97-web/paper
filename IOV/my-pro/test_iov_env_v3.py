import numpy as np
from iov_env_v3 import IoVEnvV3

def test_env():
    print("=========================================")
    print("Initializing IoV_Env_V3 Physical Engine...")
    print("=========================================")
    env = IoVEnvV3(num_vehicles=10)
    
    # --- Normal Phase ---
    print("\n[Phase 1] Normal Condition")
    obs = env.reset()
    for step in range(3):
        actions = env.action_space.sample()
        obs, reward, done, info = env.step(actions)
        print(f"Step {step+1}:")
        print(f"  Actions -> Local:{list(actions).count(0)}, MEC:{list(actions).count(1)}, Offsite:{list(actions).count(2)}, Cloud:{list(actions).count(3)}")
        print(f"  Offloading Users -> URLLC (N_U): {info['N_U']}, eMBB (N_E): {info['N_E']}")
        print(f"  Bandwidth/User -> URLLC: {info['Allocated_B_URLLC']/1e6:.2f} MHz, eMBB: {info['Allocated_B_eMBB']/1e6:.2f} MHz")
        print(f"  Interference -> ZITD_norm: {info['ZITD_norm']:.6f}, ICI (Watts): {info['ICI']:.6f}")
        print(f"  Metrics -> Avg Cost: {info['avg_cost']:.4f}, Reward: {reward:.4f}\n")
        
    # --- Traffic Flood OOD ---
    print("=========================================")
    print("[Phase 2] Triggering Spatial Traffic Flood OOD")
    print("Expect ZITD to skyrocket and Costs to increase...")
    print("=========================================")
    env.trigger_flood()
    obs = env.reset()
    for step in range(3):
        actions = env.action_space.sample()
        obs, reward, done, info = env.step(actions)
        print(f"Step {step+1}:")
        print(f"  Actions -> Local:{list(actions).count(0)}, MEC:{list(actions).count(1)}, Offsite:{list(actions).count(2)}, Cloud:{list(actions).count(3)}")
        print(f"  Offloading Users -> URLLC (N_U): {info['N_U']}, eMBB (N_E): {info['N_E']}")
        print(f"  Bandwidth/User -> URLLC: {info['Allocated_B_URLLC']/1e6:.2f} MHz, eMBB: {info['Allocated_B_eMBB']/1e6:.2f} MHz")
        print(f"  Interference -> ZITD_norm: {info['ZITD_norm']:.6f}, ICI (Watts): {info['ICI']:.6f}")
        print(f"  Metrics -> Avg Cost: {info['avg_cost']:.4f}, Reward: {reward:.4f}\n")

if __name__ == '__main__':
    test_env()
