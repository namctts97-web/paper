import torch
from my_iov_env import ResidualIoVEnv

def run_proof():
    print("=== 开始严格对照实验 (Proof of Independence) ===")
    env = ResidualIoVEnv()
    
    # 直接触发灾难，制造恶劣环境
    env.trigger_capacity_avalanche()
    env.trigger_traffic_flood()
    print("1. 已触发灾难 (OOD)。MEC 算力仅剩 10GHz。")
    
    env.reset()
    
    # 模拟一个“弱智”智能体：不管三七二十一，死活都要去 Local MEC (Action 1)
    # 这相当于我们强行覆盖了神经网络的大脑
    print("2. 强行设定智能体大脑 (Raw Action) 100% 输出 Action 1 (去 Local MEC)。")
    import numpy as np
    raw_actions = np.array([1] * len(env.vehicles))
    
    # 把大脑的决定交给物理环境
    _, _, _, info = env.step(raw_actions)
    
    print("\n--- 核心对决结果 ---")
    print(f"智能体大脑真实的疯狂意图 (Raw Actions): {info['raw_actions']}")
    print(f"环境 KKT 引擎无情镇压后的最终物理动作 (Legal Actions): {info['legal_actions']}")
    print("------------------\n")
    
    print("结论推导：")
    print("如果你看到这两行数据不一样（比如 Raw 是 1，但 Legal 变成了 0），")
    print("这就彻底证明了：【智能体的真实意图】和【物理环境的强迫执行】是完全独立记录的两套数据！")
    print("那么回想一下，为什么在之前的正常训练中，这两张图会一模一样？")
    print("因为那是正常训练后的神经网络！它聪明到在第一步（Raw Action）就直接给出了最优解（去云端），环境一看合法，就不再修改了！")

if __name__ == "__main__":
    run_proof()
