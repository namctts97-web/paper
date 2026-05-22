import numpy as np

class Heuristic_Agent:
    def __init__(self, N_vehicles=6):
        self.N = N_vehicles
        
    def select_action(self, state):
        # state 形状: (N, 6)
        # 特征含义: [x, y, D_i, lambda_i, mec_load, avg_reward]
        actions = []
        for i in range(self.N):
            # 简单的启发式规则 (Worst-Fit / Service-Aware):
            # 1. 检查是否是紧急任务 (URLLC, lambda_i > 0.5)
            # 2. 检查 MEC 负载 (mec_load > 0.8)
            lambda_i = state[i, 3]
            mec_load = state[i, 4]
            
            if lambda_i > 0.5:
                # 紧急任务，如果 MEC 未满载优先发往 MEC，否则本地执行
                if mec_load < 0.8:
                    actions.append(1) # Local MEC
                else:
                    actions.append(0) # Local
            else:
                # 普通大体积数据任务 (eMBB)，发往远端 MEC 或云端
                if mec_load < 0.9:
                    actions.append(2) # Remote MEC
                else:
                    actions.append(3) # Cloud
                    
        return np.array(actions)
