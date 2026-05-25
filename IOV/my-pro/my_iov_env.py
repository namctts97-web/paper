import gym
from gym import spaces
from scipy.special import logsumexp
import numpy as np
import random
from edge_sim_py import *
import math
from baseline.env_v2_core import get_env_params_v2
class ResidualIoVEnv(gym.Env):
    def __init__(self, ablation_mode='ours'):
        super(ResidualIoVEnv, self).__init__()
        self.ablation_mode = ablation_mode
        self.disaster_count = 0
        
        # 核心物理参数对齐 (调用 V2 接口)
        self.params = get_env_params_v2()
        self.f_local = self.params['f_local']
        self.f_mec = self.params['f_mec']
        self.f_offsite = self.params['f_offsite']
        self.f_cloud = self.params['f_cloud']
        self.p_i = self.params['p_ue']
        # 导师修正：为了让云端在真实物理中具备吸引力，必须拓宽车端到基站的 5G 无线带宽，消除上行瓶颈
        self.B = 100 * (10**6) # 100 MHz (原 20MHz 太拥挤导致网络死守本地)
        self.noise_power = self.params['sigma2']
        self.r_eo = self.params['r_eo']
        self.r_ec = self.params['r_ec']
        self.prop_eo = self.params['prop_eo']
        self.prop_ec = self.params['prop_ec']
        self.k_energy = self.params['k_energy']
        # === 严谨物理逻辑修正：遵守边缘到云端的算力递增层级 ===
        self.f_mec = 28 * 10**9      # 本地 MEC 恢复为正常的 28GHz (中小型边缘节点)
        self.f_offsite = 56 * 10**9  # 远端 MEC 恢复为 56GHz (汇聚层大节点，算力是本地的两倍)
        # 核心物理约束：虽然远端算力大，但现实中基站间的“可用回传带宽”往往是被复用和拥塞的
        self.r_eo = 50 * 10**6       # 设为 50Mbps (真实环境下的可用拥塞带宽)
        self.r_ec = 100 * 10**6      # 到云端的专线带宽 100Mbps
        
        self.T_max = 1.0               # 最大容忍时间窗 1s
        self.eta = 0.05                # 乒乓惩罚系数 (缩小以适应 DIR 比例)
        self.V_max = 1.5               # 李雅普诺夫漂移最大惩罚比例 (Lyapunov Maximum Penalty Ratio)
        
        # 初始化模拟器
        self.simulator = Simulator(
            tick_duration=1,
            tick_unit="seconds",
            stopping_criterion=lambda model: model.schedule.steps == 1000,
            resource_management_algorithm=self._algorithm_wrapper,
        )
        # 使用 EdgeAISIM 目录下的数据集作为基础拓扑
        # 注意：这里路径需要根据实际位置调整，假设在 D:\Enviroment\论文\IOV\EdgeAISIM\sample_dataset1.json
        self.simulator.initialize(input_file="d:/Enviroment/论文/IOV/EdgeAISIM/sample_dataset1.json")
        
        # 覆盖并对齐对象参数
        self.vehicles = User.all()
        self.edge_servers = EdgeServer.all()
        
        for es in self.edge_servers:
            es.cpu = self.f_mec # 设置总算力
            
        # 绝对物理保障：固定 2辆 URLLC 和 4辆 eMBB，避免出现某一代没有 URLLC 导致的统计全 0 (假直线)
        urllc_indices = random.sample(range(len(self.vehicles)), 2)
        
        for i, v in enumerate(self.vehicles):
            # 初始信道增益与速度
            v.velocity = random.uniform(10, 120) / 3.6 # km/h to m/s
            v.h_t = 1.0 
            
            # V3 物理特性绑定：任务特征 (U_i, D_i) 与 QoS 双峰高斯模型绑定
            if i in urllc_indices:
                # URLLC 任务 (极度敏感于时延，小数据极速计算)
                v.lambda_i = np.clip(np.random.normal(0.85, 0.05), 0.7, 1.0)
                v.U_i = random.uniform(10, 50) * 1024 * 8            # 10KB - 50KB (bits)
                v.D_i = random.uniform(100, 500) * (10**6)           # 0.1 - 0.5 Gcycles
            else:
                # 70% eMBB 任务，将其一分为二：AI计算型 vs 传统背景型
                v.lambda_i = np.clip(np.random.normal(0.25, 0.1), 0.0, 0.5)
                
                if random.random() < 0.5:
                    # 1. AI 计算型 eMBB (如 3D 点云检测)：归宿 -> 云端 / 边缘
                    v.U_i = random.uniform(5000, 20000) * 1024 * 8       # 5MB - 20MB (bits)
                    v.D_i = random.uniform(50, 200) * (10**9)            # 50 - 200 Gcycles (计算密集)
                else:
                    # 2. 传统数据型 eMBB (如 视频打包缓存)：归宿 -> 死守本地
                    # 导师约束：引入重数据、轻计算任务
                    v.U_i = random.uniform(20000, 50000) * 1024 * 8      # 20MB - 50MB (bits) 数据量极大
                    v.D_i = random.uniform(1, 2) * (10**9)               # 1 - 2 Gcycles 计算量极小
                
            v.mu_i = 1.0 - v.lambda_i

        # 状态记录
        self.prev_actions = {v.id: 0 for v in self.vehicles}
        self.current_step_actions = None
        
        # 净化 __init__：重置灾难标记 (导师指令)
        self.is_avalanche_triggered = False
        self.is_flood_triggered = False
        
        # 遥测记忆变量 (Telemetry)
        self.last_mec_load = 0.0
        self.last_avg_reward = 0.0
        
        # 绝对不要在这里调用 self._apply_disaster_state() ！！！

        # 定义 Gym 空间 (每个车辆 0:本地, 1:本地MEC, 2:远程MEC, 3:云)
        self.action_space = spaces.MultiDiscrete([4] * len(self.vehicles))
        # 观测空间重构：(6辆车, 7维特征) - 引入前瞻特征
        self.observation_space = spaces.Box(low=-1000, high=1e9, shape=(len(self.vehicles), 7), dtype=np.float32)

    def _apply_disaster_state(self):
        """确保 reset 之后，灾难状态依然生效 (听取导师建议，回归极简异构验证)"""
        if self.is_avalanche_triggered:
            if self.disaster_count == 1:
                # OOD 1：基站算力 (f_mec) 暴跌至 5~10GHz，光纤不动 (100Mbps)
                # 预期行为：网络会将 eMBB 任务转移到云端或远端 (Action 3 或 2)
                self.f_mec = random.uniform(5.0, 10.0) * (10**9) 
                self.shared_r_ec = 100.0 * (10**6) 
                self.shared_r_eo = 50.0 * (10**6) 
            elif self.disaster_count == 2:
                # OOD 2：基站算力暴跌至 1~5GHz，光纤拥塞 (暴跌至 1~5Mbps)
                # 预期行为：云端之路断绝，网络被迫退回本地 (Action 0)
                self.f_mec = random.uniform(1.0, 5.0) * (10**9) 
                self.shared_r_ec = random.uniform(1.0, 5.0) * (10**6) 
                self.shared_r_eo = random.uniform(1.0, 5.0) * (10**6) 
            else:
                self.f_mec = 2.0 * (10**9)
                
            for es in self.edge_servers:
                es.cpu = self.f_mec
                
        if getattr(self, 'is_flood_triggered', False):
            for v in self.vehicles:
                v.U_i *= 10.0
                v.D_i *= 10.0
                
        if getattr(self, 'is_compute_flood_triggered', False):
            for v in self.vehicles:
                v.D_i *= 50.0


    def _algorithm_wrapper(self, parameters):
        """对接底层框架的算法回调：纯执行器，仅执行合法化后的动作"""
        if self.current_step_actions is None:
            return
            
        for i, vehicle in enumerate(self.vehicles):
            action = self.current_step_actions[i]
            for app in vehicle.applications:
                for service in app.services:
                    if action == 1:
                        target = self.edge_servers[0]
                        if service.server != target:
                            service.provision(target_server=target)
                    else:
                        # 本地处理，不执行迁移
                        pass

    def calculate_fading_rate(self, vehicle):
        """Jakes AR(1) 小尺度衰落模型"""
        # 相关系数 rho 与车速反相关
        # 设定：10km/h -> 0.95, 120km/h -> 0.1
        v_kmh = vehicle.velocity * 3.6
        rho = max(0.1, 0.95 - (v_kmh - 10) * (0.85 / 110))
        
        e_t = np.random.normal(0, 1)
        vehicle.h_t = rho * vehicle.h_t + np.sqrt(1 - rho**2) * e_t
        # 数值保护：防止信道增益过小导致速率为 0
        return max(1e-3, abs(vehicle.h_t))

    def greedy_knapsack_projection(self, raw_actions, estimated_rates):
        """
        [终极学术修正] 基于带约束松弛背包问题的 KKT 闭式解 (Closed-form KKT Projection)
        与物理信道 CSI 前向耦合
        """
        legal_actions = np.copy(raw_actions)
        mec_demand_cycles = 0
        for i, a in enumerate(raw_actions):
            if a == 1:
                mec_demand_cycles += self.vehicles[i].D_i
                
        # 物理限制：基站在给定时延窗内的绝对算力极限
        max_mec_capacity = self.f_mec * self.T_max
        
        if mec_demand_cycles > max_mec_capacity:
            # [终极学术修正] 计算真实的拉格朗日梯度 (Lagrangian Gradient)
            def kkt_gradient(idx):
                v = self.vehicles[idx]
                est_rate = estimated_rates[idx] # 使用前向耦合的瞬时物理速率
                T_loc = v.D_i / self.f_local
                E_loc = self.k_energy * (self.f_local**2) * v.D_i
                Cost_loc = v.lambda_i * T_loc + v.mu_i * E_loc
                
                T_mec = v.U_i / est_rate + v.D_i / self.f_mec
                E_mec = self.p_i * (v.U_i / est_rate)
                Cost_mec = v.lambda_i * T_mec + v.mu_i * E_mec
                
                # 边际收益 / 资源消耗
                return (Cost_loc - Cost_mec) / v.D_i

            mec_vehicles = [i for i, a in enumerate(raw_actions) if a == 1]
            # 按照拉格朗日梯度从小到大排序 (优先踢掉卸载性价比最低的)
            mec_vehicles.sort(key=kkt_gradient)
            
            current_demand = mec_demand_cycles
            for i in mec_vehicles:
                if current_demand <= max_mec_capacity:
                    break
                legal_actions[i] = 0 # 物理退回：基站拒绝接入，强制本地计算
                current_demand -= self.vehicles[i].D_i
                
        return legal_actions

    def step(self, raw_actions):
        # [终极学术修正] 物理层前向耦合：双轨制宏微观容量模型
        K_micro_samples = 100
        offloading_count = sum(1 for a in raw_actions if a > 0)
        available_B = self.B / offloading_count if offloading_count > 0 else self.B
        
        estimated_rates_macro = []
        all_R_samples = []
        
        for i, v in enumerate(self.vehicles):
            # 宏观游走一次，得到此时刻的基准增益
            macro_h = self.calculate_fading_rate(v) 
            scale = max(1e-4, macro_h)
            
            # 微观快照：在 1秒 宏观跨度内采样 K 次
            h_samples = np.random.rayleigh(scale=scale, size=K_micro_samples)
            
            d_m = max(1e-4, math.sqrt(v.coordinates[0]**2 + v.coordinates[1]**2))
            path_loss = 127 + 30 * math.log10(d_m / 1000.0)
            signal_power_samples = self.p_i * (h_samples**2) * (10**(-path_loss/10))
            
            # 瞬时容量微观样本
            R_samples = available_B * np.log2(1 + signal_power_samples / self.noise_power)
            all_R_samples.append(R_samples)
            
            # GKP 准入控制需要一个宏观预估速率，此处使用遍历容量作为基准
            estimated_rates_macro.append(np.mean(R_samples))

        # 1. 经过物理信道耦合的 GKP 准入控制 (原 KKT)
        legal_actions = self.greedy_knapsack_projection(raw_actions, estimated_rates_macro)
        self.current_step_actions = legal_actions # 让 _algorithm_wrapper 去执行合法的
        
        # 2. 计算乒乓惩罚 (必须使用 legal_actions 和上一次的 legal_actions 比较)
        penalty = 0
        for i, v in enumerate(self.vehicles):
            penalty += self.eta * abs(legal_actions[i] - self.prev_actions[v.id])
            self.prev_actions[v.id] = legal_actions[i] # 更新为合法的动作
            
        # 3. 推进物理模拟器 (仅执行 1 个 tick)
        self.simulator.step()
        
        # 4. 计算真实延迟与能耗 (所有判断必须是 legal_actions)
        total_cost = 0
        actual_load_cycles = 0
        total_latency = 0
        total_energy = 0
        urllc_latency, urllc_energy, urllc_count = 0, 0, 0
        embb_latency, embb_energy, embb_count = 0, 0, 0
        success_count = 0
        
        # ==========================================
        # 核心手术 1：引入回传带宽拥塞共享
        # ==========================================
        count_offsite = sum(1 for a in legal_actions if a == 2)
        count_cloud = sum(1 for a in legal_actions if a == 3)
        
        shared_r_eo = self.r_eo / count_offsite if count_offsite > 0 else self.r_eo
        shared_r_ec = self.r_ec / count_cloud if count_cloud > 0 else self.r_ec
        # ==========================================
        
        # 统计选择本地 MEC 和远程 MEC 的车辆数，用于 Convex 算力分配 (与专家一致)
        idx_mec = np.where(np.array(legal_actions) == 1)[0]
        idx_offsite = np.where(np.array(legal_actions) == 2)[0]
        f_assigned = np.zeros(len(self.vehicles))
        
        if len(idx_mec) > 0:
            sum_sqrt_D = sum(math.sqrt(self.vehicles[idx].D_i) for idx in idx_mec)
            for idx in idx_mec:
                f_assigned[idx] = self.f_mec * (math.sqrt(self.vehicles[idx].D_i) / sum_sqrt_D)
                
        if len(idx_offsite) > 0:
            sum_sqrt_D = sum(math.sqrt(self.vehicles[idx].D_i) for idx in idx_offsite)
            for idx in idx_offsite:
                f_assigned[idx] = self.f_offsite * (math.sqrt(self.vehicles[idx].D_i) / sum_sqrt_D)
            
        reward_list = []
        
        for i, v in enumerate(self.vehicles):
            # 重新根据 legal_actions 计算最终共享带宽
            actual_offloading_count = sum(1 for a in legal_actions if a > 0)
            actual_available_B = self.B / actual_offloading_count if actual_offloading_count > 0 else self.B
            
            # 复用刚刚生成的微观速率，按带宽比例放缩
            R_samples_actual = all_R_samples[i] * (actual_available_B / available_B) if available_B > 0 else all_R_samples[i]
            
            # 双轨制计算物理速率
            if v.lambda_i > 0.6:
                # URLLC 任务：使用 Effective Capacity (有效容量) 严防深衰落中断
                theta = 0.05 # QoS 指数
                # C_eff = - (1/theta) * (logsumexp(-theta * R) - log(K))
                X = -theta * R_samples_actual
                c_eff = - (1.0 / theta) * (logsumexp(X) - math.log(K_micro_samples))
                rate = max(1e-3, c_eff)
            else:
                # eMBB 任务：使用 Ergodic Capacity (遍历容量) 
                rate = max(1e-3, np.mean(R_samples_actual))
            
            U = v.U_i
            D = v.D_i
            
            # 计算绝对物理下界代价（本地执行）
            Cost_local = v.lambda_i * (D / self.f_local) + v.mu_i * (self.k_energy * (self.f_local**2) * D)
            
            if legal_actions[i] == 0:
                T_i = D / self.f_local
                E_i = self.k_energy * (self.f_local**2) * D
            elif legal_actions[i] == 1:
                T_trans = U / rate
                T_exec = D / f_assigned[i]
                T_i = T_trans + T_exec
                E_i = self.p_i * T_trans
                actual_load_cycles += D
            elif legal_actions[i] == 2:
                T_trans = U / rate + U / shared_r_eo + self.prop_eo
                T_exec = D / f_assigned[i]
                T_i = T_trans + T_exec
                E_i = self.p_i * (U / rate)
            elif legal_actions[i] == 3:
                T_trans = U / rate + U / shared_r_ec + self.prop_ec
                T_exec = D / self.f_cloud
                T_i = T_trans + T_exec
                E_i = self.p_i * (U / rate)
                
            Cost_actual = v.lambda_i * T_i + v.mu_i * E_i
            
            total_latency += T_i
            total_energy += E_i
            total_cost += Cost_actual
            
            if v.lambda_i > 0.6:
                urllc_latency += T_i
                urllc_energy += E_i
                urllc_count += 1
            else:
                embb_latency += T_i
                embb_energy += E_i
                embb_count += 1
            
            if T_i <= self.T_max:
                success_count += 1
            else:
                x = (T_i - self.T_max) / self.T_max
                if self.ablation_mode == 'ablation_hardclip':
                    # [消融实验 A] 关闭 Arcsinh 保护，使用简单的线性罚函数 (后面会进行 Hard Clip)
                    Cost_actual = Cost_local * (1.0 + x)
                else:
                    # [终极学术修正] Arcsinh 渐进有界漂移罚函数 (Float32 精度安全)
                    penalty_ratio = 1.0 + (self.V_max - 1.0) * np.arcsinh(2.0 * x)
                    Cost_actual = Cost_local * penalty_ratio
                
            if raw_actions[i] > 0 and legal_actions[i] == 0:
                # 被拒惩罚严格遵循李雅普诺夫下界最大比例
                Cost_actual = Cost_local * self.V_max
                
            # DIR (Dimensionless Improvement Ratio) 计算
            r_i = 1.0 - (Cost_actual / Cost_local)
            
            # 张量截断 (保证理论界限)
            r_i = np.clip(r_i, 1.0 - self.V_max, 1.0)
            reward_list.append(r_i)
            
        # 5. Reward (平均 DIR - 乒乓惩罚)
        reward = np.mean(reward_list) - penalty
        obs = []
        for v in self.vehicles:
            # 专属车辆级视角矩阵构建 (完美 MDP 马尔可夫状态)
            obs.append([
                v.coordinates[0] / 1000.0,
                v.coordinates[1] / 1000.0,
                v.D_i / 1e9,
                v.lambda_i,
                self.last_mec_load,
                # 导师新增：MEC 剩余可用算力百分比 (雷区预警特征 1)
                1.0 - min(1.0, self.last_mec_load),
                # 导师新增：竞争车辆比例 (雷区预警特征 2)
                sum(1 for a in self.prev_actions.values() if a > 0) / len(self.vehicles)
            ])
            
        obs = np.array(obs, dtype=np.float32)
        done = self.simulator.schedule.steps >= 1000
        
        info = {
            "raw_actions": raw_actions.tolist(),
            "legal_actions": legal_actions.tolist(), # 输出对比
            "avg_cost": total_cost / len(self.vehicles),
            "avg_latency": total_latency / len(self.vehicles),
            "avg_energy": total_energy / len(self.vehicles),
            "urllc_latency": urllc_latency / max(1, urllc_count),
            "embb_latency": embb_latency / max(1, embb_count),
            "urllc_energy": urllc_energy / max(1, urllc_count),
            "embb_energy": embb_energy / max(1, embb_count),
            "success_rate": success_count / len(self.vehicles),
            "mec_load": actual_load_cycles / (self.f_mec * self.T_max)
        }
        
        # 更新给下一回合的遥测记忆
        self.last_mec_load = info["mec_load"]
        self.last_avg_reward = reward
        
        return obs, reward, done, info

    def reset(self):
        # 1. 保存当前的灾难状态快照 (导师指令)
        avalanche_state = getattr(self, 'is_avalanche_triggered', False)
        flood_state = getattr(self, 'is_flood_triggered', False)
        
        # 2. 调用干净的 __init__，重置出 1 倍正常的车辆数据
        self.__init__() 
        
        # 3. 恢复灾难标记
        self.is_avalanche_triggered = avalanche_state
        self.is_flood_triggered = flood_state
        
        # 4. 关键：全生命周期内，只在这里执行且仅执行 1 次灾难放大！
        self._apply_disaster_state()
        
        obs = []
        for v in self.vehicles:
            obs.append([
                v.coordinates[0] / 1000.0,
                v.coordinates[1] / 1000.0,
                v.D_i / 1e9,
                v.lambda_i,
                self.last_mec_load,
                1.0 - min(1.0, self.last_mec_load),
                sum(1 for a in self.prev_actions.values() if a > 0) / len(self.vehicles)
            ])
            
        return np.array(obs, dtype=np.float32)

    def trigger_capacity_avalanche(self):
        """触发算力雪崩"""
        self.disaster_count += 1
        self.is_avalanche_triggered = True
        self._apply_disaster_state()
        print("\n[EVENT] CRITICAL: Capacity Avalanche Triggered! MEC Capacity -> 10GHz")

    def trigger_traffic_flood(self):
        """触发流量洪峰"""
        self.is_flood_triggered = True
        self._apply_disaster_state()
        print("\n[EVENT] CRITICAL: Traffic Flood Triggered! Task Workload x10")

    def trigger_compute_flood(self):
        """触发计算型洪峰 (如紧急 AI 推理)"""
        self.is_compute_flood_triggered = True
        self._apply_disaster_state()
        print("\n[EVENT] CRITICAL: Compute Flood Triggered! Task Computation x50")

    def recover_from_avalanche(self):
        """恢复算力雪崩"""
        self.is_avalanche_triggered = False
        self.f_mec = self.params['f_mec']
        self.shared_r_ec = 10 * (10**6) # 恢复回传
        for es in self.edge_servers:
            es.cpu = self.f_mec
        print("\n[EVENT] RECOVERY: Capacity Avalanche Resolved! MEC Capacity -> 28GHz")

    def recover_from_flood(self):
        """恢复流量洪峰"""
        self.is_flood_triggered = False
        print("\n[EVENT] RECOVERY: Traffic Flood Resolved! Task Workload Normal")

    def recover_from_compute_flood(self):
        """恢复计算洪峰"""
        self.is_compute_flood_triggered = False
        print("\n[EVENT] RECOVERY: Compute Flood Resolved! Task Computation Normal")
