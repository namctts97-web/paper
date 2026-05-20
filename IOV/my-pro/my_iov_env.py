import gym
from gym import spaces
import numpy as np
import random
from edge_sim_py import *
import math
from baseline.env_v2_core import get_env_params_v2
class ResidualIoVEnv(gym.Env):
    def __init__(self):
        super(ResidualIoVEnv, self).__init__()
        
        # 核心物理参数对齐 (调用 V2 接口)
        self.params = get_env_params_v2()
        self.f_local = self.params['f_local']
        self.f_mec = self.params['f_mec']
        self.f_offsite = self.params['f_offsite']
        self.f_cloud = self.params['f_cloud']
        self.p_i = self.params['p_ue']
        self.B = self.params['B']
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
        self.eta = 0.5                 # 乒乓惩罚系数
        
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
            
        for v in self.vehicles:
            # 初始信道增益与速度
            v.velocity = random.uniform(10, 120) / 3.6 # km/h to m/s
            v.h_t = 1.0 
            
            # V3 物理特性绑定：任务特征 (U_i, D_i) 与 QoS 双峰高斯模型绑定
            if random.random() < 0.3:
                # 30% URLLC 任务 (极度敏感于时延，小数据重计算)
                v.lambda_i = np.clip(np.random.normal(0.85, 0.05), 0.7, 1.0)
                v.U_i = random.uniform(100, 500) * 1024 * 8          # 100KB - 500KB (bits)
                v.D_i = random.uniform(2000, 5000) * (10**6)         # 2 - 5 Gcycles
            else:
                # 70% eMBB 任务 (更敏感于能耗，大数据轻计算)
                v.lambda_i = np.clip(np.random.normal(0.25, 0.1), 0.0, 0.5)
                v.U_i = random.uniform(5000, 20000) * 1024 * 8       # 5MB - 20MB (bits)
                v.D_i = random.uniform(500, 2000) * (10**6)          # 0.5 - 2 Gcycles
                
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
        # 观测空间重构：(6辆车, 6维特征)，彻底抛弃长条一维数组
        self.observation_space = spaces.Box(low=-1000, high=1e9, shape=(len(self.vehicles), 6), dtype=np.float32)

    def _apply_disaster_state(self):
        """确保 reset 之后，灾难状态依然生效 (导师指令)"""
        if self.is_avalanche_triggered:
            self.f_mec = 10 * (10**9) # 算力雪崩: 10GHz
            for es in self.edge_servers:
                es.cpu = self.f_mec
                
        if self.is_flood_triggered:
            for v in self.vehicles:
                # 真实物理修正：流量洪峰只影响背景数据(eMBB)，不影响紧急控制指令(URLLC)的大小
                if v.lambda_i < 0.5:
                    v.U_i *= 10
                    v.D_i *= 10

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

    def apply_kkt_projection(self, raw_actions):
        """KKT 算力投影 (贪心背包装箱修复版：拯救归零的 MEC 负载)"""
        capacity_limit = self.f_mec * self.T_max
        legal_actions = np.copy(raw_actions)
        
        # 提取想要去 MEC 的车辆索引
        offload_indices = [i for i, action in enumerate(raw_actions) if action == 1]
                
        # 核心修正：按 QoS (lambda_i) 降序排列！优先让最怕延迟的高优 URLLC 尝试接入
        sorted_indices = sorted(offload_indices, key=lambda idx: self.vehicles[idx].lambda_i, reverse=True)
        
        # 先将所有想去本地 MEC 的任务默认重定向到算力无限的云端 (Action 3)
        for idx in offload_indices:
            legal_actions[idx] = 3
            
        accepted_workload = 0
        for idx in sorted_indices:
            task_D = self.vehicles[idx].D_i
            # 贪心装箱：只要当前任务还能塞得下 MEC，就接受它！拒绝同归于尽！
            if accepted_workload + task_D <= capacity_limit:
                legal_actions[idx] = 1 # 准许接入 MEC
                accepted_workload += task_D
                
        return legal_actions

    def step(self, raw_actions):
        # 1. 第一步绝对是：获取合法的物理动作！
        legal_actions = self.apply_kkt_projection(raw_actions)
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
            
        total_latency = 0
        for i, v in enumerate(self.vehicles):
            h_t = self.calculate_fading_rate(v)
            # 大尺度路损: 127 + 30 * log10(d_km)
            d_m = max(1e-4, math.sqrt(v.coordinates[0]**2 + v.coordinates[1]**2))
            path_loss = 127 + 30 * math.log10(d_m / 1000.0)
            
            signal_power = self.p_i * h_t * (10**(-path_loss/10))
            # FDMA 带宽平分修正
            rate = (self.B / len(self.vehicles)) * math.log2(1 + signal_power / self.noise_power)
            
            U = v.U_i
            D = v.D_i
            
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
                T_trans = U / rate + U / self.r_eo + self.prop_eo
                T_exec = D / f_assigned[i]
                T_i = T_trans + T_exec
                E_i = self.p_i * (U / rate)
            elif legal_actions[i] == 3:
                T_trans = U / rate + U / self.r_ec + self.prop_ec
                T_exec = D / self.f_cloud
                T_i = T_trans + T_exec
                E_i = self.p_i * (U / rate)
                
            total_cost += v.lambda_i * T_i + v.mu_i * E_i
            total_latency += T_i
            
        # 5. Reward
        reward = - (total_cost / len(self.vehicles)) - penalty
        obs = []
        for v in self.vehicles:
            # 专属车辆级视角矩阵构建
            obs.append([
                v.coordinates[0] / 1000.0,
                v.coordinates[1] / 1000.0,
                v.D_i / 1e9,
                v.lambda_i,
                self.last_mec_load,
                np.clip(self.last_avg_reward / 10.0, -2.0, 0.0)  # 修改 3: 避免灵辝方差污染状态
            ])
            
        obs = np.array(obs, dtype=np.float32)
        done = self.simulator.schedule.steps >= 1000
        
        info = {
            "raw_actions": raw_actions.tolist(),
            "legal_actions": legal_actions.tolist(), # 输出对比
            "avg_cost": total_cost / len(self.vehicles),
            "avg_latency": total_latency / len(self.vehicles),
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
                np.clip(self.last_avg_reward / 10.0, -2.0, 0.0)  # 修改 3: 避免灵辝方差污染状态
            ])
            
        return np.array(obs, dtype=np.float32)

    def trigger_capacity_avalanche(self):
        """触发算力雪崩"""
        self.is_avalanche_triggered = True
        self._apply_disaster_state()
        print("\n[EVENT] CRITICAL: Capacity Avalanche Triggered! MEC Capacity -> 10GHz")

    def trigger_traffic_flood(self):
        """触发流量洪峰"""
        self.is_flood_triggered = True
        self._apply_disaster_state()
        print("\n[EVENT] CRITICAL: Traffic Flood Triggered! Task Workload x10")
