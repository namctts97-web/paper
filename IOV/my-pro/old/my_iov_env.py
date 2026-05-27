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
        
        self.params = get_env_params_v2()
        
        # Absolute Contract Macro Parameters
        self.f_local = random.uniform(2.0, 5.0) * 10**9
        self.f_mec = random.uniform(30.0, 50.0) * 10**9
        self.f_offsite = random.uniform(100.0, 200.0) * 10**9
        self.f_cloud = self.params['f_cloud']
        self.p_i = self.params['p_ue']
        self.B = self.params['B']
        self.noise_power = self.params['sigma2']
        self.r_eo = self.params['r_eo']
        self.r_ec = self.params['r_ec']
        self.prop_eo = self.params['prop_eo']
        self.k_energy = self.params['k_energy']
        
        self.eta = 0.05
        self.V_max = 1.5
        
        self.simulator = Simulator(
            tick_duration=1,
            tick_unit="seconds",
            stopping_criterion=lambda model: model.schedule.steps == 1000,
            resource_management_algorithm=self._algorithm_wrapper,
        )
        self.simulator.initialize(input_file="d:/Enviroment/论文/IOV/EdgeAISIM/sample_dataset1.json")
        
        self.vehicles = User.all()
        self.edge_servers = EdgeServer.all()
        
        for es in self.edge_servers:
            es.cpu = self.f_mec
            
        urllc_indices = random.sample(range(len(self.vehicles)), 2)
        
        for i, v in enumerate(self.vehicles):
            v.velocity = random.uniform(10, 120) / 3.6
            v.h_t = 1.0 
            
            # Absolute Contract Generation
            if i in urllc_indices:
                v.task_type = 'URLLC'
                v.lambda_i = np.clip(np.random.normal(0.85, 0.05), 0.7, 1.0)
                v.U_i = random.uniform(4000, 16000)
                v.D_i = random.uniform(0.1, 1.0) * 10**6
                v.t_max = 0.003
            else:
                v.task_type = 'eMBB'
                v.lambda_i = np.clip(np.random.normal(0.25, 0.1), 0.0, 0.5)
                v.U_i = random.uniform(8.0, 16.0) * 10**6
                v.D_i = random.uniform(50.0, 100.0) * 10**9
                v.t_max = 1.5
                
            v.mu_i = 1.0 - v.lambda_i

        self.prev_actions = {v.id: 0 for v in self.vehicles}
        self.current_step_actions = None
        
        self.is_avalanche_triggered = False
        self.is_flood_triggered = False
        
        self.action_space = spaces.MultiDiscrete([4] * len(self.vehicles))
        # state: (x, y, D_ratio, lambda_i, mec_load, r_ec_ratio, flood_mult, I_mean_norm) -> shape 8
        self.observation_space = spaces.Box(low=-1000, high=1e9, shape=(len(self.vehicles), 8), dtype=np.float32)

    def _apply_disaster_state(self):
        if self.is_avalanche_triggered:
            if self.disaster_count == 1:
                self.f_mec = random.uniform(5.0, 10.0) * 10**9 
            elif self.disaster_count == 2:
                self.f_mec = random.uniform(1.0, 5.0) * 10**9 
            else:
                self.f_mec = 2.0 * 10**9
            for es in self.edge_servers:
                es.cpu = self.f_mec
                
        if getattr(self, 'is_flood_triggered', False):
            for v in self.vehicles:
                v.U_i *= 10.0
                v.D_i *= 10.0

    def _algorithm_wrapper(self, parameters):
        pass # Only a stub for edge_sim_py

    def calculate_fading_rate(self, vehicle):
        # 仅计算小尺度快衰落
        v_kmh = vehicle.velocity * 3.6
        rho = max(0.1, 0.95 - (v_kmh - 10) * (0.85 / 110))
        e_t = np.random.normal(0, 1)
        vehicle.h_t = rho * vehicle.h_t + np.sqrt(1 - rho**2) * e_t
        return max(1e-3, abs(vehicle.h_t))
        
    def _compute_cost(self, v, T, E):
        E_max = self.k_energy * (self.f_local**2) * v.D_i
        if E_max <= 0: E_max = 1e-10
        
        if v.task_type == 'URLLC':
            # Softplus-Smoothed Deadband: C^infty smoothness preventing Kink gradient tear
            beta = 10.0
            x = (T - 0.003) / 0.003
            # Prevent math.exp overflow for large x
            smooth_max = x if (beta * x) > 50 else (1.0 / beta) * math.log(1.0 + math.exp(beta * x))
            return 10.0 * math.asinh(smooth_max) + 0.01 * math.asinh(E / E_max)
        else: # eMBB
            # Isometric Arcsinh Manifold: Prevents Manifold Warping and guarantees pure Pareto
            return 1.0 * math.asinh(T / 1.5) + 1.0 * math.asinh(E / E_max)

    def greedy_knapsack_projection(self, raw_actions, estimated_rates):
        legal_actions = np.copy(raw_actions)
        mec_demand_cycles = 0
        for i, a in enumerate(raw_actions):
            if a == 1: mec_demand_cycles += self.vehicles[i].D_i
                
        max_mec_capacity = self.f_mec * 1.0
        
        if mec_demand_cycles > max_mec_capacity:
            def kkt_gradient(idx):
                v = self.vehicles[idx]
                est_rate = estimated_rates[idx]
                T_loc = v.D_i / self.f_local
                E_loc = self.k_energy * (self.f_local**2) * v.D_i
                Cost_loc = self._compute_cost(v, T_loc, E_loc)
                
                T_mec = v.U_i / est_rate + v.D_i / self.f_mec
                E_mec = self.p_i * (v.U_i / est_rate)
                Cost_mec = self._compute_cost(v, T_mec, E_mec)
                
                return (Cost_loc - Cost_mec) / v.D_i

            mec_vehicles = [i for i, a in enumerate(raw_actions) if a == 1]
            mec_vehicles.sort(key=kkt_gradient)
            
            current_demand = mec_demand_cycles
            for i in mec_vehicles:
                if current_demand <= max_mec_capacity: break
                legal_actions[i] = 0 
                current_demand -= self.vehicles[i].D_i
                
        return legal_actions

    def step(self, raw_actions):
        K_micro_samples = 100
        offloading_count = sum(1 for a in raw_actions if a > 0)
        K_RBs = self.params.get('K_RBs', 50)
        
        # 宏观干扰温度计算 (MFT)
        rho_rb = min(1.0, offloading_count / K_RBs)
        
        estimated_rates_macro = []
        all_R_samples = []
        macro_h_list = []
        
        for v in self.vehicles:
            macro_h_list.append(self.calculate_fading_rate(v))
            
        avg_macro_h = np.mean(macro_h_list)
        I_mean = rho_rb * self.p_i * avg_macro_h
        # 归一化用于状态感知 (基准化为纯噪声的倍数)
        self.I_mean_norm = I_mean / self.noise_power
        
        for i, v in enumerate(self.vehicles):
            macro_h = macro_h_list[i]
            scale = max(1e-4, macro_h)
            h_samples = np.random.rayleigh(scale=scale, size=K_micro_samples)
            d_m = max(1e-4, math.sqrt(v.coordinates[0]**2 + v.coordinates[1]**2))
            
            # 3GPP UMa Path Loss (与 env_v2_core 严格对齐，必须转为 km)
            pl_db = 128.1 + 37.6 * math.log10(d_m / 1000.0)
            # Log-Normal Shadowing
            shadowing_db = np.random.normal(0, 8)
            total_loss_db = pl_db + shadowing_db
            
            signal_power_samples = self.p_i * (h_samples**2) * (10**(-total_loss_db/10))
            
            # SINR 与 OFDMA 资源分配
            sinr_samples = signal_power_samples / (self.noise_power + I_mean)
            
            # MAC Preemption: Strict OFDMA slicing with priority weights
            N_U = sum(1 for j, vv in enumerate(self.vehicles) if vv.task_type == 'URLLC' and raw_actions[j] > 0)
            N_E = sum(1 for j, vv in enumerate(self.vehicles) if vv.task_type == 'eMBB' and raw_actions[j] > 0)
            
            # To compute hypothetical rate for non-offloading users, assume they join
            temp_N_U = N_U + (1 if v.task_type == 'URLLC' and raw_actions[i] == 0 else 0)
            temp_N_E = N_E + (1 if v.task_type == 'eMBB' and raw_actions[i] == 0 else 0)
            
            # Strict OFDMA slicing ensures sum(alloc_B) <= B
            W_U, W_E = 4.0, 1.0
            total_weight = max(1e-9, temp_N_U * W_U + temp_N_E * W_E)
            if v.task_type == 'URLLC':
                alloc_B = self.B * (W_U / total_weight)
            else:
                alloc_B = self.B * (W_E / total_weight)
                
            R_samples = alloc_B * np.log2(1 + sinr_samples)
            
            all_R_samples.append(R_samples)
            estimated_rates_macro.append(np.mean(R_samples))

        legal_actions = self.greedy_knapsack_projection(raw_actions, estimated_rates_macro)
        self.current_step_actions = legal_actions
        
        penalty = 0
        for i, v in enumerate(self.vehicles):
            penalty += self.eta * abs(legal_actions[i] - self.prev_actions[v.id])
            self.prev_actions[v.id] = legal_actions[i]
            
        self.simulator.step()
        
        total_cost, actual_load_cycles, total_latency, total_energy = 0, 0, 0, 0
        urllc_latency, urllc_energy, urllc_count = 0, 0, 0
        self.current_step_urllc_latencies = []
        embb_latency, embb_energy, embb_count = 0, 0, 0
        success_count = 0
        
        count_offsite = sum(1 for a in legal_actions if a == 2)
        count_cloud = sum(1 for a in legal_actions if a == 3)
        shared_r_eo = self.r_eo / count_offsite if count_offsite > 0 else self.r_eo
        shared_r_ec = self.r_ec / count_cloud if count_cloud > 0 else self.r_ec
        
        idx_mec = np.where(np.array(legal_actions) == 1)[0]
        idx_offsite = np.where(np.array(legal_actions) == 2)[0]
        f_assigned = np.zeros(len(self.vehicles))
        
        # Compute Preemption (严格时域切分物理逻辑):
        # URLLC 独占算力极短时间，随后 eMBB 瓜分剩余全部时域。因此双方在宏观视角下各自按照同类数量平分。
        if len(idx_mec) > 0:
            actual_N_U_mec = sum(1 for idx in idx_mec if self.vehicles[idx].task_type == 'URLLC')
            actual_N_E_mec = sum(1 for idx in idx_mec if self.vehicles[idx].task_type == 'eMBB')
            for idx in idx_mec:
                f_assigned[idx] = self.f_mec / max(1, actual_N_U_mec) if self.vehicles[idx].task_type == 'URLLC' else self.f_mec / max(1, actual_N_E_mec)
                
        if len(idx_offsite) > 0:
            actual_N_U_off = sum(1 for idx in idx_offsite if self.vehicles[idx].task_type == 'URLLC')
            actual_N_E_off = sum(1 for idx in idx_offsite if self.vehicles[idx].task_type == 'eMBB')
            for idx in idx_offsite:
                f_assigned[idx] = self.f_offsite / max(1, actual_N_U_off) if self.vehicles[idx].task_type == 'URLLC' else self.f_offsite / max(1, actual_N_E_off)
            
        reward_list = []
        
        for i, v in enumerate(self.vehicles):
            actual_offloading_count = sum(1 for a in legal_actions if a > 0)
            
            # 如果动作合法化导致越界被砍，重新计算宏观干扰
            rho_rb_actual = min(1.0, actual_offloading_count / K_RBs)
            I_mean_actual = rho_rb_actual * self.p_i * avg_macro_h
            
            # 重新缩放 R_samples (基于新的拥塞与 SINR)
            scale = max(1e-4, macro_h_list[i])
            h_samples = np.random.rayleigh(scale=scale, size=K_micro_samples)
            d_m = max(1e-4, math.sqrt(v.coordinates[0]**2 + v.coordinates[1]**2))
            pl_db = 128.1 + 37.6 * math.log10(d_m / 1000.0)
            shadowing_db = np.random.normal(0, 8)
            total_loss_db = pl_db + shadowing_db
            signal_power_samples = self.p_i * (h_samples**2) * (10**(-total_loss_db/10))
            
            sinr_samples_actual = signal_power_samples / (self.noise_power + I_mean_actual)
            
            # MAC Preemption (Actual Phase): Strict OFDMA slicing
            actual_N_U = sum(1 for j, vv in enumerate(self.vehicles) if vv.task_type == 'URLLC' and legal_actions[j] > 0)
            actual_N_E = sum(1 for j, vv in enumerate(self.vehicles) if vv.task_type == 'eMBB' and legal_actions[j] > 0)
            
            temp_actual_N_U = actual_N_U + (1 if v.task_type == 'URLLC' and legal_actions[i] == 0 else 0)
            temp_actual_N_E = actual_N_E + (1 if v.task_type == 'eMBB' and legal_actions[i] == 0 else 0)
            
            W_U, W_E = 4.0, 1.0
            total_weight_actual = max(1e-9, temp_actual_N_U * W_U + temp_actual_N_E * W_E)
            if v.task_type == 'URLLC':
                actual_alloc_B = self.B * (W_U / total_weight_actual)
            else:
                actual_alloc_B = self.B * (W_E / total_weight_actual)
            R_samples_actual = actual_alloc_B * np.log2(1 + sinr_samples_actual)
            
            if v.task_type == 'URLLC':
                theta = 0.05
                X = -theta * R_samples_actual
                c_eff = - (1.0 / theta) * (logsumexp(X) - math.log(K_micro_samples))
                rate = max(1e-3, c_eff)
            else:
                rate = max(1e-3, np.mean(R_samples_actual))
            
            U, D = v.U_i, v.D_i
            
            Cost_local = self._compute_cost(v, D / self.f_local, self.k_energy * (self.f_local**2) * D)
            
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
                # [物理灾难对齐] 严格同步外部生成的 Cloud Congestion 状态
                if getattr(self, 'force_cloud_congestion', False):
                    prop_ec_jitter = random.uniform(0.5, 1.0)
                else:
                    prop_ec_jitter = random.uniform(0.015, 0.030)
                T_trans = U / rate + U / shared_r_ec + prop_ec_jitter
                T_exec = D / self.f_cloud
                T_i = T_trans + T_exec
                E_i = self.p_i * (U / rate)
                
            Cost_actual = self._compute_cost(v, T_i, E_i)
            
            total_latency += T_i
            total_energy += E_i
            total_cost += Cost_actual
            
            if v.task_type == 'URLLC':
                urllc_latency += T_i
                self.current_step_urllc_latencies.append(T_i)
                urllc_energy += E_i
                urllc_count += 1
            else:
                embb_latency += T_i
                embb_energy += E_i
                embb_count += 1
            
            if T_i <= v.t_max: success_count += 1
                
            if raw_actions[i] > 0 and legal_actions[i] == 0:
                Cost_actual = Cost_local * self.V_max
                
            r_i = 1.0 - (Cost_actual / Cost_local)
            r_i = np.clip(r_i, 1.0 - self.V_max, 1.0)
            reward_list.append(r_i)
            
        reward = np.mean(reward_list) - penalty
        obs = []
        for v in self.vehicles:
            obs.append([
                v.coordinates[0] / 1000.0,
                v.coordinates[1] / 1000.0,
                v.D_i / (self.f_local * v.t_max),
                v.lambda_i,
                self.f_mec / (50.0 * 1e9),
                self.r_ec / (500.0 * 1e6), 
                1.0 if not getattr(self, 'is_flood_triggered', False) else 10.0,
                getattr(self, 'I_mean_norm', 0.0) # 物理层同频干扰指标
            ])
            
        obs = np.array(obs, dtype=np.float32)
        done = self.simulator.schedule.steps >= 1000
        
        info = {
            "raw_actions": raw_actions.tolist(),
            "legal_actions": legal_actions.tolist(),
            "avg_cost": total_cost / len(self.vehicles),
            "avg_latency": total_latency / len(self.vehicles),
            "avg_energy": total_energy / len(self.vehicles),
            "urllc_latency": urllc_latency / max(1, urllc_count),
            "raw_urllc_latencies": self.current_step_urllc_latencies,
            "embb_latency": embb_latency / max(1, embb_count),
            "urllc_energy": urllc_energy / max(1, urllc_count),
            "embb_energy": embb_energy / max(1, embb_count),
            "success_rate": success_count / len(self.vehicles),
            "mec_load": actual_load_cycles / (self.f_mec * 1.0)
        }
        
        return obs, reward, done, info

    def reset(self):
        avalanche_state = getattr(self, 'is_avalanche_triggered', False)
        flood_state = getattr(self, 'is_flood_triggered', False)
        self.__init__(ablation_mode=self.ablation_mode) 
        self.is_avalanche_triggered = avalanche_state
        self.is_flood_triggered = flood_state
        self._apply_disaster_state()
        self.current_step_urllc_latencies = []
        obs = []
        for v in self.vehicles:
            obs.append([
                v.coordinates[0] / 1000.0,
                v.coordinates[1] / 1000.0,
                v.D_i / (self.f_local * v.t_max),
                v.lambda_i,
                self.f_mec / (50.0 * 1e9), 
                self.r_ec / (500.0 * 1e6), 
                1.0 if not getattr(self, 'is_flood_triggered', False) else 10.0,
                0.0 # reset 时干扰为0
            ])
        return np.array(obs, dtype=np.float32)

    def trigger_capacity_avalanche(self):
        self.disaster_count += 1
        self.is_avalanche_triggered = True
        self._apply_disaster_state()
        print("\n[EVENT] CRITICAL: Capacity Avalanche Triggered!")

    def trigger_traffic_flood(self):
        self.is_flood_triggered = True
        self._apply_disaster_state()
        print("\n[EVENT] CRITICAL: Traffic Flood Triggered!")

    def recover_from_avalanche(self):
        self.is_avalanche_triggered = False
        self.f_mec = random.uniform(30.0, 50.0) * 10**9
        for es in self.edge_servers: es.cpu = self.f_mec
        print("\n[EVENT] RECOVERY: Capacity Avalanche Resolved!")

    def recover_from_flood(self):
        self.is_flood_triggered = False
        print("\n[EVENT] RECOVERY: Traffic Flood Resolved!")
