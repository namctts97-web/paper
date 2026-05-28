import gym
from gym import spaces
import numpy as np
import math
import random

class IoVEnvV3(gym.Env):
    """
    IoV_Env_V3:
    1. Strict Decoupling of S_macro and S_micro
    2. Strictly conserved OFDMA MAC allocation
    3. Zonal Interference Temperature Density (ZITD) modeled strictly as Inter-Cell Interference (ICI)
    4. Heterogeneous CVaR Barrier Cost for URLLC
    """
    def __init__(self, num_vehicles=20, grid_size=1000):
        super(IoVEnvV3, self).__init__()
        self.num_vehicles = num_vehicles
        self.grid_size = grid_size
        
        # System constraints & physical parameters
        self.B = 20e6  # 20 MHz Total Bandwidth
        
        # Noise power: Thermal noise is -174 dBm/Hz.
        # Linear scale (Watts): 10^((-174 - 30) / 10) * B
        self.noise_power = 10 ** ((-174.0 - 30.0) / 10.0) * self.B
        
        self.p_ue_dbm = 23 # 23 dBm UE transmit power
        self.p_ue = 10 ** ((self.p_ue_dbm - 30) / 10) # Watts
        
        self.f_mec_base = 50.0 * 10**9 # 50 GHz
        self.f_cloud = 200.0 * 10**9
        self.r_eo_base = 1.0 * 10**9 # 1 Gbps edge-to-edge
        self.r_ec_base = 500.0 * 10**6 # 500 Mbps edge-to-cloud
        
        # MAC Layer OFDMA Priority Weights
        self.W_U = 4.0
        self.W_E = 1.0
        
        # Action space: 0=Local, 1=MEC, 2=Offsite, 3=Cloud
        self.action_space = spaces.MultiDiscrete([4] * self.num_vehicles)
        
        # State:
        # S_macro: [ZITD_norm, MEC_Load, Core_Congestion, Surge_Mult] (4)
        # S_micro for each vehicle: [PL_linear, h_t, U_i_MB, D_i_Gcycles, is_urllc, t_max, f_local_GHz] (7 per vehicle)
        # Total state dim = 4 + 7 * num_vehicles
        self.state_dim = 4 + 7 * self.num_vehicles
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)
        
        # OOD exogenous triggers
        self.is_compute_avalanche = False
        self.is_traffic_flood = False
        self.is_core_congestion = False
        
        self.vehicles = []
        self.time_step = 0
        
    def reset(self):
        self.time_step = 0
        self.vehicles = []
        
        # Apply Exogenous OOD State
        self.f_mec = 5.0 * 10**9 if self.is_compute_avalanche else self.f_mec_base
        self.surge_mult = 10.0 if self.is_traffic_flood else 1.0
        self.r_ec = 50.0 * 10**6 if self.is_core_congestion else self.r_ec_base
            
        for i in range(self.num_vehicles):
            # URLLC probability approx 20%
            is_urllc = 1 if random.random() < 0.2 else 0
            
            x = random.uniform(0, self.grid_size)
            y = random.uniform(0, self.grid_size)
            
            h_t = np.random.rayleigh(1.0)
            
            if is_urllc:
                U_i = random.uniform(4000, 16000) * self.surge_mult # bits
                D_i = random.uniform(0.1e6, 1.0e6) * self.surge_mult # CPU cycles
                t_max = 0.003 # 3ms
            else:
                U_i = random.uniform(8.0e6, 16.0e6) * self.surge_mult # bits
                D_i = random.uniform(50.0e9, 100.0e9) * self.surge_mult # CPU cycles
                t_max = 1.5 # 1.5s
                
            f_local = random.uniform(2.0, 5.0) * 10**9
            velocity = random.uniform(10, 120) / 3.6 # m/s
            
            v = {
                'id': i, 'x': x, 'y': y, 'h_t': h_t,
                'is_urllc': is_urllc, 'U_i': U_i, 'D_i': D_i, 
                't_max': t_max, 'f_local': f_local,
                'velocity': velocity
            }
            self.vehicles.append(v)
            
        return self._get_obs()

    def _get_obs(self):
        # ==========================================
        # 1. Macro State (S_macro) Calculation
        # ==========================================
        
        # ZITD (Zonal Interference Temperature Density) 
        # Using a Gaussian Kernel to calculate spatial congestion indicator
        sigma = 200.0 # meters spreading radius
        zonal_I = 0.0
        for i in range(self.num_vehicles):
            I_i = 0.0
            for j in range(self.num_vehicles):
                if i != j:
                    d_sq = (self.vehicles[i]['x'] - self.vehicles[j]['x'])**2 + (self.vehicles[i]['y'] - self.vehicles[j]['y'])**2
                    I_i += self.p_ue * math.exp(-d_sq / (2 * sigma**2))
            zonal_I += I_i
            
        # ZITD_norm is a global macroscopic indicator (average zonal congestion normalized by noise)
        # Apply a generic macro path loss factor (e.g. 1e-13) to convert raw Tx power to received background interference
        generic_path_loss = 1e-13
        zonal_I_received = (zonal_I / self.num_vehicles) * generic_path_loss
        ZITD_norm = zonal_I_received / self.noise_power
        
        # MEC Load computation (total demand vs capacity)
        mec_demand = sum(v['D_i'] for v in self.vehicles)
        mec_load = mec_demand / self.f_mec
        
        # Core Congestion Ratio
        core_cong = self.r_ec_base / self.r_ec
        
        S_macro = [ZITD_norm, mec_load, core_cong, self.surge_mult]
        
        # ==========================================
        # 2. Micro State (S_micro) Calculation
        # ==========================================
        S_micro = []
        for v in self.vehicles:
            # Distance to center MEC (assume MEC at center of grid)
            d_i = max(10.0, math.sqrt((v['x'] - self.grid_size/2)**2 + (v['y'] - self.grid_size/2)**2))
            
            # Path loss 3GPP UMa
            pl_db = 128.1 + 37.6 * math.log10(d_i / 1000.0) + np.random.normal(0, 8)
            pl_linear = 10**(-pl_db/10)
            
            # Normalize micro features for NN stability
            S_micro.extend([
                pl_linear * 1e10,           # Rescaled PL
                v['h_t'],                   # Small scale fading
                v['U_i'] / 1e6,             # MB
                v['D_i'] / 1e9,             # Gcycles
                float(v['is_urllc']),       # Indicator
                v['t_max'],                 # Deadline
                v['f_local'] / 1e9          # GHz
            ])
            
        self.current_ZITD = ZITD_norm
        obs = np.array(S_macro + S_micro, dtype=np.float32)
        return obs

    def step(self, actions):
        # 1. Physical State Update (Micro movements and CSI Markov chain)
        for v in self.vehicles:
            v['x'] = max(0, min(self.grid_size, v['x'] + v['velocity'] * random.uniform(-1, 1)))
            v['y'] = max(0, min(self.grid_size, v['y'] + v['velocity'] * random.uniform(-1, 1)))
            
            v_kmh = v['velocity'] * 3.6
            rho = max(0.1, 0.95 - (v_kmh - 10) * (0.85 / 110))
            e_t = np.random.normal(0, 1)
            v['h_t'] = rho * v['h_t'] + math.sqrt(1 - rho**2) * e_t

        # 2. Action Execution & Strict OFDMA Resource Conservation
        # actions: 0=Local, 1=MEC, 2=Offsite, 3=Cloud
        offloading_indices = [i for i, a in enumerate(actions) if a > 0]
        N_U = sum(1 for i in offloading_indices if self.vehicles[i]['is_urllc'] == 1)
        N_E = sum(1 for i in offloading_indices if self.vehicles[i]['is_urllc'] == 0)
        
        # Strict OFDMA weighting logic. Total mapped bandwidth strictly equals self.B if active.
        total_weight = max(1e-9, N_U * self.W_U + N_E * self.W_E)
        alloc_B_URLLC = self.B * (self.W_U / total_weight) if N_U > 0 else 0.0
        alloc_B_eMBB = self.B * (self.W_E / total_weight) if N_E > 0 else 0.0
        
        # 3. Inter-Cell Interference (ICI) formulation
        # 物理学修正：ICI 是经过严重路径损耗后的到达功率，通常与热噪声在同一量级。
        # 我们利用 ZITD_norm（区域干扰密度）作为系数，建立环境拥塞对物理层信道的实质影响。
        # 假设基准状态下，ICI 约为热噪声的 1 到 2 倍；发生空间数据洪峰（OOD）时，倍数上升。
        ICI_scaling_factor = 1.0 + 0.1 * self.current_ZITD
        ICI = self.noise_power * ICI_scaling_factor
        
        # 4. Heterogeneous Cost Computation
        total_cost = 0.0
        
        for i, v in enumerate(self.vehicles):
            a_i = actions[i]
            
            d_i = max(10.0, math.sqrt((v['x'] - self.grid_size/2)**2 + (v['y'] - self.grid_size/2)**2))
            pl_db = 128.1 + 37.6 * math.log10(d_i / 1000.0) + np.random.normal(0, 8)
            pl_linear = 10**(-pl_db/10)
            
            signal_power = self.p_ue * (v['h_t']**2) * pl_linear
            
            if a_i > 0:
                alloc_B = alloc_B_URLLC if v['is_urllc'] == 1 else alloc_B_eMBB
                # SINR strictly uses Noise + Inter-Cell Interference (ICI), no Intra-Cell Interference
                sinr = signal_power / (self.noise_power + ICI)
                rate = alloc_B * math.log2(1 + sinr)
            else:
                rate = 1e-9 # Fallback to prevent DivZero
                
            rate = max(1.0, rate)
            
            T_trans, T_exec = 0.0, 0.0
            E_trans, E_exec = 0.0, 0.0
            k_energy = 1e-28
            
            E_max = k_energy * (v['f_local']**2) * v['D_i']
            if E_max <= 0: E_max = 1e-10
            
            if a_i == 0: # Local
                T_exec = v['D_i'] / v['f_local']
                E_exec = k_energy * (v['f_local']**2) * v['D_i']
            elif a_i == 1: # MEC
                T_trans = v['U_i'] / rate
                actual_N_MEC = sum(1 for a in actions if a == 1)
                f_assigned = self.f_mec / max(1, actual_N_MEC)
                T_exec = v['D_i'] / f_assigned
                E_trans = self.p_ue * T_trans
            elif a_i == 2: # Offsite
                T_trans = v['U_i'] / rate + v['U_i'] / self.r_eo_base + 0.005
                actual_N_Off = sum(1 for a in actions if a == 2)
                f_assigned = self.f_mec_base / max(1, actual_N_Off)
                T_exec = v['D_i'] / f_assigned
                E_trans = self.p_ue * (v['U_i'] / rate)
            elif a_i == 3: # Cloud
                T_trans = v['U_i'] / rate + v['U_i'] / self.r_ec + random.uniform(0.015, 0.030)
                T_exec = v['D_i'] / self.f_cloud
                E_trans = self.p_ue * (v['U_i'] / rate)
                
            T_i = T_trans + T_exec
            E_i = E_trans + E_exec
            
            # 5. CVaR Barrier and Softplus Costs
            if v['is_urllc'] == 1:
                alpha, beta, gamma = 10.0, 10.0, 0.01
                x = (T_i - v['t_max']) / v['t_max']
                # Softplus smoothed CVaR barrier
                smooth_max = x if (beta * x) > 50 else (1.0 / beta) * math.log(1.0 + math.exp(beta * x))
                cost = alpha * math.asinh(smooth_max) + gamma * (E_i / E_max)
            else:
                # Standard eMBB metric
                cost = 1.0 * (T_i / v['t_max']) + 1.0 * (E_i / E_max)
                
            total_cost += cost
            
        # Reward is negative cost (can be rescaled by RL agent)
        reward = -total_cost / self.num_vehicles
        self.time_step += 1
        done = self.time_step >= 100
        
        info = {
            "N_U": N_U, "N_E": N_E,
            "Allocated_B_URLLC": alloc_B_URLLC,
            "Allocated_B_eMBB": alloc_B_eMBB,
            "ICI": ICI,
            "ZITD_norm": self.current_ZITD,
            "avg_cost": total_cost / self.num_vehicles
        }
        
        return self._get_obs(), reward, done, info
        
    def evaluate_actions(self, actions):
        """
        无状态评估接口 (Stateless Evaluation)
        供外部启发式求解算法调用，仅返回该动作组合下的总代价（Cost）。
        不改变内部任何状态（坐标、衰落），保证启发式算法的 Fitness 函数与环境 100% 同构。
        """
        offloading_indices = [i for i, a in enumerate(actions) if a > 0]
        N_U = sum(1 for i in offloading_indices if self.vehicles[i]['is_urllc'] == 1)
        N_E = sum(1 for i in offloading_indices if self.vehicles[i]['is_urllc'] == 0)
        
        total_weight = max(1e-9, N_U * self.W_U + N_E * self.W_E)
        alloc_B_URLLC = self.B * (self.W_U / total_weight) if N_U > 0 else 0.0
        alloc_B_eMBB = self.B * (self.W_E / total_weight) if N_E > 0 else 0.0
        
        ICI_scaling_factor = 1.0 + 0.1 * getattr(self, 'current_ZITD', 1.0)
        ICI = self.noise_power * ICI_scaling_factor
        
        total_cost = 0.0
        
        for i, v in enumerate(self.vehicles):
            a_i = actions[i]
            
            d_i = max(10.0, math.sqrt((v['x'] - self.grid_size/2)**2 + (v['y'] - self.grid_size/2)**2))
            # 为了无状态纯粹性，这里忽略正态分布随机阴影，或者使用均值，保证评估稳定性
            pl_db = 128.1 + 37.6 * math.log10(d_i / 1000.0) 
            pl_linear = 10**(-pl_db/10)
            
            signal_power = self.p_ue * (v['h_t']**2) * pl_linear
            
            if a_i > 0:
                alloc_B = alloc_B_URLLC if v['is_urllc'] == 1 else alloc_B_eMBB
                sinr = signal_power / (self.noise_power + ICI)
                rate = alloc_B * math.log2(1 + sinr)
            else:
                rate = 1e-9 
                
            rate = max(1.0, rate)
            
            T_trans, T_exec = 0.0, 0.0
            E_trans, E_exec = 0.0, 0.0
            k_energy = 1e-28
            
            E_max = k_energy * (v['f_local']**2) * v['D_i']
            if E_max <= 0: E_max = 1e-10
            
            if a_i == 0: 
                T_exec = v['D_i'] / v['f_local']
                E_exec = k_energy * (v['f_local']**2) * v['D_i']
            elif a_i == 1: 
                T_trans = v['U_i'] / rate
                actual_N_MEC = sum(1 for a in actions if a == 1)
                f_assigned = self.f_mec / max(1, actual_N_MEC)
                T_exec = v['D_i'] / f_assigned
                E_trans = self.p_ue * T_trans
            elif a_i == 2: 
                T_trans = v['U_i'] / rate + v['U_i'] / self.r_eo_base + 0.005
                actual_N_Off = sum(1 for a in actions if a == 2)
                f_assigned = self.f_mec_base / max(1, actual_N_Off)
                T_exec = v['D_i'] / f_assigned
                E_trans = self.p_ue * (v['U_i'] / rate)
            elif a_i == 3: 
                T_trans = v['U_i'] / rate + v['U_i'] / self.r_ec + 0.02 # 均值
                T_exec = v['D_i'] / self.f_cloud
                E_trans = self.p_ue * (v['U_i'] / rate)
                
            T_i = T_trans + T_exec
            E_i = E_trans + E_exec
            
            if v['is_urllc'] == 1:
                alpha, beta, gamma = 10.0, 10.0, 0.01
                x = (T_i - v['t_max']) / v['t_max']
                smooth_max = x if (beta * x) > 50 else (1.0 / beta) * math.log(1.0 + math.exp(beta * x))
                cost = alpha * math.asinh(smooth_max) + gamma * (E_i / E_max)
            else:
                cost = 1.0 * (T_i / v['t_max']) + 1.0 * (E_i / E_max)
                
            total_cost += cost
            
        return total_cost

    def trigger_avalanche(self):
        self.is_compute_avalanche = True
    def trigger_flood(self):
        self.is_traffic_flood = True
    def trigger_core_congestion(self):
        self.is_core_congestion = True
    def reset_disasters(self):
        self.is_compute_avalanche = False
        self.is_traffic_flood = False
        self.is_core_congestion = False
