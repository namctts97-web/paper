import gym
from gym import spaces
import numpy as np
import math
import random
from scipy.special import logsumexp

class IoVEnvV3(gym.Env):
    """
    IoV_Env_V3: (Fully Restored Physics & Mathematics from V2)
    1. 100MHz Bandwidth, 2.0W Power, 10000G Cloud Compute
    2. KKT Greedy Knapsack Projection
    3. Asinh Cost Manifold Mapping
    4. Advantage Normalized Reward Baseline
    5. CVaR Micro Sampling for URLLC
    """
    def __init__(self, num_vehicles=10, grid_size=1000):
        super(IoVEnvV3, self).__init__()
        self.num_vehicles = num_vehicles
        self.grid_size = grid_size
        
        # System constraints & physical parameters (RESTORED)
        self.B = 100 * 1e6  # 100 MHz Total Bandwidth
        
        self.noise_power = 2 * 1e-13 # Thermal noise + base interference
        self.p_ue = 2.0 # 2.0W (33 dBm)
        
        self.f_mec_base = 50.0 * 10**9 # 50 GHz
        self.f_cloud = 10000.0 * 10**9 # 10000 GHz
        self.r_eo_base = 500.0 * 1e6 # 500 Mbps
        self.r_ec_base = 500.0 * 1e6 # 500 Mbps
        self.k_energy = 1e-27
        self.V_max = 3
        
        # MAC Layer OFDMA Priority Weights
        self.W_U = 4.0
        self.W_E = 1.0
        
        # Action space: 0=Local, 1=MEC, 2=Offsite, 3=Cloud
        self.action_space = spaces.MultiDiscrete([4] * self.num_vehicles)
        
        self.state_dim = 4 + 7 * self.num_vehicles
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)
        
        self.is_compute_avalanche = False
        self.is_traffic_flood = False
        self.is_core_congestion = False
        
        self.vehicles = []
        self.time_step = 0
        self.prev_actions = [0] * self.num_vehicles
        
    def reset(self):
        self.time_step = 0
        self.vehicles = []
        self.prev_actions = [0] * self.num_vehicles
        
        # Apply Exogenous OOD State
        base_f_mec = random.uniform(30.0, 50.0) * 10**9
        self.f_mec = random.uniform(2.0, 10.0) * 10**9 if self.is_compute_avalanche else base_f_mec
        self.surge_mult = 10.0 if self.is_traffic_flood else 1.0
        self.r_ec = random.uniform(10.0, 50.0) * 1e6 if self.is_core_congestion else self.r_ec_base
            
        for i in range(self.num_vehicles):
            is_urllc = 1 if random.random() < 0.2 else 0
            
            x = random.uniform(0, self.grid_size)
            y = random.uniform(0, self.grid_size)
            
            h_t = np.random.rayleigh(1.0)
            
            if is_urllc:
                U_i = random.uniform(4000, 16000) * self.surge_mult
                D_i = random.uniform(0.1e6, 1.0e6) * self.surge_mult
                t_max = 0.003
            else:
                U_i = random.uniform(8.0e6, 16.0e6) * self.surge_mult
                D_i = random.uniform(50.0e9, 100.0e9) * self.surge_mult
                t_max = 1.5
                
            f_local = random.uniform(2.0, 5.0) * 10**9
            velocity = random.uniform(10, 120) / 3.6
            
            v = {
                'id': i, 'x': x, 'y': y, 'h_t': h_t,
                'is_urllc': is_urllc, 'U_i': U_i, 'D_i': D_i, 
                't_max': t_max, 'f_local': f_local,
                'velocity': velocity
            }
            self.vehicles.append(v)
            
        self._generate_step_noises()
        return self._get_obs()

    def _generate_step_noises(self):
        for v in self.vehicles:
            v['current_shadowing'] = np.random.normal(0, 8)
            v['current_jitter'] = random.uniform(0.015, 0.030)

    def _get_obs(self):
        sigma = 200.0 
        zonal_I = 0.0
        for i in range(self.num_vehicles):
            I_i = 0.0
            for j in range(self.num_vehicles):
                if i != j:
                    d_sq = (self.vehicles[i]['x'] - self.vehicles[j]['x'])**2 + (self.vehicles[i]['y'] - self.vehicles[j]['y'])**2
                    I_i += self.p_ue * math.exp(-d_sq / (2 * sigma**2))
            zonal_I += I_i
            
        generic_path_loss = 1e-13
        zonal_I_received = (zonal_I / self.num_vehicles) * generic_path_loss
        ZITD_norm = zonal_I_received / self.noise_power
        
        mec_demand = sum(v['D_i'] for v in self.vehicles)
        mec_load = mec_demand / self.f_mec
        core_cong = self.r_ec_base / self.r_ec
        
        S_macro = [ZITD_norm, mec_load, core_cong, self.surge_mult]
        
        S_micro = []
        for v in self.vehicles:
            d_i = max(10.0, math.sqrt((v['x'] - self.grid_size/2)**2 + (v['y'] - self.grid_size/2)**2))
            pl_db = 128.1 + 37.6 * math.log10(d_i / 1000.0) + v['current_shadowing']
            pl_linear = 10**(-pl_db/10)
            
            S_micro.extend([
                pl_linear * 1e10,           
                v['h_t'],                   
                v['U_i'] / 1e6,             
                v['D_i'] / 1e9,             
                float(v['is_urllc']),       
                v['t_max'],                 
                v['f_local'] / 1e9          
            ])
            
        self.current_ZITD = ZITD_norm
        obs = np.array(S_macro + S_micro, dtype=np.float32)
        return obs

    def _compute_cost(self, v, T, E):
        E_max = self.k_energy * (v['f_local']**2) * v['D_i']
        if E_max <= 0: E_max = 1e-10
        
        if v['is_urllc'] == 1:
            beta = 10.0
            x = (T - v['t_max']) / v['t_max']
            smooth_max = x if (beta * x) > 50 else (1.0 / beta) * math.log(1.0 + math.exp(beta * x))
            return 10.0 * math.asinh(smooth_max) + 0.01 * math.asinh(E / E_max)
        else: 
            return 1.0 * math.asinh(T / v['t_max']) + 1.0 * math.asinh(E / E_max)

    def greedy_knapsack_projection(self, raw_actions, estimated_rates):
        legal_actions = np.copy(raw_actions)
        mec_demand_cycles = sum(self.vehicles[i]['D_i'] for i, a in enumerate(raw_actions) if a == 1)
        max_mec_capacity = self.f_mec * 1.0 # 1 second window
        
        if mec_demand_cycles > max_mec_capacity:
            def kkt_gradient(idx):
                v = self.vehicles[idx]
                est_rate = estimated_rates[idx]
                
                T_loc = v['D_i'] / v['f_local']
                E_loc = self.k_energy * (v['f_local']**2) * v['D_i']
                Cost_loc = self._compute_cost(v, T_loc, E_loc)
                
                T_mec = v['U_i'] / est_rate + v['D_i'] / self.f_mec
                E_mec = self.p_ue * (v['U_i'] / est_rate)
                Cost_mec = self._compute_cost(v, T_mec, E_mec)
                
                return (Cost_loc - Cost_mec) / v['D_i']

            mec_vehicles = [i for i, a in enumerate(raw_actions) if a == 1]
            mec_vehicles.sort(key=kkt_gradient)
            
            current_demand = mec_demand_cycles
            for i in mec_vehicles:
                if current_demand <= max_mec_capacity: break
                legal_actions[i] = 0 
                current_demand -= self.vehicles[i]['D_i']
                
        return legal_actions

    def _get_rate_samples(self, actions):
        K_micro_samples = 100
        offloading_indices = [i for i, a in enumerate(actions) if a > 0]
        N_U = sum(1 for i in offloading_indices if self.vehicles[i]['is_urllc'] == 1)
        N_E = sum(1 for i in offloading_indices if self.vehicles[i]['is_urllc'] == 0)
        
        total_weight = max(1e-9, N_U * self.W_U + N_E * self.W_E)
        alloc_B_URLLC = self.B * (self.W_U / total_weight) if N_U > 0 else 0.0
        alloc_B_eMBB = self.B * (self.W_E / total_weight) if N_E > 0 else 0.0
        
        ICI_scaling_factor = 1.0 + 0.1 * getattr(self, 'current_ZITD', 1.0)
        ICI = self.noise_power * ICI_scaling_factor
        
        rate_samples_list = []
        for i, v in enumerate(self.vehicles):
            scale = max(1e-4, v['h_t'])
            h_samples = np.random.rayleigh(scale=scale, size=K_micro_samples)
            
            d_i = max(10.0, math.sqrt((v['x'] - self.grid_size/2)**2 + (v['y'] - self.grid_size/2)**2))
            pl_db = 128.1 + 37.6 * math.log10(d_i / 1000.0) + v['current_shadowing']
            pl_linear = 10**(-pl_db/10)
            
            signal_power_samples = self.p_ue * (h_samples**2) * pl_linear
            sinr_samples = signal_power_samples / (self.noise_power + ICI)
            
            alloc_B = alloc_B_URLLC if v['is_urllc'] == 1 else alloc_B_eMBB
            R_samples = alloc_B * np.log2(1 + sinr_samples)
            R_samples = np.maximum(R_samples, 1e-9)
            rate_samples_list.append(R_samples)
            
        return rate_samples_list

    def _compute_actual_cost(self, v, a_i, rate_samples, actual_N_MEC, actual_N_Off):
        if a_i > 0:
            if v['is_urllc'] == 1:
                theta = 0.5
                X = -theta * rate_samples
                c_eff = - (1.0 / theta) * (logsumexp(X) - math.log(len(rate_samples)))
            else:
                c_eff = np.mean(rate_samples)
            c_eff = max(1.0, c_eff)
        else:
            c_eff = 1e-9
            
        T_trans, T_exec = 0.0, 0.0
        E_trans, E_exec = 0.0, 0.0
        
        if a_i == 0: 
            T_exec = v['D_i'] / v['f_local']
            E_exec = self.k_energy * (v['f_local']**2) * v['D_i']
        elif a_i == 1: 
            T_trans = v['U_i'] / c_eff
            f_assigned = self.f_mec / max(1, actual_N_MEC)
            T_exec = v['D_i'] / f_assigned
            E_trans = self.p_ue * T_trans
        elif a_i == 2: 
            T_trans = v['U_i'] / c_eff + v['U_i'] / self.r_eo_base + 0.005
            f_assigned = self.f_mec_base / max(1, actual_N_Off) 
            T_exec = v['D_i'] / f_assigned
            E_trans = self.p_ue * (v['U_i'] / c_eff)
        elif a_i == 3: 
            T_trans = v['U_i'] / c_eff + v['U_i'] / self.r_ec + v['current_jitter']
            T_exec = v['D_i'] / self.f_cloud
            E_trans = self.p_ue * (v['U_i'] / c_eff)
            
        return self._compute_cost(v, T_trans + T_exec, E_trans + E_exec)

    def step(self, raw_actions):
        rate_samples_list_raw = self._get_rate_samples(raw_actions)
        estimated_rates_macro = [np.mean(r) for r in rate_samples_list_raw]
        
        # KKT Protection
        legal_actions = self.greedy_knapsack_projection(raw_actions, estimated_rates_macro)
        
        # Recompute physical parameters with legal actions
        rate_samples_list_legal = self._get_rate_samples(legal_actions)
        
        actual_N_MEC = sum(1 for a in legal_actions if a == 1)
        actual_N_Off = sum(1 for a in legal_actions if a == 2)
        
        reward_list = []
        total_cost = 0.0
        penalty = 0.0
        eta = 0.1
        
        for i, v in enumerate(self.vehicles):
            Cost_actual = self._compute_actual_cost(v, legal_actions[i], rate_samples_list_legal[i], actual_N_MEC, actual_N_Off)
            Cost_local = self._compute_actual_cost(v, 0, rate_samples_list_legal[i], actual_N_MEC, actual_N_Off)
            
            total_cost += Cost_actual
            
            # Penalize action projection
            if raw_actions[i] > 0 and legal_actions[i] == 0:
                Cost_actual = Cost_local * self.V_max
                
            r_i = 1.0 - (Cost_actual / Cost_local)
            r_i = np.clip(r_i, 1.0 - self.V_max, 1.0)
            reward_list.append(r_i)
            
            penalty += eta * abs(legal_actions[i] - self.prev_actions[i])
            self.prev_actions[i] = legal_actions[i]
            
        reward = np.mean(reward_list) - penalty
        self.time_step += 1
        done = self.time_step >= 100
        
        for v in self.vehicles:
            v['x'] = max(0, min(self.grid_size, v['x'] + v['velocity'] * random.uniform(-1, 1)))
            v['y'] = max(0, min(self.grid_size, v['y'] + v['velocity'] * random.uniform(-1, 1)))
            
            v_kmh = v['velocity'] * 3.6
            rho = max(0.1, 0.95 - (v_kmh - 10) * (0.85 / 110))
            e_t = np.random.normal(0, 1)
            v['h_t'] = rho * v['h_t'] + math.sqrt(1 - rho**2) * e_t
            
        self._generate_step_noises()
        
        info = {
            "N_U": sum(1 for i, a in enumerate(legal_actions) if a > 0 and self.vehicles[i]['is_urllc'] == 1),
            "N_E": sum(1 for i, a in enumerate(legal_actions) if a > 0 and self.vehicles[i]['is_urllc'] == 0),
            "avg_cost": total_cost / self.num_vehicles,
            "legal_actions": legal_actions
        }
        
        return self._get_obs(), reward, done, info
        
    def evaluate_actions(self, actions):
        rate_samples_list = self._get_rate_samples(actions)
        actual_N_MEC = sum(1 for a in actions if a == 1)
        actual_N_Off = sum(1 for a in actions if a == 2)
        
        total_cost = 0.0
        for i, v in enumerate(self.vehicles):
            cost = self._compute_actual_cost(v, actions[i], rate_samples_list[i], actual_N_MEC, actual_N_Off)
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
