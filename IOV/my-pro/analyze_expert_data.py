import numpy as np
import os

def analyze_dataset(file_path='data/expert_dataset.npy'):
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return
        
    print("=" * 70)
    print("         10万条金牌专家数据集统计物理审计报告 (量纲修正版)         ")
    print("=" * 70)
    
    # 1. 加载数据
    data = np.load(file_path, allow_pickle=True)
    n_samples = len(data)
    print(f"[*] 样本总量: {n_samples} 条")
    
    # 2. 解析结构
    first_sample = data[0]
    state_shape = first_sample['state'].shape
    action_shape = first_sample['action'].shape
    n_tasks = action_shape[0]
    
    print(f"[*] 单条观测 State 维度: {state_shape} | 物理意义: {n_tasks}台车辆 * 4特征 (x, y, D_i, lambda_i)")
    print(f"[*] 单条动作 Action 维度: {action_shape} | 物理意义: {n_tasks}台车辆的决策 (0:本地, 1:本地MEC, 2:远程MEC, 3:云)")
    
    # 3. 统计指标提取
    all_states = np.array([item['state'] for item in data])
    all_actions = np.array([item['action'] for item in data])
    
    # 3.1 卸载分布统计
    # 统计四种状态的占比
    option_names = ["0:本地执行", "1:本地MEC", "2:远程MEC", "3:云端服务器"]
    
    print("\n--- 1. 动作决策层多分类统计 (Action Layer) ---")
    for op_idx, op_name in enumerate(option_names):
        op_rate = np.mean(all_actions == op_idx)
        print(f"[*] 全局平均 选择【{op_name}】比例: {op_rate:.2%}")
        
    print("\n[*] 各车辆独立决策分布情况:")
    for v_idx in range(n_tasks):
        print(f"    - 车辆 {v_idx+1}:")
        for op_idx, op_name in enumerate(option_names):
            v_op_rate = np.mean(all_actions[:, v_idx] == op_idx)
            print(f"        * 选择【{op_name}】: {v_op_rate:.2%}")
        
    # 常见卸载配置文件 (Action Profiles)
    action_profiles, counts = np.unique(all_actions, axis=0, return_counts=True)
    sorted_idx = np.argsort(counts)[::-1]
    
    print("\n[*] 最常见的 Top 5 卸载决策组合 (Action Profiles):")
    for i in range(min(5, len(sorted_idx))):
        idx = sorted_idx[i]
        profile = action_profiles[idx]
        count = counts[idx]
        percentage = count / n_samples
        print(f"    - Top {i+1}: 决策组合 {profile.tolist()} | 频次 = {count:6d} 次 | 占比 = {percentage:.2%}")
        
    # 3.2 物理特征层统计 (State Layer)
    # 恢复物理量纲: 根据 env_config / json 数据集，坐标的原始单位为千米 (km)
    # obs 中：坐标已除以 1000.0, D_i 已除以 1e9, lambda_i 原样
    # 因此乘以 1000.0 恢复出的就是原始公里数
    x_coords = all_states[:, 0::4] * 1000.0
    y_coords = all_states[:, 1::4] * 1000.0
    D_cycles = all_states[:, 2::4] * 1e9
    lambda_weights = all_states[:, 3::4]
    
    distances = np.sqrt(x_coords**2 + y_coords**2) # 单位: km
    
    print("\n--- 2. 状态特征层统计 (State Layer) ---")
    print(f"[*] 车辆到 MEC (0,0) 的直线距离分布 (千米 / 米):")
    print(f"    - 极小值: {np.min(distances):.2f} km ({np.min(distances)*1000:.1f} m)")
    print(f"    - 极大值: {np.max(distances):.2f} km ({np.max(distances)*1000:.1f} m)")
    print(f"    - 均值: {np.mean(distances):.2f} km ({np.mean(distances)*1000:.1f} m)")
    print(f"[*] 任务计算量 D_i 异构分布 (Megacycles):")
    print(f"    - 极小值: {np.min(D_cycles)/1e6:.2f} Mc | 极大值: {np.max(D_cycles)/1e6:.2f} Mc | 均值: {np.mean(D_cycles)/1e6:.2f} Mc")
    
    # 3.3 物理-决策关联分析 (Cross-Layer Correlation)
    # 分析距离近 (d < 3.5 km) vs 距离远 (d > 5.0 km) 对决策的影响
    near_mask = distances < 3.5
    far_mask = distances > 5.0
    
    # 卸载率这里定义为：选择卸载到 MEC/云端 (即 action > 0) 的比例
    near_offload = np.mean(all_actions[near_mask] > 0) if np.any(near_mask) else 0.0
    far_offload = np.mean(all_actions[far_mask] > 0) if np.any(far_mask) else 0.0
    
    # 分析计算量极大 (D > 1200Mc) vs 计算量极小 (D < 800Mc) 的卸载比例
    heavy_mask = D_cycles > 1200 * 1e6
    light_mask = D_cycles < 800 * 1e6
    
    heavy_offload = np.mean(all_actions[heavy_mask] > 0) if np.any(heavy_mask) else 0.0
    light_offload = np.mean(all_actions[light_mask] > 0) if np.any(light_mask) else 0.0
    
    print("\n--- 3. 专家算法物理规律审计 (Physics Consistency) ---")
    print(f"[*] 距离敏感度分析 (卸载定义为 Action > 0):")
    print(f"    - 距离 MEC 近邻组 (d < 3.5 km) 整体卸载比例: {near_offload:.2%}")
    print(f"    - 距离 MEC 遥远组 (d > 5.0 km) 整体卸载比例: {far_offload:.2%}")
    print(f"    - 学术结论: {'符合香农公式，近处信道优，卸载率较高' if near_offload >= far_offload else '警告: 不符合香农公式，数据异常！'}")
    
    print(f"[*] 负载敏感度分析 (卸载定义为 Action > 0):")
    print(f"    - 重计算量任务组 (D > 1200Mc) 整体卸载比例: {heavy_offload:.2%}")
    print(f"    - 轻计算量任务组 (D < 800Mc)  整体卸载比例: {light_offload:.2%}")
    print(f"    - 学术结论: {'符合计算能耗博弈，重任务由于本地算力瓶颈更倾向于卸载' if heavy_offload >= light_offload else '警告: 不符合算力分配常识，数据异常！'}")
    
    # 3.4 异构 QoS 敏感度分析 (V2 专属)
    urllc_mask = lambda_weights > 0.7
    embb_mask = lambda_weights < 0.5
    
    urllc_actions = all_actions[urllc_mask]
    embb_actions = all_actions[embb_mask]
    
    print("\n--- 4. 异构 QoS 双峰偏好专项分析 (V2 专属) ---")
    if len(urllc_actions) > 0 and len(embb_actions) > 0:
        print(f"[*] URLLC 任务 (极度怕延迟, lambda > 0.7) 决策偏好:")
        for op_idx, op_name in enumerate(option_names):
            print(f"    - 选择【{op_name}】比例: {np.mean(urllc_actions == op_idx):.2%}")
            
        print(f"\n[*] eMBB 任务 (极度怕费电, lambda < 0.5) 决策偏好:")
        for op_idx, op_name in enumerate(option_names):
            print(f"    - 选择【{op_name}】比例: {np.mean(embb_actions == op_idx):.2%}")
            
        urllc_local = np.mean(urllc_actions == 0)
        embb_local = np.mean(embb_actions == 0)
        print(f"\n[*] QoS 学术定性结论: {'完全符合预期！eMBB 因为怕费电 (发射功率 2.0W) 更多留在本地慢慢算，URLLC 拼命卸载追求低延迟。' if embb_local > urllc_local else '警告：QoS 分流失效！'}")
    
    print("=" * 70)

if __name__ == "__main__":
    analyze_dataset()
