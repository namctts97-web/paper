import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('image', exist_ok=True)

def smooth(x, weight=0.6):
    """只允许对均值使用轻度平滑(<=0.6)，绝对禁止前方差平滑"""
    res = []
    last = x[0]
    for val in x:
        smoothed = last * weight + (1 - weight) * val
        res.append(smoothed)
        last = smoothed
    return np.array(res)

def load_data(algo, metric):
    data = []
    for seed in range(1, 6):
        path = f'results/{algo}_seed_{seed}_{metric}.npy'
        if os.path.exists(path):
            data.append(np.load(path))
    if len(data) == 0:
        raise FileNotFoundError(f"No data found for {algo} {metric}")
    return np.array(data)

def get_stats(data):
    mean_raw = np.mean(data, axis=0)
    # The user strictly commanded: "方差平滑系数必须死死锁在 0.0"
    std_raw = np.std(data, axis=0) 
    
    # 仅对均值进行克制的轻度平滑以辨识主线趋势，暴露出全部的方差震荡
    mean_smooth = smooth(mean_raw, 0.6)
    return mean_smooth, std_raw

try:
    ours_cost = load_data('ours', 'cost')
    ppo_cost = load_data('ppo', 'cost')
    prior_cost = load_data('prior', 'cost')

    ours_urllc = load_data('ours', 'urllc')
    ppo_urllc = load_data('ppo', 'urllc')
    
    # 取单一 seed 的 gate 用来画图，防止跨种子平均导致阶跃变缓
    ours_gate_single = load_data('ours', 'gate')[0] 
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

x = np.arange(1, 2001)

# =====================================================================
# 1. Main Convergence Curve (True Variance Shadow)
# =====================================================================
ours_m, ours_s = get_stats(ours_cost)
ppo_m, ppo_s = get_stats(ppo_cost)
prior_m, prior_s = get_stats(prior_cost)

plt.figure(figsize=(12, 6))
plt.plot(x, ours_m, 'r-', label='OURS (Dual-Brain Res-Gating)', linewidth=2)
# 真实的爆炸式扩散方差
plt.fill_between(x, ours_m - ours_s, ours_m + ours_s, color='red', alpha=0.25)

plt.plot(x, ppo_m, 'gray', label='Standard PPO (Context-Aware)', linewidth=2)
plt.fill_between(x, ppo_m - ppo_s, ppo_m + ppo_s, color='gray', alpha=0.25)

plt.plot(x, prior_m, 'b--', label='Ablation (EWC Prior, No Gating)', linewidth=2)
plt.fill_between(x, prior_m - prior_s, prior_m + prior_s, color='blue', alpha=0.1)

plt.axvspan(500, 800, color='red', alpha=0.1, label='OOD: Capacity Avalanche')
plt.axvspan(1300, 1600, color='blue', alpha=0.1, label='OOD: Traffic Flood')

plt.title("System Convergence & OOD Resilience (True Variance Explosion)")
plt.xlabel("Episodes")
plt.ylabel("System Cost")
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('image/training_curve.png', dpi=300)

# =====================================================================
# 2. URLLC 99.99% Tail Latency Violation Rate (Global Sliding Window)
# =====================================================================
ours_u_m, ours_u_s = get_stats(ours_urllc)
ppo_u_m, ppo_u_s = get_stats(ppo_urllc)

plt.figure(figsize=(12, 4))
# Statistical Warm-up Phase
plt.axvspan(0, 250, color='lightgray', alpha=0.5, label='Statistical Warm-up Phase (Jitter)')

plt.plot(x, ours_u_m, 'r-', label='OURS True Tail Violation', linewidth=2)
plt.fill_between(x, ours_u_m - ours_u_s, ours_u_m + ours_u_s, color='red', alpha=0.2)

plt.plot(x, ppo_u_m, 'k-', label='PPO True Tail Violation', linewidth=2)
plt.fill_between(x, ppo_u_m - ppo_u_s, ppo_u_m + ppo_u_s, color='gray', alpha=0.2)

plt.axhline(y=0.01, color='green', linestyle='--', linewidth=2, label='3GPP 99.99% Reliability Deadline')

plt.axvspan(500, 800, color='red', alpha=0.1)
plt.axvspan(1300, 1600, color='blue', alpha=0.1)

plt.title("URLLC 99.99% Tail Violation Rate (100,000-Sample Global Window)")
plt.xlabel("Episodes")
plt.ylabel("Violation Rate")
# Cap the y-limit logically, exposing deep fading floor
plt.ylim(-0.005, np.max(ppo_u_m[300:]) + 0.05) 
plt.legend(loc='upper left')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('image/urllc_violation_rate.png', dpi=300)

# =====================================================================
# 3. G(t) Gating Radar ECG (Step-Response Integrity)
# =====================================================================
# Absolutely NO smoothing, showing the raw single-seed trace
plt.figure(figsize=(12, 3))
plt.plot(x, ours_gate_single, color='#00ff00', linewidth=1.2, label='Raw G(t) Activation') 
plt.fill_between(x, 0, ours_gate_single, color='#00ff00', alpha=0.1)
plt.axvspan(500, 800, color='red', alpha=0.2, label='Avalanche (OOD)')
plt.axvspan(1300, 1600, color='blue', alpha=0.2, label='Flood (OOD)')

plt.title("G(t) Gating Radar ECG (Single Seed, No Smoothing, Absolute Step-Response)")
plt.xlabel("Episodes")
plt.ylabel("G(t) Activation")
plt.ylim(-0.1, 1.1)
ax = plt.gca()
ax.set_facecolor('#111111')
plt.legend(loc='upper right', facecolor='#111111', labelcolor='white')
plt.grid(True, color='#333333', linestyle='--')
plt.tight_layout()
plt.savefig('image/gate_radar.png', dpi=300)

print("\n[OK] All true physics figures rendered successfully.")
