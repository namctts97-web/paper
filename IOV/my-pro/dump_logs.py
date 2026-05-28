import numpy as np
import os

d = np.load('model/plot_costs.npy', allow_pickle=True).item()
artifact_dir = r"C:\Users\zrd\.gemini\antigravity\brain\7c7da191-bd3e-409f-9eca-b865cb995f94"
out_path = os.path.join(artifact_dir, "step_by_step_cost_log.md")

with open(out_path, 'w', encoding='utf-8') as f:
    f.write("# 逐步代价数据详细记录 (Steps 1-200)\n\n")
    f.write("此表格记录了 4 个基线在每个时间步（Step）的 Average Cost 原始数据。\n\n")
    f.write("> [!NOTE]\n")
    f.write("> **Step 1-100**: 平稳期 (Normal)\n")
    f.write("> **Step 101-200**: OOD灾难期 (Traffic Flood)\n\n")
    f.write("| Step | Ours (Residual) | Prior DNN | Vanilla PPO | Oracle (SA) |\n")
    f.write("|---|---|---|---|---|\n")
    
    for i in range(200):
        step = i + 1
        ours_c = d['ours'][i]
        prior_c = d['prior'][i]
        ppo_c = d['ppo'][i]
        oracle_c = d['oracle'][i]
        
        if step == 101:
            f.write(f"| **{step} (OOD)** | **{ours_c:.2f}** | **{prior_c:.2f}** | **{ppo_c:.2f}** | **{oracle_c:.2f}** |\n")
        else:
            f.write(f"| {step} | {ours_c:.2f} | {prior_c:.2f} | {ppo_c:.2f} | {oracle_c:.2f} |\n")

print("Done. File generated.")
