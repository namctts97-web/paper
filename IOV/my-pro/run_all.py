import subprocess
import sys
import time

algos = ['ours', 'ppo', 'prior']
seeds = [1, 2, 3, 4, 5]

processes = []

print("Launching 15 isolated environment processes...")

for algo in algos:
    for seed in seeds:
        cmd = [sys.executable, 'run_single_seed.py', '--algo', algo, '--seed', str(seed)]
        p = subprocess.Popen(cmd)
        processes.append((p, algo, seed))
        # Slight delay to prevent file collision during initialization
        time.sleep(0.5)

print(f"All {len(processes)} processes launched. Waiting for completion...")

failed = False
for p, algo, seed in processes:
    p.wait()
    if p.returncode != 0:
        print(f"[ERROR] Process {algo} seed {seed} failed with code {p.returncode}")
        failed = True

if failed:
    print("Some processes failed!")
    sys.exit(1)
else:
    print("All processes completed successfully!")
