import numpy as np

def analyze():
    print("Loading expert data...")
    data = np.load('data/expert_dataset.npy', allow_pickle=True)
    states = np.concatenate([d['state'] for d in data]) # Shape: (600000, 8)
    actions = np.concatenate([d['action'] for d in data]) # Shape: (600000,)
    
    # Feature map: x, y, D_ratio, lambda_i, mec_load, r_ec_ratio, flood_mult, I_mean_norm
    lambdas = states[:, 3]
    I_mean = states[:, 7]
    D_ratio = states[:, 2]
    
    urllc_mask = lambdas > 0.6
    embb_mask = lambdas <= 0.6
    
    print("\n" + "="*50)
    print(f"Total Samples: {len(actions)}")
    print(f"URLLC Samples: {np.sum(urllc_mask)}")
    print(f"eMBB Samples: {np.sum(embb_mask)}")
    print("="*50)
    
    def print_dist(mask, name):
        subset_actions = actions[mask]
        subset_I_mean = I_mean[mask]
        subset_D = D_ratio[mask]
        
        unique, counts = np.unique(subset_actions, return_counts=True)
        print(f"\n[{name}] Action Distribution:")
        for a, c in zip(unique, counts):
            a_mask = subset_actions == a
            avg_I = np.mean(subset_I_mean[a_mask])
            avg_D = np.mean(subset_D[a_mask])
            print(f"  Action {a}: {c/len(subset_actions)*100:>6.2f}% | Avg I_mean_norm: {avg_I:.4f} | Avg D_ratio: {avg_D:.4f}")
            
    print_dist(urllc_mask, "URLLC")
    print_dist(embb_mask, "eMBB")

if __name__ == "__main__":
    analyze()
