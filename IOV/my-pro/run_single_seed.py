import argparse
import os
import time
import numpy as np
import torch
import collections

from my_iov_env import ResidualIoVEnv
from residual_ppo_agent import ResidualPPOAgent

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, required=True, choices=['ours', 'ppo', 'prior'])
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()

    # Isolate CPU to prevent thread starvation during parallel runs
    torch.set_num_threads(1)
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'

    set_seed(args.seed)
    env = ResidualIoVEnv()
    sample_state = env.reset()
    flat_state_dim = sample_state.flatten().shape[0]
    action_dim = len(env.vehicles) * 4

    agent = ResidualPPOAgent(state_dim=flat_state_dim, action_dim=action_dim, lr=1e-5, ablation_mode=args.algo)
    
    if args.algo == 'ours':
        agent.load_model('model/ours_converged.pth')
        agent.load_expert('model/prior_dnn_expert.pth')
    elif args.algo == 'ppo':
        agent.load_model('model/ppo_converged.pth')
    elif args.algo == 'prior':
        agent.load_expert('model/prior_dnn_expert.pth')

    global_urllc_window = collections.deque(maxlen=100000)
    
    max_episodes = 2000
    steps_per_episode = 200

    hist_cost = []
    hist_gate = []
    hist_urllc_violation = []

    for episode in range(1, max_episodes + 1):
        state = env.reset()
        ep_cost = 0
        ep_gate = []
        
        # OOD Triggers
        if episode == 500:
            env.trigger_capacity_avalanche()
        if episode == 800:
            env.is_avalanche_triggered = False
            env._apply_disaster_state()
        if episode == 1300:
            env.trigger_traffic_flood()
        if episode == 1600:
            env.is_flood_triggered = False
            env._apply_disaster_state()

        for t in range(steps_per_episode):
            if args.algo == 'prior':
                res = agent.select_action(state)
                action, logprob, val = res[0], res[1], res[2]
            else:
                res = agent.select_action(state)
                action, logprob, val = res[0], res[1], res[2]
                if args.algo == 'ours' and res[3] is not None:
                    ep_gate.append(res[3]) # gate is at index 3: delta_logits, gate, recon_error

            next_state, reward, done, info = env.step(action)
            
            if args.algo != 'prior':
                agent.store_transition((state, action, logprob, reward, done))
            
            state = next_state
            ep_cost += info['avg_cost']
            
            # Global Sliding Window for URLLC Latencies
            raw_latencies = info.get('raw_urllc_latencies', [])
            global_urllc_window.extend(raw_latencies)

            if done: break
        
        # Continual Learning Update (ONLY for PPO and OURS)
        # This causes the variance explosion the user demanded!
        if args.algo != 'prior':
            agent.update()

        avg_cost = ep_cost / steps_per_episode
        hist_cost.append(avg_cost)
        
        if len(ep_gate) > 0:
            hist_gate.append(np.mean(ep_gate))
        else:
            hist_gate.append(0.0)

        # Calculate True Violation Rate over Global Sliding Window
        if len(global_urllc_window) > 0:
            violations = sum(1 for lat in global_urllc_window if lat > 0.003)
            true_violation_rate = violations / len(global_urllc_window)
        else:
            true_violation_rate = 0.0
            
        hist_urllc_violation.append(true_violation_rate)

        if episode % 100 == 0:
            print(f"[{args.algo.upper()} | Seed {args.seed}] Ep {episode} | Cost: {avg_cost:.2f} | Gate: {hist_gate[-1]:.3f} | URLLC Viol: {true_violation_rate:.5f} (Window: {len(global_urllc_window)})")

    # Save to disk
    os.makedirs('results', exist_ok=True)
    np.save(f'results/{args.algo}_seed_{args.seed}_cost.npy', np.array(hist_cost))
    np.save(f'results/{args.algo}_seed_{args.seed}_gate.npy', np.array(hist_gate))
    np.save(f'results/{args.algo}_seed_{args.seed}_urllc.npy', np.array(hist_urllc_violation))

if __name__ == "__main__":
    main()
