# Network Design V1: Expert Imitation Prior + OOD Residual Correction

## 1. Design Decision

This project should use a two-stage policy architecture:

1. **Task-specific expert imitation prior networks**
   - `URLLCNet`: safety-first imitation network for URLLC tasks.
   - `EMBBNet`: utility-aware imitation network for eMBB tasks.

2. **OOD residual correction network**
   - `OODResidualPolicy`: only learns corrections on top of the prior policy under OOD states.
   - This module should be trained after `URLLCNet` and `EMBBNet` are stable.

In other words, the first implementation target is:

```text
state -> task router
      -> URLLCNet if task_type == URLLC
      -> EMBBNet  if task_type == eMBB
      -> safety shield / action mask
```

The residual RL stage is:

```text
final_logits = prior_logits + alpha_ood * residual_logits
final_ratio  = clip(prior_ratio + alpha_ood * residual_ratio_delta)
```

where `alpha_ood` is close to `0` in normal states and larger in OOD states.

## 2. Literature Support

- **Learning expert optimizers with DNNs**: Sun et al., *Learning to Optimize: Training Deep Neural Networks for Wireless Resource Management*, argues that a wireless resource allocation algorithm can be treated as an unknown nonlinear mapping and approximated by a DNN for fast real-time inference: https://arxiv.org/abs/1705.09412
- **URLLC is not average-delay optimization**: Bennis, Debbah, and Poor, *Ultra-Reliable and Low-Latency Wireless Communication: Tail, Risk and Scale*, motivates delay-tail/risk-aware design for URLLC instead of only optimizing average metrics: https://arxiv.org/abs/1801.01270
- **URLLC MEC offloading needs probabilistic/queue-aware constraints**: Liu, Bennis, Debbah, and Poor, *Dynamic Task Offloading and Resource Allocation for Ultra-Reliable Low-Latency Edge Computing*, models URLLC edge offloading with queue/reliability constraints: https://doi.org/10.1109/TCOMM.2019.2898573
- **VEC offloading and resource allocation with learning is established**: Liu et al., *Deep Reinforcement Learning for Offloading and Resource Allocation in Vehicle Edge Computing and Networks*, IEEE TVT 2019: https://doi.org/10.1109/TVT.2019.2935450
- **eMBB/URLLC objectives differ**: Alsenwi et al., *Intelligent Resource Slicing for eMBB and URLLC Coexistence in 5G and Beyond*, separates eMBB resource allocation and URLLC scheduling phases in an optimization-aided DRL framework: https://arxiv.org/abs/2003.07651
- **Residual learning over a base controller is valid**: Johannink et al., *Residual Reinforcement Learning for Robot Control*, shows a final policy can be formed by superposing a conventional/base controller and a learned residual controller: https://arxiv.org/abs/1812.03201
- **Multi-head auxiliary learning is standard**: Ruder, *An Overview of Multi-Task Learning in Deep Neural Networks*, summarizes shared encoders with task-specific heads: https://arxiv.org/abs/1706.05098

## 3. Shared Input Preprocessing

Both prior networks use the current expert dataset state features.

Recommended raw features:

```text
task_type_id
data_size_bits
result_size_bits
required_cycles
deadline_sec
tolerant_deadline_sec
vehicle_speed_mps
vehicle_acc_mps2
distance_to_rsu_m
lane_id
sinr_linear
downlink_sinr_linear
channel_gain_linear
tx_power_w
bandwidth_hz
uplink_rate_shannon_bps
downlink_rate_shannon_bps
mac_delay_sec
cpu_local_hz
cpu_lmec_hz
cpu_rmec_hz
cpu_cloud_hz
queue_local_len
queue_lmec_len
queue_rmec_len
queue_cloud_len
workload_local_cycles
workload_lmec_cycles
workload_rmec_cycles
workload_cloud_cycles
util_local
util_lmec
util_rmec
util_cloud
backhaul_l2r_delay_sec
backhaul_r2c_delay_sec
is_ood
ood_type_id
```

Use `log1p` for large positive-scale features:

```text
data_size_bits
result_size_bits
required_cycles
sinr_linear
downlink_sinr_linear
channel_gain_linear
uplink_rate_shannon_bps
downlink_rate_shannon_bps
cpu_*_hz
workload_*_cycles
```

Then apply training-set standardization:

```text
x_norm = (x - mean_train) / std_train
```

Important: fit normalization only on training split, then reuse it for validation/test/OOD.

## 4. URLLCNet

### 4.1 Goal

URLLCNet should not only imitate `label_action`. It should predict:

```text
action_logits        -> 5 actions
cpu_ratio            -> continuous scalar
delay_ms             -> total delay
margin_ms            -> 3ms - total delay
violation_prob       -> P(delay > 3ms)
reliability          -> estimated reliability
```

URLLC's most serious error is not wrong classification in general. It is:

```text
false safe prediction = true violation, but model predicts safe
```

### 4.2 Architecture

```text
Input(d)
  -> Linear(d, 256)
  -> LayerNorm
  -> SiLU
  -> Dropout(0.05)
  -> Linear(256, 256)
  -> LayerNorm
  -> SiLU
  -> Dropout(0.05)
  -> Linear(256, 128)
  -> LayerNorm
  -> SiLU

Heads:
  action_head      Linear(128, 5)
  cpu_ratio_head   Linear(128, 1) + sigmoid
  delay_head       Linear(128, 1) + softplus
  margin_head      Linear(128, 1)
  violation_head   Linear(128, 1) + sigmoid
  reliability_head Linear(128, 1) + sigmoid
```

Action mask during training/inference:

```text
URLLC allowed: LOCAL, LOCAL_MEC, REMOTE_MEC, CLOUD
URLLC masked:  DROP
```

### 4.3 Loss

```text
L_urllc =
    CE(masked_action_logits, label_action, class_weight)
  + 0.5 * SmoothL1(cpu_ratio_pred, label_cpu_ratio)
  + 1.0 * SmoothL1(delay_pred_ms, total_delay_ms)
  + 1.0 * SmoothL1(margin_pred_ms, deadline_margin_ms)
  + 4.0 * BCE(violation_prob, deadline_violation)
  + 0.5 * SmoothL1(reliability_pred, estimated_reliability)
  + 10.0 * illegal_drop_penalty
```

Compute `cpu_ratio` loss only for non-DROP samples. URLLC has no DROP labels, but keeping this rule makes the shared training code simpler.

### 4.4 Metrics

Priority metrics:

```text
violation_false_negative_rate
violation_recall
DROP_pred_rate
delay_MAE_ms
P95_delay_error_ms
action_accuracy
OOD_violation_recall
OOD_action_accuracy
```

The most important one is `violation_false_negative_rate`.

## 5. EMBBNet

### 5.1 Goal

EMBBNet should learn utility-aware service decisions:

```text
action_logits    -> 5 actions
cpu_ratio        -> continuous scalar
delay_sec        -> total service delay
status_logits    -> SUCCESS / DEGRADED / REJECTED
utility_score    -> expert utility
```

eMBB must distinguish:

```text
SUCCESS   delay <= deadline
DEGRADED  deadline < delay <= tolerant_deadline
REJECTED  delay > tolerant_deadline or DROP
```

### 5.2 Architecture

```text
Input(d)
  -> Linear(d, 384)
  -> LayerNorm
  -> SiLU
  -> Dropout(0.10)
  -> Linear(384, 256)
  -> LayerNorm
  -> SiLU
  -> Dropout(0.10)
  -> Linear(256, 128)
  -> LayerNorm
  -> SiLU

Heads:
  action_head    Linear(128, 5)
  cpu_ratio_head Linear(128, 1) + sigmoid
  delay_head     Linear(128, 1) + softplus
  status_head    Linear(128, 3)
  utility_head   Linear(128, 1)
```

### 5.3 Loss

```text
L_embb =
    CE(action_logits, label_action, class_weight)
  + 0.5 * SmoothL1(cpu_ratio_pred, label_cpu_ratio)
  + 1.0 * SmoothL1(delay_pred_sec, total_delay_sec)
  + 1.0 * CE(status_logits, status_id)
  + 0.5 * SmoothL1(utility_pred, utility_score)
```

Rules:

```text
cpu_ratio loss only applies when label_action != DROP
utility loss only applies when utility_score is finite
status_id mapping:
  1 -> SUCCESS
  2 -> DEGRADED
  0 -> REJECTED / DROP
```

### 5.4 Metrics

```text
macro_F1
action_accuracy
DROP_precision
DROP_recall
SUCCESS_F1
DEGRADED_F1
REJECTED_F1
delay_MAE_sec
utility_MAE
OOD_macro_F1
over_drop_rate
degraded_as_success_rate
```

## 6. Safety Shield

The safety shield is not the residual RL module. It is a deterministic inference guard.

For URLLC:

```text
1. Mask DROP.
2. If violation_prob > tau_v or margin_pred_ms < tau_m:
      use model-predicted delay/risk to choose the safest legal action;
      if candidate evaluator is available, choose the legal action with minimum predicted delay.
3. Never emit DROP.
```

Initial thresholds:

```text
tau_v = 0.35
tau_m = 0.30 ms
```

For eMBB:

```text
If status_head predicts REJECTED, allow DROP.
If status is DEGRADED, do not force DROP; keep the selected action if tolerant_delay is satisfied.
```

## 7. OODResidualPolicy

Residual policy should be trained after the two prior networks.

### 7.1 Input

```text
normalized state
prior_action_logits
prior_cpu_ratio
prior_delay_pred
prior_violation_prob / status_probs
is_ood
ood_type_id
```

### 7.2 Output

For URLLC:

```text
delta_action_logits: shape [5]
delta_cpu_ratio:    scalar in [-0.25, 0.25]
residual_gate:      scalar in [0, 1]
```

For eMBB:

```text
delta_action_logits: shape [5]
delta_cpu_ratio:    scalar in [-0.25, 0.25]
residual_gate:      scalar in [0, 1]
```

Final policy:

```text
alpha = residual_gate * is_ood_score
final_logits = prior_logits + alpha * delta_action_logits
final_ratio  = clip(prior_ratio + alpha * delta_cpu_ratio, 0.05, 1.0)
```

### 7.3 Training Recommendation

Do not train residual RL first.

Order:

```text
1. Train URLLCNet on expert data.
2. Train EMBBNet on expert data.
3. Freeze prior networks.
4. Train OODResidualPolicy in simulator.
5. Keep safety shield active for URLLC.
```

The residual reward should penalize changing normal decisions:

```text
R = task_reward - lambda_change * 1[action != prior_action] when is_ood == 0
```

For URLLC:

```text
R_urllc =
  - energy_weight * energy
  - delay_weight * delay_ms
  - huge_penalty * 1[delay > 3ms]
  - change_penalty_normal
```

For eMBB:

```text
R_embb =
  throughput_reward
  - delay_penalty
  - energy_penalty
  - drop_penalty
  - change_penalty_normal
```

## 8. Final Recommendation

Implement in this order:

```text
Phase 1:
  Feature preprocessor
  URLLCNet
  EMBBNet
  supervised training/evaluation

Phase 2:
  safety shield inference wrapper
  normal/OOD split evaluation

Phase 3:
  OODResidualPolicy
  simulator-based residual RL
```

Do not start residual RL until the two supervised prior networks are stable on both normal and OOD test sets.
