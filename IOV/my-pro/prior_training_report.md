# Prior Network Training and Inference Report

## 1. Training Environment

```text
GPU: NVIDIA GeForce RTX 5070
PyTorch: 2.11.0+cu128
CUDA available: True
Torch CUDA: 12.8
```

No additional PyTorch installation was required because the existing CUDA build supports the local GPU.

## 2. Dataset Split

The expert datasets were split with stratification.

```text
URLLC:
  train: 35000
  val:    7501
  test:   7499

eMBB:
  train: 35001
  val:    7501
  test:   7498
```

Stratification keys:

```text
URLLC: label_action + is_ood + urllc_outage
eMBB : label_action + is_ood + status_id
```

## 3. Saved Artifacts

```text
model/urllc_prior_best.pth
model/embb_prior_best.pth
model/urllc_feature_scaler.npz
model/embb_feature_scaler.npz
model/urllc_split_indices.json
model/embb_split_indices.json
model/urllc_prior_metrics.json
model/embb_prior_metrics.json
```

## 4. URLLCNet Test Results

```text
action_accuracy: 98.49%
action_macro_f1: 91.40%
delay_MAE: 0.0425 ms
violation_recall: 95.52%
violation_false_negative_rate: 4.48%
DROP_pred_rate: 0%

OOD action_accuracy: 96.72%
OOD violation_recall: 95.31%
OOD violation_false_negative_rate: 4.69%
```

Confusion matrix on held-out test set:

```text
rows=true, cols=pred

LOCAL       3697   30    5    0    0
LOCAL_MEC     51 3636   16    0    0
REMOTE_MEC     6    5   53    0    0
CLOUD          0    0    0    0    0
DROP           0    0    0    0    0
```

Violation detection:

```text
true violation samples: 67
TP: 64
FP: 14
FN: 3
TN: 7418
```

The supervised violation head is good, but URLLC should not run without a safety shield.

## 5. eMBBNet Test Results

```text
action_accuracy: 96.45%
action_macro_f1: 93.03%
delay_MAE: 0.0850 s
status_accuracy: 98.12%
status_macro_f1: 86.19%
utility_MAE: 0.0424
over_drop_rate: 0.20%

OOD action_accuracy: 94.90%
OOD action_macro_f1: 91.45%
OOD status_accuracy: 96.24%
OOD over_drop_rate: 0.62%
```

Action confusion matrix:

```text
rows=true, cols=pred

LOCAL       2976   34   35   13   14
LOCAL_MEC     26 2104   42   23    0
REMOTE_MEC    15   20 1645   17    0
CLOUD          4    9    7  457    1
DROP           4    0    1    1   50
```

Status confusion matrix:

```text
rows=true, cols=pred

REJECTED    42    0   14
SUCCESS      0 7043  108
DEGRADED     7   12  272
```

## 6. URLLC Safety Shield

The inference wrapper supports two modes:

```text
network-only
network + safety shield
```

Default safety thresholds:

```text
violation_threshold = 0.35
margin_threshold_ms = 0.30
```

Held-out test comparison:

```text
URLLC network-only:
  action_accuracy: 98.4931%
  policy_violation_rate: 0.9201%
  policy_violation_on_expert_feasible: 0.0269%
  DROP_pred_rate: 0%

URLLC with safety shield:
  action_accuracy: 98.6131%
  shield_rate: 1.2668%
  policy_violation_rate: 0.8935%
  policy_violation_on_expert_feasible: 0%
  DROP_pred_rate: 0%
```

Interpretation:

```text
The shield changes only about 1.27% of URLLC decisions.
It eliminates simulator-evaluated deadline violations on samples where the expert has a feasible action.
It slightly improves action accuracy.
```

Therefore, the current recommended URLLC inference mode is:

```text
URLLCNet + safety shield
```

## 7. Current Recommendation

Before residual RL, the system should use:

```text
URLLC:
  URLLCNet
  DROP mask
  safety shield enabled

eMBB:
  EMBBNet
  status-aware decision
  DROP allowed only when predicted/selected service is rejected
```

Residual RL should still wait until the prior inference wrapper is integrated with the simulator loop.
