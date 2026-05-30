# 网络设计 V1：专家模仿先验网络 + OOD 残差纠偏网络

## 1. 总体设计

本项目采用两阶段策略结构：

1. **任务专用专家模仿先验网络**
   - `URLLCNet`：面向 URLLC 任务的安全优先专家模仿网络。
   - `EMBBNet`：面向 eMBB 任务的效用优先专家模仿网络。

2. **OOD 残差纠偏网络**
   - `OODResidualPolicy`：只在 OOD 场景下学习对先验策略的修正。
   - 该模块应在 `URLLCNet` 和 `EMBBNet` 训练稳定后再进行训练。

当前第一阶段的推理流程为：

```text
state -> task router
      -> 如果 task_type == URLLC，进入 URLLCNet
      -> 如果 task_type == eMBB，进入 EMBBNet
      -> action mask / safety shield
```

第二阶段加入残差强化学习后：

```text
final_logits = prior_logits + alpha_ood * residual_logits
final_ratio  = clip(prior_ratio + alpha_ood * residual_ratio_delta)
```

其中 `alpha_ood` 在正常场景下接近 0，在 OOD 场景下增大。这样可以保证残差网络只做异常场景纠偏，不破坏正常时期已经学好的专家先验策略。

## 2. 文献依据

- **DNN 学习专家优化器输出**  
  Sun 等人的 *Learning to Optimize: Training Deep Neural Networks for Wireless Resource Management* 提出，可以把无线资源分配算法看成一个未知非线性映射，然后用 DNN 近似该映射，从而获得接近优化算法的性能和更快的实时推理速度。  
  https://arxiv.org/abs/1705.09412

- **URLLC 不能只优化平均时延**  
  Bennis、Debbah 和 Poor 的 *Ultra-Reliable and Low-Latency Wireless Communication: Tail, Risk and Scale* 强调，URLLC 需要关注尾部风险、可靠性和极端时延，而不能只看平均时延。  
  https://arxiv.org/abs/1801.01270

- **URLLC 边缘卸载需要队列和可靠性约束**  
  Liu、Bennis、Debbah 和 Poor 的 *Dynamic Task Offloading and Resource Allocation for Ultra-Reliable Low-Latency Edge Computing* 将 URLLC 边缘卸载建模为带队列和可靠性约束的任务卸载与资源分配问题。  
  https://doi.org/10.1109/TCOMM.2019.2898573

- **车联网边缘卸载和资源分配可用 DRL 建模**  
  Liu 等人的 *Deep Reinforcement Learning for Offloading and Resource Allocation in Vehicle Edge Computing and Networks* 说明，VEC 中的卸载和资源分配可以用深度强化学习处理。  
  https://doi.org/10.1109/TVT.2019.2935450

- **eMBB 和 URLLC 的优化目标不同**  
  Alsenwi 等人的 *Intelligent Resource Slicing for eMBB and URLLC Coexistence in 5G and Beyond* 将 eMBB 资源分配和 URLLC 调度分阶段处理，说明两类业务的目标存在明显差异。  
  https://arxiv.org/abs/2003.07651

- **残差强化学习适合在已有策略上做修正**  
  Johannink 等人的 *Residual Reinforcement Learning for Robot Control* 表明，可以将最终控制策略表示为基础控制器和学习残差策略的叠加。  
  https://arxiv.org/abs/1812.03201

- **多头辅助学习有通用深度学习依据**  
  Ruder 的 *An Overview of Multi-Task Learning in Deep Neural Networks* 总结了共享编码器和任务专用输出头的多任务学习结构。  
  https://arxiv.org/abs/1706.05098

## 3. 共享输入预处理

两个先验网络使用当前专家数据集中的状态特征。

推荐输入特征如下：

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

对数量级很大的正值特征使用 `log1p` 变换：

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

然后使用训练集均值和标准差进行标准化：

```text
x_norm = (x - mean_train) / std_train
```

注意：均值和标准差只能在训练集上计算，然后固定用于验证集、测试集和 OOD 测试集。

## 4. URLLCNet

### 4.1 网络目标

URLLCNet 不应该只学习 `label_action`。它需要同时预测：

```text
action_logits        5 类动作
cpu_ratio            CPU 分配比例
delay_ms             总时延
margin_ms            3ms - 总时延
violation_prob       P(delay > 3ms)
reliability          估计可靠性
```

URLLC 最严重的错误不是普通动作分类错误，而是：

```text
真实已经超过 3ms，但模型预测为安全
```

因此，URLLCNet 的核心目标是降低 violation 漏报。

### 4.2 网络结构

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

URLLC 训练和推理时使用动作 mask：

```text
允许动作：LOCAL, LOCAL_MEC, REMOTE_MEC, CLOUD
禁止动作：DROP
```

也就是说，URLLCNet 可以输出 5 类 logits，但 DROP 永远被 mask 掉。

### 4.3 损失函数

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

其中：

```text
CE 表示交叉熵
SmoothL1 用于连续值回归
BCE 用于 violation 二分类
```

`violation_prob` 的损失权重要高，因为 URLLC 最怕把危险样本误判成安全样本。

### 4.4 评估指标

URLLCNet 不应该只看 action accuracy。重点指标应该是：

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

其中最重要的是：

```text
violation_false_negative_rate
```

也就是“真实超 3ms 的样本里，有多少被模型误判为安全”。

## 5. EMBBNet

### 5.1 网络目标

EMBBNet 学习的是效用优先的服务决策。它需要预测：

```text
action_logits    5 类动作
cpu_ratio        CPU 分配比例
delay_sec        总服务时延
status_logits    SUCCESS / DEGRADED / REJECTED
utility_score    专家效用分数
```

eMBB 必须区分：

```text
SUCCESS   delay <= deadline
DEGRADED  deadline < delay <= tolerant_deadline
REJECTED  delay > tolerant_deadline 或 DROP
```

如果没有 `status_head`，模型很容易把“可降级服务”的样本误学成 DROP 或 SUCCESS。

### 5.2 网络结构

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

eMBBNet 比 URLLCNet 稍大一点，因为 eMBB 的动作分布和效用折中更复杂。

### 5.3 损失函数

```text
L_embb =
    CE(action_logits, label_action, class_weight)
  + 0.5 * SmoothL1(cpu_ratio_pred, label_cpu_ratio)
  + 1.0 * SmoothL1(delay_pred_sec, total_delay_sec)
  + 1.0 * CE(status_logits, status_id)
  + 0.5 * SmoothL1(utility_pred, utility_score)
```

规则：

```text
cpu_ratio loss 只在 label_action != DROP 的样本上计算
utility loss 只在 utility_score 有效时计算
status_id:
  1 -> SUCCESS
  2 -> DEGRADED
  0 -> REJECTED / DROP
```

### 5.4 评估指标

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

eMBBNet 特别要避免：

```text
过度 DROP
把 DEGRADED 误判成 SUCCESS
把 REJECTED 误判成可服务
```

## 6. Safety Shield

Safety Shield 不是残差强化学习模块，而是推理阶段的硬规则保护层。

URLLC 推理时：

```text
1. 永远 mask DROP。
2. 如果 violation_prob > tau_v 或 margin_pred_ms < tau_m：
      启动保守动作选择；
      若可调用候选动作评估器，则选择预测时延最小且合法的动作。
3. 最终永远不输出 DROP。
```

初始阈值建议：

```text
tau_v = 0.35
tau_m = 0.30 ms
```

eMBB 推理时：

```text
如果 status_head 预测 REJECTED，则允许 DROP。
如果 status_head 预测 DEGRADED，不强行 DROP，只要在 tolerant_deadline 内即可服务。
```

## 7. OODResidualPolicy

残差策略应该在两个先验网络训练完成后再做。

### 7.1 输入

```text
normalized state
prior_action_logits
prior_cpu_ratio
prior_delay_pred
prior_violation_prob 或 status_probs
is_ood
ood_type_id
```

### 7.2 输出

URLLC 和 eMBB 都可以使用类似输出：

```text
delta_action_logits  shape = [5]
delta_cpu_ratio      scalar in [-0.25, 0.25]
residual_gate        scalar in [0, 1]
```

最终策略为：

```text
alpha = residual_gate * is_ood_score
final_logits = prior_logits + alpha * delta_action_logits
final_ratio  = clip(prior_ratio + alpha * delta_cpu_ratio, 0.05, 1.0)
```

### 7.3 训练建议

不要先训练残差 RL。推荐顺序：

```text
1. 训练 URLLCNet
2. 训练 EMBBNet
3. 冻结两个 prior network
4. 在仿真环境中训练 OODResidualPolicy
5. URLLC 始终保留 safety shield
```

残差网络要惩罚正常场景下乱改 prior 决策：

```text
R = task_reward - lambda_change * 1[action != prior_action], when is_ood == 0
```

URLLC reward：

```text
R_urllc =
  - energy_weight * energy
  - delay_weight * delay_ms
  - huge_penalty * 1[delay > 3ms]
  - change_penalty_normal
```

eMBB reward：

```text
R_embb =
  throughput_reward
  - delay_penalty
  - energy_penalty
  - drop_penalty
  - change_penalty_normal
```

## 8. 实现顺序

推荐按以下顺序实现：

```text
Phase 1:
  FeaturePreprocessor
  URLLCNet
  EMBBNet
  supervised training / evaluation

Phase 2:
  safety shield inference wrapper
  normal / OOD split evaluation

Phase 3:
  OODResidualPolicy
  simulator-based residual RL
```

不要在两个监督先验网络还没有稳定前就开始 residual RL。否则 residual 学到的不是 OOD 纠偏，而是在修一个不稳定的基础策略，实验会很难解释。

## 9. 当前阶段结论

当前最应该实现的是：

```text
URLLCNet:
  安全优先
  action + cpu_ratio + delay + margin + violation + reliability

EMBBNet:
  效用优先
  action + cpu_ratio + delay + status + utility
```

这两个网络训练稳定后，再进入 OOD 残差强化学习阶段。
