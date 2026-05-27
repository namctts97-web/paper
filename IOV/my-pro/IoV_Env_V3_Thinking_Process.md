# IoV Env V3 核心理论与架构蓝图

本文档用于记录 IoV_Env_V3 从底层数学推导到系统架构蓝图的演进过程，以及每次重大理论更新与重构动作的思考日志。

## 1. 状态空间 (State Space) 严格解耦定义

为了完美契合 Residual RL 的双脑架构（OOD 异常检测门控 + 细粒度策略微调），我们将环境返回的状态空间在物理层面上严格划分为 $S_{macro}$ 和 $S_{micro}$。

### 1.1 宏观外生状态 $S_{macro}$ (专供 Autoencoder 捕捉 OOD 偏移)
$S_{macro}$ 是一个**全局尺度（Global-Scale）**的低维向量，只包含系统级别的因果指标，绝对剔除了单车的空间坐标与瞬时瑞利衰落等马尔可夫微观噪声。
- **$s_{mac}^{(1)}$ 区域干扰温度密度 (ZITD, Zonal Interference Temperature Density)**:
  反映空间拥塞导致的数据洪流。计算公式：$I_{zonal}^{(i)} = \sum_{j \neq i} P_j \cdot \exp\left(-\frac{d_{i,j}^2}{2\sigma^2}\right)$。通过邻接矩阵实现高斯衰减扩散，输入网络前进行全局归一化。
- **$s_{mac}^{(2)}$ MEC 算力负载率 (MEC Load Ratio)**:
  反映算力雪崩。计算公式：$L_{MEC} = \frac{\sum_{i \in \text{MEC}} D_i}{f_{MEC} \cdot 1\text{s}}$，代表当前时隙 MEC 的算力需求与实际可用算力上限的比例。
- **$s_{mac}^{(3)}$ 核心网拥塞率/时延衰减 (Core Network Congestion/Jitter)**:
  反映边缘到云端的回传链路质量。计算公式：$R_{ec\_ratio} = \frac{\hat{r}_{ec}}{r_{ec\_ideal}}$，其中 $\hat{r}_{ec}$ 包含突发的网络拥塞惩罚。
- **$s_{mac}^{(4)}$ 全局任务激增倍数 (Global Traffic Surge Multiplier)**:
  反映特定网格或全网的任务到达率突变（例如正常时为 $1.0$，数据洪峰时突变为 $10.0$）。

### 1.2 微观物理状态 $S_{micro}$ (专供 Prior DNN & Actor-Critic 决策)
$S_{micro}$ 是特定于每一辆车（Vehicle $i$）的**个体物理与业务特征（Agent-Specific Features）**，用于在既定的宏观环境背景下寻找最优解。
- **$s_{mic}^{(1)}$ 局部信道大尺度衰落 (Large-Scale Fading)**: 由距离 $d_i$ 决定的路径损耗 (Path Loss) 和对数正态阴影衰落 (Shadowing)。
- **$s_{mic}^{(2)}$ 局部信道小尺度衰落 (Small-Scale Fading, CSI)**: 遵循马尔可夫转移矩阵更新的瑞利快衰落 $h_t$。
- **$s_{mic}^{(3)}$ 任务数据量大小 ($U_i$)**: 待卸载任务的输入数据量（bits）。
- **$s_{mic}^{(4)}$ 任务算力需求 ($D_i$)**: 待卸载任务需要的 CPU 周期数（Cycles）。
- **$s_{mic}^{(5)}$ 任务类型标识 (Task Type Indicator)**: 区分 URLLC (如 1) 与 eMBB (如 0)。
- **$s_{mic}^{(6)}$ 任务容忍延迟红线 ($t_{max}^{(i)}$)**: 严格区分，URLLC 为 $3\text{ms}$，eMBB 为 $1.5\text{s}$。
- **$s_{mic}^{(7)}$ 车辆当前可用本地算力 ($f_{local}^{(i)}$)**: 车辆端的本地 CPU 频率。

---

## 2. 动作空间 (Action Space) 定义

环境针对 $N$ 辆车提供离散动作决策。为了便于混合整数非线性规划（MINLP）启发式算法与强化学习对齐，采用 MultiDiscrete 空间：$\mathcal{A} \in \{0, 1, 2, 3\}^N$
对于每辆车 $i$，其卸载动作 $a_i$ 定义如下：
- **$a_i = 0$ (本地计算, Local)**: 任务在车端本地处理，无传输时延，受限于本地算力 $f_{local}$ 与严苛的能耗惩罚。
- **$a_i = 1$ (MEC 计算, Edge)**: 任务卸载至最近的 MEC 边缘服务器。受限于无线 MAC 层的加权 OFDMA 竞争与 MEC 算力上限 $f_{MEC}$。
- **$a_i = 2$ (邻近边缘协同, Offsite-Edge)**: 当主 MEC 过载时，经过主 MEC 路由至邻近空闲 MEC 计算。增加一跳边缘间回传时延。
- **$a_i = 3$ (云端计算, Cloud)**: 卸载至核心网云端。享有无穷大算力 $f_{cloud}$，但需承受巨大的长时回传传输时延与高额 Jitter。

---

## 3. 系统单个时间步 (Time Slot) 执行时序图 (Event Sequence)

在强化学习的每一个时隙（Step）中，环境引擎的物理演进与计算必须严格遵守以下因果时序：

1. **宏观灾难 OOD 注入 (OOD Injection)**
   - 触发逻辑检查（是否进入 Compute Avalanche / Traffic Flood / Core Congestion 状态）。
   - 强行修改对应的环境物理外生参数（如降低 $f_{MEC}$，成倍放大 $U_i$ 和 $\lambda_i$）。

2. **微观物理状态演进 (Physical State Update)**
   - 根据车辆移动学模型更新绝对空间坐标。
   - 根据大尺度公式与马尔可夫过程计算瑞利小尺度快衰落 $h_t$。
   - 计算距离邻接矩阵，生成该时隙各区域的 ZITD (区域干扰温度密度)。

3. **观测提取与解耦输出 (Observation Yielding)**
   - 环境根据最新物理状态，向智能体组装并严格隔离输出 $S_{macro}$ 和 $S_{micro}$。

4. **动作接收与 MAC 层严格资源守恒切分 (Action Execution & strict OFDMA)**
   - 接收 RL 或启发式算法给出的决策 $\mathcal{A}$。
   - **严格统计**选择进行无线卸载（动作 1, 2, 3）的车辆数 $N_U$ 和 $N_E$。
   - 触发具有资源守恒属性的加权严格 OFDMA 切分：
     $$B_U = B \times \frac{W_U}{N_U W_U + N_E W_E}$$
     $$B_E = B \times \frac{W_E}{N_U W_U + N_E W_E}$$

5. **SINR 计算与传输率映射 (SINR & Shannon Capacity)**
   - 基于分配的带宽 $B_U/B_E$、大尺度路径损耗、瞬时瑞利衰落以及热噪声加 $I_{zonal}$，通过香农定理计算每辆车的真实上行传输速率 $R_i$。

6. **任务执行与时延/能耗统计 (Execution & Cost Aggregation)**
   - 根据动作 $a_i$ 累加路径上的传输时延与排队/计算时延，获得端到端总时延 $T_i$。
   - 累加发射功率消耗与 CPU 能耗，获得总能耗 $E_i$。

7. **异构代价函数惩罚 (Heterogeneous Cost & CVaR Barrier Computation)**
   - **eMBB**: 使用常规时延与能耗的加权平滑函数。
   - **URLLC (CVaR Barrier 代理)**: 激活爆炸性梯度屏障惩罚，严打延迟违例。
     $$Cost_{URLLC} = \alpha \cdot \text{Softplus}\left(\beta \cdot \frac{T_i - t_{max}}{t_{max}}\right) + \gamma \cdot \frac{E_i}{E_{max}}$$
   - 生成最终环境 Reward $= - Cost_{total}$。

8. **时间步推进 (Tick)**
   - 环境内部时钟推进，判断终止条件。

---

# IoV Env V3 核心构建与思考过程日志

## 4. 理论纠偏与物理悖论修正 (2026-05-27)
**思考过程**：
在收到审稿人（User）对 `IoV_Env_V3` 蓝图的终审意见后，我深刻反思了第5步中关于“OFDMA 带宽切分”与“同频干扰计算”产生的物理悖论。在严格的 OFDMA 体系下，同一基站（MEC）覆盖下不同车辆分配到的子载波是绝对正交的，因此**小区间内部干扰（Intra-Cell Interference）在理论上为 0**。如果将 ZITD（区域干扰温度密度）直接作为分母加进去，等同于破坏了 OFDMA 的数学纯洁性。

**修改动作（物理逻辑对齐）**：
- 在代码与理论注释中，严格将 $I_{zonal}$ 定义为**小区间干扰（Inter-Cell Interference, ICI）**或“周边区域其他基站频谱复用泄漏造成的宏观背景干扰”。
- 这样，本小区内的资源分配严格遵守正交守恒（$N_U \cdot B_U + N_E \cdot B_E \le B$），而外界涌入的 OOD 数据洪流会导致全局频谱底噪上升（即 $I_{zonal}$ 激增），从而完美兼顾了物理正交性与 OOD 宏观空间检测的合法性。

## 5. 状态空间与网络路由代码落地设计
**思考过程**：
为了配合 Actor-Critic 和 Autoencoder 的双脑架构，环境返回的观测值必须严格隔离。由于 Gym 原生的字典状态（Dict Space）在某些 RL 库中不易处理，我们将采用拼接的一维向量，并在环境文档和代码层严格标明切片索引。
- $S_{macro}$ (外生全局特征，4维)：ZITD, MEC_Load, Core_Congestion, Traffic_Surge_Mult。
- $S_{micro}$ (内生个体特征，7维/车)：大尺度损耗、瑞利快衰落、U_i, D_i, 任务类型、容忍延迟、本地算力。
**网络路由**：在 RL 代理代码（未来编写）中，Autoencoder 仅截取前 4 维作为输入，Actor-Critic 截取完整向量。

## 6. 从零搭建 IoV_Env_V3 动作记录
**动作记录**：
1. 创建 `iov_env_v3.py` 文件。
2. 剥离了旧版本中所有杂乱无章的 edge_sim_py 耦合，完全基于纯数学物理公式搭建轻量级、高保真的环境引擎。
3. 实现了 8 步因果时序。
4. 编写 `test_iov_env_v3.py` 以纯随机动作进行物理规律的数值验证。
