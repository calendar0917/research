# PROTOCOL — cscl-bjoc-mfic-v1 (MFIC: Matrix-Function Incidence Composition)

**状态**：预注册（CPU 科学购买门槛在任何本轮 CPU 结果产生之前冻结）
**Track**：`structure-contribution`
**Protocol id**：`cscl-bjoc-mfic-v1`
**Split tag**：`V`
**冻结方式**：`protocol_freeze.json` 绑定本文件、`configs/mfic_v1.yaml`、两份代码文件、
split / objects / prep / K 缓存 / CPU gates / CPU signal 的 sha256。冻结后任何语义改动都会使
`verify_freeze` 失败；CPU 通过后才会写 GPU pre-freeze。

---

## 0. 一句话问题

**把一个固定、标签盲、解析的矩阵函数 `K = sum_{k>=2} 0.65^k S^k` 与 BJOC 的最终原子内容做
一次图级二次型 `Q^T K Q`，用来替换 P3 reader 中的 `sum_i a_i^2` 通道，能否在不做任何图神经
消息传播的条件下，稳定超过 P3？**

CPU 阶段先回答一个更弱、更便宜的问题：这些解析矩阵特征是否对**当前 Fit 训练域**的标签变化
提供明显的额外信息？如果 CPU 购买门槛不通过，立即停止，不启动 GPU。

## 1. 严格计算边界

本轮**禁止**：

- GCN / GINE / GAT / 其他 MP-GNN；
- Transformer / Self-Attention / Graph Attention；
- 多轮可学习邻域聚合；节点或键状态的迭代更新；
- 预测 ensemble；
- 新的目标分量标签、化学公式注入、外部训练数据；
- 任何 official valid / official test 访问；
- 先做神经邻域传播再声称“最后一层没有消息传递”。

本轮**允许**：

- 现有 BJOC 一次性 Atom / Bond / Junction 编码器；
- 固定、标签无关的邻接矩阵解析函数；
- 图级张量二次型；
- 一次性共享 MLP 与非线性 Reader；
- 预处理阶段缓存 `K`（只读输入）。

矩阵 `K` **只能用于最终图级收缩**，不得写回 per-atom / per-bond hidden state。若审计发现实现
仍属于显式学习式 GNN 消息传递，立即停止，不通过改名回避约束。

## 2. 外部参考与披露（必须逐字保留在报告中）

Batatia et al., ICLR 2024, *Equivariant Matrix Function Neural Networks*,
<https://arxiv.org/html/2310.10434>。

该论文的 ZINC 报告为 **0.063 ± 0.002**，但**原模型实际使用 GCN 层**。本轮候选只借鉴
“邻接矩阵的解析矩阵函数”这一数学对象；它**不是论文模型的无 GNN 复现**，也**不能**把论文的
0.063 当作本模型的预期性能或成绩。任何把该数字归功于 MFIC 的表述都违反本协议。

## 3. 数据与 split V

- 只使用 PyG `ZINC(subset=True)` official **train** 10,000 行；official valid/test 永不加载。
- 沿用四条历史输入核对异常行的冻结排除规则：`941, 2210, 6071, 9766`，9,996 行可用。
- 身份 = 由 PyG 图重建的 canonical constitution SMILES；与 committed canonical SMILES 的身份
  划分逐行相同。
- 新 grouped split **V**：目标 `Fit 7,996 / Monitor 1,000 / Locked 1,000`（±10，按身份组贪心）。
- **显式 salt** `cscl-bjoc-mfic-v1`；断言：
  1. 实际 salt == 保存 JSON salt == 硬编码常量；
  2. 用保存的 salt + identity 元数据独立重算，三区逐行相同；
  3. 保存的 `split_sha256` 与重算一致；
  4. 换 salt probe 会改变划分；
  5. V 的任一区行哈希不与历史 E/F/S/T/U 的任何区相同，且 V 不与历史任一轮逐行相同。
- 报告 V 与 E/F/S/T/U 的完整行重叠矩阵。**披露**：同一 official-train 10k 总体已被历史多轮
  使用；V 是新的内部重采样评估，**不是独立盲测**。不得把新 salt 说成新独立数据。
- CPU 阶段只读 Fit 行的结构与其 raw `y`；Monitor / Locked 的 `y` 在 CPU 阶段保持不可达。
  CPU 特征、PCA、标准化、Ridge 系数、Ridge alpha 全部只用当前训练折拟合。

## 4. MFIC 模型

### 4.1 固定解析核

对原始分子图（去重 PyG 正反边后的真实物理原子邻接矩阵 `A`）：

```
D_ii = sum_j A_ij
S    = D^(-1/2) A D^(-1/2)        # 孤立点零度归一化按零处理
R    = solve(I - 0.65*S, I)        # 稳定线性求解，alpha=0.65 固定，不搜索
K    = R - I - 0.65*S
     = sum_{k>=2} 0.65^k S^k
```

- 必须使用稳定的 linear solve；禁止先显式数值求逆再反复相乘来模拟迭代消息传递。
- `K` 完全由结构生成，不训练、不使用 `y`，可在数据预处理时缓存；训练时只读。
- `K` 的数学/数值门禁见 §6。

### 4.2 图级二次型

原 BJOC Atom encoder 输出每原子 `h_i ∈ R^96`。加入共享线性投影：

```
q_i = Linear(96 -> 16)(h_i)
C   = Q^T K Q / max(1, n_atoms)          # 16 x 16 对称矩阵
```

取上三角含对角的 **136** 维特征。图级新 Reader 输入固定为：

```
[sum_composer(96), sum_atom(96), upper_triangle(C)(136), global(33)] = 361
```

这**取代**旧 P3 Reader 中的 `sum_i a_i^2` 96 维，而不是在 321 维上另接零初始化 residual。
Reader 固定两层 hidden=256、SiLU、dropout=0；参数量必须实测并保存：

| 模块 | 参数量 |
|---|---:|
| 共享 Atom/Bond/Junction/Composer | 265,024 |
| 共享投影 `q_proj` (96→16, bias) | 1,552 |
| 新 Reader `361→256→256→1` | 158,721 |
| **MFIC 总计** | **425,297** |
| P3 总计（对照） | 413,505 |

### 4.3 必须证明的事项

1. 没有任何 per-atom 状态接受 `K` 传播后的更新；
2. `K` 是标签盲的固定解析算子；
3. 整张图只在最终 readout 时进行 `Q^T K Q`；
4. 新分支的梯度真实存在（`q_proj`、Reader、共享 encoder/composer 均收到有限非零梯度）；
5. 节点重编号、边端点交换、batch 排列不影响结果；
6. 新 Reader 所有通道参与参数审计，不保留 dead parameter。

## 5. 对照臂

### A — P3

原始 413,505 参数 P3-R3 完整配方：同一 Fit-only masked SSL encoder、同一 raw-y L1、
Adam/cosine R3、360 epoch、wd 1e-5、clip 5、batch 128、last-5 parameter soup。

### B — MFIC-R

真实结构矩阵 `K` 与内容 `Q` 的组合，Reader 输入 361 维，参数 425,297。

### C — MFIC-U

与 MFIC-R **同参数、同初始化、同形状、同训练预算**，使用 `K` 的内容归属 Null：

```
d = trace(K) / n
o = (sum(K) - trace(K)) / (n*(n-1))        # n >= 2
K_U = d*I + o*(11^T - I)
```

`n = 1` 时单独定义 `K_U = d*I`（此时 `K = 0`，故 `K_U = 0`）。

然后：

```
C_U = Q^T K_U Q / max(1, n)
```

Null 保留每分子的 `K` 对角质量、非对角总质量和内容 `Q` 的多重集，但破坏具体结构—内容的
对应。R/U 两臂实际均预计算 `K`，前向使用相同形状的稠密矩阵乘法；不能省掉 U 的实际计算后
假装等计算成本。

**必须披露**：`K_U` **不保证**保留真实 `K` 的全部特征谱、行和或空间分布，因此 `R−U` 不能
单独被解释为严格的化学因果效应。

### 初始化与公平性（GPU 阶段）

同 seed 三臂：

- 同一 Fit-only masked SSL encoder checkpoint；
- 同一公共模块（Atom/Bond/Junction/Composer）初始权重，逐位相等；
- R/U 所有参数初始权重完全一致（同 seed 构造 + 独立断言 full-state hash 相等）；
- 同一 minibatch order；
- 同一 supervised 更新数量与 raw-y L1；
- 同一 R3：360 epoch cosine，Adam、wd 1e-5、clip 5；
- 相同 last-5 parameter soup。

Reader 形状不同，因此 P3 与 MFIC 的完整参数初始化不要求相同；公共模块必须一致。

## 6. Stage A：CPU 科学购买门槛

**这个阶段必须先完成，不运行正式 GPU。** 目的不是证明 MFIC 一定超过 P3，而是判断新的长程
矩阵特征是否对当前训练域的标签变化提供明显额外信息。

### 6.1 只使用 Fit 行

- 官方 train 10k，排除四条历史异常行；
- 只使用 split V 的 Fit 7,996 行及其 raw `y`；
- Monitor、Locked、official valid/test 标签均不可见；
- 无模型预测进入 CPU 特征；无 target components。

### 6.2 四个预固定 grouped 内部折

在 Fit 的 identity groups 上按 `sha256(salt|fold|identity)` 排序，整组贪心放入当前最小的折；
四折在读取任何 `y` 之前固定并写盘。每折训练 = 其余三折，验证 = 当前折。identity group 绝不
跨折。

### 6.3 CPU-Base

已有原始结构汇总特征：

- 28 维 atom 类别计数；
- 4 维 bond 类别计数；
- 原始 global33；
- 3 维原始规模统计 `n_atoms, n_bonds, n_junctions`。

共 **68** 维。无 target components、无化学公式、无外部训练数据。

### 6.4 CPU-Matrix

在相同 Base 上增加解析矩阵二次型：

- 投影 `Q` 来自**仅当前训练折原子**上拟合的无监督 PCA（16 维，中心化，不做额外缩放）；
- 验证折原子只能 `transform`；
- `C = Q^T K Q / max(1,n)` 的上三角含对角 136 维；
- 总维度 **204**。

### 6.5 Ridge 探针

两臂均：

- 同样的标准化规则：训练折独立拟合 mean/std（零方差列 scale=1），验证折只 transform；
- 相同的 Ridge 回归，固定 `alpha=1.0`；
- 截距由训练折 raw `y` 均值给出；
- 无超参数搜索；相同评估分子与 label access。

这个 Ridge 探针**不是**最终 BJOC 性能估计，只检验新增矩阵信息在简单统计模型下是否有明显
标签相关收益。

### 6.6 CPU 购买门槛（全部满足才通过）

1. CPU-Matrix 四折平均 MAE 比 CPU-Base 低至少 **0.020**；
2. 至少 **3/4** 折方向一致（`MAE_matrix < MAE_base`）；
3. 去掉每折 Base 最差的 10 个分子后，平均方向仍改善（平均 `delta < 0`）；
4. ordinary (`y > -3`) 平均方向仍改善（平均 `delta < 0`）；
5. 图同构、矩阵对称、有限数值和置换不变性门禁全部通过；
6. 计算新特征不依赖任何 official valid/test 或模型预测。

**阈值是资源购买启发式标准**，不能把 CPU Ridge 通过解释为神经网络的必然收益；它也可能
错过非线性模型才能利用的信息。本轮接受这种保守的资源取舍。

若不通过：

```
MATRIX_FEATURE_NO_CHEAP_SIGNAL
```

停止、报告，不启动 GPU，不换 alpha、不换 PCA 维度、不换 kernel、不换 Ridge。

## 7. CPU 数学门禁（与 §6.6 第 5 条对应）

独立核验：

- `K`、`S` 对称；`R` 数值有限；
- `I - alpha*S` 的最小特征值严格正；
- `K` 与 `sum_{k=2}^{40} alpha^k S^k` 的截断级数一致（浮点容差）；
- 同一图重编号后，`K` 变换 `P K P^T` 与 `Q^T K Q` 计算不变；
- 简单链、单环、融合环、多环、孤立点都正确；
- `K` 不是一个仅由 `n_atoms` 决定的常数通道；
- 真实 `K` 与 `K_U` 在非退化示例上不同；
- 两种矩阵在模型中均实际被消费；
- CPU 合成例中可以区分某些有限短阶 walk 统计相同的非同构图；
- 不能宣称 synthetic witness 代表真实 ZINC 泛化收益；
- 检查全图 `sum(R)` 是否退化为常数；本协议**不使用** `sum(R)` 作为特征。

## 8. Stage B（仅 CPU 通过后冻结并执行）

### 8.1 split V

目标 Fit 7,996 / Monitor 1,000 / Locked 1,000。显式 salt，独立校验 row hashes 与 identity
分组。报告与历史 E/F/S/T/U 的交叠。同一个 official-train 10k 池已被多轮使用，不得将新划分
称为独立盲测。

### 8.2 模型与训练

- P3、MFIC-R、MFIC-U 三臂；
- 两个 seed：0、1；
- 同 seed 三臂共享：Fit-only masked SSL encoder checkpoint、公共模块初始权重、R/U 全部初始
  权重、minibatch order、supervised 更新数量、raw-y L1、R3 配方、last-5 soup；
- 预算上限 **3.0 GPU-hours**，包含 smoke、SSL、监督训练、预测与失败重试；
- `res` GPU0 跑 seed 0、GPU1 跑 seed 1；所有远端操作走 `rr`；单 GPU pool 内可见设备通常是
  `cuda:0`，不能把物理 GPU1 直接映射成进程内 `cuda:1`；
- GPU smoke 若预计正式总时间超过 **2.7 GPU-hours**，停止而不是偷偷缩短训练。

### 8.3 预测冻结与评分

两 seed、三臂、Monitor/Locked 全部生成 label-blind 预测、checkpoint、哈希并冻结。之后只读取
Monitor 标签。

主比较：

- `MFIC-R − P3`：绝对 MAE 是否改善？
- `MFIC-R − MFIC-U`：真实拓扑内容对应是否比解析 content-null 更好？
- `MFIC-U − P3`：只是更换一般二次型容量是否就有收益？

主指标：raw-y MAE；报告两 seed、分子级配对 95% CI（5,000 次 bootstrap，分子为抽样单位，
固定 seed 0）、ordinary、tail、top-10-removed、RMSE、Fit MAE、逐 seed 差异、计算成本与结构
特征参与度。

**Monitor 主购买门槛（全部满足）**：

1. `R − P3 ≤ −0.004`；
2. 两 seed `R − P3` 同向（均 < 0）；
3. 配对 CI 上界 < 0；
4. ordinary 和 top-10-removed 均改善；
5. RMSE 与 tail 无明显灾难性退化（RMSE 退化 ≤ +0.005；tail 退化 ≤ +0.005）；
6. `R − U ≤ −0.003`，且两 seed 同向（均 < 0）；
7. 真实结构矩阵对最终预测存在非退化响应（label-free swap：`mean|Δpred| ≥ 0.0025` 且
   `max|Δpred| ≥ 0.02`）。

条件 6 用于支持“结构对应有效”的机制主张。若 R 超过 P3 但未超过 U，可记录一般函数族收益，
但不能声称真实拓扑绑定贡献。

若 Monitor 任何关键条件失败：

```
MFIC_NO_USEFUL_GAIN
```

立即停止；Locked 标签保持未读，不做 full10k、official test、alpha/维度/rank/Reader rescue。

只有 Monitor 通过，才读取已经冻结的 Locked 预测对应标签一次；同样条件确认。

两个区都通过，才报告：

```
MFIC_GENERALIZATION_CANDIDATE
```

这仅允许**提出**下一次独立 full10k 确认，不自动执行，也不能声称已达到 test 0.06。

## 9. 排重声明

本协议不是以下历史路线的重命名或 rescue：

| 历史路线 | 历史结果 | 与 MFIC 的区别 |
|---|---|---|
| SAC-v1 二阶键对张量 | Locked `SACR−P3 −0.0019`（CI 跨零），Monitor 不复制；`R−U` 单区弱信号 | SAC 使用**可学习** pair MLP + 线图距离桶 R1/R2/R3 的图级池化；MFIC 使用**固定解析**全长度 walk resolvent `K`，无 pair MLP、无关系桶、无学习邻接 |
| P3 bond-relation bridge | 零初始化门控塌缩，`swap` 预测差为 0 | MFIC 没有零初始化门控；`K` 直接进入最终二次型，梯度从第一步就存在 |
| STRC shared additive residual | `STRC_NO_PREDICTIVE_GAIN` | STRC 是共享 per-type 加性标量；MFIC 是 16 维内容投影的图级二次型 |
| DLP / Soft5 distance pooling | `SOFT5_NO_USEFUL_GAIN` | 距离池化替换 pair 矩桶；MFIC 替换 `sum a_i^2` 通道，使用解析 resolvent 核 |
| GLRC / cross-bond q / Readout 二阶矩 | 关闭或 CI 跨零 | MFIC 的新颖点不是“再加一个二阶矩”，而是**固定矩阵函数 + 最终图级收缩 + content-null 对照**；但它仍是图核/二次型家族，不能宣称全新函数类 |
| BJOC predictive bottleneck | 标签样本量是主瓶颈，输入不可观测性未成立 | MFIC 不增加标签或外部数据；它检验的是“固定长程矩阵内容是否提供新的标签相关信息” |

**诚实声明**：邻接矩阵的解析函数、resolvent/谱核和图级二次型在文献中并不新颖。本轮的贡献
是一个受控的、非消息传递的**放置检验**：把固定解析核放在最终 readout，与同形状 content-null
和原 P3 在同一 SSL/配方/预算下比较。若没有增益，不得把“矩阵被使用”解释成预测收益。

## 10. 交付物

`tracks/structure-contribution/results/mfic_v1/`：

- 本协议与 CPU/GPU freeze；
- split V、split compare、input audit；
- CPU 四折 Ridge 信号报告；
- K/R/K_U 数学与数值正确性测试；
- 参数与梯度审计；
- CPU 购买门槛与 decision JSON；
- 若 GPU 被购买：budget ledger、SSL checkpoint 哈希、预测冻结清单、LabelGuard 访问日志、
  Monitor/Locked 评分、最终 decision；
- 偏差与 amendments；
- 简洁中文 `REPORT.md`；
- `STATE.yaml` 指针更新（不覆盖历史）。
