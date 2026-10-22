# zinc_shared_factor_v1 — 预注册（实验前冻结）

Track `structural-sufficiency`。基于仓库快照 `f446ba5a02e7ede882e249b4f8c5503eebb722e0`，
新目录、新协议 `zinc-shared-factor-v1`，不修改任何旧实验的定义或结果。

## 1. 问题

在**同一 P3 载体、同一 P3 接入口（`W_s`）、同一 `bf_data` S3/S4/F4 对象、同一 train-only
SSL 初始化、同一 recipe 与同一 47,400 updates 预算**下：

1. **实用问题（T vs D）**：用“每个真实原子一个共享隐状态 + 小片段上的精确收缩”替换稠密内容
   MLP，能否在完整 valid MAE 上得到更好的泛化？
2. **机制问题（T vs N）**：若有效，收益是否依赖共享变量保留的**连接关系**？N 与 T 参数、输入
   宽度、MLP 结构、初始值完全相同，只把非根原子在不同键中的共享状态拆成独立副本。

T 与 D 的差别**不止一条共享约束**（内容函数、输入宽度、参数化都不同），因此 T−D 只回答
实用效果；T−N 才是共享变量机制的对照。本实验不预设提升，也不预设新颖性。

## 2. 固定不变项

- official PyG ZINC `subset=True`，train 10,000 / valid 1,000；raw-y L1；不裁剪标签、不删除极端
  分子、不改 loss 权重；**本轮不访问 official test**（无 test 命令，runner 不加载 test cache）。
- 对象：`zinc_branching_fragment_v1/code/bf_data.py` 的 S3 / S4 / F4，完整类型化诱导邻接
  （含所有额外弦边），root 固定在 slot 0；多重性与 content-key 语义、split-local id 语义不变。
  不增加结构家族。
- 载体：`ad2_models.P3NullModel`（413,505 参数）+ 原始输入/global33 + `198→128` 零初始化
  `W_s`（25,344 参数）+ 每 root×family occurrence 加权 `mean / mean_sq / log1p(count) /
  existence`（3×66=198；空集合严格为零）。
- 初始化：`torch.manual_seed(seed)` 后按历史顺序构建 P3 核心 → 只加载 train-only masked-SSL
  encoders（`pretrain_encoder_s0.pt`，SHA256 `006fabc1…bfa97`）→ 追加 `W_s`（精确 0）与内容
  分支 → A/S/MLP 用隔离 generator（seed 404+500·seed）覆盖；D 沿用上一轮 DensePhi 的
  seed（embeddings 101+500·seed、phi 202+500·seed）→ seed 0 与上一轮 D **逐字节相同**
  （`466d9481…`）。不加载任何 supervised best/final。
- seed 1 复用同一份 SSL 权重（这不是独立的 SSL 预训练重复），只改变监督训练的随机初始化与
  数据顺序（shuffle seed = seed + 101）。
- 每步根据当前参数重新计算内容表示（autograd 保留）；只缓存结构/索引张量。

## 3. T 的精确定义

`C=16` 通道，`R=4` 状态（本轮不扫描）。参数：

- `A[28,C,R]`：`q_c(s|a) = softmax_s A[a,c,s]`；
- `K_{b,c}(s,t) = exp(tanh(S_{b,c}(s,t)))`，`S` 用对称上三角（含对角线）参数化，
  共 `3·C·R(R+1)/2 = 480` 个参数（`3` 键型 × `16` 通道 × `10` 个上三角元素）；
- 所有家族、所有出现位置共享 A 与 S；无 whole-pattern / 节点编号 embedding、无学习缓存状态。

对对象 `H`（根 `r`，真实原子 `a_i`，诱导邻接真实边集 `E`，键型 `b_ij∈{1,2,3}`，`m=|E|`）：

```
T_H,c(s) = Σ_{s_i, i≠r} [ Π_{i≠r} q_c(s_i|a_i) · Π_{(i,j)∈E} K_{b_ij,c}(s_i,s_j) ] ,  s_r = s
```

- 精确枚举 `R^{n-1}`（4 节点 64、5 节点 256）个联合赋值，在 log 空间用 logsumexp 求和；
  **不是** belief propagation，也没有多层邻居隐藏向量更新。
- 无边（`b=0`）与 padding 都是乘法单位 1：在 log 空间加 0，不存在可学习的“无边”势；
  padding 槽位没有状态、不参与求和、也不进入 `m` 的计数。
- 同一个真实原子无论参与几条键，都只有一个状态变量。

内容输入（133 维）：

```
q_c(s|a_root) 展平 (64) | log(T_H,c(s))/max(m,1) 展平 (64) | family one-hot (3) | n/5 | m/10
```

`phi_T`：`Linear(133,64) → SiLU → Linear(64,32)`（有 bias；无 dropout / LayerNorm / skip /
辅助损失）。T 天然对非根节点重排不变，因此**不枚举 6/24 个排列**；D 保留原有的精确置换平均。

参数：内容模块 `1,792 + 480 + 8,576 + 2,080 = 12,928`；总计
`413,505 + 25,344 + 12,928 = 451,777`（D = 449,601，差 2,176 = 0.48%）。T/N 没有内容 embedding
表（D 的 `E_atom[28,8]+E_pair[4,4]=240` 不计入）。

## 4. N 的精确定义

保留同一个 `q`、`K`、根状态与完全相同的 133 维输入与 MLP，只把 `T_H,c(s)` 换成

```
N_H,c(s) = Π_{(r,j)∈E} [ Σ_t q_c(t|a_j) K_{b,c}(s,t) ]
           · Π_{(i,j)∈E, i,j≠r} [ Σ_{u,v} q_c(u|a_i) q_c(v|a_j) K_{b,c}(u,v) ]
```

非根原子每次出现在一条键里都用独立状态副本：保留每条键的类型与两端原子类型，移除共享非根
原子带来的**键间依赖**。N 独立训练（同初始值、同 batch 顺序、同完整预算），不是把训练好的 T
在推理时切开。

## 5. 训练前必须通过的检查（`code/sf_checks.py` → `analysis/cpu_checks.json`）

1. 向量化 T/N 与独立 float64 暴力枚举一致（4 节点 / 5 节点 / 带 padding 布局）。
2. 根保持的非根重排不变；图内 occurrence 顺序不改变结果。
3. 结构见证：根 0 接叶子 1..4，H1 加边 (1,2),(2,3)，H2 加边 (1,2),(3,4)，四个叶子同类型、
   所有键同类型、原子/键类型计数相同；在**预先固定**的非退化随机参数（`WITNESS_SEED=20261022`）
   下 T 可区分、N 完全一致。不通过时查实现，不筛种子。
4. 纯星形（无叶子间边）T 与 N 一致；孤立额外节点是单位和的 no-op。
5. 根/非根类型、键型改变都有正确执行路径；padding 槽位写入垃圾值必须**逐位**无影响。
6. 全部参数在 optimizer 中；`W_s=0` 时内容参数梯度为 0（预期）；打开 `W_s` 后 q、K、phi 梯度
   非零，且一步更新后内容参数确实移动。
7. D 的 init/forward/partial-epoch 与上一轮 LF-D 逐位一致；T/N 初始参数一致；checkpoint 重载
   预测一致；train/valid content digest 与上一轮审计一致。
8. 参数计数 `449,601 / 451,777 / 451,777`，79 steps × 600 = 47,400 updates，warmup/cosine 端点，
   同 seed 三臂 batch order 完全一致。

这些检查只证明实现一致与局部区分性质，不证明表达力、泛化或新颖性。

## 6. Recipe（与上一轮正式 recipe 相同）

AdamW 单组、peak 1e-3 / min 1e-5 / wd 1e-5 / betas (0.9,0.999) / eps 1e-8、warmup 500、
per-step cosine over 47,400、clip 5、fp32、batch 128（79 updates/epoch）、600 epochs、
每 epoch 全量 valid、strict `<` 选单个 best、无 ensemble/soup/EMA/SWA/早停。

## 7. 预算与阶段门槛（探索性购买门槛，不是显著性标准）

- 每臂（含计时/检查/打分）硬上限 **1 GPU 小时**；全程硬上限 **6 GPU 小时**（失败重试计入）。
- 固定 horizon 600 epochs。若实测 epoch 时间投影超过 3,600 s：**不缩短候选**，停止并报告成本
  （先做不改变数学定义的实现优化）。
- 两卡并行：`res-gpu0` / `res-gpu1`（`rr`，process backend），不改用 `res-2`，不停止他人任务。

阶段 A（T seed0 ∥ D seed0，≤2 GPU-h）**同时**满足才进入 B：

1. `best_valid(T,s0) ≤ best_valid(D,s0) − 0.003`；
2. `best_valid(T,s0) ≤ 0.075`；
3. 20-epoch 滑动中位数支持 T 更好：`best20_window_median(T) < best20_window_median(D)` 且
   `tail100_median(T) < tail100_median(D)`（不是只有一个孤立最低点）。

阶段 B（T seed1 ∥ D seed1）：`best_valid(T,s1) < best_valid(D,s1)` 且
`best20_window_median(T,s1) < best20_window_median(D,s1)` 才算方向重复；否则停止并报告不稳定，
不挑最好 seed 当结果。

阶段 C（N seed0 ∥ N seed1）：仅在 B 重复后执行；N 使用对应 seed 的 T 初始状态 hash 与相同
batch order hash。

数据出现后不改门槛、不加 seed/通道/状态数/温度/层数/结构家族/预训练。

## 8. 交付与分析

主指标：完整 valid MAE（保留全部 1,000 个分子）。每臂每 seed 报告 best-valid、best epoch、
eval-mode train@best、final-valid、最优 20-epoch 滑动中位数（及窗口终点）、tail-100 中位数、
完整曲线、参数、运行时间、显存、代码/环境来源；同 seed 的 T−D（运行 N 后加 T−N）；
radius-2 seen/unseen × ordinary/tail 分组的 n / MAE / 对总体 MAE 的贡献；unseen 组合的原子/键
类型是否在训练中出现过；改善是否集中于单个极端样本（仅诊断，不改主结果）；T/N 表示差异与
训练后塌缩诊断（辅助证据）。分子级 bootstrap 只表示分子抽样不确定性，不代表跨 seed 方差；
两个 seed 也不足以估计可靠的 seed 方差。

## 9. 与既有工作的关系（报告里核对）

- **GCKN**（Chen et al., NeurIPS 2020）：基于 walk/path 核的卷积，用路径上的局部邻域求和
  构成节点表示；与本轮的“对象内精确配分”不同，但同属“显式局部结构化聚合、不用 MP 迭代”。
- **Deep Homomorphism Networks**（Nguyen et al., ICLR 2023）：用有界树深度的子图同态计数
  作为结构特征、并用可学习权重组合；其树状张量积与“共享状态 vs 独立副本”的差别正是本轮的
  机制对照（树/无环 ⇒ 独立副本足够，环 ⇒ 共享才有依赖）。本轮不是同态计数的实现，也不主张
  首创因子化或 no-MP。
- **加权图同态配分函数**：`T_H,c(s)` 是一个小片段上的带权（q 为顶点权、K 为边权）配分函数，
  即 `Z_H = Σ_{σ} Π_v q(σ_v) Π_e K(σ_u,σ_v)` 的根条件版本；N 是其树化/边独立近似。
  本轮把它当**局部内容特征**接进既有载体，而不是求 graphet、也不用它是完备不变量。
  因此本轮不声称首创因子化、结构组合或 no-MP。
