# SSRC Phase 2a protocol — 原子级局部迭代计算是否给成熟 P3E 编码器带来真实增量

Protocol ID: `ssrc-phase2a-zinc-v1`
Date: 2026-10-20
Status: **FROZEN_PRE_GPU**（设计、门禁、子群、判定规则与 seed-1 购买条件在任何 Phase 2a
GPU 训练开始之前固定；写完后只允许实现错误修正，不允许改臂/宽度/配方/阈值）
Upstream: `results/phase1e/REPORT.md`、`notes/phase1e_protocol.md`、
`results/phase1b/REPORT.md`、`results/phase1e/strong_baseline.json`
Frozen code: `code/phase2a/`（只读 Phase 1b/1e 产物；不重训、不改写任何历史轨迹）

> 本轮的主协议仍然是 **SSRC 内部 Fit 8,000 / Monitor 1,000**。
> official ZINC valid/test 不加载；内部 Reserved 不评估、不用于选择。
> **披露**：Phase 1e 为标签公式分析曾加载全部 10,000 条 official-train 标签，
> 因此内部 Reserved 已有历史分析暴露，本轮不得再声称它"从未被读取"。

---

## 0. 排重：必须先证明本轮不是旧失败模型改名

代码审查（`tracks/structure-contribution/code/*.py` 全量 + `notes/*.md`）结论：
**在本仓库的全部历史 BJOC 变体中，`a = AtomEncoder(atom_feat)` 之后从未被任何
邻居消息更新过。** 所有已测扩展都发生在 `b`/`w`/`c`/reader 或图级统计上。

| 历史族 | 具体操作 | 判定 | 与本轮差异 |
|---|---|---|---|
| `p3_bond_relation_bridge_v1` | `c' = c + γ⊙Σ_f K_ef φ(b_e,b_f)`，γ 零初始化 | `CASE_D_NO_GAIN_ON_P3`；γ≈3e-4，残差为 c 的 6.6e-5，换核预测变化 0 | 那是在 **bond 级、composer 之后**、且零初始化门控；本轮在 **atom 级、AtomEncoder 之后、BondEncoder 之前**，标准初始化、无门控 |
| `sac_v1` / `bjoc_cross_bond_interaction_v1` | 图级二阶关系张量 `T_s`，只进 reader | `CASE_D_NO_GAIN` / 中性 | 图级汇总量，不改变 `a` 的感受野 |
| `distance_pooling_v1` / `DLP`（`bjoc_architecture_breakthrough_v1`） | 静态 distance-conditioned pair moments 追加到 reader | `DLP_NO_GAIN` | 一次性 pair 统计，无迭代、不更新原子状态 |
| junction ownership / Null-Organized / GLRC | `W_b` 归属、全局门控 | `GLRC_NO_GAIN` | 与原子状态无关 |
| `mfic_v1` / `bond_random_walk_organization_v1` / `bond_rw_local_vs_multiscale_v1` | 物理邻接上的矩阵函数/多尺度 bond 随机游走 pair 核 | `MFIC_NO_USEFUL_GAIN` / 中性 | bond–bond 核，`a` 仍是一次性的 |
| `bjoc_representation_optimization_v1` / `bjoc_typed_path_ring_features_v1` | 额外输入通道（typed path、ring block） | 无实质收益 | 增加输入统计，不增加迭代计算 |
| SSL 目标 / recipe / SAM / tail 加权 | 训练侧 | 已饱和/负结果 | 本轮不改配方 |

因此本轮唯一候选家族是**真正的两类插入式原子级计算**：Arm F（逐原子、不看邻居）
与 Arm M（两轮真实物理边消息传递）。两者都不使用零初始化门控、不引入环大小/目标
公式/额外人工描述符、不改 reader 维度。

---

## 1. 问题

在 BJOC-P3E 级成熟静态结构编码器上，加入**真实物理边上的少量原子状态传播**（2 轮），
是否比**参数匹配的非交互逐原子计算**（2 层 NodeFFN）产生更好的 ZINC 泛化？
以及，这条新路径是否超过冻结的 P3E（Monitor MAE 0.091071）？

区别的核心是**有效感受野**：P3E 的 `a_i` 是零跳逐原子表示，`b_uv` 只看两个端点，
`w_ijk` 只看两条键。Arm M 让 `a_i` 在进入 `BondEncoder` 之前看到 2 跳邻域，
从而改变后续 Bond/Junction 对象编码的有效输入。

---

## 2. 模型设计（冻结）

记 `dim = atom_out = 96`。两臂都在 `a -> a'` 处插入，其后**逐字节复用** P3E 的
`encode_bonds / encode_junctions / group_to_bonds(NULL) / group_composer / reader`。

公共块（两臂完全相同）：

```
_LocalBlock(z) = W2 · SiLU(W1 · LayerNorm(z))          # pre-norm FFN
h_{k+1} = h_k + _LocalBlock(input_k)
```

| | Arm F (`node_ffn`) | Arm M (`local_mp`) |
|---|---|---|
| 每轮输入 `input_k` | `h_k` | `h_k + msg_k` |
| `msg_k` | 无 | `Σ_{u→v ∈ E_phys} SiLU(h_k[u] + e_uv)`，双向边 |
| `e_uv` | 无 | `Wb · bond_feat_uv`（7→96），两轮共享 |
| 层数/轮数 | 2 | 2，权重不共享 |
| 归一化 | `LayerNorm(z)` | 同（对 `h+msg` 做 LN） |
| 残差 | `h + block(·)` | 同 |
| 门控 | 无 | 无 |
| 新增参数 | `2·(193H+288)` | `2·(193H+288) + 7·96+96` |
| H = 128 | 49,984 | 50,752（+1.54%） |
| 总参数 | 463,489 | 464,257（P3E 的 1.121× / 1.122×） |

- **禁止**：环大小、目标公式分量、任何新的人工结构描述符、global attention、
  多轮扫描（只测 2 层）、bond 状态更新。
- 初始化：两臂共享的 P3E 主干（atom/bond/junction/composer/reader + buffers）
  必须与 `bp.build_bjoc_null(seed=0)` **逐位相同**；随后覆盖为 Fit-only SSL encoder
  状态（sha `37282569…`）。新增模块用专用 `torch.Generator` 初始化（非零初始化），
  F 与 M 的两层 `W1/W2` 初始值相同；梯度在训练第 1 步即非零。
- 下游 reader 输入维度保持 321（`compose_out 96 + 2·atom_out 192 + GLOBAL_DIM 33`）。

---

## 3. 数据与隔离

- ZINC `subset=True` official train 10,000；内部 split 复用
  `results/phase1b/split_zinc_phase1b.json`（sha `fbca5fd5…`）：Fit 8,000 / Monitor 1,000 /
  Reserved 1,000，三方 row 集合互不相交（代码断言）。
- 标签 vault：`results/phase1b/zinc_labels_fit_monitor.json`（sha `631dae5f…`），
  只含 Fit+Monitor 原始 y。监督训练只用 Fit 行 raw y，L1。
- SSL 初始化：直接复用 Phase 1e 冻结的 Fit-only masked atom/bond encoder
  （`results/phase1e/baseline/pretrain_s0.pt`，文件 sha `0d045e60…`，
  encoder sha `37282569…`），**不重跑 SSL**。
- P3E 参照：直接复用冻结 soup（sha `94ba0f5f…`）与冻结预测
  （`monitor_pred_s0.npz`，sha `fc698ccc…`）。**不重训 P3E**；`replay` 阶段从 soup
  重算 Monitor 预测并断言与冻结数组逐位一致。
- official valid/test 不加载；Reserved 不评分。Monitor 标签在预测冻结之后才读。

---

## 4. 训练配方（冻结，与 P3E 完全相同）

R3：360 epoch、batch 128、Adam lr 1e-3 → cosine → 1e-5、wd 1e-5、clip 5.0、
raw-y L1、last-5 epoch 参数平均 soup、shuffle seed = `seed + 101`、
`steps_per_epoch = 63`。两臂使用与 P3E **相同的 batch 顺序哈希**
（`d142e307…`），由同一 `bp.shuffle_batches` 生成，作为审计项断言。
不早停、不选模型、不按 Monitor/Reserved 调任何超参。

---

## 5. GPU 前 CPU 门禁（全部通过才允许训练）

1. Atom/Bond/Junction 行号与端点映射：`bond_index` 每条无向键恰一次、端点真实、
   `bond_feat` 与 raw `edge_attr` 对应；Arm M 构造的双向边集合 = 正向 + 反向。
2. 原子重编号、边方向交换、batch 重排：整图预测变化 ≤ 1e-5（两臂）。
3. Arm F 的 `a'` 对物理邻接扰动**不敏感**（≤1e-7）；Arm M 敏感（>1e-4）。
4. Arm M 的分支确实使用真实邻居与键属性：重连（保度）改变输出；置换 `bond_feat`
   改变输出；对邻居 `h_u` 的梯度非零。
5. Arm M 输出不是 P3E `a` 的复制：`||a'-a|| / ||a||` 与余弦相似度均需显示实质变化。
6. 两臂新增模块全部参数在第 1 次 backward 后梯度非零；训练后分支未被旁路。
7. 共享 P3E 输入/SSL 状态/训练行号/优化配方哈希一致；batch 顺序哈希一致。
8. 模型前向只读 `batch` 的允许键；批次里没有 y、没有 cycle 分量、没有 Reserved/valid/test。
9. `kind="none"` 时插入模型与 `src.BJOC` 前向**逐位一致**（max|Δ| = 0），
   reader 输入维度与块顺序不变。
10. 小 batch、环分子、非环分子、复杂分子的 forward/backward 均有限。

另加：`replay` 阶段断言 soup/SSL 文件哈希、encoder 哈希、冻结预测逐位一致、
P3E MAE = 0.091071（1e-9）。

---

## 6. 评估

主指标：**原始完整 ZINC 目标**（含人工 cycle penalty）的 Monitor MAE / RMSE。
同时报告：

- `MAE(F) - MAE(M)`（`Gain_MP`，正 = 真实传播更好）；
- `MAE(P3E) - MAE(M)`（`Gain_P3E`）；
- `MAE(P3E) - MAE(F)`（`Gain_F`）；
- 配对 bootstrap（按分子，5000 次，95% percentile CI），单 seed **仅发现性**；
- 子群：`label_complement`（957）、`label_penalised`（43）、`rdkit_large`（40）、
  `penalty_only_rdkit_missed`（3）、**cycle-basis 编号敏感**（4，仅诊断）；
- 误差尾部：AE 分位数（p50/p90/p95/p99/max）、top-1%/top-5% 误差份额、
  去掉 top-1%/top-5% 后的 MAE；
- 效率：可训练参数、前向 GEMM FLOPs（同一固定 batch 的 `FlopCounterMode`）、
  实际 step 时间、完整训练 GPU-hours、推理时间、峰值显存。
- 子群只能描述，**不能**用于选择配置或训练专用分支。

---

## 7. 预注册判定规则

主对比顺序：F vs M（因果），M vs P3E（实用），F vs P3E（容量）。

实用门槛：`ΔMAE ≥ 0.004`（沿用 Phase 1b/1d 的 ZINC meaningful band 的方向修正；
0.004 是本轮"值得改成熟系统"的最小幅度）。

| 状态 | 条件（按顺序判定） |
|---|---|
| `OPTIMIZATION_UNRESOLVED` | 任一臂出现非有限损失、梯度异常、soup Fit-MAE > 0.05、或训练曲线明显未收敛（末 5 epoch train MAE 中位数 > 0.01） |
| `STRONG_BACKBONE_MP_INCREMENT` | `Gain_MP ≥ 0.002` 且 `Gain_P3E ≥ 0.004` 且 RMSE 不劣于 P3E 超过 +0.005 且 complement 子群增益 ≥ 0.002 |
| `CAPACITY_ONLY_INCREMENT` | `Gain_MP < 0.002`（含 F 更好）且 `Gain_F ≥ 0.004` 且 `Gain_P3E ≥ 0.004` |
| `NONCOMPETITIVE_COMPUTE_TRADEOFF` | `Gain_P3E ≥ 0.004` 且 `Gain_P3E < 0.010` 且前向 FLOPs 比 ≥ 1.5×（本轮预计 ~1.19×，正常不会触发；若实测触发则如实记录） |
| `NO_PRACTICAL_GAIN_OVER_P3E` | 其余（`Gain_P3E < 0.004` 且 `Gain_F < 0.004`） |
| `INSUFFICIENT_EVIDENCE` | 只允许在产物/哈希/隔离校验失败时使用；不得用于掩盖负结果 |

解释规则（写进报告，不许事后改）：
- M 优于 F 且优于 P3E → 局部迭代计算在强编码器上有应用增量。
- M 只优于 F、两者都不优于 P3E → 传播在这个新模块家族内部更好，但没有证明值得改成熟系统。
- F、M 都改进 P3E 且接近 → 收益主要来自新增容量/训练结构，不是邻接传播。
- F 优于 M → 降低"局部传播有利"的先验。
- 全部未改善 → 停止本轮新分支，不做自动超参 rescue。

### seed 1 购买条件（全部满足才买）

1. `Gain_P3E ≥ 0.004`；
2. `Gain_MP ≥ 0.002`（明确可见的传播优势）；
3. RMSE 不劣于 P3E 超过 +0.005；
4. complement（957）MAE 改善 ≥ 0.002，即收益不是几乎全部来自 cycle-penalty 子群或少数极端分子；
5. 新增模块梯度正常、无优化故障；
6. 实测 GPU-hours / FLOPs 相对收益可接受（≤ 1.5× P3E FLOPs 且 ≤ 0.6 GPU-h/臂）。

满足则训练 F/M seed 1（P3E 使用同 seed 的匹配基线**仅在需要时**购买；本轮不预置）。
两 seed 方向冲突 → `UNRESOLVED`，不再买 seed。

---

## 8. 预算

上限 **4.0 A100 GPU-hours**（含 smoke、失败作业、SSL、监督、推理）。
预计：smoke 0.02 h + F 0.16 h + M 0.18 h + 推理 0.01 h ≈ 0.37 h（seed 0）；
若买 seed 1，再 ≈ 0.35 h。SSL/P3E 重训 = 0 h（复用冻结产物）。

---

## 9. 明确不允许

- 按 Monitor 结果搜索 GINE 层数/宽度/学习率/正则；
- 新增关系池化、attention token、ensemble、新 SSL 目标、ZINC-Full、official test；
- 为 cycle penalty 增加标签规则通道或按子群训练专用分支；
- 在失败的新模块上反复改门控/LR/初始化救援；
- 把"P3E + GINE 有用"当作论文新颖点（GraphGPS 等已结合局部 MP 与结构编码）。
