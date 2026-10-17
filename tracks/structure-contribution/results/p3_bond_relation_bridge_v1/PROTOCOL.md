# cscl-p3-bond-relation-bridge-v1 — 预注册协议

状态：**冻结草案（在任何 GPU 训练、任何 Monitor/Locked 标签读取之前）**
split：**F**（新的内部重划分；salt `cscl-p3-bond-relation-bridge-v1`）
上游：`bond_relation_attribution_v1`（`CASE_C_INCONCLUSIVE`，其冻结判定不被本轮修改）

---

## 1. 问题

在保留 BJOC-P3 的 atom/bond/junction 输入、SSL 初始化、R3 配方与全部参数的前提下，
**加入一次真实 Bond–Bond 一跳邻接的、共享 pair MLP 的局部非加性组合，能否给 P3 带来
可复现的增量预测收益？该收益是否超过同容量、同行和的稠密 Uniform 关系分支？**

三臂（仅此三臂，不做架构/宽度/学习率/epoch/算子搜索）：

| 臂 | 关系分支 | 新增参数 |
|---|---|---|
| `P3` | 无（原始 BJOC-Null P3） | 0 |
| `P3R` | 真实线图邻居上的行归一化一跳关系 | pair MLP + 零初始化残差 |
| `P3U` | 全 bond 稠密均匀关系（同容量、同行和） | 完全相同 |

## 2. 关系核（R / U）

对分子 `m` 的 `n = n_bonds` 个物理 bond，线图邻接 `A`（`A_ef = 1` 当且仅当 bond e 与
bond f 共享恰好一个原子；对角线为 0；无多重边）。记 `deg(e) = Σ_f A_ef`。

- **R**：`K^R_ef = A_ef / deg(e)`（`deg(e)=0` 时为 0）。行和 = 1（有邻居时），
  对角线严格 0。**无自连接、无 lazy 项**（与历史 `lm.k1_matrix` 不同）。
- **U**：`K^U_ef = 1/(n-1)`（`e ≠ f`，`n ≥ 2`）；`n = 1` 时全零。行和 = 1，对角线严格 0。

**同行和匹配**：两核在 `n ≥ 2` 的每个 bond 上行和都恰为 1，总质量都恰为 `n`，对角线都为 0。
因此 R−U 的差异**不是**行和/自连接差异，而是：(i) 支撑（真实邻居 vs 全体）、
(ii) 权重集中度/熵、(iii) FLOPs。R 稀疏、U 稠密这一点**未被隔离**，见 §7。

退化：线图是完全图时 `R ≡ U`（`n=2` 恒成立；`n=3` 仅当三键两两相邻）。审计必须报告
这种分子的数量与占比。

## 3. 模型：零初始化残差桥

`c_e = group_composer([b_e, W_e, V_e, log1p(n_b), bond_feat_e])` 是 P3 的 per-bond composed
表示；`z = Σ_e c_e` 进入 reader。新增：

```
r_e = Σ_{f≠e} K_ef · φ(b_e, b_f)            # φ 对称，输入 [b_e+b_f, |b_e−b_f|]
c'_e = c_e + γ ⊙ r_e                        # γ ∈ R^96，零初始化
z'  = Σ_e c'_e ;  reader_in' = [z', a_sum, a_sq, g]
```

- `φ`：`Linear(192→32) → SiLU → Linear(32→96)`，**P3R 与 P3U 共用同一实现、同一
  初始化（seed 决定）**；`γ` 逐通道零初始化。新增参数 = 12,416 (φ) + 96 (γ) = **12,512**；
  总参数 `422,945`（两臂完全相同；P3 为 `413,505`）。
- 初始化时 `γ = 0` ⇒ `c' = c` 逐位相同 ⇒ **三臂初始预测完全一致**（CPU 门禁逐位断言 0.0）。
- 训练不冻结任何 P3 主体参数；不改变 reader / composer / encoder 结构。
- 实现方式：wrapper 调用 `base.forward_features`，只替换 `z` 块（`reader_in[:, :96]`），
  不修改 `structural_relation_composer_v1.py`。

## 4. 数据、划分与隔离

- 官方 ZINC-12K `train` 10,000 行；4 行 committed-SMILES 与 PyG 图不一致
  （`941, 2210, 6071, 9766`）按统一规则排除 ⇒ **9,996 行**（与上一轮一致）。
- 身份 = 由 PyG 图重建的 canonical constitution SMILES（去立体化学）；已验证它与
  committed-SMILES 身份给出相同划分。按 `sha256(salt|identity)` 排序贪心填充，
  身份组不跨区；目标 **fit 7,996 / monitor 1,000 / locked 1,000**（允许贪心组边界导致
  ±10 的偏差，实际值记录在 `split_F.json`）。
- 模型/超参选择**不使用** split E 的 Monitor/Locked，也不使用 split F 的 Monitor/Locked；
  本轮三臂的差异只来自 §2–§3 的预注册设计。
- **不读取 official valid / official test**；Monitor/Locked 的 `.y` 在预处理、SSL、训练、
  预测阶段通过 `LabelGuard` 不可达；预测先冻结哈希，之后**单次**读取两区标签。
- 计算 checkout 只携带提交进 Git 的 JSON 证据；被 gitignore 的无标签 `.pt` 中间产物
  （objects/pairs）由 `stage-inputs` 依**冻结的** split/standardiser/pair 总数确定性重建并
  语义校验（amendment-1，见 `AMENDMENTS.md`），不读标签。
- 限制：split F 与历史各轮共用同一 10k 分子总体（不同划分、不同 SSL、不同轨迹），
  因此**不是独立外部验证**；绝对 MAE 也不与历史轮次直接可比。

## 5. 训练与预算

- 每 seed 一个 Fit-only 全量 masked SSL（`bp.pretrain_masked`，mode `atom_bond`，
  7,560 updates，无标签），三臂共享该编码器状态（含 junction 组）。
- 三臂：R3 配方，360 epochs × 63 steps = **22,680 updates**，batch 128，raw-y L1，
  Adam 1e-3→1e-5 逐步 cosine，wd 1e-5，clip 5，last-5 soup；batch order = seed+101 的
  `randperm`（与历史一致）。
- seed 0 → GPU0，seed 1 → GPU1，并行；每 seed 依次 `P3, P3R, P3U`。
- **GPU 预算上限 4.0 GPU-hours**（SSL+训练计入 ledger；预测前向约 1 GPU-min/seed 不入账，
  在报告中披露）。若 ledger 超限立即停止并按 §8 报告不完整结果。
- 若 CPU 门禁失败：修复工程缺陷或停止；**不替换模型设计**。

## 6. CPU 门禁（正式 GPU 之前全部通过）

1. R/U 对角线为 0、行和 = 1（`n ≥ 2`）、由 pair list 重建的稠密核与定义逐位一致、
   bond 槽位置换下关系等价（重编号不变性）。
2. 同 seed 下 P3R 与 P3U 的 base 参数、pair MLP 参数、`γ`、初始化哈希完全一致；
   三臂 batch order 哈希一致；参数审计 422,945 / 422,945 / 413,505。
3. `γ = 0` 时增强模型输出与 P3 **逐位相同**（同一 batch，max|Δ| = 0.0）。
4. `γ ≠ 0` 时 R 与 U 都改变前向；同一模型换核输出不同；pair MLP 的梯度在若干
   optimizer step 后非零（第 1 步 `γ` 梯度非零、φ 梯度为 0 是零初始化残差的已知性质，
   必须如实记录）。
5. 关系分支不读取任何标签（只消费 `b`、pair 索引与权重）；审计 junction 分组路径
   未被改动（`base` 模块参数/输出不变），并记录与历史 Organized/Null、GLRC 的差异。
6. 普通图、单键图（`n_b=1`）、环图（含 `n_b=2/3` 的退化）上 R/U 均可稳定前向，
   空 pair 列表可用。
7. 训练循环与 `bp.train_supervised`（冻结 P3 训练器）在同一 seed/同一 32 分子
   1-epoch 上 soup 逐位一致（本机；远端按 1e-6 float32 噪声地板记录）。
8. 所有核构造、模型配置、评分与判定代码在 freeze 中哈希固定；freeze 之后不得改动。

## 7. 混杂与不可声称的结论

- **R−U 混有支撑稀疏性/FLOPs 差异**：R 只在真实邻居上求和（平均 ~2–3 项），U 在全 bond
  上求和（n−1 项）。因此 R−U 不能单独识别“真实连接归属”的因果效应；这是**性能**对照，
  不是因果识别。
- 该分支接近一次 **bond-line-graph message passing**（一跳、对称 pair MLP、行归一化），
  不能声称“完全没有消息传递”。它不是 GNN/attention/迭代式消息传递：单次、固定核、
  无多层传播。
- 与历史 junction 归属实验（Organized/Null 分组、GLRC 门控）可能**信息冗余**：P3 的
  junction moments 已经广播了局部环境；新增项只用 bond 表示，不含 junction 内容，
  但两者都由同一 atom/bond 编码器产生。
- 新增容量（+3.0% 参数）与额外 pair 运算本身由 `P3U` 控制；`P3U − P3` 是容量/一般
  pair 交互的估计。
- 同一 10k 分子总体的历史暴露使本轮**不是**独立确认；结论只针对 split F 的内部对照。
- **换核诊断（预注册）**：在标签读取之前，用冻结的 P3R 权重配 U 核、P3U 权重配 R 核各做一次
  Monitor/Locked 前向（`predict-cross`，`cross_*` 文件），与同臂自己的核比较预测差。
  它只用于机制证据（学到的 γ,φ 是否依赖具体核），不参与任何选择或判定。

## 8. 主判定（在看到任何本轮结果之前冻结）

主区域 **Locked**（n≈1,000），复现区域 **Monitor**。配对 bootstrap：按分子重采样
5,000 次，seed 0（与历史评分一致）。效应尺度 **MEANINGFUL_DELTA = 0.004 MAE**。

对每个对比 `A − B`（负 = A 更好）：

- `GAIN`：`Δ_mean ≤ −0.004` 且两 seed 的 Δ < 0 且 bootstrap CI 上界 < 0 且普通区
  （`y > −3`）Δ ≤ 0。
- `LOSS`：`Δ_mean ≥ +0.004` 且两 seed 的 Δ > 0 且 CI 下界 > 0。
- 否则 `NEUTRAL`。

`replicated`：该对比在 Monitor 上 `Δ_mean < 0` 且两 seed 的 Δ < 0（不设幅度门槛）。

判定（优先级从高到低，互斥）：

- **A**：`P3R−P3` GAIN 且 replicated，且 `P3R−P3U` GAIN 且 replicated。
- **B**：`P3R−P3U` GAIN 且 replicated，但 `P3R−P3` 不是 GAIN。
- **C**：`P3R−P3` GAIN 且 replicated，且 `P3U−P3` GAIN 且 replicated，
  但 `P3R−P3U` 不是 GAIN。
- **E**（预注册的残余标签）：`P3R−P3` 是 GAIN，但不满足 A/B/C
  （即未在 Monitor 复现，或未同时胜过 U）→ `CASE_E_R_GAIN_NOT_ATTRIBUTED`。
- **D**：`P3R−P3` 不是 GAIN → `CASE_D_NO_GAIN_ON_P3`。

语义：只有 A 才构成“关系组织对强 backbone 有额外预测价值”的初步证据；B/C/D/E 都
**不**购买新的 ZINC 结构模块实验。seed 或区域不一致时如实报告不确定性，不事后调参。

## 9. 交付

`results/p3_bond_relation_bridge_v1/`：`PROTOCOL.md`、`protocol_freeze.json`、
`split_F.json`、`relation_audit.json`、`cpu_gates.json`、`reuse_audit.json`、
`ssl_F_s*.json`、`train_{P3,P3R,P3U}_F_s*.json`、`prediction_manifest_F.json`、
`holdout_read.lock`、`holdout_score_F.json`、`decision.json`、
`per_molecule_errors_*.csv`、`post_hoc_bridge.json`、`gpu_budget_ledger.json`、
`evidence_manifest.json`、`REPORT.md`、`predictions/*.npz`（含 `cross_*` 换核诊断）。
