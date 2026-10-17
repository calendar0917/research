# PROTOCOL — cscl-bjoc-sac-v1 (BJOC-SAC)

**状态**：预注册（在任何本轮 GPU 轨迹开始之前冻结）
**Track**：`structure-contribution`；**Protocol id**：`cscl-bjoc-sac-v1`；**Split tag**：`S`
**冻结方式**：`protocol_freeze.json` 绑定本文件、`configs/sac_v1.yaml`、两份代码文件、split /
objects / relations / gates / input audit 的 sha256。冻结后任何语义改动都会使 `verify_freeze` 失败。

---

## 0. 一句话问题

**结构限定的二阶交互张量（非消息传递、图级收缩），能否在保留 BJOC 强内容编码能力的同时，
改善真实分子的 held-out 预测？** 并把它拆成三个可反驳的对照：

1. `SACR` 是否比同轮 `P3` 更准？
2. 增量来自**真实结构关联**，还是**一般二阶容量**（`SACU` 内容 Null）？
3. 关系张量是否真的参与了最终预测（换 Null 会改变预测）？

## 1. 历史证据（只读，不覆盖旧决策）

| 证据 | 结论 | 本轮如何吸收 |
|---|---|---|
| `cscl_v0_gpu` | 严格加性关系显著不足（additive 0.4255 vs opaque 0.4043） | 不做纯加性分解；保留 opaque reader |
| `cscl_task_adaptive_structure_units_v1` | 可学习划分与 opaque reader 均不及 BJOC（最好的 F-O 0.1232 vs BJOC-Null 0.1043） | 不引入可学习划分；沿用 P3 的 Null composer |
| `bond_relation_attribution_v1` | 真实一跳关系在弱 backbone 上有稳定收益（R−D −0.0175，R−U −0.0192），但 R−P3 +0.0152 | 关系本身有信息；弱骨干上的收益不等于强骨干收益 |
| `p3_bond_relation_bridge_v1` | 零初始化关系残差在 P3 上塌缩（|γ|→1e-4/1e-11，换核 Δpred=0），R−P3 −0.000099 NEUTRAL | **禁止零初始化融合门控**；关系张量必须直接进 reader |
| `bjoc_representation_optimization_v1` | typed short-path / ring features 过拟合（fit −0.0138，locked +0.0124） | 不新增手工键型/环型特征；关系只作用于**已学到的** `c_e` |
| `bjoc_recipe_tuning_v1` / `bjoc_r3_frozen_terminal_test_v1` | R3 配方；P3 历史 exposed-test 0.069822 | 三臂共用冻结 R3；本轮**不读** official test |

## 2. 架构（唯一的新增部分）

复用原 BJOC 的 atom / bond / junction encoder、masked SSL、Null composer 及其 per-bond
`c_e ∈ R^96`（`structural_relation_composer_v1` 一字不改）。只重设计图级组合与 Reader 输入。

```
q_s(e) = tanh(W_s c_e)                          W_s: 96 -> 24, s = 1,2,3
R1 = {e<f : dist_line(e,f) = 1}                 (两条物理 bond 共享端点)
R2 = {e<f : dist_line(e,f) = 2}
R3 = {e<f : 同一含环双连通块 且 dist_line(e,f) >= 3}
T_s = sum_{(e,f) in R_s} (q_s(e) * q_s(f)) / sqrt(max(1, |R_s|))
readout_input = concat(P3_original_reader_input, T_1, T_2, T_3,
                       log1p(|R_1|), log1p(|R_2|), log1p(|R_3|))
y_hat = Reader(readout_input)                   # 3 层 SiLU MLP (396-256-256-1)
```

* `dist_line` 是**物理 bond 线图**（PyG 正反边先按 `unique_undirected_bonds` 去重）上的无权最短距离；
  `R1/R2/R3` 两两不交叠、无自配对、无序。
* 含环双连通块 = Tarjan 双连通分量中边数 ≥ 3 的块。桥自成一个 1 边块，永不进入 `R3`；spiro 两个环是
  两个块，跨环配对不进 `R3`；fused 环共享边，同一块；不连通分量不可能同块。
* **没有** `h_e <- aggregate(neighbors)` 形式的可学习更新：`c_e` 前向不被刷新，`T_s` 直接在图级做
  收缩，`index_add` 只出现在图级 pair 归约（一次），不是逐 bond 邻域聚合。
* Reader 与 P3 同深度（2 hidden + 输出层）同宽（256），SiLU，**无门控残差**。P3 的内容路径保留：
  Reader 第一层的前 321 列与 bias、以及后续所有层，直接取自 P3 的初始 reader；新增 75 列使用标准
  独立 `uniform(-1/sqrt(fan_in), +1/sqrt(fan_in))` 初始化。因此 P3 的 reader 是 SAC reader 在
  `T→0` 时的特例。

**参数**（脚本审计，非估算）：

| 组件 | 参数 |
|---|---|
| P3 主干（encoders + composer + reader） | 413,505 |
| `-` 被替换掉的 P3 reader | −148,481 |
| `+` SAC reader (396-256-256-1) | +167,681 |
| `+` 三个 96→24 投影（含 bias） | +6,984 |
| **SAC 总计** | **439,689**（其中新增 26,184，占 P3 的 +6.3%） |

500k 上限满足。P3 reader 参数被**移除**而非保留为死参数，因此不存在虚报。

## 3. 三个实验臂

| 臂 | 关系张量 | 参数 | 初始化 |
|---|---|---|---|
| `P3` | 无（原 BJOC-Null） | 413,505 | 冻结 P3 scratch init + 同 seed SSL |
| `SACR` | 真实 `R1/R2/R3` | 439,689 | 同一 P3 共享主干 init；SAC 新增参数由固定 generator 生成 |
| `SACU` | 内容归属 Null（见 §4） | 439,689 | 与 `SACR` **逐位相同** |

`SACR`/`SACU` 是同一个类、同一条前向代码路径，唯一区别是传进 `graph_tensors` 的
`(src, dst, weight)` pair 张量。

## 4. 内容 Null（`SACU`）

对每个 `s`，设真实关系对数 `m_s = |R_s|`、bond 数 `n`：

```
T_s^U = [m_s / C(n,2)] * sum_{e<f} (q_s(e) * q_s(f)) / sqrt(max(1, m_s))
```

若 `n < 2` 或 `m_s = 0`，张量严格为零。性质（全部由 CPU gate 断言）：

* 与 `R_s` 相同的总关系质量 `m_s`（权重和 = `m_s`）、相同对象/内容/参数/初始化；
* 无对角自连接、对 bond 槽位置换不变（只依赖内容多重集与 `m_s`）；
* 与真实 bond-pair ↔ 结构对应完全无关。

**必须写明的限制**：`SACU` 不是精确的 row-sum/degree-matched Null，也**不是等 FLOPs 对照**
（`SACU` 的 pair 数 = `C(n,2)`，约为 `R1+R2+R3` 的 1.5–5 倍）。因此 `SACR − SACU` **不能**单独
证明"真实连接归属"的纯因果作用；只能证明"在给定总质量与内容下，把质量放到真实结构对上是否更好"。

## 5. 数据与划分

* 数据：**只**使用标准 PyG `ZINC(subset=True)` official **train** 10,000 行。训练与 SSL 不使用
  任何外部 ZINC-Full 分子，**不加载** official valid / test（源码扫描 + LabelGuard 双重保证）。
* 剔除历史 4 行 SMILES/PyG 不一致（`941, 2210, 6071, 9766`），9,996 行；身份 = 由 PyG 图重建的
  canonical constitution SMILES（与 committed SMILES 的 identity 划分逐行相同）。
* **新盐** `cscl-bjoc-sac-v1`，显式传参（禁止 builder 默认值）。已断言：
  1. 实际参与 split 的 salt == 保存元数据中的 salt == 硬编码常量；
  2. 用元数据 salt 独立重算贪心划分，与保存的 fit/monitor/locked 行完全一致；
  3. 用历史盐 `cscl-bond-relation-attribution-v1` / `cscl-p3-bond-relation-bridge-v1` 能逐行复现
     split E / F（这正是上一轮 D-1 的回归测试）；
  4. 新区间的行哈希与历史 E/F 的任何区间**都不相同**；若相同则训练前停止；
  5. 换盐 probe 会改变划分（盐是 load-bearing 的）。
* 规模：约 7,996 / 1,000 / 1,000（±10 容差，按身份组贪心填充），身份组不跨区。
* **披露**：同一 10k 总体已被历史多轮使用。split S **不是**全新的盲测人群，只是一次新的划分。
  `split_compare_S.json` 记录新旧区间重叠矩阵与"曾被任何历史区间使用过的行比例"。
  因此本轮所有结果只能作为**同轮配对比较**证据，不能作为独立外部验证。

## 6. CPU 单测门禁（必须先全部通过才 freeze）

| gate | 内容 |
|---|---|
| G1 | 物理 bond 去重：PyG 正反边 == 2×无向键，bond 对象仅一份（400 Fit + 100 Locked 真实行） |
| G2 | R1/R2/R3 两两不交叠（真实行 + 合成图），记录覆盖率 |
| G3 | 环块语义：链（R3=0）、单环 C6（m=[6,6,3]）、spiro（2 块、跨块远距对存在但 R3=0）、fused（1 块、R3>0）、disconnected（不可达对存在、R3=0）、单键/双键退化 |
| G4 | 原子重编号 + 键端点交换 + batch 重排下预测不变（SAC 与 P3） |
| G5 | 同 seed 初始化：SACR/SACU 全参数哈希相同；P3 共享主干（encoders+composer）哈希与 SAC 相同；reader 前 321 列/后续层与 P3 逐位相同；参数 413,505 / 439,689 ≤ 500k |
| G6 | 梯度活性：projection / reader / atom / bond / junction encoders 全部非零（R 与 U 两种关系集） |
| G7 | **关系归属 witness**：保持 bond 内容多重集不变、只改变内容到结构位置的对映（非自同构），`T_R` 改变 ≥1e-3 而 `T_U` 不变 ≤1e-5 |
| G8 | 无消息传递：`c`/`z` 与 base 前向逐位相同；`q_s(e)` 只依赖自身 `c_e`；`T_s` 与显式双重循环参考一致；`index_add` 只出现在图级 pair 归约；禁用模块/令牌扫描为空 |
| G9 | 退化/极端小图数值稳定：单键、双键、三元环、空关系集合、`n<2` 时 U 严格零、无 NaN/Inf |
| G10 | 标签纪律：LabelGuard 正控；本轮代码无 official valid/test 加载路径 |
| G11 | 训练器等价：本模块的 P3 训练循环与冻结 `bjoc_perfmech_long_v1.train_supervised` 在 64 分子 2 epoch 下 **soup 逐位相同**、batch order 哈希相同 |
| G12 | U 张量恒等式：与规范公式（显式双重循环）在 3 个合成分子 × 3 关系上 max|err| ≤ 2e-5 |

> CPU witness 只证明表达力与实现正确性，**不是** GPU GO 的科学信号。

## 7. 训练配方（三臂完全相同）

* Fit-only masked atom+bond SSL：7,560 updates，batch 128，Adam 1e-3，wd 1e-5，clip 5，mask 15%/15%，
  RNG = `SeedSequence([seed, mol_id, visit])`。每 seed 一个 SSL checkpoint，三臂共享。
* 监督：raw-y L1，batch 128，R3 = 360 epoch cosine，Adam 1e-3→1e-5，wd 1e-5，clip 5，
  last-5 parameter soup。`steps_per_epoch = ceil(n_fit/128)`。
* 同 seed 下三臂 batch order 相同（`torch.Generator(seed + 101)` + `randperm`）。
* **不做**宽度/学习率/dropout/优化器/epoch 搜索；不追加 seed。
* GPU：`res` 的 GPU0 = seed 0，GPU1 = seed 1（各自串行 SSL → P3 → SACR → SACU）。
  预算上限 **4 GPU-hours**（先跑真实 batch smoke 实测吞吐，投影超限则停止）。

## 8. 冻结、评分与判定

1. **预测冻结**：28 个预测文件（3 臂 × 2 区 × 2 seed 的 own；SACR↔U 换核 2 × 2 区 × 2 seed；
   `T→0` 消融 2 臂 × 2 区 × 2 seed）先写盘并计算合并 sha256；
   同时冻结**换核敏感性**诊断（标签无关）。
2. **单次读标签**：之后才创建 `holdout_read.lock` 并读 Monitor+Locked 的 `y`，整轮只读一次。
3. **主指标**：raw-y MAE（Locked 为主判定区，Monitor 为复现区）。同时报告 RMSE、median AE、
   ordinary (`y > -3`) / tail (`y <= -3`)、top-10-removed、逐分子 paired bootstrap 95% CI
   （5,000 次重采样，分子为抽样单位）。
   **CI 语义限定**：它主要反映**固定两 seed 条件下**的分子抽样不确定性，不是跨训练 seed 的完整不确定性。
4. **预注册强正信号**（"值得进一步独立确认"，非"已达 0.06"）：

   | # | 条件 |
   |---|---|
   | S1 | `SACR − P3 ≤ −0.004`（seed 均值，Locked） |
   | S2 | `SACR − SACU ≤ −0.003`（seed 均值，Locked） |
   | S3 | 两个对比在**两个 seed 上方向一致**（均 < 0） |
   | S4 | 两个对比的分子级 paired 95% CI 上界 < 0 |
   | S5 | ordinary 与 top-10-removed 均不恶化（Δ ≤ 0） |
   | S6 | 已训练的 `SACR` 真实依赖关系张量：Locked 上换内容 Null 后 `mean|Δpred| ≥ 0.0025` 且 `max|Δpred| ≥ 0.02`（标签无关，冻结预测时已确定） |

5. **分类（有序、互斥）**：

   | case | 条件 | 解释 |
   |---|---|---|
   | A | S1–S6 全真 | 结构关系机制收益（仍需独立确认） |
   | B | `SACR−SACU` 达 GAIN 且复现，但 `SACR−P3` 不达阈值 | 关系组织有作用，模型性能不足 |
   | C | `SACR` 与 `SACU` 都优于 P3，但 `SACR−SACU` 不达阈值 | **一般二阶函数族收益**，非关系机制 |
   | E | `SACR−P3` 达阈值但不满足其它条件 | 有收益但无法归因 |
   | D | `SACR−P3` 不达阈值 | 无收益 |

   `GAIN` 的冻结定义（与 P3 轮一致）：seed 均值 ≤ −阈值、两 seed 同号、CI 上界 < 0、ordinary Δ ≤ 0。

6. **停止规则**：未达强正标准 → **不启动 full10k**、**不读 official test**、**不更换 rank / kernel /
   读出宽度 / 初始化**做挽救、**不追加 seed**、不自动进入下一轮架构搜索。
   达标准也只允许"申请下一阶段"，不等于 full10k/test 一定到 0.060。
   若 `SACR` 改善 P3 但不改善 `SACU` → 归类为一般二阶收益；若 `SACR` 改善 `SACU` 但不改善 P3 →
   关系组织有作用但模型不足。两者都不得表述为"真实连接归属的因果作用"。

## 9. 已知限制（预注册即披露）

1. `SACU` 不是等 FLOPs / row-sum 匹配对照；`SACR−SACU` 混合了"支撑选择 + per-pair 权重 + FLOPs"。
2. `R1/R2/R3` 的覆盖率依分子而变（链无 `R3`、稠环全覆盖）；`T_s` 的尺度由 `sqrt(m_s)` 归一，
   但分子间 `m_s` 分布差异可能引入容量上的异质性。
3. 关系张量虽然"图级"，但梯度会把 pair 耦合带回 `c_e`；这不是前向消息传递，也不等价于"无交互"。
4. 同一 10k 总体已被历史多轮使用（见 §5 披露）；本轮不是盲测。
5. Reader 的 75 个新列初始化与 P3 reader 不同（形状不同，不可能逐位相同）；P3 与 SAC 的对照因此
   包含"新输入列 + 新参数"的容量差异，而 `SACU` 是这一差异的严格同容量对照。
6. 两 seed 的 seed 间差异在 P3 轮已达到 0.0077（Locked 0.0872 vs 0.0795），对 <0.002 的效应没有
   分辨力；0.004/0.003 阈值已吸收这一点。

## 10. 交付

`results/sac_v1/`：`PROTOCOL.md`、`protocol_freeze.json`、`split_S.json`、`split_compare_S.json`、
`input_audit.json`、`objects_manifest_S.json`、`relations_manifest_S.json`、`relation_audit.json`、
`cpu_gates.json`、`staged_inputs_S.json`、`smoke/smoke_S.json`、`ssl_S_s*.json`、
`train_{P3,SACR,SACU}_S_s*.json`、`prediction_manifest_S.json`、`holdout_read.lock`、
`holdout_score_S.json`、`decision.json`、`per_molecule_errors_*.csv`、`post_hoc_S.json`、
`gpu_budget_ledger.json`、`evidence_manifest.json`、`predictions/*.npz`、
`DEVIATIONS.md`、`REPORT.md`（简洁中文科研判断）。
