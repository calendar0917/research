# SSRC Phase 1a protocol — same-information E₂ S vs X screening

Protocol ID: `ssrc-phase1a-lipo-v1`  
日期: 2026-10-19  
状态: **FROZEN_PRE_GPU**（本文件与同 commit 的代码/split/config 构成冻结；GPU 后不再改口径）  
上游: `notes/phase0_corrigendum.md`（纠错优先）, `notes/research_core.md`,
`results/phase0/REPORT.md`

> 本轮是**探索性 screening**，不是最终显著性检验，也不是“消息传递是否必要”的
> 最终回答。任何结果都不能直接证明或否定整个 SSRC 方向。

---

## 1. 唯一研究问题

> 在输入精确相同的 \(E_2\) rooted-patch 多重集时，跨 patch 的学习式交互
> 是否优于仅进行局部共享编码和图级池化？这种优势是否在小样本下更明显？

三个模型：

| 模型 | 输入 | 计算 | 角色 |
|---|---|---|---|
| **S** | 每个原子的精确 radius-2 rooted patch；patch 内节点/边原始属性 + root flag + 到根距离 | 共享 2 层 GINE（patch 内）；逐 token 残差 FFN；mean pooling；统一回归头 | 无跨 patch 交互 |
| **X** | 与 S **完全相同**的 patch 张量与 mask | 与 S 相同的局部 GINE 与回归头；patch token 之间 2 层全局 self-attention；无位置编码 | 同信息、跨 patch 交互 |
| **G** | 原始分子图 + `edge_index` + 原始原子/键属性 | 2 层 GINE；mean pooling；统一回归头 | 应用性能参照，信息访问不同 |

**S 不是“完全没有消息传递”**：它在每个 patch 内部有局部 GINE。它只是没有
跨 patch 的 attention/迭代/incidence。**X 仍然只是 \(E_2\) 多重集的函数**，
不能突破真正的 \(E_2\) 碰撞。

### 1.1 信息一致性论证

- S/X 的 batch 只含 `patch_node_feat`, `patch_edge_index`, `patch_edge_attr`,
  `patch_batch`, `patch_to_mol`, `patch_pos`, `patch_mask`, `mol_n_patches`,
  `y`, `mol_index`；没有原图 `edge_index`、patch 间 adjacency、root 原子编号。
- `patch_pos` 只用于把 token 放进 `[B, max_patches, d]` 存储张量；不进入任何
  特征，也不加位置编码。X 的 attention 对 token 置换等变，pooling 置换不变。
- S/X 的局部编码器和回归头使用**同一 seed、完全相同的 state_dict 初始化**；
  X 只额外拥有 attention 栈，S 只额外拥有逐 token FFN。
- S 与 X 的参数预算：**59,011 vs 59,075**（S/X = 0.9989，差 0.11%）。
  - S 的逐 token 残差 FFN 是真实计算（2 个 64→128→64 残差块），
    不是无作用的参数装饰。
- 固定 64 分子 batch 的 forward FLOPs（`FlopCounterMode`，PyG scatter 可能未计入，
  是下界）：S 622.3M，X 877.7M，G 170.5M。**X 的 FLOPs 约是 S 的 1.41×**；
  这是跨 patch attention 的真实计算代价，必须如实报告，不能藏起来。若 X 胜出，
  “更多算力”是未排除的替代解释之一；若 S 胜出，则 X 在更多算力下仍未胜出。
- G 的宽度固定为 96、2 层，参数 53,379（约为 S/X 的 0.90×），是一次性容量匹配
  选择；它是真实容量，不是 dummy 参数。G 的 FLOPs 明显更低（170.5M），
  因此 \(G-S\)/\(G-X\) 只作应用参照，不作信息或算力因果分解。

---

## 2. 数据与冻结 split

### 2.1 来源

- Lipophilicity（MoleculeNet / DeepChem 重分发），4200 行；
  CSV sha256 `aed41590cb30609d51d8e08ad3ff06495a76e80e211358801f596b10da69bacd`。
- 结构输入：committed label-free raw graph
  `tracks/structure-contribution/results/bjoc_cross_task_lipophilicity_v1/dataset/lipo_raw_v1.json.gz`
  （sha256 `8416837ea3e56e3e5218865391c88913bb16948b48b7a61be1ee890ea4dda6dd`）。
  GPU 主机不需要 RDKit。
- 官方 split：DeepChem 2.8.0 scaffold，train/valid/test = 3360/420/420；
  本协议只使用官方 **train 3360 行**建立内部 Fit/Monitor/Reserved。
- 官方 valid/test 标签**不读**；内部 Reserved 标签**不读**。
- GPU 主机只使用提交的 `results/phase1a/labels_fit_monitor.json`
  （sha256 `0260a390…`）：恰好 2,856 条 Fit+Monitor 标签，无 Reserved、
  无官方 valid/test。它由 `freeze_labels.py` 从 CSV 流式生成；
  生成时 Reserved/valid/test 标签不进入输出映射。

### 2.2 内部 split（structure-only, label-blind）

| 分区 | 行数 | 用途 |
|---|---:|---|
| Fit | 2352 | n=2352 训练；n≈250 子集也从中选 |
| Monitor | 504 | 本轮探索性比较/早停 |
| Reserved | 504 | **本轮完全不读标签、不评分** |

冻结文件：`results/phase1a/split_phase1a.json`（sha256
`9fa96d49efd94cd2ff381922f83a07f963d832064c75469657b4ce5f68c332b6`）。

| 分区 | row-index sha256（Phase 0 comma 口径） | scaffold 组数 |
|---|---|---:|
| Fit | `c474e38cdecf10071965527409901403ec8b670b96fe4a3c42452531dd0210e9` | 725 |
| Monitor | `fe81cc7904a6da9378f5bf5ae7452ac71629adad83a2a98b917f87299a85f39f` | 421 |
| Reserved | `bc28c5d37c2fe2fbe39c01d5bf2ab9c56630eee2d0a4454a108d150ab2f42d86` | 422 |

- 两组 scaffold 重叠：Fit∩Monitor = 0，Fit∩Reserved = 0，Monitor∩Reserved = 0。
- 复算与 Phase 0 candidate **逐行一致**（`phase0_candidate_check` 全 true）。
- 这不是重新筛选多个 split；就是 Phase 0 已提出的 2352/504/504 候选，
  经结构-only 复算后冻结。

### 2.3 n≈250 子集（已冻结）

- 规则：只从 Fit 的 scaffold 组中选；为免一个超大 scaffold 主导小样本，
  只考虑组大小 ≤ 10 的组；按 scaffold 字符串排序后用
  `random.Random(20261018)` 洗牌；按整组加入直到 ≥250。实际 **n=253**，
  114 个 scaffold 组。
- 行号 sha256 `81a3000c9b5116911e1c8dfb2505ce4a7a0b34022784bed13d8b82af897e0df3`。
- 所有模型在同一 n 使用**完全相同**的 253 行；标签只用于训练/标准化。

### 2.4 历史盲性披露

Phase 0 曾对**全部 3360 个官方 train 标签**做过探索性 LOO/置换诊断。
因此本轮内部 Reserved 不能宣称“从研究一开始就完全盲”。Phase 1a 的纪律是：

- 不读取 Reserved 的逐行标签；
- 不计算 Reserved 的任何分数/指标；
- 不在任何模型选择、早停、报告主表中使用 Reserved；
- Phase 1a 的标签读取器只保留 Fit/Monitor 行，Reserved 标签不进入返回对象
  （CPU gate `label_isolation` 记录 `reserved_labels_retained=0`）。

---

## 3. 模型与配方（一次性冻结）

### 3.1 S

- 输入 patch：radius 2；节点特征 = 原子原始 one-hot（15 类 + UNK，
  17 槽）+ charge（7）+ H（7）+ chirality（6）+ root flag（1）+ 距离
  one-hot（3）= **41** 维；边特征 = bond class（6）+ stereo（6）= **12** 维。
  不使用 ring flag、全图距离、global 描述符或原子编号。
- 局部编码：`Linear(41→64)` + 2 层 GINEConv（hidden 64，edge_dim 12，
  train_eps）+ 每层 residual + LayerNorm + dropout 0.1；patch token =
  patch 内节点 mean。
- 逐 token 残差 FFN：2 个 `LayerNorm → Linear(64→128) → SiLU → Dropout →
  Linear(128→64)` 残差块；无 token 间交互。
- mean pooling over patch tokens；回归头 = `LayerNorm → Linear(64→64) →
  SiLU → Dropout → Linear(64→1)`。
- 参数 **59,011**。

### 3.2 X

- 与 S 相同的 patch 张量、局部 GINE、回归头、初始化。
- 2 层 `TransformerEncoderLayer`（d=64，4 heads，FFN 128，GELU，
  `norm_first=True`，dropout 0.1），`src_key_padding_mask=~patch_mask`；
  **无位置编码**。
- masked mean pooling over valid tokens；同一回归头。
- 参数 **59,075**（S/X=0.9989）。

### 3.3 G

- 输入：原始分子图节点特征（37 维原子原始属性）+ 边特征（12 维）；
  `edge_index` 双向。
- `Linear(37→96)` + 2 层 GINEConv（hidden 96，edge_dim 12，train_eps）+
  residual + LayerNorm + dropout 0.1；mean pooling over nodes；回归头宽度 96。
- 参数 **53,379**；FLOPs 170.5M（fixed 64-molecule batch，下界）。
- 只作应用参照；不作信息价值分解。

### 3.4 统一训练配方（全部模型相同）

| 项 | 值 |
|---|---|
| loss | L1（MAE） |
| optimizer | AdamW，lr 1e-3，weight decay 1e-5 |
| schedule | cosine，T_max=200，eta_min=1e-5 |
| batch | 64 |
| max epochs | 200 |
| early stop | Monitor MAE，patience 30，min_delta 1e-4 |
| grad clip | 5.0 |
| dropout | 0.1 |
| seeds | 0, 1, 2 |
| n | 253（冻结子集）与 2352（完整 Fit） |
| 数据顺序 | 同 n/seed 的所有模型共用 `default_rng([seed, epoch])` 的同一 permutation |
| 目标标准化 | 只用当前训练行的 mean/std；Monitor 用同一变换 |
| 不做 | LR/深度/结构编码/SSL/架构搜索；不做结果驱动追加种子 |

### 3.5 运行矩阵与预算

3 模型 × 2 n × 3 seeds = **18 条轨迹**。  
累计 GPU 预算上限 **4 GPU-hours**（单卡占用时间之和，不是墙钟）。
若接近预算，停止并报告已完成的有效结果；不追加 seed/架构。

---

## 4. CPU 门禁（GPU 前，全部通过才允许 GPU）

2026-10-19 本地运行：`uv run python .../phase1a_gates.py` → **8/8 PASS**
（`results/phase1a/cpu_gates.json`；JSON 为本地证据）。

| # | 门禁 | 结果 |
|---|---|---|
| 1 | split 2352/504/504、small=253、scaffold 组不重叠、Phase 0 candidate 逐行一致 | PASS |
| 2 | label vault 只保留 Fit/Monitor；Reserved 标签 retained=0；错误 purpose 被拒 | PASS |
| 3 | C24/caterpillar/grid 的 \(E_2(d6)=E_2(d9)\) 精确成立；S/X 对碰撞类预测差 ≤ 6e-8 | PASS |
| 4 | patch 顺序、原子重编号、patch 内边顺序置换：S/X 输出差 ≤ 6e-8 | PASS |
| 5 | S/X batch key 集合精确；注入 `full_edge_index` 被 guard 拒绝；G batch 不含 patch 字段 | PASS |
| 6 | S/X 共享局部编码器/头 state_dict 逐 tensor 相同；同 seed 可复现；数据 permutation 可复现 | PASS |
| 7 | 32 样本 × 3 模型 forward/backward + 2 epoch 训练；loss 有限、梯度非零 | PASS |
| 8 | 参数/FLOPs/预处理成本已实测；S/X 参数比 0.9989；FLOPs 差异如实记录 | PASS |

GPU 门禁还要在 `res` 上实测 **GPU 峰值显存、实际 step 时间、实际 GPU 秒数**，
并写入每条 run JSON；这不改变 CPU 门禁的通过状态，但必须报告。

任何关键输入一致性或数据隔离门禁失败 → 停止 GPU，提交修复报告。

---

## 5. 主端点与分析计划

### 5.1 主比较

\[
\mathrm{Adv}_{\rm set}(n):=\mathrm{MAE}_S(n)-\mathrm{MAE}_X(n)
\]
正值 = 同信息下 X 更好。

关注：

1. \(\mathrm{Adv}_{\rm set}(253)\)
2. \(\mathrm{Adv}_{\rm set}(2352)\)
3. 小样本优势：\(\mathrm{Adv}_{\rm set}(253)-\mathrm{Adv}_{\rm set}(2352)\)

每个量报告 3 个 seed 的逐 seed 值、seed 均值、seed 范围；不把 seed 当独立数据
复制。主指标 = Monitor 上 best-monitor MAE（早停选中）；同时报告 last-epoch
MAE 作为对早停乐观偏差的敏感性检查。

### 5.2 辅助比较

\[
\mathrm{MAE}_S-\mathrm{MAE}_G,\qquad
\mathrm{MAE}_X-\mathrm{MAE}_G
\]
只用于判断与完整图 GINE 的应用性能差距；**不**直接作信息价值/因果分解。

### 5.3 不确定性

- 单 scaffold split、3 seeds、Monitor 用于早停；不是独立终局确认。
- 分子级 bootstrap 按 **scaffold 组** cluster（Monitor 421 组），
  5000 次；同一 resample 同时计算 S/X，保持配对。
- 对交互项，固定 seed 均值后对 Monitor scaffold 组做 cluster bootstrap；
  这捕获 Monitor 抽样不确定性，**不**捕获训练集/scaffold split 不确定性。
- 任何 CI 都附带上述条件。CI 跨零只表示“不确定”，**不**表示等价。
- 不做等效性检验，因此不写“等价/无差异”；只写“本 pilot 未能区分”。

### 5.4 实际意义阈值（预先固定）

- \(|Adv| < 0.02\) MAE：本 pilot 视为**不具实际意义**的微小差异；
- \(0.02 \le |Adv| < 0.05\)：弱/不确定；
- \(|Adv| \ge 0.05\)：可能具有实际意义，但仍需看 seed 一致性与 CI。
- 这些阈值是探索性判断辅助，不是正式显著性/等效性边界。

---

## 6. 判定规则（预先固定）

- **S/X 均未学到基本有效预测**（Monitor MAE 不优于“训练均值常数预测器”，
  或 R²≤0 且训练/监控曲线平坦）→ 先排查实现和优化；正确仍不行则停止该
  pilot，不推断机制不存在。
- **X 在 3/3 seeds 稳定优于 S，且小样本效应具有实际意义**
  （\(\mathrm{Adv}_{\rm set}(253)>0\) 在 3/3 seeds；均值 ≥0.02；
  交互 \(\mathrm{Adv}(253)-\mathrm{Adv}(2352)>0\)；CI 与实际意义阈值支持）
  → `PROMISING_SAME_INFORMATION_INTERACTION`。下一轮设计独立 split 或第二任务
  确认；**不自动做新架构**。
- **S≈X，G 较好** → `UNRESOLVED_INFORMATION_VS_INDUCTIVE_BIAS`；
  不能声称全图信息是原因。若继续，下一轮加固定的一跳/边 incidence 信息，
  用信息访问匹配的非迭代与迭代对照。
- **S≈X≈G** → `NO_CLEAR_INTERACTION_VALUE_IN_THIS_PILOT`；不自动扩大模型/数据集；
  先判断是否存在能改变结论的未测因素。
- **X 只在更大 n 上优于 S，或结果与 seed 强相关** → 如实报告，
  不硬解释成样本效率收益。
- 一轮结果不能直接证明或否定整个 SSRC 方向。

---

## 7. 不做什么

- 不训练旧草案的 \(S_{\rm bag}\)、\(S(G)\) 或五臂 oracle 分解。
- 不做新增结构编码家族搜索、SSL、外部数据、ZINC 搜索。
- 不读官方 valid/test、不读本轮 Reserved、不用它们做任何选择。
- 不因为结果不理想而改模型/口径/阈值；不做结果驱动的追加种子。
- 不把 \(G-S\)/\(G-X\) 解释为纯信息增益。
- 不把“差异不显著”写成“等价”。

---

## 8. 复现命令

```bash
# 1) 重新生成/核对冻结 split（structure-only；不读标签）
uv run python tracks/structural-sufficiency/code/phase1a/freeze_split.py

# 2) CPU 门禁（8/8）
uv run python tracks/structural-sufficiency/code/phase1a/phase1a_gates.py

# 3) 单元测试
uv run pytest -q tracks/structural-sufficiency/code/phase1a/tests

# 4) 单条轨迹（本地 CPU smoke）
uv run python tracks/structural-sufficiency/code/phase1a/run_phase1a.py \
  --model S --n small --seed 0 --device cpu --epochs 2 --patience 1

# 5) GPU：18 条轨迹，经 rr 在 res 上执行（示例）
rr deploy res
rr run res ssrc-p1a-S-small-s0 --pool res-gpu0 --result tracks/structural-sufficiency/runs/phase1a \
  -- python tracks/structural-sufficiency/code/phase1a/run_phase1a.py \
     --model S --n small --seed 0 --device cuda \
     --out-dir tracks/structural-sufficiency/runs/phase1a
```

---

## 9. 冻结清单

| 项 | 值 |
|---|---|
| split | `results/phase1a/split_phase1a.json` sha256 `9fa96d49…` |
| raw graph | sha256 `8416837e…` |
| labels artifact | `results/phase1a/labels_fit_monitor.json` sha256 `0260a390…`（Fit+Monitor only） |
| data code | `phase1a_data.py` sha256 `4fa0e432…`（label artifact 支持） |
| model code | `phase1a_models.py` sha256 `160c050a…` |
| train code | `phase1a_train.py` sha256 `0c5c6738…` |
| runner | `run_phase1a.py` sha256 `219495af…` |
| gates | `phase1a_gates.py` sha256 `5e3066b1…` |
| config | `configs/phase1a.yaml`（与本文一致） |
| CPU gates | 8/8 PASS；S/X params 59,011/59,075；FLOPs 622.3M/877.7M/170.5M |

> 冻结 commit：包含本文件与上述代码/split/config 的 commit。GPU 结果的
> `git_commit` 字段必须与该 commit 一致；若不一致，结果作废并重新部署。
