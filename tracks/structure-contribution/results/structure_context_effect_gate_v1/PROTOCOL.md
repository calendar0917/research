# PROTOCOL — structure-context-effect-gate-v1 (Stage B)

**轨道**: `tracks/structure-contribution/`
**阶段**: Stage B —— 固定分子预测器上的"上下文条件化结构响应"验证（gate，不是新方法轮）
**性质**: 预先注册（pre-registered）。所有分析对象定义、干预定义、参考方案、特征块、
模型类、交叉验证方案、统计量、对照与判断规则在本文件中冻结；`REPORT.md` 只报告本协议
声明的统计量并据其判断。本文件在**读取任何模型干预响应之前**写入磁盘。

**边界（整个 Stage B 不变）**

- 只使用冻结的 Lipophilicity train-only 结构对象（`objects_train.pt`，3 360 条）与
  `bjoc_cross_task_lipophilicity_v1` 已训练好的 **Arm P（BJOC-Lipo-SSL）soup seed 0/1**
  检查点；
- **不读取**任何 Lipophilicity valid/test 标签、不读取 ZINC official test / Locked、
  不读取任何性质的 valid/test 量；
- **不训练、不更新任何模型权重**；模型以 `eval()` 冻结模式在 CPU 上做前向；
- 不新建解释网络、不做神经架构搜索、不做字典/上下文半径/数据集的重新搜索；
- 所有结果只称为"**固定模型的 Bond 来源替换响应**"，不称为真实化学贡献或因果效应；
- 不修改原 Stage A / BJOC / CSCL / HardConcept 的冻结产物，不修改
  `notes/research_core.md`。

---

## 0. 问题

Stage A（`structure_context_support_v1`）确认：同一真实局部结构 `S` 在不同 Lipophilicity
train-only 分子中有独立支持的、经组成/尺寸/环数控制后仍存在的上下文 `CT`（C4，93 个类型）。

Stage B 检验：

> 对一个**冻结**的分子性质预测器，同一 `S` 的**模型内部响应**能否被真实上下文关系稳定解释，
> 并在控制组成和非线性读出之后保留增量价值？

三个竞争解释（在同一固定响应目标上比较）：

- **B0 — Structure-only**：只允许 `S` 身份（及训练组中估计的该 `S` 的典型响应）；
- **B1 — Structure + global controls**：允许 `S`、分子尺寸、组成、环数与固定全局特征；
- **B2 — B1 + real context**：在 B1 的全部输入上，额外使用 Stage A 已冻结的外层结构与关系上下文。

---

## 1. 冻结输入

| 角色 | 路径 | sha256（运行时校验） |
| --- | --- | --- |
| Stage A 实例清单 | `results/structure_context_support_v1/analysis/instances.npz` | `80970087965952baf63821e965418380aca88fc0463286bb3ff6bef10175d327` |
| Stage A 总结 | `results/structure_context_support_v1/summary.json` | （运行时记录） |
| Lipophilicity train 结构对象 | `results/bjoc_cross_task_lipophilicity_v1/objects_train.pt` | `6d681cb0eaaab8c448564993c3740b24204e52e9e430f8c366189aa0061c7c86` |
| 冻结 SSL 模型 seed 0 | `results/bjoc_cross_task_lipophilicity_v1/soup_P_s0.pt` | `1a8292acf0477ce89899d48d913f0c50d651eab00c4d92d757580f847836a304` |
| 冻结 SSL 模型 seed 1 | `results/bjoc_cross_task_lipophilicity_v1/soup_P_s1.pt` | `505f840ea90783bdbd6761e2290e4f875670bcdf3f293a8aa5d9225582c542b0` |
| 全局特征标准化 | `results/bjoc_cross_task_lipophilicity_v1/standardizer_v1.json` | （运行时记录） |
| 目标量纲常数（train-only） | `train_P_s0.json` 的 `target_normalization`（mean 2.162904761904762 / scale 1.210992810122257） | （运行时记录） |

模型配置固定为 `bjoc_lipophilicity_adapter_v1.LipoConfig()`（419 713 参数，`organization_mode=NULL`，
见 `train_P_s0/1.json` 的 `parameter_audit`）。运行时校验参数量与参数字典哈希；在 valid 结构上
重放预测，要求与冻结的 `predictions/valid_P_s{0,1}.npz` 的最大绝对差 ≤ 5e-6 且文件 sha256 与
`prediction_manifest_valid.json` 一致（不读取任何标签）。

`labels_loaded=false`、`valid_test_labels_loaded=false`、`weights_updated=false` 写入 `summary.json`。

## 2. 局部结构 `S` 到 BJOC 化学键的对应（必须先验证）

Stage A 的实例以（`mol_uid` = `objects_train.pt` 中的位置，`u, v` = RDKit SMILES 解析后的重原子
索引，锚点键）给出。BJOC 的物理 Bond 由 `cu.unique_undirected_bonds` 生成：每条物理键一次、
`u < v`、按 `(u, v)` 排序。对应验证（对全部 3 360 个 train 分子）：

1. `build_lipo_objects` 的 `n_atoms` == RDKit 重原子数；
2. 模型 `bond_index` 的键集合 == RDKit 键集合（无序端点对）；
3. 键型一致：RDKit `SINGLE/DOUBLE/TRIPLE/AROMATIC` ↔ 模型 `bond_type` 0/1/2/3（芳香一致）；
4. 每个 Stage A 实例的 `(u, v)` 在该分子的 `bond_index` 中恰好出现一次。

任何一项失败即停止（`NO_JUSTIFIED_CONTEXT_EFFECT_METHOD`，实现失败），不进入响应计算。
验证结果写入 `analysis/bond_correspondence.json`。

## 3. 固定分析对象（在本文件写入后、读响应前生成并冻结）

**主分析数据集**：Stage A `summary.json::datasets.lipo__T1__primary.decision_stats.exact_stratum_type_keys`
列出的 **93 个 C4 类型**（在 `(n_heavy, 重原子分子式, n_rings)` 完全相同的分子之间仍有 ≥2 个各由
≥2 个分子独立支持的上下文）。

**分析池定义**（只使用 Stage A 实例清单、SMILES、以及 RDKit 重算的分子级分层键）：

对每个 C4 实例 `i`（键 `(u, v)`，类型 `s`，上下文 `ct`，分子 `m`）：

- 分层键 `stratum(m) = (n_heavy, heavy_formula, n_rings)`（与 Stage A `exact_stratum` 完全一致；
  `heavy_formula` 用与 Stage A 相同的 `_heavy_formula` 规则重算，并与 `instances.npz` 的
  `mol_scaffold` 交叉校验）；
- 在 `(s, stratum)` 组内，称 `ct` 是**独立支持的**当且仅当该 `ct` 在该组内被 **≥2 个不同分子**支持；
- 保留组内满足"**≥2 个独立支持上下文**"的所有实例，且实例的 `ct` 属于这些独立支持上下文；
- 其余实例（所在组不满足，或 `ct` 只有 1 个分子支持）**不进入分析池**。

**去重单位**：主单位 `u = (mol_uid, s_key, ct_key)`。
- 同一分子中的多个 `S` 实例不得当成独立观测；
- 同一单位内多个键的响应取算术平均（若 `S` 与 `ct` 精确相同，这些键在模型输入上同构，响应
  应一致；脚本记录单位内响应标准差作为进一步证据）；
- 单位权重 = 该单位包含的键实例数，在描述统计中报告；预测评估不重复加权。

**结构支持门槛（先于响应固定）**：
- 分析池独立骨架数 ≥ 30；
- 有"跨骨架上下文对比"的类型数 ≥ 10（即同一类型下存在两个独立支持上下文，其骨架集合不相交）；
- 满足"≥2 个独立支持上下文、且其中 ≥2 个各由 ≥2 个骨架支持"的 `(s, stratum)` 组数 ≥ 30。

若任一门槛不满足：只报告统计能力不足（`CONTEXT_EFFECT_INCONCLUSIVE`，理由=支持不足），
不进行任何字典/上下文半径/数据集搜索，不通过放宽定义制造结果。

分析池在响应计算前写入 `analysis/analysis_pool.npz` 与 `analysis/pool_manifest.json`（含
sha256、C4 类型键、计数、门槛检查结果）。

## 4. 固定模型的 Bond 来源替换响应

### 4.1 干预位置与传播

对分析池中的每个键实例 `(m, b)`（`b` 为模型物理 Bond 索引）：

1. 用冻结模型前向计算自然编码 `b_b^nat = encode_bonds(a(m))[b]`；
2. 把 `b` 位置的 Bond 编码替换为预定义参考编码 `d`（见 §4.2），**其余一切输入不变**：
   - 保留原始分子、原子编码、全部未干预 Bond 编码、所有原始特征；
   - 重新计算 `w = encode_junctions(a, b')`（只依赖各 Junction 自身涉及的两条 Bond 编码）；
   - 重新计算 `W, V, n_b = group_to_bonds(w, mode=NULL)`（模型自身的全局分组）；
   - 重新计算 `c = group_composer([b', W, V, log1p(n_b), bond_feat])`；
   - 重新计算 `Z` 与 `reader_in`，读出预测；
3. 响应 `R(m, b) = f(intervened) − f(natural)`（模型归一化输出尺度；报告时另换算 raw-scale，
   用冻结的 train-only target normalization 常数）。

实现不得修改冻结模型代码，而是在本目录脚本中复制同一前向路径并加显式覆盖；必须满足：

- 无覆盖时 `forward_with_overrides` 与 `model.forward` 逐位一致（差值 ≤ 1e-7）；
- 用该键自己的自然编码覆盖时 `R == 0`（≤ 1e-9）；
- 覆盖后 `b'[b] == d`，未干预 Bond 的 `b` 不变；
- 只有与该键共享 Junction 的 `w` 行改变，其余 `w` 行不变；
- 覆盖目标键后，`W/V` 与 `c` 只在模型定义下变化，不做任何"在 reader 输入上减去一项"的操作。

**局部性记录**：由于冻结模型 `organization_mode=NULL`，单键替换会经由
`W_b = n_b · mean_G(w)` 的全局均值通道影响所有 Bond；这是模型语义的一部分，报告中如实说明，
不把它当作"单键局部贡献"。

### 4.2 参考编码（三套，全部与标签、与响应无关）

参考编码的**选择规则**只使用结构与冻结模型的自然编码，不读取任何响应，不使用任何性质标签：

- **REF-A（主口径，observed type-anchor）**：按物理键型 `c ∈ {SINGLE, DOUBLE, TRIPLE, AROMATIC}`
  在全部 3 360 个 train-only 分子的自然 Bond 编码上取类均值 `μ_c`，取类内 L2 距离最小者
  `d_c = argmin_j ||b_j − μ_c||₂`（并列时取 `(mol_uid, bond_id)` 最小者）。`d_c` 是真实观测到的
  自然编码。
- **REF-B（class-mean）**：`d_c = μ_c`（自然编码的类均值；分布内参考，与既有
  `bjoc_source_attribution_feasibility_v1` 的 μ 参照语义一致，但不是单个观测编码）。
- **REF-C（random natural donors）**：对每个键实例，从**同类键且不属于目标分子**的 pool 中，
  用固定种子 `SeedSequence([20261017, mol_uid, bond_id, c])` 抽 `J = 8` 个自然编码，
  响应取 8 次独立干预响应的均值；编码距离取 8 次的均值（并记录 min/max/std）。

每套参考都记录：编码距离 `||d − b^nat||₂`（及 cosine）、响应大小、符号。

### 4.3 正确性/反证检查（全部写入 `summary.json`）

1. **恒等控制**：对全部键实例用自身编码替换，响应应为 0（最大 |R| ≤ 1e-9）；
2. **参数量与哈希**：模型参数量、state_dict sha256 与冻结清单/运行记录核对；
3. **预测重放**：valid 结构上的预测文件与冻结 manifest 的 sha256 一致；
4. **跨 seed 响应一致**：同一单位、同一参考下 seed 0/1 的响应 Spearman 相关与符号一致率；
5. **跨参考一致**：A/B/C 之间响应的 Spearman 相关与符号一致率；
6. **单位内一致性**：同一 `(m, s, ct)` 内多键响应的标准差；
7. **上下文确实超出 `S`**：B2 的 CT 特征在**同一 `S` 内**存在非零方差（逐特征报告），
   并在 §5 的构造上确认这些特征只来自 `S` 之外的外层/关系轴。

## 5. 三个竞争解释 B0 / B1 / B2（同一目标、同一模型类、同一 CV）

**预测目标**：主口径为 REF-A 下每个单位 `(m, s, ct)` 的**有符号响应** `R`（模型归一化尺度）。
REF-B、REF-C 下重复全部流程作为参考稳定性检验；绝对响应 `|R|` 只作描述性报告。

**单位级特征块**（全部只由结构/冻结模型自然编码定义，不含任何标签）：

- `X_S`：93 个 C4 类型 one-hot（按单位数降序、再按键字典序固定顺序）；
- `X_G`（全局控制）：`[n_heavy, log1p(n_heavy), n_bonds, n_rings, n_ring_systems, n_arom_rings,
  cyclomatic = n_bonds − n_heavy + n_components]`、重原子元素计数（`C,N,O,S,P,F,Cl,Br,I,B,Si`
  共 11 维）、冻结模型本身使用的 38 维 `global_feat`；
- `X_CT`（真实上下文，只在 B2 使用）：外层壳统计（patch 之外、与 patch 直接相连的原子的
  原子类计数 15 维 + 总数 1 维；patch→外层键的键类计数 4 维；外层节点外部度直方图 4 维）、
  环关系轴（锚点是否在环、锚点最小环大小分桶、过锚点环数、锚点所属环系原子数，共 4 维）、
  杂原子关系轴（patch 到外部杂原子的最小距离、≤2 距离的杂原子数，共 2 维）、
  `ring_dist`（1 维）。`X_CT` 是 Stage A `outer_key` + 三轴关系的确定性粗化，不包含 `S` 内部信息。

**模型类**（B0/B1/B2 完全相同、容量相同，固定方案，不做架构搜索）：
`sklearn.ensemble.HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_depth=3,
min_samples_leaf=20, l2_regularization=1.0, random_state=0, early_stopping=False)`。
- B0：输入 `X_S`；
- B1：输入 `[X_S, X_G]`；
- B2：输入 `[X_S, X_G, X_CT]`。

**交叉验证**（主口径：按 Murcko 骨架分组；敏感性：按分子分组）：
`GroupKFold(n_splits=5)`，groups = 骨架（主）/ 分子（敏感性）。同一分子的多个 `S` 实例、
同一骨架的全部单位始终落在同一折。所有指标在 out-of-fold 预测上汇总。

**描述性上界**：`B2x = B1 + ct_key 的 out-of-fold 目标编码`（训练折内估计、平滑 k=10、
未见 CT 回退到训练折全局均值），只用于回答"更高分辨率的精确 CT 身份是否有额外价值"，
不参与主判断。

**指标**：MAE（主）、RMSE、OOF R²。配对改善 `Δ = MAE(B1) − MAE(B2)`，`>0` 表示 B2 更好。
不确定度：按骨架的 cluster bootstrap（10 000 次重采样，重采样固定 OOF 预测）给出
`Δ` 的 95% 分位区间；并报告逐骨架改善比例、逐类型 `Δ` 分布。

## 6. 关键反证（全部预注册）

1. **CT 打乱对照**：在每个 `(s, stratum)` 组内随机置换单位间的 `X_CT` 行（保持样本量与组内
   特征多重集），重训 B2（B1 不变），同一 CV，得到 `Δ*`；重复 `n_shuffle = 100` 次。
   经验单侧 p 值 `p_shuffle = (1 + #{Δ* ≥ Δ_obs}) / 101`。
2. **高频结构主导**：按类型报告 `Δ_s`；报告整体改善中前 1/3/5 高频类型单位占比；
   预注册"去掉分析池中单位数最多的前 3 个类型"后重跑 B1/B2 的 `Δ`。
3. **非独立单元主导**：按骨架报告改善比例；预注册"去掉分析池中分子数最多的前 3 个骨架后"
   重跑 B1/B2 的 `Δ`。
4. **编码距离解释**：`B1d = B1 + [编码距离及 log 编码距离]`，比较 `MAE(B1d) − MAE(B2)`
   与 `Δ`；并报告单位级响应与编码距离的 Spearman 相关。
5. **符号对参考的敏感性**：A/B/C 与 seed 0/1 之间的符号一致率与 Spearman 相关；
   预注册"稳定"阈值：全部（参考×seed）两两对中，
   最小 Spearman ρ ≥ 0.5 且最小符号一致率 ≥ 0.7（`|R| < 1e-6` 视为零符号）。
6. **上下文是否超出 `S`**：逐特征报告同一 `S` 内的方差比例（非零方差特征数 / 总特征数），
   并给出仅用 `X_CT` 的 B2-only 描述性结果（不含 `X_S`/`X_G`），用于确认 CT 信息真实存在。

## 7. 判断规则（预先冻结）

记 `Δ(S, M)` = REF-`S`、模型 seed `M` 下骨架分组 OOF 的 `MAE(B1) − MAE(B2)`；
`CI_lo(S, M)` = 骨架 bootstrap 2.5% 分位；`p_shuffle(S, M)` 同 §6.1（主口径 `S=A`）。

- `response_defined` = §4.1/§4.3 的 1–4 项实现检查全部通过（含恒等控制、对应验证、
  预测重放、跨 seed 中位相关 > 0.5 且符号一致率 ≥ 0.7）。
- `reference_stable` = §6.5 的阈值满足。
- `improvement(M)` = 三个参考方案中**至少 2 个**满足 `Δ > 0` 且 `CI_lo > 0`。
- `both_seeds` = `improvement(0)` 且 `improvement(1)`。
- `null_exceeded` = 主口径 `Δ(A, M)` 对两个 seed 都超过各自 shuffle null 的 95% 分位。
- `nontrivial` = 三个参考中至少 2 个满足 `Δ ≥ max(0.02 · MAE_B1, shuffle null 95% 分位)`。
- `support_ok` = §3 的三个结构支持门槛全部满足。

决策（按顺序，第一条命中即为结论）：

```
1. 若 response_defined 失败（干预/对应/重放不可靠）        -> NO_JUSTIFIED_CONTEXT_EFFECT_METHOD
2. 若 reference_stable 失败（响应对参考定义非常不稳定）    -> NO_JUSTIFIED_CONTEXT_EFFECT_METHOD
3. 若没有任何 (S, M) 组合满足 Δ>0（B2 不优于 B1）         -> NO_JUSTIFIED_CONTEXT_EFFECT_METHOD
4. 若不满足 both_seeds 或 improvement(0)/improvement(1) 有失败
   （即只在部分 seed/参考下出现增量）                      -> CONTEXT_EFFECT_INCONCLUSIVE
5. 若 support_ok 失败（独立骨架/跨骨架供给不足）          -> CONTEXT_EFFECT_INCONCLUSIVE
6. 若 null_exceeded 失败 或 nontrivial 失败                -> CONTEXT_EFFECT_INCONCLUSIVE
7. 否则                                                    -> CONTEXT_EFFECT_SIGNAL_PRESENT
```

无论哪种结果，本轮都不训练新模型。`B2` 优于 `B1` **不是**成功标准；成功标准是获得能决定
研究路线的证据（含反证）。

## 8. 复现

```bash
uv run --group lipo python \
  tracks/structure-contribution/results/structure_context_effect_gate_v1/code/structure_context_effect_gate_v1.py all
uv run pytest -q tracks/structure-contribution/results/structure_context_effect_gate_v1/tests
```

子命令：`pool`（生成并冻结分析池）、`correspondence`、`responses`、`predict`（B0/B1/B2 与对照）、
`all`。输出全部落在本目录：

```
results/structure_context_effect_gate_v1/
  REPORT.md
  PROTOCOL.md
  summary.json
  analysis/analysis_pool.npz
  analysis/pool_manifest.json
  analysis/bond_correspondence.json
  analysis/responses_seed0.npz / responses_seed1.npz
  analysis/features.npz
  analysis/predictor_metrics.json
  analysis/shuffle_null.json
  code/structure_context_effect_gate_v1.py
  tests/test_structure_context_effect_gate_v1.py
```

随机性只有三处且固定：REF-C 的参考抽取（种子 `20261017`）、打乱对照（种子 `0..99`）、
bootstrap（种子 `20261018`）。prediction/model 前向为确定性（`eval()`、dropout 0、CPU）。

## 9. 已知限制（写入报告，不隐藏）

1. 冻结模型是 `NULL` 组织模式：单键替换的效应包含全局均值通道，响应不是单键局部贡献；
2. `CT` 只是 radius-2 邻域 + 少量关系轴，更远的长程上下文不在定义内；
3. Lipophilicity train-only 分子高度同源；即使按骨架分组，仍有系列化合物共享局部环境；
4. 参考编码是模型内部反事实，不是真实化学编辑，不能外推为化学因果效应；
5. 只有一个性质目标（Lipophilicity）与一个模型族，不外推；
6. 不把 IG/单点遮蔽一致性当作独立忠实度证据（本轮完全不使用 IG）。
```
