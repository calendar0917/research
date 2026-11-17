# PROTOCOL — lcrc-auto-subgraph-faithfulness-pilot-v1

本文件是本轮的**规则声明（跑之前冻结）**。机器可读版本在
`protocols/lcrc-auto-subgraph-faithfulness-pilot-v1.yaml`（控制面协议）与
`configs/experiment.yaml`（工程配置）；冻结常量的哈希在
`analysis/protocol_freeze.json`（`freeze_hash`）。

变化规则：本文件与 `code/as_common.FROZEN_CONSTANTS` 一经正式运行不得为了结果好看而修改。
只允许修**工程实现**（索引 / dtype / 路径 / 账目字段），并写入 `README.md` 的“工程修复记录”。

---

## 0. 身份、对象与边界

| 项 | 冻结值 |
|---|---|
| 实验 ID | `lcrc-auto-subgraph-faithfulness-pilot-v1` |
| 模型 | `lcrc_readout_context_pilot_small_v1` **Arm A = 原始 LCRC-C1**：`sum_v f(h_v) + rho([g, mean(h), mean(h^2), log1p(n_atoms)])` |
| 训练 | **复用**既有 360-epoch、seed 0 轨迹；本轮不训练 |
| 任务 | 官方 `ogbg-molesol`（902/113/113）、`ogbg-mollipo`（3360/420/420） |
| 性质 | 单 seed、小数据、**开发性解释可行性验证**；不是 benchmark，也不是化学因果验证 |

**判断对象是模型对显式结构信息的依赖，不是物理删键后的真实化学性质变化。**

禁止：新解释网络、额外消息传递/注意力/读出、第二 seed、超参搜索、按 Test 选 checkpoint、
扩展数据集或搜索空间、用 ChemBench random122 替换官方 split、自行生成 split。

## 1. 数据与模型

* 只调用 OGB `get_idx_split()` 对应的既有冻结索引（`<dataset>_{train,valid,test}_idx`），
  四个冻结缓存按真实文件字节 SHA-256 在每个阶段校验。
* split 尺寸固定；Train/Valid/Test 两两无交集且覆盖全表（prepare 阶段断言）。
* 目标标准化（mean、population std，ddof=0）**只来自 Train**。
* 模型 checkpoint / 轨迹 / 预测与源轮次 `analysis/run_manifest.json` 记录的 SHA-256 逐项核对；
  **任一身份不匹配即停止**，不得静默当作缺失重训。
* 每 epoch 的全量 Train/Valid/Test 评估必须在轨迹中完整存在（360/360）；
  选中 epoch = 最小 Valid RMSE（平局取更早，epoch 0 不参与），并独立重算核对。
* Test 在本轮每 epoch 都评估，报告一律标注 `DEVELOPMENT_TEST`，且**从不**参与任何选择。

若预测重放、checkpoint 身份或轨迹覆盖不通过，则停止该数据集，不使用它生成任何解释结论。

## 2. 干预：`CONDITIONAL_S1_S2_EVIDENCE_MASK`

对每个分子每条**去重后的真实物理键** `e` 建立一个可微标量门控 `m[e] ∈ [0,1]`；
仅在解释阶段使用，原始模型权重全部冻结、`eval()`、`requires_grad=False`。

### 2.1 S1 门控

对于物理键 `e=(u,v)`：

* 该键送入两个端点 `b_sum` 的 `BondEncoder` 输出分别乘 `m[e]`；
* 由该键产生的两个有向邻居编码 `nbr_emb` 项分别乘 `m[e]`。

C1 没有 `b_mean` 分支，因此**不引入** b_mean 门控。
**不修改** `AtomEncoder` 输入、`BondEncoder` 参数或原始物理键列表。

### 2.2 S2 门控

对每条非回退二跳路径 occurrence `p = (root -> middle -> end)`，其两条物理键为 `e1,e2`：

```
w[p] = m[e1] * m[e2]
```

`w[p]` 乘在该 occurrence 的**全部**贡献上：type embedding、endpoint embedding、
joint embedding/projection、closure embedding、以及 `content = a[end] * learned_gate`。
因此当两条键中任意一条被屏蔽时，该 occurrence 恰好移除 **一次**（它是某个求和中的**一项**）。

### 2.3 不门控的输入

`atom_feat`、`atom_scalar`、`global`（42 维）、原子数 `n_atoms`、原子编码 `a_v`、
以及未被显式门控的其它结构代理信息全部保持不变。

**该干预在全部产物中一律命名为 `CONDITIONAL_S1_S2_EVIDENCE_MASK`，不得称为真实分子删键、
删基团或完整化学反事实。**

## 3. occurrence 映射与模型等价性

* 按 `molecular_graph.enumerate_s1` / `enumerate_s2` 的**原始确定性顺序**重建
  occurrence → `(root, middle, end, physical_bond_id_1, physical_bond_id_2)` 映射；
  缓存中没有 `middle` 与物理键 ID，不得凭路径类别猜测。
* 对每个被解释分子逐项校验重建结果与缓存一致：
  `nbr_code`、`path_cat`、`path_end63`、`path_end`、`path_closure`、`nbr_ptr`、`path_ptr`。
  任一 occurrence 无法正确映射 → 该数据集禁止继续生成解释结论（`ENGINEERING_GATE=FAIL`）。
* 门控全 1 时必须复现原始冻结 Arm A：预测、`local_sum`、`rho` 及其总和，
  容差 `tol_raw(x) = 1e-5 + 1e-6 * max(1,|x|)`（原始单位）。
* 解释前后 checkpoint / state hash 不变。

## 4. 自动发现重要子图

### 4.1 单键敏感度

全部 `m[e]=1`，以原始目标单位的预测计算一次 `g[e] = ∂pred_raw/∂m[e]`：
一次前向 + 一次反向，不构造训练损失，不更新参数。
保存 `edge_id, atom_u, atom_v, bond_type, gradient_raw`。梯度 NaN/Inf → 工程门禁失败。

### 4.2 候选结构

候选 `S` 是原始分子物理键的子集，其涉及原子构成连通图。搜索范围固定：

* 2–6 条物理键；可含环的一部分或完整小环；不要求对应任何 SMARTS 基团；
* 不跨不连通原子集合；
* 不依据任务标签、原子名或人工规则（酰胺/酯等）挑选候选。

超过 6 键的结构本轮**未覆盖**，不能据此断言模型不依赖它们。

### 4.3 发现分数（固定）

```
A(S) = |sum_{e in S} g[e]| / sqrt(|S|)
```

先**有符号求和**再取绝对值，**不是**梯度绝对值之和。该分数只用于**发现候选**，
不是最终解释忠实度指标。

### 4.4 搜索算法（固定）

* `k=1`：对全部单键排序（不截断）。
* `k=2`：枚举**所有**共享端点的两键集合（不依赖单键预筛选）。
* `k=3..6`：从上一层保留的 top-12 候选出发，每次加入一条与现有子图相邻的物理键。
* 每个大小层最多保留 12 个不同候选；同分时优先键数少，再按排序后的物理键 ID 字典序。
* 不重复评估同一键集合。
* 在所有 `k=2..6` 候选中按 `A(S)` 排序，依次选 Top-3；后续候选与已选候选的
  键集合 Jaccard 重叠须 ≤ 0.5；不足 3 个就如实输出不足 3 个。
* **完整候选与排序过程必须落盘**；不得事后按真实屏蔽效果重新挑选 Top-3。

## 5. 独立验证：实际模型响应

对 Top-3 重新执行冻结模型的真实前向：

* `y0`：全部门控为 1 的原始预测；
* `yS`：候选子图的键门控全部为 0、其余为 1；
* `delta(S) = y0 - yS`。

保存 `abs_delta`、`signed_delta`、`delta / train_target_std`。
Top-1 额外计算全部选中键 `m = 0.5` 的剂量响应，并记录与 `m=0` 的方向是否一致。

### 5.1 S1 / S2 分解（Top-1）

* `S1_ONLY`：只屏蔽候选键的 S1 贡献，S2 保持原样；
* `S2_ONLY`：只屏蔽候选键涉及的 S2 occurrence，S1 保持原样；
* `S1_S2`：同时屏蔽两者（= Top-1 主干预，独立重算并核对一致）。

```
interaction_delta = delta(S1_S2) - delta(S1_ONLY) - delta(S2_ONLY)
```

该值只是**当前冻结函数的条件性交互诊断**，不是独立化学交互能。

### 5.2 局部求和 / 全图读出分解

分别保存干预前后的

* `L = sum_v f(h_v)`
* `P = rho([g, mean(h), mean(h^2), log1p(n_atoms)])`

必须核对 `delta_total = delta_L + delta_P`（原始单位，`tol_raw` 内）。

仅用于描述、**不是唯一贡献百分比**：

```
rho_share    = |delta_P| / (|delta_L| + |delta_P| + 1e-12)
cancellation = (|delta_L| + |delta_P| - |delta_total|) / (|delta_L| + |delta_P| + 1e-12)
```

## 6. 两类固定对照

### 对照 A：匹配的随机连通子图

对每个自动发现的 Top-1 子图，尝试生成 16 个不同的随机连通子图。条件：

1. 来自**同一原始分子**；
2. 物理键数量与 Top-1 相同；
3. 不与 Top-1 完全相同；
4. 被屏蔽的 S2 occurrence 数与 Top-1 的相对差 ≤ 25%（分母 `max(1, n_s2_top1)`）。

实现：固定 RNG（`SeedSequence([20261117, dataset, split, row_id])`），最多 512 次随机增长尝试，
合格候选按确定性生成次序取前 16 个。合格对照少于 8 个 → `INSUFFICIENT_MATCHED_NULL`，
该分子的匹配对照结论记为缺失；**匹配成功率必须报告，不得只报告成功匹配的分子**。
另存的“只匹配键数”随机集合仅作次要展示，**不进入主对照指标**。

### 对照 B：原子局部贡献启发式

复用 Arm A 已有的逐原子局部输出 `c_v = f(h_v)`：

```
B(S) = sum_{v in atoms(S)} |c_v| / sqrt(|S|)
```

使用与 §4.4 相同的搜索规则、范围、beam width 与 tie-break 选出该启发式的 Top-1，
再用**完全相同的 S1/S2 门控干预**计算真实 `abs_delta`。
`c_v` **不**被称为原子的唯一因果贡献，只当作廉价、直接的对比方法。

## 7. 样本选择与运行规模

* 每数据集从官方 Train / Valid / Test 各选 **64** 个分子（共 2×3×64 = 384）。
* 规则：对 `sha256(20261117|dataset|split|official_row_id)` 升序排序（同键按 row id 升序）取前 64。
  **禁止**依据标签大小、预测误差、是否含某基团、梯度大小或 Test 结果筛选。
* 保存每个 split 的样本 ID 列表与 sha256。
* 无法解析/映射的图记录原始 row id 与原因，**不得用另一个分子补齐**。
* 先在每数据集 Valid 前 8 个选中分子上做工程 smoke；全部门禁通过后才执行正式 384 分子。
* 训练曲线仍按官方**全量** Train/Valid/Test；64 个样本的限制只用于解释计算。

## 8. 必须实现的正确性测试（`tests/`）

1. 全 1 门控与原始模型预测、`local_sum`、`rho` 等价（Train/Valid/Test）。
2. S1 两端点、S2 两条物理键的 occurrence 映射正确；含三角闭合路径与重复（反向）路径。
3. 同一 occurrence 同时碰到被屏蔽键时只移除一次（与“从路径列表删除该 occurrence”逐位比对）。
4. `m=0.5` 的连续门控能正常反向传播。
5. 8 个固定分子若干条键、中央有限差分 `epsilon=1e-3` 验证 autograd 梯度；
   报告绝对/相对误差；近零梯度不使用不稳定的相对误差做判定。
6. 原子重编号后预测与映射回原始键的梯度数值等价（≥ 8 个分子）。
7. 单图推理与 batch 推理一致；无跨分子污染（含梯度层面的检验）。
8. `delta_total == delta_L + delta_P`（原始单位闭合）。
9. 重载 checkpoint 结果一致；解释前后参数 hash 不变。
10. Train/Valid/Test ID 无交集；目标标准化只来自 Train。

等价、索引、数值非有限等硬错误 → 测试失败并停止正式实验。不得放宽容差或删去失败样本。

## 9. 报告指标与判读

所有指标按 dataset × split 分列，不合并成单一总分。

### 9.1 结构发现

可解释分子覆盖率；Top-1/2/3 键数与涉及原子数；Top-1 `|delta|` 与
`|delta| / train_target_std`；Top-1 的 S1/S2 occurrence 扰动数量；
`g` 方向与真实门控差分方向一致的比例；连续梯度近似与真实（有限）干预之间的偏差。

### 9.2 相对对照

对匹配成功的分子：

```
effect_ratio      = |delta(Top1)| / (median_e |delta(matched_random_e)| + 1e-12)
effect_advantage  = |delta(Top1)| - median_e |delta(matched_random_e)|
```

报告 `effect_ratio > 1` 的分子比例、相对局部贡献启发式的 `abs_delta` 差、
以分子为重采样单位的 2000 次 bootstrap 区间。

bootstrap 只反映**固定 checkpoint、固定数据集**下的样本不确定性，
**不得**表述为跨 seed 稳定性。比值不能单独证明发现有用：若所有变化接近 0，比值再大也没有意义。
额外报告 `|delta(Top1)| ≥ 0.01 × train_target_std` 的分子比例（1% 只是本轮 pilot 的效果分辨尺度）。

### 9.3 架构诊断

汇总 `rho_share` 分布、`cancellation` 分布、`S1_ONLY` / `S2_ONLY` / `S1_S2` 响应、
S1/S2 交互项分布、高梯度但低真实响应的失败案例、高局部输出但低真实响应的失败案例、
Top-1 对该分子 S1 code / S2 occurrence 的覆盖率。

### 9.4 固定开发性判读（不强行合成单一成功标签）

**ENGINEERING_GATE**：全部核心等价、映射与回溯测试通过 → `PASS`，否则 `FAIL` 并停止。

**DISCOVERY_SIGNAL**：某数据集的 Valid 与 Test 同时满足
①有足够的匹配对照样本（匹配成功的分子数 ≥ max(32, 80% 已解释分子)）；
②`effect_ratio` 中位数 ≥ 1.5；③`effect_ratio > 1` 的比例 ≥ 70%；
④至少 50% 的分子达到 1% Train 标签标准差的预测变化
→ `PILOT_SIGNAL`，否则 `WEAK_OR_INCONCLUSIVE` 并列出未通过条件。
这只是开发性投入规则，不是统计显著性，也不是科学结论。

**ARCHITECTURE_DIAGNOSIS**：只如实输出 `rho_share`、抵消、S1/S2 交互、覆盖率等证据；
**不**自动推导“必须改读出”或“必须改结构算子”。

失败时必须区分：工程实现失败 / 候选搜索失败 / 梯度与有限干预不一致 /
结构信息屏蔽后变化过小 / 无足够匹配对照 / 全图读出混合严重 /
搜索范围或固定原子与全图特征造成的不可识别性。
**禁止**把解释失败直接写成“LCRC 没有学到结构信息”。

## 10. 可视化

RDKit 高亮展示 Top-3 连通子图。每数据集固定输出：

* Valid 中按官方 row id 确定性选取的前 8 个分子；
* Test 中按官方 row id 确定性选取的前 8 个分子；
* 另展示预测响应最大 2 个与最小 2 个**诊断案例**，明确标记为结果驱动选择。

每个分子图附：官方 row id、SMILES、真实标签、预测值、Top-1/2/3 高亮原子与物理键 ID、
梯度发现分数、真实屏蔽预测差、随机匹配对照效果、S1/S2 与局部/全图读出响应。

可以在**完成自动排序之后**用 RDKit 或现有规则标注片段是否对应常见化学结构，
但该标注不得参与搜索、评分、筛选或训练。

## 11. 资源与远端执行

* 远端只用 `res` 的两个空闲 GPU：`res-gpu0` = `ogbg-molesol`，`res-gpu1` = `ogbg-mollipo`。
* 全部远端执行经 `rr`（`rr doctor` → `rr deploy` → `rr run --pool ...` → `rr jobs/status/logs` → `rr pull`）；
  禁止手工 SSH + `nohup`/`sbatch`，禁止远端改代码，禁止覆盖历史实验目录。
* 预算：不重复已验证的 A 模型训练；每 GPU worker 正式解释软上限 **0.5 GPU·h**、硬上限 **1.0 GPU·h**。
  原模型中间编码按分子批量缓存；候选搜索只用一次梯度计算完成；实际干预按候选批量前向。
  到达硬预算即停止，保留已完成样本与缺失清单，不增加资源、不因结果弱而延长预算。

## 12. 交付物

`PROTOCOL.md`、`README.md`、`code/`、`tests/`、`analysis/preflight.json`、
`analysis/training_source.json`、`results/<dataset>/`、`analysis/main_table.csv`、
`analysis/architecture_diagnostics.csv`、`analysis/figures/`、`analysis/run_manifest.json`、
`REPORT.md`、`DECISION.md`。

不把数百个原始分子图提交 Git；提交代表性图与压缩后的可复现结果。

## 13. 术语纪律

* 干预 = `CONDITIONAL_S1_S2_EVIDENCE_MASK`（显式结构证据的条件性屏蔽）。
* `rho_share` / `cancellation` = 描述性读出分解，不是唯一贡献百分比。
* `c_v = f(h_v)` = 冻结模型的逐原子局部输出，不是原子的唯一因果贡献。
* bootstrap 区间 = 固定 checkpoint/数据集下的样本不确定性，不是跨 seed 稳定性。
* 1% Train 标签标准差 = 本轮 pilot 的效果分辨尺度，不是普适重要性阈值。
