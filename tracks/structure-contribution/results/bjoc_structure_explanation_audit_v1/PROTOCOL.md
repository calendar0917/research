# BJOC-Null 结构信息流与解释可行性审计 v1 — PROTOCOL（冻结）

* 命名空间：`bjoc_structure_explanation_audit_v1`
* 轨道：`tracks/structure-contribution`
* 状态：本文件在任何审计数值产生之前写入（冻结协议）。只做冻结模型的静态依赖审计与 CPU 推理干预，**不训练、不选模、不看任何标签**。
* 相关已有审计（只读复用，不修改）：
  * `results/bjoc_junction_sufficiency_audit_v1/`（B1/B2/B3/B4 的 Null ownership blindness、reader 4 block 置零、ridge R²、reader 距离标尺）
  * `results/bjoc_architecture_breakthrough_v1/REPORT.md`（GLRC/DLP 负结果、reader 通路实验口径）
  * `results/cscl_task_adaptive_structure_units_v1/REPORT.md`（additive 分解 vs opaque reader 的历史证据）
* 待审计冻结模型：BJOC-Null（413,505 params）+ full-10k masked atom/bond SSL 编码器 + R3 recipe（360 epoch cosine，1e-3→1e-5），两个 seed 的 soup：
  * `results/bjoc_recipe_tuning_v1/soup_R3_seed0.pt`
  * `results/bjoc_recipe_tuning_v1/soup_R3_seed1.pt`

## 1. 审计问题与非目标

回答五个问题：

* **A**：raw graph → 固定结构特征 → `a_i` → `b_uv` → `w_ijk` → `W_b/V_b/n_b` → `c_b` → reader_in → prediction 的逐层真实计算依赖是什么；从哪一层开始失去严格局部结构依赖。
* **B**：冻结模型四条 reader 输入通路（`z_bond`=Σc，`sum_atom`，`sum_atom_squared`，`global_topology`）的干预敏感性与冗余/旁路程度。
* **C**：远端 junction 的编码表示 `w_j` 是否会通过 Null 分组公式进入任意 Bond 的 `c_e`。
* **D**：能否不重新训练，定义对“连通子结构”具有非平凡忠实度的解释接口。
* **E**：哪些结论是公式必然、哪些是冻结模型数值干预、哪些仍无证据。

**非目标**：不报告 MAE/RMSE；不做模型选择；不训练；不做真实化学编辑（不删除原子/键、不做官能团替换）；不实现 Motif 搜索或可学习结构单元；不修改任何已有模型定义、权重、配置、报告。

## 2. 数据纪律

* 只使用官方 train 结构的冻结缓存 `results/bjoc_full10k_pretraining_transfer_v1/objects_train10k.pt`（`MolObjects` 不含 `y` 字段，构建时已由 Phase-A gate 证明无标签）。
* **不加载、不读取、不引用**官方 valid / 官方 test 的结构与标签；不读取任何 Monitor / Locked 标签；不使用历史 valid/test 指标进行任何选择。
* 本次审计不计算任何监督指标（MAE/RMSE 等），因此不存在用标签做判断的路径。
* 归一化只使用 `prep_train10k.json` 中已冻结的 train-only `global_mean/global_scale`（只读）。

## 3. 冻结模型与复现口径

* 用 `run_bjoc_full10k_pretraining_transfer_v1._make_downstream_model(seed, norm, "cpu")` 重建结构，`load_state_dict(soup_R3_seed{seed}.pt, strict=True)`。
* `model.eval()` + `torch.no_grad()`；冻结权重，禁止任何 optimizer step。
* 参数必须恰为 413,505；reader_in 必须恰为 321 维 = `[z(96), sum_atom(96), sum_atom_squared(96), global_topology(33)]`。
* checkpoint、对象缓存、prep 文件的 SHA-256 与既有记录（`bjoc_full10k_pretraining_transfer_v1/REPORT.md`、`bjoc_recipe_tuning_v1/REPORT.md`）不一致即停止并报告阻塞，不允许静默替换模型。

## 4. 分子样本选择（预先固定的哈希规则，label-blind）

对每一行 `i`，由冻结对象缓存计算规范键：

```
key_bytes(i) = atom_feat(float32 C-contig) || bond_index(int64) || bond_type(int64)
               || junc_index(int64) || junc_bond(int64)
rank(i) = 按 sha256(key_bytes(i)).hexdigest() 升序（同哈希按行号升序）
```

* **诊断集**：`rank 0..255`（256 个分子）。
* **smoke 子集**：`rank 0..15`（16 个分子，只用于先跑通流程）。
* **参照集**：`rank 256..1279`（1024 个分子），与诊断集不重叠。
* 选择只看结构张量字节，不看任何标签或历史指标；规则在运行前冻结（本节）。
* 基本信息统计（`n_atoms, n_bonds, n_junctions`）可从缓存读出，用于参照替换的匹配键。

## 5. 干预语义（严格区分“真实化学编辑”）

以下所有干预都是**内部表示的反事实重组**，不是分子编辑，也不对应任何真实化学操作：

1. **B2-zero**：把指定 reader 输入块置为精确 0。
2. **B2-ref（匹配参照替换）**：对每个诊断分子，用参照集中基本结构统计 `(n_atoms, n_bonds, n_junctions)` 的 L1 最近邻（并列取哈希 rank 小者）的同一块替换目标块。这是“自然表示的重组”，不是结构编辑。
3. **B3-remote-junction**：在编码完成后，对选定的单个 junction `j*` 的 `w_{j*}` 加一个确定性的扰动 `α·δ`（`δ` = 该分子内 `w` 逐维标准差；`α ∈ {0.25, 0.5, 1.0, 2.0}`），然后用原始 Null 公式重算 `W_b, V_b, c_b, reader_in, pred`。其他编码表示（`a,b,w` 的其余行）冻结不变。`(e*, j*)` 取图上距离最大的 Bond–Junction 对（并列取索引小者）；距离定义为 junction 原子集 `{i,j,k}` 与 bond 端点 `{u,v}` 的最短路径距离的最小值。
4. **C-bond-subset（仅限 Bond 汇总通路）**：只在 `z = Σ_b c_b` 中把选定 bond 集合 `S` 的 `c_b` 行置 0；`sum_atom / sum_atom_squared / global_topology` 完全不变。任何由该干预得到的结论只能限定为“Bond 汇总通路解释”，不得声称解释了整个分子预测。
5. **C-local-decomposition（诊断用途）**：用同一 composer 权重，把 `W_b=V_b=0`（保留 `b_b, log1p(n_b), bond_feat`）计算 `c_b^local`，用于分解每个 `c_b` 中“全局项 vs 局部项”的量级。这是诊断性反事实，不是模型真实使用的表示。

## 6. 预注册统计口径

* 干预差异一律为 `Δpred = pred_intervened − pred_natural`，报告 mean / median / P90 的 `|Δ|`（原始 y 单位）。
* 两个 seed 的方向与量级：报告每 seed 的 `mean Δ`、`median |Δ|`，以及两个 seed 的 per-molecule `Δ` 的 Pearson / Spearman 相关与符号一致率。
* 零替换 vs 匹配参照替换一致性：报告两者判断（每块、每 seed）的符号一致率与 per-molecule rank 相关。
* 分布外程度：
  * 逐维 z-score：用参照集 321 维 mean/std 标准化后，被替换块的平均 `|z|`；自然块作为对照。
  * 多变量最近邻：在参照集标准化的 321 维空间中，modified reader_in 到最近自然 reader_in（参照集 ∪ 诊断集自然值，排除自身）的距离，与自然值的最近邻距离之比（median / P90）。
* 通路冗余：岭回归（λ=1e-2，标准化后）用参照集拟合“其他三块 → 目标块”，在诊断集报告 R²；另报 321 维内 4 块之间的线性 CKA 矩阵。旧审计的置零结果只作**复现**，不作为新发现。
* B3 公式验证：对每个干预，数值重算的 `ΔW_b` 必须等于 `n_b·(α·δ/n_junc)`（float32 舍入内）；`α=0` 时必须逐位复现原 `c` 与 `pred`。
* B3 距离/规模规律：按 `d(b,j*)` 分桶报告 `mean |Δc_b|`；报告 `|Δc|` 与距离的 Spearman 相关、`|Δpred|` 与 `n_bonds`/`n_junctions` 的 Spearman 相关；报告组织模式对照（同一权重、同一扰动下 `organized` 分组只有 `j*` 的两个 bond 的 `W` 变化）。
* Phase C：固定子集大小 `k = min(8, n_bonds)`；连通子集 = 从 bond 0（或随机种子 bond）出发的 bond-BFS 球；随机连通子集 8 次；随机任意子集 8 次；空集与全 Bond 集必报（`Δ=0` 与最大差异两个校验）。用配对比较（每分子自身对照）报告 connected − random 差异。若连通与随机在归一化后无实质区别，判定“Bond 锚点不能构成忠实连通子结构解释接口”。

## 7. 证据分级（报告中必须标注）

1. **公式必然（algebraic）**：可直接从模块公式证明，并附一种实现级验证。
2. **冻结模型数值干预（numeric intervention）**：在冻结权重上的可观测量，结论受干预分布外性与 Reader 非线性限制。
3. **无证据 / 未决（unsupported）**：当前审计没有提供支持。

报告用词禁令：不把 reader 块置零/替换称为“独立贡献”；不把内部向量掩蔽称为“删除键/删除子结构/官能团替换”；不用预测差异反推化学因果。

## 8. 停止条件

* 任一步骤出现 SHA-256 不一致、模型重建失败、参数数 ≠ 413,505、`forward` 与 `reader(forward_features)` 不一致（>1e-6）、原子重标号或 Bond 顺序置换改变预测（>1e-5）、Null 公式验证失败 → 立即停止依赖该结果的后续阶段并报告阻塞。
* Phase C：若在 Phase A/B 已经证明 `c_b` 具有不可分离的图级项，且找不到“选中连通子结构 ⇒ 有限、明确、忠实于模型实际计算”的内部干预，则停止 Phase C 的实现，只报告接口不充分与缺失条件。
* 不为了得到更“好看”的结论调整评价方式、样本或阈值。

## 9. 交付物

* `PROTOCOL.md`（本文件）
* `sample_manifest.json`：哈希规则、诊断/参照行索引与统计
* `source_manifest.json`：checkpoint/缓存/code SHA-256、环境、行索引
* `phase_a_dependency.json`：逐层依赖表、A1/A2/A3 验证
* `phase_b_baseline.json`：B1 一致性
* `phase_b_pathways.json`：B2 通路敏感性、冗余、OOD
* `phase_b_nonlocal.json`：B3 远端 junction 传播
* `phase_c_interface.json`：C 接口探针（或停止记录）
* `metrics.json`：汇总统计
* `REPORT.md`：按证据强度组织的判断与 Q1–Q5 回答
* 审计代码 `code/bjoc_structure_explanation_audit_v1.py` 与针对性测试 `tests/test_bjoc_structure_explanation_audit_v1.py`
