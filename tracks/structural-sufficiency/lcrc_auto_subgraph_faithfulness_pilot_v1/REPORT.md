# REPORT — lcrc-auto-subgraph-faithfulness-pilot-v1

冻结的 LCRC-C1（Arm A）在官方 `ogbg-molesol` / `ogbg-mollipo` 上的**自动连通子图解释可行性**开发性验证。单 seed、每划分 64 个分子、单次干预条件 `CONDITIONAL_S1_S2_EVIDENCE_MASK`。

---

## 1. 一页摘要

* **ENGINEERING_GATE**：`PASS`（关键门禁见第 4 节）。
* **DISCOVERY_SIGNAL（总体）**：`WEAK_OR_INCONCLUSIVE`。
* `ogbg-molesol`：`WEAK_OR_INCONCLUSIVE`，未通过条件：valid:effect_ratio_median_ge_1p5, test:effect_ratio_median_ge_1p5。
* `ogbg-mollipo`：`WEAK_OR_INCONCLUSIVE`，未通过条件：test:effect_ratio_median_ge_1p5, test:effect_ratio_gt1_fraction_ge_0p70。
* 关键提醒：本轮的干预是**显式结构证据的条件性屏蔽**，等价于在冻结函数的 S1/S2 求和里把对应项乘 0，**不是**真实分子删键、删基团或完整化学反事实。

## 2. 实验边界（冻结，不可事后调整）

| 项 | 冻结值 |
|---|---|
| 实验 ID | `lcrc-auto-subgraph-faithfulness-pilot-v1` |
| 模型 | `lcrc_readout_context_pilot_small_v1` Arm A = 原始 LCRC-C1：`sum_v f(h_v) + rho([g, mean(h), mean(h^2), log1p(n_atoms)])` |
| 训练 | 复用既有 360-epoch 轨迹（seed 0），本轮**不重训** |
| 数据集 | 官方 `ogbg-molesol` / `ogbg-mollipo`，OGB `get_idx_split()` 冻结索引 |
| 解释样本 | 每数据集 Train/Valid/Test 各 64 个（共 384） |
| 样本选择 | `sha256(20261117 + dataset + split + row_id)` 升序取前 64，与标签/误差/梯度无关 |
| 搜索空间 | 连通子图，物理键数 2–6，每层 beam 12，Top-3，Jaccard ≤ 0.5 |
| 发现分数 | `A(S) = abs(sum_e g[e]) / sqrt(k)`（有符号先求和再取绝对值，k = 键数） |
| 干预 | `CONDITIONAL_S1_S2_EVIDENCE_MASK`，门控 `m[e] ∈ [0,1]`，剂量点 m = 0.5 |
| 对照 A | 同分子同键数随机连通子图，匹配 S2 occurrence 数（相对差 ≤ 25%），RNG 20261117，最多 512 次尝试，取前 16 个 |
| 对照 B | 原子局部输出启发式 `B(S) = sum_v abs(f(h_v)) / sqrt(k)`（仅作廉价对比，不称因果） |
| 效果分辨尺度 | 1% 的 Train 标签标准差（仅本轮 pilot 尺度） |
| bootstrap | 2000 次，分子为重采样单位，seed 20261117 |
| 预算 | 每 GPU worker 软上限 0.5 GPU·h，硬上限 1.0 GPU·h |

## 3. 数据、模型与复用身份核验

| dataset | checkpoint sha256 | epochs | test evals | selected epoch | Valid RMSE | Test RMSE | replay |
|---|---|---|---|---|---|---|---|
| `ogbg-molesol` | 5a396b4675ff | 360 | 360 | 248 | 0.8753 | 0.8545 | True |
| `ogbg-mollipo` | 1b96644df50a | 360 | 360 | 275 | 0.7086 | 0.7254 | True |

* 两个数据集的 checkpoint / trajectory / `best_predictions.npz` 均与源轮次 `analysis/run_manifest.json` 记录的 SHA-256 一致；每个 epoch 都执行了 Train/Valid/Test 三个完整划分的评估（360/360）。
* 选中 epoch = 该冻结轨迹的最小 Valid RMSE（平局取更早 epoch），本轮已独立重算并核对。
* 全量 Train/Valid/Test 预测由重载 checkpoint 逐项重放（见 `analysis/preflight.json` 的 `checkpoint_replay`），Test 在本轮只作为 `DEVELOPMENT_TEST` 报告。

## 4. ENGINEERING_GATE

| dataset | gate | passed | detail |
|---|---|---|---|
| `ogbg-molesol` | DATA_IDENTITY | yes | frozen caches + official split sizes/disjointness |
| `ogbg-molesol` | TRAINING_IDENTITY | yes | checkpoint/trajectory/manifest SHA-256 and 360-epoch 3-split coverage |
| `ogbg-molesol` | CHECKPOINT_REPLAY | yes | reloaded checkpoint reproduces stored predictions |
| `ogbg-molesol` | MODEL_EQUIVALENCE_ALL_ONES | yes | all-ones gates reproduce frozen arm A on Train/Valid/Test rows |
| `ogbg-molesol` | OCCURRENCE_MAPPING | yes | every S1/S2 occurrence rebuilt from enumerate_s1/enumerate_s2 and verified against the cache |
| `ogbg-molesol` | BATCHING_ISOLATION | yes | single vs batch and order permutation |
| `ogbg-molesol` | ATOM_RENUMBERING | yes | prediction and bond gradients equivariant under atom relabelling |
| `ogbg-molesol` | FINITE_DIFFERENCE | yes | central FD eps=1e-3 vs autograd |
| `ogbg-molesol` | PARAMETER_HASH_UNCHANGED | yes | no parameter was modified by the explanation path |
| `ogbg-molesol` | PARAMETER_BUDGET | yes | 228674 total / 15298 readout |
| `ogbg-molesol` | SPLIT_IDS_DISJOINT_TRAIN_ONLY_STD | yes | Train/Valid/Test disjoint; target mean/std from Train only |
| `ogbg-mollipo` | DATA_IDENTITY | yes | frozen caches + official split sizes/disjointness |
| `ogbg-mollipo` | TRAINING_IDENTITY | yes | checkpoint/trajectory/manifest SHA-256 and 360-epoch 3-split coverage |
| `ogbg-mollipo` | CHECKPOINT_REPLAY | yes | reloaded checkpoint reproduces stored predictions |
| `ogbg-mollipo` | MODEL_EQUIVALENCE_ALL_ONES | yes | all-ones gates reproduce frozen arm A on Train/Valid/Test rows |
| `ogbg-mollipo` | OCCURRENCE_MAPPING | yes | every S1/S2 occurrence rebuilt from enumerate_s1/enumerate_s2 and verified against the cache |
| `ogbg-mollipo` | BATCHING_ISOLATION | yes | single vs batch and order permutation |
| `ogbg-mollipo` | ATOM_RENUMBERING | yes | prediction and bond gradients equivariant under atom relabelling |
| `ogbg-mollipo` | FINITE_DIFFERENCE | yes | central FD eps=1e-3 vs autograd |
| `ogbg-mollipo` | PARAMETER_HASH_UNCHANGED | yes | no parameter was modified by the explanation path |
| `ogbg-mollipo` | PARAMETER_BUDGET | yes | 228674 total / 15298 readout |
| `ogbg-mollipo` | SPLIT_IDS_DISJOINT_TRAIN_ONLY_STD | yes | Train/Valid/Test disjoint; target mean/std from Train only |

| dataset | FD bonds | FD max abs err | FD max rel err | all-ones max abs err | renumber grad max abs err |
|---|---|---|---|---|---|
| `ogbg-molesol` | 32 | 2.7396e-04 | 0.0034 | 4.9275e-07 | 1.2319e-07 |
| `ogbg-mollipo` | 32 | 1.8916e-04 | 6.4409e-04 | 2.8885e-07 | 2.5275e-07 |

* 全 1 门控与冻结 Arm A 的预测、`local_sum`、`rho` 在 Train/Valid/Test 行上**在冻结容差内一致**（`1e-5 + 1e-6*max(1,|x|)`，原始单位；实测最大误差见上表，比容差小 1–2 个数量级，残差来自门控乘法在浮点累加顺序上的差异，不是模型差异）。
* S1/S2 occurrence 映射（含三角闭合路径与重复路径）逐项与冻结缓存比对；同一 occurrence 被两条被屏蔽键触碰时，按“从路径列表中删除该 occurrence”验证只移除一次。
* 解释前后 checkpoint / state hash 不变；原子重编号后预测与键梯度等价。

## 5. 解释样本

| dataset | split | n | row_ids sha256 |
|---|---|---|---|
| `ogbg-molesol` | train | 64 | `a591262dd898748d` |
| `ogbg-molesol` | valid | 64 | `401f8881dd42f5d5` |
| `ogbg-molesol` | test | 64 | `f75c68cc74655b54` |
| `ogbg-mollipo` | train | 64 | `befcb8c4a9de821e` |
| `ogbg-mollipo` | valid | 64 | `6cbd2b9c375135cd` |
| `ogbg-mollipo` | test | 64 | `a44ac0789c4b45dc` |

选择规则与标签、预测误差、分子内容、梯度、Test 结果均无关；完整 ID 见 `results/<dataset>/samples.json`。

## 6. 结构发现与真实屏蔽响应（各 split 分列）

| dataset | split | n | coverage | Top1 k | Top1 atoms | Top1 S2 occ | median abs(delta) | median abs(delta)/std | abs(delta) ≥ 1% std |
|---|---|---|---|---|---|---|---|---|---|
| `ogbg-molesol` | train | 64 | 100.0% | 6.0 | 6.0 | 10.0 | 1.7105 | 0.8276 | 100.0% |
| `ogbg-molesol` | valid | 64 | 100.0% | 6.0 | 6.0 | 12.0 | 1.6945 | 0.8199 | 100.0% |
| `ogbg-molesol` | test | 64 | 100.0% | 6.0 | 7.0 | 12.0 | 2.1905 | 1.0599 | 100.0% |
| `ogbg-mollipo` | train | 64 | 100.0% | 3.0 | 4.0 | 6.0 | 0.8406 | 0.6938 | 100.0% |
| `ogbg-mollipo` | valid | 64 | 100.0% | 4.0 | 5.0 | 6.0 | 0.8213 | 0.6779 | 98.4% |
| `ogbg-mollipo` | test | 64 | 100.0% | 4.0 | 5.0 | 6.0 | 0.6820 | 0.5629 | 100.0% |

| dataset | split | sign agreement (Top1) | sign agreement (Top1-3) | median abs(linearisation err) | median rel err |
|---|---|---|---|---|---|
| `ogbg-molesol` | train | 92.2% | 92.4% | 0.5969 | 0.3797 |
| `ogbg-molesol` | valid | 98.4% | 97.9% | 0.6405 | 0.4199 |
| `ogbg-molesol` | test | 100.0% | 100.0% | 0.8810 | 0.4953 |
| `ogbg-mollipo` | train | 78.1% | 79.2% | 0.9258 | 0.5536 |
| `ogbg-mollipo` | valid | 87.5% | 85.4% | 0.8239 | 0.5513 |
| `ogbg-mollipo` | test | 76.6% | 78.6% | 0.8450 | 0.6352 |

`sign agreement` = 用 `sum_{e in S} g[e]` 的符号预测真实屏蔽差 `delta = y0 - yS` 的符号的比例；`linearisation err` = `|delta - sum g|`，即在 m=1 处的连续梯度近似与真实有限屏蔽之间的偏差。

| dataset | n | Top-1 键数分布 (k:count) | 落在 6 键上限的比例 |
|---|---|---|---|
| `ogbg-molesol` | 192 | 2:15 3:7 4:15 5:23 6:132 | 68.8% |
| `ogbg-mollipo` | 192 | 2:63 3:35 4:22 5:22 6:50 | 26.0% |

落在 6 键上限的比例越高，说明该数据集的 `A(S)` 在冻结搜索空间边界上仍在增长；此时报告的真实屏蔽效果是**当前搜索空间下的下界**，而不是该分子的最优可解释结构。这是搜索范围的诊断，本轮不改变 2–6 键的冻结上限。

## 7. 相对两类对照

| dataset | split | matched n | success | median effect_ratio | ratio>1 | median advantage | random median abs(delta) | ratio CI95 | heuristic median abs(delta) | advantage vs heuristic | Top1 > heuristic |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `ogbg-molesol` | train | 57/64 | 89.1% | 1.287 | 77.2% | 0.4025 | 1.2371 | [1.104, 1.388] | 1.3116 | 0.3173 | 67.2% |
| `ogbg-molesol` | valid | 63/64 | 98.4% | 1.293 | 84.1% | 0.4629 | 1.3494 | [1.210, 1.431] | 1.3612 | 0.5698 | 70.3% |
| `ogbg-molesol` | test | 61/64 | 95.3% | 1.386 | 83.6% | 0.5330 | 1.4248 | [1.246, 1.550] | 1.4563 | 0.5703 | 67.2% |
| `ogbg-mollipo` | train | 64/64 | 100.0% | 1.672 | 70.3% | 0.3608 | 0.5074 | [1.335, 2.085] | 1.1774 | -0.0817 | 42.2% |
| `ogbg-mollipo` | valid | 64/64 | 100.0% | 1.568 | 71.9% | 0.2945 | 0.4937 | [1.283, 2.224] | 0.8083 | 0.0210 | 50.0% |
| `ogbg-mollipo` | test | 64/64 | 100.0% | 1.286 | 56.2% | 0.1407 | 0.5233 | [0.875, 1.672] | 0.7532 | -0.1270 | 45.3% |

`effect_ratio = |delta(Top1)| / (median_e |delta(matched random_e)| + 1e-12)`；`effect_advantage = |delta(Top1)| - median_e |delta(matched random_e)|`。bootstrap 区间只反映**固定 checkpoint、固定数据集**下的样本不确定性，不是跨 seed 稳定性。匹配对照不足 8 个的分子记为 `INSUFFICIENT_MATCHED_NULL`，其匹配结论按缺失处理。
* `INSUFFICIENT_MATCHED_NULL` 分子数：31 / 384（row ids: `ogbg-molesol`/train/78, `ogbg-molesol`/train/820, `ogbg-molesol`/train/332, `ogbg-molesol`/train/605, `ogbg-molesol`/train/505, `ogbg-molesol`/train/354, `ogbg-molesol`/train/474, `ogbg-molesol`/train/491, `ogbg-molesol`/train/1076, `ogbg-molesol`/train/112, `ogbg-molesol`/train/403, `ogbg-molesol`/train/1081 … 其余 19 个见 `analysis/molecule_table.csv` 的 `matched_status` 列）

## 8. 架构诊断：S1 / S2 / 交互 / 局部求和 / 全图读出

| dataset | split | n | median abs(S1_ONLY) | median abs(S2_ONLY) | median abs(S1_S2) | S2 > S1 | median interaction | median abs(interaction) |
|---|---|---|---|---|---|---|---|---|
| `ogbg-molesol` | train | 64 | 0.1636 | 1.4545 | 1.7105 | 89.1% | -0.3177 | 0.3569 |
| `ogbg-molesol` | valid | 64 | 0.1717 | 1.7379 | 1.6945 | 100.0% | -0.3294 | 0.4259 |
| `ogbg-molesol` | test | 64 | 0.1240 | 1.8353 | 2.1905 | 98.4% | -0.3718 | 0.6005 |
| `ogbg-mollipo` | train | 64 | 0.1785 | 0.7397 | 0.8406 | 89.1% | 0.0520 | 0.3871 |
| `ogbg-mollipo` | valid | 64 | 0.2201 | 0.9625 | 0.8213 | 89.1% | 0.0514 | 0.2440 |
| `ogbg-mollipo` | test | 64 | 0.1892 | 0.7286 | 0.6820 | 84.4% | -0.0483 | 0.2649 |

| dataset | split | rho_share med | q25 | q75 | P dominates | cancellation med | cancellation > 0.5 | base rho_share | local share | S1 coverage | S2 coverage |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `ogbg-molesol` | train | 0.142 | 0.072 | 0.369 | 20.3% | 0.073 | 12.5% | 0.435 | 0.858 | 47.7% | 32.6% |
| `ogbg-molesol` | valid | 0.092 | 0.057 | 0.162 | 3.1% | 0.125 | 9.4% | 0.333 | 0.908 | 28.6% | 20.0% |
| `ogbg-molesol` | test | 0.065 | 0.049 | 0.139 | 0.0% | 0.111 | 6.2% | 0.258 | 0.935 | 28.2% | 20.0% |
| `ogbg-mollipo` | train | 0.121 | 0.061 | 0.290 | 7.8% | 2.607e-09 | 7.8% | 0.331 | 0.879 | 11.5% | 6.3% |
| `ogbg-mollipo` | valid | 0.160 | 0.065 | 0.263 | 4.7% | 4.064e-09 | 6.2% | 0.296 | 0.840 | 13.8% | 7.9% |
| `ogbg-mollipo` | test | 0.136 | 0.055 | 0.223 | 4.7% | 6.757e-17 | 3.1% | 0.333 | 0.864 | 10.1% | 5.7% |

| dataset | split | dose same direction | median abs(delta at m=0.5)/abs(delta at m=0) | median abs(delta)/std | p90 abs(delta)/std |
|---|---|---|---|---|---|
| `ogbg-molesol` | train | 92.2% | 0.576 | 0.8276 | 1.4768 |
| `ogbg-molesol` | valid | 98.4% | 0.554 | 0.8199 | 1.4551 |
| `ogbg-molesol` | test | 100.0% | 0.444 | 1.0599 | 1.6119 |
| `ogbg-mollipo` | train | 82.8% | 0.686 | 0.6938 | 1.7820 |
| `ogbg-mollipo` | valid | 89.1% | 0.763 | 0.6779 | 1.6786 |
| `ogbg-mollipo` | test | 76.6% | 0.734 | 0.5629 | 1.3432 |

`rho_share = |delta_P| / (|delta_L| + |delta_P| + 1e-12)` 与 `cancellation = (|delta_L| + |delta_P| - |delta_total|) / (|delta_L| + |delta_P| + 1e-12)` **只用于描述**，不是唯一贡献分解，也不是因果贡献百分比。`S1 coverage` / `S2 coverage` 是 Top-1 子图在该分子上触及的 S1 code / S2 occurrence 比例。

## 9. 失败案例

* 梯度线性预测大、真实屏蔽响应相对小的案例：48 条。选法：每个 dataset×split 取 `abs(sum_e g[e])` 最大的四分位，再在其中按真实 `abs(delta)` 升序取 8 个（结果驱动的诊断选择，不参与任何排序或筛选）。下表列出前 6 条。
  * `ogbg-molesol`/train row 26: 线性预测 abs(sum g) = 2.0876, 真实屏蔽 abs(delta) = 1.2444, 实际/线性预测 = 0.596
  * `ogbg-molesol`/train row 849: 线性预测 abs(sum g) = 2.2743, 真实屏蔽 abs(delta) = 1.4605, 实际/线性预测 = 0.642
  * `ogbg-molesol`/train row 112: 线性预测 abs(sum g) = 2.3704, 真实屏蔽 abs(delta) = 1.4848, 实际/线性预测 = 0.626
  * `ogbg-molesol`/train row 1070: 线性预测 abs(sum g) = 2.1156, 真实屏蔽 abs(delta) = 1.6527, 实际/线性预测 = 0.781
  * `ogbg-molesol`/train row 230: 线性预测 abs(sum g) = 2.0414, 真实屏蔽 abs(delta) = 1.6970, 实际/线性预测 = 0.831
  * `ogbg-molesol`/train row 1126: 线性预测 abs(sum g) = 2.6301, 真实屏蔽 abs(delta) = 1.8186, 实际/线性预测 = 0.691
* 局部输出启发式选出的子图真实响应相对小的案例：48 条。选法同上（先取启发式 `B(S)` 的 Top-1 真实响应最大四分位，再按真实 `abs(delta)` 升序取 8 个）。
  * `ogbg-molesol`/train row 354: 启发式子图 abs(delta) = 2.1554, 发现子图 abs(delta) = 1.6444
  * `ogbg-molesol`/train row 837: 启发式子图 abs(delta) = 2.0896, 发现子图 abs(delta) = 2.1463
  * `ogbg-molesol`/train row 1031: 启发式子图 abs(delta) = 2.2161, 发现子图 abs(delta) = 2.2161
  * `ogbg-molesol`/train row 491: 启发式子图 abs(delta) = 2.2029, 发现子图 abs(delta) = 2.3466
  * `ogbg-molesol`/train row 43: 启发式子图 abs(delta) = 2.5900, 发现子图 abs(delta) = 2.5243
  * `ogbg-molesol`/train row 375: 启发式子图 abs(delta) = 2.9022, 发现子图 abs(delta) = 2.5407

## 10. 判读

* **ENGINEERING_GATE**：`PASS`。
* **`ogbg-molesol` DISCOVERY_SIGNAL** = `WEAK_OR_INCONCLUSIVE`

| split | condition | value | rule | passed |
|---|---|---|---|---|
| valid | effect_floor_fraction_ge_0p50 | 1.000 | >= 0.50 of molecules reach 1% of the Train target standard deviation | yes |
| valid | effect_ratio_gt1_fraction_ge_0p70 | 0.841 | >= 0.70 | yes |
| valid | effect_ratio_median_ge_1p5 | 1.293 | >= 1.5 | no |
| valid | enough_matched_controls | 63.000 | matched molecules >= max(32, 80% of interpreted molecules) | yes |
| test | effect_floor_fraction_ge_0p50 | 1.000 | >= 0.50 of molecules reach 1% of the Train target standard deviation | yes |
| test | effect_ratio_gt1_fraction_ge_0p70 | 0.836 | >= 0.70 | yes |
| test | effect_ratio_median_ge_1p5 | 1.386 | >= 1.5 | no |
| test | enough_matched_controls | 61.000 | matched molecules >= max(32, 80% of interpreted molecules) | yes |

* **`ogbg-mollipo` DISCOVERY_SIGNAL** = `WEAK_OR_INCONCLUSIVE`

| split | condition | value | rule | passed |
|---|---|---|---|---|
| valid | effect_floor_fraction_ge_0p50 | 0.984 | >= 0.50 of molecules reach 1% of the Train target standard deviation | yes |
| valid | effect_ratio_gt1_fraction_ge_0p70 | 0.719 | >= 0.70 | yes |
| valid | effect_ratio_median_ge_1p5 | 1.568 | >= 1.5 | yes |
| valid | enough_matched_controls | 64.000 | matched molecules >= max(32, 80% of interpreted molecules) | yes |
| test | effect_floor_fraction_ge_0p50 | 1.000 | >= 0.50 of molecules reach 1% of the Train target standard deviation | yes |
| test | effect_ratio_gt1_fraction_ge_0p70 | 0.562 | >= 0.70 | no |
| test | effect_ratio_median_ge_1p5 | 1.286 | >= 1.5 | no |
| test | enough_matched_controls | 64.000 | matched molecules >= max(32, 80% of interpreted molecules) | yes |

* `DISCOVERY_SIGNAL` 是**跑之前冻结的开发性投入规则**，不是统计显著性，也不是科学结论；未通过时只说明本轮证据不足以支持继续投入。
* `ARCHITECTURE_DIAGNOSIS` 只如实输出 `rho_share`、抵消、S1/S2 交互与覆盖率证据；本轮**不**据此推导“必须改读出”或“必须改结构算子”。

## 11. 不确定性与限制

* 单 training seed（0）、单 checkpoint；所有数字都是 development evidence，没有跨 seed 稳定性证据。
* bootstrap 只描述固定 checkpoint / 固定数据集内的分子抽样不确定性。
* 搜索空间仅覆盖 2–6 条物理键的连通子图；更大的结构未被覆盖，不能据此断言模型不依赖它们。
* 干预是条件性证据屏蔽：`atom_feat`、`atom_scalar`、`global`、原子数与原子编码保持不变，因此它不是化学反事实，也不是“模型没有学到结构信息”的证明。
* 每划分只解释 64 个分子（固定种子哈希排序），覆盖全量划分的统计推断不在本轮范围。
* 执行顺序：8 分子/split 的工程 smoke 先在本地 CPU 上跑通（同一份代码、同一份冻结缓存），随后在 `res` 上以 `stage=preflight` 在 GPU 上重跑全部等价/映射/有限差分/重编号/批量门禁；两者都通过后才启动两个正式 worker。smoke 结果本身不进入任何报告指标。
* 若某个分子无法解析或映射不一致，本轮记录原始 row id 与原因，不用另一个分子补齐。

## 12. 产物

| path | content |
|---|---|
| `PROTOCOL.md` | 冻结实验定义与解释边界 |
| `analysis/preflight.json` | 数据/缓存/checkpoint/等价性/映射/梯度验证 |
| `analysis/training_source.json` | 复用身份、360 epoch 轨迹、选中 epoch、checkpoint hash |
| `analysis/main_table.csv` | 各 split 主要效果指标 |
| `analysis/architecture_diagnostics.csv` | S1/S2、局部/全图读出诊断 |
| `analysis/molecule_table.csv` | 每分子明细（可重放） |
| `analysis/bootstrap.json` | 分子级 bootstrap 区间 |
| `analysis/failure_cases.json` | 高梯度/高局部输出但低响应的失败案例 |
| `analysis/decision.json` | 冻结判读规则的结果 |
| `analysis/run_manifest.json` | commit、协议哈希、缓存/checkpoint 哈希、环境、GPU 耗时 |
| `analysis/figures/` | 分子高亮图 40 张 + 分布图 |
| `results/<dataset>/` | 样本 ID、候选搜索全log、逐分子干预记录、预算 |

* 产出逐分子记录的 GPU worker commit：`2e202646b1560e857f92549e7fa72af55fe6ca85`（两个 worker 各占一个 GPU：`res-gpu0` = ogbg-molesol，`res-gpu1` = ogbg-mollipo）。本文件由 CPU 侧 aggregate 阶段在 commit `0baa611d7cdd20c57f60415881e82fbf00df7eb7` 上重新生成，aggregate 不修改任何权重或逐分子记录。
* 协议哈希 `5b2ed182d1a6077edd8dd902ed504f67da9e03eaeaec20cc235defeec1a58118`；GPU 耗时：`ogbg-molesol` 6.1 s (0.0017 GPU·h)；`ogbg-mollipo` 6.3 s (0.0018 GPU·h)（软上限 0.5 GPU·h / 硬上限 1.0 GPU·h per worker；未触发任何预算停止）。
* 执行环境（来自 `rr` 的运行记录）：`as-lipo` stage=dataset_worker pool=res-gpu1 node=a100-2 driver=550.163.01 torch=cu124 state=completed；`as-mol` stage=dataset_worker pool=res-gpu0 node=a100-2 driver=550.163.01 torch=cu124 state=completed；`as-pf0` stage=preflight pool=res-gpu0 node=a100-2 driver=550.163.01 torch=cu124 state=completed；`as-pf1` stage=preflight pool=res-gpu1 node=a100-2 driver=550.163.01 torch=cu124 state=completed。

完整图像只在 `analysis/figures/` 中保存有限张（每数据集固定 8 Valid + 8 Test + 4 诊断案例 + 2 分布图）；数百个原始分子图不入 Git，可由运行记录恢复。

