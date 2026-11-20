# REPORT — 冻结模型的真实编辑迁移评估（不训练）

**Round** `response_dictionary_task_transfer_s0_v1`  
**Test policy** `DEVELOPMENT_TEST`；Test 只用于本报告的冻结评估，不选择 checkpoint、字典、alpha、子集或参照。  
**Frozen structure commit** `d48df88cc9cace8b9a25b6e0e80ecc944dac895e`  
**Results commit** 本文件所在的 results commit（最终答复中给出完整 SHA）。  
**Budget** CPU only；`0 GPU`，`0 A100·h`。结构目录构建 12.8 s；正式冻结评估 0.91 s wall / 10.7 s CPU；最大 RSS 160 MB。远低于 45 min 硬上限。

本轮没有训练、没有优化器、没有新 seed、没有新数据集，也没有做 LR/K/rank/panel 扫描。旧轮次的代码、结果、官方 split、字典、alpha、projector 与编辑定义均未修改；旧目录只读。

---

## 0. 冻结对象与输入核对

本轮只使用以下冻结对象：

| object | frozen state | role |
|---|---|---|
| RESP | seed0 best-Val epoch **63** | 原生 full-`F(G)` 预测 |
| JOINT | seed0 best-Val epoch **85** | 固定控制 |
| ADD | seed0 best-Val epoch **91** | 固定控制 |
| `D_RESPONSE` / `D_LATENT` / `D_FREQ` | K16, **HP_CERTIFIED** | 字典投影预测 |
| official split | OGB Lipo official scaffold split | 唯一 split |

全分子旧指标的重放全部通过（18/18 个 model×split 检查；`max_abs_diff = 0.0`，阈值 `1e-5`）：

| model | Train RMSE / MAE | Valid RMSE / MAE | Test RMSE / MAE |
|---|---:|---:|---:|
| RESP | 0.34751 / 0.26425 | 0.62874 / 0.48208 | 0.67603 / 0.52352 |
| JOINT | 0.32122 / 0.24775 | 0.66418 / 0.48900 | 0.69986 / 0.53947 |
| ADD | 0.30178 / 0.23130 | 0.67625 / 0.50102 | 0.71208 / 0.54927 |
| `D_RESPONSE` | 0.35317 / 0.26954 | 0.62502 / 0.48094 | 0.67842 / 0.52619 |
| `D_LATENT` | 0.38242 / 0.28893 | 0.62893 / 0.48238 | 0.67936 / 0.52580 |
| `D_FREQ` | 0.35691 / 0.27178 | 0.63069 / 0.48463 | 0.68016 / 0.52722 |

预测文件、checkpoint 与旧结果表的 sha256、epoch、字典版本和全部分子级 hash 均记录在 `run_manifest.json`；`analysis/frozen_checks.json` 记录 18 个检查与 `prediction_hashes`。旧 `pairs.csv` 的 `split_a/split_b` 有 **331/823** 与 official split 不一致，本轮只用 official row id 重新取 split，不沿用旧 split 列；这 331 个不一致已在结构审计中复现（`old_stored_split_mismatches_reproduced = 331`）。

---

## 1. 唯一的新真实编辑目录（label-free，先冻结）

### 1.1 定义

在全部 4200 个原始 Lipo 分子、official split、全部 raw OGB 原子/键列保持原样的前提下：

- 只取单连通分量、无显式 H 的分子；**立体化学保留**，不删除任何 raw atom/bond 列。
- 对每个 **非环单键桥**（bridge，且 OGB bond type 为 SINGLE）枚举所有合法切分；不局限于模型的 4 个 selected views。
- 变量侧 `S`：1–8 个 heavy atoms。
- 核心侧 `C`：补集，至少 10 个 heavy atoms，且至少为每个 parent 的 75%。
- `C` 的同一性：在完整属性图上判定（9 列原子、3 列键、端口、切键列），用 exact coloured-incidence-graph canonical serialization；`core_uid` 由 exact canonical string + cut-bond columns 分组，**不是** WL/hash 分桶。`c_key`/`core_id` 只是紧凑标签；`core_table.csv` 保存每个 exact core 的 canonical serialization。
- 同一 unordered pair 只保留一个定义：max `C` size → `C` canonical key → sorted `S_A,S_B` keys → cut-edge indices。
- 方向：`S` native key 字典序较小者为 A；`delta = B − A`。
- `transformation_id = (ordered S_A, S_B, cut-bond attrs)`；不包含核心 C，因此同一 transformation 可以跨不同 C/分量出现。

### 1.2 结构审计

| item | count |
|---|---:|
| eligible molecules (single component, no explicit H) | 4,199 |
| excluded multi-component rows | 1（official row 1646, valid） |
| explicit-H rows | 0 |
| duplicate canonical rows | 0 |
| molecules with ≥1 legal cut | 4,036 |
| molecules without legal cut | 163 |
| bridges total | 43,304 |
| single bridges | 36,888 |
| non-single bridges skipped | 6,416 |
| legal candidate cuts | 23,197 |
| reassembly verified | **23,197 / 23,197**（0 failures） |
| exact core groups (all candidates) | 19,396 |
| pair-bearing exact cores | 779 |
| unordered real-edit pairs | **2,597** |
| pair-bearing components | **438** |
| distinct transformations | 1,714 |
| pairs with >1 candidate definition | 934（max 6） |
| stereo-bearing molecules | 1,186 |

`S` 越界、`C` 过小、`C` 比例不足的候选分别记录为 46,920 / 526 / 3,133。每个候选都通过 parent `verify_reassembly`；`candidate_cuts.csv.gz` 含逐候选端口、cut-bond 列与 `reassembly_verified`。

**父轮 literal filter 审计**：parent `rd_fragment.enumerate_views` 额外检查 `bond_feat[:,2] != 0`。OGB 的第 2 个 bond 列是 `is_conjugated`，不是 `is_in_ring`；bridge 检验本身已经保证非环。本轮因此只按 bond type SINGLE 取合法切分，并记录该 literal filter 会额外排除 6,374 个合法候选 / 12,664 个 single-bridge 实例（仅审计，不参与筛选）。

### 1.3 与旧 823 目录的关系

| status | old pairs | meaning |
|---|---:|---|
| `EXACT_SAME_DEFINITION` | **609** | 同一 unordered row pair，且 exact `C`、port、cut-bond 列完全相同 |
| `PRESENT_DIFFERENT_DEFINITION` | **32** | 同一 row pair 仍在，但旧 core 在 full attributes 下不精确；新定义取了更大/另一个 exact core |
| `ABSENT` | **182** | 旧 core 在 full attributes 下不精确，新目录没有该 row pair |
| old row pairs present in new | 641 | 609 + 32 |
| old pairs not exact under full attributes | 214 | 208 `CUT_DIFF` + 6 `C_AND_CUT_DIFF` |
| new-only row pairs | **1,956** | 其中 249 个 `S_max=1–4`（全部 old-stereo-ineligible），1,707 个 `S_max=5–8`（1,137 个 both stereo-eligible，570 个 ineligible） |

旧目录的 **27 个 Train–Test exact pairs** 全部保留；它们的旧指标被本轮逐 pair 精确复现：RESP pair-MAE / CE-MAE = 0.43669 / 0.35680，JOINT = 0.40969 / 0.36194，ADD = 0.41467 / 0.39490（与旧 follow-up REPORT 完全一致）。旧 32 个“同 row pair 不同定义”和 182 个“absent”全部是 Train–Train，不影响旧 E_T 的 27 对。

---

## 2. 支持集与相对旧目录的增量

### 2.1 官方 split cells

| scope | pairs | molecules | non-Train mols | Test mols | Valid mols | components | cores | transforms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Train–Train | 2,133 | 1,389 | 0 | 0 | 0 | 391 | 693 | 1,375 |
| Train–Valid | 157 | 207 | 62 | 0 | 62 | 57 | 72 | 148 |
| Train–Test | 260 | 292 | 84 | 84 | 0 | 68 | 95 | 243 |
| Valid–Valid | 6 | 12 | 12 | 0 | 12 | 6 | 6 | 6 |
| Valid–Test | 23 | 38 | 38 | 20 | 18 | 16 | 18 | 23 |
| Test–Test | 18 | 26 | 26 | 26 | 0 | 11 | 12 | 17 |
| **ALL** | **2,597** | **1,594** | **162** | **91** | **71** | **438** | **779** | **1,714** |

### 2.2 固定支持集

| scope | definition | pairs | molecules | non-Train mols | Test mols | Valid mols | components | cores | transforms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **E_T** | ≥1 Test endpoint | **301** | 317 | 109 | **91** | 18 | **72** | 102 | 277 |
| **E_HH_T** | no Train endpoint, ≥1 Test | **41** | 56 | 56 | **38** | 18 | **23** | 26 | 39 |
| **E_STRICT** | E_HH_T + C in no legal Train cut + same transformation in Train–Train another component | **2** | 4 | 4 | **2** | 2 | **2** | 2 | 2 |
| E_STRICT_C_ONLY | C-only audit of E_STRICT | 2 | 4 | 4 | 2 | 2 | 2 | 2 | 2 |

E_STRICT **不为空**；只有两对：`nt00401_00984`（rows 984/401，valid/test）和 `nt00893_01864`（rows 1864/893，valid/test）。两对的 `core_seen_train_full=0`、`transform_seen_train_other_component=1`；E_STRICT 与 C-only 版本在这里相同，说明在该数据上 full-attribute 精确性没有进一步缩小 strict support。

### 2.3 相对旧 27/7/7 的增量

旧 held-out 编辑支持是 **27 Train–Test pairs / 7 components / 7 Test molecules**；旧目录没有 both-non-Train 对。新目录：

| item | old | new | increment |
|---|---:|---:|---:|
| E_T pairs | 27 | **301** | +274 |
| E_T components | 7 | **72** | +65 |
| E_T Test molecules | 7 | **91** | +84 |
| E_HH_T pairs / components / molecules | 0 | **41 / 23 / 56** | +41 / +23 / +56 |
| E_STRICT pairs / components / molecules | 0 | **2 / 2 / 4** | +2 / +2 / +4 |

E_T 的 72 个分量中，7 个含旧 exact pair，**65 个是全新的 pair-bearing components**；E_T 的 91 个 Test 分子中 84 个是新的。E_HH_T 的 41 对全部是 new-only（0 个旧 exact pair）；E_STRICT 也全部是 new-only。**但 E_T 含 260 个 Train–Test 对，不能把它称为“新核心”或“两端全留出”的泛化；只有 E_HH_T/E_STRICT 是两端非 Train。**

### 2.4 词典库存与 selected-view 覆盖（只作覆盖，不作筛选）

既有词典的 Train `S` 词表为 702 个 word：

| scope | pairs | S_A,S_B both / one / none in inventory | pairs with shared selected C view |
|---|---:|---:|---:|
| ALL | 2,597 | 1,657 / 428 / 512 | 915（35.2%） |
| E_T | 301 | 138 / 67 / 96 | 101（33.6%） |
| E_HH_T | 41 | 14 / 11 / 16 | 14（34.1%） |
| E_STRICT | 2 | 1 / 0 / 1 | **0（0%）** |

“shared selected C” 指两个 endpoint 的固定 ≤4 selected views 中存在同一个 exact C+cut；它只是解释覆盖字段，**没有**用于改变 evaluation subset。E_STRICT 两对都不在 selected-view 覆盖内，因此任何只围绕 4 个 selected views 的 word 解释都不能自动覆盖这两对。

---

## 3. 冻结标签结果

所有指标都是 `delta = B − A` 上的误差；`gain = error(reference) − error(candidate)`，正值表示 candidate 更好。`CE-MAE` 先在每个 component 内对 pair 误差取平均，再对 component 等权平均。完整逐 scope/逐 model 数值在 `analysis/metrics_by_scope.csv`、`analysis/paired_gains.csv`、`analysis/dictionary_delta_table.csv`、`analysis/E_HH_T_pairs.csv`、`analysis/E_STRICT_pairs.csv`。

### 3.1 Pair-equal MAE（括号内为 RMSE）

| scope | pairs | comps | RESP | JOINT | ADD | `D_RESPONSE` | `D_LATENT` | `D_FREQ` | ZERO |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 2,597 | 438 | 0.3403 (0.4480) | **0.3190 (0.4199)** | 0.3215 (0.4280) | 0.3435 (0.4517) | 0.3526 (0.4641) | 0.3460 (0.4577) | 0.6737 (0.9136) |
| E_T | 301 | 72 | 0.4047 (0.5156) | **0.4007 (0.4984)** | 0.4295 (0.5439) | 0.4047 (0.5160) | 0.4077 (0.5201) | 0.4035 (0.5190) | 0.6886 (0.9195) |
| E_HH_T | 41 | 23 | 0.4124 (0.5219) | 0.4037 (0.4959) | 0.4657 (0.6031) | 0.4030 (0.5156) | **0.3952 (0.5024)** | 0.4195 (0.5306) | 0.7020 (0.9034) |
| E_STRICT | 2 | 2 | 0.3261 (0.3823) | 0.2074 (0.2621) | 0.2182 (0.2636) | 0.2947 (0.3608) | 0.2967 (0.3623) | 0.3239 (0.3793) | 0.2450 (0.2593) |

### 3.2 Component-equal MAE

| scope | RESP | JOINT | ADD | `D_RESPONSE` | `D_LATENT` | `D_FREQ` | ZERO |
|---|---:|---:|---:|---:|---:|---:|---:|
| ALL | 0.3380 | **0.3135** | 0.3275 | 0.3402 | 0.3542 | 0.3465 | 0.6305 |
| E_T | 0.3904 | 0.3889 | 0.4173 | 0.3898 | **0.3873** | 0.3875 | 0.5643 |
| E_HH_T | 0.3989 | 0.4008 | 0.4311 | 0.3889 | **0.3813** | 0.4171 | 0.6766 |
| E_STRICT | 0.3261 | **0.2074** | 0.2182 | 0.2947 | 0.2967 | 0.3239 | 0.2450 |

组件等权会改变小排名：E_T 的 pair-equal 第一是 JOINT，但 CE-MAE 第一是 `D_LATENT`；E_HH_T 的 pair-equal 第一是 `D_LATENT`，CE-MAE 也是 `D_LATENT`，而 JOINT/RESP 的 CE 顺序与 pair-equal 不完全一致。两种口径下 RESP 都没有稳定优势。

### 3.3 官方 cells 的紧凑视图（pair-equal MAE）

| cell | pairs / comps | RESP | JOINT | ADD | `D_RESPONSE` | ZERO |
|---|---:|---:|---:|---:|---:|---:|
| Train–Train | 2,133 / 391 | 0.3268 | **0.3014** | 0.2994 | 0.3304 | 0.6629 |
| Train–Valid | 157 / 57 | 0.3961 | 0.3983 | 0.4096 | 0.4005 | 0.7824 |
| Train–Test | 260 / 68 | 0.4035 | **0.4002** | 0.4238 | 0.4049 | 0.6865 |
| Valid–Valid | 6 / 6 | 0.4314 | **0.3790** | 0.4472 | 0.4272 | 0.9083 |
| Valid–Test | 23 / 16 | 0.4085 | 0.4069 | 0.4500 | **0.3949** | 0.6952 |
| Test–Test | 18 / 11 | 0.4175 | **0.3997** | 0.4857 | 0.4134 | 0.7106 |
| ALL | 2,597 / 438 | 0.3403 | **0.3190** | 0.3215 | 0.3435 | 0.6737 |

### 3.4 `TRAIN_TRANSFORM_MEAN` 有定义的同一子集

`TRAIN_TRANSFORM_MEAN` 只用 Train–Train 对、按 component 等权平均、排除被评估 pair 的整个 component；无支持时为 NA，绝不 0-fill。下表是**有表格参照的同一批 pair**，不是全 scope：

| scope | supported / scope | coverage | comps | RESP | JOINT | ADD | `D_RESPONSE` | `D_LATENT` | `D_FREQ` | ZERO | TABLE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 1,071 / 2,597 | 41.2% | 322 | 0.3183 | 0.2970 | 0.2976 | 0.3225 | 0.3340 | 0.3238 | 0.6191 | 0.3977 |
| E_T | 55 / 301 | 18.3% | 33 | 0.3900 | 0.3764 | 0.3811 | 0.3897 | 0.3906 | 0.3930 | 0.5984 | 0.3919 |
| E_HH_T | 6 / 41 | 14.6% | 5 | 0.4399 | 0.3700 | 0.2999 | 0.4364 | 0.4233 | 0.4405 | 0.4433 | **0.2836** |
| E_STRICT | 2 / 2 | 100% | 2 | 0.3261 | 0.2074 | 0.2182 | 0.2947 | 0.2967 | 0.3239 | 0.2450 | **0.1358** |

E_T 上表格均值与 RESP 几乎相同（0.3919 vs 0.3900；paired gain +0.0019，CI 跨 0）。E_HH_T/E_STRICT 上表格均值点估计更好，但只有 **6 / 2** 个支持的 pair、5 / 2 个 component；E_STRICT 的 CI 基于 2 个 component，不能当作稳定结论。E_T 的表格覆盖率只有 18.3%，因此“表格均值在 E_T 上不差”本身也只适用于这个小支持子集。

### 3.5 Paired gains 与 2000 次 component bootstrap

固定 2000 次、NumPy seed **1703**、同一 scope 内所有 model/comparison 使用同一组 component 重采样；少于 2 个有效 component 时 CI 记 NA。下表只列 MAE gain 与 95% CI（完整 RMSE/CE gain 在同名 CSV）：

| scope | comparison | n / K | ΔMAE | 95% CI | 读法 |
|---|---|---:|---:|---:|---|
| E_T | RESP vs JOINT | 301 / 72 | −0.0040 | [−0.0301, +0.0230] | JOINT 点估计略好，CI 跨 0 |
| E_T | RESP vs ADD | 301 / 72 | +0.0248 | [−0.0071, +0.0559] | RESP 点估计更好，CI 跨 0 |
| E_T | `D_RESPONSE` vs RESP | 301 / 72 | +0.0001 | [−0.0045, +0.0049] | 实质相同 |
| E_T | `D_RESPONSE` vs `D_LATENT` | 301 / 72 | +0.0030 | [−0.0066, +0.0141] | 无分辨 |
| E_T | `D_RESPONSE` vs `D_FREQ` | 301 / 72 | −0.0011 | [−0.0108, +0.0073] | 无分辨 |
| E_HH_T | RESP vs JOINT | 41 / 23 | −0.0087 | [−0.0424, +0.0277] | JOINT 点估计略好，CI 跨 0 |
| E_HH_T | RESP vs ADD | 41 / 23 | +0.0533 | [+0.0006, +0.0983] | RESP 更好；小支持 |
| E_HH_T | `D_RESPONSE` vs RESP | 41 / 23 | +0.0094 | [+0.0015, +0.0180] | 字典略好 |
| E_HH_T | `D_RESPONSE` vs `D_LATENT` | 41 / 23 | −0.0079 | [−0.0203, +0.0015] | CI 跨 0 |
| E_HH_T | `D_RESPONSE` vs `D_FREQ` | 41 / 23 | +0.0164 | [−0.0006, +0.0443] | CI 跨 0 |
| E_STRICT | RESP vs JOINT | 2 / 2 | −0.1187 | [−0.1579, −0.0796] | JOINT 更好；2 对 |
| E_STRICT | RESP vs ADD | 2 / 2 | −0.1079 | [−0.1595, −0.0564] | ADD 更好；2 对 |
| E_STRICT | `D_RESPONSE` vs RESP | 2 / 2 | +0.0315 | [+0.0226, +0.0403] | 字典略好；2 对 |
| E_STRICT | `D_RESPONSE` vs `D_LATENT` | 2 / 2 | +0.0021 | [+0.0017, +0.0024] | 差异极小 |
| E_STRICT | `D_RESPONSE` vs `D_FREQ` | 2 / 2 | +0.0293 | [+0.0183, +0.0403] | 字典更好；2 对 |

`TRAIN_TRANSFORM_MEAN` 的 paired gain：

| scope | n / K | RESP vs TABLE ΔMAE | 95% CI | TABLE CE-MAE / RESP CE-MAE |
|---|---:|---:|---:|---:|
| ALL supported | 1,071 / 322 | +0.0794 | [+0.0462, +0.1146] | 0.4038 / 0.3145 |
| E_T supported | 55 / 33 | +0.0019 | [−0.0829, +0.0832] | 0.3633 / 0.3774 |
| E_HH_T supported | 6 / 5 | −0.1563 | [−0.4404, +0.0369] | 0.2273 / 0.4407 |
| E_STRICT supported | 2 / 2 | −0.1903 | [−0.3555, −0.0251] | 0.1358 / 0.3261 |

**CI 跨 0 只表示在该支持和该抽样下方向未分辨，不表示等价；CI 也不代表跨训练 seed 的稳定性。**

### 3.6 Influence：E_T 逐一去掉一个 Test 分子

对 E_T 的 91 个 Test 分子，逐个删除该分子的所有 E_T pair，重算 gain（`analysis/influence_E_T.json`）：

| comparison | full gain | min | max | sign reversals / 91 |
|---|---:|---:|---:|---:|
| RESP vs JOINT | −0.0040 | −0.0090 | **+0.0028** | 3 |
| RESP vs ADD | +0.0248 | +0.0167 | +0.0321 | 0 |
| RESP vs TABLE | +0.0019 | −0.0212 | +0.0229 | 14 |
| `D_RESPONSE` vs RESP | +0.0001 | −0.0011 | +0.0010 | 30 |

RESP vs JOINT 的“JOINT 略好”在删除 3 个 Test 分子后会翻转；RESP vs ADD 的符号在 91 次删除中不翻转，但其 CI 在 E_T 上跨 0；RESP vs TABLE 和 `D_RESPONSE` vs RESP 都容易翻转。没有任何一个 E_T 比较在“方向 + CI + influence”三方面同时稳定。

### 3.7 字典 delta 保真 vs 对真实 `delta_y` 的误差

“fidelity”是字典预测的 delta 对**原生 RESP** delta 的 RMSE；“label error”是对真实 `delta_y` 的 MAE。两者分表报告：

| scope | native RESP label MAE | `D_RESPONSE` fidelity RMSE / label MAE | `D_LATENT` fidelity RMSE / label MAE | `D_FREQ` fidelity RMSE / label MAE |
|---|---:|---:|---:|---:|
| ALL | 0.3403 | 0.03077 / 0.3435 | 0.09110 / 0.3526 | 0.06132 / 0.3460 |
| E_T | 0.4047 | 0.03415 / 0.4047 | 0.09822 / 0.4077 | 0.09485 / 0.4035 |
| E_HH_T | 0.4124 | 0.03515 / 0.4030 | 0.12741 / 0.3952 | 0.12421 / 0.4195 |
| E_STRICT | 0.3261 | 0.03266 / 0.2947 | 0.03059 / 0.2967 | **0.00310 / 0.3239** |

`D_RESPONSE` 在 ALL/E_T/E_HH_T 上都是三字典中保真最好的；但在 E_T 上它的 label MAE 与原生 RESP 相同（0.4047），在 ALL 上反而略差（0.3435 vs 0.3403）。`D_LATENT` 在 E_HH_T 上保真更差但 label MAE 更低（0.3952 vs RESP 0.4124）；`D_FREQ` 在 E_STRICT 上保真极好（0.0031）但 label MAE 更差（0.3239 vs `D_RESPONSE` 0.2947）。保真与 label 优势不是同一件事。

### 3.8 反例（不是 cherry-pick，按误差大小列出的具体 pair）

- **E_STRICT `nt00893_01864`**：`delta_y=+0.33`；RESP 预测 `-0.1955`（误差 `-0.5255`），JOINT `-0.0376`（`-0.3676`），ADD `-0.0361`（`-0.3661`），ZERO `0.0000`（`-0.3300`），TABLE `+0.1600`（`-0.1700`）。RESP 比 ZERO 和 JOINT/ADD/TABLE 都差。
- **E_STRICT `nt00401_00984`**：`delta_y=+0.16`；RESP 误差 `-0.1267`，ZERO `-0.1600`，JOINT `+0.0472`，TABLE `-0.1017`。RESP 比 ZERO 好，但比 JOINT/ADD/TABLE 差；两对合起来 RESP 的 MAE（0.3261）仍差于 ZERO（0.2450）。
- **E_T `nt00202_02121`**：`delta_y=-1.26`；RESP 误差 `+1.3861`，JOINT `+1.3807`，ZERO `+1.2600`；ZERO 最好。
- **E_T `nt00037_01164`**：`delta_y=-1.78`；RESP 误差 `+1.3771`，JOINT `+1.0269`，ADD `+1.0251`；RESP 更差。
- **E_HH_T `nt00807_01729`**：`delta_y=+3.16`；RESP 误差 `-0.9736`，JOINT `-0.7381`，ADD `-1.3600`，ZERO `-3.1600`；RESP 不是最好。

这些反例说明：即使 RESP 在 E_T/E_HH_T 的总 MAE 好于 ZERO，它也会在个别真实编辑上方向错误，且并不稳定优于 JOINT/ADD/表格均值。

---

## 4. 回答 A–D

### A. 新目录是否增加了真正独立的官方 held-out 支持，尤其是两端非 Train / 新核心？

**增加了，但“真正独立”的程度分层明显。** 相对旧目录的 27 pairs / 7 components / 7 Test molecules，新 E_T 达到 **301 pairs / 72 components / 91 Test molecules**；其中 65 个 component 和 84 个 Test 分子是旧目录没有的。E_HH_T（无 Train endpoint、≥1 Test）从 **0** 增加到 **41 pairs / 23 components / 56 molecules（38 Test + 18 Valid）**，全部是 new-only。E_STRICT（两端非 Train + C 不在任何合法 Train 切分 + 同一 transformation 在别的分量的 Train–Train 出现过）从 **0** 增加到 **2 pairs / 2 components / 4 molecules**。

因此：**E_T 和 E_HH_T 的独立官方支持确实扩大了**；但 E_T 的 301 对里 260 对是 Train–Test，只能说明“涉及留出分子的编辑响应”，不能当作新核心或两端全留出的泛化。真正严格的新核心、两端非 Train 支持只有 **E_STRICT 的 2 对 / 2 分量 / 4 分子**，只能作为存在性证据，不能支撑规律性结论。旧 27 个 exact Train–Test pairs 全部保留且指标逐对复现；新 E_T 的增量主要来自 274 个 new-only 对。

### B. 在可比较且有支持的集合上，RESP 对真实编辑变化是否优于既定对照？支持多大、哪些反例？

**没有稳定地优于既定对照。** 分层结论：

- **E_T（301 pairs / 72 components）**：RESP pair-MAE 0.4047；JOINT 0.4007（RESP 差 0.0040，CI [−0.0301,+0.0230]）；ADD 0.4295（RESP 好 0.0248，CI [−0.0071,+0.0559]）；`D_RESPONSE` 0.4047（相同）；ZERO 0.6886（RESP 明显好于 zero）。CE-MAE 上 RESP 0.3904，JOINT 0.3889，`D_LATENT` 0.3873，`D_RESPONSE` 0.3898。对 JOINT 的负 gain 在 leave-one-Test-molecule-out 中有 3/91 次翻转，说明连这 0.004 也不稳。
- **E_HH_T（41 pairs / 23 components）**：RESP 0.4124；JOINT 0.4037（RESP 差 0.0087，CI 跨 0）；ADD 0.4657（RESP 好 0.0533，CI [+0.0006,+0.0983]）；`D_RESPONSE` 0.4030，`D_LATENT` 0.3952（都比 RESP 好）；ZERO 0.7020。RESP 唯一 CI 在 0 以上的优势是对 ADD，但这是 41 对 / 23 分量的单一支持，CE gain 也只有 +0.0322。
- **E_STRICT（2 pairs / 2 components）**：RESP 0.3261，差于 JOINT 0.2074、ADD 0.2182、`D_RESPONSE` 0.2947、`D_LATENT` 0.2967、ZERO 0.2450；只有 `D_FREQ` 0.3239 与它接近。TABLE 0.1358 最好。两对的具体反例见 §3.8；这是本轮最强的反例。
- **表格均值支持子集**：E_T 只有 55/301（18.3%，33 components），RESP vs TABLE +0.0019（CI 跨 0）；E_HH_T 只有 6/41（14.6%，5 components），RESP vs TABLE −0.1563（CI 跨 0）；E_STRICT 2/2，RESP vs TABLE −0.1903（CI [−0.3555,−0.0251]，但 K=2）。表格均值在 E_HH_T/E_STRICT 的点估计更好，但支持太小，且 E_T 覆盖率仅 18.3%，不能据此说“表格均值普遍更好”。

**支持大小**：E_T 301/72、E_HH_T 41/23、E_STRICT 2/2；表格参照进一步缩小到 55/33、6/5、2/2。**反例**：E_STRICT 两对全部是反例（尤其 `nt00893_01864`）；E_T 的 `nt00202_02121`、`nt00037_01164`，E_HH_T 的 `nt00807_01729`。结论是：RESP 相对 ZERO 在 E_T/E_HH_T 上有明显更低的 MAE（但 E_STRICT 反转），不能据此说它捕捉了唯一的真实机制；相对 JOINT/ADD/字典/表格均值没有稳定优势。RESP 只在 E_T/E_HH_T 的 ADD 对照上有一个方向为正、但 E_T CI 跨 0 的点估计。

### C. 响应字典保留了多少原生模型的真实任务信号，还是只保留了冻结输出？

**`D_RESPONSE` 很好地保留了原生 RESP 的“变化函数”，但没有把它变成更好的真实标签预测。** 保真 RMSE 为 ALL 0.0308、E_T 0.0342、E_HH_T 0.0352、E_STRICT 0.0327，均明显好于 `D_LATENT`（0.091/0.098/0.127/0.031）和 `D_FREQ`（0.061/0.095/0.124/0.003）在 ALL/E_T/E_HH_T 上的保真。就“字典是否复现原生 RESP 的 delta”而言，`D_RESPONSE` 是三字典中最好的；这就是它保留的“冻结输出”部分。

但 label-level 的增量很小且不一致：`D_RESPONSE` 的 MAE 在 ALL 为 0.3435（RESP 0.3403，略差），E_T 为 0.4047（与 RESP 相同），E_HH_T 为 0.4030（RESP 0.4124，好 0.0094，CI [+0.0015,+0.0180]），E_STRICT 为 0.2947（RESP 0.3261，好 0.0315，但只有 2 对）。它没有在 E_T 上带来可分辨的 label 改进，也没有改变“RESP/字典都不是 E_STRICT 上最好”的事实。`D_LATENT` 在 E_HH_T 上保真更差但 label MAE 更低（0.3952）；`D_FREQ` 在 E_STRICT 上保真极好但 label MAE 更差（0.3239）。所以：**响应字典保留的是原生模型的 delta 形状（fidelity），不是额外的真实任务信号；保真优势与 label 优势不能互相替代。** 与“仅保留冻结输出”相比，它增加了一层约 0.03 RMSE 的投影误差，并在小支持上偶然平滑出 0.01–0.03 的 MAE 改善，但没有一致的任务级增益。

### D. 还有什么仍然无法回答？

- **跨训练状态/seed 的稳定性**：本轮只有一个冻结状态（RESP e63、JOINT e85、ADD e91、HP_CERTIFIED 字典）；bootstrap 只条件于这个状态，不能回答换 seed/配方后排序是否保持。
- **E_STRICT 的 2 对是否代表任何总体**：2 个 component、4 个分子，其中一对主导了 RESP 的误差；不能据此建立“新核心编辑”的规律，也不能估计总体错误率。
- **E_HH_T / 表格支持子集的稳定性**：E_HH_T 只有 41 对 / 23 分量，表格支持只有 6 对 / 5 分量；E_STRICT 的表格 CI 只有 2 个 component。点估计的优劣不足以判定。
- **编辑定义与支持规模的因果作用**：本轮固定了 S≤8、C≥10、C≥75%、单键桥、full-attribute exact 匹配；没有（也不允许）扫描更大 S、不同 C 阈值或不同 matching。因此无法回答“换一个合法编辑定义会不会改变结论”。
- **selected views / 词典 702→136 压缩的任务充分性**：shared selected C 只覆盖 915/2597（35.2%），E_STRICT 为 0/2；本轮只报告覆盖，没有把 selected-view 解释当作因果机制，也没有重新估计 702→136 的 S 词表压缩。
- **误差来源分解**：无法把 RESP 的 edit error 分解为“模型没学到”“编辑定义噪声”“标签/数据噪声”或“核心/侧链不匹配”；本轮没有训练或干预。
- **E_T 的 Train–Test 部分**：即使 E_T 总体好于 ZERO，也不能把它写成两端留出或新核心泛化；只有 E_HH_T/E_STRICT 支持那种说法，而后者只有 2 对。
- **是否值得进入下一轮**：本轮不回答，也不启动；由研究者决定。

---

## 5. 复现与产物

```bash
cd /home/calendar/code/research
uv run python tracks/structural-sufficiency/response_dictionary_task_transfer_s0_v1/code/tt_catalogue.py
uv run python tracks/structural-sufficiency/response_dictionary_task_transfer_s0_v1/code/tt_evaluate.py
uv run python tracks/structural-sufficiency/response_dictionary_task_transfer_s0_v1/code/tt_run.py
```

- 结构冻结：`analysis/structure/structure_freeze.json`；`structure_hash = cbed04e4c07cdd8d124717bac08d3335ab4b7f41fffb6da2493fa80423c25992`。
- 逐候选切分：`analysis/structure/candidate_cuts.csv.gz`；exact core 表（含 canonical serialization）：`analysis/structure/core_table.csv`。
- 目录/分量/官方映射/旧目录对比：`analysis/structure/catalogue_pairs.csv`、`components.csv`、`pair_components.csv`、`official_rows.csv`、`old_pair_comparison.csv`、`new_source_strata.csv`、`support_by_cell.csv`、`support.json`、`exclusions.json`。
- 标签结果：`analysis/pairs.csv`、`metrics_by_scope.csv`、`metrics_table_mean_subset.csv`、`paired_gains.csv`、`bootstrap_plan.json`、`dictionary_delta_table.csv`、`influence_E_T.json`、`E_HH_T_pairs.csv`、`E_STRICT_pairs.csv`、`frozen_full_metrics.csv`、`frozen_checks.json`、`summary.json`。
- 运行记录：`run_manifest.json`（code commit、input/artifact sha256、命令、wall/CPU、max RSS、GPU=none、`gpu_hours_a100=0.0`、exceptions）。

两个提交号：结构/协议/代码提交为 `d48df88cc9cace8b9a25b6e0e80ecc944dac895e`；results 提交为包含本 REPORT 的 commit（最终答复给出完整 SHA）。大缓存与 checkpoint 未进入 Git；旧轮次文件未改动。
