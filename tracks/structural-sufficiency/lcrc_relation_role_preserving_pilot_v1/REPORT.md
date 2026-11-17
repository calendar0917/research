# 保留 C1 读出的 S2 结构角色 pilot（RR_REAL / RR_CONTROL）

轮次 `lcrc-relation-role-preserving-pilot-v1`。一切数字均为 **DEVELOPMENT**，Test 列是 **DEVELOPMENT_TEST**。

本轮只训练四个单元：RR_REAL / RR_CONTROL × Lipophilicity / WC_LOGP，seed 0，官方 `ogbg-mollipo` 划分 3360/420/420，Rf 配方 360 epoch。C1、GINE、oracle 基线一律只读复用，不重训、不重选、不重评。

## 1. 本轮问题

在**完全保留** C1 原子编码、键编码、S1、typed-S2、LocalMLP、`sum_v f(h_v)` 与完整 `rho([g, mean h, mean h², log1p n])` 的条件下，只把每条 S2 出现的 200 维记录按“同一拓扑结构内部 / 跨拓扑结构边界”分成两组、各乘一个逐通道可训练 gamma：

1. 预测是否恢复到 C1(Rf) 水平（≤ 1.10× Valid RMSE）？
2. 真实结构边界角色（RR_REAL）是否优于同参数量、同初始化、同配方的任意伪分组（RR_CONTROL）？
3. 对 WC_LOGP 已知原子贡献的捕获是否改善？
4. 若有改善，证据来自预测还是结构解释？
5. 是否值得继续研究显式 block/interface Token？

## 2. 冻结复用与门禁

复用的冻结拓扑分块 / S2 归属缓存来自 `lcrc_relation_interface_routing_pilot_v1`，逐文件 SHA-256 + **逐数组重算** 校验：

| split | routing cache sha256 | fingerprint match | arrays match | rebuild s |
|---|---|---|---|---|
| train | 8fc85c7fba7c8bd1… | True | True | 4.04 |
| valid | 5e43da2eb3bd99e3… | True | True | 0.51 |
| test | 890407976d3e9b9c… | True | True | 0.51 |

门禁（`analysis/routing_invariants.json`，phase = `post_training`）：**11 项**，pass = **True**，failing = []。

| gate | pass |
|---|---|
| 10_wc_logp_closure | True |
| 11_group_sizes | True |
| 1_frozen_reuse | True |
| 2_role_partition | True |
| 3_s2_conservation | True |
| 4_gamma_one_equivalence | True |
| 5_data_readout_identity | True |
| 6_relabelling | True |
| 7_parameter_identity | True |
| 8_label_oracle_blindness | True |
| 9_trained_identity | True |

关键门禁的含义：

- **角色划分完备性**：每条 S2 occurrence 恰好属于一组（REAL：`owner(e1)==owner(e2)`；CONTROL：S2 类型哈希），无遗漏、无重复；
- **S2 聚合守恒**：逐原子 `Q_internal + Q_cross == 冻结 C1 的 S2 聚合`（typed 104 维 + content 96 维）；
- **gamma=1 等价**：在 canonical C1 初始状态上，局部输入列、`h`、`f(h)`、`mean h`、`mean h²`、`rho`、预测与 `cr_models.ArmA` 完全等价；
- **重编号不变**：原子/键重编号后，分块双射、角色计数、逐原子角色聚合与预测不变；
- **C1 输入/归一化/读出未改**：433 列输入的非 S2 列逐位相同，S2 列在守恒容差内相同；Train-only 归一化与目标 scaler 复算一致；`f`/`rho` 权重与 canonical 状态一致；
- **参数同一**：REAL/CONTROL 参数量与初始状态哈希完全相同；
- **无标签/oracle 进入分组**：CONTROL 角色仅由 S2 类型码决定，REAL 角色仅由冻结分块决定；标签置乱后角色与批次特征不变。

## 3. 架构与参数预算

| 项 | 参数量 |
|---|---|
| C1（只读基线） | 228674 |
| trunk（atom/bond/S1/S2/LocalMLP） | 213376 |
| f = MLP 64→64→1 | 4225 |
| rho = MLP 171→64→1 | 11073 |
| gamma_internal + gamma_cross | 400 |
| RR_REAL / RR_CONTROL 总计 | 229074 |
| 相对 C1 增量 | 400 |

两模型参数量完全相同（ratio = 1.000000），初始状态哈希相同 = `496b0b5adc2ce1b3…`；gamma 初值全为 1：**是**。

S2 角色输入：`Q_new = gamma_internal ⊙ Q_internal + gamma_cross ⊙ Q_cross ∈ R^200`，代替原来的 `Q_internal + Q_cross`；其余 233 列（a 96 + b_sum 96 + nbr 32 + atom scalars 9）与 C1 完全相同。

### 两组 occurrence 的规模（报告，不强制对齐）

| split | model | groups | n(group0) | n(group1) | share0 | share1 | n1/n0 | 两组的分子数 |
|---|---|---|---|---|---|---|---|---|
| train | RR_REAL | ['internal', 'cross'] | 160152 | 117160 | 57.75% | 42.25% | 0.7316 | 3354 |
| train | RR_CONTROL | ['hash0', 'hash1'] | 191180 | 86132 | 68.94% | 31.06% | 0.4505 | 3355 |
| valid | RR_REAL | ['internal', 'cross'] | 21144 | 14410 | 59.47% | 40.53% | 0.6815 | 420 |
| valid | RR_CONTROL | ['hash0', 'hash1'] | 24749 | 10805 | 69.61% | 30.39% | 0.4366 | 420 |
| test | RR_REAL | ['internal', 'cross'] | 21716 | 14412 | 60.11% | 39.89% | 0.6637 | 420 |
| test | RR_CONTROL | ['hash0', 'hash1'] | 25372 | 10756 | 70.23% | 29.77% | 0.4239 | 420 |

Train 上两组规模差异：RR_REAL internal/cross = 160152 / 117160 (差 15.50%)；RR_CONTROL hash0/hash1 = 191180 / 86132 (差 37.88%)。

CONTROL 的分组只依赖 S2 类型码 (path_cat, path_end63, path_closure) 的固定 blake2b 哈希，不含原子索引、bond id、row_id、标签或 oracle；该哈希在训练前冻结在 `cache/`，盐值 `lcrc-relation-role-preserving-pilot-v1/s2-type-hash/control`。

## 4. Lipophilicity：预测、恢复与角色增量

### 4.1 预测（best-Valid checkpoint；Test = DEVELOPMENT_TEST）

| model | split | RMSE | MAE | epoch | params | comparability |
|---|---|---|---|---|---|---|
| RR_REAL | train | 0.2519 | 0.1942 | 148 | 229074 | CURRENT_ROUND |
| RR_REAL | valid | 0.7071 | 0.5203 | 148 | 229074 | CURRENT_ROUND |
| RR_REAL | test | 0.7610 | 0.5540 | 148 | 229074 | CURRENT_ROUND |
| RR_CONTROL | train | 0.2598 | 0.1936 | 115 | 229074 | CURRENT_ROUND |
| RR_CONTROL | valid | 0.6945 | 0.4880 | 115 | 229074 | CURRENT_ROUND |
| RR_CONTROL | test | 0.7069 | 0.5136 | 115 | 229074 | CURRENT_ROUND |
| C1 (frozen, Rf) | train | 0.2391 |  | 124 | 228674 | DIRECTLY_COMPARABLE |
| C1 (frozen, Rf) | valid | 0.7069 | 0.5079 | 124 | 228674 | DIRECTLY_COMPARABLE |
| C1 (frozen, Rf) | test | 0.7293 | 0.5324 | 124 | 228674 | DIRECTLY_COMPARABLE |
| GINE-G0 (frozen, Rf) | train | 0.2552 |  | 183 | 377350 | DIRECTLY_COMPARABLE |
| GINE-G0 (frozen, Rf) | valid | 0.6161 | 0.4670 | 183 | 377350 | DIRECTLY_COMPARABLE |
| GINE-G0 (frozen, Rf) | test | 0.6800 | 0.5092 | 183 | 377350 | DIRECTLY_COMPARABLE |
| GINE-Gx (frozen, Rf) | train | 0.2664 |  | 128 | 379014 | DIRECTLY_COMPARABLE |
| GINE-Gx (frozen, Rf) | valid | 0.6307 | 0.4790 | 128 | 379014 | DIRECTLY_COMPARABLE |
| GINE-Gx (frozen, Rf) | test | 0.6749 | 0.5202 | 128 | 379014 | DIRECTLY_COMPARABLE |

历史 C1(Rf) 与 GINE 行是只读基线；GINE 仅作参照，不进入任何判据。

### 4.2 预测是否恢复（PREDICTION_PRESERVED）

预登记判据：RR_REAL Lipophilicity **Valid** RMSE ≤ 1.10 × C1(Rf) Valid RMSE。

| model | valid RMSE | valid / C1 | valid − C1 | test RMSE | test / C1 | epoch | pass |
|---|---|---|---|---|---|---|---|
| RR_REAL | 0.7071 | 1.0004 | 0.0003 | 0.7610 | 1.0434 | 148 | True |
| RR_CONTROL | 0.6945 | 0.9825 | -0.0124 | 0.7069 | 0.9692 | 115 | True |

C1(Rf) 基线：Valid = 0.7069，Test = 0.7293，Epoch = 124，参数量 = 228674（`tracks/structural-sufficiency/gnn_route_clarification_8h_v1/analysis/tables/stage1.csv`）。

**PREDICTION_PRESERVED = True**；实际差距 = REAL Valid 0.0003 (100.04% of C1)。CONTROL 是否也恢复 = True。

### 4.3 REAL − CONTROL 的同分子配对 bootstrap

约定：`diff = RMSE(REAL) − RMSE(CONTROL)`，负值表示 REAL 更好；同一 split 上同一批分子，2000 次重采样，seed 20261117。

| split | n | RMSE REAL | RMSE CONTROL | diff | 95% CI diff | P(REAL better) | CI 排除 0 |
|---|---|---|---|---|---|---|---|
| train | 3360 | 0.2519 | 0.2598 | -0.0079 | [-0.0165, 0.0001] | 0.9735 | False |
| valid | 420 | 0.7071 | 0.6945 | 0.0126 | [-0.0328, 0.0572] | 0.2710 | False |
| test | 420 | 0.7610 | 0.7069 | 0.0541 | [0.0104, 0.1036] | 0.0055 | False |

### 4.4 gamma 学到的值与该冻结模型的消融响应

| model | |g_int−1| mean | max | share>0.05 | |g_cross−1| mean | max | share>0.05 | ratio cross/internal |
|---|---|---|---|---|---|---|---|
| RR_REAL | 0.0719 | 0.3325 | 55.00% | 0.0946 | 0.4796 | 67.00% | 1.317 |
| RR_CONTROL | 0.0704 | 0.3005 | 52.00% | 0.0642 | 0.3655 | 49.00% | 0.912 |

| model | split | learned | both=1 | internal=1 | cross=1 | swap (诊断) |
|---|---|---|---|---|---|---|
| RR_REAL | valid | 0.7071 | 0.7015 | 0.7037 | 0.7055 | 0.7009 |
| RR_REAL | test | 0.7610 | 0.7623 | 0.7597 | 0.7637 | 0.7727 |
| RR_CONTROL | valid | 0.6945 | 0.6940 | 0.6943 | 0.6951 | 0.6983 |
| RR_CONTROL | test | 0.7069 | 0.6999 | 0.7034 | 0.7042 | 0.7017 |

| model | split | mean |Δpred| | median |Δpred| | max |Δpred| | ΔRMSE (both=1 − learned) |
|---|---|---|---|---|---|
| RR_REAL | valid | 0.0569 | 0.0436 | 0.3550 | -0.0056 |
| RR_REAL | test | 0.0563 | 0.0425 | 0.4566 | 0.0013 |
| RR_CONTROL | valid | 0.0551 | 0.0426 | 0.2884 | -0.0005 |
| RR_CONTROL | test | 0.0569 | 0.0452 | 0.3609 | -0.0070 |

`both=1` 即把角色缩放完全关掉、退回 C1 聚合；表中的 ΔRMSE 是该冻结模型**真实使用**角色缩放的程度。`swap` 只是诊断，不进入判据。

## 5. WC_LOGP：预测、闭合、oracle 定位与屏蔽响应

### 5.1 预测与闭合

| model | split | RMSE | MAE | epoch | params | comparability |
|---|---|---|---|---|---|---|
| RR_CONTROL | train | 0.0298 | 0.0231 | 220 | 229074 | CURRENT_ROUND |
| RR_CONTROL | valid | 0.1092 | 0.0498 | 220 | 229074 | CURRENT_ROUND |
| RR_CONTROL | test | 0.1049 | 0.0531 | 220 | 229074 | CURRENT_ROUND |
| RR_REAL | train | 0.0293 | 0.0226 | 340 | 229074 | CURRENT_ROUND |
| RR_REAL | valid | 0.1179 | 0.0509 | 340 | 229074 | CURRENT_ROUND |
| RR_REAL | test | 0.0966 | 0.0507 | 340 | 229074 | CURRENT_ROUND |
| C1 (frozen, R3) | valid | 0.0626 | 0.0433 | 237 | 228674 | RECIPE_DIFFERS |
| C1 (frozen, R3) | test | 0.0781 | 0.0489 | 237 | 228674 | RECIPE_DIFFERS |

真实 WC_LOGP 原子贡献闭合（`sum_v c_v_oracle == WC_LOGP`）：

| split | rows | max |graph − label| | max |Σ atom − label| |
|---|---|---|---|
| train | 3360 | 0.0000000000 | 0.0000000000 |
| valid | 420 | 0.0000000000 | 0.0000000000 |
| test | 420 | 0.0000000000 | 0.0000000000 |

| model | split | rows | max |pred − (Σf + rho + bias)| | max |pred − saved| | RMSE recomputed |
|---|---|---|---|---|---|
| RR_REAL | valid | 420 | 0.0000002690 | 0.0000000000 | 0.1179 |
| RR_REAL | test | 420 | 0.0000001537 | 0.0000000000 | 0.0966 |
| RR_CONTROL | valid | 420 | 0.0000003074 | 0.0000000000 | 0.1092 |
| RR_CONTROL | test | 420 | 0.0000001537 | 0.0000000000 | 0.1049 |

闭合恒等式在标准化空间成立（`pred_std = Σf + rho`），raw 空间的常数截距 `bias = mean(y_train)` 显式记入 `analysis/wc_logp_closure.json`（本 task 取 3.2698；同一 task 两模型同值）。

### 5.2 Oracle enrichment（冻结候选池 + 相同匹配对照）

候选池 = 与上一轮**完全相同**的冻结拓扑 block 集合；每个模型用自己的**原生 block 屏蔽响应**`score_i = |Δ_i| / sqrt(max(1, n_bonds_i))` 选候选（无标签、无 oracle）。随机对照复用同一函数、同一 seed（20261117）与同一匹配规则（键数 + 原子数，≥8 个，最多 16 个），capture = 候选原子集合 |WC_LOGP 原子贡献| / 全分子质量。

| model | split | n | matched | mean enrich | median | positive | CI mean | CI median | capture | control mean | signal |
|---|---|---|---|---|---|---|---|---|---|---|---|
| RR_REAL | valid | 420 | 309 | 0.0023 | -0.0017 | 48.54% | [-0.0017, 0.0065] | [-0.0055, 0.0056] | 0.0959 | 0.0936 | False |
| RR_REAL | test | 420 | 310 | 0.0028 | 0.0015 | 52.58% | [-0.0019, 0.0073] | [-0.0034, 0.0063] | 0.0977 | 0.0949 | False |
| RR_CONTROL | valid | 420 | 308 | 0.0031 | -0.0020 | 47.08% | [-0.0011, 0.0072] | [-0.0063, 0.0045] | 0.0958 | 0.0927 | False |
| RR_CONTROL | test | 420 | 309 | 0.0024 | 0.0012 | 52.43% | [-0.0024, 0.0071] | [-0.0043, 0.0056] | 0.0980 | 0.0956 | False |

| split | n shared | mean enrich REAL−CONTROL | 95% CI | P(REAL>CONTROL) | REAL signal | CI 排除 0 | REAL positive | CONTROL positive |
|---|---|---|---|---|---|---|---|---|
| valid | 299 | -0.0006 | [-0.0023, 0.0010] | 0.5033 | False | False | 47.83% | 46.15% |
| test | 303 | 0.0001 | [-0.0011, 0.0013] | 0.5000 | False | False | 52.48% | 52.48% |

### 5.3 同一候选的模型屏蔽响应

对两个模型各自发现的候选（同一分子、同一 block、同一屏蔽语义），**两个模型都**重新前向一次，比较 |Δ|；同时报告逐分子全 block |Δ| 向量的 Spearman 与 argmax 一致率。

| split | candidate from | n | mean |Δ REAL| | mean |Δ CONTROL| | mean diff (REAL−CONTROL) | 95% CI | P(REAL stronger) |
|---|---|---|---|---|---|---|---|
| valid | RR_REAL | 420 | 0.3127 | 0.3222 | -0.0095 | [-0.0130, -0.0061] | 0.3810 |
| valid | RR_CONTROL | 420 | 0.3080 | 0.3225 | -0.0145 | [-0.0177, -0.0114] | 0.3119 |
| test | RR_REAL | 420 | 0.3142 | 0.3238 | -0.0095 | [-0.0125, -0.0066] | 0.3810 |
| test | RR_CONTROL | 420 | 0.3113 | 0.3245 | -0.0132 | [-0.0160, -0.0103] | 0.3190 |

| split | n molecules | mean Spearman |Δ| | median Spearman |Δ| | same argmax |
|---|---|---|---|---|
| valid | 420 | 0.9039 | 0.9451 | 89.29% |
| test | 420 | 0.9169 | 0.9500 | 90.95% |

## 6. 决策

| flag | value |
|---|---|
| CONTROL_ALSO_PRESERVED | True |
| OLD_MODELS_RETRAINED | False |
| ONLY_C1_READOUT_RECOVERY | True |
| ORACLE_LOCALISATION_SIGNAL | False |
| PREDICTION_PRESERVED | True |
| PREDICTION_RECOVERED_BOTH | True |
| ROLE_INCREMENT_ORACLE | False |
| ROLE_INCREMENT_PREDICTIVE | False |
| ROLE_INCREMENT_SHOWN | False |
| ROLE_NOT_SUPPORTED | False |
| TEST_USED_FOR_SELECTION | False |

**判定 = `ONLY_C1_READOUT_RECOVERY`**

1. **保留 C1 读出后预测是否恢复**：REAL Valid RMSE = 0.7071（C1 的 1.0004×，差距 0.0003），CONTROL = 0.6945（0.9825×）。PREDICTION_PRESERVED = `True`。

2. **真实结构边界角色是否优于同容量伪分组**：Lipo Valid ΔRMSE(REAL−CONTROL) = 0.0126，95% CI [-0.0328, 0.0572]，P(REAL better) = 0.2710；Test Δ = 0.0541，95% CI [0.0104, 0.1036]。预测增量 credit = `False`。

3. **WC_LOGP 已知贡献捕获是否改善**：Valid paired enrichment Δ = -0.0006 (95% CI [-0.0023, 0.0010], n=299)；Test Δ = 0.0001 (95% CI [-0.0011, 0.0013], n=303)。oracle 增量 credit = `False`。

4. **改善来自预测还是结构解释**：ROLE_INCREMENT_SHOWN = `False`，ROLE_INCREMENT_PREDICTIVE = `False`，ROLE_INCREMENT_ORACLE = `False`。ONLY_C1_READOUT_RECOVERY = `True`，ROLE_NOT_SUPPORTED = `False`。

5. **是否值得继续显式 block/interface Token**：见 `DECISION.md` 的下一步建议；本轮证据仅为单 seed DEVELOPMENT，Valid 上两模型 |Δ| 的 mean Spearman = 0.9039，argmax 一致率 = 89.29%。

## 7. 口径、局限与未做的事

- single training seed per unit; no multi-seed stability claim
- official Test is DEVELOPMENT_TEST: evaluated every epoch, never used for selection
- the historical C1(Rf) baseline is read-only and was not re-trained; the Lipophilicity comparison is DIRECTLY_COMPARABLE (same split/target/scaler/seed/recipe/canonical init)
- the WC_LOGP C1 baseline is recipe R3 and is flagged RECIPE_DIFFERS
- no historical architecture result of the previous round enters any decision rule
- oracle controls are matched on bond and atom count only, not on composition

- no historical model re-trained, re-selected or re-evaluated
- no new seed, no hyper-parameter search, no extra dataset
- no partition extension / learnable grouping
- no Test-based selection or tuning

### 不可直接比较的旧结果（仅附录，不进主表、不进判据）

| task | model | split | RMSE | params | label |
|---|---|---|---|---|---|
| lipo | R1 (previous round, frozen) | valid | 0.8398 | 241860 | NOT_DIRECTLY_COMPARABLE |
| lipo | R1 (previous round, frozen) | test | 0.8027 | 241860 | NOT_DIRECTLY_COMPARABLE |
| lipo | R2 (previous round, frozen) | valid | 0.8116 | 258756 | NOT_DIRECTLY_COMPARABLE |
| lipo | R2 (previous round, frozen) | test | 0.7939 | 258756 | NOT_DIRECTLY_COMPARABLE |
| wc_logp | R1 (previous round, frozen) | valid | 0.1501 | 241860 | NOT_DIRECTLY_COMPARABLE |
| wc_logp | R1 (previous round, frozen) | test | 0.1170 | 241860 | NOT_DIRECTLY_COMPARABLE |
| wc_logp | R2 (previous round, frozen) | valid | 0.1640 | 258756 | NOT_DIRECTLY_COMPARABLE |
| wc_logp | R2 (previous round, frozen) | test | 0.1542 | 258756 | NOT_DIRECTLY_COMPARABLE |

### 运行环境

- 四个训练单元的 commit: `3fa9de73b0e5b949bfa1110f92cc23fe33f5e372`（分析/汇总在本 commit）
- commit: `6dbeffab40f60f8aa2e8e647cc508d1d265a9c95` (branch `exp/lcrc-explanation-task-relevance-v1`, dirty=False)
- hostname: `a100-2`；platform: `Linux-6.8.0-138-generic-x86_64-with-glibc2.35`
- torch: `12.4`, CUDA available = `True`, devices = `['NVIDIA A100-SXM4-40GB']`
- 冻结输入 sha256: `{'c1_trajectory': '384dfe74b1bbd856bb167c0c8d0bc7dce2dc1d51da5353d57cbb3950da4dce56', 'control_table': '7cf802171ff220a360ddb4cced4c6a8572361e69ba67a963c095bf7cffb94436', 'oracle_contrib': '5690c0e3c7ba0ed51fbe30541e4a83a49abf174c6440aaebfb46a3c5cbc48257', 'routing_caches': {'test': '890407976d3e9b9c136b83eb5390c099af39358cfe91433914d6aec073fca6cd', 'train': '8fc85c7fba7c8bd11e230f0f278f9be53df70d181eae568eeedb29fbbc2500dc', 'valid': '5e43da2eb3bd99e36ae761c2bdef22833d6f263821172129fad5ec28967d1ec3'}, 'stage1_table': '93852b406c5071f0401b8c8beab8e89bb7b07386a2c0e3f864581f780d503bcd'}`

