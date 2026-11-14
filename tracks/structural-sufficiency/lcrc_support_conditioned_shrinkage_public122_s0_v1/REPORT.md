# 支持度条件化的联合块收缩 — 报告（中文）

- round: `lcrc-support-conditioned-shrinkage-public122-s0-v1`
- 状态: **DEVELOPMENT / DEVELOPMENT_TEST**（Test 参与路线选择：`TEST_USED_FOR_ROUTE_SELECTION`；单 seed 0；不称独立最终测试）
- 判定: **SPLIT_OR_SELECTION_SENSITIVE**
- 冻结 key: `K_FINE`；购买 fit: ['100', '25']；完成正式训练 10 条
- 执行 regime: cpu — 见 §7（`res` 部署被共享文件系统写满阻断，记录为基础设施失败）

## 0. 四个必答问题

**① 代理是否有可判别性？**
结构上**通过**：`K_FINE` 与 `K_JOINT` 在 Lipo fit100/fit25 全部门槛组合均 PASS，按固定优先规则取 `K_FINE`。fit100 的门槛实际值：E1_active_groups=3080.000✓ · E1_active_keys=2242.000✓ · E2_spread_fullfit=0.460✓ · E3_spread_oof=0.538✓ · E4_unknown_share=0.116✓ · E5_high_share=0.714✓ · E5_low_share=0.227✓。

active 根仅占全部根的一小部分（Lipo train 14.7%），因为 `J_full ≡ J_marg` 的根本来就不受干预（该根内所有 path 归到同一个 `b = path_cat % 5` 组，边际和恒等于全和）；inactive 根一律 `a=1`。实测门控跨度为 `q90-q10`：完整 fit 0.460、OOF 0.538（门槛 0.20）。

**② 选择性是否超过两种均匀及反向？**
- fit100（相对增益，正=SELECT 更好）：BASE: Valid -1.17% / Test +2.76%；UNIFORM: Valid +8.38% / Test -2.88%；UNIFORM_OOF: Valid +3.92% / Test +3.02%；REVERSE: Valid +5.11% / Test +2.50%
- fit25（相对增益，正=SELECT 更好）：BASE: Valid +4.47% / Test +3.25%；UNIFORM: Valid +4.47% / Test +3.32%；UNIFORM_OOF: Valid +2.11% / Test +3.68%；REVERSE: Valid +4.78% / Test +3.06%

配对 canonical-structure cluster bootstrap（2000 次，固定选中权重，共用同任务索引）在 §2 表里给出区间；区间跨 0 的项一律如实标注。

**③ 低标签与全量是否一致？**
- fit100：SELECT>BASE 两 split RMSE≥1% = False；MAE 不恶化 = True；同时优于两种 UNIFORM 与 REVERSE = False
- fit25：SELECT>BASE 两 split RMSE≥1% = True；MAE 不恶化 = False；同时优于两种 UNIFORM 与 REVERSE = True
- valid · SELECT_vs_BASE：gain(fit25)−gain(fit100) = 0.0425，区间 [-0.0276, +0.1151]
- valid · SELECT_vs_UNIFORM：gain(fit25)−gain(fit100) = -0.0251，区间 [-0.0734, +0.0322]
- valid · SELECT_vs_UNIFORM_OOF：gain(fit25)−gain(fit100) = -0.0107，区间 [-0.0635, +0.0603]
- test · SELECT_vs_BASE：gain(fit25)−gain(fit100) = 0.0081，区间 [-0.0367, +0.0597]
- test · SELECT_vs_UNIFORM：gain(fit25)−gain(fit100) = 0.0413，区间 [-0.0131, +0.0949]
- test · SELECT_vs_UNIFORM_OOF：gain(fit25)−gain(fit100) = 0.0099，区间 [-0.0469, +0.0660]

**④ 下一步保留什么、停止什么？**
- 保留：`SPLIT_OR_SELECTION_SENSITIVE` —— 同一 panel 里 SELECT 的符号在 Valid 与 Test、fit100 与 fit25 之间翻转（§2、§3 逐项给出），因此**不**升级为可复现信号。
- 停止：不再追加 seed、不再搜索第三个 key / 新的 alpha 形状、不采购 BACE 面板（`NOT_PURCHASED_BY_BUDGET`：Lipo 未过晋升门槛）；若继续，必须先解决**选点口径与 split 冲突**（R_SEL 用 Valid 选点，而收益出现在 Test），而不是加大干预强度。

## 1. 结构准备与 Train-only 可行性（零训练、零标签）

| dataset | 候选键数 K_FINE / K_JOINT | active 根占比 train / valid / test |
|---|---|---|
| Lipophilicity | 6891 / 3772 | 14.7% / 14.2% / 14.7% |
| BACE | 2307 / 1262 | 11.4% / 12.4% / 12.0% |

| dataset | fit | key | E1 groups | E1 keys | E2 spread | E3 spread | E4 unknown | E5 low | E5 high | pass |
|---|---|---|---|---|---|---|---|---|---|---|
| BACE | 100 | K_FINE | 1181 | 468 | 0.504 | 0.521 | 0.045 | 0.109 | 0.857 | PASS |
| BACE | 100 | K_JOINT | 1181 | 352 | 0.502 | 0.513 | 0.026 | 0.078 | 0.889 | FAIL |
| Lipophilicity | 100 | K_FINE | 3080 | 2242 | 0.460 | 0.538 | 0.116 | 0.227 | 0.714 | PASS |
| Lipophilicity | 100 | K_JOINT | 3080 | 1606 | 0.489 | 0.510 | 0.079 | 0.161 | 0.789 | PASS |
| Lipophilicity | 25 | K_FINE | 761 | 1081 | 0.475 | 0.542 | 0.239 | 0.448 | 0.411 | PASS |
| Lipophilicity | 25 | K_JOINT | 761 | 823 | 0.488 | 0.537 | 0.178 | 0.348 | 0.531 | PASS |

冻结结果：`KEYS_PASS_BOTH_N`，key = `K_FINE`，购买 ['100', '25']；`a_oof` = F100=0.6873, F25=0.6602。
结构加权平均 `a_SELECT` 在 fit 参照权重下 = 0.700000000000（容差 1e-10）。

`K_JOINT` 在 BACE 上因 E5 low-share 未过（0.075 < 0.10），但 Lipo 两 key 都过，按优先级仍取 `K_FINE`；若采购 BACE 也沿用同一 key，不另选粒度。

自计数剂量差（fit 表加权均值 − OOF 表加权均值）：F100=0.0127、F25=0.0398。

**关键等价关系**：inactive 根的 `J_full−J_marg ≡ 0`，因此 `a` 在那些根上无作用；`SELECT` 与 `UNIFORM` 的加权均值都被固定为 0.70（容差 1e-10，见 `analysis/freeze.json` 的锁检查），所以两者的差别**纯粹是 active 根上 a 的分配**（低支持→0.40–0.55，高支持→0.94–1.00），不是总收缩剂量。`REVERSE` 是同剂量、反方向的分配；`UNIFORM_OOF` 是同方向、常数剂量的分配。这是本轮的四个臂恰好构成 2×2（选择性 × 方向）对照的原因。

自计数口径：Train 查询使用**同一完整 fit 表**，因此 fit 结构会计入自身；OOF 表（4/5 结构）只用于门槛与 `a_oof` 常数，正式模型始终用完整 fit 表。支持计数是**结构支持度**，不是独立实验标签数。

## 2. 五臂主表与预先固定的四项比较

### fit100

| arm | update | Train_fit | Valid | Test | Test MAE | params | wall (s) |
|---|---|---|---|---|---|---|---|
| BASE | 5443 | 0.1716 | 0.6488 | 0.5954 | 0.4361 | 228674 | 1343.5 |
| UNIFORM | 6221 | 0.1544 | 0.7163 | 0.5628 | 0.4081 | 228674 | 1478.4 |
| UNIFORM_OOF | 6221 | 0.1526 | 0.6831 | 0.5970 | 0.4352 | 228674 | 1476.7 |
| SELECT | 6221 | 0.1458 | 0.6563 | 0.5790 | 0.4221 | 228674 | 1473.3 |
| REVERSE | 5832 | 0.1544 | 0.6917 | 0.5938 | 0.4300 | 228674 | 1535.5 |

| 比较 | Valid 增益 | Valid 相对 | Test 增益 | Test 相对 | Test ΔMAE | bootstrap(结构簇, Test) |
|---|---|---|---|---|---|---|
| SELECT vs BASE | -0.0076 | -1.17% | 0.0164 | +2.76% | -0.0140 | [-0.0118, +0.0489] (跨0) |
| SELECT vs UNIFORM | 0.0600 | +8.38% | -0.0162 | -2.88% | 0.0140 | [-0.0475, +0.0183] (跨0) |
| SELECT vs UNIFORM_OOF | 0.0268 | +3.92% | 0.0180 | +3.02% | -0.0131 | [-0.0175, +0.0578] (跨0) |
| SELECT vs REVERSE | 0.0354 | +5.11% | 0.0149 | +2.50% | -0.0079 | [-0.0185, +0.0497] (跨0) |

| 对照 vs BASE | Valid 相对 | Test 相对 |
|---|---|---|
| UNIFORM | -10.41% | +5.48% |
| UNIFORM_OOF | -5.29% | -0.27% |
| REVERSE | -6.62% | +0.26% |

### fit25

| arm | update | Train_fit | Valid | Test | Test MAE | params | wall (s) |
|---|---|---|---|---|---|---|---|
| BASE | 7387 | 0.0509 | 0.7804 | 0.7562 | 0.5458 | 228674 | 1422.6 |
| UNIFORM | 972 | 0.2718 | 0.7805 | 0.7567 | 0.5822 | 228674 | 1528.8 |
| UNIFORM_OOF | 583 | 0.4274 | 0.7616 | 0.7596 | 0.5781 | 228674 | 1052.3 |
| SELECT | 5638 | 0.0624 | 0.7455 | 0.7317 | 0.5526 | 228674 | 1557.8 |
| REVERSE | 5832 | 0.0708 | 0.7830 | 0.7547 | 0.5575 | 228674 | 1012.1 |

| 比较 | Valid 增益 | Valid 相对 | Test 增益 | Test 相对 | Test ΔMAE | bootstrap(结构簇, Test) |
|---|---|---|---|---|---|---|
| SELECT vs BASE | 0.0349 | +4.47% | 0.0245 | +3.25% | 0.0069 | [-0.0098, +0.0629] (跨0) |
| SELECT vs UNIFORM | 0.0349 | +4.47% | 0.0251 | +3.32% | -0.0296 | [-0.0114, +0.0620] (跨0) |
| SELECT vs UNIFORM_OOF | 0.0161 | +2.11% | 0.0279 | +3.68% | -0.0254 | [-0.0169, +0.0717] (跨0) |
| SELECT vs REVERSE | 0.0375 | +4.78% | 0.0231 | +3.06% | -0.0049 | [-0.0035, +0.0473] (跨0) |

| 对照 vs BASE | Valid 相对 | Test 相对 |
|---|---|---|
| UNIFORM | -0.00% | -0.07% |
| UNIFORM_OOF | +2.41% | -0.45% |
| REVERSE | -0.33% | +0.20% |

主选择规则 `R_SEL`（50 点共同 grid，Valid RMSE 最小，同分取较早）；Test 从不参与选点。

Murcko-scaffold 簇敏感性（同一重采样索引，Test）：
- fit100: SELECT_vs_BASE=0.0164 [-0.0130, 0.0497]；SELECT_vs_REVERSE=0.0149 [-0.0222, 0.0501]；SELECT_vs_UNIFORM=-0.0162 [-0.0505, 0.0184]；SELECT_vs_UNIFORM_OOF=0.0180 [-0.0173, 0.0634]
- fit25: SELECT_vs_BASE=0.0245 [-0.0089, 0.0659]；SELECT_vs_REVERSE=0.0231 [-0.0056, 0.0516]；SELECT_vs_UNIFORM=0.0251 [-0.0141, 0.0673]；SELECT_vs_UNIFORM_OOF=0.0279 [-0.0205, 0.0769]

**晋升门槛（§5，冻结）逐项判定**

| fit | SELECT>BASE 两 split RMSE≥1% | MAE 不恶化(≤1%) | 优于 2×UNIFORM+REVERSE 两 split | pass |
|---|---|---|---|---|
| 100 | False | True | False | False |
| 25 | True | False | True | False |

选点敏感性（为什么判定为 split/selection 敏感）：
- F100|BASE: R_SEL update=5443，A_SEL update=5443，W_SEL(5 点均值) update=5443
- F100|UNIFORM: R_SEL update=6221，A_SEL update=3888，W_SEL(5 点均值) update=6221
- F100|UNIFORM_OOF: R_SEL update=6221，A_SEL update=6221，W_SEL(5 点均值) update=6221
- F100|SELECT: R_SEL update=6221，A_SEL update=6221，W_SEL(5 点均值) update=6221
- F100|REVERSE: R_SEL update=5832，A_SEL update=5443，W_SEL(5 点均值) update=5832
- F25|BASE: R_SEL update=7387，A_SEL update=3499，W_SEL(5 点均值) update=7387
- F25|UNIFORM: R_SEL update=972，A_SEL update=972，W_SEL(5 点均值) update=972
- F25|UNIFORM_OOF: R_SEL update=583，A_SEL update=3305，W_SEL(5 点均值) update=583
- F25|SELECT: R_SEL update=5638，A_SEL update=5638，W_SEL(5 点均值) update=5638
- F25|REVERSE: R_SEL update=5832，A_SEL update=2333，W_SEL(5 点均值) update=5832

在 fit100 上，三个受门控的臂（UNIFORM / UNIFORM_OOF / SELECT）恰好都选到同一个 grid 点，而它们的 Valid/Test 排序相反（Valid: SELECT<UNIFORM_OOF<UNIFORM；Test: UNIFORM<SELECT<UNIFORM_OOF）。在 fit25 上 UNIFORM 与 UNIFORM_OOF 的 R_SEL 落在极早的 grid 点（早期即过拟合），说明该规则在低标签下对单点噪声非常敏感——这正是把它作为**唯一主规则并如实报告**的理由，而不是事后换规则。

## 3. 低标签（fit25 = 840 行嵌套子集）与全量

| split | 比较 | gain fit100 | gain fit25 | gain25−gain100 | 区间 |
|---|---|---|---|---|---|
| valid | SELECT_vs_BASE | -0.0076 | 0.0349 | 0.0425 | [-0.0276, +0.1151] |
| valid | SELECT_vs_UNIFORM | 0.0600 | 0.0349 | -0.0251 | [-0.0734, +0.0322] |
| valid | SELECT_vs_UNIFORM_OOF | 0.0268 | 0.0161 | -0.0107 | [-0.0635, +0.0603] |
| test | SELECT_vs_BASE | 0.0164 | 0.0245 | 0.0081 | [-0.0367, +0.0597] |
| test | SELECT_vs_UNIFORM | -0.0162 | 0.0251 | 0.0413 | [-0.0131, +0.0949] |
| test | SELECT_vs_UNIFORM_OOF | 0.0180 | 0.0279 | 0.0099 | [-0.0469, +0.0660] |

两个 N 是同 seed 的两个拟合量；不能当作样本复杂度或噪声下限估计。

## 4. 支持解释与影响度（兼容性分析，不是因果分解）

### fit100：负担阈值（Train-only OOF 的 q25/q75）= 0.20174 / 0.41424

| split | group | n | RMSE BASE | RMSE UNIFORM | RMSE SELECT | mean ΔSSE SELECT−BASE | vs UNIFORM |
|---|---|---|---|---|---|---|---|
| valid | low | 133 | 0.4081 | 0.4175 | 0.4227 | 0.01213 | 0.00433 |
| valid | high | 109 | 0.9889 | 1.1654 | 1.0083 | 0.03890 | -0.34146 |
| test | low | 127 | 0.5110 | 0.4950 | 0.4857 | -0.02516 | -0.00904 |
| test | high | 108 | 0.7442 | 0.7165 | 0.7620 | 0.02674 | 0.06720 |

unknown-key 占比（active 根内）：valid 0.099、test 0.093；无 active 根的分子占比：valid 0.074、test 0.098。

### fit25：负担阈值（Train-only OOF 的 q25/q75）= 0.22076 / 0.46201

| split | group | n | RMSE BASE | RMSE UNIFORM | RMSE SELECT | mean ΔSSE SELECT−BASE | vs UNIFORM |
|---|---|---|---|---|---|---|---|
| valid | low | 126 | 0.5457 | 0.5217 | 0.5153 | -0.03228 | -0.00660 |
| valid | high | 95 | 1.0708 | 1.1333 | 1.0260 | -0.09377 | -0.23169 |
| test | low | 126 | 0.6987 | 0.7057 | 0.6482 | -0.06813 | -0.07790 |
| test | high | 105 | 0.9781 | 0.9607 | 0.9583 | -0.03837 | -0.00465 |

unknown-key 占比（active 根内）：valid 0.180、test 0.200；无 active 根的分子占比：valid 0.074、test 0.098。

### 影响度：按本轮 BASE 绝对误差降序固定删除（五臂删同一行集）

| fit | split | 口径 | BASE | UNIFORM | UNIFORM_OOF | SELECT | REVERSE |
|---|---|---|---|---|---|---|---|
| 100 | valid | 全样本 | 0.6488 | 0.7163 | 0.6831 | 0.6563 | 0.6917 |
| 100 | valid | 删最大 0.01 | 0.5135 | 0.5379 | 0.4949 | 0.5099 | 0.5304 |
| 100 | valid | 删最大 0.05 | 0.4311 | 0.4641 | 0.4392 | 0.4513 | 0.4563 |
| 100 | test | 全样本 | 0.5954 | 0.5628 | 0.5970 | 0.5790 | 0.5938 |
| 100 | test | 删最大 0.01 | 0.5524 | 0.5275 | 0.5544 | 0.5430 | 0.5585 |
| 100 | test | 删最大 0.05 | 0.4671 | 0.4458 | 0.4786 | 0.4667 | 0.4860 |
| 25 | valid | 全样本 | 0.7804 | 0.7805 | 0.7616 | 0.7455 | 0.7830 |
| 25 | valid | 删最大 0.01 | 0.6719 | 0.6851 | 0.6741 | 0.6599 | 0.6867 |
| 25 | valid | 删最大 0.05 | 0.5757 | 0.6046 | 0.6147 | 0.5759 | 0.5947 |
| 25 | test | 全样本 | 0.7562 | 0.7567 | 0.7596 | 0.7317 | 0.7547 |
| 25 | test | 删最大 0.01 | 0.6925 | 0.7322 | 0.7353 | 0.6887 | 0.7069 |
| 25 | test | 删最大 0.05 | 0.5838 | 0.6465 | 0.6630 | 0.6037 | 0.6146 |

辅规则与窗口：
- F100|BASE: Test 最后 10 个 grid 点中位数 0.5991，grid 最好 0.5902
- F100|REVERSE: Test 最后 10 个 grid 点中位数 0.5928，grid 最好 0.5862
- F100|SELECT: Test 最后 10 个 grid 点中位数 0.5945，grid 最好 0.5719
- F100|UNIFORM: Test 最后 10 个 grid 点中位数 0.5718，grid 最好 0.5571
- F100|UNIFORM_OOF: Test 最后 10 个 grid 点中位数 0.6126，grid 最好 0.5849

## 5. 干预强度与结构追溯（内部干预，不是重训收益）

### fit100（SELECT 选中权重，update=6221）

| split | 分子数 | 预测改变分子比例 | max |Δp| | p50 |Δp| | p99 |Δp| |
|---|---|---|---|---|---|
| test | 420 | 0.9024 | 0.546623 | 0.025164 | 0.266792 |
| valid | 420 | 0.9262 | 0.282746 | 0.022426 | 0.202267 |

- init 权重（按分子先聚合；`a_active` 是该分子**全部根**的 a 均值，inactive 根计为 1）：‖J_full−J_marg‖ 中位 4.9459，‖(1−a)(J_full−J_marg)‖ 中位 1.6032，比值中位 0.0420，a_active 中位 0.9714

- selected 权重（按分子先聚合；`a_active` 是该分子**全部根**的 a 均值，inactive 根计为 1）：‖J_full−J_marg‖ 中位 9.1025，‖(1−a)(J_full−J_marg)‖ 中位 2.6167，比值中位 0.0681，a_active 中位 0.9714

- final 权重（按分子先聚合；`a_active` 是该分子**全部根**的 a 均值，inactive 根计为 1）：‖J_full−J_marg‖ 中位 9.1103，‖(1−a)(J_full−J_marg)‖ 中位 2.6169，比值中位 0.0685，a_active 中位 0.9714

- 五对输入别名（上游已存在）：5 对，全部 |Δp| ≤ 2e-5 = True
### fit25（SELECT 选中权重，update=5638）

| split | 分子数 | 预测改变分子比例 | max |Δp| | p50 |Δp| | p99 |Δp| |
|---|---|---|---|---|---|
| test | 420 | 0.9024 | 0.261789 | 0.015466 | 0.157486 |
| valid | 420 | 0.9262 | 0.856240 | 0.013046 | 0.251529 |

- init 权重（按分子先聚合；`a_active` 是该分子**全部根**的 a 均值，inactive 根计为 1）：‖J_full−J_marg‖ 中位 6.2772，‖(1−a)(J_full−J_marg)‖ 中位 1.7320，比值中位 0.0472，a_active 中位 0.9640

- selected 权重（按分子先聚合；`a_active` 是该分子**全部根**的 a 均值，inactive 根计为 1）：‖J_full−J_marg‖ 中位 9.8710，‖(1−a)(J_full−J_marg)‖ 中位 3.0125，比值中位 0.0726，a_active 中位 0.9640

- final 权重（按分子先聚合；`a_active` 是该分子**全部根**的 a 均值，inactive 根计为 1）：‖J_full−J_marg‖ 中位 9.8508，‖(1−a)(J_full−J_marg)‖ 中位 3.0202，比值中位 0.0733，a_active 中位 0.9640

- 五对输入别名（上游已存在）：5 对，全部 |Δp| ≤ 2e-5 = True

## 6. 结论与边界

- 判定：**SPLIT_OR_SELECTION_SENSITIVE**
- 即使出现正向信号，也只支持**这一**结构代理、这一算子、这一配方、这一 split 与单 seed 下的学习偏置；不证明校准置信度、普遍稀有结构因果、VC 复杂度下降或化学因果。
- `a` 改变了学习函数，不能仅因参数量相同（228 674）声称完整函数族表达力相同；五对别名预测不变也不等于全体表达边界未变。
- 机制依赖（门控是否被使用）与预测收益分别表述：§5 是机制/干预证据，§2 是性能证据。
- 只比较两个事先定义的均匀常数（0.70 与 `a_oof`），不声称击败所有可能的均匀收缩强度。

## 7. 成本、执行 regime 与事故记录

- 完成作业 10 条（4 并发，实际墙钟约 1.1 h）；进程墙钟合计 3.86 h（各 run 之和，不除以并发）；GPU·h 合计 0.000 A100·h（本轮**未占卡**，见下）。
- 手数上限：墙钟目标 3.0 h / 硬 4.0 h；GPU 软 2.0 / 硬 3.0 A100·h；正式训练上限 15 条。
- **基础设施事故**：`res` 的共享 `/home`（44 TB）在本轮期间 100% 写满（`df -k` 可用 0 KB），`rr deploy res` 在上传 git bundle 时失败（`No space left on device`），因此 `res-gpu0/res-gpu1` 无法使用；本地与 `res` 之间也没有可用的写路径。为完成已授权的科学清单，全部正式训练改在**同一**本地 CPU regime 执行（torch 2.5.1+cpu、无 CUDA、单机 16 核）。这是执行 regime 的偏离，已在 `analysis/source_manifest.json` 与`analysis/run_manifest.json` 逐 run 记录；同一 panel 的五臂共享同一 regime，因此内部可比。跨 regime（`res` A100 与本地 CPU）数字不可互换。
- 冒烟：7 个短作业（各 ≤8 updates），参数量一致 = True，共享初始 state = {'BACE': True, 'Lipophilicity': True}。

## 8. 产物

- `PROTOCOL.md`
- `configs/experiment.yaml`
- `code/`
- `tests/`
- `analysis/preflight.json`
- `analysis/key_selection.json`
- `analysis/freeze.json`
- `analysis/source_manifest.json`
- `analysis/run_manifest.json`
- `analysis/comparisons.json`
- `analysis/intervention.json`
- `analysis/support_groups.json`
- `analysis/influence.json`
- `analysis/budget.json`
- `analysis/main_table.csv`
- `analysis/figures/performance_two_N.png`
- `analysis/figures/support_feasibility.png`
- `analysis/figures/support_groups.png`

