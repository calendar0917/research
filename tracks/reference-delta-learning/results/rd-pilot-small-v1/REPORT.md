# Reference-delta learning pilot — rd-pilot-small-v1

官方 OGB `ogbg-molesol` / `ogbg-mollipo`，官方 Train/Validation/Test 划分，单训练 seed=0，`test_policy: always`（Test 作开发评估集，不参与任何选择）。本文件由 `stage=aggregate` 依据两个 worker 的产物生成；评价修订为 `rd-evalfix-v1`。

> **评价纠错（rd-evalfix-v1）。** 原 C 评价在 Val/Test 上把当前 split 的 query 局部行号误当成 Train 行号读取特征；参照与标签却是真正的 Val/Test。因此旧 C 的 Val/Test 曲线、选中状态、预测、比较、案例与路线判定无效，已在 **不重训任何 booster** 的前提下重算。旧报告中的“C 只能选第 1 轮”“越训练越不泛化”“NO_JOINT_INCREMENT”等表述已撤回。根因、撤回范围与模型 SHA 见 `CORRIGENDUM_evalfix_v1.md`。

## 问题与四条臂

在**静态分子输入完全相同**（同一 2059 维 ECFP4+描述符矩阵、同一官方划分、同一 Train 目标标准化）的前提下：

**联合学习两个分子的性质差**（推理时用 Train 参照还原成绝对值），是否优于直接绝对回归、Train 邻居标签直接预测、以及“直接模型 + 邻居残差修正”？

| 臂 | 定义 | 训练 |
| --- | --- | --- |
| **D** | `D_t(G) = mu + sigma * f_t(x_G)` | 800 轮 |
| **N** | `N(G) = sum_j w_j y_j` | 无 |
| **C** | `delta_t(G,R) = sigma/2 [h_t(z(G,R)) - h_t(z(R,G))]`，`C_t(G) = sum_j w_j (y_j + delta_t(G,R_j))` | 1600 轮 |
| **R** | `R_t(G) = D_t(G) + sum_j w_j (y_j - D_t(R_j))` | 复用 D |

`z(i,j) = [x_i, x_j, x_i - x_j]`（6177 维）；参照一律来自 Train，按 Tanimoto 取 Top-64，排除自身与完全相同 SMILES 的行；`w_j` 由冻结的 6 个 (K, 权重) 配置之一给出（`K ∈ {4,16,64}`，uniform 或 softmax τ=0.1）。C 的反对称化是**推理定义**的一部分（`delta(G,R) = -delta(R,G)`），不是事后修补；`delta` 是原始单位差值，乘 `sigma` 之后**不再加 `mu`**。

判定（投入门槛，不是统计显著性界限）：C 需在 **Val 与 Test 同时** 比 D、N、R 各好 ≥3% 才算 `strong_joint_task`；另一任务不得差于任何一条臂超过 5%；`SUBSTANTIVE_PROGRESS` 要求两个 split 都满足 `c <= 1.05 b` 或关闭一半差距；`COMPETITIVE` 要求 C 的 Val 与 Test 都不超过最优可核验参照 B 的 105%。

## 主表（每任务每臂：Val 选中状态，原始单位）

| dataset | arm | preset | round | used_rounds | trained_rounds | config | train_rmse | valid_rmse | test_rmse | train_mae | valid_mae | test_mae |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ogbg-molesol | D | H1 | 798 | 798 | 800 | none | 0.3700 | 0.9270 | 0.8159 | 0.2842 | 0.7146 | 0.6539 |
| ogbg-molesol | N | constant | 0 | 0 | 0 | K4_softmax_tau_0.1 | 1.1797 | 1.5293 | 1.5846 | 0.8862 | 1.2576 | 1.2464 |
| ogbg-molesol | R | H1 | 669 | 669 | 800 | K4_softmax_tau_0.1 | 0.4438 | 0.8814 | 0.8282 | 0.3359 | 0.6912 | 0.6624 |
| ogbg-molesol | C | H2 | 1577 | 1577 | 1600 | K4_softmax_tau_0.1 | 0.4298 | 0.8775 | 0.8906 | 0.3173 | 0.6575 | 0.7015 |
| ogbg-molesol | G | reused | - | - | - | none | 0.3299 | 0.9353 | 0.9354 | 0.2371 | 0.7095 | 0.7288 |
| ogbg-molesol | L | reused | - | - | - | none | 0.3462 | 0.8686 | 0.8570 | 0.2562 | 0.6533 | 0.6686 |
| ogbg-mollipo | D | H2 | 800 | 800 | 800 | none | 0.4728 | 0.7747 | 0.7628 | 0.3579 | 0.6004 | 0.5905 |
| ogbg-mollipo | N | constant | 0 | 0 | 0 | K16_softmax_tau_0.1 | 0.8555 | 0.9970 | 0.9178 | 0.6278 | 0.7621 | 0.7141 |
| ogbg-mollipo | R | H1 | 790 | 790 | 800 | K16_softmax_tau_0.1 | 0.5247 | 0.7448 | 0.7135 | 0.3848 | 0.5641 | 0.5358 |
| ogbg-mollipo | C | H2 | 1595 | 1595 | 1600 | K16_softmax_tau_0.1 | 0.5050 | 0.7130 | 0.6879 | 0.3669 | 0.5414 | 0.5110 |
| ogbg-mollipo | G | reused | - | - | - | none | 0.2691 | 0.6091 | 0.6456 | 0.2054 | 0.4624 | 0.4847 |
| ogbg-mollipo | L | reused | - | - | - | none | 0.2631 | 0.7052 | 0.7139 | 0.1943 | 0.5121 | 0.5252 |

D/N/R/C 均为本轮产物：D、C 每个 preset 一条正式轨迹，N 为常量（无训练），R 复用 D 的树前缀并重新选择检索配置。G/L 为 44300426 的复用点参照，行级 row_id/y 已核验并复算 RMSE；未重训。

## 曲线摘要（同一配方/配置：选中轮 / 最后一轮 / 最后 20 轮均值）

| dataset | arm | rounds | valid_selected | valid_final | valid_mean_last20 | test_selected | test_final | test_mean_last20 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ogbg-molesol | D | 800 | 0.9270 | 0.9271 | 0.9274 | 0.8159 | 0.8163 | 0.8154 |
| ogbg-molesol | N | 0 | 1.5293 | 1.5293 | 1.5293 | 1.5846 | 1.5846 | 1.5846 |
| ogbg-molesol | R | 800 | 0.8814 | 0.8824 | 0.8829 | 0.8282 | 0.8373 | 0.8366 |
| ogbg-molesol | C | 1600 | 0.8775 | 0.8779 | 0.8778 | 0.8906 | 0.8906 | 0.8905 |
| ogbg-mollipo | D | 800 | 0.7747 | 0.7747 | 0.7748 | 0.7628 | 0.7628 | 0.7630 |
| ogbg-mollipo | N | 0 | 0.9970 | 0.9970 | 0.9970 | 0.9178 | 0.9178 | 0.9178 |
| ogbg-mollipo | R | 800 | 0.7448 | 0.7451 | 0.7450 | 0.7135 | 0.7131 | 0.7134 |
| ogbg-mollipo | C | 1600 | 0.7130 | 0.7130 | 0.7130 | 0.6879 | 0.6880 | 0.6881 |

最后一轮与最后 20 轮均值只作曲线摘要报告，从不替代 Val 选中的状态；N 为常量预测，三者相同。选中轮早于最后一轮时，二者的差就是该配方在本任务上的过拟合/漂移幅度。

## C 相对 D / N / R 的差距与配对不确定性

| dataset | comparison | split | rmse_c | rmse_b | delta_rmse | relative_delta | ci95_low | ci95_high | drop_one_delta | drop_one_sign_flip |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ogbg-molesol | C-D | valid | 0.8775 | 0.9270 | -0.0495 | -0.0534 | -0.1469 | 0.0436 | -0.0794 | no |
| ogbg-molesol | C-N | valid | 0.8775 | 1.5293 | -0.6518 | -0.4262 | -0.8654 | -0.4346 | -0.6065 | no |
| ogbg-molesol | C-R | valid | 0.8775 | 0.8814 | -0.0039 | -0.0044 | -0.0769 | 0.0668 | -0.0306 | no |
| ogbg-molesol | C-D | test | 0.8906 | 0.8159 | 0.0747 | 0.0915 | 0.0043 | 0.1456 | 0.0801 | no |
| ogbg-molesol | C-N | test | 0.8906 | 1.5846 | -0.6940 | -0.4380 | -0.9176 | -0.4733 | -0.6478 | no |
| ogbg-molesol | C-R | test | 0.8906 | 0.8282 | 0.0624 | 0.0753 | 0.0017 | 0.1217 | 0.0564 | no |
| ogbg-mollipo | C-D | valid | 0.7130 | 0.7747 | -0.0617 | -0.0796 | -0.0917 | -0.0299 | -0.0634 | no |
| ogbg-mollipo | C-N | valid | 0.7130 | 0.9970 | -0.2841 | -0.2849 | -0.3559 | -0.2184 | -0.2819 | no |
| ogbg-mollipo | C-R | valid | 0.7130 | 0.7448 | -0.0318 | -0.0427 | -0.0510 | -0.0127 | -0.0329 | no |
| ogbg-mollipo | C-D | test | 0.6879 | 0.7628 | -0.0749 | -0.0982 | -0.1036 | -0.0464 | -0.0768 | no |
| ogbg-mollipo | C-N | test | 0.6879 | 0.9178 | -0.2299 | -0.2504 | -0.2959 | -0.1660 | -0.2308 | no |
| ogbg-mollipo | C-R | test | 0.6879 | 0.7135 | -0.0256 | -0.0358 | -0.0446 | -0.0062 | -0.0260 | no |

bootstrap 按 query 分子重采样 2000 次（RNG 1703，固定模型与训练参照，pair 不作为独立样本）；`drop_one_*` 为移除该比较中参照模型误差最大的同一个 query 后重算的 ΔRMSE。区间只描述固定模型下的重采样不确定性。

## 冻结判定

| dataset | arm_C_val | arm_C_test | best_reference | best_valid | best_test | competitive | strong_joint_task | not_regressing | val_progress | test_progress | substantive |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ogbg-molesol | 0.8775 | 0.8906 | L | 0.8686 | 0.8570 | yes | no | no | yes | yes | yes |
| ogbg-mollipo | 0.7130 | 0.6879 | G | 0.6091 | 0.6456 | no | yes | yes | no | no | no |

**ogbg-molesol**

- 相对增益 gain_b = 1 - RMSE_C/RMSE_b：D_test=-0.0915, D_valid=0.0534, N_test=0.4380, N_valid=0.4262, R_test=-0.0753, R_valid=0.0044
- strong_joint_task=no（六条件：D_test=no, D_valid=yes, N_test=yes, N_valid=yes, R_test=no, R_valid=no）
- 另一任务 5% 条件：D_test=no, D_valid=yes, N_test=yes, N_valid=yes, R_test=no, R_valid=yes
- valid: J=0.8814 b=0.8686 c=0.8775 half_gap_closed=no competitive_5pct=yes progress_pass=yes
- test: J=0.8159 b=0.8570 c=0.8906 half_gap_closed=not_applicable competitive_5pct=yes progress_pass=yes

**ogbg-mollipo**

- 相对增益 gain_b = 1 - RMSE_C/RMSE_b：D_test=0.0982, D_valid=0.0796, N_test=0.2504, N_valid=0.2849, R_test=0.0358, R_valid=0.0427
- strong_joint_task=yes（六条件：D_test=yes, D_valid=yes, N_test=yes, N_valid=yes, R_test=yes, R_valid=yes）
- 另一任务 5% 条件：D_test=yes, D_valid=yes, N_test=yes, N_valid=yes, R_test=yes, R_valid=yes
- valid: J=0.7448 b=0.6091 c=0.7130 half_gap_closed=no competitive_5pct=no progress_pass=no
- test: J=0.7135 b=0.6456 c=0.6879 half_gap_closed=no competitive_5pct=no progress_pass=no

- JOINT_DELTA_SIGNAL: **CROSS_TASK_TRADEOFF**
- SUBSTANTIVE_PROGRESS: **no**
- EXECUTION_PASS: **yes**
- WORTH_STRUCTURAL_COMPARATOR_FOLLOWUP: **no**

## C 的拟合层级（pair / 分子 Train / Val / Test）

| dataset | config | pair_rmse_std_selected | pair_rmse_std_antisym_selected | mean_last20_pair_rmse_std | train_rmse_molecule | valid_rmse_molecule | test_rmse_molecule |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ogbg-molesol | K4_softmax_tau_0.1 | 0.2550 | 0.2546 | 0.2544 | 0.4298 | 0.8775 | 0.8906 |
| ogbg-mollipo | K16_softmax_tau_0.1 | 0.6019 | 0.6013 | 0.6024 | 0.5050 | 0.7130 | 0.6879 |

## C 的曲线与拟合层级

**ogbg-molesol**（C-H2，配置 `K4_softmax_tau_0.1`，选中轮 = 1577 / 共 1600 轮）

- 训练 pair 加权 RMSE（标准化目标）：第 1 轮 1.2842 → 第 1600 轮 0.2539
- 分子 Train RMSE：第 1 轮 1.1543 → 第 1600 轮 0.4286（下降；该 split 全曲线最好 0.4286，第 1600 轮）
- Val RMSE：第 1 轮 1.4998 → 第 1600 轮 0.8779（下降；该 split 全曲线最好 0.8775，第 1577 轮）
- Test RMSE：第 1 轮 1.5522 → 第 1600 轮 0.8906（下降；该 split 全曲线最好 0.8890，第 1311 轮）
- 对照：N 的 Val/Test = 1.5293/1.5846，D 的 Val/Test = 0.9270/0.8159
- 与自身基线 N 相比：Val 上 C 在 1600 轮中有 1600 轮优于 N（N = 1.5293），Test 上 1600/1600 轮（N = 1.5846）
- 读法：分子 Train 曲线下降，Val 下降，Test 下降；Val 全曲线最小 RMSE 在第 1577 轮（0.8775），Val-selected 状态是第 1577 轮（Val/Test 0.8775/0.8906）。这是固定配方下的曲线描述；选中仍只由完整 Val RMSE 决定。

**ogbg-mollipo**（C-H2，配置 `K16_softmax_tau_0.1`，选中轮 = 1595 / 共 1600 轮）

- 训练 pair 加权 RMSE（标准化目标）：第 1 轮 1.3233 → 第 1600 轮 0.6015
- 分子 Train RMSE：第 1 轮 0.8476 → 第 1600 轮 0.5048（下降；该 split 全曲线最好 0.5048，第 1600 轮）
- Val RMSE：第 1 轮 0.9883 → 第 1600 轮 0.7130（下降；该 split 全曲线最好 0.7130，第 1595 轮）
- Test RMSE：第 1 轮 0.9099 → 第 1600 轮 0.6880（下降；该 split 全曲线最好 0.6879，第 1595 轮）
- 对照：N 的 Val/Test = 0.9970/0.9178，D 的 Val/Test = 0.7747/0.7628
- 与自身基线 N 相比：Val 上 C 在 1600 轮中有 1600 轮优于 N（N = 0.9970），Test 上 1600/1600 轮（N = 0.9178）
- 读法：分子 Train 曲线下降，Val 下降，Test 下降；Val 全曲线最小 RMSE 在第 1595 轮（0.7130），Val-selected 状态是第 1595 轮（Val/Test 0.7130/0.6879）。这是固定配方下的曲线描述；选中仍只由完整 Val RMSE 决定。

### 同一参照还原机制的对照：R vs D

| dataset | D_valid | R_valid | gain_valid | D_test | R_test | gain_test |
| --- | --- | --- | --- | --- | --- | --- |
| ogbg-molesol | 0.9270 | 0.8814 | 0.0492 | 0.8159 | 0.8282 | -0.0151 |
| ogbg-mollipo | 0.7747 | 0.7448 | 0.0386 | 0.7628 | 0.7135 | 0.0647 |

R 与 C 共用同一套 Tanimoto 参照、同一批权重配置和同一还原公式；差别只在 R 的锚点是已训练的绝对模型 D，而 C 的差值 h 是从 Train-Train 对里从零学的。R 相对 D 的增益：ogbg-molesol Val +4.9% / Test -1.5%；ogbg-mollipo Val +3.9% / Test +6.5%。Test 上 R 低于 D 的任务：ogbg-molesol。该对照是描述性的，不参与冻结判定。

## 参照覆盖（最高合法参照 Tanimoto 的固定分箱）

| dataset | split | bin_low | bin_high | n_queries | D_rmse | N_rmse | R_rmse | C_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ogbg-molesol | train | 0.0000 | 0.2000 | 7 | 0.4258 | 1.8607 | 0.5760 | 0.5903 |
| ogbg-molesol | train | 0.2000 | 0.4000 | 101 | 0.4287 | 1.4547 | 0.5150 | 0.4809 |
| ogbg-molesol | train | 0.4000 | 0.6000 | 350 | 0.3794 | 1.1873 | 0.4306 | 0.4023 |
| ogbg-molesol | train | 0.6000 | 0.8000 | 317 | 0.3513 | 0.9877 | 0.4171 | 0.4216 |
| ogbg-molesol | train | 0.8000 | 1.0000 | 127 | 0.3334 | 1.3001 | 0.4736 | 0.4679 |
| ogbg-molesol | valid | 0.0000 | 0.2000 | 11 | 0.8681 | 1.4160 | 0.8252 | 0.9000 |
| ogbg-molesol | valid | 0.2000 | 0.4000 | 51 | 0.9353 | 1.6020 | 0.9261 | 0.8984 |
| ogbg-molesol | valid | 0.4000 | 0.6000 | 42 | 0.9421 | 1.5222 | 0.8534 | 0.8712 |
| ogbg-molesol | valid | 0.6000 | 0.8000 | 7 | 0.7486 | 1.0719 | 0.8156 | 0.8052 |
| ogbg-molesol | valid | 0.8000 | 1.0000 | 2 | 1.2255 | 1.7248 | 0.8101 | 0.5106 |
| ogbg-molesol | test | 0.0000 | 0.2000 | 12 | 0.9315 | 1.8732 | 0.9844 | 0.9559 |
| ogbg-molesol | test | 0.2000 | 0.4000 | 56 | 0.8885 | 1.3848 | 0.8846 | 0.9063 |
| ogbg-molesol | test | 0.4000 | 0.6000 | 36 | 0.6955 | 1.8131 | 0.6857 | 0.8413 |
| ogbg-molesol | test | 0.6000 | 0.8000 | 7 | 0.6602 | 1.0722 | 0.7740 | 0.8834 |
| ogbg-molesol | test | 0.8000 | 1.0000 | 2 | 0.2616 | 1.9823 | 0.6853 | 0.9258 |
| ogbg-mollipo | train | 0.0000 | 0.2000 | 5 | 0.6970 | 1.8574 | 0.8666 | 1.0357 |
| ogbg-mollipo | train | 0.2000 | 0.4000 | 459 | 0.6148 | 1.2695 | 0.7125 | 0.6948 |
| ogbg-mollipo | train | 0.4000 | 0.6000 | 539 | 0.5678 | 1.0213 | 0.6081 | 0.5776 |
| ogbg-mollipo | train | 0.6000 | 0.8000 | 1351 | 0.4392 | 0.7495 | 0.4913 | 0.4753 |
| ogbg-mollipo | train | 0.8000 | 1.0000 | 1006 | 0.3740 | 0.6152 | 0.4024 | 0.3794 |
| ogbg-mollipo | valid | 0.0000 | 0.2000 | 2 | 1.3475 | 0.4193 | 1.4408 | 1.1459 |
| ogbg-mollipo | valid | 0.2000 | 0.4000 | 141 | 0.9230 | 1.2124 | 0.9194 | 0.9024 |
| ogbg-mollipo | valid | 0.4000 | 0.6000 | 104 | 0.8258 | 1.0755 | 0.7486 | 0.7007 |
| ogbg-mollipo | valid | 0.6000 | 0.8000 | 121 | 0.5880 | 0.7326 | 0.5623 | 0.5217 |
| ogbg-mollipo | valid | 0.8000 | 1.0000 | 52 | 0.5461 | 0.6888 | 0.5017 | 0.4813 |
| ogbg-mollipo | test | 0.0000 | 0.2000 | 3 | 1.2801 | 0.8841 | 1.2211 | 0.8870 |
| ogbg-mollipo | test | 0.2000 | 0.4000 | 136 | 0.9233 | 1.0794 | 0.9087 | 0.8845 |
| ogbg-mollipo | test | 0.4000 | 0.6000 | 104 | 0.6823 | 0.9694 | 0.6125 | 0.5899 |
| ogbg-mollipo | test | 0.6000 | 0.8000 | 126 | 0.6601 | 0.7589 | 0.5867 | 0.5564 |
| ogbg-mollipo | test | 0.8000 | 1.0000 | 51 | 0.6297 | 0.6672 | 0.5362 | 0.5391 |

最高相似度分位数：{"ogbg-molesol": {"test": {"max": 1.0, "mean": 0.380290171873253, "median": 0.36734694242477417, "min": 0.14516128599643707, "n": 113, "q25": 0.28333333134651184, "q75": 0.4615384638309479}, "train": {"max": 1.0, "mean": 0.6052579647818056, "median": 0.5882353186607361, "min": 0.0, "n": 902, "q25": 0.5, "q75": 0.7142857313156128}, "valid": {"max": 0.9090909361839294, "mean": 0.3888052682169771, "median": 0.3888888955116272, "min": 0.14814814925193787, "n": 113, "q25": 0.27272728085517883, "q75": 0.460317462682724}}, "ogbg-mollipo": {"test": {"max": 1.0, "mean": 0.5408604620468049, "median": 0.52452152967453, "min": 0.14754098653793335, "n": 420, "q25": 0.3524251878261566, "q75": 0.7124697417020798}, "train": {"max": 1.0, "mean": 0.6740983620773824, "median": 0.7321428656578064, "min": 0.17073170840740204, "n": 3360, "q25": 0.5381054282188416, "q75": 0.8142856955528259}, "valid": {"max": 1.0, "mean": 0.5421870468627839, "median": 0.5225576460361481, "min": 0.1603773534297943, "n": 420, "q25": 0.35384616255760193, "q75": 0.7092802822589874}}}
空箱保留；阈值未搜索。覆盖子群上的局部优势不能替代完整任务结论。

## 参照计算账本（每任务 3 个 Val query，仅计算分解）

### ogbg-molesol · median_abs_error_C · row_id=318 · SMILES=Nc1ncnc2nc[nH]c12

y=-2.120000  pred_D=-1.625685  pred_C=-1.585709  |err_C|=0.534291  config=K4_softmax_tau_0.1  K_eff=4

| slot | ref_row_id | ref_smiles | similarity | weight | y_ref_raw | delta_raw | y_ref_plus_delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 26 | Cn1c(=O)n(C)c2nc[nH]c2c1=O | 0.2973 | 0.4029 | -1.3900 | -0.4457 | -1.8357 |
| 1 | 81 | Cc1ncnc2nccnc12 | 0.2353 | 0.2167 | -0.4660 | -0.5334 | -0.9994 |
| 2 | 585 | Nc1c(C)c[nH]c(=O)n1 | 0.2222 | 0.1902 | -1.4580 | -0.4236 | -1.8816 |
| 3 | 927 | Nc1nc(=O)[nH]cc1F | 0.2222 | 0.1902 | -0.9720 | -0.4564 | -1.4284 |

其余参照加权和 = 0.000000000；Top5 加权和 = -1.585708663；合计 = -1.585708663（与 pred_C 差 7.66e-09）

### ogbg-molesol · largest_improvement_vs_D · row_id=1025 · SMILES=OCC2OC(OC1(CO)OC(CO)C(O)C1O)C(O)C(O)C2O

y=0.790000  pred_D=-1.221598  pred_C=0.586777  |err_C|=0.203223  config=K4_softmax_tau_0.1  K_eff=4

| slot | ref_row_id | ref_smiles | similarity | weight | y_ref_raw | delta_raw | y_ref_plus_delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 875 | OCC1OC(O)(CO)C(O)C1O | 0.4211 | 0.5182 | 0.6400 | -0.0802 | 0.5598 |
| 1 | 341 | OCC1OC(O)C(O)C(O)C1O | 0.4000 | 0.4198 | 0.7400 | -0.0883 | 0.6517 |
| 2 | 1026 | OC1C(O)C(O)C(O)C(O)C1O | 0.1562 | 0.0367 | 0.3500 | 0.0300 | 0.3800 |
| 3 | 102 | C1OC(O)C(O)C(O)C1O | 0.1190 | 0.0253 | 0.3900 | -0.0273 | 0.3627 |

其余参照加权和 = 0.000000000；Top5 加权和 = 0.586776562；合计 = 0.586776562（与 pred_C 差 8.20e-10）

### ogbg-molesol · largest_regression_vs_D · row_id=1116 · SMILES=COP(=S)(OC)Oc1ccc(Sc2ccc(OP(=S)(OC)OC)cc2)cc1

y=-6.237000  pred_D=-6.483160  pred_C=-7.378256  |err_C|=1.141256  config=K4_softmax_tau_0.1  K_eff=4

| slot | ref_row_id | ref_smiles | similarity | weight | y_ref_raw | delta_raw | y_ref_plus_delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 229 | COP(=S)(OC)Oc1ccc(SC)c(C)c1 | 0.5000 | 0.5071 | -4.5700 | -3.0344 | -7.6044 |
| 1 | 274 | COP(=S)(OC)Oc1ccc(N(=O)=O)c(C)c1 | 0.4250 | 0.2395 | -4.0400 | -3.4020 | -7.4420 |
| 2 | 646 | COP(=S)(OC)Oc1cc(Cl)c(Cl)cc1Cl | 0.3714 | 0.1402 | -5.7200 | -1.6982 | -7.4182 |
| 3 | 748 | CCOP(=S)(OCC)Oc1ccc(cc1)S(C)=O | 0.3500 | 0.1132 | -2.3000 | -3.8802 | -6.1802 |

其余参照加权和 = 0.000000000；Top5 加权和 = -7.378255844；合计 = -7.378255844（与 pred_C 差 2.27e-08）

### ogbg-mollipo · median_abs_error_C · row_id=1266 · SMILES=C[C@@H]1CN(CCN1C(=O)[C@@H]2CCCC[C@H]2C(=O)NC3(CC3)C#N)c4ccc5c(C)nn(C)c5c4

y=2.700000  pred_D=2.271331  pred_C=2.275799  |err_C|=0.424201  config=K16_softmax_tau_0.1  K_eff=16

| slot | ref_row_id | ref_smiles | similarity | weight | y_ref_raw | delta_raw | y_ref_plus_delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2150 | COc1ccc(cc1OC)N2CCN([C@H](C)C2)C(=O)[C@@H]3CCCC[C@H]3C(=O)NC4(CC4)C#N | 0.6757 | 0.2751 | 1.8500 | 0.1519 | 2.0019 |
| 1 | 2389 | C[C@@H]1CN(CCN1C(=O)[C@@H]2CCCC[C@H]2C(=O)NC3(CC3)C#N)c4ccccc4 | 0.6714 | 0.2637 | 2.3500 | 0.0120 | 2.3620 |
| 2 | 2877 | COc1ccc(nc1OC)N2CCN([C@H](C)C2)C(=O)[C@@H]3CCCC[C@H]3C(=O)NC4(CC4)C#N | 0.5679 | 0.0936 | 2.3000 | 0.3592 | 2.6592 |
| 3 | 285 | COc1ccc(cc1OC)N2CCN(CC2)C(=O)[C@@H]3CCCC[C@H]3C(=O)NC4(CC4)C#N | 0.5128 | 0.0540 | 1.4000 | 0.5859 | 1.9859 |
| 4 | 2443 | COc1ccc(cc1)N2CCN(CC2)C(=O)[C@@H]3CCCC[C@H]3C(=O)NC4(CC4)C#N | 0.4872 | 0.0418 | 1.7400 | 0.4952 | 2.2352 |

其余参照加权和 = 0.652787066；Top5 加权和 = 1.623012153；合计 = 2.275799219（与 pred_C 差 9.15e-10）

### ogbg-mollipo · largest_improvement_vs_D · row_id=1762 · SMILES=CS(=O)(=O)c1ccc(cc1)[C@H](CC2CCCC2)C(=O)Nc3nccs3

y=3.880000  pred_D=2.915886  pred_C=3.788362  |err_C|=0.091639  config=K16_softmax_tau_0.1  K_eff=16

| slot | ref_row_id | ref_smiles | similarity | weight | y_ref_raw | delta_raw | y_ref_plus_delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2600 | CS(=O)(=O)c1ccc(cc1)[C@@H](CC2CCCC2)C(=O)Nc3nccs3 | 1.0000 | 0.9298 | 3.8600 | 0.0000 | 3.8600 |
| 1 | 2302 | O=C(Nc1nccs1)[C@H](CC2CCOCC2)c3ccc(cc3)S(=O)(=O)C4CC4 | 0.6833 | 0.0392 | 2.5500 | 0.1723 | 2.7223 |
| 2 | 972 | Fc1cnc(NC(=O)[C@H](CC2CCOCC2)c3ccc(cc3)S(=O)(=O)C4CC4)s1 | 0.4714 | 0.0047 | 3.1900 | -0.0373 | 3.1527 |
| 3 | 2396 | Fc1cnc(NC(=O)C(CC2CCOCC2)c3ccc(cc3)S(=O)(=O)C4CC4)s1 | 0.4714 | 0.0047 | 3.3000 | -0.0373 | 3.2627 |
| 4 | 3462 | CS(=O)(=O)c1ccc(cc1Cl)[C@@H](CC2CCCC2)C(=O)Nc3cnc(cn3)[C@H](O)CO | 0.4533 | 0.0039 | 2.5800 | 0.3434 | 2.9234 |

其余参照加权和 = 0.050979772；Top5 加权和 = 3.737381825；合计 = 3.788361597（与 pred_C 差 6.15e-11）

### ogbg-mollipo · largest_regression_vs_D · row_id=1036 · SMILES=CC1(C)Cc2c(c3ccccc3)c(c(CC(=O)O)n2C1)c4ccc(Cl)cc4

y=3.070000  pred_D=2.596468  pred_C=1.848096  |err_C|=1.221904  config=K16_softmax_tau_0.1  K_eff=16

| slot | ref_row_id | ref_smiles | similarity | weight | y_ref_raw | delta_raw | y_ref_plus_delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1228 | OC(=O)Cc1csc(n1)c2ccc(Cl)cc2 | 0.3393 | 0.1182 | -0.2200 | 1.2997 | 1.0797 |
| 1 | 1984 | Cc1c(Sc2ccc(Cl)cc2)c3cc(ccc3n1CC(=O)O)c4ccccc4 | 0.3088 | 0.0871 | 3.7000 | -1.5859 | 2.1141 |
| 2 | 1439 | CN(C1CCN(CCC(c2ccccc2)c3ccccc3)CC1)C(=O)Cc4ccc(Cl)cc4 | 0.2857 | 0.0692 | 3.5000 | -2.1388 | 1.3612 |
| 3 | 4059 | CN(C)C(=O)C(CCN1CCC(O)(CC1)c2ccc(Cl)cc2)(c3ccccc3)c4ccccc4 | 0.2857 | 0.0692 | 3.4700 | -0.8811 | 2.5889 |
| 4 | 3913 | CC(C)N1CCC(CC1)N(C(=O)Cc2ccccc2)c3ccc(Cl)cc3 | 0.2794 | 0.0649 | 2.3000 | -1.0236 | 1.2764 |

其余参照加权和 = 1.180283783；Top5 加权和 = 0.667811851；合计 = 1.848095634（与 pred_C 差 4.69e-09）

## 单分子推理成本（描述性）

| dataset | arm | device | references_per_query | us_per_molecule | molecules_timed | config |
| --- | --- | --- | --- | --- | --- | --- |
| ogbg-molesol | C | cpu | 4 | 916.9540 | 113 | K4_softmax_tau_0.1 |
| ogbg-molesol | D | cpu | 0 | 152.6730 | 1128 | - |
| ogbg-molesol | N | cpu | 4 | 0.1760 | 1128 | K4_softmax_tau_0.1 |
| ogbg-molesol | R | cpu | 4 | 0.4480 | 1128 | K4_softmax_tau_0.1 |
| ogbg-mollipo | C | cpu | 16 | 1588.4030 | 420 | K16_softmax_tau_0.1 |
| ogbg-mollipo | D | cpu | 0 | 128.0560 | 4200 | - |
| ogbg-mollipo | N | cpu | 16 | 0.0910 | 4200 | K16_softmax_tau_0.1 |
| ogbg-mollipo | R | cpu | 16 | 0.2340 | 4200 | K16_softmax_tau_0.1 |

| dataset | design_matrix_seconds | retrieval_table_seconds | pair_list_seconds | pair_matrix_seconds | pair_matrix_rows |
| --- | --- | --- | --- | --- | --- |
| ogbg-molesol | 0.0102 | 0.0695 | 0.4824 | 0.0016 | 452 |
| ogbg-mollipo | 0.0553 | 1.3525 | 2.4256 | 0.1082 | 6720 |

定义：`us_per_molecule` = 用选中模型做一次预测的中位耗时 / 分子数（CPU，nthread=4，重载后的 booster，无 batching/调优）；`references_per_query` 是该臂选中配置的 `K_eff`。D 不需要参照；N 只做加权求和；R 行只计“取参照 + 加权残差”这一步（锚点模型 D 的前向已单列在 D 行，实际 R 推理 ≈ D + R 两行之和）；C 需要正反两向的 6177 维 pair 特征与两次树遍历，因此成本最高，且只在 Test split 上计时（pair 矩阵是最贵的部分），其余臂覆盖三个 split。一次性成本里，`retrieval_table_seconds` 覆盖三个 split 的 Top-64 全表，`pair_list_seconds` 是一次性冻结配对，`pair_matrix_seconds` 是 C 在 Test 上构建选中配置 pair 特征的时间（这些都不进入每轮评测）。

## 资源账本

| dataset | arm | preset | rounds | train_seconds | eval_seconds | matrix_build_seconds | seconds_per_round | gpu_peak_mb | trained_at | model_sha256 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ogbg-molesol | D | H1 | 800 | 2.2388 | 1.4098 | 0.0000 | 0.0028 | 528.0000 | 2026-11-16T14:04:54 | 719c3ab2fde6d7b3f6391aca5b2bdeca397cf3943b49110568e2975aad5b9024 |
| ogbg-molesol | D | H2 | 800 | 2.6820 | 1.3196 | 0.0000 | 0.0034 | 532.0000 | 2026-11-16T14:04:58 | 000df6502fb675f4605b0b2b6defc65fe569639e199dcb1607f37db941b792ec |
| ogbg-molesol | C | H1 | 1600 | 15.1288 | 16.6034 | 2.8841 | 0.0095 | 1134.0000 | 2026-11-16T14:05:47 | 9415143d5c8212a7ef18fe5d5611b5e9563a6149d00f032fbca81f53e96b06b8 |
| ogbg-molesol | C | H2 | 1600 | 16.5472 | 27.0891 | 2.2398 | 0.0103 | 1146.0000 | 2026-11-16T14:06:56 | 32e73107b8f5b268cf5c4f3f21751e7a2411261c9a37fc17a47691621b4122bd |
| ogbg-mollipo | D | H1 | 800 | 2.2088 | 2.7391 | 0.0000 | 0.0028 | 536.0000 | 2026-11-16T14:05:11 | 7532858029f2179e3ab18a16cd32d795eba71f52196cc6ee1405b32e776737d9 |
| ogbg-mollipo | D | H2 | 800 | 2.1399 | 3.0802 | 0.0000 | 0.0027 | 540.0000 | 2026-11-16T14:05:16 | 2e28a5bf8c9ed835da3841ab43960e38e3e8620a81d59a958bc1531b4be3fd6f |
| ogbg-mollipo | C | H1 | 1600 | 34.5486 | 62.6228 | 11.7230 | 0.0216 | 10670.0000 | 2026-11-16T14:08:21 | 9e75ae321be2c0bba316f5e56c3a72617bfd2a03ba7fcfed0e4c8125e01cf2e5 |
| ogbg-mollipo | C | H2 | 1600 | 38.2668 | 102.0933 | 10.8613 | 0.0239 | 10694.0000 | 2026-11-16T14:12:43 | 78c128dc67aae760b5490bdcbc6d7031ea9767421ce048fd87771305aae9ccd3 |

- 轨迹算力合计（训练 + 每轮三集评测 + 一次性 pair 矩阵构建）= 358.4 s = 0.100 GPU·h；两个 worker 各独占一张 A100，并行执行。
- 两个 worker 训练时最后一次执行的墙钟合计 = 49.7 s（每 GPU 一个训练 worker；本轮的启动失败与 resume 重跑另计，均记录在 `rr jobs`）。
- 本次 `rd-evalfix-v1` 新增评价墙钟（两任务合计）= 376.9 s；只加载既有 booster，CPU 评价，新增训练 0 条、新增 GPU 计算 = 0；按独占分配计约 0.105 GPU·h（GPU 空闲等待）。
- worker 进程 peak RSS：ogbg-molesol 2432 MB，ogbg-mollipo 6260 MB。GPU 峰值是 nvidia-smi 采样的整卡占用，只作上界参考。
- 预算：soft 3 GPU·h / hard 6 GPU·h（含 smoke 与重跑）；本轮轨迹算力远低于 soft 预算。
- 每轮评价成本已与一次性 pair 矩阵构建分开记录；本次修正后 C 的评测/训练时间比 = ogbg-molesol/H1 1.10x，ogbg-molesol/H2 1.64x，ogbg-mollipo/H1 1.81x，ogbg-mollipo/H2 2.67x。这是“每轮完整三集评测”的直接代价（K 越大、查询越多，代价越高）。

## 评价纠错与重跑范围（rd-evalfix-v1）

- 根因：`pair_matrix` / `pair_feature_blocks` 原实现固定读取 `task.x["train"]`。Train–Train 训练对因此正确，但 C 的 Val/Test 评价 query 也被映射到 Train 矩阵；参照表、权重和评价标签仍对应真正的 Val/Test query，取数身份不一致。
- 修复：两个构造函数改为 keyword-only 的显式 `query_x` / `reference_x` 两个输入域；训练对两边都传 Train，评价 query 传当前 split、参照传 Train。反向对仍由`BlockLayout(reverse=True)` 在正确构造的正向块上交换 X1/X2 并对 X3 取负。
- 撤回：旧 C 的 Val/Test 曲线、Val-selected preset/round/config、`pred_C.npz`、`pred_C_delta.npz`、C-D/C-N/C-R 比较、paired bootstrap、drop-one、参照覆盖中的 C 误差、三个 Val 案例账本，以及由旧 C 状态支撑的“C 退化/第 1 轮/NO_JOINT_INCREMENT”结论。
- 保留：八个 D/C booster、D/N/R 的选中状态与预测、G/L 点参照、参照表、训练 pair、样本权重和六种推理配置均未改变；本次新增训练 0 条。
- 重算：从两个既有 C booster 各提取全部 1600 棵树，逐前缀重算三 split × 六配置的分子级 RMSE/MAE；Train 分子曲线与训练 pair 曲线在相同状态下的修复前后差均为 0。

## 数据与模型身份

| dataset | graph_cache_sha256 | ecfp_sha256 | split_sizes | split_hash_valid | reference_hash | pair_hash | pair_records | mu | sigma |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ogbg-molesol | 3b7a8e16ca9c7ab240add84fbe26050dc81e22359bcc36f1c8a8111b77aec24a | 653fb0190abc4af43822ec643bacbcbdbfdd7555f5e632574c9f61728d574f2e | {'test': 113, 'train': 902, 'valid': 113} | dec602694cc57677 | a8dcf896959a7f650e1b216eac101ba40333573775d094df7c3049fa412c868f | cd49adf5df4fff78920747235199742c8f6a5faed3ba439a0e0a20b3f1f2c128 | 57728 | -2.8669 | 2.0667 |
| ogbg-mollipo | fe002287fa9c581278bc5430f5b4a5b26b35d9d4d5207d138f15956d93123c7c | 7990ce4c90a0fdb032897076e19221b62b1361fc98f36fea248cf097bb424ae4 | {'test': 420, 'train': 3360, 'valid': 420} | ebcbd43531f9eb25 | ebf968a2e2c235e2117271fcd29d9a9ae6a58d7d4d4bb387ebe4f11962fb2f07 | e4e3fafe7a928e3a719f5150a42286d349bf3074cfbac9694d64992d233bfeee | 215040 | 2.1622 | 1.2115 |

## 正确性检查（EXECUTION_PASS 的依据）

| dataset | max_abs_diff_D_raw | max_abs_margin_diff_C | max_abs_metric_diff | pass |
| --- | --- | --- | --- | --- |
| ogbg-molesol | 0.0000 | 0.0000 | 0.0000 | yes |
| ogbg-mollipo | 0.0000 | 0.0000 | 0.0000 | yes |

## 复用参照 G / L 的核验

| dataset | arm | available | point_only | train_rmse | valid_rmse | test_rmse | recompute_abs_diff_valid | recompute_abs_diff_test | source_commit | sha256 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ogbg-molesol | G | yes | yes | 0.3299 | 0.9353 | 0.9354 | 0.0000 | 0.0000 | 4430042683bfad26c02391c10d17a26932716812 | c829aa82facfbbdef20c0fc12577436cc8a70c5fc0fa083eada50de62850b0c3 |
| ogbg-molesol | L | yes | yes | 0.3462 | 0.8686 | 0.8570 | 0.0000 | 0.0000 | 4430042683bfad26c02391c10d17a26932716812 | 3d500494fd03bb35fd711a377d14ea9371bcf8251aa986bd0c58fe749c5bb88d |
| ogbg-mollipo | G | yes | yes | 0.2691 | 0.6091 | 0.6456 | 0.0000 | 0.0000 | 4430042683bfad26c02391c10d17a26932716812 | 4163b8c723f4a030ad88878a0ac74a2f1be3ca71ad2b6ea709385bb9bf0f2b46 |
| ogbg-mollipo | L | yes | yes | 0.2631 | 0.7052 | 0.7139 | 0.0000 | 0.0000 | 4430042683bfad26c02391c10d17a26932716812 | 9354c8a1bca7346fd8146d7525aa22a62b4050fee726fe0e10c0b0d4679493ee |

## 结论

- 执行有效性：EXECUTION_PASS = yes；参照轴：REFERENCE_COMPLETE。
- 联合差值信号：CROSS_TASK_TRADEOFF（strong_joint_tasks=['ogbg-mollipo']）。
- 实质进展：SUBSTANTIVE_PROGRESS = no。
- 是否值得下一步设计结构比较器：no。这只表示“值得进一步设计结构比较器”，不表示已获得结构解释、化学机制、组合泛化或第一梯队性能。

本轮的 3%/5%/50% 是投入门槛，不是统计显著性界限；负结果只针对本配方。
