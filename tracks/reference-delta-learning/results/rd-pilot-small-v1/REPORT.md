# Reference-delta learning pilot — rd-pilot-small-v1

官方 OGB `ogbg-molesol` / `ogbg-mollipo`，官方 Train/Validation/Test 划分，单训练 seed=0，`test_policy: always`（Test 作开发评估集，不参与任何选择）。本文件由 `stage=aggregate` 依据两个 GPU worker 的产物生成。

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
| ogbg-molesol | C | H2 | 1 | 1 | 1600 | K4_softmax_tau_0.1 | 1.1543 | 1.5297 | 1.5824 | 0.8671 | 1.2655 | 1.2478 |
| ogbg-molesol | G | reused | - | - | - | none | 0.3299 | 0.9353 | 0.9354 | 0.2371 | 0.7095 | 0.7288 |
| ogbg-molesol | L | reused | - | - | - | none | 0.3462 | 0.8686 | 0.8570 | 0.2562 | 0.6533 | 0.6686 |
| ogbg-mollipo | D | H2 | 800 | 800 | 800 | none | 0.4728 | 0.7747 | 0.7628 | 0.3579 | 0.6004 | 0.5905 |
| ogbg-mollipo | N | constant | 0 | 0 | 0 | K16_softmax_tau_0.1 | 0.8555 | 0.9970 | 0.9178 | 0.6278 | 0.7621 | 0.7141 |
| ogbg-mollipo | R | H1 | 790 | 790 | 800 | K16_softmax_tau_0.1 | 0.5247 | 0.7448 | 0.7135 | 0.3848 | 0.5641 | 0.5358 |
| ogbg-mollipo | C | H2 | 1 | 1 | 1600 | K16_softmax_tau_0.1 | 0.8476 | 0.9991 | 0.9181 | 0.6216 | 0.7631 | 0.7155 |
| ogbg-mollipo | G | reused | - | - | - | none | 0.2691 | 0.6091 | 0.6456 | 0.2054 | 0.4624 | 0.4847 |
| ogbg-mollipo | L | reused | - | - | - | none | 0.2631 | 0.7052 | 0.7139 | 0.1943 | 0.5121 | 0.5252 |

D/N/R/C 均为本轮产物：D、C 每个 preset 一条正式轨迹，N 为常量（无训练），R 复用 D 的树前缀并重新选择检索配置。G/L 为 44300426 的复用点参照，行级 row_id/y 已核验并复算 RMSE；未重训。

## 曲线摘要（同一配方/配置：选中轮 / 最后一轮 / 最后 20 轮均值）

| dataset | arm | rounds | valid_selected | valid_final | valid_mean_last20 | test_selected | test_final | test_mean_last20 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ogbg-molesol | D | 800 | 0.9270 | 0.9271 | 0.9274 | 0.8159 | 0.8163 | 0.8154 |
| ogbg-molesol | N | 0 | 1.5293 | 1.5293 | 1.5293 | 1.5846 | 1.5846 | 1.5846 |
| ogbg-molesol | R | 800 | 0.8814 | 0.8824 | 0.8829 | 0.8282 | 0.8373 | 0.8366 |
| ogbg-molesol | C | 1600 | 1.5297 | 2.9708 | 2.9709 | 1.5824 | 3.0342 | 3.0341 |
| ogbg-mollipo | D | 800 | 0.7747 | 0.7747 | 0.7748 | 0.7628 | 0.7628 | 0.7630 |
| ogbg-mollipo | N | 0 | 0.9970 | 0.9970 | 0.9970 | 0.9178 | 0.9178 | 0.9178 |
| ogbg-mollipo | R | 800 | 0.7448 | 0.7451 | 0.7450 | 0.7135 | 0.7131 | 0.7134 |
| ogbg-mollipo | C | 1600 | 0.9991 | 1.5526 | 1.5522 | 0.9181 | 1.4629 | 1.4624 |

最后一轮与最后 20 轮均值只作曲线摘要报告，从不替代 Val 选中的状态；N 为常量预测，三者相同。选中轮早于最后一轮时，二者的差就是该配方在本任务上的过拟合/漂移幅度。

## C 相对 D / N / R 的差距与配对不确定性

| dataset | comparison | split | rmse_c | rmse_b | delta_rmse | relative_delta | ci95_low | ci95_high | drop_one_delta | drop_one_sign_flip |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ogbg-molesol | C-D | valid | 1.5297 | 0.9270 | 0.6027 | 0.6502 | 0.4181 | 0.7897 | 0.6418 | no |
| ogbg-molesol | C-N | valid | 1.5297 | 1.5293 | 0.0005 | 0.0003 | -0.0111 | 0.0135 | 0.0022 | no |
| ogbg-molesol | C-R | valid | 1.5297 | 0.8814 | 0.6483 | 0.7356 | 0.4643 | 0.8296 | 0.6905 | no |
| ogbg-molesol | C-D | test | 1.5824 | 0.8159 | 0.7665 | 0.9394 | 0.5529 | 0.9899 | 0.7848 | no |
| ogbg-molesol | C-N | test | 1.5824 | 1.5846 | -0.0022 | -0.0014 | -0.0127 | 0.0086 | -0.0022 | no |
| ogbg-molesol | C-R | test | 1.5824 | 0.8282 | 0.7542 | 0.9106 | 0.5447 | 0.9746 | 0.7739 | no |
| ogbg-mollipo | C-D | valid | 0.9991 | 0.7747 | 0.2244 | 0.2897 | 0.1549 | 0.2946 | 0.2376 | no |
| ogbg-mollipo | C-N | valid | 0.9991 | 0.9970 | 0.0021 | 0.0021 | 0.0001 | 0.0041 | 0.0021 | no |
| ogbg-mollipo | C-R | valid | 0.9991 | 0.7448 | 0.2543 | 0.3415 | 0.1894 | 0.3267 | 0.2681 | no |
| ogbg-mollipo | C-D | test | 0.9181 | 0.7628 | 0.1552 | 0.2035 | 0.0876 | 0.2248 | 0.1668 | no |
| ogbg-mollipo | C-N | test | 0.9181 | 0.9178 | 0.0003 | 0.0003 | -0.0019 | 0.0024 | 0.0002 | no |
| ogbg-mollipo | C-R | test | 0.9181 | 0.7135 | 0.2046 | 0.2867 | 0.1408 | 0.2687 | 0.2176 | no |

bootstrap 按 query 分子重采样 2000 次（RNG 1703，固定模型与训练参照，pair 不作为独立样本）；`drop_one_*` 为移除该比较中参照模型误差最大的同一个 query 后重算的 ΔRMSE。区间只描述固定模型下的重采样不确定性。

## 冻结判定

| dataset | arm_C_val | arm_C_test | best_reference | best_valid | best_test | competitive | strong_joint_task | not_regressing | val_progress | test_progress | substantive |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ogbg-molesol | 1.5297 | 1.5824 | L | 0.8686 | 0.8570 | no | no | no | no | no | no |
| ogbg-mollipo | 0.9991 | 0.9181 | G | 0.6091 | 0.6456 | no | no | no | no | no | no |

**ogbg-molesol**

- 相对增益 gain_b = 1 - RMSE_C/RMSE_b：D_test=-0.9394, D_valid=-0.6502, N_test=0.0014, N_valid=-0.0003, R_test=-0.9106, R_valid=-0.7356
- strong_joint_task=no（六条件：D_test=no, D_valid=no, N_test=no, N_valid=no, R_test=no, R_valid=no）
- 另一任务 5% 条件：D_test=no, D_valid=no, N_test=yes, N_valid=yes, R_test=no, R_valid=no
- valid: J=0.8814 b=0.8686 c=1.5297 half_gap_closed=no competitive_5pct=no progress_pass=no
- test: J=0.8159 b=0.8570 c=1.5824 half_gap_closed=not_applicable competitive_5pct=no progress_pass=no

**ogbg-mollipo**

- 相对增益 gain_b = 1 - RMSE_C/RMSE_b：D_test=-0.2035, D_valid=-0.2897, N_test=-0.0003, N_valid=-0.0021, R_test=-0.2867, R_valid=-0.3415
- strong_joint_task=no（六条件：D_test=no, D_valid=no, N_test=no, N_valid=no, R_test=no, R_valid=no）
- 另一任务 5% 条件：D_test=no, D_valid=no, N_test=yes, N_valid=yes, R_test=no, R_valid=no
- valid: J=0.7448 b=0.6091 c=0.9991 half_gap_closed=no competitive_5pct=no progress_pass=no
- test: J=0.7135 b=0.6456 c=0.9181 half_gap_closed=no competitive_5pct=no progress_pass=no

- JOINT_DELTA_SIGNAL: **NO_JOINT_INCREMENT_THIS_RECIPE**
- SUBSTANTIVE_PROGRESS: **no**
- EXECUTION_PASS: **yes**
- WORTH_STRUCTURAL_COMPARATOR_FOLLOWUP: **no**

## C 的拟合层级（pair / 分子 Train / Val / Test）

| dataset | config | pair_rmse_std_selected | pair_rmse_std_antisym_selected | mean_last20_pair_rmse_std | train_rmse_molecule | valid_rmse_molecule | test_rmse_molecule |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ogbg-molesol | K4_softmax_tau_0.1 | 1.2842 | 1.2842 | 0.2544 | 1.1543 | 1.5297 | 1.5824 |
| ogbg-mollipo | K16_softmax_tau_0.1 | 1.3233 | 1.3233 | 0.6024 | 0.8476 | 0.9991 | 0.9181 |

## C 的退化机制（曲线层面）

**ogbg-molesol**（C-H2，配置 `K4_softmax_tau_0.1`，选中轮 = 1 / 共 1600 轮）

- 训练 pair 加权 RMSE（标准化目标）：第 1 轮 1.2842 → 第 1600 轮 0.2539
- 分子 Train RMSE：第 1 轮 1.1543 → 第 1600 轮 0.4286（该 split 全曲线最好 0.4286）
- Val RMSE：第 1 轮 1.5297 → 第 1600 轮 2.9708（该 split 全曲线最好 1.5297）
- Test RMSE：第 1 轮 1.5824 → 第 1600 轮 3.0342（该 split 全曲线最好 1.5824）
- 对照：N 的 Val/Test = 1.5293/1.5846，D 的 Val/Test = 0.9270/0.8159
- 与自身基线 N 相比：Val 上 C 在 1600 轮中有 0 轮优于 N（N = 1.5293），Test 上 2/1600 轮（N = 1.5846）
- 读法：pair 层与分子 Train 层单调变好，Val/Test 单调变差，且 C 在 Val 上的最好轮是第 1 轮——即“几乎不输出差值”的初始状态。差值模型把 Train-Train 的训练对拟合得越好，还原到未见 scaffold 的分子上偏差越大。

**ogbg-mollipo**（C-H2，配置 `K16_softmax_tau_0.1`，选中轮 = 1 / 共 1600 轮）

- 训练 pair 加权 RMSE（标准化目标）：第 1 轮 1.3233 → 第 1600 轮 0.6015
- 分子 Train RMSE：第 1 轮 0.8476 → 第 1600 轮 0.5048（该 split 全曲线最好 0.5048）
- Val RMSE：第 1 轮 0.9991 → 第 1600 轮 1.5526（该 split 全曲线最好 0.9991）
- Test RMSE：第 1 轮 0.9181 → 第 1600 轮 1.4629（该 split 全曲线最好 0.9181）
- 对照：N 的 Val/Test = 0.9970/0.9178，D 的 Val/Test = 0.7747/0.7628
- 与自身基线 N 相比：Val 上 C 在 1600 轮中有 0 轮优于 N（N = 0.9970），Test 上 0/1600 轮（N = 0.9178）
- 读法：pair 层与分子 Train 层单调变好，Val/Test 单调变差，且 C 在 Val 上的最好轮是第 1 轮——即“几乎不输出差值”的初始状态。差值模型把 Train-Train 的训练对拟合得越好，还原到未见 scaffold 的分子上偏差越大。

### 同一参照还原机制的对照：R vs D

| dataset | D_valid | R_valid | gain_valid | D_test | R_test | gain_test |
| --- | --- | --- | --- | --- | --- | --- |
| ogbg-molesol | 0.9270 | 0.8814 | 0.0492 | 0.8159 | 0.8282 | -0.0151 |
| ogbg-mollipo | 0.7747 | 0.7448 | 0.0386 | 0.7628 | 0.7135 | 0.0647 |

R 与 C 共用同一套 Tanimoto 参照、同一批权重配置和同一还原公式；差别只在 R 的锚点是已训练的绝对模型 D，而 C 的差值 h 是从 Train-Train 对里从零学的。R 相对 D 的增益：ogbg-molesol Val +4.9% / Test -1.5%；ogbg-mollipo Val +3.9% / Test +6.5%。也就是说，在有强绝对值锚点时参照残差修正是有效的（molesol 的 Test 是唯一例外，R 比 D 差约 1.5%），本轮的负结果落在“用 Train 对从零联合学习差值”这一步，而不是落在“检索 + 还原”机制本身。该对照是描述性的，不参与冻结判定。

## 参照覆盖（最高合法参照 Tanimoto 的固定分箱）

| dataset | split | bin_low | bin_high | n_queries | D_rmse | N_rmse | R_rmse | C_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ogbg-molesol | train | 0.0000 | 0.2000 | 7 | 0.4258 | 1.8607 | 0.5760 | 1.8177 |
| ogbg-molesol | train | 0.2000 | 0.4000 | 101 | 0.4287 | 1.4547 | 0.5150 | 1.4291 |
| ogbg-molesol | train | 0.4000 | 0.6000 | 350 | 0.3794 | 1.1873 | 0.4306 | 1.1624 |
| ogbg-molesol | train | 0.6000 | 0.8000 | 317 | 0.3513 | 0.9877 | 0.4171 | 0.9646 |
| ogbg-molesol | train | 0.8000 | 1.0000 | 127 | 0.3334 | 1.3001 | 0.4736 | 1.2691 |
| ogbg-molesol | valid | 0.0000 | 0.2000 | 11 | 0.8681 | 1.4160 | 0.8252 | 1.4181 |
| ogbg-molesol | valid | 0.2000 | 0.4000 | 51 | 0.9353 | 1.6020 | 0.9261 | 1.5993 |
| ogbg-molesol | valid | 0.4000 | 0.6000 | 42 | 0.9421 | 1.5222 | 0.8534 | 1.5260 |
| ogbg-molesol | valid | 0.6000 | 0.8000 | 7 | 0.7486 | 1.0719 | 0.8156 | 1.0421 |
| ogbg-molesol | valid | 0.8000 | 1.0000 | 2 | 1.2255 | 1.7248 | 0.8101 | 1.7991 |
| ogbg-molesol | test | 0.0000 | 0.2000 | 12 | 0.9315 | 1.8732 | 0.9844 | 1.8571 |
| ogbg-molesol | test | 0.2000 | 0.4000 | 56 | 0.8885 | 1.3848 | 0.8846 | 1.3729 |
| ogbg-molesol | test | 0.4000 | 0.6000 | 36 | 0.6955 | 1.8131 | 0.6857 | 1.8255 |
| ogbg-molesol | test | 0.6000 | 0.8000 | 7 | 0.6602 | 1.0722 | 0.7740 | 1.0835 |
| ogbg-molesol | test | 0.8000 | 1.0000 | 2 | 0.2616 | 1.9823 | 0.6853 | 1.9771 |
| ogbg-mollipo | train | 0.0000 | 0.2000 | 5 | 0.6970 | 1.8574 | 0.8666 | 1.8442 |
| ogbg-mollipo | train | 0.2000 | 0.4000 | 459 | 0.6148 | 1.2695 | 0.7125 | 1.2580 |
| ogbg-mollipo | train | 0.4000 | 0.6000 | 539 | 0.5678 | 1.0213 | 0.6081 | 1.0113 |
| ogbg-mollipo | train | 0.6000 | 0.8000 | 1351 | 0.4392 | 0.7495 | 0.4913 | 0.7429 |
| ogbg-mollipo | train | 0.8000 | 1.0000 | 1006 | 0.3740 | 0.6152 | 0.4024 | 0.6093 |
| ogbg-mollipo | valid | 0.0000 | 0.2000 | 2 | 1.3475 | 0.4193 | 1.4408 | 0.4246 |
| ogbg-mollipo | valid | 0.2000 | 0.4000 | 141 | 0.9230 | 1.2124 | 0.9194 | 1.2160 |
| ogbg-mollipo | valid | 0.4000 | 0.6000 | 104 | 0.8258 | 1.0755 | 0.7486 | 1.0779 |
| ogbg-mollipo | valid | 0.6000 | 0.8000 | 121 | 0.5880 | 0.7326 | 0.5623 | 0.7322 |
| ogbg-mollipo | valid | 0.8000 | 1.0000 | 52 | 0.5461 | 0.6888 | 0.5017 | 0.6889 |
| ogbg-mollipo | test | 0.0000 | 0.2000 | 3 | 1.2801 | 0.8841 | 1.2211 | 0.8964 |
| ogbg-mollipo | test | 0.2000 | 0.4000 | 136 | 0.9233 | 1.0794 | 0.9087 | 1.0810 |
| ogbg-mollipo | test | 0.4000 | 0.6000 | 104 | 0.6823 | 0.9694 | 0.6125 | 0.9655 |
| ogbg-mollipo | test | 0.6000 | 0.8000 | 126 | 0.6601 | 0.7589 | 0.5867 | 0.7614 |
| ogbg-mollipo | test | 0.8000 | 1.0000 | 51 | 0.6297 | 0.6672 | 0.5362 | 0.6669 |

最高相似度分位数：{"ogbg-molesol": {"test": {"max": 1.0, "mean": 0.380290171873253, "median": 0.36734694242477417, "min": 0.14516128599643707, "n": 113, "q25": 0.28333333134651184, "q75": 0.4615384638309479}, "train": {"max": 1.0, "mean": 0.6052579647818056, "median": 0.5882353186607361, "min": 0.0, "n": 902, "q25": 0.5, "q75": 0.7142857313156128}, "valid": {"max": 0.9090909361839294, "mean": 0.3888052682169771, "median": 0.3888888955116272, "min": 0.14814814925193787, "n": 113, "q25": 0.27272728085517883, "q75": 0.460317462682724}}, "ogbg-mollipo": {"test": {"max": 1.0, "mean": 0.5408604620468049, "median": 0.52452152967453, "min": 0.14754098653793335, "n": 420, "q25": 0.3524251878261566, "q75": 0.7124697417020798}, "train": {"max": 1.0, "mean": 0.6740983620773824, "median": 0.7321428656578064, "min": 0.17073170840740204, "n": 3360, "q25": 0.5381054282188416, "q75": 0.8142856955528259}, "valid": {"max": 1.0, "mean": 0.5421870468627839, "median": 0.5225576460361481, "min": 0.1603773534297943, "n": 420, "q25": 0.35384616255760193, "q75": 0.7092802822589874}}}
空箱保留；阈值未搜索。覆盖子群上的局部优势不能替代完整任务结论。

## 参照计算账本（每任务 3 个 Val query，仅计算分解）

### ogbg-molesol · median_abs_error_C · row_id=360 · SMILES=CN(C)C(=O)Nc1ccc(c(Cl)c1)n2nc(oc2=O)C(C)(C)C

y=-4.328000  pred_D=-4.275640  pred_C=-3.235483  |err_C|=1.092517  config=K4_softmax_tau_0.1  K_eff=4

| slot | ref_row_id | ref_smiles | similarity | weight | y_ref_raw | delta_raw | y_ref_plus_delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 731 | CN(C)C(=O)Nc1ccc(Cl)c(Cl)c1 | 0.4898 | 0.3487 | -3.8000 | 0.1057 | -3.6943 |
| 1 | 407 | CN(C)C(=O)Nc1ccc(C)c(Cl)c1 | 0.4706 | 0.2877 | -3.4830 | 0.1057 | -3.3773 |
| 2 | 837 | COc1ccc(NC(=O)N(C)C)cc1Cl | 0.4444 | 0.2215 | -2.5640 | 0.0117 | -2.5523 |
| 3 | 264 | CN(C)C(=O)Nc1ccc(Cl)cc1 | 0.4000 | 0.1420 | -2.8900 | 0.0026 | -2.8874 |

其余参照加权和 = 0.000000000；Top5 加权和 = -3.235483175；合计 = -3.235483175（与 pred_C 差 2.15e-09）

### ogbg-molesol · largest_improvement_vs_D · row_id=939 · SMILES=OC(CC(c1ccccc1)c3c(O)c2ccccc2oc3=O)c4ccc(cc4)c5ccc(Br)cc5

y=-4.445000  pred_D=-7.171814  pred_C=-4.418051  |err_C|=0.026949  config=K4_softmax_tau_0.1  K_eff=4

| slot | ref_row_id | ref_smiles | similarity | weight | y_ref_raw | delta_raw | y_ref_plus_delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 561 | CC(=O)CC(c1ccccc1)c3c(O)c2ccccc2oc3=O | 0.5472 | 0.5858 | -3.8930 | 0.1057 | -3.7873 |
| 1 | 583 | CC(=O)CC(c1ccc(Cl)cc1)c2c(O)c3ccccc3oc2=O | 0.5000 | 0.3655 | -5.8390 | 0.1057 | -5.7333 |
| 2 | 471 | c1ccccc1C(O)c2ccccc2 | 0.2292 | 0.0244 | -2.5500 | 0.1057 | -2.4443 |
| 3 | 562 | c1ccccc1C(O)C(O)c2ccccc2 | 0.2292 | 0.0244 | -1.9300 | 0.1057 | -1.8243 |

其余参照加权和 = 0.000000000；Top5 加权和 = -4.418051388；合计 = -4.418051388（与 pred_C 差 3.28e-09）

### ogbg-molesol · largest_regression_vs_D · row_id=866 · SMILES=CC=C(C(=CC)c1ccc(O)cc1)c2ccc(O)cc2

y=-4.950000  pred_D=-4.823862  pred_C=-1.439809  |err_C|=3.510191  config=K4_softmax_tau_0.1  K_eff=4

| slot | ref_row_id | ref_smiles | similarity | weight | y_ref_raw | delta_raw | y_ref_plus_delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 843 | Oc1ccc(O)cc1 | 0.4706 | 0.3701 | -0.1700 | -0.1057 | -0.2757 |
| 1 | 999 | Cc1ccc(O)cc1 | 0.4286 | 0.2431 | -0.7300 | -0.1057 | -0.8357 |
| 2 | 534 | COC(=O)c1ccc(O)cc1 | 0.4231 | 0.2301 | -1.8270 | -0.1057 | -1.9327 |
| 3 | 353 | CCC(C(CC)c1ccc(O)cc1)c2ccc(O)cc2 | 0.3846 | 0.1567 | -4.4300 | 0.0263 | -4.4037 |

其余参照加权和 = 0.000000000；Top5 加权和 = -1.439809035；合计 = -1.439809035（与 pred_C 差 2.84e-09）

### ogbg-mollipo · median_abs_error_C · row_id=1741 · SMILES=COc1ccc(CN2CCc3c(C2)[nH]c4ccccc34)cc1OC

y=3.100000  pred_D=2.988888  pred_C=2.512071  |err_C|=0.587929  config=K16_softmax_tau_0.1  K_eff=16

| slot | ref_row_id | ref_smiles | similarity | weight | y_ref_raw | delta_raw | y_ref_plus_delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 4139 | CC(=O)N[C@@H]1CC2(CCN(Cc3ccc4[nH]c5ccccc5c4c3)CC2)c6ccccc16 | 0.3289 | 0.1001 | 2.9600 | -0.0079 | 2.9521 |
| 1 | 3694 | COc1ccc(CC(=O)N(C)C2CCN(CCC(c3ccccc3)c4ccccc4)CC2)cc1OC | 0.3288 | 0.0999 | 3.4000 | -0.0063 | 3.3937 |
| 2 | 2454 | COc1cc2CC(CC3CCN(Cc4ccccc4)CC3)C(=O)c2cc1OC | 0.3143 | 0.0864 | 2.4100 | -0.0063 | 2.4037 |
| 3 | 2243 | COc1cc(CNc2nc3ccccc3[nH]2)cc(OC)c1OC | 0.2923 | 0.0694 | 2.3600 | -0.0079 | 2.3521 |
| 4 | 3595 | COc1ccccc1OCCNCC(O)COc2cccc3[nH]c4ccccc4c23 | 0.2917 | 0.0689 | 3.2700 | -0.0079 | 3.2621 |

其余参照加权和 = 1.282102766；Top5 加权和 = 1.229968463；合计 = 2.512071230（与 pred_C 差 2.08e-10）

### ogbg-mollipo · largest_improvement_vs_D · row_id=1049 · SMILES=Clc1ccc(cc1)S(=O)(=O)Nc2nnc(s2)c3ccccc3

y=1.140000  pred_D=2.822873  pred_C=0.957668  |err_C|=0.182332  config=K16_softmax_tau_0.1  K_eff=16

| slot | ref_row_id | ref_smiles | similarity | weight | y_ref_raw | delta_raw | y_ref_plus_delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 3301 | Cc1nnc(NS(=O)(=O)c2ccc(N)cc2)s1 | 0.4565 | 0.1666 | -1.3300 | 0.0433 | -1.2867 |
| 1 | 1139 | Nc1nnc(s1)c2ccccc2 | 0.3902 | 0.0859 | 1.8000 | 0.0433 | 1.8433 |
| 2 | 2950 | CCOC(=O)c1ccccc1c2csc(NS(=O)(=O)c3ccc(Cl)cc3)n2 | 0.3871 | 0.0832 | 2.4300 | -0.0079 | 2.4221 |
| 3 | 2305 | O=S(=O)(Nc1ccccc1c2ccccc2)c3ccccc3 | 0.3864 | 0.0826 | 3.9600 | -0.0063 | 3.9537 |
| 4 | 3083 | Fc1ccc(F)c(NS(=O)(=O)c2ccc(Cl)cc2)c1 | 0.3830 | 0.0799 | 2.9000 | 0.0079 | 2.9079 |

其余参照加权和 = 0.253331892；Top5 加权和 = 0.704335703；合计 = 0.957667594（与 pred_C 差 5.14e-10）

### ogbg-mollipo · largest_regression_vs_D · row_id=1220 · SMILES=COc1cccc(COCC(O)CN2CCC(CC2)C(=O)N)c1

y=-0.500000  pred_D=0.260279  pred_C=2.412827  |err_C|=2.912827  config=K16_softmax_tau_0.1  K_eff=16

| slot | ref_row_id | ref_smiles | similarity | weight | y_ref_raw | delta_raw | y_ref_plus_delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 3429 | NC(=O)c1cccc(OC2CCN(Cc3ccccc3)CC2)c1 | 0.3582 | 0.1019 | 1.9200 | 0.0079 | 1.9279 |
| 1 | 3719 | CN[C@@H](C)C(=O)N[C@H](C(=O)N[C@H]1CCCN(CCc2cccc(OC)c2)C1)C(C)(C)C | 0.3415 | 0.0862 | 1.4700 | 0.0197 | 1.4897 |
| 2 | 1657 | COc1cccc(c1)C(COCc2cc(cc(c2)C(F)(F)F)C(F)(F)F)N3CCNCC3 | 0.3378 | 0.0831 | 3.1000 | -0.0079 | 3.0921 |
| 3 | 3227 | COc1cc(ccc1N)C(=O)NC[C@@H](O)CN2CCC(CC2)Oc3ccc(Cl)c(Cl)c3 | 0.3293 | 0.0763 | 3.1200 | -0.0079 | 3.1121 |
| 4 | 3899 | NC(=O)C1CCN(CC(=O)Nc2ccc3OCOc3c2)CC1 | 0.3143 | 0.0657 | 0.4000 | 0.0433 | 0.4433 |

其余参照加权和 = 1.564257633；Top5 加权和 = 0.848569339；合计 = 2.412826972（与 pred_C 差 1.01e-10）

## 资源账本

| dataset | arm | preset | rounds | train_seconds | eval_seconds | matrix_build_seconds | seconds_per_round | gpu_peak_mb | model_sha256 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ogbg-molesol | D | H1 | 800 | 2.2388 | 1.4098 | 0.0000 | 0.0028 | 528.0000 | 719c3ab2fde6d7b3f6391aca5b2bdeca397cf3943b49110568e2975aad5b9024 |
| ogbg-molesol | D | H2 | 800 | 2.6820 | 1.3196 | 0.0000 | 0.0034 | 532.0000 | 000df6502fb675f4605b0b2b6defc65fe569639e199dcb1607f37db941b792ec |
| ogbg-molesol | C | H1 | 1600 | 15.1288 | 30.3028 | 2.8841 | 0.0095 | 1134.0000 | 9415143d5c8212a7ef18fe5d5611b5e9563a6149d00f032fbca81f53e96b06b8 |
| ogbg-molesol | C | H2 | 1600 | 16.5472 | 49.8026 | 2.2398 | 0.0103 | 1146.0000 | 32e73107b8f5b268cf5c4f3f21751e7a2411261c9a37fc17a47691621b4122bd |
| ogbg-mollipo | D | H1 | 800 | 2.2088 | 2.7391 | 0.0000 | 0.0028 | 536.0000 | 7532858029f2179e3ab18a16cd32d795eba71f52196cc6ee1405b32e776737d9 |
| ogbg-mollipo | D | H2 | 800 | 2.1399 | 3.0802 | 0.0000 | 0.0027 | 540.0000 | 2e28a5bf8c9ed835da3841ab43960e38e3e8620a81d59a958bc1531b4be3fd6f |
| ogbg-mollipo | C | H1 | 1600 | 34.5486 | 136.8870 | 11.7230 | 0.0216 | 10670.0000 | 9e75ae321be2c0bba316f5e56c3a72617bfd2a03ba7fcfed0e4c8125e01cf2e5 |
| ogbg-mollipo | C | H2 | 1600 | 38.2668 | 212.2467 | 10.8613 | 0.0239 | 10694.0000 | 78c128dc67aae760b5490bdcbc6d7031ea9767421ce048fd87771305aae9ccd3 |

- 轨迹算力合计（训练 + 每轮三集评测 + 一次性 pair 矩阵构建）= 579.3 s = 0.161 GPU·h；两个 worker 各独占一张 A100，并行执行。
- 两个 worker 最后一次执行的墙钟合计 = 52.1 s（每 GPU 一个训练 worker；本轮的启动失败与 resume 重跑另计，均记录在 `rr jobs`）。
- worker 进程 peak RSS：ogbg-molesol 2457 MB，ogbg-mollipo 6258 MB。GPU 峰值是 nvidia-smi 采样的整卡占用，只作上界参考。
- 预算：soft 3 GPU·h / hard 6 GPU·h（含 smoke 与重跑）；本轮轨迹算力远低于 soft 预算。
- 每轮评价成本已与一次性 pair 矩阵构建分开记录；C 的评测成本是训练本身的 2-6 倍，这是“每轮完整三集评测”的直接代价（K 越大、查询越多，代价越高）。

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
- 联合差值信号：NO_JOINT_INCREMENT_THIS_RECIPE（strong_joint_tasks=[]）。
- 实质进展：SUBSTANTIVE_PROGRESS = no。
- 是否值得下一步设计结构比较器：no。这只表示“值得进一步设计结构比较器”，不表示已获得结构解释、化学机制、组合泛化或第一梯队性能。

本轮的 3%/5%/50% 是投入门槛，不是统计显著性界限；负结果只针对本配方。
