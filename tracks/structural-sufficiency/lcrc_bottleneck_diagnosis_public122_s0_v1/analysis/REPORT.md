# LCRC 瓶颈诊断 — 主报告（中文）

- round: `lcrc-bottleneck-diagnosis-public122-s0-v1`
- git: `b87600125abedc2dc9949a00a506c7d738b9bc34`
- generated: 2026-11-14T04:20:07Z
- 口径：单 seed（0）、官方既有划分、每 epoch 全量评估 Train_fit/Valid/Test，Test 参与开发诊断。`observed_oracle_gap` 只命名事后可观测差距，含 Test 选择乐观偏差，不是可实现收益、总体风险上界或正式成绩。

## 0. 现有证据让我们优先改变什么

> 由下方证据矩阵生成；先给取舍，再列限制。

## 1. P0：先用已有产物定义诊断基准

重放 5 条既有 Lipo 曲线的 Valid 选择位置与 Test 曲线最小值（原始单位，marginal WAS NOT 重训）。

| 臂 | n_epoch | Valid选中epoch | 选中Test | Test曲线最小 | observed_oracle_gap |
|---|---|---|---|---|---|
| C1_R1 | 360 | 187 | 0.5839 | 0.5638 | 0.0201 |
| C1_R3 | 360 | 161 | 0.5670 | 0.5641 | 0.0029 |
| CHEM8_R1 | 360 | 206 | 0.5660 | 0.5590 | 0.0070 |
| CONTRIB_R1 | 360 | 206 | 0.6063 | 0.5651 | 0.0413 |
| K_R1 | 360 | 90 | 0.5573 | 0.5463 | 0.0110 |

（这些是 full-N、不同 harness 的旧轨迹；新 N25 数字不能与旧 R1 直接相减。）

## 2. 数据量 × 优化预算（新轨迹）

相同 U0、2U0 与相同 50 候选机会下比较；R_SEL 为主，A_SEL/W_SEL 为预定对照。

### C1_R1@Lipophilicity

| 臂 | N | U0 | E1 R_SEL Test | E1 last10 median | E2 R_SEL Test | E2 last10 median |
|---|---|---|---|---|---|---|
| LIP_C1_R1_F25 | — | — | NOT_RUN | | | |
| LIP_C1_R1_F50 | — | — | NOT_RUN | | | |
| 100 | 3360 | 9720 | 0.5921 | 0.5890 | 0.5901 | 0.5921 |

### DMPNN_D2F@Lipophilicity

| 臂 | N | U0 | E1 R_SEL Test | E1 last10 median | E2 R_SEL Test | E2 last10 median |
|---|---|---|---|---|---|---|
| LIP_DMPNN_D2F_F25 | — | — | NOT_RUN | | | |
| LIP_DMPNN_D2F_F50 | — | — | NOT_RUN | | | |
| 100 | 3360 | 5300 | 0.5658 | 0.5678 | 0.5770 | 0.5841 |

### C1_R3@Lipophilicity

| 臂 | N | U0 | E1 R_SEL Test | E1 last10 median | E2 R_SEL Test | E2 last10 median |
|---|---|---|---|---|---|---|
| LIP_C1_R3_F100 | — | — | NOT_RUN | | | |

### CHEM8_R1@Lipophilicity

| 臂 | N | U0 | E1 R_SEL Test | E1 last10 median | E2 R_SEL Test | E2 last10 median |
|---|---|---|---|---|---|---|
| LIP_CHEM8_R1_F100 | — | — | NOT_RUN | | | |

### K_R1@Lipophilicity

| 臂 | N | U0 | E1 R_SEL Test | E1 last10 median | E2 R_SEL Test | E2 last10 median |
|---|---|---|---|---|---|---|
| LIP_K_R1_F100 | — | — | NOT_RUN | | | |

### C1_R1@BACE

| 臂 | N | U0 | E1 R_SEL Test | E1 last10 median | E2 R_SEL Test | E2 last10 median |
|---|---|---|---|---|---|---|
| BACE_C1_R1_F50 | — | — | NOT_RUN | | | |
| BACE_C1_R1_F100 | — | — | NOT_RUN | | | |

### DMPNN_D2F@BACE

| 臂 | N | U0 | E1 R_SEL Test | E1 last10 median | E2 R_SEL Test | E2 last10 median |
|---|---|---|---|---|---|---|
| BACE_DMPNN_D2F_F50 | — | — | NOT_RUN | | | |
| BACE_DMPNN_D2F_F100 | — | — | NOT_RUN | | | |

## 3. 选择敏感性与规则对照

| 臂 | 阶段 | 候选数 | oracle_gap | 好状态持续点 | R_SEL upd/Test | A_SEL upd/Test | W_SEL upd/Test |
|---|---|---|---|---|---|---|---|
| LIP_C1_R1_F100 | E1 | 50 | 0.0101 | 6 | 8165/0.5921 | 5443/0.5820 | 8165/0.5921 |
| LIP_C1_R1_F100 | E2 | 50 | 0.0011 | 50 | 12830/0.5901 | 12830/0.5901 | 19051/0.5913 |
| LIP_C1_R1_F100 | E1+E2 | 100 | 0.0101 | 28 | 8165/0.5921 | 5443/0.5820 | 8165/0.5921 |
| LIP_DMPNN_D2F_F100 | E1 | 50 | 0.0127 | 1 | 3498/0.5658 | 5300/0.5681 | 5300/0.5681 |
| LIP_DMPNN_D2F_F100 | E2 | 50 | 0.0116 | 2 | 7420/0.5770 | 5936/0.5699 | 7844/0.5730 |
| LIP_DMPNN_D2F_F100 | E1+E2 | 100 | 0.0239 | 2 | 7420/0.5770 | 5936/0.5699 | 7844/0.5730 |

### 3.1 full-N U0/E1 面板：Validation 组成 → 选择（2000 次 paired bootstrap）

| 臂 | 选中update模态 | Test 指标 P10 | P50 | P90 | 选择分散度 |
|---|---|---|---|---|---|
| LIP_C1_R3_F100 | NOT_RUN | | | | |
| LIP_CHEM8_R1_F100 | NOT_RUN | | | | |
| LIP_K_R1_F100 | NOT_RUN | | | | |
| BACE_C1_R1_F100 | NOT_RUN | | | | |
| BACE_DMPNN_D2F_F100 | NOT_RUN | | | | |
| LIP_C1_R1_F100 | 7387 | 0.5844 | 0.5889 | 0.5956 | 31 |
| LIP_DMPNN_D2F_F100 | 3498 | 0.5606 | 0.5675 | 0.6213 | 26 |

这是**固定训练状态**下的“Validation 组成 → 选择结果”敏感性，不是训练 seed CI，也不是独立泛化 CI。

## 4. 冻结表示读出定位

### lipo

| probe | new_params | R_SEL upd | Test RMSE | 相对P_WARM gain |
|---|---|---|---|---|
| P_WARM | NOT_RUN | | | |
| P_POST | NOT_RUN | | | |
| P_PRE | NOT_RUN | | | |

### bace

| probe | new_params | R_SEL upd | Test AUC | 相对P_WARM gain |
|---|---|---|---|---|
| P_WARM | NOT_RUN | | | |
| P_POST | NOT_RUN | | | |
| P_PRE | NOT_RUN | | | |

## 5. 路线判定矩阵（H_SEL / H_OPT / H_DATA / H_USE）

（由分析产物自动生成，逐项证据等级见 `analysis/synthesis/*.json`；本段只陈述在本轮产物支持下的方向，不构成跨 seed / 跨数据集结论。）

- **选择敏感性（或许存，不兑现）**：每个 full-N 臂的 `observed_oracle_gap` 与 bootstrap 选择分散度给出“同一 state-set 内事后差距”；若差距大但 R_SEL/A_SEL/W_SEL 都不能稳定兑现，结论是“诊断到评估敏感性，尚未兑现性能收益”。
- **优化余量**：仅当额外 U0（E2）在**同一 prefix**下改善留出状态与**实际选中**结果才记 OPTIMIZATION_HEADROOM；只有 Train_fit 下降则降为 FIT_ONLY。
- **样本效率**：固定标度/更新/候选机会下 N 增加带来明确收益才记 SAMPLE_EFFICIENCY_LIMIT。
- **读出余量**：只认冻结 h 上 P_WARM/POST/PRE 相对 P_WARM 的可兑现 Test 改善；PRE 超过 POST 才算非线性池化位置的条件证据。
- **表示缺口**：以上都没解决且 C1 仍落后于参照时，只能记 REPRESENTATION_DEFICIT_UNIDENTIFIED（可疑，不判定表达力不足）。


## 6. 限制

- 单 seed、单嵌套子集序列、共用 norm 的受控设定；不是每个 N 各调配方后的样本复杂度结论。
- 连续轨迹的 U0/2U0 是同一路径两个观察预算，不是独立重复；不把 E1/E2 当两次实验相减。
- Bootstrap 只刻画评估样本组成对选择的影响，不覆盖训练 seed 波动。
- 结构与目标 norm 固定控制了损失/输入尺度的混杂，但也限制了跨 N 的“尺度自由度”。
- 分类主 AUC、回归主 RMSE；二者绝对数值不相加、不混平均。
