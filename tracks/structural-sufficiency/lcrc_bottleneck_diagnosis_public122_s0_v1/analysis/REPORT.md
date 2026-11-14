# LCRC 瓶颈诊断 — 主报告（中文）

- round: `lcrc-bottleneck-diagnosis-public122-s0-v1`  ·  git: `3e38d4a8402d9a0c6d7344c6509db4fd08c40b23`  ·  generated: 2026-11-14T06:12:20Z
- 口径：单 seed（0）、官方既有划分、每 epoch 全量评估 Train_fit/Valid/Test，Test 参与开发诊断。`observed_oracle_gap` 只命名事后可观测差距，含 Test 选择乐观偏差，不是可实现收益、总体风险上界或正式成绩。**不同干预的增益不能相加。**

## 0. 结论先行

| 假设 | 判定 | 证据等级 |
|---|---|---|
| **H_SEL** 选择（Validation 选择/有限样本波动） | `SELECTION_SENSITIVITY` | direct |
| **H_OPT** 优化预算（同信息下更多优化） | `FIT_ONLY` | direct |
| **H_DATA** 监督样本量/覆盖 | `SAMPLE_EFFICIENCY_LIMIT` | direct |
| **H_USE** 冻结表示的读出利用 | `READOUT_NO_PURCHASE` | direct |
| **H_REP** 原输入缺任务信息（本轮不直接检验） | `not_established` | suspected_only |

**一句话取舍**：在本轮这个固定学习过程里，最明确、可复核的限制是**监督样本量（H_DATA）**；其次是**选择敏感性已被测到但没有可兑现的替代规则（H_SEL）**；**额外优化只降低 Train_fit、不再改善留出集（H_OPT → FIT_ONLY）**；**冻结 h 上没有买到读出余量（H_USE = 无采购）**。因此本轮不支持“表达力不足”这类默认结论，但也没有拿到能宣布表示缺口被排除的证据（H_REP 仍可疑、未被检验）。

支撑上面第一句的直接数字（Lipo，Test RMSE，R_SEL，同 U0/2U0 与同 50 候选机会）：

- `C1_R1@Lipophilicity`: N 840→3360，E1_R_SEL +0.1587，E2_R_SEL +0.1644，E1_A_SEL +0.1688，E2_A_SEL +0.1639，E1_W_SEL +0.1599，E2_W_SEL +0.1632（ruler 0.01）
- `DMPNN_D2F@Lipophilicity`: N 840→3360，E1_R_SEL +0.1897，E2_R_SEL +0.2065，E1_A_SEL +0.1640，E2_A_SEL +0.2136，E1_W_SEL +0.1855，E2_W_SEL +0.2106（ruler 0.01）

## 1. P0：用已有产物定义诊断基准（未重训任何旧臂）

| 臂 | n_epoch | Valid选中epoch | 选中Test | Test曲线最小 | observed_oracle_gap |
|---|---|---|---|---|---|
| C1_R1 | 360 | 187 | 0.5839 | 0.5638 | 0.0201 |
| C1_R3 | 360 | 161 | 0.5670 | 0.5641 | 0.0029 |
| CHEM8_R1 | 360 | 206 | 0.5660 | 0.5590 | 0.0070 |
| CONTRIB_R1 | 360 | 206 | 0.6063 | 0.5651 | 0.0413 |
| K_R1 | 360 | 90 | 0.5573 | 0.5463 | 0.0110 |

复现了提示里的两个例子（C1_R1 .58394/.56381、CONTRIB .60632/.56505）与 K 的 .55728。旧轨迹是 full-N、另一个 harness 的 360-epoch 口径；**新 N=25/50 的数字不能与旧 R1 直接相减**。

## 2. 数据量 × 优化预算（新轨迹，同一连续路径到 2U0）

![data budget](figures/data_budget.png)

**C1_R1@Lipophilicity**　主指标 RMSE

| fit | N | U0 | E1 R_SEL RMSE | E1 last10 | E2 R_SEL RMSE | E2 last10 | Train_fit | GPU·h |
|---|---|---|---|---|---|---|---|---|
| 25 | 840 | 9720 | 0.7508 | 0.7519 | 0.7545 | 0.7547 | 0.053→0.052 | 0.352 |
| 50 | 1680 | 9720 | 0.6416 | 0.6530 | 0.6537 | 0.6536 | 0.095→0.090 | 0.283 |
| 100 | 3360 | 9720 | 0.5921 | 0.5890 | 0.5901 | 0.5921 | 0.126→0.125 | 0.263 |

**DMPNN_D2F@Lipophilicity**　主指标 RMSE

| fit | N | U0 | E1 R_SEL RMSE | E1 last10 | E2 R_SEL RMSE | E2 last10 | Train_fit | GPU·h |
|---|---|---|---|---|---|---|---|---|
| 25 | 840 | 5300 | 0.7555 | 0.7817 | 0.7835 | 0.7910 | 0.053→0.033 | 0.043 |
| 50 | 1680 | 5300 | 0.6332 | 0.6680 | 0.6706 | 0.6860 | 0.140→0.080 | 0.048 |
| 100 | 3360 | 5300 | 0.5658 | 0.5678 | 0.5770 | 0.5841 | 0.228→0.167 | 0.044 |

**C1_R3@Lipophilicity**　主指标 RMSE

| fit | N | U0 | E1 R_SEL RMSE | E1 last10 | E2 R_SEL RMSE | E2 last10 | Train_fit | GPU·h |
|---|---|---|---|---|---|---|---|---|
| 100 | 3360 | 9720 | 0.5730 | 0.5864 | 0.5866 | 0.5878 | 0.124→0.117 | 0.253 |

**CHEM8_R1@Lipophilicity**　主指标 RMSE

| fit | N | U0 | E1 R_SEL RMSE | E1 last10 | E2 R_SEL RMSE | E2 last10 | Train_fit | GPU·h |
|---|---|---|---|---|---|---|---|---|
| 100 | 3360 | 9720 | 0.5917 | 0.5904 | 0.5897 | 0.5912 | 0.126→0.121 | 0.265 |

**K_R1@Lipophilicity**　主指标 RMSE

| fit | N | U0 | E1 R_SEL RMSE | E1 last10 | E2 R_SEL RMSE | E2 last10 | Train_fit | GPU·h |
|---|---|---|---|---|---|---|---|---|
| 100 | 3360 | 9720 | 0.5859 | 0.5852 | 0.5838 | 0.5868 | 0.108→0.109 | 0.356 |

**C1_R1@BACE**　主指标 ROC-AUC　（分类只跑至 U0，无 E2 leg）

| fit | N | U0 | E1 R_SEL ROC-AUC | E1 last10 | E2 R_SEL ROC-AUC | E2 last10 | Train_fit | GPU·h |
|---|---|---|---|---|---|---|---|---|
| 50 | 605 | 3600 | 0.7630 | 0.7600 | — | — | 1.000→— | 0.055 |
| 100 | 1210 | 3600 | 0.8176 | 0.8150 | — | — | 0.999→— | 0.062 |

**DMPNN_D2F@BACE**　主指标 ROC-AUC　（分类只跑至 U0，无 E2 leg）

| fit | N | U0 | E1 R_SEL ROC-AUC | E1 last10 | E2 R_SEL ROC-AUC | E2 last10 | Train_fit | GPU·h |
|---|---|---|---|---|---|---|---|---|
| 50 | 605 | 1900 | 0.7400 | 0.7415 | — | — | 0.999→— | 0.009 |
| 100 | 1210 | 1900 | 0.7960 | 0.7868 | — | — | 0.983→— | 0.008 |

读法：**N 每翻一倍，两个学习器的 Test 都明确变好**（H_DATA）。而 E1→E2 的额外优化在所有臂上都是 **Train_fit 变好、留出集不变或更差**（H_OPT 只到 FIT_ONLY）。注意 C1 的 Train_fit RMSE 远低于 D-MPNN（≈0.126 vs ≈0.228）却留出更差——“训练误差更低”显然不能当能力证据。

### 2.1 full-N 结构变体对照（另一层信息：加结构描述符本身有多有用）

| 臂 | 输入 | E1 R_SEL Test | E1 last10 | E2 R_SEL Test |
|---|---|---|---|---|
| LIP_C1_R1_F100 | atom/bond 图特征（C1 基线） | 0.5921 | 0.5890 | 0.5901 |
| LIP_CHEM8_R1_F100 | 冻结 8 个 chem8 原子列 | 0.5917 | 0.5904 | 0.5897 |
| LIP_K_R1_F100 | chem8 + Q8 图贡献残差(K) | 0.5859 | 0.5852 | 0.5838 |
| LIP_C1_R3_F100 | C1_R1，wd 1e-2 | 0.5730 | 0.5864 | 0.5866 |
| LIP_DMPNN_D2F_F100 | D-MPNN（不同架构与预算） | 0.5658 | 0.5678 | 0.5770 |

CHEM8 与 C1 基线几乎同水平，K 略好——**本轮不据此宣布“结构信息足够”或“表达力不足”**：这只说明在这三个固定配方下，换/加结构描述的收益小于数据量与选择摆动的量级。

## 3. 选择敏感性：差距存在，但换规则兑现不了

![update curves](figures/update_curves.png)

| 臂 | 阶段 | 候选 | observed_oracle_gap | 好状态最长持续点 | R_SEL upd/Test | A_SEL upd/Test | W_SEL upd/Test |
|---|---|---|---|---|---|---|---|
| LIP_C1_R1_F25 | E1 | 50 | 0.0150 | 9 | 8359/0.7508 | 8359/0.7508 | 9137/0.7520 |
| LIP_C1_R1_F25 | E2 | 50 | 0.0028 | 50 | 19051/0.7545 | 14774/0.7540 | 19051/0.7545 |
| LIP_C1_R1_F25 | E1+E2 | 100 | 0.0188 | 59 | 19051/0.7545 | 8359/0.7508 | 19051/0.7545 |
| LIP_DMPNN_D2F_F25 | E1 | 50 | 0.0311 | 1 | 1590/0.7555 | 1272/0.7321 | 1802/0.7536 |
| LIP_DMPNN_D2F_F25 | E2 | 50 | 0.0017 | 4 | 5936/0.7835 | 5936/0.7835 | 5936/0.7835 |
| LIP_DMPNN_D2F_F25 | E1+E2 | 100 | 0.0311 | 1 | 1590/0.7555 | 1272/0.7321 | 1802/0.7536 |
| LIP_C1_R1_F50 | E1 | 50 | 0.0119 | 2 | 3110/0.6416 | 2333/0.6297 | 6804/0.6539 |
| LIP_C1_R1_F50 | E2 | 50 | 0.0018 | 42 | 17885/0.6537 | 14774/0.6535 | 18274/0.6550 |
| LIP_C1_R1_F50 | E1+E2 | 100 | 0.0119 | 2 | 3110/0.6416 | 2333/0.6297 | 18274/0.6550 |
| LIP_DMPNN_D2F_F50 | E1 | 50 | 0.0097 | 3 | 1590/0.6332 | 3392/0.6495 | 3710/0.6521 |
| LIP_DMPNN_D2F_F50 | E2 | 50 | 0.0049 | 5 | 5406/0.6706 | 6254/0.6671 | 5830/0.6701 |
| LIP_DMPNN_D2F_F50 | E1+E2 | 100 | 0.0097 | 3 | 1590/0.6332 | 3392/0.6495 | 3710/0.6521 |
| LIP_C1_R1_F100 | E1 | 50 | 0.0101 | 6 | 8165/0.5921 | 5443/0.5820 | 8165/0.5921 |
| LIP_C1_R1_F100 | E2 | 50 | 0.0011 | 50 | 12830/0.5901 | 12830/0.5901 | 19051/0.5913 |
| LIP_C1_R1_F100 | E1+E2 | 100 | 0.0101 | 28 | 8165/0.5921 | 5443/0.5820 | 8165/0.5921 |
| LIP_DMPNN_D2F_F100 | E1 | 50 | 0.0127 | 1 | 3498/0.5658 | 5300/0.5681 | 5300/0.5681 |
| LIP_DMPNN_D2F_F100 | E2 | 50 | 0.0116 | 2 | 7420/0.5770 | 5936/0.5699 | 7844/0.5730 |
| LIP_DMPNN_D2F_F100 | E1+E2 | 100 | 0.0239 | 2 | 7420/0.5770 | 5936/0.5699 | 7844/0.5730 |
| LIP_C1_R3_F100 | E1 | 50 | 0.0211 | 3 | 6221/0.5730 | 5443/0.5783 | 6610/0.5841 |
| LIP_C1_R3_F100 | E2 | 50 | 0.0006 | 50 | 13219/0.5866 | 12636/0.5872 | 13414/0.5883 |
| LIP_C1_R3_F100 | E1+E2 | 100 | 0.0211 | 3 | 6221/0.5730 | 5443/0.5783 | 6610/0.5841 |
| LIP_CHEM8_R1_F100 | E1 | 50 | 0.0089 | 7 | 8165/0.5917 | 5249/0.5878 | 9526/0.5896 |
| LIP_CHEM8_R1_F100 | E2 | 50 | 0.0010 | 50 | 15552/0.5897 | 11858/0.5890 | 19440/0.5912 |
| LIP_CHEM8_R1_F100 | E1+E2 | 100 | 0.0089 | 16 | 8165/0.5917 | 5249/0.5878 | 19440/0.5912 |
| LIP_K_R1_F100 | E1 | 50 | 0.0187 | 3 | 5249/0.5859 | 5443/0.5771 | 6026/0.5907 |
| LIP_K_R1_F100 | E2 | 50 | 0.0000 | 27 | 13219/0.5838 | 13219/0.5838 | 13414/0.5847 |
| LIP_K_R1_F100 | E1+E2 | 100 | 0.0187 | 3 | 5249/0.5859 | 5443/0.5771 | 6026/0.5907 |
| BACE_C1_R1_F50 | E1 | 50 | 0.0592 | 18 | 2808/0.7630 | 288/0.7871 | 3024/0.7609 |
| BACE_C1_R1_F50 | E1+E2 | 50 | 0.0592 | 18 | 2808/0.7630 | 288/0.7871 | 3024/0.7609 |
| BACE_DMPNN_D2F_F50 | E1 | 50 | 0.0322 | 5 | 1254/0.7400 | 76/0.6763 | 1558/0.7411 |
| BACE_DMPNN_D2F_F50 | E1+E2 | 50 | 0.0322 | 5 | 1254/0.7400 | 76/0.6763 | 1558/0.7411 |
| BACE_C1_R1_F100 | E1 | 50 | 0.0261 | 2 | 576/0.8176 | 288/0.7891 | 1080/0.8252 |
| BACE_C1_R1_F100 | E1+E2 | 50 | 0.0261 | 2 | 576/0.8176 | 288/0.7891 | 1080/0.8252 |
| BACE_DMPNN_D2F_F100 | E1 | 50 | 0.0138 | 2 | 836/0.7960 | 76/0.7120 | 912/0.8071 |
| BACE_DMPNN_D2F_F100 | E1+E2 | 50 | 0.0138 | 2 | 836/0.7960 | 76/0.7120 | 912/0.8071 |

### 3.1 full-N 面板：Validation 组成 → 选择（2000 次 paired bootstrap + 逐行 jackknife）

| 臂 | 模态选中update | Test P10 | P50 | P90 | 不同选中数 | jackknife Test 极差 | 选择稳定度 |
|---|---|---|---|---|---|---|---|
| LIP_C1_R1_F100 | 7387 | 0.5844 | 0.5889 | 0.5956 | 31 | 0.0111 | 0.979 |
| LIP_C1_R3_F100 | 5443 | 0.5730 | 0.5783 | 0.5841 | 24 | 0.0053 | 0.995 |
| LIP_CHEM8_R1_F100 | 8165 | 0.5846 | 0.5898 | 0.5917 | 27 | 0.0057 | 0.998 |
| LIP_K_R1_F100 | 5249 | 0.5771 | 0.5859 | 0.5885 | 19 | 0.0027 | 0.998 |
| LIP_DMPNN_D2F_F100 | 3498 | 0.5606 | 0.5675 | 0.6213 | 26 | 0.0019 | 0.998 |
| BACE_C1_R1_F100 | 576 | 0.7891 | 0.8223 | 0.8332 | 39 | 0.0076 | 0.954 |
| BACE_DMPNN_D2F_F100 | 836 | 0.7944 | 0.7996 | 0.8071 | 35 | 0.0000 | 1.000 |

替代规则（`A_SEL` 对 `R_SEL`）：6/13 条为正，中位增益 0.0000；`W_SEL`：5/13，中位 -0.0012。判定 `NO_CLEAR_AVOIDABLE_LIMIT`。

这两段必须分开读：**bootstrap/jackknife 说的是“同一批训练状态下，验证集组成会让选中结果漂移”**（结果：确实漂移——Lipo C1_R1 的 jackknife Test 极差 0.0111，和整条 E2 优化增益同量级；DMPNN F100 的 bootstrap P90=0.6213 说明尾部会挑到早期坏状态）；**规则对照说的是“有没有一条事先固定的替代规则能稳定兑现”**（结果：没有）。所以是诊断到评估敏感性，而不是已经拿到性能收益。

## 4. 优化余量与唯一重启臂

判定 `FIT_ONLY`。逐臂 E1→E2 的留出增益与 Train_fit 增益（正号=更好）：

| 臂 | 留出增益 | Train_fit 增益 | 标签 |
|---|---|---|---|
| LIP_C1_R1_F25 | -0.0037 | 0.0018 | `FIT_ONLY` |
| LIP_C1_R1_F50 | -0.0121 | 0.0054 | `FIT_ONLY` |
| LIP_C1_R1_F100 | 0.0020 | 0.0010 | `FIT_ONLY` |
| LIP_DMPNN_D2F_F25 | -0.0280 | 0.0207 | `FIT_ONLY` |
| LIP_DMPNN_D2F_F50 | -0.0373 | 0.0605 | `FIT_ONLY` |
| LIP_DMPNN_D2F_F100 | -0.0112 | 0.0604 | `FIT_ONLY` |
| LIP_C1_R3_F100 | -0.0136 | 0.0069 | `FIT_ONLY` |
| LIP_CHEM8_R1_F100 | 0.0021 | 0.0051 | `FIT_ONLY` |
| LIP_K_R1_F100 | 0.0020 | -0.0005 | `NO_EFFECT` |

重启触发条件（冻结）：Train RMSE 降幅 < 0.05 且 Test 增益 < 0.01。实测：Train 降幅 0.0082、Test 增益 0.0020 → `FIRE_RESTART`。
已按协议执行 1 条重启臂（`LIP_C1_R1_F100_RESTART`：从 `LIP_C1_R1_F100/u0_snapshot.pt` 恢复 model+Adam+数据迭代器+RNG，以原 peak lr 重启 cosine 再跑 U0 步，不 reset Adam）。

### 4.1 重启臂 vs 低 lr 延长（同 prefix、同 50 点 E2 网格）

| 方案 | R_SEL 选中update | E2 R_SEL Test | E2 last10 Test | E2 last10 Train_fit | observed_oracle_gap |
|---|---|---|---|---|---|
| 低 lr 延长（F100 的 E2 leg） | 12830 | 0.5901 | 0.5921 | 0.125 | 0.0011 |
| 重启臂（peak lr 重跑 U0 步） | 18857 | 0.5930 | 0.5924 | 0.125 | 0.0042 |

重启相对低 lr 延长的 Test 增益 = -0.0029（正号=重启更好）。两者 Train_fit 几乎相同、留出集都不改善；即 **换一种优化路径也换不出泛化收益**，进一步把 H_OPT 钉在 FIT_ONLY。

## 5. 冻结表示的读出定位

![readout](figures/readout.png)

源：`LIP_C1_R1_F100/u0_snapshot.pt`（update = U0 = 9720 的终点状态，不含任何按 Valid/Test 选 epoch 的步骤；见 PROTOCOL 附录 A1）。三条探针共用同一份冻结 `h`，新增参数都恰为 14705，step0 与 P_WARM 等价（最大绝对差 ~1e-7，float32 噪声）。

| probe | new_params | R_SEL 选中 | Valid RMSE | Test RMSE | Test 曲线最小 | observed_oracle_gap |
|---|---|---|---|---|---|---|
| P_WARM | 0 | 0 | 0.7254 | 0.5897 | 0.5823 | 0.0074 |
| P_POST | 14705 | 0 | 0.7254 | 0.5897 | 0.5823 | 0.0074 |
| P_PRE | 14705 | 0 | 0.7254 | 0.5897 | 0.5823 | 0.0074 |

采购门槛（冻结）：Lipo Valid 与 Test RMSE 都需相对改善 ≥1%。判定 `READOUT_NO_PURCHASE` → BACE 探针 `NOT_RUN`（未采购，不算“无效”）。

**三条探针的 R_SEL 都落在 step 0**：头在 1800 步里把 Train_fit 从 0.127 降到 0.099，Valid 却从 0.7254 单调升到 ≈0.7369，所以“按 Valid 选”必然回到未训练状态——这本身就是“读出侧没有可兑现余量”的直接证据，而不是探针没跑起来。若改用事后 oracle（Test 曲线最小），三条都只到 0.5823@180 步，相对 step0 的 observed_oracle_gap 仅 ≈0.0074（仍含 Test 选择乐观偏差，且低于 0.01 的实用尺）。

**必须说清的实现限制**：`P_POST` 与 `P_PRE` 的预测最大差仅 5.24e-04（对 P_WARM 是 1.85e-02）。原因是 delta 末层零初始化 + 两条支路串联：新增非线性支路在 1800 步内几乎没被用起来。因此 **PRE vs POST 这一对照在本轮是退化的，“池化位置”没有结论**；能说的只是：在这个预算下，多给 14.7k 参数的非线性头**没有买到** Test 改善。

## 6. 判读边界（明确不做的推断）

- 不把 oracle gap、数据增益、probe 增益相加去解释“总误差的百分之多少”。
- 不把 U0/2U0 当两次独立实验；它们是同一路径的两个观察预算。
- C1 的 Train_fit 更低却留出更差，只能说明它在记忆结构身份，不能说明“欠训练”或“表达力不足”。
- 单 seed、单嵌套子集序列、共用 norm 的受控设定：这是固定学习过程的样本效率，不是调优后的样本复杂度。
- 新轨迹不承诺逐位复现旧口径（E1 只在 50 点网格上选，旧口径在 360 个 epoch 上选；target norm 也不同）。
- BACE 只到 U0、每格 50 候选，且分类主 AUC、回归主 RMSE，绝对数字不混平均。

## 7. 预算、执行与一次基础设施事故

- regime：`res` / a100-2 / 2×A100-SXM4-40GB / driver 550.163.01 / torch 2.5.1+cu124；pool `res-gpu0`+`res-gpu1`，每 pool 一条进程。
- 正式轨迹：**13 条主体 + 3 条 Lipo 读出探针 + 1 条重启臂**（上限 20，未超）。保留产物计费 GPU·h = **2.18**；按 job 墙钟统计：有用 182.8 min、重复浪费 94.3 min。
- 事故：local scheduler (code/lb_queue.py v1) treated a transiently empty `rr jobs` listing as 'no job running' and relaunched LIP_C1_R1_F25 (5 extra) and LIP_C1_R1_F50 (2 extra) into the same --out-dir. All extras were cancelled within ~8 min. Detected and fixed the same hour: fail-closed reconciliation + local launch ledger + settle window (v2).
- 完整性：every retained arm satisfies updates_run == Umax, grid_updates length 100 (Lipo) / 50 (BACE), last grid update == Umax, and curve.csv/grid_predictions agree on the raw-unit Test RMSE of every grid state; the retained LIP_C1_R1_F25 payload is the last-finishing survivor's complete write, reproduced deterministically by seed 0.
- 结论影响：重复臂与保留臂同 seed、同 commit、同数据顺序，属于同一确定性轨迹的重复执行，不产生新信息也不用作独立重复；只消耗预算。阻塞类失败 0 条（3 次即时失败 0.3 min 内修复重启）。

## 8. 产物与复现

```bash
uv run python code/run.py prepare                       # 子集/norm/缓存指纹
uv run python code/run.py train --model C1_R1 --dataset Lipophilicity --fit 25
uv run python code/run.py analyze && uv run python code/run.py report
uv run python -m pytest tests -q                        # 14 项不变量
```

- 机器量：`analysis/synthesis/{prior_evidence,selection,data_budget,readout,gates}.json`、`analysis/run_manifest.json`、`analysis/incident.json`、`analysis/queue_history.json`。
- 每条轨迹：`train.json`（含 U0/Umax、初始 state hash、共用 norm 指纹、50 点网格、三规则选中权重）、`curve.csv`（每 epoch + 每网格点的全量三划分评估）、`grid_predictions.npz`（每个 grid 状态逐样本预测）、`u0_snapshot.pt`、`best_{R,A,W}_SEL.pt`、`last.pt`、`step0.pt`。
- 权重副本：`runs/` 本地保留（`.gitignore` 排除大文件），远端 `res` 同路径可恢复。
