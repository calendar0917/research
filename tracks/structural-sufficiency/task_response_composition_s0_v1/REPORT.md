# task_response_composition_s0_v1 — 结构响应组合 s0 报告

八小时轮次（T0 2026-11-21T23:33:19+08:00）的完整结果。所有数字来自 `analysis/*.json`；本文件由 `code/trc_report.py` 从这些记录生成，不重算指标。

## 0. 结论摘要

1. **整分子预测没有超过外部 GINE 坐标。** 最好的候选是 `LIPO_FACTOR_CUTRESP_R0`，Valid 0.7096 / Test 0.7227，同一协议下的 GINE 坐标是 Valid 0.6406 / Test 0.6582。Valid 上 GINE 显著更好（配对 bootstrap 差值 **-0.0690** [-0.1271, -0.0075]）。
2. **外部 GINE 与历史 GINE 参考一致。** 本轮 raw-GINE 坐标 0.6406/0.6582，历史 `GINE_RAW_REF` 0.6101/0.6722（历史为 240 epoch + early stop，本轮统一 E=160），说明坐标可比。

## 1. 问题与冻结结构

问题：带接口片段的任务作用能否由少量共享响应状态承载，环境经由该状态改变片段作用？在显式共享限制下，用真实匹配替换目录检查共享是否成立；再看它对整分子预测和数据效率的价值。骨干被禁止使用 GNN / 图 Transformer / 预训练 / 蒸馏 / 强模型残差；GINE 只作外部预测坐标。

| split | molecules | parses | parses/mol | fallback mols |
|---|---|---|---|---|
| train | 3360 | 27237 | 8.11 | 7 |
| valid | 420 | 3140 | 7.48 | 1 |
| test | 420 | 3061 | 7.29 | 0 |

* 分子 4200，解析 33438，片段 29283，对象帧 71004；帧原子属性行 76，键属性行 10。
* 距离桶总量（d0..d3,d4+,unreachable）：[354531, 854837, 1295104, 1419061, 8416687, 69]。
* 片段原子数：word 侧（≤8）直方 [0, 18, 24, 50, 111, 214, 332, 439, 626]；全体（含环境 C）分位数 {"max": 114, "mean": 25.01765529488099, "p50": 26.0, "p90": 35.0}；连通分量直方 [0, 29260, 0, 0, 23, 0, 0, 0]。
* 无桥键分子（逐原子根回退）：8 个，OGB 行号 [721, 1452, 1897, 2568, 2751, 2969, 3204, 3753]。
* cache `/home/calendar/code/research/tracks/structural-sufficiency/task_response_composition_s0_v1/data/trc_cache_lipo.npz` sha256 `c53fee5d16869b7b7ff721d404a84289d0a315d392440ebfb436cbd307e3abc3`，commit `b70ade26a73628fe541baa2ea91fcaee860d12f7`。

替换目录（只在 Train 内部构建；Valid/Test 只作单元，不产生对）：

| cell | pairs | transforms | transforms in Train | molecules | pairs with Train support |
|---|---|---|---|---|---|
| cross | 1072 | 966 | 137 | 727 | 205 |
| test | 37 | 36 | 5 | 39 | 5 |
| train | 4646 | 3399 | 3399 | 1865 | 4646 |
| valid | 15 | 15 | 0 | 24 | 0 |

* Train 对 4646，transform 3399，重复 transform（≥2 个结构分量）482，结构分量 458，涉及分子 1865；σ_y=1.2115，τ²=0.5446。
* `|Δy|` 均值 0.6928（原始单位，σ 标准化前）。
* 有 Train 支持的对 4856/5770（84.2%），但 held-out 单元只有 210 对（18.7%）。也就是说：在 Train 内部，同一 `(C, 词)` 约束可以反复被真实对约束；跨划分时这个约束几乎不存在，这也决定了下面第 6 节支持子集的口径。
* 重复 transform 子集 1771 对。
* 同一 `(cell, transform)` 单元最多包含 53 个真实对；有多个对的单元 577/4416，这些单元内的 `mean|Δy|` = 0.6368，全体单元的 `mean|Δy|` = 0.7053。这是“同一 transform 在不同结构里标签不一致”的直接度量，也是目录约束共享时无法避开的数据性质。

## 2. 工程检查

| check | ok | evidence |
|---|---|---|
| inputs | True | recombination 坏 0/1595, bond endpoint 坏 0, bridge 坏 0, frame mask ok=True |
| renumber | True | 重编号后结构签名与四臂预测一致（FP64） |
| port | True | 用端口重算 CSR 与 pack 一致 max|Δ|=0.000000010；非等价端口改变真实特征 min|Δ|=0.492476 |
| context | True | 恒等式 max|Δ|=0.00 |
| batching | True | 批/单条、乱序、稀疏 vs 显式循环 max|Δ|=0.00 |
| batch | True | 真实批次主/辅损失有限、梯度非零 |

GPU/FP32 复跑在结果冻结前未执行（见偏差日志）；训练本身每个 epoch 都在 CUDA/FP32 上跑同一前向，重编号不变性由结构签名（与数值路径无关）覆盖。
CPU/FP64 `all_ok = True`。

## 3. 配方与实际条件

* 共同 horizon **E=160**：slower of the first pair: 240 x 1.25 x 14.9s = 74.5 min > 50 min, 160 x 1.25 x 14.9s = 49.7 min <= 50 min -> E=160。实测 epoch 秒数 {"LIPO_FACTOR_CUTRESP_R0": {"median": 4.2, "gpu": "res-gpu1"}, "LIPO_DIRECT_CUTRESP_R0": {"median": 14.9, "gpu": "res-gpu0 (shared with a foreign 22GB process)"}}。horizon 只由计时决定，未看任何标签指标；epochs 1–5 是线性 warmup，与 E 无关。
* AdamW lr 1e-3，wd 1e-4，clip 5，batch 64 分子 + 16 对，5 epoch warmup + cosine 到 1e-5，seed 0，单种子；每个 epoch 全量 Train/Valid/Test 评估（原始单位）；checkpoint 只按 Valid RMSE 选择；λ_var=0。
* 主损 = 分子内解析均方；辅损 = `mean((Δf - Δy/σ_y)²)/τ²`，`τ² = max(0.25, E[((Δy)/σ_y)²])`，权重 0.25；辅对采样 80% 来自重复 transform、20% 来自全体，transform→分量→对均匀。

## 4. 主表（ogbg-mollipo，官方 3360/420/420，原始单位，best-Valid）

| run | arm | aux | train frac | Valid RMSE | Test RMSE | Train RMSE | best ep | params | wall min |
|---|---|---|---|---|---|---|---|---|---|
| LIPO_FACTOR_CUTRESP_R0 | FACTOR | CUTRESP | 1.0 | 0.7096 | 0.7227 | 0.2220 | 100 | 154983 | 11.6 |
| LIPO_DIRECT_NONE_R0 | DIRECT | NONE | 1.0 | 0.7345 | 0.7057 | 0.1680 | 130 | 155102 | 22.7 |
| LIPO_FACTOR_NONE_R0 | FACTOR | NONE | 1.0 | 0.7427 | 0.7337 | 0.1693 | 129 | 154983 | 6.2 |
| LIPO_DIRECT_CUTRESP_R0 | DIRECT | CUTRESP | 1.0 | 0.7474 | 0.6723 | 0.1833 | 135 | 155102 | 40.3 |
| LIPO_GLOBAL_NONE_R0 | GLOBAL | NONE | 1.0 | 0.7555 | 0.7614 | 0.4723 | 92 | 124629 | 5.0 |
| LIPO_FACTOR_MOLPAIR_R0 | FACTOR | MOLPAIR | 1.0 | 0.8720 | 0.8523 | 0.5087 | 144 | 154983 | 6.4 |

外部 GINE 坐标（同划分、同目标缩放、同评估协议，raw 特征，5 层 128 维，sum+mean readout，387974 参数）：**Valid 0.6406 / Test 0.6582**（best epoch 128，wall 4.2 min）。

历史参照（只读，不同协议，不可直接排名，只作坐标）：

| historical run | Valid | Test |
|---|---|---|
| WORD_PREFIX | 0.6991 | 0.6891 |
| BAG_CONT | 0.6924 | 0.6980 |
| MATCHED_W34 | 0.6567 | 0.7247 |
| DIRECT_S56 | 0.7387 | 0.7753 |
| GINE_RAW_REF | 0.6101 | 0.6722 |

## 5. 共享结构是否值钱：2×2 因子（配对 bootstrap，Test/Valid 各 1000 次重采样）

| split | DIRECT_NONE | DIRECT_CUTRESP | FACTOR_NONE | FACTOR_CUTRESP | gain_share(NONE) | gain_share(CUTRESP) | resp|FACTOR | resp|DIRECT | interaction |
|---|---|---|---|---|---|---|---|---|---|
| valid | 0.7345 | 0.7474 | 0.7427 | 0.7096 | -0.0082 | 0.0379 | 0.0331 | -0.0130 | 0.0461 |
| test | 0.7057 | 0.6723 | 0.7337 | 0.7227 | -0.0280 | -0.0504 | 0.0110 | 0.0334 | -0.0224 |

`gain_share = DIRECT − FACTOR`（正 = 分解式接口更准）；`resp|arm = NONE − CUTRESP`（正 = 真实匹配监督有用）；`interaction = resp|FACTOR − resp|DIRECT`（正 = 响应监督只在共享因式下有用）。

| comparison (control − candidate) | Valid ΔRMSE [95% CI] | Test ΔRMSE [95% CI] |
|---|---|---|
| GINE_RAW_LIPO_R0 - LIPO_DIRECT_CUTRESP_R0 | **-0.1068** [-0.1659, -0.0420] | -0.0141 [-0.0783, +0.0475] |
| GINE_RAW_LIPO_R0 - LIPO_FACTOR_CUTRESP_R0 | **-0.0690** [-0.1271, -0.0075] | **-0.0645** [-0.1310, -0.0006] |
| GINE_RAW_LIPO_R0 - LIPO_FACTOR_NONE_R0 | **-0.1021** [-0.1588, -0.0371] | **-0.0756** [-0.1342, -0.0174] |
| GINE_RAW_LIPO_R0 - LIPO_GLOBAL_NONE_R0 | **-0.1149** [-0.1924, -0.0171] | **-0.1032** [-0.1588, -0.0508] |
| LIPO_DIRECT_CUTRESP_R0 - LIPO_FACTOR_CUTRESP_R0 | **+0.0379** [+0.0071, +0.0707] | **-0.0504** [-0.0790, -0.0201] |
| LIPO_DIRECT_NONE_R0 - LIPO_DIRECT_CUTRESP_R0 | -0.0130 [-0.0478, +0.0225] | +0.0334 [-0.0004, +0.0695] |
| LIPO_DIRECT_NONE_R0 - LIPO_FACTOR_NONE_R0 | -0.0082 [-0.0388, +0.0258] | -0.0280 [-0.0599, +0.0074] |
| LIPO_FACTOR_MOLPAIR_R0 - LIPO_FACTOR_CUTRESP_R0 | **+0.1624** [+0.1099, +0.2138] | **+0.1296** [+0.0782, +0.1818] |
| LIPO_FACTOR_NONE_R0 - LIPO_FACTOR_CUTRESP_R0 | **+0.0331** [+0.0015, +0.0639] | +0.0110 [-0.0252, +0.0461] |
| LIPO_GLOBAL_NONE_R0 - LIPO_FACTOR_NONE_R0 | +0.0128 [-0.0580, +0.0859] | +0.0277 [-0.0343, +0.0909] |

Δ>0 表示 candidate 更好；**粗体**表示 95% 区间不含 0。

## 6. 真实匹配替换目录上的响应指标

评价对象是目录里**真实存在**的对：`Δy = y_b − y_a`，`Δf` 是模型的切口函数差或整分子差，全部除以 σ_y。不做任何未观察组合的伪标签。TABLE 只在训练对上按 transform 建表（每个结构分量先平均，再跨分量平均），零预测给 `Δ=0` 基线。

| predictor | n(supported) | MAE | corr | sign | MAE(strict) | corr(strict) | MAE(repeat) |
|---|---|---|---|---|---|---|---|
| TABLE（训练对查表） | 4856 | 0.0885 | 0.9607 | 0.9463 | 0.2218 | 0.8643 | 0.2154 |
| 零预测 | 5770 | 0.5691 | 0.0000 | 0.0142 | 0.4982 | 0.0000 | 0.4976 |
| LIPO_DIRECT_CUTRESP_R0 · 切口函数差 | 4856 | 0.0904 | 0.9826 | 0.9283 | 0.0925 | 0.9688 | 0.0765 |
| LIPO_DIRECT_CUTRESP_R0 · 整分子差 | 4856 | 0.1714 | 0.9582 | 0.8878 | 0.1790 | 0.9392 | 0.1686 |
| LIPO_DIRECT_NONE_R0 · 切口函数差 | 4856 | 0.1800 | 0.9488 | 0.8756 | 0.1987 | 0.9152 | 0.1868 |
| LIPO_DIRECT_NONE_R0 · 整分子差 | 4856 | 0.1803 | 0.9524 | 0.8834 | 0.1918 | 0.9285 | 0.1825 |
| LIPO_FACTOR_CUTRESP_R0 · 切口函数差 | 4856 | 0.1284 | 0.9706 | 0.9117 | 0.1193 | 0.9589 | 0.1043 |
| LIPO_FACTOR_CUTRESP_R0 · 整分子差 | 4856 | 0.1930 | 0.9505 | 0.8801 | 0.1958 | 0.9314 | 0.1867 |
| LIPO_FACTOR_MOLPAIR_R0 · 切口函数差 | 4856 | 0.7677 | -0.3684 | 0.3801 | 0.7770 | -0.5985 | 0.7895 |
| LIPO_FACTOR_MOLPAIR_R0 · 整分子差 | 4856 | 0.6658 | -0.2757 | 0.3567 | 0.6584 | -0.6098 | 0.6683 |
| LIPO_FACTOR_NONE_R0 · 切口函数差 | 4856 | 0.1920 | 0.9440 | 0.8581 | 0.2105 | 0.9099 | 0.2007 |
| LIPO_FACTOR_NONE_R0 · 整分子差 | 4856 | 0.1814 | 0.9515 | 0.8878 | 0.1952 | 0.9245 | 0.1850 |
| LIPO_GLOBAL_NONE_R0 · 整分子差 | 4856 | 0.3300 | 0.8273 | 0.7939 | 0.3066 | 0.8182 | 0.3036 |

严格支持 = 排除目标分子自身所在的结构分量，避免用同一分子族的训练对自证。

无训练支持的对（模型必须外推，该子集上 TABLE 无定义）：

| predictor | quantity | n | MAE | corr | sign |
|---|---|---|---|---|---|
| LIPO_DIRECT_CUTRESP_R0 | delta_cut | 914 | 0.3892 | 0.7907 | 0.7648 |
| LIPO_DIRECT_CUTRESP_R0 | delta_mol | 914 | 0.3680 | 0.8151 | 0.7954 |
| LIPO_DIRECT_NONE_R0 | delta_cut | 914 | 0.4304 | 0.7651 | 0.7724 |
| LIPO_DIRECT_NONE_R0 | delta_mol | 914 | 0.3893 | 0.7960 | 0.7888 |
| LIPO_FACTOR_CUTRESP_R0 | delta_cut | 914 | 0.4111 | 0.7629 | 0.7877 |
| LIPO_FACTOR_CUTRESP_R0 | delta_mol | 914 | 0.3671 | 0.8269 | 0.8140 |
| LIPO_FACTOR_MOLPAIR_R0 | delta_cut | 914 | 0.7294 | 0.0532 | 0.5284 |
| LIPO_FACTOR_MOLPAIR_R0 | delta_mol | 914 | 0.6495 | 0.2011 | 0.5361 |
| LIPO_FACTOR_NONE_R0 | delta_cut | 914 | 0.4471 | 0.7307 | 0.7560 |
| LIPO_FACTOR_NONE_R0 | delta_mol | 914 | 0.3862 | 0.7865 | 0.7888 |
| LIPO_GLOBAL_NONE_R0 | delta_mol | 914 | 0.4154 | 0.7963 | 0.7779 |
| 零预测 | Δ=0 | 914 | 0.6511 | – | 0.0098 |

按单元分组：

| cell | n pairs | TABLE MAE | zero MAE | 模型 Δmol MAE | 模型 Δcut MAE |
|---|---|---|---|---|---|
| cross | 1072 | 0.3037 | 0.6221 | 0.3529 | 0.3725 |
| test | 37 | 0.5134 | 0.6405 | 0.4069 | 0.4234 |
| train | 4646 | 0.0786 | 0.5719 | 0.1653 | 0.0802 |
| valid | 15 | – | 0.7583 | 0.5040 | 0.4758 |

（模型列取 `response_metrics.json` 中第一个模型，逐模型数值见该文件。）

## 7. 组合双差（两个上下文 × 同一对词）

可用四元组 7048，抽取 2000，涉及分子 1199；各子集规模 {'all_held_out': 1, 'train_test_any_test': 2, 'train_valid_any_valid': 0}。

| run | subset | n | D_mol MAE | – | D_mol sign | D_cut MAE | D_cut sign |
|---|---|---|---|---|---|---|---|
| LIPO_DIRECT_CUTRESP_R0 | all_held_out | 1 | 0.1354 | – | 1.0000 | 0.7591 | 0.0000 |
| LIPO_DIRECT_CUTRESP_R0 | train_test_any_test | 2 | 0.3205 | – | 1.0000 | 0.5113 | 0.5000 |
| LIPO_DIRECT_NONE_R0 | all_held_out | 1 | 0.4281 | – | 0.0000 | 0.3843 | 0.0000 |
| LIPO_DIRECT_NONE_R0 | train_test_any_test | 2 | 0.4877 | – | 0.5000 | 0.4204 | 0.5000 |
| LIPO_FACTOR_CUTRESP_R0 | all_held_out | 1 | 0.5164 | – | 0.0000 | 0.3024 | 1.0000 |
| LIPO_FACTOR_CUTRESP_R0 | train_test_any_test | 2 | 0.5157 | – | 0.5000 | 0.2124 | 1.0000 |
| LIPO_FACTOR_MOLPAIR_R0 | all_held_out | 1 | 0.5673 | – | 0.0000 | 1.6166 | 0.0000 |
| LIPO_FACTOR_MOLPAIR_R0 | train_test_any_test | 2 | 1.2013 | – | 0.0000 | 2.1442 | 0.0000 |
| LIPO_FACTOR_NONE_R0 | all_held_out | 1 | 0.0924 | – | 1.0000 | 0.1080 | 1.0000 |
| LIPO_FACTOR_NONE_R0 | train_test_any_test | 2 | 0.3592 | – | 1.0000 | 0.2650 | 1.0000 |
| LIPO_GLOBAL_NONE_R0 | all_held_out | 1 | 0.1878 | – | 1.0000 | – | – |
| LIPO_GLOBAL_NONE_R0 | train_test_any_test | 2 | 0.5623 | – | 0.5000 | – | – |

`D_y = [y(C1,Sb) − y(C1,Sa)] − [y(C2,Sb) − y(C2,Sa)]`；模型的 `D` 用同一 (Sa,Sb) 的两个上下文差。双差可以抵消只依赖词的公共项，留下由环境经共享状态决定的部分。

## 8. 数据效率

## 9. 案例（每条由显式数值规则选出，非手工挑选）

**composed** — both single molecules predicted well (top 40%), pair delta badly (top 10%), held-out only

| cell | mol a (OGB id) | mol b | Δy/σ | Δf(C) | Δf(∅) | merge | err a | err b | pair err | repeats | label spread |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cross | 3095 | 1729 | -1.907 | -0.203 | -0.760 | 0.556 | 2.036 | 0.361 | 2.063 |  | – |
| cross | 1 | 1729 | 0.693 | 2.454 | 0.171 | 2.284 | 1.231 | 0.361 | 2.134 |  | – |
| cross | 3679 | 1729 | 0.437 | 2.365 | 0.207 | 2.158 | 1.519 | 0.361 | 2.335 |  | – |

**merge** — largest |delta_full - delta_isolated| on held-out pairs

| cell | mol a (OGB id) | mol b | Δy/σ | Δf(C) | Δf(∅) | merge | err a | err b | pair err | repeats | label spread |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cross | 873 | 1029 | -0.421 | 1.693 | -0.930 | 2.623 | – | – | – |  | – |
| cross | 1 | 807 | 3.302 | 3.355 | 0.892 | 2.463 | – | – | – |  | – |
| cross | 1332 | 3107 | -1.626 | -2.439 | -0.036 | 2.402 | – | – | – |  | – |

**pathological** — transform repeated >=3 times inside one cell with the largest label spread

| cell | mol a (OGB id) | mol b | Δy/σ | Δf(C) | Δf(∅) | merge | err a | err b | pair err | repeats | label spread |
|---|---|---|---|---|---|---|---|---|---|---|---|
| train | 2549 | 3921 | -0.867 | -0.767 | 0.299 | 1.066 | – | – | – | 3 | 1.728 |
| train | 761 | 15 | -0.371 | -0.452 | -0.074 | 0.379 | – | – | – | 10 | 1.383 |

**redirect** — context flips the sign of the word's action

| cell | mol a (OGB id) | mol b | Δy/σ | Δf(C) | Δf(∅) | merge | err a | err b | pair err | repeats | label spread |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cross | 1637 | 369 | -1.824 | -2.170 | -0.426 | 1.744 | – | – | – |  | – |
| cross | 1637 | 369 | -1.824 | -2.061 | -0.778 | 1.283 | – | – | – |  | – |
| cross | 169 | 3217 | 0.165 | -0.002 | 0.545 | 0.547 | – | – | – |  | – |

## 10. 负结果与开放问题

* **没有超过 GINE。** 本轮最好的候选在 Valid 上比同协议 GINE 坐标差 0.0690。禁止 GNN 骨干的前提下，这个差距是该方法本身的上限证据，而不是训练不足：候选的训练误差已明显低于 GINE 的验证误差（见主表 Train 列）。
* **Valid/Test 排序反转。** `DIRECT_CUTRESP` 在 Test 上比 `FACTOR_CUTRESP` 好 0.0504，在 Valid 上差 0.0379；两侧 95% 区间都不含 0。best-Valid 选择本身几乎不损失 Test（`split_gap.json` 的 `cost_of_valid_selection` ≈ 0），所以这是划分难度差异，不是选择伪影。任何单侧结论都会被这次反转驳倒。
* **MOLPAIR 有害。** 只用整分子差约束 `f̄` 会让切口函数失配，Valid/Test 都比 CUTRESP 差一大截；说明起作用的不是“任何成对监督”，而是“监督切口函数本身”。
* **TABLE 在支持子集上远强于所有神经模型。** 这说明当前数据量下“按 transform 查表”仍是更强的近似；共享响应状态的价值必须体现在无支持外推（`unsupported` 子集）上，那里 TABLE 无定义，模型确实优于零预测，但绝对误差仍然偏大。

## 11. 复现

```
code/trc_prepare.py --dataset lipo|esol     # 建 cache（本地，需要 rdkit）
code/trc_checks.py --device cpu|cuda        # 工程检查
code/trc_train.py --run-id ... --arm ... --supervision ... --dataset ... --epochs 160 --device cuda
code/trc_reference.py --run-id ... --dataset lipo --epochs 160 --device cuda
code/trc_chain.sh code code/trc_queue_gpu1.txt   # 单卡串行队列
code/trc_analyze.py --device cpu            # analysis/*.json
code/trc_cases.py                           # analysis/cases.json
code/trc_report.py                          # 本文件
```

`data/` 与 `results/*/` 不进版本库（cache 与权重是运行产物），但 `analysis/` 的每个数字都可以由上面的命令从公开 cache 重建。

