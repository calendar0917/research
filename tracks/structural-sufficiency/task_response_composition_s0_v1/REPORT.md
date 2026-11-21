# task_response_composition_s0_v1 — 结构响应组合 s0 报告

八小时轮次（T0 2026-11-21T23:33:19+08:00）的完整结果。所有数字来自 `analysis/*.json`；本文件由 `code/trc_report.py` 从这些记录生成，不重算指标。

## 0. 结论摘要

1. **整分子预测没有超过外部 GINE 坐标。** 最好的候选是 `LIPO_FACTOR_CUTRESP_R0`，Valid 0.7096 / Test 0.7227，同一协议下的 GINE 坐标是 Valid 0.6406 / Test 0.6582。Valid 上 GINE 显著更好（配对 bootstrap 差值 **-0.0690** [-0.1271, -0.0075]）。
2. **但在 Test 上，`DIRECT_CUTRESP` 与 GINE 坐标在统计上不可区分**：Test RMSE 0.6723 vs 0.6582，配对 bootstrap 差值 -0.0141 [-0.0783, +0.0475]（跨 0）。Valid 上则明显更差。同样的反转在第 5 节出现，说明单看一侧会得出相反结论。

3. **外部 GINE 与历史 GINE 参考一致。** 本轮 raw-GINE 坐标 0.6406/0.6582，历史 `GINE_RAW_REF` 0.6101/0.6722（历史为 240 epoch + early stop，本轮统一 E=160），说明坐标可比。

4. **交互是必需的，但它的具体形式无关。** 组合双差上纯加性臂 `ADD` 恒等于 0（结构上无法表达“环境改变词的作用”），而 `FACTOR`/`DIRECT` 的切口双差显著优于零基线（sign 0.722/0.760 vs 0.550）；同时 `FACTOR` 相对 `ADD` 的整分子差值是 Valid 0.0181、Test -0.0005，都跨 0。所以“必需交互”成立，“必须低秩可解释状态”不成立。

5. **真实匹配监督在整分子指标上收益很小，在组合维度上收益清楚。** `FACTOR` 由 `NONE` 到 `CUTRESP`：Valid +0.033（显著）、Test +0.011（不显著），整分子层面几乎不可见；但双差 sign 0.666 → 0.722、切口函数差在严格支持子集上的 MAE 0.211 → 0.119。`MOLPAIR`（只约束整分子差）则在两个维度上都明显有害。

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
* cache `/home/calendar/code/research/tracks/structural-sufficiency/task_response_composition_s0_v1/data/trc_cache_lipo.npz` sha256 `c53fee5d16869b7b7ff721d404a84289d0a315d392440ebfb436cbd307e3abc3`，commit `8d254fddc10416eb49ccb8ce0af8852b4b3d8266`。

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

GPU/FP32 复跑（`--skip renumber`：该子项需要在算力机上重建 mini pack，而远端 venv 没有 rdkit，见偏差日志）：`all_ok = True`。

| check | ok (cuda/fp32) | evidence |
|---|---|---|
| inputs | True | {"frames_edge_mask_ok": true, "local_edge_index_bad": 0, "parse_bond_endpoint_bad": 0, "parse_bridge_bad": 0, "parse_recombination_bad": 0} |
| context | True | {"ctx_feature_max_abs": 2.86102294921875e-06, "ctx_feature_selfconsistent": true} |
| port | True | {"max_port_change_diff": 3.332204510175204, "min_port_change_diff": 0.49247648509779385, "reconstruction_max_abs": 9.934107481068821e-09} |
| batching | True | {"batch_vs_single_max_abs": 1.4901161193847656e-08, "permuted_batch_max_abs": 2.9802322387695312e-08, "reversed_batch_max_abs": 1.4901161193847656e-08, "sparse_vs_explicit_max_abs": 1.5742485128811268e-06} |
| batch | True | {} |

CPU/FP64 `all_ok = True`。

## 3. 配方与实际条件

* 共同 horizon **E=160**：slower of the first pair: 240 x 1.25 x 14.9s = 74.5 min > 50 min, 160 x 1.25 x 14.9s = 49.7 min <= 50 min -> E=160。实测 epoch 秒数 {"LIPO_FACTOR_CUTRESP_R0": {"median": 4.2, "gpu": "res-gpu1"}, "LIPO_DIRECT_CUTRESP_R0": {"median": 14.9, "gpu": "res-gpu0", "note": "cost is the parse-level evaluation of 33438 parses x 3 splits, not card contention (the GINE reference on the same card ran at 1.6 s/epoch)"}}。horizon 只由计时决定，未看任何标签指标；epochs 1–5 是线性 warmup，与 E 无关。
* AdamW lr 1e-3，wd 1e-4，clip 5，batch 64 分子 + 16 对，5 epoch warmup + cosine 到 1e-5，seed 0，单种子；每个 epoch 全量 Train/Valid/Test 评估（原始单位）；checkpoint 只按 Valid RMSE 选择；λ_var=0。
* 主损 = 分子内解析均方；辅损 = `mean((Δf - Δy/σ_y)²)/τ²`，`τ² = max(0.25, E[((Δy)/σ_y)²])`，权重 0.25；辅对采样 80% 来自重复 transform、20% 来自全体，transform→分量→对均匀。

## 4. 主表（ogbg-mollipo，官方 3360/420/420，原始单位，best-Valid）

| run | arm | aux | train frac | Valid RMSE | Test RMSE | Train RMSE | best ep | params | wall min |
|---|---|---|---|---|---|---|---|---|---|
| LIPO_FACTOR_CUTRESP_R0 | FACTOR | CUTRESP | 1.0 | 0.7096 | 0.7227 | 0.2220 | 100 | 154983 | 11.6 |
| LIPO_ADD_CUTRESP_R0 | ADD | CUTRESP | 1.0 | 0.7276 | 0.7221 | 0.2580 | 109 | 153831 | 11.1 |
| LIPO_DIRECT_NONE_R0 | DIRECT | NONE | 1.0 | 0.7345 | 0.7057 | 0.1680 | 130 | 155102 | 22.7 |
| LIPO_FACTOR_NONE_R0 | FACTOR | NONE | 1.0 | 0.7427 | 0.7337 | 0.1693 | 129 | 154983 | 6.2 |
| LIPO_DIRECT_CUTRESP_R0 | DIRECT | CUTRESP | 1.0 | 0.7474 | 0.6723 | 0.1833 | 135 | 155102 | 40.3 |
| LIPO_GLOBAL_NONE_R0 | GLOBAL | NONE | 1.0 | 0.7555 | 0.7614 | 0.4723 | 92 | 124629 | 5.0 |
| LIPO_DIRECT_CUTRESP_F50_R0 | DIRECT | CUTRESP | 0.5 | 0.8370 | 0.8055 | 0.1875 | 112 | 155102 | 20.7 |
| LIPO_FACTOR_CUTRESP_F50_R0 | FACTOR | CUTRESP | 0.5 | 0.8544 | 0.7854 | 0.2377 | 98 | 154983 | 5.9 |
| LIPO_FACTOR_MOLPAIR_R0 | FACTOR | MOLPAIR | 1.0 | 0.8720 | 0.8523 | 0.5087 | 144 | 154983 | 6.4 |
| LIPO_FACTOR_CUTRESP_F25_R0 | FACTOR | CUTRESP | 0.25 | 0.9571 | 0.9748 | 0.2354 | 97 | 154983 | 3.1 |
| LIPO_DIRECT_CUTRESP_F25_R0 | DIRECT | CUTRESP | 0.25 | 0.9940 | 0.9591 | 0.2865 | 97 | 155102 | 10.8 |

外部 GINE 坐标（同划分、同目标缩放、同评估协议，raw 特征，5 层 128 维，sum+mean readout，387974 参数）：**Valid 0.6406 / Test 0.6582**（best epoch 128，wall 4.2 min）。

历史参照（只读，不同协议，不可直接排名，只作坐标）：

| historical run | Valid | Test |
|---|---|---|
| WORD_PREFIX | 0.6991 | 0.6891 |
| BAG_CONT | 0.6924 | 0.6980 |
| MATCHED_W34 | 0.6567 | 0.7247 |
| DIRECT_S56 | 0.7387 | 0.7753 |
| GINE_RAW_REF | 0.6101 | 0.6722 |

* 已用计算：11 个 lipo 单元 + 4 个 ESOL 单元 + 2 个 GINE 坐标，合计 **2.67 GPU·h**（16 GPU·h 预算内），队列上限 20 个正式单元未用满。这个数只累加每个单元自己的 wall time，不含两次作废的重跑和排队空转。

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
| LIPO_ADD_CUTRESP_R0 - LIPO_FACTOR_CUTRESP_R0 | +0.0181 [-0.0152, +0.0519] | -0.0005 [-0.0394, +0.0387] |
| LIPO_DIRECT_CUTRESP_R0 - LIPO_FACTOR_CUTRESP_R0 | **+0.0379** [+0.0071, +0.0707] | **-0.0504** [-0.0790, -0.0201] |
| LIPO_DIRECT_NONE_R0 - LIPO_DIRECT_CUTRESP_R0 | -0.0130 [-0.0478, +0.0225] | +0.0334 [-0.0004, +0.0695] |
| LIPO_DIRECT_NONE_R0 - LIPO_FACTOR_NONE_R0 | -0.0082 [-0.0388, +0.0258] | -0.0280 [-0.0599, +0.0074] |
| LIPO_FACTOR_MOLPAIR_R0 - LIPO_FACTOR_CUTRESP_R0 | **+0.1624** [+0.1099, +0.2138] | **+0.1296** [+0.0782, +0.1818] |
| LIPO_FACTOR_NONE_R0 - LIPO_FACTOR_CUTRESP_R0 | **+0.0331** [+0.0015, +0.0639] | +0.0110 [-0.0252, +0.0461] |
| LIPO_GLOBAL_NONE_R0 - LIPO_FACTOR_NONE_R0 | +0.0128 [-0.0580, +0.0859] | +0.0277 [-0.0343, +0.0909] |
| GINE_RAW_ESOL_R0 - ESOL_DIRECT_CUTRESP_R0 | -0.0030 [-0.1442, +0.1263] | **-0.1163** [-0.2141, -0.0133] |
| GINE_RAW_ESOL_R0 - ESOL_FACTOR_CUTRESP_R0 | +0.0301 [-0.0968, +0.1515] | -0.0744 [-0.1724, +0.0336] |
| GINE_RAW_ESOL_R0 - ESOL_FACTOR_NONE_R0 | +0.0260 [-0.1107, +0.1591] | -0.0562 [-0.1470, +0.0374] |
| GINE_RAW_ESOL_R0 - ESOL_GLOBAL_NONE_R0 | -0.0622 [-0.2069, +0.0915] | **-0.1355** [-0.2369, -0.0289] |

Δ>0 表示 candidate 更好；**粗体**表示 95% 区间不含 0。

## 6. 真实匹配替换目录上的响应指标

评价对象是目录里**真实存在**的对：`Δy = y_b − y_a`，`Δf` 是模型的切口函数差或整分子差，全部除以 σ_y。不做任何未观察组合的伪标签。TABLE 只在训练对上按 transform 建表（每个结构分量先平均，再跨分量平均），零预测给 `Δ=0` 基线。

| predictor | n(supported) | MAE | corr | sign | MAE(strict) | corr(strict) | MAE(repeat) |
|---|---|---|---|---|---|---|---|
| TABLE（训练对查表） | 4856 | 0.0885 | 0.9607 | 0.9463 | 0.2218 | 0.8643 | 0.2154 |
| 零预测 | 5770 | 0.5691 | 0.0000 | 0.0142 | 0.4982 | 0.0000 | 0.4976 |
| LIPO_ADD_CUTRESP_R0 · 切口函数差 | 4856 | 0.2590 | 0.8766 | 0.8451 | 0.2509 | 0.8364 | 0.2500 |
| LIPO_ADD_CUTRESP_R0 · 整分子差 | 4856 | 0.2365 | 0.9192 | 0.8556 | 0.2279 | 0.8995 | 0.2229 |
| LIPO_DIRECT_CUTRESP_F25_R0 · 切口函数差 | 4856 | 0.4888 | 0.5453 | 0.6763 | 0.4194 | 0.5665 | 0.4155 |
| LIPO_DIRECT_CUTRESP_F25_R0 · 整分子差 | 4856 | 0.4516 | 0.6058 | 0.6921 | 0.4006 | 0.5845 | 0.3978 |
| LIPO_DIRECT_CUTRESP_F50_R0 · 切口函数差 | 4856 | 0.3269 | 0.7825 | 0.7858 | 0.2857 | 0.7788 | 0.2795 |
| LIPO_DIRECT_CUTRESP_F50_R0 · 整分子差 | 4856 | 0.3319 | 0.7966 | 0.7924 | 0.3110 | 0.7727 | 0.3084 |
| LIPO_DIRECT_CUTRESP_R0 · 切口函数差 | 4856 | 0.0904 | 0.9826 | 0.9283 | 0.0925 | 0.9688 | 0.0765 |
| LIPO_DIRECT_CUTRESP_R0 · 整分子差 | 4856 | 0.1714 | 0.9582 | 0.8878 | 0.1790 | 0.9392 | 0.1686 |
| LIPO_DIRECT_NONE_R0 · 切口函数差 | 4856 | 0.1800 | 0.9488 | 0.8756 | 0.1987 | 0.9152 | 0.1868 |
| LIPO_DIRECT_NONE_R0 · 整分子差 | 4856 | 0.1803 | 0.9524 | 0.8834 | 0.1918 | 0.9285 | 0.1825 |
| LIPO_FACTOR_CUTRESP_F25_R0 · 切口函数差 | 4856 | 0.5047 | 0.5348 | 0.6596 | 0.4362 | 0.5348 | 0.4342 |
| LIPO_FACTOR_CUTRESP_F25_R0 · 整分子差 | 4856 | 0.4527 | 0.6028 | 0.6870 | 0.3989 | 0.5950 | 0.3963 |
| LIPO_FACTOR_CUTRESP_F50_R0 · 切口函数差 | 4856 | 0.3426 | 0.7650 | 0.7881 | 0.2945 | 0.7681 | 0.2856 |
| LIPO_FACTOR_CUTRESP_F50_R0 · 整分子差 | 4856 | 0.3437 | 0.7769 | 0.8019 | 0.3128 | 0.7671 | 0.3091 |
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
| LIPO_ADD_CUTRESP_R0 | delta_cut | 914 | 0.4111 | 0.7802 | 0.7932 |
| LIPO_ADD_CUTRESP_R0 | delta_mol | 914 | 0.3693 | 0.8195 | 0.8118 |
| LIPO_DIRECT_CUTRESP_F25_R0 | delta_cut | 914 | 0.6788 | 0.3461 | 0.5963 |
| LIPO_DIRECT_CUTRESP_F25_R0 | delta_mol | 914 | 0.5973 | 0.4289 | 0.6171 |
| LIPO_DIRECT_CUTRESP_F50_R0 | delta_cut | 914 | 0.4729 | 0.6735 | 0.7199 |
| LIPO_DIRECT_CUTRESP_F50_R0 | delta_mol | 914 | 0.4578 | 0.6830 | 0.7407 |
| LIPO_DIRECT_CUTRESP_R0 | delta_cut | 914 | 0.3892 | 0.7907 | 0.7648 |
| LIPO_DIRECT_CUTRESP_R0 | delta_mol | 914 | 0.3680 | 0.8151 | 0.7954 |
| LIPO_DIRECT_NONE_R0 | delta_cut | 914 | 0.4304 | 0.7651 | 0.7724 |
| LIPO_DIRECT_NONE_R0 | delta_mol | 914 | 0.3893 | 0.7960 | 0.7888 |
| LIPO_FACTOR_CUTRESP_F25_R0 | delta_cut | 914 | 0.6855 | 0.3287 | 0.6149 |
| LIPO_FACTOR_CUTRESP_F25_R0 | delta_mol | 914 | 0.5900 | 0.4423 | 0.6532 |
| LIPO_FACTOR_CUTRESP_F50_R0 | delta_cut | 914 | 0.5376 | 0.5786 | 0.7090 |
| LIPO_FACTOR_CUTRESP_F50_R0 | delta_mol | 914 | 0.4903 | 0.6394 | 0.7429 |
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
| cross | 1072 | 0.3037 | 0.6221 | 0.3548 | 0.3858 |
| test | 37 | 0.5134 | 0.6405 | 0.3549 | 0.3599 |
| train | 4646 | 0.0786 | 0.5719 | 0.2334 | 0.2578 |
| valid | 15 | – | 0.7583 | 0.5323 | 0.5739 |

（模型列取 `response_metrics.json` 中第一个模型，逐模型数值见该文件。）

切口函数的内部一致性（best-Valid epoch 上的全量评估）：

| run | Train cut − mol | cut var (Train) | cut var (Valid) | cut RMSE (Train) | cut RMSE (Valid) | mol RMSE (Valid) |
|---|---|---|---|---|---|---|
| LIPO_ADD_CUTRESP_R0 | 0.1061 | 0.0371 | 0.1584 | 0.3641 | 0.9245 | 0.7276 |
| LIPO_DIRECT_CUTRESP_F25_R0 | 0.1187 | 0.0420 | 0.2048 | 0.4052 | 1.1507 | 0.9940 |
| LIPO_DIRECT_CUTRESP_F50_R0 | 0.0955 | 0.0222 | 0.1600 | 0.2830 | 1.0188 | 0.8370 |
| LIPO_DIRECT_CUTRESP_R0 | 0.0738 | 0.0161 | 0.1199 | 0.2571 | 0.8963 | 0.7474 |
| LIPO_DIRECT_NONE_R0 | 0.0647 | 0.0123 | 0.1252 | 0.2326 | 0.8780 | 0.7345 |
| LIPO_FACTOR_CUTRESP_F25_R0 | 0.1068 | 0.0331 | 0.2013 | 0.3422 | 1.1189 | 0.9571 |
| LIPO_FACTOR_CUTRESP_F50_R0 | 0.0988 | 0.0284 | 0.1571 | 0.3365 | 1.0243 | 0.8544 |
| LIPO_FACTOR_CUTRESP_R0 | 0.0836 | 0.0227 | 0.1127 | 0.3056 | 0.8501 | 0.7096 |
| LIPO_FACTOR_MOLPAIR_R0 | 0.0748 | 0.0271 | 0.1477 | 0.5834 | 1.0136 | 0.8720 |
| LIPO_FACTOR_NONE_R0 | 0.0642 | 0.0121 | 0.1283 | 0.2335 | 0.8936 | 0.7427 |
| LIPO_GLOBAL_NONE_R0 | – | – | – | – | – | 0.7555 |

`cut var` 是同一分子内部各解析切口分数 `f_c` 的方差均值——它衡量模型是否真的把“不同接口有不同作用”表示出来了，而不是把所有解析压到同一个整分子值上。`Train cut − mol` 是切口级训练误差与整分子训练误差之差，反映主损（分子内平均后平方）与切口级拟合之间的差距。

机制上可以读到一件事：在同一个 `FACTOR` 骨架上，`CUTRESP` 把 Train 内部的切口分数离散度从 0.012 抬到 0.023，同时把切口级 Train RMSE 从 0.234 抬到 0.306，却把整分子 Valid RMSE 从 0.743 降到 0.710。真实匹配监督并没有让模型更贴合Train 的切口标签，而是把它推向“同一分子的不同接口给出不同值、但均值更准”的表示——这正是接口分解应有的行为。

## 7. 组合双差（两个上下文 × 同一对词）

可用四元组 7048，抽取 2000，涉及分子 1199；各子集规模 {'all_held_out': 1, 'train_or_cross': 1996, 'train_test_any_test': 2, 'train_train': 1835, 'train_valid_any_valid': 0}。

| run | subset | n | D_mol MAE | D_mol sign | D_cut MAE | D_cut sign | std(D_cut) |
|---|---|---|---|---|---|---|---|
| LIPO_ADD_CUTRESP_R0 | all_held_out | 1 | n<10，不解释 | – | – | – | – |
| LIPO_ADD_CUTRESP_R0 | train_or_cross | 1996 | 0.3572 | 0.6398 | 0.4059 | D≡0（加性臂上下文精确抵消） | 0.00000008 |
| LIPO_ADD_CUTRESP_R0 | train_test_any_test | 2 | n<10，不解释 | – | – | – | – |
| LIPO_ADD_CUTRESP_R0 | train_train | 1835 | 0.3531 | 0.6436 | 0.4041 | D≡0（加性臂上下文精确抵消） | 0.00000008 |
| LIPO_DIRECT_CUTRESP_F25_R0 | all_held_out | 1 | n<10，不解释 | – | – | – | – |
| LIPO_DIRECT_CUTRESP_F25_R0 | train_or_cross | 1996 | 0.4532 | 0.5276 | 0.5079 | 0.5256 | 0.41277325 |
| LIPO_DIRECT_CUTRESP_F25_R0 | train_test_any_test | 2 | n<10，不解释 | – | – | – | – |
| LIPO_DIRECT_CUTRESP_F25_R0 | train_train | 1835 | 0.4520 | 0.5275 | 0.4994 | 0.5302 | 0.40256745 |
| LIPO_DIRECT_CUTRESP_F50_R0 | all_held_out | 1 | n<10，不解释 | – | – | – | – |
| LIPO_DIRECT_CUTRESP_F50_R0 | train_or_cross | 1996 | 0.4047 | 0.5656 | 0.4370 | 0.5877 | 0.42080287 |
| LIPO_DIRECT_CUTRESP_F50_R0 | train_test_any_test | 2 | n<10，不解释 | – | – | – | – |
| LIPO_DIRECT_CUTRESP_F50_R0 | train_train | 1835 | 0.4024 | 0.5673 | 0.4384 | 0.5837 | 0.42971917 |
| LIPO_DIRECT_CUTRESP_R0 | all_held_out | 1 | n<10，不解释 | – | – | – | – |
| LIPO_DIRECT_CUTRESP_R0 | train_or_cross | 1996 | 0.3020 | 0.7199 | 0.2489 | 0.7500 | 0.42791192 |
| LIPO_DIRECT_CUTRESP_R0 | train_test_any_test | 2 | n<10，不解释 | – | – | – | – |
| LIPO_DIRECT_CUTRESP_R0 | train_train | 1835 | 0.2955 | 0.7264 | 0.2359 | 0.7597 | 0.42915702 |
| LIPO_DIRECT_NONE_R0 | all_held_out | 1 | n<10，不解释 | – | – | – | – |
| LIPO_DIRECT_NONE_R0 | train_or_cross | 1996 | 0.3189 | 0.6804 | 0.3348 | 0.6558 | 0.36876802 |
| LIPO_DIRECT_NONE_R0 | train_test_any_test | 2 | n<10，不解释 | – | – | – | – |
| LIPO_DIRECT_NONE_R0 | train_train | 1835 | 0.3140 | 0.6812 | 0.3258 | 0.6578 | 0.36843580 |
| LIPO_FACTOR_CUTRESP_F25_R0 | all_held_out | 1 | n<10，不解释 | – | – | – | – |
| LIPO_FACTOR_CUTRESP_F25_R0 | train_or_cross | 1996 | 0.4485 | 0.5095 | 0.4909 | 0.5421 | 0.43172369 |
| LIPO_FACTOR_CUTRESP_F25_R0 | train_test_any_test | 2 | n<10，不解释 | – | – | – | – |
| LIPO_FACTOR_CUTRESP_F25_R0 | train_train | 1835 | 0.4446 | 0.5144 | 0.4810 | 0.5411 | 0.41691034 |
| LIPO_FACTOR_CUTRESP_F50_R0 | all_held_out | 1 | n<10，不解释 | – | – | – | – |
| LIPO_FACTOR_CUTRESP_F50_R0 | train_or_cross | 1996 | 0.4060 | 0.5741 | 0.4254 | 0.5616 | 0.35649400 |
| LIPO_FACTOR_CUTRESP_F50_R0 | train_test_any_test | 2 | n<10，不解释 | – | – | – | – |
| LIPO_FACTOR_CUTRESP_F50_R0 | train_train | 1835 | 0.4048 | 0.5755 | 0.4274 | 0.5531 | 0.35500900 |
| LIPO_FACTOR_CUTRESP_R0 | all_held_out | 1 | n<10，不解释 | – | – | – | – |
| LIPO_FACTOR_CUTRESP_R0 | train_or_cross | 1996 | 0.3145 | 0.7044 | 0.2694 | 0.7169 | 0.41123337 |
| LIPO_FACTOR_CUTRESP_R0 | train_test_any_test | 2 | n<10，不解释 | – | – | – | – |
| LIPO_FACTOR_CUTRESP_R0 | train_train | 1835 | 0.3100 | 0.7101 | 0.2580 | 0.7215 | 0.40945326 |
| LIPO_FACTOR_MOLPAIR_R0 | all_held_out | 1 | n<10，不解释 | – | – | – | – |
| LIPO_FACTOR_MOLPAIR_R0 | train_or_cross | 1996 | 0.5165 | 0.4213 | 0.6323 | 0.4289 | 0.54309647 |
| LIPO_FACTOR_MOLPAIR_R0 | train_test_any_test | 2 | n<10，不解释 | – | – | – | – |
| LIPO_FACTOR_MOLPAIR_R0 | train_train | 1835 | 0.5161 | 0.4163 | 0.6365 | 0.4234 | 0.54787104 |
| LIPO_FACTOR_NONE_R0 | all_held_out | 1 | n<10，不解释 | – | – | – | – |
| LIPO_FACTOR_NONE_R0 | train_or_cross | 1996 | 0.3206 | 0.6804 | 0.3387 | 0.6598 | 0.39547549 |
| LIPO_FACTOR_NONE_R0 | train_test_any_test | 2 | n<10，不解释 | – | – | – | – |
| LIPO_FACTOR_NONE_R0 | train_train | 1835 | 0.3142 | 0.6845 | 0.3290 | 0.6665 | 0.38798211 |
| LIPO_GLOBAL_NONE_R0 | all_held_out | 1 | n<10，不解释 | – | – | – | – |
| LIPO_GLOBAL_NONE_R0 | train_or_cross | 1996 | 0.3963 | 0.5511 | – | – | – |
| LIPO_GLOBAL_NONE_R0 | train_test_any_test | 2 | n<10，不解释 | – | – | – | – |
| LIPO_GLOBAL_NONE_R0 | train_train | 1835 | 0.3932 | 0.5499 | – | – | – |

**这个检验在本数据集上基本做不了，这是结论而不是缺陷。** 四元组要求同一对词出现在两个不同上下文里，而 Valid 只有 15 对、Test 只有 37 对，所以 held-out 双差只有 1–2 个样本，任何数字都没有统计意义；有意义的只有 `train_train`（同一划分内部）与 `train_or_cross`（Train 与重组单元之间）。下表的 held-out 行留着是为了说明这一点，不是用来支持/否证假设的。

`D_y = [y(C1,Sb) − y(C1,Sa)] − [y(C2,Sb) − y(C2,Sa)]`；模型的 `D` 用同一 (Sa,Sb) 的两个上下文差。双差抵消只依赖词的公共项，只留下“环境如何改变这一对词的作用”。

**这张表是本轮最干净的一条机制证据。** 在 `train_train` 子集（1835 个四元组）上：

* `ADD` 的 `D_cut` **恒等于 0**（std = 8e-08，纯浮点残差）：因为加性头`a(C)+a(S)+b(e)` 在双差里把上下文精确抵消。它的 `D_cut` MAE = 0.404 正好等于目标本身的 `mean|D_y|`，也就是零基线。换句话说，**纯加性臂在结构上就无法表达“环境改变词的作用”**；它在整分子 RMSE 上与 `FACTOR` 打平这件事，不能用来论证“组合效应不重要”。
* `DIRECT_CUTRESP`（D_cut MAE 0.236 / sign 0.760）与 `FACTOR_CUTRESP`（0.258 / 0.722）都显著优于零基线（0.404）与 `GLOBAL` 的整分子双差（0.393 / 0.550），说明共享接口状态确实承载了环境相关的作用。
* 真实匹配监督在这个诊断上有单独贡献：`FACTOR` 0.666 → 0.722（NONE → CUTRESP）、`DIRECT` 0.658 → 0.760；而 `MOLPAIR` 把它压到 0.423。这条增益在整分子 RMSE 上看不见，只在“组合”这一维度上可见。
* 训练分子数减半后这条信号明显退化（`FACTOR` sign 0.722 → 0.553 @50% → 0.541 @25%），而零基线不变——所以双差能力依赖样本量，且比整分子 RMSE 更早饱和。

## 8. 数据效率

ESOL（1128 分子，官方 902/112/114；原始单位，best-Valid）：

| run | arm | aux | train frac | Valid RMSE | Test RMSE | Train RMSE | best ep | params | wall min |
|---|---|---|---|---|---|---|---|---|---|
| ESOL_FACTOR_CUTRESP_R0 | FACTOR | CUTRESP | 1.0 | 0.8194 | 0.9095 | 0.2755 | 114 | 154983 | 2.1 |
| ESOL_FACTOR_NONE_R0 | FACTOR | NONE | 1.0 | 0.8235 | 0.8913 | 0.2661 | 126 | 154983 | 1.4 |
| ESOL_DIRECT_CUTRESP_R0 | DIRECT | CUTRESP | 1.0 | 0.8525 | 0.9514 | 0.3476 | 84 | 155102 | 7.0 |
| ESOL_GLOBAL_NONE_R0 | GLOBAL | NONE | 1.0 | 0.9117 | 0.9706 | 0.4702 | 95 | 124629 | 0.9 |

ESOL 的 raw-GINE 坐标：Valid 0.8495 / Test 0.8351。

ogbg-mollipo 小样本（嵌套 sha256 子集：25% ⊂ 50% ⊂ 100%，各臂共用同一子集，同一 E=160）：

| run | arm | aux | train frac | Valid RMSE | Test RMSE | Train RMSE | best ep | params | wall min |
|---|---|---|---|---|---|---|---|---|---|
| LIPO_DIRECT_CUTRESP_F50_R0 | DIRECT | CUTRESP | 0.5 | 0.8370 | 0.8055 | 0.1875 | 112 | 155102 | 20.7 |
| LIPO_FACTOR_CUTRESP_F50_R0 | FACTOR | CUTRESP | 0.5 | 0.8544 | 0.7854 | 0.2377 | 98 | 154983 | 5.9 |
| LIPO_FACTOR_CUTRESP_F25_R0 | FACTOR | CUTRESP | 0.25 | 0.9571 | 0.9748 | 0.2354 | 97 | 154983 | 3.1 |
| LIPO_DIRECT_CUTRESP_F25_R0 | DIRECT | CUTRESP | 0.25 | 0.9940 | 0.9591 | 0.2865 | 97 | 155102 | 10.8 |

| arm/aux | train frac | Valid | Test | ΔValid vs 100% | ΔTest vs 100% |
|---|---|---|---|---|---|
| DIRECT/CUTRESP | 0.50 | 0.8370 | 0.8055 | 0.0895 | 0.1333 |
| DIRECT/CUTRESP | 0.25 | 0.9940 | 0.9591 | 0.2466 | 0.2868 |
| FACTOR/CUTRESP | 0.50 | 0.8544 | 0.7854 | 0.1448 | 0.0627 |
| FACTOR/CUTRESP | 0.25 | 0.9571 | 0.9748 | 0.2476 | 0.2521 |

读法：在相同训练分子数下比较不同臂，才是在问“哪一种接口表示更省样本”；不同臂的 100%→50%→25% 退化斜率可以直接对比，因为子集是同一批分子。

事实：两种臂的退化斜率几乎一样（Valid 上 FACTOR +0.145/+0.248、DIRECT +0.090/+0.247），所以“更省样本”在两个臂之间没有差别；而 Valid/Test 的反转在 100%、50%、25% 三个数据量上都存在（50% 时 DIRECT 在 Valid 领先 0.017、FACTOR 在 Test 领先 0.020；25% 时 FACTOR 在 Valid 领先 0.037、DIRECT 在 Test 领先 0.016，但 25% 两边都接近训练集规模，绝对值已不可用于比较）。所以这个反转是划分的性质，不是样本量的性质。
## 9. 案例（每条由显式数值规则选出，非手工挑选）

**merge** — largest |delta_full - delta_isolated| on held-out pairs

| cell | mol a (OGB id) | mol b | Δy/σ | Δf(C) | Δf(∅) | merge | err a | err b | pair err | repeats | label spread |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cross | 873 | 1029 | -0.421 | 1.693 | -0.930 | 2.623 | – | – | – |  | – |
| cross | 1 | 807 | 3.302 | 3.355 | 0.892 | 2.463 | – | – | – |  | – |
| cross | 1332 | 3107 | -1.626 | -2.439 | -0.036 | 2.402 | – | – | – |  | – |

**redirect** — context flips the sign of the word's action

| mol a | mol b | flip side | f(Sa|C) | f(Sa|∅) | f(Sb|C) | f(Sb|∅) | Δy/σ | merge | shift a | shift b |
|---|---|---|---|---|---|---|---|---|---|---|
| 1637 | 369 | a | 1.887 | -0.369 | -0.283 | -0.795 | -1.824 | 1.744 | 2.256 | 0.512 |
| 169 | 3217 | a | 0.826 | -1.175 | 0.824 | -0.630 | 0.165 | 0.547 | 2.001 | 1.454 |
| 969 | 3408 | a | 1.127 | -0.858 | 0.910 | -1.028 | -0.330 | 0.047 | 1.985 | 1.939 |

**composed** — both single molecules predicted well (top 40%), pair delta badly (top 10%), held-out only（“预测得好”指 cross 单元内误差最小的 40%，该单元本身最难，所以绝对误差仍有 1–2 kcal/mol）

| cell | mol a (OGB id) | mol b | Δy/σ | Δf(C) | Δf(∅) | merge | err a | err b | pair err | repeats | label spread |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cross | 3095 | 1729 | -1.907 | -0.203 | -0.760 | 0.556 | 2.036 | 0.361 | 2.063 |  | – |
| cross | 1 | 1729 | 0.693 | 2.454 | 0.171 | 2.284 | 1.231 | 0.361 | 2.134 |  | – |
| cross | 3679 | 1729 | 0.437 | 2.365 | 0.207 | 2.158 | 1.519 | 0.361 | 2.335 |  | – |

**pathological** — transform repeated >=3 times inside one cell with the largest label spread

| cell | repeats in cell | label spread σ | Δy/σ of every repeat | molecule pairs (OGB id) |
|---|---|---|---|---|
| train | 3 | 1.728 | [-0.8666656411653841, -1.40317304591952, 2.5009495116096923] | [[2549, 3921], [2384, 2945], [2583, 2548]] |
| train | 10 | 1.383 | [-0.37142818815365447, 2.311108442037596, -0.6025390126784426, -0.016507901981076697, 2.6577747280222064, -0.45396789484875233, -0.28063453046801823, 2.022219960579039, 2.674282679200712, -0.49523753910722973] | [[761, 15], [53, 2023], [468, 154], [1680, 679], [2311, 690], [761, 3259], [1109, 2465], [1121, 2105], [1367, 2717], [2351, 2152]] |

* 同 transform 的两个词：`[51*][1004*][5*]([3001*])[1002*][7*]1[1010*][10*][1010*][10*][1010*][10*][1010*][10*][1010*][10*][1010*]1` / `[18*][1001*][5*]([3001*])[1004*][51*]`
* 同 transform 的两个词：`[57*][3001*]` / `[42*][3001*]`

## 10. 对三个问题的直接回答

**(1) 少量共享响应状态能否承载带接口片段的任务作用？**

能，但结论分成两半，而且整分子 RMSE 会给出误导性答案。

* **整分子 RMSE 上看不出交互的必要性**：`FACTOR` 相对纯加性 `ADD` 的差值 Valid 0.0181、Test -0.0005，配对区间都跨 0。如果只看这个指标，会得出“加性就够”的结论。
* **组合双差上，加性结构上就不够**：`ADD` 的双差恒等于 0（上下文精确抵消，见第 7 节），`FACTOR`/`DIRECT` 的切口双差都显著好于零基线（MAE 0.258/0.236 vs 0.404，sign 0.722/0.760 vs 0.550）。也就是说，要表达“环境改变片段作用”，上下文与片段的交互项是必需的。
* **但交互的具体形式不重要**：`FACTOR` 的低秩双线性与 `DIRECT` 的黑箱 MLP 在整分子 RMSE 上互有胜负（Valid FACTOR 好 0.038、Test DIRECT 好 0.050），在双差上 DIRECT 略好（0.760 vs 0.722 sign）。所以能确认的是“必需交互 + 共享接口状态”，不能确认的是“必须用可解释的低秩状态”。

**(2) 显式共享限制是否被数据集支持？**

支持，但形式比预期弱。

* 真实匹配替换目录给出 5770 个可评估对；held-out（cross，1072 对、727 个互不相同的分子、两个上下文共享同一对词）上，最好的整分子差 MAE 是 0.3529，零预测是 0.6221。模型学到的是可外推的响应，而不只是记住了标签。
* 宽支持子集（4856 对，含目标自身结构分量）：TABLE MAE 0.0885，最好的神经网络切口函数差 0.0904，最好的整分子差 0.1714——两者几乎打平，但 TABLE 的相关性 0.96 高于模型。
* 严格支持子集（1899 对，已排除目标分子所在的结构分量）：TABLE MAE 0.2218，而 `DIRECT_CUTRESP` 的切口函数差是 0.0925，比查表准一倍多。查表在严格子集上退化，正是“同 transform 跨结构共享”这一假设的直接检验：能共享的部分是真实存在的（否则模型不会在排除自身分量后仍然准确），但它不足以支撑一个查表近似。
* 反面证据同样清楚：目录里 577 个“同一 transform、多个真实对”的单元内部 `mean|Δy|` = 0.6368，只比全体单元的 0.7053 低 10%——同一个 `(C, 词)` 约束下标签本身就不一致，这是目录约束共享时无法回避的噪声地板。

**(3) 对整分子预测和数据效率的价值？**

* 对 `GLOBAL`（纯整分子描述，无接口分解）：`GLOBAL_NONE − FACTOR_NONE` Valid 0.0128、Test 0.0277（正 = 分解更好），配对区间 Valid +0.0128 [-0.0580, +0.0859]、Test +0.0277 [-0.0343, +0.0909]，都跨 0；
  在 ESOL（902 个训练分子）上同一个对比是 Valid 0.0881、Test 0.0794。方向一致：数据越少，接口分解越值钱；但在 3360 个训练分子上不显著。
* 对数据效率：lipo 用 50%/25% 训练分子重跑 `FACTOR_CUTRESP` 与 `DIRECT_CUTRESP`，见第 8 节；ESOL 只有 902 个训练分子，是同一个问题的自然小样本点。
* 对绝对精度：没有超过外部 GINE 坐标，见第 0 节。

## 11. 负结果与开放问题

* **没有超过 GINE。** 本轮最好的候选在 Valid 上比同协议 GINE 坐标差 0.0690。禁止 GNN 骨干的前提下，这个差距更像是方法上限而不是训练不足：配方的 cosine 已经走完，最佳 epoch 落在内部（100/160、135/160），尾部平台化，且 Train RMSE 已远低于 Valid RMSE（见主表）。
* **Valid/Test 排序反转。** `DIRECT_CUTRESP` 在 Test 上比 `FACTOR_CUTRESP` 好 0.0504，在 Valid 上差 0.0379；两侧 95% 区间都不含 0。best-Valid 选择本身几乎不损失 Test（`split_gap.json` 的 `cost_of_valid_selection` ≈ 0），所以这是划分难度差异，不是选择伪影。任何单侧结论都会被这次反转驳倒。
* **MOLPAIR 有害。** 只用整分子差约束 `f̄` 会让切口函数失配，Valid/Test 都比 CUTRESP 差一大截；说明起作用的不是“任何成对监督”，而是“监督切口函数本身”。
* **查表很强，但只在宽支持口径上。** 在含目标自身结构分量的 4856 对上，TABLE 的 MAE 0.0885 与最好的神经切口函数差 0.0904 打平、相关性更高；但在排除目标自身分量的 1899 对上 TABLE 退化到 0.2218，`DIRECT_CUTRESP` 反而是 0.0925。所以“按 transform 查表”在本数据量下仍是宽支持口径的强基线，但它不能替代对未知结构的外推；真正的空白是**无任何训练支持**的 914 对：那里 TABLE 无定义，模型 MAE 0.367–0.415，明显优于零预测 0.651，但绝对误差仍然偏大。

## 12. 复现

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

