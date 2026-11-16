# LCRC 显式上下文读出的两个小数据集可行性实验

**轮次** `lcrc-readout-context-pilot-small-v1` · **状态** DEVELOPMENT · **所有 Test 数字都是 `DEVELOPMENT_TEST`** ·
**分支** `lcrc-readout-context-pilot-small-v1`（依据版本 `9b3dba098915`）·
**正式轨迹执行 commit** `ee72310d9290` · **freeze hash** `5a6d48aaef012aac…` ·
**生成时间** 2026-11-16T23:23:41 · **正式 GPU 墙钟** 0.659 A100-h

> 本报告是**单 seed、两个小任务的可行性实验**。Test 划分经用户明确授权每 epoch 全量评估
> （`test_policy=always`），但**从未**参与 checkpoint、宽度、LR 或任何判读量的选择。
> 文中没有任何跨任务平均、没有把单 seed 点估计写成泛化结论、也不构成 benchmark 提交。

**一句话判读**：这套「共享局部 trunk + 预算匹配的显式加性读出」**可实现、可逐项回溯**
（TRACE 两任务全通过），但在本配方下 **molesol 上 C 损失了 A 的性能**、**mollipo 上 C 与 A 打平**，
而 **C 相对 B 的上下文增量在 Val/Test 之间变号**，因此**本轮不支持"两个显式上下文交互带来可用增益"**；
与复用的冻结 GINE 相比，仅 molesol 的 A 落在 5% 带内。建议**停止这一读出形式的本轮扩检**，
保留可复用的机制与账目代码。

## 1. 训练了什么

| 任务 | Train/Val/Test | A 读出参数 | B 读出参数 | C 读出参数 | M 活跃列 | 常量列 | 数据审计 |
|---|---|---|---|---|---|---|---|
| ogbg-molesol | 902/113/113 | 15298 | 15296 (H=207) | 15330 (H=200) | 34 | 8 | 是 |
| ogbg-mollipo | 3360/420/420 | 15298 | 15326 (H=206) | 15290 (H=198) | 36 | 6 | 是 |

三臂**共用同一个局部 trunk**（逐位等于 C1 的局部通路，同一规范初值），只有读出不同：

* **A**：冻结的 C1 原读出 `sum_v f(h_v) + rho([z_g, mean h, mean h^2, log1p n])`，不改一处；
* **B**：`b_global + sum_v c0(h_v) + sum_j u_j(z_j)`，`M` 为 Train 上跨度 > 1e-8 的原始 global 列（molesol 34 / mollipo 36），常量列在 B/C 中恒输出 0，不建无训练信号的随机头；
* **C**：B 再加两项**预先命名**的上下文交互 `a_size(z)·sum_v c_size(h_v)` 与 `a_ring(z)·sum_v c_ring(h_v)`，上下文只允许 `log1p_n_atoms` 与 `ring_atom_fraction` 两个字段。

`H` 由预算匹配决定：A 的 15298 个读出参数就是预算，B/C 的 `H` 取使名义读出参数量落在**一个宽度步长**以内的整数；整数 `H` 无法精确相等，实际差值为 ±36 以内，`analysis/protocol_freeze.json` 记录了分配与公式。

冻结缓存（**真实文件字节** SHA-256，正式运行前在计算节点上重新校验）：

* `ogbg_molesol_sq-pilot-cache-v1.npz` — `3b7a8e16ca9c7ab2…`
* `ogbg_molesol_sq-pilot-lipo-raw-v1.npz` — `a3ab9f7e22798a8f…`
* `ogbg_mollipo_sq-pilot-cache-v1.npz` — `fe002287fa9c5812…`
* `ogbg_mollipo_sq-pilot-lipo-raw-v1.npz` — `3005aa55e766d83a…`

共享 trunk 初值（三臂逐位相同，并在独立子进程中复现）：

* ogbg-molesol：trunk `1d075977ddb49ffe…`，整状态 `73d907c9ad7aa794…`
* ogbg-mollipo：trunk `30262df9366f83c3…`，整状态 `bde55cb3635e005d…`

冻配方（三臂同一套）：seed 0、360 epoch 跑满（无 early stopping）、AdamW peak 0.0003 → min 1e-05 逐步 cosine、wd 0.001、batch 128、clip 5.0、FP32、无 warmup/EMA/SWA/soup、每 epoch 全量评估三划分，选择只看最小 Val RMSE。


## 2. 主表（原始目标单位，逐任务分开，不跨任务平均）

| 任务 | 臂 | 来源 | 总参数 | best epoch | Train RMSE | Val RMSE | Test RMSE | Train MAE | Val MAE | Test MAE | final Val | final Test | last20 Val | last20 Test | updates | 训练 s | 评估 s | 纯推理 s | GPU-h | ckpt hash |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ogbg-molesol | A | 本轮 | 228674 | 248 | 0.3489 | 0.8753 | 0.8545 | 0.2596 | 0.6595 | 0.6695 | 0.9343 | 0.9053 | 0.9149 | 0.8890 | 2880 | 98.7 | 37.4 | 0.091 | 0.045 | 5a396b4675ff |
| ogbg-molesol | B | 本轮 | 228672 | 167 | 0.3966 | 0.9184 | 1.3853 | 0.2912 | 0.7042 | 0.8721 | 0.9526 | 1.4625 | 0.9529 | 1.4546 | 2880 | 92.0 | 36.8 | 0.088 | 0.042 | 69c313df089f |
| ogbg-molesol | C | 本轮 | 228706 | 127 | 0.4876 | 0.9980 | 1.1012 | 0.3900 | 0.7471 | 0.8094 | 1.0439 | 1.1524 | 1.0310 | 1.1440 | 2880 | 101.3 | 37.6 | 0.092 | 0.045 | 5659e76cf737 |
| ogbg-molesol | G | 复用冻结 | 377094 | 179 | 0.3299 | 0.9353 | 0.9354 | 0.2371 | 0.7095 | 0.7288 | 0.9963 | 0.9479 | 0.9936 | 0.9473 |  | 244.8 |  |  | — | — |
| ogbg-mollipo | A | 本轮 | 228674 | 275 | 0.2639 | 0.7086 | 0.7254 | 0.1949 | 0.5123 | 0.5292 | 0.7158 | 0.7418 | 0.7137 | 0.7401 | 9720 | 385.1 | 199.4 | 0.541 | 0.178 | 1b96644df50a |
| ogbg-mollipo | B | 本轮 | 228702 | 116 | 0.4052 | 0.7589 | 0.7290 | 0.3050 | 0.5402 | 0.5466 | 0.8122 | 0.7507 | 0.8087 | 0.7491 | 9720 | 355.1 | 194.8 | 0.534 | 0.168 | 8ae167f7e3c7 |
| ogbg-mollipo | C | 本轮 | 228666 | 133 | 0.3912 | 0.6967 | 0.7404 | 0.2936 | 0.5136 | 0.5539 | 0.7428 | 0.7664 | 0.7397 | 0.7644 | 9720 | 394.9 | 200.2 | 0.543 | 0.181 | 9c9a326026c1 |
| ogbg-mollipo | G | 复用冻结 | 377350 | 168 | 0.2691 | 0.6091 | 0.6456 | 0.2054 | 0.4624 | 0.4847 | 0.6326 | 0.6695 | 0.6337 | 0.6690 |  | 923.6 |  |  | — | — |

* `best epoch` = 最小 Val RMSE 的那一个 epoch；同一冻结状态的 Test 列就是 `DEVELOPMENT_TEST`。
* `final` 是同一条轨迹第 360 epoch 的状态；`last20` 是最后 20 个 epoch 的算术均值——曲线摘要，不是 ensemble，也不给 single checkpoint 编造预测 CI。
* `G` 是 `sq-pilot-small-v1` 的冻结 GINE，**原样复用、本轮不重训**（`REUSED_FROZEN_REFERENCE`）；它的输入通路不同、LR 预设更强（`1e-3` 对 本轮的 `3e-4`），只作强性能坐标。
* `GPU-h` 一列：本轮的 6 条轨迹按各自 worker 的 job 墙钟、以该臂 `train+eval` 秒数占比分摊（分摊明细见 §12.1 与 `analysis/resource_ledger.json`）；复用参照的 GINE 轨迹为 0。


### 2.1 同一冻结状态的 Val → Test 迁移差

选中状态是 Val RMSE 的极小点，所以这张表问的是：这个 Val 数字在 Test 上还剩多少。molesol 的 Val 只有 113 个分子，Val 上的小改进本身噪声很大。

| 任务 | 臂 | Val RMSE | Test RMSE | Test − Val | Test / Val | last20 Test − Val | last20 Test / Val |
|---|---|---|---|---|---|---|---|
| ogbg-molesol | A | 0.8753 | 0.8545 | -0.0208 | 0.976 | -0.0259 | 0.972 |
| ogbg-molesol | B | 0.9184 | 1.3853 | 0.4669 | 1.508 | 0.5016 | 1.526 |
| ogbg-molesol | C | 0.9980 | 1.1012 | 0.1032 | 1.103 | 0.1130 | 1.110 |
| ogbg-molesol | G | 0.9353 | 0.9354 | 0.0001 | 1.000 | -0.0463 | 0.953 |
| ogbg-mollipo | A | 0.7086 | 0.7254 | 0.0168 | 1.024 | 0.0264 | 1.037 |
| ogbg-mollipo | B | 0.7589 | 0.7290 | -0.0298 | 0.961 | -0.0597 | 0.926 |
| ogbg-mollipo | C | 0.6967 | 0.7404 | 0.0437 | 1.063 | 0.0247 | 1.033 |
| ogbg-mollipo | G | 0.6091 | 0.6456 | 0.0365 | 1.060 | 0.0353 | 1.056 |

A/B/C 中最大的选中态差距出现在 ogbg-molesol/B（Test = 1.51× Val）。这个量级的差距说明该臂的 Val 选中点在本题上不是一个可靠的性能坐标；本报告把它列出来而不是抹平，且全文任何地方都不用跨 split 平均值。


## 3. 三个主对照 + 样本不确定性

| 任务 | 对照 | Δ Val | 比值 Val | 95% CI (Val) | Δ Test | 比值 Test | 95% CI (Test) | Δ final Val | Δ final Test | Δ last20 Val | Δ last20 Test |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ogbg-molesol | B-A | 0.0431 | 1.049 | [-0.054, 0.151] | 0.5308 | 1.621 | [0.073, 1.096] | 0.0183 | 0.5572 | 0.0380 | 0.5655 |
| ogbg-molesol | C-A | 0.1227 | 1.140 | [0.022, 0.243] | 0.2467 | 1.289 | [0.044, 0.480] | 0.1096 | 0.2471 | 0.1161 | 0.2550 |
| ogbg-molesol | C-B | 0.0797 | 1.087 | [-0.057, 0.242] | -0.2840 | 0.795 | [-0.647, 0.068] | 0.0913 | -0.3101 | 0.0780 | -0.3105 |
| ogbg-molesol | A-G | -0.0600 | 0.936 | [-0.160, 0.042] | -0.0809 | 0.913 | [-0.206, 0.036] | -0.0619 | -0.0426 | -0.0787 | -0.0583 |
| ogbg-molesol | C-G | 0.0627 | 1.067 | [-0.083, 0.205] | 0.1658 | 1.177 | [-0.060, 0.421] | 0.0476 | 0.2045 | 0.0374 | 0.1967 |
| ogbg-mollipo | B-A | 0.0503 | 1.071 | [-0.014, 0.112] | 0.0037 | 1.005 | [-0.052, 0.062] | 0.0965 | 0.0089 | 0.0950 | 0.0089 |
| ogbg-mollipo | C-A | -0.0119 | 0.983 | [-0.055, 0.034] | 0.0150 | 1.021 | [-0.034, 0.064] | 0.0270 | 0.0246 | 0.0260 | 0.0243 |
| ogbg-mollipo | C-B | -0.0622 | 0.918 | [-0.141, 0.005] | 0.0113 | 1.016 | [-0.024, 0.043] | -0.0695 | 0.0157 | -0.0690 | 0.0154 |
| ogbg-mollipo | A-G | 0.0996 | 1.163 | [0.021, 0.181] | 0.0798 | 1.124 | [0.032, 0.129] | 0.0832 | 0.0723 | 0.0800 | 0.0711 |
| ogbg-mollipo | C-G | 0.0876 | 1.144 | [0.025, 0.150] | 0.0948 | 1.147 | [0.043, 0.144] | 0.1102 | 0.0968 | 0.1060 | 0.0955 |

Δ = 左 − 右（负值偏向左侧），比值为左/右。区间是**配对分子 bootstrap** 的 2.5/97.5 分位（2000 次重采样，RNG 1703，同一 split 内所有模型共用同一次重采样，每次重采样重算 RMSE/MAE）。
它**条件于当前单 seed 选中状态**：不估计训练 seed 方差、不修复 Val 选择、也不消除 Test 的开发暴露。

逐行 drop-one 敏感性（是否由一行决定方向）：

| 任务 | split | 对照 | Δ RMSE（全量） | 去掉一行后的 Δ 范围 | 翻号行数 | 单行最大位移 |
|---|---|---|---|---|---|---|
| ogbg-molesol | valid | B-A | 0.0431 | [0.0207, 0.0606] | 0 | 0.0224 |
| ogbg-molesol | valid | C-A | 0.1227 | [0.0718, 0.1365] | 0 | 0.0509 |
| ogbg-molesol | valid | C-B | 0.0797 | [0.0175, 0.0992] | 0 | 0.0622 |
| ogbg-molesol | test | B-A | 0.5308 | [0.1930, 0.5565] | 0 | 0.3377 |
| ogbg-molesol | test | C-A | 0.2467 | [0.1397, 0.2662] | 0 | 0.1070 |
| ogbg-molesol | test | C-B | -0.2840 | [-0.3298, -0.0533] | 0 | 0.2307 |
| ogbg-mollipo | valid | B-A | 0.0503 | [0.0184, 0.0591] | 0 | 0.0319 |
| ogbg-mollipo | valid | C-A | -0.0119 | [-0.0195, 0.0003] | 1 | 0.0122 |
| ogbg-mollipo | valid | C-B | -0.0622 | [-0.0663, -0.0210] | 0 | 0.0412 |
| ogbg-mollipo | test | B-A | 0.0037 | [-0.0056, 0.0161] | 7 | 0.0125 |
| ogbg-mollipo | test | C-A | 0.0150 | [0.0051, 0.0272] | 0 | 0.0122 |
| ogbg-mollipo | test | C-B | 0.0113 | [0.0066, 0.0157] | 0 | 0.0047 |

_删除后的数字只作诊断，官方分数永远是全量 split。_


## 4. 定向实现检查（正式训练开始前）

| 任务 | 门禁数 | 全部通过 | 失败数 |
|---|---|---|---|
| ogbg-molesol | 13 | 是 | 0 |
| ogbg-mollipo | 13 | 是 | 0 |

| 门禁 | ogbg-molesol | ogbg-mollipo |
|---|---|---|
| armA_equivalence | 通过 | 通过 |
| checkpoint_reload | 通过 | 通过 |
| data | 通过 | 通过 |
| dropout_shared | 通过 | 通过 |
| gine_reference_reuse | 通过 | 通过 |
| gradients | 通过 | 通过 |
| invariance | 通过 | 通过 |
| ledger_closure | 通过 | 通过 |
| parameters | 通过 | 通过 |
| roundtrip_and_zero_readout | 通过 | 通过 |
| scale_equivalence | 通过 | 通过 |
| shared_init | 通过 | 通过 |
| single_variable_functions | 通过 | 通过 |

其中与 prompt 直接对应的是：A 与只读 `DissectLCRC("C1")` 在同一状态、同一真实 batch 上预测/aux 逐位一致；三臂共同 trunk 初值逐位相等（并在新进程复现）；`u_j` 初值逐位相等；`q(0)=0` 且向量化与逐标量一致、无被中心化吃掉的输出 bias；每次前向只有一次 dropout 且被所有局部头共用；raw/标准化往返与「零输出 → Train 均值」正确；新参数梯度有限可达；eval 批量拆分、分子隔离、原子重编号不变性；账目闭合；`a_k×2 / c_k÷2` 尺度等价；checkpoint 保存/重载后预测与全部账目重放一致；以及冻结 GINE 复用门禁（逐样本重算 RMSE/MAE、360 epoch 曲线与 best/final/last20 身份）。


## 5. 回溯：每个数还能不能重新拆开

容差全程是预先冻结的 `tol_raw(x) = 1e-5 + 1e-6·max(1,|x|)`（原始单位）；失败不用事后放宽阈值。

账目定义（与 prompt §十 一致）：`intercept_raw = mu + sigma·b_global`，`base_root_v_raw = sigma·c0(h_v)`，`global_j_raw = sigma·u_j(z_j)`，`interaction_{v,k}_raw = sigma·a_k(z_k)·c_k(h_v)`，总和 = `pred_raw`。**只有 intercept 带 mu**，每个 root 与每个交互都只带 sigma。

### 5.1 诊断 1：全量 Val/Test 预测重建

| 模型 | split | 行数 | max abs error | median abs error | 容差 | 判定 |
|---|---|---|---|---|---|---|
| ogbg-molesol/B | valid | 113 | 2.69e-07 | 2.31e-08 | 2.03e-05 | 通过 |
| ogbg-molesol/B | test | 113 | 4.70e-07 | 2.31e-08 | 1.91e-05 | 通过 |
| ogbg-molesol/C | valid | 113 | 5.48e-07 | 5.68e-08 | 1.97e-05 | 通过 |
| ogbg-molesol/C | test | 113 | 4.86e-07 | 7.03e-08 | 1.93e-05 | 通过 |
| ogbg-mollipo/B | valid | 420 | 1.97e-07 | 1.92e-08 | 1.72e-05 | 通过 |
| ogbg-mollipo/B | test | 420 | 2.43e-07 | 1.92e-08 | 1.56e-05 | 通过 |
| ogbg-mollipo/C | valid | 420 | 2.71e-07 | 1.95e-08 | 1.51e-05 | 通过 |
| ogbg-mollipo/C | test | 420 | 2.02e-07 | 2.06e-08 | 1.51e-05 | 通过 |

逐分子账目：`analysis/ledgers/<task>_<arm>_molecules.csv.gz`（含 row_id、split、smiles、y_raw、pred_raw、n_atoms、intercept、sum_base、每项 global_j、两个 a_k/S_k/sum_interaction、重建值与误差）；逐 root 账目：`analysis/ledgers/<task>_<arm>_atoms.csv.gz`（含官方 row_id、分子内 root index、root 原子类型、c0/c_size/c_ring、base_root_raw、两个 interaction_root_raw、S1/S2 occurrence 区间、以及该 root 冻结输入窗口的 sha256 定位键）；42 列 g、active mask 与标准化均值/尺度可由冻结缓存 + 特征 spec 查回，`results/<task>/budget.json` 记录了 active/inactive 列名、`H` 的预算公式、scaler 与 split 尺寸。

### 5.2 选中 checkpoint 的保存/重载重放

| 任务 | 臂 | Val 预测最大差 | Test 预测最大差 | 重载状态 hash | 判定 |
|---|---|---|---|---|---|
| ogbg-molesol | A | 0.0000020 | 0.0000015 | ac89f8ba9c0a | 通过 |
| ogbg-molesol | B | 0.0000025 | 0.0000017 | d182cb7ca5da | 通过 |
| ogbg-molesol | C | 0.0000030 | 0.0000025 | 10b21544b72f | 通过 |
| ogbg-mollipo | A | 0.0000018 | 0.0000015 | 3b72ef730698 | 通过 |
| ogbg-mollipo | B | 0.0000040 | 0.0000014 | 8f0f56a3f8b5 | 通过 |
| ogbg-mollipo | C | 0.0000035 | 0.0000016 | 7bf8802725c0 | 通过 |

### 5.3 诊断 2：固定局部输入干预（输入通道干预，不是化学删边）

每个 Val 分子单图 collate，删掉该分子第 0 条 typed-S2 occurrence（同时删掉对应 `path_cat/path_end63/path_end/path_closure` 条目并修正分子内 `path_ptr`），**不改** raw graph、S1、atom scalars 和 global；然后比较完整预测变化与账目变化之和。

| 任务 | 分子数 | 重建失败数 | 判定 |
|---|---|---|---|
| ogbg-molesol | 32 | 0 | 通过 |
| ogbg-mollipo | 32 | 0 | 通过 |

所有样本下，其它 root 的 base 项与全部 global 项变化 ≤1e-6；arm C 的 `a_k` 完全不变（上下文向量没动）。变化由被影响 root 的 base 与两个交互项合计重建。被删 occurrence 的 root/end/键型与其 middle 候选集记录在 `analysis/interventions.json`（middle 映射不唯一时如实列出候选）。**A 仍然通过 rho 响应**（mean h / mean h² / n），按实际记录——那正是本轮要处理的纠缠，不记作实现错误。

### 5.4 诊断 3：固定 h 的上下文条件响应

对同样的 32 个 Val 分子，固定已算出的 h，把某个上下文从**观测值**推到 Train Q75（Q25/Q75 由同一 Train norm 换算），同时正确更新 `u_j` 与 `a_k`；闭式 `Δ = Δu_k + Δa_k·S_k` 与独立整前向比较。

| 任务 | 上下文字段 | RMS Δu（主效应） | RMS Δ 交互 | 失败数 | 判定 |
|---|---|---|---|---|---|
| ogbg-molesol | log1p_n_atoms | 0.0834 | 0.4709 | 0 | 通过 |
| ogbg-molesol | ring_atom_fraction | 0.1645 | 0.0688 | 0 | 通过 |
| ogbg-mollipo | log1p_n_atoms | 0.2636 | 0.2111 | 0 | 通过 |
| ogbg-mollipo | ring_atom_fraction | 0.1919 | 0.1838 | 0 | 通过 |

`Δu` 是**单项 global 主效应**、`Δ 交互` 是**被局部状态加权的交互效应**，两列分开正是为了避免把「数值进去了」和「结构被读得不一样了」混成同一个「结构效应」。这里都是条件模型干预，混合后的输入不保证是真实分子。未变的上下文与局部项全程不变。

### 5.5 诊断 5：等价尺度检验

只读复制 C 的选中状态，把某 `a_k` 输出 ×2、对应 `c_k` 输出 ÷2：

| 任务 | Val 行数 | 预测最大差 | a 比值 | 判定 |
|---|---|---|---|---|
| ogbg-molesol | 113 | 0.0000000 | 2.0000–2.0000 | 通过 |
| ogbg-mollipo | 420 | 0.0000000 | 2.0000–2.0000 | 通过 |

因此 `a_k` 或 `c_k` **单独的参数范数不能当作重要性指标**。这只验证该等价变换，不声称解决了全部归因不唯一问题。

### 5.6 A 轴判定：TRACE

| 任务 | 判定 |
|---|---|
| ogbg-molesol | TRACE_PASS |
| ogbg-mollipo | TRACE_PASS |

`TRACE_PASS` 只表示账目闭合：预测重建、两类干预重建、尺度/批量/重编号等价、checkpoint 重放都通过。它**不是**「证明了真实化学贡献」——root 是重叠的局部输入环境，不是可加的独立官能团。


## 6. 交互是真被用了，还是只有常数/均摊效应

用选中模型对 **Train 根**做节点等权均值 `c̄_k`，把分子级交互精确分解为 `I_k = a_k·[n_atoms·c̄_k + Σ_v (c_k(v) − c̄_k)]`，分别报告「常数根项」与「结构偏离项」的 RMS。这是来自 Train 的参考分解，不重拟合权重；RMS 之比可以因抵消 >1，**不写成互斥的误差解释百分比**。

| 任务 | split | 字段 | RMS 交互（原始） | 常数根项 | 结构偏离项 | 根间 std/RMS 中位数 | 单根分子数 |
|---|---|---|---|---|---|---|---|
| ogbg-molesol | valid | log1p_n_atoms | 2.1770 | 2.5750 | 1.2043 | 0.660 | 0 |
| ogbg-molesol | valid | ring_atom_fraction | 0.2456 | 0.0634 | 0.2044 | 0.869 | 0 |
| ogbg-molesol | test | log1p_n_atoms | 2.7540 | 2.8402 | 1.1726 | 0.636 | 0 |
| ogbg-molesol | test | ring_atom_fraction | 0.2793 | 0.0660 | 0.2401 | 0.897 | 0 |
| ogbg-mollipo | valid | log1p_n_atoms | 0.2290 | 0.0206 | 0.2352 | 0.990 | 0 |
| ogbg-mollipo | valid | ring_atom_fraction | 0.2768 | 0.2058 | 0.1308 | 0.932 | 0 |
| ogbg-mollipo | test | log1p_n_atoms | 0.1895 | 0.0168 | 0.1873 | 0.987 | 0 |
| ogbg-mollipo | test | ring_atom_fraction | 0.3075 | 0.2164 | 0.1528 | 0.918 | 0 |

工程读数（预先冻结）：某交互只有在 Val、Test 与 Q25→Q75 干预三处 RMS 都 ≥ Train 目标标准差的 1% 时才记 `RESOLVED_FUNCTIONAL_RESPONSE`，否则 `WEAK_AT_1PCT_TARGET_SCALE`；根间差异只看 `n_atoms≥2` 的分子，`std_v(c_k)/max(RMS_v(c_k),1e-12)` 的 Val/Test 中位数 <0.05 记 `LOW_ROOT_SPECIFICITY_AT_5PCT`，否则只记 `NONUNIFORM_ROOT_RESPONSE`。0.01 是本轮 1% 目标标准差的读数分辨率，不是化学或统计显著性门槛；`NONUNIFORM` 也不是化学机制证明。

| 任务 | 字段 | RMS（Val） | RMS（干预） | 根间 std/RMS 中位数 | 响应读数 | 根特异性读数 |
|---|---|---|---|---|---|---|
| ogbg-molesol | log1p_n_atoms | 1.0534 | 0.4709 | 0.660 | RESOLVED_FUNCTIONAL_RESPONSE | NONUNIFORM_ROOT_RESPONSE |
| ogbg-molesol | ring_atom_fraction | 0.1188 | 0.0688 | 0.869 | RESOLVED_FUNCTIONAL_RESPONSE | NONUNIFORM_ROOT_RESPONSE |
| ogbg-mollipo | log1p_n_atoms | 0.1890 | 0.2111 | 0.990 | RESOLVED_FUNCTIONAL_RESPONSE | NONUNIFORM_ROOT_RESPONSE |
| ogbg-mollipo | ring_atom_fraction | 0.2285 | 0.1838 | 0.932 | RESOLVED_FUNCTIONAL_RESPONSE | NONUNIFORM_ROOT_RESPONSE |

结论：在两个任务上，交互项**在数值上确实被使用**（RMS 达到 0.19–1.05 个目标标准差，根间差异明显，不是常数倍乘），但**它的方向在 Val/Test 之间不稳定**（见第 7 节 C 轴）。即「机制被用上了」与「机制带来可复现的增益」在本轮是两件事，不能混为一谈。


## 7. 四个预先冻结的判读轴（分别输出，不合成一个 PASS）

| 任务 | 轴 | 判定 |
|---|---|---|
| ogbg-molesol | A. 回溯 | TRACE_PASS |
| ogbg-molesol | B. 性能保留 | PERFORMANCE_LOSS_THIS_RECIPE |
| ogbg-molesol | C. 上下文增量 | SPLIT_OR_SELECTION_SENSITIVE |
| ogbg-molesol | D. 对 G 的竞争力 | WITHIN_5PCT_OF_THIS_GINE (A) |
| ogbg-mollipo | A. 回溯 | TRACE_PASS |
| ogbg-mollipo | B. 性能保留 | PRESERVED_POINT_ESTIMATE / HORIZON_CONSISTENT |
| ogbg-mollipo | C. 上下文增量 | SPLIT_OR_SELECTION_SENSITIVE |
| ogbg-mollipo | D. 对 G 的竞争力 | BEHIND_THIS_GINE |

### ogbg-molesol

* **B 性能保留（C 对 A，容差 1.05×）**：选中态比值 Val 1.140、Test 1.289；last20 比值 1.127 / 1.287。
* **C 上下文增量（C 对 B，阈值 +2.00%）**：

| 表面 | C 相对 B 的相对增益 |
|---|---|
| last20_test | +21.35% |
| last20_valid | -8.19% |
| selected_test | +20.50% |
| selected_valid | -8.67% |

  达到阈值：selected_test, last20_test；为负：selected_valid, last20_valid；Val/Test 是否变号：是。

* **D 对 G 的竞争力（各臂分别对复用 G）**：

| 臂 | Val RMSE | Test RMSE | Val / G | Test / G | 两个 split 都在 5% 内 |
|---|---|---|---|---|---|
| A | 0.8753 | 0.8545 | 0.936 | 0.913 | 是 |
| B | 0.9184 | 1.3853 | 0.982 | 1.481 | 否 |
| C | 0.9980 | 1.1012 | 1.067 | 1.177 | 否 |

  G 是输入通路不同、LR 预设更强（1e-3 对 3e-4）的强性能坐标：差距不归因于单个算子，也不构成任何方向的 SOTA 声明。

### ogbg-mollipo

* **B 性能保留（C 对 A，容差 1.05×）**：选中态比值 Val 0.983、Test 1.021；last20 比值 1.036 / 1.033。
* **C 上下文增量（C 对 B，阈值 +2.00%）**：

| 表面 | C 相对 B 的相对增益 |
|---|---|
| last20_test | -2.06% |
| last20_valid | +8.53% |
| selected_test | -1.56% |
| selected_valid | +8.20% |

  达到阈值：selected_valid, last20_valid；为负：selected_test, last20_test；Val/Test 是否变号：是。

* **D 对 G 的竞争力（各臂分别对复用 G）**：

| 臂 | Val RMSE | Test RMSE | Val / G | Test / G | 两个 split 都在 5% 内 |
|---|---|---|---|---|---|
| A | 0.7086 | 0.7254 | 1.163 | 1.124 | 否 |
| B | 0.7589 | 0.7290 | 1.246 | 1.129 | 否 |
| C | 0.6967 | 0.7404 | 1.144 | 1.147 | 否 |

  G 是输入通路不同、LR 预设更强（1e-3 对 3e-4）的强性能坐标：差距不归因于单个算子，也不构成任何方向的 SOTA 声明。


## 8. A 是否复现了冻结的 C1 轨迹（实现可信度的旁证）

A 是**未改动**的 C1 读出，所以它的轨迹应与 `sq-pilot-small-v1` 的冻结 `L` 轨迹一致，只差 prompt 冻结的那一处差异：batch 顺序（本轮 `rng(0)` 逐 epoch permutation，历史跑用 `shuffle_offset=917000`）。

| 任务 | peak LR 相同 | epoch 相同 | L 的 best epoch | A 的 best epoch | L Val | A Val | Δ Val | Δ Test | Δ last20 Val |
|---|---|---|---|---|---|---|---|---|---|
| ogbg-molesol | 是 | 是 | 248 | 248 | 0.8686 | 0.8753 | +0.77% | -0.30% | +0.84% |
| ogbg-mollipo | 是 | 是 | 275 | 275 | 0.7052 | 0.7086 | +0.49% | +1.61% | +0.29% |

**同一 best epoch、1% 量级的差异**，正是「同一配方 + 不同 shuffle 约定」的预期特征；这不是新实验，也不能当作 seed 稳定性证据。


## 9. 直接回答 prompt §十三 的 7 个问题

**1. C 相对 A 保住了多少性能？哪些 split 或训练阶段冲突？**

ogbg-molesol：C/A 的 Val 比值 1.140、Test 比值 1.289（C 0.9980/1.1012 vs A 0.8753/0.8545），last20 比值 1.127/1.287；ogbg-mollipo：C/A 的 Val 比值 0.983、Test 比值 1.021（C 0.6967/0.7404 vs A 0.7086/0.7254），last20 比值 1.036/1.033。

mollipo 两个 split 都在 5% 容差内（`PRESERVED_POINT_ESTIMATE / HORIZON_CONSISTENT`），molesol 两个都超（`PERFORMANCE_LOSS_THIS_RECIPE`）。冲突点在 **molesol 的 Test**：B 的选中态 Test 比 Val 差 1.51 倍，说明 113 个分子的 Val 极小点在该任务上选不到可靠状态；C 的 Test 反而好于 B，但 Val 更差——这是「Val 选点噪声」而不是「C 在 Test 上更稳」的证据。

**2. C 相对 B 的增量多大，配对区间与 drop-one 怎样？**

ogbg-molesol：选中态 ΔVal 0.0797（比值 1.087）、ΔTest -0.2840；drop-one 后 Val 的 Δ 范围 [0.0175, 0.0992]（翻号 0 行），Test 范围 [-0.3298, -0.0533]（翻号 0 行）；ogbg-mollipo：选中态 ΔVal -0.0622（比值 0.918）、ΔTest 0.0113；drop-one 后 Val 的 Δ 范围 [-0.0663, -0.0210]（翻号 0 行），Test 范围 [0.0066, 0.0157]（翻号 0 行）。

区间（第 3 节）在两个任务、两个 split 上都跨 0，且方向在 Val/Test 之间相反；drop-one 也显示方向不由单行决定（除 mollipo Val 的 C-A 有 1 行翻号，C-B 无）。因此按预冻结规则记 `SPLIT_OR_SELECTION_SENSITIVE` / `SPLIT_OR_SELECTION_SENSITIVE`：**不宣称上下文增益**。

**3. 交互是否真的被使用，是否只是规模/均摊效应？**

ogbg-molesol/log1p_n_atoms：RMS 交互 2.1770 原始单位（常数根项 2.5750、结构偏离项 1.2043），根间 std/RMS 中位数 0.660；ogbg-molesol/ring_atom_fraction：RMS 交互 0.2456 原始单位（常数根项 0.0634、结构偏离项 0.2044），根间 std/RMS 中位数 0.869；ogbg-mollipo/log1p_n_atoms：RMS 交互 0.2290 原始单位（常数根项 0.0206、结构偏离项 0.2352），根间 std/RMS 中位数 0.990；ogbg-mollipo/ring_atom_fraction：RMS 交互 0.2768 原始单位（常数根项 0.2058、结构偏离项 0.1308），根间 std/RMS 中位数 0.932。

两个字段在 Val/Test 的 RMS 都远超 1% 目标标准差的读数分辨率，且根间差异大（不是常数倍乘）；规模效应（`n·c̄_k`）与结构偏离项都被单独量化。所以**机制在数值层面被使用了**；但第 7 节的性能证据显示它**没有产生可复现的方向性收益**——两者必须分开陈述。

**4. 回溯能覆盖到哪一级，哪些内部计算仍不能解释？**

覆盖到：**分子级**（intercept / sum_base / 每个 global_j / 两个 a_k·S_k / 交互和 / 重建值）与**root 级**（c0、c_size、c_ring、base_root_raw、两个 interaction_root_raw、S1/S2 occurrence 区间、冻结输入窗口的 sha256 定位键）；以及**干预级**（删一条 typed-S2 occurrence 的账目变化可逐项重建）和**尺度等价**（a_k×2 / c_k÷2 精确不变）。全部 Val/Test 放行（最大误差 5.5e-7 原始单位）。

仍不能解释的是：**root 内部哪些内部路径或子结构最『负责』这个 c_k**——最后一步归因不唯一，因为 root 环境按定义是重叠的、typed-S2 数远多于原子数；本轮的 root 账目不是「独立化学片段可加」。`a_k`/`c_k` 的参数范数也不是重要性指标（诊断 5）。

**5. 与强 GINE 相比还有多少差距？**

| 任务 | 臂 | G Val | G Test | Δ Val vs G | Δ Test vs G | 比值 Val/Test |
|---|---|---|---|---|---|---|
| ogbg-molesol | A | 0.9353 | 0.9354 | -0.0600 | -0.0809 | 0.936 / 0.913 |
| ogbg-molesol | B | 0.9353 | 0.9354 | -0.0169 | 0.4499 | 0.982 / 1.481 |
| ogbg-molesol | C | 0.9353 | 0.9354 | 0.0627 | 0.1658 | 1.067 / 1.177 |
| ogbg-mollipo | A | 0.6091 | 0.6456 | 0.0996 | 0.0798 | 1.163 / 1.124 |
| ogbg-mollipo | B | 0.6091 | 0.6456 | 0.1498 | 0.0834 | 1.246 / 1.129 |
| ogbg-mollipo | C | 0.6091 | 0.6456 | 0.0876 | 0.0948 | 1.144 / 1.147 |

molesol 上 **A** 已在 5% 带内（甚至略好于 G），但 **B/C 不在**；mollipo 上三臂全部落后 12–25%。G 是不同输入通路且用更强 LR 预设（1e-3）的强坐标，所以这只表示「距离」，不表示某个算子的责任，也不构成「击败 SOTA」或「被 SOTA 击败」的结论。

**6. 有无未完成项、执行偏离、影响结果的 bug？**

没有未完成项：6 条正式轨迹全部跑满 360 epoch、每 epoch 三划分全量评估（`test_eval_count=360`），两任务各 13 个定向检查全通过；G 复用门禁通过，因此**没有**触发 fallback 重训。执行偏离只有一处、且是 prompt 冻结的：batch 顺序用 `rng(0)` permutation（历史跑用 `shuffle_offset=917000`），因此 A 与冻结 L 有 ~1% 差异（第 8 节）。

工程 bug 全部在正式训练前发现并修复，记录在 `analysis/engineering_log.md`；修复只涉及实现（索引、dtype、路径、账目字段、报告模板），**没有**为了让结果好看而改科学定义，也没有作废重跑的正式轨迹（`resource_ledger.json` 的 `wasted_reruns` 为空）。

**7. 下一轮应保留 B、保留 C，还是停止这一读出形式？**

按预冻结规则：**B 未保住 A 的性能（molesol 超容差），C 相对 B 也没有稳定的增量**，所以规则给出的结论是**停止把这一读出形式继续扩大**——既不建议保留 C，也不建议在没有新信息的情况下改用「更简单」的 B 作为主读出（B 在 molesol 上同样损失）。

可以保留的是**机制与账目代码**：共享 trunk 构造、预算匹配读出、账目/干预/尺度等价的检查套件与分析驱动，它们与具体读出形式解耦。任何后续尝试都需要新的预注册，并在一个「上下文确实应该有用」的任务上，把「数值进去了」与「结构被读得不一样了」分开验证。本轮不追加实验。


## 10. 本轮不声称什么

* **只有一个 seed。** 每条对照都是单条轨迹；配对 bootstrap 只描述「给定冻结状态」下的分子不确定性，
  不估计训练方差，也不是等价性检验（区间跨 0 不能写成「等价」）。
* **Val 选择在两侧都偏乐观。** 每个数都是 Val 上的最好 epoch（113 / 420 个分子）；三臂之间因为
  规则、split 相同而可比，但绝对数字不是泛化估计。
* **Test 是 `DEVELOPMENT_TEST`。** 它每 epoch 被看，且只作为开发评估列报；任何拿它做选择的读者都会使它失效。
  本轮没有这样做。
* **G 是坐标不是对照。** 冻结 GINE 输入通路不同、LR 预设更强，差距不能归因于单个算子。
* **没有化学结论。** root 是重叠的局部输入环境；root 窗口重叠，不是独立官能团，内部路径的最后一步归因不唯一。
* **没有架构搜索。** 读出形式在跑之前就固定在代码里；换第三个上下文字段、加宽 `a_k`、改 `H` 都属于另一轮。
* **只有两个小任务。** 没有 ZINC / PCBA / public-122 / MoleculeNet benchmark 声明。

## 11. 复现入口

```bash
# 本地身份与门禁（CPU，约 1 分钟）
uv run pytest -q tracks/structural-sufficiency/lcrc_readout_context_pilot_small_v1/tests
bash tracks/structural-sufficiency/lcrc_readout_context_pilot_small_v1/run_remote.sh \
  --purpose "prepare" --set stage=prepare

# 正式训练：一个 GPU 一个 worker，各自一个数据集
rr deploy res
rr run res lcrc-rc-molesol --pool res-gpu0 \
  --result tracks/structural-sufficiency/lcrc_readout_context_pilot_small_v1/results \
  --result tracks/structural-sufficiency/lcrc_readout_context_pilot_small_v1/analysis \
  -- bash tracks/structural-sufficiency/lcrc_readout_context_pilot_small_v1/run_remote.sh \
     --purpose "formal: ogbg-molesol" --set stage=dataset_worker --set dataset=ogbg-molesol
# 另一个池/数据集：--pool res-gpu1 --set dataset=ogbg-mollipo

# 拉回并从零重算全部表格、账目、干预、图与判定（纯 CPU）
bash tracks/structural-sufficiency/lcrc_readout_context_pilot_small_v1/run_remote.sh \
  --purpose "aggregate" --set stage=aggregate
```

正式轨迹执行 commit 见 §12 的 `commit` 列（由 rr 记录），分支 `lcrc-readout-context-pilot-small-v1`，
依据版本 `9b3dba098915`。执行来源（host / GPU / driver / torch / commit）
逐 run 记录在 `analysis/run_provenance.json`，由 rr 控制面读取，不是手填。

## 12. 执行来源与真实成本

读自控制面 run 记录（`rr status --json`），不是手填：

| experiment | pool | node | driver | torch | commit | 可见 GPU | 状态 | exit | 开始 | 结束 |
|---|---|---|---|---|---|---|---|---|---|---|
| lcrc-rc-molesol | res-gpu0 | a100-2 | 550.163.01 | 2.5.1+cu124 | ee72310d9290 | 0 | completed | 0x0 | 2026-11-16T22:31:39+08:00 | 2026-11-16T22:39:35+08:00 |
| lcrc-rc-mollipo | res-gpu1 | a100-2 | 550.163.01 | 2.5.1+cu124 | ee72310d9290 | 1 | completed | 0x0 | 2026-11-16T22:31:51+08:00 | 2026-11-16T23:03:29+08:00 |

### 12.1 真实 GPU 成本

| experiment | run_id | pool | job 墙钟 s | A100-h |
|---|---|---|---|---|
| lcrc-rc-molesol | lcrc-rc-molesol-20261116-223138-ad73b71e | res-gpu0 | 476 | 0.132 |
| lcrc-rc-mollipo | lcrc-rc-mollipo-20261116-223149-6b1569e9 | res-gpu1 | 1898 | 0.527 |

| 任务 | 臂 | 训练 s（trainer 侧） | 评估 s | 三划分纯推理 s | updates |
|---|---|---|---|---|---|
| ogbg-molesol | A | 98.7 | 37.4 | 0.091 | 2880 |
| ogbg-molesol | B | 92.0 | 36.8 | 0.088 | 2880 |
| ogbg-molesol | C | 101.3 | 37.6 | 0.092 | 2880 |
| ogbg-mollipo | A | 385.1 | 199.4 | 0.541 | 9720 |
| ogbg-mollipo | B | 355.1 | 194.8 | 0.534 | 9720 |
| ogbg-mollipo | C | 394.9 | 200.2 | 0.543 | 9720 |

两个 worker 合计 **0.659 A100-h** job 墙钟（各自独占一张卡、每卡只有一个训练进程、无 fan-out）；另有 1 个 A100 0.003 h 的 smoke（3 epoch × 3 臂，单独 `_smoke` 目录，从不作为结论）。prepare / 聚合分析 / 画图全部在本地 CPU 完成，不占 GPU。软预算 2.0 A100-h、硬预算 3.0 A100-h：远低于上限。

分析阶段不接触 GPU 主机，只读拉回的产物；`analysis/run_manifest.json` 列出 6 条正式轨迹的 run_id / commit / 签名 / checkpoint hash 与复用参照的 hash。

