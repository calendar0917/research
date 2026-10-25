# REPORT — P3-Chromatic Bond Transformer v0

**Round:** `zinc-p3-chromatic-bond-v0` · **Track:** structural-sufficiency
**Commit:** `55bc122` · **Seed:** 0 · **Official test:** 未读取（`official_test_loaded: false`）

## 1. 五个问题的直接回答

1. **仓库里是否已有等价实现？没有。** DBT/GOS 是 per-head softmax + per-channel value gate；GOS 的 `Ksm` per-channel 分支从未运行，且 5-D mask/`dim=1` query-axis softmax 均不合法；GRIT/STA/NR/BCCA/RC/ESOC 均为 per-head 或单头。公开 CSA 层确实在 `num_heads*out_dim` 特征轴上做 key-axis softmax，但本轮是 P3 bond object 上的研究组合，**不是 CSA 复现**，0.056 不是本轮预期。
2. **per-channel chromatic attention 是否胜过标量控制？正式问题未被有效检验；描述性结果是没有。** C 的 best-valid 为 **0.081068**，S 为 **0.077129**；`gain = S_best - C_best = -0.003939`（C 更差）。配对 bootstrap 的 C−S 平均 AE 差为 **+0.003939**，95% CI `[-0.004712, +0.017630]`，C 优于 S 的重采样概率仅 **0.296**。正式分类因稳定性屏幕为 `UNSTABLE_OR_UNDERFIT`（见 §5），不能据此写“color 原理上失败”。
3. **是否达到/超过历史 DCC-L？C 没有；S 在同 seed 锚点附近，但未达 4-seed 均值。** C best 0.081068、W20 0.082853，均未达 competitive 线（best ≤0.078537、W20 ≤0.080）。S best 0.077129、W20 0.078399，达到 competitive 线，并与 DCC-L seed-0（0.078537）统计上不可区分（S−DCC-L 平均 AE 差 `-0.001408`，95% CI `[-0.012297, +0.006314]`）。历史 4-seed 均值 **0.075990** 仍低于 S，不能声称 family-level 超越。
4. **结构/环特征是否被使用？是，但未转化为 color 的泛化优势。** C 学到的 `rel_channel` 非零且通道多样：各层 `channel_dev_tv` 约 **0.055–0.079**，有效 relation-logit 标准差约 **0.195–0.288**，权重 RMS 约 **0.115–0.155**；S 对应量约为 0。冻结权重下关闭 ring 列（前 256 个 valid 分子）使 C 的 MAE 从 0.11089 升至 0.13486（**+0.02397**），S 从 0.08428 升至 0.11948（**+0.03520**），说明环特征确实参与预测。代价在尾部：`y ≤ -3`（n=65）上 C 的 MAE **0.28677**，S 为 **0.18868**（C 差 **+0.09809**）；普通组（n=935）C 略好（0.06677 vs 0.06937）。
5. **决策与下一步？** 冻结决策规则先运行稳定性屏幕，C 在 epoch 34 出现 `|Δvalid| = 0.11008 > 0.05`，S 在 epoch 24 出现 `0.09686 > 0.05`，故正式分类为 **`UNSTABLE_OR_UNDERFIT`**，本轮性能问题“未被有效检验”，不重跑、不调参、不换 seed。描述性 counterfactual（不改变正式 decision）为 **`COMPETITIVE_COLOR_UNPROVEN`**：S 竞争、C 不竞争且无 color gain。下一轮若要检验同一机制，应事先把稳定性标准改为**晚窗口/平滑曲线**（例如 epoch ≥200 或滑动中位数），并保留多 seed；本轮不做事后修改。

## 2. 运行与预算

| 项 | C | S |
|---|---:|---:|
| 参数量 | 477,017 | 477,017 |
| 初始状态 SHA-256 | `01a545e6…` | `01a545e6…` |
| best-valid | **0.081068** @376 | **0.077129** @335 |
| W20（起点） | 0.082853 (532) | 0.078399 (555) |
| tail-100 median | 0.082914 | 0.078442 |
| final valid | 0.082954 | 0.078456 |
| train eval-mode MAE @best | 0.01225 | 0.01777 |
| A100·h（run-level） | 0.764 | 0.806 |

- 34/34 个 CPU check 全部通过（参数、init 相等、C/S 初值相等、梯度可达、环特征、batch 隔离、等变性、边界、train/eval、attention、cache、hash、test 拒绝）。
- 关系 cache：train 6,492,991 个 bond pair，valid 644,472 个；ring pair 比例 0.13640/0.13672；cycle 长度覆盖 3–18。
- 计时投影 **2.065 A100·h**（soft 3.5 / hard 4.0），正式两卡实际 **1.570 A100·h**；无 `torch.compile`、无 EMA/SWA/soup、无 supervised DCC-L init。
- DCC-L seed-0 历史锚点：best 0.078537 @489，W20 0.079082，tail-100 0.079112；4-seed 均值 0.075990 仅作背景线。

## 3. C/S 配对与分组

| 指标 | C | S | C−S |
|---|---:|---:|---:|
| ordinary MAE (`y > -3`, n=935) | 0.06677 | 0.06937 | **−0.00261** |
| tail MAE (`y ≤ -3`, n=65) | 0.28677 | 0.18868 | **+0.09809** |
| DCC-L worst-10 移除后 MAE (n=990) | 0.06536 | 0.06644 | −0.00108 |

C 在普通组略好，但尾部严重退化，触发 rule 1 的 catastrophic 条件（tail Δ >0.010）。DCC-L worst-10 移除后 C/S 接近，说明 DCC-L 的优势部分集中在少数困难分子；该移除是描述性固定集合，不是正式检验。

## 4. 机制诊断（best checkpoint，前 256 个 valid 分子）

- C：`rel_channel` 权重 RMS 0.115–0.155；`channel_dev_tv` 0.055–0.079；有效 relation-logit std 0.195–0.288；ring-off 使 MAE +0.02397，平均 |Δpred| 0.0511，最大 2.132。
- S：`channel_dev_tv ≈ 3.5e-8`、有效 std 0.0（数值零，符合 within-head mean 定义）；ring-off 使 MAE +0.03520，平均 |Δpred| 0.0556，最大 2.694。
- 两臂 train/eval 预测最大差均为 0.0；attention 行和误差 <6e-7，非相邻 bond pair 仍有 0.84–0.91 的 attention mass；C 的通道多样性是真实的，不是初始化残留。

## 5. 稳定性屏幕与决策

| 曲线 | max \|Δvalid\| after ep10 | 位置 | after ep200 |
|---|---:|---:|---:|
| C | 0.11008 | ep34 | 0.02204 |
| S | 0.09686 | ep24 | 0.03350 |
| DCC-L seed-0（历史） | 0.10583 | ep25 | 0.02352 |

冻结规则要求 `max |Δvalid| > 0.05 (t>10)` 即 `UNSTABLE_OR_UNDERFIT`。C/S 均触发；而且历史 DCC-L 曲线也会触发同一屏幕，说明它主要捕捉高 LR 早期的 valid 噪声，而不是本轮机制特有的不稳定。两臂在 epoch 200 后最大跳变已降至 0.022/0.034，tail-100 稳定，train eval-mode MAE 远低于 0.03；但**正式 decision 仍按预注册规则执行**，不能事后放宽。

`analysis/decision.json` 的 category 为 `UNSTABLE_OR_UNDERFIT`；`analysis/decision_notes.json` 仅记录上述屏幕校准和 counterfactual，不改变 decision。

## 6. 限制

单 seed pilot；C/S 共享参数与初值但**不是 effective-capacity matched**；valid 是暴露开发集；DCC-L 锚点是历史值、非配对随机变量；ring-off 只在 256 个固定 valid 分子上计算；本轮不声称 CSA 复现或机制新颖性。所有数字仅对 valid，未读取 official test。

**一句话结论：** 在这个预注册 pilot 中，per-channel chromatic attention 没有表现出优于标量控制的证据；正式问题被早期 valid 跳变触发的稳定性屏幕判为“未有效检验”，因此 category 是 `UNSTABLE_OR_UNDERFIT`，而不是“color 被证伪”。
