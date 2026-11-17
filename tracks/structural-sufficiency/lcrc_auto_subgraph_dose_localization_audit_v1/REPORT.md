# REPORT — lcrc-auto-subgraph-dose-localization-audit-v1

对 `lcrc-auto-subgraph-faithfulness-pilot-v1`（交付提交 `147b1d81`）的**只读审计轮**。本轮不重新选择 Top-1、不重新训练、不更换 seed、不修改架构、不扩大搜索范围；只把随机对照的匹配量从 `n_s2_internal` 改成真实被门控移除的 `n_s2_affected`，并测量屏蔽响落在哪些原子根节点上。

## 0. 一页结论（先回答五个问题）

### `ogbg-molesol`

1. **原对照的剂量匹配是否存在实质性口径错误？** `SUBSTANTIVE`。原匹配量是 `n_s2_internal`（两条键都在 S 内），真实被移除的是 `n_s2_affected`（至少一条键在 S 内）。原 Top-1 的 occurrence 数中位 internal = 12.0000, affected = 22.0000；原已接受对照中 22.8% 在修正后的 affected 口径下超出 25% 容差（三划分合并；分划分见 §2；affected 相对差中位 0.1333）。
2. **VALID（修正对照）**：median effect ratio = 1.2699（bootstrap CI low 1.2236），ratio>1 = 83.6%，median effect advantage = 0.4352，判定 `ATTENUATED`。
3. **TEST（修正对照）**：median effect ratio = 1.2618（bootstrap CI low 1.2067），ratio>1 = 79.4%，median effect advantage = 0.5263，判定 `ATTENUATED`。
4. **Top-1 是否仍明显优于随机对照和局部启发式？** 修正对照下的 Valid/Test 判定为 `ATTENUATED` / `ATTENUATED`；Top-1 的真实屏蔽效应大于局部启发式的分子比例为 70.3%（Valid）/ 67.2%（Test）。
5. **影响集中在自身根节点还是扩散？** train: 外部根节点绝对响应占比中位 0.1248，>0.5 的比例 9.4%，判定 `LOCAL_CONCENTRATED`；valid: 外部根节点绝对响应占比中位 0.2123，>0.5 的比例 7.8%，判定 `LOCAL_CONCENTRATED`；test: 外部根节点绝对响应占比中位 0.1398，>0.5 的比例 3.1%，判定 `LOCAL_CONCENTRATED`。

### `ogbg-mollipo`

1. **原对照的剂量匹配是否存在实质性口径错误？** `SUBSTANTIVE`。原匹配量是 `n_s2_internal`（两条键都在 S 内），真实被移除的是 `n_s2_affected`（至少一条键在 S 内）。原 Top-1 的 occurrence 数中位 internal = 6.0000, affected = 14.0000；原已接受对照中 36.8% 在修正后的 affected 口径下超出 25% 容差（三划分合并；分划分见 §2；affected 相对差中位 0.2000）。
2. **VALID（修正对照）**：median effect ratio = 1.8267（bootstrap CI low 1.2984），ratio>1 = 71.7%，median effect advantage = 0.3022，判定 `SURVIVES`。
3. **TEST（修正对照）**：median effect ratio = 1.2894（bootstrap CI low 0.9834），ratio>1 = 59.7%，median effect advantage = 0.1780，判定 `NOT_SUPPORTED`。
4. **Top-1 是否仍明显优于随机对照和局部启发式？** 修正对照下的 Valid/Test 判定为 `SURVIVES` / `NOT_SUPPORTED`；Top-1 的真实屏蔽效应大于局部启发式的分子比例为 50.0%（Valid）/ 45.3%（Test）。
5. **影响集中在自身根节点还是扩散？** train: 外部根节点绝对响应占比中位 0.3476，>0.5 的比例 29.7%，判定 `MIXED`；valid: 外部根节点绝对响应占比中位 0.3366，>0.5 的比例 23.4%，判定 `MIXED`；test: 外部根节点绝对响应占比中位 0.3460，>0.5 的比例 21.9%，判定 `MIXED`。

> 说明：第 3 条回答中的两个数据集各自独立，不要把 molesol 和 mollipo 的数字混成一行。

## 1. 边界、身份与不可比声明

* 模型：`lcrc_readout_context_pilot_small_v1` **Arm A = LCRC-C1**，`sum_v f(h_v) + rho([g, mean(h), mean(h^2), log1p(n_atoms)])`；checkpoint / 轨迹 SHA-256 与原轮记录一致，本轮**没有训练**。
* 数据：官方 `ogbg-molesol` / `ogbg-mollipo` 冻结缓存与官方 `get_idx_split()`；每划分 64 个 row id 与 原轮 `samples.json` 逐项相同（重新按同一规则选择并比对哈希）。
* 解释对象：原轮已选定的 Top-1 物理键集合被**逐字复用**；梯度、发现分数、搜索日志均从原轮读取，并逐分子重算核对（任一不一致即停止）。
* 干预仍是 `CONDITIONAL_S1_S2_EVIDENCE_MASK`（`m[e] ∈ {0,1}` 的冻结前向重算），**不是**化学删键/删基团/完整反事实。Test 仅作 `DEVELOPMENT_TEST` 报告。
* **两轮对照协议不同**：`pilot_internal_matched` 与 `corrected_affected_matched` 的匹配量、接受规则都不同。下文所有对比都保留协议列，旧数字不能冒充已修正结果。

## 2. Q1 — 原对照的剂量匹配口径

S2 门控权重是 `w[p] = m[e1] * m[e2]`，因此 `m[e]=0` 时实际被移除的 occurrence 条件是 `e1 in S OR e2 in S`。原轮用于匹配对照的是 `n_s2_internal`（`e1 in S AND e2 in S`），两者相差的正是**边界 occurrence**（恰好一条键在 S 内）。

| dataset | split | Top1 n_s2_internal | Top1 n_s2_affected | pilot null: median internal rel diff | pilot null: median affected rel diff | pilot null: frac outside affected 25% | corrected null: median affected rel diff | corrected null: frac outside affected 25% | corrected null: distinct draws | corrected null: frac draws rejected by affected rule |
|---|---|---|---|---|---|---|---|---|---|---|
| `ogbg-molesol` | train | 12.0000 | 20.0000 | 0.0000 | 0.1111 | 13.8% | 0.1111 | 0.0% | 952.0000 | 23.0% |
| `ogbg-molesol` | valid | 12.0000 | 24.0000 | 0.0000 | 0.1429 | 22.4% | 0.1000 | 0.0% | 1469.0000 | 33.6% |
| `ogbg-molesol` | test | 12.0000 | 22.0000 | 0.0000 | 0.1667 | 30.0% | 0.1111 | 0.0% | 1766.0000 | 43.9% |
| `ogbg-mollipo` | train | 6.0000 | 14.0000 | 0.0000 | 0.2000 | 42.4% | 0.1429 | 0.0% | 2082.0000 | 58.7% |
| `ogbg-mollipo` | valid | 6.0000 | 16.0000 | 0.0000 | 0.2000 | 36.0% | 0.1429 | 0.0% | 1795.0000 | 46.6% |
| `ogbg-mollipo` | test | 6.0000 | 14.0000 | 0.0000 | 0.1667 | 32.2% | 0.1429 | 0.0% | 1562.0000 | 36.4% |

`pilot null` 列来自对原轮**已接受**对照键集合的重新计数（不改动原轮存储的效果值）；`corrected null` 列是本轮新生成的对照。修正对照的 `frac outside affected 25%` 恒为 0，因为 25% 容差就是接受准则；它右侧的“候选拒绝率”才说明该准则有多严格。

原轮**按它自己声明的规则执行是正确的**（pilot null 的 internal 相对差最大 0.25，见 `analysis/old_control_dose.csv`）；问题在于它匹配的量不是门控实际移除的量：修正口径下原对照的 affected 剂量与 Top-1 相差一个边界项，`boundary = affected - internal` 的典型规模见 `median_affected_minus_internal`。

## 3. Q2 — 修正后的 Valid / Test 对照

| dataset | split | protocol | matched n | success | median effect ratio | ratio CI95 | ratio>1 | median advantage | advantage CI95 | median random abs(delta) |
|---|---|---|---|---|---|---|---|---|---|---|
| `ogbg-molesol` | valid | pilot | 63.0000 | 0.9844 | 1.2931 | [1.2104, 1.4314] | 84.1% | 0.4629 | [0.2996, 0.6118] | 1.3494 |
| `ogbg-molesol` | valid | corrected | 61.0000 | 0.9531 | 1.2699 | [1.2236, 1.4850] | 83.6% | 0.4352 | [0.2748, 0.6255] | 1.1667 |
| `ogbg-molesol` | test | pilot | 61.0000 | 0.9531 | 1.3859 | [1.2460, 1.5503] | 83.6% | 0.5330 | [0.3209, 0.8588] | 1.4248 |
| `ogbg-molesol` | test | corrected | 63.0000 | 0.9844 | 1.2618 | [1.2067, 1.4273] | 79.4% | 0.5263 | [0.2881, 0.6829] | 1.2910 |
| `ogbg-mollipo` | valid | pilot | 64.0000 | 1.0000 | 1.5677 | [1.2829, 2.2240] | 71.9% | 0.2945 | [0.1074, 0.5227] | 0.4937 |
| `ogbg-mollipo` | valid | corrected | 60.0000 | 0.9375 | 1.8267 | [1.2984, 2.0489] | 71.7% | 0.3022 | [0.1030, 0.6094] | 0.5618 |
| `ogbg-mollipo` | test | pilot | 64.0000 | 1.0000 | 1.2864 | [0.8747, 1.6725] | 56.2% | 0.1407 | [-0.0972, 0.3599] | 0.5233 |
| `ogbg-mollipo` | test | corrected | 62.0000 | 0.9688 | 1.2894 | [0.9834, 1.7205] | 59.7% | 0.1780 | [-0.0088, 0.3553] | 0.4803 |

`effect_ratio = |delta(Top1)| / (median_e |delta(control_e)| + 1e-12)`；`effect_advantage = |delta(Top1)| - median_e |delta(control_e)|`。bootstrap：2000 次、以分子为重采样单位、seed 20261117；它只描述固定 checkpoint / 固定数据集下的抽样不确定性，不是跨 seed 稳定性。

### 3.1 修正后判定（沿用原轮冻结的投入规则）

| dataset | split | protocol | verdict | median ratio | ratio>1 | ratio CI low | effect floor frac | matched n |
|---|---|---|---|---|---|---|---|---|
| `ogbg-molesol` | valid | pilot | ATTENUATED | 1.2931 | 84.1% | 1.2104 | 100.0% | 63.0000 |
| `ogbg-molesol` | valid | corrected | ATTENUATED | 1.2699 | 83.6% | 1.2236 | 100.0% | 61.0000 |
| `ogbg-molesol` | test | pilot | ATTENUATED | 1.3859 | 83.6% | 1.2460 | 100.0% | 61.0000 |
| `ogbg-molesol` | test | corrected | ATTENUATED | 1.2618 | 79.4% | 1.2067 | 100.0% | 63.0000 |
| `ogbg-mollipo` | valid | pilot | SURVIVES | 1.5677 | 71.9% | 1.2829 | 98.4% | 64.0000 |
| `ogbg-mollipo` | valid | corrected | SURVIVES | 1.8267 | 71.7% | 1.2984 | 98.4% | 60.0000 |
| `ogbg-mollipo` | test | pilot | NOT_SUPPORTED | 1.2864 | 56.2% | 0.8747 | 100.0% | 64.0000 |
| `ogbg-mollipo` | test | corrected | NOT_SUPPORTED | 1.2894 | 59.7% | 0.9834 | 100.0% | 62.0000 |

判定规则（跑前冻结）：`SURVIVES` = 四项条件全过且 ratio 的 bootstrap CI low > 1；`ATTENUATED` = median ratio > 1 且 CI low ≥ 1 但未全过；其余 `NOT_SUPPORTED`。四项条件 = median ratio ≥ 1.5、ratio>1 比例 ≥ 70%、1% std 效应底线比例 ≥ 50%、匹配分子数 ≥ max(32, 80%×已解释分子)。

这里的 `SURVIVES/ATTENUATED/NOT_SUPPORTED` 是本轮为**两种协议并排比较**而冻结的三档映射，不是原轮的 `PILOT_SIGNAL / WEAK_OR_INCONCLUSIVE` 二档标签；原轮自己的判定见其 `DECISION.md`。

### 3.2 分子级配对变化（同一分子、两个对照协议）

| dataset | split | paired molecules | median ratio delta (corrected - pilot) | ratio delta CI95 | median advantage delta | frac ratio decreased |
|---|---|---|---|---|---|---|
| `ogbg-molesol` | train | 44.0000 | -1.4382e-07 | [-0.0446, 8.3204e-08] | -3.6956e-07 | 54.5% |
| `ogbg-molesol` | valid | 61.0000 | 1.9850e-06 | [-1.3904e-07, 0.0598] | 5.8514e-07 | 45.9% |
| `ogbg-molesol` | test | 61.0000 | -4.3706e-06 | [-0.1535, 7.9169e-07] | -6.1593e-07 | 54.1% |
| `ogbg-mollipo` | train | 53.0000 | 0.0757 | [1.0127e-06, 0.1670] | 0.0390 | 34.0% |
| `ogbg-mollipo` | valid | 60.0000 | 0.0019 | [-0.0809, 0.1009] | 0.0074 | 46.7% |
| `ogbg-mollipo` | test | 62.0000 | 3.4965e-04 | [-6.3490e-07, 0.0611] | 9.2244e-05 | 43.5% |

| dataset | split | matched under both protocols | pilot-only | corrected-only |
|---|---|---|---|---|
| `ogbg-molesol` | train | 44 | 13 | 1 |
| `ogbg-molesol` | valid | 61 | 2 | 0 |
| `ogbg-molesol` | test | 61 | 0 | 2 |
| `ogbg-mollipo` | train | 53 | 11 | 0 |
| `ogbg-mollipo` | valid | 60 | 4 | 0 |
| `ogbg-mollipo` | test | 62 | 2 | 0 |

| dataset | split | paired molecules with identical accepted control set | median control-set Jaccard | paired molecules with a changed set | median ratio delta on changed sets | CI95 |
|---|---|---|---|---|---|---|
| `ogbg-molesol` | train | 34.1% | 0.798 | 29 | -0.1112 | [-0.1905, 7.7316e-08] |
| `ogbg-molesol` | valid | 14.8% | 0.600 | 52 | 0.0239 | [-1.1531e-07, 0.1340] |
| `ogbg-molesol` | test | 9.8% | 0.524 | 55 | -0.0210 | [-0.2405, 1.1206e-06] |
| `ogbg-mollipo` | train | 1.9% | 0.455 | 52 | 0.0778 | [1.7236e-06, 0.1692] |
| `ogbg-mollipo` | valid | 1.7% | 0.455 | 59 | 0.0039 | [-0.0831, 0.1246] |
| `ogbg-mollipo` | test | 1.6% | 0.455 | 61 | 6.9852e-04 | [-1.4462e-06, 0.0611] |

配对差的中位数接近 0 是预期的：同一分子的随机流不变，只有接受准则改变，因此多数分子的候选集合高度重叠甚至完全相同。聚合中位数的移动同时包含**分子集合变化**（上表 pilot-only / corrected-only）、**对照集合变化**（本表）与两个协议下不同的对照值，三者不能混为一谈。

## 4. Q3 — Top-1 相对随机对照与局部启发式

| dataset | split | median Top1 abs(delta) | median heuristic abs(delta) | median advantage | Top1 > heuristic (pilot record) | Top1 > heuristic (recomputed Top1) |
|---|---|---|---|---|---|---|
| `ogbg-molesol` | train | 1.7105 | 1.3116 | 0.3173 | 67.2% | 71.9% |
| `ogbg-molesol` | valid | 1.6945 | 1.3612 | 0.5698 | 70.3% | 70.3% |
| `ogbg-molesol` | test | 2.1905 | 1.4563 | 0.5703 | 67.2% | 67.2% |
| `ogbg-mollipo` | train | 0.8406 | 1.1774 | -0.0817 | 42.2% | 43.8% |
| `ogbg-mollipo` | valid | 0.8213 | 0.8083 | 0.0210 | 50.0% | 50.0% |
| `ogbg-mollipo` | test | 0.6820 | 0.7532 | -0.1270 | 45.3% | 45.3% |

局部输出启发式 `B(S) = sum_v |f(h_v)| / sqrt(|S|)` 的子图与效果**直接复用原轮记录**（不重新搜索）；Top-1 的 `|delta|` 用本轮重算值（与原轮在冻结容差内一致）。

## 5. Q4 — 屏蔽响应的根节点定位

| dataset | split | n | outside fraction median | IQR | >0.5 | <=0.25 | median inside abs response | median outside abs response | verdict |
|---|---|---|---|---|---|---|---|---|---|
| `ogbg-molesol` | train | 64 | 0.1248 | [0.0571, 0.2820] | 9.4% | 71.9% | 1.6011 | 0.2672 | LOCAL_CONCENTRATED |
| `ogbg-molesol` | valid | 64 | 0.2123 | [0.0994, 0.3792] | 7.8% | 57.8% | 1.6718 | 0.4317 | LOCAL_CONCENTRATED |
| `ogbg-molesol` | test | 64 | 0.1398 | [0.0710, 0.2920] | 3.1% | 68.8% | 2.0467 | 0.3004 | LOCAL_CONCENTRATED |
| `ogbg-mollipo` | train | 64 | 0.3476 | [0.1921, 0.5195] | 29.7% | 37.5% | 0.7578 | 0.3700 | MIXED |
| `ogbg-mollipo` | valid | 64 | 0.3366 | [0.1816, 0.4814] | 23.4% | 39.1% | 0.8750 | 0.3730 | MIXED |
| `ogbg-mollipo` | test | 64 | 0.3460 | [0.2378, 0.4628] | 21.9% | 32.8% | 0.7666 | 0.3901 | MIXED |

### 5.1 S2 剂量与根节点分布

| dataset | split | median n_s2_internal | median n_s2_boundary | median n_s2_affected | internal / total | boundary / total | affected / total | boundary / affected | affected occurrences rooted outside |
|---|---|---|---|---|---|---|---|---|---|
| `ogbg-molesol` | train | 10.0000 | 7.0000 | 18.0000 | 0.3258 | 0.2500 | 0.6667 | 0.5000 | 0.2143 |
| `ogbg-molesol` | valid | 12.0000 | 12.0000 | 24.0000 | 0.2000 | 0.2660 | 0.4655 | 0.5584 | 0.2500 |
| `ogbg-molesol` | test | 12.0000 | 10.0000 | 22.0000 | 0.2000 | 0.1695 | 0.3871 | 0.5000 | 0.2111 |
| `ogbg-mollipo` | train | 6.0000 | 10.0000 | 14.0000 | 0.0632 | 0.1160 | 0.1650 | 0.6667 | 0.3274 |
| `ogbg-mollipo` | valid | 6.0000 | 10.0000 | 16.0000 | 0.0789 | 0.1354 | 0.1921 | 0.6000 | 0.2958 |
| `ogbg-mollipo` | test | 6.0000 | 10.0000 | 14.0000 | 0.0572 | 0.1118 | 0.1623 | 0.6429 | 0.3063 |

这里同时报告两种 S2 覆盖率：**internal coverage**（`n_s2_internal / n_s2_total`）与 **affected coverage**（`n_s2_affected / n_s2_total`）。原轮报告的 “S2 coverage” 是前者，不能当作被门控移除的 occurrence 比例。

### 5.2 与 Top-1 键数、梯度分数、真实屏蔽效应的对应关系

| dataset | split | Spearman vs k | vs abs(score A) | vs abs(delta_total) | vs n_s2_affected | vs boundary/affected | vs affected rooted outside |
|---|---|---|---|---|---|---|---|
| `ogbg-molesol` | train | -0.019 | 0.031 | -0.311 | 0.353 | 0.702 | 0.756 |
| `ogbg-molesol` | valid | -0.207 | 0.139 | -0.568 | 0.529 | 0.677 | 0.703 |
| `ogbg-molesol` | test | -0.053 | 0.419 | -0.688 | 0.638 | 0.618 | 0.725 |
| `ogbg-mollipo` | train | -0.302 | 0.105 | -0.022 | 0.135 | 0.579 | 0.637 |
| `ogbg-mollipo` | valid | -0.202 | 0.133 | -0.243 | 0.078 | 0.579 | 0.608 |
| `ogbg-mollipo` | test | -0.312 | 0.263 | -0.081 | -0.127 | 0.472 | 0.540 |

按 Top-1 键数分组的 outside fraction：

| dataset | split | Top1 k | n | outside fraction median | median abs(delta_total) | median n_s2_affected | median boundary/affected |
|---|---|---|---|---|---|---|---|
| `ogbg-molesol` | train | 2 | 10 | 0.1418 | 0.4029 | 5.0000 | 0.5833 |
| `ogbg-molesol` | train | 3 | 2 | 0.1379 | 1.0423 | 12.0000 | 0.4375 |
| `ogbg-molesol` | train | 4 | 7 | 0.1156 | 1.6444 | 14.0000 | 0.3333 |
| `ogbg-molesol` | train | 5 | 11 | 0.1214 | 1.3993 | 16.0000 | 0.4286 |
| `ogbg-molesol` | train | 6 | 34 | 0.1330 | 2.3011 | 23.0000 | 0.4773 |
| `ogbg-molesol` | valid | 2 | 4 | 0.2137 | 0.4176 | 10.0000 | 0.8000 |
| `ogbg-molesol` | valid | 3 | 1 | 0.6624 | 0.0950 | 20.0000 | 0.8000 |
| `ogbg-molesol` | valid | 4 | 6 | 0.2524 | 1.1802 | 17.0000 | 0.6307 |
| `ogbg-molesol` | valid | 5 | 5 | 0.3757 | 1.2027 | 24.0000 | 0.5833 |
| `ogbg-molesol` | valid | 6 | 48 | 0.1633 | 2.2081 | 25.0000 | 0.5359 |
| `ogbg-molesol` | test | 2 | 1 | 0.2390 | 0.4932 | 10.0000 | 0.8000 |
| `ogbg-molesol` | test | 3 | 4 | 0.3103 | 0.4137 | 13.0000 | 0.5125 |
| `ogbg-molesol` | test | 4 | 2 | 0.1020 | 0.8985 | 18.0000 | 0.3333 |
| `ogbg-molesol` | test | 5 | 7 | 0.1205 | 1.2227 | 20.0000 | 0.5000 |
| `ogbg-molesol` | test | 6 | 50 | 0.1375 | 2.4320 | 22.0000 | 0.5000 |
| `ogbg-mollipo` | train | 2 | 23 | 0.5171 | 0.6345 | 10.0000 | 0.8000 |
| `ogbg-mollipo` | train | 3 | 14 | 0.2701 | 0.8503 | 12.0000 | 0.6458 |
| `ogbg-mollipo` | train | 4 | 6 | 0.2110 | 0.4153 | 15.0000 | 0.5714 |
| `ogbg-mollipo` | train | 5 | 8 | 0.3476 | 1.1819 | 20.0000 | 0.5417 |
| `ogbg-mollipo` | train | 6 | 13 | 0.2965 | 0.8719 | 24.0000 | 0.5455 |
| `ogbg-mollipo` | valid | 2 | 17 | 0.4175 | 0.7885 | 10.0000 | 0.8000 |
| `ogbg-mollipo` | valid | 3 | 13 | 0.2119 | 0.6004 | 12.0000 | 0.5000 |
| `ogbg-mollipo` | valid | 4 | 8 | 0.2542 | 0.9359 | 15.0000 | 0.5982 |
| `ogbg-mollipo` | valid | 5 | 8 | 0.2712 | 1.1821 | 21.0000 | 0.5227 |
| `ogbg-mollipo` | valid | 6 | 18 | 0.3269 | 1.0632 | 26.0000 | 0.5420 |
| `ogbg-mollipo` | test | 2 | 23 | 0.4506 | 0.5495 | 12.0000 | 0.8333 |
| `ogbg-mollipo` | test | 3 | 8 | 0.2814 | 0.9505 | 13.0000 | 0.6458 |
| `ogbg-mollipo` | test | 4 | 8 | 0.3650 | 0.9176 | 17.0000 | 0.5857 |
| `ogbg-mollipo` | test | 5 | 6 | 0.2672 | 1.2496 | 17.0000 | 0.4365 |
| `ogbg-mollipo` | test | 6 | 19 | 0.2971 | 0.5190 | 26.0000 | 0.5385 |

相关性只用于描述这批 64×3 分子上的对应关系；样本量小、单 checkpoint，不作为机制证明。按指示，本轮**不**用这些指标自动推导“模型架构必须修改”。

## 6. Q5 — 本轮支持 / 不支持 / 仍无法判断

**支持**

* 原轮的随机对照确实按 `n_s2_internal`（两条键都在 S 内）匹配，而冻结门控 `w[p]=m[e1]*m[e2]` 实际移除的是 `n_s2_affected`（至少一条键在 S 内）；两者相差边界 occurrence。
* 在冻结前向上，`n_s2_affected` 恰好等于被置零的 S2 权重数（逐分子 1→0 集合比对通过），修正后的对照可以在同一模型上复现。
* 修正对照下逐分子的 effect ratio / advantage，以及分子级 bootstrap 区间；两数据集、三划分分开报告。
* 屏蔽引起的局部读出变化在根节点上的分解：子图内根节点与外部根节点各自的绝对响应、外部占比与闭合检查。

**不支持**

* 任何跨 seed / 跨 checkpoint 的稳定性结论（本轮只有一个冻结 checkpoint）。
* 任何化学反事实解释：`atom_feat` / `atom_scalar` / `global` / 原子数特征并未被门控，屏蔽不是删键或删基团。
* “模型不使用结构”或“架构必须修改”这类结论；本轮没有做架构对照。
* 新的 Top-1、新的搜索空间、新的架构或新的数据集。

**仍无法判断**

* 修正后的判定是否能外推到每划分 64 个分子之外。
* 外部根节点的响应对预测质量是否必要（本轮没有做去掉外部响应的对照）。
* 换一种匹配零假设（例如按边界比例或按权重质量匹配）是否会改变排序；本轮只跑了 affected 剂量匹配。

## 7. 工程门禁与复现

* ENGINEERING_GATE: `PASS`
* 原轮指标重放（用原轮逐分子记录重算并与原 `main_table.csv` 比对）：`PASS`
* 协议哈希：`2d638f7d3f5aed55ce46642e74d1467c8caa24166c77fce6fd7a46d57d1ad5fc`
* PROTOCOL.md sha256：`43f0836f3a55af4ae073e1d07632047ce661b112da8d2e1fbb3e719236204214`
* Top-1 屏蔽预测与原轮逐分子比对：384 个分子中 0 个超出冻结容差；最大绝对偏差 signed_delta 2.9565e-06、delta_L 2.4637e-06、delta_rho 1.1554e-06、pred_raw 1.4443e-06（CPU↔GPU 浮点差，容差按原轮口径）。
* 运行 provenance（commit、工作区状态、代码/协议/输入哈希、环境与峰值内存）见 `analysis/run_manifest.json`，不在本文件重复嵌入 commit（否则每次提交都会改变报告本身）。
* 上游 results commit：`2e202646b1560e857f92549e7fa72af55fe6ca85`

| dataset | gate | passed | detail |
|---|---|---|---|
| `ogbg-molesol` | DATA_IDENTITY | yes | frozen caches + official split sizes/disjointness |
| `ogbg-molesol` | PILOT_SAMPLE_IDENTITY | yes | the pilot's 64 row ids per split are reproduced exactly |
| `ogbg-molesol` | TRAINING_IDENTITY | yes | checkpoint/trajectory SHA-256, 360-epoch 3-split coverage, selected-epoch argmin |
| `ogbg-molesol` | MODEL_EQUIVALENCE_ALL_ONES | yes | all-ones gates reproduce frozen arm A on every split |
| `ogbg-molesol` | OCCURRENCE_MAPPING | yes | occurrence map rebuilt and verified again on the smoke molecules |
| `ogbg-molesol` | TOP1_SOURCE_IDENTITY | yes | pilot Top-1 bond set, gradient, discovery score, search-log entry and masked effect replay |
| `ogbg-molesol` | S2_AND_OR_ACCOUNTING | yes | masking S removes exactly the e1-in-S OR e2-in-S occurrences on the frozen forward |
| `ogbg-molesol` | S2_WEIGHT_COUNT | yes | the real s2_weight tensor has exactly n_s2_affected zeros and n_s2_unaffected ones |
| `ogbg-molesol` | LOCALIZATION_CLOSURE | yes | sum_v delta_c_v == delta_L and delta_L + delta_rho == delta_total |
| `ogbg-molesol` | PARAMETER_HASH_UNCHANGED | yes | no parameter was modified by the audit path |
| `ogbg-molesol` | PARAMETER_BUDGET | yes | 228674 total / 15298 readout |
| `ogbg-mollipo` | DATA_IDENTITY | yes | frozen caches + official split sizes/disjointness |
| `ogbg-mollipo` | PILOT_SAMPLE_IDENTITY | yes | the pilot's 64 row ids per split are reproduced exactly |
| `ogbg-mollipo` | TRAINING_IDENTITY | yes | checkpoint/trajectory SHA-256, 360-epoch 3-split coverage, selected-epoch argmin |
| `ogbg-mollipo` | MODEL_EQUIVALENCE_ALL_ONES | yes | all-ones gates reproduce frozen arm A on every split |
| `ogbg-mollipo` | OCCURRENCE_MAPPING | yes | occurrence map rebuilt and verified again on the smoke molecules |
| `ogbg-mollipo` | TOP1_SOURCE_IDENTITY | yes | pilot Top-1 bond set, gradient, discovery score, search-log entry and masked effect replay |
| `ogbg-mollipo` | S2_AND_OR_ACCOUNTING | yes | masking S removes exactly the e1-in-S OR e2-in-S occurrences on the frozen forward |
| `ogbg-mollipo` | S2_WEIGHT_COUNT | yes | the real s2_weight tensor has exactly n_s2_affected zeros and n_s2_unaffected ones |
| `ogbg-mollipo` | LOCALIZATION_CLOSURE | yes | sum_v delta_c_v == delta_L and delta_L + delta_rho == delta_total |
| `ogbg-mollipo` | PARAMETER_HASH_UNCHANGED | yes | no parameter was modified by the audit path |
| `ogbg-mollipo` | PARAMETER_BUDGET | yes | 228674 total / 15298 readout |

| dataset | device | wall s | compute s | peak RSS MB | molecules |
|---|---|---|---|---|---|
| `ogbg-molesol` | cpu | 31.12 | 27.58 | 704.5 | 192 |
| `ogbg-mollipo` | cpu | 58.35 | 44.38 | 797.1 | 192 |

* 执行顺序：`preflight`（身份门禁 + 8 分子全链路 smoke）→ 两个数据集的 `audit` → CPU `aggregate`。smoke 记录不进入任何报告指标。
* 单 seed、单 checkpoint、无训练；bootstrap 只反映分子抽样不确定性。

## 8. 产物

| path | content |
|---|---|
| `PROTOCOL.md` | 跑前冻结的审计定义与判读规则 |
| `analysis/preflight.json` | 身份、样本、等价性、门禁与 smoke 记录 |
| `analysis/old_vs_corrected.csv` | 两种对照协议并排的主要指标（不回填、不混池） |
| `analysis/matched_controls.csv` | 修正对照逐子图记录（Top-1 + 全部匹配候选） |
| `analysis/old_control_dose.csv` | 原轮已接受对照在修正口径下的剂量重算 |
| `analysis/localization.csv` | 逐分子根节点响应定位与闭合检查 |
| `analysis/localization_summary.json` | 分组统计、Spearman 对应关系、按 k 分箱 |
| `analysis/bootstrap.json` | 分子级 bootstrap 区间（两种协议） |
| `analysis/decision.json` | 冻结规则下的判定与五问回答 |
| `analysis/run_manifest.json` | commit、协议哈希、输入/代码哈希、环境与内存记录 |
| `results/<dataset>/<split>/records.json` | 逐分子完整审计记录（可重放） |
| `results/<dataset>/<split>/candidates.npz` | 全部随机候选的三个剂量计数与接受标记 |
| `results/<dataset>/<split>/atom_response.csv` | 逐原子 `c_v` 与 `delta_c_v` |

