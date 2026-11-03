# LCRC-P1：CHEM8 × MULTICONTRIB8 交互的可重复性与 COMBO 的跨划分稳定性

- 轮次编号：`lcrc-generalization-factorial-p1-v1`
- 协议：`PROTOCOL.md`（冻结于 2026-11-03，`analysis/protocol_freeze.json` 记录 8 个分析代码文件与 20 个必需单元的 SHA-256）
- 执行：`rr` / host `res` / 池 `res-gpu0`、`res-gpu1` / NVIDIA A100-SXM4-40GB / driver 550.163.01 / torch 2.5.1+cu124 / python 3.12.14
- GPU 账：**1.1381 / 2.0 A100 GPU-小时**（10 个报告单元 + 2 个作废重跑，见 §8）
- Test 口径：`DEVELOPMENT_TEST`。Test 从未参与训练、从未参与 epoch/checkpoint 选择；**本报告不把 Test 称作独立最终确认集**

---

## 0. 判决（先说结论）

1. **Q1（交互是否 seed 可重复）：不可重复。** Test 上交互项 `I` 三个 seed 为 −0.0207 / −0.0090 / **+0.0109**（MSE 单位），符号在第 3 个 seed 翻转；三个 seed 的 scaffold-cluster bootstrap 95% 区间全部跨 0。Validation 上更明显：+0.1496 / +0.0396 / **−0.1495**，均值 +0.0132，seed 间标准差 0.1512。**没有证据支持"CHEM8 + MULTICONTRIB8 存在稳定协同"这一假设。**
2. **Q2（COMBO 相对 BASE 是否跨划分稳定）：Test RMSE 7 个 (split, seed) 里 6 个改善，平均 −0.0189（122）/ −0.0305（123）/ −0.0088（124）**；MAE 6/7 个方向一致。但逐 seed 的 95% 区间只在 2/7 个 Test 组合里排除 0，且 random124 seed 0 出现 **+0.0067 的劣化**。方向"大体一致、逐点不显著"，不能宣称"已建立"。
3. **Q3（尾部）：CVaR90-AE 的改善方向一致（Test 6/7 为负）但没有一个区间排除 0**；而 `LABEL_SELECTED_POSTHOC_TAIL` 分析里 6/7 个 Test 组合的区间排除 0 —— 这一差异主要来自"用 BASE 自己的误差选样本"的选择效应，**不可以**用 post-hoc tail 的结果替代尾部风险主张。
4. **Q4（增益是否集中于少数分子）：不集中。** Test 上 COMBO 改善的分子比例 0.488–0.569（几乎就是 50/50），在净收益为正的 6 个 Test 组合里，前 5 个分子对净 SSE 下降的贡献份额为 0.38–0.77；但单个分子仍可造成 −1.2 ~ −4.0 SSE 的反向损失。**这是"面上小幅、尾部仍有重尾损伤"的形态，不是"少数英雄式翻盘"。**
5. **Q5（是否只对 scaffold-unseen 有效）：不是，方向甚至相反。** random123 seed 1 的 Test 上，seen 组 ΔRMSE = −0.0543 [−0.0799, −0.0260]，unseen 组 = +0.0084 [−0.0385, +0.0594]；random122 三个 seed 的 seen/unseen 差值几乎相同。最接近 Train 的 `max_train_tanimoto` Q4 在 122 Test 上最稳定地改善。**COMBO 的增益更像"熟悉化学空间内的小幅稳定改进"，不是 unseen 泛化能力。**

一句话：**本轮没有复现出可重复的 CHEM8×MULTICONTRIB 交互，也没有建立 COMBO 的统计显著优势；能站住的只有"COMBO 在三个划分的 Test 上都不比 BASE 差、平均约 1.6%–5.2% 的 RMSE 小幅改善、且该改善不是靠少数分子或 unseen 子群"这一弱陈述。**

---

## 1. 本轮实际做了什么

| 划分 | seed | BASE | CHEM8 | MULTICONTRIB8 | COMBO |
|---|---|---|---|---|---|
| random122 | 0 | 复用 | 复用 | 复用 | 复用 |
| random122 | 1 | 复用 | **新训** | **新训** | 复用 |
| random122 | 2 | 复用 | **新训** | **新训** | 复用 |
| random123 | 0 | **新训** | — | — | 复用 |
| random123 | 1 | **新训** | — | — | **新训** |
| random124 | 0 | **新训** | — | — | 复用 |
| random124 | 1 | **新训** | — | — | **新训** |

- 共 20 个报告单元：10 个复用（`lcrc_performance_longrun_v1/analysis/runs/`），10 个本轮新训。
- 新训只补协议列出的缺口：random122 的 CHEM8 s1/s2、MULTICONTRIB8 s1/s2；random123/124 的 BASE s0/s1 与 COMBO s1。
- 没有新增模型结构、损失、特征；没有调任何超参；没有更换或挑选 seed；没有开始 P2。
- 复用的严格性由机器检查保证（`analysis/integrity_checks.json`，449 项全通过）：
  - `training_code_hashes_identical_across_all_units`：20 个单元的 11 个训练代码文件（`cl_train.py`、`cl_chem8.py`、`cl_model.py`、`cl_data.py`、`cl_common.py`、`prep/*`、`v2/mb2_*`）哈希**完全相同**（`n_distinct_code_hash_sets = 1`）。历史上不同 commit SHA 只是仓库层面的提交差异，训练相关文件逐字节相同。
  - `frozen_recipe_identical_across_all_units`：0 处不符（360 epoch、AdamW、峰值 3e-4 → 最小 1e-5 余弦、wd 1e-3、batch 128、dropout 0.10、clip 5.0、`shuffle_generator_seed = 917000 + seed`）。
  - `execution_regime_identical_across_all_units`：同一主机 `a100-2`、同一 torch/CUDA、同一 GPU 型号（1 个环境）。
  - 每个划分的 `target_norm` 与其 Train 划分严格一致且 `source: train_split_only`；BASE 输入缓存在同划分内逐单元一致；扩展层零初始化在每单元 `passed: true`。

---

## 2. Q1 —— 四臂交互项是否跨 seed 可重复

定义（冻结）：`I = MSE(COMBO) − MSE(CHEM8) − MSE(MULTICONTRIB8) + MSE(BASE)`，单位 logD²，负值 = 有利于组合。

| 部分 | seed | I | scaffold-bootstrap 95% CI | 主效应 CHEM8 | 主效应 MULTICONTRIB8 |
|---|---|---|---|---|---|
| Valid | 0 | **+0.1496** | [−0.0076, +0.4483] | −0.0184 | −0.1519 |
| Valid | 1 | **+0.0396** | [−0.0283, +0.1548] | −0.0031 | −0.1248 |
| Valid | 2 | **−0.1495** | [−0.5316, +0.0508] | −0.0123 | −0.0474 |
| Test | 0 | **−0.0207** | [−0.0528, +0.0120] | −0.0051 | +0.0040 |
| Test | 1 | **−0.0090** | [−0.0453, +0.0237] | +0.0038 | −0.0128 |
| Test | 2 | **+0.0109** | [−0.0249, +0.0483] | −0.0074 | −0.0275 |

- Validation：1/3 为负，均值 +0.0132，seed 间 sd 0.1512。
- Test：2/3 为负，均值 −0.0063，seed 间 sd 0.0159。
- 6 个区间**没有一个**排除 0。

**判读。** 交互项在两个部分都发生符号翻转，因此"交互方向"不是可重复量。更要紧的是交互背后的实质：

- Validation 上 MULTICONTRIB8 的主效应很大（均值 −0.108 MSE，即 RMSE 0.6258 vs BASE 0.7085），但 COMBO 并没有把它保住（seed 0/1 的 `I` 为正，说明 COMBO 的 MSE 比"加法预测"更差，即组合**破坏**了 MULTICONTRIB8 的 Validation 收益）。
- Test 上 MULTICONTRIB8 的主效应缩到 −0.0121，COMBO−BASE 的 MSE 下降 ≈ −0.0217，其中交互项贡献 −0.0063（约 29%）、两个主效应贡献约 71%。也就是说 **Test 上 COMBO 的小幅优势并不是"协同"带来的**，而且这个 −0.0063 的交互本身符号不稳。

**因此 Q1 的答案是：不成立。** 上一轮（P0，`combo_attribution.json`）观察到的是"Validation 上的交互方向不一致"；本轮把三 seed 的 Test 列进来之后，结论更强：交互项在 Test 上同样符号翻转，量级（0.009–0.021 logD²）只是同单元重训噪声（§7：MSE 约 0.004）的 2–5 倍，且没有一个 seed 的区间排除 0。

**混合来源的额外保留。** seed 0 的四臂全部复用（同一批次训练），seed 1/2 的 CHEM8、MULTICONTRIB8 是本轮新训、而 BASE、COMBO 是复用。因此"交互的 seed 间方差"里混有"新训 vs 复用"这一无法完全分离的成分。用 §7 实测的重训噪声量化：Validation 上单臂重训可动 0.010 RMSE ≈ 0.014 MSE，四臂组合到 `I` 的量级约 0.028 MSE，而观测到的 seed 间 sd 是 0.151 —— **Validation 的交互方差远大于重训噪声，是真实的 seed 依赖**；Test 上重训噪声对 `I` 的贡献约 0.008 MSE，观测 sd 0.0159，只有约 2 倍，**Test 的交互方差无法与重训噪声干净区分**。

---

## 3. Q2 —— COMBO 相对 BASE 的稳定性（random122/123/124）

### 3.1 逐单元指标（Test，raw logD 单位）

| 划分 | 臂 | seed | RMSE | MAE | P90-AE | CVaR90-AE | 来源 |
|---|---|---|---|---|---|---|---|
| 122 | BASE | 0/1/2 | 0.5704 / 0.5634 / 0.5779 | 0.4241 / 0.4243 / 0.4277 | 0.8787 / 0.8996 / 0.9556 | 1.3101 / 1.2621 / 1.3015 | 复用 |
| 122 | CHEM8 | 0/1/2 | 0.5660 / 0.5667 / 0.5715 | 0.4245 / 0.4225 / 0.4173 | 0.8575 / 0.8890 / 0.9627 | 1.3042 / 1.2645 / 1.3135 | 复用/新/新 |
| 122 | MULTICONTRIB8 | 0/1/2 | 0.5739 / 0.5519 / 0.5536 | 0.4260 / 0.4142 / 0.4154 | 0.9122 / 0.8511 / 0.8827 | 1.3007 / 1.2448 / 1.2468 | 复用/新/新 |
| 122 | COMBO | 0/1/2 | 0.5510 / 0.5472 / 0.5567 | 0.4120 / 0.4101 / 0.4112 | 0.8905 / 0.8576 / 0.8782 | 1.2443 / 1.2247 / 1.2713 | 复用 |
| 123 | BASE | 0/1 | 0.5747 / 0.5984 | 0.4155 / 0.4370 | 0.9337 / 0.9901 | 1.3011 / 1.3290 | 新训 |
| 123 | COMBO | 0/1 | 0.5374 / 0.5748 | 0.3912 / 0.4235 | 0.8619 / 0.9853 | 1.2194 / 1.2803 | 复用/新 |
| 124 | BASE | 0/1 | 0.5152 / 0.5636 | 0.3912 / 0.4303 | 0.8575 / 0.9068 | 1.1271 / 1.2298 | 新训 |
| 124 | COMBO | 0/1 | 0.5219 / 0.5393 | 0.3981 / 0.4067 | 0.8366 / 0.9172 | 1.1494 / 1.1997 | 复用/新 |

seed 平均（122 为 3 个 seed，123/124 各 2 个）：Test RMSE 122 0.5706 → 0.5517（−3.3%），123 0.5866 → 0.5561（−5.2%），124 0.5394 → 0.5306（−1.6%）。

### 3.2 配对差（COMBO − BASE，负值有利于 COMBO）

| 划分 | seed | Test ΔRMSE [95% CI] | Test ΔMAE [95% CI] | Test ΔCVaR90-AE | Test 改善分子比例 | Valid→Test 同号 |
|---|---|---|---|---|---|---|
| 122 | 0 | −0.0194 [−0.0461, +0.0050] | −0.0121 [−0.0316, +0.0064] | −0.0659 | 0.529 | 是 / 是 / 否 |
| 122 | 1 | −0.0162 [−0.0407, +0.0096] | −0.0142 [−0.0330, +0.0050] | −0.0374 | 0.529 | 是 / 是 / 是 |
| 122 | 2 | −0.0212 [−0.0491, +0.0081] | −0.0165 [−0.0406, +0.0074] | −0.0302 | 0.521 | 是 / 是 / 是 |
| 123 | 0 | **−0.0373 [−0.0659, −0.0080]** | **−0.0243 [−0.0443, −0.0042]** | −0.0817 | 0.555 | 否 / 是 / 否 |
| 123 | 1 | −0.0236 [−0.0504, +0.0058] | −0.0135 [−0.0359, +0.0086] | −0.0488 | 0.517 | 是 / 是 / 是 |
| 124 | 0 | **+0.0067 [−0.0164, +0.0299]** | **+0.0069 [−0.0108, +0.0246]** | +0.0223 | 0.488 | 否 / 否 / 否 |
| 124 | 1 | **−0.0243 [−0.0492, −0.0025]** | **−0.0236 [−0.0435, −0.0052]** | −0.0301 | 0.569 | 是 / 是 / 否 |

（末列依次为 RMSE / MAE / CVaR90 的 Validation↔Test 同号性。）

- **RMSE**：Test 上 6/7 个 (split, seed) 为负，均值 −0.0189 / −0.0305 / −0.0088；95% 区间只有 2/7 排除 0。Validation→Test 同号 5/7（123 s0、124 s0 翻转）。
- **MAE**：Test 上 6/7 为负；同号 6/7（124 s0 翻转）。**MAE 方向一致性强于 RMSE**，但效应更小（−0.012 ~ −0.024）。
- **P90-AE / CVaR90-AE**：见 §4。
- 复用的 random123/124 COMBO seed 0 恰好是两个最好（0.5374、0.5219）与一个最差（124 seed 0 的 +0.0067 反超）的源头；**seed 0 的两个比较是"复用 COMBO vs 新训 BASE"，其差值里混有 §7 的重训噪声（Test ≈ 0.002–0.004 RMSE）**，因此 124 seed 0 的 +0.0067 属于"不可区分于噪声的反向"，而不是"已证实的恶化"。

**判读。** 一致的弱方向是：**COMBO 在三个划分的 Test 上平均比 BASE 好 1.6%–5.2%，MAE 方向 6/7 一致；但没有任何单一 seed 达到 95% 显著性，且 124 seed 0 为反向。** 这与上一轮 `lcrc_performance_longrun_v1` 的 COMBO lipo 专家效应（Test 0.5429 vs BASE 0.5620，五个 run）量级一致，本轮把该效应放到三个划分上检验，结论是"方向稳定、强度弱"。

---

## 4. Q3 —— CVaR90-AE 与 post-hoc 尾部的区别

| 划分 | Test CVaR90-AE BASE→COMBO（seed 均值） | 配对 ΔCVaR90 区间排除 0 | `LABEL_SELECTED_POSTHOC_TAIL` Test ΔRMSE（区间排除 0 的个数） |
|---|---|---|---|
| 122 | 1.2912 → 1.2467 | 0/3 | −0.111 / −0.114 / −0.128（3/3） |
| 123 | 1.3150 → 1.2498 | 0/2 | −0.171 / −0.156（2/2） |
| 124 | 1.1784 → 1.1745 | 0/2 | −0.048 / −0.090（1/2） |

- **CVaR90-AE**（每个模型自己最差 `ceil(10%×420)=42` 个分子的平均绝对误差）在 Test 上 6/7 为负、幅度 −0.030 ~ −0.082，但 **7 个区间全部跨 0**。P90-AE 同样：122 0.9113→0.8755、123 0.9619→0.9236、124 0.8822→0.8769，逐 seed 区间基本跨 0。
- **post-hoc tail**（用同 seed BASE 的绝对误差取 top 10%）给出的 ΔRMSE 是 −0.05 ~ −0.17，比整体效应大 3–8 倍，Test 上 7/7 为负且 6/7 区间排除 0。这个放大是**选择效应**：被选中的分子本来就是 BASE 犯大错的分子，任何与之相关的模型在该子集上都会显得更好。**因此该结果标记为 `LABEL_SELECTED_POSTHOC_TAIL`，不可作为"尾部修复能力"的证据，也不可替代 P90/CVaR90 主张。**

**判读（Q3）：** 尾部方向的证据是"一致但未建立"。**本轮不能宣称 COMBO 降低了尾部风险。** 唯一可以说的是：CVaR90-AE 的点估计在三个划分、6/7 个 (split, seed) 上都不比 BASE 差，且没有出现"整体变好、尾部变坏"的反向模式。

---

## 5. Q4 —— 增益是集中在少数分子，还是弥散

逐分子平方误差差 `ΔSE = SE(COMBO) − SE(BASE)`，负 = COMBO 更好。

| 划分 | 部分 | seed | 净 SSE 变化 | 改善/恶化分子数 | 前 5 分子占净下降份额 | 单分子最大反向损失 |
|---|---|---|---|---|---|---|
| 122 | Valid | 0/1/2 | +8.71 / +37.10 / +87.84 | 224/196、208/212、216/204 | 1.07 / 0.15 / 0.07 | −6.24 / −40.94 / −86.01 |
| 122 | Test | 0/1/2 | +9.14 / +7.53 / +10.10 | 222/198、222/198、219/201 | 0.63 / 0.77 / 0.73 | −2.84 / −1.91 / −1.31 |
| 123 | Test | 0/1 | +17.43 / +11.63 | 233/187、217/203 | 0.38 / 0.68 | −4.01 / −2.53 |
| 124 | Test | 0/1 | −2.91 / +11.26 | 205/215、239/181 | −1.83 / 0.45 | −1.20 / −2.14 |

- Test 上**改善分子比例 0.488–0.569**：几乎是 50/50 的弥散改善，而不是"少数分子翻盘"。净收益为正的 6 个 Test 组合中，前 5 分子的份额 0.38–0.77 < 1，说明净收益由较多分子的中等改善累积而成（124 seed 0 净收益为负，该份额在那里无意义）。
- 但反向尾部很重：单个分子可贡献 −1.2 ~ −4.0 SSE，前 5 个负向分子合计 −5.3 ~ −9.1 SSE（123 Test seed 0：净 +17.4，最差单分子 −4.0，前 5 负向合计 −9.1）。
- Validation 上形态完全不同：seed 1/2 的净收益几乎全部由**单个分子**（row 2087 等）造成（前 5 份额 0.15/0.07，最大单分子损失 −40.9/−86.0 与净收益同量级）。这说明 Validation 上的"大效应"是重尾单分子驱动的，随划分而消失。

**判读（Q4）：** Test 上 COMBO 的优势是**弥散的中等改善 + 少量重尾损伤**；不存在"少数恢复 + 其余受损"的结构。这也解释了为什么 CVaR90 的方向改善但幅度不显著：改善并不专属于尾部。

---

## 6. Q5 —— 是否只对 scaffold-unseen 分子有效

`scaffold_seen` = 该分子 Bemis–Murcko scaffold（无环分子用规范 SMILES）出现在**同一划分的 Train** 中。

| 划分 | seed | Test seen：ΔRMSE [95% CI] | Test unseen：ΔRMSE [95% CI] |
|---|---|---|---|
| 122 | 0 | −0.0216 [−0.0420, +0.0020] | −0.0185 [−0.0618, +0.0229] |
| 122 | 1 | −0.0153 [−0.0435, +0.0149] | −0.0171 [−0.0571, +0.0248] |
| 122 | 2 | −0.0195 [−0.0608, +0.0178] | −0.0231 [−0.0634, +0.0177] |
| 123 | 0 | **−0.0441 [−0.0828, −0.0043]** | −0.0308 [−0.0717, +0.0120] |
| 123 | 1 | **−0.0543 [−0.0799, −0.0260]** | +0.0084 [−0.0385, +0.0594] |
| 124 | 0 | +0.0056 [−0.0258, +0.0343] | +0.0078 [−0.0263, +0.0408] |
| 124 | 1 | −0.0206 [−0.0496, +0.0095] | −0.0279 [−0.0649, +0.0077] |

- random122 三个 seed 的 seen/unseen 差值几乎为零（−0.022 vs −0.019、−0.015 vs −0.017、−0.020 vs −0.023）。
- random123 seed 1 是唯一"显著"的子群结果，而它指向 **seen** 组（−0.0543，区间排除 0），unseen 组为 **+0.0084**。
- 描述性辅助：`max_train_tanimoto` Q4（最接近 Train 的四分位）在 122 Test 上两个 seed 的区间都排除 0（−0.0345、−0.0416），而 Q1（最远）不稳定。

**判读（Q5）：** 没有任何 scaffold-unseen 特异性；若说有倾向，是**偏向已见 scaffold / 高相似度分子**。这与"COMBO 只是在小幅改良局部化学空间内的拟合"一致，而不是"补上了 unseen 泛化缺口"。

---

## 7. 同单元重训噪声（本轮意外获得的硬约束）

第一批 GPU 单元因为包装脚本把产物写到了错误的目录名下（见 §8），两个单元各自**从零重训了一次**。这给出了同 split × 同 arm × 同 seed × 同代码哈希 × 同主机 × 同 driver/torch 的**直接重训差**（`analysis/replay_nondeterminism.json`）：

| 单元 | Validation RMSE 差 | Test RMSE 差 | best epoch 差 |
|---|---|---|---|
| 122 / CHEM8 / s1 | 0.0113 | 0.0018 | 11 |
| 122 / CHEM8 / s2 | 0.0094 | 0.0036 | 41 |

- **Validation 可动 ~0.010 RMSE，Test 可动 ~0.003 RMSE**（即 MSE 层面约 0.014 / 0.004）。
- 后果：任何 ≤0.003 RMSE 的 Test 差异、≤0.010 的 Validation 差异，**都不能归因于臂本身**，无论 bootstrap 把分子重采样多少次。
- 本报告所有 bootstrap 区间只量化"评测集抽样不确定性"，**不包含也不消解重训噪声**；两者必须分开读。

---

## 8. 执行、预算与偏差

### 8.1 GPU 账（`analysis/gpu_ledger.json`）

| 项目 | 值 |
|---|---|
| 预算上限 | 2.0 A100 GPU-小时 |
| 已计入（含作废重跑） | **1.1381** |
| 余额 | 0.8619 |
| 报告单元 | 10 个新训（`n_completed = 10`） |
| 作废重跑 | 2 个（路径命名缺陷，**全部计入**） |
| 单 run 训练墙钟 | 270–400 s |

10 个复用单元的训练时间记在 `lcrc_performance_longrun_v1` 的预算，不重复计入本上限。

### 8.2 作废与冻结修订（全部记录在 `analysis/protocol_freeze.json` 的 `revisions`）

1. **rev 1**：`run_unit.py` 把新训产物写到复用单元的命名方案下（`analysis/runs/<arm>_<split>_s<seed>`）。只影响目录名，不影响训练逻辑/配方/seed/指标。发现时**尚未对任何新数字做分析**；两个单元重训，作废单元的时间计入预算。
2. **rev 2**：GPU 账本必须把作废尝试计入，而不是静默丢弃。
3. **rev 3**：`rr pull` 把 run bundle 解到 `.rr/pulled/<run_id>/result/<declared path>`，新增 `dispatch.py sync` 把声明的子树搬进 `analysis/runs/<unit_key>`；纯产物搬运。
4. **rev 4**：账本重试记录去重、（实验名, 是否作废）成对只计一次费用、作废单元使用自身记录的 run id 与墙钟。修好后总计入 1.1381 h。

四次修订都发生在**任何 Q1–Q5 数值被解读之前**，且都不改变训练、指标或统计定义。

### 8.3 已知偏差（与协议一致）

- 复用的 10 个单元此前已在 P0/P4b 轮次中被查看过；本轮不是对它们的盲测。
- 122/123/124 三个划分来自同一个 4,200 分子池（`Lipop.csv.gz`，SHA-256 `34280f0a…`），**三者的 Test 不是独立队列**；跨划分只能作"稳定性检查"，不能当作三次独立重复。
- seed 不是独立分子；bootstrap 以 Bemis–Murcko scaffold 为簇（Test 上 350–361 簇 / 420 分子，最大簇占 1.2%–2.9%，三个划分均 `low_cluster_support: false`）。
- 部分 (split, seed) 混合了复用与新训单元（§2），差值里含 §7 的重训噪声。
- `source_audit.json` 中 `git_dirty: true` 指的是**本次构建自身重写了分析输出文件**；每个训练单元记录的 `git_dirty_tracked` 均为 `false`，且 20 个单元训练代码哈希完全一致。
- 未删除任何离群分子，未按 Test 挑选 epoch 或 checkpoint。

### 8.4 完整性

`analysis/integrity_checks.json`：**449 项检查，449 通过，0 失败**，覆盖：CSV 唯一性/列名、分子解析、每个划分的尺寸与行哈希与冻结 run 记录一致、划分内行唯一、划分间行隔离、三划分覆盖全池、跨划分同 row_id 同 SMILES/同标签、target 归一化 Train-only 且同划分内一致、BASE 输入缓存一致、共享初始化哈希与 BASE 一致、扩展层零初始化、代码哈希/配方/执行环境跨单元一致、单元可用性。

### 8.5 Test 使用台账

Test 只在 checkpoint 冻结后被读取一次（`test_labels_read: true` 仅出现在训练结束后的评估阶段，`test_used_for_development: false`，`validation_used_for_checkpoint_selection: true`）。Test 数值一律标注 `DEVELOPMENT_TEST`，**不构成独立最终确认**。

---

## 9. 与既有结论的关系

- `lcrc_performance_longrun_v1` 的判决是 `NO_LCRC_B_SIGNAL_THIS_ROUND`；本轮不推翻它：CHEM8/MULTICONTRIB 的收益在 Validation 上存在、在 Test 上塌缩（MULTICONTRIB8 主效应 −0.108 → −0.012 MSE）。
- P0 的结论"rooted-radius-2 支撑度对 LCRC−PNA 差距无可测增量"与本轮一致：本轮不引入任何支撑度加权模型。
- 本轮**新增**的可复用事实：(a) 该训练配方在同单元重训下 Test RMSE 可动 ~0.003；(b) MULTICONTRIB8 的 Validation 优势不能迁移到 Test（约 9 倍衰减）；(c) COMBO 在三个划分的 Test 上一致优于 BASE 的方向性，但强度落在重训噪声的 5–10 倍、且逐 seed 不显著。

## 10. 不可声称的内容

- 不可声称"CHEM8 × MULTICONTRIB8 存在协同"或"不存在协同"——只能说在本轮 3 个 seed 上**方向不可重复**。
- 不可声称 COMBO 显著优于 BASE，也不可声称在 random124 上显著劣于 BASE。
- 不可声称 COMBO 降低尾部风险（CVaR90 区间全跨 0）；不可用 post-hoc tail 数字替代。
- 不可声称 COMBO 改善 unseen 泛化；证据方向相反。
- 不可把三个划分当作三次独立复现；不可把 Validation 与 Test 的数字放在同一显著性口径下比较（Validation 参与 checkpoint 选择，系统性乐观）。
- 不可把本轮的 Test 称为"独立最终确认集"。

## 11. 产物清单

| 文件 | 内容 |
|---|---|
| `PROTOCOL.md` | 冻结协议（问题、单元清单、复用规则、配方、统计口径、预算） |
| `REPORT.md` | 本文件 |
| `analysis/protocol_freeze.json` | 协议 SHA-256、8 个分析代码文件哈希、运行前审计、4 次修订记录 |
| `analysis/source_audit.json` | CSV/划分 pickle/代码/每单元产物文件的 SHA-256 |
| `analysis/run_reuse_manifest.json` | 20 个单元的来源、复用判据、逐文件哈希 |
| `analysis/metrics_by_split_seed_arm.csv` | 40 行（20 单元 × Valid/Test）的 RMSE/MAE/P90-AE/CVaR90-AE/SSE/bias |
| `analysis/per_molecule_predictions.csv` | 16,800 行逐分子预测、误差、子群标签 |
| `analysis/factorial_interaction.json` | `I` 的逐 seed 点估计、主效应、bootstrap 区间、方向一致性 |
| `analysis/paired_uncertainty.json` | COMBO−BASE 的 ΔRMSE/ΔMAE/ΔP90/ΔCVaR 与 scaffold-cluster bootstrap |
| `analysis/tail_risk.json` | CVaR90/P90 与 `LABEL_SELECTED_POSTHOC_TAIL` 两套尾部口径 |
| `analysis/scaffold_subgroups.json` | seen/unseen 与 heavy-atom/TPSA/max-Train-Tanimoto 四分位 |
| `analysis/concentration_diagnostics.json` | 逐分子 ΔSE 集中度、前 5/前 10 份额、最差单分子损失 |
| `analysis/integrity_checks.json` | 449 项完整性检查 |
| `analysis/gpu_ledger.json` | 12 条执行记录（10 报告 + 2 作废）与预算余额 |
| `analysis/replay_nondeterminism.json` | §7 的重训噪声测量 |
| `analysis/dispatch_plan.json` | 5 波 10 单元的发射计划与命令 |
| `analysis/manifest.json` | 产物清单与哈希 |
| `analysis/runs/<unit_key>/` | 每单元 8 个文件（checkpoint、curve、train/valid/test 预测、train.json、provenance）；按 `.gitignore` 不入库，哈希在 `source_audit.json` |
| `code/` | `p1_common.py`、`p1_stats.py`、`p1_sources.py`、`p1_analysis.py`、`run_unit.py`、`run_analysis.py`、`dispatch.py`、`write_freeze.py` |
| `tests/` | 27 个 CPU 测试（指标算术、CVaR 尾长、行对齐、bootstrap 确定性、冻结常量、单元清单完整性、冻结驱动哈希/配方） |

## 12. 下一步（明确不在本轮）

不做 P2、不做新架构、不做超参搜索、不因为"124 seed 0 反向"或"交互符号翻转"而重跑或换 seed。若要把 COMBO 的小幅优势提升到可主张的强度，需要的是**多 seed × 多划分的重复训练**（每个 (split, arm) ≥ 5 个 seed，以使 seed 间方差与 §7 的重训噪声分开），这属于下一轮设计决策，不在本轮范围内。
