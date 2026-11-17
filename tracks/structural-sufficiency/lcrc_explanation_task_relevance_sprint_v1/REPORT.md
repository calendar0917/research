# LCRC Explanation Task Relevance Sprint v1 — Report

**Round**: `lcrc_explanation_task_relevance_sprint_v1`  
**Dataset**: `ogbg-mollipo` official scaffold split (3360 / 420 / 420)  
**Frozen pairs**: 823 (791 Train–Train, 5 Train–Valid, 27 Train–Test, 0 both-non-Train)  
**Models**: frozen LCRC-C1 (`L_lipophilicity/Rf`, epoch 124) and GINE-G0 (`G0_lipophilicity/Rf`, epoch 183); WC_LOGP oracle models `L_WC_LOGP` (epoch 333) and `T_WC_LOGP` (epoch 355). No retraining, no new seeds, no explainer training, no re-optimisation of `effect_ratio`.  
**Execution**: `res` / A100, `res-gpu0` = Main A, `res-gpu1` = Main B, both under `rr` at commit `a16b2224`; 0.048 GPU·h total (budget 2 GPU·h per worker).  
**Test**: all Test numbers are `DEVELOPMENT_TEST`, not a held-out final benchmark.

---

## 0. 五个问题的直接回答（中文摘要）

### Q1. 自动发现的结构是否比随机结构更接近编辑位置？

**LCRC：没有达到可支持的程度。** 相对同分子、同 bond 数随机 connected control 的编辑区富集：

| 指标 | LCRC | GINE | LCRC−GINE |
|---|---:|---:|---:|
| Atom Jaccard advantage | 0.0157 [−0.0048, 0.0373] | **0.0219 [0.0002, 0.0453]** | −0.0063 [−0.0336, 0.0195] |
| Atom recall advantage | 0.0168 [−0.0135, 0.0487] | **0.0504 [0.0150, 0.0895]** | −0.0336 [−0.0777, 0.0106] |
| Bond recall advantage | 0.0103 [−0.0160, 0.0376] | **0.0389 [0.0108, 0.0712]** | −0.0286 [−0.0657, 0.0075] |
| Attachment-boundary hit advantage | 0.0309 [−0.0167, 0.0837] | **0.0838 [0.0261, 0.1545]** | −0.0529 [−0.1177, 0.0142] |

括号为 shared-endpoint component bootstrap 95% CI（231 个 component，2000 次，seed 20261117）。LCRC 的四个区间都跨 0；GINE 的三个区间下界 > 0。**结论：GINE 的 Top-1 比同尺寸随机 connected subgraph 更接近编辑位置；LCRC 没有证明这一点。**

一个必须同时报告的反例基线：使用 LCRC 自己的 atom local output 做同一个 k=2..6、beam=12 的搜索（`LOCAL_LCRC`），编辑区富集显著更强：

- Atom Jaccard advantage **0.0619 [0.0322, 0.0954]**
- Atom recall advantage **0.0752 [0.0342, 0.1207]**
- Bond recall advantage **0.0871 [0.0484, 0.1300]**

所以“LCRC 的表示不能定位编辑区”不是唯一解释；**当前的 gradient-subgraph 打分/搜索本身是一个明显的瓶颈**。

### Q2. 发现的结构是否比随机结构更好地捕获 WC_LOGP contribution change？

**是，LCRC 和 GINE 都优于随机；但优势是零膨胀的均值效应，且局部 atom-output 基线远强于两者。**

| 模型 | Oracle enrichment（Top-1 − random median） | Oracle Top-1 capture | 候选优于 random median 的比例 | 中位 enrichment |
|---|---:|---:|---:|---:|
| LCRC | **0.1186 [0.0895, 0.1485]** | 0.1221 [0.0922, 0.1524] | 16.7% | 0.0 |
| GINE | **0.0984 [0.0665, 0.1342]** | 0.1050 [0.0726, 0.1414] | 14.3% | 0.0 |
| LOCAL_LCRC | **0.3244 [0.2876, 0.3649]** | 0.3289 [0.2917, 0.3697] | 34.1% | 0.0 |

- LCRC 的 `n_s2_affected`-matched 补充 control：enrichment **0.1182 [0.0881, 0.1490]**，与同尺寸 control 几乎相同，说明这个结果不是 S2 dose 不匹配造成的。
- LCRC−GINE 差异：**+0.0201 [−0.0213, 0.0611]**，不显著。
- 关键 caveat：中位 enrichment = 0，只有 16.7% / 14.3% 的候选超过随机中位数；正的均值来自少数“命中大块 contribution mass”的候选。不能把它写成“大多数候选都能捕获 WC_LOGP 变化”。

WC_LOGP contribution change 的空间分布也解释了为什么这件事难：在 823 个 pair 上，**92.6% 的 |δc| mass 落在 added/deleted variable fragment 上，core 只有 7.4%**（median variable fraction = 1.0；54.3% 的 pair core 完全为 0）。也就是说，oracle 定位目标与结构编辑区高度重叠，但候选只有 k=2..6 个 bond，常常覆盖不到那个 fragment。

### Q3. LCRC 与 GINE 各自有什么优势？是否被预测准确率混淆？

**预测层面是“混合”的，不是一方全面更好：**

| 指标 | LCRC | GINE | LCRC−GINE |
|---|---:|---:|---:|
| Valid RMSE（全分子） | 0.7069 | **0.6161** | — |
| Test RMSE（DEVELOPMENT_TEST） | 0.7293 | **0.6800** | — |
| Pair \|Δŷ−Δy\| | **0.2584 [0.2305, 0.2873]** | 0.2866 [0.2617, 0.3131] | **−0.0283 [−0.0462, −0.0085]** |
| Direction accuracy（\|Δy\|≥0.1） | **0.9088 [0.8847, 0.9313]** | 0.8925 [0.8627, 0.9165] | +0.0163 [−0.0049, 0.0377] |

两者都通过 `prediction_accurate` 门槛（Valid/Test RMSE ≤ 0.80，direction ≥ 0.75）。LCRC 的 **pair change** 更准，GINE 的 **absolute molecule RMSE** 更好。

**结构解释层面：**

- Oracle：LCRC 0.1186 vs GINE 0.0984；LCRC 点估计略高，但 paired difference 不显著。
- Real-label：GINE 的 atom Jaccard / bond recall / boundary hit 三个 advantage 的 CI 下界都 > 0；LCRC 三个都跨 0。LCRC−GINE 的点估计全为负，但三个 paired CI 都跨 0。
- **没有证据支持 LCRC 有 unique advantage**；`NO_UNIQUE_LCRC_ADVANTAGE_SHOWN`。

**是否被预测准确率混淆？** 没有看到 LCRC 的定位失败主要由“预测更差”解释：

- LCRC 在 pair change/direction 上其实更好；
- LCRC 的 oracle enrichment 在 `pair_accurate` 组为 0.1188，在 `pair_inaccurate` 组为 0.1182；direction-correct 组 0.1172，direction-wrong 组 0.0898（wrong 组 n=67，CI 较宽）；
- 见 Q4：LCRC 最大的失败格是“预测准确但结构没定位到”，而不是“预测错”。

因此，**LCRC 的 real-label 定位失败更像方法/候选结构问题，而不是预测准确率问题**。但 GINE 的 absolute RMSE 更好，不能完全排除它对 GINE real-label 优势的贡献；这不是一个受控的 architecture-only 比较。

### Q4. 失败更多来自预测不准的 pair，还是预测准确但定位错误的 pair？

按 `pair_accurate = |Δŷ−Δy| ≤ 0.25`，`localisation_success = atom_recall ≥ 0.5 或 attachment-boundary hit`（pair 两端点取平均）划分 823 个 pair：

| 模型 | accurate & localized | **accurate & mislocalized** | inaccurate & localized | inaccurate & mislocalized |
|---|---:|---:|---:|---:|
| LCRC | 164 (19.9%) | **331 (40.2%)** | 106 (12.9%) | 222 (27.0%) |
| GINE | 193 (23.5%) | 235 (28.6%) | 145 (17.6%) | 250 (30.4%) |

- **LCRC：最大的单一失败格是“预测准确但定位错误”（331/823 = 40.2%）**；在全部 659 个非“双成功”pair 中，accurate-mislocalized 占 50.2%，inaccurate 占 49.8%，几乎对半。
- **GINE：失败更多来自预测不准**；395/630 = 62.7% 的非双成功 pair 属于 inaccurate，accurate-mislocalized 只有 235 个。
- 只看方向可评估的 735 个 pair：LCRC 289 accurate-mislocalized vs 301 inaccurate；GINE 202 vs 365。结论方向一致：**LCRC 的瓶颈更偏定位，GINE 的瓶颈更偏预测**。

### Q5. 证据支持哪个 architecture direction，还是应该暂停 claim？

- **LCRC-specific structural explanation：暂停 claim。** LCRC 的 prediction 是准确的、oracle localization 也可靠，但 real-label localization 没有超过随机，且没有 unique advantage；对应标签 `CHEMICAL_RELEVANCE_UNRESOLVED`（scope: LCRC）和 `NO_UNIQUE_LCRC_ADVANTAGE_SHOWN`。
- **GINE direction：可以继续 structural explanation。** GINE 同时满足 prediction accurate、oracle localization reliable、real-label localization supported；对应 `CONTINUE_STRUCTURAL_EXPLANATION`（scope: GINE）。
- **但不要把它写成“gradient search 已经足够好”。** 同一个搜索空间里，LCRC atom-output 局部基线在 oracle enrichment 上是 0.3244，远高于 LCRC gradient 的 0.1186 和 GINE gradient 的 0.0984；这指向 **方法/候选打分瓶颈**，而不是单纯“LCRC 表示不行”。下一步应该先做 method-level 改进（例如 contribution-aware candidate scoring、局部输出与 gradient 的组合、或更大/不同族的候选），再判断 architecture-specific claim。

**最终标签（`analysis/decision.json`）**：

```text
CONTINUE_STRUCTURAL_EXPLANATION          scope: GINE
CHEMICAL_RELEVANCE_UNRESOLVED            scope: LCRC
NO_UNIQUE_LCRC_ADVANTAGE_SHOWN           scope: LCRC, GINE
lcrc_interpretation: PAUSE_LCRC_STRUCTURAL_CLAIM
```

`label_scope` 是本次报告为了区分两个模型而增加的**报告约定**，不改变协议中预先冻结的阈值；协议阈值本身未调参。

---

## 1. P0 identity 与 P1 implementation gates

### 1.1 P0 identity（全部通过）

| 项目 | 结果 |
|---|---|
| Official split | Train 3360, Valid 420, Test 420；与 `split/scaffold/*.csv.gz` 的 set 和 order 都一致 |
| Split disjoint & cover | 是；intersection 全 0，union 4200 |
| Graph cache SHA-256 | 与源 round 记录一致 |
| Pair catalogue / atom mapping / oracle npz SHA-256 | 与源 round 记录一致 |
| 823 pairs / 1646 endpoints | 全部通过结构 verifier；0 个 `NOT_EVALUABLE` |
| Core atom mapping | atom label、core bond presence/type、variable-fragment external bond 全部一致 |
| Core automorphism invariance | 2936 次检查，0 失败；700 个 pair 声明了多个 core isomorphism，但 admissible automorphism 下 oracle delta 不变 |
| WC_LOGP graph closure | 4200 行最大误差 2.42e-14 |
| WC_LOGP pair closure | 823 个 pair 全部闭合；最大误差 4.88e-15（tol ≥ 2e-8） |
| Checkpoint SHA/参数数/kind/replay | 4/4 通过；stored prediction 最大绝对差 3.07e-6（tolerance `1e-5 + 1e-6|x|`） |

Checkpoint identity：

| Model | Epoch | Valid RMSE | Params | SHA-256 |
|---|---:|---:|---:|---|
| L_lipophilicity/Rf | 124 | 0.70688484 | 228,674 | `9ef542e5…` |
| G0_lipophilicity/Rf | 183 | 0.61607729 | 377,350 | `04282be5…` |
| L_WC_LOGP | 333 | 0.11440873 | 228,674 | `4522a636…` |
| T_WC_LOGP | 355 | 0.10327919 | 377,350 | `5e812d7e…` |

### 1.2 P1 implementation gates（全部通过）

| Gate | 结果 |
|---|---|
| LCRC all-ones == original | 通过（8 个 sample endpoints，2 个 task） |
| GINE shared physical-bond message gate all-ones == original | 通过；both directions、all layers；finite-difference 与 renumbering 通过 |
| LCRC finite difference | 128/128 通过（eps=1e-3 central difference，tol `1e-2 + 1e-2·max(1,|fd|)`） |
| GINE finite difference | 同上，128/128 |
| Atom + physical-bond renumbering | 16/16 通过 |
| Edit-region consistency | 1646/1646：`edit_atoms == atoms(edit_bonds)`，attachment/changed bonds 都在 edit bonds 内 |
| Oracle closure | 823 pair closure + 4200 graph rows 全部通过 |
| Label-blind discovery API | `discover_from_gradient(mol, grad_raw, score_name)`；无 label/target/oracle/edit/pair 参数 |
| Candidate re-selection | 无；Top-1 primary，Top-3 supplementary，candidates 由固定 gradient 规则一次产生 |

`analysis/p1_gates.json` 的 `all_passed = true`。

### 1.3 Pair support（必须分 stratum 读）

| Stratum | Pairs | Shared-endpoint components | Support |
|---|---:|---:|---|
| Train–Train | 791 | 228 | **OK** |
| Train–Valid | 5 | 4 | **LOW_SUPPORT** |
| Train–Test | 27 | 7 | **LOW_SUPPORT** |
| Both non-Train | 0 | 0 | **NO_SUPPORT** |
| At least one non-Train | 32 | 11 | **LOW_SUPPORT** |
| All evaluable | 823 | 231 | **OK** |

所有正式推断只用 component bootstrap；823 个 pair 不是 823 个独立样本。Train–Valid/Train–Test 的数字只作描述，不作为正式 claim；both-non-Train stratum 为 `NO_SUPPORT`，不能报 positive。

---

## 2. Main A — real Lipophilicity

### 2.1 Prediction quality

| 指标 | LCRC | GINE |
|---|---:|---:|
| Valid RMSE | 0.7069 | 0.6161 |
| Test RMSE（DEVELOPMENT_TEST） | 0.7293 | 0.6800 |
| Pair \|Δŷ−Δy\| | 0.2584 [0.2305, 0.2873] | 0.2866 [0.2617, 0.3131] |
| Direction accuracy（735 个 \|Δy\|≥0.1） | 0.9088 [0.8847, 0.9313] | 0.8925 [0.8627, 0.9165] |

LCRC 的 pair change 更准；GINE 的 absolute RMSE 更好。两者都过 `prediction_accurate`。

### 2.2 Structural localisation vs same-size random controls

Raw Top-1 means（不是 advantage）：

| 模型 | Top-1 k (mean) | Atom Jaccard | Atom recall | Bond recall | Boundary hit |
|---|---:|---:|---:|---:|---:|
| LCRC | 3.33 | 0.090 | 0.140 | 0.098 | 0.210 |
| GINE | 3.73 | 0.100 | 0.190 | 0.141 | 0.283 |
| LOCAL_LCRC | — | 0.140 | 0.207 | 0.182 | 0.207 |

Random controls 的均值约为：atom Jaccard 0.074–0.078，atom recall 0.124–0.139，bond recall 0.088–0.102，boundary 0.179–0.199。Advantage 才是正式比较对象：

| metric | model | point | 95% CI | n | comps | support |
|---|---:|---:|---|---:|---:|---|
| Atom Jaccard advantage | LCRC | 0.0157 | [−0.0048, 0.0373] | 823 | 231 | OK |
| Atom Jaccard advantage | GINE | **0.0219** | **[0.0002, 0.0453]** | 823 | 231 | OK |
| Atom Jaccard advantage | LOCAL_LCRC | **0.0619** | **[0.0322, 0.0954]** | 823 | 231 | OK |
| Atom recall advantage | LCRC | 0.0168 | [−0.0135, 0.0487] | 823 | 231 | OK |
| Atom recall advantage | GINE | **0.0504** | **[0.0150, 0.0895]** | 823 | 231 | OK |
| Atom recall advantage | LOCAL_LCRC | **0.0752** | **[0.0342, 0.1207]** | 823 | 231 | OK |
| Bond recall advantage | LCRC | 0.0103 | [−0.0160, 0.0376] | 823 | 231 | OK |
| Bond recall advantage | GINE | **0.0389** | **[0.0108, 0.0712]** | 823 | 231 | OK |
| Bond recall advantage | LOCAL_LCRC | **0.0871** | **[0.0484, 0.1300]** | 823 | 231 | OK |
| Boundary hit advantage | LCRC | 0.0309 | [−0.0167, 0.0837] | 823 | 231 | OK |
| Boundary hit advantage | GINE | **0.0838** | **[0.0261, 0.1545]** | 823 | 231 | OK |
| Boundary hit advantage | LOCAL_LCRC | 0.0173 | [−0.0248, 0.0631] | 823 | 231 | OK |

**Train–Train（正式 support OK）**：结论与 all 一致。LCRC atom Jaccard +0.0190 [−0.0016, 0.0405]，GINE +0.0241 [0.0026, 0.0467]；LCRC bond recall +0.0124 [−0.0140, 0.0404]，GINE +0.0418 [0.0135, 0.0723]；LCRC boundary +0.0340 [−0.0146, 0.0893]，GINE +0.0830 [0.0279, 0.1450]。

**Train–Test（27 pairs / 7 comps, LOW_SUPPORT）**：LCRC atom Jaccard −0.0576 [−0.0865, −0.0070]，GINE −0.0355 [−0.0931, 0.0591]；只是描述，不能据此做 Test claim。

### 2.3 LCRC vs GINE paired differences

| metric | LCRC−GINE | 95% CI | 解释 |
|---|---:|---|---|
| Atom Jaccard advantage | −0.0063 | [−0.0336, 0.0195] | 不显著；LCRC 没有 unique advantage |
| Atom recall advantage | −0.0336 | [−0.0777, 0.0106] | 不显著；点估计偏向 GINE |
| Bond recall advantage | −0.0286 | [−0.0657, 0.0075] | 不显著；点估计偏向 GINE |
| Boundary hit advantage | −0.0529 | [−0.1177, 0.0142] | 不显著；点估计偏向 GINE |

GINE 的点估计在三个 real-label 指标上都更高，但没有显著优于 LCRC；因此既不能说 LCRC 更好，也不能说 GINE 显著更好。**可说的是：LCRC 没有显示出 unique advantage。**

---

## 3. Main B — WC_LOGP known generated contributions

### 3.1 Contribution change 的空间分布

| 项目 | mean | median |
|---|---:|---:|
| Core \|δc\| mass | 0.1077 | 0.0 |
| Variable (A+B) \|δc\| mass | 1.3396 | 1.2658 |
| Total \|δc\| mass | 1.4473 | 1.4435 |
| Core fraction | 0.0739 | 0.0 |
| Variable fraction | **0.9261** | **1.0** |

- 100% 的 pair variable fraction > 0.5；69.99% > 0.9；54.31% 的 pair core contribution change 完全为 0。
- 因此，oracle mass capture 与 edit localization 的目标高度重叠；候选如果覆盖不到 variable fragment，就会同时丢失这两个信号。

### 3.2 Oracle enrichment / mass capture

| 模型 | Oracle enrichment | Oracle Top-1 capture | n endpoints | comps | support |
|---|---:|---:|---:|---:|---|
| LCRC | **0.1186 [0.0895, 0.1485]** | 0.1221 [0.0922, 0.1524] | 1646 | 231 | OK |
| GINE | **0.0984 [0.0665, 0.1342]** | 0.1050 [0.0726, 0.1414] | 1646 | 231 | OK |
| LOCAL_LCRC | **0.3244 [0.2876, 0.3649]** | 0.3289 [0.2917, 0.3697] | 1646 | 231 | OK |
| LCRC−GINE | +0.0201 [−0.0213, 0.0611] | +0.0171 [−0.0231, 0.0575] | 1646 | 231 | OK |

- LCRC 的 S2-matched 补充 control：**0.1182 [0.0881, 0.1490]**；与 same-size control 几乎一致，说明 LCRC 的 oracle capture 不是 S2 dose 不匹配造成的。该 control 不用于 GINE 的 equal-dose 声明。
- **零膨胀**：LCRC/GINE/LOCAL 的 median enrichment 都是 0；只有 16.7% / 14.3% / 34.1% 的候选超过 random median。均值 CI > 0 是“少数命中大 mass”的结果。
- Train–Train：LCRC 0.1219 [0.0912, 0.1521]，GINE 0.0999 [0.0671, 0.1365]，LOCAL 0.3283 [0.2904, 0.3666]。
- Train–Test（LOW_SUPPORT）：LCRC 0.0252 [0.0012, 0.0833]，GINE 0.0501 [0.0051, 0.1401]；只作描述。
- Train–Valid（LOW_SUPPORT）：LCRC 0.1000 [0.0000, 0.3750]，GINE 0.1237 [0.0000, 0.3750]。
- Both-non-Train：`NO_SUPPORT`（0 pair）。

### 3.3 贡献变化分布在 added/deleted 还是 core？

按 pair 统计：**92.6% 的 |δc| 变化在 added/deleted variable atoms 上，core 只有 7.4%**。这既支持“编辑区是主要变化来源”，也说明 oracle localization 与真实编辑定位在此数据上并不是两个独立任务；两者的一致性需要谨慎解读。

---

## 4. Failure modes 与 prediction confounding

### 4.1 四格失败表（all 823 pairs）

| 模型 | accurate & localized | accurate & mislocalized | inaccurate & localized | inaccurate & mislocalized |
|---|---:|---:|---:|---:|
| LCRC | 164 (19.9%) | **331 (40.2%)** | 106 (12.9%) | 222 (27.0%) |
| GINE | 193 (23.5%) | 235 (28.6%) | 145 (17.6%) | 250 (30.4%) |

- LCRC 的失败集中在 **accurate-mislocalized**；GINE 的失败更多是 **inaccurate**。这与 Q3/Q4 的结论一致。
- Direction-evaluable 735 pairs：LCRC 145/289/100/201；GINE 168/202/130/235。
- Train–Train 791 pairs：LCRC 161/325/102/203；GINE 188/231/140/232。
- At-least-one-non-Train 32 pairs：LCRC 3/6/4/19；GINE 5/4/5/18（LOW_SUPPORT，仅描述）。

### 4.2 预测分层下的 oracle enrichment

| 模型 | direction-correct | direction-wrong | pair-accurate | pair-inaccurate |
|---|---:|---:|---:|---:|
| LCRC | 0.1172 [0.0871, 0.1497] | 0.0898 [0.0379, 0.1547] | 0.1188 [0.0852, 0.1549] | 0.1182 [0.0810, 0.1580] |
| GINE | 0.0942 [0.0620, 0.1259] | 0.1422 [0.0813, 0.2098] | 0.0934 [0.0589, 0.1283] | 0.1039 [0.0659, 0.1491] |

LCRC 的 oracle capture 在预测准确/不准确组几乎相同；GINE 在 direction-wrong 组甚至更高。**没有证据表明 oracle localization 主要由 pair prediction accuracy 驱动。** real-label localization 的失败也不能被“预测不准”单独解释。

---

## 5. 决策

`analysis/decision.json`：

- `prediction_accurate = true`（两模型 Valid/Test RMSE ≤ 0.80，direction ≥ 0.75）。
- `oracle_localization_reliable = {LCRC: true, GINE: true}`（component bootstrap 95% 下界 > 0，点估计 ≥ 0.02）。
- `real_label_localization_supported = {LCRC: false, GINE: true}`。
- `unique_lcrc_advantage = false`；`no_unique_lcrc_advantage = true`。
- Labels：
  - `CONTINUE_STRUCTURAL_EXPLANATION`，scope **GINE**
  - `CHEMICAL_RELEVANCE_UNRESOLVED`，scope **LCRC**
  - `NO_UNIQUE_LCRC_ADVANTAGE_SHOWN`，scope **LCRC + GINE**
- `lcrc_interpretation = PAUSE_LCRC_STRUCTURAL_CLAIM`。

**一句话**：GINE 路线的 structural explanation 值得继续；LCRC 的 structural explanation claim 应暂停，直到方法/候选打分改进后重新验证。当前最强的信号不是“LCRC 架构独特”，而是“gradient subgraph search 还没有充分利用 LCRC 的 atom-level output”。

---

## 6. Figures

所有图在 `analysis/figures/`：

| 文件 | 内容 |
|---|---|
| `localisation_and_oracle_forest.png` | 左：real-label localization advantages（LCRC/GINE/LOCAL）；右：WC_LOGP oracle enrichment |
| `pair_prediction_scatter.png` | 823 pair 的 Δy vs Δŷ，LCRC/GINE 两面板 |
| `failure_modes.png` | 四格 failure mode 堆叠柱 |
| `oracle_capture_distribution.png` | 零膨胀的 oracle enrichment 分布（log density） |
| `molecule_cases.png` | 3 个真实 molecule 的 edit region / LCRC Top-1 / GINE Top-1 / WC_LOGP high-|δc| atoms |

`molecule_cases.png` 的三个 case 是**结果驱动选取**的（success / failure / inconsistent），不是随机样本；选取规则与 pair id 写在 `analysis/figure_cases.json`。图中红色 = 结构编辑区；蓝色 = LCRC Top-1；绿色 = GINE Top-1；橙色 = top-|δc| WC_LOGP atoms。

---

## 7. 限制与不能说的话

1. **823 pairs 不是独立样本**：它们共享分子和 231 个 shared-endpoint components；所有 CI 都是 component bootstrap。不能把 n=823 当作独立样本量。
2. **分层支持不足**：Train–Valid 5 pairs/4 comps、Train–Test 27 pairs/7 comps 是 `LOW_SUPPORT`；both-non-Train 是 `NO_SUPPORT`。不能说 Test 上“显著”或“泛化”。
3. **Test 是 DEVELOPMENT_TEST**：所有 Test 数字只作开发期证据，不是最终 held-out benchmark。
4. **oracle 不是化学因果**：WC_LOGP 是生成的 atom-contribution oracle；`mass_capture` 衡量的是“是否覆盖已知生成贡献”，不是“化学上真正重要”。
5. **候选空间限制**：只搜索 k=2..6、beam=12 的 connected subgraphs；不能外推到其他候选族或更大子图。
6. **零膨胀**：oracle enrichment 的中位数是 0；均值 CI > 0 不等于“多数候选有效”。
7. **local heuristic 的强基线**：`LOCAL_LCRC` 使用 LCRC 的 atom output，不是独立训练的新 explainer；它说明当前 gradient 打分有明显提升空间，但不能反过来证明“LCRC 表示一定好”。
8. **GINE 的 real-label 优势不是受控 architecture-only 比较**：GINE absolute RMSE 更好、候选平均 k 略大；虽然 random controls 做了同尺寸匹配，仍不能完全分离 representation 与 prediction accuracy 的贡献。
9. **figure case 选择是结果驱动的**：只用于解释，不用于统计推断。

---

## 8. 复现入口

```bash
# local P0/P1
cd tracks/structural-sufficiency/lcrc_explanation_task_relevance_sprint_v1
PYTHONPATH=code uv run python code/xr_prepare.py --device cpu
PYTHONPATH=code uv run python code/xr_gates.py --device cpu
PYTHONPATH=code uv run pytest -q tests

# formal workers (committed revision, via rr)
cd /home/calendar/code/research
rr run res xr-taskrel-measured --pool res-gpu0 --gpus 1 --cpus 8 --mem 16G --time 02:00:00 \
  --result tracks/structural-sufficiency/lcrc_explanation_task_relevance_sprint_v1/results/measured \
  -- python tracks/structural-sufficiency/lcrc_explanation_task_relevance_sprint_v1/code/xr_worker.py measured \
     --out-dir tracks/structural-sufficiency/lcrc_explanation_task_relevance_sprint_v1/results/measured --device cuda

rr run res xr-taskrel-oracle --pool res-gpu1 --gpus 1 --cpus 8 --mem 16G --time 02:00:00 \
  --result tracks/structural-sufficiency/lcrc_explanation_task_relevance_sprint_v1/results/oracle \
  -- python tracks/structural-sufficiency/lcrc_explanation_task_relevance_sprint_v1/code/xr_worker.py oracle \
     --out-dir tracks/structural-sufficiency/lcrc_explanation_task_relevance_sprint_v1/results/oracle --device cuda

# aggregate + figures + summary
PYTHONPATH=code uv run python code/xr_aggregate.py --n-boot 2000
PYTHONPATH=code uv run python code/xr_strata.py
PYTHONPATH=code uv run python code/xr_figures.py
PYTHONPATH=code uv run python code/xr_summary.py
```

关键文件：

- `PROTOCOL.md`：冻结协议（identity、discovery、metrics、统计、阈值）。
- `analysis/data_identity.json`、`pair_support.json`、`model_replay.json`、`p1_gates.json`。
- `analysis/measured_pair_results.csv`、`oracle_localization_results.csv`、`model_comparison.csv`、`stratified_localization.csv`、`failure_modes.csv`。
- `analysis/decision.json`、`summary_metrics.json`、`run_manifest.json`、`figure_cases.json`。
- `results/measured/`、`results/oracle/`：1646 endpoints、完整 candidate log、worker summary。
- `analysis/figures/`：5 张图。

**Execution regime**: `res` (process backend), node `a100-2`, driver 550.163.01, torch 2.5.1+cu124, CUDA 12.4, commit `a16b222466cb`; 2 workers × 1 A100, total 0.048 GPU·h, 0 errors.
