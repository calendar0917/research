# DECISION — LCRC Explanation Task Relevance Sprint v1

## 决策

**暂停 LCRC-specific structural explanation claim；继续 GINE 路线的 structural explanation；不声称 LCRC 有 unique advantage。**

```text
CONTINUE_STRUCTURAL_EXPLANATION          scope: GINE
CHEMICAL_RELEVANCE_UNRESOLVED            scope: LCRC
NO_UNIQUE_LCRC_ADVANTAGE_SHOWN           scope: LCRC, GINE
lcrc_interpretation: PAUSE_LCRC_STRUCTURAL_CLAIM
```

`label_scope` 是为了区分两个模型而增加的**报告约定**；协议中预先冻结的阈值没有修改，也没有根据 Valid/Test 结果调参。

## 依据（全部为 component bootstrap，231 个 shared-endpoint components，2000 reps）

### 1. Prediction gate: PASS

| | LCRC | GINE |
|---|---:|---:|
| Valid RMSE | 0.7069 | 0.6161 |
| Test RMSE（DEVELOPMENT_TEST） | 0.7293 | 0.6800 |
| Direction accuracy（\|Δy\|≥0.1） | 0.9088 | 0.8925 |

两模型都满足 `prediction_accurate`（RMSE ≤ 0.80；direction ≥ 0.75）。因此不能把整体结果归为 `PREDICTION_CONFOUNDED`。

### 2. Oracle localization gate: PASS for both

- LCRC oracle enrichment **0.1186 [0.0895, 0.1485]**
- GINE oracle enrichment **0.0984 [0.0665, 0.1342]**
- LCRC S2-matched control **0.1182 [0.0881, 0.1490]**
- LCRC−GINE **+0.0201 [−0.0213, 0.0611]**（不显著）

两者都超过同尺寸随机 control，且点估计 ≥ 0.02；因此不是 `EXPLANATION_METHOD_OR_REPRESENTATION_BOTTLENECK`（该标签要求 oracle localization 不可靠）。

**但注意**：只有 16.7% 的 LCRC 候选、14.3% 的 GINE 候选超过 random median；中位 enrichment = 0。正的均值来自少数命中大 contribution mass 的候选。不能写成“多数候选都捕获了 WC_LOGP 变化”。

### 3. Real-label localization gate: GINE PASS, LCRC FAIL

GINE 的三个 advantage 的 95% CI 下界都 > 0：

- Atom Jaccard +0.0219 [0.0002, 0.0453]
- Bond recall +0.0389 [0.0108, 0.0712]
- Boundary hit +0.0838 [0.0261, 0.1545]

LCRC 的三个都跨 0：

- Atom Jaccard +0.0157 [−0.0048, 0.0373]
- Bond recall +0.0103 [−0.0160, 0.0376]
- Boundary hit +0.0309 [−0.0167, 0.0837]

因此：

- GINE：`CONTINUE_STRUCTURAL_EXPLANATION`。
- LCRC：oracle localization 可靠，但 real-label pair evidence 不足 → `CHEMICAL_RELEVANCE_UNRESOLVED`。

### 4. LCRC vs GINE: no unique LCRC advantage

| metric | LCRC−GINE | 95% CI |
|---|---:|---|
| Atom Jaccard advantage | −0.0063 | [−0.0336, 0.0195] |
| Atom recall advantage | −0.0336 | [−0.0777, 0.0106] |
| Bond recall advantage | −0.0286 | [−0.0657, 0.0075] |
| Boundary hit advantage | −0.0529 | [−0.1177, 0.0142] |

点估计全为负、CI 全跨 0。**不能声称 LCRC 比 GINE 更会定位，也不能声称 GINE 显著更好；能说的是 LCRC 没有显示出 unique advantage。**

### 5. Failure mode: LCRC 的瓶颈更像定位，不是预测

823 pair 的四格：

| 模型 | accurate & localized | accurate & mislocalized | inaccurate & localized | inaccurate & mislocalized |
|---|---:|---:|---:|---:|
| LCRC | 164 | **331** | 106 | 222 |
| GINE | 193 | 235 | 145 | 250 |

- LCRC 的最大单一失败格是 **accurate & mislocalized**（40.2%），不是 prediction error。
- GINE 的失败更多是 inaccurate（62.7% 的非双成功 pair）。
- LCRC 的 oracle enrichment 在 pair-accurate 与 pair-inaccurate 组几乎相同（0.1188 vs 0.1182）；说明 LCRC 的失败不能简单归因于“预测不准”。

## 为什么不给 LCRC CONTINUE

LCRC 同时满足：

- prediction accurate；
- oracle localization reliable；

但**不满足** real-label localization supported。协议给 `CONTINUE_STRUCTURAL_EXPLANATION` 的条件是三者同时成立；LCRC 缺第三条，因此不能给 LCRC `CONTINUE`。GINE 三条都成立，所以 `CONTINUE` 的 scope 只写 GINE。

## 关键方法 caveat

`LOCAL_LCRC`（用 LCRC 自己的 atom local output 做同一 k=2..6、beam=12 搜索）在 oracle enrichment 上是 **0.3244 [0.2876, 0.3649]**，远高于 LCRC gradient 的 0.1186 和 GINE gradient 的 0.0984。它说明：

- 当前 gradient-subgraph 打分/搜索是一个明显的 method-level 瓶颈；
- “LCRC 表示不能解释”不是唯一解释；
- 下一步应优先做候选打分/候选族的 method-level 改进，再谈 architecture-specific claim。

这个 baseline 不是新训练的 explainer，也不参与协议决策；它只用于诊断“方法是否还有空间”。

## 什么会改变这个决策

1. LCRC 在 real-label localization 的至少一个 primary advantage（atom Jaccard、bond recall、boundary hit）上，component bootstrap 95% 下界 > 0；
2. 或 LCRC 在 oracle enrichment 上显著优于 GINE（paired CI 下界 > 0）且 real-label 不退化；
3. 或 method-level 改进（contribution-aware scoring / 候选族）使 LCRC 的 oracle 与 real-label 同时超过 GINE，且预测准确率仍过 gate。

在这些出现之前，LCRC-specific structural explanation 应保持 `PAUSE`；GINE 路线可继续。

## 不能说的话

- 不能说 LCRC 在 Test 上泛化好：Train–Test 只有 27 pairs / 7 components，`LOW_SUPPORT`，且 Test 是 `DEVELOPMENT_TEST`。
- 不能说 both-non-Train 有任何证据：该 stratum 是 `NO_SUPPORT`（0 pair）。
- 不能说 WC_LOGP capture 等于化学因果：它是生成贡献 oracle。
- 不能说“多数候选都捕获了 WC_LOGP 变化”：中位 enrichment 为 0。
- 不能说 LCRC 的失败完全与预测无关：GINE 的 absolute RMSE 更好，仍可能对 real-label 优势有贡献；这不是受控 architecture-only 比较。
