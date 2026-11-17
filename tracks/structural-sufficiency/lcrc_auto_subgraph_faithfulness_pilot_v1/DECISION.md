# DECISION — lcrc-auto-subgraph-faithfulness-pilot-v1

只依据本轮**跑之前冻结**的判读规则，不制定后续新模型。

## 1. ENGINEERING_GATE

`PASS`

## 2. DISCOVERY_SIGNAL

总体：`WEAK_OR_INCONCLUSIVE`

### `ogbg-molesol` — `WEAK_OR_INCONCLUSIVE`

| split | condition | value | rule | passed |
|---|---|---|---|---|
| valid | effect_floor_fraction_ge_0p50 | 1.000 | >= 0.50 of molecules reach 1% of the Train target standard deviation | yes |
| valid | effect_ratio_gt1_fraction_ge_0p70 | 0.841 | >= 0.70 | yes |
| valid | effect_ratio_median_ge_1p5 | 1.293 | >= 1.5 | no |
| valid | enough_matched_controls | 63.000 | matched molecules >= max(32, 80% of interpreted molecules) | yes |
| test | effect_floor_fraction_ge_0p50 | 1.000 | >= 0.50 of molecules reach 1% of the Train target standard deviation | yes |
| test | effect_ratio_gt1_fraction_ge_0p70 | 0.836 | >= 0.70 | yes |
| test | effect_ratio_median_ge_1p5 | 1.386 | >= 1.5 | no |
| test | enough_matched_controls | 61.000 | matched molecules >= max(32, 80% of interpreted molecules) | yes |

未通过条件：`valid:effect_ratio_median_ge_1p5`, `test:effect_ratio_median_ge_1p5`

### `ogbg-mollipo` — `WEAK_OR_INCONCLUSIVE`

| split | condition | value | rule | passed |
|---|---|---|---|---|
| valid | effect_floor_fraction_ge_0p50 | 0.984 | >= 0.50 of molecules reach 1% of the Train target standard deviation | yes |
| valid | effect_ratio_gt1_fraction_ge_0p70 | 0.719 | >= 0.70 | yes |
| valid | effect_ratio_median_ge_1p5 | 1.568 | >= 1.5 | yes |
| valid | enough_matched_controls | 64.000 | matched molecules >= max(32, 80% of interpreted molecules) | yes |
| test | effect_floor_fraction_ge_0p50 | 1.000 | >= 0.50 of molecules reach 1% of the Train target standard deviation | yes |
| test | effect_ratio_gt1_fraction_ge_0p70 | 0.562 | >= 0.70 | no |
| test | effect_ratio_median_ge_1p5 | 1.286 | >= 1.5 | no |
| test | enough_matched_controls | 64.000 | matched molecules >= max(32, 80% of interpreted molecules) | yes |

未通过条件：`test:effect_ratio_median_ge_1p5`, `test:effect_ratio_gt1_fraction_ge_0p70`

## 3. 依据的关键数字

| dataset | split | median effect_ratio | ratio>1 | abs(delta) ≥ 1% std | matched n | median abs(delta)/std |
|---|---|---|---|---|---|---|
| `ogbg-molesol` | valid | 1.293 | 84.1% | 100.0% | 63/64 | 0.8199 |
| `ogbg-molesol` | test | 1.386 | 83.6% | 100.0% | 61/64 | 1.0599 |
| `ogbg-mollipo` | valid | 1.568 | 71.9% | 98.4% | 64/64 | 0.6779 |
| `ogbg-mollipo` | test | 1.286 | 56.2% | 100.0% | 64/64 | 0.5629 |

## 4. ARCHITECTURE_DIAGNOSIS（只报告，不自动推导结论）

| dataset | split | rho_share med | cancellation med | abs(S1) med | abs(S2) med | interaction med | dose consistency |
|---|---|---|---|---|---|---|---|
| `ogbg-molesol` | train | 0.142 | 0.073 | 0.1636 | 1.4545 | -0.3177 | 92.2% |
| `ogbg-molesol` | valid | 0.092 | 0.125 | 0.1717 | 1.7379 | -0.3294 | 98.4% |
| `ogbg-molesol` | test | 0.065 | 0.111 | 0.1240 | 1.8353 | -0.3718 | 100.0% |
| `ogbg-mollipo` | train | 0.121 | 2.607e-09 | 0.1785 | 0.7397 | 0.0520 | 82.8% |
| `ogbg-mollipo` | valid | 0.160 | 4.064e-09 | 0.2201 | 0.9625 | 0.0514 | 89.1% |
| `ogbg-mollipo` | test | 0.136 | 6.757e-17 | 0.1892 | 0.7286 | -0.0483 | 76.6% |

## 5. 若结果不足，属于哪一类失败（按证据归类，可多选）

* **工程实现失败**：`NO` — 所有核心等价/映射/回溯门禁均通过。
* **候选搜索失败**：`NO` — 可解释分子覆盖率最小 100.0%（候选池在全部已解释分子上都非空）。
* **梯度与有限干预不一致**：`PARTIAL` — Top-1 符号一致率最低 76.6%；中位线性化绝对误差见 `main_table.csv`。
* **结构信息屏蔽后变化过小**：`NO` — 达到 1% Train 标签标准差的分子比例最低 98.4%。
* **无足够匹配对照**：`NO` — 匹配对照成功率最低 89.1%（其余分子记为 INSUFFICIENT_MATCHED_NULL）。
* **全图读出混合严重**：`NOT_EVIDENCED` — rho_share 中位数最高 0.160，cancellation 中位数最高 0.125。
* **搜索范围 / 固定原子与全图特征造成的不可识别性**：`BOUNDED_BY_DESIGN` — 只搜索 2–6 键连通子图；atom_feat/atom_scalar/global 与原子数不被门控，因此任何依赖这些输入的响应都不在本干预的可识别范围内。

禁止把解释失败直接写成“LCRC 没有学到结构信息”。本轮的干预只屏蔽显式 bond-derived S1/S2 证据，其余输入（原子特征、原子标量、global42、原子数）保持不变。

