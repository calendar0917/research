# DECISION — 保留 C1 读出的 S2 结构角色 pilot

轮次 `lcrc-relation-role-preserving-pilot-v1`。结论只覆盖本轮预登记判据，不外推到其他架构、其他数据集或其他 seed。

## 判定

**`ONLY_C1_READOUT_RECOVERY`**

| flag | value |
|---|---|
| CONTROL_ALSO_PRESERVED | True |
| OLD_MODELS_RETRAINED | False |
| ONLY_C1_READOUT_RECOVERY | True |
| ORACLE_LOCALISATION_SIGNAL | False |
| PREDICTION_PRESERVED | True |
| PREDICTION_RECOVERED_BOTH | True |
| ROLE_INCREMENT_ORACLE | False |
| ROLE_INCREMENT_PREDICTIVE | False |
| ROLE_INCREMENT_SHOWN | False |
| ROLE_NOT_SUPPORTED | False |
| TEST_USED_FOR_SELECTION | False |

## 五个问题的直接回答

1. **保留 C1 读出后预测是否恢复**：REAL Valid RMSE = 0.7071（C1 的 1.0004×，差距 0.0003），CONTROL = 0.6945（0.9825×）。PREDICTION_PRESERVED = `True`。

2. **真实结构边界角色是否优于同容量伪分组**：Lipo Valid ΔRMSE(REAL−CONTROL) = 0.0126，95% CI [-0.0328, 0.0572]，P(REAL better) = 0.2710；Test Δ = 0.0541，95% CI [0.0104, 0.1036]。预测增量 credit = `False`。

3. **WC_LOGP 已知贡献捕获是否改善**：Valid paired enrichment Δ = -0.0006 (95% CI [-0.0023, 0.0010], n=299)；Test Δ = 0.0001 (95% CI [-0.0011, 0.0013], n=303)。oracle 增量 credit = `False`。

4. **改善来自预测还是结构解释**：ROLE_INCREMENT_SHOWN = `False`，ROLE_INCREMENT_PREDICTIVE = `False`，ROLE_INCREMENT_ORACLE = `False`。ONLY_C1_READOUT_RECOVERY = `True`，ROLE_NOT_SUPPORTED = `False`。

5. **是否值得继续显式 block/interface Token**：见 `DECISION.md` 的下一步建议；本轮证据仅为单 seed DEVELOPMENT，Valid 上两模型 |Δ| 的 mean Spearman = 0.9039，argmax 一致率 = 89.29%。

## 预登记判据的命中情况

| split | evidence | point estimate | 95% CI | P(REAL better) | credit |
|---|---|---|---|---|---|
| lipo/train | predictive | -0.0079 | [-0.0165, 0.0001] | 0.9735 | False |
| lipo/valid | predictive | 0.0126 | [-0.0328, 0.0572] | 0.2710 | False |
| lipo/test | predictive | 0.0541 | [0.0104, 0.1036] | 0.0055 | False |
| wc_logp/valid | oracle | -0.0006 | [-0.0023, 0.0010] | 0.5033 | False |
| wc_logp/test | oracle | 0.0001 | [-0.0011, 0.0013] | 0.5000 | False |

判据（训练前冻结）：预测增量要求 95% 配对 CI 在 REAL 一侧排除 0 且 P(REAL better) ≥ 0.95；oracle 增量还要求 REAL 的 enrichment 本身满足“≥ 60% 匹配分子为正 + 均值为正”，并在同一批匹配分子上与 CONTROL 做配对比较。

## 角色缩放是否被真正使用

| model | |g_int−1| mean | |g_cross−1| mean | Valid mean |Δpred| (both=1) | Valid both=1 RMSE | Valid learned RMSE |
|---|---|---|---|---|---|
| RR_REAL | 0.0719 | 0.0946 | 0.0569 | 0.7015 | 0.7071 |
| RR_CONTROL | 0.0704 | 0.0642 | 0.0551 | 0.6940 | 0.6945 |

## 下一步建议

结论是 `ONLY_C1_READOUT_RECOVERY`：保留 C1 读出后两个模型都能恢复预测，但真实结构边界角色没有给出预登记强度的额外收益。因此**不建议**仅凭本轮证据去投入显式 block/interface Token 的预测力路线；若仍要做，必须先提出一个新的、能区分“读出恢复”与“角色信息”的预登记设计（例如在匹配候选上做更强的外部结构判据），而不是继续加容量。**本轮到此停止。**

## 口径限制

- single training seed per unit; no multi-seed stability claim
- official Test is DEVELOPMENT_TEST: evaluated every epoch, never used for selection
- the historical C1(Rf) baseline is read-only and was not re-trained; the Lipophilicity comparison is DIRECTLY_COMPARABLE (same split/target/scaler/seed/recipe/canonical init)
- the WC_LOGP C1 baseline is recipe R3 and is flagged RECIPE_DIFFERS
- no historical architecture result of the previous round enters any decision rule
- oracle controls are matched on bond and atom count only, not on composition

## 未做的事

- no historical model re-trained, re-selected or re-evaluated
- no new seed, no hyper-parameter search, no extra dataset
- no partition extension / learnable grouping
- no Test-based selection or tuning

