# DECISION — 关系–接口路由 pilot

轮次 `lcrc-relation-interface-routing-pilot-v1`。结论只覆盖本轮预登记的判据，不外推到论文级结论。

## 判定

| 标记 | 值 |
|---|---|
| CAPACITY_CONFOUNDED | 否 |
| INTERFACE_CONTEXT_INCREMENT_NOT_SHOWN | 是 |
| INTERFACE_CONTEXT_REGRESSION_WHERE | wc_logp/test |
| INTERFACE_CONTEXT_SIGNIFICANT_REGRESSION | 是 |
| OLD_MODELS_RETRAINED | 否 |
| ORACLE_LOCALISATION_SIGNAL | 是 |
| PREDICTION_RETENTION_FAIL | 是 |
| RHO_DOMINATES_STRUCTURE | 否 |
| TEST_USED_FOR_SELECTION | 否 |
| partition_gate | PARTITION_USABLE |

**下一步建议：`PAUSE_FIXED_TOPOLOGY_PARTITION`**

## 三个命题

### A. 固定拓扑分块 + 接口路由是否保留了预测力

* 每个模型的两个训练单元都在 1.10× 以内：R1 = 否，R2 = 否
* 至少一个模型通过：否

### B. 接口是否需要两端 block 的上下文

* `lipo/test`：ΔRMSE(R2−R1) = -0.008825，95% CI [-0.073002, 0.057045]，P(R2 更好) = 0.596
* `lipo/valid`：ΔRMSE(R2−R1) = -0.028208，95% CI [-0.091266, 0.038352]，P(R2 更好) = 0.776
* `wc_logp/test`：ΔRMSE(R2−R1) = 0.037197，95% CI [0.017857, 0.056033]，P(R2 更好) = 0.000
* `wc_logp/valid`：ΔRMSE(R2−R1) = 0.013967，95% CI [-0.020106, 0.043476]，P(R2 更好) = 0.218
* 增量显著：否；容量混淆：否

### C. 结构解释是否有意义

* oracle 定位有信号：是
* 原生 ledger 闭合：是
* 备注：ledger closure is a property of the architecture, not evidence of chemical correctness

## 口径限制

* Only the Lipophilicity unit has a directly comparable C1 baseline (recipe Rf, seed 0, same split and scaler); the WC_LOGP baseline is recipe R3, so its retention ratio mixes recipe and architecture.
* The frozen architecture replaces the C1 readout as a whole: the 27-term atom sum becomes a ~10-term block sum with block-internal weighted-mean pooling, and rho(g, mean h, mean h^2, log1p n) becomes rho_g(g, log1p n). The measured gap is the cost of that whole replacement, not of routing alone.
* The oracle controls are matched on bond count and atom count, not on atom/bond composition, so a block that is chemically atypical for its size can look good.
* All numbers are single-seed DEVELOPMENT / DEVELOPMENT_TEST evaluations.

## 未做的事

* 没有重训任何旧模型，没有新 seed，没有 Test 选型；
* 没有把接口上下文之外的架构改动带入结论；
* 没有对 Lipophilicity 的下降做“调参救回来”的尝试——那会破坏单 seed 预登记。
