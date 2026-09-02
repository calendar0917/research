# Attributed Beam8 INIT-incidence follow-up

> 协议：`tracks/ksvd/docs/KSVD_ATTRIBUTED_BEAM8_INIT_INCIDENCE_FOLLOWUP_PROTOCOL_20260814.md`  
> 判定：`BEAM8_INIT_INCIDENCE_BELOW_GATE`

| variant | balanced accuracy |
|---|---:|
| GINE_ONLY | 0.7268 ± 0.0239 |
| INIT_FULL_TRUE | 0.7354 ± 0.0113 |
| INIT_FULL_SHUFFLED | 0.7445 ± 0.0150 |
| INIT_BAG_BROADCAST | 0.7097 ± 0.0313 |
| INIT_NO_RELATION | 0.7490 ± 0.0126 |

## Paired attribution

| comparison | mean | W/T/L |
|---|---:|---:|
| classification_increment | +0.0086 | 2/0/1 |
| binding | -0.0091 | 2/0/1 |
| localization | +0.0257 | 3/0/0 |
| relation_metadata | -0.0136 | 1/0/2 |

## Diagnostic checks

- classification_increment：`True`；
- binding：`False`；
- localization：`True`；
- relation_metadata：`False`；

## Boundary

- 本轮是看到 FINAL Stage A 后的机制诊断，不是独立确认实验。
- GINE_ONLY 与 INIT_FULL_TRUE 读取冻结 Stage-A folds；新增控制按相同 split/checkpoint/seeds 训练。
