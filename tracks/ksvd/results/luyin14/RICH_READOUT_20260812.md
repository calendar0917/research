# luyin14 rich sparse-code readout 迁移审计

> 日期：2026-08-12  
> 协议：`tracks/ksvd/docs/KSVD_LUYIN14_RICH_READOUT_PROTOCOL_20260812.md`  
> 判定：`RICH_STATISTICS_HELP_WITHOUT_KSVD_UPDATE_ATTRIBUTION`

## 1. 三个主 gate

| gate | pass | 预注册要求 |
|---|---:|---|
| FINAL rich > FINAL coarse | `True` | 至少 2/4 数据集 mean >= +.01 且 wins >= 6/9 |
| FINAL rich > INIT rich | `False` | 至少 2/4 数据集 mean >= +.01 且 wins >= 6/9 |
| STATS+FINAL rich > STATS | `False` | 至少 2/4 数据集 mean >= +.01 且 wins >= 6/9 |

## 2. Balanced accuracy 主表

| dataset | STATS | INIT coarse | FINAL coarse | INIT rich | FINAL rich | STATS+INIT rich | STATS+FINAL rich |
|---|---:|---:|---:|---:|---:|---:|---:|
| IMDB-BINARY | 0.708 | 0.631 | 0.632 | 0.668 | 0.672 | 0.664 | 0.676 |
| IMDB-MULTI | 0.463 | 0.445 | 0.448 | 0.470 | 0.473 | 0.472 | 0.467 |
| MUTAG | 0.856 | 0.697 | 0.712 | 0.711 | 0.718 | 0.769 | 0.766 |
| PTC_MR | 0.554 | 0.517 | 0.521 | 0.536 | 0.523 | 0.553 | 0.527 |

## 3. 预注册 paired deltas

| dataset | rich-coarse | W/T/L | FINAL-INIT rich | W/T/L | STATS+rich-STATS | W/T/L |
|---|---:|---:|---:|---:|---:|---:|
| IMDB-BINARY | +0.039 | 9/0/0 | +0.004 | 6/0/3 | -0.033 | 1/0/8 |
| IMDB-MULTI | +0.024 | 9/0/0 | +0.003 | 6/0/3 | +0.004 | 6/0/3 |
| MUTAG | +0.005 | 5/0/4 | +0.007 | 5/0/4 | -0.090 | 1/0/8 |
| PTC_MR | +0.003 | 4/0/5 | -0.013 | 1/0/8 | -0.027 | 3/0/6 |

## 4. 节点特征数据

| dataset | feature | feature+stats | feature+FINAL rich | feature+stats+FINAL rich | rich over feature+stats |
|---|---:|---:|---:|---:|---:|
| MUTAG | 0.825 | 0.874 | 0.761 | 0.772 | -0.102 (1/0/8) |
| PTC_MR | 0.564 | 0.585 | 0.575 | 0.559 | -0.026 (3/0/6) |

## 5. 结论

高阶 sparse-code 统计有用，但 FINAL 没有稳定优于同一 INIT；不能把增益归因于 K-SVD updates。后续若继续，应研究合法原型/patch metric，而不是增加 K-SVD 轮次。

解释边界：本实验冻结现有 patch/K/T/folds；没有扫描 classifier、读出变体或网络深度。
