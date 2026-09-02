# luyin14 uncompressed RAW patch relation terminal screen

> 日期：2026-08-12  
> 协议：`tracks/ksvd/docs/KSVD_LUYIN14_RAW_RELATION_PROTOCOL_20260812.md`  
> 判定：`RAW_RELATION_TERMINAL_NO_GO`

## 1. Gate

- TRUE > SHUFFLED binding：`False`；
- TRUE > RAW bag：`True`；
- STATS+TRUE > STATS：`False`。

## 2. Balanced accuracy 与 paired delta

| dataset | STATS | RAW bag | TRUE | SHUFFLED | TRUE-SHUFFLED | W/T/L | TRUE-BAG | W/T/L | STATS+TRUE-STATS | W/T/L |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IMDB-BINARY | 0.708 | 0.629 | 0.672 | 0.666 | +0.006 | 6/0/3 | +0.043 | 9/0/0 | -0.033 | 1/1/7 |
| IMDB-MULTI | 0.463 | 0.450 | 0.458 | 0.455 | +0.003 | 6/0/3 | +0.008 | 8/0/1 | -0.003 | 5/0/4 |
| MUTAG | 0.856 | 0.691 | 0.718 | 0.723 | -0.005 | 2/4/3 | +0.027 | 6/1/2 | -0.101 | 0/0/9 |
| PTC_MR | 0.554 | 0.529 | 0.528 | 0.529 | -0.001 | 4/1/4 | -0.001 | 4/2/3 | -0.020 | 2/0/7 |

## 3. 结论

未压缩 RAW token 也没有形成跨数据集、stats 外的 relation 增益。停止 patch-graph/Transformer 下游路线。
