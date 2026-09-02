# Mutagenicity global-statistics residual decomposition

> 协议：`tracks/ksvd/docs/KSVD_MUTAGENICITY_GLOBAL_STATS_DECOMPOSITION_PROTOCOL_20260814.md`  
> parity：`True`

| variant | balanced accuracy over 18 units | delta vs GINE |
|---|---:|---:|
| GINE_FROZEN | 0.7331 ± 0.0482 | +0.0000 |
| ATTR_MEAN | 0.7629 ± 0.0337 | +0.0298 |
| ATTR_MAX | 0.7525 ± 0.0490 | +0.0194 |
| ATTR_SUM | 0.7592 ± 0.0210 | +0.0260 |
| ATTR_ALL | 0.7607 ± 0.0408 | +0.0276 |
| STRUCT_ONLY | 0.7792 ± 0.0132 | +0.0461 |
| GLOBAL_FULL | 0.7867 ± 0.0121 | +0.0535 |

## Sufficiency relative to GLOBAL_FULL

| component | GLOBAL−component | W/T/L | near-sufficient (<=0.5pt) |
|---|---:|---:|---:|
| ATTR_MEAN | +0.0238 | 17/0/1 | False |
| ATTR_MAX | +0.0342 | 18/0/0 | False |
| ATTR_SUM | +0.0275 | 18/0/0 | False |
| ATTR_ALL | +0.0260 | 15/0/3 | False |
| STRUCT_ONLY | +0.0075 | 13/0/5 | False |

## Boundary

- 所有统计只在 outer-train 上归一化，GINE/GLOBAL_FULL 与原实验精确 parity。
- near-sufficient 只表示该统计族足以解释大部分校准，不表示因果机制。
- 本结果用于筛选后续真实 TUD 数据集，不授权继续扩大 Mutagenicity 上的 Beam8 模型。
