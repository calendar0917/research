# Beam8 conditional increment after global statistics

> 协议：`tracks/ksvd/docs/KSVD_ATTRIBUTED_BEAM8_CONDITIONAL_INCREMENT_PROTOCOL_20260814.md`  
> 判定：`BEAM8_INCREMENT_BEYOND_GLOBAL_STATS_NOT_ESTABLISHED`

| variant | balanced accuracy over 18 units |
|---|---:|
| GINE_FROZEN | 0.7331 ± 0.0482 |
| GLOBAL_STATS_STAGE1 | 0.7867 ± 0.0121 |
| GLOBAL_PLUS_BEAM8_FULL | 0.7859 ± 0.0112 |
| GLOBAL_PLUS_BEAM8_CODE_ONLY | 0.7858 ± 0.0125 |
| GLOBAL_PLUS_BEAM8_HIST_ONLY | 0.7857 ± 0.0108 |
| GLOBAL_PLUS_BEAM8_RANDOM_DICTIONARY | 0.7860 ± 0.0112 |

## Paired attribution

| comparison | mean | W/T/L | split3/4 | model0/1/2 |
|---|---:|---:|---:|---:|
| full_vs_global | -0.0008 | 9/1/8 | -0.0025 / +0.0009 | +0.0001 / -0.0021 / -0.0004 |
| full_vs_code_only | +0.0001 | 9/0/9 | -0.0009 / +0.0011 | +0.0004 / -0.0009 / +0.0008 |
| full_vs_hist_only | +0.0002 | 6/0/12 | -0.0008 / +0.0011 | +0.0026 / -0.0019 / -0.0002 |
| full_vs_random_dictionary | -0.0002 | 9/0/9 | -0.0015 / +0.0012 | +0.0006 / -0.0014 / +0.0004 |

## Frozen checks

- increment：`False`；
- beats_hist_only：`False`；
- beats_random_dictionary：`False`；
- both_splits：`False`；
- model_majority：`False`；
- parity：`True`；

## Boundary

- 第一阶段先冻结 GLOBAL_STATS，第二阶段才读取 Beam8，因此 FULL−GLOBAL 是条件增量。
- 本轮是结果可见后的机制探索；通过时仍需未见 split 或外部 attributed TUD 确认。
- 失败表示当前 Beam8 BAG 未证明具有普通图统计之外的分类价值。
