# Beam8/NCI1 compact-relation follow-up

> 协议：`tracks/ksvd/docs/KSVD_BEAM8_NCI1_COMPACT_RELATION_FOLLOWUP_20260813.md`  
> 判定：`COMPACT_RELATION_BELOW_GATE`

| variant | train bacc | test bacc | gap |
|---|---:|---:|---:|
| RAW_BAG | 0.7214 | 0.6696 ± 0.0040 | 0.0518 |
| RAW_COMPACT_TRUE | 0.7278 | 0.6733 ± 0.0074 | 0.0545 |
| RAW_COMPACT_SHUFFLED | 0.7268 | 0.6650 ± 0.0037 | 0.0618 |
| INIT_BAG | 0.7162 | 0.6779 ± 0.0080 | 0.0383 |
| INIT_COMPACT_TRUE | 0.7255 | 0.6801 ± 0.0052 | 0.0454 |
| FINAL_BAG | 0.7231 | 0.6754 ± 0.0035 | 0.0477 |
| FINAL_COMPACT_TRUE | 0.7279 | 0.6759 ± 0.0039 | 0.0519 |
| FINAL_COMPACT_SHUFFLED | 0.7252 | 0.6696 ± 0.0025 | 0.0556 |

## Paired attribution

| comparison | mean | W/T/L |
|---|---:|---:|
| raw_true_vs_shuffled | +0.0083 | 3/0/0 |
| raw_true_vs_bag | +0.0036 | 1/1/1 |
| final_true_vs_shuffled | +0.0063 | 2/0/1 |
| final_true_vs_bag | +0.0005 | 2/0/1 |
| final_vs_init_true | -0.0041 | 0/0/3 |

- wide FINAL TRUE−BAG：`-0.0049`；
- compact 相对 wide 的 TRUE−BAG 改善：`+0.0054`；

## Frozen checks

- compact_binding：`False`；
- compact_incremental：`True`；
- capacity_diagnosis：`True`；

## Boundary

- 本轮是在 s8/o2 结果可见后注册的容量诊断，不是新的独立确认实验。
- compact 仍不能超过 BAG 时，不继续扫描手工关系统计量。
