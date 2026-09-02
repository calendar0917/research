# luyin14 edge-aware joint dictionary strict Stage A

> 日期：2026-08-12  
> 协议：`tracks/ksvd/docs/KSVD_LUYIN14_TUD_RADIUS2_PROTOCOL_20260813.md`  
> 判定：`EDGE_AWARE_JOINT_STAGE_A_NO_GO`

## 1. 诊断

| dataset | typed patch types | INIT recon | FINAL recon |
|---|---:|---:|---:|
| Mutagenicity | 424 | 0.6026 | 0.2828 |

## 2. Balanced accuracy

| dataset | GINE | FINAL TRUE | FINAL SHUFFLED | INIT TRUE |
|---|---:|---:|---:|---:|
| Mutagenicity | 0.681 | 0.751 | 0.751 | 0.751 |

## 3. Paired deltas

| dataset | FINAL-GINE | W/T/L | TRUE-SHUFFLED | W/T/L | FINAL-INIT | W/T/L |
|---|---:|---:|---:|---:|---:|---:|
| Mutagenicity | +0.070 | 3/0/0 | +0.000 | 1/0/2 | +0.000 | 2/0/1 |

## 4. 结论

edge-aware shared code 未通过 Stage A；不在这两个小数据集上继续增加融合容量。
