# luyin14 edge-aware joint dictionary strict Stage A

> 日期：2026-08-12  
> 协议：`tracks/ksvd/docs/KSVD_LUYIN14_EDGE_AWARE_JOINT_PROTOCOL_20260812.md`  
> 判定：`EDGE_AWARE_JOINT_STAGE_A_NO_GO`

## 1. 诊断

| dataset | typed patch types | INIT recon | FINAL recon |
|---|---:|---:|---:|
| MUTAG | 13 | 0.2291 | 0.0498 |
| PTC_MR | 17 | 0.4285 | 0.1900 |

## 2. Balanced accuracy

| dataset | GINE | FINAL TRUE | FINAL SHUFFLED | INIT TRUE |
|---|---:|---:|---:|---:|
| MUTAG | 0.773 | 0.805 | 0.777 | 0.781 |
| PTC_MR | 0.498 | 0.499 | 0.511 | 0.548 |

## 3. Paired deltas

| dataset | FINAL-GINE | W/T/L | TRUE-SHUFFLED | W/T/L | FINAL-INIT | W/T/L |
|---|---:|---:|---:|---:|---:|---:|
| MUTAG | +0.032 | 1/0/2 | +0.028 | 1/1/1 | +0.024 | 2/0/1 |
| PTC_MR | +0.001 | 2/0/1 | -0.012 | 0/0/3 | -0.049 | 0/0/3 |

## 4. 结论

edge-aware shared code 未通过 Stage A；不在这两个小数据集上继续增加融合容量。
