# luyin14 edge-aware joint dictionary strict Stage A

> 日期：2026-08-12  
> 协议：`tracks/ksvd/docs/KSVD_LUYIN14_EDGE_AWARE_JOINT_PROTOCOL_20260812.md`  
> 判定：`EDGE_AWARE_JOINT_STAGE_A_NO_GO`

## 1. 诊断

| dataset | typed patch types | INIT recon | FINAL recon |
|---|---:|---:|---:|
| MUTAG | 13 | 0.2188 | 0.0602 |
| PTC_MR | 17 | 0.4174 | 0.1831 |

## 2. Balanced accuracy

| dataset | GINE | FINAL TRUE | FINAL SHUFFLED | INIT TRUE |
|---|---:|---:|---:|---:|
| MUTAG | 0.769 | 0.733 | 0.797 | 0.738 |
| PTC_MR | 0.577 | 0.503 | 0.553 | 0.538 |

## 3. Paired deltas

| dataset | FINAL-GINE | W/T/L | TRUE-SHUFFLED | W/T/L | FINAL-INIT | W/T/L |
|---|---:|---:|---:|---:|---:|---:|
| MUTAG | -0.036 | 1/0/2 | -0.063 | 0/1/2 | -0.004 | 1/0/2 |
| PTC_MR | -0.074 | 0/0/3 | -0.049 | 1/0/2 | -0.034 | 0/0/3 |

## 4. 结论

edge-aware shared code 未通过 Stage A；不在这两个小数据集上继续增加融合容量。
