# luyin14 edge-aware joint dictionary strict Stage A

> 日期：2026-08-12  
> 协议：`tracks/ksvd/docs/KSVD_LUYIN14_EDGE_AWARE_JOINT_PROTOCOL_20260812.md`  
> 判定：`EDGE_AWARE_JOINT_ADVANCES`

## 1. 诊断

| dataset | typed patch types | INIT recon | FINAL recon |
|---|---:|---:|---:|
| MUTAG | 13 | 0.2106 | 0.0551 |
| PTC_MR | 17 | 0.4232 | 0.1775 |

## 2. Balanced accuracy

| dataset | GINE | FINAL TRUE | FINAL SHUFFLED | INIT TRUE |
|---|---:|---:|---:|---:|
| MUTAG | 0.690 | 0.824 | 0.745 | 0.750 |
| PTC_MR | 0.516 | 0.526 | 0.529 | 0.523 |

## 3. Paired deltas

| dataset | FINAL-GINE | W/T/L | TRUE-SHUFFLED | W/T/L | FINAL-INIT | W/T/L |
|---|---:|---:|---:|---:|---:|---:|
| MUTAG | +0.134 | 3/0/0 | +0.079 | 2/0/1 | +0.074 | 2/0/1 |
| PTC_MR | +0.010 | 2/0/1 | -0.003 | 1/0/2 | +0.003 | 2/0/1 |

## 4. 结论

edge-aware shared code 通过 Stage A；扩展 split seeds 1/2。
