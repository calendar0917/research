# luyin14 edge-aware joint dictionary strict Stage A

> 日期：2026-08-12  
> 协议：`tracks/ksvd/docs/KSVD_LUYIN14_TUD_EXTERNAL_VALIDATION_PROTOCOL_20260813.md`  
> 判定：`EDGE_AWARE_JOINT_ADVANCES`

## 1. 诊断

| dataset | typed patch types | INIT recon | FINAL recon |
|---|---:|---:|---:|
| Mutagenicity | 19 | 0.3602 | 0.1399 |

## 2. Balanced accuracy

| dataset | GINE | FINAL TRUE | FINAL SHUFFLED | INIT TRUE |
|---|---:|---:|---:|---:|
| Mutagenicity | 0.681 | 0.752 | 0.719 | 0.713 |

## 3. Paired deltas

| dataset | FINAL-GINE | W/T/L | TRUE-SHUFFLED | W/T/L | FINAL-INIT | W/T/L |
|---|---:|---:|---:|---:|---:|---:|
| Mutagenicity | +0.070 | 3/0/0 | +0.033 | 3/0/0 | +0.039 | 2/0/1 |

## 4. 结论

edge-aware shared code 通过 Stage A；扩展 split seeds 1/2。
