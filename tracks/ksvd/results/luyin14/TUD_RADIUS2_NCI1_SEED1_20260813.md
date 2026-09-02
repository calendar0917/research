# luyin14 结构—属性共享稀疏码 strict Stage A

> 日期：2026-08-12  
> 协议：`tracks/ksvd/docs/KSVD_LUYIN14_TUD_RADIUS2_PROTOCOL_20260813.md`  
> 判定：`JOINT_MULTIVIEW_DICTIONARY_ADVANCES`

## 1. 数据与字典诊断

| dataset | unique structure | unique joint | INIT recon | FINAL recon |
|---|---:|---:|---:|---:|
| NCI1 | 159 | 1119 | 0.5499 | 0.2546 |

## 2. Balanced accuracy

| dataset | GIN | JOINT FINAL TRUE | JOINT SHUFFLED | JOINT INIT |
|---|---:|---:|---:|---:|
| NCI1 | 0.702 | 0.760 | 0.729 | 0.758 |

## 3. Paired deltas

| dataset | FINAL-GIN | W/T/L | TRUE-SHUFFLED | W/T/L | FINAL-INIT | W/T/L |
|---|---:|---:|---:|---:|---:|---:|
| NCI1 | +0.058 | 3/0/0 | +0.031 | 3/0/0 | +0.002 | 2/0/1 |

## 4. 结论

共享稀疏码通过 Stage A；扩展 split seeds 1/2 后再决定是否增加 relation-aware encoder。
