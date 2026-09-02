# luyin14 结构—属性共享稀疏码 strict Stage A

> 日期：2026-08-12  
> 协议：`tracks/ksvd/docs/KSVD_LUYIN14_JOINT_MULTIVIEW_DICTIONARY_PROTOCOL_20260812.md`  
> 判定：`JOINT_MULTIVIEW_DICTIONARY_STAGE_A_NO_GO`

## 1. 数据与字典诊断

| dataset | unique structure | unique joint | INIT recon | FINAL recon |
|---|---:|---:|---:|---:|
| MUTAG | 4 | 33 | 0.0282 | 0.0128 |
| PTC_MR | 6 | 135 | 0.2600 | 0.1340 |

## 2. Balanced accuracy

| dataset | GIN | JOINT FINAL TRUE | JOINT SHUFFLED | JOINT INIT |
|---|---:|---:|---:|---:|
| MUTAG | 0.769 | 0.738 | 0.706 | 0.742 |
| PTC_MR | 0.522 | 0.536 | 0.502 | 0.484 |

## 3. Paired deltas

| dataset | FINAL-GIN | W/T/L | TRUE-SHUFFLED | W/T/L | FINAL-INIT | W/T/L |
|---|---:|---:|---:|---:|---:|---:|
| MUTAG | -0.032 | 1/0/2 | +0.031 | 2/0/1 | -0.004 | 1/0/2 |
| PTC_MR | +0.014 | 2/0/1 | +0.034 | 2/0/1 | +0.052 | 3/0/0 |

## 4. 结论

共享稀疏码未通过 Stage A；不进入 cross-attention。需要改变 patch 语义或训练目标，不能继续把问题归因于融合头过于简单。
