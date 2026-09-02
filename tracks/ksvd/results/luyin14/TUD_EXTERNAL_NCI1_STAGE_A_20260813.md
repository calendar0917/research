# luyin14 结构—属性共享稀疏码 strict Stage A

> 日期：2026-08-12  
> 协议：`tracks/ksvd/docs/KSVD_LUYIN14_TUD_EXTERNAL_VALIDATION_PROTOCOL_20260813.md`  
> 判定：`JOINT_MULTIVIEW_DICTIONARY_STAGE_A_NO_GO`

## 1. 数据与字典诊断

| dataset | unique structure | unique joint | INIT recon | FINAL recon |
|---|---:|---:|---:|---:|
| NCI1 | 8 | 307 | 0.2147 | 0.0998 |

## 2. Balanced accuracy

| dataset | GIN | JOINT FINAL TRUE | JOINT SHUFFLED | JOINT INIT |
|---|---:|---:|---:|---:|
| NCI1 | 0.692 | 0.736 | 0.721 | 0.749 |

## 3. Paired deltas

| dataset | FINAL-GIN | W/T/L | TRUE-SHUFFLED | W/T/L | FINAL-INIT | W/T/L |
|---|---:|---:|---:|---:|---:|---:|
| NCI1 | +0.044 | 3/0/0 | +0.015 | 2/0/1 | -0.013 | 1/0/2 |

## 4. 结论

共享稀疏码未通过 Stage A；不进入 cross-attention。需要改变 patch 语义或训练目标，不能继续把问题归因于融合头过于简单。
