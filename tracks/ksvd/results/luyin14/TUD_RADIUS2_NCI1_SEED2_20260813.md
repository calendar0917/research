# luyin14 结构—属性共享稀疏码 strict Stage A

> 日期：2026-08-12  
> 协议：`tracks/ksvd/docs/KSVD_LUYIN14_TUD_RADIUS2_PROTOCOL_20260813.md`  
> 判定：`JOINT_MULTIVIEW_DICTIONARY_STAGE_A_NO_GO`

## 1. 数据与字典诊断

| dataset | unique structure | unique joint | INIT recon | FINAL recon |
|---|---:|---:|---:|---:|
| NCI1 | 159 | 1119 | 0.5370 | 0.2591 |

## 2. Balanced accuracy

| dataset | GIN | JOINT FINAL TRUE | JOINT SHUFFLED | JOINT INIT |
|---|---:|---:|---:|---:|
| NCI1 | 0.725 | 0.742 | 0.750 | 0.763 |

## 3. Paired deltas

| dataset | FINAL-GIN | W/T/L | TRUE-SHUFFLED | W/T/L | FINAL-INIT | W/T/L |
|---|---:|---:|---:|---:|---:|---:|
| NCI1 | +0.016 | 2/0/1 | -0.009 | 1/0/2 | -0.021 | 1/0/2 |

## 4. 结论

共享稀疏码未通过 Stage A；不进入 cross-attention。需要改变 patch 语义或训练目标，不能继续把问题归因于融合头过于简单。
