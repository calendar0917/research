# luyin14 patch-local KSVD × node-feature 多模态融合

> 日期：2026-08-12  
> 协议：`tracks/ksvd/docs/KSVD_LUYIN14_PATCH_LOCAL_MULTIMODAL_PROTOCOL_20260812.md`  
> 判定：`PATCH_LOCAL_MULTIMODAL_ALIGNMENT_NO_GO`

## 1. Gate

- added_value: `False`；
- binding: `False`；
- ksvd_learning: `False`；
- beats_naive: `True`；

## 2. Balanced accuracy

| dataset | feature+stats | naïve +FINAL rich | INIT true cross | FINAL true cross | FINAL shuffled cross |
|---|---:|---:|---:|---:|---:|
| MUTAG | 0.874 | 0.772 | 0.826 | 0.817 | 0.824 |
| PTC_MR | 0.585 | 0.559 | 0.563 | 0.567 | 0.565 |

## 3. Paired deltas

| dataset | TRUE-base | W/T/L | TRUE-SHUFFLED | W/T/L | FINAL-INIT | W/T/L | TRUE-naïve |
|---|---:|---:|---:|---:|---:|---:|---:|
| MUTAG | -0.057 | 2/0/7 | -0.007 | 4/1/4 | -0.009 | 3/1/5 | +0.045 |
| PTC_MR | -0.018 | 3/0/6 | +0.003 | 4/1/4 | +0.004 | 4/0/5 | +0.009 |

## 4. 结论

compact bilinear 局部对齐未形成稳定增益。现有失败不能仅归因于图级 concat 太粗；在增加 cross-attention 容量前，需要先改变结构专家或 patch 语义。
