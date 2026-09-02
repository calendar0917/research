# luyin14 节点级 KSVD–属性融合 strict Stage A

> 日期：2026-08-12  
> 协议：`tracks/ksvd/docs/KSVD_LUYIN14_NODE_LEVEL_FUSION_PROTOCOL_20260812.md`  
> 判定：`NODE_LEVEL_FUSION_STAGE_A_NO_GO`

## 1. Balanced accuracy

| dataset | GIN | concat TRUE | FiLM TRUE | FiLM SHUFFLED | FiLM INIT |
|---|---:|---:|---:|---:|---:|
| MUTAG | 0.750 | 0.781 | 0.841 | 0.789 | 0.841 |
| PTC_MR | 0.531 | 0.535 | 0.519 | 0.537 | 0.519 |

## 2. Paired Stage-A deltas

| dataset | FiLM-GIN | W/T/L | TRUE-SHUFFLED | W/T/L | FINAL-INIT | W/T/L | concat-GIN |
|---|---:|---:|---:|---:|---:|---:|---:|
| MUTAG | +0.091 | 2/0/1 | +0.051 | 2/0/1 | +0.000 | 0/3/0 | +0.032 |
| PTC_MR | -0.012 | 0/0/3 | -0.019 | 1/0/2 | +0.000 | 0/3/0 | +0.004 |

## 3. 结论

严格 checkpoint 下节点级 concat/FiLM 未通过。图级融合过粗不是唯一问题；当前局部 adjacency KSVD token 本身仍缺少稳定任务互补性。
