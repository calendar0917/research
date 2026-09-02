# luyin14 OOF 多视图融合：节点/统计专家 + KSVD 专家

> 日期：2026-08-12  
> 协议：`tracks/ksvd/docs/KSVD_LUYIN14_OOF_MULTIVIEW_FUSION_PROTOCOL_20260812.md`  
> 判定：`CURRENT_KSVD_EXPERT_LACKS_STABLE_COMPLEMENTARITY`

## 1. 专家与融合主表（balanced accuracy）

| dataset | base view | BASE | INIT | FINAL | FINAL avg | FINAL scalar | FINAL gate | FINAL stack | best OOF Δ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| IMDB-BINARY | STATS | 0.708 | 0.668 | 0.672 | 0.687 | 0.707 | 0.694 | 0.707 | -0.002 |
| IMDB-MULTI | STATS | 0.463 | 0.470 | 0.473 | 0.469 | 0.466 | 0.474 | 0.468 | +0.010 |
| MUTAG | FEATURE_STATS | 0.874 | 0.711 | 0.718 | 0.802 | 0.858 | 0.850 | 0.831 | -0.016 |
| PTC_MR | FEATURE_STATS | 0.585 | 0.536 | 0.523 | 0.554 | 0.591 | 0.588 | 0.558 | +0.006 |

## 2. OOF 融合相对 BASE

| dataset | scalar Δ/W | gate Δ/W | stack Δ/W |
|---|---:|---:|---:|
| IMDB-BINARY | -0.002 (5/0/4) | -0.014 (3/0/6) | -0.002 (5/0/4) |
| IMDB-MULTI | +0.002 (7/0/2) | +0.010 (8/0/1) | +0.004 (7/0/2) |
| MUTAG | -0.016 (0/4/5) | -0.024 (2/3/4) | -0.042 (1/2/6) |
| PTC_MR | +0.006 (6/1/2) | +0.003 (3/1/5) | -0.027 (0/0/9) |

## 3. K-SVD update 归因：FINAL fusion − INIT fusion

| dataset | scalar | gate | stack |
|---|---:|---:|---:|
| IMDB-BINARY | +0.003 (5/1/3) | -0.001 (4/1/4) | +0.002 (5/1/3) |
| IMDB-MULTI | -0.001 (4/0/5) | -0.003 (4/0/5) | -0.007 (2/0/7) |
| MUTAG | -0.000 (3/3/3) | +0.008 (4/3/2) | +0.012 (5/0/4) |
| PTC_MR | +0.010 (5/1/3) | -0.004 (2/0/7) | -0.008 (3/1/5) |

## 4. 融合器行为

| dataset | FINAL scalar alpha | FINAL gate weight | OOF disagreement | struct-only correct | base-only correct |
|---|---:|---:|---:|---:|---:|
| IMDB-BINARY | +0.101 | 0.382 | 0.240 | 0.098 | 0.142 |
| IMDB-MULTI | +0.045 | 0.353 | 0.252 | 0.097 | 0.100 |
| MUTAG | +0.040 | 0.293 | 0.241 | 0.054 | 0.187 |
| PTC_MR | -0.025 | 0.302 | 0.427 | 0.192 | 0.235 |

## 5. 结论

严格 OOF 融合也未建立稳定互补性。现有 concat 的失败不只是融合器过于简单，而是当前 KSVD expert 在真实数据上不能稳定修正 base expert 的错误；不应直接增加 cross-attention 容量。
