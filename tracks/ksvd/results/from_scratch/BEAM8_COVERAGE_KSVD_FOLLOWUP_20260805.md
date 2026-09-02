# Beam8 coverage operating point：rooted-canonical KSVD follow-up

> 日期：2026-08-05
> 协议：`tracks/ksvd/docs/KSVD_BEAM8_COVERAGE_OPERATING_POINT_PROTOCOL_20260805.md`
> 判定：`KEEP_BASE_WITH_RESIDUAL_SECOND_CHANNEL_GEOMETRY_UNRESOLVED`

## 1. 判定

- passing FAIR95 geometries：`[]`；
- FAIR95 KSVD Pareto：`[]`；
- BASE KSVD Pareto：`['s8_o2_BASE', 's10_o3_BASE', 's12_o4_BASE']`；
- geometry 尚未冻结：三者在 reconstruction、dictionary size 与 per-graph code length 间互不支配。

- 条件性结论：patch-only 表示偏向 FAIR95；若 residual 作为显式第二通道，则 BASE 更优。当前标签只对应后一种系统。

| geometry | seed | full RMSE reduction | corrected ratio | observed ratio | full≥2% | corrected≤+1% | observed≤+2% | recall nonlower | pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| s8_o2 | 970101 | 0.0904 | 1.0695 | 1.0190 | True | False | True | True | False |
| s8_o2 | 970102 | 0.0929 | 1.0695 | 1.0178 | True | False | True | True | False |
| s8_o2 | 970103 | 0.0938 | 1.0716 | 1.0184 | True | False | True | True | False |
| s10_o3 | 970101 | 0.1041 | 1.1142 | 1.0194 | True | False | True | True | False |
| s10_o3 | 970102 | 0.1024 | 1.1254 | 1.0267 | True | False | False | True | False |
| s10_o3 | 970103 | 0.1055 | 1.1169 | 1.0205 | True | False | False | True | False |
| s12_o4 | 970101 | 0.1082 | 1.1449 | 1.0068 | True | False | True | True | False |
| s12_o4 | 970102 | 0.1061 | 1.1507 | 1.0128 | True | False | True | True | False |
| s12_o4 | 970103 | 0.1079 | 1.1427 | 1.0051 | True | False | True | True | False |

## 2. Graph-balanced means across cover seeds

| branch | patches | dict/code scalars | stage | patch error | observed RMSE/F1 | full RMSE | corrected RMSE | full recall/F1 |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| s8_o2_BASE | 27.71 | 672/83.1 | raw | 0.0000 | 0.0000/1.0000 | 0.1763 | 0.0000 | 0.9138/0.9545 |
| s8_o2_BASE | 27.71 | 672/83.1 | final | 0.2819 | 0.2335/0.9587 | 0.2441 | 0.1681 | 0.8939/0.9164 |
| s8_o2_FAIR95 | 30.66 | 672/92.0 | raw | 0.0000 | 0.0000/1.0000 | 0.1261 | 0.0000 | 0.9607/0.9800 |
| s8_o2_FAIR95 | 30.66 | 672/92.0 | final | 0.2946 | 0.2378/0.9547 | 0.2215 | 0.1799 | 0.9364/0.9359 |
| s10_o3_BASE | 17.65 | 1080/53.0 | raw | 0.0000 | 0.0000/1.0000 | 0.2198 | 0.0000 | 0.8738/0.9321 |
| s10_o3_BASE | 17.65 | 1080/53.0 | final | 0.3871 | 0.3120/0.9047 | 0.3150 | 0.2252 | 0.8259/0.8467 |
| s10_o3_FAIR95 | 22.06 | 1080/66.2 | raw | 0.0000 | 0.0000/1.0000 | 0.1239 | 0.0000 | 0.9620/0.9806 |
| s10_o3_FAIR95 | 22.06 | 1080/66.2 | final | 0.4117 | 0.3190/0.8926 | 0.2822 | 0.2519 | 0.8977/0.8760 |
| s12_o4_BASE | 12.00 | 1584/36.0 | raw | 0.0000 | 0.0000/1.0000 | 0.2476 | 0.0000 | 0.8357/0.9091 |
| s12_o4_BASE | 12.00 | 1584/36.0 | final | 0.4522 | 0.3569/0.8618 | 0.3567 | 0.2564 | 0.7663/0.7896 |
| s12_o4_FAIR95 | 16.69 | 1584/50.1 | raw | 0.0000 | 0.0000/1.0000 | 0.1186 | 0.0000 | 0.9650/0.9822 |
| s12_o4_FAIR95 | 16.69 | 1584/50.1 | final | 0.4809 | 0.3598/0.8471 | 0.3184 | 0.2939 | 0.8606/0.8327 |

## 3. Geometry candidates

| candidate | full RMSE | corrected RMSE | full recall | dict/code scalars | Pareto |
|---|---:|---:|---:|---:|---:|
| s8_o2_BASE | 0.2441 | 0.1681 | 0.8939 | 672/83.1 | True |
| s10_o3_BASE | 0.3150 | 0.2252 | 0.8259 | 1080/53.0 | True |
| s12_o4_BASE | 0.3567 | 0.2564 | 0.7663 | 1584/36.0 | True |

## 4. 结果解释

- 不加 residual 时，FAIR95 在三种 geometry 上都稳定降低 full RMSE 并提高 full edge recall；所以不能说额外覆盖没有价值。
- 加 exact residual 后，FAIR95 的 compression-only RMSE 相对 BASE 分别恶化约 7.0%、11.9%、14.6%；因此在显式 residual 第二通道系统中，不应继续把 residual 边搬入固定 K24/T3 的 patch channel。
- BASE residual 不是随机噪声：coverage 审计显示其主要偏向 zero-common-neighbor 与跨社区边。因此 residual 若用于下游，必须作为可见 edge token/channel，而不能只在最终解码时悄悄置 1。

## 5. Error decomposition audit

最大 squared-error decomposition delta：`9.714e-17`。

`uncorrected full RMSE² = RAW coverage RMSE² + residual-corrected compression RMSE²`；residual 只修未观察真实边，不修 observed-pair KSVD 错误。

## 6. 边界

- 每个 geometry/checkpoint/cover seed 独立训练 train-only dictionary；没有跨 test graph 泄漏。
- 三折 test graph 数为 27/27/18；汇总按 test graph 数加权，而不是错误地对三折等权。
- FAIR95 是否通过由三 seed 中至少两 seed的预注册 gate决定，不用 pooled mean 掩盖 seed failure。
- BASE 与 FAIR95 的 observed-pair 集合不同；observed RMSE gate 混合了共享 pair 误差与新增 hard-pair 难度，不解释为同一 pair 上的纯退化。
- residual sidecar 的 bit 成本未进入 KSVD gate；本轮结论只比较 reconstruction，不是新的 bit-codec 主张。
- 若多个 geometry 位于 Pareto，最终选择仍需明确偏好 dictionary size 或 per-graph code length；不伪造单一总标量。
- 本轮只检验 residual 作为 reconstruction sidecar，不检验 residual edge token 的下游价值。
