# ENZYMES Beam8 canonical-slot attribute invariance protocol

> 日期：2026-08-14  
> 状态：分类前冻结。

对64个 ENZYMES graphs 各作3次固定随机 node relabeling。cover、canonical colors、slot order、
continuous slot tensor、within-patch shuffled control 与 node-incidence 均重新从 relabeled graph
计算，不复制原图结果。

对 `SLOT_FULL_TRUE` 和 `SLOT_FULL_WITHIN_PATCH_SHUFFLED` 分别要求：

- token row exact/allclose match = 1.0；
- token multiset match = 1.0；
- compact graph readout match = 1.0；
- mapped orbit-safe node-incidence equivariance = 1.0。

chain exact match 仅作 diagnostic。全部 required checks 通过才允许运行 split9/10 分类。

