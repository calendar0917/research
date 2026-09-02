# Patch-chain 节点身份与 bit-rate 审计

> 日期：2026-08-02  
> 协议：`tracks/ksvd/docs/KSVD_PATCH_CHAIN_IDENTITY_RATE_PROTOCOL_20260802.md`  
> 总判定：`REVISE_PATCH_CHAIN_REPRESENTATION_CLAIM`

## 1. Relabel-resampling

图数：18；每图 cover seeds：3。

| metric | mean |
|---|---:|
| mapped replay patch adjacency match | 1.0000 |
| mapped replay transition match | 1.0000 |
| resampled exact ordered-chain match | 0.0000 |
| resampled patch-set Jaccard | 0.1916 |
| resampled local-vector row match | 0.0345 |
| resampled transition-map match | 0.0303 |
| absolute edge-coverage delta | 0.0070 |
| absolute pair-coverage delta | 0.0058 |
| absolute RAW full-RMSE delta | 0.0065 |

严格重编号判定：`SAMPLER_NOT_STRICTLY_RELABEL_EQUIVARIANT`。

Mapped replay 为 1 只说明一条已经生成的抽象 cover 可以随 node permutation 一起搬运。重新运行 sampler 后，chain/slot substrate 是否复现是更强、也更相关的检验。

## 2. 完整 payload 下界

| payload | mean bits / graph |
|---|---:|
| direct upper-triangle bitset | 1225.0 |
| direct enumerative exact code | 1177.2 |
| patch identity optimistic fixed-field estimate | 812.3 |
| patch identity fixed-width ordered IDs | 1066.7 |
| RAW local slots + identity estimate | 1612.3 |
| RAW unique observed pairs + identity estimate | 1464.4 |

RAW unique estimate / direct bitset：`1.195`；图级不更小比例：`0.667`。

RAW 判定：`RAW_CHAIN_NOT_BIT_COMPETITIVE_WITH_ADJACENCY`。

按目标度数拆分：

| degree | patches | identity bits | RAW unique bits | / bitset | KSVD coefficient break-even bits |
|---:|---:|---:|---:|---:|---:|
| 15 | 14.00 | 642.0 | 1187.6 | 0.969 | 8.88 |
| 20 | 18.00 | 822.0 | 1483.2 | 1.211 | 2.46 |
| 25 | 21.33 | 973.0 | 1722.4 | 1.406 | -1.04 |

`degree=15` 在该乐观显式编码下接近 bitset break-even；随着密度和 matched patch budget 增加，identity sidecar 很快成为主导成本。

## 3. KSVD break-even

在先支付 optimistic identity estimate 和每个 nonzero 的 5-bit atom index 后，为了不超过 1225-bit adjacency，每个约 `3×patch_count` coefficient 平均只剩 `3.43` bits。

判定：`KSVD_REQUIRES_UNVALIDATED_AGGRESSIVE_QUANTIZATION`。这个预算尚未包含 dictionary、quantizer、completion model、framing 或误差校验，因此不是已经实现的 codec rate。

## 4. 结论

当前 Beam8 的 coverage/reconstruction 对重编号扰动较稳定，但具体 chain、local slots 和 transition maps 不是严格 relabel-equivariant。另一方面，在本协议对 patch chain 有利的显式身份编码下，平均 RAW payload 仍不能胜过直接 bit-packed adjacency。该结果否定当前表示已经 bit-competitive 的主张，不构成对所有可能联合熵编码的不可行性证明。

因此应把现有方法限定为：

> 完整已知图上的结构引导 patch extractor 与可选局部 sparse representation。

不能继续无条件表述为编号无关图表示，或已经证明优于直接邻接矩阵的 bit-level compression。后续必须拆成 labeled rate-distortion 与 ID-free structural representation 两条不同路线。
