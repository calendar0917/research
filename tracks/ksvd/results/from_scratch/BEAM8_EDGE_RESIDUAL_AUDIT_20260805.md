# Beam8 edge repair、residual 与 hybrid rate 审计

> 日期：2026-08-05  
> 协议：`tracks/ksvd/docs/KSVD_BEAM8_EDGE_RESIDUAL_PROTOCOL_20260805.md`  
> 判定：`REPAIR_POLICY_FOUND_BUT_NOT_GLOBAL_CODEC`

## 1. 设置与总判定

72 graphs × 3 cover seeds；Beam8/R1；s=10, o=3；最多 60 patches。

- invariants：`True`；
- BASE+residual vs EDGE100 gate：`True`（图级 residual preferred=1.000）；
- HYBRID_AFTER_BASE gate：`False`；
- global proxy rate gate：`False`；
- 存在正长度 patch 前缀胜过 direct enumerative 的运行比例：`0.000`。

## 2. 核心发现

从 BASE 延长到 EDGE100 平均增加 `8.59` 个 patch （p50=9，p90=10，max=11）。

用 BASE + residual 代替补到 EDGE100，平均节省 `348.1` proxy bits （min=162，p50=353，p90=433）。

HYBRID_AFTER_BASE 恰好停在 BASE 的运行比例为 `1.000`；即在当前显式成本模型下，BASE 之后没有一个额外 patch 的边际收益足以抵消 identity + KSVD code 成本。

每次运行中最便宜的正长度 patch 前缀相对 direct enumerative 仍平均多 `27.7` bits，最接近的一次也多 `8` bits。

因此，本轮找到的是 **Beam8 表示已经存在时最省的 coverage-completion policy：固定 BASE 后直接记录 residual edges**；它精确补齐未观察真实边，但不修正已观察 pair 上的 KSVD 重构/量化误差。没有找到胜过直接整图编码的 labeled-adjacency codec。

## 3. Checkpoints

| checkpoint | reach | patches | edge cover | pair cover | residual edges | RAW RMSE | identity | framing | KSVD proxy | residual bits | hybrid bits | raw exact bits |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ZERO | 1.000 | 0.00 | 0.0000 | 0.0000 | 499.61 | 0.6352 | 0.0 | 6.0 | 0.0 | 1175.3 | 1181.3 | 1181.3 |
| BASE | 1.000 | 17.65 | 0.8750 | 0.5281 | 59.52 | 0.2186 | 600.9 | 6.0 | 688.5 | 278.1 | 1573.5 | 1531.9 |
| EDGE90 | 1.000 | 18.97 | 0.9099 | 0.5597 | 45.11 | 0.1907 | 645.6 | 6.0 | 739.7 | 226.1 | 1617.5 | 1563.4 |
| EDGE95 | 1.000 | 21.67 | 0.9571 | 0.6169 | 21.53 | 0.1316 | 737.0 | 6.0 | 845.0 | 130.3 | 1718.3 | 1628.9 |
| EDGE99 | 1.000 | 25.05 | 0.9941 | 0.6802 | 3.06 | 0.0469 | 851.4 | 6.0 | 976.8 | 32.2 | 1866.3 | 1722.8 |
| EDGE100 | 1.000 | 26.25 | 1.0000 | 0.7025 | 0.00 | 0.0000 | 892.0 | 6.0 | 1023.6 | 8.9 | 1930.5 | 1767.4 |
| HYBRID_ALL | 1.000 | 0.00 | 0.0000 | 0.0000 | 499.61 | 0.6352 | 0.0 | 6.0 | 0.0 | 1175.3 | 1181.3 | 1181.3 |
| HYBRID_AFTER_BASE | 1.000 | 17.65 | 0.8750 | 0.5281 | 59.52 | 0.2186 | 600.9 | 6.0 | 688.5 | 278.1 | 1573.5 | 1531.9 |

`hybrid bits = prefix-length framing + optimistic canonical-set identity + K24/T3/q8 code proxy + residual subset`。RAW RMSE 只包含未覆盖真实边的 zero-fill coverage error。

## 4. Seed robustness

| seed | EDGE100 reach | residual preferred | BASE bits | EDGE100 patch-only bits* | hybrid-after bits | hybrid reduction | hybrid gate |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 950101 | 1.000 | 1.000 | 1573.9 | 1924.0 | 1573.9 | 0.000 | False |
| 950102 | 1.000 | 1.000 | 1572.3 | 1917.8 | 1572.3 | 0.000 | False |
| 950103 | 1.000 | 1.000 | 1574.3 | 1922.9 | 1574.3 | 0.000 | False |

* EDGE100 patch-only mean 只在 60 patches 内达到 100% 的图上计算；不可达图自动视为 residual policy 更可行，但不混入该均值。

## 5. Direct codec comparison

| seed | HYBRID_ALL | bitset | direct enumerative | below both |
|---:|---:|---:|---:|---:|
| 950101 | 1181.3 | 1225.0 | 1175.3 | False |
| 950102 | 1181.3 | 1225.0 | 1175.3 | False |
| 950103 | 1181.3 | 1225.0 | 1175.3 | False |

## 6. Family × degree

| family | degree | BASE patches | BASE edge | BASE residual | BASE hybrid bits | HYBRID_AFTER patches | HYBRID_AFTER bits | EDGE100 reach |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| block | 15 | 13.75 | 0.8505 | 55.67 | 1297.0 | 13.75 | 1297.0 | 1.000 |
| block | 20 | 17.62 | 0.9090 | 45.46 | 1531.9 | 17.62 | 1531.9 | 1.000 |
| block | 25 | 21.50 | 0.9449 | 34.33 | 1763.1 | 21.50 | 1763.1 | 1.000 |
| regular | 15 | 14.00 | 0.7973 | 76.00 | 1377.2 | 14.00 | 1377.2 | 1.000 |
| regular | 20 | 18.00 | 0.8542 | 72.92 | 1637.0 | 18.00 | 1637.0 | 1.000 |
| regular | 25 | 21.00 | 0.8805 | 74.71 | 1839.1 | 21.00 | 1839.1 | 1.000 |
| small_world | 15 | 14.00 | 0.8581 | 53.21 | 1304.5 | 14.00 | 1304.5 | 1.000 |
| small_world | 20 | 18.00 | 0.8897 | 55.17 | 1587.2 | 18.00 | 1587.2 | 1.000 |
| small_world | 25 | 21.00 | 0.8908 | 68.25 | 1824.2 | 21.00 | 1824.2 | 1.000 |

## 7. 路线结论与边界

- 对 **已经选择 Beam8 作为结构表示** 的路线：采用 `BASE + residual edge subset`，不要继续增加 repair patches；它能消除 coverage error。只有 RAW patch values 被精确保留，或另有 compression-error correction 时，整体图恢复才是 exact。
- 对 **labeled adjacency bit compression** 的路线：当前显式 patch-chain 编码应停止；不再为证明该主张追加实际 K24/T3/q8 量化实验。
- 对 **结构表示/下游学习** 的路线：Beam8、rooted-canonical slots 与共享字典仍可继续，因为其价值目标不是击败 direct adjacency codec。
- 若未来研究 sampler-induced joint entropy coding，应作为新协议重新立项；当前 canonical-set 公式不是该分布下的信息论下界，不能据此宣称所有可能图 codec 都不可行。

边界：

- 输入是完整已知图；缺边输入不能把 unobserved 解释为 0。
- residual 是 encoder 从完整输入算出的确定性 sidecar，不是标签泄漏。
- 已计入 0..maximum_patches 的 prefix-length framing；仍未计 codec mode、dictionary、quantizer、checksum 等其他 framing/模型开销。
- canonical identity 和 K24/T3/q8 都是偏向 patch 方法的 optimistic proxy；即使假设 dictionary 免费且 8-bit coefficient 无失真，正长度 patch 前缀也没有胜过 direct enumerative。真实量化只会增加成本或失真。
- `HYBRID_ALL` 允许选择 0 patch，因此若 direct enumerative 本身最便宜，会诚实地停在 0；这只约束 labeled-adjacency codec 主张，不否定 Beam8 的结构表示用途。
