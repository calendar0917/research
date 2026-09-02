# Beam8 边覆盖、残差边与混合停止协议

> 日期：2026-08-05  
> 状态：结果前冻结  
> 起点：`docs/luyin/luyin13.txt` 与后续讨论。  
> 范围：完整已知、无向、无自环二值图；只研究 cover coverage 与显式 payload 代理，不使用 graph labels。

## 1. 研究问题

当前 Beam8/R1 在 `s=10,o=3,m=1.5` 下取得较高真实边覆盖，但没有显式观察全部节点对。本轮不再把“所有节点对都进入 patch”作为压缩硬约束，而检验以下更窄命题：

1. 若未观察 pair 默认解码为 0，继续增加 Beam8 patch 达到 90%/95%/99%/100% edge coverage 分别需要多少 patch？
2. 在当前固定 Beam8 前缀后，直接编码遗漏真实边，是否比继续增加完整 patch 更便宜？
3. 沿同一条 Beam8 链，按“patch payload 代理 + residual-edge sidecar”自动选停止前缀，是否优于固定预算与纯 patch repair？
4. 即使混合停止优于纯 patch repair，计入 node-identity/transition sidecar 后是否可能低于直接 adjacency bitset 或直接 enumerative edge code？

本轮不声称已经实现最终熵编码器。所有 bit 数必须标为显式编码代理或 optimistic proxy。

## 2. 数据与冻结设置

正式设置：

```text
graph_bank_seed = 810001
families = regular / small_world / block
n = 50
degrees = 15 / 20 / 25
8 graphs per cell = 72 graphs
cover seeds = 950101 / 950102 / 950103
patch size s = 10
overlap o = 3
base budget multiplier m = 1.5
Beam = 8
candidate restarts = 1
maximum patches = 60
KSVD proxy: K=24, T=3, coefficient bits q=8
```

第一轮允许先用 18 图 × 1 seed smoke run；正式结论必须来自上面的 72 图 × 3 seeds。

每张图只生成一次最多 60 个 patch 的 Beam8 chain；所有预算点均取这条链的前缀，避免不同停止条件改变早期随机路径。

## 3. Cover 与残差定义

对前缀 `P_1,...,P_t`：

```text
observed pairs E_obs_all = union of all within-patch node pairs
covered true edges E_cov = E_true intersect E_obs_all
residual true edges E_res = E_true minus E_cov
unobserved pairs U = C(n,2) minus E_obs_all
```

未观察 pair 的默认预测为 0。因此 RAW zero-fill coverage RMSE 为：

```text
sqrt(|E_res| / C(n,2))
```

它只度量 coverage error，不包含 KSVD quantization/reconstruction error。

记录以下前缀：

- `ZERO`：0 patch，全部真实边作为 residual；
- `BASE`：旧 `patch_budget(..., multiplier=1.5)`；
- `EDGE90/95/99/100`：第一次达到对应 edge coverage 的前缀；
- `HYBRID_ALL`：在 `0..max_patches` 全部前缀中，使总 proxy bits 最小者；
- `HYBRID_AFTER_BASE`：限制 `t >= BASE` 后的最小者。

若 60 patches 内未达到某阈值，必须报告失败比例，不得把最后一个前缀伪装成达标。

## 4. Bit accounting

设 `M=C(n,2)`，patch 数为 `t`。

### 4.1 直接基线

- direct bitset：`M` bits；
- direct enumerative exact：`ceil(log2 C(M,m)) + ceil(log2(M+1))`，其中 `m=|E_true|`。

### 4.2 Residual edge sidecar

接收端已由 patch memberships 知道 unobserved pair universe `U`。这里“unobserved 默认 0”只是解码约定；遗漏真实边必须由 residual sidecar 付费后才能消除 coverage error。若 observed pair values 来自有损 KSVD，residual edge sidecar 并不修正其重构/量化误差；因此这里只称 coverage-complete，不称整个 codec exact。精确 residual-subset sidecar 代理为：

```text
ceil(log2(|U|+1)) + ceil(log2 C(|U|, |E_res|))
```

该编码记录 residual edge 数和它们在 unobserved-pair universe 中的子集；不逐边固定宽度保存两个 endpoint。

### 4.3 Patch identity / transition

同时报告两种身份成本：

1. `ordered_identity_bits`：ordered membership fixed-field estimate：
   - first patch：`log2 P(n,s)`；
   - 每个后继 patch：`log2 C(s,o) + log2 P(n-s,s-o)`；这里 `n-s` 是 exact-overlap 下上一 patch 之外的合法节点全集（同时修正旧 identity-rate runner 使用 `n-o` 的宽松估算）；
2. `canonical_set_identity_bits`：假设给定 membership 与 root 后，其余 rooted-canonical slots 可由 patch topology 导出：
   - first patch：`log2 C(n,s) + log2 s`；
   - 每个后继 patch：`log2 C(s,o) + log2 C(n-s,s-o) + log2(s-o)`；最后一项编码 Beam8 当前从新节点中选出的 root。

第二项仍忽略 automorphism 下 node-map ambiguity、framing 和实现开销，是有意偏向 patch 方法的 optimistic proxy，不得称为严格信息论下界。

### 4.4 Prefix framing

Hybrid stream 必须传输停止前缀长度 `t∈[0,max_patches]`：

```text
ceil(log2(max_patches+1))
```

在冻结的 `max_patches=60` 下为 6 bits。它计入所有 patch/hybrid checkpoint，包括 `t=0`；direct bitset/enumerative standalone baseline 不支付该 hybrid framing。codec mode、dictionary、quantizer、checksum 等其他 framing 仍未计入。

### 4.5 KSVD code proxy

冻结：

```text
per patch = T * (ceil(log2 K) + q)
K=24, T=3, q=8
```

即每 patch 39 bits。该 proxy 假设 dictionary 已跨图摊销，且不计 dictionary、quantizer、均值、framing、误差校验；也不保证 8-bit coefficient 下的实际 distortion。

主要混合成本：

```text
canonical_hybrid_bits
= prefix framing bits
+ canonical_set_identity_bits
+ KSVD code proxy bits
+ residual sidecar bits
```

并同时报告 ordered-identity 版本。`t=0` 时 patch identity/code 成本为 0，但仍需 hybrid prefix framing，因此等于 direct enumerative edge code再加6-bit停止字段；若完全不用 hybrid codec，则应直接选择 standalone enumerative baseline。

### 4.6 RAW exact control

为了避免把 KSVD proxy 当作已实现 codec，额外报告一个 coverage + observed-value exact control：

```text
canonical_raw_exact_bits
= prefix framing bits
+ canonical_set_identity_bits
+ unique observed pair values
+ residual sidecar bits
```

它是显式保存所有 unique observed 0/1 pair values 的 exact control。

## 5. 主要比较与冻结判定

### 5.1 Repair efficiency

对每张图比较：

- `BASE + residual` 的 added residual bits；
- 从 BASE 延长到首次 EDGE100 的 added patch identity + KSVD proxy bits。

若 EDGE100 不可达，则该图直接记为 residual preferred。

`RESIDUAL_REPAIR_PREFERRED` gate：

1. 100% edge coverage 达成率不影响 residual branch 可行性；
2. 至少 2/3 cover seeds 中，`BASE + residual` 平均 total canonical proxy bits 低于 `EDGE100 + no residual`；
3. 图级 residual-preferred fraction >= 0.75。

### 5.2 Hybrid stopping

`HYBRID_STOPPING_USEFUL` gate：

1. `HYBRID_AFTER_BASE` 平均 canonical proxy bits 相对 `BASE+residual` 至少降低 1%；
2. 且不高于 `EDGE100+no residual`（只在 EDGE100 可达图比较）；
3. 三个 seeds 均满足前两项。

### 5.3 Global codec claim

只有 `HYBRID_ALL` 在三个 seeds 中都同时低于 direct bitset 和 direct enumerative exact，才可标记：

```text
HYBRID_PROXY_RATE_COMPETITIVE
```

否则标记：

```text
REPAIR_POLICY_FOUND_BUT_NOT_GLOBAL_CODEC
```

若连 repair/hybrid gate 也不通过，则标记：

```text
NO_RATE_JUSTIFIED_EDGE_REPAIR_POLICY
```

即使 proxy rate gate 通过，也必须在实际 rooted-canonical KSVD quantization/distortion 审计前保留“proxy”限定。

## 6. 必报结果

- 各 checkpoint 的 patch count、edge/pair coverage、residual edge count、RAW zero-fill RMSE；
- ordered/canonical identity bits、KSVD proxy bits、residual bits、RAW exact bits、总 hybrid bits；
- EDGE90/95/99/100 达成率和达到阈值所需 patch 数分布；
- HYBRID_ALL 与 HYBRID_AFTER_BASE 的停止 patch 数分布；
- family × degree 分层；
- 三 cover seeds 的独立汇总；
- exact overlap、connected patch、single-chain invariants；
- 与 1225-bit bitset、direct enumerative exact 的比率。

## 7. 解释边界

- 本轮输入是完整已知图；若输入本身缺边，unobserved pair 不能默认解释为 0。
- residual edge sidecar 使用原图 truth，因为它是 encoder 对完整输入计算的确定性残差，不是 test-label leakage。
- 本轮只显式计入 prefix-length framing；其余 codec mode、dictionary、quantizer、checksum 等成本未计，因此仍偏向 patch 路线。
- RAW zero-fill RMSE 只反映未覆盖真实边；不能替代 KSVD 后的实际 full RMSE。
- KSVD bit 数是冻结的 optimistic proxy，不是已经验证的量化 codec。
- 若 direct enumerative 在 `t=0` 处成为全局最优，不表示 Beam8 对 representation/downstream 无用，只否定其在当前显式 labeled-adjacency rate accounting 下的压缩优势。
