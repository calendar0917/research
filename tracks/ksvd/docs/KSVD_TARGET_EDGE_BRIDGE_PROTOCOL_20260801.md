# KSVD 连续 patch cover：target-edge bridge 第二轮协议

> 日期：2026-08-01  
> 状态：正式结果不可见前冻结  
> 前置：第一轮 `FAIL_SMALL_PATCH_GRAPH_RECOVERY`；连续性和 stitching invariants 通过，但 single frontier 在 block 图上滞留于局部社区。

## 1. 唯一研究问题

保持第一轮数据、patch 大小、overlap 和预算全部不变，只改变 patch 链的跨区域调度：

> 显式选择低覆盖区域中的未覆盖边，并通过最短 bridge 引导下一 patch，能否修复单条局部 frontier 在 modular graph 上的覆盖失败？如果单链仍失败，允许少量连续 segments 是否足够？

本轮仍不训练 KSVD、不使用 labels、不做分类。

## 2. 数据与预算

复用第一轮固定 graph bank：

```text
graph_bank_seed = 810001
families = regular, small_world, block
n_nodes = 50
target average degree = 15, 20, 25
graphs/family/degree = 8
patch_size = 10
target_overlap = 5
edge_capacity_multiplier = 1.5
```

使用新的 sampler audit seeds：

```text
820101, 820102, 820103
```

同一 graph/seed 下所有方法共享第一轮 `patch_budget()` 得到的 patch 数量。

## 3. 方法

### 3.1 Frozen controls

- `independent_walk`：第一轮独立 walk；
- `frontier_cover`：第一轮单链局部贪心，不修改实现。

### 3.2 edge_target_bridge

保持单条连续链，每次 transition 必须恰好重叠 5 个节点：

1. 统计每个节点尚未观察到的 incident true edges；
2. 从全图未覆盖边中选择 endpoint deficit 最大的 target edge；
3. 从当前 patch 中寻找 anchor，并在禁止重新使用本轮 dropped nodes 的条件下求到 target edge 的最短路径；
4. bridge path 与 target edge 占用不超过 5 个新 slots；
5. retained slots 和剩余 slots 按新增未覆盖边、未观察 node-pair、低覆盖节点的顺序贪心补足。

target edge 和 bridge 使用的都只是输入图结构，不使用 graph labels。

### 3.3 multi_chain_target

固定 `segment_length=4`：

- 每个 segment 的第一个 patch 直接以全局 endpoint deficit 最大的未覆盖边为 seed；
- segment 内后续 patch 使用 `edge_target_bridge`；
- segment 之间允许 reset，不要求 overlap；
- segment 内 transition 必须恰好 overlap 5；
- 不根据 graph family、degree 或运行结果修改 segment length。

这是结构边界对照：若它成功而单链失败，说明一般图更适合“多条局部连续链 + segment 关系”，而不是强制一条全局图像式序列。

## 4. 新增审计

除第一轮全部指标外，增加：

- `patch_connected_rate`；
- `segment_count`；
- `continuous_transition_fraction`；
- `within_segment_overlap_mean`；
- `within_segment_jaccard_gap`；
- block-only edge/pair coverage；
- target edge hit rate：被选择的 target edge 是否确实进入下一 patch；
- bridge length mean/max。

所有方法继续要求：

- observed-pair consistency/accuracy = 1；
- mapped replay patch adjacency/transition maps = 1。

## 5. 冻结 gates

### 5.1 Single-chain gate

`edge_target_bridge` 必须在至少 2/3 seeds 同时满足：

1. mean edge coverage 比 frozen `frontier_cover` 高至少 `0.03`；
2. block-only mean edge coverage高至少 `0.05`；
3. mean pair coverage比 frontier 降低不超过 `0.03`；
4. continuous transition fraction = `1.0`；
5. within-segment overlap mean = `5.0`；
6. within-segment Jaccard gap > `0`；
7. patch connected rate >= `0.99`；
8. target edge hit rate = `1.0`。

### 5.2 Multi-chain gate

若 single-chain 不通过，`multi_chain_target` 必须在至少 2/3 seeds 同时满足：

1. mean edge coverage 比 frontier 高至少 `0.05`；
2. block-only mean edge coverage高至少 `0.08`；
3. mean pair coverage不低于 frontier；
4. continuous transition fraction >= `0.70`；
5. within-segment overlap mean = `5.0`；
6. within-segment Jaccard gap > `0`；
7. patch connected rate >= `0.99`；
8. target edge hit rate = `1.0`。

## 6. 判定

- `PASS_SINGLE_CHAIN_TARGET_BRIDGE`：单链修复成功，下一轮可进入 raw/KSVD stitched reconstruction；
- `SELECT_MULTI_CHAIN_PATCH_COVER`：只有 multi-chain 通过，后续对象改为 patch segments + segment relation；
- `FAIL_TARGET_EDGE_SCHEDULING`：两者都失败，停止固定 `s=10,b=5` 调度搜索，转入 patch-size/overlap 容量曲线；
- `FAIL_TARGET_BRIDGE_INVARIANTS`：拼接、mapped replay、连通性或 target-hit 出错。

## 7. 解释边界

通过仍只说明结构 cover 改善，不说明：

- learned atom 已经有 motif 语义；
- patch 序列是唯一或严格 permutation-invariant 的；
- 多 segment 的顺序具有跨图共享语义；
- KSVD、Transformer 或下游分类已经有效。
