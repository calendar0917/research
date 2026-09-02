# Continuous cover：slot-persistent ordering 审计协议

> 日期：2026-08-01  
> 状态：正式结果不可见前冻结  
> 前置：target-bridge sampler 通过；construction-order KSVD 的 stitched gate 全过，但 patch INIT→FINAL reduction 只有 `5.4%`，未达到冻结的 `10%` optimization gate。

## 1. 研究问题

当前相邻 patch 虽共享恰好 5 个节点，但共享节点可能从前一 patch 的任意 slot 移动到后一 patch 的前 5 个 slots。transition map 能恢复对应，普通 KSVD 却看不到这张 map。

本轮只改变 local ordering：

> 若让重叠节点在相邻 patch 中保持相同 slot，能否在完全相同的 node sets、覆盖率、预算和字典容量下，提高 KSVD patch/stitch reconstruction 与 overlap consistency？

## 2. 两个 matched branches

### construction_order

冻结第二、三轮的当前顺序：retained nodes 排在新 patch 前部，bridge/new nodes 随后。

### persistent_slot_order

对冻结后的 cover 做纯重排：

1. 每个 segment 首 patch 保持原顺序；
2. 对相邻 patch 的 shared node，令其 next slot 等于 previous slot；
3. 新节点按 construction order 依次填入空 slots；
4. 不改变 node set、center、target edge、segment、patch 数量或 graph coverage。

目标是 continuous transition 的 shared-slot persistence rate = `1.0`。

## 3. 数据与训练

完全复用 stitched reconstruction 协议：

```text
graph_bank_seed = 810001
cover_seed = 830101
72 graphs, deterministic 3-fold
s=10, overlap=5
K=24, T=3, T_min=1, updates=25
PCA rank=3
threshold=0.5
```

每个 branch 独立使用相同 train graph IDs 拟合 coordinate mean、INIT、FINAL 和 PCA3；test graphs 只编码。

## 4. Invariants

每张图必须满足：

- 两个 branches 的 position-wise node sets 完全相同；
- node/edge/pair coverage 完全相同；
- persistent continuous shared-slot persistence = `1.0`；
- RAW observed RMSE/disagreement = 0；
- mapped replay patch adjacency/transition maps = 1。

## 5. Registered gate

`persistent_slot_order` 通过必须同时满足：

1. 3/3 folds FINAL patch error < INIT；
2. mean patch INIT→FINAL relative reduction >= `0.10`；
3. persistent FINAL observed RMSE 比 construction FINAL 至少低 `0.02` relative；
4. persistent FINAL overlap disagreement 比 construction FINAL 至少低 `0.02` relative；
5. persistent FINAL observed F1 与 full edge recall 均不低于 construction FINAL；
6. persistent FINAL observed RMSE 低于 persistent PCA3；
7. 每折 nondead >=20/24、maximum activation share <0.50。

## 6. 判定

- `PASS_SLOT_PERSISTENT_KSVD`：全部通过；continuous local slots 成为后续方法定义；
- `PERSISTENT_COORDINATES_HELP_BUT_BELOW_GATE`：相对 construction 的 RMSE/disagreement/F1 比较通过，但 10% optimization gate 未通过；保留为表示改进，不升级强 KSVD claim；
- `REJECT_SLOT_PERSISTENT_ORDERING`：没有稳定改善，说明 transition correspondence 不能仅靠重排交给 vanilla KSVD；
- `FAIL_SLOT_PERSISTENCE_INVARIANTS`：node sets、coverage、RAW 或 mapped replay 不一致。

## 7. 边界

slot persistence 是 traversal-relative coordinate transport，不是图的唯一 canonical coordinates。通过也不能说明不同图的 slot 0–9 具有共同绝对语义。
