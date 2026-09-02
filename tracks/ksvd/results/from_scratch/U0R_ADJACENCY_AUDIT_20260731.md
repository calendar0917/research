# U0-R 邻接表示审计结果

> 日期：2026-07-31
>
> 本实验不训练 KSVD；它检查 exact canonicalization 是否不仅同构不变，而且具有可供线性字典学习使用的结构几何。

## 1. 总判定

**SELECT_WALK_ORDER_PENDING_U0P**

Canonical adjacency is invariant but its linear geometry fails at least one gate. Walk-order adjacency preserves sampler-slot semantics and is eligible, pending the separate U0-P signal-exposure gate.

## 2. 三个 audit seeds

| seed | patches | CAN invariance | CAN one-flip mean | CAN amplify >1 | CAN severe >=4 | CAN GED Spearman | WALK GED Spearman | WALK relabel invariance |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 732001 | 960 | 1.0000 | 3.2567 | 0.6208 | 0.3692 | 0.5792 | 0.6342 | 1.0000 |
| 732002 | 960 | 1.0000 | 3.2728 | 0.6238 | 0.3754 | 0.6402 | 0.6985 | 1.0000 |
| 732003 | 960 | 1.0000 | 3.2682 | 0.6217 | 0.3728 | 0.5890 | 0.6827 | 1.0000 |

## 3. Generator / sampler sanity

| seed | graph failures | swap attempts mean | trace length mean/p95/max | induced edges mean/min/max |
|---:|---:|---:|---:|---:|
| 732001 | 0 | 57.45 | 8.05/12.00/21 | 5.56/5/9 |
| 732002 | 0 | 57.35 | 7.92/12.00/21 | 5.54/5/9 |
| 732003 | 0 | 56.81 | 8.12/12.00/35 | 5.55/5/9 |

## 4. 如何理解两个表示

- `R-CAN` 的成功点是：同一个 rooted isomorphism class 只有一个 vector；它不自动保证相邻 graph classes 在 15 维空间中也相邻。
- `R-WALK` 的 coordinate 不是 canonical graph role，而是首次发现顺序。它牺牲同一 subgraph 在不同 walk 下的唯一表示，换取固定 sampler-slot 语义和严格的一条边/一个坐标连续性。
- continuous KSVD atom 不需要本身是合法 adjacency；真正需要避免的是输入坐标因 canonical relabeling 产生大幅、无结构依据的跳变。

## 5. 下一步

以 R-WALK 进入 U0-P signal-exposure gate；在 U0-P 通过前不运行 KSVD。R-CAN 仅作为不变性 baseline。
