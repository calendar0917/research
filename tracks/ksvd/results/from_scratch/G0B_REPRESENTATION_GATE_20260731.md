# G0B-R Canonical Representation Gate 结果

> 日期：2026-07-31  
> 本实验不运行 KSVD；它只检查加入 motif variation 后，canonical adjacency 是否仍允许定义稳定、可辨识的 motif atom。

## 1. 总判定

**FAIL_NOISY_CANONICAL_REPRESENTATION**

The clean control passes, but at least one noisy motif condition loses label identifiability or fixed core-coordinate semantics. Do not interpret a subsequent KSVD failure as an optimization failure.

## 2. 条件汇总

| condition | variants | collision mass | Bayes accuracy | robust fixed-core F1 (worst motif) | medoid macro accuracy | gate |
|---|---:|---:|---:|---:|---:|---:|
| R0_clean | 4 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | PASS |
| R1_add_one_noncore_exhaustive | 46 | 0.4773 | 0.8920 | 0.5909 | 0.5379 | FAIL |
| R2_edge_flip_p005 | 8000 | 0.9849 | 0.8263 | 0.7074 | 0.6840 | FAIL |

## 3. Hidden core 在 canonical coordinates 中是否稳定

| condition | motif | variants | role-identifiable rate | deterministic fixed F1 | optimistic fixed F1 | robust fixed F1 |
|---|---|---:|---:|---:|---:|---:|
| R0_clean | triangle | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| R0_clean | four_cycle | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| R0_clean | three_star | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| R0_clean | five_path | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| R1_add_one_noncore_exhaustive | triangle | 12 | 1.0000 | 0.8333 | 0.8333 | 0.8333 |
| R1_add_one_noncore_exhaustive | four_cycle | 11 | 1.0000 | 0.9318 | 0.9318 | 0.9318 |
| R1_add_one_noncore_exhaustive | three_star | 12 | 0.8333 | 0.8333 | 0.8889 | 0.8333 |
| R1_add_one_noncore_exhaustive | five_path | 11 | 0.1818 | 0.6136 | 0.6818 | 0.5909 |
| R2_edge_flip_p005 | triangle | 2000 | 0.9310 | 0.8588 | 0.8960 | 0.8588 |
| R2_edge_flip_p005 | four_cycle | 2000 | 0.8975 | 0.8324 | 0.8639 | 0.8287 |
| R2_edge_flip_p005 | three_star | 2000 | 0.7605 | 0.8148 | 0.8753 | 0.8005 |
| R2_edge_flip_p005 | five_path | 2000 | 0.5350 | 0.7340 | 0.7845 | 0.7074 |

其中：

- `role-identifiable rate`：canonical graph 的所有最小排列是否都把 hidden core 映射到同一个 support；
- `deterministic`：使用实现选出的第一个最小排列；
- `optimistic`：对每个 graph automorphism 选择最有利的 hidden-core mapping；
- `robust`：要求固定 support 对所有等价 mapping 都成立，是进入显式 motif atom 学习最严格、最可信的指标。

## 4. Cross-family collisions

### R0_clean: 0 个 collision vectors，equal-prior mass=0.0000

无 cross-family collision。

### R1_add_one_noncore_exhaustive: 2 个 collision vectors，equal-prior mass=0.4773

- motifs=['triangle', 'three_star']，mass=0.250000，vector=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
- motifs=['four_cycle', 'five_path']，mass=0.227273，vector=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0]

### R2_edge_flip_p005: 57 个 collision vectors，equal-prior mass=0.9849

- motifs=['triangle', 'four_cycle', 'three_star', 'five_path']，mass=0.128000，vector=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0]
- motifs=['triangle', 'three_star']，mass=0.123000，vector=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1]
- motifs=['triangle', 'four_cycle', 'five_path']，mass=0.116750，vector=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]
- motifs=['triangle', 'three_star', 'five_path']，mass=0.116500，vector=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
- motifs=['triangle', 'four_cycle', 'three_star', 'five_path']，mass=0.075625，vector=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
- motifs=['four_cycle', 'three_star', 'five_path']，mass=0.062000，vector=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0]
- motifs=['triangle', 'four_cycle', 'three_star', 'five_path']，mass=0.052750，vector=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1]
- motifs=['triangle', 'four_cycle', 'three_star', 'five_path']，mass=0.047250，vector=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1]
- 其余 49 个见 JSON。

## 5. 科学解释

G0 的 clean patches 只有四种 exact canonical vectors，因此 maximin initialization 可以直接枚举 vocabulary。G0B-R 检查的是：一旦同一 motif 允许结构变化，生成器定义的 hidden motif family 是否仍由无类型邻接图唯一决定，以及 hidden core 是否仍位于一致的线性坐标。

如果 noisy condition 未通过 representation gate，后续直接运行 KSVD 将无法区分：

1. KSVD 没有学到 motif；
2. 相同观测图对应多个 hidden motif 解释；
3. exact canonicalization 虽然对单图置换不变，但不同 variants 的 core 坐标不一致。

此时正确动作不是增加 restart，而是重新定义表示或研究命题，例如使用 rooted/typed canonicalization、显式节点/边属性，或放弃“atom 是固定 adjacency edge mask”的强解释。
