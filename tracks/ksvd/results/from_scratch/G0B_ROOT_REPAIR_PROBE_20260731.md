# G0B-R Root Repair Probe

> 日期：2026-07-31  
> 目的：区分“纯 canonical adjacency 的问题能否由结构性 root 修复”与“必须额外提供稳定 anchor 信息”。不运行 KSVD。

## 1. 汇总

| nuisance | representation | collision mass | Bayes accuracy | robust core F1 | medoid accuracy | gate |
|---|---|---:|---:|---:|---:|---:|
| R1_add_one_noncore_exhaustive | unrooted | 0.4773 | 0.8920 | 0.5909 | 0.5379 | FAIL |
| R1_add_one_noncore_exhaustive | structural_max_degree_root | 0.4773 | 0.8920 | 0.6591 | 0.6705 | FAIL |
| R1_add_one_noncore_exhaustive | oracle_root0 | 0.2386 | 0.9148 | 0.6136 | 0.6875 | FAIL |
| R2_edge_flip_p005 | unrooted | 0.9849 | 0.8263 | 0.7074 | 0.6840 | FAIL |
| R2_edge_flip_p005 | structural_max_degree_root | 0.9849 | 0.8263 | 0.7482 | 0.7039 | FAIL |
| R2_edge_flip_p005 | oracle_root0 | 0.9173 | 0.8921 | 0.7350 | 0.7742 | FAIL |

表示条件：

- `unrooted`：原始 exact canonical adjacency；
- `structural_max_degree_root`：只用 observed patch 的最大度节点作为 root；最大度 tie 仍做置换不变最小化，不增加外部信息；
- `oracle_root0`：生成器 node 0 作为可见 root，模拟 patch extractor 提供稳定 anchor。它不是 motif label，但属于额外 side information。

## 2. 每种 motif 的 robust fixed-core F1

| nuisance | representation | triangle | four_cycle | three_star | five_path |
|---|---|---:|---:|---:|---:|
| R1_add_one_noncore_exhaustive | unrooted | 0.8333 | 0.9318 | 0.8333 | 0.5909 |
| R1_add_one_noncore_exhaustive | structural_max_degree_root | 1.0000 | 0.8182 | 0.9444 | 0.6591 |
| R1_add_one_noncore_exhaustive | oracle_root0 | 1.0000 | 0.8182 | 0.9444 | 0.6136 |
| R2_edge_flip_p005 | unrooted | 0.8588 | 0.8287 | 0.8005 | 0.7074 |
| R2_edge_flip_p005 | structural_max_degree_root | 0.9307 | 0.7674 | 0.8895 | 0.7482 |
| R2_edge_flip_p005 | oracle_root0 | 0.9550 | 0.8496 | 0.9248 | 0.7350 |

## 3. 判定逻辑

1. 如果 `structural_max_degree_root` 修复，则可以继续研究无需额外语义的 rooted structural patch；
2. 如果只有 `oracle_root0` 修复，则 route 需要一个由采样器、节点类型或领域属性提供的稳定 anchor；
3. 如果 oracle root 仍失败，则单个 root 不足以定义跨 variant 的固定 adjacency motif atom；应改用 typed representation、invariant statistics 或弱化 atom 解释。

## 4. 当前结论

Neither a graph-observable max-degree root nor the generator-supplied root makes all noisy motif families identifiable with stable fixed core coordinates. Do not proceed to noisy KSVD with a fixed adjacency-mask atom interpretation.
