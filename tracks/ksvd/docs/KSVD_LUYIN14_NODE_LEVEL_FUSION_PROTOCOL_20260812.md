# luyin14：节点级 KSVD–属性融合 strict screen

> 日期：2026-08-12  
> 状态：结果前冻结  
> 数据：MUTAG / PTC_MR；不构造数据集。

## 1. 研究问题

前述 luyin14 主要把 KSVD codes 池化成图向量再分类。导师所说“把系数表征拼接到节点
特征后面”对应另一种接入方式：每个节点拥有自己的局部结构 code，结构模态在 GNN 消息
传递之前或过程中与属性模态交互。

旧 MUTAG 自检使用 test-fold epoch maximum，且没有 PTC_MR、INIT、节点打乱绑定与 strict
checkpoint。本轮只回答：

> 严格 train-only checkpoint 下，节点级 KSVD code 的 concat 或 FiLM 是否稳定补充属性 GIN？

## 2. 节点结构 token

对每个节点 `v`：

- patch 为 `v ∪ N(v)`，最多 8 节点；超限时按 train-label-free 的 degree/global-WL key截断；
- `v` 是 rooted singleton color，patch 内使用 exact rooted-canonical slots；
- vector 为 8×8 padded adjacency upper triangle（28D）；
- dictionary 仅由 outer-train graph 的 node patches 拟合；`K24/T3/iterations5`；
- 同一 centering 下分别编码 deterministic INIT 与 FINAL，token 为 `|x_v|`；
- token normalization 只使用 outer-train nodes。

`SHUFFLED` 在每张图内部置换 node tokens，保留 token multiset、节点特征和图结构，只打破
local structure code 与节点位置/属性的绑定。

## 3. 分类模型

共同 backbone：3-layer GIN、hidden64、layer-wise graph sum readout、dropout0.5。

- `GIN_ONLY`：原始 node attributes；
- `FINAL_CONCAT_TRUE`：输入 `[x_v; s_v]`；
- `FINAL_FILM_TRUE`：每层后使用 `s_v` 产生 zero-init bounded FiLM residual；
- `FINAL_FILM_SHUFFLED`：同容量、打乱 node binding；
- `INIT_FILM_TRUE`：与 FINAL FiLM 相同，只有 dictionary 未做 K-SVD updates。

FiLM 形式：

`h <- h * (1 + 0.1*tanh(gamma(s))) + 0.1*beta(s)`，gamma/beta 输出层零初始化。

## 4. Strict checkpoint

- Stage A：outer split seed 0，3 folds，model seed 0；
- 每个 outer-train 内固定 80/20 stratified inner validation；
- 最多 100 epochs、patience 25，只用 inner-valid balanced accuracy 选 epoch；
- 重置模型、optimizer 和 DataLoader RNG，在完整 outer-train 上训练恰好 selected epoch；
- outer-test 只评估一次；同 fold/mode 使用固定 seeds；
- loss 为普通 cross entropy；主指标 balanced accuracy。

## 5. Stage A 晋级

FiLM 只有同时满足才进入 split seeds 1/2：

1. `FINAL_FILM_TRUE - GIN_ONLY` mean `>=+0.01` 且至少 2/3 folds 为正；
2. `FINAL_FILM_TRUE - FINAL_FILM_SHUFFLED` mean `>=+0.005` 且至少 2/3 为正；
3. `FINAL_FILM_TRUE - INIT_FILM_TRUE` mean为正；
4. 至少一个数据集通过 1--3，另一个相对 GIN_ONLY 不低于 `-0.01`。

concat 作为旧建议的 strict replication，不单独授权扩展；只有 FiLM gate 通过才继续。若失败，
说明当前局部 adjacency KSVD code 即使以正确节点位置接入 GNN，也没有稳定互补性；下一步应
改变 patch token 语义，而不是增加 cross-attention heads。

