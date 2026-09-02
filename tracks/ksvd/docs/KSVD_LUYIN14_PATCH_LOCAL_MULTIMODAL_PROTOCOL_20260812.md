# luyin14：patch-local KSVD × node-feature 多模态对齐协议

> 日期：2026-08-12  
> 状态：结果前冻结  
> 数据：MUTAG / PTC_MR，不构造新数据集。

## 1. 与已有融合的区别

已有 concat 只把全图 node-feature readout 与全图 KSVD readout 拼接。它不知道：

> 某个 dictionary atom 的激活，具体发生在具有哪些节点特征组成的 patch 上。

本轮借鉴 compact bilinear multimodal pooling，在 patch 层绑定两种模态，再做低维图级读出。

## 2. 冻结局部对齐表示

对每张图：

1. FAIR95 patch `p` 的节点特征组成 `f_p`：patch 成员节点原始 feature 的均值；
2. KSVD code activation `a_p=|x_p|`；
3. 在图内分别对每个 atom activation 和每个 feature channel 做中心化/RMS normalization；
4. 计算 compact bilinear correlation：`C = A_norm^T F_norm / n_patches`，维度 `K×F`；
5. flatten `C`。outer-train TRUE matrices 上拟合 PCA16；TRUE/SHUFFLED train/test 均使用同一
   TRUE-train PCA basis；
6. 最终分类输入为 `FEATURE_STATS + PCA16(C)`。

`SHUFFLED` 在每张图内部置换 patch feature rows，只打破 code 与 patch feature 的对应；两种
模态的边际分布、patch count、dictionary、PCA basis 与 classifier capacity保持不变。

INIT 与 FINAL 分别独立计算 cross matrix 和 train-only PCA，不共享 basis；必须成对报告。

## 3. 固定协议

- 数据集：MUTAG、PTC_MR；
- FAIR95/rooted-canonical、`K24/T3/iterations5`；
- split seeds `0/1/2`，3-fold；
- dictionary、PCA、scaler、LR 均为 outer-train only；
- PCA rank 固定 16，不扫描；classifier 仍为 `C=1` logistic；
- 同时重报 `FEATURE_STATS` 与 naïve `FEATURE_STATS+FINAL_RICH`。

## 4. 晋级规则

1. `FINAL_TRUE_CROSS - FEATURE_STATS`：至少一个数据集 mean `>=+0.01` 且 wins `>=6/9`，
   另一个不低于 `-0.01`；
2. `FINAL_TRUE_CROSS - FINAL_SHUFFLED_CROSS`：同样满足上述跨数据集规则；
3. `FINAL_TRUE_CROSS - INIT_TRUE_CROSS` 两数据集 mean 平均为正，且至少一个数据集为正；
4. TRUE cross 必须优于同折 naïve rich concat 的 mean。

四项同时通过，才进入 patch-token cross-attention/FiLM；否则现有 patch/node-feature 局部绑定
没有可迁移线性信号，不通过增加 attention heads 或 hidden width补救。

