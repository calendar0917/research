# 导师 50-node 真实子图：Grouped KSVD 人工结论与 relation-token 下一步

> 日期：2026-08-06  
> 自动报告：`tracks/ksvd/results/mentor_subgraphs/GROUPED_KSVD_FOLLOWUP_20260806.md`  
> 正式判定：`KSVD_GROUPED_RECONSTRUCTION_READY_FOR_RELATION_TOKEN_ABLATION`

## 1. 可以确认的结论

### 1.1 普通 KSVD 在真实子图上具有稳定的 train-only added value

在 root-candidate-grouped 主视图中，六个 geometry/checkpoint branches 均满足：

- 3/3 folds `FINAL < INIT`；
- INIT→FINAL graph-count-weighted patch-error reduction 为 `21.5%–30.8%`；
- 24/24 atoms 在所有 FINAL folds 中均为 nondead；
- maximum activation share 仅约 `0.060–0.105`，没有单 atom 坍缩；
- FINAL observed-pair RMSE 在每折均不差于 INIT；
- FINAL 同时优于 train-patch RANDOM dictionary 与 train-only PCA3；
- random-reference 上改善同方向，没有 root-grouped 系统性反转。

这比此前 synthetic Beam8 audit 中约 5–6% 的弱 KSVD gain 更强。当前最稳妥的定位是：

> 在 frozen rooted-canonical Beam8 patches 上，普通 K24/T3 KSVD 是一个真实、稳定的 sparse compressor；它值得进入 matched relation-token attribution，但仍不等价于语义 motif learner。

### 1.2 Root grouping 没有消除 KSVD gain

root-candidate-grouped 三折 root intersection 严格为 0，而各 branch 的 FINAL reconstruction 与 random-reference 非常接近。例如：

- `s8/o2 BASE` patch error：random `0.2251`，grouped `0.2265`；
- `s10/o3 BASE`：random `0.3071`，grouped `0.3061`；
- `s12/o4 BASE`：random `0.3710`，grouped `0.3693`。

因此本轮没有看到 KSVD gain 主要来自相同 root 的直接记忆。

但这不是 source-disjoint 泛化：root-grouped test source nodes 中约 `89.3%–91.2%` 在 train union 中出现，test source edges 中约 `80.6%–83.2%` 已在 train 出现。报告必须继续称为 root-candidate-grouped/transductive-overlap-aware，而不是完全 inductive source-graph generalization。

### 1.3 Canonical patch 模板重复随 patch size 快速下降

root-grouped 下 exact-seen test patch occurrence：

- `s8/o2`：BASE `0.7476`，FAIR95 `0.7054`；
- `s10/o3`：BASE `0.3423`，FAIR95 `0.2913`；
- `s12/o4`：BASE `0.1694`，FAIR95 `0.1320`。

这说明小 patch vocabulary 更重复、更容易复用；大 patch 的 exact structural templates 更长尾。尽管如此，三种 geometry 的 KSVD optimization gate 都通过，因此大 patch branch 的 gain 不能仅解释为 exact patch memorization。

## 2. Geometry 与 operating point

### 2.1 若 reconstruction quality 优先，`s8/o2` 是明确首选

root-grouped FINAL：

| branch | patch err | observed RMSE | full RMSE | corrected RMSE | full recall | dict/code scalars |
|---|---:|---:|---:|---:|---:|---:|
| s8/o2 BASE | 0.2265 | 0.1849 | 0.1816 | 0.1067 | 0.8864 | 672 / 59.6 |
| s8/o2 FAIR95 | 0.2504 | 0.1943 | 0.1466 | 0.1238 | 0.9591 | 672 / 70.0 |
| s10/o3 BASE | 0.3061 | 0.2378 | 0.2185 | 0.1433 | 0.8536 | 1080 / 39.6 |
| s10/o3 FAIR95 | 0.3277 | 0.2444 | 0.1817 | 0.1655 | 0.9315 | 1080 / 50.8 |
| s12/o4 BASE | 0.3693 | 0.2754 | 0.2445 | 0.1739 | 0.8155 | 1584 / 28.9 |
| s12/o4 FAIR95 | 0.3914 | 0.2792 | 0.2114 | 0.1992 | 0.8899 | 1584 / 38.8 |

`s8/o2` 的 dictionary 最小，同时 reconstruction最好；代价是 per-graph codes最多。因此：

- 默认质量端点：`s8/o2`；
- 最短 code 端点：`s12/o4`；
- `s10/o3` 是中间成本 control。

自动 Pareto 将六个 branches 全保留是正确的，因为 dictionary size、code length、uncorrected/corrected error 与 recall 之间没有统一总标量；但这不代表六个 branch 同等适合作为默认下游输入。

### 2.2 BASE 与 FAIR95 回答的是两种不同系统目标

相对 BASE，FAIR95 在三个 geometry 上：

- uncorrected full RMSE 降低约 `13.5%–19.3%`；
- full edge recall 提升约 `0.073–0.078`；
- 但 patch error 增加约 `6.0%–10.5%`；
- observed-pair RMSE 增加约 `1.4%–5.1%`；
- residual-corrected full RMSE 增加约 `14.5%–15.9%`；
- per-graph code scalars 增加约 `17.6%–34.5%`。

因此：

- **patch-only / 不提供 residual sidecar**：FAIR95 更适合，因为它显著降低未观察边造成的 full-graph error；
- **显式 residual edge-token/sidecar 系统**：BASE 更适合，因为 residual被独立暴露后，BASE 的 patch compression error更低、code更短。

不能把 FAIR95 的高覆盖解释为对所有系统都更优。

## 3. Density 解释

所有 branches 在五个 strata 都保持 FINAL reconstruction有效，但误差随密度有明显结构：

- `s8/o2 BASE` FINAL patch error 从 `lt5≈0.265`、`d5_10≈0.262` 下降到 `ge25≈0.176`；
- `s10/o3 BASE` 从约 `0.359/0.359` 降到 `0.240`；
- `s12/o4 BASE` 从约 `0.417/0.437` 降到 `0.299`。

大 patch 在稀疏图上更难压缩；小 patch 对整个密度范围更稳。下一阶段的主表仍需按 density strata 报告，不应只用 pooled mean。

## 4. 下一阶段建议：只做 matched relation-token attribution

本轮通过后，允许进入一次低容量、冻结的 downstream mechanism audit，但不建议立即进行大规模 Transformer 搜索。

### 4.1 主 branches

优先比较：

1. `s8/o2 BASE`：默认 compression + explicit residual-token 系统；
2. `s8/o2 FAIR95`：patch-only 高覆盖对照；
3. `s12/o4 BASE`：短 code Pareto端点；
4. `s10/o3 BASE`：中间 geometry control，可在计算受限时放入附表。

不需要首轮把六个 branches 全部扩展成复杂模型。BASE/FAIR95 的机制差异比同 checkpoint 下三 geometry 的穷举更重要。

### 4.2 必须匹配的 token controls

同一 folds、同一 covers、同一 readout容量下比较：

- `RAW_PATCH`：rooted-canonical raw adjacency token；
- `PCA3`：train-only dense rank-3 token；
- `RANDOM_T3`：random train-patch dictionary sparse code；
- `INIT_T3`：deterministic maximin sparse code；
- `FINAL_T3`：learned KSVD sparse code；
- `FINAL_T3 + RELATION`：加入 overlap slot map、center distance、segment/transition；
- `FINAL_T3 + RELATION + RESIDUAL_EDGE_TOKEN`：仅用于 BASE；
- `SHUFFLED_RELATION`：关系绑定打乱的必要负对照。

关键 attribution 是：

```text
FINAL - INIT                 = dictionary learning added value
FINAL - PCA3/RANDOM          = sparse learned basis added value
TRUE_RELATION - SHUFFLED     = relation binding added value
BASE+RESIDUAL - FAIR95       = explicit residual channel vs moving edges into patch channel
```

### 4.3 下一轮 go/no-go

只有以下条件同时满足，才值得进入更强 Transformer/GNN：

- `FINAL_T3` 在 root-grouped held-out downstream metric 上稳定优于 `INIT_T3`；
- `TRUE_RELATION` 稳定优于 `SHUFFLED_RELATION`；
- BASE residual-token branch 不依赖解码时隐藏地使用 ground-truth residual，输入端必须显式可见；
- gain 在至少四个 density strata 不发生系统性反向；
- model capacity、训练预算和 early stopping 完全匹配。

若 reconstruction gain未转化为 downstream gain，应保留 KSVD 为 compressor/baseline，不扫描 K/T/深度来救结果。

## 5. 仍需向导师确认

1. 文件名 `subgraphs_50_20_...` 中 `20` 的含义；
2. first insertion node 是否是真实 sampling root；
3. 10000 图是否来自同一张 2805-node source graph；
4. 是否有其他 batch、sampling seed、labels 或下游任务定义。

没有 graph/patch/downstream labels 时，当前结果只能支持 unsupervised reconstruction substrate，不能独立完成监督 relation-token 效用判定。
