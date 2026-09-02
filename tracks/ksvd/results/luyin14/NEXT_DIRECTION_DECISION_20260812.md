# luyin14 后续方向决策：读出、structured pursuit 与 RAW relation

> 日期：2026-08-12  
> 范围：在 `ROUTE_CLOSURE_20260812.md` 之后继续审计并运行最小可证伪实验。  
> 总结：当前 adjacency-patch KSVD 下游路线已足够闭环；下一步必须更换研究问题，而不是继续修同一 readout/fusion。

## 1. 这次实际继续探索了什么

先审计仓库，避免重复已有路线：

- motif-slot 已经继续到 connected occurrence / atom--occurrence incidence；不是空白方向；
- masked-context SSL latent patch metric 已经做过，KSVD held-out reconstruction 明显优于
  PCA/random，但下游没有 KSVD-specific 增益；
- dictionary consensus、Hungarian alignment、code averaging、reconstruction confidence
  weighting 已经做过，结果中性或负；
- real-patch occurrence bipartite 有弱正机制信号，但 vocabulary/generalization 仍是瓶颈。

因此本轮只运行三个此前仍有独立信息增益的 screen。

### 1.1 MolHIV-style rich sparse-code readout 迁移

固定原 luyin14 sampler、FAIR95 patches、`K24/T3/u5`、folds 与线性头，只把 coarse
`usage/mean/RMS` 换成完整 10 组 code 分布统计、逐 patch 重构误差和 patch count；同时强制
比较同一 patch/centering 下的 INIT 与 FINAL。

| gate | 结果 |
|---|---|
| FINAL rich > FINAL coarse | **通过**：IMDB-BINARY `+0.039`、IMDB-MULTI `+0.024`，均 9/9 wins |
| FINAL rich > INIT rich | **失败**：四数据集均未达到 `+0.01 & 6/9`；PTC_MR 为 `-0.013` |
| STATS+FINAL rich > STATS | **失败**：`-0.033/+0.004/-0.090/-0.027` |

准确解释：粗读出确实丢信息，但这些额外信息不能稳定归因于 K-SVD updates，也没有越过简单
graph statistics。IMDB 上 rich 的主要收益还大量来自 reconstruction/count sidecar：相对
rich-no-recon，IMDB-BINARY `+0.033`、IMDB-MULTI `+0.016`。

### 1.2 Relation-conditioned structured pursuit

固定一档 `rho=0.25`、两轮 support refinement，让真实 overlap/chain 邻居的上一轮 supports
参与当前 patch 的 atom ranking；设置保持关系图权重/度/谱、只打乱 patch binding 的
SHUFFLED control。

| dataset | TRUE−independent | TRUE−SHUFFLED | support agreement | recon change |
|---|---:|---:|---:|---:|
| IMDB-BINARY | `+0.006` | `+0.011` | `0.356→0.488` | `+271%` |
| IMDB-MULTI | `-0.008` | `-0.002` | `0.327→0.414` | `+225%` |
| MUTAG | `-0.018` | `-0.014` | `0.118→0.263` | `+66%` |
| PTC_MR | `-0.003` | `+0.020` | `0.097→0.222` | `+33%` |

这个 screen 成功提高了支持一致度，却以大幅破坏重构几何为代价；三个主 gate 全失败。它说明
“邻居 support 更像”本身不是目标，必须在明确 reconstruction budget 下谈 structured coding。
但由于 RAW relation 后续也没有 stats 外增量，本数据组上不值得继续扫描 `rho` 或换 solver。

### 1.3 未压缩 RAW patch relation 终局对照

完全去掉 KSVD，用 rooted-canonical 28D adjacency patch token，比较 TRUE/SHUFFLED binding。

| dataset | TRUE−RAW bag | TRUE−SHUFFLED | STATS+TRUE−STATS |
|---|---:|---:|---:|
| IMDB-BINARY | `+0.043`，9/9 | `+0.006`，6/9 | `-0.033` |
| IMDB-MULTI | `+0.008`，8/9 | `+0.003`，6/9 | `-0.003` |
| MUTAG | `+0.027`，6/1/2 | `-0.005` | `-0.101` |
| PTC_MR | `-0.001` | `-0.001` | `-0.020` |

TRUE 相对弱 bag 在两个数据集通过，但 TRUE 不稳定胜 SHUFFLED，且所有数据集都没有越过
graph stats。故不能把现有失败简单归因于 K-SVD compression，也不应据此进入 Transformer
或 patch-graph message passing。

## 2. 现在的方法为什么不行

综合所有证据，当前瓶颈不是单点实现 bug，而是四层错位。

### 2.1 任务错位

IMDB 的主要标签信号可被 size/degree/triangle 等低阶统计读取；MUTAG/PTC_MR 又有强节点标签。
当前 adjacency patch channel要么重复这些统计，要么在小样本上线性拼接高维噪声。它不是这些
任务上不可替代的信息源。

### 2.2 目标错位

K-SVD 优化局部欧氏重构，不优化类别相关性、跨 scaffold 稳定性或 patch composition。
FINAL reconstruction 可以优于 INIT/PCA/random，却不保证 FINAL graph classifier 优于 INIT。
本轮 rich 的 FINAL−INIT 接近 0，再次直接确认这个错位。

### 2.3 对象错位

bag/rich readout 丢掉 occurrence 和连接位置；但强行加 relation 或 support smoothness 又会把
局部 patch 差异抹平。MolHIV 的真实原型二部传播说明 occurrence 是有价值的对象，但该信号
较弱，且不能弥补 vocabulary 跨 scaffold 泛化。

### 2.4 评估错位

小 TUD 数据上 240--300 维结构通道相对样本数过大，graph stats/节点特征基线已经很强。
在这里继续堆 readout、MLP 或 Transformer，最可能得到的是 fold-specific 方差，而不是更清楚
的方法归因。

## 3. 可能方向排序

### 第一优先：更换下游问题，构造“关系确实必要”的 matched benchmark

这是当前唯一能同时保留 patch/KSVD 研究初衷、又不和已失败证据正面冲突的方向。

首个 benchmark 应满足：

1. 图大小、边数、degree histogram、triangle/cycle counts 在类别间匹配；
2. 标签只由相同 local patches 的不同空间组合/远程关系决定；
3. train/test 使用 motif-composition 或 graph-family held-out split；
4. 先要求 `RAW_TRUE > RAW_SHUFFLED > RAW_BAG`，再测试 INIT/FINAL KSVD 是否保留该增益；
5. 最后才加入真实数据，不先训练 Transformer。

这个方向回答的新问题是：

> 当低阶统计被控制、任务必须读取 patch composition 时，稀疏字典能否在压缩率与关系任务
> 性能之间形成可解释 Pareto，而不是“在任意 TUD 分类上是否加分”。

它允许 KSVD 以 compressor/denoiser 身份形成可辩护贡献；若 FINAL 在固定 rate 下仍低于 RAW/
INIT，则可以干净停止整个 downstream claim。

### 第二优先：multi-cover consistency，目标改为稳定表示而非立刻提分类

rooted-canonical 已修复 local slot 任意性，但 relabel 后重新采样的 code cosine 仍约 `0.70`；
真正未完整闭环的是 sampler/view variance。可以对同一图生成 2--4 个 deterministic covers，做：

- cover-level invariant token matching；
- consensus/average readout；
- masked-patch relation consistency；
- INIT/FINAL 与 RAW matched controls。

晋级条件应首先是 relabel+resample cosine `>=0.90`、重构误差恶化 `<=5%`，然后才看新 matched
benchmark 的 relation task。不要再用 IMDB/MUTAG 单独分类增益选择 consistency 权重。

### 第三优先：若允许弱化 KSVD 核心，转向 observed real-prototype occurrence

仓库中最可靠的“高阶对象”正证据来自真实 observed prototypes：atom--occurrence bipartite
相对 matched node MIL 在 8k scaffold development 为 `+0.016`，full official test ensemble
仍为 `+0.0033`。这说明 incidence mechanism 有效，但当前不是强最终模型。

若走这条路，方法核心应明确改为：

```text
permutation-invariant real local prototype
-> connected occurrence
-> bounded atom--occurrence propagation
```

KSVD 只作为 matched compressor/control，不再要求它提供最终 vocabulary。继续实验必须换新的
未触碰 benchmark 或 official-train-only split；不能再根据已经查看的 MolHIV test 微调。

### 低优先探索：reconstruction-constrained graph-fused coding

本轮否定的是固定 hard support prior，不是所有 structured sparse coding 数学形式。若未来新
benchmark 已先证明 RAW relation 有必要，可以考虑：

```text
min_X ||Y-DX||^2 + lambda * sum_(i,j) w_ij ||x_i-x_j||_1
subject to reconstruction degradation <= 5%
```

但在当前四 TUD 数据上，RAW relation 本身没有 stats 外价值，所以现在实现正式 graph-fused
solver 的信息增益很低，不列为近期任务。

## 4. 明确停止清单

- 不继续扩大 `K/T/iterations/restarts`；
- 不继续 rich readout 变体或非线性 classifier 搜索；
- 不继续同一 TUD 数据上的 concat/gate/token/Transformer；
- 不扫描 structured-pursuit `rho/rounds`；
- 不把 SSL latent metric、dictionary consensus、occurrence lifting 当作未做过的新方向；
- 不因为 IMDB-BINARY 的孤立正值声称 relation 已成立；
- 不恢复 labeled-adjacency global codec 主张。

## 5. 建议的下一项具体工作

如果目标仍是形成新的方法结果，建议下一项工作冻结为：

> **Matched-composition benchmark v0：控制全图低阶统计，只改变 patch 的空间组合；用
> RAW/INIT/FINAL × BAG/TRUE/SHUFFLED 做 3 seeds × 3 held-out-family folds。**

它应先是一个无神经网络的线性-probe机制实验。只有 RAW relation 和 FINAL-vs-INIT 两个 gate
都通过，才实现小型 relation-aware encoder；否则把 KSVD 永久固定为 reconstruction/
compression diagnostic，并结束 downstream 分类路线。

## 6. 产物

- rich readout：`RICH_READOUT_20260812.md` / `rich_readout_20260812.json`；
- structured pursuit：`STRUCTURED_PURSUIT_20260812.md` / `structured_pursuit_20260812.json`；
- RAW relation：`RAW_RELATION_20260812.md` / `raw_relation_20260812.json`；
- 三份结果前冻结协议在 `tracks/ksvd/docs/KSVD_LUYIN14_*_PROTOCOL_20260812.md`；
- runners/tests 在 `tracks/ksvd/code/run_luyin14_*.py` 与 `test_luyin14_*.py`。

