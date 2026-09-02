# 真实纯结构数据上的 KSVD go/no-go 结论（2026-07-30）

> 这份结论回答的是：**普通欧氏 KSVD 能否被当作“显式、自适应、可解释的结构词汇学习器”继续作为主线？** 它不回答最终 MolHIV/CIN 上限，也不否定“学习显式高阶结构”这个更大的研究命题。

## 1. 已完成的证据链

### 数据与协议

- 真实、structure-only 数据：IMDB-BINARY、IMDB-MULTI、REDDIT-BINARY；不使用节点/边属性。
- IMDB 使用 cleaned 版本作为主要机制判断，避免同构重复图膨胀。
- 主要评估：3 split seeds × 5-fold outer CV；所有 KSVD/PCA/dictionary 仅在 outer-train 拟合。
- 条件增益：分类器正则 C 由 outer-train 内 3-fold CV 选择；控制 graph size、edge count、density、degree histogram、component、triangle、cycle-rank 等统计。
- matched controls：raw patch、PCA、random real columns、clustered real-patch dictionary，以及给每类 content 拼接**同一个** relation-graph block。
- 语义审计：atom nearest-real-patch、negative mass、projectable proxy、结构 signature purity、跨折 Hungarian matching。

### 产物

- 初始真实数据审计：`docs/REAL_STRUCTURE_KSVD_IMDB_3SEED_20260730.md`
- patch substrate：`docs/REAL_STRUCTURE_PATCH_AUDIT_20260730.md`
- IMDB matched conditional：`docs/REAL_STRUCTURE_CONDITIONAL_IMDB_MATCHED_20260730.md`
- REDDIT R2/max_nodes=24 matched conditional：`docs/REAL_STRUCTURE_CONDITIONAL_REDDIT_R2_N24_MATCHED_20260730.md`
- IMDB atom audit：`docs/REAL_STRUCTURE_ATOM_AUDIT_20260730.md`
- REDDIT atom audit：`docs/REAL_STRUCTURE_ATOM_AUDIT_REDDIT_R2_N24_20260730.md`
- projected-KSVD 最终门槛：`docs/REAL_STRUCTURE_PROJECTED_KSVD_20260730.md`

## 2. 结果总表

下表均为 balanced accuracy。`KSVD+graph`、`real+graph`、`raw+graph` 使用相同 relation-graph block，因此其差异才可用于判断 dictionary-specific gain。

| 设置 | stats | relation graph | raw+graph | real+graph | KSVD content | KSVD+graph | true atom-position relation |
|---|---:|---:|---:|---:|---:|---:|---:|
| IMDB-BINARY cleaned/B0 | 0.7533 | 0.7466 | **0.7654** | 0.7590 | 0.7479 | 0.7600 | 0.7413 |
| IMDB-MULTI cleaned/B0 | 0.5009 | 0.5055 | 0.5443 | 0.5545 | 0.5372 | **0.5570** | 0.5211 |
| REDDIT-BINARY raw/R2, max_nodes=24 | 0.8432 | **0.8550** | 0.8188 | 0.8372 | 0.8323 | 0.8490 | 0.8083 |

补充尺度检查：

- REDDIT B0：relation graph=0.8575，KSVD+graph=0.8375；B0 patches 中 61.9% 只是单边。
- REDDIT R2/max_nodes=12：relation graph=0.8610，KSVD+graph=0.8488；62.0% patches 触及 12-node 上限。
- 将 R2 上限扩大到 24 后，KSVD 仍未超过 relation-graph-only；因此结论不是单纯由 max_nodes=12 截断造成。

## 3. 分问题判断

### Q1：KSVD 是否提供超出 size/degree 的条件信息？

**不稳定。**

- IMDB-BINARY：KSVD content 相对 stats 为 -0.0055；raw patch 为 +0.0135。
- IMDB-MULTI：KSVD content 为 +0.0362，但 real-patch content 为 +0.0529。
- REDDIT R2/max_nodes=24：KSVD content 为 -0.0108。

因此能确认的是“某些 patch 表征包含额外信息”，不能确认“该信息来自 KSVD 特有的字典学习”。

### Q2：在 matched relation block 下，KSVD 是否优于 PCA/raw/real-patch？

**只有一个边缘正例，且不跨数据集稳定。**

- IMDB-BINARY：KSVD+graph 比 raw+graph 低 0.0053；比 real+graph 高 0.0010，但三个 seed 的方向不一致。
- IMDB-MULTI：KSVD+graph 比 real+graph 高 0.0025，seed means 为 `[+0.0129, -0.0133, +0.0079]`，仍不稳定。
- REDDIT R2/max_nodes=24：KSVD+graph 比 real+graph 高 0.0118，但 **relation-graph-only 又比 KSVD+graph 高 0.0060**。这说明该设置中可靠增益来自 patch relation topology，而不是 KSVD content。

### Q3：atom 是合法、可解码、跨折稳定的 graphlet vocabulary 吗？

**没有通过。**

| 设置 | KSVD projectable proxy | KSVD nearest-signature agreement | real-patch agreement |
|---|---:|---:|---:|
| IMDB-BINARY cleaned/B0 | 0.554 | 0.717 | 0.733 |
| IMDB-MULTI cleaned/B0 | 0.496 | 0.670 | 0.692 |
| REDDIT-BINARY R2/max_nodes=24 | 0.158 | 0.412 | 0.435 |

- IMDB B0 patch population本身有约 82% clique，within-graph unique fraction 只有约 0.29–0.31；高 top-k purity 很大程度上来自简单团结构重复。
- REDDIT R2 更丰富，但自由 KSVD atom 的 nearest cosine 降到 0.863、negative mass 升到 0.077，只有 15.8% 满足 projectable proxy。
- clustered real-patch dictionary 天然 100% legal/projectable，语义稳定性并不低于 KSVD。

所以“向量方向可匹配”不能升级为“已发现合法 graphlet”。

### Q4：true relation 优于 shuffle，能否证明 composition？

**不能。当前结果恰好说明只做 shuffle control 会误判。**

true relation 确实常高于 shuffled relation，但它相对更低容量的 `content+graph` 明显更差：

- IMDB-BINARY：-0.0187；
- IMDB-MULTI：-0.0360；
- REDDIT R2/max_nodes=24：-0.0407，15/15 folds 全部下降。

解释是：正确 binding 比随机 binding 少一些破坏，但当前 atom-position relation readout 本身仍是有害的高维/错配表示。它没有证明组合结构已被有效利用。

### Q5：有没有值得保留的正信号？

**有，但正信号不属于普通 KSVD。**

REDDIT-BINARY 上 relation-graph-only 在 B0、R2/12、R2/24 三个尺度均稳定高于 stats，增益约 +0.012 到 +0.018，且三个 split seeds 同方向。这支持：

> patch 之间的 overlap/distance/coverage 等显式关系可能包含超出全局 degree statistics 的信息。

但加入 KSVD content 后反而低于 relation-graph-only；因此目前应把它定位为“显式 patch-relation signal”，而不是“KSVD vocabulary + composition 已成立”。

## 4. Go / no-go

### No-go：作为当前论文 headline 的普通 KSVD

以下命题暂时停止：

> 在 WL patch 向量上运行无约束欧氏 KSVD，就能自动学得共享、合法、稳定、可组合的结构词汇，并由此形成对 CIN 有竞争力的新模型。

停止理由不是单次准确率低，而是四条机制证据同时未通过：

1. 没有稳定 KSVD-specific conditional gain；
2. 自由 atom 不保证合法，真实数据越丰富时 projectability 越低；
3. semantic stability 不优于 clustered real patches；
4. exact atom-position relation 在所有主设置中都低于 content+graph。

### Go：保留 KSVD 的有限角色

KSVD 仍可作为：

- patch-space compression/coding baseline；
- 初始化器；
- reconstruction 与 task relevance 错位的诊断工具；
- 产生候选方向后再投影到合法真实 patch 的中间步骤。

但除非后续出现新的机制证据，不再把 unconstrained KSVD atom 直接称为 motif/graphlet/cell。

## 5. Projected-KSVD 最终有界实验（已完成）

已严格执行预注册比较：

1. clustered real-patch medoids；
2. final-projected KSVD：普通 KSVD 后，以 Hungarian matching 投影到 distinct real training patches；
3. iterative projected KSVD：每轮 dictionary update 后投影，并重新 OMP。

所有 projected atoms 均为训练折中真实、非零、signature-distinct 的 normalized patch，因此 `projectability=1`。协议保持 outer 5-fold × 3 split seeds、inner 3-fold 选择分类器 C、16 atoms、T=2、10 iterations，并继续控制 stats。

### 5.1 主要结果

下表为 `stats + content` balanced accuracy：

| 设置 | raw | PCA | clustered real | unconstrained KSVD | final projected | iterative projected |
|---|---:|---:|---:|---:|---:|---:|
| IMDB-BINARY cleaned/B0 | **0.7668** | 0.7549 | 0.7544 | 0.7479 | 0.7536 | 0.7480 |
| IMDB-MULTI cleaned/B0 | 0.5463 | 0.5172 | 0.5538 | 0.5400 | 0.5410 | **0.5673** |
| REDDIT-BINARY R2/max_nodes=24 | 0.8202 | 0.8325 | 0.8252 | 0.8328 | 0.8318 | **0.8333** |

结果不是“projection 完全无效”，而是**没有达到跨数据集、跨 seed 的方法证据**：

- IMDB-BINARY：final/iterative 相对 raw 分别为 -0.0133/-0.0189；没有超过最简单的 raw mean。
- IMDB-MULTI：iterative projected 平均比 clustered real patch 高 +0.0135、比 raw 高 +0.0210，是唯一值得记录的弱正例；但两个比较在 split seed=1 分别为 -0.0129/-0.0223，未通过 3/3 seed 稳定性。
- REDDIT：final/iterative 相对 PCA 分别为 -0.0007/+0.0008，方向随 seed 改变；更重要的是二者 `stats+content` 均低于 stats，而 `relation-graph-only=0.8550` 仍显著高于 projected+graph（0.8360/0.8403）。
- final/iterative projection 均提高了训练 patch reconstruction error；合法性不是免费获得的，但重构变化本身也不能解释分类排序。

### 5.2 预注册门槛结论

两个 projected 变体都没有在任何数据集上同时满足：

- 相对 clustered-real、PCA、raw 三者均为 3/3 split seeds 正向；
- 控制 stats 后保留稳定增益；
- 并在至少两个结构差异明显的数据集复现。

因此最终判定为 **NO-GO**：

> KSVD（包括普通、final-projected、iterative-projected）正式从方法主线降为 baseline、初始化器或候选生成器；不再追加 atoms、T、RW、top-k、projection strength 等普通搜索。

这同时排除了一个重要替代解释：此前失败不只是因为自由 atom 不合法。即使强制每个 atom 都是合法真实 patch，稳定的 KSVD-specific task gain 仍未出现。


### 5.3 不使用 WL 的替代解释复核

为检验上述 no-go 是否只是 WL histogram 信息损失造成，又在完全相同协议下比较了两种 non-WL 表示：

- `canonical`：nauty exact canonical labeling 后的完整 adjacency + node mask；
- `rooted canonical`：进一步保留 sampler center 的同构角色。

完整证据见：

- `docs/REAL_STRUCTURE_NONWL_AUDIT_20260730.md`；
- `docs/CANONICAL_VS_WL_PROJECTED_KSVD_20260730.md`；
- `docs/NONWL_PROJECTED_KSVD_FINAL_20260730.md`。

当前 WL 在理论上确实不能区分所有非同构图，例如 triangular prism 与 `K3,3`；但在 IMDB-BINARY、IMDB-MULTI、REDDIT-BINARY 实际采到的 patches 中，WL→exact unrooted canonical 没有观察到 type collision。Reddit 上观察到了同一个 unrooted patch 对应多个 center roles，但 rooted canonical 也没有通过门槛。

三种表示下的 `stats + content` 主结果为：

| 设置 | 表示 | raw | PCA | clustered real | unconstrained | final projected | iterative projected |
|---|---|---:|---:|---:|---:|---:|---:|
| IMDB-BINARY | WL | **0.7668** | 0.7549 | 0.7544 | 0.7479 | 0.7536 | 0.7480 |
|  | canonical | 0.7440 | 0.7552 | **0.7579** | 0.7478 | 0.7517 | 0.7415 |
|  | rooted | **0.7502** | 0.7495 | 0.7403 | 0.7375 | 0.7431 | 0.7265 |
| IMDB-MULTI | WL | 0.5463 | 0.5172 | 0.5538 | 0.5400 | 0.5410 | **0.5673** |
|  | canonical | 0.4978 | 0.4768 | 0.4894 | 0.5091 | **0.5485** | 0.4918 |
|  | rooted | **0.5128** | 0.5119 | 0.4853 | 0.5044 | 0.5095 | 0.5106 |
| REDDIT-BINARY | WL | 0.8202 | 0.8325 | 0.8252 | 0.8328 | 0.8318 | **0.8333** |
|  | canonical | 0.8160 | 0.8333 | 0.8375 | 0.8378 | **0.8418** | 0.8335 |
|  | rooted | 0.8115 | 0.8362 | 0.8362 | 0.8402 | 0.8327 | **0.8417** |

canonical final-projected 在 IMDB-MULTI 对自身三个 baseline 达到 3/3 seed 正向，并在 Reddit 相对 WL 对应方法提高 +0.0100；这说明表示几何会影响 real prototype selection，不能说 non-WL 完全无作用。但它没有在第二个数据集同时稳定超过自身 raw/PCA/real-patch，rooted 也没有通过任何数据集，且 Reddit 的最佳字典 content 仍低于 relation-graph-only=0.8550。

因此 no-go 得到进一步加强：

> 失败不只是自由 atom 不合法，也不能主要归因于 WL type collision 或 patch center 丢失。exact adjacency 保留完整结构后仍没有跨数据集的 KSVD-specific task gain；主要矛盾更可能是欧氏重构几何与任务相关结构相似性不对齐。

“增加多维排序依据”不再进入实验队列。它只是 canonical labeling 的不严格近似，仍会受到 tie/node-id 的置换问题；若后续继续研究，应改成合法离散候选上的结构 metric/selector 与 exact incidence，而不是继续为邻接向量设计排序。

## 6. 对更大科研 idea 的影响

更大的 idea 仍然成立，但需要改写为：

> 学习 dataset-level shared、legal、decodable、cross-environment stable 的结构 prototype，并恢复其 exact occurrence/incidence；结构关系的收益必须优于 matched content 与 shuffled/removed-relation controls。

这与 CIN 的区别可以很清楚：CIN 使用预定义 cell/cycle lifting；我们的目标是**学习哪些合法高阶结构应成为 cell，以及它们如何跨图复用**。创新点不应来自模仿 CIN 的消息传递外形，而应来自：

1. vocabulary 是数据自适应学习的，而不是枚举全部环；
2. prototype 本身合法、可回指、可解码；
3. occurrence/incidence 是精确的，而不是只有 graph-level soft code；
4. vocabulary 在结构 OOD/environment shift 下可匹配、可复用。

Projected-KSVD 已未通过最终门槛，因此后续不再以 KSVD 为 learned vocabulary headline。若进入 MolHIV/CIN-style 路线，应从“合法候选结构的学习式选择 + exact occurrence/incidence + OOD stability”出发，并把 KSVD 仅作为 baseline/初始化器。
