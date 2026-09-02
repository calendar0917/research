# KSVD 邻接坐标语义与 Phase 2/3 回顾

> 日期：2026-08-01  
> 目的：回答“统一排序能否让邻接矩阵和字典原子有意义”，并回顾 `~/life/code/keyan` 中 Phase 2、Phase 3 的真实思路。

## 1. 结论先行

全局统一采用一种排序规则，确实能给邻接矩阵建立一个**统一的操作性坐标系**。例如：

```text
slot 0 = root/center
slot 1 = 排名最高的候选节点
slot 2 = 排名第二的候选节点
...
coordinate (i,j) = slot i 与 slot j 是否相连
```

在这个定义下，KSVD atom 可以被解释为：

> root 与若干“结构排名槽位”之间、以及这些槽位彼此之间的典型连续连接模式。

但这只保证了**rank-slot meaning（排序槽位语义）**，并不自动保证：

1. 同构图在任意重编号后一定选择同一批抽象节点；
2. 度数相同的节点一定进入相同槽位；
3. cutoff 附近的等价节点不会因为 node ID 改变而被替换；
4. 图只改变一条边时，整个排序不会大幅翻转；
5. atom 等于一个内在、可命名、与排序无关的 graph motif。

因此准确说法是：

> **统一排序可以让 atom 在一个统一坐标系内有条件地、统计地有意义；它不能单独证明 atom 是置换不变的内在图原子。**

## 2. “邻接矩阵有意义”实际包含四个不同问题

### 2.1 坐标定义是否一致

同一个 `(i,j)` 在所有 patch 中是否表达同一类角色关系？

- WALK first-discovery order：表达“第 `i` 个被随机游走首次发现的节点”。
- center + degree order：表达“第 `i` 个按度数排名的邻居/候选节点”。
- rooted canonical order：表达“rooted isomorphism canonical labeling 给出的第 `i` 个节点”。

只要全流程统一，三者都有坐标定义；区别在于定义是否稳定、是否严格不变，以及是否适合线性建模。

### 2.2 节点选择是否不变

排序之前，必须先决定固定大小 patch 里放哪些节点。若候选节点多于容量，cutoff 处有多个结构等价节点，则 node ID tie-break 会导致重编号后选中不同抽象节点。

当前 IMDB capped n-hop 审计中，最强 local-signature selector 仍有：

- cutoff tie：`0.7404`；
- selected abstract node-set match：`0.3440`；
- ranked adjacency match：`0.9903`；
- rooted canonical output match：`0.9952`。

这说明“输出向量大多相同”和“实际选中了同一批节点”不是一回事。IMDB 的 clique-like 邻域中，很多候选节点彼此等价；换掉其中一个节点后，邻接向量可能仍一样，但不能声称 selector 本身唯一。

### 2.3 严格不变是否等于适合 KSVD

不等于。exact canonicalization 可以解决离散身份问题，却可能产生排序不连续：只改一条边，canonical label 可能整体换位，使向量在欧氏空间中跳很远。

KSVD 需要的不只是“同一对象得到同一编号”，还希望：

> 结构相近的 patch 在向量空间里也相近。

因此必须同时审计：

- relabel invariance；
- one-edge perturbation continuity；
- graph-distance 与 vector-distance 的对应；
- learned basis 的跨 fold 稳定性。

### 2.4 字典是否跨图共享

即使 patch 坐标统一，如果每张图单独学习一套字典，graph A 的 atom 3 与 graph B 的 atom 3 也未必是同一个结构方向。

要讨论“数据集级原子语义”，至少需要：

- 在训练集上学习一个 shared/global dictionary；或
- 对 per-graph dictionaries 做显式 atom matching/alignment。

当前从零路线采用 outer-train-fold global dictionary，因此比 Phase 3 的典型 per-graph dictionary 更适合讨论跨图 atom identity。

## 3. Phase 2 当时的真实思路

主要文件：

- `~/life/code/keyan/phase2_structure_classification/plan.md`
- `~/life/code/keyan/phase2_structure_classification/src/features.py`

### 3.1 方法定义

Phase 2 的主流程是：

```text
每个节点的 1-hop ego
→ center 放 slot 0
→ 邻居按 degree descending 排序
→ 截断/zero-pad 到 s×s
→ 全训练集 shared dictionary
→ 每个节点 sparse code
→ 每个 atom 的 mean/std/min/max/median
→ 图级 MLP 或 DeepSets
```

初始设置为 `s=7, K=20, T=3`，后续重点扩大到 `s=15/25, K=50, T=5`。

### 3.2 Phase 2 最重要的发现不是“排序解决了置换”

它最重要的发现是把 KSVD 前后的损失拆开：

1. **Truncation loss**：固定 `s` 之前已经丢失了多少完整 ego 信息；
2. **Reconstruction error**：KSVD 对已经截断后的信号又丢了多少信息。

IMDB-BINARY 的记录为：

| s | truncation loss | nonzero-region MSE | accuracy |
|---:|---:|---:|---:|
| 7 | 64.0% | 0.000142 | 68.7% |
| 15 | 19.8% | 0.002626 | 72.3% |
| 25 | 1.9% | 0.005233 | 73.9% |

Phase 2 因此得出一个仍然有效的原则：

> 完美重建一个严重残缺的 patch，不如有损地重建一个更完整的 patch。

这个原则对当前路线很重要：不能只报告 FINAL 相对当前 7-node patch 的 reconstruction improvement，还要审计 patch 形成之前的 selection/coverage loss。

### 3.3 Phase 2 没有解决的事情

Phase 2 的 degree sorting 是工程上的统一坐标系，不是严格 permutation-invariance 证明：

- degree ties 没有结构化消歧；
- `argsort` 的 tie 顺序会继承输入/node-index 顺序；
- 截断边界若有同度节点，选中集合也会随编号变化；
- 实现把 diagonal/self-loop 也放进了矩阵，padding 与真实低度 ego 需要额外区分。

因此 Phase 2 atom 最准确的解释是“center/degree-rank slot patterns”，而不是无条件的 graph motif。

## 4. Phase 3 当时的真实思路

主要文件：

- `~/life/code/keyan/phase3_egosvd_xgboost/plan.md`
- `~/life/code/keyan/phase3_egosvd_xgboost/phase3_progress_2026-05-16.md`
- `~/life/code/keyan/phase3_egosvd_xgboost/src/ego_sampler.py`
- `~/life/code/keyan/phase3_egosvd_xgboost/src/ksvd_features.py`

### 4.1 方法定义

Phase 3 的常见流程是：

```text
每个节点一个 ego patch
→ degree-ranked padded adjacency
→ KSVD
→ dictionary/code 的 per-atom statistics 与 Gram features
→ 加入 macro topology 和 node-feature summaries
→ XGBoost
```

patch size 使用：

```text
subnet_size = ceil(avg_degree × 1.8)
并限制在 [4, n_nodes]
```

显式比较过的排序包括：

- `ego_degree_desc`
- `center_first_degree_desc`
- `sample_new_like_sorted`

代码最终都以 degree descending 为主，并以 node ID 为最终 tie-break。

### 4.2 Phase 3 对置换问题的态度

Phase 3 并没有认为“统一排序已经证明不变性”。其 plan 明确写到：

> flattened adjacency 对节点排序高度敏感，当前坐标系可能与导师实现不同。

因此当时的策略是把 sampler、ordering、center placement、KSVD initialization、OMP mode 和 evaluation protocol 都参数化，执行“先对齐，再诊断”。

IMDB quick test 中两种 ordering 都得到 `77.60%`；进展报告把排序差异总结为 `<1pp`，说明它不是当时与导师分数差距的主因。但这只是 downstream sensitivity 结果，不是严格的不变性证明。

### 4.3 Phase 3 的更大问题：很多实验使用 per-graph dictionary

Phase 3 的典型 pipeline 是每张图单独学习 `K=8` 字典，再汇聚：

- dictionary atom 的 mean/std/max；
- sparse-code atom 的 mean/std/max；
- sparse-code Gram matrix；
- macro/node summaries。

因此它更像“用每张图内部 sparse factorization 的统计量描述该图”，而不是学习一个数据集共享 motif vocabulary。没有 atom alignment 时，不应把不同图中的 atom `k` 当作共同语义单元。

Phase 3 后期也加入了 global dictionary，并在 ENZYMES 上发现 global dictionary 优于 per-graph dictionary；但主要历史结果仍需按具体配置分别解释。

### 4.4 Phase 3 真正定位到的主混淆

当时最大的表面分数差异不是 ordering，而是 evaluation protocol：

- Protocol A 在 test fold 上 early stopping，存在信息泄漏；
- Protocol B 使用诚实的 repeated CV。

同一特征在 IMDB-BINARY 上从约 `78.0%` 降至 `73.6% ± 4.2%`。因此 Phase 3 的 benchmark 数字不能直接作为“atom 语义已被验证”的证据。

## 5. Phase 2、Phase 3 与当前路线的关系

| 方面 | Phase 2 | Phase 3 | 当前从零路线 |
|---|---|---|---|
| patch | fixed/expanded 1-hop ego | dynamic ego | fixed 7-node WALK |
| ordering | center + degree-desc | degree-desc variants | first-discovery order |
| padding | 有 | 有 | 无 |
| 字典 | shared/global | 典型为 per-graph，后有 global | outer-train-fold global |
| 主要关注 | classification、coverage | 复现对齐、graph statistics | falsification、INIT→FINAL 归因 |
| 置换保证 | 未证明 | 识别为风险但未证明 | 显式 relabel/canonical audit |
| atom 含义 | degree-rank slots | 多为图内 factor statistics | walk-position continuous basis |

三条路线可以拼成一个更完整的认识：

1. Phase 2 提醒我们先审计 **selection/coverage loss**；
2. Phase 3 提醒我们 ordering、sampler、dictionary scope、evaluation protocol 必须分别对齐；
3. 当前路线进一步要求区分 **INIT 与 FINAL**，不能把初始化器、宏观统计或测试泄漏的作用算给 KSVD updates。

## 6. 下一步建议：先做 matched coordinate-semantics audit

不建议现在直接把 WALK 全部换成 degree sorting 后重跑分类。那会同时改变 sampler output distribution、coordinate system 和 dictionary，无法判断变化来自哪里。

下一步只固定同一批 7-node WALK-selected node sets，然后重排同一 induced subgraph：

1. `walk_order`：当前 first-discovery order；
2. `center_degree_order`：center first，其余按 patch 内 degree descending，node ID 仅作为被审计的 tie-break；
3. `center_signature_order`：center first，按 distance/degree/triangle/WL-like local signature；
4. `rooted_canonical_order`：对固定 node set 做 exact rooted canonicalization。

先不做 downstream model selection，只报告：

- relabel vector match；
- tie frequency 与 rank-flip rate；
- one-edge perturbation 后的 vector discontinuity；
- vector distance 与 rooted GED 的关系；
- INIT→FINAL held-out reconstruction；
- effective atom count/coherence；
- 跨 fold atom matching；
- 每个 atom 的 top-activating patch 是否形成稳定 topology distribution。

判定方式：

- 若 rank/signature ordering 在不损害连续性的前提下明显提高 atom alignment，可把方法定位为 **rank-slot KSVD**；
- 若 canonical ordering 不变性最好但 perturbation geometry 明显恶化，不应仅因“严格唯一”而采用；
- 若所有 adjacency ordering 都无法同时满足 invariance 与 continuity，应承认 raw flattened adjacency 不是合适的 KSVD signal，转向 permutation-invariant patch descriptors；
- 若要严格处理 cutoff ties，应考虑完整 orbit/set aggregation，但这将不再是 vanilla fixed-vector adjacency KSVD。

## 7. 当前最稳妥的方法定位

在完成上述审计前，当前 atom 最准确的名称不是“graph motif atom”，而是：

> **sampler-relative ordered-adjacency sparse basis atom**。

若改为统一 center/signature 排序并通过审计，可以升级为：

> **rooted rank-slot structural basis atom**。

只有当 selection、ordering、dictionary alignment 和 perturbation geometry 都得到更强保证后，才适合讨论“内在图原子”或“motif vocabulary”。

## 8. 导师脚本的真实方法定位

仓库中被作为导师原始管线参考的主要文件是：

- `~/life/code/keyan/phase3_egosvd_xgboost/MUTAG_cls.py`
- `~/life/code/keyan/phase3_egosvd_xgboost/temp.py`

其中 `MUTAG_cls.py` 依赖仓库外的：

```python
PreLibs.spComponents.sparseRepresentation.ksvd.KSVD
PreLibs.spComponents.sparseRepresentation.sample.sample_new
```

`temp.py` 保存了 `sample_new` 和 `KSVD` 的实现片段，但 `transformNetwork()` 缺失。因此当前可以恢复导师方法的总体数据流和绝大多数数值细节，但不能声称已经知道导师原始的精确节点排序。`src/sampler_supervisor.py` 中的 center-first degree-desc 是后来的代理实现，不是原始函数本身。

### 8.1 导师不是在学习数据集共享的 motif dictionary

导师脚本对每张图单独执行：

\[
Y_G \approx D_G X_G,
\]

其中：

- `Y_G`：图 `G` 中每个节点对应的 ego adjacency vectors；
- `D_G`：只属于图 `G` 的字典；
- `X_G`：只属于图 `G` 的 sparse codes。

随后不保留完整 `D_G` 和 `X_G`，而是压缩为固定维度统计量：

```text
D_G: 每个 atom 的 mean/std/max
X_G: 每个 atom 的 abs-mean/std/abs-max
X_G X_G^T: atom co-activation Gram
```

因此导师思路最准确的定位是：

> **用 per-graph sparse factorization 描述一张图内部的局部结构分布，再把 factorization statistics 当作图级特征。**

它不是：

> 在整个数据集上学习一套共享的、可跨图命名的 motif atoms。

### 8.2 为什么导师可以使用动态 patch size

导师脚本对每张图计算：

```text
subnet_size = ceil(avg_degree × 1.8)
clamp 到 [4, n_nodes]
```

不同图的 adjacency dimension 可以不同，因为每张图单独学习 `D_G`；最后只输出 `mean/std/max/Gram`，图级特征维度仍固定。

这是一个重要取舍：

| 设计 | 优点 | 代价 |
|---|---|---|
| per-graph dictionary + dynamic `s_G` | 覆盖更灵活，不要求全数据集统一 adjacency dimension | 没有天然的跨图 atom identity |
| shared dictionary + fixed `s` | atom index 跨图共享，可形成 vocabulary | 必须处理截断、padding 和统一坐标 |

Phase 2 和当前路线更接近第二种；导师脚本属于第一种。

### 8.3 导师的完整分类器不是纯 KSVD

`MUTAG_cls.py` 的图级向量包含三条支线：

1. KSVD micro/Gram features；
2. `density/max_degree/gini/max_k_core/triangle_density`；
3. MUTAG 的 7 维 atom-type concentration。

最后使用 XGBoost + Optuna。因此导师脚本的分数表示的是：

> sparse factorization statistics + 宏观拓扑 + 化学语义的融合分类性能。

它不能单独证明 KSVD atoms 有增量，也不能证明纯结构 KSVD 达到了相同分数。

### 8.4 排序在导师方法中的作用弱于当前路线

导师只使用每个 atom 的 `mean/std/max`，这些统计量本身会丢弃 atom 中具体哪个 adjacency coordinate 取了什么值。也就是说，它更关注：

- atom 整体密度/波动；
- atom 使用强度；
- atoms 的共激活关系；

而不是 atom matrix 的具体可视化结构。

排序仍会影响 KSVD 的样本几何，但导师方法并不要求每个 atom 的 `(i,j)` 都具有可直接命名的跨图语义。这也是 Phase 3 中排序变化通常小于 1pp 的一个合理解释。

### 8.5 导师脚本中需要谨慎解释的实现点

1. `transformNetwork()` 缺失，精确 ordering 未知；
2. padding 使用节点编号 `0`，而 PyG 图中 `0` 通常是真实节点，按保存片段存在 padding collision 风险；
3. OMP 没有显式给出固定稀疏度，Phase 3 后来的环境审计发现它可能退化为近似 1-sparse；
4. per-graph atoms 没有显式跨图 matching；SVD 初始化和 `transformPos` 只能提供弱顺序/符号锚定；
5. test fold 被用作 XGBoost early-stopping eval set，而且 Optuna 在同一组 folds 上选择并报告结果，存在评估偏高；
6. feature importance 在全部数据上重新拟合，只能描述相关性，不能作为 atom 因果解释。

## 9. 四个阶段最简单的心智模型

```text
导师脚本：
    KSVD = 每张图自己的“局部结构谱/摘要器”

Phase 2：
    KSVD = 全数据集共享的“结构词典学习器”

Phase 3：
    把导师的摘要器工程化、模块化，并诊断复现差距

当前路线：
    回到共享词典目标，但要求用 reconstruction、INIT→FINAL、controls
    和 atom stability 逐层证明它真的成立
```

因此当前路线和导师脚本不是同一个科学问题：

- 导师问：这种 factorization statistics 能不能帮助图分类？
- 当前路线问：KSVD 能不能从图 patch 中自发现一套共享、稳定、可泛化并最终对任务有增量的结构 basis？
