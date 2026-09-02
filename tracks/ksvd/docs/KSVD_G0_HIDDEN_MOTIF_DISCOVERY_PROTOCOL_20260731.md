# KSVD G0：完整大图中的隐藏 motif discovery 协议

> 日期：2026-07-31  
> 状态：**冻结 v0**  
> 定位：E0/E1 只是 implementation / assumption audit；G0 才开始检查“learner 未获得人工字典时，能否从完整图产生的无标签 patch 中得到可解释结构原子”。

## 1. 本轮唯一问题

> 当完整大图由若干隐藏局部 motif cell 组成，而 learner 只获得从这些图中提取、经过置换不变 canonicalization 的无标签 patch 向量时，一次可部署初始化加普通 KSVD 能否恢复 motif vocabulary 及每个 motif 的出现位置？

这里必须区分三个命题：

1. **patch 暴露条件成立**：提取器确实给出了包含完整 motif 的 patch；
2. **initializer discovery**：初始化本身已经找到了 motif prototypes；
3. **KSVD refinement/discovery**：初始化尚未恢复 vocabulary，KSVD 更新后才恢复。

G0 不允许把 2 冒充为 3。

## 2. G0 回答与不回答的边界

G0 比 E1 前进一步：

- atom 不再由 learner 人工指定；
- 数据不再直接按给定的 `Y=D*X` 矩阵接口生成；
- 先生成完整大图，再从图中提取 patch；
- 完整图节点编号被全局随机置换；
- learner 不获得 motif label、cell 位置或真实字典。

但 G0 仍是受控 discovery sanity check，而不是真实大图路线的最终证明，因为：

- patch node set 由 oracle cell boundary 给出；
- 每个 patch 第一版只含一个无噪声 motif；
- exact canonicalization 可以在 6 节点上穷举 `6! = 720` 个排列；
- 暂不处理随机游走是否能找到这些 cell，也不处理 patch 间关联。

因此，G0 成功只说明：

> **在 motif 已被 patch extractor 完整暴露后，dictionary pipeline 能否无标签地形成可解释 vocabulary。**

它不能说明随机游走、真实大图采样或 downstream classification 已经可行。

## 3. 完整图生成

每张图包含：

- 12 个 cell；
- 每个 cell 6 个节点；
- 总节点数 72；
- 4 种隐藏 motif：triangle、4-cycle、3-star、5-path；
- motif 概率固定为 `0.35 / 0.25 / 0.25 / 0.15`；
- train 100 张图，test 30 张图；
- train/test 各自强制至少出现一次每种 motif。

cell 内部仅放置对应 motif 边。随后使用只跨 cell 的 bridge backbone 连接全部 72 个节点，使完整图连通，同时保证每个 cell 的 induced subgraph 不被 bridge 改写。最后对整张图的 72 个节点做一次全局随机置换。

生成器保存隐藏 cell node sets 和 motif IDs 供评估使用，但 learner 接口只接收 patch matrix。

## 4. Oracle patch extraction 与 exact canonicalization

第一版故意使用 oracle cell node set 提取 6-node induced patch，以隔离 dictionary discovery 本体。

对每个 patch：

1. 按当前全局节点编号取得 induced adjacency；
2. 穷举 720 个节点排列；
3. 对每个排列取邻接上三角 15 维 binary vector；
4. 选择 lexicographically smallest vector 作为 canonical representation。

必须通过 permutation-invariance 自测。固定语义槽位和 motif-specific 排序均禁止。

## 5. 现实初始化预算

每个 data seed 只运行以下两个预先冻结的单次条件，不做 model selection：

### 5.1 Primary：deterministic maximin

- 从真实训练 patch columns 中选原子；
- 首先选择最大范数列，tie 用列向量 lexicographic order；
- 后续选择与已选集合最大绝对 cosine 最小的列；
- 所有 tie 均确定性处理；
- 只运行一次。

### 5.2 Secondary：fixed random-column baseline

- 固定 initialization seed `0`；
- 从训练 patch columns 无放回选 4 列；
- 只运行一次。

**禁止：**增加 restart、查看 test/真值后选 seed、按恢复指标选择候选。

## 6. 最关键的 attribution control

每种初始化都必须从同一初始字典运行两次：

- `INIT`：`n_iter=0`，只做 OMP 评估；
- `FINAL`：`n_iter=25`，执行 KSVD 更新。

解释规则：

1. INIT 差、FINAL 好：支持 KSVD refinement/discovery；
2. INIT 已好、FINAL 也好：discovery 主要来自 initializer，不能归功于 KSVD；
3. reconstruction/code 好但 atom 无法 decode：只能称 latent structural basis，不能称 motif atom；
4. 单次初始化跨 data seeds 不稳定：属于 operational robustness 失败，不能用更多 restart 掩盖。

## 7. 冻结配置

- data seeds：`20260731..20260740`；
- train/test graphs：100/30；
- cells per graph：12；
- patch size：6，vector dimension 15；
- `K=4`；
- `T=1`，`T_min=1`；
- FINAL iterations：25；
- KSVD internal seed：0，仅用于确定性的 dead-atom handling；
- 无内部 nuisance edge、无 edge flip、无 motif deformation；
- train/test 完全分离。

## 8. 指标

### 8.1 Atom-to-motif

评估时才将 learned atoms 与四个隐藏 motif 做最优 permutation/sign matching：

- matched mean/minimum absolute cosine；
- label-free relative atom decode：`abs(w_e) >= alpha * max(abs(atom))`；
- `alpha = 0.5` 为主阈值，同时记录 `0.3 / 0.5 / 0.7`；
- decoded edge precision/recall/F1；
- exact decoded motif count；
- decoded 非孤立部分是否连通。

不得用真实 motif edge 数做 top-k decode。

### 8.2 Code-to-occurrence

matching 后比较每个 patch 的 one-hot hidden occurrence：

- support precision/recall/F1；
- 每种 motif occurrence precision/recall/F1；
- macro occurrence F1；
- exact occurrence assignment accuracy。

motif label 只用于冻结评估，不进入初始化、KSVD 或 sparse coding。

### 8.3 Reconstruction

辅助记录：

- train/test relative reconstruction；
- binary edge precision/recall/F1；
- exact patch reconstruction。

这些指标不能替代 atom 和 occurrence recovery。

### 8.4 跨数据稳定性

对 10 个独立 data seeds 的单次 learned dictionaries 做 pairwise optimal matching，报告 mean/minimum matched atom cosine。这是 operational stability；同一数据上大量碰随机 seed 不算稳定性。

## 9. G0 判定

对 primary deterministic condition，FINAL discovery gate 为：

- 10/10 data controls 通过；
- mean atom cosine 的跨 seed 最小值 `>= 0.99`；
- primary decode threshold 0.5 下，每个 seed 4/4 exact motifs；
- occurrence macro F1 的跨 seed最小值 `>= 0.99`；
- exact occurrence accuracy 的跨 seed最小值 `>= 0.99`；
- pairwise dictionary stability minimum `>= 0.99`。

若 FINAL 通过，再根据 INIT 判定贡献：

- INIT 不通过、FINAL 通过：`PASS_KSVD_REFINEMENT`；
- INIT 也通过：`PASS_INITIALIZER_DISCOVERY_ONLY`；
- FINAL 不通过：`FAIL_SINGLE_RUN_DISCOVERY`；
- 数据、canonicalization 或 oracle control 失败：`FAIL_DATA_OR_EVALUATOR`。

random-column condition 是预注册的弱初始化诊断，不用于替代 primary，也不允许选择较好者形成 ensemble。

## 10. 下一层只由结果决定

最可能的 G0 结果是 deterministic maximin 在 `n_iter=0` 已找齐四种完全重复 prototype。若如此，G0 只证明 pipeline 正确，下一层 G0B 必须加入 **within-motif variation / nuisance edges**，让训练集中不再只有四种唯一列，才能真正检验 KSVD 是否从一族变体中提炼稳定 motif。

若 random INIT 差而 random FINAL 好，可作为 KSVD 有 refinement 能力的附加证据；但 primary attribution 仍按冻结规则报告。
