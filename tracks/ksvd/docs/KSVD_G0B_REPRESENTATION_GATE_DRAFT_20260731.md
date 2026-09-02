# KSVD G0B 草案：先过 representation gate，再加 within-motif variation

> 日期：2026-07-31  
> 状态：**设计草案，尚未冻结、尚未运行**  
> 前置结果：G0 primary 为 `PASS_INITIALIZER_DISCOVERY_ONLY`；fixed-random 单次条件从 INIT 0/10 提升到 FINAL 10/10，最晚在 iteration 10 严格恢复。

## 1. 为什么不能直接“随机加边然后跑 KSVD”

G0 只有四种 exact canonical columns。下一步当然应增加同一 motif 内部的结构变化，但 exact canonical adjacency 存在一个必须先隔离的问题：

> 对两个非同构的 noisy variants，canonical labeling 虽然各自置换不变，却不保证隐藏 motif core 被放到相同的 15 个坐标上。

因此，若直接加入 nuisance edges 后 KSVD atom 无法 decode，至少有两种完全不同的原因：

1. KSVD 没有从变体中提炼 motif；
2. canonical adjacency representation 本身没有提供跨变体一致的线性坐标。

这正对应 `luyin11.txt` 中“邻接矩阵置换不变性”和“重建误差不能代表图表征”的问题。G0B 必须先做 representation gate，不能把两类失败混在一起。

## 2. G0B-R：representation-only gate

本阶段不运行 KSVD。

### 2.1 数据

沿用 G0 四种 6-node motif。每个 patch 在隐藏 motif core 外加入受控 nuisance：

- R0：无 nuisance，作为回归控制；
- R1：恰加 1 条非 core edge；
- R2：独立 edge flip probability 0.05；
- 暂不混入 patch sampling error。

对每个 variant，canonicalizer 除返回 15 维 vector，还必须返回产生 lexicographic minimum 的确定性 permutation，并把生成器中的 hidden core edges 映射到 canonical coordinates。

### 2.2 三个必要检查

#### A. Label identifiability

检查是否存在同一个 canonical patch vector 由不同 hidden motif families 生成。

报告：

- cross-family collision count/rate；
- `P(motif | canonical vector)` 的最大后验准确率，即 representation-level Bayes upper bound。

若不同 hidden motifs 在观测上完全相同，则任何无标签 learner 都不可能恢复生成器赋予的语义。

#### B. Core-coordinate consistency

对每个 motif family 和每个 canonical edge coordinate，计算：

```text
该坐标在多少 variants 中对应 hidden core edge
```

报告：

- per-coordinate core occupancy；
- 最佳固定 support 对 hidden core 的 mean/min F1；
- 是否存在一个无需知道每个 variant 真值、但跨 variants 稳定的固定 motif support。

若 core edge 在 canonical coordinates 中大幅漂移，则“一个 KSVD atom = 一个固定 motif edge mask”的命题在该表示下先天不成立。

#### C. Prototype separability

不学习字典，只计算每个 motif family 的 canonical-vector centroid/medoid：

- within-family cosine/distance；
- between-family cosine/distance；
- nearest-centroid/nearest-medoid oracle classification；
- family silhouette-like separation。

这是 dictionary learning 前的可分性上限，不是 downstream 任务结果。

### 2.3 R gate

只有同时满足以下条件才进入 G0B-K：

- cross-family collision rate `<= 1%`；
- representation Bayes accuracy `>= 0.95`；
- 每种 motif 的最佳固定 core support mean F1 `>= 0.90`；
- nearest-medoid family accuracy `>= 0.90`。

失败时应先改变 representation，而不是调 KSVD/restart。候选改变包括：

- rooted / typed canonicalization；
- 节点或边属性辅助 canonicalization；
- graph-invariant statistics；
- 不把 atom 解释为固定 adjacency edge mask，改成 latent structural basis。

## 3. G0B-K：有变体时的单次 dictionary discovery

仅在 G0B-R 通过后运行。

### 3.1 第一版冻结建议

- 先只用 R1：每个 patch 恰有一个 nuisance edge；
- train/test graphs 与 G0 相同：100/30，每图 12 cells；
- 10 data seeds；
- K=4，T=1，25 iterations；
- primary deterministic maximin 单次；
- secondary fixed random-column seed 0 单次；
- INIT `n_iter=0` 与 FINAL `n_iter=25`；
- 不增加 restart，不按真值选择模型。

K=4、T=1 的问题是：KSVD 是否把每个 motif family 的一组 variants 压缩成一个稳定 atom。只有之后才考虑 `core atoms + nuisance atoms` 的 K>4、T=2 分解。

### 3.2 评价不再假设 exact true dictionary

由于 noisy variants 不再严格满足给定的 `Y=D*X`，评价应改成：

- atom 与 representation gate 得到的 motif centroid/medoid/core support 的匹配；
- atom relative-threshold decode 的 core precision/recall/F1；
- code 对 hidden motif occurrence 的 macro F1/accuracy；
- train/test reconstruction 仅为辅助；
- INIT→FINAL 增益；
- 跨 datasets 的单次 atom stability。

### 3.3 关键分支

1. **INIT 差、FINAL 好**：出现比 G0 更可信的 KSVD motif-family refinement；
2. **INIT 已好**：maximin/medoid discovery 仍是主要机制，KSVD 必要性存疑；
3. **occurrence 好、atom core decode 差**：可称 latent family basis，不称显式 motif atom；
4. **representation gate 好但 FINAL 差**：才是 dictionary objective/optimization 的直接负证据；
5. **representation gate 先失败**：停止 KSVD 调参，优先解决表示。

## 4. 当前最重要的结论

下一步不应是“把 synthetic 变复杂一点”这么笼统，而应拆成：

```text
G0B-R：canonical representation 是否允许跨 variant 的固定 motif 语义？
    ↓ 通过后
G0B-K：单次 KSVD 是否能从 motif family 中提炼稳定 atoms？
```

这样才能知道失败发生在置换不变表示，还是发生在 KSVD 本身。
