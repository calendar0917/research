# KSVD 从零路线：IMDB-BINARY R1-A label-free basis characterization

> 日期：2026-08-01  
> 状态：**结果不可见前冻结**  
> 前置：R0-P/R0-D 通过；R0-A/R0-B/R0-C task utility 失败，普通无监督 task branch 已停止。  
> 本轮角色：不再问 classification added value；只刻画 KSVD 作为 sparse graph-patch compressor / basis discovery 的质量。

## 1. 研究问题

在 raw IMDB-BINARY 上，单初始化、无人工 atom vocabulary 的 FINAL dictionary 是否表现为：

1. 比 INIT 更跨 fold 稳定的真实 patch basis；
2. 比 fixed Gaussian 更接近真实 held-out patches；
3. 每个 atom 都有可定位的 top-activating real patches，而不是抽象噪声方向；
4. atom usage 覆盖不同 local density regimes，而不是单一 clique 模式。

本轮**不使用 labels 作为 gate**。任何 label composition 只作 descriptive diagnostics。

## 2. 数据与 freeze

直接复用 R0-D 的 registered artifacts，不重新训练字典：

```text
raw IMDB-BINARY
views = stratified + exact-isomorphism-grouped
outer split seed = 731301
patch sampling seed = 20260731
patch size = 7
patch dimension = 21
patches/graph = min(n,24)
K = 12
T = 2
T_min = 1
updates = 25
INIT = deterministic maximin
restarts = 0
centering = outer-train coordinate mean
```

输入：

- `tracks/ksvd/results/from_scratch/imdb_binary_r0d_dictionary_audit_20260731.json`

对每个 outer fold 只重新编码 patches，不更新 dictionary、不扫描 K/T/iterations。

## 3. Per-atom characterizations

对 INIT 与 FINAL 分别在 held-out test patches 上编码后报告：

### 3.1 Usage

- activation frequency；
- activation share；
- graph coverage：至少激活一次的 test graph 比例；
- mean absolute coefficient among active patches。

### 3.2 Real-patch proximity

- 每个 atom 与全部 outer-train centered patches 的最大 absolute cosine；
- 对应 nearest real patch 的 edge count；
- 同指标的 fixed Gaussian control。

### 3.3 Top-activating patches

每个 atom 取 held-out 中 absolute coefficient 最大的 `top_k=5` patches，报告：

- mean absolute coefficient；
- edge-count mean/std；
- unique walk-vector count；
- unique rooted-canonical signature count；
- 主导 edge-count mass。

### 3.4 Continuous atom mass

在 centered atom 上报告：

- L1 mass；
- positive-coordinate mass；
- maximum absolute coordinate。

不把 continuous atom 阈值化后强行命名为 motif。

## 4. Cross-fold stability

复用 R0-D 的 exact maximum-cosine assignment：

- INIT matched atom cosine mean/min；
- FINAL matched atom cosine mean/min；
- FINAL−INIT stability gain。

额外：对每个 fold 的 FINAL atoms，报告与同 fold INIT 的 matched cosine，衡量 KSVD update 是否把 atoms 推离初始化但仍保持跨 fold 可对齐。

## 5. Controls

1. **Gaussian proximity control**：同一 centering 下的 fixed Gaussian dictionary nearest real-patch cosine。  
2. **INIT proximity/stability control**：同一 fold 的 INIT 作为 pre-update baseline。  
3. **Label-free requirement**：主 gate 不看任何 BA、label composition 或 classifier。  
4. **No restart selection**：只分析 R0-D 已保存的唯一 deterministic dictionaries。

## 6. Registered characterization gates

对每个 view 必须同时满足：

1. FINAL test non-dead atoms = 12/12 在全部 5 folds；
2. FINAL cross-fold matched atom cosine mean `>= 0.70`；
3. FINAL cross-fold matched cosine mean 严格高于 INIT；
4. FINAL mean nearest real-patch absolute cosine 高于 Gaussian，且 margin `>= 0.10`；
5. FINAL mean nearest real-patch absolute cosine 不低于 INIT；
6. 平均每个 atom 的 top-5 unique walk vectors `>= 2.0`，避免 trivial single-vector collapse；
7. 跨 atom 的 top-5 edge-count means 的标准差 `>= 1.0`，避免所有 atoms 只抓住同一 density regime。

两个 view 都通过：

> `PASS_R1A_BASIS_CHARACTERIZATION`

任一 view 失败：

> `FAIL_R1A_BASIS_CHARACTERIZATION`

## 7. 允许与不允许的解释

如果通过，可以说：

> 无人工 atom vocabulary 的 KSVD 在 raw IMDB 上发现了比 INIT 更跨 fold 稳定、比 Gaussian 更接近真实 patches、并覆盖多种 local density regimes 的 sparse basis。

仍不能说：

- atoms 都是可命名 motifs；
- basis 对分类有增量；
- 达到公开 benchmark SOTA；
- reconstruction 自动等价于 task discovery。

如果失败：

1. 不增加 restart 或扫描 K/T；
2. 不把 failure 改写成 classification 失败；
3. 只能在新协议中检查更强的 multi-view/attribute compressor 问题，或接受当前 basis 仅 reconstruction-valid 但结构刻画不足。

## 8. 与 task branch 的关系

R1-A 是 compressor/basis-discovery 分支的第一步。它不重新打开 R0-A/B/C 的 classification gate，也不用 label-conditioned objective。若未来进入 task-aware dictionary，必须另立命题，不能回写到 R1-A。

## 9. Execution record（协议冻结后填写）

完整运行已于 2026-08-01 执行。正式 classification 为
`FAIL_R1A_BASIS_CHARACTERIZATION`，但 failure 只来自第 6 节 condition 5。

执行后发现该 condition 结构性失配：deterministic maximin INIT 的每列本来就是一条
outer-train centered real patch 再归一化，因此 INIT nearest-real absolute cosine 按构造恒为
`1.0`；任何离开单条样本、形成连续 basis direction 的 FINAL atom 都不可能超过它。

因此：

1. 保留原 protocol 和 formal FAIL，不事后删 gate 或改成 PASS；
2. 不把 formal FAIL 解释为其余 basis evidence 全部失败；
3. R1-A 对 overall basis characterization 记为 **protocol-misspecified / inconclusive**；
4. 后续只允许无 gate 的 descriptive consensus basis atlas，不再从同一结果构造新 confirmatory gate。

结果：

- `tracks/ksvd/results/from_scratch/IMDB_BINARY_R1A_BASIS_CHARACTERIZATION_20260801.md`
- `tracks/ksvd/results/from_scratch/imdb_binary_r1a_basis_characterization_20260801.json`
