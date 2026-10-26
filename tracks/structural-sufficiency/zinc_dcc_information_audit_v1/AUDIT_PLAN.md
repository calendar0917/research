# ZINC DCC-L 结构信息与泛化瓶颈审计 v1 — 冻结计划

**Track:** `tracks/structural-sufficiency/zinc_dcc_information_audit_v1/`
**冻结时间（UTC）:** 2026-10-26T01:32:08Z
**仓库 HEAD:** `58e6837b91507aae3feb9ad9370ef75d1959b7ec`
**工作树状态:** 只有本 track 的新文件；另有一个本轮开始前已存在的未跟踪目录
`zinc_frozen_bond_representation_audit_v1/`（不是本 track 的产物，本轮不修改它）。
**性质:** 诊断审计，不是新架构搜索、超参数搜索、性能调参或模型修补。

> 本文件在任何 A1–A4 结果产生之前写定。所有依赖 `y` 的门禁、特征定义、
> 划分和评价标准以本文件 + `configs/audit_v1.json` 为准。后续如果需要修订，
> 必须另写 `AMENDMENT_*.md`，不得就地改写本文件。

---

## 0. 一句话问题

**DCC-L 的结构统计与图级读出是否遗漏、抹平或难以利用某些对 ZINC 泛化有价值的
联合结构关系？如果存在，它是信息表示限制、有限模型计算限制，还是有限样本
可学习性问题？**

必须分开的四件事：

- **H-A（代数不可辨识）:** 特定聚合/池化算子对“键状态与键位置的对应关系”
  是否在 activation level 上不可辨识。
- **H-B（非冗余性）:** 新增联合统计相对**完整 513 维** DCC-L reader 输入
  在现实分子分布上是否包含不可重建的独立变异。
- **H-C（预测增量）:** 新增联合统计是否给独立分子上的标签预测带来稳定增量。
- **H-D（难例解释）:** 上述现象是否解释低支持、未见局部环境或尾部难例的误差。

**A 不自动推出 B；B 不自动推出 C；C 不自动推出模型创新或 0.05x 可达性。**

---

## 1. 数据、checkpoint 与访问协议

### 1.1 数据

| 项 | 值 |
|---|---|
| 数据集 | official PyG `ZINC(subset=True)`，raw-y |
| 本轮允许 | official **train** 10,000 个结构；official valid 只在“不参与选择/门禁/审计头训练”的约束下**不加载**；official **test 完全不打开** |
| 结构 cache | `tracks/structural-sufficiency/zinc_arch_discovery_v1/cache/ad_train.pt` |
| cache sha256 | 运行前记录在 `analysis/provenance.json` |
| P3 objects | `tracks/structure-contribution/results/bjoc_full10k_pretraining_transfer_v1/objects_train10k.pt` |
| 标签 | 只用于 (i) Fit8k 的 DCC-L 监督训练，(ii) ProbeFit 的审计头拟合，(iii) AuditEval 在冻结预测后的一次性评分。**不用于任何结构统计、划分、门禁、特征选择或标准化** |
| 官方 valid | 不作为本轮评价输入；不参与 checkpoint 选择、审计头训练、门禁或最终主表 |
| 官方 test | **不构造、不加载、不索引、不读取 labels/predictions/cache**；代码中若出现 `split="test"` 必须抛异常 |

### 1.2 冻结 checkpoint（A1/A2/A3/A4 的模型侧对象）

优先使用至少两个 DCC-L seed：

| 名称 | 种子 | 来源 | 用途 |
|---|---:|---|---|
| `D0` | 0 | `zinc_dual_channel_composer_v1/runs/L_s0/best.pt`（本地） | A1/A2/A3/A4 主 |
| `D1` | 1 | `zinc_performance_search_v1/runs/B_R0_s1/best.pt`（远端 `res`，拉回后计算 SHA） | A1/A2/A3/A4 第二 seed |
| `D2` | 2 | `zinc_performance_search_v1/runs/B_R0_s2/best.pt`（远端，可选） | 只作稳健性 |
| `D3` | 3 | `zinc_performance_search_v1/runs/B_R0_s3/best.pt`（远端，可选） | 只作稳健性 |

- 每个 checkpoint 必须记录：文件 SHA256、`artifacts.json`/`search.json` 里的
  seed/epoch/参数数/初始 state hash、以及实际加载后的 `state_dict` 结构。
- 不允许用 official valid 重算 MAE 来“验证”checkpoint；只核对文件 hash、
  metadata 和参数契约。若必须用预测文件做一致性检查，只允许读取 train 行。
- `B-R0` 与 `L` 的 forward 契约相同：`DualChannel(organized_local=True)`，
  494,193 参数。`B-R0_s0` 就是历史 `L_s0`，不重复训练。

### 1.3 标准化

- 冻结 DCC-L checkpoint 内的 `global_mean/global_scale` 来自
  `prep_train10k.json`（full official-train 10k 的无标签结构统计）。这是冻结
  模型的一部分，**不是**本轮新训练；A1/A2/A3/A4 必须如实披露它覆盖了
  ProbeFit/AuditEval 的无标签结构。
- B 阶段新训练 DCC-L 时，标准化只能由 **Fit8k** 的结构行计算；不得使用
  ProbeFit/AuditEval/valid/test 的统计。

### 1.4 访问审计

- 所有 `torch.load`/文件读取通过一个记录 wrapper；任何路径包含 `ad_valid.pt`、
  `ad_test.pt`、`valid`、`test` 的 cache 访问都必须被阻断并记录。
- 输出 `analysis/label_access_audit.json`：每个阶段读取了哪些文件、是否含
  `labels`、是否读 `y`、是否读 official valid/test。
- 训练脚本的环境变量和命令行不得包含 test 路径。

---

## 2. 冻结的结构定义

### 2.1 物理键

每个分子使用 cache 的**无向物理键**一次：`bond_index` 的 `u<v` 行，
`bond_ptr` 分片。PyG 的双向 edge storage 不重复计数。

### 2.2 键类型

**实测（冻结，先于所有结果）:** 官方 train cache 的 249,279 条物理键的
`bond_type` 只取 `{1,2,3}`（单/双/三键），没有第 4 类（芳香）在数据中出现。
`BOND_CATEGORIES=4` 只是对象 builder 的预留宽度。因此：

- **主统计 `J_raw` 使用实际观察到的 3 类词表 `{1,2,3}`，每关系 6 个无序类型对。**
  这是对任务书中“4 类/10 对/40 维”模板的**预注册偏离**，理由是数据中没有
  第 4 类；不用一个恒为零的 slot 制造虚假维度。
- 为兼容任务书的 4-slot 口径，额外计算 **4-slot 版本**（6 个真实对 + 4 个
  含空 slot 的对），只作 `compatibility_diagnostic` 保存；**不进入任何
  ridge/MLP/门禁/审计头**。
- 主统计维度：J_raw = 4 关系 × (6 频率 + `log1p(pair_count)` + `empty_mask`)
  = **32 维**；J_latent = **4 维**；总计 **36 维**，低于 52 维上限。

### 2.3 四种键对关系（分别定义、可重叠、分别归一化）

对分子 `G` 的物理键集合 `E_G`，构造 line graph `L(G)`：节点是物理键，
两节点相邻 iff 两条键共享一个原子。`d_L(b,c)` 是 line graph 最短路距离。

- **R1:** `d_L(b,c) = 1`（共享原子）。
- **R2:** `d_L(b,c) = 2`。
- **R3:** `3 ≤ d_L(b,c) < ∞` 且 `b,c` 在同一连通分量。
- **R4:** `b,c` 同属于至少一个**含环的 biconnected block**。含环 block
  定义为边数 ≥3 的 biconnected component（即该 block 内至少有一个环）。
  使用 `networkx.biconnected_components`；不把 bridge/单边 block 算入。

每种关系内部只取无序键对 `b<c`，不重复、不跨分子。关系之间允许重叠；
`N_r = |R_r|` 是关系内无序对数。

### 2.4 `J_raw`：结构条件的键类型联合分布

对每个关系 `r`：

```
count_r[t] = #{(b,c) in R_r : unordered_type_pair(type_b,type_c)=t}
p_r[t]     = count_r[t] / N_r        if N_r > 0 else 0
J_raw     = [p_r[t] for r in R1..R4, t in 6 pairs]
            + [log1p(N_r) for r]
            + [1{N_r=0} for r]
```

- 频率在每个关系内独立归一化；不把重叠关系当作互斥分解。
- 空关系：6 个频率全 0，`log1p(0)=0`，mask=1。
- 不使用标签、学习得到的 target 派生信号、logP/SA/cycle penalty。

### 2.5 `J_latent`：训练后键内容的关系条件协同变化

对冻结 backbone 的 `c_loc ∈ R^{n_b × 96}`：

```
h_b = c_loc[b] - mean_{b' in G}(c_loc[b'])
T_r = (1/N_r) * sum_{(b,c) in R_r} dot(h_b,h_c) / 96    if N_r>0 else 0
J_latent = [T_R1, T_R2, T_R3, T_R4]
```

- 分子内中心化；不使用标签。
- 空关系置 0，并用 `J_raw` 的 empty mask 标识。
- A3/A4 的 `J_latent` 由被审计的冻结 checkpoint 产生；B 阶段由对应的
  Fit8k backbone checkpoint 产生（逐 backbone seed 重算），不跨 seed 复用。
- 这是“键内容在真实拓扑中的对应关系”的 4 维摘要；不是类型边际，也不是
  图级二阶通道的重新命名。

### 2.6 null 统计（信息对照，不是合法分子）

- `J_raw_null`：每个分子内对 `bond_type` 做固定种子的置换，保持类型直方图，
  重算 `J_raw`。**这可能破坏化学价态，只是信息对照，不得称为合法分子。**
- `J_latent_null`：每个分子内对 `c_loc` 的行做固定置换，保持 c_loc 多重集，
  重算 `J_latent`。这只是“键内容与键位置对应关系”的计算对照。
- 置换种子：`SeedSequence([20261026, molecule_index])`，在结果前冻结。

---

## 3. 阶段 A1 — 真实信息流前向审计（CPU，0 GPU）

对至少 `D0`、`D1` 两个 checkpoint，在固定 1,000 个 train 分子上：

1. 用 hooks/wrapper 捕获 `a`（原子状态）、`b`（键状态）、`w`（junction 状态）、
   `c_glob`、`c_loc`、reader 输入 `z`（必须 513 维）、最终 `f_DCC(z)`。
2. 核对：
   - `c_loc` 使用真实 `junc_bond` 归属：`W_l[b]=Σ_{j touches b} w_j`，
     `V_l[b]=Σ w_j²`，与 captured `c_loc` 的 composer 输入逐项一致。
   - `c_glob` 的 Null grouping：`W_g[b]=n_b*mean_G(w)`，
     `V_g[b]=n_b*mean_G(w²)`，与 captured 输入一致。
   - 513 维切片：`[z(96), a_sum(96), a_sq(96), g(33), zl(96), zl2(96)]`；
     逐切片来源、尺度、是否 train-only 标准化。
   - `reader(z)` 与原始 `forward` parity：max abs diff ≤1e-5（float32）；
     记录实际值。
   - 输入来源审计：所有特征只来自结构、训练得到的表示或 train-only 标准化；
     无 valid/test、无跨 split 绑定缓存。
3. 不变性（单分子 + 小 batch）：
   - atom relabel（随机置换原子，重排 atom/bond/junction 行与索引）；
   - bond permutation（物理键顺序置换，重排所有键级张量）；
   - endpoint reversal（`bond_index` 两端交换、`junc_index` 的 i/k 交换、
     `junc_bond` 两列交换；注意 `bond_feat` 的 ring 标志对称）；
   - batch regrouping（同分子与不同 companion 分子组成 batch）。
   记录 max abs Δpred、finite、以及是否在 1e-5 内。
4. 非退化性指标（对 1,000 分子，逐对象）：
   - 有效秩/participation ratio：`PR = (Σλ)²/Σλ²`（within-molecule 中心化
     对键对象；图级对 z）。
   - 奇异值/特征值 top1/top5/top10 份额、90%/95% 能量维数。
   - 图内键状态变异：per-molecule pairwise RMS / bond RMS、coefficient of
     variation、`frac(pairwise distance < 1%*mean norm)`。
   - 常量通道比例：std < 1e-6 的通道占比。
   - `c_loc` 的 PR 必须单独报告；不能把初始化低秩结论直接推广到训练后权重。

**A1 输出:** `analysis/forward_audit.json`、`analysis/nondegeneracy.json`、
`analysis/provenance.json`、`analysis/label_access_audit.json`。

---

## 4. 阶段 A2 — 代数不可辨识性与真实结构响应

### 4.1 A2.1 纯代数反例（独立 float64）

固定一个 6 键 path 的 line graph。构造两组 96 维键状态 `X`、`Y`，满足：

- `sum(X) = sum(Y)`（float64，逐元素）
- `sum(X²) = sum(Y²)`（float64，逐元素）
- 其他 reader 输入块完全相同
- 键状态与键位置的对应发生改变
- 一个依赖键间对应关系的统计量（例如固定 line graph 上的
  `Σ_{edges} <x_b,x_c>` 或关系条件 `T_r`）发生非零变化

用独立 numpy float64 实现验证；把两组状态分别装进实际 DCC-L reader 的
513 维输入，验证 `reader(z_X) == reader(z_Y)`（max abs diff 0）。同时报告
反例的数学构造和它**不能**证明什么：这不是两张真实分子图的不可辨识性，
也不是标签反事实。

### 4.2 A2.2 冻结真实模型的响应

在 200 个真实 train 分子上：

- 保持同一分子的 `c_loc` 多重集，在键位置间做**预先固定**的置换
  （`SeedSequence([20261026, molecule])`，与 A3 的 null 同源）。
- 只替换 `c_loc` 的行，保持 `c_glob`、`a_sum/a_sq`、`g` 不变；重算
  `zl=Σc_loc`、`zl2=Σc_loc²`；检查 pooled reader 输入和预测是否不变。
- 检查明确依赖对应关系的联合统计（`T_R1..T_R4`、line-graph 相邻内积）
  是否变化；报告 |Δ| 分布、变化分子比例。
- 同时报告真实 `c_loc` 的图内变异（PR、pairwise RMS、CV），避免把
  “所有键状态近乎相同”误读为信息丢失。

**边界:** 这不是合法分子图的重构，也不是标签反事实；只证明冻结计算的
依赖结构。真实 `c_loc` 可能已经编码自身周围的对应关系，`global33` 也含
图结构；因此 A2 不自动推出“真实分子信息缺失”。

**A2 输出:** `analysis/algebraic_counterexample.json`、
`analysis/rebinding_response.json`。

---

## 5. 阶段 A3 — 两个候选统计族的实现与检查

### 5.1 实现

`code/joint_stats.py`：

- `line_graph_relations(bond_index, n_atoms)` → R1–R4 pairs；
- `biconnected_ring_blocks(bond_index, n_atoms)` → R4；
- `j_raw(bond_type, relations)` → 32 维；
- `j_latent(c_loc, relations)` → 4 维；
- `j_raw_null` / `j_latent_null`（固定种子）；
- 独立 float64 穷举参考 `reference_j_raw/j_latent`（双重循环，不调主实现）。

### 5.2 必做检查（全部通过才允许进入 A4）

1. 主实现与独立穷举 float64 参考在 100 个随机分子上一致（≤1e-9 相对/绝对）。
2. 图重标号（atom relabel）：J_raw/J_latent 不变（在 1e-8 内；J_latent 的
   c_loc 需按键重排同步置换）。
3. 键顺序置换：不变。
4. endpoint flip：不变。
5. 缓存与在线计算一致：缓存文件 hash 与重算一致。
6. 边界：空图（无键）、单键、无环树、单环、多环/稠合环。
7. 训练前后分布：D0 vs D1 的 J_latent 均值/方差/稀疏比例。
8. 与 `global33`、`z` 的相关性：Pearson/Spearman，逐维和块级。
9. 与 null 统计比较：J_raw_null、J_latent_null 的分布和相关性。
10. 训练/评估一致：同一 checkpoint 的 eval 模式前后一致；无 dropout。

**A3 输出:** `analysis/joint_stats_checks.json`、
`analysis/joint_stats_distribution.json`、`analysis/joint_stats_corr.json`。

---

## 6. 阶段 A4 — 不使用标签的条件冗余审计

### 6.1 问题

**完整 `z ∈ R^513` 在现实分子分布上是否足以重建 J_raw 和 J_latent？**

### 6.2 固定特征与重建器

- 特征：
  - `z`：完整 513 维（主）；
  - `g`：`global33`（33 维，主对照）；
  - `z_no_g`：去掉 z 中 33 维 global 切片的 480 维（消融，辅助）。
- 目标：`J_raw`（32 维，主）、`J_latent`（4 维，主）。
- 标签无关划分：
  - `Fit8k`（8,000）：拟合 ridge/MLP；
  - `ProbeFit`（1,000）：out-of-sample 主报告；
  - `AuditEval`（1,000）：只作 label-free 的第二 out-of-sample 检查；
    不参与任何选择。
- 线性 Ridge：
  - X/Y 标准化只用 Fit8k 的 mean/std；
  - alpha 网格固定为 `[1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]`；
  - 5-fold CV on Fit8k，按 mean squared error 选 alpha；每个目标块共享
    一个 alpha（多输出 ridge），不做逐维调参；
  - 低方差维：Fit8k std < 1e-6 或 < 1e-3×该块 median std 的维度标记为
    `degenerate`，单独列出，不进入门禁/总体 R²。
- 非线性敏感性：
  - 固定 MLP：`in→128→out`，SiLU，AdamW lr 1e-3，wd 1e-4，batch 256，
    max 300 epochs，early stopping on 内部 1,000 行 holdout（Fit8k 内按
    固定 hash 取最后 1,000 行），patience 20，选 holdout MSE 最小；
  - 对 `z`/`g` 分别与 `J_raw`/`J_latent` 组合；
  - 只作敏感性检查，不替代主 ridge 结论。

### 6.3 报告

- 逐维 out-of-sample R²、块级 R²（方差加权）、normalized RMSE；
- 残差协方差特征值、PR、top1/top5 份额；
- `z→J` vs `g→J` vs `z_no_g→J` 的差异；
- J_latent 是否重现 RCC 的高冗余现象；
- null 统计的可重建性（J_raw_null、J_latent_null）作为对照；
- 2 个冻结 seed 的独立结果（不取平均最优）。

### 6.4 门禁（STOP_BEFORE_NEW_GPU）

- 对每个候选块（J_raw、J_latent）和每个冻结 seed，计算 `z→J` 的块级
  out-of-sample R²（ProbeFit）。
- **如果所有有效候选块在所有冻结 seed 上 R² ≥ 0.98，且没有有意义的独立
  残差变异（残差方差份额 ≤2%，或残差 PR 低且无稳定结构），则
  `STOP_BEFORE_NEW_GPU`。**
- 这是工程成本筛选，不是 Bayes 信息充分性定理。
- 如果至少一个块在两个 seed 上都 R² < 0.98，且 ProbeFit/AuditEval 差异
  <0.02，则认为存在稳定非冗余成分，进入 B。
- 不允许根据标签挑选最有利的维度/关系子集；后续报告完整预定义块。

---

## 7. 阶段 B — 独立分子上的预测增量检验（仅在 A 门禁通过后）

### 7.1 划分（label-free，先于任何 B 结果）

- 只用 official train 10,000 个结构。
- 内容键：`WL_hash(3 iterations, node atom type, edge bond type) +
  n_atoms + n_bonds`；若 WL 碰撞，用 `networkx.is_isomorphic` 精确核验。
  同构/重复结构必须同角色，不能跨 Fit/ProbeFit/AuditEval 泄漏。
- 结构分层：`n_atoms` 固定 bins `[≤20, 21–24, 25–28, ≥29]` ×
  `n_ring_systems` bins `[0, 1–3, ≥4]`（来自 `global_feat[:,6]`，label-free）。
- 每个 stratum 内按 `sha256(content_key)` 升序，按 80/10/10 分配；
  用 largest-remainder 调整到精确 8,000/1,000/1,000。
- 发布 `configs/split_fit8k_probe_audit.json`：row IDs、content hashes、
  每个 stratum 的计数、角色 hash、重复检查结果。
- 旧 Fit4k/Fit8k/Monitor/Locked 结果只作背景；新 AuditEval 只称为
  “本轮内部审计评价”，不声称摆脱历史选择偏差。

### 7.2 DCC-L 参照训练

- 原生 `DualChannel` / B-R0 forward，494,193 参数。
- Fit8k，两个监督 seed（0, 1），并行在 `res` 两张 GPU。
- 优化器/配方：AdamW，lr 1e-3→1e-5，wd 1e-5，batch 128，raw-y L1，
  warmup 500 updates，clip 5，dropout 0，**精确 47,400 updates**。
- checkpoint：**final-update checkpoint**（第 47,400 update 的模型），
  不用 ProbeFit/AuditEval/official valid 选择。训练中只记录 train/monitor
  （Fit8k 内部固定 500 行）健康度。
- Fit-only 标准化：global33、root/block scalar 的 mean/scale 只由 Fit8k 行
  计算。
- SSL：
  - 优先在 Fit8k 上按旧 masked-SSL 机制重训一个 encoder（记录成本）；
  - 两个监督 seed 可以复用一个 SSL encoder，但必须披露这只估计部分训练
    随机性；
  - 如果只能使用历史 full-10k SSL，标记 `transductive-unsupervised`，
    不得声称严格 inductive；
  - 禁止静默退化。
- 训练前用 2–3 个真实 epoch profiling 投影两张 GPU 的合计成本；到硬 cap
  停止并如实报告，不缩短 horizon 伪装完整实验。

### 7.3 审计头（不训练新 backbone）

冻结 DCC-L 全部参数。在 ProbeFit（1,000）上训练三个小型、同规则正则化的
残差诊断器：

- **Z:** 输入完整 `z`，预测 `y - f_DCC(z)`。
- **ZJ:** 输入完整 `z` + J_raw + J_latent（36 维），预测同一残差。
- **ZN:** 输入完整 `z` + `J_hat`（36 维），其中 `J_hat` 仅由 `z` 重建
  （A4 的 ridge，Fit8k label-free 拟合；逐 backbone seed 重拟合），
  作为“增加输入维度但没有新增信息”的维度匹配控制。
- 原始 `f_DCC(z)` 作为无校准参照。
- 主诊断器 Ridge；alpha 网格固定
  `[1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]`，只在 ProbeFit 内部 5-fold CV
  选择；不做逐维/逐样本调参。
- 预注册 robust 敏感性：固定 `HuberRegressor(epsilon=1.35, alpha=1e-4)`
  一次，不调参；只作敏感性。
- J_hat 重建器与所有标签访问隔离：只用 Fit8k 的 z/J，不读 ProbeFit 标签。
- 对每个 backbone seed：
  1. 冻结并哈希 backbone checkpoint；
  2. 冻结 ProbeFit 上的所有诊断器；
  3. 生成 AuditEval 全部预测；
  4. 哈希预测文件；
  5. **最后**读取 AuditEval 标签并一次性评分。

### 7.4 主端点与统计

- `Gain_J = MAE(Z) - MAE(ZJ)`；正值表示联合统计带来增益。
- `Gain_vs_N = MAE(ZN) - MAE(ZJ)`；正值表示优于等维冗余控制。
- 两个 backbone seed 独立报告，不只报平均最优。
- 逐分子配对 bootstrap（10,000 次，固定重采样）估计固定模型下的样本
  不确定性；**不能解释为训练种子不确定性**。
- 对照：MAE、RMSE、median AE；best 10 worst rows 的贡献和去除后结果；
  两个 seed 的效应方向；诊断头相对原始 DCC 的收益；训练拟合与 AuditEval
  差距；Z/ZJ/ZN 预测相关性和残差相关性。
- 不做 test 评价。

### 7.5 泛化分层（标签只在冻结后用于解释）

支持度严格按 Fit8k 的结构计算：

- `full_support`：所有 radius-2 原子环境至少有 5 个独立 Fit 分子支持；
- `low_support`：没有 0 支持，但含 1–4 个 Fit 分子支持的环境；
- `unseen`：存在 Fit8k 中完全未见的 radius-2 环境。

另做预先声明的诊断分层：

- 普通标签 `y > -3` 与尾部 `y ≤ -3`；
- 分子大小（固定 bins）；
- ring-rich（`n_rings ≥ 4`）；
- 原 cycle penalty 有无影响（仅在标签审计可复现且口径一致时）。

所有标签定义的分组只能在冻结后用于解释，不许用于训练加权、模型选择或
参数搜索。小样本分层必须给 n、区间和单分子驱动现象。

### 7.6 正式决定规则（写入 `decision.json`）

- **A. NONREDUNDANT_AND_TASK_RELEVANT**：新统计相对完整 z 稳定非冗余；
  ZJ 在两个监督 seed 上都优于 Z；ZJ 也优于 ZN；主增益 ≥0.003 MAE；
  配对区间与去极端行检查支持不是少数行伪影。
- **B. NONREDUNDANT_BUT_NOT_PREDICTIVE**：有真实结构增量，但没有稳定
  标签收益。结论是“当前表示/数据/诊断函数下未发现预测价值”，不否定
  全部关系建模。
- **C. REDUNDANT_OR_RECONSTRUCTIBLE**：统计可由完整现有表示稳定重建；
  停止该统计族，优先考虑可学习性或结构归纳偏置。
- **D. CAPACITY_OR_CALIBRATION_ONLY**：ZJ 有改善，但 Z 或 ZN 同样改善；
  新增联合统计没有独立收益。
- **E. INCONCLUSIVE**：种子、区间、分层或训练健康度不足。
- 可同时标注 `BENCHMARK_TAIL_DOMINATED` 等诊断标签，但不得覆盖主分类。
- 只有 A 才在报告中提出下一代原创方法的最小计算原则；本轮不训练新架构。

---

## 8. 预算与停止规则

### 8.1 GPU 预算

| 项 | 预计 A100·h |
|---|---:|
| A1–A4（CPU，前向/统计/ridge/MLP 敏感性） | 0 |
| B: Fit8k masked-SSL（1 个 encoder） | ~0.45 |
| B: DCC-L Fit8k 2 个监督 seed（精确 47,400 updates） | ~0.70 |
| B: 2–3 epoch profiling + 启动/评估 | ~0.10 |
| **合计（若 B 执行）** | **~1.25** |
| 软 cap | **4.0** |
| 硬 cap | **6.0** |

- 所有 GPU 运行通过 `rr run res ... --pool res-gpu{0,1}`；记录 job occupancy。
- 训练前 profiling；若投影超过剩余预算，停止，不缩短 horizon 伪装完整实验。
- 失败运行、取消原因、CPU 时间都记入 `RUN_LEDGER.md`。

### 8.2 停止规则

1. A3 任一必做检查失败 → 修实现并重跑；不得带错误进入 A4。
2. A4 门禁满足 → `STOP_BEFORE_NEW_GPU`，不训练 B。
3. 预算硬 cap 到达 → 停止并报告已完成部分。
4. 训练出现 NaN/发散/参数不在 optimizer → 停止，修复后重跑；错误运行不进入正式结果。
5. 任何 official valid/test 标签被读入评价 → 立即停止，报告违规，不把结果当正式。
6. 不使用 official valid 选 checkpoint；若实现上不得不评估 valid 作监控，
   必须在计划中声明并证明不参与选择。

---

## 9. 交付物

- `AUDIT_PLAN.md`（本文件）
- `METHOD.md`：实际统计定义、可辨识性反例、数据访问协议
- `REPORT.md`：结果、诊断、竞争解释、局限、七个明确答案
- `decision.json`：机器可读判定
- `analysis/`：前向审计、统计冗余、模型训练、分层与误差结果
- `code/`：可复现审计脚本和相关测试
- `RUN_LEDGER.md`：GPU 运行、hash、commit、成本
- `configs/audit_v1.json`：冻结常量
- `configs/split_fit8k_probe_audit.json`（若 B）

---

## 10. 非声明

1. 不声称“结构信息一定不足”或“DCC-L 一定充分”。
2. 不把 A2 的 activation-level 反例外推为真实分子图的不可辨识性。
3. 不把 B 的审计头包装成新的 ZINC 预测方法或性能 champion。
4. 不把关联/依赖性写成因果。
5. 不把 official valid 的历史暴露说成新鲜盲测。
6. 不把 test 数字作为本轮评价输入。
7. 不为了提交研究产出而编造新架构；若结论是 `NO_JUSTIFIED_NEW_MODEL`，
   直接写这一结论。
