# SCSB v1 — 支持度条件的结构回退算子（预注册）

轮次：`zinc_support_conditioned_backoff_v1`
协议：`ssrc-zinc-support-conditioned-backoff-v1`
冻结配置：`configs/scb_v1.json`（机器可读的唯一权威）
状态：**Phase 0 CPU 门控已通过（13/13）；official test 未构造、未读取。**

---

## 0. 一句话问题

DCC-L 的真实局部 junction-to-bond 内容（`W_local = sum_{j touches b} w_j`）提供了预测价值，
但在训练支持度为 0 的 radius-2 结构上，它是否在做不可靠的细粒度外推？用一个**与标签无关、
与具体键位置绑定**的训练支持度，把这条局部统计连续地收缩到模型**已有的**图级均场统计
（`group_to_bonds`），能否在不增加参数、不增加特征模块、不改变训练配方的前提下，
改善 Fit 未见 radius-2 环境上的 raw-y MAE？

两个必须分开的命题：

1. **总量命题**：把局部统计按支持度收缩到均场，整体是否有用？（S vs L）
2. **位置命题**：收益是否来自“支持度 ↔ 具体键位置”的匹配，而不是仅仅改变了每个分子的
   支持度分布/收缩量？（S vs P）

S 只有同时通过这两条，才算机制成立。

---

## 1. 边界与不可改变项

- 官方 ZINC-subset：train=10,000、valid=1,000、test=1,000。
- 不添加任何 ZINC-full 或其他数据集的带标签样本，不使用伪标签、蒸馏、模型集成。
- 新模型与参照只使用相同的官方 train 标签；单模型 ≤500k 参数；无递归消息传递。
- **official test 在本轮任何阶段都不读取、不评分、不用于选择。**
- Phase 1 只在 official train 10k 内部工作；**official valid 在 Phase 1 不读取**。
  Phase 2 才使用 official valid 做最后开发确认；仍不读 official test。
- 所有比较臂使用**完全相同的内部训练集**。Phase 1 的 L 臂直接使用在该 Fit8k 上、
  用同一初始化/批序/配方/47,400 updates 训练出的冻结 DCC-L checkpoint，不重训。

---

## 2. 先例核对：本轮与它们的数学差异

这一节是防止“换一个 backoff 词就声称新颖”的硬约束。每一项都写清楚**保留了什么、
改变了什么、为什么不是同一个数学对象**。

### 2.1 DCC-L 原型（`zinc_dual_channel_composer_v1`）

DCC-L 同时有两条键级通道：

```
c_glob_b = GroupComposer([b_b, Wg_b, Vg_b, log1p(n_b), bond_feat_b])
c_loc_b  = LocalComposer([b_b, Wl_b, Vl_b, log1p(n_b), bond_feat_b])

Wg_b = n_b * mean_{j in graph} w_j          (group_to_bonds 均场)
Vg_b = n_b * mean_{j in graph} w_j^2
Wl_b = sum_{j touches b} w_j                (真实 incidence 求和)
Vl_b = sum_{j touches b} w_j^2
```

本轮**只**把 `Wl_b, Vl_b` 替换为

```
Wbar_b = alpha_b * Wl_b + (1 - alpha_b) * Wg_b
Vbar_b = alpha_b * Vl_b + (1 - alpha_b) * Vg_b
```

然后送入**原来的** `local_composer`。`alpha_b ∈ [0,1)`，由键两端原子的 radius-2 签名
训练支持度 `s_b` 给出，`alpha_b = s_b/(s_b+4)`。

- 保留：全部 494,193 个参数、reader、两个 composer 的宽度/深度、输入布局、SSL 初始化、
  优化器、批序、更新数。
- 改变：局部 composer 的**输入充分统计量**在每条键上的取值。
- `alpha=1` 时 `Wbar=Wl, Vbar=Vl`，与 DCC-L 逐位一致；`alpha=0` 时
  `Wbar=Wg, Vbar=Vg`，局部 composer 的输入与全局 composer 的输入**完全相同**
  （不是“去掉局部通道”，而是把局部通道回退到同一个均场统计上）。
- 无新参数、无新模块、无新特征、无额外监督。

### 2.2 `zinc_hierarchical_backoff`（KSVD 2026-09-04）

- 它的对象：一个**离散的 typed-WL 词表特征向量**。radius-3 精确 token 若在训练词表中
  未出现，就整块回退到同中心、较低 radius 的 token 计数（每块 4,098 维），
  再由线性/ridge 模型 + Optuna 拟合。
- 它的 `alpha` 不存在：回退是**0/1 的块替换**，不是连续权重；没有 per-bond 位置，
  没有可训练的 composer，没有图级均场通道。
- 本轮的对象：**一个已训练非线性 composer 的逐键输入统计量**；回退是连续的
  `s/(s+4)` 凸组合，混合目标是模型**自己的** `group_to_bonds` 均场统计；
  所有参数共享，端到端重训。
- 因此不是“层次 backoff 的再实现”，而是**在固定函数类内部改变每条键的充分统计量**。
  另外，hierarchical backoff 的历史结果是**正**的（valid +0.0177 / test +0.0303），
  而本轮不借用它的正结果作为先验；它的特征空间、模型和评测都不同。

### 2.3 `compositional_patch_sharing_oracle`（KSVD 2026-09-10）与 frozen patch sharing

- 它的对象：**冻结推理**下，把 rare/OOV 的 radius-2 精确 **token embedding** 替换/混合为
  结构近邻或高频 donor 的 embedding；不改训练，不改 composer。
- 结果：所有 5 个变体在 4/4 seed 上变差；KNN8 ≈ random donor，且略差于 frequent mean；
  伤害集中在被干预的 rare/OOV 分子。结论是“冻结 token embedding 的 donor 替换没有
  可用的共享信号”。
- 本轮的对象不是 token embedding，也不是 donor：局部统计 `Wl/Vl` 是**该键自己的**
  incidence 聚合；混合目标是**同一张图的**均场统计，不是别的分子的 embedding；
  没有近邻搜索、没有 donor、没有 KNN、没有 token 词表。并且本轮**重新训练整个模型**，
  让 composer 自己适配混合后的输入，而不是在冻结权重上做替换。
- 这个先例是本轮最强的**反面先验**：它说明“拿别处的统计替换稀有内容”很可能有害。
  本轮不否认这一点；本轮的 P 臂正是用来区分“替换/收缩本身”与“支持度与位置的匹配”。
  如果 S 没有超过 P，本轮会把结果读成对该先例的又一次确认，而不是“换对象就没事”。

### 2.4 `zinc_support_transfer_probe_v1`（2026-10-23）

- 它的对象：冻结基座 + 一个**后验加性纠错头** `f_base + m_F * h(X)`，10.5k 参数，
  末层零初始化，在 C_dev 上选步；选择器 24/24 选 step 0（恒等），强行应用一致变差。
- 它的 `m_F` 是分子级 0/1 门（含 n₂=0 原子），不是逐键连续权重；它学习的是
  “对零支撑结构的残差修正”，不是把已有局部统计收缩到均场。
- 本轮：无新参数、无新头、无 0/1 门、无后验修正；改变的是**基座训练时的逐键输入**，
  并且从同一初始化端到端训练。它不试图学“每个环境一个修正常数”，而是让低支持键
  自动少用局部统计、多用图级均场。
- 该先例的负结果**不能**直接推出本轮也负；但它把“加性纠错头”这一族关掉了，本轮
  不再走那一族。

### 2.5 `zinc_support_weighted_objective_v1`（2026-10-27）

- 它的对象：**loss 的分子级标量权重** `w_i`，由分子内 1–4 支持原子比例导出；
  不改前向、不改参数、不改数据。结果是 valid 变差（配对均值 +0.0030），
  且最稀有三分位 **+0.0149**（两 seed 同向）；训练拟合反而更好。
- 它的机制结论：低支持环境是**方差问题**，需要“收缩/共享统计量”，而不是更多梯度注意力。
- 本轮正是对这句话的直接回应，但作用点相反：不是给低支持分子**更多 loss 权重**，
  而是在**前向**里让低支持键**更少使用**局部统计、更多使用共享的图级均场；
  权重函数从分子级标量变成**逐键** `s/(s+4)`，且训练目标保持均匀 L1。
- 数学上：SWO 改的是 `sum_i w_i |f(x_i)-y_i|` 的 `w_i`；本轮改的是 `f` 内部的
  `Wl_b, Vl_b`。两者不是同一个变量，也不是同一个干预层。
- SWO 的负结果是本轮最强的**正面动机**（它明确指向收缩），同时也是**反面警告**：
  如果 S 只是“把低支持样本的预测往均值拉”，P 也会得到同样的分布收益；S 必须超过 P。

### 2.6 `zinc_rooted_branch_operator_v0`（RBO v0）

- 它的对象：一个独立的 331k 参数模型，用根内 1/2/3 体分支、删边分支多重集、
  关系加权二体等**新构件**去组合被支持的局部片段；不做 DCC composer，无 r2 key embedding。
- 结果：Q_target 0.1284 vs P3-minus 0.1243（+0.0041，更差），未见/已见误差倍率没有改善。
- 本轮：不新增构件、不新增参数、不换主体；只对 DCC-L 已有的局部统计做支持度条件收缩。
  它不测试“被支持的构件能否组合”，而测试“已有局部统计在零/低支持时该不该少用”。
  所以 RBO 的负结果不覆盖本轮，但同样是一个警告：未见环境的问题可能不是“局部构件缺失”。

### 2.7 DCC 的 C 臂（Null-local / 均场 local 输入）

DCC 的 C 臂把 local composer 的输入**全局**替换为 `n_b * graph_mean(w)`（同一均场统计），
与 L 同参数、同初始化。历史结果：L−C = −0.00383（L 更好，CI 上界贴 0）；
在 signature-unseen 组 L−C = −0.00512（L 更好）。

这是本轮最重要的**反面先验**：在所有键上都用均场，明显不如真实 incidence 求和；
即使在未见签名组，L 也不差于 C。因此：

- 本轮不是 C 臂的重复：C 是**全局 0/1 替换**，S 是**逐键、支持度连续**的混合；
- 高支持键仍保留局部统计（alpha≈0.95+），只有低支持键被明显拉向均场；
- P 臂保留了每个分子的 alpha 分布，只破坏位置对应，所以 S−P 才是“位置是否有用”的
  干净对照。
- 如果 S 没有超过 L，最自然的解释就是 C 臂先验是对的：DCC-L 的局部统计即使在低支持
  环境也比均场好，或者至少不差；本轮会如实记录并关闭机制。

### 2.8 与 `zinc_composition_transfer_v1` 的 supervision-support 干预

- 它是**数据侧**干预：把带目标 key 的 donor 分子加入/移出训练集（`F_plus` / `F_minus`），
  改变的是标签支撑，不是前向统计；标签数量/组成变了。
- 本轮不改训练集、不加标签、不改 loss；支持度只由结构计数导出，所有臂共享同一 Fit。
  因此不是 supervision-support 干预的重复。

### 2.9 与 GOS / 全局算子搜索

- GOS route A 是在键 token 上做**全局注意力混合**（幅值敏感核 + 关系门），
  新增 57k 参数和一层新的全局算子；它不按支持度、也不回退到均场。
- 本轮无新算子、无注意力、无新参数；只改局部 composer 的输入统计量。
  两者的数学对象和参数预算都不同。

### 2.10 一句话的差异声明

> 本轮不是“层次特征 backoff”，不是“稀有 token embedding 的 donor 共享”，不是
> “后验加性纠错头”，不是“样本加权”，不是“新的分支/关系算子”，也不是“把局部通道
> 全局换成均场”。它是：**在一个固定的 DCC-L 函数类内部，用逐键、标签无关、训练集
> 自排除的 radius-2 支持度，把该键的局部 incidence 统计连续收缩到同一模型的图级均场
> 统计；并用同分布的键内置换臂隔离“位置对应”与“支持度分布”。**

---

## 3. 支持度与 alpha 的精确定义（冻结）

### 3.1 `k2` 签名

对每个原子 `v`，取**精确 typed radius-2 根签名**：

```
r1(v) = (atom_type(v), sorted[(bond_type(v,u), atom_type(u)) for u in N(v)])
r2(v) = (r1(v), sorted[(bond_type(v,u), r1(u)) for u in N(v)])
```

这与 `zinc_dccl_bottleneck_v1` 的 `dca_common.radius2_signature_sets` / 审计
`audit_coverage.radius_signatures` 完全一致（中心原子类型 + 排序后的
(键类型, 邻居 radius-1 签名)）。没有新的 key 定义。

### 3.2 支持度 `s`

- **单位**：Fit 分子上的 document frequency（一个分子对同一签名最多贡献 1 次，
  即使该签名在分子内出现多次）。这与审计的 `signature_coverage`（`freq[s] += 1`
  对 `set(sigs)` 做）和 bottleneck 的“≥5 个训练分子”分层一致。
- **范围**：只从模型当前的 **Fit** 集合构建。Phase 1 的 Fit = bjoc `fit_rows`（8,000）；
  Phase 2 的 Fit = 官方 train 10,000。Monitor / Locked / official valid / official test
  **不进入支持表**。
- **训练自排除**：对 Fit 分子 `m`，`s = df(sig) - 1`（因为 `m` 自己一定在 `df` 里，
  且每个签名只贡献 1 次）。对非 Fit 查询分子，`s = df(sig)`。
  统一 estimand 是“**除查询分子本身以外的 Fit 分子数**”：
  - Fit 分子：`df - 1`；
  - 新分子：`df`（新分子不在 Fit 中，不会自我贡献）。
- **键级**：`s_b = min(s_u, s_v)`，`u,v` 是键的两个端点。这是任务指定的规则。
- **alpha**：`alpha_b = s_b / (s_b + 4)`。`s_b=0 -> 0`，`s_b=1 -> 0.2`，
  `s_b=4 -> 0.5`，`s_b=10 -> 0.714`，`s_b=20 -> 0.833`，`s_b->inf -> 1`。
  注意：有限支持度下 `alpha<1` 恒成立；L 臂是唯一真正的 `alpha=1`。

### 3.3 置换臂 P

对每个分子 `m`，取它所有键的 alpha 向量 `a_m`，用固定 salt `20261028` 和
`SeedSequence([salt, row])` 生成 `perm`，令 `a'_m = a_m[perm]`。
逐分子多重集**精确不变**；只有 alpha 与具体键位置的对应被破坏。
置换只依赖（salt, row），不依赖标签、模型、batch 或训练顺序。

### 3.4 Phase 0 实测 alpha 分布（冻结，`cache/alpha_manifest.json`）

| 集合 | mean | median | alpha=0 | alpha≤0.5 | s=0 键 | s1–4 均值 | s5–19 均值 | s≥20 均值 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fit（LOO） | 0.7859 | 0.9216 | 5.79% | — | 11,558 (5.79%) | 0.3445 | 0.7067 | 0.9537 |
| Locked（全部键） | 0.7840 | 0.9200 | 5.37% | 17.17% | 1,326 (5.37%) | 0.3419 | 0.7059 | 0.9538 |
| Locked primary（278 分子） | 0.6419 | — | 19.15% | 33.02% | 1,263 (19.15%) | 0.3256 | 0.7015 | 0.9539 |

P 臂：逐分子多重集逐位相同；87.4% 的 primary 键 alpha 改变；逐分子
corr(alpha_S, alpha_P) 均值 0.00073（池化 0.1186 由分子间均值差造成）。

---

## 4. 数据划分（复用冻结的 label-blind 划分）

| 集合 | n | 来源 | 用途 |
|---|---:|---|---|
| Fit | 8,000 | `cscl-predictive-bottleneck-v1-A` fit_rows | 唯一训练集；支持表 |
| Monitor | 1,000 | 同 split monitor_rows | 训练中每 epoch 监控 + best-monitor 选点 |
| Locked | 1,000 | 同 split locked_rows | Phase 1 只读一次，训练结束后由独立 score 脚本读取 |

- 该 split 是 canonical-SMILES grouped、label-blind、duplicate 不跨 sides；Fit/Monitor/Locked
  互斥且覆盖 10k（Phase 0 重新检查）。
- 为什么复用而不新建：它是唯一已有**匹配 DCC-L Fit8k 两 seed 轨迹**的划分，
  因此 L 臂可以零成本、零重训、完全同 Fit/同初始化/同批序；新建划分会引入新的
  结构选择自由度并迫使重训 L，噪声和成本都更大。其 Locked 的 primary 组 n=278，
  功效计算见 §5.3。
- 分组（只用于评估，不用于训练/支持/alpha/选点）：

| 组 | n（Locked） | 定义 |
|---|---:|---|
| `primary` | 278 | `label_excess == 0` 且至少一个原子的 k2 在 Fit 中 df=0 |
| `unseen_all` | 287 | 至少一个 k2 在 Fit 中 df=0 |
| `low4_nopen` | 388 | 无 unseen，含 df 1–4 原子，无人工环惩罚 |
| `full_nopen` | 303 | 所有原子 df≥5，无人工环惩罚 |
| `ordinary_seen` | 691 | `low4_nopen + full_nopen` |
| `ordinary_label` | 932 | `y > -3`（标签分组，仅报告） |
| `tail` | 68 | `y <= -3` |
| `penalty_all` | 31 | `label_excess > 0` |

- `label_excess` 来自冻结的 `zinc_long_cycle_audit` 的 `*_cycle_audit_label.csv`；
  它对应 `label_excess = max(0, basis_max - 6)`（与 `zema_audit` 的
  `penalty = label_excess > 0` 完全一致），只用于评估分组（“无人工环惩罚”），
  不进入训练、支持度、alpha 或选点。
- 规模/长环统计：primary 组平均原子数 22.03，exact-longest≥7 比例 41.2%；
  Locked 其余 723 行为 23.31 / 32.6%。这两组统计在 Phase 0 冻结并报告；
  不放宽 primary 定义、不按标签剔除样本。

---

## 5. 三个臂与训练

### 5.1 臂

| 臂 | alpha | 说明 |
|---|---|---|
| **L** | 1（不传 `bond_alpha`） | 原始 DCC-L 前向；冻结 Fit8k checkpoint（DCB） |
| **S** | 真实训练支持度 | 本轮的候选 |
| **P** | 同分子内置换 | 保留每分子 alpha 分布，破坏位置对应 |

### 5.2 训练配方（三臂完全相同）

- DCC-L / `organized_local=True`，494,193 参数；同一 `DR.make_builder("L")` 初始化，
  同一 SSL encoder（seed 0 / 1），同一初始 state hash。
- AdamW lr 1e-3 → 1e-5 per-step cosine，warmup 500，wd 1e-5，dropout 0，
  clip 5，batch 128，raw-y L1，**47,400 updates**（Fit8k：63 steps/epoch，
  752 full epochs + 24 steps = 47,400；最后一段 partial epoch 仍做一次 Monitor 评估）。
- Fit8k 的标准化只用 Fit8k（`dcb_train.fit_only_norm`），与 DCB L 完全相同。
- 每 epoch 在 Monitor 上评估；best-monitor = `monitor_mae` 严格最小；final = 最后一步。
- Phase 1 不评估 Locked；Locked 只在所有训练结束后由独立 score 脚本读一次。
- Phase 2 用官方 train 10k，官方 valid 作为每 epoch monitor 与 best-valid 选点
  （与历史 DCC-L full-10k 协议一致），仍不读 test。

### 5.3 功效（Phase 0 冻结）

用冻结 L 的两 seed 在 primary 组上的逐分子 AE 差作为**保守**的配对 SD
（S 与 L 同初始化/同批序，真实配对 SD 预计更小）：

- n_primary = 278；SD(|AE_s0 − AE_s1|) ≈ 0.0863。
- 单 seed MDE80 ≈ 2.8 × 0.0863 / √278 ≈ **0.0145**；两 seed 均值 MDE80 ≈ **0.0103**。
- 预注册目标 0.015 高于单 seed MDE80，设计对目标是**刚好足够**的；这不是一个
  可以分辨 0.005 效应的实验。若真实效应只有 0.005–0.010，本轮很可能得到
  NEAR_MISS/失败，按预注册停止，不追加 seed。
- L 两 seed 在 primary 的 MAE 漂移 = 0.1219 − 0.1166 = 0.0052，作为训练噪声地板报告。

---

## 6. 指标与对比

### 6.1 主指标

**primary group raw-y MAE**：Locked 中无人工环惩罚、且含至少一个 Fit 未见
radius-2 环境的分子（n=278）。不使用任何环惩罚修正，直接 raw-y。

### 6.2 次指标（全部报告，不全部作为门）

- 整体 Locked raw MAE（1000 行）与 **natural-prevalence tier-weighted** MAE：
  `w_unseen=604/2000, w_low4=760/2000, w_full=636/2000`（由 Monitor+Locked 的
  label-free tier 计数冻结；sum=1）。
- `unseen_all` 287、`low4_nopen` 388、`full_nopen` 303、`ordinary_seen` 691、
  `ordinary_label`（y>-3）932、`tail`（y≤-3）68、`penalty_all` 31。
- W20 = 20-epoch 滑动中位数的逐 epoch 最小值；T100 = 最后 100 个 epoch 的中位数。
  两者都不做 checkpoint 选择。
- `train@best` = best-monitor checkpoint 在 Fit 上的 eval-mode MAE。
- 分子级 paired bootstrap（10,000 次、固定 seed、同一组 row resamples）。
- key-cluster bootstrap：按分子的 Fit-unseen k2 签名集合聚类（primary 组 274 个 cluster，
  270 个 singleton）；cluster 重采样。
- 逐分子 ΔAE、drop-top-1 / drop-top-10、最大 |ΔAE| 行、seed 间漂移。

### 6.3 对比

对每个 seed 和 seed 均值：

- `S − L`：总量命题（支持度收缩是否有用）。
- `S − P`：位置命题（支持度是否匹配到了正确的键）。
- `L − P`：辅助读数（如果 S≈L≈P，说明整个 alpha 干预无效；如果 P 明显更差，
  说明位置对应重要但总量不成立）。

负值 = 前一个臂更好。

---

## 7. 预注册判定（Phase 1）

### 7.1 PASS 条件（全部满足才进入 Phase 2）

对 `S−L` 和 `S−P` **分别**：

1. **primary**：seed 均值 Δ ≤ −0.015，且两个 seed 的 Δ 都 < 0；
   seed 均值逐分子 ΔAE 的 10k molecule bootstrap 95% CI 上界 < 0。
2. **overall**：natural-prevalence tier-weighted 整体 seed 均值 Δ ≤ −0.003，
   两个 seed 都 < 0；seed 均值 bootstrap 95% CI 上界 < 0。
3. **ordinary safety**：`ordinary_seen` 的 `max(Δ_s0, Δ_s1) ≤ +0.002`。
4. **seed stability**：`max(Δ_s) − min(Δ_s) ≤ 0.015`。
5. **drop-top-k**：seed 均值 primary Δ 在去掉 |ΔAE| 最大的 1 行和 10 行后仍 ≤ −0.010
   且符号不变。
6. **key-cluster**：seed 均值 primary Δ 的 key-cluster bootstrap 符号不变，
   95% CI 上界 < 0。
7. **safety**：`tail` 的 `max(Δ_s) ≤ +0.010`，`ordinary_label` 的
   `max(Δ_s) ≤ +0.005`。若任一 safety 失败，判定 `STOP_UNSAFE`，即使 1–6 通过。

### 7.2 非 PASS 的分类（不允许救援）

- **NEAR_MISS_STOP**：primary seed 均值在 (−0.015, −0.010]，两 seed 同向、CI<0，
  但未达 −0.015。停止；不追加 seed、不改 alpha 分母/支持度定义。
- **STOP_MECHANISM_NOT_SUPPORTED**：
  - `S−P` 通过但 `S−L` 不通过：位置对应有信号，但总量无净收益；
  - `S−L` 通过但 `S−P` 不通过：收益来自支持度分布/整体收缩，而不是位置匹配；
  - 两者都不过。
- **STOP_UNSTABLE**：任一主对比的 seed 符号不一致，或 seed 间差 > 0.015。
- **STOP_UNSAFE**：ordinary/tail 安全门失败。
- 任何非 PASS 都关闭该机制；不调 alpha 分母 4、不换 k2 定义、不换 count unit、
  不改 min-of-endpoints 规则、不加层/宽/参数、不换优化器/调度、不加 seed。

### 7.3 为什么这个门槛不是“移动球门”

- primary 0.015 是任务给定的量级；n=278 的单 seed MDE80≈0.0145，两 seed 均值
  MDE80≈0.0103，因此门槛与功效匹配。
- “两 seed 同向 + seed 均值 + CI + drop-top-k + key-cluster + stability”是对
  “训练随机性/极端单行贡献”的预注册约束，不是事后挑选。
- 如果数值在 0.010–0.015 之间，按 NEAR_MISS_STOP 处理；报告真实数字，不把它
  重写成成功，也不追加计算。

---

## 8. Phase 2（仅在 Phase 1 PASS 后）

- 官方 train 10,000 训练 S 两 seed；L = 历史 full-10k DCC-L B-R0 seed0/1
  （best-valid，494,193 参数，同配方）；官方 valid 做最后开发确认；**不读 test**。
- 官方 valid 的 primary = 无人工环惩罚、含 full-train 未见 k2 的分子（n=241）。
- 门槛：primary seed 均值 Δ(S−L) ≤ −0.010、两 seed 同向、CI<0；
  overall tier-weighted Δ ≤ −0.003、两 seed 同向；
  ordinary_seen ≤ +0.002；tail ≤ +0.010。n=241 的阈值从 0.015 降到 0.010
  是**在 Phase 2 之前**写死的（功效原因），不是因为 Phase 1 结果调整。
- 若 Phase 2 在结构外推与整体性能两方面都稳定，才写 bond-only BOSI 改写设计；
  本轮不实现、不提前移植、不增加新分支。若不通过，关闭机制。

---

## 9. Phase 0 CPU 门控（不占 GPU；失败即停止）

必须全部通过并写入 `analysis/cpu_checks.json`：

1. **划分隔离**：Fit/Monitor/Locked 互斥、覆盖 10k、hash 与冻结 JSON 一致；
   Fit 与 Monitor/Locked 无重复分子组交叉。
2. **支持表 Fit-only**：只从 Fit 构建；把 Monitor/Locked 从数据中删除后支持表 hash 不变；
   把标签随机置换后支持表 hash 不变；源码扫描无 labels 进入支持/alpha。
3. **训练自排除**：对 25 个 Fit 分子暴力重算 `s = df - 1`；
   把查询分子加入 Fit 后再用 LOO 计算，支持度不变（对 Fit 和 Locked 查询都检查）；
   非 Fit 查询用 df。
4. **置换不变性**：对 50 个分子做原子重编号 + 键端点交换 + 键序随机化；
   逐原子签名、逐键 alpha（按映射）不变。
5. **原模型复现**：`SCBDualChannel` 在 `alpha=None` 和 `alpha=1` 时与原始
   `DualChannel` 的 forward 逐位一致；加载 DCB Fit8k L checkpoint 后，
   Locked 预测与 DCB 保存的预测在容差 ≤5e-6 内一致（CPU/GPU 浮点差）；
   参数 494,193；state keys/shapes 相同；初始 state hash 与 DCB 记录相同。
6. **梯度/参数**：所有参数有有限梯度；local composer 第 2 步梯度非零；
   alpha 路径无 NaN；没有新增参数；优化器覆盖全部参数。
7. **alpha 分布有意义**：
   - primary 组 ≥5% 的键 `alpha ≤ 0.5`（实际约 33%）；
   - `s=0` 的键 `alpha=0`；`s≥20` 的键平均 `alpha ≥ 0.85`（实际约 0.954）；
   - 训练 LOO 与 Locked 推理的 alpha 分布形状可解释；
   - P 臂每分子 alpha 多重集与 S **逐位相同**；≥50% 的键 alpha 改变；
     分子内 corr(alpha_S, alpha_P) < 0.05（实测逐分子均值 0.00073；池化 0.1186 是分子间均值差造成的 Simpson 假象，不是位置对应）。
   若 alpha 近似常数、或 unseen/low-support 没有真正走向均场通路，停止。
8. **合成图端点语义**：在 3 原子 2 键的合成图上，
   `alpha=0 -> Wmix=Wg, Vmix=Vg`，`alpha=1 -> Wmix=Wl, Vmix=Vl`；
   `alpha=0` 时 local composer 的输入与 group composer 的输入逐位相同。
9. **缓存/checkpoint**：`ad_train.pt` / `ad_valid.pt` 存在且 hash 记录；
   DCB Fit8k L checkpoints 存在且 hash 与记录一致；split/support/alpha manifest 写入。
10. **无 test 访问**：源码扫描 + 运行时断言；所有 JSON 写 `official_test_loaded: false`。

Phase 0 若发现“支持度差异无意义”或“训练/推理一致性无法保证”，**直接停止**，
不启动 GPU。

---

## 10. 预算

| 项 | 计划 GPU·h |
|---|---:|
| Phase 0（CPU） | 0 |
| Phase 1：S×2 + P×2（Fit8k，2 卡并行两波） | 1.48 |
| Phase 1：L×2 | 0（冻结 DCB checkpoint） |
| Phase 2：S×2（full 10k，2 卡并行） | 0.74 |
| score / smoke | 0.08 |
| **合计** | **2.30** |

软上限 3 A100·h，硬上限 4。若冻结 L checkpoint 在远端不可用，
备用方案是重训 L 两 seed（+0.74），计入硬上限并在报告中说明。
不跑 seed 2/3、不跑宽度/分母/支持度/优化器扫描。

---

## 11. 不许做的事（NO_RESCUE）

- 不调 alpha 分母 4、不换支持度定义（document frequency / min-of-endpoints / k2）、
  不改训练/推理的支持度口径。
- 不加层、宽度、特征、参数、消息传递、第二分支、token 表、donor 表。
- 不改优化器、学习率、schedule、batch、更新数、loss。
- 不加 seed（除预注册的 2 个）、不加 arm、不换 checkpoint 选择规则。
- 不用 official test；Phase 1 不用 official valid。
- 不把 NEAR_MISS 重写成 PASS；不把“S 比 P 好但比 L 差”包装成机制成功。
- 不因为“预算还剩”而追加实验。

---

## 12. 最终报告必须回答的 5 个问题

1. 支持度条件混合是否比相同参数的原始 local composer 有实用收益？（S−L）
2. 收益是否来自支持度与具体局部结构位置的匹配，而不是支持度分布本身？（S−P）
3. 是否真的修复未见结构，而不只修复人工长环尾部？（primary 无惩罚组 vs tail）
4. 能否达到或稳定超过当前 DCC-L？离单模型 test 0.05x 的目标还有多远？
   （不读 test；只报告 valid/Fit 内部与历史 0.0663 test 的距离，标注为外推）
5. 如果失败，该机制是否值得彻底关闭？（按预注册停止规则）

---

## 13. 复现入口（待实现后填入）

```bash
# Phase 0 CPU gate
uv run python tracks/structural-sufficiency/zinc_support_conditioned_backoff_v1/code/scb_checks.py

# Phase 1 formal runs (after commit + rr deploy)
rr run res scb-S-s0 --pool res-gpu0 --gpus 1 --time 1:00:00 -- \
  python tracks/structural-sufficiency/zinc_support_conditioned_backoff_v1/code/scb_train.py \
  train --arm S --fit fit8k --seed 0 --device cuda:0
# ... S-s1 on res-gpu1, P-s0/s1 likewise

# one frozen score read after all training
rr run res scb-score-p1 --pool res-local --cpus 8 --time 1:00:00 -- \
  python tracks/structural-sufficiency/zinc_support_conditioned_backoff_v1/code/scb_score.py \
  --phase phase1 --device cpu

# local analysis
uv run python tracks/structural-sufficiency/zinc_support_conditioned_backoff_v1/code/scb_analyze.py --phase phase1
```

正式命令与输出路径以 `analysis/run_records.json` 和 `REPORT.md` 为准。
