# KSVD / MolHIV 实验审计与突破路线

**日期：2026-07-26**  
**适用范围：** 当前仓库中已完成的 MolHIV standalone、graph residual、localized node-token、JK、atom-conditioned readout、显式 ring、多尺度、字典变体与 terminal test 实验。  
**协议边界：** 后续模型选择不得再使用 official-valid；official-test 已被历史实验暴露，只能作为已污染的历史终点，不能再反馈到配置选择。

---

## 1. 总判断

### 1.1 路线确实走窄了，但被测透的是“token injection”，不是 KSVD 本身

当前绝大多数实验共享同一表示核心：

```text
radius-2 centered ego patch
-> 848-d fixed vector
-> D32 / OMP T3 / fixed KSVD dictionary
-> 32-d sparse code
-> projection / gate / readout / residual engineering
```

后续变化主要发生在 sparse code 已经生成之后：换 gate、注入层、router、bilinear、JK、graph residual、atom readout、容量、正则化和 ensemble。它们没有改变 dictionary atom 在图中的角色，也没有创造新的消息传递拓扑。因此，当前负结果更准确的表述是：

> **“固定局部 sparse code 作为 GINE side feature”已经基本测透；不能据此断言 KSVD representation learning 本身无效。**

### 1.2 现有证据不支持“当前 KSVD 模型已经稳定到 0.80”

- 5-seed frozen official-test：KSVD `0.75189 ± 0.01514`，GINE-h64 `0.75270 ± 0.01469`。
- fixed 双分支 ensemble terminal test：`0.76191 ± 0.01176`。
- 该 ensemble 与普通 two-GINE seed-pair mean 几乎相同，因此不能归因为 KSVD-specific gain。
- fixed split 1729 上 atom-additive readout 的 model-seed mean 为 `0.79403 ± 0.00724`，但固定 model seed0 换三组 inner splits 后，KSVD 三次都低于 matched GINE；PCA 与 KSVD 的 split mean 几乎相同。

所以现在距离 CIN 的约 `0.80–0.81` 仍有实质差距，且核心差距是跨 scaffold 泛化与方法归因，而不是参数量。

### 1.3 KSVD 尚有两条真实正证据

1. standalone rich-statistics readout 中，KSVD 对 random/PCA 有稳定优势；这说明字典不是完全随机噪声。
2. localized KSVD 在若干固定 split/seed 上可改善均值或方差；这说明 GINE 能利用该信号，但当前利用方式不稳。

这两点足以支持继续研究，但不足以支持继续围绕同一 token fusion 做小修小补。

---

## 2. 当前实验矩阵审计

| 维度 | 已测试内容 | 结果 | 判断 |
|---|---|---:|---|
| Backbone 容量 | h64/l4、h80/l3、参数匹配 h70 | KSVD h64/l4 `-0.00150`；h80/l3 `-0.01872` | 容量不是当前瓶颈；停止普通 scaling |
| Dictionary 容量 | D48/T4 | `-0.07070` | 明确停止扩大 D/T |
| 多尺度 | r1+r2 三种组合 | `-0.0046` 到 `-0.0133` | 普通 concat/并行多尺度已失败 |
| Token 注入 | all/前1层/前2层、gate scale、LR、dropout、L2、warmup、EMA | 无稳定晋级 | 注入优化空间基本耗尽 |
| 自适应融合 | node router | `-0.01498`，3/3 失败 | 更自由的 router 放大过拟合 |
| 高阶交互 | node-local、support mask、edge-neighbor、dynamic bilinear | 约 `-0.011` 到 `-0.027` | 在固定 code 上加二阶分支无效 |
| Graph context | graph-global KSVD context | 稳定性略改善，准确率未晋级 | 可保留作稳定性 ablation，不是突破 |
| Atom-conditioned readout | additive 96 参数、specific 2048 参数 | additive 只在固定 split 好；跨 split 不胜 GINE；PCA 匹配 | 证明 basis-conditioned readout 可行，未证明 KSVD-specific |
| Dictionary sampling | graph-balanced、cluster-balanced、mixture | full-inner 约 `-0.01` 到 `-0.02` | 频率均衡/分域不是缺失机制 |
| Task-aware dictionary | positive reweight、conditional、label augmentation、atom selection | 未通过 random/PCA matched controls | 不再直接对 label-aware KSVD 扫参 |
| Stability | seed consensus、atom/code averaging、reconstruction weighting | 中性或负 | 普通 dictionary-seed instability 不是主因 |
| Unrolling | bounded unrolled KSVD variants | 全负 | 固定 848-d 空间内端到端微调不是突破 |
| 显式 ring | broadcast/readout/typed stats | 最佳仅 `+0.00077`，1/3 wins | 现有实现未做 incidence message passing；不应转成手工 ring 主线 |
| Ensemble | GINE + KSVD graph residual | test `0.76191` | 主要是双模型方差压缩，非 KSVD 归因 |
| Standalone | KSVD rich statistics | KSVD 5/5 胜 random；绝对性能低 | 最强 KSVD-specific 证据，说明信息存在但下游机制不对 |

---

## 3. 三个此前没有被充分正视的问题

### 3.1 内部模型选择目标与最终 scaffold 目标不一致

当前 `run_molhiv_node_tokens_inner.py` 使用 `StratifiedShuffleSplit`，即 official-train 内的随机分层拆分。它控制了标签比例，却没有隔离 molecular scaffold。

这意味着：

- inner selection 主要奖励同分布局部模式复用；
- official-valid/test 考察的是新 scaffold；
- KSVD 恰好最容易学习训练集中高频局部 patch，因此 random inner split 可能系统性高估它；
- n=8000 时每个 960-graph inner-valid 只有 36 个正例，epoch AUC 噪声很大。

**优先修复：** 用 SMILES 的 Bemis–Murcko scaffold 作为 group，在 official-train 内建立 3 个固定 `StratifiedGroupKFold` development folds。后续所有新路线先过这三折，停止 random inner split 作为主晋级依据。

### 3.2 当前 patch 并非完全“无手工 ring”

`centered_ego_vector()` 的 base 调用了 `labeled_wl_ring_patch_features()`，其中额外加入 13 维人工 cycle/ring statistics：cycle rank、cyclic edge ratio、3–8 环长度 histogram、aromatic/ring atom ratio 等。

此外，标准 OGB atom features 本身包含 aromatic/ring flags；这一点 GINE baseline 也会看到，和 CIN 显式构造 cycle cells 不是同一强度的先验。但当前 KSVD patch 的 13 维额外 cycle statistics 必须披露，否则“完全由 KSVD 自学习 motif”的叙事不够干净。

**建议：** 新主路线至少同时维护两个 cache：

1. `wl_chem_clean`：去掉额外 13 维 cycle statistics；
2. `wl_chem_ring_legacy`：保留当前 848-d representation，作为性能/历史对照。

如果 clean 版本只有小幅下降，应优先采用 clean 版本；如果下降很大，则必须把 ring statistics 明确写成输入先验，不能声称纯自学习。

### 3.3 当前 sparse code 没有成为图中的实体

目前 dictionary atom ID 在经过 projection 后很容易被 GINE 混合掉；atom 只作为 32 个坐标轴存在，不拥有状态、边界或独立更新。模型也没有因 KSVD assignment 获得新的 communication path。

因此即使 sparse code 有信息，它也只能作为已有 GINE node state 的偏置。3-layer GINE 已覆盖大量 radius-2/3 局部信息，冗余是预期结果。

---

## 4. 主突破：KSVD-induced sparse motif lifting

核心转变：

> **不再把 KSVD code 当作普通特征，而是把 sparse assignment 变成 atom 与 learned motif entities 之间的 incidence relation。KSVD 从 side feature 变成 message-passing topology 的生成器。**

这仍然不同于 CIN：

- CIN 通过人工定义的 cycle lifting 创建高阶 cells；
- 本路线的 motif 类型和 incidence 权重来自 train-only KSVD/OMP；
- 不输入显式 scaffold；clean 版本不输入额外 cycle histogram；
- 高阶对象是数据学习出的 motif slot/occurrence，而不是预先枚举的 ring 类型。

### 4.1 Pilot P0：graph-local motif slots（最快、最强归因）

对每个 molecule 创建 `D=32` 个 graph-local motif slot。节点 sparse code 定义加权二部图：

```text
A[v,j] = 1[j in supp(z_v)] * |z_vj|
sign[v,j] = sign(z_vj)
```

在 GINE layer 1 后只做一轮：

```text
atom h_v
 -> sparse atom-to-slot aggregation using A[v,j]
 -> slot update conditioned on dictionary atom ID e_j and sign/magnitude stats
 -> sparse slot-to-atom broadcast using normalized A[v,j]
 -> zero-init residual gate
 -> remaining GINE layers
```

性质：

- KSVD assignment 直接定义额外 adjacency；
- 同一 molecule 内激活相同 learned motif 的原子可以交换信息；
- 只有一个 GINE backbone；
- scatter edges 约为 `T * num_atoms`，T=3，运行开销可控；
- 用 rank-16 bottleneck 时预计新增约 8k–12k 参数，总参数约 46k–50k；
- residual gate 零初始化，初始预测严格等于 matched GINE。

这比 atom-additive readout 更有可能产生真正增益，因为 motif information 在后续 GINE 层之前参与表示更新，而不是只在最终 logit 上加一个分数。

### 4.2 Pilot P1：motif occurrence hyperedges（更接近 learned higher-order cells）

若 P0 有正信号，再保留空间 occurrence：

1. 每个中心 atom 的 radius-2 patch 做 top-1 或 top-T OMP assignment；
2. 创建 occurrence node `o=(center, atom_id)`；
3. occurrence 与该 ego patch 的成员原子建立 incidence edges；
4. occurrence embedding 包含 KSVD atom ID、coefficient sign/magnitude、center/hop encoding；
5. 做一轮 `atom -> occurrence -> atom` 更新。

首版只保留 top-1，避免 occurrence 数和运行时间扩大约 3 倍。P1 的优势是保留 motif 的空间实例和边界；缺点是 incidence edge 数约为 `num_atoms * ego_size`，比 P0 更贵，而且固定 radius-2 ego 本身仍可能贡献 shortcut，因此必须有 no-ID topology control。

### 4.3 Pilot P2：task-aware occurrence selector（通过 P0/P1 后再做）

固定无监督 KSVD dictionary，不用 HIV label 更新 dictionary；仅让下游学习哪些 KSVD occurrence 应被 lift：

```text
g_o = sigmoid(q([z_o, reconstruction_error_o, h_center]))
```

并加稀疏/熵约束，限制激活预算。这样任务信息只选择 learned motif occurrence，不重新定义 motif。它比历史 adaptive token router 更合理，因为 gate 控制的是高阶实体是否存在，而不是对同一个 side feature 做逐层自由缩放。

---

## 5. 必须配置的 matched controls

任何 incidence 结果都不能只比较 GINE。最低对照矩阵：

| Control | 保留什么 | 打破什么 | 回答的问题 |
|---|---|---|---|
| matched GINE | backbone/seed/data order | 所有 motif lifting | 总增益是否存在 |
| KSVD incidence | 完整 sparse support、系数、ID | 无 | candidate |
| PCA incidence | 相同 D/T/参数/拓扑实现 | KSVD objective | 是否只是一般 basis |
| random-patch incidence | 相同 D/T/参数/实现 | learned dictionary | 是否只是随机局部分组 |
| graph-wise shuffled KSVD IDs | 每图 support degree、系数分布 | 跨图 atom semantics | 是否需要稳定 motif identity |
| no-ID sparse transport | KSVD incidence support/weights | dictionary atom identity embedding | 是否只是通用 sparse pooling |
| P1 ego no-ID | 完全相同 radius-2 incidence | KSVD type | 是否只是 radius-2 shortcut |

可选强对照：参数匹配的 learned dense slots。若它也同样有效，贡献应归因于 generic slot message passing，而不是 KSVD。

---

## 6. 新 development protocol

### 6.1 数据隔离

- 只使用 official-train 构造 scaffold groups、字典与 development folds；
- official-valid 不再用于新路线选择；
- official-test 不再用于任何反馈；
- 后续真正 terminal claim 应转移到新的 benchmark 或预注册外部 split。

### 6.2 三阶段漏斗，兼顾运行时间

#### Stage A：静态可行性审计（无神经训练）

检查每个 scaffold fold：

- atom 使用率、死 atom 数；
- 每图 active slots 数；
- train/held-out scaffold 的 assignment JS divergence；
- graph-wise shuffled/PCA/random 的度分布是否严格匹配；
- occurrence/edge 数和预计显存。

停止条件：held-out scaffold 上大面积 dead/novel support collapse，或 incidence 图退化为单一常见 atom hub。

#### Stage B：快速 pilot

```text
n=8000
3 scaffold folds
model seed=0
20–30 epochs
selection-only
P0 controls: GINE / KSVD / PCA / random / shuffled / no-ID
```

当前 n=8000 selection run 通常为几十秒量级；P0 scatter 开销预计远低于再跑一个 GINE。完整 18 个 run 应以“约几十分钟而非多小时”为目标，并记录真实 wall time。

初筛晋级门槛：

- KSVD vs GINE mean delta `>= +0.003`；
- 至少 2/3 scaffold folds 获胜；
- KSVD 同时超过 PCA、random、shuffled 和 no-ID；
- fold standard deviation 不明显恶化；
- 总参数 `< 50k`，单 epoch 时间不超过 GINE 的 `1.5x`。

#### Stage C：确认

仅对通过 Stage B 的一个配置：

```text
3 scaffold folds x model seeds 0/1/2
```

正式晋级门槛：

- paired mean delta vs GINE `>= +0.005`；
- 至少 6/9 paired wins；
- KSVD-specific controls 仍保持正差；
- 参数量增幅不超过约 35%；
- 不触碰 official-valid/test。

---

## 7. 第二突破：latent patch metric + KSVD bottleneck

如果 incidence mechanism 有用但 KSVD/PCA 差异仍小，问题在 patch space，而不是 message passing。

当前 KSVD 在固定 848-d、hash/WL/人工统计混合空间中优化欧氏重构。新的版本应：

1. 在 official-train patch 上训练小型自监督 local encoder；
2. 目标使用 masked center atom/bond reconstruction、局部 augment consistency 或 context prediction；
3. encoder 不使用 HIV label；
4. 冻结 encoder，在 32–64 维 latent space 中做 KSVD/OMP；
5. sparse code 进入 P0/P1 incidence lifting，而不是回到普通 token injection。

优点：

- 去掉额外 ring histogram 后仍可学习 chemistry-preserving metric；
- KSVD 保持离散、稀疏 bottleneck；
- 与已失败 unrolled KSVD 不同：它改变的是字典学习空间，而不是在旧 848-d sufficient statistics 上多迭代几步。

风险：引入 encoder 后 attribution 更难，因此必须同时比较 latent-PCA、latent-kmeans/random 和“encoder without KSVD”。

---

## 8. 第三突破：外部无标签 molecular dictionary pretraining

当前 dictionary 只覆盖 MolHIV train scaffolds。一个保持 KSVD 初衷、直接针对 scaffold coverage 的方向是：

- 从较大、无标签的 molecular corpus 采样局部 patches；
- 预训练 clean KSVD dictionary；
- 在 MolHIV 上只做 OMP encoding 和 incidence downstream；
- 不使用外部任务标签；
- 与 MolHIV-train-only KSVD、external PCA、external random-patch 严格匹配。

这可能比继续增加 D 更有效：目标不是更多参数，而是让固定 D32 atoms 覆盖更广的化学局部空间。风险是 domain mismatch 和数据重叠审计，因此只列为 incidence pilot 后的中期路线。

---

## 9. 暂停清单

在新的 scaffold protocol 和 incidence pilot 完成前，暂停：

- width/depth 普通 scaling；
- D/T 扩大；
- 新 token gate、router、dropout、EMA、warmup 扫描；
- 同一 code 上继续堆 bilinear/readout；
- 显式 ring/scaffold 主路线；
- 双 GINE ensemble 作为 KSVD 方法主结果；
- 任何 official-valid/test 后验选择。

---

## 10. 最终优先级

1. **立即：** 建立 official-train-only 的 3 个 scaffold development folds。
2. **主 pilot：** 实现 P0 graph-local KSVD motif-slot incidence，一轮、单 GINE、zero-init、<50k 参数。
3. **同时：** 增加 clean patch cache，去掉额外 13 维 cycle statistics；legacy cache 只作对照。
4. **通过后：** P1 spatial motif occurrences，验证 learned higher-order lifting。
5. **仍无 KSVD-specific gain：** latent self-supervised patch metric + KSVD bottleneck。
6. **仍卡 scaffold coverage：** 外部无标签 dictionary pretraining，或转移到新 benchmark 做预注册验证。

一句话路线：

> **从“把 KSVD code 塞进 GINE”转向“由 KSVD sparse assignment 构造可学习的高阶 incidence topology”，并用真正的 scaffold-held-out inner folds选择模型。**

---

## 11. 与相关工作的边界

需要在论文叙事中主动承认：unsupervised motif vocabulary、motif convolution、cell-complex lifting 和 learnable lifting 已有相关工作。我们的潜在贡献不能只写成“增加 motif nodes”，而应明确为：

1. KSVD/OMP 学习连续、稀疏、可重构的 motif dictionary；
2. sparse support 直接生成 weighted incidence topology；
3. dictionary atom identity 跨图共享、occurrence 在图内实例化；
4. 在 MolHIV 上用 PCA/random/shuffled/no-ID matched controls 做 KSVD-specific attribution；
5. 参数量与运行时间显著低于双-backbone或高容量 cellular model。

相关一手工作包括：CIN/Cellular WL、Molecular graph motif convolution、Differentiable Cell Complex Module、以及 2026 年的 task-aware differentiable lifting。它们共同说明“lifting/topology，而不只是附加特征”是合理方向；同时也意味着方法新颖性必须落在 **KSVD-induced sparse weighted lifting 与严谨归因**，不能只落在“motif incidence”四个字上。
