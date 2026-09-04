# 2026-08-29 之后 MolHIV / ZINC 实验整理

> 整理范围：2026-08-29 至 2026-09-04 工作区中，与 `ogbg-molhiv`、PyG `ZINC(subset=True)`、局部 patch、KSVD、统计读出和下游模型有关的实验。
>
> 这份文档按“实验问题和证据链”整理，而不是按文件日期罗列日志。不同实验的特征对象、下游模型、数据规模和评估协议不同，数值只在同一协议内比较。

## 0. 先给结论

原先的模板：

```text
2-radius 采样 → KSVD → mean/std → 下游分类
```

需要改成下面这条更准确的主线：

```text
定义不变的局部对象
    → 训练集内拟合表示变换（KSVD / PCA / prototype / token vocabulary）
    → 保留局部对象在分子内的分布或关系
    → 图级统计读出
    → 下游任务（MolHIV 分类；ZINC 回归）
    → 按固定 split 和冻结规则评估
```

原因是：8 月 29 日以后，确实继续做了 radius-2 和 KSVD 归因，但后续大量有效实验已经转向 **不使用 KSVD 的 invariant rooted-WL、typed motif、hierarchical backoff 和中心级条件模型**。因此不能把这些模型的收益都归因于 KSVD。

当前最重要的判断如下：

| 问题 | 当前结论 |
|---|---|
| 局部对象怎么定义？ | 必须先通过 node relabel audit；MolHIV 采用完整、all-centre、invariant radius-2 ego；ZINC 的主对象逐步扩大到 all-centre radius-3。 |
| 采样最重要的因素是什么？ | 不是简单增加中心数量，而是保留全中心并扩大有效上下文/节点对覆盖。ZINC 的 radius-3 比 radius-1/2 更有用。 |
| KSVD 是否带来下游增益？ | KSVD 能显著降低重构误差，但 `FINAL` 没有稳定优于 `RAW` 或 `INIT`；普通无监督 KSVD update 暂不能视为任务增益来源。 |
| MolHIV 的统计读出哪里有效？ | 对每个中心 patch 保留 population distribution，尤其是 `std`，明显优于只取一个 mean；跨中心 `cross_cov` 也有机制和迁移信号。 |
| MolHIV 当前性能参考是什么？ | 非 KSVD 的 `S+marginal` XGBoost 在 official-valid 五 seed 均值为 `0.8413`，ensemble 为 `0.8450`；这是统计路线结果，不是 KSVD 结果。 |
| ZINC 当前最有力的统计路线是什么？ | collision-free typed-WL count 之后，hierarchical exact-token backoff 达到 valid `0.3546`、refit test `0.3455` MAE；它关闭了 KSVD。 |
| ZINC 是否已经验证了中心条件关系？ | 中心级 `[s_v,a_v,s_v*a_v]` 条件融合通过了 shuffle 机制控制；但关系传播和多 seed 稳定性仍需补充。 |
| 当前最值得复核的结果是什么？ | ZINC exact radius-2 path-conditioned pooling 的 valid/test MAE 为 `0.1872/0.1381`，但目前是单 seed 神经模型，证据等级低于完整多次复核的 XGBoost 结果。 |

## 1. 数据集与任务

| 数据集 | 数据来源与划分 | 任务 | 指标 | 主要风险 |
|---|---|---|---|---|
| `ogbg-molhiv` | OGB official scaffold split；train/valid/test = `32901/4113/4113`；正例数约为 `1232/81/130` | HIV 活性二分类 | ROC-AUC，越高越好 | 正例少、scaffold shift 明显；official test 只有 130 个正例 |
| `ZINC(subset=True)` | PyG official split；train/valid/test = `10000/1000/1000` | 连续分子性质回归 | MAE，越低越好 | 不同切片的目标尺度不同，不能直接横比切片绝对 MAE |

这里的“MolHIV 下游分类”与“ZINC 下游回归”必须分开汇报。ZINC 不能沿用 MolHIV 的 AUC 语言，也不能把两个数据集的分数放进同一排行榜。

### 1.1 统一对象

对图 (G) 中的中心原子 (v)，半径为 (r) 的局部对象可写成：

```text
P(G,v,r) = 以 v 为中心、保留距离 ≤ r 节点的 induced ego graph
```

之后的关键选择包括：

- 中心是全部原子，还是每图只保留少量中心；
- patch 是 raw typed object、rooted-WL role，还是 KSVD code/reconstruction；
- 图级读出只保留均值，还是保留中心群体的方差、分位数、极值和关系；
- 是否保留不同中心之间的距离、重叠和条件对应关系。

## 2. 实验路线总览

| 时间 | 主要问题 | 主要结果 | 路线决定 |
|---|---|---|---|
| 8 月 29 日 | 导师固定特征路线能否用 typed patch + KSVD + XGBoost 复现 | 固定特征 `S` 可行；K-SVD reconstruction 误差下降，但分类不随之提高 | 不把普通 KSVD update 当成任务增益 |
| 8 月 30 日 | ZINC 的半径、长程对象、属性—结构绑定和 KSVD 归因 | radius-3 更有用；binding 可检测，但 relation/conditional readout 未稳定超过 raw | 保留 radius-3；暂停盲扫 K/T 和高维 joint |
| 8 月 31 日 | 旧 radius-2 patch 是否是有效图对象 | 发现 node-ID tie-break、patch 截断和类别 modulo 碰撞 | 旧 `0.8307` 结果降级为历史调试结果，重建 invariant object |
| 9 月 1 日 | 真正缺的是 KSVD，还是结构编码和 graph readout | rooted-WL 修复了 coarse role；mean → distribution、跨中心统计带来主要增量 | 重点转向 population distribution 和 cross-centre interaction |
| 9 月 2–3 日 | MolHIV 交互块是否能跨 split；ZINC 是否需要中心级条件模型 | MolHIV 的 `S+marginal` 是 valid 主线；ZINC 中心条件融合通过 shuffle；relation propagation 未稳定 | MolHIV 不再用 test 调参；ZINC 继续核查 typed/hierarchical/path route |
| 9 月 4 日 | ZINC typed token、层级回退、显式 pair/path readout | hierarchical backoff 优于普通 typed count；composition 和 conditioned match 未通过；path pooling 数值很强但仅单 seed | 主线采用 hierarchical backoff；path pooling 作为待复核候选 |

## 3. 采样：尝试过什么，以及采样问题如何收敛

### 3.1 MolHIV：从“少量 radius-2 patch”改成“完整 all-centre radius-2 对象”

#### 早期 radius-2 typed proxy

导师路线代理使用每个原子为中心的 radius-2 patch，尝试过：

- 每图最多 8 个中心；
- 保留所有原子中心；
- topology-only patch、atom/bond typed patch 和 typed-slot 统计；
- patch adjacency、atom histogram、bond histogram 等固定维度对象。

结果表明，少量中心会把一个分子的局部环境分布压得过粗；保留所有中心后，`mean/std` 等 population readout 才有机会表达“分子里有多少种局部环境、环境是否异质”。

但这版对象随后审计失败，不能继续做正式机制归因：

- 局部邻接的 tie-break 依赖 node ID，随机重标号后特征改变；
- `14.12%` patch 被截断，adjacency 与完整 atom/bond histogram 的范围不一致；
- atom category `%16` 造成 `16` 个真实类别碰撞；
- 冻结模型在重标号后单图预测最大漂移为 `0.3454`。

因此，历史 `S+R_raw≈0.8307` 只能作为调试阶段数字，不能作为“KSVD 或纯结构机制成立”的证据。

#### 修正后的对象

后续统一采用：

```text
完整 radius-2 induced ego
→ 每个原子都作为中心
→ topology-only rooted-WL role
→ strict atom/bond semantics
→ graph-level aggregation
```

修正后的 invariant object 通过了 topology、attribute 和 joint readout 的随机重标号审计。此后 exact topology certificate 主要用于碰撞和覆盖审计，不再简单用更高维 exact vocabulary 替代 rooted-WL 主通道。

### 3.2 ZINC：比较 radius-1/2/3、截断上限和多尺度

ZINC 长程 factorial 固定全部原子为中心，比较 radius-1、2、3，并对 radius-3 比较 `max_nodes=12` 和 `max_nodes=20`。

| 对象 | 平均 node-pair coverage | patch truncation | 固定参数下 `global+raw+KSVD-final` valid/test MAE |
|---|---:|---:|---:|
| radius-1 | `0.2391` | `0` | `0.5436 / 0.5674` |
| radius-2 | `0.4952` | `0.0001` | `0.5430 / 0.5599` |
| radius-3, cap=12 | `0.6787` | `0.0615` | **`0.5225 / 0.5400`** |
| radius-3, cap=20 | `0.6987` | `0` | `0.5256 / 0.5276` |

节点覆盖率在所有半径下都是 `1.0`，主要变化是节点对和局部上下文覆盖。把 cap 从 12 增到 20 基本消除了截断，但 valid 变化很小、test 略稳，说明**半径带来的上下文增量比消除截断更关键**。

多尺度 `radius-2 + radius-3` 在四个非重叠切片上都优于单尺度，但相对最佳单尺度的 MAE 改善为：

```text
0.01066 / 0.00549 / 0.01501 / 0.00471
```

预先规定的晋级门槛要求两片都至少改善 `0.01`，因此多尺度只记为弱互补，不进入 full 主线；大样本单尺度采用 radius-3。

#### 长程关系采样的尝试

为解释 radius-3 的增益，曾把 radius-2 局部对象之间的距离、位置和 atom-pair relation 单独统计。两个非重叠切片中，一片略有正增益，另一片反而下降；`true-shuffle` 也未稳定通过门槛。因此当前不能把 radius-3 增益简化为“图级远距离 atom pair 直方图”。

**采样结论：**

1. MolHIV 保留完整 all-centre radius-2 invariant object；
2. ZINC 保留 all-centre radius-3 作为单尺度主对象，cap=20 用于避免明显截断；
3. 不再默认把 radius-2 当作所有数据集的固定答案；
4. 下一步若研究长程，应改进“局部对象如何连接”，而不是继续堆简单 pair token。

## 4. 处理与表示：KSVD 以及后来替代 KSVD 的路线

### 4.1 KSVD 的实际作用

对训练 patch 矩阵 (Y)，共享 KSVD 字典学习可抽象为：

```text
Y ≈ D X
```

- (D)：跨图共享的 dictionary atoms；
- (X)：OMP 稀疏系数；
- `INIT`：初始字典或零次更新的稀疏重构；
- `FINAL`：经过若干次 K-SVD update 的稀疏重构；
- `RAW`：不经过字典压缩的原始局部对象统计。

正式实验中，字典和所有重构参数只用训练部分拟合。这里最重要的对照不是“重构误差是否下降”，而是：**下游指标是否随 K-SVD update 同步改善。**

### 4.2 MolHIV 的 KSVD 归因

#### 导师固定特征代理

导师真实的 69D/624D 上游 schema、payload 和 best parameters 没有在仓库中恢复，因此只能做 proxy。一个固定特征代理包含：

```text
S = 显式拓扑 + atom composition + bond composition
T/R = patch code 或 typed reconstruction readout
下游 = XGBoost(binary:logistic)
```

official-train/valid 结果如下（不同视图共用本阶段的 fixed feature protocol）：

| 视图 | valid ROC-AUC | 解释 |
|---|---:|---|
| `S` | `0.7817 ± 0.0066` | 显式拓扑与化学组成的强基线 |
| `S+T_init` | `0.7717 ± 0.0103` | 初始稀疏重构没有超出 `S` |
| `S+T_final` | `0.7624 ± 0.0090` | K-SVD update 后反而下降 |
| `R_raw` | `0.6649 ± 0.0057` | typed reconstruction proxy 的原始统计 |
| `R_init` / `R_final` | `0.6521 / 0.6542` | update 的任务增量接近于零 |
| `S+R_raw` / `S+R_final` | `0.7510 / 0.7477` | 都没有超过显式 `S` |

另一轮 train-only search 的结果中，`S+R_raw` 可以略高于 `S`，但 `R_final` 仍明显不占优。slot-aligned proxy 也显示 raw 比全局统计 proxy 好，但这支持的是“统计对象与对齐方式重要”，不是“KSVD update 有效”。

#### invariant reconstruction screen

在修正后的 invariant radius-2 joint patch 上，只使用 official-train 内部 scaffold folds 做机制筛选：

| 视图 | 内部三折平均 AUC |
|---|---:|
| `S+joint L2` | `0.6920` |
| `S+bilinear` | `0.7087` |
| `S+PCA reconstruction` | `0.7145` |
| `S+KSVD INIT` | **`0.7323`** |
| `S+KSVD FINAL` | `0.7184` |

`FINAL` 的重构误差比 `INIT` 更低，但分类 AUC 下降约 `1.39pt`。这说明当前有效的更像是**真实训练 patch prototype / 任务友好的稀疏坐标**，而不是继续优化无监督重构目标。

#### MolHIV KSVD 结论

- 共享字典、invariant patch 和稀疏 readout 可以携带结构信息；
- KSVD 是健康的 reconstruction learner；
- 但普通无监督 `FINAL` 没有稳定的 MolHIV label increment；
- `S+R_raw` 或 distribution readout 的收益不能直接命名为 KSVD 收益；
- 若要重新打开 KSVD，应另立 task-aligned dictionary protocol，不能继续在同一个 official validation 上盲扫 `K/T/Beam`。

### 4.3 ZINC 的 KSVD 归因

ZINC factorial 使用 train-only `K=24, T=3, 6 iterations`，同时比较 raw、INIT、FINAL。训练 patch 重构误差确实下降：

| radius | typed INIT | typed FINAL |
|---|---:|---:|
| 1 | `0.1414` | `0.0592` |
| 2 | `0.2728` | `0.1760` |
| 3 | `0.4267` | `0.3220` |
| 3-wide | `0.4329` | `0.3306` |

但任务指标没有同步改善：radius-3 standalone typed code 从 INIT `1.0330` 变为 FINAL `1.0613`，topology code 也出现类似现象；valid 和 test 对 INIT/FINAL 的偏好还会反转。

**ZINC 结论：** KSVD 当前定位为压缩器和 diagnostic side channel。后续 typed-WL count、typed match、hierarchical backoff、中心条件网络和 path pooling 均明确关闭 KSVD，以免把 token/readout 的收益混到字典更新上。

### 4.4 不使用 KSVD 的表示处理

8 月 31 日以后，表示处理逐渐转成以下几类：

| 处理方式 | 做法 | 作用 |
|---|---|---|
| rooted-WL role | 从 topology-only patch 迭代生成 root/shell/degree 角色 | 修复 `shell×degree×cycle` coarse role 过粗的问题 |
| exact topology certificate | 用 colored incidence graph certificate 做审计和条件分组 | 检查碰撞、覆盖和 OOV；不直接作为默认高维主通道 |
| typed-WL token | 对每个中心保留 0/1/2/3 层 typed token | 支持 motif count、match 和层级回退 |
| empirical prototype | 从训练 patch 中选真实 prototype，再做 prefix/response match | 检查“真实局部对象”是否比抽象方向更适合下游 |
| PCA/SVD | 只在训练部分拟合低秩坐标 | 作为交互块压缩和 scope sensitivity control |
| center-level encoder | 保留每个中心的结构和属性 token，再做条件融合 | 避免在 graph-level mean 前丢失中心对应关系 |

## 5. 统计读出：从 mean 到 population distribution 和条件关系

### 5.1 MolHIV：mean 不是充分读出

设一个图有中心 patch rows (z_{G,1},ldots,z_{G,n})。早期只做：

```text
graph feature = mean_v z(G,v)
```

后续恢复了中心群体的统计：

```text
mean / std / min / max / quantiles / RMS / abs-mean / nonzero rate / usage ...
```

在相同的 clean rooted-WL radius-2 对象上，结果是：

| 视图 | 维度 | official-valid AUC |
|---|---:|---:|
| `T+A mean` | 154 | `0.7945 ± 0.0033` |
| `T+A distribution` | 1793 | **`0.8172 ± 0.0028`** |
| `S+T+A mean` | 359 | `0.8022 ± 0.0020` |
| `S+T+A distribution` | 1998 | **`0.8276 ± 0.0043`** |

快速 block ablation 显示：

| readout block | valid AUC |
|---|---:|
| mean | `0.7936` |
| mean + std | **`0.8187`** |
| mean + min/max | `0.8020` |
| mean + q10–q90 | `0.7940` |
| mean + abs-mean/RMS/nonzero | `0.8075` |
| all 12 blocks | `0.8172` |

这里的核心不是“维度越多越好”，而是 `std` 保留了一个分子内部局部环境的异质性：一个分子可能同时包含少数特殊环境和大量普通环境，单一 mean 会把这种结构抹平。

### 5.2 MolHIV：结构—属性关系的三种层次

后续把融合拆成三种不同对象：

1. **marginal**：结构和属性各自的分布；
2. **within-patch binding**：同一 patch 内 topology role 与 attribute 的联合/中心化联合；
3. **cross-centre interaction**：同一分子不同中心环境之间的结构—属性共同变化。

对应的统计形式可以写成：

```text
binding = P(role, attribute) - P(role)P(attribute)
cross_cov = Cov_v(topology_row_v, attribute_row_v)
```

主要发现：

- rooted-WL 比 coarse role 明显好：内部三折 `T` 从 `0.6608` 提升到 `0.6941`，factorized raw 为 `0.7327`；
- exact orbit 位置绑定没有通过 matched shuffle，且 exact topology 长尾跨 scaffold 覆盖不足；
- patch-level topology–attribute pairing 可以被 shuffle control 检测，`true - patch-pair-shuffle` 约 `+2.36pt`，但没有稳定超过最强 marginal baseline；
- cross-centre covariance 在内部不重叠 scale-up 中相对 marginal 增加约 `0.0035`，与 binding 合并后相对 marginal 增加约 `0.0102`，但需要注意这是 train-only mechanism screen，不是最终 official-valid 榜单。

### 5.3 ZINC：marginal、typed count、match 和 hierarchical backoff

ZINC 上的读出从低维 marginal 逐步扩展为 typed token 的计数和层级匹配：

| 读出 | 含义 | 结果判断 |
|---|---|---|
| `S+marginal` | global structure/chemistry + radius-2/3 局部 mean/std/context | 稳定、低成本 baseline |
| typed motif count | 对 0/1/2/3 层 typed rooted-WL token 做 raw count 和按中心数归一化 | 明显优于低维 marginal |
| typed match | 训练集内频繁 token prototype，按同中心 prefix match 汇总 mean/max/response | 有效，但弱于 hierarchical backoff |
| hierarchical backoff | 高阶 exact token 未命中时回退到同中心的低阶 token | 当前最强 XGBoost 统计路线 |
| hierarchical composition | 在 backoff 上再加入 cross-level、cross-centre、rarity 和 molecule-composition | 在当前协议中反而变差 |
| conditional match gate | 用 topology 条件化 attribute prototype，再和 hierarchical base 拼接 | mechanism gate 失败，不进入 full |

hierarchical backoff 的关键不是单纯扩大维度，而是让高阶结构在 OOV/长尾时保留可解释的低阶信息：

```text
level-3 exact token
    → 同中心 level-2 token
    → level-1 token
    → level-0 token
```

这比直接把所有高阶 token 当作互不相关的 one-hot 更适合 ZINC 的长尾局部对象。

### 5.4 ZINC：条件关系的机制证据与性能边界

ZINC 的 center-level network 对每个中心保留结构 token (s_v) 和属性 token (a_v)，然后比较：

```text
center_concat       = [s_v, a_v]
conditional_fusion  = [s_v, a_v, s_v * a_v]
conditional_relation= conditional_fusion + centre-pair propagation
```

在官方 full split 的单 seed fast run 中：

| 模式 | valid MAE | refit test MAE |
|---|---:|---:|
| attribute-only | `0.4943` | `0.5308` |
| structure-only | `1.1191` | `1.1587` |
| center concat | `0.2695` | `0.2722` |
| conditional fusion | `0.2387` | **`0.2100`** |
| conditional relation | **`0.2193`** | `0.2382` |

随后固定预算的 shuffle confirmation 中，真实中心对应明显优于图内置乱：

| 模式 | valid MAE | refit test MAE |
|---|---:|---:|
| conditional fusion | `0.2163` | `0.2126` |
| attribute-row shuffle | `0.3268` | `0.3186` |

这证明“哪个属性属于哪个中心结构环境”确实有信息；但 `conditional_relation` 的关系传播收益在 valid/test 上没有稳定一致，暂时不继续堆关系层。

## 6. 下游结果：MolHIV 分类

### 6.1 Official-valid 与 controlled test

下表是中心交互路线冻结后的主对照。每个视图的参数先在 official-train scaffold folds 上选择；test 只在冻结后评估。test 为五个模型 seed 的均值，分别列出严格 train-only 和 train+valid refit 两种口径。

| 视图 | official-valid AUC | strict train-only test | train+valid refit test |
|---|---:|---:|---:|
| `S` | `0.7916` | `0.7432` | `0.7540` |
| `S+marginal` | **`0.8413`** | `0.7889` | `0.7853` |
| `S+cross_cov` | `0.8393` | `0.8009` | `0.7998` |
| `S+binding` | `0.8389` | `0.7897` | `0.7880` |
| `S+cross_cov+binding` | `0.8382` | **`0.8087`** | `0.7956` |

解释边界：

- **性能主线（按 official-valid 选择）**是 `S+marginal`，不是复杂 interaction；
- **机制主线**是跨中心 `cross_cov`，binding 更适合作为条件辅助块；
- strict test 中 `S+both` 较强，但 official test 只有 130 个正例，且更早路线已经查看过 MolHIV test，因此只能称 controlled terminal evaluation，不称 untouched test；
- PCA-16 的 `cross_cov` sensitivity 在 valid 为 `0.8372`、strict test 为 `0.8036`，没有证明它全面优于 PCA-8 或 marginal。

### 6.2 KSVD、XGBoost、MLP 和可学习融合的关系

做过的下游模型并不是同一层级的竞争：

- KSVD rich readout 主要用于 standalone attribution，回答“KSVD 是否比 random/PCA 有信息”；
- XGBoost 适合读取稀疏的 typed distribution、count 和 interaction blocks；
- MLP 作为 readout control，positive weight=10 时 `S+marginal` valid `0.8307`，test `0.7328`；center fusion valid `0.8271`，test `0.7414`，说明下游损失和 readout 会显著改变 valid→test 迁移；
- clean rooted-WL 的两个独立 XGBoost expert 做固定 50/50 late fusion，train+valid refit test 五 seed mean 为 `0.7825`，十模型 ensemble 为 `0.7839`；这支持误差互补，但仍不是 KSVD 增益；
- MolHIV center conditional neural fusion 的 valid/test 为 `0.8146/0.7019`，未解决 scaffold 泛化问题。

## 7. 下游结果：ZINC 回归

下面的结果来自不同表示/模型协议，不能把它们当作严格同预算排行榜；但可以用来说明路线如何逐步改善。

| 表示/模型 | KSVD | valid MAE | refit test MAE | 证据与状态 |
|---|---|---:|---:|---|
| `S` global baseline（r3 count 实验） | 否 | `0.5607` | `0.5924` | XGBoost；低维参考 |
| `S+marginal`，radius-3 typed-WL | 否 | `0.5444` | `0.5519` | XGBoost，20-trial；稳定 baseline |
| `S+WL-count`，0/1/2/3 typed token count | 否 | `0.3709` | `0.3765` | XGBoost，collision-free vocabulary；明显增益 |
| `S+typed_match` | 否 | `0.4100` | `0.3951` | XGBoost；优于 marginal，弱于 backoff |
| `S+hierarchical_backoff` | 否 | **`0.3546`** | **`0.3455`** | XGBoost，train-only vocabulary + lower-order fallback；当前强统计主线 |
| `S+hierarchical_composition` | 否 | `0.3786` | `0.4025` | 加入 composition/rarity 后变差，停止 |
| exact r2 patch direct relation + XGBoost | 否 | `0.5154` | `0.5460` | 显式 pair relation 不足以解释长程增益 |
| exact r2 patch direct relation + MLP | 否 | `0.4562` | `0.4544` | 可学习 readout；仍未与 backoff 同协议比较 |
| centre conditional fusion network | 否 | `0.2163` | `0.2126` | 单一网络协议；shuffle confirmation 通过 |
| centre conditional relation fast | 否 | `0.2193` | `0.2382` | 单 seed fast；关系传播不稳定 |
| exact r2 path-conditioned pooling | 否 | `0.1872` | **`0.1381`** | 单 seed、约 323k 参数；数值很强，必须多 seed/独立复核 |

其中最后三行不是 XGBoost，同样不能直接和 `0.3546` 做严格优劣宣判。path pooling 目前只是“值得复现的候选”，还没有完成多 seed、不同 split 和独立实现核对。

### 7.1 ZINC 机制筛选的失败也有价值

为了判断 typed statistics 的增益来自哪里，做过以下控制：

- radius-2 joint v1：true 相对 shuffle 稳定更好，但没有稳定超过 typed raw；
- conditional joint v2：两个大切片都检测到 binding，但相对 typed raw 为 `-0.02189/+0.00797`，未过 `+0.01` 晋级门槛；
- centered residual + SVD16：true 相对 shuffle 有间隔，但两片相对 typed raw 为 `+0.02567/-0.00155`，停止；
- hierarchical topology-conditioned match：三折 true-match MAE `0.4017`，shuffle `0.4024`，true-shuffle 仅 `0.0007`，且三折都没有超过 hierarchical base，gate 失败；
- hierarchical composition：增加更多跨层、跨中心、稀有度和分子组成特征后 valid/test 反而恶化。

因此当前应把“binding 存在”和“binding 形成稳定 target increment”严格分开：前者得到机制支持，后者尚未成立。

## 8. 评估协议：结果应该怎样解读

### 8.1 数据泄漏边界

所有正式路线遵守以下原则：

1. 字典、PCA/SVD、token vocabulary、prototype bank、standardizer 只在训练范围拟合；
2. MolHIV 的下游超参数在 official-train 的 scaffold folds 内选择，再看 official-valid；
3. ZINC 的 XGBoost 超参数通常在 official-train 内三折 shuffled KFold 选择，再评估 official-valid；
4. test 只在视图、参数、seed 和 refit 规则冻结后做终端核对；
5. 机制 screen、smoke、fast run 只能决定是否晋级，不能冒充最终 leaderboard。

### 8.2 MolHIV 的特殊说明

MolHIV official test 在更早的实验路线中已经被查看过，因此本整理中的 test 结果全部标记为 **controlled terminal evaluation**。它们仍然可以回答“冻结后的方案在 test 上如何表现”，但不能声称 test 完全 untouched，也不能根据 test 排名回改 view、PCA rank、融合权重或超参数。

### 8.3 ZINC 的特殊说明

ZINC 的 valid/test 结果大多采用：

```text
train-only vocabulary/feature fitting
→ official-valid 选择或报告
→ 冻结后 train+valid refit
→ official-test
```

ZINC 不同实验的下游模型差异很大：固定参数 XGBoost、20-trial XGBoost、单 seed MLP、单 seed center network 和单 seed path pooling 不能只按一个 MAE 表排序。尤其是 path pooling 的 `0.1381` 需要先完成重复实验，才能成为主结果。

### 8.4 旧结果与当前结果的边界

- 旧 radius-2 typed-slot `0.8307` 因 invariance、截断和类别碰撞问题降级；
- 导师真实 69D/624D schema 尚未恢复，当前同维度实现只能称 proxy；
- MolHIV `0.8413` 是 clean non-KSVD `S+marginal` XGBoost 结果；
- ZINC `0.3546` 是 non-KSVD hierarchical backoff XGBoost 结果；
- 任何这些数字都不能反写成“KSVD 已经带来同样增益”。

## 9. 当前推荐的两条可复现 pipeline

### 9.1 MolHIV 推荐主线

```text
完整 all-centre invariant radius-2 ego
    → rooted-WL topology + strict chemistry
    → 每个中心形成 local row
    → mean/std/all12 population distribution
    → 加 global S statistics
    → XGBoost(binary:logistic)
```

当前定位：

- `S+marginal` 是 official-valid 性能参考；
- `cross_cov` 是最值得在新 outer scaffold splits 上复核的交互块；
- binding 和 late fusion 作为机制/误差互补候选；
- ordinary unsupervised KSVD 只保留为 raw/INIT/FINAL compressor diagnostic。

### 9.2 ZINC 推荐主线

```text
all-centre radius-3 typed local object
    → collision-free typed-WL token vocabulary（train-only）
    → exact token count + same-centre lower-order hierarchical backoff
    → XGBoost(reg:absoluteerror)
```

当前定位：

- radius-3 是单尺度采样 baseline；
- hierarchical backoff 是当前较完整、可审计的统计路线；
- center conditional fusion 和 path pooling 是可学习候选，但需要多 seed 复核；
- hierarchical composition、conditioned match 和简单 long-range pair relation 暂停；
- ordinary KSVD 不进入 ZINC 当前性能主线。

## 10. 后续工作与停止项

### 可以继续

1. MolHIV：在新的 repeated scaffold outer splits 上复核 `S+marginal`、`cross_cov` 和受约束 late fusion；不再使用当前 official test 调参。
2. ZINC：对 path-conditioned pooling 做多 seed、独立 split 和参数冻结复核；同时与 hierarchical backoff 做同口径比较。
3. 若必须恢复 KSVD：先取得导师真实 69D/624D schema、patch payload、聚合定义和 best parameters，再做 exact feature audit。

### 暂停或关闭

- 在同一 validation 上继续扫 MolHIV 的 `K/T/Beam`、KSVD iteration、attention 和 router；
- 把 ZINC 的简单 pair token、high-dimensional conditional histogram 或 composition block 继续加宽；
- 用 MolHIV test 的排名反向选择模型；
- 把 valid/test 上的非 KSVD 强结果写成 KSVD 贡献。

## 11. 主要结果来源

- 导师固定特征与 KSVD 代理：[`MENTOR_CONCEPT_ROUTE_VERDICT_20260829.md`](../../tracks/ksvd/results/luyin16/MENTOR_CONCEPT_ROUTE_VERDICT_20260829.md)、[`MENTOR_FUSED_PROXY_SEARCH_20260829.md`](../../tracks/ksvd/results/luyin16/MENTOR_FUSED_PROXY_SEARCH_20260829.md)
- patch 审计与 invariant 修正：[`PATCH_OBJECT_AUDIT_20260831.md`](../../tracks/ksvd/results/luyin16/PATCH_OBJECT_AUDIT_20260831.md)、[`INVARIANT_PATCH_MECHANISM_SCREEN_20260831.md`](../../tracks/ksvd/results/luyin16/INVARIANT_PATCH_MECHANISM_SCREEN_20260831.md)
- MolHIV 读出诊断：[`READOUT_DIAGNOSIS_20260901.md`](../../tracks/ksvd/results/luyin16/READOUT_DIAGNOSIS_20260901.md)
- MolHIV 结构—属性与中心交互：[`CLEAN_STRUCTURAL_ROLE_FUSION_20260901.md`](../../tracks/ksvd/results/luyin16/CLEAN_STRUCTURAL_ROLE_FUSION_20260901.md)、[`CROSS_CENTER_INTERACTION_OFFICIAL_VALID_20260902.md`](../../tracks/ksvd/results/luyin16/CROSS_CENTER_INTERACTION_OFFICIAL_VALID_20260902.md)、[`CROSS_CENTER_INTERACTION_OFFICIAL_TEST_20260902.md`](../../tracks/ksvd/results/luyin16/CROSS_CENTER_INTERACTION_OFFICIAL_TEST_20260902.md)
- ZINC radius/KSVD factorial：[`ZINC_LONG_RANGE_FACTORIAL_20260830.md`](../../tracks/ksvd/results/luyin16/ZINC_LONG_RANGE_FACTORIAL_20260830.md)、[`ZINC_MULTISCALE_RADIUS_20260830.md`](../../tracks/ksvd/results/luyin16/ZINC_MULTISCALE_RADIUS_20260830.md)
- ZINC 机制筛选：[`ZINC_MECHANISM_ROUTE_SYNTHESIS_20260830.md`](../../tracks/ksvd/results/luyin16/ZINC_MECHANISM_ROUTE_SYNTHESIS_20260830.md)、[`ZINC_LONG_RANGE_OBJECT_RELATION_20260830.md`](../../tracks/ksvd/results/luyin16/ZINC_LONG_RANGE_OBJECT_RELATION_20260830.md)
- ZINC typed/hierarchical 主线：[`ZINC_STEP_A_MOTIF_COUNT_20260903.md`](../../tracks/ksvd/results/luyin16/ZINC_STEP_A_MOTIF_COUNT_20260903.md)、[`ZINC_TYPED_MATCH_20260904.md`](../../tracks/ksvd/results/luyin16/ZINC_TYPED_MATCH_20260904.md)、[`ZINC_HIERARCHICAL_BACKOFF_20260904.md`](../../tracks/ksvd/results/luyin16/ZINC_HIERARCHICAL_BACKOFF_20260904.md)、[`ZINC_HIERARCHICAL_COMPOSITION_20260904.md`](../../tracks/ksvd/results/luyin16/ZINC_HIERARCHICAL_COMPOSITION_20260904.md)
- ZINC 中心条件与 path pooling：[`ZINC_CONDITIONAL_FUSION_SHUFFLE_20260903.md`](../../tracks/ksvd/results/luyin16/ZINC_CONDITIONAL_FUSION_SHUFFLE_20260903.md)、[`ZINC_CENTER_RELATION_NETWORK_OFFICIAL_FAST_20260903.md`](../../tracks/ksvd/results/luyin16/ZINC_CENTER_RELATION_NETWORK_OFFICIAL_FAST_20260903.md)、[`ZINC_PATCH_PATH_POOLING_20260904.json`](../../tracks/ksvd/results/luyin16/ZINC_PATCH_PATH_POOLING_20260904.json)

> 时间核对：工作区另有一份文件名为 `ZINC_CONDITIONAL_STATISTICS_20260906.md` 的结果。由于本整理截止日为 2026-09-04，且文件名日期超出当前日期，该结果暂不纳入主结论，待确认实际实验日期后再补入。
