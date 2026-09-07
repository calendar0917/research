# `tracks/ksvd/` 实验路线总结

## 核心问题

图的结构信息和节点/边属性信息能否分别表示、再有意义地融合？K-SVD 在其中到底是任务表征学习器，还是仅做压缩？

围绕这个问题，我们在 MolHIV 和 ZINC 两个数据集上展开了一系列方案。以下按方案组织，不按日期拼凑。

## 基本流程与数据集

```text
局部采样（radius-2/3）
→ 可选处理（K-SVD 字典学习）
→ 统计读出（mean / std / 分布）
→ 下游模型（XGBoost / MLP）
```

- **MolHIV**：分子二分类，预测 HIV 活性。官方骨架划分，train 32901 / valid 4113 / test 4113。指标为 ROC-AUC，越高越好。valid 正样本仅约 81 个，0.01-0.02 的 AUC 差异不能过度解释。test 非未被接触的干净测试集，而是冻结后的受控终端评估。
- **ZINC**：分子性质回归，目标是 penalized logP。官方 `subset=True` 划分 10000/1000/1000。指标为 MAE，越低越好。

评估协议：字典、词表、PCA 等无监督变换只在 train 上拟合；用 official-train 内部骨架交叉验证做调参和机制筛选；冻结后评估 official-valid，部分实验再用 train+valid 重训后检查 test。`5000/500` 切片和 smoke 只用于方向判断，不作正式结论。

### 关键术语速查

- **patch**：以某个原子为中心、按 radius 扩展的局部子图。
- **typed**：把原子类型、键类型编进 patch 向量。
- **rooted**：以中心原子为根，给邻居分配规范化槽位，使 patch 不依赖节点编号顺序。
- **WL token（结构角色 token）**：用 Weisfeiler-Lehman 算法对 patch 做多轮哈希，生成结构角色编码。每轮迭代信息更细，round-0 最粗，round-3 最细。
- **typed-WL token**：原子类型 + 键类型 + WL 结构角色组合成的精确 token，每个 token 唯一对应一种局部结构-属性模式。
- **K-SVD INIT / FINAL**：INIT 是字典初始化，FINAL 是完成 K-SVD 更新迭代后的字典。
- **marginal（边际统计）**：对结构角色 T 和属性 A 分别统计，不保留二者的联合关系。
- **binding**：同一 patch 内 `P(T,A) - P(T)×P(A)`，衡量局部环境里结构和属性是否偏离独立。
- **cross_cov（跨中心协方差）**：同一分子中各局部环境的 T/A 行向量之间的协方差，衡量分子内部环境间的联合变化。
- **readout**：把一个分子里所有中心 patch 的向量汇总成一个图级向量的操作。

---

## 方案一：显式统计基线（S + XGBoost）

```text
对每个分子提取全局拓扑统计与化学组成统计
→ 拼成向量 S（拓扑 18 维 + 原子组成 174 维 + 键组成 13 维 = 205 维）
→ 输入 XGBoost
```

**结果（MolHIV official-valid）：**

| 特征块 | 维度 | valid AUC |
|---|---:|---:|
| 纯拓扑 | 18 | 0.6831 |
| 原子组成 | 174 | 0.7066 |
| 键组成 | 13 | 0.5993 |
| 原子+键组成 | 187 | 0.7196 |
| 拓扑+化学（即 S） | 205 | **0.7817** |

**说明：** 强信号来自拓扑与化学组成的互补，不是任何一方单独。S 是有效的非 GNN 基线，后续所有方案都以它为比较锚点。这一步排除了"纯结构特征就能达到 0.78"的误读。

结果入口：[`MENTOR_CONCEPT_ROUTE_VERDICT_20260829.md`](./MENTOR_CONCEPT_ROUTE_VERDICT_20260829.md)、[`PURE_STRUCTURAL_FUSION_20260829.md`](./PURE_STRUCTURAL_FUSION_20260829.md)

---

## 方案二：K-SVD 字典稀疏码

```text
每个节点 radius-2 采样
→ 对 patch 展平，训练 K-SVD 字典、记录稀疏系数
→ 对系数做统计（mean 等）
→ 输入 XGBoost
```

**结果（MolHIV official-valid）：**

| 视角 | valid AUC |
|---|---:|
| 结构 token，字典初始化 | 0.6718 |
| 结构 token，字典最终更新 | 0.6521 |
| S + 结构 token（初始化） | 0.7717 |
| S + 结构 token（最终更新） | 0.7624 |

K-SVD 把平均重建误差从 0.1732 降到 0.1204，但分类反而变差。纯拓扑代理同样如此：结构统计 0.7723，加上最终重建结果后降到 0.7367。

**结果（ZINC official，radius-3 K=24 T=3）：**

| 视角 | valid MAE | test MAE |
|---|---:|---:|
| 全局统计+属性 | 0.6122 | 0.6382 |
| radius-3 + 原始/FINAL | 0.5225 | 0.5400 |

K-SVD 重建误差持续下降（初始化 0.4267 → 最终 0.3220），但任务 MAE 不同步；有时初始化版本在 test 上反而更好。

在方案七的框架中同样如此：原始 patch 为 valid 0.1872 / test 0.1381，K-SVD K=64 T=4 压缩后降到 valid 0.2582 / test 0.2439。

**说明：** 这是最一致也最重要的负结果。在所有阶段、两个数据集上，K-SVD 的无监督重建目标与下游任务目标都不一致——重建误差下降不等于分类/回归变好。K-SVD 当前更像压缩器，而非任务表征学习器。如果未来重新引入字典学习，必须引入任务对齐目标，这本身已是一个新的研究命题。

结果入口：[`MENTOR_CONCEPT_ROUTE_VERDICT_20260829.md`](./MENTOR_CONCEPT_ROUTE_VERDICT_20260829.md)、[`ZINC_LONG_RANGE_FACTORIAL_20260830.md`](./ZINC_LONG_RANGE_FACTORIAL_20260830.md)、[`ZINC_KSVD_EXACT_PATCH_PATH_POOLING_20260905.md`](./ZINC_KSVD_EXACT_PATCH_PATH_POOLING_20260905.md)

---

## 方案三：分布读出

```text
每个节点 radius-2 采样（修正后的置换不变 patch 对象）
→ 对所有中心 patch 的每一维求 mean / std / 分位数 / min-max 等 12 种统计量
→ 拼成图级向量
→ 输入 XGBoost
```

**前置修正：patch 对象审计。** 旧对象存在三个问题：用节点编号作 tie-break 导致重标号后 patch 顺序变化；邻接表只保留 8 个节点但属性统计用了完整 ego，范围不一致；原子类别取模造成 55 个原始类碰撞到 16 个桶。旧对象上的历史 0.8307 因此失效。修正后使用完整 radius-2 ego、直接 OGB 原始类别、rooted 不变对象，重标号漂移归零。修正后的对象成为后续所有 MolHIV 机制实验的基础。

**结果（MolHIV official-valid）：**

| 读出方式 | 维度 | valid AUC |
|---|---:|---:|
| 仅 mean | 154 | 0.7936 |
| mean + std | 308 | **0.8187** |
| mean + min/max | 462 | 0.8020 |
| mean + 全部 12 块统计 | 1793 | 0.8172 |

| 视角 | 维度 | valid AUC | test AUC (重训) |
|---|---:|---:|---:|
| S + 结构+属性 mean | 359 | 0.8128 | 0.7697 |
| S + 结构+属性 mean+std | 513 | 0.8266 | 0.7723 |
| S + 结构+属性 全部12块 | 1998 | **0.8341** | 0.7779 |

**说明：** 这是路线的一个关键转折。之前的读出把所有中心 patch 压成一个均值，会丢失分子内部的局部环境异质性——"局部环境均匀"和"少数特殊环境 + 大量普通环境"均值相同但性质可能不同。保留分布（尤其 std）带来显著增益。这说明主要瓶颈一度不是融合算子不够深，而是过早丢掉了局部环境分布。

结果入口：[`PATCH_OBJECT_AUDIT_20260831.md`](./PATCH_OBJECT_AUDIT_20260831.md)、[`READOUT_DIAGNOSIS_20260901.md`](./READOUT_DIAGNOSIS_20260901.md)

---

## 方案四：结构-属性交互统计

```text
在同一个不变 patch 内定义结构角色 T 与属性 A
→ 联合统计 P(T,A)，中心化 binding = P(T,A) - P(T)P(A)
→ 跨中心协方差 cross_cov：各局部环境的 T/A 行如何共同变化
→ 与 S + 边际统计 拼接，经 PCA 降维后输入 XGBoost
```

binding 和 cross_cov 是两个不同层次：binding 衡量同一局部环境内 T 和 A 是否对应，cross_cov 衡量同一分子中各局部环境的 T/A 是否共同变化。MolHIV 的交互块原始维度约 11045 维，经 PCA 降到 8 维；下面的 ZINC 版本分别对 binding 和 cross_cov 做 train-only PCA 降到 16 维。

**ZINC 上的早期平行检测：** ZINC 上此前做过 within-patch binding 检测（joint v1、conditional joint v2、centered residual SVD16），都只在小切片（5000/500）上运行。binding 信号可检测（true 稳定优于 shuffle），但均未超过 typed raw 的门槛，因此当时没有进入 full pipeline，也没有继续做 cross_cov；下面补充的是后续独立的官方 full、无 WL 直接结构版。

**ZINC official full 的直接结构版确认（无 WL）：** 为检验上述机制是否必须依赖 WL token，补做了一个官方 `10000/1000/1000`、seed=0 的 full pipeline。每个原子中心取 radius-3 induced ego patch；结构角色 `T` 直接由节点 `(shell, induced degree, cycle)` 和边 `(无序 shell pair, cycle)` 定义，属性 `A` 由中心原子 one-hot、patch 原子直方图、incident-bond 直方图和 patch bond 直方图定义。binding 为同一 patch 内的 `P(T,A)-P(T)P(A)`，cross_cov 为各中心的 `T/A` 行之间的协方差。只用 train 拟合 PCA 和调参，3-fold train-only Optuna 选模型；valid 使用 official-train 拟合，test 使用冻结参数在 train+valid 上重训。整个版本不使用 WL、K-SVD、GNN 或 attention。

| 视角 | 维度 | train CV MAE | valid MAE | test MAE（train+valid 重训） |
|---|---:|---:|---:|---:|
| `S` | 62 | 0.563881 | 0.559099 | 0.592362 |
| `S + direct-marginal` | 395 | 0.470485 | 0.460278 | 0.445707 |
| `S + cross_cov` | 411 | 0.453161 | 0.441877 | 0.435351 |
| `S + binding` | 411 | 0.449619 | **0.440913** | **0.413400** |
| `S + cross_cov + binding` | 427 | **0.444800** | 0.448394 | 0.422868 |

这是一个 seed 的结果：交叉验证选择 `S + cross_cov + binding`，但单独的 binding 视角在 official-valid 和 train+valid refit test 上更好；相对 `S + direct-marginal`，`S + binding` 的 valid/test 改善分别为 0.019365/0.032307，`S + cross_cov` 分别改善 0.018401/0.010356。结构重标号不变性审计通过，最大漂移为 `2.38e-7`。因此，直接统计结构角色也能形成有效的结构-属性交互，且在这个 ZINC seed 上不需要 WL；但组合项的排序仍有 split/选择波动，不能据此宣称已得到稳定最终模型。

**结果（MolHIV 内部骨架交叉验证）：**

| 视角 | mean AUC |
|---|---:|
| S + 边际统计 | 0.7098 |
| S + 跨中心协方差 | 0.7133 |
| S + binding | 0.7148 |
| S + 两者 | **0.7200** |

S + 两者相对边际统计 +0.0102，3/3 折均胜；相对匹配双置乱 +0.0102，3/3 折均胜。这是机制阶段最重要的正结果。

**结果（MolHIV official-valid）：**

| 视角 | 骨架交叉验证 | valid AUC |
|---|---:|---:|
| S + 边际统计 | 0.7871 | **0.8413** |
| S + 跨中心协方差 | 0.7900 | 0.8393 |
| S + binding | 0.7888 | 0.8389 |
| S + 两者 | **0.7907** | 0.8382 |

冻结 test 中，cross_cov 约 0.800，S+两者在严格 train-only 下为 0.8087，五模型集成 0.8135；但 train+valid 重训且 PCA 重拟合后 S+两者降到 0.7956，交互块对 PCA 坐标系和骨架偏移敏感。

另外测试了 cross-attention、FiLM、MoE 等可学习融合，在当前对象和预算下都没有超过简单边际统计，甚至干扰属性分支。低容量 norm/bilinear gate 也没有超过 S+两者。

**说明：** 需要严格区分两件事：能检测到结构和属性不独立，与该依赖能稳定提高新分子预测。前者有较多证据（多个 shuffle 实验都显示真实优于置乱）；后者在 MolHIV 上尚未完全成立，而 ZINC 的直接结构 full pipeline 在单 seed 上给出了正向确认（尤其是 `S + binding` 的 test MAE 0.413400）。跨中心协方差和 binding 都值得保留，但需要更多 seed/切分验证，当前不能把其中任一项表述为稳定最终主模型。

结果入口：[`ROLE_ATTRIBUTE_BINDING_20260901.md`](./ROLE_ATTRIBUTE_BINDING_20260901.md)、[`CROSS_CENTER_INTERACTION_ROUTE_VERDICT_20260902.md`](./CROSS_CENTER_INTERACTION_ROUTE_VERDICT_20260902.md)、[`CROSS_CENTER_INTERACTION_OFFICIAL_VALID_20260902.md`](./CROSS_CENTER_INTERACTION_OFFICIAL_VALID_20260902.md)、[`ZINC_DIRECT_BINDING_CROSS_COV_20260905.md`](./ZINC_DIRECT_BINDING_CROSS_COV_20260905.md)

---

## 方案五：高维 typed motif 与层级回退

```text
每个节点 radius-3 采样
→ 计算 typed-WL token（原子类型 + 键类型 + WL 结构角色，精确无碰撞）
→ 对 train 中出现的 token 做计数统计
→ 对 test 中未出现的 token，层级回退到低阶 token
→ 输入 XGBoost
```

先看 radius 的影响（ZINC official，Optuna 调参后）：

| radius | valid MAE | test MAE (重训) |
|---|---:|---:|
| radius-1 | 0.5151 | 0.5475 |
| radius-2 | 0.5139 | 0.5478 |
| radius-3 | **0.4953** | 0.5271 |
| radius-3 宽截断 | 0.4967 | **0.5080** |

radius-3 有明确正信号，但"只调大 radius"不能解释全部差距，最好 MAE 仍约 0.50。节点对覆盖率：radius-1 0.239、radius-2 0.495、radius-3 0.679。把最大节点数从 12 提到 20 后截断率接近 0，但 valid 基本不变，说明增量来自感受野扩大而非截断减少。

引入高维 typed motif 后（ZINC official）：

| 视角 | 维度 | valid MAE | test MAE |
|---|---:|---:|---:|
| S（radius-3） | 62 | 0.5607 | 0.5924 |
| WL token 计数 | 16392 | 0.4104 | 0.4293 |
| S + WL token 计数 | 16454 | **0.3709** | 0.3765 |
| S + 层级回退 | 41048 | **0.3546** | **0.3455** |

层级回退的含义：当高阶 WL token（如 round-3）在训练词表中未出现时，不直接标为 OOV 丢弃，而是回退到低阶 round（round-3→round-2→round-1→round-0），保留信息。回退后的总维度约 41048 维。

继续加入跨层、跨中心、稀有度等特征反而变差：valid 0.3786、test 0.4025。typed match（前缀匹配 + 响应摘要）为 valid 0.4100 / test 0.3951，不如层级回退；拓扑条件门控相对 shuffle 仅好 0.0007，未通过门槛。

**说明：** 高维 typed motif 比低维边际统计强很多，说明 ZINC 需要更细的局部结构/属性模式。层级回退解决了部分长尾 token 和跨 split 迁移问题。但信息越多不等于越好——过多高维统计引入冗余和分布敏感性。

结果入口：[`ZINC_LONG_RANGE_FACTORIAL_20260830.md`](./ZINC_LONG_RANGE_FACTORIAL_20260830.md)、[`ZINC_STEP_A_MOTIF_COUNT_20260903.md`](./ZINC_STEP_A_MOTIF_COUNT_20260903.md)、[`ZINC_HIERARCHICAL_BACKOFF_20260904.md`](./ZINC_HIERARCHICAL_BACKOFF_20260904.md)、[`ZINC_TYPED_MATCH_20260904.md`](./ZINC_TYPED_MATCH_20260904.md)

---

## 方案六：中心条件融合与条件统计

这个方案的核心问题是：**不要只看边际统计，要保留"结构角色如何调节属性分布"。**

之前方案三的分布读出对每个 patch 统计 mean/std，但结构角色和属性是分别统计的（边际统计），没有保留"给定某种结构角色，属性是什么分布"这层条件关系。方案六把这层关系显式建出来。

### 6a 条件融合（MLP 版）

```text
对每个中心节点：
→ 确定结构角色 r（通过 WL 得到）
→ 把结构角色向量与对齐属性向量做乘法交互
→ 让结构角色直接调节属性权重
→ 小型 MLP 输出
```

这是在同一个中心内，让结构信息调制属性权重，而不是简单拼接。shuffle 对照组保留融合结构但打乱同中心的属性对应关系——如果融合明显优于 shuffle，说明"同一中心的结构-属性对应"确实有信号。

**结果（ZINC official，radius-3）：**

| 视角 | valid MAE | test MAE |
|---|---:|---:|
| 条件融合 | **0.2163** | **0.2126** |
| 同中心属性 shuffle | 0.3268 | 0.3186 |

条件融合比 shuffle 好约 0.11 MAE，说明中心条件对应关系有真实信号。但关系传播（在中心间通过显式距离/重叠做消息传递）为 0.2193 / 0.2382，没有稳定优于条件融合，增益不能归因于消息传递。

MolHIV 上同类条件融合为 valid 0.8146 / test 0.7019，不能迁移到骨架测试集。

### 6b 条件统计（可分解版）

```text
对每个中心节点：
→ 统计 E[1{结构角色=r} × 对齐属性向量]
→ 不构造精确联合 token（避免高维稀疏），只保留一阶条件关系
→ 多尺度版本在 WL round 0/1/2/3 四个层级都做条件统计并拼接
→ 输入 XGBoost
```

**结果（ZINC official）：**

| 视角 | 维度 | valid MAE | test MAE |
|---|---:|---:|---:|
| 分解式条件统计 | 526 | 0.5011 | 0.5115 |
| 条件统计 radius-3 | 4622 | 0.4841 | 0.5029 |
| 条件统计多尺度 | 16910 | **0.4488** | **0.4713** |

**说明：** 把"结构角色下的属性分布"保留下来比单纯增加全局标量更有效。多尺度（同时保留粗粒度和细粒度的角色-属性条件关系）优于单层。但维度和稀疏性增长很快（526→4622→16910），仍存在过拟合和跨 split 稳定性问题。条件融合（MLP，0.2163）明显优于条件统计（XGBoost，0.4488），说明这层关系用可学习交互比手工统计更有效。ZINC 上有效的中心条件模型不能直接迁移到 MolHIV 骨架测试。

结果入口：[`ZINC_CONDITIONAL_FUSION_SHUFFLE_20260903.md`](./ZINC_CONDITIONAL_FUSION_SHUFFLE_20260903.md)、[`ZINC_CONDITIONAL_STATISTICS_20260906.md`](./ZINC_CONDITIONAL_STATISTICS_20260906.md)、[`MOLHIV_CENTER_CONDITIONAL_FUSION_20260903.md`](./MOLHIV_CENTER_CONDITIONAL_FUSION_20260903.md)

---

## 方案七：精确 patch 路径 + MLP

这个方案的核心问题是：**能否保留精确的局部结构身份和中心间关系，但不做 GNN 消息传递？**

方案五的 typed-WL token 虽然精确，但用计数统计送入 XGBoost；方案六的条件融合用 MLP 但没有显式建模中心间关系。方案七把两者结合：精确 patch 编码 + 显式中心对关系 + MLP。

```text
对每个中心 patch：
→ 生成精确 patch 描述符（146 维，编码每层的原子/键组成、大小、环等）
→ 过 MLP 编码成 embedding

对所有中心对（pair）：
→ 用关系描述符（23 维：距离、重叠、是否同根、是否同 patch）
→ 按最短路径距离分桶（6 个距离区间）
→ 每桶内做置换不变池化（sum / sum-of-squares / log-count）

→ 一个 MLP head 输出图级预测
```

关键设计：不靠消息传递迭代传播信息，而是通过显式的中心对关系 + 距离分桶池化直接建模跨中心关系。保留显式结构关系，但不做 GNN 式隐式消息传递。

**结果（ZINC official）：**

| 变体 | 说明 | 参数量 | valid MAE | test MAE |
|---|---|---:|---:|---:|
| 精确 patch 路径 原始 | 146 维→隐层 | 323,009 | 0.1872 | **0.1381** |
| 因式分解 rank-20 | 146→20 维瓶颈→隐层，低秩正则化 | 228,861 | **0.1759** | 0.1427 |
| CIN 宽度 | 不同宽度配置 | — | 0.1802 | 0.2011 |
| 混合 | 多配置混合 | — | 0.1839 | 0.1385 |
| K-SVD 压缩 | K=64 T=4 | — | 0.2582 | 0.2439 |
| 层级关系上下文（最新） | 146 维→隐层 +1 次零初始化中心—上下文关系更新，无消息传递 | 277,053 | 0.1816 | **0.1346** |

因式分解 rank-20 的含义：把 146 维 patch 描述符先压到 20 维瓶颈再展开，参数减少 30%，valid 更好但 test 略降——低秩正则化在 valid 上有帮助但没迁移到 test。

**结果（MolHIV official）：**

| 变体 | valid AUC | test AUC |
|---|---:|---:|
| 精确 patch 路径 MLP | 0.8028 | 0.7852 |
| K-SVD patch 路径 MLP | 0.8290 | 0.7803 |
| 壳层矩（按距离分层统计） | 0.8225 | 0.7879 |
| 壳层分布 | 0.7969 | **0.7941** |

**说明：** 这条路线在 ZINC 上把差距从约 0.5 推进到 0.14-0.18，是当前最有潜力的显式路线。K-SVD 压缩（0.2582 / 0.2439）明显不如原始，再次确认压缩损失任务有用的身份信息。但 valid/test 的最佳变体不一致，不能把多次搜索中的最优单点当作稳定最终结果。MolHIV 上 K-SVD patch 路径 valid 有提升但 test 没有超过原始，不同读出排序也不稳定。最后一行的"层级关系上下文"是 0905 晚间的后续实验：在精确 patch 路径的池化基础上，保留一次零初始化、可训练的中心—上下文关系更新（仍无消息传递），test 提升到 **0.1346**，是当前 ZINC 最佳数字；它仍是单次运行，尚无种重复验证。

结果入口：[`ZINC_PATCH_PATH_POOLING_20260904.md`](./ZINC_PATCH_PATH_POOLING_20260904.md)、[`ZINC_PATCH_PATH_POOLING_FACTORIZED_RANK20_20260905.md`](./ZINC_PATCH_PATH_POOLING_FACTORIZED_RANK20_20260905.md)、[`MOLHIV_PATCH_PATH_POOLING_20260904.md`](./MOLHIV_PATCH_PATH_POOLING_20260904.md)

---

## 方案八：统一关系 patch 模型

这个方案的核心问题是：**能否把前面所有有效机制塞进一个统一框架，同时解决两个数据集？**

此前 ZINC 和 MolHIV 各用一套模型。方案八把精确 token、层级回退、K-SVD 初始化、单 patch 矩、跨中心协方差、距离分桶中心对池化等机制全部融合进一个统一 encoder。

```text
patch encoder：输入结构+属性描述符（ZINC 840 维 / MolHIV 797 维）
→ 精确 token + 层级回退 + K-SVD 初始化的 prototype code
→ 单 patch 不变矩（mean/std/max）+ patch 内结构-属性联合矩
pair encoder：输入关系描述符
→ 距离分桶的跨中心联合矩 + 无序中心对 sketch
→ 置换不变池化（sum / sum-of-squares / max / log-count）
→ 一个 graph-level head
```

无消息传递、无 attention、无 K-SVD 更新（只用 K-SVD 初始化）、无集成。

**结果：**

| 数据集 | 变体 | 参数量 | valid | test (重训) |
|---|---|---:|---:|---:|
| ZINC | 统一模型 v1（粗哈希 + OOV 回退） | 143,777 | MAE 0.6582 | MAE 0.6666 |
| ZINC | 统一模型 v2（typed 精确根证书） | 166,509 | MAE 0.1930 | MAE 0.1640 |
| MolHIV | OOV-only token | 115,693 | AUC 0.8383 | AUC 0.7727 |
| MolHIV | 精确 rooted token | 325,957 | AUC 0.8052 | AUC 0.7746 |

注意：ZINC 统一模型 v1 valid 0.6582 **远差于**方案七的 0.1872——v1 的粗哈希 + OOV 回退在 ZINC 上丢失了太多精确身份，v2 换上 ZINC 专用的精确根证书后回升到 0.1930/0.1640，接近但未超过方案七。MolHIV 补上精确 token 后覆盖率从 0 到 1.0，但 valid 反而从 0.8383 降到 0.8052。更细的 token identity 同时增加了参数量和过拟合风险。

**说明：** 统一了计算框架，但没有统一归纳偏置。ZINC 更需要细粒度局部 token、长程上下文和回归友好的连续表征；MolHIV 的强项是低维、强正则化的统计分布和骨架鲁棒的化学组成。把所有机制塞进一个模型反而不如专门设计的方案七。当前不应为了"模型统一"而牺牲已验证的有效归纳偏置。

结果入口：[`UNIFIED_RELATIONAL_PATCH_ZINC_20260905.md`](./UNIFIED_RELATIONAL_PATCH_ZINC_20260905.md)、[`UNIFIED_RELATIONAL_PATCH_MOLHIV_20260905.md`](./UNIFIED_RELATIONAL_PATCH_MOLHIV_20260905.md)、[`UNIFIED_TYPED_RELATIONAL_PATCH_ENCODER_ZINC_20260905.md`](./UNIFIED_TYPED_RELATIONAL_PATCH_ENCODER_ZINC_20260905.md)、[`UNIFIED_TYPED_RELATIONAL_PATCH_ENCODER_MOLHIV_20260905.md`](./UNIFIED_TYPED_RELATIONAL_PATCH_ENCODER_MOLHIV_20260905.md)、[`UNIFIED_EXACT_ROOTED_RELATIONAL_PATCH_ENCODER_MOLHIV_20260905.md`](./UNIFIED_EXACT_ROOTED_RELATIONAL_PATCH_ENCODER_MOLHIV_20260905.md)

---

## 方案九：结构化可解释诊断模型

在方案七、八之后做了两个"结构化"模型（`structured_patch_relational_model`）：把单 patch 的拓扑/属性分开编码、内部低秩 binding、距离分桶 pair 乘积，但**图级 head 用线性层**，使预测成为结构/属性/patch 内 binding/几何/跨中心各分量/全局上下文的**精确加和分解**——设计意图是归因诊断，不是提分。

- ZINC：`0.2441 / 0.2059`（valid/test，test 为 train+valid 重训）；MolHIV：`0.7948 / 0.7530`。
- 均低于方案七（ZINC 0.1872/0.1381；MolHIV 0.8028/0.7852），说明线性加和读出头不足以继承 MLP 读出的性能；但本次运行保证语句可做逐分量分解，是后续解释实验的基础，不是候选主模型。

结果入口：[`STRUCTURED_PATCH_RELATIONAL_MODEL_ZINC_20260905.md`](./STRUCTURED_PATCH_RELATIONAL_MODEL_ZINC_20260905.md)、[`STRUCTURED_PATCH_RELATIONAL_MODEL_MOLHIV_20260905.md`](./STRUCTURED_PATCH_RELATIONAL_MODEL_MOLHIV_20260905.md)

---

## 当前判断

**1. K-SVD 是压缩器，不是任务表征学习器。** 在所有阶段、两个数据集上，重建误差下降都与下游任务改善脱钩，甚至负相关。当前不能把贡献表述为"无监督 K-SVD 学到了任务最优结构原型"。

**2. 结构的作用不止一个层次。** 至少三层：局部上下文（radius-3 > radius-1/2）、局部环境分布（std/分位数 > mean）、环境间联合变化（跨中心协方差有机制证据）。"把结构做好"不是单纯调大 radius，而是定义完整的不变局部对象并保留其分布与联合变化。

**3. 结构-属性 binding 可检测但不可稳定预测。** 多个 shuffle 实验证明真实优于置乱，但跨骨架、跨 PCA 范围下排序不稳定。必须区分"检测到关系"与"关系能稳定提升预测"。

**4. 统计读出不弱，关键是统计对象。** 只统计全图标量：弱；对 patch 求 mean：偏弱；保留 mean/std/分布：有效；进一步保留跨中心协方差：有机制价值。没有证据表明必须用消息传递或 Transformer 才能超过显式统计。

**5. ZINC 和 MolHIV 的瓶颈不同。** ZINC 从 0.5 推进到 0.13-0.18（当前最佳 test `0.1346`），离 0.1 仍有缺口，主要来自精确 patch 复用率低、边际统计丢掉联合出现方式、高维稀疏过拟合。MolHIV valid 已很强（0.84），瓶颈是骨架测试泛化，不是模型表达能力不足。

**6. "模型统一"不是当前收益来源。** 统一 encoder（方案八）在 MolHIV 可用 OOV 归纳偏置保持 0.838,在 ZINC 必须用精确根证书才能回到 0.19；结构化线性诊断（方案九）不提升性能。性能收益目前集中在"精确局部身份 + 显式关系 + MLP 读出"（方案七）这一组合上，跨数据集的统一架构收益尚不明确。

---

## 暂不应继续的方向

- 盲扫 K-SVD 的字典大小/稀疏度/迭代轮数，把重建误差当任务指标
- 无稳定性门槛的情况下增加精确轨道数 / WL 轮数 / 维度
- 在同一 official-valid/test 上反复选融合权重
- 堆 attention / Transformer / FiLM / 消息传递层
- 把 smoke 或单 seed 最好数字当正式结论

---

## 下一步主线

```text
不变局部对象
→ patch 分布读出
→ 跨中心结构异质性
→ 任务对齐的低维交互
→ 一个下游 head
```

需先回答两个问题：
1. 哪些跨中心交互在新骨架划分上仍然稳定？
2. ZINC 的 0.13–0.18 差距来自 patch 身份、关系距离、读出统计，还是缺少全分子上下文？（层级关系上下文已把 test 推到 0.1346，但单次运行；下一步应做该模型的种重复与逐步消融来回答此问题。）

只有这两个问题有稳定答案后，才有必要决定是否引入任务对齐的字典学习、数据集条件化的 head，或更深的融合网络。
