# RESEARCH_PROPOSAL — 结构覆盖度预训练假设（H-COVER）

* 轮次：`bjoc_predictive_research_redirection_v1`
* Track：`tracks/structure-contribution`
* 性质：研究方案（**不执行**）。随附证据见 `EVIDENCE_MAP.md`，协议草案见 `EXPERIMENT_PLAN.md`。
* 本轮结论的边界（先说清楚）：
  1. **模块级结论：`NO_JUSTIFIED_NEW_MODULE`。** 现有证据不支持再向 BJOC-Null 添加任何新的结构模块
     （ownership / pair-channel / 加性残差 / 结构词表 / 监督期 SSL），理由见 `EVIDENCE_MAP.md` §2、§5。
  2. **机制级结论：恰好有一个未被检验、且在证据上优先级最高的机制值得一次实验** —— 预训练**结构覆盖度**
     （H-COVER）。它不新增任何参数或模块，只改变已确认正结果（masked SSL）所依赖的数据侧机制。
  3. 如果研究者要求的是"**方法创新**"而不是"瓶颈判定 + 性能验证"，那么本轮的诚实结论同样应写作
     `NO_JUSTIFIED_NEW_EXPERIMENT`：H-COVER 不是新方法（见 §6）。两者边界在 §0.1 给出。

## 0.1 两个结论的边界

| 问题 | 结论 |
|---|---|
| 有没有一个新的**结构表示模块**（新通路/新聚合/新读出/新 SSL 目标）值得买 GPU？ | 没有。任何邻近设计要么与已关闭 family 同构，要么与 masked atom/bond CE 数学等价，要么信息可由现有输入导出。 |
| 有没有一个**机制假设**值得一次资源有限、对照严格的实验？ | 有，且只有一个：**在匹配算力下，扩大预训练结构分布是否改善下游泛化**（H-COVER）。它测的是瓶颈归属，不是方法创新。 |
| 如果该假设失败？ | 见 §7；程序应停止扩展 BJOC 的表示/训练空间，转入独立确认与第二个分子端点或重新定位。 |

---

## 1. 一句话命题（7.1）

> 在控制相同的 masked atom/bond SSL 目标、相同的 7,560 次预训练更新、相同的 BJOC-Null 架构（413,505 参数）、
> 相同的 R3 监督配方、相同的 fine-tune 数据（official-train 10k）与相同 seed 协议后，把 SSL 的**结构分布**
> 从 official-train 10k 子集扩展到完整 ZINC 训练集（220,011 个未标注结构，排除 official valid/test 结构），
> 预计通过改善 encoder 对**少见局部环境（atom/bond/junction context）的初始化质量**，
> 使内部 held-out 的 seed-mean raw-y MAE 相对冻结 R3 至少改善 **0.004**，且改善不集中于极端单点分子。

形式化：设 `D_N` 为预训练结构分布（大小 N），`θ_SSL(N)` 为固定 7,560 次更新后的 encoder 状态，
`FT_R3` 为冻结的监督配方。命题是

```
M(D_220k) < M(D_10k) − 0.004,    M(·) := seed-mean held-out MAE of FT_R3(θ_SSL(·)),
```

其中三臂 `D_10k`（复用）、`D_50k`、`D_220k` 共享 objective、更新数、batch、优化器、mask 比例与 seed。

---

## 2. 为什么值得研究

1. **它是唯一正结果（SSL）的唯一未被检验的维度。** SSL 初始化是跨 split/seed/任务复现的最大增益
   （ZINC valid −0.0156 / exposed test −0.0207；Lipo valid −11.10% RMSE，5/5 seeds）。
   现有证据只说"同目标更多曝光饱和"（bond CE 0.015–0.019，plateau by visit ~80），
   从未测过**覆盖**（`EVIDENCE_MAP.md` §1.1、§3 B4）。
2. **样本效率瓶颈已被独立确认。** 同更新数下 4k→8k 标签量带来 −0.0373 MAE（~4× 额外优化效应），
   且 4k 拟合更好、泛化更差（`results/bjoc_predictive_bottleneck_v1/REPORT.md`）。
   在协议禁止外部标注数据的前提下，能增加"监督"的合法途径只有自监督结构覆盖。
3. **它是零参数、零架构风险的干预。** 模型、对象层级、可归因前向、解释审计管线全部不变；
   若失败，不会留下"复杂模块"债务；若成功，可立刻被所有后续解释研究复用。
4. **它对研究者三项优先级同时是正的**：性能（唯一可能的有意义增益方向）、结构信息真实作用
   （检验"结构覆盖 → 泛化"这一可观察因果链）、解释基础（前向不变、可归因性不受影响）。

---

## 3. 最强反对证据（必须写在设计之前）

1. **"8k→10k 的 diversity 证据"不干净。** 8k 轮与 10k 轮的 P−B 来自不同 split、不同基线、
   不同 readout（locked-dev vs official-valid），不是同轮剂量-反应；`bjoc_full10k_pretraining_transfer_v1`
   自己给出的是"exposure 饱和"判断。**H-COVER 的正面先验实际上比它看起来弱。**
2. **增加未标注结构不增加极端标签样本。** tail 失败有 coverage/statistics 签名，
   但它首先是 **label-space** 尾部；假设极端分子的结构近邻本来就稀少，则结构覆盖也无济于事。
   因此 H-COVER 的增益目标只能定在 bulk/ordinary + 总体 MAE，不能承诺修 tail。
3. **标签越少，SSL 越有用。** Lipo（3,360 标签）的 SSL 增益（11%）远大于 ZINC（10k 标签）的 6%；
   这暗示 ZINC 上 SSL 的边际价值可能已接近耗尽，扩大未标注语料可能只带来很小的增量。
4. **持久误差可能是模型家族偏差而非数据不足。** 四模型共同高误差集中在大型环富分子、残差相关 0.907–0.949
   （`bjoc_predictive_bottleneck_v1` §3）——若是归纳偏置问题，加数据只会缓慢改善而不会消除。
5. **高方差历史。** 同一 J-fit 到 1e-6，但 fine-tune 种子间差 ~0.002–0.003；
   若真实效应 <0.003，两 seed 无法分辨（recipe 轮 seed-1 仅 −0.0017 即为例证）。
   → 我们的最小实质效应阈值必须 ≥0.004，并预先接受"测不出小效应"。

---

## 4. 数学设计（7.2）

### 4.1 输入/干预位置

* **模型输入**：不变（raw PyG `x / edge_index / edge_attr` → BJOC 对象层级）。
* **改动位置**：只改**预训练阶段采样的结构分布** `p_SSL(G)`，不改 encoder/composer/reader 的任何公式。
* **参数**：新增 0；预训练头是训练期模块，推理前移除（与现协议一致）。

现机制（`code/bjoc_masked_pretraining_v1.py`，`code/run_bjoc_full10k_pretraining_transfer_v1.py::pretrain_full10k`）：

```
D_10k = official ZINC subset=True train（10,000 行）
for step in 1..7560:
    G ~ Uniform(D_10k)（按 epoch 打乱）
    A_mask, B_mask ~ mask(G; SeedSequence(seed, mol_id, visit))       # 15% / 15%
    G_m = rebuild(raw_codes masked, A_mask, B_mask)                    # 全部 BJOC 特征重建
    L = CE(atom_head(E_atom(G_m)), hidden atom types)
      + CE(bond_head(E_bond(G_m)),  hidden physical bond types)
    θ ← Adam(L, lr=1e-3, wd=1e-5, clip=5), batch=128
```

新机制（H-COVER）：

```
D_N ∈ { D_10k,  D_50k,  D_220k }        # D_50k ⊂ D_220k；D_10k ⊂ D_50k ⊂ D_220k
D_220k = PyG ZINC subset=False train（220,011 行）− 与 official valid/test 结构碰撞的行（Stage 0 审计）
for step in 1..7560:                     # 完全相同的更新预算
    G ~ Uniform(D_N)
    …（其余逐字不变：mask 规则、rebuild、loss、优化器、seed）
```

因此三臂的**唯一差异**是结构分布 `D_N`；由此

```
views_per_molecule = 967,680 / N ∈ {96.8, 19.4, 4.4}
```

形成一条 **diversity–exposure 剂量轴**：若"覆盖"是有效机制，MAE 应随 log N 上升而下降；
若"每分子曝光"才是有效机制，顺序应相反（10k 最好）。这条曲线本身就是机制判据，而不只是性能读数。

### 4.2 Fine-tune（三臂完全相同，且与冻结 R3 逐项匹配）

```
θ ← FT_R3(θ_SSL(N), seed)：official train 10k raw-y，
    Adam 1e-3→1e-5 per-step cosine，wd 1e-5，clip 5，batch 128，
    360 epochs = 28,440 updates，raw-y L1，last-5 epoch soup。
composer/reader 初始化、per-seed batch order、种子常量与 R3 完全一致（哈希核对）。
```

### 4.3 信息依赖范围

* SSL 只依赖 raw 结构（`x / edge_index / edge_attr`）；y 不进入任何代码路径（沿用 AST 扫描 + `labels_read:false`）。
* fine-tune 只依赖 official-train 10k 的 y；valid 只在预测全部冻结后单次读取。
* 语料扩展不引入任何新的跨对象通路：fine-tune 的 forward 与冻结 R3 bit-level 同构
  （同参数、同输入，只有权重不同）。

### 4.4 与现有 BJOC 的关系 / 为什么不是重复

* 不新增 ownership / pair / residual / 词表 / 新 SSL 目标；不改聚合或读出；
  不触碰已关闭 family（`EVIDENCE_MAP.md` §5）。
* **不针对尾部加权/重采样**：目标样本分布仍是 `Uniform(D_N)`，不按标签、不按结构稀有度重采样；
  尾部若有改善，只来自"共同出现的局部环境被覆盖"这一间接通道（预测机制见 §7.1），
  且必须在报告中按 ordinary/tail 分开披露（tail 改善不作为主要卖点）。
* **语料是真实分子**（ZINC train 原始结构，无合成/截断/加噪结构）；标签从不参与语料构建。
* **防泄漏**：valid/test 的结构指纹与 D_220k 求交并排除碰撞行；SSL 只读结构张量，
  不读任何标签缓存（AST + label-access 计数）。
* 与"原 SSL 增加同等预训练计算量"的区别：更新数、batch、优化器逐项相同，算力匹配；
  变化的是样本分布，不是步数。
* 与"原 BJOC-R3 增加同等监督更新"的区别：fine-tune 完全冻结。

---

## 5. 最强竞争解释与控制（7.3）

| 若 MAE 提升，最可能被错误归因为 | 具体控制 |
|---|---|
| 训练计算量更多 | 预训练更新数、batch、优化器、LR、clip 逐项相同；fine-tune 配方逐项相同；GPU 记录每臂 wall 与 steps |
| 参数量增加 | 参数为 0 增量；逐臂 `parameter_audit` 断言 413,505（预训练头推理前移除） |
| 初始化/seed 运气 | 同 seed 协议；seed-0 筛选 + 预注册 seed-1 确认；composer/reader init 与 batch order 哈希对齐 |
| 暴露次数（每分子被看到的次数） | **三臂剂量轴**：10k（96.8）→50k（19.4）→220k（4.4）；若收益随曝光减少而上升，才能归因覆盖 |
| 优化噪声/不同 mask 流 | mask RNG 规则逐字不变（`SeedSequence(seed, mol_id, visit)`）；每臂记录预训练 CE/acc 曲线与最终 encoder hash |
| 验证集选择偏差 | 单次冻结读取；valid 是暴露 dev 集（披露）；禁止在 valid 上做任何 per-arm 选择之外的决定；不得读 test |
| 极端样本驱动 | 预注册 ordinary / tail / top-10-removed / 单分子占比条件（见 `EXPERIMENT_PLAN.md` §Stage 2） |
| 数据泄漏（valid/test 结构进入 SSL） | Stage 0 结构碰撞审计：排除与 official valid/test raw-graph 哈希碰撞的行；全流程只读结构不读标签 |
| 学习率/训练时长变化 | 无：LR/schedule/epoch 全部继承 R3 冻结常量 |

**最关键的对照**：同一 mask 目标、同一算力下的 **10k vs 220k**（预测的主对比）；
**50k 剂量臂**用于区分"覆盖效应"与"曝光稀释"两种解释（这是本设计相对任何单一 arm 实验的核心增量）。

---

## 6. 为什么不是直接选择已有方法（7.4）+ 文献与新颖性审查

### 6.1 简单强替代方案

* **基线**：冻结的 R3（SSL-10k + R3；`soup_R3_seed{0,1}.pt`，valid 0.079844 / exposed test 0.069822）。
  它是当前 system-level 最强，也是 220k 臂必须击败的对象。
* **等算力对照**：10k 语料下增加 SSL 更新数。已有证据表明同目标曝光饱和
  （bond acc 0.995、CE 0.015、plateau by visit ~80；`results/bjoc_full10k_pretraining_transfer_v1/REPORT.md` §5），
  因此**不单独购买**该臂，而是把 50k 剂量臂作为等价的信息对照（同样花算力、更多分子、更少曝光）。
* **容量匹配对照**：无需购买——H-COVER 零参数增量，容量混杂在结构上被排除。
* **破坏目标结构的对照**：不适用（本机制不引入结构关系）；对照是数据分布本身。

### 6.2 聚焦文献核验（只查与机制直接相关的工作）

| 主题 | 已核对记录 | 对 H-COVER 的含义 |
|---|---|---|
| masked graph modeling | GraphMAE，KDD 2022，DOI `10.1145/3534678.3539321`（arXiv 2205.10803）；GraphMAE2，arXiv 2304.04779 | 本机制的预训练目标本身是既有方法；不主张原创 |
| atom/bond-level SSL | Hu et al., *Strategies for Pre-training GNNs*，arXiv 1905.12265（AttrMasking / ContextPred） | "mask 属性从上下文重建"有直接先例；不主张原创 |
| molecular SSL + 大规模未标注语料 | GROVER，arXiv 2007.02835（摘要原文："We pre-train GROVER with 100 million parameters on 10 million unlabelled molecules"）；MolCLR，arXiv 2102.10056；GraphCL，arXiv 2010.13902 | "扩大未标注语料做 SSL"是成熟实践；因此 **H-COVER 不是方法创新，只能是瓶颈实验** |
| fragment / motif 级预训练 | Fragment-based Pretraining and Finetuning on Molecular Graphs，arXiv 2310.03274 | 本轮不使用 fragment 目标；与既有 fragment 预训练无重叠 |
| 大规模分子表示预训练 | Mole-BERT（ChemRxiv DOI `10.26434/chemrxiv-2023-dngg4`，正式 venue 待核对）；Uni-Mol（ChemRxiv DOI `10.26434/chemrxiv-2022-jjm0j`，ICLR 2023 待核对） | 再次说明 corpus scaling 不是新机制 |
| graph-level readout / 长程信息 | 未做新核验（本轮不涉及 readout 改动）；仓内既有证据见 `results/bjoc_junction_sufficiency_audit_v1/REPORT.md` B2、`results/distance_pooling_bottleneck_v1/` | fallback（scale-separated readout）若启动，需另行核验 Set2Set/Janossy/DeepSets 类工作 |

回答任务 §6 的五个问题：

1. **与最接近方法的数学等价性**：H-COVER 的 SSL objective 与 GraphMAE/Hu et al. 的 masked reconstruction
   基本同构；H-COVER 不声称差异。实质差异只在于它初始化的对象是 **BJOC 一次性非消息传递层级**
   （atom→bond→junction→static incidence→sum composer），而现有文献多在 GNN/Transformer 上验证。
   这个差异是**系统差异**，不是新算法。
2. **单一 ZINC-12k 任务能否支撑新颖性**：不能。因此本轮不把 H-COVER 当作创新点，只当作
   "系统内部的瓶颈判定 + 性能验证"；论文级新颖性必须来自系统组合与解释框架（见 §9、最终 Q5）。
3. **最可能形成实质差异的设计点**：剂量轴（同算力、同目标、10k/50k/220k）+ 与冻结 R3 的逐项匹配
   + 覆盖度诊断（held-out 局部环境在预训练语料中的出现率与 per-molecule 误差变化的关系）。
4. **不能声称为原创的**：masked atom/bond SSL；"更多未标注数据改善预训练"；用大语料做 self-supervised
   pretraining；任何 corpus scaling 规律。
5. **相关工作总结**：核心机制已被覆盖。若研究者要求方法新颖性，应判 `NO_JUSTIFIED_NEW_EXPERIMENT`；
   若接受"性能/瓶颈实验"定位，则 H-COVER 仍值得一轮（§0.1 的边界）。

---

## 7. 预期改善机制、失败后的判断更新、预算（7.5–7.7）

### 7.1 预期机制链

```
D_220k 覆盖更多原子/键/junction 局部环境类型（label-free 可测）
   → encoder 在 SSL 后对这些环境有更稳定的编码（预训练 CE 可能更高但 downstream 更好）
   → fine-tune 在更宽的结构先验上开始
   → held-out MAE 改善；且改善应集中在"局部环境在 10k 中稀少、在 220k 中被覆盖"的分子
```

**机制验收诊断**（预注册、标签盲、不参与选择）：
`cov_10k(e)` / `cov_220k(e)` = held-out 分子局部环境类型 e 在预训练语料中的出现率；
报告 ΔMAE 与 `Δcov` 的相关；同时报告预训练 CE/acc 曲线（预期：220k 臂 CE 更高、acc 更低，
但这不是失败条件——它说明任务更难，重点看 downstream）。

### 7.2 失败后的研究判断更新（预注册）

| 结果 | 更新 |
|---|---|
| 220k/50k 均 ≥ baseline（或 CI 跨零、单 seed 获益） | **结构覆盖不是有效杠杆**。结合模块全负证据，停止扩展 BJOC 表示/训练空间；转独立确认 + 第二端点（Lipophilicity/ESOL 类）或重新定位到解释研究。 |
| 单调改善但 <0.004 | 覆盖方向存在但效应低于可操作阈值；不追（不扩语料、不调步数），记录为"小效应"。 |
| 220k 改善、50k 更差 | 非单调 → 覆盖/曝光 trade-off；不缩放，仅记录。 |
| 220k 改善且 ≥0.004、两 seed 同向 | 进入 §Stage 4：在**新内部 split**上独立确认（需新的预注册），再谈是否值得外部端点复现。 |
| ordinary 改善但 RMSE/tail 明显恶化 | 判为分布级权衡而非泛化增益，停止（沿用历史 gate）。 |

### 7.3 计算预算估算

| 项 | 单 seed 估算 | 依据 |
|---|---|---|
| SSL（7,560 updates，batch 128） | 0.47–0.50 GPU-h | full10k SSL0/SSL1 = 1779 s / 1666 s（a100-2, `results/bjoc_full10k_pretraining_transfer_v1/REPORT.md`） |
| R3 fine-tune（28,440 updates） | 0.19–0.20 GPU-h | recipe 轮 R3 678–693 s |
| 单臂单 seed 小计 | ~0.70 GPU-h | |
| 计划：50k s0 + 220k s0 + 胜者 s1（+ smoke） | ~2.1–2.5 GPU-h | 上限 3.5 GPU-h |
| CPU | valid predict/score/审计按历史流程，无新 GPU | |

数据成本：full ZINC raw 已在本地（`data/ZINC/raw/train.pickle` 220,011 行；无需下载）；
Stage 0 需实测 220k 对象构建吞吐/内存并决定缓存或流式。

### 7.4 对解释研究的意义（§IX 最低要求）

* 模型 forward 不变：Atom/Bond 编码、Junction 内容、local/global 混合方式、聚合不可逆性都与现有
  `results/bjoc_junction_sufficiency_audit_v1/REPORT.md` B1–B4 的审计对象一致；本干预**不新增**任何混合通路。
* 覆盖度诊断本身对解释有用：它回答"哪些局部环境在预训练中根本没有被见过"，
  是未来 source-level attribution 的**分母**（没有见过的环境谈不上稳定贡献）。
* 不把任何 internal sensitivity 当化学解释的纪律不变（沿用 `results/bjoc_source_ownership_v1/REPORT.md` §3 的措辞）。

---

## 8. 交付的下一项对照（对应任务 §8 Stage 1）

* Base：冻结 R3（`soup_R3_seed{0,1}.pt`）。
* Candidate A（剂量点 1）：SSL on `D_50k`（deterministic hash 排序前 50k）→ R3。
* Candidate B（主候选）：SSL on `D_220k`（full train − valid/test 结构碰撞行）→ R3。
* 主对比：`B − Base`；机制轴：`A` 的位置决定"覆盖 vs 曝光"方向。
* 逐项匹配：SSL objective/mask/optimizer/updates；fine-tune 数据/batch order/composer-reader init/R3 配方。
* 数据纪律：SSL 只读结构；official valid 单次冻结读取；official test 与 Locked 永不加载；不读任何新标签。

（完整 Stage 0–4 见 `EXPERIMENT_PLAN.md`。）

---

## 9. 论文级含义（回答最终 Q5 的一半）

即使 H-COVER 成功，**它也不构成方法创新**；它的价值是把"数据侧瓶颈"钉死或排除，
从而决定论文的主张形态。当前可支持论文的资产是：
(1) 一次性非消息传递对象层级在 ZINC 上达到 0.0698 exposed test（目标带内）；
(2) masked structural SSL 的跨任务迁移证据；
(3) 系统化的结构模块负结果 + 严格 matching 纪律；
(4) 可信归因审计框架（完整性/稳定性/来源隔离）。
**缺的证据**：未暴露 split/task 上的独立确认；SSL 收益的"表示 vs 优化条件"分离；一个真正的方法新颖点
（目前最接近的是"一次性 incidence 组合 + 共享 junction 对象"的非 GNN 可归因系统，而非任何单个 SSL 目标）。

---

## PROPOSED_RESEARCH_CORE_UPDATE

> 以下仅为建议，**不改写** `notes/research_core.md`；变更需研究者最终确认。

1. **仍然保留的科学目标**：跨分子共享、结构上可对齐的单元—关系表示；不以消息传递为核心机制；
   预测性能与解释可信度同时成立；结构贡献必须能追溯到具体对象（atom/bond/junction/unit-relation）；
   禁止用无法解释的全局旁路掩盖分解代价。
2. **应降级为"历史实现限制"的条款**：
   - "预测必须严格加性分解"（`notes/research_core.md` §1/§5）——已被 TASU（加性 Reader 损失 ~0.04 MAE）
     与 STRC/GLRC/DLP 负结果挑战；研究者最新判断明确接受非线性 Reader 与邻域聚合。
   - "所有结构贡献需要唯一分摊到单元"与"排除非线性全局 Reader"——同上，应改为"可审计的信息来源与
     可复核的干扰实验"，而不是强制加和恒等式。
   - "无隐藏全局非线性旁路"保留为**信息披露要求**（哪部分依赖全局统计、哪部分依赖局部），而非架构禁令。
3. **已受负证据挑战的假设**（保留历史、不追溯修改）：
   - H1（显式单元—关系分解同时买到性能与解释）——`notes/decisions/decision_cscl_v0_h1_refuted_20261010.md`；
   - "真实 ownership 组织有独立预测价值"——O/S/GLRC/Organized/D 全负（`EVIDENCE_MAP.md` §2）；
   - "结构词表/签名可作核心解释对象"——`results/bjoc_signature_conditional_value_v1/REPORT.md` 关线；
   - "严格加性不牺牲性能"——TASU 的直接测量。
4. **建议的新研究核心表述（一句话）**：
   > 在非消息传递的一次性对象层级上，构造**可审计来源**的分子预测系统：
   > 局部结构信息经非线性但可追踪的聚合进入预测；解释主张必须区分计算来源、内部敏感性与真实化学含义，
   > 并以"结构信息是否带来可复现的泛化/迁移证据"为第一优先级。
5. **需要研究者确认的变更**：上述 §2 的降级是否正式改写 research_core §1/§5；
   是否接受"创新性来自系统组合与解释框架，而非单个 SSL 目标"的论文定位；
   以及 H-COVER 是否按 `EXPERIMENT_PLAN.md` 预注册执行。
