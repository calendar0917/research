# NEXT_DIRECTION — ZINC no-MP 性能差距与原创机制边界复核

**结论：`NO_NEW_MODEL_HYPOTHESIS`。** 基准 `calendar0917/research@405fc7d`；只读核查三套公开源码及本仓库既有实验，未训练新模型、未使用 GPU、未读取 official test/valid。文中的 DCC-L 官方 test 数字仅引用历史冻结报告，并非本次重算。

## 1. 能从真实前向和训练协议确认什么？

| 模型 | 实际信息流与读出 | 特征、规模和训练配方 | 已公开的成绩/核查边界 |
|---|---|---|---|
| **DCC-L（本仓库）** | 原子/键/junction 一次性编码；真实 junction→bond 归属的 `c_loc` 与图级共享 junction 的 `c_glob`；`[Σc_glob,Σa,Σa²,global33,Σc_loc,Σc_loc²]` 共 513 维，经 MLP 读出；**无递归状态更新** | 494,193 参数；显式原子半径统计、环与距离等 `global33`，masked-SSL 初始化；batch 128、600 epoch＝47,400 updates、L1 | 四 seed official-valid **0.07599**；历史已暴露的 test **0.06732**，不作为新盲测 |
| **ESA+（Buterez et al., 2025）** | 边 token 初态 `MLP([x_src,x_dst,e])`；共享端点的边对用 masked self-attention，随后可交错全局 self-attention；每层**更新边内容**；最后以 **32 个 learned PMA seed** 做注意力池化和回归 | 论文的 ZINC-12k ESA+ 含 **RWSE + ring-membership 特征 + 调优配方**；公开模型源码能确认 M/S/P 计算，但**未发现可对应这条成绩的冻结 subset 配方、精确参数数和训练时长** | 论文 Table 6 **0.052±0.001**；不能把 demo 命令误当论文复现 |
| **GRIT（Ma et al., 2023）** | 节点 `h_i` 与所有节点对 `e_ij` 共同演化；21 步 RRWP 初始化相对节点对，边调制 attention 的 logit/content，**逐层更新节点对状态**，配 degree scaler；节点 **sum + MLP** 读出 | 官方代码：473,473 参数、10 层、宽 64、8 头，batch 32、2000 epoch 上限、约 **626k updates**；本仓库测得每 epoch ≈35.2 s，含 valid 的完整单 seed **预计约 20.5 A100·h** | 文献 ZINC-12k **0.059±0.002**；仓库只完成前向/成本审计，**未复现性能** |
| **CSA-rings（Menegaux et al., 2023）** | 节点 token、**node-pair 级关系滤波器**；各特征通道独立进行 key-axis softmax，并把 edge-value 消息加入节点；10 层更新节点，**逐节点 MLP 后 sum**；配置 `None+CSA` 无 GNN local branch | 公开 ZINC Rings 配方：宽 64、8 头、batch 32、2000 epoch 上限、RWSE（1–20）、环长≤18的同环节点对边扩充；**未在本仓库验证实际参数/耗时或复现成绩** | 文献 ZINC-12k **约 0.056**；不能将收益单独归于 chromatic attention |

**代码与口径关键发现：**

1. ESA 的 [模型前向](https://github.com/davidbuterez/edge-set-attention/blob/861a0b12d445ae80cc69b3948af623b4331321d4/esa/models.py) 和 [层实现](https://github.com/davidbuterez/edge-set-attention/blob/861a0b12d445ae80cc69b3948af623b4331321d4/esa/masked_layers.py) 明确支持 M/S/P；但公开 [ZINC loader](https://github.com/davidbuterez/edge-set-attention/blob/861a0b12d445ae80cc69b3948af623b4331321d4/data_loading/data_loading.py) 写的是 **`subset=False`（250k）**、train-label StandardScaler；[训练器](https://github.com/davidbuterez/edge-set-attention/blob/861a0b12d445ae80cc69b3948af623b4331321d4/esa/train.py) 为 `max_epochs=-1` + early stopping。README 也声明 benchmark 示例参数并非论文最优配置。因此 **0.052 是论文证据，不是本次从公开 demo 核验过的 12k checkpoint**。ESA 论文的 mask/layer-order 消融也不提供 ZINC 上同参数、同特征、同时间的 all-unmasked 对照。
2. GRIT 的 [ZINC YAML](https://github.com/LiamMa/GRIT/blob/6c988ea600a606fbb49a2246c64a2d37396b3ab5/configs/GRIT/zinc-GRIT-RRWP.yaml)、[真实层](https://github.com/LiamMa/GRIT/blob/6c988ea600a606fbb49a2246c64a2d37396b3ab5/grit/layer/grit_layer.py) 与本仓库 [COST_REPORT](zinc_grit_reference_preparation_v1/COST_REPORT.md) 对齐。2000 epoch 是**配置上限**，不是已核实的论文最终训练步数；相对 DCC-L 的计算差距很大。GRIT 的 `layers_pre_mp=0`，但仍有逐层**关系注意力与状态更新**，不能描述成“静态统计读出”。
3. CSA 的 [ZINC Rings YAML](https://github.com/inria-thoth/csa/blob/74167c024b8475e043d4eb23a077fb3eae620110/configs/CSA/zinc-CSA%2BRings.yaml)、[逐通道 attention](https://github.com/inria-thoth/csa/blob/74167c024b8475e043d4eb23a077fb3eae620110/graphgps/layer/csa_layer.py)、[ring→node-pair 扩充](https://github.com/inria-thoth/csa/blob/74167c024b8475e043d4eb23a077fb3eae620110/graphgps/transform/transforms.py) 确实提供显式**成对环关系**；这比 DCC-L 的全图环统计更细，但与 ZINC 人工 cycle penalty 相关，**不是已隔离的通用组合泛化机制**。
4. 三者的 published MAE 均不是本仓库中与 DCC-L **同特征、同标签变换、同更新次数、同 checkpoint 选择**的配对实验。ESA+ 的 exact ZINC-12k 参数/训练成本、GRIT/CSA 的实际收敛 epoch 仍未核实；不得填猜测值。

## 2. 两条最可信的“性能差距解释”及其反证

### H1：**池化前的持久、反复的关系状态更新**，而非静态一跳编码，改善组合规则复用

**源码依据：** ESA 的边 token 经 M/S 多层更新后才进入 PMA；GRIT 在每层同时更新 `h_i` 与 `e_ij`。DCC-L 仅完成一次 junction→bond 聚合，随后以矩求和读出。多层共享计算在理论上可能使稀有局部配置由常见组成部分迁移而来，但**没有任何现有对照证明它对 DCC-L 的 unseen radius-2 分子有益**。

**历史反证：** [DBT](zinc_directed_bond_transformer_v1/REPORT.md) 已用 4 层全局有关系的 directed-bond attention、1500 epoch，test **0.06940**，没有超过 DCC-L；[GOS 勘误](zinc_global_operator_search_v1/ERRATUM.md) 的实际算子是 dot-attention + relation content gate，test **0.06779**，也没有形成领先；[object-token attention](results/zinc_structural_token_attention_v1/REPORT.md) 相比 P3 更差。它们不等同 ESA 的交错 mask/PMA，但**足以否定“多层注意力一般会自动解决低支持泛化”**。若未来重开，必须在同等特征和训练预算下看到稳定、足够大的 unseen-group 增益，且独立于 cycle-tail；否则推翻该解释。**创新边界：** edge-state refinement + attention 已被 ESA/GRIT 实施；仅复制不是新方法。

### H2：**保留对象之间的相对结构关系**，而非只提供节点/键统计边际

**源码依据：** GRIT 以 `(D^{-1}A)^k_{ij}`（k≤21）的相对编码作为**节点对**输入；CSA 让 RWSE 与同环关系成为**节点对级滤波器**。DCC-L 虽含 diameter、mean distance、ring histogram，但在最终 reader 不显式保留“哪些键/原子构成哪个远距/同环关系”。这在信息上有差别，不等于标签必需。

**历史反证：** [DCC 信息审计及修复](zinc_dcc_information_audit_v1/FOLLOWUP_DIAGNOSTIC.md) 严格证明 pooled-state 的 activation-level 碰撞，但 `J_raw/J_latent` 的鲁棒 ProbeFit OOF **无可测条件收益**；[RCC](zinc_relational_contrast_contraction_v0/REPORT.md) 的关系统计约 **99.4% 可由既有 reader 线性重建**；DBT 已直接输入 exact path-count/distance 的关系并未获胜；[chromatic pilot](zinc_p3_chromatic_bond_v0/REPORT.md) 单 seed C 0.08107 vs matched scalar 0.07713，且其预注册早期不稳定门禁过严，**不能说原理已证伪**。新的相对关系若要成为方法命题，需先证明其增量既**不可由现有表示重建**，又在对称、等参数、独立 holdout 中改善预测；单纯添加 RWSE/ring/path 或 per-channel bias 已有 GRIT/CSA/SPSE/MoSE 先例。**创新边界：** 这本身也不是原创机制。

## 3. 为什么现在不应启动新模型

跨轮最稳定的现象是：DCC-L Fit8k 训练 MAE ≈0.0004，而独立 AuditEval 上充分支持 / 低支持 / 未见 radius-2 环境的原始 MAE 约为 **0.048/0.060/0.124**；历史 Fit4k→Fit8k 配对实验在相同更新数下有明显改善。现象支持**有限标签下的组合可学习性困难**，但不识别原因属于信息缺失、归纳偏置、目标噪声还是监督效率。ZINC 标签的 NetworkX basis cycle penalty 甚至对部分原子顺序敏感（见 [phase1e](results/phase1e/REPORT.md)），使 ring 相关收益更难归因为化学规律。

**最终研究决定：`NO_NEW_MODEL_HYPOTHESIS`，不是 `ALL_ATTENTION_FAILED`。** H1/H2 是未充分识别的**性能解释**，不是两条获得批准的原创模型设计；“边状态迭代”“相对路径/RWSE/环”“注意力池化”已有公开先例，而本仓库更接近它们的多个实施均未产生可复现的目标级增益。当前没有一个同时满足**预测收益线索、与旧实验本质不同、原创性**的可实施假设。

**行动：** 冻结 DCC-L 作为基线，结束 R1–R4/J、普通 attention、ring/path 特征加法的 GPU 搜索；不为复现一个并无创新的 GRIT 论文数字花约 20.5 A100·h。若以论文创新为首要目标，可将主问题转为“给定同一显式结构信息，哪种可学习机制在小样本、未见结构上提高效率”，优先选择**实验测量性质和 scaffold-disjoint** 任务以降低人工标签混杂，且需正视 [Phase1a](results/phase1a/REPORT.md) 中同信息 attention 未改善的既有反证。重新训练的前提必须是：先写出与 ESA/GRIT/CSA 及本仓库 DBT/GOS/STA/SCM/J **不同的最小计算命题**，给出独立于强模型补丁的理论/反例依据，并预先冻结同特征、同预算、支持度分层与真实 held-out 指标。达不到这些条件，维持暂停，不再用架构试错代替研究命题。

**外部论文：** [ESA+，Nature Communications 2025](https://www.nature.com/articles/s41467-025-60252-z)；[GRIT，ICML 2023](https://proceedings.mlr.press/v202/ma23c.html)；[CSA，arXiv 2023](https://arxiv.org/abs/2304.10933)。所有公开源码均核对于本次记录的固定 commit；本文件只使用原有报告数字，不对外部成绩做新复现。