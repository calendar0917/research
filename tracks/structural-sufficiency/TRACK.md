# Track: SSRC — Structural Sufficiency & Relational Computation

| 项 | 内容 |
|----|------|
| slug | `structural-sufficiency` |
| 状态 | **active（主线）；Phase 0.5 纠错完成；Phase 1a–1d 完成；Phase 1e 完成 → 路线判定 `Direction A`（ZINC 大环收益是标签 `-cycles` 人工项造成的基准伪影；SSRC 臂族在同 split 上不具竞争力；下一步 = 真实性质任务 Phase C，设计已冻结、未训练）；**Phase 2a 完成 → `CAPACITY_ONLY_INCREMENT`（P3E 之上的小模块带来 +0.0055 MAE，来自容量而非边上的传播）；Phase 2b 完成 → `NO_PRACTICAL_ADVANTAGE`（标准 ZINC-12k：M valid +0.003056，one-shot test +0.001273 未确认；F valid null；M 不作为下一性能基座）；`zinc_rooted_moment_v1` 完成 → `NO_STABLE_GAIN_OVER_P`（精确 rooted 矩条件化，C3 无可重复贡献）** |
| 创建 | 2026-10-18 |
| 主线位置 | **SSRC（新主线）**：显式结构充分性 → 关系计算；旧 `structure-contribution` 标记 **paused / 非主线**，全部历史保留，不覆盖、不改写 |
| Phase 0 结论 | **REFRAME_NEEDED**：原表述“MP 是否必要”已被文献覆盖且实验易混淆；重述为“精确 \(E_r\) 的信息充分性 vs 有限样本可学习性”，条件性 GO 到受控 pilot |
| Phase 0.5 | **纠错完成**：三层概念、风险分解符号、pilot 方向/停止规则、Eijkelboom train/test 口径、ESA+ 文献、S(G) oracle 降级；见 `notes/phase0_corrigendum.md` |
| Phase 1a | **COMPLETE（R1+R2）**：36/36 轨迹；n=2352 S 在 6/6 draws 优于 X（pooled Adv −0.0387），n=253 为 5/6（pooled −0.0230）；交互 +0.0157 但仅 3/6 draws 为正、CI 跨零。总 1.21 GPU-h |
| Phase 1a 判定 | **`NO_CLEAR_INTERACTION_VALUE_IN_THIS_PILOT`**；限定 `S_AT_LEAST_AS_GOOD`、`NO_STABLE_SMALL_N_X_ADVANTAGE` |
| Phase 1b | **COMPLETE**：5 臂 × 2 任务 seed 0；B1 broken-adjacency 控制；B2 matched-compute 控制；ZINC seed 1 复现。总 1.60 GPU-h |
| Phase 1b 判定 | **`PURSUE_RELATIONAL_SUFFICIENCY`**：S0→R1 ZINC +0.0571/+0.0992；broken-R1 更差 +0.0706；M1_long ≈ M2（Δ=−0.0027，CI 跨零）⇒ 迭代增益是计算量不是机制 |
| Phase 1c 判定 | **`CORRESPONDENCE_SURVIVES_CAPACITY` + `ABSOLUTE_GAP_PERSISTS`**：Gain_pair +0.0682/+0.0309（两 seed 均值 +0.0495）；Gain_over_capacity +0.0529；R1_size 最好 seed 0.2366 vs G_ref 0.1653（gap +0.0713）；M_keep_edge 0.2370 vs R1_size 0.2366（同配方中性；预注册匹配计算被优化混淆） |
| Phase 1d 判定（**Phase 1e 已下调口径，见下一行**） | **`PAIRWISE_INTERACTION_PROMISING`（限定 `LARGE_RING_TAIL_SPECIFIC`）+ `CAPACITY_SCALING_PROMISING` + `REPRESENTATION_OR_INDUCTIVE_BIAS_GAP`**：R_incidence 0.2797 vs R1_size 0.2366（Gain_joint +0.0431，CI +0.0121…+0.0763），但 96% 无大环分子上 incidence 反而好 −0.0281；4% 大环上 R1 好 +1.751。R1_size_cap 0.1935（Gain_cap +0.0431，CI +0.0192…+0.0688），整体 G_ref gap +0.0282；96% 上 gap 仅 +0.0174，4% 尾部 +0.286。FLOPs 2.10× G_ref，记 `EFFICIENCY_OBSTACLE` |
| Phase 1e 判定 | **`BENCHMARK_CYCLE_ARTIFACT_DOMINATES`（主）+ `UNRESOLVED_LABEL_MECHANISM`（次，cycle basis 顺序依赖）+ `R1_IS_NOT_COMPETITIVE`** → **路线 `Direction A`**：标签是 `-max(0, NetworkX Paton basis 最长环 − 6)`（非置换不变，非 PyG 文档口径）；Monitor 40 个 RDKit 大环 ≡ 43 个标签受罚分子（purity 1.0、recall 0.93），受罚 4% 占各臂绝对误差 29–53%，cycle 项占该组标签量级 68%；**任何 SSRC 臂在任一子组都不优于 G_ref**（R1_size 在大环 40 上比 G_ref 差 +0.72，Phase 1d 的 +1.751 只是对更差 sibling 的相对值）；零训练条件路由 0.1922（仍比 G_ref 差 +0.0269）；严格隔离的单 GPU 强基线 0.0911 vs G_ref 0.1653（Δ −0.0742，CI −0.0965…−0.0545，补集上也 −0.0446）。0.607 GPU-h |
| Phase 2a 判定 | **`CAPACITY_ONLY_INCREMENT`**：在冻结 P3E 之后插入小模块（新参数 +12%、前向 FLOPs 1.15×），两 seed 均稳定改善 P3E **+0.005542 MAE**（0.0911 → 0.0853）；但把该模块换成真实物理边上的 2 轮 GINE **没有额外收益**：`Gain_MP` = +0.003764（seed 0）/ −0.000055（seed 1），符号跨 seed 冲突且小于对照臂的 seed 间波动（0.0039）。⇒ 进益来自新增逐原子容量，不是边上的交互。0.933 GPU-h（8 runs，预算 23.3%） |
| Phase 2b 判定 | **`NO_PRACTICAL_ADVANTAGE`**：标准 ZINC-12k（official train 10k / valid 1k / test 1k，seed-matched full-10k SSL，R3）。valid：P3 0.079844 / F 0.079616 / M 0.076788；M **+0.003056**（两 seed 正，drop-top-1 +0.002683，drop-top-10 P3-error +0.002004，complement +0.002992；CI95 −1.8e-5…+0.00623，仍跨零）；F +0.000228（seed 1 负）。`Gain_MP` +0.002828，但 seed 0 仅 +0.000033，分离由 seed 1 驱动。one-shot test：M 0.068549 vs P3 0.069822，**+0.001273**，CI95 −0.002258…+0.004725，band 与 CI 均 fail；F 未上 test（唯一 finalist 规则）。0.947 GPU-h（37.9% 预算），无 GPU 失败。 |
| Phase 2a 协议缺陷（需长期携带） | 预注册的 **绝对 RMSE 门禁（Δ≤+0.005）在 n=1000、max\|y\|≈8 的 Monitor 上是单分子统计量**：seed 0 Δ=+0.005208（差 0.000208），其中单个置换不稳定标签分子（official-train row 2052）贡献 MSE 差的 1.59×；seed 1 Δ=−0.002090，同一位形反向。**阈值未事后修改**；seed 0 产物保留 `INSUFFICIENT_EVIDENCE`，miss 与诊断写入 `seed1_purchase_record.md` 后才买 seed 1 |
| Phase 1e 基准缺陷（需长期携带） | 任何 ZINC `logP_SA_cycle` 聚合数字都混入了一个隐藏的环大小任务；历史数字（如 R1 0.260161、G_ref 0.165290）仍是对该 target 的有效测量，但**不得当作干净的性质预测结果**；未来 ZINC 声明必须单列或排除受罚子集 |

## 问题（一句话）

给定显式结构表示 \(E_r\)，学习式关系计算（消息传递、显式对象交互、迭代注意力）
什么时候是必要的——是在**补充 \(E_r\) 缺失的信息**，还是在信息不变的前提下
**改善可学习性/样本效率/泛化**？

## 范围

### 做

- 严格区分三层：**A 有限样本身份可辨识性**（\(E_r\) 是否碰撞）、
  **B 预测任务信息充分性**（是否有 \(P(Y|G)=P(Y|E_r)\) / 相同 Bayes 风险）、
  **C 有限样本可学习性**（同输入下不同假设类的泛化/样本效率/代价）。
- 用精确带属性 rooted-radius-\(r\) 诱导子图多重集 \(E_r\) 做 CPU 碰撞证书；
  精确同构/规范表示，不用普通哈希；节点重标号不变性；无标签/编号/种子泄漏。
- 文献边界审计：明确“强编码减少 MP 需求”**不是**本路线的新贡献；补审
  ESA+（Buterez et al. 2025）与 Shen & Dong 2026 的术语/对象差异。
- 真实分子：Lipophilicity；内部 Fit/Monitor/Reserved（2352/504/504）+
  冻结 n=253 子集；官方 valid/test 与 Reserved 不读。
- Phase 1a 只比较**同信息**的 S（无跨 patch 交互）与 X（跨 patch attention），
  外加 G（原始图 GINE）应用参照；不训练 \(S_{\rm bag}\)、\(S(G)\) 或五臂 oracle。
- 允许并鼓励有边界的负结果。

### 不做

- **不**重开旧轨 BJOC 已关闭的模块/配方搜索族（junction ownership、固定签名、
  STRC 加性残差、pair/距离通道、SSL 目标搜索、配方搜索、尾部加权）。
- **不**继承旧模型的硬约束（不一定非 GNN、不一定非消息传递；架构不预设）。
- **不**声称“结构编码减少消息传递需求”本身新颖（Eijkelboom 2023 已做）。
- **不**设计新架构作为本轮目标；Phase 1a 只做受控同信息比较。
- **不**做 ZINC 排行榜搜索、外部无标签预训练、或覆盖 frozen 结果。
- **不**读官方 valid/test 或本轮内部 Reserved 的逐行标签/评分；**不**用它们做
  模型选择、早停或主表。
- **不**把“训练集结构指纹唯一”当作“统计充分/可学习/可泛化”；**不**把
  \(E_r\) 的 LOO 组均值当作共享 patch 模型的替代。
- **不**把 Dynamic 访问 `edge_index` 而 Static 只看无序 patch 多重集的差异，
  解释为“同信息下动态更强”；**不**把 \(G-S\)/\(G-X\) 直接解释为纯信息增益。
- **不**在结果出来后改模型、口径、阈值或追加 seed/架构。

## 协议 / 评估

- 复用：Lipophilicity DeepChem 2.8.0 scaffold split（train/valid/test =
  3360/420/420；2408 scaffold 组；跨集泄漏 0）；label-free `MolObjects`；
  scaffold 复现校验；label vault 隔离机制。
- 本轮只做 **train-only** 分析；官方 valid/test 标签未读。
- Phase 1a 内部 scaffold split：**已冻结** 2352/504/504，scaffold 组重叠 0；
  另冻结 n=253 小样本子集（114 个 scaffold 组）。见
  `results/phase1a/split_phase1a.json`（sha256 `9fa96d49…`）。
- `protocol_id`：`ssrc-phase1a-lipo-v1`；协议见 `notes/phase1a_protocol.md`。
- 主表：Phase 0 数字源是 `results/phase0/analysis/*.json` 与
  `results/phase0/REPORT.md`；Phase 1a 结果只写 `results/phase1a/REPORT.md`，
  不与其它轨混表。

## 进度

| 阶段 | 状态 |
|------|------|
| 问题与范围 | 完成（Phase 0 重述） |
| 精读 / 概念 | 完成（7 篇 + GSN；ESA+ 未唯一识别） |
| Phase 0 CPU | 完成：17 tests；三家族 C24 碰撞；正对照；真实数据可行性 |
| Phase 0 train-only 诊断 | 完成：\(E_2/E_3\) 近唯一；scaffold 更统计高效 |
| Phase 0.5 纠错 | 完成：三层概念、风险符号、pilot 方向、文献口径、S(G) 降级 |
| Phase 1a 协议/代码 | 完成：S/X/G、split、label vault、8/8 CPU gates（远端复跑通过）、8 tests |
| Phase 1a GPU screening | **完成**：R1+R2 共 36/36 exit 0；1.208 GPU-h 实际占用（训练循环 1.120 h），≤4 h 预算 |
| Phase 1a 结果 | **S≥X**：n=2352 6/6 draws S 更好；n=253 5/6 draws S 更好；交互 3/6 为正、CI 跨零；无稳定小样本 X 优势；G 未支配 S |
| Phase 1a 复现性 | R2 在 clean worktree 重跑同 18 条；同 seed run-to-run 最大差：n=253 S 0.040、n=2352 S/X ≤0.012；full-n 结论稳，small-n 噪声与效应同量级 |
| Phase 1b 协议/代码 | 完成：S0/R1/M1/M2/G_ref、ZINC 内部 8k/1k/1k、Lipo 复用、label vault、23/23 CPU gates、7 tests、broken-adjacency 重连 |
| Phase 1b GPU | **完成**：Phase A 10/10、B1 1/1、B2 1/1、D 2/2，全部 exit 0；1.601 GPU-h（占用 1.642 h），20% 预算 |
| Phase 1b 结果 | ZINC：S0 0.3173 / R1 0.2602 / M2 0.2522 / G_ref 0.1653；Lipo：S0 0.5610 / R1 0.5521 / M2 0.5470 / G_ref 0.5644；两任务均超常数基线 |
| Phase 1c 协议/代码 | 完成：S_cap_size / R1_size / R1_marginal_size / M_keep_edge；ZINC 8k/1k/1k；17/17 CPU gates；18 tests；data isolation PASS |
| Phase 1c GPU | **完成**：10/10 run JSON exit 0；0.981 GPU-h run-level（1.003 h 占用），≤2 h；Phase A 0.474 h、B 0.225 h、C 0.281 h |
| Phase 1c 结果 | Gain_pair +0.0682/+0.0309（均值 +0.0495）；Gain_over_capacity +0.0529；R1_size 最好 seed 0.2366，G_ref gap +0.0713；M_keep 0.2370 vs R1_size 0.2366（同配方中性，匹配计算对照被优化混淆） |
| Phase 1d 协议/代码 | 完成：R_incidence / R1_size_cap / R_incidence_cap（后者未跑）；ZINC 8k/1k/1k；19/19 CPU gates；25 tests；data isolation PASS；Phase A 10k 结构特征表 |
| Phase 1d GPU | **完成**：3/3 run JSON exit 0（1 smoke + 2 full）；0.4642 GPU-h run-level（0.4706 h 占用），≤2 h；B 0.1706 h、C 0.2935 h |
| Phase 1d 结果 | R_incidence 0.2797；B Gain_joint +0.0431（CI +0.0121…+0.0763），但 96% 互补组 −0.0281、4% 大环 +1.751；R1_size_cap 0.1935，Gain_cap +0.0431（CI +0.0192…+0.0688），G_ref gap +0.0282；96% gap +0.0174、4% tail gap +0.286；FLOPs 2.10× |
| Phase 2a 协议/代码 | 完成：`LocalConfig`/`NodeFFN2`/`LocalMP2`、有向边映射、参数与 FLOP 审计、9/9 gate 条目（覆盖 10 项预注册检查）、20 tests、冻结预测写入 |
| Phase 2a GPU | **完成**：8/8 run exit 0；0.933 GPU-h（占用 0.933 h），≤4 h；prep/smoke/score 合计 31 s，训练循环 0.908 h |
| Phase 2a 结果 | 两 seed 均值 MAE：P3E 0.090887 / F 0.087199 / M 0.085345；Gain_P3E +0.005675 / +0.005409；Gain_MP +0.003764 / −0.000055；957 补集 0.0770 / 0.0731 / 0.0712；43 受罚组 0.4007 / 0.4002 / 0.3998 |
| Phase 2a 判定 | **`CAPACITY_ONLY_INCREMENT`**：容量进益真实可复现，交互进益为零；seed 0 单独看会误判 M>F（F 的 seed-0 轨迹比自身 seed-1 差 0.0039） |
| Phase 2b 协议/代码 | 完成：`PROTOCOL.md`/`configs/phase2b.yaml` 预注册；P3 复用 audit 10/10、CPU gates 10/10（G4 容差 1e-4 处理远端 float 噪声）、FLOPs 1.147×/1.151×、16 tests；train 记录全部 init/encoder/batch-order 精确匹配 |
| Phase 2b GPU | **完成**：smoke + F s0/s1 + M s0/s1 + one-shot test forward；4/4 formal exit 0；0.947 GPU-h（训练循环 0.929 h），≤2.5 h；无 GPU 失败 |
| Phase 2b 结果 | valid P3 0.079844 / F 0.079616 / M 0.076788；test P3 0.069822 / M 0.068549；M gain valid +0.003056、test +0.001273（CI 跨零）；F valid +0.000228（不 promote）；M 唯一 finalist |
| Phase 2b 判定 | **`NO_PRACTICAL_ADVANTAGE`**：valid 仅阈值边缘通过，one-shot test 未确认；M 的 +12.3% 参数 / +15.1% FLOPs / +38% 训练墙不换来确定收益；M/F 冻结保留为 borderline/negative 证据 |
| Phase 2c-sup（`zinc_support_transfer_probe_v1`）协议/代码 | 完成：`PROTOCOL.md` 冻结（commit `b0e287f`）+ 两处修订 A1（标准化尺度下限/头 lr/选择时机）、A2（C_dev trim5 选择统计量，均在见到新数值前写入）；新 honest holdout（F 7467 / C_fit 1156 / C_dev 306 / Q_A 361 / Q_B 310 / Q_bg 400），六条隔离门禁全过；7 tests；B/G/S 三臂、末层零初始化、总参数 424066/423986 ≤ 500000 |
| Phase 2c-sup GPU | **完成**：2/2 formal exit 0（`b652cd73e89f` 提交，a100-2，driver 550.163.01，torch 2.5.1+cu124）；**0.488 GPU·h**（s0 0.2194 / s1 0.2689），另两次启动因远端缺 git-ignored manifest 在 <1 min 内失败；3.0 h 预算的 17% |
| Phase 2c-sup 结果（修正后） | 24/24 个头（G/S × 2 wd × 3 init × 2 base seed）在 C_dev 上**全部选中 step 0**（=不纠错），**保留完整曲线的 12 个臂未裁剪 MAE 同样在 step 0 最小**（所以不是 trim5 造成的；另 12 个 wd=1e-3 臂只存了 trim5 水平）；选中臂与 B 逐分子相同（Δ=0.0000）；头确实在拟合（C_fit L1 0.120→0.027/0.054）但 C_dev 未裁剪 0.114→0.166/0.180、trim5 0.092→0.147/0.158；强行用到 1500 步则 Q_A +0.041/+0.036（s0）、Q_B +0.071/+0.122、Q_bg +0.012/+0.013（CI 在有害侧排除 0）；**S vs G 无一致符号**（Q_A 上 S 略好、Q_B 上更差）；oracle 整包平移上限 0.0011 < 目标所需 0.003；旧包 12/12 同向 |
| Phase 2c-sup 更正 A3/A4/A5（0 GPU） | **A3**：`n1 = sup2[atom_k1]` 把半径 1 的 ID 拿去查半径 2 的支持表（`zst_head.Features` 与 `zst_common.per_atom_support`），G/S 都受影响；修正为按分子计数 k1，新增"重编号不变"（修正后 0.0 / 修正前 4022）与"逐分子暴力计数"（0.0）两条检查；**复用原两个基座、原划分、原冻结预测重训全部 24 个头（CPU）**，选择结论不变。**A4**：初版"同键族内部也不可预测"与已保存曲线不符（seed 0 的 S 未裁剪在 step 50 为 0.0878 < 基座 0.0966），撤回，改为"选择器 4/4 选 0；未裁剪仅在 1/4 格下探且不跨 seed 复现"。**A5**：撤回"条件均值≈0"与"所有可学习纠错上限 0.0011"，改为限定陈述（L1 对应条件中位数；oracle 常数只约束整包平移；零头配对差是同义反复） |
| Phase 2c-sup 判定（限定） | **`NO_TRANSFERABLE_CORRECTION_FOR_THIS_HEAD_FAMILY`（分支 D，限定）**：在"加性 + m_F 门控 + 这组特征 + L1 + C_dev(trim5)"这一族里没有跨结构收益；未裁剪曲线同向；旧包与另一 seed 同向。**该判定不外推到其它纠错族**。§5 全 10k valid 门禁未开、official valid/test 未读、shuffle 控制未做（S/G 符号混合时无解释力）。初版 `NO_TRANSFERABLE_CORRECTION__RESIDUAL_IS_LABEL_NOISE_AT_THIS_RESOLUTION` 保留在历史（commit `ccd6110`）中 |
| `zinc_rooted_moment_v1` 协议/代码 | 完成：`PROTOCOL.md` + `configs/rmc_v1.json` 冻结（`0255e32`）；精确 rooted 矩恒等式 `F=μ+(2aC2q+C3[q,q])/Z`；23 项 CPU gate + 5 个单测；缓存逐字节可复现（本地 = res，跨 worker 数/线程数） |
| `zinc_rooted_moment_v1` GPU | **完成**：6/6 formal exit 0（`d1faaa2`，a100-2，driver 550.163.01，torch 2.5.1+cu124）；**2.546 GPU·h formal / 2.564 含失败**，上限 5.0；两个 A100 并行，墙面 2 h 28 min |
| `zinc_rooted_moment_v1` 结果 | best-valid / T100：P 0.079472/0.081451（s0）、0.079627/0.080546（s1）；M 0.088184/0.089459、0.086726/0.087902；G 0.087109/0.090084、0.085643/0.086843。M−P T100 **+0.008008 / +0.007356**；G−P **+0.008634 / +0.006297**；M−G **−0.000626 / +0.001059**（符号翻转，mean +0.000216） |
| `zinc_rooted_moment_v1` 判定 | **`NO_STABLE_GAIN_OVER_P`**：矩读出的额外通路本身约 −0.0077 MAE；C3 偏斜项在重新训练下无重复贡献（固定权重下仅 +0.0003…+0.0005）；ordinary/unseen/tail 全负，drop-row172 与 drop-top10 同向；official test **未读**（触发 false） |
| `zinc_performance_search_v1` 协议 | 完成：`PLAN.md` + `DECISIONS.md`（D1–D7）+ `LEDGER.md`；预注册 A/B 两主体 × R0–R3 配方 + 一个针对性额外配置 R0L；C 主体按"先计时、单 run ≤2 h"约束执行 |
| `zinc_performance_search_v1` 工程 | GRIT 10 层/64 宽/8 头（473,473 par，RRWP 1..20）稠密 padded 端口实现并验 parity（forward 1.7e-6、grad 9.6e-5、RRWP 2.4e-7）；实测 30.97/18.65/14.84 s/epoch（bs 32/64/128）→ ≥4 h/run，**超出 2 h 上限**，按预案回退 DBT；EMA 策略因拷贝方向 bug 整轮作废（已修，未重跑） |
| `zinc_performance_search_v1` 结果 | 冻结 **B-R0**（DCC-L 494,193 par，R0）：valid 0.075990±0.002137（4 seeds），**test 0.067319±0.000709，range 0.066267–0.067785**；A-R0 基线 4 seeds valid 0.080761±0.003033、test 0.070879±0.001830（B 的 4 个 seed 在 test 上全部优于 A 的 4 个）；A-R1 valid 0.087570/test 0.075147；B-R1 0.079246/0.069022；A-R0L 0.079491/0.071844；C-DBT600 0.082479/0.071329 |
| `zinc_performance_search_v1` 判定 | **`TARGET_NOT_MET_BEST_TEST_0.066267`**：单模型 test 0.060 未达到（best 0.066267，差 0.006267）；所有已试杠杆（lr/wd/horizon/第三主体/双通道）收益 ≤0.005；官方 test 于冻结短名单后一次性读取（14 条，valid 复核 14/14），之后无调参；总 8.39 GPU·h（含 0.97 失败/取消） |
| `zinc_relation_composition_probe_v1` 协议/代码 | 完成：`PLAN.md` + `configs/probe_v1.json`；只比较**关系状态写入规则**（D：写入依赖当前键 token 状态；S：只依赖固定结构），两支参数/张量/初始化/配方逐位相同、都用真实 `xi` 做 token 更新；键上 lifted RRWP `R_bc[k]=(M P^k M^T)_bc`（无标签、train/valid 离线预算、可重建）；**7/7 工程套件 PASS**（关系对四端点枚举 2.9e-8、D/S 初始状态逐张量相同、旧 513 宽读出层不存在、机制/掩码/数值/checkpoint）；470,321 参数 ≤500k |
| `zinc_relation_composition_probe_v1` GPU | **完成**：2/2 formal exit 0（`7464c0d30441`，a100-2，两张 A100 各一 run 并行）；实测 **6.68 s/epoch → 1.113 GPU·h / 600-epoch run**；按事前代价规则第 3 级**只跑 seed 0 的 D/S 对**（4 run 在预算内放不下）；训练 **2.223 GPU·h**（含计时/诊断共 2.240），墙面约 3.6 h（超 3 h 硬上限约 30–40 min：实现阶段 + 远端部署阻塞 + 两处 runner 修复，均非额外 GPU 工作；未因此补跑任何 run） |
| `zinc_relation_composition_probe_v1` 结果 | best-valid / W20 / T100 / final：D 0.084493 / 0.094786 / 0.100461 / 0.095439；S 0.081838 / 0.103041 / 0.110148 / 0.099211；B-R0（历史参照）0.078537 / 0.079082 / n/a / 0.079076。**D−S 符号在行间翻转**：best-valid +0.002655（偏 S），W20 −0.008255、T100 −0.009686、final −0.003772（偏 D）；配对 bootstrap CI 跨 0（[−0.0038, +0.0097]，D 更好的行占 0.492）；两支在稳定量上都比 B-R0 差（D +0.0157、S +0.0240） |
| `zinc_relation_composition_probe_v1` 判定 | **`NO_STABLE_D_S_DIFFERENCE_AND_WORSE_THAN_B_AT_THIS_HORIZON`（建议停止这条线）**：单 seed 无法在任一方向分离 D/S，且新主体整体不如已确认的 B-R0。**但「机制未接上」被排除**：固定权重下只把 D 的关系写入换成 S，valid 0.084493→0.122155（**+0.0377**，mean \|Δpred\| 0.085；未训练对照只有 2.5e-4）；本地重前向复现报告 best-valid 差 4.06e-09。D 的 train@best 0.0236 远低于 S 的 0.0535 → 更像**主体在该日程/容量下不划算（拟合进 train、不转移到 valid）+ valid 曲线不稳定**，而非「关系没被使用」。official test **未加载未评分**，0.05× 目标未评估 |
| `zinc_relation_composition_probe_v1` 更正/限制 | 单 seed（seed 0）边界；600 epochs 是声明探索日程而非 GRIT 原生配方；B-R0 为历史参照（非本轮重训、非同配方对照）；行 bootstrap 只描述 1000 分子样本、不代表训练噪声；墙面比 3 h 硬上限超出约 30–40 min（来自远端部署阻塞与两处 runner 修复，不是额外 GPU 工作） |
| `zinc_relation_bn_diagnostic_v1` 协议/代码 | 完成：`PLAN.md` + 一个复用源模型的诊断脚本（无优化器/无 backward/不改参数）；只改派生副本的 BN `running_mean/var`；四种状态 O（原 eval）/B（batch 统计、不写 buffer）/T（B+原 attention dropout，门控）/C（train-only 逐层重估）；校准按真实 BN 调用顺序（11 个 BN）逐个做，FP64 Chan/Welford 流式合并、真实条目加权、`running_var=M2/(n−1)`；official test 未构造未加载，valid 只做最后评估 |
| `zinc_relation_bn_diagnostic_v1` GPU | **完成**：`bn-D`(res-gpu0, 97.5 s) / `bn-S`(res-gpu1, 70.5 s) 均 exit 0；**0.047 GPU·h**（上限 0.5），墙面约 0.75 h（含一次被终止的 CPU smoke 误判、一次部署重试与写作） |
| `zinc_relation_bn_diagnostic_v1` 结果 | **B 完全解释 train 侧差距**：train MAE D_best 0.023578→**0.007683**、D_final 0.033238→**0.007610**、S_best 0.053499→**0.009829**、S_final 0.108899→**0.007541**（在线 0.0082–0.0091，\|差\|≤0.0011）；T 与 B 同量级（0.0084–0.0101）⇒ dropout 不贡献。**C 不能恢复 train**（0.0318/0.0318/0.0749/0.0700），valid 变化符号混合（−0.0131/−0.0005/+0.0048/+0.0108），被旧 BN 选出的 best 反而变差。逐层看 11 个 BN 中 10 个的 running 统计与 train 全局矩已一致（方差比中位 0.991–1.001），只有 `layers.0.bn_h1` 明显陈旧（均值漂移 0.50/最大 1.70、方差比低至 0.45）⇒ 不是"统计量估错"，而是**逐 batch 归一化本身是学到的函数** |
| `zinc_relation_bn_diagnostic_v1` 判定 | **`BN_BATCH_STATISTICS_EXPLAIN_THE_TRAIN_GAP__GLOBAL_RECALIBRATION_RECOVERS_NEITHER_TRAIN_NOR_VALID`（诊断结束，停止）**：train/eval 差距 100% 定位到 BN 统计来源，但**没有可修的推理统计损失**——重估全局矩既不恢复 train 也不一致改善 valid（最优 S_best 0.0818→0.0866 变差）。上一轮 D/S 判定不变（eval 口径 valid 数字未被污染，剩余差距是泛化差距）；上轮 D→S 固定权重干预的 +0.0377 可能含激活统计失配，按要求未重做。检查全过：O 重放差 ≤1.3e-08、参数 hash 一致、仅 BN running 变化、逐层自洽 6.3e-08、重分批 1.4e-06、源 checkpoint 未改 |
| 交付 | Phase 0 REPORT + 纠错文件 + Phase 1a 协议/REPORT + Phase 1b 协议/决策日志/REPORT/run manifest + Phase 1c 协议/纠错/REPORT/run manifest/analysis + Phase 1d 协议/REPORT/CPU audit/run manifest/analysis + STATE/TRACK |

## 入口

| 路径 | 用途 |
|------|------|
| `notes/research_core.md` | 权威定义（已加纠错 banner）、toy 结论、真实数据诊断 |
| `notes/phase0_corrigendum.md` | **Phase 0.5 纠错优先文件**：三层概念、风险符号、pilot 方向、文献 |
| `notes/prior_work_boundary.md` | 文献审计表；ESA+/Shen&Dong 增补 |
| `notes/phase1a_protocol.md` | Phase 1a 冻结协议、门禁、端点、判定规则 |
| `configs/phase1a.yaml` | 冻结配置/hash/预算 |
| `results/phase0/REPORT.md` | Phase 0 完整报告（顶部有纠错链接） |
| `results/phase1a/split_phase1a.json` | 冻结 split（structure-only） |
| `results/phase1a/REPORT.md` | Phase 1a 结果报告（主表、CI、判定、限制） |
| `results/phase1a/run_manifest.tsv` | R1 18 条 run 的 rr run_id/时间/占用 |
| `results/phase1a/run_manifest_r2.tsv` | R2 clean-worktree 18 条 run |
| `results/phase1a/analysis_summary.json` | R1 无逐行标签的汇总分析 |
| `results/phase1a/analysis_summary_r2.json` | R2 汇总分析 |
| `results/phase1a/analysis_replicates.json` | R1+R2 pooled/复现性分析 |
| `notes/phase1c_protocol.md` | Phase 1c 冻结协议、门禁、端点、预算、判定规则 |
| `configs/phase1c.yaml` | Phase 1c 冻结配置/hash/预算 |
| `results/phase1c/REPORT.md` | Phase 1c 完整报告（7 问、CPU 诊断、主表、限制、下一步） |
| `results/phase1c/run_manifest.tsv` | Phase 1c 10 条 run 的 rr run_id/GPU 秒/output hash |
| `results/phase1c/analysis_phase1c.json` | Phase 1c 主分析（配对 bootstrap、seed 复现、M_keep 诊断） |
| `results/phase1c/run_validation.json` | 数据隔离/provenance 校验（PASS） |
| `results/phase1c/cpu_gates_phase1c.json` | Phase 1c 17/17 CPU gates |
| `results/phase1c/readout_diagnostics.json` | CPU 读出/尺寸/环诊断 |
| `runs/phase1c/` | 逐 run JSON（Monitor 预测；git-ignored） |
| `code/phase1c/` | 模型、训练 runner、分析、gates、tests |
| `results/phase2b/PROTOCOL.md` | Phase 2b 预注册协议（standard ZINC-12k、P3 复用、F/M 四跑、valid promotion、one-shot test） |
| `results/phase2b/REPORT.md` | Phase 2b 完整报告（P3 复用 audit、valid/test 主表、7 问、成本、GPU 账本、最终 verdict） |
| `results/phase2b/p3_reuse_audit.json` / `cpu_gates_phase2b.json` / `flops_phase2b.json` | P3 复用 audit 10/10、CPU gates 10/10、FLOP 审计 |
| `results/phase2b/valid_decision.json` / `valid_metrics.json` / `valid_prediction_freeze.json` | valid 单次读标签、promotion 判定（M）、冻结预测 hash |
| `results/phase2b/test_decision.json` / `test_metrics.json` / `test_prediction_freeze.json` | one-shot test 判定（未确认）与冻结预测 hash |
| `results/phase2b/train_*.json` / `valid_freeze_*.json` / `soup_*.pt` | 四跑训练记录、valid 预测 freeze、soup（hash 已入报告） |
| `results/phase2b/gpu_ledger.json` / `run_manifest.tsv` | GPU 账本（0.947 h）与 rr run_id/commit/hash 清单 |
| `code/phase2b/` | 复用 audit、gates、R3 runner、valid/test 评分、16 tests |
| `notes/phase2a_protocol.md` | Phase 2a 冻结协议（臂定义、参数匹配、10 项 CPU gate、判定带、seed-1 购买规则、no-go 清单） |
| `results/phase2a/REPORT.md` | Phase 2a 完整报告（主表、子群、尾部诊断、8 问、决策链、GPU 账本、复现命令） |
| `results/phase2a/seed1_purchase_record.md` | seed-1 购买记录（六条件核对、RMSE 门禁 miss 与其构成、披露的事后判断、停止规则），写于任何 seed-1 训练之前 |
| `results/phase2a/analysis_phase2a.json` / `analysis_seed1_phase2a.json` | 逐 seed 主分析（指标、分组、配对对比、bootstrap） |
| `results/phase2a/decision_phase2a.json` / `decision_seed1_phase2a.json` / `round_verdict_phase2a.json` | 机械判定产物：seed 0 `INSUFFICIENT_EVIDENCE`（冻结规则原样保留）/ seed 1 与轮次 `CAPACITY_ONLY_INCREMENT` |
| `results/phase2a/posthoc_phase2a.json` | 尾部事后诊断（MSE 差 CI、单分子贡献份额、图函数补集 RMSE） |
| `results/phase2a/run_manifest.tsv` | Phase 2a 8 条 run 的 rr run_id / commit / GPU 秒 / artifact sha256 |
| `results/phase2a/cpu_gates_phase2a.json` / `flops_phase2a.json` / `replay.json` | CPU gates、FLOP 审计、P3E 复现（max|d| 9.537e-07） |
| `code/phase2a/` | 模型、runner、seed-1 评分、事后诊断、tests |
| `notes/phase1e_protocol.md` | Phase 1e 冻结协议（A1–A4 / B1–B3 / C / 路线表、阈值、post-hoc 纪律声明） |
| `results/phase1e/REPORT.md` | Phase 1e 完整报告（6 问回答、基准缺陷、路线、非声明） |
| `results/phase1e/label_audit.json` | A1 标签机制（混淆矩阵、重建、置换稳定性） |
| `results/phase1e/error_attribution.json` | A2 分组误差归因与对 G_ref 的配对对比 |
| `results/phase1e/zero_training_routing.json` | A3 零训练条件路由诊断（post-hoc） |
| `results/phase1e/reference_audit.json` | B1 四类协议分离表 + 污染审计 + B2 预算 |
| `results/phase1e/strong_baseline.json` | B2 严格隔离强基线结果与 bootstrap |
| `results/phase1e/decision.json` | A4 规则判定与路线选择（含阈值记录） |
| `results/phase1e/phase_c_design.md` | Phase C 任务选择与预注册设计（仅设计，未训练） |
| `results/phase1e/run_manifest.tsv` | Phase 1e 工件 sha256 + 代码 commit |
| `notes/phase1d_protocol.md` | Phase 1d 冻结协议、Phase A 事实、B/C 判定规则、预算 |
| `configs/phase1d.yaml` | Phase 1d 冻结配置/hash/端点/预算 |
| `results/phase1d/REPORT.md` | Phase 1d 完整报告（A/B/C、尾部分解、限制、下一步） |
| `results/phase1d/analysis_phase1d_cpu.json` | Phase A 逐分子描述性审计 |
| `results/phase1d/structural_features.npz` | 10k 结构-only 特征表 |
| `results/phase1d/analysis_phase1d.json` | Phase B/C 主分析（配对 bootstrap、subgroup、cost） |
| `results/phase1d/run_manifest.tsv` | Phase 1d 3 条 run 的 rr run_id/GPU 秒/output hash |
| `results/phase1d/run_validation.json` | 数据隔离/provenance 校验（PASS） |
| `results/phase1d/cpu_gates_phase1d.json` | Phase 1d 19/19 CPU gates |
| `runs/phase1d/` | 逐 run JSON（Monitor 预测；git-ignored） |
| `code/phase1d/` | 模型、训练 runner、分析、gates、tests |
| `runs/phase1a/` | 逐 run JSON（含 Monitor 预测，git-ignored） |
| `code/phase1a/` | S/X/G、数据、训练、CPU gates、runner、tests |
| `results/phase0/code/ssrc_toy.py` | 精确 \(E_r\)、pynauty/exhaustive/networkx 交叉验证、oracle |
| `results/phase0/tests/` | 17 个 Phase 0 CPU 测试 |
| `docs/tracks/structural-sufficiency.md` | 站点入口 |

## 禁止

- 与其它轨结果混表横比（除非单独 deliverable 写清协议）。
- 在 `tracks/` 之外另立仓库扩张本课题。
- 用官方 test 数字做路线选择；用 valid 数字反复调参后仍称“预注册”。
- 把 toy 的 \(E_2\) 碰撞外推为“真实分子上 MP 必要/不必要”。
- 把一次 pilot 的 null 结果包装成“机制不存在”的普遍结论。
- 把 Phase 2a 的 +0.0055 写成“局部传播/消息传递机制”的进益：它是 +12% 参数换来的容量进益，GINE 版与 FFN 版无可分辨差异。
- 在 n=1000、max|y|≈8 的 Monitor 上再用**绝对 RMSE 阈值**做稳定性门禁（单个分子即可翻转符号）；改用配对 MSE 差 CI 或图函数补集。

## 下一步

00. **`zinc_rooted_moment_v1` 改写下一步（历史，2026-10-23）**：**精确 rooted 矩条件化**这条线在本预算/本读出形式下关闭。
   - 事实：M−P T100 = +0.008008 / +0.007356（两 seed），G−P = +0.008634 / +0.006297；M−G 符号翻转（−0.000626 / +0.001059）；ordinary、unseen+ordinary、tail 全负；drop-row172/drop-top10 同向；C3 在固定权重下仅 +0.0003…+0.0005，重新训练无可重复收益。
   - 因此**不**再试：尺度集、ε、读出宽度/深度、更多 seed、更多视图、把 C3 换成更高阶矩或换基函数。
   - **唯一允许的下一步**（若重开）：做一个与 P **参数匹配且 step 0 逐位等价**的注入（例如把矩读出经**零初始化标量门**接入现有 P3 reader），先证明“不付通路代价”再谈矩的信息价值；两 seed、T100 主端点、未达冻结触发不读 test。
   - 会改写判定的证据：零初始化门版本下 M−P 两 seed T100 同向改善且均值 ≥0.003，且 M−G 至少不反向。

00a. **`zinc_support_transfer_probe_v1`（含 A3/A4/A5 更正）已改写下一步（历史，2026-10-23）**：raw-y 上的**加性纠错头/残差后处理**这条线**关闭**。
   - 事实（修正后）：24/24 个头在 C_dev 上选"不纠错"，未裁剪曲线同样在 step 0 最小；强行应用在三个包上一致变差（+0.012…+0.122）；S 与 G 无一致优劣；旧包 12/12 同向；oracle 整包平移上限 0.0011 < 0.003。k1 特征 bug 修好后结论不变。
   - 因此**不**再试：不同头容量/更多 init/更多 seed/更多 wd/更细的原子特征/per-k2 表/把纠错头接到别的 checkpoint/换成分组或交叉拟合的残差挖掘。两次在同一位置失败，剩余空间（整包平移 0.0011）低于目标所需 0.003。
   - **下一次 GPU 投入的方向**：端到端重新学习结构表示（能直接表达"未见键类型"的表示），在与 official 10k 一致的协议上评估；而不是继续从冻结 P3 的残差里找突破。
   - 会改变判定的证据：在**完全不含 query 标签**的固定规则下，query MAE 改善 ≥0.003、键簇 bootstrap 区间不含 0、并在第二个基座 seed 上复现。这需要新表示，不需要新头。

00. **Phase 2b 已改写下一步（历史）**：标准 ZINC-12k 上 M 的 valid +0.003056 未在 one-shot test 重现（+0.001273，CI 跨零），F 为 null；**M 不作为下一性能基座，P3 保持基座**。M/F 冻结保留为 borderline/negative 证据。若重启该线，唯一允许的科学问题是：**3–5 seed + 参数匹配 trunk/readout 对照**，检验 M 的 valid gain 与 M−F separation 是否可复现，还是 seed-1 波动；必须新预注册，不自动开 Phase 2c。

00b. **Phase 2a（历史）已改写下一步**：ZINC 上“在强编码器后追加局部传播”这一线**停止**。
   - 事实：+12% 参数的小模块在冻结 P3E 之后稳定带来 +0.0055 MAE（0.0911→0.0853，两 seed 一致）；但把该模块换成真实边上的 2 轮 GINE 没有额外收益（`Gain_MP` +0.0038 / −0.0001，符号跨 seed 冲突）。Phase 1b 的“迭代≈计算量”与本轮“邻接≈无邻接”是同一结论的两种形式。
   - 因此**不**再试宽度/深度/gate 初始化/更多 seed/新 SSL 目标/cycle 路由/ensemble。
   - 若仍要检验交互价值，必须是**新一轮预注册**，且满足：目标依赖 P3E 感受野可证不足的图统计量（先做零训练诊断）；每臂 ≥2 seed；尾部用鲁棒判据；否则不动 GPU。
   - Phase C（Lipophilicity）设计仍冻结未训练，优先级高于任何 ZINC 追加实验；但若做 Phase C，必须先把其 RMSE/尾部判据改成配对 MSE 差形式。

000. **`zinc_relation_bn_diagnostic_v1` 改写下一步（最新，2026-10-24）**：BN 推理统计**不是**上一轮 valid 差距的原因。
     train 侧的 eval-mode 误差 100% 由"BN 用 running 统计而非 batch 统计"决定（B 把四份 checkpoint 全部拉回在线水平，
     |差|≤0.0011；T 说明 dropout 无贡献），但用 train-only **逐层**重估全局矩（C）既不恢复 train（0.032–0.075 vs 0.0075–0.0098），
     也不一致改善 valid（−0.0131…+0.0108，best 反而变差）。逐层证据显示 10/11 个 BN 的 running 统计本就接近精确全局矩
     （只有 `layers.0.bn_h1` 陈旧）⇒ 差距来自"逐 batch 归一化是学到的函数"，不是"统计量估歪"。
     结论：**不存在可修的推理统计损失**；valid 差距是泛化问题，本轮没有给它新的杠杆。
     下一步**停止**：不要为此加 seed、换归一化类型或加长日程；若将来重启，只能作为"训练时归一化是否需要改变"的新预注册问题，
     并且必须重新训练（本轮是固定权重诊断，不能外推）。另：上轮 D→S 固定权重干预(+0.0377)可能含激活统计失配，未重做。

00. **`zinc_relation_composition_probe_v1` 改写下一步（历史，2026-10-24）**：把**关系状态的写入规则**条件化到当前键 token 状态（D），
     在同一日程/同一参数量级下**没有**买到相对"只由固定结构写入"（S）的收益，也整体不如已确认的 B-R0。
     seed 0 单对上：best-valid 偏 S（+0.002655），W20/T100/final 偏 D（−0.0083/−0.0097/−0.0038），配对 bootstrap CI 跨 0；
     两支在稳定量上分别比 B-R0 差 +0.0157（D）/ +0.0240（S）。**"机制没接上"这个解释被排除**：固定权重下把 D 的写入换成 S，
     valid 从 0.084493 崩到 0.122155（+0.0377），说明关系状态确实被后续层读出；同时 D 的 train@best 0.0236 远低于 S 0.0535，
     即 D 把容量拟合进了训练集却没有转移到 valid。
     因此这一格子里**最可疑的不是主体而是日程/稳定性**：本轮按设计移除了 per-graph 归一化、core attention dropout 0.2，
     两条 valid 曲线的 W20 都比各自 best-valid 高 0.010–0.021，best-valid 只是"幸运 epoch"。
     下一步只允许：(a) 先建立**稳定**的 valid 曲线（恢复 per-graph 归一化或等效稳定化）再重做 D/S 对照，且要求两支 train@best 同量级；
     (b) 若不做 (a)，则关闭该主体，不再加 seed、不再调 lr/wd/horizon。**不要**在抖动曲线上继续解读 D/S 的 0.003 级差异。

00. **`zinc_performance_search_v1` 改写下一步（历史，2026-10-23）**：在"≤500k 参数 / 单模型 / ~0.35 GPU·h per run"的格子里，
    性能优先搜索**没有**把 ZINC test 推到 0.060（最好 B-R0 = 0.066267，4 seeds 0.0663–0.0678；A-R0 0.0709；C/DBT 0.0713）。
    已试杠杆（R0/R1、1200-epoch 长日程、双通道 composer、第三主体）单点收益都 ≤0.005，而差距是 0.0063。
    下一轮只允许两条路：(a) 给 GRIT/DBT 原生 horizon 的预算（GRIT 实测 ≥4 h/run 稠密、稀疏 ~9.5 s/epoch；DBT 全 horizon 3.7 GPU·h）；
    (b) 换设定（更大模型 / 更多数据 / 不同 readout）。**不要**在同一格子里继续调 lr/wd/horizon。
    附带结论：EMA 策略本轮作废（实现 bug，已修未验）；valid 名次不能替代 test 名次（seed 级反例在本轮报告 §4）。

0. **Phase 1e 已改写下一步**：ZINC 大环线的收益被证明主要是标签 `-cycles` 人工项；SSRC 臂族在同 split 上与隔离强基线差 0.074 MAE。因此：
   - **停止** SSRC/R1 架构线在 ZINC 上的任何继续优化（宽度/seed/深度/global-token/迭代臂/ring-aware token 全部停止，理由已升级为基准缺陷 + 竞争力不足，而非"审计无 headroom"）。
   - 下一个科学步骤 = Phase C（Lipophilicity 冻结 scaffold split），设计见 `results/phase1e/phase_c_design.md`；分两层：Tier 1 家族内 H1/H2/H3（新增 `D8` 深度臂），Tier 2 在冻结强骨干上比较 `RO_pool` vs `RO_rel` 以判定 H2。总预算 ≈1.6 GPU-h。**必须先预注册再跑**。
   - 任何 ZINC 数字今后必须单列受罚子集，禁止用聚合值做跨轮比较。

1. ~~不再跑 GPU 宽度/seed/深度/global-token/迭代臂~~（已被 Phase 1e 升级，保留历史）。Phase 1d 已显示：additive incidence 在 96% 普通分子上足够或略好；joint pairing 与 capacity 的增益集中在 4% 大环尾部；R1_size_cap 整体仍比 G_ref 差 +0.0282 MAE，且 FLOPs 为 2.10×。继续加宽或加 seed 不回答剩余问题。
2. **唯一允许的下一步**：CPU-only 大环可预测性审计。用冻结的 R1_size / R1_size_cap / G_ref 逐分子预测与 Phase 1d 结构-only 特征，检查 40 个大环分子是否存在 patch 可观测 headroom、误差是否由 cycle-size/ring-fusion 特征驱动，以及 ring-aware local token 是否可能在不破坏 96% 互补组的前提下有帮助。该审计不训练模型、不选模型。
3. 判定/停止：
   - 若审计没有预注册的 **≥0.02 MAE** 机会：**停止 SSRC 架构线**，把 Phase 1d 写成 tail-specific mechanism + absolute-gap negative 的最终报告；不追加模块。
   - 若审计显示 ≥0.02 MAE 机会：只允许在**下一轮**预注册**一个** ring-aware local representation 诊断；不得做宽度/深度/半径/seed grid；若互补组退化，或尾部增益 <0.02 / CI 跨零，立即停止。
4. **不**读官方 valid/test 或内部 Reserved 做选择；**不**把单次 null 写成“消息传递/深度普遍无用”；**不**把 `R_incidence` 写成提出的新方法，也不把大环尾部收益外推为一般化学结论。
