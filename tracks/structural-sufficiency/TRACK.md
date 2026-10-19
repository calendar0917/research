# Track: SSRC — Structural Sufficiency & Relational Computation

| 项 | 内容 |
|----|------|
| slug | `structural-sufficiency` |
| 状态 | **active（主线）；Phase 0.5 纠错完成；Phase 1a 完成；Phase 1b 完成；Phase 1c 完成（关系对应通过容量/读出/边缘边际控制；绝对 G_ref 差距仍存；下一步是 CPU 残差审计）** |
| 创建 | 2026-10-18 |
| 主线位置 | **SSRC（新主线）**：显式结构充分性 → 关系计算；旧 `structure-contribution` 标记 **paused / 非主线**，全部历史保留，不覆盖、不改写 |
| Phase 0 结论 | **REFRAME_NEEDED**：原表述“MP 是否必要”已被文献覆盖且实验易混淆；重述为“精确 \(E_r\) 的信息充分性 vs 有限样本可学习性”，条件性 GO 到受控 pilot |
| Phase 0.5 | **纠错完成**：三层概念、风险分解符号、pilot 方向/停止规则、Eijkelboom train/test 口径、ESA+ 文献、S(G) oracle 降级；见 `notes/phase0_corrigendum.md` |
| Phase 1a | **COMPLETE（R1+R2）**：36/36 轨迹；n=2352 S 在 6/6 draws 优于 X（pooled Adv −0.0387），n=253 为 5/6（pooled −0.0230）；交互 +0.0157 但仅 3/6 draws 为正、CI 跨零。总 1.21 GPU-h |
| Phase 1a 判定 | **`NO_CLEAR_INTERACTION_VALUE_IN_THIS_PILOT`**；限定 `S_AT_LEAST_AS_GOOD`、`NO_STABLE_SMALL_N_X_ADVANTAGE` |
| Phase 1b | **COMPLETE**：5 臂 × 2 任务 seed 0；B1 broken-adjacency 控制；B2 matched-compute 控制；ZINC seed 1 复现。总 1.60 GPU-h |
| Phase 1b 判定 | **`PURSUE_RELATIONAL_SUFFICIENCY`**：S0→R1 ZINC +0.0571/+0.0992；broken-R1 更差 +0.0706；M1_long ≈ M2（Δ=−0.0027，CI 跨零）⇒ 迭代增益是计算量不是机制 |
| Phase 1c 判定 | **`CORRESPONDENCE_SURVIVES_CAPACITY` + `ABSOLUTE_GAP_PERSISTS`**：Gain_pair +0.0682/+0.0309（两 seed 均值 +0.0495）；Gain_over_capacity +0.0529；R1_size 最好 seed 0.2366 vs G_ref 0.1653（gap +0.0713）；M_keep_edge 0.2370 vs R1_size 0.2366（同配方中性；预注册匹配计算被优化混淆） |

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
| 交付 | Phase 0 REPORT + 纠错文件 + Phase 1a 协议/REPORT + Phase 1b 协议/决策日志/REPORT/run manifest + Phase 1c 协议/纠错/REPORT/run manifest/analysis + STATE/TRACK |

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

## 下一步

1. **不再加机制模块**：不加 global attention token、更深残差主干、新结构编码/距离通道、SSL，也不再加一轮 update。Phase 1c 的关系对应收益通过容量/读出/边缘边际控制，但绝对 G_ref 差距仍为 +0.0713 MAE（最好 seed）；同配方 M_keep_edge 0.2370 vs R1_size 0.2366，没有 update 收益。
2. **唯一允许的下一步**：CPU-only 残差审计。用冻结的 R1_size/G_ref 逐分子预测，把 `|err_R1|−|err_Gref|` 按 patch 可观测变量（n_atoms/n_bonds、环数、degree、bond-type counts、root/first-shell degree、patch size、重复 coarse key 数、patch-key entropy）分解；对比 S_cap / marginal，判断剩余差距是否集中在 patch token 信息不足的分子。
3. 判定/停止：
   - 若**没有** patch 可观测变量解释 **≥0.02 MAE**，或残差呈均匀/噪声状：**停止 SSRC 路线**，报告 absolute-gap negative；不追加模块。
   - 若某个变量解释 ≥0.02 MAE：只预注册**一个** compute-matched 诊断，只改该变量；增益 <0.02 MAE 或 CI 跨零即停止；不做机制 sweep。
   - `S_cap_size` seed 1 只是低优先级稳健性检查，不属于预注册 seed gate；在残差审计之前不跑。
4. **不**读官方 valid/test 或内部 Reserved 做选择；**不**把单次 null 写成“消息传递/深度普遍无用”；**不**把 `R1_marginal_size` 写成化学因果实验。
