# Track: SSRC — Structural Sufficiency & Relational Computation

| 项 | 内容 |
|----|------|
| slug | `structural-sufficiency` |
| 状态 | **active（主线）；Phase 0.5 纠错完成；Phase 1a 预 GPU 冻结** |
| 创建 | 2026-10-18 |
| 主线位置 | **SSRC（新主线）**：显式结构充分性 → 关系计算；旧 `structure-contribution` 标记 **paused / 非主线**，全部历史保留，不覆盖、不改写 |
| Phase 0 结论 | **REFRAME_NEEDED**：原表述“MP 是否必要”已被文献覆盖且实验易混淆；重述为“精确 \(E_r\) 的信息充分性 vs 有限样本可学习性”，条件性 GO 到受控 pilot |
| Phase 0.5 | **纠错完成**：三层概念、风险分解符号、pilot 方向/停止规则、Eijkelboom train/test 口径、ESA+ 文献、S(G) oracle 降级；见 `notes/phase0_corrigendum.md` |
| Phase 1a | **FROZEN_PRE_GPU**：同信息 S/X + G 参照；split 2352/504/504 + n=253；CPU gates 8/8；18 条轨迹、≤4 GPU-h |

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
| Phase 1a 协议/代码 | 完成：S/X/G、split、label vault、8/8 CPU gates、7 tests |
| Phase 1a GPU screening | **待执行**：18 条轨迹，≤4 GPU-h；结果后写 REPORT |
| 交付 | Phase 0 REPORT（含纠错链接）+ 纠错文件 + Phase 1a 协议 + STATE/TRACK |

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
| `results/phase1a/REPORT.md` | Phase 1a 结果报告（GPU 后生成） |
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

1. **提交 Phase 0.5 + Phase 1a 冻结**：纠错文件、协议、config、split、代码、
   CPU gates、tests。
2. **在 `res` 上执行 18 条探索性轨迹**（S/X/G × n=253/2352 × seeds 0/1/2），
   只读 Fit/Monitor；记录 GPU 峰值显存、step 时间、实际 GPU 秒数；≤4 GPU-h。
3. 汇总主端点 \(\mathrm{Adv}_{\rm set}(n)=\mathrm{MAE}_S-\mathrm{MAE}_X\) 与
   小样本交互 \(\mathrm{Adv}(253)-\mathrm{Adv}(2352)\)；按 scaffold cluster
   bootstrap 给条件性 CI；写 `results/phase1a/REPORT.md`。
4. 按预注册判定：`PROMISING_SAME_INFORMATION_INTERACTION` /
   `UNRESOLVED_INFORMATION_VS_INDUCTIVE_BIAS` /
   `NO_CLEAR_INTERACTION_VALUE_IN_THIS_PILOT` /
   `IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`。
5. **不**自动做新架构、追加 seed、扩大 n 或读 Reserved/test。
