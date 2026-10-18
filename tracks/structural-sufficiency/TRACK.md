# Track: SSRC — Structural Sufficiency & Relational Computation

| 项 | 内容 |
|----|------|
| slug | `structural-sufficiency` |
| 状态 | **active（主线）** |
| 创建 | 2026-10-18 |
| 主线位置 | **SSRC（新主线）**：显式结构充分性 → 关系计算；旧 `structure-contribution` 标记 **paused / 非主线**，全部历史保留，不覆盖、不改写 |
| Phase 0 结论 | **REFRAME_NEEDED**：原表述“MP 是否必要”已被文献覆盖且实验易混淆；重述为“精确 \(E_r\) 的信息充分性 vs 有限样本可学习性”，条件性 GO 到受控 pilot |

## 问题（一句话）

给定显式结构表示 \(E_r\)，学习式关系计算（消息传递、显式对象交互、迭代注意力）
什么时候是必要的——是在**补充 \(E_r\) 缺失的信息**，还是在信息不变的前提下
**改善可学习性/样本效率/泛化**？

## 范围

### 做

- 形式化区分 **问题 A（信息可辨识性：\(E_r\) 是否碰撞）** 与
  **问题 B（有限样本/预算下的可学习性与泛化）**。
- 用精确带属性 rooted-radius-\(r\) 诱导子图多重集 \(E_r\) 做 CPU 碰撞证书；
  精确同构/规范表示，不用普通哈希；节点重标号不变性；无标签/编号/种子泄漏。
- 文献边界审计：明确“强编码减少 MP 需求”**不是**本路线的新贡献。
- 真实分子可行性：Lipophilicity 优先（train-only 内部分割、可复用资产、
  公平对照、成本/FLOPs 报告）；ESOL 备选；ZINC-12k 只作辅助。
- 若重述被接受：一个**信息访问受控**的最小 GPU pilot，比较
  \(S(E_r)\), \(D(E_r)\), \(S(G)\), \(D(G)\) 四臂；参数/读出/预算匹配；
  预注册主端点与停止规则。
- 允许并鼓励有边界的负结果。

### 不做

- **不**重开旧轨 BJOC 已关闭的模块/配方搜索族（junction ownership、固定签名、
  STRC 加性残差、pair/距离通道、SSL 目标搜索、配方搜索、尾部加权）。
- **不**继承旧模型的硬约束（不一定非 GNN、不一定非消息传递；架构不预设）。
- **不**声称“结构编码减少消息传递需求”本身新颖（Eijkelboom 2023 已做）。
- **不**设计新架构作为本轮目标；pilot 只做受控机制比较。
- **不**做 ZINC 排行榜搜索、外部无标签预训练、或覆盖 frozen 结果。
- **不**读官方 valid/test/locked 标签做新选择；**不**使用 `res`。
- **不**把“训练集结构指纹唯一”当作“统计充分/可学习/可泛化”。
- **不**把 Dynamic 访问 `edge_index` 而 Static 只看无序 patch 多重集的差异，
  解释为“同信息下动态更强”。

## 协议 / 评估

- 复用：Lipophilicity DeepChem 2.8.0 scaffold split（train/valid/test =
  3360/420/420；2408 scaffold 组；跨集泄漏 0）；label-free `MolObjects`；
  scaffold 复现校验；label vault 隔离机制。
- 本轮只做 **train-only** 分析；官方 valid/test 标签未读。
- 候选内部 scaffold split（**未冻结，不是协议**）：2352/504/504，
  scaffold 组重叠 0。
- 未来 pilot 的 `protocol_id`：`ssrc-lipo-pilot-v1`（必须先冻结 row indices、
  hash、配置、主端点，再训练）。
- 主表：本轮无性能主表；Phase 0 的唯一数字源是
  `results/phase0/analysis/*.json` 与 `results/phase0/REPORT.md`。

## 进度

| 阶段 | 状态 |
|------|------|
| 问题与范围 | 完成（Phase 0 重述） |
| 精读 / 概念 | 完成（7 篇 + GSN；ESA+ 未唯一识别） |
| 最小实验（CPU） | 完成：17 tests；三家族 C24 碰撞；正对照；真实数据可行性 |
| 真实分子 train-only 诊断 | 完成：\(E_2/E_3\) 近唯一；scaffold 更统计高效 |
| GPU pilot | **未执行**；条件性预注册草案见 `results/phase0/REPORT.md` §7 |
| 交付 | Phase 0 REPORT + 本 TRACK + STATE + notes |

## 入口

| 路径 | 用途 |
|------|------|
| `notes/research_core.md` | 权威定义、问题 A/B、toy 结论、真实数据诊断、假设 |
| `notes/prior_work_boundary.md` | 文献审计表；已覆盖/部分覆盖/重复风险 |
| `results/phase0/REPORT.md` | Phase 0 完整报告 + 判断 + 条件性 pilot 草案 |
| `results/phase0/code/ssrc_toy.py` | 精确 \(E_r\)、pynauty/exhaustive/networkx 交叉验证、oracle |
| `results/phase0/code/run_phase0.py` | CPU 诊断驱动；输出 `analysis/phase0_results.json` |
| `results/phase0/code/lipo_feasibility_audit.py` | 真实数据/协议/成本审计（不读标签） |
| `results/phase0/code/train_only_er_information_audit.py` | train-only LOO/置换诊断（只读 train labels） |
| `results/phase0/tests/` | 17 个 CPU 测试 |
| `docs/tracks/structural-sufficiency.md` | 站点入口 |

## 禁止

- 与其它轨结果混表横比（除非单独 deliverable 写清协议）。
- 在 `tracks/` 之外另立仓库扩张本课题。
- 用官方 test 数字做路线选择；用 valid 数字反复调参后仍称“预注册”。
- 把 toy 的 \(E_2\) 碰撞外推为“真实分子上 MP 必要/不必要”。
- 把一次 pilot 的 null 结果包装成“机制不存在”的普遍结论。

## 下一步

1. **等待用户对重述的裁决**：
   - 接受“信息充分性 vs 统计可学习性” → 进入条件性 pilot 预注册；
   - 不接受 → 按 Phase 0 判断停止原表述（`STOP_THIS_FORMULATION`）。
2. 若接受：冻结 `ssrc-lipo-pilot-v1`（内部 split 行号/hash、四臂配置、
   参数/FLOP 匹配规则、5 seeds、主端点、停止规则）。
3. 实现 \(S_{\rm bag}, S_{\rm patch}, D(E_r), D(G), S(G)\)；先 CPU smoke test，
   再 1×A100 跑学习曲线（n=250/500/1000/2352 × 5 seeds）。
4. 主端点：\(D(G)-S_{\rm patch}(E_r)\) 在 n=250 的配对差，以及
   \(\Delta(250)-\Delta(2352)\) 的交互；所有结论带 seed 级 CI。
