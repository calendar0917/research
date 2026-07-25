# RW × KSVD 结构表征 — 调研报告（收口稿 v0.1）

> 面向导师讨论。数字均来自本轨 `results/`；协议见正文。  
> 详细文献：[notes/rw_survey.md](../notes/rw_survey.md) v0.2

---

## 1. 问题

**结构如何被显式、可还原地编码？**  
GNN 多隐式；GSN/CIN 显式但人设子结构；组内路线是 **字典学习（KSVD）** 从数据分解结构原子。

一阶邻域基线过粗、边冗余。导师提出：用 **随机游走** 构造待分解局部结构，再 KSVD。  
本调研回答：

1. RW 在图学习中有哪些成熟用法？我们占哪一格？  
2. 如何接到 KSVD 且保持「可还原」？  
3. 现有证据能证明什么、不能证明什么？

---

## 2. 相关工作定位（一表）

| 角色 | 代表 | 输出 | 我们 |
|------|------|------|------|
| 游走核 | RW / marginalized kernel | 图相似度 | 借「游走=结构指纹」；不走核+SVM |
| 游走嵌入 | DeepWalk, node2vec | 节点向量 | 借 **多 walk、\(p,q\)**；不要 Skip-gram |
| 子图采样 | GraphSAINT 等 | 诱导子图训 GNN | 借 **诱导 \(G[S]\)**；改为 KSVD 的 \(Y\) |
| 游走 PE | RWPE | 统计位置特征 | **不可还原**对照 |
| 显式结构 GNN | GSN, CIN | 计数 / cell MP | 同显式动机；原子 **学** 而非手数 |
| **本组** | RW→诱导→KSVD | \(D,X\)，节点 \(s_v\) | 可还原分解 + 结构通道 |

**定位句：** 不处在核/嵌入/PE/CIN 终局，而在 **采样形态（SAINT/n2v）× 可还原字典（KSVD）**。

精读：`docs/literature/deep/` 下 deepwalk, node2vec, graphsaint, rw_graph_kernel, rwpe, gsn, cin。

---

## 3. 方法（表征）

### 3.1 管道

对每个节点 \(v\)：采 \(r\) 个 patch（1-hop 或从 \(v\) 出发的 RW）→ 诱导子图 → 向量 \(y_i\) → 共享字典 \(D\) 上 \(x_i=\mathrm{OMP}(D,y_i)\) → \(s_v=\mathrm{pool}_i(x_i)\)。  
下游可：仅结构分类 / 与属性 concat / residual / gate 进 GIN。

### 3.2 关键实现选择

| 选择 | 理由 |
|------|------|
| 诱导子图而非纯路径 | 保留边关系，利于「结构块」 |
| 共享 \(D\)（train-only） | 每图字典使系数不可比（合成实验 0.55→0.86） |
| 节点级 \(s_v\) 而非图级广播 | 广播使全图结构通道相同，MUTAG 上严重掉点 |
| 多 walk + pool | 对齐 DeepWalk/n2v 习惯；单 walk 过噪 |

### 3.3 与 CIN 的关系

问题同类（显式结构，分子动机）。**实现与量级不同**（完整 cell GNN vs 结构通道探针）。**禁止**用 MUTAG 乐观数字与 CIN molhiv AUC 混比。

---

## 4. 证据（分层）

### 4.1 机制层 A — 强

- **C4 vs C8**（同 \(n,|E|\)）：严格 1-hop ~0.85；RW+共享 KSVD ~1.0。  
- 可视化：环上 B0 常为 P3，RW 可盖住 C4（`results/viz/`）。  
- 负对照：错误任务上 RW 可劣于 B0（distant-triangle）。

→ **更大感受野采样在「闭包 motif」上必要且有效。**

### 4.2 表征层 B — 中

- 共享字典必要（合成三角任务）。  
- 重建误差仅作 **同 patch 族** 诊断；**不可**用 B0 recon≈0 对 RW recon≈0.1 比表征优劣。  
- 多 walk：\(s_v=\mathrm{mean}_i x_i\) 已实现；过程指标（\|S\|、patch recon）可报。

### 4.3 下游层 C — 弱

协议：**Xu GIN 全局 epoch**（与论文 10-fold 选 epoch 习惯对齐）。

| MUTAG | Acc % |
|-------|-------|
| GIN only | 89.4 ± 5.8（贴文献 89.4） |
| 节点结构 B0/RW + 多种融合 | 最好约 **+1pt** 量级且不稳；硬拼 RW 可掉到 ~84 |

→ **当前结构通道在 MUTAG 上未证明稳定下游增益。**  
NCI1 / molhiv：**未完成**完整对照。

---

## 5. 结论（可对导师说）

1. **文献：** RW 有核/嵌入/采样/PE 四类成熟用法；本组定位为「诱导子图采样 + 可还原字典」，不是再做一个 node2vec 或 CIN。  
2. **机制：** 在 C4 探针上，RW（大于 1-hop）相对一阶 **有效**；一阶有理论上限。  
3. **表征：** 共享字典与节点级系数是正确方向；图级广播是错误融合。  
4. **下游：** MUTAG 上 **尚未**证明稳定涨点；不能声称分子任务已有效。  
5. **与作业匹配：** 完成「RW 能否/如何接入 KSVD」的调研与可行性；**未**完成「刷过 GSN/CIN」或方法定型。

---

## 6. 过程指标补全（录音 R1/R4/R5/R6 · 2026-07-24）

实现：`code/coverage_sample.py` · `code/run_coverage_rf.py` · 报告 `results/COVERAGE_RF_SUMMARY.md`。

| 项 | 结果摘要 |
|----|----------|
| **不硬删** | 仅边权 ×γ；`hard_delete=False` |
| **非全点种子** | degree_stratified / uncovered；预算 `max_walks` |
| **覆盖 vs 重复** | 同预算下 stratified+decay：walks=8 即 cover=1.0，repeat 低于 random/无 decay |
| **感受野曲线** | C4 类图：B0 **c4_hit=0**；RW L=4→0.33，L=6→0.89，L=8→0.99，L=12→1.0 |

→ 录音中「约束采样 + 感受野」从概念落实为 **可复现过程主表**（仍不宣称下游 SOTA）。

### 6.1 图级主线闭环（2026-07-25 补实）

默认算法 **CoverageRW-KSVD-Readout**（`notes/graph_level_algorithm.md`）：

| 层 | 结果 |
|----|------|
| 过程（C4 集 200 图） | coverage：walks≈5.7 且 cover=1.0；B0：walks=14 |
| 闭环 C4 分类 | **coverage 91.0%** vs B0 82.0% vs degree 82.0% |
| 对照三角任务 | B0=100%（1-hop 可见三角）；coverage 79% → **不是处处 RW 更优** |

实现：`code/graph_level.py` · `python -m code.run_graph_level_solid`。
## 7. 局限（主动写清）

- 文献覆盖/自回避专文仍可再补。  
- \(p,q,r,L\) 多为启发式，非 MUTAG 网格最优。  
- 结构向量仍是邻接 pad + 低维系数。  
- 下游 MUTAG 弱；无 OGB 主表。  
- 证据 **A+过程强、C 弱**。

---

## 8. 建议下一步（请导师拍板）
| 选项 | 内容 | 目的 |
|------|------|------|
| **T1 做实表征** | 过程指标面板（C4 覆盖、\|S\|、同族 recon）；小网格 \(r,L\) | 不依赖 Acc 的选模 |
| **T2 中等数据** | NCI1 同 Xu 协议一行 GIN±结构 | 补 C 层 |
| **T3 收口** | 本报告 + 开题式开放问题，暂停加模块 | 阶段交付 |
| **T4 远期** | molhiv + 原子融合 + 更强 head | 才碰 CIN 量级 |

**RW 可行性 / 录音达标 / CIN 路线**：见 [../notes/rw_feasibility_and_cin_path.md](../notes/rw_feasibility_and_cin_path.md)。

**推荐：** 先 **T3 交付本报告** 讨论；过程侧已补，若继续实验优先 **T2**，勿先 T4。

---

## 9. 附件索引

| 路径 | 内容 |
|------|------|
| `notes/rw_survey.md` | 文献 v0.2 |
| `notes/definition.md` | KSVD 定义 |
| `notes/rw_params.md` | \(p,q,r\) 与多 walk |
| `notes/shared_dict.md` | 共享字典决策史 |
| `notes/rw_feasibility_and_cin_path.md` | 录音达标 + CIN 路径 |
| `results/COVERAGE_RF_SUMMARY.md` | 覆盖驱动 + RF 曲线 |
| `results/C4_SUMMARY.md` | 机制主证据 |
| `results/FUSION_*.md` | 下游融合 |
| `results/viz/` | 采样可视化 |
| `docs/literature/deep/*` | 精读 |

---

*报告版本 0.2 · 含覆盖/RF 过程指标 · 下游 MUTAG 弱、无 OGB 主表*
