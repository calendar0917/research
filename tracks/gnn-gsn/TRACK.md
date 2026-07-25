# Track: Kernel–GNN–GSN 图分类（归档）

| 项 | 内容 |
|----|------|
| slug | `gnn-gsn` |
| 状态 | **archived** |
| 创建 | 2026-07-23（迁入工作台；实现仍在旧仓） |
| 主线位置 | Kernel → GNN → GSN |

## 问题（一句话）

结构编码谱系上，用**统一图分类协议**做 Kernel / GIN / GSN 的最小实证，并分清 paper 自检 vs strict 主横比。

## 范围

- **已做**：概念谱系；GSN / WL 精读；双协议；本仓 GIN/WL + 官方 GSN；home-GSN 消融；导师可读表  
- **不做（本轨）**：全库 Errica 嵌套、扩 COLLAB 等 social 全量、简化 GSN 进主结论  
- **实现位置**：完整代码与 JSON **不搬迁**，见旧仓（下表）

## 协议 / 评估

| protocol_id | 角色 |
|-------------|------|
| `strict-v0` | **主表 B** 横比 |
| `paper-optimistic` / 官方 paper | **表 A** 自检，不与 strict 混比 |
| `official-strict` | 官方仓严格校准 |

细则与决策记录：旧仓 `paper/docs/experiments/index.md`、`paper/experiments/results/SCHEMA.md`。

## 进度

| 阶段 | 状态 |
|------|------|
| 概念与谱系 | 完成 → 本仓 `docs/literature/deep/concepts.md` |
| GSN / WL 精读 | 完成 → deep/ 压缩 + 旧仓全文 |
| 双协议 + 复现 | 完成 → 旧仓 experiments |
| 导师交付 | 可对接 → 旧仓 xlsx / mentor_brief |
| 本工作台 | **归档**；新工作去 ksvd 等轨 |

## 入口（只读指针 → 旧仓 `../../paper`）

| 旧路径 | 内容 |
|--------|------|
| `paper/docs/research_guide.md` | 该阶段完整手册与数字速查 |
| `paper/docs/experiments/` | 协议说明、xlsx、csv |
| `paper/experiments/` | `common.py` / `run_*.py` / `results/` / `official_gsn/` |
| `paper/experiments/results/REGISTRY.md` | 结果登记 |

本轨 `code/`、`results/` **故意留空脚手架**：避免双份真相；查数去旧仓。

## 结论要点（不写论文精度表）

- 结构方法要分 **目的维**（表达力 A vs 下游 B），协议不混比。  
- GSN 主结果认 **官方实现**；简化 GSN 仅附录/消融。  
- NCI1 上 GIN 差距更像 **超参/协议** 而非「实现完全错了」一类假说（详见旧 guide）。  
- 与 KSVD：都显式结构；GSN 手选计数，KSVD 字典学习（见 concepts）。

## 禁止

- 在本轨或旧仓继续扩张为「新专题主战场」  
- 把 paper 轨数字与 strict 主表横比当排名  
- 把简化 GSN 写进主结论替换官方

## 下一步

无（archived）。新课题：`../../scripts/init_track.sh ksvd "…"`
