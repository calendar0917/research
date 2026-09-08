# Track: Kernel–GNN–GSN 图分类（归档 · 旧仓已遗失）

| 项 | 内容 |
|----|------|
| slug | `gnn-gsn` |
| 状态 | **archived**（2026-09 核验：旧仓已不可访问，见下） |
| 创建 | 2026-07-23（迁入工作台；实现原在旧仓，旧仓现已遗失） |
| 主线位置 | Kernel → GNN → GSN |

## 问题（一句话）

结构编码谱系上，用**统一图分类协议**做 Kernel / GIN / GSN 的最小实证，并分清 paper 自检 vs strict 主横比。

## 旧仓遗失声明（重要）

本轨原指向旧仓 `../../paper`（独立 GitHub 私有仓 `calendar0917/paper`）。2026-09 核验：

- GitHub 上该仓不可访问；本地 `~/code/paper` 不存在；本工作台 git 历史（自迁移起）**从未包含** `paper/experiments/`、`common.py`、`official_gsn/`、`results/REGISTRY.md` 中任何文件
- 原实现（`common.py` / `run_*.py` / `official_gsn/` / `results/`）与历史数字**随旧仓遗失，无法恢复**
- 已修复本工作台全部指向旧仓的悬空指针（见本轨与 `docs/` 各处；`verify_expressivity.py` 已改为自包含）

## 职责移交：权威实现 → `gsn-replication` 轨

| 旧仓内容（已遗失） | 现在 | 位置 |
|---|---|---|
| `official_gsn/`（官方 GSN 封装/适配） | 官方仓 vendor + runner | `tracks/gsn-replication/code/vendor/` + `run_official.py` |
| `common.py`（统一协议框架） | 各轨自实现（不再有共享 common） | `run_strict.py`（严格协议）/ `run_wl_subtree_kernel.py`（wl 轨） |
| `run_*.py`（实验入口） | 配置驱动入口 | `run_strict.py --config …` / `run_official.py --config …` |
| `results/`（历史数字 + REGISTRY） | **历史数字不保留**；新结果 | `tracks/gsn-replication/results/`（git-ignored；协议 `gsn-strict-social-v1`） |
| `docs/experiments/`（双协议细则） | 协议文档 | `tracks/gsn-replication/{docs,notes}/README.md`、`docs/SERVER_RUNBOOK.md` |

## 协议 / 评估（历史设计 → 现行）

| protocol_id | 角色 | 现状 |
|---|---|---|
| `strict-v0`（旧） | 主表 B 横比 | 历史设计；细则随旧仓遗失 |
| `paper-optimistic`（旧） | 表 A 自检 | 历史设计 |
| `official-strict`（旧） | 官方仓严格校准 | → 演进为 `gsn-strict-social-v1`（gsn-replication 轨，可用） |
| `gsn-strict-social-v1` | **现行严格协议**（10 seeds × 10×10 CV，val 选 epoch） | `tracks/gsn-replication` |

## 进度

| 阶段 | 状态 |
|------|------|
| 概念与谱系 | 完成 → 本仓 `docs/literature/deep/concepts.md` |
| GSN / WL 精读 | 完成 → deep/ 压缩版（完整旧笔记随旧仓遗失） |
| 双协议 + 复现 | 旧实现遗失；**重做** → gsn-replication 轨（进行中） |
| 导师交付 | 历史表遗失；如有需要基于新轨重建 |

## 结论要点（仍有效，已由 gsn-replication 继承）

- 结构方法要分 **目的维**（表达力 A vs 下游 B），协议不混比。
- GSN 主结果认 **官方实现**；简化 GSN 仅附录/消融。
- 与 KSVD：都显式结构；GSN 手选计数，KSVD 字典学习（见 concepts）。

## 禁止

- 在本轨继续扩张为「新专题主战场」
- 把本轨任何数字与现行轨横比（历史数字不存在；新旧协议也不同）

## 下一步

无（archived）。GSN 现行工作 → `tracks/gsn-replication`。
