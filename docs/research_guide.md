# 工作手册（全局）

> 规矩 + 当前进度。单篇公式细节放轨内 notes / 文献深读。  
> 旧仓：`../paper`（只读，勿在那边开新专题）。

---

## 0. 文档地图

| 路径 | 放什么 | 不要放 |
|------|--------|--------|
| **本文件** | 怎么做、全局进度、active 轨 | 实验数字明细 |
| `tracks/*/TRACK.md` | 该课题问题/状态/禁止项 | 第二份全局计划 |
| `docs/literature/deep/` | 概念谱系 + 精读（压缩版） | 堆论文 Acc |
| `docs/luyin/` | 录音原文（只存档） | 平行「解读全集」 |
| `docs/deliverables/` | 对外材料（导师页） | 按日堆 mentor_brief |
| `tracks/*/results/` | 该轨 JSON / registry | 写进精读笔记 |
| `../paper` | GNN 阶段完整代码与结果 | 继续扩张新线 |

**一事一处。** 进度：全局 §2 + 轨内 TRACK；概念：`deep/concepts.md`。

---

## 1. 主线（跨轨）

结构如何被编码：

**Graph Kernel**（预定义结构特征 + 核）→ **GNN/GIN**（消息传递、≤1-WL）→ **GSN**（显式子结构计数 + MPNN）→ **KSVD**（字典学习出的结构原子）

各轨只做主线的一段；对照表见 [concepts.md](literature/deep/concepts.md)。

---

## 2. 进度（改这一节）

| Track | 状态 | 要点 |
|-------|------|------|
| gnn-gsn | **archived** | 双协议图分类 + 官方 GSN；[TRACK](../tracks/gnn-gsn/TRACK.md)；实现 `../paper` |
| ksvd | **active** | 定义 + [RW 调研 v0.1](../tracks/ksvd/notes/rw_survey.md)；[TRACK](../tracks/ksvd/TRACK.md) |
| hod-gnn-replication | **active** | HOD-GNN 基线复现审计；[TRACK](../tracks/hod-gnn-replication/TRACK.md)；独立官方环境 |

### 下一步

1. ksvd：图级取样已做实 `results/GRAPH_LEVEL_SOLID_SUMMARY.md`（C4 闭环 coverage>B0）
2. ksvd：进入 `luyin16` 结构独立性、泄漏和长距离诊断阶段
3. hod-gnn-replication：按 `TRACK.md` 维护 paper-seeds 与 official-seeds 两套协议
4. 需要历史 GNN 数字 → `../paper` xlsx/registry（不拷进本仓主表）
5. 录音 → `docs/luyin/`（当前 KSVD 阶段主来源：`luyin16`；历史机制：`luyin10`、`luyin3`/`luyin4`）


---

## 3. 怎么做研究

### 3.1 分层

```
问题意识 → 谱系动机 → 少而深精读 → 最小实验 → 缺口
```

### 3.2 每篇四问

1. 旧方法缺陷？  
2. 关键一步 / 假设与代价？  
3. 目的？（A 表达力 / B 下游分类 / C 其它，**分表**）  
4. 与主线（Kernel–GNN–GSN–KSVD）近远？

### 3.3 AI

| 可以 | 不行 |
|------|------|
| 翻译、检索、代码、润色、事后质检 | 代写四问终版、代判目的、只读 AI 综述称已读 |

### 3.4 口诀

```
不跳层；四问；宁少深读；AI 不代懂；
主线过滤；结果只进轨内 results；不跨轨混协议。
```

---

## 4. Track 约定

| 字段 | 含义 |
|------|------|
| `active` | 正在做 |
| `paused` | 暂停 |
| `archived` | 收工，只查不扩主结论 |
| `pending` | 未 init 或仅占位 |

新建：

```bash
./scripts/init_track.sh <slug> "<一句话标题>"
```

---

## 5. 新会话启动

1. 读本文件 §2  
2. 打开当前 active 轨的 `TRACK.md`  
3. 实验只进该轨 `code/` + `results/`  
4. 需要旧 GNN 细节 → `../paper/docs/research_guide.md`（只读）
