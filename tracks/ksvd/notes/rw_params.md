# 子结构怎么选 · p/q 怎么定

## 节点级结构（拼 GIN 时）

对每个节点 \(v\)：

| 模式 | \(S_v\) 怎么来 | 之后 |
|------|----------------|------|
| **B0** | \(S=\{v\}\cup N(v)\)，可 cap 到 \(m\) | 诱导 \(G[S]\) → 邻接展平 → 共享 \(D\) 的 OMP 系数 \(s_v\) |
| **RW** | 从 \(v\) 出发 **r 条** node2vec 游走；每条 → \(S_i\) → \(y_i\) → \(x_i=\mathrm{OMP}(D,y_i)\)；\(s_v=\mathrm{pool}_i(x_i)\)（mean/max/mean_max） | 诱导 + 系数池化 |

**不是**「从图上抠预定义三角形列表」；是 **以节点为中心的采样邻域**。

### RW 超参（实现：`node_struct.NodeStructConfig` + `sample.node2vec_walk`）

| 符号 | 含义 | 本仓常用 |
|------|------|----------|
| \(L\) | `walk_length` 最多走几步 | 6 |
| \(m\) | `max_nodes` 节点集大小上限 | 8 |
| \(p\) | return：刚走来的边，回退权重 \(\propto 1/p\) | 见下 |
| \(q\) | in-out：远离上一跳的点权重 \(\propto 1/q\) | 见下 |
| no_backtrack | 尽量不立刻走回上一节点 | True |

node2vec 直觉：

- **\(p\) 大**：少回头  
- **\(q>1\)**：更贴局部（偏 BFS）  
- **\(q<1\)**：更往外扩（偏 DFS）  
- **\(p=q=1\)**：近似均匀二阶游走（DeepWalk 味）

### p,q 是网格搜出来的吗？

**没有在 MUTAG 上为融合做系统网格。** 来源是：

| 设定 | 用于 |
|------|------|
| \(p=0.5,q=2\) | 偏局部；C4/节点 GIN 实验里用过（希望围着环转） |
| \(p=q=1\) | 默认/对照 |
| 多种 \((p,q,L,m)\) | **C4 合成探针**里扫过采样 Acc，不是 GIN 超参搜索 |

融合脚本 CLI：`--p --q --walk_length --max_nodes`。

---

## 图级 KSVD（早期 smoke）

可对全图采多条 RW / 每节点 B0，堆成 \(Y\) 再学字典——与「每节点一条 patch → \(s_v\)」是两条线。

## 覆盖驱动采样（录音 R4/R5/R6）

实现：`code/coverage_sample.py`

| 项 | 做法 |
|----|------|
| 种子 | `degree_stratified` / `uncovered` / `random`；**非必须全点** |
| 软惩罚 | 轨迹边权 × `edge_decay`（默认 0.7） |
| 硬删 | **禁止**（`hard_delete=False`） |
| 早停 | `traj_edge_cover ≥ cover_target` 或用尽 `max_walks` |
| 指标 | `traj_edge_cover`, `traj_edge_repeat`, `mean_|S|`, 以及 C4 的 `c4_hit` RF 曲线 |

```bash
python -m code.run_coverage_rf
```
