# Follow-up experiments

## 1. Triangle vs long-cycle (controlled)

Meta: `{'task': 'triangle_vs_longcycle', 'n_nodes': 16, 'edges_per_graph': 16, 'n_per_class': 100, 'class0_triangle_rate': 1.0}`

| method | Acc |
|--------|-----|
| degree_stats | 0.575 ± 0.047 |
| B0 | 0.585 ± 0.064 |
| B1 | 0.560 ± 0.087 |
| M0 | 0.600 ± 0.059 |
| M0_local | 0.600 ± 0.059 |

## 2. MUTAG fusion

Meta: `{'name': 'MUTAG', 'n_graphs': 188, 'n_classes': 2, 'mean_n': 17.930851063829788, 'mean_e': 19.79255319148936, 'n_with_node_features': 188, 'node_feat_dim': 7}`

### Baselines

| method | Acc |
|--------|-----|
| node_attr_only | 0.857 ± 0.083 |
| degree_stats | 0.878 ± 0.058 |
| attr+degree | 0.878 ± 0.047 |

### With KSVD structure

| method | struct | +attr | +attr+deg |
|--------|--------|-------|-----------|
| B0 | 0.665 | 0.799 | 0.846 |
| B1 | 0.660 | 0.818 | 0.825 |
| M0 | 0.649 | 0.787 | 0.829 |

## 解读（短）

### 合成（三角形 vs 长环）

- 同 \(n\)、同 \(|E|\) 后，**度特征应不再满分**（见 json 里 `degree_stats`）。
- KSVD 各采样 Acc **~0.56–0.60**，略高于随机 0.5，**未形成可靠结构判别**。
- 可能原因：诱导邻接+pad 排序破坏了局部三角信号；读出过粗；每图独立字典，系数不可比。

### MUTAG 融合

| 设定 | Acc（约） |
|------|-----------|
| 节点属性 mean/max/sum | **0.86** |
| 度统计 | **0.88** |
| 结构-only KSVD | ~0.65–0.67 |
| 结构+属性 | ~0.79–0.82（**低于** 属性 alone） |
| 结构+属性+度 | ~0.83–0.85（仍 **≤** 度/属性基线） |

→ 当前 KSVD 向量更像 **噪声通道**：拼上去拖累线性分类器，而不是互补信息。

### 对 molhiv 的含义

仅 LR + 现读出，上 molhiv **大概率不理想**（见对话分析）；要先修「结构向量是否携带可分信息」，或上更强 head / 端到端。

