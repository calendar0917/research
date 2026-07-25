# 图级默认算法：CoverageRW-KSVD-Readout

> 对齐 luyin10：**一系列子图代表图 → KSVD → readout → 图分类**。  
> 与节点级 `node_struct`（每点 \(s_v\)）分离。

## 输入输出

- 输入：无向图 \(G=(V,E)\)
- 输出：图向量 \(s_G\in\mathbb{R}^d\)；训练时还有共享字典 \(D\)

## 算法步骤

1. **种子**（非全点）：`degree_stratified`，预算 `max_walks=12`  
2. **游走**：node2vec，`p=0.5`, `q=2.0`, `L=8`，不立刻回头  
3. **软降权**：轨迹边 `w ← w × 0.7`；**永不删边**  
4. **Patch**：访问节点去重 cap 到 `m=8` → **诱导子图** \(G[S]\)  
5. **早停**：轨迹边覆盖率 ≥ `0.95` 或用尽预算  
6. **学字典**（仅 train 图）：所有 patch 向量堆成 \(Y\)，KSVD 得 \(D\)（atoms=12, T=3）  
7. **编码一图**：同样采样得 \(Y_g\)，\(x_j=\mathrm{OMP}(D,y_j)\)，  
   \(s_G = \mathrm{readout}(X)\) 拼 energy/usage（录音「能量」）  
   - 池化可选：`mean` / `max` / **`attn`（MIL attention）** — `ksvd.pool_X`  
8. **分类**：\(s_G\) → 标准化 + LogisticRegression  

## 过程指标（图级）

| 指标 | 含义 |
|------|------|
| traj_edge_cover | 轨迹边覆盖率 |
| traj_edge_repeat | 多余重复质量 |
| n_walks | 实际 walk 数 |
| mean \|S\| | patch 规模 |

## 成功标准（本轨）

| 层 | 标准 |
|----|------|
| 过程 | 同 cover 下 walks 少于全点 B0；repeat 不爆炸 |
| 闭环 | C4 任务上 coverage 模式 Acc **明显高于** B0 与随机 |

## 明确不做

- 节点分类、GIN 融合（另模块）  
- 用低 repeat 约束节点级 \(s_v\)（会抽空节点邻域）  

## 复现

```bash
python -m code.run_graph_level_solid
```
