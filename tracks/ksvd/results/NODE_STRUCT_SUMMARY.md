# Node-level structure + GIN（paper-optimistic）

## 融合方式（本次正确版）

```
train fold 上所有节点的 patch → 学共享 D
对每个节点 v：
  S_v = 1-hop(v) 或 从 v 出发的 RW
  y_v = vec(诱导 G[S_v])
  x_v^struct = |OMP(D, y_v)|   # 维数 = n_atoms = 12
  x_v' = [x_v^attr ; x_v^struct]
→ GIN(x')
```

| 对照 | 做法 | 节点结构维 |
|------|------|------------|
| gin_only | 仅属性 | 0 → in_dim≈7 |
| **gin_node_B0** | 每节点 1-hop 系数 | **12** → in_dim≈19 |
| **gin_node_RW** | 每节点 RW patch 系数 | **12** → in_dim≈19 |
| gin_graph_bcast | 图级 108 维广播（旧） | 108 复制 → in_dim≈115 |

协议：`paper-optimistic-gin-nodestruct-v0`（10-fold，epoch@max held-out Acc）。  
代码：`code/node_struct.py` · `code/run_node_struct_gin.py`

---

## MUTAG 结果

| variant | Acc % |
|---------|-------|
| **gin_only** | **94.6 ± 2.5** |
| **gin_node_B0** | **93.6 ± 4.1** |
| **gin_node_RW** | **93.1 ± 4.2** |
| gin_graph_bcast（旧） | **83.5 ± 6.1** |

---

## 解读

1. **节点级 ≫ 图级广播**：93.6% vs 83.5%（约 +10pt）→ 你的理论质疑被实验支持。  
2. **相对 gin_only**：节点级约 −1pt，统计上接近、**未证明增益**，但**不再灾难性降分**。  
3. B0 ≈ RW：MUTAG 小分子上 1-hop 系数已够；RW 优势仍主要在 C4 类合成探针。  
4. 维数合理：12 维系数 vs 以前 108 维图摘要硬塞。

---

## 结论

- **融合设计**：应使用 **每节点系数**，不要图级广播。  
- **当前节点级拼接**：合理基线；在 MUTAG 乐观 GIN 上 **持平略低**，未超过纯 GIN。  
- 若要增益：残差 `h←h+W s_v`、门控、或结构进边特征（更接近 GSN），而非继续加 raw concat 维数。
