# DeepWalk（压缩精读）

> 概念：[concepts.md](concepts.md) · 组内：[rw_survey](../../../tracks/ksvd/notes/rw_survey.md)

| 项 | 内容 |
|----|------|
| 短名 | DeepWalk |
| 标题 | DeepWalk: Online Learning of Social Representations |
| 作者 / 出处 | Perozzi et al.；KDD 2014 |
| 链接 | [doi](https://doi.org/10.1145/2623330.2623732) |
| 层级 | C |
| 日期 | 2026-07-24 |

## 一句话

把图上的**均匀随机游走**当成「句子」，用 Skip-gram 学节点嵌入。

## 四问

### 1. 旧方法缺陷？

- 社交/信息网节点表示多靠手工特征或谱方法。  
- 谱方法贵、难扩展；难直接吃「序列式」分布假设。

### 2. 关键一步 / 假设与代价？

- 采样：从每个点出发多条固定长度均匀 RW。  
- 目标：Skip-gram 预测窗口内共现节点。  
- 假设：结构近邻在游走共现中可观察（分布假说搬到图）。  
- 代价：嵌入**不可还原为子图**；无 \(p,q\) 形态控制（后由 node2vec 补）。

### 3. 目的？

**C**：节点表示 → 多标签分类等；非图同构表达力主叙事。

### 4. 与主线 Kernel–GNN–GSN–KSVD？

| | 关系 |
|--|------|
| Kernel | 都用路径/游走统计；核是相似度，这里是**嵌入** |
| GNN | 更早、无监督邻域序列 |
| **KSVD 轨** | 只借「**多条 walk 采样**」习惯；**不要** Skip-gram 终点 |

## 数据流

```
G → 每点 r 条均匀 RW → 当句子 → Skip-gram → f(v)
```

组内截断在 walk / 诱导 \(G[S]\)，再进 KSVD。

## 边界

- 不证明图分类 SOTA；不控制 BFS/DFS 形态。  
- \(p=q=1\) 的 node2vec 近似 DeepWalk 采样。

## 口述 3 点

1. 游走 = 假句子。  
2. 多 walk 是标配，不是可选装饰。  
3. 我们要轨迹当**结构信号源**，不要词向量。
