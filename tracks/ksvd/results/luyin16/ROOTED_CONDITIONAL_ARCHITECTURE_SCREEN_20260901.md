# MolHIV：可学习结构–属性融合架构快筛（2026-09-01）

## 目的

这轮不是继续增加统计量或调 XGBoost，而是检验一个完整的可学习 patch 模型：

```text
完整 radius-2 rooted patch
  ├─ topology-only GIN（只看邻接、root、shell）→ structure token
  └─ atom/bond attribute set encoder（不看邻接）
          ↓
  structure-query set cross-attention
          ↓
  gated-attention MIL graph readout
```

所有节点都是中心，patch 是完整 induced ego；原子属性使用 40D strict semantics
（去除 degree/ring），键属性使用 13D compact semantics。模型没有读 node ID，
没有 K-SVD，也没有 cycle 手工特征。

## 协议

- 只使用 official-train 内部三个 Bemis–Murcko scaffold folds；每折固定抽取
  1200 train / 600 held-out graph。
- hidden=48、2 层 topology GIN、15 epoch、batch=24、seed=0；没有用 outer
  held-out 选择 epoch 或超参数。
- 对照：structure-only、attribute-only、simple concat、cross-attention。
- `random` null 在同一图内任意交换 attribute patch；`size_matched` null 只在
  节点数和边数相同的 patch 之间交换，因此保留了 patch 尺寸配对。
- 重标号审计通过；official-valid/test 未编码、未评估。

原始输出：
[`rooted_cross_attention_patch_network/summary.json`](rooted_cross_attention_patch_network/summary.json)
和
[`rooted_cross_attention_patch_network/summary.md`](rooted_cross_attention_patch_network/summary.md)。

## 结果

| view | fold 0 | fold 1 | fold 2 | mean |
|---|---:|---:|---:|---:|
| structure | 0.6130 | 0.5800 | 0.6584 | 0.6171 |
| attribute | 0.7068 | 0.6750 | 0.6232 | **0.6683** |
| concat | 0.6165 | 0.6082 | 0.6743 | 0.6330 |
| cross-attention | 0.6156 | 0.6229 | 0.6504 | 0.6297 |
| cross-attention + random shuffle | 0.5000 | 0.6274 | 0.6290 | 0.5855 |
| cross-attention + size-matched shuffle | 0.5592 | 0.6625 | 0.6549 | 0.6255 |

关键差值：

- cross-attention − concat：`−0.00335`，只赢 `1/3` fold；
- cross-attention − size-matched shuffle：`+0.00414`，只赢 `1/3` fold；
- cross-attention − random shuffle：`+0.04421`，但这个差值不能作为细粒度机制证据，
  因为 random shuffle 同时破坏了 patch 尺寸/边数的配对；
- attribute − structure：`+0.05122`，属性边际仍是主要可学习信号。

此前的 pooled MoE 和 FiLM 版本也未通过：

- pooled MoE：mean `0.5534`，低于 concat `0.6330`；
- FiLM：mean `0.5837`，低于 concat `0.6330`。

## 判定

这次探索给出的是一个清晰的负结果：

1. **没有证据表明“同一 patch 的细粒度结构–属性对应”在当前 MolHIV scaffold
   迁移上带来稳定标签增量。** random null 的 gap 主要由尺寸配对混淆解释；在
   size-matched null 下几乎消失。
2. **WL/统计量不是唯一瓶颈的说法得到支持，但可学习 cross-attention 也没有超过
   属性边际。** 结构分支单独较弱，简单拼接还会干扰属性分支。
3. 这不是进入 Optuna、增加 expert 数或上 official test 的理由。继续在这个
   radius-2 patch 内堆融合层，预期收益很低。

## 下一步建议

应停止“局部 patch set 的细粒度融合”作为主路线。若仍要做一次架构实验，应该
改变研究对象而不是再改融合算子：使用标准 edge-aware GNN 在**整分子图**上同时
建模节点/键属性，再把局部 patch token 作为可选 residual context；先与 clean
GINE 基线在同一 train-only scaffold protocol 比较。只有该全图模型明显优于
当前 clean 统计基线，才值得讨论局部 token 的增量；否则保留导师统计路线但把
0.8022 视为未通过不变性审计的历史上限，不再用它指导 K-SVD/维度搜索。

