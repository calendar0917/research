# luyin14：TU 中等规模真实数据外部验证协议

> 日期：2026-08-13  
> 状态：结果前冻结  
> 数据：Mutagenicity / NCI1；不构造数据集。

## 1. 目的

MUTAG/PTC_MR 上 edge-aware joint dictionary 在 split seed 0 出现机制信号，但没有跨三个 split
稳定。MolHIV 已参与过多轮方法选择，不再作为新的独立确认数据。本轮选择两个约 4k graphs
的 TU 数据集，在不增加模型容量的情况下做固定协议迁移：

- **Mutagenicity**：4,337 graphs，14 类 node labels、3 类 edge labels；检验完整 edge-aware
  structure–attribute shared code；
- **NCI1**：4,110 graphs，37 类 node labels、无 edge labels；检验 shared code 是否依赖
  typed edge，作为 no-edge matched control。

数据规模、字段和类别比例可以在实验前审计；不得根据下游结果改变表示或超参数。

## 2. 固定表示

共同设置：

- 每节点 rooted `v∪N(v)` patch，最多 8 nodes；
- Mutagenicity/NCI1 最大度数均为 4，没有节点触发邻居截断；非连通图因此不需要全图
  GLOBAL-WL tie-breaker，直接做每个 rooted ego 的 canonicalization；
- 中心节点属性 + 一跳邻居属性均值作为 attribute view；
- structure/attribute blocks 分别 outer-train center + RMS scale；
- shared-code K-SVD：`K=24, T=3, updates=5`，最多 3,000 outer-train node patches；
- code 通过 zero-init bounded FiLM 注入 3-layer hidden64 GNN；
- dropout 0.5、Adam lr 0.01、最多 100 epochs、patience 25。

数据集差异：

- Mutagenicity：28 upper-triangle slots × 3 edge types，基线为读取相同 edge labels 的 GINE；
- NCI1：28D binary adjacency，基线为 GIN；不得为 NCI1 人工补 edge types。

## 3. Controls

每个数据集固定比较：

- `BASE_ONLY`：GIN 或 GINE；
- `FINAL_TRUE`：FINAL joint sparse code；
- `FINAL_SHUFFLED`：每图内打乱 attribute context 后重新拟合和编码；
- `INIT_TRUE`：相同 initialization、0 次 K-SVD updates。

所有 dictionary/scaler/token normalization 只使用 outer-train graphs。SHUFFLED 保持每图内
structure 和 attribute 的边际 multiset，只破坏节点级共同出现关系。

## 4. Stage A 与停止规则

- split seed 0，3 stratified folds，model seed 0；
- outer-train 内固定 80/20 stratified validation 选 epoch；
- 完整 outer-train 重训固定 selected epoch，test fold 只评一次；
- PyTorch 单线程 deterministic algorithms；
- 不扫描 K/T、patch size、block weight、FiLM strength、hidden 或 checkpoint。

每个数据集独立通过以下四项才扩展 split seeds 1/2：

1. `FINAL_TRUE - BASE_ONLY >= +0.01`，至少 2/3 folds 正；
2. `FINAL_TRUE - FINAL_SHUFFLED >= +0.005`，至少 2/3 folds 正；
3. `FINAL_TRUE - INIT_TRUE > 0`，至少 2/3 folds 正；
4. FINAL reconstruction 低于 INIT。

判读：

- Mutagenicity 通过：支持 edge-aware joint code 值得多 split 验证；
- NCI1 通过：说明 shared-code 机制不完全依赖 edge labels；
- 仅 Mutagenicity 通过：贡献可能依赖 typed chemical bonds；
- 两者都失败：停止在 TUD 上增加 cross-attention/融合容量，KSVD 保留为 compressor/control。
