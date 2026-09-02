# Attributed Beam8 frozen residual repeated-SHUFFLED binding audit

> 日期：2026-08-14  
> 状态：multi-split binding gate 失败后冻结；repeated-shuffle 结果不可见。

## 1. 研究问题

Multi-split 验证中，TRUE residual 稳定超过 GINE 与 BAG，但 TRUE−SHUFFLED 只有
`+0.29pt`，且 split seed 2 为负。原实验每个 fold 只使用一个确定性 SHUFFLED realization，
因此本轮回答：

> binding gate 失败是否只是单次置换方差，还是正确 patch→node 对应确实不能稳定超过
> 同图、同 row-multiset 的错误对应？

## 2. 冻结设计

- 数据与 outer splits：TU Mutagenicity，split seeds 0/1/2，各 3 folds；
- model seed 固定为 0；
- 使用现有 multi-split 结果中冻结的 base epochs 与 TRUE test scores；
- 每个 fold 重训同一 inner/full GINE，但不重新选择 base epoch；
- Beam8 INIT dictionary、normalization、orbit-safe incidence 与原实验完全相同；
- 每个 fold 运行 8 个 SHUFFLED realizations，repeat 0 使用原始 shuffle seed；
- 每个 realization 独立走 inner residual epoch selection 与 full outer-train retraining；
- GINE states/logits 预计算后冻结，所有 shuffle heads 使用相同 cache、初始化、loader seed、
  容量和优化器。

SHUFFLED 仍只在同图、同 automorphism-orbit size 的 canonical orbit groups 间置换，保持 row
multiset，不改变任何 graph-level Beam8 content。

## 3. Validity check

对每个 fold，repeat 0 的 GINE 与 SHUFFLED test balanced accuracy 必须与原始冻结 JSON 一致；
任一差值超过 `1e-12`，审计立即失败，不解释后续 repeats。

## 4. Diagnostic gate

共 9 folds × 8 shuffles = 72 个 TRUE−SHUFFLED comparisons。只有全部满足才认为 exact binding
获得 repeated-shuffle 支持：

1. 72 个 comparisons 的 mean `≥+0.5pt`；
2. TRUE 胜率 `≥60%`，即至少 44/72；
3. 至少 6/9 fold-level mean deltas 为正；
4. 至少 2/3 split-level mean deltas 为正；
5. 最差 split-level mean 不低于 `−0.5pt`。

这是观察到单-shuffle 失败后的机制诊断，不升级为独立 confirmatory evidence。通过后仍需新的
未见 split 确认；失败则认为当前分类收益来自 graph-specific Beam8 content 与 node
heterogeneity，而非精确 patch→atom binding。
