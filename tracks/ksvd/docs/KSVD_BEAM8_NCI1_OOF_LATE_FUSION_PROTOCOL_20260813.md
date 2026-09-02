# Beam8/NCI1 OOF late-fusion protocol

> 日期：2026-08-13  
> 状态：compact follow-up 结果可见后注册；多模态分类接入诊断。

## 1. 问题

Beam8/s8/o2 的关系绑定可检测，但单独结构分支低于完整 node-feature+graph-statistics
基线。本轮不再增加手工关系特征，而采用低容量 prediction-level late fusion，检验
Beam8 chain 是否提供主属性分支之外的互补分类信息。

## 2. 严格 OOF 结构

每个固定 outer fold 内：

1. 主分支：`FEATURE_STATS` linear head；
2. 结构分支：`INIT_BAG / INIT_COMPACT_TRUE / INIT_COMPACT_SHUFFLED` linear head；
3. 在 outer-train 内做固定 3-fold inner OOF，得到两个无泄漏 decision logits；
4. 只在两个 OOF logits 上训练 `LogisticRegression(C=1)` meta head；
5. 两个 base heads 在完整 outer-train 重拟合，产生 outer-test logits，交给冻结 meta head。

不用 outer-test 选择权重，不训练 MLP、gate、attention 或 patch GNN。选择 INIT 是因为
compact follow-up 中 FINAL 在 3/3 folds 低于 INIT；同时报告 FINAL_TRUE late fusion 作为
KSVD update 对照，不据此调字典。

## 3. 必报

- `FEATURE_STATS`；
- `INIT_BAG`、`INIT_TRUE`；
- `FUSION_INIT_BAG`、`FUSION_INIT_TRUE`、`FUSION_INIT_SHUFFLED`；
- `FUSION_FINAL_TRUE`。

## 4. 冻结 gate

- fusion increment：`FUSION_INIT_TRUE - FEATURE_STATS >= 1 point`，且至少 2/3 folds 为正；
- relation-specific fusion：`FUSION_INIT_TRUE - FUSION_INIT_SHUFFLED >= 0.5 point`，
  且至少 2/3 folds 为正；
- chain-specific fusion：`FUSION_INIT_TRUE - FUSION_INIT_BAG >= 0.5 point`，
  且至少 2/3 folds 为正；
- KSVD update：`FUSION_FINAL_TRUE - FUSION_INIT_TRUE >= 1 point`，且至少 2/3 folds 为正。

前三项通过才支持 Beam8 chain 是有效的多视图补充分支；仅 fusion increment 通过而关系
对照失败，只能支持一般结构 late fusion。全部失败则 NCI1 上停止扩大分类融合模型。

