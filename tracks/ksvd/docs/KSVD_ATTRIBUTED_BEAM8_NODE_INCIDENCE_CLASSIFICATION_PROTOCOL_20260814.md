# Attributed Beam8 node-incidence classification protocol

> 日期：2026-08-14  
> 状态：node-incidence feasibility 与 orbit-safe equivariance gate 通过后冻结，分类结果不可见。

## 1. 核心假设

图级 pooling 在结构—属性融合前丢失 node–patch correspondence。将 Beam8 patch sparse code
经 node–patch incidence 回写到原图节点，再由 edge-aware GINE 使用，可能把“可检测的 patch
relation”转化为分类增量。

## 2. 数据与表示

- 数据：完整 TU Mutagenicity；split seed 0，3-fold stratified CV；
- Beam8：合法 attributed `s8/o2/Beam8/R1/m1.5 BASE`，不加入 typed anchor；
- dictionary：outer-train patches，`K24/T3/5 updates/3000 patches`；
- node token：patch RAW、INIT 或 FINAL code 拼接 patch atom histogram，经 incidence mean/max
  回写；另含 slot distribution、center rate、patch position、patch relation degree、count/coverage；
- 所有 node incidence 先在 attributed automorphism orbit 内平均，再使用 outer-train node-only
  normalization；
- backbone：3-layer GINE，hidden64，读取原始 node/edge attributes；node incidence 通过每层
  zero-init bounded FiLM 注入。初始模型严格等于 GINE_ONLY。

## 3. 必报变体

- `GINE_ONLY`；
- `RAW_FULL_TRUE`；
- `INIT_FULL_TRUE`；
- `FINAL_FULL_TRUE`；
- `FINAL_FULL_SHUFFLED`：在同图、同 orbit-size 的 attributed canonical orbits 之间置换
  node incidence，保留 node-feature row multiset与容量；
- `FINAL_BAG_BROADCAST`：将同图 FINAL incidence mean 广播到全部节点；
- `FINAL_NO_RELATION`：保留 localized code incidence 与 count/coverage，slot/center/position/
  patch-relation channels 置零。

## 4. Strict checkpoint

- outer-train 内固定 80/20 stratified inner validation；
- 最多 80 epochs、patience 20，只按 inner-valid balanced accuracy 选 epoch；
- 重置模型、optimizer、DataLoader RNG，在完整 outer-train 训练 selected epoch；
- outer-test 只评估一次；所有 variant 同 fold 使用同一 model/data seeds；
- 主指标 balanced accuracy。

## 5. 冻结 gate

FINAL 要转化为可归因分类收益，必须同时满足：

1. `FINAL_FULL_TRUE − GINE_ONLY ≥ +1pt`，至少 2/3 folds 正；
2. `FINAL_FULL_TRUE − FINAL_FULL_SHUFFLED ≥ +0.5pt`，至少 2/3 folds 正；
3. `FINAL_FULL_TRUE − FINAL_BAG_BROADCAST ≥ +0.5pt`，至少 2/3 folds 正；
4. `FINAL_FULL_TRUE − FINAL_NO_RELATION ≥ +0.5pt`，至少 2/3 folds 正；
5. `FINAL_FULL_TRUE − INIT_FULL_TRUE > 0`，至少 2/3 folds 正。

第 1 项说明分类增量；第 2/3 项说明 node binding/localization；第 4 项说明 Beam8
slot/chain/position metadata；第 5 项才支持 ordinary KSVD updates。若只通过部分 gate，按对应
机制保留结论，不得合并声称“Beam8+KSVD 分类成功”。
