# Attributed Beam8 frozen-GINE residual calibration protocol

> 日期：2026-08-14  
> 状态：localized INIT multi-model-seed 失败后冻结，residual 结果不可见。

## 1. 研究问题

joint FiLM training 会改变 GINE optimization trajectory，localized INIT 的 seed0 正增益在
model seeds 1/2 上反转。本轮冻结已经训练好的 GINE，只训练一个小型 Beam8 residual head，
回答：

> 条件于固定 GINE node states/logits，Beam8 localized INIT 是否仍包含独立分类残差信息？

## 2. 数据与结构特征

- 数据：TU Mutagenicity，split seed 0，3 outer folds，model seed 0；
- GINE：3 layers、hidden64、edge attributes、layer-wise sum prediction；
- Beam8：attributed s8/o2 BASE；outer-train deterministic maximin INIT dictionary，K24/T3；
- node feature：INIT code + patch atom histogram 的 orbit-safe incidence mean/max；不使用
  slot/position/relation metadata，不运行普通 KSVD updates。

## 3. Frozen residual head

冻结 GINE 后读取最终节点状态 `h_v` 与结构特征 `s_v`：

`u_v = tanh(W_h h_v) ⊙ tanh(W_s s_v)`，rank=16

对 `u_v` 做 graph sum pooling，再经零初始化线性层得到 residual logits：

`logits = stopgrad(GINE_logits) + W_o sum_v u_v`

只训练 `W_h/W_s/W_o`，GINE 参数与 BatchNorm 状态全部冻结。

## 4. 变体

- `GINE_FROZEN`：零 residual；
- `TRUE_RESIDUAL`：正确 orbit-safe localized incidence；
- `SHUFFLED_RESIDUAL`：同图、同 orbit-size canonical orbits 间置换结构 rows，保持 row multiset；
- `BAG_RESIDUAL`：同图结构均值广播到所有节点。

## 5. Strict two-stage checkpoint

每个 outer fold：

1. outer-train 固定 80/20 inner split；
2. 仅 inner-train 训练 GINE，inner-valid 选择 base epoch（max80/patience20）；
3. 重训 inner GINE 到 base epoch并冻结；仅 inner-train 训练 residual，inner-valid 选择 residual
   epoch（max60/patience15）；
4. 重训 full outer-train GINE 到固定 base epoch并冻结；full outer-train 训练 residual 到固定
   residual epoch；
5. outer-test 只评估一次。

各 residual variant 共用同一 inner/full frozen GINE、base logits、seeds 与容量。

## 6. Gate

- residual increment：TRUE−GINE `≥+0.5pt`，至少 2/3 folds 正；
- conditional binding：TRUE−SHUFFLED `≥+0.5pt`，至少 2/3 folds 正；
- conditional localization：TRUE−BAG `≥+0.5pt`，至少 2/3 folds 正。

三项全通过才扩展 model seeds。失败时停止 Mutagenicity 上的 Beam8 分类增量路线，不进入
atom gate、graph-supervised dictionary 或 cross-attention。
