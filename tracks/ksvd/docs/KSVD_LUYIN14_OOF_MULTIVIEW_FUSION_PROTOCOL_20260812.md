# luyin14：真实数据上的 OOF 多视图融合协议

> 日期：2026-08-12  
> 状态：结果前冻结  
> 数据：只使用四个现有 TUD 数据集，不构造新 benchmark。

## 1. 为什么这一轮不同于已有 concat/gate

已有实验把数十维节点/图统计与 72--248 维 KSVD readout 直接拼接，或在同一小网络中联合
训练两个高维分支。这要求小数据分类器同时完成模态内建模、尺度校准和模态融合，很容易让
弱结构分支污染强节点/统计分支。

本轮采用 prediction-level multi-view fusion：

1. base view 与 KSVD view 各自独立训练、各自产生 class probabilities；
2. outer-train 内通过严格 cross-fitting 得到每个训练图的 out-of-fold (OOF) 预测；
3. 融合器只读取低维 logits、confidence 和 disagreement；
4. outer-test 只由在完整 outer-train 上重训的两个专家和 OOF 冻结融合器预测。

这样融合器从未看到任何 base/KSVD 专家对其自身训练样本的 in-sample 预测。

## 2. 冻结数据与专家

- 数据集：`IMDB-BINARY / IMDB-MULTI / MUTAG / PTC_MR`；
- outer split：seeds `0/1/2`，每个 3-fold stratified CV；
- inner cross-fit：每个 outer-train 内固定 3-fold stratified，seed=`1729+outer_seed`；
- 所有 inner OOF KSVD dictionary 均只使用对应 inner-train graphs；inner-valid patch 不进入
  centering、initialization 或 K-SVD updates；
- outer-test expert 使用完整 outer-train 拟合的 dictionary/classifier；
- FAIR95、rooted-canonical patch、`K24/T3/iterations5` 与 rich readout 完全沿用前一轮。

base view：

- IMDB：`STATS`；
- MUTAG/PTC_MR：`FEATURE_STATS`（node feature mean/max/sum + graph stats）。

structure view：

- `INIT_RICH`；
- `FINAL_RICH`。

所有一级专家均使用 `StandardScaler + LogisticRegression(C=1, max_iter=5000)`。

## 3. 冻结融合器

对 INIT/FINAL 分别报告：

1. `FIXED_AVG`：两专家概率 0.5/0.5 平均，无训练参数；
2. `LOGIT_ADD`：centered log-probability 相加；
3. `OOF_SCALAR`：固定 base logit 系数为 1，仅用 OOF log-loss 学一个
   `alpha in [-1,1]`：`L = L_base + alpha L_struct`，附 `0.01 alpha^2`；
4. `OOF_GATE`：用 OOF 学三参数置信门：
   `w=sigmoid(b0+b1(conf_struct-conf_base)+b2*JS(base,struct))`，输出
   `(1-w)p_base+w p_struct`，参数范围 `[-6,6]`、L2=`0.01`；
5. `OOF_STACK`：读取 base/structure centered logits 的低容量 logistic meta-classifier，
   `C=0.1`。

不扫描权重、C、gate width 或 inner folds。FINAL 与 INIT 必须用完全相同的融合协议。

## 4. 必报归因

- component：BASE / INIT / FINAL；
- 每种 FINAL fusion 相对 BASE 的 paired delta；
- 同种 `FINAL fusion - INIT fusion`，判断 K-SVD updates 是否贡献；
- OOF scalar alpha、gate 平均结构权重；
- base/structure OOF error disagreement：结构专家是否真的在 base 错误样本上提供互补命中。

## 5. 晋级规则

单数据集稳定通过仍要求 mean paired delta `>=+0.01` balanced accuracy 且 wins `>=6/9`。

### 5.1 融合 gate

至少一种 OOF 融合器相对 BASE：

- 至少 2/4 数据集稳定通过；
- MUTAG/PTC_MR 至少一个通过，另一个不低于 BASE 超过 1 point。

### 5.2 KSVD attribution gate

通过融合器的 FINAL 版本必须相对相同 INIT 版本：

- 至少 2/4 数据集 mean 为正；
- 跨四数据集 mean delta 为正；
- 不允许主要收益只来自 fixed average 的方差压缩。

两项同时通过，才继续探索更深的 multimodal interaction，例如 token-level cross-attention。
若融合 gate 通过但 attribution gate 失败，则只说明多视图集成有效，不能归因为 K-SVD。
若 OOF 融合全部失败，则问题不是 concat 太简单，而是当前 KSVD expert 没有稳定互补误差；
不进入更高容量融合。

