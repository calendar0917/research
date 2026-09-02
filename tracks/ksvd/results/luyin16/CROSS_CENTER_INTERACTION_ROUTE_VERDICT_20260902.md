# MolHIV 中心级结构–属性交互路线：确认与边界（历史机制阶段）

> **后续冻结评估提示（2026-09-02）**：本文记录的是 train-only mechanism
> screen/scale-up，仍然有效地说明了交互对象的机制方向，但其中“`S+both` 作为
> 当前终点”的性能判断已被后续正式评估补充。请以
> [`CROSS_CENTER_INTERACTION_OFFICIAL_VALID_20260902.md`](CROSS_CENTER_INTERACTION_OFFICIAL_VALID_20260902.md)
> 和 [`CROSS_CENTER_INTERACTION_OFFICIAL_TEST_20260902.md`](CROSS_CENTER_INTERACTION_OFFICIAL_TEST_20260902.md)
> 为冻结的 valid/test 数字来源。

后续结果显示：official-valid 独立调参的最高均值是 `S+marginal=0.841285`；在
controlled official-test 上，严格 train-only 的 `S+both=0.808749`（五 seed
ensemble `0.813525`），而 `S+cross_cov=0.800937`。因此机制结论保留，但不能
再把某一个 split 上的 `S+both` 排名直接写成跨 split 的唯一主线；跨三种 PCA
scope 读法更稳定的是 `cross_cov`，binding 应视为条件辅助块。

日期：2026-09-02  
协议：`luyin16-molhiv-cross-center-interaction-*`

## 结论先行

当前最有希望的对象不是“单个 patch 内精确的 role–attribute binding”，而是：

> 先描述一个分子中不同中心环境的结构–属性联合变化（`cross_cov`），再让
> patch 内 binding 作为条件修正项进入同一个下游模型。

这已经通过了一次完全不重叠的 3000/1500 scale-up：

| 比较 | 平均 AUC 增量 | 折级胜负 |
|---|---:|---:|
| `S+both` − `S+marginal` | **+0.010243** | **3/3** |
| `S+both` − matched double-shuffle | **+0.010226** | **3/3** |
| `S+both` − `S+cross_cov` | +0.006714 | 2/3 |
| `S+both` − `S+binding` | +0.005179 | 2/3 |
| `S+both` − late score fusion | +0.005179 | 2/3 |

因此可以把 `S+both` 作为当前路线的可解释候选终点；但还不能说已经得到一个
最终稳定的 0.80+ 模型，也不能把它包装成深度多模态预训练已经成功。随后做的
低容量 gate 筛选没有超过 `S+both`，所以没有继续堆叠显式深融合算子。

## 三阶段证据

三阶段都只在 official-train scaffold folds 上做监督评分，XGBoost 参数和
model seeds 固定为同一组；高维交互块只用每折 train 拟合 PCA-8。

| 阶段 | train/valid 每折 | 图级 union | `S+marginal` | `S+cross_cov` | `S+binding` | `S+both` |
|---|---:|---:|---:|---:|---:|---:|
| discovery | 1200/600 | 5121 | 0.672615 | 0.680463 | 0.674361 | 0.684422 |
| disjoint confirmation | 1200/600 | 5073 | 0.726045 | 0.729040 | 0.727783 | 0.730418 |
| disjoint scale-up | 3000/1500 | 11045 | 0.709778 | 0.713307 | 0.714842 | **0.720022** |

confirmation 排除了 discovery 的全部 5121 个 dataset indices；scale-up 又同时
排除了前两轮的 10194 个 indices。三轮之间图级不重叠，且每折 train/valid
不重叠。所有 true/null readout 的随机重标号审计通过，最大漂移为
`1.79e-7`。

## 机制拆解

### 1. `cross_cov` 是真实但不够稳定的主轴

在 scale-up 中，`S+cross_cov` 比中心置乱平均高 `+0.005330`，比 marginal 高
`+0.003529`，两个 gate 都是 2/3。它说明同一分子不同中心的结构–属性共同变化
不是纯粹的中心边际统计；但 fold 0 相对 marginal 为负，所以不能单独把它当作
最终路线。

### 2. patch 内 binding 单独并没有被确认

scale-up 中 `S+binding` 比 marginal 高 `+0.005064`，但比 patch-shuffle 低
`-0.001585`。这表示 binding 块可能有预测容量，却没有稳定证据表明“精确的
role–attribute 对应关系”本身是跨折可复现的机制。discovery 的早期正结果因此
不应继续解释为 binding 已经成立。

### 3. binding 在真实 cross 环境下成为条件修正

最关键的 mixed-null 是：保持真实 `cross_cov`，只把 binding 置乱。scale-up 中

`S+both` − (`cross_true + binding_shuffled`) = **+0.008416，3/3 folds**；

在 9 个 fold×model-seed 配对中有 8 个为正（平均 +0.008416）。这说明 binding
不是一个独立稳定的边际信号，但在真实的跨中心结构–属性环境已经确定以后，
它能提供额外的局部修正。

反方向的 mixed-null（保持真实 binding、置乱 cross）为 **+0.005971，2/3 folds**，
9 个 fold×seed 配对中 6 个为正。它较弱，支持一个有层次的解释：

```text
中心间联合异质性（主上下文）
        ↓
patch 内结构–属性 binding（条件修正）
        ↓
图级分类
```

### 4. 为什么 feature-level joint 比 late fusion 好

固定 50/50 的 marginal/binding score late fusion 在 scale-up 为 0.714842，而
同样信息放进一个 XGBoost feature view 的 `S+both` 为 0.720022。这个差异支持
“让下游模型看到两个块并进行条件分裂”比先各自出分数再平均更合适；但它仍然是
可解释的 PCA 块 + 树模型交互，不等于已经验证了 attention 或深度多模态网络。

### 5. 显式深融合 gate 的反证

在同一第三组 scale-up 上，固定 `S+both` 的两个 PCA 块，只额外加入一个低容量
门控或双线性块：

| 候选 | 平均 AUC | 相对 `S+both` | 折级胜负 |
|---|---:|---:|---:|
| `S+both+norm_gate` | 0.719682 | −0.000340 | 1/3 |
| `S+both+bilinear_gate` | 0.711673 | −0.008349 | 0/3 |

`norm_gate` 相对自己的 double-null 仍有 +0.003234，但它没有超过真实的
`S+both`；双线性 gate 同时低于真实和 null。这个结果很有用：它说明“有条件
交互”是真实的，但不需要再人为指定一个显式乘法/门控结构，XGBoost 在现有
低维块上做树分裂已经能承载这部分条件关系。

## 对当前路线的判断

1. **不需要因为 GINE 低于 0.80 就回头诉诸 GINE。** GINE 学的是消息传递平滑
   的端到端表示，当前显式统计保留了 GINE 可能丢掉的中心分布、稀有属性和跨中心
   异质性。两者的归纳偏置不同，显式对象更强并不矛盾。
2. **“结构 + 属性”不应再理解为一次拼接。** 目前证据支持两个层次：
   `cross_cov` 负责分子内部环境组织，binding 负责在该上下文下校正局部角色。
3. **暂缓结构独立预训练。** 先把条件关系固定并验证稳定性；否则会得到两个各自
   可重构、但语义空间未必对齐的分支，正好放大当前“割裂”问题。
4. **K-SVD 暂时只做压缩/诊断。** 现阶段没有证据表明无监督 dictionary update
   会创造这个交互信号；先不要把它引入机制判定。

## 当前路线边界

低容量 gate 已按预先约定完成且未超过 `S+both`，因此当前路线定格为：

```text
invariant radius-2 all-centre statistics
    + marginal topology/attribute distributions
    + cross-centre topology–attribute covariance
    + patch binding as a conditional auxiliary block
    → fixed XGBoost
```

如果还要投入预算，优先做冻结表示后的稳定性/解释性分析（fold bootstrap、
特征块消融、校准和错误分层），或在预注册的正式 train/validation 协议上做一次
单点复核；不建议现在再引入 GINE、attention、结构独立预训练或 K-SVD task
update。这样可以把“机制已被确认”和“架构仍可继续堆叠”明确分开。

## 可复现实验产物

- runner：[cross_center_interaction_screen.py](../../experiments/luyin16/cross_center_interaction_screen.py)
- discovery：[summary.md](cross_center_interaction_screen/summary.md)
- confirmation：[summary.md](cross_center_interaction_confirmation/summary.md)
- scale-up：[summary.md](cross_center_interaction_scaleup/summary.md)
- 配置：[confirmation yaml](../../configs/luyin16/cross_center_interaction_confirmation.yaml)、[scale-up yaml](../../configs/luyin16/cross_center_interaction_scaleup.yaml)
- 测试：[test_cross_center_interaction_screen.py](../../tests/test_cross_center_interaction_screen.py)
