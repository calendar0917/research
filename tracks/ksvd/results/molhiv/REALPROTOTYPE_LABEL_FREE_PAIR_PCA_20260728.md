# Real-prototype 无标签 pair-PCA 路线总结（2026-07-28）

## 一句话结论

> full official-train 三折达到预先固定的全部条件；下一步应先冻结配置，再只评估一次 official valid。

## 8k：三个训练随机种子

这里改变的只是模型训练中的随机性；结构词汇、PCA 压缩方向和数据划分保持不变。

| seed | compact base | covariance PCA | 增益 | real−打乱结构归属 | base 胜折数 | 对照胜折数 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.760277 | 0.762440 | +0.002162 | +0.002098 | 3/3 | 2/3 |
| 1 | 0.760277 | 0.762638 | +0.002361 | +0.002152 | 3/3 | 2/3 |
| 2 | 0.760277 | 0.762363 | +0.002086 | +0.002163 | 3/3 | 2/3 |

固定三 seed 概率平均：

- real fold AUC：`0.771541 / 0.703586 / 0.812166`
- mean gain over compact base：`+0.002154`
- real − assignment-shuffled：`+0.002077`

### 为什么 covariance PCA 更可信

它不看分子标签，只观察 outer-fit 分子里哪些具体结构组合经常一起变化。因此，保留下来的方向来自数据中反复出现的结构规律，而不是任意随机方向。

相反，correlation PCA 会先把每个组合都缩放到近似同等重要。这会把非常少见、估计不稳定的组合放大；8k 结果表明这种处理平均没有收益。

## Full official-train 三折开发验证

这仍然只是在 official train 内部做开发验证。official valid/test 的编码和评估次数都保持为 0。

- compact base：`0.788469 / 0.743996 / 0.806008`，mean `0.779491`
- covariance PCA：`0.792660 / 0.750703 / 0.804660`，mean `0.782674`
- covariance 增益：`+0.004191 / +0.006706 / -0.001348`，mean `+0.003183`
- covariance real − assignment-shuffled：`+0.004712 / +0.001832 / +0.000198`，mean `+0.002248`
- correlation PCA mean gain：`+0.000863`

### 预先固定的晋级条件

- PASS — `mean_gain_at_least_0.002`
- PASS — `at_least_2_of_3_base_wins`
- PASS — `real_mean_beats_assignment_shuffled`
- PASS — `at_least_2_of_3_matched_control_wins`

**总判断：PASS。**

需要注意，这不是“每一折都提高”：fold 2 相对 compact base 为 `-0.001348`。不过真实结构组合在三折都优于“打乱节点到原型归属”的对照，说明平均收益并不只是增加 144 个参数带来的容量效果。

## 当前研究解释

1. 单个局部结构是否出现还不够；距离 1/2 内的具体结构组合确实可能补充弱信息。
2. 但组合空间很大，不能任意压缩。随机投影的结果不稳定，说明信号不是“随便保留一些组合”就能得到。
3. covariance PCA 的稳定性说明，更有希望的是保留在许多训练分子中反复共同变化、变化幅度较大的组合。
4. 这条路线仍然不是把字典完全按标签训练；结构词汇和压缩都不看标签，只有最后的性质预测器看标签。这样能减少过拟合和数据泄漏风险。
5. official valid/test 当前均未用于编码、选参或评估。

## 与随机压缩路线的区别

此前 rank-8 随机 pair sketch 在三个 projection seeds 上不稳定：平均增益不足，且真实结构组合并不能持续胜过打乱归属对照。这里的 covariance PCA 不依赖随机方向，而是从训练分子里反复出现的共同变化中确定压缩轴；8k 三个训练 seeds 和 full 三折都保持正的平均收益。这是当前最关键的机制性进展。

## 工程与审计说明

full vocabulary 的目标×候选相似度矩阵达到约 11.5–16.1 亿个 float32 元素，直接在内存中构造会触发 OOM。runner 已改为：分块构造完全相同的正余弦矩阵、用磁盘映射保存、再按候选分块计算贪心增益。8k 回归检查中，`farthest` 与 `scaffold_facility` 的 32 个原型行和 prototype hash 与旧实现完全一致。这个修改只改变存储方式，不改变选择目标。

## 冻结状态

已写入 `label_free_pair_pca_official_valid_freeze_v1.json`。冻结内容包括：PCA64、两组各 32 个真实原型、top-3 分配、距离 1/2、covariance PCA rank 144、compact base 残差上限 0.3125、pair residual 上限 0.25、30 epochs，以及 pair-head seeds 0/1/2 的固定概率平均。

在冻结时：official valid/test 编码与评估次数仍为 0。下一步若运行 official valid，只允许按 manifest 做一次固定评估；不得根据 valid 重新选择 rank、优化器、correlation PCA 或随机投影。

## 后续纪律

- 若 full 门槛通过：先写冻结清单，再进行一次 official valid；不能看到 valid 后再改 rank、残差上限或重新启用失败分支。
- 若 full 门槛失败：保留 compact exact-distance 关系模型，停止 pair-PCA，不评估 official valid/test。
