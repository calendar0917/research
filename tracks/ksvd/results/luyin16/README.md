# luyin16 results

本目录是 luyin16 阶段的结果入口。只提交小型 Markdown 摘要、配置快照、registry 或 checksum；原始 JSON、NPZ、PT、log 和缓存由 `.gitignore` 排除。

建议每个结果摘要回答：问题、协议、数据划分、输入分支、结果、限制和下一步。

`*_smoke/` 目录只验证管线，不进入 Git，也不能用于研究结论。导师思路概念复现的正式结果应来自冻结配置 `mentor-concept-replication-v1`，并明确区分 validation-only 开发结果和显式授权后的 terminal test。

导师概念复现阶段的综合判定见 [`MENTOR_CONCEPT_ROUTE_VERDICT_20260829.md`](MENTOR_CONCEPT_ROUTE_VERDICT_20260829.md)。那一阶段的完整 official validation 覆盖 chem code readout、S 特征块、pure-topology patch 和 624D typed-reconstruction proxy，均未评估 test；后续中心级路线的受控 test 评估另见本文末尾链接。

后续新增的 typed-slot proxy v2 和 occurrence-sum development 对照也已纳入上述判定：slot 对齐有部分支持，sum 聚合被否定，但导师 exact schema 仍未恢复。

ZINC 长程、统计对象、采样覆盖、K-SVD INIT/FINAL 与属性—结构解耦的完整
factorial 结论见 [`ZINC_LONG_RANGE_FACTORIAL_20260830.md`](ZINC_LONG_RANGE_FACTORIAL_20260830.md)。
对应 suite 已 8/8 完成；official test 仅作为冻结协议的终端核对，后续不得据此调参。

两个非重叠 train/valid 切片上的统计绑定、typed-shuffle 与 patch-relation 快筛见
[`ZINC_MECHANISM_SCREEN_20260830.md`](ZINC_MECHANISM_SCREEN_20260830.md)。当前结论是：
结构--属性绑定信号可复现，但 joint v1 与粗 relation 尚不满足 full 晋级条件；
下一步只筛一个 permutation-invariant `conditional joint v2`。

该 v2 已完成并作出终止判断，见
[`ZINC_CONDITIONAL_JOINT_V2_20260830.md`](ZINC_CONDITIONAL_JOINT_V2_20260830.md)：
binding 机制通过 shuffle 控制，但高维条件直方图未稳定超过 typed raw，因此不进入
full、Optuna 或 K-SVD。

进一步的 centered residual + train-only SVD16 终止快筛见
[`ZINC_CONDITIONAL_RESIDUAL_SVD16_20260830.md`](ZINC_CONDITIONAL_RESIDUAL_SVD16_20260830.md)。
它再次通过 true-shuffle binding 控制，但没有跨切片稳定补充 typed raw；因此
ZINC 上的结构--属性统计展开路线停止，后续转回采样覆盖和跨 patch 长程关系。

距离≥3 的 local-object/atom relation 快筛见
[`ZINC_LONG_RANGE_OBJECT_RELATION_20260830.md`](ZINC_LONG_RANGE_OBJECT_RELATION_20260830.md)。
pair vocabulary 覆盖充分，但预测增益与 position-shuffle 间隔均未跨切片稳定。

上述机制实验的统一结论与唯一下一步见
[`ZINC_MECHANISM_ROUTE_SYNTHESIS_20260830.md`](ZINC_MECHANISM_ROUTE_SYNTHESIS_20260830.md)。

radius-2、radius-3 与多尺度互补的终止快筛见
[`ZINC_MULTISCALE_RADIUS_20260830.md`](ZINC_MULTISCALE_RADIUS_20260830.md)。
multiscale 四片方向一致但未稳定达到 `0.01`，当前冻结 radius-3 raw 为 ZINC
单尺度 baseline。

MolHIV 上 exact rooted topology、automorphism-orbit binding 与
topology-conditioned chemistry 的 train-only 快筛见
[`EXACT_CONDITIONAL_FUSION_20260901.md`](EXACT_CONDITIONAL_FUSION_20260901.md)。
结论是：精确 orbit 位置绑定 no-go；patch 级 topology--attribute 配对通过 matched
shuffle 机制控制，但直接拼接仍未超过 `S+WL+attribute`，只允许再做一次
cross-fitted conditional residual gate。

中心级结构--属性交互快筛见
[`CROSS_CENTER_INTERACTION_20260901.md`](CROSS_CENTER_INTERACTION_20260901.md)。
当前 `cross-centre covariance` 相对 marginal 为 `+0.00785`、3/3 folds，
相对 centre-shuffle 为 `+0.00491`、2/3 folds；patch 内 binding 单独未胜
matched shuffle。该方向获得一次 train-only 确认实验权限，但尚未进入
official-valid、深度融合、预训练或 K-SVD update。

后续完全不重叠 confirmation、3000/1500 scale-up 与 mixed-null 的最终判定见
[`CROSS_CENTER_INTERACTION_ROUTE_VERDICT_20260902.md`](CROSS_CENTER_INTERACTION_ROUTE_VERDICT_20260902.md)。
`S+both` 在第三组图上相对 marginal 为 `+0.01024`、相对 matched double-shuffle
为 `+0.01023`，均为 3/3 folds；机制更接近“跨中心联合异质性作为上下文，patch
binding 作为条件修正”，而不是两个独立分支的粗拼接。随后测试的 norm/bilinear
低容量 gate 未超过 `S+both`，因此**机制阶段**冻结在“统计交互 + XGBoost”，不再向
深度融合、预训练或 K-SVD task update 扩展。

随后完成的 official-valid 冻结搜索与 controlled official-test 见：

- [`CROSS_CENTER_INTERACTION_OFFICIAL_VALID_20260902.md`](CROSS_CENTER_INTERACTION_OFFICIAL_VALID_20260902.md)：
  每个视图 16-trial Optuna 只在 official-train scaffold folds 调参；valid 上
  `S+marginal` 五 seed 均值 `0.841285`、ensemble `0.845047`，高于独立调参的
  `S+both` `0.838222/0.841259`。
- [`CROSS_CENTER_INTERACTION_OFFICIAL_TEST_20260902.md`](CROSS_CENTER_INTERACTION_OFFICIAL_TEST_20260902.md)：
  五个冻结视图均在 test 评估，test 不参与选择或调参；严格 train-only 下
  `S+both=0.808749`、ensemble `0.813525`，`S+cross_cov=0.800937`、ensemble
  `0.805587`。test 只有 130 个 positive，差异应视为泛化证据而非精确排名。
- [`CROSS_CENTER_INTERACTION_TEST_PCA_CONTROL_20260902.md`](CROSS_CENTER_INTERACTION_TEST_PCA_CONTROL_20260902.md)：
  固定 train-only PCA、只增加 train+valid 标签重训后，`S+both=0.803644`
  （ensemble `0.807049`）；相较 PCA 也重拟合的 `0.795642`，说明交互坐标系
  漂移是当前重要稳定性问题。

因此当前应分开表述：`S+marginal` 是 valid 上的简洁性能参考，
`cross_cov/binding` 是有机制含义的交互块，而 `S+both` 在独立 test 检查中显示
出最强的迁移潜力，但尚不足以声称一个不依赖 PCA scope 的最终稳定模型。下一步
更应把 `cross_cov` 作为稳定核心、把 binding 作为条件辅助块，并优先做冻结坐标下的
bootstrap/校准/错误分层，而不是继续引入 GINE、attention、
结构独立预训练或 K-SVD task update。
