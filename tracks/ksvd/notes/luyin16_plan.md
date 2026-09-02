# luyin16 阶段计划

来源：[`docs/luyin/luyin16.txt`](../../../docs/luyin/luyin16.txt)。本阶段不改变 luyin14 已冻结的 no-go 结论，重点重新确认 KSVD 结构表征的功能定位。

代码入口：可复用逻辑从 `../src/ksvd_research/` 导入；实验 runner 只新增到 `../experiments/luyin16/`。`../code/` 保留历史复现，不作为本阶段的新文件入口。

## 阶段问题

1. 结构表征脱离 GNN 后是否具有可重复的独立价值？
2. 字典 `D`、稀疏系数 `X`、节点属性是否在当前代码中被正确区分和使用？
3. 局部采样/K-SVD 在回归或长距离任务上的不足，是否来自感受野限制？

## 进入实验前的审计

- 核对 TU 数据集和回归数据集的确切名称、任务和指标。
- 固定数据划分和随机种子；HIV 使用官方 scaffold split。
- 字典、标准化参数和特征选择只在训练部分拟合；验证/测试只编码和评估。
- 记录下游输入是 `D`、`X`、重建结果、节点属性还是结构统计。
- 给每次运行记录 `protocol_id`、代码版本、环境版本和输出路径。

## 最小实验矩阵

在相同 split、seed、分类器和预算下比较：

```text
structure-stats
raw/init sparse code
trained K-SVD code
node attributes
GNN baseline
structure + attributes
```

只有结构-only 显示稳定信号后，才比较 concat、residual、gate/FiLM 等少量融合方式；暂不引入 cross-attention 或新的 patch Transformer。

## 长距离诊断

固定特征维度和评估协议，比较 radius-1、radius-2、更大局部范围或图级结构，并同时记录覆盖率、patch 规模、重建误差和下游指标。

## 结果判定

本阶段结束时只回答四件事：结构是否独立有效、KSVD 更新是否有任务归因、融合是否产生稳定增益、局部感受野是否解释回归失败。所有原始输出放 `results/luyin16/`，只将小型摘要和 manifest 纳入版本控制。

## 导师思路概念复现 v1

在拿不到导师上游特征代码时，使用 `mentor-concept-replication-v1` 复现研究结构而非具体数值：

- `S_v1`：显式拓扑统计 + OGB 原子/键 categorical composition；
- `T_init_v1`：train-only 初始化字典的 rich sparse-code readout；
- `T_final_v1`：从同一初始化进行 K-SVD 更新后的 matched readout；
- 下游：固定参数 `XGBClassifier(binary:logistic)`；
- 主要差值：`T_final−T_init`、`S+T_final−S`；
- 默认只报告 train/valid，冻结后才允许显式运行 official test。

执行顺序固定为：300 图 smoke 验证管线；8000 图、K32 development 检查信号与成本；冻结配置后运行完整 official split；最后才显式授权一次 test 评估。development 子采样结果不作为最终 MolHIV 数字。

首次 development 运行完成后，K32 设置冻结为 `mentor-concept-replication-v1-k32-official-valid`，用于完整 official validation 复核。该复核不承担 K64 容量搜索，也不允许查看 test。

在 K32 chem-patch 复核后，只增加一个 matched topology-patch 对照：采样、字典预算、稀疏度、分类器和 official split 全部不变，仅令 `patch_feat=topo`。它是普通无监督 K-SVD 路线的停止判定实验，不是新一轮超参数搜索。

## 2026-08-29 阶段结论

完整 official validation 表明：显式拓扑与化学组成的固定特征配合 XGBoost 有效（`S=0.7817`），但 ordinary unsupervised K-SVD 的 chem/topology code readout 和 624D typed-reconstruction proxy 均没有稳定的 FINAL−INIT 或 S 外增量。普通 K/T、Beam、fusion 扫描停止；综合证据与后续权限见 `../results/luyin16/MENTOR_CONCEPT_ROUTE_VERDICT_20260829.md`。导师 exact replication 仍需 69/624D 上游 schema、payload 和 XGBoost best params。

补充的 typed-slot proxy v2 显示，保留 `8×64 + 28×4 = 624` 的 slot-aligned typed object 后，raw view 从全局分布 proxy 的 `0.6649` 提升到 `0.7014`，且 `FINAL−INIT=+0.0216`；occurrence-sum development 仅 `0.5821`，因此“统计对象/对齐重要”得到部分支持，但仍不足以解释 `S=0.7817` 或导师结果。

2026-08-30 的 T-readout 消融进一步定位了损失位置：raw `mean(Y)=0.7014`、
reconstruction `mean(DX)=0.6687`、absolute residual `0.7116`；12-trial
`S+T_raw` 虽在随机 inner-CV 比 S 高 `0.0014`，但 official-valid 低
`0.0159` 且 0/5 seed wins。当前 typed-slot T schema 关闭，详见
`../results/luyin16/T_READOUT_ABLATION_20260830.md`。后续候选统一采用
3-trial scaffold-aware screen；明显弱项不再完整 Optuna/多 seed。

2026-08-30 的 radius-2 全中心实验改变了阶段判断：每个原子作为中心，保留
全部距离≤2 的局部 patch，再做 `12×52=624D` typed distribution，
`S+R_raw` official-valid 经 train-only scaffold screen 和 12-trial Optuna
达到 `0.8307`；冻结后 train+valid refit test 为 `0.7804`，`S+R_final`
为 `0.8022`。同预算 pure-topology `12×28=336D` 对照 valid 为
`S+R_raw=0.8100`、`S+R_final=0.8177`，test refit 分别为 `0.7627/0.7751`。
因此当前 luyin16 可进入“局部结构分布对象及其任务增益”的机制研究阶段，
但不能把增益单独归因于 K-SVD 更新；导师真实 69/624D schema 仍待取得。

## 2026-08-30 ZINC 长程诊断

官方 PyG ZINC-12K `10000/1000/1000` split 的可恢复 suite 已 8/8 完成，详见
`../results/luyin16/ZINC_LONG_RANGE_FACTORIAL_20260830.md`。固定 `max_nodes=12`
时，radius-1/2/3 的节点对覆盖率为 `0.239/0.495/0.679`，融合 view 的 fixed
valid MAE 为 `0.5436/0.5430/0.5225`；6-trial Optuna 后为
`0.5151/0.5139/0.4953`。radius-3 的长程增量成立。

radius-3 将 `max_nodes` 从 12 增至 20 后，截断率从 `0.0615` 降至近 0，
但 valid 基本不变，说明截断不是主要解释。K-SVD 虽持续降低重建误差，
FINAL 对任务的改善却不稳定，尤其 official test 上常由 INIT 更优。因此下一阶段
冻结 test，不再扫 K/T；优先做 permutation-invariant object、block-wise joint
statistics 和跨 patch 距离/重叠关系，再用 train folds 做 INIT/FINAL 归因。

## 2026-08-30 ZINC 机制快筛

radius-2 的两个非重叠 `5000/500` train/valid 切片表明：`global+typed raw`
均稳定优于 global；joint v1 相对图内 attribute-shuffle 的平均 MAE 间隔分别为
`0.02195/0.01914`，支持结构--属性绑定信号存在。但 joint v1 未超过 typed raw，
而 raw+relation 对 raw 的增益为 `0.02525/0.00578`，未跨切片通过 `0.01`
门槛。因此 joint/relation 均不直接晋级 10K/full，也不引入 K-SVD 或 Optuna。

下一轮只允许一个 `conditional joint v2` 快筛：用 patch 大小、边数、cycle rank、
中心度和 radius shell 大小等 permutation-invariant 结构签名，统计条件 atom/bond
分布，并保留图内 shuffle 归因。先跑两个非重叠 `2000/200` 切片；完整数字、
晋级规则和限制见
`../results/luyin16/ZINC_MECHANISM_SCREEN_20260830.md`。

conditional joint v2 随后完成两个 `2000/200` 和两个 `5000/500` 切片。
大切片 true 相对 shuffle 的平均 MAE 间隔为 `0.01559/0.03609`，binding 机制
成立；但相对 typed raw 的增益为 `-0.02189/+0.00797`，预测门槛失败。
因此不做 10K/1K、Optuna 或 K-SVD。若继续，只允许一次 train-only 16D SVD
的 centered conditional residual 快筛；详见
`../results/luyin16/ZINC_CONDITIONAL_JOINT_V2_20260830.md`。

最后的 centered conditional residual + train-only SVD16 快筛也已完成。两个
`2000/200` 切片 true 相对 shuffle 的平均间隔为 `0.04408/0.02089`，但相对
typed raw 的增益为 `+0.02567/-0.00155`，未跨切片通过严格 gate。因此停止
结构--属性联合统计、SVD 维度和 K-SVD 搜索。三轮控制共同支持“binding 存在，
但不是稳定的 ZINC 标签增量”；下一重点转回 radius-3 已支持的采样覆盖和跨 patch
长程关系。详见
`../results/luyin16/ZINC_CONDITIONAL_RESIDUAL_SVD16_20260830.md`。

跨 patch 长程关系随后用距离≥3 的 atom-pair relation 和 position-shuffle 做了
两个 `2000/200` 切片。relation 相对 raw 的增益为 `+0.00591/-0.02809`，
shuffle 间隔也未跨切片稳定；训练 pair coverage 为 100%，可排除词表覆盖问题。
因此 radius-3 增益不能简化为 long-distance atom-pair bag。当前唯一下一步是
radius-2、radius-3 与 radius-2+3 的 matched 多尺度快筛。完整阶段综合判断见
`../results/luyin16/ZINC_MECHANISM_ROUTE_SYNTHESIS_20260830.md`。

最终的 r2/r3 multiscale 快筛也已完成。r2+r3 在四个切片均为最优，但相对最佳
单尺度的增益为 `0.01066/0.00549/0.01501/0.00471`，未稳定达到 `0.01`，
因此不进入 full。5000 图时 r3 在两片均优于 r2，支持更大感受野需要更多样本。
当前冻结 ZINC 单尺度 baseline 为 radius-3 typed raw（cap20），multiscale 只保留
为弱候选；详见 `../results/luyin16/ZINC_MULTISCALE_RADIUS_20260830.md`。

## 2026-08-31 MolHIV 对象审计与机制 gate

全中心 radius-2 历史对象未通过基本审计：重标号导致 typed readout 最大漂移
`0.3665`、冻结 XGBoost 预测最大漂移 `0.3454`；14.12% patch 存在“8 节点
邻接 + 完整 ego 属性”的范围错位；原子 `%16` 在 55 个已观察类别上发生系统
碰撞。因此旧 `S+R_raw=0.8307` 只保留为历史协议数字，不再承担机制结论。

修正对象改用完整 radius-2 ego 的 root-shell/度、shell-pair 边统计，以及直接
OGB atomic-number/bond-type 类别。128 图、每图 3 次重标号的 topology、attribute、
joint readout 漂移均精确为 0。随后只在 official-train 的三个 scaffold folds 上
做固定预算筛查：

- `S+joint RAW-S`：`+0.0548/+0.0446/-0.0195`，均值 `+0.0266`，2/3 胜；
- `S+attributes RAW-S` 均值 `+0.0310`，强于 topology 的 `+0.0112`；
- joint standalone 在 3/3 folds 都低于 attributes standalone，局部属性是主信号；
- FINAL 将 valid reconstruction error 相对 INIT 降低约 20.5%–26.3%，但
  `S+FINAL-S+INIT=-0.0286/-0.0340/+0.0402`，均值 `-0.00746`，仅 1/3 胜。

阶段判定：保留 invariant all-center local statistics；将普通 K-SVD update 继续
定位为 reconstruction/compression diagnostic，而不是已成立的任务增益来源。
不再运行 ZINC、K/T/iteration 或 Optuna 搜索。若继续机制研究，唯一优先控制是
在已胜出的 attribute block 上比较 empirical INIT、Gaussian/PCA compression 与
FINAL；它回答“INIT 增益来自真实 patch 原型还是通用降噪”，不属于性能扫参。

## 2026-09-01 role × attribute binding screen

为直接检验“结构角色与属性的对应关系是否带来任务信息”，新增了不依赖
K-SVD 的最小模型 `luyin16-molhiv-role-attribute-binding-screen-v1`：每个完整
radius-2 ego 中，结构角色只由 shell、诱导度数 bin 和局部 cycle membership
决定；原子/键属性独立使用 compact OGB semantics。对每个 patch 分别计算
`P(role,attribute)`、`P(role)`、`P(attribute)`，并以
`P(role,attribute)-P(role)P(attribute)` 作为 centered binding。图内独立打乱属性
实体作为保留两端边际的 null control。

对象审计在 64 个 official-train 图、每图 2 次重标号下通过，所有真值块最大漂移
不超过 `2.4e-7`。官方 train 内部三 scaffold folds（每折 6000/3000）中，固定
XGBoost 的 `S+binding` 平均 AUC=`0.7393`，`S`=`0.6939`；binding 相对
`S+marginals` 为 `+0.0315`、相对 `S` 为 `+0.0454`，true 相对两次 shuffle
平均为 `+0.0701`，三折均胜。更换 XGBoost seed `[0,1,2]` 后平均仍为
`S+binding=0.7380`、`S=0.6981`。

冻结到 official-train → official-valid 后，`S=0.7775`、`S+binding=0.8017`，
相对 `S+marginals` 为 `+0.0172`，相对 `S` 为 `+0.0243`；但 centered
binding 对 shuffle 的平均差仅 `+0.0014`，未过预注册 `+0.003` 机制门槛。
未中心化 joint 对 shuffle 的差为 `+0.0170`，因此当前可下的结论是：

1. 结构—属性联合表示在 MolHIV 上有稳定的绝对任务增量；
2. 结构角色—属性依赖在内部 scaffold folds 中很强，但在 official-valid 外部
   split 上，不能把全部增量归因于 centered binding 的因果式对应关系；
3. 节点 binding（平均 `0.7405`）强于边 binding（`0.7158`），下一步不应先加
   cross-attention、K/T 或字典容量，而应先做节点角色定义/分层统计的单一机制对照。

原始输出：
`../results/luyin16/role_attribute_binding_screen/`、
`../results/luyin16/role_attribute_binding_screen_seedconfirm/` 和
`../results/luyin16/role_attribute_binding_screen_official_valid/`。官方 test
未编码或评估。

## 2026-09-01 cross-centre interaction 快筛

在固定 invariant all-centre radius-2 rooted-WL patch 上，新增不依赖 GINE、
attention 或 K-SVD 的中心级交互：对每张图计算拓扑行与属性行在不同中心之间的
中心化协方差，并用每折 train-only PCA-8 压缩。三组 `1200/600`
official-train scaffold 子样本中，`S+marginal+cross_cov` 相对
`S+marginal` 平均 `+0.00785`、3/3 folds；相对 centre-shuffle 平均
`+0.00491`、2/3 folds。加入 patch 内 binding 后相对 marginal 为
`+0.01181`、3/3 folds，但 binding 单独不胜 matched patch-shuffle。

因此当前可行性判断改为：优先确认“跨中心结构--属性联合异质性”，而不是继续
扩大单 patch 精确 binding、cross-attention 或 K-SVD。该结果只来自 train-only
快筛，尚不进入 official-valid；限制与唯一确认步骤见
`../results/luyin16/CROSS_CENTER_INTERACTION_20260901.md`。
