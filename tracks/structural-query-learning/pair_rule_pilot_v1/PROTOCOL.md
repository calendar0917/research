# 冻结协议｜sq-lipo-pair-rule-v1

本文件是本轮**唯一科学协议**的人类可读版本。机器可读版本为
`tracks/structural-query-learning/protocols/sq-lipo-pair-rule-v1.yaml`（控制面
计算 `protocol_hash`），运行配置为
`pair_rule_pilot_v1/configs/sq-lipo-pair-rule-v1.yaml`。协议在正式远端运行前冻结；
任何改动都会改变协议 hash，并使旧产物不可比。

- 试验 ID：`sq-lipo-pair-rule-v1`
- 设计参照：`main@0d2ef5f4f72fbda3efbdbf590f2d0b02ec7da007`
- 唯一数据集：OGB `ogbg-mollipo`
- 官方 scaffold 划分：Train / Val / Test = **3360 / 420 / 420**
- 唯一训练 seed：**0**
- 所有正式臂：**180 epoch**，每 epoch 全量评估 Train / Val / Test
- 指标：原始 logD 单位 RMSE（主）与 MAE（次）
- `test_policy: always`；Test 逐 epoch 观测，标为 **DEVELOPMENT_TEST**
- 最佳状态：最小 Val RMSE；精确相同取较早 epoch
- 每臂同时报告 Val 选中状态与 epoch 180 状态

## 1. 数据与身份

唯一身份来源是 `gr_data.bundle("ogbg-mollipo", verify=True)`。复用已提交的图缓存
和 Lipo 原始缓存，SHA-256 分别为

```
fe002287fa9c581278bc5430f5b4a5b26b35d9d4d5207d138f15956d93123c7c  (graph cache)
3005aa55e766d83a237b1659ad2cabb51b8702eef6d82c203fdecf9cb3c42cc3  (lipo raw)
```

`prepare` 必须断言：三划分互斥、并集恰好覆盖官方 row ID `0..4199`、每个 split 的
row ID 与标签逐项一致、Train-only 标签标准化可逆、每个 split 的行数等于官方数字。
所有新矩阵按各 split 的**升序官方 row ID**写入。所有对照按官方 row ID 对齐。

图张量来自缓存：`node_ptr` / `bond_ptr` 切片、`bond_u/bond_v` 本图索引、物理键
唯一（`u < v`）。原子列语义和双键类型用 OGB 的解码表校验；缓存第三根键列是 OGB
的**共轭**编码（复用的 `sq_data.BOND_COLUMN_NAMES` 文本标签是旧命名，值不变）。

## 2. 规则类型与共同输入

### 2.1 原子规则类型

原子 `v` 的规则类型是整数元组

```
t(v) = (atom_feat[0], atom_feat[3], atom_feat[7], atom_feat[8], role)
       元素            形式电荷       芳香          在环          额外角色
```

全部使用 OGB 原始类别索引。若 C 原子与 O 以双键相连，该 C 的 `role=1`，该 O 的
`role=2`，其余为 `0`；一个原子命中一次即记录一次。`role=1/2` 只描述这个局部
C=O/O=C 环境，不引入任何标签。

### 2.2 共同输入（五组，按此顺序拼接）

每个计数对应一个规则组，组内两列 `u(n) = [min(n,1), max(n-1,0)]`。组内排序使用
稳定的规范字符串键；B 组保留 `sq_bank.build_base_bank` 的原始返回顺序。

1. **四个具名整图计数**：原子数、物理键数、连通分量数、环秩
   `n_bonds - n_atoms + n_components`。
2. **固定 B 查询库**：用 `sq_data.packed_split`、`sq_data.train_vocabulary`、
   `sq_bank.build_base_bank` 的原默认值 `cap=512`、`min_molecules=8` 构造。每个
   查询取 `sq_bank.query_columns(...)` 的首列（根节点命中次数），命中实例是对应
   根原子。
3. **带类型的局部两键路径 S2**：枚举 `u-v-w`，两根物理键均存在且 `u != w`；在
   同一中心的无序邻居对上实现，保证每条无向路径只计一次。键包含 `t(u), t(v),
   t(w)`、两根键的完整 OGB 三列类型、以及端点 `u,w` 是否直接相邻；正反向完整键
   取字典序较小者为规范键。候选需在至少 16 个 Train 分子中出现且 Train 计数非常
   量；按 Train 命中分子数降序、规范键升序，最多保留 256 个。
4. **端点类型对总数 `C_t`**：对每张图所有 `u < v` 锚点对，`t = sorted(t(u),
   t(v))`。对第 2.3 节最终槽位词表中出现的每种 `t`，加入全图计数 `C_t`。包含任何
   距离和不可达锚点对。
5. **整图距离边际 `H_d`**：对全部无序锚点对，按最短物理键路长加入距离箱
   `1,2,3,4,5,6,7,8,9+,unreachable`。`9+` 只含有限距离。总对数
   `T = n_atoms*(n_atoms-1)/2 = Σ_d H_d`。

恒定边际列按原样保留在输入、曲线和账目中。

### 2.3 REAL 槽位与 NULL 边际

REAL 的扩展规则键是 `(t, d)`，命中实例是一对实际锚点 `u < v`。只取有限距离箱
`3,4,5,6,7,8,9+` 为候选。筛选只用 Train：至少 16 个 Train 分子命中、原始计数在
Train 非常量；按命中分子数降序、规范键升序，最多 512 条。槽位和类别顺序在
Val/Test 特征构造前锁定，供所有 split 以及 REAL / NULL / MLP 共用。

REAL 每槽两列为 `u(K_(t,d))`，`K_(t,d)` 是实际锚点对计数。

NULL 的同一关系槽只用 `C_t, H_d, T`。在同一张图内把距离箱均匀重分配给所有锚点
对，虚拟命中次数 `K ~ Hypergeom(T, H_d, C_t)`；两列是同一基函数的精确期望：

```
mean_K = C_t * H_d / T
p_hit  = Hypergeom(T, H_d, C_t).sf(0)
null_cols = [p_hit, mean_K - p_hit]
```

任一边际为零或 `T=0` 时两列均为零。第二列仅对负舍入误差（`|x| <= 1e-10`）截为
零；更大负值视为实现故障。REAL 与 NULL 的扩展差异按行统计：至少一个已入库槽位的
任一**原始**基函数列绝对差 `> 1e-8`（缩放前检查）。若关系库为空，REAL/NULL 使用
合法零列，共同输入照常评估，并记录“结构对照覆盖不足”。

## 3. 五条正式轨迹

| 臂 | 输入 | 读出 | 设备 |
|---|---|---|---|
| BASE | 共同输入 | 分组稀疏加性，FULL 与 -32 | CPU |
| NULL | 共同输入 + 同槽位边际期望 | 同上 | CPU |
| REAL | 共同输入 + 同槽位真实关系 | 同上 | CPU |
| MLP | REAL 的完全相同输入、列序、缩放 | `D→128→64→1` MLP | GPU1 `cuda:0` |
| C1 | 官方图特征，仓库原模型 | LCRC-C1 / NO_BOND_MEAN | GPU0 `cuda:0` |

三条加性臂共享同一组共同列、同一尺度、同一数值的 `lambda`；FULL 与 -32 是同一条
训练轨迹的两种读出，不是两条训练。

### 3.1 加性求解

- 标签用官方 Train 的 mean / 总体 std（`ddof=0`）标准化；统一使用 `b.scaler`，
  结果只反标准化一次。
- 每个原始基函数列除以 `max(sqrt(mean_train(column^2)), 0.1)`。共同列在三臂中共用
  尺度；REAL / NULL 扩展各自用自己的 Train 列计算。
- 固定目标

  ```
  ||Xw + b - z||^2 / (2*n_train) + lambda * Σ_r ||w_r||_2
  ```

  每规则两列同组，偏置免惩罚。求解时中心化 Train 列和标准化标签以消元偏置；预测
  使用未中心化的缩放矩阵与解析偏置 `mean(z) - mean_train(X) @ w`。
- `lambda = 0.02 * lambda_max_REAL`，其中 `lambda_max_REAL` 是 REAL 全 Train
  中心化光滑损失在零权重处的最大组梯度 L2 范数。三臂使用同一数值。
- 全批次 `float64` FISTA；零权重初始化，`t=1, L=1`；每 epoch 一次近端更新。若

  ```
  f(w_new) > f(q) + grad(q)·(w_new-q) + (L/2)||w_new-q||^2 + 1e-12
  ```

  则 `L ← 2L` 并从同一 `q` 重算，接受后沿用该 `L`。组近端为
  `max(0, 1 - lambda/(L||v_r||_2)) * v_r`；零范数直接为零。标准 FISTA 更新
  `t_new=(1+sqrt(1+4t^2))/2`、`q_new=w_new+((t-1)/t_new)(w_new-w_old)`。
- 每 epoch 记录光滑损失、惩罚、目标、`L`、回溯次数、活跃组数；出现非有限值即工程
  失败。
- 每 epoch 同时输出 FULL 与 -32。FULL 使用全部权重。-32 只保留 `||w_r||_2 >
  1e-10` 中按 **Train 上该组输出的标准差**降序的前 32 组，平手按预先固定的组顺序；
  保留组权重不变，用保留组重算 Train 偏置。FULL 与 -32 各自保存 Val 最佳 epoch；
  REAL-32 的最佳 epoch 还保存同权重的 FULL 预测，用于同权重测截取代价。
- raw logD 输出：规则组分数乘 Train 标签 std，标签 mean 只加进偏置一次。两列全零
  的规则组分数为零。

### 3.2 MLP 与 C1

- MLP 输入是 REAL FULL 特征和同一尺度；结构 `D→128→64→1`，两隐藏层 SiLU +
  dropout 0.10，输出标准化标量。建模前 `torch.manual_seed(0)`；PyTorch Linear
  默认初始化；无其他输入分支。每 epoch 用
  `np.random.default_rng(0)` 单一流对官方 Train 本地行索引作排列，并复用
  `gr_common.lr_at("Rf", step, 3360, 180)`。优化器 AdamW（`0.9,0.999`，
  `eps=1e-8, wd=1e-5`），峰值 lr `1e-3`，5 epoch warmup，余弦到 `1e-5`，
  batch 128 保留尾批，标准化 MSE，梯度裁剪 5，FP32；eval 模式、eval batch 256、
  全量行。
- C1 用 `gr_models.ModelSet(b).lcrc()` 原构造，核对 **228674** 参数；用
  `gr_train.run_trajectory` 在独立目录训练 180 epoch，`recipe="Rf"`，
  task 的 key/stage/dataset/model 与协议一致；设备 `cuda:0`。GPU0 只跑 C1。

## 4. 实现核验（正式 worker 启动前）

`runs/prep/preflight.json` 记录误差与规模；任一项失败即停止，修复后用同一协议重验。

1. **行与划分**：缓存指纹、三 split row ID 与标签逐项一致、数量正确、互斥且覆盖
   `0..4199`；Train-only 标签标准化往返；尺度计算只读取 Train 行（对 Val 行施加扰动
   不改变已存尺度）。
2. **计数与不变性**：手工小链、带孤立点小图（物理键、S2、不可达、长距箱）；固定
   8 个官方 Train 行随机重标号后全部计数逐项不变；每张图核对 `Σ_d H_d = T`、
   `Σ_t C_t = T`、有限 `d>=3` 的类型-距离表之和等于对应的有限对数。
3. **可分辨关系的正对照**：同一 18 节点环，O/N 锚点分别相隔 6、8；物理对象边际、
   整图距离边际、局部 S2 计数相同；完整计数器中 O–N 距离 6 槽位在两图分别为 1 和
   0；两图该槽的 NULL 特征相同。人工词表强制保留该槽，不经过真实 Train 过滤。
4. **NULL 与求解**：`T=4,H_d=2,C_t=2` 时两列应为 `5/6, 1/6`，第一列在 `[0,1]`，
   两列和等于 `mean_K=1`；REAL 前处理的实际干预覆盖被记录；组近端闭式解与
   FULL/-32 偏置复算通过。
5. **运行**：两张卡各自用小批 Train 样本做前向与一次更新，检查输出/损失/梯度有限；
   smoke 独立标记，正式训练总是从固定初始化重新开始。

## 5. 指标与解释复核

- 保存每个正式 epoch 三个 split 的 RMSE、MAE、样本数；保存每种读出的 Val 最佳与
  epoch 180 的官方 row ID、raw 标签、raw 预测。用 `sq_data.official_metric` 从
  保存预测独立重算 RMSE 和 MAE，读回值容差 `1e-6` raw。
- 对 REAL-32 的 Val 最佳状态导出 `analysis/rules.json`：规范规则、两列 Train 尺度、
  两项 raw 单位系数（拟合权重 × Train 标签 std ÷ 列尺度）、raw 偏置、epoch、
  候选/模型 hash。独立规则重放入口从原图重新计算支持项，复现 Val/Test 全部预测，
  `max_abs_error <= 1e-6` raw。
- 对升序前 20 个 Val 分子列出非零规则中绝对贡献前 3 项；显示带符号分数、计数、
  对应锚点对 / S2 路径 / 局部根 ID，排序稳定。对全 Val 中最小 row ID 的实际命中
  关系样例，验证仅将该规则计数减一时的带符号预测差等于导出规则函数的单项差；没有
  活跃关系样例时标为不适用。统计 Val/Test 上使用真实关系项的分子比例、关系项非零
  贡献和规则数分布。
- 配对 cluster bootstrap：2000 次，seed 1703；Bemis–Murcko 骨架（去手性）为簇；
  无环空骨架用规范去手性 SMILES 分组；解析失败用 row ID 单独成组并计数。每次重抽
  相同数量簇、保留簇内全部行、两模型共用抽样行；报告差的 2.5%/97.5% 分位。
  Val/Test 各自保留官方划分，bootstrap 分组只用于评估。

## 6. 预定判读

主表使用各臂最小 Val RMSE 状态；另写固定 epoch 180 表。主对比对 BASE、NULL、REAL
使用 -32：

```
gain(control) = RMSE(control) - RMSE(REAL-32)     正数 = REAL 更好
```

关系轴按顺序取第一个满足的判定：

| 条件 | 判定 |
|---|---|
| Val 覆盖 < 84 个分子 | `LOW_RELATION_COVERAGE` |
| 覆盖足够，REAL-32 在 Val 无任何真实关系项绝对值 > `1e-6` logD | `NO_ACTIVE_RELATION_32` |
| 两个 Val gain ≥ 0.02、两个 Val 95% 区间下界 > 0、两个 Test gain > 0 | `BINDING_SIGNAL` |
| 两个 Val gain > 0，但任一 Val 区间下界 ≤ 0 或任一 Test gain ≤ 0 | `UNRESOLVED_BINDING` |
| Val 对 BASE-32 的 gain > 0，同时对 NULL-32 的 gain ≤ 0 | `ADDITION_NOT_BINDING` |
| 其余 | `NO_CLEAR_BINDING_SIGNAL` |

性能轴：

```
gap = RMSE(REAL-32) - RMSE(C1)
```

Val 与 Test 的 gap 都 `<= 0.04` 且规则重放通过 → `NEAR_C1`，否则 `NOT_NEAR_C1`。
两轴同时达到 `BINDING_SIGNAL` 与 `NEAR_C1` 才报 `PROMISING_PILOT`。若本轮 live
C1 最佳 Val RMSE 比历史 `0.7052` 高超过 `0.04`，附记 `WEAK_LIVE_REFERENCE`。

瓶颈线索（0.02 logD 为解读尺度）：报告 REAL-32 选中 epoch 的 FULL 与 -32 差、两者
各自选中状态的差、REAL FULL 与 MLP 的差及两者相对 C1 的表现。同权重去掉 32 项预算
使 RMSE 至少降低 0.02 → 优先定位压缩；MLP 的 Val/Test 都进入 C1 的 0.04 范围且 Val
比 FULL 至少低 0.02 → 优先定位读出/容量/拟合；两个读出 Val 都落后 C1 超过 0.04 →
记录当前证据词表可行性受限并报告 Train 拟合与末段目标变化；其余只报告差值、方向与
区间，不声称清楚定位。

## 7. 产物与完成定义

- `pair_rule_pilot_v1/PROTOCOL.md`（本文件）、配置、协议、study、运行入口。
- `runs/prep/`：`preflight.json`、规则词表、列序、尺度、哈希、矩阵、各 split 身份。
- 每臂 `status.json`、`curve.csv`、最佳/epoch180 预测；加性臂的 FULL/-32 文件名与
  表名一致；可恢复训练状态。
- `analysis/summary.json`、`bootstrap.csv`、`rules.json`、
  `explanation_cases.jsonl`、`resource_ledger.json`。
- `REPORT.md`：主要结果、两张表、两轴判定、瓶颈线索、复算结果、历史背景与不确定性。

大矩阵、模型权重、预测数组和恢复文件放在 git-ignored 运行目录；报告中保留可追溯
路径、代码 commit、协议 hash、运行 ID、设备环境和实际用时。五条正式轨迹完成、指标
复算与解释复算通过后才给出完整结论。

## 8. 执行与资源

- 远端统一通过 `rr`；先 `rr deploy res --pool res-gpu0` 与 `res-gpu1` 部署同一个
  已提交版本。
- `prepare` 在 `res-gpu1`（1 GPU / 4 CPU）完成共享缓存、CPU 预检和 MLP smoke。
- `worker_c1` 在 `res-gpu0` 只跑 C1；`worker_rules` 在 `res-gpu1` 先跑 MLP，再在
  同一 worker 顺序跑 BASE、NULL、REAL（CPU 数学库 ≤ 4 线程）。
- 从首个远端准备任务启动计，远端墙钟上限 **60 分钟**、A100 占卡上限
  **2 A100·小时**（含 smoke 和恢复）。CPU 求解时间另计。
- `rr pull` 后本地运行 `stage=aggregate`，生成分析、报告和小型复现记录。
