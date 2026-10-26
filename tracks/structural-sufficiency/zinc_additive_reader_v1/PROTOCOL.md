# PROTOCOL — `zinc_additive_reader_v1`

**状态：正式训练前冻结。** 协议、配置 `configs/ar_v1.json`、代码与 CPU gate 产物
`analysis/cpu_checks.json` 在第一条正式轨迹启动前提交到 Git；训练开始后不再修改
结构、配方、阈值、seed 数或分析口径。

**边界：official test 不构造、不读取、不评估；official valid 不加载、不用于选点、
不报告 MAE。** 本轮所有监督标签、Monitor 选点与 Locked 评分都来自 official-train
缓存内的冻结 BJOC 行。

---

## 1. 问题与假设

已验证的事实（`zinc_dccl_bottleneck_v1`）：DCC-L 在 Fit8k 上的 Locked MAE 为
0.0836 / 0.0834（两 seed），误差集中在 Fit8k 支撑不足的局部环境；Fit4k→Fit8k
的 −0.038 说明瓶颈是**共享函数的估计方差**，而不是某个可查表的区域缺陷。

本轮只检验一个受控机制假设：

> **H_ADD**：在有限标签下，DCC-L 的损失部分来自“先把全部结构状态聚合成一个
> 宽向量，再交给自由度很高的整图 MLP”；把预测约束成“共享局部贡献之和 + 小型
> 全局修正”，在相同输入信息、相同局部状态、近参数量的条件下，能改善未见结构的
> Locked 泛化。

**不是**架构搜索，不声称论文原创性，不把内部 Locked 收益换算成 official test 成绩。

## 2. 固定数据与角色

复用 `tracks/structure-contribution/results/bjoc_predictive_bottleneck_v1/split_a.json`
的冻结划分（10k official-train 缓存内的划分）：

| 角色 | 行数 | 用途 |
|---|---:|---|
| Fit8k | 8000 | 监督训练 + **仅 Fit8k** 标准化 + 支撑统计 |
| Monitor | 1000 | 每 epoch 评估；严格最小 Monitor MAE 选 best checkpoint |
| Locked | 1000 | **四条正式轨迹全部完成并冻结后一次性读取**；不参与任何选择 |
| official valid | — | 不加载、不评估、不报告 |
| official test | — | 不构造、不读取、不评估 |

行号 SHA-256（int64 little-endian）：

```
fit8k   1cf6a2ec73ff5dfc182d72a7d4cd0148d31f3ae2488ee9943132f1aa51d7455a
monitor d493dc7fc547e062ee16301bb42f350d5eb6d5d773215254c0d84098a410933b
locked  9cbefd5e84cf7f4e8f7314280f35f636a0b18753a2f93a724b935e65a208cf6d
```

三者互不相交、并为 10,000；Fit4k（4000）嵌套于 Fit8k。`ar_common.split_rows()`
在每次运行时重新计算并核对 hash；不匹配即拒绝运行。

## 3. 模型（只替换 reader）

两臂保留 DCC-L 的全部局部结构计算：`P3Encoders`、`group_composer`、
`local_composer`、junction-to-bond 真实 ownership、原有确定性结构输入与
Fit-only 标准化。记

- `c_loc_b ∈ R^96`：现有 local composer 输出；
- `c_glob_b ∈ R^96`：现有 global composer 输出；
- `a_v ∈ R^96`：现有 atom encoder 输出；
- `g ∈ R^33`：现有标准化 global33；

则共享局部网络与全局摘要为

```
h_b = phi(c_loc_b) ∈ R^32          phi: 96 -> 64 (SiLU) -> 32
S   = sum_b h_b ∈ R^32
G   = [ sum_b c_glob_b (96), sum_v a_v (96), sum_v a_v^2 (96), g (33) ] ∈ R^321
```

**A — Additive Reader**

```
y_A = w^T S + psi(G)
w   = Linear(32, 1, bias=False), 可训练、非零初始化
psi = 321 -> 256 (SiLU) -> 256 (SiLU) -> 1
```

`psi` 不接收 `S`、`sum c_loc` 或 `sum c_loc^2`；没有 per-bond ID、per-pattern
lookup、逐结构记忆表；局部贡献跨分子共享参数；没有把局部分支关闭的零初始化门。
`w` 的初值非零（CPU check 3/4 记录范数与梯度）。

**D — Dense Reader**

```
y_D = R([S, G])
R   = 353 -> 256 (SiLU) -> 256 (SiLU) -> 1
```

A/D 的 `phi` 结构、输入状态、`S/G` 张量完全相同；hidden width 由**参数匹配**
预注册为 256，而不是根据性能选择。

### 参数审计（预注册值，代码必须精确复现）

| 模块 | A | D |
|---|---:|---:|
| encoders | 202,048 | 202,048 |
| group_composer | 62,976 | 62,976 |
| local_composer | 31,536 | 31,536 |
| phi (96→64→32) | 8,288 | 8,288 |
| w | 32 | 0 |
| psi | 148,481 | 0 |
| R | 0 | 156,673 |
| **总计** | **453,361** | **461,521** |

`|A−D| / max(A,D) = 1.768% ≤ 2%`，两者 ≤500k。参数差异完全来自 D 的第一层多
接收 32 维 `S`；这是本对照的归纳偏置变量，不是额外数据。

**已知不可完全同构**：A 与 D 的 reader 容量在“输入列 × hidden”上匹配，但
参数分布不同。因此这是一个具体的归纳偏置对照，不能声称隔离了所有容量分布差异。

## 4. 初始化（不得从监督 DCC-L 初始化）

`ar_models.build_pair(norm, seed)`：

1. 用历史同一 `dcc_models.build_initial_state(norm, seed)` 生产 P3 参考与 DCC 的
   共享 trunk：
   - `P3Encoders` 只加载 **train-only masked-SSL** `pretrain_encoder_s{seed}.pt`
     （seed 0 SHA-256 `006fabc1…bfa97`）；
   - `group_composer` 来自该 P3 随机参考；`local_composer` 来自历史
     `seed + 90731` 流；
2. `phi` 在 `seed + 20011` 流初始化一份，A/D 逐张量复制；
3. A 的 `w/psi` 在 `seed + 20021` 流初始化，D 的 `R` 在 `seed + 20022` 流初始化；
4. **不加载任何 `best_monitor.pt` / `final.pt` 监督 checkpoint**。历史 DCC-L
   checkpoint 只用于推理重放（CPU check 1）和读取既有 Locked 预测，绝不进入
   新模型初值。

`build_pair` 在返回前断言：A/D 的 shared trunk + `phi` 逐张量相同；参数 ≤500k；
相对差异 ≤2%；否则抛错。

## 5. 训练配方（与 `dcb_train.py` 一致）

| 项 | 值 |
|---|---|
| 数据 | Fit8k（8000 行） |
| total updates | **47,400**（63 updates/epoch × 753 epochs） |
| batch | 128 |
| optimizer | AdamW，lr 1e-3 → 1e-5 per-step cosine，warmup 500，wd 1e-5 |
| loss | raw-y L1 |
| clip | gradient norm 5，非有限梯度立即失败 |
| dropout | 0 |
| precision | FP32 |
| batch order | `torch.Generator().manual_seed(seed + 101)` + `randperm(8000)`（与历史相同） |
| 标准化 | `fit_only_norm(Fit8k)`；A/D 使用逐数组相同的 norm，hash 记录 |
| 选点 | 每 epoch 全量 Monitor；严格最小 Monitor MAE（`<`，平局保留更早 epoch） |
| checkpoint | 保存 `best_monitor.pt` 与 `final.pt`、逐 epoch 曲线、Monitor 预测 |
| 训练集拟合 | best/final checkpoint 上对全部 8000 Fit8k 行做固定模型 eval-mode MAE |

不启用 EMA、soup、辅助监督、早停、结构统计、attention、递归 MP、外部数据、
teacher 或 ensemble。

## 6. 训练前 CPU gate（10/10，缺一不训练）

`uv run python tracks/structural-sufficiency/zinc_additive_reader_v1/code/ar_checks.py`
必须输出 `all_pass=true` 并写入 `analysis/cpu_checks.json`（绑定同一 config sha）。
检查组：

0. 冻结 split 行 hash、角色、互斥性、Fit4k 嵌套；
1. **历史 DCC-L 生产 forward 重放**：加载冻结 `best_monitor.pt`（仅推理），
   逐张量核对 `a / c_glob / c_loc / z / a_sum / a_sq / g / G`；与冻结 Monitor
   预测重放；确认监督 checkpoint ≠ SSL 初始化；
2. A/D 对相同原始图的 `G` 与初始 `S` 逐位相同；`G` 的 4 段维度 96/96/96/33=321，
   33 维 global 全部进入且只进入一次；
3. A/D 的 shared encoders/composers/phi 逐张量相同；两个 seed 的 shared 初始化
   不同；encoders 与 SSL 文件逐位一致；w/psi/R 非零；
4. 所有参数 requires_grad、反向梯度有限且非零；A 的 `phi → w → loss` 与
   `phi → S → w` 均有非零梯度；无冻结/零门；
5. 原子重编号、无向边反转、单分子 vs batch、batch 顺序均不改变预测（容差 2e-5）；
6. 图间无信息混合；空图、单键、一般环、融合环前向有限；
7. **加性分离**：固定 `G`，只改一个 `h_b` 时 A 的预测差恒等于 `w^T δ`，且与
   其他键 `h` 无关（spread ≤1e-5）；D 的同一干预依赖其他 `h`（spread >1e-4）；
8. 逐图计数/求和与 loop 一致；`S`/`wS`/`psi` 尺度有限；初始化输出非退化；
9. 静态 + 运行时审计：official valid/test 不在任何代码路径；训练/标准化只触
   `ad_train.pt`；Fit/Monitor/Locked 互斥；norm 只来自 8000 Fit 行。

不满足时先修代码；不得先训练再解释。

## 7. Pilot（先只跑 A_s0、D_s0）

第一阶段只读 Monitor，不读 Locked。记
`A0 = A_s0 Monitor MAE`，`D0 = D_s0 Monitor MAE`，
`C0 = 历史 DCC-L s0 Monitor MAE = 0.12315087235253304`。

**停止规则（原文执行）**

1. 若 `A0 > D0 + 0.005` 且 `A0 > C0 + 0.010` → 停止；
2. 若 `A0 > C0 + 0.010` 且 `D0 > C0 + 0.010` → 停止。

任一条触发：不跑 seed 1、不读 Locked、本轮结束，报告负结果。两条都不触发：
跑 A_s1、D_s1。**看到 seed 0 后不修改结构、宽度、激活、分支数、配方或阈值。**

## 8. 四轨迹完成后的冻结与一次性 Locked

1. `ar_analyze.py` 先核对四条 `train.json` 的 `locked_read=false`、
   `official_valid_loaded=false`、`official_test_loaded=false` 与 47,400 updates；
2. 写 `analysis/freeze.json`：A/D 的 `best_monitor.pt` / `final.pt` /
   两个 predictions / train.json / curve.csv 的 SHA-256，历史 DCC-L 对应文件
   hash，config 与 PROTOCOL.md hash，Git commit；
3. freeze 写入后才允许 `one_shot_locked`：对每条 best/final checkpoint 计算一次
   Locked 预测；A 额外保存 `wS`、`psi(G)`、`S` 范数；写
   `analysis/locked_read.json`（`read_count = 1`）。任何重跑只加载已保存数组，
   不再对 checkpoint 做 Locked 前向；
4. `final.pt` 只作辅助，主表一律 best-monitor。

## 9. 分析口径

- 固定 Locked 分组（**支撑量只用 Fit8k 计算**；精确 typed radius-2 rooted signature）：
  - `all_supported`：每个原子环境在 Fit8k 中出现于 ≥5 个分子；
  - `low_support`：无未见原子，但至少一个原子支撑 1–4；
  - `any_unseen`：至少一个原子环境在 Fit8k 中为 0。
- ordinary = `y > −3`；extreme-tail = `y ≤ −3`；主 MAE 始终保留全部分子。
- 配对比较：A vs D、A vs 历史 DCC-L、D vs 历史 DCC-L；逐 seed 与 seed mean；
  报告 ΔMAE（candidate − reference）、逐分子 |AE| 配对差、10,000 次固定分子
  bootstrap 95% 区间、改善分子比例、组内 MAE。
- bootstrap 只描述固定模型上的分子抽样不确定性，不替代跨 seed 复现。
- top-10 敏感性：每对比较按 `max(AE_a, AE_b)` 剔除 10 个分子后重算 ΔMAE。
- A 诊断：`wS` 与 `psi` 的均值/方差/分位、与 y 的相关、`std(wS)/std(psi)`、
  `mean|wS|/(mean|wS|+mean|psi|)`、**把局部分支置零（只保留 psi）后的 Locked MAE**。
- 历史 DCC-L 只读 `runs/fit8k_s{0,1}/predictions_best_monitor.pt` 与 checkpoint
  hash；不重训、不改写。

## 10. 事前研究决策（按优先级）

1. **局部通路塌缩 → `ADDITIVITY_NOT_SUPPORTED`**：两个 A seed 中任一满足
   `std(wS) < 0.05·std(psi)` 且 `|MAE_zero_local − MAE_full| < 0.001`。
2. **`PROMISING_READER_BIAS`**：两个 A seed 的 Locked MAE 都低于对应的 D 与
   历史 DCC-L；A seed-mean 相对 D 与 DCC-L 的收益都 ≥0.003；ordinary 组
   seed-mean 相对两者的收益都 ≥0.002；剔除 top-10 AE 后相对两者的 seed-mean
   gain 仍为正；且 `low_support` 或 `any_unseen` 中至少一组：两个 seed 的
   A-vs-D 与 A-vs-DCCL gain 都 >0，且该组 seed-mean gain 相对两者都 ≥0.003。
   只有此情形才建议下一轮 official 10k 验证。
3. **`ADDITIVITY_NOT_SUPPORTED`（明确更差）**：A seed-mean 比 D 差 ≥0.003，
   或比历史 DCC-L 差 ≥0.003。
4. **`CAPACITY_ONLY`**：非 PROMISING、非上一条；A 与 D 的 seed-mean 都比历史
   DCC-L 低 ≥0.003；且 A-vs-D 不可靠（|A−D| seed-mean <0.003，或并非所有 seed
   的 A−D 都为正）。此时只声称“缩小/重组 reader 的可能正则化作用”，不声称
   局部可加机制成立。
5. **`INCONCLUSIVE`**：其余（依赖 seed、尾部，或全部效应 <0.003）。不加第三个
   seed 救援。

**讨论 0.05x 的必要条件**：只有 A 相对历史 DCC-L 的 Locked 收益达到约
0.005–0.007 且跨两个 seed 稳健时，才允许讨论它是否有望缩小通向 official test
0.05x 的差距；不得把内部 Locked 增益直接换算成 official test 成绩。

## 11. 预算与远程执行

- 硬上限：**2.0 A100 GPU-hours**，含 smoke、正式轨迹、失败作业。
- 先 `res` 计时 smoke（短 horizon、同一代码路径），按实测 epoch 时间投影
  753 epochs；若投影的 4 条正式轨迹 >2.0 h，则停止，**不缩短 47,400 updates**。
- 正式轨迹：`res` 两张 A100 并行，`rr run` + `--pool res-gpu0/1`；
  保留 run ID、commit SHA、部署 marker、实际 GPU model/driver/torch。
- 失败作业也计入 GPU 账目。Phase 1 最多 2 条，pilot 通过后最多再加 2 条。
- 不启动任何原版 DCC-L 训练。

## 12. 交付

`tracks/structural-sufficiency/zinc_additive_reader_v1/`：

- `PROTOCOL.md`（本文件）、`configs/ar_v1.json`
- `code/ar_{common,models,train,checks,analyze}.py`、`code/tests/`
- `analysis/cpu_checks.json`、`freeze.json`、`locked_read.json`、
  `locked_preds/`、`summary.json`、`decision.json`、`tables.md`（若进入正式阶段）
- `REPORT.md`：观察 / 解释 / 决策分开；失败也如实报告，禁止追加 reader 搜索、
  补种子或重训 DCC-L。
