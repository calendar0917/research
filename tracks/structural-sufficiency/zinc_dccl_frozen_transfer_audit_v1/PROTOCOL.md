# zinc_dccl_frozen_transfer_audit_v1 — 冻结预注册协议

**Track:** `tracks/structural-sufficiency/zinc_dccl_frozen_transfer_audit_v1/`
**冻结时间（写入本文件时，在任何 probe 训练或 Monitor 评分之前）:** 2026-10-28
**仓库 HEAD（冻结时）:** `a9daf4d64f9945c9e1105b040f2b99f23a5fa81a`
**性质:** 审计 + 便宜的冻结特征 probe。**不训练 DCC-L，不引入任何新分子标签，
不读 official valid / test，不读内部 Locked1k，不启动 HCR。**
**计算:** 仅 CPU（本机）；两个空闲 A100 不使用。

> 本文件在任何 `z` 导出、任何 probe 训练、任何 Monitor 评分之前写成。
> 所有阈值、维度、步数、正则化、reader 形状、head 选择方式与分组定义以本文件
> + `configs/fta_v1.json` 为准。结果出来之后**不得**修改它们；需要修订必须另写
> `AMENDMENT_*.md`，并说明该修订在什么信息之前/之后做出。

---

## 0. 唯一核心问题

**DCC-L 从 Fit4k 增至 Fit8k 的显著泛化改善，有多少可以由“已经训练出来的、
可被相同低容量 reader 利用的表示变化”解释？**

拆成三个可测的子问题（不相互替代）：

- **Q1（可提取性）** 固定形状、固定预算、固定标签的 64 隐层 reader，在
  `z_8k` 上是否比在 `z_4k` 上更准？（`Δ_transfer`）
- **Q2（通道定位）** 改进出现在 atom+global、真实局部 composer 矩+global、
  还是完整 513 维组合上？
- **Q3（机制购买依据）** 若 `Δ_transfer ≈ 0`，则“优先共享低阶贡献、抑制高阶
  残差自由度”（HCR）缺少本轮前提，应关闭该假设；若 `Δ_transfer` 显著且集中在
  某通道，则该通道是下一次配对训练的唯一候选。

**本轮不回答**：reader 因果贡献（无法从冻结表示中识别）、任何 official
valid/test 数字、任何新架构是否更好。

---

## 1. 冻结输入（只读，记录 sha256）

| 对象 | 路径（仓库相对） | sha256 |
|---|---|---|
| train cache（official ZINC subset train 10k，结构 + y） | `tracks/structural-sufficiency/zinc_arch_discovery_v1/cache/ad_train.pt` | `d85d9875878a3cc0df27aeb8ad80e6cddf809e9d8265d77ef423a91bef4e19e3` |
| BJOC 冻结 10k 内部划分 | `tracks/structure-contribution/results/bjoc_predictive_bottleneck_v1/split_a.json` | `53cd2cec29fb658bdbc789845df36b2194a8c8635140d61b4173ce064b48678e` |
| BJOC 冻结 Fit4k 嵌套子集 | `tracks/structure-contribution/results/bjoc_predictive_bottleneck_v1/fit4k_subset.json` | `a9c657a06ceb81de72fa1e6991369ab94b4f648f87e170fc173621d89160f060` |
| Fit4k trunk 轨迹记录（s0 / s1） | `…/zinc_dccl_bottleneck_v1/runs/fit4k_s{0,1}/train.json` | `8867cb95…` (s0) / 见 `analysis/representation_freeze.json` |
| Fit8k trunk 轨迹记录（s0 / s1） | `…/zinc_dccl_bottleneck_v1/runs/fit8k_s{0,1}/train.json` | `2efe2f26…` (s0) / 见 `analysis/representation_freeze.json` |

**冻结 checkpoint（唯一模型侧对象）** = `zinc_dccl_bottleneck_v1` 的
**最终更新 checkpoint**（`step = 47400`，`selection = "final"`），**不是**
Monitor 选择的 `best_monitor.pt`：

| 条件 id | 文件 | sha256 |
|---|---|---|
| `fit4k_s0` | `runs/fit4k_s0/final.pt` | `2be27014ffa8070153cffe69002cbffa6db0e5dd148d9d2549f3f8450c67eeea` |
| `fit4k_s1` | `runs/fit4k_s1/final.pt` | `e7df0e2f6b68eb7a57a20aa448c7c0c2f18b1d9cda1363adf4d6cc31f5c9cb21` |
| `fit8k_s0` | `runs/fit8k_s0/final.pt` | `40baa5f6dced7bf1edbe37af52e89901f7e5c4e018b14713c001ce376391a340` |
| `fit8k_s1` | `runs/fit8k_s1/final.pt` | `cb8aac08e922a0c7962a1c67a7e744664ef4830bf6025ae6b5daab8ba18fb36b` |

四个 checkpoint 都是 `DualChannel(organized_local=True)`（`P3N`），
494,193 参数，`T = 47,400` updates（Fit4k：1482 epochs；Fit8k：753 epochs），
Fit-only 标准化，raw-y L1，batch 128，AdamW 1e-3→1e-5 cosine，warmup 500，
wd 1e-5，clip 5。二者的**唯一**差异是训练行数与标准化统计（均由同一冻结
nested split 决定），记录在各自 `train.json`。

**冻结行集**（行内 hash = 升序 int64 小端字节的 sha256）：

| 行集 | n | sha256 |
|---|---:|---|
| `fit4k` | 4,000 | `3f6b5d69097476b02f5c6fe0cc180e564180fb0da5a241bb5ba1c5b88e7b3ecd` |
| `fit8k` | 8,000 | `1cf6a2ec73ff5dfc182d72a7d4cd0148d31f3ae2488ee9943132f1aa51d7455a` |
| `monitor` | 1,000 | `d493dc7fc547e062ee16301bb42f350d5eb6d5d773215254c0d84098a410933b` |
| `locked` | 1,000 | `9cbefd5e84cf7f4e8f7314280f35f636a0b18753a2f93a724b935e65a208cf6d`（**本轮不加载**） |

`fit4k ⊂ fit8k`；`monitor ∩ fit8k = ∅`；`monitor ∩ fit4k = ∅`；四者并为 10,000。

---

## 2. A 阶段 — 冻结表示导出

### 2.1 导出内容

对四个条件，用**真实 forward**（`model(batch)`，`eval()`，`torch.no_grad()`，
batch 256）在行集 `probe_fit ∪ probe_sel ∪ monitor`（5,000 行）上导出

```text
z = [ sum(c_glob) 96 | sum(a) 96 | sum(a^2) 96 | g 33 | sum(c_loc) 96 | sum(c_loc^2) 96 ]
```

块索引（半开区间，与 `zinc_dcc_information_audit_v1` A1 完全一致）：

| 块 | 索引 | dim |
|---|---|---:|
| `c_glob_sum` | `[0, 96)` | 96 |
| `atom_sum` | `[96, 192)` | 96 |
| `atom_sq` | `[192, 288)` | 96 |
| `global33`（trunk 自身标准化后） | `[288, 321)` | 33 |
| `c_loc_sum` | `[321, 417)` | 96 |
| `c_loc_sq` | `[417, 513)` | 96 |

`g` 是**该 trunk 自己的** `(global_feat − global_mean) / global_scale`；不做任何
跨条件重标定（下游 reader 的标准化会处理尺度，见 §3.2）。

### 2.2 必须通过的导出门禁（任一失败 ⇒ 本轮 `INVALID_RUN`，不得继续评分）

- **E1 reader 输入对齐**：forward hook 捕获的 reader 输入 = 导出 `z`（max|Δ| = 0）。
- **E2 独立重算**：用显式的 `encoders → group_to_bonds → group_composer →
  local_grouping → local_composer → index_add` 路径重算 `z`，与 E1 的
  `z` 逐元素一致（浮点 `max|Δ| ≤ 1e-6`；实现上应观察到 0.0）。
- **E3 预测重放**：把导出 `z` 送入**同一个** checkpoint 的 `reader`，最大预测差
  ≤ 1e-6，且与 `model(batch)` 的逐分子预测一致。
- **E4 批量不变性**：batch 64 与 batch 256 导出的 `z` 一致（`max|Δ| / max|z| ≤ 1e-7`，
  约 0.84 float32 ulp；见 `AMENDMENT_01.md`。该判据在首次执行时、任何 probe/Monitor
  之前由绝对 1e-6 修订，原因是 float32 分块 GEMM 在 `O(10^2)` 量级的 `z` 上产生
  1e-5 量级的绝对舍入差，相对量级仅 ~3e-8）。
- **E5 与冻结轨迹交叉核对**：Monitor 上的原始 MAE 必须等于该 checkpoint
  `train.json` 的 `eval_reads.final.monitor_mae`（差 ≤ 1e-6）。这一条把本轮
  导出钉死在历史冻结产物上。

### 2.3 描述性统计（**不是门禁，不得据此判定信息不足**）

在 `probe_fit` 3,000 行（以及 `monitor` 1,000 行作稳定性对照）上，按块报告：

- 每维方差、块协方差 trace、块内 participation ratio（PR = (Σλ)²/Σλ²）、
  块占全空间 trace 份额；
- 标准化后 `z` 的数值秩：按奇异值相对容差 `1e-2 / 1e-4 / 1e-6 × σ_max`
  定义秩，以及 95% / 99% 平方 Frobenius 能量所需秩、PR；
- 线性 CKA（行中心化）跨 seed（4k↔4k、8k↔8k）、跨规模（4k↔8k，同 seed 与
  交叉 seed）、全 513 与逐块。

**解读纪律**：低 PR / 低秩只说明“有效方向少”，不说明信息不足，也不进入判定。

---

## 3. B 阶段 — 相同标签、相同 reader 的训练

### 3.1 冻结 probe 划分（标签无关）

1. 对 `fit4k` 每一行计算**纯结构指纹**
   `sha256("n_atoms|sorted atom types|sorted (min(u,v),max(u,v),bond_type)")`
   （不读 y、不读任何模型输出）。
2. 指纹相同的行组成一个不可拆分组（`fit4k` 内最大组 2，共 2 组）。
3. 每个组按 `sha256("zinc-dccl-frozen-transfer-audit-v1-probe-split:" + 指纹)` 升序排序。
4. 按该顺序贪心加入 `probe_fit`，**取累计 ≤ 3000 的最大前缀**（整组加入，
   不拆组）；`fit4k \ probe_fit` = `probe_sel`。
5. 冻结结果（**已在本文件写入前用纯结构计算得到**，不含任何标签）：
   `probe_fit` n = 3,000，`probe_sel` n = 1,000，二者互斥且并为 `fit4k`；
   `probe_fit` 行 hash `63f7c2a5643531935fc21b729b6a7bb3aef3618d296682d6bd37d545867c7bfb`，
   `probe_sel` 行 hash `da081fa2b0541971f6cb3a58f0c0becc108c657ed391926f7e7da6a49df8f36a`。
6. 运行时会重算并核对上述 hash；不匹配即拒绝运行。

`probe_fit` 的行**只在**以下用途读 y：训练新 reader。`probe_sel` 的行**只在**
以下用途读 y：每 500 步选择 reader checkpoint。`monitor` 的行**只在**所有 cell
冻结之后统一读一次 y（§4）。三方互斥。

### 3.2 新 reader（每个 cell 完全相同）

```text
Linear(513, 64) → SiLU → Linear(64, 1)
```

- 初始化：PyTorch 默认 `Linear` 初始化，`torch.manual_seed(head_init)`
  （`head_init ∈ {0, 1}`）。**同一 `head_init` 在所有表示条件下产生逐位相同的
  初始权重**（运行时核对 init state hash 相等）。
- 损失：raw-y L1。优化器：AdamW，`lr = 1e-3`（**常数**，无 schedule、无
  warmup），`weight_decay = 1e-5`，默认 betas。梯度裁剪 5.0。
- batch = 128，总 **10,000 updates**（3,000 行 ⇒ 24 updates/epoch，417 epochs）。
- 数据顺序：`torch.Generator().manual_seed(head_init + 1729)`，每个 epoch 重新
  `randperm(probe_fit)`；不同表示条件使用**相同**顺序。
- 特征标准化：**只用同一条件的 `probe_fit` 3,000 行**逐列 `mean/std`
  （`scale = max(std, 1e-6)`），同一变换用于 `probe_fit`、`probe_sel`、`monitor`。
- 每 500 updates（500, 1000, …, 10000）在 `probe_sel` 上算 L1；**best = 最小
  probe_sel MAE（并列取更早的 step）**；同时保存最后一个 update 的 `final`。
  `best` 是主 checkpoint，`final` 只作稳健性报告。
- CPU、FP32、`torch.set_num_threads(1)`，逐 cell 串行（可复现优先）。
- 全程 `requires_grad = False` 于 trunk：本轮没有任何 trunk 权重进入优化器；
  `z` 只来自 §2 的真实 forward。

### 3.3 Cell 清单（主实验 24 格 + 5 个对照）

| # | 条件（trunk） | head_init | 特征块 |
|---|---|---|---:|
| 1–8 | `fit4k_s0`, `fit4k_s1`, `fit8k_s0`, `fit8k_s1` | 0, 1 | `full` (513) |
| 9–16 | 同上 | 0, 1 | `atom_global`（`atom_sum`+`atom_sq`+`global33` = 225 列，其余置零） |
| 17–24 | 同上 | 0, 1 | `local_global`（`global33`+`c_loc_sum`+`c_loc_sq` = 225 列，其余置零） |

分块 cell 使用与 `full` **相同的 head 形状**（513 输入、不用的列置零）、
相同的标签、相同预算、相同初始化（按 `head_init` 对齐）、相同标准化参数
（按列取自 `full` 的 `probe_fit` 统计，再置零非块列）。

对照（**不参与判定，只用于仪器诊断**）：

| # | 对照 | 定义 |
|---|---|---|
| C1–C4 | readout-recovery | trunk ∈ 4 条件，`head_init = 0`，`full`，目标 = **该 trunk 自己的 prediction**（在 `probe_fit` 上），`probe_sel` 上评分 ⇒ 检验 64 隐层头能否从 `z` 复现 trunk 自己的函数 |
| C5 | label-permutation null | `fit8k_s0`，`head_init = 0`，`full`，`probe_fit` 的 y 用固定置换（`seed = 20261028`）打乱，其余不变 |

### 3.4 复现性

指定 cell（`fit8k_s0 / full / init 0`）训练两次，`probe_sel` MAE 差必须
≤ 1e-6（逐位一致）；两次的 final Monitor 预测 `max|Δ| ≤ 1e-5`。

计数口径：训练 cell 总计 **30** 个 = 24 主实验 + 4 recovery + 1 label-null + 1 复现。
其中只有 24 个主实验与 §6 的判定有关。

---

## 4. 一次性 Monitor 评分

所有 cell（含对照）训练结束、且 `probe_sel` 指标与 head state hash 写入
`analysis/probe_freeze.json` 之后，**一次**加载 `monitor` 标签，对每个 cell 的
`best` 与 `final` 在 1,000 个 Monitor 行上算预测与逐分子 AE。

- `monitor` 行不属于任何 trunk 的训练集（已核对 `∩fit4k = ∩fit8k = ∅`）。
- 原始 trunk 的 Monitor 预测在 §2 已导出（`final` checkpoint）；本轮同时报告
  其在 Monitor 上的 `4k→8k` MAE 变化，以及历史冻结的 Locked 数字（仅引用，
  不重算、不加载 Locked）。
- 官方 valid / test 与内部 Locked1k **不加载、不预测、不评分**。

---

## 5. 分组（全部在 Monitor 1,000 行上；定义先于此固定）

支撑定义沿用原冻结口径：对每个原子取**精确 typed radius-2 rooted signature**；
某环境的“支持”= 该 signature 出现在多少个 fit 分子中；
`full` = 全部原子支持 ≥5；`low` = 无 0 支持但存在 1–4 支持；
`unseen` = 至少一个原子 0 支持（分子级 `any-unseen`）。

| 组 id | 定义 |
|---|---|
| `overall` | 全部 1,000 行 |
| `ordinary_nopenalty` | 结构侧人工环惩罚 = 0：`max(nx.cycle_basis) ≤ 6`（含无环），**标签无关** |
| `any_unseen_fit4k` | `any-unseen` w.r.t. **Fit4k** 支撑表，**且** `ordinary_nopenalty` |
| `any_unseen_fit8k` | `any-unseen` w.r.t. **Fit8k** 支撑表，**且** `ordinary_nopenalty` |
| `low_support_fit4k` / `low_support_fit8k` | `low`（无 0 支持、含 1–4 支持） |
| `full_support_fit4k` / `full_support_fit8k` | `full`（全部支持 ≥5） |
| `newly_covered_4k_to_8k` | `any-unseen`(Fit4k) 且 **非** `any-unseen`(Fit8k)：4k→8k 新获得覆盖 |
| `still_unseen_both` | 同时是 `any-unseen`(Fit4k) 与 `any-unseen`(Fit8k) |
| `extreme_tail` | `y ≤ −3`（**仅诊断**，历史口径；不用于选择或门禁以外的解释） |
| `non_tail` | `y > −3` |
| `drop_fixed_top10` | 去掉“**固定影响最大 10 行**”后的 990 行。固定规则（与任何 probe 结果无关）：按 **raw Fit4k trunk（两 seed 平均）** 的 Monitor 绝对误差降序取前 10 行 |

对每个 cell 与每个组报告：n、MAE、组内逐分子 AE 均值；Δ 报告为
`MAE(probe on z_4k) − MAE(probe on z_8k)`（**正值 = 8k 表示更好读**），
分别给出 4 个 cell、两 trunk seed 的 head-init 均值、两 head-init 的 trunk-seed 均值、
四格总均值，以及 Monitor 逐分子配对 bootstrap 95% 区间（5,000 次，描述性）。

---

## 6. 判定规则（预注册，机械执行）

记：

- `Δ_s = mean_{init∈{0,1}} Δ_transfer(fit4k_s − fit8k_s)`，`s ∈ {0,1}`，`best` checkpoint，`overall` 组；
- `Δ_all = (Δ_0 + Δ_1) / 2`；
- `raw_gain = mean_s [ MAE(raw_4k_s on Monitor) − MAE(raw_8k_s on Monitor) ]`（final checkpoint）。

**执行顺序**（先 U，再 TTS，再 ROCS，否则 UNRESOLVED）：

### 6.1 `PROBE_UNINFORMATIVE`（仪器坏掉）

任一成立：

- **U1** 任一 readout-recovery 对照在 `probe_sel` 上的 MAE **> 0.15**
  （目标为 trunk 自身预测；该阈值是“头连 trunk 自己的函数都复现不了”的宽松
  上界，远大于 mechanics smoke 中的 ~0.06）；
- **U2** 任一 `full` 实标签 cell 在 best checkpoint 上的 `probe_fit` 训练 L1
  **> 0.10**（严重欠拟合）；
- **U3** `max_s |Δ_s(init 0) − Δ_s(init 1)| > 0.010`（初始化不稳定性与判定
  门槛同量级）；
- **U4** §3.4 复现性检查不通过。

### 6.2 `TRUNK_TRANSFER_SIGNAL`

全部成立：

- **T1** `Δ_0 ≥ 0.010` 且 `Δ_1 ≥ 0.010`；
- **T2** `any_unseen_fit4k` 与 `any_unseen_fit8k` 两个组上，`Δ_s > 0` 对
  `s = 0, 1` 都成立（“无惩罚未见结构组改善方向一致”）；
- **T3** `non_tail` 组上 `Δ_0 ≥ 0.010` 且 `Δ_1 ≥ 0.010`（“不是极端尾部独占”）。

### 6.3 `READOUT_OR_COADAPTATION_SUSPECTED`

`raw_gain ≥ 0.025`（原始 4k→8k 增益仍然很大）**且**

- `Δ_all < 0.005`，**或** `sign(Δ_0) ≠ sign(Δ_1)`（方向不稳定）。

显式限制：该标签**不是**“证明 reader 是原因”，只表示“固定读者从 8k 表示里
重新提取不到原始增益的量级”。

### 6.4 其他 ⇒ `UNRESOLVED`

（例如 `0.005 ≤ Δ_all` 但未达 T1；或 T1 成立但 T2/T3 不成立。）

**不论判定如何：本轮不自动启动 HCR、不换 reader、不新增监督训练、不读取
official valid/test 或 Locked。**

---

## 7. 报告口径与禁止事项

- 必须报告**全部 4 格 × 3 分块**，不能只报最好的 head；组均值只是汇总，不替代格子。
- `final` checkpoint 的结果必须与 `best` 并列报告（不用于判定）。
- 不得把 `Δ_transfer` 与 `raw_gain` 之差解释为严格的 reader 因果贡献。
- 不得用 PR/秩/CKA 的低值推断“信息不足”。
- 不得用 official valid/test 或 Locked 做任何选择；不得读取 `ad_valid.pt`/`ad_test.pt`。
- 不得在 Monitor 上做任何超参、步数、维度或正则化的搜索；Monitor 只读一次。
- 不得事后修改 `probe_fit`/`probe_sel`、reader 形状、学习率、步数、选择规则或阈值。

---

## 8. 交付

`PROTOCOL.md`、`configs/fta_v1.json`、`code/`、`analysis/representation_freeze.json`
（含逐 checkpoint / 逐 `z` 数组 sha256）、`analysis/representation_stats.json`、
`analysis/probe_freeze.json`、`analysis/probe_monitor_metrics.json`、
`analysis/per_molecule_monitor.csv`、`analysis/fta_decision.json`、
`analysis/cpu_ledger.json`、`analysis/cpu_checks.json`、`REPORT.md`、`DECISION.md`。

`REPORT.md` 最后必须逐条回答：

1. 样本数量改善是否可以被固定读出头从新表示中重新提取？
2. 改善更明显出现在 atom、local-composer 还是完整特征组合？
3. HCR 的“优先共享低阶贡献、抑制高阶残差自由度”是否有更充分的购买依据？
4. 如果没有依据，应该关闭哪个具体假设？
