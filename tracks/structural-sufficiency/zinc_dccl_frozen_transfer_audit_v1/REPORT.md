# DCC-L 冻结表示迁移审计 — `zinc_dccl_frozen_transfer_audit_v1`

**判定：`TRUNK_TRANSFER_SIGNAL`**

**一句话结论：** 在完全相同的 3,000 个标签、完全相同的 33k 参数 reader
（`Linear(513,64)→SiLU→Linear(64,1)`）、完全相同的预算下，**Fit8k 的 513 维
reader 输入比 Fit4k 的同一输入好读 +0.0297 MAE**（Monitor1k，4 格均值，
配对 bootstrap 95% CI `[+0.0226, +0.0390]`）。原始 4k→8k 网络在同一批行的
MAE 增益是 **+0.0387**，因此**约 77% 的样本数量增益可以由一个固定低容量
reader 从新表示中重新提取**（seed 0：64%，seed 1：90%）。它不是“尾部独占”，
也不集中在人工环惩罚分子上：在 full 513 上，4/4 格、两个 trunk seed 以及除
36 行受罚子集（CI 跨零）外的每一个支撑分组，Δ 都同号为正。

**边界：** 官方 valid / test 与内部 Locked1k 全程未加载、未预测、未评分；
未训练任何 DCC-L；未使用任何额外分子标签；**0 GPU 小时**（CPU ≈ 3.6 min，见 §9）。
本轮的 `Δ_transfer` 与原始 `raw_gain` 之差**不能**解释为 reader 的因果贡献
（见 §8）。

---

## 1. 这一轮到底做了什么

对照的是 `zinc_dccl_bottleneck_v1` 的四个 Python 冻结 checkpoint：

| 条件 | 文件 | 训练行 | updates |
|---|---|---|---:|
| `fit4k_s0/s1` | `runs/fit4k_s{0,1}/final.pt` | 4,000 | 47,400（1482 epochs） |
| `fit8k_s0/s1` | `runs/fit8k_s{0,1}/final.pt` | 8,000 | 47,400（753 epochs） |

用 **`final`（固定最终更新）** 而不是 Monitor 选出的 best。四个 checkpoint 的
文件 sha256 与 `zinc_dccl_bottleneck_v1/…/train.json` 一起冻结在
`configs/fta_v1.json` 与 `analysis/representation_freeze.json`。

流程（全部 CPU，单线程 FP32，逐 cell 串行）：

1. **导出**：对每个 trunk 跑真实 forward，在 reader 入口捕获
   `z = [Σc_glob 96 | Σa 96 | Σa² 96 | global33 33 | Σc_loc 96 | Σc_loc² 96]`（513 维），
   行集 = `probe_fit(3000) ++ probe_sel(1000) ++ monitor(1000)`。
2. **划分**：Fit4k 按**纯结构指纹**（不含 y、不含模型输出）哈希切成
   3,000 probe-fit + 1,000 probe-selection；整组不拆，行 hash 冻结。
3. **训练**：24 个主 cell = {4k,8k}×{s0,s1}×{atom+global, local+global, full}×{init 0,1}；
   每格 10,000 updates、batch 128、AdamW lr 1e-3（常数）、wd 1e-5、clip 5、
   raw-y L1，每 500 步在 probe-selection 选点。trunk 全冻结（`requires_grad=False`，
   `z` 只由真实 forward 产生）。
4. **一次性 Monitor 评分**：所有 cell 冻结后，统一在 Monitor1k 读一次标签评分。
5. **判定**：按 `PROTOCOL.md` §6 的机械规则执行（`PROTOCOL.md` 在导出前写完；
   唯一的事前修订是 `AMENDMENT_01.md` 的 E4 数值容差，且发生在任何 probe /
   Monitor 之前）。

---

## 2. A. 冻结表示：对齐与描述性统计

### 2.1 导出门禁（全部通过，见 `analysis/representation_freeze.json`）

| 门 | 含义 | 四个条件的实测 |
|---|---|---|
| E1 | hook 捕获的 reader 输入 = 导出的 `z` | 0.0（按构造） |
| E2 | 用显式 `encoders→group→composer→index_add` 独立重算 `z` | **0.0** |
| E3 | 把导出 `z` 重放回同一 checkpoint 的 reader | **0.0** |
| E4 | batch 64 vs batch 256 的 `z` 一致性 | 2.4e-6…1.14e-5，相对 `max|z|` **1.1e-8…3.5e-8**（≤1e-7，见 `AMENDMENT_01.md`） |
| E5 | 导出预测的 Monitor MAE = 冻结 `train.json` 的 `eval_reads.final.monitor_mae` | 差 ≤ **1e-6**（4/4） |

E5 把本轮导出钉死在历史冻结轨迹上：raw Monitor MAE =
`fit4k 0.162686 / 0.167007`、`fit8k 0.123827 / 0.128514`。

### 2.2 描述性统计（`analysis/representation_stats.json`，**不是门禁**）

probe-fit 3,000 行上的 full-513 谱摘要：

| 条件 | trace(cov) | participation ratio | rank 99% 能量 | rank(1e-6·σmax) |
|---|---:|---:|---:|---:|
| fit4k_s0 | 2935 | 3.51 | 82 | 483 |
| fit4k_s1 | 3214 | 3.36 | 89 | 489 |
| fit8k_s0 | 2411 | 4.71 | 96 | 494 |
| fit8k_s1 | 2681 | 4.09 | 96 | 494 |

块方差份额（`fit4k_s0`，其余条件同量级）：`Σc_loc²` 0.326、`Σc_glob` 0.283、
`Σa` 0.176、`Σa²` 0.129、`Σc_loc` 0.076、`global33` 0.011。
块内 participation ratio 很低（`Σc_glob` 1.78、`Σc_loc` 1.56、`Σc_loc²` 1.20），
**这正是历史 A1 观察到的低秩读出**；按纪律，低秩**不**被解释为信息不足，也
不参与判定。

线性 CKA（probe-fit）：

| 对 | full 513 | `Σa/Σa²` | `Σc_loc/Σc_loc²` | `Σc_glob` |
|---|---:|---:|---:|---:|
| 4k_s0 ↔ 4k_s1（同规模跨 seed） | 0.963 | 0.983 / 0.951 | 0.941 / 0.937 | 0.928 |
| 8k_s0 ↔ 8k_s1（同规模跨 seed） | 0.925 | 0.982 / 0.950 | 0.915 / 0.866 | 0.867 |
| 4k_s0 ↔ 8k_s0（跨规模同 seed） | 0.928 | **0.995 / 0.989** | 0.957 / 0.905 | 0.888 |
| 4k_s1 ↔ 8k_s1（跨规模同 seed） | 0.929 | **0.994 / 0.983** | 0.931 / 0.889 | 0.895 |

**唯一可读的描述性事实**：4k→8k 几乎没改变 atom 边际块（CKA 0.98–0.995），
但明显改变了 ownership 条件化的 composer 块（`Σc_loc²` 0.87–0.91、`Σc_glob`
0.87–0.90）。这与 §4 的通道定位一致，但 CKA 本身不作证据使用。

---

## 3. B. 相同标签、相同 reader：主结果

`Δ_transfer = MAE(probe on z_4k) − MAE(probe on z_8k)`，**正值 = 8k 表示更好读**。
主 checkpoint = probe-selection 上的 best；`final` 只作稳健性。

### 3.1 全部 24 个 cell（`analysis/report_tables.md` §1/§3a）

full 513（12 格中 4 个主格 × 2 个 head init；此处展示 4 个 seed/init 组合）：

| trunk seed | head init | MAE 4k | MAE 8k | Δ |
|---|---:|---:|---:|---:|
| 0 | 0 | 0.164350 | 0.140058 | **+0.024292** |
| 0 | 1 | 0.165418 | 0.140322 | **+0.025096** |
| 1 | 0 | 0.172839 | 0.138695 | **+0.034143** |
| 1 | 1 | 0.177634 | 0.142373 | **+0.035261** |

- **4/4 格同号**，幅度 0.0243–0.0353。
- **head-init 波动**：`|Δ(i0)−Δ(i1)|` = 0.00080（s0）/ 0.00112（s1），
  远小于效应量 → 判定不是某个初始化的运气。
- **trunk-seed 波动**：0.0247 vs 0.0347（差 0.0100）；两个 seed 都过 0.010 门。
- 组均值：`Δ_s0 = +0.0247`、`Δ_s1 = +0.0347`、`Δ_all = +0.0297`。
- **`final` checkpoint** 的 `Δ_all = +0.0256`（同号，幅度小 14%）。

### 3.2 与原始网络的关系（Monitor1k，同一批分子）

| 条件 | probe MAE（两 init 均值） | 原始 trunk MAE（final） | probe − raw |
|---|---:|---:|---:|
| fit4k_s0 | 0.16488 | 0.16269 | +0.00220 |
| fit4k_s1 | 0.17524 | 0.16701 | +0.00823 |
| fit8k_s0 | 0.14019 | 0.12383 | +0.01636 |
| fit8k_s1 | 0.14053 | 0.12851 | +0.01202 |

即：固定 33k reader 在 4k 表示上几乎复现了原网络的水平（+0.002…+0.008），
在 8k 表示上留下 **+0.012…+0.016** 没吃到。原始 4k→8k 增益 **+0.03868**
（Monitor，final；历史 Locked best-Monitor 为 +0.0377/+0.0369），
固定 reader 提取 **+0.02970 → 76.8%**（s0 63.8%，s1 89.7%）。

### 3.3 分块探针（同 head 形状、未用列置零、同标签同预算）

| block | 列数 | MAE 4k | MAE 8k | Δ abs | Δ rel | Δ 的两个 trunk seed |
|---|---:|---:|---:|---:|---:|---|
| `atom_global`（Σa，Σa²，global33） | 225 | 0.3062 | 0.3035 | +0.0026 | +0.85% | **−0.0029 / +0.0081（异号）** |
| `local_global`（global33，Σc_loc，Σc_loc²） | 225 | 0.5584 | 0.5224 | **+0.0360** | +6.46% | +0.0216 / +0.0505 |
| **full**（513） | 513 | 0.1701 | 0.1404 | **+0.0297** | +17.46% | +0.0247 / +0.0347 |

两个分块**列数完全相同（225）**、head 形状/预算/标签完全相同，所以
“atom 通道 vs local composer 通道”是容量匹配的比较。读法必须带三条限制：

1. 三个 block 的**绝对**误差水平差 2–3 倍（0.17 / 0.31 / 0.56），跨 block 的
   绝对 Δ **不能**当作“信息贡献”的可加分解；相对 Δ 也只是一个粗定位。
2. `atom_global` 单独预测能力弱、且 Δ 在两个 trunk seed 上**异号**，因此正确的
   陈述是“在这个固定 reader 下，atom+global 通道几乎不承载 4k→8k 的可提取增益”，
   而不是“atom 通道没有信息”。
3. 分块是定位辅助，**不用于选择最高成绩的模型**；判定只看 full 513。

---

## 4. C. 结构分组与解释（full 513，best，Monitor1k）

支撑定义沿用原冻结口径（精确 typed radius-2 signature 出现在多少个 fit 分子中；
≥5 支持 / 1–4 低支持 / 0 未见）；“无人工环惩罚”用**结构侧**判据
（`max(nx.cycle_basis) ≤ 6`，含无环），与标签无关。

| 组 | n | MAE 4k | MAE 8k | Δ | Δ_s0 | Δ_s1 | 相对 | 配对 bootstrap 95% |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| overall | 1000 | 0.1649 | 0.1402 | **+0.0297** | +0.0247 | +0.0347 | +17.5% | [+0.0226, +0.0390] |
| ordinary（无人工环惩罚） | 964 | 0.1095 | 0.0886 | +0.0263 | +0.0209 | +0.0316 | +22.9% | [+0.0215, +0.0309] |
| 人工环惩罚（`max_ring>6`） | 36 | 1.6475 | 1.5214 | +0.1220 | +0.1261 | +0.1178 | +7.4% | [−0.0055, +0.3258] |
| any-unseen（Fit4k 支撑，无惩罚） | 423 | 0.1518 | 0.1214 | +0.0330 | +0.0304 | +0.0356 | +20.6% | [+0.0238, +0.0420] |
| any-unseen（Fit8k 支撑，无惩罚） | 301 | 0.1698 | 0.1403 | +0.0329 | +0.0295 | +0.0364 | +18.3% | [+0.0213, +0.0444] |
| low-support（Fit4k） | 405 | 0.1660 | 0.1459 | +0.0282 | +0.0201 | +0.0364 | +16.8% | [+0.0152, +0.0493] |
| low-support（Fit8k） | 359 | 0.0946 | 0.0756 | +0.0253 | +0.0190 | +0.0315 | +25.5% | [+0.0179, +0.0326] |
| full-support（Fit4k） | 155 | 0.0625 | 0.0528 | +0.0214 | +0.0098 | +0.0330 | +30.4% | [+0.0129, +0.0316] |
| full-support（Fit8k） | 324 | 0.1736 | 0.1525 | +0.0299 | +0.0211 | +0.0387 | +17.0% | [+0.0148, +0.0545] |
| 4k→8k 新获得覆盖 | 123 | 0.1082 | 0.0755 | +0.0326 | +0.0327 | +0.0326 | +29.4% | [+0.0202, +0.0465] |
| 两个 fit 都仍未见 | 317 | 0.2355 | 0.2007 | +0.0345 | +0.0348 | +0.0342 | +14.2% | [+0.0233, +0.0461] |
| 极端标签 `y ≤ −3`（仅诊断） | 74 | 0.9089 | 0.8407 | +0.0748 | +0.0682 | +0.0814 | +8.2% | [+0.0074, +0.1806] |
| 非极端（`y > −3`） | 926 | 0.1054 | 0.0842 | +0.0261 | +0.0212 | +0.0310 | +23.6% | [+0.0214, +0.0310] |
| 去掉“固定影响最大 10 行” | 990 | 0.1074 | 0.0872 | +0.0260 | +0.0202 | +0.0319 | +22.9% | [+0.0215, +0.0307] |

读法：

1. **不是尾部独占。** 极端尾（74 行，占 4k 时绝对误差质量的 41%）只贡献了
   固定 reader 增益的 **18.6%**（`0.0748×74 / (0.0297×1000)`）；非尾 926 行贡献
   81.4%，且自身就达到 +0.0261。
2. **不是“补上未见环境的查表项”。** “4k→8k 新获得覆盖”（123 行）+0.0326 与
   “两个 fit 都仍未见”（317 行）+0.0345 几乎相同；**已完全支撑**的分子同样
   受益（+0.0214…+0.0299）。这与 `zinc_dccl_bottleneck_v1` 的“增益是与分子在 4k
   时误差成正比的估计方差、而非覆盖补洞”结论一致。
3. **不是人工环惩罚的产物。** 无惩罚分子的增益（+0.0263，相对 +22.9%）比受罚
   子集更稳；受罚 36 行的 CI 跨零（且本仓库已多次记录该标签项依赖环 basis 顺序）。
4. **方向在 4/4 cell 与全部分组上一致为正**（受罚 36 行除外，其 CI 宽）。
5. 去掉固定影响最大 10 行（规则：raw Fit4k 两 seed 的 Monitor 绝对误差降序前 10，
   **与任何 probe 结果无关**）后 Δ 只从 0.0297 降到 0.0260。

---

## 5. D. 判定：逐条对照预注册规则

| 门 | 阈值 | 实测 | 结果 |
|---|---|---|---|
| **U1** readout-recovery（trunk 自身预测，probe-selection L1） | ≤ 0.15 | 0.0479 / 0.0551 / 0.0575 / 0.0609 | **PASS** |
| **U2** best checkpoint 的 probe-fit 训练 L1 | ≤ 0.10 | max 0.0339 | **PASS** |
| **U3** 跨 head-init 的 `|Δ(i0)−Δ(i1)|` | ≤ 0.010 | 0.00112 | **PASS** |
| **U4** 复现 cell（8k_s0/full/init0 重跑） | ≤1e-6 | probe-selection L1 差 **0.0**，best state hash 逐位相同 | **PASS** |
| **T1** `Δ_s ≥ 0.010`（s=0,1） | 0.010 | +0.02469 / +0.03470 | **PASS** |
| **T2** 无惩罚 any-unseen（两套支撑定义）× 两 seed 均 > 0 | >0 | Fit4k 档 (n=423) +0.0304/+0.0356；Fit8k 档 (n=301) +0.0295/+0.0364 | **PASS** |
| **T3** non-tail `Δ_s ≥ 0.010` | 0.010 | +0.02121 / +0.03097 | **PASS** |

⇒ **`TRUNK_TRANSFER_SIGNAL`**。

对照 `READOUT_OR_COADAPTATION_SUSPECTED` 的两个条件：原始增益确实很大
（`raw_gain = +0.0387 ≥ 0.025`），但 `Δ_all = +0.0297` 不小于 0.005，且两个
trunk seed **同号**，因此该标签不成立。

仪器控制（不参与判定，`analysis/report_tables.md` §4）：

- **readout-recovery**：同一个 33k head 在 Monitor 上复现各自 trunk 函数到
  0.0465 / 0.0594 / 0.0623 / 0.0511 —— 说明 64 隐层头的容量足以承载“读一个
  513 维向量”的接口工作；它不是在拟合噪声。
- **label-permutation null**：打乱 4k 标签后，Monitor MAE 1.651，常数中位数
  预测器 1.539 —— 没有泄漏，空头甚至略差于常数（符合预期）。
- **复现**：`probe_selection` L1 与 best state 逐位相同。

---

## 6. E. 四个必答问题

**1）样本数量改善是否可以被固定读出头从新表示中重新提取？**
**可以，而且大部分（≈77%）。** 同一 33k 参数 reader、同一 3,000 个标签下，
`Δ_transfer = +0.0297`（CI `[+0.0226, +0.0390]`），对应原始网络同批分子
`+0.0387` 增益的 76.8%；两个 trunk seed 分别为 63.8% 与 89.7%，4/4 格同号。
所以“4k→8k 的改善主要属于更宽的 256-wide reader / 其 co-adaptation”这一解释
**不能**成为主要解释。剩下约 **23%（≈0.009 MAE）** 在本轮里**无法归因**：
它既可能是 64 隐层头的容量限制，也可能是 8k trunk 的 reader 与表示共同适配出的
函数（本轮设计无法区分，见 §8）。

**2）改善更明显出现在 atom、local-composer 还是完整特征组合？**
**local composer 通道，而不是 atom 通道。** 在容量完全匹配（同为 225 列输入、
同 head 形状与预算）的条件下：

- `local_global`（global33 + Σc_loc + Σc_loc²）：**+0.0360**（相对 +6.5%），
  两个 trunk seed 同号（+0.0216 / +0.0505）；
- `full` 513：**+0.0297**（相对 +17.5%），两个 seed 同号；
- `atom_global`（Σa + Σa² + global33）：**+0.0026**（相对 +0.85%），
  **两个 trunk seed 异号**（−0.0029 / +0.0081）。

这与 CKA 的描述性观察一致（4k↔8k 的 atom 块 CKA 0.98–0.995，几乎没变；
`Σc_loc²` 只有 0.87–0.91）。**注意**：三个 block 的绝对误差水平不同
（0.17 / 0.31 / 0.56），所以这不是可加的贡献分解，也不是“local 通道更好”的
模型选择结论；它只说明**可提取的样本数量增益集中在 ownership 条件化的
真实局部 composer 矩上**。

**3）HCR 的“优先共享低阶贡献、抑制高阶残差自由度”是否有更充分的购买依据？**
**有一部分通道证据，但没有“抑制高阶自由度”的证据；作为本轮的直接购买理由不充分。**

- 支持的部分：标签数量改善确实落在“共享的、由真实 junction→bond 归属计算出的
  局部贡献”通道上（Σc_loc 与 Σc_loc²），而不是 atom 边际；这正是 HCR 想要的
  那种“共享低阶贡献”的载体。
- 不支持的部分：(i) 承载信号的分块**包含二阶矩 Σc_loc²**，本轮没有把
  “一阶 vs 二阶”分开，因此“抑制高阶残差自由度”没有被检验，更没有被支持；
  (ii) 用一个 **33k** 参数的 reader 已经能拿到 77%，HCR 若只是再收缩 reader
  的自由度，其可购买空间被限制在**未被提取的 ~23%（≈0.009 MAE）**之内；
  (iii) 本轮没有证据表明那 23% 是由“高阶自由度”造成的（它同样可以由 64 隐层
  的容量或 8k 的 reader–表示共同适配解释）。
- 因此 HCR 若要开，前提不是本轮结果，而是一个**新的预注册**：在 ≥2 seed 上
  直接比较“共享低阶贡献 + 抑制高阶残差”的 forward 与参数匹配的对照，并冻结
  `any-unseen` 与 non-tail 两个端点。本轮**不启动**。

**4）如果没有依据，应该关闭哪个具体假设？**
关闭的假设是：**“DCC-L 的 Fit4k→Fit8k 增益主要是 reader（256-wide MLP）的
读出/共同适配产物”——作为主要解释关闭。** 证据是固定 33k reader 在 4/4 格、
两个 trunk seed 上重新提取到整体增益的 64–90%，且 14 个分组里除 36 行受罚子集
（CI 跨零）外全部同号为正。
保留但**未授权**的假设只有一个：**“余下 ~23% 由低容量 readout 的容量/优化或
trunk–reader 共同适配造成”**。检验它需要“同表示、不同 reader 容量”的配对实验
（而不是 HCR），且必须新预注册；本轮不买。

---

## 7. 结论对不同口径的稳健性

| 口径 | 结果 |
|---|---|
| 主 checkpoint（probe-selection best） | Δ_all = **+0.0297** |
| `final` checkpoint（第 10,000 update） | Δ_all = **+0.0256**（同号） |
| 只用 head init 0 / 1 | +0.02922 / +0.03018 |
| 只用 seed 0 / 1 | +0.02469 / +0.03470 |
| 去掉固定影响最大 10 行 | +0.02603 |
| 去掉极端标签（`y≤−3`） | +0.02609 |
| 只算无人工环惩罚 | +0.02625 |
| 只算 any-unseen（Fit4k，无惩罚，n=423） | +0.03298 |
| 只算 any-unseen（Fit8k，无惩罚，n=301） | +0.03294 |

没有任何一个口径把符号翻负；最小的口径（seed 0、non-tail）仍为 +0.0212。

---

## 8. 限制与非声明

1. **不是因果分解。** `Δ_transfer` 与 `raw_gain` 的差（≈0.009）**不能**读作
   “reader 的因果贡献”：trunk 与 reader 是一起训练的，冻结表示上的差分只说明
   “可被这个 head 重新提取多少”，不说明“原 reader 贡献多少”。
2. **probe 是下界式的测量。** head 只有 33k 参数、只训练 10,000 updates、
   只用 3,000 标签；它在 4k 表示上就已经比原 trunk 差 0.002–0.008。因此
   `Δ_transfer` 应读作“这个预算下可提取的部分”，而不是“表示里包含的全部”。
3. **3,000 vs 4,000/8,000 标签量不同**：两个 head 用同一 3,000 标签，所以
   head 之间的比较是干净的；但 head 的绝对误差不能与 trunk 的绝对误差直接比。
4. **分块探针的绝对 Δ 不可跨块相加或排序**（误差基线差 2–3 倍）；分块只作定位。
5. **Monitor1k 是诊断集，不是第二个泛化基准。** 它不参与任何 probe 选择
   （选择只在 probe-selection 1,000 行上）；但历史上它被用于 4k/8k trunk 的
   best-checkpoint 选择——本轮使用的是不依赖它的 `final` checkpoint。
6. **`probe_fit/probe_sel` 是 Fit4k 的哈希划分**，与 `Fit8k` 无关；
   因此两个表示的 head 训练/选择行完全一致，这一点是本轮最关键的控制。
7. **描述性统计不进入判定**：PR 1.2–4.7、数值秩 483–494、CKA 0.87–0.995 都只是
   描述；低 PR 不被解释为信息不足。
8. 未开启：HCR、任何新架构、任何 GPU 训练、official valid/test、Locked1k。

---

## 9. 计算账本（0 GPU）

| 阶段 | 墙钟 |
|---|---:|
| 导出 4 个 checkpoint 的 `z`（5,000 行 ×4） | 5.1 s |
| 描述性统计（块方差 / PR / 秩 / CKA） | 2.4 s |
| 30 个 probe cell（单线程、串行；24 主格 158.4 s + 6 控制格 40.2 s） | 198.5 s |
| 一次性 Monitor 评分（60 个 best/final 头） | 2.8 s |
| 分析（分组 + 判定） | 0.5 s |
| CPU gate（91 项） | 7.6 s |
| **合计** | **217.0 s（0.060 CPU·h）** |

GPU：**0 小时**；两个空闲 A100 全部保留。峰值 RSS ≈ 2.8 GB（train cache + 4 份 `z`）。
明细见 `analysis/cpu_ledger.json`、`runs/ledger.jsonl`。

---

## 10. 产物与复现

```bash
cd tracks/structural-sufficiency/zinc_dccl_frozen_transfer_audit_v1/code
uv run python fta_export.py     # 真实 forward 导出 z + E1–E4
uv run python fta_stats.py      # 描述性统计
uv run python fta_probe.py      # 30 个 head cell（含 recovery / null / repro）
uv run python fta_score.py      # 一次性 Monitor 评分（E5 在此核对）
uv run python fta_analyze.py    # 分组 + 判定
uv run python fta_checks.py     # CPU gates（91/91）
uv run python fta_ledger.py     # 计算账本
uv run pytest -q test_fta.py    # 10 个快速单测
```

| 产物 | 内容 |
|---|---|
| `PROTOCOL.md` / `configs/fta_v1.json` | 冻结协议与机器可读常量 |
| `AMENDMENT_01.md` | 唯一的（数值容差）事前修订 |
| `analysis/representation_freeze.json` | checkpoint/z 的 sha256、E1–E5、行 hash |
| `analysis/representation_stats.json` / `.md` | 块方差、PR、数值秩、CKA |
| `AMENDMENT_02.md` | 评分后的事后代码修正：`any_unseen` 分组按协议与 `ordinary` 相交（阈值/设计未变） |
| `analysis/probe_freeze.json` + `analysis/probe_cells/*.json` | 30 个 cell 的完整训练记录（曲线、选点、init hash） |
| `analysis/probe_monitor_metrics.json` | 一次性 Monitor 评分（best + final） |
| `analysis/per_molecule_monitor.csv` | 逐分子：y、环/支撑特征、4 个 raw AE、30 个 best-head 预测与 AE |
| `analysis/probe_monitor_predictions.npz` | 1,000×66 冻结预测矩阵（30 cell × best/final + 4 raw + y + 行号） |
| `analysis/fta_decision.json` / `report_tables.md` | 机械判定与全部表格 |
| `analysis/cpu_checks.json` | 91 项 CPU gate |
| `analysis/cpu_ledger.json` | 计算账本 |
| `runs/z/*.npz`、`runs/heads/*.pt` | 冻结 `z` 与 head 状态（git-ignored，sha256 已入 `analysis/`） |
