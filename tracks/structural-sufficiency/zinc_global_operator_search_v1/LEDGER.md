# GOS-v1 实验账本

比分单位：官方 ZINC-subset（train 10,000 / valid 1,000），raw-y MAE，单模型，
部署参数 ≤500k，无预测融合 / seed 平均 / checkpoint soup。
`valid` 是开发集（已被历史多轮使用），不是盲测。test 只在冻结后一次性读取。

初始化：所有 run 都从 seed 配对的 train-only masked-SSL 编码器
（`pretrain_encoder_s0.pt` / `..._s1.pt`）加载 atom/bond/junction encoders，
其余模块随机初始化（同一 seed ⇒ 相同初态 sha，见各行 init 前缀）。
配方：`RECIPE B`（AdamW lr 1e-3→1e-5、wd 1e-5、batch 128、500 步 warmup、
per-step cosine、clip 5、raw-y L1、600 epoch = 47,400 updates、全量 valid 每 epoch、
strict best-valid 选点）。除 A-reg 一行外全部同配方。

| # | run | route | seed | commit | params | best-valid | ep | final | GPU·h | 备注 |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---|
| 1 | A_s0 | A | 0 | 19652e0 | 460,017 | 0.075919 | 402 | 0.076529 | 0.833 | 首个主结果 |
| 2 | C0_s0 | C0 | 0 | 19652e0 | 378,993 | 0.080462 | 419 | 0.081799 | 0.528 | 载体对照 |
| 3 | K_s0 | K | 0 | b245e88 | 452,457 | 0.081812 | 329 | 0.083661 | 0.869 | 有符号非竞争核（输） |
| 4 | A_s1 | A | 1 | b245e88 | 460,017 | 0.074577 | 472 | 0.076508 | 0.843 | 第二 seed |
| 5 | C0_s1 | C0 | 1 | b245e88 | 378,993 | 0.080463 | 349 | 0.082159 | 0.527 | 载体第二 seed |
| 6 | Asm_s0 | Asm | 0 | e07da30 | 459,717 | 0.080797 | 316 | 0.082095 | 0.785 | 机制对照：普通归一化 attention（输） |
| 7 | reg_A_s0 | A | 0 | e07da30 | 460,017 | 0.086080 | 372 | 0.112909 | 0.986 | 唯一修正版：wd 1e-3 + dropout 0.05（输，后期发散） |
| 8 | N0_s0 | N0 | 0 | d65d949 | 400,545 | 0.081145 | 478 | 0.081781 | 0.466 | 机制对照：键内容进 reader、无混合（输） |
| 9 | ho_A_s0 | A | 0 | c1008b5 | 460,017 | 0.081299 | 364 | 0.083032 | 0.752 | 留出检验：9,000 行训练 |
| 10 | ho_C0_s0 | C0 | 0 | c1008b5 | 378,993 | 0.085241 | 384 | 0.086642 | 0.469 | 留出检验：9,000 行训练 |
| 11 | A160_s0 | A (reader 160) | 0 | 233826d | 488,817 | 0.080073 | 423 | 0.080833 | 0.836 | 预注册宽度补充：未过门 |
| 12 | A160_s1 | A (reader 160) | 1 | 233826d | 488,817 | 0.081810 | 330 | 0.082745 | 0.939 | 同上 |
| — | gos-timingA | A | 0 | 19652e0 | 460,017 | — | 2 ep | — | 0.003 | 计时探针（5.5 s/epoch） |
| — | gos-test / gos-test2 | A s0/s1 | — | 859828c | — | — | — | — | ~0.10 | 冻结后 reporting-only test（第二次是 import 修复后的 --reread） |

合计 **≈8.9 GPU·h**（12 个 600-epoch run + 探针 + 2 次 test 评估）；上限 16（硬）/14（软）。

失败/浪费：无训练失败。`gos-timingA` 2 epoch 探针 0.003 GPU·h；
`gos-prep`（关系缓存构建，CPU）不计 GPU。

## 参考锚（历史，同一数据/配方/指标）

| 参考 | params | valid（seed 均值/range） | test 均值 | 来源 |
|---|---:|---|---:|---|
| DCC-L (B-R0, 4 seeds) | 494,193 | 0.075990 ± 0.002137 (0.073837–0.078537) | 0.067319 | `zinc_performance_search_v1` |
| DCC-L s0 | 494,193 | 0.078537 | 0.066267 | 同上 |

## 关键单调性检查（写在同一 epoch 上，不用“候选 ep100 vs 参考最终”）

见 `analysis/main_table.md` 的 matched-epochs 表。

## 终端 test 冻结规则（在 test 读取前写定，2026-10-25T18:10Z）

冻结家族 = **route A + 配方 B**，seed 0/1（预注册主配置），checkpoint = 各自 best-valid。
`Areg_s0`（同架构、R2 配方：wd 1e-3 + dropout 0.05，唯一一次有诊断依据的修正：
A 的 online train MAE 在 best epoch 已降到 0.013、训练末期 0.0005，valid 在 ep400 后进入平台）
的处置规则：

- 若 `Areg_s0` valid **≤ 0.0735**（比 A_s0 好 ≥0.0024，即超过本家族 seed 差 0.0013 的 ~2 倍且
  超过参考家族 seed sd），则作为同家族第三 checkpoint 一起冻结（但仍只有 1 个 seed，
  只能称"配方方向的单 seed 迹象"）。
- 否则 `Areg_s0` 只作 valid 诊断，不读 test，家族仍是 A_s0 + A_s1。

`N0`（同载体 + 键内容进 reader、无全局混合）是机制对照，**不是**候选，不进 test。
C0/K/Asm 同样不读 test。冻结文件 `analysis/test_freeze.json` 在第一次读 test 前写入，
之后不允许改 seed / 改 checkpoint / 改配方。

## 训练集内部留出检验（`HOLDOUT_SEED=20261025`，A/C0 在 9000 行上重训，1000 行留出）

留出行只来自 official train（不含 valid/test 标签），本轮任何开发决定都没看过这 1000 行。

| arm | 训练行 | ep | official valid | **留出 1000 MAE** | 训练(kept) MAE | GPU·h |
|---|---:|---:|---:|---:|---:|---:|
| A（route A，配方 B） | 9,000 | 364 | 0.081299 | **0.114666** | 0.015609 | 0.752 |
| C0 | 9,000 | 384 | 0.085241 | **0.124116** | 0.015727 | 0.469 |

配对逐分子差（A − C0）：**−0.009450**，bootstrap 95% CI **[−0.014008, −0.005082]**，A 更优 54.8%。
稳健性：去掉 C0 最差 1 行 −0.008619；去掉 AE 最大 10 行 −0.006267；去掉 5 个极端标签行 −0.007208。
**中位数 AE 几乎相同**（0.043296 vs 0.043118）：增益集中在尾部/大误差分子，而不是中位分子。
（留出 MAE 绝对值偏高是因为这 1000 行里含 y=−42.0 等极端标签分子，official valid 只有 −20.3 的尾部；
两层都用同一批行做配对比较，因此不影响差值结论。）

结论：A 相对同载体 C0 的增益**在新鲜 1000 分子上复现**（+0.0095，CI 不含 0），且**不是尾部单点**。
它没有转化为对 DCC-L 的 test 优势（见 REPORT §4）。

## 预注册（2026-10-25T20:15Z，在启动 A160 之前写定）

动机（只用 valid/历史，不看 A 的 test 数字做设计）：C0（128 宽 reader）valid 0.080462，
而 DCC-L（同 composers、256 宽 reader）4 seed 均值 0.075990；同一 seed 下 s0 是 0.080462 vs 0.078537。
即"压缩 reader"在本轮自己的对照里值 ~0.002–0.0045 valid。A 的算子值 ~0.0052（两 seed + 留出复现）。
因此唯一一个待检验的**加性**假设是：把 reader 放到 500k 部署上限允许的最宽（160 宽，
总参数 488,817），A 的算子增益是否与 reader 容量叠加。

- 家族：`A160` = route A + 配方 B + reader width 160，seed 0/1，其余一切不变（不是网格：
  160 是"能塞进 500k 的最大宽度"这一条规则选出来的）。
- 冻结规则：只有当 A160 两 seed 的 best-valid 均值 **≤ 0.0735**（比 A 家族再好 ≥0.0017，
  即超过 A 家族内部 seed 差 0.0013）并且两个 seed 都好于 A 的同 seed 值时，才冻结 A160 家族并做
  **第二次、reporting-only** 的 test 读取；否则不读 test，报告保留 A 家族结果。
- 这是本轮最后一次新增配置；此后无论结果如何都不再加模型。

## ERRATUM 与两个补救臂（预注册，在启动之前写定）

起因：`MixA` 的 `ken.unsqueeze(-1)` 把 key-energy 项广播到 **query** 轴，行 softmax 精确抵消。
route A 实际训练的是 `softmax(q·k/√d+β(r))` + `g=1+tanh(γ(r))` 门控 + AdaRMSNorm。
详见 `ERRATUM.md`；核实 gate 见 `analysis/errata_checks.json`（17/17 通过，含 fp64 语义、
冻结 checkpoint 的 legacy vs dot 比较、梯度抽查、初态 sha 与历史逐位一致）。

不重跑、不改写任何历史 run / checkpoint / REPORT 数字。

### 两个臂（各一条轨迹，seed 0，route A，配方 B，reader 128，dropout 0）

| 臂 | tag | 算术 | 唯一变化 |
|---|---|---|---|
| L2G | `l2g_A_s0` | `score=(q·k − ½‖k‖²)/√d + β(r)`（`unsqueeze(-2)`），`m=Σ a·g·v` | 打分项（真 L2） |
| GN | `gn_A_s0` | 原 dot 打分，`w=a·g`，`m=Σ (w/Σw)·v` | 门控权重归一化 |

- 数据：official train 内 9,000 行训练（`HOLDOUT_SEED=20261025`，与 `ho_A_s0` 同一划分，
  sha `39295fd7…` 已核对），official valid 1,000 行选点，内部 1,000 行**只在 checkpoint 选定后评估一次**。
- 初态：`build_model(route='A', seed=0)` 重建（ssl-only + 随机 composer/core/reader），
  init sha = `4b8c59f8…36ffd9` = `ho_A_s0` 的初态（已逐位核实）。因此与 `ho_A_s0` 的差别只有开关，
  与 `A_s0` 的差别只有开关 + 训练行数（历史参照用 `ho_A_s0`，不用 `A_s0`）。
- 历史参照：`ho_A_s0` official valid **0.081299**（ep364），内部留出 **0.114666**。
- 内部留出已被查看过一次（上一轮），**不再称新鲜盲测**；SSL 预训练可能见过其无标签结构。

### 停止规则（写在此处，事后不改）

1. NaN / 语义错误（训练与评估走的不是同一个 forward、holdout 行错位、gate 归一化导致 NaN）→ 停下修错，
   不得解释为算法失败。
2. 同进度比较（ep100/200/300/400/500/600 对 `ho_A_s0` 的同 epoch valid）：若在 **ep300 与 ep400 两个窗口**
   都比 `ho_A_s0` 差 **≥0.004**，且 ep400 的差没有比 ep300 收窄（收窄 <0.0005）→ 停止该臂，标明截断。
3. 否则跑满 600 epoch；若预算（合计 2.0 A100·h）在跑满前耗尽 → 按已完成部分报告，不超支。

### 预算

合计上限 **2.0 A100·h**（两卡相加，含故障与评估）。按历史 `ho_A_s0` 的 0.752 GPU·h/600ep 估算，
两臂 ≈1.5 GPU·h，余量 ~0.5 用于可能的提前停止/重跑错误。
GPU0 = L2G，GPU1 = GN（`res-gpu0` / `res-gpu1` 池，一张 A100 各一个）。

### 结果解释的边界（先写清）

- 单 seed、单配置；不因 CI 跨零称等价，不因训练结束后关门变差称“训练收益”。
- 相对差值全部用同一 9,000 行划分的 `ho_A_s0` 作参照，并给 official valid 与 internal holdout 两个口径。
- 分组只作描述：按**旧 A 的固定逐分子 AE** 在 holdout 上分三档（低/中/高），看新臂的差是否集中在高误差档；
  全量 MAE 一律保留，不剔除样本、不据此选 checkpoint。
