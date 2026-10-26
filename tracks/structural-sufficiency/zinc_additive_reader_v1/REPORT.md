# REPORT — `zinc_additive_reader_v1`

**类型：** 受控机制可行性实验（不是架构搜索，不声称原创性）
**状态：** 四条正式轨迹完成；best-Monitor checkpoint 冻结后一次性 Locked 推理；预注册判定 **`INCONCLUSIVE`**
**协议：** `PROTOCOL.md`（SHA-256 `4f413b587e16788be5a1c288ae94a5114716b0ce5c9d5bba34386b9c131d8e0b`）
**配置：** `configs/ar_v1.json`（SHA-256 `a436d9beef111214312d24750c53320fea63e3133d99e7323b2d28c66aeadd9b`）

---

## 摘要

本轮只检验一个假设：DCC-L 的有限标签泛化瓶颈，是否部分来自“把所有结构状态聚合成
宽向量后交给高自由度整图 MLP”；如果把预测约束成“共享局部贡献之和 + 小型全局修正”，
是否能改善未见结构上的 Locked 泛化。

在参数量匹配（A 453,361；D 461,521；相差 1.768%）、相同 trunk、相同输入、相同
Fit8k 配方、相同 47,400 updates、相同 Monitor 选点的条件下：

- **A 比 D 略好，但幅度低于预注册的 0.003 门槛。** Locked best 的 seed-mean：
  A `0.084543`，D `0.087184`；A−D `−0.002641`（负值表示 A 更好），95% bootstrap
  CI `[−0.005912, +0.000747]`；两个 seed 方向一致（s0 `−0.003438`，s1 `−0.001844`）。
- **A 没有超过未重训的历史 DCC-L。** A−C `+0.001045`（正值表示 A 更差），两个 seed
  都略差（s0 `+0.000820`，s1 `+0.001270`）；C 的 seed-mean 为 `0.083498`。
- **目标组 `any_unseen` 没有收益，反而两 seed 都差于 C。** A−C 在 unseen 组
  `+0.006440`（A 更差；s0 `+0.006499`，s1 `+0.006381`）；A−D 在该组几乎为零
  （`−0.000563`）。`low_support` 上 A 比 D 好 `0.004363`（CI 不含 0），但比 C 只
  好 `0.002152`（CI 含 0，且仅 s0 为正）。
- **局部分支确实在工作，不是“没学会/被忽略”。** A 的 `wS` 标准差为 `0.851/0.779`，
  `psi` 为 `1.131/1.225`；局部项占平均绝对贡献 `43.3%/44.9%`；把 `wS` 置零后
  Locked MAE 从 `0.0844/0.0847` 恶化到 `0.6906/0.7822`。CPU 干预检查也确认 A 的
  局部扰动严格等于 `wᵀδ`、与其他键无关；D 则与其他 `h` 交互。
- **结论：** 预注册类别为 **`INCONCLUSIVE`**。具体到“加性 reader 能改善未见结构
  泛化”这一核心方向，证据是负面的；本轮不足以进入 official 10k，也不支持继续在
  Locked 上搜索 reader。A 相对 D 的小幅优势说明 reader 的聚合方式**有影响**，但
  影响太小、且没有出现在目标组上，不能支撑“这是主要瓶颈”的强主张。

---

## 1. 协议执行与审计

### 1.1 固定数据与角色

复用 `tracks/structure-contribution/results/bjoc_predictive_bottleneck_v1/split_a.json`
的冻结 10k official-train 内划分，未创建新 split：

| 角色 | 行数 | 行号 SHA-256（int64 LE） |
|---|---:|---|
| Fit8k | 8,000 | `1cf6a2ec73ff5dfc182d72a7d4cd0148d31f3ae2488ee9943132f1aa51d7455a` |
| Monitor | 1,000 | `d493dc7fc547e062ee16301bb42f350d5eb6d5d773215254c0d84098a410933b` |
| Locked | 1,000 | `9cbefd5e84cf7f4e8f7314280f35f636a0b18753a2f93a724b935e65a208cf6d` |

三者互不相交、并为 10,000；Fit4k（4,000）嵌套于 Fit8k。所有运行重新计算并核对
行 hash；不匹配即拒绝运行。标准化只使用 Fit8k，`norm_hash` 四条轨迹完全一致：
`bbfb4ce733b7433cfe18fa922b8b990a37ba17aac103ddf5a51fe0110f4db207`。

Locked 支撑分组只用 Fit8k 的精确 typed radius-2 rooted signature 计算：

| 组 | 定义 | Locked 分子数 |
|---|---|---:|
| `all_supported` | 每个原子环境在 Fit8k 中出现于 ≥5 个分子 | 312 |
| `low_support` | 无未见原子，但至少一个原子支撑 1–4 | 401 |
| `any_unseen` | 至少一个原子环境在 Fit8k 中出现 0 次 | 287 |

Fit8k 共有 11,015 个唯一环境；Locked 标签均值 `0.05670`、标准差 `1.90959`，
`y ≤ −3` 的极端尾 68 个分子。分组互斥且覆盖全部 1,000 行。

### 1.2 模型与参数

两臂保留 DCC-L 的全部局部结构计算（`P3Encoders`、`group_composer`、
`local_composer`、真实 junction-to-bond ownership、原有确定性结构输入与
Fit-only 标准化）。共享局部网络与全局摘要为

```
h_b = phi(c_loc_b) in R^32,   phi: 96 -> 64 (SiLU) -> 32
S   = sum_b h_b
G   = [ sum_b c_glob_b (96), sum_v a_v (96), sum_v a_v^2 (96), g (33) ] in R^321
```

| 模块 | A（Additive） | D（Dense） |
|---|---:|---:|
| encoders | 202,048 | 202,048 |
| group_composer | 62,976 | 62,976 |
| local_composer | 31,536 | 31,536 |
| `phi` (96→64→32) | 8,288 | 8,288 |
| `w` (32→1, no bias) | 32 | 0 |
| `psi` (321→256→256→1) | 148,481 | 0 |
| `R` (353→256→256→1) | 0 | 156,673 |
| **总计** | **453,361** | **461,521** |

`|A−D| / max(A,D) = 1.768% ≤ 2%`，两者均 ≤500k。A 的 `psi` 不接收 `S`、
`Σc_loc` 或 `Σc_loc²`；没有 per-bond/pattern lookup；没有零初始化门；`w` 非零
初始化。A/D 的 `phi`、trunk、输入 `S/G` 张量逐位相同；初始化只来自历史
train-only masked-SSL encoder 与历史 composer 随机流，**没有加载任何监督 DCC-L
checkpoint 作为初值**。

需要明确的两个容量事实：

1. A 与 D 是彼此匹配的（相差 1.768%），所以 **A vs D 是容量匹配的 reader 归纳
   偏置对照**。
2. A、D 的 reader 都比 DCC-L 小：A reader 148,513（`w`+`psi`），D reader
   156,673，C reader 197,633；A 总参数比 C 少 40,832（8.26%），D 少 32,672
   （6.61%）。因此 **A vs C 仍混有 reader 容量差异**，不能把 A−C 全部归因于
   加性约束；这是本轮预注册时就接受的限制。

### 1.3 训练配方与预算

四条正式轨迹与 `dcb_train.py` 一致：Fit8k、47,400 updates（63 updates/epoch ×
753 epochs）、batch 128、AdamW 1e-3→1e-5 per-step cosine、warmup 500、weight
decay 1e-5、raw-y L1、gradient clip 5、dropout 0、FP32、每 epoch 全量 Monitor、
严格最小 Monitor 选点（平局保留更早 epoch）、batch order `seed+101`。未使用
EMA、soup、辅助监督、早停、attention、递归 MP、外部数据、teacher 或 ensemble。

| run | 参数 | Monitor best | Locked best | Locked final | train@best | best epoch | GPU-h |
|---|---:|---:|---:|---:|---:|---:|---:|
| A_s0 | 453,361 | 0.126245 | 0.084394 | 0.084551 | 0.006971 | 560 | 0.3610 |
| A_s1 | 453,361 | 0.124696 | 0.084693 | 0.084542 | 0.004564 | 608 | 0.3481 |
| D_s0 | 461,521 | 0.125478 | 0.087832 | 0.085457 | 0.019176 | 392 | 0.4008 |
| D_s1 | 461,521 | 0.128676 | 0.086537 | 0.086783 | 0.011971 | 495 | 0.4199 |
| C_s0（历史，不重训） | 494,193 | 0.123151 | 0.083573 | — | — | 437 | 0.3685 |
| C_s1（历史，不重训） | 494,193 | 0.127772 | 0.083423 | — | — | 552 | 0.3691 |

GPU 账本：正式轨迹合计 **1.5299 A100-hours**；四次 smoke 合计 **0.0076**；总计
**1.5375 < 2.0**。Smoke 先在 `res` 上测时（A 315 updates 投影 0.4445 h/run，
D 315 updates 投影 0.4433 h/run），正式轨迹实际为 0.348–0.420 h/run，峰值显存
约 102–105 MiB。主机 `a100-2`，2×A100-SXM4-40GB，driver 550.163.01，
torch 2.5.1+cu124，CUDA 12.4。完整逐 run 账本与曲线见
`analysis/training_ledger.json`；预算审计见 `analysis/protocol_consistency.json`。

### 1.4 训练前 CPU gate

正式训练前 10/10 PASS；在最终代码上又复跑一次 10/10 PASS（`analysis/cpu_checks.json`，
SHA-256 `8effe6afc7fa4dd1fe65964ca26e0f6e2f0da65df4782174e51cffdc6565423d`）。
关键项：

- 历史 DCC-L 生产 forward 重放：`a/c_glob/c_loc/z/a_sum/a_sq/g/G` 逐张量核对；
  Monitor 重放 MAE `0.1231508676847443`；最大数值差 `3.58e-7`（组件差 0.0）。
- A/D 对同一原始图的 `G` 与初始 `S` 逐位相同；`G` 四段维度 96/96/96/33=321。
- A/D 共享 encoder/composer/`phi` 逐张量相同；两个 seed 的共享初始化不同；
  encoder 与 SSL 文件逐位一致；`w/psi/R` 非零。
- 所有参数有梯度；`phi→w→loss` 与 `phi→S→w` 梯度非零；无冻结/零门。
- 原子重编号、无向边反转、单分子 vs batch、batch 顺序：预测差 ≤2e-5。
- 图间无信息混合；空图、单键、普通环、融合环前向有限。
- **加性分离：** 固定 `G`、只改一个 `h_b` 时，A 的 `Δy` 恒等于 `wᵀδ`，与其他
  `h` 无关（spread `7.15e-7`）；D 的同一干预依赖其他 `h`（spread `9.89e-4`）。
- 静态 + 运行时审计：official valid/test 不在任何代码路径；训练/标准化只触
  Fit8k；Fit/Monitor/Locked 互斥。
- 四条 `train.json` 均记录 `locked_read=false`、`official_valid_loaded=false`、
  `official_test_loaded=false`；47,400 updates、753 epochs、同一 norm hash。

### 1.5 一次性 Locked 读取与执行偏差

四条正式轨迹完成后，`ar_analyze.py` 先写 `analysis/freeze.json`（checkpoint、
预测、train.json、curve、config、PROTOCOL、Git commit 的 SHA-256），再对每个
best/final checkpoint 做一次 Locked 前向；A 额外保存 `wS`、`psi(G)`、`S` 范数。
`analysis/locked_read.json` 记录 `read_count=1`、1000 行、行 hash、8 个预测文件
及其 SHA-256。

必须诚实记录的执行偏差（不改变任何模型、配方、阈值或判定）：

1. 第一次 `ar_analyze.py` 在写完 `A_s0_best_monitor.npz` 后，因 `final.pt` 没有
   `monitor_mae` 字段而崩溃；第二次复用该 npz（没有再次做 Locked 前向），计算其余
   7 个数组，随后因历史 MAE 一致性阈值 `1e-9` 过严（重算与存储值差约 `5e-9`）
   而崩溃；第三次因 `locked_ae_best` 别名缺失而崩溃；第四次成功生成
   summary/decision/tables；之后又修正了 top-10 敏感性符号的**决策输入**符号错误，
   重跑一次得到最终表格。**8 个 checkpoint 的 Locked 前向各只做了一次**，没有
   checkpoint 被第二次前向；没有因为看到 Locked 结果而修改模型、训练、阈值或
   决策规则。
2. 上面的分析重跑会重新读取冻结的 Locked 标签来计算指标（不是重新推理）。这是
   工程 bug 修复，不是对 Locked 的反复“试探”；所有判定规则在训练前已冻结，
   且最终 verdict 在第一次完整读取时已经是 `INCONCLUSIVE`，后续修复没有改变它。
3. 正式部署前，远端有一个无关审计 track 的未跟踪 analysis 文件导致 `rr deploy`
   拒绝；在远端把这些文件移到备份目录后部署成功。没有改科学代码或数据。
4. 第一次 D smoke 在 GPU1 上投影偏高（0.581 h/run），用 315-update smoke 复核为
   0.443 h/run；正式 D 实际 0.401/0.420 h/run，预算仍低于 2.0 h。

**没有**构造、读取或评估 official test；**没有**加载 official valid；**没有**
用 Locked 做 checkpoint 选择；**没有**重训历史 DCC-L。历史 C 条目里的
`official_valid_loaded=true` 只描述旧轮次曾描述性读取 official valid，本轮没有
任何 official valid 路径。

---

## 2. 观察

### 2.1 主结果：A 略好于 D，但没有超过 C

Locked best-Monitor 的 seed-mean：

```
A_mean = 0.084543
D_mean = 0.087184
C_mean = 0.083498      (历史 DCC-L，未重训)

A - D  = -0.002641     (A 更好)
A - C  = +0.001045     (A 更差)
D - C  = +0.003686     (D 更差)
```

逐 seed（Δ = candidate − reference；负值 = candidate 更好）：

| comparison | seed | ΔMAE | 95% bootstrap CI | 改善分子比例 |
|---|---:|---:|---|---:|
| A vs D | 0 | −0.003438 | [−0.008099, +0.001869] | 0.542 |
| A vs D | 1 | −0.001844 | [−0.005960, +0.002570] | 0.525 |
| A vs D | seed-mean | −0.002641 | [−0.005912, +0.000747] | 0.560 |
| A vs C | 0 | +0.000820 | [−0.004055, +0.006492] | 0.520 |
| A vs C | 1 | +0.001270 | [−0.002664, +0.005470] | 0.486 |
| A vs C | seed-mean | +0.001045 | [−0.002503, +0.004804] | 0.524 |
| D vs C | 0 | +0.004258 | [+0.000400, +0.008159] | 0.492 |
| D vs C | 1 | +0.003114 | [−0.000267, +0.006521] | 0.473 |
| D vs C | seed-mean | +0.003686 | [+0.001075, +0.006392] | 0.475 |

要点：

- A−D 两个 seed 都是负的（A 更好），但 seed-mean 只有 `0.00264`，低于预注册的
  `0.003` 材料性门槛，CI 含 0；这是一个**方向一致但幅度不足**的弱信号。
- A−C 两个 seed 都是正的（A 更差），幅度约 `0.001`；A 没有超过历史 DCC-L。
- D−C 的 seed-mean CI 不含 0（D 更差约 `0.0037`），但逐 seed CI 有一个含 0；
  D 是容量匹配的 dense 对照，不是“DCC-L 的等容量复现”。

### 2.2 分组：低支撑有局部信号，未见结构没有收益，尾部分歧

seed-mean 分组（Δ = A − reference；负值 = A 更好）：

| 组 | n | A vs D Δ [95% CI] | A vs C Δ [95% CI] |
|---|---:|---:|---:|
| all-supported | 312 | −0.002339 [−0.007114, +0.004060] | +0.000192 [−0.004622, +0.006521] |
| low-support | 401 | **−0.004363 [−0.008307, −0.000294]** | −0.002152 [−0.006821, +0.003134] |
| any-unseen | 287 | −0.000563 [−0.008653, +0.007850] | **+0.006440 [−0.002304, +0.015674]** |
| ordinary (y>−3) | 932 | −0.002334 [−0.005471, +0.000945] | −0.000038 [−0.003351, +0.003438] |
| tail (y≤−3) | 68 | −0.006852 [−0.026658, +0.013991] | **+0.015884 [−0.007869, +0.044763]** |

逐 seed 的 `low_support` A−C：s0 `−0.004644`，s1 `+0.000340`；`any_unseen` A−C：
s0 `+0.006499`，s1 `+0.006381`。也就是说：

- **low-support：** A 比 D 好 `0.00436`，seed-mean CI 不含 0；但 A 比 C 只
  好 `0.00215`，CI 含 0，且只有 s0 为正。可以谨慎说“加性 reader 在稀有但已见
  环境上比同容量 dense reader 略稳”，不能说“超过 DCC-L”。
- **any-unseen：** A 与 D 基本打平（`−0.00056`），但 A 比 C 差 `0.00644`，两个
  seed 都是同方向。**目标组没有出现假设预期的改善。**
- **tail：** A 比 C 差 `0.01588`，A 比 D 好 `0.00685`，但 A−D 的逐 seed 符号
  相反（s0 `−0.0247`，s1 `+0.0110`）。尾部只有 68 个分子，方差很大，不应单独
  解释。
- **ordinary：** A 与 C 基本打平（`−0.00004`），A 比 D 好 `0.00233`。A 相对 D
  的优势不是“只靠尾部”；但 A 相对 C 的劣势主要集中在尾部/少数大误差分子。

### 2.3 top-10 敏感性与“少数极端分子”

按每对比较的 `max(AE_a, AE_b)` 剔除 10 个分子后：

| comparison | 原始 ΔMAE | 剔除 top-10 后 ΔMAE | 解释 |
|---|---:|---:|---|
| A vs D | −0.002641 | **−0.004315** | A 的优势在典型分子上更大，不是尾部驱动 |
| A vs C | +0.001045 | **−0.001657** | A 的总体劣势集中在少数最大误差分子；剔除后 A 反而略好 |
| D vs C | +0.003686 | **+0.002870** | D 更差是较广泛的，不只是极少数分子 |

这组结果很重要：A 相对 C 的总体差距不是“全面变差”，而是**尾部/极端分子上的
表达力不足**；A 相对 D 的优势则在典型分子上仍然存在。两者都没有达到 0.003 的
材料性门槛（A vs C 剔除后也只有 `0.00166`）。

### 2.4 A 的局部分支诊断

| run | std(wS) | std(psi) | corr(wS,y) | corr(psi,y) | 局部贡献占比 | MAE full | MAE zero-local | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A_s0 | 0.8512 | 1.1307 | +0.9165 | +0.9463 | 0.4334 | 0.084394 | 0.690613 | +0.606220 |
| A_s1 | 0.7791 | 1.2247 | +0.8898 | +0.9488 | 0.4489 | 0.084693 | 0.782199 | +0.697506 |

`wS` 与 `psi` 的 Locked 分位（best checkpoint）：

- A_s0：`wS` q05/q50/q95 = `−1.480/0.254/1.153`；`psi` q05/q50/q95 =
  `−2.121/0.198/1.530`。
- A_s1：`wS` q05/q50/q95 = `−0.892/0.596/1.533`；`psi` q05/q50/q95 =
  `−2.809/−0.175/1.169`。

局部项不是装饰：它贡献了约 43–45% 的平均绝对预测幅度，与 y 的相关约 0.89–0.92；
把它置零会让 Locked MAE 恶化约 0.61–0.70。`std(wS)/std(psi) = 0.753/0.636`，
远高于预注册的塌缩阈值 0.05；`delta_zero_local` 也远大于 0.001。因此
**没有任何 A seed 出现局部通路塌缩**。这也说明：本轮的负结果不是“A 没学会用
局部项”，而是“用了局部项也不足以超过 DCC-L，尤其不足以改善未见结构”。

`wS` 与 `psi` 的相关为 0.78/0.74，这是两个组件都在拟合同一标签的自然结果，不是
加性结构被破坏；CPU 干预检查已经严格验证了 A 的局部扰动与其他 `h` 无关。

### 2.5 训练曲线与拟合

- A 的 best-Monitor 出现在 ep 560/608，D 在 ep 392/495；A 在 best 时的 Fit8k
  拟合 MAE 为 `0.00697/0.00456`，D 为 `0.0192/0.0120`。A 的 best 更晚、训练集
  拟合更好。
- 到 final 时两者都把 Fit8k 拟合到约 `4.7–5.3e-4`，说明两个 reader 的容量都足以
  拟合训练集；差异不在“能不能拟合”，而在泛化/选点。
- final checkpoint 的 Locked（辅助，不用于主判定）：A `0.084551/0.084542`，
  D `0.085457/0.086783`；A 仍比 D 好，但 seed-mean 只有约 `0.00157`，小于
  best 的 `0.00264`。D_s0 的 final 比它的 best 好约 `0.0024`，说明 best-Monitor
  选点对 D_s0 有一定噪声；这不改变主表按协议使用 best 的结论。

---

## 3. 解释

### 3.1 A vs D：加性约束确实改变了 reader，但只是小幅正则化

A 与 D 共享全部 trunk、输入、`S/G` 和参数量级；唯一区别是 reader 的归纳偏置
（A：`wᵀS + ψ(G)`；D：`R([S,G])`）。A 在两个 seed 的 Locked 上都更好，且
low-support 的 seed-mean CI 不含 0，top-10 剔除后优势更大。这支持一个弱结论：

> 在**相同容量**下，把局部状态先做成共享的 32 维 sum、再用一个低维线性头读取，
> 比让 dense MLP 自己从 `[S,G]` 里学聚合，略更抗有限标签噪声；这个效应在典型
> 分子和低支撑分子上可见。

但它只有 `0.00264`，低于预注册的 0.003，CI 含 0，且 `any_unseen` 上 A−D 基本
为零。因此不能把它升级为“reader 聚合是主要瓶颈”。更准确的表述是：
**reader 的聚合方式是一个小尺度的正则化因素，不是本轮可检测的主要瓶颈。**

D 的 final 训练拟合与 A 一样好，说明 D 并非容量不足；D 的 best 更早、训练拟合
更差，提示 dense reader 在 Monitor 上的选点更噪、或者更早进入某种过拟合/欠拟合
折中。A 的 best 更晚、训练拟合更好但 Locked 略好，符合“加性 sum 降低了局部
组合的方差”的解释，但幅度有限。

### 3.2 A vs C：没有超过 DCC-L，且差异集中在尾部/极端分子

A 比 C 差约 `0.00105`，两个 seed 同方向但幅度很小；ordinary 组基本打平
（`−0.00004`），tail 组差 `0.01588`，top-10 剔除后 A 反而略好 `0.00166`。
这组模式说明：

- A 在“普通分子”上已经与 DCC-L 相当；加性 reader 没有破坏一般的局部化学信号。
- A 的差距主要在少数极端分子：这些分子可能需要跨局部状态的**上下文相关修正**
  （例如一个局部环境的意义依赖其他局部环境/整体构象），而 `wᵀS + ψ(G)` 的局部项
  被严格禁止与其他 `h` 交互，`ψ(G)` 又只看到全局求和，无法做逐局部的上下文调制。
- CPU 加性分离检查正好量化了这一表达力差异：A 的局部干预严格等于 `wᵀδ`，D 的
  同一干预会随其他 `h` 改变。这不是实现 bug，而是设计使然。

因此，**“把 reader 约束成局部可加 + 小全局修正”在本参数化下没有带来未见结构
收益；它把模型从 D 的尾部表达力拉向 A 的可加结构，但拉回的收益不足以补偿失去
的交互表达力。** 这是一个机制上可解释的负结果，而不是单纯的 seed 噪声。

需要保留的替代解释：A 和 D 的 reader 都比 C 小（A 少 8.26%，D 少 6.61%），
A−C 的差距可能有一部分是容量，而不是加性约束。A vs D 是容量匹配的，所以
“A 比 D 好”可以较干净地归因于归纳偏置；但“A 比 C 差”不能完全归因于加性约束。
本轮的设计只能做到这个程度；若要分离容量，需要另一个预注册的、与 C reader 等
容量的 dense 对照，而不是本轮能回答的问题。

### 3.3 low-support / any-unseen / tail：目标组没有兑现

假设的核心是“未见结构泛化”。实际分组结果是：

- `low_support`（已见但稀有）：A 比 D 好，说明共享局部 sum 对稀有环境的方差
  控制有一点帮助；但相对 C 的优势只有 `0.00215`，且只有 s0 为正，不能称为稳健。
- `any_unseen`（真正未见环境）：A 与 D 打平，A 比 C 差 `0.00644`，两个 seed
  一致。**这是对 H_ADD 最直接的反证。**
- tail：A 比 C 差，A 比 D 好但逐 seed 符号翻转；68 个分子，不能单独支撑结论。

所以，A 相对 D 的小幅总体优势不是“未见结构泛化”带来的，而更可能是
low-support/ordinary 上的正则化；目标组反而略差。这个区分是本轮最重要的
观察之一：**“加性 reader 有正则化效果”与“加性 reader 改善未见结构”不是同一
件事；本轮只支持前者的一小部分，不支持后者。**

### 3.4 局部分支确实贡献，失败不是“没用到局部项”

A 的 `wS` 幅度、相关性、零局部 MAE 都表明局部分支在强烈参与预测；CPU 检查还
证明 A 的局部扰动严格可加、D 的不可加。因此：

- 不是初始化把 `w` 压成零；
- 不是梯度没有传到 `phi→w`；
- 不是模型忽略了 `c_loc`；
- 不是“局部项没学会”。

而是：**即使局部项被充分使用，这个“局部可加 + 全局小修正”的函数族仍然不足以
超过 DCC-L 的整图 reader，尤其在未见环境和极端尾部。** 这使负结果更有信息量：
它否证的是“只要强迫局部可加就能解决瓶颈”的强版本，而不是“模型没优化好”。

### 3.5 与历史结果的关系

历史 DCC-L 的 Locked 是 `0.08357/0.08342`，本轮 A 是 `0.08439/0.08469`，
D 是 `0.08783/0.08654`。A 没有复制 DCC-L 的优势；D 也没有。历史 DCC-L 的
reader 更大（197,633 vs A 148,513 / D 156,673），且其 dense reader 允许局部状态
之间的交互。本轮的 A/D 都把 reader 缩小并改写，结果都略差于 C；其中 A 比 D 好，
说明“缩小 reader + 加性 sum”有一点正则化收益，但不足以弥补容量/交互的损失。

历史 DCC-L 在训练时曾评估 Locked（用于日志），但 best checkpoint 由 Monitor
选择；本轮没有重训、没有改历史文件。历史 C 的 official valid 读取也只发生在
旧轮次，本轮未触碰。这个协议差异不影响“C 的 best-Monitor 预测是冻结的”这一
比较，但应在解释时记住：C 不是在本轮协议下重新训练的。

### 3.6 统计与协议限制

- 只有 2 个 seed；Locked 1,000 行；tail 68 行。大多数 CI 含 0；`A−D` 的
  seed-mean CI 含 0，`A−C` 的 CI 含 0。
- 历史同数学 ZINC 的 run-to-run 漂移记录到约 `0.0026`（valid），历史 DCC-L
  Monitor 两 seed 差 `0.00462`。**小于 0.003 的 Locked 差异不应被当作可靠
  材料性效应**；这正是预注册把 0.003 作为门槛的原因。
- A vs D 是容量匹配的；A/D vs C 不是。A−C 的容量混淆无法在本轮消除。
- A/D 的 reader 初始化流不同（`seed+20021` vs `seed+20022`），所以 A−D 还包含
  reader 初始化噪声；2 个 seed 无法把 `0.00264` 与这部分方差干净分开。A/D 的
  reader 参数数量也略有差异（A 148,513 vs D 156,673），总参数匹配但分布不同。
- best-Monitor 选点本身有噪声：D_s0 的 final 在 Locked 上比 best 好 `0.0024`；
  主表按协议用 best，final 只作辅助。A 的 final 与 best 基本一致。
- 分析阶段有工程重跑（见 1.5），但 8 个 Locked 前向各一次；没有根据 Locked
  结果改模型/阈值/判定。这个偏差应公开，不能假装是“一次运行、零重跑”。

---

## 4. 决策

### 4.1 预注册判定

按 `PROTOCOL.md` 的优先级：

```
verdict = INCONCLUSIVE
```

理由（机器生成，`analysis/decision.json`）：

- A_mean `0.084543`；D_mean `0.087184`；C_mean `0.083498`。
- `gain_A_vs_D = 0.002641`（<0.003）；`gain_A_vs_C = −0.001045`（没有超过 C）；
  `gain_D_vs_C = −0.003686`（D 更差）。
- 逐 seed A<D 为真，但 A<C 为假；ordinary gain vs C/D =
  `0.000038/0.002334`；top-10 gain vs C/D = `0.001880/0.004582`（符号已修正；
  但 overall/ordinary 仍不达标）。
- 两个关键组都不稳定：`low_support` 的 A-vs-C 只有 s0 为正；`any_unseen` 两个
  seed 都差于 C。无局部通路塌缩。

没有触发 `ADDITIVITY_NOT_SUPPORTED`（A 没有比 D 或 C 差 ≥0.003），也没有触发
`CAPACITY_ONLY`（D 没有比 C 好 ≥0.003；A−D 虽小但逐 seed 都为正）。因此落在
`INCONCLUSIVE`。按协议**不加第三个 seed 救援**。

### 4.2 五个问题

**1）A vs D，谁更好？**

A 在两个 seed 的 Locked best 上都更好：A `0.084394/0.084693`，D
`0.087832/0.086537`；seed-mean A−D `−0.002641`（A 更好），改善分子比例
`0.560`。方向一致，但幅度低于 0.003，CI 含 0（`[−0.005912, +0.000747]`），
final checkpoint 的差距更小（约 `0.00157`）。结论：**A 略好于 D，但不是可靠
的材料性优势；只能算弱信号。**

**2）A vs 未重训的历史 DCC-L，谁更好？**

历史 DCC-L 更好。A−C `+0.001045`（A 更差），两个 seed 都是正的
（`+0.000820/+0.001270`）；ordinary 组基本打平，tail 组 A 差 `0.01588`。
A 没有超过未重训的 DCC-L，更不能说“加性 reader 修复了 DCC-L 的瓶颈”。

**3）收益是在 low-support/unseen，还是只是尾部噪声？**

- 对 D：A 的收益不是尾部噪声；ordinary `−0.00233`、low-support `−0.00436`，
  top-10 剔除后 `−0.00432`，在典型分子上仍在。但 `any_unseen` 基本为零
  （`−0.00056`）。
- 对 C：A 没有总体收益（`+0.00105` 更差）；ordinary 打平，tail 更差
  `+0.01588`；`any_unseen` 更差 `+0.00644`。top-10 剔除后 A 略好
  `0.00166`，说明总体差距集中在少数极端分子，不是全面退化。
- **目标组 `any_unseen` 没有收益；两个 seed 都比 C 差约 0.0064。**
  low-support 上 A 比 D 好、比 C 只有不稳健的小优势。尾部 68 个分子方差大，
  不能把 A 的问题简单归为“尾部噪声”；unseen 组的负结果是独立的、方向一致的。

**4）局部分支到底有没有贡献？**

有，而且很强。`wS` 的 std 为 `0.851/0.779`，`psi` 为 `1.131/1.225`；局部项
占平均绝对贡献 `43.3%/44.9%`；`corr(wS,y)=0.917/0.890`；把 `wS` 置零后
Locked MAE 从 `0.0844/0.0847` 恶化到 `0.6906/0.7822`（Δ `+0.606/+0.698`）。
没有塌缩。CPU 干预检查也确认 A 的局部扰动严格等于 `wᵀδ`、与其他 `h` 无关。
所以本轮否证的不是“局部项没被使用”，而是“局部项可加就足够”。

**5）够不够进入 official 10k，还是应该关闭这条路线？**

不够。预注册的 `PROMISING_READER_BIAS` 条件没有满足：A 没有在两个 seed 都超过
C；A−C 为负；A−D 只有 `0.00264 < 0.003`；`any_unseen` 没有稳定收益；没有一组
同时满足 A-vs-D 与 A-vs-C 的 ≥0.003 稳定增益。**不建议进入 official 10k。**
建议关闭这一具体“加性 reader / reader 聚合是主要瓶颈”的路线，而不是继续在
Locked 上搜索 reader 变体。若未来要继续，应该是一个新的、预注册的问题（例如
与 C reader 等容量的 dense 对照，或专门针对尾部/上下文交互的机制），而不是
本轮结果的“再调一调”。

### 4.3 是否讨论 official-test 0.05x

不讨论。协议规定只有在 A 对历史 DCC-L 的 Locked 增益约 `0.005–0.007` 且稳健时
才允许讨论到 official-test `0.05x` 的距离；本轮的 A−C 是 `+0.001045`（A 更差），
不满足门槛。**不做任何 official-test 分数换算。**

### 4.4 可以带走的结论

1. 在容量匹配下，**加性局部 sum + 小全局修正**比 dense reader 略好，尤其在
   low-support/ordinary；这是对“reader 聚合方式有影响”的弱支持。
2. 这个优势**不足以超过原始 DCC-L**，也**没有出现在未见结构组**；目标组反而
   两 seed 都差约 0.0064。
3. 局部分支被强烈使用，因此负结果不是优化失败；是函数族表达力的限制。
4. 总体判定 `INCONCLUSIVE`；不追加 seed，不进入 official 10k，不把内部 Locked
   差异换算成官方成绩。
5. 负结果应被保留：它把“加性 reader 能解决有限标签泛化”的强版本降级为
   “加性 reader 是一个小尺度正则化，但不是主要瓶颈”。

---

## 5. 复现与资产

### 5.1 运行 ID / commit

| 项 | 值 |
|---|---|
| 训练代码（seed 0） | `9be30bf9ca4faacb41607f4b91f51e99e4214eaf` |
| Pilot 决策 commit | `9a088883ed4a` |
| 训练代码（seed 1，与 seed 0 代码相同 + pilot 记录） | `9a088883ed4a` |
| freeze commit | `677ac820da0098217a25470908a4074d423b244f` |
| 分析结果首次 commit | `6c018e76a2f454e88cf5253cdfac1c6a6e457b0f` |
| top-10 符号修正 commit | `732f91a08ba6d5e1201b58cb433dc7b053e69940` |
| A_s0 | `ar-A-s0-20261026-131924-4b7a9566` |
| D_s0 | `ar-D-s0-20261026-131924-88ec5078` |
| A_s1 | `ar-A-s1-20261026-134845-67062193` |
| D_s1 | `ar-D-s1-20261026-134845-919835a1` |
| smoke A（128/315） | `ar-smoke-A-20261026-131349-1f66c209` / `ar-smoke-A5-20261026-131628-daf47ae7` |
| smoke D（128/315） | `ar-smoke-D-20261026-131349-ab3620f6` / `ar-smoke-D5-20261026-131806-4d7ea9ac` |

主机 `a100-2`，2×A100-SXM4-40GB，driver 550.163.01，torch 2.5.1+cu124，
CUDA 12.4；远端 marker 在 `res` 上按 commit 部署。历史 DCC-L 文件未修改。

### 5.2 关键资产 SHA-256

| 文件 | SHA-256 |
|---|---|
| `configs/ar_v1.json` | `a436d9beef111214312d24750c53320fea63e3133d99e7323b2d28c66aeadd9b` |
| `PROTOCOL.md` | `4f413b587e16788be5a1c288ae94a5114716b0ce5c9d5bba34386b9c131d8e0b` |
| `analysis/cpu_checks.json` | `8effe6afc7fa4dd1fe65964ca26e0f6e2f0da65df4782174e51cffdc6565423d` |
| `analysis/freeze.json` | `ecc3a7de6ac4dd85e679c5d09e5490b413bc3d791e272939dacafba4e6b2dda4` |
| `analysis/locked_read.json` | `2c28471b10851510f5a50dd134b189d92058215a01750889bcbeef8526eae7bb` |
| `analysis/summary.json` | `efa180f3e2fa77e598dbf2e81caaf3b059727e86edd58c4905e4101ef31a8880` |
| `analysis/decision.json` | `28eb984679f1608a054c46282e8d2514fc7571b8cb6038b0bd6dbb286d844ac6` |
| `analysis/tables.md` | `fa163043440d6c63c9e9eca6e6ff31a92f5a211827a50c3badc79c1132a3e209` |
| `analysis/training_ledger.json` | `8140592d4d93b5b0a4eaf812d82acff8e5880b81c09b750a55cbd04ec596ecf5` |

A/D best-Monitor checkpoint（完整 hash 见 `freeze.json`）：

- A_s0 `69dbcd62181ace1cc46d0f41562d4506b0b8c27d3dfd580c6daa155dda6d3434`
- A_s1 `a9ccc70f4e4e41590e0564e5fd847274f8e1dfe0817ce9c66b6a8311113c4bc9`
- D_s0 `89209cca81f5b8c70d44a1e98bbc15f8300ef41ad190834e94ed09adc8ba9f19`
- D_s1 `f641708db3b81fce230d93f681ef2ecf2295db1821eec6f2e59bccbcad2c5d4f`

历史 DCC-L best-Monitor：

- C_s0 `6a145dc69e81aed9c1f41f572a6956f8a7351dacba9ddcaa2daf4e71930f5817`
- C_s1 `882fae17de42ef007b79750e6851e73dd414316a3c2efa15b2468892ce1ff35c`

### 5.3 文件

```
tracks/structural-sufficiency/zinc_additive_reader_v1/
  PROTOCOL.md
  REPORT.md
  configs/ar_v1.json
  code/ar_common.py
  code/ar_models.py
  code/ar_train.py
  code/ar_checks.py
  code/ar_analyze.py
  code/ar_ledger.py
  code/tests/test_ar_models.py
  analysis/cpu_checks.json
  analysis/freeze.json
  analysis/locked_read.json
  analysis/locked_preds/*.npz
  analysis/summary.json
  analysis/decision.json
  analysis/tables.md
  analysis/training_ledger.json
  analysis/a_component_quantiles.json
  analysis/protocol_consistency.json
  runs/{A_s0,D_s0,A_s1,D_s1}/          # git-ignored; hashes in freeze.json
  runs_smoke/*/                         # git-ignored; smoke ledger in training_ledger.json
```

`runs/` 与 `runs_smoke/` 是本地/远端训练产物，不作为 Git 资产提交；checkpoint 与
预测的 SHA-256 已写入 `freeze.json`，训练曲线与逐 run 账本已写入
`training_ledger.json`。Locked 逐分子预测 npz 已随分析结果提交（只含行号与预测，
不含标签）。

---

## 6. 最终一句话

**在本轮预注册的受控条件下，加性 reader 比容量匹配的 dense reader 略好
（Locked seed-mean −0.00264），但没有超过未重训的 DCC-L（+0.00105），也没有
改善未见结构（any-unseen 比 DCC-L 差 +0.00644）；局部通路强烈参与，故负结果
不是优化失败，而是“局部可加 + 小全局修正”这一函数族的表达力限制。判定
`INCONCLUSIVE`；不进入 official 10k，关闭这一具体 reader-bias 路线。**
