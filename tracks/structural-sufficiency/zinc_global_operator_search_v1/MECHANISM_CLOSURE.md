# GOS 机制收束报告

范围：official valid + 冻结 checkpoint + 1 条新轨迹；**未读 official test**。总成本
**0.749 A100·h**（1 条 600-epoch 轨迹；零 GPU 分析不计）。valid 已被历史多轮使用，不是盲测。

## 1. A 与 DCC-L 是否改善不同类型的困难分子？

**没有。** 8 个同 seed 配对（A s0/s1 × DCC-L s0–s3）的 signed error 相关 0.79–0.96，
absolute error 相关 0.80–0.96；逐分子 A 更优比例仅 49–52%。用 DCC-L s0 的最差 10%（100 个）
做固定集合：A_s0/A_s1 各重合 65 个（Jaccard 0.48）；用 DCC-L seed 平均误差定义时重合 68 个
（Jaccard 0.52）。该集合贡献 A 总绝对误差的 44.3%、DCC-L 的 44.4%——两者在同样的尾部上花掉
差不多比例的总误差。

按规模（原子数三分位）、环数（0–1/2/3/≥4）、旧 r2 seen/unseen 分层后，**没有任何一组**在两个
A seed 上同向、且 bootstrap 区间排除 0。唯一两个 seed 同向的是 2 环分子：A −0.00255
（95% CI [−0.00595, +0.00090]），幅度小于噪声，不能称为类别优势。规模、≥4 环、unseen、tail
的 per-seed 差值都翻号；tail 上 A 家族平均反而差 +0.0071（CI 跨零）。A 家族 valid 平均只比
DCC-L 好 0.00074（0.075248 vs 0.075990），不是“在某一类困难分子上更好”，而是整体小幅、
seed 相关的差异。

## 2. gate mass 的变化主要是平均尺度、分子级还是 query 级？

**主要是 query 级。** 冻结 `ho_A_s0` 上 mass 均值 0.349、中位数 0.086、5–95% 分位
[0.00025, 1.74]；52.6% 的 query < 0.1，21.4% < 0.01。质量方差分解：

| 来源 | 占比 |
|---|---:|
| layer/head 平均尺度 | 26.1% |
| 不同分子之间 | 20.6% |
| **同一分子内不同 query 之间** | **53.3%** |

去掉 layer/head 均值后，分子内 query 方差占剩余方差的 **72.1%**。`A_s0` 同向且更强：中位数
0.003，分子内占 mass 方差 58.4%。分子平均 mass 对 [log 键数、环数、环原子比例、最长环、unseen]
的 R² 为 0.13–0.77（中位 0.55，随 head 变），说明分子级部分与已有结构量有关；但分子内
log-mass 对局部键特征（环键、端点度和、双键）的 R² 中位仅 **0.023**（0.001–0.067）。
所以 mass 不是一个固定 layer/head 缩放，也不是已有数量特征的简单重复；它带有大量 query 级、
关系特异的自由度。这只是机制描述，不能当成监督收益。

## 3. Constant-Mass 能否取代动态 mass？

单轨迹 `cmass_A_s0`（route A、seed 0、同一 9,000/1,000 划分、配方 B、460,029 参数、12 个 `rho`）
跑满 600 epoch，**0.749 GPU·h**；5 项正确性检查全过（非 rho 初态逐位相同、fp64 公式等价、
padding/batch 不变、梯度有限、checkpoint 重载）。`rho` 初值只来自 9,000 行无标签结构的公共初始化
平均 mass（0.85–1.16）。

| 口径 | cmass | ho_A_s0 | cmass − A |
|---|---:|---:|---:|
| best-valid@ep | 0.084467@397 | 0.081299@364 | **+0.003168** |
| W20 | 0.086145 | 0.083016 | +0.003129 |
| 内部留出 1,000 | 0.117924 | 0.114666 | **+0.003258** |

留出配对 bootstrap 95% CI 为 **[−0.000563, +0.007018]**，cmass 更优分子 48.8%，中位 AE 只差
+0.00064；去掉 cmass 最大 10 行后差 +0.00131，5% trimmed mean +0.00130。晚段同 epoch
（300–600）平均差约 +0.0034，不是单个 epoch 或单个异常分子造成。学到的 `softplus(rho)` 为
0.58–0.97：它只学到了全局尺度，没有复现 A 的 query 级质量抑制（中位 mass 0.086；A_s0 0.003）。

**按预注册判定：INCONCLUSIVE。** 两端点都差约 0.0032，方向一致、且不是单点；但留出区间跨零，
去掉最大 10 行后的差 0.0013 低于预注册的“明显落后”稳健门槛 0.0020。因此不能把它写成“动态
mass 已被证明必要”，也不能说“Constant-Mass 等价”。能确定的是：**Constant-Mass 不能无代价地
取代动态 mass；缺失的 query 级抑制只值约 0.003 MAE，接近本任务的种子/尾部噪声。**

## 4. 下一步

- **A vs DCC-L**：没有可用的“A 更擅长的困难类别”；两者主要错在同一批分子上，按结构分层路由
  没有依据。
- **gate mass**：query 级变化最大，但 A_s0 的 cell 尺度本身也很大（log10 方差 83% 在 layer/head），
  所以“动态 mass”同时包含固定尺度与 query 抑制两部分；本轮只关闭了后者的性能问题。
- **性能希望**：没有值得继续购买的证据。Constant-Mass 没有超过 A，A 也没有超过 DCC-L 的 test
  锚；不为一个约 0.003 的机制细节再买 seed 或新臂。
- **决定：STOP。** 保留机制结论（mass 是 query 级、非固定缩放；Constant-Mass 有小幅代价但证据
  不足以判必要），不启动第二 seed、不把 mass 拼进 reader、不做组合臂；本轮不提升新候选。

## 5. 证据与成本

`analysis/paired_A_vs_dccl.{json,md}`、`analysis/gate_mass.{json,md}`、
`analysis/cmass_checks.json`、`analysis/cmass_init.json`、`analysis/cmass_result.json`、
`analysis/holdout_A_cmass_s0.json`。训练 0.749 GPU·h（硬上限 1.0）；未读 test。
