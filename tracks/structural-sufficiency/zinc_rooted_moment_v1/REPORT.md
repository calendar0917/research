# zinc_rooted_moment_v1：把邻域端点特征分布的二/三阶矩接到 root 上，能改善 ZINC 回归吗？

**Protocol:** `ssrc-zinc-rooted-moment-v1`
**冻结提交（PROTOCOL + config + code + CPU 检查一起）:** `0255e32`（缓存可复现性补丁 `d7d1a71`，`checks --rebuild` `52e3824`，分析期补丁 `d273ed2` / `c2a98c0` / `6575bc3`）
**正式 run 代码:** `d1faaa2`（六个 `train.json` 都记录此 commit；之后的 commit 只改分析/探针代码，不改训练路径）
**状态:** 完成。六个正式 run（P/M/G × seed 0/1）全部完成；**official test 未读取**（冻结触发条件不成立）。
**自动 verdict:** `NO_STABLE_GAIN_OVER_P`；`h1_supported=false`，`h2_supported=false`。

---

## 0. 一句话结论

把每个原子的“以该原子为 root 的多尺度游走邻域”压缩成**端点特征分布的二阶矩 C2 与三阶中心矩 C3**，再用 root 自己的 query 通过正二次核读出（`F = μ + (2aC2q + C3[q,q])/Z`），**没有**改善 ZINC valid：M 相对同轮 P 在两个 seed 上分别差 **+0.008008 / +0.007356**（T100，`delta = M − P`，正=更差；均值 **+0.007682**），G（同参数、同输入、同初态，只把 C3 置零的高斯闭合）也同样差 **+0.008634 / +0.006297**（均值 **+0.007465**）。也就是说，损失几乎全部来自“多出这条 moment 读出通路”本身，而不是来自三阶偏斜：M−G 在 T100 上两个 seed **方向相反**（−0.000626 / +0.001059），best-valid 上虽两个 seed 都是 M 略差（+0.0011）但 CI 跨 0。official test 因预注册触发条件不成立而没有打开，本轮**不能**声称任何 test 数字。

---

## 1. 六个问题与直接回答

**Q1（H1）M 是否在两个 seed 上都优于同轮 P？均值增益是否 ≥0.003？**
否。T100：M 相对 P 的增益（正=更好）为 **−0.008008（s0）/ −0.007356（s1）**，均值 **−0.007682**；best-valid 为 **−0.008712 / −0.007099**（均值 −0.007906，95% 配对 bootstrap CI **[+0.002307, +0.013378]**，即“M 更差”的区间不含 0）。两个 seed 方向一致，但方向是**变差**。H1 不成立。

**Q2（H2）C3 偏斜项本身是否有可重复贡献（M−G）？**
没有可重复证据。T100 的 M−G 增益为 **−0.000626（s0，M 略好）/ +0.001059（s1，M 略差）**，符号翻转，均值 −0.000216；best-valid 两个 seed 都是 M 略差（+0.001075 / +0.001083，均值 +0.001079，CI **[−0.001729, +0.003901]** 跨 0）。固定权重诊断显示 C3 在 M 的已训权重下**确实贡献了一点**（关掉 C3 后 valid MAE 变差 +0.000285 / +0.000492），但这点贡献不足以在重新训练中体现为稳定优势。H2 不成立（“C3 完全无用”也不对，见 Q4）。

**Q3 损失/增益在分层上如何分布？**
M−P（best-valid，seed 均值）：
`ordinary_935 +0.006858`、`unseen+ordinary +0.017314`、`seen_ordinary +0.003544`、`tail_65 +0.022981`、`seen_tail +0.026706`、`unseen_tail +0.017393`、**去掉 row 172 后 +0.006729**、**去掉最大 10 个分子后 +0.007800**（s0 +0.008442 / s1 +0.007135）。
即：损失普遍存在、在 unseen 与 tail 上更大，row 172 与最大 10 个分子都**不能**解释这个负结论。**没有任何一个分组在两个 seed 上同向改善**：`ring_rich_213` 的 seed 均值为 −0.001174，但它来自两个 seed 方向相反（s0 **+0.002244**、s1 **−0.004592**）；`ring_rich_unseen_47` 更极端（+0.026605 / −0.035858，只有 47 个分子）。这类均值不能当作机制证据。final checkpoint 同向（ordinary +0.006186、unseen+ordinary +0.017013、tail +0.029787）。

**Q4 模型是否真的在用 rooted moments？**
是，且实现与缓存在数值上被验证：
- `q = 0` 时 `F ≡ μ`、`log Z ≡ log(1+ε)`：真实缓存上 `max|F−μ| = 0.0`，`max|log Z − log 1.05| = 4.47e-8`。
- query 不是摆设：训练后 `‖q‖` 均值 **3.38–3.41**（上限 4，接近饱和），4 个头 3.3–3.5。
- C3 项相对线性项（`2aC2q`）的范数比中位数 **0.020–0.022**，随尺度单调上升（k=0 为 0，k=12 为 0.036）——C3 存在但只占 ~2%。
- `log Z` 极小（逐尺度均值 0.006–0.039），说明核几乎没被“归一化”推向非线性区。
- 固定权重把 C3 关掉：M_s0 best MAE `0.088184 → 0.088470`（+0.000285），M_s1 `0.086726 → 0.087219`（+0.000492），预测平均变化 0.0028 / 0.0035、最大 0.1285 / 0.0715。方向说明 C3 在**给定权重**下帮了一点点，而不是纯噪声。

**Q5 与旧工作和文献的边界在哪里？**
见 §7 的表。一句话：本轮的新意很窄——**精确、无标签、root 条件化**的端点特征分布二阶/三阶矩摘要，加上**同参数同输入的高斯闭合对照**；不新在“核/多项式注意力/多尺度预计算/rooted readout”这些组件上，也不新在“用矩做特征”这个想法上。

**Q6 成本、可复现性与 official test 政策？**
六个正式 run 合计 **2.546 GPU·h**；含 1 次失败的 timing、2 次失败的 probe 与 2 次成功的预跑/探针，共 **2.564 GPU·h**（上限 5.0，预注册 formal+pre-formal ≤4.7）。墙面时间 20:01→22:29（约 2 h 28 min）。CPU 侧：全量缓存 **26–37 s**（14–24 workers，峰值 RSS 8.9–9.3 GB，train 6.74 GB / valid 0.67 GB），23 项 CPU gate 与 5 个单元测试全绿；缓存字节在“不同 worker 数 / 不同 `OMP_NUM_THREADS`”下**逐字节一致**（train `5e830939…`，valid `277f02cf…`），本地与 res 上哈希相同。official test **未打开**：预注册的 U/分组触发条件（两 seed 均改善、均值 ≥0.003、ordinary best+final 均改善、unseen+ordinary final 不恶化且均值 ≥0.005）三项全 false，因此本轮没有 test 数字，也不允许用它补救。

---

## 2. 方法与实现（冻结内容摘要）

### 2.1 数学核心

无向简单邻接 `A`，`B = I + A`，尺度 `K = {0,1,2,3,4,6,8,12}`，`W_k = B^k`（整数，用增量矩阵幂计算，并与暴力游走枚举逐元素相等）。对 root `i`：

```
m_ik = Σ_j W_k[i,j]          p_ik(j) = W_k[i,j] / m_ik
μ_ik = Σ_j p_ik(j) u_j
C2_ik = Σ_j p_ik(j) (u_j−μ)(u_j−μ)^T
C3_ik = Σ_j p_ik(j) (u_j−μ)⊗(u_j−μ)⊗(u_j−μ)
κ(q,u) = (1 + qᵀu)² + ε,  ε = 0.05,  a = 1 + qᵀμ,  Z = a² + qᵀC2q + ε
F_ik(q) = μ_ik + [2a·C2_ik q + C3_ik[q,q]] / Z_ik  ≡  Σ_j p_ik(j)κ(q,u_j)u_j / Σ_j p_ik(j)κ(q,u_j)
```

`Z ≥ ε` 恒成立（`a² ≥ 0`、`C2 ⪰ 0`），全流程无 clamp。**C3 不是三节点 motif、不是三角/环计数**：它是同一端点特征分布的**三阶中心矩（偏斜）**；本轮任何文字都不把它写成最短路、共振或子结构。

缓存（**每个 root 原子**、每个尺度、每个视图）：`μ`(12)、`C2`(78 packed，含对角)、`C3`(364 packed：78 个 `a≤b≤c` + 286 个 `a<b<c` 混合项，多重数已折入)、`log m`(1) = 455 个 float32；每原子 7,280 个 float32（29.1 kB）。train 231,664 原子 = **6.74 GB**，valid 23,083 原子 = **0.67 GB**。CPU float64 构建、float32 存储。

一维见证（固定先验 `(0.5,0.5)`，`q=1`，与协议一致）：A `0.3846153846`、B（含 C3）`0.4526064213`、B 的高斯闭合 `0.3846153846`、B 的三阶中心矩 `0.08838834765`；直接求和与矩形式差 `~1e-14`（门槛 `<1e-10`）。

### 2.2 端点特征（固定，不训练）

`u_raw = [raw_atom_108 ‖ frozen_SSL_atom_96]`（204 维）。`memory_atom_96` 来自冻结的 M1 atom encoder
`tracks/structure-contribution/results/bjoc_full10k_pretraining_transfer_v1/pretrain_encoder_s0.pt`
（SHA-256 `006fabc1…`），26,336 参数，`requires_grad=False`，**不在 optimizer 里**。两视图 `R_v ∈ R^{204×12}`，生成器 seed 7301/7302，`u_j^{(v)} = tanh(standardize(u_raw)_j R_v)/√12`。列标准化与投影都由 train-only 统计冻结进 `cache/rmc_norm.pt`。

### 2.3 模型（单预测器，415,250 online 参数）

```
(a,b,w) = P3Encoders(batch)                       # 冻结架构、权重在线
e_i = MLP(290→128→96)([a_i, mean_j b_ij, mean_j w_ij, log1p deg_i, log1p njunc_i])
q   = 4·tanh(MLP(96→48)(e_i)).reshape(4,12)/√12    # head 0/1→view0，head 2/3→view1，各读 8 个尺度
readout = [e(96) ‖ F(4·8·12=384) ‖ logZ(32) ‖ logm(8) ‖ global33 ‖ deg(1) ‖ log1p n_atoms ‖ log1p n_bonds]  # 556
h_i = MLP(556→224→96) ; t_i = MLP(96→128→1)
ŷ = graph_bias + Σ_i t_i / train_mean_n_atoms
```

无 LayerNorm/BatchNorm/dropout，无残差门控/融合，无旧 P3 预测旁路。

| 块 | 参数 |
|---|---:|
| P3 encoders（在线） | 202,048 |
| e_mlp 290→128→96 | 49,632 |
| q_proj 96→48 | 4,656 |
| h_mlp 556→224→96 | 146,368 |
| t_mlp 96→128→1 | 12,545 |
| graph_bias | 1 |
| **M / G / (U) 合计（optimizer）** | **415,250** |
| 冻结 memory atom encoder | 26,336（不进 optimizer） |
| **部署合计** | **441,586**（< 500,000） |
| P（逐字 `P3NullModel`） | 413,505 |

### 2.4 三个臂

- **P**：`P3NullModel` 逐字实例化（与 nr_v1 相同路径）。其 seed-0 初始 state 哈希 `c58e7567…` 与 nr_v1 P 的初始哈希**完全相同**（本轮单独验证过），说明 P 路径没有漂移。
- **M**：上式，含 C3。
- **G**：`use_c3=False`，同一公式里把 C3 贡献置零（同均值同协方差的高斯闭合）；同参数、同初态（同一 canonical state 克隆）、同数据顺序、同 readout 维度。M/G 的初始 state 哈希逐 seed 相同（s0 `4939993b…`，s1 `0c79af57…`）。
- **U（未运行）**：分子内 uniform 分布控制，只在 §1-Q6 的触发条件成立时才启动。

## 3. 冻结与检查

23 项 CPU gate（脚本 `checks`，真实缓存 + 合成数据；`analysis/cpu_checks.json`）全部通过：见证值、恒等式（`max|F| 1.6e-15`，`max|Z| 2.1e-14`，门槛 1e-10）、`q=0`、点质量、单原子全尺度、packed vs full、`C2` PSD（最小特征值 −1.0e-10 为 float32 舍入）、`k=0` 精确零、`W_k` vs 暴力枚举（精确相等）、大 q 的 `Z ≥ ε`（真实缓存 `min Z = 1.008`，无 clamp）、`dF/dq` vs 直接求和 autograd（8.9e-16）、参数量（P 413,505 / M=G=415,250 / 部署 441,586）、P/M/G 配对初态、M 强制 C3=0 与 G 输出**逐位相等**（0.0）、encoder replay（0.0）、空邻接（`deg=0`、`njunc=0` 的 `mean_b/mean_w` 精确为 0）、memory encoder 缓存（5.5e-9 < 2e-6）、真实缓存 `q=0`、batch vs 单分子（4.7e-10）、反向 batch（0.0）、重编号/键方向不变性（≤6.9e-18）、梯度（27 个参数张量的梯度全部有限；除 `graph_bias` 在对称 L1 符号下可能抵消外各组非零，`graph_bias` 在 sum-loss 下梯度非零；optimizer 恰好覆盖 415,250）、冻结统计哈希。另有 5 个单元测试（`code/tests/test_rmc_math.py`，1 s 内）覆盖见证值、恒等式与梯度、packed/full、`W_k`、置换/单原子。

## 4. 结果

### 4.1 主表（valid，raw-y MAE，越低越好）

| arm | seed | best-valid | epoch | final(600) | W20 | T100 | eval-train@best | eval-train@final | params | wall h |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P | 0 | 0.079472 | 390 | 0.081388 | 0.081383 | 0.081451 | 0.014776 | 0.000616 | 413,505 | 0.313 |
| P | 1 | 0.079627 | 412 | 0.080566 | 0.080509 | 0.080546 | 0.012623 | 0.000577 | 413,505 | 0.354 |
| M | 0 | 0.088184 | 365 | 0.089441 | 0.089416 | 0.089459 | 0.016741 | 0.001248 | 415,250 | 0.456 |
| M | 1 | 0.086726 | 392 | 0.087955 | 0.087737 | 0.087902 | 0.011721 | 0.001390 | 415,250 | 0.497 |
| G | 0 | 0.087109 | 232 | 0.090089 | 0.090013 | 0.090084 | 0.028821 | 0.001335 | 415,250 | 0.415 |
| G | 1 | 0.085643 | 461 | 0.086855 | 0.086741 | 0.086843 | 0.007154 | 0.001459 | 415,250 | 0.473 |

`W20` = 任意连续 20 epoch 的最小 valid 中位数；`T100` = 最后 100 epoch 的中位数。两者都只是持续性诊断，不是另一个可选 checkpoint。

### 4.2 对比（`delta = MAE(first) − MAE(second)`，正=first 更差）

| contrast | seed | best Δ | 95% CI | final Δ | T100 Δ | ordinary Δ | unseen+ordinary Δ | ring-rich Δ |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| M−P | 0 | +0.008712 | [+0.001776, +0.015859] | +0.008052 | **+0.008008** | +0.007619 | +0.020102 | +0.002244 |
| M−P | 1 | +0.007099 | [+0.000741, +0.013107] | +0.007388 | **+0.007356** | +0.006096 | +0.014526 | −0.004592 |
| M−P | mean | +0.007906 | [+0.002307, +0.013378] | +0.007720 | +0.007682 | +0.006858 | +0.017314 | −0.001174 |
| G−P | 0 | +0.007637 | [+0.000516, +0.014794] | +0.008700 | +0.008634 | +0.007300 | +0.015360 | −0.001235 |
| G−P | 1 | +0.006016 | [−0.000090, +0.012257] | +0.006288 | +0.006297 | +0.004524 | +0.014201 | +0.003907 |
| G−P | mean | +0.006826 | [+0.001515, +0.011973] | +0.007494 | +0.007465 | +0.005912 | +0.014780 | +0.001336 |
| M−G | 0 | +0.001075 | [−0.004718, +0.007245] | −0.000648 | **−0.000626** | +0.000318 | +0.004743 | +0.003479 |
| M−G | 1 | +0.001083 | [−0.002737, +0.004812] | +0.001100 | **+0.001059** | +0.001573 | +0.000325 | −0.008499 |
| M−G | mean | +0.001079 | [−0.001729, +0.003901] | +0.000226 | +0.000216 | +0.000946 | +0.002534 | −0.005287 |

（CI 来自逐分子配对 bootstrap，10,000 次，先在同分子上对 seed 取均值，不把 2,000 条预测当独立。ring-rich 的 seed 均值列只为展示方向：两个 seed 符号相反。）
读法：M 与 G **都**比 P 差约 0.007–0.008，CI 基本不含 0；M−G 在 T100 上符号翻转，best-valid 上两 seed 同向但 CI 跨 0。

### 4.3 敏感性与尾部

- **row 172**：M_s0 的 AE = 5.8844、M_s1 = 6.5336（P：4.2731 / 5.7780）。去掉 row 172 后 M−P 仍为 **+0.006729（best）/+0.005910（final）**，tau 结论不依赖它。
- **去掉最大 10 个分子**：M−P = +0.008442 / +0.007135，同向。
- **tail vs ordinary**：损失在 tail 更大（best +0.022981，final +0.029787），而 ordinary 也稳定为负（+0.006858 / +0.006186）。

## 5. 机制诊断（`analysis/probe_summary.json`，固定权重、无梯度）

- `q=0` 恒等：`max|F−μ| = 0.0`、`max|log Z − log 1.05| = 4.47e-8`（真实缓存）。
- query 尺度：`‖q‖` 均值 3.38–3.41（4 个头 3.3–3.5），接近 `4·tanh` 的上限，说明读出确实在工作点上被使用。
- C3/线性项范数比：中位数 0.020–0.022，逐尺度 0 → 0.036；即 C3 是**小而随尺度增长**的修正。
- `log Z`：逐尺度均值 0.006–0.039（`Z ≈ 1`），核没有被推入强非线性。
- 关掉 C3（同一权重）：valid MAE 变差 **+0.000285（s0）/ +0.000492（s1）**，预测平均变化 0.0028 / 0.0035。**方向 = C3 有小幅正贡献**，与“重新训练 M vs G 无稳定差异”并不矛盾：前者度量给定权重下 C3 的作用，后者度量优化结果。

## 6. 预算、执行与可复现性

### 6.1 GPU ledger（含失败；`analysis/gpu_ledger.json`）

| item | run id | state | GPU h |
|---|---|---|---:|
| timing（CUDA reset 失败） | `rmc-timing-20261023-205056-beff8836` | failed | 0.0019 |
| timing（完成） | `rmc-timing-20261023-205334-2d3e072d` | completed | 0.0100 |
| formal P_s0 | `rmc-P_s0-20261023-205603-5dd9e3a0` | completed | 0.3186 |
| formal P_s1 | `rmc-P_s1-20261023-205616-1f6a0415` | completed | 0.3608 |
| formal M_s0 | `rmc-M_s0-20261023-211754-1d52c34b` | completed | 0.4622 |
| formal G_s1 | `rmc-G_s1-20261023-211907-4c925094` | completed | 0.4792 |
| formal G_s0 | `rmc-G_s0-20261023-214906-f04aae13` | completed | 0.4214 |
| formal M_s1 | `rmc-M_s1-20261023-214917-536749ff` | completed | 0.5039 |
| probe（索引错误 ×2） | `rmc-probe-20261023-222241-553b0ce2`、`…-222531-c79abaed` | failed | 0.0036 |
| probe（完成） | `rmc-probe-20261023-222902-64eb86f2` | completed | 0.0019 |

**正式合计 2.5461 GPU·h，含 pre-formal 与失败共 2.5636 GPU·h / 5.0 上限**（预注册 formal+pre-formal ≤4.7，余量 2.1）。CPU 任务（checks、clean、ledger）不占 GPU。两个 A100 并行，墙面 2 h 28 min。详细 JSON 见 `analysis/gpu_ledger.json`。

### 6.2 每个 run 的来源

六个 `runs/<arm>_s<seed>/run_extra.json` 均记录：协议 id、git commit（`d1faaa2…`）、config SHA-256、norm SHA-256、moment cache 的 train/valid SHA-256、SSL checkpoint 哈希与参数数（26,336）、online/deployment 参数量、`use_c3`、初态/ best / final 的 state 哈希、数据顺序流哈希、逐 epoch 曲线哈希、best/final 的 **回放一致性 MAE**（与保存在曲线里的数字差 < 5e-6 才通过）、`official_test_loaded=false`。`analysis/summary.json` 把这些聚合成机器可读的一处。

### 6.3 可复现性细节

- 缓存逐字节可复现：不同 worker 数（2 vs 8）与不同 `OMP_NUM_THREADS`（1 vs 默认）下 train/valid/norm 全部同哈希；本地与 res 同哈希。为此，游走矩与投影都用 `np.einsum(optimize=False)`（不用 BLAS 多线程归约），冻结 memory encoder 在 `torch.set_num_threads(1)` 下运行。
- P 与 nr_v1 的 P 初始 state 哈希相同；但两轮 600-epoch 轨迹不同（本轮 P_s0 best 0.079472 vs nr_v1 0.078611）。原因是 CUDA `index_add_` 的原子加与训练本身的混沌性，**不是**代码漂移。这给出一条重要的噪声基线：同架构同 seed 的 best-valid 在轮间可差 ~0.001–0.003，这也是本轮把 T100/两 seed/逐分子配对放在首位的原因。
- 峰值显存 7.55 GB（A100-40GB），其中约 7.4 GB 是 moment 缓存。

## 7. 边界：与旧工作、与文献

| 旧的轮次 | 与本轮的关系 |
|---|---|
| A / B 臂（`zinc_recipe_trunk_v2`） | 研究归一化位置与自身特征门控；没有矩，没有 root 条件读出 |
| BCCA（`zinc_bcca_factorial_v1`） | atom↔bond/junction 的 cross-attention；是学习的注意力，不是精确矩摘要 |
| SF（`zinc_structure_features_v1`） | 标量结构特征注入；是计数，不是端点特征分布的矩，也没有 query |
| NR（`zinc_native_relation_v1`） | 关系类型 ↔ token pair 的一轮消息传递；无矩、无 root query |
| R0 / R1（`notes/prior_work_boundary.md`） | 只用 rooted **count** 特征；无 C2/C3、无核读出 |

文献侧（均**不**声称是本轮提出的）：
**Linear Attention**（Katharopoulos et al., 2020；Choromanski et al., 2021）——`κ(q,u)=φ(q)ᵀφ(u)` 的核视角，本轮的正二次核就是它的一个具体特征映射；
**TaylorShift**（Nguyen et al., 2025）与多项式/核 GNN——`(1+qᵀu)²` 是二阶多项式核，本轮的 C3 项正是把它展开到三阶后的“偏斜修正”；
**SIGN**（Frasca et al., 2020）与 **MM-GNN**（Zhu et al., 2021）——多尺度邻域摘要的预计算思路，本轮的 `W_k` 缓存与它们同源；
**GRIT**（Ma et al., 2023）——rooted、query 条件化的读出，本轮的 `q_i` 读出在形式上与它同类。
本轮唯一较窄的贡献是：**精确、无标签**的端点特征分布 **二/三阶中心矩**，加上一个**同参数同输入的高斯闭合（C3=0）对照**，并用一维见证与直接求和恒等式把“矩 = 加权求和”钉死。

## 8. 不能声称什么 / 下一步

**不能声称：**
- 不能声称 rooted moments 在 ZINC 上有帮助——两个 seed、两个端点（T100 与 best-valid）、ordinary 与 unseen+ordinary 全部为负。
- 不能声称 C3（偏斜）无用：固定权重下它贡献 +0.0003…+0.0005；但**重新训练**下 M−G 无稳定优势，所以只能说“在本预算与这个读出形式下，C3 的可重复增益未被证实”。
- 不能把 ring-rich 的 seed 均值（M−P −0.0012、M−G −0.0053）写成机制结论：两个 seed 的符号相反（M−P +0.0022 / −0.0046），`ring_rich_unseen_47` 只有 47 个分子，均值完全由 1–2 个分子支配。
- 不能把“G 与 M 都差 0.007”解释为“矩无用”：M/G 共用同一条额外读出通路（556 维 h_mlp、逐 root 的 F/logZ/logm 拼接），本轮的对照**无法**把“通路代价”与“矩信息”分开；这正是下一轮该控制的变量。
- 不能给出任何 official test 数字（未触发），也不能用 valid 数字替代 test。
- 不能声称 C3 编码了环/子结构/共振，或与最短路有关。

**下一步（若要再花 GPU，只做这一件事）：** 用一个**参数与 P 完全对齐**的读出（例如把 moment 读出做成对 P3 reader 的零初始化条件：`t_i = t_i^{P3} + 0`，或只让 moment 通路通过一个零初始化标量门进入），使 step 0 与 P **逐位等价**，再问“矩信息能否在不付通路代价的前提下被读进来”。如果不能，这条线应关闭，而不是调宽度/尺度集/ε/更多 seed。

## 9. 交付物

- `PROTOCOL.md`、`configs/rmc_v1.json`（冻结协议与配置）
- `code/rmc_moments.py`（数学核心：walk powers、packed 矩、恒等读出、直接求和对照、一维见证）
- `code/rmc_data.py`（冻结端点特征、缓存构建/存取、collate）
- `code/rmc_models.py`（RMC 模型、P/M/G/U 构造、参数与初态审计）
- `code/rmc_runner.py`（prep/checks/smoke/timing/train/score/analyze/probe/test）
- `code/rmc_ledger.py`（`rr jobs` → GPU ledger）
- `code/tests/test_rmc_math.py`（5 个快速单元测试）
- `analysis/cpu_checks.json`（23 项 gate）、`analysis/prep.json`（缓存字节/时间/RSS/哈希）
- `analysis/curves/*.csv`（逐 epoch 曲线，6 个 run）
- `analysis/scores/*.json`、`analysis/per_molecule/*.csv`（逐分子预测与分组）
- `analysis/main_table.{md,json}`、`analysis/decision.md`、`analysis/summary.json`（机器可读）
- `analysis/probe_summary.json`（固定权重机制诊断）
- `analysis/gpu_ledger.json`（含失败的 GPU 台账）
