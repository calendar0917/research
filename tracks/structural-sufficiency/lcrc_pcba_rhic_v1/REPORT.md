# REPORT — LCRC-RHIC v1

**轮次**：`lcrc-pcba-rhic-v1`
**模型**：`LCRC-RHIC`（seed 0，40 epochs，唯一一次正式训练）
**数据**：官方 `ogbg-molpcba`，官方 scaffold split（350,343 / 43,793 / 43,793，128 tasks）
**决策标签**：**`NO_CLEAR_ADVANCE`（无清晰增益）**

> 一句话结论：RHIC 的“分支内 + 根内两级共现聚合”在结构上确实被使用（机制诊断全部通过），
> 并把 **Validation** 从 CPC 的 0.265944 提升到 **0.268734**（+0.00279），
> 但 **Test** 从 CPC 的 0.262356 降到 **0.258416**（−0.00394）；
> 它优于 W2 与 RC-REAL，却不及平坦路径的 CPC 和迭代式的 W2-DYNAMIC。
> 预注册的 `PERFORMANCE_ADVANCE`（Valid ≥ 0.280 且 Test ≥ 0.270）与 `PARTIAL_ADVANCE`
> 均未达到，也不满足 `REGRESSION` 的双侧条件，因此按规则判为 `NO_CLEAR_ADVANCE`。

本报告中的数字均为本次唯一正式运行的实测值；`Test@40` 与 `Test best-Valid` 来自
封存后的唯一一次官方 Test 打分。除特别注明外，AP 为 **127 个可打分 Test 任务**上的
官方 macro AP（task 45 不可打分）。**106 任务高支撑集**定义为官方 Valid 与 Test
正例均 ≥ 20（由 CPC 轮冻结并由本轮重新校验）；其余 **22 个低支撑任务**单独报告，
绝不作为“可打分任务总数”。

---

## 0. 摘要

### OBSERVED（观测）

| 指标 | RHIC | CPC | W2-DYNAMIC | W2 |
|---|---:|---:|---:|---:|
| 参数量 | **1,117,253** | 1,043,333 | 1,128,468 | 994,324 |
| Valid best AP | **0.268734** (ep25) | 0.265944 (ep36) | 0.275067 (ep25) | 0.265273 (ep24) |
| Valid @40 AP | **0.266492** | 0.263993 | 0.268791 | 0.261867 |
| Test best-Valid AP | **0.258416** | 0.262356 | 0.262226 | 0.253617 |
| Test @40 AP | **0.254685** | 0.261492 | 0.260136 | 0.250986 |
| Train AP @40 | 0.605333 | 0.592958 | 0.654260 | 0.581875 |
| Train BCE @40 | 0.023889 | 0.024554 | 0.022108 | 0.025215 |
| Valid BCE @40 | 0.046720 | 0.046690 | 0.048017 | 0.046538 |
| GPU-hours | 1.9181 | 1.7620 | 1.7340 | 1.3949 |

- Valid 上：RHIC > W2-DYNAMIC? **否**（低 0.00633）；RHIC > CPC（+0.00279），> RC-REAL（+0.00102），> W2（+0.00346）。
- Test 上：RHIC > W2（+0.00480），> RC-REAL（+0.00248）；< CPC（−0.00394），< W2-DYNAMIC（−0.00381）。
- 106 高支撑集（Test）：RHIC 0.279189，CPC 0.280599，W2-DYNAMIC 0.286440，W2 0.275816；
  RHIC 相对 CPC **−0.001410**，48/106 改善、58/106 变差。
- 106 高支撑集（Valid）：RHIC 0.295093，CPC 0.291600；RHIC 相对 CPC **+0.003493**，67/106 改善。
- Valid→Test 落差：RHIC **0.010318**，CPC 0.003588，W2 0.011656，W2-DYNAMIC 0.012841。

### INTERPRETATION（解释）

1. 两级分支共现不是“死模块”。在冻结的 best-Valid checkpoint 上，
   `delta=0` 使 Valid AP 从 0.268739 掉到 0.218122（−0.050618），
   根内分支分配打乱后掉到 0.259601（−0.009139）；`q_b`/`cross_branch` 的非零比例、
   门控与残差尺度都表明模型确实在使用分支结构和二阶共现项。
2. 但“被使用”没有转化为“可迁移”。RHIC 的 Train AP 明显高于 CPC（0.6053 vs 0.5930），
   Train BCE 更低，而 Valid→Test 落差从 CPC 的 0.0036 扩大到 0.0103；
   同时残差尺度 `R = RMS(delta)/RMS(h0)` 从第 1 轮的 0.056 持续升到第 40 轮的 0.663
   （best epoch 25 时 0.554），门控均值从 0.527 升到 0.712。
   这与“二阶共现项在记忆 scaffold-specific 的分支统计量”一致：
   它提升了拟合与 Validation，但在 official scaffold shift 下更脆弱。
3. 与平坦 CPC 相比，层级化的收益只出现在 Valid 侧且幅度很小（+0.0028；
   高支撑 Valid +0.0035）；Test 侧反而回退 0.0039。Test 的损失并非均匀分布：
   127 个可打分任务中 57 升 70 降，逐任务中位数仅 −0.00067，
   但少数任务大幅下降（task 8 −0.1164、task 68 −0.0836、task 43 −0.0804），
   同时也有大幅上升（task 109 +0.0716、task 119 +0.0652、task 104 +0.0564）。
   这说明模型是在任务间“重新分配”性能，而不是整体变强。

### DECISION（决策）

按 §8 的预注册规则：

- `PERFORMANCE_ADVANCE`：Valid 0.268734 < 0.280 → 不满足。
- `PARTIAL_ADVANCE`：
  - 强单侧条件要求 Valid ≥ 0.273 且 Test ≥ 0.265 → 不满足；
  - 双侧改进条件要求 Valid > 0.267944 且 Test > 0.264356，且高支撑 Δ ≥ +0.002 →
    Test 0.258416 < 0.264356，不满足。
- `REGRESSION`：要求 Valid < 0.262944 **且** Test < 0.259356（或高支撑 Δ < −0.003）→
  Valid 0.268734 不满足；高支撑 Δ −0.001410 > −0.003，也不满足。
- 机制门（§7）全部通过 → 不是 `MECHANISM_INVALID`。

因此最终标签为 **`NO_CLEAR_ADVANCE`**：Valid 侧有小的提升，Test 侧有小的回退；
在单 seed 下这些差异不足以确立统计显著性，但方向相反，既不足以称为前进，
也不构成预注册意义上的回归。

---

## 1. 研究问题与预注册

本轮的固定问题：

> 在 W2 的局部编码与 readout 完全不变、且**不做任何迭代式隐状态传播**的前提下，
> 把 CPC-v1 的根化两跳路径 token 按“第一跳有向分支 `(i,j)`”做两级分组——
> 分支内保留均值与**不同 token 两两共现**均值，根内保留分支值的均值与
> **不同分支两两共现**均值——能否比平坦 CPC 更接近 W2-DYNAMIC 的增益？

预注册文件为 `PROTOCOL.md`（在第一次正式训练前冻结）。本轮只训练一个模型、
一个 seed、40 epochs；不训练第二个架构/seed/搜索，不重训 W2/CPC/DYNAMIC。
Test 只在封存后被读取一次，且从不参与模型选择。

---

## 2. 模型与参数预算

### 2.1 主路径（一次写入，无迭代）

```python
a, b, scal = base._encode(batch)                # 一次
h0 = base.drop(base.local(batch, a, b, scal))   # 一次

delta = RHIC(h0, b, path_index, bond_index)     # 一次，只读 h0/b
h = h0 + delta

logits = base.post_local(h, batch)              # W2 readout，一次
```

`h0, delta, h ∈ R^{N_atoms×128}`；`b ∈ R^{N_bonds×192}`。
RHIC 不读取 `h`，不回写原子状态，没有第二步传播。
`delta` 只写回根原子；`post_local` 只被调用一次。

### 2.2 路径与分支

路径沿用 CPC 的根化简单两跳路径 `i--j--k`（`i≠j, j≠k, i≠k`），
枚举完整、无采样/截断。分支是第一跳的**有向**物理键：

```
branch_id = 2 * physical_bond_id + direction
direction = 0 若 bond_index[e] == (i,j)，否则 1
```

`(i,j)` 与 `(j,i)` 是不同分支；`n_b=0` 的分支不编码、不计数；
`d_i=0` 的根 `delta_i=0`。分支 id 在运行时由 `cpc_bond_ij + bond_index`
导出，**没有新建缓存**。

### 2.3 两级统计（精确公式）

分支 `b=(i,j)`，其 `n_b` 个不同路径 token 为 `v_p ∈ R^64`：

```
mu_b  = (1/n_b) Σ_p v_p
D_b   = Σ_p (v_p − mu_b)²
q_b   = mu_b² − D_b / (n_b(n_b−1))
      = (S_b² − T_b) / (n_b(n_b−1)),  S_b=Σv_p, T_b=Σv_p²
q_b   = 0 exactly if n_b < 2
c_b   = [r_i(32), m_j(32), e_ij(16), mu_b(64), q_b(64), log1p(n_b)(1)]   # 209
beta_b= branch2(SiLU(branch1(c_b)))                                       # 64
```

根 `i`，其 `d_i` 个不同分支值为 `beta_b`：

```
mean_branch_i = (1/d_i) Σ_b beta_b
cross_branch_i= mean_branch_i² − D_i / (d_i(d_i−1))       # D_i=Σ(beta_b−mean)²
cross_branch_i= 0 exactly if d_i < 2
g_i           = [r_i(32), mean_branch_i(64), cross_branch_i(64),
                 log1p(path_count_i)(1), log1p(d_i)(1)]                   # 162
delta_i       = root2(SiLU(root1(g_i)))                                   # 128
delta_i       = 0 exactly if d_i = 0
```

`q_b` / `cross_branch` 是**不同元素有序对**的均值，不含自乘积；
实现用 `S²−T` 的稳定等价式，`D` 直接累加偏差平方，避免大数相减。

### 2.4 参数预算（精确到个位）

| 组 | 参数 |
|---|---:|
| W2 base（atom/bond encoders, local, f, rho） | 848,096 |
| W2 readout（post_local） | 146,228 |
| **W2 小计** | **994,324** |
| Token：LayerNorm(128) | 256 |
| Token：3×(128·32+32) | 12,384 |
| Token：bond 192·16+16 | 3,088 |
| Token：path1 192·96+96 | 18,528 |
| Token：path2 96·64+64 | 6,208 |
| Token：gate 96·1+1 | 97 |
| **Token 小计** | **40,561** |
| Branch：branch1 209·96+96 | 20,160 |
| Branch：branch2 96·64+64 | 6,208 |
| **Branch 小计** | **26,368** |
| Root：root1 162·192+192 | 31,296 |
| Root：root2 192·128+128 | 24,704 |
| **Root 小计** | **56,000** |
| **RHIC 小计** | **122,929** |
| **总计** | **1,117,253** |

相对 W2：+122,929；相对 CPC：+73,920；相对 W2-DYNAMIC：−11,215。

### 2.5 初始化与 epoch-0 恒等

- W2 seed-0 状态按位加载，trunk fingerprint 与历史 W2 一致（`5f7101a8…`）。
- RHIC 模块在 `torch.random.fork_rng(devices=[])` 内构造，使用私有
  generator（seed `0·1_000_003 + 2002`），不推进全局 torch RNG。
- `root2.weight` 与 `root2.bias` 置零，因此 epoch 0 时 `delta ≡ 0`，logits 与未训练 W2 相同。

实测：CPU 上 `max|Δlogit| = 0`；CUDA 上 `1.907e-6`（≤1e-4）；
CPU↔CUDA RHIC logits 差 `3.815e-6`。epoch-0 Valid AP 为 0.029063055
（历史未训练 W2 0.029063332，差 2.77e-7）。

---

## 3. 数据与训练协议

- 官方 `ogbg-molpcba`，官方 scaffold split，未新建任何 split。
- Train/Valid 标签来自冻结的 NaN-masked 工件；官方 Test 标签在训练与机制分析期间
  全为 NaN，封存后才通过 `cc_data.read_official_test_labels` 打开一次。
- 归一化：沿用 W2 的 Train-only 归一化，不重算。
- 批顺序：历史 W2 seed-0 的 40-epoch 计划，`global_plan_sha256` 与
  `consumed40_prefix_sha256` 在训练开始时校验；batch size 128，2,738 steps/epoch，
  共 109,520 steps。
- 优化器：AdamW，单参数组，W2 与 RHIC 同 LR；weight decay 1e-3，
  betas (0.9,0.999)，eps 1e-8。
- LR：1-epoch 线性 warmup 3e-5 → 3e-4，然后 109,520 步逐步 cosine 3e-4 → 1e-5。
- dropout 0.10（继承 W2，不改），grad clip 5.0，FP32，masked BCE。
- 不早停：40 epochs 全部训练并在每个 epoch 后计算官方 Validation AP；
  best-Valid 由 `argmax Valid AP`（并列取最早）选出，`@40` 是预注册终点。

---

## 4. 训练前审计（A–E，全部通过）

### A. 结构与缓存身份

- 复用 CPC 缓存：437,929 行、**34,369,144** 条路径、658,409,864 字节，
  `rows_sha256 85b8d38b…`、`cache_sha256 caad3e90…`；无任何 split 的标签。
- 全缓存分支审计（34,369,144 条路径）：方向不唯一 0、middle 错配 0、
  第二键错配 0、分支 id 越界 0，`(root,middle)` 与分支 id 一一对应
  （unique branches 11,655,674 = unique root-middle keys）。
- 11 个手算小图（单原子、零路径键、链、三角、方环、枝树、多路径环、K4 等）全部通过；
  独立 Python 枚举与模型的分支集合、计数、`q_b`/`cross_branch` 零值一致。
- 真实 batch 上独立 NumPy float64 复算：`mu` 3.7e-9、`q` 7.3e-10、
  `mean_branch` 5.0e-9、`cross_branch` 1.0e-9（最大绝对差）。
- 分组见证（witness）：图 A `{x,x},{y,y}` 与图 B `{x,y},{x,y}` 的 token 多重集相同，
  CPC 全局均值最大差 2.98e-8，但 RHIC 的 `branch_q` 差 0.0728、
  `branch_input` 差 0.1210、根 `delta` 差 0.00250。证明“同样的 CPC 均值、不同的分支划分”
  会被 RHIC 区分。

### B. 一次性传播

hooks 计数：`atom_encoder=1, bond_encoder=1, local=1, drop=1, rhic=1, f=1, rho=1,
readout=1, post_local=1`。RHIC 收到的正是 `drop(local(...))` 与 bond-encoder 输出；
`post_local` 收到的正是 `h0+delta`；远端端点原子的 `h0` 改变会改变根 `delta`
（路径读取），但不存在“更新后的原子状态被再次读取”的路径。

### C. 梯度与预算

- 10 warm-up + 100 measured 个正式计划 batch。
- step 1：零初始化 `root2` 使门关闭，只有 `root2.weight/bias` 有非零梯度；
  step 3/10/100：**24/24 个新张量梯度全部非零**。
- 60.37 ms/step（p50）、68.05 ms（p90），峰值显存 253 MB；
  40-epoch 投影 **1.9227 GPU-h**（目标 ≤3.0，硬顶 ≤4.0）。

### D. Test 隔离

训练脚本与机制脚本都断言 Test 标签在封存前不是有限值；checkpoint 记录
`test_evaluated: false`。

### E. 数值一致性

见 §2.5；epoch-0 恒等与 CPU↔CUDA 一致性均通过。

### Train 结构覆盖率（冻结 Train 结构）

| 统计 | 值 |
|---|---:|
| Train 原子数 | 9,001,316 |
| 平均分支数 / 根 | 1.982（p95 3，max 4） |
| 平均路径数 / 根 | 3.006（p95 5，max 11） |
| 根有 ≥2 个分支 | 0.7814 |
| 根有 ≥2 条路径 | 0.9574 |
| 根至少有一个多路径分支 | **0.7767** |
| 根有 ≥2 个多路径分支 | 0.1793 |
| 分支中 `n_b≥2` 的比例 | **0.4881**（8,708,273 / 17,842,385） |
| 分子至少有一个多路径分支 | **0.9994** |
| 分子有 ≥2 个多路径分支 | 0.9714 |

结论：结构机会充分存在（几乎每个分子都有多路径分支，约一半分支 `n_b≥2`），
因此后面的负结果不能归因于“结构没有被枚举到”。

---

## 5. 正式训练结果

- 状态 `COMPLETED`，40/40 epochs，109,520 steps，无 resume，无早停。
- best epoch = **25**，best Valid AP = **0.268734**；epoch 40 Valid AP = **0.266492**。
- Train AP@40 = **0.605333**，Train BCE@40 = **0.023889**，Valid BCE@40 = **0.046720**。
- GPU 时间 **1.9181 h**（目标 3.0 h，硬顶 4.0 h），峰值显存 261 MB。
- 14 次梯度裁剪事件；FP32；dropout 0.10。

曲线要点（每 epoch 的官方 Validation）：

| epoch | Valid AP | R = RMS(δ)/RMS(h0) | gate mean | Train BCE |
|---:|---:|---:|---:|---:|
| 1 | 0.095876 | 0.056 | 0.527 | 0.0617 |
| 5 | 0.217466 | 0.352 | 0.550 | 0.0378 |
| 10 | 0.249062 | 0.369 | 0.580 | 0.0333 |
| 15 | 0.266059 | 0.440 | 0.618 | 0.0308 |
| 20 | 0.267904 | 0.507 | 0.655 | 0.0290 |
| **25** | **0.268734** | 0.554 | 0.672 | 0.0275 |
| 30 | 0.265671 | 0.616 | 0.696 | 0.0263 |
| 35 | 0.266738 | 0.648 | 0.708 | 0.0256 |
| 40 | 0.266492 | 0.663 | 0.712 | 0.0252 |

`R` 单调上升、门控持续打开，而 Valid 在 epoch 25 后不再提升：
这是“残差容量继续增大、泛化不再改善”的典型形态。

---

## 6. 主结果表

### 6.1 五个模型的完整对比

| Model | Params | Best ep | Valid best | Valid @40 | Test best-Valid | Test @40 | Train AP@40 | Train BCE | Valid BCE | GPU h |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| W2 | 994,324 | 24 | 0.265273 | 0.261867 | 0.253617 | 0.250986 | 0.581875 | 0.025215 | 0.046538 | 1.3949 |
| RC-REAL | 1,051,022 | 33 | 0.267718 | 0.266950 | 0.255941 | 0.256167 | 0.604281 | 0.023895 | 0.047169 | 1.9186 |
| LCRC-CPC | 1,043,333 | 36 | 0.265944 | 0.263993 | 0.262356 | 0.261492 | 0.592958 | 0.024554 | 0.046690 | 1.7620 |
| W2-DYNAMIC | 1,128,468 | 25 | 0.275067 | 0.268791 | 0.262226 | 0.260136 | 0.654260 | 0.022108 | 0.048017 | 1.7340 |
| **LCRC-RHIC** | **1,117,253** | **25** | **0.268734** | **0.266492** | **0.258416** | **0.254685** | 0.605333 | 0.023889 | 0.046720 | 1.9181 |

所有 Test 数字由同一段代码在同一份官方标签上重算：OGB 官方
`Evaluator("ogbg-molpcba")` 与独立 sklearn per-task AP 的差异为 **5.6e-17**
（即完全一致）；127 个任务可打分，task 45 不可打分。

### 6.2 127-task macro、106-task high-support、22-task low-support

| Model | 127-task Test macro | 106 HS Valid | 106 HS Test | 22 LS Valid | 22 LS Test |
|---|---:|---:|---:|---:|---:|
| W2 | 0.253617 | 0.289269 | 0.275816 | — | — |
| RC-REAL | 0.255941 | 0.293230 | 0.277303 | — | — |
| LCRC-CPC | 0.262356 | 0.291600 | 0.280599 | 0.136443 | 0.170274 |
| W2-DYNAMIC | 0.262226 | 0.300639 | 0.286440 | — | — |
| **LCRC-RHIC** | **0.258416** | **0.295093** | **0.279189** | **0.135686** | **0.153562** |

说明：
- 22 个低支撑任务中 **21 个**在 Test 上可打分（task 45 无正例，不可打分）；
  它们只作为诊断集，**不**被当作“可打分任务全集的替代”。
- RHIC 相对 CPC：HS Valid **+0.003493**（67/106 改善），HS Test **−0.001410**
  （48/106 改善、58/106 变差）；低支撑 Test 从 0.170274 降到 0.153562。
- 全 127 任务逐任务比较：RHIC vs CPC 为 57 升 / 70 降，逐任务中位数 −0.000672。

### 6.3 逐任务 delta 的分布（127 个可打分任务）

| Test 正例数 | 任务数 | RHIC−CPC 平均 ΔAP | RHIC 更好的任务数 |
|---|---:|---:|---:|
| 20–49 | 12 | −0.00440 | 8 |
| 50–99 | 22 | −0.00227 | 9 |
| 100–199 | 19 | +0.00144 | 11 |
| ≥200 | 55 | −0.00260 | 21 |

最大回退：task 8（−0.1164，Test 0.196 vs CPC 0.312）、task 68（−0.0836）、
task 43（−0.0804）、task 3（−0.0504）、task 21（−0.0427）。
最大改善：task 109（+0.0716）、task 119（+0.0652）、task 104（+0.0564）、
task 106（+0.0532）、task 86（+0.0439）。

---

## 7. 机制诊断（仅 Validation，冻结 best-Valid checkpoint）

三种 eval-mode 前向，全部在官方 Validation（43,793 行）上：

| 干预 | Valid AP | Valid BCE | 与 real 的平均 |Δlogit| | 逐任务 real 更好 / 更差 |
|---|---:|---:|---:|---:|
| `real` | **0.268739** | **0.045669** | — | — |
| `delta_zero`（`h=h0`） | 0.218122 | 0.049582 | 1.2566 | 119 / 8 |
| `branch_assignment_shuffle`（根内分支分配置换，seed 0） | 0.259601 | 0.046327 | 0.4900 | 102 / 25 |

- `real − delta_zero = +0.050618`；`real − shuffle = +0.009139`。
- 打乱保持每个根的路径 token 多重集、根归属、路径数与各分支大小，
  因此 `+0.009139` 只能来自“哪些 token 被分到同一个分支”。
- 全 Validation 聚合使用量：`R=0.5442`，`gate_mean=0.6687 ± 0.1895`，
  活跃分支比例 0.9343，多路径分支比例 0.4747，`q_b` 非零比例 0.4747，
  `cross_branch` 非零比例 0.8282，根有 ≥2 分支比例 0.8283，
  根有多路径分支比例 0.7664。
- 机制门 8 项全部通过（残差非零、尺度合理、分支活跃、多路径分支存在、
  `q_b` 非零、`cross_branch` 非零、`delta` 改变预测、分组改变预测），
  因此 `mechanism_valid = true`，标签不是 `MECHANISM_INVALID`。

注意解释边界：`delta_zero` 与 shuffle 都是对“联合训练后的网络”的分布外干预，
它们度量的是**冻结预测对已学分支结构的依赖**，不是一个独立的因果泛化增益。

---

## 8. 七个问题的回答

### Q1. RHIC 是否在官方 Validation / Test 上超过 W2、RC-REAL、CPC、W2-DYNAMIC？

**OBSERVED**：
- Valid：RHIC 超过 W2（+0.00346）、RC-REAL（+0.00102）、CPC（+0.00279）；
  **没有**超过 W2-DYNAMIC（−0.00633）。
- Test：RHIC 超过 W2（+0.00480）、RC-REAL（+0.00248）；
  **没有**超过 CPC（−0.00394）或 W2-DYNAMIC（−0.00381）。
- 参数 1,117,253，介于 CPC 与 W2-DYNAMIC 之间。

**INTERPRETATION**：在两个单跳/平坦基线（W2、RC-REAL）之上有稳定的小幅提升；
但一旦对手也使用路径结构（CPC）或迭代传播（DYNAMIC），RHIC 在 Test 上落后。
“分支层级”并没有比“平坦路径”带来更好的迁移。

### Q2. 是否达到预注册目标（Valid 0.280，Test 0.270）？

**OBSERVED**：Valid 0.268734（差 0.011266），Test 0.258416（差 0.011584）。未达到。
**INTERPRETATION**：差距不是舍入或评测噪声级别；即使考虑单 seed 的不确定性，
0.011 也不是本轮可以合理归因于“运气”的量级。

### Q3. 层级分支组合相对平坦 CPC 是否带来增量？

**OBSERVED**：Valid +0.00279（HS Valid +0.00349，67/106 改善）；
Test −0.00394（HS Test −0.00141，48/106 改善）。
**INTERPRETATION**：在拟合/Validation 侧有小的正增量，在 scaffold-shift 的 Test 侧
没有；净效应为负。若只看 Valid 会误判为“有效”，Test 结果否定了这一判断。

### Q4. 若有增益，是广泛的还是集中的？

**OBSERVED**：Valid 侧较广（67/106 高支撑任务改善，HS 平均 +0.0035）；
Test 侧不是整体上升，而是重分配：48 升 / 58 降，中位数 −0.00067，
少数任务大幅下降（task 8、68、43），同时少数任务大幅上升（task 109、119、104）。
**INTERPRETATION**：不是“普遍更好”，而是“在某些分支统计模式上更好、在另一些上更差”。
这种重分配在 single-seed 下无法证明净收益。

### Q5. 机制诊断显示了什么？

**OBSERVED**：`real 0.268739` ≫ `delta_zero 0.218122`（+0.050618），
`real` > `shuffle 0.259601`（+0.009139）；`q_b`、`cross_branch` 大量非零；
`R=0.544`；所有机制门通过。
**INTERPRETATION**：模型确实依赖 RHIC 残差，而且依赖“具体的分支分组”，
不是仅仅依赖 token 的多重集。机制是“活的”，但其增益没有迁移到 Test。

### Q6. 成本（参数、GPU、BCE/AP 权衡）与 one-shot 约束？

**OBSERVED**：+122,929 参数（相对 W2）、+73,920（相对 CPC）、−11,215（相对 DYNAMIC）；
1.9181 GPU-h（<3.0 目标，<4.0 硬顶）；Train AP 0.605333、Train BCE 0.023889、
Valid BCE 0.046720；与 RC-REAL 的拟合指标几乎相同（0.604281 / 0.023895），
Train AP 高于 CPC（0.592958），低于 DYNAMIC（0.654260）。
one-shot 审计：每个模块恰好一次调用，RHIC 不读取 `h`，`post_local` 只调用一次。
**INTERPRETATION**：计算成本完全在预算内；它的“容量”主要体现在 Train 拟合，
而不是 Test 迁移；Valid BCE 与 CPC 基本相同，说明概率校准没有实质改善。

### Q7. 决策标签是什么？对“结构充分性”假设意味着什么？

**DECISION**：`NO_CLEAR_ADVANCE`。
**INTERPRETATION**：
- 对“在 one-shot 约束下，进一步层级化/二阶共现化路径结构能提升泛化”这一具体主张，
  本轮给出的是**不支持**的证据：Valid 的小幅提升以 Test 的回退为代价，
  且相对 CPC 的净 Test 变化为 −0.0039。
- 这**不**等于“结构无用”：RHIC 仍优于 W2/RC-REAL，且机制诊断显示
  分支分组被实际使用。更准确的结论是：**在官方 scaffold split 上，
  二阶分支共现带来的额外表达能力主要拟合了不可迁移的分支特异统计量**；
  继续沿这条路线增加层级/共现阶数，预期不会自动转化为 Test 增益。
- 与 DYNAMIC 的对比（Valid −0.0063，Test −0.0038）说明：
  迭代传播带来的增益并不能由“一次性、更复杂的路径聚合”替代。

---

## 9. 决策与含义

按预注册规则得到 `NO_CLEAR_ADVANCE`，理由已在 §0 列出。需要强调：

1. 这不是“没有跑完”或“机制失效”。训练 40/40 完成，机制门全部通过，
   Test 只读一次，best-Valid 与 @40 两个 checkpoint 都已封存并打分。
2. 这也不是预注册意义上的 `REGRESSION`：Test 确实低于 CPC 0.0039，
   但 Valid 高于 CPC 0.0028，规则要求两侧同时低于 CPC−0.003 才判回归。
3. 因而正确的读法是“**Valid/Test 方向相反的小效应**”：
   在 106 高支撑集上 Valid +0.0035、Test −0.0014；
   在全 127 任务上 Valid +0.0028、Test −0.0039。
   以 single seed 的精度，这些都不足以支撑“前进”的结论。

对后续轮次的启示（不属于本轮预注册结论，仅作为研究判断）：

- 若继续 one-shot 路线，应优先约束二阶项的**迁移性**（例如对 `q_b` 做跨 scaffold
  的归一化/去相关、或限制其有效秩），而不是继续增大层级或参数量。
- 若目标是接近 DYNAMIC，证据更支持“允许有限次传播”，而不是“把一次性聚合做得更复杂”。
- 任何后续尝试都应保留本轮的两个“护栏”：Valid-only 选择 + 封存后单次 Test；
  以及机制诊断（delta-zero / 分组 shuffle），否则无法区分“模块被使用”与“模块有增益”。

---

## 10. 偏差、限制与复现

### 偏差（均在预注册容差内）

1. epoch-0 Valid AP：GPU 0.029063055 vs 历史未训练 W2 0.029063332（Δ2.77e-7，
   容差 1e-4；CPU/GPU 浮点差异）。
2. Validation replay：freeze best 0.268748297 vs 记录 0.268734186（Δ1.41e-5）；
   epoch40 0.266496537 vs 0.266491960（Δ4.58e-6）；机制 real 0.268739448
   （Δ5.26e-6）。均 ≤1e-4。
3. 运行记录的 `git_state` 含 6 个 `??` 未跟踪文件（早期失败探针留下的
   `analysis/*_failed*.json`、`gpu_profile.json` 等）；**没有 tracked 文件被修改**，
   训练代码为 commit `1b9bfa5d4484`。freeze manifest 的 `git_state` 同样只列出
   刚写出、尚未提交的 `test_selection.json`（`??`），代码为 commit `c4208ab1d37a`。
4. GPU 执行域：`res` 单机 `a100-2`，pool `res-gpu1`，driver 550.163.01，
   torch 2.5.1+cu124；所有 GPU 时间来自同一 regime。

### 限制

- **单 seed**：本轮只训练 seed 0，没有 seed 方差或置信区间；
  Valid +0.0028 / Test −0.0039 这样的量级不能单独证明总体效应。
- **Test 是已暴露的工程比较集**：不是盲泛化估计，也从不用于选择。
- **`delta_zero` / shuffle 是分布外干预**：它们说明“冻结预测依赖什么”，
  不构成“去掉模块后重新训练”的因果消融。
- **task 45 不可打分**；127 是 Test 可打分任务数，22 个低支撑任务不是“可打分总体”。

### 复现锚点（SHA-256）

| 对象 | SHA-256 |
|---|---|
| best-Valid checkpoint | `0d1b40d35bc540ded85d6caf620aa5f761ada4f0f8a2114d125172a479322ede` |
| epoch-40 checkpoint | `e434e6699c66ee69cfe0d116c26c6cff93b47cdba8a8ad726ae242c5a2c4a3af` |
| last checkpoint | `92155cb66f109ea8908841456af020aaabcc7cd78c9a7732901d0d93ef73f203` |
| train curve | `c73a98a2dd2770b348f9a5d250bcc0094967b9e1e072b7f387fa56d0f49fe65b` |
| branch usage | `54bdfd1bfb81f276777e7c00886123b8c126c59271aab870e66c4e920b750ffb` |
| Test freeze seal | `7848561815f57ffe64eb105e1db821bbf0aa8b8949a1ab1fe7766efcbdba7e04` |
| RHIC Test logits (best) | `ccba1ad514043c9637303da540ddd1431bc74f0e93efe889cc0529a2ff386968` |
| RHIC Test logits (@40) | `49e71511f3bd156b5fa578890bad8693a1d810a97a28f4d5c2757f9a62781913` |
| 官方 Test 标签文件 | `ae9430cf3c2c7697c0318ef1a9f32d6536e4ade48e7e09a81da9ade9fb7bb7a2` |
| Test rows | `9171aa54ed4070dcbd6697b68d57560c7015c118168de4fa63b5e05a02c3cd62` |

产物：`analysis/structure.json`、`analysis/coverage.json`、`analysis/witness.json`、
`analysis/preflight.json`、`analysis/mechanism.json`、`analysis/usage_full_valid.json`、
`analysis/test_prediction_manifest.json`、`analysis/test_scores.json`、
`analysis/decision.json`、`analysis/per_task.json`；
运行产物在 `runs/LCRC_RHIC/`（git-ignored，已从 GPU 主机拉回）。

---

## 11. 结语

RHIC 是一次“机制成立、但增益不迁移”的干净负结果：
两级分支共现确实被模型使用，确实把 Validation 推高到 0.2687，
但没有跨过 CPC 的 Test 水平，更没有接近 0.270 / 0.280 的目标。
在官方 scaffold split 与单 seed 的证据下，本轮的结论是
**`NO_CLEAR_ADVANCE`**：层级化的一次性路径聚合不是通往 W2-DYNAMIC 增益的有效替代，
且其额外容量有明确的“拟合更好、迁移更差”的代价。
