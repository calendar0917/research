# LCRC-CPC v1 — 中文研究报告

**轮次**：`lcrc-pcba-cpc-v1`  
**正式模型**：`LCRC-CPC`，seed 0，40 epochs，唯一一次正式训练  
**状态**：训练、Validation-only 机制诊断、Test 预测封存、单次 Test 评分和预定判读均已完成。  
**Test 角色**：`EXPOSED_TEST_ENGINEERING_COMPARISON_SET`。Test 已暴露，不是 blind 泛化估计，不参与模型选择。

## 首页结论

| 问题 | 结论 | 最强证据 | 反对证据或限制 |
|---|---|---|---|
| 联合条件化是否成立 | **成立（机制层面）** | 非加性配对见证：两组 root/middle/end 边际、路径数和键特征相同，additive reference 差值 **0.0**，真实 CPC `delta` 差 **1.003e-3**、`context` 差 **2.042e-3**；训练后 `delta=0` 使 Validation AP 从 **0.265944** 降到 **0.195371**，配对打乱降到 **0.255172** | 见证只证明 CPC 能响应 `(middle,endpoint)` 配对；不证明历史 W2 在真实分子上必然无法区分，也不是因果性能证明 |
| 是否满足无迭代传播 | **是** | 一次 forward 中 `_encode`/`local`/`drop`/`CPC`/`post_local` 各 1 次；CPC 输入 `h0` 与 `drop(local(...))` 最大差 **7.15e-7**（GPU）；改变远端 endpoint 的 `h0` 直接改变 root `delta`，最大差 **9.30e-3** | 允许且实际使用了一次“路径级跨原子读取”；这不是“完全没有广义 message aggregation” |
| 是否改善 W2 | **分裂：Test 有广泛改善，Validation 无可靠改善** | Validation best **0.265944** vs W2 **0.265273**（**+0.000671**，任务 bootstrap 95% CI 含 0）；Test best **0.262356** vs W2 **0.253617**（**+0.008739**，87/127 任务改善，任务 CI **[+0.001122,+0.017881]**） | Validation 高支持任务 67/106 改善、均值仅 **+0.002331**，CI 含 0；分子 bootstrap 的 Test 差值 CI **[-0.000069,+0.016094]** 也刚跨 0 |
| 恢复了多少 Dynamic 增益 | **Validation 远未达到；Test 点估计达到但高支持任务落后** | `R_valid=0.0685`，`R_test=1.0151`；Test@40 `R=1.1482`；Validation 75% 目标 **0.272620** 未达到 | 高支持 106 任务上 CPC **0.280599** vs Dynamic **0.286440**（**−0.005841**，CI 不含 0）；Test macro 优势由 22 个低支持任务（**+0.028710** vs W2）驱动；不能声称恢复了 Dynamic 的计算能力 |
| 是否值得继续 | **不值得基于本轮结果直接追加第二 seed 或架构扩展** | Validation 主指标无可靠增益；Test 又已暴露，且高支持任务落后 Dynamic | 若研究者继续，应先解决确认性问题：独立/更稳健的划分、预注册高支持主终点；本轮到此停止 |

**预定判读标签：`CPC_PARTIAL_GAIN`。**

---

## 1. 模型与机制

### 1.1 数据流

```python
a, b, scal = base._encode(batch)                  # 一次
h0 = base.drop(base.local(batch, a, b, scal))     # local/drop 各一次

delta = CPC(h0, b, path_index)                    # 一次，只读同一 h0/b
h_cpc = h0 + delta
logits = base.post_local(h_cpc, batch)            # 继承 W2 readout，一次
```

`h0`、`delta`、`h_cpc` 为 `[N_atoms,128]`，`b` 为 W2 bond encoder 输出 `[N_bonds,192]`。CPC 不读取 `h_cpc`，也不把更新后的 middle/endpoint 写回后再传播。

### 1.2 有根两键路径

对每个 root `i`，枚举 `j ∈ N(i)`、`k ∈ N(j)\{i}`，保留所有合法 `(i,j,k)`，包括三角环、不同 middle 到同一 endpoint 的重数，以及 `(i,j,k)`/`(k,j,i)` 两个方向。没有采样、截断、最短路去重、virtual node 或人工远距离边。

路径缓存从冻结 PCBA S2 缓存构造，只追加：

`cpc_ptr, cpc_middle, cpc_end, cpc_bond_ij, cpc_bond_jk`

以及每分子的 `row_atom_ptr/row_bond_ptr` 和每行计数。全量缓存实际为：

| 项 | 数值 |
|---|---:|
| 分子行数 | 437,929 |
| 原子数 | 11,373,137 |
| CPC 路径数 | 34,369,144 |
| 缓存大小 | 658,409,864 bytes（约 0.658 GB） |
| 枚举 schema | `rooted-two-bond-simple-path-v1` |
| 缓存行 ID sha256 | `85b8d38b10293ddd5f431c6feae3aa9c22ec56f19a3a2367c4da2c170d9c1642` |
| 源 cache sha256 | `caad3e90…` |
| feature spec hash | `7270f767…` |
| split partition sha256 | `c872bd63…` |
| Train/Valid label artifact sha256 | `5c3b5fbe…` |

逐 root 的 CPC 路径数、endpoint 多重集合、两条 bond ID 的真实连接性、三原子同分子性均与冻结 S2 逐条核对通过；没有 `i==k` 回退 walk。

### 1.3 参数与初始化

| 模块 | 参数 |
|---|---:|
| 共享 LayerNorm(128) | 256 |
| 三个 Linear(128,32) | 12,384 |
| Linear(192,16) bond projection | 3,088 |
| Path Linear(192,96) | 18,528 |
| Path Linear(96,64) | 6,208 |
| Gate Linear(96,1) | 97 |
| Output Linear(65,128) | 8,448 |
| **CPC 新增** | **49,009** |
| W2 原始 | 994,324 |
| **总计** | **1,043,333** |

W2 部分与历史 seed-0 构造器逐 tensor bitwise 相同（trunk fingerprint `5f7101a8…`）。CPC 使用私有 RNG `seed*1_000_003+1001`，不推进 W2 公共 RNG；`Output Linear` 的 weight/bias 初始为零。因此 epoch 0 的 `CPC logits == 历史未训练 W2 logits`：CPU 最大差 **0.0**，GPU 最大差 **1.91e-6**（容差 1e-4）。CPU/CUDA logits 最大差 **3.81e-6**。

### 1.4 机制门槛

训练前的 CPU/GPU preflight 全部通过：

* 小图枚举、全量缓存、原子重编号、键方向反转、跨分子隔离、零路径 `delta=0`；
* 非加性配对见证：additive reference 差 **0.0**，CPC `delta` 差 **1.003e-3**；
* 单次调用计数：`atom_encoder=1, bond_encoder=1, local=1, drop=1, cpc=1, f=1, rho=1, readout=1`；`post_local=1`；
* CPC 输入 `h0` 与 `drop(local(...))` 最大差 **7.15e-7**；
* 远端 endpoint 改变引起 root `delta` 最大差 **9.30e-3**；
* 新参数梯度：step 1 只有 output 层有效（零输出初始化所致），step 3、10、100 时 **18/18** 个新 tensor 均获得非零梯度；
* 100 个真实 batch profile：step p50 **58.7 ms**，40 epoch 投影 **1.861 GPU-h**。

## 2. 唯一正式训练

| 项 | 值 |
|---|---|
| 数据 | `ogbg-molpcba`，Train 350,343 / Valid 43,793 / Test 43,793，128 tasks |
| batch / epochs | 128 / 40（109,520 updates） |
| 优化器 | AdamW，weight decay 1e-3，betas (0.9,0.999)，eps 1e-8 |
| LR | 1 epoch warmup 3e-5→3e-4，随后 per-step cosine 3e-4→1e-5 |
| dropout / clip / precision | 0.10 / 5.0 / FP32 |
| loss | masked BCE（NaN target 不进入） |
| batch order | 历史 W2 seed-0 40-epoch plan，`global_plan_sha256=06f43729…`；prefix `1a24dfca…` |
| 选择规则 | 官方 Validation AP argmax；并列取最早 epoch |
| 结果 | best epoch **36**，best Valid **0.265944328**；epoch 40 Valid **0.263993025** |
| Train AP @40 | **0.592958021**；Train/Valid gap **0.328965** |
| checkpoint | best sha `7957f2aa…`；epoch40 sha `031d2fa6…` |
| 成本 | 内部 **1.7620 GPU-h**；含 preflight/freeze/mechanism 分配共 **1.8447 GPU-h**（目标 2.8，硬上限 3.5） |
| 显存峰值 | 236,069,888 B allocated / 501,219,328 B reserved |

正式 run `status=COMPLETED`，40/40 个 epoch 均完成官方 Validation AP 评估；`test_evaluated=false`。运行 commit 为 `4864a6ed…`；代码树 clean（`environment.git_dirty=false`）。`git_state.dirty=true` 只来自远端两个与本轮无关的 S3 未跟踪审计文件。

## 3. 性能

### 3.1 主表

| 模型 | 参数量 | Best Valid AP | Valid@40 | Best-Valid Test AP | Test@40 | GPU-h |
|---|---:|---:|---:|---:|---:|---:|
| W2 | 994,324 | 0.265273316 | 0.261866918 | 0.253616863 | 0.250985846 | 1.3949 |
| RC-REAL | 1,051,022 | 0.267717702 | 0.266950286 | 0.255940917 | 0.256167166 | 1.9186 |
| W2-DYNAMIC | 1,128,468 | 0.275067186 | 0.268791116 | 0.262226049 | 0.260136315 | 1.7340 |
| **LCRC-CPC** | **1,043,333** | **0.265944328** | **0.263993025** | **0.262356011** | **0.261492308** | **1.7620** |

### 3.2 相对 Dynamic 的恢复比例

`R = (AP_CPC − AP_W2)/(AP_DYNAMIC − AP_W2)`，使用历史 best checkpoint；不根据结果重定义 75%。

| 划分/checkpoint | CPC | W2 | Dynamic | R |
|---|---:|---:|---:|---:|
| Validation best | 0.265944328 | 0.265273316 | 0.275067186 | **0.0685** |
| Validation @40 | 0.263993025 | 0.261866918 | 0.268791116 | **0.3071** |
| Exposed Test best | 0.262356011 | 0.253616863 | 0.262226049 | **1.0151** |
| Exposed Test @40 | 0.261492308 | 0.250985846 | 0.260136315 | **1.1482** |

Validation 75% 门槛 **0.272620** 未达到；Test 75% 门槛 **0.260070** 在点估计上达到。Test 点估计略高于 Dynamic（+0.000130），但高支持任务均值显著低于 Dynamic，且 Test 是暴露集。

### 3.3 高支持与低支持分解

共同高支持集：Validation 和 Test 均至少有 20 个正例的 106 个任务。Test 和 Validation 分别报告。

| 分层（Test, best） | N | CPC | W2 | Dynamic | CPC−W2 | 改善/退化 vs W2 | CPC−Dynamic |
|---|---:|---:|---:|---:|---:|---:|---:|
| 高支持 | 106 | 0.280598722 | 0.275816057 | 0.286440162 | **+0.004782665** | **72 / 34** | **−0.005841440** |
| 低支持 | 22 | 0.170273756 | 0.141563792 | 0.140002432 | **+0.028709964** | 15 / 6 | +0.030271324 |
| 全部已评分任务 | 127 | 0.262356011 | 0.253616863 | 0.262226049 | **+0.008739148** | 87 / 40 | +0.000129962 |

| 分层（Validation, best） | N | CPC | W2 | Dynamic | CPC−W2 | 改善/退化 vs W2 | CPC−Dynamic |
|---|---:|---:|---:|---:|---:|---:|---:|
| 高支持 | 106 | 0.291600244 | 0.289268942 | 0.300638511 | **+0.002331302** | **67 / 39** | **−0.009038267** |
| 低支持 | 22 | 0.136443042 | 0.144152539 | 0.145992878 | −0.007709497 | 8 / 13 | −0.009549836 |
| 全部已评分任务 | 127 | 0.265944328 | 0.265273316 | 0.275067186 | **+0.000671012** | 75 / 52 | **−0.009122857** |

Test 的 macro 改善来自低支持任务的大幅提升；高支持任务只比 W2 高 **+0.004783**，且仍比 Dynamic 低 **−0.005841**。Validation 高支持任务只有 67/106 改善，均值和 CI 都支持不了“广泛改善”的结论。

### 3.4 描述性不确定性

不是显著性检验。任务 bootstrap 对 20,000 次任务重采样；分子 bootstrap 对 300 次 Test 分子重采样。

| 比较（best） | Δ | 95% 区间 | 正差异比例 |
|---|---:|---:|---:|
| Test 全部任务：CPC−W2 | +0.008739 | [+0.001122,+0.017881] | 0.9885 |
| Test 全部任务：CPC−Dynamic | +0.000130 | [−0.009967,+0.011742] | 0.4860 |
| Test 高支持：CPC−W2 | +0.004783 | [+0.000340,+0.009134] | 0.9820 |
| Test 高支持：CPC−Dynamic | −0.005841 | [−0.010381,−0.001148] | 0.0069 |
| Validation 全部任务：CPC−W2 | +0.000671 | [−0.005565,+0.006958] | 0.5796 |
| Validation 高支持：CPC−W2 | +0.002331 | [−0.002628,+0.007314] | 0.8231 |
| Validation 高支持：CPC−Dynamic | −0.009038 | [−0.015681,−0.003028] | 0.0015 |

分子 bootstrap 的 Test 全部任务 CPC−W2 为 **+0.007972**，95% 区间 **[−0.000069,+0.016094]**；Dynamic−W2 为 **+0.008656**，区间 **[+0.000807,+0.016563]**。任务 bootstrap 忽略任务内标签噪声，分子 bootstrap 包含它，因此前者更窄；两者都不支持“已证明显著优于 W2”。

## 4. 分支是否真正参与预测

固定 32 行 Validation 诊断 batch，每个 epoch 记录：

| epoch | R = RMS(delta)/RMS(h0) | gate mean | gate std | 每 root gate std 均值 | RMS(z) | RMS(context) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0908 | 0.6379 | 0.1765 | 0.0384 | 0.4331 | 0.2666 |
| 10 | 0.3773 | 0.6501 | 0.2882 | 0.1294 | 0.7249 | 0.2812 |
| 20 | 0.4480 | 0.6725 | 0.2951 | 0.1322 | 0.7915 | 0.3396 |
| 30 | 0.5266 | 0.7047 | 0.3033 | 0.1382 | 0.8876 | 0.4158 |
| 36 | 0.5337 | 0.7151 | 0.3009 | 0.1364 | 0.9039 | 0.4321 |
| 40 | 0.5398 | 0.7182 | 0.3001 | 0.1361 | 0.9098 | 0.4392 |

全量 Validation 上，best checkpoint 的 `R=0.490205`，`RMS(delta)=0.533487`，`RMS(h0)=1.088295`；gate mean **0.7212**、std **0.2980**、p1 **0.0027**、p99 **0.9979**；每原子平均 **3.0732** 条路径，zero-path 原子仅 **61/1,185,291 = 5.15e-5**。分支数值上明显活跃，不是“参数存在但输出接近零”。

18 个新 tensor 在 step 3/10/100 全部有非零梯度；到 epoch 36，各 CPC 模块梯度范数仍在 `3.2e-3`（bond_proj）到 `2.5e-2`（path1）量级，权重范数持续增长。没有关键梯度长期消失。

## 5. 冻结 checkpoint 的 Validation-only 干预

只使用 best-Valid checkpoint，不重训、不重选、不调阈值。

| 干预 | Valid AP | Valid BCE | 相对 real 的 logit RMS 变化 |
|---|---:|---:|---:|
| real CPC | **0.265944176** | 0.046584398 | — |
| `delta=0` | **0.195371183** | 0.052053991 | 20.22% |
| 同 root 内打乱 middle–endpoint 配对 | **0.255172222** | 0.047169266 | 8.28% |

`delta=0` 使 AP 下降 **0.070573**，配对打乱下降 **0.010772**；两者都改变了 99.99% 以上的 logit 项。打乱后路径没有跨分子，root/路径数/键多重集固定；第二个 bond ID 与新的 endpoint 不保证化学连接，这是刻意的 OOD 配对干预。

这些下降说明网络确实依赖训练后的 CPC 路径，但**不能**解释为“CPC 相对 W2 的独立因果收益”，因为主干和 readout 是联合训练的，`delta=0` 是分布外输入。

## 6. 必须回答的五个问题

### Q1. CPC 是否真正实现了已学习语义的联合条件化？

**Observed**：是，机制层面成立。证据不是模块名，而是：

1. CPC 的真实输入是 W2 `drop(local(...))` 的同一 `h0`（GPU 最大差 7.15e-7）和同一 `b`；
2. 非加性见证中两组路径的 root/middle/endpoint 边际、路径数、bond 特征完全相同，additive reference 差 **0.0**，而真实 CPC 的 `delta` 差 **1.003e-3**、`context` 差 **2.042e-3**；
3. token 显式包含 `r[i]*d[k]` 与 `m[j]*d[k]`，不是纯加和；
4. 训练后全部 18 个新 tensor 有梯度，全量 Validation `R=0.490`，gate 非退化；
5. `delta=0` 与配对打乱都实质改变预测。

**Interpretation**：CPC 确实在做“已学习原子表示之间的具体路径配对组合”，而不是仅给每个原子加一个局部标量。

**限制**：见证是固定 `h0` 的合成实验；它不证明真实分子上 W2 无法编码相同信息，也不证明这种机制必然带来更好的泛化。

### Q2. CPC 是否确实没有迭代隐藏状态传播？

**Observed**：是。真实 forward hook/计数显示 `_encode`、`local`、`drop`、`CPC`、`f`、`rho`、readout、`post_local` 各 **1** 次；CPC 的 forward 只发生 **1** 次；`post_local` 只读取一次 `h_cpc`。改变一条合成路径的远端 endpoint `h0_k`，root `delta_i` 直接改变 **9.30e-3**，但没有任何 `h_cpc` 再次进入 CPC。

**Interpretation**：这是一次路径级跨原子读取加一次 root 聚合；不是两层 MP，也不是 `h1=CPC(h0); h2=CPC(h1)`。

**限制**：不能声称“完全没有广义 message aggregation”；本轮允许并实际使用了路径上的跨原子 `h0` 读取。

### Q3. CPC 相比 W2 是否有广泛收益？

**Observed**：

* Test macro：**+0.008739**，87/127 改善，任务 bootstrap CI [+0.001122,+0.017881]；
* Test 高支持：**+0.004783**，72/106 改善，CI [+0.000340,+0.009134]；
* Test 低支持：**+0.028710**，15/22 改善，是 macro 提升的主要来源；
* Validation macro：**+0.000671**，75/127 改善，CI 含 0；
* Validation 高支持：**+0.002331**，67/106 改善，CI 含 0；
* Validation 低支持：**−0.007709**，8/22 改善。

**Interpretation**：在已暴露 Test 上，CPC 相对 W2 有可观察的广泛改善；但在正式用于选择的 Validation 上，增益很小且不能与 0 区分。Test 的增益不是所有分层一致：高支持只小幅改善，低支持大幅改善。

**限制**：单个 seed；Test 已暴露；不能仅凭 1–2 个稀疏 assay 宣称成功，也不能把 Test 的方向当作独立泛化证据。

### Q4. CPC 恢复了多少 Dynamic 的收益？

**Observed**：

* `R_valid = 0.0685`（best），`R_valid@40 = 0.3071`；Validation 未达到 75% 目标；
* `R_test = 1.0151`（best），`R_test@40 = 1.1482`；Test 点估计达到/略超过 Dynamic；
* 参数量 1,043,333，比 Dynamic 少 **85,135**，比 W2 多 **49,009**；
* GPU 成本 1.7620 h，比 Dynamic 多 **0.0280 h**；
* best epoch 36（Dynamic 25）；CPC best 比 @40 高 0.001951（Validation）、0.000864（Test）；
* 高支持 Test：CPC−Dynamic **−0.005841**，CI 不含 0；Validation **−0.009038**，CI 不含 0；
* Validation→Test 下降：CPC **0.003588**，Dynamic **0.012841**，W2 **0.011656**。

**Interpretation**：Test 的 macro 点估计与 Dynamic 几乎相同，但这是由低支持任务和 Test 暴露/分布差异驱动的；高支持任务上 CPC 显著低于 Dynamic，Validation 上则远未达到 Dynamic。因此不能声称恢复了两层迭代动态传播的能力。

**限制**：Test 是工程比较集；`R_test > 1` 不能升级为“匹配 Dynamic”的强结论。高支持均值差距与任务 bootstrap 都反对这一强结论。

### Q5. 这是否改变无迭代 MP 研究方向的判断？

**Observed**：一个固定的、非迭代的有根两键路径联合条件化模块可以被训练、可以产生非加性配对响应，并在已暴露 Test 上超过 W2；但它在 Validation 上没有可靠增益，在高支持任务上没有追上 Dynamic。

**Interpretation**：结果支持“一次性路径级联合计算可以学到有用的跨原子交互”，但不支持“它可以替代迭代 hidden-state propagation”，也不支持“所有非迭代方法都无法达到 Dynamic”。它更像是：在这个固定配方和单 seed 下，CPC 学到的是有用的、但与 Dynamic 不同且迁移不稳定的信号。

**Decision**：**不值得基于本轮结果直接追加第二 seed、扩大宽度/路径长度或改 attention。** 如果研究者要继续，最低成本的下一步不是再训练一个变体，而是先建立独立确认：使用未暴露的评估划分（或嵌套 scaffold 验证），预注册高支持任务主终点，并把 Test 仅作为工程描述。本轮在唯一正式训练、冻结评分和报告后停止。

## 7. 事件与偏差

1. **GPU preflight 的三次工程失败**：`_permute_atoms_with_paths` 在 CUDA batch 上调用了 CPU permutation helper；随后修正为 CPU 副本 + 回迁，并把 GPU 数值不变量容差从精确相等放宽到 `1e-5`/`1e-6`。另修正了一个 CPU/GPU model 设备混用和远端 sensitivity 的设备放置。全部是审计/测试代码修复，模型公式、参数、路径和训练配方未变。
2. **W2 参考 logits 哈希拼写错误**：`cpc_common` 中 W2_best_valid 哈希有一个字符换位（`eaf2` vs `aef2`）。在 Test freeze 前用 BOC manifest 和实际 `.npy` 重新核验并修正；没有数据或预测被替换。
3. **`cc_data` 导入顺序**：首次 freeze 因 `cpc_test.py` 在 `cpc_common` 设置路径前导入 `cc_data` 而失败；只调整导入顺序。
4. **Test 高支持集定义错误（已记录）**：首次 `score` 在封存后读取 Test 标签并计算了内存中的数组，但在“高支持集应为 106”的断言处失败，**没有写出 `test_scores.json`**。原因是临时实现用了 Test + Dynamic 正例数（108），而协议要求 Validation + Test 均 ≥20（106）。修正为读取 Validation best-epoch 正例数后重跑；模型、checkpoint、阈值、选择规则和 Test 预测均未改变。最终 `test_scores.json` 来自修正后的单次写出。
5. **远端 `git_state` 脏标记**：正式 run 的 `git_state.dirty=true` 来自两个与本轮无关的 S3 未跟踪分析文件；代码 commit 为 `4864a6ed…`，`environment.git_dirty=false`。正式训练没有使用未提交代码。
6. **成本**：100-batch profile 投影 1.861 GPU-h；实际内部 1.7620 GPU-h；含所有 GPU pool 分配的 ledger 为 1.8447 GPU-h，均低于 2.8 目标。结构缓存 0.658 GB，低于 1 GiB 目标。
7. **未做**：没有第二 seed、第二架构、STATIC/MP 臂、路径采样、Test 驱动调参或 checkpoint 重选。

## 8. 可复现性与工件

关键哈希/身份：

* W2 参考：best Valid `0.265273316`，Test `0.253616863`；
* Dynamic 参考：best Valid `0.275067186`，Test `0.262226049`；
* LCRC-CPC best checkpoint sha256 `7957f2aa4bc524950960f4560cce0c3cacc46090aaf770d85305d018d71cf27a`；
* LCRC-CPC epoch40 checkpoint sha256 `031d2fa636da490bb47ab66a1933ce7b0cab3898dec55bf22c636832940ac453`；
* Test row IDs sha256 `9171aa54ed4070dcbd6697b68d57560c7015c118168de4fa63b5e05a02c3cd62`；
* Test freeze seal `72c8adeee6c0a90a56feefd218e90e933375ed743c6ef3b7fb9eea39df7d7493`；
* best Test logits sha256 `9dbf4e8ff997d1fd922313828086598744122d7eae794669b146822de983dfda`；
* epoch40 Test logits sha256 `581f9220b23f96038d56cd9e45b7d76e7a553cd6c618e4910b834a00eb098fb1`。

工件：

`analysis/structure_cache.json`、`preflight.json`、`gpu_profile.json`、`mechanism.json`、`test_prediction_manifest.json`、`test_scores.json`、`per_task_analysis.json`、`decision.json`、`uncertainty.json`、`gpu_ledger.json`、`report_tables.md`、`training_curve.csv`、`branch_usage.json`、`per_task_validation.csv`、`selected_checkpoint.json`。

`runs/` 下的 checkpoint 和 658 MB 路径缓存按仓库惯例被 `.gitignore` 忽略；训练曲线、分支使用、checkpoint 选择和 Test 冻结/评分工件已复制或写入 `analysis/`。
