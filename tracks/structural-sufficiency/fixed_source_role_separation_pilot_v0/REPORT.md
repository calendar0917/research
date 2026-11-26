# FSR v0 — 固定来源结构计算的中心—邻域角色分离（短程验证报告）

Round ID：`fixed-source-role-separation-pilot-v0`；协议 `fsr-v0`
数据：`ogbg-molpcba` 官方 scaffold split（Train 350343 / Valid 43793）；seed 0；**Test 未运行**
历史参照：`direct_structural_computation_recurrence_v0/results/runs/FIXED1_seed0`（真实 seed-0 FIXED1 曲线，25 epoch，已完成）。只用其 **12-epoch 前缀**，从不使用 25-epoch best（0.275847）。

## 结论（先回答五个问题）

| 问题 | 回答 |
|---|---|
| 1. RSSC 是否超过 TIED？ | **是。** 同 12 epoch、同 order plan、同更新数：best Valid AP `0.267029 vs 0.260582`，**+0.006447**；endpoint `+0.007614`（task bootstrap 95% 区间 `[+0.00212, +0.01363]`）；高支持任务 `+0.00617`（best）/`+0.00306`（endpoint）。 |
| 2. RSSC 是否超过同 horizon FIXED1？ | **否。** FIXED1 的 12-epoch 前缀 best = `0.269992`，RSSC `0.267029`，**−0.002963**（高支持 `−0.00049`，73 任务改善 / 54 退化，bootstrap 区间跨 0）。TIED 同样低于 FIXED1（**−0.009410**，区间 `[−0.01616,−0.00290]`）。 |
| 3. 改善来自训练拟合还是 Validation 泛化？ | 相对 TIED：两者都有，但拟合增幅远大于泛化（best Train AP `+0.0381`，best Valid AP `+0.0064`）。相对 FIXED1：**只有训练拟合更高**（Train AP `0.45465 vs 0.44792`，`+0.0067`），Valid AP 反而低 `0.0030`。 |
| 4. 新分支是否实际参与计算？ | **强烈参与。** RSSC `D=AB` 的 `‖D‖F/‖W‖F≈0.494`，修正项/fc1 项 RMS 比 ≈0.34–0.36；冻结置零该分支后 Valid AP 从 `0.2680` 掉到 `0.2351`（−0.0320，112/127 任务退化，logits 几乎全部改变）。TIED 分支较窄（R≈0.233，比值 ≈0.105–0.109）。 |
| 5. 是否值得继续研究角色分离？ | **本轮不建议继续。** 角色分离确实优于参数绑定（+0.0064，endpoint 区间不含 0），但**没有带来超过原始 FIXED1 的性能推进**（−0.0030；最后三 epoch 均值 −0.0007）。按预注册规则判定 `ROLE_DIFFERENCE_WITHOUT_PERFORMANCE_ADVANCE`。 |

**正式判定**：`ROLE_DIFFERENCE_WITHOUT_PERFORMANCE_ADVANCE`（无 `BUDGET_LIMITED` / `REFERENCE_MISSING` / `HORIZON_TRUNCATED`；两臂均完整 12 epoch）。

## 1. 实现与继承

- 完全继承 FIXED1：`h0 → LN1 → 邻域 SiLU 聚合 → h1 → (LN2(h1), LN2(h0)) → 固定来源聚合 → h2 → post_local`。atom/bond encoder、S1/S2、dropout、`f`、`rho`、四头 attention readout、残差、归一化、维度一律未改；未新增预测头；未把 `h1[src]`/`h2[src]` 作为邻居来源。
- 唯一变化：两个 block 的**非线性前**线性项各加一条独立低秩分支
  `a = fc1(z+m) + A(B(x))`，`x = z+m`（TIED）或 `z−m`（RSSC），rank = 32，B 为继承 fan-in 均匀初始化（私有 generator 20261126），A 全零；两个 block 各自独立。
- 真实参数清点（两臂相同）：`1,153,044 = 1,128,468 + 24,576`；新增 `2 × (128×32 + 32×256)`。继承部分与历史 FIXED1 初始状态**逐位一致**（W2 trunk 指纹 `5f7101a8…`、new 指纹 `f7c24ee4…`、71 个继承 state tensor）。
- 两臂初始参数逐位一致（max abs diff = 0）。

## 2. 正确性检查（训练前）

`analysis/checks.json`，**15/15 通过**（full 模式，真实缓存 + 真实 batch）：

- B0 数据身份：feature spec / norm / split / label artifact 与历史审计值逐项相等，行数 350343/43793/43793，Test 标签在训练 artifact 中全为 NaN。
- B1 参数量；B2 两臂初值逐位一致 + 继承初值等于历史 FIXED1；B3 `A=0` 时两臂在合成图与真实 Train/Valid batch 上**精确**恢复 FIXED1 初始前向（logits/h1/h2/m2 max abs diff = 0）。
- B4 非零 A（并临时给 fc2 小随机权重使分支可观测）：`z+m` 相同、`z−m` 不同时 TIED 的 pre-activation/输出在 fp32 舍入内不变（≤1e-6），RSSC 明显变化（>1e-3）。
- B5 RSSC 双形式等价（float64）：`W(z+m)+A(B(z−m)) = (W+D)z + (W−D)m`，bias 只加一次，相对误差 ~1e-15。
- B6 第二层邻域聚合只依赖 `LN2(h0)`：扰动 `h1` 不改变 `m2`（逐位），扰动 `h0` 改变 `m2`，且 `m2 == Σ_j SiLU(LN2(h0)[j]+e_ji)`。
- B7 原子重编号、边序、键方向反转不变（fp32 散射累加误差 ≤2.9e-6）。
- B8 A/B 梯度可达：首步 A/B 梯度为 0（继承的 `fc2=0` 残差初始化所致），第 2 步 A 梯度非零、第 3 步 B 梯度非零；显式非零 A 时 B 梯度立即可达。
- B9 空邻域（单原子、无边）：`m2=0`、有限输出。
- B10 训练/机制代码路径不读取 Test（静态检查），本轮没有任何 Test 评价。

## 3. 训练与预算

- 正式 horizon：preflight（真实 batch assembly + GPU step + 全量 Valid/Train 评价）在 `res-gpu1` 实测 step 28.1 ms、Valid 6.6 s、Train 38.1 s，投影两臂 12 epoch = **24.2 min/臂** → 选择 **12 epochs**（未压缩 cosine，使用 40-epoch schedule 前缀）。
- 两臂并行、相同 batch 序列与更新数：AdamW、wd 1e-3、clip 5、dropout 0.10、masked BCE、首 epoch warmup `3e-5→3e-4`、seed-0 冻结 plan（`06f437…b6f6`）。
- 每 epoch 全量 Valid AP/BCE；epoch 4/8/12 全量 Train AP/BCE；结束时补评 best checkpoint 的 Train 指标。主 checkpoint = horizon 内最高 Valid AP，平局取更早 epoch。
- **GPU 时间**：TIED `0.7178` GPU-h，RSSC `0.7152` GPU-h，合计 **1.433 GPU-h**；另加 preflight/并发探测/完整检查/finalize 约 `0.07 GPU-h`，总计约 **1.50 GPU-h**（软上限 1.6，未超）。
- **墙钟**：T0 ≈ 01:19，训练 01:52–02:36（两臂并行，每臂约 43 min；两进程共享单卡使每 epoch 约 180–190 s，高于单进程投影的 137 s，但 soft wall 未触发），结果分析与报告至约 02:50。总计约 90 min（触及硬上限，见偏差）。

## 4. 结果

`analysis/main_table.csv`、`analysis/epoch_curves.csv`（含 FIXED1 历史前缀，`source=historical_seed0_run`）：

| arm | best epoch | best Valid AP | endpoint Valid AP | last-3 mean AP | best Train AP | best Train BCE | Valid BCE | params | GPU-h |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TIED | 10 | 0.260582 | 0.259415 | 0.259558 | 0.416502 | 0.031035 | 0.045285 | 1,153,044 | 0.718 |
| RSSC | 12 | **0.267029** | **0.267029** | **0.262524** | 0.454651 | 0.029436 | 0.045210 | 1,153,044 | 0.715 |
| FIXED1（历史 12-epoch 前缀） | 12 | 0.269992 | 0.269992 | 0.263256 | 0.447919 | 0.029790 | 0.045382 | 1,128,468 | — |

任务级（127 个官方可评分任务，`analysis/paired_comparisons.json`）：

| 比较 | 平均 ΔAP | 高支持（正负例≥20） | 改善/退化 | task bootstrap 95% | 去掉 \|Δ\| 最大 3 任务 |
|---|---:|---:|---:|---:|---:|
| RSSC − TIED（best） | **+0.006447** | +0.006168 | 82 / 44 | [−0.00713, +0.01747] | +0.008918 |
| RSSC − TIED（endpoint） | +0.007614 | +0.003058 | 77 / 50 | [+0.00212, +0.01363] | +0.003889 |
| RSSC − FIXED1（best/endpoint） | −0.002963 | −0.000491 | 73 / 54 | [−0.01583, +0.00705] | +0.001920 |
| TIED − FIXED1（best） | −0.009410 | −0.006661 | 54 / 72 | [−0.01616, −0.00290] | −0.007815 |

曲线层面的稳健性：最后 3 epoch 均值 RSSC `0.262524` > TIED `0.259558`（+0.00297），RSSC 略低于 FIXED1（`0.263256`，−0.00073）。最后 5 epoch 均值 RSSC `0.258607` > TIED `0.254644`。bootstrap 只表示当前固定训练结果的任务不确定性，不是 seed 方差（本轮仅 seed 0）。

## 5. 轻量机制诊断（`analysis/mechanism.json`）

以各臂 best checkpoint、真实 Valid batch 记录：

| 量 | TIED (best ep10) | RSSC (best ep12) |
|---|---:|---:|
| `‖D‖F`（block0 / block1） | 3.310 / 3.313 | 7.285 / 7.330 |
| `‖W‖F` | 14.23 / 14.25 | 14.73 / 14.81 |
| `R = ‖D‖F/‖W‖F` | 0.233 / 0.233 | 0.494 / 0.495 |
| `cos(D, W)` | +0.30 / +0.31 | **−0.14 / −0.13** |
| RMS `fc1(z+m)`（block1） | 3.49 | 3.40 |
| RMS 低秩修正（block1） | 0.379 | 1.211 |
| 修正/fc1 比（block1） | 0.109 | 0.356 |
| RMS `z` / `m`（block1） | 1.075 / 1.379 | 1.090 / 1.320 |
| A/B 梯度范数（真实 Valid 反传） | 0.0040–0.0045 / 0.0033–0.0038 | 0.0071–0.0072 / 0.0049–0.0055 |

- RSSC 学到的 `D` 与 `W` 近似正交甚至轻微负相关，而 TIED 的 `D` 与 `W` 同向；RSSC 的修正项尺度约为 TIED 的 3 倍，说明解除绑定后分支承担了明显不同的方向。
- **冻结干预（仅 RSSC best checkpoint）**：将新增分支置零（A=0，等价 D=0，其余全部不变），Valid AP `0.26703 → 0.23505`（**−0.03198**），112/127 任务退化，15 个改善，logits 平均绝对变化 0.746、RMS 0.989，99.9999% 的预测项发生 >1e-6 变化。**该分支是主要计算路径之一**；但这只是“模型是否使用分支”的检查，不代表训练该分支能提升泛化。
- 3 个真实 Valid 分子（row 23/26/27，均 29 原子）中心/邻域非线性前分解：RSSC block1 中心项 RMS 1.48–1.55、邻域项 2.91–3.10（邻域占主导）；TIED 为 1.46–1.53 / 2.52–2.69；两种写法的重构相对误差 ≤2.2e-7（fp32 舍入级），即角色分离在数值上精确成立。
- 未做结构词、化学解释、规则搜索或归因分析；未做 SREN 解释流程。

## 6. 判读（预注册阈值）

- `ROLE_SEPARATION_PROMISING` 需要 RSSC 相对 TIED **且** 同 horizon FIXED1 均 ≥ +0.003 AP。实际：相对 TIED `+0.0064`（满足），相对 FIXED1 `−0.0030`（不满足）。
- `GENERIC_PARAMETERIZATION_SIGNAL` 需要两臂均 ≥ +0.003 于 FIXED1；实际 TIED `−0.0094`、RSSC `−0.0030`（不满足）。
- `ROLE_DIFFERENCE_WITHOUT_PERFORMANCE_ADVANCE`：RSSC 优于 TIED（≥0.003）但未超过 FIXED1 → **命中**。
- 高支持任务与后期曲线无冲突（RSSC 高支持 +0.006；last-3 高于 TIED）。因此不是“少数低支持任务”造成的差异；但整体仍低于未改动的 FIXED1。

## 7. 预算、偏差与未完成事项

- 训练单元：2 个（TIED、RSSC），均完整 12 epoch；无多 seed、无蒸馏/集成/预训练/辅助监督；未运行 Test；未自动启动下一轮。
- **偏差 1（算力调度）**：`res-gpu0` 被其他用户的进程长期占用（`wujiayu`, 20.5 GB, 84–97% util），未抢占；两臂改为**并发运行于 `res-gpu1`**（各自 CUDA_VISIBLE_DEVICES 相同，进程级并发）。并发探测显示 step 时间几乎不变（28.9/29.9 ms vs 单进程 28.1 ms），但每 epoch 实际 180–190 s（高于 preflight 投影），仍直接决定 12 epochs 而非降级 8。
- **偏差 2（RSSC 运行记录修复）**：RSSC 的 12 个 epoch、曲线、best/last checkpoint、best-checkpoint 评价均由原训练进程正常产出；随后在**训练结束后的机制诊断**中因 numpy/torch 混用崩溃（`ndarray.abs`）。用 `code/frs_finalize.py` 从已保存的 best checkpoint 与现有 artifact 重算 `mechanism.json` 并重建 `run.json`（`repair_note` 已写入；崩溃现场保留为 `run.failed.json`）。训练本身未重训，结果与曲线未受影响。
- **偏差 3**：TIED 未做冻结置零干预（协议只要求 RSSC）；其分支参与度由 D/修正尺度与梯度给出。
- 未完成/未做（均为本轮范围外）：seed 方差、Test 评价、解释规律发现、rank/架构搜索、更长训练。历史同 horizon 参照完整可用，因此没有 `REFERENCE_MISSING`；两臂均完成 12 epoch，因此没有 `BUDGET_LIMITED`。
- 产物：`PROTOCOL.md`、`code/`（模型、训练、检查、机制、分析、manifest、finalize）、`analysis/{checks,preflight_timing,main_table,epoch_curves,paired_comparisons,mechanism,summary}.json|csv`、`results/manifest.json`（best/last checkpoint 路径 + SHA256，共 55.8 MB，留在 `runs/`）。
