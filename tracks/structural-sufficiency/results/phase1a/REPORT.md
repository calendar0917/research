# SSRC Phase 1a report — same-information \(E_2\) S vs X screening

状态: **COMPLETE（R1 + R2 两个完整 replicate）**  
协议: `ssrc-phase1a-lipo-v1`  
科学冻结 commit: `f73334a`（corrigendum + protocol + code + split）  
训练代码 commit: `5465171`（仅增加 Fit/Monitor-only 标签工件与加载路径；模型/配置未变）  
运行: `res`，2×A100-SXM4-40GB；R1 2026-10-18 21:59–22:25，R2 22:40–23:07 (+08:00)  
决策: **`NO_CLEAR_INTERACTION_VALUE_IN_THIS_PILOT`**  
方向限定: **`S_AT_LEAST_AS_GOOD`；`NO_STABLE_SMALL_N_X_ADVANTAGE`**

> 一句话：在精确相同的 \(E_2\) patch 多重集下，跨 patch self-attention（X）
> 没有表现出优于共享局部编码 + 池化（S）的价值。n=2352 时 S 在 **6/6**
> 个 draw（2 replicates × 3 seeds）更好；n=253 时 S 在 **5/6** 个 draw
> 更好，但没有稳定的 X 优势。小样本交互点估计为正，但 6 个 draw 中只有
> 3 个为正，且两个 replicate 的 CI 都跨零，不能作为样本效率收益。

---

## 1. 这个 pilot 能回答与不能回答什么

**能回答（在本任务/本 split/本配方内）**

- 给定完全相同的 \(E_2\) patch 输入、相同局部编码器与回归头、匹配参数预算，
  加两层全局 self-attention 是否改善 Monitor 上的有限样本泛化？
- 这种改善是否在 n=253 比 n=2352 更大？
- 与普通 full-graph GINE 参照相比，S/X 的应用性能大概在什么位置？
- 同一冻结配方在 clean worktree 重跑时，结果有多稳定？

**不能回答**

- \(E_2\) 是否任务信息充分（\(P(Y|G)=P(Y|E_2)\)、Bayes 风险是否相同）。
  S/X 都只是 \(E_2\) 的函数；S 表现好不等于 \(E_2\) 充分。
- “消息传递是否必要”的一般结论；本轮没有 \(S_{\rm bag}\)、\(S(G)\) 或五臂
  oracle 分解。
- 跨 patch 交互在更强正则化/更长训练/不同 attention 配方下是否会有价值；
  本轮只用一个冻结配方，不做调参。
- 对其它数据集、其它任务、其它 scaffold split 的泛化。

---

## 2. 信息输入一致性

| 模型 | 输入 | 跨 patch 计算 | 参数 | forward FLOPs（CPU-gate 固定 64 分子 batch，下界） |
|---|---|---|---:|---:|
| **S** | 每个原子的精确 radius-2 rooted patch；patch 内原始原子/键属性 + root + 距离 | 无；patch 内 2 层 GINE + 逐 token 残差 FFN + mean pooling | 59,011 | 622.3M |
| **X** | 与 S **逐 tensor 相同**的 patch 张量/mask | 相同局部 GINE/头 + 2 层全局 self-attention（无位置编码） | 59,075 | 877.7M |
| **G** | 原始分子图 + `edge_index` + 原始属性 | full-graph 2 层 GINE + mean pooling | 53,379 | 170.5M |

- S/X 的 batch 只含 patch-only keys；原图 `edge_index`、patch 间 adjacency、
  root 原子编号均不在输入中。`patch_pos` 只用于 dense 存储，不进入特征，
  也不加位置编码。
- S/X 共享同一 seed 的局部编码器与回归头初始化；CPU gate 逐 tensor 验证相同。
- S/X 参数比 = 59,011/59,075 = **0.9989**。
- CPU-gate 固定 batch 上 X/S = **1.41×**。实际训练 batch 的 run JSON 值为：
  n=253 时 X 944.1M / S 622.3M = **1.52×**；n=2352 时 X 936.6M / S 584.2M =
  **1.60×**（随 batch 组成变化）。这是真实计算差异；若 X 胜出，“更多算力”
  会是未排除的替代解释。**本轮 X 未胜出**，但解释 S 的优势时仍不能把
  “交互”与“更容易优化/更少算力”完全分开。
- \(E_2\) 碰撞门禁：C24/caterpillar/grid 的 d=6 vs d=9 有精确相同的
  \(E_2\) 多重集；S/X 对这些碰撞类的预测差 ≤ 6e-8。两者都不能突破
  \(E_2\) 的信息上限。

---

## 3. 数据、split 与盲性

| 项 | 值 |
|---|---|
| 数据集 | Lipophilicity（MoleculeNet/DeepChem 重分发），4200 行 |
| 官方 split | scaffold 3360/420/420；**只用官方 train 3360 行** |
| 内部 split | Fit/Monitor/Reserved = **2352/504/504**；scaffold 组重叠 0 |
| 小样本 | **n=253**，114 个 scaffold 组（Fit 内组大小 ≤10，整组抽取） |
| Monitor | 504 行 / 421 个 scaffold 组；只用于早停与比较 |
| Reserved | **未读标签、未评分、未进入任何主表** |
| 官方 valid/test | **未读** |
| split sha256 | `9fa96d49efd94cd2ff381922f83a07f963d832064c75469657b4ce5f68c332b6` |
| 标签工件 sha256 | `0260a3909eec30900df646bc760e65360315f4a1398c7092c7c72d04a4e404f7` |

**历史盲性披露（必须保留）**：Phase 0 的 LOO/置换诊断用过全部 3360 个官方
train 标签，因此本轮内部 Reserved 不是“从研究一开始就完全盲”。Phase 1a 的
纪律是**不读取/不评分 Reserved**；这由 `scope` 字段和 CPU gate
`label_isolation` 记录（`reserved_labels_retained=0`）。本轮不是确认性盲测。

---

## 4. 实际执行、replicate 与预算

### 4.1 两个完整 replicate

| Replicate | 目的 | 运行环境 | 结果 |
|---|---|---|---|
| **R1** | 预注册的 18 条筛选轨迹 | 共享 checkout，commit `5465171`；worktree 有**无关**的 untracked lock 文件 | 18/18 exit 0 |
| **R2** | 同设计、同 seeds 的 clean 复现（因 R1 的 `git_dirty=true` 与一次同 seed 重跑差异而做） | `/tmp/ssrc-clean` clean worktree，同一 commit `5465171` | 18/18 exit 0，全部 `git_dirty=false` |

- R1 的 `git_dirty=true` 来自 `tracks/structure-contribution/results/...` 下
  与本 track 无关的 untracked `*.lock` 文件；SSRC 代码 `git diff` 为空，
  Phase 1a 从未 import/读取这些路径。R2 在 clean worktree 重跑，消除该 provenance
  歧义。**R1 没有被丢弃或替换**；两套结果都报告。
- 两个 replicate 的 `git_commit` 都是 `5465171`，`config_sha256` 都是
  `863e82975afbee0122e279f0bd134fe5f6871dabe1e3e2e1af8ec98181b223eb`。
- 每套 18 条 = 3 模型 × 2 n × 3 seeds；**没有**结果驱动的额外 seed、架构或
  超参搜索。R2 是同一 seeds 的重复，不是新 seed。

### 4.2 预算

| 项 | R1 | R2 | 合计 |
|---|---:|---:|---:|
| rr 实际 GPU 占用（started→completed） | 0.619 h | 0.588 h | **1.208 h** |
| 训练循环内 `gpu_seconds` | 0.575 h | 0.545 h | **1.120 h** |
| 预处理 | 26.9 s | 28.3 s | 55.3 s |
| 峰值 GPU 显存 | 90.6 MB | 90.6 MB | — |

- 预算上限 4 GPU-h；两个 replicate 合计约 **1.21 GPU-h**，约 **30%**。
- 低 GPU 占用是因为模型/数据小、单条轨迹短，不改变单 split/3 seeds 的统计限制。

### 4.3 运行间非确定性（必须报告）

一次 preliminary clean 同 seed 检查（S / n=253 / seed=0）得到 Monitor MAE
0.8902，而 R1 同配置为 0.9072；这促成了完整 R2。R2 的同一个 run 为 0.9095。
三个同 seed draw 为 0.8902 / 0.9072 / 0.9095，说明 **GPU 训练存在可观的
run-to-run 非确定性**（scatter/attention 的非确定归约在 200 epoch 内放大）。

R1 vs R2 同 (model, n, seed) 的 Monitor MAE 最大绝对差：

| n | S | X | G |
|---|---:|---:|---:|
| 253 | **0.0400** | 0.0242 | 0.0165 |
| 2352 | 0.0110 | 0.0112 | 0.0191 |

因此：

- **n=2352** 的 S−X 差（pooled −0.0387）明显大于同 seed run-to-run 噪声
  （≤0.011），且 6/6 draws 同向；这是本轮最稳的信号。
- **n=253** 的 S−X 差（pooled −0.0230）与 run-to-run 噪声同量级；
  5/6 draws 同向，但两个 replicate 的 CI 都跨零。只能写“S 不劣于 X /
  未见 X 优势”，不能写精确的小样本效应。
- 3 seeds 不是 3 个完全独立的训练过程；它们是 3 个 seed draw 加上 run 噪声。
  所有 CI 只覆盖 Monitor 抽样，不覆盖 run-to-run 或 train/split 不确定性。

---

## 5. 主结果

主指标：Monitor 上 best-monitor MAE（早停选中），越低越好。  
\(\mathrm{Adv}_{\rm set}(n)=\mathrm{MAE}_S(n)-\mathrm{MAE}_X(n)\)；
**正值 = X 更好，负值 = S 更好**。

### 5.1 R1 逐 seed（预注册那 18 条）

| n | seed | S | X | G | Adv_set (S−X) | Adv_last (S−X) | S−G | X−G |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 253 | 0 | 0.9072 | 0.8930 | 0.9018 | **+0.0142** | −0.0186 | +0.0054 | −0.0088 |
| 253 | 1 | 0.8711 | 0.9258 | 0.9124 | −0.0547 | −0.0438 | −0.0413 | +0.0134 |
| 253 | 2 | 0.8777 | 0.9242 | 0.9229 | −0.0465 | −0.0496 | −0.0451 | +0.0014 |
| 2352 | 0 | 0.5668 | 0.6067 | 0.5942 | −0.0399 | −0.0421 | −0.0275 | +0.0125 |
| 2352 | 1 | 0.5459 | 0.5966 | 0.5753 | −0.0507 | −0.0663 | −0.0295 | +0.0213 |
| 2352 | 2 | 0.5629 | 0.6029 | 0.5835 | −0.0400 | −0.0489 | −0.0206 | +0.0194 |

### 5.2 R2 逐 seed（clean worktree，同设计）

| n | seed | S | X | G | Adv_set (S−X) | Adv_last (S−X) | S−G | X−G |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 253 | 0 | 0.9095 | 0.9171 | 0.8966 | −0.0076 | — | +0.0129 | +0.0205 |
| 253 | 1 | 0.8874 | 0.9258 | 0.9025 | −0.0384 | — | −0.0151 | +0.0233 |
| 253 | 2 | 0.9177 | 0.9228 | 0.9063 | −0.0051 | — | +0.0114 | +0.0165 |
| 2352 | 0 | 0.5697 | 0.5978 | 0.6035 | −0.0281 | — | −0.0338 | −0.0057 |
| 2352 | 1 | 0.5569 | 0.5854 | 0.5752 | −0.0285 | — | −0.0183 | +0.0102 |
| 2352 | 2 | 0.5639 | 0.6091 | 0.5644 | −0.0451 | — | −0.0005 | +0.0447 |

> R2 的 `Adv_last` 未在此表展开；分析 JSON 中有完整值。R2 last-epoch 的
> 方向与 R1 一致（S 在两个 n 都不差于 X）。

### 5.3 汇总（每套 3 seeds；pooled 6 draws）

| n | 模型 | R1 mean | R2 mean | pooled mean [6-draw range] |
|---:|---|---:|---:|---:|
| 253 | S | **0.8853** | **0.9049** | **0.8951** [0.8711, 0.9177] |
| 253 | X | 0.9143 | 0.9219 | 0.9181 [0.8930, 0.9258] |
| 253 | G | 0.9123 | 0.9018 | 0.9071 [0.8966, 0.9229] |
| 2352 | S | **0.5585** | **0.5635** | **0.5610** [0.5459, 0.5697] |
| 2352 | X | 0.6021 | 0.5974 | 0.5997 [0.5854, 0.6091] |
| 2352 | G | 0.5843 | 0.5811 | 0.5827 [0.5644, 0.6035] |

| 量 | R1 | R2 | pooled |
|---|---:|---:|---:|
| Adv_set(253), mean [range] | −0.0290 [−0.0547,+0.0142] | −0.0170 [−0.0384,−0.0051] | **−0.0230** [−0.0547,+0.0142] |
| X-better draws, n=253 | 1/3 | 0/3 | **1/6** |
| Adv_set(2352), mean [range] | −0.0435 [−0.0507,−0.0399] | −0.0339 [−0.0452,−0.0281] | **−0.0387** [−0.0507,−0.0281] |
| X-better draws, n=2352 | 0/3 | 0/3 | **0/6** |
| 交互 Adv(253)−Adv(2352), mean [range] | +0.0145 [−0.0066,+0.0541] | +0.0169 [−0.0099,+0.0401] | **+0.0157** [−0.0099,+0.0541] |
| 交互为正的 draws | 1/3 | 2/3 | **3/6** |

Per-replicate scaffold-cluster bootstrap（Monitor 421 组，5000 次）：

| 量 | R1 95% CI | R2 95% CI |
|---|---:|---:|
| Adv_set(253) | [−0.0618, +0.0040] | [−0.0446, +0.0111] |
| Adv_set(2352) | [−0.0720, −0.0148] | [−0.0596, −0.0062] |
| 交互 | [−0.0295, +0.0569] | [−0.0212, +0.0548] |

- **n=2352**：S 在 6/6 draws 更好；两个 replicate 的 bootstrap CI 都在零以下。
  在单 split/3 seeds/两 replicates 的限制下，这是最清楚的信号。
- **n=253**：S 在 5/6 draws 更好；pooled mean −0.0230，但两个 CI 都跨零，
  且 run-to-run 噪声可达 0.040。不能声称 X 在小样本更好，也不能声称等价。
- **交互**：pooled +0.0157，但只有 3/6 draws 为正，两个 CI 都跨零。
  点估计为正的原因是“S 在小 n 的优势变小”，**不是 X 在小 n 胜出**。
  小样本“X 更好”的假设未得到支持。

### 5.4 常数基线

训练均值预测所有 Monitor 行：n=253 为 0.9553，n=2352 为 0.9652。
所有模型在所有 36 条轨迹上都优于常数基线（S 改善约 0.05–0.42，
X 约 0.03–0.37），所以不是“没学会”的实现失败。

### 5.5 G 参照

- pooled n=2352：S 0.5610，G 0.5827，X 0.5997；S−G = −0.0217。
- pooled n=253：S 0.8951，G 0.9071，X 0.9181；S−G = −0.0120。
- 但 R2 的 n=253 上 G（0.9018）略优于 S（0.9049），R1 反过来；
  G 的位置不稳定。G 只有 53k 参数、170M FLOPs，信息访问不同，
  所以 **不能**把 \(G-S\) 解释为“全图信息没有价值”。本轮没有看到
  full-graph GINE 相对 patch-local S 的稳定优势。

---

## 6. 诊断：为什么 X 没有更好

- **过拟合更大**：R1 n=2352 时 X 的 train MAE 均值 0.227，Monitor 0.602；
  S 的 train 0.285，Monitor 0.559。X 更能拟合训练集，但泛化更差。
  R2 同向（X train 更低、Monitor 更高）。
- **最优 epoch 不固定**：X 的 full-n best epoch 在 R1 为 135/98/80，
  R2 为 95/66/70；S 为 148/139/110 与 115/97/119。X 往往更早停止。
- **成本更高**：X 的固定 batch step time 中位数约 20.9–24.4 ms，
  S 约 14.0–14.3 ms；X 的 actual-batch FLOPs 是 S 的 1.52–1.60×。X 用更多算力却没有
  更好的 Monitor 泛化。
- **不能把原因唯一归为“交互”**：X 同时改变了优化景观、正则化与算力。
  本轮只能说“在这个冻结配方下，加入跨 patch attention 没有带来价值”，
  不能断言“任何跨 patch 交互都无用”，也不能断言“S 的归纳偏置本质上更优”。

---

## 7. 判定

**正式标签：`NO_CLEAR_INTERACTION_VALUE_IN_THIS_PILOT`**  
**方向限定：`S_AT_LEAST_AS_GOOD`；`NO_STABLE_SMALL_N_X_ADVANTAGE`**

理由：

1. 预注册的 `PROMISING_SAME_INFORMATION_INTERACTION` 需要 X 稳定优于 S，
   且小样本交互有实际意义；Full-n 是 0/6 X-better，小 n 只有 1/6 X-better，
   两条都不满足。
2. `UNRESOLVED_INFORMATION_VS_INDUCTIVE_BIAS` 的原条件是 S≈X 且 G 较好；
   G 没有稳定优于 S，且 full-n 上 S 明确优于 X，所以不选它。
3. `IMPLEMENTATION_OR_OPTIMIZATION_FAILURE` 不成立：三个模型都显著优于
   常数基线，训练 loss 明显下降，没有 NaN/梯度爆炸/全零预测。
4. 预注册 `NO_CLEAR_INTERACTION_VALUE_IN_THIS_PILOT` 的字面条件是
   “S≈X≈G”；本轮并非精确相等（full-n S>X，6/6 draws），但其**实质判据**——
   没有观察到跨 patch 交互的正价值——明确满足。这里如实标注这一偏差，
   不把负结果粉饰成“等价”。

**这不是“消息传递/交互普遍无用”的结论。** 它只说明：在这个任务、这个
\(E_2\) 表示、这个 2 层 GINE + 2 层 attention 配方、这个单 split/3 seeds/
两 replicates 的探索性设置下，跨 patch attention 没有提供可检测的收益。

---

## 8. 下一步（只给一个方向）

**方向：在独立 scaffold split（或第二个任务）上原样重复冻结的 S/X 对比，
不做任何架构/超参搜索。**

- 目的：检验 `S ≥ X` 是否是本 split 的偶然，而不是立刻换模型。
- 成功/停止条件：
  - 若独立 split/第二任务上仍然是 **S ≥ X**（尤其 full-n 3/3 seeds 或
    CI 支持），则**停止 cross-patch-attention 这条分支**；下一轮转向
    “信息访问”实验（固定加入一跳/边 incidence 信息，匹配非迭代 vs 迭代），
    而不是继续调 attention。
  - 若 X 在独立 split 上稳定胜出，再检查是否是优化/正则化差异
    （预注册对照），而不是直接宣布“交互有效”。
  - 若 S/X 仍无法区分，保持 `UNRESOLVED`，不扩大模型/数据搜索。
- 不自动做：新架构、更多 seed、更多 n、Reserved/test 评分、ZINC 搜索。

---

## 9. 限制与未完成

- 单 scaffold split、3 seeds、两个 replicates；不是确认性实验。
- scaffold-cluster bootstrap 只覆盖 Monitor 抽样不确定性，**不覆盖**
  训练集/scaffold split 或 run-to-run 不确定性；R2 的 run-to-run 噪声
  已单独报告。
- 没有做等效性检验；CI 跨零只写“未能区分”，不写“等价”。
- X 的 actual-batch FLOPs 是 S 的 1.52–1.60×（CPU-gate 固定 batch 为 1.41×），
  是真实混淆；虽然 X 没赢，但不能据此把 S 的优势
  完全归因于“没有交互”。
- G 是不同信息访问、不同参数量的应用参照，不能作信息/算力因果分解。
- FLOPs 用 `FlopCounterMode`，PyG scatter 可能未计入，是下界。
- 没有读/评分内部 Reserved 或官方 valid/test；因此本轮没有确认性泛化数字。
- 没有做 attention 深度/正则化/位置编码/学习率搜索（协议禁止）；
  因此不能回答“调过的 X 是否会更好”。
- 没有做 per-scaffold 或标签区间的探索性误差分析；避免结果驱动的新分析。
- R1 的 `git_dirty=true` 来自无关 untracked locks；R2 clean worktree 用于
  消除该 provenance 歧义。R2 的 `rr` 元数据指向共享 checkout 的 analysis-only
  commit `65525ba`，但实际训练进程在 `/tmp/ssrc-clean` 的 `5465171` clean
  worktree 中执行；每条 R2 run JSON 的 `environment.git_commit=5465171`、
  `git_dirty=false`。`5465171..65525ba` 只新增
  `analyze_phase1a.py`、REPORT、analysis/manifest，训练/模型/数据代码 diff 为空。

---

## 10. 复现与文件

| 内容 | 路径 |
|---|---|
| 冻结协议 | `notes/phase1a_protocol.md` |
| 纠错文件 | `notes/phase0_corrigendum.md` |
| 配置/hash | `configs/phase1a.yaml` |
| split | `results/phase1a/split_phase1a.json` |
| Fit/Monitor 标签工件 | `results/phase1a/labels_fit_monitor.json` |
| R1 provenance / 分析 | `results/phase1a/run_manifest.tsv`, `analysis_summary.json` |
| R2 provenance / 分析 | `results/phase1a/run_manifest_r2.tsv`, `analysis_summary_r2.json` |
| 两 replicate 汇总 | `results/phase1a/analysis_replicates.json` |
| 逐 run JSON（含 Monitor 预测，git-ignored） | `runs/phase1a/`, `runs/phase1a_r2/` |
| 分析代码 | `code/phase1a/analyze_phase1a.py`, `analyze_phase1a_replicates.py` |
| 训练/模型/数据代码 | `code/phase1a/` |
| CPU gates | `results/phase1a/CPU_GATES.md`（完整 JSON 本地 `cpu_gates.json`） |

```bash
# CPU gates + tests
uv run python tracks/structural-sufficiency/code/phase1a/phase1a_gates.py
uv run pytest -q tracks/structural-sufficiency/code/phase1a/tests

# R1 / R2 / pooled analysis
uv run python tracks/structural-sufficiency/code/phase1a/analyze_phase1a.py \
  --runs tracks/structural-sufficiency/runs/phase1a \
  --out tracks/structural-sufficiency/results/phase1a/analysis_summary.json
uv run python tracks/structural-sufficiency/code/phase1a/analyze_phase1a.py \
  --runs tracks/structural-sufficiency/runs/phase1a_r2 \
  --out tracks/structural-sufficiency/results/phase1a/analysis_summary_r2.json
uv run python tracks/structural-sufficiency/code/phase1a/analyze_phase1a_replicates.py
```

**失败/排除记录**：

- 早期 orchestration 把 `--n` 写成 `full`，导致 9 次 full-n 启动在 argparse
  阶段退出（exit 2，无结果 JSON）；另有一次错误的 worker 参数分组，只产生了
  2 条有效的 S-small 轨迹，随后被正式 R1 run 覆盖。最终 R1 18 条全部 exit 0。
- R2 启动前两次因本地 commit 与 remote commit 不一致被 `rr` 拒绝（无结果），
  加 `--allow-stale` 后在 clean worktree 运行；最终 R2 18 条全部 exit 0。
- 一次 preliminary clean 同 seed 检查（S/253/seed0 = 0.8902）不属于 R1/R2
  的 36 条主表，但作为 run-to-run 非确定性的证据披露；它促成了完整 R2。
  没有用任何失败/额外 draw 做模型选择或 cherry-pick。
