# SSRC Phase 1a report — same-information \(E_2\) S vs X screening

状态: **COMPLETE**  
协议: `ssrc-phase1a-lipo-v1`  
科学冻结 commit: `f73334a`（corrigendum + protocol + code + split）  
运行 commit: `5465171`（仅增加 Fit/Monitor-only 标签工件与加载路径；模型/配置未变）  
运行时间: 2026-10-18 21:59–22:25 (+08:00)，`res`，2×A100-SXM4-40GB  
决策: **`NO_CLEAR_INTERACTION_VALUE_IN_THIS_PILOT`**  
方向限定: **`S_AT_LEAST_AS_GOOD`；`NO_STABLE_SMALL_N_X_ADVANTAGE`**

> 一句话：在精确相同的 \(E_2\) patch 多重集下，跨 patch self-attention（X）没有
> 表现出优于共享局部编码 + 池化（S）的价值；在 n=2352 上 S 在 3/3 seeds
> 更好，在小 n 上 S 平均也更好但没有稳定优于 X。小样本“X 相对更好”的
> 交互点估计为正，但由单个 seed 驱动，CI 跨零，不能作为样本效率收益。

---

## 1. 这个 pilot 能回答与不能回答什么

**能回答（在本任务/本 split/本配方内）**

- 给定完全相同的 \(E_2\) patch 输入、相同局部编码器与回归头、匹配参数预算，
  加两层全局 self-attention 是否改善 Monitor 上的有限样本泛化？
- 这种改善是否在 n=253 比 n=2352 更大？
- 与一个普通 full-graph GINE 参照相比，S/X 的应用性能大概在什么位置？

**不能回答**

- \(E_2\) 是否任务信息充分（\(P(Y|G)=P(Y|E_2)\)、Bayes 风险是否相同）。
  S 和 X 都只是 \(E_2\) 的函数；S 表现好不等于 \(E_2\) 充分。
- “消息传递是否必要”的一般结论；本轮没有 \(S_{\rm bag}\)、\(S(G)\) 或五臂
  oracle 分解。
- 跨 patch 交互在更强正则化/更长训练/不同 attention 配方下是否会有价值；
  本轮按协议只用一个冻结配方，不做调参。
- 对其它数据集、其它任务、其它 scaffold split 的泛化。

---

## 2. 信息输入一致性

| 模型 | 输入 | 跨 patch 计算 | 参数 | forward FLOPs（固定 64 分子 batch，下界） |
|---|---|---|---:|---:|
| **S** | 每个原子的精确 radius-2 rooted patch；patch 内原始原子/键属性 + root + 距离 | 无；patch 内 2 层 GINE + 逐 token 残差 FFN + mean pooling | 59,011 | 622.3M |
| **X** | 与 S **逐 tensor 相同**的 patch 张量/mask | 相同局部 GINE/头 + 2 层全局 self-attention（无位置编码） | 59,075 | 877.7M |
| **G** | 原始分子图 + `edge_index` + 原始属性 | full-graph 2 层 GINE + mean pooling | 53,379 | 170.5M |

- S/X 的 batch 只含 patch-only keys；原图 `edge_index`、patch 间 adjacency、
  root 原子编号均不在输入中。`patch_pos` 只用于 dense 存储，不进入特征，
  也不加位置编码。
- S/X 共享同一 seed 的局部编码器与回归头初始化；CPU gate 逐 tensor 验证相同。
- S/X 参数比 = 59,011/59,075 = **0.9989**。
- X 的 forward FLOPs 是 S 的 **1.41×**。这是真实计算差异；若 X 胜出，
  “更多算力”会是未排除的替代解释。**本轮 X 未胜出**，所以这不是对 X 的
  有利辩解；但解释 S 的优势时仍不能把“交互”与“更容易优化/更少算力”完全分开。
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

## 4. 实际执行与预算

- 运行矩阵：3 模型 × 2 n（253/2352）× 3 seeds = **18 条轨迹**；
  **没有**结果驱动的额外 seed、架构或超参搜索。
- 18/18 条最终 run `exit_code=0`；每条 JSON 的 `git_commit` 均为
  `54651711af787757ab6866097d512c08caaad289`，`config_sha256` 均为
  `863e82975afbee0122e279f0bd134fe5f6871dabe1e3e2e1af8ec98181b223eb`。
- **实际 GPU 占用（rr started→completed，含启动/预处理）**：**0.619 GPU-h**
  （2229 s）。
- **训练循环内 `gpu_seconds` 合计**：**0.575 GPU-h**（2071 s）。
- 预处理合计：26.9 s；峰值 GPU 显存：**90.6 MB**（X, n=253）。
- 预算上限 4 GPU-h；实际用量约 **15%**。
- 两条早期 orchestration 尝试因 CLI 参数拼写（`full` 而非 `2352`）失败，
  未产生结果 JSON，已排除；最终 18 条全部成功。详见
  `run_manifest.tsv` 与下文“失败/排除”。

> 低 GPU 占用不是因为“训练很轻”，而是模型小、数据小、单条轨迹短；
> 这不改变单 split/3 seeds 的统计限制。

---

## 5. 主结果

主指标：Monitor 上 best-monitor MAE（早停选中），越低越好。  
\(\mathrm{Adv}_{\rm set}(n)=\mathrm{MAE}_S(n)-\mathrm{MAE}_X(n)\)；
**正值 = X 更好，负值 = S 更好**。

### 5.1 逐 seed

| n | seed | S | X | G | Adv_set (S−X) | Adv_last (S−X) | S−G | X−G |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 253 | 0 | 0.9072 | 0.8930 | 0.9018 | **+0.0142** | −0.0186 | +0.0054 | −0.0088 |
| 253 | 1 | 0.8711 | 0.9258 | 0.9124 | −0.0547 | −0.0438 | −0.0413 | +0.0134 |
| 253 | 2 | 0.8777 | 0.9242 | 0.9229 | −0.0465 | −0.0496 | −0.0451 | +0.0014 |
| 2352 | 0 | 0.5668 | 0.6067 | 0.5942 | −0.0399 | −0.0421 | −0.0275 | +0.0125 |
| 2352 | 1 | 0.5459 | 0.5966 | 0.5753 | −0.0507 | −0.0663 | −0.0295 | +0.0213 |
| 2352 | 2 | 0.5629 | 0.6029 | 0.5835 | −0.0400 | −0.0489 | −0.0206 | +0.0194 |

常数基线（训练均值预测所有 Monitor 行）：n=253 为 0.9553，
n=2352 为 0.9652。所有模型在所有 seed 上都优于常数基线
（S−const −0.048 到 −0.419；X−const −0.029 到 −0.369），所以不是
“没学会”的实现失败。

### 5.2 汇总

| n | 模型 | Monitor MAE mean [range] | 相对常数基线 |
|---:|---|---:|---:|
| 253 | S | **0.8853** [0.8711, 0.9072] | −0.070 |
| 253 | X | 0.9143 [0.8930, 0.9258] | −0.041 |
| 253 | G | 0.9123 [0.9018, 0.9229] | −0.043 |
| 2352 | S | **0.5585** [0.5459, 0.5668] | −0.407 |
| 2352 | X | 0.6021 [0.5966, 0.6067] | −0.363 |
| 2352 | G | 0.5843 [0.5753, 0.5942] | −0.381 |

| 量 | mean | 3-seed range | 方向一致的 seed 数 | scaffold-cluster bootstrap 95% CI |
|---|---:|---:|---:|---:|
| Adv_set(253) | **−0.0290** | [−0.0547, +0.0142] | 1/3 正（即 2/3 S 更好） | [−0.0618, +0.0040] |
| Adv_set(2352) | **−0.0435** | [−0.0507, −0.0399] | 0/3 正（3/3 S 更好） | [−0.0720, −0.0148] |
| 交互 Adv(253)−Adv(2352) | **+0.0145** | [−0.0066, +0.0541] | 1/3 正 | [−0.0295, +0.0569] |
| Adv_last(253) | −0.0373 | [−0.0496, −0.0186] | 0/3 正 | — |
| Adv_last(2352) | −0.0525 | [−0.0663, −0.0421] | 0/3 正 | — |

- **n=2352**：S 在 3/3 seeds 优于 X，平均 0.0435 MAE；bootstrap CI 在零以下。
  在单 split/3 seeds 的限制下，这是本 pilot 中最清楚的信号。
- **n=253**：S 平均优于 X 0.0290，但 seed 0 方向相反；CI 跨零。
  不能声称 X 在小样本更好，也不能声称两者等价。
- **交互**：点估计 +0.0145 是“X 在小 n 相对更好”的方向，但只有 seed 0
  为正（+0.0541），seed 1/2 为负；CI 跨零。**小样本优势假设未得到稳定支持。**
- 用 last-epoch 指标替代 best-monitor 后，结论不变（S 在两个 n 都更好，
  full-n 差异更大）。所以主结果不是早停乐观偏差造成的。

### 5.3 G 参照

- n=2352：S−G = **−0.0258**（S 更好），CI [−0.0621, +0.0081]；
  X−G = +0.0177（X 更差），CI [−0.0199, +0.0538]。
- n=253：S−G = **−0.0270**，CI [−0.0591, +0.0059]；
  X−G = +0.0022，CI [−0.0214, +0.0255]。
- 点估计上 S 在两个 n 都优于普通 full-graph GINE G，但 CI 跨零；
  G 只有 53k 参数、170M FLOPs，且信息访问不同，所以**不能**把
  \(G-S\) 解释为“全图信息没有价值”。这只是一个应用参照：
  本轮没有看到 full-graph GINE 相对 patch-local S 的明显优势。

---

## 6. 诊断：为什么 X 没有更好

- **过拟合更大**：n=2352 时 X 的 train MAE 均值 0.227，Monitor 0.602；
  S 的 train 0.285，Monitor 0.559。X 更能拟合训练集，但泛化更差。
  n=253 也有类似但更嘈杂的模式（X train 0.682 vs S 0.573）。
- **最优 epoch 不固定**：X 在 full-n 的 best epoch 为 135/98/80，
  S 为 148/139/110；X 往往更早停止，说明 Monitor 更早开始变差。
- **成本更高**：X 的固定 batch step time 中位数约 20.9–24.4 ms，
  S 约 14.0–14.3 ms；X 的 FLOPs 是 S 的 1.41×。X 用更多算力却没有
  更好的 Monitor 泛化。
- **不能把原因唯一归为“交互”**：X 同时改变了优化景观、正则化与算力。
  本轮只能说“在这个冻结配方下，加入跨 patch attention 没有带来价值”，
  不能断言“任何跨 patch 交互都无用”，也不能断言“S 的归纳偏置本质上更优”。

---

## 7. 判定

**正式标签：`NO_CLEAR_INTERACTION_VALUE_IN_THIS_PILOT`**  
**方向限定：`S_AT_LEAST_AS_GOOD`；`NO_STABLE_SMALL_N_X_ADVANTAGE`**

理由：

1. 预注册的 `PROMISING_SAME_INFORMATION_INTERACTION` 需要 X 在 3/3 seeds
   稳定优于 S，且小样本交互有实际意义；两条都不满足。
2. `UNRESOLVED_INFORMATION_VS_INDUCTIVE_BIAS` 的原条件是 S≈X 且 G 较好；
   本轮 G 没有明显优于 S，且 full-n 上 S 明确优于 X，所以不选它。
3. `IMPLEMENTATION_OR_OPTIMIZATION_FAILURE` 不成立：三个模型都显著优于
   常数基线，训练 loss 明显下降，没有 NaN/梯度爆炸/全零预测。
4. 预注册 `NO_CLEAR_INTERACTION_VALUE_IN_THIS_PILOT` 的字面条件是
   “S≈X≈G”；本轮并非精确相等（full-n S>X），但其**实质判据**——
   没有观察到跨 patch 交互的正价值——明确满足。这里如实标注这一偏差，
   不把负结果粉饰成“等价”。

**这不是“消息传递/交互普遍无用”的结论。** 它只说明：在这个任务、这个
\(E_2\) 表示、这个 2 层 GINE + 2 层 attention 配方、这个单 split/3 seeds
的探索性设置下，跨 patch attention 没有提供可检测的收益。

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
    （例如冻结步数、权重衰减、dropout 的预注册对照），而不是直接宣布
    “交互有效”。
  - 若 S/X 仍无法区分，保持 `UNRESOLVED`，不扩大模型/数据搜索。
- 不自动做：新架构、更多 seed、更多 n、Reserved/test 评分、ZINC 搜索。

---

## 9. 限制与未完成

- 单 scaffold split、3 seeds、Monitor 同时用于早停与比较；不是确认性实验。
- scaffold-cluster bootstrap 只覆盖 Monitor 抽样不确定性，**不覆盖**
  训练集/scaffold split 不确定性；CI 不能当作跨 split 的泛化保证。
- 没有做等效性检验；CI 跨零只写“未能区分”，不写“等价”。
- X 的 1.41× FLOPs 是真实混淆；虽然 X 没赢，但不能据此把 S 的优势
  完全归因于“没有交互”。
- G 是不同信息访问、不同参数量的应用参照，不能作信息/算力因果分解。
- FLOPs 用 `FlopCounterMode`，PyG scatter 可能未计入，是下界。
- 没有读/评分内部 Reserved 或官方 valid/test；因此本轮没有确认性泛化数字。
- 没有做 attention 深度/正则化/位置编码/学习率搜索（协议禁止）；
  因此不能回答“调过的 X 是否会更好”。
- 没有做 per-scaffold 或标签区间的探索性误差分析；避免结果驱动的新分析。

---

## 10. 复现与文件

| 内容 | 路径 |
|---|---|
| 冻结协议 | `notes/phase1a_protocol.md` |
| 纠错文件 | `notes/phase0_corrigendum.md` |
| 配置/hash | `configs/phase1a.yaml` |
| split | `results/phase1a/split_phase1a.json` |
| Fit/Monitor 标签工件 | `results/phase1a/labels_fit_monitor.json` |
| 18 条 run 的 provenance | `results/phase1a/run_manifest.tsv` |
| 分析汇总（无逐行标签） | `results/phase1a/analysis_summary.json` |
| 逐 run JSON（含 Monitor 预测，git-ignored） | `runs/phase1a/`（本地；远端 `res` 同路径） |
| 分析代码 | `code/phase1a/analyze_phase1a.py` |
| 训练/模型/数据代码 | `code/phase1a/` |
| CPU gates | `results/phase1a/CPU_GATES.md`（完整 JSON 本地 `cpu_gates.json`） |

```bash
# CPU gates + tests
uv run python tracks/structural-sufficiency/code/phase1a/phase1a_gates.py
uv run pytest -q tracks/structural-sufficiency/code/phase1a/tests

# 分析已拉回的 18 条 run
uv run python tracks/structural-sufficiency/code/phase1a/analyze_phase1a.py \
  --runs tracks/structural-sufficiency/runs/phase1a \
  --out tracks/structural-sufficiency/results/phase1a/analysis_summary.json
```

**失败/排除记录**：早期 orchestration 把 `--n` 写成 `full`，导致 9 次
full-n 启动在 argparse 阶段退出（exit 2，无结果 JSON）；另有一次错误的
worker 参数分组，只产生了 2 条有效的 S-small 轨迹，随后被正式 run 覆盖。
最终 18 条 run 全部 exit 0；失败启动不计入结果，也不用于任何选择。
