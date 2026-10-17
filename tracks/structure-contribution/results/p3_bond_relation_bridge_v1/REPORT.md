# cscl-p3-bond-relation-bridge-v1 — 报告

**结论（第一段）**：在保持 BJOC-P3 全部 atom/bond/junction 输入、SSL 初始化、R3 配方与
全部 413,505 个参数不变的前提下，加一条共享 pair MLP 的**真实一跳 Bond–Bond 关系**残差，
**不能**给当前最强 P3 带来任何有意义的预测收益：Locked（n=1,000，两 seed）上
`P3+R − P3 = −0.000099`（95% CI `[−0.002205, +0.001967]`，两 seed 异号，Monitor 上反向
`+0.001127`），远低于预注册的 `0.004` 实用门槛，判定 **NEUTRAL**；它是否超过同容量、
同行和的稠密 Null 分支的答案是“方向对但同样不显著”：`P3+R − P3+U = −0.001306`
（CI `[−0.003435, +0.000762]`，两 seed 同为负但在 Monitor 上反号 `+0.001127`），也是
**NEUTRAL**。机制上这不是“关系没被用上一点”，而是**这条分支基本没被用上**：零初始化门控
`γ` 在训练后被优化到 `|γ|≈3.2e-4`（seed 1 甚至 `5.3e-11`），部署模型的残差只有
`|c|` 的 `6.6e-5`，**把 R 核换成 U 核后预测变化为 0**。按预注册判定为
**`CASE_D_NO_GAIN_ON_P3`**：该融合路线在 ZINC 上停止，不追加 seed/架构/超参，
也不购买新的 ZINC 结构模块实验。**重要偏差**：本轮 split F 因盐值默认参数 bug
与上一轮 split E 逐行相同（见 §6），所以这是“同一批 2,000 个评测分子上的两 seed 配对测量”，
**不是**独立复现。

---

## 1. 问题与预注册

新协议 `cscl-p3-bond-relation-bridge-v1`（与已冻结的 `bond_relation_attribution_v1`
无关，后者 `CASE_C_INCONCLUSIVE` 不被修改）。三臂：

| 臂 | 关系分支 | 参数 |
|---|---|---|
| `P3` | 无（原 BJOC-Null，junction objects 入 reader） | 413,505 |
| `P3+R` | `r_e = Σ_{f≠e} K^R_ef φ(b_e,b_f)`，`K^R=A_line/deg`，`c'=c+γ⊙r`，`γ` 零初始化 | 422,945 |
| `P3+U` | 同实现、同参数、同初始化，`K^U_ef=1/(n_b−1)`（`e≠f`） | 422,945 |

`φ`：`Linear(192→32) → SiLU → Linear(32→96)`（对称输入 `[b_e+b_f, |b_e−b_f|]`），
`γ∈R^96` 逐通道零初始化；**不改** `structural_relation_composer_v1.py`，只重算 reader
输入的 `z` 块。两核对角线严格 0、有邻居时行和恰为 1、总质量相等（R/U 每分子都是
`n_bonds`），因此 R−U **不是**行和/自连接伪影；差异只在支撑、集中度与 FLOPs。

## 2. 执行与冻结

- 冻结：`protocol_freeze.json`（amendment 1），code `p3_bond_relation_bridge_v1.py
  ee6dd258…` / driver `263c793b…`，config `090a77b3…`，11 个输入文件哈希；commit
  `14e3c7927aa5`。
- 8/8 CPU 门禁通过：核性质与重编号不变性；两新增臂参数/初始化/顺序完全一致
  （state hash 相同）；`γ=0` 时输出与 P3 **逐位相同（max|Δ|=0.0）**；非零 `γ` 下 R/U
  都改变前向；`φ` 梯度第 1 步恰为 0、第 2 步起非零（零初始化残差的已知性质，如实记录）；
  标签路径与 junction 块未被改动；单键/环/空 pair 边界稳定；**与冻结训练器 soup 逐位一致
  （0.0）**。45 个单元测试通过。
- split：9,996 行（排除 4 行历史 SMILES/PyG 不一致）、9,993 个身份组、7,996/1,000/1,000、
  身份组跨区 0（**但见 §6 D-1：实际盐值沿用历史默认值，划分与 E 相同**）。
- SSL：Fit-only masked atom/bond，7,560 updates，1,632 s / 1,677 s；atom acc
  0.9264 / 0.9305，bond acc 0.9953 / 0.9941；`full_encoder == encoder_final`。
- 监督训练：22,680 updates × 360 epochs、batch 128、raw-y L1、Adam 1e-3→1e-5、wd 1e-5、
  clip 5、last-5 soup；墙钟 P3 533/617 s、P3R 617/717 s、P3U 640/730 s；
  **GPU ledger 1.9896 h / 4.0 h**。
- 预测：20 个文件（含 8 个换核诊断）先冻结哈希
  `4205b23856dd3a985e7ea630d959cf15ac13cb62a8726f201d1dfc0c0b045002`，
  之后**单次**读取 Monitor+Locked 标签（`holdout_read.lock` 07:27:41Z；
  monitor y `531953fd…`、locked y `e0e798a8…`）。official valid/test 全程未加载。

## 3. 结果

### 3.1 Locked（主判定区，n=1,000，两 seed 平均）

| 臂 | MAE | seed0 / seed1 | RMSE | 中位 AE | 普通区 MAE (n=942) | 尾部 MAE (n=58) |
|---|---|---|---|---|---|---|
| P3 | 0.083320 | 0.087169 / 0.079471 | 0.210180 | 0.05059 | 0.074559 | 0.225610 |
| P3+R | **0.083221** | 0.086222 / 0.080220 | 0.204690 | 0.05076 | 0.074463 | 0.225451 |
| P3+U | 0.084526 | 0.086886 / 0.082167 | 0.208644 | 0.05062 | 0.075324 | 0.233990 |

| 对比（负 = 前者更好） | Δ MAE | 95% CI | 判定 | seed0 / seed1 | 普通区 Δ | 尾部 Δ | top-10 剔除后 Δ | 逐分子更优比例 |
|---|---|---|---|---|---|---|---|---|
| **P3+R − P3** | −0.000099 | [−0.002205, +0.001967] | NEUTRAL | −0.000947 / **+0.000749** | −0.000096 | −0.000160 | +0.000659 | 0.497 |
| **P3+R − P3+U** | −0.001306 | [−0.003435, +0.000762] | NEUTRAL | −0.000664 / −0.001948 | −0.000860 | −0.008539 | −0.000952 | 0.525 |
| **P3+U − P3** | +0.001207 | [−0.000684, +0.003109] | NEUTRAL | −0.000283 / +0.002697 | +0.000765 | +0.008380 | +0.001235 | 0.481 |

### 3.2 Monitor（复现区，n=1,000）

| 臂 | MAE | seed0 / seed1 | RMSE | 普通区 | 尾部 |
|---|---|---|---|---|---|
| P3 | 0.090082 | 0.090357 / 0.089808 | 0.282766 | 0.072501 | 0.360722 |
| P3+R | 0.091210 | 0.095115 / 0.087304 | 0.284395 | 0.072352 | 0.381486 |
| P3+U | 0.090082 | 0.090214 / 0.089950 | 0.283561 | 0.072800 | 0.356121 |

| 对比 | Δ MAE | 95% CI | seed0 / seed1 | 普通区 Δ | 尾部 Δ |
|---|---|---|---|---|---|
| P3+R − P3 | **+0.001127** | [−0.001082, +0.003313] | +0.004758 / −0.002504 | −0.000148 | +0.020764 |
| P3+R − P3+U | **+0.001127** | [−0.000939, +0.003267] | +0.004901 / −0.002646 | −0.000447 | +0.025365 |
| P3+U − P3 | −0.000000 | [−0.002147, +0.001974] | −0.000143 / +0.000142 | +0.000299 | −0.004601 |

三个对比在 Locked 与 Monitor 上全部 **NEUTRAL**，`replicated` 全为 false
（R−P3 在 Monitor 上反而是 +0.0011，尾部 +0.0208）。RMSE 上 P3+R 比 P3 低 0.0055
（−2.6%），但 MAE 不动、剔除 10 个最差分子后反而 `+0.000659`，且换核预测差为 0 —— 这是
训练轨迹噪声，不作为收益（RMSE 也不是预注册判定指标）。

## 4. 机制：分支为什么是“死的”

| 证据 | P3+R seed0 | P3+R seed1 | P3+U seed0 | P3+U seed1 |
|---|---|---|---|---|
| `mean|γ|`（训练后 soup） | 3.2e-4 | 5.3e-11 | 1.3e-4 | 4.2e-8 |
| `γ` 非零通道比例 | 3.1% | 0.0% | 4.2% | 1.0% |
| `mean|γ⊙r| / mean|c|` | 6.6e-5 | 2.4e-18 | 2.2e-5 | 1.8e-12 |
| 末端 `φ` 梯度范数 | 1.9e-5 | 7.5e-12 | 1.3e-6 | 3.8e-9 |
| 首步 `γ` 梯度范数 | 0.0220 | 0.0206 | 0.0234 | 0.0218 |

1. **门控塌缩**：`γ` 从 0 出发，Adam 第一步确实给出非零梯度（0.021），但 22,680 步后
   只剩 `1e-4` 量级（seed 1 到 `1e-11`）；分支对前向的贡献是 `|c|` 的 0.007%。
   于是 `φ` 的梯度（正比于 `γ`）比首步小 3–10 个数量级，pair MLP 基本没学到东西。
2. **换核诊断**：把训练好的 P3+R 权重配 U 核、P3+U 权重配 R 核，Locked 上预测差
   `mean = 0.0`、`max = 0`（冻结前完成、不参与判定）。即**部署模型完全不依赖核**，
   所以 R−U 的任何差异都不可能来自“真实连接”。
3. 因此 Locked 上 `R−U = −0.0013`（两 seed 同负）与 Monitor 上 `+0.0011`（反号）之间的
   摆动，属于同一 Fit 数据上“多了 12,512 个几乎不激活的参数”导致的轨迹分叉，
   与 P3 的 seed 间差异（0.0872 vs 0.0795）同量级。

核审计（1,200 分子，标签无关）：对角线 0、行和 1、由 pair list 重建稠密核**逐位精确**；
R 每分子 34.6 个 pair（占稠密 11.5%），U 为 312.1（95.8% 稠密）；线图平均度 2.77
（最大 6）；`R≡U` 退化分子 0 个（ZINC-12K 最少 8 个键，线图从不是完全图）。

## 5. 判定

`CASE_D_NO_GAIN_ON_P3`（预注册规则 D）：
`P3+R` 未能在 `0.004` 实用尺度上改进 P3（Locked `−0.000099`，CI 过 0，Monitor 反向），
且未能在任何区域形成稳定优势。按预注册：
- **停止**“给 P3 加一跳 Bond 关系残差”这条融合路线；
- **不**追加 seed、宽度、初始化、学习率或新算子来“救”这个结果；
- **不**购买新的 ZINC 结构模块实验。

同时必须说清楚这个判定**不**等于“真实 Bond–Bond 关系没有信息”：本轮观察到的是
“**这个零初始化残差 + R3 配方**下，强 P3 主干不把预测工作分配给新分支”。要区分
“信息无用”和“优化没接上”，需要另立协议（例如：直接监督关系残差、非零初始门控、
把关系项接到更窄的骨干、或在更难的标签上评估）——本轮没有购买这些，也不应据此轮
数据声称任何一方。

## 6. 执行偏差（详见 `DEVIATIONS.md`）

- **D-1（重要）**：`cmd_split` 调用 `bra.build_identity_split` 时未传 `salt`，函数回落到
  历史默认盐 `cscl-bond-relation-attribution-v1`，因此 **split F 与 split E 逐行相同**
  （fit/monitor/locked 行哈希完全一致）。`split_F.json` 里写的本轮盐值是错误元数据。
  后果：本轮评测分子 = 上一轮评测分子，**不是独立复现**，两轮结果不可合并为独立证据；
  但三臂对照本身仍是同 Fit、同 SSL、同顺序、同配方、无标签泄漏的配对比较，
  `CASE_D` 的臂间差值不受此影响。未用换盐重跑补救：新的 1,000 行 Locked 里约 80%
  会是本轮已训练过的行，属于更严重的泄漏；且单次读标签已用掉。
- **D-2**：budget ledger 只计 SSL 与监督训练（1.9896 h）；预测/换核/评分/post-hoc
  的短前向（<5 min）未计入，已在 evidence manifest 注明。

## 7. 限制

1. **同一 10k 分子总体**：历史多轮已反复使用该总体；本轮（尤其因 D-1）不构成独立外部
   验证。绝对 MAE 也不与历史轮次直接可比（本轮 P3 Locked 0.083320 vs 上一轮 0.082591，
   差异来自 SSL 来源/轨迹，不是同一模型）。
2. **R/U 的稀疏性没有隔离**：R 每分子 34.6 项、U 312.1 项，FLOPs 与支撑同时变化；
   本轮只是“性能对照”，不能把 R−U 解释为“真实连接归属”的因果效应。
3. **接近 bond-line-graph message passing**：一跳、行归一化、对称 pair MLP、单次聚合，
   不能声称“完全没有消息传递”；它不是多层 GNN/attention，但也不是无消息传递的模型。
4. **与历史 junction 归属实验可能信息冗余**：P3 的 junction moments 已经广播局部环境；
   新分支虽然只用 bond 表示、不含 junction 内容，但共享同一 atom/bond 编码器，
   独立性无法保证。
5. **容量控制有限**：`+3.0%` 参数由 P3+U 控制；U−P3 为 `+0.0012`（Locked）/`−0.0000`
   （Monitor），说明“加参数/一般 pair 交互”本身没有收益，但这不是等 FLOPs 对照。
6. **单次读取、两 seed**：seed 间差异（P3 自身 0.0872 vs 0.0795）大于任何臂间差值，
   因此本轮对 `<0.002` 量级的效应没有分辨力；这一点在判定中被 `0.004` 门槛吸收。
7. official valid / official test 全程未加载；四个历史 SMILES/PyG 不一致行按统一规则排除。

## 8. 交付与证据

`results/p3_bond_relation_bridge_v1/`：`PROTOCOL.md`、`AMENDMENTS.md`、`DEVIATIONS.md`、
`protocol_freeze.json`（+ `protocol_freeze_amendment1.json`）、`split_F.json`、
`relation_audit.json`、`cpu_gates.json`（8/8）、`ssl_F_s{0,1}.json`、
`train_{P3,P3R,P3U}_F_s{0,1}.json`、`prediction_manifest_F.json`（20 文件）、
`holdout_read.lock`、`holdout_score_F.json`、`decision.json`、
`per_molecule_errors_{monitor,locked}.csv`、`post_hoc_bridge.json`、
`gpu_budget_ledger.json`、`evidence_manifest.json`（69 artifacts / 263.5 MB，
本地 69/69 哈希一致）、`staged_inputs_F.json`、`smoke/smoke_F.json`、`predictions/*.npz`。

复现入口：
```bash
uv run python tracks/structure-contribution/code/run_p3_bond_relation_bridge_v1.py \
  {split,verify-inputs,objects,kernels,cpu-gates,freeze,stage-inputs,smoke,ssl,train,predict,predict-cross,freeze-predictions,score-holdout,post-hoc,finalize}
```
