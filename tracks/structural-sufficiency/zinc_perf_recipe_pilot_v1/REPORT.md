# ZPR-v1 — ZINC 性能调试 + 配方试点：结论报告

- Round：`zinc-perf-recipe-pilot-v1`
- 决策提交：`fe5d60f`（决策冻结于 official test 读取之前）
- 训练执行：`res` / A100-40GB / torch 2.5.1+cu124，commit `1f7ada7`（两臂）
- 预算实耗：**0.868 A100·h**（job wall 累计：G0 0.371 + G1 0.462 + smoke 0.009 + divergence 0.022 + test read 0.004；两臂 trainer 自计 0.827），上限 1.5
- 正式轨迹：两条，均 seed 0，600 epoch / 47,400 updates，`status = completed`

本报告只回答被指定的 5 个问题。所有 valid 数字标 `VALID_AS_DEV_DESCRIPTIVE`（valid 参与了
checkpoint 选择），所有 test 数字标 `TEST_AS_DEV_DESCRIPTIVE`（test 在仓库中已被使用过，
不是新的盲测）。两者都不得用于回选 β / EMA / epoch / 结构。

---

## 1. 失败样本是否有共同结构特征？哪些解释成立？

**有，且集中在“训练集没覆盖的局部环境”这一项上。** 判定范围是 LCRC-A s0 与 DCC-L s0 的
test 逐分子 AE top-20 并集，共 26 行，其中 14 行同时在两个模型里（`SHARED_FAILURE`），
12 行只在一个模型里（`MODEL_COMPLEMENTARY`）。逐行表见
`analysis/phase_a_table.md`，机器可读见 `analysis/phase_a_summary.json`。

| 解释 | 成立的证据 | 强度 |
|---|---|---|
| **radius-2 typed 局部环境在 train 10k 中未出现** | 14 个共同失败行中 10 行含 ≥1 个 DF=0 的 radius-2 环境；244/1000 test 行含这类原子，却承担 LCRC 44.6% / DCC 47.6% 的总 AE（单行 AE 高出 3 倍量级）；按原子看，失败行常见 2–10 个未覆盖原子 / 14–30 个原子 | **主因**，与 valid 侧审计（25% 行承担 50% AE）在同一量级上复现 |
| **人工环惩罚（`cycle = -max(0, max(nx.cycle_basis)-6)`）** | 52/1000 test 行带此惩罚，承担 LCRC 14.8% / DCC 14.4% 的总 AE；4 个共同失败行（298、634、702、402）带惩罚；惩罚使目标移动 3.28–6.72 个 z 单位，而模型只能看到环数 → 结构上不可恢 | **成立，但性质是标签构造因子**，不是模型没表征好的结构 |
| **模型互补** | 12 行只进单侧 top-20；固定 50/50 融合对这些行的平均 AE 改善 +0.163（LCRC-only 侧），但对 DCC-only 侧是 **−0.101（变差）**；整体融合 test 0.058831 vs 0.066782/0.066267 | 成立且**不对称** |
| **标签或数据异常（行本身坏了）** | 未成立。3 行（208、680、770）graph→RDKit 身份无法验证，2 行（924、976）标签重构残差 3.4，但这 5 行全部是**低误差行**（AE 0.02–0.11），不驱动任何失败行 | 只作为数据侧 caveat |
| 残余（无上述两因子） | 共同失败行里只有 **336** 一行 r2 支持充足、无惩罚，仍未解释 | 单个残余案例 |

两点必须一起读：

- **“环惩罚”这个解释里有一半是 cycle-basis 伪影。** 52 个惩罚行里有 **6 行**（230、402、547、
  670、687、728）的惩罚来自某一次 `nx.cycle_basis` 抽到非最小基环（如稠环的外周长环），
  其 RDKit 最小环基里**没有任何 ≥7 元环**（`max_mcb ≤ 6`，`rdkit_max_ring_size ≤ 6`，
  `longest_simple_cycle = 8`）。9 行 test 存在 basis≠MCB 不一致。见
  `analysis/phase_a_basis_artifact.json`。行 402（DCC AE 3.09，全场最差）正是这类。
- **标签重构残差是可解释的系统项，不是随机噪声。** 8 个 `cycle=-2` 行的重构残差
  一致为 0.0619，`cycle=-1` 行中位 ~0.11、`cycle=0` 行中位 0.002 —— 因为存档标签用的是 GVAE
  流水线自己的 z 标准化常数，与从冻结属性表重算的常数略有差异，残差随 |cycle| 线性放大。
  阈值 0.5（远高于该系统项、远低于两个真实错配的 3.41/3.44）的选取依据见
  `analysis/phase_a_label_provenance.json`。

## 2. SmoothL1 是否优于 L1？

**不优于，方向相反。** 两臂除损失外完全同源（同一 init 哈希 `11c698ae`、同一 SSL 权重与
缓存、同一 `randperm` 顺序、同一 AdamW/cosine/warmup、同一 batch 与 47,400 updates）。

| 判据（预注册 §2.4） | 要求 | 实测 Δ (G1−G0) | 通过 |
|---|---|---:|---|
| full valid | Δ ≤ −0.003 | **+0.00298** | 否 |
| ordinary | Δ < 0 | **+0.00407** | 否 |
| minus row172 | Δ < 0 | **+0.00537** | 否 |

**verdict = `NO_COMPELLING_RECIPE_GAIN`**（`analysis/decision.json`）。

所有 bulk 指标都一致变差：ordinary +0.0041、r2_unseen_ordinary +0.0023、seen_ordinary
+0.0046、minus_fixed_worst10 +0.0080、W20 +0.0047、T100 +0.0047。唯一变好的是 tail
（y ≤ −3，0.2768 vs 0.2895，−0.0127）。test（单次、描述性）方向一致且更大：
G0 0.067030 / G1 0.073041（**+0.00601**），p90 0.1391→0.1509，max AE 1.53→2.51。

机制上这是可以预期的，并非“加了鲁棒性反而变差”的悖论：本模型在 best epoch 的训练 MAE
只有 0.0034（G0）/0.0139（G1），而 β=0.05，于是 **绝大多数训练样本落在 Huber 的二次区间**，
每样本梯度是 `d/β`（约为 L1 的 1/20）而非 ±1。也就是说 G1 实际更像“对易样本降权 20 倍的
L2”，把梯度预算挪给了少数困难样本——这正好解释了 tail 变好、bulk 变差、训练集拟合差 4 倍
（0.0139 vs 0.0034）这三点同时出现。注意：这个观察**不构成**调 β 的许可，本轮不做任何 β 调整。

## 3. EMA 是否有收益？

**没有可复现的收益。** 被动 EMA（decay 0.999，第 11,850 step＝25% updates 起，与在线模型不共享
RNG、不改变 optimizer 轨迹；通过性校验见 `analysis/cpu_checks.json` 的 C6/C7 与
`runs/*/train.json` 的 `ema.audit`）在各自轨迹内部比较：

| 臂 | online best | EMA best | Δ (EMA−online) | 方向 |
|---|---:|---:|---:|---|
| G0（L1） | 0.07923 | 0.07966 | **+0.00043** | EMA 更差 |
| G1（SmoothL1） | 0.08221 | 0.08247 | **+0.00025** | EMA 更差 |
| G0 test | 0.067030 | 0.066940 | −0.00009 | EMA 略好 |
| G1 test | 0.073041 | 0.071899 | −0.00114 | EMA 略好 |

valid 上两臂 EMA 都更差，test 上两臂 EMA 都更好 —— **符号在 split 之间翻转**，量级
(1e-4…1e-3) 低于重跑噪声带，因此只能判定为“无可测收益”。另外 EMA 起点在 25% updates，
而两臂 best 分别出现在 epoch 508 / 310，EMA 的有效平均窗口与 best 附近并不对齐，这也让
“EMA 应当更好”的先验在本配方下不成立。**不部署 EMA，也不因为 EMA 再开一轮。**

## 4. 是否值得进一步确认？

**不值得。** 三条理由，都写在数据里：

1. 结论方向是负的：候选在 full/ordinary/minus-row172 三条判据上全部朝变差方向失败，不是
   “接近阈值但没到”。按预注册规则，这种情况不进入确认轮。
2. 本轮能分辨的最小效应大于被测量：配对 bootstrap 95% CI 为 **[−0.0057, +0.0098]**（1000 个
   valid 分子），跨 0；同时“同一实验跑两次”的噪声带是：早期相位最大 valid 差异 0.427、
   选择时刻的持续偏移 0.00265（G0 vs 2026-10-28 同配方重跑）、历史同配方 seed 间差 0.0039。
   被检验的效应量 0.003 与噪声带同阶 —— 即使换成 4 个 seed，也只能把 CI 压到 ~0.0015，
   而那需要 ≥4×2 条 600-epoch 轨迹，远超本轮 1.5 A100·h 的预算与“仅 seed 0”的约束。
3. 没有可确认的机制假设：唯一显示“收益”的子群是 tail（n=77、y ≤ −3，LCRC 侧 AE 0.29），
   且它的方向与 bulk 相反；把它当成下一轮的目标就是把单 seed 的 0.0127 噪声当信号。

配套的诚实结论：即使 Δ 达到 −0.003 门槛，本轮也只允许写成 `seed0 candidate`，不自动开 seed 1。
现在连门槛都没有达到，所以连 candidate 都不成立。

## 5. 停止，还是另立确认轮？

**停止。** 不新增 seed、不调 β、不调 EMA decay/起点、不改结构、不由此派生新方向。

若将来确实要回答“损失形状对 ZINC 的作用”，需要的不是更多 seed，而是**先解决执行噪声**：
本轮 `zpr_divergence.py` 证明冻结 trainer 在同一张卡上连续跑两次，第 4 个 epoch 的 valid MAE
就差 0.427（init 哈希相同、首个 batch 行序相同、torch/CUDA 相同、commit 相同）。这是 CUDA
原子加（`index_add_`/scatter）顺序不确定 + 早期训练不稳定的放大，不是 trainer 差异。在该
执行机制下，任何 <0.003 的配方效应都无法用曲线级证据判定，需要先把训练路径做成可复现
（确定性 scatter 或固定分块归约），否则确认轮只是在测量 GPU 调度。**这条路线本轮不开**，
仅记录为前置条件。

---

## 6. 执行偏差与口径记录

1. **C4 判据修正（训练后、读 test 前）**：预注册要求“L1 控制臂复现 2026-10-28 的 600-epoch
   曲线到 1e-5”。实测最大 valid 差异 0.3585（第 9 epoch），随后稳定在 ~0.0026 的偏移。
   为此补做 `zpr_divergence.py`（同一进程内用冻结 trainer 连跑两次 12 epoch）：冻结 trainer
   **对自身**的差异是 0.427，对历史曲线是 0.397/0.177，均 ≥ 控制臂对历史的 0.359 —— 即该判据
   在本执行机制下不可达，而非控制臂违规。改用的可达判据（frozen 输入逐位一致 + 控制臂发散
   不大于冻结 trainer 自发散）通过，见 `analysis/c4_resolution.json`。
2. **Phase A 解释口径澄清（在 Phase A 脚本内、看 test 误差之前）**：`possible_explanation`
   只在“失败行”上给判（失败行 = 两模型 AE top-20 并集），否则任意低误差行也可能因符号规则
   被标成 `MODEL_COMPLEMENTARY`，失去意义。`UNRESOLVED` 保留为不可达回退分支。
   两模型都大误差的行必然落在某侧 top-20 内，所以“共同失败 ⊆ 交集”不是简化而是定义。
3. **标签重构阈值 0.5**：见 §1 第 2 点，依据是先测出系统项（≤0.15）与真实错配（3.41/3.44）
   之间的空档；阈值不参与任何选择，只影响 caveat 的标注。
4. **`runs/` 不再进 Git**（新增 `.gitignore` 规则），远端生成的 JSON 也改写到 `runs/` 下，
   避免与已提交的 analysis 副本在下次 `rr deploy` 冲突。
5. **test 读取次数 = 1**（`analysis/test_read.json` 的 `read_count`，脚本内有 once-guard；
   冻结文件在读取前写入）。读取有效性的独立验证：用冻结的逐分子预测 CSV 复算历史基线，
   LCRC 0.0667823695（存档 0.0667823703，差 7e-10）、DCC 0.0662665578（存档 0.066267，
   差 4.4e-7）—— 说明 test 标签顺序与特征化路径都没问题。
6. **分组口径可比性**：test 分组用的 radius-2 typed 环境规则，先在 valid 上复现冻结 anchor
   （251/251 完全一致）后才允许读 test；见 `analysis/test_read.json` 的
   `grouping_rule_selfcheck`。

## 7. 产物索引

| 文件 | 内容 |
|---|---|
| `PROTOCOL.md` | 预注册协议（含 §2.4 决策规则） |
| `analysis/phase_a_table.md` | 指定 11 行 + top-20 并集逐行诊断短表 |
| `analysis/phase_a_summary.json` | 解释桶、机制检验、标签/数据发现 |
| `analysis/phase_a_basis_artifact.json` | cycle-basis 伪影（6 行）与 MCB 对照 |
| `analysis/phase_a_label_provenance.json` | 标签重构系统项 vs 真实错配 |
| `analysis/main_table.{md,json}` | 两臂全部预注册指标 + 配对统计 + EMA 臂内比较 |
| `analysis/decision.json` | 决策（`NO_COMPELLING_RECIPE_GAIN`）+ test 读取记录 |
| `analysis/c4_resolution.json` | C4 判据修正与证据 |
| `analysis/divergence.json` | 冻结 trainer 重跑对照（执行噪声带） |
| `analysis/cpu_checks.json` | C0–C8 正确性门禁原始输出 |
| `analysis/test_read.json`、`test_per_molecule.csv` | 单次 test 读取（`TEST_AS_DEV_DESCRIPTIVE`） |
| `runs/{G0,G1}_s0/` | 曲线、checkpoint、逐分子 valid 预测、train.json |
| `code/`、`tests/` | 训练器、Phase A、对照、比较、test 读取、31 个纯数学测试 |
