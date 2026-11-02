# LCRC Lipophilicity 模型解剖：勘误 + 冻结诊断（follow-up v1）

- **源轮**：`tracks/structural-sufficiency/lcrc_lipo_model_dissection_v1/`
- **源轮审查 commit**：`4a393476aace617666642d87e0443ea3c3af87e0`（本轮实际 checkout 一致）
- **本轮性质**：erratum + frozen-model diagnostic。**新训练 = 0；新 Test 推理/重算 = 0**。
- **运行环境**：CPU-only；6 个已选 checkpoint（A0/C1 × seeds 0/1/2），24 次 condition 评估；全部脚本合计约 10 s。
- **数据边界**：只用 repeat 122 Valid（420 行）的已有预测与已有 checkpoint；Test 只做 `main_runs.csv` 中**已公开 RMSE/MAE 标量**的算术修正，标为 `EXPOSED_TEST_SCALAR_DIFF_ONLY`，不是新的 Test 评估/选择。
- **本轮目录**：所有新代码、表格、诊断只在本目录；源轮文件、checkpoint、predictions、freeze、bootstrap、Test access log 均未改动；`STATE.yaml` 未改。

---

## 0. 一页结论

### 1) 跨 repeat 聚合是否混用了不同划分的参照模型？哪些原轮数值/表述要勘误？

**是，有一个确定的实现 bug，但只污染了 `contrasts_*.json` / `markdown.md` 的 contrast 表，不是 72 个 run 或 cross-repeat 表。** 原 `dissect_tables.py::_seed_diffs` 用 `(variant, seed)` 作为字典键；先累积 repeat 122，再累积 123/124，后写覆盖前写。结果：只在 122 出现的变体，其 `A0` 参照在部分 cell 被替换成了 **124 的 A0**；C1/B5 则直接变成了 124 的配对。对 **11 个以 A0@124 为参照的 cell**，Valid 的统一偏移是 **+0.0692998**（FULL122 mean 0.7065861 − FULL124 mean 0.6372863），Test 是 **+0.0378856**；C1/B5 是 candidate/reference 都取自 124，不能用一个统一偏移概括。

- 原 `analysis/tables/contrasts_valid.json`、`contrasts_test_exposed.json`、`analysis/tables/markdown.md` 的 `contrasts_122` 表中，**15 个 contrast 行需要修正**；例如 `A2−A0` 原 **+0.077999** → 修正 **+0.008699**，`A3−A0` 原 **+0.060550** → **−0.008750**，`A_INTERACTION` 原 **+0.014753** → **−0.054547**。
- **不受影响**：`main_runs.csv`、每 run 记录、`cross_repeat.json`（按 repeat 分组后再平均）、repeat-122 bootstrap 的 point estimate、所有 Valid 预测、checkpoint、compact 选择。原 REPORT 的大部分主结论引用的是 repeat-122/bootstrap 值，因此**不是整份报告作废**；作废的是上述 contrast JSON/Markdown 的错误行及引用它们的句子。

详见 §2 和 §2.6 的 11 条勘误表。

### 2) A3/A5 的 “RMSE 下降、MAE 上升” 发生在哪些样本？RMSE 改善是否集中在少数极端误差？

**是，A3/A5 的均值 RMSE 改善几乎完全由同一个极端样本 row 2087 提供；去掉它后，3/3 seeds 的 RMSE 都变差。** row 2087（`CC(N)(COP(=O)(O)O)C(=O)O`，y=2.5，Valid 位置 212）是 FULL 三个 seed 的 **|error| 第 1 名**；在 seed 0/1/2 分别占 FULL 总 SSE 的 **43.96% / 41.01% / 54.21%**。A3 在三个 seed 上把它的 SSE 降低 **7.81 / 32.65 / 3.74**，A5 降低 **14.18 / 6.63 / 40.71**；排除该行后：

- A3 outside-row2087 ΔRMSE = **+0.03534 / +0.01799 / +0.01510**（均值 **+0.02281**）
- A5 outside-row2087 ΔRMSE = **+0.03835 / +0.01104 / +0.03822**（均值 **+0.02920**）

排除固定 top-5/top-10 后 worse 更大。因此“RMSE 更低”是**单个极端高估被纠正**的算术结果，不是 420 个样本普遍变好。

**MAE 的代价则是分散的。** A3 平均 **232/420** 个样本 |error| 变大（188 个变小），median ΔAE 为 **+0.02884**（3/3 seeds 为正）；A5 平均 **220.7/420** 变大（199.3 变小），median ΔAE 为 **+0.01200**（2/3 seeds 为正，seed 1 为 −0.00204）。所以 A3/A5 是典型的 **RMSE 被一个极端样本拉低、MAE 被大量小误差抬高**，不能概括为“去掉一条读出路径几乎免费”。

### 3) 对实际训练的 FULL/C1，分别替换 S2 的 J 配对统计与 C content 加权统计，内部表示/预测各变多少？

**在实际训练的 6 个 checkpoint 上，J 替换是主导项，C 替换很小但非零；两者不是严格可加。** 用 `JCLocalContent` 子类只替换 `jc_joint_agg` 或 `jc_content_agg`，其他输入/参数/权重完全不变：

- **内部表示**：J_ONLY 覆盖 **1,632/11,465 roots（14.2%）**、**389/420 molecules（92.6%）**；逐 root 的 |Δh| 中位数约 **7e-7**（多数 root 不动），p95 **4.03–4.80**，max **9.56–13.38**。C_ONLY 覆盖 **2,566 roots（22.4%）**、**410 molecules（97.6%）**；|Δh| 中位数约 **1.3e-7**，p95 **0.16–0.28**，max **0.39–0.77**。J 的内部扰动比 C 大一个数量级。
- **预测**：J_ONLY 的 Valid RMSE 变化 **+0.000525 … +0.023837**（6/6 checkpoint 变差，均值 **+0.01231**）；C_ONLY **+0.000214 … +0.002487**（RMSE 6/6 变差，均值 **+0.001125**；MAE 均值 +0.001209，逐 checkpoint 混合）；BOTH **+0.000646 … +0.023625**（均值 **+0.012585**）。J 解释了大部分输出损失，C 在输出上接近“几乎免费但非零”。
- **门控不是常数，但也不是处处变化**：path-weighted gate mean **1.038–1.075**、sd **0.063–0.088**，范围约 **0.75–1.27**；Valid 的 57,325 个 (root, bt2) 组中 39,471 个为空、5,320 个 singleton、12,534 个 multi；只有 **23.5%** 的 multi 组组内 gate sd > 1e-12（组内 sd 中位数为 0）。因此 C 的“小”是**训练后门控变化稀疏且输出敏感度低**，不能解释成“C 信息无用”。
- **可加性**：BOTH − J_ONLY − C_ONLY 的非线性残差 mean |r| **0.00385–0.00555**、max **0.0505–0.1036**；不能把 J/C 当成精确的方差分解。

### 4) 哪些问题本轮能答，哪些不能？

**能答**：原轮哪些 contrast 数字被跨 repeat bug 污染；A3/A5 的 RMSE 改善在样本层面来自哪里；实际训练权重下 J 与 C 的**条件性**表示/预测敏感度。

**不能答**：重训后单读出模型能否保持/缩小差距；A3/A5 的 MAE 代价在更大数据/更多 seed 下是否仍成立；Test 泛化或盲测；row 2087 是标签问题、样本问题还是模型失败；J/C 的化学因果作用；C1 的跨轮稳健性。所有这些都需要新训练、更多数据、或新的（盲）Test，不在本轮“零训练”边界内。

### 5) 新训练/新 Test 计数与缺失 cells

- 新训练/微调/epoch 重选：**0**。
- 新 Test 推理、重算、标签读取：**0**；Test 只用已有 `main_runs.csv` 的 RMSE/MAE 标量做算术修正。
- 加载 checkpoint：**6**；condition 评估：**24 = 6 checkpoint × {NATIVE, J_ONLY, C_ONLY, BOTH}**；没有 condition 失败。
- 身份/缺失检查：72 个 `(repeat, variant, seed)` 身份，repeat 122 = 54，123/124 各 9；每个已有 run 的 Valid n=420；预定义 contrast 的缺失 cell 均按 “unavailable” 显式列出，没有静默丢 cell。

---

## 1. 输入、身份与对齐核对

### 1.1 主运行表身份

`code/followup_common.py::check_main_rows` 从 `main_runs.csv` 读出 72 个唯一 `(repeat, variant, seed)`：

- repeat 122：18 variants × 3 seeds = **54**
- repeat 123：A0/C1/B5 × 3 = **9**
- repeat 124：A0/C1/B5 × 3 = **9**
- 每个已有 run 的 `valid_n = 420`

同一 `(repeat, variant, seed)` 的重复行、非预期 variant、非预期 split 都会报错；本轮没有触发。

### 1.2 预测对齐

`code/error_concentration.py` 用 **row_id**（不是位置）对齐 A0/A3/A5/C1/B5 × 3 seeds 的 repeat-122 Valid 预测；所有 run 的 `y_true` 逐行完全相同，row_id 顺序一致，无缺失/重复。来源是原轮 `analysis/predictions/valid_*.npz`，没有重新推理。

### 1.3 row 2087 的 1.15e-6 上下文差

- 保存的 Valid 预测：`pred = -7.021600776`，`|e| = 9.521600776`，SSE = 90.660881337。
- 原 single-molecule attribution JSON：`pred = -7.021601929`，`|e| = 9.521601929`，SSE = 90.660903295。
- 差 **1.15e-6**，是 batching/数值精度差，不是 row/label mismatch。B 全部使用保存的 Valid 预测；attribution 值仅记录用于审计，**没有覆盖原文件**。

### 1.4 一致性/版本

- 源轮 checkout 为 `4a393476aace617666642d87e0443ea3c3af87e0`；除本轮新目录外工作树干净。
- `input_manifest.json`：缺失输入 0；源树与 commit 一致 = true；没有把 Test 标签/预测文件列入输入。
- **协议元数据差异（已记录，未静默忽略）**：`analysis/protocol_freeze.json` 里有两个 code hash 与当前 checkout 不一致：`code/dissect_tables.py`（freeze 后修复 `compact_selection['selected']`）和 `code/dissect_intervene.py`（新增 `--out-tag`）。当前 checkout 本身与 `4a393476` 的 git blob 一致；这两个文件的改动不触碰 `_seed_diffs`、contrast 生成、per-run 记录或 Valid/Test 标量列。本轮分析**没有 import 或执行这两个脚本**，而是独立重写了 corrected tables，因此依赖的是 run 记录/预测/checkpoint 本身，而不是这两个变更后的代码路径。该判断和 hash 差异均写入 `input_manifest.json`，作为限制保留。严格说，这是 **freeze 元数据与 checkout 的 provenance 不一致**，不是本轮所用数据 `main_runs.csv` / Valid 预测 / checkpoint 的一致性问题；本轮没有把这两个脚本当作分析输入执行。

### 1.5 Test 边界

- 未读取 Test 标签、Test 预测文件、Test 样本级数据。
- `analysis/corrected_tables/corrected_contrasts_test_exposed.json` 中每个 cell 都标 `EXPOSED_TEST_SCALAR_DIFF_ONLY`；输入只有 `main_runs.csv` 的 Test RMSE/MAE 标量。
- 原 `analysis/test_access_log.json` 未修改；本轮没有新增 Test access 记录。

---

## 2. Work A：跨 repeat bug、修正表与勘误

### 2.1 bug 机制（可复现）

原 `dissect_tables.py` 的 `_seed_diffs` 先按 repeat 122、123、124 顺序累积，键为 `(variant, seed)`；后一个 repeat 的同名 cell 覆盖前一个。对只在 122 出现的 variant，reference 在部分 seed 上从 `A0@122` 变成 `A0@124`；对 C1/B5，candidate/reference 都变成 124。

复现脚本 `code/corrected_tables.py::reproduce_legacy_bug` 明确按原顺序构造 `(variant, seed)` 字典；输出在 `legacy_affected_differences.json`。它不是“重新解释原代码”，而是把错误的 keying 重放一遍，确认能逐值复现旧表。

### 2.2 修正实现

- 所有比较都是 **same-repeat, same-seed paired**：每个 repeat 先算 `candidate − reference` 的 3-seed 均值；需要跨 repeat 汇总时，只对“每个 repeat 已经 same-seed 配对后的均值”做**等权描述性平均**。
- identity 键包含 `(repeat, variant, seed)`；缺臂时返回 `unavailable` + `missing_cells`，不补值、不跨 repeat 填空。
- 预定义 A/B/C 对比复用原轮定义；另加 `A4−A0`、`A5−A0` 两个 **supplemental anchor**（在 JSON 中标为 supplemental，不改变 legacy 复现的预定义列表）。
- 交互项按原定义：`I_A = (A2−A0) − (A4−A3) = A2−A0−A4+A3`；`I_B = (B1−A0) − (B3−B2) = B1−A0−B3+B2`。

### 2.3 修正后的核心 Valid 表（repeat 122，3 seeds）

| contrast | corrected ΔValid RMSE | corrected ΔValid MAE | 旧错误值（如有） |
|---|---:|---:|---:|
| A2−A0 (NO_G) | **+0.008699** | +0.014598 | +0.077999 |
| A3−A0 (NO_ADD) | **−0.008750** | +0.026642 | +0.060550 |
| A4−A0 (POOL_ONLY) | **+0.054496** | +0.099394 | —（原表无此 predefined 行） |
| A5−A0 (ADD_ONLY) | **−0.012461** | +0.014903 | —（原表无此 predefined 行） |
| B0−A0 (TYPED_S2_OFF) | **+0.054173** | +0.152656 | +0.123473 |
| B4−A0 (MARGINAL_ALL) | **+0.007029** | +0.009469 | +0.076329 |
| B5−A0 (MARGINAL_BT2) | **+0.010716** | +0.016795 | +0.008964（124） |
| C1−A0 (NO_BOND_MEAN) | **−0.001279** | +0.000902 | −0.000575（124） |
| A4−A3 | **+0.063246** | +0.072752 | +0.063246（未受影响） |
| A4−A2 | **+0.045797** | +0.084796 | +0.045797（未受影响） |
| A_INTERACTION | **−0.054547** | −0.058154 | +0.014753 |
| B_INTERACTION | **−0.004907** | +0.007005 | +0.064393 |

A3/A5 的 per-seed Valid RMSE 差：

- A3: **+0.013903, −0.044468, +0.004314**
- A5: **+0.005611, −0.003041, −0.039952**
- C1: **+0.015743, −0.020612, +0.001033**
- B5: **−0.004873, +0.025957, +0.011064**

Test 标量示例（**仅暴露标量算术修正**，不是新的 Test 评估）：`A2−A0 +0.056930 → +0.019044`，`B0−A0 +0.217216 → +0.179331`，`C1−A0 +0.002955 → +0.001430`，`A_INTERACTION −0.029818 → −0.067703`；全部 15 行见 `corrected_contrasts_test_exposed.json`。

### 2.4 跨 repeat 核验

原 `cross_repeat.json` 的每个 `(repeat, variant)` cell 与 `main_runs.csv` 重算值逐字段一致；`cross_repeat_check.json` 共 45 个字段，max residual **0.0**，容差 1e-9。因此跨 repeat 表本身没有混 repeat bug；它的问题是**独立单位解释**，不是数值 bug。

C1/B5 的 same-seed、per-repeat 修正值（Valid RMSE）：

| contrast | 122 | 123 | 124 | 等权描述平均 |
|---|---:|---:|---:|---:|
| C1−A0 | −0.001279 | −0.001903 | −0.000575 | **−0.001252** |
| B5−A0 | +0.010716 | +0.018306 | +0.008964 | **+0.012662** |

这三个 repeat 共享同一批分子（只是划分不同），因此该平均不是独立数据估计；本轮**没有**为它构造新的 CI。

### 2.5 bootstrap point / anchor 核验

- 原 `analysis/bootstrap.json` 的 Valid contrast **point estimate**（36 个字段：17 个普通对比 × RMSE/MAE + 2 个交互 × RMSE；交互没有 MAE point）与修正后的 repeat-122 值一致，max residual **6.03e-8**（容差 1e-6）；`bootstrap_point_check.json` 记录 `all_points_agree=true`。
- 原 bootstrap 的 CI 是对 repeat-122 的 scaffold bootstrap；本轮**没有**重跑 CI，也没有把修正后的 Test 标量用于选择。
- 核心 anchor 检查（A2/A3/A4/A5/B0/B4/B5/C1 与 A4−A2/A4−A3/A_INTERACTION/B_INTERACTION）最大残差 **3.79e-8**；`anchor_checks.json` 为 `all_passed=true`。

### 2.6 勘误表（原位置 → 修正 → 依据 → 影响）

> 行号指源轮 `REPORT.md` / `MODEL.md` 在审查 commit 中的位置。

| # | 原位置 / 原说法 | 修正 | 依据 | 影响范围 |
|---|---|---|---|---|
| E1 | `analysis/tables/contrasts_valid.json`、`contrasts_test_exposed.json`、`markdown.md` 的 `contrasts_122`：`A2−A0=+0.077999`、`A3−A0=+0.060550`、`B1−A0=+0.080195`、`B2−A0=+0.089172`、`B0−A0=+0.123473`、`B4−A0=+0.076329`、`B5−A0=+0.008964`、`B6−A0=+0.083377`、`C0−A0=+0.097256`、`C1−A0=−0.000575`、`C2−A0=+0.084932`、`C3−A0=+0.081343`、`C4−A0=+0.094942`、`A_INTERACTION=+0.014753`、`B_INTERACTION=+0.064393` | 全部改为 same-repeat/same-seed 配对；见 §2.3 corrected 值和 `corrected_contrasts_valid.json` / `.csv` | `_seed_diffs` 的 `(variant, seed)` 后写覆盖；**11 个** 直接 A0@124 参照行统一偏移 **+0.0692998**；C1/B5 是 124 值；两个 interaction 由修正后的组件重算。Test 的 **11 个** 直接 A0 参照行偏移 **+0.0378856**，C1/B5 同样取自 124，interaction 同样重算 | **需替换上述两个 JSON、Markdown contrast 表及所有引用句**；不影响 run 记录、`cross_repeat.json`、bootstrap point/CI、Valid 预测、checkpoint、C1 选择 |
| E2 | REPORT §3.2 line 158：`A_INTERACTION = (A4−A3) − (A2−A0)`；REPORT §4.2 line 210：`B_INTERACTION = (B3−B1) − (B2−A0)` | `I_A = (A2−A0) − (A4−A3) = A2−A0−A4+A3`（数值仍 **−0.054547**）；`I_B = (B1−A0) − (B3−B2) = B1−A0−B3+B2`（数值仍 **−0.004907**）。原 B 公式给出的是 `−I_B` | 源轮 bootstrap JSON 的 label 本来就是 `A2-A0-A4+A3`；修正后数值与负号同时自洽；B 的 written formula 与表中 −0.0049 不相容 | Q1/Q3 的“负交互/近似可加”方向不变；两个公式与符号都必须改，不能把它们写成反向 |
| E3 | REPORT §0 Q2 line 23、§3.4 / §6.2：A4“同时删两者”/“去掉两条读出”/“reduce readout path almost free” | A4 **只删 g 与 f/L，保留 pooled readout `rho_h([mean_h, mean_h², log1p n])`**；A3/A5 各保留一条读出。`A4−A0=+0.054496`，`A4−A2=+0.045797`，`A4−A3=+0.063246`。A3/A5 的 RMSE 改善伴随 MAE 上升，且集中在 row 2087；不是“免费删读出” | `MODEL.md` A4 = POOL_ONLY；`REPORT.md` line 156/157 的 `A4−A2`/`A4−A3` 定义；B 的 sample-level 结果 §3 | Q2 的“冗余在读出、两条各自够用”只能保留为**条件性 performance tradeoff**；不能把 A4 当作“两个读出都删”的证据 |
| E4 | REPORT §2.1 line 93：`f_v = w·h_v + b` | `f` 是 **64→64→1 的 MLP**（4,225 参数，hidden SiLU），不是线性 `w·h+b`；`L=Σ_v f(h_v)` | `MODEL.md` line 58/120；checkpoint state dict 中 `f.0/f.2` 两层权重 | 任何把 `f` 当线性/纯加性参数化的解释都要撤回；A3/A5/C4 的差异不是“线性头 vs 池化”的简单对比 |
| E5 | REPORT §0 Q5 line 26、§2.1 line 96、§8.1 line 377：`pred = target_mean + Σ_v f(h_v) + L + ρ`；以及 `pred=target_mean+L+ρ` | 标准单位：`y = L + rho`，`L = Σ_v f(h_v)`；原始单位：`pred = μ + σ(L+rho) = μ + Σ_v c_v_raw + rho_raw`，`c_v_raw=σ f(h_v)`，`rho_raw=σ rho`。**不应再加一次 `Σ f`，也不应把标准单位 `L+rho` 与原始单位 `target_mean` 混写** | `MODEL.md` §7 line 191 的原始单位分解本来是正确的；`REPORT.md` line 400–416 的数值恒等式也按正确公式成立 | Q5 的公式必须改；原 attribution 数值/残差本身不需要重算，但“pred 的组成”表述要统一 |
| E6 | REPORT §0 Q1 line 22、§3.3 lines 163/165、`MODEL.md` §8 line 211：`g` 与 `h` “共享同一份解释力 / 同一份工作两条路径 / 信息高度重叠” | 数据支持的是**条件性 performance interaction**：`I_A<0`，即删除效应不按加法叠加；这**不等于**证明 `g` 与 `h` 承载同一份信息，也不等于同一潜在变量 | `A_INTERACTION` 是删除对比的交互，不是信息分解；原 `MODEL.md` 自己也说“不能解释为唯一物理贡献” | Q1 可保留“互为条件性替代/冗余路由”的 performance 说法；应删除“同一份信息”的本体论结论 |
| E7 | REPORT §0 Q2 line 23、§3.4、§6.2：A3/A5 “各自都够用”，只差 MAE 一条容差，RMSE 几乎免费 | A3/A5 的 RMSE **均值**改善分别 **−0.008750/−0.012461**，但 MAE 分别 **+0.026642/+0.014903**；A3 平均 232/420、A5 220.7/420 个样本 |error| 变大；RMSE 改善在排除 row 2087 后 3/3 seeds 变差 | B 的 paired decomposition（§3）；`REPORT.md` line 303 也承认“分布形状会变” | Q2 结论改为“RMSE/MAE tradeoff，且 RMSE gain 集中”；不能作为单读出可删/可购买的证据 |
| E8 | REPORT §4.3 line 217：B3(+0.036) 与 B4/B5(≤0.011) 的约 0.025 gap “由类型/末端/内容的边际分布提供”，暗示可把 B0−B5 当 marginal 的定量贡献 | B0 与 B5 是**不同控制**：B0 删整个 typed-S2 块（−70,592 参数），B5 保持参数不变、只做 marginal 重参数化；两者之差同时含容量、优化、参数化差异，不能分解成“marginal 的百分比”。可报告 corrected `B0−A0=+0.054173`、`B5−A0=+0.010716`，差 **+0.043457** 描述性存在，但不做因果归因 | `MODEL.md` B0/B5 定义；B4/B5 的 coverage/初始化 caveat（§8.4） | Q3 的“joint 本身几乎不值钱、marginal 是主要来源”只能作为**操作性描述**；不能给精确百分比或“marginal 单独贡献 0.025” |
| E9 | REPORT §0 Q4 line 25、§5.3 lines 251–256：scalars “必须留”、2nd-order “必须留”、closure “建议保留”、C0“也不能删” | 这些是**本轮购买规则/当前协议下的保留建议**，不是普遍必要性证明；C1 通过 0.010/0.020/0.010 是**实用性容差**，不是等价性检验（原报告 line 30/513 已有此 caveat，应贯穿 Q4 措辞） | 规则是四条件同时满足；n=3 seeds；C1 的 Valid −0.0013 不能证明信息无损失 | Q4 的 “must keep” 应软化为 “在本轮/本协议下不通过删除”；不外推到所有任务/模型规模 |
| E10 | `MODEL.md` §7 line 200：“无局部求和头的变体（A1/A3/A4/A5）不产生原子贡献图” | **A1/A3/A4 无局部求和头；A5 = ADD_ONLY 保留 `f`/`L`，只是没有 `rho`**，因此 A5 仍有逐原子贡献图（`b0` 为全图常数项） | `MODEL.md` line 133/173；A5 结构定义；`REPORT.md` Q5 也用 A5 的 `L+b0` | attribution/接口说明需改；不影响任何数值，但影响“哪些变体可做原子回溯”的表述 |
| E11 | REPORT §4.3 line 219、§8.4：用 `path_gate=1` 初始化 + 人工非恒定门探针（joint 0.587/content 1.000）说明“结构机会” | 人工探针是**结构机会**，不是训练后 checkpoint 的实际门控；本轮 C 直接测实际训练权重：path-weighted gate mean 1.038–1.075、sd 0.063–0.088、范围约 0.75–1.27；但只有 23.5% 的 multi 组组内 sd>0，content 替换输出影响小 | C 的 `gate_summary.json` / `summary.json`；§4.3 | Q3 可保留“初始化点 content coverage=0，因此 B4/B5 不能证明 content 无用”；但必须把**人工探针**与**训练后实际门控**分开，不能用前者替代后者 |

**没有发现需要勘误的其它相反表述**：源 `MODEL.md` 对 A3 的 “去掉 f/L，只用 rho” 与 A5 的 “只有 L+常数 b0” 在变体定义表中是正确的；只有 §7 line 200 的“无局部头变体列表”把 A5 错列进去。A2/A3/A5 的源 REPORT 主表数值（如 A5=0.6941、A3=0.6978、A4=0.7611）来自 repeat-122，未受 bug 影响。

### 2.7 受影响/不受影响文件清单

**必须替换（错误值）**

- `analysis/tables/contrasts_valid.json`（15 个 contrast 行）
- `analysis/tables/contrasts_test_exposed.json`（同 15 行，Test scalar only）
- `analysis/tables/markdown.md` 的 `contrasts_122` 表（同 15 行；其 bootstrap CI 列来自未受影响的 bootstrap，故会出现“point +0.0780 但 CI [−0.012,+0.026]”的内部不一致）

**保持原样（未受 bug 影响，已核验）**

- `analysis/main_runs.csv`、`analysis/runs/*`、`analysis/protocol_freeze.json`、`analysis/tables/cross_repeat.json`、`analysis/tables/variant_means.csv`、`size_bins.csv`、`analysis/bootstrap.json` 的 point/CI、`analysis/predictions/valid_*.npz`、`analysis/freeze/*`、`analysis/test_access_log.json`、所有 checkpoint。

`analysis/tables/markdown.md` 的 `main_122`、`means_122`、`cross_repeat` 表没有被 `_seed_diffs` 污染；`cross_repeat` 的 45 个字段已逐一对 `main_runs.csv` 核验。

---

## 3. Work B：误差集中度与 A3/A5 的 RMSE/MAE 分解

### 3.1 固定集合的定义（避免用候选结果挑样本）

- **per-seed FULL top-k**：只用 FULL 的 repeat-122 Valid |error| 排序，`k=1/5/10`；tie-break 为 row_id 升序；同一 seed 的候选变体复用该集合。
- **seed0 common top-k**：用 seed 0 FULL 排序得到的固定 row_ids，所有 seed 复用（敏感性检查）。
- **row 2087**：源报告 Q5 的 rank-0 样本，先验固定，不是从 A3/A5 结果中挑出的“outlier”。
- **top-|ΔSE| 显示表**：只用于贡献排序展示，不是评估子集；`top5_abs_delta_se_display.csv` 有明确说明。

### 3.2 每 seed 结果

| 变体 | seed | ΔRMSE | ΔMAE | row2087 ΔSSE | outside-row2087 ΔRMSE | AE 改善/恶化 | median ΔAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| A3 | 0 | +0.013903 | +0.057190 | −7.813 | **+0.035340** | 173 / 247 | +0.051954 |
| A3 | 1 | −0.044468 | +0.005214 | −32.647 | **+0.017993** | 202 / 218 | +0.010678 |
| A3 | 2 | +0.004314 | +0.017522 | −3.741 | **+0.015104** | 189 / 231 | +0.023874 |
| A5 | 0 | +0.005611 | +0.034208 | −14.181 | **+0.038354** | 192 / 228 | +0.025226 |
| A5 | 1 | −0.003041 | −0.005825 | −6.632 | **+0.011038** | 211 / 209 | −0.002043 |
| A5 | 2 | −0.039952 | +0.016327 | −40.711 | **+0.038216** | 195 / 225 | +0.012822 |
| C1 | 0 | +0.015743 | +0.020076 | +4.718 | +0.010466 | 185 / 235 | +0.024166 |
| C1 | 1 | −0.020612 | −0.009193 | −6.063 | −0.012806 | 228 / 192 | −0.005932 |
| C1 | 2 | +0.001033 | −0.008176 | +3.896 | −0.007833 | 225 / 195 | −0.012267 |
| B5 | 0 | −0.004873 | −0.000980 | −2.151 | −0.001610 | 213 / 207 | −0.005365 |
| B5 | 1 | +0.025957 | +0.015844 | +8.271 | +0.015412 | 199 / 221 | +0.009262 |
| B5 | 2 | +0.011064 | +0.035519 | −10.048 | +0.038962 | 176 / 244 | +0.036533 |

（Δ = candidate − FULL；正 = 更差。row2087 的 ΔSSE 是该单行对总 SSE 变化的贡献，不等于 RMSE 差的百分比。）

### 3.3 固定集合的敏感性（均值 over 3 seeds）

| 变体 | all ΔRMSE / ΔMAE | 排除 row2087 ΔRMSE / ΔMAE | 排除 per-seed top5 ΔRMSE / ΔMAE | 排除 per-seed top10 ΔRMSE / ΔMAE | AE 改善/恶化（平均） |
|---|---:|---:|---:|---:|---:|
| A3 | −0.008750 / +0.026642 | **+0.022812 / +0.028816** | +0.038671 / +0.033928 | +0.048796 / +0.038144 | 188.0 / 232.0 |
| A5 | −0.012461 / +0.014903 | **+0.029203 / +0.017457** | +0.034322 / +0.018995 | +0.040013 / +0.020784 | 199.3 / 220.7 |
| C1 | −0.001279 / +0.000902 | −0.003391 / +0.000847 | +0.002634 / +0.002906 | +0.005386 / +0.003888 | 212.7 / 207.3 |
| B5 | +0.010716 / +0.016795 | +0.017588 / +0.016933 | +0.021898 / +0.018045 | +0.025503 / +0.019245 | 196.0 / 224.0 |

关键点：

- A3/A5 的 outside-row2087 **3/3 seeds 都是正**（更差）；排除 top-5/top-10 后仍然正且更大。RMSE 的负均值完全依赖 row 2087 的极端 SSE 降低。
- A3 的 MAE 代价是 3/3 seeds、全样本的（median ΔAE 均为正）；A5 的 MAE 均值被 seed 0/2 拉高，seed 1 略好，但整体仍是 broad cost（220.7/420 恶化）。
- C1 没有 row2087 集中模式：ΔRMSE 符号跨 seed 混合，outside-row2087 均值 −0.003391（略好），MAE 约 0.0009（几乎中性）。
- B5 是“均值更差、但不是单行集中”的模式：ΔRMSE 在 2/3 seeds 为正；outside-row2087 均值为 **+0.017588**（seed 0 仅 −0.001610，seed 1/2 为正）；AE 224/420 恶化。它和 A3/A5 的“RMSE 被一个极端行改善、MAE 分散变差”不是同一种模式。

### 3.4 row 2087 的具体诊断

- `row_id=2087`，位置 `212`（0-based），SMILES `CC(N)(COP(=O)(O)O)C(=O)O`，`y=2.5`。
- FULL seed 0 预测 `−7.021600776`，`|e|=9.521600776`；FULL seed 1/2 也把它排为 |error| 第 1（|e|≈8.94/11.13），说明它不是 seed 0 的偶然尾部。
- 该分子同时含磷酸/羧酸片段；本报告**不**据此做化学归因，只把它当作固定样本诊断。
- A3/A5 在三个 seed 上都降低其 SSE；C1/B5 的 ΔSSE 跨 seed 变号（C1 +4.72/−6.06/+3.90；B5 −2.15/+8.27/−10.05），进一步说明 row2087 不是所有简化变体的共同“修复点”。

### 3.5 回答 Q2（限定版）

1. **发生样本**：A3/A5 的 RMSE 均值改善主要发生在 row 2087；排除它后，两个变体在 3/3 seeds 上都更差。固定 top-5/top-10 的排除结果同向且更大。
2. **是否集中在少数极端误差**：是。row 2087 一行就占 FULL seed 0 总 SSE 的 43.96%（seed 1/2 为 41.01%/54.21%）；A3/A5 的 SSE 改善足以把总 RMSE 均值拉负。
3. **MAE 是否广泛变差**：A3 是（232/420，3/3 seeds median 正）；A5 是总体上（220.7/420，2/3 seeds median 正，seed 1 轻微例外）。所以“RMSE 改善、MAE 恶化”不是矛盾，而是**平方误差被极端高估主导、绝对误差被大量小误差主导**的算术表现。
4. **C1/B5 是否类似**：不。C1 小且 seed 混合；B5 的均值 worse（2/3 seeds），但不是“单个极端行改善 + 其余变差”的集中模式。

---

## 4. Work C：实际训练权重下的 J/C 分离替换

### 4.1 对象与条件

- 加载 **6 个已经由 Valid RMSE 选出的 repeat-122 checkpoint**：A0（FULL）与 C1（COMPACT），各 seeds 0/1/2；**没有重训、没有重选 epoch、没有改 checkpoint**。
- `JCLocalContent` 是冻结 `M.LocalContent` 的子类，只增加两个独立聚合开关：
  - `jc_joint_agg="native"` 或 `"marginal_bt2"`（J）
  - `jc_content_agg="native"` 或 `"marginal_bt2"`（C）
- 每个 checkpoint 评估 **NATIVE / J_ONLY / C_ONLY / BOTH** 四个条件；共 **24** 次 condition 评估。
- `eval()`、`torch.no_grad()`、CPU；输入是 repeat-122 Valid 全 420 行；没有 Test。

### 4.2 实现等价性/parity

- NATIVE 重放 vs 原保存预测：max |Δpred| **1.30e-6 … 2.16e-6**（容差 1e-4）；ΔRMSE ≤ 6.74e-8，ΔMAE ≤ 2.58e-8。
- BOTH（两个都换成 marginal_bt2）vs 原轮 `s2_agg_override="marginal_bt2"`：max |Δ| **0.0**（每个 checkpoint）。
- state dict hash、batch hash 在评估前后不变；单开关条件中，未切换的 T/E/K（以及 J_ONLY 的 C、C_ONLY 的 J）max |Δ| = **0.0**。
- 5 个代数检查全部通过：singleton = full、empty group = 0、joint bias 计 `n_g` 次、content 常数门等于 marginal、两个分子不跨组；`implementation_checks.json` 为 `all_passed=true`。

### 4.3 gate 实际统计（Valid 420 行）

| 指标 | A0 seed 0/1/2 | C1 seed 0/1/2 |
|---|---|---|
| path-weighted gate mean | 1.0601 / 1.0699 / 1.0752 | 1.0384 / 1.0634 / 1.0635 |
| path-weighted gate sd | 0.0872 / 0.0807 / 0.0880 | 0.0635 / 0.0819 / 0.0851 |
| min / max | 0.917–1.222 / 0.751–1.237 / 0.833–1.266 | 0.882–1.190 / 0.788–1.266 / 0.845–1.233 |

- 结构计数（每个 checkpoint 相同）：57,325 个 `(root, bt2)` 组；**39,471 空 / 5,320 singleton / 12,534 multi**；multi 组路径数中位 2、最大 6。
- 只有 **23.5%** 的 multi 组组内 gate sd > 1e-12；multi 组 sd 的 median = 0，p95 ≈ 0.09–0.13，max ≈ 0.13–0.17。path-weighted sd 0.06–0.09 与“组内变异稀疏、跨组差异存在”一致；不能把它当作组内逐路径差异的度量。
- 因此实际训练门控**不是常数**，但内容替换能起作用的组内变异很稀疏；这与 C 的小输出效应一致。

### 4.4 内部向量变化（相对 NATIVE，root-level）

覆盖定义：该 root 的 J 或 C 子向量在替换前后满足 `maxabs(Δ) > 1e-6 + 1e-5·max(scale)`（`coverage_threshold_rule`）；不是简单的“非零元素”。

| 统计 | J_ONLY（A0/C1 全部 6 个 checkpoint） | C_ONLY（全部 6 个） |
|---|---:|---:|
| changed roots | **1,632 / 11,465 (14.23%)** | **2,566 / 11,465 (22.38%)** |
| changed molecules | **389 / 420 (92.62%)** | **410 / 420 (97.62%)** |
| median \|Δ\| | ~7e-7 | ~1.3e-7 |
| p90 \|Δ\| | 3.14–3.44 | 0.11–0.18 |
| p95 \|Δ\| | 4.03–4.80 | 0.16–0.28 |
| max \|Δ\| | 9.56–13.38 | 0.39–0.77 |

- J 的变化集中在少数 roots，但幅度很大；C 的变化覆盖更多 roots/molecules，但幅度小一个数量级。
- 逐 root 的 Δh（LocalMLP 输出）也被记录：J_ONLY mean 0.269–0.319、p95 2.15–2.56、max 6.21–7.54；C_ONLY mean 0.029–0.054、p95 0.17–0.33、max 1.43–2.19；BOTH 与 J_ONLY 接近。

### 4.5 输出响应（24 个 condition 的测量值）

NATIVE 指标（RMSE / MAE）：

| checkpoint | A0 | C1 |
|---|---:|---:|
| seed 0 | 0.700750 / 0.407941 | 0.716493 / 0.428017 |
| seed 1 | 0.681180 / 0.415705 | 0.660568 / 0.406512 |
| seed 2 | 0.737828 / 0.396013 | 0.738861 / 0.387837 |

条件响应（Δ = condition − NATIVE；所有值都是 same-checkpoint、same-Valid-420）：

| checkpoint | J_ONLY ΔRMSE / ΔMAE | C_ONLY ΔRMSE / ΔMAE | BOTH ΔRMSE / ΔMAE |
|---|---:|---:|---:|
| A0 s0 | +0.023837 / +0.025080 | +0.000821 / +0.001046 | +0.023625 / +0.024745 |
| A0 s1 | +0.006211 / +0.019808 | +0.001479 / +0.002670 | +0.005177 / +0.019311 |
| A0 s2 | +0.019051 / +0.026710 | +0.000214 / −0.000997 | +0.019873 / +0.026916 |
| C1 s0 | +0.009182 / +0.011255 | +0.001309 / +0.002250 | +0.010209 / +0.012099 |
| C1 s1 | +0.000525 / +0.014123 | +0.002487 / +0.002848 | +0.000646 / +0.014450 |
| C1 s2 | +0.015053 / +0.022505 | +0.000443 / −0.000561 | +0.015978 / +0.023699 |

（ΔMAE 来自 `summary.json` 的逐 checkpoint 值；C_ONLY 的 ΔRMSE 6/6 为正，但 ΔMAE 有 2 个负值，均值仍为 +0.001209。）

- **J_ONLY**：ΔRMSE 均值 **+0.01231**，6/6 变差，范围 +0.000525…+0.023837；ΔMAE 均值 **+0.019914**。
- **C_ONLY**：ΔRMSE 均值 **+0.001125**，6/6 变差，范围 +0.000214…+0.002487；ΔMAE 均值 **+0.001209**，但逐 checkpoint 为 +0.001046/+0.002670/−0.000997/+0.002250/+0.002848/−0.000561（2 个轻微负值，MAE 效应是 mixed/近中性）。
- **BOTH**：ΔRMSE 均值 **+0.012585**，6/6 变差，范围 +0.000646…+0.023625；ΔMAE 均值 **+0.020203**。
- **A0 vs C1**：J_ONLY 均值 A0 **+0.01637**、C1 **+0.00825**；C_ONLY 均值 A0 **+0.00084**、C1 **+0.00141**。C1 的 J 敏感度更低，但方向一致；C1 的 C 敏感度与 A0 接近。
- **BOTH vs J_ONLY**：均值差很小（RMSE +0.000275、MAE +0.000289），但逐样本不是严格可加；非线性残差 mean |r| 0.00385–0.00555、max 0.0505–0.1036。

### 4.6 回答 Q3（限定版）

1. **J 与 C 必须分开**：J_ONLY 的内部变化幅度（p95 4.03–4.80，max 9.56–13.38）和输出变化（ΔRMSE 均值 +0.01231）都远大于 C_ONLY（p95 0.16–0.28，max 0.39–0.77；ΔRMSE 均值 +0.001125）。
2. **C 不是“无用”**：6/6 checkpoint 的 C_ONLY 都使 Valid RMSE 略微变差，内部表示覆盖 2,566 roots / 410 molecules；只是实际训练门控在组内变化稀疏，输出敏感度低。
3. **J 的损伤也不是灾难性的**：J_ONLY 在 6/6 checkpoint 都变差，但 C1 seed 1 只有 +0.000525；原轮 B5（J+C 一起换成 marginal）为 +0.010716，方向相同、量级同阶，但 magnitude 依赖 seed/weights。
4. **不能把两者相加当精确分解**：BOTH 接近 J_ONLY 但非线性残差 max 0.05–0.10；这是干预的非可加性，不是“J 占 90%、C 占 10%”的方差分解。
5. **不能外推为可删除性**：全部是冻结权重、Valid 420、3 seeds；没有重训、没有 Test、没有跨数据集验证。

---

## 5. 事实 / 受限解释 / 未回答

### 5.1 事实（可直接复核）

- 跨 repeat bug 的键覆盖机制可复现；Valid 偏移 **+0.0692998**，Test 标量偏移 **+0.0378856**。
- 15 个 contrast 行在 `contrasts_valid.json` / `contrasts_test_exposed.json` / `markdown.md` 中错误；修正表与 anchor/bootstrap/cross-repeat 核验通过。
- A3/A5 的 RMSE 均值改善依赖 row 2087；排除后 3/3 seeds 变差；MAE 代价分散。
- 6 个 checkpoint 上 J/C 替换的 coverage、Δh、Δpred、ΔRMSE 均已逐 checkpoint 记录；BOTH 与原 `marginal_bt2` 完全一致。
- 新训练 0、新 Test 0、新服务器/付费资源 0；CPU-only。

### 5.2 受限解释（可以说，但要带条件）

- “J 替换比 C 替换影响大”是**当前 6 个训练权重、Valid 420、这组干预**的条件性结论，不是结构必要性。
- “A3/A5 的 RMSE 改善来自极端样本”是**样本分解**结论；不说明模型在其它数据上也会这样，也不说明 MAE 代价不重要。
- “C1 通过实用性容差”是购买决策；不是“与 FULL 等价”，也不是“信息无损失”。
- “训练后 gate 不是常数”是描述性统计；不能据此说 gate 学到了化学选择性。
- 原轮 `g`/`h` 的负交互可以解释为**条件性 performance 替代**；不能说它们共享同一份信息。

### 5.3 未回答

- 重训 A3/A5/单读出模型后，RMSE/MAE tradeoff 是否保持？需要新训练。
- row 2087 是标签噪声、分布外样本、还是模型系统性失败？需要标签审计/外部数据，不能从现有 Valid 单独判定。
- C1 在更多 seed/repeat/数据规模下是否仍通过容差？本轮只有 3 seeds、共享分子的 3 个划分。
- J/C 对 Test 泛化、下游化学任务、不同模型规模的影响；本轮不读 Test 样本级数据。
- J/C 内部的路径级机制（哪些路径/分子驱动输出变化）；本轮只有 root/molecule 级聚合与固定样本，不做因果路径归因。

---

## 6. 复现、测试与文件

### 6.1 命令（从本目录运行）

```bash
cd tracks/structural-sufficiency/lcrc_lipo_model_dissection_followup_v1

# A: corrected tables + legacy reproduction + anchor/bootstrap/cross-repeat checks
uv run python code/corrected_tables.py

# B: per-sample error decomposition and fixed hard sets
uv run python code/error_concentration.py

# C: frozen J/C interventions on 6 selected checkpoints (24 conditions)
uv run python code/frozen_jc_interventions.py

# regression tests (9)
uv run pytest code/test_followup.py -q

# input manifest / hash audit
uv run python code/build_input_manifest.py
```

本次实测（CPU）：A **1.34 s**，B **1.53 s**，C **3.83 s**，tests **1.9 s**（pytest 内部计时），manifest **1.62 s**；合计 **约 10 s**。C 的 24 次 condition 评估不涉及反向传播/优化器。

### 6.2 测试与核验

- `code/test_followup.py`：**9 passed**（~1.9 s），覆盖 identity、same-repeat 配对、legacy 复现、anchor、cross-repeat、bootstrap point、预测对齐、row2087、J/C parity。
- `anchor_checks.json`：`all_passed=true`，max residual 3.79e-8。
- `bootstrap_point_check.json`：36 个 point 字段（17 个普通对比 × RMSE/MAE + 2 个交互 × RMSE），max residual 6.03e-8，`all_points_agree=true`。
- `cross_repeat_check.json`：45 个字段，max residual 0.0，`all_fields_match=true`。
- `input_manifest.json`：missing=0，72 identities，源树与 commit 一致=true，Test 输入=0；两个 freeze 后 code hash 差异已列出。

### 6.3 关键产物

| 文件 | 内容 |
|---|---|
| `analysis/corrected_tables/corrected_contrasts_valid.json` / `.csv` | 所有预定义对比 + A4−A0/A5−A0 的 same-repeat/same-seed 修正值、per-seed、missing、cross-repeat 描述平均 |
| `analysis/corrected_tables/corrected_contrasts_test_exposed.json` / `.csv` | 仅 Test RMSE/MAE 标量的算术修正；`EXPOSED_TEST_SCALAR_DIFF_ONLY` |
| `analysis/corrected_tables/legacy_affected_differences.json` | 原 bug 复现值、旧−新差、错误来源 `(candidate_repeat, reference_repeat, seed)` |
| `analysis/corrected_tables/anchor_checks.json` | 核心 anchor 对照 |
| `analysis/corrected_tables/cross_repeat_check.json` | 原 cross_repeat 对 main_runs 的 45 字段核验 |
| `analysis/corrected_tables/bootstrap_point_check.json` | 原 bootstrap point 对修正值的 36 字段核验 |
| `analysis/error_concentration/paired_samples.csv` | 420 行 × seed × variant 的 y/pred/error/ΔSSE/ΔAE |
| `analysis/error_concentration/error_decomposition.json` | 每变体每 seed 的 ΔRMSE/ΔMAE/gross gain/loss/median ΔAE + 固定集合 |
| `analysis/error_concentration/hard_sets.json` | 固定 FULL top1/5/10、seed0 common sets、row2087 |
| `analysis/error_concentration/summary.json` | B 的汇总 |
| `analysis/trained_jc/summary.json` | 6 checkpoint × 4 condition 的 Δpred/Δh/gate/parity/非线性残差 |
| `analysis/trained_jc/gate_summary.json` | gate 均值/sd/分位/组计数（路径加权，非未使用词表） |
| `analysis/trained_jc/per_molecule_conditions.csv` | 420 行 × 6 checkpoint × 4 condition 的预测 |
| `analysis/trained_jc/implementation_checks.json` | 5 个代数检查 |
| `analysis/trained_jc/failures.json` | 空列表（无失败 condition） |
| `input_manifest.json` | 输入清单、hash、Test 排除、协议差异决策 |

### 6.4 边界声明

- 本轮没有训练、微调、epoch 重选或超参选择；没有新 baseline；没有外部文献/SOTA；没有新服务器/付费资源；没有 push。
- 没有把 outlier-removed 指标当作新的主结果；row 2087 的排除分析只作为**诊断**，并明确标注。
- 没有修改源轮文件、checkpoint、predictions、freeze、bootstrap、Test access log；没有修改 `STATE.yaml`。
