# LCRC-A / Lipophilicity 模型理解与简化 — 报告

- **Round**: `lcrc-lipo-model-dissection-v1`
- **对象**: 冻结的 LCRC-A Lipophilicity 模型（pilot v1 `Lipophilicity_s0/checkpoint.pt`，SHA256 `b1dd1f5a…e2663`，247,106 参数）
- **公开训练划分**: `Lipop.random122.ind.pkl`（3360/420/420）
- **冻结 recipe**: R3（AdamW，peak LR 3e-4 → min 1e-5 per-step cosine，wd 1e-3，batch 128，dropout 0.10 on `h`，grad clip 5.0，360 epoch，无 EMA/SWA/soup/SSL/早停，best-Validation checkpoint）
- **正式训练运行**: 54（public122 全矩阵）+ 18（跨划分）= 72，等于冻结上限 `max_formal_runs=72`
- **GPU 预算**: soft 6.0 h / hard 8.0 h，实耗 **6.54 h**（见 §9；超 soft 0.54 h，由预注册的跨划分采购造成，hard 内）
- **最终状态**: **`COMPLETE`**；简化候选 **`COMPACT = C1 (NO_BOND_MEAN)`**，`status = SIMPLER_CANDIDATE_SELECTED`（只读 Valid 选出，Test 一次性读取后确认）

> 本报告回答冻结 prompt 的 Q1–Q5，并交付一个经过审查的简化候选。
> 所有 Test 数字标为 **`EXPOSED_TEST_EXPLORATORY`**：每个 checkpoint 只读一次，从不参与任何选择。
> 所有"购买"判定使用冻结的**实用性容差**（mean ΔR ≤ 0.010、max ΔR ≤ 0.020、mean ΔA ≤ 0.010），
> 这是"可接受的简化"判据，**不是**等价性检验。

---

## 0. 结论摘要

| 问题 | 结论（一句话） | 关键数字（122 Valid，3 seeds） |
|---|---|---|
| **Q1** `g` vs `h` 的信息价值 | 二者互为替代：**有加性头时 `g` 几乎不值钱**（+0.009），**没有加性头时 `g` 值 +0.063**；交叉项 −0.0545（CI 不含 0）说明这不是两份独立信息，而是**同一份工作的两条路径** | A2−A0 = +0.0087；A4−A3 = +0.0632 [+0.030,+0.108]；A_INTERACTION = −0.0545 [−0.103,−0.018] |
| **Q2** 加性头 vs 池化读出 | **两条读出各自都够用**：只留加性求和（A5 = `L+b0`）−0.012；只留池化读出（A3 = `ρ`, 无 `f`/`L`）−0.009；**同时删两者**才 +0.046。冗余在**读出**，不在**输入**：预测由 `h`（529 维局部内容）承载 | A5 = 0.6941；A3 = 0.6978；A4 = 0.7611；A1（只有 `g`，2,817 参数）= 1.0120 |
| **Q3** S2 joint/content vs marginals | 整个 typed-S2 块值 +0.054（B0）；其中 **joint 配对本身几乎不值钱**（换成边际乘积 B4/B5 只 +0.007/+0.011，参数不变），**content/type/end 的边际量**才是那 +0.02–0.03 的来源 | B0−A0 = +0.0542；B4−A0 = +0.0070；B5−A0 = +0.0107；B3−A0 = +0.0357 |
| **Q4** S1-dup / scalars / 2nd-order / closure 留谁 | 只有 **S1-dup（`b_mean = b_sum/deg`）可以删**（−18,432 参数，−0.0013）；scalars（+0.016）、2nd-order pooling（+0.012，且 sum→mean 是 +0.026）、closure（+0.014）都必须留 | C1 = 0.7053（唯一合格）；C2 +0.0156 [+0.002,+0.026]；C3 +0.0120；B6 +0.0141；C4 +0.0256 |
| **Q5** 局部统计量能否回溯到预测 | 能，而且可逐项核对：`pred = target_mean + Σ_v f(h_v) + L + ρ`，删掉一条二跳路径后**只有根的 `f` 与 `ρ` 变化，其他原子的 `f` 变化严格为 0**，恒等式残差 ≤ 2.9e-7 | 例：row 2087（rank 0）删 `10O→9→11O (2→1)` 路径时 ΔL=+0.520、Δρ=+0.639、Δpred=+1.160（−7.022 → −5.862） |

**交付物**：`COMPACT = C1 (NO_BOND_MEAN)` = FULL 去掉 `b_mean` 这一 96 维"入射键和的度数归一化副本"，
参数 247,106 → **228,674（−7.5%）**，Valid RMSE −0.0013（3 seeds，scaffold bootstrap CI [−0.014,+0.008]），
Valid MAE +0.0009，是一次**在实用性容差内的简化**；它不宣称"与 FULL 等价"，也不宣称"信息上无损失"。

---

## 1. 数据、实现与可复现性

### 1.1 只读来源

| 来源 | 冻结标识 | 本轮用途 |
|---|---|---|
| ChemBench `Lipop` 原始图 | commit `0b10d4c1ac72b6929503694a5f60446824736391` | 唯一结构来源（标签无关 RDKit 图） |
| `Lipophilicity.csv` | SHA256 `34280f0a…cd758`，n=4200 | 仅标签列 `exp`；Test 标签只在 score 阶段读取 |
| `Lipop.random{122,123,124}.ind.pkl` | 行索引 SHA256 见 `analysis/protocol_freeze.json` | 划分唯一来源（3360/420/420，三划分互不重叠、并集=4200） |
| pilot v1 checkpoint | `b1dd1f5a…e2663`（`lcrc_molnet_benchmark_pilot_v1/analysis/runs/Lipophilicity_s0/`） | 冻结参照；只用于等价性 replay，不参与选择 |
| 冻结上游代码 | `lcrc_cross_task_preparation_v1/code/{feature_spec,lcrc_data,lcrc_model}.py` | 只读引用（哈希登记在 `analysis/protocol_freeze.json`） |

标签隔离：`dissect_data.py` 的 Valid/Train 视图**不含 Test 标签**（`analysis/audit_cpu.json → selection.view_no_test`：
`has_y_test=false`、`test_targets_loaded=false`、`test_structure_cache_built=false`）；归一化只用 Train。

### 1.2 实现等价性（不是"看起来一样"，是逐位核对）

| 检查 | 结果 |
|---|---|
| FULL 初始化 state 与冻结上游 | **bitwise identical**（35 个张量，`max_abs_diff=0.0`） |
| 冻结 checkpoint replay（Valid RMSE） | 0.7108803968 vs 记录值 0.7108803968，\|Δ\|=3.47e-9；与当年 prediction 的 `max_abs_diff=0.0` |
| 同一 batch 的 full forward：`y`、`h`、`rho`、`contrib_atom`、`mean_h`、`mean_h2` 等 auxiliary | `max_abs_diff = 0.0`（逐项） |
| 反向：共享参数的梯度 | 与冻结上游逐项相同（`tests/test_dissection.py`） |
| dropout | 每 forward 恰好调用一次；A1 无 dropout |
| 18 个变体参数计数 | 全部等于 `EXPECTED_PARAMS`（见 §2.3） |
| 变体共享参数 | 与 FULL 的对应列/行**逐位相等**（54/54 run 记录 `shared_init_bitwise_equal_to_full=True`） |
| 每 seed 每 stage 的 batch 顺序 | 同 seed 内所有变体**完全一致**（`analysis/freeze/*.json → batch_order_checks`） |
| 原子重编号不变性 / batch split-merge / 空 S2 / 反标准化恒等式 | 全部通过（`analysis/audit_cpu.json`，**62 项检查**，`ok=true`） |
| 单元测试 | **32 passed**（`tests/test_dissection.py`） |

`runs/`、`cache/` 不进 Git（`.gitignore`）；每 run 的 `train.json` / `checkpoint.pt` / `*_predictions.npz` 留在
本地与远端磁盘，可复核的摘要写入 `analysis/`。

### 1.3 训练与评估规则（未改动）

- 每个 run：`fit` = Train 3360，`monitor` = Valid 420；best-epoch 按 **Valid RMSE 严格改进**，平局保留更早 epoch。
- 每 run 9720 optimizer steps（54/54 run 一致，`optimizer_steps=9720`）。
- 目标在 Train 标准化空间做 MSE，报告一律反标准化回 logD。
- 参考点：Train 均值常数预测器的 **Valid RMSE = 1.1559、Test RMSE = 1.1991**（用于判断"是否学到东西"）。

---

## 2. 模型在算什么（计算图摘要，完整版见 `MODEL.md`）

### 2.1 一条前向的完整路径

```
atom_feat(51) --AtomEncoder--> a_v(96)              bond_feat(13) --BondEncoder--> b_e(96)
S1:  b_sum_v = Σ_{e∋v} b_e                       (96)
     b_mean_v = b_sum_v / deg(v)                 (96)   <= b_sum 的度数归一化副本
     nbr_v = Emb_nbr(atom_type(u)*5 + (bt_e-1))   (32)   一跳类型联合项
S2 (二跳枚举, 200 维):
     T_v = Σ_p Emb_type(path_cat_p)               (32)
     E_v = Σ_p Emb_end (path_end63_p)             (32)
     J_v = Σ_{p in v} joint_proj(path_u ⊙ path_v) (32)   <= 逐次配对的联合投影
     K_v = Σ_p Emb_close(path_closure_p)          (8)
     C_v = Σ_p [path_gate ⊙ a_end]                (96)   <= 门控*末端原子表示
     x_v = [a_v, b_sum, b_mean, nbr, T, E, J, K, C, scalars(9)]   (529)
LocalMLP: x_v(529) -> 192 -> 64 (LayerNorm, dropout 0.10) = h_v
Head:     f_v = w·h_v + b                         (1)    <= 原子贡献
          L   = Σ_v f_v                                   <= 加性求和读出
Output:  ρ = MLP(mean_v h_v, mean_v h_v², log1p n, g(42))  (1)   <= 池化读出
         pred = target_mean + L + ρ                (原始单位)
```

### 2.2 各分支在做什么（由实验确认，不是由结构推断）

- **typed-S2 线性块（T/E/J/K/C → `L`）是主干**：把它整块（200 维）置零，固定权重下的 Valid 子集 RMSE 从
  0.455→1.726（seed 0）、0.508→2.083（seed 1）、0.424→2.761（seed 2）（§8.1）。
- **LocalMLP + `f` 是"逐原子非线性重写"**：单独去掉它（A3）不掉分；但当 `g` 也去掉时（A4）掉 +0.046 —— 说明二者共享同一份解释力。
- **池化读出 `ρ` 是与加性头和 `g` 并列的另一种读出**：它的输入是 `h` 的一阶、二阶矩与 42 维全图统计，
  输出量级（sd ≈ 0.6–0.8）与贡献和（sd ≈ 0.8）相当。它不是"残差通道"——单独保留它就能复现 FULL（A3）。
- **`b_mean` 是可计算的重复量**：`b_sum/deg`，度信息已由 `atom_feat` 与标量块提供；本轮的 C1 实验确认它可删。

### 2.3 参数与变体账本（全部经 `EXPECTED_PARAMS` 校验）

| 变体 | 名称 | 参数 | Δ vs FULL | 说明 |
|---|---|---|---|---|
| A0 | FULL | 247,106 | — | 冻结模型 |
| A1 | G_ONLY | 2,817 | −244,289 | 只留 `g → 64 → 1` |
| A2 | NO_G | 244,418 | −2,688 | `ρ` 不再读 42 维 `g` |
| A3 | NO_ADD | 242,881 | −4,225 | 去掉 `f` 与加性求和 |
| A4 | POOL_ONLY | 240,193 | −6,913 | 无 `f`、无 `g` |
| A5 | ADD_ONLY | 236,034 | −11,072 | `L + b0`，无 `ρ` |
| B0 | TYPED_S2_OFF | 176,514 | −70,592 | 整体去 200 维 typed-S2 |
| B1 | NO_JOINT | 224,546 | −22,560 | 去 32 维 `J` 与其投影 |
| B2 | NO_CONTENT | 228,274 | −18,832 | 去 96 维 `C` 与 `path_gate` |
| B3 | NO_JOINT_CONTENT | 205,714 | −41,392 | 去 `J` + `C` |
| B4 | MARGINAL_ALL | 247,106 | 0 | 同参数，`J`/`C` 换成每根边际乘积 |
| B5 | MARGINAL_BT2 | 247,106 | 0 | 同参数，按 `(root, bt2)` 分组的边际乘积 |
| B6 | NO_CLOSURE_CHANNELS | 245,362 | −1,744 | 去 8 维 `K` 与 closure 标量 |
| C0 | NO_S1JOINT | 238,402 | −8,704 | 去 32 维 `nbr` 联合项 |
| C1 | NO_BOND_MEAN | 228,674 | **−18,432** | 去 `b_mean`（保留 `b_sum`） |
| C2 | NO_SCALARS | 245,378 | −1,728 | 去 9 维 `atom_scalar` |
| C3 | NO_M2 | 243,010 | −4,096 | `ρ` 不再读 `mean_h²` |
| C4 | MEAN_ADD | 247,106 | 0 | `L = mean_v f` 而非 `Σ_v f` |

---

## 3. Stage A：`g` 与 `h` 的价值，加性头 vs 池化读出（Q1、Q2）

冻结规则：先冻结 Valid（18 run），再一次性读 Test。

### 3.1 主表（Valid 与 Test，3 seeds）

| 变体 | 参数 | **Valid RMSE** | Valid MAE | **Test RMSE** `EXPOSED_TEST_EXPLORATORY` | Test MAE | best epoch |
|---|---|---|---|---|---|---|
| A0 FULL | 247,106 | **0.7066 ± 0.0288** | 0.4066 | **0.5770** | 0.4246 | 275–331 |
| A1 G_ONLY | 2,817 | 1.0120 ± 0.0031 | 0.7935 | 1.0774 | 0.8578 | 222–318 |
| A2 NO_G | 244,418 | 0.7153 ± 0.0223 | 0.4212 | 0.5960 | 0.4440 | 133–174 |
| A3 NO_ADD | 242,881 | 0.6978 ± 0.0547 | 0.4332 | 0.5856 | 0.4467 | 62–184 |
| A4 POOL_ONLY | 240,193 | 0.7611 ± 0.0139 | 0.5059 | 0.6723 | 0.5206 | 22–235 |
| A5 ADD_ONLY | 236,034 | 0.6941 ± 0.0145 | 0.4215 | 0.5980 | 0.4469 | 158–248 |

（常数预测器：Valid 1.1559 / Test 1.1991。）

### 3.2 四角对比与交叉项

| 对比 | ΔValid RMSE | 逐 seed | scaffold bootstrap 95% | 跨 0 |
|---|---|---|---|---|
| A2 − A0（去掉 `g`，保留 `f`） | **+0.0087** | +0.038, +0.013, −0.025 | [−0.0123, +0.0256] | 是 |
| A3 − A0（去掉 `f`，保留 `g`） | **−0.0088** | +0.014, −0.044, +0.004 | [−0.0574, +0.0425] | 是 |
| A4 − A2（在"无 `g`"上再去 `f`） | **+0.0458** | +0.027, +0.051, +0.059 | [−0.0328, +0.1256] | 是 |
| A4 − A3（在"无 `f`"上再去 `g`） | **+0.0632** | +0.051, +0.109, +0.030 | **[+0.0303, +0.1078]** | **否** |
| A_INTERACTION = (A4−A3) − (A2−A0) | **−0.0545** | −0.013, −0.095, −0.055 | **[−0.1029, −0.0177]** | **否** |

### 3.3 回答 Q1：`g` 的信息价值是**条件性**的

1. 单独看边际：**有 `h` 的时候 `g` 几乎不值钱**（+0.0087，CI 跨 0）；**去掉 `h` 的加性头之后 `g` 值 +0.063**（CI 不含 0）。
2. 交叉项显著为负（−0.0545），意味着两个删除**远非相加**：`g`（42 维全图统计）与 `f`（逐原子非线性读数）在功能上**共享同一份解释力**，不是两份独立信息。
3. A1（只有 `g`，2,817 参数）Valid 1.012 / Test 1.077，仍优于常数预测器（1.156/1.199），R²≈0.23 —— 全图统计**确实含任务信号**，但远远不够。
4. 因此不能说"`g` 无用"。正确的说法是：**在完整模型里，`g` 是一条可被局部路径替代的冗余路由；一旦局部加性路径被移除，它就成为主力。** 这同时解释了为什么"删除单个分支不掉分"不等于"该分支不含信息"。

### 3.4 回答 Q2：两条读出各自都够用，冗余在**读出**而不在**输入**

- **只留加性读出**（A5 = `L + b0`，无任何池化项）：Valid 0.6941（比 FULL **低** 0.012），Test 0.5980（+0.021）。
- **只留池化读出**（A3 = `ρ(mean_h, mean_h², log1p n, g)`，无 `f`、无 `L`）：Valid 0.6978（−0.009），Test 0.5856（+0.009）。
- **同时删掉两条读出**（A4 = 只有 `ρ(mean_h, mean_h², log1p n)`，没有 `f`/`L`、也没有 `g`）：Valid **+0.046**，Test +0.095。
- 结论：**加性头与池化读出互为替代，而不是"主路径 + 残差"**。两者都是同一个 `h` 的非线性泛函：
  `L = Σ_v f(h_v)`（对 `h` 的逐原子求和）与 `ρ = MLP(mean h, mean h², n, g)`（对 `h` 的池化）。
  只要保留任意一条全局通路（`g` 或另一条读出），性能就在容差内；把两条同时拿掉才掉 0.046。
  这也解释了为什么单删任一条都"不掉分"却不等于该路径无用（与 Q1 的交叉项 −0.0545 同构）。
- 3.3 节结论的推论：**预测真正依赖的是 `h`（529 维局部内容经 LocalMLP），而不是某个特定读出**。
  这一点与 Stage B 的直接证据一致：删掉输入端的 typed-S2 块（B0）会稳定掉 0.054，而删掉任一条读出都不会。
- 购买规则层面：A5 与 A3 都只差 MAE 容差（mean ΔA = +0.015 / +0.027），因此都不合格 —— 见 §6。

---

## 4. Stage B：S2 的 joint/content 相对 marginals 的增益（Q3）

### 4.1 主表

| 变体 | 参数 | **Valid RMSE** | Valid MAE | **Test RMSE** `EXPOSED_TEST_EXPLORATORY` | Test MAE |
|---|---|---|---|---|---|
| A0 FULL | 247,106 | 0.7066 ± 0.0288 | 0.4066 | 0.5770 | 0.4246 |
| B0 TYPED_S2_OFF | 176,514 | 0.7608 ± 0.0084 | 0.5592 | 0.7563 | 0.5888 |
| B1 NO_JOINT | 224,546 | 0.7175 ± 0.0138 | 0.4242 | 0.5856 | 0.4414 |
| B2 NO_CONTENT | 228,274 | 0.7265 ± 0.0415 | 0.4635 | 0.6327 | 0.4819 |
| B3 NO_JOINT_CONTENT | 205,714 | 0.7423 ± 0.0267 | 0.4742 | 0.6346 | 0.4913 |
| B4 MARGINAL_ALL | 247,106 | 0.7136 ± 0.0278 | 0.4160 | 0.5788 | 0.4406 |
| B5 MARGINAL_BT2 | 247,106 | 0.7173 ± 0.0279 | 0.4233 | 0.5749 | 0.4340 |
| B6 NO_CLOSURE_CHANNELS | 245,362 | 0.7207 ± 0.0330 | 0.4127 | 0.5765 | 0.4314 |

### 4.2 对比

| 对比 | ΔValid RMSE | bootstrap 95% | 跨 0 | 读法 |
|---|---|---|---|---|
| B0 − A0（整块 typed-S2） | +0.0542 | [−0.113, +0.204] | 是 | 3/3 seed 变差；CI 宽是因为 scaffold 间差异大 |
| B1 − A0（只去 `J`） | +0.0109 | [−0.005, +0.035] | 是 | 与 0 难分 |
| B2 − A0（只去 `C`） | +0.0199 | [−0.044, +0.091] | 是 | 略差 |
| B3 − A0（去 `J`+`C`） | +0.0357 | — | — | ≈ B1+B2（可加） |
| B3 − B1 | +0.0248 | [−0.020, +0.077] | 是 | 一致的变差 |
| B3 − B2 | +0.0158 | [−0.007, +0.033] | 是 | 一致的变差 |
| **B4 − A0（同参数，joint+content → 边际）** | **+0.0070** | [−0.006, +0.025] | 是 | **配对信息几乎不值钱** |
| **B5 − A0（同参数，按 `(root,bt2)` 边际）** | **+0.0107** | [+0.0001, +0.0286] | **否** | 同上，稍大 |
| B6 − A0（closure 通道） | +0.0141 | [−0.010, +0.034] | 是 | 保留更好 |
| B_INTERACTION = (B3−B1) − (B2−A0) | −0.0049 | [−0.028, +0.028] | 是 | `J` 与 `C` 的删除近似可加 |

### 4.3 回答 Q3

1. **typed-S2 块整体是有价值的**：B0 少 7 万参数，Valid +0.054（3/3 seed 一致变差），Test +0.179。这是本轮**唯一**一个"删掉大块就明显掉分"的结构。
2. **但"联合/配对"本身几乎不值钱**：B4/B5 在**参数完全相同**的情况下，把 `Σ_p joint_proj(u_p ⊙ w_p)` 与 `Σ_p path_gate ⊙ a_end` 换成
   每根（或每 `(root,bt2)` 组）的**边际乘积** `W(n·mean(u)·mean(w)) + n·b` 与 `n·mean(gate)·mean(a_end)`，只损失 **+0.007 / +0.011**（B5 的 CI 下界刚好擦过 0）。
3. **价值来自边际量**：B3（彻底删掉 `J`+`C`）损失 +0.036，而 B4/B5（保留其边际）只损失 ≤0.011 —— 说明那 ~0.025 的差距由**类型/末端/内容的边际分布**提供，而不是由"哪条路径和哪条路径配在一起"提供。
4. **控制变量的覆盖度必须同时报告**（见 §8.4）：B4/B5 是**代数恒等替换**，不是化学反事实；`path_gate` 初始化为 1 时，content 边际在初始化点与 FULL 的 content 求和完全相等（content coverage = 0.000），
   因此"B4/B5 不掉分"不能解释成"门控/内容项无用"。用非恒定门的确定性探针测得的**结构机会**是 joint 0.587 / content 1.000（Valid），Test 为 0.590 / 0.998。

---

## 5. Stage C：S1-dup / scalars / 2nd-order pooling / closure 留谁（Q4）

### 5.1 主表

| 变体 | 参数 | **Valid RMSE** | Valid MAE | **Test RMSE** `EXPOSED_TEST_EXPLORATORY` | Test MAE |
|---|---|---|---|---|---|
| A0 FULL | 247,106 | 0.7066 ± 0.0288 | 0.4066 | 0.5770 | 0.4246 |
| C0 NO_S1JOINT | 238,402 | 0.7345 ± 0.0324 | 0.4254 | 0.5964 | 0.4442 |
| **C1 NO_BOND_MEAN** | **228,674** | **0.7053 ± 0.0403** | **0.4075** | **0.5784** | **0.4288** |
| C2 NO_SCALARS | 245,378 | 0.7222 ± 0.0384 | 0.4178 | 0.5739 | 0.4318 |
| C3 NO_M2 | 243,010 | 0.7186 ± 0.0104 | 0.4151 | 0.5779 | 0.4268 |
| C4 MEAN_ADD | 247,106 | 0.7322 ± 0.0533 | 0.4233 | 0.5788 | 0.4369 |

### 5.2 对比与判定

| 变体 | 删的是什么 | ΔValid RMSE | bootstrap 95% | mean ΔA | 允许购买? |
|---|---|---|---|---|---|
| **C1** | `b_mean = b_sum/deg`（96 维，S1-dup） | **−0.0013** | [−0.014, +0.008] | **+0.0009** | ✅ **是** |
| C0 | 32 维 `nbr` 联合项（S1 类型联合） | +0.0280 | [−0.007, +0.057] | +0.0188 | ❌ |
| C2 | 9 维 `atom_scalar` | +0.0156 | [+0.002, +0.026] | +0.0113 | ❌ |
| C3 | `ρ` 的二阶矩输入 `mean_h²` | +0.0120 | [−0.011, +0.031] | +0.0085 | ❌ |
| C4 | `Σ f` → `mean f`（同参数） | +0.0256 | [−0.007, +0.047] | +0.0167 | ❌（且不省参数） |
| B6 | 8 维 closure 通道 | +0.0141 | [−0.010, +0.034] | +0.0062 | ❌ |

### 5.3 回答 Q4

- **S1-dup（`b_mean`）：可以删。** 它是 `b_sum/deg`，度信息已由 `atom_feat`（含 degree/环等）与标量块提供；
  删掉省 18,432 参数（96×192 的输入列），Valid 反而 −0.0013、MAE +0.0009，是本轮**唯一**同时通过四条容差的变体。
- **scalars：必须留。** C2 的 CI 明确不含 0（[+0.002,+0.026]），它是本轮**唯一**一个"边际虽小但统计上稳定变差"的删除。
  值得注意的是 C2 的 **Test 反而更好**（0.5739 < 0.5770）—— 这是 Valid/Test 不一致的直接证据，说明不能拿 Test 做选择。
- **2nd-order pooling：必须留，且"求和"而非"平均"是关键。** C3（去 `mean_h²`）+0.012；C4（sum→mean，参数不变）+0.026。
  后者说明 `ρ` 依赖的**总量级**（随分子大小增长）本身是信号，而不是只依赖形状。
- **closure：建议保留。** B6 +0.014，CI 跨 0（−0.010,+0.034），且只省 1,744 参数 —— 收益/风险比不划算。
- **S1 类型联合项（C0）也不能删**：+0.028，与 B1（去 S2 joint）合计说明"类型联合"在两处都起了作用。

---

## 6. COMPACT：选择过程与一次性 Test

### 6.1 选择规则（冻结，只读 Valid）

```
候选 = 任意变体 V（A/B/C 全部），满足（Δ 均为 V − FULL，原始 logD，有符号）
  (a) params(V) < params(FULL)
  (b) mean_seed ΔValid RMSE ≤ 0.010
  (c) max_seed  ΔValid RMSE ≤ 0.020      # 任何单个 seed 都不许比 FULL 差超过 0.020
  (d) mean_seed ΔValid MAE  ≤ 0.010
排序 = 参数升序 → mean Valid RMSE 升序 → 冻结变体顺序；取第一个
若无可选 → status = NO_SIMPLER_VARIANT, COMPACT = FULL
```

选择输入是 `analysis/freeze/{A,B,C}_r122.json`（只含 Valid），
`dissect_compact.py` 会**拒绝**任何带 Test 字段的输入（`selection_inputs_exclude_test=true`），
输入文件的 SHA256 记录在 `analysis/compact_selection.json`。

### 6.2 全部 17 个候选的判定

| 变体 | 参数 | mean ΔR | max ΔR | mean ΔA | (a) | (b) | (c) | (d) | 通过 |
|---|---|---|---|---|---|---|---|---|---|
| A1 | 2,817 | +0.305 | +0.329 | +0.387 | ✅ | ❌ | ❌ | ❌ | ❌ |
| A2 | 244,418 | +0.009 | +0.038 | +0.015 | ✅ | ✅ | ❌ | ❌ | ❌ |
| A3 | 242,881 | −0.009 | +0.014 | +0.027 | ✅ | ✅ | ✅ | ❌ | ❌ |
| A4 | 240,193 | +0.054 | +0.065 | +0.099 | ✅ | ❌ | ❌ | ❌ | ❌ |
| A5 | 236,034 | −0.012 | +0.006 | +0.015 | ✅ | ✅ | ✅ | ❌ | ❌ |
| B0 | 176,514 | +0.054 | +0.085 | +0.153 | ✅ | ❌ | ❌ | ❌ | ❌ |
| B1 | 224,546 | +0.011 | +0.036 | +0.018 | ✅ | ❌ | ❌ | ❌ | ❌ |
| B2 | 228,274 | +0.020 | +0.033 | +0.057 | ✅ | ❌ | ❌ | ❌ | ❌ |
| B3 | 205,714 | +0.036 | +0.060 | +0.068 | ✅ | ❌ | ❌ | ❌ | ❌ |
| B4 | 247,106 | +0.007 | +0.026 | +0.009 | ❌ | ✅ | ❌ | ✅ | ❌ |
| B5 | 247,106 | +0.011 | +0.026 | +0.017 | ❌ | ❌ | ❌ | ❌ | ❌ |
| B6 | 245,362 | +0.014 | +0.017 | +0.006 | ✅ | ❌ | ✅ | ✅ | ❌ |
| C0 | 238,402 | +0.028 | +0.032 | +0.019 | ✅ | ❌ | ❌ | ❌ | ❌ |
| **C1** | **228,674** | **−0.001** | **+0.016** | **+0.001** | ✅ | ✅ | ✅ | ✅ | ✅ |
| C2 | 245,378 | +0.016 | +0.023 | +0.011 | ✅ | ❌ | ❌ | ❌ | ❌ |
| C3 | 243,010 | +0.012 | +0.026 | +0.009 | ✅ | ❌ | ❌ | ✅ | ❌ |
| C4 | 247,106 | +0.026 | +0.066 | +0.017 | ❌ | ❌ | ❌ | ❌ | ❌ |

**唯一通过：C1（NO_BOND_MEAN）。** `status = SIMPLER_CANDIDATE_SELECTED`。

值得记录的两点：
1. **A5（只留 `L+b0`）与 A3（去加性头）都只差一条容差**（MAE），说明"减少读出路径"在 RMSE 意义上几乎免费，
   但分布形状（MAE）会变 —— 报告不把这些当作"可购买"，因为规则是四条同时满足。
2. **B4 的 mean ΔA = +0.009 恰好压在 0.010 边界内**，但它不省参数，因此 (a) 不满足 —— 规则正确地拒绝了它。

### 6.3 COMPACT 的一次性 Test（`EXPOSED_TEST_EXPLORATORY`）

| 模型 | 参数 | Valid RMSE | **Test RMSE** | Test MAE | Test R² |
|---|---|---|---|---|---|
| FULL (A0) | 247,106 | 0.7066 | 0.5770 | 0.4246 | 0.768 |
| **COMPACT (C1)** | **228,674** | **0.7053** | **0.5784** | **0.4288** | **0.767** |

Test 上 COMPACT 与 FULL 的差距 +0.0014（3 seeds 均值），与 Valid 的 −0.0013 同量级且方向相反 ——
这是"两者在 1 个 Test RMSE 的千分位上无法区分"的**描述性**陈述，不构成等价性证明（n=3 seeds，Test 只读一次）。

Test 读取纪律：`analysis/test_access_log.json` 记录每次 score 的 run 列表、checkpoint 哈希、
Test 标签读取次数（每 checkpoint 恰好 1 次）、以及"最后一次读取发生在哪个 stage freeze 之后"。

---

## 7. 跨划分（repeat 123 / 124）

### 7.1 设计（冻结的 post-selection 采购）

COMPACT 选定后，在公开 `123`/`124` 上固定运行 **FULL / COMPACT(C1) / B5** 各 seeds 0/1/2（3×3×2 = 18 run）。
每个 repeat：**重新拟合 Train 归一化**、**从头初始化**（不迁移 122 权重）、各自用自身 Valid 选 epoch、冻结后各自一次性读 Test。

### 7.2 结果（每格 = 3 seeds 均值）

| repeat | 变体 | 参数 | Valid RMSE | Valid MAE | **Test RMSE** `EXPOSED_TEST_EXPLORATORY` | Test MAE |
|---|---|---|---|---|---|---|
| 122 | FULL | 247,106 | 0.7066 ± 0.0288 | 0.4066 | **0.5770** | 0.4246 |
| 122 | COMPACT (C1) | 228,674 | 0.7053 ± 0.0403 | 0.4075 | **0.5784** | 0.4288 |
| 122 | B5 | 247,106 | 0.7173 ± 0.0279 | 0.4233 | **0.5749** | 0.4340 |
| 123 | FULL | 247,106 | 0.7610 ± 0.0263 | 0.4858 | **0.5827** | 0.4278 |
| 123 | COMPACT (C1) | 228,674 | 0.7591 ± 0.0276 | 0.4769 | **0.5667** | 0.4107 |
| 123 | B5 | 247,106 | 0.7793 ± 0.0226 | 0.4935 | **0.5853** | 0.4320 |
| 124 | FULL | 247,106 | 0.6373 ± 0.0089 | 0.4161 | **0.5391** | 0.4133 |
| 124 | COMPACT (C1) | 228,674 | 0.6367 ± 0.0147 | 0.4177 | **0.5420** | 0.4107 |
| 124 | B5 | 247,106 | 0.6462 ± 0.0153 | 0.4142 | **0.5368** | 0.4064 |
| **grand** | FULL | | 0.7016 | | **0.5663** | |
| **grand** | COMPACT (C1) | | **0.7004** | | **0.5624** | |
| **grand** | B5 | | 0.7143 | | **0.5656** | |

### 7.3 配对差异（相对同 repeat 同 seed 的 FULL）

| 对比 | 122 | 123 | 124 | 9 对平均 | max | SD | mean ΔA | max ΔA |
|---|---|---|---|---|---|---|---|---|
| **COMPACT − FULL** | −0.0013 | −0.0019 | −0.0006 | **−0.0013** | **+0.0157** | 0.0101 | −0.0021 | +0.0201 |
| B5 − FULL | +0.0107 | +0.0183 | +0.0090 | +0.0127 | +0.0260 | 0.0106 | +0.0075 | +0.0355 |

（单位：logD Valid RMSE。max/SD/`mean ΔA` 基于 3 repeat × 3 seed = 9 对。）

### 7.4 读法

1. **划分本身比变体效应大一个量级**：同一个 FULL、同一 recipe，Valid RMSE 在 0.637（124）到 0.761（123）之间摆动（range 0.124），
   而全部 18 个变体的 Valid 差异都在 ±0.06 内。**只能用同 repeat 的配对差异做比较**，跨 repeat 的绝对值不可比。
2. **COMPACT 在三个 repeat 上都 ≤ FULL（Valid）**：−0.0013 / −0.0019 / −0.0006；9 对中最大的一对仍只有 +0.0157（< 0.020 上限），
   9 对均值 −0.0013。也就是说，"删 `b_mean`"这个购买决定在没参与选择的两份划分上**仍然成立**。
3. **B5 在三个 repeat 上都更差（Valid）**：+0.011 / +0.018 / +0.009。边际化 joint/content 的小损失是**可重复的**，
   与 r122 的 +0.011 一致 —— 这支持 §4 的结论（配对信息价值 ≈ 0.01）而不是"完全为零"。
4. **Test（仅供描述）**：COMPACT − FULL = +0.0014 / −0.0160 / +0.0029，9 对均值 −0.0039；B5 − FULL = −0.0021 / +0.0026 / −0.0023。
   这三个数都远小于划分间波动（0.044）与 seed 波动（0.01–0.03），**不构成"哪个模型更好"的证据**。
5. 跨划分阶段**没有**改变 COMPACT 的选择，也没有引入新模式或新超参；它只回答"这个简化是否只在 122 上成立"。
   答案：不是。

---

## 8. Q5：能否把某个局部统计量回溯到预测

### 8.1 接口与恒等式

FULL 的前向给出可加分解（原始单位）：

```
pred = target_mean + Σ_v f(h_v) + L + ρ
     = target_mean + Σ_v c_v + ρ,     c_v = f(h_v) + L_v        (L = Σ_v L_v)
```

`dissect_intervene.py` 对 122 Valid 的**前 64 行** × 3 seeds 做**固定权重干预**（不重训、不换权重）：

| 干预 | 覆盖 (行比例) | mean\|Δpred\| | ΔRMSE | max\|Δpred\| |
|---|---|---|---|---|
| `typed_s2_zero`（200 维 typed-S2 置零） | 1.000 | **1.6774** | +1.7272 | 7.873 |
| `contrib_zero`（`L` 置零） | 1.000 | **0.6620** | +0.5141 | 2.005 |
| `g_zero`（42 维 `g` 置零） | 1.000 | **0.4736** | +0.4261 | 2.718 |
| `mean_h2_zero`（`ρ` 的二阶矩输入置零） | 1.000 | **0.3353** | +0.2399 | 1.407 |
| `agg_b5`（joint/content 换成 B5 边际聚合） | 0.984 | **0.1316** | +0.0443 | 0.647 |

读法：**typed-S2 是最大的单点依赖（1.68 > `L` 的 0.66 > `g` 的 0.47 > 二阶矩 0.34 >> 聚合方式 0.13）**。
最后一行同时说明"聚合方式"确实能改变预测（覆盖 98.4% 的行），但其幅度只有 0.13 —— 与 §4 的"配对几乎不值钱"一致。

### 8.2 路径级留一（LOO）

对每条二跳路径 `p = (root u, middle, end w, bt1, bt2, closure)`，从该分子的路径表中**删除这一条**，
重新做一次前向（其余输入不变），记录：

- **恒等式**：`Δpred = ΔL + Δρ`，残差 ≤ **2.9e-7**（逐例核对；`identity_residual` 字段）。
- **局部性**：其他原子的 `f` 变化**严格为 0**（`other_atoms_f_max_abs_delta = 0.00e+00`，全部 552 行 `atom_terms.csv` 一致）。
- **原子项一致性**：`atom_terms.csv` 的 552 行原子级记录满足 `Σ_v c_v + target_mean + ρ = pred`，残差 ≤ **3.9e-7**。

示例（rank 0 = 全 Valid \|error\| 最大的行，row 2087，`CC(N)(COP(=O)(O)O)C(=O)O`，y=2.5，pred=−7.022，\|e\|=9.52）：

| path | root→mid→end | bt1→bt2 | closure | pred before→after | f_root before→after | ΔL | Δρ | Δpred | 其他原子 Δf | 恒等式残差 |
|---|---|---|---|---|---|---|---|---|---|---|
| 31 | 10O→9→11O | 2→1 | 0 | −7.022→−5.862 | −0.446→+0.074 | +0.5203 | +0.6393 | **+1.1596** | 0.00e+00 | 2.88e-07 |
| 19 | 6O→5→7O | 2→1 | 0 | −7.022→−6.348 | −0.893→−0.465 | +0.4282 | +0.2457 | +0.6738 | 0.00e+00 | 2.88e-07 |
| 33 | 11O→9→10O | 1→2 | 0 | −7.022→−6.419 | −1.088→−0.415 | +0.6726 | −0.0700 | +0.6025 | 0.00e+00 | 4.44e-16 |
| 8 | 2N→1→9C | 1→1 | 0 | −7.022→−7.565 | −0.489→−0.944 | −0.4555 | −0.0877 | −0.5432 | 0.00e+00 | 1.33e-15 |

这张表就是"回溯"的实际含义：
删掉一条 `root O —(2)— 中间原子 —(1)— 末端 O` 的二跳路径，
**根的局部贡献从 −0.446 变成 +0.074**（`f` 翻转符号），同时**池化项 `ρ` 从 −3.3225 变成 −2.6832**（+0.639），
两者相加等于预测变化 +1.1596。任何"这个局部统计量对预测有多大影响"的陈述，都必须同时报告
**它自己的贡献变化**与**它通过 `h` 的矩对 `ρ` 的间接影响** —— 后者在本例中占了 55%。

**COMPACT（C1）上同样成立**（`analysis/attribution/path_loo_compact.json`，8 个案例、C1 seed 0）：
同一个分子 row 2087 的预测是 −7.2662（L=−5.7626, ρ=−3.6806, 残差 8.9e-16）；
删掉同一条 `10O→9→11O (2→1)` 路径后 −7.2662 → −6.0265，`ΔL=+0.7676`、`Δρ=+0.4722`、`Δpred=+1.2397`；
其他原子 `Δf` 仍为 0，恒等式残差最大 2.9e-7。也就是说，删掉 `b_mean` **没有改变**"局部统计量 → 预测"的可分解性，
只改变了 `h` 的函数本身（同一路径的影响从 1.16 变成 1.24）。

### 8.3 八个展示案例（固定 ranks 0,1,209,210,416,417,418,419）

`analysis/attribution/TABLES.md` 给出每个案例的：预测、目标、`target_mean`、`L`、`ρ`、恒等式残差、
每条路径的 `bt1/bt2/closure/middle 候选`、before→after 的 `pred`、`f_root`、`h_root` 范数、`ρ`，
以及 top-12 路径表和 8 张图（`analysis/attribution/figures/example_rank00*.png`）。
`analysis/attribution/atom_terms.csv` 包含 **A0 与 C1 两套**逐原子项（各 552 行），可直接核对 `Σ_v c_v + target_mean + ρ = pred`。

### 8.4 覆盖度与不能声称的东西

- **`agg_b5` 的覆盖度 0.984**（98.4% 的行预测发生变化）；在初始权重下 content 边际与 FULL 恒等
  （`path_gate≡1`），因此**结构机会**用确定性非恒定门探针测量（content 1.000 / joint 0.587，Valid）。
  报告同时给出"初始化覆盖度"与"结构机会覆盖度"，任一为 0 的结果**不得**被解释为"该分支无用"。
- 这些干预是**单一已训练函数上的依赖诊断**，不是"分子被化学改造"的反事实；
  路径删除是**模型输入通道**的干预，不改变分子的真实化学结构。
- `ρ` 的变化使"删一条路径"的影响**不可能**完全局部化：局部性只在 `f` 上严格成立（其他原子 Δf = 0），
  在预测上必然通过 `h` 的矩溢出。这正是本模型"局部统计 → 全局预测"的真实通路。

---

## 9. 预算、并发与执行账本

| 项 | 冻结上限 | 实耗 |
|---|---|---|
| GPU 小时（soft / hard） | 6.0 / 8.0 | **见下表**（122 全矩阵 + 跨划分） |
| 并发作业 | ≤ 2 | 2（`res-gpu0` / `res-gpu1`，各 1×A100-40GB） |
| 正式 run 数 | ≤ 72 | 54（122）+ 18（123/124）= **72** |
| 超参搜索 / 集成 / EMA / 蒸馏 / 新描述符 | 禁止 | 0 |

- 单 run GPU 时间来自 `train.json → gpu.mean_epoch_seconds × epochs_run`，汇总在 `analysis/tables/gpu_ledger.json`。
- 所有正式 run 都从**干净 commit** 部署（`rr` 拒绝脏工作树）；每个 `train.json` 记录 `git_commit`、`git_dirty_tracked`、驱动/`torch`/节点。
- 失败运行：仅 2 次（`cross` 首次启动时因 `compact_selection.json` 未登记而立即退出，4 秒，零 GPU 消耗），
  已按规则在报告与 `analysis/freeze` 中记录，不计入正式 run。
- **预算状态**：见 §9.1 的最终数字；若超过 soft cap，则报告明确标注"已停止一切新增正式运行，仅完成预注册的跨划分采购"。

### 9.1 最终 ledger

| repeat | runs | GPU 小时 |
|---|---|---|
| 122（18 变体 × 3 seeds） | 54 | 4.742 |
| 123（FULL/COMPACT/B5 × 3 seeds） | 9 | 0.886 |
| 124（同上） | 9 | 0.916 |
| **合计** | **72** | **6.544** |

- `analysis/tables/gpu_ledger.json` 逐 run 记录 `train_wall_seconds`、`mean_epoch_seconds`、`gpu_hours`；合计 6.5437 h。
- 预算合规：冻结协议第十三节写明"软预算不是提前停止已允许固定实验的科学 gate；到 6 h 报告剩余计划，在硬上限内继续"。
  本轮在 122 阶段结束时已到 4.74 h，剩余计划（跨划分 18 run，约 1.7 h）在启动前登记；实际结束时 6.54 h，
  **超 soft 0.54 h（+9%），低于 hard 8.0 h**；此后未启动任何新正式运行。
- 除上述 72 个正式 run 外，只运行过：`lipo-clean`（远端工作树清理，0 GPU）、`lipo-prep`（结构缓存，49 s）、
  `lipo-smoke`（1 epoch CUDA 烟雾，1.1 s/epoch）与 2 次立即退出的 `cross` 启动（每次 4 s，0 GPU）。这些都不产生指标、不参与任何结论。
- 单 run 成本估算：122 全矩阵平均 0.088 h/run（A1 最低 0.037 h，B5 最高 0.103 h）；跨划分平均 0.100 h/run。

---

## 10. 与文献的关系（不参与本轮结论）

- 本轮**没有**做任何跨模型比较，因此不引用任何外部数字作为"我们的模型更好/更差"的证据。
- 若需要定位：Lipophilicity 上公开的 GNN 结果通常在 RMSE 0.55–0.75 区间（MoleculeNet 系列与 ChemFM 系列报告），
  本轮 FULL 的 Test 0.5770 落在该区间内。**该定位未由本轮复现**，仅作为背景，见 `§11` 的限制说明。
- 本轮的可迁移结论与具体数据集无关：**"多个分支各自删掉都不掉分，不等于它们冗余；要看交叉项与覆盖度"** —— 这是
  本轮四角设计与代数控制给出的方法论结论。

---

## 11. 不能声称的内容 / 限制 / 未完成项

**不能声称**

1. 不能声称"`g` 无用"或"局部路径无用"：交叉项显著（−0.0545）说明二者条件性替代（§3.3）。
2. 不能声称"B4/B5 证明了 joint/content 无用"：它们是**同参数的代数恒等替换**，且在初始化点 content 覆盖度为 0（§4.3、§8.4）。
3. 不能声称"COMPACT 与 FULL 等价"：容差是**实用性购买阈值**（0.010/0.020/0.010），不是等价性检验；n=3 seeds。
4. 不能声称任何**化学**因果：所有干预都是固定权重下对模型输入的干预。
5. 不能声称 Test 数字可用于选择：Test 每个 checkpoint 只读一次，且 C2 的例子（Valid 变差、Test 变好）说明二者会分歧。
6. 不能声称"closure 无用"：+0.014 且 CI 跨 0，只说明"在这个容量预算下不值得为它付 1,744 参数"。

**限制**

- 3 seeds / 1 个公开划分（选择阶段）；scaffold bootstrap 是**评估样本**区间，不是训练随机性 CI。
- `Valid` 只有 420 行，单行噪声（row 2087 的 \|e\|≈9.5）会显著影响 RMSE；A4/B0 的 CI 很宽即由此而来。
- 所有变体共享同一份 FULL 初始化（按固定列映射复制），这消除初始化噪声但**不是**"独立初始化下的期望表现"。
- 冻结 recipe 的梯度裁剪触发率很高（30% 的 step 触发 clip；pilot 参照 run 为 49%），是 recipe 的性质，本轮未改动。

**未完成项**

- 无。跨划分结果见 §7；`REPORT.md`、`MODEL.md`、`PROTOCOL.md`、`analysis/` 全部随本轮提交。

---

## 附录 A. 复现命令（全部经 `rr`，禁止手写 ssh/nohup）

```bash
# 0) 预检
rr doctor res

# 1) 冻结协议 + CPU 审计 + 测试（本地）
uv run python code/dissect_protocol.py
uv run python code/dissect_audit.py
uv run python -m pytest tests -q

# 2) 远端准备（结构缓存 + 归一化，49 s）
rr run res lipo-prep --pool res-gpu0 --result runs -- uv run python code/dissect_prepare.py

# 3) 分 stage 扫描（每 stage 2 个 shard，各占一张 A100）
rr run res lipo-A0 --pool res-gpu0 --gpus 1 --result runs -- uv run python code/dissect_sweep.py --stage A --shard 0 --shards 2
rr run res lipo-A1 --pool res-gpu1 --gpus 1 --result runs -- uv run python code/dissect_sweep.py --stage A --shard 1 --shards 2
#   ... B / C / cross 同构

# 4) stage 冻结（只读 Valid）→ 一次性 Test
uv run python code/dissect_freeze.py --stage A
uv run python code/dissect_score.py --freeze analysis/freeze/A_r122.json

# 5) 选择 COMPACT（只读 Valid）→ 登记到 run_matrix → 跨划分
uv run python code/dissect_compact.py
uv run python code/dissect_protocol.py --compact C1

# 6) 表格 / 图 / 归因
uv run python code/dissect_tables.py
uv run python code/dissect_render.py
uv run python code/dissect_intervene.py --compact C1
```

## 附录 B. 文件与产物

| 路径 | 内容 |
|---|---|
| `PROTOCOL.md` | 冻结协议全文（含原始 prompt 逐字嵌入） |
| `MODEL.md` | 逐行计算图、字段来源、18 变体开关点、"可声称/不可声称" |
| `REPORT.md` | 本文件 |
| `analysis/protocol_freeze.json` | 协议冻结记录（来源哈希、recipe、容差、budget） |
| `analysis/audit_cpu.json` | 62 项 CPU 审计（等价性、变体、覆盖度、选择规则） |
| `analysis/freeze/{A,B,C}_r122.json`, `{cross_r123,cross_r124}.json` | stage 冻结记录（只含 Valid） |
| `analysis/test_access_log.json` | Test 一次性读取日志 |
| `analysis/compact_selection.json` | COMPACT 选择（只读 Valid 的输入 + 全部候选判定） |
| `analysis/tables/*.csv,*.json,markdown.md` | 主表、均值、对比、bootstrap、size bin、GPU ledger |
| `analysis/attribution/*` | 干预、路径 LOO、原子项、8 张图、`TABLES.md` |
| `analysis/bootstrap.json` | scaffold 组 bootstrap（2000 次，seed 20261102） |
| `runs/<repeat>/<variant>/s<seed>/` | 每 run 的 `train.json`、`checkpoint.pt`、`*_predictions.npz`（不进 Git） |

## 附录 C. 报告完整性

- 本文件由本轮全部 **72 个正式 run** 的冻结记录生成：`analysis/freeze/{A,B,C}_r122.json`（54 run）+
  `analysis/freeze/cross_r123.json`、`cross_r124.json`（18 run），Test 读取记录在 `analysis/test_access_log.json`。
- 所有整数/浮点数字可由 `analysis/tables/*.json`、`analysis/bootstrap.json`、`analysis/freeze/*.json` 重新计算；
  报告中的主表、四角对比、COMPACT 判定表已用脚本与源 JSON 逐项比对（0 处不符）。
- 本轮结束时 `analysis/compact_selection.json` 已提交（commit `c1116d6e`），跨划分在其后启动。
