# Compact-v6: Attribute Branch Viability Repair

> 实验对象：`luyin16-zinc-hierarchical-patch-relation-context-compact-v6-viable`
> 对照：`...-compact-v6-factorized-role`（original，zero-init，**已 collapse**）
> 基线：historical compact-v4-hinge seed-0 valid `0.17006561887910357`
> 协议：`zinc-context-gap`（PyG ZINC `subset=True`，train→valid selection，frozen
> checkpoint；**test 未加载**）
> 代码：`experiments/luyin16/compact_v6_viability_repair.py`（诊断/门）、
> `experiments/luyin16/compact_v6_viability_summary.py`（表/图/decision）、
> `zinc_patch_path_pooling.py`（`attribute_fusion_init` 最小修复）
> config：`configs/luyin16/zinc_compact_v6_viable.yaml`
> 测试：`tests/test_compact_v6_viability_repair.py`（8 tests）
> 数据/表：`results/compact_v6_viability/`

---

## 0. 本阶段唯一问题

原始 compact-v6 的 attribute branch 在 frozen seed-0 上：

```
std(e_attribute) ≈ 2.3e-10
type-shuffle / role-shuffle  max |Δpred| = 0.0（精确）
```

因此 original v6 = **collapsed / unused**。但 topology–attribute factorization
假设**尚未被有效检验**。本阶段：先用 gradient trace 找到 collapse 的**明确机制**，
做**唯一最小修复**，再用 100-step viability gate 证明 branch 可训练、可影响预测，
最后用 fresh seed0 决定该 factorization 是 GO 还是永久 CLOSE。

**边界**：本阶段不是新架构。除 attribute branch 可训练性外，架构逐位不动。

---

## 1. Diagnosis：original collapse 的机制

### 1.1 single-batch forward / gradient trace（`original_single_batch_trace.json`）

seed 0、当前训练数据、当前初始化、一个真实 training batch（2,945 patches）。
forward 统计（节选）：

| 节点 | mean | std | 备注 |
|---|---:|---:|---|
| raw atom role features | 0.366 | 0.388 | 有信息（39% 结构性 0） |
| raw bond role features | 0.343 | 0.361 | 有信息 |
| atom type embeddings | 0.514 | 1.365 | 有信息 |
| bond type embeddings | −0.073 | 0.553 | 有信息 |
| atom encoder output | 0.176 | 0.188 | 有信息 |
| bond encoder output | 0.045 | 0.062 | 有信息 |
| atom / bond pooled | 0.177 / 0.045 | 0.168 / 0.059 | 有信息 |
| attr fusion hidden | −0.034 | 0.175 | 有信息 |
| **final e_attribute** | **0.0** | **0.0** | **zero-init，恒为 0** |
| concatenated patch input | −0.013 | 0.733 | — |
| patch_encoder first output | 0.014 | 0.430 | — |

`loss.backward()` 后（该 batch）：

| 参数 | grad 状态 | grad_norm |
|---|---|---:|
| `attribute_encoder.fusion.2.weight` | nonzero | 2.93e-3 |
| `attribute_encoder.fusion.2.bias` | nonzero | 8.10e-3 |
| `attribute_encoder.{atom,bond}_mlp.*` | **exactly_zero** | 0.0 |
| `attribute_encoder.{atom,bond}_embedding.*` | **exactly_zero** | 0.0 |
| `patch_encoder.layers.0.weight[:, -8:]`（新增 8D 列） | **exactly_zero** | 0.0 |
| `patch_encoder.layers.0/1/4`（既有列） | nonzero | 0.16 / 0.012 / 0.155 |

`grad_e_attribute` 本身 **nonzero**（std 7.5e-6），所以 final projection 自己
第一步能走；但 `W_final = 0 → ∂h/∂W_up = W_final^T · g = 0`，**整数个 upstream
attribute encoder 在 step 0 拿到精确 0 梯度**；同时 `e_attribute = 0` 也让
patch_encoder 新增 8 列拿到精确 0 梯度。

### 1.2 optimizer audit（`optimizer_parameter_audit.csv/json`）

```
attribute_branch_parameters    : 14
attribute_branch_all_registered: true
attribute_branch_missing       : []
num_parameters / trainable     : 62 / 62
num_optimizer_groups           : 1
```

**排除 Case G0-B（optimizer registration bug）**。所有 attribute 参数
`requires_grad=True`，都在唯一的 Adam group 里。

### 1.3 raw attribute representation 是否有信息

`attribute_input_variance.json`：train/valid 各抽 1000 patches。

- atom role 每维 std ≈ 0.30（8 维全 >0.12）；unique atom types 17/16；
- raw `(type, role)` primitive 组合 >700 种。

`adversarial_representation_check.json`：构造同 topology、同 atom/bond
multiset、不同 typed placement 的一对 patch：

```
count_view_identical   : true
factorized_view_differs: true
placement_pair_ok      : true
```

**排除 Case G0-D（attribute input 本身恒定）与 role/primitive 构造 bug**。

### 1.4 collapse 的实际位置（frozen original checkpoint）

在 original frozen seed-0 checkpoint 上逐层统计 **patch 维度**方差
（跨 patch 的 std，而非跨 feature）：

| 层 | 跨 patch std |
|---|---:|
| `h_atom`（occurrence） | 5.2e-6 |
| `h_bond`（occurrence） | 2.4e-5 |
| pooled atom / bond（`fusion_in`） | 1.1e-6 |
| `fusion_hidden` | 1.0e-8 |
| **`e_attribute`** | **7.6e-12** |
| `e_attribute` norm（mean / std） | 2.749e-3 / 7.2e-12 |

即：**collapse 不是发生在 final projection**（其 `weight.norm = 7.9e-3`，非零），
而是发生在 attribute encoder 的**非线性隐藏层**——它们退化成近常数函数
（输入 occurrence std 0.212，输出 `h_atom` occurrence std 仅 5.2e-6）。
`e_attribute` 缩成一个 patch 不变的常向量，被 patch_encoder 后面的 LayerNorm
吸收 → 预测对 shuffle **精确不变**。original frozen 的 `bond_mlp.0.weight`
norm 正好 **0.0**，`bond_embedding` ≈ 1.6e-6。

### 1.5 10-step micro trace（`original_10step_trace.csv`）

| step | loss | grad_W_final | grad_attr_upstream | grad_W_attr | W_final norm | e_attr patch_std |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.565 | 7.0e-3 | **0.0** | **0.0** | 6.3e-3 | 6.0e-5 |
| 1 | 1.507 | 9.4e-3 | 2.4e-5 | 2.5e-4 | 1.0e-2 | 9.8e-5 |
| 9 | 1.392 | 8.3e-3 | 7.1e-5 | 8.9e-4 | 2.4e-2 | 2.3e-4 |

zero-init 的封锁只持续 **step 0**；从 step 1 起 upstream 和 patch 新列都恢复。
所以「zero-init 一步致死」**不成立**。为公平对比，另跑了 original 100-step
（`original_100step_trace.csv`）：step 99 时 `W_final=0.023`、patch_std 9.5e-5
（**刚好压在 1e-4 V1 阈值下方**）、loss 0.369。

### 1.6 Diagnosis（Case G0-A + 训练动力学吸引子）

> original attribute branch collapse 的机制：
> **final fusion projection 零初始化** ⇒ step 0 `e_attribute=0`，
> 于是 (a) 所有 upstream attribute 参数、(b) patch_encoder 新增 8 列
> 都拿到**精确零梯度**；只有 10→8 final projection 能学习。branch 因此以
> 「被延迟、且未被 patch 路径使用」的状态起步，在 canonical 60-epoch 训练中
> optimizer 选择用 v4 路径单独拟合目标，attribute encoder 的非线性层退化成
> 近常数函数，`e_attribute` 缩成常向量并被后续 LayerNorm 吸收，最终
> **精确 shuffle-invariant**。

**Q1–Q6 的确定回答**：

- **Q1 为什么 collapse**：zero-init final projection 使 branch 在 step 0 与
  loss 断开（upstream 与 patch 新列精确零梯度），加上 v4 路径已能拟合目标，
  训练收敛到忽略 placement 的解；encoder 非线性层退化 → 常向量 → 被 LayerNorm
  吸收 → shuffle-invariant。
- **Q2 哪一层**：block 发生在 final projection（step 0）；**collapse 发生在
  attribute encoder 的 MLP 隐藏层**（`h_atom`/`h_bond` 近常数），`e_attribute`
  跨 patch std 7.6e-12。
- **Q3 step0 哪些梯度为 0**：`atom_mlp.*`、`bond_mlp.*`、`atom_embedding.*`、
  `bond_embedding.*`、`patch_encoder.layers.0.weight[:, -8:]` = 精确 0；
  `fusion.2.weight/bias` = 非零；`grad_e_attribute` = 非零（std 7.5e-6）。
- **Q4 optimizer 是否包含全部参数**：是。14/14 attribute 参数注册，单 group，
  `requires_grad=True`（Case G0-B 排除）。
- **Q5 patch_encoder 新增 8D 列是否收到梯度**：step 0 **精确 0**（因 e_attribute=0）；
  step 1 起非零（2.5e-4），在 100-step repaired 中持续非零；original frozen
  checkpoint 中这些列的 weight norm = 0.4535（训练过，不是被 zero）。
- **Q6 最小修复改了什么**：**只改** final attribute fusion projection 的初始化：
  `zeros` → `normal(mean=0, std=0.01)`（`attribute_fusion_init=small_normal`）。
  scale=0.01 预注册、不 sweep。attribute dim / role features / MLP 深度 /
  patch descriptor / topology branch / pooling / graph head / loss / optimizer /
  LR / weight decay / epoch / batch size 全部不动。

---

## 2. Viability：100-step gate（`repaired_100step_trace.csv`, `repaired_viability_gate.json`）

固定 1024-molecule train subset，100 optimizer steps，repaired init。

| gate | 结果 | 判定 |
|---|---|---|
| **V1 non-collapse** | `std(e_attribute)=3.87e-4` (>1e-4) | PASS |
| **V2 gradients alive** | final projection / upstream encoder / patch attr cols 末 5 步全 >1e-9 | PASS |
| **V3 prediction sensitivity（role-shuffle）** | max |Δpred| = 3.70e-5，mean 7.22e-6 | PASS |
| **V4 branch ablation（e_attribute=0）** | max |Δpred| = 1.38e-2，mean 5.61e-3 | PASS |

`all_pass = true`。**修复后的 branch 确实可以接收非零输入、获得有效梯度、
产生非退化表示、并真实影响预测。** （此阶段不要求 shuffled MAE 更差。）

对照 original 100-step：patch_std 9.5e-5、`W_final` 0.023、loss 0.369；
repaired：patch_std 3.9e-4、`W_final` 0.084、loss 0.419。repaired branch
明显更活跃（`W_final` 3.6×，patch_std 4×），代价是前 100 步拟合略慢
（toy 阶段，不构成 benchmark 结论）。

**Q7**：100 step 后 `std(e_attribute)`（跨 patch）= **3.87e-4**。
**Q8**：attr-zero 影响（100 step toy）mean 5.61e-3 / max 1.38e-2。
**Q9**：role-association shuffle 影响（100 step toy）mean 7.22e-6 / max 3.70e-5。

---

## 3. Fresh seed0 full validation（`zinc_compact_v6_viable.yaml`, seed 0）

从头训练（不复用任何 v4/v6 checkpoint），canonical 60 epochs，
`params=101,083`，`runtime=404s`，frozen best-valid @ epoch 59。

| model | params | valid MAE | Δ (v4 − v6) |
|---|---:|---:|---:|
| v4 historical（promoted canonical reference） | 99,613 | 0.1700656 | — |
| **compact-v6-viable seed0** | 101,083 | **0.1702090** | **−0.000143** |

**Q11**：fresh seed0 Δ = **−0.000143**（gate ≥ +0.003 → **FAIL**）。

### 3.1 seed0 mechanism diagnostics（frozen best-valid checkpoint）

`seed0_mechanism_diagnostics.json`（valid n=1000）：

| diagnostic | clean | shuffled/zeroed | mean |Δpred| | max |Δpred| |
|---|---:|---:|---:|---:|
| clean | 0.1702090 | — | — | — |
| attr-zero | — | 0.1702007 | 5.14e-5 | 2.95e-4 |
| attribute-type shuffle | — | 0.17020903 | 1.15e-7 | 2.03e-6 |
| role-association shuffle | — | 0.17020904 | 1.22e-7 | 1.43e-6 |

branch 统计：

| 量 | original (collapsed) | repaired-viable seed0 |
|---|---:|---:|
| `e_attribute` patch_std | 7.6e-12 | **3.62e-7** |
| `e_attribute` norm mean | 2.749e-3 | 6.73e-3 |
| `e_attribute` norm std | 7.2e-12 | 6.2e-8 |
| `fusion.2.weight` norm | 7.9e-3 | 3.47e-2 |
| patch_encoder attr-col weight norm | 0.4535 | 0.4655 |

**关键结论**：修复使 branch 活跃度提高约 4–5 个数量级（patch_std 7.6e-12 →
3.6e-7），但 **canonical 全量训练后 branch 再次退化为近常数输出**：
placement 相关信号仍在 float 噪声量级（type/role shuffle mean ≈ 1.2e-7），
低于预注册的「mean |Δpred| > 1e-4」机制门槛。attr-zero 仍给出可测的常数偏移
（mean 5.1e-5），但 role/type shuffle 不再可测。

> 也就是说：**修复足以通过 100-step viability gate，却不足以让 branch 在
> 60-epoch 真实协议下存活**。collapse 不只是 zero-init 的直接影响，而是一个
> 「v4 路径足以拟合目标 ⇒ placement 无持续梯度 ⇒ encoder 退化」的训练动力学吸引子。

**Q10**：完整 seed0 checkpoint 中 branch **没有被真实使用**——placement
（type/role shuffle）效应 ≈ 1e-7（数值噪声），branch 输出为近常量。
**Q13**：seed1 **未运行**（seed0 性能门失败，按预注册规则 STOP AT SEED0）。

### 3.2 Bulk safety（`seed0_difficulty_quintiles.csv`）

复用冻结的 historical difficulty quintiles（每 q n=200）：

| quintile | v4 MAE | repaired-v6 MAE | Δ (v4 − v6) |
|---:|---:|---:|---:|
| Q1 | 0.040662 | 0.069999 | **−0.029337** |
| Q2 | 0.068468 | 0.086792 | **−0.018324** |
| Q3 | 0.097396 | 0.110492 | −0.013097 |
| Q4 | 0.143851 | 0.138023 | +0.005827 |
| Q5 | 0.496876 | 0.445738 | **+0.051138** |

- overall Δ = **−0.000758**（Q1–Q3 退化抵消了 Q5 的改善）；
- **Q1+Q2 combined degradation = +0.023831 > +0.002 → FAIL**。

**Q12**：easy Q1/Q2 **不安全**。修复后的 viable 模型再次呈现
**同一 redistribution 指纹**（easy bulk 变差、hard tail 变好）：
Q1–Q3 全部退化，全部收益来自 Q5。这与 corrected exact tokenizer 及 original
collapsed v6 的指纹一致。

---

## 4. 结果判定（`decision_record.json`, `seed01_results.csv`）

| gate | 阈值 | 实测 | 判定 |
|---|---|---:|---|
| seed0 performance Δ | ≥ +0.003 | −0.000143 | **FAIL** |
| Q1/Q2 combined degradation | ≤ +0.002 | +0.023831 | **FAIL** |
| seed0 mechanism（mean |Δpred|, role-shuffle） | > 1e-4 | 1.22e-7 | **FAIL** |
| 100-step viability（修复后） | V1–V4 all | all_pass | PASS |

按 §27：seed0 性能门失败 ⇒ **STOP AT SEED0**，不跑 seed1。

```
seed01_results.csv
seed 0 : v4 0.170066 | viable 0.170209 | Δ −0.000143 | completed
seed 1 : not_run (STOP AT SEED0: performance gate failed)
```

---

## 5. 科学边界（§40）

**已经成立**：original compact-v6 实验没有使用其 attribute branch
（step-0 upstream 精确零梯度；frozen 输出精确 shuffle-invariant）。

**本阶段新成立**：分支 collapse 的机制已被 gradient trace 证实；用唯一最小修复
（nonzero final projection init）后，branch 在 100-step 上**可训练、非退化、
可影响预测**（V1–V4 pass）。

**仍未成立**：topology–attribute factorization 有用。修复后的 branch 在全量
seed0 上再次退化，性能 Δ = −0.000143（< +0.003），bulk gate FAIL。

---

## 6. 最终判断（§42 Outcome 2）

# BRANCH REPAIRED, ARCHITECTURE NO-GO

- branch 已证明可以修活（100-step V1–V4 pass）；
- seed0 性能 **负**（−0.000143）；
- 全量训练后 branch 仍近常数、placement shuffle ≈ 1e-7；
- easy bulk Q1/Q2 明显退化（+0.0238 > 0.002）。

下一步：**永久关闭 topology–attribute factorization**。
不再调 scale / dim / role feature / optimizer。

**Q14 = B. close factorization permanently.**

---

## 7. Root-cause / viability 一览（Q1–Q14）

| # | 问题 | 回答 |
|---:|---|---|
| Q1 | 为什么 collapse | zero-init final projection → step0 与 loss 断开；训练收敛到忽略 placement 的解，encoder 退化常向量并被 LayerNorm 吸收 |
| Q2 | collapse 在哪层 | block 在 final projection（step0）；collapse 在 attribute encoder 隐藏层，`e_attribute` patch_std 7.6e-12 |
| Q3 | step0 哪些梯度为 0 | 全部 upstream（embedding/MLP）+ patch_encoder 新 8 列为精确 0；final projection 与 `grad_e_attribute` 非零 |
| Q4 | optimizer 是否正确包含全部参数 | 是，14/14，单 group，requires_grad=True |
| Q5 | patch_encoder 新 8D 是否收到梯度 | step0 精确 0（e_attr=0），step1 起非零；frozen original 列权重 norm 0.4535 |
| Q6 | 最小修复改了什么 | 仅 final fusion projection init：zeros → normal(std=0.01) |
| Q7 | 100 step 后 std(e_attribute) | 3.87e-4（跨 patch） |
| Q8 | attr-zero 对预测影响 | 100-step mean 5.6e-3 / max 1.4e-2；全量 seed0 mean 5.1e-5 / max 2.95e-4 |
| Q9 | role-shuffle 对预测影响 | 100-step mean 7.2e-6 / max 3.7e-5；全量 seed0 mean 1.2e-7 / max 1.4e-6 |
| Q10 | 完整 seed0 中 branch 是否真实使用 | 否（placement 效应≈1e-7，输出近常数） |
| Q11 | fresh seed0 ΔMAE | −0.000143 |
| Q12 | easy Q1/Q2 是否安全 | 否，degradation +0.0238 > 0.002 |
| Q13 | seed1 是否复现 | 未运行（STOP AT SEED0） |
| Q14 | A 还是 B | **B. close factorization permanently** |

---

## 8. 复现命令

```bash
# original diagnosis（不改源码行为：attribute_fusion_init 默认 zero）
uv run python -m tracks.ksvd.experiments.luyin16.compact_v6_viability_repair --stage cache
uv run python -m tracks.ksvd.experiments.luyin16.compact_v6_viability_repair --stage original_trace
uv run python -m tracks.ksvd.experiments.luyin16.compact_v6_viability_repair --stage optimizer_audit
uv run python -m tracks.ksvd.experiments.luyin16.compact_v6_viability_repair --stage original_10step
uv run python -m tracks.ksvd.experiments.luyin16.compact_v6_viability_repair --stage input_variance
uv run python -m tracks.ksvd.experiments.luyin16.compact_v6_viability_repair --stage adversarial
# repaired viability
uv run python -m tracks.ksvd.experiments.luyin16.compact_v6_viability_repair --stage repaired_100step --steps 100
# fresh seed0（test blocked）
uv run research run zinc_patch_path_pooling \
  --config tracks/ksvd/configs/luyin16/zinc_compact_v6_viable.yaml \
  --study zinc-context-gap --purpose "compact-v6-viable seed0 (test blocked)" \
  --mode scratch --seed 0
# mechanism + quintiles + tables/figures/decision
uv run python -m tracks.ksvd.experiments.luyin16.compact_v6_viability_repair --stage seed_mechanism  --run-dir <seed0-run> --output tracks/ksvd/results/compact_v6_viability
uv run python -m tracks.ksvd.experiments.luyin16.compact_v6_viability_repair --stage seed_quintiles  --run-dir <seed0-run> --output tracks/ksvd/results/compact_v6_viability
uv run python -m tracks.ksvd.experiments.luyin16.compact_v6_viability_summary
uv run pytest tracks/ksvd/tests/test_compact_v6_viability_repair.py
```

seed0 viable run id：`runs/2026/09/10/20260910-183159-63fd1616`。

## 9. Figures

- Figure 1：original 10-step gradient norms by branch stage（定位 step-0 collapse）
  · `results/compact_v6_viability/figures/fig1_original_10step_gradients.png`
- Figure 2：repaired 100-step（`e_attribute` patch std / `W_final` / upstream grad）
  · `results/compact_v6_viability/figures/fig2_repaired_100step.png`
- Figure 3：seed0 clean vs attr-zero vs role-shuffle prediction-difference 分布
  · `results/compact_v6_viability/figures/fig3_seed0_prediction_delta.png`
