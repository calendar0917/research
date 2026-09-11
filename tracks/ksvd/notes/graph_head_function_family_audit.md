# Graph-head function family audit — low-rank cross-coordinate interaction

> 问题：compact-v4 已经得到的 frozen graph representation `R ∈ R^302` 是否**基本包含
> 足够信息**，但当前 downstream regression function family 并不是最 sample-efficient
> 的选择？特别测试：与同参数量 generic ReLU MLP 相比，显式 **low-rank
> cross-coordinate interaction** head（rank-4 Factorization Machine）是否更适合
> `R → y`？
>
> 模块：`experiments/luyin16/zinc_graph_head_function_family.py`
> 结果：`results/graph_head_function_family/`
> 测试：`tests/test_graph_head_function_family.py`（13 tests，全过）
> 结论：**INCONCLUSIVE（Case E）**。Stage 1（backbone seed 0）命中预注册的
> EFFICIENCY ADVANCE（mean Δsmall = +0.00337，4/5 folds），但 Stage 2（backbone
> seed 1）**没有复现**（mean Δsmall = +0.00042，2/5 folds）；pooled mean Δsmall =
> +0.00190 < +0.0025，fold-averaged 仅 3/5 为正，两 stage 的 paired bootstrap 95%
> CI 下界都为负。**不对 FM 分支扩大预算**：不扫 rank、不做 E2E、不做 hybrid。

---

## 1. Motivation

连续六条「给 compact-v4 补一个新的 structural statistic / residual adapter」的路线
都 NO-GO：

- frozen richer readout → NO-GO；
- pair endpoint association → NO-GO；
- centre-incidence covariance / co-occurrence → NO-GO；
- explicit triadic binding → NO-GO；
- additive coordinatewise quantile-hinge basis → NO-GO；
- （更早的）local structural enrichment → NO-GO。

这些结果共同把问题从「representation 里缺了哪个局部统计量」推向：

> **信息可能已经足够，问题在 downstream function family / statistical
> learnability。**

本阶段只隔离一个变量：**same representation + same L1 objective + same data，
只改变最终函数族**，并测试一个明确的 inductive bias：
**explicit low-rank pairwise interaction**。

## 2. Why structural-statistic search was stopped

三次 witness 诊断都显示「被丢弃的局部统计量」要么 target-free 存在但 task-irrelevant
（pair endpoint association ΔR = −0.00699，0/5；centre covariance ΔM = −0.00136，
2/5），要么显式三体 binding 不如 matched unbound control（ΔU = −0.00185，0/5）。
继续叠加更精细的局部统计量边际收益递减，且每次都付出新的 representation + target
audit confound。因此本阶段不再动 representation，而是审计 `R → y` 的函数族。

## 3. Representation vs head family vs loss

三条轴必须分开：

- **Representation**：有哪些信息可用（`R` 的 302 个坐标）。
- **Head family**：`R → y` 用哪种函数类（additive / generic MLP / low-rank
  interaction / tree）。
- **Loss**：训练更愿意压低哪类误差。

本阶段固定 representation（frozen v4）与 loss（L1/MAE），只动 head family。
最重要的区分是：本轮**不是** residual adapter `yhat = yhat_0 + Δ(R)`，而是
**direct head** `yhat = f(R)`。

## 4. Why additive hinge failure motivates interaction

上一轮 function-basis audit 表明：

- `f(R) = Σ_j f_j(R_j)`（coordinatewise additive nonlinear，quantile-hinge
  basis）输给同预算 generic ReLU MLP，mean Δsmall = −0.01539，0/5 folds；
- 但 hinge terms 本身是 live 的（内部置零使 MAE 上升 +0.0478，5/5）。

正确的解读不是「function parameterisation 不重要」，而是：

> **nonlinear response 有价值，但 additive coordinatewise 结构可能正是瓶颈；
> 缺的可能是 coordinate interaction。**

这正是上一轮 decision record 里登记的 `revisit_if`：
「an MAE-oriented structured function basis with explicit coordinate interactions
rather than a purely additive one」。本轮把这个 hypothesis 变成一次预注册实验。

## 5. Direct head protocol

所有新 head：

```
yhat = f(R)          # direct prediction
```

不使用 `yhat_0` 作为输入。这是与 frozen adapters 的关键区别：frozen adapter 回答
「原预测之后还能修什么」，direct head 回答「同一个 representation 本身更适合哪种
downstream function family」。

- 每个 fold 用 `head-fit 7200` 训练；
- 用 `head-selection 800` 选 best-selection checkpoint；
- 在 untouched `outer-heldout 2000` 上做一次 final evaluation。
- **H0**：原 backbone checkpoint 里的 original graph head，在
  outer-heldout 2000 上预测，作为 jointly-trained reference。

## 6. OOF data protocol

只使用 official train universe（10000 molecules）；official valid/test **从未加载**。
每个 frozen OOF fold 提供真实 nested structure：

```
7200 backbone-fit  (inner_train)
 800 backbone-selection (inner_valid)
2000 outer-heldout (outer fold)
```

这直接复用 `zinc_oof_difficulty_audit._fold_slices()`（K=5，fold_seed=0，
inner split seed 固定）。与之前 frozen audits 的 `2000 → 1200/400/400` 不同：本轮让
head 在接近真实 ZINC 规模的 7200 上训练，但最终比较仍在 backbone 从未训练过的 2000
molecules 上完成。

representation export：对每个 fold 用 frozen backbone 一次 forward 覆盖全部 10000
molecules，导出 `R`（302D）、`yhat_0`、`target`、`subset_index`、`role`
（0=fit, 1=selection, 2=holdout）。Gate 0 全部精确通过（见 §12），holdout `R` 与
既有 `frozen_state_export_v4_centre_incidence` 逐元素 max diff = **0.0**。

## 7. Tiny MLP (H1)

primary generic control：

```
302 → 5 → 2 → 1, ReLU
params = 302*5+5 + 5*2+2 + 2*1+1 = 1530
```

## 8. Strong MLP (H2)

stronger generic reference（故意不做参数匹配）：

```
302 → 13 → 13 → 1, ReLU
params = 302*13+13 + 13*13+13 + 13*1+1 = 4135
```

H2 必须存在：如果 FM > tiny MLP 但 H2 ≫ FM，正确结论只能是 *FM improves
small-budget efficiency*，不能说 *FM is a better head overall*。

## 9. Low-rank Factorization Machine (E, rank 4)

对 fit-only standardized `z ∈ R^302`：

```
yhat = b + w^T z + Σ_{i<j} <v_i, v_j> z_i z_j
```

`v_i ∈ R^r`，**固定 r = 4，禁止 rank sweep**。计算使用标准恒等式：

```
Σ_{i<j} <v_i,v_j> z_i z_j
  = 1/2 Σ_f [ (Σ_i v_if z_i)^2 - Σ_i v_if^2 z_i^2 ]
```

不构造 302×302 interaction matrix。

为什么是合理 hypothesis：tiny MLP 允许 generic cross-coordinate interaction，但必须
自己从数据里发现组合方式；FM 明确假设重要的二阶 interaction 可由一个 **low-rank**
matrix `A = V V^T` 表达。测试的是明确 inductive bias，不是又一个 basis expansion。

与前期 pair endpoint audit 不同：那里检查的是 pair level `q_ij` 里被丢弃的
endpoint association；本轮的 interaction 发生在**最终 graph representation `R`
内部**，其坐标已经综合了 unary / pair / global / topology。

## 10. CatBoost-MAE secondary reference

预注册的固定 MAE preset（`loss_function=MAE`, `eval_metric=MAE`, `depth=6`,
`learning_rate=0.03`, `iterations=1000`, `random_seed=0`, `use_best_model=True`,
`early_stopping_rounds=100`, `verbose=False`），输入与 neural heads **完全相同**的
standardized `z`，不额外喂 raw descriptors。

当前环境 **catboost 未安装**。按预注册规则：记录 `unavailable`，实验继续；
不为 secondary control 大改环境。记录文件：
`results/graph_head_function_family/catboost_secondary_status.json`
（`available: false`, `extra_features: []`）。没有 `catboost_secondary_results.csv`。

因此本轮**没有 tree-style reference 数字**，Q14 = unavailable。也不引用/重跑
XGBoost sweep（§29）。

## 11. Parameter accounting

| head | architecture | params |
|---|---|---:|
| H0 | frozen original head `302→64→32→1`（jointly trained） | 0（reference） |
| Hlinear | `302 → 1` | 303 |
| **H1** | `302 → 5 → 2 → 1` | **1530** |
| **FM** | FM(rank=4): bias + linear(302) + 302×4 factors | **1511** |
| H2 | `302 → 13 → 13 → 1` | 4135 |

FM 内部：interaction 302×4 = 1208，linear 302，bias 1 → **1511**。
H1/FM mismatch = `|1530−1511|/1511 = 1.257%` ≤ 3%，parameter-match gate 通过。

## 12. Stage 0 — representation integrity / reconstruction gate

`checkpoint_inventory.json`：10/10 checkpoints present，complete seeds `[0,1]`。
`representation_integrity.json`：seeds 0 与 1 各 5/5 folds 全通过。

| gate | 内容 | 结果 |
|---|---|---|
| G1 | stored `target` == official-train label `y_stored` | max 0.0 |
| G2 | role membership == (7200 fit, 800 selection, 2000 holdout) | pass |
| G3 | re-feed stored `R` through frozen head == stored `yhat_0` | max 0.0 |
| G4 | holdout `yhat_0` == frozen OOF fold prediction | max 0.0 |
| G5 | holdout `R` == corrected centre-incidence export `R` | max 0.0（seed 0；seed 1 n/a） |

`R_dim = 302`，布局 `unary 97 + pair-bucket moments 165 + global 32 + topology 8`。
fingerprint 记录 tokenizer_version（`typed_tokenizer_v1_historical`）、
vocabulary_fingerprint、checkpoint_fingerprint（state `.pt` SHA-256）、
config_fingerprint、split_fingerprint。

## 13. Preprocessing / training protocol

- fit-only coordinate standardisation：`mean_j, std_j` 只由该 fold 的 head-fit 7200
  计算；`z_j = (R_j−mean_j)/max(std_j,1e-6)`，`std_j < 1e-6` 的 degenerate 坐标
  `z_j = 0`。selection/evaluation 只使用 fit statistics。每 fold degenerate 坐标
  11–84 / 302（见 `preprocessing_stats.json`）。
- loss = L1/MAE；optimizer = Adam(lr=1e-3, weight_decay=0)；统一
  **deterministic mini-batch 512**（同一 order 规则），best-selection checkpoint
  = min selection MAE；head_seed = 0。
- horizon 由 fold0/seed0 convergence trace 决定（见下）。

**关于 batch size 的说明（与 frozen adapter 的 full-batch 不同）**：在 7200
fit molecules 上先用 frozen-adapter 的 full-batch 跑了 convergence sanity，发现
400/800 full-batch step 严重欠训练（H1 selection MAE 400→800 仍下降 0.025）。为
避免把「optimization speed」误当成「function family capacity」，本轮改用固定的
deterministic mini-batch 512，对所有 head 完全一致，无 LR/optimizer/batch sweep。
这是一个 pre-registered protocol choice，不是 head 之间的差异。

Convergence trace（fold0, seed0, mini-batch 512，selection MAE）：

| epoch | Hlinear | H1 | H2 | FM |
|---:|---:|---:|---:|---:|
| 100 | 0.19603 | 0.18473 | 0.17755 | 0.22312 |
| 200 | 0.19474 | 0.18418 | 0.17630 | 0.21184 |
| 400 | 0.19564 | 0.18449 | 0.17647 | 0.22433 |
| 800 | 0.19441 | 0.18316 | 0.17681 | 0.22782 |

`max|400−800| = 3.5e-3 > 1e-4`（H1/Hlinear 仍在改善，FM 在 200 后转差 →
best-selection checkpoint 生效）⇒ 预注册规则选 **H = 800**。

Non-collapse / dead-model 诊断（§33，均记录在 `stage*_fold_results.csv` /
`stage*_bootstrap.json`）：H1 initial output std 0.013，H2 0.070，FM 0.325；
initial gradient norm H1 0.377，H2 0.739，FM 非零；final output std H1 1.896，
H2 1.916，FM 2.081（target std ≈ 1.98）。没有 branch collapse。

## 14. Stage 1 results (backbone seed 0 × 5 folds × init 0)

Direct-head evaluation MAE（outer-heldout 2000，mean over folds）：

| H0 | Hlinear (303) | H1 (1530) | H2 (4135) | **FM (1511)** | FM no-interaction |
|---:|---:|---:|---:|---:|---:|
| 0.175809 | 0.187802 | 0.175539 | **0.170051** | 0.172165 | 0.286558 |

| fold | H0 | H1 | H2 | FM | Δsmall = H1−FM | Δstrong = H2−FM |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.178035 | 0.172059 | 0.169355 | 0.165169 | +0.006891 | +0.004186 |
| 1 | 0.192990 | 0.190264 | 0.183997 | 0.184941 | +0.005323 | −0.000944 |
| 2 | 0.178089 | 0.170942 | 0.169067 | 0.184359 | −0.013417 | −0.015292 |
| 3 | 0.169940 | 0.181104 | 0.168304 | 0.167557 | +0.013546 | +0.000747 |
| 4 | 0.159989 | 0.163328 | 0.159533 | 0.158801 | +0.004527 | +0.000732 |
| **mean** | 0.175809 | 0.175539 | 0.170051 | **0.172165** | **+0.003374** | **−0.002114** |

- mean Δsmall = **+0.003374**（median +0.005323），**4/5** folds positive。
- mean Δstrong = **−0.002114**（0/5? 实际 3/5 positive，但 fold 2 大幅为负主导），
  即 FM 比 H2 差 **+0.002114**（在预注册 0.0015–0.004 efficiency band 内）。
- mean Δorig = MAE(H0) − MAE(FM) = **+0.003643**（FM 略胜 jointly-trained H0）。
- bootstrap（stratified molecule-level paired）：Δsmall 95% CI
  `[−0.004398, +0.008361]`，P(>0)=0.835。
- Stage-1 verdict：**EFFICIENCY ADVANCE**（mean Δsmall ≥ 0.003，4/5 positive，
  FM−H2 = +0.00211 ≤ 0.004）。按预注册 §40，这个信号值得 second backbone
  replication。

## 15. Stage 2 / replication decision (backbone seed 1 × 5 folds × init 0)

| H0 | Hlinear | H1 | H2 | **FM** | FM no-interaction |
|---:|---:|---:|---:|---:|---:|
| 0.174253 | 0.186197 | 0.176349 | **0.170508** | 0.175931 | 0.283952 |

| fold | H0 | H1 | H2 | FM | Δsmall | Δstrong |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.178228 | 0.172966 | 0.173622 | 0.185074 | −0.012108 | −0.011452 |
| 1 | 0.192265 | 0.208017 | 0.190456 | 0.190055 | +0.017962 | +0.000401 |
| 2 | 0.171642 | 0.167978 | 0.164973 | 0.170251 | −0.002274 | −0.005278 |
| 3 | 0.169318 | 0.173346 | 0.168894 | 0.172354 | +0.000992 | −0.003459 |
| 4 | 0.159812 | 0.159437 | 0.154593 | 0.161923 | −0.002486 | −0.007330 |
| **mean** | 0.174253 | 0.176349 | 0.170508 | **0.175931** | **+0.000417** | **−0.005424** |

- mean Δsmall = **+0.000417**（median −0.002274），**2/5** folds positive
  —— 低于 Stage-1 CLEAR-NO-GO 线 +0.0005，实质为 0。
- mean Δstrong = **−0.005424**，FM 明显差于 H2。
- bootstrap Δsmall 95% CI `[−0.004378, +0.004220]`，P(>0)=0.604。

**Replication decision**：Stage 1 efficiency signal **没有复现**。这正是预注册
Case E（unstable / fold flip / seed-sensitive）的情形。

## 16. Pooled final decision

两 backbone × 5 folds pooled（`pooled_backbone_results.csv`）：

| head | pooled mean MAE |
|---|---:|
| H0 | 0.175031 |
| Hlinear | 0.186999 |
| H1 | 0.175944 |
| H2 | **0.170280** |
| FM | 0.174048 |

- pooled mean **Δsmall = +0.001896**（< 0.0025 GO 门槛）。
- pooled mean **Δstrong = −0.003769**（FM 差于 H2）。
- pooled mean Δorig = +0.000982。
- fold-averaged Δsmall（先平均两 backbone，再逐 fold）：`[−0.002608, +0.011642,
  −0.007845, +0.007269, +0.001021]` → **3/5** 为正（< 4/5 门槛）。
- 两 backbone mean Δsmall 均 > 0，但 paired bootstrap 95% CI 下界在两个 stage
  都为负（final GO 要求 lower bound > 0）。
- final GO criteria：`pooled ≥ 0.0025` FAIL，`backbone_consistent_positive` PASS，
  `fold_averaged ≥ 4/5` FAIL，`CI lower > 0` FAIL，`fm_interaction_alive` PASS。

⇒ **INCONCLUSIVE（Case E）**。`final_frozen_decision.json`，
`e2e_authorized = false`。

## 17. Interaction mechanism analysis (Stage 1 advanced)

在 backbone seed 0 的 5 个 trained FM 上做 contribution decomposition 与
frozen-inference ablation（`fm_interaction_contributions.csv`,
`fm_interaction_ablation.csv`, `fm_mechanism_summary.json`）：

| quantity | value |
|---|---:|
| std(linear contribution) | 1.8524 |
| std(interaction contribution) | 0.9053 |
| mean absolute interaction contribution | 0.2233 |
| interaction / prediction scale | 0.4272 |
| interaction / linear scale | 0.4887 |
| MAE increase when V=0 (no retrain) | **+0.1144** |
| folds with positive ablation delta | **5/5** |

结论：**the interaction term is alive and carries substantial signal**
（去掉 interaction 使 MAE 上升 0.114）。因此 Stage 1 的微小 efficiency signal
在机制上不是「FM 退化成 linear」。但由于跨 backbone 不稳定，仍不能声称
low-rank interaction 是一个可靠的 head-family advantage。

## 18. End-to-end replacement if authorized

**未运行。** 只有 frozen FM 正式 GO 才授权 E2E-FM（§53–§62）。本轮 frozen verdict
是 INCONCLUSIVE，final gate 未过，`e2e_status.json` 记录 `authorized: false`。
不运行 seed0/seed1 matched pair、不做 MLP+FM hybrid、不扫 rank、不调 LR/weight
decay。`shared_initialization_audit.json` 与 `e2e_seed_results.csv` 因此不存在；
common-initialization primitive（`_copy_shared_upstream`，逐 tensor 复制并记录
hash）已实现并被 Test 12 覆盖，以备未来 GO 使用。

---

## Q1–Q18

- **Q1** frozen `R` 真实维度/组成？`R ∈ R^302` = `unary 97 + pair-bucket moments
  165 + global 32 + topology 8`。见 `representation_integrity.json`。
- **Q2** 为什么 additive hinge 失败让 cross-coordinate interaction 值得测试？
  additive reader 输给 generic MLP（−0.01539，0/5），但 hinge terms 是 live 的
  （内部置零 +0.0478，5/5）→ 说明 nonlinear response 有价值，瓶颈在「无 coordinate
  interaction」；这正是上一轮 decision 的 `revisit_if`。
- **Q3** FM rank4 参数量？**1511**（bias 1 + linear 302 + interaction 1208）。
- **Q4** matched tiny MLP 参数量？**1530**（302→5→2→1）；mismatch 1.257%。
- **Q5** strong MLP 参数量？**4135**（302→13→13→1）。
- **Q6** H0 outer-heldout MAE？seed0 **0.175809**；seed1 **0.174253**；pooled
  **0.175031**。
- **Q7** Linear MAE？seed0 0.187802；seed1 0.186197；pooled 0.186999。
- **Q8** Tiny MLP MAE？seed0 0.175539；seed1 0.176349；pooled 0.175944。
- **Q9** Strong MLP MAE？seed0 0.170051；seed1 0.170508；pooled **0.170280**。
- **Q10** FM MAE？seed0 0.172165；seed1 0.175931；pooled 0.174048。
- **Q11** mean Δsmall？seed0 **+0.003374**（4/5）；seed1 +0.000417（2/5）；pooled
  **+0.001896**。
- **Q12** mean Δstrong？seed0 −0.002114；seed1 −0.005424；pooled **−0.003769**
  （FM 差于 H2）。
- **Q13** fold 方向？seed0 Δsmall = `[+,+,−,+,+]`；seed1 = `[−,+,−,+,−]`；
  fold-averaged = `[−,+,−,+,+]`（3/5 正）。方向不稳定。
- **Q14** CatBoost-MAE 结果？**unavailable**（包未安装，按预注册记录，不阻塞）。
- **Q15** 是否需要 second backbone？Stage 1 命中 EFFICIENCY ADVANCE ⇒ 按预注册
  **必须**跑 second backbone；跑了，且**未复现**。
- **Q16** 若 FM GO，收益是否真正依赖 interaction？非 GO。但机制分析显示
  interaction term 非退化（std 0.905，ablation +0.114，5/5），即 Stage 1 信号若
  存在确实来自 interaction，而非 linear fallback。
- **Q17** 若 E2E 运行，seed0/seed1 是否复现？E2E **未授权 / 未运行**。
- **Q18** 最终属于？**INCONCLUSIVE（Case E）**。不是 STRONG GO、不是
  PARAMETER-EFFICIENCY GO、不是 TREE-FAMILY SIGNAL、也不是干净的 NO-GO：
  Stage-1 小 efficiency signal 在同协议 second backbone 上消失。

## Final verdict

**INCONCLUSIVE — Case E（unstable）— DO NOT EXPAND BUDGET.**

- Stage 1（backbone seed 0）确实命中 EFFICIENCY ADVANCE：FM 比同预算 H1 好
  +0.00337（4/5），比 jointly-trained H0 好 +0.00364，但比 H2 差 +0.00211。
- Stage 2（backbone seed 1）**未复现**：mean Δsmall = +0.00042（2/5），FM 差于 H0
  与 H2。
- Pooled：Δsmall = +0.00190 < 0.0025，fold-averaged 3/5 为正，CI 下界为负。
- 因此 **不授权** end-to-end FM replacement；**不扫 rank**（8/16/learned）；
  **不做** DeepFM / field-aware FM / CrossNet / polynomial network / bilinear head
  sweep；**不做** MLP+FM hybrid。

**Proven（在本 protocol 范围内）：**

- rank-4 low-rank pairwise interaction head 在 frozen compact-v4 `R` 上**不显示
  可复现的**、超过同预算 generic ReLU MLP 的 parameter-efficiency advantage。
- 但它也**不是**一个坏 head：pooled MAE 0.17405 与 H1 0.17594、H0 0.17503 同量级，
  且 interaction term 明确 live（ablation +0.114，5/5）。
- 独立描述性发现：一个 strong direct MLP（H2, 4135）在 untouched 2000 上 pooled
  MAE 0.17028，**优于** jointly-trained original head H0 0.17503，说明 frozen `R`
  仍可被更强的 direct reader 利用，但收益极不稳定。

**Not proven：**

- 不能断言「所有 low-rank / interaction head family 都无效」；只关闭了
  **rank-4 FM on this frozen R** 这一条具体 hypothesis。
- 不能断言「tree-style reader 不适合 MAE」；CatBoost unavailable，没有 tree
  evidence。
- 不能断言 end-to-end FM 一定无效：frozen `R` 由 MLP head 共同训练得到，latent
  geometry 可能天然 MLP-friendly。Frozen NO-GO/INCONCLUSIVE 不自动授权更多 head
  family 搜索，也不授权 E2E。

**下一步（按预注册）：**

> 关闭 frozen-`R` explicit low-rank interaction head 这条线；不扫 rank、不做 E2E。
> 若继续，问题应上升为 representation-family / objective-aligned structured
> computation，并重新预注册自己的 witness。允许的独立 follow-up 只有：一个全新的、
> 预先注册的 function-family hypothesis（不是 FM 的变体），或先解决 CatBoost
> available 后的 tree-style secondary reference。

---

## 复现入口

```
uv run python -m tracks.ksvd.experiments.luyin16.zinc_graph_head_function_family <stage>
# stages: inventory spec cache integrity splits preprocessing convergence
#         stage1 stage1b stage2 mechanism catboost figures decision all
uv run pytest tracks/ksvd/tests/test_graph_head_function_family.py
```

输出目录：`tracks/ksvd/results/graph_head_function_family/`。
