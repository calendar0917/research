# CJP-SSL v1 — 最终报告

**任务书**：`CJP-SSL — Conditional Joint-Path Self-Supervision for LCRC-A`
**执行日期**：2026-10-29
**执行环境**：`res`（`a100-2`，2 × A100-SXM4-40GB，torch 2.5.1+cu124，CUDA 12.4 driver 550.x）
**代码冻结**：`configs/cjp_ssl_v1.json`（sha256 `06c1603f18a75bf5…`）；冻结基线 commit `19aa94b2…`（本任务开始时的 repo HEAD，记录在 `config.git_commit_at_freeze`），运行期的实际 commit 见 §11.6。该 config 在 Phase A 前只追加了一个 `diagnostic_rows` 块（由历史 `lcrcA_s0` 预测导出，**在看到本轮任何 GPU 结果之前冻结**），训练相关字段自首个 commit `86a9eac` 起未变；所有 12 个正式 run 使用的都是 `06c1603f…` 版本。
**数据边界**：只读取官方 PyG ZINC `subset=True` train 的 `x/edge_index/edge_attr`；**official valid 仅用于冻结的评估协议**；**official test 从未被加载**（代码中无 test 路径，`analysis/*.json` 均记录 `official_test_loaded: false`）

---

## 0. 结论（先行）

| 判决 | 值 | 依据（seed-mean，delta = 前者 − 后者，负值表示前者更好） |
|---|---|---|
| **D `NO_GAIN`** | **成立** | MARG−A-REPLAY `+0.00752` `[+0.00114, +0.01509]`；JOINT−A-REPLAY `+0.00572` `[+0.00027, +0.01230]`；两 seed 同向变差；ordinary、去 row172、r2-unseen ordinary 全部无改善 |
| A `LOCAL_SSL_SUPPORTED` | 不成立 | 无一 arm 达到「两 seed 改善 + seed-mean ≥0.003 + ordinary/去 row172 同向改善」 |
| B `JOINT_BINDING_SUPPORTED` | 不成立 | JOINT−MARG `−0.00180` `[−0.00632, +0.00232]`，两 seed 符号相反；ordinary `+0.00077`、r2-unseen ordinary `+0.00522`（JOINT 更差） |
| C `MARG_ONLY_GAIN` | 不成立 | MARG 本身未达到 A 的门槛 |
| 次级标签 `JOINT_PRETEXT_LEARNED_NO_DOWNSTREAM_GAIN` | **成立** | JOINT held-out pretext CE seed-mean `0.2774` < MARG `0.2873`，JOINT raw-logit 交互 RMS `3.0–3.2`（确有非可加结构），但下游 full MAE 无收益 |

一句话：**在新协议下，conditional joint-path SSL 预训练 LocalContent 不能改善 LCRC-A；旧 atom/bond SSL 的贡献（≈ −0.0081 MAE，−9.3%）被干净复现且远大于任何新协议效应。因此不应把本方案设为默认预训练。**

---

## 1. 冻结要求 → 实现 → 证据

| 冻结要求 | 实现 | 证据 |
|---|---|---|
| LCRC-A 推理不变，部署参数 237,519 | 只在 `LocalContent` 前面加 pretext head；`build_supervised_model` 用 `DEPLOY_PARAMETERS` 断言 | 8 个监督 run 全部 `params=237519`；G12 导出/部署检查 |
| pretext head `Linear(192,128)→SiLU→Linear(128,63)` | `cjp_models.PretextHead` | 32,831 params；G12 |
| `atom_encoder`/`bond_encoder` 冻结，`f`/`rho` 不参与预训练 | `build_ssl_model(..., freeze_encoders=True)`；优化器只含 local+head | 预训练前后 encoder SHA 完全相同（`encoder_bit_identical=true`，4/4 run）；grad norm 记录只含 `local`/`head` 两组；G10/G11 |
| 复用历史 atom/bond SSL checkpoint | `lc_models.ssl_path(seed)`，只加载 `atom_encoder`/`bond_encoder`（`junction_encoder` 不加载） | s0 `006fabc1…`、s1 `ff8d6474…`，与冻结 config 一致 |
| 只用 train 的 `x/edge_index/edge_attr` | `cjp_data.load_raw_train` 只取三字段；`cjp_gates.G1` 静态扫描 + 动态 poison 测试 | G1 pass；pretrain payload `data.labels_read_by_ssl=false` |
| 按 canonical fingerprint 分组，≈9,500/500，重复不跨 split | typed WL（5 轮）指纹；按固定 SHA 排序切分 | fit `82490f79…` / diag `695323da…` / order `ca90733c…`；9,997 唯一指纹、3 组重复（6 分子，互相同构） |
| 所有有向真实二跳路径 `(v,w,u)`，每次访问 ≤3 条顶点不相交路径，`SeedSequence([seed, molecule_id, visit_index, 20261029])` | `cjp_data.sample_paths` | G8（重编号/反向/batching 不变性）；mask RNG 与 batch RNG 分离（`[seed, 20261030]`） |
| MARG/JOINT 看到完全相同的 path/batch | 两 arm 共用同一 `prepare_visit` 输出，各自独立构建模型 | `same_samples_across_arms` 两 seed 全 true（masked batch SHA、first-batch query SHA、split、coverage 全等） |
| `x[u]:=27`，物理键 `(w,u)` 两条有向记录 `:=0`；从 masked raw graph 经 `build_mol_objects` 重建 P3；S1/S2 可见性过滤；拓扑不变；train-only 归一化 | `apply_mask` + `C.M.build_mol_objects` + `compose_masked`；归一化来自冻结 P3 统计 | G2（关掩码后与冻结 cache **max_abs_diff=0.0**，10,000 分子）、G3（独立 naive 参考实现逐元素一致）、G5（两条有向记录都置 0）、G6（mask token 不被当作真实类型）、G7（无 mask 前向与 LCRC 完全一致） |
| 目标 `3*raw_atom_type(u)+(raw_bond_type(w,u)-1) ∈ [0,62]` | `cjp_data.queries` | G6（63 类、值域/词表检查）；不是 28×4 |
| MARG = CE on `flatten(L_marg)`；JOINT = CE on raw `flatten(L)`；同 head/优化器/算力 | 同一 `PretextHead`，仅 logits 变换不同 | G9（2×2 XOR 反例嵌入真实 21×3 空间 + 投影残差 ≤1e-6；MARG/JOINT 共享 init hash `a3e8d8ad…`） |
| 损失 = 分子内选中路径平均 CE，再对有路径分子平均 | `cjp_models.per_molecule_loss` | 与任务书公式一致；空路径分子被排除（本次 0 个） |
| AdamW 1e-3 常数、WD 1e-5、batch 128、clip 5、7,560 updates、dropout 0、FP32 | `PRETRAIN_RECIPE`，逐项断言 | 4 个预训练 run 均 7,560 updates；`recipe` 快照在 payload |
| 导出最后一步 LocalContent | `local_state.pt` + `pretext_head_state.pt`，附 SHA | 4/4 导出；监督阶段 `module_sha256.local` == 预训练 `local_state.state_sha256`（见 §6 转移校验） |
| 四个监督 arm，仅 `local` 状态不同，其余初始化一致 | `cjp_sup.py` 临时 patch `lc_models.build_model`，其余直接调用冻结 `lc_train.train` | `cross_arm_module_sharing`：encoders/f/rho 在 replay/marg/joint 间完全相同；MARG vs JOINT 仅 `local` 不同；数据顺序在 4 arm 间逐 seed 完全一致（`first_batch_rows` 相等） |
| 冻结 `lc_train.train` 配方（600 ep / 47,400 upd / batch128 / AdamW 1e-3→1e-5 / warmup500 / cosine / WD1e-5 / clip5 / 每 epoch full valid / best-MAE checkpoint） | 不复制、直接 import 冻结模块（G0 校验其 SHA） | 8/8 run `epochs=600, updates=47400`；lr 曲线与冻结实现一致；`git_commit` 记录在 payload |
| 从不加载 official test | 无 test 路径 | `official_test_loaded: false`（全部 JSON） |

---

## 2. CPU 门禁（G0–G12 + lineage）

`analysis/cpu_checks.json` → `all_pass: true`；`uv run pytest -q tracks/structural-sufficiency/zinc_cjp_ssl_v1/tests` → **13 passed**（46.9 s）。

| gate | 内容 | 结果 |
|---|---|---|
| G0 | 冻结 trainer 血统（`lc_train`/`lc_models`/`lc_data`/`lc_common` SHA） | pass |
| G1 | label 隔离（静态扫描 + 动态 poison `y`） | pass |
| G2 | 关掩码后与冻结 cache 逐字节一致（10,000 分子，`max_abs_diff=0.0`） | pass（25.2 s） |
| G3 | masked 重建 vs 独立 naive 参考实现（含 S1/S2 可见性、两条有向记录） | pass |
| G4 | hidden-truth swap：210 atom + 210 bond swap 全部改变预测 | pass |
| G5 | 两条有向记录都被 mask | pass |
| G6 | 无 mask token 被当作真实类型；63 类词表/值域 | pass |
| G7 | 无 mask 前向与冻结 LCRC 完全一致 | pass |
| G8 | 重编号 / 反向 / batching 不变性 | pass |
| G9 | MARG 因子分解证明（2×2 XOR 反例 + 21×3 空间）+ 共享 init hash | pass |
| G10 | 梯度隔离（encoder/f/rho 无梯度） | pass |
| G11 | LocalContent 全部张量有限且非零梯度（如 `mlp.lins.0` 2.545、`path_emb` 0.0207） | pass |
| G12 | 导出/部署兼容（237,519 params，MARG/JOINT 仅 `local` 不同） | pass |

---

## 3. Phase A：基线复现

| seed | 历史 LCRC-A best | 同轮 A-REPLAY best | Δ | 历史 −row172 | 同轮 −row172 | Δ |
|---|---|---|---|---|---|---|
| 0 | 0.076582 | **0.076116** | −0.00047 | 0.070076 | 0.070165 | +0.00009 |
| 1 | 0.080504 | **0.082301** | +0.00180 | 0.070115 | 0.070559 | +0.00044 |

复现证据链（比终点指标更强）：

1. `initial_state_sha256` 与历史 run 完全相同（s0 `11c698ae…`，s1 `d9422859…`）；
2. `first_batch_rows` 与历史 run 完全相同；
3. **第 1 个 epoch 的 full-valid MAE 与历史一致到 1e-7/1e-8**（s0：0.6003845175 vs 0.6003844830；s1：0.6090951305 vs 0.6090951543），第 2 epoch 同样一致到 1e-6/1e-7；
4. 去 row172 后两 seed 偏差仅 +0.00009/+0.00044 —— 与「同一 pipeline 在 GPU 上非确定性放大」一致。

**全量 MAE 的复现偏差几乎全部来自 valid:0172 这一行**（`y=−20.34`）：row172 绝对误差在四次同 pipeline 运行中为 6.02（replay s0）/ 6.58（历史 s0）/ 10.46（历史 s1）/ 11.81（replay s1）。单这一行的误差差 5.8 就等价于 full MAE 0.0058。这与任务书的诊断设计一致；所有主比较都使用**同轮 A-REPLAY**，历史数字只用于一致性检查。

---

## 4. Phase B/C：预训练诊断

7,560 updates / batch 128 / 4 次运行；MARG 与 JOINT 在每个 seed 内看到**完全相同**的 mask、batch、路径。

| run | held-out CE（init→final） | class prior CE | cond. prior CE | raw-logit interaction RMS | eff. rank | act RMS | swap Δ | cross-mol shuffle Δ | wall |
|---|---|---|---|---|---|---|---|---|---|
| MARG s0 | 4.0695 → **0.3182** | 1.6102 | 1.4198 | 4.382（init 0.169） | 22.85 → 29.41 | 0.757 → 0.970 | +3.58 | +8.21 | 0.472 h |
| MARG s1 | 4.1135 → **0.2564** | 1.6123 | 1.4258 | 4.320（init 0.141） | 26.35 → 28.27 | 0.820 → 0.966 | +3.24 | +7.28 | 0.478 h |
| JOINT s0 | 4.0589 → **0.2972** | 1.6102 | 1.4198 | 3.209（init 0.169） | 22.85 → 29.49 | 0.757 → 0.951 | +3.86 | +7.61 | 0.488 h |
| JOINT s1 | 4.1238 → **0.2577** | 1.6123 | 1.4258 | 2.973（init 0.141） | 26.35 → 29.80 | 0.820 → 0.958 | +3.25 | +7.18 | 0.485 h |

- **掩码覆盖**（4 run 完全相同）：原子 12.94%、键 12.02%，平均 3.00 条路径/分子，0 个分子无路径。每条被选路径恰好 mask 1 个原子（`u`）与 1 条物理键（`(w,u)` 的两条有向记录）；root `v` 保持可见。
- **同样本校验**：两 seed 的 `masked_batch_sha256`、`first_batch_query_sha256`、split、coverage 在 MARG/JOINT 间全等（`same_samples_across_arms` 全 true）。
- **可学性**：held-out CE 从 ≈4.06–4.12 降到 0.26–0.32，显著优于 class prior（1.61）与 conditional prior（1.42）。这不是「没学会」。
- **干预**：swap `h_v↔h_w` 使 loss 从 ≈0.28 升到 3.2–3.9；跨分子 shuffle 升到 7.2–8.2。head 同时依赖 root 身份、middle 身份与该分子自身结构。
- **非可加性**：JOINT raw logits 的交互分量 RMS 2.97–3.21（init 0.14–0.17），说明 joint 目标确实被拟合出非可加结构。
- **表示几何**：LocalContent 有效秩 22.9–26.4 → 28.3–29.8，activation RMS 0.76–0.82 → 0.95–0.97；预训练把局部表示拉高秩、放大尺度。
- **梯度**：最后一步 `local.mlp.lins.0.weight` 0.27–0.36、`nbr_emb` 0.011–0.015、head 0.209–0.422；每组只有 `local`/`head` 两个梯度组，encoder 组**没有任何梯度条目**（冻结），f/rho 不在优化器中。
- **注意（实现细节）**：MARG 的损失只依赖可加投影，原始 logits 的不可加分量位于该损失的零空间中，训练中自由漂移（0.17→4.3+）。因此上表 MARG 的 raw-interaction RMS 不能解读为「学到的交互」；判断 learned joint binding 应看 JOINT 的 2.97–3.21。MARG 实际用于 CE 的 logits 严格可加。

---

## 5. 监督结果（official valid 1,000，raw y，best-by-valid checkpoint）

| arm | seed 0 | seed 1 | seed mean | 相对 A-REPLAY（mean） |
|---|---|---|---|---|
| **A-REPLAY**（历史 SSL encoders + 标准 LocalContent） | **0.076116** | **0.082301** | **0.079209** | — |
| SCRATCH（无 SSL，标准 LocalContent） | 0.083585 | 0.091069 | 0.087327 | +0.00812 |
| MARG-SSL | 0.088419 | 0.085031 | 0.086725 | +0.00752 |
| JOINT-SSL | 0.083514 | 0.086343 | 0.084929 | +0.00572 |

历史 LCRC-A 参考：0.076582 / 0.080504（mean 0.078543）。

| run | best ep | W20 | T100 | final | train@best（eval） | params | wall |
|---|---|---|---|---|---|---|---|
| A-REPLAY s0 | 419 | 0.077020 | 0.077050 | 0.077062 | 0.00936 | 237,519 | 0.341 h |
| A-REPLAY s1 | 473 | 0.082456 | 0.082496 | 0.082495 | 0.00499 | 237,519 | 0.435 h |
| SCRATCH s0 | 424 | 0.084240 | 0.084301 | 0.084334 | 0.01267 | 237,519 | 0.355 h |
| SCRATCH s1 | 477 | 0.091623 | 0.091689 | 0.091699 | 0.00666 | 237,519 | 0.466 h |
| MARG-SSL s0 | 442 | 0.089060 | 0.089093 | 0.089041 | 0.00731 | 237,519 | 0.343 h |
| MARG-SSL s1 | 445 | 0.085265 | 0.085299 | 0.085278 | 0.00750 | 237,519 | 0.391 h |
| JOINT-SSL s0 | 505 | 0.083763 | 0.083804 | 0.083792 | 0.00517 | 237,519 | 0.443 h |
| JOINT-SSL s1 | 441 | 0.086788 | 0.086818 | 0.086834 | 0.00768 | 237,519 | 0.481 h |

（W20/T100/final 与 best 几乎相同 → best checkpoint 不是单点噪声；所有 arm 的 `train@best` 都远低于 valid，无过拟合迹象。）

分组 MAE（seed 0 / seed 1）：

| arm | full | −row172 | −A worst10 | ordinary | tail | r2-unseen ord | seen ord | ring-rich unseen |
|---|---|---|---|---|---|---|---|---|
| A-REPLAY | 0.0761/0.0823 | 0.0702/0.0706 | 0.0641/0.0638 | 0.0645/0.0635 | 0.243/0.353 | 0.1343/0.1311 | 0.0424/0.0421 | 0.225/0.289 |
| SCRATCH | 0.0836/0.0911 | 0.0750/0.0785 | 0.0692/0.0714 | 0.0702/0.0706 | 0.276/0.385 | 0.1414/0.1478 | 0.0477/0.0462 | 0.261/0.324 |
| MARG-SSL | 0.0884/0.0850 | 0.0778/0.0745 | 0.0713/0.0667 | 0.0642/0.0665 | 0.437/0.352 | 0.1353/0.1363 | 0.0417/0.0443 | 0.319/0.307 |
| JOINT-SSL | 0.0835/0.0863 | 0.0725/0.0754 | 0.0652/0.0683 | 0.0662/0.0660 | 0.332/0.379 | 0.1456/0.1365 | 0.0411/0.0437 | 0.335/0.308 |

组大小：full 1,000；−row172 999；−worst10 990；ordinary（`y>−3`）935；tail（`y≤−3`）65；r2-unseen ordinary 225；r2-unseen 251；seen ordinary 710；ring-rich unseen 47。

---

## 6. 配对比较（seed-mean；delta = first − second，负 = first 更好）

| contrast | mean Δ | CI95 | ordinary | −row172 | −worst10 | r2-unseen ord | wins（s0/s1） | median（s0/s1） |
|---|---|---|---|---|---|---|---|---|
| A-REPLAY − SCRATCH | **−0.00812** | [−0.01409, −0.00274] | −0.00642 | −0.00638 | −0.00628 | −0.01187 | 0.548 / 0.545 | −0.00404 / −0.00362 |
| MARG − A-REPLAY | +0.00752 | [+0.00114, +0.01509] | +0.00134 | +0.00574 | +0.00499 | +0.00312 | 0.491 / 0.484 | +0.00054 / +0.00080 |
| JOINT − A-REPLAY | +0.00572 | [+0.00027, +0.01230] | +0.00210 | +0.00358 | +0.00278 | +0.00834 | 0.522 / 0.480 | −0.00189 / +0.00116 |
| JOINT − MARG | −0.00180 | [−0.00632, +0.00232] | +0.00077 | −0.00217 | −0.00221 | +0.00522 | 0.511 / 0.499 | −0.00061 / +0.00003 |

逐 seed 的 `MARG−A`：s0 `+0.01230 [+0.00112,+0.02650]`，s1 `+0.00273 [−0.00262,+0.00808]`；`JOINT−A`：s0 `+0.00740 [−0.00198,+0.02055]`，s1 `+0.00404 [−0.00080,+0.00921]`。bootstrap 为按分子的 10,000 次配对重采样。

**读法**：

1. **旧 SSL**：两个 seed、每一个预定义子群（ordinary、tail、去 row172、去 worst10、r2-unseen ordinary、seen ordinary、ring-rich unseen）都一致变好，wins fraction > 0.54，median Δ < 0。这是本报告唯一稳健的正效应。
2. **新 LocalContent SSL**：full MAE 与去 row172 两个 seed 都更差；ordinary 与 r2-unseen ordinary 也无改善。MARG s0 的 ordinary −0.00028 是 935 行上的 3e-4 级差异，s1 立即反号为 +0.00296，不能算收益。
3. **JOINT vs MARG**：seed-mean 略偏 JOINT（−0.0018），但两 seed 符号相反、CI 跨 0；ordinary 与 r2-unseen ordinary 反而 JOINT 更差。

---

## 7. seed-1 gate 的实际执行

任务书 gate：若 MARG 与 JOINT **都**比同步 A_s0 差 ≥ +0.010 full MAE，且去极端行后仍 ≥ +0.005、ordinary/unseen 均更差 → 停止 seed1；否则购买 seed1。

实际（s0）：MARG +0.01230（满足 ≥0.010）、JOINT +0.00740（不满足）、JOINT −row172 +0.00235（不满足 ≥0.005）。**gate 未触发 → 按协议执行了 Phase C（seed1 预训练 + 监督）与 Phase D（SCRATCH s0+s1）**，未因「看起来不好」提前停止。seed1 结果（MARG +0.00273、JOINT +0.00404）与 s0 同向，未改变判决。

---

## 8. 五个必须回答的问题

**Q1. 旧 SSL 对 LCRC-A 到底贡献多少？**
**−0.00812 MAE（−9.3% 相对），CI95 [−0.01409, −0.00274]，两 seed、所有子群一致。** 具体：ordinary −0.00642、tail −0.0325、去 row172 −0.00638、r2-unseen ordinary −0.01187、seen ordinary −0.00469、ring-rich unseen −0.0355。旧 atom/bond SSL 是真实、可复现、非单行驱动的收益。

**Q2. 预训练 LocalContent 是否比标准初始化更有利？**
**不。** MARG +0.00752 `[+0.00114,+0.01509]`、JOINT +0.00572 `[+0.00027,+0.01230]`（正 = 更差），两 seed 同向。去 row172（+0.0057/+0.0036）与 ordinary（+0.0013/+0.0021）也都不支持「被 row172 掩盖的真实收益」。

**Q3. 联合预测是否优于边际？**
**下游没有证据（−0.00180，CI 跨 0，两 seed 反号）；pretext 上 JOINT 更好（CE 0.2774 vs 0.2873，seed mean），且确实学到非可加交互（RMS 3.0–3.2）。** 即 `JOINT_PRETEXT_LEARNED_NO_DOWNSTREAM_GAIN`。逐 seed 的 pretext CE 并不稳定：s0 JOINT 0.2972 < MARG 0.3182，s1 JOINT 0.2577 > MARG 0.2564。

**Q4. 改善是否发生在 ordinary 与 r2-unseen ordinary，而不是单个异常分子？**
**没有任何改善。** 两个 arm 相对 A-REPLAY 的 ordinary 为 +0.00134/+0.00210（更差），r2-unseen ordinary 为 +0.00312/+0.00834（更差）。相反，**旧 SSL 的收益**正好发生在 ordinary（−0.00642）与 r2-unseen ordinary（−0.01187）上。

**Q5. 现在是否应把它设为默认预训练方式？**
**不应该。** 判决 D：停止当前 conditional path SSL 方案（仅否定本任务书固定的路径/目标/初始化协议）。默认预训练应继续使用旧 atom/bond SSL；LocalContent 用标准初始化。

---

## 9. 机制讨论

1. **为什么 full MAE 在本任务里如此难读**：valid 只有 1,000 行，其中 `y=−20.34` 的 row172 一行独占 0.006–0.013 的 MAE。同一条 pipeline（相同 init、相同数据顺序、第 1 epoch 一致到 1e-7）在这一行上的误差可以是 6.0 或 11.8。因此「full MAE 差 0.006」本身不构成机制证据；本报告同时给出 −row172、−worst10、ordinary、tail、unseen 五个切面。**所有切面都指向同一结论**：新 SSL 无收益，旧 SSL 有收益。
2. **预训练确实学会了目标，但没迁移**：held-out CE 0.26–0.32 远低于 conditional prior 1.42，干预实验证明 head 使用了 root/middle/分子三重信息，JOINT 的 raw-logit 交互 RMS 3.0–3.2。失败点在迁移，不在优化。
3. **预训练改变了 LocalContent 的几何**：有效秩 +1.9–6.6、activation RMS +0.14–0.21。这与「SSL 初始化把局部表示推向更高秩、更大尺度」一致；在冻结的监督配方（warmup 500、peak LR 1e-3、per-step cosine、600 epochs）下，这种初始化没有转化为更好的解，反而在 full MAE 上更差。任务书禁止调参，因此这是「同一配方 + 不同初始化」的干净比较，而不是对 SSL 理念的一般性否定。
4. **旧 SSL 有效的可能原因**：atom/bond encoder 是所有下游读取的共享前端；它的预训练直接改变 frozen 输入特征。LocalContent 只是 133k 参数的局部模块，且其输入已被冻结 encoder 决定；在同一配方下，重新初始化它带来的差异更容易被监督信号在 600 epoch 内抹平/覆盖。
5. **MARG 与 JOINT 的差别在下游消失**：pretext 上 JOINT 拟合了非可加交互（RMS 3.2 vs MARG 严格可加），但下游 ordinary/unseen 反而略差。说明「能拟合 joint 交互」与「该交互对下游有用」是两件事。

---

## 10. 限制（明确不外推）

- **2 个 seed**；结论是 seed-mean + 方向一致性，不是显著性的强断言。CI 由按分子 bootstrap 得到，未覆盖 seed 间方差。
- **full MAE 被单行主导**；判决按任务书使用 full MAE，但同时报告了去极端行/子群结果，且结论不依赖 row172。
- 预训练只跑 **7,560 updates**、单一 LR/WD/掩码率/头宽（全部冻结，未调参）；不能排除其他预算下的不同结果。
- 监督配方是为「旧 SSL + 标准 LocalContent」设计的；本任务禁止为其改动，因此结论限定于该配方。
- 只测 LCRC-A（`f`/`rho` 不参与预训练），不外推到 LCRC-B/C 或其他下游。
- 只测 ZINC 10k train 的官方 valid 1,000；official test 从未加载。

---

## 11. 交付物、哈希与预算

### 11.1 文件

| 交付 | 路径 |
|---|---|
| 设计/冻结记录 | `DESIGN.md` |
| 代码 | `code/cjp_common.py`, `cjp_data.py`, `cjp_models.py`, `cjp_pretrain.py`, `cjp_sup.py`, `cjp_gates.py`, `cjp_freeze.py`, `cjp_analysis.py`, `cjp_ledger.py` |
| 测试 | `tests/test_cjp_gates.py`（13 passed） |
| 冻结配置 | `configs/cjp_ssl_v1.json`（sha256 `06c1603f18a75bf5…`） |
| CPU 门禁 | `analysis/cpu_checks.json`（`all_pass: true`） |
| pretext 摘要 | `analysis/pretext_summary.json` |
| 监督摘要 | `analysis/supervised_summary.json` |
| 配对比较 | `analysis/paired_comparisons.json` |
| 逐分子 | `analysis/per_molecule_valid.csv` |
| GPU 账本 | `analysis/gpu_ledger.json` |
| 自动摘要 | `analysis/summary.md` |
| 报告/决策 | `REPORT.md`, `NEXT_DECISION.md` |

### 11.2 冻结 SHA（关键）

| 对象 | SHA256（前 16） |
|---|---|
| 冻结 config | `06c1603f18a75bf5` |
| lc_models / lc_data / lc_train / lc_common | 见 `configs/cjp_ssl_v1.json:reference`（G0 校验） |
| LCRC train cache / valid cache | `8777494f7745ff5a` / `aae2caa50d322a25` |
| 归一化指纹 | `8c3e88137f18ca68` |
| fit / diag index | `82490f7944d5ef7c` / `695323dafb81e763` |
| fingerprint order | `ca90733c43699291` |
| 旧 SSL ckpt s0 / s1 | `006fabc11efede09` / `ff8d6474896c2744` |
| LocalContent 标准 init（s0） | `58160a29209475b7` |
| MARG/JOINT 共享 pretext init | `a3e8d8adf16e0df2` |

### 11.3 监督 checkpoint（best / final，前 16）

| run | init state | best | final |
|---|---|---|---|
| A-REPLAY s0 | `11c698ae669dbee1` | `036fbd9b32472ad3` | `d999269f58592e89` |
| A-REPLAY s1 | `d9422859dc47bbe2` | `abe0bcbddd75115b` | `d8b19ebbcce31b34` |
| SCRATCH s0 | `e7e26cbf34ddcfb5` | `314e6a869dad0793` | `e72f87cde588c560` |
| SCRATCH s1 | `0bea72728e13dae7` | `4ef35c5dce641cfa` | `1696f871b4ad6d2b` |
| MARG-SSL s0 | `18d15ce75d653782` | `8a2cb5658f2808aa` | `08161d8960ee09c8` |
| MARG-SSL s1 | `f306930986be9acc` | `1f7f9f871b7bd803` | `6a05ae885e0f657e` |
| JOINT-SSL s0 | `2e9922a2aedb549d` | `59f33be5bb12fe0b` | `1dfcbf9aa159673d` |
| JOINT-SSL s1 | `c810d9d6670d5ccf` | `9dbbc0789f9e35d8` | `4ce93f97eee085fb` |

### 11.4 预训练导出（file SHA256，前 16）

| run | local_state.pt | pretext_head.pt |
|---|---|---|
| pretrain_marg_s0 | `13fbcf5c73e033d8` | `87695f2120e65552` |
| pretrain_marg_s1 | `d2ae8fd1284833ce` | `5338145364dc6e89` |
| pretrain_joint_s0 | `57dfed91e7ef4ebf` | `9ff170a0c63368d3` |
| pretrain_joint_s1 | `de0b30d26cd5ba0c` | `226eb0d5a34b5703` |

转移校验：监督 run 的 `module_sha256.local` 与对应预训练 `local_state.state_sha256` **完全相等**（MARG s0 `81790726…`、JOINT s0 `3427a123…` 等，4/4），且 encoders/f/rho 在 arm 间相同、MARG 与 JOINT 仅 `local` 不同。

### 11.5 预算

| 项 | 值 |
|---|---|
| GPU·h（12 个正式 run，计入账本） | **5.177** |
| 其中预训练 | 1.922 h（4 run） |
| 其中监督 | 3.255 h（8 run） |
| smoke（未入账，仅供参考） | ≈0.02 h |
| 软上限 10 / 硬上限 16 GPU·h | 未超 |
| wall clock（首次 smoke 07:56 → 最后一个 run 11:12） | 3 h 16 min（硬上限 8 h，未超） |
| official test 加载 | 从未 |

GPU 账本明细：`analysis/gpu_ledger.json`（含 device regime：`res / A100-SXM4-40GB / torch 2.5.1+cu124`）。

### 11.6 执行 regime（provenance）

| 阶段 | rr job（run id） | 代码 commit |
|---|---|---|
| smoke pre / sup | `cjp-smoke-pre` (12b332d6) / `cjp-smoke-sup` (fe3c32a9) | `86a9eac2e5e7` |
| Phase A replays | `cjp-A-replay-s0` (bae52a72) / `-s1` (67e77e85) | `86a9eac2e5e7` |
| Phase B pretrain s0 | `cjp-pre-marg-s0` (28430b2e) / `cjp-pre-joint-s0` (bb193e1b) | `ab48d7c0ccc3` |
| Phase B supervised s0 | `cjp-margssl-s0` (aa789f1f) / `cjp-jointssl-s0` (65e3fb0f) | `ab48d7c0ccc3` |
| Phase C pretrain s1 | `cjp-pre-marg-s1` (d8408c4c) / `cjp-pre-joint-s1` (56a4ae6c) | `ab48d7c0ccc3` |
| Phase C supervised s1 | `cjp-margssl-s1` (c5e5d975) / `cjp-jointssl-s1` (41e59d84) | `3abf25c009b3` |
| Phase D scratch | `cjp-scratch-s0` (962c29d1) / `cjp-scratch-s1` (50c23196) | `3abf25c009b3` |

三个运行 commit 之间 `cjp_common.py`/`cjp_data.py`/`cjp_models.py`/`cjp_sup.py` **完全未变**；`cjp_pretrain.py` 只在 Phase A 之后增加了逐 batch mask-SHA 记录（诊断字段，不参与计算），4 个预训练 run 全部使用同一版本；`cjp_analysis.py` 在 Phase D 前只增加了转移校验与 markdown 输出，不改变任何数值定义。每个 payload 自带 `git_commit` 与 device regime，可用 `analysis/*.json` 复核。

---

## 12. 可复现命令

```bash
RR=/home/calendar/.pi/agent/skills/remote-research-runner/bin/rr
# 预训练
$RR run res cjp-pre-marg-s0 --pool res-gpu0 --result tracks/structural-sufficiency/zinc_cjp_ssl_v1/runs \
  -- python tracks/structural-sufficiency/zinc_cjp_ssl_v1/code/cjp_pretrain.py --arm MARG --seed 0 --device cuda:0
# 监督（replay/scratch/margssl/jointssl）
$RR run res cjp-A-replay-s0 --pool res-gpu0 --result tracks/structural-sufficiency/zinc_cjp_ssl_v1/runs \
  -- python tracks/structural-sufficiency/zinc_cjp_ssl_v1/code/cjp_sup.py --mode replay --seed 0 --device cuda:0
# 分析与账本（本地）
uv run python tracks/structural-sufficiency/zinc_cjp_ssl_v1/code/cjp_analysis.py
uv run python tracks/structural-sufficiency/zinc_cjp_ssl_v1/code/cjp_ledger.py
```
