# LCRC-v1 最终报告 — Local Contribution + Relation-Aware Composition

**Round:** `zinc-local-relation-composition-v1`  
**Track:** `tracks/structural-sufficiency/zinc_local_relation_composition_v1/`  
**状态:** 完成（6/6 正式 run：A/R/N × seed 0/1；test 已由用户要求做一次 post-hoc 描述性读取：A_s0 = 0.066782；非 gate 通过）  
**一句话结论:** 不带 junction token 的 A 在 seed 0 名义上优于 DCC-L seed 0，但两个 seed 的均值没有超过 DCC-L 四 seed 均值；R 相对 N 的收益在两个 seed 上符号翻转；机制检查显示关系读头坍缩为近似常数偏置，真实结构对应没有被冻结模型使用。预注册 gate 未通过；用户随后明确要求对最佳 valid 单模型 A_s0 读一次 test，得到 **0.066782**，落在 DCC-L s0 0.066267 与四 seed 均值 0.067319 之间，统计上与 DCC-L 不可区分。该读取是 post-hoc、非 confirmatory，不改变 stop 结论；本轮仍然没有 `<0.060` 的信号。

---

## 1. 执行摘要

| 项目 | 结果 |
|---|---|
| 正式 run | 6 个：A_s0 / A_s1 / R_s0 / R_s1 / N_s0 / N_s1 |
| 单模型参数 | A = 237,519；R = N = 269,664（均 < 500k） |
| 训练 | 每个 run 600 epochs = 47,400 updates；raw-y L1；单模型 |
| 正式训练 GPU 时间 | **2.9501 GPU·h**（目标 6，硬上限 8） |
| 最大并发 | 2 × A100-40GB（`res-gpu0` / `res-gpu1`） |
| 最佳 valid MAE | **A_s0 = 0.076582** |
| A 两 seed 均值 | 0.078543 |
| R 两 seed 均值 | 0.079032 |
| N 两 seed 均值 | 0.080075 |
| 历史 DCC-L valid 四 seed 均值 | 0.075990（未在本轮重训） |
| 历史 DCC-L test | s0 = 0.066267；四 seed 均值 = 0.067319（未在本轮重训） |
| R−N | seed0 **−0.003950**，seed1 **+0.001866**；符号翻转 |
| test | **A_s0 post-hoc = 0.066782**；pre-registered gate 未通过；仅描述性，非 confirmatory |
| CPU gates | 8/8 PASS（`analysis/cpu_checks.json`） |

**最重要的四个事实：**

1. **A 的局部主体是有效的，但优势不稳定。** A_s0 0.076582，优于 DCC-L seed0 0.078537（逐分子 paired Δ = −0.001956，CI 含 0）；A_s1 0.080504，明显差于 DCC-L seed1 0.073837。A 两 seed 均值 0.078543，比 DCC-L 四 seed 均值 0.075990 差 0.002553。不能声称“达到或超过 DCC-L”。
2. **R 没有稳定优于 N。** seed0 R 比 N 好 0.003950，seed1 R 比 N 差 0.001866；均值 R−N = −0.001042，但两个 seed 方向相反。即使 seed0 的 R−N 增益，其 paired bootstrap 95% CI 也包含 0（[−0.009299, +0.000882]）。
3. **关系通道在功能上是关闭的。** R 的 `Δ_rel` 在分子间的极差只有 ~2×10⁻⁵（两个 seed）；冻结权重把真实归属换成 mean-field 归属，valid MAE 变化 ≤1×10⁻⁸；把 `rel.head` 置零只改变常数偏置（R_s0 −1.4×10⁻⁵；R_s1 +5.95×10⁻³）。因此 R 与 N 的差异不是“真实结构对应”的证据，而是零初始化读头坍缩为常数偏置后的优化轨迹差异。
4. **用户在 gate 失败后要求读了一次 test，A_s0 = 0.066782。** 它比 DCC-L s0 0.066267 差 +0.000515，比 DCC-L 四 seed 均值 0.067319 好 −0.000537；paired test bootstrap CI 跨 0（[−0.007302, +0.007178]）。这是单 seed、post-hoc、test 已暴露的描述性记录，不能作为“超过 DCC-L”的 confirmatory 证据，也没有改变 stop 决定。

---

## 2. 合规、数据与 provenance

### 2.1 数据与选择

- 只使用官方 PyG ZINC `subset=True` 的 **train 10,000 / valid 1,000 / test 1,000**；没有创建新的 Fit/Monitor/Locked split。
- 训练只用 train；标准化只用 train 拟合（`global33` 复用冻结 P3 train stats，9 个原子标量用 train mean/std）；valid 每 epoch 全量评估并严格按最小 valid MAE 选一个 checkpoint；训练/验证阶段未构造 test，所有 6 个 run 冻结后才按用户明确要求做了一次 post-hoc test 读取（见 2.5）。
- 没有使用 EMA、soup、seed averaging、ensemble、蒸馏、辅助标签、目标分项监督。
- 没有使用 RDKit logP/SA、目标环惩罚公式值、历史模型预测/残差作为输入。
- 输入只含原始图的 label-free 拓扑统计、已有 `global33`，以及已有 train-only masked-SSL 的 `atom_encoder` / `bond_encoder` 权重（正常微调）。
- 没有加载任何监督 DCC-L 权重；没有 junction encoder / junction token / per-wedge token；没有迭代 hidden-state message passing。

### 2.2 训练配方（完全冻结）

AdamW；peak LR 1e-3 → min 1e-5；500-step warmup 后 per-step cosine；weight decay 1e-5；batch 128；600 epochs（79 steps/epoch × 600 = 47,400 updates）；raw-y L1；grad clip 5；dropout 0；每 epoch 全量 valid；严格最小 valid MAE 选一个 checkpoint；无 EMA/soup/seed averaging。

### 2.3 代码与权重 provenance

| 项目 | 值 |
|---|---|
| A_s0 / R_s0 commit | `42c01a3c7011` |
| N_s0 / R_s1 commit | `d2d3743b9560` |
| A_s1 / N_s1 commit | `b9367655c5de` |
| 训练 forward 代码 | 上述三个 commit 的 `lc_models.py` / `lc_data.py` / `lc_train.py` / `lc_common.py` **逐字节相同**（`git diff --stat` 为空）；后续 commit 只增加分析脚本、metadata fingerprint 和 timing 记录 |
| SSL 权重 | `tracks/structure-contribution/results/bjoc_full10k_pretraining_transfer_v1/pretrain_encoder_s0.pt`，sha256 `006fabc11efede091420fc2126ef2ef1c46cb3948ab91648a96843c0352bfa97` |
| SSL 加载模块 | 只有 `atom_encoder`、`bond_encoder`；`junction_path_loaded=false`；可用 keys 里虽有 `junction_encoder`，但**没有加载** |
| 模型 junction 检查 | `has_junction_module=false`；`state_keys_with_junction=[]` |
| train cache content sha256 | `ded004e8a0c03af31413b0f8d0e8223d80c6ba424a2cbd34db42e16c9b85459b` |
| valid cache content sha256 | `f28a0cceed34444c39523e1d301a02350a67087c15b3de725850e45976212aef9` |
| norm fingerprint | `8c3e88137f18ca6841758b5779b57d7362d9dd57c2c29db2067f822cafd1c89f` |
| feature version | `lcrc-v1-atom108-s1joint-s2typedpath` |
| CPU gates | 8/8 PASS，`analysis/cpu_checks.json`（commit `40d2036`；之后仅 metadata/analysis 变化） |

### 2.4 每个 run 的 checkpoint / 时间

| run | arm | seed | commit | params | best epoch | best valid | GPU·h | best checkpoint sha256（前 16） |
|---|---|---:|---|---:|---:|---:|---:|---|
| lcrcA_s0 | A | 0 | `42c01a3c7011` | 237,519 | 476 | 0.076582 | 0.3576 | `06340bcd9e45af03` |
| lcrcA_s1 | A | 1 | `b9367655c5de` | 237,519 | 454 | 0.080504 | 0.4909 | `7225fe11fce0a24d` |
| lcrcN_s0 | N | 0 | `d2d3743b9560` | 269,664 | 481 | 0.081541 | 0.5060 | `541f72408af73212` |
| lcrcN_s1 | N | 1 | `b9367655c5de` | 269,664 | 416 | 0.078608 | 0.4963 | `de1a5f88673037ca` |
| lcrcR_s0 | R | 0 | `42c01a3c7011` | 269,664 | 486 | 0.077591 | 0.5374 | `51cf7941c8314d85` |
| lcrcR_s1 | R | 1 | `d2d3743b9560` | 269,664 | 419 | 0.080474 | 0.5619 | `c9f7baf36071ee7e` |

正式训练合计 **2.9501 GPU·h**；prep/timing 为 CPU 或数秒级 GPU，合计仍 <3.0 GPU·h。未超过 6 GPU·h 目标，更未接近 8 GPU·h 硬上限。

### 2.5 test 访问记录

- 预注册 gate 在**任何** test 访问之前评估：`analysis/seed_summary.json` 记录 `pass=false`、`decision=do_not_read_test`。
- 用户随后明确要求“读一下 test”；对 **A_s0**（最佳 valid 单模型，0.076582）执行一次 `lc_test_reporting.py --run lcrcA_s0 --allow-any-arm`。
- 脚本先写 `analysis/test_freeze.json`（选择、checkpoint sha256、config sha256、重算 valid MAE），再构建 label-free test feature cache，最后只读一次 test labels；`test_report.json` 记录 `read_count: 1`。
- `test_freeze.json` 明确记录 `protocol_compliant=false`、`user_override_post_hoc=true`；这不是 gate 通过，也不是 confirmatory test。读取发生在所有模型/seed/epoch 冻结之后，没有用 test 做任何选择。
- test cache 现在存在（`cache/lc_test.pt`，label-free features）；逐分子预测在 `analysis/test_per_molecule_lcrcA_s0.csv`，NPZ 在 `predictions/test_lcrcA_s0.npz`（本地 gitignored）。
- 历史 test 已暴露，不是新盲测；6 个训练 run 的 `train.json` 仍记录 `official_test_loaded: false`（训练期间确实没有读）。
- 本次执行使用的是加了两处修正的 `lc_test_reporting.py`（`--allow-any-arm`、`L.D.official_labels`）；`test_freeze.json` 的 `git_commit=36972d7` 是读取前冻结的仓库状态，checkpoint/config sha256 已在该文件中冻结，脚本最终版本见本报告 commit。

---

## 3. 主结果

### 3.1 主表（official valid-1000，raw-y MAE）

| run | arm | seed | params | updates | best valid | best ep | W20 | T100 | final valid | train@best | GPU·h |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lcrcA_s0 | A | 0 | 237,519 | 47,400 | **0.076582** | 476 | 0.077333 | 0.077359 | 0.077366 | 0.005268 | 0.358 |
| lcrcA_s1 | A | 1 | 237,519 | 47,400 | 0.080504 | 454 | 0.081191 | 0.081252 | 0.081197 | 0.006855 | 0.491 |
| lcrcN_s0 | N | 0 | 269,664 | 47,400 | 0.081541 | 481 | 0.081948 | 0.082008 | 0.082014 | 0.004858 | 0.506 |
| lcrcN_s1 | N | 1 | 269,664 | 47,400 | 0.078608 | 416 | 0.079217 | 0.079239 | 0.079223 | 0.010238 | 0.496 |
| lcrcR_s0 | R | 0 | 269,664 | 47,400 | 0.077591 | 486 | 0.078056 | 0.078080 | 0.078084 | 0.005133 | 0.537 |
| lcrcR_s1 | R | 1 | 269,664 | 47,400 | 0.080474 | 419 | 0.080795 | 0.080818 | 0.080775 | 0.008065 | 0.562 |

历史 DCC-L anchor（未重训）：valid 四 seed 均值 **0.075990**；per-seed `s0=0.078537, s1=0.073837, s2=0.074677, s3=0.076911`。

W20/T100/final 三条末段曲线与 best 非常接近，说明没有明显的 late divergence；best checkpoint 都在第 416–486 epoch，训练充分。

### 3.2 两 seed 均值与逐 seed 差

| arm | seeds | best valid per seed | mean |
|---|---:|---|---:|
| A | 0,1 | 0.076582, 0.080504 | **0.078543** |
| R | 0,1 | 0.077591, 0.080474 | 0.079032 |
| N | 0,1 | 0.081541, 0.078608 | 0.080075 |

| seed | A | R | N | DCC-L | R−N | R−A | N−A | A−DCC-L | R−DCC-L |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.076582 | 0.077591 | 0.081541 | 0.078537 | **−0.003950** | +0.001009 | +0.004959 | −0.001956 | −0.000946 |
| 1 | 0.080504 | 0.080474 | 0.078608 | 0.073837 | **+0.001866** | −0.000030 | −0.001895 | +0.006667 | +0.006637 |

**结论：**

- `A` 的两 seed 均值最好，但只比 R 好 0.000489，比 N 好 0.001532；都远小于 seed 间波动（A 的 seed 差 0.003922）。
- `R−N` 在 seed0 为 −0.003950，在 seed1 为 +0.001866，**符号翻转**，不能称为稳定收益。
- 逐 seed 对 DCC-L 的比较是描述性的：A_s0/R_s0 名义上优于 DCC-L_s0，但 A_s1/R_s1 明显差于 DCC-L_s1；两 seed 均值均差于 DCC-L 四 seed 均值。

### 3.3 Paired per-molecule 对比（bootstrap 只重采样 valid 分子，不重采样训练 seed）

| first | second | seed | MAE first | MAE second | Δ | 95% CI | first better |
|---|---|---:|---:|---:|---:|---|---:|
| R_s0 | N_s0 | 0 | 0.077591 | 0.081541 | −0.003950 | [−0.009299, +0.000882] | 0.487 |
| R_s1 | N_s1 | 1 | 0.080474 | 0.078608 | +0.001866 | [−0.003039, +0.007221] | 0.502 |
| R_s0 | A_s0 | 0 | 0.077591 | 0.076582 | +0.001009 | [−0.002623, +0.004884] | 0.476 |
| R_s1 | A_s1 | 1 | 0.080474 | 0.080504 | −0.000030 | [−0.004216, +0.004138] | 0.492 |
| N_s0 | A_s0 | 0 | 0.081541 | 0.076582 | +0.004959 | [+0.000224, +0.010287] | 0.491 |
| N_s1 | A_s1 | 1 | 0.078608 | 0.080504 | −0.001895 | [−0.008308, +0.003417] | 0.497 |
| A_s0 | DCC-L_s0 | 0 | 0.076582 | 0.078537 | −0.001956 | [−0.008850, +0.004355] | 0.508 |
| A_s1 | DCC-L_s1 | 1 | 0.080504 | 0.073837 | +0.006667 | [−0.003205, +0.019993] | 0.532 |
| R_s0 | DCC-L_s0 | 0 | 0.077591 | 0.078537 | −0.000946 | [−0.008910, +0.006125] | 0.498 |
| R_s1 | DCC-L_s1 | 1 | 0.080474 | 0.073837 | +0.006637 | [−0.003197, +0.018709] | 0.525 |

CI 均包含 0；paired bootstrap 只描述 valid 分子抽样，不能替代跨 seed 复现。**不能把 A_s0 或 R_s0 的单点优势当成稳定结论。**

---

## 4. 分组结果

| group | n | A_s0 | A_s1 | N_s0 | N_s1 | R_s0 | R_s1 | DCC-L_s0 | DCC-L_s1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 1000 | 0.076582 | 0.080504 | 0.081541 | 0.078608 | 0.077591 | 0.080474 | 0.078537 | 0.073837 |
| ordinary (`y>-3`) | 935 | 0.063545 | **0.062916** | 0.066942 | 0.063712 | 0.064805 | 0.064181 | 0.065537 | 0.063085 |
| tail (`y≤-3`) | 65 | 0.264113 | 0.333502 | 0.291543 | 0.292886 | **0.261503** | 0.314848 | 0.265538 | 0.228501 |
| row172 | 1 | 6.575232 | 10.459000 | 7.242666 | 8.202539 | 6.488779 | 9.752165 | 6.099347 | 5.418310 |
| r2-unseen | 251 | 0.145477 | 0.148758 | 0.159810 | 0.150767 | 0.149342 | 0.152520 | 0.150957 | 0.135445 |
| r2-unseen ordinary | 225 | 0.130915 | 0.133092 | 0.141189 | 0.134875 | 0.135708 | 0.138540 | 0.130773 | 0.121821 |
| ring-rich (`cycle_rank≥4`) | 213 | 0.100707 | 0.099069 | 0.117450 | **0.096037** | 0.096731 | 0.097593 | 0.111694 | 0.099212 |
| ring-rich unseen | 47 | 0.259189 | 0.281068 | 0.335394 | 0.262203 | 0.268687 | 0.269037 | 0.317617 | 0.264888 |
| worst-10 fixed | 10 | 1.468935 | 1.825165 | 1.603475 | 1.553974 | **1.401975** | 1.815320 | 1.656936 | 1.280082 |
| all − row172 | 999 | 0.070076 | 0.070115 | 0.074373 | 0.070476 | 0.071173 | 0.070793 | 0.072510 | 0.068487 |
| all − worst10 | 990 | 0.062517 | 0.062881 | 0.066168 | 0.063706 | 0.064213 | 0.062950 | 0.062594 | 0.061653 |
| tail − row172 | 64 | 0.165502 | 0.175291 | 0.182931 | 0.169298 | **0.164202** | 0.167390 | 0.174385 | 0.147410 |

**读法：**

- 新主体在 **ordinary** 上并不弱：A_s1 的 ordinary 0.062916 甚至是全表最好；A_s0 0.063545、R_s1 0.064181 也与 DCC-L 同档。
- 差距主要在 **tail / row172 / worst-10**。A_s1 的 tail 0.3335、row172 10.459 把整体从 0.076 量级拉到 0.0805。R_s0 的 tail 0.2615、row172 6.489 是 R 里最好的，但 R_s1 又回到 0.3148/9.752。
- `all − worst10` 对 A_s0 是 0.062517；也就是说，**即使把固定 worst-10 全部做到零误差，整体也只有 0.062517×0.990 = 0.061892；其余 990 个分子的 MAE 仍是 0.062517，仍 >0.060**。`<0.060` 不能只靠修几个极端点。

---

## 5. 机制诊断：关系通道到底有没有被使用？

### 5.1 真实关系输入不是退化的

CPU G6（512 个 train 分子）：

| 指标 | 值 |
|---|---:|
| 平均合格原子对数 / 分子 | 207.5（范围 45–532） |
| 6 类非空率 | [1.000, 0.945, 1.000, 0.324, 1.000, 0.021] |
| 6 类平均计数 | [26.3, 7.8, 57.2, 2.2, 113.8, 0.15] |
| ≥2 个非空类别的分子比例 | 100% |
| R 与 N 内容不同的分子比例 | 100%（mean L2 0.454） |
| 从 A 池化特征 ridge 重建 R 内容，common cats R² | 0.702（约 30% 方差线性不可重建） |
| 同一 ridge 的 permutation control | −1.44 |

所以：**输入侧是真实的、非退化的，也不是完全冗余**。R 的类别内容确实随分子变化；在最佳 checkpoint 上，R 的类别均值 between-molecule std 是 within-molecule SEM 的 2.7–3.7 倍（N 的全局均值只有 0.21–0.22 倍，主要被 pair 采样噪声支配）。

### 5.2 但是读头坍缩为常数

| run | Δ_rel mean | Δ_rel 分子间极差 | rel.head norm | head 置零 ΔMAE | 固定权重 R↔N 内容切换 ΔMAE | fresh ridge (hidden) valid | fresh ridge (raw M) valid |
|---|---:|---:|---:|---:|---:|---:|---:|
| R_s0 | 0.000470 | 2.0×10⁻⁵ | 4.25×10⁻⁴ | −1.39×10⁻⁵ | **9×10⁻⁹** | 0.077873 | 0.078090 |
| R_s1 | 0.027211 | 1.6×10⁻⁵ | 5.95×10⁻⁴ | +5.95×10⁻³ | **1.8×10⁻⁹** | 0.080524 | 0.080377 |
| N_s0 | 0.004193 | 1.83×10⁻⁴ | 2.23×10⁻³ | +3.57×10⁻⁴ | 1.5×10⁻⁶ | 0.081630 | 0.081777 |
| N_s1 | 0.018663 | 4.79×10⁻⁴ | 7.21×10⁻³ | +1.67×10⁻³ | 4.15×10⁻⁴ | 0.078482 | 0.078854 |

（fresh ridge 是后验诊断：在 8,000 个 train 分子上拟合线性头，2,000 个 held-out train + valid 评估；不是可部署模型，也不参与选型。）

要点：

- **R 的 `Δ_rel` 在分子间几乎是常数**：极差 ~2×10⁻⁵，而 valid MAE 是 0.077–0.080。读头学到的是一条全局偏置，不是 per-molecule 的结构修正。
- **固定权重下把真实归属换成 mean-field 归属，R 的 valid MAE 变化 ≤1×10⁻⁸**。这是最强的反证：冻结模型对“真实对应 vs 均值场”完全不敏感。若真实对应被使用，这个切换不可能只有 10⁻⁹ 量级。
- **head 置零**对 R_s0 甚至略微改善（−1.4×10⁻⁵），对 R_s1 则是去掉一个有用的常数偏置（+5.95×10⁻³）。两种情况下它都只是 calibration，不是结构信息。
- **fresh ridge 也救不回来**：在 hidden 或 raw M 上拟合线性头，最好也只有 ~1×10⁻⁴ 的改善，和 permutation control 同量级；R_s0 的 ridge 甚至比 base 更差。关系内容没有可线性读出的 residual signal。
- `cat_emb_norm` 仍有 6–7（嵌入没有坍缩），说明不是 embedding 死亡；死亡的是 **content → correction** 的映射：共享 MLP + zero-init head 在 L1 残差上收敛到了常数。

### 5.3 为什么关系头会坍缩？

可能解释（按证据强度排序）：

1. **mean-over-pairs 把 pair-level 信号平均掉了。** 每个类别有 26–114 个 pair（类别 3/5 更少），类别均值主要反映“类别偏移”（距离桶/环块），而不是分子特异的残差；30% 非冗余方差存在，但幅度小、方向弱。
2. **zero-init 线性头的梯度是弱信号。** 初始 `head=0`，关系分支对 shared trunk 的梯度也接近 0；L1 的 per-molecule 梯度噪声大，头更容易学到一个全局常数（降低平均 |error|）而不是 content 映射。head norm 最终只有 4×10⁻⁴–7×10⁻³。
3. **A 主体已经很强，残差小而噪声大。** train eval MAE 只有 0.005–0.0102；在这么小的残差上，关系内容的边际信号很容易被 batch 噪声和 tail 的极端样本淹没。
4. **类别不平衡。** cat5 只有 2.1% 分子非空，cat3 32.4%，cat1 94.5%；稀有类别得不到足够梯度，共享 MLP 倾向忽略它们。
5. **seed 噪声 / 优化轨迹。** R 和 N 只差一个 content 聚合方式，但 zero-init 分支会改变 shared trunk 的早期梯度；在 600 epoch、强 cosine 退火下，这种微小差异可以放大成 0.004 的 valid 差异。seed0 的 R>N 与 seed1 的 R<N 正是这种不稳定性的表现。


### 5.4 Post-hoc test 读取（用户要求；非协议 gate）

**选择：** A_s0，best valid = 0.076582，checkpoint sha256 `06340bcd9e45af03e887e706031b6ea14c719cac2b68ba716ce3d686bb21fdad`。预注册 gate 失败（R−N 方向翻转；R mean 0.079032 > 0.079），所以这次读取在 `test_freeze.json` 中记录为 `protocol_compliant=false`、`user_override_post_hoc=true`。

| 指标 | A_s0 | DCC-L s0 | DCC-L s1 | DCC-L s2 | DCC-L s3 | DCC-L 4-seed mean |
|---|---:|---:|---:|---:|---:|---:|
| test MAE | **0.066782** | 0.066267 | 0.067688 | 0.067538 | 0.067785 | 0.067319 |
| test RMSE | 0.139026 | — | — | — | — | — |
| p90 AE | 0.137302 | — | — | — | — | — |
| max AE | 2.008445 | — | — | — | — | — |
| ordinary (`y>-3`) | 0.063410 (n=923) | 0.060392 | 0.062325 | 0.061729 | 0.063093 | — |
| tail (`y≤-3`) | 0.107208 (n=77) | 0.136686 | 0.131978 | 0.137169 | 0.124026 | — |

**Paired test bootstrap（10k，逐分子；机器可读表见 `analysis/test_compare.{md,json}`）：**

| 对比 | Δ MAE (A_s0 − DCC-L) | 95% CI | A better rate |
|---|---:|---|---:|
| vs DCC-L s0 | +0.000516 | [−0.007302, +0.007178] | 0.502 |
| vs DCC-L s1 | −0.000906 | [−0.007180, +0.005060] | 0.513 |
| vs DCC-L s2 | −0.000756 | [−0.006870, +0.005134] | 0.500 |
| vs DCC-L s3 | −0.001002 | [−0.006396, +0.004330] | 0.503 |

**读法：**

- A_s0 的 test MAE **落在 DCC-L 四个 seed 的范围内**：比 s0 差 +0.000515，比 s1/s2/s3 好 0.0008–0.0010；所有 paired CI 都跨 0。单 seed、post-hoc，**不能声称超过 DCC-L**。
- 与 valid 的分组模式相反：test 上 A_s0 **ordinary 更差**（0.063410 vs DCC-L 0.0604–0.0631），**tail 更好**（0.107208 vs DCC-L 0.1240–0.1367）。test 只有 77 个 tail，且 test 的极端误差远小于 valid（max AE 2.008 vs valid row172 6.575）；valid 的 row172 病态点在 test 中不存在。因此 test 的“tail 更好”不能直接外推到 valid 的 tail 问题。
- test 数字与 valid 数字不可直接比较：A_s0 valid 0.076582、test 0.066782；DCC-L 也是 valid 0.078537、test 0.066267。test 整体更容易，且历史 test 已暴露。
- **决策不变：STOP。** 这次读取是用户要求的描述性检查，不是 gate 通过，也不能用来改模型/seed/epoch；R 机制仍然为空，A 仍然只有 seed0 的 test 证据。

---

## 6. 五个问题

### Q1. 不带 junction token 的局部表示，能否与 DCC-L 竞争？

**观察结果：**

- A_s0 = **0.076582**，优于 DCC-L seed0 0.078537（Δ = −0.001956，paired CI [−0.008850, +0.004355]，win rate 0.508）。
- A_s1 = 0.080504，差于 DCC-L seed1 0.073837（Δ = +0.006667）。
- A 两 seed 均值 **0.078543**，比 DCC-L 四 seed 均值 0.075990 差 **0.002553**。
- A 的 ordinary MAE 0.0629–0.0635，和 DCC-L 同档；差距集中在 tail/row172。
- A 的局部贡献占预测尺度 71–78%（`local_share_median`），h 的 participation ratio 4.75–4.99，没有表示坍缩；A 的 `Σf` 和 `ρ` 都在工作。
- 用户要求下的 post-hoc test：A_s0 = 0.066782，落在 DCC-L s0 0.066267 与 4-seed mean 0.067319 之间，所有 paired CI 跨 0；单 seed、非 gate，仍不能作为稳定竞争优势。

**可能解释：**

- A 的局部主体确实能承载 ZINC 的大部分 ordinary 化学；不需要 junction token 也能到 DCC-L 的 ordinary 水平。
- 但 A 对 tail/极端分子的稳定性不足，seed1 的 tail 0.3335、row172 10.459 把均值拉坏；两 seed 均值因此没有超过 DCC-L。
- A_s0 的“优于 DCC-L seed0”是单点、描述性的；CI 含 0，不能当成稳定优势。

**下一步决定：**

- 本轮 test 已在用户要求下做了一次 post-hoc 读取；**不再重复读 test**，也不以 A_s0 单点声称“达到/超过 DCC-L”。
- 如果未来要继续 A 这条线，只值得做 **3–4 seed 的重复确认**（同一配方、同一条停止规则），并预注册 tail/row172 分层报告；在均值稳定 ≤0.079 或明确优于 DCC-L 之前，不把 test 当作 confirmatory。
- 本轮不再为 A 启动补丁实验。

### Q2. R 是否优于 N？真实结构对应是否带来稳定收益？

**观察结果：**

- seed0：R 0.077591 vs N 0.081541 → **R−N = −0.003950**（R 好）。
- seed1：R 0.080474 vs N 0.078608 → **R−N = +0.001866**（R 差）。
- 均值 R−N = **−0.001042**；方向翻转，不是稳定收益。
- 即使 seed0 的 R−N，paired bootstrap CI 也包含 0（[−0.009299, +0.000882]）。
- R−A：seed0 +0.001009，seed1 −0.000030；均值 +0.000489（R 略差）。
- N−A：seed0 +0.004959，seed1 −0.001895；均值 +0.001532（N 略差）。

**可能解释：**

- R/N 的差异主要来自 zero-init 关系分支对 shared trunk 的早期扰动和常数偏置，而不是 content 的真实归属；因此对 seed 敏感、方向不稳定。
- seed1 的 N 是六个新 run 中该 seed 最好的（0.078608），说明“mean-field + 常数校准”本身就能在某个 seed 上表现很好；R 没有稳定优势。
- 两 seed 的 R−N 均值虽然数值上 ≤ −0.001，但“两个 seed 一致”不成立，不能作为机制证据。即使把 R valid mean 0.079032 四舍五入为 0.079，`consistent R<N` 仍为 false；test 仍不读。

**下一步决定：**

- **不再重复读 test**；不把 R 作为“真实关系组合有效”的正结果。
- 本轮停止；不启动 N_s2 / R_s2 或任何 patch。

### Q3. 若 R 优于 N，是真实对应还是容量/冗余？若否，失败在哪里？

**观察结果：**

- R 的输入是真实的、非退化的：207.5 对/分子、6 类覆盖、100% 分子 R≠N 内容、between-molecule 方差是 SEM 的 2.7–3.7 倍；G6 还显示 common categories 有约 30% 方差不能由 A 池化特征线性重建。
- 但训练后的 R **没有使用**这个输入：
  - `Δ_rel` 分子间极差 ~2×10⁻⁵（R_s0/R_s1），几乎是常数。
  - 固定权重把真实归属换成 mean-field，valid MAE 变化 **≤1×10⁻⁸**（R 两个 seed）。
  - `rel.head` norm 只有 4×10⁻⁴–6×10⁻⁴；head 置零只改变常数偏置。
  - fresh ridge（hidden 或 raw M）没有可用的 residual signal；permutation control 同量级。
- 因此 seed0 的 R>N **不是**真实对应的证据；它是常数偏置/优化轨迹差异。R 的额外 32k 参数也没有被有效利用——容量不是解释，坍缩才是。

**可能解释：**

- 关系内容的主要变化是“类别偏移”，不是“分子特异残差”；mean-over-pairs 把这些偏移压成了近似常数。
- zero-init head + L1 + 强共享 trunk 使 content→correction 的映射欠拟合；头用常数偏置降低平均误差，比学 content 更容易。
- 30% 非冗余方差是“相对于 A 池化特征的线性冗余”，不等于“对 residual 有预测力”；这两件事被本轮结果明确区分开了。

**下一步决定：**

- **停止**。不通过“去掉 weight decay”“放大 head”“换初始化”“加 category attention”等 patch 来抢救；这些都属于事后补丁，违反本轮预注册的停止规则。
- 若未来开新轮，必须预注册新的**关系读出假设**（例如分布分位数/直方图、learned query pooling、pair-level auxiliary supervision），并用 3+ seed 验证；不能把本轮 R 的 seed0 单点当作依据。
- 本轮结论：**真实对应存在，但没有被读出使用；R−N 的性能差异不能归因于 correspondence。**

### Q4. 最佳新单模型离 DCC-L 多远？有没有支持继续追求 test MAE <0.060 的信号？

**观察结果：**

- 最佳新单模型：**A_s0 = 0.076582**。
  - vs DCC-L 四 seed 均值 0.075990：差 **+0.000592**（约 0.78%）。
  - vs DCC-L seed0 0.078537：好 **−0.001956**（单点、CI 含 0）。
  - A 两 seed 均值 0.078543：差 **+0.002553**。
  - R 两 seed 均值 0.079032：差 **+0.003042**。
- 新主体在 ordinary 上已接近/达到 DCC-L：A_s1 ordinary 0.062916，A_s0 0.063545；DCC-L 四 seed ordinary 0.0631–0.0655。
- 差距集中在 tail：A_s0 tail 0.2641、A_s1 tail 0.3335；DCC-L s1/s2 tail 0.2285/0.2162。
- `<0.060` 的距离：best 0.076582 需要再降 **0.016582**（相对 21.7%）。即使把固定 worst-10 全部做到 0，整体也只有 **0.061892**（其余 990 个分子 MAE 0.062517），仍 >0.060；因此 `<0.060` 需要 ordinary 也显著改善，而不是只修几个极端点。
- 没有观察到任何“接近 0.060”的曲线/分组信号：tail 仍在 0.26–0.33，row172 在 6.5–10.5，r2-unseen 在 0.145–0.153。
- post-hoc test：A_s0 = 0.066782，比 DCC-L s0 0.066267 差 +0.000515，比 DCC-L 4-seed mean 0.067319 好 −0.000537；单 seed、CI 跨 0，不构成优势。test 的 tail 更容易（max AE 2.008，valid row172 6.575），不能据此说 `<0.060` 更近。

**可能解释：**

- 当前 A 的主体已经到 DCC-L 的 ordinary 平台；剩余误差是 tail/extreme 的长尾，而不是平均化学的表示不足。
- 关系组合没有针对 tail；它的常数偏置对 tail 没有实质帮助。
- `<0.060` 需要新的 tail/extreme 机制（或不同的数据/监督），不是本轮 LCRC 结构的自然延伸。

**下一步决定：**

- test 已在用户要求下读了一次；**不再重复读**；不声称 `<0.060` 可期。
- 本轮停止。若未来追求 `<0.060`，应把它作为一个**新的、预注册的 tail/extreme 泛化问题**，而不是继续在 LCRC 上加关系模块。
- 对 DCC-L 的“接近”只作为描述性 anchor；DCC-L 未在本轮重训，不能做严格同配方对照。

### Q5. 失败模式诊断（观察结果 / 可能解释 / 下一步决定）

**观察结果：**

1. 关系头坍缩：R 的 `Δ_rel` 几乎是常数（极差 ~2×10⁻⁵），head norm 4–6×10⁻⁴；固定权重切换真实/mean-field 归属，MAE 变化 ≤1×10⁻⁸。
2. R−N 方向翻转：seed0 −0.003950，seed1 +0.001866；不是稳定收益。
3. R−A 几乎为零：seed0 +0.001009，seed1 −0.000030；关系分支没有给 A 带来稳定增益。
4. A 的 seed 波动：A_s0 0.076582 vs A_s1 0.080504；差异主要来自 tail/row172（A_s1 tail 0.3335 vs A_s0 0.2641；row172 10.459 vs 6.575）。
5. 最佳模型仍远离 `<0.060`：best 0.076582；`all − worst10` 仍 0.062517（整体零 worst10 为 0.061892）；tail 0.26–0.33。

**可能解释：**

- **读出瓶颈：** mean-over-pairs + zero-init linear head 在 L1 残差上收敛到常数偏置；content 有方差，但没有被转换成 per-molecule correction。
- **信号瓶颈：** 关系内容的主要方差是类别偏移，不是 residual 方向；30% 非冗余方差 ≠ 可预测残差。
- **优化瓶颈：** 关系分支对 shared trunk 的早期梯度很弱；seed 间的微小差异会被 600 epoch cosine 放大，造成 R/N 方向翻转。
- **度量瓶颈：** 65 个 tail + 10 个固定 worst 主导 aggregate；任何不改善 tail 的结构改动都只能在 0.078–0.081 之间摆动。

**下一步决定：**

- **本轮停止，不启动更多补丁实验；不再重复读 test。**
- 保留 A 作为“无 junction 的强 ordinary 基线”的证据，但不把它升级为“超过 DCC-L”的结论；若继续，只做预注册的 3–4 seed A 复现。
- 关系方向若未来重启，必须是新轮：预注册 distributional/attention relation summary、pair-level auxiliary supervision、tail-stratified 评估和 3+ seed；本轮不通过改 head/初始化/正则来“救”R。
- 归档 `REPORT.md`、`analysis/`、`runs/` 和 `analysis/cpu_checks.json`；test 保持未读。

---

## 7. 预算与产物

### 7.1 预算

| 项目 | GPU·h |
|---|---:|
| A_s0 | 0.3576 |
| A_s1 | 0.4909 |
| N_s0 | 0.5060 |
| N_s1 | 0.4963 |
| R_s0 | 0.5374 |
| R_s1 | 0.5619 |
| **合计** | **2.9501** |
| 目标 / 硬上限 | 6 / 8 |

最大并发 2 张 A100；没有超过预算。prep 是 CPU；timing run 是数秒级。

### 7.2 产物

| 文件 | 内容 |
|---|---|
| `REPORT.md` | 本报告 |
| `PROTOCOL.md` | 预注册协议（训练前冻结；状态已更新） |
| `DESIGN.md` | 设计与风险 |
| `configs/lcrc_v1.json` | 冻结配置 |
| `analysis/cpu_checks.json` | 8/8 CPU gates |
| `analysis/main_table.{md,json}` | 主表 |
| `analysis/contrasts.{md,json}` | 逐分子 paired 对比 + bootstrap |
| `analysis/groups.{md,json}` | 固定分组 |
| `analysis/seed_summary.{md,json}` | 两 seed 汇总 + test gate |
| `analysis/mechanism.json` | 尺度、head-zero、R/N mode swap、类别覆盖、h participation |
| `analysis/relation_probe.json` | 后验 relation-content ridge / variance 诊断 |
| `analysis/per_molecule_valid.csv` | 逐分子 valid 预测与 AE |
| `analysis/test_freeze.json` | post-hoc test 前的选择/checkpoint/config 冻结记录（`protocol_compliant=false`） |
| `analysis/test_report.json` | A_s0 post-hoc test 指标（read_count=1） |
| `analysis/test_per_molecule_lcrcA_s0.csv` | A_s0 逐分子 test 预测与 AE |
| `analysis/test_compare.{md,json}` | A_s0 与历史 DCC-L s0–s3 test 的逐分子 paired 比较（描述性） |
| `analysis/summary.json` | 机器可读汇总 |
| `analysis/norm.json` | train-only 标准化 |
| `analysis/timing.json` | 计时 |
| `runs/*/best.pt`, `runs/*/train.json`, `runs/*/curve.csv` | 每个 run 的 checkpoint、provenance、曲线 |

### 7.3 最终声明

- 预注册 gate 未通过；用户随后明确要求对 A_s0 做了一次 **post-hoc** test 读取：0.066782（`analysis/test_report.json`）。`test_freeze.json` 记录 `protocol_compliant=false`、`user_override_post_hoc=true`；该数字是描述性记录，不是 confirmatory，也没有用于任何模型/seed/epoch 选择。
- 历史 DCC-L valid 四 seed 均值 0.075990 和 per-seed 值来自既有 `zinc_performance_search_v1`，**本轮没有重训 DCC-L**；所有与 DCC-L 的逐 seed 对比都是描述性的，不是同配方、同 seed 的严格对照。
- 没有 ensemble、distillation、extra labels、EMA/soup/seed averaging；没有 RDKit 目标描述符；没有 junction token；没有加载监督 DCC-L 权重；没有迭代 hidden-state message passing。
- 结论仍是**混合/偏负**：A 的局部主体在 ordinary 上是有竞争力的，但没有稳定超过 DCC-L；R 的“真实对应”输入存在且非退化，但读头没有使用它；post-hoc test 也没有给出稳定优势；本轮没有 `<0.060` 的信号，因此停止。
