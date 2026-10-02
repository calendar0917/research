# ZINC + 字典路线：当前 Full 的机制与性能瓶颈审计（`zinc_full_bottleneck_audit_v1`）

Date: 2026-10-02 · 只读机制审计 + 一个冻结监督探针 · **官方 test 从未实例化**
对象：`E2E-DictEnv-Scale-v1` 的父 Full soup（`SCALE-FULL-seed0_soup_state.pt`，
sha256 `17f5fcc3cdf32462657006ec89ebf2ecc72ab6fbc954b926af248aca374574eb`）。
协议：`tracks/ksvd/protocols/zinc-full-bottleneck-audit-v1.yaml`。
run：`tracks/ksvd/runs/2026/10/02/20261002-175703-469df2f8`（probe 阶段）。
证据目录：`tracks/ksvd/results/zinc_full_bottleneck_audit_v1/`。

---

## 1. 审计摘要（一页）

**最可能的问题：不是某个新模块没加，而是 (a) 模型已进入记忆区、valid 尾部泛化差；
(b) 结构—原子语义的“节点侧绑定”在当前 Full 中已经彻底死亡（恒为 0），所以研究主张里
“每个 occurrence 上结构 × 原子语义”的机制在实际预测中并不存在。**

已确认事实（可复放，见 §3–§5）：

1. **节点分支完全死亡。** `W_A_S`、`W_A_C`、`node_encoder.0.weight` 在当前 soup 里都是
   float32 denormal（absmax ≈ 7.05e-38，norm = 0.0）。前向中 `node_slot` 的
   `zero_frac = 1.0`，`node_out` 逐行 std = 0（row0 == row1）。因此 446 维 fusion 输入里
   有 144 维（节点槽）是**常量**，被 `fusion.0.bias` 吸收。结构字典码只通过**边侧绑定**
   与 `Sem108` 直接接口进入模型。
2. **任务字典不是稀疏字典。** 288 维任务码平均 250.9 个非零（87.1%，p50=255），
   `λ1=0.05, λ2=0.01, 16 步 ISTA` 下软阈值几乎不激活；低维重构相对误差 0.0588。
   它承重（zero-code ΔMAE≈1.44），但更像一个按行 RMS 归一化的稠密 144→288→144 变换，
   而不是“稀疏共享字典”。这与既有 CODE 负结果一致，但 CODE 只关闭了“码矩统计量”。
3. **误差由极负尾部和单个异常行主导。** calibrated valid MAE = 0.115066；
   id172（`y=-20.3405`，`p_raw=-1.3886`）单独贡献 0.018918（占绝对误差和 16.4%），
   去掉它 MAE=0.096244；top1%/5%/10% 分别占误差和的 25.3%/39.8%/50.6%；
   train-fit 最低十分位（y<-2.55）valid MAE=0.3812，占误差和 30.8%。
4. **尾部是泛化失败，不是表示装不下。** train 上 y<-3 的 716 行拟合良好
   （signed err -0.015，回归斜率 dp/dy=1.03），valid 上 y<-3 的 65 行严重欠拟合
   （signed err -0.381，斜率 0.287）。模型在 train 尾部能表示，在 valid 尾部不能迁移。
5. **两个条件探针均为负。** 在冻结 `[1,H2]` 凸头上加
   (P1) 446 维 pre-fusion 接口矩、或 (P2) 结构码 × 原子类型联合矩，valid 改善分别为
   **-0.000272 / -0.000131**，远低于 0.003 门槛；同宽随机投影对照仅 +4.5e-7。
   所以这两个位置**没有线性可取的剩余信号**，也没有“只是宽度/正则化”的假象。

**证据强度与边界：** 上述 1–4 是当前 checkpoint 上可复放的事实（§3–§5）；
5 是“固定 `[1,H2]` 凸头 + 两个具体块”的负结果，只关闭这两个组合，不关闭整个表示/读出方向。
探针本身有数值阳性对照（`[1,H2]` 复现 CODE 基线 0.115024），但 H2 有效秩仅 ~2（78% 单元死），
因此它**探不出非线性信息**——负结果不能推出“上游已充分”。

**下一步：不值得购买新的“加模块 / 扩参”训练。** 最有价值的下一步是补一条针对尾部的
可恢复性/覆盖证据（§9），而不是再 warm/从零训练一个局部模块。

---

## 2. 阶段 A — 当前对象与协议核对表

| 项 | 值 / 证据 |
|---|---|
| repo revision | `924515b8f4931d019ed481f5f9382eb90e1bf28a`（dirty：untracked jointbond 结果、trajectory review、install-arch.sh） |
| study / protocol | 焦点 study `zinc-context-gap`（`research context`）；本审计自登记 protocol `zinc-full-bottleneck-audit-v1`，与既有 claim/decision 不冲突 |
| 记录是否已被更新 | untracked 的 `e2e_dictenv_jointbond_v1`（09-30）与 `..._decay_diagnostic_v1`（09-30）是 **Sem108 父模型**的旁支，早于 scale_v1 Full（10-02）；最新主线对象仍是本文的 Full soup。jointbond 报告本身确认 Sem108 父的 `W_A_S/W_A_C` 已塌缩，本审计在当前 Full 上复核 |
| 当前 phase | `zinc-upstream-portfolio-v1-stop-no-purchased-candidate`（`research context`） |
| checkpoint | `results/e2e_dictenv_scale_v1/checkpoints/SCALE-FULL-seed0_soup_state.pt`，文件 sha256 `17f5fcc3…574eb`（已核对） |
| checkpoint 类型 | **参数 soup**：320 轮中 official-valid 最低的 5 个 epoch `[268,276,289,299,317]` 的参数平均（`soup.json`） |
| 训练拟合量 | 结构字典 `sdb32`（`results/sdb_v0/dictionary.pt`，sha `925d573a…`）；公共子空间 q1（`results/e2e_dictenv_common_subspace_dictionary_v1/common_subspace.json`）；patch/anchor/topology 标准化只在 official train 拟合 |
| 数据 | PyG ZINC `subset=True` official train 10000 / valid 1000；split fingerprint `58c69506…f28a`；test 未实例化 |
| 前向 replay | 用 `sc.build_scale_model(..., FULL, scale_seed=0)` + `cm.C6_MASK`；raw valid MAE = **0.1191540920053958**（与发布值逐位一致）；eval-mode train MAE = 0.0453967 |
| 残差/标定符号 | 定义 `r = y − pred`；train 上 `b = median(r) = −0.03428781`；输出 `pred + b`。calibrated train/valid = 0.0329501 / **0.11506585458567133**（与 bias diagnostic 一致） |
| 校准身份 | 标定是**推理包装**（未写回 checkpoint）；每个模型各自用自己 train 中位残差，绝不用 valid |
| dtype/device/随机性 | float32、CPU 8 threads、eval()、无 dropout 生效（C6 eval 路径无随机）；soup 加载后无 RNG |
| warm-80 control | 记录证据（未由本审计重放）：`pilot_control.json` raw 0.11441372、calibrated 0.11439933、bias −0.003617、members `[76..80]`、soup sha `ad93338e…`。与 trajectory review 一致 |
| 轨迹复核 | `notes/zinc_upstream_portfolio_v1_trajectory_review.md`（untracked）与 `results/e2e_dictenv_upstream_portfolio_v1/full_trajectory.png`（存在）已读/已核对 |

**选模效应（区分训练改善 / 偏置吸收 / 选模平均）：**
父 soup 成员 valid 均值 0.124375，soup 0.119154 → 平均效应 0.005221；
固定尾窗（316–320）成员均值 0.130574，top-5-by-valid 比它好 0.006198。
即 0.119154 是**在报告用的同一 1000 行上选出来的**，乐观偏差上界约 0.006 量级；
control 用固定末 5 轮，是更公平的协议。matched calibrated gain（父 0.115066 vs control
0.114399）= **+0.000667**，与 training-protocol-audit 的 +0.000610 同量级。

**阶段 A 判断：** 没有阻断后续审计的 replay/协议错误；现有证据继续支持
“简单延长同配置训练是低优先级”。但**必须**把 raw/calibrated 并报、把
“top-5-by-valid soup”与“固定窗”区分开。

---

## 3. 阶段 B — 实际机制链路与信息保留

### 3.1 链路图（实际代码）

```
raw graph
  └─ patch_cont φ65 (train-fit 标准化; FEC-S0)
       └─ CSSD 公共坐标:  c = φ·U (U:65×1), r = φ − c Uᵀ
            └─ tied-IHT(Dbar_perp, r; s=8, 10 步) → α_res[32]
                 coord z = [c/rms ; α_res] ∈ R^33          (每个 occurrence)
       ┌────────────────────────────┬─────────────────────────────┐
  节点侧绑定(死亡)               边侧绑定(存活)              直接语义接口
  u=(z@W_A_S)(q_atom@W_A_C)/√96  g=[c_u+c_v;|c_u−c_v|;c_u*c_v]  [Sem108 ; size2]
  按 (root,shell=3) index_add     ue=(g@W_E_S)(q_bond@W_E_C)/√48     = 110
  node_encoder 96→64→48          按 (root,shellpair=6) index_add
        = 常量(144)               edge_encoder 48→48→32 (=192)
       └──────────────┬──────────────┴──────────────┘
                fused = [110 ; 144 ; 192] = 446
                     └─ fusion: 446→342→144 (SiLU) → h[144]
                          └─ 任务字典 bridge (行 RMS 归一化 + 16 步 ISTA)
                               E = scale·(α @ V_L) ∈ R^144
       ┌──────────────────────────┴───────────────────────────┐
   unary pool_moments(E)                              pair 路径
   [ΣE ; ΣE² ; log1p(count)] = 289        u=E@W_P[144→48]; relation=MLP(15→96→48)
                                          gate=1+tanh(distance_gate[bucket])
                                          pair_input=[l+r;|l−r|;(l*r)*gate;rel]∈R^192
                                          pair_value=MLP(192→192→48)
                                          按 distance bucket pool → 5×97 = 485
       └──────────────────────────┬───────────────────────────┘
             R = [unary 289 ; pair 485 ; global 32 ; topology 8] = 814
                  └─ reader 814→39→39→1 → prediction
```

**没有 pair→node/patch 写回；没有 message passing；没有图索引；16 步 ISTA 只在单个
occurrence 内做。** “无消息传递”在此实现中的实际含义是：局部环境 E 一旦算出，
pair 之间只做一次性的对称聚合（sum/|diff|/product），不再回写节点或局部环境。

### 3.2 阶段/参数表（Full 408,651）

| 阶段 | 形状 | 参数 | 来源 | 当前状态 |
|---|---|---|---|---|
| 结构字典 D | 65×32 | 2,080 | train 拟合（CSSD/sdb32） | col norm 0.113–1.016 |
| 节点绑定 W_A_S/W_A_C | 33×96 / 28×96 | 5,856 | 学习 | **死亡（denormal）** |
| 边绑定 W_E_S/W_E_C | 99×48 / 4×48 | 4,944 | 学习 | 存活（norm 1.76 / 1.12） |
| node_slot_encoder | 96→64→48 | 9,328 | 学习 | 第 0 层权重死亡→输出常量 |
| edge_slot_encoder | 48→48→32 | 3,920 | 学习 | 存活 |
| Sem108+size2 接口 | 110 | 0（拼接） | 固定构造 | 直接进入 fusion |
| fusion | 446→342→144 | **202,266** | 学习 | 输入 144 维为常量 |
| 任务字典 D_L/V_L | 144×288 / 288×144 | 82,944 | 学习 | 码 87% 稠密 |
| pair_projection | 144→48 | 6,912 | 学习 | 存活 |
| relation_encoder | 15→96→48 | 6,384 | 学习 | 存活 |
| distance_gate | 5×48 | 240 | 学习 | 存活 |
| pair_encoder | 192→192→48 | 46,704 | 学习 | 存活 |
| global_encoder | 62→32→32 | 3,136 | 学习 | 存活（C6 下无化学直方图） |
| topology_encoder | 25→16→8 | 552 | 学习 | 存活 |
| reader | 814→39→39→1 | 33,385 | 学习 | H2 有效秩 ~2，78% 死单元 |

**Full 相对 Small（106,925）新增 301,726：fusion +145,788（48.3%）、任务字典 +73,728（24.4%）、
pair_encoder +41,376（13.7%）、reader +29,250（9.7%）、pair_projection +6,144、
relation_encoder +5,280、distance_gate +160。** 绑定与 slot encoder **没有扩宽**。
所以“扩参数”几乎全部落在 fusion（处理固定 446 输入）和任务字典上，而这两处正是本审计
探针覆盖的位置。

### 3.3 绑定/聚合的精确含义与可区分性

节点绑定对每个 occurrence 计算 `u = (z_i·W_A_S) ⊙ (q_a·W_A_C)`，再按 `(root, shell)`
求和。这是一个**双线性**联合：它保留 (结构码方向, 原子类型) 的乘性共现，但
`(z·A)⊙(q·C)` 对每个输出维只保留一个标量积；同一 shell 内不同 occurrence 的顺序、
以及“哪个结构码配哪个邻居原子”的更高阶关系被求和抹平（这正是 clean-mechanism round
用 independence null `(Σa)(Σc)/(n√D)` 检验的东西，C6 下 node Δ 平均 +0.0093、edge Δ +0.0401）。
边绑定同理，但 `g=[c_u+c_v;|c_u−c_v|;c_u*c_v]` 额外保留端点结构码的对称/反对称组合。
**当前 Full 中节点侧这个联合恰好为 0**，所以上述节点侧保留性在预测中不发生。

不可区分/未检查的结构区别（待验证假设，不是结论）：
- `Σ_{occ}` 抹平 occurrence 顺序与多重集；相同 (shell 计数, 原子类型计数) 的不同图在节点侧
  不可区分。
- 边侧按 6 个 shellpair 求和，保留方向吗？`env_bond_shellpair` 由无序 pair 定义，
  故键方向被对称化（同一 shellpair 内 u,v 顺序不影响 `g` 的前两块，但 `c_u*c_v` 对称、
  `|c_u−c_v|` 对称，所以方向不可区分）。这是设计选择，本审计未测其代价。

### 3.4 损失、重构约束与两套字典

训练总 loss（`train_cssd`）：
`L = L1(pred, y) + λ·E[ ‖r − α_res Dbar_perpᵀ‖² / (‖φ‖²+ε) ]`，`λ = H1_LAMBDA ≈ 33.96`。
**结构重构约束是显式加在总 loss 里的**（CSSD residual 重构）；**任务字典的
低维重构 `‖x − α Dbarᵀ‖²` 只是诊断，从不进入 loss**（`bridge_reconstruction_diagnostic`）。
两套字典必须分开：
- **上游结构字典 D（65×32, s=8）**：IHT 稀疏，coord 8 个非零（`coord_nnz_per_row_mean=8`），
  健康（active 32/32）。
- **任务 latent dictionary D_L/V_L（144×288）**：码 87% 非零，几乎稠密；承重但与“稀疏”无关。

### 3.5 哪些操作丢信息、哪些只是坐标变换

- **可逆/坐标变换：** 行 RMS 归一化 `x=h/scale` 与 `E=scale·(α V_L)` 成对，尺度可逆；
  `[Sem108;size2]` 是拼接，不丢信息。
- **可能丢信息（待验证）：** 节点/边按 shell 求和；pair 的对称组合；reader 的 814→39→39。
  本审计的探针只证明“pre-fusion 接口矩 / 节点联合矩在 `[1,H2]` 凸头上无线性剩余信号”，
  不能证明这些操作永久丢信息。
- **已确认的“信息入口”事实：** C6 mask 在训练与评估中都把
  global atom/bond 直方图、unary count、pair count、relation path_count 置零。
  clean-mechanism round 的 GS1（graph-chemistry row shuffle）在 C6 上 ΔMAE 恰为 0.0000，
  即这些通道在 C6 中被结构性弃用；所以“把全局化学直方图加回来”不是本审计要买的方向。

---

## 4. 阶段 B — 当前 checkpoint 的功能性检查

`audit_full.json` 与 `phase_c_stats.json` 记录全部数值。

**分支尺度（valid，graph-weighted，`audit_full.json::branches`）：**

| 分支 | absmean | absmax | zero_frac | 备注 |
|---|---|---|---|---|
| node_slot | 0.0 | 0.0 | **1.000** | 完全死亡 |
| edge_slot | 1.53e-4 | 0.372 | 0.992 | 构造稀疏（多数 (root,shellpair) 无键） |
| node_out | 5.84e-3 | 0.191 | 0.0 | **逐行常量**（per-row std=0） |
| edge_out | 1.36e-3 | 0.548 | 0.0 | 存活 |
| E (bridge out) | 0.154 | — | — | per-dim std 0.009–1.10 |

**两套字典：**
- 结构字典 D：col norm 均值 0.616，min 0.113，max 1.016。
- 任务字典：mean nonzero 250.89/288 = 0.871，p50 255，p95 275；
  `‖E−h‖/‖h‖=0.63`，per-row corr(E,h)=0.85；E 参与比 4.32，top1 0.448。
  → 任务码接近稠密时，bridge 更像“低秩、按行归一化的稠密编码器”，不是稀疏字典。

**图级表示 R（valid）谱：** stable rank 3.46，participation ratio 8.05，
top1 0.289，top5 0.609。分块：unary_first SR 1.60、unary_second 1.73、
pair_all SR 3.99、global 1.69、topology 1.23。**R 的有效信号只有 ~3–8 维**，
主要由 size/count 主导。

**reader：** H2 stable rank 1.46，participation ratio 1.98，top1 0.685，top5 0.974，
**78.4% 单元恒为 0**。即 `[1,H2]` 凸头实际只有 ~2 个有效方向——这是 CODE 探针能力上限的
量化解释，也是本审计 D1 的 `PROBE_UNDERPOWERED` 边界。

**Sem108 旧结论在当前 Full 是否成立：** 旧 jointbond 审计在 Sem108 父 checkpoint 上
报告 `W_A_S/W_A_C` 塌缩到 denormal。**本审计确认当前 Full soup 同样如此**，并进一步
确认 `node_encoder.0.weight` 也塌缩、`node_out` 逐行常量。机制是自锁的：
`W_A` 塌缩 → node_slot≡0 → `node_encoder.0.weight` 的输入梯度为 0 → weight decay 把该权重
也推向 0；而含 bias 的 encoder 在输入为 0 时仍输出常量，所以下游看不出“缺失”。
这不是“某个投影方向失活”，而是**整条节点路径失活**。

**梯度去向：** 发布记录 `summary.json` 给出 MAE-only task grad：`D_L 9.63e-2`、
`V_L 2.43e-1`，两者都收到任务梯度且移动（Frobenius 16.5 / 12.8）；zero-code ΔMAE 1.44、
reset-to-init ΔMAE 1.33 → bridge 承重。但“承重”不等于“稀疏字典优于稠密对照”。

**局部干预（复用已有证据，不重复做）：** clean-mechanism round 已对
C6 检查点做过节点/边 independence-null 替换（node Δ 均值 +0.0093、edge +0.0401）；
本审计不重做，只指出**当前 Full 的节点侧已经是 0，无法再被该探针测量**。

---

## 5. 阶段 C — 基线与误差预算

`valid_per_graph.csv`（列：`split, graph_id, y, pred_raw, train_fitted_bias,
pred_calibrated, abs_error, fixed_group`）与 `phase_c_stats.json`。

**总体（official train 10000 / valid 1000）：**

| | train | valid |
|---|---:|---:|
| raw MAE (eval-mode) | 0.045397 | 0.119154 |
| calibrated MAE | 0.032950 | 0.115066 |
| median abs (cal) | 0.023415 | 0.059156 |
| mean signed (cal) | +0.001271 | **−0.022459** |

valid calibrated 上仍残留 −0.0225 的系统偏置（train 中位残差标定后），说明 train/valid
的残差分布不完全同构。

**误差集中度（calibrated valid）：**

| 量 | 值 |
|---|---:|
| id172 贡献 | 0.018918（占绝对误差和 16.4%） |
| 去掉 id172 的 MAE | 0.096244 |
| top1% / top5% / top10% 误差和占比 | 25.3% / 39.8% / 50.6% |
| central 90% MAE | 0.063195 |
| central 95% MAE | 0.072950 |

**按 train-fit target 分位（valid）：**

| 组 | n | y 区间 | MAE | 误差和占比 | 组内 signed |
|---|---:|---|---:|---:|---:|
| q0 | 93 | [−20.34, −2.55] | **0.3812** | 30.8% | −0.258 |
| q1 | 156 | [−2.52, −1.05] | 0.1069 | 14.5% | −0.028 |
| q2 | 256 | [−1.04, 0.44] | 0.1107 | 24.6% | −0.001 |
| q3 | 252 | [0.44, 1.41] | 0.0778 | 17.0% | +0.000 |
| q4 | 149 | [1.41, 2.11] | 0.0664 | 8.6% | +0.023 |
| q5 | 94 | [2.11, 3.74] | 0.0543 | 4.4% | +0.028 |

train 上同分位 MAE 为 0.0576 / 0.0379 / 0.0290 / 0.0291 / 0.0280 / 0.0277。
**即：模型能拟合 train 尾部（0.058），valid 尾部 0.381，差 6.6×；中央 q2 的
valid/train 比也达 3.8×。** 这更像泛化失败而非“表示装不下”。

**尾部 vs 其余（y < train p5 = −3.55）：** 44 行，MAE 0.684，占误差和 26.2%；
y<-3 的 valid 65 行 signed err −0.381、回归斜率 dp/dy=0.287（train 同段 716 行斜率 1.03、
signed err −0.015）。**valid 预测最小值 −10.26，而 valid 标签最小 −20.34。**

**覆盖/稀有度（固定输入度量：size + 原子类型直方图 + 键类型直方图，train 标准化，1-NN）：**

| 组 | n | 到最近 train 邻居距离（mean/median） | \|y − y_nn\| | NN 属于 train 尾部的比例 |
|---|---:|---|---:|---:|
| valid 尾部 (y<p5) | 44 | 1.573 / 1.414 | 2.85 | 43.2% |
| valid 其余 | 956 | 0.911 / 0.812 | 0.90 | 3.2% |

另外：高误差 top10% 在标准化 topology 特征上的 1-NN 距离 0.0485 vs 其余 0.0047；
`corr(target, |res|) = −0.383`；`corr(nodecount, |res|) = −0.015`（尺寸本身不是主因）。
**尾部在固定化学/尺寸指纹上确实更孤立，且最近邻标签差很大**——覆盖是尾部的一个
竞争解释（该指纹较粗，且 1-NN 整体 MAE 0.988 远差于模型，所以只作定性证据）。

**id172 身份核对：** `ids=172`，`y=-20.340497970581055`，`p_base=-1.3885539770126343`，
`abs_err_raw=18.95194`，calibrated abs 18.91766；与交接报告一致。**不删除、不改标签、
不只报排除后成绩。**

**固定 `id%5` 分组（描述性）：** MAE 0.0979 / 0.0893 / **0.1739** / 0.1106 / 0.1036；
group 2 明显更差（含 id172）。仅作描述，不是独立验证集。

**阶段 C 结论（有数量级）：** 尾部（44 行，4.4%）贡献约 26% 误差和；单个 id172 贡献
约 0.019 MAE；即使完美修好 id172，valid 也只到 ~0.096。中央 90% 仍有 0.063 的 MAE。
所以**只修极端点不够，但“尾部 + 中央普遍偏大”共同存在**；这既不是单纯覆盖不足，
也不是单纯表示不足。

---

## 6. 阶段 D — 条件预测探针

**冻结配置**（`protocols/zinc-full-bottleneck-audit-v1.yaml`，fit 前写定）：

| 名称 | 设计 | 宽度 | 结果 valid MAE | gain vs baseline | status |
|---|---|---:|---:|---:|---|
| baseline B | `[1, H2]` | 40 | 0.115024135 | — | CONVERGED |
| P1 | `[1, H2, Z]`，Z = 每图 `[Σz, Σz²]`（446 维 pre-fusion 接口） | 932 | 0.115295754 | **−0.000272** | CONVERGED |
| P2 | `[1, H2, NJ]`，NJ = `Σ_i outer(coord_i, onehot(atom_i))`（33×28） | 964 | 0.115155179 | **−0.000131** | CONVERGED |
| C1 | `[1, H2, RP892]`，H2 的固定随机投影 | 932 | 0.115023773 | +3.6e-7 | CONVERGED |
| C2 | `[1, H2, RP924]` | 964 | 0.115023690 | +4.5e-7 | CONVERGED |

求解器：复用 upstream prototype `fit_mae`（MAE+L2 ADMM，`λ=1e-5`，gap 1e-6），
train 拟合、valid 评估，缩放/RMS 只用 train。

**数值阳性对照：** 用冻结 reader 权重从 R 重算 H2 并复现 frozen 预测，
max abs diff = 1.3e-6（`audit_full.json::reader_replay_max_abs_diff`）；
`[1,H2]` refit 复现 CODE 基线 0.115023784（本审计 0.115024135）。探针数值合格。

**判定：** 两个候选块都**未过 0.003 门槛**，且方向为负；同宽对照几乎为 0，
排除“只是 head 变宽/正则化”的解释。P2 对 id172 有 +0.0056 的改善但整体为负，
说明节点联合矩能帮极端单点、不能帮普遍行。分组方向也不一致（各 3/5）。

**边界（必须写清）：**
- 这是 **frozen-representation head split**：backbone 已看过全部 train 标签，
  不是端到端 OOF，也不是干净泛化证据。
- 负结果只关闭“P1 这个接口矩 / P2 这个联合矩 + `[1,H2]` 凸头”两个组合；
  不关闭整个表示方向、不证明字典无用、不证明所有读出都不行。
- H2 有效秩 ~2、78% 死单元 → 该凸头**探不出非线性信息**；若未来要问非线性可恢复性，
  需要换更强的读出或先修复 H2 的塌缩，并另立协议。
- 未做逐图 bootstrap / 多 seed（确定性求解器）；valid 已被反复探索，是探索性证据。

---

## 7. 判断与后续方案

### 7.1 分类（按任务表）

| 类别 | 是否成立 | 证据 | 决策 |
|---|---|---|---|
| **实现/协议问题** | **成立（已定位）** | 节点绑定整条路径 denormal 且自锁；soup 用报告集选模（乐观偏差 ~0.006） | 记录；节点修复若要买，须单独 pilot；选模改为固定窗/独立 split |
| 上游表示/对应关系受限 | 部分 | 节点侧结构×原子对应在预测中不存在；但 P2 联合矩凸头无增益 | 不购买“修节点绑定”的从零训练 |
| 信息存在但读取困难 | 未确认 | `[1,H2]` 与 `[1,R]` 线性头都到不了更好；H2 有效秩 ~2 | 需要更强读出/非线性可恢复性证据 |
| **泛化/覆盖/尾部** | **证据最强** | train 尾部拟合 0.058 vs valid 0.381；valid 尾部孤立；id172 0.019 | 若买训练，只买针对尾部的配对 pilot |
| 训练/数值问题 | 次要 | 节点塌缩是数值/自锁现象，但 P2 显示无性能上限 | 不泛化为“多训一点” |
| 暂未定位 | 是（非线性部分） | 凸头能力受限 | 补最小追加证据（§7.3） |

### 7.2 已确认事实 / 支持但未确认 / 未知

**已确认（可复放）：** 见 §1 的 1–4；节点死亡、任务码 87% 稠密、误差预算、
train/valid 尾部斜率差、两个探针负结果、C6 通道弃用。

**支持但未确认：** 尾部差主要来自“覆盖/稀有度”而非纯噪声（1-NN 指纹证据较粗）；
节点塌缩是“Sem108 使其冗余”还是“训练病理”（两者都能解释，本审计未区分）。

**未知：** 是否存在任何非线性可恢复的剩余信号；固定窗/独立选模下父 soup 的真实水平；
尾部标签本身的噪声/不可约成分；H2 塌缩（78% 死）是否是 reader 的瓶颈。

### 7.3 下一步（最多一个方向；若不该训练则给下一条证据）

**不推荐购买任何“加模块 / 扩参 / 从零 320 轮”的 run。** 依据：两个最有前途的压缩位置
（pre-fusion 接口、节点联合）条件探针均为负且低于门槛；R3/DROP/CODE 也都失败；
warm-80 的 matched calibrated gain 仅 +0.000667。

**最有价值的下一条证据（不训练）：**
> 在冻结 R 上做一次**尾部可恢复性检查**：用 train 内部固定折（不是 official valid）
> 拟合一个更强的读出（例如 2 层宽 MLP 或梯度提升），比较它对
> 中央行 vs 尾行的 train 内部泛化误差，并与当前 reader 对照。
> 目的：区分“R 里对尾部有可泛化信息但当前 reader 读不出”（→ 买读出/接口改动）
> 与“R 对尾部没有可泛化信息、尾部本质靠覆盖/噪声”（→ 不买架构，改选模/报告口径）。

**若未来确要买一个训练 pilot，唯一值得买的是**（须新 preregistration）：
- 假设：尾部欠拟合来自**优化/损失几何**而非表示。最小配对：
  共享父 soup，warm-80，对照 vs 一个只改“尾行相对权重/分位损失”的臂，
  其余（数据、lr、wd、clip、固定末 5 轮 soup、train-median 标定）完全一致。
- 通过门槛：calibrated valid gain ≥ 0.003 且 > 同维控制，且包含/排除 id172 方向一致，
  且固定 `id%5` 分组 ≥4/5 同向。
- 失败则关闭“尾部欠拟合=损失几何问题”，不追加变体。
- 但**本轮不启动**：先补 §7.3 的冻结 R 可恢复性证据更便宜、更判别。

**若想修节点绑定**（研究主张的机制完整性）：这是一个“机制对齐”而非“性能”动作。
最小 pilot 是：重新初始化 `W_A_S/W_A_C` 并短 warm，检验 node-slot 是否恢复非零、
以及是否带来 ≥0.003 的 calibrated valid 改善。**但 P2 的负结果预示收益上限很低**，
所以优先级低于尾部证据。

---

## 8. 复现清单

- **代码/协议：** `protocols/zinc-full-bottleneck-audit-v1.yaml`；
  `configs/luyin16/zinc_full_bottleneck_audit_v1.yaml`；
  runner `tracks/ksvd/src/ksvd_research/runners/zinc_full_bottleneck_audit_v1.py`
  （已注册进 `runners/__init__.py`）。
- **审计脚本：** `results/zinc_full_bottleneck_audit_v1/audit_full.py`（replay+分支）、
  `stats_phase_c.py`（误差预算）、`extract_probe_features.py`（特征）、
  `probe_phase_d.py`（凸头拟合）。
- **run：** `uv run research run zinc_full_bottleneck_audit_v1 --purpose ...`
  → `tracks/ksvd/runs/2026/10/02/20261002-175703-469df2f8`（probe，56.8 s）。
- **checkpoint：** `results/e2e_dictenv_scale_v1/checkpoints/SCALE-FULL-seed0_soup_state.pt`
  sha `17f5fcc3…574eb`；结构字典 `results/sdb_v0/dictionary.pt` sha `925d573a…`；
  subspace `results/e2e_dictenv_common_subspace_dictionary_v1/common_subspace.json`。
- **数据/缓存：** PyG ZINC official train/valid；`results/zinc_static_dictionary_pair/cache/`
  + `results/e2e_dictenv_p1/cache/`；handoff `results/zinc_dictionary_real_data_handoff/`
  （R / reader / p_base / y，split fingerprint `58c69506…`）。
- **随机性：** 固定 seed 0；控制投影 seed 20261002；确定性 ADMM，无 bootstrap。
- **未执行：** 任何 backbone 训练、任何新 320/80 轮 run、官方 test 读取、RDKit 新描述符。
- **产物：** `audit_full.json`、`phase_c_stats.json`、`valid_per_graph.csv`、
  `probe_features.npz`、`phase_d_probe.json`、`probe_features_meta.json`。

---

## 9. 五问直答

1. **Full 的新增参数是否主要用在真正受限的位置？**
   不是。新增 301,726 中 48.3% 在 fusion（446→342→144）、24.4% 在任务字典，
   但 fusion 的 144 维节点输入是常量、任务码 87% 稠密且 CODE 已证其矩无增益。
   绑定与 slot encoder 完全没扩。**支持“扩参位置不在当前受限处”的证据是分支统计与
   两个负探针；不能判断的是**：换一个更大的读出或非线性头是否会改变结论（凸头能力受限）。

2. **字典/绑定/聚合/读出分别在做什么，哪些参与预测？**
   结构字典 D 提供 coord（IHT s=8）；**节点绑定不参与（恒 0）**；边绑定参与且是
   结构—键语义的主要入口；`Sem108` 直接接口参与；fusion 参与（但 144/446 输入为常量）；
   任务字典参与（zero-code ΔMAE 1.44）但非稀疏；unary/pair 聚合参与；reader 参与，
   但 H2 有效秩 ~2、78% 死单元。C6 下全局化学直方图/count/path_count 不参与。

3. **剩余误差更像哪一类？**
   最像**泛化/尾部覆盖**：train 尾部可拟合（斜率 1.03）、valid 尾部严重回缩（斜率 0.29），
   尾部在固定化学/尺寸指纹上更孤立；单个 id172 占 16.4% 误差和，但中央 90% 仍有 0.063 MAE。
   **无法排除**“信息读取”与“表示入口”在非线性层面仍有责任，因为本审计的凸头探针
   在 H2 上能力有限（`PROBE_UNDERPOWERED` 边界）。所以排序是
   “泛化/尾部 ≳ 读取/表示（未定） > 训练数值（仅节点塌缩这一具体异常）”。

4. **哪些常见改动现在可以暂缓？**
   - 加节点绑定容量 / 修节点绑定：暂缓（P2 联合矩凸头无增益；且机制上 Sem108 可能已冗余）。
   - 加 pre-fusion / fusion 宽度：暂缓（P1 负、R 有效秩低、capacity_localization_v2 也负）。
   - 加任务码矩/换 code 统计量：暂缓（CODE 已负，本审计不重复）。
   - 加回全局化学直方图：暂缓（clean-mechanism GS1 Δ=0）。
   - 继续同配置延长训练：暂缓（trajectory review + matched calibrated gain 0.000667）。
   边界：这些结论都基于**当前 checkpoint + 固定探针/短 warm**，不覆盖“换非线性读出”
   或“从零训练”的未测情形。

5. **下一步只允许一个小实验，选择什么？**
   尚不该训练。选择补 **§7.3 的“冻结 R 尾部可恢复性”审计证据**：
   在 train 内部固定折上比较强读出对尾行 vs 中央行的可泛化误差，判断尾部是
   “可读但当前 reader 读不出”还是“R 对尾部没有可泛化信息”。
   这条证据最便宜、最能区分“买读出改动”与“不买架构、只改选模/报告口径”。
