# Second-Hop Composition Pilot — REPORT

**Round:** `zinc-secondhop-composition-pilot-v1`
**Track:** `tracks/structural-sufficiency/zinc_secondhop_composition_pilot_v1/`
**状态:** 阶段一完成，**按冻结门槛未购买阶段二**（无 seed 1）
**预注册:** `PROTOCOL.md`（sha256 `010f7869…`，见 `analysis/provenance.json`）在写任何训练代码/跑任何 GPU 之前提交（commit `73ebbb2`）
**冻结 config:** `configs/shc_v1.json`（sha256 `eb693b28…`）
**实现 commit:** `73ebbb2`（输入/模型/门禁）+ `97b360f`（train.json 复用守卫）；两次 GPU 运行都在 `97b360f` 上
**上游审计:** `zinc_path_binding_sufficiency_audit_v1`（REDUNDANT / NO-GO on DFPB；建议唯一最小实验 = 第二跳组成注入）
**复用主体:** `zinc_local_relation_composition_v1` 的 LCRC-A（代码/cache/A_s0/A_s1 只读）
**official test:** **未加载**（任何阶段、任何理由）；`official_test_loaded=false`
**GPU:** 2 × A100-40GB（`res`，`res-gpu0` / `res-gpu1`），driver 550.163.01，torch 2.5.1+cu124；**1.048 GPU·h**（目标 1，硬上限 3）

---

## 0. 一句话结论

**第二跳的真实配对不是 LCRC-A 的 MAE 瓶颈。** 在完全冻结的配方下，把 `[B_v, C]`（96+32 = 128 维）
接进 `LocalContent` 后：网络**确实在用**这组统计（128/128 新列离开零、`z_new/z_old≈0.30`、
冻结权重下切换 C 通道改变 100% 分子的预测），但 primary 指标 **valid full raw-y MAE = 0.078976**
比冻结 A_s0（0.076582）**差 +0.00240**；真正的配对（Real）比同边际同 path count 的均场（Null, 0.082897）
好 **−0.00392**，方向一致但 95% CI 跨零，且几乎全部来自高 `|y|` 尾部（5 行占尾部效应的 64%）。
在 935 个 ordinary 行上 Real 比 A 好 **−0.00105**（广而小：52% 行变好、中位 −0.00090、top-5 只占 9.2%），
但这是 0.001 量级、低于预注册的 0.003 购买门槛，也小于历史 A 的 seed 间差（0.00392）。
所以：**S2 缺口在数学/表示上真实存在（配对通道非空），但在本配置下没有可观测到值得购买的 MAE 收益**——
它不是实际瓶颈；本轮停止该配置，不据此否定所有可学习投影。

---

## 1. 设计（冻结内容的摘要，详见 `PROTOCOL.md`）

主干 = LCRC-A，逐位一致：

* `a_v = atom_encoder(atom_feat_v)`，`b_e = bond_encoder([a_u+a_v, |a_u−a_v|, a_u⊙a_v, bond_feat_e])`
* `h_v = LN(MLP([a_v, Σb, mean b, S1joint, S2_path, S2_content, scalars]))`（529 → 192 → 64）
* `y = Σ_v f(h_v) + ρ(global33, mean h, mean h², log(1+|V|))`

唯一改动：`LocalContent` 输入追加 128 维（529 → 657）：

```
n_v        = 该根的有序非回溯二跳路径数（与冻结 cache 的 path_ptr 逐原子一致）
B_v        = (1/√n_v) Σ_p b_{e2(p)}                       ∈ R^96
U_p        = P_b b_{e2(p)}   (P_b: 96→32, 固定, 不训练)
V_p        = P_a a_{u(p)}    (P_a: 96→32, 固定, 不训练)
C_real(v)  = (1/√n_v) Σ_p (U_p ⊙ V_p)                     ∈ R^32     ← Real arm
C_null(v)  = √n_v (mean_p U_p) ⊙ (mean_p V_p)             ∈ R^32     ← Null arm
n_v = 0 → 三者全零；n_v = 1 → C_real = C_null = U_p ⊙ V_p（逐位校验）
```

* `b_{e2}` 用**同一个** `bond_encoder` 作用在第二跳键上；没有新 encoder。
* 均场只在**同一 root 内部**取：不跨 root、不跨分子、不跨 batch。
* `P_b/P_a` 来自单一冻结种子 `20261105`（`randn(96,32)/√96`，先 B 后 A），sha256
  `af217bc6…` / `fae06e1b…`，`requires_grad=False`，所有 arm/seed 共享，**没有用 valid 选投影**。
* 扩宽方式是**原位**把历史 `Linear(529,192)` 变成 `Linear(657,192)`：新 128 列权重全零、bias 不变、
  其余张量逐位等于历史 A 初始化 ⇒ **step-0 输出 = A（max|Δy| = 0.0）**。
* 参数：A 237,519 → 新 262,095（≤ 500k）。无 attention / junction / 新 reader / 新目标分项。

Real/Null 的差别**只有** `C_real` vs `C_null`：同一 `n_v`、同一 `B_v`、同一 `{U_p}`/`{V_p}` 边际、
同一参数结构/初值/数据顺序（full-state hash 相同）。这正是"只有真实配对不同"的对照。

---

## 2. CPU 门禁（`analysis/cpu_checks.json`，8/8 PASS，0 GPU）

| gate | 结果 |
|---|---|
| **G1 路径枚举** | 从冻结 cache 原始邻接重新枚举：train 692,716 / valid 69,040 条；`path_cat/path_end63/path_end/path_closure/nbr_code/root 顺序` 全部 mismatch **0**；`root=end`、`e1=e2`、`e1` 关联 `(v,w)`、`e2` 关联 `(w,u)`、`e2` 端点是 `u` 全部 **0**；与审计 `paths.json` 的路径数一致 |
| **G2 历史初值一致** | 新旧 base 模型在 seed 0/1 上 `state_hash` 与历史记录**逐位相等**（`11c698ae…` / `d9422859…`），全部张量 `torch.equal`；扩宽后除首层新 128 列（全 0）外无差异；step-0 `old vs base vs Real vs Null` max\|Δ\| = **0.0**；参数 262,095；无 junction/attention |
| **G3 Real/Null** | full-state hash 相同；step-1 新列梯度范数 10.85 = 0.197×旧列，128/128 列非零，全部有限；`W_new` 0 → 0.156（单步 lr 1e-3 检查）；30 步 loss 1.605→1.165、`W_new`=0.997；checkpoint 保存/重载预测逐位相等 |
| **G4 不变性** | 原子重编号 / 键反向 / batch 16 vs 8+8 vs 单分子 / 行序置换，在**非零**新权重下（Real 与 null 通道、Null arm）：最大差 **1.07e-6**（原子置换下的 h），键反向 7.2e-7，batch 拆分 2.4e-7，行序 0.0；tolerance 1e-5 |
| **G5 边际/配对语义** | 合成反例：同边际同 path count，`B`/`C_null` 逐位相同、`C_real` 差 0.707；真实数据 root 内交换 64 个 n=2 root 的配对：`B` 0.0、`C_null` 0.0、`C_real` 3.24、`y` 0.314（null 通道 `y` 变化 0.0）；`n_v≤1` 原子 `C_real ≡ C_null`（max 0.0），`n_v≥2` 差 3.01 |
| **G6 尺度/覆盖** | valid 256 分子：`x_old_rms 1.269`、`B 0.783`、`C_real 0.350`、`C_null 0.267`、新块 0.701；`C_real/C_null` RMS 比 **1.31**（带 [0.5,2]）；`n_v=0` 0 个、`n_v=1` 239、`n_v≥2` 5625；`C_real≠C_null` 96.2% 原子 |
| **G7 评估路径** | A_s0/A_s1 checkpoint 载入扩宽模型：MAE 与冻结 npz 重放值差 **1.0e-8 / 7.8e-9**（相对 0.0765816361 / 0.0805037835），预测最大差 3.8e-6（CPU vs GPU kernel），checkpoint sha 前缀一致 |
| **G8 守卫** | 参数 262,095 ≤ 500k；投影跨 arm/seed 完全一致且不可训练；`build_supplement("test")` 抛错；test 未加载 |

补充（冻结过程中的一次披露）：历史 `train.json` 里 `best.valid_mae`（epoch 循环）与
`best_valid_mae_eval`（checkpoint 重放）相差 ~3e-9（GPU kernel 非确定性），
`predictions/valid_best.npz` 重算与**重放值**逐位一致；本报告所有 A 参照都取 npz 重算值。

---

## 3. 主结果（阶段一，seed 0，官方 valid 1,000 行）

主指标：**official-valid 全量 raw-y MAE（best epoch）**。`A_s0` 为冻结历史预测（不重训）。

| model | **full** | drop row172 | drop fixed-worst10 | ordinary (935) | r2-unseen ord (225) | seen ord (710) | tail (65) | r2-unseen (251) | ring-rich (≥4 环) | worst-10 mean AE | best ep | W20 | T100 | train MAE (eval) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **A_s0** (frozen) | **0.076582** | 0.070076 | 0.062517 | 0.063545 | 0.130915 | 0.042195 | 0.264113 | 0.145477 | 0.100707 | 1.46894 | 476 | 0.077333 | 0.077359 | 0.005268 |
| **Real_s0** | **0.078976** | 0.069955 | 0.062419 | **0.062500** | **0.125420** | 0.042561 | 0.315978 | 0.144972 | 0.101900 | 1.71818 | 368 | 0.079500 | 0.079534 | 0.012544 |
| **Null_s0** | **0.082897** | 0.072317 | 0.065149 | 0.063813 | 0.129254 | 0.043075 | 0.357420 | 0.152694 | 0.103747 | 1.83999 | 522 | 0.083033 | 0.083056 | 0.002387 |

读法（先把三件事分开）：

1. **primary 指标上两个新 arm 都不如 A**：Real +0.00240、Null +0.00632（越大越差）。
   `W20/T100/final` 同方向（Real 0.07950/0.07953/0.07953 vs A 0.07733/0.07736/0.07737），
   所以这不是"best epoch 选早了"。
2. **去掉已知异常行后 Real 与 A 基本打平**：drop row172 = −0.000121，drop fixed worst-10 =
   −0.000099。完整 valid 上 Real 的恶化 **+0.002516 来自 row172 单行**（`single_row_share = 1.051`），
   row172（`y = −20.3405`，ZINC `−cycles` 标签人工项的已知异常行）的 AE：A 6.575 → Real 9.091 → Null 10.653。
3. **普通行上 Real 是广而小的改善**：935 个 ordinary 行 Real−A = −0.001045，
   52.0% 行变好、中位 −0.00090、top-5 行只占该效应的 9.2%；tail 的均值恶化 +0.0519 但 tail 中位数是
   −0.0036（即尾部多数行也略好，均值被 5 行拉坏：top-5 占 56.5%）。

---

## 4. 逐分子配对对比（同 1,000 行；负 = 前者更好）

| 对比 | full | drop row172 | drop fixed-worst10 | bootstrap 95% CI | win rate | 单行最大占比 | ordinary | r2-unseen ord | seen ord | tail |
|---|---|---|---|---|---|---|---|---|---|---|
| **Real − Null** | **−0.003921** | −0.002362 | −0.002730 | [−0.010734, +0.002133] | 0.489 | 0.424 | −0.001310 | **−0.003834** | −0.000514 | −0.041435 |
| **Real − A** | **+0.002395** | −0.000121 | −0.000099 | [−0.004085, +0.010414] | 0.519 | **1.051** | −0.001045 | −0.005495 | +0.000366 | +0.051865 |
| **Null − A** | **+0.006316** | +0.002241 | +0.002631 | [−0.001280, +0.017168] | 0.508 | 0.646 | +0.000268 | −0.001661 | +0.000880 | +0.093307 |

对 full 差的贡献分解（group 内 ΔAE 之和 / 1000）：

| 对比 | ordinary | tail | row172 | fixed-worst10 |
|---|---|---|---|---|
| Real − Null | −0.001227 | −0.002694 | −0.001561 | −0.001218 |
| Real − A | −0.000976 | **+0.003371** | **+0.002516** | +0.002492 |
| Null − A | +0.000251 | **+0.006065** | **+0.004077** | +0.003711 |

行结构（判断"是否广"）：

| 对比 · 组 | n | mean ΔAE | median ΔAE | 变好行占比 | top-5 行占该组 \|Δ\| 之和 |
|---|---|---|---|---|---|
| Real−Null · ordinary | 935 | −0.00131 | **+0.00083** | 48.6% | 8.5% |
| Real−Null · r2-unseen ord | 225 | −0.00383 | −0.00184 | 50.7% | 18.7% |
| Real−Null · tail | 65 | −0.04144 | −0.00778 | 53.8% | **64.1%** |
| Real−A · ordinary | 935 | −0.00104 | −0.00090 | **52.0%** | 9.2% |
| Real−A · r2-unseen ord | 225 | −0.00549 | −0.00578 | 53.8% | 20.1% |
| Real−A · seen ord | 710 | +0.00037 | −0.00054 | 51.4% | 4.3% |
| Real−A · tail | 65 | +0.05186 | −0.00362 | 50.8% | **56.5%** |

三个必须一起读的结论：

* **Real 比 Null 好是方向一致的**（full、两个稳健变体、r2-unseen ordinary、ordinary 全是负号），
  说明"真实配对"不是空的；但 **CI 跨零**、ordinary 的中位 Δ 是 **+0.0008**（略差），
  且 full 效应主要是 5 个尾部大行（尾部 top-5 占 64%）。它是"小 + 尾部集中"，不是广泛效应。
* **Null（同内容、同边际、错误配对）严格差于 A**（+0.00632），也就是说：**第二跳内容本身不是白给的礼物**，
  把它以均场方式塞进去是净损失。Real 只是把这个损失减小了约三分之二。
* **Real 相对 A 的普通行改善是广而小的**（median −0.0009、52% 行、top-5 9.2%），
  但量级 0.001 小于预注册门槛 0.003，也小于历史 A 的 seed 间差 **0.00392**
  （A_s0 0.076582 vs A_s1 0.080504）。**单 seed 下连 Real−A 的符号都不能说已确立**。

---

## 5. 机制诊断（best checkpoint，冻结权重）

| run | `W_new/W_old` | `W_new` 键块 / 配对块范数 | step-1 `g_new/g_old` | 新列非零列数 | `z_new/z_old`（首层激励） | C 切换 mean\|Δpred\| | C 切换 ΔMAE | h participation ratio（init → best） |
|---|---|---|---|---|---|---|---|---|
| Real_s0 | 0.589 | 14.00 / 12.14 | 0.1967 | 128/128 | **0.296** | 0.03706 | **+0.013022**（0.078976→0.091999） | 6.96 → 4.31 |
| Null_s0 | 0.627 | 14.20 / 14.08 | 0.1949 | 128/128 | 0.278 | 0.03218 | **+0.010173**（0.082897→0.093070） | 6.96 → 4.99 |

* `W_new` 轨迹（Real）：0.152 @ep1 → 7.05 @ep50 → 15.69 @ep200 → 18.67 @ep400 → 18.88 @ep600；
  配方里的 500-step warmup 让"step-1 权重"只有 3.1e-4（CPU 门禁用固定 lr 1e-3 单独验证过它确实会动）。
* 输入尺度：best epoch 的 valid 上 `x_old_rms 1.525`、`B 1.468`、`C_real 1.571`、新块 1.494——
  新块与旧 X 同量级（这是"输入尺度"的实际读数，不是标准化）。
* `n_v=1` 的 451 个原子（3.8%）上 `C_real ≡ C_null`；`n_v≥2` 的 11,348 个原子上两者不同
  （96.2%），所以 Real/Null 的差别确实只在多路径 root 上。
* **C 切换的正确读法**：它测的是"冻结函数对 C 通道的依赖"，不是配对的反事实收益。
  切换后预测几乎全变、MAE 变差 +0.0130（Real）/ +0.0102（Null），说明网络把 C 当成真实输入在用；
  但被替换掉的通道是**训练分布外**的，所以这个数**不能**当作"真实配对值 0.013"。

---

## 6. 阶段一购买门槛（预注册的机械判定）

| 条件 | 结果 | 数值 |
|---|---|---|
| **条件 0（不崩）** `max(Real, Null) ≤ A_s0 + 0.005` | **FAIL** | 0.082897 > 0.081582 |
| **信号 A（vs A，任一变体 ≤ −0.003 且 full ≤ +0.002）** | **FAIL** | 最好的变体是 drop_row172 **−0.000121** |
| **信号 B（Real vs Null，任一变体 ≤ −0.003 且 full ≤ +0.002）** | **PASS** | full −0.003921（drop 变体 −0.0024/−0.0027 未达阈值） |
| **信号 C（r2-unseen ordinary ≤ −0.010 且 seen ordinary ≤ +0.002）** | **FAIL** | r2-unseen ord −0.003834（未达 −0.010）；seen ord −0.000514（满足后半） |
| **买阶段二** | **NO** | 条件 0 未过 ⇒ 不跑 seed 1 |

**这是一个"信号 B 通过、但被条件 0 否决"的停止点。** 我按冻结的合取式执行，没有事后放宽
条件 0（例如改成"只有携带信号的 arm 不能退化"就会让阶段二启动；那是另一个预注册决定，不是本轮的）。
因此：**无 seed 1、无阶段二、无第三阶段**。

停止时的事实边界：Real−Null 的 −0.0039 本身就在 seed 噪声带内（A 的两个 seed 相差 0.00392），
单 seed 无法把"配对有用"从执行漂移里分离出来；这也正是阶段二本来要买的东西，而门槛没有允许。

---

## 7. 直接回答本轮的核心问题

> **S2 的第二跳信息缺口，究竟是一个影响 MAE 的实际瓶颈，还是一个数学上存在、却没有可观测预测收益的表示差异？**

在本轮冻结的配置（固定投影、128 维零初始化注入、原配方、1 个 seed、valid 1,000 行）下，答案是：

1. **不是 MAE 瓶颈。** 真实配对 arm 的 primary 指标比 A **差 +0.00240**；两个稳健变体上与 A 打平
   （−0.0001）；最好的广而小改善（ordinary −0.00105）低于购买门槛，也小于历史 A 的 seed 间差。
   若 S2 缺口是瓶颈，**加上它应该让 A 变好，而不是变差**；观测到的方向相反（且这个反向由 row172 主导）。
2. **但也不是"纯数学差异、完全无信号"。** 在**同边际、同 path count、只换配对**的 Real/Null 对照里，
   Real 在每一个预定义组和每一个稳健变体上方向都更好（full −0.0039、drop172 −0.0024、
   dropW10 −0.0027、r2-unseen ordinary −0.0038），说明配对确实被网络利用（C 切换 +0.013 也证实它被使用）。
   只是这个信号**小、CI 跨零、且集中在高 `|y|` 尾部**，在 ordinary 行上中位数甚至是 +0.0008。
3. **损失/收益的结构**：真实配对把"第二跳内容注入"的净损失从 +0.0063（Null）降到 +0.0024（Real），
   并在 ordinary 行上把符号翻成 −0.0010；代价是尾部（尤其 row172）大幅恶化。也就是说，
   这个操作**重新分配**了误差，而不是降低了总误差。

因此本轮的实际结论是：**这个固定投影 + 128 维注入方式没有可购买的预测收益；S2 缺口在表示上是真的，
在 MAE 上不是瓶颈。** 按用户给定的决策含义："Real、Null 都不优于 A ⇒ 当前固定投影和接入方式没有预测收益，
停止该配置；不据此否定所有可学习投影。"

---

## 8. 必报指标清单（对照 PROTOCOL §5）

| 要求 | 位置 |
|---|---|
| 官方 valid 全量 raw-y MAE（主指标） | §3；`analysis/summary.json::stage1` |
| Real−Null / Real−A / Null−A，逐 seed（本轮仅 s0） | §4 |
| ordinary / r2-unseen ordinary / tail | §3–§4 |
| 去掉 row172、固定 worst-10 的影响 | §3–§4（含单行占比与贡献分解） |
| best epoch、W20、T100、eval-mode train MAE | §3 |
| 冻结 Real 权重后 C_real→C_null 的预测/MAE 变化 | §5（Real 与 Null 双向） |
| 新增列权重、梯度、激活尺度 | §5（init / step-1 / best；`z_new/z_old`；新块 RMS） |
| 参数、初始化一致性、数据/代码/checkpoint SHA、GPU 时间 | §9 + `analysis/provenance.json`、`analysis/provenance_gpu.json` |
| 不把 row172 的整体改善解释为广泛泛化 | §3 第 2 条、§4 行结构（row172 在 Real−A 中 share = 1.051） |

---

## 9. Provenance

| 项目 | 值 |
|---|---|
| code commit（两次 GPU 运行） | `97b360f15ae3` |
| 预注册 commit | `73ebbb2`（PROTOCOL + config + code + CPU gates） |
| PROTOCOL sha256 / config sha256 | `analysis/provenance.json` |
| train cache | content `ded004e8…`（10,000 mol / 231,664 atom / 692,716 path）；**远程文件 sha `1aef7f21…` 与历史 A 运行记录一致**；本地副本文件 sha 为 `8777494f…`（同 content，不同 torch 序列化字节） |
| valid cache | content `f28a0cea…`（1,000 / 23,083 / 69,040）；远程文件 sha `514e8c72…`（同上） |
| 第二跳补充表 | content `train 747fe048…` / `valid b085c4ec…`（远程重建与本地一致；`torch.save` 文件字节不稳定，故以 content sha 为门禁） |
| A_s0 checkpoint | `06340bcd9e45af03…`，valid 0.0765816361（npz 重放；epoch 循环记录 0.0765816392） |
| A_s1 checkpoint | `7225fe11fce0a24d…`，valid 0.0805037835 |
| Real_s0 checkpoint | best `97e4d02539ea5dd3…`，valid 0.0789764001 @ ep368 |
| Null_s0 checkpoint | best `116c084f6cfe43e1…`，valid 0.0828973208 @ ep522 |
| 投影 | `P_b af217bc694d2…`，`P_a fae06e1b3264…`（seed 20261105，不可训练） |
| run IDs | `sh-real-s0-20261028-221352-63a5c9ff`、`sh-null-s0-20261028-221413-b0170283` |
| execution regime | `res` / `a100-2`，2×A100-SXM4-40GB，driver 550.163.01，torch 2.5.1+cu124，CUDA 12.4，python 3.12.14 |
| GPU 时间 | Real 0.490 h + Null 0.556 h = **1.046 h**；+ 远程 2-epoch smoke 0.002 h ≈ **1.048 GPU·h**（硬上限 3） |
| official test | **未加载** |

**provenance 语义**：`analysis/provenance.json` 与 `cpu_checks.json` 里的
`git_commit=dc53fd7…`、`git_dirty_tracked=true` 是**生成时刻**的状态：代码停在
`dc53fd7`（阶段一代码 + 报告），脏的只有本次重新生成的分析产物文件本身（时间戳/该字段）。
所有数值在 `dc53fd7` 上重跑后逐位复现（Real 0.078976@368、Null 0.082897@522、
补充表 content sha `747fe048…`/`b085c4ec…` 不变）。产物随后单独提交。

**smoke 披露**：第一次 2-epoch smoke 忘记传 `--out`，写进了默认的 `runs/shReal_s0`（远程）；
正式 Real_s0 用 `--force` 覆盖了它（同一 seed/arm，`epochs=600`）。同一 smoke 在本地 CPU 复现，
valid 数字与远程 GPU **逐位相同**（0.433685 / 0.322342）——这也是本模型 CPU/GPU 确定性的一次实测。
另：`--force` 覆盖前已经存在的 `train.json` 会被新 payload 完全替换，报告以 `epochs=600` 的 payload 为准。

---

## 10. 统计与解释上的限制（必须与结论一起读）

1. **单 seed**。A 的 seed 间差 0.00392 大于本轮所有 Real−A 的观测差；Real−A 的符号在单 seed 下
   不能视为确立。冻结门槛因此拒绝购买 seed 1。
2. **只有一个固定投影**（预注册只允许一个；没有投影种子搜索）。因此"真实配对的投影读出"只测了一个
   32 维随机草图，不能推广到"所有可学习投影"。
3. **容量/优化混淆**：两个新 arm 都是同一个 +128 列注入；Null 更差说明内容注入本身有代价
   （尾部 +0.093）。因此 Real 的 +0.0024 不能被读成"配对把 A 搞坏了"，只能读成
   "注入带来的额外自由度在这个固定预算下没有回本，而真实配对把代价减小了约 2/3"。
4. **row172 是已知标签人工项异常行**（`y=−20.34`，ZINC `−cycles`），它的 A 误差本来就 6.58；
   本轮 Real 在该行 9.09。所有 full-valid 的比较都必须同时看 drop_row172。
5. **`C_real` 与 `C_null` 的边际（一阶矩）相同，但联合分布不同，输入 RMS 也差 1.31 倍**；
   这是"只换配对"设计的固有属性（不是缺陷），但意味着 Real−Null 不是"同分布、只换标签"的干预。
6. 本轮没有读 test；任何 test 结论都没有。

---

## 11. 复现

```bash
cd /home/calendar/code/research
# CPU（~40 s，0 GPU）：路径门禁 + 8 项 CPU gate
uv run python tracks/structural-sufficiency/zinc_secondhop_composition_pilot_v1/code/sh_cpu_checks.py
uv run pytest tracks/structural-sufficiency/zinc_secondhop_composition_pilot_v1/tests -q

# 回到本 commit 的部署，在 res 上（两次运行，各约 0.5 GPU-h）
rr doctor res && rr deploy res
rr run res sh-real-s0 --pool res-gpu0 --gpus 1 --time 1:30:00 \
  --result tracks/structural-sufficiency/zinc_secondhop_composition_pilot_v1/runs/shReal_s0 -- \
  python tracks/structural-sufficiency/zinc_secondhop_composition_pilot_v1/code/sh_train.py \
    train --arm Real --seed 0 --device cuda:0
rr run res sh-null-s0 --pool res-gpu1 --gpus 1 --time 1:30:00 \
  --result tracks/structural-sufficiency/zinc_secondhop_composition_pilot_v1/runs/shNull_s0 -- \
  python tracks/structural-sufficiency/zinc_secondhop_composition_pilot_v1/code/sh_train.py \
    train --arm Null --seed 0 --device cuda:0

# 本地分析（读 runs/ + 冻结 A 预测 + 冻结分组）
uv run python tracks/structural-sufficiency/zinc_secondhop_composition_pilot_v1/code/sh_analysis.py --stage 1
```

产出：`analysis/{cpu_checks,supplement,provenance,provenance_gpu,stage1_gate,summary}.json`、
`analysis/tables.md`、`analysis/per_molecule_valid.csv`；`runs/sh{Real,Null}_s0/`
（git-ignored，含 checkpoint / curve / predictions）。

---

## 12. 决定

见 `DECISION.md`：**停止该配置**（固定投影 + 128 维注入），不购买 seed 1 / 阶段二 / 第三阶段；
不据此否定所有可学习投影，也不重启 DFPB；后续任何动作需要新的预注册。
