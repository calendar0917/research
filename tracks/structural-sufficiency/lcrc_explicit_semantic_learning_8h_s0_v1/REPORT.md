# LCRC 显式结构主线：语义参数共享、可学习性与预测性能 —— 本轮报告

- round_id / protocol_id: `lcrc-explicit-semantic-learning-8h-s0-v1`
- 轨道目录: `tracks/structural-sufficiency/lcrc_explicit_semantic_learning_8h_s0_v1/`
- 执行环境: res 单机，双 A100(40G)，`res-gpu0`/`res-gpu1` 各一训练进程，池内逻辑 `cuda:0`，无 DDP
- 单训练 seed=0；官方 Train/Val/Test；每个 epoch 全量评三划分；选点只看 Validation，Test 仅报告并用于本轮路线取舍（已在 `analysis/test_access_log.json` 记录 20 条读取）

---

## 0. 结论速览（所改变的判断）

1. **机制层得到支持**：共享原子/键成分的参数共享确实在语义对齐时把不同路径组合的梯度耦合起来（重叠数 0/1/2 → 余弦核 0.082/0.316/0.632），而置换控制 FP 的核近乎平坦（~0.18–0.22）。真实分子编辑响应上 FS 也最好（edit MAE 0.293 < LS 0.308 < LP 0.334 / FP 0.334），符号一致率 FS 0.901 最高。
2. **性能层没有获得同向、可放大的进步**：在 Lipo 上 FS 相对匹配查表 LS 在 **Valid 上系统性更差**（R: −3.40%；Rwd: −3.40%；W0: −5.01%；N25: −3.65%；N50: −0.30%），只在 **Test 上更好**（R: +1.55%；Rwd: +4.88%；W0: +5.09%；N25: +0.99%；N50: +3.09%）。用固定 update(共同 grid) 逐点看，这个方向是稳定的：FS 在 Test 上 41–46/50 个 grid 点更优，却在 Valid 上仅 7–12/50 个点更优。
3. **DiD_semantics 方向不稳**：Lipo 上 Test DiD 为正（R: +0.0123；Rwd: +0.0129），Valid DiD 为负（R: −0.0280；Rwd: −0.0232）；ESOL 上 Valid DiD 微弱为正（+0.0021）而 Test DiD 为负（−0.0289）。500 次 scaffold 簇 bootstrap 的配对/DiD 区间**全部跨 0**。
4. **置换控制不能把 Test 增益判给"真实化学语义"**：FP（置换）相对 LP 在 Test 上也拿到约 +1% 量级、与 FS 相近的收益点；因此 Test 上的小改善更像**一般神经重参数化/参数共享正则**，而非真实组合语义独有的收益。
5. **放大开关 E 不成立 → 不购买 P4（PCBA/HIV）**。剩余 GPU 未用于扫新架构。

机械路线标签（可同时成立，不强行互斥）：
- `MECHANISM_SIGNAL_NO_PERFORMANCE_ADVANCE`（机制对照支持，但未胜过当前最佳查表/显式基线）
- `GENERIC_REPARAMETERIZATION`（FS−LS 与 FP−LP 量级相近，真实语义 DiD 未获同向支持）
- `NO_USEFUL_SIGNAL`（就"预测性能进步"这一层而言）

---

## 1. 预算与实际消耗（诚实记录）

| 项 | 计划 | 实际 |
|---|---|---|
| Wall（含实现/部署/训练/分析/交付） | 目标 7.5h / 硬 8h | 见文末，约 4.8h 达到"结果锁定"（训练全部完成），剩余用于分析/交付 |
| GPU | 软 14 / 硬 16 A100·h | 计费作业合计 ≈ **3.83 A100·h**（另加 smoke/preflight/成本探测约 0.35h）→ 约 **4.2 A100·h** |
| 正式研究训练条数 | ≤ 23 | **20 条**（P1=11, P2=5, P3=4；P4 三条未购买；其余为诊断） |

实际完成的 20 条主训练（arm 名 = `family_budget_dataset_recipe`）：

- **P1 核心（Lipo 满 Train）**：`LS/F/S/LP/FP _N100_lipo_R`
- **P1 wd0 对**：`LS/FS _N100_lipo_W0`
- **P1 Rwd 分支**：`B/LS/FS _N100_lipo_Rwd` + 补 `LP/FP _N100_lipo_Rwd`（5 条）
- **P2 跨任务（ESOL）**：`B/LS/FS/LP/FP _N100_esol_R`
- **P3 标签预算**：`LS/FS _N25_lipo_R`、`LS/FS _N50_lipo_R`

未购买：P4_*_HIV（证据开关 E 不成立），以及 PCBA 三臂。

诊断（不计入 23）：preflight（CPU/GPU）、selfcheck、GPU smoke（2ep）、FS 成本探测（实为 360ep，已作为 FS_R 真实结果回收）、机制/support/kernel/edit 分析、廉价显式基线（CPU）。

---

## 2. 方法本体（本轮唯一改变）

方法主体仍是 LCRC（同一批显式局部对象、原子/键/路径统计、LocalMLP、原读出），**无消息传递/attention/graph head/D-MPNN/新 loss**。

唯一改动：把原六个自由表（`path_emb,path_u` 400×32；`nbr_emb,end_emb,path_v` 80×32；`path_gate` 400×1，共 33,680 参数）由两个共享成分表生成——
`A:Embedding(16,16)`、`B:Embedding(5,8)`（296 参数），经 `u(p)=[A(mid),B(b1),B(b2)]`（32维）、`v(e)=[A(atom),B(bond)]`（24维）驱动 2 个路径 MLP(32→197→32)、3 个二元 MLP(24→40→32)、gate `1+MLP(32→23→1)`（末层清零→初值恒 1）。生成器 33,689 参数，整模型 **FS=228,683**（比自由查表 C1 的 228,674 多 9）。codec 按原口径精确解码：`p//25, (p%25)//5, p%5`；`e//5, e%5`；`path_end63` 实际为 80 类。

**四臂成对设计**（在 `analysis/preflight.json`、selfcheck 中已验证）：
- LS：自由表初始 = FS 的 θ0 物化表，之后自由训练；
- FS：语义生成器，真实 mid/atom/bond 组成解码；
- LP：自由表初始 = FP 的 θ0 物化表；
- FP：同 FS 生成器与 θ0，但对复合 code 先做固定双射（`π400/σ80`, seed 20261114）再解码。

epoch0 预测在每对内精确相等（CPU atol 1e-6 / GPU max|Δ|≤4.29e-06），gate 初值 1，五个向量 bank 用固定 affine buffer 标准化到 C1 初表逐列 μ/σ（常数，不拟合样本）。

---

## 3. 主表：四臂 + 各配方/预算/跨任务

选点口径：**主口径 = Validation-only 的 50 点共同 grid（`round(j·U/50)`）**；辅助 = best-Val epoch。下表为 grid 选点（Val 最小）的 Val/Test RMSE。

### 3.1 Lipo N100, R（核心）

| arm | sel j | Valid RMSE | Test RMSE |
|---|---|---|---|
| LS | 31 | **0.7115** | 0.7202 |
| FS | 29 | 0.7357 | **0.7091** |
| LP | 23 | 0.7461 | 0.7040 |
| FP | 13 | 0.7423 | 0.7051 |

- Δ_sem(Test)=LS−FS=+0.0112；Δ_perm(Test)=LP−FP=−0.0011；**DiD_test=+0.0123**
- Δ_sem(Val)=−0.0242；Δ_perm(Val)=+0.0038；**DiD_val=−0.0280**

### 3.2 其它配方/预算的 FS vs LS（grid 选点，相对改善 %）

| 组 | Valid: LS→FS | FS相对 | Test: LS→FS | FS相对 |
|---|---|---|---|---|
| lipo W0 | 0.6987→0.7337 | **−5.01%** | 0.7108→0.6747 | **+5.09%** |
| lipo Rwd | 0.7046→0.7286 | **−3.40%** | 0.7151→0.6802 | **+4.88%** |
| lipo N50 | 0.7765→0.7788 | −0.30% | 0.8108→0.7858 | **+3.09%** |
| lipo N25 | 0.7936→0.8226 | **−3.65%** | 0.8679→0.8593 | +0.99% |
| esol R | 0.8892→0.8740 | **+1.71%** | 0.8851→0.8670 | **+2.04%** |

**读法**：Lipo 全部配方一致呈现"Valid 更差、Test 更好"；ESOL 是唯一两划分同向为正的数据集（但其 DiD 在 Test 上为负，见下）。

### 3.3 DiD_semantics（grid 选点 与 逐 grid 点均值两口径）

| 组 | 口径 | DiD_Valid | DiD_Test |
|---|---|---|---|
| lipo R | grid 选点 | −0.0280 | +0.0123 |
| lipo R | 逐点均值(50点) | −0.0221 | +0.0093 |
| lipo Rwd | grid 选点 | −0.0232 | +0.0129 |
| lipo Rwd | 逐点均值 | −0.0206 | +0.0232 |
| esol R | grid 选点 | +0.0021 | −0.0289 |
| esol R | 逐点均值 | −0.0025 | −0.0411 |

**没有一个组在 Valid 与 Test 上同时给出正的 DiD。**

### 3.4 不确定性（500 次 Murcko scaffold 簇 bootstrap，Exp(1) 共同权重，seed 20261114）

Lipo R（best-epoch 选点）：

| 对比 | Split | 点估计(RMSE 差) | 95% CI |
|---|---|---|---|
| FS vs LS | Valid | −0.0252 | [−0.0546, +0.0052] |
| FS vs LS | Test | +0.0117 | [−0.0232, +0.0467] |
| FP vs LP | Valid | −0.0027 | [−0.0399, +0.0296] |
| FP vs LP | Test | +0.0076 | [−0.0225, +0.0387] |
| **DiD** | Valid | −0.0225 | [−0.0593, +0.0133] |
| **DiD** | Test | +0.0041 | [−0.0380, +0.0523] |

**全部区间跨 0**；这只是固定训练状态下的样本扰动区间，不代表训练方差，按协议**不因跨 0 加 seed 或改阈值**。

---

## 4. 机制诊断：到底"共享了什么"

### 4.1 跨 code 梯度耦合核（零新训练，path_emb，23 个真实 Train df code）

固定 Rademacher 方向（seed 20261115），`K_ab = J_a·J_b / (|J_a||J_b|)`，按原 mid/b1/b2 的同位置成分重叠数分组（排除 a=b）：

| 重叠成分数 | FS 平均 K | FP 平均 K |
|---|---|---|
| 0 | 0.0818 | 0.1828 |
| 1 | **0.3159** | 0.2173 |
| 2 | **0.6316** | 0.2040 |

- FS：**真实成分重叠越多，梯度耦合越强**（0→0.082→0.316→0.632），说明共享是语义对齐的。
- FP：**三点近乎持平（~0.18–0.22）**，双射破坏了原子/键组成的真实对应，耦合被抹平。
- 这是"共享机制的实现与语义对齐"证据，**不是化学因果**，也不能仅凭非零核就断言预测收益。

### 4.2 真实分子编辑响应（复用 `lcrc_lipo_edit_supervision_public122_s0_v1/analysis/pairs.csv`）

canonical SMILES 一一映射到 OGB 当前行，823 个唯一无序对；按当前官方 split 分格，每格 <20 component 标 LOW_SUPPORT 只描述。component 等权 `MAE(|Δpred−Δy|)` 与符号一致率（|Δy|≥0.1）：

| 模型 | edit MAE | sign agreement | n |
|---|---|---|---|
| **FS** | **0.2926** | **0.9005** | 823 |
| LS | 0.3076 | 0.8831 | 823 |
| LP | 0.3339 | 0.8656 | 823 |
| FP | 0.3341 | 0.8790 | 823 |

FS 在真实编辑响应上最好，且优于其匹配查表 LS 与置换控制 FP。**注意**：这仍是相关/描述证据，不把分量变化解释为唯一官能团贡献。

### 4.3 code 支持度切片（退化，据协议"少箱则不分箱"）

Train-LOO 联合 code 的 df≤5 占比三分位为 `[0,0]`（几乎所有 Train 联合 code 都 df>5）；Val/Test 按该阈值分箱后 bins=`[410,0,10]`/`[404,0,16]`。**分箱退化，按预注册规则报告"不分箱"，不再另找阈值。** 逐 split token 加权未见比例见 `p0_audit.json`（path 未见 Test 0.45%，联合未见 0.91%）。

---

## 5. 廉价显式统计坐标（CPU，Lipo+ESOL）

袋级显式统计 X（global42 缩放 + nbr80 + path400 + end80 + closure2 + (path,end)32000 = 32,604 维稀疏），Train 拟合列 scale，Ridge(α=10) 与固定超参 XGBRegressor（无搜索/early stop）：

| 数据集 | Ridge Test RMSE | XGB Test RMSE | 本轮最佳 LCRC Test |
|---|---|---|---|
| Lipo | 1.0457 | 0.9869 | FS 0.7091 / LS 0.7202 |
| ESOL | 1.8274 | 1.5523 | FS 0.8670 |

廉价显式统计显著弱于 LCRC；但 XGB 的稀疏零走"缺失"路径并学得默认方向（已在 `explicit_baseline.json` 声明，不当作数据空缺）。**这不能解释成"所有局部表示无用"**，只说明该袋级统计不是更强的当前坐标。

---

## 6. 逐项回答

**Q1 上轮排除了什么、没识别什么？哪些旧文字要收窄？**
上轮机器表复算确认：B Test=0.7211、D2=0.7457、G=0.7023（`p0_audit.json` 三条与本轮一致）。上轮**没有**识别出"缺容量/归纳偏置"这一根因；固定来源干预未定位到 H2 特有的收益。需收窄的旧文字：(a) S 臂代码含 `bond_proj`，PROTOCOL "no bond term" 措辞不准，但不重训、不判整轮作废；(b) R2 的 Val=0.7419 差于 B=0.7371，旧报告"R2 的 Val 也改善"不成立；(c) H2 的 size-standardised gap(0.1159) 与未标准化 gap(~0.1170) 不可混用，须各自报支持数（H2 group 356 / other 64）。

**Q2 在 LCRC 本体内，共享训练是否超过匹配初值查表？是否需要真实组合语义？**
- 就 **Test** 而言，FS 在 Lipo 各配方都比匹配 LS 好（+1%~+5%），ESOL 上也好（+2.04%）。
- 就 **Valid**（预注册选点所依据的划分）而言，FS 在 Lipo **每个配方都更差**，仅 ESOL 更好。
- 置换控制 FP 在 Test 上也能拿到 ~+1% 量级、与 FS 相当的点 → Test 的小改善**可能是通用重参数化**而非真实组合语义。
- 结论：**不需要把收益归给"真实组合语义"**；语义对齐确实改变了梯度耦合与编辑响应（机制层），但**没有变成 Valid 上可选的性能进步**。

**Q3 收益来自初始化、参数空间正则、标签预算，还是更广的实际预测改善？**
- 初始化(LS vs 原 B)：LS 相对 B 确有改善（B_Rwd Test 0.7188、LS_Rwd 0.7151；B_R Test 0.7211、LS_R Test 0.7202），说明"把 FS 的 θ0 物化成初表"这一小步有效——这是 **INITIALIZATION** 信号。
- 训练共享(FS vs LS)：在 Valid 上为负，不支持。
- wd：W0 / Rwd 都保持"Valid 变差、Test 变好"的同一符号，非某一 wd 独有 → 不是纯 weight-decay 作用，也不是 wd 能救。
- 标签预算：N25/N50 与满 N 方向一致（Test 更好、Valid 更差），**无额外的小标签专属收益**。
- 综合：真正稳定的只有 Init/重参数化层面的小效应，**没有更广的实际预测进步**。

**Q4 改善是否出现在结构风险与真实编辑响应中？证据链哪层断开？**
- 编辑响应：FS 最好（0.293），证据链在**解释层**成立。
- 结构风险切片：Train 联合 code 支持度过高、分箱退化，**未能**在"低支持族"上定位到收益（证据链这段断开）。
- 断点在于：机制/编辑层支持 → **性能可分选层（Valid）为负** → 无法把机制优势转成可放大主张。

**Q5 与当前最佳 LCRC、廉价显式统计、已有 GNN 相比，保住了什么、还缺什么？**
- 相对最佳查表候选（LS_W0 Valid 0.6987 / Test 0.7108；LS_Rwd 0.7046/0.7151；B_Rwd 0.7130/0.7188）：FS 在 Valid 上都不如；Test 上 FS_W0(0.6747)/FS_Rwd(0.6802)/FS_R(0.7091) 均低于这些查表候选，但这是以牺牲 Valid 换来的（且最低 Test 的臂全是生成器）。
- 相对廉价显式统计：LCRC 明显更好。
- 相对已有 GNN（G Test 0.7023）：FS_W0 Test 0.6747 / FS_Rwd 0.6802 已低于 G，但同样以 Valid 更差为代价；G 是另一架构坐标，仅并列参考。
- 保住：LCRC 本体作为强显式基线，且语义共享不劣化 Test。
- 还缺：Valid 上真正超过最佳查表的证据、以及跨划分同向的 DiD。

**Q6 是否应继续语义共享方向、仅保留初始化/配方，还是关闭此假设？**
按预定规则：**关闭"语义共享带来可放大预测收益"这一具体假设**；保留可复用的两点认识——(1) FS 的 θ0 物化作为初始化有小而稳的 Test 收益（Init 信号）；(2) 语义对齐确实改变梯度耦合与编辑响应（机制结论，可作为解释证据，但非性能主张）。**不启动新的模块设计，不扫新架构。**

---

## 7. 用户最终短答

- **Wall / GPU**：约 **4.8h** 达到训练全部完成并锁定结果（其余用于分析/交付，未越过 8h 硬墙）；GPU 计费合计约 **4.2 A100·h**（软 14 / 硬 16 之内）。
- **完成清单**：P1 核心 4 + W0 对 2 + Rwd 5；P2 ESOL 5；P3 预算 4（N25/N50 的 LS/FS）——共 **20 条**主训练（在 ≤23 之内）；外加 preflight/selfcheck/smoke/成本/机制/support/kernel/edit/廉价基线诊断。
- **未购买清单**：P4（PCBA 三臂或 HIV 三臂）——证据开关 E 不成立；按要求不为凑预算另扫架构。
- **最好模型与比较口径**：按预注册 Validation-only 选点，Lipo 上跨所有臂**Valid 最优**是 **LS_W0（Valid 0.6987 / Test 0.7108）**；**Test 最低**的是 **FS_W0（0.6747）**、次为 FP_Rwd(0.6759)/FS_Rwd(0.6802)，但它们的 Valid（0.7337/0.7440/0.7286）均**劣于**最佳查表 LS_W0/Rwd。比较口径均为官方 scaffold Val/Test RMSE（raw logD），MAE 同报。
- **研究主张改变了什么**：证明了"语义对齐的参数共享"在**机制层**真实存在（耦合核随重叠分级、编辑响应最佳），但**没有**在预注册的 Valid 选点上变成可放大的性能进步；Test 上的小改善更像通用重参数化/初始化效应，不能被判给真实化学语义。因此**关闭该具体假设**，只保留 Init/机制两点认识。
- **仍是单 seed 探索的部分**：所有 arm 仅 seed=0；DiD 的置换控制仅一个双射(seed 20261114)；scaffold bootstrap 区间全部跨 0 属样本扰动、非训练方差。这些都不构成"已确认的性能主张"。

---

## 8. 产物索引

- 协议/配置：`protocols/lcrc-explicit-semantic-learning-8h-s0-v1.yaml`、`configs/`（manifest、run_matrix、rr 覆盖）
- 每 run 记录：`analysis/results/<arm>/{metrics.json,curve.csv,grid.json,pred_*.npz}`
- 主表/配对/DiD/标签：`analysis/tables_all_arms.json`、`tables_comparisons.json`、`tables_did.json`、`tables_did_grid.json`、`tables_bootstrap_R.json`、`tables_judgements.json`、`final_evidence.json`
- 支持度/梯度核/编辑：`analysis/mech_support_fs.json`、`mech_kernel_fs.json`、`mech_kernel_fp.json`、`mech_edit.json`
- 廉价显式基线：`analysis/explicit_baseline.json`
- 阶段/队列/test-access：`analysis/stage_decisions.json`、`analysis/dispatch_state.json`、`analysis/test_access_log.json`
- 图（5 张，`analysis/figures/`）：`fig1_curves.png`、`fig2_paired_did.png`、`fig3_budget.png`、`fig4_support_kernel.png`、`fig5_cost_edit.png`
- 来源说明：`SOURCE_NOTES.md`
