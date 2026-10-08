# zinc_cssd_graph_information_sufficiency_v1 — 冻结 CSSD 消费者的图级聚合信息充分性诊断

状态：预注册（先于任何带标签评分冻结；witness/`R_G` 导出与恢复检查是不带标签的）。
日期：2026-10-09。Track：ksvd；Study：zinc-context-gap。
代码：`tracks/ksvd/experiments/luyin16/zinc_cssd_graph_information_sufficiency_v1.py`。
Runner：`zinc_cssd_graph_information_sufficiency_v1`（非 terminal，official valid/test 永不加载）。

## 1. 科学问题（本轮唯一问题）

在冻结的 CSSD 字典与下游消费者（`zinc_cssd_consumer_replacement_v1` 的 DICT_s0/DICT_s1）
已经固定的前提下，**图级聚合 `R_G` 是否丢失了对 `g` 有用的、由真实图结构决定的局部字典
组合信息**？

具体被测对象：CSSD 字典原子支持（alpha 的 top-8 二值支持 `b[v,k]`）与**真实物理原子的
patch 共覆盖**之间的三方绑定（字典结构原子 — 物理图节点 — 多个重叠 patch）。它只针对
任务brief中的第 3 点（局部环境与关系对象被压缩成图级统计的过程），同时以容量对照、
伪见证对照和两种子复核控制另外三点（phi65 表达能力、CSSD 稀疏表示本身、预测头/训练
过程）作为竞争解释。

## 2. 当前 CSSD 的实际信息流（用真实前向代码核对，2026-10-09）

```
原始分子图 G（原子/键，ZINC 化学）
  │  每个 root 一条真实 radius-2 邻域；节点 ID 以 env_occ_{node,root,shell} 保留
  ▼
phi65（每 root 纯拓扑显式坐标，65D；无化学属性；fsar_r2_ar0.build_phi）
  │  root_codes 读 data.dict_phi（原始 phi 坐标）
  ▼
CSSD 解码（LocalTupleEncoderMCSSD.cssd_decode，冻结 buffer，不可训练）
    c     = phi @ U                     (U 65×1, q=1)
    r     = phi − c @ Uᵀ
    alpha = tied_iht_codes(Dbar, r, s=8, steps=10)   ← 32D、恰 8-sparse
    phi_hat = c @ Uᵀ + alpha @ Dbarᵀ
  │  ⚠ alpha 在此处被丢弃：消费者只见 phi_hat（重构）。
  │  ⚠ DeployFull.code() 返回零占位；forward 的 aux["coord"] 不是 CSSD alpha。
  ▼
化学元组（prev.tuple_features）：x(v,a,t) = [标准化的 phi_hat(65) ; onehot28(root 原子)
    ; onehot28(邻居原子) ; onehot4(键型)] = 125D；权重 w_J = 真实键 incidence C/d
  │  真实化学在此进入；每个 root 内是元组多重集（root 内顺序被丢弃）
  ▼
M 编码器 e(v) = Σ_tuples w_J · SiLU(x @ Āᵀ)（64D）→ W_loc 直注 fusion 第 1 层
    （kappa 缩放；W_loc 零初始化、训练后 ‖W_loc‖≈7.4 —— 稀疏残差的真实读出通道）
  │  与 Sem108(patch_cont 108D) + size2(anchor 2D) 在 fusion[0] 相加（110→342）
  ▼
E_v = MLPBridge(fusion(...))（144D 局部环境态；一次生成，之后不被 pair 更新）
  ├─→ unary = pool_moments(E)（289 = 144+144+1；C6_MASK 置零 count 块）
  ├─→ pair 分支：u = pair_projection(E)（144→48，所有 pair 共享一个投影）
  │      q_uv = pair_encoder([left+right ; |left−right| ; (left⊙right)·gate ; relation])
  │      relation = relation_encoder(pair_relation 子集)（静态关系，无回写）
  │      relation_readout = pool_pair_moments(q, batch[pair_index[0]], pair_bucket)
  │      （5 桶 ×(48+48+1)=485；C6_MASK 置零 count 块；pair 图分组正确）
  ├─→ global_encoder(global_context)（32；C6_MASK 置零 atom/bond 直方图组）
  └─→ topology_encoder(topology_features)（8；topology25→16→8）
  ▼
R_G = [unary 289 ; pair 485 ; global 32 ; topology 8] = **814D**（reader 输入）
  ▼
ComponentReader 814→39→39→(39→2)：(ĥ_ell, ĥ_s)，g_raw = ĥ_ell + ĥ_s = h
  ▼
y_raw = h + q_raw（q_raw = 冻结 Q(topology25) soup）；y_cal = y_raw + b_y
```

核实记录（本轮以代码与实例化模型再次验证）：

- `R_G` 实际宽度是 **814D**（E=144、PAIR_HIDDEN=48），不是历史 clarity audit 的
  302D（E=48、pair 16）；以实例化的 DICT_s0 模型 `reader.net[0]: Linear(814→39)` 为准。
- `DeployFull.code()` 返回 `(n, 33)` 零占位（65×1 common + 32 atoms 的形状）；前向返回的
  `aux["coord"]` 是该占位，**绝不能当作 CSSD alpha**。本轮的 alpha 全部从冻结基底
  （`cssd_basis.npz` 的 U/common_rms/D）+ `data.dict_phi` 用与部署算子逐式相同的方法
  （`zreuse._root_codes` 的数学）提取，并在导出阶段与部署路径 `cssd_decode` 数值核对。
- `semantic_interface` 不读 coord（只用 `data.patch_cont`/`anchor`），因此 phi65 进入
  消费者的唯一通道是 `root_codes` 内的 tuple 特征（经 phi_hat）。
- `pool_pair_moments_masked` 的图分组是正确的 `batch[pair_index[0]]`（历史 Q1 错误已修）。
- C6_MASK（`e2e_dictenv_clean_mechanism_v1.C6_MASK`，本轮全部前向使用）：
  `global_zero_groups=("atom_histogram","bond_histogram")`、`unary_zero_blocks=("count",)`、
  `pair_zero_blocks=("count",)`、`relation_zero_groups=("path_count",)`。

### 哪里可能丢信息（本诊断的靶点）

- 保留：真实节点 ID/patch incidence（env_occ_*，E 的构造直接按 root/shell 收集真实原子
  的码）；真实键 incidence（w_J 与 pair_relation）；字典信息在 `cssd_decode` 处解码、在
  `W_loc` 注入处融合；拓扑（phi65/topology25）与化学（one-hot28/28/4、Sem108）分离。
- 多重集统计：每个 root 的元组聚合（Σ w·SiLU(x Āᵀ)）、`R_G` 的 unary/pair 一二阶矩。
- **丢失候选**：`R_G` 只有 (i) root 级 E 的一/二阶矩，(ii) pair 级 q_uv 按距离桶的一/二阶
  矩，(iii) global/topology。任何"多个 root 的字典支持经由同一真实原子耦合"的三阶共覆盖
  统计都不在 `R_G` 中。`W_k(G) = Σ_x C(n[x,k],3) / (Σ_x C(|C(x)|,3)+ε)` 恰是这样的统计。

### 绕过 CSSD 的路径（如实列出）

化学 one-hot28/28/4（元组特征）、J incidence 权重、Sem108+size2、静态 relation、
global_context、topology25、reader、Q 全部不经过 CSSD；CSSD 只替换 tuple 特征里的 phi65
块。因此本轮任何正信号都**不能**宣称"字典独有价值"，只能宣称"当前消费者的 `R_G` 聚合
之外存在可利用的、与字典支持—真实 patch 关联有关的增量预测信号"。

## 3. 与历史阴性研究的区别（新颖性表；防重复实验）

| 历史实验（全部在旧 compact-v4-hinge 上） | 被测对象 | 与本轮 `W` 的实质区别 |
|---|---|---|
| frozen_conditional_readout_sufficiency（NO-GO） | 冻结 `{h'_i},{q_ij}` 的一般 set readout（学习型摘要） | `W` 不是学习型 set 摘要，而是无参数、按**真实原子共覆盖**归组的三阶共现计数；对象是 CSSD 字典支持，旧图里没有字典对象 |
| pair_endpoint_association（NO-GO） | 对称 pair 映射丢弃的跨坐标 endpoint 外积关联（pair 内） | `W` 是跨 **root**、以共享真实原子为锚的共覆盖；不是 pair 内任何坐标关联 |
| centre_incidence_cooccurrence（NO-GO） | 一个 centre 上入射 q 的跨**通道**协方差 | `W` 是同一**单个**通道（字典原子 k）在多个共享原子的 root 上的出现共现——"共享锚多重性"，不是同锚跨通道协方差 |
| triadic_relation_binding（NO-GO） | `(q_ij, q_ik, q_jk)` 三条 pairwise 关系的显式闭合绑定 | 绑定轴不同：旧实验绑定一个 patch 三元组的两两关系；`W` 绑定"同一物理原子被 ≥3 个 root 覆盖"与各 root 的字典支持。状态也不同（学习出的 16D pair 态 vs 冻结字典二值支持） |
| raw_graph_patch_sufficiency（未反驳） | G_raw→patch 系统的预神经边界 | 不同边界；本轮在 `R_G` 聚合处，不在预神经特征处 |

- `W` 不是当前 `R_G` 已包含的统计量（由构造：`R_G` 无任何跨 root 三阶统计）。
- 见证不依赖标签：alpha 只依赖冻结基底 + phi65；C(x) 只依赖 env_occ_*。 witnessed
  代码路径不 import targets（导出阶段结构上不加载 targets.npz）。
- 机制见证（sham）：分子内固定种子置换支持行、incidence 不动 → 保全部边缘
  （每分子每原子使用次数、|C(x)|、分母），只破坏"支持—原子覆盖"绑定。24 分子探针：
  平均 ‖W_real−W_shuf‖₁ ≈ 0.86，24/24 非零 → 机制活着（导出阶段在全部 10k 行复核）。
- 与旧三元实验的等价性风险：旧 triad 用的是学习出的 pair 态及其闭合；即便把 `W` 看成
  "某种三元统计"，它的输入（CSSD 支持与真实 incidence 的绑定）在旧表示与旧实验里都不
  存在，且旧结论不能自动搬到 CSSD（本轮明确验证而非假设）。
- 若正信号：支持"当前冻结 CSSD 消费者的图级统计之外，存在与真实字典—patch 关联有关的
  增量预测信号"（结论 A 的措辞）；**不**直接宣布字典组合方法成立。
- 若无正信号（机制见证有效的前提下）：足以停止本候选（结论 B），不救场、不加新统计量。

## 4. 冻结的见证定义（先于任何 g 评估）

对每分子（unbatched，root 行 = 原始原子行 0..n−1）：

- `alpha = tied_iht_codes(Dbar, phi − (phi@U)@Uᵀ, s=8, steps=10)`（部署算子的精确数学；
  冻结 `cssd_basis.npz`：U 65×1、Dbar 65×32）。`b[v,k] = 1[alpha[v,k] ≠ 0]`。
- `C(x) = {v : (v,x) ∈ env_occ}`（复用 env 缓存 `env_occ_node`/`env_occ_root`；每 (v,x)
  恰一条，root 自身 shell=0 包含在内）。
- `n[x,k] = Σ_{v∈C(x)} b[v,k]`；`W_k = Σ_x C(n[x,k],3) / (Σ_x C(|C(x)|,3) + 1e-9)`，
  `C(m,3)=m(m−1)(m−2)/6`。W ∈ R^32；分母为 0（无原子被 ≥3 root 覆盖）→ W=0 并计数。
- **sham**：`b_shuf[v,:] = b[π(v),:]`，`π = default_rng(20261021 + gid).permutation(n)`，
  gid = 该行全局 train 行号（local_mol_id）；固定点计数记录；W_shuf 同式。
- 记录每分子诊断：分母、被 ≥3 root 覆盖的原子数、固定点数、`‖W_real−W_shuf‖₁`。
- 不加原子/键/环/RDKit 特征；不更新隐状态；无注意力；纯冻结态只读数学。

**机制有效性门（无标签，先于评分检查）**：fit 行上 mean‖W_real−W_shuf‖₁ ≥ 0.1 且
分母>0 的分子中 ≥95% 距离非零；分母=0 比例记录。探针 0.86/24·24 预期通过；若不过 → STOP(C)。

## 5. 冻结的导出与评分协议（先于任何带标签评分）

### 恢复与导出检查（无标签）

1. 对象哈希：DICT_s0/s1 的 soup/init/last state、cssd_basis、Q_soup、fold/targets/payload/
   prep 对 `source_manifest.json` 逐项核对。
2. 预测一致性：恢复模型重放 fit+dev 的 h/ell_hat/s_hat 对已存 npz ≤1e-4（fp32 保存精度；
   GPU 不强求位级）。
3. 真 `R_G`：`forward_pre_hook` 挂在 `model.reader` 上捕获输入（814D）；绝不用
   `aux["coord"]`。
4. root 码对齐：env 缓存 phi 行 == `data.dict_phi` 行（全部 10k）；occ 索引范围、(v,x)
  唯一性；batch 化前后 alpha/W 一致；真实分子节点重标号（phi/atom/occ 同步重映射）不变；
  batch 顺序/拼接不变。
5. alpha 算子等价：`zreuse._root_codes` 的 phi_hat == 部署 `cssd_decode(phi)`（≤1e-6）。
6. 标签隔离：witness/R 导出阶段不读 targets；评分阶段才加载。
7. 来源披露：CSSD 基底历史 refit 含任务监督（非纯无监督）；本轮不宣称端到端无监督。

### 评分设计（冻结；标签只在此阶段读取）

冻结模型/基底/编码器/Q 不动。小容量残差头：`g_new = g_frozen + head(输入)`，
head = 2 隐层 MLP（13,13，ReLU），L1 损失，Adam lr 1e-3，full-batch，最多 1500 epochs，
按分组 selection 子集（train 组的 20%）早停（patience 300，最优 checkpoint）。
输入按 fold-train 行标准化（零方差列 scale=1）。初始化 seed 20261021（三臂同一构造顺序，
隐层初始化共享；第一层因输入宽度 814/846 不同而形状不同——如实披露）。
三臂（同优化器/步数/正则/训练行/选择协议，无任何宽度/lr/搜索）：

| 臂 | 输入 | 参数 | 作用 |
|---|---|---|---|
| R-only | R(814) | 10,791 | 容量参照 |
| R+W_shuf | [R;W_shuf](846) | 11,207 | 破坏绑定的伪见证（每分子每原子使用次数/|C(x)|/分母全保留） |
| R+W_real | [R;W_real](846) | 11,207 | 真实见证 |

- 参数差 +416 = W 块的第一层输入权重（32×13），属输入维度差而非隐层容量差；
  **决定性的 Δg_sham 对照两臂架构完全相同**（容量干净）。
- 主指标：**未校准 g_raw MAE**。fit 上 5 折分组 OOF（按 canonical SMILES 分组，
  `default_rng(20261021)` 对排序后的唯一 SMILES 组做固定划分；同分子重复形式不跨折）；
  Δg_R = MAE(R-only) − MAE(R+W_real)，Δg_sham = MAE(R+W_shuf) − MAE(R+W_real)。
  两个冻结 DICT seed 分开训练/报告（g_frozen 与 R 取各自 run）。
- 次指标（探索性）：dev 1999 行（全 fit 训练的头；历史开发比较集，非独立 confirm）；
  g_new = h_dev + head；同时报 y_raw = g_new + q_raw（Q 不动）。
- 分层：常见 k=0、k 组、n_nodes 三分位；top-20 分子对 Δ 的贡献份额（防单分子驱动）。
- 不确定性：canonical-SMILES 组配对 bootstrap 2000 次（seed 20261021，臂与 seed 共享组
  抽样；每次抽样的两 seed Δ 先平均再取 CI）。**训练 seed 不确定性不伪装成 bootstrap CI**；
  头为单 init seed。OOF 头 ≠ 整模型独立 OOF（backbone 本身在这些 fit 标签上训练过）。

### 决策规则（冻结；不得事后移动门槛）

- 依据：完整 240-epoch 训练轨迹噪声 ±0.001–0.003（历史多轮实测）；本设计为共享数据/
  协议的配对头比较（8001 行、架构相同），抽样噪声低一个量级（历史同类审计 2000 行
  CI ±0.002–0.003 → 8001 行更紧）。0.003 = 能改变全模型决策的量级；0.001 = 本设计能
  可靠分辨的下限。
- **GO(A)**：两 seed 方向一致且 seed 平均的 OOF `Δg_R ≥ +0.0010` 且 `Δg_sham ≥ +0.0005`
  且 Δg_sham 的组配对 bootstrap CI95 下界 > 0；机制门通过；top-20 分子贡献份额 < 50%。
  - 若 further `Δg_R ≥ +0.003`：决策尺度信号；0.001–0.003：真实但低于决策尺度，需要
    独立 backbone/split 确认协议才能升级。
- **NO-GO(B)**：不满足 GO 且机制门通过 → "本轮没有发现当前 CSSD 消费者遗漏该特定
  字典—patch 组合统计的有效证据"；停止本候选，不加新统计量。
- **STOP(C)**：机制门失败（W_real≈W_shuf）、见证与已有统计等价、恢复/一致性检查失败、
  评估来源不可核验 → 明确停止，不编造、不自动开始另一套实验。

### 预算

全部本地 CPU（导出 + 36 个小头训练）。**0 GPU**；不用 res-2（除非本地恢复失败——
所需 checkpoint 已确认本地在位）。不训练任何 backbone，不访问 official valid/test。

## 6. 范围与不做什么

- 不改冻结主模型的输入/`R_G`/标签/Q；sham 只存在于见证计算内部。
- 不做第二个见证量、不做宽度/阈值搜索、不为正/负结果补 seed。
- 不把本轮结果说成：字典独有价值、所有高阶关系无用、或聚合充分性的一般证明。
