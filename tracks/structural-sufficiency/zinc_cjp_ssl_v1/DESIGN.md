# DESIGN — CJP-SSL v1（Conditional Joint-Path Self-Supervision for LCRC-A）

**Round id:** `zinc-conditional-joint-path-ssl-v1`
**目录:** `tracks/structural-sufficiency/zinc_cjp_ssl_v1/`
**状态:** 实现/门禁在 GPU 之前冻结（`configs/cjp_ssl_v1.json`）

> 结果与判决见 `REPORT.md` 与 `NEXT_DECISION.md`。本文件保持冻结，未因结果修改。

本文件忠实记录任务书冻结的科学方案与实现层决定；不重写研究目标，不追加
teacher/student、contrastive loss、codebook、外部数据、额外图模块或注意力层。

---

## 1. 冻结命题

主命题：**在不改变 LCRC-A 推理函数的情况下，使用无标签的二跳条件预测预训练
LocalContent，能否提高有限性质标签下的泛化能力？**

次命题：**联合预测真实的终点原子类型与第二条物理键类型，是否比仅学习它们的
条件边际分布更有利于下游预测？**

参考基线（历史 LCRC-A，不重训也可核对）：seed0 valid `0.076581636`，
seed1 valid `0.080503784`，两 seed 均值 `0.078542710`；600 epoch / 47,400
updates；237,519 参数。历史 exposed test A_s0 `0.066782` 只作背景，本 round
**不读取 official test**（prep/训练/分析全部拒绝 test）。

## 2. 部署架构（完全冻结）

推理模型严格为原 LCRC-A（`lc_models.LCRC("A")`）：

- atom encoder → 96；bond encoder → 96；`LocalContent` → 64；
- S1/S2 embedding、gated endpoint content、局部 MLP、LayerNorm 不变；
- `f`、`rho` 不变；
- 不增加推理输入列、预测头或状态；部署参数严格 **237,519**（G12 检查）；
- SSL 专用预测头只存在于预训练代码，监督 checkpoint 不含它（G12 检查
  `pretext_head_absent_from_supervised_state=True`）；
- 未修改 `lc_models.py` / `lc_data.py` / `lc_train.py` / `lc_common.py`
  （SHA 记录在 config；G0 检查 `cjp_sup` 直接调用冻结的 `lc_train.train`）。

## 3. 预训练样本

### 3.1 数据与划分

- 只读官方 PyG ZINC `subset=True` **train** 的原始 `x` / `edge_index` /
  `edge_attr`；`y` 从不访问（G1 用标签投毒验证）。
- 结构按 typed Weisfeiler-Lehman 指纹（5 轮、128-bit）分组；组按
  `sha256(fingerprint)` 排序，按组整体填入 fit，剩余为 diag。重复结构不跨组；
  多成员组用 `networkx.is_isomorphic`（含原子/键标签）核验。
- 实际划分：10,000 → **9,500 fit / 500 diag**，fingerprint 10k 中 9,997 唯一，
  3 个重复组（6 个分子），全部同构；`fit_index_sha256=82490f79…`，
  `diag_index_sha256=695323da…`。
- 划分只用于自监督。性质监督仍用官方完整 train 10k；official valid 1k 只作
  开发评估/选 checkpoint；official test 从不构造。

### 3.2 路径选择（冻结算法）

对每个分子枚举全部有向真实二跳路径 `p=(v,w,u)`（`v≠w≠u`、`v≠u`、两条都是
真实物理键、不回溯、不创造虚构节点/键）。每次访问：

1. `numpy.random.SeedSequence([seed, molecule_id, visit_index, 20261029])`
   （`molecule_id` = 官方 train 原始行号；`visit_index` = 该分子在本 run 内第
   几次被访问的计数器）；
2. 打乱全部合法路径；
3. 按打乱顺序贪心选取三条顶点互不重叠的路径；
4. 最多 3 条，不足则用实际数量；
5. 无合法路径 → 该分子 pretext loss 跳过，不构造虚假目标。

mask RNG 与 batch RNG（`SeedSequence([seed, 20261030])`）严格分离。两个 arm
在同一 seed 下逐次访问的分子、visit_index、路径样本与数据顺序完全相同
（`same_samples_across_arms`：逐 batch mask SHA 聚合值必须相等）。

### 3.3 遮盖

对每条选中 `(v,w,u)`：原子 `v,w` 与第一键 `(v,w)` 保持可见；
`x[u] := 27`；物理键 `(w,u)` 的**两条** PyG 有向记录 `:= 0`；不删边、不改拓扑。
路径顶点互不重叠。目标在模型输入之外：

```
target = 3 * raw_atom_type(u) + (raw_bond_type(w,u) - 1) ∈ [0, 62]
```

原始语义为 21 个有效 atom categories × 3 个有效 physical bond categories，
**63 类**；28×4 编码槽不被当成 112 类（G6 词表检查）。

## 4. 遮盖后的 LCRC 特征（关键正确性）

顺序固定为：

1. 先改原始 `x/edge_attr`；
2. 用冻结 P3 `structural_relation_objects_v1.build_mol_objects` 对**遮盖后的
   原始图**重建 atom108、bond7、global33；
3. 由 SSL-only masked LCRC builder 重建 S1/S2/scalar 输入。

builder 显式携带遮盖后的 raw type 与 mask boolean（不使用
`argmax(atom_feat[:21])`，因为 27 不属于真实 21 类）。

- **S1**：只有邻居 atom type 与 bond type 都可见时，`(type(u), bond)` 才进入
  categorical embedding 聚合；含 27 或 0 的记录被省略。
- **S2**：只有 `type(w)`、`type(u)`、`bond1`、`bond2` 四项全部可见时，才产生
  189/63 类 typed-path 记录、closure 与 gated endpoint contribution；否则整条
  路径不进入 categorical 聚合。
- 纯拓扑（degree、最短路径距离、真实路径总数、closure 比例、关系对）仍按真实
  拓扑计算；categorical record 被省略不等于拓扑路径不存在。
- 所有 scalar 标准化沿用冻结 train-only mean/std（`analysis/norm.json`，
  fingerprint `8c3e8813…`），从不重新拟合。
- 无遮盖时，SSL-only builder 必须与冻结 LCRC train cache 逐字节一致（G2）。

## 5. 两个 SSL arm

共同初始化：`LCRC("A")` 按原 seed 构造 → 载入历史 atom/bond SSL checkpoint
（`lc_models.ssl_path(seed)`，SHA s0 `006fabc1…`、s1 `ff8d6474…`）。
`atom_encoder`、`bond_encoder` 冻结；`LocalContent` 全部可训练；`f`、`rho`
不参加预训练；只额外加一个丢弃式预测头。两 arm 的模型+头初始 hash 必须相同
（G9）。

预测输入：`h_v = LocalContent(masked graph, v)`，
`h_w = LocalContent(masked graph, w)`，`q = concat(h_v, h_w, h_v*h_w)`（192 维）。
同一个头：`Linear(192,128) → SiLU → Linear(128,63)`。logits 视为 `[21,3]`。

- **Arm MARG**：`L_marg[a,b] = mean_b L[a,b] + mean_a L[a,b] - mean_ab L[a,b]`，
  对 `flatten(L_marg)` 做普通 63-way CE。目标函数严格限制为两个条件边际之和。
- **Arm JOINT**：直接对 `flatten(L)` 做同样的 63-way CE。

不添加任何辅助 atom CE / bond CE / 对比损失 / teacher / EMA / 自蒸馏。

**Loss**：一个分子内先对其选中路径取平均 CE；再对 batch 中有有效路径的分子
取平均（不按路径数给大分子加权）。两个 arm 使用完全相同的样本、相同的
63-way CE。

结构性数学单元测试（G9）：
`softmax(L_marg) == softmax(row_mean) ⊗ softmax(col_mean)`（误差 <1e-6）；
MARG 投影的非加性残差在浮点容差内为 0；在真实 21×3 空间内嵌入 2×2 XOR
交互反例（`[[2,-2],[-2,2]]`），其 MARG 投影恒为 0、JOINT 残差 max=2.0、
总变差 0.168；两 arm 全参数初始 hash 相同。

## 6. 预训练配方（冻结）

每 seed：fit 9,500 / diag 500（仅结构）；AdamW，constant LR `1e-3`，
WD `1e-5`，batch **128 molecules**，grad clip 5，**7,560 updates**，dropout 0，
FP32，单模型。导出**最后一步** LocalContent 权重，不通过诊断集选 checkpoint。

诊断（训练前/后）：63-class CE、类先验 CE、以可见 `(type(w), type(e1))` 分组的
条件频率先验 CE（fit-only Laplace +1）、held-out-structure CE（500 diag）、
selected path 数与 atom/bond mask coverage、LocalContent 各模块梯度、输入交换
（`h_v↔h_w`）与跨 molecule shuffle 对 pretext loss 的影响、JOINT logits 的
不可加性交互 RMS、LocalContent effective rank / activation RMS。全部是诊断量。

## 7. CPU 门禁（`analysis/cpu_checks.json`）

`cjp_gates.py` 实现 G0–G12，全部 CPU；任何 GPU 任务之前必须全 PASS：

| gate | 内容 |
|---|---|
| G0 | `cjp_sup` 直接调用冻结 `lc_train.train`，不重写训练循环；recipe 相同 |
| G1 | SSL 代码无 label 路径（静态扫描）；标签投毒后 input/loss/grad 完全不变 |
| G2 | 关闭 masking：SSL-only builder 与冻结 LCRC train cache 逐字节一致 |
| G3 | masked raw graph → P3 rebuild 与独立参考 compose 一致；差异只在 dependency map；mask token 占据 own slot；隐藏真值不在 own slot |
| G4 | 隐藏真值交换 210(atom)+210(bond) 次：masked model input 逐位不变、target 改变、mol 张量不变 |
| G5 | 每个被遮物理键的两条有向记录同时为 0；`edge_index` 从不修改；改动数 = 2×路径数 |
| G6 | S1/S2 无 mask token 当合法 type；所有 categorical index 在既有词表内；count 与 visibility 规则一致 |
| G7 | 无遮盖时模型输出与原 LCRC forward 逐位相同（max abs diff = 0） |
| G8 | 原子共同重编号 equivariance、键端点反向、batch 拆分合并不变性 |
| G9 | MARG 概率严格因子化；MARG/JOINT 初始 hash 一致；2×2 交互反例 |
| G10 | 只有 LocalContent + pretext head 收到 SSL gradient；`f/rho`、atom/bond encoder 无梯度 |
| G11 | `path_emb/path_u/path_v/path_gate/mlp/norm` 等可达模块梯度非零且有限；逐行检查被访问的 embedding 行 |
| G12 | 保存/重载/导出与部署结构兼容；参数 237,519；监督 state 不含 pretext head；四 arm 的模块共享关系正确 |

失败只允许修正确性 bug，不得改变科研定义。

## 8. 四个正式监督 arm（统一性质训练）

每个 seed 都从同一标准 `LCRC("A")` 构造过程开始（`torch.manual_seed(seed)`）。

| arm | atom/bond SSL | LocalContent 初值 |
|---|---|---|
| **A-REPLAY** | 载入历史 atom/bond SSL | 同 seed 标准初始化（同步基线） |
| **SCRATCH** | 不载入 | 与 A-REPLAY 逐张量相同 |
| **MARG-SSL** | 载入 | MARG 预训练 state |
| **JOINT-SSL** | 载入 | JOINT 预训练 state |

`f/rho` 在四 arm 中逐张量相同（G12）。统一训练：官方 train10k 监督 raw-y L1，
官方 valid1k 每 epoch 全量评估并选 full valid 最低 MAE 单 checkpoint；batch128，
AdamW，peak LR 1e-3、min LR 1e-5，warmup 500，per-step cosine，WD 1e-5，
clip 5，600 epoch / 47,400 updates，全部模块可微调；无 soup/EMA/ensemble/额外
loss。实现方式是给冻结的 `lc_train.train` 打一个**仅替换初始化**的补丁
（`lc_models.build_model` 在调用期间被替换，返回后恢复），因此 batch 顺序、
参数名、初值、数据归一化、LR 轨迹、checkpoint 语义逐字节沿用冻结实现。

## 9. 调度与预算

`res` 的两张 A100（`res-gpu0`/`res-gpu1`），经 `rr` 部署/运行。墙钟硬上限 8 h，
GPU 硬上限 16 GPU·h（软上限 10 GPU·h）。

- **Phase A** 双卡并行：A-REPLAY s0 / s1，核对 0.076582 / 0.080504。
- **Phase B** 双卡并行：MARG/JOINT s0 预训练 → MARG/JOINT s0 监督。
- **Phase C** 双卡并行：MARG/JOINT s1 预训练 → s1 监督。
- **Phase D** 双卡并行：SCRATCH s0 / s1。

Phase B 之后的 seed1 gate：若两 arm 都比同步 A_s0 差 ≥+0.010 full MAE，且
去固定极端行后仍 ≥+0.005、ordinary/unseen 均更差，则停 seed1；否则买 seed1。
预算不足时 Phase D 最后做，绝不使用剩余预算开新目标/teacher/结构搜索。

## 10. 性能分析（预先固定）

Primary：official valid 1,000 raw-y MAE。主比较：

1. `A-REPLAY − SCRATCH`（旧 encoder SSL 的效应）
2. `MARG − A-REPLAY`（预训练 LocalContent 的效应）
3. `JOINT − MARG`（联合 vs 因子化条件目标）
4. `JOINT − A-REPLAY`（最终净收益）

所有 paired contrasts 同 seed、同 valid 分子。报告：full MAE；valid:0172 removed；
历史 A 固定 worst-10 removed；ordinary `y>-3`；tail `y<=-3`；
r2-unseen ordinary；seen ordinary；wins fraction；median absolute-error delta；
paired row-bootstrap CI95（10,000）；best epoch、W20、T100、final；
eval-mode train MAE @ best；实际参数量；pretext NLL/conditional prior；
SSL 前后 LocalContent effective rank / activation RMS；每 seed 哈希与 GPU 计时。

## 11. 判决规则（冻结）

- **A `LOCAL_SSL_SUPPORTED`**：MARG 或 JOINT 相对同轮 A-REPLAY，两 seed 均改善，
  seed-mean full MAE 至少改善约 0.003，ordinary 与去 row172 也改善，
  r2-unseen ordinary 不明显反向恶化。
- **B `JOINT_BINDING_SUPPORTED`**：在 A 的绝对收益基础上，JOINT 比 MARG
  seed-mean 至少改善约 0.002、两 seed 同向、ordinary/unseen 不反向恶化。
- **C `MARG_ONLY_GAIN`**：MARG 优于 A，但 JOINT 不优于 MARG。
- **D `NO_GAIN`**：MARG、JOINT 都未达到实用收益 → 停止当前 Conditional Path
  SSL 方案（只否定本任务书的固定路径/目标/初始化协议）。
- 若 JOINT pretext CE 更好但下游 MAE 无改善 → 只能报
  `JOINT_PRETEXT_LEARNED_NO_DOWNSTREAM_GAIN`。

## 12. 交付

`code/`（masked builder、mask sampler、projection head、pretrain、supervised
integration、analysis）、`tests/`（G0–G12）、`configs/cjp_ssl_v1.json`、
`analysis/pretext_summary.json`、`analysis/supervised_summary.json`、
`analysis/paired_comparisons.json`、`REPORT.md`、`NEXT_DECISION.md`、
checkpoint SHA、源代码 commit、完整 GPU·h 账本。

## 13. 冻结产物（GPU 前）

- `analysis/cpu_checks.json`：G0–G12 全 PASS。
- `configs/cjp_ssl_v1.json`：上游 SHA（lc_models/lc_data/lc_train/lc_common、
  LCRC cache、norm、两个 SSL checkpoint）、划分 SHA、全部配方、四 arm 定义、
  预算与判决策略。
- 关键 SHA：fit `82490f7944d5ef7c2b5ebd9a563f8fd1ec92aebc284e5bef7bd9bba32db8fbef`，
  diag `695323dafb81e763928aba2966e97e5f1c7698f6c80f6a2b16a02144ecbb402f`，
  fingerprint order `ca90733c43699291ac06b1ab8a0dd0b974a14dd2b81bed9617e87114dd1c812c`。
