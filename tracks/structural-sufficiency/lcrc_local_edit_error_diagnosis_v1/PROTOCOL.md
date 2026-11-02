# PROTOCOL — LCRC 局部编辑响应误差诊断（冻结执行协议）

Track: `tracks/structural-sufficiency/lcrc_local_edit_error_diagnosis_v1/`
Round ID / protocol ID: `lcrc-local-edit-error-diagnosis-v1`
冻结时间：2026-11-02（Asia/Shanghai）
参照 checkout：`e080b94b60e5b5536e338faae34a144d0147eac2`（本轮未回退任何用户改动）

本文件是本轮的**冻结规则集**。任何科学定义（配对规则、划分算法、标签权限、
探针定义、阈值、预算、报告规则）以本文件 + `code/edit_common.py` 的常量为准。
本文件在读取任何 Diagnose/Audit 标签之前写出；之后只允许追加"实现映射"和
"事后记录"两节，不得修改已冻结的规则。

---

## 0. 研究问题与判定（摘要）

对两个只差一次小片段替换的真实分子（A→B，方向见 §2.6）：

```
d_pred  = pred(B) - pred(A)          (原始 logD)
d_y     = y(B)    - y(A)
e_delta = d_pred  - d_y              ← 本轮的可检验目标
```

* **H1（次要）**：某些片段变换的 `e_delta` 在不同核心结构上重复出现。
* **H2（主要）**：显式回溯给出的分支响应信息（`dL_var / dL_core / dR`）比
  变换类别先验和同容量全图响应参照，更能预测**新核心结构**上的 `e_delta`。

`e_delta` 不是原子/路径的真实物理贡献，也不是化学因果效应；`e_delta` 是当前
函数的**局部编辑响应误差**。所有分量都是"当前函数的计算分解"，不是观测误差的
唯一分配。

判定只用 §7 的预注册资源决策规则。**不追求官方榜单成绩**，本轮不上报任何
"性能提升"结论。

---

## 1. 数据范围与标签权限

| 项 | 值 |
|---|---|
| 数据 | ChemBench `Lipop`（`Lipophilicity.csv`），SHA256 `34280f0a…cd758` |
| 划分文件 | `Lipop.random122.ind.pkl`，SHA256 `bf7acb17…a7d` |
| 本轮母集 | **仅** random122 的 **Train 3360 行**；行序列 SHA256 `4e8d42a9…af96` |
| 结构来源 | `raw_attributes_v1.json.gz`，SHA256 `1127607d…51be` |
| 受保护标签 | random122 的 Valid/Test 标签、旧 Reserved 标签 —— **不读、不用、不评估** |
| 标签作用域 | `dissect_source.load_lipop(122, labels="train_valid")` 之后再取 Train 段，Test 目标字节从不解码 |
| 审计性质 | `WITHIN_DATASET_PROSPECTIVE_AUDIT`（不是全新独立盲测） |

**标签权限（代码强制，非约定）**：`code/edit_data.py::labels(partition, purpose)`

| purpose | 允许读标签的分区 |
|---|---|
| `train` | fit |
| `select` | fit, select |
| `probe_fit` | fit, select, diagnose |
| `audit_score` | fit, select, diagnose, audit |

额外规则：
* Audit 标签只有在 `analysis/probe_freeze.json` 存在后才可读；
* Audit 标签**只允许读一次**（第二次直接抛错）；
* 每次读取都追加到 `analysis/label_access_ledger.json`（含 UTC 时间、行哈希、
  commit、`test_targets_decoded=false`）。
* 归一化（`g`/`atom_scalar` 的 mean/scale、目标 mean/std）**只用 Fit**：
  `code/edit_data.py::train_norm/target_norm`。

---

## 2. P0 — 真实分子对构造（先不训练）

### 2.1 分子级资格（结构判定，不看标签）

单连通分子；无显式氢原子；canonical SMILES 不含 `@` `/` `\`（排除立体化学）；
全部原子为重原子。

### 2.2 切割规则

只切断**非环单键**。用 RDKit `FragmentOnBonds(bond, addDummies=True)`，dummy 取代
被切键，因此连接原子的价态与其在母体中一致。两侧分别鉴定为：

* **变量片段**：1–4 个重原子；
* **共同核心**：≥10 个重原子，且 ≥ 母体重原子数的 75%。

（两侧同时满足这两组约束不可能，方向唯一。）

### 2.3 保留的属性

节点标签 `(元素, 是否芳香, 形式电荷, 总氢数, 是否连接点)`；边标签
`SINGLE/DOUBLE/TRIPLE/AROMATIC/OTHER`。氢数在"切断后加 dummy"的价态下取，因此
A、B 两侧在连接原子上可比。

### 2.4 核心同一性

1. 用带属性图上的 Weisfeiler–Lehman 迭代细化签名做**候选分桶**（`edit_graph.py::wl_signatures`）；
2. 桶内再用**显式带标号图同构**（带回溯剪枝，含连接点标记）判定核心同一性。

WL 只用于生成候选；同一性由同构判定。本轮统计：8 027 个 WL 桶 / 9 482 条切割，
精确同构检查 1 455 次，`n_classes_with_multiple_canonical_strings=0`（未观察到
WL 假候选），与设计阶段预筛（610 次候选检查、无假候选）一致。

### 2.5 一对多定义消解

同一分子对可能由多个切割产生。**每个分子对只保留一个定义**，选择序：
`(core_heavy 降序, 核心 canonical 字符串升序, 变量片段 A 字符串, 变量片段 B 字符串)`。
本轮 610 个候选定义 → 509 个唯一定义（49 次被更优定义取代，
`superseded_definition=49`）。

### 2.6 方向

按**带连接点的规范化变量片段字符串**（RDKit canonical SMILES，dummy 同位素归一）
排序，较小者为 A、较大者为 B。变量片段同构、或两分子 canonical 结构相同的情形
不作为结构变换（`same_variable_fragment_skipped`）。反向查询必须严格变号
（`edit_trace.py::verify_all`，全 509 对）。

### 2.7 不变量与已完成的复核

* 每条切割都验证 `core_atoms ∪ var_atoms = {0..n_atoms-1}`；
* 509 对全部核心映射都是**保标号同构**且把连接点送到连接点（`gate_core_mapping`）；
* 核心自同构群非平凡的对有 442/509 —— 故逐原子展示只在映射唯一或按自同构轨道
  合并后进行，聚合量 `dL_core` 不受映射选择影响；
* 原子重编号下核心类别与连接点不变量（`gate_renumbering_structure`）。

### 2.8 与设计期预筛的差异（诚实记录）

设计期预筛报告 501 对 / 502 参与分子 / 228 类变换 / 110 对同重原子数。
本轮从原始结构独立重算得到 **509 对 / 502 参与分子 / 211 类变换 / 113 对同重原子数**。
差异来源：本轮用 RDKit 2026.09.1 重新枚举并自行规范化字符串（预筛未提交代码），
且"每对保留一个定义"的消解序与预筛的贪心可能不同。**分子数 502 完全一致**，
说明配对集合高度重合；数量差异未通过放宽任何冻结条件来弥补。

---

## 3. 分组划分（在读任何本轮标签之前冻结）

**不可拆分单位** = 二部图 `分子 —配对— 分子`、`分子 —共用匹配核心— 核心同构类`、
`分子 —相同完整分子— 分子` 的连通分组。因此：

* 任何配对都不可能跨划分；
* 任何共用**同一实际匹配核心**的配对必然同组（本轮额外检查并合并）；
* 相同分子/重复行同组。

**分配算法**（`code/edit_split.py::allocate_split`，只用结构与盐）：

1. 目标：`fit/select/diagnose/audit = 0.60/0.10/0.15/0.15 × 3360`；
2. 单位排序：`(size 降序, sha256("20261102|unit_id") 升序)`；
3. 先放**含配对的单位**，再用**无配对的单位**补足；
4. 每个单位放入当前**相对缺口** `(target_p - filled_p)/target_p` 最大的分区，
   平局按 `fit > select > diagnose > audit`。

盐 = `20261102`。该算法在读取任何标签前实现并冻结。

**结果**（`analysis/component_split.json`，逐分区行 id 与哈希）：

| 分区 | 行 | 配对 | 参与分子 | 变换类 | 有配对组件 | 孤立配对 | 贪心不相交配对 |
|---|---|---|---|---|---|---|---|
| fit | 2016 | 301 | 300 | 144 | 176 | 63 | 134 |
| select | 336 | 50 | 51 | 40 | 18 | 10 | 22 |
| diagnose | 504 | 84 | 76 | 68 | 22 | 15 | 34 |
| audit | 504 | 74 | 75 | 61 | 26 | 15 | 32 |

* 门槛（Diagnose 与 Audit 分别）：≥40 对、≥15 个有配对组件、≥60 个不同分子。
  **全部通过** → `status = OK`，允许 GPU 训练。
* 变换类别覆盖：Diagnose 68 类、Audit 61 类、**共享 19 类**（其余为 unseen，
  在报告里分层给出，不重拟合）。
* **Murcko scaffold 重叠**（报告，不作为划分单位）：diagnose|audit Jaccard 0.084，
  其余 0.059–0.095。本划分是**核心/配对组件不重叠**，**不等价于 Murcko
  scaffold-disjoint**，报告不得换名。

---

## 4. 预测模型与 GPU 执行（恰好 6 个训练单元）

| 单元 | 模型 | 参数 | 角色 |
|---|---|---|---|
| `C1:s{0,1,2}` | C1 = `NO_BOND_MEAN` | 228 674 | **主结论** |
| `A0:s{0,1,2}` | FULL | 247 106 | 仅参照（不替换主模型结论） |

* 数据：`fit` 训练，`select` 选 checkpoint。Diagnose/Audit 从不参与 encoder 训练
  或 epoch 选择。
* 配方 R3（与上游冻结配方逐项一致）：AdamW，LR 3e-4→1e-5 **per-step** cosine，
  betas (0.9,0.999)，eps 1e-8，wd 1e-3，batch 128，dropout 0.10（作用在共享 `h` 上，
  每 forward 一次），grad clip 5，**360 完整 epochs**，无早停、无 EMA/SWA/soup/SSL、
  无超参搜索。2016 行 → 16 steps/epoch → **5 760 updates**。
* 选择规则：**Select RMSE（原始单位）严格更小**，平局取更早 epoch。
* 同 seed 的 C1/FULL 共享同一 FULL 初始 state（CPU `torch.manual_seed(seed)` 构造后
  按固定列删除映射），batch 顺序相同；`shared_init_bitwise_equal_to_full` 逐 run 记录。
* 输出：`runs/edit_test_s<seed>_<model>/{train.json,curve.csv,checkpoint.pt,
  fit_predictions.npz,select_predictions.npz}`；checkpoint 与曲线 sha256 写入
  `analysis/checkpoint_manifest.json`。
* **不**复用旧 public122 checkpoint 评估本轮 holdout（它训练过本轮分区中的分子）；
  旧 checkpoint 仅用于 §5 的实现等价性复播。
* 预算：**硬上限 4.0 A100 GPU·h**（含 smoke、推理、失败），常规预计 0.5–1.5。
  一次同配置基础设施重试允许；超预算时先缩推理批次/转 CPU，**不改配方、不增 seed**。
  6 个有效单元完成后**不追加** seed、不做结构/超参搜索。
* 执行：`rr doctor/deploy/run/jobs/status/logs/pull`；`res` 上 `res-gpu0`/`res-gpu1`
  各至多 1 个训练进程，总并发 ≤2。wave 计划（`configs/run_matrix.json`）：
  w1 `gpu0 C1:s0 / gpu1 A0:s0`，w2 `gpu0 A0:s1 / gpu1 C1:s1`，w3 `gpu0 C1:s2 / gpu1 A0:s2`。

---

## 5. P1 — 每对的完整统计回溯

对 6 个模型/seed 分别完整编码 A、B（**两端都重新计算完整结构统计**），在标准化单位下

```
L      = sum_v f(h_v)                 y_hat = L + rho        pred = mu + sigma*(L+rho)
dL_var  = sum_{B var} f(h_v)  - sum_{A var} f(h_v)
dL_core = sum_{B core} f(h_v) - sum_{A core} f(h_v)
dR      = rho(B) - rho(A)
d_pred  = sigma * (dL_var + dL_core + dR)
```

* `mu` 在差分中抵消；**不得**重复加一次 `sum f`，**不得**混用标准化与原始单位。
* 恒等式残差阈值：FP32 主判据 `1e-5` 原始 logD（绝对），相对 `1e-5`。
* `dL_core` 是匹配核心上的**求和**，不依赖任取的同构映射；逐原子展示只在映射唯一
  时进行，否则按自同构轨道合并并显式报告映射歧义。
* 核心原子签名必须覆盖 forward 的**真实输入**：`atom_feat`、标准化 `atom_scalar`、
  入射键 `(bond_feat, 端点 atom_feat)`、一跳 `nbr_code`、逐根二跳记录
  `(path_cat, path_end63, closure, 端点 atom_feat)`。在这些签名相同且映射明确的核心
  原子上，`h` 与 `f(h)` 必须不变。
* 对签名改变的核心根，记录具名统计变化（原子类型/原子标量分量/入射键多重集/一跳
  码/二跳记录增删）、`J/C/T/E/K` 通道变化（`Δ` 的 L1/L∞）、`h`/`f` 变化与全图 `rho`
  变化；`g` 的变化同样由两个合法分子重算后报告。
* 规约性（gauge）：至少展示一个**不改变任何预测**的坐标变换
  `c'_v = c_v + k, rho' = rho - k·n_atoms`（原始单位 `k = 0.5 logD`），并显示
  `dL_var` 与 `dR` 会随 `k·(n_var_B - n_var_A)` 移动。主结果始终以 `d_pred/e_delta`
  为锚。
* 必须核验**所有**配对的正反向变号、原子重编号不变性、batch 拆分/合并一致性
  （不能只核验展示案例）。

---

## 6. P2 — Diagnose 上的有限、预定义诊断探针

目标：预测 `e_delta`（原始 logD）。每个模型/seed **分别**拟合；禁止用 ensemble mean
提升主成绩。所有方法共享**同一个**变换类别先验项 `b_t`（方向固定；反向时符号翻转；
Audit 未见类别的 `b_t=0`）。

| 方法 | 定义 | 连续系数数 |
|---|---|---|
| `NATIVE` | `e_hat = 0` | 0 |
| `FAMILY` | `e_hat = b_t` | 0 |
| `GLOBAL` | `b_t + a1·d_pred + a2·d_nheavy + a3·d_nhetero` | 3 |
| `TRACE` | `b_t + c1·dL_var_raw + c2·dL_core_raw + c3·dR_raw` | 3 |
| `LOCAL_ONLY`（次要消融） | `b_t + c1·dL_var_raw + c2·dL_core_raw` | 2 |

* 加权 ridge，**无截距**。目标 = `e_delta`。每组件总权重相同、组件内配对均分；
  loss = 各组件 MSE 之和。所有已缩放系数与类别项共用**固定 L2 = 5**
  （小样本收缩选择，不是通用最优，**不做 lambda 搜索**）。
* 连续差分只除以 **Diagnose 的 component-equal-weight RMS**，不做均值平移（保持反向
  奇对称）；RMS=0 的列整列记 0 并报告有效秩。
* 主参照 `BEST_BASE`：只用 Diagnose 的 **leave-one-pair-component-out** 开发交叉验证，
  取组件等权 MAE 最低者，平局顺序 `NATIVE > FAMILY > GLOBAL`。**TRACE 是唯一 trace
  候选**（不从多个 trace 方案中挑一个）。
* 交叉验证每一折**重算**缩放与类别项；某类别只出现在被留出组件时，该折该类项为 0。
* 冻结后再一次性访问 Audit：`analysis/probe_freeze.json`（系数、缩放、BEST_BASE、
  缺失类别策略、代码哈希、commit）。
* Select 只负责 checkpoint；Diagnose 不参与 encoder 学习/epoch 选择；两者不混用。
* 探针系数不得解释为"多少真实误差属于哪个分支"。
* **描述性分析（不拟合）**：在 Diagnose 里按组件等权平均 `|e_delta|`，取最多 3 类
  "至少 3 个不同组件"的变换，记录方向与具体统计变化；到 Audit 只检查同类变换能否
  复现该方向，缺少支持则标 `UNAVAILABLE`。不得用 Audit 另挑"最漂亮"的变换。

---

## 7. P3 — 一次 Audit 评分与判定

主指标：**组件等权 `MAE(e_delta - e_hat)`**（称为"诊断探针误差"，也等于用 `e_hat`
做一次局部响应校正后的 pair-difference MAE）。主要比较固定 C1 上
`TRACE` vs `BEST_BASE`；FULL 仅参照。

同时报告：native pair MAE/RMSE、每模型/seed 的分子级 MAE/RMSE、pair 残差偏差、
seen/unseen 类别分层、变量片段同重原子数子集、组件数、有效不同分子数。
分子级表现只作**载体核验**。

`Gain = MAE(BEST_BASE) - MAE(TRACE)`（正值 = 回溯有增量）。先算每 seed 的组件等权
指标，再描述性平均 3 个 seed；对**相同 Audit 组件**做配对 bootstrap
（`seed=20261102`，2000 次），**所有 seed/方法共用同一重采样**。报告
"条件于这 6 个已训练模型"的 CI，另列 seed 离散度；**不得**把 3 seeds × N pairs 当
独立样本量。

预注册资源决策（不是普适科学阈值）：

| 判定 | 条件 |
|---|---|
| `PROMISING_DIAGNOSTIC_INTERFACE` | 平均 `Gain ≥ 0.03` logD、≥2/3 seeds 为正、组件配对 CI 排除 0、且无单一组件决定正负 |
| `PROMISING_BUT_UNRESOLVED` | 有上述量级但 CI 跨零或种子明显反向；本轮仍停止，不自动加样本/seed |
| `NO_INCREMENT_THIS_PROBE` | 达不到量级、整体反向或弱于简单对照 |
| `TRACEABLE_ONLY` | 回溯恒等式成立但回溯探针无增量（能力与用途的区别） |
| `DATA_INSUFFICIENT` / `EXECUTION_INCOMPLETE` | 对应边界下不输出有效性结论 |

影响度检查（固定）：去掉 native 误差最大的**一个组件**后重算；另报告去掉 native
误差最大**单分子及其全部关联配对**后的结果。排除分析只作诊断，不替代主指标。

闭环检查：若把 `d_pred - e_hat` 当单分子标量势的差分，配对图中的三角形
（不足则用可用基本环）应相消。报告 `e_hat` 的闭环残差；无三角形时报告
`UNAVAILABLE`。非零闭环不使误差预测实验作废，但**禁止**包装成"已修复 LCRC 单分子
预测"。

---

## 8. 记录与身份

* 所有身份使用完整键：`round/partition/model/seed/row_id|pair_id`；
  join 只按 `row_id`/`pair_id`，不按位置；缺失显式 `unavailable`，不跨划分补齐。
* 交付物清单见 `REPORT.md` §0 与 `analysis/RUNBOOK.md`。

## 9. 不能声称的内容

唯一错误路径；真实化学因果；普适结构充分性；优于所有图解释器；全新独立外部验证；
"已修复 LCRC 单分子预测器"；性能/榜单结论。也**不得**把仅删除一条二跳记录的
输入通道干预称作"分子编辑"（本轮主对象是真实、合法分子对）。

---

## 10. 实现映射（在本文件冻结后追加）

| 协议要素 | 代码入口 |
|---|---|
| 常量/哈希/规则 | `code/edit_common.py` |
| 带标号图、WL、显式同构、自同构轨道 | `code/edit_graph.py` |
| P0 配对构造 | `code/edit_pairs.py` |
| P0 单位与划分分配 | `code/edit_split.py` |
| 结构 cache / Fit-only 归一化 / 标签权限 | `code/edit_data.py` |
| 构建 cache（每主机一次） | `code/edit_prepare.py` |
| 单个训练单元（R3） | `code/edit_train.py` |
| CPU 正确性闸门（pre/post） | `code/edit_audit.py` |
| P1 成对回溯 + 全局核验 + gauge | `code/edit_trace.py` |
| P2/P3 探针、冻结、Audit、bootstrap、影响度、闭环 | `code/edit_probe.py` |
| 案例图（固定规则挑选） | `code/edit_figures.py` |
| run matrix（6 单元） | `configs/run_matrix.json` |
