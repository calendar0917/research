# DECISIONS — `zinc_adaptive_arch_search_v1`

每轮一条 Decision Log（约一页）。格式：Hypothesis / Intervention / Control /
Observation / Alternative explanation / Decision / Next。

---

## D1 — 方向 A 第一步：BOSI-QP（A1）与匹配 null（A0）

**Hypothesis**
BOSI 的伙伴矩是"先聚合再使用"：`M[v,e] = Σ_f T_k(x[v,f])`，目标键状态只在之后
的 Side 里才与它相遇。若把目标半键状态放进**每一个 pair 的共享非线性编码**（聚合之前），
局部组合规则可以以更少的样本被学到，从而改善 official-valid MAE。

**Intervention**（`code/aas_models.py`，commit `451fc8d`）
`P[v,e] = Σ_{f≠e} Phi([x[v,e], x[v,f], |x[v,e]-x[v,f]|, x[v,e]·x[v,f]])`，
`Phi = 128→64→32`（SiLU，共享），`P` 追加到**未改动**的 BOSI-B Side 输入
（617→649；新增 32 列零初始化）。其余（矩块、closure、Side/Compose/Reader、配方、
SSL 初始化、batch 顺序）逐项不变。488,321 参数。

**Control**（A0）
同参数、同初始化、同优化器、同数据顺序；只把 `q_e = x[v,e]` 换成**同一伙伴集合的均值**
`q_null[v,e]`。A0 仍能看到伙伴内容（经 Phi 的非线性 + 与均值的交互），所以它隔离的
正是"目标键自己是否参与每个 pair 的计算"，而不是"是否多做了一次非线性编码"。

**CPU 门控**（`analysis/cpu_checks.json`，11/11 PASS）
步 0 前向与 BOSI-B **逐位相同**（max\|Δ\|=0.0）；A1/A0 除新分支外初始张量与 BOSI-B
逐张量相同；`P` 为 0 的比例 8.25%（= 度 1 端点的空伙伴集，严格 0）；伙伴行置换下
`P` 差 3e-8；扰动单个目标半键只移动它自己的 `P` 行（A0 的自身行不动，跨集合泄漏 0）；
`Phi` 在 step 0 梯度为 0、step 1 起非零有限；40 步拟合 loss 1.54→0.61。

**Observation**
见 `REPORT.md` 与 `analysis/summary_a.json`（运行完成后回填）。

**Alternative explanation**
若 A1 ≈ A0 且两者都优于 B，则收益来自"前聚合的非线性内容编码"，而不是目标条件化。
若 A1/A0 都劣于 B，需先排除输入尺度/优化异常（已由 C5 型尺度检查与 step-0 逐位一致排除），
之后才判该机制无益。

**Decision**：待定（见下方 D2）。

**Next**：A1 结果落地后按 PLAN 的预注册区间决定 DEEPEN / NONLINEAR_ONLY / STOP_A。

---

## D2 — 方向 B：SQBC（S1 有序 vs S0 对称）

**Hypothesis**
DCC-L 的强项是"learned junction 内容 + 真实 ownership 汇入键"；BOSI 的弱项是没有
把 pair 当作**有角色的有序对象**。若把键级 pair 计算做成**目标键有显式角色**的共享
非线性算子（S1），并且只保留一次静态组合，则比"同信息但目标盲"的对称楔形（S0）
更能改善低支持/未见局部环境。

**Intervention**（`code/aas_sqbc.py`，commit `4ae20ac`）
独立无递归主体：`u_e=Wq b_e, w_e=Wk b_e`（96→32），对每个共享中心的有序键对
`(e,f)` 计算 `w_ef = Phi([u_e,w_f,u_e·w_f,|u_e-w_f|, a_v, rel, log1p n])`，
`P[v,e] = Σ_{f≠e} w_ef`，`c_e = Compose([b_e, P_u+P_v, |P_u-P_v|, P_u·P_v, bond_feat,
log1p n_junc])`，读出 `[Σc, Σc², Σa, Σa², global33] → 288 → 288 → 1`。381,505 参数。
没有 junction encoder、没有 384 维矩 bank、没有图级关系通道、没有 attention。

**Control**（S0）
**逐张量相同的初始状态**（`initial_state_bit_identical=true`），只是 pair 输入换成
无序对的基本对称不变量 `[u+w, |u-w|, (u+w)|u-w|, u·w]`——这是 `{u,w}` 的完全不变式。
因此两支的差别**只在**"目标键是否有非对称角色"，参数/容量/初始化/配方完全相同。

**CPU 门控**（`analysis/cpu_checks_b.json`，10/10 PASS）
参数形状与初值逐位相同；S1/S0 前向不同；角色互换（诊断开关）下 S1 的 `P` 变化
>1e-3 而 S0 **逐位不变**；空伙伴集 `P=0` 且全有限；原子重标号/键反向/键序/批次重组
不变性 ≤1.5e-7（**键反向最初失败 3.2e-3，因为把 `P_u,P_v` 放进了不对称槽位**；
已改为只在键级使用对称组合，见 `aas_sqbc.py` 注释——这是本轮唯一一次实现修复，
在正式训练前完成）；所有模块梯度有限非零。

**Observation**
见 `REPORT.md` / `analysis/summary_b.json`。

**Decision**：待定。
