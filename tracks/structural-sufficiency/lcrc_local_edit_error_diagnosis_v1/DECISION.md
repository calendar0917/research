# DECISION — `lcrc_local_edit_error_diagnosis_v1`

Round: `lcrc-local-edit-error-diagnosis-v1` · 冻结协议：本目录 `PROTOCOL.md`
判定只由协议 §7 的预注册规则给出，`REPORT.md` 中的 bootstrap / 影响度 /
闭环 / gauge 均不改判定。

---

## 判定

**`NO_INCREMENT_THIS_PROBE`**

C1（NO_BOND_MEAN，主模型）在一次性 Audit 上：

* mean `Gain = Audit_MAE(BEST_BASE) − Audit_MAE(TRACE)` = **+0.0015** raw logD
  （预注册实质阈值 **0.03**）
* 逐 seed：+0.0030 / +0.0036 / −0.0022（2/3 正号）
* 组件配对 bootstrap 95% CI：[−0.0108, +0.0114]（跨 0）
* 最大单组件占绝对增益 0.17–0.22（无单点决定符号）

A0（FULL，参照）mean Gain = +0.0033（3/3 正号），同样低于阈值 —— 化简与否
不改变结论。

---

## 一、局部编辑误差是否可预测？

**否，至少不能用这套线性分支归因特征预测。**

最直接的证据不是 `Gain` 的数值，而是 Audit 上**每个单元的最佳方法都是
`NATIVE`（= 预测 e_delta = 0）或与它相差 < 0.003 的 `FAMILY`**：

| unit | 最佳方法 | 最佳 MAE | TRACE MAE |
|---|---|---|---|
| C1:s0 | NATIVE | 0.4070 | 0.4093 |
| C1:s1 | TRACE | 0.4144 | 0.4144 |
| C1:s2 | FAMILY | 0.4145 | 0.4217 |
| A0:s0 | FAMILY | 0.4221 | 0.4234 |
| A0:s1 | LOCAL_ONLY | 0.3923 | 0.3927 |
| A0:s2 | FAMILY | 0.3913 | 0.3965 |

迹在 Diagnose 上把 CV MAE 压到 0.31–0.37，到 Audit 上所有方法一起退化到
0.39–0.42，且退化到"零预测"水平。这说明可学习的是 **Diagnose 这 84 对上的
偶然结构**，不是可迁移的误差规律。

`Gain > 0` 的机制要说清楚：`BEST_BASE` 在 Diagnose 上选了 `GLOBAL`，
而 `GLOBAL` 在 Audit 上比零预测器差 0.004–0.006；拿它作参照，
任何略优于此的方法都会得到正的 `Gain`。**这是 baseline 选择规则的产物，不是信息。**

## 二、迹恒等式成立吗？恒等式成立意味着什么？

成立，且非常干净：`d_pred = dL_var + dL_core + dR`，6 个 checkpoint 上
最大绝对残差 1.1e-6 raw logD；反序（5.7e-7）、批量拆分（4.8e-7）、
原子重编号（8.3e-7）都不破坏它。

但恒等式成立**只**说明分解是同一 forward 的代数重排，
**不**说明任何一项携带可预测误差的信息。本轮正是把这两件事分开：
恒等式 100% 通过，探针增量 ~0。

数值上 `|dL_var|`（0.33–0.36）确实是 `|dL_core|`（0.12–0.15）的 2–3 倍，
若只看这一条容易误读为"变体支路主导误差"——**不成立**，占比大 ≠ 可预测。

## 三、化简（A0 → C1）改变了什么？

* 训练/选择层面几乎没变：Select RMSE 0.680–0.688，A0/C1 差异 ≤ 0.0046 且方向不一致。
* 误差可预测性层面也没变：A0 +0.0033、C1 +0.0015，都远低于 0.03。
* 参数少 18 432，训练时间同量级（222–250 s / 360 epoch）。

结论：**删掉 `b_mean` 在这个诊断任务上既没有代价也没有收益**；
它没有把"误差可解释性"变好或变坏。

## 四、闭环与 gauge 说明了什么？

* 闭环：30 个三角，mean |Σ ê| = 0.0081，max 0.0560。残差非零但很小 →
  预测器**近似**路径无关、并非严格闭合。因此**不能**声称"单分子预测器被固定"。
* gauge：等价变换 `c_v + k`、`rho − k·n_atoms` 使 `d_pred` 逐位不变（0.0），
  但让 Audit MAE 变化 0.038–0.40。分支归因的系数与 gauge 强耦合，
  所以这套系数**只能**当误差预测器用，不能当化学贡献分解读。

## 五、下一步（若继续该方向）

1. **换特征族**再试一次，而不是换正则：本轮的失败点在于特征在 Diagnose 上
   过拟合；结构统计量（`dL_var/dR`）已被证否，值得试的是
   逐原子 `Δf/Δh` 的分布矩、或 mapping 质量指标（`n_core_atoms_changed`、
   `max_abs_dh_unchanged`）。
2. **扩大 Audit 的配对组件数**：26 个组件给出的 CI 半宽约 ±0.011，
   要分辨 0.03 的效应至少需要 ~4 倍组件数。
3. **把零预测器写进 baseline 集合**（例如把 `NATIVE` 固定为必选 baseline），
   避免 `BEST_BASE` 选到比零预测器更差的候选而制造虚假正增益。
   这一条是**协议缺陷**，应在本轮之后作为 amendment 记录，不得回溯修改本轮结果。
4. 若要检验"误差是否可用结构统计量解释"，应改成**预注册的效应量 + 功率分析**，
   而不是在同一个 Diagnose 上做模型选择。

---

## 可复现的最小证据链

```
analysis/probe_freeze.json         探针在 Audit 之前冻结（含 commit 与代码哈希）
analysis/probe_audit_scores.json   一次性 Audit（重复运行会拒绝）
analysis/probe_report.json         verdict + bootstrap + 影响度 + 闭环 + gauge
analysis/trace_verification.json   恒等式/反序/批量/重编号
analysis/pair_traces.csv           3054 条配对迹
analysis/label_access_ledger.json  标签访问顺序审计
```
