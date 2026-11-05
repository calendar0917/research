# PROTOCOL — 同种编辑、不同核心的上下文效应筛选

Round: `lcrc-edit-context-screen-v1`
结果标签：`EXPLORATORY_EXPOSED_DATA`
说明：本文件逐项照录执行指令给定的实验定义；第 5 节只是实现映射，不新增研究内容。

---

## 1. 本轮唯一研究问题

**同一种局部化学结构替换，在不同分子核心中的真实性质变化可能不同。冻结的 LCRC 是否能
预测这种上下文差异？**

本轮不研究新的误差校正器，不研究 S2-off 重训练，不研究 attention，不设计新的解释指标。
直接使用已有实验数据、真实标签及冻结模型预测。

## 2. 数据边界

原始数据来自 `lcrc_local_edit_error_diagnosis_v1`。已知历史资产：

* 509 个 matched molecular pairs。
* 211 种有方向的变换类别。
* 原有 Fit / Select / Diagnose / Audit 配对组件划分。
* LCRC-A0 和 C1 各三个 seed。
* 现有预测回溯及实验标签审计。

**重要限制：历史 Audit 已经曝光，因此本轮所有结果仅为 `EXPLORATORY_EXPOSED_DATA`。不得称为独立确认。**

用于评价上下文效应的样本限定为原有 `diagnose` 与 `audit` 分区，不使用 Fit/Select 作为
模型泛化评分样本。

不要重训 A0/C1，也不要用任何已观察到的标签效应大小重新筛选数据。

## 3. 先做不看标签的样本可行性检查

对于两次编辑 p、q，仅在满足全部条件时构建上下文比较：

* `class_id` 完全相同，即方向相同的 `var_a => var_b` 结构替换。
* 所属 `component` 不同。
* 所属 `core_class` 不同。
* 不共享原始分子。
* 两个分子对均来自原有 Diagnose/Audit。

每个 `(class_id, core_class, component)` 内有多次重复编辑时先取该单元的平均效应，再生成
不同组件、不同核心之间的比较，减少重复配对的权重。

先输出：合格变换家族数、合格独立组件数、合格独立核心数、合格上下文比较数、贪心选择互不
重叠组件后还能保留多少比较、每个家族的组件/核心数量分布。

可行性最低要求：至少 5 个合格变换家族、15 个不同配对组件、20 次跨独立组件的上下文比较。
这些是本轮预设的数据支持门槛，不是通用的统计学要求。

若未满足，直接输出 `INSUFFICIENT_CONTEXT_SUPPORT`，记录缺口并停止实验 S。不得放宽结构
匹配条件、引入 Fit 样本凑数或自动搜索其他数据集。

## 4. 指标定义与筛选判断

对于每个真实 matched molecular pair p：

* `dy_p = y(B_p) - y(A_p)`，单位 raw logD。
* `dp_p = pred(B_p) - pred(A_p)`，单位 raw logD。

对两个同类变换、不同核心的配对单元 p、q：

* `context_y = dy_p - dy_q`
* `context_pred = dp_p - dp_q`

固定上下文无关参照：`context_pred_zero = 0`。主指标：

`ContextGain = MAE(context_pred_zero, context_y) - MAE(context_pred, context_y)`

正值表示模型比"同一种编辑在所有核心中作用相同"更能预测上下文差异。

统计时：

1. 每个变换家族内部先对上下文比较取平均。
2. 再对变换家族等权平均。
3. 不允许一个具有大量组合的家族主导总分。
4. 同时报告组件数、核心数、家族数、原始比较数及互不重叠比较数。
5. 若进行 bootstrap，以家族作为主要聚类单位，并明确不同家族可能共享组件，区间只是探索性估计。

模型：

* 主方法：冻结 C1 seed0、seed1、seed2。
* 参考方法：冻结 A0 seed0、seed1、seed2。
* 不训练新模型，不 ensemble 调参。

附加对照：在各变换家族内部，对预测的 `dy` 在独立核心之间随机置换 1000 次，破坏结构上下文
与预测响应的一一对应，但保留该家族预测值分布。计算置换对照的 ContextGain 分布。

同时输出在 `|context_y| >= 0.2 raw logD` 的子集上预测上下文差异方向的正确率；此阈值仅作为
固定的描述性分层。

本轮预设的筛选判断（不做显著性宣称，只给出预先约定的资源决策标签）：
`PILOT_PROMISING` 需同时满足：

* C1 三个 seed 中至少两个 ContextGain 为正。
* C1 三 seed 平均 ContextGain 至少达到 `+0.05 raw logD`。
* C1 的结果优于同家族随机置换对照的中位数。
* 不存在删除单个变换家族后主结果方向全部反转的情况。

否则标记 `PILOT_NOT_ESTABLISHED`，并说明主要原因是支持不足、效应太小、随机性不稳定还是
被少数家族主导。无论标签如何，都必须输出完整数据与不确定性；不得自行改变阈值以使实验通过。
这些标签仅用于**是否值得购买下一轮独立验证数据**，不等于论文结论。

## 5. 实现映射（不是新的科学定义）

| 指令项 | 本目录实现 |
|---|---|
| 结构输入 | `pairs.csv`（509 对）、`component_split.json`（`row_to_component`）、`pair_traces.csv`（`partition` 列）；只读，不修改历史文件 |
| 组件判定 | 每条 pair 的 `row_a`、`row_b` 必须落在同一组件（实测 509/509）；不同组件之间分子集合互斥（全局检查，报告为 `molecule_disjointness`） |
| 单元 | `(class_id, core_class, component)`；单元内多对先取均值（本轮每个单元只有 1 对，代码仍按均值实现并有单元测试） |
| 比较 | `ec_units.build_comparisons`：同 `class_id`、不同 `component`、不同 `core_class` 的所有无序单元对 |
| 贪心不重叠 | 按 `(class_id, component_p, component_q, core_class_p, core_class_q)` 排序，组件未被用过则接受；纯结构、不看标签 |
| 预测 | `pair_traces.csv` 的 `d_pred_raw`（冻结模型，不重新推理、不重训）；并校验 `d_pred = pred_b - pred_a` |
| 标签 | 历史同一条冻结路径 `dissect_source.load_lipop(122, labels="train_valid")`；`dy = y(row_b) - y(row_a)` |
| 标签链路核验 | ① 用 `probe_audit_scores.json` 的 74 个 Audit 对 ×6 单元校验 `d_pred` 与 `d_pred - e_delta`；② 独立重算历史 frozen 的 component-equal-weight NATIVE MAE 并逐单元比对（容差 1e-9） |
| 家族等权 | 家族内先算两个 MAE，家族 gain = 差值，再对家族等权平均（`ec_score.family_gains` + `primary_gain`） |
| bootstrap | 以家族为重采样单位，2000 次，固定 seed 20261105（seed-mean 用 20261106） |
| 置换对照 | 家族内对"单元的预测 dy（dp）"随机置换 1000 次；同一置换 index 在 6 个模型单元间共享（common random numbers）；真实 `context_y` 不变 |
| 方向正确率 | 阈值 `|context_y| >= 0.2`；同时给出 pooled 与家族等权结果；零预测差算错 |
| 决策 | `ec_score.decide`：四条预设准则；"主结果方向"用 C1 三 seed 平均的家族等权 ContextGain 判定 leave-one-family-out |
| 额外输出 | `context_family_gains.csv`（家族级 gain）与 `context_scores.json.influence`（前几名家族占比）用于少数家族影响分析 |

## 6. 交付物

`PROTOCOL.md`、`analysis/eligibility.json`、`analysis/context_units.csv`、
`analysis/context_comparisons.csv`、`analysis/context_scores.json`、
`analysis/permutation_control.json`、`REPORT.md`。
