# DECISION — lcrc-auto-subgraph-dose-localization-audit-v1

本文件只依据本轮**跑之前冻结**的规则作答；不制定后续结构搜索、配对实验或新架构。

## 1. ENGINEERING_GATE

`PASS`

原轮指标重放（用原轮记录重算 vs 原 `main_table.csv`）：`PASS`

下文所有 `SURVIVES/ATTENUATED/NOT_SUPPORTED` 都是本轮为并排比较而冻结的三档映射，不是原轮的 `PILOT_SIGNAL / WEAK_OR_INCONCLUSIVE` 二档标签。

## 2. 五问判定

### `ogbg-molesol`

**Q1 原对照剂量匹配口径** — `SUBSTANTIVE`

* 原 Top-1 的 internal / affected occurrence 中位：12.0000 / 22.0000。原已接受对照中 22.8% 在 affected 口径下超出 25% 容差（三划分合并；分划分见 REPORT §2）；affected 相对差中位 0.1333。
* 结论边界：原轮匹配的是**子图内部二跳路径数**，不是门控实际移除的 occurrence 数；因此旧对照不是“同剂量”对照。这不是说原轮的效应计算有错，而是它的零假设不同。

**Q2 修正后的 Valid/Test effect ratio 与 advantage**

| split | protocol | median effect ratio | median advantage | ratio>1 | verdict |
|---|---|---|---|---|---|
| valid | pilot | 1.2931 | 0.4629 | 84.1% | ATTENUATED |
| valid | corrected | 1.2699 | 0.4352 | 83.6% | ATTENUATED |
| test | pilot | 1.3859 | 0.5330 | 83.6% | ATTENUATED |
| test | corrected | 1.2618 | 0.5263 | 79.4% | ATTENUATED |

**Q3 Top-1 是否仍明显优于随机对照与局部启发式**

* valid: 修正对照判定 `ATTENUATED`（median ratio 1.2699，CI low 1.2236）；Top-1 真实 |delta| > 局部启发式的比例 70.3%，median advantage 0.5698。
* test: 修正对照判定 `ATTENUATED`（median ratio 1.2618，CI low 1.2067）；Top-1 真实 |delta| > 局部启发式的比例 67.2%，median advantage 0.5703。

**Q4 影响是集中在自身根节点还是扩散到周围根节点**

| split | outside fraction median | IQR | >0.5 | boundary/affected | affected rooted outside | verdict |
|---|---|---|---|---|---|---|
| train | 0.1248 | [0.0571, 0.2820] | 9.4% | 0.5000 | 0.2143 | LOCAL_CONCENTRATED |
| valid | 0.2123 | [0.0994, 0.3792] | 7.8% | 0.5584 | 0.2500 | LOCAL_CONCENTRATED |
| test | 0.1398 | [0.0710, 0.2920] | 3.1% | 0.5000 | 0.2111 | LOCAL_CONCENTRATED |

### `ogbg-mollipo`

**Q1 原对照剂量匹配口径** — `SUBSTANTIVE`

* 原 Top-1 的 internal / affected occurrence 中位：6.0000 / 14.0000。原已接受对照中 36.8% 在 affected 口径下超出 25% 容差（三划分合并；分划分见 REPORT §2）；affected 相对差中位 0.2000。
* 结论边界：原轮匹配的是**子图内部二跳路径数**，不是门控实际移除的 occurrence 数；因此旧对照不是“同剂量”对照。这不是说原轮的效应计算有错，而是它的零假设不同。

**Q2 修正后的 Valid/Test effect ratio 与 advantage**

| split | protocol | median effect ratio | median advantage | ratio>1 | verdict |
|---|---|---|---|---|---|
| valid | pilot | 1.5677 | 0.2945 | 71.9% | SURVIVES |
| valid | corrected | 1.8267 | 0.3022 | 71.7% | SURVIVES |
| test | pilot | 1.2864 | 0.1407 | 56.2% | NOT_SUPPORTED |
| test | corrected | 1.2894 | 0.1780 | 59.7% | NOT_SUPPORTED |

**Q3 Top-1 是否仍明显优于随机对照与局部启发式**

* valid: 修正对照判定 `SURVIVES`（median ratio 1.8267，CI low 1.2984）；Top-1 真实 |delta| > 局部启发式的比例 50.0%，median advantage 0.0210。
* test: 修正对照判定 `NOT_SUPPORTED`（median ratio 1.2894，CI low 0.9834）；Top-1 真实 |delta| > 局部启发式的比例 45.3%，median advantage -0.1270。

**Q4 影响是集中在自身根节点还是扩散到周围根节点**

| split | outside fraction median | IQR | >0.5 | boundary/affected | affected rooted outside | verdict |
|---|---|---|---|---|---|---|
| train | 0.3476 | [0.1921, 0.5195] | 29.7% | 0.6667 | 0.3274 | MIXED |
| valid | 0.3366 | [0.1816, 0.4814] | 23.4% | 0.6000 | 0.2958 | MIXED |
| test | 0.3460 | [0.2378, 0.4628] | 21.9% | 0.6429 | 0.3063 | MIXED |

## 3. Q5 支持 / 不支持 / 仍无法判断

**支持**

* 原轮的随机对照确实按 `n_s2_internal`（两条键都在 S 内）匹配，而冻结门控 `w[p]=m[e1]*m[e2]` 实际移除的是 `n_s2_affected`（至少一条键在 S 内）；两者相差边界 occurrence。
* 在冻结前向上，`n_s2_affected` 恰好等于被置零的 S2 权重数（逐分子 1→0 集合比对通过），修正后的对照可以在同一模型上复现。
* 修正对照下逐分子的 effect ratio / advantage，以及分子级 bootstrap 区间；两数据集、三划分分开报告。
* 屏蔽引起的局部读出变化在根节点上的分解：子图内根节点与外部根节点各自的绝对响应、外部占比与闭合检查。

**不支持**

* 任何跨 seed / 跨 checkpoint 的稳定性结论（本轮只有一个冻结 checkpoint）。
* 任何化学反事实解释：`atom_feat` / `atom_scalar` / `global` / 原子数特征并未被门控，屏蔽不是删键或删基团。
* “模型不使用结构”或“架构必须修改”这类结论；本轮没有做架构对照。
* 新的 Top-1、新的搜索空间、新的架构或新的数据集。

**仍无法判断**

* 修正后的判定是否能外推到每划分 64 个分子之外。
* 外部根节点的响应对预测质量是否必要（本轮没有做去掉外部响应的对照）。
* 换一种匹配零假设（例如按边界比例或按权重质量匹配）是否会改变排序；本轮只跑了 affected 剂量匹配。

## 4. 明确不做的事

* 不重新选择 Top-1、不重新训练、不换 seed、不改架构、不扩大搜索范围、不新增数据集。
* 不把本轮结果自动升级为“结构搜索该继续”或“LCRC-v2 必须改读出/改算子”的架构结论。
* 不把 `pilot_internal_matched` 与 `corrected_affected_matched` 混成同一张主表或同一个结论。

