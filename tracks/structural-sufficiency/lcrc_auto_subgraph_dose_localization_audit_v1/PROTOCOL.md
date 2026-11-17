# PROTOCOL — lcrc-auto-subgraph-dose-localization-audit-v1

本文件是本轮的**规则声明（跑之前冻结）**。机器可读版本在
`protocols/lcrc-auto-subgraph-dose-localization-audit-v1.yaml`（控制面协议）与
`configs/experiment.yaml`（工程配置）；冻结常量的哈希在
`analysis/protocol_freeze.json`（`freeze_hash`）。

变化规则：本文件与 `code/au_common.FROZEN_CONSTANTS` 一经正式运行不得为了结果好看而修改。
只允许修**工程实现**（索引 / dtype / 路径 / 账目字段 / 内存分块），并记录在 `README.md`。

---

## 0. 身份、对象与边界

| 项 | 冻结值 |
|---|---|
| 实验 ID | `lcrc-auto-subgraph-dose-localization-audit-v1` |
| 上游轮次 | `lcrc-auto-subgraph-faithfulness-pilot-v1`（交付提交 `147b1d81`） |
| 模型 | `lcrc_readout_context_pilot_small_v1` **Arm A = LCRC-C1**：`sum_v f(h_v) + rho([g, mean(h), mean(h^2), log1p(n_atoms)])` |
| 训练 | **不训练**；只复用既有 360-epoch、seed 0 checkpoint |
| 任务 | 官方 `ogbg-molesol`（902/113/113）、`ogbg-mollipo`（3360/420/420） |
| 解释样本 | 上游轮次每划分 64 个官方 row id，逐项复用 |
| Top-1 | 上游轮次已选定的物理键集合，**逐字复用** |
| 性质 | 只读审计 / 对照协议修正 / 响应定位诊断；不是新搜索、不是新架构、不是化学因果验证 |

**禁止**：重新选择 Top-1、重新训练、更换 seed、修改模型或读出、改变 2–6 键搜索范围、
新增数据集、重新解释另一批分子、用 Test 做任何选择、把两个对照协议混成同一主表。

## 1. 冻结输入（只读）

* 官方 OGB split 索引与四个冻结缓存（真实文件字节 SHA-256 在每个数据集阶段校验）。
* 上游 `results/<dataset>/{train,valid,test}/molecules.json`：Top-1 键集合、梯度、
  发现分数、剂量点记录、局部启发式记录、搜索日志（`candidates.npz`）。
* 上游 `analysis/main_table.csv`：用于核对本轮的“旧指标重算”是否与上游一致。
* 源轮次 checkpoint / 轨迹：SHA-256 与上游记录逐项一致。
* 上游样本选择规则：`sha256(20261117|dataset|split|row_id)` 升序取前 64；
  本轮重新选择并与上游 `samples.json` 逐项比对 row id、cache row 与哈希。

任一身份不匹配 → 停止该数据集，不使用它生成任何结论。

## 2. S2 剂量：internal / boundary / affected

对分子的一条去重物理键集合 `S`，S2 occurrence `p=(root -> middle -> end)` 的两条键为
`e1, e2`。门控权重为 `w[p] = m[e1] * m[e2]`。当 `m[e]=0`（`e in S`）时：

| 量 | 定义 | 门控后权重 |
|---|---|---|
| `n_s2_internal(S)` | `e1 in S AND e2 in S` | 0（两条键都屏蔽，仍只移除一次） |
| `n_s2_boundary(S)` | 恰好一条键在 `S` 内 | 0 |
| `n_s2_affected(S)` | `e1 in S OR e2 in S` | 0 |
| `n_s2_unaffected(S)` | 两条键都不在 `S` 内 | 1 |

恒等式：`n_s2_affected = n_s2_internal + n_s2_boundary`；
`n_s2_affected + n_s2_unaffected = n_s2_total`。

**声明**：上游轮次用于匹配随机对照的是 `n_s2_internal`，它**不是**实际被屏蔽的
occurrence 数。本轮把 `n_s2_affected` 作为唯一的主要匹配量，并同时报告三者。

对每个原 Top-1 与每个随机候选都记录：
`n_s2_internal`、`n_s2_boundary`、`n_s2_affected`、`n_s2_unaffected`、`n_s2_total`，
以及边界 occurrence 的“第一条键被屏蔽 / 第二条键被屏蔽”拆分和
“affected occurrence 的根原子在子图内 / 外”的拆分。

### 2.1 真实门控等价检查（每个分子）

对 `m[e]=0 (e in S)`，必须同时满足：

1. **权重计数**：冻结前向实际消费的 `w[p]` 张量中，恰好有 `n_s2_affected` 个 0、
   `n_s2_unaffected` 个 1，没有分数权重；
2. **前向等价**：把恰好 `n_s2_affected` 个 occurrence 从路径列表中删除后，
   冻结模型的前向输出与“S2 门控置零”（S1 保持全开）在原始目标单位下一致，
   容差 `1e-5 + 1e-6*max(1,|x|)`；
3. **删除一次**：两条键都被屏蔽的 occurrence 计一次删除；任一不满足即停止。

## 3. 修正后的随机对照（A 组）

对每个原 Top-1，键集合**不变**，重新生成同分子随机连通子图：

| 规则 | 冻结值 |
|---|---|
| 匹配量 | `n_s2_affected` |
| 相对差 | `|n_aff(c) - n_aff(Top1)| / max(1, n_aff(Top1)) <= 0.25` |
| 物理键数 | 与 Top-1 完全相同 |
| 排除 | 与 Top-1 完全相同的键集合 |
| 连通性 | 必须连通（生成器保证，仍逐项断言） |
| RNG | `20261117`，`SeedSequence([seed, dataset_index, split_index, row_id])` |
| 生成器 | 上游轮次同一个 `grow_connected_subgraph`（随机流不变，只改接受准则） |
| 尝试上限 | 512 次候选增长 |
| 接收上限 | 前 16 个符合条件的不同候选 |
| 最小数量 | 少于 8 个 → `INSUFFICIENT_MATCHED_NULL`（不补造、不放宽、不替换） |

**不得**再用 `n_s2_internal` 作为主要匹配条件。

效果定义（原始目标单位）：

```
delta(S)        = y0 - yS
effect_ratio    = |delta(Top1)| / (median_e |delta(control_e)| + 1e-12)
effect_advantage= |delta(Top1)| - median_e |delta(control_e)|
```

Top-1 的屏蔽预测必须与上游已保存的真实预测/差值一致（逐字段：
`signed_delta`、`delta_L`、`delta_rho`、`pred_raw_masked`），容差为原始单位
`1e-5 + 1e-5*max(1,|x|)`（CPU 重放与 GPU 记录之间的浮点差）。不一致 → 停止并报告。

bootstrap：2000 次、以分子为重采样单位、seed 20261117、percentile 区间；
只描述固定 checkpoint / 固定数据集内的抽样不确定性。

上游数字与修正数字**并排**报告：上游行标为 `pilot_internal_matched`，本轮行标为
`corrected_affected_matched`；两行不可混池、不可互相替代。

## 4. 响应定位（B 组）

对每个原 Top-1，记录所有原子根节点 `v` 的局部输出 `c_v = f(h_v)`：

* `delta_c_v = c_v(original) - c_v(masked)`（同一冻结函数、同一分子）；
* `inside` = 至少一条被屏蔽物理键的端点原子；`outside` = 其余原子；
* `inside_abs_response = sum_{v in inside} |delta_c_v|`；
* `outside_abs_response = sum_{v in outside} |delta_c_v|`；
* `outside_response_fraction = outside_abs_response / (inside + outside)`；
* 同时保存 `n_s2_internal`、`n_s2_boundary`、`n_s2_affected` 与
  `delta_L`、`delta_rho`、`delta_total`。

必须满足（原始单位容差 `1e-5 + 1e-6*max(1,|x|)`）：

```
sum_v delta_c_v == delta_L
delta_L + delta_rho == delta_total
```

**声明**：inside/outside 是“局部读出响应落在哪些根节点”，**不是**独立、唯一的化学贡献分配。

额外报告：分子级 outside fraction 的中位数与四分位、`>0.5` 的比例、
`boundary/affected` 比例、`affected` occurrence 中根原子在子图外的比例，
以及这些指标与 Top-1 键数、`|score_A|`、`|delta_total|` 的 Spearman 对应关系
（描述性，不做机制断言）。

## 5. 判读规则（跑前冻结）

### 5.1 原对照口径缺陷

* `SUBSTANTIVE`：原已接受对照中，在 affected 口径下超出 25% 容差的比例 ≥ 20%，
  或 affected 相对差中位数 > 0.25；
* `MARGINAL`：比例 > 0 但不满足上条；
* `NONE`：比例为 0。

### 5.2 修正后的对照优势（沿用上游冻结的投入规则，便于并排比较）

对 Valid 与 Test 分别判定：

* 四项条件：`median effect_ratio >= 1.5`；`ratio>1` 比例 `>= 0.70`；
  `1% Train std` 效应底线比例 `>= 0.50`；匹配分子数 `>= max(32, 80%×已解释分子)`。
* `SURVIVES`：四项全过且 ratio 的 bootstrap CI low `> 1`；
* `ATTENUATED`：`median ratio > 1` 且 CI low `>= 1`，但未全过；
* `NOT_SUPPORTED`：其余。

### 5.3 响应定位

* `LOCAL_CONCENTRATED`：分子级 outside fraction 中位数 `<= 0.25`；
* `SPREAD_TO_SURROUNDING_ROOTS`：中位数 `> 0.50`；
* `MIXED`：其余。

## 6. 工程门禁

`DATA_IDENTITY`、`PILOT_SAMPLE_IDENTITY`、`TRAINING_IDENTITY`、
`MODEL_EQUIVALENCE_ALL_ONES`、`OCCURRENCE_MAPPING`、`TOP1_SOURCE_IDENTITY`、
`S2_AND_OR_ACCOUNTING`、`S2_WEIGHT_COUNT`、`LOCALIZATION_CLOSURE`、
`PARAMETER_HASH_UNCHANGED`、`PARAMETER_BUDGET`。

任一失败 → `ENGINEERING_GATE=FAIL`，不进入正式 audit 阶段。

## 7. 执行与资源

* 可以在 CPU 上完成的部分先完成；GPU 只用于必要的冻结模型批量推理。
* 每个分子一个小的重复批次（BASE + Top-1 + 至多 16 个对照），按
  `MAX_VARIANTS_PER_FORWARD = 8` 分块；单次前向的原子/键数有硬上限保护。
* 逐 split 落盘，峰值内存不得随 split 数或对照数增长；记录 `peak_rss_mb`。
* 不训练、单 seed、不新增数据集、不修改上游任何文件。

## 8. 产物

`PROTOCOL.md`、`REPORT.md`、`DECISION.md`、`analysis/old_vs_corrected.csv`、
`analysis/matched_controls.csv`、`analysis/old_control_dose.csv`、
`analysis/localization.csv`、`analysis/localization_summary.json`、
`analysis/preflight.json`、`analysis/bootstrap.json`、`analysis/decision.json`、
`analysis/run_manifest.json`、`analysis/protocol_freeze.json`、代码与测试、
`results/<dataset>/...` 逐分子记录。
