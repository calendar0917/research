# ERRATA — `lcrc_performance_route_8h_public122_s0_v1`

| 项 | 值 |
| --- | --- |
| 轮次 | `lcrc_performance_route_8h_public122_s0_v1`（8 h 轮，public `random122`，seed 0） |
| 本文件日期 | 2026-11-14 |
| 触发 | 定向复核：8 h 轮中有 5 处实现/派生缺陷会影响**路线判断**，另有 1 处协议级 D-MPNN 调度错误 |
| 范围 | 只做实现修正、核验与重算；**不重新设计科研路线**，不扩模型 / seed / 数据集 |
| 新训练 | 仅 2 次（`DMPNN_LR3E4_FIX`、`DMPNN_LR1E3_FIX`），合计 **0.0413 A100·h**（上限 0.5） |
| 旧产物 | 全部旧 checkpoint / `curve.csv` / `predictions_*.npz` / `train.json` / 旧分析表**逐字节未改**；修正量另存为 `*_corrected.json` / `support_v2_*.npz` |
| 结论性质 | 仍是单 seed、单轮、Test 参与过路线选择。本文件只修正**实现与口径**，不增加任何外部有效性 |

本文件是唯一勘误的可引用口径；`REPORT.md` 中受影响的原句已就地标注 `【勘误 Ex】` 并指向本文件。
**`REPORT.md` 不是全部作废**：身份核对（§1）表明所有 Lipophilicity 臂的预测与指标身份成立，
因此 M/K/配方/迁移等**未被实现缺陷触及**的训练数字在核对后依然可用。

复现：

```bash
uv run python code/run.py correct                     # 全部修正分析（CPU，<1 min）
uv run python code/run.py correct --stages tail       # 单项
uv run python code/run.py schedule-trace              # D-MPNN 玩具调度追踪（无数据、无 GPU）
uv run pytest tests -q                                # 14 个回归测试
```

---

## 0. 摘要：五条修正 + 一条新证据

| 编号 | 缺陷 | 影响面 | 修正后的一句话 |
| --- | --- | --- | --- |
| **E1** | `tail_view` 取**最小**误差行当"要删除的尾部" | §5.1、Q4 的"不靠尾部"整段 | K 的 Test 增益**确实随砍尾快速衰减**（4.56 % → 3.56 % → 1.13 %），Valid 增益在砍 5 % 后**变负**（+13.16 % → −2.37 %） |
| **E2** | `build_support` 数的是**根原子**不是**分子** | §3.2、S 家族全部结论 | 四个 S 臂是 `root-support legacy` 实验，**没有检验**注册的分子支撑假说；修正后 κ 128.5→86、`a_bar(4κ)` 0.911→0.843 |
| **E3** | 梯度诊断对**全部参数**取范数、aux 未乘 0.1 | §4.2、Q3 的"优化负担"机制 | "aux/main = 9.64 / 20.4"在正确加权下是 **0.96 / 2.04**；保留权重上共享主干比值 ≈ **0.61**。"优化负担是已证实原因"**降级为待检验解释** |
| **E4** | `Adam(lr=peak_lr)` × `LambdaLR(desired/init)` | §2.4、Q1 的"D-MPNN 距离" | 实际 lr 是协议的 **3×（D1）/ 10×（D2）**；两条旧结果标 `DMPNN_LEGACY_LR_MISMATCH`。按协议重跑的 D-MPNN 主比较（按 Valid 选）Test 只领先 C1_R1 **+2.91 %**，不是 +7.35 % |
| **E5** | `run_table` 生成于 `_curve_best_epoch` 修复之前，且读错 `commit` 键 | 附录、机器表 | 原始记录正确、**派生表过期**。`best_epoch` 由 curve 重算（K=90、C1_R1=187），`commit` 由 `provenance.git_commit` 恢复，`epochs_run` 单列（LONG 720 / HIV 160） |
| 7 | （非缺陷）用已有曲线做学习曲线检查 | §2.5、K 的路线判定 | K 的 **Test** 优势只在 epoch≈60–120 的窄窗口存在；epoch 187 处 K 的 Test 0.5847 **差于** C1_R1 的 0.5839；240–360 窗口 Test 中位数 K 0.5923 vs 0.5786 |

---

## 1. 身份核对（第一步，先做，先于任何解释）

产物：`analysis/corrected_manifest.json`、`analysis/run_table_corrected.{json,csv}`。
每条 run 逐字段核对：`curve.csv` 行数 → optimizer updates → curve 的 Valid argbest →
`train.json`（是否存在 `best_epoch`/`epochs_run` 键）→ `predictions_best.npz` 的 row ids / 标签 / 单位 / 指标 →
（若有权重）CPU 重播 checkpoint。**不从事后叙述回填机器表**：新表只由 `runs/*/train.json` + `curve.csv` 生成。

### 1.1 结果

* 25 条 Lipophilicity 臂（23 条旧 + 勘误轮新增 2 条，含全部 10 条被点名的身份）：
  **`prediction_identity = VERIFIED`**。具体满足
  ① `predictions_best.npz` 的 `*_rows` 与发布划分逐元素相等（行 id 与 sha256 记录在案）；
  ② `*_y` 与发布标签逐元素相等；③ 只用 Train 的 mean/scale 反标准化一次；
  ④ 重算 RMSE/MAE 与 `train.json.metrics.best` 差 ≤ 1e-6；
  ⑤ 与 `curve.csv` 在 best-Valid epoch 那一行差 ≤ 1e-6；⑥ `predictions_last.npz` 与 curve 最后一 epoch 一致。
* 只有 **3 个旧 checkpoint 被保留**（`runs/` 是 git-ignore，其余在轮末删盘）：
  `C1_R1`、`A_LOGP`、`S_SELECT_K1` → 三者 `weight_identity = VERIFIED`
  （CPU 重播：max |Δpred| 2.6e-6 / 2.3e-6 / 1.7e-6；RMSE 完全对上）。
  两个新增 D-MPNN run 的 `best.pt` 在盘上 → `VERIFIED_METADATA`（见下）。
  其余 20 条 `weight_identity = NOT_RETAINED`。因此：
  **主性能数字（全部基于保存的预测）身份成立；只有"重新从权重算一遍"的检查（机制面板、# 检查点梯度）受保留范围限制。**
* **关于"对 base 与 K 做 CPU 重播"（prompt 点名项）**：base `C1_R1` 已重播并 `VERIFIED`（max|Δpred| = 2.6e-6，
  RMSE 逐位对上）；**`K_COMBO` 的 `best.pt` 在本机不存在**（`runs/K_COMBO__Lipophilicity/` 只剩
  `train.json`/`curve.csv`/`diagnostics.json`/`predictions_*.npz`），所以 K 的重播**不可恢复**，记为
  `weight_identity = NOT_RETAINED`（不是遗漏）。K 的身份因此由**三条相互独立**的路径确认并全部通过：
  ① `predictions_best.npz` 的 row ids/labels/单位、② 重算指标 vs `train.json`、③ best-Valid 与最后一 epoch
  两处 vs `curve.csv`。若要真正重播 K，需要重训一次（本轮未做，也不计入 0.5 GPU·h）。
* **D-MPNN 的权重核对只做到元数据层**（`VERIFIED_METADATA`）：`best.pt` 自己记录的
  `best_epoch` 与 curve argbest 相等，`best_metric_value` 与该 epoch 的 `valid_rmse` 差 ≤1e-6，
  参数量 318 301 与 `train.json.model.parameters` 一致。**不是**前向重播（D-MPNN 重播需要定向边/特征缓存，
  未做）；其预测身份已由 `predictions_best.npz` 独立确认。
* 重播时发现一个真实的可复现性约束（已记入 manifest）：`S_SELECT_K1` 的预测**依赖 `cache/support_*.npz`**，
  支撑表不在 checkpoint 内。只 `load_state_dict` 而不喂 `path_key_count` 会得到 0.57373（而不是 0.57393）。
  结论：S 臂的 checkpoints 是"权重 + 冻结支撑表"才能复现，两者都在库内，可复现，但必须成对使用。

### 1.2 E5：派生表过期 vs 记录损坏

| 现象 | 判定 | 依据 |
| --- | --- | --- |
| `run_table.csv` 大量 `best_epoch=1` | **派生表过期** | 该表 `created_utc=2026-11-13T21:31:14Z`，而修复 `_curve_best_epoch` 的提交 `d6e8d1f4` 在 21:34:51 才落地。旧实现用 `value > best`（**最大化** valid_rmse），所以选中的是 Valid 最差的 epoch：R1 的 epoch 1 valid_rmse=3.06 正是最大值 → `best_epoch=1`。修复后重算：K=90、C1_R1=187、C1_R3=161、A0_R0=321…与 `REPORT.md` 附录手写值一致（**报告是对的，机器表是旧的**） |
| `commit=None` 全部 | **取字段错误** | 表读 `provenance["commit"]`；记录里键名是 `provenance["git_commit"]`。已恢复（K/R1 为 `7df6c3c7`/`4fe07a0c`） |
| LONG `epochs=360`、HIV `epochs=360` | **列语义错误** | 该列取自 `recipe["epochs"]`（协议声明），实际 `epochs_run` 是 720 / 160（curve 行数亦为 720 / 160）。新表单列 `epochs_run` 与 `epochs_declared` |
| DMPNN 记录无 `best_epoch` | **记录缺字段** | 两条 DMPNN 的 `train.json` 从未写顶层 `best_epoch`（只写 `recipe.epochs_run`）。已在 `lcpr_dmpnn.py` 补写该字段（对已完成的旧 run 仍以 curve argbest=64 / 59 为准，并标明来源） |

**没有只为"表一致"去改任何原始记录**；旧 `run_table.{json,csv}` 保留原样并在 `corrected_manifest.json` 里标注
`superseded_by: analysis/run_table_corrected.{json,csv}`。

---

## 2. E1 — 尾部：修正 drop 集合

**缺陷**（`lcpr_analysis.tail_view`）：`np.argsort(ae)[:k]` 返回**最小**误差的 k 行，而所有调用方把它当作
"要删掉的集合"（键名就是 `exclude_top_*`）。自证矛盾：旧表里 base Test RMSE 由 0.58394 **升到**
0.58674（1 %）/ 0.59909（5 %）——删掉最大残差不可能抬高 RMSE。

**修正**：按 base 绝对误差**降序**，平局用**发布 global row id 升序**；
`k = max(1, round(frac·n))` 不变；同一比较中所有模型删除**完全同一组 base 行**；
主结果始终是全样本。position 与 global row id 分列（例：Valid 的第 212 个位置 = 发布行 **2087**）。

产物：`analysis/tail_corrected.json`、图 `analysis/figures/tail_corrected_cumulative_dsse.png`。

### 2.1 K 对 C1_R1（base 定义 drop 集合）

| split | 口径 | base 同行 RMSE | 臂同行 RMSE | 增益 | 被删行占 base SSE |
| --- | --- | --- | --- | --- | --- |
| Test | 全样本 420 | 0.58394 | 0.55728 | **+4.56 %** | — |
| Test | 删最大 1 %（k=4, 416 行） | 0.54282 | 0.52351 | +3.56 % | 14.4 % |
| Test | 删最大 5 %（k=21, 399 行） | 0.46340 | 0.45816 | **+1.13 %** | 40.2 % |
| Valid | 全样本 420 | 0.65263 | 0.56674 | **+13.16 %** | — |
| Valid | 删最大 1 %（k=4） | 0.50817 | 0.48764 | +4.04 % | 39.9 % |
| Valid | 删最大 5 %（k=21） | 0.42316 | 0.43321 | **−2.37 %** | 60.1 % |

→ 旧报告"去掉 5 % 后增益升到 +4.94 %、`tail_only_gain=false`、K 的增益**不靠尾部**"必须撤回。
**正确说法**：K 的增益**高度集中在 base 最差的行**；把 base 最差 5 % 删掉后，Test 只剩 +1.13 %，Valid **变负**。

### 2.2 同一 drop 集合下的三臂交叉表（base=C1_R1 定义）

| 臂 | Test 全样本 | Test 删 1 % | Test 删 5 % | Valid 全样本 | Valid 删 5 % |
| --- | --- | --- | --- | --- | --- |
| `K_COMBO` | +4.56 % | +3.56 % | **+1.13 %** | +13.16 % | **−2.37 %** |
| `C1_CHEM8` | +3.08 % | +0.54 % | **−0.71 %** | −0.20 % | −2.45 % |
| `C1_CONTRIB` | −3.83 % | −4.15 % | −6.03 % | +17.63 % | −2.06 % |
| `C1_R3` | +2.90 % | +2.59 % | **−0.60 %** | −0.14 % | −5.75 % |

→ 在**被保留的行**（= 去掉冻结 base 最难 5 % 之后的 399 行）上，本轮**没有任何臂**明显领先；
K 是唯一在 Test 删 5 % 后仍为正的臂，
但幅度只有 +1.13 %。

### 2.3 K 对 R3 / K 对 CHEM8（pair 自己的 control 定义 drop 集合）

| 比较 | Test 全样本 | Test 删 1 % | Test 删 5 % | Valid 全样本 |
| --- | --- | --- | --- | --- |
| `C1_R3` vs `K_COMBO`（control=K） | −1.74 % | −0.55 % | −7.90 % | −15.31 % |
| `C1_CHEM8` vs `K_COMBO`（control=K） | −1.56 % | −2.77 % | −8.64 % | −15.38 % |

注意符号与口径：这两行的 control 是 **K**，负增益 = 对手更差。把它读成"K 越砍尾越强"是**错的**——
它说明 **K 自己的残差尾部比 CHEM8/R3 更重**（K 的最大 21 个残差占它 Test SSE 的 40.8 %）。
所以"K 比同配方 CHEM8 好"这件事**同时依赖用谁的尾部定义行集**，这正是必须两种口径都报的原因。

### 2.4 增益坐在哪个误差层（K 对 C1_R1，按 base 绝对误差五分位）

| 分位（易→难） | Test 相对增益 | Test 改善行比例 | Valid 相对增益 | Valid 改善行比例 |
| --- | --- | --- | --- | --- |
| Q1 | −272.5 % | 0.20 | −272.5 % | 0.19 |
| Q2 | −55.7 % | 0.45 | −62.2 % | 0.40 |
| Q3 | −26.5 % | 0.58 | −22.4 % | 0.51 |
| Q4 | **+6.9 %** | 0.67 | −1.2 % | 0.69 |
| Q5（最难） | **+10.6 %** | 0.64 | **+11.7 %** | 0.76 |

（相对增益在 Q1 上分母极小，只看符号与量级，不要读成百分比。）

**"Valid 和 Test 的收益是否由同一种样本承担？"→ 是，同一种误差层，不是同一批行**（两个 split 行不交）。
两个 split 的增益剖面几乎同形：**在 base 已经拟合好的 60 % 行上 K 明显更差，只有最难的两成明显更好**。
这既解释了 Valid 的大增益为何在砍尾后消失，也说明"K 的 Valid +13.2 %"不是"整体更准"。

同时注意 MAE（同一个全样本）：Test 上 K **0.4136** 差于 `C1_CHEM8` 0.4095 与 `C1_R3` 0.4123；
Valid 上 K 0.3879 优于 R1 0.4043。也就是说 **K 的 Test RMSE 优势来自平方误差（尾部），不是中位误差**。

---

## 3. E2 — S 家族：定义错了，假说没有被检验

**缺陷**（`lcpr_data.build_support` 195–200 行）：`for a in range(n_atoms)` 逐**根原子**累计，
同一分子里多个根携带同一个 key 就被记多次 → `c` 是**根支撑**。注册定义（`PROTOCOL.md`、
`configs/experiment.yaml: support_rule.c_key`）写的是"包含该 key 的**不同 Train 分子**数"。
反证早已在库里：`analysis/prepare.json` 里 BACE `count_max = 17425`，而 BACE Train 只有 **1210** 个分子。

**修正**（`lcpr_data.build_support_v2` + `_mol_key_counts`）：对每个 Train 分子用 `atom_ptr` 界定其根范围，
再由 `path_ptr` 定位该分子的**全部** path，整分子的 key 集合**只计一次**；写入新文件
`cache/support_v2_{dataset}.npz`，旧 `support_*.npz` **未动**。
断言（同时是 `tests/test_lcpr_corrections.py` 的用例）：`1 ≤ c_mol ≤ n_Train`、`c_mol ≤ c_root`、
同分子复制同一 key 的根不增加计数、两分子共有该 key 则 `c_mol = 2`、未知 key 的映射约定与旧实现一致。

产物：`analysis/support_v2_summary.json`。

| 数据集 | Train 分子 / 根原子 | key 数 | `c_root` max（旧） | `c_mol` max（新） | κ 旧→新 | `a_bar(κ)` 旧→新 | `a_bar(4κ)` 旧→新 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Lipophilicity | 3360 / 90 863 | 114 | **44 125** | 3320 | 128.5 → **86** | 0.9691 → **0.9487** | 0.9111 → **0.8430** |
| ESOL | 902 / 12 114 | 103 | 4448 | 554 | 35 → **20** | 0.9345 → 0.9078 | 0.8398 → 0.7601 |
| BACE | 1210 / 40 720 | 83 | 17 425 | 1204 | 156 → **79** | 0.9414 → 0.9066 | 0.8470 → 0.7324 |

逐 key 的 `c_mol / c_root` 中位数 = **0.500**（Lipophilicity/ESOL），只有 15.8 % 的 key 两者相等。

各 split 的 α 分布（Lipophilicity，κ=86）：Train/Valid/Test 的 α 中位数都是 **0.9698**，
p05 ≈ 0.836，`frac(c ≤ 1)` ≤ 0.06 %。→ 修正后的收缩**依然很轻**，但 4κ 档从"几乎不收缩"（0.911）
变成"可见收缩"（0.843），因此"收缩越轻越好"这条经验在 4κ 档**不再能直接外推**。

**状态与结论**

* 四个 S 臂（`S_SELECT_K1/K4`、`S_UNIFORM_K1/K4`）的训练输入是**根支撑** →
  统一改称 **`root-support legacy`** 实验。它们检验的是"根支撑收缩"，**不是**注册的分子支撑假说。
* 本轮**不重训 S**（预算与 prompt 约束），因此 **"分子支撑假说是否值得购买"目前未被回答**。
* 明确禁止的表述：把 v2 支撑表换进旧 S 权重重算的预测称作"修正后的 S 结果"；或把"修好实现"当成
  自动值得再开一轮搜索。旧 S 数字（Test +1.71 % / +2.40 %、Valid 全负）保留在案，作为
  **root-support legacy 的事实记录**。

---

## 4. E3 — 辅助梯度：能算的算清，不能恢复的撤回

**缺陷**（`lcpr_train._grad_diagnostics`）：
① 对 `model.parameters()`（**全部**参数）求范数——包含主任务私有头 `f`/`rho` 与 aux 私有头 `q_aux`/`rho_aux`，
所以不是"共享 trunk 上的实际辅助更新占比"；② `grad_aux_l2` 用的是**未加权**的 aux loss，
而真实损失是 `loss_main + 0.1 · loss_aux`。因此报告里 "0.59 / 9.64 / 20.4 倍" 三个数字都不是训练里生效的量。

**能恢复的**：只补 0.1 权重（作用域仍是全部参数，历史标量无法再拆分）：

| 臂 | step | 记录值 aux/main | 乘 0.1 后（全部参数） |
| --- | --- | --- | --- |
| `A_LOGP` | 1 / 100 / 1000 / last | 0.229 / 0.593 / **8.10** / 0.528 | 0.023 / 0.059 / **0.810** / 0.053 |
| `A_MW` | 100 | **9.64** | **0.964** |
| `A_SHUFFLE` | 100 | **20.4** | **2.04** |

→ "aux 梯度压过主梯度 10–20 倍、成了事实上的主任务"这句话在正确加权下**最多是 0.96–2.04 倍**，
且这是**全部参数**口径（私有头贡献了大部分范数）。

**不能恢复的**：step 1/100/1000 的 **checkpoint 没有保存**（`runs/A_MW__*`、`runs/A_SHUFFLE__*` 只剩
`train.json`/`curve.csv`/`diagnostics.json`/`predictions_*.npz`），所以无法从旧标量还原"共享 trunk 上的比值"，
也不能靠重训恢复那条轨迹（重训不是"恢复"，是新证据）。

**能新算的**（`CHECKPOINT_DIAGNOSTIC`，明确不是原训练时的梯度）：在保留的 best 权重上，
eval 模式、按 row hash 固定的 4 个 Train minibatch（128/批），同一次 forward 上分别求
`g_main_shared`、`g_aux_shared`。共享主干 = `atom_encoder.* + bond_encoder.* + local.*`（23 个张量），
排除 `f/rho/q_aux/rho_aux`（16 个张量）。产物：`analysis/grad_checkpoint_diagnostic.json`。

| 臂 | checkpoint | ‖0.1·g_aux_shared‖/‖g_main_shared‖（4 批均值，括号内为逐批） | 加权 cosine（4 批均值 + 范围） | 全部参数加权比（4 批均值） | 全局 clip 系数 ‖g‖/5（4 批均值） |
| --- | --- | --- | --- | --- | --- | --- |
| `A_LOGP` | **保留** | **0.605**（0.54 / 0.65 / 0.49 / 0.74） | **+0.062**（单批 +0.096 / −0.173 / +0.138 / +0.185） | 0.810（单批 0.67–1.08） | **0.253**（单批 0.19–0.29；**不会被 clip**） |
| `A_MW` | 未保留 | — | — | — | — |
| `A_SHUFFLE` | 未保留 | — | — | — | — |

**必须撤回 / 降级的三句话**

1. 撤回："aux 头被喂得越好、主任务越安全；喂不动的 aux 目标就是纯噪声源"——这条因果链的证据
   （10–20 倍压过）不成立，且在保留权重上加权后的辅助贡献是**小量且弱对齐**（0.61 / cos≈0.06）。
2. 降级："优化负担是已证实原因" → **待检验的解释**。主任务无稳健联合收益（Test 排序真目标 > 打乱 > MolWt，
   但无一超过基线，且 Valid 全负）这个**事实**不变。
3. 收敛 WC_LOGP aux R² ≈ 0.976 只说明"联合训练后的辅助预测能力"，**不证明**原 C1 表示早已包含该信息，
   也**不证明**主任务不需要该信息。MolWt 是**组成相关的计算任务**，不能写成"无信息量"或"与 logP 无关"。

---

## 5. E4 — D-MPNN 调度：实际运行确实用了错误 lr

**缺陷**（`lcpr_dmpnn.train_dmpnn`）：`Adam(lr=peak_lr)` 而 `_noam_lambda` 返回 `desired_lr / init_lr`；
`LambdaLR` 的实际 lr = `peak_lr × desired_lr / init_lr`。于是 D1 实际 init/peak/final =
**3e-4 / 9e-4 / 3e-4**，D2 = **1e-3 / 1e-2 / 1e-3**，而协议要求 1e-4 / (3e-4 或 1e-3) / 1e-4。

**判定链条**（不只看配置字段）：

1. 两条旧 run 的 `provenance.code_hashes["code/lcpr_dmpnn.py"] = fb0618ef…`，与含缺陷的版本**逐字节相同**
   → 执行的就是这段代码。
2. 玩具 optimizer（单参数 Adam + 真 `LambdaLR`，全程 5300 步 = 53 步/epoch × 100）：
   首个 update 3e-4 / 9e-4，warmup 端点（step 106）9e-4 / 1e-2，末端 3e-4 / 1e-3。
3. 实际 `curve.csv:lr_end`：D1 = 6.0e-4 / 9.0e-4 / 3.0e-4，D2 = 5.5e-3 / 1.0e-2 / 1.0e-3
   —— 与玩具"legacy"轨迹逐端点相等（≤1e-9），与"fixed"轨迹（2.0e-4 / 5.5e-4 首 epoch）不相等。
4. `scheduler.step()` 在 `optimizer.step()` **之后**（这一点本来就是对的，`lr_end` 记录的是"下一步将使用的 lr"）。

→ 两条旧结果标 **`DMPNN_LEGACY_LR_MISMATCH`**，保留在案但不再代表"固定协议的 D1/D2 分数"。
前向 parity（max|Δ|=0，容差 1e-6）**不检验优化器调度**，不能据此说 recipe 正确。

**修正与新训练**（唯一改动：optimizer 基准 lr = `init_lr` = 1e-4；`LambdaLR` 定义不变；
其余超参/特征/初始化/批序/100 epoch/seed 0 全部不变；每 epoch 全量 Train/Valid/Test；按 Valid 选 epoch）：

| run | 配方 | 基准 lr | 实测 lr（首/峰/末） | best-Valid epoch | Valid RMSE | Valid MAE | Test RMSE | Test MAE | 参数 | GPU·h |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `DMPNN_LR1E3_FIX` | D2F | 1e-4 | 1e-4 / 1e-3 / 1e-4 | 92 | **0.64509** | 0.38277 | **0.56694** | 0.41292 | 318 301 | 0.0211 |
| `DMPNN_LR3E4_FIX` | D1F | 1e-4 | 1e-4 / 3e-4 / 1e-4 | 95 | 0.73322 | 0.42633 | 0.58468 | 0.45568 | 318 301 | 0.0202 |
| `DMPNN_LR3E4`（legacy） | D1 | 3e-4 | 3e-4 / 9e-4 / 3e-4 | 64 | 0.73645 | 0.39785 | 0.54102 | 0.40079 | 318 301 | 0.0224 |
| `DMPNN_LR1E3`（legacy） | D2 | 1e-3 | 1e-3 / 1e-2 / 1e-3 | 59 | 0.65301 | 0.42429 | 0.61125 | 0.44974 | 318 301 | 0.9692 |

**按冻结规则（Valid 选配方）主比较 = `DMPNN_LR1E3_FIX`**；另一条按 prompt 要求作为 Test 参考列出，不挑最好 Test。

| 对照 | ΔValid | ΔTest | Test ΔRMSE | 95 % CI（Test，配对簇 bootstrap，**RMSE 单位**） | p |
| --- | --- | --- | --- | --- | --- |
| `DMPNN_LR1E3_FIX` vs `C1_R1` | **+1.16 %** | **+2.91 %** | +0.0170 | [−0.0330, +0.0658] | 0.506 |
| `DMPNN_LR3E4_FIX` vs `C1_R1`（参考） | −12.35 % | −0.13 % | −0.0007 | [−0.0465, +0.0421] | 0.961 |
| `DMPNN_LR3E4` legacy vs `C1_R1` | −12.84 % | +7.35 % | +0.0429 | [−0.0086, +0.0943] | 0.106 |

**参数与复现细节**：两次 FIX 训练均从 commit `57a4777d` 启动，其 `code/lcpr_dmpnn.py` 的 sha256 为 **`f6994af0…`**
（与 legacy 的 `fb0618ef…` 不同，因为基准 lr 的修复就在这个文件里）。此后又加了一处**纯簿记补丁**
（只向 `train.json` 多写 `best_epoch`/`epochs_run`/`checkpoint_rule` 三个字段），使当前工作树的
`lcpr_dmpnn.py` 为 `66f4c7cc…`；该补丁**不改变训练语义**，但会使重跑时的 code hash 与 FIX run 不同，故在此记明。
两次 FIX run 的 `train.json` 因此仍无顶层 `best_epoch`（与 E5 所述历史行为一致），其 best epoch 取自 curve argbest。

两点必须一起说：

* **协议口径下"D-MPNN 距离"只有 +2.91 %（Valid +1.16 %），不是 +7.35 %**，而且区间跨 0。
  旧的 +7.35 % 属于**协议外**配方（有效峰值 9e-4、衰减底 3e-4），不能继续当 D1 的分数。
* `DMPNN_LR3E4_FIX` 在 100 epoch 内**明显欠训练**（Train RMSE 0.401，而 legacy D1 的 Train RMSE 为 0.265），
  所以"D1 用 3e-4 峰值"这条协议配置在这套预算下**本身也不成立**——这是事实记录，不是新结论。
  K 的 Test 0.5573 / Valid 0.5667 因此在**两条 split 上都不落后于**这个 D-MPNN 参考；
  但 D-MPNN 只有 1 个 seed、1 个宽度配置，仍是**单点坐标**，不是 D-MPNN 的上限。

---

## 6. 学习曲线（零新增训练）：K 的优势在哪儿

用已有 5 条完整 epoch 曲线的标量（`analysis/learning_curves.json`）。**不新建 checkpoint、不平均权重、
不把 Test 选出的 epoch 当独立结果；逐 epoch 相关的点不做 iid 显著性检验。**

Test RMSE（`train`/`valid` 见文件）：

| epoch | C1_R1 | C1_R3 | C1_CHEM8 | C1_CONTRIB | **K_COMBO** |
| --- | --- | --- | --- | --- | --- |
| 30 | 0.748 | 0.695 | 0.690 | 0.700 | 0.662 |
| 60 | 0.671 | 0.612 | 0.651 | 0.621 | 0.588 |
| 90 | 0.597 | 0.598 | 0.579 | 0.600 | **0.557** |
| 120 | 0.594 | 0.593 | 0.608 | 0.587 | 0.576 |
| 180 | 0.578 | 0.595 | 0.583 | 0.592 | 0.582 |
| 240 | 0.576 | 0.583 | 0.573 | 0.603 | 0.589 |
| 360 | 0.579 | 0.583 | 0.568 | 0.605 | 0.592 |
| 窗口中位数 60–120 | 0.5996 | 0.5897 | 0.5964 | 0.5904 | **0.5812** |
| 窗口中位数 120–240 | 0.5791 | 0.5824 | 0.5743 | 0.5966 | 0.5838 |
| 窗口中位数 240–360 | **0.5786** | 0.5831 | **0.5681** | 0.6039 | 0.5923 |

* K 的 Test 优势**只存在于 60–120 的窄窗口**；从 180 起 K 的 Test **不再优于 C1_R1**，
  240–360 窗口 K 反而更差（0.5923 vs 0.5786），而同一窗口 K 的 Valid 仍更好（0.6249 vs 0.6631）。
* 核对 checkpoint 选择位置：K 的 best-Valid 在 **epoch 90**，比 base 的 187 早 **97** 个 epoch。
  **在同一 epoch（187）上取 K 的 Test = 0.5847，比 C1_R1 的 0.5839 更差。**
  → K 的 Test 优势有相当一部分是"不同的 Valid 最小点落在不同位置"造成的，而不是"同一训练长度下更准"。
* `C1_CHEM8` 的 best-Test 在 226、best-Valid 在 206，late-window Test 中位数最好（0.5681）：
  同样存在"选择位置"效应，作为 K 的配对对照时也应当记住这一点。
* 同一 epoch 对照（均取 **187**，即 `C1_R1` 的 best-Valid 位置）：Test 为 K 0.5847 / R1 0.5839 /
  `C1_CHEM8` 0.5786 / `C1_R3` 0.5774；Valid 为 K 0.6235 / R1 0.6526 / CHEM8 0.6632 / R3 0.6579。
  即：**同一训练长度下，K 的 Test 不如两个更便宜的对照，但 Valid 好很多。**
  这是"两条臂各自在 Valid 上选点"的后果，而不是"K 更准"。

---

## 7. 收窄后的路线判断（研究者可讨论的决定）

### 7.1 必须区分的四个数（K，Lipophilicity，单 seed）

| 比较 | 口径 | Test | Valid | Test MAE | 结论 |
| --- | --- | --- | --- | --- | --- |
| K vs `C1_R1` | 基线（Valid 选出的 R1） | **+4.56 %**（p=0.069） | **+13.16 %**（p=0.084） | 0.4136 vs 0.4288（更好） | 相对"vanilla 基线"的坐标 |
| K vs `C1_R3` | 换配方（R3 的 Test 本来最好） | **+1.71 %**（p=0.573） | +13.28 % | 0.4134 vs 0.4123（更差） | **不是**纯架构效应 |
| K vs `C1_CHEM8` | **同 R1 配方**，只差贡献残差 | **+1.54 %**（ΔRMSE +0.0087，CI [−0.0271, +0.0440]，p=0.638） | +13.33 %（ΔRMSE +0.0872，CI [−0.0094, +0.2028]，p=0.128） | 0.4136 vs 0.4095（更差） | 同配方增量：点估计小、区间跨 0 |
| K 在 base 最差 5 % 行被删后 | base 定义 drop 集合 | **+1.13 %** | **−2.37 %** | — | 尾部集中 |

→ 可说的话：**K 在 Lipophilicity 上有一个强而一致的 Valid 侧改善与一个小而不显著的 Test 侧改善，
后者集中在 base 最难的行上，且换成同配方对照后只有 ~1.5 %。**

### 7.2 其它必须收窄的原句

* **A0 桥接**：必须与 `A0_R0` 比（Valid 0.69954 → 0.68537，Test 0.58010 → 0.57565，两个 split 都更好），
  **不能**拿 `C1_R1` 当参照去否定桥接。原表里"弱（−5.0 % / +1.4 %）"是与 C1_R1 比出来的，口径不对。
* **WIDE**：`WIDE_LOCAL` 只排除了**这一种增容配置**；**LONG** 改了调度 horizon（720 epoch）。
  两者都不能证明"模型普遍不欠容量 / 预算无价值"。
* **迁移**：ESOL/BACE/FreeSolv/HIV 都**没有**建立跨任务收益。HIV 点差极小（Test ROC-AUC −0.0011）
  只能说**未见增益**，不能说"已证明等价"。
* **单 seed**：筛选后 bootstrap 的区间跨 0 **不等于**"无效"；也**不为追 p<.05 自动加 seed**。
* 不接受"换 Validation 不训练就完成交叉验证"；官方划分保持不动（不重划分、不扩 Valid、不训练跨骨干 K）。

### 7.3 决定

* **K 保留为 Lipophilicity 候选，不升级为通用 LCRC-B。**
  理由是它确有跨 split 的正向信号（Valid 稳定为正，且 3 个窗口都更好），但
  ① Test 增益与同配方增量都很小且区间跨 0；② 增益集中在难行（砍 5 % 后 Valid 变负、Test 仅 +1.1 %）；
  ③ 它的 Test 优势依赖 epoch 选择位置。下一轮若继续 K，**必须**多 seed + 固定 epoch 对照 + 同时报 MAE 与尾部口径。
* **最值得继续的研究问题**（按可证伪性排序，仍不属本轮结论）：
  1. **K 的"难行收益"是不是真的**：把 K 与 `C1_CHEM8` 的配对增量做成 3 seed、预先固定 drop-set 口径，
     并同时报告 Q1–Q3（容易行）是否系统性变差。这是本轮唯一在修正后仍然**方向一致**的信号。
  2. **分子支撑假说（E2）仍未被检验**：v2 支撑表已经算好，`root-support legacy` 的四个臂不能复用；
     值不值得再买一次 S 搜索**目前未定**（需先决定是否接受"收缩本身很轻、天花板很低"的先验）。
  3. **D-MPNN 的协议配方本身需要重定**：`D1F`（峰值 3e-4、100 epoch）欠训练，
     说明"100 epoch + 3e-4"这组协议超参不是可用的参考坐标；正确参照应当先固定预算再谈距离。
  4. **辅助监督**：需要保存训练期梯度（共享主干、加权）才能回答"优化负担"，本轮只能撤回不能替代。

---

## 8. 产物与受影响清单

| 新增（本文件相关） | 内容 |
| --- | --- |
| `analysis/corrected_manifest.json` | 25 条 Lipo 臂逐字段身份 + 权重重播（3 条 `VERIFIED` / 2 条 D-MPNN `VERIFIED_METADATA` / 20 条 `NOT_RETAINED`）+ 旧表缺陷 |
| `analysis/run_table_corrected.{json,csv}` | 37 条正式训练（35 旧 + 2 新），由原始记录重建的机器表（`best_epoch` / `epochs_run` / `git_commit` / `optimizer_base_lr` / `optimizer_updates` / checkpoint 保留标记） |
| `analysis/tail_corrected.json` | E1 的 6 个 pair × 3 split 全口径（含五分位、最大改善/恶化行、累计 ΔSSE） |
| `analysis/figures/tail_corrected_cumulative_dsse.png` | 按 base |error| 降序的累计 ΔSSE（K vs C1_R1，Valid/Test） |
| `analysis/paired_corrected.json` | K vs CHEM8/R3/R1 与 D-MPNN 各配对 bootstrap |
| `analysis/support_v2_summary.json` + `cache/support_v2_*.npz` | E2 的分子支撑表、κ / a_bar / α 分布、与旧值差异 |
| `analysis/grad_checkpoint_diagnostic.json` | E3 的 `CHECKPOINT_DIAGNOSTIC`（共享主干、加权、clip 系数）+ 旧标量的加权修正 |
| `analysis/dmpnn_schedule_audit.json` | E4 玩具追踪、实际 `lr_end`、code hash 对齐、两个新 run |
| `analysis/learning_curves.json` | 固定 epoch 与窗口中位数、各模型的选择位置 |
| `code/lcpr_correction.py`, `tests/test_lcpr_corrections.py` | 修正实现与 14 个回归测试 |
| `runs/DMPNN_LR3E4_FIX__Lipophilicity`, `runs/DMPNN_LR1E3_FIX__Lipophilicity` | E4 的两次新训练（其余旧 run 目录未改） |
| `analysis/errata_reruns/` | 上述两次训练的 `train.json`/`curve.csv`/`predictions_best.npz` **逐字节副本**（因为 `runs/` 被 git-ignore，新证据不能只留在本机磁盘上）+ README |

`REPORT.md` 中被本文件修正或收窄的原句已在原处加 `【勘误 Ex】` 标注；旧表 `analysis/tail_summary.json`、
`analysis/run_table.{json,csv}`、`analysis/paired_comparisons.json` **保持原样**，标注 `SUPERSEDED`，
以便任何人核对差异。

**成本**：本地 CPU（准备/面板/重算/写作）不计；新增 GPU 训练 2 次、**0.0413 A100·h**（进程墙钟 78 s + 81 s，
各 1 张 A100），远低于 0.5 GPU·h 与 2 h 墙钟上限。
