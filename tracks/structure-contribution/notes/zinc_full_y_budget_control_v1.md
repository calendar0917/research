# zinc-full-y-budget-control-v1 — Full-Y 训练预算对照（fit 7200 vs 官方 train 10000，seed 0）

Track: `structure-contribution` · 日期: 2026-10-11 · 协议: `configs/zinc_full_y_budget_control_v1.yaml`（冻结 commit `a5409ee`）
Driver: `code/run_full_y_budget_control_v1.py`（audit / replay / train / predict / score 五阶段）
Tests: `tests/test_full_y_budget_control_v1.py`（13 passed）
诊断探针: `code/probe_seed0_determinism_v1.py`（只读，不写任何 artifact）

> **本轮只回答两个问题**
> Q1：把 Full-Y raw-y-only 训练从 fit_inner 7200 扩到 official-train 10000（同代码/同超参/同 soup 规则/同评测），
> official-validation raw-y MAE 是否真的改善？改善是否达到预冻结的 A/B 带？
> Q2：即使改善成立，绝对水平（0.1186）离 0.09 / 0.08 目标仍有 +0.029 / +0.039 的差距
> ——是否仍需要一个新预测架构（而非继续买预算/调参）？

> **披露（预注册时已写明，此处再次置顶）**
> Δ 是**整个训练预算机制**的变化（行数 +38.9%、优化步数 +38.6%、fit-only 预处理重拟合），
> **不是** 2800 行标签的孤立因果效应；official validation 是历史轮次已暴露过的仪器
> （`tracks/ksvd`、`cscl-v3d` 均读过），本轮是同一仪器上的受控对照，不是新的盲测。

---

## 1. Phase A：历史 10k checkpoint 复用审计（`historical_reuse_audit.json`）

目的：本轮最多允许 2 条 seed-0 正式轨迹。若历史 10k Y-arm checkpoint 与当前协议**逐字匹配**，
则复用它可以只跑 1 条新轨迹（7200），把预算全部留给对照本身。

| 检查 | 结果 |
|---|---|
| runner 源码 SHA vs `frozen_eval_manifest.json` | 匹配（`7a86a135…b063fc723`） |
| `Y_seed0_raw_soup_state.pt` / `all_train_prep.npz` SHA | 匹配 `7792c560…` / `a7dbcc43…`（last/init state 同样匹配） |
| soup state hash vs `Y_seed0.json` 记录 | 匹配 `6c1e81ce…`；soup members [236..240] |
| 训练配方逐项（epochs 240 / steps 18960 / lr 1e-3 / wd 1e-5 / clip 5 / batch 128 / λ 33.95873017865987 / shuffle seed 101） | 全等 |
| AST 审计: 训练解压只读 `decomp["y"]`、`decomp["g"]`；c/k/gid 仅出现在 `np.savez_compressed` 导出路径 | 通过 |
| 重建损失 `reconstruction_loss(phi, coord)` 标签无关 | 通过 |
| 参数量 408651（两 prep） | 通过 |
| init state 差异键 | 仅 `['D','U','common_rms']` 数据 buffer；**参数逐位相同** |
| 非 GNN 扫描（无 message passing / attention / 新分支） | 通过 |
| prep7200 `fit_idx == fit_inner`；all-train prep `fit_idx == arange(10000)` | 通过 |
| y 对齐（缓存 y == decomp y == 官方数据集 y） | 逐位相等（max diff 0.0）；g+c−y < 1e-9 |

**全行 forward replay 门**（`protocol_replay.json`）：用当前代码对历史 soup 在全部 10000 训练行上重建
预测并与历史导出 `Y_seed0_fit_pred.npz` 比对
`max|dpred| = 5.72e-06`（本地）/ `3.82e-06`（res），`MAE diff = 5.7e-09` → ok。

结论：**REUSE_10K**。10k 臂 = 历史 seed-0 Y checkpoint（`Y_seed0_raw_soup_state.pt`，soup 236–240），
本轮只新增 1 条轨迹（7200 臂）。历史 F7200 的 monitor-top-5 soup 仅作背景参考，不作为臂。

## 2. 训练（res / a100-2 / pool res-gpu1 / cuda:0 / 240 epochs / FP32 / 无 AMP·scheduler）

| | F7200（新跑） | F10000（复用历史） |
|---|---:|---:|
| 训练行 | 7200（fit_inner） | 10000（official train） |
| batches/epoch × epochs | 57 × 240 | 79 × 240 |
| steps | 13,680 | 18,960 |
| wall（GPU1） | 1206.8 s（5.03 s/epoch） | 1372.8 s（5.56 s/epoch） |
| init state sha256 | `fc69a75a…` | `93c2f23c…`（仅数据 buffer 不同） |
| soup（236–240）state sha256 | `8382c49c…` | `6c1e81ce…` |
| fit-only MAE（soup 在各自 fit 集） | 0.038478 | 0.038638 |
| 内部 monitor / dev MAE | 0.114280 / 0.113951 | 0.118258 / 0.116413（历史记录） |
| official valid raw MAE | **0.134058** | **0.118618** |

F7200 运行：`rr run res full-y-budget-f7200 --pool res-gpu1`，run_id `full-y-budget-f7200-20261011-204542-8d5fb6d5`，
commit `92462c3`，exit 0，device `cuda:0` = `NVIDIA A100-SXM4-40GB`，`official_valid_loaded=false`、`official_test_loaded=false`。
F10000 元数据 `F10000_train_or_reuse_meta.json` 记录复用哈希与 step 数（18960）。

## 3. Phase B：冻结预测 + 冻结评测程序，然后只读一次 official-valid y

`predict`（本地 CPU，**不读任何 y**）：
- F7200 行：`rpv2.apply_prep_fit7200` + `rpv2.predict_raw`（新 last-5 soup）
- F10000 行：`zftd.apply_prep_train_only` + `rpv2.predict_raw`（历史 soup）
- 写出 `F7200_valid_preds.npz` / `F10000_valid_preds.npz` / `valid_predictions.npz`
  （sha256 `f8e9abb3…` / `e59400cd…` / `f82cd75d…`），随后写 `freeze_manifest.json`
  （checkpoint/prep/数据/预测文件哈希 + `extraction_program_sha256` + `config_sha256`）。

**评测路径验证（都是"重放已记录的旧数字"，不涉及本轮臂选择）：**

| 交叉检查 | max\|Δ\| |
|---|---:|
| F10000 CPU 重放 vs 历史 `heldout_valid_predictions.csv` 的 `Y_raw_0` | **0.0**（逐位相等） |
| 历史 F7200 monitor-top-5 soup CPU 重放 vs `cscl_v3d_official_eval/valid_F_preds.npz` | **0.0**（逐位相等） |
| official-valid y vs 历史 CSV `y` 列 | 0.0 |
| 行对齐（dict_atom == 数据集 x） | 0 mismatches（1000/1000） |

即：本轮的 F10000 预测与历史导出**完全一致**，0.118618 这个数字由本轮代码独立复现；
两个臂走的是同一条特征/预测路径。

`score`：预测文件哈希复核通过后才读 y（ZINC official val，1000 行）。读取次数 = **2**：
第 1 次在组装 `decision.json` 时因 band 标签 KeyError 崩溃（指标已算出、阈值/规则未变），
第 2 次为最终运行；两次都使用同一批 sha256 冻结的预测文件。该披露写在 `decision.json.scoring_provenance`。

## 4. 结果（official validation，1000 分子，raw-y 单位）

| 指标 | F7200 | F10000 |
|---|---:|---:|
| **MAE** | 0.134058 | **0.118618** |
| RMSE | 0.668938 | 0.653580 |
| 中位绝对误差 | 0.072359 | 0.060427 |
| p90 / p95 | 0.247060 / 0.359641 | 0.224687 / 0.308601 |
| max 绝对误差 | 20.1360 | 19.9057 |
| top-10 误差质量占比 | 22.5% | 25.3% |

**配对结果** `d_i = |e_F10000,i| − |e_F7200,i|`（5000 次重采样，seed 0）：

- Δ = **−0.015440**，bootstrap 95% CI **[−0.022937, −0.008017]**（不含 0）
- F10000 更好的分子 592/1000（59.2%），F7200 更好 408（40.8%）
- 预冻结带：Δ ≤ −0.010 → **band A（budget_gain_clear）**；CI 不跨 0，`ci_conflict=false`，
  `final_label = budget_gain_clear`
- 极端行稳健性：去掉历史已知极值行 `valid:0172`（y=−20.34，单行占 F7200 误差质量 15.0%）
  Δ' = −0.015225（`not_dominated_by_extreme_rows=true`）；去掉两臂各自最大误差行同样 −0.015225

**子群**（见 `subgroup_analysis.json`）：

| 子群 | n | MAE F7200 | MAE F10000 | Δ |
|---|---:|---:|---:|---:|
| y ≤ −3 | 65 | 0.548284 | 0.523922 | −0.024362 |
| y > −3 | 935 | 0.105262 | 0.090441 | −0.014820 |
| 原子数 ≤ 20 | 266 | 0.138956 | 0.139336 | **+0.000381** |
| 原子数 > 20 | 734 | 0.132283 | 0.111109 | −0.021174 |
| 环秩 0–1 | 76 | 0.153986 | 0.133407 | −0.020578 |
| 环秩 2–3 | 711 | 0.129815 | 0.119728 | −0.010087 |
| 环秩 ≥ 4 | 213 | 0.141112 | 0.109634 | −0.031478 |
| y 四分位 q1…q4 | 250 each | 0.2311 / 0.1319 / 0.0989 / 0.0744 | 0.2155 / 0.1181 / 0.0799 / 0.0610 | −0.0156 / −0.0138 / −0.0190 / −0.0133 |

改善在 y 带、环秩、四分位上一致；**唯一例外是小分子（≤20 原子）**：+0.0004（无收益）。
误差质量高度集中：去掉 `valid:0172` 后 F7200 0.114036 / F10000 0.098810（由已记录数字算术推出，
未再读 y）。

## 5. 绝对水平：Q2

| | valid MAE | 距 0.09 | 距 0.08 |
|---|---:|---:|---:|
| F7200 | 0.134058 | +0.044058 | +0.054058 |
| F10000 | **0.118618** | **+0.028618** | **+0.038618** |

- 0.118618 与历史 F10000 记录**完全相同**（该臂复用历史 checkpoint，评测路径逐位复现）。
- 背景（非本轮臂，规则不同，仅参考）：
  - 历史 F7200 monitor-top-5 soup valid raw = 0.132953（比本轮 F7200 last-5 好 −0.001105，
    即 soup 选择规则在 7200 臂上的 valid 波动量级 ≈ 1e−3）；
  - 匹配对照必须用同一 soup 规则，所以主数字用 last-5（协议禁止用 monitor/valid 选 soup）。
- 想达到 valid 0.09，还差 **0.0286**（相对 24%）；达到 0.08 差 0.0386（相对 33%）。
  历史 official-test 的 Full-Y 数字（0.094589）属于旧配置，**不能**用来宣称本轮配置达到 0.09；
  本轮**未评测、也不会评测** official test。

## 6. 威胁与披露

1. **Δ 的语义**：整个预算机制（行 +38.9%、步数 +38.6%、预处理重拟合），不是 2800 行的孤立因果效应。
2. **仪器暴露**：official validation 在历史轮次已被读取（ksvd / cscl-v3d / strong 相关轮），
   本轮是受控对照；不存在"未见过的 valid"。
3. **轨迹级随机性（本轮新证据）**：历史 F7200 与本次 F7200 同配方/同 seed/同 GPU/同 prep，
   但 ep1 train MAE 不同（0.906464 vs 0.872567）。`probe_seed0_determinism_v1` 在同一进程内
   连续跑两个**重新 seed** 的 1-epoch 复制：
   - init state hash 两次相同（`fc69a75a…`，且与正式运行的 init 相同 → 初始化可复现）；
   - 首个 batch 的 loss 逐位相同，但 step 之后的权重哈希已不同、epoch-1 MAE 已相差
     **0.910273 vs 0.902745**（正式 run 0.872567）——即同一 GPU 上"同 seed 重跑"的 epoch-1
     差异（~0.008–0.04）本身就与历史/本次差异同量级，来源是 GPU 端 kernel 非确定性（如
     `index_add_` 类原子累加）经 Adam 类符号式更新放大；
   - 本地 CPU 两个复制**逐位相同**（0.876541 两次，全部哈希相同）→ 非确定性来自 GPU 执行路径。
   影响：Δ 是"臂级"对照，不能被解读为"同 seed 逐位可复现的受控变量"，也不能把单臂 valid
   差异全部归因于预算；该随机性同样作用于两臂，且 CI 已含配对噪声。
4. **单 seed、单轨迹**：无 seed 1、无第三次轨迹、无超参搜索（协议禁止）；因此无法从本数据
   估计 seed 间方差，只能用配对 bootstrap 与上述轨迹噪声探针定性说明。
5. **小分子子群无效**：扩预算的收益集中在 ≥20 原子的分子；小分子（n=266）Δ≈0。
6. **official test**：本轮完全未访问；任何 test 结论都不属于本配置。
7. **`valid_predictions.npz` 不含 y**（`y_in_file=false`）；y 只在 score 阶段读入内存。

## 7. 结论

- **Q1：是，预算收益真实存在，且达到预冻结的 A 带。** F10000 − F7200 = **−0.015440**
  [−0.022937, −0.008017]（5000 次配对 bootstrap，seed 0），59.2% 分子上更好；
  子群方向一致；对极端行稳健。作为对照的"准匹配"历史数字（+0.014797）被本轮严格匹配的
  −0.015440 取代（差异来自 soup 规则与噪声，见 §6.3）。
  但它只是"更多数据/更多步数"的产物：fit MAE 两臂几乎相同（0.038478 vs 0.038638），
  而 monitor→valid 的泛化缺口才是绝对水平的主要来源。
- **Q2：是，绝对性能仍远未达标，需要新的预测架构/表示，而不是继续买预算。**
  最好可达 valid raw MAE = 0.118618，距 0.09 差 0.0286（24%），距 0.08 差 0.0386；
  20.5% 的误差质量来自 top-10 行，单行 `valid:0172`（y=−20.34）≈15%——误差是重尾的，
  说明缺口不是"平均拟合不够"，而是少数极端/长尾结构上的系统性失败；
  小分子子群完全不吃预算红利。
  因此下一轮的单一问题是**预测架构/表示**（在 y-only 合同内），而不是预算、超参或 soup 选择。

## 8. 产物

`results/zinc_full_y_budget_control_v1/`（git-ignored）：

| 文件 | 内容 |
|---|---|
| `historical_reuse_audit.json` | Phase A 硬检查 + 配方等值 + REUSE_10K 决策 |
| `protocol_replay.json` | 10000 行 forward replay 门（max\|dpred\| / MAE diff / ok） |
| `data_prep_audit.json` | 两 prep 的 fit_idx / 参数量 / init 差异键 / y 对齐 |
| `F7200_train_meta.json` | 新 7200 臂：曲线、soup、init/soup 哈希、wall、内部 MAE |
| `F10000_train_or_reuse_meta.json` | 复用历史 10k 臂的哈希/配方/steps 记录 |
| `F7200_last5_soup_state.pt` | 新臂 checkpoint（sha256 `baf2e6b8…`） |
| `F7200_fit_pred.npz` | 新臂 fit 集预测（内部诊断） |
| `valid_predictions.npz` + `F7200/F10000_valid_preds.npz` | 冻结的 official-valid 预测（无 y） |
| `freeze_manifest.json` | 预测/checkpoint/prep 哈希 + 评测程序哈希 |
| `valid_metrics.json` | 主指标、分位数、fit MAE、背景、标签交叉核验、阈值差距 |
| `paired_bootstrap.json` | 配对有符号差、CI95、重采样数/seed、更好分子比例 |
| `subgroup_analysis.json` | 子群 + 极端行稳健性 + `valid:0172` 诊断 |
| `decision.json` | 预冻结带判定、CI 冲突标记、绝对目标差距、读取披露 |
| `seed0_determinism_probe_{gpu,cpu}.json` | §6.3 的 GPU/CPU 复现性探针原始输出 |
