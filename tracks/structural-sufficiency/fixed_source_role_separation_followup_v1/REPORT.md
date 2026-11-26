# FSF v1 — RSSC 25-epoch 短程延续验证报告

Round ID：`fixed-source-role-separation-followup-v1`；协议 `fsf-v1`
数据：`ogbg-molpcba` 官方 scaffold split（Train 350343 / Valid 43793 / 128 tasks），seed 0，**Test 未读取、未评价**
只读来源：`fixed_source_role_separation_pilot_v0/runs/RSSC_seed0/checkpoint_last.pt`（epoch 12, step 32856）
历史坐标：`direct_structural_computation_recurrence_v0/results/runs/FIXED1_seed0` 的真实 seed-0 **25-epoch prefix**

## 首页结论

**补齐 25-epoch horizon 没有改变上一轮的研究判断。**

上一轮（12-epoch prefix）的判定是 `ROLE_DIFFERENCE_WITHOUT_PERFORMANCE_ADVANCE`：RSSC 优于 TIED，但没有超过原始 FIXED1。本轮把 horizon 补到 25 后，RSSC 的 best Valid AP 为 **0.27602765**，FIXED1 25-epoch best 为 **0.27584552**，差 **+0.000182**；高支持任务反而低 **−0.001352**；127 个任务 63 改善 / 64 退化；paired task bootstrap 95% 区间 **\[−0.00599, +0.00616\]** 跨零。按预注册阈值判定 **`NEAR_BASELINE`**，不是 `PERFORMANCE_ADVANCE`。

因此：**不追加 rank、gate、width、seed，不进一步延长训练；结束当前 RSSC 架构扩展路线。**

一个需要明确记录的细微点：RSSC 的训练拟合确实进一步上升（best Train AP `0.609293` vs FIXED1 `0.599532`，**+0.009760**；Train BCE 也更好），但 Validation 只得到 `+0.00018` 的 best AP，Train–Valid AP gap 从 `0.32369` 扩大到 `0.33327`。方向上接近“拟合增强但泛化未同步改善”，但预注册的 `FIT_WITHOUT_GENERALIZATION_GAIN` 阈值要求 `ΔTrain AP ≥ +0.010`，实际 `+0.009760` 未达到，故正式标签保持 `NEAR_BASELINE`，不做事后阈值修改。

## 1. 恢复与正确性验证（正式训练前）

`analysis/resume_checks.json`：**9/9 通过**；`analysis/timing.json`；`analysis/attempts.json` 记录三次 checks-only、两次正式启动（其中 4 次在训练前停止，最后 1 次完成）。

| 检查 | 结果 |
|---|---|
| C1 checkpoint identity | v0 `checkpoint_last.pt` SHA-256 `68aa67f6…`，与 v0 manifest 一致；epoch 12 / step 32856；75 个 state tensor；1,153,044 参数 |
| C2 epoch-12 重放 | Valid AP `0.26703045` vs 记录 `0.26702866`（Δ `1.79e-6`）；BCE `0.045210025835` vs `0.045210025863`（Δ `2.8e-11`）；127 任务逐任务 AP 最大差 `1.86e-4`；同进程重复重放的 AP 差 `8.6e-7`、逐任务最大差同为 `1.86e-4`，说明逐任务差异是 CUDA scatter 顺序级评价噪声，不是权重不一致（state fingerprint 逐位相等） |
| C3 optimizer | 69 个 state 全为 step 32856；moment 形状匹配、有限；与 checkpoint 逐位相等；单 param group，betas/eps/wd 与 recipe 一致 |
| C4 scheduler / next LR | global step 32856；总步数 109520；下次更新 LR `0.00024670457946132966`，严格小于 epoch-12 末 LR `0.00024670788387859356`；epoch 13–25 各 epoch 末 LR 与历史 FIXED1 曲线差 **0.0**（未压缩、未重启 schedule） |
| C5 order plan / 下一次 batch | 全局 plan `06f437…b6f6`；epoch 13 的 row-order hash、batch-plan hash、前 128 个 row IDs、batch 数、末 batch 大小均与历史 DSCR `training_order.json` 完全一致；epoch 13–25 全部为 Train 行，Valid/Test 行数 0 |
| C6 data identity | feature/norm/split/label/cache 与 v0 记录逐项相等；Train/Valid/Test 行数 350343/43793/43793；Test 标签全 NaN |
| C7 model/loss/recipe | 1,153,044 参数、rank 32、dropout 0.10、width 128、batch 128、AdamW/weight decay/clip 与 checkpoint 一致；数据/模型/损失/训练计算相关文件的 git blob 与 checkpoint 提交 `5517f5ae…` **全部相同** |
| C8 参数 / RNG | 恢复后 state fingerprint 等于 checkpoint；Python/NumPy/torch CPU/CUDA RNG hash 与 checkpoint 保存值逐位相等；不等于随机初始化指纹 |
| C9 Test | plan 无 Test 行、Test 标签全 NaN、checkpoint `test_scored=false`；本轮所有 evaluation 调用带 `split == "test"` 运行时守卫 |

**下一步连续性**：独立 probe 与正式训练的第一 step 对比，loss 绝对差 `1.86e-9`，LR 差 0，grad-norm 差 0，更新后参数 `state_matches_probe=true`（逐 tensor 最大差 ≤1e-5），RNG 恢复 hash 与 checkpoint 相等。即没有重新初始化 optimizer/参数/RNG，也没有从 `checkpoint_best` 重启训练逻辑。

## 2. 训练与预算

- 从 epoch 13 继续到 25，共 13 个 epoch / 35,594 次更新；global step 32856 → 68450。
- 每 epoch 全量 Valid AP/BCE；epoch 16、20、25 全量 Train AP/BCE；结束时补评 best checkpoint（epoch 25）的完整 Train AP/BCE。
- 所有计算继承 RSSC v0：rank 32、两个低秩 block、原始 FIXED1 主干与 `post_local`、masked BCE、原始 AdamW、dropout 0.10、batch 128、原始 40-epoch cosine 前缀。
- 实测 step 中位 **50.92 ms**，epoch train 投影 `139.43 s`；实际 13 个 epoch 的 train 时间合计 `1797.97 s`（30.0 min），逐 epoch `128.2–146.0 s`，Valid 合计 `104.66 s`（6.8–9.9 s/epoch）。
- 预算门禁投影剩余 `40.25 min`，实际剩余 `35.3 min`；偏差 `−4.9 min`（约 −12%，投影保守）。
- **正式 run GPU 时间 `0.5889 GPU-h`**（`gpu_seconds=2120.2`；job 墙钟 35.6 min）。加上 checks-only/失败尝试/GPU 探测，总计约 **0.64 GPU-h**，低于 0.8 目标、远低于 1.0 硬上限。
- 本轮从任务开始（03:09 UTC）到分析完成约 **59 min**，在 60 min 墙钟内；恢复检查 03:16–03:28，正式训练 03:28–04:04，分析至 04:08。正式训练 35.6 min，其余为恢复验证、预算门禁、拉取与分析。
- 未运行 Test；未做多 seed、蒸馏、集成、超参数搜索；只使用 `res-gpu1`，未抢占 `res-gpu0` 上其他用户的进程。

## 3. 性能结果

主指标始终是**官方全部可评分任务 macro AP（Valid）**。

| arm | best epoch | best Valid AP | endpoint Valid AP | last-3 mean | last-5 mean | best Train AP | best Train BCE | Valid BCE at best | params | GPU-h |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RSSC (resumed 13→25) | 25 | **0.27602765** | **0.27602765** | 0.27513785 | 0.27397061 | 0.60929280 | 0.02382143 | 0.04654344 | 1,153,044 | 0.589 |
| FIXED1 (historical 25 prefix) | 25 | 0.27584552 | 0.27584552 | 0.27426651 | 0.27335849 | 0.59953249 | 0.02422221 | 0.04637688 | 1,128,468 | — |

`analysis/main_table.csv`、`analysis/epoch_curves.csv`（RSSC 与 FIXED1 各 25 行）；完整 45 列训练曲线在 `runs/RSSC_resume_seed0/train_curve.csv`（含 v0 epoch 1–12 与新增 13–25，`source` 列区分）。

曲线层面的关键事实：

- 12-epoch prefix 仍复现上一轮：RSSC `0.26702866` vs FIXED1 `0.26999159`，**−0.00296293**。
- 13–25 段 RSSC 多数 epoch 略高；但最终 best/endpoint 只高 `+0.000182`。
- last-3 mean `+0.000871`，last-5 mean `+0.000612`：后期曲线略好，但幅度远低于 +0.003 的实际收益门槛。
- 高支持任务（Valid 正负例均 ≥20，107 个任务）RSSC `0.29713254` vs FIXED1 `0.29848497`，**−0.00135244**；52 改善 / 55 退化。
- Valid BCE 略差 `+0.00016657`；Train BCE 略好 `−0.00040078`。
- Train–Valid AP gap：RSSC `0.33326515` vs FIXED1 `0.32368697`，扩大 `+0.00957818`；BCE gap 扩大 `+0.00056735`。即拟合更强、泛化没有同步。

逐任务（`analysis/task_deltas.csv`，best checkpoint 对 best checkpoint）：

| 统计 | 值 |
|---|---:|
| 可配对任务 | 127 |
| 平均 ΔAP | **+0.00018213** |
| 中位 ΔAP | −0.00016653 |
| 改善 / 退化 / 持平（>1e-6） | **63 / 64 / 0** |
| 改善 >0.005 / 退化 <−0.005 | 43 / 49 |
| ΔAP 标准差 | 0.03516 |
| task bootstrap 2000 次 95% CI | **\[−0.00598929, +0.00616344\]** |
| bootstrap 重采样中正比例 | 0.516 |
| 去掉 \|Δ\| 最大 3 个任务后均值 | **−0.00049474**（去掉 task 2/108/31） |

最大绝对差任务：task 2 `−0.18182`（低支持，正例 16）、task 108 `+0.16154`（正例 1）、task 31 `+0.10476`（正例 6）。去掉这三个任务后平均差仍为负。逐任务重排其实很明显（43 个任务改善 >0.005、49 个退化 >0.005，ΔAP 标准差 0.0352），但正负几乎相抵，macro AP 只差 `+0.00018`；因此不存在可重复的、方向一致的实质收益。

## 4. 轻量机制诊断（仅记录，不重跑完整解释）

`analysis/mechanism_light.json|csv`；best checkpoint（epoch 25）与 epoch 12 对比：

| 量 | epoch 12 | epoch 25 (best) | 变化 |
|---|---:|---:|---:|
| `‖D‖F/‖W‖F` block0 | 0.49440 | 0.52020 | +0.02581 |
| `‖D‖F/‖W‖F` block1 | 0.49499 | 0.52291 | +0.02792 |
| `cos(D, W)` block0 / block1 | −0.143 / −0.133 | −0.152 / −0.128 | 基本不变（仍近正交/轻微负相关） |
| 修正/fc1 RMS 比 block0 | 0.34278 | 0.33972 | −0.00306 |
| 修正/fc1 RMS 比 block1 | 0.35957 | 0.37888 | +0.01931 |
| `‖W‖F` block0 / block1 | 14.73 / 14.81 | 16.47 / 16.74 | 主干权重尺度增大 |
| `‖D‖F` block0 / block1 | 7.28 / 7.33 | 8.57 / 8.75 | 低秩项尺度增大 |

低秩角色分离分支在 25 epoch 后仍然实际参与计算（`R≈0.52`，修正/fc1≈0.34–0.38），但它的继续训练只把 Train AP 推高约 0.01，没有转化为可重复的 Validation 收益。按本轮范围，**没有重跑完整结构贡献解释脚本，也没有重复冻结删除分支实验**。

## 5. 判定

按训练前冻结的规则：

- `PERFORMANCE_ADVANCE` 需要 `Δbest ≥ +0.003`，且 endpoint / last5 / high-support 不明显冲突。实际 `Δbest=+0.000182`，不满足。
- `REGRESSION` 需要 `Δbest ≤ −0.005`，或 `Δbest ≤ −0.003` 且 endpoint/last5 同时落后。实际不满足。
- `FIT_WITHOUT_GENERALIZATION_GAIN` 需要 `ΔTrain AP ≥ +0.010` 且 `Δbest ≤ 0` 且 `Δend ≤ 0`。实际 `ΔTrain=+0.009760`、`Δbest=+0.000182`，两个条件都差一点，不满足。
- 落在 `−0.003 < Δbest < +0.003` 且无预算/完整性失败、无符号冲突 → **`NEAR_BASELINE`**。

**正式判定：`NEAR_BASELINE`。** 这是单 seed 探索性结论，不是确认性泛化结论。25-epoch horizon 下 RSSC 与原始 FIXED1 的实际预测收益基本持平；没有值得增加架构复杂度的收益。按任务要求，结束 RSSC 架构扩展路线，不再追加 rank/gate/width/seed 或延长训练。

TIED 未延长，仍只保留原来 12-epoch 机制比较的身份；本轮没有把 W2 或 W2_JOINT 当作 FIXED1。

## 6. 偏差、失败尝试与产物

- **失败尝试（均未产生任何训练 epoch）**：C3 optimizer 比较实现错误；C2/C8 未在检查前恢复 RNG 且 logits-SHA 判据过严；第一次预算门禁使用双重 replay 时间和 2100 s cap 得到 `NO_GO`；第一次正式启动在 first-step parity 遇到 0 元素 buffer 的 `max()`。修复后正式 run 从 epoch 12 干净恢复。详见 `analysis/attempts.json`。
- **预算偏差**：投影剩余 40.25 min，实际 35.3 min；GPU 0.589 h（目标 0.8，硬上限 1.0）；总墙钟约 59 min（60 min 内）。没有因为预算截断 horizon。
- **评价噪声**：epoch-12 重放的 Valid AP 与历史差 `1.79e-6`，逐任务最大差 `1.86e-4`；同一进程重复重放的逐任务最大差也正好 `1.86e-4`，因此采用“AP/BCE/逐任务容差 + state fingerprint 逐位相等”的判据。主表使用训练时全量官方 Valid macro AP 曲线值（与 FIXED1 同口径）；best-checkpoint 补评的 AP 为 `0.27602646`，与曲线 `0.27602765` 差 `1.2e-6`。
- **产物**：`PROTOCOL.md`、`round/analysis/{resume_checks,timing,budget_projection,attempts,epoch_curves,main_table,task_deltas,paired_comparisons,mechanism_light,summary}.json|csv`、`runs/RSSC_resume_seed0/`（run.json、完整曲线、逐任务、best/last checkpoint、light mechanism）、`results/manifest.json`。checkpoint 大文件留在 `runs/`，路径 + SHA-256 见 manifest；未修改历史 v0 目录。
- **Test**：本轮没有读取或评价 Test；所有 checkpoint `test_scored=false`、`test_used_for_selection=false`。
