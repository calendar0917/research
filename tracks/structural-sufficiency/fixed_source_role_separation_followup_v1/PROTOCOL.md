# FSR follow-up v1 — RSSC 短程延续验证（协议，训练前冻结）

- Round: `tracks/structural-sufficiency/fixed_source_role_separation_followup_v1/`
- Round ID: `fixed-source-role-separation-followup-v1`；Protocol id `fsf-v1`
- 只读复用：`fixed_source_role_separation_pilot_v0/`（RSSC 的 epoch-12 `checkpoint_last.pt`、曲线、逐任务结果、机制诊断）
- 历史比较坐标：`direct_structural_computation_recurrence_v0/results/runs/FIXED1_seed0` 的真实 seed-0 **25-epoch prefix**（不是 W2，不是 W2_JOINT）
- 数据：`ogbg-molpcba` 官方 scaffold split（Train 350343 / Valid 43793 / 128 tasks）；seed 0；**Test 不读取、不评价**
- 本轮只回答一个问题：**在完全相同的 25-epoch horizon 下，RSSC 能否超过原始 FIXED1 的实际预测收益？**

本文件在正式训练开始前写入。开始训练后不修改模型、horizon、评价口径、比较阈值或判读规则。

## 1. 继承与恢复

- 模型、数据和训练计算逐字继承 RSSC v0 / FIXED1：rank 32、两个低秩角色分离 block、原始 FIXED1 主干与 `post_local`、masked BCE、AdamW、dropout 0.10、batch 128、原始 40-epoch per-step cosine（horizon 只取前缀，不压缩、不重启 LR）。
- 从 RSSC v0 的 **`checkpoint_last.pt`（epoch 12, global step 32856）** 恢复：model state、optimizer state、scheduler/global step、Python/NumPy/torch CPU/CUDA RNG、原始 40-epoch order plan、已完成 epoch 与历史最佳指标（best epoch 12, best Valid AP 0.2670286554582952）。
- **不从 `checkpoint_best.pt` 重新启动训练逻辑**，不重新初始化 optimizer，不重启或压缩 LR schedule。若必要恢复状态缺失：`RESUME_STATE_MISSING`，停止，不从头训练。
- 历史目录只读；新产物全部写入本轮目录。

## 2. 训练与评价

- 从 epoch 13 继续到 epoch 25（13 个 epoch，35,594 次更新；global step 32856 → 68450）。
- 每 epoch 全量官方 Valid AP/BCE（127 个可评分任务）；在 epoch **16、20、25** 评价完整 Train AP/BCE；结束时补评最佳 checkpoint 的完整 Train AP/BCE。
- 合并 v0 epoch 1–12 与新增 epoch 13–25，按完整 25-epoch prefix 中最高官方 Valid macro AP 选最佳 checkpoint；平局取更早 epoch。若 epoch 12 仍最佳，则最佳 checkpoint 就是 v0 的 epoch-12 best。
- 不运行 Test。训练/评价路径带运行时 Test 守卫（`split == "test"` 直接抛错）。

## 3. 恢复正确性检查（正式训练前）

使用已有验证代码，不做重复的大规模 preflight：

1. checkpoint manifest 与 SHA-256；epoch-12 checkpoint 的 Valid 重放与历史 AP/BCE、逐任务 AP、logits 指纹一致。
2. optimizer 全部 69 个 state 的 step = 32856、moment 形状有限；param group 与 recipe 一致；scheduler global step = 32856、总步数 = 109520。
3. 下一次更新的 LR = `scheduled_lr(32856)`，且 epoch 13–25 各 epoch 末 LR 与历史 FIXED1 曲线逐项一致；epoch 13 的 batch row IDs / row-order hash 与原始 40-epoch plan 的历史记录完全一致；epoch 13–25 全部为 Train 行、无 Valid/Test 行。
4. 恢复后参数指纹等于 checkpoint；RNG hash 等于 checkpoint 保存的 RNG；一次真实 step 的 loss/LR/grad-norm/更新后参数与独立 probe 一致（容差内）。
5. 数据身份（feature/norm/split/label/cache）、模型参数清点、masked BCE、recipe 与 checkpoint 所在提交逐项一致（计算路径文件 git blob 未变）。
6. Test 标签全为 NaN、plan 无 Test 行、本轮无任何 Test 评价。

任何一项硬失败 → 不启动正式训练，写 `RESUME_STATE_MISSING` / `RESUME_CHECK_FAILED`。

## 4. 预算门禁

正式训练前用真实 epoch-13 batch 测 20 个 step 的中位耗时，加上实测 Valid 评价与保守 Train 评价估计，投影剩余 13 epoch + 4 次 Train 评价 + 最终评估。若投影无法在墙钟剩余预算内完成 epoch 25，或 GPU-hour 超过 1.0，则不启动正式训练，记录 `BUDGET_LIMITED`。目标 GPU-hour 0.8，硬上限 1.0；墙钟硬上限 60 min。

## 5. 比较坐标与指标

主指标：官方全部可评分任务 macro AP（Valid）。

对照：真实 FIXED1 seed-0 25-epoch prefix（best Valid AP 0.27584552136653284；epoch-25 Valid AP 同值；best Train AP 0.5995324907052489；epoch-25 Train BCE 0.024222211976071915；高支持任务 AP 从 `per_task_ap.csv` epoch 25 复核）。

报告：

- 完整 Validation 曲线（epoch 1–25）与 best Valid AP；
- epoch-25 endpoint AP、最后五个 epoch 平均 AP；
- best checkpoint 的 Train AP/BCE、Valid BCE；
- Train–Valid gap 的变化；
- 高支持任务（Valid 正负例均 ≥20）AP；
- 127 个任务的逐任务改善/退化数、paired task bootstrap（2000 次，seed 0，95% 区间）、去掉绝对差最大 3 个任务后的均值；
- RSSC 与 FIXED1 的 best-checkpoint 逐任务差值表。

## 6. 判读（训练前冻结）

记 `Δbest = RSSC best Valid AP − FIXED1 25-prefix best Valid AP`，`Δend = RSSC epoch-25 AP − FIXED1 epoch-25 AP`，`Δlast5 = RSSC epoch 21–25 均值 − FIXED1 epoch 21–25 均值`，`Δhigh = 高支持任务 AP 差`，`Δtrain = RSSC best Train AP − FIXED1 best Train AP`。

- `PERFORMANCE_ADVANCE`：`Δbest ≥ +0.003` 且 `Δend ≥ −0.001` 且 `Δlast5 ≥ −0.001` 且 `Δhigh ≥ −0.001`。
- `REGRESSION`：`Δbest ≤ −0.005`，或（`Δbest ≤ −0.003` 且 `Δend ≤ −0.001` 且 `Δlast5 ≤ −0.001`）。
- `FIT_WITHOUT_GENERALIZATION_GAIN`：`Δtrain ≥ +0.010` 且 `Δbest ≤ 0` 且 `Δend ≤ 0`。
- `NEAR_BASELINE`：`−0.003 < Δbest < +0.003` 且不满足上述回归/拟合判定。
- `INCONCLUSIVE`：预算/恢复失败/训练不完整；或 `Δbest` 与 `Δend`、`Δlast5`、`Δhigh` 出现符号冲突且两侧幅度均 ≥0.003。

优先级：预算/完整性 → INCONCLUSIVE；FIT_WITHOUT_GENERALIZATION_GAIN；REGRESSION；PERFORMANCE_ADVANCE；NEAR_BASELINE；冲突 → INCONCLUSIVE。

单 seed 探索性结论，不是确认性泛化结论。若没有 `PERFORMANCE_ADVANCE`，按任务要求结束当前 RSSC 架构扩展路线；不追加 rank、gate、width、seed 或进一步延长训练。

## 7. 轻量诊断

只记录两个 block 的 `‖D‖F/‖W‖F`、低秩修正项与原始 `fc1(z+m)` 的 RMS 比、相对 epoch 12 的变化、Train–Valid gap 变化。不重跑完整结构贡献解释脚本，不重复冻结删除分支实验。

## 8. 交付物

`PROTOCOL.md`、`REPORT.md`、`analysis/resume_checks.json`、`analysis/timing.json`、`analysis/epoch_curves.csv`（1–25）、`analysis/main_table.csv`、`analysis/task_deltas.csv`、`analysis/paired_comparisons.json`、`analysis/mechanism_light.json`、`analysis/summary.json`、`results/manifest.json`、`runs/RSSC_resume_seed0/`（曲线、逐任务、checkpoint、run.json）。报告首页直接回答：补齐 horizon 是否改变了上一轮的研究判断。
