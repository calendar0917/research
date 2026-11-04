"""Render ``REPORT.md`` from the committed run facts and comparison tables."""

from __future__ import annotations

import csv

import wb_common as C

FORMAL_RUN_ID = "lcrc-pepweighted-p2-seed0-20261104-102145-361ca7ae"
WINDOW_UTC = "2026-11-04T02:21:48Z – 2026-11-04T02:57:45Z"
SMOKE_SECONDS = (22.46757023734972, 20.17748290579766)


def pct(x: float) -> str:
    return f"{100.0 * float(x):.2f}"


def sgn(x: float) -> str:
    return f"{float(x):+.2f}"


def main() -> int:
    facts = C.read_json(C.ANALYSIS / "run_facts.json")
    run = C.read_json(C.FORMAL_RUN_DIR / "run.json")
    gates = C.read_json(C.ANALYSIS / "gates.json")
    hist_run = C.read_json(C.HISTORY_RUN_JSON)
    pos = C.read_json(C.POS_WEIGHT_JSON)
    per_label = list(csv.DictReader(open(C.RESULTS / "per_label_comparison.csv", encoding="utf-8")))
    curve = list(csv.DictReader(open(C.FORMAL_RUN_DIR / "curve.csv", encoding="utf-8")))
    by_epoch = {int(row["epoch"]): row for row in curve}
    hist = facts["history"]
    cur = facts["current"]
    weights = [float(row["pos_weight"]) for row in pos["labels"]]

    milestones = [1, 60, 120, 180, 221, 231, 240]
    milestone_rows = []
    for epoch in milestones:
        row = by_epoch.get(epoch)
        if row is None:
            continue
        milestone_rows.append(
            f"| {epoch} | {float(row['train_loss']):.5f} | {pct(row['valid_metric'])} | "
            f"{float(row['grad_norm_mean']):.2f} / {float(row['grad_norm_max']):.2f} | "
            f"{float(row['lr']):.3e} | {'是' if row['is_best'] == 'True' else ''} |"
        )

    pos_rows = []
    for row in pos["labels"]:
        pos_rows.append(
            f"| {row['label_index']} | {row['label']} | {row['train_pos']} | {row['train_neg']} | "
            f"{100.0 * float(row['train_pos_rate']):.2f}% | {float(row['pos_weight']):.6f} |"
        )

    req_rows = []
    train_gap_rows = []
    for row in per_label:
        req_rows.append(
            f"| {row['label']} | {int(row['train_pos'])}/{int(row['train_n'])} | "
            f"{100.0 * float(row['train_pos_rate']):.2f}% | {float(row['pos_weight']):.6f} | "
            f"{pct(row['history_valid_ap'])} | {pct(row['new_valid_ap'])} | "
            f"{sgn(row['delta_valid_ap_pp'])} |"
        )
        train_gap_rows.append(
            f"| {row['label']} | {pct(row['history_train_ap'])} | {pct(row['new_train_ap'])} | "
            f"{sgn(row['delta_train_ap_pp'])} | {pct(row['history_train_valid_gap'])} | "
            f"{pct(row['new_train_valid_gap'])} | {sgn(row['gap_change_pp'])} |"
        )

    improved = [row for row in per_label if float(row["delta_valid_ap_pp"]) > 0]
    declined = [row for row in per_label if float(row["delta_valid_ap_pp"]) < 0]
    improved_txt = "、".join(
        f"{row['label']} ({float(row['delta_valid_ap_pp']):+.4f} pp)" for row in improved
    ) or "无"
    largest_declines = sorted(declined, key=lambda row: float(row["delta_valid_ap_pp"]))[:3]
    decline_txt = "、".join(
        f"{row['label']} ({float(row['delta_valid_ap_pp']):+.2f} pp)" for row in largest_declines
    )
    gap_down = [row for row in per_label if float(row["gap_change_pp"]) < 0]
    gap_up = [row for row in per_label if float(row["gap_change_pp"]) > 0]

    steps_total = sum(int(row["steps"]) for row in curve)
    clip_share = 100.0 * float(run["grad_clip_events"]) / float(steps_total)
    hist_clip_share = 100.0 * float(hist_run["grad_clip_events"]) / float(steps_total)
    total_gpu_hours = (float(run["seconds"]) + sum(SMOKE_SECONDS)) / 3600.0
    final_row = curve[-1]
    fall = float(run["best_valid_macro_ap_pct"]) - float(final_row["valid_metric"]) * 100.0

    text = f"""# LCRC Peptides-func Weighted BCE Diagnostic v1 — 事实报告

- **Round ID**：`{C.ROUND_ID}`
- **结果标签**：`{C.RESULT_LABEL}`
- **唯一干预**：训练 loss 由普通 BCE 改为基于官方 Train 标签频率的 `pos_weight` 加权 BCE。
- **正式代码 commit**：`{run['environment']['git_commit']}`（`git_tracked_dirty = {run['environment']['git_tracked_dirty']}`；`git_dirty = {run['environment']['git_dirty']}` 仅因远端未跟踪的 `runs/` 输出目录）
- **正式 run id**：`{FORMAL_RUN_ID}`；实验名 `lcrc-pepweighted-p2-seed0`
- **机器 / 执行窗口（UTC）**：`res` / `a100-2`，1× NVIDIA A100-SXM4-40GB（`CUDA_VISIBLE_DEVICES=0`），driver 550.163.01，torch {run['environment']['torch']}（CUDA {run['environment']['torch_cuda']}），Python {run['environment']['python']}；{WINDOW_UTC}
- **报告口径**：只记录可核验事实、差值与实验局限；不给出“架构问题已解决/已证明”的结论。

> **Test 政策**：正式训练与分析入口安装 `sys.addaudithook`，任何打开历史 Test artifact 的行为立即失败。
> 本轮 `test_evaluated = {run['test_evaluated']}`，`test_access_attempts = {run['test_access_attempts']}`，
> 训练进程 `guard.attempts = {run['guard']['attempts']}`。正式训练进程没有读取 Test 数据、Test 标签或 Test 预测。
>
> **Test 授权修订（2026-11-04T03:36:43Z）**：本报告冻结后，用户明确授权对已冻结 checkpoint 做一次
> post-hoc terminal Test 评估。正式训练记录仍为 `test_evaluated = false`；Test 只用于事后比较，
> 未参与任何选择。结果见 `TEST_REPORT.md`。

---

## 0. 执行摘要

| 项 | 事实 |
|---|---|
| 正式训练 | 成功完成，status = `{run['status']}`，240/240 epoch，无早停、无 OOM、无时间上限阻塞 |
| 唯一干预 | Weighted BCEWithLogitsLoss，10 个 Train-only `pos_weight` |
| 架构 / 参数量 | LCRC-C1 / NO_BOND_MEAN；{run['parameters']:,} 参数；init hash 与历史 seed0 完全相同 |
| best Validation macro AP | **{pct(run['best_valid_macro_ap'])}%** @ epoch {run['best_epoch']} |
| 历史普通 BCE seed0 | {pct(hist['valid_macro_ap'])}% @ epoch {hist['best_epoch']} |
| 本轮 − 历史 | **{sgn(cur['delta_valid_pp'])} pp**（Validation macro AP） |
| Train macro AP @ best | {pct(run['train_macro_ap_per_label_mean'])}%（历史 {pct(hist['train_macro_ap'])}%，{sgn(cur['delta_train_pp'])} pp） |
| Train/Validation macro gap | {pct(cur['train_valid_gap'])} pp（历史 {pct(hist['train_valid_gap'])} pp，{sgn(cur['delta_gap_pp'])} pp） |
| GPU / 时间 | 正式 {cur['gpu_hours']:.3f} A100 GPU-hours；{cur['seconds']:.1f} s（上限 1.25 h / 4200 s） |
| 正确性门禁 | `analysis/gates.json`：{len(gates['checks'])} 项全部通过 = {gates['all_passed']} |
| 正式训练 Test 访问 | 0 次；`test_evaluated = false`（post-hoc 授权评估见 `TEST_REPORT.md`） |

---

## 1. 唯一干预与冻结不变量

`pos_weight` 逐标签定义为

```
pos_weight_j = train_negative_j / train_positive_j
```

只用官方 Train split 计算；没有截断、平滑、开方、归一化、上限或强制 `max(1, w)`。`antimicrobial` 的权重小于 1，`antiparasitic` 的权重约为 49。

**没有修改**：LCRC 架构、S1/S2、原子/键特征、readout、数据划分、特征缓存、normalization、初始化映射、AdamW 参数、peak LR、weight decay、dropout、batch size、gradient clipping、cosine horizon、early-stop 规则、checkpoint 选择指标。除 loss 外，训练配方与 Diagnostic v1 P2 相同（240 / 80 / 40；peak LR 3e-4；wd 1e-3；dropout 0.1；batch 64；clip 5.0；horizon 240）。

结构门禁（`run.json → structural_gate`）：

- 参数量 **{run['parameters']:,}**，与历史 Fast v1 / Diagnostic v1 相同；
- `init_state_hash` = `{run['model_info']['init_state_hash']}`，与 Fast v1 seed0 和 Diagnostic v1 seed0 均相同；
- feature spec hash、norm fingerprint、train/valid cache SHA-256 与 Diagnostic v1 完全一致；
- batch size 64，未触发 32/16 fallback；
- 因此本 run 是 loss 的单变量对照（单 seed）。

正确性检查在正式训练前完成，`analysis/gates.json` 覆盖：Train-only 权重与计数严格对应、标签顺序/维度/dtype/device、`ones(10)` 与普通 BCE 的数值及梯度一致、极端不平衡手工公式、缺失标签 mask 不广播错误、梯度有限且 AdamW 可更新参数、Test guard 拦截并记录、训练入口源码不引用 Test artifact。

---

## 2. 十个标签的 Train 正负计数与 `pos_weight`

来源：`analysis/pos_weight.json`（Train-only；`valid_used = false`，`test_used = false`）。权重为精确比值；GPU 上的 loss tensor 以 float32 保存，与精确比值的最大差异为 5.9e-7。

| idx | label | Train pos | Train neg | Train 阳性率 | pos_weight |
|---:|---|---:|---:|---:|---:|
{chr(10).join(pos_rows)}

`antiparasitic`：Train 217 正例，`pos_weight = {weights[8]:.10f}`。`antimicrobial`：Train 6,820 正例，`pos_weight = {weights[4]:.10f}`（小于 1，未截断）。

---

## 3. 正式训练事实

- 运行：`Peptides-func / LCRC-C1 / seed 0`，从 scratch 初始化，max 240 epoch，min 80，patience 40。
- 实际跑满 **240 epoch**；`stopped_early = false`。best Validation macro AP 在 **epoch {run['best_epoch']}**，距离 240 上限 19 个 epoch；之后到 240 没有新的严格提升。
- best checkpoint 的 Validation macro AP（选择指标）：**{pct(run['best_valid_macro_ap'])}%**；重新加载 checkpoint 后复算 {pct(run['reload_valid_macro_ap'])}%，差值 {run['reload_valid_delta']:.2e}（< 1e-4）。
- best checkpoint 的 Train macro AP：**{pct(run['train_macro_ap_per_label_mean'])}%**；逐标签 reloaded Validation AP 的均值为 {pct(run['valid_macro_ap_per_label_mean'])}%（与选择指标相差 {abs(100.0 * (run['valid_macro_ap_per_label_mean'] - run['best_valid_macro_ap'])):.4f} pp，来自 checkpoint reload / CUDA reduction）。
- 最后 epoch（240）：Train weighted BCE loss = {float(final_row['train_loss']):.5f}，Validation macro AP = {pct(final_row['valid_metric'])}%；比 best 低 {fall:.2f} pp。
- 分阶段最佳：epoch 1–120 为 {pct(run['phase_first_120']['best_valid_macro_ap'])}%@{run['phase_first_120']['best_epoch']}；epoch 121–240 为 {pct(run['phase_121_240']['best_valid_macro_ap'])}%@{run['phase_121_240']['best_epoch']}。

典型 epoch（weighted BCE loss 的绝对数值不能与普通 BCE 直接比较）：

| epoch | Train weighted BCE | Validation macro AP % | grad norm mean / max | LR | best |
|---:|---:|---:|---:|---:|---|
{chr(10).join(milestone_rows)}

资源与数值：

- 正式运行时间 **{cur['seconds']:.2f} s**（{cur['gpu_hours']:.4f} A100 GPU-hours；上限 1.25 h）。另有两次最小 CUDA 工程 smoke（22.47 s、20.18 s，不用于结果）；GPU 总计约 **{total_gpu_hours:.3f} h**，仍低于 1.25 h。
- GPU peak：allocated **{run['gpu_peak_allocated_bytes'] / 1048576:.2f} MiB**，reserved **{run['gpu_peak_reserved_bytes'] / 1048576:.2f} MiB**。
- 梯度：全训练 {steps_total:,} 次更新，clip 事件 **{run['grad_clip_events']:,}**（{clip_share:.2f}%）；历史普通 BCE 为 {int(hist_run['grad_clip_events']):,}（{hist_clip_share:.2f}%）。最大未裁剪梯度 **{run['grad_norm_max']:.2f}**（历史 {float(hist_run['grad_norm_max']):.2f}）。没有非有限 loss 或梯度；没有 OOM、batch fallback 或时间上限阻塞。
- 最佳 epoch 的 Train weighted BCE = {run['best_epoch_train_loss']:.5f}；epoch 梯度 mean/max = {run['best_epoch_grad_norm_mean']:.3f}/{run['best_epoch_grad_norm_max']:.3f}。

---

## 4. 三路方法对照

| 方法 | Seed | Loss | Best epoch | Train AP | Validation AP |
|---|---:|---|---:|---:|---:|
| 历史 LCRC-C1 | 0 | 普通 BCE | {hist['best_epoch']} | {pct(hist['train_macro_ap'])}% | {pct(hist['valid_macro_ap'])}% |
| 本轮 LCRC-C1 | 0 | Weighted BCE | {run['best_epoch']} | {pct(run['train_macro_ap_per_label_mean'])}% | {pct(run['best_valid_macro_ap'])}% |
| ECFP4 + LightGBM | 0 | 独立树模型目标 | — | 不适用 | {pct(facts['ecfp_background']['valid_macro_ap'])}% |

- 本轮 − 历史（Validation macro AP）：**{sgn(cur['delta_valid_pp'])} pp**（{pct(hist['valid_macro_ap'])}% → {pct(run['best_valid_macro_ap'])}%）。
- 本轮 − 历史（Train macro AP）：**{sgn(cur['delta_train_pp'])} pp**。
- ECFP4 + LightGBM 为固定背景参照，不重训、不是单变量对照；其与 LCRC 的差值不能解释为 loss 或架构效应。
- **普通 BCE 与加权 BCE 的训练 loss 数值不可直接比较**：两者对正例的权重不同，loss 的绝对尺度不同。

---

## 5. 逐标签 Validation AP：改善与退步

| label | Train pos / n | Train 阳性率 | pos_weight | 历史 Valid AP % | 本轮 Valid AP % | Δ pp |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(req_rows)}

- 改善的标签：{improved_txt}。
- 退步的标签数量：**{len(declined)}/10**；最大退步为 {decline_txt}。
- `antihypertensive` 的 +0.0017 pp 在两位小数下为 0.00 pp，应视为基本持平；其余 9 个标签均为负差值。
- macro AP 的 −3.73 pp 不是由单一标签造成：`antiviral`、`cell_cell_communication`、`anticancer` 三个标签的 Δ 合计为 −20.98 pp（按 10 标签均值折合 −2.10 pp，约占 macro 降幅的 56%）。

### 5.1 逐标签 Train AP 与 Train/Validation gap

gap = Train AP − Validation AP（正值表示 Train 更高）。

| label | 历史 Train AP % | 本轮 Train AP % | ΔTrain pp | 历史 gap pp | 本轮 gap pp | Δgap pp |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(train_gap_rows)}

- macro Train/Validation gap：{pct(hist['train_valid_gap'])} pp → {pct(cur['train_valid_gap'])} pp，变化 **{sgn(cur['delta_gap_pp'])} pp**。
- 逐标签 gap 下降的标签：{len(gap_down)} 个；上升的标签：{len(gap_up)} 个（{", ".join(row['label'] for row in gap_up)}）。
- 该 gap 收窄的同时 Train AP（{sgn(cur['delta_train_pp'])} pp）与 Validation AP（{sgn(cur['delta_valid_pp'])} pp）都下降，因此不能把 gap 变小单独解释为泛化改善。

---

## 6. 对八个验收问题的直接回答

1. **Weighted BCE 是否成功执行？是否严格保持同一架构与训练配置？**
   是。正式 run `{run['status']}`，240/240 epoch；参数量 {run['parameters']:,}，init hash、feature spec、norm fingerprint、train/valid cache 与历史 seed0 完全一致；batch 64、无 fallback；除 loss 外全部配方冻结。执行记录注记：远端 `git_dirty=true` 来自未跟踪的 `runs/` 输出，`git_tracked_dirty=false`。
2. **10 个标签的 `pos_weight` 分别是多少？**
   按标签顺序：{", ".join(f"{name}={w:.6f}" for name, w in zip(C.LABELS, weights))}。
3. **Weighted BCE 的最佳 Validation macro AP 是多少？**
   **{pct(run['best_valid_macro_ap'])}%**（epoch {run['best_epoch']}），reload 复算 {pct(run['reload_valid_macro_ap'])}%。
4. **相比原普通 BCE 的 60.62%，提高还是降低了多少百分点？**
   降低 **{abs(cur['delta_valid_pp']):.4f} pp**（60.6229% → {pct(run['best_valid_macro_ap'])}%）。
5. **哪些标签的 AP 改善最多？哪些下降？**
   仅 `antihypertensive` 为正（+0.0017 pp，基本持平）；其余 9 个标签全部下降。下降最多的是 `antiviral`（{float(next(r['delta_valid_ap_pp'] for r in per_label if r['label']=='antiviral')):+.2f} pp）、`cell_cell_communication`（{float(next(r['delta_valid_ap_pp'] for r in per_label if r['label']=='cell_cell_communication')):+.2f} pp）、`anticancer`（{float(next(r['delta_valid_ap_pp'] for r in per_label if r['label']=='anticancer')):+.2f} pp）。
6. **最佳 checkpoint 在第几个 epoch？是否再次接近 240 epoch 上限？**
   epoch **{run['best_epoch']}**；跑满 240 epoch，没有早停；best 距上限 19 个 epoch，之后 19 个 epoch 没有新的严格提升。仍属于接近 240 上限才达到最佳。
7. **Train/Validation gap 如何变化？**
   macro gap {pct(hist['train_valid_gap'])} pp → {pct(cur['train_valid_gap'])} pp，**{sgn(cur['delta_gap_pp'])} pp**。但 Train AP 和 Validation AP 同时下降（{sgn(cur['delta_train_pp'])} pp / {sgn(cur['delta_valid_pp'])} pp），所以这不是“泛化更好”的证据。
8. **是否发生数值不稳定、过大的梯度或资源阻塞？**
   没有非有限 loss/梯度、OOM、batch fallback 或时间/预算阻塞。梯度显著大于历史：clip 事件 {run['grad_clip_events']:,}/{steps_total:,}（{clip_share:.2f}%）对比历史 {int(hist_run['grad_clip_events']):,}（{hist_clip_share:.2f}%），最大未裁剪梯度 {run['grad_norm_max']:.2f} 对比 {float(hist_run['grad_norm_max']):.2f}；训练靠固定 clip 5.0 完成。正式 GPU 使用 {cur['gpu_hours']:.3f} h（含两次最小 smoke 约 {total_gpu_hours:.3f} h），低于 1.25 h；实际耗时 {cur['seconds']:.1f} s，低于 4200 s。

---

## 7. 产物、哈希与复现

- 代码 commit：`{run['environment']['git_commit']}`；正式 run：`{FORMAL_RUN_ID}`。
- checkpoint：`runs/weighted_bce_seed0/checkpoint.pt`，SHA-256 `{run['checkpoint_sha256']}`。
- curve：`runs/weighted_bce_seed0/curve.csv`，SHA-256 `{run['curve_sha256']}`。
- 配置：`configs/protocol.json`；协议：`PROTOCOL.md`。
- 权重：`analysis/pos_weight.json`；门禁：`analysis/gates.json`（{len(gates['checks'])} 项，全部通过 = {gates['all_passed']}）。
- 逐 epoch 训练记录：`runs/weighted_bce_seed0/curve.csv`；选择记录：`runs/weighted_bce_seed0/selection.json`。
- 对照表：`results/validation_comparison.csv`、`results/per_label_comparison.csv`。
- 资源：`analysis/resource_ledger.json`（GPU budget 1.25 h；formal {cur['gpu_hours']:.4f} h；peak allocated {run['gpu_peak_allocated_bytes'] / 1048576:.2f} MiB；reserved {run['gpu_peak_reserved_bytes'] / 1048576:.2f} MiB）。
- Test：`test_evaluated = false`，`test_access_attempts = []`，guard attempts = `[]`。

复现命令（仓库根目录）：

```bash
# Train-only weights + gates + tests
uv run python tracks/structural-sufficiency/lcrc_peptides_weighted_bce_v1/code/wb_compute_pos_weight.py
uv run python tracks/structural-sufficiency/lcrc_peptides_weighted_bce_v1/code/wb_check.py
uv run pytest -q tracks/structural-sufficiency/lcrc_peptides_weighted_bce_v1/tests

# The single formal run (committed revision, res-gpu0, 1 A100)
rr run res lcrc-pepweighted-p2-seed0 --pool res-gpu0 --gpus 1 --cpus 4 --mem 32G --time 01:15:00 \\
  --result tracks/structural-sufficiency/lcrc_peptides_weighted_bce_v1/runs/weighted_bce_seed0 -- \\
  uv run python tracks/structural-sufficiency/lcrc_peptides_weighted_bce_v1/code/wb_p2_train.py \\
  --device cuda --out-dir tracks/structural-sufficiency/lcrc_peptides_weighted_bce_v1/runs/weighted_bce_seed0 \\
  --max-seconds 4200

# Pull + comparison tables
rr pull lcrc-pepweighted-p2-seed0
uv run python tracks/structural-sufficiency/lcrc_peptides_weighted_bce_v1/code/wb_analyze.py
```

---

## 8. 实验局限

- **单 seed、单配置**：只有一个 seed 0 和一个预先冻结的加权 BCE 配置；没有第二次训练、没有调参、没有根据 Validation 结果选择第二套配置。因此无法估计 seed 方差，也不能把差值分解为 loss 效应与优化噪声。
- **原始训练/选择为 Validation-only**：正式训练与 checkpoint 选择只使用 Train/Validation，Test 未参与；用户随后授权的 post-hoc Test 评估见 `TEST_REPORT.md`，不能反推训练时使用过 Test。
- **目标函数已改变**：普通 BCE 与 weighted BCE 的 loss 尺度不同，loss 数值不能用于判断哪个目标“训练得更好”。
- **单变量对照的边界**：ECFP4 + LightGBM 是不同模型/不同目标的固定背景参照，不是单变量对照。
- **指标细节**：`best_valid_macro_ap` 是训练中实时模型的 checkpoint 选择指标；逐标签 AP 来自 reloaded checkpoint，reload 差值为 {run['reload_valid_delta']:.2e}。
- **gap 解释限制**：Train/Validation gap 收窄伴随两个 AP 同时下降，不能单独解释为泛化改善。
- **执行环境**：远端 `git_dirty=true` 仅由未跟踪的 `runs/` 输出目录触发；实际执行 commit 为 `{run['environment']['git_commit']}`，`git_tracked_dirty=false`。
- 用户随后授权的 post-hoc Test 评估见 `TEST_REPORT.md`；它没有改变本文件的 Validation-only 训练结论，也没有参与任何选择。原正式训练 `run.json` 仍记录 `test_evaluated = false`。
- 本轮到此停止；不提出、不执行下一轮实验。上述事实不足以支持“架构问题已解决”或“架构问题已证明”的结论。
"""
    (C.ROUND / "REPORT.md").write_text(text, encoding="utf-8")
    C.log(f"wrote {C.track_rel(C.ROUND / 'REPORT.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
