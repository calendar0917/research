"""Render ``TEST_REPORT.md`` for the authorized post-hoc Test evaluation."""

from __future__ import annotations

import csv

import wb_common as C

ROUND_ID = "lcrc-peptides-weighted-bce-v1"


def pct(x: float) -> str:
    return f"{100.0 * float(x):.2f}"


def sgn(x: float) -> str:
    return f"{float(x):+.2f}"


def main() -> int:
    data = C.read_json(C.ANALYSIS / "test_eval.json")
    gates = C.read_json(C.ANALYSIS / "test_gates.json")
    auth = data["authorization"]
    per_label = list(
        csv.DictReader(open(C.RESULTS / "test_per_label_comparison.csv", encoding="utf-8"))
    )
    by_id = {row["id"]: row for row in data["checkpoints"]}
    fast = by_id["fastv1_seed0_120ep"]
    ordinary = by_id["diagnostic_ordinary_bce_240ep"]
    weighted = by_id["weighted_bce_240ep"]
    comparison = data["comparison"]

    stored_valid_delta = 100.0 * (
        float(weighted["stored_validation_macro_ap"]) - float(ordinary["stored_validation_macro_ap"])
    )
    test_delta = float(comparison["weighted_minus_ordinary_test_pp"])
    cpu_valid_delta = float(comparison["weighted_minus_ordinary_validation_pp"])
    test_rows = []
    for row in [fast, ordinary, weighted]:
        test_rows.append(
            f"| {row['id']} | {row['best_epoch']} | {pct(row['stored_validation_macro_ap'])} | "
            f"{pct(row['recomputed_validation_macro_ap'])} | {pct(row['test_macro_ap'])} | "
            f"{sgn(row['test_minus_validation_pp'])} |"
        )

    per_label_rows = []
    for row in per_label:
        per_label_rows.append(
            f"| {row['label']} | {float(row['pos_weight']):.6f} | {pct(row['fastv1_test_ap'])} | "
            f"{pct(row['ordinary_test_ap'])} | {pct(row['weighted_test_ap'])} | "
            f"{sgn(row['delta_test_ap_pp'])} | {pct(row['ordinary_valid_ap'])} | "
            f"{pct(row['weighted_valid_ap'])} | {sgn(row['delta_valid_ap_pp'])} |"
        )

    improved_test = [row for row in per_label if float(row["delta_test_ap_pp"]) > 0]
    declined_test = [row for row in per_label if float(row["delta_test_ap_pp"]) < 0]
    ranked_test = sorted(per_label, key=lambda row: float(row["delta_test_ap_pp"]))
    worst_test = ranked_test[:3]
    best_test = ranked_test[-1]
    improved_txt = (
        "、".join(
            f"{row['label']} ({float(row['delta_test_ap_pp']):+.4f} pp)" for row in improved_test
        )
        or "无"
    )
    worst_txt = "、".join(
        f"{row['label']} ({float(row['delta_test_ap_pp']):+.2f} pp)" for row in worst_test
    )
    per_label_max_sanity = 100.0 * float(fast["stored_test_per_label_delta"])
    ordinary_vs_fast_test = 100.0 * (float(ordinary["test_macro_ap"]) - float(fast["test_macro_ap"]))
    weighted_vs_fast_test = 100.0 * (float(weighted["test_macro_ap"]) - float(fast["test_macro_ap"]))

    text = f"""# LCRC Peptides-func Weighted BCE v1 — 授权后 Test 评估

- **Round ID**：`{ROUND_ID}`
- **修订 ID**：`{data['amendment_id']}`
- **结果标签**：`{data['result_label']}`
- **授权**：用户明确授权访问官方 Test split；授权消息：“{auth['authorization_message']}”
- **授权时间（UTC）**：`{auth['authorization_received_utc']}`
- **评估对象**：已冻结的 3 个 checkpoint（Fast v1 120ep 仅作 Test 管线 sanity；Diagnostic 普通 BCE 240ep；本轮 weighted BCE 240ep）
- **评估环境**：本地 CPU，torch {data['environment']['torch']}（CUDA {data['environment']['torch_cuda']} 编译版，CPU 前向），Python {data['environment']['python']}；耗时 {data['duration_seconds']:.2f} s
- **执行状态**：commit `{data['environment']['git_commit']}`，`git_tracked_dirty = {data['environment']['git_tracked_dirty']}`；`git_dirty = {data['environment']['git_dirty']}` 仅因当时尚未跟踪的 Test 评估/报告文件。评估脚本在运行后未再改动。
- **训练**：没有训练、没有反向传播、没有调参、没有重新选择 checkpoint。
- **Test 用途**：只用于事后报告与比较；不参与任何 checkpoint、seed、超参数或配置选择。

> **重要边界**：正式训练进程的 `run.json` 仍然是 `test_evaluated = false`；本文件是训练结束后、checkpoint 已冻结后进行的单独 post-hoc terminal Test 评估。原始 `REPORT.md` 的 Validation-only 结果不变。Test 一旦被查看，后续若再据其做选择，就必须声明为 Test-based selection；本轮没有这样做。

---

## 1. Test 管线完整性

| 检查 | 结果 |
|---|---|
| Test cache SHA-256 | `{data['test_split']['cache_sha256']}`，与历史 Fast v1 `test_cache_identity.cache_test_sha256` 一致 |
| Test labels | `{data['test_split']['labels']}`；shape `{data['test_split']['n']} × 10`，有限且二值 |
| Test row identity | `{data['test_split']['row_identity_status']}`（Diagnostic v1 已核验 2,331 行） |
| feature spec hash | `{data['test_split']['feature_spec_hash']}`，与训练 spec 一致 |
| Test pipeline sanity | 重新评估 Fast v1 seed0：Test macro AP **{pct(fast['test_macro_ap'])}%**，历史记录 **{pct(fast['stored_test_macro_ap'])}%**；差值 **{100.0 * float(fast['stored_test_delta']):.4f} pp**，逐标签最大差值 **{per_label_max_sanity:.4f} pp** |
| Test AP 独立重算 | 所有 checkpoint 的逐标签 AP 与 sklearn `average_precision_score` 重算差值为 0 |
| 概率范围 | 所有 Test 概率有限且在 `[0, 1]` |
| 门禁 | `analysis/test_gates.json`：{len(gates['checks'])} 项全部通过 = `{gates['all_passed']}` |

Fast v1 Test 复现与历史记录一致，说明 Test cache、Test labels、模型重建和 AP 计算管线一致。CPU 与历史 GPU 的差异量级为 1e-4 到 1e-3 AP。

---

## 2. Test 主结果

| checkpoint | best epoch | 历史/选择 Validation AP % | 本次 CPU 复算 Validation AP % | **Test AP %** | Test − CPU Validation pp |
|---|---:|---:|---:|---:|---:|
{chr(10).join(test_rows)}

- **普通 BCE 240ep**：Test macro AP **{pct(ordinary['test_macro_ap'])}%**。
- **Weighted BCE 240ep**：Test macro AP **{pct(weighted['test_macro_ap'])}%**。
- **Weighted − ordinary（Test）**：**{sgn(test_delta)} pp**。
- Weighted − ordinary（历史/选择 Validation）：**{sgn(stored_valid_delta)} pp**；本次 CPU 复算 Validation：**{sgn(cpu_valid_delta)} pp**。

**方向一致性**：Validation 上 weighted BCE 低 3.73 pp；Test 上低 3.30 pp。两个划分给出同方向的结论：本轮加权 BCE 没有超过普通 BCE。

### 2.1 与历史 Fast v1 的 Test 数值

| 方法 | Validation AP % | Test AP % |
|---|---:|---:|
| Fast v1 ordinary BCE 120ep（历史记录；本次复算） | 53.73 | **51.69** |
| Diagnostic ordinary BCE 240ep（本次 Test 评估） | 60.62（历史选择值） | **58.22** |
| Weighted BCE 240ep（本次 Test 评估） | 56.89（历史选择值） | **54.92** |

普通 BCE 240ep 相对 Fast v1 的 Test 变化：**{sgn(ordinary_vs_fast_test)} pp**；weighted BCE 240ep 相对 Fast v1：**{sgn(weighted_vs_fast_test)} pp**；weighted 相对 ordinary 240ep：**{sgn(test_delta)} pp**。

---

## 3. 逐标签 Test AP

| label | pos_weight | Fast v1 Test AP % | ordinary Test AP % | weighted Test AP % | Δ Test pp | ordinary Valid AP % | weighted Valid AP % | Δ Valid pp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(per_label_rows)}

- Test 上改善的标签：**{improved_txt}**。
- Test 上退步的标签数量：**{len(declined_test)}/10**；最大退步为 {worst_txt}。
- 最小 Test 变化为 `{best_test['label']}`（{float(best_test['delta_test_ap_pp']):+.4f} pp），仍然为负。
- 在 Validation 上唯一非负的 `antihypertensive`（+0.0017 pp）在 Test 上为 **{float(next(row['delta_test_ap_pp'] for row in per_label if row['label']=='antihypertensive')):+.2f} pp**，没有复现 Validation 的持平。
- Test 与 Validation 的逐标签变化方向并非完全一致，但 macro 方向一致。

---

## 4. 与 Validation 的关系

- 普通 BCE 240ep：Validation {pct(ordinary['recomputed_validation_macro_ap'])}% → Test {pct(ordinary['test_macro_ap'])}%，Test − Validation **{sgn(ordinary['test_minus_validation_pp'])} pp**。
- Weighted BCE 240ep：Validation {pct(weighted['recomputed_validation_macro_ap'])}% → Test {pct(weighted['test_macro_ap'])}%，Test − Validation **{sgn(weighted['test_minus_validation_pp'])} pp**。
- Fast v1 120ep：Validation {pct(fast['recomputed_validation_macro_ap'])}% → Test {pct(fast['test_macro_ap'])}%，Test − Validation **{sgn(fast['test_minus_validation_pp'])} pp**。
- 三个 checkpoint 的 Test 都低于各自 Validation。Weighted 的 Test−Validation 差距（{sgn(weighted['test_minus_validation_pp'])} pp）小于 ordinary（{sgn(ordinary['test_minus_validation_pp'])} pp），但 weighted 的 Validation 和 Test 绝对 AP 都更低；不能把“差距更小”解释为泛化更好。

---

## 5. ECFP4 + LightGBM 的 Test 状态

**没有评估。** 原因：Diagnostic v1 只保存了 ECFP4 + LightGBM 的 Validation 预测，没有持久化 10 个 LightGBM 模型；重新训练 ECFP 属于新的 ECFP run，超出“Test 访问授权”的范围，也被原协议禁止。因此本轮 Test 表只覆盖 LCRC checkpoint。

历史 Fast v1 报告中出现的 ECFP4 + LightGBM Test 74.6% 使用的是 Train+Validation 训练，和本轮 Diagnostic 的 Train-only ECFP 不是同一设置，不能作为本轮 Test 对照。

---

## 6. 结论性事实与限制

1. **Test 主比较**：普通 BCE 240ep Test **{pct(ordinary['test_macro_ap'])}%**；weighted BCE 240ep Test **{pct(weighted['test_macro_ap'])}%**；weighted 低 **{abs(test_delta):.4f} pp**。
2. **与 Validation 一致**：Validation 低 3.7337 pp，Test 低 {abs(test_delta):.4f} pp；两个划分方向一致。
3. **逐标签**：Test 上 10 个标签全部为负；最大降幅为 {worst_txt}。
4. **管线可信度**：Fast v1 seed0 Test 复算 {pct(fast['test_macro_ap'])}%，历史 {pct(fast['stored_test_macro_ap'])}%，差 {100.0 * float(fast['stored_test_delta']):.4f} pp；Test cache/labels/row identity 全部通过。
5. **不是 Test 选择实验**：checkpoint 已在 Test 评估前由 Validation 冻结；Test 只用于事后比较，未用于选择。若以后使用这些 Test 数值选模型，性质会变成 Test-based selection。
6. **仍是单 seed、单配置**：无法估计 seed 方差；不能把 −3.30 pp 分解为 loss、优化或数据噪声。
7. **ECFP 缺失**：没有 ECFP Test 数值；不能用 74.6% 的 Train+Validation 文献值替代。
8. **架构结论**：这些 Test 数值只说明该单 seed/单配置下 weighted BCE 的 Test AP 更低；不能支持“架构问题已解决/已证明”的结论。

---

## 7. 产物

- 授权：`analysis/test_access_authorization.json`
- 修订配置：`configs/test_access_amendment.json`
- Test 评估记录：`analysis/test_eval.json`
- Test 门禁：`analysis/test_gates.json`（{len(gates['checks'])} 项全部通过 = `{gates['all_passed']}`）
- Test 对照：`results/test_comparison.csv`
- 逐标签 Test 对照：`results/test_per_label_comparison.csv`
- 预测（本地、未纳入 Git）：`runs/test_eval/*_test_predictions.npz`
- 复现命令：

```bash
uv run python tracks/structural-sufficiency/lcrc_peptides_weighted_bce_v1/code/wb_test_eval.py \\
  --device cpu --out-dir tracks/structural-sufficiency/lcrc_peptides_weighted_bce_v1/runs/test_eval
```

- 原始 Validation 报告：`REPORT.md`
- 原始训练 `run.json` 仍记录 `test_evaluated = false`；Test 访问仅发生在上述 post-hoc 入口。
"""
    (C.ROUND / "TEST_REPORT.md").write_text(text, encoding="utf-8")
    C.log(f"wrote {C.track_rel(C.ROUND / 'TEST_REPORT.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
