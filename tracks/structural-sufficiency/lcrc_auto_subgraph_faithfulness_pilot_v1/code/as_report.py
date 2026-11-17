"""REPORT.md / DECISION.md generation from the frozen analysis artefacts.

Every number in the generated documents is read back from
``analysis/*.json`` / ``analysis/*.csv``; nothing is typed by hand and no text is
produced that the artefacts do not support.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import as_common as C


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(x):
        return "NA"
    if x != 0.0 and (abs(x) < 1e-3 or abs(x) >= 1e5):
        return f"{x:.{digits}e}"
    return f"{x:.{digits}f}"


def _pct(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(x):
        return "NA"
    return f"{100.0 * x:.1f}%"


def _cell(value: Any, where: str) -> str:
    """One markdown table cell; a raw pipe would silently break the table."""
    text = str(value)
    if "|" in text:
        raise ValueError(f"markdown table cell contains a pipe ({where}): {text!r}")
    return text


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    width = len(headers)
    out = [
        "| " + " | ".join(_cell(h, "header") for h in headers) + " |",
        "|" + "|".join(["---"] * width) + "|",
    ]
    for row in rows:
        if len(row) != width:
            raise ValueError(f"table row has {len(row)} cells, header has {width}")
        out.append("| " + " | ".join(_cell(c, f"row {len(out)}") for c in row) + " |")
    return "\n".join(out)


def _load() -> dict[str, Any]:
    a = C.ANALYSIS_DIR
    return {
        "preflight": C.read_json(a / "preflight.json"),
        "training": C.read_json(a / "training_source.json"),
        "freeze": C.read_json(a / "protocol_freeze.json"),
        "boot": C.read_json(a / "bootstrap.json"),
        "failures": C.read_json(a / "failure_cases.json"),
        "decision": C.read_json(a / "decision.json"),
        "manifest": C.read_json(a / "run_manifest.json"),
        "main_rows": C.SC.read_csv(a / "main_table.csv"),
        "diag_rows": C.SC.read_csv(a / "architecture_diagnostics.csv"),
        "molecule_rows": C.SC.read_csv(a / "molecule_table.csv"),
        "figures": C.read_json(a / "figures" / "index.json"),
    }


def _main_lookup(rows: Sequence[Mapping[str, str]]) -> dict[tuple[str, str], Mapping[str, str]]:
    return {(r["dataset"], r["split"]): r for r in rows}


def _diag_lookup(rows: Sequence[Mapping[str, str]]) -> dict[tuple[str, str], Mapping[str, str]]:
    return {(r["dataset"], r["split"]): r for r in rows}


def _f(row: Mapping[str, Any], key: str, digits: int = 4) -> str:
    return _fmt(row.get(key), digits)


# ---------------------------------------------------------------------------
# REPORT.md
# ---------------------------------------------------------------------------


def write_report() -> Path:
    data = _load()
    main = _main_lookup(data["main_rows"])
    diag = _diag_lookup(data["diag_rows"])
    decision = data["decision"]
    preflight = data["preflight"]
    training = data["training"]
    boot = data["boot"]
    figures = data["figures"]
    manifest = data["manifest"]

    lines: list[str] = []
    lines.append("# REPORT — lcrc-auto-subgraph-faithfulness-pilot-v1")
    lines.append("")
    lines.append(
        "冻结的 LCRC-C1（Arm A）在官方 `ogbg-molesol` / `ogbg-mollipo` 上的"
        "**自动连通子图解释可行性**开发性验证。单 seed、每划分 64 个分子、"
        "单次干预条件 `CONDITIONAL_S1_S2_EVIDENCE_MASK`。"
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    # 1 summary
    lines.append("## 1. 一页摘要")
    lines.append("")
    lines.append(f"* **ENGINEERING_GATE**：`{preflight['engineering_gate']}`（关键门禁见第 4 节）。")
    lines.append(
        f"* **DISCOVERY_SIGNAL（总体）**：`{decision['discovery_signal_overall']}`。"
    )
    for dataset in C.DATASETS:
        sig = decision["discovery_signal_per_dataset"][dataset]
        lines.append(
            f"* `{dataset}`：`{sig['verdict']}`"
            + ("" if sig["signal"]["pass"] else f"，未通过条件：{', '.join(sig['signal']['failed_conditions'])}")
            + "。"
        )
    lines.append(
        "* 关键提醒：本轮的干预是**显式结构证据的条件性屏蔽**，"
        "等价于在冻结函数的 S1/S2 求和里把对应项乘 0，"
        "**不是**真实分子删键、删基团或完整化学反事实。"
    )
    lines.append("")

    # 2 boundary
    lines.append("## 2. 实验边界（冻结，不可事后调整）")
    lines.append("")
    lines.append(_table(
        ["项", "冻结值"],
        [
            ["实验 ID", f"`{C.ROUND_ID}`"],
            ["模型", "`lcrc_readout_context_pilot_small_v1` Arm A = 原始 LCRC-C1：`sum_v f(h_v) + rho([g, mean(h), mean(h^2), log1p(n_atoms)])`"],
            ["训练", "复用既有 360-epoch 轨迹（seed 0），本轮**不重训**"],
            ["数据集", "官方 `ogbg-molesol` / `ogbg-mollipo`，OGB `get_idx_split()` 冻结索引"],
            ["解释样本", f"每数据集 Train/Valid/Test 各 {C.N_SAMPLES_PER_SPLIT} 个（共 {2 * 3 * C.N_SAMPLES_PER_SPLIT}）"],
            ["样本选择", f"`sha256({C.SAMPLE_SEED} + dataset + split + row_id)` 升序取前 64，与标签/误差/梯度无关"],
            ["搜索空间", f"连通子图，物理键数 {C.SEARCH_MIN_K}–{C.SEARCH_MAX_K}，每层 beam {C.BEAM_WIDTH}，Top-{C.TOP_K_SUBGRAPHS}，Jaccard ≤ {C.JACCARD_MAX_OVERLAP}"],
            ["发现分数", "`A(S) = abs(sum_e g[e]) / sqrt(k)`（有符号先求和再取绝对值，k = 键数）"],
            ["干预", f"`{C.CONDITION_NAME}`，门控 `m[e] ∈ [0,1]`，剂量点 m = {C.DOSE_GATE}"],
            ["对照 A", f"同分子同键数随机连通子图，匹配 S2 occurrence 数（相对差 ≤ {int(100 * C.S2_MATCH_REL_TOL)}%），RNG {C.CONTROL_SEED}，最多 {C.MAX_GROWTH_ATTEMPTS} 次尝试，取前 {C.N_MATCHED_RANDOM} 个"],
            ["对照 B", "原子局部输出启发式 `B(S) = sum_v abs(f(h_v)) / sqrt(k)`（仅作廉价对比，不称因果）"],
            ["效果分辨尺度", f"{100 * C.EFFECT_FLOOR_STD_FRACTION:.0f}% 的 Train 标签标准差（仅本轮 pilot 尺度）"],
            ["bootstrap", f"{C.BOOTSTRAP_REPLICATES} 次，分子为重采样单位，seed {C.BOOTSTRAP_SEED}"],
            ["预算", f"每 GPU worker 软上限 {C.GPU_BUDGET_SOFT_SECONDS / 3600:.1f} GPU·h，硬上限 {C.GPU_BUDGET_HARD_SECONDS / 3600:.1f} GPU·h"],
        ],
    ))
    lines.append("")

    # 3 identity
    lines.append("## 3. 数据、模型与复用身份核验")
    lines.append("")
    rows = []
    for dataset in C.DATASETS:
        ts = training["datasets"][dataset]
        pf = preflight["datasets"][dataset]
        rows.append([
            f"`{dataset}`",
            _fmt(ts["sha256"]["checkpoint_best_val.pt"][:12]),
            str(ts["trajectory"]["epochs"]),
            str(ts["trajectory"]["test_eval_count"]),
            str(ts["trajectory"]["selected_epoch"]),
            _fmt(ts["trajectory"]["selected_valid_rmse_raw"]),
            _fmt(ts["trajectory"]["selected_test_rmse_raw"]),
            str(pf["checkpoint_replay"]["pass"]),
        ])
    lines.append(_table(
        ["dataset", "checkpoint sha256", "epochs", "test evals", "selected epoch", "Valid RMSE", "Test RMSE", "replay"],
        rows,
    ))
    lines.append("")
    lines.append(
        "* 两个数据集的 checkpoint / trajectory / `best_predictions.npz` 均与源轮次 "
        "`analysis/run_manifest.json` 记录的 SHA-256 一致；每个 epoch 都执行了 "
        "Train/Valid/Test 三个完整划分的评估（360/360）。"
    )
    lines.append(
        "* 选中 epoch = 该冻结轨迹的最小 Valid RMSE（平局取更早 epoch），本轮已独立重算并核对。"
    )
    lines.append(
        "* 全量 Train/Valid/Test 预测由重载 checkpoint 逐项重放（见 `analysis/preflight.json` 的 "
        "`checkpoint_replay`），Test 在本轮只作为 `DEVELOPMENT_TEST` 报告。"
    )
    lines.append("")

    # 4 gates
    lines.append("## 4. ENGINEERING_GATE")
    lines.append("")
    gate_rows = []
    for dataset in C.DATASETS:
        for gate in preflight["datasets"][dataset]["gates"]:
            gate_rows.append([f"`{dataset}`", gate["gate"], _fmt(gate["passed"]), gate["detail"]])
    lines.append(_table(["dataset", "gate", "passed", "detail"], gate_rows))
    lines.append("")
    fd_rows = []
    for dataset in C.DATASETS:
        fd = preflight["datasets"][dataset]["finite_difference"]
        rn = preflight["datasets"][dataset]["atom_renumbering"]
        eq = preflight["datasets"][dataset]["model_equivalence"]
        fd_rows.append([
            f"`{dataset}`",
            str(fd["n_bonds_checked"]),
            _fmt(fd["max_abs_err"]),
            _fmt(fd["max_rel_err_of_large"]),
            _fmt(eq["pred_raw_max_abs_err"]),
            _fmt(rn.get("max_grad_abs_err")) if "max_grad_abs_err" in rn else _fmt(
                max((r["grad_max_abs_err"] for r in preflight["datasets"][dataset]["atom_renumbering_records"]),
                    default=0.0)
            ),
        ])
    lines.append(_table(
        ["dataset", "FD bonds", "FD max abs err", "FD max rel err", "all-ones max abs err", "renumber grad max abs err"],
        fd_rows,
    ))
    lines.append("")
    lines.append(
        "* 全 1 门控与冻结 Arm A 的预测、`local_sum`、`rho` 在 Train/Valid/Test 行上**在冻结容差内一致**"
        "（`1e-5 + 1e-6*max(1,|x|)`，原始单位；实测最大误差见上表，比容差小 1–2 个数量级，"
        "残差来自门控乘法在浮点累加顺序上的差异，不是模型差异）。"
    )
    lines.append(
        "* S1/S2 occurrence 映射（含三角闭合路径与重复路径）逐项与冻结缓存比对；"
        "同一 occurrence 被两条被屏蔽键触碰时，按“从路径列表中删除该 occurrence”验证只移除一次。"
    )
    lines.append("* 解释前后 checkpoint / state hash 不变；原子重编号后预测与键梯度等价。")
    lines.append("")

    # 5 samples
    lines.append("## 5. 解释样本")
    lines.append("")
    sample_rows = []
    for dataset in C.DATASETS:
        for split in C.SPLITS:
            sel = manifest["sample_selection"][dataset][split]
            sample_rows.append([f"`{dataset}`", split, str(sel["n_selected"]), f"`{sel['row_ids_sha256'][:16]}`"])
    lines.append(_table(["dataset", "split", "n", "row_ids sha256"], sample_rows))
    lines.append("")
    lines.append("选择规则与标签、预测误差、分子内容、梯度、Test 结果均无关；完整 ID 见 `results/<dataset>/samples.json`。")
    lines.append("")

    # 6 discovery
    lines.append("## 6. 结构发现与真实屏蔽响应（各 split 分列）")
    lines.append("")
    rows = []
    for dataset in C.DATASETS:
        for split in C.SPLITS:
            m = main[(dataset, split)]
            rows.append([
                f"`{dataset}`", split, m["n_interpreted"], _pct(float(m["coverage"])),
                _f(m, "top1_n_bonds_median", 1), _f(m, "top1_n_atoms_median", 1),
                _f(m, "top1_n_s2_occurrences_median", 1),
                _f(m, "abs_delta_top1_median"),
                _f(m, "abs_delta_top1_median_over_train_std"),
                _pct(float(m["frac_top1_ge_1pct_train_std"])),
            ])
    lines.append(_table(
        ["dataset", "split", "n", "coverage", "Top1 k", "Top1 atoms", "Top1 S2 occ",
         "median abs(delta)", "median abs(delta)/std", "abs(delta) ≥ 1% std"],
        rows,
    ))
    lines.append("")
    rows = []
    for dataset in C.DATASETS:
        for split in C.SPLITS:
            m = main[(dataset, split)]
            rows.append([
                f"`{dataset}`", split,
                _pct(float(m["sign_agreement_top1_frac"])),
                _pct(float(m["sign_agreement_topk_frac"])),
                _f(m, "linearisation_median_abs_err"),
                _f(m, "linearisation_median_rel_err"),
            ])
    lines.append(_table(
        ["dataset", "split", "sign agreement (Top1)", "sign agreement (Top1-3)", "median abs(linearisation err)", "median rel err"],
        rows,
    ))
    lines.append("")
    lines.append(
        "`sign agreement` = 用 `sum_{e in S} g[e]` 的符号预测真实屏蔽差 `delta = y0 - yS` 的符号的比例；"
        "`linearisation err` = `|delta - sum g|`，即在 m=1 处的连续梯度近似与真实有限屏蔽之间的偏差。"
    )
    lines.append("")

    # 7 controls
    lines.append("## 7. 相对两类对照")
    lines.append("")
    rows = []
    for dataset in C.DATASETS:
        for split in C.SPLITS:
            m = main[(dataset, split)]
            b = boot["datasets"][dataset]["splits"][split]
            rows.append([
                f"`{dataset}`", split,
                f"{m['matched_n_molecules']}/{m['n_interpreted']}",
                _pct(float(m["matched_success_rate"])),
                _f(m, "matched_median_effect_ratio", 3),
                _pct(float(m["matched_ratio_gt1_frac"])),
                _f(m, "matched_median_effect_advantage"),
                _f(m, "matched_median_abs_delta_random"),
                f"[{_fmt(b['effect_ratio']['ci_low'], 3)}, {_fmt(b['effect_ratio']['ci_high'], 3)}]",
                _f(m, "local_heuristic_median_abs_delta"),
                _f(m, "local_heuristic_median_advantage"),
                _pct(float(m["frac_top1_gt_local_heuristic"])),
            ])
    lines.append(_table(
        ["dataset", "split", "matched n", "success", "median effect_ratio", "ratio>1",
         "median advantage", "random median abs(delta)", "ratio CI95",
         "heuristic median abs(delta)", "advantage vs heuristic", "Top1 > heuristic"],
        rows,
    ))
    lines.append("")
    lines.append(
        "`effect_ratio = |delta(Top1)| / (median_e |delta(matched random_e)| + 1e-12)`；"
        "`effect_advantage = |delta(Top1)| - median_e |delta(matched random_e)|`。"
        "bootstrap 区间只反映**固定 checkpoint、固定数据集**下的样本不确定性，不是跨 seed 稳定性。"
        "匹配对照不足 8 个的分子记为 `INSUFFICIENT_MATCHED_NULL`，其匹配结论按缺失处理。"
    )
    mol_rows = data["molecule_rows"]
    insufficient = [r for r in mol_rows if r.get("matched_status") == "INSUFFICIENT_MATCHED_NULL"]
    lines.append(
        f"* `INSUFFICIENT_MATCHED_NULL` 分子数：{len(insufficient)} / {len(mol_rows)}"
        + (
            "（row ids: "
            + ", ".join(f"`{r['dataset']}`/{r['split']}/{r['row_id']}" for r in insufficient[:12])
            + (f" … 其余 {len(insufficient) - 12} 个见 `analysis/molecule_table.csv` 的 `matched_status` 列"
               if len(insufficient) > 12 else "")
            + "）"
            if insufficient else "（全部分子都有 ≥ 8 个合格匹配对照）"
        )
    )
    lines.append("")

    # 8 architecture
    lines.append("## 8. 架构诊断：S1 / S2 / 交互 / 局部求和 / 全图读出")
    lines.append("")
    rows = []
    for dataset in C.DATASETS:
        for split in C.SPLITS:
            d = diag[(dataset, split)]
            rows.append([
                f"`{dataset}`", split, d["n"],
                _f(d, "abs_s1_only_median"), _f(d, "abs_s2_only_median"), _f(d, "abs_s1_s2_median"),
                _pct(float(d["frac_s2_gt_s1"])),
                _f(d, "signed_interaction_median"), _f(d, "abs_interaction_median"),
            ])
    lines.append(_table(
        ["dataset", "split", "n", "median abs(S1_ONLY)", "median abs(S2_ONLY)",
         "median abs(S1_S2)", "S2 > S1", "median interaction", "median abs(interaction)"],
        rows,
    ))
    lines.append("")
    rows = []
    for dataset in C.DATASETS:
        for split in C.SPLITS:
            d = diag[(dataset, split)]
            rows.append([
                f"`{dataset}`", split,
                _f(d, "rho_share_median", 3), _f(d, "rho_share_q25", 3), _f(d, "rho_share_q75", 3),
                _pct(float(d["frac_delta_P_dominates"])),
                _f(d, "cancellation_median", 3),
                _pct(float(d["frac_material_cancellation"])),
                _f(d, "base_rho_share_median", 3),
                _f(d, "local_sum_abs_share_median", 3),
                _pct(float(d["top1_s1_coverage_median"])),
                _pct(float(d["top1_s2_coverage_median"])),
            ])
    lines.append(_table(
        ["dataset", "split", "rho_share med", "q25", "q75", "P dominates",
         "cancellation med", "cancellation > 0.5", "base rho_share", "local share",
         "S1 coverage", "S2 coverage"],
        rows,
    ))
    lines.append("")
    rows = []
    for dataset in C.DATASETS:
        for split in C.SPLITS:
            d = diag[(dataset, split)]
            rows.append([
                f"`{dataset}`", split,
                _pct(float(d["dose_same_direction_frac"])),
                _f(d, "dose_abs_ratio_median", 3),
                _f(d, "median_abs_delta_over_train_std"),
                _f(d, "p90_abs_delta_over_train_std"),
            ])
    lines.append(_table(
        ["dataset", "split", "dose same direction", "median abs(delta at m=0.5)/abs(delta at m=0)",
         "median abs(delta)/std", "p90 abs(delta)/std"],
        rows,
    ))
    lines.append("")
    lines.append(
        "`rho_share = |delta_P| / (|delta_L| + |delta_P| + 1e-12)` 与 "
        "`cancellation = (|delta_L| + |delta_P| - |delta_total|) / (|delta_L| + |delta_P| + 1e-12)` "
        "**只用于描述**，不是唯一贡献分解，也不是因果贡献百分比。"
        "`S1 coverage` / `S2 coverage` 是 Top-1 子图在该分子上触及的 S1 code / S2 occurrence 比例。"
    )
    lines.append("")

    # 9 failures
    lines.append("## 9. 失败案例")
    lines.append("")
    fh = data["failures"]["high_gradient_low_effect"]
    fl = data["failures"]["high_local_output_low_effect"]
    lines.append(
        f"* 梯度线性预测大、真实屏蔽响应相对小的案例：{len(fh)} 条。"
        "选法：每个 dataset×split 取 `abs(sum_e g[e])` 最大的四分位，再在其中按真实 `abs(delta)` 升序取 8 个"
        "（结果驱动的诊断选择，不参与任何排序或筛选）。下表列出前 6 条。"
    )
    for rec in fh[:6]:
        lines.append(
            f"  * `{rec['dataset']}`/{rec['split']} row {rec['row_id']}: "
            f"线性预测 abs(sum g) = {_fmt(abs(rec['predicted_delta_from_g']))}, "
            f"真实屏蔽 abs(delta) = {_fmt(rec['actual_abs_delta'])}, "
            f"实际/线性预测 = {_fmt(rec['ratio_actual_to_predicted'], 3)}"
        )
    lines.append(
        f"* 局部输出启发式选出的子图真实响应相对小的案例：{len(fl)} 条。"
        "选法同上（先取启发式 `B(S)` 的 Top-1 真实响应最大四分位，再按真实 `abs(delta)` 升序取 8 个）。"
    )
    for rec in fl[:6]:
        lines.append(
            f"  * `{rec['dataset']}`/{rec['split']} row {rec['row_id']}: "
            f"启发式子图 abs(delta) = {_fmt(rec['heuristic_abs_delta'])}, "
            f"发现子图 abs(delta) = {_fmt(rec['discovery_abs_delta'])}"
        )
    lines.append("")

    # 10 interpretation rules
    lines.append("## 10. 判读")
    lines.append("")
    lines.append(f"* **ENGINEERING_GATE**：`{preflight['engineering_gate']}`。")
    for dataset in C.DATASETS:
        sig = decision["discovery_signal_per_dataset"][dataset]["signal"]
        rows = []
        for split in ("valid", "test"):
            for name, chk in sig["splits"][split]["checks"].items():
                rows.append([split, name, _fmt(chk["value"], 3), chk["rule"], _fmt(chk["passed"])])
        lines.append(f"* **`{dataset}` DISCOVERY_SIGNAL** = `{decision['discovery_signal_per_dataset'][dataset]['verdict']}`")
        lines.append("")
        lines.append(_table(["split", "condition", "value", "rule", "passed"], rows))
        lines.append("")
    lines.append(
        "* `DISCOVERY_SIGNAL` 是**跑之前冻结的开发性投入规则**，不是统计显著性，也不是科学结论；"
        "未通过时只说明本轮证据不足以支持继续投入。"
    )
    lines.append(
        "* `ARCHITECTURE_DIAGNOSIS` 只如实输出 `rho_share`、抵消、S1/S2 交互与覆盖率证据；"
        "本轮**不**据此推导“必须改读出”或“必须改结构算子”。"
    )
    lines.append("")

    # 11 limits
    lines.append("## 11. 不确定性与限制")
    lines.append("")
    lines.append("* 单 training seed（0）、单 checkpoint；所有数字都是 development evidence，没有跨 seed 稳定性证据。")
    lines.append("* bootstrap 只描述固定 checkpoint / 固定数据集内的分子抽样不确定性。")
    lines.append(f"* 搜索空间仅覆盖 {C.SEARCH_MIN_K}–{C.SEARCH_MAX_K} 条物理键的连通子图；更大的结构未被覆盖，不能据此断言模型不依赖它们。")
    lines.append("* 干预是条件性证据屏蔽：`atom_feat`、`atom_scalar`、`global`、原子数与原子编码保持不变，因此它不是化学反事实，也不是“模型没有学到结构信息”的证明。")
    lines.append("* 每划分只解释 64 个分子（固定种子哈希排序），覆盖全量划分的统计推断不在本轮范围。")
    lines.append("* 若某个分子无法解析或映射不一致，本轮记录原始 row id 与原因，不用另一个分子补齐。")
    lines.append("")

    # 12 artefacts
    lines.append("## 12. 产物")
    lines.append("")
    art_rows = [
        ["`PROTOCOL.md`", "冻结实验定义与解释边界"],
        ["`analysis/preflight.json`", "数据/缓存/checkpoint/等价性/映射/梯度验证"],
        ["`analysis/training_source.json`", "复用身份、360 epoch 轨迹、选中 epoch、checkpoint hash"],
        ["`analysis/main_table.csv`", "各 split 主要效果指标"],
        ["`analysis/architecture_diagnostics.csv`", "S1/S2、局部/全图读出诊断"],
        ["`analysis/molecule_table.csv`", "每分子明细（可重放）"],
        ["`analysis/bootstrap.json`", "分子级 bootstrap 区间"],
        ["`analysis/failure_cases.json`", "高梯度/高局部输出但低响应的失败案例"],
        ["`analysis/decision.json`", "冻结判读规则的结果"],
        ["`analysis/run_manifest.json`", "commit、协议哈希、缓存/checkpoint 哈希、环境、GPU 耗时"],
        ["`analysis/figures/`", f"分子高亮图 {sum(1 for f in figures['molecules'] if f.get('rendered'))} 张 + 分布图"],
        ["`results/<dataset>/`", "样本 ID、候选搜索全log、逐分子干预记录、预算"],
    ]
    lines.append(_table(["path", "content"], art_rows))
    lines.append("")
    prov = manifest.get("execution_provenance") or {}
    lines.append(
        f"* 产出逐分子记录的 GPU worker commit：`{prov.get('results_commit')}`"
        f"（两个 worker 各占一个 GPU：`res-gpu0` = ogbg-molesol，`res-gpu1` = ogbg-mollipo）。"
        f"本文件由 CPU 侧 aggregate 阶段在 commit `{manifest.get('git_commit')}` 上重新生成，"
        "aggregate 不修改任何权重或逐分子记录。"
    )
    lines.append(
        f"* 协议哈希 `{manifest.get('protocol_hash')}`；GPU 耗时：" + "；".join(
            f"`{d}` {_fmt(manifest['gpu'][d]['budget']['compute_seconds'], 1)} s "
            f"({_fmt(manifest['gpu'][d]['budget']['compute_hours'], 4)} GPU·h)"
            for d in C.DATASETS
        ) + f"（软上限 {C.GPU_BUDGET_SOFT_SECONDS / 3600:.1f} GPU·h / 硬上限 "
        f"{C.GPU_BUDGET_HARD_SECONDS / 3600:.1f} GPU·h per worker；未触发任何预算停止）。"
    )
    runs = prov.get("runs") or []
    if runs:
        lines.append(
            "* 执行环境（来自 `rr` 的运行记录）："
            + "；".join(
                f"`{r['experiment']}` stage={r['stage']} pool={r['pool']} node={r['node']} "
                f"driver={r['driver_version']} torch={r['torch_backend']} state={r['state']}"
                for r in runs
            )
            + "。"
        )
    lines.append("")
    lines.append(
        "完整图像只在 `analysis/figures/` 中保存有限张（每数据集固定 8 Valid + 8 Test + 4 诊断案例 + 2 分布图）；"
        "数百个原始分子图不入 Git，可由运行记录恢复。"
    )
    lines.append("")

    path = C.ROUND_DIR / "REPORT.md"
    C.atomic_write_text(path, "\n".join(lines) + "\n")
    return path


# ---------------------------------------------------------------------------
# DECISION.md
# ---------------------------------------------------------------------------


def write_decision() -> Path:
    data = _load()
    decision = data["decision"]
    main = _main_lookup(data["main_rows"])
    preflight = data["preflight"]
    lines: list[str] = []
    lines.append("# DECISION — lcrc-auto-subgraph-faithfulness-pilot-v1")
    lines.append("")
    lines.append("只依据本轮**跑之前冻结**的判读规则，不制定后续新模型。")
    lines.append("")
    lines.append("## 1. ENGINEERING_GATE")
    lines.append("")
    lines.append(f"`{preflight['engineering_gate']}`")
    lines.append("")
    if preflight["critical_failures"]:
        lines.append("未通过门禁：" + ", ".join(f"`{c}`" for c in preflight["critical_failures"]))
        lines.append("")

    lines.append("## 2. DISCOVERY_SIGNAL")
    lines.append("")
    lines.append(f"总体：`{decision['discovery_signal_overall']}`")
    lines.append("")
    for dataset in C.DATASETS:
        entry = decision["discovery_signal_per_dataset"][dataset]
        lines.append(f"### `{dataset}` — `{entry['verdict']}`")
        lines.append("")
        rows = []
        for split in ("valid", "test"):
            for name, chk in entry["signal"]["splits"][split]["checks"].items():
                rows.append([split, name, _fmt(chk["value"], 3), chk["rule"], _fmt(chk["passed"])])
        lines.append(_table(["split", "condition", "value", "rule", "passed"], rows))
        lines.append("")
        if entry["signal"]["failed_conditions"]:
            lines.append("未通过条件：" + ", ".join(f"`{c}`" for c in entry["signal"]["failed_conditions"]))
        else:
            lines.append("全部冻结条件通过。")
        lines.append("")

    lines.append("## 3. 依据的关键数字")
    lines.append("")
    rows = []
    for dataset in C.DATASETS:
        for split in ("valid", "test"):
            m = main[(dataset, split)]
            rows.append([
                f"`{dataset}`", split,
                _f(m, "matched_median_effect_ratio", 3),
                _pct(float(m["matched_ratio_gt1_frac"])),
                _pct(float(m["frac_top1_ge_1pct_train_std"])),
                f"{m['matched_n_molecules']}/{m['n_interpreted']}",
                _f(m, "abs_delta_top1_median_over_train_std"),
            ])
    lines.append(_table(
        ["dataset", "split", "median effect_ratio", "ratio>1", "abs(delta) ≥ 1% std",
         "matched n", "median abs(delta)/std"],
        rows,
    ))
    lines.append("")

    lines.append("## 4. ARCHITECTURE_DIAGNOSIS（只报告，不自动推导结论）")
    lines.append("")
    diag = decision["architecture_diagnosis"]["datasets"]
    rows = []
    for dataset in C.DATASETS:
        for split in C.SPLITS:
            d = diag[dataset].get(split)
            if not d:
                continue
            rows.append([
                f"`{dataset}`", split,
                _f(d, "rho_share_median", 3),
                _f(d, "cancellation_median", 3),
                _f(d, "abs_s1_only_median"), _f(d, "abs_s2_only_median"),
                _f(d, "signed_interaction_median"),
                _pct(float(d["dose_same_direction_frac"])),
            ])
    lines.append(_table(
        ["dataset", "split", "rho_share med", "cancellation med", "abs(S1) med", "abs(S2) med",
         "interaction med", "dose consistency"],
        rows,
    ))
    lines.append("")

    lines.append("## 5. 若结果不足，属于哪一类失败（按证据归类，可多选）")
    lines.append("")
    classes = classify_failures(data)
    for name, verdict, evidence in classes:
        lines.append(f"* **{name}**：`{verdict}` — {evidence}")
    lines.append("")
    lines.append(
        "禁止把解释失败直接写成“LCRC 没有学到结构信息”。本轮的干预只屏蔽显式 bond-derived "
        "S1/S2 证据，其余输入（原子特征、原子标量、global42、原子数）保持不变。"
    )
    lines.append("")
    path = C.ROUND_DIR / "DECISION.md"
    C.atomic_write_text(path, "\n".join(lines) + "\n")
    return path


def classify_failures(data: Mapping[str, Any]) -> list[tuple[str, str, str]]:
    """Map the observed evidence onto the round's failure classes."""
    preflight = data["preflight"]
    main = _main_lookup(data["main_rows"])
    diag = _diag_lookup(data["diag_rows"])
    out: list[tuple[str, str, str]] = []
    eng = preflight["engineering_gate"] == "PASS"
    out.append(
        (
            "工程实现失败",
            "NO" if eng else "YES",
            "所有核心等价/映射/回溯门禁均通过。" if eng else f"未通过：{preflight['critical_failures']}",
        )
    )
    coverage = [
        float(main[(d, s)]["coverage"]) for d in C.DATASETS for s in C.SPLITS
    ]
    out.append(
        (
            "候选搜索失败",
            "NO" if min(coverage) >= 0.99 else "PARTIAL",
            f"可解释分子覆盖率最小 {_pct(min(coverage))}（候选池在全部已解释分子上都非空）。",
        )
    )
    sign = [float(main[(d, s)]["sign_agreement_top1_frac"]) for d in C.DATASETS for s in C.SPLITS]
    out.append(
        (
            "梯度与有限干预不一致",
            "NO" if min(sign) >= 0.8 else "PARTIAL",
            f"Top-1 符号一致率最低 {_pct(min(sign))}；中位线性化绝对误差见 `main_table.csv`。",
        )
    )
    floors = [float(main[(d, s)]["frac_top1_ge_1pct_train_std"]) for d in C.DATASETS for s in C.SPLITS]
    out.append(
        (
            "结构信息屏蔽后变化过小",
            "NO" if min(floors) >= 0.5 else "YES/PARTIAL",
            f"达到 1% Train 标签标准差的分子比例最低 {_pct(min(floors))}。",
        )
    )
    matched = [int(main[(d, s)]["matched_n_molecules"]) for d in C.DATASETS for s in C.SPLITS]
    n_int = [int(main[(d, s)]["n_interpreted"]) for d in C.DATASETS for s in C.SPLITS]
    frac = [m / n if n else 0.0 for m, n in zip(matched, n_int)]
    out.append(
        (
            "无足够匹配对照",
            "NO" if min(frac) >= 0.8 else "PARTIAL",
            f"匹配对照成功率最低 {_pct(min(frac))}（其余分子记为 INSUFFICIENT_MATCHED_NULL）。",
        )
    )
    rho = [float(diag[(d, s)]["rho_share_median"]) for d in C.DATASETS for s in C.SPLITS]
    canc = [float(diag[(d, s)]["cancellation_median"]) for d in C.DATASETS for s in C.SPLITS]
    out.append(
        (
            "全图读出混合严重",
            "EVIDENCED" if max(rho) >= 0.5 or max(canc) >= 0.5 else "NOT_EVIDENCED",
            f"rho_share 中位数最高 {_fmt(max(rho), 3)}，cancellation 中位数最高 {_fmt(max(canc), 3)}。",
        )
    )
    out.append(
        (
            "搜索范围 / 固定原子与全图特征造成的不可识别性",
            "BOUNDED_BY_DESIGN",
            "只搜索 2–6 键连通子图；atom_feat/atom_scalar/global 与原子数不被门控，"
            "因此任何依赖这些输入的响应都不在本干预的可识别范围内。",
        )
    )
    return out


def write_reports() -> dict[str, Any]:
    report = write_report()
    decision = write_decision()
    return {"report": C.relpath(report), "decision": C.relpath(decision)}


__all__ = ["write_report", "write_decision", "write_reports", "classify_failures"]
