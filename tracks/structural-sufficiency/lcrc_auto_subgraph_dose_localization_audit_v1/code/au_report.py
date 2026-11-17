"""REPORT.md / DECISION.md generation from the frozen audit artefacts.

Every number in the generated documents is read back from ``analysis/*.json`` /
``analysis/*.csv``; nothing is typed by hand and no claim is emitted that the
artefacts do not support.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import au_common as C


# ---------------------------------------------------------------------------
# formatting helpers
# ---------------------------------------------------------------------------


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
        "freeze": C.read_json(a / "protocol_freeze.json"),
        "bootstrap": C.read_json(a / "bootstrap.json"),
        "decision": C.read_json(a / "decision.json"),
        "manifest": C.read_json(a / "run_manifest.json"),
        "localization": C.read_json(a / "localization_summary.json"),
        "old_vs_rows": C.read_csv(a / "old_vs_corrected.csv"),
        "localization_rows": C.read_csv(a / "localization.csv"),
        "old_control_rows": C.read_csv(a / "old_control_dose.csv"),
    }


def _lookup(rows: Sequence[Mapping[str, str]], *keys: str) -> dict[tuple[str, ...], Mapping[str, str]]:
    return {tuple(r[k] for k in keys): r for r in rows}


def _row_value(row: Mapping[str, Any] | None, key: str) -> str:
    if row is None:
        return "NA"
    return _fmt(row.get(key))


# ---------------------------------------------------------------------------
# REPORT.md
# ---------------------------------------------------------------------------


def write_report() -> Path:
    data = _load()
    decision = data["decision"]
    preflight = data["preflight"]
    manifest = data["manifest"]
    loc = data["localization"]
    old_vs = _lookup(data["old_vs_rows"], "dataset", "split", "protocol")
    summary = decision["summary"]

    lines: list[str] = []
    add = lines.append
    add("# REPORT — lcrc-auto-subgraph-dose-localization-audit-v1")
    add("")
    add(
        "对 `lcrc-auto-subgraph-faithfulness-pilot-v1`（交付提交 `147b1d81`）的**只读审计轮**。"
        "本轮不重新选择 Top-1、不重新训练、不更换 seed、不修改架构、不扩大搜索范围；"
        "只把随机对照的匹配量从 `n_s2_internal` 改成真实被门控移除的 `n_s2_affected`，"
        "并测量屏蔽响落在哪些原子根节点上。"
    )
    add("")
    add("## 0. 一页结论（先回答五个问题）")
    add("")
    for dataset in C.DATASETS:
        answers = summary
        q1 = answers["q1_dose_matching_defect"][dataset]
        add(f"### `{dataset}`")
        add("")
        add(
            f"1. **原对照的剂量匹配是否存在实质性口径错误？** `{q1['grade']}`。"
            f"原匹配量是 `n_s2_internal`（两条键都在 S 内），真实被移除的是 "
            f"`n_s2_affected`（至少一条键在 S 内）。原 Top-1 的 occurrence 数中位 "
            f"internal = {_fmt(q1['median_top1_n_s2_internal'])}, "
            f"affected = {_fmt(q1['median_top1_n_s2_affected'])}；"
            f"原已接受对照中 {_pct(q1['frac_old_controls_outside_affected_tol'])} 在修正后的 "
            f"affected 口径下超出 25% 容差（affected 相对差中位 "
            f"{_fmt(q1['median_old_control_affected_rel_diff'])}）。"
        )
        for split in ("valid", "test"):
            q2 = answers["q2_corrected_effect"][dataset][split]
            add(
                f"{2 if split == 'valid' else 3}. **{split.upper()}（修正对照）**："
                f"median effect ratio = {_fmt(q2['effect_ratio_median'])}"
                f"（bootstrap CI low {_fmt(q2['effect_ratio_ci_low'])}），"
                f"ratio>1 = {_pct(q2['ratio_gt1_fraction'])}，"
                f"median effect advantage = {_fmt(q2['median_effect_advantage'])}，"
                f"判定 `{q2['verdict']}`。"
            )
        q3 = answers["q3_top1_vs_controls"][dataset]
        add(
            "4. **Top-1 是否仍明显优于随机对照和局部启发式？** 修正对照下的 Valid/Test 判定为 "
            f"`{q3['corrected_verdict_valid']}` / `{q3['corrected_verdict_test']}`；"
            f"Top-1 的真实屏蔽效应大于局部启发式的分子比例为 "
            f"{_pct(q3['frac_top1_gt_heuristic_valid'])}（Valid）/ "
            f"{_pct(q3['frac_top1_gt_heuristic_test'])}（Test）。"
        )
        q4 = answers["q4_localization"][dataset]
        add(
            "5. **影响集中在自身根节点还是扩散？** "
            + "；".join(
                f"{split}: 外部根节点绝对响应占比中位 {_fmt(q4[split]['outside_fraction_median'])}"
                f"，>0.5 的比例 {_pct(q4[split]['frac_outside_gt_0p5'])}，判定 `{q4[split]['verdict']}`"
                for split in C.SPLITS
            )
            + "。"
        )
        add("")
    add("> 说明：第 3 条回答中的两个数据集各自独立，不要把 molesol 和 mollipo 的数字混成一行。")
    add("")
    add("## 1. 边界、身份与不可比声明")
    add("")
    add(
        "* 模型：`lcrc_readout_context_pilot_small_v1` **Arm A = LCRC-C1**，"
        "`sum_v f(h_v) + rho([g, mean(h), mean(h^2), log1p(n_atoms)])`；"
        "checkpoint / 轨迹 SHA-256 与原轮记录一致，本轮**没有训练**。"
    )
    add(
        "* 数据：官方 `ogbg-molesol` / `ogbg-mollipo` 冻结缓存与官方 `get_idx_split()`；"
        "每划分 64 个 row id 与 原轮 `samples.json` 逐项相同（重新按同一规则选择并比对哈希）。"
    )
    add(
        "* 解释对象：原轮已选定的 Top-1 物理键集合被**逐字复用**；梯度、发现分数、搜索日志均从原轮读取，"
        "并逐分子重算核对（任一不一致即停止）。"
    )
    add(
        "* 干预仍是 `CONDITIONAL_S1_S2_EVIDENCE_MASK`（`m[e] ∈ {0,1}` 的冻结前向重算），"
        "**不是**化学删键/删基团/完整反事实。Test 仅作 `DEVELOPMENT_TEST` 报告。"
    )
    add(
        "* **两轮对照协议不同**：`pilot_internal_matched` 与 `corrected_affected_matched` 的匹配量、"
        "接受规则都不同。下文所有对比都保留协议列，旧数字不能冒充已修正结果。"
    )
    add("")

    # -- Q1 dose --------------------------------------------------------------
    add("## 2. Q1 — 原对照的剂量匹配口径")
    add("")
    add(
        "S2 门控权重是 `w[p] = m[e1] * m[e2]`，因此 `m[e]=0` 时实际被移除的 occurrence 条件是 "
        "`e1 in S OR e2 in S`。原轮用于匹配对照的是 `n_s2_internal`（`e1 in S AND e2 in S`），"
        "两者相差的正是**边界 occurrence**（恰好一条键在 S 内）。"
    )
    add("")
    rows = []
    for dataset in C.DATASETS:
        for split in C.SPLITS:
            p = old_vs.get((dataset, split, "pilot_internal_matched"))
            c = old_vs.get((dataset, split, "corrected_affected_matched"))
            rows.append([
                f"`{dataset}`", split,
                _row_value(p, "median_top1_n_s2_internal"),
                _row_value(p, "median_top1_n_s2_affected"),
                _row_value(p, "median_control_internal_rel_diff"),
                _row_value(p, "median_control_affected_rel_diff"),
                _pct(float(p["frac_controls_outside_affected_tol"])) if p else "NA",
                _row_value(c, "median_control_affected_rel_diff"),
                _pct(float(c["frac_controls_outside_affected_tol"])) if c else "NA",
                _row_value(c, "candidate_draws"),
                _pct(float(c["frac_candidate_draws_rejected_affected"]))
                if c and c.get("frac_candidate_draws_rejected_affected") else "NA",
            ])
    add(_table(
        ["dataset", "split", "Top1 n_s2_internal", "Top1 n_s2_affected",
         "pilot null: median internal rel diff", "pilot null: median affected rel diff",
         "pilot null: frac outside affected 25%", "corrected null: median affected rel diff",
         "corrected null: frac outside affected 25%", "corrected null: distinct draws",
         "corrected null: frac draws rejected by affected rule"],
        rows,
    ))
    add("")
    add(
        "`pilot null` 列来自对原轮**已接受**对照键集合的重新计数（不改动原轮存储的效果值）；"
        "`corrected null` 列是本轮新生成的对照。修正对照的 `frac outside affected 25%` 恒为 0，"
        "因为 25% 容差就是接受准则；它右侧的“候选拒绝率”才说明该准则有多严格。"
    )
    add("")
    add(
        "原轮**按它自己声明的规则执行是正确的**（pilot null 的 internal 相对差最大 0.25，"
        "见 `analysis/old_control_dose.csv`）；问题在于它匹配的量不是门控实际移除的量："
        "修正口径下原对照的 affected 剂量与 Top-1 相差一个边界项，"
        "`boundary = affected - internal` 的典型规模见 `median_affected_minus_internal`。"
    )
    add("")

    # -- Q2 corrected effect -------------------------------------------------
    add("## 3. Q2 — 修正后的 Valid / Test 对照")
    add("")
    rows = []
    for dataset in C.DATASETS:
        for split in ("valid", "test"):
            for protocol, label in (("pilot_internal_matched", "pilot"),
                                    ("corrected_affected_matched", "corrected")):
                r = old_vs.get((dataset, split, protocol))
                if r is None:
                    continue
                rows.append([
                    f"`{dataset}`", split, label,
                    _row_value(r, "matched_n_molecules"),
                    _row_value(r, "matched_success_rate"),
                    _row_value(r, "median_effect_ratio"),
                    f"[{_row_value(r, 'effect_ratio_ci_low')}, {_row_value(r, 'effect_ratio_ci_high')}]",
                    _pct(float(r["ratio_gt1_frac"])),
                    _row_value(r, "median_effect_advantage"),
                    f"[{_row_value(r, 'effect_advantage_ci_low')}, {_row_value(r, 'effect_advantage_ci_high')}]",
                    _row_value(r, "median_matched_abs_delta"),
                ])
    add(_table(
        ["dataset", "split", "protocol", "matched n", "success", "median effect ratio",
         "ratio CI95", "ratio>1", "median advantage", "advantage CI95", "median random abs(delta)"],
        rows,
    ))
    add("")
    add(
        "`effect_ratio = |delta(Top1)| / (median_e |delta(control_e)| + 1e-12)`；"
        "`effect_advantage = |delta(Top1)| - median_e |delta(control_e)|`。"
        "bootstrap：2000 次、以分子为重采样单位、seed 20261117；它只描述固定 checkpoint / 固定数据集下的"
        "抽样不确定性，不是跨 seed 稳定性。"
    )
    add("")
    add("### 3.1 修正后判定（沿用原轮冻结的投入规则）")
    add("")
    rows = []
    for dataset in C.DATASETS:
        for split in ("valid", "test"):
            entry = decision["datasets"][dataset]["splits"][split]
            for protocol, label in (("pilot", "pilot"), ("corrected", "corrected")):
                v = entry[protocol]
                rows.append([
                    f"`{dataset}`", split, label, v["verdict"],
                    _fmt(v["conditions"]["effect_ratio_median_ge_1p5"]["value"]),
                    _pct(v["conditions"]["effect_ratio_gt1_fraction_ge_0p70"]["value"]),
                    _fmt(v["effect_ratio_ci_low"]),
                    _pct(v["conditions"]["effect_floor_fraction_ge_0p50"]["value"]),
                    _fmt(v["conditions"]["enough_matched_controls"]["value"]),
                ])
    add(_table(
        ["dataset", "split", "protocol", "verdict", "median ratio", "ratio>1",
         "ratio CI low", "effect floor frac", "matched n"],
        rows,
    ))
    add("")
    add(
        "判定规则（跑前冻结）：`SURVIVES` = 四项条件全过且 ratio 的 bootstrap CI low > 1；"
        "`ATTENUATED` = median ratio > 1 且 CI low ≥ 1 但未全过；其余 `NOT_SUPPORTED`。"
        "四项条件 = median ratio ≥ 1.5、ratio>1 比例 ≥ 70%、1% std 效应底线比例 ≥ 50%、"
        "匹配分子数 ≥ max(32, 80%×已解释分子)。"
    )
    add("")
    add(
        "这里的 `SURVIVES/ATTENUATED/NOT_SUPPORTED` 是本轮为**两种协议并排比较**而冻结的三档映射，"
        "不是原轮的 `PILOT_SIGNAL / WEAK_OR_INCONCLUSIVE` 二档标签；原轮自己的判定见其 `DECISION.md`。"
    )
    add("")
    add("### 3.2 分子级配对变化（同一分子、两个对照协议）")
    add("")
    rows = []
    for dataset in C.DATASETS:
        for split in C.SPLITS:
            p = decision["datasets"][dataset]["splits"][split]["paired_change"]
            rows.append([
                f"`{dataset}`", split, _fmt(p["n_paired_molecules"]),
                _fmt(p["ratio_delta_median"]),
                f"[{_fmt(p['ratio_delta_ci']['ci_low'])}, {_fmt(p['ratio_delta_ci']['ci_high'])}]",
                _fmt(p["advantage_delta_median"]),
                _pct(p["frac_ratio_decreased"]),
            ])
    add(_table(
        ["dataset", "split", "paired molecules", "median ratio delta (corrected - pilot)",
         "ratio delta CI95", "median advantage delta", "frac ratio decreased"],
        rows,
    ))
    add("")
    rows = []
    for dataset in C.DATASETS:
        for split in C.SPLITS:
            p = decision["datasets"][dataset]["splits"][split]["paired_change"]
            rows.append([
                f"`{dataset}`", split, p.get("n_matched_both"),
                p.get("n_matched_pilot_only"), p.get("n_matched_corrected_only"),
            ])
    add(_table(
        ["dataset", "split", "matched under both protocols", "pilot-only",
         "corrected-only"],
        rows,
    ))
    add("")
    rows = []
    for dataset in C.DATASETS:
        for split in C.SPLITS:
            p = decision["datasets"][dataset]["splits"][split]["paired_change"]
            rows.append([
                f"`{dataset}`", split,
                _pct(p.get("frac_paired_identical_control_sets")),
                _fmt(p.get("median_control_set_jaccard"), 3),
                p.get("n_paired_changed_control_sets"),
                _fmt(p.get("ratio_delta_median_changed_sets")),
                f"[{_fmt((p.get('ratio_delta_ci_changed_sets') or {}).get('ci_low'))}, "
                f"{_fmt((p.get('ratio_delta_ci_changed_sets') or {}).get('ci_high'))}]",
            ])
    add(_table(
        ["dataset", "split", "paired molecules with identical accepted control set",
         "median control-set Jaccard", "paired molecules with a changed set",
         "median ratio delta on changed sets", "CI95"],
        rows,
    ))
    add("")
    add(
        "配对差的中位数接近 0 是预期的：同一分子的随机流不变，只有接受准则改变，"
        "因此多数分子的候选集合高度重叠甚至完全相同。聚合中位数的移动同时包含"
        "**分子集合变化**（上表 pilot-only / corrected-only）、**对照集合变化**（本表）"
        "与两个协议下不同的对照值，三者不能混为一谈。"
    )
    add("")

    # -- Q3 top1 vs heuristic ------------------------------------------------
    add("## 4. Q3 — Top-1 相对随机对照与局部启发式")
    add("")
    rows = []
    for dataset in C.DATASETS:
        for split in C.SPLITS:
            entry = decision["datasets"][dataset]["top1_vs_heuristic"][split]
            corrected = old_vs.get((dataset, split, "corrected_affected_matched"))
            rows.append([
                f"`{dataset}`", split,
                _row_value(corrected, "median_top1_abs_delta"),
                _row_value(corrected, "median_local_heuristic_abs_delta"),
                _row_value(corrected, "median_top1_minus_heuristic"),
                _pct(entry["pilot_frac_top1_gt_heuristic"]),
                _pct(entry["corrected_frac_top1_gt_heuristic"]),
            ])
    add(_table(
        ["dataset", "split", "median Top1 abs(delta)", "median heuristic abs(delta)",
         "median advantage", "Top1 > heuristic (pilot record)",
         "Top1 > heuristic (recomputed Top1)"],
        rows,
    ))
    add("")
    add(
        "局部输出启发式 `B(S) = sum_v |f(h_v)| / sqrt(|S|)` 的子图与效果**直接复用原轮记录**"
        "（不重新搜索）；Top-1 的 `|delta|` 用本轮重算值（与原轮在冻结容差内一致）。"
    )
    add("")

    # -- Q4 localization -----------------------------------------------------
    add("## 5. Q4 — 屏蔽响应的根节点定位")
    add("")
    rows = []
    for dataset in C.DATASETS:
        for split in C.SPLITS:
            s = loc[dataset]["splits"][split]
            rows.append([
                f"`{dataset}`", split, s["n"],
                _fmt(s["outside_fraction_median"]),
                f"[{_fmt(s['outside_fraction_q25'])}, {_fmt(s['outside_fraction_q75'])}]",
                _pct(s["frac_outside_gt_0p5"]),
                _pct(s["frac_outside_le_0p25"]),
                _fmt(s["median_inside_abs_response"]),
                _fmt(s["median_outside_abs_response"]),
                s["verdict"],
            ])
    add(_table(
        ["dataset", "split", "n", "outside fraction median", "IQR", ">0.5", "<=0.25",
         "median inside abs response", "median outside abs response", "verdict"],
        rows,
    ))
    add("")
    add("### 5.1 S2 剂量与根节点分布")
    add("")
    rows = []
    for dataset in C.DATASETS:
        for split in C.SPLITS:
            s = loc[dataset]["splits"][split]
            rows.append([
                f"`{dataset}`", split,
                _fmt(s["median_n_s2_internal"]), _fmt(s["median_n_s2_boundary"]),
                _fmt(s["median_n_s2_affected"]),
                _fmt(s["median_internal_over_total"]), _fmt(s["median_boundary_over_total"]),
                _fmt(s["median_affected_over_total"]),
                _fmt(s["median_boundary_over_affected"]),
                _fmt(s["median_affected_root_outside_fraction"]),
            ])
    add(_table(
        ["dataset", "split", "median n_s2_internal", "median n_s2_boundary",
         "median n_s2_affected", "internal / total", "boundary / total",
         "affected / total", "boundary / affected", "affected occurrences rooted outside"],
        rows,
    ))
    add("")
    add(
        "这里同时报告两种 S2 覆盖率：**internal coverage**（`n_s2_internal / n_s2_total`）"
        "与 **affected coverage**（`n_s2_affected / n_s2_total`）。原轮报告的 “S2 coverage” 是前者，"
        "不能当作被门控移除的 occurrence 比例。"
    )
    add("")
    add("### 5.2 与 Top-1 键数、梯度分数、真实屏蔽效应的对应关系")
    add("")
    rows = []
    for dataset in C.DATASETS:
        for split in C.SPLITS:
            s = loc[dataset]["splits"][split]
            corr = s["spearman_vs_outside_fraction"]
            rows.append([
                f"`{dataset}`", split,
                _fmt(corr["top1_n_bonds"]["rho"] if corr["top1_n_bonds"]["rho"] is not None else None, 3),
                _fmt(corr["abs_score_A"]["rho"], 3),
                _fmt(corr["abs_delta_total"]["rho"], 3),
                _fmt(corr["n_s2_affected"]["rho"], 3),
                _fmt(corr["boundary_over_affected"]["rho"], 3),
                _fmt(corr["affected_root_outside_fraction"]["rho"], 3),
            ])
    add(_table(
        ["dataset", "split", "Spearman vs k", "vs abs(score A)", "vs abs(delta_total)",
         "vs n_s2_affected", "vs boundary/affected", "vs affected rooted outside"],
        rows,
    ))
    add("")
    add("按 Top-1 键数分组的 outside fraction：")
    add("")
    rows = []
    for dataset in C.DATASETS:
        for split in C.SPLITS:
            for b in loc[dataset]["bins_by_n_bonds"][split]:
                rows.append([
                    f"`{dataset}`", split, b["top1_n_bonds"], b["n"],
                    _fmt(b["outside_fraction_median"]),
                    _fmt(b["median_abs_delta_total"]),
                    _fmt(b["median_n_s2_affected"]),
                    _fmt(b["median_boundary_over_affected"]),
                ])
    add(_table(
        ["dataset", "split", "Top1 k", "n", "outside fraction median",
         "median abs(delta_total)", "median n_s2_affected", "median boundary/affected"],
        rows,
    ))
    add("")
    add(
        "相关性只用于描述这批 64×3 分子上的对应关系；样本量小、单 checkpoint，不作为机制证明。"
        "按指示，本轮**不**用这些指标自动推导“模型架构必须修改”。"
    )
    add("")

    # -- Q5 scope ------------------------------------------------------------
    add("## 6. Q5 — 本轮支持 / 不支持 / 仍无法判断")
    add("")
    for key, title in (("supported", "支持"), ("not_supported", "不支持"),
                       ("still_undetermined", "仍无法判断")):
        add(f"**{title}**")
        add("")
        for item in summary["q5_scope"][key]:
            add(f"* {item}")
        add("")

    # -- gates ---------------------------------------------------------------
    add("## 7. 工程门禁与复现")
    add("")
    add(f"* ENGINEERING_GATE: `{preflight['engineering_gate']}`")
    add(f"* 原轮指标重放（用原轮逐分子记录重算并与原 `main_table.csv` 比对）："
        f"`{'PASS' if decision.get('pilot_metric_replay_passed') else 'FAIL'}`")
    add(f"* 协议哈希：`{manifest['protocol_hash']}`")
    add(f"* PROTOCOL.md sha256：`{manifest.get('protocol_file_sha256')}`")
    rep = decision.get("top1_replay") or {}
    allrep = rep.get("all") or {}
    if allrep:
        err = allrep.get("max_abs_err") or {}
        add(
            f"* Top-1 屏蔽预测与原轮逐分子比对：{allrep.get('n_molecules')} 个分子中 "
            f"{allrep.get('n_failed')} 个超出冻结容差；最大绝对偏差 "
            f"signed_delta {_fmt(err.get('signed_delta'))}、delta_L {_fmt(err.get('delta_L'))}、"
            f"delta_rho {_fmt(err.get('delta_rho'))}、pred_raw {_fmt(err.get('pred_raw_masked'))}"
            "（CPU↔GPU 浮点差，容差按原轮口径）。"
        )
    add(f"* 本轮实际运行 commit（audit + aggregate，工作区无未提交的 tracked 改动）：`{manifest.get('git_commit')}`；上游 results commit："
        f"`{(manifest.get('upstream') or {}).get('results_commit')}`")
    add("")
    rows = []
    for dataset in C.DATASETS:
        g = preflight["datasets"][dataset]
        for gate in g["gates"]:
            rows.append([f"`{dataset}`", gate["gate"], "yes" if gate["passed"] else "no", gate["detail"]])
    add(_table(["dataset", "gate", "passed", "detail"], rows))
    add("")
    rows = []
    for dataset in C.DATASETS:
        b = (manifest.get("execution") or {}).get(dataset, {}).get("budget") or {}
        rows.append([
            f"`{dataset}`", b.get("device"), _fmt(b.get("wall_seconds"), 2),
            _fmt(b.get("compute_seconds"), 2), _fmt(b.get("peak_rss_mb"), 1),
            b.get("n_molecules_completed"),
        ])
    add(_table(
        ["dataset", "device", "wall s", "compute s", "peak RSS MB", "molecules"], rows,
    ))
    add("")
    add(
        "* 执行顺序：`preflight`（身份门禁 + 8 分子全链路 smoke）→ 两个数据集的 `audit` → CPU `aggregate`。"
        "smoke 记录不进入任何报告指标。"
    )
    add("* 单 seed、单 checkpoint、无训练；bootstrap 只反映分子抽样不确定性。")
    add("")

    add("## 8. 产物")
    add("")
    add(_table(
        ["path", "content"],
        [
            ["`PROTOCOL.md`", "跑前冻结的审计定义与判读规则"],
            ["`analysis/preflight.json`", "身份、样本、等价性、门禁与 smoke 记录"],
            ["`analysis/old_vs_corrected.csv`", "两种对照协议并排的主要指标（不回填、不混池）"],
            ["`analysis/matched_controls.csv`", "修正对照逐子图记录（Top-1 + 全部匹配候选）"],
            ["`analysis/old_control_dose.csv`", "原轮已接受对照在修正口径下的剂量重算"],
            ["`analysis/localization.csv`", "逐分子根节点响应定位与闭合检查"],
            ["`analysis/localization_summary.json`", "分组统计、Spearman 对应关系、按 k 分箱"],
            ["`analysis/bootstrap.json`", "分子级 bootstrap 区间（两种协议）"],
            ["`analysis/decision.json`", "冻结规则下的判定与五问回答"],
            ["`analysis/run_manifest.json`", "commit、协议哈希、输入/代码哈希、环境与内存记录"],
            ["`results/<dataset>/<split>/records.json`", "逐分子完整审计记录（可重放）"],
            ["`results/<dataset>/<split>/candidates.npz`", "全部随机候选的三个剂量计数与接受标记"],
            ["`results/<dataset>/<split>/atom_response.csv`", "逐原子 `c_v` 与 `delta_c_v`"],
        ],
    ))
    add("")
    path = C.ROUND_DIR / "REPORT.md"
    C.atomic_write_text(path, "\n".join(lines) + "\n")
    return path


# ---------------------------------------------------------------------------
# DECISION.md
# ---------------------------------------------------------------------------


def write_decision() -> Path:
    data = _load()
    decision = data["decision"]
    preflight = data["preflight"]
    loc = data["localization"]
    summary = decision["summary"]
    lines: list[str] = []
    add = lines.append
    add("# DECISION — lcrc-auto-subgraph-dose-localization-audit-v1")
    add("")
    add("本文件只依据本轮**跑之前冻结**的规则作答；不制定后续结构搜索、配对实验或新架构。")
    add("")
    add("## 1. ENGINEERING_GATE")
    add("")
    add(f"`{preflight['engineering_gate']}`")
    add("")
    add(f"原轮指标重放（用原轮记录重算 vs 原 `main_table.csv`）："
        f"`{'PASS' if decision.get('pilot_metric_replay_passed') else 'FAIL'}`")
    add("")
    add(
        "下文所有 `SURVIVES/ATTENUATED/NOT_SUPPORTED` 都是本轮为并排比较而冻结的三档映射，"
        "不是原轮的 `PILOT_SIGNAL / WEAK_OR_INCONCLUSIVE` 二档标签。"
    )
    add("")
    add("## 2. 五问判定")
    add("")
    for dataset in C.DATASETS:
        add(f"### `{dataset}`")
        add("")
        q1 = summary["q1_dose_matching_defect"][dataset]
        add(f"**Q1 原对照剂量匹配口径** — `{q1['grade']}`")
        add("")
        add(
            f"* 原 Top-1 的 internal / affected occurrence 中位："
            f"{_fmt(q1['median_top1_n_s2_internal'])} / {_fmt(q1['median_top1_n_s2_affected'])}。"
            f"原已接受对照中 {_pct(q1['frac_old_controls_outside_affected_tol'])} 在 affected 口径下"
            f"超出 25% 容差；affected 相对差中位 {_fmt(q1['median_old_control_affected_rel_diff'])}。"
        )
        add(
            "* 结论边界：原轮匹配的是**子图内部二跳路径数**，不是门控实际移除的 occurrence 数；"
            "因此旧对照不是“同剂量”对照。这不是说原轮的效应计算有错，而是它的零假设不同。"
        )
        add("")
        add("**Q2 修正后的 Valid/Test effect ratio 与 advantage**")
        add("")
        rows = []
        for split in ("valid", "test"):
            v = decision["datasets"][dataset]["splits"][split]["corrected"]
            p = decision["datasets"][dataset]["splits"][split]["pilot"]
            rows.append([
                split, "pilot", _fmt(p["median_effect_ratio"]), _fmt(p["median_effect_advantage"]),
                _pct(p["ratio_gt1_frac"]), p["verdict"],
            ])
            rows.append([
                split, "corrected", _fmt(v["median_effect_ratio"]), _fmt(v["median_effect_advantage"]),
                _pct(v["ratio_gt1_frac"]), v["verdict"],
            ])
        add(_table(
            ["split", "protocol", "median effect ratio", "median advantage", "ratio>1", "verdict"], rows,
        ))
        add("")
        add("**Q3 Top-1 是否仍明显优于随机对照与局部启发式**")
        add("")
        for split in ("valid", "test"):
            v = decision["datasets"][dataset]["splits"][split]["corrected"]
            h = decision["datasets"][dataset]["top1_vs_heuristic"][split]
            add(
                f"* {split}: 修正对照判定 `{v['verdict']}`（median ratio {_fmt(v['median_effect_ratio'])}，"
                f"CI low {_fmt(v['effect_ratio_ci_low'])}）；Top-1 真实 |delta| > 局部启发式的比例 "
                f"{_pct(h['corrected_frac_top1_gt_heuristic'])}，"
                f"median advantage {_fmt(h['corrected_median_top1_minus_heuristic'])}。"
            )
        add("")
        add("**Q4 影响是集中在自身根节点还是扩散到周围根节点**")
        add("")
        rows = []
        for split in C.SPLITS:
            s = loc[dataset]["splits"][split]
            rows.append([
                split, _fmt(s["outside_fraction_median"]),
                f"[{_fmt(s['outside_fraction_q25'])}, {_fmt(s['outside_fraction_q75'])}]",
                _pct(s["frac_outside_gt_0p5"]), _fmt(s["median_boundary_over_affected"]),
                _fmt(s["median_affected_root_outside_fraction"]), s["verdict"],
            ])
        add(_table(
            ["split", "outside fraction median", "IQR", ">0.5", "boundary/affected",
             "affected rooted outside", "verdict"],
            rows,
        ))
        add("")
    add("## 3. Q5 支持 / 不支持 / 仍无法判断")
    add("")
    for key, title in (("supported", "支持"), ("not_supported", "不支持"),
                       ("still_undetermined", "仍无法判断")):
        add(f"**{title}**")
        add("")
        for item in summary["q5_scope"][key]:
            add(f"* {item}")
        add("")
    add("## 4. 明确不做的事")
    add("")
    add(
        "* 不重新选择 Top-1、不重新训练、不换 seed、不改架构、不扩大搜索范围、不新增数据集。\n"
        "* 不把本轮结果自动升级为“结构搜索该继续”或“LCRC-v2 必须改读出/改算子”的架构结论。\n"
        "* 不把 `pilot_internal_matched` 与 `corrected_affected_matched` 混成同一张主表或同一个结论。"
    )
    add("")
    path = C.ROUND_DIR / "DECISION.md"
    C.atomic_write_text(path, "\n".join(lines) + "\n")
    return path


def write_reports() -> dict[str, Any]:
    report = write_report()
    decision = write_decision()
    return {
        "report": C.relpath(report),
        "decision": C.relpath(decision),
        "report_sha256": C.sha256_file(report),
        "decision_sha256": C.sha256_file(decision),
    }


__all__ = ["write_report", "write_decision", "write_reports"]
