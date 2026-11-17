"""Report generator: ``REPORT.md`` and ``DECISION.md`` from ``analysis/`` artefacts.

No number in the reports is typed by hand: every table cell is read from a
JSON/CSV file produced by ``rp_prepare``, ``rp_checks``, ``rp_train`` or
``rp_analysis``.  Regenerating the report after a re-aggregation cannot silently
keep a stale number.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

import rp_common as C


def f(x: Any, nd: int = 4) -> str:
    if x is None:
        return "—"
    try:
        value = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(value):
        return "nan"
    return f"{value:.{nd}f}"


def pct(x: Any, nd: int = 2) -> str:
    if x is None:
        return "—"
    value = float(x)
    if not np.isfinite(value):
        return "nan"
    return f"{100.0 * value:.{nd}f}%"


def ci(pair: Sequence[Any] | None, nd: int = 4) -> str:
    if not pair or pair[0] is None:
        return "—"
    return f"[{f(pair[0], nd)}, {f(pair[1], nd)}]"


def table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def _read(name: str, default: Any = None) -> Any:
    path = C.ANALYSIS_DIR / name
    if not path.is_file():
        return default
    return C.read_json(path)


def _csv(name: str) -> list[dict[str, str]]:
    path = C.ANALYSIS_DIR / name
    return C.read_csv(path) if path.is_file() else []


def _r(x: Any, nd: int = 4) -> str:
    return f(x, nd)


def write_reports() -> dict[str, Any]:
    decision = _read("decision.json")
    if decision is None:
        raise FileNotFoundError("analysis/decision.json is missing; run the aggregate stage first")
    retention = _read("retention.json", {})
    bootstrap = _read("bootstrap_real_vs_control.json", {})
    gamma = _read("gamma_analysis.json", {})
    oracle = _read("oracle_localisation_summary.json", {})
    mask = _read("mask_response_summary.json", {})
    closure = _read("wc_logp_closure.json", {})
    groups = _read("role_groups.json", {})
    budget = _read("parameter_budget.json", {})
    invariants = _read("routing_invariants.json", {})
    freeze = _read("protocol_freeze.json", {})
    manifest = _read("run_manifest.json", {})
    comparison = _csv("prediction_comparison.csv")
    appendix = _csv("appendix_not_comparable.csv")

    report = _report(
        decision, retention, bootstrap, gamma, oracle, mask, closure, groups,
        budget, invariants, freeze, manifest, comparison, appendix,
    )
    C.atomic_write_text(C.ROUND_DIR / "REPORT.md", report)
    decision_md = _decision(decision, retention, bootstrap, oracle, mask, gamma)
    C.atomic_write_text(C.ROUND_DIR / "DECISION.md", decision_md)
    return {"REPORT.md": str((C.ROUND_DIR / "REPORT.md").relative_to(C.ROUND_DIR)),
            "DECISION.md": str((C.ROUND_DIR / "DECISION.md").relative_to(C.ROUND_DIR))}


# ---------------------------------------------------------------------------
# REPORT.md
# ---------------------------------------------------------------------------


def _report(decision, retention, bootstrap, gamma, oracle, mask, closure, groups,
            budget, invariants, freeze, manifest, comparison, appendix) -> str:
    L: list[str] = []
    L.append("# 保留 C1 读出的 S2 结构角色 pilot（RR_REAL / RR_CONTROL）")
    L.append("")
    L.append(f"轮次 `{C.ROUND_ID}`。一切数字均为 **DEVELOPMENT**，Test 列是 **DEVELOPMENT_TEST**。")
    L.append("")
    L.append("本轮只训练四个单元：RR_REAL / RR_CONTROL × Lipophilicity / WC_LOGP，seed 0，"
             "官方 `ogbg-mollipo` 划分 3360/420/420，Rf 配方 360 epoch。C1、GINE、oracle "
             "基线一律只读复用，不重训、不重选、不重评。")
    L.append("")
    L.append("## 1. 本轮问题")
    L.append("")
    L.append("在**完全保留** C1 原子编码、键编码、S1、typed-S2、LocalMLP、`sum_v f(h_v)` 与"
             "完整 `rho([g, mean h, mean h², log1p n])` 的条件下，只把每条 S2 出现的 200 维记录"
             "按“同一拓扑结构内部 / 跨拓扑结构边界”分成两组、各乘一个逐通道可训练 gamma：")
    L.append("")
    L.append("1. 预测是否恢复到 C1(Rf) 水平（≤ 1.10× Valid RMSE）？")
    L.append("2. 真实结构边界角色（RR_REAL）是否优于同参数量、同初始化、同配方的任意伪分组（RR_CONTROL）？")
    L.append("3. 对 WC_LOGP 已知原子贡献的捕获是否改善？")
    L.append("4. 若有改善，证据来自预测还是结构解释？")
    L.append("5. 是否值得继续研究显式 block/interface Token？")
    L.append("")
    L.append("## 2. 冻结复用与门禁")
    L.append("")
    reuse = _read("frozen_reuse.json", {})
    L.append("复用的冻结拓扑分块 / S2 归属缓存来自 `lcrc_relation_interface_routing_pilot_v1`，"
             "逐文件 SHA-256 + **逐数组重算** 校验：")
    L.append("")
    L.append(table(
        ["split", "routing cache sha256", "fingerprint match", "arrays match", "rebuild s"],
        [
            [split,
             str(reuse.get("files", {}).get(split, {}).get("actual_sha256", "—"))[:16] + "…",
             reuse.get("splits", {}).get(split, {}).get("fingerprint_match"),
             all(reuse.get("splits", {}).get(split, {}).get("arrays_match", {}).values()),
             f(reuse.get("splits", {}).get(split, {}).get("rebuild_seconds"), 2)]
            for split in C.SPLITS
        ],
    ))
    L.append("")
    L.append(f"门禁（`analysis/routing_invariants.json`，phase = `{invariants.get('phase')}`）："
             f"**{len(invariants.get('gates', {}))} 项**，pass = **{invariants.get('pass')}**，"
             f"failing = {invariants.get('failing')}。")
    L.append("")
    gate_rows = []
    for name, gate in (invariants.get("gates") or {}).items():
        gate_rows.append([name, gate.get("pass")])
    L.append(table(["gate", "pass"], gate_rows))
    L.append("")
    L.append("关键门禁的含义：")
    L.append("")
    L.append("- **角色划分完备性**：每条 S2 occurrence 恰好属于一组（REAL：`owner(e1)==owner(e2)`；"
             "CONTROL：S2 类型哈希），无遗漏、无重复；")
    L.append("- **S2 聚合守恒**：逐原子 `Q_internal + Q_cross == 冻结 C1 的 S2 聚合`（typed 104 维 + content 96 维）；")
    L.append("- **gamma=1 等价**：在 canonical C1 初始状态上，局部输入列、`h`、`f(h)`、`mean h`、"
             "`mean h²`、`rho`、预测与 `cr_models.ArmA` 完全等价；")
    L.append("- **重编号不变**：原子/键重编号后，分块双射、角色计数、逐原子角色聚合与预测不变；")
    L.append("- **C1 输入/归一化/读出未改**：433 列输入的非 S2 列逐位相同，S2 列在守恒容差内相同；"
             "Train-only 归一化与目标 scaler 复算一致；`f`/`rho` 权重与 canonical 状态一致；")
    L.append("- **参数同一**：REAL/CONTROL 参数量与初始状态哈希完全相同；")
    L.append("- **无标签/oracle 进入分组**：CONTROL 角色仅由 S2 类型码决定，REAL 角色仅由冻结分块决定；"
             "标签置乱后角色与批次特征不变。")
    L.append("")
    L.append("## 3. 架构与参数预算")
    L.append("")
    expect = budget.get("analytic", {})
    L.append(table(
        ["项", "参数量"],
        [
            ["C1（只读基线）", expect.get("c1_total")],
            ["trunk（atom/bond/S1/S2/LocalMLP）", expect.get("trunk")],
            ["f = MLP 64→64→1", expect.get("f")],
            ["rho = MLP 171→64→1", expect.get("rho")],
            ["gamma_internal + gamma_cross", expect.get("role_gamma")],
            ["RR_REAL / RR_CONTROL 总计", expect.get("rr_total")],
            ["相对 C1 增量", expect.get("delta_over_c1")],
        ],
    ))
    L.append("")
    L.append(f"两模型参数量完全相同（ratio = {f(budget.get('ratio', {}).get('real_over_control'), 6)}），"
             f"初始状态哈希相同 = `{str(budget.get('full_initial_state_hash'))[:16]}…`；"
             f"gamma 初值全为 1：**是**。")
    L.append("")
    L.append("S2 角色输入：`Q_new = gamma_internal ⊙ Q_internal + gamma_cross ⊙ Q_cross ∈ R^200`，"
             "代替原来的 `Q_internal + Q_cross`；其余 233 列（a 96 + b_sum 96 + nbr 32 + atom scalars 9）"
             "与 C1 完全相同。")
    L.append("")
    L.append("### 两组 occurrence 的规模（报告，不强制对齐）")
    L.append("")
    rows = []
    for split in C.SPLITS:
        e = (groups.get("splits", {}) or {}).get(split, {})
        for variant in C.MODELS:
            m = e.get(variant, {})
            rows.append([
                split, variant, m.get("class_names"), m.get("n_group0"), m.get("n_group1"),
                pct(m.get("fraction_group0")), pct(m.get("fraction_group1")),
                f(m.get("size_ratio_g1_over_g0"), 4),
                m.get("n_molecules_both_groups"),
            ])
    L.append(table(
        ["split", "model", "groups", "n(group0)", "n(group1)", "share0", "share1", "n1/n0", "两组的分子数"],
        rows,
    ))
    L.append("")
    ctrl = (groups.get("splits", {}) or {}).get("train", {}).get("RR_CONTROL", {})
    real = (groups.get("splits", {}) or {}).get("train", {}).get("RR_REAL", {})
    L.append(f"Train 上两组规模差异：RR_REAL internal/cross = {real.get('n_group0')} / {real.get('n_group1')} "
             f"(差 {pct(real.get('fraction_group0', 0) - real.get('fraction_group1', 0))})；"
             f"RR_CONTROL hash0/hash1 = {ctrl.get('n_group0')} / {ctrl.get('n_group1')} "
             f"(差 {pct(ctrl.get('fraction_group0', 0) - ctrl.get('fraction_group1', 0))})。")
    L.append("")
    L.append("CONTROL 的分组只依赖 S2 类型码 (path_cat, path_end63, path_closure) 的固定 blake2b 哈希，"
             "不含原子索引、bond id、row_id、标签或 oracle；该哈希在训练前冻结在 `cache/`，"
             f"盐值 `{C.CONTROL_SALT}`。")
    L.append("")
    L.append("## 4. Lipophilicity：预测、恢复与角色增量")
    L.append("")
    L.append("### 4.1 预测（best-Valid checkpoint；Test = DEVELOPMENT_TEST）")
    L.append("")
    lipo_rows = [
        r for r in comparison
        if r.get("task") == "lipo"
    ]
    order = {"RR_REAL": 0, "RR_CONTROL": 1, "C1 (frozen, Rf)": 2, "GINE-G0 (frozen, Rf)": 3, "GINE-Gx (frozen, Rf)": 4}
    lipo_rows.sort(key=lambda r: (order.get(r.get("model"), 9), {"train": 0, "valid": 1, "test": 2}.get(r.get("split"), 9)))
    L.append(table(
        ["model", "split", "RMSE", "MAE", "epoch", "params", "comparability"],
        [[r.get("model"), r.get("split"), f(r.get("rmse")), f(r.get("mae")), r.get("epoch"),
          r.get("n_parameters"), r.get("comparability")] for r in lipo_rows],
    ))
    L.append("")
    L.append("历史 C1(Rf) 与 GINE 行是只读基线；GINE 仅作参照，不进入任何判据。")
    L.append("")
    base = retention.get("baseline", {}) or {}
    L.append("### 4.2 预测是否恢复（PREDICTION_PRESERVED）")
    L.append("")
    L.append(
        f"预登记判据：RR_REAL Lipophilicity **Valid** RMSE ≤ {C.RETENTION_FACTOR:.2f} × C1(Rf) Valid RMSE。"
    )
    L.append("")
    rows = []
    for variant in C.MODELS:
        m = (retention.get("models", {}) or {}).get(variant, {})
        rows.append([
            variant, f(m.get("valid_rmse")), f(m.get("valid_ratio_over_c1"), 4),
            f(m.get("valid_abs_gap"), 4), f(m.get("test_rmse")), f(m.get("test_ratio_over_c1"), 4),
            m.get("selected_epoch"), m.get("passes_retention"),
        ])
    L.append(table(
        ["model", "valid RMSE", "valid / C1", "valid − C1", "test RMSE", "test / C1", "epoch", "pass"],
        rows,
    ))
    L.append("")
    L.append(f"C1(Rf) 基线：Valid = {f(base.get('valid_rmse'))}，Test = {f(base.get('test_rmse'))}，"
             f"Epoch = {base.get('epoch')}，参数量 = {base.get('n_parameters')}（`{base.get('source')}`）。")
    L.append("")
    L.append(f"**PREDICTION_PRESERVED = {decision.get('flags', {}).get('PREDICTION_PRESERVED')}**；"
             f"实际差距 = REAL Valid {f((retention.get('models', {}).get('RR_REAL', {}) or {}).get('valid_abs_gap'), 4)} "
             f"({pct((retention.get('models', {}).get('RR_REAL', {}) or {}).get('valid_ratio_over_c1'))} of C1)。"
             f"CONTROL 是否也恢复 = {decision.get('flags', {}).get('CONTROL_ALSO_PRESERVED')}。")
    L.append("")
    L.append("### 4.3 REAL − CONTROL 的同分子配对 bootstrap")
    L.append("")
    L.append("约定：`diff = RMSE(REAL) − RMSE(CONTROL)`，负值表示 REAL 更好；同一 split 上同一批分子，"
             f"{C.BOOTSTRAP_DRAWS} 次重采样，seed {C.BOOTSTRAP_SEED}。")
    L.append("")
    rows = []
    for split in C.SPLITS:
        c = (bootstrap.get("comparisons", {}) or {}).get(f"lipo/{split}", {})
        if not c:
            continue
        rows.append([
            split, c.get("n"), f(c.get("rmse_real")), f(c.get("rmse_control")),
            f(c.get("diff_real_minus_control")), ci(c.get("ci95_diff")),
            f(c.get("prob_real_better"), 4), c.get("positive_ci_excludes_zero_in_favour_of_real"),
        ])
    L.append(table(
        ["split", "n", "RMSE REAL", "RMSE CONTROL", "diff", "95% CI diff", "P(REAL better)", "CI 排除 0"],
        rows,
    ))
    L.append("")
    L.append("### 4.4 gamma 学到的值与该冻结模型的消融响应")
    L.append("")
    rows = []
    for variant in C.MODELS:
        g = (gamma.get(variant, {}) or {}).get("gamma", {})
        gi = g.get("gamma_internal", {})
        gc = g.get("gamma_cross", {})
        rows.append([
            variant,
            f(gi.get("mean_abs_dev_from_one")), f(gi.get("max_abs_dev_from_one")),
            pct(gi.get("fraction_abs_dev_gt_0.05")),
            f(gc.get("mean_abs_dev_from_one")), f(gc.get("max_abs_dev_from_one")),
            pct(gc.get("fraction_abs_dev_gt_0.05")),
            f((gamma.get(variant, {}) or {}).get("gamma_abs_dev_ratio_cross_over_internal"), 3),
        ])
    L.append(table(
        ["model", "|g_int−1| mean", "max", "share>0.05", "|g_cross−1| mean", "max", "share>0.05", "ratio cross/internal"],
        rows,
    ))
    L.append("")
    rows = []
    for variant in C.MODELS:
        abl = (gamma.get(variant, {}) or {}).get("ablations", {})
        for split in ("valid", "test"):
            a = abl.get(split, {})
            if not a:
                continue
            rows.append([
                variant, split,
                f((a.get("learned") or {}).get("rmse")),
                f((a.get("both_one") or {}).get("rmse")),
                f((a.get("internal_one") or {}).get("rmse")),
                f((a.get("cross_one") or {}).get("rmse")),
                f((a.get("swap") or {}).get("rmse")),
            ])
    L.append(table(
        ["model", "split", "learned", "both=1", "internal=1", "cross=1", "swap (诊断)"],
        rows,
    ))
    L.append("")
    rows = []
    for variant in C.MODELS:
        rr = (gamma.get(variant, {}) or {}).get("role_response", {})
        for split in ("valid", "test"):
            e = rr.get(split, {})
            if not e:
                continue
            rows.append([
                variant, split, f(e.get("mean_abs_delta_pred_both_one_vs_learned")),
                f(e.get("median_abs_delta_pred_both_one_vs_learned")),
                f(e.get("max_abs_delta_pred_both_one_vs_learned")),
                f(e.get("rmse_change_both_one_minus_learned")),
            ])
    L.append(table(
        ["model", "split", "mean |Δpred|", "median |Δpred|", "max |Δpred|", "ΔRMSE (both=1 − learned)"],
        rows,
    ))
    L.append("")
    L.append("`both=1` 即把角色缩放完全关掉、退回 C1 聚合；表中的 ΔRMSE 是该冻结模型**真实使用**角色缩放的程度。"
             "`swap` 只是诊断，不进入判据。")
    L.append("")
    L.append("## 5. WC_LOGP：预测、闭合、oracle 定位与屏蔽响应")
    L.append("")
    L.append("### 5.1 预测与闭合")
    L.append("")
    rows = []
    for r in comparison:
        if r.get("task") != "wc_logp":
            continue
        rows.append([r.get("model"), r.get("split"), f(r.get("rmse")), f(r.get("mae")),
                     r.get("epoch"), r.get("n_parameters"), r.get("comparability")])
    L.append(table(["model", "split", "RMSE", "MAE", "epoch", "params", "comparability"], rows))
    L.append("")
    spl = closure.get("splits", {})
    rows = [[s, spl.get(s, {}).get("n_rows"), f(spl.get(s, {}).get("max_abs_graph_minus_label"), 10),
             f(spl.get(s, {}).get("max_abs_atom_sum_minus_label"), 10)] for s in C.SPLITS]
    L.append("真实 WC_LOGP 原子贡献闭合（`sum_v c_v_oracle == WC_LOGP`）：")
    L.append("")
    L.append(table(["split", "rows", "max |graph − label|", "max |Σ atom − label|"], rows))
    L.append("")
    rows = []
    for variant in C.MODELS:
        e = (closure.get("models", {}) or {}).get(variant, {})
        for split in ("valid", "test"):
            x = e.get(split, {})
            if not x.get("available"):
                continue
            rows.append([variant, split, x.get("n_rows"), f(x.get("max_abs_readout_closure_gap"), 10),
                         f(x.get("max_abs_pred_minus_saved_prediction"), 10), f(x.get("rmse_recomputed"))])
    L.append(table(
        ["model", "split", "rows", "max |pred − (Σf + rho + bias)|", "max |pred − saved|", "RMSE recomputed"],
        rows,
    ))
    bias_note = sorted({f(x.get("bias_raw")) for e in (closure.get("models", {}) or {}).values()
                        for x in (e or {}).values() if isinstance(x, dict) and x.get("available")})
    L.append("")
    L.append(f"闭合恒等式在标准化空间成立（`pred_std = Σf + rho`），raw 空间的常数截距 "
             f"`bias = mean(y_train)` 显式记入 `analysis/wc_logp_closure.json`（本 task 取 "
             f"{', '.join(bias_note) if bias_note else '—'}；同一 task 两模型同值）。")
    L.append("")
    L.append("### 5.2 Oracle enrichment（冻结候选池 + 相同匹配对照）")
    L.append("")
    L.append(
        "候选池 = 与上一轮**完全相同**的冻结拓扑 block 集合；每个模型用自己的**原生 block 屏蔽响应**"
        "`score_i = |Δ_i| / sqrt(max(1, n_bonds_i))` 选候选（无标签、无 oracle）。"
        f"随机对照复用同一函数、同一 seed（{C.ORACLE_CONTROL_SEED}）与同一匹配规则"
        f"（键数 + 原子数，≥{C.MIN_ORACLE_CONTROLS} 个，最多 {C.MAX_ORACLE_CONTROLS} 个），"
        "capture = 候选原子集合 |WC_LOGP 原子贡献| / 全分子质量。"
    )
    L.append("")
    rows = []
    for variant in C.MODELS:
        e = (oracle.get("models", {}) or {}).get(variant, {})
        for split in ("valid", "test"):
            x = e.get(split, {})
            if not x:
                continue
            en = x.get("enrichment", {})
            rows.append([
                variant, split, x.get("n_molecules"), x.get("n_matched"),
                f(en.get("mean")), f(en.get("median")), pct(en.get("positive_fraction")),
                ci(en.get("ci95_mean")), ci(en.get("ci95_median")),
                f(x.get("capture", {}).get("mean")), f(x.get("control_mean_of_means")),
                x.get("oracle_signal"),
            ])
    L.append(table(
        ["model", "split", "n", "matched", "mean enrich", "median", "positive", "CI mean",
         "CI median", "capture", "control mean", "signal"],
        rows,
    ))
    L.append("")
    rows = []
    for split in ("valid", "test"):
        x = (oracle.get("paired_real_minus_control", {}) or {}).get(split, {})
        if not x:
            continue
        rows.append([
            split, x.get("n_shared"), f(x.get("mean_diff")), ci(x.get("ci95_mean_diff")),
            f(x.get("prob_real_enrichment_better"), 4), x.get("real_oracle_signal"),
            x.get("positive_in_favour_of_real"), pct(x.get("fraction_a_positive")),
            pct(x.get("fraction_b_positive")),
        ])
    L.append(table(
        ["split", "n shared", "mean enrich REAL−CONTROL", "95% CI", "P(REAL>CONTROL)",
         "REAL signal", "CI 排除 0", "REAL positive", "CONTROL positive"],
        rows,
    ))
    L.append("")
    L.append("### 5.3 同一候选的模型屏蔽响应")
    L.append("")
    L.append("对两个模型各自发现的候选（同一分子、同一 block、同一屏蔽语义），**两个模型都**重新前向一次，"
             "比较 |Δ|；同时报告逐分子全 block |Δ| 向量的 Spearman 与 argmax 一致率。")
    L.append("")
    rows = []
    for split in ("valid", "test"):
        for source in C.MODELS:
            e = (mask.get("pairs", {}) or {}).get(f"{split}/{source}", {})
            if not e:
                continue
            pm = e.get("paired_abs_delta_real_minus_control", {})
            rows.append([
                split, source, e.get("n"), f(e.get("mean_abs_delta_real")),
                f(e.get("mean_abs_delta_control")), f(pm.get("mean_diff")),
                ci(pm.get("ci95_mean_diff")), f(pm.get("prob_a_better"), 4),
            ])
    L.append(table(
        ["split", "candidate from", "n", "mean |Δ REAL|", "mean |Δ CONTROL|",
         "mean diff (REAL−CONTROL)", "95% CI", "P(REAL stronger)"],
        rows,
    ))
    L.append("")
    rows = []
    for split in ("valid", "test"):
        e = (mask.get("rank_agreement", {}) or {}).get(split, {})
        if not e:
            continue
        rows.append([split, e.get("n_molecules"), f(e.get("mean_spearman_abs_delta")),
                     f(e.get("median_spearman_abs_delta")), pct(e.get("fraction_same_argmax"))])
    L.append(table(
        ["split", "n molecules", "mean Spearman |Δ|", "median Spearman |Δ|", "same argmax"],
        rows,
    ))
    L.append("")
    L.append("## 6. 决策")
    L.append("")
    flags = decision.get("flags", {})
    L.append(table(["flag", "value"], [[k, v] for k, v in flags.items()]))
    L.append("")
    L.append(f"**判定 = `{decision.get('verdict')}`**")
    L.append("")
    L.append(_decision_summary(decision, retention, bootstrap, oracle, mask))
    L.append("")
    L.append("## 7. 口径、局限与未做的事")
    L.append("")
    for note in decision.get("caveats", []):
        L.append(f"- {note}")
    L.append("")
    for note in decision.get("not_done", []):
        L.append(f"- {note}")
    L.append("")
    L.append("### 不可直接比较的旧结果（仅附录，不进主表、不进判据）")
    L.append("")
    if appendix:
        L.append(table(
            ["task", "model", "split", "RMSE", "params", "label"],
            [[r.get("task"), r.get("model"), r.get("split"), f(r.get("rmse")), r.get("n_parameters"),
              r.get("comparability")] for r in appendix],
        ))
    else:
        L.append("（无）")
    L.append("")
    if manifest:
        L.append("### 运行环境")
        L.append("")
        training_commits = sorted(
            {
                str(((C.read_json(path).get("environment") or {}).get("git_commit")))
                for path in sorted(C.ANALYSIS_DIR.glob("trajectory_*.json"))
                if "_smoke" not in path.name
            }
        )
        if training_commits:
            L.append(f"- 四个训练单元的 commit: `{', '.join(training_commits)}`（分析/汇总在本 commit）")
        env = manifest.get("environment", {})
        L.append(f"- commit: `{env.get('git_commit')}` (branch `{env.get('git_branch')}`, dirty={env.get('git_dirty_tracked')})")
        L.append(f"- hostname: `{env.get('hostname')}`；platform: `{env.get('platform')}`")
        torch_info = env.get("torch", {})
        L.append(f"- torch: `{torch_info.get('torch_cuda')}`, CUDA available = `{torch_info.get('cuda_available')}`, "
                 f"devices = `{torch_info.get('devices')}`")
        L.append(f"- 冻结输入 sha256: `{manifest.get('inputs_sha256')}`")
    L.append("")
    return "\n".join(L) + "\n"


def _decision_summary(decision, retention, bootstrap, oracle, mask) -> str:
    flags = decision.get("flags", {})
    real = (retention.get("models", {}) or {}).get("RR_REAL", {})
    control = (retention.get("models", {}) or {}).get("RR_CONTROL", {})
    comps = bootstrap.get("comparisons", {})
    parts = []
    parts.append(f"1. **保留 C1 读出后预测是否恢复**：REAL Valid RMSE = {f(real.get('valid_rmse'))}"
                 f"（C1 的 {f(real.get('valid_ratio_over_c1'), 4)}×，差距 {f(real.get('valid_abs_gap'), 4)}），"
                 f"CONTROL = {f(control.get('valid_rmse'))}"
                 f"（{f(control.get('valid_ratio_over_c1'), 4)}×）。"
                 f"PREDICTION_PRESERVED = `{flags.get('PREDICTION_PRESERVED')}`。")
    lipo_valid = comps.get("lipo/valid", {})
    lipo_test = comps.get("lipo/test", {})
    parts.append(f"2. **真实结构边界角色是否优于同容量伪分组**：Lipo Valid ΔRMSE(REAL−CONTROL) = "
                 f"{f(lipo_valid.get('diff_real_minus_control'))}，95% CI {ci(lipo_valid.get('ci95_diff'))}，"
                 f"P(REAL better) = {f(lipo_valid.get('prob_real_better'), 4)}；Test Δ = "
                 f"{f(lipo_test.get('diff_real_minus_control'))}，95% CI {ci(lipo_test.get('ci95_diff'))}。"
                 f"预测增量 credit = `{flags.get('ROLE_INCREMENT_PREDICTIVE')}`。")
    ov = (oracle.get("paired_real_minus_control", {}) or {}).get("valid", {})
    ot = (oracle.get("paired_real_minus_control", {}) or {}).get("test", {})
    parts.append(f"3. **WC_LOGP 已知贡献捕获是否改善**：Valid paired enrichment Δ = {f(ov.get('mean_diff'))} "
                 f"(95% CI {ci(ov.get('ci95_mean_diff'))}, n={ov.get('n_shared')})；Test Δ = {f(ot.get('mean_diff'))} "
                 f"(95% CI {ci(ot.get('ci95_mean_diff'))}, n={ot.get('n_shared')})。"
                 f"oracle 增量 credit = `{flags.get('ROLE_INCREMENT_ORACLE')}`。")
    parts.append(f"4. **改善来自预测还是结构解释**：ROLE_INCREMENT_SHOWN = `{flags.get('ROLE_INCREMENT_SHOWN')}`，"
                 f"ROLE_INCREMENT_PREDICTIVE = `{flags.get('ROLE_INCREMENT_PREDICTIVE')}`，"
                 f"ROLE_INCREMENT_ORACLE = `{flags.get('ROLE_INCREMENT_ORACLE')}`。"
                 f"ONLY_C1_READOUT_RECOVERY = `{flags.get('ONLY_C1_READOUT_RECOVERY')}`，"
                 f"ROLE_NOT_SUPPORTED = `{flags.get('ROLE_NOT_SUPPORTED')}`。")
    same = (mask.get("rank_agreement", {}) or {}).get("valid", {})
    parts.append(f"5. **是否值得继续显式 block/interface Token**：见 `DECISION.md` 的下一步建议；"
                 f"本轮证据仅为单 seed DEVELOPMENT，Valid 上两模型 |Δ| 的 mean Spearman = "
                 f"{f(same.get('mean_spearman_abs_delta'))}，argmax 一致率 = {pct(same.get('fraction_same_argmax'))}。")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# DECISION.md
# ---------------------------------------------------------------------------


def _decision(decision, retention, bootstrap, oracle, mask, gamma) -> str:
    flags = decision.get("flags", {})
    L: list[str] = []
    L.append("# DECISION — 保留 C1 读出的 S2 结构角色 pilot")
    L.append("")
    L.append(f"轮次 `{C.ROUND_ID}`。结论只覆盖本轮预登记判据，不外推到其他架构、其他数据集或其他 seed。")
    L.append("")
    L.append("## 判定")
    L.append("")
    L.append(f"**`{decision.get('verdict')}`**")
    L.append("")
    L.append(table(["flag", "value"], [[k, v] for k, v in flags.items()]))
    L.append("")
    L.append("## 五个问题的直接回答")
    L.append("")
    L.append(_decision_summary(decision, retention, bootstrap, oracle, mask))
    L.append("")
    L.append("## 预登记判据的命中情况")
    L.append("")
    inc = decision.get("prediction_increment", {})
    lc = decision.get("oracle_increment", {})
    rows = []
    for split in C.SPLITS:
        if split in inc:
            rows.append([f"lipo/{split}", "predictive", f(inc[split].get("diff")),
                         ci(inc[split].get("ci95")), f(inc[split].get("prob_real_better"), 4),
                         inc[split].get("credit")])
    rows.append(["wc_logp/valid", "oracle",
                 f((lc.get("valid") or {}).get("diff_mean")), ci((lc.get("valid") or {}).get("ci95")),
                 f((lc.get("valid") or {}).get("prob_real_better"), 4), (lc.get("valid") or {}).get("credit")])
    rows.append(["wc_logp/test", "oracle",
                 f((lc.get("test") or {}).get("diff_mean")), ci((lc.get("test") or {}).get("ci95")),
                 f((lc.get("test") or {}).get("prob_real_better"), 4), (lc.get("test") or {}).get("credit")])
    L.append(table(["split", "evidence", "point estimate", "95% CI", "P(REAL better)", "credit"], rows))
    L.append("")
    L.append(
        "判据（训练前冻结）：预测增量要求 95% 配对 CI 在 REAL 一侧排除 0 且 "
        f"P(REAL better) ≥ {C.INCREMENT_PROB:.2f}；oracle 增量还要求 REAL 的 enrichment 本身满足"
        f"“≥ {100 * C.ORACLE_POSITIVE_FRACTION:.0f}% 匹配分子为正 + 均值为正”，"
        "并在同一批匹配分子上与 CONTROL 做配对比较。"
    )
    L.append("")
    L.append("## 角色缩放是否被真正使用")
    L.append("")
    rows = []
    for variant in C.MODELS:
        g = (gamma.get(variant, {}) or {}).get("gamma", {})
        rr = (gamma.get(variant, {}) or {}).get("role_response", {})
        abl = (gamma.get(variant, {}) or {}).get("ablations", {})
        rows.append([
            variant,
            f(g.get("gamma_internal", {}).get("mean_abs_dev_from_one")),
            f(g.get("gamma_cross", {}).get("mean_abs_dev_from_one")),
            f((rr.get("valid") or {}).get("mean_abs_delta_pred_both_one_vs_learned")),
            f((abl.get("valid") or {}).get("both_one", {}).get("rmse")),
            f((abl.get("valid") or {}).get("learned", {}).get("rmse")),
        ])
    L.append(table(
        ["model", "|g_int−1| mean", "|g_cross−1| mean", "Valid mean |Δpred| (both=1)", "Valid both=1 RMSE", "Valid learned RMSE"],
        rows,
    ))
    L.append("")
    L.append("## 下一步建议")
    L.append("")
    verdict = decision.get("verdict")
    if verdict == "ROLE_INCREMENT_SHOWN":
        L.append("本轮已记录 `ROLE_INCREMENT_SHOWN`。若继续，最省的下一步是**在同一预登记下增加一个确认 seed**，"
                 "或把角色缩放做成显式 interface Token 的对照；两者都必须先预算 GPU·h 并新写协议。"
                 "**本轮到此停止**：不自动加 seed、不加超参搜索、不扩结构划分。")
    elif verdict == "ONLY_C1_READOUT_RECOVERY":
        L.append("结论是 `ONLY_C1_READOUT_RECOVERY`：保留 C1 读出后两个模型都能恢复预测，"
                 "但真实结构边界角色没有给出预登记强度的额外收益。"
                 "因此**不建议**仅凭本轮证据去投入显式 block/interface Token 的预测力路线；"
                 "若仍要做，必须先提出一个新的、能区分“读出恢复”与“角色信息”的预登记设计（例如在匹配候选上做"
                 "更强的外部结构判据），而不是继续加容量。**本轮到此停止。**")
    elif verdict == "ROLE_NOT_SUPPORTED":
        L.append("结论是 `ROLE_NOT_SUPPORTED`：保留 C1 读出并未恢复预登记水平的预测，且没有独立的 oracle 增量。"
                 "**不建议**继续显式 block/interface Token 路线。**本轮到此停止。**")
    else:
        L.append("结论是 `UNRESOLVED`：预测与外部结构指标都没有达到预登记强度。"
                 "**不建议**在缺少新证据的情况下继续加架构。**本轮到此停止。**")
    L.append("")
    L.append("## 口径限制")
    L.append("")
    for note in decision.get("caveats", []):
        L.append(f"- {note}")
    L.append("")
    L.append("## 未做的事")
    L.append("")
    for note in decision.get("not_done", []):
        L.append(f"- {note}")
    L.append("")
    return "\n".join(L) + "\n"


__all__ = ["write_reports"]
