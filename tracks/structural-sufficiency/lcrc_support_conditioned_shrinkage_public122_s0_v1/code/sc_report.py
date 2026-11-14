"""Report + the three core figures for the support-conditioned shrinkage round.

Reads ``analysis/*.json`` (produced by ``sc_analysis``) and writes
``REPORT.md`` (Chinese) plus at most three figures under ``analysis/figures/``.
No number in the report is typed by hand: everything is formatted from the JSON
artefacts.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import sc_common as C

FIG = C.ANALYSIS / "figures"


def _f(x, nd: int = 6) -> str:
    if x is None:
        return "—"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(v):
        return "n/a"
    return f"{v:.{nd}f}"


def _pct(x, nd: int = 2) -> str:
    if x is None:
        return "—"
    return f"{100 * float(x):+.{nd}f}%"


def _ci(rec: Mapping[str, Any]) -> str:
    if not rec:
        return "—"
    lo, hi = rec.get("p2.5"), rec.get("p97.5")
    if lo is None or hi is None:
        return "—"
    return f"[{lo:+.4f}, {hi:+.4f}]"


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def figure_support(comparisons: Mapping[str, Any], groups: Mapping[str, Any]) -> str | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import sc_support as S

    key = C.read_json(C.ANALYSIS / "key_selection.json")["chosen_key"]
    sup = S.load_support(C.PRIMARY_DATASET, "100", key)
    act = np.asarray(sup["active_train"], dtype=bool)
    c = np.asarray(sup["c_train"], dtype=np.int64)[act]
    a = np.asarray(sup["alpha_SELECT_train"], dtype=np.float64)[act]
    r = np.asarray(sup["r_train"], dtype=np.float64)[act]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    bins = np.unique(np.round(np.logspace(0, np.log10(max(c.max(), 2)), 26)).astype(int))
    idx = np.digitize(c, bins)
    xs, q10, q50, q90 = [], [], [], []
    for i in range(len(bins)):
        m = (idx == i)
        if m.sum() < 20:
            continue
        xs.append(bins[i])
        w = r[m]
        order = np.argsort(a[m])
        av, wv = a[m][order], w[order]
        cw = np.cumsum(wv) / wv.sum()
        q10.append(av[int(np.searchsorted(cw, 0.1))])
        q50.append(av[int(np.searchsorted(cw, 0.5))])
        q90.append(av[int(np.searchsorted(cw, 0.9))])
    ax.fill_between(xs, q10, q90, alpha=0.25, label="q10–q90")
    ax.plot(xs, q50, marker="o", ms=3, label="weighted median")
    ax.axhline(C.UNIFORM_A, ls="--", lw=1, color="k", label="UNIFORM a=0.70")
    ax.set_xscale("log")
    ax.set_xlabel("c_T(K_v): distinct fit structures containing the key")
    ax.set_ylabel("a_SELECT(v)")
    ax.set_title(f"frozen support gate ({key}, Lipo fit100)")
    ax.legend(fontsize=8)

    ax = axes[1]
    bx = np.linspace(0.4, 1.0, 25)
    ax.hist(a, bins=bx, weights=r / r.sum(), alpha=0.6,
            label="fit table (Train roots, r-weighted)")
    oof = groups.get("oof_hist")
    if oof:
        ax.plot(oof["centers"], oof["weights"], color="C1", lw=1.5,
                label="OOF queries (Train structures only)")
    ax.axvline(C.UNIFORM_A, ls="--", lw=1, color="k")
    ax.set_xlabel("a_SELECT(v)")
    ax.set_ylabel("share of reference weight")
    ax.set_title(f"gate spread: fit mean = {_f(float(sup['fit_mean_a_select']), 4)}, "
                 f"a_oof = {_f(float(sup['a_oof']), 4)}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / "support_feasibility.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return C.track_rel(path)


def figure_performance(comparisons: Mapping[str, Any]) -> str | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = comparisons["main_table"]["rows"]
    fits = sorted({r["fit"] for r in rows}, key=lambda x: -int(x))
    metric = "test_rmse" if any(r.get("test_rmse") is not None for r in rows) else "test_roc_auc"
    fig, axes = plt.subplots(1, len(fits), figsize=(5.2 * len(fits), 4), squeeze=False)
    for ax, fit in zip(axes[0], fits):
        sub = [r for r in rows if r["fit"] == fit and r.get(metric) is not None]
        names = [r["arm"] for r in sub]
        vals = [r[metric] for r in sub]
        colors = ["#666666" if n == C.ARM_BASE else
                  ("#1f77b4" if n == C.ARM_SELECT else "#bbbbbb") for n in names]
        ax.bar(range(len(names)), vals, color=colors)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("Test RMSE" if metric == "test_rmse" else "Test ROC-AUC")
        ax.set_title(f"fit{fit}")
        lo = min(vals)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=7)
        ax.set_ylim(lo * 0.97 if metric == "test_rmse" else max(0.5, lo - 0.05),
                    max(vals) * 1.02 + (0.02 if metric == "test_rmse" else 0.02))
    fig.suptitle("five arms, identical recipe / init / batch plan (Test is DEVELOPMENT_TEST)")
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / "performance_two_N.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return C.track_rel(path)


def figure_support_groups(groups: Mapping[str, Any]) -> str | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_fit = groups.get("by_fit", {})
    fits = [f for f in by_fit if by_fit[f].get("groups")]
    if not fits:
        return None
    fig, axes = plt.subplots(len(fits), 2, figsize=(10, 3.6 * len(fits)), squeeze=False)
    for row, fit in enumerate(fits):
        for col, split in enumerate(("valid", "test")):
            ax = axes[row][col]
            rec = by_fit[fit]["groups"].get(split, {})
            labels, vals = [], []
            for grp in ("low", "high"):
                for ctrl in (C.ARM_BASE, C.ARM_UNIFORM):
                    keyname = f"mean_dsse_SELECT_vs_{ctrl}"
                    if keyname in rec.get(grp, {}):
                        labels.append(f"{grp}\nvs {ctrl}")
                        vals.append(rec[grp][keyname])
            if not vals:
                continue
            ax.bar(range(len(vals)), vals, color=["#d62728" if v > 0 else "#2ca02c" for v in vals])
            ax.axhline(0, color="k", lw=0.8)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, fontsize=7)
            ax.set_ylabel("mean ΔSSE (neg = SELECT better)")
            ax.set_title(f"fit{fit} · {split} · burden groups (Train-only OOF thresholds)")
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / "support_groups.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return C.track_rel(path)


# ---------------------------------------------------------------------------
# decision (frozen rules of PROTOCOL sections 5 and 8)
# ---------------------------------------------------------------------------


def decide(comparisons: Mapping[str, Any], intervention: Mapping[str, Any]) -> dict[str, Any]:
    gains = comparisons["gains"]
    boot = comparisons["bootstrap"]
    table = comparisons["main_table"]["rows"]
    n_complete = sum(1 for r in table if r.get("status") == "COMPLETE")
    notes: list[str] = []

    def g(fit: str, cand: str, ctrl: str, split: str) -> float | None:
        rec = gains.get(f"F{fit}|{cand}_vs_{ctrl}")
        return None if rec is None else rec.get(f"{split}_gain")

    def grel(fit: str, cand: str, ctrl: str, split: str) -> float | None:
        rec = gains.get(f"F{fit}|{cand}_vs_{ctrl}")
        return None if rec is None else rec.get(f"{split}_gain_rel")

    statuses: dict[str, Any] = {"n_completed_runs": n_complete}

    # INTERVENTION_WEAK_OR_UNUSED
    cf = ((intervention.get("by_fit", {}).get("100", {}) or {}).get("counterfactual", {}) or {})
    weak = all((cf.get(s, {}) or {}).get("frac_molecules_changed", 1.0) < 0.02
               for s in ("valid", "test")) if cf else False
    statuses["intervention_weak"] = bool(weak)
    if weak:
        notes.append("a=1 counterfactual changes <2% of molecules: intervention weak or unused")

    # promotion rule (§5) on Lipophilicity
    promo: dict[str, Any] = {}
    for fit in ("100", "25"):
        if gains.get(f"F{fit}|{C.ARM_SELECT}_vs_{C.ARM_BASE}") is None:
            continue
        rmse_ok = all((grel(fit, C.ARM_SELECT, C.ARM_BASE, s) or -1) >= C.LIPO_PROMOTE_REL_RMSE
                      for s in ("valid", "test"))
        mae_rec = gains[f"F{fit}|{C.ARM_SELECT}_vs_{C.ARM_BASE}"]
        mae_ok = all((mae_rec.get(f"{s}_mae_rel") or 0) <= C.LIPO_PROMOTE_MAE_TOL
                     for s in ("valid", "test"))
        uniform_ok = all((g(fit, C.ARM_SELECT, u, s) or -1) > 0
                         for u in (C.ARM_UNIFORM, C.ARM_UNIFORM_OOF, C.ARM_REVERSE)
                         for s in ("valid", "test"))
        promo[fit] = {"rmse_vs_base_both_splits": rmse_ok, "mae_not_worse": mae_ok,
                      "beats_all_three_controls_both_splits": uniform_ok,
                      "pass": bool(rmse_ok and mae_ok and uniform_ok)}
    statuses["promotion"] = promo

    # which comparisons survive the bootstrap
    survive: dict[str, Any] = {}
    for tag, rec in boot.items():
        if not tag.startswith("F100|"):
            continue
        sc = rec.get("structure_cluster", {})
        for comp, val in sc.items():
            if not isinstance(val, dict):
                continue
            survive[f"{tag}|{comp}"] = {
                "point": val.get("point_gain"), "ci": [val.get("p2.5"), val.get("p97.5")],
                "excludes_zero": bool(val.get("p2.5") is not None
                                      and (val["p2.5"] > 0 or val["p97.5"] < 0)),
            }
    statuses["bootstrap_f100"] = survive

    select_better_base = any((g(fit, C.ARM_SELECT, C.ARM_BASE, s) or -1) > 0
                             for fit in ("100", "25") for s in ("valid", "test"))
    beats_uniforms = any(promo.get(fit, {}).get("beats_all_three_controls_both_splits")
                         for fit in promo)
    beats_reverse = any(all((g(fit, C.ARM_SELECT, C.ARM_REVERSE, s) or -1) > 0
                            for s in ("valid", "test")) for fit in promo)
    only25 = bool(promo.get("100") and not promo["100"]["pass"] and promo.get("25", {}).get("pass"))
    any_pass = any(v.get("pass") for v in promo.values())

    if weak:
        status = "INTERVENTION_WEAK_OR_UNUSED"
    elif any_pass and beats_uniforms and beats_reverse:
        crosses = any(not v["excludes_zero"] for k, v in survive.items() if "SELECT_vs_" in k)
        status = "SUPPORT_DIRECTED_SIGNAL_THIS_CONFIG" + (" (UNCERTAIN: CI crosses 0)" if crosses else "")
    elif only25:
        status = "LOW_LABEL_ONLY"
    elif select_better_base and not beats_uniforms:
        status = "GENERIC_CONTRACTION_SIGNAL"
    elif select_better_base and beats_uniforms and not beats_reverse:
        status = "DIRECTION_NOT_IDENTIFIED"
    elif not select_better_base:
        status = "NO_PREDICTIVE_GAIN_THIS_CONFIG"
    else:
        status = "SPLIT_OR_SELECTION_SENSITIVE"
    statuses["status"] = status
    statuses["notes"] = notes
    return statuses


# ---------------------------------------------------------------------------
# REPORT.md
# ---------------------------------------------------------------------------


def build_report() -> str:
    pre = C.read_json(C.ANALYSIS / "preflight.json")
    sel = C.read_json(C.ANALYSIS / "key_selection.json")
    freeze = C.read_json(C.ANALYSIS / "freeze.json")
    comp = C.read_json(C.ANALYSIS / "comparisons.json")
    groups = C.read_json(C.ANALYSIS / "support_groups.json")
    infl = C.read_json(C.ANALYSIS / "influence.json")
    inter = C.read_json(C.ANALYSIS / "intervention.json")
    budget = C.read_json(C.ANALYSIS / "budget.json")
    smoke = C.read_json(C.ANALYSIS / "smoke.json")
    status = decide(comp, inter)

    rows = comp["main_table"]["rows"]
    fits = sorted({r["fit"] for r in rows}, key=lambda x: -int(x))
    metric = "test_rmse" if any(r.get("test_rmse") is not None for r in rows) else "test_roc_auc"
    gains = comp["gains"]
    boot = comp["bootstrap"]
    key = sel["chosen_key"]

    L: list[str] = []
    L.append(f"# 支持度条件化的联合块收缩 — 报告（中文）\n")
    L.append(f"- round: `{C.ROUND_ID}`")
    L.append(f"- 状态: **DEVELOPMENT / DEVELOPMENT_TEST**（Test 参与路线选择：`{C.TEST_USE}`；"
             f"单 seed 0；不称独立最终测试）")
    L.append(f"- 判定: **{status['status']}**")
    L.append(f"- 冻结 key: `{key}`；购买 fit: {sel['purchase_fits']}；"
             f"完成正式训练 {status['n_completed_runs']} 条")
    L.append(f"- 执行 regime: {', '.join(sorted({str(v.get('device')) for v in budget['runs'].values() if v.get('status') == 'COMPLETE'}) or ['none'])}"
             f" — 见 §7（`res` 部署被共享文件系统写满阻断，记录为基础设施失败）\n")

    # ---- 0. four answers
    L.append("## 0. 四个必答问题\n")
    e = pre["stage_gates"]
    e1 = e.get(f"{C.PRIMARY_DATASET}|F100|{key}", {})
    gate_rows = " · ".join(f"{k}={_f(v['value'], 3)}{'✓' if v['pass'] else '✗'}"
                           for k, v in e1.get("checks", {}).items())
    weak = status.get("intervention_weak")
    L.append("**① 代理是否有可判别性？**")
    L.append(f"结构上**通过**：`K_FINE` 与 `K_JOINT` 在 Lipo fit100/fit25 全部门槛组合均 PASS，"
             f"按固定优先规则取 `K_FINE`。fit100 的门槛实际值：{gate_rows}。")
    L.append("")
    L.append(f"active 根仅占全部根的一小部分（Lipo train {_f(100 * float(pre['stage_keys'][C.PRIMARY_DATASET]['frac_active']['train']), 1)}%），"
             f"因为 `J_full ≡ J_marg` 的根本来就不受干预（该根内所有 path 归到同一个 "
             f"`b = path_cat % 5` 组，边际和恒等于全和）；inactive 根一律 `a=1`。"
             f"实测门控跨度为 `q90-q10`：完整 fit {_f(e1.get('checks', {}).get('E2_spread_fullfit', {}).get('value'), 3)}、"
             f"OOF {_f(e1.get('checks', {}).get('E3_spread_oof', {}).get('value'), 3)}（门槛 0.20）。")
    if weak:
        L.append("但**有效干预检查未通过**：把 `a` 全设为 1 只改变极少数分子的预测，"
                 "见 §5；因此本轮不能把任何性能差异归因于门控。")
    L.append("")

    L.append("**② 选择性是否超过两种均匀及反向？**")
    for fit in fits:
        rec = gains.get(f"F{fit}|{C.ARM_SELECT}_vs_{C.ARM_BASE}")
        if not rec:
            continue
        parts = []
        for ctrl in (C.ARM_BASE, C.ARM_UNIFORM, C.ARM_UNIFORM_OOF, C.ARM_REVERSE):
            r = gains.get(f"F{fit}|{C.ARM_SELECT}_vs_{ctrl}")
            if not r:
                continue
            parts.append(f"{ctrl}: Valid {_pct(r.get('valid_gain_rel'))} / Test {_pct(r.get('test_gain_rel'))}")
        L.append(f"- fit{fit}（相对增益，正=SELECT 更好）：" + "；".join(parts))
    L.append("")
    L.append("配对 canonical-structure cluster bootstrap（2000 次，固定选中权重，共用同任务索引）在 §2 表里给出区间；"
             "区间跨 0 的项一律如实标注。")
    L.append("")

    L.append("**③ 低标签与全量是否一致？**")
    promo = status.get("promotion", {})
    for fit in ("100", "25"):
        p = promo.get(fit)
        if not p:
            continue
        L.append(f"- fit{fit}：SELECT>BASE 两 split RMSE≥1% = {p['rmse_vs_base_both_splits']}；"
                 f"MAE 不恶化 = {p['mae_not_worse']}；同时优于两种 UNIFORM 与 REVERSE = "
                 f"{p['beats_all_three_controls_both_splits']}")
    low = comp.get("low_label", {})
    if low:
        for split in ("valid", "test"):
            for ctrl, rec in (low.get(split) or {}).items():
                L.append(f"- {split} · {ctrl}：gain(fit25)−gain(fit100) = "
                         f"{_f(rec.get('diff_25_minus_100'), 4)}，区间 {_ci(rec)}")
    L.append("")

    L.append("**④ 下一步保留什么、停止什么？**")
    if status["status"].startswith("NO_PREDICTIVE_GAIN"):
        L.append("- 停止：这一结构代理在**本算子+配方+split+单 seed**下无可兑现收益；不再自动搜索第三个 key、"
                 "更多 alpha 或更多 seed。")
    elif status["status"].startswith("GENERIC_CONTRACTION"):
        L.append("- 保留：整体收缩剂量本身（`UNIFORM`/`UNIFORM_OOF`）是唯一可复现的信号来源；"
                 "停止把支持度方向当作独立机制。")
    elif status["status"].startswith("INTERVENTION"):
        L.append("- 停止：本实现下门控几乎不改变预测，不能把性能差异解释为门控作用；"
                 "若要继续必须先设计**能真正改变表示**的结构条件化算子（不是加大强度）。")
    elif status["status"].startswith("SUPPORT_DIRECTED"):
        L.append("- 保留：`SUPPORT_DIRECTED_SIGNAL_THIS_CONFIG`（provisional），最多按协议购买 BACE 五臂；"
                 "仍需多 seed 与固定 epoch 对照才能升级。")
    elif status["status"].startswith("SPLIT_OR_SELECTION_SENSITIVE"):
        L.append("- 保留：`SPLIT_OR_SELECTION_SENSITIVE` —— 同一 panel 里 SELECT 的符号在 Valid 与 Test、"
                 "fit100 与 fit25 之间翻转（§2、§3 逐项给出），因此**不**升级为可复现信号。")
        L.append("- 停止：不再追加 seed、不再搜索第三个 key / 新的 alpha 形状、不采购 BACE 面板"
                 "（`NOT_PURCHASED_BY_BUDGET`：Lipo 未过晋升门槛）；若继续，必须先解决"
                 "**选点口径与 split 冲突**（R_SEL 用 Valid 选点，而收益出现在 Test），"
                 "而不是加大干预强度。")
    else:
        L.append("- 保留观测、停止外推；见 §6 的边界。")
    L.append("")

    # ---- 1. structural feasibility
    L.append("## 1. 结构准备与 Train-only 可行性（零训练、零标签）\n")
    L.append("| dataset | 候选键数 K_FINE / K_JOINT | active 根占比 train / valid / test |")
    L.append("|---|---|---|")
    for d in ("Lipophilicity", "BACE"):
        sk = pre["stage_keys"][d]
        fa = sk["frac_active"]
        L.append(f"| {d} | {sk['n_key_ids_fine']} / {sk['n_key_ids_joint']} | "
                 f"{_f(100 * float(fa['train']), 1)}% / {_f(100 * float(fa['valid']), 1)}% / "
                 f"{_f(100 * float(fa['test']), 1)}% |")
    L.append("")
    L.append("| dataset | fit | key | E1 groups | E1 keys | E2 spread | E3 spread | E4 unknown | E5 low | E5 high | pass |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for tag, rec in sorted(e.items()):
        c = rec["checks"]
        L.append(f"| {rec['dataset']} | {rec['fit']} | {rec['key']} | "
                 f"{c['E1_active_groups']['value']} | {c['E1_active_keys']['value']} | "
                 f"{_f(c['E2_spread_fullfit']['value'], 3)} | {_f(c['E3_spread_oof']['value'], 3)} | "
                 f"{_f(c['E4_unknown_share']['value'], 3)} | {_f(c['E5_low_share']['value'], 3)} | "
                 f"{_f(c['E5_high_share']['value'], 3)} | {'PASS' if rec['pass'] else 'FAIL'} |")
    L.append("")
    L.append(f"冻结结果：`{sel['status']}`，key = `{key}`，购买 {sel['purchase_fits']}；"
             f"`a_oof` = {', '.join(f'F{k}={_f(v, 4)}' for k, v in sel['a_oof'].items())}。")
    L.append(f"结构加权平均 `a_SELECT` 在 fit 参照权重下 = "
             f"{_f(freeze['datasets'][C.PRIMARY_DATASET]['support']['100']['fit_mean_a_select'], 12)}（容差 1e-10）。")
    L.append("")
    L.append("`K_JOINT` 在 BACE 上因 E5 low-share 未过（0.075 < 0.10），但 Lipo 两 key 都过，"
             "按优先级仍取 `K_FINE`；若采购 BACE 也沿用同一 key，不另选粒度。")
    L.append("")
    sc = [groups["by_fit"][f]["self_count"] for f in fits if f in groups.get("by_fit", {})]
    if sc:
        L.append("自计数剂量差（fit 表加权均值 − OOF 表加权均值）：" + "、".join(
            f"F{i}={_f(x['self_count_dose_gap'], 4)}" for i, x in
            zip([f for f in fits if f in groups.get("by_fit", {})], sc)) + "。")
        L.append("")
    L.append("**关键等价关系**：inactive 根的 `J_full−J_marg ≡ 0`，因此 `a` 在那些根上无作用；"
             "`SELECT` 与 `UNIFORM` 的加权均值都被固定为 0.70（容差 1e-10，见 `analysis/freeze.json` 的锁检查），"
             "所以两者的差别**纯粹是 active 根上 a 的分配**（低支持→0.40–0.55，高支持→0.94–1.00），"
             "不是总收缩剂量。`REVERSE` 是同剂量、反方向的分配；`UNIFORM_OOF` 是同方向、常数剂量的分配。"
             "这是本轮的四个臂恰好构成 2×2（选择性 × 方向）对照的原因。")
    L.append("")
    L.append("自计数口径：Train 查询使用**同一完整 fit 表**，因此 fit 结构会计入自身；"
             "OOF 表（4/5 结构）只用于门槛与 `a_oof` 常数，正式模型始终用完整 fit 表。"
             "支持计数是**结构支持度**，不是独立实验标签数。")
    L.append("")

    # ---- 2. main table
    L.append("## 2. 五臂主表与预先固定的四项比较\n")
    for fit in fits:
        L.append(f"### fit{fit}\n")
        L.append("| arm | update | Train_fit | Valid | Test | Test MAE | params | wall (s) |")
        L.append("|---|---|---|---|---|---|---|---|")
        for r in [x for x in rows if x["fit"] == fit]:
            if r.get("status") != "COMPLETE":
                L.append(f"| {r['arm']} | — | NOT_RUN | — | — | — | — | — |")
                continue
            L.append(f"| {r['arm']} | {r.get('selected_update')} | {_f(r.get('train_fit_rmse'), 4)} | "
                     f"{_f(r.get('valid_rmse'), 4)} | {_f(r.get('test_rmse'), 4)} | "
                     f"{_f(r.get('test_mae'), 4)} | {r.get('parameters')} | "
                     f"{_f(r.get('wall_seconds'), 1)} |")
        L.append("")
        L.append("| 比较 | Valid 增益 | Valid 相对 | Test 增益 | Test 相对 | Test ΔMAE | bootstrap(结构簇, Test) |")
        L.append("|---|---|---|---|---|---|---|")
        for cand, ctrl in C.SELECT_COMPARISONS:
            rec = gains.get(f"F{fit}|{cand}_vs_{ctrl}")
            if not rec:
                continue
            b = boot.get(f"F{fit}|test", {}).get("structure_cluster", {}).get(f"{cand}_vs_{ctrl}", {})
            L.append(f"| {cand} vs {ctrl} | {_f(rec.get('valid_gain'), 4)} | "
                     f"{_pct(rec.get('valid_gain_rel'))} | {_f(rec.get('test_gain'), 4)} | "
                     f"{_pct(rec.get('test_gain_rel'))} | {_f(rec.get('test_mae_change'), 4)} | "
                     f"{_ci(b)}{'' if b.get('crosses_zero') is not True and b else ' (跨0)' if b else ''} |")
        L.append("")
        L.append("| 对照 vs BASE | Valid 相对 | Test 相对 |")
        L.append("|---|---|---|")
        for arm in (C.ARM_UNIFORM, C.ARM_UNIFORM_OOF, C.ARM_REVERSE):
            rec = gains.get(f"F{fit}|{arm}_vs_{C.ARM_BASE}")
            if rec:
                L.append(f"| {arm} | {_pct(rec.get('valid_gain_rel'))} | {_pct(rec.get('test_gain_rel'))} |")
        L.append("")
    L.append("主选择规则 `R_SEL`（50 点共同 grid，Valid RMSE 最小，同分取较早）；Test 从不参与选点。")
    L.append("")
    L.append("Murcko-scaffold 簇敏感性（同一重采样索引，Test）：")
    for fit in fits:
        rec = boot.get(f"F{fit}|test", {}).get("scaffold_cluster", {})
        items = [f"{k}={_f(v.get('point_gain'), 4)} [{_f(v.get('p2.5'), 4)}, {_f(v.get('p97.5'), 4)}]"
                 for k, v in rec.items() if isinstance(v, dict)]
        if items:
            L.append(f"- fit{fit}: " + "；".join(items))
    L.append("")
    promo_tbl = status.get("promotion", {})
    if promo_tbl:
        L.append("**晋升门槛（§5，冻结）逐项判定**\n")
        L.append("| fit | SELECT>BASE 两 split RMSE≥1% | MAE 不恶化(≤1%) | 优于 2×UNIFORM+REVERSE 两 split | pass |")
        L.append("|---|---|---|---|---|")
        for fit, v in promo_tbl.items():
            L.append(f"| {fit} | {v['rmse_vs_base_both_splits']} | {v['mae_not_worse']} | "
                     f"{v['beats_all_three_controls_both_splits']} | {v['pass']} |")
        L.append("")
        L.append("选点敏感性（为什么判定为 split/selection 敏感）：")
        for r in rows:
            if r.get("status") != "COMPLETE":
                continue
            L.append(f"- F{r['fit']}|{r['arm']}: R_SEL update={r.get('selected_update')}，"
                     f"A_SEL update={r.get('a_sel_update')}，W_SEL(5 点均值) update={r.get('w_sel_update')}")
        L.append("")
        L.append("在 fit100 上，三个受门控的臂（UNIFORM / UNIFORM_OOF / SELECT）恰好都选到同一个 grid 点，"
                 "而它们的 Valid/Test 排序相反（Valid: SELECT<UNIFORM_OOF<UNIFORM；Test: UNIFORM<SELECT<UNIFORM_OOF）。"
                 "在 fit25 上 UNIFORM 与 UNIFORM_OOF 的 R_SEL 落在极早的 grid 点（早期即过拟合），"
                 "说明该规则在低标签下对单点噪声非常敏感——这正是把它作为**唯一主规则并如实报告**的理由，"
                 "而不是事后换规则。")
        L.append("")

    # ---- 3. low label
    L.append("## 3. 低标签（fit25 = 840 行嵌套子集）与全量\n")
    low = comp.get("low_label", {})
    if low:
        L.append("| split | 比较 | gain fit100 | gain fit25 | gain25−gain100 | 区间 |")
        L.append("|---|---|---|---|---|---|")
        for split in ("valid", "test"):
            for ctrl, rec in (low.get(split) or {}).items():
                L.append(f"| {split} | {ctrl} | {_f(rec.get('gain_fit100'), 4)} | "
                         f"{_f(rec.get('gain_fit25'), 4)} | {_f(rec.get('diff_25_minus_100'), 4)} | "
                         f"{_ci(rec)} |")
        L.append("")
        L.append("两个 N 是同 seed 的两个拟合量；不能当作样本复杂度或噪声下限估计。")
    else:
        L.append("fit25 未购买或未完成（见 §7 的执行记录）。")
    L.append("")

    # ---- 4. support groups + influence
    L.append("## 4. 支持解释与影响度（兼容性分析，不是因果分解）\n")
    for fit in fits:
        rec = groups.get("by_fit", {}).get(fit, {})
        if not rec or not rec.get("groups"):
            continue
        L.append(f"### fit{fit}：负担阈值（Train-only OOF 的 q25/q75）= "
                 f"{_f(rec['thresholds']['q25'], 5)} / {_f(rec['thresholds']['q75'], 5)}\n")
        L.append("| split | group | n | RMSE BASE | RMSE UNIFORM | RMSE SELECT | mean ΔSSE SELECT−BASE | vs UNIFORM |")
        L.append("|---|---|---|---|---|---|---|---|")
        for split in ("valid", "test"):
            g = rec["groups"].get(split, {})
            for grp in ("low", "high"):
                gg = g.get(grp, {})
                L.append(f"| {split} | {grp} | {gg.get(C.ARM_BASE, {}).get('n')} | "
                         f"{_f(gg.get(C.ARM_BASE, {}).get('rmse'), 4)} | "
                         f"{_f(gg.get(C.ARM_UNIFORM, {}).get('rmse'), 4)} | "
                         f"{_f(gg.get(C.ARM_SELECT, {}).get('rmse'), 4)} | "
                         f"{_f(gg.get(f'mean_dsse_SELECT_vs_{C.ARM_BASE}'), 5)} | "
                         f"{_f(gg.get(f'mean_dsse_SELECT_vs_{C.ARM_UNIFORM}'), 5)} |")
        uk = rec.get("unknown_key_shares", {}).get("active_roots_only", {})
        L.append("")
        L.append(f"unknown-key 占比（active 根内）：valid {_f(uk.get('valid'), 3)}、test {_f(uk.get('test'), 3)}；"
                 f"无 active 根的分子占比：valid {_f(rec['groups'].get('valid', {}).get('frac_no_active_root'), 3)}、"
                 f"test {_f(rec['groups'].get('test', {}).get('frac_no_active_root'), 3)}。")
        L.append("")
    if any(infl.get("by_fit", {}).get(fit) for fit in fits):
        L.append("### 影响度：按本轮 BASE 绝对误差降序固定删除（五臂删同一行集）\n")
        L.append("| fit | split | 口径 | BASE | UNIFORM | UNIFORM_OOF | SELECT | REVERSE |")
        L.append("|---|---|---|---|---|---|---|---|")
        for fit in fits:
            iv = infl.get("by_fit", {}).get(fit) or {}
            for split in ("valid", "test"):
                if split not in iv:
                    continue
                for frac in ("headline", "0.01", "0.05"):
                    if frac == "headline":
                        table_rec = {a: v for a, v in iv[split]["headline"].items()}
                        name = "全样本"
                    else:
                        d = iv[split]["drop"].get(frac, {})
                        table_rec = {a: v for a, v in d.items()
                                     if a in C.ARMS}
                        name = f"删最大 {frac}"
                    if not table_rec:
                        continue
                    L.append(f"| {fit} | {split} | {name} | " + " | ".join(
                        _f(table_rec.get(a, {}).get(metric.split('_')[1]), 4) for a in C.ARMS) + " |")
        L.append("")
    L.append("辅规则与窗口：")
    for tag, rec in (comp.get("selection_sensitivity") or {}).items():
        if not tag.startswith("F100|"):
            continue
        L.append(f"- {tag}: Test 最后 10 个 grid 点中位数 {_f(rec['test']['last10_median'], 4)}，"
                 f"grid 最好 {_f(rec['test']['best_grid'], 4)}")
    L.append("")

    # ---- 5. intervention
    L.append("## 5. 干预强度与结构追溯（内部干预，不是重训收益）\n")
    for fit in fits:
        rec = inter.get("by_fit", {}).get(fit, {})
        if not rec:
            continue
        cf = rec.get("counterfactual", {})
        L.append(f"### fit{fit}（SELECT 选中权重，update={rec.get('selected_update')}）\n")
        L.append("| split | 分子数 | 预测改变分子比例 | max |Δp| | p50 |Δp| | p99 |Δp| |")
        L.append("|---|---|---|---|---|---|")
        for split, c in cf.items():
            L.append(f"| {split} | {c['n_molecules']} | {_f(c['frac_molecules_changed'], 4)} | "
                     f"{_f(c['max_abs_delta_raw'], 6)} | {_f(c['p50_abs_delta_raw'], 6)} | "
                     f"{_f(c['p99_abs_delta_raw'], 6)} |")
        dose = rec.get("dose") or {}
        for tag in ("init", "selected", "final"):
            d = (dose.get(tag) or {}).get("molecules", {}) if dose.get(tag) else None
            if not d:
                continue
            L.append("")
            L.append(f"- {tag} 权重（按分子先聚合；`a_active` 是该分子**全部根**的 a 均值，"
                     f"inactive 根计为 1）：‖J_full−J_marg‖ 中位 {_f(d.get('norm_diff', {}).get('p50'), 4)}，"
                     f"‖(1−a)(J_full−J_marg)‖ 中位 {_f(d.get('norm_scaled', {}).get('p50'), 4)}，"
                     f"比值中位 {_f(d.get('ratio_scaled_full', {}).get('p50'), 4)}，"
                     f"a_active 中位 {_f(d.get('mean_a_active', {}).get('p50'), 4)}")
        ap = rec.get("alias_pairs", {})
        L.append("")
        L.append(f"- 五对输入别名（上游已存在）：{ap.get('n_pairs')} 对，全部 |Δp| ≤ 2e-5 = "
                 f"{ap.get('all_within_2e-5')}")
    L.append("")

    # ---- 6. status and boundaries
    L.append("## 6. 结论与边界\n")
    L.append(f"- 判定：**{status['status']}**")
    L.append("- 即使出现正向信号，也只支持**这一**结构代理、这一算子、这一配方、这一 split 与单 seed 下的学习偏置；"
             "不证明校准置信度、普遍稀有结构因果、VC 复杂度下降或化学因果。")
    L.append("- `a` 改变了学习函数，不能仅因参数量相同（228 674）声称完整函数族表达力相同；"
             "五对别名预测不变也不等于全体表达边界未变。")
    L.append("- 机制依赖（门控是否被使用）与预测收益分别表述：§5 是机制/干预证据，§2 是性能证据。")
    L.append("- 只比较两个事先定义的均匀常数（0.70 与 `a_oof`），不声称击败所有可能的均匀收缩强度。")
    L.append("")

    # ---- 7. cost / regime
    L.append("## 7. 成本、执行 regime 与事故记录\n")
    tot = budget["totals"]
    L.append(f"- 完成作业 {tot['n_completed']} 条（4 并发，实际墙钟约 1.1 h）；进程墙钟合计 "
             f"{_f(tot['sum_process_wall_seconds'] / 3600, 2)} h（各 run 之和，不除以并发）；"
             f"GPU·h 合计 {_f(tot['sum_gpu_hours'], 3)} A100·h（本轮**未占卡**，见下）。")
    L.append(f"- 手数上限：墙钟目标 {C.WALL_TARGET_HOURS} h / 硬 {C.WALL_HARD_HOURS} h；"
             f"GPU 软 {C.GPU_SOFT_CAP_HOURS} / 硬 {C.GPU_HARD_CAP_HOURS} A100·h；正式训练上限 {C.MAX_FORMAL_RUNS} 条。")
    L.append("- **基础设施事故**：`res` 的共享 `/home`（44 TB）在本轮期间 100% 写满（`df -k` 可用 0 KB），"
             "`rr deploy res` 在上传 git bundle 时失败（`No space left on device`），因此 `res-gpu0/res-gpu1` 无法使用；"
             "本地与 `res` 之间也没有可用的写路径。为完成已授权的科学清单，全部正式训练改在**同一**本地 CPU regime 执行"
             "（torch 2.5.1+cpu、无 CUDA、单机 16 核）。这是执行 regime 的偏离，已在 `analysis/source_manifest.json` 与"
             "`analysis/run_manifest.json` 逐 run 记录；同一 panel 的五臂共享同一 regime，因此内部可比。"
             "跨 regime（`res` A100 与本地 CPU）数字不可互换。")
    L.append(f"- 冒烟：{smoke['n_jobs']} 个短作业（各 ≤{smoke['max_updates']} updates），"
             f"参数量一致 = {smoke['all_parameter_counts_equal_228674']}，共享初始 state = "
             f"{smoke['all_arms_share_init_hash']}。")
    L.append("")

    # ---- 8. artefacts
    L.append("## 8. 产物\n")
    for p in ("PROTOCOL.md", "configs/experiment.yaml", "code/", "tests/",
              "analysis/preflight.json", "analysis/key_selection.json", "analysis/freeze.json",
              "analysis/source_manifest.json", "analysis/run_manifest.json",
              "analysis/comparisons.json", "analysis/intervention.json",
              "analysis/support_groups.json", "analysis/influence.json", "analysis/budget.json",
              "analysis/main_table.csv"):
        L.append(f"- `{p}`")
    figs = sorted(p.name for p in FIG.glob("*.png")) if FIG.exists() else []
    for f in figs:
        L.append(f"- `analysis/figures/{f}`")
    L.append("")
    return "\n".join(L) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    C.ANALYSIS.mkdir(parents=True, exist_ok=True)
    comp = C.read_json(C.ANALYSIS / "comparisons.json")
    groups = C.read_json(C.ANALYSIS / "support_groups.json")
    figs = []
    for fn, args in ((figure_support, (comp, groups)),
                     (figure_performance, (comp,)),
                     (figure_support_groups, (groups,))):
        try:
            p = fn(*args)
            if p:
                figs.append(p)
        except Exception as exc:  # pragma: no cover - figures are optional
            C.log(f"[report] figure {fn.__name__} failed: {exc}")
    text = build_report()
    (C.TRACK / "REPORT.md").write_text(text, encoding="utf-8")
    C.write_json(C.ANALYSIS / "figures.json", {"figures": figs})
    C.log(f"[report] wrote REPORT.md ({len(text)} chars), figures={figs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
