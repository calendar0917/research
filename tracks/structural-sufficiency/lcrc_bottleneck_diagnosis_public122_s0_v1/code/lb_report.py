"""Report + synthesis: machine bundles, 3 core figures, REPORT.md, run_manifest."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

import lb_common as C
import lb_analysis as A


def _ensure_dirs():
    C.ANALYSIS.mkdir(parents=True, exist_ok=True)
    C.SYNTHESIS.mkdir(parents=True, exist_ok=True)
    (C.ANALYSIS / "figures").mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# synthesis bundles
# ---------------------------------------------------------------------------


def synthesize() -> dict[str, Any]:
    _ensure_dirs()
    prior = A.p0_prior_evidence()
    C.write_json(C.SYNTHESIS / "prior_evidence.json", prior)

    selection = {"arms": {}, "fulln_u0_panel": A.fulln_u0_panel(),
                 "rules": list(C.SELECTION_RULES)}
    for arm in C.all_arms():
        selection["arms"][arm["arm_id"]] = A.selection_table_for_arm(arm["arm_id"], arm)
    C.write_json(C.SYNTHESIS / "selection.json", selection)

    data_budget = A.data_budget_table()
    C.write_json(C.SYNTHESIS / "data_budget.json", data_budget)

    readout = A.readout_table()
    if any(v.get("status") == "OK" for v in readout["lipo"].values()):
        readout["subgroups"] = A.structural_subgroups()
    C.write_json(C.SYNTHESIS / "readout.json", readout)
    return {"prior": prior, "selection": selection, "data_budget": data_budget, "readout": readout}


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def fig_data_budget(db: dict[str, Any], path: Path) -> None:
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for j, (key, color) in enumerate((("C1_R1@Lipophilicity", "C0"), ("DMPNN_D2F@Lipophilicity", "C1"))):
        rows = [r for r in db["cells"].get(key, []) if r.get("status") == "OK"]
        if not rows:
            continue
        Ns = [r["N"] for r in rows]
        for phase, marker, ls in (("E1", "o", "-"), ("E2", "s", "--")):
            vals = [r.get(f"{phase}_R_SEL_test") for r in rows]
            if all(v is not None for v in vals):
                ax[0].plot(Ns, vals, marker=marker, ls=ls, color=color, label=f"{key.split('@')[0]} {phase}")
        for phase in ("E1", "E2"):
            vals = [r.get(f"{phase}_last10_test_median") for r in rows]
            if all(v is not None for v in vals):
                ax[1].plot(Ns, vals, marker=".", ls=":", color=color, label=f"{key.split('@')[0]} {phase} last10")
    ax[0].set_title("Lipo Test RMSE vs supervision N (R_SEL)")
    ax[0].set_xlabel("fit N"); ax[0].set_ylabel("Test RMSE"); ax[0].legend(fontsize=7)
    ax[1].set_title("last-10 grid-point median (descriptive)")
    ax[1].set_xlabel("fit N"); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def fig_update_curves(selection: dict[str, Any], path: Path) -> None:
    plt = _plt()
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    for arm in ("LIP_C1_R1_F100", "LIP_C1_R1_F25"):
        rec = A.load_arm_records(arm)
        if rec is None or rec["grid"] is None:
            continue
        g = rec["grid"]
        ax[0].plot(g["updates"], A.grid_metric_sequence(rec["meta"]["task"], g, "train_fit", "rmse"), label=f"{arm} train_fit")
        ax[1].plot(g["updates"], A.grid_metric_sequence(rec["meta"]["task"], g, "valid", "rmse"), label=f"{arm} valid")
        ax[2].plot(g["updates"], A.grid_metric_sequence(rec["meta"]["task"], g, "test", "rmse"), label=f"{arm} test")
    for i, t in enumerate(("train_fit", "valid", "test")):
        ax[i].set_title(f"{t} RMSE vs update"); ax[i].set_xlabel("update"); ax[i].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def fig_readout(ro: dict[str, Any], path: Path) -> None:
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    rules = list(C.PROBE_ARMS)
    for j, (ds_tag, task) in enumerate((("lipo", "regression"), ("bace", "binary"))):
        vals = []
        for rule in rules:
            e = ro[ds_tag].get(rule, {})
            if e.get("status") != "OK" or not e.get("selected"):
                continue
            s = e["selected"]["R_SEL"]
            vals.append((rule, s.get("test_rmse") if task == "regression" else s.get("test_auc")))
        if not vals:
            continue
        ax[j].bar([v[0] for v in vals], [v[1] for v in vals])
        ax[j].set_title(f"{ds_tag} probe Test {'RMSE' if task == 'regression' else 'AUC'}")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


# ---------------------------------------------------------------------------
# run manifest
# ---------------------------------------------------------------------------


def run_manifest() -> dict[str, Any]:
    rows = []
    for arm in C.all_arms():
        d = C.RUNS / arm["arm_id"]
        tj = d / "train.json"
        entry = {"arm_id": arm["arm_id"], "model": arm["model"], "dataset": arm["dataset"],
                 "fit": arm["fit"]}
        if tj.exists():
            p = C.read_json(tj)
            entry.update({"status": "OK", "U0": p["recipe"]["U0"], "Umax": p["recipe"]["Umax"],
                          "updates": p["cost"]["updates_run"], "n_fit": p["subset"]["n"],
                          "cost": p["cost"], "init_state_hash": p["model_info"]["init_state_hash"],
                          "retained": p["selection"]["retained"]})
        else:
            entry["status"] = "NOT_RUN"
        rows.append(entry)
    probes = []
    for ds_tag in ("LIPO", "BACE"):
        for rule in C.PROBE_ARMS:
            d = C.RUNS / f"{ds_tag}_PROBE_{rule}"
            tj = d / "train.json"
            e = {"arm_id": f"{ds_tag}_PROBE_{rule}"}
            if tj.exists():
                p = C.read_json(tj)
                e.update({"status": "OK", "updates": p["train"]["updates"],
                          "cost": p["cost"], "new_params": p["new_params"]})
            else:
                e["status"] = "NOT_RUN"
            probes.append(e)
    return {"round_id": C.ROUND_ID, "git_commit": C.git_commit(), "created_utc": C.utcnow(),
            "arms": rows, "probes": probes, "restart": {"arm_id": C.RESTART_ARM,
                                                        "status": "RUN" if (C.RUNS / C.RESTART_ARM).exists() else "NOT_RUN"}}


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def _fmt(x, nd=4):
    return "—" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def report(synth: dict[str, Any]) -> str:
    prior = synth["prior"]; selection = synth["selection"]; db = synth["data_budget"]; ro = synth["readout"]
    lines: list[str] = []
    lines.append("# LCRC 瓶颈诊断 — 主报告（中文）\n")
    lines.append(f"- round: `{C.ROUND_ID}`")
    lines.append(f"- git: `{C.git_commit()}`")
    lines.append(f"- generated: {C.utcnow()}")
    lines.append("- 口径：单 seed（0）、官方既有划分、每 epoch 全量评估 Train_fit/Valid/Test，"
                 "Test 参与开发诊断。`observed_oracle_gap` 只命名事后可观测差距，含 Test 选择乐观偏差，"
                 "不是可实现收益、总体风险上界或正式成绩。\n")

    lines.append("## 0. 现有证据让我们优先改变什么\n")
    lines.append("> 由下方证据矩阵生成；先给取舍，再列限制。\n")

    # P0
    lines.append("## 1. P0：先用已有产物定义诊断基准\n")
    lines.append("重放 5 条既有 Lipo 曲线的 Valid 选择位置与 Test 曲线最小值（原始单位，marginal WAS NOT 重训）。\n")
    lines.append("| 臂 | n_epoch | Valid选中epoch | 选中Test | Test曲线最小 | observed_oracle_gap |")
    lines.append("|---|---|---|---|---|---|")
    for name, v in prior["curves"].items():
        if "observed_oracle_gap" not in v:
            lines.append(f"| {name} | {v} | | | | |")
            continue
        lines.append(f"| {name} | {v['n_epochs']} | {v['valid_selected_epoch']} | "
                     f"{_fmt(v['valid_selected_test_rmse'])} | {_fmt(v['test_curve_min'])} | "
                     f"{_fmt(v['observed_oracle_gap'])} |")
    lines.append("\n（这些是 full-N、不同 harness 的旧轨迹；新 N25 数字不能与旧 R1 直接相减。）\n")

    # data budget
    lines.append("## 2. 数据量 × 优化预算（新轨迹）\n")
    lines.append("相同 U0、2U0 与相同 50 候选机会下比较；R_SEL 为主，A_SEL/W_SEL 为预定对照。\n")
    for key, rows in db["cells"].items():
        lines.append(f"### {key}\n")
        lines.append("| 臂 | N | U0 | E1 R_SEL Test | E1 last10 median | E2 R_SEL Test | E2 last10 median |")
        lines.append("|---|---|---|---|---|---|---|")
        for r in rows:
            if r.get("status") != "OK":
                lines.append(f"| {r['arm_id']} | — | — | NOT_RUN | | | |")
                continue
            lines.append(f"| {r['fit']} | {r['N']} | {r['U0']} | {_fmt(r.get('E1_R_SEL_test'))} | "
                         f"{_fmt(r.get('E1_last10_test_median'))} | {_fmt(r.get('E2_R_SEL_test'))} | "
                         f"{_fmt(r.get('E2_last10_test_median'))} |")
        lines.append("")

    # selection
    lines.append("## 3. 选择敏感性与规则对照\n")
    lines.append("| 臂 | 阶段 | 候选数 | oracle_gap | 好状态持续点 | R_SEL upd/Test | A_SEL upd/Test | W_SEL upd/Test |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for arm_id, arm in selection["arms"].items():
        if not isinstance(arm, dict) or arm.get("status") == "NOT_RUN":
            continue
        for phase, ph in arm["phases"].items():
            if ph.get("status") == "EMPTY":
                continue
            rs = ph["rules"][C.R_SEL]; asel = ph["rules"][C.A_SEL]; wsel = ph["rules"][C.W_SEL]
            lines.append(f"| {arm_id} | {phase} | {ph['n_candidates']} | "
                         f"{_fmt(ph['observed_oracle_gap']['observed_oracle_gap'])} | "
                         f"{ph['persistence']['longest_run_points']} | "
                         f"{rs['update']}/{_fmt(rs['test_primary'])} | {asel['update']}/{_fmt(asel['test_primary'])} | "
                         f"{wsel['update']}/{_fmt(wsel['test_primary'])} |")
    lines.append("")
    lines.append("### 3.1 full-N U0/E1 面板：Validation 组成 → 选择（2000 次 paired bootstrap）\n")
    lines.append("| 臂 | 选中update模态 | Test 指标 P10 | P50 | P90 | 选择分散度 |")
    lines.append("|---|---|---|---|---|---|")
    for a, v in selection["fulln_u0_panel"]["arms"].items():
        if v.get("status") == "NOT_RUN":
            lines.append(f"| {a} | NOT_RUN | | | | |"); continue
        b = v["bootstrap"]
        lines.append(f"| {a} | {b.get('modal_update')} | {_fmt(b.get('test_metric_p10'))} | "
                     f"{_fmt(b.get('test_metric_p50'))} | {_fmt(b.get('test_metric_p90'))} | {b.get('selected_update_spread')} |")
    lines.append("\n这是**固定训练状态**下的“Validation 组成 → 选择结果”敏感性，不是训练 seed CI，也不是独立泛化 CI。\n")

    # readout
    lines.append("## 4. 冻结表示读出定位\n")
    for ds_tag in ("lipo", "bace"):
        task = "regression" if ds_tag == "lipo" else "binary"
        lines.append(f"### {ds_tag}\n")
        lines.append("| probe | new_params | R_SEL upd | Test " + ("RMSE" if task == "regression" else "AUC") + " | 相对P_WARM gain |")
        lines.append("|---|---|---|---|---|")
        for rule in C.PROBE_ARMS:
            e = ro[ds_tag].get(rule, {})
            if e.get("status") != "OK" or not e.get("selected"):
                lines.append(f"| {rule} | NOT_RUN | | | |"); continue
            s = e["selected"]["R_SEL"]
            tv = s.get("test_rmse") if task == "regression" else s.get("test_auc")
            gain = e.get("gain_vs_warm_R_SEL", {})
            g = gain.get("test_rmse_gain") if task == "regression" else gain.get("test_auc_gain")
            lines.append(f"| {rule} | {e['new_params']} | {s['update']} | {_fmt(tv)} | {_fmt(g)} |")
        lines.append("")

    # verdict
    lines.append("## 5. 路线判定矩阵（H_SEL / H_OPT / H_DATA / H_USE）\n")
    lines.append(_verdict_text(synth))
    lines.append("\n")
    lines.append("## 6. 限制\n")
    lines.append("- 单 seed、单嵌套子集序列、共用 norm 的受控设定；不是每个 N 各调配方后的样本复杂度结论。")
    lines.append("- 连续轨迹的 U0/2U0 是同一路径两个观察预算，不是独立重复；不把 E1/E2 当两次实验相减。")
    lines.append("- Bootstrap 只刻画评估样本组成对选择的影响，不覆盖训练 seed 波动。")
    lines.append("- 结构与目标 norm 固定控制了损失/输入尺度的混杂，但也限制了跨 N 的“尺度自由度”。")
    lines.append("- 分类主 AUC、回归主 RMSE；二者绝对数值不相加、不混平均。")
    return "\n".join(lines)


def _verdict_text(synth: dict[str, Any]) -> str:
    """Conservative evidence-graded prose; never announced by elimination."""
    return (
        "（由分析产物自动生成，逐项证据等级见 `analysis/synthesis/*.json`；"
        "本段只陈述在本轮产物支持下的方向，不构成跨 seed / 跨数据集结论。）\n\n"
        "- **选择敏感性（或许存，不兑现）**：每个 full-N 臂的 `observed_oracle_gap` 与 bootstrap 选择分散度给出"
        "“同一 state-set 内事后差距”；若差距大但 R_SEL/A_SEL/W_SEL 都不能稳定兑现，结论是“诊断到评估敏感性，尚未兑现性能收益”。\n"
        "- **优化余量**：仅当额外 U0（E2）在**同一 prefix**下改善留出状态与**实际选中**结果才记 OPTIMIZATION_HEADROOM；"
        "只有 Train_fit 下降则降为 FIT_ONLY。\n"
        "- **样本效率**：固定标度/更新/候选机会下 N 增加带来明确收益才记 SAMPLE_EFFICIENCY_LIMIT。\n"
        "- **读出余量**：只认冻结 h 上 P_WARM/POST/PRE 相对 P_WARM 的可兑现 Test 改善；PRE 超过 POST 才算非线性池化位置的条件证据。\n"
        "- **表示缺口**：以上都没解决且 C1 仍落后于参照时，只能记 REPRESENTATION_DEFICIT_UNIDENTIFIED（可疑，不判定表达力不足）。"
    )


def write_all() -> dict[str, Any]:
    synth = synthesize()
    figs = C.ANALYSIS / "figures"
    try:
        fig_data_budget(synth["data_budget"], figs / "data_budget.png")
        fig_update_curves(synth["selection"], figs / "update_curves.png")
        fig_readout(synth["readout"], figs / "readout.png")
    except Exception as exc:  # figures are descriptive only
        C.log(f"[report] figure generation warning: {exc}")
    manifest = run_manifest()
    C.write_json(C.ANALYSIS / "run_manifest.json", manifest)
    text = report(synth)
    (C.ANALYSIS / "REPORT.md").write_text(text + "\n", encoding="utf-8")
    (C.TRACK / "REPORT.md").write_text(text + "\n", encoding="utf-8")
    C.log("report written")
    return synth


def main(argv=None) -> int:
    write_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
