"""Tables, figures and the Chinese report for the round.

Everything here reads only files written by the training runs and by
``lp_analysis`` / ``lp_mechanism``; nothing is recomputed from a model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import lp_common as C
import lp_analysis as A


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "–"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "n/a"
    return f"{number:.{digits}f}"


def _pct(value: Any) -> str:
    if value is None:
        return "–"
    return f"{100.0 * float(value):+.2f}%"


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------


def main_table(tables: Mapping[str, Any]) -> str:
    lines = [
        "| 臂 | 参数 | best-Val ep | Val RMSE | Test RMSE | last Val | last Test | 末窗 Val | 末窗 Test | Test MAE(best) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in A.MAIN_ARMS:
        if arm not in tables:
            continue
        row = tables[arm]
        best = row["best_valid"]
        last = row["last_epoch"]
        late = row["late_window"]
        lines.append(
            "| {arm} | {p} | {ep} | {v} | {t} | {lv} | {lt} | {wv} | {wt} | {mae} |".format(
                arm=arm,
                p=_fmt(row["parameters"]),
                ep=_fmt(row["best_epoch"]),
                v=_fmt(best["valid"]["rmse"]),
                t=_fmt(best["test"]["rmse"]),
                lv=_fmt(last["valid"]["rmse"]),
                lt=_fmt(last["test"]["rmse"]),
                wv=_fmt(late["valid_rmse"]),
                wt=_fmt(late["test_rmse"]),
                mae=_fmt(best["test"]["mae"]),
            )
        )
    lines += [
        "",
        "`末窗` = 最后 20 个 epoch（G 为最后 10 个）逐 epoch 指标的均值，不是预测 ensemble；"
        "`last` = 最后一轮同一权重的全量评估。全部为原始 logD 单位。",
    ]
    return "\n".join(lines)


def comparison_table(comparisons: Mapping[str, Any], judgements: Mapping[str, Any]) -> str:
    lines = [
        "| 比较 | Val(best) Δ% | Test(best) Δ% | Test(last) Δ% | Test(末窗) Δ% | Test MAE Δ% | 赢率 | 删尾1% Test Δ% | 删尾5% Test Δ% | 标签 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, comp in comparisons.items():
        verdict = judgements.get(name, {})
        best = comp["best_valid"]
        last = comp["last_epoch"]
        late = comp["late_window"]
        test_mol = comp["per_molecule"]["test"]
        lines.append(
            "| {name} | {vb} | {tb} | {tl} | {tw} | {mae} | {win} | {t1} | {t5} | {label} |".format(
                name=name,
                vb=_pct(best["valid"]["rel_improvement"]),
                tb=_pct(best["test"]["rel_improvement"]),
                tl=_pct(last["test"]["rel_improvement"]),
                tw=_pct(late["test_rel_improvement"]),
                mae=_pct(best["test"]["rel_mae_improvement"]),
                win=_fmt(test_mol["win_rate"], 3),
                t1=_pct(test_mol["trim_0.01"]["rel_improvement"]),
                t5=_pct(test_mol["trim_0.05"]["rel_improvement"]),
                label=verdict.get("label", "–"),
            )
        )
    lines += [
        "",
        "正 Δ% = 候选优于 control。删尾集合按 control 的绝对误差排序取最差 1%/5%，两侧删同一批行；"
        "完整官方指标不变。",
    ]
    return "\n".join(lines)


def bootstrap_table(bootstrap: Mapping[str, Any]) -> str:
    if not bootstrap:
        return "（未计算）"
    lines = [
        "| split | 比较 | ΔRMSE 点估计 | 2.5% | 97.5% | ΔMAE 点估计 | 2.5% | 97.5% |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, payload in bootstrap.get("paired_deltas", {}).items():
        for split in ("valid", "test"):
            entry = payload.get(split)
            if not entry:
                continue
            lines.append(
                f"| {split} | {name} | {_fmt(entry['delta_rmse_point'])} | {_fmt(entry['lo'])} | "
                f"{_fmt(entry['hi'])} | {_fmt(entry['delta_mae_point'])} | {_fmt(entry['mae_lo'])} | {_fmt(entry['mae_hi'])} |"
            )
    lines += [
        "",
        f"{bootstrap.get('replicates')} 次 Murcko scaffold 簇正权重 Exp(1) 扰动，所有模型共享同一组权重；"
        "区间描述固定模型的样本不确定性，不代表训练 seed 方差。",
    ]
    return "\n".join(lines)


def group_table(groups: Mapping[str, Any]) -> str:
    lines = ["### 群体规模", ""]
    lines.append("| split | 分子数 | H1 分子 | H2 分子 | H3 分子 | 缺失 radius-2 根 | 解析失败 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for split in C.SPLITS:
        entry = groups["n"][split]
        lines.append(
            f"| {split} | {entry['molecules']} | {entry['H1_molecules']} | {entry['H2_molecules']} | "
            f"{entry['H3_molecules']} | {entry['missing_roots']} | {entry['parse_failed']} |"
        )
    lines += ["", "### size-standardised MAE gap（B 的 best-Val 点选组，Test 冻结后读取）", ""]
    lines.append("| 群体 | split | B gap | 可估计 | 群体 n | 非群体 n |")
    lines.append("|---|---|---:|---|---:|---:|")
    for name in ("H1", "H2", "H3"):
        for split in ("valid", "test"):
            profile = groups["profiles"][name][split].get("B")
            if not profile:
                continue
            lines.append(
                f"| {name} | {split} | {_fmt(profile.get('gap'))} | "
                f"{'是' if profile.get('estimable') else '否'} | {profile.get('n_group')} | {profile.get('n_other')} |"
            )
    selection = groups.get("selection", {})
    lines += [
        "",
        f"选组结果：`{selection.get('label')}`"
        + (f"（{selection.get('group')}，Val gap={_fmt(selection.get('selected_valid_gap'))}）" if selection.get("group") else ""),
        "",
    ]
    lines.append("| 群体 | split | 臂 | MAE(群体) | MAE(非群体) | gap | gap 2.5% | gap 97.5% |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|")
    for name in ("H1", "H2", "H3"):
        for split in ("valid", "test"):
            boot = groups.get("bootstrap", {}).get(name, {}).get(split, {})
            for arm, entry in (boot.get("models") or {}).items():
                profile = groups["profiles"][name][split][arm]
                lines.append(
                    f"| {name} | {split} | {arm} | {_fmt(profile.get('mae_group'))} | {_fmt(profile.get('mae_other'))} | "
                    f"{_fmt(entry['gap_point'])} | {_fmt(entry['gap_lo'])} | {_fmt(entry['gap_hi'])} |"
                )
    lines.append("")
    lines.append("| 群体 | split | 比较 | Δgap 点估计 | 2.5% | 97.5% |")
    lines.append("|---|---|---|---:|---:|---:|")
    for name in ("H1", "H2", "H3"):
        for split in ("valid", "test"):
            boot = groups.get("bootstrap", {}).get(name, {}).get(split, {})
            for comp, entry in (boot.get("paired_group_deltas") or {}).items():
                lines.append(
                    f"| {name} | {split} | {comp} | {_fmt(entry['point'])} | {_fmt(entry['lo'])} | {_fmt(entry['hi'])} |"
                )
    return "\n".join(lines)


def intervention_table(interventions: Mapping[str, Any]) -> str:
    lines = []
    for tag, entry in interventions.get("checkpoints", {}).items():
        lines.append(f"### D2 冻结权重（{tag}，epoch {entry.get('epoch')}）")
        lines.append("")
        lines.append("| split | NORMAL RMSE | REUSE_ALL RMSE | Δpred max | FIXED_ALL RMSE | Δpred max | REUSE_TARGET RMSE | Δpred max | REUSE_CONTROL RMSE | Δpred max | 恒等式残差 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for split, payload in entry["splits"].items():
            target = payload.get("reuse_target", {})
            control = payload.get("reuse_control", {})
            identity = max(
                float(target.get("identity_max_abs_residual", 0.0) or 0.0),
                float(control.get("identity_max_abs_residual", 0.0) or 0.0),
            )
            lines.append(
                f"| {split} | {_fmt(payload['normal_rmse'])} | {_fmt(payload['reuse_all_rmse'])} | "
                f"{_fmt(payload['reuse_all_delta_pred_max'])} | {_fmt(payload['fixed_all_rmse'])} | "
                f"{_fmt(payload['fixed_all_delta_pred_max'])} | {_fmt(target.get('rmse'))} | "
                f"{_fmt(target.get('delta_pred_max'))} | {_fmt(control.get('rmse'))} | "
                f"{_fmt(control.get('delta_pred_max'))} | {_fmt(identity, 8)} |"
            )
        lines.append("")
        for split, payload in entry["splits"].items():
            match = payload.get("match")
            if match:
                lines.append(f"* {split} 匹配：`{match.get('label')}` 目标根 {match.get('n_target_roots')} / 控制根 {match.get('n_control_roots')}")
        lines.append("")
    if not lines:
        lines = ["（未做干预：缺少 D2 冻结权重）"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def figures(tables: Mapping[str, Any], comparisons: Mapping[str, Any], groups: Mapping[str, Any], interventions: Mapping[str, Any]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C.FIGURES.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    # 1. learning curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for split, ax in zip(C.SPLITS, axes):
        for arm in A.MAIN_ARMS:
            if arm not in tables:
                continue
            curve = A.load_curve(arm)
            epochs = [int(r["epoch"]) for r in curve]
            ax.plot(epochs, [float(r[f"{split}_rmse"]) for r in curve], label=arm, linewidth=1.2)
        ax.set_title(f"{split} RMSE (raw logD)")
        ax.set_xlabel("epoch")
        ax.set_ylabel("RMSE")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    path = C.FIGURES / "learning_curves.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(C.track_rel(path))

    # 2. RMSE / MAE comparison + tail sensitivity
    arms = [a for a in A.MAIN_ARMS if a in tables]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    x = np.arange(len(arms))
    width = 0.38
    for ax, split, title in zip(axes[:2], ("valid", "test"), ("Validation (best-Val)", "Test (best-Val)")):
        rmse_vals = [tables[a]["best_valid"][split]["rmse"] for a in arms]
        mae_vals = [tables[a]["best_valid"][split]["mae"] for a in arms]
        ax.bar(x - width / 2, rmse_vals, width, label="RMSE")
        ax.bar(x + width / 2, mae_vals, width, label="MAE")
        ax.set_xticks(x)
        ax.set_xticklabels(arms)
        ax.set_title(title)
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8)
    ax = axes[2]
    names = list(comparisons)
    trimmed1 = [comparisons[n]["per_molecule"]["test"]["trim_0.01"]["rel_improvement"] for n in names]
    trimmed5 = [comparisons[n]["per_molecule"]["test"]["trim_0.05"]["rel_improvement"] for n in names]
    full = [comparisons[n]["best_valid"]["test"]["rel_improvement"] for n in names]
    xx = np.arange(len(names))
    ax.bar(xx - 0.26, full, 0.26, label="full Test")
    ax.bar(xx, trimmed1, 0.26, label="drop worst 1%")
    ax.bar(xx + 0.26, trimmed5, 0.26, label="drop worst 5%")
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_xticks(xx)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("relative RMSE improvement")
    ax.set_title("tail sensitivity (Test, best-Val)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = C.FIGURES / "rmse_mae_tail.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(C.track_rel(path))

    # 3. group gaps
    if groups:
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        labels = []
        values = []
        lows = []
        highs = []
        for name in ("H1", "H2", "H3"):
            boot = groups.get("bootstrap", {}).get(name, {}).get("test", {})
            entry = (boot.get("models") or {}).get("B")
            if not entry:
                continue
            labels.append(f"{name} (B, test)")
            values.append(entry["gap_point"])
            lows.append(entry["gap_lo"])
            highs.append(entry["gap_hi"])
            d2 = (boot.get("models") or {}).get("D2")
            if d2:
                labels.append(f"{name} (D2, test)")
                values.append(d2["gap_point"])
                lows.append(d2["gap_lo"])
                highs.append(d2["gap_hi"])
        if labels:
            xx = np.arange(len(labels))
            ax.bar(xx, values, 0.55, yerr=[np.subtract(values, lows), np.subtract(highs, values)], capsize=4)
            ax.axhline(0, color="k", linewidth=0.8)
            ax.set_xticks(xx)
            ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
            ax.set_ylabel("size-standardised MAE gap (logD)")
            ax.set_title("structural groups: failure gap on Test")
            ax.grid(alpha=0.3, axis="y")
            fig.tight_layout()
            path = C.FIGURES / "groups.png"
            fig.savefig(path, dpi=140)
            plt.close(fig)
            written.append(C.track_rel(path))
        else:
            plt.close(fig)

    # 4. source interventions
    checkpoints = interventions.get("checkpoints", {}) if interventions else {}
    if checkpoints:
        fig, axes = plt.subplots(1, len(checkpoints), figsize=(6.0 * len(checkpoints), 4.0), squeeze=False)
        for ax, (tag, entry) in zip(axes[0], checkpoints.items()):
            variants = ["normal_rmse", "reuse_all_rmse", "fixed_all_rmse"]
            labels = ["NORMAL", "REUSE_ALL", "FIXED_ALL"]
            splits = list(entry["splits"])[:2]
            xx = np.arange(len(variants))
            for offset, split in enumerate(splits):
                payload = entry["splits"][split]
                ax.bar(xx + (offset - 0.5) * 0.36, [payload.get(v, np.nan) for v in variants], 0.36, label=split)
            ax.set_xticks(xx)
            ax.set_xticklabels(labels)
            ax.set_title(f"D2 {tag}: layer-2 source replacement")
            ax.set_ylabel("RMSE (logD)")
            ax.grid(alpha=0.3, axis="y")
            ax.legend(fontsize=8)
        fig.tight_layout()
        path = C.FIGURES / "interventions.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        written.append(C.track_rel(path))
    return written


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------


def write_tables(
    tables: Mapping[str, Any],
    comparisons: Mapping[str, Any],
    judgements: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
    groups: Mapping[str, Any],
    interventions: Mapping[str, Any],
) -> list[str]:
    parts = [
        "# 主表（每臂 best-Val / last / 末窗；原始 logD 单位）",
        "",
        main_table(tables),
        "",
        "# 主比较（control → candidate）",
        "",
        comparison_table(comparisons, judgements.get("pairs", {})),
        "",
        f"臂级判定：**{judgements.get('arm_level', {}).get('label')}**",
        "",
        "# 配对 scaffold 簇扰动区间",
        "",
        bootstrap_table(bootstrap),
        "",
        "# 结构群体",
        "",
        group_table(groups) if groups else "（未计算）",
        "",
        "# 冻结来源干预",
        "",
        intervention_table(interventions) if interventions else "（未做）",
        "",
    ]
    text = "\n".join(parts)
    path = C.TABLES / "TABLES.md"
    path.write_text(text + "\n", encoding="utf-8")
    return [C.track_rel(path)]
