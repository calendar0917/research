"""Render the frozen JSON/CSV tables as Markdown for REPORT.md.

Numbers are formatted from ``analysis/tables/*`` and ``analysis/bootstrap.json``
only; nothing is recomputed or re-selected here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import dissect_common as C

TABLES = C.ANALYSIS / "tables"


def _f(x: Any, nd: int = 4) -> str:
    if x is None or x == "":
        return "—"
    return f"{float(x):.{nd}f}"


def _signed(x: Any, nd: int = 4) -> str:
    if x is None or x == "":
        return "—"
    return f"{float(x):+.{nd}f}"


def main_table(rows: Sequence[Mapping[str, Any]]) -> str:
    head = (
        "| repeat | variant | seed | params | best ep | Train RMSE | Valid RMSE | Valid MAE | Valid R² | "
        "Test RMSE | Test MAE | GPU s |\n|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    body = []
    for r in rows:
        body.append(
            f"| {r['repeat']} | {r['variant']} | {r['seed']} | {r['params_trainable']} | {r['best_epoch']} | "
            f"{_f(r['train_rmse'])} | {_f(r['valid_rmse'])} | {_f(r['valid_mae'])} | {_f(r['valid_r2'], 3)} | "
            f"{_f(r['test_rmse'])} | {_f(r['test_mae'])} | {_f(r['train_wall_seconds'], 0)} |"
        )
    return head + "\n".join(body) + "\n"


def means_table(rows: Sequence[Mapping[str, Any]]) -> str:
    head = (
        "| variant | name | seeds | params | Valid RMSE mean | SD | min | max | Valid MAE mean | Test RMSE mean | "
        "best ep (min–max) |\n|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    body = []
    for r in rows:
        if int(float(r["n_seeds"])) < 1:
            continue
        body.append(
            f"| {r['variant']} | {r['variant_name']} | {r['n_seeds']} | {r['params_trainable']} | "
            f"{_f(r['valid_rmse_mean'])} | {_f(r.get('valid_rmse_sd'))} | {_f(r.get('valid_rmse_min'))} | "
            f"{_f(r.get('valid_rmse_max'))} | {_f(r['valid_mae_mean'])} | {_f(r.get('test_rmse_mean'))} | "
            f"{r['best_epoch_min']}–{r['best_epoch_max']} |"
        )
    return head + "\n".join(body) + "\n"


def contrasts_table(rows: Sequence[Mapping[str, Any]], boot: Mapping[str, Any] | None) -> str:
    head = (
        "| contrast | n seeds | ΔValid RMSE mean | SD | per seed | ΔValid MAE mean | scaffold-bootstrap 95% | crosses 0 |\n"
        "|---|---|---|---|---|---|---|---|\n"
    )
    body = []
    for r in rows:
        if not r.get("available"):
            body.append(f"| {r['contrast']} | 0 | — | — | — | — | — | — |")
            continue
        ci = "—"
        cross = "—"
        if boot and r["contrast"] in boot:
            lo, hi = boot[r["contrast"]]["d_rmse_mean_ci95"]
            ci = f"[{lo:+.4f}, {hi:+.4f}]"
            cross = "yes" if boot[r["contrast"]]["ci_crosses_zero"] else "no"
        body.append(
            f"| {r['contrast']} | {r['n_seeds']} | {_signed(r['d_rmse_mean'])} | {_f(r.get('d_rmse_sd'))} | "
            f"{r['d_rmse_per_seed']} | {_signed(r.get('d_mae_mean'))} | {ci} | {cross} |"
        )
    return head + "\n".join(body) + "\n"


def cross_table(rows: Sequence[Mapping[str, Any]]) -> str:
    head = "| variant | repeat | seeds | Valid RMSE mean | SD | Valid MAE mean | Test RMSE mean | Test MAE mean |\n|---|---|---|---|---|---|---|---|\n"
    body = []
    for r in rows:
        for repeat, d in r["per_repeat"].items():
            body.append(
                f"| {r['variant']} | {repeat} | {d['n_seeds']} | {_f(d['valid_rmse_mean'])} | {_f(d['valid_rmse_sd'])} | "
                f"{_f(d['valid_mae_mean'])} | {_f(d['test_rmse_mean'])} | {_f(d['test_mae_mean'])} |"
            )
        body.append(
            f"| {r['variant']} | **all** | — | {_f(r['grand_valid_rmse_mean'])} | — | — | "
            f"{_f(r['grand_test_rmse_mean'])} | — |"
        )
    return head + "\n".join(body) + "\n"


def bins_table(rows: Sequence[Mapping[str, Any]], variants: Sequence[str]) -> str:
    if not rows:
        return "_no size-bin table_\n"
    head = "| atom-count bin | n Valid | " + " | ".join(f"{v} RMSE (Δ)" for v in variants) + " |\n"
    head += "|---" * (2 + len(variants)) + "|\n"
    body = []
    for r in rows:
        cells = []
        for v in variants:
            rmse = r.get(f"{v}_rmse")
            d = r.get(f"{v}_d_rmse")
            cells.append(f"{_f(rmse)} ({_signed(d)})" if rmse is not None else "—")
        lo = r["atom_count_low"]
        hi = r["atom_count_high"]
        if lo in ("", None):
            label = f"≤{int(float(hi))}"
        elif hi in ("", None):
            label = f">{int(float(lo))}"
        else:
            label = f"{int(float(lo)) + 1}–{int(float(hi))}"
        body.append(f"| {label} | {r['n_valid']} | " + " | ".join(cells) + " |")
    return head + "\n".join(body) + "\n"


def main() -> int:
    import csv

    def load(name: str) -> list[dict[str, Any]]:
        p = TABLES / name
        if not p.exists():
            return []
        with p.open(newline="") as fh:
            return list(csv.DictReader(fh))

    rows = load("main_runs.csv")
    means = load("variant_means.csv")
    boot = C.read_json(C.ANALYSIS / "bootstrap.json")["valid"]["contrasts"] if (C.ANALYSIS / "bootstrap.json").exists() else None
    contrasts = C.read_json(TABLES / "contrasts_valid.json") if (TABLES / "contrasts_valid.json").exists() else []
    cross = C.read_json(TABLES / "cross_repeat.json") if (TABLES / "cross_repeat.json").exists() else []
    bins = load("size_bins.csv")
    out = {
        "main_122": main_table([r for r in rows if int(float(r["repeat"])) == C.MAIN_REPEAT]),
        "means_122": means_table([r for r in means if int(float(r["repeat"])) == C.MAIN_REPEAT]),
        "contrasts_122": contrasts_table(contrasts, boot),
        "cross_repeat": cross_table(cross),
        "size_bins": bins_table(bins, ["A0", "B0", "B4", "B5", "C4"]),
        "main_cross": main_table([r for r in rows if int(float(r["repeat"])) != C.MAIN_REPEAT]),
    }
    text = []
    for key, body in out.items():
        text.append(f"## {key}\n\n{body}\n")
    (TABLES / "markdown.md").write_text("\n".join(text))
    C.log(f"[tables] markdown.md written ({sum(len(v) for v in out.values())} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
