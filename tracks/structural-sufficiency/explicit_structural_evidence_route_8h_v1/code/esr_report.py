"""Render the report tables from the frozen analysis JSON/CSV files.

    python esr_report.py --analysis analysis --out analysis/report_tables.md

Nothing here computes science: every number is read from an artifact written by
``esr_collect.py`` / ``esr_mechanism.py`` (which in turn read ``runs/*/run.json``).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import esr_common as C

PRIMARY = {"LIPO": "RMSE", "ESOL": "RMSE", "TOX21": "macro ROC-AUC", "PCBA": "macro AP"}


def _md_table(rows: Sequence[Sequence[Any]], header: Sequence[str]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(_fmt(v) for v in row) + " |")
    return "\n".join(out)


def _fmt(value: Any) -> str:
    if value is None or value == "":
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def training_table(rows: list[dict[str, str]]) -> str:
    body = []
    for r in rows:
        if r.get("source") != "this_round":
            continue
        body.append(
            [
                r["dataset"],
                r["arm"],
                int(float(r["parameters"])),
                int(float(r["epochs"])),
                int(float(r["best_epoch"])),
                float(r["primary_best_valid"]),
                float(r["primary_final_epoch"]),
                round(float(r["wall_minutes"]), 1),
                round(float(r["gpu_hours"]), 3),
            ]
        )
    body.sort(key=lambda x: (x[0], -x[5] if x[0] in ("TOX21", "PCBA") else x[5]))
    return _md_table(body, ["dataset", "arm", "params", "epochs", "best ep", "best Valid", "final ep", "wall min", "GPU-h"])


def pcba_table(rows: list[dict[str, str]], decisions: Mapping[str, Any], paired: Mapping[str, Any]) -> str:
    body = []
    hist = {r["arm"]: r for r in rows if r.get("source") == "historical_reference" and r["dataset"] == "PCBA"}
    for r in rows:
        if r["dataset"] != "PCBA" or r.get("source") != "this_round":
            continue
        arm = r["arm"]
        p = next(
            (v for k, v in paired.get("pcba_vs_historical", {}).get("paired_vs_historical", {}).get("W2_JOINT", {}).items() if k == arm),
            None,
        )
        body.append(
            [
                arm,
                int(float(r["parameters"])),
                int(float(r["epochs"])),
                int(float(r["best_epoch"])),
                float(r["primary_best_valid"]),
                float(r["best_within_25_prefix"]) if r.get("best_within_25_prefix") else None,
                float(r["primary_final_epoch"]),
                (f"{p['delta_point']:+.4f} [{p['delta_ci95'][0]:+.4f}, {p['delta_ci95'][1]:+.4f}]" if p else "—"),
                (f"{p['improved_tasks']}/{p['degraded_tasks']}" if p else "—"),
            ]
        )
    body.sort(key=lambda x: -x[4])
    for name, r in hist.items():
        body.append([f"*{name}* (historical)", r.get("parameters"), r.get("epochs"), "", r.get("primary_best_valid"), "", "", "", ""])
    table = _md_table(body, ["arm", "params", "epochs", "best ep", "best Valid AP", "best in 25-ep prefix", "final ep AP", "Δ vs W2_JOINT [95% CI]", "tasks +/−"])
    levels = decisions.get("levels") or {}
    tail = []
    if levels:
        tail.append(
            _md_table(
                [
                    ["W2 best within 25 ep (frozen)", decisions["reference_points"]["W2_best_within_25"], "—"],
                    ["W2 prefix-25 final (frozen)", decisions["reference_points"]["W2_prefix25_final"], "—"],
                    ["FIXED1 prefix-25 final (frozen)", decisions["reference_points"]["FIXED1_prefix25_final"], "—"],
                    ["best explicit in this round", levels["L1"]["value"], decisions["best_explicit"]["arm"]],
                    ["L1 threshold (+0.005 over W2)", levels["L1"]["threshold"], "reached" if levels["L1"]["reached"] else "not reached"],
                    ["L2 threshold (within 0.003 of FIXED1)", levels["L2"]["threshold"], "reached" if levels["L2"]["reached"] else "not reached"],
                    ["L3 / L4 (0.280 / 0.290)", "—", (f"L3 {'yes' if levels['L3']['reached'] else 'no'}, L4 {'yes' if levels['L4']['reached'] else 'no'}")],
                ],
                ["PCBA reference / level", "macro AP", "status"],
            )
        )
    return table + "\n\n" + "\n\n".join(tail)


def small_table(rows: list[dict[str, str]], decisions: Mapping[str, Any], paired: Mapping[str, Any], dataset: str) -> str:
    body = []
    base = next((float(r["primary_best_valid"]) for r in rows if r["dataset"] == dataset and r["source"] == "this_round" and r["arm"] == "BASE"), None)
    for r in rows:
        if r["dataset"] != dataset or r.get("source") != "this_round":
            continue
        arm = r["arm"]
        key = f"BASE__vs__{arm}" if f"BASE__vs__{arm}" in paired.get("datasets", {}).get(dataset, {}) else f"{arm}__vs__BASE"
        p = paired.get("datasets", {}).get(dataset, {}).get(key)
        rel = None
        if base and dataset in ("LIPO", "ESOL"):
            rel = (float(r["primary_best_valid"]) - base) / base
        delta = None
        if base and dataset == "TOX21":
            delta = float(r["primary_best_valid"]) - base
        body.append(
            [
                arm,
                int(float(r["parameters"])),
                int(float(r["best_epoch"])),
                float(r["primary_best_valid"]),
                (f"{rel * 100:+.2f}%" if rel is not None else (f"{delta:+.4f}" if delta is not None else None)),
                (f"{p['delta_point']:+.4f} [{p['delta_ci95'][0]:+.4f}, {p['delta_ci95'][1]:+.4f}]" if p else "—"),
            ]
        )
    body.sort(key=lambda x: x[3], reverse=dataset == "TOX21")
    return _md_table(body, ["arm", "params", "best ep", f"best Valid {PRIMARY[dataset]}", "vs BASE", "paired Δ vs BASE [95% CI]"])


def accounting_table(diagnostics: Mapping[str, Any]) -> str:
    body = []
    for e in diagnostics.get("entries", []):
        acc = e.get("accounting") or {}
        iv = e.get("interventions") or {}
        row = [
            e["dataset"],
            e["arm"],
            ("yes" if e.get("accounting_exact") else ("no" if e.get("accounting_exact") is False else "n/a")),
            acc.get("max_abs_reconstruction_error"),
            (f"{iv['full_primary']:.4f}" if "full_primary" in iv else "—"),
        ]
        for key in ("zero_relation_tokens", "zero_local_tokens", "zero_h0", "zero_global_descriptors"):
            ab = (iv.get("ablations") or {}).get(key)
            row.append(f"{ab['primary']:.4f} ({ab['delta']:+.4f})" if ab else "—")
        body.append(row)
    return _md_table(
        body,
        ["dataset", "arm", "accounting exact", "max |recon error|", "full Valid", "relation off", "local off", "h0 = 0", "global off"],
    )


def rules_table(rules: Mapping[str, Any]) -> str:
    body = []
    for entry in rules.get("datasets", []):
        if entry.get("skipped"):
            body.append([entry["dataset"], entry["arm"], "skipped", entry["skipped"], "", ""])
            continue
        top = (entry.get("rooted_structure_rules") or [None])[0]
        body.append(
            [
                entry["dataset"],
                entry["arm"],
                entry.get("n_signatures_passing_support"),
                entry.get("n_signatures_total"),
                len(entry.get("relation_rules") or []),
                (f"{top['element']}/deg{top['degree']} ({top['n_molecules']} mol, {top['n_scaffolds']} scaffolds, mean {top['mean_contribution']:+.4f})" if top else "—"),
            ]
        )
    return _md_table(body, ["dataset", "arm", "rules kept", "signatures", "relation buckets", "strongest rooted rule"])


def efficiency_table(rows: list[dict[str, str]]) -> str:
    body = []
    for r in rows:
        body.append(
            [
                r["dataset"],
                r["arm"],
                r["parameters"],
                r.get("step_ms_p50"),
                r.get("valid_eval_seconds"),
                (int(float(r["peak_allocated_bytes"])) / 2**20 if r.get("peak_allocated_bytes") else None),
                round(float(r["wall_minutes"]), 1),
                round(float(r["gpu_hours"]), 3),
            ]
        )
    body.sort(key=lambda x: (x[0], x[5] if isinstance(x[5], float) else 0))
    return _md_table(body, ["dataset", "arm", "params", "step ms p50", "Valid eval s", "peak MiB", "wall min", "GPU-h"])


def decisions_table(dec: Mapping[str, Any]) -> str:
    body = []
    for name, entry in (dec.get("triggers") or {}).items():
        body.append([name, "yes" if entry.get("triggered") else "no", entry.get("rule", "")])
    for ds, entry in (dec.get("small_data") or {}).items():
        for arm, vals in entry.get("arms", {}).items():
            body.append(
                [
                    f"{ds} {arm}",
                    "signal" if vals.get("signal") else "no signal",
                    (f"rel {vals['relative_delta'] * 100:+.2f}%" if "relative_delta" in vals else f"Δ {vals.get('absolute_delta', 0):+.4f}"),
                ]
            )
    return _md_table(body, ["decision", "fired", "rule / value"])


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="render report tables")
    ap.add_argument("--analysis", default=str(C.ANALYSIS))
    ap.add_argument("--out", default=str(C.ANALYSIS / "report_tables.md"))
    args = ap.parse_args(argv)
    analysis = Path(args.analysis)
    rows = _csv(analysis / "main_table.csv")
    dec = json.loads((analysis / "phase_decisions.json").read_text())
    paired = json.loads((analysis / "paired_comparisons.json").read_text())
    diag = json.loads((analysis / "mechanism_diagnostics.json").read_text())
    rules = json.loads((analysis / "structure_rules.json").read_text())
    eff = _csv(analysis / "efficiency_table.csv")
    parts = [
        "## Training\n",
        training_table(rows),
        "\n## PCBA\n",
        pcba_table(rows, dec, paired),
        "\n## Lipo\n",
        small_table(rows, dec, paired, "LIPO"),
        "\n## Tox21\n",
        small_table(rows, dec, paired, "TOX21"),
        "\n## ESOL\n",
        small_table(rows, dec, paired, "ESOL"),
        "\n## Accounting and interventions\n",
        accounting_table(diag),
        "\n## Structure rules\n",
        rules_table(rules),
        "\n## Efficiency\n",
        efficiency_table(eff),
        "\n## Decisions\n",
        decisions_table(dec),
        "",
    ]
    Path(args.out).write_text("\n".join(parts), encoding="utf-8")
    C.log(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
