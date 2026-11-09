"""Post-score comparison tables and decision-band classification for BOC v1.

Reads only frozen artifacts: Test scores produced after the prediction freeze,
Validation curves, audits, profiling and the Valid-only zeroing diagnostic.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

CODE = Path(__file__).resolve().parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import boc_common as C  # noqa: E402


def band(value: float) -> str:
    if value >= C.DECISION_BANDS["strong"]:
        return "strong"
    if value >= C.DECISION_BANDS["meaningful"]:
        return "meaningful"
    if value >= C.DECISION_BANDS["small"]:
        return "small"
    return "no_evidence"


def _curve_stats(run: Mapping[str, Any]) -> dict[str, Any]:
    path = C.REPO / str(run.get("curve", ""))
    if not path.is_file():
        return {}
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    if not rows:
        return {}
    tail = rows[-5:]

    def mean(key: str) -> float | None:
        vals = [float(r[key]) for r in tail if r.get(key) not in (None, "", "nan")]
        return float(np.mean(vals)) if vals else None

    def mx(key: str) -> float | None:
        vals = [float(r[key]) for r in rows if r.get(key) not in (None, "", "nan")]
        return float(max(vals)) if vals else None

    return {
        "last5_mean": {k: mean(k) for k in (
            "grad_l2_bond_channel",
            "grad_l2_ring_channel",
            "grad_l2_main_fusion",
            "grad_l2_w2_trunk",
            "h0_rms",
            "m_rms",
            "r_rms",
            "z_b_rms",
            "fusion_input_rms",
            "h_rms",
            "ring_nonzero",
        )},
        "max": {k: mx(k) for k in ("grad_l2_ring_channel", "r_rms", "ring_nonzero")},
        "final": {k: rows[-1].get(k) for k in ("h0_rms", "m_rms", "r_rms", "z_b_rms", "fusion_input_rms", "h_rms")},
    }


def _score_table(scores: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for label, m in scores["models"].items():
        rows.append(
            {
                "label": label,
                "model": m["model"],
                "epoch": m["epoch"],
                "params": None,
                "valid_ap": m["valid_ap"],
                "test_ap_official": m["test_ap_official"],
                "test_ap_independent": m["test_ap_independent"],
                "delta_official_independent": m["test_ap_abs_delta"],
                "test_bce": m["test_bce"],
                "valid_test_delta": m["valid_test_ap_delta"],
                "n_tasks_scored": m["n_tasks_scored"],
            }
        )
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", default=str(C.ANALYSIS / "test_scores.json"))
    parser.add_argument("--selection", default=str(C.ANALYSIS / "selected_checkpoint.json"))
    parser.add_argument("--profile", default=str(C.ANALYSIS / "profile.json"))
    parser.add_argument("--zeroing", default=str(C.ANALYSIS / "zeroing_sensitivity.json"))
    parser.add_argument("--out-json", default=str(C.ANALYSIS / "comparison.json"))
    parser.add_argument("--out-md", default=str(C.ANALYSIS / "report_tables.md"))
    args = parser.parse_args(argv)
    scores = C.read_json(Path(args.scores))
    selection = C.read_json(Path(args.selection))
    profile = C.read_json(Path(args.profile)) if Path(args.profile).is_file() else {}
    zeroing = C.read_json(Path(args.zeroing)) if Path(args.zeroing).is_file() else {}
    audits = {
        name: C.read_json(C.ANALYSIS / name)
        for name in ("structure_audit.json", "model_audit.json")
        if (C.ANALYSIS / name).is_file()
    }

    runs: dict[str, Any] = {}
    for arm in ("FULL", "BOND"):
        path = C.RUNS / arm / "run.json"
        runs[arm] = C.read_json(path) if path.is_file() else {}

    table = _score_table(scores)
    by_label = {r["label"]: r for r in table}
    ref = {
        "W2": selection["references"]["W2"],
        "RC-REAL": selection["references"]["RC-REAL"],
    }
    for arm in ("FULL", "BOND"):
        run = runs.get(arm, {})
        for label in (f"BOC_{arm}_best_valid", f"BOC_{arm}_epoch40"):
            if label in by_label:
                by_label[label]["params"] = int(run.get("parameters", selection["models"][arm]["parameters"]))
                by_label[label]["gpu_hours"] = float(run.get("gpu_hours", float("nan")))
                by_label[label]["gpu_peak_reserved_bytes"] = int(run.get("gpu_peak_reserved_bytes", 0))
                by_label[label]["epoch_seconds_mean"] = float(run.get("epoch_seconds_mean", float("nan")))

    primary = by_label.get("BOC_FULL_best_valid", {})
    primary_ap = primary.get("test_ap_official")
    decision = {
        "primary_comparison": "BOC-FULL best-Valid vs W2 best-Valid (official Test AP, same pipeline)",
        "primary_test_ap": primary_ap,
        "band": band(float(primary_ap)) if primary_ap is not None else None,
        "bands": dict(C.DECISION_BANDS),
        "vs_W2": primary.get("test_ap_official", float("nan")) - ref["W2"]["test_ap_best_brief"]
        if primary_ap is not None
        else None,
        "vs_RC_REAL": primary.get("test_ap_official", float("nan")) - ref["RC-REAL"]["test_ap_best"]
        if primary_ap is not None
        else None,
    }
    if primary_ap is not None:
        decision["interpretation"] = {
            "strong": ">=0.270: strong evidence",
            "meaningful": "0.265-0.270: meaningful improvement",
            "small": "0.258-0.265: small improvement",
            "no_evidence": "<=0.258: no evidence",
        }[decision["band"]]

    out: dict[str, Any] = {
        "schema": C.SCHEMA_COMPARISON,
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "test_role": C.TEST_ROLE,
        "decision": decision,
        "table": table,
        "references": ref,
        "support_strata": scores.get("support_strata", {}),
        "label_stats": scores.get("label_stats", {}),
        "valid_test_gap": {
            label: row["valid_test_delta"] for label, row in by_label.items() if row.get("valid_test_delta") is not None
        },
        "curves": {arm: _curve_stats(run) for arm, run in runs.items()},
        "zeroing_valid_only": zeroing.get("arms", {}),
        "profile": profile.get("arms", {}),
        "audits": {
            name: {
                "passed": audit.get("passed"),
                "n_checks": audit.get("n_checks"),
                "n_passed": audit.get("n_passed"),
                "sha256": C.sha256_file(C.ANALYSIS / name),
            }
            for name, audit in audits.items()
        },
        "instance_counts": {
            "physical_bonds": 12_309_186,
            "bond_neighbour_entries": 34_369_144,
            "bond_pair_entries": 34_610_219,
            "ring_systems": 1_100_845,
            "ring_atom_memberships": 7_343_607,
            "ring_pairs": 23_424_393,
            "external_substituents": 2_386_191,
            "cache_bytes": 1_645_218_822,
            "cache_seconds": 219.9,
        },
        "independent_vs_official": {
            row["label"]: row["delta_official_independent"] for row in table
        },
        "test_exposure": scores.get("test_exposure_history"),
        "descriptive_only": True,
    }
    C.write_json(args.out_json, out)

    lines = [
        "# LCRC-BOC v1 — comparison tables",
        "",
        f"Test role: `{C.TEST_ROLE}`.  Decision bands are engineering bands, not significance tests.",
        "",
        "## Primary table",
        "",
        "| model | epoch | params | best Valid AP | Test AP (official) | independent AP | Valid→Test | GPU h |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    ref_rows = [
        ("W2 best-Valid", ref["W2"]["best_epoch"], ref["W2"]["parameters"], ref["W2"]["best_valid_ap"], ref["W2"]["test_ap_best_brief"], ref["W2"]["gpu_hours"]),
        ("W2 @40", 40, ref["W2"]["parameters"], ref["W2"]["valid_ap_at_40"], C.W2_TEST_AP_40_BRIEF, ref["W2"]["gpu_hours"]),
        ("RC-REAL best-Valid", ref["RC-REAL"]["best_epoch"], ref["RC-REAL"]["parameters"], ref["RC-REAL"]["best_valid_ap"], ref["RC-REAL"]["test_ap_best"], ref["RC-REAL"]["gpu_hours"]),
        ("RC-REAL @40", 40, ref["RC-REAL"]["parameters"], ref["RC-REAL"]["valid_ap_at_40"], ref["RC-REAL"]["test_ap_40"], ref["RC-REAL"]["gpu_hours"]),
    ]
    for name, epoch, params, valid, test, gpu in ref_rows:
        lines.append(f"| {name} | {epoch} | {params:,} | {valid:.6f} | {test:.6f} | — | — | {gpu:.4f} |")
    for arm in ("FULL", "BOND"):
        for label in (f"BOC_{arm}_best_valid", f"BOC_{arm}_epoch40"):
            row = by_label.get(label)
            if not row:
                continue
            lines.append(
                f"| {label} | {row['epoch']} | {int(row['params']):,} | {row['valid_ap']:.6f} | "
                f"{row['test_ap_official']:.6f} | {row['test_ap_independent']:.6f} | "
                f"{row['valid_test_delta']:+.6f} | {row['gpu_hours']:.4f} |"
            )
    lines += ["", f"Primary band: **{decision.get('band')}** ({decision.get('interpretation', 'n/a')}).", ""]
    lines += ["## Support strata (Test positives per task)", ""]
    for key, val in scores.get("support_strata", {}).items():
        lines.append(
            f"* `{key}`: n={val.get('n_tasks')} mean Δ={val.get('mean_delta')} "
            f"median Δ={val.get('median_delta')} improved={val.get('improved')}"
        )
    lines += ["", "## Channel norms / gradients (last 5 epochs mean)", ""]
    for arm, stats in out["curves"].items():
        if not stats:
            continue
        m = stats["last5_mean"]
        lines.append(
            f"* `{arm}`: h0 {m['h0_rms']:.4f}, m {m['m_rms']:.4f}, r {m['r_rms']:.4f}, "
            f"h {m['h_rms']:.4f}; grad L2 bond {m['grad_l2_bond_channel']:.4f}, "
            f"ring {m['grad_l2_ring_channel']:.4f}, fusion {m['grad_l2_main_fusion']:.4f}, "
            f"W2 {m['grad_l2_w2_trunk']:.4f}"
        )
    lines += ["", "## Zeroing sensitivity (official Validation only)", ""]
    for arm, block in out["zeroing_valid_only"].items():
        modes = block["modes"]
        lines.append(
            f"* `{arm}`: both {modes['both_channels']['valid_ap']:.6f}, "
            f"ring-zero {modes['ring_zero']['valid_ap']:.6f} "
            f"({modes['ring_zero']['delta_vs_both_channels']:+.6f}), "
            f"bond-zero {modes['bond_zero']['valid_ap']:.6f} "
            f"({modes['bond_zero']['delta_vs_both_channels']:+.6f}), "
            f"both-zero {modes['both_zero']['valid_ap']:.6f}"
        )
    lines += ["", "## Audits", ""]
    for name, block in out["audits"].items():
        lines.append(f"* `{name}`: passed={block['passed']} ({block['n_passed']}/{block['n_checks']}), sha `{block['sha256'][:12]}`")
    Path(args.out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
