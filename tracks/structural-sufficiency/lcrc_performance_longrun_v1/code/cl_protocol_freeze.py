"""Pre-register the Sprint-04 performance round before the first GPU job.

Writes ``analysis/protocol_freeze.json``.  It pins:

* the three bottleneck hypotheses and the single-change arms;
* the anchor datasets and the full conditional plan P1-P5;
* the R3 recipe and the HIV short protocol;
* the checkpoint, Test and selection rules;
* the historical BASE reference values and their source files;
* every training-code hash and the committed chem8 artefact.

It refuses to run without a passing P0 test log.  The freeze itself is committed
after the code commit; each run records the deployed commit and this file's
sha256.

Usage
-----
    uv run python code/cl_protocol_freeze.py --p0-report analysis/p0_tests.txt
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Sequence

import cl_common as C  # noqa: I001
import cl_chem8 as CH


def _read_benchmark() -> dict[str, dict[str, Any]]:
    path = C.V2 / "analysis/benchmark_table.csv"
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    out = {}
    for row in rows:
        out[row["dataset"]] = {
            "metric": row["primary_metric"],
            "train": float(row["train"]),
            "valid": float(row["valid"]),
            "test": float(row["test"]),
            "best_epoch": int(row["best_epoch"]),
            "parameters": int(row["parameters"]),
            "gpu_hours": float(row["gpu_hours"]),
            "source": C.track_rel(path),
            "source_sha256": C.sha256_file(path),
            "note": "Sprint-02 BASE, seed 0, R3 recipe, 360 epochs",
        }
    return out


def _read_lipo_repeats() -> dict[str, dict[str, Any]]:
    path = C.TASK03 / "analysis/lipo_repeat_comparison.csv"
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not row["arm"].startswith("LCRC-A"):
            continue
        repeat = str(row["repeat"])
        out[repeat] = {
            "metric": "rmse",
            "train": float(row["train_rmse"]),
            "valid": float(row["valid_rmse"]),
            "test": float(row["test_rmse"]),
            "best_epoch": int(row["best_epoch"]) if row["best_epoch"] else None,
            "gpu_seconds": float(row["gpu_seconds"]) if row["gpu_seconds"] else None,
            "source": C.track_rel(path),
            "source_sha256": C.sha256_file(path),
            "note": "Sprint-01/03 BASE, seed 0, R3 recipe, 360 epochs",
        }
    return out


def _read_matched_baseline() -> dict[str, dict[str, Any]]:
    path = C.V2 / "analysis/matched_baseline.csv"
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        row["dataset"]: {
            "xgb_valid": float(row["xgb_valid"]),
            "xgb_test": float(row["xgb_test"]),
            "same_test_rows": row["same_test_rows"].strip().lower() == "true",
            "source": C.track_rel(path),
            "source_sha256": C.sha256_file(path),
            "hyperparameters": "frozen ECFP4 (2048) + 217 RDKit descriptors + XGBoost; no tuning",
        }
        for row in rows
    }


def build(p0_report: Path) -> dict[str, Any]:
    if not p0_report.exists():
        raise SystemExit(f"P0 test report {p0_report} is missing; run the P0 tests first")
    report = p0_report.read_text(encoding="utf-8")
    if "passed" not in report or "failed" in report.splitlines()[-1].lower():
        raise SystemExit(f"P0 report {p0_report} does not look like a passing run")
    if not C.CHEM8_MANIFEST.exists():
        raise SystemExit("chem8 manifest is missing")
    import ta_data as TD

    repeats = {str(s): TD.lipo_repeat_split(s) for s in (122, 123, 124)}
    payload: dict[str, Any] = {
        "round_id": C.ROUND_ID,
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "git_dirty_tracked": C.git_dirty(),
        "git_commit_note": (
            "HEAD when this file was written (the code commit). The commit that contains this file "
            "is the one deployed; each run records it as git_commit."
        ),
        "code_hashes": C.code_hashes(),
        "p0": {
            "report": C.track_rel(p0_report),
            "sha256": C.sha256_file(p0_report),
            "tail": "\n".join(report.strip().splitlines()[-3:]),
            "gate": "all added modules pass before any GPU job",
        },
        "chem8": {
            "manifest": C.track_rel(C.CHEM8_MANIFEST),
            "sha256": C.sha256_file(C.CHEM8_MANIFEST),
            "channels": list(CH.CHANNEL_NAMES),
            "definition": (
                "label-free RDKit CIP R/S/unassigned-tetra, Lipinski HBD/HBA atom roles, and "
                "SP/SP2/SP3 hybridisation appended to the frozen lipo_v1 atom features"
            ),
            "artifacts": {
                d: {
                    "path": C.track_rel(CH.artifact_path(d)),
                    "sha256": C.sha256_file(CH.artifact_path(d)),
                    "n_molecules": int(C.read_json(C.CHEM8_MANIFEST)["datasets"][d]["n_molecules"]),
                    "n_fallback": int(C.read_json(C.CHEM8_MANIFEST)["datasets"][d]["n_fallback"]),
                }
                for d in C.V2C.DATASET_ORDER
            },
        },
        "hypotheses": {
            "A_local_chemistry": "LCRC-A's local atom input omits deterministic chemistry; CHEMLOCAL8 adds it.",
            "B_global_statistics": "rho compresses the h distribution to two moments; RHOSHAPE adds max and MAD.",
            "C_readout": "a single additive scalar contribution is too simple; MULTICONTRIB8 adds 8 latent additive channels.",
            "null": "none of these is the main bottleneck; report NO_LCRC_B_SIGNAL_THIS_ROUND.",
        },
        "arms": {
            "chem8": {
                "single_change": "widen atom_encoder.lins.0 from 51 to 59; new columns zero",
                "parameters_default": 248130,
            },
            "rhoshape": {
                "single_change": "rho input + [max_v h_v, mean_v |h_v - mean h|] (128 dims); new columns zero",
                "parameters_default": 255298,
            },
            "multicontrib": {
                "single_change": "Q: 64->64->8 graph sum, delta=MLP([Q_graph, log1p(n)]); delta last layer zero",
                "parameters_default": 252139,
            },
            "combo_chem_contrib": {"single_change": "CHEMLOCAL8 + MULTICONTRIB8", "parameters_default": 253163},
            "combo_chem_rho": {"single_change": "CHEMLOCAL8 + RHOSHAPE", "parameters_default": 256322},
            "combo_rho_contrib": {"single_change": "RHOSHAPE + MULTICONTRIB8", "parameters_default": 260331},
            "base": {"single_change": "none; frozen LCRC-A", "parameters_default": 247106},
        },
        "recipes": {
            "default": {
                "optimizer": C.RECIPE["optimizer"],
                "peak_lr": C.RECIPE["peak_lr"],
                "min_lr": C.RECIPE["min_lr"],
                "weight_decay": C.RECIPE["weight_decay"],
                "batch_size": C.RECIPE["batch_size"],
                "dropout": C.RECIPE["dropout"],
                "gradient_clip": C.RECIPE["gradient_clip"],
                "epochs": C.RECIPE["epochs"],
                "scheduler": C.RECIPE["scheduler"],
                "early_stopping": False,
            },
            "hiv_short": {
                "epochs": C.HIV_RECIPE["epochs"],
                "minimum_epochs": C.HIV_RECIPE["minimum_epochs"],
                "early_stop_patience": C.HIV_RECIPE["early_stop_patience"],
                "scheduler_horizon_epochs": C.HIV_RECIPE["scheduler_horizon_epochs"],
                "applies_to": ["HIV-BASE-SHORT", "HIV-WINNER-SHORT"],
                "loss": "unweighted BCEWithLogitsLoss",
            },
        },
        "checkpoint_rule": {
            "regression": "lowest Validation RMSE, strict improvement, ties keep the earlier epoch",
            "binary": "highest Validation ROC-AUC, tie -> lower Validation BCE, then earlier epoch",
            "multitask": "highest Validation macro ROC-AUC, tie -> lower masked BCE, then earlier epoch",
            "restore_before_report": True,
            "recheck_tolerance": "analytic metric resolution, at least 1e-5",
        },
        "test_policy": {
            "mode": "DEVELOPMENT_TEST",
            "read": "once per run, after the best-Validation checkpoint is frozen",
            "selection": "Test may inform the next stage's go/no-go, never the checkpoint/epoch/HP/width",
            "checkpoint_selection_uses_test": False,
            "ledger": "analysis/test_access_ledger.json (append-only, one entry per run)",
        },
        "anchors": {
            "Lipophilicity": "random122 (published repeat 0)",
            "ESOL": "random122 (published repeat 0)",
            "BACE": "scaffold122 (published repeat 0)",
            "seed": 0,
        },
        "splits": {
            "Lipophilicity_repeats": {
                str(s): {
                    "file": repeats[str(s)]["file"],
                    "sha256": repeats[str(s)]["sha256"],
                    "rows_sha256": repeats[str(s)]["rows_sha256"],
                    "sizes": repeats[str(s)]["sizes"],
                }
                for s in (122, 123, 124)
            },
            "transfer": {
                "FreeSolv": "random122",
                "BBBP": "scaffold122",
                "Tox21": "random122",
                "ogbg-molhiv": "official OGB scaffold split",
            },
        },
        "base_reference": {
            "anchors": {d: _read_benchmark()[d] for d in ("Lipophilicity", "ESOL", "BACE")},
            "lipophilicity_repeats": _read_lipo_repeats(),
            "transfer": {d: _read_benchmark()[d] for d in ("FreeSolv", "BBBP", "Tox21", "ogbg-molhiv")},
            "matched_xgboost": _read_matched_baseline(),
        },
        "plan": {
            "P1": {
                "question": "Do the three single changes move the three anchors?",
                "runs": [r for r in C.PLANNED_RUNS if r["stage"] == "P1"],
                "n_gpu_runs": 9,
                "selection_rule": (
                    "STRONG_SIGNAL: >=2 anchors improve both Validation and Test, or one large gain with no "
                    "catastrophic regression elsewhere; WEAK_SIGNAL: one anchor or mixed Valid/Test; "
                    "NO_SIGNAL: broad regression or Train-only gain. Promote at most two single arms."
                ),
            },
            "P2": {
                "question": "If two directions are positive, does one combination add value?",
                "max_gpu_runs": 3,
                "rule": (
                    "Construct exactly one combo, preferring the strongest representation candidate + the "
                    "strongest readout candidate. Do not build the triple unless all three singles are clearly "
                    "positive. If only one arm is positive, skip P2 and promote that arm."
                ),
            },
            "P3": {
                "question": "Is a winner gain still visible after matching parameter capacity?",
                "runs": [r for r in C.PLANNED_RUNS if r["stage"] == "P3"],
                "n_gpu_runs": 4,
                "scale_m": dict(C.SCALE_M_CONFIG),
                "constraint": "BASE-SCALE-M and WINNER-SCALE-M must both be 320k-450k parameters; only local_hidden may be nudged",
            },
            "P4": {
                "question": "Does the frozen winner transfer across tasks?",
                "runs": [r for r in C.PLANNED_RUNS if r["stage"] == "P4"],
                "n_gpu_runs": 7,
                "hiv_note": "HIV-BASE-SHORT and HIV-WINNER-SHORT must both run; no 360-epoch HIV comparison",
            },
            "P5": {
                "question": "Are the gains stable across seeds?",
                "max_gpu_runs": 12,
                "rule": "Up to three tasks; winner seeds 1,2 and matched BASE seeds 1,2 where absent; do not fill the budget for its own sake.",
            },
        },
        "budget": {
            "wall_time_hours": 8.0,
            "hard_cap_a100_gpu_hours": 15.0,
            "planned_upper_bound_gpu_hours": 15.0,
            "reserve_for_sync_analysis_report_git": True,
        },
        "decision_rules": {
            "A_chem": "CHEMLOCAL8 wins -> recommend LCRC-Chem; next round local chemistry, not interaction",
            "B_stats": "RHOSHAPE wins -> recommend LCRC-Stats; next round explicit set statistics",
            "C_composition": "MULTICONTRIB8 wins -> recommend LCRC-Composition; next round multiple latent contribution axes",
            "D_combo": "COMBO clearly best -> name LCRC-B and decompose the two contributions",
            "E_null": "no stable gain -> NO_LCRC_B_SIGNAL_THIS_ROUND and stop these three directions",
        },
        "non_goals": [
            "no Graph Transformer / self-attention / pair attention",
            "no new message-passing layer",
            "no SSL / prototype dictionary / DIST / PATH / G2L",
            "no ZINC architecture search",
            "no large hyper-parameter search",
        ],
    }
    C.write_json(C.PROTOCOL_FREEZE, payload)
    C.log(f"wrote {C.track_rel(C.PROTOCOL_FREEZE)}")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="freeze the Sprint-04 protocol")
    parser.add_argument("--p0-report", default=str(C.ANALYSIS / "p0_tests.txt"))
    args = parser.parse_args(argv)
    build(Path(args.p0_report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
