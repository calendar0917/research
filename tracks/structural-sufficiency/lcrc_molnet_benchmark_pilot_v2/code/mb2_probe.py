"""P3: FreeSolv / BACE Validation-only recipe probes (no Test access at all).

Four pre-registered candidates (``C.PROBE_CANDIDATES``), each a *single* parameter
change against R3:

    FreeSolv  dropout 0.10 -> 0.00
    FreeSolv  dropout 0.10 -> 0.20
    BACE      dropout 0.10 -> 0.20
    BACE      weight decay 1e-3 -> 1e-2

The FreeSolv and BACE Test sets were already read in Sprint 01, so this module
*never* reads a Test label: it only collects the Train/Validation records that the
probe runs wrote.  It refuses a probe run whose ``train.json`` claims Test access,
and it refuses to write a Test metric for a probe.

BASE rows come read-only from Sprint 01 (``lcrc_molnet_benchmark_pilot_v1``), which
used the identical recipe, split and seed.

Usage
-----
    uv run python .../code/mb2_probe.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Sequence

import mb2_common as C

PROBE_COLUMNS = [
    "dataset",
    "config",
    "kind",
    "dropout",
    "weight_decay",
    "overrides_vs_R3",
    "epochs_run",
    "best_epoch",
    "valid_primary_metric",
    "valid_primary_value",
    "train_primary_value",
    "train_valid_gap",
    "valid_secondary_name",
    "valid_secondary_value",
    "final_epoch_valid_primary_value",
    "best_minus_final_valid",
    "delta_valid_vs_base",
    "delta_relative_vs_base",
    "train_wall_seconds",
    "gpu_hours",
    "peak_memory_allocated_bytes",
    "checkpoint_sha256",
    "artifact",
    "test_labels_read",
    "test_metric_reported",
    "verdict",
    "note",
]


def _primary(task: str, metrics: dict[str, Any]) -> float:
    return float(
        metrics["rmse"] if task == "regression" else metrics["roc_auc"]
    )


def _secondary(task: str, metrics: dict[str, Any]) -> tuple[str, float | None]:
    if task == "regression":
        return "mae", float(metrics["mae"])
    return "pr_auc", float(metrics["pr_auc"])


def _from_train_json(path: Path, *, dataset: str, config: str, kind: str) -> dict[str, Any]:
    payload = C.read_json(path)
    if payload.get("dataset") != dataset:
        raise SystemExit(f"{C.track_rel(path)} is for {payload.get('dataset')}, not {dataset}")
    if payload["data"].get("test_labels_read") or payload["data"].get("test_cache_built"):
        raise SystemExit(f"{C.track_rel(path)} reports Test access; a probe must be Validation-only")
    task = payload["task"]
    valid = payload["metrics"]["valid"]
    train = payload["metrics"]["train"]
    ok = "valid_rmse_raw" if task == "regression" else "valid_roc_auc"
    curve = Path(C.REPO / payload["checkpoint"]["curve_path"])
    final_value = None
    if curve.exists():
        rows = list(csv.DictReader(curve.open(encoding="utf-8")))
        if rows:
            final_value = float(rows[-1][ok])
    sec_name, sec_value = _secondary(task, valid)
    valid_primary = _primary(task, valid)
    train_primary = _primary(task, train)
    return {
        "dataset": dataset,
        "config": config,
        "kind": kind,
        "dropout": float(payload["recipe"]["dropout"]),
        "weight_decay": float(payload["recipe"]["weight_decay"]),
        "overrides_vs_R3": payload["recipe"].get("overrides_vs_R3") or {},
        "epochs_run": int(payload["recipe"]["epochs_run"]),
        "best_epoch": int(payload["checkpoint"]["best_epoch"]),
        "valid_primary_metric": payload["checkpoint"]["best_metric_name"],
        "valid_primary_value": valid_primary,
        "train_primary_value": train_primary,
        "train_valid_gap": (valid_primary - train_primary) if task == "regression" else (train_primary - valid_primary),
        "valid_secondary_name": sec_name,
        "valid_secondary_value": sec_value,
        "final_epoch_valid_primary_value": final_value,
        "best_minus_final_valid": (final_value - valid_primary) if final_value is not None else None,
        "train_wall_seconds": float(payload["timing"]["train_wall_seconds"]),
        "gpu_hours": float(payload["timing"]["train_wall_seconds"]) / 3600.0,
        "peak_memory_allocated_bytes": int(payload["gpu"]["peak_memory_allocated_bytes"]),
        "checkpoint_sha256": payload["checkpoint"]["sha256"],
        "artifact": C.track_rel(path),
        "test_labels_read": False,
        "test_metric_reported": False,
        "note": "",
    }


def build() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for dataset in ("FreeSolv", "BACE"):
        base_path = C.OLD_V1_RUNS / f"{dataset}_s0/train.json"
        if not base_path.exists():
            missing.append(f"BASE {dataset}: {C.track_rel(base_path)}")
            continue
        rows.append(_from_train_json(base_path, dataset=dataset, config="R3_BASE", kind="frozen_benchmark_base"))
    for name in C.PROBE_ORDER:
        spec = C.PROBE_CANDIDATES[name]
        path = C.RUNS / f"probe_{name}_s{C.TRAIN_SEED}/train.json"
        if not path.exists():
            missing.append(f"PROBE {name}: {C.track_rel(path)}")
            continue
        payload = C.read_json(path)
        if not payload.get("probe", {}).get("probe"):
            raise SystemExit(f"{C.track_rel(path)} is not marked as a probe run")
        rows.append(_from_train_json(path, dataset=spec["dataset"], config=name, kind="validation_only_probe"))

    base_by_dataset = {r["dataset"]: r for r in rows if r["kind"] == "frozen_benchmark_base"}
    for row in rows:
        base = base_by_dataset.get(row["dataset"])
        if base is None or row["kind"] == "frozen_benchmark_base":
            row["delta_valid_vs_base"] = 0.0 if base is not None else None
            row["delta_relative_vs_base"] = 0.0 if base is not None else None
            continue
        delta = row["valid_primary_value"] - base["valid_primary_value"]
        row["delta_valid_vs_base"] = delta
        row["delta_relative_vs_base"] = delta / base["valid_primary_value"] if base["valid_primary_value"] else None

    # a deliberately conservative verdict language, fixed before the numbers exist
    for row in rows:
        if row["kind"] == "frozen_benchmark_base":
            row["verdict"] = "base"
            continue
        delta = row["delta_valid_vs_base"]
        better = delta < 0 if row["valid_primary_metric"] == "valid_rmse" else delta > 0
        magnitude = abs(row["delta_relative_vs_base"] or 0.0)
        if not better:
            row["verdict"] = "no_validation_improvement__does_not_solve_the_gap"
            row["note"] = (
                "the single-parameter change did not improve Validation; this says the finite regularisation "
                "probe did not help, not that the model is fundamentally mis-specified"
            )
        elif magnitude >= 0.05:
            row["verdict"] = "large_validation_gain__candidate_for_next_round"
            row["note"] = (
                "Validation improved by >=5% relative on a single seed; treat as a candidate recipe for a "
                "multi-seed / new-split confirmation, never as proven generalisation"
            )
        else:
            row["verdict"] = "small_validation_gain__within_single_seed_noise"
            row["note"] = (
                "Validation moved less than 5% relative on one seed; the effect is not separable from seed "
                "and split noise at this sample size"
            )

    manifest = {
        "schema": "lcrc-molnet-v2-probe-manifest-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "candidates": {
            name: {
                "dataset": C.PROBE_CANDIDATES[name]["dataset"],
                "overrides": dict(C.PROBE_CANDIDATES[name]["overrides"]),
                "hypothesis": C.PROBE_CANDIDATES[name]["hypothesis"],
            }
            for name in C.PROBE_ORDER
        },
        "rows": len(rows),
        "missing": missing,
        "test_access": {
            "test_labels_read": False,
            "test_inference_run": False,
            "test_metric_reported": False,
            "why": (
                "FreeSolv and BACE Test labels were already read in Sprint 01; these probes are "
                "Validation-only diagnostics and are excluded from the seven-dataset main table"
            ),
        },
        "selection_rule_honoured": "the four candidates were fixed before any run and none was added or dropped",
        "seed": C.TRAIN_SEED,
        "epochs_per_candidate": int(C.RECIPE["epochs"]),
    }
    return rows, manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="assemble the Validation-only probe table")
    parser.add_argument("--out", default=str(C.ANALYSIS / "validation_tuning_probe.csv"))
    args = parser.parse_args(argv)
    rows, manifest = build()
    path = Path(args.out)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PROBE_COLUMNS)
        writer.writeheader()
        for row in rows:
            out = {k: row.get(k) for k in PROBE_COLUMNS}
            if isinstance(out["overrides_vs_R3"], dict):
                out["overrides_vs_R3"] = C.stable_json(out["overrides_vs_R3"])
            writer.writerow(out)
    C.write_json(C.ANALYSIS / "probe_manifest.json", manifest)
    C.log(f"wrote {C.track_rel(path)} ({len(rows)} rows); missing: {manifest['missing']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
