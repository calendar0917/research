"""Build every LCRC-P1 analysis artefact from frozen run outputs.

Read-only with respect to the historical rounds: the only inputs are

* the three frozen ChemBench Lipophilicity splits and the shared label table,
* the ten frozen ``lcrc_performance_longrun_v1`` units (reused),
* the ten units trained in this round (fresh).

The scoring order is fixed and enforced: alignment and integrity checks run
first and abort the build on a blocking failure, then the tables are written.

Usage::

    python .../code/run_analysis.py            # full build
    python .../code/run_analysis.py --audit-only
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from p1_common import (
    ANALYSIS,
    ARMS,
    BOOTSTRAP_SEED,
    EXPECTED_SIZES,
    FROZEN_RECIPE_KEYS,
    LONGRUN,
    LONGRUN_FREEZE,
    N_BOOT,
    PART_ORDER,
    PROTOCOL_FREEZE,
    Q2_ARMS,
    ROUND,
    ROUND_ID,
    SEEDS,
    SHUFFLE_OFFSET,
    SPLIT_ORDER,
    TOL,
    TOP_FRACTION,
    analysis_code_hashes,
    git_commit,
    git_dirty,
    read_json,
    sha256_file,
    track_rel,
    utcnow,
    verify_analysis_freeze,
    write_json,
)
import p1_analysis as A
import p1_sources as SRC
import p1_stats as S


# ---------------------------------------------------------------------------
# audit assembly
# ---------------------------------------------------------------------------


def unit_source_record(unit: SRC.Unit) -> dict[str, Any]:
    train_json = unit.train_json
    return {
        "unit_key": unit.key,
        "split": unit.split,
        "arm": unit.arm,
        "seed": unit.seed,
        "origin": unit.origin,
        "run_dir": track_rel(unit.run_dir),
        "files": {
            "train.json": sha256_file(unit.run_dir / "train.json"),
            "checkpoint.pt": sha256_file(unit.run_dir / "checkpoint.pt"),
            "valid_predictions.npz": sha256_file(unit.run_dir / "valid_predictions.npz"),
            "test_predictions.npz": sha256_file(unit.run_dir / "test_predictions.npz"),
            "test_eval.json": sha256_file(unit.run_dir / "test_eval.json"),
            "curve.csv": sha256_file(unit.run_dir / "curve.csv"),
        },
        "historical_git_commit": train_json.get("git_commit"),
        "protocol_id": train_json.get("protocol_id"),
        "code_hashes": train_json.get("code_hashes"),
        "recipe": {k: train_json.get("recipe", {}).get(k) for k in FROZEN_RECIPE_KEYS}
        | {"shuffle_generator_seed": train_json.get("recipe", {}).get("shuffle_generator_seed")},
        "environment": {
            "hostname": train_json.get("hostname"),
            "torch": train_json.get("torch"),
            "torch_cuda": train_json.get("torch_cuda"),
            "cuda_device_name": train_json.get("cuda_device_name"),
            "python": train_json.get("python"),
            "device": train_json.get("device"),
            "precision": train_json.get("precision"),
        },
        "target_norm": train_json.get("data", {}).get("target_norm"),
        "cache_fingerprints": {
            "train": train_json.get("data", {}).get("train_cache_fingerprint"),
            "train_base": train_json.get("data", {}).get("train_cache_base_fingerprint"),
            "valid": train_json.get("data", {}).get("valid_cache_fingerprint"),
            "valid_base": train_json.get("data", {}).get("valid_cache_base_fingerprint"),
            "train_norm": train_json.get("data", {}).get("train_norm_fingerprint"),
        },
        "split_record_rows": train_json.get("split_record", {}).get("rows"),
        "source": train_json.get("source"),
        "best_epoch": train_json.get("checkpoint", {}).get("best_epoch"),
        "epochs_run": train_json.get("checkpoint", {}).get("epochs_run"),
        "parameters": train_json.get("model", {}).get("parameters"),
        "arm_label": train_json.get("model", {}).get("name"),
        "recorded_metrics": {"valid": train_json.get("metrics", {}).get("valid"), "test": train_json.get("metrics", {}).get("test")},
        "recomputed_metrics": unit.metrics,
        "selection_uses_test": train_json.get("checkpoint", {}).get("selection_uses_test"),
        "test_used_for_development": train_json.get("test_used_for_development"),
        "extension_zero_init": train_json.get("model", {}).get("extension_zero_init"),
        "base_copy": train_json.get("model", {}).get("base_copy"),
        "base_init_state_hash": train_json.get("model", {}).get("base_init_state_hash"),
        "init_state_hash": train_json.get("model", {}).get("init_state_hash"),
        "shared_encoder_state_hash": train_json.get("model", {}).get("shared_encoder_state_hash"),
        "timing": train_json.get("timing"),
    }


def _unique(values: Sequence[Any]) -> list[Any]:
    seen = []
    for value in values:
        if value not in seen:
            seen.append(value)
    return seen


def cross_unit_checks(units: Mapping[tuple[str, str, int], SRC.Unit]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    records = [unit_source_record(u) for u in units.values()]

    code_sets = {json.dumps(r["code_hashes"], sort_keys=True) for r in records}
    checks.append(
        {
            "check": "training_code_hashes_identical_across_all_units",
            "passed": len(code_sets) == 1,
            "detail": {"n_distinct_code_hash_sets": len(code_sets)},
        }
    )
    reference_recipe = None
    recipe_mismatch = []
    for r in records:
        recipe = dict(r["recipe"])
        recipe.pop("shuffle_generator_seed", None)
        if reference_recipe is None:
            reference_recipe = recipe
        elif recipe != reference_recipe:
            recipe_mismatch.append({"unit": r["unit_key"], "recipe": recipe})
    checks.append(
        {
            "check": "frozen_recipe_identical_across_all_units",
            "passed": not recipe_mismatch,
            "detail": {"n_mismatches": len(recipe_mismatch), "mismatches": recipe_mismatch[:3]},
        }
    )
    shuffle_bad = [
        r["unit_key"]
        for r in records
        if int(r["recipe"].get("shuffle_generator_seed", -1)) != int(r["seed"]) + SHUFFLE_OFFSET
    ]
    checks.append(
        {
            "check": "shuffle_generator_seed_is_seed_plus_frozen_offset",
            "passed": not shuffle_bad,
            "detail": {"offset": SHUFFLE_OFFSET, "violations": shuffle_bad},
        }
    )

    env_sets = {json.dumps({k: v for k, v in r["environment"].items() if k != "hostname"}, sort_keys=True) for r in records}
    hosts = _unique([r["environment"]["hostname"] for r in records])
    checks.append(
        {
            "check": "execution_regime_identical_across_all_units",
            "passed": len(env_sets) == 1,
            "detail": {
                "n_distinct_environments": len(env_sets),
                "hostnames": hosts,
                "environments": [json.loads(x) for x in env_sets],
                "rule": "same host, same torch/CUDA build, same device model; driver recorded in the GPU ledger",
            },
        }
    )

    for split in SPLIT_ORDER:
        norms = _unique(
            [
                json.dumps({k: r["target_norm"][k] for k in ("mean", "scale", "n", "source")}, sort_keys=True)
                for r in records
                if r["split"] == split
            ]
        )
        checks.append(
            {
                "check": f"target_norm_identical_within_split_{split}",
                "passed": len(norms) == 1,
                "detail": [json.loads(x) for x in norms],
            }
        )
        # base caches must be byte-equal for base / multicontrib / chem8(fresh)
        base_fp = {
            (r["cache_fingerprints"]["train_base"], r["cache_fingerprints"]["valid_base"])
            for r in records
            if r["split"] == split
            and r["arm"] in ("base", "multicontrib", "chem8", "combo_chem_contrib")
        }
        # for base/multicontrib the *_base fingerprints are null; the direct
        # fingerprints must still agree with the chem8 arms' base fingerprints
        direct = {
            (r["cache_fingerprints"]["train"], r["cache_fingerprints"]["valid"])
            for r in records
            if r["split"] == split and r["arm"] in ("base", "multicontrib")
        }
        chem8_base = {
            (r["cache_fingerprints"]["train_base"], r["cache_fingerprints"]["valid_base"])
            for r in records
            if r["split"] == split and r["arm"] in ("chem8", "combo_chem_contrib")
        }
        checks.append(
            {
                "check": f"base_input_cache_identical_within_split_{split}",
                "passed": (
                    len(direct) <= 1
                    and len(chem8_base) <= 1
                    and (not direct or not chem8_base or direct == chem8_base)
                ),
                "detail": {
                    "base_and_multicontrib_fingerprints": sorted(direct),
                    "chem8_arms_base_fingerprints": sorted(chem8_base),
                    "deferred_until_base_units_exist": bool(not direct),
                },
            }
        )
        norm_fp = _unique([r["cache_fingerprints"]["train_norm"] for r in records if r["split"] == split])
        checks.append(
            {
                "check": f"train_only_standardiser_fingerprint_identical_{split}",
                "passed": len(norm_fp) == 1,
                "detail": norm_fp,
            }
        )

    # the same seed must produce the same initialisation for the shared trunk
    for split in SPLIT_ORDER:
        for seed in SEEDS[split]:
            entries = [r for r in records if r["split"] == split and r["seed"] == seed and r["arm"] in ("base", "chem8", "multicontrib", "combo_chem_contrib")]
            if not entries:
                continue
            base_hash = {r["base_init_state_hash"] for r in entries if r["base_init_state_hash"]}
            base_entry = [r for r in entries if r["arm"] == "base"]
            if base_entry:
                base_hash.add(base_entry[0]["init_state_hash"])
            checks.append(
                {
                    "check": f"shared_init_hash_matches_base_seed_{split}_s{seed}",
                    "passed": len(base_hash) == 1,
                    "detail": sorted(str(x) for x in base_hash),
                }
            )

    return checks


def integrity_payload(all_checks: Sequence[dict[str, Any]], *, phase: str) -> dict[str, Any]:
    failed = [c for c in all_checks if not c.get("passed")]
    return {
        "round_id": ROUND_ID,
        "phase": phase,
        "generated_at": utcnow(),
        "n_checks": len(all_checks),
        "n_passed": len(all_checks) - len(failed),
        "n_failed": len(failed),
        "all_passed": not failed,
        "blocking_policy": (
            "a failed check blocks the paired comparison that depends on it; the build aborts "
            "instead of silently mixing incomparable units"
        ),
        "failed": failed,
        "checks": list(all_checks),
    }


def completeness_checks(units: Mapping[tuple[str, str, int], SRC.Unit]) -> list[dict[str, Any]]:
    checks = []
    for split in SPLIT_ORDER:
        arm_list = ARMS if split == "random122" else Q2_ARMS
        for arm in arm_list:
            for seed in SEEDS[split]:
                present = (split, arm, seed) in units
                checks.append(
                    {
                        "check": f"unit_available_{split}_{arm}_s{seed}",
                        "passed": bool(present),
                        "detail": None if present else "MISSING: this comparison cell is not built",
                    }
                )
    return checks


# ---------------------------------------------------------------------------
# reuse manifest
# ---------------------------------------------------------------------------


def reuse_manifest(units: Mapping[tuple[str, str, int], SRC.Unit], split_audit: Mapping[str, Any]) -> dict[str, Any]:
    entries = []
    for spec in SRC.unit_registry():
        unit = units.get((spec["split"], spec["arm"], spec["seed"]))
        entry: dict[str, Any] = {
            "unit_key": spec["key"],
            "split": spec["split"],
            "arm": spec["arm"],
            "seed": spec["seed"],
            "origin": spec["origin"],
            "source_run": spec["source_run"],
            "run_dir": track_rel(spec["run_dir"]),
            "available": bool(unit is not None),
        }
        if unit is not None:
            record = unit_source_record(unit)
            entry.update(
                {
                    "files": record["files"],
                    "historical_git_commit": record["historical_git_commit"],
                    "code_hashes": record["code_hashes"],
                    "best_epoch": record["best_epoch"],
                    "epochs_run": record["epochs_run"],
                    "parameters": record["parameters"],
                    "recorded_valid_rmse": (record["recorded_metrics"]["valid"] or {}).get("rmse"),
                    "recomputed_valid_rmse": record["recomputed_metrics"]["valid"]["rmse"],
                    "recorded_test_rmse": (record["recorded_metrics"]["test"] or {}).get("rmse"),
                    "recomputed_test_rmse": record["recomputed_metrics"]["test"]["rmse"],
                    "check_summary": {
                        "n": len(unit.checks),
                        "n_failed": int(sum(1 for c in unit.checks if not c["passed"])),
                    },
                    "reuse_decision": (
                        "REUSED" if spec["origin"] == "reused" else "TRAINED_THIS_ROUND"
                    ),
                }
            )
        else:
            entry["reuse_decision"] = "MISSING_AT_BUILD_TIME"
        entries.append(entry)

    return {
        "round_id": ROUND_ID,
        "generated_at": utcnow(),
        "reuse_rule": (
            "A historical unit is reused only if all of the following hold: same dataset/split/arm/seed "
            "identity, byte-identical training-code hashes (all 12 files), identical frozen R3 recipe "
            "knobs (shuffle seed = seed + 917000), identical Train-only target standardiser, identical "
            "frozen split row hashes, strict row-order alignment of the prediction files, exact label "
            "reproduction, metric reproduction within 1e-5, checkpoint selection from Validation only, "
            "and an identical execution regime (host / torch / CUDA / GPU model)."
        ),
        "units": entries,
        "missing_units": [
            {
                "unit_key": spec["key"],
                "split": spec["split"],
                "arm": spec["arm"],
                "seed": spec["seed"],
                "run_dir": track_rel(spec["run_dir"]),
            }
            for spec in SRC.unit_registry()
            if spec["origin"] == "fresh" and (spec["split"], spec["arm"], spec["seed"]) not in units
        ],
        "split_audit_reference": {
            name: {
                "rows_sha256": split_audit["splits"][name]["rows_sha256"],
                "sha256": split_audit["splits"][name]["sha256"],
            }
            for name in SPLIT_ORDER
        },
    }


# ---------------------------------------------------------------------------
# main build
# ---------------------------------------------------------------------------


def build(*, audit_only: bool = False, n_boot: int | None = None, check_freeze: bool = True) -> dict[str, Any]:
    if check_freeze:
        freeze = verify_analysis_freeze()
    else:
        # development-only escape hatch: the reported artefacts must be built
        # with the freeze check ON (see PROTOCOL.md §9)
        freeze = {"development_mode": True, "analysis_code_hashes": analysis_code_hashes()}
    ANALYSIS.mkdir(parents=True, exist_ok=True)

    print("[1/6] loading frozen splits and molecule records", flush=True)
    splits, split_audit = SRC.load_splits(build_support=True, verbose=True)

    print("[2/6] loading units", flush=True)
    units = SRC.load_all_units(splits, verbose=True)
    unit_checks = [c for unit in units.values() for c in unit.checks]

    print("[3/6] cross-unit and completeness checks", flush=True)
    cross_checks = cross_unit_checks(units)
    complete_checks = completeness_checks(units)
    all_checks = split_audit["checks"] + unit_checks + cross_checks + complete_checks
    integrity = integrity_payload(all_checks, phase="final" if len(units) == len(SRC.unit_registry()) else "partial")

    source_audit = {
        "round_id": ROUND_ID,
        "generated_at": utcnow(),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "analysis_code_freeze": freeze,
        "analysis_code_hashes": analysis_code_hashes(),
        "inputs": {
            "chembench_csv": split_audit["csv"],
            "splits": split_audit["splits"],
            "frozen_training_protocol": {
                "path": track_rel(LONGRUN_FREEZE),
                "sha256": sha256_file(LONGRUN_FREEZE),
                "round": track_rel(LONGRUN),
            },
        },
        "units": [unit_source_record(unit) for unit in units.values()],
        "completeness": {
            "n_units_expected": len(SRC.unit_registry()),
            "n_units_available": len(units),
            "n_reused": int(sum(1 for u in units.values() if u.origin == "reused")),
            "n_fresh": int(sum(1 for u in units.values() if u.origin == "fresh")),
            "expected_units": [
                {"split": u["split"], "arm": u["arm"], "seed": u["seed"], "origin": u["origin"], "key": u["key"]}
                for u in SRC.unit_registry()
            ],
        },
        "exposure": {
            "test_policy": "DEVELOPMENT_TEST",
            "checkpoint_selection": "Validation RMSE only (frozen in every unit)",
            "test_used_for_training": False,
            "test_used_within_run_for_epoch_selection": False,
        },
    }
    write_json(ANALYSIS / "source_audit.json", source_audit)
    write_json(ANALYSIS / "run_reuse_manifest.json", reuse_manifest(units, split_audit))
    write_json(ANALYSIS / "integrity_checks.json", integrity)

    if not integrity["all_passed"]:
        non_completeness = [
            c for c in all_checks if not c["passed"] and not c["check"].startswith("unit_available_")
        ]
        if non_completeness:
            raise SystemExit(
                f"integrity checks failed (non-completeness): {len(non_completeness)}; "
                "see analysis/integrity_checks.json"
            )
        missing = [c for c in all_checks if not c["passed"]]
        if len(missing) == len([c for c in all_checks if c["check"].startswith("unit_available_")]):
            print(
                json.dumps({"status": "no_units_trained_yet", "missing": len(missing)}, indent=1),
                flush=True,
            )
            return {"integrity": integrity, "units": len(units)}

    if audit_only:
        return {"integrity": integrity, "units": len(units)}

    boot = int(n_boot if n_boot is not None else N_BOOT)

    print("[4/6] per-molecule features and metric tables", flush=True)
    frames = {}
    frame_meta = {}
    for split in SPLIT_ORDER:
        frame, meta = SRC.molecule_frame(splits[split])
        frames[split] = (frame, meta)
        frame_meta[split] = meta
        print(f"  {split}: {len(frame)} query rows, edges {meta['quantile_edges']}", flush=True)

    metrics_frame = A.metrics_table(units)
    metrics_frame.to_csv(ANALYSIS / "metrics_by_split_seed_arm.csv", index=False)

    per_mol = A.per_molecule_table(units, frames)
    per_mol.to_csv(ANALYSIS / "per_molecule_predictions.csv", index=False)

    print("[5/6] paired uncertainty, interaction, tail risk, subgroups", flush=True)
    paired = A.paired_uncertainty(units, splits, n_boot=boot)
    paired["direction_consistency"] = A.direction_consistency(paired)
    write_json(ANALYSIS / "paired_uncertainty.json", paired)

    interaction = A.factorial_interaction(units, splits, n_boot=boot)
    write_json(ANALYSIS / "factorial_interaction.json", interaction)

    tail = A.tail_risk(units, splits, n_boot=boot)
    write_json(ANALYSIS / "tail_risk.json", tail)

    subgroups = A.scaffold_subgroups(units, frames, splits, n_boot=boot)
    subgroups["feature_frame_source"] = frame_meta
    write_json(ANALYSIS / "scaffold_subgroups.json", subgroups)

    concentration = A.concentration_diagnostics(units, splits)
    write_json(ANALYSIS / "concentration_diagnostics.json", concentration)

    print("[6/6] manifest", flush=True)
    manifest = {
        "round_id": ROUND_ID,
        "generated_at": utcnow(),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "protocol_freeze": track_rel(PROTOCOL_FREEZE),
        "bootstrap": {"n_boot": boot, "unit": "Bemis-Murcko scaffold cluster", "seed": BOOTSTRAP_SEED},
        "tail": {"fraction": TOP_FRACTION, "cvar": "ceil(0.10*n) worst molecules of the model itself"},
        "units": {
            "n_total": len(units),
            "n_reused": int(sum(1 for u in units.values() if u.origin == "reused")),
            "n_fresh": int(sum(1 for u in units.values() if u.origin == "fresh")),
        },
        "outputs": [
            "analysis/source_audit.json",
            "analysis/run_reuse_manifest.json",
            "analysis/integrity_checks.json",
            "analysis/metrics_by_split_seed_arm.csv",
            "analysis/per_molecule_predictions.csv",
            "analysis/paired_uncertainty.json",
            "analysis/factorial_interaction.json",
            "analysis/tail_risk.json",
            "analysis/scaffold_subgroups.json",
            "analysis/concentration_diagnostics.json",
            "analysis/gpu_ledger.json",
        ],
        "reproduce": [
            "uv run python tracks/structural-sufficiency/lcrc_generalization_factorial_p1_v1/code/run_analysis.py",
            "uv run pytest -q tracks/structural-sufficiency/lcrc_generalization_factorial_p1_v1/tests",
        ],
    }
    write_json(ANALYSIS / "manifest.json", manifest)
    return {
        "integrity": {"n_checks": integrity["n_checks"], "n_failed": integrity["n_failed"]},
        "units": len(units),
        "metrics_rows": len(metrics_frame),
        "per_molecule_rows": len(per_mol),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LCRC-P1 analysis build")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--n-boot", type=int, default=None)
    parser.add_argument(
        "--no-freeze-check",
        action="store_true",
        help="development only: skip the analysis-code freeze check (never used for reported artefacts)",
    )
    args = parser.parse_args(argv)
    result = build(audit_only=args.audit_only, n_boot=args.n_boot, check_freeze=not args.no_freeze_check)
    print(json.dumps(result, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
