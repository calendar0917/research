"""Freeze the round: ``configs/protocol.json``, ``configs/run_matrix.json`` and
``analysis/protocol_freeze.json``.

The TU cross-validation scheme is chosen here from *measured hardware cost only*
(``analysis/cost_measurement.json``), never from a score.  The projection formulas
are printed into the frozen record so the arithmetic is auditable:

    unit_gpu_hours = mean_epoch_seconds * formal_epochs / 3600
    family_total   = sum(unit_gpu_hours) * COST_SAFETY_FACTOR + fixed overheads
    allowed        = family_total + reserve <= GPU_SOFT_CAP_HOURS
                     and projected wall-clock makespan <= 7 h

Priority order (protocol §8.2): A (10-fold x 2 seeds) -> B (5-fold x 3 seeds) ->
C (5-fold x 2 seeds, RESOURCE_LIMITED).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

import ptb_common as C

#: Fixed, pre-registered GPU cost of the frozen local-edit training + trace stage
#: (6 units x R3 on 2016 rows + a CPU trace pass).  Taken from that round's own
#: protocol estimate (0.05 GPU-h/run) with a 1.5x margin.
LOCAL_EDIT_GPU_HOURS = 0.50
#: Local-edit is a wall-clock cost on one GPU, not part of the molecule family.
WALL_CLOCK_LIMIT_HOURS = 7.0


def _cost_rows(payload: Mapping[str, Any]) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    out: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row in payload["rows"]:
        out[(row["family"], row["dataset"], row["model"])] = row
    return out


def project_molecule(rows, factor: float) -> dict[str, Any]:
    per_unit = {}
    total = 0.0
    for dataset in C.MOLECULE_ORDER:
        epochs = int(C.MOLECULE_DATASETS[dataset]["epochs"])
        # conservative: use the slower of the two variants for both, because a
        # single 3-epoch measurement can be biased by CUDA warm-up
        epoch_s = max(float(rows[("molecule", dataset, m)]["mean_epoch_seconds"]) for m in C.MODELS)
        overhead = max(
            max(float(rows[("molecule", dataset, m)]["process_wall_seconds"]) - float(rows[("molecule", dataset, m)]["train_wall_seconds"]), 0.0)
            for m in C.MODELS
        )
        for model in C.MODELS:
            unit_h = (epoch_s * epochs + overhead) / 3600.0
            per_unit[f"{dataset}:{model}"] = {
                "mean_epoch_seconds": epoch_s,
                "epochs": epochs,
                "overhead_seconds": overhead,
                "unit_gpu_hours": unit_h,
            }
            total += unit_h * len(C.MOLECULE_SEEDS)
    return {
        "per_dataset_model": per_unit,
        "n_units": len(per_unit) * len(C.MOLECULE_SEEDS),
        "gpu_hours_raw": total,
        "gpu_hours_projected": total * factor,
    }


def project_tu(rows, scheme: Mapping[str, Any], factor: float, cache_build_by_dataset: Mapping[str, float]) -> dict[str, Any]:
    per_unit = {}
    raw = 0.0
    n_splits = int(scheme["n_splits"])
    seeds = tuple(scheme["seeds"])
    for dataset in C.TU_DATASETS:
        epoch_s = max(float(rows[("tu", dataset, m)]["mean_epoch_seconds"]) for m in C.MODELS)
        overhead = max(
            max(float(rows[("tu", dataset, m)]["process_wall_seconds"]) - float(rows[("tu", dataset, m)]["train_wall_seconds"]), 0.0)
            for m in C.MODELS
        )
        for model in C.MODELS:
            unit_h = (epoch_s * int(C.TU_EPOCHS) + overhead) / 3600.0
            n_units = n_splits * len(seeds)
            per_unit[f"{dataset}:{model}"] = {
                "mean_epoch_seconds": epoch_s,
                "epochs": int(C.TU_EPOCHS),
                "overhead_seconds": overhead,
                "unit_gpu_hours": unit_h,
                "units": n_units,
            }
            raw += unit_h * n_units
    cache_build = float(sum(cache_build_by_dataset.get(d, 0.0) for d in C.TU_DATASETS)) * n_splits
    cache_hours = cache_build / 3600.0
    return {
        "scheme": dict(scheme),
        "per_dataset_model": per_unit,
        "n_units": len(per_unit) * n_splits * len(seeds),
        "fold_cache_build_gpu_hours": cache_hours,
        "gpu_hours_raw": raw + cache_hours,
        "gpu_hours_projected": (raw + cache_hours) * factor,
    }


def makespan_hours(molecule_h: float, local_edit_h: float, tu_h: float) -> float:
    """Two-GPU makespan when units may be split freely between the two pools."""
    total = molecule_h + local_edit_h + tu_h
    return max(total / 2.0, max(molecule_h, local_edit_h, tu_h))


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cost", default=str(C.ANALYSIS / "cost_measurement.json"))
    ap.add_argument("--force-scheme", default=None, choices=list(C.TU_SCHEMES))
    ap.add_argument("--out-protocol", default=str(C.CONFIGS / "protocol.json"))
    ap.add_argument("--out-matrix", default=str(C.CONFIGS / "run_matrix.json"))
    ap.add_argument("--out-freeze", default=str(C.ANALYSIS / "protocol_freeze.json"))
    args = ap.parse_args(argv)

    cost = C.read_json(args.cost)
    rows = _cost_rows(cost)
    factor = float(C.COST_SAFETY_FACTOR)
    mol = project_molecule(rows, factor)
    cache_build_by_dataset = {
        row["dataset"]: float(row["fold_cache_build_seconds"])
        for key, row in rows.items()
        if key[0] == "tu" and key[2] == C.MODELS[0]
    }

    candidates = []
    chosen = None
    for name in ("A_10x2", "B_5x3", "C_5x2"):
        scheme = dict(C.TU_SCHEMES[name])
        tu = project_tu(rows, scheme, factor, cache_build_by_dataset)
        total = mol["gpu_hours_projected"] + LOCAL_EDIT_GPU_HOURS + tu["gpu_hours_projected"] + float(
            C.EXPLICIT_RESERVE_GPU_HOURS
        )
        span = makespan_hours(mol["gpu_hours_projected"] + LOCAL_EDIT_GPU_HOURS, 0.0, tu["gpu_hours_projected"])
        entry = {
            "name": name,
            "scheme": scheme,
            "tu_gpu_hours_projected": tu["gpu_hours_projected"],
            "n_tu_units": tu["n_units"],
            "family_total_with_reserve_gpu_hours": total,
            "within_gpu_soft_cap": bool(total <= C.GPU_SOFT_CAP_HOURS),
            "projected_wall_clock_hours": span,
            "within_wall_clock": bool(span <= WALL_CLOCK_LIMIT_HOURS),
            "degraded": bool(scheme.get("degraded", False)),
        }
        entry["eligible"] = bool(entry["within_gpu_soft_cap"] and entry["within_wall_clock"])
        candidates.append(entry)
        if chosen is None and entry["eligible"]:
            chosen = entry

    if args.force_scheme:
        chosen = next(e for e in candidates if e["name"] == args.force_scheme)
        chosen["forced_by_flag"] = True

    if chosen is None:
        chosen = candidates[-1]
        chosen["requires_degradation"] = True

    scheme = chosen["scheme"]
    tu_units = C.tu_units(scheme)
    mol_units = C.mol_units()

    protocol = {
        "schema": "lcrc-ptb-protocol-v1",
        "round_id": C.ROUND_ID,
        "protocol_id": C.PROTOCOL_ID,
        "reference_head": C.REFERENCE_HEAD,
        "created_utc": C.utcnow(),
        "frozen_before_formal_training": True,
        "models": {
            "A0": {
                "label": "LCRC-A frozen computational graph",
                "parameters_single_task": C.EXPECTED_PARAMS_SINGLE["A0"],
                "graph": "AtomEncoder -> BondEncoder -> S1(b_sum,b_mean,typed nbr) -> S2(T,E,J,K,C) -> LocalMLP+LayerNorm -> dropout(h) -> L=sum_v f(h_v) -> rho(global,mean h,mean h^2,log1p n) -> L+rho",
            },
            "C1": {
                "label": "LCRC-C1 = A0 minus the 96-d b_mean input block",
                "parameters_single_task": C.EXPECTED_PARAMS_SINGLE["C1"],
                "parameters_single_task_delta": C.EXPECTED_PARAM_DELTA,
                "role": "pre-registered main model; deltas in the Test results do not re-select it",
            },
        },
        "recipe": {
            **{k: v for k, v in C.RECIPE.items() if k != "epochs_by_dataset"},
            "epochs_by_dataset": C.RECIPE["epochs_by_dataset"],
            "molhiv_epoch_cap_never_extended": True,
        },
        "molecule_experiment": {
            "datasets": list(C.MOLECULE_ORDER),
            "seeds": list(C.MOLECULE_SEEDS),
            "units": [{"dataset": d, "model": m, "seed": s} for (d, m, s) in mol_units],
            "n_units": len(mol_units),
            "data": "ChemBench published splits (frozen bytes) for ESOL/FreeSolv/BACE/BBBP/Tox21; official OGB scaffold split for ogbg-molhiv",
            "test_policy": "terminal: no Test read until the dataset's checkpoint manifest is frozen",
            "result_label": C.RESULT_LABEL_MOL,
        },
        "tu_experiment": {
            "datasets": list(C.TU_DATASETS),
            "scheme": scheme,
            "scheme_name": chosen["name"],
            "n_units": len(tu_units),
            "outer_folds": "StratifiedKFold(n_splits=%d, shuffle=True, random_state=%d)"
            % (int(scheme["n_splits"]), int(scheme["fold_seed"])),
            "inner_valid": "stratified 10%% of outer-train, random_state=%d+fold_id" % C.TU_INNER_VALID_SEED_BASE,
            "epochs": int(C.TU_EPOCHS),
            "recipe": {
                "optimizer": "AdamW",
                "peak_lr": 1e-3,
                "min_lr": 1e-5,
                "weight_decay": 1e-4,
                "batch_size": 64,
                "dropout": 0.10,
                "gradient_clip": 5.0,
                "loss": "CrossEntropyLoss (unweighted)",
                "scheduler": "per_step_cosine",
                "selection": "max Valid accuracy; tie -> lower Valid CE; tie -> earlier epoch",
            },
            "result_label": C.TU_RESULT_LABEL,
            "resource_limited": bool(scheme.get("degraded", False) or chosen.get("requires_degradation")),
        },
        "cost_projection": {
            "safety_factor": factor,
            "molecule": mol,
            "candidates": candidates,
            "chosen": chosen["name"],
            "local_edit_gpu_hours_allowance": LOCAL_EDIT_GPU_HOURS,
            "explicit_reserve_gpu_hours": C.EXPLICIT_RESERVE_GPU_HOURS,
            "wall_clock_limit_hours": WALL_CLOCK_LIMIT_HOURS,
            "rule": (
                "A if molecule+local-edit+TU(10x2)+reserve <= 14 GPU-h AND makespan <= 7 h; "
                "else B if the same holds for TU(5x3); else C (5x2) marked RESOURCE_LIMITED"
            ),
        },
        "budget": {
            "gpu_soft_cap_hours": C.GPU_SOFT_CAP_HOURS,
            "gpu_hard_cap_hours": C.GPU_HARD_CAP_HOURS,
            "wall_clock_cap_hours": C.WALL_CLOCK_CAP_HOURS,
            "reporting_reserve_hours": C.REPORTING_RESERVE_HOURS,
        },
        "statistics": {
            "bootstrap_seed": C.STATS_BOOTSTRAP_SEED,
            "bootstrap_replicates": C.STATS_BOOTSTRAP_REPLICATES,
            "paired_unit": "molecule row for molecule tasks; unbreakable pair component for local-edit; outer fold for TU",
        },
        "label_policy": {
            "molecule_test": "EXPOSED_TEST_EXPLORATORY - already read in earlier rounds; not a fresh blind test",
            "tu": "outer fold evaluated once per unit after the checkpoint hash is written",
            "local_edit": "frozen in lcrc_local_edit_error_diagnosis_v1 (Audit one-shot, after probe freeze)",
        },
    }

    matrix = {
        "schema": "lcrc-ptb-run-matrix-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "molecule": [{"unit": C.mol_run_name(d, m, s), "dataset": d, "model": m, "seed": s} for (d, m, s) in mol_units],
        "tu": [
            {
                "unit": C.tu_run_name(d, m, f, s),
                "dataset": d,
                "model": m,
                "fold": f,
                "seed": s,
            }
            for (d, m, f, s) in tu_units
        ],
        "tu_scheme": chosen["name"],
        "local_edit": {
            "track": "tracks/structural-sufficiency/lcrc_local_edit_error_diagnosis_v1",
            "units": C.read_json(C.EDIT / "configs/run_matrix.json")["allowed_units"],
            "note": "executed under that round's own frozen protocol",
        },
        "n_units_total": len(mol_units) + len(tu_units) + 6,
    }
    C.write_json(args.out_protocol, protocol)
    C.write_json(args.out_matrix, matrix)

    freeze = {
        "schema": "lcrc-ptb-protocol-freeze-v1",
        "round_id": C.ROUND_ID,
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "git_dirty_tracked": C.git_dirty(),
        "reference_head": C.REFERENCE_HEAD,
        "protocol_sha256": C.sha256_text(C.stable_json(protocol)),
        "run_matrix_sha256": C.sha256_text(C.stable_json(matrix)),
        "protocol_json_sha256": C.sha256_file(args.out_protocol),
        "run_matrix_json_sha256": C.sha256_file(args.out_matrix),
        "code_hashes": C.code_hashes(),
        "reused_modules": C.reused_module_hashes(),
        "env": C.env_info(),
        "cost_measurement": {
            "path": C.track_rel(args.cost),
            "sha256": C.sha256_file(args.cost),
            "env": cost.get("env"),
        },
        "chosen_tu_scheme": chosen["name"],
        "candidates": candidates,
    }
    C.write_json(args.out_freeze, freeze)
    C.log(f"frozen TU scheme = {chosen['name']} ({chosen['n_tu_units']} TU units, {len(mol_units)} molecule units)")
    print(C.stable_json({"scheme": chosen["name"], "candidates": candidates}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
