"""Freeze the protocol before any formal run.

Writes, deterministically:

* ``configs/variants.json``   - machine-readable branch definitions
* ``configs/run_matrix.json`` - the pre-registered 54-cell main matrix plus the
  cross-split rule (FULL + COMPACT + B5 on repeats 123/124, deduplicated)
* ``analysis/protocol_freeze.json`` - source/split hashes, code hashes, the R3
  recipe, budget and selection tolerances

``PROTOCOL.md`` (the executed prompt + the execution mapping) is written
separately by the researcher and hashed here.
"""

from __future__ import annotations

import argparse
from typing import Any

import dissect_common as C
import dissect_data as D
import dissect_model as M
import dissect_source as S


def variant_payload() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in C.VARIANT_ORDER:
        local_cols, rho_cols = C.variant_cols(key)
        out[key] = {
            "order": list(C.VARIANT_ORDER).index(key),
            "name": C.VARIANT_NAMES[key],
            "stage": C.VARIANT_DEFS[key]["stage"],
            "narrative": C.VARIANT_DEFS[key]["narrative"],
            "definition": dict(C.VARIANT_DEFS[key]),
            "local_cols": list(local_cols),
            "n_local_cols": len(local_cols),
            "rho_cols": list(rho_cols),
            "n_rho_cols": len(rho_cols),
            "expected_trainable_parameters": C.EXPECTED_PARAMS[key],
        }
    return out


def build_run_matrix(compact: str | None = None) -> dict[str, Any]:
    cells = [f"r{C.MAIN_REPEAT}:{v}" for v in C.VARIANT_ORDER for _ in (0,)]
    return {
        "schema": "lcrc-lipo-dissection-run-matrix-v1",
        "protocol_id": C.PROTOCOL_ID,
        "main_repeat": C.MAIN_REPEAT,
        "policy": {
            "repeats": list(C.REPEATS),
            "test_repeats": [123, 124],
            "seeds": list(C.SEEDS),
            "max_concurrent_training_jobs": 2,
            "gpu_soft_cap_hours": C.GPU_SOFT_CAP_HOURS,
            "gpu_hard_cap_hours": C.GPU_HARD_CAP_HOURS,
            "max_formal_runs": 72,
            "execution_order": [f"{v}:s{s}" for v in C.VARIANT_ORDER for s in C.SEEDS],
            "stage_order": ["A", "B", "C", "COMPACT", "cross_split"],
            "variant_order": list(C.VARIANT_ORDER),
        },
        "main_cells": [
            {"repeat": C.MAIN_REPEAT, "variant": v, "seed": s, "run_name": C.run_name(C.MAIN_REPEAT, v, s)}
            for v in C.VARIANT_ORDER
            for s in C.SEEDS
        ],
        "allowed_cells": cells,
        "cross_split_rule": {
            "repeats": [123, 124],
            "variants": ["A0", "B5"],
            "compact_variant": compact,
            "deduplicate": True,
            "note": (
                "Each repeat trains FULL, the selected COMPACT and B5 with seeds 0/1/2 from scratch; "
                "identical variants are trained once. COMPACT is chosen from public122 Validation only."
            ),
        },
        "n_main_runs": len(C.VARIANT_ORDER) * len(C.SEEDS),
        "n_cross_split_runs_max": 2 * 3 * 3,
        "n_formal_runs_max": len(C.VARIANT_ORDER) * len(C.SEEDS) + 2 * 3 * 3,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Freeze the LCRC-A dissection protocol")
    parser.add_argument("--compact", default=None, help="set only after COMPACT selection (amendment)")
    args = parser.parse_args(argv)

    spec = D.frozen_spec()
    source_report: dict[str, Any] = {}
    for repeat in C.REPEATS:
        src = S.load_lipop(repeat, labels="none")
        source_report[f"r{repeat}"] = {
            "split_file": src["split_file"],
            "split_file_sha256": src["split_file_sha256"],
            "split_rows_sha256": src["split_rows_sha256"],
            "split_sizes": src["sizes"],
            "labels_scope": src["labels_scope"],
        }

    variants = variant_payload()
    C.write_json(C.CONFIGS / "variants.json", variants)
    matrix = build_run_matrix(args.compact)
    C.write_json(C.CONFIGS / "run_matrix.json", matrix)

    from mb_common import RECIPE as PILOT_RECIPE

    for key, value in C.RECIPE.items():
        if key in ("adam_betas", "adam_eps"):
            continue
        if key in PILOT_RECIPE and PILOT_RECIPE[key] != value:
            raise ValueError(f"recipe mismatch on {key}: {value!r} != frozen {PILOT_RECIPE[key]!r}")

    protocol_md = C.TRACK / "PROTOCOL.md"
    freeze = {
        "schema": "lcrc-lipo-dissection-protocol-freeze-v1",
        "created_utc": C.utcnow(),
        "protocol_id": C.PROTOCOL_ID,
        "round_id": C.ROUND_ID,
        "git_commit": C.git_commit(),
        "git_dirty_tracked": C.git_dirty(),
        "protocol_md": {
            "path": C.track_rel(protocol_md),
            "sha256": C.sha256_file(protocol_md) if protocol_md.exists() else None,
        },
        "source": {
            "repository": "https://github.com/shenwanxiang/ChemBench",
            "commit": C.CHEMBENCH_COMMIT,
            "csv": C.track_rel(C.LIPOP_CSV),
            "csv_sha256": C.LIPOP_CSV_SHA256,
            "repeats": source_report,
        },
        "feature_spec": {
            "name": spec.name,
            "hash": spec.hash(),
            "atom_feat_dim": spec.atom_feat_dim,
            "bond_feat_dim": spec.bond_feat_dim,
            "atom_scalar_dim": spec.n_scalars,
            "global_dim": spec.global_dim,
            "n_nbr_codes": spec.n_nbr_codes,
            "n_path_codes": spec.n_path_codes,
            "local_layout": {k: list(v) for k, v in C.LOCAL_LAYOUT.items()},
            "rho_layout": {k: list(v) for k, v in C.RHO_LAYOUT.items()},
        },
        "model": {
            "width_config": dict(M.DEFAULT_CONFIG),
            "variant_order": list(C.VARIANT_ORDER),
            "stage_variants": {k: list(v) for k, v in C.STAGE_VARIANTS.items()},
            "expected_params": dict(C.EXPECTED_PARAMS),
            "variants_json_sha256": C.sha256_file(C.CONFIGS / "variants.json"),
            "run_matrix_sha256": C.sha256_file(C.CONFIGS / "run_matrix.json"),
        },
        "recipe": {
            **{k: (list(v) if isinstance(v, tuple) else v) for k, v in C.RECIPE.items()},
            "epochs": int(C.RECIPE["epochs"]),
            "steps_per_epoch": 27,
            "optimizer_steps": 27 * int(C.RECIPE["epochs"]),
        },
        "selection": {
            "main_repeat": C.MAIN_REPEAT,
            "seeds": list(C.SEEDS),
            "tolerances_logD": {
                "mean_valid_rmse_delta": C.TOL_MEAN_DR,
                "max_seed_valid_rmse_delta": C.TOL_MAX_DR,
                "mean_valid_mae_delta": C.TOL_MEAN_DA,
            },
            "tolerance_status": "practical purchase rules, not statistical equivalence",
            "order": "fewest parameters, then lower mean Valid RMSE, then variant order",
            "if_no_candidate": "COMPACT = FULL",
            "selection_inputs": ["public122 Validation metrics", "trainable parameter counts"],
            "selection_excludes": ["Test metrics"],
        },
        "evaluation": {
            "exposed_test_status": "EXPOSED_TEST_EXPLORATORY",
            "bootstrap_replicates": C.BOOTSTRAP_REPLICATES,
            "bootstrap_seed": C.BOOTSTRAP_SEED,
            "bootstrap_grouping": "Bemis-Murcko scaffold; empty scaffold keyed by canonical SMILES",
            "attribution_ranks": list(C.ATTRIB_RANKS),
        },
        "budget": {
            "gpu_soft_cap_hours": C.GPU_SOFT_CAP_HOURS,
            "gpu_hard_cap_hours": C.GPU_HARD_CAP_HOURS,
            "max_formal_runs": 72,
            "max_concurrent": 2,
        },
        "code_hashes": C.code_hashes(),
        "reused_module_hashes": C.reused_module_hashes(),
        "cpu_equivalence": {
            "eval_abs_tolerance": C.CPU_EQUIV_TOL,
            "train_mode_abs_tolerance": C.CPU_EQUIV_TOL,
            "valid_replay_tolerance": C.REPRO_TOL,
            "reference": dict(C.REFERENCE),
        },
    }
    C.write_json(C.ANALYSIS / "protocol_freeze.json", freeze)
    C.log(f"wrote {C.track_rel(C.ANALYSIS / 'protocol_freeze.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
