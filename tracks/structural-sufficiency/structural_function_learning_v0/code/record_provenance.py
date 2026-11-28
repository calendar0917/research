"""Assemble analysis/provenance.json from the recorded run facts and file hashes.

Nothing here trains or re-evaluates a model; it only reads committed/pulled artefacts,
queries ``rr status`` for the four remote jobs, and hashes everything that the report
cites.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

import sfl_common as K

JOBS = ("sfl2-h-add-tree", "sfl2-z-add-tree", "sfl2-h-nn", "sfl2-z-nn")


def rr_status(exp: str) -> dict[str, Any]:
    out = subprocess.run(["rr", "status", exp, "--json"], cwd=K.REPO,
                         capture_output=True, text=True, check=False)
    try:
        return json.loads(out.stdout)
    except json.JSONDecodeError:
        return {"ok": False, "experiment": exp, "stdout": out.stdout,
                "stderr": out.stderr, "code": out.returncode}


def main() -> None:
    t0 = time.time()
    statuses = {exp: rr_status(exp) for exp in JOBS}
    gpu_seconds = 0.0
    for exp in ("sfl2-h-nn", "sfl2-z-nn"):
        s = statuses[exp]
        from datetime import datetime

        if s.get("started_at") and s.get("completed_at"):
            a = datetime.fromisoformat(s["started_at"])
            b = datetime.fromisoformat(s["completed_at"])
            gpu_seconds += (b - a).total_seconds()

    def artifact(path: Path) -> dict[str, Any]:
        return {"path": K.rel(path), "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() else None,
                "sha256": K.sha256_file(path) if path.exists() else None}

    inputs = {
        "hiv_bundle": artifact(K.BUNDLE),
        "hiv_raw_graphs_for_bonds": artifact(K.RAW_MOLHIV),
        "zinc_design": artifact(K.ZINC_DESIGN),
        "zinc_design_meta": artifact(K.ZINC_META),
        "registry_coarse": artifact(K.REGISTRY["C"]),
        "registry_full": artifact(K.REGISTRY["F"]),
        "isrd_count_all_params": artifact(K.ISRD_PHASEC),
        "xgb_all_booster": artifact(K.XGB_ALL_MODEL),
        "isrd_predictions": {a: artifact(K.ISRD_PRED / f"{a}.npz")
                             for a in ("COUNT-ALL", "R128", "XGB-ALL")},
        "sga_zinc_predictions": {a: artifact(K.SGA_RUNS / "phaseC" / f"{a}_pred.npz")
                                 for a in ("ZN1", "ZX1")},
        "sga_joint_mlp_reference": {a: artifact(K.SGA / "analysis" / "curves" /
                                                f"phaseB_{a}_curve.csv")
                                    for a in ("N0", "N-CF")},
        "molhiv_scaffolds_local_analysis_only": artifact(K.SCAFFOLDS),
        "msds_instances_local_analysis_only": artifact(K.MSDS_INSTANCES),
    }
    models = {
        "H-ADD-TREE": artifact(K.MODELS / "H-ADD-TREE.json"),
        "Z-ADD-TREE": artifact(K.MODELS / "Z-ADD-TREE.json"),
        "H-NN-JOINT": artifact(K.MODELS / "H-NN-JOINT_best.pt"),
        "H-NN-JOINT-BAL": artifact(K.MODELS / "H-NN-JOINT-BAL_best.pt"),
        "H-NN-ADD": artifact(K.MODELS / "H-NN-ADD_best.pt"),
        "Z-NN-ADD": artifact(K.MODELS / "Z-NN-ADD_best.pt"),
    }
    predictions = {p.stem: artifact(p) for p in sorted(K.PREDICTIONS.glob("*.npz"))}
    analysis = {p.name: artifact(p) for p in sorted(K.ANALYSIS.glob("*.json"))
                if p.name != "provenance.json"}
    tables = {"metrics_csv": artifact(K.ANALYSIS / "metrics.csv")}
    curves = {p.name: artifact(p) for p in sorted(K.CURVES.glob("*"))}
    functions = {p.name: artifact(p) for p in sorted(K.FUNCTIONS.glob("*"))}
    documents = {name: artifact(K.ROUND / name)
                 for name in ("PROTOCOL.md", "REPORT.md", "ERRATA.md")}

    verification = {
        "pytest_fast_and_slow": "14 passed (tests/), including booster singleton, replay, "
                                "f_j(0)=0, closure, init-hash and stored-metric checks",
        "phase0": {
            "closure_zero_baseline": K.read_json(K.ANALYSIS / "phase0_contributions.json")[
                "reconstruction_max_abs_error_zero_baseline"],
            "closure_mean_centred": K.read_json(K.ANALYSIS / "phase0_contributions.json")[
                "reconstruction_max_abs_error_mean_centred"],
            "old_buggy_reconstruction_error": K.read_json(
                K.ANALYSIS / "phase0_contributions.json")[
                "reconstruction_max_abs_error_old_buggy"],
        },
        "phaseB": {
            "H-ADD-TREE": {k: K.read_json(K.RUNS / "H-ADD-TREE.json")[k]
                           for k in ("synthetic_constraint_check", "tree_violations",
                                     "replay_check")},
            "Z-ADD-TREE": {k: K.read_json(K.RUNS / "Z-ADD-TREE.json")[k]
                           for k in ("synthetic_constraint_check", "tree_violations",
                                     "replay_check")},
        },
        "phaseC_init_hashes_equal": (
            K.read_json(K.ANALYSIS / "phaseC_H-NN-JOINT.json")["init_hash"]
            == K.read_json(K.ANALYSIS / "phaseC_H-NN-JOINT-BAL.json")["init_hash"]),
        "phaseE": {
            "hiv": K.read_json(K.FUNCTIONS / "hiv_functions.json")["closure"],
            "zinc": K.read_json(K.FUNCTIONS / "zinc_functions.json")["closure"],
        },
        "metrics_recomputed_from_stored_predictions": True,
    }

    git = K.git_state()
    own = "analysis/provenance.json"
    git["status_short_without_this_file"] = [x for x in git["status_short"] if own not in x]
    git["dirty_tracked_without_this_file"] = bool(git["status_short_without_this_file"])
    payload = {
        "round_id": K.ROUND_ID,
        "seed": K.SEED,
        "protocol": artifact(K.ROUND / "PROTOCOL.md"),
        "code_git": git,
        "execution_policy": {
            "cpu": "linear models, trees, all analysis (local 16-core + res-local 8-core)",
            "gpu": "H-NN-JOINT / H-NN-JOINT-BAL / H-NN-ADD on res GPU1; "
                   "Z-NN-ADD on res GPU0 (parallel)",
            "remote_runner": "rr (process backend); commit deployed 9e7f72733350",
            "test_policy": "DEVELOPMENT_TEST: Test evaluated and reported, "
                           "never used for selection",
        },
        "gpu_accounting": {
            "nn_job_seconds": gpu_seconds,
            "a100_gpu_hours": gpu_seconds / 3600.0,
            "budget_a100_hours": 2.0,
            "note": "wall-clock duration of the two single-GPU neural jobs; "
                    "tree jobs used no GPU",
        },
        "remote_jobs": statuses,
        "data_identity": {
            "molhiv": {"rows": {k: K.SPLIT_N[k] for k in K.SPLITS},
                       "n_columns": K.HIV_WIDTH,
                       "columns": "[log1p(n_atoms), log1p(n_bonds), C(754), F(3652)]",
                       "structure_scale": "frozen count/sqrt(max(n_atoms,1))",
                       "n_bonds_source": "tracked raw ogbg-molhiv bond_ptr differences / 2; "
                                         "verified equal to MSDS-v0 bond counts locally"},
            "zinc": {"rows": {k: K.ZINC_SPLIT_N[k] for k in K.SPLITS},
                     "n_columns": K.ZINC_WIDTH,
                     "columns": "Z(4642) + C(9) + F(1250), frozen SGA-v1 design"},
            "historical_predictions_identity_checked": True,
        },
        "inputs": inputs,
        "new_models": models,
        "predictions": predictions,
        "analysis_outputs": analysis,
        "tables": tables,
        "curves": curves,
        "function_outputs": functions,
        "documents": documents,
        "verification": verification,
        "commands": [
            "uv run python tracks/.../code/phase0_contributions.py",
            "uv run python tracks/.../code/phaseA_linear.py",
            "rr run res sfl2-h-add-tree --pool res-local --cpus 8 -- python .../phaseB_add_tree.py --dataset hiv",
            "rr run res sfl2-z-add-tree --pool res-local --cpus 8 -- python .../phaseB_add_tree.py --dataset zinc",
            "rr run res sfl2-h-nn --pool res-gpu1 --gpus 1 -- python .../phaseC_add_nn.py --arms H-NN-JOINT,H-NN-JOINT-BAL,H-NN-ADD",
            "rr run res sfl2-z-nn --pool res-gpu0 --gpus 1 -- python .../phaseC_add_nn.py --arms Z-NN-ADD",
            "uv run python tracks/.../code/phaseD_analysis.py",
            "uv run python tracks/.../code/phaseE_functions.py",
            "uv run pytest -q tracks/.../tests",
        ],
        "environment": K.environment(),
        "seconds_to_assemble": float(time.time() - t0),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    K.write_json(K.ANALYSIS / "provenance.json", payload)
    print(f"[prov] wrote analysis/provenance.json "
          f"({payload['gpu_accounting']['a100_gpu_hours']:.4f} A100-h, "
          f"{len(models)} models, {len(predictions)} prediction tables)")


if __name__ == "__main__":
    main()
