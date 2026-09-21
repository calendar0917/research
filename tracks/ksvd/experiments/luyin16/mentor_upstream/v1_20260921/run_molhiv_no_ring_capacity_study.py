#!/usr/bin/env python3
"""Run the MolHIV no-ring semantic-cross capacity study.

For each requested dictionary size this controller reproduces the historical
seed-0 structural dictionary protocol, builds the capacity-specific 693-D
composition+reconstructed-typed backbone, and evaluates that backbone plus the
three existing semantic-cross views with XGBoost seeds 0--9 on the official
OGB scaffold split.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PATCH_DIR = ROOT / "results/molhiv_historical_m13_seed0/cache_M13_q999"
COMPOSITION = ROOT / "results/molhiv_restored_legacy_v1/composition_69_exact.npz"
REFERENCE = (
    ROOT
    / "results/molhiv_historical_seed0_features_ablation_6views_xgb_10seed/auc_objective_summary.json"
)
RESULT_ROOT = ROOT / "results/molhiv_no_ring_capacity_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capacities", nargs="+", type=int, default=[16, 32, 128])
    parser.add_argument("--sparsity", type=int, default=8)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    return parser.parse_args()


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def complete(paths: list[Path]) -> bool:
    return all(path.is_file() for path in paths)


def run_stage(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("$ " + subprocess.list2cmdline(command), flush=True)
    print(f"  log: {log_path}", flush=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n=== START {datetime.now().isoformat()} ===\n")
        handle.write("$ " + subprocess.list2cmdline(command) + "\n")
        handle.flush()
        result = subprocess.run(
            command,
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        handle.write(
            f"=== END {datetime.now().isoformat()} exit={result.returncode} ===\n"
        )
    if result.returncode:
        raise RuntimeError(f"Stage failed ({result.returncode}); see {log_path}")


def summarize(result_root: Path, capacities: list[int], sparsity: int) -> None:
    payload: dict[str, object] = {
        "dataset": "ogbg-molhiv",
        "protocol": "official scaffold split; dictionary seed 0; XGBoost seeds 0-9; fixed historical parameters",
        "sparsity": sparsity,
        "capacities": {},
    }
    # Keep the already completed K=64 experiment in the same comparison file.
    sources = {
        64: ROOT / "results/molhiv_no_ring_semantic_cross_v1/summary.json",
        **{
            k: result_root / f"K{k}_s{sparsity}" / "evaluation/summary.json"
            for k in capacities
        },
    }
    for k, path in sorted(sources.items()):
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            payload["capacities"][str(k)] = {
                "source": str(path.resolve()),
                "baseline": data.get("baseline"),
                "views": data.get("views", {}),
                "best_by_validation": data.get("best_by_validation"),
            }
    save_json(result_root / "capacity_summary.json", payload)


def main() -> None:
    args = parse_args()
    capacities = list(dict.fromkeys(int(k) for k in args.capacities))
    sparsity = int(args.sparsity)
    if not capacities or any(k <= 0 or sparsity <= 0 or sparsity > k for k in capacities):
        raise ValueError(f"Invalid capacities/sparsity: {capacities}, s={sparsity}")
    result_root = args.result_root.resolve()
    result_root.mkdir(parents=True, exist_ok=True)
    save_json(
        result_root / "run_manifest.json",
        {
            "capacities": capacities,
            "sparsity": sparsity,
            "dictionary_seed": 0,
            "classifier_seeds": args.seeds,
            "official_split": True,
            "dictionary_selection_monitor": "valid (historical reproduction)",
            "cross_modes": ["patch_mass", "root_cond", "mass_root"],
            "started": datetime.now().isoformat(),
        },
    )

    for k in capacities:
        local = result_root / f"K{k}_s{sparsity}"
        ksvd_dir = local / "ksvd"
        diagnostic_dir = local / "diagnostic"
        evaluation_dir = local / "evaluation"
        cross_dir = local / "cross_cache"
        logs = local / "logs"
        for path in (ksvd_dir, diagnostic_dir, evaluation_dir, cross_dir, logs):
            path.mkdir(parents=True, exist_ok=True)

        if not complete(
            [
                ksvd_dir / "dictionary_scaled.npy",
                ksvd_dir / "dictionary_raw.npy",
                ksvd_dir / "code_indices.npy",
                ksvd_dir / "code_values.npy",
                ksvd_dir / "code_nnz.npy",
                ksvd_dir / "summary.json",
            ]
        ):
            run_stage(
                [
                    sys.executable,
                    "-u",
                    str(ROOT / "molhiv_online_structural_ksvd_full.py"),
                    "train",
                    "--cache-dir",
                    str(PATCH_DIR),
                    "--out-dir",
                    str(ksvd_dir),
                    "--seed",
                    "0",
                    "--n-atoms",
                    str(k),
                    "--sparsity",
                    str(sparsity),
                    "--epochs",
                    "5",
                    "--graphs-per-batch",
                    "16",
                    "--patches-per-graph",
                    "8",
                    "--dictionary-sweeps",
                    "1",
                    "--svd-mode",
                    "power",
                    "--power-iterations",
                    "6",
                    "--replay-capacity",
                    "65536",
                    "--replay-per-batch",
                    "192",
                    "--init-patches-per-graph",
                    "4",
                    "--init-max-patches",
                    "65536",
                    "--block-scaling",
                    "equal_energy",
                    "--eval-every",
                    "1",
                    "--selection-monitor",
                    "valid",
                    "--monitor-train-graphs",
                    "500",
                    "--monitor-valid-graphs",
                    "500",
                    "--eval-patches-per-graph",
                    "8",
                ],
                logs / "01_ksvd.log",
            )

        if not complete(
            [
                diagnostic_dir / "recon_decoded_typed_pool.npy",
                diagnostic_dir / "diagnostic_metadata.npz",
                diagnostic_dir / "build_manifest.json",
            ]
        ):
            run_stage(
                [
                    sys.executable,
                    "-u",
                    str(ROOT / "run_molhiv_ksvd_three_diagnostics.py"),
                    "build",
                    "--ksvd-script",
                    str(ROOT / "molhiv_online_structural_ksvd_full.py"),
                    "--ksvd-cache-dir",
                    str(PATCH_DIR),
                    "--ksvd-result-dir",
                    str(ksvd_dir),
                    "--expected-atoms",
                    str(k),
                    "--expected-sparsity",
                    str(sparsity),
                    "--composition-cache",
                    str(COMPOSITION),
                    "--out-dir",
                    str(diagnostic_dir),
                    "--edge-threshold",
                    "0.5",
                    "--progress-every",
                    "100",
                ],
                logs / "02_diagnostics.log",
            )

        run_stage(
            [
                sys.executable,
                "-u",
                str(ROOT / "run_molhiv_no_ring_semantic_cross.py"),
                "--patch-dir",
                str(PATCH_DIR),
                "--code-dir",
                str(ksvd_dir),
                "--diagnostic-dir",
                str(diagnostic_dir),
                "--reference-summary",
                str(REFERENCE),
                "--result-dir",
                str(evaluation_dir),
                "--cross-dir",
                str(cross_dir),
                "--expected-atoms",
                str(k),
                "--expected-sparsity",
                str(sparsity),
                "--seeds",
                *[str(seed) for seed in args.seeds],
                "--include-base",
                "--n-jobs",
                str(args.n_jobs),
                "--checkpoint-every",
                "100",
                "--resume",
            ],
            logs / "03_evaluation.log",
        )
        summarize(result_root, capacities, sparsity)

    summarize(result_root, capacities, sparsity)
    save_json(
        result_root / "COMPLETE.json",
        {"completed": datetime.now().isoformat(), "capacities": capacities},
    )
    print("MolHIV no-ring capacity study complete.", flush=True)


if __name__ == "__main__":
    main()
