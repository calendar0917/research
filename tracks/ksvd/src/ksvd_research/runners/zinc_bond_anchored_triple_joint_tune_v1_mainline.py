"""Runner: BondAnchoredTriple-JointTune-v1 mainline performance screen.

Implements
``tracks/ksvd/notes/e2e_dictenv_bond_anchored_triple_joint_tune_v1_preregistration.md``:
the frozen Sem108 + C6 dictionary coordinates, the frozen six BAT-v1
normalisation buffers, the BAT-v1 M1 soup hot start (F + 366-D Reader), and one
seed-0 40-epoch run that jointly fine-tunes the edge binding, fusion, pair path
and triple composition + Reader with a fixed Top-5 soup over epochs 21-40.

Local CPU only.  The official ZINC **valid** split is evaluation only and the
official **test** split is never instantiated; a granted ``test_access`` is
refused.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

import torch

from ksvd_research.runtime.fingerprints import zinc_fingerprints
from ksvd_research.runtime.manifest import RunContext, RunResult
from ksvd_research.runtime.paths import REPO_ROOT, TRACK_ROOT, resolve_path

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tracks.ksvd.experiments.luyin16 import (  # noqa: E402
    e2e_dictenv_bond_anchored_triple_joint_tune_v1 as jt,
)
from tracks.ksvd.experiments.luyin16 import (  # noqa: E402
    zinc_bond_anchored_triple_joint_tune_v1_mainline as stages,
)

EXPECTED_SPLIT_SIZES = {"train": 10_000, "val": 1_000, "test": 1_000}
DEFAULT_CONFIG = TRACK_ROOT / "configs/luyin16/zinc_bond_anchored_triple_joint_tune_v1_mainline.yaml"


def build_runner():
    from ksvd_research.runner_api import Runner

    return Runner(
        name="zinc_bond_anchored_triple_joint_tune_v1_mainline",
        default_config=DEFAULT_CONFIG,
        default_study="zinc-context-gap",
        description=(
            "BondAnchoredTriple-JointTune-v1: fixed dictionary coordinates, joint "
            "representation+composition fine-tune from the BAT-v1 M1 soup, 40-epoch "
            "seed-0 screen, local CPU"
        ),
        run_module=sys.modules[__name__],
    )


def fingerprints(config: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    data_root = resolve_path(config["data"]["root"])
    payload = zinc_fingerprints(data_root, expected_sizes=EXPECTED_SPLIT_SIZES)
    return payload, payload["split_fingerprint"]


def run(config: Mapping[str, Any], context: RunContext) -> RunResult:
    if context.test_access == "granted":
        raise RuntimeError(
            "BondAnchoredTriple-JointTune-v1 is a non-terminal screen; it must never "
            "run with test access granted"
        )
    if not torch.cuda.is_available():
        torch.set_num_threads(int(config.get("runtime", {}).get("torch_threads", jt.THREADS)))

    stages.stage_structure()
    stages.stage_references()
    stages.stage_preflight()
    stages.stage_correctness()
    stages.stage_train()
    summary = stages.stage_analysis(run_id=context.run_id)

    context.artifact_dir.mkdir(parents=True, exist_ok=True)
    (context.artifact_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    for source, name in (
        (stages.RESULTS_DIR / "structure_report.json", "structure_report.json"),
        (stages.RESULTS_DIR / "references.json", "references.json"),
        (stages.RESULTS_DIR / "preflight.json", "preflight.json"),
        (stages.RESULTS_DIR / "correctness.json", "correctness.json"),
        (stages.RESULTS_DIR / "run_seed0.json", "run_seed0.json"),
        (stages.RESULTS_DIR / "curve_seed0.csv", "curve_seed0.csv"),
        (stages.RESULTS_DIR / "soup.json", "soup.json"),
        (stages.RESULTS_DIR / "REPORT.md", "REPORT.md"),
        (stages.RESULTS_DIR / "DECISION.md", "DECISION.md"),
    ):
        if source.exists():
            (context.artifact_dir / name).write_text(
                Path(source).read_text(encoding="utf-8"), encoding="utf-8"
            )

    metrics: dict[str, Any] = {
        "measure": "soup_valid_mae",
        "valid_mae": float(summary["M_joint_soup"]),
        "valid_soup_mae": float(summary["M_joint_soup"]),
        "valid_best_mae": float(summary["best_valid_mae"]),
        "M_joint_soup": float(summary["M_joint_soup"]),
        "M_start": float(summary["M_start"]),
        "M_parent": float(summary["M_parent"]),
        "M_parent_recorded": float(summary["M_parent_recorded"]),
        "Delta_vs_start": float(summary["Delta_vs_start"]),
        "Delta_vs_parent": float(summary["Delta_vs_parent"]),
        "best_valid_mae": float(summary["best_valid_mae"]),
        "best_epoch": int(summary["best_epoch"]),
        "last_10_valid_mean": float(summary["last_10_valid_mean"]),
        "final_train_mae": float(summary["final_train_mae"]),
        "soup_members": list(summary["soup_members"]),
        "epochs_run": int(summary["epochs_run"]),
        "completed": bool(summary["completed"]),
        "verdict": str(summary["verdict"]),
        "correctness_all_passed": bool(summary["correctness_all_passed"]),
        "hot_start_passed": bool(summary["hot_start"]["passed"]),
        "hot_start_abs_diff": float(summary["hot_start"]["abs_diff"]),
        "source_files_unchanged": bool(summary["source_files_unchanged"]),
        "frozen_subset_unchanged": bool(summary["frozen_subset_unchanged"]),
        "trainable_params": int(summary["parameter_accounting"]["trainable_params"]),
        "full_model_params": int(summary["parameter_accounting"]["total_registered_params"]),
        "wall_clock_s": float(summary["wall_clock_s"]),
        "seconds_per_epoch": float(summary["seconds_per_epoch"]),
        "peak_rss_mb": float(summary["peak_rss_mb"]),
        "split_sizes": {"train": 10_000, "valid": 1_000, "test": None},
        "test_access": context.test_access,
        "official_test_loaded": False,
        "seed": 0,
        "device": str(summary["device"]),
    }
    (context.artifact_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(
        f"[bond-anchored-triple-joint-tune] verdict={metrics['verdict']} "
        f"M_joint_soup={metrics['M_joint_soup']:.9f} M_start={metrics['M_start']:.9f} "
        f"delta_vs_start={metrics['Delta_vs_start']:+.9f}",
        flush=True,
    )
    return RunResult(
        metrics=metrics,
        status="completed",
        artifacts=[
            "artifacts/summary.json",
            "artifacts/metrics.json",
            "artifacts/structure_report.json",
            "artifacts/references.json",
            "artifacts/preflight.json",
            "artifacts/correctness.json",
            "artifacts/run_seed0.json",
            "artifacts/curve_seed0.csv",
            "artifacts/soup.json",
            "artifacts/REPORT.md",
            "artifacts/DECISION.md",
        ],
    )


__all__ = ["build_runner", "fingerprints", "run", "stages"]
