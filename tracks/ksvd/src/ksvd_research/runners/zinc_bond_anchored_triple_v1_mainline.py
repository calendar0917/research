"""Runner: BondAnchoredTriple-v1 mainline performance screen.

Implements
``tracks/ksvd/notes/e2e_dictenv_bond_anchored_triple_v1_preregistration.md``:
the frozen ``CSSD-Sem108 + C6`` parent soup, its cached pair tokens and Reader
input, the exact bond-anchored triple objects, and one seed-0 80-epoch run of
the new ``F`` + re-initialised Reader path with a fixed Top-5 soup.

CPU-first (the repository execution regime).  The official ZINC **valid** split
is evaluation only and the official **test** split is never instantiated; a
granted ``test_access`` is refused.
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
    e2e_dictenv_bond_anchored_triple_v1 as bat,
)
from tracks.ksvd.experiments.luyin16 import (  # noqa: E402
    zinc_bond_anchored_triple_v1_mainline as stages,
)

EXPECTED_SPLIT_SIZES = {"train": 10_000, "val": 1_000, "test": 1_000}
DEFAULT_CONFIG = TRACK_ROOT / "configs/luyin16/zinc_bond_anchored_triple_v1_mainline.yaml"


def build_runner():
    from ksvd_research.runner_api import Runner

    return Runner(
        name="zinc_bond_anchored_triple_v1_mainline",
        default_config=DEFAULT_CONFIG,
        default_study="zinc-context-gap",
        description=(
            "BondAnchoredTriple-v1: frozen Sem108+C6 parent, three-environment "
            "common-endpoint static composition, 80-epoch seed-0 screen, CPU"
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
            "BondAnchoredTriple-v1 is a non-terminal screen; it must never run "
            "with test access granted"
        )
    if not torch.cuda.is_available():
        torch.set_num_threads(int(config.get("runtime", {}).get("torch_threads", bat.THREADS)))

    stages.stage_references()
    stages.stage_preflight()
    stages.stage_cache()
    stages.stage_standardizers()
    stages.stage_correctness()
    run_payload = stages.stage_train()
    summary = stages.stage_analysis(run_id=context.run_id)

    context.artifact_dir.mkdir(parents=True, exist_ok=True)
    (context.artifact_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    for source, name in (
        (stages.RESULTS_DIR / "references.json", "references.json"),
        (stages.RESULTS_DIR / "preflight.json", "preflight.json"),
        (stages.RESULTS_DIR / "cache_report.json", "cache_report.json"),
        (stages.RESULTS_DIR / "standardizers.json", "standardizers.json"),
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
        "M_soup": float(summary["M_soup"]),
        "M_parent_replay": float(summary["M_parent_replay"]),
        "M_parent_recorded": float(summary["M_parent_recorded"]),
        "delta_soup_minus_parent": float(summary["delta_soup_minus_parent"]),
        "best_valid_mae": float(summary["best_valid_mae"]),
        "best_epoch": int(summary["best_epoch"]),
        "last_20_valid_mean": float(summary["last_20_valid_mean"]),
        "final_train_mae": float(summary["final_train_mae"]),
        "soup_members": list(summary["soup_members"]),
        "epochs_run": int(summary["epochs_run"]),
        "completed": bool(summary["completed"]),
        "verdict": str(summary["verdict"]),
        "correctness_all_passed": bool(summary["correctness_all_passed"]),
        "parent_state_unchanged": bool(summary["parent_state_unchanged"]),
        "parent_replay_passed": bool(
            stages._read_json(stages.RESULTS_DIR / "references.json")["replay"]["passed"]
        ),
        "trainable_params": int(summary["parameter_accounting"]["trainable_params"]),
        "full_model_params": int(summary["parameter_accounting"]["full_model_params"]),
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
        f"[bond-anchored-triple] verdict={metrics['verdict']} "
        f"M_soup={metrics['M_soup']:.9f} M_parent={metrics['M_parent_replay']:.9f} "
        f"delta={metrics['delta_soup_minus_parent']:+.9f}",
        flush=True,
    )
    return RunResult(
        metrics=metrics,
        status="completed",
        artifacts=[
            "artifacts/summary.json",
            "artifacts/metrics.json",
            "artifacts/references.json",
            "artifacts/preflight.json",
            "artifacts/cache_report.json",
            "artifacts/standardizers.json",
            "artifacts/correctness.json",
            "artifacts/run_seed0.json",
            "artifacts/curve_seed0.csv",
            "artifacts/soup.json",
            "artifacts/REPORT.md",
            "artifacts/DECISION.md",
        ],
    )


__all__ = ["build_runner", "fingerprints", "run", "stages"]
