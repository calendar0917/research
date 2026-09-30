"""Runner: JointBond-Decay-Diagnostic-v1 (fixed-parent branch-trainability probe).

Implements
``tracks/ksvd/notes/e2e_dictenv_jointbond_decay_diagnostic_v1_preregistration.md``:
the frozen ``CSSD-Sem108`` soup as an un-updated parent, a fresh v1 JointBond
branch initialisation, and two 200-step arms that differ **only** in Adam's
weight decay (``1e-5`` coupled L2 vs ``0.0``) on one repeated 128-graph
official-train fit batch.

This is a short diagnostic, not a performance round: no full training, no soup,
no second seed, no architecture change.  CPU only with 8 threads; the official
ZINC **valid** and **test** splits are never loaded and a granted
``test_access`` is refused.
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
    zinc_jointbond_decay_diagnostic_v1 as stages,
)

EXPECTED_SPLIT_SIZES = {"train": 10_000, "val": 1_000, "test": 1_000}
DEFAULT_CONFIG = TRACK_ROOT / "configs/luyin16/zinc_jointbond_decay_diagnostic_v1.yaml"


def build_runner():
    from ksvd_research.runner_api import Runner

    return Runner(
        name="zinc_jointbond_decay_diagnostic_v1",
        default_config=DEFAULT_CONFIG,
        default_study="zinc-context-gap",
        description="JointBond-Decay-Diagnostic-v1 (fixed parent, WD vs NO-WD, CPU)",
        run_module=sys.modules[__name__],
    )


def fingerprints(config: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    data_root = resolve_path(config["data"]["root"])
    payload = zinc_fingerprints(data_root, expected_sizes=EXPECTED_SPLIT_SIZES)
    return payload, payload["split_fingerprint"]


def run(config: Mapping[str, Any], context: RunContext) -> RunResult:
    if context.test_access == "granted":
        raise RuntimeError(
            "JointBond-Decay-Diagnostic-v1 is a diagnostic; it must never run with test access granted"
        )
    if not torch.cuda.is_available():
        torch.set_num_threads(int(config.get("runtime", {}).get("torch_threads", stages.THREADS)))

    stages.stage_references()
    stages.stage_preflight()
    stages.stage_correctness()
    run_payload = stages.stage_run()
    summary = stages.stage_analysis()

    preflight = stages._read_json(stages.RESULTS_DIR / "preflight.json")
    correctness = stages._read_json(stages.RESULTS_DIR / "correctness.json")
    data = stages._read_json(stages.RESULTS_DIR / "data.json")

    context.artifact_dir.mkdir(parents=True, exist_ok=True)
    (context.artifact_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    for source, name in (
        (stages.RESULTS_DIR / "preflight.json", "preflight.json"),
        (stages.RESULTS_DIR / "correctness.json", "correctness.json"),
        (stages.RESULTS_DIR / "data.json", "data.json"),
        (stages.RESULTS_DIR / "decay_diagnostic.json", "decay_diagnostic.json"),
        (stages.RESULTS_DIR / "records_wd.json", "records_wd.json"),
        (stages.RESULTS_DIR / "records_nowd.json", "records_nowd.json"),
        (stages.RESULTS_DIR / "trace_wd.csv", "trace_wd.csv"),
        (stages.RESULTS_DIR / "trace_nowd.csv", "trace_nowd.csv"),
        (stages.RESULTS_DIR / "REPORT.md", "REPORT.md"),
    ):
        if source.exists():
            (context.artifact_dir / name).write_text(
                Path(source).read_text(encoding="utf-8"), encoding="utf-8"
            )

    metrics: dict[str, Any] = {
        "measure": "d_fit_mae",
        "wd_d_fit": float(summary["per_arm"]["wd"]["d_fit"]),
        "nowd_d_fit": float(summary["per_arm"]["nowd"]["d_fit"]),
        "wd_fit_mae_step0": float(summary["per_arm"]["wd"]["fit_mae_step0"]),
        "wd_fit_mae_final": float(summary["per_arm"]["wd"]["fit_mae_final"]),
        "nowd_fit_mae_step0": float(summary["per_arm"]["nowd"]["fit_mae_step0"]),
        "nowd_fit_mae_final": float(summary["per_arm"]["nowd"]["fit_mae_final"]),
        "nowd_probe_mae_final": float(summary["per_arm"]["nowd"]["probe_mae_final"]),
        "verdict": str(summary["verdict"]),
        "nowd_learns": bool(summary["questions"]["1_branch_learns_fit"]["nowd_learns"]),
        "wd_learns": bool(summary["questions"]["1_branch_learns_fit"]["wd_learns"]),
        "parent_state_unchanged": bool(all(run_payload["parent_state_unchanged"].values())),
        "coupled_l2_confirmed": bool(
            correctness["gates"]["D8_coupled_l2_not_adamw"]["coupled_l2_confirmed"]
        ),
        "correctness_all_passed": bool(correctness["all_passed"]),
        "branch_params": int(summary["per_arm"]["wd"]["optimizer_numel"]),
        "steps_per_arm": int(preflight["steps_per_arm"]),
        "fit_size": int(preflight["fit_size"]),
        "probe_size": int(preflight["probe_size"]),
        "fit_fingerprint": str(data["fit_fingerprint"]["sha256"]),
        "probe_fingerprint": str(data["probe_fingerprint"]["sha256"]),
        "wall_clock_s": float(run_payload["wall_clock_s"]),
        "split_sizes": {"train": 10_000, "valid": None, "test": None},
        "test_access": context.test_access,
        "official_test_loaded": False,
        "seed": int(preflight.get("seed", stages.SEED)),
    }
    (context.artifact_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(
        f"[jointbond-decay] verdict={metrics['verdict']} "
        f"d_fit(wd)={metrics['wd_d_fit']:+.3e} d_fit(nowd)={metrics['nowd_d_fit']:+.3e}",
        flush=True,
    )
    return RunResult(
        metrics=metrics,
        status="completed",
        artifacts=[
            "artifacts/summary.json",
            "artifacts/metrics.json",
            "artifacts/preflight.json",
            "artifacts/correctness.json",
            "artifacts/data.json",
            "artifacts/decay_diagnostic.json",
            "artifacts/records_wd.json",
            "artifacts/records_nowd.json",
            "artifacts/trace_wd.csv",
            "artifacts/trace_nowd.csv",
            "artifacts/REPORT.md",
        ],
    )


__all__ = ["build_runner", "fingerprints", "run", "stages"]
