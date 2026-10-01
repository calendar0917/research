"""Runner: E2E-DictEnv-RoleCorr-Increment-v2 (appended-block increment screen).

Implements
``tracks/ksvd/notes/e2e_dictenv_rolecorr_increment_v2_preregistration.md``.

Route 1 (``model.stage=route1``) keeps the full frozen K32/s8 structural budget
and appends one 16-wide frozen block at coordinate width 49:

* ``EXTRA-STRUCT``  A — frozen RoleCorr K16/s4 structural residual block;
* ``CORR-ADD``      B — frozen RoleCorr scaler + K16/s4 correspondence block;
* ``CORR-PCA-ADD``  C — train-fitted PCA16 of the same correspondence object.

Route 2 (``model.stage=route2``, conditional) prepares the frozen K48/s12
dictionary over the train-only scale-balanced 709-D joint input plus the PCA48
control; the route-2 training/inference stages are only implemented once the
route-1 gate misses and the round is explicitly extended.

The official ZINC **test** split is never instantiated; a granted
``test_access`` is refused by this runner.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Mapping

from ksvd_research.runtime.fingerprints import zinc_fingerprints
from ksvd_research.runtime.manifest import RunContext, RunResult
from ksvd_research.runtime.paths import REPO_ROOT, TRACK_ROOT, resolve_path

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tracks.ksvd.experiments.luyin16 import (  # noqa: E402
    zinc_e2e_dictenv_rolecorr_increment_v2 as stages,
)

EXPECTED_SPLIT_SIZES = {"train": 10_000, "val": 1_000, "test": 1_000}
DEFAULT_CONFIG = TRACK_ROOT / "configs/luyin16/zinc_e2e_dictenv_rolecorr_increment_v2.yaml"

ARTIFACTS = (
    "cache_check.json",
    "verify_reused.json",
    "pca16.json",
    "correctness.json",
    "smoke.json",
    "run_EXTRA-STRUCT.json",
    "run_CORR-ADD.json",
    "run_CORR-PCA-ADD.json",
    "curve_EXTRA-STRUCT.csv",
    "curve_CORR-ADD.csv",
    "curve_CORR-PCA-ADD.csv",
    "soup_EXTRA-STRUCT.json",
    "soup_CORR-ADD.json",
    "soup_CORR-PCA-ADD.json",
    "interventions.json",
    "summary.json",
    "REPORT.md",
    "DECISION.md",
    "joint_standardizers.json",
    "joint_objects.json",
    "joint_pca48.json",
)


def build_runner():
    from ksvd_research.runner_api import Runner

    return Runner(
        name="zinc_e2e_dictenv_rolecorr_increment_v2",
        default_config=DEFAULT_CONFIG,
        default_study="zinc-context-gap",
        description=(
            "RoleCorr-Increment-v2: width-49 appended-block screen over the full "
            "frozen K32/s8 structural budget (extra-structural vs correspondence "
            "vs PCA16), device-parametric (cpu/cuda), official-test blocked"
        ),
        run_module=sys.modules[__name__],
    )


def fingerprints(config: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    data_root = resolve_path(config["data"]["root"])
    payload = zinc_fingerprints(data_root, expected_sizes=EXPECTED_SPLIT_SIZES)
    return payload, payload["split_fingerprint"]


def _config_stage(config: Mapping[str, Any]) -> str:
    model = config.get("model", {})
    stage = str(model.get("stage", "route1"))
    if stage not in ("route1", "route2"):
        raise ValueError(f"model.stage must be 'route1' or 'route2', got {stage!r}")
    return stage


def _copy_artifacts(context: RunContext) -> list[str]:
    context.artifact_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for name in ARTIFACTS:
        source = stages.RESULTS_DIR / name
        if not source.exists():
            continue
        (context.artifact_dir / name).write_text(
            source.read_text(encoding="utf-8"), encoding="utf-8"
        )
        copied.append(f"artifacts/{name}")
    for name in ("per_molecule_errors.npz",):
        source = stages.RESULTS_DIR / name
        if source.exists():
            (context.artifact_dir / name).write_bytes(source.read_bytes())
            copied.append(f"artifacts/{name}")
    return copied


def _metrics(summary: Mapping[str, Any], stage: str, device: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "measure": "soup_valid_mae",
        "stage": stage,
        "seed": int(stages.SEED),
        "device": str(device),
        "epochs": int(summary.get("epochs", stages.TRAIN_EPOCHS)),
        "coordinate_dim": int(summary.get("coordinate_dim", 49)),
        "split_sizes": {"train": 10_000, "valid": 1_000, "test": None},
        "official_valid_loaded": True,
        "official_test_loaded": False,
        "verdict": str(summary.get("verdict", "INCOMPLETE")),
        "correctness_all_passed": bool(summary.get("correctness_all_passed", False)),
    }
    for key in (
        "M_A_soup",
        "M_B_soup",
        "M_C_soup",
        "M_A_best",
        "M_B_best",
        "M_C_best",
        "improvement_B_minus_A",
        "screen_abs_gate",
        "screen_gate_fired",
    ):
        if key in summary:
            metrics[key] = summary[key]
    metrics["valid_mae"] = summary.get("M_B_soup")
    metrics["valid_soup_mae"] = summary.get("M_B_soup")
    metrics["valid_best_mae"] = summary.get("M_B_best")
    mechanism = summary.get("mechanism") or {}
    for key in (
        "G_block_zero_B",
        "G_block_shuffle_B",
        "G_object_shuffle_B",
        "block_load_bearing_B",
    ):
        if key in mechanism:
            metrics[key] = mechanism[key]
    if "sparse_vs_dense" in summary:
        metrics["sparse_advantage"] = bool(summary["sparse_vs_dense"]["sparse_advantage"])
        metrics["B_minus_C"] = float(summary["sparse_vs_dense"]["B_minus_C"])
    return metrics


def run(config: Mapping[str, Any], context: RunContext) -> RunResult:
    if context.test_access == "granted":
        raise RuntimeError(
            "RoleCorr-Increment-v2 is a non-terminal screen; it must never run "
            "with test access granted"
        )
    stage = _config_stage(config)
    runtime = config.get("runtime", {})
    device = str(runtime.get("device", stages.DEFAULT_DEVICE))
    configured = stages.configure(
        epochs=int(config.get("model", {}).get("epochs", stages.TRAIN_EPOCHS)),
        threads=int(runtime.get("torch_threads", stages.THREADS)),
        device=device,
    )
    print(
        f"[increment] stage={stage} device={configured['device']} "
        f"epochs={configured['epochs']} commit={stages._git_commit()}",
        flush=True,
    )
    if stage == "route1":
        stages.run_route1(list(stages.ROUTE1_STAGES))
    else:
        stages.run_route2(list(stages.ROUTE2_STAGES))

    summary_path = stages.RESULTS_DIR / "summary.json"
    if summary_path.exists():
        summary = stages._read_json(summary_path)
    else:
        summary = {"verdict": "INCOMPLETE", "device": configured["device"]}
    artifacts = _copy_artifacts(context)
    metrics = _metrics(summary, stage, configured["device"])
    (context.artifact_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    artifacts.append("artifacts/metrics.json")
    print(f"[increment] verdict={metrics['verdict']}", flush=True)
    return RunResult(metrics=metrics, status="completed", artifacts=artifacts)


__all__ = ["build_runner", "fingerprints", "run", "stages"]
