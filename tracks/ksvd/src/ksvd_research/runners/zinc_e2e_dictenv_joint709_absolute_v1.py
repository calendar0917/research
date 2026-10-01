"""Runner: E2E-DictEnv-Joint709-Absolute-v1 (absolute-performance CPU screen).

One candidate only: the frozen K48/s12 joint environment dictionary over the
train-only scale-balanced 709-D input (structural residual 65 + Sem108 108 +
RoleCorr correspondence 536), coordinate ``[c~ ; IHT_10(alpha_48)]`` width 49.
Everything else is the frozen Sem108 parent (C6 mask, bindings, fusion,
relation features, pooling, backend, reader, optimiser, 320 epochs, Top-5 soup).

``model.stage=prepare`` builds the train-only scaler / caches / frozen
dictionary / reconstruction diagnostic (scratch use); ``model.stage=screen``
runs the correctness gates, the smoke and the formal seed-0 training.

This round has **no control arm**: no PCA48, no random / shuffled /
no-dictionary training.  The official ZINC **test** split is never
instantiated; a granted ``test_access`` is refused by this runner.
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
    zinc_e2e_dictenv_joint709_absolute_v1 as stages,
)

EXPECTED_SPLIT_SIZES = {"train": 10_000, "val": 1_000, "test": 1_000}
DEFAULT_CONFIG = TRACK_ROOT / "configs/luyin16/zinc_e2e_dictenv_joint709_absolute_v1.yaml"

ARTIFACTS = (
    "verify_frozen.json",
    "joint_standardizers.json",
    "joint_scaler_check.json",
    "cache_check_joint.json",
    "joint_objects.json",
    "reconstruction.json",
    "correctness.json",
    "smoke.json",
    "run_JOINT-SPARSE.json",
    "curve_JOINT-SPARSE.csv",
    "soup_JOINT-SPARSE.json",
    "interventions.json",
    "summary.json",
    "REPORT.md",
    "DECISION.md",
)


def build_runner():
    from ksvd_research.runner_api import Runner

    return Runner(
        name="zinc_e2e_dictenv_joint709_absolute_v1",
        default_config=DEFAULT_CONFIG,
        default_study="zinc-context-gap",
        description=(
            "Joint709-Absolute-v1: single-candidate absolute-performance CPU screen of "
            "the frozen K48/s12 dictionary over the train-only balanced 709-D joint "
            "input (struct 65 + Sem108 108 + RoleCorr 536), width 49, official-test blocked"
        ),
        run_module=sys.modules[__name__],
    )


def fingerprints(config: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    data_root = resolve_path(config["data"]["root"])
    payload = zinc_fingerprints(data_root, expected_sizes=EXPECTED_SPLIT_SIZES)
    return payload, payload["split_fingerprint"]


def _config_stage(config: Mapping[str, Any]) -> str:
    model = config.get("model", {})
    stage = str(model.get("stage", "screen"))
    if stage not in ("prepare", "screen", "all"):
        raise ValueError(f"model.stage must be 'prepare', 'screen' or 'all', got {stage!r}")
    return stage


def _copy_artifacts(context: RunContext) -> list[str]:
    context.artifact_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for name in ARTIFACTS:
        source = stages.RESULTS_DIR / name
        if not source.exists():
            continue
        (context.artifact_dir / name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        copied.append(f"artifacts/{name}")
    for name in ("per_molecule_errors.npz",):
        source = stages.RESULTS_DIR / name
        if source.exists():
            (context.artifact_dir / name).write_bytes(source.read_bytes())
            copied.append(f"artifacts/{name}")
    return copied


def _metrics(stage: str, device: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "measure": "soup_valid_mae",
        "stage": stage,
        "seed": int(stages.SEED),
        "device": str(device),
        "coordinate_dim": int(stages.core.COORD_DIM),
        "joint_dim": int(stages.core.JOINT_DIM),
        "iht_steps": int(stages.core.IHT_STEPS),
        "s": int(stages.core.JOINT_SPARSITY),
        "split_sizes": {"train": 10_000, "valid": 1_000, "test": None},
        "official_valid_loaded": True,
        "official_test_loaded": False,
    }
    summary_path = stages.RESULTS_DIR / "summary.json"
    if summary_path.exists():
        summary = stages._read_json(summary_path)
        band = summary.get("band", {})
        metrics.update(
            {
                "verdict": str(summary.get("verdict", "INCOMPLETE")),
                "epochs": int(summary.get("epochs", stages.TRAIN_EPOCHS)),
                "epochs_run": int(summary.get("epochs_run", 0)),
                "completed": bool(summary.get("completed", False)),
                "valid_mae": float(summary.get("soup_valid_mae", float("nan"))),
                "valid_soup_mae": float(summary.get("soup_valid_mae", float("nan"))),
                "valid_best_mae": float(summary.get("best_valid_mae", float("nan"))),
                "best_epoch": int(summary.get("best_epoch", 0)),
                "absolute_band": str(band.get("band", "unknown")),
                "absolute_recommendation": str(band.get("recommendation", "")),
                "soup_train_mae": float(summary.get("soup_train_mae", float("nan"))),
                "train_valid_gap": float(summary.get("train_valid_gap", float("nan"))),
                "zero_joint_code_delta": float(
                    summary.get("zero_joint_code", {}).get("delta", float("nan"))
                ),
                "row_shuffle_mean_delta": float(
                    summary.get("row_shuffle_mean_delta", float("nan"))
                ),
                "reconstruction_valid_rel_err": float(
                    summary.get("reconstruction", {}).get(
                        "valid_relative_error_tied_iht", float("nan")
                    )
                ),
                "active_atoms": int(summary.get("code_usage", {}).get("active_atoms", 0)),
                "effective_atoms": float(
                    summary.get("code_usage", {}).get("effective_atoms", float("nan"))
                ),
                "correctness_all_passed": bool(summary.get("correctness_all_passed", False)),
                "smoke_passed": bool(summary.get("smoke_passed", False)),
                "trainable_parameters": int(
                    summary.get("parameters", {}).get("trainable", 0)
                ),
                "total_parameters": int(summary.get("parameters", {}).get("total", 0)),
                "wall_clock_s": float(summary.get("wall_clock_s", float("nan"))),
            }
        )
    else:
        metrics["verdict"] = "PREPARE_OR_INCOMPLETE"
    return metrics


def run(config: Mapping[str, Any], context: RunContext) -> RunResult:
    if context.test_access == "granted":
        raise RuntimeError(
            "Joint709-Absolute-v1 is a non-terminal screen; it must never run "
            "with test access granted"
        )
    stage = _config_stage(config)
    runtime = config.get("runtime", {})
    device = str(runtime.get("device", "cpu"))
    configured = stages.configure(
        epochs=int(config.get("model", {}).get("epochs", stages.TRAIN_EPOCHS)),
        threads=int(runtime.get("torch_threads", stages.THREADS)),
        device=device,
    )
    print(
        f"[joint709] stage={stage} device={configured['device']} "
        f"epochs={configured['epochs']} threads={configured['threads']} "
        f"commit={stages._git_commit()}",
        flush=True,
    )
    if stage in ("prepare", "all"):
        stages.run_stages("prepare")
    if stage in ("screen", "all"):
        stages.run_stages("screen")
    artifacts = _copy_artifacts(context)
    metrics = _metrics(stage, configured["device"])
    (context.artifact_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    artifacts.append("artifacts/metrics.json")
    print(f"[joint709] verdict={metrics['verdict']}", flush=True)
    return RunResult(metrics=metrics, status="completed", artifacts=artifacts)


__all__ = ["build_runner", "fingerprints", "run", "stages"]
