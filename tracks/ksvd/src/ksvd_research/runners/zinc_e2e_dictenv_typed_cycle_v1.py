"""Runner: E2E-DictEnv-Typed-Cycle-v1 (static typed chordless-cycle object, ZINC CPU).

One candidate only: the frozen Small ``LatentBridgeSEM108`` (106,925 params)
plus a shared typed-cycle encoder whose per-ring latent is decoded by the
**same** ``D_L``/``V_L`` task dictionary; the pooled ring code (97) enters the
same reader (399 -> 13 -> 13 -> 1).  Exactly 133,002 parameters, no second
dictionary, no message passing, no write-back.

``model.stage=screen`` runs the ring cache, the 8 acceptance gates, the CPU
timing gate, the 24-step smoke, one seed-0 320-epoch CPU run, the frozen
inference probes and the analysis.  The official ZINC **test** split is never
instantiated; a granted ``test_access`` is refused.
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
    zinc_e2e_dictenv_typed_cycle_v1 as stages,
)

EXPECTED_SPLIT_SIZES = {"train": 10_000, "val": 1_000, "test": 1_000}
DEFAULT_CONFIG = TRACK_ROOT / "configs/luyin16/zinc_e2e_dictenv_typed_cycle_v1.yaml"

ARTIFACTS = (
    "summary.json",
    "REPORT.md",
    "DECISION.md",
    "screen.json",
    "preflight.json",
    "preregistration_snapshot.json",
    "timing.json",
    "smoke.json",
    "run_seed0.json",
    "train_skipped.json",
)


def build_runner():
    from ksvd_research.runner_api import Runner

    return Runner(
        name="zinc_e2e_dictenv_typed_cycle_v1",
        default_config=DEFAULT_CONFIG,
        default_study="zinc-context-gap",
        description=(
            "Typed-cycle-v1: frozen Small + shared-dictionary typed ring objects "
            "(133,002 params), seed-0 320-epoch CPU screen, test blocked"
        ),
        run_module=sys.modules[__name__],
    )


def fingerprints(config: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    data_root = resolve_path(config["data"]["root"])
    payload = zinc_fingerprints(data_root, expected_sizes=EXPECTED_SPLIT_SIZES)
    return payload, payload["split_fingerprint"]


def _copy_artifacts(context: RunContext) -> list[str]:
    context.artifact_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for name in ARTIFACTS:
        source = stages.RESULTS_DIR / name
        if not source.exists():
            continue
        (context.artifact_dir / name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        copied.append(f"artifacts/{name}")
    for name, source in (
        ("correctness.json", stages.AUDIT_DIR / "correctness.json"),
        ("parameter_audit.json", stages.AUDIT_DIR / "parameter_audit.json"),
        ("typed_cycle_probes.json", stages.MECHANISM_DIR / "typed_cycle_probes.json"),
        ("ring_cache.json", stages.CACHE_DIR / "ring_cache.json"),
    ):
        if source.exists():
            (context.artifact_dir / name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
            copied.append(f"artifacts/{name}")
    return copied


def _metrics() -> dict[str, Any]:
    summary_path = stages.RESULTS_DIR / "summary.json"
    metrics: dict[str, Any] = {
        "measure": "soup_valid_mae",
        "stage": "screen",
        "seed": int(stages.SEED),
        "device": "cpu",
        "parameters": 133002,
        "official_valid_loaded": True,
        "official_test_loaded": False,
        "split_sizes": {"train": 10_000, "valid": 1_000, "test": None},
    }
    if summary_path.exists():
        summary = stages._read_json(summary_path)
        metrics.update(
            {
                "verdict": str(summary.get("verdict", "INCOMPLETE")),
                "status": str(summary.get("status", "completed")),
                "completed": bool("M_S" in summary),
                "valid_mae": float(summary.get("M_S", float("nan"))),
                "valid_soup_mae": float(summary.get("M_S", float("nan"))),
                "valid_best_mae": float(summary.get("best_valid_mae", float("nan"))),
                "best_epoch": int(summary.get("best_epoch", 0)),
                "soup_train_mae": float(summary.get("soup_train_mae", float("nan"))),
                "train_valid_gap": float(summary.get("train_valid_gap", float("nan"))),
                "wall_clock_s": float(summary.get("wall_clock_s", float("nan"))),
                "seconds_per_epoch": float(summary.get("seconds_per_epoch", float("nan"))),
                "correctness_all_passed": bool(summary.get("correctness_all_passed", False)),
                "timing_authorized": bool(summary.get("timing", {}).get("formal_run_authorized", False)),
                "zero_ring_delta_mae": float(
                    (summary.get("zero_ring") or {}).get("delta_mae", float("nan"))
                ),
                "ring_encoder_task_grad": float(summary.get("ring_encoder_task_grad", float("nan"))),
            }
        )
    else:
        metrics["verdict"] = "INCOMPLETE"
    return metrics


def run(config: Mapping[str, Any], context: RunContext) -> RunResult:
    if context.test_access == "granted":
        raise RuntimeError(
            "Typed-cycle-v1 is a validation-only screen; it must never run with test access granted"
        )
    model = config.get("model", {})
    stage = str(model.get("stage", "screen"))
    if stage != "screen":
        raise ValueError(f"model.stage must be 'screen', got {stage!r}")
    runtime = config.get("runtime", {})
    stages.configure(
        epochs=int(model.get("epochs", stages.TRAIN_EPOCHS)),
        threads=int(runtime.get("torch_threads", stages.THREADS)),
        device=str(runtime.get("device", "cpu")),
    )
    print(f"[typed-cycle] commit={stages._git_commit()}", flush=True)
    stages.run_stages("all")
    artifacts = _copy_artifacts(context)
    metrics = _metrics()
    (context.artifact_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    artifacts.append("artifacts/metrics.json")
    print(f"[typed-cycle] verdict={metrics['verdict']}", flush=True)
    return RunResult(metrics=metrics, status="completed", artifacts=artifacts)


__all__ = ["build_runner", "fingerprints", "run", "stages"]
