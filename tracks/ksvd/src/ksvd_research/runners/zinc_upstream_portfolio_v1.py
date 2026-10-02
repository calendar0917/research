"""Runner: E2E-DictEnv-Upstream-Portfolio-v1 (three bounded upstream candidates).

One immutable ``ScaleSpec`` Full family, one shared control and three
single-variation arms (``code`` / ``drop`` / ``r3``).  The official ZINC test
split is never instantiated; a granted ``test_access`` is refused.

``model.stage`` selects the work of one run:

* ``stage0_code`` — frozen-parent native-code-moment export + fixed convex
  conditional fit and the CODE purchase gate (no training);
* ``pilot`` — the shared 80-epoch warm-start pilot of ``model.arm``;
* ``screen`` — the fresh-initialisation 320-epoch screen of ``model.arm``;
* ``audit`` — Small-preset closed-form parameter audit (construction only).
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
    zinc_upstream_portfolio_v1 as stages,
)

EXPECTED_SPLIT_SIZES = {"train": 10_000, "val": 1_000, "test": 1_000}
DEFAULT_CONFIG = TRACK_ROOT / "configs/luyin16/zinc_upstream_portfolio_v1.yaml"

STAGES = ("stage0_code", "pilot", "screen", "audit")


def build_runner():
    from ksvd_research.runner_api import Runner

    return Runner(
        name="zinc_upstream_portfolio_v1",
        default_config=DEFAULT_CONFIG,
        default_study="zinc-context-gap",
        description=(
            "Upstream portfolio v1: control + CODE / DROP / R3 single-variation arms on the "
            "frozen Full scale family; stage0 conditional, 80-epoch warm pilot, 320-epoch "
            "fresh screen; test blocked"
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
            "upstream continuum is non-terminal; it must never run with test access granted"
        )
    model = config.get("model", {})
    stage = str(model.get("stage", "pilot"))
    arm = str(model.get("arm", "control"))
    if stage not in STAGES:
        raise ValueError(f"model.stage must be one of {STAGES}, got {stage!r}")
    runtime = config.get("runtime", {})
    device = stages.resolve_device(str(runtime.get("device", "cpu")))
    if device.type == "cpu":
        import torch

        torch.set_num_threads(int(runtime.get("torch_threads", 8)))
    epochs = int(model.get("epochs", stages.PILOT_EPOCHS if stage == "pilot" else stages.SCREEN_EPOCHS))
    lr = float(model.get("lr", stages.PILOT_LR if stage == "pilot" else stages.SCREEN_LR))
    print(f"[upstream] stage={stage} arm={arm} device={device} epochs={epochs} lr={lr}", flush=True)
    if stage == "stage0_code":
        payload = stages.stage0_code(device)
    elif stage == "pilot":
        payload = stages.pilot(arm, device, epochs=epochs, lr=lr)
    elif stage == "screen":
        payload = stages.screen(arm, device, epochs=epochs, lr=lr)
    else:
        payload = stages.small_parameter_audit()
    stages._write_json(stages.RESULTS_DIR / f"last_{stage}_{arm}.json", payload)
    (context.artifact_dir / "metrics.json").write_text(
        json.dumps({key: value for key, value in payload.items() if key not in ("soup_state", "curve", "valid_predictions_raw", "valid_predictions_calibrated", "valid_targets")}, indent=2, default=str),
        encoding="utf-8",
    )
    metrics = {
        "measure": {
            "stage0_code": "conditional_valid_mae",
            "pilot": "pilot_calibrated_valid_mae",
            "screen": "screen_calibrated_valid_mae",
            "audit": "small_parameter_audit",
        }[stage],
        "stage": stage,
        "arm": arm,
        "device": str(device),
        "official_valid_loaded": True,
        "official_test_loaded": False,
    }
    if stage in ("pilot", "screen"):
        metrics.update(
            {
                "raw_valid_mae": float(payload["raw_valid_mae"]),
                "calibrated_valid_mae": float(payload["calibrated_valid_mae"]),
                "soup_members": payload["members"],
                "wall_clock_s": float(payload["wall_clock_s"]),
                "seconds_per_epoch": float(payload["seconds_per_epoch"]),
                "peak_gpu_memory_mb": float(payload["peak_gpu_memory_mb"]),
                "parameters": int(payload["parameter_audit"]["actual_parameters"]),
            }
        )
    elif stage == "stage0_code":
        metrics.update(
            {
                "baseline_h39_valid_mae": float(payload["baseline_h39_valid_mae"]),
                "code_valid_mae": float(payload["code_valid_mae"]),
                "code_gain_vs_h39": float(payload["code_gain_vs_h39"]),
                "passed": bool(payload["passed"]),
                "decision": str(payload["decision"]),
            }
        )
    else:
        metrics.update({"all_exact": bool(payload["all_exact"])})
    (context.artifact_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[upstream] done stage={stage} arm={arm}", flush=True)
    return RunResult(metrics=metrics, status="completed", artifacts=["artifacts/metrics.json"])


__all__ = ["build_runner", "fingerprints", "run", "stages"]
