"""Runner: E2E-DictEnv-Scale-v1 (unified Small/Full task-dictionary scaling).

One candidate only: the Sem108 local object with the task path widened by a
single ``ScaleSpec`` multiplier.  ``m = 1`` reproduces the frozen latent-bridge
model bit-for-bit; ``m = 3`` (Full, 408,651 parameters) widens fusion, the
shared task dictionary, the static-relation path and the graph reader while
every fixed module (structural dictionary, bindings, slot encoders, global /
topology auxiliaries, C6 mask) stays identical.  No message passing, no
attention, no node/edge hidden state, no write-back.

``model.stage=screen`` runs the acceptance gates, the CPU timing gate, the
3-epoch smoke, the formal Full seed-0 320-epoch CPU training (only when the
measured budget fits 4 hours), the frozen probes and the analysis.
``model.stage=prepare`` is a no-op (the round reuses the frozen Sem108 /
common-subspace objects and their audits).

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
    zinc_e2e_dictenv_scale_v1 as stages,
)

EXPECTED_SPLIT_SIZES = {"train": 10_000, "val": 1_000, "test": 1_000}
DEFAULT_CONFIG = TRACK_ROOT / "configs/luyin16/zinc_e2e_dictenv_scale_v1.yaml"

ARTIFACTS = (
    "historical_references.json",
    "preflight.json",
    "preregistration_snapshot.json",
    "parameter_audit.json",
    "correctness.json",
    "timing.json",
    "smoke.json",
    "run_seed0.json",
    "train_skipped.json",
    "curve_seed0.csv",
    "soup.json",
    "summary.json",
    "REPORT.md",
    "DECISION.md",
)


def build_runner():
    from ksvd_research.runner_api import Runner

    return Runner(
        name="zinc_e2e_dictenv_scale_v1",
        default_config=DEFAULT_CONFIG,
        default_study="zinc-context-gap",
        description=(
            "Scale-v1: unified Small/Full task-dictionary family (m=1 106,925; m=3 408,651), "
            "Full seed-0 CPU screen gated by real forward/backward timing, test blocked"
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
    return copied


def _metrics(stage: str, device: str) -> dict[str, Any]:
    stages_module = stages
    metrics: dict[str, Any] = {
        "measure": "soup_valid_mae",
        "stage": stage,
        "seed": int(stages_module.SEED),
        "device": str(device),
        "scale_multiplier": int(stages_module.sc.FULL.m),
        "task_dim": int(stages_module.sc.FULL.d),
        "task_atoms": int(stages_module.sc.FULL.k),
        "pair_dim": int(stages_module.sc.FULL.p),
        "bridge_steps": int(stages_module.lb.BRIDGE_STEPS),
        "bridge_lambda1": float(stages_module.lb.BRIDGE_LAMBDA1),
        "bridge_lambda2": float(stages_module.lb.BRIDGE_LAMBDA2),
        "split_sizes": {"train": 10_000, "valid": 1_000, "test": None},
        "official_valid_loaded": True,
        "official_test_loaded": False,
    }
    summary_path = stages_module.RESULTS_DIR / "summary.json"
    if summary_path.exists():
        summary = stages_module._read_json(summary_path)
        parameters = summary.get("parameters", {})
        timing = summary.get("timing", {})
        smoke = summary.get("smoke", {})
        bridge = summary.get("bridge", {})
        code = bridge.get("code_density", {})
        metrics.update(
            {
                "verdict": str(summary.get("verdict", "INCOMPLETE")),
                "case": str(summary.get("case", "")),
                "status": str(summary.get("status", "completed")),
                "completed": bool("M_S" in summary),
                "epochs": int(summary.get("epochs", stages_module.TRAIN_EPOCHS)),
                "small_parameters": int(parameters.get("small", 0)),
                "full_parameters": int(parameters.get("full", 0)),
                "full_task_dictionary_parameters": int(parameters.get("full_task_dictionary", 0)),
                "full_body_parameters": int(parameters.get("full_body", 0)),
                "timing_authorized": bool(
                    summary.get("gates", {}).get("timing_authorized", False)
                ),
                "correctness_all_passed": bool(
                    summary.get("gates", {}).get("correctness_all_passed", False)
                ),
                "smoke_passed": bool(summary.get("gates", {}).get("smoke_passed", False)),
                "predicted_total_with_margin_s": float(
                    timing.get("predicted_total_seconds_with_margin", float("nan"))
                ),
                "smoke_seconds_per_epoch": float(smoke.get("seconds_per_epoch", float("nan"))),
                "smoke_final_train_mae": float(smoke.get("final_train_mae", float("nan"))),
            }
        )
        if "M_S" in summary:
            metrics.update(
                {
                    "valid_mae": float(summary.get("M_S", float("nan"))),
                    "valid_soup_mae": float(summary.get("M_S", float("nan"))),
                    "valid_best_mae": float(summary.get("best_valid_mae", float("nan"))),
                    "best_epoch": int(summary.get("best_epoch", 0)),
                    "absolute_band": str(summary.get("performance_band", "unknown")),
                    "soup_train_mae": float(summary.get("soup_train_mae", float("nan"))),
                    "train_valid_gap": float(summary.get("train_valid_gap", float("nan"))),
                    "delta_vs_latent_bridge_background": float(
                        summary.get("backgrounds", {}).get("delta_vs_latent_bridge", float("nan"))
                    ),
                    "delta_vs_sem108_background": float(
                        summary.get("backgrounds", {}).get("delta_vs_sem108", float("nan"))
                    ),
                    "zero_code_delta_mae": float(bridge.get("zero_code_delta_mae", float("nan"))),
                    "permutation_mean_delta_mae": float(
                        bridge.get("permutation_mean_delta_mae", float("nan"))
                    ),
                    "reset_to_init_delta_mae": float(
                        bridge.get("reset_to_init_delta_mae", float("nan"))
                    ),
                    "bridge_task_grad_D_L": float(bridge.get("task_grad_D_L", float("nan"))),
                    "bridge_task_grad_V_L": float(bridge.get("task_grad_V_L", float("nan"))),
                    "code_mean_nonzero": float(code.get("mean_nonzero", float("nan"))),
                    "wall_clock_s": float(summary.get("wall_clock_s", float("nan"))),
                    "seconds_per_epoch": float(summary.get("seconds_per_epoch", float("nan"))),
                }
            )
    else:
        metrics["verdict"] = "PREPARE_OR_INCOMPLETE"
    return metrics


def run(config: Mapping[str, Any], context: RunContext) -> RunResult:
    if context.test_access == "granted":
        raise RuntimeError(
            "Scale-v1 is a non-terminal screen; it must never run with test access granted"
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
        f"[scale] stage={stage} device={configured['device']} epochs={configured['epochs']} "
        f"threads={configured['threads']} commit={stages._git_commit()}",
        flush=True,
    )
    if stage == "prepare":
        print("[scale] prepare is a no-op: Sem108 / SDB / common objects and audits are reused", flush=True)
    if stage in ("screen", "all"):
        stages.run_stages("chain")
    artifacts = _copy_artifacts(context)
    metrics = _metrics(stage, configured["device"])
    (context.artifact_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    artifacts.append("artifacts/metrics.json")
    print(f"[scale] verdict={metrics['verdict']}", flush=True)
    return RunResult(metrics=metrics, status="completed", artifacts=artifacts)


__all__ = ["build_runner", "fingerprints", "run", "stages"]
