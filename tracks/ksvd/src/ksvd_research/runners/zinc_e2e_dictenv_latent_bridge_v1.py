"""Runner: E2E-DictEnv-Latent-Bridge-v1 (SEM108 + local task dictionary bridge).

One candidate only: the frozen Sem108 computation with one shared
low-dimensional task dictionary inserted at the local fusion output
``h[48] -> alpha[96] -> E[48]``, read unchanged by the unary pooling, the
static pair computation and the single graph readout.  Two new parameters only
(``D_L`` 48x96, ``V_L`` 96x48); no graph index, no message passing, no
write-back, no residual bypass.

``model.stage=screen`` runs the acceptance gates, the smoke, the formal seed-0
320-epoch CPU training, the frozen bridge probes and the analysis.
``model.stage=prepare`` is a no-op for this round (the candidate reuses the
required Sem108 / SDB / common-subspace objects; the old Joint709 prepare is
never invoked).

This round has **no** control arm: no dense/PCA bridge, no random dictionary
arm, no second seed.  The official ZINC **test** split is never instantiated;
a granted ``test_access`` is refused by this runner.
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
    zinc_e2e_dictenv_latent_bridge_v1 as stages,
)

EXPECTED_SPLIT_SIZES = {"train": 10_000, "val": 1_000, "test": 1_000}
DEFAULT_CONFIG = TRACK_ROOT / "configs/luyin16/zinc_e2e_dictenv_latent_bridge_v1.yaml"

ARTIFACTS = (
    "historical_references.json",
    "preflight.json",
    "preregistration_snapshot.json",
    "parameter_audit.json",
    "correctness.json",
    "smoke.json",
    "run_seed0.json",
    "curve_seed0.csv",
    "soup.json",
    "summary.json",
    "REPORT.md",
    "DECISION.md",
)


def build_runner():
    from ksvd_research.runner_api import Runner

    return Runner(
        name="zinc_e2e_dictenv_latent_bridge_v1",
        default_config=DEFAULT_CONFIG,
        default_study="zinc-context-gap",
        description=(
            "Latent-Bridge-v1: frozen Sem108 + a shared local task dictionary "
            "(h[48] -> alpha[96] -> E[48], tied unrolled ISTA, 9,216 new params), CPU, test blocked"
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
    metrics: dict[str, Any] = {
        "measure": "soup_valid_mae",
        "stage": stage,
        "seed": int(stages.SEED),
        "device": str(device),
        "bridge_dim": int(stages.lb.BRIDGE_DIM),
        "bridge_atoms": int(stages.lb.BRIDGE_ATOMS),
        "bridge_steps": int(stages.lb.BRIDGE_STEPS),
        "bridge_lambda1": float(stages.lb.BRIDGE_LAMBDA1),
        "bridge_lambda2": float(stages.lb.BRIDGE_LAMBDA2),
        "split_sizes": {"train": 10_000, "valid": 1_000, "test": None},
        "official_valid_loaded": True,
        "official_test_loaded": False,
    }
    summary_path = stages.RESULTS_DIR / "summary.json"
    if summary_path.exists():
        summary = stages._read_json(summary_path)
        bridge = summary.get("bridge", {})
        parameters = summary.get("parameters", {})
        code = bridge.get("code_density", {})
        metrics.update(
            {
                "verdict": str(summary.get("verdict", "INCOMPLETE")),
                "case": str(summary.get("case", "")),
                "epochs": int(summary.get("epochs", stages.TRAIN_EPOCHS)),
                "completed": True,
                "valid_mae": float(summary.get("M_S", float("nan"))),
                "valid_soup_mae": float(summary.get("M_S", float("nan"))),
                "valid_best_mae": float(summary.get("best_valid_mae", float("nan"))),
                "best_epoch": int(summary.get("best_epoch", 0)),
                "absolute_band": str(summary.get("performance_band", "unknown")),
                "sem108_background_soup_mae": float(
                    summary.get("sem108_background", {}).get("soup_valid_mae", float("nan"))
                ),
                "delta_vs_sem108_background": float(
                    summary.get("sem108_background", {}).get("delta_vs_background", float("nan"))
                ),
                "zero_bridge_code_delta_mae": float(bridge.get("zero_code_delta_mae", float("nan"))),
                "zero_bridge_code_delta_pred_rms": float(
                    bridge.get("zero_code_delta_pred_rms", float("nan"))
                ),
                "permutation_mean_delta_mae": float(
                    bridge.get("permutation_mean_delta_mae", float("nan"))
                ),
                "reset_to_init_delta_mae": float(
                    bridge.get("reset_to_init_delta_mae", float("nan"))
                ),
                "bridge_task_grad_D_L": float(bridge.get("task_grad_D_L", float("nan"))),
                "bridge_task_grad_V_L": float(bridge.get("task_grad_V_L", float("nan"))),
                "bridge_D_L_movement": float(bridge.get("D_L_movement_frobenius", float("nan"))),
                "bridge_V_L_movement": float(bridge.get("V_L_movement_frobenius", float("nan"))),
                "bridge_learned": bool(bridge.get("learned", False)),
                "bridge_load_bearing": bool(bridge.get("load_bearing", False)),
                "code_mean_nonzero": float(code.get("mean_nonzero", float("nan"))),
                "code_l0_p95": float(code.get("l0_p95", float("nan"))),
                "sem108_body_parameters": int(parameters.get("sem108_body", 0)),
                "bridge_parameters": int(parameters.get("bridge", 0)),
                "candidate_parameters": int(parameters.get("candidate", 0)),
                "wall_clock_s": float(summary.get("wall_clock_s", float("nan"))),
                "seconds_per_epoch": float(summary.get("seconds_per_epoch", float("nan"))),
                "correctness_all_passed": bool(
                    summary.get("gates", {}).get("correctness_all_passed", False)
                ),
                "smoke_passed": bool(summary.get("gates", {}).get("smoke_passed", False)),
            }
        )
    else:
        metrics["verdict"] = "PREPARE_OR_INCOMPLETE"
    return metrics


def run(config: Mapping[str, Any], context: RunContext) -> RunResult:
    if context.test_access == "granted":
        raise RuntimeError(
            "Latent-Bridge-v1 is a non-terminal screen; it must never run with test access granted"
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
        f"[latent-bridge] stage={stage} device={configured['device']} "
        f"epochs={configured['epochs']} threads={configured['threads']} "
        f"commit={stages._git_commit()}",
        flush=True,
    )
    if stage == "prepare":
        print("[latent-bridge] prepare is a no-op: Sem108 / SDB / common objects are reused", flush=True)
    if stage in ("screen", "all"):
        stages.run_stages("screen")
    artifacts = _copy_artifacts(context)
    metrics = _metrics(stage, configured["device"])
    (context.artifact_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    artifacts.append("artifacts/metrics.json")
    print(f"[latent-bridge] verdict={metrics['verdict']}", flush=True)
    return RunResult(metrics=metrics, status="completed", artifacts=artifacts)


__all__ = ["build_runner", "fingerprints", "run", "stages"]
