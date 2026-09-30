"""Runner: E2E-DictEnv-RoleCorr-v1 (role<->attribute correspondence object).

Implements
``tracks/ksvd/notes/e2e_dictenv_rolecorr_v1_preregistration.md``: the frozen
``CSSD-Sem108 + C6`` pipeline with the dictionary coordinate replaced by a
shared sparse dictionary over the 536-D role x attribute correspondence object
``C`` (``D_S16 + D_C16 + common1`` = 33 coordinates), screened against the
frozen pure-topology coordinate (``SDB K32/s8 + common1``).

Two invocations are defined by the pre-registration:

* ``model.stage=primary`` — cache / scaler / audit / dictionaries / correctness
  / smoke / train (arms ``TOPO`` and ``CORR``, seed 0) / interventions /
  analysis;
* ``model.stage=controls`` — only authorised after the 2 % screening gate has
  fired; trains the frozen controls ``CORR-SHUF`` and ``CORR-PCA`` and writes
  the final verdict.

CPU-first (the repository execution regime).  The official ZINC **valid** split
is evaluation only and the official **test** split is never instantiated; a
granted ``test_access`` is refused.
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
    zinc_e2e_dictenv_rolecorr_v1 as stages,
)

EXPECTED_SPLIT_SIZES = {"train": 10_000, "val": 1_000, "test": 1_000}
DEFAULT_CONFIG = TRACK_ROOT / "configs/luyin16/zinc_e2e_dictenv_rolecorr_v1.yaml"

ARTIFACTS = (
    "audit.json",
    "cache_meta_train.json",
    "cache_meta_valid.json",
    "correctness.json",
    "smoke.json",
    "standardizers.json",
    "dictionaries.json",
    "run_TOPO.json",
    "run_CORR.json",
    "curve_TOPO.csv",
    "curve_CORR.csv",
    "soup_TOPO.json",
    "soup_CORR.json",
    "interventions.json",
    "controls_objects.json",
    "control_interventions.json",
    "run_CORR-SHUF.json",
    "run_CORR-PCA.json",
    "summary.json",
    "REPORT.md",
    "DECISION.md",
)


def build_runner():
    from ksvd_research.runner_api import Runner

    return Runner(
        name="zinc_e2e_dictenv_rolecorr_v1",
        default_config=DEFAULT_CONFIG,
        default_study="zinc-context-gap",
        description=(
            "RoleCorr-v1: frozen-dictionary Sem108+C6 with a shared sparse "
            "dictionary over the 536-D role x attribute correspondence object; "
            "seed-0 screen with frozen shuffle/PCA controls, CPU"
        ),
        run_module=sys.modules[__name__],
    )


def fingerprints(config: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    data_root = resolve_path(config["data"]["root"])
    payload = zinc_fingerprints(data_root, expected_sizes=EXPECTED_SPLIT_SIZES)
    return payload, payload["split_fingerprint"]


def _config_epochs(config: Mapping[str, Any]) -> int:
    model = config.get("model", {})
    return int(model.get("epochs", stages.TRAIN_EPOCHS))


def _config_stage(config: Mapping[str, Any]) -> str:
    model = config.get("model", {})
    stage = str(model.get("stage", "primary"))
    if stage not in ("primary", "controls"):
        raise ValueError(f"model.stage must be 'primary' or 'controls', got {stage!r}")
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


def _metrics(summary: Mapping[str, Any], stage: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "measure": "soup_valid_mae",
        "stage": stage,
        "seed": int(stages.SEED),
        "device": "cpu",
        "epochs": int(stages.TRAIN_EPOCHS),
        "split_sizes": {"train": 10_000, "valid": 1_000, "test": None},
        "official_valid_loaded": True,
        "official_test_loaded": False,
        "verdict": str(summary["verdict"]),
        "relative_improvement": float(summary["relative_improvement"]),
        "screen_gate": float(summary["screen_gate"]),
        "screen_gate_fired": bool(summary["screen_gate_fired"]),
        "correctness_all_passed": bool(summary["correctness_all_passed"]),
        "audit_passed": bool(summary["audit_passed"]),
    }
    if "M_A_soup" in summary:
        metrics.update(
            {
                "valid_mae": float(summary["M_B_soup"]),
                "valid_soup_mae": float(summary["M_B_soup"]),
                "valid_best_mae": float(summary["M_B_best"]),
                "M_A_soup": float(summary["M_A_soup"]),
                "M_B_soup": float(summary["M_B_soup"]),
                "M_A_best": float(summary["M_A_best"]),
                "M_B_best": float(summary["M_B_best"]),
            }
        )
    inter = summary.get("interventions") or {}
    for key in ("M_A0", "M_C0", "M_S0", "G_A0", "G_C0", "G_S0", "G_Cshuf"):
        if key in inter:
            metrics[key] = float(inter[key])
    if "controls" in summary:
        for key in ("M_C", "M_D", "G_C_vs_B", "G_PCA_vs_B"):
            metrics[key] = float(summary["controls"][key])
        metrics["answers"] = dict(summary["answers"])
    return metrics


def run(config: Mapping[str, Any], context: RunContext) -> RunResult:
    if context.test_access == "granted":
        raise RuntimeError(
            "RoleCorr-v1 is a non-terminal screen; it must never run with test "
            "access granted"
        )
    stage = _config_stage(config)
    configure = stages.configure(
        epochs=_config_epochs(config),
        threads=int(config.get("runtime", {}).get("torch_threads", stages.THREADS)),
    )
    print(
        f"[rolecorr] stage={stage} epochs={configure['epochs']} "
        f"threads={configure['threads']} commit={stages._git_commit()}",
        flush=True,
    )
    if stage == "primary":
        stages.run_primary(list(stages.PRIMARY_STAGES))
    else:
        stages.run_controls(list(stages.CONTROL_STAGES))

    summary = stages._read_json(stages.RESULTS_DIR / "summary.json")
    artifacts = _copy_artifacts(context)
    metrics = _metrics(summary, stage)
    (context.artifact_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    artifacts.append("artifacts/metrics.json")
    print(
        f"[rolecorr] verdict={metrics['verdict']} "
        f"M_A={metrics.get('M_A_soup')} M_B={metrics.get('M_B_soup')} "
        f"rel={metrics['relative_improvement']:+.4%}",
        flush=True,
    )
    return RunResult(metrics=metrics, status="completed", artifacts=artifacts)


__all__ = ["build_runner", "fingerprints", "run", "stages"]
