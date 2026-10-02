"""Runner: frozen-Full ZINC bottleneck audit (read-only conditional probe).

Protocol: ``tracks/ksvd/protocols/zinc-full-bottleneck-audit-v1.yaml``.
Stages (``model.stage``):

* ``replay``   — rebuild the frozen ``SCALE-FULL-seed0`` soup and replay
  raw/calibrated eval-mode train/valid MAE plus per-branch statistics;
* ``features`` — extract the frozen graph-level probe features to npz;
* ``probe``    — fit the ``[1,H2]`` baseline, the two frozen candidate blocks and
  the same-width controls with the reused MAE+L2 ADMM solver.

No backbone parameter is updated; the official ZINC test split is never
instantiated.  A granted ``test_access`` is refused.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Mapping

from ksvd_research.runtime.manifest import RunContext, RunResult
from ksvd_research.runtime.paths import REPO_ROOT, TRACK_ROOT

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tracks.ksvd.results.zinc_full_bottleneck_audit_v1 import (  # noqa: E402
    extract_probe_features,
    probe_phase_d,
)

DEFAULT_CONFIG = TRACK_ROOT / "configs/luyin16/zinc_full_bottleneck_audit_v1.yaml"
STAGES = ("replay", "features", "probe")
CHECKPOINT_SHA = "17f5fcc3cdf32462657006ec89ebf2ecc72ab6fbc954b926af248aca374574eb"


def build_runner():
    from ksvd_research.runner_api import Runner

    return Runner(
        name="zinc_full_bottleneck_audit_v1",
        default_config=DEFAULT_CONFIG,
        default_study="zinc-context-gap",
        description=(
            "Read-only frozen-Full ZINC bottleneck audit: soup replay + error budget + "
            "two conditional head probes on the frozen representation; test blocked"
        ),
        run_module=sys.modules[__name__],
    )


def fingerprints(config: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    payload = {
        "checkpoint_sha256": CHECKPOINT_SHA,
        "split_fingerprint": "58c69506df857bd8452481c6dcdd608ad230ba1a1aa90624850068fdd8faf28a",
        "official_test_loaded": False,
    }
    return payload, str(payload["split_fingerprint"])


def run(config: Mapping[str, Any], context: RunContext) -> RunResult:
    if context.test_access == "granted":
        raise RuntimeError("the bottleneck audit is non-terminal; test access must be blocked")
    import torch

    runtime = config.get("runtime", {})
    if str(runtime.get("device", "cpu")) != "cpu":
        raise RuntimeError("the frozen-Full bottleneck audit is CPU-only in this round")
    torch.set_num_threads(int(runtime.get("torch_threads", 8)))
    stage = str(config.get("model", {}).get("stage", "probe"))
    if stage not in STAGES:
        raise ValueError(f"model.stage must be one of {STAGES}, got {stage!r}")
    print(f"[full-audit] stage={stage}", flush=True)

    if stage == "replay":
        from tracks.ksvd.results.zinc_full_bottleneck_audit_v1 import audit_full

        audit_full.main()
        metrics = {"measure": "replay", "official_test_loaded": False}
    elif stage == "features":
        extract_probe_features.main()
        metrics = {"measure": "probe_features", "official_test_loaded": False}
    else:
        if not probe_phase_d.FEAT.exists():
            extract_probe_features.main()
        probe_phase_d.main()
        payload = json.loads((probe_phase_d.OUT / "phase_d_probe.json").read_text(encoding="utf-8"))
        base = float(payload["baseline"]["valid_mae"])
        metrics = {
            "measure": "conditional_valid_mae",
            "baseline_h2_valid_mae": base,
            "p1_valid_mae": float(payload["P1_prefix_interface"]["valid_mae"]),
            "p1_gain": float(payload["P1_prefix_interface"]["gain_vs_baseline"]),
            "p2_valid_mae": float(payload["P2_node_joint"]["valid_mae"]),
            "p2_gain": float(payload["P2_node_joint"]["gain_vs_baseline"]),
            "control_max_gain": max(
                float(payload["C1_control_h2proj_892"]["gain_vs_baseline"]),
                float(payload["C2_control_h2proj_924"]["gain_vs_baseline"]),
            ),
            "gate_gain_min": float(payload["gate"]["gain_min"]),
            "official_test_loaded": False,
        }
    (context.artifact_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[full-audit] done stage={stage}", flush=True)
    return RunResult(metrics=metrics, status="completed", artifacts=["artifacts/metrics.json"])


__all__ = ["build_runner", "fingerprints", "run"]