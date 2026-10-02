"""Runner: nonlinear head probe on the frozen Full ZINC soup.

Protocol: ``tracks/ksvd/protocols/zinc-full-nonlinear-probe-v1.yaml``.
Three frozen configs (A=[R], B=[R,z], C=[R,q(R)]) x two fixed seeds, plus a
parent tail recheck and a node-weight member check.  No backbone update; the
official ZINC test split is never instantiated.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Mapping

from ksvd_research.runtime.manifest import RunContext, RunResult
from ksvd_research.runtime.paths import REPO_ROOT, TRACK_ROOT

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tracks.ksvd.results.zinc_full_nonlinear_probe_v1 import nonlinear_probe  # noqa: E402

DEFAULT_CONFIG = TRACK_ROOT / "configs/luyin16/zinc_full_nonlinear_probe_v1.yaml"
CHECKPOINT_SHA = "17f5fcc3cdf32462657006ec89ebf2ecc72ab6fbc954b926af248aca374574eb"


def build_runner():
    from ksvd_research.runner_api import Runner

    return Runner(
        name="zinc_full_nonlinear_probe_v1",
        default_config=DEFAULT_CONFIG,
        default_study="zinc-context-gap",
        description=(
            "Frozen-Full ZINC nonlinear head probe (A=[R], B=[R,z], C=[R,q(R)]; "
            "2 seeds) + parent tail recheck; no backbone training; test blocked"
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
        raise RuntimeError("the nonlinear probe is non-terminal; test access must be blocked")
    import torch

    runtime = config.get("runtime", {})
    if str(runtime.get("device", "cpu")) != "cpu":
        raise RuntimeError("the nonlinear probe is CPU-only in this round")
    torch.set_num_threads(int(runtime.get("torch_threads", 8)))

    nonlinear_probe.main()
    payload = json.loads((nonlinear_probe.OUT / "nonlinear_probe.json").read_text(encoding="utf-8"))
    agg = payload["two_seed_aggregates"]
    metrics = {
        "measure": "nonlinear_head_valid_mae",
        "parent_cal_valid_mae": payload["constants"]["parent_cal_valid_mae"],
        "A_mean_gain": agg["A"]["mean_gain_vs_parent"],
        "B_mean_gain": agg["B"]["mean_gain_vs_parent"],
        "C_mean_gain": agg["C"]["mean_gain_vs_parent"],
        "B_minus_A_mean_gain": agg["B_minus_A"]["mean_gain"],
        "B_minus_C_mean_gain": agg["B_minus_C"]["mean_gain"],
        "gate_gain": payload["constants"]["gate_gain"],
        "frozen_epochs": payload["frozen"]["frozen_epochs"],
        "seconds_per_epoch": payload["frozen"]["speed_test_seconds_per_epoch"],
        "wall_clock_seconds": payload["wall_clock_seconds"],
        "official_test_loaded": False,
    }
    (context.artifact_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("[nonlinear-probe] done", flush=True)
    return RunResult(metrics=metrics, status="completed", artifacts=["artifacts/metrics.json"])


__all__ = ["build_runner", "fingerprints", "run"]