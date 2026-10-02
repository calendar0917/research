"""Runner: zinc_full_reader_wd_pilot_v1 (paired reader-weight-decay warm pilot).

Protocol: ``tracks/ksvd/protocols/zinc-full-reader-wd-pilot-v1.yaml``.
Stages: ``timing`` (one control epoch -> freeze H) and ``pair`` (control +
candidate at the frozen H).  No backbone architecture change, no test access.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Mapping

from ksvd_research.runtime.manifest import RunContext, RunResult
from ksvd_research.runtime.paths import REPO_ROOT, TRACK_ROOT

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tracks.ksvd.results.zinc_full_reader_wd_pilot_v1 import wd_pilot  # noqa: E402

DEFAULT_CONFIG = TRACK_ROOT / "configs/luyin16/zinc_full_reader_wd_pilot_v1.yaml"
STAGES = ("timing", "pair")
CHECKPOINT_SHA = "17f5fcc3cdf32462657006ec89ebf2ecc72ab6fbc954b926af248aca374574eb"


def build_runner():
    from ksvd_research.runner_api import Runner

    return Runner(
        name="zinc_full_reader_wd_pilot_v1",
        default_config=DEFAULT_CONFIG,
        default_study="zinc-context-gap",
        description=(
            "Paired warm-start pilot: control vs 10x reader weight decay on the frozen "
            "Full soup (Adam coupled L2, last-5 soup); no test access"
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
        raise RuntimeError("reader-wd pilot is non-terminal; test access must be blocked")
    import torch

    runtime = config.get("runtime", {})
    if str(runtime.get("device", "cpu")) != "cpu":
        raise RuntimeError("this pilot is CPU-only")
    threads = int(runtime.get("torch_threads", 8))
    torch.set_num_threads(threads)
    model = config.get("model", {})
    stage = str(model.get("stage", "timing"))
    if stage not in STAGES:
        raise ValueError(f"model.stage must be one of {STAGES}")

    wd_pilot.OUT.mkdir(parents=True, exist_ok=True)
    if stage == "timing":
        payload = wd_pilot.timing_run(torch.device("cpu"), threads)
        metrics = {"measure": "timing", **payload}
    else:
        h = int(model["h"])
        mult = float(model.get("reader_wd_mult", 10))
        results = wd_pilot.pair_run(torch.device("cpu"), h, mult)
        ev = wd_pilot.evaluate(results)
        wd_pilot.write_valid_csv(results)
        four = wd_pilot.four_forwards()
        summary = {
            "protocol_version": "zinc_full_reader_wd_pilot_v1",
            "stage": "pair", "h": h, "reader_wd_mult": mult,
            "control": {k: v for k, v in results["control"].items() if "valid_predictions" not in k},
            "candidate": {k: v for k, v in results["candidate"].items() if "valid_predictions" not in k},
            "evaluation": ev,
            "four_forwards": four,
            "checkpoint_sha256": CHECKPOINT_SHA,
            "official_test_loaded": False,
        }
        (wd_pilot.OUT / "summary.json").write_text(
            json.dumps(summary, indent=2, default=float), encoding="utf-8")
        (wd_pilot.OUT / "four_forwards.json").write_text(
            json.dumps(four, indent=2, default=float), encoding="utf-8")
        metrics = {"measure": "pair", "h": h,
                   "control_cal_valid_mae": results["control"]["calibrated_valid_mae"],
                   "candidate_cal_valid_mae": results["candidate"]["calibrated_valid_mae"],
                   **ev["gate"]}
    (context.artifact_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, default=float), encoding="utf-8")
    print(json.dumps(metrics, indent=2, default=float), flush=True)
    return RunResult(metrics=metrics, status="completed", artifacts=["artifacts/metrics.json"])


__all__ = ["build_runner", "fingerprints", "run"]