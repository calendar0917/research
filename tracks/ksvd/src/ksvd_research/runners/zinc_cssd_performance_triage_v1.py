"""Runner: ZINC CSSD performance triage v1 (read-only error attribution on
the saved predictions of ``zinc_cssd_consumer_replacement_v1``).

Single stage ``analyze`` (CPU, seconds, no torch, no GPU, no model loading, no
forward replays, no official valid/test).  Everything is computed from the
saved NumPy/JSON products of the promoted consumer-replacement round
(commit 32b8c96, four formal runs at revision a308e3490bfe):

* row alignment (gid/y vs targets at the saved fold positions),
* the stored identities (y = g + c; y_raw ≈ h + q_raw; h = ell_hat + s_hat),
* the frozen-Q object checks (one shared soup hash; bit-identical q_raw),
* the one-shot terminal-eval MAE reproduction at full precision,
* the e_g/e_Q/e_y error synthesis (component MAEs never summed into a
  budget; B_Q = net effect of the current Q error),
* the four k-group contributions with full-split-N normalisation and exact
  add-back checks,
* the dev top-20 by the two-seed mean DICT |e_y| with the located
  gid=3775/1424 localisation kept from the q-spotcheck note,
* the component cancellation gap (erratum support) and the two text errata
  of the previous round's REPORT (no historical number is modified).

Outputs: ``source_manifest.json``, ``error_summary.json``,
``group_contributions.csv``, ``top20.csv`` into the run artifact directory
(and the module's results directory for the round REPORT).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

from ksvd_research.runtime.fingerprints import zinc_fingerprints
from ksvd_research.runtime.manifest import RunContext, RunResult
from ksvd_research.runtime.paths import REPO_ROOT, TRACK_ROOT, resolve_path

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ksvd_research.runner_api import RunnerError  # noqa: E402

from tracks.ksvd.experiments.luyin16 import (  # noqa: E402
    zinc_cssd_performance_triage_v1 as stages,
)

EXPECTED_SPLIT_SIZES = {"train": 10_000, "val": 1_000, "test": 1_000}
DEFAULT_CONFIG = TRACK_ROOT / "configs/luyin16/zinc_cssd_performance_triage_v1.yaml"

STAGES = ("analyze",)


def build_runner():
    from ksvd_research.runner_api import Runner

    return Runner(
        name="zinc_cssd_performance_triage_v1",
        default_config=DEFAULT_CONFIG,
        default_study="zinc-context-gap",
        description=(
            "Read-only performance triage of the saved predictions of "
            "zinc_cssd_consumer_replacement_v1 (fit 8001 / dev 1999 rows): "
            "e_g/e_Q/e_y error synthesis, frozen-Q object checks, terminal-eval "
            "reproduction, four k-group contributions (full-split N, exact "
            "add-back), dev top-20 with the located long-cycle pair, the "
            "component cancellation gap and the previous round's two text "
            "errata; no training, no GPU, no model loading, official "
            "valid+test blocked"
        ),
        run_module=sys.modules[__name__],
    )


def fingerprints(config: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    data_root = resolve_path(config["data"]["root"])
    payload = zinc_fingerprints(data_root, expected_sizes=EXPECTED_SPLIT_SIZES)
    return payload, payload["split_fingerprint"]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def run(config: Mapping[str, Any], context: RunContext) -> RunResult:
    if context.test_access == "granted":
        raise RuntimeError(
            "zinc_cssd_performance_triage_v1 is a read-only analysis; it must "
            "never run with test access granted"
        )
    model_cfg = config.get("model", {})
    stage = str(model_cfg.get("stage", "analyze"))
    if stage not in STAGES:
        raise RunnerError(f"model.stage must be one of {STAGES}, got {stage!r}")

    result = stages.analyze(out_dir=stages.RESULTS_DIR)

    artifacts = []
    for name in ("source_manifest.json", "error_summary.json",
                 "group_contributions.csv", "top20.csv"):
        src = stages.RESULTS_DIR / name
        (context.artifact_dir / name).write_text(
            src.read_text(encoding="utf-8"), encoding="utf-8"
        )
        artifacts.append(f"artifacts/{name}")

    sm = result["seed_mean"]
    groups_dev = result["groups"]["splits"]["dev"]["seed_mean"]["DICT"]
    bq_total = sm["DICT/dev"]["B_Q"]
    tail_bq = groups_dev[3]["B_Q_group"]  # k<=-3
    km2_bq = groups_dev[2]["B_Q_group"]   # k=-2
    c_y_k0 = groups_dev[0]["C_y"]
    mae_y = sm["DICT/dev"]["MAE_y"]
    metrics = {
        "measure": "zinc_cssd_performance_triage_v1::analyze",
        "stage": stage,
        "purpose": "saved-prediction performance attribution and next-candidate design",
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "seconds": result["seconds"],
        "one_shot": True,
        "read_only": True,
        "dict_dev_mean_MAE_y": sm["DICT/dev"]["MAE_y"],
        "dict_dev_mean_MAE_g": sm["DICT/dev"]["MAE_g"],
        "dict_dev_mean_B_Q": sm["DICT/dev"]["B_Q"],
        "avg_delta_dev": sm["avg_delta/dev"],
        "k0_share_of_dict_dev_MAE_y": c_y_k0 / mae_y,
        "q_tail_share_of_dict_dev_B_Q": (tail_bq + km2_bq) / bq_total,
        "k_leq_minus3_share_of_dict_dev_B_Q": tail_bq / bq_total,
        "budget_priority": "g-generalization-readout",
        "purchased_candidate": "none",
    }
    checks_pass = all(
        [v for v in result["frozen_q"].values() if isinstance(v, bool)]
    ) and all(
        v["all_addback_checks_pass"] for v in result["addback_checks"].values()
    ) and all(
        v["y_raw_match_within_1e-8"] and v["g_raw_match_within_1e-8"]
        for v in result["terminal_eval_reproduction"].values()
    ) and all(
        c["gid_matches_targets_rows"] and c["y_matches_targets_rows"]
        for c in result["alignment"].values() if isinstance(c, dict)
    )
    metrics["all_checks_pass"] = bool(checks_pass)
    _write_json(context.artifact_dir / "metrics.json", metrics)
    print(
        f"[cssd-performance-triage-v1] analyze done in {result['seconds']:.1f}s; "
        f"all_checks_pass={metrics['all_checks_pass']}",
        flush=True,
    )
    return RunResult(metrics=metrics, status="completed", artifacts=artifacts)
