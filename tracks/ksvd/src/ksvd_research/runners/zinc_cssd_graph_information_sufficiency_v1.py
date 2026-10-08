"""Runner: ZINC CSSD graph information sufficiency v1.

With the frozen CSSD dictionary and the frozen DICT consumers of
``zinc_cssd_consumer_replacement_v1`` held fixed, does the graph-level
aggregation R_G (the reader input, verified 814D) discard task-relevant
information about the three-way binding between CSSD dictionary-atom supports
and the REAL radius-2 patch co-coverage of physical atoms?

``model.stage``:

* ``witness-export``  label-free W_real/W_shuf (32D each) + mechanism
                      diagnostics for all 10000 train rows; the frozen basis
                      operator math is used (never the zero-placeholder
                      ``DeployFull.code()``/``aux["coord"]``); CPU.
* ``restore-checks``  frozen DICT_s0/s1 restore, object hashes, prediction
                      replay vs the saved per-run npz, alpha operator
                      equivalence vs the deployed ``cssd_decode``, witness
                      batch/relabel invariances, label-isolation field check;
                      CPU.
* ``rg-export``       capture the true reader input R (814D) for fit+dev rows
                      of both frozen seeds via a reader forward-pre-hook, with
                      replay checks; CPU.
* ``heads``           the frozen 3-arm residual-head protocol (R-only /
                      R+W_shuffled / R+W_real), 5-fold canonical-SMILES-grouped
                      OOF on fit rows + exploratory dev scoring, group-paired
                      bootstrap, the frozen decision branches A/B/C; labels
                      are read ONLY here; CPU.
* ``report``          assemble the report manifest; CPU.

Non-terminal: official ZINC valid and test are never instantiated; a granted
``test_access`` is refused.  No backbone training, no architecture search.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

from ksvd_research.runtime.fingerprints import zinc_fingerprints
from ksvd_research.runtime.manifest import RunContext, RunResult
from ksvd_research.runtime.paths import REPO_ROOT, TRACK_ROOT, resolve_path

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ksvd_research.runner_api import Runner, RunnerError  # noqa: E402

from tracks.ksvd.experiments.luyin16 import (  # noqa: E402
    zinc_cssd_graph_information_sufficiency_v1 as stages,
)

EXPECTED_SPLIT_SIZES = {"train": 10_000, "val": 1_000, "test": 1_000}
DEFAULT_CONFIG = TRACK_ROOT / "configs/luyin16/zinc_cssd_graph_information_sufficiency_v1.yaml"

STAGES = ("witness-export", "restore-checks", "rg-export", "heads", "report")


def build_runner():
    return Runner(
        name="zinc_cssd_graph_information_sufficiency_v1",
        default_config=DEFAULT_CONFIG,
        default_study="zinc-context-gap",
        description=(
            "Frozen CSSD dictionary + frozen DICT consumers (s0/s1): does the "
            "graph-level aggregation R_G (reader input, 814D) discard "
            "task-relevant information about the binding between CSSD "
            "dictionary-atom supports and REAL patch co-coverage of physical "
            "atoms?  Label-free witness W_k = sum_x C(n[x,k],3) / (sum_x "
            "C(|C(x)|,3)+eps) from the deployed alpha operator (never "
            "aux['coord']) + env-cache incidence; SHAM = within-molecule "
            "support-row permutation (marginals preserved, binding destroyed); "
            "frozen 3-arm residual-head protocol (R-only 10,791 params vs "
            "R+W_shuffled / R+W_real 11,207 params), 5-fold canonical-SMILES "
            "grouped OOF on the 8001 fit rows + exploratory dev scoring, "
            "group-paired bootstrap (2000 draws, seed 20261021, shared picks), "
            "frozen decision branches A/B/C with thresholds fixed in the "
            "protocol BEFORE any labelled scoring; mechanism gate (real-vs-sham "
            "L1) required before scoring; dev = the 1999 historical development "
            "rows (never a new confirm); heads-OOF != whole-model OOF "
            "(disclosed); official valid+test blocked; 0 GPU, no backbone "
            "training, no architecture search"
        ),
        run_module=sys.modules[__name__],
    )


def fingerprints(config: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    data_root = resolve_path(config["data"]["root"])
    payload = zinc_fingerprints(data_root, expected_sizes=EXPECTED_SPLIT_SIZES)
    return payload, payload["split_fingerprint"]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def run(config: Mapping[str, Any], context: RunContext) -> RunResult:
    if context.test_access == "granted":
        raise RuntimeError(
            "zinc_cssd_graph_information_sufficiency_v1 is non-terminal; it must never run with test access granted"
        )
    model_cfg = config.get("model", {})
    stage = str(model_cfg.get("stage", "witness-export"))
    if stage not in STAGES:
        raise ValueError(f"model.stage must be one of {STAGES}, got {stage!r}")
    runtime = config.get("runtime", {})
    device_name = str(runtime.get("device", "cpu"))
    out_dir = stages.RESULTS_DIR
    print(f"[cssd-graph-info-sufficiency-v1] stage={stage} device={device_name}", flush=True)

    def _metrics(payload: Mapping[str, Any]) -> dict[str, Any]:
        base = {
            "measure": f"zinc_cssd_graph_information_sufficiency_v1::{stage}",
            "stage": stage,
            "official_valid_loaded": False,
            "official_test_loaded": False,
        }
        base.update({k: v for k, v in payload.items() if k in (
            "all_passed", "seconds", "label_free",
        )})
        if stage == "witness-export" and "mechanism_gate" in payload:
            base["mechanism_gate_passed"] = payload["mechanism_gate"]["passed"]
        if stage == "heads" and "decision" in payload:
            base["reading_branch"] = payload["decision"]["branch"]
            base["dG_R_two_seed_mean"] = payload["decision"]["dG_R_two_seed_mean"]
            base["dG_sham_two_seed_mean"] = payload["decision"]["dG_sham_two_seed_mean"]
        return base

    if stage == "witness-export":
        result = stages.stage_witness_export(out_dir=out_dir)
        _write_json(context.artifact_dir / "witness_export.json", result)
        metrics = _metrics(result)
        _write_json(context.artifact_dir / "metrics.json", metrics)
        return RunResult(metrics=metrics, status="completed", artifacts=["artifacts/witness_export.json"])

    if stage == "restore-checks":
        result = stages.stage_restore_checks(out_dir=out_dir)
        _write_json(context.artifact_dir / "restore_checks.json", result)
        metrics = _metrics(result)
        _write_json(context.artifact_dir / "metrics.json", metrics)
        return RunResult(metrics=metrics, status="completed", artifacts=["artifacts/restore_checks.json"])

    if stage == "rg-export":
        result = stages.stage_rg_export(out_dir=out_dir, device_name=device_name)
        _write_json(context.artifact_dir / "rg_export.json", result)
        metrics = _metrics(result)
        _write_json(context.artifact_dir / "metrics.json", metrics)
        return RunResult(metrics=metrics, status="completed", artifacts=["artifacts/rg_export.json"])

    if stage == "heads":
        result = stages.stage_heads(out_dir=out_dir)
        _write_json(context.artifact_dir / "heads_results.json", result)
        metrics = _metrics(result)
        _write_json(context.artifact_dir / "metrics.json", metrics)
        return RunResult(metrics=metrics, status="completed", artifacts=["artifacts/heads_results.json"])

    if stage == "report":
        result = stages.stage_report(out_dir=out_dir)
        _write_json(context.artifact_dir / "report_manifest.json", result)
        metrics = _metrics(result)
        _write_json(context.artifact_dir / "metrics.json", metrics)
        return RunResult(metrics=metrics, status="completed", artifacts=["artifacts/report_manifest.json"])

    raise RunnerError(f"unhandled stage {stage}")
