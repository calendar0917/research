"""Runner: ZINC CSSD g-component attribution v1 (frozen-consumer ell/s
component error attribution on the four frozen soups of
``zinc_cssd_consumer_replacement_v1``).

Two stages (CPU, seconds; no training, no GPU, no official valid/test):

* ``export`` — label-free: restore the four frozen soups, replay-check the
  promoted prediction npz, verify the strict reader identity
  ``h == ell_hat + s_hat`` on the real forward output, export the missing
  per-row dev component predictions, and fix the pre-registered label-free
  structure groupings (node terciles / coarse atom composition / coarse bond
  composition) with their validation cross-checks.
* ``analyze`` — the pre-registered labeled analysis: component synthesis
  (MAE/bias/cancellation per run, split, k-group), k=0 subgroup tables with
  DICT-RAW paired deltas, top-20 concentration shares, and canonical-SMILES
  group paired bootstrap CIs (2000 draws, seed 20261022).

dev (1999 rows = the historical development set) is exploratory localization,
never a confirmation.  See
``tracks/ksvd/notes/zinc_cssd_g_component_attribution_v1.md`` (frozen before
any labeled analysis).
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

from ksvd_research.runner_api import Runner, RunnerError  # noqa: E402

from tracks.ksvd.experiments.luyin16 import (  # noqa: E402
    zinc_cssd_g_component_attribution_v1 as stages,
)

EXPECTED_SPLIT_SIZES = {"train": 10_000, "val": 1_000, "test": 1_000}
DEFAULT_CONFIG = TRACK_ROOT / "configs/luyin16/zinc_cssd_g_component_attribution_v1.yaml"

STAGES = ("export", "analyze")


def build_runner() -> Runner:
    return Runner(
        name="zinc_cssd_g_component_attribution_v1",
        default_config=DEFAULT_CONFIG,
        default_study="zinc-context-gap",
        description=(
            "Frozen-consumer g-component attribution on the four frozen soups of "
            "zinc_cssd_consumer_replacement_v1: label-free dev component export "
            "with replay/identity checks, ell/s error synthesis per run/split/k-group, "
            "k=0 subgroup tables (node terciles / coarse atom / coarse bond composition) "
            "with DICT-RAW paired deltas and canonical-SMILES group bootstrap CIs, "
            "top-20 concentration; no training, no feature search, no official "
            "valid/test; dev is exploratory only"
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
            "zinc_cssd_g_component_attribution_v1 must never run with test "
            "access granted"
        )
    model_cfg = config.get("model", {})
    stage = str(model_cfg.get("stage", "export"))
    if stage not in STAGES:
        raise RunnerError(f"model.stage must be one of {STAGES}, got {stage!r}")
    runtime_cfg = config.get("runtime", {})
    device_name = str(runtime_cfg.get("device", "cpu"))

    if stage == "export":
        result = stages.stage_export(out_dir=stages.RESULTS_DIR, device_name=device_name)
        replay = result["replay_and_identity"]
        metrics = {
            "measure": "zinc_cssd_g_component_attribution_v1::export",
            "stage": stage,
            "purpose": "label-free frozen-soup dev component export + structure groupings",
            "official_valid_loaded": False,
            "official_test_loaded": False,
            "seconds": result["seconds"],
            "label_free": True,
            "atom_mapping_crosscheck_max_abs_diff": result["structure_groupings"][
                "atom_mapping_crosscheck"
            ]["max_abs_diff"],
            "max_replay_diff": max(
                v["dev_h_max_abs_diff"] for v in replay.values()
            ),
            "max_identity_gap": max(
                v["dev_identity_max_abs"] for v in replay.values()
            ),
        }
        metrics["all_checks_pass"] = bool(
            metrics["max_replay_diff"] <= stages.REPLAY_TOL
            and metrics["max_identity_gap"] <= stages.IDENTITY_TOL
            and result["structure_groupings"]["atom_mapping_crosscheck"]["passed"]
            and result["structure_groupings"]["canonical_smiles_groups_cross_fit_dev"] == 0
        )
        artifacts = ["artifacts/export.json"]
        for name in stages.RUN_NAMES:
            src = stages.RESULTS_DIR / f"dev_components_{name}.npz"
            (context.artifact_dir / src.name).write_bytes(src.read_bytes())
            artifacts.append(f"artifacts/{src.name}")
        src = stages.RESULTS_DIR / "structure_assignments.npz"
        (context.artifact_dir / src.name).write_bytes(src.read_bytes())
        artifacts.append("artifacts/structure_assignments.npz")
        _write_json(context.artifact_dir / "metrics.json", metrics)
        print(
            f"[cssd-g-component-attribution-v1] export done in {result['seconds']:.1f}s; "
            f"all_checks_pass={metrics['all_checks_pass']}",
            flush=True,
        )
        return RunResult(metrics=metrics, status="completed", artifacts=artifacts)

    # stage == "analyze"
    result = stages.analyze(out_dir=stages.RESULTS_DIR)
    dev = result["splits"]["dev"]["two_seed_mean"]
    boot = result["k0_bootstrap"]["overall"]
    conc_k0 = result["k0_concentration"]["k0"]["e_s"]["DICT"]["top_share"]
    addback_ok = all(
        all(
            abs(v) < 1e-9
            for k, v in per_run["addback"].items()
            if k.startswith("addback")
        )
        for split in result["k_groups"].values()
        for per_run in split["per_run"].values()
    )
    metrics = {
        "measure": "zinc_cssd_g_component_attribution_v1::analyze",
        "stage": stage,
        "purpose": "frozen-consumer ell/s component error attribution (exploratory, dev)",
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "dev_DICT_MAE_ell": dev["MAE_ell"]["DICT"],
        "dev_DICT_MAE_s": dev["MAE_s"]["DICT"],
        "dev_DICT_MAE_g": dev["MAE_g"]["DICT"],
        "dev_RAW_MAE_ell": dev["MAE_ell"]["RAW"],
        "dev_RAW_MAE_s": dev["MAE_s"]["RAW"],
        "dev_RAW_MAE_g": dev["MAE_g"]["RAW"],
        "dev_DICT_triangle_gap": dev["triangle_gap"]["DICT"],
        "dev_DICT_opposite_sign_fraction": dev["opposite_sign_fraction"]["DICT"],
        "k0_top20_share_of_abs_s_DICT": conc_k0,
        "bootstrap_delta_MAE_s_point": boot["e_s"]["point"],
        "bootstrap_delta_MAE_s_ci95": boot["e_s"]["ci95"],
        "bootstrap_delta_MAE_ell_point": boot["e_ell"]["point"],
        "bootstrap_delta_MAE_ell_ci95": boot["e_ell"]["ci95"],
    }
    metrics["all_checks_pass"] = bool(
        addback_ok
        and all(v < stages.IDENTITY_TOL for k, v in result["alignment"].items() if k.endswith("identity_gap"))
    )
    artifacts = []
    for name in (
        "component_summary.json", "subgroup_tables.csv", "concentration.csv",
        "source_manifest.json",
    ):
        src = stages.RESULTS_DIR / name
        (context.artifact_dir / name).write_text(
            src.read_text(encoding="utf-8"), encoding="utf-8"
        )
        artifacts.append(f"artifacts/{name}")
    _write_json(context.artifact_dir / "metrics.json", metrics)
    print(
        f"[cssd-g-component-attribution-v1] analyze done; "
        f"dev DICT MAE_s {metrics['dev_DICT_MAE_s']:.5f} "
        f"MAE_ell {metrics['dev_DICT_MAE_ell']:.5f} "
        f"MAE_g {metrics['dev_DICT_MAE_g']:.5f}; "
        f"all_checks_pass={metrics['all_checks_pass']}",
        flush=True,
    )
    return RunResult(metrics=metrics, status="completed", artifacts=artifacts)
