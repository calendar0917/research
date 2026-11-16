"""Runner ``lcrc-readout-context-pilot-small-v1``: prepare / smoke / dataset_worker / aggregate.

One run = one stage.  The two parallel GPU workers are

    --set stage=dataset_worker --set dataset=ogbg-molesol   (pool res-gpu0)
    --set stage=dataset_worker --set dataset=ogbg-mollipo   (pool res-gpu1)

and each executes the frozen per-dataset sequence of the round brief:

    1 verify the four frozen caches and audit both datasets
    2 preflight checks (cr_checks, all 25 gates) -> analysis/checks.json
    3 arm A (C1 readout, unchanged) : 360 epochs, full Train/Val/Test every epoch
    4 arm B (additive: roots + active globals)
    5 arm C (B + size / ring context interaction)
    6 per-arm trajectory.json, checkpoints, prediction matrices, ledgers

Every step persists its own JSON/npz and a signature marker so an interrupted
worker resumes instead of restarting, and so the aggregate stage never needs the
GPU host again.

Only engineering knobs are overridable (stage, dataset, resume, epochs for the
smoke stage).  The scientific constants live in the code modules and are hashed
into ``analysis/protocol_freeze.json``.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import cr_common as C
import cr_data as D
import cr_models as M
import cr_train as T
from cr_common import Ledger, stage_timer

DEFAULT_CONFIG = C.ROUND_DIR / "configs" / f"{C.ROUND_ID}.yaml"
SMOKE_EPOCHS = 3


def build_runner():
    from ksvd_research.runner_api import Runner

    return Runner(
        name=C.ROUND_ID,
        default_config=DEFAULT_CONFIG,
        default_study=C.STUDY_ID,
        description=(
            "LCRC explicit context readout feasibility pilot on official ogbg-molesol / "
            "ogbg-mollipo: budget-matched additive readouts A/B/C on one frozen local trunk "
            "(test_policy=always, seed 0)"
        ),
        run_module=sys.modules[__name__],
    )


# ---------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------


def fingerprints(config: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    """Dataset fingerprint from the committed, label-free caches."""
    payload: dict[str, Any] = {"round_id": C.ROUND_ID}
    for dataset in C.DATASETS:
        cache = C.SQL_DATA / D.cache_filename(dataset)
        raw = C.SQL_DATA / D.lipo_raw_filename(dataset)
        split_payload: dict[str, Any] = {}
        for split in C.SPLITS:
            mesh = np.load(cache, allow_pickle=False)
            try:
                idx = np.asarray(mesh[C.SPLIT_FILE[split]], dtype=np.int64)
            finally:
                mesh.close()
            split_payload[split] = {
                "n": int(idx.size),
                "sha256": C.sha256_array(idx),
            }
        payload[dataset] = {
            "graph_cache_sha256": C.sha256_file(cache) if cache.is_file() else "missing",
            "lipo_raw_sha256": C.sha256_file(raw) if raw.is_file() else "missing",
            "splits": split_payload,
        }
    split_fingerprint = C.sha256_json(
        {d: payload[d]["splits"] for d in C.DATASETS if isinstance(payload.get(d), dict)}
    )
    return payload, split_fingerprint


# ---------------------------------------------------------------------------
# io
# ---------------------------------------------------------------------------


def _write(path: Path, payload: Any) -> None:
    C.atomic_write_json(path, payload)


def _read(path: Path) -> Any:
    return C.read_json(path)


def run(config: Mapping[str, Any], context) -> Any:
    from ksvd_research.runtime.manifest import RunResult

    if context.test_access != "granted":
        raise RuntimeError(
            "this round requires test_policy=always; the control plane blocked official test access"
        )
    stage = str(config.get("stage", "dataset_worker"))
    ledger_dir = Path(context.artifact_dir)
    ledger = Ledger(ledger_dir / "ledger.jsonl")
    t0 = time.perf_counter()
    if stage == "prepare":
        metrics = _stage_prepare(config, ledger)
    elif stage == "smoke":
        metrics = _stage_smoke(config, ledger)
    elif stage == "dataset_worker":
        metrics = _stage_dataset_worker(config, ledger, ledger_dir)
    elif stage == "aggregate":
        import cr_analysis

        metrics = cr_analysis.aggregate(datasets=_datasets(config), ledger=ledger)
    else:
        raise ValueError(f"unknown stage {stage!r}")
    metrics = {**metrics, "stage": stage, "wall_seconds": time.perf_counter() - t0}
    _write(ledger_dir / f"{stage}_summary.json", {**metrics, "ledger": ledger.summary()})
    return RunResult(
        metrics=metrics,
        status="completed",
        artifacts=[f"artifacts/{stage}_summary.json", "artifacts/ledger.jsonl"],
    )


def _datasets(config: Mapping[str, Any]) -> list[str]:
    raw = config.get("dataset", "all")
    if isinstance(raw, (list, tuple)):
        names = [str(x) for x in raw]
    else:
        names = [str(raw)]
    if "all" in names:
        return list(C.DATASETS)
    bad = [n for n in names if n not in C.DATASETS]
    if bad:
        raise ValueError(f"unknown dataset(s) {bad}; allowed {C.DATASETS} or 'all'")
    return names


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------


def _budget_record(bundle: D.Bundle) -> dict[str, Any]:
    specs = M.arm_specs(bundle)
    out: dict[str, Any] = {}
    for arm, spec in specs.items():
        out[arm] = {
            "readout_form": {
                "A": "f(h_v) summed over roots + rho([g, mean h, mean h^2, log1p n])  [unchanged C1]",
                "B": "b_global + sum_v c0(h_v) + sum_j u_j(z_j) over active globals",
                "C": "B + a_size(z_size) * sum_v c_size(h_v) + a_ring(z_ring) * sum_v c_ring(h_v)",
            }[arm],
            "hidden": spec.hidden,
            "n_active_global": spec.n_active,
            "nominal_readout_params": spec.readout_params_nominal,
            "budget": spec.budget,
        }
    budget = int(M.READOUT_BUDGET)
    spread = [rec["nominal_readout_params"] for rec in out.values()]
    return {
        "readout_budget_A": budget,
        "assignment": out,
        "nominal_param_spread": int(max(spread) - min(spread)),
        "match_rule": (
            "A is the frozen C1 readout (its parameter count is the budget; no C1 change is allowed). "
            "H for B and C is chosen so the nominal readout count is within one hidden-width step "
            "(+/-48 / +/-66 params) of A; exact equality is impossible for integer H."
        ),
    }


def _stage_prepare(config: Mapping[str, Any], ledger: Ledger) -> dict[str, Any]:
    C.set_global_seeds(C.TRAIN_SEED)
    metrics: dict[str, Any] = {"round_id": C.ROUND_ID, "datasets": {}}
    audits: dict[str, Any] = {}
    budgets: dict[str, Any] = {}
    for dataset in C.DATASETS:
        with stage_timer(ledger, "prepare.verify_frozen", dataset=dataset):
            verified = D.verify_frozen_files(dataset)
        with stage_timer(ledger, "prepare.build_bundle", dataset=dataset):
            bundle = D.build_bundle(dataset)
        with stage_timer(ledger, "prepare.audit", dataset=dataset):
            audit = D.audit_dataset(dataset)
        budget = _budget_record(bundle)
        audits[dataset] = audit
        budgets[dataset] = budget
        metrics["datasets"][dataset] = {
            "frozen_files": verified,
            "split_sizes": bundle.split_sizes,
            "n_active_global": int(bundle.active_global["n_active"]),
            "n_inactive_global": len(bundle.active_global["inactive_indices"]),
            "active_global_names": list(bundle.active_global["active_names"]),
            "inactive_global_names": list(bundle.active_global["inactive_names"]),
            "context_fields": list(C.CONTEXT_FIELDS),
            "target_scaler": {k: float(bundle.scaler[k]) for k in ("mean", "std")},
            "feature_spec_hash": bundle.extra["spec_hash"],
            "norm_fingerprint": bundle.extra["norm_fingerprint"],
            "build_seconds": bundle.build_seconds,
            "readout_budget": budget,
        }

    canonical_hashes = {}
    for dataset in C.DATASETS:
        _b = D.build_bundle(dataset)
        canonical_hashes[dataset] = M.state_hashes(M.canonical_c1_state(_b.spec, _b.norm))
    freeze = {
        "round_id": C.ROUND_ID,
        "protocol_id": C.PROTOCOL_ID,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "design_basis_commit": "9b3dba098915b0cdd800767f61b523681ee9e25d",
        "round_commit": C.git_commit(),
        "round_branch": C.git_branch(),
        "seed": C.TRAIN_SEED,
        "seed_semantics": "single training seed; no repeat, no multi-seed claim, no best-of-N",
        "arms": list(C.ARMS),
        "context_fields": list(C.CONTEXT_FIELDS),
        "context_quantiles": list(C.CONTEXT_QUANTILES),
        "active_span_eps": C.ACTIVE_SPAN_EPS,
        "recipe": T.recipe_record(),
        "budget": budgets,
        "canonical_init": canonical_hashes,
        "frozen_caches": C.EXPECTED_CACHE_SHA256,
        "reused_readonly": {
            "sq-pilot-small-v1": sorted(
                str(p.relative_to(C.REPO_ROOT)) for p in (C.SQL_CODE / "sq_common.py", C.SQL_CODE / "sq_data.py")
            ),
            "sq_runner": str((C.SQL_CODE / "sq_runner.py").relative_to(C.REPO_ROOT)),
            "sq_results_method_G": str((C.SQL_RESULTS).relative_to(C.REPO_ROOT)),
            "lcrc_cross_task_preparation_v1": str(C.PREP_CODE.relative_to(C.REPO_ROOT)),
            "lcrc_lipo_model_dissection_v1": str(C.DISSECT_CODE.relative_to(C.REPO_ROOT)),
        },
        "code_sha256": T.code_hashes(),
        "checks_path": str((C.ANALYSIS_DIR / "checks.json").relative_to(C.REPO_ROOT)),
        "analysis_scales": {
            "trace_tolerance": "1e-5 + 1e-6 * max(1, |x|) in raw target units",
            "bootstrap_replicates": 2000,
            "bootstrap_seed": C.BOOTSTRAP_SEED,
            "retention_tolerance_ratio": 1.05,
            "context_gain_threshold_relative": 0.02,
            "interaction_resolution_std": 0.01,
            "root_specificity_threshold": 0.05,
        },
        "test_policy": "always",
        "environment": C.environment_snapshot(),
    }
    freeze["freeze_hash"] = C.sha256_json({k: v for k, v in freeze.items() if k != "freeze_hash"})
    C.ensure_dir(C.ANALYSIS_DIR)
    _write(C.ANALYSIS_DIR / "protocol_freeze.json", freeze)
    _write(C.ANALYSIS_DIR / "dataset_audit.json", audits)
    metrics["freeze_hash"] = freeze["freeze_hash"]
    metrics["protocol_freeze"] = str((C.ANALYSIS_DIR / "protocol_freeze.json").relative_to(C.REPO_ROOT))
    metrics["all_audits_pass"] = bool(all(a["pass"] for a in audits.values()))
    if not metrics["all_audits_pass"]:
        raise RuntimeError("dataset audit failed; refusing to continue")
    return metrics


# ---------------------------------------------------------------------------
# smoke
# ---------------------------------------------------------------------------


def _stage_smoke(config: Mapping[str, Any], ledger: Ledger) -> dict[str, Any]:
    """Three epochs per arm on molesol: proves the loop, never a reported number."""
    import cr_checks

    epochs = int(config.get("smoke_epochs", SMOKE_EPOCHS))
    dataset = str(config.get("dataset", C.DATASETS[0]))
    if dataset == "all":
        dataset = C.DATASETS[0]
    out: dict[str, Any] = {"dataset": dataset, "epochs": epochs, "arms": {}}
    with stage_timer(ledger, "smoke.checks", dataset=dataset):
        out["checks"] = {
            "all_pass": cr_checks.check_single_variable_functions()["pass"],
        }
    bundle = D.build_bundle(dataset)
    canonical = M.canonical_c1_state(bundle.spec, bundle.norm)
    specs = M.arm_specs(bundle)
    for arm in C.ARMS:
        with stage_timer(ledger, f"smoke.train.{arm}", dataset=dataset):
            result = T.run_trajectory(
                bundle,
                arm,
                arm_spec=specs[arm],
                canonical_state=canonical,
                out_dir=T.smoke_dir(dataset, arm),
                device=_device(config),
                epochs=epochs,
                resume=bool(config.get("resume", True)),
                ledger=ledger,
                purpose="smoke",
            )
        out["arms"][arm] = {
            "selected_epoch": result["selected"]["epoch"],
            "selected_valid_rmse_raw": result["selected"]["valid_rmse_raw"],
            "selected_test_rmse_raw": result["selected"]["test_rmse_raw"],
            "n_parameters": result["n_parameters"],
            "total_updates": result["total_updates"],
            "wall_seconds": result["train_seconds"],
            "note": "smoke only; never reported as a scientific result",
        }
    out["all_arms_completed"] = all(a["total_updates"] > 0 for a in out["arms"].values())
    return out


def _device(config: Mapping[str, Any]) -> str:
    import torch

    requested = str(config.get("device", "cuda:0"))
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return requested


# ---------------------------------------------------------------------------
# dataset worker
# ---------------------------------------------------------------------------


def _stage_dataset_worker(config: Mapping[str, Any], ledger: Ledger, artifact_dir: Path) -> dict[str, Any]:
    import cr_checks

    names = _datasets(config)
    if len(names) != 1:
        raise RuntimeError("the dataset_worker stage takes exactly one dataset")
    dataset = names[0]
    device = _device(config)
    metrics: dict[str, Any] = {"dataset": dataset, "device": device, "arms": {}}

    with stage_timer(ledger, "worker.verify_frozen", dataset=dataset):
        metrics["frozen_files"] = D.verify_frozen_files(dataset)
    with stage_timer(ledger, "worker.checks", dataset=dataset):
        checks = cr_checks.run_all(datasets=[dataset], tmp_dir=artifact_dir / "_checkpoints")
        _write(C.ANALYSIS_DIR / f"checks_{dataset}.json", checks)
        merged = C.ANALYSIS_DIR / "checks.json"
        if merged.is_file():
            # keep the union of both workers so the aggregate stage sees every gate
            previous = _read(merged)
            seen = {(c["check"], str(c.get("dataset"))) for c in previous.get("checks", [])}
            for check in checks["checks"]:
                if (check["check"], str(check.get("dataset"))) not in seen:
                    previous.setdefault("checks", []).append(check)
            previous["n_checks"] = len(previous["checks"])
            failed = [
                c["check"] + ":" + str(c.get("dataset")) for c in previous["checks"] if not c.get("pass")
            ]
            previous["failed"] = failed
            previous["all_pass"] = bool(not failed)
            previous["datasets"] = sorted({str(c.get("dataset")) for c in previous["checks"] if c.get("dataset")})
            _write(merged, previous)
        else:
            _write(merged, checks)
    metrics["checks_all_pass"] = bool(checks["all_pass"])
    if not checks["all_pass"]:
        raise RuntimeError(f"{dataset}: preflight checks failed: {checks['failed']}")

    with stage_timer(ledger, "worker.build_bundle", dataset=dataset):
        bundle = D.build_bundle(dataset)
    canonical = M.canonical_c1_state(bundle.spec, bundle.norm)
    specs = M.arm_specs(bundle)
    _write(
        C.RESULTS_DIR / dataset / "budget.json",
        {
            "dataset": dataset,
            "readout_budget_A": int(M.READOUT_BUDGET),
            "arms": {
                arm: {
                    "hidden": specs[arm].hidden,
                    "nominal_readout_params": specs[arm].readout_params_nominal,
                    "budget": specs[arm].budget,
                }
                for arm in C.ARMS
            },
            "n_active_global": int(bundle.active_global["n_active"]),
            "active_global_names": list(bundle.active_global["active_names"]),
            "inactive_global_names": list(bundle.active_global["inactive_names"]),
            "target_scaler": {k: float(bundle.scaler[k]) for k in ("mean", "std")},
            "split_sizes": bundle.split_sizes,
            "feature_spec_hash": bundle.extra["spec_hash"],
            "norm_fingerprint": bundle.extra["norm_fingerprint"],
        },
    )
    for arm in C.ARMS:
        with stage_timer(ledger, f"worker.train.{arm}", dataset=dataset, arm=arm):
            result = T.run_trajectory(
                bundle,
                arm,
                arm_spec=specs[arm],
                canonical_state=canonical,
                out_dir=T.trajectory_dir(dataset, arm),
                device=device,
                epochs=None,
                resume=bool(config.get("resume", True)),
                ledger=ledger,
                purpose="formal",
            )
        metrics["arms"][arm] = {
            "selected_epoch": result["selected"]["epoch"],
            "selected_train_rmse_raw": result["selected"]["train_rmse_raw"],
            "selected_valid_rmse_raw": result["selected"]["valid_rmse_raw"],
            "selected_test_rmse_raw": result["selected"]["test_rmse_raw"],
            "selected_valid_mae_raw": result["selected"]["valid_mae_raw"],
            "selected_test_mae_raw": result["selected"]["test_mae_raw"],
            "final_valid_rmse_raw": result["final"]["valid_rmse_raw"],
            "final_test_rmse_raw": result["final"]["test_rmse_raw"],
            "mean_last20_valid_rmse_raw": result["last20"]["mean_last20_valid_rmse_raw"],
            "mean_last20_test_rmse_raw": result["last20"]["mean_last20_test_rmse_raw"],
            "n_parameters": result["n_parameters"],
            "readout_parameters": result["readout_parameters"],
            "hidden": result.get("hidden"),
            "total_updates": result["total_updates"],
            "train_seconds": result["train_seconds"],
            "eval_seconds_total": result["eval_seconds_total"],
            "test_eval_count": result["test_eval_count"],
            "test_policy": result["test_policy"],
            "resumed_skip": bool(result.get("resumed_skip", False)),
        }
    metrics["test_policy"] = "always"
    metrics["readout_budget"] = {arm: specs[arm].readout_params_nominal for arm in C.ARMS}
    metrics["n_active_global"] = int(bundle.active_global["n_active"])
    return metrics
