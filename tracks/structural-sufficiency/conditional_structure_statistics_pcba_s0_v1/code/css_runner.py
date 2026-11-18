"""Runner ``css-pcba-s0-v1``: conditional structure statistics on PCBA, seed 0.

One ``research run`` = one stage:

* ``prepare``   - CPU only: freeze every identity (data / norm / split / cache /
                  order plan / canonical initial state / parameter counts / run
                  matrix / reuse manifest); writes ``analysis/*.json``.
* ``preflight`` - the measurement gates of ``css_preflight``; writes
                  ``analysis/preflight.json`` and ``analysis/cost_projection.json``.
* ``train``     - one formal arm trajectory (``--set arm=E|L|S``); 40 epochs,
                  every epoch fully evaluated on Train / Validation / Test.
* ``analyze``   - the pre-registered analysis, including the S purchase rule.

Only engineering knobs are overridable (stage, arm, device, resume, epochs for
smoke runs).  Every scientific constant lives in the code and is hashed into
``analysis/frozen_config.json``.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import css_common as C

DEFAULT_CONFIG = C.CONFIGS / "config.yaml"


def build_runner():
    from ksvd_research.runner_api import Runner

    return Runner(
        name=C.ROUND_ID,
        default_config=DEFAULT_CONFIG,
        default_study=C.STUDY_ID,
        description=(
            "Conditional structure statistics (EARLY_TASK E / LATE_TASK L / EARLY_SHARED S): "
            "one from-scratch subject over frozen S1/S2 structural occurrences, official "
            "ogbg-molpcba, seed 0, 40 epochs of the inherited recipe, full three-split "
            "evaluation every epoch (DEVELOPMENT_TEST)"
        ),
        run_module=sys.modules[__name__],
    )


# ---------------------------------------------------------------------------
# fingerprints (small files only: no 4.5 GB cache load at plan time)
# ---------------------------------------------------------------------------


def fingerprints(config: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    pilot = C.PILOT
    spec_path = pilot / "analysis" / "pcba_spec.json"
    split_path = pilot / "analysis" / "pcba_split.json"
    labels_path = pilot / "analysis" / "pcba_labels_trainvalid.npy"
    manifest_path = pilot / "analysis" / "cache_manifest.json"
    cpc_meta = C.CPC_ROUND / "runs" / "cpc_cache" / "meta.json"
    payload: dict[str, Any] = {
        "round_id": C.ROUND_ID,
        "dataset": C.DATASET,
        "feature_spec_hash": C.FEATURE_SPEC_HASH,
        "norm_fingerprint": C.NORM_FINGERPRINT,
        "split_partition_sha256": C.SPLIT_PARTITION_SHA256,
        "cache_sha256": C.CACHE_SHA256,
        "label_artifact_sha256": C.LABEL_ARTIFACT_SHA256,
        "files": {
            "spec": [C.track_rel(spec_path), C.sha256_file(spec_path) if spec_path.is_file() else "missing"],
            "split": [C.track_rel(split_path), C.sha256_file(split_path) if split_path.is_file() else "missing"],
            "labels": [C.track_rel(labels_path), C.sha256_file(labels_path) if labels_path.is_file() else "missing"],
            "cache_manifest": [
                C.track_rel(manifest_path),
                C.sha256_file(manifest_path) if manifest_path.is_file() else "missing",
            ],
            "cpc_meta": [C.track_rel(cpc_meta), C.sha256_file(cpc_meta) if cpc_meta.is_file() else "missing"],
            "official_graph_label": [
                C.track_rel(C.OGB_GRAPH_LABEL),
                C.sha256_file(C.OGB_GRAPH_LABEL) if C.OGB_GRAPH_LABEL.is_file() else "missing",
            ],
        },
        "split_sizes": {"train": C.N_TRAIN, "valid": C.N_VALID, "test": C.N_TEST},
        "plan": dict(C.PLAN),
        "param_counts": dict(C.EXPECTED_PARAMETERS),
        "task_chunk": int(config.get("task_chunk", C.TASK_CHUNK_DEFAULT)),
        "dropout_generator_seed": int(C.DROPOUT_GENERATOR_SEED),
        "canonical_init_seed": int(C.CANONICAL_INIT_SEED),
        "test_policy": "always",
        "test_role": C.TEST_ROLE,
    }
    return payload, C.SPLIT_PARTITION_SHA256


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------


def run(config: Mapping[str, Any], context) -> Any:
    from ksvd_research.runtime.manifest import RunResult

    if context.test_access != "granted":
        raise RuntimeError(
            "this round evaluates the official Test split every epoch as DEVELOPMENT_TEST; "
            "the control plane blocked test access"
        )
    stage = str(config.get("stage", "prepare"))
    ledger = C.Ledger(Path(context.artifact_dir) / "ledger.jsonl")
    t0 = time.perf_counter()
    if stage == "prepare":
        metrics = _stage_prepare(config, ledger)
    elif stage == "preflight":
        metrics = _stage_preflight(config, ledger)
    elif stage == "train":
        metrics = _stage_train(config, ledger)
    elif stage == "analyze":
        metrics = _stage_analyze(config, ledger)
    else:
        raise ValueError(f"unknown stage {stage!r}")
    metrics["wall_seconds"] = float(time.perf_counter() - t0)
    _write_resource_ledger(ledger, config, metrics)
    return RunResult(status="completed", metrics=metrics, artifacts=[])


def _stage_prepare(config: Mapping[str, Any], ledger) -> dict[str, Any]:
    import torch

    import css_data as D
    import css_model as M

    with C.stage_timer(ledger, "prepare", device="cpu", purpose="freeze"):
        frozen = {
            "schema": "css-pcba-s0-v1-frozen-config-v1",
            "round_id": C.ROUND_ID,
            "protocol_id": C.PROTOCOL_ID,
            "study_id": C.STUDY_ID,
            "seed": int(C.SEED),
            "canonical_init_seed": int(C.CANONICAL_INIT_SEED),
            "dropout_generator_seed": int(C.DROPOUT_GENERATOR_SEED),
            "task_chunk_default": int(C.TASK_CHUNK_DEFAULT),
            "recipe": dict(C.RECIPE),
            "lr_schedule": C.lr_audit(),
            "plan": dict(C.PLAN),
            "expected_parameters": dict(C.EXPECTED_PARAMETERS),
            "parameter_breakdown": dict(C.PARAMETER_BREAKDOWN),
            "arms": {k: dict(v) for k, v in C.ARMS.items()},
            "identities": {
                "feature_spec_hash": C.FEATURE_SPEC_HASH,
                "norm_fingerprint": C.NORM_FINGERPRINT,
                "split_partition_sha256": C.SPLIT_PARTITION_SHA256,
                "cache_sha256": C.CACHE_SHA256,
                "label_artifact_sha256": C.LABEL_ARTIFACT_SHA256,
                "official_graph_label_sha256": C.sha256_file(C.OGB_GRAPH_LABEL),
                "cpc_cache_meta_sha256": C.sha256_file(
                    C.CPC_ROUND / "runs" / "cpc_cache" / "meta.json"
                ),
            },
            "s_gate": dict(C.S_GATE),
            "historical_anchors": dict(C.HISTORICAL),
            "budget": dict(C.BUDGET),
            "test_role": C.TEST_ROLE,
            "math": {
                "front_end": "atom 51->256->192; bond 589->320->192 (direction invariant); "
                "S1 576->192->96; S2 961->256->96; key 96->16 tanh; value 96->32",
                "query": "q=tanh(q0 + [qt_t] + Wc c_G), Wc bias-free, q0=qt=Wc=0 at init",
                "gate": "w(s) = 1 + 0.5 tanh(s), s = dot(k, q)/sqrt(16); no softmax, no learnable scale",
                "E": "z_f,v,t = sum_p w(s_p,t) v_f,p / sqrt(max(n_f(v),1))",
                "L": "z_f,v,t = w(dot(mean_p k_f,p, q_G,t)/sqrt(16)) sum_p v_f,p / sqrt(max(n_f(v),1))",
                "root": "j = Linear_static([a_v, sum_e b_e, scalar_v]) + Linear_stat([z1,z2]); "
                "h = LayerNorm(SiLU(j)); dropout 0.10 on h during training only",
                "readout": "p_G,t = [sum h/sqrt(n), mean h, mean h^2, c_G, log1p n]; "
                "r = MLP(235->192->96); logit = wout[t].r + bout[t]",
            },
            "created_utc": C.utcnow(),
            "git_commit": C.git_commit(),
        }
        C.atomic_write_json(C.ANALYSIS / "frozen_config.json", frozen)
        C.atomic_write_json(
            C.ANALYSIS / "run_matrix.json",
            {
                "round_id": C.ROUND_ID,
                "arms": [
                    {
                        "arm": arm,
                        "name": C.ARMS[arm]["name"],
                        "conditioning": C.ARMS[arm]["conditioning"],
                        "task_query": bool(C.ARMS[arm]["task_query"]),
                        "assigned_pool": C.ARMS[arm]["assigned_pool"],
                        "parameters": int(C.EXPECTED_PARAMETERS[arm]),
                        "status": "primary" if arm in ("E", "L") else "conditional_on_s_gate",
                    }
                    for arm in C.ARM_ORDER
                ],
                "seed": int(C.SEED),
                "epochs": int(C.TOTAL_EPOCHS),
                "batch_size": int(C.BATCH_SIZE),
                "eval_policy": "full Train+Valid+Test after every epoch; DEVELOPMENT_TEST",
            },
        )
        dev = D.load_task(with_cpc=False)
        init_audit = M.audit_initial_states(dev.task.norm)
        state = M.canonical_init_state()
        torch.save(
            {
                "schema": C.SCHEMA_CANONICAL_INIT,
                "round_id": C.ROUND_ID,
                "arm_reference": "E (S omits qt; L is identical to E)",
                "canonical_init_seed": int(C.CANONICAL_INIT_SEED),
                "state_dict": state,
                "shared_fingerprint": init_audit["canonical_shared_state_fingerprint"],
                "created_utc": C.utcnow(),
            },
            C.ANALYSIS / "canonical_init_E.pt",
        )
        C.atomic_write_json(
            C.ANALYSIS / "canonical_init.json",
            {
                "schema": C.SCHEMA_CANONICAL_INIT,
                "file": C.track_rel(C.ANALYSIS / "canonical_init_E.pt"),
                "file_sha256": C.sha256_file(C.ANALYSIS / "canonical_init_E.pt"),
                "shared_fingerprint": init_audit["canonical_shared_state_fingerprint"],
                "canonical_init_seed": int(C.CANONICAL_INIT_SEED),
                "parameter_audit": init_audit,
            },
        )
        C.atomic_write_json(
            C.ANALYSIS / "identities.json",
            {
                "round_id": C.ROUND_ID,
                "feature_spec_hash": C.FEATURE_SPEC_HASH,
                "norm_fingerprint": C.NORM_FINGERPRINT,
                "split_partition_sha256": C.SPLIT_PARTITION_SHA256,
                "cache_sha256": C.CACHE_SHA256,
                "label_artifact_sha256": C.LABEL_ARTIFACT_SHA256,
                "official_graph_label_sha256": C.sha256_file(C.OGB_GRAPH_LABEL),
                "plan_global_sha256": C.PLAN["global_plan_sha256"],
                "plan_consumed40_prefix_sha256": C.PLAN["consumed40_prefix_sha256"],
                "cpc_cache": {
                    "path": C.track_rel(C.CPC_ROUND / "runs" / "cpc_cache"),
                    "meta_sha256": C.sha256_file(C.CPC_ROUND / "runs" / "cpc_cache" / "meta.json"),
                    "n_paths": int(C.read_json(C.CPC_ROUND / "runs" / "cpc_cache" / "meta.json")["n_paths"]),
                },
                "label_view_audit": dev.label_view_audit(),
            },
        )
        C.atomic_write_json(C.ANALYSIS / "reuse_manifest.json", _reuse_manifest(dev))
    return {
        "stage": "prepare",
        "parameters": dict(C.EXPECTED_PARAMETERS),
        "init_audit_passed": bool(init_audit["passed"]),
        "identities": {
            "feature_spec_hash": C.FEATURE_SPEC_HASH,
            "norm_fingerprint": C.NORM_FINGERPRINT,
            "split_partition_sha256": C.SPLIT_PARTITION_SHA256,
            "plan_global_sha256": C.PLAN["global_plan_sha256"],
        },
    }


def _reuse_manifest(dev) -> dict[str, Any]:
    return {
        "schema": "css-pcba-s0-v1-reuse-manifest-v1",
        "round_id": C.ROUND_ID,
        "reused": [
            {
                "asset": "lip_v1 PCBA feature cache (atom 51 / bond 13 / scalar 9 / global 42)",
                "source": C.track_rel(C.PILOT / "cache" / "pcba_v1"),
                "verification": "inherited lr_data.load_task identity gates: feature spec hash, "
                "norm fingerprint, split partition hash, label artifact hash, cache file "
                "count/bytes/manifest sha; no rebuild",
                "identity": {"feature_spec_hash": C.FEATURE_SPEC_HASH, "cache_sha256": C.CACHE_SHA256},
            },
            {
                "asset": "official ogbg-molpcba Train-only normalisation",
                "source": "pcba_scale_pilot_v1 analysis (fit_split=train_only, 350,343 rows)",
                "verification": "norm_fingerprint equality (no re-fit)",
                "identity": {"norm_fingerprint": C.NORM_FINGERPRINT},
            },
            {
                "asset": "official split (350,343 / 43,793 / 43,793)",
                "source": C.track_rel(C.PILOT / "analysis" / "pcba_split.json"),
                "verification": "partition sha256 + inherited overlap/size gates",
                "identity": {"split_partition_sha256": C.SPLIT_PARTITION_SHA256},
            },
            {
                "asset": "frozen 40-epoch seed-0 order plan",
                "source": "sg_data.make_plan40 (regenerated, not copied)",
                "verification": "global plan hash + consumed-40 prefix hash equality",
                "identity": dict(C.PLAN),
            },
            {
                "asset": "rooted two-bond path (S2) structure cache incl. middle atom and both "
                "physical bond ids",
                "source": C.track_rel(C.CPC_ROUND / "runs" / "cpc_cache"),
                "verification": "cache meta schema/rows/paths + per-batch structural audit "
                "(counts identity, bond endpoints, closure vs frozen path_closure); no rebuild",
                "identity": {"meta_sha256": C.sha256_file(C.CPC_ROUND / "runs" / "cpc_cache" / "meta.json")},
            },
            {
                "asset": "S1 directed occurrence index",
                "source": "derived from the frozen bond_index/bond_batch at collate time",
                "verification": "S1 == 2 x physical bonds and endpoint consistency on every audited batch",
                "identity": {"derived": True},
            },
            {
                "asset": "PinnedStager host-to-device transfer",
                "source": "lcrc_pcba_pipeline_perf_v1/code/pf_pipeline.py",
                "verification": "already gated in that round; unchanged tensors/dtypes",
            },
            {
                "asset": "recipe + per-step LR schedule",
                "source": "sg_common.SCRATCH_RECIPE / scheduled_lr(phase=3)",
                "verification": "imported, not re-derived; audit printed in frozen_config.json",
            },
            {
                "asset": "official Test targets",
                "source": C.track_rel(C.OGB_GRAPH_LABEL),
                "verification": "raw Train/Valid blocks reproduce the frozen artefact bit-for-bit; "
                "Test block read once into a separate development view",
                "identity": {"sha256": C.sha256_file(C.OGB_GRAPH_LABEL)},
            },
        ],
        "not_reused_and_not_rebuilt": [
            {
                "asset": "historical LCRC backbones / W2 / W3 / Dynamic / CPC / RC checkpoints",
                "reason": "this round's subject is from scratch; no weights are loaded, and no "
                "historical training is repeated. Their published Validation/Test numbers are "
                "cited as performance anchors only.",
            },
            {
                "asset": "missing historical per-epoch Train/Test curves",
                "reason": "reported as missing; not re-trained to fill old curves",
            },
        ],
        "label_view": dev.label_view_audit(),
        "created_utc": C.utcnow(),
    }


def _stage_preflight(config: Mapping[str, Any], ledger) -> dict[str, Any]:
    import css_preflight as PF

    device = str(config.get("device", "cuda:0"))
    gates = config.get("gates")
    with C.stage_timer(ledger, "preflight", device=device, purpose="gates"):
        payload = PF.run_preflight(
            device_name=device,
            gates=None if not gates else (gates if isinstance(gates, (list, tuple)) else [gates]),
        )
    summary = {
        "stage": "preflight",
        "passed": payload["passed"],
        "failing_gates": payload["failing_gates"],
        "gates_run": payload["gates_run"],
        "elapsed_seconds": payload["elapsed_seconds"],
        "preflight_gpu_seconds": payload["preflight_gpu_seconds"],
    }
    if "cost_projection" in payload:
        summary["cost_projection"] = {
            k: payload["cost_projection"][k]
            for k in (
                "projected_arm_hours",
                "projected_arm_hours_with_20pct_margin",
                "projected_two_arm_gpu_hours",
                "within_gpu_soft_budget",
                "within_gpu_hard_budget",
            )
        }
    return summary


def _stage_train(config: Mapping[str, Any], ledger) -> dict[str, Any]:
    import torch

    import css_data as D
    import css_train as T

    arm = str(config.get("arm", "E"))
    if arm not in C.ARMS:
        raise ValueError(f"arm must be one of {C.ARM_ORDER}")
    if arm == "S" and not (C.ANALYSIS / "s_gate.json").is_file():
        raise RuntimeError("S is gated: analysis/s_gate.json with purchased=true is required")
    if arm == "S":
        gate = C.read_json(C.ANALYSIS / "s_gate.json")
        if not bool(gate.get("purchased")):
            raise RuntimeError("S purchase rule said NOT_PURCHASED; refusing to train S")
    device_name = str(config.get("device", "cuda:0"))
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    dev = D.load_task(with_cpc=True)
    out_dir = C.RUNS / str(config.get("out_subdir") or C.ARMS[arm]["name"])
    with C.stage_timer(ledger, f"train_{arm}", device=str(device), purpose="formal"):
        summary = T.train(
            dev,
            arm=arm,
            out_dir=out_dir,
            device=device,
            epochs=int(config.get("epochs", C.TOTAL_EPOCHS)),
            task_chunk=int(config.get("task_chunk", C.TASK_CHUNK_DEFAULT)),
            resume=bool(config.get("resume", True)),
            log_every=int(config.get("log_every", 200)),
            prefetch_depth=int(config.get("prefetch_depth", T.PREFETCH_DEPTH)),
            steps_per_epoch=config.get("steps_per_epoch"),
            eval_limit=config.get("eval_limit"),
        )
    _update_registry(arm, summary, run_id=str(getattr(ledger, "run_id", "")) or "")
    return {
        "stage": "train",
        "arm": arm,
        "epochs_run": summary["epochs_run"],
        "best_epoch": summary["best_epoch"],
        "best_valid_ap": summary["best_valid_ap"],
        "parameters": summary["parameters"],
        "wall_seconds": summary["wall_seconds"],
        "device": summary["device"],
        "outputs": C.track_rel(out_dir),
    }


def _stage_analyze(config: Mapping[str, Any], ledger) -> dict[str, Any]:
    import css_analyze as A

    device = str(config.get("device", "cpu"))
    with C.stage_timer(ledger, "analyze", device="cpu", purpose="analysis"):
        payload = A.run_analysis(device=device, mechanism_rows=int(config.get("mechanism_rows", C.MECHANISM_VALID_ROWS)))
    gate = payload["s_gate"]
    C.atomic_write_json(C.ANALYSIS / "s_gate.json", gate)
    _write_resource_ledger(ledger, config, {})
    return {
        "stage": "analyze",
        "arms": [row["arm"] for row in payload["main_table"]],
        "d40": gate["d40_valid_ap_E_minus_L"],
        "s_purchased": gate["purchased"],
        "s_label": gate["label"],
        "best_valid_ap": {row["arm"]: row["best_valid_ap"] for row in payload["main_table"]},
        "epoch40_test_ap": {row["arm"]: row["epoch40_state"]["test_ap"] for row in payload["main_table"]},
    }


def _update_registry(arm: str, summary: Mapping[str, Any], *, run_id: str) -> None:
    path = C.ANALYSIS / "run_registry.json"
    registry = C.read_json(path) if path.is_file() else {"round_id": C.ROUND_ID, "runs": []}
    registry["runs"] = [
        r
        for r in registry["runs"]
        if not (r.get("arm") == arm and r.get("seed") == summary.get("seed"))
    ]
    registry["runs"].append(
        {
            "arm": arm,
            "arm_name": C.ARMS[arm]["name"],
            "seed": int(summary["seed"]),
            "epochs_run": int(summary["epochs_run"]),
            "best_epoch": int(summary["best_epoch"]),
            "best_valid_ap": float(summary["best_valid_ap"]),
            "parameters": int(summary["parameters"]),
            "wall_seconds": float(summary["wall_seconds"]),
            "gpu_hours": float(summary["wall_seconds"]) / 3600.0,
            "device": summary["device"],
            "host": summary.get("host", {}).get("hostname"),
            "gpu_names": summary.get("host", {}).get("gpu_names"),
            "git_commit": summary.get("git_commit"),
            "run_dir": C.track_rel(C.RUNS / C.ARMS[arm]["name"]),
            "control_plane_run_id": run_id,
            "registered_utc": C.utcnow(),
        }
    )
    registry["total_gpu_hours"] = float(sum(r["gpu_hours"] for r in registry["runs"]))
    C.atomic_write_json(path, registry)


def _write_resource_ledger(ledger, config: Mapping[str, Any], metrics: Mapping[str, Any]) -> None:
    summary = ledger.summary()
    registry_path = C.ANALYSIS / "run_registry.json"
    registry = C.read_json(registry_path) if registry_path.is_file() else {"runs": []}
    training_gpu_hours = float(sum(r.get("gpu_hours", 0.0) for r in registry.get("runs", [])))
    payload = {
        "schema": "css-pcba-s0-v1-resource-ledger-v1",
        "round_id": C.ROUND_ID,
        "stage": str(config.get("stage", "")),
        "stage_entries": summary["entries"],
        "stage_gpu_seconds": summary["gpu_seconds_total"],
        "stage_gpu_hours": summary["gpu_hours_total"],
        "training_registry_gpu_hours": training_gpu_hours,
        "total_gpu_hours_estimate": float(summary["gpu_hours_total"] + training_gpu_hours),
        "budget": dict(C.BUDGET),
        "updated_utc": C.utcnow(),
    }
    previous = C.ANALYSIS / "resource_ledger.json"
    if previous.is_file():
        old = C.read_json(previous)
        merged = old.get("stage_entries", []) + payload["stage_entries"]
        payload["stage_entries"] = merged
        payload["stage_gpu_hours"] = float(
            sum(float(e.get("gpu_seconds", 0.0) or 0.0) for e in merged) / 3600.0
        )
        payload["total_gpu_hours_estimate"] = float(payload["stage_gpu_hours"] + training_gpu_hours)
    C.atomic_write_json(previous, payload)


__all__ = ["DEFAULT_CONFIG", "build_runner", "fingerprints", "run"]
