"""Runner ``sq-lipo-pair-rule-v1``: prepare / worker_c1 / worker_rules / aggregate.

One run = one stage.  The remote execution regime is

    prepare       GPU1 pool, one process: identity, frozen matrices, CPU
                  preflight, MLP GPU smoke
    worker_c1     GPU0 pool: C1 only
    worker_rules  GPU1 pool: MLP on cuda:0, then BASE / NULL / REAL on CPU
    aggregate     local CPU after ``rr pull``

Everything a later stage needs is written into the stage result directory that
``rr run --result`` declares, so an interrupted worker resumes from its own
status/signature instead of recomputing a finished arm.
"""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import sq_pair_common as C
import sq_pair_data as PD
import sq_pair_fista as F
import sq_pair_mlp as M
import sq_pair_preflight as PF
import sq_pair_rules as R

import sq_data as D

DEFAULT_CONFIG = C.CONFIG_PATH
SPLITS = C.SPLITS


def build_runner():
    from ksvd_research.runner_api import Runner

    return Runner(
        name=C.ROUND_ID,
        default_config=DEFAULT_CONFIG,
        default_study=C.STUDY_ID,
        description=(
            "Lipo explicit structural-rule pilot: grouped sparse additive base / "
            "marginal-NULL / real anchor-pair relation readouts, MLP probe and live "
            "C1 on official ogbg-mollipo (test_policy=always, seed 0, 180 epochs)"
        ),
        run_module=sys.modules[__name__],
    )


# ---------------------------------------------------------------------------
# identity / paths
# ---------------------------------------------------------------------------


def fingerprints(config: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    payload = D.load_cache(C.DATASET)
    graph_path = D.cache_path(C.DATASET)
    lipo_path = D.lipo_raw_path(C.DATASET)
    entry = {
        "round_id": C.ROUND_ID,
        "dataset": C.DATASET,
        "graph_cache_sha256": C.sha256_file(graph_path) if graph_path.is_file() else None,
        "lipo_raw_sha256": C.sha256_file(lipo_path) if lipo_path.is_file() else None,
        "split_sizes": {split: int(R.sorted_split_ids(payload, split).size) for split in SPLITS},
        "split_row_id_sha256": {
            split: C.sha256_array(R.sorted_split_ids(payload, split)) for split in SPLITS
        },
        "label_hash_all_rows": C.sha256_array(np.asarray(payload["labels"], dtype=np.float64)),
    }
    return entry, C.sha256_json(entry)


def _dir(config: Mapping[str, Any], key: str, default: Path) -> Path:
    raw = config.get(key)
    if raw in (None, ""):
        return default
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = C.REPO_ROOT / path
    return path


def _prep_dir(config: Mapping[str, Any]) -> Path:
    return _dir(config, "prep_dir", C.RUNS_DIR / "prep")


def _c1_dir(config: Mapping[str, Any]) -> Path:
    return _dir(config, "c1_dir", C.RUNS_DIR / "c1")


def _rules_dir(config: Mapping[str, Any]) -> Path:
    return _dir(config, "rules_dir", C.RUNS_DIR / "rules")


def _analysis_dir(config: Mapping[str, Any]) -> Path:
    return _dir(config, "analysis_dir", C.ANALYSIS_DIR)


def _write_stage_status(
    stage: str,
    out: Path,
    payload: Mapping[str, Any],
    *,
    started: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    provenance = dict(started or C.run_provenance())
    body = {
        **payload,
        "stage": stage,
        "run_provenance": provenance,
        "started_at": provenance.get("started_at"),
        "started_epoch": float(provenance.get("started_epoch", time.time())),
    }
    body = C.finish_provenance(body)
    C.write_json(Path(out) / "status.json", body)
    return body


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def run(config: Mapping[str, Any], context) -> Any:
    from ksvd_research.runtime.manifest import RunResult

    if context.test_access != "granted":
        raise RuntimeError(
            "this round requires test_policy=always; the control plane blocked official test access"
        )
    stage = str(config.get("stage", "prepare"))
    ledger = C.ledger(stage)
    t0 = time.perf_counter()
    started = C.run_provenance()
    if stage == "prepare":
        metrics, out = _stage_prepare(config, ledger, started)
    elif stage == "worker_c1":
        metrics, out = _stage_worker_c1(config, ledger, started)
    elif stage == "worker_rules":
        metrics, out = _stage_worker_rules(config, ledger, started)
    elif stage == "aggregate":
        import sq_pair_analysis as AN

        metrics, out = AN.aggregate(config, ledger)
    else:
        raise ValueError(f"unknown stage {stage!r}")
    metrics = {**metrics, "stage": stage, "wall_seconds": time.perf_counter() - t0}
    artifact_dir = Path(context.artifact_dir)
    C.write_json(artifact_dir / f"{stage}_summary.json", {**metrics, "ledger": ledger.summary()})
    return RunResult(
        metrics=metrics,
        status="completed",
        artifacts=[
            f"artifacts/{stage}_summary.json",
            "artifacts/ledger.jsonl",
            str(Path(out) / "status.json"),
        ],
    )


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------


def _stage_prepare(
    config: Mapping[str, Any], ledger, started: Mapping[str, Any]
) -> tuple[dict[str, Any], Path]:
    out = C.ensure_dir(_prep_dir(config))
    payload = R.load_payload_verified()
    with C.SC.stage_timer(ledger, "identity_verify"):
        identity = R.verify_identity(payload)
        bundle_identity = R.load_bundle_identity()
    C.write_json(out / "identity.json", {"identity": identity, "bundle": bundle_identity})
    with C.SC.stage_timer(ledger, "build_rule_library"):
        summary = R.build_all(payload, out)
    device = "cuda:0" if _cuda_available() else "cpu"
    with C.SC.stage_timer(ledger, "mlp_smoke", device=device):
        mlp_smoke = M.smoke_mlp(
            prep_dir=out,
            device=device,
            n_rows=int(config.get("smoke_mlp_rows", 16)),
        )
    C.write_json(out / "mlp_smoke.json", mlp_smoke)
    with C.SC.stage_timer(ledger, "preflight"):
        preflight = PF.run_preflight(
            payload=payload,
            prep_dir=out,
            bundle_identity=bundle_identity,
            mlp_smoke=mlp_smoke,
            ledger=ledger,
        )
    status = _write_stage_status(
        "prepare",
        out,
        {
            "state": "completed",
            "dataset": C.DATASET,
            "summary": summary,
            "identity": identity,
            "bundle_identity": bundle_identity,
            "preflight": {
                "pass": preflight["pass"],
                "n_failed": preflight["n_failed"],
                "path": str(out / "preflight.json"),
            },
            "mlp_smoke": mlp_smoke,
            "protocol_hash": C.protocol_hash(),
            "code_hashes": C.code_hashes(),
            "device": device,
            "prep_dir": str(out),
        },
        started=started,
    )
    metrics = {
        "n_b_records": summary["n_b_records"],
        "n_s2_selected": summary["n_s2_selected"],
        "n_relation_slots": summary["n_relation_slots"],
        "n_common_columns": summary["n_common_columns"],
        "n_extension_columns": summary["n_extension_columns"],
        "coverage_valid": summary["coverage"]["valid"]["n_rows_with_extension_difference"],
        "preflight_pass": preflight["pass"],
        "mlp_smoke_pass": bool(mlp_smoke["pass"]),
        "prep_dir": str(out),
        "status": status["state"],
    }
    return metrics, out


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------------------
# C1 worker
# ---------------------------------------------------------------------------


def _load_prep_for_worker(prep_dir: Path) -> dict[str, Any]:
    if not (prep_dir / "rules_vocabulary.json").is_file():
        raise FileNotFoundError(f"frozen prep artefacts missing at {prep_dir}; run stage=prepare first")
    if not (prep_dir / "status.json").is_file():
        raise FileNotFoundError(f"prep status.json missing at {prep_dir}")
    status = C.read_json(prep_dir / "status.json")
    if status.get("state") != "completed":
        raise RuntimeError(f"prep is not complete: {status.get('state')}")
    if not C.read_json(prep_dir / "preflight.json").get("pass"):
        raise RuntimeError("prep preflight did not pass; refusing to run a formal arm")
    return {
        "status": status,
        "vocabulary": C.read_json(prep_dir / "rules_vocabulary.json"),
        "column_order": C.read_json(prep_dir / "column_order.json"),
        "hashes": C.read_json(prep_dir / "hashes.json"),
        "summary": C.read_json(prep_dir / "summary.json"),
    }


def _stage_worker_c1(
    config: Mapping[str, Any], ledger, started: Mapping[str, Any]
) -> tuple[dict[str, Any], Path]:
    import sq_pair_c1 as C1

    out = C.ensure_dir(_c1_dir(config))
    prep = _load_prep_for_worker(_prep_dir(config))
    C.write_json(
        out / "prep_reference.json",
        {
            "prep_dir": str(_prep_dir(config)),
            "prep_status_sha256": C.sha256_file(_prep_dir(config) / "status.json"),
            "rules_vocabulary_sha256": prep["hashes"]["rules_vocabulary_sha256"],
            "graph_cache_sha256": prep["hashes"]["graph_cache_sha256"],
        },
    )
    device = str(config.get("device_c1", "cuda:0"))
    with C.SC.stage_timer(ledger, "c1_run", device=device):
        summary = C1.run_c1(
            out_dir=out / "C1",
            device=device,
            smoke=True,
        )
    # uniform top-level names for the report/aggregate
    shutil.copy2(out / "C1" / "curve.csv", out / "curve.csv")
    shutil.copy2(out / "C1" / "best_predictions.npz", out / "best_predictions.npz")
    shutil.copy2(out / "C1" / "final_predictions.npz", out / "epoch180_predictions.npz")
    _write_stage_status(
        "worker_c1",
        out,
        {
            "state": "completed",
            "arm": "C1",
            "epochs": int(C.EPOCHS),
            "recipe": C.C1_RECIPE,
            "device": device,
            "n_parameters": int(summary["n_parameters"]),
            "selected": summary["selected"],
            "epoch180": summary["final"],
            "best_epoch": int(summary["selected"]["epoch"]),
            "train_seconds": float(summary.get("train_seconds", 0.0)),
            "eval_seconds": float(summary.get("eval_seconds", 0.0)),
            "seconds_total": float(summary.get("seconds_total", 0.0)),
            "init": summary.get("init"),
            "smoke_pass": bool((summary.get("smoke") or {}).get("pass")),
            "task": summary.get("task"),
            "paths": {
                "c1_dir": str(out / "C1"),
                "curve": str(out / "curve.csv"),
                "best_predictions": str(out / "best_predictions.npz"),
                "epoch180_predictions": str(out / "epoch180_predictions.npz"),
                "trajectory": str(out / "C1" / "trajectory.json"),
            },
        },
        started=started,
    )
    metrics = {
        "arm": "C1",
        "best_epoch": int(summary["selected"]["epoch"]),
        "best_valid_rmse": float(summary["selected"]["valid_rmse_raw"]),
        "test_rmse_best": float(summary["selected"]["test_rmse_raw"]),
        "test_rmse_epoch180": float(summary["final"]["test_rmse_raw"]),
        "n_parameters": int(summary["n_parameters"]),
        "smoke_pass": bool((summary.get("smoke") or {}).get("pass")),
    }
    return metrics, out


# ---------------------------------------------------------------------------
# rules worker (MLP + three additive arms)
# ---------------------------------------------------------------------------


def _lambda_max_real(prep_dir: Path) -> dict[str, Any]:
    system = F.load_system(prep_dir, "REAL")
    mean_x = system.x_train.mean(axis=0)
    mean_z = float(system.z_train.mean())
    zc = system.z_train - mean_z
    xc = system.x_train - mean_x[None, :]
    lam_max, norms = F.lambda_max_groups(xc, zc, system.n_train, system.groups)
    return {
        "lambda_max_real": float(lam_max),
        "max_group_gradient_norms": float(max(norms) if norms else 0.0),
        "n_groups": int(system.n_groups),
        "n_columns": int(system.n_columns),
    }


def _stage_worker_rules(
    config: Mapping[str, Any], ledger, started: Mapping[str, Any]
) -> tuple[dict[str, Any], Path]:
    out = C.ensure_dir(_rules_dir(config))
    prep_dir = _prep_dir(config)
    prep = _load_prep_for_worker(prep_dir)
    C.write_json(
        out / "prep_reference.json",
        {
            "prep_dir": str(prep_dir),
            "prep_status_sha256": C.sha256_file(prep_dir / "status.json"),
            "rules_vocabulary_sha256": prep["hashes"]["rules_vocabulary_sha256"],
            "scales_sha256": prep["hashes"]["scales_sha256"],
            "matrices_sha256": prep["hashes"]["matrices_sha256"],
        },
    )
    resume = bool(config.get("resume", True))
    device = str(config.get("device_mlp", "cuda:0")) if _cuda_available() else "cpu"
    arms: dict[str, Any] = {}

    # ---- MLP on the GPU first -------------------------------------------
    with C.SC.stage_timer(ledger, "MLP", device=device):
        arms["MLP"] = M.fit_mlp(
            prep_dir=prep_dir,
            out_dir=out / "MLP",
            device=device,
            epochs=int(config.get("epochs", C.EPOCHS)),
            resume=resume,
            ledger=ledger,
        )
    C.write_json(out / "MLP" / "status.json", arms["MLP"])

    # ---- one frozen lambda for all additive arms ------------------------
    with C.SC.stage_timer(ledger, "lambda_max_real"):
        lam = _lambda_max_real(prep_dir)
    C.write_json(out / "lambda_max_real.json", lam)

    # ---- BASE / NULL / REAL on CPU --------------------------------------
    for arm in C.ADDITIVE_ARMS:
        with C.SC.stage_timer(ledger, f"additive_{arm}", arm=arm):
            arm_summary = F.fit_additive(
                prep_dir=prep_dir,
                arm=arm,
                out_dir=out / arm,
                lambda_max_real=float(lam["lambda_max_real"]),
                epochs=int(config.get("epochs", C.EPOCHS)),
                resume=resume,
                ledger=ledger,
            )
        arms[arm] = arm_summary

    status = _write_stage_status(
        "worker_rules",
        out,
        {
            "state": "completed",
            "device_mlp": device,
            "cpu_threads": int(config.get("cpu_threads", 4)),
            "lambda_max_real": lam,
            "arms": {
                arm: {
                    "state": arm_summary.get("state"),
                    "best": arm_summary.get("best"),
                    "epoch180": arm_summary.get("epoch180"),
                    "n_parameters": arm_summary.get("n_parameters"),
                    "n_columns": arm_summary.get("n_columns"),
                    "n_groups": arm_summary.get("n_groups"),
                    "lambda": arm_summary.get("lambda"),
                    "fit_seconds": arm_summary.get("fit_seconds"),
                    "train_seconds": arm_summary.get("train_seconds"),
                }
                for arm, arm_summary in arms.items()
            },
            "paths": {
                "mlp": str(out / "MLP"),
                "base": str(out / "BASE"),
                "null": str(out / "NULL"),
                "real": str(out / "REAL"),
            },
        },
        started=started,
    )
    metrics = {
        "device_mlp": device,
        "mlp_best_valid_rmse": float(arms["MLP"]["best"]["valid_rmse_raw"]),
        "mlp_test_rmse_best": float(arms["MLP"]["best"]["scores"]["test"]["rmse_raw"]),
        "base_minus32_best_valid_rmse": float(arms["BASE"]["best"]["-32"]["valid_rmse_raw"]),
        "null_minus32_best_valid_rmse": float(arms["NULL"]["best"]["-32"]["valid_rmse_raw"]),
        "real_minus32_best_valid_rmse": float(arms["REAL"]["best"]["-32"]["valid_rmse_raw"]),
        "real_full_best_valid_rmse": float(arms["REAL"]["best"]["FULL"]["valid_rmse_raw"]),
        "lambda": float(lam["lambda_max_real"]),
        "status": status["state"],
    }
    return metrics, out
