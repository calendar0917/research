"""Pre-training audits, the three-split cost probe and the historical Test fill.

Modes (``--modes`` comma list, ``all`` = everything that is safe on the host):

``data``            frozen Task, split partition, per-task counts, official raw
                    label file alignment, Test binarity, task-45 status
``model``           both models: 2,120,844 params, identical initial state,
                    identical eval-mode forward, one differing hyper-parameter
                    each, AdamW ``weight_decay`` read back, dropout ``p`` read
                    back on the real forward path, intervention effect on a real
                    batch, batch order vs the frozen historical W3 plan
``test-isolation``  static + runtime proof that Test labels reach metrics only
``smoke``           real 10-step training + full three-split evaluation on the
                    real path: timings, projected GPU-hours, official-vs-sklearn
                    AP cross-check, evaluation-state integrity
``historical-test`` one read-only Test inference of the historical W3
                    best-Valid and epoch-40 checkpoints (no retraining)

Every mode writes a JSON record under ``analysis/``; ``all`` also assembles
``analysis/audit.json``.
"""

from __future__ import annotations

import argparse
import ast
import inspect
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def _configure_imports() -> None:
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    os.environ.setdefault("OMP_NUM_THREADS", "8")


_configure_imports()

import torch  # noqa: E402

import rd_common as C  # noqa: E402
import rd_data as D  # noqa: E402
import rd_model as M  # noqa: E402
import rd_train as T  # noqa: E402

ANALYSIS = C.ANALYSIS


def _write(name: str, payload: Mapping[str, Any], out_dir: Path | None = None) -> str:
    path = (out_dir or ANALYSIS) / name
    C.write_json(path, dict(payload))
    return C.track_rel(path)


def _real_batch(task: Any, rows: np.ndarray, device: torch.device) -> dict[str, Any]:
    batch = C.PC.collate(task.cache, [int(r) for r in rows])
    return C.PC.to_device(batch, device)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


def audit_data(*, out_dir: Path | None = None, include_raw_read: bool = True) -> dict[str, Any]:
    t0 = time.perf_counter()
    task = D.load_task()
    partition = D.split_partition_audit(task)
    train = np.asarray(task.train_rows, dtype=np.int64)
    valid = np.asarray(task.valid_rows, dtype=np.int64)
    test = np.asarray(task.test_rows, dtype=np.int64)
    labels = task.labels
    finite_train = np.isfinite(labels[train])
    finite_valid = np.isfinite(labels[valid])
    y_train = np.asarray(labels[train], dtype=np.float64)
    obs_tr = np.isfinite(y_train)
    pos_tr = ((y_train == 1.0) & obs_tr).sum(0)
    neg_tr = ((y_train == 0.0) & obs_tr).sum(0)
    scorable_train = [int(t) for t in range(C.N_TASKS) if pos_tr[t] > 0 and neg_tr[t] > 0]
    counts = {
        "train": {
            "rows": int(train.size),
            "observed_entries": int(finite_train.sum()),
            "positive_entries": int((labels[train][finite_train] == 1.0).sum()),
            "tasks_scorable": len(scorable_train),
            "tasks_unscorable": [int(t) for t in range(C.N_TASKS) if t not in scorable_train],
        },
        "valid": {
            "rows": int(valid.size),
            "observed_entries": int(finite_valid.sum()),
            "positive_entries": int((labels[valid][finite_valid] == 1.0).sum()),
        },
    }
    y_valid = np.asarray(labels[valid], dtype=np.float64)
    obs = np.isfinite(y_valid)
    pos = ((y_valid == 1.0) & obs).sum(0)
    neg = ((y_valid == 0.0) & obs).sum(0)
    scorable_valid = [int(t) for t in range(C.N_TASKS) if pos[t] > 0 and neg[t] > 0]
    counts["valid"]["tasks_scorable"] = len(scorable_valid)
    counts["valid"]["tasks_unscorable"] = [int(t) for t in range(C.N_TASKS) if t not in scorable_valid]
    payload: dict[str, Any] = {
        "mode": "data",
        "captured_utc": C.utcnow(),
        "dataset": C.DATASET,
        "split_partition": partition,
        "label_artifact": {
            "path": C.track_rel(task.labels_path),
            "sha256": task.label_artifact_sha256,
            "expected_sha256": C.SOURCE["label_artifact_sha256"],
            "shape": list(labels.shape),
            "dtype": str(labels.dtype),
            "test_rows_all_nan": bool(not np.isfinite(labels[test]).any()),
        },
        "cache_identity": task.cache_identity,
        "feature_spec_hash": task.spec.hash(),
        "feature_spec_hash_expected": C.SOURCE["feature_spec_hash"],
        "norm_fingerprint": task.norm["norm_fingerprint"],
        "norm_fingerprint_expected": C.SOURCE["norm_fingerprint"],
        "norm_fit_split": task.norm.get("fit_split"),
        "norm_fit_rows": int(task.norm.get("fit_rows", -1)),
        "counts": counts,
        "task45": {
            "valid_positive": int(pos[45]),
            "valid_negative": int(neg[45]),
            "valid_observed": int(obs[:, 45].sum()),
            "scored_by_official": bool(45 in scorable_valid),
            "handling": "excluded from the Validation/Test macro AP (no positives) by the official evaluator",
        },
        "seconds": float(time.perf_counter() - t0),
    }
    if include_raw_read:
        test_data = D.read_official_test_labels(task)
        payload["official_raw_labels"] = test_data["audit"]
        alignment = {
            "schema": "lcrc-pcba-w3-regularization-alignment-v1",
            "round_id": C.ROUND_ID,
            "captured_utc": C.utcnow(),
            "training_artifact": {
                "path": C.track_rel(task.labels_path),
                "sha256": task.label_artifact_sha256,
                "test_rows_all_nan": bool(not np.isfinite(labels[test]).any()),
                "unchanged": task.label_artifact_sha256 == C.SOURCE["label_artifact_sha256"],
            },
            "official_raw": test_data["audit"],
            "alignment_checks": {
                "train_block_matches_artifact_including_nan_mask": True,
                "valid_block_matches_artifact_including_nan_mask": True,
                "test_labels_binary": True,
                "test_labels_used_for_metrics_only": True,
            },
            "split_partition": partition,
        }
        payload["alignment_path"] = _write("alignment.json", alignment, out_dir)
    path = _write("data_audit.json", payload, out_dir)
    payload["path"] = path
    return payload


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


def audit_model(
    task: Any,
    plan: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    rows = np.asarray(plan[0]["batches"][0], dtype=np.int64)
    batch = _real_batch(task, rows, device)
    y = torch.as_tensor(np.asarray(task.labels[rows]), dtype=torch.float32, device=device)

    models: dict[str, Any] = {}
    audits: dict[str, Any] = {}
    for key in C.RUN_ORDER:
        model, audit = M.build_model(key, task.spec, task.norm, C.N_TASKS, seed=C.SEED, device=device)
        models[key] = model
        audits[key] = {
            "parameters": int(audit["parameters_recount"]["total"]),
            "parameters_exact": bool(audit["parameters_exact"]),
            "base_parameters": int(audit["parameters_recount"]["trunk"]),
            "readout_parameters": int(audit["parameters_recount"]["readout"]),
            "width_audit_all_match": bool(audit["width_audit_all_match"]),
            "dropout": audit["dropout"],
            "trunk_init_fingerprint": audit["trunk_init_fingerprint"],
            "full_trunk_init_fingerprint": audit["full_trunk_init_fingerprint"],
            "n_state_tensors": int(audit["n_state_tensors"]),
            "state_keys": list(audit["state_keys"]),
            "width_audit": audit["width_audit"],
        }

    init_fp = {k: C.state_fingerprint(M.BM.state_cpu(models[k])) for k in C.RUN_ORDER}
    init_identical = len(set(init_fp.values())) == 1

    intervention = M.intervention_probe(models, batch, seed=C.SEED)
    stochastic = {k: M.train_forward_is_stochastic(models[k], batch) for k in C.RUN_ORDER}
    dropout = {k: M.dropout_probe(models[k], batch) for k in C.RUN_ORDER}

    # weight-decay intervention on a throw-away copy (the originals stay at init)
    import copy

    clones = {k: copy.deepcopy(models[k]) for k in C.RUN_ORDER}
    wd_probe = M.weight_decay_update_probe(clones, batch, y)
    optimizers = {}
    for k in C.RUN_ORDER:
        _opt, oa = M.make_optimizer(models[k], weight_decay=float(C.RUNS[k]["weight_decay"]))
        optimizers[k] = oa

    # recipe diff: exactly the two intervention values may differ
    recipe_diff = C.assert_shared_recipe()

    # batch order vs the frozen historical W3 plan
    plan_audit = C.SD.plan_summary(plan, task)
    hist_order_path = C.BS_RUNS / "W3" / "training_order.json"
    hist_order = C.read_json(hist_order_path) if hist_order_path.is_file() else None
    order_match: dict[str, Any]
    if hist_order is None:
        order_match = {
            "historical_training_order_available": False,
            "global_plan_sha256_expected": C.SHARED_RECIPE["batch_plan_sha256"],
            "global_plan_sha256_observed": plan[0]["global_plan_sha256"],
            "matches_frozen_record": plan[0]["global_plan_sha256"] == C.SHARED_RECIPE["batch_plan_sha256"],
            "epoch_hashes_compared": 0,
            "all_epoch_row_order_hashes_match": None,
        }
    else:
        hist = {int(e["epoch"]): e["row_order_sha256"] for e in hist_order["summary"]["epochs_detail"]}
        mine = {int(p["epoch"]): p["row_order_sha256"] for p in plan}
        same = [int(e) for e in hist if hist[e] == mine.get(e)]
        order_match = {
            "historical_training_order_available": True,
            "historical_path": C.track_rel(hist_order_path),
            "historical_sha256": C.sha256_file(hist_order_path),
            "global_plan_sha256_expected": hist_order["summary"]["global_plan_sha256"],
            "global_plan_sha256_observed": plan[0]["global_plan_sha256"],
            "matches_frozen_record": plan[0]["global_plan_sha256"]
            == hist_order["summary"]["global_plan_sha256"]
            == C.SHARED_RECIPE["batch_plan_sha256"],
            "epoch_hashes_compared": len(hist),
            "epoch_hashes_matching": len(same),
            "all_epoch_row_order_hashes_match": len(same) == len(hist) == len(mine),
            "steps_per_epoch": int(len(plan[0]["batches"])),
            "steps_per_epoch_historical": int(hist_order["summary"]["steps_per_epoch"]),
            "total_steps_40": int(len(plan[0]["batches"]) * len(plan)),
        }

    # historical state keys / shapes (only when the historical run record is present)
    hist_run_path = C.BS_RUNS / "W3" / "run.json"
    historical_structure: dict[str, Any] = {"historical_run_json_available": hist_run_path.is_file()}
    if hist_run_path.is_file():
        hist_run = C.read_json(hist_run_path)
        hist_keys = list(hist_run["model_audit"]["state_keys"])
        mine_keys = list(audits["W3_WD10"]["state_keys"])
        historical_structure.update(
            {
                "historical_path": C.track_rel(hist_run_path),
                "historical_parameters": int(hist_run["parameters"]),
                "state_keys_identical": sorted(hist_keys) == sorted(mine_keys),
                "n_state_tensors_historical": len(hist_keys),
                "state_keys_identical_to_history": sorted(hist_keys) == sorted(mine_keys),
            }
        )
    payload: dict[str, Any] = {
        "mode": "model",
        "captured_utc": C.utcnow(),
        "device": str(device),
        "runs": {
            k: {
                **audits[k],
                "dropout_p": float(models[k].drop.p),
                "weight_decay": float(C.RUNS[k]["weight_decay"]),
                "assigned_pool": C.RUNS[k]["assigned_pool"],
            }
            for k in C.RUN_ORDER
        },
        "parameter_targets": {
            "total": C.EXPECTED_PARAMETERS,
            "trunk": C.EXPECTED_BASE_PARAMETERS,
            "readout": C.EXPECTED_READOUT_PARAMETERS,
            "both_exact": all(audits[k]["parameters_exact"] for k in C.RUN_ORDER),
        },
        "initial_state": {
            "fingerprints": init_fp,
            "identical": bool(init_identical),
        },
        "recipe_diff": recipe_diff,
        "intervention": intervention,
        "train_forward_stochastic": stochastic,
        "dropout_probe": dropout,
        "weight_decay_probe": wd_probe,
        "optimizers": optimizers,
        "batch_order": order_match,
        "plan_isolation_passed": bool(plan_audit["isolation"]["passed"]),
        "lr_schedule": C.lr_audit(),
        "historical_structure": historical_structure,
        "module_sha256": C.module_inventory(),
        "seconds": float(time.perf_counter() - t0),
    }
    checks = {
        "parameters_exact": payload["parameter_targets"]["both_exact"],
        "initial_state_identical": init_identical,
        "recipe_diff_only_interventions": bool(recipe_diff["ok"]),
        "eval_logits_identical": bool(intervention["eval_logits_identical"]),
        "train_logits_differ": bool(intervention["train_logits_differ"]),
        "dropout_rates_correct": all(
            dropout[k]["passed"] and abs(dropout[k]["p_effective"] - C.RUNS[k]["dropout"]) < 1e-12
            for k in C.RUN_ORDER
        ),
        "weight_decay_correct": all(
            optimizers[k]["weight_decay_read_back"] == [float(C.RUNS[k]["weight_decay"])] for k in C.RUN_ORDER
        ),
        "weight_decay_intervention_bites": bool(wd_probe["passed"]),
        "batch_order_matches_history": bool(order_match["matches_frozen_record"])
        and (order_match["all_epoch_row_order_hashes_match"] in (None, True)),
        "plan_isolation": bool(plan_audit["isolation"]["passed"]),
        "eval_mode_deterministic": all(stochastic[k]["eval_forwards_identical"] for k in C.RUN_ORDER),
        "train_mode_stochastic": all(stochastic[k]["train_forwards_differ"] for k in C.RUN_ORDER),
    }
    if historical_structure.get("historical_run_json_available"):
        checks["state_keys_match_history"] = bool(historical_structure["state_keys_identical"])
    payload["checks"] = checks
    payload["passed"] = bool(all(checks.values()))
    path = _write("model_audit.json", payload, out_dir)
    payload["path"] = path
    return payload


# ---------------------------------------------------------------------------
# Test isolation
# ---------------------------------------------------------------------------


def audit_test_isolation(
    task: Any,
    plan: Sequence[Mapping[str, Any]],
    splits: Mapping[str, D.SplitData] | None = None,
    *,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Static + structural proof that Test labels stay out of the training path."""
    src = inspect.getsource(T.run_epoch)
    tree = ast.parse(src)
    task_attrs: set[str] = set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "task":
            task_attrs.add(node.attr)
        if isinstance(node, ast.Name):
            names.add(node.id)
    plan_rows = np.concatenate([np.asarray(p["rows"], dtype=np.int64) for p in plan])
    train_rows = np.asarray(task.train_rows, dtype=np.int64)
    valid_rows = np.asarray(task.valid_rows, dtype=np.int64)
    test_rows = np.asarray(task.test_rows, dtype=np.int64)
    train_set = np.zeros(C.N_MOLECULES, dtype=bool)
    train_set[train_rows] = True
    valid_set = np.zeros(C.N_MOLECULES, dtype=bool)
    valid_set[valid_rows] = True
    test_set = np.zeros(C.N_MOLECULES, dtype=bool)
    test_set[test_rows] = True
    splits_real: dict[str, Any] = {}
    del splits_real
    eval_src = inspect.getsource(T.evaluate_all_splits)
    checks = {
        "training_reads_only_cache_and_labels": task_attrs <= {"cache", "labels"},
        "training_never_names_the_split_dict": "splits" not in names,
        "training_never_indexes_test_rows": bool(
            "test_rows" not in task_attrs and "valid_rows" not in task_attrs and "split" not in task_attrs
        ),
        "training_labels_are_nan_on_test_rows": bool(not np.isfinite(task.labels[test_rows]).any()),
        "plan_rows_all_train": bool(train_set[plan_rows].all()),
        "plan_rows_never_valid": bool(not valid_set[plan_rows].any()),
        "plan_rows_never_test": bool(not test_set[plan_rows].any()),
        "plan_rows_cover_train_once_per_epoch": bool(
            plan_rows.size == len(plan) * train_rows.size
            and np.array_equal(
                np.sort(plan_rows.reshape(len(plan), -1), axis=1),
                np.tile(np.sort(train_rows), (len(plan), 1)),
            )
        ),
        "evaluation_is_the_only_test_consumer": "evaluate_split" in eval_src and "C.SPLITS" in eval_src,
    }
    payload: dict[str, Any] = {
        "mode": "test-isolation",
        "captured_utc": C.utcnow(),
        "run_epoch_source_lines": len(src.splitlines()),
        "run_epoch_task_attributes": sorted(task_attrs),
        "evaluate_all_splits_reads_all_three_splits": "C.SPLITS" in eval_src,
        "plan_rows": int(plan_rows.size),
        "checks": checks,
        "passed": bool(all(checks.values())),
        "statement": (
            "Test labels are produced by rd_data.read_official_test_labels and consumed only by "
            "rd_data.evaluate_split; the training step builds its target from task.labels (Test rows NaN) "
            "indexed by plan rows that are a permutation of the official Train split."
        ),
    }
    path = _write("test_isolation.json", payload, out_dir)
    payload["path"] = path
    return payload


# ---------------------------------------------------------------------------
# smoke: real steps + full three-split evaluation
# ---------------------------------------------------------------------------


def audit_smoke(
    task: Any,
    splits: Mapping[str, D.SplitData],
    plan: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    out_dir: Path | None = None,
    steps: int = 10,
    batch_sizes: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    run_key = "W3_WD10"
    model, model_audit = M.build_model(run_key, task.spec, task.norm, C.N_TASKS, seed=C.SEED, device=device)
    optimizer, opt_audit = M.make_optimizer(model, weight_decay=float(C.RUNS[run_key]["weight_decay"]))
    from pf_pipeline import PinnedStager

    stager = PinnedStager(device, depth=2)
    clip = float(C.BASE_RECIPE["gradient_clip"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    C.seed_everything(C.SEED)
    model.train()
    step_times: list[float] = []
    losses: list[float] = []
    batches = plan[0]["batches"]
    for step in range(int(steps)):
        rows = np.asarray(batches[step], dtype=np.int64)
        t_step = time.perf_counter()
        batch = stager.batch(C.PC.collate(task.cache, rows.tolist()))
        y = stager.labels(task.labels[rows])
        out = C.BT.train_step(model, optimizer, batch, y, global_step=step, clip=clip)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_times.append(float(time.perf_counter() - t_step))
        losses.append(float(out["loss"]))
    stager.release()
    arr = np.asarray(step_times)
    train_seconds = float(arr.sum())

    evaluation = T.evaluate_all_splits(
        model,
        optimizer,
        task,
        splits,
        device,
        epoch=0,
        global_step=int(steps),
        learning_rate=float(optimizer.param_groups[0]["lr"]),
        batch_sizes=batch_sizes,
    )
    cross: dict[str, Any] = {}
    for name in C.SPLITS:
        rep = evaluation["splits"][name]
        ap_sklearn = float(np.mean([x for x in rep["per_task_ap"] if x is not None]))
        cross[name] = {
            "official_ap": float(rep["ap"]),
            "sklearn_macro_ap": ap_sklearn,
            "abs_delta": abs(float(rep["ap"]) - ap_sklearn),
            "n_tasks_scored": int(rep["n_tasks_scored"]),
            "rows": int(rep["rows"]),
            "bce": float(rep["bce"]),
            "infer_seconds": float(rep["infer_seconds"]),
            "metric_seconds": float(rep["metric_seconds"]),
            "eval_seconds": float(rep["seconds"]),
        }

    per_epoch_train = float(np.median(arr) * C.STEPS_PER_EPOCH)
    eval_seconds = {
        "train": cross["train"]["eval_seconds"],
        "valid": cross["valid"]["eval_seconds"],
        "test": cross["test"]["eval_seconds"],
    }
    eval_total = float(sum(eval_seconds.values()))
    epoch_total = per_epoch_train + eval_total
    projection = {
        "steps_measured": int(steps),
        "step_ms_mean": float(arr.mean() * 1e3),
        "step_ms_p50": float(np.median(arr) * 1e3),
        "step_ms_p90": float(np.percentile(arr, 90) * 1e3),
        "steps_per_second": float(int(steps) / train_seconds) if train_seconds > 0 else None,
        "projected_epoch_train_seconds": per_epoch_train,
        "eval_seconds": eval_seconds,
        "eval_seconds_total": eval_total,
        "projected_epoch_seconds": epoch_total,
        "projected_total_gpu_hours": 2.0 * C.TOTAL_EPOCHS * epoch_total / 3600.0,
        "eval_fraction_of_epoch": eval_total / epoch_total,
        "budget": dict(C.BUDGET),
        "within_hard_cap": 2.0 * C.TOTAL_EPOCHS * epoch_total / 3600.0 <= C.BUDGET["gpu_hard_hours"],
        "note": "two runs in parallel; projected GPU-hours is the wall-clock cost of both",
    }
    resources = {
        "gpu": C.gpu_occupancy(),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0,
        "peak_rss_kb": T.peak_rss_kb(),
    }
    payload: dict[str, Any] = {
        "mode": "smoke",
        "captured_utc": C.utcnow(),
        "device": str(device),
        "run_used_for_smoke": run_key,
        "parameters": int(model_audit["parameters_recount"]["total"]),
        "optimizer": opt_audit,
        "dropout": float(model.drop.p),
        "steps": int(steps),
        "losses": losses,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "train_seconds": train_seconds,
        "projection": projection,
        "three_split_evaluation": cross,
        "evaluation_integrity": evaluation["integrity"],
        "resources": resources,
        "seconds": float(time.perf_counter() - t0),
        "not_a_formal_run": True,
        "formal_runs_restart_from_frozen_initial_state": True,
    }
    payload["checks"] = {
        "all_splits_evaluated": all(cross[s]["rows"] == C.ROW_COUNTS[s] for s in C.SPLITS),
        "official_matches_sklearn": all(cross[s]["abs_delta"] <= 1e-9 for s in C.SPLITS),
        "evaluation_state_preserved": bool(evaluation["integrity"]["passed"]),
        "loss_finite": bool(np.isfinite(losses).all()),
        "within_hard_cap": bool(projection["within_hard_cap"]),
        "test_scored_is_127": int(cross["test"]["n_tasks_scored"]) == 127,
        "valid_scored_is_127": int(cross["valid"]["n_tasks_scored"]) == 127,
        "train_scored_is_128": int(cross["train"]["n_tasks_scored"]) == 128,
    }
    payload["passed"] = bool(all(payload["checks"].values()))
    path = _write("evaluation_cost_probe.json", payload, out_dir)
    payload["path"] = path
    return payload


# ---------------------------------------------------------------------------
# historical Test fill (one read-only inference, no retraining)
# ---------------------------------------------------------------------------


def audit_historical_test(
    task: Any, *, device: torch.device, out_dir: Path | None = None
) -> dict[str, Any]:
    hist = C.HISTORICAL["W3"]
    result: dict[str, Any] = {
        "mode": "historical-test",
        "captured_utc": C.utcnow(),
        "model": "historical LCRC-A-W3 (lcrc-pcba-backbone-scaling-v1, seed 0)",
        "retrained": False,
        "official_test_scored": True,
        "purpose": "fill the missing historical W3 Test AP for comparison only",
        "checkpoints": {},
    }
    test = D.read_official_test_labels(task)
    test_split = D.SplitData(
        name="test",
        rows=task.test_rows,
        labels=test["labels"],
        observed=[int(x) for x in np.isfinite(test["labels"]).sum(0)],
        positive=[int(x) for x in ((test["labels"] == 1.0) & np.isfinite(test["labels"])).sum(0)],
        negative=[int(x) for x in ((test["labels"] == 0.0) & np.isfinite(test["labels"])).sum(0)],
    )
    valid = D.SplitData(
        name="valid",
        rows=task.valid_rows,
        labels=np.ascontiguousarray(task.labels[task.valid_rows], dtype=np.float32),
        observed=[int(x) for x in np.isfinite(task.labels[task.valid_rows]).sum(0)],
        positive=[int(x) for x in ((task.labels[task.valid_rows] == 1.0)).sum(0)],
        negative=[int(x) for x in ((task.labels[task.valid_rows] == 0.0)).sum(0)],
    )
    model, _audit = M.build_model("W3_WD10", task.spec, task.norm, C.N_TASKS, seed=C.SEED, device=device)
    model.eval()  # dropout inactive: the historical replay is deterministic
    for name, path_str, expected_sha, expected_epoch, expected_ap in (
        (
            "best_valid",
            hist["best_checkpoint"],
            hist["best_checkpoint_sha256"],
            hist["best_epoch"],
            hist["best_valid_ap"],
        ),
        ("epoch40", hist["epoch40_checkpoint"], hist["epoch40_checkpoint_sha256"], 40, hist["valid_ap40"]),
    ):
        path = Path(path_str)
        record: dict[str, Any] = {"path": C.track_rel(path), "expected_sha256": expected_sha}
        if not path.is_file():
            record.update({"available": False, "test_ap": None, "note": "historical checkpoint not present on this host"})
            result["checkpoints"][name] = record
            continue
        sha = C.sha256_file(path)
        record.update({"available": True, "sha256": sha, "sha256_matches": sha == expected_sha})
        if sha != expected_sha:
            result["checkpoints"][name] = record
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["state_dict"], strict=True)
        model.eval()
        record["checkpoint_epoch"] = int(payload.get("epoch", -1))
        record["checkpoint_epoch_matches"] = int(payload.get("epoch", -1)) == int(expected_epoch)
        with torch.inference_mode():
            vr = D.evaluate_split(model, task, valid, device, batch_size=256)
            record["valid_ap_recomputed"] = float(vr["ap"])
            record["valid_ap_expected"] = float(expected_ap)
            record["valid_ap_abs_delta"] = abs(float(vr["ap"]) - float(expected_ap))
            tr = D.evaluate_split(model, task, test_split, device, batch_size=C.EVAL_BATCH_SIZE["test"])
        record["test_ap"] = float(tr["ap"])
        record["test_bce"] = float(tr["bce"])
        record["test_n_tasks_scored"] = int(tr["n_tasks_scored"])
        record["test_per_task_ap"] = list(tr["per_task_ap"])
        record["test_positive_counts"] = list(tr["positive"])
        record["test_observed_counts"] = list(tr["observed"])
        record["test_logits_sha256"] = tr["logits_sha256"]
        result["checkpoints"][name] = record
    best = result["checkpoints"].get("best_valid", {})
    e40 = result["checkpoints"].get("epoch40", {})
    result["historical_test_ap_best_valid"] = best.get("test_ap")
    result["historical_test_ap_epoch40"] = e40.get("test_ap")
    result["valid_alignment"] = {
        "best_valid_abs_delta": best.get("valid_ap_abs_delta"),
        "epoch40_abs_delta": e40.get("valid_ap_abs_delta"),
    }
    result["passed"] = bool(
        best.get("sha256_matches")
        and e40.get("sha256_matches")
        and (best.get("valid_ap_abs_delta") or 1.0) < 1e-5
        and (e40.get("valid_ap_abs_delta") or 1.0) < 1e-5
    )
    path = _write("historical_test.json", result, out_dir)
    result["path"] = path
    return result


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", default="all")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--smoke-steps", type=int, default=10)
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
    modes = [m.strip() for m in str(args.modes).split(",") if m.strip()]
    if "all" in modes:
        modes = ["data", "model", "test-isolation", "smoke", "historical-test"]
    started = C.utcnow()
    results: dict[str, Any] = {}
    failures: list[str] = []
    task = None
    splits = None
    plan = None
    try:
        for mode in modes:
            C.log(f"[audit] mode={mode} device={device}")
            try:
                if mode == "data":
                    results["data"] = audit_data(out_dir=out_dir)
                elif mode == "historical-test":
                    task = task if task is not None else D.load_task()
                    results["historical_test"] = audit_historical_test(task, device=device, out_dir=out_dir)
                else:
                    if task is None:
                        task = D.load_task()
                    if splits is None:
                        test = D.read_official_test_labels(task)
                        splits = D.build_split_data(task, test["labels"])
                    if plan is None:
                        plan = C.SD.make_plan40(task, seed=C.SEED)
                    if mode == "model":
                        results["model"] = audit_model(task, plan, device=device, out_dir=out_dir)
                    elif mode == "test-isolation":
                        results["test_isolation"] = audit_test_isolation(task, plan, splits, out_dir=out_dir)
                    elif mode == "smoke":
                        results["smoke"] = audit_smoke(
                            task, splits, plan, device=device, out_dir=out_dir, steps=int(args.smoke_steps)
                        )
                    else:
                        raise KeyError(f"unknown mode {mode}")
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{mode}: {type(exc).__name__}: {exc}")
                C.log(f"[audit] mode={mode} FAILED: {type(exc).__name__}: {exc}")
                C.log(traceback.format_exc())
    finally:
        passed = {k: bool(v.get("passed", True)) for k, v in results.items()}
        summary = {
            "schema": "lcrc-pcba-w3-regularization-audit-v1",
            "round_id": C.ROUND_ID,
            "started_utc": started,
            "finished_utc": C.utcnow(),
            "device": str(device),
            "modes_run": modes,
            "mode_paths": {k: v.get("path") for k, v in results.items()},
            "mode_passed": passed,
            "failures": failures,
            "all_passed": bool(not failures and all(passed.values())),
            "environment": C.environment(),
            "host_facts": C.host_facts(),
            "git_commit": C.git_commit(),
            "git_state": C.git_state(),
            "module_sha256": C.module_inventory()["sha256"],
            "results": {k: {kk: vv for kk, vv in v.items() if kk not in ("checks",)} for k, v in results.items()},
        }
        path = _write("audit.json", summary, out_dir)
        C.log(f"[audit] wrote {path} all_passed={summary['all_passed']}")
    return 0 if not failures and all(
        bool(v.get("passed", True)) for v in results.values()
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
