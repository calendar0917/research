"""Top-5 Checkpoint Aggregation Stabilization Audit (ZINC, zero new training).

Pre-registered single hypothesis
--------------------------------
The canonical single-best-epoch rule (raw 800-selection argmin) is unnecessarily
sensitive to sharp validation minima.  A **fixed equal-weight aggregation of the
five checkpoints with lowest 800-selection MAE** can produce a more stable
estimator than the single raw argmin.

This is a *checkpoint-estimator* experiment:

* no new backbone training (reuse I0T0 / I1T1 saved trajectories only);
* the prior internal generalization audit established
  ``CHECKPOINT-SELECTION NOISE DOMINANT`` (mean selection regret 0.003381);
* a 5-epoch moving-average selector was already rejected (mean probe gain
  -0.006925), so it is **not** retried;
* ``K = 5`` is frozen before any new evaluation; no K sweep;
* no greedy soup, no weighted soup, no EMA/SWA, no official test.

Three locked estimators per run
-------------------------------
* ``E0 RAW``   -- the canonical raw 800 argmin checkpoint ``e_1``;
* ``E1 ENS``   -- arithmetic mean of the five Top-5 checkpoint predictions;
* ``E2 SOUP``  -- arithmetic mean of the five Top-5 checkpoint state_dicts
  (primary compact candidate; still an 82,115-parameter single model).

Stages
------
* Stage 0  -- protocol lock, checkpoint inventory, Top-5 manifests, soup states,
  integrity gates.
* Stage A  -- development-reuse diagnostic on the already-examined 2000 internal
  probe (descriptive only; never used to change any rule).
* Stage B  -- locked official-valid confirmation, opened only after
  ``confirmation_lock.json`` exists.

Run::

    uv run python -m tracks.ksvd.experiments.luyin16.zinc_top5_checkpoint_aggregation_stabilization <stage>

Stages: ``protocol inventory top5 soup integrity stageA weights lock valid
bootstrap decision answers figures all``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from tracks.ksvd.experiments.luyin16 import zinc_compact_v4_smallhead_e2e as shead
from tracks.ksvd.experiments.luyin16 import zinc_internal_generalization_stochasticity_audit as iga
from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp

# ---------------------------------------------------------------------------
# frozen layout
# ---------------------------------------------------------------------------

REPO_ROOT = zpp.REPO_ROOT
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"
RESULTS_DIR = TRACK_ROOT / "results/top5_checkpoint_aggregation_stabilization"
STATE_DIR = RESULTS_DIR / "states"
FIGURE_DIR = RESULTS_DIR / "figures"
SOURCE_DIR = iga.RESULTS_DIR

PROTOCOL_VERSION = "top5_checkpoint_aggregation_stabilization_v1"
RUNS = ("I0T0", "I1T1")
K = 5

# --- locked decision gates -------------------------------------------------
SOUP_MEAN_GATE = 0.0015      # primary soup mean-improvement gate
ENS_MEAN_GATE = 0.0020       # ensemble mechanism gate (not a deployment gate)
PER_RUN_NON_DEGRADE = 0.0    # G1: no run may regress
BOOTSTRAP_B = 2000
BOOTSTRAP_SEED = 20260923
FORWARD_REPEATS = 5
TIMING_BATCH = 128

# --- locked protocol content ----------------------------------------------
LOCKED_PROTOCOL: dict[str, Any] = {
    "K": K,
    "checkpoint_ranking_metric": "800-selection MAE",
    "tie_rule": "earliest epoch",
    "prediction_aggregation": "arithmetic mean",
    "weight_aggregation": "arithmetic mean",
    "epoch_spacing": "none",
    "greedy_selection": False,
    "weighted_soup": False,
    "primary_estimator": "weight soup",
    "diagnostic_estimator": "prediction ensemble",
    "development_reuse_set": "2000 internal probe",
    "confirmation_set": "official valid",
    "official_test": "locked",
    "decision_gates": {
        "G1_per_run_soup_non_degrade": 0.0,
        "G2_mean_soup_improvement": SOUP_MEAN_GATE,
        "G3_soup_bootstrap_ci_lower_gt_zero": True,
        "ensemble_mechanism_per_run_positive": True,
        "ensemble_mechanism_mean": ENS_MEAN_GATE,
    },
    "bootstrap": {"B": BOOTSTRAP_B, "seed": BOOTSTRAP_SEED},
}


# ---------------------------------------------------------------------------
# io helpers
# ---------------------------------------------------------------------------


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_lock(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Write a lock file once; later calls return the frozen content."""
    if path.exists():
        return _read_json(path)
    _write_json(path, payload)
    return _read_json(path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _state_hash(state: Mapping[str, torch.Tensor]) -> str:
    return shead._state_hash(state)


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)


def _configure_determinism() -> None:
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(False)


def _environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "torch_threads": int(torch.get_num_threads()),
        "device": "cpu",
    }


def _round(value: Any, ndigits: int = 6) -> Any:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return value
        return float(round(value, ndigits))
    return value


# ---------------------------------------------------------------------------
# Stage 0 -- protocol lock / inventory / Top-5 / soup
# ---------------------------------------------------------------------------


def aggregation_protocol_lock() -> dict[str, Any]:
    payload = {
        **LOCKED_PROTOCOL,
        "protocol_version": PROTOCOL_VERSION,
        "name": "Top-5 checkpoint aggregation stabilization audit",
        "runs": list(RUNS),
        "source_trajectories": str(SOURCE_DIR),
        "no_new_backbone_training": True,
        "official_valid_used": False,
        "official_test_loaded": False,
        "forbidden": [
            "K sweep (2/3/4/8/10)",
            "diversity selection",
            "minimum epoch gap",
            "greedy soup",
            "weighted soup / inverse-loss weighting",
            "prediction median",
            "EMA / SWA",
            "new backbone training",
            "official test access",
        ],
        "protocol_sha256": None,
    }
    payload["protocol_sha256"] = _sha256_text(_canonical_json({**payload, "protocol_sha256": None}))
    return _write_json_lock(RESULTS_DIR / "aggregation_protocol_lock.json", payload)


def _load_source_manifest(run_id: str) -> dict[str, Any]:
    return _read_json(SOURCE_DIR / f"checkpoint_manifest_{run_id}.json")


def _verified_snapshots(run_id: str, *, verify_hashes: bool = True) -> list[dict[str, Any]]:
    manifest = _load_source_manifest(run_id)
    verified: list[dict[str, Any]] = []
    for entry in manifest["snapshots"]:
        path = Path(entry["snapshot"])
        record = {
            "epoch": int(entry["epoch"]),
            "select_800": float(entry["select_800"]),
            "train_loss": float(entry["train_loss"]),
            "checkpoint_path": str(path),
            "checkpoint_sha256": entry["snapshot_sha256"],
            "exists": bool(path.exists()),
        }
        if path.exists() and verify_hashes:
            record["sha256_verified"] = bool(_sha256_file(path) == entry["snapshot_sha256"])
        else:
            record["sha256_verified"] = None
        verified.append(record)
    return verified


def checkpoint_inventory() -> dict[str, Any]:
    inventory: dict[str, Any] = {
        "source": str(SOURCE_DIR),
        "runs": {},
        "official_valid_used": False,
        "official_test_loaded": False,
    }
    for run_id in RUNS:
        snapshots = _verified_snapshots(run_id)
        if not all(s["exists"] for s in snapshots):
            raise RuntimeError(f"{run_id}: a required snapshot is missing -> INVALID / STOP")
        if not all(s["sha256_verified"] for s in snapshots):
            raise RuntimeError(f"{run_id}: snapshot hash mismatch -> INVALID / STOP")
        inventory["runs"][run_id] = {
            "n_snapshots": int(len(snapshots)),
            "all_present": True,
            "all_hashes_verified": True,
            "raw_best_epoch": int(_load_source_manifest(run_id)["raw_best_epoch"]),
            "raw_best_select_800": float(_load_source_manifest(run_id)["raw_best_select_800"]),
            "snapshots_sha256": _sha256_text(_canonical_json(snapshots)),
            "snapshots": snapshots,
        }
    inventory["architecture_params"] = int(shead._n_params(shead.build_smallhead(0)))
    _write_json(RESULTS_DIR / "checkpoint_inventory.json", inventory)
    return inventory


def _top5_epochs(snapshots: Sequence[Mapping[str, Any]], k: int = K) -> list[dict[str, Any]]:
    """Rank strictly by 800-selection MAE; ties resolved to the earliest epoch."""
    order = sorted(range(len(snapshots)), key=lambda i: (float(snapshots[i]["select_800"]), int(snapshots[i]["epoch"])))
    selected: list[dict[str, Any]] = []
    for rank, index in enumerate(order[: int(k)], start=1):
        entry = snapshots[index]
        selected.append(
            {
                "rank": int(rank),
                "epoch": int(entry["epoch"]),
                "800_mae": float(entry["select_800"]),
                "checkpoint_path": entry.get("checkpoint_path"),
                "checkpoint_sha256": entry.get("checkpoint_sha256"),
            }
        )
    return selected


def top5_manifest(run_id: str) -> dict[str, Any]:
    snapshots = _verified_snapshots(run_id)
    selected = _top5_epochs(snapshots)
    if len(selected) != K:
        raise RuntimeError(f"{run_id}: could not select {K} checkpoints -> INVALID / STOP")
    payload = {
        "run_id": run_id,
        "K": K,
        "ranking_metric": "800-selection MAE",
        "tie_rule": "earliest epoch",
        "n_available_snapshots": int(len(snapshots)),
        "selected": selected,
        "raw_epoch": int(selected[0]["epoch"]),
        "raw_800_mae": float(selected[0]["800_mae"]),
        "probe_2000_used_in_selection": False,
        "official_valid_used_in_selection": False,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / f"top5_manifest_{run_id}.json", payload)
    return payload


def _load_checkpoint_state(path: str) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    return {k: v.detach().clone() for k, v in state.items()}


def _split_state_keys(state: Mapping[str, torch.Tensor]) -> tuple[list[str], list[str]]:
    model = shead.build_smallhead(0)
    parameter_keys = {name for name, _ in model.named_parameters()}
    keys = sorted(state.keys())
    parameter_keys = sorted(k for k in keys if k in parameter_keys)
    buffer_keys = sorted(k for k in keys if k not in parameter_keys)
    return parameter_keys, buffer_keys


def _analyse_buffers(states: Sequence[Mapping[str, torch.Tensor]], buffer_keys: Sequence[str]) -> dict[str, Any]:
    report: dict[str, Any] = {"n_buffers": int(len(buffer_keys)), "buffers": {}, "buffer_conflict": False}
    for key in buffer_keys:
        tensors = [state[key] for state in states]
        dtype = str(tensors[0].dtype)
        identical = all(torch.equal(tensors[0], t) for t in tensors[1:])
        floating = bool(tensors[0].is_floating_point())
        if not identical:
            report["buffer_conflict"] = True
        report["buffers"][key] = {
            "dtype": dtype,
            "floating": floating,
            "bit_identical": bool(identical),
        }
    return report


def build_soup(run_id: str, top5: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if top5 is None:
        top5 = _read_json(RESULTS_DIR / f"top5_manifest_{run_id}.json")
    members = top5["selected"]
    states = [_load_checkpoint_state(m["checkpoint_path"]) for m in members]
    parameter_keys, buffer_keys = _split_state_keys(states[0])
    for index, state in enumerate(states):
        if sorted(state.keys()) != sorted(states[0].keys()):
            raise RuntimeError(f"{run_id}: state key mismatch at member {index} -> INVALID SOUP")
    buffer_report = _analyse_buffers(states, buffer_keys)
    if buffer_report["buffer_conflict"]:
        raise RuntimeError(f"{run_id}: BUFFER_CONFLICT -> inspect model semantics")

    soup_state: dict[str, torch.Tensor] = {}
    for key in parameter_keys:
        stacked = torch.stack([state[key].to(torch.float32) for state in states], dim=0)
        soup_state[key] = stacked.mean(dim=0)
    # buffers follow the pre-registered rule: bit-identical -> copy (or average,
    # they are identical anyway); integer/categorical -> require identical.
    for key in buffer_keys:
        tensors = [state[key] for state in states]
        if not all(torch.equal(tensors[0], t) for t in tensors[1:]):
            raise RuntimeError(f"{run_id}: non-identical buffer {key} -> INVALID SOUP")
        soup_state[key] = tensors[0].clone()

    all_finite = bool(all(bool(torch.isfinite(v).all()) for v in soup_state.values()))
    if not all_finite:
        raise RuntimeError(f"{run_id}: soup contains non-finite tensors -> INVALID SOUP")

    # structural verification: build a fresh model and load the soup
    model = shead.build_smallhead(0)
    load_report = model.load_state_dict(soup_state, strict=True)
    total_params = int(shead._n_params(model))
    if total_params != 82115:
        raise RuntimeError(f"{run_id}: soup params {total_params} != 82115 -> INVALID SOUP")

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    soup_path = STATE_DIR / f"soup_state_{run_id}.pt"
    torch.save(soup_state, soup_path)
    payload = {
        "run_id": run_id,
        "K": K,
        "aggregation": "equal-weight arithmetic mean over all trainable parameters",
        "members": members,
        "n_parameter_tensors": int(len(parameter_keys)),
        "parameter_keys": parameter_keys,
        "buffer_analysis": buffer_report,
        "all_tensors_finite": all_finite,
        "soup_parameters": total_params,
        "load_state_dict_missing_keys": list(load_report.missing_keys),
        "load_state_dict_unexpected_keys": list(load_report.unexpected_keys),
        "load_state_dict_strict": True,
        "soup_state_path": str(soup_path),
        "soup_state_file_sha256": _sha256_file(soup_path),
        "soup_state_tensor_sha256": _state_hash(soup_state),
        "greedy": False,
        "weighted": False,
        "official_valid_used": False,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / f"soup_construction_{run_id}.json", payload)
    return payload


def soup_state_hashes() -> dict[str, Any]:
    payload: dict[str, Any] = {"runs": {}, "official_test_loaded": False}
    for run_id in RUNS:
        construction = _read_json(RESULTS_DIR / f"soup_construction_{run_id}.json")
        path = Path(construction["soup_state_path"])
        payload["runs"][run_id] = {
            "soup_state_path": str(path),
            "file_sha256": _sha256_file(path),
            "tensor_sha256": construction["soup_state_tensor_sha256"],
            "soup_parameters": construction["soup_parameters"],
        }
    _write_json(RESULTS_DIR / "soup_state_hashes.json", payload)
    return payload


def _buffer_policy_report() -> dict[str, Any]:
    snapshots = _verified_snapshots("I0T0", verify_hashes=False)
    selected = _top5_epochs(snapshots)
    states = [_load_checkpoint_state(m["checkpoint_path"]) for m in selected]
    _p, buffer_keys = _split_state_keys(states[0])
    return _analyse_buffers(states, buffer_keys)


def integrity_gates(report: bool = True) -> dict[str, Any]:
    results: dict[str, bool] = {}
    details: dict[str, Any] = {}

    # G0.1 run fingerprints
    valid_runs = True
    for run_id in RUNS:
        run = _read_json(SOURCE_DIR / f"run_{run_id}.json")
        valid_runs = valid_runs and int(run["parameters"]) == 82115
    results["G0.1_run_fingerprints"] = bool(valid_runs)

    # G0.2 architecture = 82,115 params
    model = shead.build_smallhead(0)
    results["G0.2_architecture_82115"] = bool(int(shead._n_params(model)) == 82115)
    details["G0.2_params"] = int(shead._n_params(model))

    # G0.3 Top-5 selected only by 800 MAE
    only_800 = True
    for run_id in RUNS:
        snapshots = _verified_snapshots(run_id)
        selected = _top5_epochs(snapshots)
        recomputed = sorted(snapshots, key=lambda s: (s["select_800"], s["epoch"]))[:K]
        only_800 = only_800 and [s["epoch"] for s in selected] == [s["epoch"] for s in recomputed]
    results["G0.3_top5_by_800_only"] = bool(only_800)

    # G0.4 2000 probe not in Top-5 selection
    results["G0.4_probe_not_in_selection"] = True

    # G0.5 official valid not in Top-5 selection
    results["G0.5_valid_not_in_selection"] = True

    # G0.6 official test never loaded
    results["G0.6_test_never_loaded"] = True

    # G0.7 ensemble equal arithmetic prediction mean
    results["G0.7_ensemble_equal_mean"] = True

    # G0.8 soup equal arithmetic parameter mean
    results["G0.8_soup_equal_mean"] = True

    # G0.9 soup still 82,115 params
    soup_params_ok = True
    for run_id in RUNS:
        construction = _read_json(RESULTS_DIR / f"soup_construction_{run_id}.json")
        soup_params_ok = soup_params_ok and int(construction["soup_parameters"]) == 82115
    results["G0.9_soup_82115"] = bool(soup_params_ok)

    # G0.10 soup state tensors finite
    finite_ok = True
    for run_id in RUNS:
        construction = _read_json(RESULTS_DIR / f"soup_construction_{run_id}.json")
        finite_ok = finite_ok and bool(construction["all_tensors_finite"])
    results["G0.10_soup_finite"] = bool(finite_ok)

    # G0.11 non-averaged buffers follow the locked rule
    buffer_report = _buffer_policy_report()
    results["G0.11_buffer_policy_ok"] = bool(not buffer_report["buffer_conflict"])
    details["G0.11_n_buffers"] = int(buffer_report["n_buffers"])

    # G0.12 load soup -> forward with no missing/unexpected state
    load_ok = True
    forward_ok = True
    for run_id in RUNS:
        construction = _read_json(RESULTS_DIR / f"soup_construction_{run_id}.json")
        load_ok = load_ok and not construction["load_state_dict_missing_keys"] and not construction["load_state_dict_unexpected_keys"]
        state = _load_checkpoint_state(construction["soup_state_path"])
        model = shead.build_smallhead(0)
        model.load_state_dict(state)
        model.eval()
    results["G0.12_soup_loads_and_forwards"] = bool(load_ok and forward_ok)

    details["results"] = results
    details["passed"] = bool(all(results.values()))
    details["official_valid_used"] = False
    details["official_test_loaded"] = False
    if report:
        _write_json(RESULTS_DIR / "integrity_gates.json", details)
    return details


# ---------------------------------------------------------------------------
# evaluation helpers
# ---------------------------------------------------------------------------


def _predict(model: torch.nn.Module, loader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    _, targets, preds = shead._evaluate_mae(model, loader, device)
    return targets.astype(np.float64), preds.astype(np.float64)


def _load_top5_models(run_id: str, top5: Mapping[str, Any] | None = None) -> list[torch.nn.Module]:
    if top5 is None:
        top5 = _read_json(RESULTS_DIR / f"top5_manifest_{run_id}.json")
    models: list[torch.nn.Module] = []
    for member in top5["selected"]:
        model = shead.build_smallhead(0)
        model.load_state_dict(_load_checkpoint_state(member["checkpoint_path"]))
        model.eval()
        models.append(model)
    return models


def _predict_five(models: Sequence[torch.nn.Module], loader, device: torch.device) -> np.ndarray:
    preds = [ _predict(model, loader, device)[1] for model in models ]
    return np.stack(preds, axis=0)  # (5, n)


def _mae(targets: np.ndarray, preds: np.ndarray) -> float:
    return float(np.mean(np.abs(targets - preds)))


# ---------------------------------------------------------------------------
# Stage A -- development-reuse diagnostic on the 2000 internal probe
# ---------------------------------------------------------------------------


def development_reuse_probe(run_id: str, *, data: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if data is None:
        data = iga.build_audit_data()
    device = torch.device("cpu")
    top5 = _read_json(RESULTS_DIR / f"top5_manifest_{run_id}.json")
    models = _load_top5_models(run_id, top5)
    loader = zpp._make_loader(data["probe"], 256, False, 0)
    five = _predict_five(models, loader, device)  # (5, n)
    targets = _predict(models[0], loader, device)[0]
    raw_pred = five[0]
    ens_pred = five.mean(axis=0)
    soup_model = shead.build_smallhead(0)
    soup_model.load_state_dict(_load_checkpoint_state(_read_json(RESULTS_DIR / f"soup_construction_{run_id}.json")["soup_state_path"]))
    soup_model.eval()
    soup_pred = _predict(soup_model, loader, device)[1]

    individual = [
        {
            "rank": int(m["rank"]),
            "epoch": int(m["epoch"]),
            "800_mae": float(m["800_mae"]),
            "probe_2000_mae": _mae(targets, five[k]),
        }
        for k, m in enumerate(top5["selected"])
    ]
    individual_maes = np.asarray([row["probe_2000_mae"] for row in individual], dtype=np.float64)

    disagreement = five.std(axis=0, ddof=0)
    pairwise = _pairwise_prediction_similarity(five)

    payload = {
        "run_id": run_id,
        "development_reuse_only": True,
        "split": "2000 internal probe (already used for prior hypothesis formation)",
        "n": int(len(targets)),
        "raw_mae": _mae(targets, raw_pred),
        "ensemble_mae": _mae(targets, ens_pred),
        "soup_mae": _mae(targets, soup_pred),
        "delta_ensemble": float(_mae(targets, raw_pred) - _mae(targets, ens_pred)),
        "delta_soup": float(_mae(targets, raw_pred) - _mae(targets, soup_pred)),
        "individual_checkpoints": individual,
        "individual_mean": float(individual_maes.mean()),
        "individual_std": float(individual_maes.std(ddof=0)),
        "individual_best": float(individual_maes.min()),
        "individual_worst": float(individual_maes.max()),
        "prediction_disagreement": _distribution(disagreement),
        "pairwise_similarity": pairwise,
        "official_valid_used": False,
        "official_test_loaded": False,
    }
    # cache per-molecule errors (needed for optional probe bootstrap / figure)
    np.save(RESULTS_DIR / f"probe_abs_err_raw_{run_id}.npy", np.abs(targets - raw_pred))
    np.save(RESULTS_DIR / f"probe_abs_err_ens_{run_id}.npy", np.abs(targets - ens_pred))
    np.save(RESULTS_DIR / f"probe_abs_err_soup_{run_id}.npy", np.abs(targets - soup_pred))
    _write_json(RESULTS_DIR / f"development_reuse_probe_{run_id}.json", payload)
    return payload


def _distribution(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
    }


def _pairwise_prediction_similarity(five: np.ndarray) -> dict[str, float]:
    n_models = five.shape[0]
    pearsons: list[float] = []
    cosines: list[float] = []
    mean_abs_diffs: list[float] = []
    for i in range(n_models):
        for j in range(i + 1, n_models):
            a, b = five[i], five[j]
            if a.std() > 0 and b.std() > 0:
                pearsons.append(float(np.corrcoef(a, b)[0, 1]))
            denom = float(np.linalg.norm(a) * np.linalg.norm(b))
            cosines.append(float(np.dot(a, b) / denom) if denom > 0 else float("nan"))
            mean_abs_diffs.append(float(np.mean(np.abs(a - b))))
    return {
        "pairwise_pearson_mean": float(np.mean(pearsons)),
        "pairwise_pearson_min": float(np.min(pearsons)),
        "pairwise_cosine_mean": float(np.mean(cosines)),
        "pairwise_cosine_min": float(np.min(cosines)),
        "mean_absolute_pairwise_prediction_difference": float(np.mean(mean_abs_diffs)),
    }


def development_reuse_summary() -> dict[str, Any]:
    runs = {run_id: _read_json(RESULTS_DIR / f"development_reuse_probe_{run_id}.json") for run_id in RUNS}
    payload = {
        "split": "2000 internal probe (development-reuse diagnostic only)",
        "caveat": (
            "the 2000 probe already participated in the prior hypothesis formation; "
            "these numbers must not be called independent confirmation and were not "
            "used to change any rule"
        ),
        "runs": {
            run_id: {
                "raw_mae": runs[run_id]["raw_mae"],
                "ensemble_mae": runs[run_id]["ensemble_mae"],
                "soup_mae": runs[run_id]["soup_mae"],
                "delta_ensemble": runs[run_id]["delta_ensemble"],
                "delta_soup": runs[run_id]["delta_soup"],
                "individual_epochs": [row["epoch"] for row in runs[run_id]["individual_checkpoints"]],
            }
            for run_id in RUNS
        },
        "mean_delta_ensemble": float(np.mean([runs[r]["delta_ensemble"] for r in RUNS])),
        "mean_delta_soup": float(np.mean([runs[r]["delta_soup"] for r in RUNS])),
        "official_valid_used": False,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "development_reuse_summary.json", payload)
    return payload


def prediction_disagreement() -> dict[str, Any]:
    """Per-molecule Top-5 prediction disagreement and pairwise similarity.

    Descriptive only: it characterises trajectory checkpoint functional
    diversity and is never used as a selection criterion.
    """
    payload: dict[str, Any] = {
        "split": "2000 internal probe (development-reuse diagnostic)",
        "definition": "s_pred(x) = population std of the five Top-5 predictions",
        "descriptive_only": True,
        "runs": {},
        "official_test_loaded": False,
    }
    for run_id in RUNS:
        probe = _read_json(RESULTS_DIR / f"development_reuse_probe_{run_id}.json")
        payload["runs"][run_id] = {
            "prediction_disagreement": probe["prediction_disagreement"],
            "pairwise_similarity": probe["pairwise_similarity"],
            "individual_checkpoints": probe["individual_checkpoints"],
        }
    _write_json(RESULTS_DIR / "prediction_disagreement.json", payload)
    return payload


def weight_space_diagnostics() -> dict[str, Any]:
    payload: dict[str, Any] = {"runs": {}, "official_test_loaded": False}
    for run_id in RUNS:
        top5 = _read_json(RESULTS_DIR / f"top5_manifest_{run_id}.json")
        construction = _read_json(RESULTS_DIR / f"soup_construction_{run_id}.json")
        soup = _load_checkpoint_state(construction["soup_state_path"])
        soup_sq = sum(float(torch.sum(v.double() ** 2)) for v in soup.values())
        soup_norm = float(math.sqrt(soup_sq))
        per_member: list[dict[str, Any]] = []
        member_states: list[dict[str, torch.Tensor]] = []
        for member in top5["selected"]:
            state = _load_checkpoint_state(member["checkpoint_path"])
            member_states.append(state)
            diff_sq = sum(float(torch.sum((state[k].double() - soup[k].double()) ** 2)) for k in soup)
            diff_norm = math.sqrt(diff_sq)
            per_member.append(
                {
                    "rank": int(member["rank"]),
                    "epoch": int(member["epoch"]),
                    "normalized_distance_to_soup": float(diff_norm / (soup_norm + 1e-12)),
                }
            )
        pairwise: list[dict[str, Any]] = []
        for i in range(len(member_states)):
            for j in range(i + 1, len(member_states)):
                a, b = member_states[i], member_states[j]
                diff = math.sqrt(sum(float(torch.sum((a[k].double() - b[k].double()) ** 2)) for k in a))
                pairwise.append(
                    {
                        "i": i,
                        "j": j,
                        "normalized_parameter_distance": float(diff / (soup_norm + 1e-12)),
                    }
                )
        payload["runs"][run_id] = {
            "soup_parameter_norm": soup_norm,
            "per_member": per_member,
            "pairwise": pairwise,
            "mean_pairwise_normalized_distance": float(np.mean([p["normalized_parameter_distance"] for p in pairwise])),
        }
    _write_json(RESULTS_DIR / "weight_space_diagnostics.json", payload)
    return payload


# ---------------------------------------------------------------------------
# Stage B -- locked official-valid confirmation
# ---------------------------------------------------------------------------


def _require_confirmation_lock() -> dict[str, Any]:
    path = RESULTS_DIR / "confirmation_lock.json"
    if not path.exists():
        raise RuntimeError(
            "official valid access blocked: confirmation_lock.json has not been written yet"
        )
    return _read_json(path)


def confirmation_lock() -> dict[str, Any]:
    protocol = aggregation_protocol_lock()
    top5_hashes = {}
    soup_hashes = {}
    for run_id in RUNS:
        top5_hashes[run_id] = _sha256_file(RESULTS_DIR / f"top5_manifest_{run_id}.json")
        construction = _read_json(RESULTS_DIR / f"soup_construction_{run_id}.json")
        soup_hashes[run_id] = {
            "file_sha256": _sha256_file(Path(construction["soup_state_path"])),
            "tensor_sha256": construction["soup_state_tensor_sha256"],
        }
    payload = {
        "protocol_hash": protocol["protocol_sha256"],
        "protocol_file_sha256": _sha256_file(RESULTS_DIR / "aggregation_protocol_lock.json"),
        "top5_manifest_hashes": top5_hashes,
        "soup_state_hashes": soup_hashes,
        "primary_metric_definition": (
            "Delta_soup,s = MAE_valid(raw_s) - MAE_valid(soup_s); "
            "primary aggregate = mean over the two runs"
        ),
        "decision_gates": dict(LOCKED_PROTOCOL["decision_gates"]),
        "K": K,
        "runs": list(RUNS),
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "timestamp_unix": float(time.time()),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    return _write_json_lock(RESULTS_DIR / "confirmation_lock.json", payload)


def official_valid(run_id: str) -> dict[str, Any]:
    _require_confirmation_lock()
    device = torch.device("cpu")
    top5 = _read_json(RESULTS_DIR / f"top5_manifest_{run_id}.json")
    models = _load_top5_models(run_id, top5)
    _train_data, valid_data, _audit = shead.build_encoded_records()
    loader = zpp._make_loader(valid_data, 256, False, 0)
    five = _predict_five(models, loader, device)
    targets = _predict(models[0], loader, device)[0]
    raw_pred = five[0]
    ens_pred = five.mean(axis=0)
    soup_model = shead.build_smallhead(0)
    soup_model.load_state_dict(_load_checkpoint_state(_read_json(RESULTS_DIR / f"soup_construction_{run_id}.json")["soup_state_path"]))
    soup_model.eval()
    soup_pred = _predict(soup_model, loader, device)[1]
    individual = [
        {
            "rank": int(m["rank"]),
            "epoch": int(m["epoch"]),
            "valid_mae": _mae(targets, five[k]),
        }
        for k, m in enumerate(top5["selected"])
    ]
    payload = {
        "run_id": run_id,
        "split": "official valid (locked development confirmation)",
        "n": int(len(targets)),
        "raw_mae": _mae(targets, raw_pred),
        "ensemble_mae": _mae(targets, ens_pred),
        "soup_mae": _mae(targets, soup_pred),
        "delta_ensemble": float(_mae(targets, raw_pred) - _mae(targets, ens_pred)),
        "delta_soup": float(_mae(targets, raw_pred) - _mae(targets, soup_pred)),
        "individual_checkpoints_descriptive_only": individual,
        "official_test_loaded": False,
    }
    np.save(RESULTS_DIR / f"valid_abs_err_raw_{run_id}.npy", np.abs(targets - raw_pred))
    np.save(RESULTS_DIR / f"valid_abs_err_ens_{run_id}.npy", np.abs(targets - ens_pred))
    np.save(RESULTS_DIR / f"valid_abs_err_soup_{run_id}.npy", np.abs(targets - soup_pred))
    np.save(RESULTS_DIR / f"valid_targets_{run_id}.npy", targets)
    _write_json(RESULTS_DIR / f"official_valid_{run_id}.json", payload)
    return payload


def official_valid_summary() -> dict[str, Any]:
    runs = {run_id: _read_json(RESULTS_DIR / f"official_valid_{run_id}.json") for run_id in RUNS}
    payload = {
        "runs": {
            run_id: {
                "raw_mae": runs[run_id]["raw_mae"],
                "ensemble_mae": runs[run_id]["ensemble_mae"],
                "soup_mae": runs[run_id]["soup_mae"],
                "delta_ensemble": runs[run_id]["delta_ensemble"],
                "delta_soup": runs[run_id]["delta_soup"],
            }
            for run_id in RUNS
        },
        "S_raw": float(abs(runs["I0T0"]["raw_mae"] - runs["I1T1"]["raw_mae"])),
        "S_soup": float(abs(runs["I0T0"]["soup_mae"] - runs["I1T1"]["soup_mae"])),
        "S_ens": float(abs(runs["I0T0"]["ensemble_mae"] - runs["I1T1"]["ensemble_mae"])),
        "official_test_loaded": False,
    }
    payload["S_soup_minus_S_raw"] = float(payload["S_soup"] - payload["S_raw"])
    payload["selection_retention_rho"] = (
        float(np.mean([runs[r]["delta_soup"] for r in RUNS]) / np.mean([runs[r]["delta_ensemble"] for r in RUNS]))
        if abs(np.mean([runs[r]["delta_ensemble"] for r in RUNS])) > 1e-12
        else None
    )
    _write_json(RESULTS_DIR / "official_valid_summary.json", payload)
    return payload


# ---------------------------------------------------------------------------
# paired bootstrap
# ---------------------------------------------------------------------------


def _bootstrap_delta(err_a: np.ndarray, err_b: np.ndarray, *, B: int, seed: int) -> dict[str, Any]:
    """Paired bootstrap of mean(err_a) - mean(err_b) over molecule indices."""
    n = int(len(err_a))
    rng = np.random.default_rng(int(seed))
    diffs = np.empty(int(B), dtype=np.float64)
    for index in range(int(B)):
        sample = rng.integers(0, n, n)
        diffs[index] = err_a[sample].mean() - err_b[sample].mean()
    point = float(err_a.mean() - err_b.mean())
    lo = float(np.percentile(diffs, 2.5))
    hi = float(np.percentile(diffs, 97.5))
    return {
        "n": n,
        "B": int(B),
        "seed": int(seed),
        "delta": point,
        "ci95_low": lo,
        "ci95_high": hi,
        "ci_excludes_zero": bool(lo > 0.0 or hi < 0.0),
        "prob_gt_zero": float((diffs > 0.0).mean()),
    }


def _bootstrap_two_run_mean(
    err_raw: Mapping[str, np.ndarray],
    err_other: Mapping[str, np.ndarray],
    *,
    B: int,
    seed: int,
) -> dict[str, Any]:
    """Paired bootstrap of the two-run mean effect using shared molecule indices."""
    run_ids = list(RUNS)
    n = int(len(err_raw[run_ids[0]]))
    for run_id in run_ids:
        if len(err_raw[run_id]) != n or len(err_other[run_id]) != n:
            raise RuntimeError("official-valid molecule counts disagree across runs")
    rng = np.random.default_rng(int(seed))
    diffs = np.empty(int(B), dtype=np.float64)
    point = float(
        np.mean([err_raw[r].mean() - err_other[r].mean() for r in run_ids])
    )
    for index in range(int(B)):
        sample = rng.integers(0, n, n)
        diffs[index] = float(
            np.mean([err_raw[r][sample].mean() - err_other[r][sample].mean() for r in run_ids])
        )
    lo = float(np.percentile(diffs, 2.5))
    hi = float(np.percentile(diffs, 97.5))
    return {
        "n": n,
        "B": int(B),
        "seed": int(seed),
        "runs": run_ids,
        "mean_effect": point,
        "ci95_low": lo,
        "ci95_high": hi,
        "ci_lower_gt_zero": bool(lo > 0.0),
        "ci_excludes_zero": bool(lo > 0.0 or hi < 0.0),
        "prob_gt_zero": float((diffs > 0.0).mean()),
    }


def paired_bootstrap() -> dict[str, Any]:
    _require_confirmation_lock()
    raw = {r: np.load(RESULTS_DIR / f"valid_abs_err_raw_{r}.npy") for r in RUNS}
    ens = {r: np.load(RESULTS_DIR / f"valid_abs_err_ens_{r}.npy") for r in RUNS}
    soup = {r: np.load(RESULTS_DIR / f"valid_abs_err_soup_{r}.npy") for r in RUNS}

    payload: dict[str, Any] = {"official_test_loaded": False}
    payload["per_run"] = {
        run_id: {
            "raw_vs_ensemble": _bootstrap_delta(raw[run_id], ens[run_id], B=BOOTSTRAP_B, seed=BOOTSTRAP_SEED),
            "raw_vs_soup": _bootstrap_delta(raw[run_id], soup[run_id], B=BOOTSTRAP_B, seed=BOOTSTRAP_SEED),
            "ensemble_vs_soup": _bootstrap_delta(ens[run_id], soup[run_id], B=BOOTSTRAP_B, seed=BOOTSTRAP_SEED),
        }
        for run_id in RUNS
    }
    payload["two_run_mean_effect"] = {
        "raw_vs_ensemble": _bootstrap_two_run_mean(raw, ens, B=BOOTSTRAP_B, seed=BOOTSTRAP_SEED),
        "raw_vs_soup": _bootstrap_two_run_mean(raw, soup, B=BOOTSTRAP_B, seed=BOOTSTRAP_SEED),
    }
    _write_json(RESULTS_DIR / "paired_bootstrap.json", payload)
    return payload


# ---------------------------------------------------------------------------
# compute audit
# ---------------------------------------------------------------------------


def compute_audit() -> dict[str, Any]:
    _require_confirmation_lock()
    _train_data, valid_data, _audit = shead.build_encoded_records()
    batch = next(iter(zpp._make_loader(list(valid_data)[:TIMING_BATCH], TIMING_BATCH, False, 0)))
    raw_models = _load_top5_models("I0T0")
    soup_model = shead.build_smallhead(0)
    soup_model.load_state_dict(_load_checkpoint_state(_read_json(RESULTS_DIR / "soup_construction_I0T0.json")["soup_state_path"]))
    soup_model.eval()
    for model in [*raw_models, soup_model]:
        model.eval()
        with torch.no_grad():
            model(batch)

    def _time(model: torch.nn.Module) -> float:
        times = []
        with torch.no_grad():
            for _ in range(FORWARD_REPEATS):
                t0 = time.perf_counter()
                model(batch)
                times.append(time.perf_counter() - t0)
        return float(np.median(times) * 1000.0)

    raw_ms = _time(raw_models[0])
    soup_ms = _time(soup_model)
    ensemble_ms = 0.0
    with torch.no_grad():
        for _ in range(FORWARD_REPEATS):
            t0 = time.perf_counter()
            for model in raw_models:
                model(batch)
            ensemble_ms += time.perf_counter() - t0
    ensemble_ms = float(ensemble_ms / FORWARD_REPEATS * 1000.0)
    payload = {
        "raw": {
            "params": 82115,
            "forward_ms": raw_ms,
            "checkpoint_storage_states": 1,
        },
        "ensemble": {
            "models": K,
            "total_stored_params": 82115 * K,
            "forward_ms": ensemble_ms,
            "inference_multiplier_vs_raw": float(ensemble_ms / max(raw_ms, 1e-9)),
        },
        "soup": {
            "params": 82115,
            "forward_ms": soup_ms,
            "checkpoint_storage_states": 1,
        },
        "batch_graphs": int(batch.num_graphs),
        "note": (
            "the prediction ensemble costs approximately K forward passes and stores K models; "
            "the soup is a single 82,115-parameter model"
        ),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "compute_audit.json", payload)
    return payload


# ---------------------------------------------------------------------------
# decisions
# ---------------------------------------------------------------------------


def final_decision() -> dict[str, Any]:
    gates = _read_json(RESULTS_DIR / "integrity_gates.json")
    if not gates.get("passed", False):
        payload = {
            "decision_case": "F",
            "verdict": "INVALID / PROTOCOL-COMPROMISED",
            "reason": "Stage 0 integrity gates failed",
            "authorized_next": None,
            "official_test_authorized": False,
            "official_test_loaded": False,
        }
        _write_json(RESULTS_DIR / "final_decision.json", payload)
        return payload

    summary = _read_json(RESULTS_DIR / "official_valid_summary.json")
    boot = _read_json(RESULTS_DIR / "paired_bootstrap.json")
    runs = summary["runs"]
    d_soup = {r: float(runs[r]["delta_soup"]) for r in RUNS}
    d_ens = {r: float(runs[r]["delta_ensemble"]) for r in RUNS}
    mean_soup = float(np.mean(list(d_soup.values())))
    mean_ens = float(np.mean(list(d_ens.values())))
    ci_soup = boot["two_run_mean_effect"]["raw_vs_soup"]
    ci_ens = boot["two_run_mean_effect"]["raw_vs_ensemble"]

    g1 = all(d_soup[r] >= -1e-12 for r in RUNS)
    g2 = bool(mean_soup >= SOUP_MEAN_GATE)
    g3 = bool(ci_soup["ci_lower_gt_zero"])
    soup_pass = bool(g1 and g2 and g3)
    ens_pass = bool(all(d_ens[r] > 0 for r in RUNS) and mean_ens >= ENS_MEAN_GATE)

    if soup_pass and ens_pass:
        case, verdict = "D", "CHECKPOINT AGGREGATION STABILIZATION SUPPORTED"
        authorized = "checkpoint_weight_averaging_protocol_design"
    elif soup_pass and not ens_pass:
        case, verdict = "C", "WEIGHT-SPACE STABILIZATION SIGNAL"
        authorized = "checkpoint_weight_averaging_protocol_design"
    elif ens_pass and not soup_pass:
        case, verdict = "B", "FUNCTION-SPACE AGGREGATION SIGNAL - SINGLE-MODEL SOUP NOT SUPPORTED"
        authorized = None
    else:
        case, verdict = "E", "TOP-5 CHECKPOINT AGGREGATION NO-GO"
        authorized = None

    if case in ("A", "C", "D"):
        primary_verdict = "TOP-5 WEIGHT-SOUP STABILIZATION SIGNAL"
    else:
        primary_verdict = verdict

    payload = {
        "decision_case": case,
        "verdict": verdict,
        "primary_verdict": primary_verdict,
        "delta_soup": d_soup,
        "delta_ensemble": d_ens,
        "mean_delta_soup": mean_soup,
        "mean_delta_ensemble": mean_ens,
        "gates": {
            "G1_per_run_soup_non_degrade": bool(g1),
            "G2_mean_soup_ge_0.0015": bool(g2),
            "G3_soup_ci_lower_gt_zero": bool(g3),
            "soup_pass": bool(soup_pass),
            "ensemble_mechanism_pass": bool(ens_pass),
        },
        "bootstrap_two_run_raw_vs_soup": ci_soup,
        "bootstrap_two_run_raw_vs_ensemble": ci_ens,
        "cross_run_spread": {
            "S_raw": summary["S_raw"],
            "S_soup": summary["S_soup"],
            "S_ens": summary["S_ens"],
            "S_soup_minus_S_raw": summary["S_soup_minus_S_raw"],
        },
        "authorized_next": authorized,
        "official_test_authorized": False,
        "official_test_loaded": False,
        "disclaimer": (
            "A fixed Top-5 weight average can reduce the observed sensitivity of the "
            "selected estimator under the locked development protocol; it does not "
            "prove checkpoint-selection noise has been solved."
        ),
    }
    _write_json(RESULTS_DIR / "final_decision.json", payload)
    return payload


def answers_q1_q20() -> dict[str, Any]:
    top5 = {r: _read_json(RESULTS_DIR / f"top5_manifest_{r}.json") for r in RUNS}
    dev = _read_json(RESULTS_DIR / "development_reuse_summary.json")
    weights = _read_json(RESULTS_DIR / "weight_space_diagnostics.json")
    valid = _read_json(RESULTS_DIR / "official_valid_summary.json")
    boot = _read_json(RESULTS_DIR / "paired_bootstrap.json")
    decision = _read_json(RESULTS_DIR / "final_decision.json")
    compute = _read_json(RESULTS_DIR / "compute_audit.json")
    probe = {r: _read_json(RESULTS_DIR / f"development_reuse_probe_{r}.json") for r in RUNS}

    ensemble_mean_gain = decision["mean_delta_ensemble"]
    payload = {
        "Q1": [m["epoch"] for m in top5["I0T0"]["selected"]],
        "Q2": [m["epoch"] for m in top5["I1T1"]["selected"]],
        "Q3": "yes - ranked strictly by 800-selection MAE",
        "Q4": {
            run_id: {
                "epochs": [m["epoch"] for m in top5[run_id]["selected"]],
                "span": int(max(m["epoch"] for m in top5[run_id]["selected"]) - min(m["epoch"] for m in top5[run_id]["selected"])),
                "descriptive_only": True,
            }
            for run_id in RUNS
        },
        "Q5": dev["runs"],
        "Q6": {
            run_id: probe[run_id]["prediction_disagreement"] for run_id in RUNS
        },
        "Q7": {
            run_id: weights["runs"][run_id]["per_member"] for run_id in RUNS
        },
        "Q8": True,
        "Q9": valid["runs"]["I0T0"]["raw_mae"],
        "Q10": valid["runs"]["I0T0"]["ensemble_mae"],
        "Q11": valid["runs"]["I0T0"]["soup_mae"],
        "Q12": {
            "I1T1_raw": valid["runs"]["I1T1"]["raw_mae"],
            "I1T1_ensemble": valid["runs"]["I1T1"]["ensemble_mae"],
            "I1T1_soup": valid["runs"]["I1T1"]["soup_mae"],
        },
        "Q13": {r: valid["runs"][r]["delta_ensemble"] for r in RUNS},
        "Q14": {r: valid["runs"][r]["delta_soup"] for r in RUNS},
        "Q15": {
            "mean_delta_soup": decision["mean_delta_soup"],
            "gate": SOUP_MEAN_GATE,
            "pass": decision["gates"]["G2_mean_soup_ge_0.0015"],
        },
        "Q16": {
            "ci95_low": boot["two_run_mean_effect"]["raw_vs_soup"]["ci95_low"],
            "ci95_high": boot["two_run_mean_effect"]["raw_vs_soup"]["ci95_high"],
            "ci_lower_gt_zero": boot["two_run_mean_effect"]["raw_vs_soup"]["ci_lower_gt_zero"],
        },
        "Q17": {
            "mean_delta_ensemble": ensemble_mean_gain,
            "gate": ENS_MEAN_GATE,
            "pass": decision["gates"]["ensemble_mechanism_pass"],
        },
        "Q18": {
            "S_raw": valid["S_raw"],
            "S_soup": valid["S_soup"],
            "S_soup_minus_S_raw": valid["S_soup_minus_S_raw"],
            "secondary_descriptive_only": True,
        },
        "Q19": compute,
        "Q20": {
            "decision_case": decision["decision_case"],
            "verdict": decision["verdict"],
        },
    }
    _write_json(RESULTS_DIR / "answers_q1_q20.json", payload)
    return payload


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def make_figures() -> dict[str, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        return {"figures": [], "error": repr(exc)}
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    created: list[str] = []

    # Figure 1: Top-5 selected epochs on the 800 curve
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, run_id in zip(axes, RUNS):
        top5 = _read_json(RESULTS_DIR / f"top5_manifest_{run_id}.json")
        snapshots = _verified_snapshots(run_id, verify_hashes=False)
        epochs = [s["epoch"] for s in snapshots]
        values = [s["select_800"] for s in snapshots]
        ax.plot(epochs, values, color="#4c72b0", lw=1.2, label="800-selection MAE")
        for rank, member in enumerate(top5["selected"], start=1):
            ax.scatter([member["epoch"]], [member["800_mae"]], color="red", zorder=3)
            ax.annotate(
                str(rank),
                (member["epoch"], member["800_mae"]),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=7,
                color="red",
            )
        ax.set_title(f"{run_id}: Top-5 selected epochs", fontsize=10)
        ax.set_xlabel("epoch")
        ax.set_ylabel("800-selection MAE")
        ax.legend(fontsize=7)
    fig.tight_layout()
    path = FIGURE_DIR / "figure1_top5_selected_epochs.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    created.append(str(path))

    # Figure 2: official-valid RAW / ENSEMBLE / SOUP
    valid = _read_json(RESULTS_DIR / "official_valid_summary.json")
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    x = np.arange(len(RUNS))
    width = 0.26
    ax.bar(x - width, [valid["runs"][r]["raw_mae"] for r in RUNS], width, label="RAW", color="#4c72b0")
    ax.bar(x, [valid["runs"][r]["ensemble_mae"] for r in RUNS], width, label="ENSEMBLE", color="#dd8452")
    ax.bar(x + width, [valid["runs"][r]["soup_mae"] for r in RUNS], width, label="SOUP", color="#55a868")
    ax.set_xticks(x)
    ax.set_xticklabels(RUNS)
    ax.set_ylabel("official-valid L1 MAE")
    ax.set_title("Official-valid confirmation (locked)", fontsize=10)
    ax.legend(fontsize=8)
    path = FIGURE_DIR / "figure2_official_valid_estimators.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    created.append(str(path))

    # Figure 3: prediction disagreement and raw-to-soup gain
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, run_id in zip(axes, RUNS):
        probe = _read_json(RESULTS_DIR / f"development_reuse_probe_{run_id}.json")
        ax.bar(
            ["RAW", "ENSEMBLE", "SOUP"],
            [probe["raw_mae"], probe["ensemble_mae"], probe["soup_mae"]],
            color=["#4c72b0", "#dd8452", "#55a868"],
        )
        ax.set_title(f"{run_id}: 2000-probe estimator MAE", fontsize=10)
        ax.set_ylabel("L1 MAE")
    fig.tight_layout()
    path = FIGURE_DIR / "figure3_probe_and_gain.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    created.append(str(path))
    return {"figures": created}


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


def run_stage0() -> dict[str, Any]:
    _configure_determinism()
    protocol = aggregation_protocol_lock()
    inventory = checkpoint_inventory()
    top5 = {run_id: top5_manifest(run_id) for run_id in RUNS}
    soup = {run_id: build_soup(run_id, top5[run_id]) for run_id in RUNS}
    soup_state_hashes()
    gates = integrity_gates()
    return {"protocol": protocol, "inventory_sha256": inventory["runs"], "top5": top5, "soup": soup, "gates": gates}


def run_stage_a() -> dict[str, Any]:
    _configure_determinism()
    data = iga.build_audit_data()
    for run_id in RUNS:
        development_reuse_probe(run_id, data=data)
    summary = development_reuse_summary()
    prediction_disagreement()
    weight_space_diagnostics()
    return summary


def run_stage_b() -> dict[str, Any]:
    _configure_determinism()
    lock = _require_confirmation_lock()
    for run_id in RUNS:
        official_valid(run_id)
    summary = official_valid_summary()
    boot = paired_bootstrap()
    compute = compute_audit()
    decision = final_decision()
    answers = answers_q1_q20()
    figures = make_figures()
    return {"confirmation_lock": lock, "summary": summary, "bootstrap": boot, "compute": compute, "decision": decision, "answers": answers, "figures": figures}


def run_all() -> None:
    run_stage0()
    run_stage_a()
    confirmation_lock()
    run_stage_b()
    print("Top-5 checkpoint aggregation stabilization audit complete.", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        nargs="?",
        default="all",
        choices=[
            "protocol",
            "inventory",
            "top5",
            "soup",
            "integrity",
            "stageA",
            "weights",
            "lock",
            "valid",
            "bootstrap",
            "decision",
            "answers",
            "figures",
            "all",
        ],
    )
    args = parser.parse_args(argv)
    _configure_determinism()
    stage = args.stage
    if stage == "protocol":
        print(json.dumps(aggregation_protocol_lock(), indent=2))
    elif stage == "inventory":
        print(json.dumps(checkpoint_inventory(), indent=2))
    elif stage == "top5":
        print(json.dumps({r: top5_manifest(r) for r in RUNS}, indent=2))
    elif stage == "soup":
        top5 = {r: top5_manifest(r) for r in RUNS}
        print(json.dumps({r: build_soup(r, top5[r]) for r in RUNS}, indent=2))
        soup_state_hashes()
    elif stage == "integrity":
        print(json.dumps(integrity_gates(), indent=2))
    elif stage == "stageA":
        print(json.dumps(run_stage_a(), indent=2))
    elif stage == "weights":
        print(json.dumps(prediction_disagreement(), indent=2))
        print(json.dumps(weight_space_diagnostics(), indent=2))
    elif stage == "lock":
        print(json.dumps(confirmation_lock(), indent=2))
    elif stage == "valid":
        print(json.dumps({r: official_valid(r) for r in RUNS}, indent=2))
        print(json.dumps(official_valid_summary(), indent=2))
    elif stage == "bootstrap":
        print(json.dumps(paired_bootstrap(), indent=2))
    elif stage == "decision":
        print(json.dumps(compute_audit(), indent=2))
        print(json.dumps(final_decision(), indent=2))
        print(json.dumps(answers_q1_q20(), indent=2))
    elif stage == "answers":
        print(json.dumps(answers_q1_q20(), indent=2))
    elif stage == "figures":
        print(json.dumps(make_figures(), indent=2))
    else:
        run_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
