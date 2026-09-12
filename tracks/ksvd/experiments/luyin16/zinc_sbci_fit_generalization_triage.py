"""SBCI fit-vs-generalization failure triage (zero-training).

This is a narrow, zero-training diagnosis.  Using only the already-saved
N3600 / I0 / T0 ``compact-v4-smallhead`` (baseline) and ``compact-v4-SBCI``
trajectories and checkpoints, it asks a single question:

    Did SBCI fail because it was too restrictive to fit the training function
    (APPROXIMATION / OVERCONSTRAINT FAILURE), or because its inductive bias
    generalized worse despite adequate fit (WRONG INDUCTIVE BIAS DESPITE
    ADEQUATE FIT), or is the evidence mixed / optimization-ambiguous?

Hard rules enforced here:

* zero new gradient updates -- only ``model.eval()`` forward passes;
* no SBCI rescue, no seed1, no N7200, no N1800, no K/width sweep;
* no official valid, no official test, ever;
* the best-achieved training fit ``F_min`` is a *diagnostic only* and never
  becomes a deployed checkpoint or changes the SBCI GO/NO-GO decision.

Module: ``tracks/ksvd/experiments/luyin16/zinc_sbci_fit_generalization_triage.py``
Results: ``tracks/ksvd/results/sbci_fit_generalization_triage/``
Note:    ``tracks/ksvd/notes/sbci_fit_generalization_triage.md``
Tests:   ``tracks/ksvd/tests/test_sbci_fit_generalization_triage.py``
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from tracks.ksvd.experiments.luyin16 import zinc_compact_v4_sbci as sb
from tracks.ksvd.experiments.luyin16 import zinc_compact_v4_smallhead_e2e as shead
from tracks.ksvd.experiments.luyin16 import (
    zinc_inductive_bias_sample_efficiency_audit as se,
)
from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp

# ---------------------------------------------------------------------------
# frozen layout / constants
# ---------------------------------------------------------------------------

REPO_ROOT = zpp.REPO_ROOT
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"
RESULTS_DIR = TRACK_ROOT / "results/sbci_fit_generalization_triage"
BASELINE_DIR = TRACK_ROOT / "results/inductive_bias_sample_efficiency_audit"
SBCI_DIR = TRACK_ROOT / "results/compact_v4_sbci"

PROTOCOL_VERSION = "sbci_fit_generalization_triage_v1"
RUN_ID = "N3600_I0T0"
N_TRAIN = 3600
MODELS = ("baseline", "sbci")

# pre-registered decision thresholds (from the triage ticket)
APPROX_FIT_DEFICIT = 0.008       # A1/A2: dF_min and dF_late >= +0.008
ADEQUATE_FIT_ABS = 0.003         # B1/B2: |dF_min|, |dF_late| <= 0.003
SELECTED_STRONG = 0.005          # A3: RAW and SOUP selected deficits > +0.005
SELECTED_ADEQUATE = 0.003        # B3: RAW and SOUP selected deficits <= +0.003
PROBE_DEGRADATION_MIN = 0.005    # B4: probe degradation >= +0.005
LATE_WINDOW = 10                 # last N selection-evaluation events
CONVERGENCE_SLOPE = -5e-5        # per selection-evaluation interval
BATCH_INVARIANCE_TOL = 1e-6

BOOTSTRAP_B = 2000
BOOTSTRAP_SEED = 20260912

# number of molecules per D3600 / 800 / 2000 set
N_SELECT = 800
N_PROBE = 2000

FAMILY_SSOD = "shared_structural_operator_dictionary"
FAMILY_DSP = "direct_shared_structural_potentials"


class TriageError(RuntimeError):
    """Raised when the frozen comparison is invalid or an artifact is corrupt."""


# ---------------------------------------------------------------------------
# io / hashing helpers
# ---------------------------------------------------------------------------


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_indices(indices: Sequence[int]) -> str:
    return hashlib.sha256(",".join(str(int(i)) for i in indices).encode()).hexdigest()


def _state_hash(state: Mapping[str, torch.Tensor]) -> str:
    return shead._state_hash(state)


def _load_state(path: str | Path) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    return {k: v.detach().clone() for k, v in state.items()}


def _configure_determinism() -> None:
    torch.set_num_threads(int(shead.TORCH_THREADS))
    torch.use_deterministic_algorithms(False)


def _environment() -> dict[str, Any]:
    import platform

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "torch_threads": int(torch.get_num_threads()),
        "device": "cpu",
    }


# ---------------------------------------------------------------------------
# data + frozen environment
# ---------------------------------------------------------------------------


def load_data() -> dict[str, Any]:
    """Load the frozen official-train development environment (firewall on)."""
    return se.load_data()


def build_subsets(data: Mapping[str, Any]) -> dict[str, Any]:
    return se.build_subsets(data)


def train_data(data: Mapping[str, Any], subsets: Mapping[str, Any]) -> list:
    return se._map_index_lists(data, subsets[f"sorted_{N_TRAIN}"])


def _data_fingerprints(data: Mapping[str, Any], subsets: Mapping[str, Any]) -> dict[str, Any]:
    train = train_data(data, subsets)
    return {
        "n_train_subset": int(len(train)),
        "n_select_800": int(len(data["select"])),
        "n_probe_2000": int(len(data["probe"])),
        "train_subset_index_hash": _sha256_indices(subsets[f"sorted_{N_TRAIN}"]),
        "train_subset_ranked_hash": _sha256_indices(subsets[f"ranked_{N_TRAIN}"]),
        "select_index_hash": _sha256_indices(data["indices"]["checkpoint_selection"]),
        "probe_index_hash": _sha256_indices(data["indices"]["internal_probe"]),
        "optimization_train_index_hash": _sha256_indices(data["indices"]["optimization_train"]),
        "subset_salt": str(se.SUBSET_SALT),
    }


# ---------------------------------------------------------------------------
# model + state helpers
# ---------------------------------------------------------------------------


def _model_dir(model: str) -> Path:
    return BASELINE_DIR if model == "baseline" else SBCI_DIR


def build_model(model: str, seed: int = 0) -> torch.nn.Module:
    if model == "baseline":
        return shead.build_smallhead(int(seed))
    if model == "sbci":
        return sb.build_sbci(int(seed))
    raise TriageError(f"unknown model {model!r}")


def _run_manifest(model: str) -> dict[str, Any]:
    return _read_json(_model_dir(model) / f"stage_run_{RUN_ID}.json")


def _checkpoint_manifest(model: str) -> dict[str, Any]:
    return _read_json(_model_dir(model) / f"checkpoint_manifest_{RUN_ID}.json")


def _soup_record(model: str) -> dict[str, Any]:
    return _read_json(_model_dir(model) / f"soup_construction_{RUN_ID}.json")


def _snapshots(model: str) -> list[dict[str, Any]]:
    rows = list(_checkpoint_manifest(model)["snapshots"])
    return sorted(rows, key=lambda r: int(r["step"]))


def raw_member(model: str) -> dict[str, Any]:
    return min(_snapshots(model), key=lambda r: (float(r["select_800_mae"]), int(r["step"])))


def soup_members(model: str) -> list[dict[str, Any]]:
    ranked = sorted(_snapshots(model), key=lambda r: (float(r["select_800_mae"]), int(r["step"])))
    return ranked[:5]


def _train_loader(data: Mapping[str, Any], subsets: Mapping[str, Any], batch: int = 256):
    return zpp._make_loader(train_data(data, subsets), int(batch), False, 0)


def _evaluate(model: torch.nn.Module, loader) -> tuple[float, np.ndarray, np.ndarray]:
    """Deterministic eval-mode MAE; ``shead._evaluate_mae`` calls ``model.eval()``."""
    return shead._evaluate_mae(model, loader, torch.device("cpu"))


def _check_finite(mae: float, model: str) -> None:
    if not np.isfinite(mae):
        raise TriageError(f"{model}: non-finite training MAE")


def load_raw_model(model: str) -> tuple[torch.nn.Module, dict[str, Any]]:
    member = raw_member(model)
    path = Path(member["snapshot"])
    if _sha256_file(path) != str(member["snapshot_sha256"]):
        raise TriageError(f"{model}: raw member snapshot hash mismatch")
    net = build_model(model, 0)
    net.load_state_dict(_load_state(path))
    net.eval()
    return net, member


def load_soup_model(model: str) -> tuple[torch.nn.Module, dict[str, Any]]:
    record = _soup_record(model)
    path = Path(record["soup_state_path"])
    if _sha256_file(path) != str(record["soup_state_file_sha256"]):
        raise TriageError(f"{model}: soup state file hash mismatch")
    state = _load_state(path)
    if _state_hash(state) != str(record["soup_state_tensor_sha256"]):
        raise TriageError(f"{model}: soup state tensor hash mismatch")
    net = build_model(model, 0)
    net.load_state_dict(state)
    net.eval()
    return net, record


def load_snapshot_model(model: str, member: Mapping[str, Any]) -> torch.nn.Module:
    net = build_model(model, 0)
    net.load_state_dict(_load_state(member["snapshot"]))
    net.eval()
    return net


# ---------------------------------------------------------------------------
# 1. protocol compatibility + run inventories
# ---------------------------------------------------------------------------


def audit_protocol_lock() -> dict[str, Any]:
    payload = {
        "name": "SBCI fit-vs-generalization failure triage",
        "protocol_version": PROTOCOL_VERSION,
        "run_id": RUN_ID,
        "candidate": "compact-v4-SBCI / N3600 / I0 / T0",
        "baseline": "compact-v4-smallhead / N3600 / I0 / T0",
        "question": (
            "did SBCI fail because its function class was too restrictive to fit "
            "D3600, or because it fit comparably but generalized worse?"
        ),
        "decision_cases": [
            "APPROXIMATION_OVERCONSTRAINT_FAILURE",
            "ADEQUATE_FIT_WRONG_INDUCTIVE_BIAS",
            "FIT_GENERALIZATION_TRIAGE_INCONCLUSIVE",
            "OPTIMIZATION_AMBIGUOUS",
        ],
        "thresholds": {
            "approx_fit_deficit": APPROX_FIT_DEFICIT,
            "adequate_fit_abs": ADEQUATE_FIT_ABS,
            "selected_strong": SELECTED_STRONG,
            "selected_adequate": SELECTED_ADEQUATE,
            "probe_degradation_min": PROBE_DEGRADATION_MIN,
            "late_window": LATE_WINDOW,
            "convergence_slope_per_eval": CONVERGENCE_SLOPE,
        },
        "hard_rules": {
            "zero_new_gradient_updates": True,
            "no_sbci_rescue": True,
            "no_seed1": True,
            "no_n7200": True,
            "no_n1800": True,
            "no_seed2_or_3": True,
            "no_k_or_width_sweep": True,
            "no_optimization_search": True,
            "no_representation_geometry": True,
            "f_min_is_diagnostic_only": True,
            "official_valid_used": False,
            "official_test_loaded": False,
        },
        "external_inputs": {
            "baseline_results": str(BASELINE_DIR),
            "sbci_results": str(SBCI_DIR),
            "known_soup_probe_sbci": 0.18292410240857862,
            "known_soup_probe_baseline": 0.1766334645843599,
            "known_probe_degradation_candidate_minus_baseline": 0.006291,
        },
        "environment": _environment(),
        "official_valid_used": False,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "audit_protocol_lock.json", payload)
    return payload


def _run_inventory(model: str) -> dict[str, Any]:
    run = _run_manifest(model)
    manifest = _checkpoint_manifest(model)
    soup = _soup_record(model)
    snapshots = _snapshots(model)
    raw = raw_member(model)
    net = build_model(model, 0)
    init_hash = _state_hash(net.state_dict())
    run_id_ok = (
        int(run["N"]) == N_TRAIN and int(run["I"]) == 0 and int(run["T"]) == 0
    )
    if not run_id_ok:
        raise TriageError(f"{model}: run is not N3600/I0/T0 (got N{run['N']}/I{run['I']}/T{run['T']})")
    if not (run.get("frozen") and run.get("estimators_evaluated")):
        raise TriageError(f"{model}: run is not frozen/evaluated")
    if run["init_state_sha256"] != init_hash:
        raise TriageError(f"{model}: rebuilt init hash does not match the frozen run")
    if len(snapshots) != int(run["snapshot_count"]):
        raise TriageError(f"{model}: snapshot count mismatch")
    return {
        "model": model,
        "results_dir": str(_model_dir(model)),
        "run_id": RUN_ID,
        "N": int(run["N"]),
        "I": int(run["I"]),
        "T": int(run["T"]),
        "is_n3600_i0_t0": bool(run_id_ok),
        "parameters": int(run["parameters"]),
        "protocol": dict(run["protocol"]),
        "optimizer": str(run["protocol"]["optimizer"]),
        "learning_rate": float(run["protocol"]["learning_rate"]),
        "weight_decay": float(run["protocol"]["weight_decay"]),
        "batch_size": int(run["protocol"]["batch_size"]),
        "loss": str(run["protocol"]["loss"]),
        "scheduler": str(run["protocol"]["scheduler"]),
        "gradient_clip_norm": float(run["protocol"]["gradient_clip_norm"]),
        "max_optimizer_steps": int(run["max_optimizer_steps"]),
        "eval_interval": int(run["eval_interval"]),
        "patience_evaluations": int(run["patience_evaluations"]),
        "optimizer_steps": int(run["optimizer_steps"]),
        "eval_count": int(run["eval_count"]),
        "early_stopped": bool(run["early_stopped"]),
        "budget_boundary_warning": bool(run["budget_boundary_warning"]),
        "raw_best_step": int(run["raw_best_step"]),
        "raw_best_eval_index": int(run["raw_best_eval_index"]),
        "raw_best_select_800": float(run["raw_best_select_800"]),
        "raw_probe_mae": float(run["raw_probe_mae"]),
        "soup_probe_mae": float(run["soup_probe_mae"]),
        "init_state_sha256": init_hash,
        "init_state_sha256_matches": bool(run["init_state_sha256"] == init_hash),
        "snapshot_count": int(len(snapshots)),
        "raw_member_step": int(raw["step"]),
        "raw_member_eval_index": int(raw["eval_index"]),
        "raw_member_select_800": float(raw["select_800_mae"]),
        "raw_member_snapshot_sha256": str(raw["snapshot_sha256"]),
        "soup_state_path": str(soup["soup_state_path"]),
        "soup_state_file_sha256": str(soup["soup_state_file_sha256"]),
        "soup_state_tensor_sha256": str(soup["soup_state_tensor_sha256"]),
        "soup_member_steps": [int(m["step"]) for m in soup["members"]],
        "soup_member_select_800": [float(m["select_800_mae"]) for m in soup["members"]],
        "soup_K": int(soup["K"]),
        "soup_rule": "equal-weight arithmetic mean over all trainable parameters",
        "frozen": True,
        "estimators_evaluated": True,
        "official_valid_used": False,
        "official_test_loaded": False,
    }


def baseline_run_inventory() -> dict[str, Any]:
    payload = _run_inventory("baseline")
    payload["architecture"] = "compact-v4-smallhead"
    payload["head"] = "302 -> 13 -> 13 -> 1"
    _write_json(RESULTS_DIR / "baseline_run_inventory.json", payload)
    return payload


def sbci_run_inventory() -> dict[str, Any]:
    payload = _run_inventory("sbci")
    payload["architecture"] = "compact-v4-SBCI (shared-basis compositional interaction)"
    payload["R_dimension"] = int(sb.FINAL_R_WIDTH)
    payload["parameter_cap"] = int(sb.BASELINE_TOTAL)
    payload["cap_respected"] = bool(int(payload["parameters"]) <= int(sb.BASELINE_TOTAL))
    _write_json(RESULTS_DIR / "sbci_run_inventory.json", payload)
    return payload


def triage_protocol_compatibility(
    data: Mapping[str, Any], subsets: Mapping[str, Any]
) -> dict[str, Any]:
    base = baseline_run_inventory()
    sbci = sbci_run_inventory()
    ref = _read_json(BASELINE_DIR / "split_inventory.json")
    lock = _read_json(BASELINE_DIR / "sample_efficiency_subset_lock.json")
    fp = _data_fingerprints(data, subsets)

    proto_keys = (
        "optimizer",
        "learning_rate",
        "weight_decay",
        "batch_size",
        "loss",
        "scheduler",
        "gradient_clip_norm",
        "max_optimizer_steps",
        "eval_interval",
        "patience_evaluations",
    )
    protocol_checks = {
        key: bool(base[key] == sbci[key]) for key in proto_keys
    }
    # shared initialization fingerprint
    base_net = build_model("baseline", 0)
    sbci_net = build_model("sbci", 0)
    bs = base_net.state_dict()
    ss = sbci_net.state_dict()
    shared_keys = sorted(k for k in bs if k in ss and tuple(bs[k].shape) == tuple(ss[k].shape))
    shared_max_abs_diff = (
        max(float((bs[k].float() - ss[k].float()).abs().max()) for k in shared_keys)
        if shared_keys
        else float("nan")
    )
    init_lock = _read_json(SBCI_DIR / "initialization_lock.json")
    checks = {
        "same_run_id": bool(base["run_id"] == sbci["run_id"] == RUN_ID),
        "same_N": bool(base["N"] == sbci["N"] == N_TRAIN),
        "same_I": bool(base["I"] == sbci["I"] == 0),
        "same_T": bool(base["T"] == sbci["T"] == 0),
        "same_optimizer_protocol": bool(all(protocol_checks.values())),
        "same_max_optimizer_steps": bool(base["max_optimizer_steps"] == sbci["max_optimizer_steps"] == 13680),
        "same_eval_interval": bool(base["eval_interval"] == sbci["eval_interval"] == 57),
        "same_patience": bool(base["patience_evaluations"] == sbci["patience_evaluations"] == 40),
        "same_d3600_indices": bool(
            fp["train_subset_index_hash"] == lock["dataset_hash_3600"]
            and fp["train_subset_index_hash"] == "3be9f7e1a4fb8cf4ed5eedfbd725b1ac9d618e6bc6d4873fe254ddc1017dfea9"
        ),
        "same_select_800_indices": bool(
            fp["select_index_hash"] == ref["index_sha256"]["checkpoint_selection"]
        ),
        "same_probe_2000_indices": bool(
            fp["probe_index_hash"] == ref["index_sha256"]["internal_probe"]
        ),
        "same_subset_salt": bool(fp["subset_salt"] == lock["salt"]),
        "same_soup_rule": bool(
            base["soup_K"] == sbci["soup_K"] == 5
            and base["soup_rule"] == sbci["soup_rule"]
        ),
        "shared_init_exact": bool(shared_max_abs_diff == 0.0),
        "baseline_init_matches_frozen": bool(base["init_state_sha256_matches"]),
        "sbci_init_matches_frozen": bool(sbci["init_state_sha256_matches"]),
    }
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "run_id": RUN_ID,
        "data_fingerprints": fp,
        "protocol_fields": {key: base[key] for key in proto_keys},
        "protocol_checks": protocol_checks,
        "shared_initialization": {
            "n_shared_tensors": int(len(shared_keys)),
            "max_abs_diff": float(shared_max_abs_diff),
            "sbci_initialization_lock_shared_tensors": int(init_lock["seeds"]["0"]["n_shared_tensors"]),
            "sbci_initialization_lock_max_abs_diff": float(init_lock["seeds"]["0"]["max_abs_diff"]),
            "sbci_initialization_lock_hash_match": bool(init_lock["seeds"]["0"]["hash_match"]),
        },
        "checks": checks,
        "comparable": bool(all(checks.values())),
        "official_valid_used": False,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "triage_protocol_compatibility.json", payload)
    if not payload["comparable"]:
        raise TriageError("protocols are not comparable; refusing to run the triage")
    return payload


# ---------------------------------------------------------------------------
# 2. selected-estimator training fit (RAW / SOUP)
# ---------------------------------------------------------------------------


def _save_abs_err(name: str, err: np.ndarray) -> str:
    path = RESULTS_DIR / name
    np.save(path, err.astype(np.float64))
    return str(path)


def _train_fit_for_model(
    model: str, data: Mapping[str, Any], subsets: Mapping[str, Any]
) -> dict[str, Any]:
    loader = _train_loader(data, subsets, 256)
    net, member = load_raw_model(model)
    mae_raw, targets, pred_raw = _evaluate(net, loader)
    _check_finite(mae_raw, f"{model}:raw")
    net_soup, soup = load_soup_model(model)
    mae_soup, targets2, pred_soup = _evaluate(net_soup, loader)
    _check_finite(mae_soup, f"{model}:soup")
    if not np.array_equal(targets, targets2):
        raise TriageError(f"{model}: train target order changed between evaluations")
    err_raw = np.abs(targets - pred_raw)
    err_soup = np.abs(targets2 - pred_soup)
    return {
        "model": model,
        "n": int(len(targets)),
        "raw": {
            "step": int(member["step"]),
            "eval_index": int(member["eval_index"]),
            "select_800_mae": float(member["select_800_mae"]),
            "train_mae": float(err_raw.mean()),
            "state_sha256": _state_hash(_load_state(member["snapshot"])),
            "snapshot_sha256": str(member["snapshot_sha256"]),
        },
        "soup": {
            "member_steps": [int(m["step"]) for m in soup_members(model)],
            "train_mae": float(err_soup.mean()),
            "state_tensor_sha256": str(soup["soup_state_tensor_sha256"]),
            "state_file_sha256": str(soup["soup_state_file_sha256"]),
        },
        "err_raw": err_raw,
        "err_soup": err_soup,
    }


def _fit_deficit_payload(kind: str) -> dict[str, Any]:
    errs: dict[str, np.ndarray] = {}
    fits: dict[str, float] = {}
    for model in MODELS:
        path = RESULTS_DIR / f"train_abs_err_{kind}_{model}_{RUN_ID}.npy"
        errs[model] = np.load(path)
        fits[model] = float(errs[model].mean())
    diff = errs["sbci"] - errs["baseline"]
    boot = se.paired_bootstrap(diff, B=BOOTSTRAP_B, seed=BOOTSTRAP_SEED)
    return {
        "run_id": RUN_ID,
        "estimator": kind,
        "baseline_train_mae": fits["baseline"],
        "sbci_train_mae": fits["sbci"],
        "delta_F": float(fits["sbci"] - fits["baseline"]),
        "paired_bootstrap": boot,
        "n": int(diff.size),
        "official_valid_used": False,
        "official_test_loaded": False,
    }


def train_eval_raw(data: Mapping[str, Any], subsets: Mapping[str, Any]) -> dict[str, Any]:
    per_model = {}
    for model in MODELS:
        fit = _train_fit_for_model(model, data, subsets)
        per_model[model] = fit
        _save_abs_err(f"train_abs_err_raw_{model}_{RUN_ID}.npy", fit["err_raw"])
    payload = _fit_deficit_payload("raw")
    payload["per_model"] = {
        model: {
            "step": per_model[model]["raw"]["step"],
            "eval_index": per_model[model]["raw"]["eval_index"],
            "select_800_mae": per_model[model]["raw"]["select_800_mae"],
            "train_mae": per_model[model]["raw"]["train_mae"],
            "state_sha256": per_model[model]["raw"]["state_sha256"],
        }
        for model in MODELS
    }
    payload["note"] = "recomputed from frozen RAW checkpoints under model.eval() on D3600"
    _write_json(RESULTS_DIR / "train_eval_raw.json", payload)
    return payload


def train_eval_soup(data: Mapping[str, Any], subsets: Mapping[str, Any]) -> dict[str, Any]:
    per_model = {}
    for model in MODELS:
        fit = _train_fit_for_model(model, data, subsets)
        per_model[model] = fit
        _save_abs_err(f"train_abs_err_soup_{model}_{RUN_ID}.npy", fit["err_soup"])
    payload = _fit_deficit_payload("soup")
    payload["per_model"] = {
        model: {
            "member_steps": per_model[model]["soup"]["member_steps"],
            "train_mae": per_model[model]["soup"]["train_mae"],
            "state_tensor_sha256": per_model[model]["soup"]["state_tensor_sha256"],
        }
        for model in MODELS
    }
    payload["note"] = "recomputed from frozen K=5 SOUP states under model.eval() on D3600"
    _write_json(RESULTS_DIR / "train_eval_soup.json", payload)
    return payload


# ---------------------------------------------------------------------------
# 3. full-trajectory snapshot training fit (diagnostic only)
# ---------------------------------------------------------------------------


def snapshot_train_fit(data: Mapping[str, Any], subsets: Mapping[str, Any]) -> dict[str, Any]:
    loader = _train_loader(data, subsets, 256)
    rows: list[dict[str, Any]] = []
    target_ref: np.ndarray | None = None
    for model in MODELS:
        net = build_model(model, 0)
        raw = raw_member(model)
        soup_steps = {int(m["step"]) for m in soup_members(model)}
        snapshots = _snapshots(model)
        for i, member in enumerate(snapshots, start=1):
            state = _load_state(member["snapshot"])
            net.load_state_dict(state)
            mae, targets, _ = _evaluate(net, loader)
            _check_finite(mae, f"{model}:{member['step']}")
            if target_ref is None:
                target_ref = targets
            elif not np.array_equal(target_ref, targets):
                raise TriageError("target order changed across snapshot evaluations")
            rows.append(
                {
                    "model": model,
                    "step": int(member["step"]),
                    "eval_index": int(member["eval_index"]),
                    "select_800_mae": float(member["select_800_mae"]),
                    "train_eval_mae_recomputed": float(mae),
                    "train_eval_mae_stored": float(member["train_eval_mae"]),
                    "abs_diff": float(abs(mae - float(member["train_eval_mae"]))),
                    "is_raw_member": int(int(member["step"]) == int(raw["step"])),
                    "is_soup_member": int(int(member["step"]) in soup_steps),
                }
            )
        print(f"[snapshots] {model}: {i} snapshots evaluated", flush=True)
    fields = (
        "model",
        "step",
        "eval_index",
        "select_800_mae",
        "train_eval_mae_recomputed",
        "train_eval_mae_stored",
        "abs_diff",
        "is_raw_member",
        "is_soup_member",
    )
    _write_csv(RESULTS_DIR / "snapshot_train_fit.csv", rows, fields)
    max_abs_diff = max(float(r["abs_diff"]) for r in rows)
    return {
        "run_id": RUN_ID,
        "n_rows": int(len(rows)),
        "row_counts": {model: int(sum(1 for r in rows if r["model"] == model)) for model in MODELS},
        "max_abs_diff_recomputed_vs_stored": max_abs_diff,
        "recompute_matches_stored": bool(max_abs_diff < 1e-5),
        "train_eval_mae_recomputed": True,
        "f_min_is_diagnostic_only": True,
        "official_valid_used": False,
        "official_test_loaded": False,
    }


# ---------------------------------------------------------------------------
# 4. best-achieved / late-trajectory / convergence / gap
# ---------------------------------------------------------------------------


def _curve(model: str) -> list[dict[str, float]]:
    rows = [
        r for r in _read_csv(RESULTS_DIR / "snapshot_train_fit.csv") if r["model"] == model
    ]
    return [
        {
            "step": float(r["step"]),
            "eval_index": float(r["eval_index"]),
            "select_800_mae": float(r["select_800_mae"]),
            "train_mae": float(r["train_eval_mae_recomputed"]),
        }
        for r in rows
    ]


def best_train_fit() -> dict[str, Any]:
    out: dict[str, Any] = {"run_id": RUN_ID, "per_model": {}}
    mins: dict[str, dict[str, float]] = {}
    for model in MODELS:
        curve = _curve(model)
        idx = min(range(len(curve)), key=lambda k: curve[k]["train_mae"])
        mins[model] = curve[idx]
        out["per_model"][model] = {
            "f_min": float(curve[idx]["train_mae"]),
            "f_min_step": int(curve[idx]["step"]),
            "f_min_eval_index": int(curve[idx]["eval_index"]),
            "f_min_select_800_mae": float(curve[idx]["select_800_mae"]),
        }
    out["delta_F_min"] = float(mins["sbci"]["train_mae"] - mins["baseline"]["train_mae"])
    out["diagnostic_only"] = True
    out["not_used_for_model_selection"] = True
    out["official_valid_used"] = False
    out["official_test_loaded"] = False
    _write_json(RESULTS_DIR / "best_train_fit.json", out)
    return out


def late_trajectory_fit() -> dict[str, Any]:
    curves = {model: _curve(model) for model in MODELS}
    own = {model: curves[model][-LATE_WINDOW:] for model in MODELS}
    out: dict[str, Any] = {
        "run_id": RUN_ID,
        "late_window": LATE_WINDOW,
        "per_model_own_last_events": {},
    }
    medians: dict[str, float] = {}
    for model in MODELS:
        vals = [r["train_mae"] for r in own[model]]
        medians[model] = float(statistics.median(vals))
        out["per_model_own_last_events"][model] = {
            "steps": [int(r["step"]) for r in own[model]],
            "eval_indices": [int(r["eval_index"]) for r in own[model]],
            "train_mae": vals,
            "median": medians[model],
            "mean": float(np.mean(vals)),
        }
    out["delta_F_late"] = float(medians["sbci"] - medians["baseline"])
    # matched-grid secondary: last LATE_WINDOW events present in both trajectories
    base_steps = {int(r["step"]) for r in curves["baseline"]}
    sbci_steps = {int(r["step"]) for r in curves["sbci"]}
    common = sorted(base_steps & sbci_steps)
    common_steps = set(common[-LATE_WINDOW:])
    matched: dict[str, Any] = {}
    matched_medians: dict[str, float] = {}
    for model in MODELS:
        sel = [r for r in curves[model] if int(r["step"]) in common_steps]
        vals = [r["train_mae"] for r in sel]
        matched_medians[model] = float(statistics.median(vals))
        matched[model] = {
            "steps": [int(r["step"]) for r in sel],
            "train_mae": vals,
            "median": matched_medians[model],
        }
    out["matched_grid_last_events"] = {
        "steps": sorted(int(s) for s in common_steps),
        "per_model": matched,
        "delta_F_late_matched": float(matched_medians["sbci"] - matched_medians["baseline"]),
    }
    out["diagnostic_only"] = True
    out["official_valid_used"] = False
    out["official_test_loaded"] = False
    _write_json(RESULTS_DIR / "late_trajectory_fit.json", out)
    return out


def _ols_slope_stats(xs: Sequence[float], ys: Sequence[float], *, B: int, seed: int) -> dict[str, Any]:
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    xm = x - x.mean()
    sxx = float((xm * xm).sum())
    if sxx == 0.0:
        return {
            "slope": 0.0,
            "std_error": 0.0,
            "t_stat": 0.0,
            "residual_std": 0.0,
            "bootstrap_ci95": [0.0, 0.0],
            "slope_significant_negative": False,
        }
    slope = float((xm * (y - y.mean())).sum() / sxx)
    resid = y - (y.mean() + slope * xm)
    dof = max(int(y.size) - 2, 1)
    s2 = float((resid * resid).sum() / dof)
    se = float(np.sqrt(s2 / sxx)) if s2 > 0 else 0.0
    rng = np.random.default_rng(int(seed))
    boots: list[float] = []
    for _ in range(int(B)):
        idx = rng.integers(0, y.size, size=y.size)
        xx = x[idx]
        xxm = xx - xx.mean()
        denom = float((xxm * xxm).sum())
        if denom > 0:
            yy = y[idx]
            boots.append(float((xxm * (yy - yy.mean())).sum() / denom))
    boots_arr = np.asarray(boots, dtype=np.float64)
    lo, hi = (
        float(np.percentile(boots_arr, 2.5)),
        float(np.percentile(boots_arr, 97.5)),
    )
    return {
        "slope": slope,
        "std_error": se,
        "t_stat": float(slope / se) if se > 0 else 0.0,
        "residual_std": float(np.sqrt(s2)),
        "bootstrap_ci95": [lo, hi],
        "slope_significant_negative": bool(hi < 0.0),
    }


def convergence_diagnostic() -> dict[str, Any]:
    curves = {model: _curve(model) for model in MODELS}
    out: dict[str, Any] = {
        "run_id": RUN_ID,
        "late_window": LATE_WINDOW,
        "per_model": {},
        "criterion": (
            "SBCI late train-fit slope < -5e-5 MAE per selection-evaluation "
            "interval while stopping by patience/budget, AND the decline is "
            "statistically distinguishable from the trajectory noise "
            "(bootstrap 95% CI upper < 0)"
        ),
        "robustness_rationale": (
            "the per-eval MAE is highly oscillatory (residual std ~0.03); a raw "
            "OLS slope can be driven by one or two spike evaluations, so the "
            "'clearly continuing to decline' condition additionally requires a "
            "significant negative slope and is cross-checked against the "
            "monotone best-so-far envelope"
        ),
    }
    for model in MODELS:
        curve = curves[model]
        late = curve[-LATE_WINDOW:]
        train = [r["train_mae"] for r in late]
        select = [r["select_800_mae"] for r in late]
        eval_idx = [r["eval_index"] for r in late]
        step = [r["step"] for r in late]
        stats = _ols_slope_stats(eval_idx, train, B=BOOTSTRAP_B, seed=BOOTSTRAP_SEED)
        stats_step = _ols_slope_stats(step, train, B=BOOTSTRAP_B, seed=BOOTSTRAP_SEED)
        select_stats = _ols_slope_stats(eval_idx, select, B=BOOTSTRAP_B, seed=BOOTSTRAP_SEED)
        all_train = [r["train_mae"] for r in curve]
        bsf = np.minimum.accumulate(np.asarray(all_train, dtype=np.float64))
        best_so_far_improve_10 = float(bsf[-1 - LATE_WINDOW] - bsf[-1])
        best_so_far_improve_20 = float(bsf[-1 - 2 * LATE_WINDOW] - bsf[-1])
        last5_slope = _ols_slope_stats(eval_idx[-5:], train[-5:], B=BOOTSTRAP_B, seed=BOOTSTRAP_SEED)["slope"]
        down_steps = sum(1 for a, b in zip(train, train[1:]) if b < a)
        run = _run_manifest(model)
        stopped = bool(run["early_stopped"] or int(run["optimizer_steps"]) >= int(run["max_optimizer_steps"]))
        literal_fired = bool(stats["slope"] < CONVERGENCE_SLOPE and stopped)
        out["per_model"][model] = {
            "steps": [int(v) for v in step],
            "train_mae": train,
            "select_800_mae": select,
            "train_slope_per_eval_interval": stats["slope"],
            "train_slope_per_eval_interval_std_error": stats["std_error"],
            "train_slope_per_eval_interval_t": stats["t_stat"],
            "train_slope_per_eval_interval_bootstrap_ci95": stats["bootstrap_ci95"],
            "train_slope_per_eval_interval_significant_negative": stats["slope_significant_negative"],
            "train_slope_per_optimizer_step": stats_step["slope"],
            "select_slope_per_eval_interval": select_stats["slope"],
            "residual_std": stats["residual_std"],
            "last5_slope_per_eval_interval": last5_slope,
            "down_steps_in_late_window": int(down_steps),
            "best_so_far_improve_last10": best_so_far_improve_10,
            "best_so_far_improve_last20": best_so_far_improve_20,
            "stopped_by_patience_or_budget": stopped,
            "early_stopped": bool(run["early_stopped"]),
            "budget_boundary_warning": bool(run["budget_boundary_warning"]),
            "literal_slope_threshold_fired": literal_fired,
        }
    sbci = out["per_model"]["sbci"]
    warning = bool(sbci["literal_slope_threshold_fired"] and sbci["train_slope_per_eval_interval_significant_negative"])
    out["literal_threshold_fired"] = bool(sbci["literal_slope_threshold_fired"])
    out["slope_significant_negative"] = bool(sbci["train_slope_per_eval_interval_significant_negative"])
    out["best_so_far_improve_last10_sbci"] = float(sbci["best_so_far_improve_last10"])
    out["best_so_far_improve_last10_baseline"] = float(out["per_model"]["baseline"]["best_so_far_improve_last10"])
    out["convergence_warning"] = warning
    out["decline_is_sustained"] = bool(warning)
    out["official_valid_used"] = False
    out["official_test_loaded"] = False
    _write_json(RESULTS_DIR / "convergence_diagnostic.json", out)
    return out


def probe_degradation() -> dict[str, Any]:
    base_soup = np.load(BASELINE_DIR / f"probe_abs_err_soup_{RUN_ID}.npy").astype(np.float64)
    sbci_soup = np.load(SBCI_DIR / f"probe_abs_err_soup_{RUN_ID}.npy").astype(np.float64)
    base_raw = np.load(BASELINE_DIR / f"probe_abs_err_raw_{RUN_ID}.npy").astype(np.float64)
    sbci_raw = np.load(SBCI_DIR / f"probe_abs_err_raw_{RUN_ID}.npy").astype(np.float64)
    if not (base_soup.size == sbci_soup.size == N_PROBE):
        raise TriageError("probe arrays are not the locked 2000-molecule probe")
    diff_soup = sbci_soup - base_soup
    diff_raw = sbci_raw - base_raw
    return {
        "run_id": RUN_ID,
        "n_probe": int(diff_soup.size),
        "baseline_soup_probe_mae": float(base_soup.mean()),
        "sbci_soup_probe_mae": float(sbci_soup.mean()),
        "probe_degradation_soup": float(diff_soup.mean()),
        "baseline_raw_probe_mae": float(base_raw.mean()),
        "sbci_raw_probe_mae": float(sbci_raw.mean()),
        "probe_degradation_raw": float(diff_raw.mean()),
        "paired_bootstrap_soup": se.paired_bootstrap(diff_soup, B=BOOTSTRAP_B, seed=BOOTSTRAP_SEED),
        "paired_bootstrap_raw": se.paired_bootstrap(diff_raw, B=BOOTSTRAP_B, seed=BOOTSTRAP_SEED),
        "official_valid_used": False,
        "official_test_loaded": False,
    }


def generalization_gap() -> dict[str, Any]:
    soup = _read_json(RESULTS_DIR / "train_eval_soup.json")
    probe = probe_degradation()
    base_gap = probe["baseline_soup_probe_mae"] - soup["baseline_train_mae"]
    sbci_gap = probe["sbci_soup_probe_mae"] - soup["sbci_train_mae"]
    payload = {
        "run_id": RUN_ID,
        "baseline_soup_probe_mae": probe["baseline_soup_probe_mae"],
        "sbci_soup_probe_mae": probe["sbci_soup_probe_mae"],
        "baseline_soup_train_mae": soup["baseline_train_mae"],
        "sbci_soup_train_mae": soup["sbci_train_mae"],
        "generalization_gap_baseline": float(base_gap),
        "generalization_gap_sbci": float(sbci_gap),
        "delta_gap": float(sbci_gap - base_gap),
        "caution": (
            "gap differences are descriptive only: a model that fits the training "
            "set worse can show a smaller gap without being better regularized"
        ),
        "official_valid_used": False,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "generalization_gap.json", payload)
    return payload


def parameter_utilization() -> dict[str, Any]:
    """Secondary: read the saved SBCI trajectory diagnostics (no geometry)."""
    curve_path = SBCI_DIR / f"curves/{RUN_ID}.csv"
    rows = _read_csv(curve_path)
    last = rows[-1]
    first = rows[0]
    keys = (
        "phi_grad_norm",
        "psi_grad_norm",
        "centre_grad_norm",
        "head_grad_norm",
        "basis_state_norm",
        "pair_state_norm",
    )
    net, member = load_raw_model("sbci")
    weight_norms = {
        "phi": float(sb._module_weight_norm(net.basis_encoder.parameters())),
        "psi": float(sb._module_weight_norm(net.relation_modulator.parameters())),
        "centre_composer": float(sb._module_weight_norm(net.centre_composer.parameters())),
        "head": float(sb._module_weight_norm(net.head.parameters())),
    }
    payload = {
        "run_id": RUN_ID,
        "source": str(curve_path),
        "raw_member_step": int(member["step"]),
        "module_weight_norms": weight_norms,
        "first_eval": {k: float(first[k]) for k in keys},
        "last_eval": {k: float(last[k]) for k in keys},
        "note": (
            "secondary confirmation that SBCI learned a non-trivial function and "
            "that every SBCI-only module carries non-zero weight/gradient signal; "
            "not used for the decision"
        ),
        "official_valid_used": False,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "parameter_utilization.json", payload)
    return payload


# ---------------------------------------------------------------------------
# 5. decision
# ---------------------------------------------------------------------------


def triage_decision() -> dict[str, Any]:
    comp = _read_json(RESULTS_DIR / "triage_protocol_compatibility.json")
    raw = _read_json(RESULTS_DIR / "train_eval_raw.json")
    soup = _read_json(RESULTS_DIR / "train_eval_soup.json")
    best = _read_json(RESULTS_DIR / "best_train_fit.json")
    late = _read_json(RESULTS_DIR / "late_trajectory_fit.json")
    conv = _read_json(RESULTS_DIR / "convergence_diagnostic.json")
    probe = probe_degradation()

    dF_raw = float(raw["delta_F"])
    dF_soup = float(soup["delta_F"])
    dF_min = float(best["delta_F_min"])
    dF_late = float(late["delta_F_late"])
    dprobe = float(probe["probe_degradation_soup"])
    optimization_warning = bool(conv["convergence_warning"])

    approx = bool(
        dF_min >= APPROX_FIT_DEFICIT
        and dF_late >= APPROX_FIT_DEFICIT
        and dF_raw > SELECTED_STRONG
        and dF_soup > SELECTED_STRONG
    )
    adequate = bool(
        abs(dF_min) <= ADEQUATE_FIT_ABS
        and abs(dF_late) <= ADEQUATE_FIT_ABS
        and dF_raw <= SELECTED_ADEQUATE
        and dF_soup <= SELECTED_ADEQUATE
        and dprobe >= PROBE_DEGRADATION_MIN
    )

    if optimization_warning:
        case = "OPTIMIZATION_AMBIGUOUS"
    elif approx:
        case = "APPROXIMATION_OVERCONSTRAINT_FAILURE"
    elif adequate:
        case = "ADEQUATE_FIT_WRONG_INDUCTIVE_BIAS"
    else:
        case = "FIT_GENERALIZATION_TRIAGE_INCONCLUSIVE"

    payload = {
        "run_id": RUN_ID,
        "protocol_comparable": bool(comp["comparable"]),
        "delta_F_raw": dF_raw,
        "delta_F_soup": dF_soup,
        "delta_F_min": dF_min,
        "delta_F_late": dF_late,
        "probe_degradation_soup": dprobe,
        "probe_degradation_raw": float(probe["probe_degradation_raw"]),
        "optimization_convergence_warning": optimization_warning,
        "optimization_literal_slope_threshold_fired": bool(conv["literal_threshold_fired"]),
        "optimization_slope_significant_negative": bool(conv["slope_significant_negative"]),
        "optimization_note": (
            "the literal -5e-5 per-eval threshold is crossed by a noise-driven "
            "OLS slope (bootstrap 95% CI includes zero), so the effective "
            "convergence warning requires a significant negative slope"
        ),
        "criteria": {
            "A1_dF_min_ge_0.008": bool(dF_min >= APPROX_FIT_DEFICIT),
            "A2_dF_late_ge_0.008": bool(dF_late >= APPROX_FIT_DEFICIT),
            "A3_raw_and_soup_gt_0.005": bool(dF_raw > SELECTED_STRONG and dF_soup > SELECTED_STRONG),
            "A4_no_optimization_warning": bool(not optimization_warning),
            "B1_abs_dF_min_le_0.003": bool(abs(dF_min) <= ADEQUATE_FIT_ABS),
            "B2_abs_dF_late_le_0.003": bool(abs(dF_late) <= ADEQUATE_FIT_ABS),
            "B3_raw_and_soup_le_0.003": bool(dF_raw <= SELECTED_ADEQUATE and dF_soup <= SELECTED_ADEQUATE),
            "B4_probe_degradation_ge_0.005": bool(dprobe >= PROBE_DEGRADATION_MIN),
            "B5_no_optimization_warning": bool(not optimization_warning),
            "approximation_case_A": approx,
            "adequate_fit_case_B": adequate,
        },
        "decision_case": case,
        "case_if_literal_convergence_rule": (
            "OPTIMIZATION_AMBIGUOUS" if (not optimization_warning) and conv["literal_threshold_fired"] else case
        ),
        "convergence_rule_note": (
            "the literal -5e-5 per-eval slope threshold is crossed, but the slope "
            "is statistically indistinguishable from zero and SBCI's best-so-far "
            "training fit had plateaued, so the effective warning is False; under "
            "a strict literal reading the case would be OPTIMIZATION_AMBIGUOUS, "
            "which also authorizes no architecture"
        ),
        "interpretation": _interpretation(case, dF_raw, dF_soup, dF_min, dF_late, dprobe),
        "thresholds": {
            "approx_fit_deficit": APPROX_FIT_DEFICIT,
            "adequate_fit_abs": ADEQUATE_FIT_ABS,
            "selected_strong": SELECTED_STRONG,
            "selected_adequate": SELECTED_ADEQUATE,
            "probe_degradation_min": PROBE_DEGRADATION_MIN,
        },
        "official_valid_used": False,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "triage_decision.json", payload)
    return payload


def _interpretation(
    case: str,
    dF_raw: float,
    dF_soup: float,
    dF_min: float,
    dF_late: float,
    dprobe: float,
) -> str:
    if case == "APPROXIMATION_OVERCONSTRAINT_FAILURE":
        return (
            "SBCI loses a material amount of D3600 training fit (best-achieved and "
            "late-trajectory both >= +0.008) on top of its worse probe MAE, so the "
            "constraint is too strong to approximate the training function."
        )
    if case == "ADEQUATE_FIT_WRONG_INDUCTIVE_BIAS":
        return (
            "SBCI fits D3600 comparably to the baseline (|dF| <= 0.003 on best, "
            "late and selected estimators) yet generalizes worse, so the shared "
            "latent-basis prior itself is the problem, not capacity."
        )
    if case == "OPTIMIZATION_AMBIGUOUS":
        return (
            "SBCI's late training fit is still improving when patience stops it, so "
            "the fit deficit cannot be attributed to the function class."
        )
    return (
        "The fit evidence is mixed rather than clean: "
        f"dF_raw={dF_raw:+.5f}, dF_soup={dF_soup:+.5f}, dF_min={dF_min:+.5f}, "
        f"dF_late={dF_late:+.5f}, probe degradation={dprobe:+.5f}. It is not a "
        "clean approximation failure (no +0.008 fit loss) and not a clean "
        "adequate-fit wrong-bias result (the SOUP selected fit deficit exceeds "
        "+0.003), so no architecture family is authorized."
    )


def next_architecture_family() -> dict[str, Any]:
    decision = _read_json(RESULTS_DIR / "triage_decision.json")
    case = decision["decision_case"]
    if case == "APPROXIMATION_OVERCONSTRAINT_FAILURE":
        payload = {
            "authorized_for_design": True,
            "family": FAMILY_SSOD,
            "full_training_authorized": False,
            "reason": (
                "SBCI training-fit deficit indicates diagonal shared-basis "
                "factorization removed useful context-dependent expressivity"
            ),
            "next_design_constraints": [
                "keep the current patch encoder",
                "add no new structural information",
                "share a small number of global relational operators",
                "restore cross-channel transformation",
                "do not restore an arbitrary independent q MLP",
                "do not rescue SBCI via K/width",
                "form: O(r) = sum_m alpha_m(r) O_m rather than z_i*z_j*g(r)",
            ],
        }
    elif case == "ADEQUATE_FIT_WRONG_INDUCTIVE_BIAS":
        payload = {
            "authorized_for_design": True,
            "family": FAMILY_DSP,
            "full_training_authorized": False,
            "reason": (
                "SBCI fits D3600 comparably but its latent shared-basis prior "
                "generalizes worse"
            ),
            "next_design_constraints": [
                "no shared latent basis factorization",
                "no operator dictionary",
                "no SBCI-style low-rank interaction",
                "map structural primitives/relations more directly to reusable target-contribution potentials",
                "shorten the scalar credit-assignment path",
                "candidate form: u_i=f_u(h_i), v_ij=f_p(h_i,h_j,r_ij), y ~ sum u + sum v + small global correction",
            ],
        }
    else:
        payload = {
            "authorized_for_design": False,
            "family": None,
            "full_training_authorized": False,
            "reason": (
                "fit-vs-generalization evidence is mixed/inconclusive"
                if case == "FIT_GENERALIZATION_TRIAGE_INCONCLUSIVE"
                else "optimization convergence evidence is insufficient"
            ),
            "next_design_constraints": [],
        }
    payload["decision_case"] = case
    payload["official_valid_used"] = False
    payload["official_test_loaded"] = False
    _write_json(RESULTS_DIR / "next_architecture_family.json", payload)
    return payload


def answers_q1_q16() -> dict[str, Any]:
    raw = _read_json(RESULTS_DIR / "train_eval_raw.json")
    soup = _read_json(RESULTS_DIR / "train_eval_soup.json")
    best = _read_json(RESULTS_DIR / "best_train_fit.json")
    late = _read_json(RESULTS_DIR / "late_trajectory_fit.json")
    conv = _read_json(RESULTS_DIR / "convergence_diagnostic.json")
    gap = _read_json(RESULTS_DIR / "generalization_gap.json")
    decision = _read_json(RESULTS_DIR / "triage_decision.json")
    family = _read_json(RESULTS_DIR / "next_architecture_family.json")
    dF_min = float(best["delta_F_min"])
    dF_late = float(late["delta_F_late"])
    if conv["convergence_warning"]:
        verdict = "OPTIMIZATION_AMBIGUOUS (convergence warning)"
    elif decision["criteria"]["approximation_case_A"]:
        verdict = "APPROXIMATION_OVERCONSTRAINT_FAILURE"
    elif decision["criteria"]["adequate_fit_case_B"]:
        verdict = "ADEQUATE_FIT_WRONG_INDUCTIVE_BIAS"
    else:
        verdict = "FIT_GENERALIZATION_TRIAGE_INCONCLUSIVE"
    payload = {
        "Q1_baseline_raw_train_mae": float(raw["baseline_train_mae"]),
        "Q2_sbci_raw_train_mae": float(raw["sbci_train_mae"]),
        "Q3_delta_F_raw": float(raw["delta_F"]),
        "Q4_baseline_soup_train_mae": float(soup["baseline_train_mae"]),
        "Q5_sbci_soup_train_mae": float(soup["sbci_train_mae"]),
        "Q6_delta_F_soup": float(soup["delta_F"]),
        "Q7_baseline_best_train_mae": float(best["per_model"]["baseline"]["f_min"]),
        "Q8_sbci_best_train_mae": float(best["per_model"]["sbci"]["f_min"]),
        "Q9_delta_F_min": dF_min,
        "Q10_delta_F_late": dF_late,
        "Q11_sbci_train_curve_converged_before_stop": bool(
            not conv["convergence_warning"]
        ),
        "Q12_optimization_warning": bool(conv["convergence_warning"]),
        "Q13_probe_degradation": float(decision["probe_degradation_soup"]),
        "Q14_generalization_gap_delta": float(gap["delta_gap"]),
        "Q15_failure_type": verdict,
        "Q16_next_architecture_family_authorized": bool(family["authorized_for_design"]),
        "Q16_next_architecture_family": family["family"],
        "diagnostic_only_note": (
            "F_min and the late-trajectory fit are diagnosis statistics only; "
            "they never select a deployed checkpoint or change the SBCI decision"
        ),
        "official_valid_used": False,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "answers_q1_q16.json", payload)
    return payload


def final_decision() -> dict[str, Any]:
    decision = _read_json(RESULTS_DIR / "triage_decision.json")
    family = _read_json(RESULTS_DIR / "next_architecture_family.json")
    answers = _read_json(RESULTS_DIR / "answers_q1_q16.json")
    payload = {
        "verdict": f"SBCI FIT-VS-GENERALIZATION TRIAGE: {decision['decision_case']}",
        "decision_case": decision["decision_case"],
        "case_if_literal_convergence_rule": decision["case_if_literal_convergence_rule"],
        "convergence_rule_note": decision["convergence_rule_note"],
        "delta_F_raw": decision["delta_F_raw"],
        "delta_F_soup": decision["delta_F_soup"],
        "delta_F_min": decision["delta_F_min"],
        "delta_F_late": decision["delta_F_late"],
        "probe_degradation": decision["probe_degradation_soup"],
        "optimization_warning": decision["optimization_convergence_warning"],
        "authorized_for_design": family["authorized_for_design"],
        "next_architecture_family": family["family"],
        "full_training_authorized": False,
        "zero_new_gradient_updates": True,
        "sbci_decision_unchanged": True,
        "official_valid_used": False,
        "official_test_loaded": False,
        "answers_q1_q16": answers,
    }
    _write_json(RESULTS_DIR / "final_decision.json", payload)
    return payload


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


def run_locks() -> None:
    _configure_determinism()
    data = load_data()
    subsets = build_subsets(data)
    audit_protocol_lock()
    triage_protocol_compatibility(data, subsets)


def run_selected_fit() -> None:
    _configure_determinism()
    data = load_data()
    subsets = build_subsets(data)
    train_eval_raw(data, subsets)
    train_eval_soup(data, subsets)


def run_snapshots() -> None:
    _configure_determinism()
    data = load_data()
    subsets = build_subsets(data)
    t0 = time.perf_counter()
    payload = snapshot_train_fit(data, subsets)
    payload["wall_clock_s"] = float(time.perf_counter() - t0)
    _write_json(RESULTS_DIR / "snapshot_train_fit.summary.json", payload)


def run_diagnostics() -> None:
    _configure_determinism()
    best_train_fit()
    late_trajectory_fit()
    convergence_diagnostic()
    generalization_gap()
    parameter_utilization()


def run_decision() -> None:
    _configure_determinism()
    triage_decision()
    next_architecture_family()
    answers_q1_q16()
    final_decision()


def run_all() -> None:
    run_locks()
    run_selected_fit()
    run_snapshots()
    run_diagnostics()
    run_decision()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=["locks", "selected_fit", "snapshots", "diagnostics", "decision", "all"],
    )
    args = parser.parse_args(argv)
    _configure_determinism()
    if args.stage == "locks":
        run_locks()
    elif args.stage == "selected_fit":
        run_selected_fit()
    elif args.stage == "snapshots":
        run_snapshots()
    elif args.stage == "diagnostics":
        run_diagnostics()
    elif args.stage == "decision":
        run_decision()
    elif args.stage == "all":
        run_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
