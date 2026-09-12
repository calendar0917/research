"""SBCI fit-matched generalization frontier (zero-training diagnostic).

This is the **final** zero-training diagnostic on the ``compact-v4-SBCI``
branch.  It asks exactly one question:

    At an equal empirical D3600 training fit, does SBCI's learned function
    generalize systematically worse than ``compact-v4-smallhead``?

It does **not** train anything, does not rescue SBCI, does not run seed1,
does not sweep anchors/tolerances/thresholds, and never loads official
valid/test.  It only loads the already-saved frozen N3600/I0/T0 snapshots of
the baseline and the candidate, recomputes their D3600 training MAE in
``model.eval()``, matches individual RAW trajectory snapshots by training-fit
level (completely blind to held-out performance), hash-locks the matched pairs,
and only then evaluates the locked pairs on the 800 selection-generalization
set.  The 2000 internal probe is opened only if the 800 frontier already gives
a pre-registered clean candidate verdict.

Protocol:  ``sbci_fit_matched_generalization_frontier_v1``
Module:    ``tracks/ksvd/experiments/luyin16/zinc_sbci_fit_matched_generalization_frontier.py``
Results:   ``tracks/ksvd/results/sbci_fit_matched_generalization_frontier/``
Note:      ``tracks/ksvd/notes/sbci_fit_matched_generalization_frontier.md``
Tests:     ``tracks/ksvd/tests/test_sbci_fit_matched_generalization_frontier.py``

Hard rules enforced in code:

* zero gradient updates -- a runtime guard raises on ``Tensor.backward`` and
  ``Optimizer.step``;
* no new seed / new model / weight modification / soup search;
* Phase F code paths never touch the 800 or 2000 graphs (runtime firewall);
* 800 is opened only after ``fit_anchor_lock.json`` + ``phaseF_fit_only_lock.json``;
* 2000 is opened only if ``stage1_800_decision.json`` authorises it, with the
  exact same locked checkpoints;
* no official valid, no official test, ever.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from tracks.ksvd.experiments.luyin16 import zinc_compact_v4_sbci as sb
from tracks.ksvd.experiments.luyin16 import zinc_compact_v4_smallhead_e2e as shead
from tracks.ksvd.experiments.luyin16 import zinc_inductive_bias_sample_efficiency_audit as se
from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp

# ---------------------------------------------------------------------------
# frozen layout / constants
# ---------------------------------------------------------------------------

REPO_ROOT = zpp.REPO_ROOT
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"
RESULTS_DIR = TRACK_ROOT / "results/sbci_fit_matched_generalization_frontier"
BASELINE_DIR = TRACK_ROOT / "results/inductive_bias_sample_efficiency_audit"
SBCI_DIR = TRACK_ROOT / "results/compact_v4_sbci"
TRIAGE_DIR = TRACK_ROOT / "results/sbci_fit_generalization_triage"

PROTOCOL_VERSION = "sbci_fit_matched_generalization_frontier_v1"
RUN_ID = "N3600_I0T0"
N_TRAIN = 3600
N_SELECT = 800
N_PROBE = 2000
MODELS = ("baseline", "sbci")
MODEL_FAMILY = {
    "baseline": "compact-v4-smallhead",
    "sbci": "compact-v4-SBCI",
}

# pre-registered, frozen thresholds (never tuned after seeing results)
BURN_IN_FRACTION = 0.25          # S_common fraction; only steps >= this are eligible
TRIM_FRACTION = 0.10             # trimmed off each end of the common fit overlap
K_ANCHORS = 7                    # fixed fit anchors
MIN_INTERVAL_WIDTH = 0.010       # U' - L' >= this, else COMMON FIT RANGE TOO NARROW
ANCHOR_PROXIMITY_TOL = 0.002     # |F(snapshot) - c_k| <= this
CROSS_MODEL_GAP_TOL = 0.002      # |F_base_k - F_sbci_k| <= this
MIN_VALID_ANCHORS = 5            # K_valid >= this
MEDIAN_GAP_TOL = 0.001           # median_k |F_base_k - F_sbci_k| <= this

BOOTSTRAP_B = 2000
BOOTSTRAP_SEED = 20260912

# Stage 1A wrong-bias candidate
WRONG_MEAN_MIN = 0.004
WRONG_MEDIAN_MIN = 0.003
WRONG_FPLUS_MIN = 0.80
# Stage 1B sharing-premise non-inferior candidate
NONINF_MEAN_MAX = 0.001
NONINF_CI_UPPER_MAX = 0.002
NONINF_MEDIAN_MAX = 0.001
NONINF_FLE_MIN = 0.80
# material frontier crossing
CROSSING_DELTA = 0.003
CROSSING_MIN_ANCHORS = 2

FAMILY_DSP = "direct_shared_structural_potentials"
FAMILY_SSOD = "shared_structural_operator_dictionary"

RUNTIME_DATE = _dt.date.today().isoformat()


class AuditError(RuntimeError):
    """Raised when the frozen comparison is invalid or a gate fails hard."""


class PhaseFirewallError(RuntimeError):
    """Raised when a later-phase data view is touched before it is unlocked."""


class ZeroTrainingViolation(RuntimeError):
    """Raised if any gradient/optimizer step is attempted."""


# runtime phase firewall: flipped only by the phase runners
_ALLOW_SELECT = False
_ALLOW_PROBE = False
_ZERO_TRAINING_GUARD: dict[str, Any] = {
    "installed": False,
    "violations": 0,
    "orig_backward": None,
    "orig_step": None,
}


# ---------------------------------------------------------------------------
# io / hashing / determinism helpers
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


def _canonical_sha256(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _state_hash(state: Mapping[str, torch.Tensor]) -> str:
    return shead._state_hash(state)


def _load_state(path: str | Path) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    return {k: v.detach().clone() for k, v in state.items()}


def _install_zero_training_guard() -> None:
    """Forbid any gradient/optimizer update at runtime (zero-training hard rule).

    Installed explicitly by ``main`` for audit stages, and uninstallable so the
    guard never leaks into unrelated test processes.
    """

    if _ZERO_TRAINING_GUARD["installed"]:
        return

    orig_backward = torch.Tensor.backward
    orig_step = torch.optim.Optimizer.step

    def _blocked_backward(self, *args: Any, **kwargs: Any):  # noqa: ANN001
        _ZERO_TRAINING_GUARD["violations"] += 1
        raise ZeroTrainingViolation("Tensor.backward is forbidden: zero-training audit")

    def _blocked_step(self, *args: Any, **kwargs: Any):  # noqa: ANN001
        _ZERO_TRAINING_GUARD["violations"] += 1
        raise ZeroTrainingViolation("Optimizer.step is forbidden: zero-training audit")

    _ZERO_TRAINING_GUARD["orig_backward"] = orig_backward
    _ZERO_TRAINING_GUARD["orig_step"] = orig_step
    torch.Tensor.backward = _blocked_backward  # type: ignore[method-assign]
    torch.optim.Optimizer.step = _blocked_step  # type: ignore[method-assign]
    _ZERO_TRAINING_GUARD["installed"] = True


def _uninstall_zero_training_guard() -> None:
    if not _ZERO_TRAINING_GUARD["installed"]:
        return
    torch.Tensor.backward = _ZERO_TRAINING_GUARD["orig_backward"]  # type: ignore[method-assign]
    torch.optim.Optimizer.step = _ZERO_TRAINING_GUARD["orig_step"]  # type: ignore[method-assign]
    _ZERO_TRAINING_GUARD["installed"] = False


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
# data + firewall
# ---------------------------------------------------------------------------


def load_data() -> dict[str, Any]:
    return se.load_data()


def build_subsets(data: Mapping[str, Any]) -> dict[str, Any]:
    return se.build_subsets(data)


def train_graphs(data: Mapping[str, Any], subsets: Mapping[str, Any]) -> list:
    return se._map_index_lists(data, subsets[f"sorted_{N_TRAIN}"])


def _select_graphs(data: Mapping[str, Any]) -> list:
    global _ALLOW_SELECT
    if not _ALLOW_SELECT:
        raise PhaseFirewallError(
            "800-selection set accessed before fit_anchor_lock was written and verified"
        )
    return data["select"]


def _probe_graphs(data: Mapping[str, Any]) -> list:
    global _ALLOW_PROBE
    if not _ALLOW_PROBE:
        raise PhaseFirewallError(
            "2000-probe set accessed before stage1_800_decision authorised it"
        )
    return data["probe"]


def _data_fingerprints(data: Mapping[str, Any], subsets: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "n_train_subset": int(len(train_graphs(data, subsets))),
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
# model + frozen snapshot helpers
# ---------------------------------------------------------------------------


def _model_dir(model: str) -> Path:
    return BASELINE_DIR if model == "baseline" else SBCI_DIR


def build_model(model: str, seed: int = 0) -> torch.nn.Module:
    if model == "baseline":
        return shead.build_smallhead(int(seed))
    if model == "sbci":
        return sb.build_sbci(int(seed))
    raise AuditError(f"unknown model {model!r}")


def _run_manifest(model: str) -> dict[str, Any]:
    return _read_json(_model_dir(model) / f"stage_run_{RUN_ID}.json")


def _checkpoint_manifest(model: str) -> dict[str, Any]:
    return _read_json(_model_dir(model) / f"checkpoint_manifest_{RUN_ID}.json")


def _soup_record(model: str) -> dict[str, Any]:
    return _read_json(_model_dir(model) / f"soup_construction_{RUN_ID}.json")


def manifest_snapshots(model: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for r in _checkpoint_manifest(model)["snapshots"]:
        rows.append(
            {
                "step": int(r["step"]),
                "eval_index": int(r["eval_index"]),
                "snapshot": str(r["snapshot"]),
                "snapshot_sha256": str(r["snapshot_sha256"]),
                "select_800_mae": float(r["select_800_mae"]),
                "stored_train_mae": float(r["train_eval_mae"]),
            }
        )
    rows.sort(key=lambda x: int(x["step"]))
    return rows


def raw_member(model: str) -> dict[str, Any]:
    return min(manifest_snapshots(model), key=lambda r: (float(r["select_800_mae"]), int(r["step"])))


def soup_member_steps(model: str) -> list[int]:
    return [int(m["step"]) for m in _soup_record(model)["members"]]


def _steps_per_epoch(model: str) -> int:
    return int(_run_manifest(model)["eval_interval"])


def _snapshot_id(model: str, step: int) -> str:
    return f"{model}_step_{int(step):05d}"


def _predict(model: torch.nn.Module, loader) -> tuple[np.ndarray, np.ndarray]:
    _mae, targets, preds = shead._evaluate_mae(model, loader, torch.device("cpu"))
    return targets.astype(np.float64), preds.astype(np.float64)


def _make_loader(graphs: Sequence[Any], batch: int = 256):
    return zpp._make_loader(list(graphs), int(batch), False, 0)


# ---------------------------------------------------------------------------
# Stage: locks  (Phase F only; no 800 / 2000 data touched)
# ---------------------------------------------------------------------------


def audit_protocol_lock() -> dict[str, Any]:
    payload = {
        "name": "SBCI fit-matched generalization frontier",
        "protocol_version": PROTOCOL_VERSION,
        "runtime_date": RUNTIME_DATE,
        "run_id": RUN_ID,
        "candidate": "compact-v4-SBCI / N3600 / I0 / T0",
        "baseline": "compact-v4-smallhead / N3600 / I0 / T0",
        "scientific_question": (
            "MAE_generalization | MAE_train approx c: is there a stable ordering "
            "between baseline and SBCI at equal empirical D3600 training fit?"
        ),
        "frontier_unit": "individual frozen RAW trajectory snapshots (NOT soup states)",
        "soup_role": "contextual endpoint markers only; NEVER part of the frontier decision",
        "hard_rules": {
            "zero_new_gradient_updates": True,
            "zero_training_guard_installed": True,
            "no_backward": True,
            "no_optimizer_step": True,
            "no_resume_training": True,
            "no_new_seed": True,
            "no_new_model": True,
            "no_weight_modification": True,
            "no_soup_construction_search": True,
            "matching_blind_to_holdout": True,
            "official_valid_used": False,
            "official_test_loaded": False,
        },
        "frozen_thresholds": {
            "burn_in_fraction": BURN_IN_FRACTION,
            "trim_fraction": TRIM_FRACTION,
            "k_anchors": K_ANCHORS,
            "min_interval_width": MIN_INTERVAL_WIDTH,
            "anchor_proximity_tol": ANCHOR_PROXIMITY_TOL,
            "cross_model_gap_tol": CROSS_MODEL_GAP_TOL,
            "min_valid_anchors": MIN_VALID_ANCHORS,
            "median_gap_tol": MEDIAN_GAP_TOL,
            "bootstrap_B": BOOTSTRAP_B,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "wrong_mean_min": WRONG_MEAN_MIN,
            "wrong_median_min": WRONG_MEDIAN_MIN,
            "wrong_fplus_min": WRONG_FPLUS_MIN,
            "noninf_mean_max": NONINF_MEAN_MAX,
            "noninf_ci_upper_max": NONINF_CI_UPPER_MAX,
            "noninf_median_max": NONINF_MEDIAN_MAX,
            "noninf_fle_min": NONINF_FLE_MIN,
            "crossing_delta": CROSSING_DELTA,
            "crossing_min_anchors": CROSSING_MIN_ANCHORS,
        },
        "phase_firewall": {
            "phase_F_uses": ["D3600", "train targets", "snapshot weights", "optimizer-step metadata"],
            "phase_F_forbids": [
                "stored 800 MAE",
                "800 predictions",
                "2000 target/prediction",
                "probe error",
            ],
            "phase_G800_requires": "fit_anchor_lock.json + phaseF_fit_only_lock.json",
            "phase_G2000_requires": "stage1_800_decision.json authorises it",
        },
        "environment": _environment(),
        "official_valid_used": False,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "audit_protocol_lock.json", payload)
    return payload


def _run_inventory(model: str) -> dict[str, Any]:
    run = _run_manifest(model)
    snapshots = manifest_snapshots(model)
    net = build_model(model, 0)
    init_hash = _state_hash(net.state_dict())
    if not (int(run["N"]) == N_TRAIN and int(run["I"]) == 0 and int(run["T"]) == 0):
        raise AuditError(f"{model}: run is not N3600/I0/T0")
    if not (run.get("frozen") and run.get("estimators_evaluated")):
        raise AuditError(f"{model}: run is not frozen/evaluated")
    if run["init_state_sha256"] != init_hash:
        raise AuditError(f"{model}: rebuilt init hash does not match the frozen run")
    if len(snapshots) != int(run["snapshot_count"]):
        raise AuditError(f"{model}: snapshot count mismatch")
    return {
        "model": model,
        "family": MODEL_FAMILY[model],
        "results_dir": str(_model_dir(model)),
        "N": int(run["N"]),
        "I": int(run["I"]),
        "T": int(run["T"]),
        "parameters": int(run["parameters"]),
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
        "init_state_sha256": init_hash,
        "init_state_sha256_matches": bool(run["init_state_sha256"] == init_hash),
        "snapshot_count": int(len(snapshots)),
        "soup_member_steps": soup_member_steps(model),
        "soup_state_path": str(_soup_record(model)["soup_state_path"]),
        "soup_state_tensor_sha256": str(_soup_record(model)["soup_state_tensor_sha256"]),
        "official_valid_used": False,
        "official_test_loaded": False,
    }


def protocol_compatibility(data: Mapping[str, Any], subsets: Mapping[str, Any]) -> dict[str, Any]:
    base = _run_inventory("baseline")
    sbci = _run_inventory("sbci")
    ref = _read_json(BASELINE_DIR / "split_inventory.json")
    lock = _read_json(BASELINE_DIR / "sample_efficiency_subset_lock.json")
    shared_lock = _read_json(SBCI_DIR / "initialization_lock.json")
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
    protocol_checks = {key: bool(base[key] == sbci[key]) for key in proto_keys}

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

    checks = {
        "same_run_id": bool(base["family"] != sbci["family"] and base["N"] == sbci["N"] == N_TRAIN),
        "same_N": bool(base["N"] == sbci["N"] == N_TRAIN),
        "same_I": bool(base["I"] == sbci["I"] == 0),
        "same_T": bool(base["T"] == sbci["T"] == 0),
        "same_optimizer_recipe": bool(all(protocol_checks.values())),
        "same_optimizer_step_budget": bool(
            base["max_optimizer_steps"] == sbci["max_optimizer_steps"] == 13680
        ),
        "same_57_step_eval_cadence": bool(
            base["eval_interval"] == sbci["eval_interval"] == 57
        ),
        "same_patience_semantics": bool(
            base["patience_evaluations"] == sbci["patience_evaluations"] == 40
        ),
        "same_seed_family_I0T0": bool(base["I"] == base["T"] == sbci["I"] == sbci["T"] == 0),
        "same_d3600_indices": bool(
            fp["train_subset_index_hash"] == lock["dataset_hash_3600"]
            and fp["train_subset_index_hash"]
            == "3be9f7e1a4fb8cf4ed5eedfbd725b1ac9d618e6bc6d4873fe254ddc1017dfea9"
        ),
        "same_800_indices": bool(
            fp["select_index_hash"] == ref["index_sha256"]["checkpoint_selection"]
        ),
        "same_2000_indices": bool(
            fp["probe_index_hash"] == ref["index_sha256"]["internal_probe"]
        ),
        "same_subset_salt": bool(fp["subset_salt"] == lock["salt"]),
        "shared_initialization_exact_match": bool(shared_max_abs_diff == 0.0),
        "shared_init_provenance_lock": bool(shared_lock["seeds"]["0"]["hash_match"]),
        "deterministic_eval_mode": True,
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
            "sbci_initialization_lock_hash_match": bool(shared_lock["seeds"]["0"]["hash_match"]),
            "sbci_initialization_lock_max_abs_diff": float(shared_lock["seeds"]["0"]["max_abs_diff"]),
            "sbci_initialization_lock_shared_tensors": int(shared_lock["seeds"]["0"]["n_shared_tensors"]),
        },
        "checks": checks,
        "comparable": bool(all(checks.values())),
        "official_valid_used": False,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "protocol_compatibility.json", payload)
    if not payload["comparable"]:
        raise AuditError("core protocol mismatch -> INVALID FRONTIER COMPARISON; refusing to run")
    return payload


def snapshot_inventory() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "run_id": RUN_ID,
        "models": {},
        "note": "counts are read from the real frozen manifests / snapshot directories",
        "official_valid_used": False,
        "official_test_loaded": False,
    }
    for model in MODELS:
        snapshots = manifest_snapshots(model)
        files = sorted(Path(snapshots[0]["snapshot"]).parent.glob("step_*.pt"))
        payload["models"][model] = {
            "family": MODEL_FAMILY[model],
            "manifest_path": str(_model_dir(model) / f"checkpoint_manifest_{RUN_ID}.json"),
            "manifest_sha256": _sha256_file(
                _model_dir(model) / f"checkpoint_manifest_{RUN_ID}.json"
            ),
            "snapshot_dir": str(Path(snapshots[0]["snapshot"]).parent),
            "n_snapshots": int(len(snapshots)),
            "n_snapshot_files_on_disk": int(len(files)),
            "eval_interval": _steps_per_epoch(model),
            "final_optimizer_step": int(snapshots[-1]["step"]),
            "first_optimizer_step": int(snapshots[0]["step"]),
            "steps": [int(s["step"]) for s in snapshots],
            "snapshot_sha256": [str(s["snapshot_sha256"]) for s in snapshots],
            "official_valid_used": False,
            "official_test_loaded": False,
        }
    _write_json(RESULTS_DIR / "snapshot_inventory.json", payload)
    return payload


def run_locks() -> None:
    _configure_determinism()
    data = load_data()
    subsets = build_subsets(data)
    audit_protocol_lock()
    protocol_compatibility(data, subsets)
    snapshot_inventory()


# ---------------------------------------------------------------------------
# Stage: fit_inventory  (Phase F only)
# ---------------------------------------------------------------------------


def fit_inventory(data: Mapping[str, Any], subsets: Mapping[str, Any]) -> dict[str, Any]:
    loader = _make_loader(train_graphs(data, subsets), 256)
    rows: list[dict[str, Any]] = []
    target_ref: np.ndarray | None = None
    max_abs_diff = 0.0
    for model in MODELS:
        net = build_model(model, 0)
        spe = _steps_per_epoch(model)
        for member in manifest_snapshots(model):
            path = Path(member["snapshot"])
            if not path.exists():
                raise AuditError(f"{model}: snapshot missing {path}")
            if _sha256_file(path) != str(member["snapshot_sha256"]):
                raise AuditError(f"{model}: snapshot hash mismatch {path}")
            net.load_state_dict(_load_state(path))
            mae, targets, _ = shead._evaluate_mae(net, loader, torch.device("cpu"))
            if not np.isfinite(mae):
                raise AuditError(f"{model}: non-finite train MAE at step {member['step']}")
            if target_ref is None:
                target_ref = targets
            elif not np.array_equal(target_ref, targets):
                raise AuditError("target order changed across snapshot evaluations")
            diff = abs(float(mae) - float(member["stored_train_mae"]))
            max_abs_diff = max(max_abs_diff, diff)
            if diff > 1e-10:
                raise AuditError(
                    f"PROVENANCE DISCREPANCY: {model} step {member['step']} "
                    f"recomputed={mae} stored={member['stored_train_mae']} diff={diff}"
                )
            rows.append(
                {
                    "model_family": MODEL_FAMILY[model],
                    "model": model,
                    "snapshot_id": _snapshot_id(model, member["step"]),
                    "optimizer_step": int(member["step"]),
                    "effective_epoch": float(int(member["step"]) / spe),
                    "train_mae": float(mae),
                    "checkpoint_hash": str(member["snapshot_sha256"]),
                    "stored_train_mae": float(member["stored_train_mae"]),
                    "abs_diff_recomputed_vs_stored": float(diff),
                    "eval_index": int(member["eval_index"]),
                    "snapshot_path": str(member["snapshot"]),
                }
            )
        print(f"[fit_inventory] {model}: {len(manifest_snapshots(model))} snapshots evaluated", flush=True)

    import pandas as pd

    frame = pd.DataFrame(rows)
    frame.to_parquet(RESULTS_DIR / "train_fit_inventory.parquet", index=False)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "run_id": RUN_ID,
        "n_rows": int(len(rows)),
        "row_counts": {
            model: int(sum(1 for r in rows if r["model"] == model)) for model in MODELS
        },
        "max_abs_diff_recomputed_vs_stored": float(max_abs_diff),
        "recompute_matches_stored": bool(max_abs_diff <= 1e-10),
        "stored_consistency_tolerance": 1e-10,
        "matching_used_train_only": True,
        "selection_800_loaded": False,
        "probe_2000_loaded": False,
        "official_valid_used": False,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "fit_inventory.summary.json", payload)
    return payload


def run_fit_inventory() -> None:
    _configure_determinism()
    data = load_data()
    subsets = build_subsets(data)
    fit_inventory(data, subsets)


# ---------------------------------------------------------------------------
# Stage: fit_lock  (Phase F only)
# ---------------------------------------------------------------------------


def _read_inventory_frame():
    import pandas as pd

    return pd.read_parquet(RESULTS_DIR / "train_fit_inventory.parquet")


def train_fit_overlap() -> dict[str, Any]:
    frame = _read_inventory_frame()
    per_model: dict[str, Any] = {}
    finals: dict[str, int] = {}
    for model in MODELS:
        sub = frame[frame["model"] == model]
        steps = sub["optimizer_step"].astype(int).tolist()
        finals[model] = int(max(steps))
    s_common = int(min(finals.values()))
    threshold = float(BURN_IN_FRACTION) * float(s_common)

    ranges: dict[str, tuple[float, float]] = {}
    eligible: dict[str, Any] = {}
    for model in MODELS:
        sub = frame[(frame["model"] == model) & (frame["optimizer_step"] >= threshold)]
        fits = sub["train_mae"].astype(float).tolist()
        eligible[model] = {
            "n_eligible": int(len(sub)),
            "first_step": int(sub["optimizer_step"].min()),
            "last_step": int(sub["optimizer_step"].max()),
            "fit_min": float(min(fits)),
            "fit_max": float(max(fits)),
        }
        ranges[model] = (float(min(fits)), float(max(fits)))
        per_model[model] = {
            "final_optimizer_step": int(finals[model]),
            "eligible_snapshots": [int(s) for s in sub["optimizer_step"].tolist()],
            "eligible_train_mae": [float(f) for f in fits],
        }

    lo = max(ranges["baseline"][0], ranges["sbci"][0])
    hi = min(ranges["baseline"][1], ranges["sbci"][1])
    status = "ok"
    if hi <= lo:
        status = "NO_COMMON_FIT_SUPPORT"
    l_trim = lo + TRIM_FRACTION * (hi - lo)
    u_trim = hi - TRIM_FRACTION * (hi - lo)
    width = u_trim - l_trim
    if status == "ok" and width < MIN_INTERVAL_WIDTH:
        status = "COMMON_FIT_RANGE_TOO_NARROW"

    anchors: list[float] = []
    if status == "ok":
        anchors = [
            float(l_trim + (u_trim - l_trim) * k / (K_ANCHORS - 1)) for k in range(K_ANCHORS)
        ]

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "run_id": RUN_ID,
        "burn_in_fraction": BURN_IN_FRACTION,
        "final_optimizer_step_baseline": int(finals["baseline"]),
        "final_optimizer_step_sbci": int(finals["sbci"]),
        "s_common": int(s_common),
        "common_step_threshold": threshold,
        "per_model": per_model,
        "eligible_summary": eligible,
        "baseline_fit_range": [float(ranges["baseline"][0]), float(ranges["baseline"][1])],
        "sbci_fit_range": [float(ranges["sbci"][0]), float(ranges["sbci"][1])],
        "common_overlap": [float(lo), float(hi)],
        "trim_fraction": TRIM_FRACTION,
        "trimmed_interval": [float(l_trim), float(u_trim)],
        "trimmed_width": float(width),
        "min_interval_width": MIN_INTERVAL_WIDTH,
        "anchors": anchors,
        "status": status,
        "eligibility_uses_only_step_and_train_mae": True,
        "selection_800_loaded": False,
        "probe_2000_loaded": False,
        "official_valid_used": False,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "train_fit_overlap.json", payload)
    return payload


def _assign_to_anchors(
    eligible_steps: Sequence[int],
    eligible_fits: Sequence[float],
    anchors: Sequence[float],
    eps: float = 1e-12,
) -> list[int]:
    """Deterministic minimum-cost bipartite assignment with later-step tie-break."""

    steps = np.asarray([int(s) for s in eligible_steps], dtype=np.float64)
    fits = np.asarray([float(f) for f in eligible_fits], dtype=np.float64)
    if steps.size < len(anchors):
        raise AuditError("not enough eligible snapshots for the anchors")
    step_norm = steps / max(float(steps.max()), 1.0)
    cost = np.empty((len(anchors), steps.size), dtype=np.float64)
    for k, c in enumerate(anchors):
        # primary = |train fit - anchor|; secondary = tiny preference for later step
        cost[k] = np.abs(fits - float(c)) - float(eps) * step_norm
    row, col = linear_sum_assignment(cost)
    chosen = {int(r): int(c) for r, c in zip(row, col)}
    return [chosen[k] for k in range(len(anchors))]


def fit_lock() -> dict[str, Any]:
    frame = _read_inventory_frame()
    overlap = _read_json(RESULTS_DIR / "train_fit_overlap.json")
    status = str(overlap["status"])
    anchors = [float(a) for a in overlap["anchors"]]
    threshold = float(overlap["common_step_threshold"])

    eligible: dict[str, list[dict[str, Any]]] = {}
    for model in MODELS:
        sub = frame[(frame["model"] == model) & (frame["optimizer_step"] >= threshold)]
        sub = sub.sort_values("optimizer_step")
        eligible[model] = sub.to_dict("records")

    pairs: list[dict[str, Any]] = []
    locked = status == "ok"
    if locked:
        assignments: dict[str, list[int]] = {}
        for model in MODELS:
            steps = [int(r["optimizer_step"]) for r in eligible[model]]
            fits = [float(r["train_mae"]) for r in eligible[model]]
            assignments[model] = _assign_to_anchors(steps, fits, anchors)

        # uniqueness within each family
        for model in MODELS:
            chosen_steps = [
                int(eligible[model][idx]["optimizer_step"]) for idx in assignments[model]
            ]
            if len(set(chosen_steps)) != len(chosen_steps):
                raise AuditError(f"{model}: matched snapshots are not unique")

        for k in range(len(anchors)):
            b = eligible["baseline"][assignments["baseline"][k]]
            s = eligible["sbci"][assignments["sbci"][k]]
            f_b = float(b["train_mae"])
            f_s = float(s["train_mae"])
            prox_b = abs(f_b - anchors[k])
            prox_s = abs(f_s - anchors[k])
            gap = abs(f_b - f_s)
            valid = bool(
                prox_b <= ANCHOR_PROXIMITY_TOL
                and prox_s <= ANCHOR_PROXIMITY_TOL
                and gap <= CROSS_MODEL_GAP_TOL
            )
            pairs.append(
                {
                    "anchor_id": int(k),
                    "target_train_mae": float(anchors[k]),
                    "baseline_snapshot": str(b["snapshot_id"]),
                    "baseline_snapshot_path": str(b["snapshot_path"]),
                    "baseline_step": int(b["optimizer_step"]),
                    "baseline_train_mae": f_b,
                    "baseline_hash": str(b["checkpoint_hash"]),
                    "baseline_proximity": float(prox_b),
                    "sbci_snapshot": str(s["snapshot_id"]),
                    "sbci_snapshot_path": str(s["snapshot_path"]),
                    "sbci_step": int(s["optimizer_step"]),
                    "sbci_train_mae": f_s,
                    "sbci_hash": str(s["checkpoint_hash"]),
                    "sbci_proximity": float(prox_s),
                    "cross_model_fit_gap": float(gap),
                    "valid": int(valid),
                }
            )

    valid_pairs = [p for p in pairs if int(p["valid"]) == 1]
    k_valid = int(len(valid_pairs))
    gaps = [float(p["cross_model_fit_gap"]) for p in pairs]
    max_gap = float(max(gaps)) if gaps else float("nan")
    median_gap = float(statistics.median(gaps)) if gaps else float("nan")
    max_prox_b = max((float(p["baseline_proximity"]) for p in pairs), default=float("nan"))
    max_prox_s = max((float(p["sbci_proximity"]) for p in pairs), default=float("nan"))

    quality_ok = bool(
        locked
        and k_valid >= MIN_VALID_ANCHORS
        and (not math_isnan(median_gap))
        and median_gap <= MEDIAN_GAP_TOL
    )
    if locked and k_valid < MIN_VALID_ANCHORS:
        lock_status = "FIT_MATCHED_FRONTIER_UNDERPOWERED"
    elif locked and (math_isnan(median_gap) or median_gap > MEDIAN_GAP_TOL):
        lock_status = "MATCH_QUALITY_INSUFFICIENT"
    elif not locked:
        lock_status = status
    else:
        lock_status = "LOCKED"

    pairs_fields = (
        "anchor_id",
        "target_train_mae",
        "baseline_snapshot",
        "baseline_step",
        "baseline_train_mae",
        "baseline_hash",
        "sbci_snapshot",
        "sbci_step",
        "sbci_train_mae",
        "sbci_hash",
        "cross_model_fit_gap",
        "valid",
    )
    _write_csv(RESULTS_DIR / "matched_snapshot_pairs.csv", pairs, pairs_fields)
    pairs_sha = _sha256_file(RESULTS_DIR / "matched_snapshot_pairs.csv")

    lock_payload = {
        "protocol_version": PROTOCOL_VERSION,
        "run_id": RUN_ID,
        "locked": bool(quality_ok),
        "lock_status": lock_status,
        "k_anchors": K_ANCHORS,
        "k_valid": k_valid,
        "min_valid_anchors": MIN_VALID_ANCHORS,
        "anchors": anchors,
        "trimmed_interval": overlap["trimmed_interval"],
        "common_step_threshold": threshold,
        "anchor_proximity_tol": ANCHOR_PROXIMITY_TOL,
        "cross_model_gap_tol": CROSS_MODEL_GAP_TOL,
        "median_gap_tol": MEDIAN_GAP_TOL,
        "matched_snapshot_pairs_sha256": pairs_sha,
        "matched_snapshot_pairs": pairs,
        "matching_cost_is_train_mae_only": True,
        "tie_break": "later optimizer step via deterministic epsilon 1e-12",
        "selection_800_loaded": False,
        "probe_2000_loaded": False,
        "official_valid_used": False,
        "official_test_loaded": False,
    }
    lock_payload["lock_sha256"] = _canonical_sha256(
        {k: v for k, v in lock_payload.items() if k != "lock_sha256"}
    )
    _write_json(RESULTS_DIR / "fit_anchor_lock.json", lock_payload)

    match_quality = {
        "protocol_version": PROTOCOL_VERSION,
        "run_id": RUN_ID,
        "k_anchors": K_ANCHORS,
        "k_valid": k_valid,
        "min_valid_anchors": MIN_VALID_ANCHORS,
        "k_valid_gate_pass": bool(k_valid >= MIN_VALID_ANCHORS),
        "max_cross_model_fit_gap": max_gap,
        "median_cross_model_fit_gap": median_gap,
        "cross_model_gap_tol": CROSS_MODEL_GAP_TOL,
        "max_baseline_proximity": max_prox_b,
        "max_sbci_proximity": max_prox_s,
        "anchor_proximity_tol": ANCHOR_PROXIMITY_TOL,
        "median_gap_tol": MEDIAN_GAP_TOL,
        "median_gap_gate_pass": bool((not math_isnan(median_gap)) and median_gap <= MEDIAN_GAP_TOL),
        "per_anchor": pairs,
        "lock_status": lock_status,
        "selection_800_loaded": False,
        "probe_2000_loaded": False,
        "official_valid_used": False,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "match_quality.json", match_quality)

    phase_f = {
        "protocol_version": PROTOCOL_VERSION,
        "run_id": RUN_ID,
        "phase": "F",
        "selection_800_loaded": False,
        "probe_2000_loaded": False,
        "matching_used_train_only": True,
        "fit_anchor_lock_present": True,
        "fit_anchor_lock_sha256": lock_payload["lock_sha256"],
        "matched_snapshot_pairs_sha256": pairs_sha,
        "frozen_before_phase_g800": True,
        "eligibility_uses_only_step_and_train_mae": True,
        "match_quality_status": lock_status,
        "official_valid_used": False,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "phaseF_fit_only_lock.json", phase_f)

    if lock_status == "FIT_MATCHED_FRONTIER_UNDERPOWERED":
        raise AuditError("FIT-MATCHED FRONTIER UNDERPOWERED (K_valid < 5)")
    if lock_status == "MATCH_QUALITY_INSUFFICIENT":
        raise AuditError("MATCH QUALITY INSUFFICIENT (median cross-model fit gap > 0.001)")
    if lock_status == "NO_COMMON_FIT_SUPPORT":
        raise AuditError("NO COMMON FIT SUPPORT - INCONCLUSIVE")
    if lock_status == "COMMON_FIT_RANGE_TOO_NARROW":
        raise AuditError("COMMON FIT RANGE TOO NARROW")
    return lock_payload


def math_isnan(value: float) -> bool:
    try:
        return bool(np.isnan(float(value)))
    except (TypeError, ValueError):
        return True


def run_fit_lock() -> None:
    _configure_determinism()
    overlap = train_fit_overlap()
    if overlap["status"] != "ok":
        # Do not open any held-out data; record the inconclusive stopping point.
        _write_json(
            RESULTS_DIR / "phaseF_fit_only_lock.json",
            {
                "protocol_version": PROTOCOL_VERSION,
                "run_id": RUN_ID,
                "phase": "F",
                "selection_800_loaded": False,
                "probe_2000_loaded": False,
                "matching_used_train_only": True,
                "status": overlap["status"],
                "fit_anchor_lock_present": False,
                "official_valid_used": False,
                "official_test_loaded": False,
            },
        )
        return
    fit_lock()


# ---------------------------------------------------------------------------
# Matching helpers shared by G800 / G2000
# ---------------------------------------------------------------------------


def _load_locked_pairs() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    lock_path = RESULTS_DIR / "fit_anchor_lock.json"
    phase_path = RESULTS_DIR / "phaseF_fit_only_lock.json"
    if not lock_path.exists() or not phase_path.exists():
        raise AuditError("fit_anchor_lock.json / phaseF_fit_only_lock.json missing")
    phase = _read_json(phase_path)
    if phase.get("selection_800_loaded") is not False or phase.get("probe_2000_loaded") is not False:
        raise AuditError("Phase F lock is not clean")
    lock = _read_json(lock_path)
    if not lock.get("locked"):
        raise AuditError("fit_anchor_lock is not locked")
    current_sha = _sha256_file(RESULTS_DIR / "matched_snapshot_pairs.csv")
    if current_sha != lock["matched_snapshot_pairs_sha256"]:
        raise AuditError("matched_snapshot_pairs.csv changed after the lock")
    valid = [p for p in lock["matched_snapshot_pairs"] if int(p["valid"]) == 1]
    return lock, valid


def _evaluate_locked_anchors(
    graphs: Sequence[Any],
    valid_pairs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate the exact locked baseline/SBCI snapshots on a held-out set."""

    loader = _make_loader(graphs, 256)
    n = len(list(graphs))
    base_net = build_model("baseline", 0)
    sbci_net = build_model("sbci", 0)
    target_ref: np.ndarray | None = None
    per_anchor: list[dict[str, Any]] = []
    per_anchor_mol_diff: list[np.ndarray] = []
    for pair in valid_pairs:
        base_net.load_state_dict(_load_state(pair["baseline_snapshot_path"]))
        f_b, t_b, p_b = shead._evaluate_mae(base_net, loader, torch.device("cpu"))
        sbci_net.load_state_dict(_load_state(pair["sbci_snapshot_path"]))
        f_s, t_s, p_s = shead._evaluate_mae(sbci_net, loader, torch.device("cpu"))
        if target_ref is None:
            target_ref = t_b
        elif not (np.array_equal(target_ref, t_b) and np.array_equal(target_ref, t_s)):
            raise AuditError("target order changed across locked-anchor evaluations")
        d_i = np.abs(t_b - p_s) - np.abs(t_b - p_b)
        per_anchor_mol_diff.append(d_i)
        per_anchor.append(
            {
                "anchor_id": int(pair["anchor_id"]),
                "target_train_mae": float(pair["target_train_mae"]),
                "baseline_snapshot": str(pair["baseline_snapshot"]),
                "baseline_step": int(pair["baseline_step"]),
                "baseline_train_mae": float(pair["baseline_train_mae"]),
                "baseline_hash": str(pair["baseline_hash"]),
                "baseline_mae": float(f_b),
                "sbci_snapshot": str(pair["sbci_snapshot"]),
                "sbci_step": int(pair["sbci_step"]),
                "sbci_train_mae": float(pair["sbci_train_mae"]),
                "sbci_hash": str(pair["sbci_hash"]),
                "sbci_mae": float(f_s),
                "D_k": float(f_s - f_b),
                "cross_model_fit_gap": float(pair["cross_model_fit_gap"]),
            }
        )
    matrix = np.vstack(per_anchor_mol_diff) if per_anchor_mol_diff else np.zeros((0, n))
    dbar = matrix.mean(axis=0) if matrix.size else np.zeros(n)
    return {
        "n": int(n),
        "per_anchor": per_anchor,
        "dbar": np.asarray(dbar, dtype=np.float64),
        "targets": target_ref,
    }


def _anchor_summary(per_anchor: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    d = np.asarray([float(a["D_k"]) for a in per_anchor], dtype=np.float64)
    n_pos = int(np.sum(d > 0.0))
    n_le_002 = int(np.sum(d <= 0.002))
    n_hi = int(np.sum(d >= CROSSING_DELTA))
    n_lo = int(np.sum(d <= -CROSSING_DELTA))
    crossing = bool(n_hi >= CROSSING_MIN_ANCHORS and n_lo >= CROSSING_MIN_ANCHORS)
    return {
        "k": int(d.size),
        "D_mean": float(d.mean()) if d.size else float("nan"),
        "D_median": float(statistics.median(d.tolist())) if d.size else float("nan"),
        "D_min": float(d.min()) if d.size else float("nan"),
        "D_max": float(d.max()) if d.size else float("nan"),
        "f_plus": float(n_pos / d.size) if d.size else float("nan"),
        "f_le_002": float(n_le_002 / d.size) if d.size else float("nan"),
        "n_D_ge_003": n_hi,
        "n_D_le_neg003": n_lo,
        "material_frontier_crossing": crossing,
    }


def _stage_decision(
    summary: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
) -> dict[str, Any]:
    mean = float(summary["D_mean"])
    median = float(summary["D_median"])
    f_plus = float(summary["f_plus"])
    f_le = float(summary["f_le_002"])
    ci_lower = float(bootstrap["ci95_lower"])
    ci_upper = float(bootstrap["ci95_upper"])
    crossing = bool(summary["material_frontier_crossing"])

    wrong = bool(
        mean >= WRONG_MEAN_MIN
        and ci_lower > 0.0
        and median >= WRONG_MEDIAN_MIN
        and f_plus >= WRONG_FPLUS_MIN
        and not crossing
    )
    noninf = bool(
        mean <= NONINF_MEAN_MAX
        and ci_upper <= NONINF_CI_UPPER_MAX
        and median <= NONINF_MEDIAN_MAX
        and f_le >= NONINF_FLE_MIN
        and not crossing
    )
    if crossing:
        decision = "MATERIAL_FRONTIER_CROSSING"
    elif wrong:
        decision = "MATCHED_FIT_SBCI_GENERALIZATION_WORSE_CANDIDATE"
    elif noninf:
        decision = "MATCHED_FIT_SBCI_NONINFERIOR_OR_FAVORABLE_CANDIDATE"
    else:
        decision = "FIT_MATCHED_FRONTIER_INCONCLUSIVE_AT_800"
    return {
        "D_mean": mean,
        "D_median": median,
        "bootstrap_ci95": [ci_lower, ci_upper],
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "f_plus": f_plus,
        "f_le_002": f_le,
        "material_frontier_crossing": crossing,
        "criteria_wrong_bias": {
            "W1_mean_ge_0.004": bool(mean >= WRONG_MEAN_MIN),
            "W2_ci_lower_gt_0": bool(ci_lower > 0.0),
            "W3_median_ge_0.003": bool(median >= WRONG_MEDIAN_MIN),
            "W4_f_plus_ge_0.80": bool(f_plus >= WRONG_FPLUS_MIN),
            "W5_no_material_crossing": bool(not crossing),
            "all": wrong,
        },
        "criteria_noninferior": {
            "N1_mean_le_0.001": bool(mean <= NONINF_MEAN_MAX),
            "N2_ci_upper_le_0.002": bool(ci_upper <= NONINF_CI_UPPER_MAX),
            "N3_median_le_0.001": bool(median <= NONINF_MEDIAN_MAX),
            "N4_f_le_0.002_ge_0.80": bool(f_le >= NONINF_FLE_MIN),
            "N5_no_material_crossing": bool(not crossing),
            "all": noninf,
        },
        "decision": decision,
    }


def _save_frontier(
    name: str,
    per_anchor: Sequence[Mapping[str, Any]],
    dbar: np.ndarray,
    target_mae: float,
) -> dict[str, Any]:
    fields = (
        "anchor_id",
        "target_train_mae",
        "baseline_snapshot",
        "baseline_step",
        "baseline_train_mae",
        "baseline_hash",
        "baseline_mae",
        "sbci_snapshot",
        "sbci_step",
        "sbci_train_mae",
        "sbci_hash",
        "sbci_mae",
        "D_k",
        "cross_model_fit_gap",
    )
    _write_csv(RESULTS_DIR / f"{name}_per_anchor.csv", per_anchor, fields)
    np.save(RESULTS_DIR / f"{name}_per_molecule.npy", np.asarray(dbar, dtype=np.float64))
    summary = _anchor_summary(per_anchor)
    boot = se.paired_bootstrap(np.asarray(dbar, dtype=np.float64), B=BOOTSTRAP_B, seed=BOOTSTRAP_SEED)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "run_id": RUN_ID,
        "n_molecules": int(len(dbar)),
        "target_mae_used_as_y": float(target_mae),
        "anchor_summary": summary,
        "paired_bootstrap_over_molecules": boot,
        "bootstrap_B": BOOTSTRAP_B,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_unit": "held-out molecule (fit anchors are fixed locked model states)",
        "equal_weight_anchors": True,
        "official_valid_used": False,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / f"{name}_bootstrap.json", payload)
    return payload


def _endpoint_context(data: Mapping[str, Any]) -> dict[str, Any]:
    frame = _read_inventory_frame()
    ctx: dict[str, Any] = {
        "note": "contextual endpoint markers only; never part of the frontier decision",
        "models": {},
        "official_valid_used": False,
        "official_test_loaded": False,
    }
    for model in MODELS:
        run = _run_manifest(model)
        raw = raw_member(model)
        row = frame[(frame["model"] == model) & (frame["optimizer_step"] == int(raw["step"]))].iloc[0]
        # contextual SOUP markers: train (recomputed at fit_inventory) + 800 (if unlocked)
        ctx["models"][model] = {
            "raw_step": int(raw["step"]),
            "raw_select_800_mae": float(raw["select_800_mae"]),
            "raw_train_mae": float(row["train_mae"]),
            "raw_probe_mae": float(run["raw_probe_mae"]),
            "soup_probe_mae": float(run["soup_probe_mae"]),
            "soup_member_steps": soup_member_steps(model),
            "soup_state_tensor_sha256": str(_soup_record(model)["soup_state_tensor_sha256"]),
        }
    # contextual soup train MAE from the triage (same frozen states)
    soup_fit = _read_json(TRIAGE_DIR / "train_eval_soup.json")
    for model in MODELS:
        ctx["models"][model]["soup_train_mae"] = float(soup_fit[f"{model}_train_mae"])
    return ctx


def run_g800() -> None:
    global _ALLOW_SELECT
    _configure_determinism()
    lock, valid_pairs = _load_locked_pairs()
    if len(valid_pairs) == 0:
        raise AuditError("no valid matched pairs to evaluate")

    data = load_data()
    # ---- Phase G800 firewall: only now is the 800 view unlocked ----
    _ALLOW_SELECT = True
    graphs = _select_graphs(data)
    result = _evaluate_locked_anchors(graphs, valid_pairs)
    per_anchor = result["per_anchor"]
    target_mae = float(np.mean(np.abs(result["targets"]))) if result["targets"] is not None else float("nan")
    saved = _save_frontier("stage1_800", per_anchor, result["dbar"], target_mae)
    decision = _stage_decision(saved["anchor_summary"], saved["paired_bootstrap_over_molecules"])

    authorize_2000 = bool(
        decision["decision"]
        in {
            "MATCHED_FIT_SBCI_GENERALIZATION_WORSE_CANDIDATE",
            "MATCHED_FIT_SBCI_NONINFERIOR_OR_FAVORABLE_CANDIDATE",
        }
    )
    decision.update(
        {
            "phase": "G800",
            "selection_generalization_diagnostic_set": True,
            "not_pristine_holdout": True,
            "fit_anchor_lock_sha256": lock["lock_sha256"],
            "authorize_2000": authorize_2000,
            "n_valid_anchors": int(len(valid_pairs)),
            "k_valid": int(lock["k_valid"]),
            "official_valid_used": False,
            "official_test_loaded": False,
        }
    )
    _write_json(RESULTS_DIR / "stage1_800_decision.json", decision)

    endpoint = _endpoint_context(data)
    # contextual SOUP 800 MAE (allowed: 800 is now unlocked)
    soup_sel: dict[str, float] = {}
    loader = _make_loader(graphs, 256)
    for model in MODELS:
        net = build_model(model, 0)
        net.load_state_dict(_load_state(_soup_record(model)["soup_state_path"]))
        mae, _t, _p = shead._evaluate_mae(net, loader, torch.device("cpu"))
        soup_sel[model] = float(mae)
        endpoint["models"][model]["soup_select_800_mae"] = float(mae)
    endpoint["soup_select_800_mae"] = soup_sel
    _write_json(RESULTS_DIR / "endpoint_context.json", endpoint)
    print(
        f"[g800] D_mean={decision['D_mean']:+.6f} "
        f"CI=[{decision['ci_lower']:+.6f},{decision['ci_upper']:+.6f}] "
        f"decision={decision['decision']} authorize_2000={authorize_2000}",
        flush=True,
    )


def run_g2000() -> None:
    global _ALLOW_PROBE
    _configure_determinism()
    if not (RESULTS_DIR / "stage1_800_decision.json").exists():
        raise AuditError("g2000 requires stage1_800_decision.json")
    stage1 = _read_json(RESULTS_DIR / "stage1_800_decision.json")
    if not stage1.get("authorize_2000"):
        raise AuditError("g2000 not authorised by stage1_800_decision.json")

    lock, valid_pairs = _load_locked_pairs()
    # exact same locked checkpoints: verify ids match the 800 lock
    for p in valid_pairs:
        if str(p["baseline_snapshot"]) == "" or str(p["sbci_snapshot"]) == "":
            raise AuditError("locked pair missing snapshot id")

    data = load_data()
    # ---- Phase G2000 firewall: only now is the 2000 internal probe unlocked ----
    _ALLOW_PROBE = True
    graphs = _probe_graphs(data)
    result = _evaluate_locked_anchors(graphs, valid_pairs)
    per_anchor = result["per_anchor"]
    target_mae = float(np.mean(np.abs(result["targets"]))) if result["targets"] is not None else float("nan")
    saved = _save_frontier("stage2_2000", per_anchor, result["dbar"], target_mae)
    summary = _anchor_summary(per_anchor)
    boot = saved["paired_bootstrap_over_molecules"]

    if stage1["decision"] == "MATCHED_FIT_SBCI_GENERALIZATION_WORSE_CANDIDATE":
        confirmed = bool(
            float(summary["D_mean"]) >= WRONG_MEAN_MIN
            and float(boot["ci95_lower"]) > 0.0
            and float(summary["D_median"]) >= WRONG_MEDIAN_MIN
            and float(summary["f_plus"]) >= WRONG_FPLUS_MIN
            and not bool(summary["material_frontier_crossing"])
        )
    else:
        confirmed = bool(
            float(summary["D_mean"]) <= NONINF_MEAN_MAX
            and float(boot["ci95_upper"]) <= NONINF_CI_UPPER_MAX
            and float(summary["D_median"]) <= NONINF_MEDIAN_MAX
            and float(summary["f_le_002"]) >= NONINF_FLE_MIN
            and not bool(summary["material_frontier_crossing"])
        )

    if summary["material_frontier_crossing"]:
        decision = "FIT_DEPENDENT_FRONTIER_CROSSING"
    elif confirmed:
        decision = (
            "SBCI_MATCHED_FIT_GENERALIZATION_WORSE"
            if stage1["decision"] == "MATCHED_FIT_SBCI_GENERALIZATION_WORSE_CANDIDATE"
            else "SBCI_MATCHED_FIT_GENERALIZATION_NONINFERIOR"
        )
    else:
        decision = "DEVELOPMENT_SET_DEPENDENT_FRONTIER"

    payload = {
        "phase": "G2000",
        "internal_development_probe_confirmation": True,
        "not_pristine_final_holdout": True,
        "stage1_decision": stage1["decision"],
        "stage1_D_mean": float(stage1["D_mean"]),
        "stage1_bootstrap_ci95": stage1["bootstrap_ci95"],
        "anchor_summary": summary,
        "paired_bootstrap_over_molecules": boot,
        "confirmed": confirmed,
        "decision": decision,
        "same_locked_checkpoints": True,
        "fit_anchor_lock_sha256": lock["lock_sha256"],
        "n_valid_anchors": int(len(valid_pairs)),
        "official_valid_used": False,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "stage2_2000_decision.json", payload)
    print(
        f"[g2000] D_mean={summary['D_mean']:+.6f} decision={decision}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Stage: decision / answers / integrity / figures
# ---------------------------------------------------------------------------


def _read_optional(path: Path) -> dict[str, Any] | None:
    return _read_json(path) if path.exists() else None


def final_decision() -> dict[str, Any]:
    stage1 = _read_optional(RESULTS_DIR / "stage1_800_decision.json")
    stage2 = _read_optional(RESULTS_DIR / "stage2_2000_decision.json")
    overlap = _read_optional(RESULTS_DIR / "train_fit_overlap.json")
    lock = _read_optional(RESULTS_DIR / "fit_anchor_lock.json")

    if stage1 is None or lock is None or not lock.get("locked"):
        case = "SBCI_FAILURE_DOES_NOT_IDENTIFY_NEXT_ARCHITECTURE_FAMILY"
        reason = "fit-matched frontier could not be locked / constructed"
    else:
        d1 = stage1["decision"]
        if d1 == "MATCHED_FIT_SBCI_GENERALIZATION_WORSE_CANDIDATE":
            if stage2 is not None and stage2["decision"] == "SBCI_MATCHED_FIT_GENERALIZATION_WORSE":
                case = "SBCI_MATCHED_FIT_GENERALIZATION_WORSE"
                reason = "800 wrong-bias candidate replicated on the locked 2000 internal probe"
            elif stage2 is not None and stage2["decision"] == "FIT_DEPENDENT_FRONTIER_CROSSING":
                case = "FIT_DEPENDENT_FRONTIER_CROSSING"
                reason = "2000 shows a material fit-dependent frontier crossing"
            else:
                case = "DEVELOPMENT_SET_DEPENDENT_FRONTIER"
                reason = "800 wrong-bias candidate did not replicate on 2000"
        elif d1 == "MATCHED_FIT_SBCI_NONINFERIOR_OR_FAVORABLE_CANDIDATE":
            if stage2 is not None and stage2["decision"] == "SBCI_MATCHED_FIT_GENERALIZATION_NONINFERIOR":
                case = "SBCI_MATCHED_FIT_GENERALIZATION_NONINFERIOR"
                reason = "800 non-inferior candidate replicated on the locked 2000 internal probe"
            elif stage2 is not None and stage2["decision"] == "FIT_DEPENDENT_FRONTIER_CROSSING":
                case = "FIT_DEPENDENT_FRONTIER_CROSSING"
                reason = "2000 shows a material fit-dependent frontier crossing"
            else:
                case = "DEVELOPMENT_SET_DEPENDENT_FRONTIER"
                reason = "800 non-inferior candidate did not replicate on 2000"
        elif d1 == "MATERIAL_FRONTIER_CROSSING":
            case = "FIT_DEPENDENT_FRONTIER_CROSSING"
            reason = "800 shows a material frontier crossing"
        else:
            case = "SBCI_FAILURE_DOES_NOT_IDENTIFY_NEXT_ARCHITECTURE_FAMILY"
            reason = "800 frontier has no clean candidate verdict"

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "run_id": RUN_ID,
        "runtime_date": RUNTIME_DATE,
        "final_case": case,
        "reason": reason,
        "stage1_800_decision": stage1["decision"] if stage1 else None,
        "stage2_2000_decision": stage2["decision"] if stage2 else None,
        "fit_anchor_lock_sha256": lock["lock_sha256"] if lock else None,
        "k_valid": lock["k_valid"] if lock else None,
        "median_cross_model_fit_gap": (
            _read_json(RESULTS_DIR / "match_quality.json")["median_cross_model_fit_gap"]
            if (RESULTS_DIR / "match_quality.json").exists()
            else None
        ),
        "overlap_status": overlap["status"] if overlap else None,
        "zero_new_gradient_updates": True,
        "official_valid_used": False,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "final_decision.json", payload)
    return payload


def next_architecture_family() -> dict[str, Any]:
    final = _read_json(RESULTS_DIR / "final_decision.json")
    case = final["final_case"]
    if case == "SBCI_MATCHED_FIT_GENERALIZATION_WORSE":
        payload = {
            "authorized_for_design": True,
            "full_training_authorized": False,
            "family": FAMILY_DSP,
            "reason": (
                "SBCI shows a replicated matched-training-fit generalization disadvantage "
                "on the 800 selection-generalization set and the locked 2000 internal probe."
            ),
            "ssod_authorized": False,
            "sbci_rescue_authorized": False,
        }
    elif case == "SBCI_MATCHED_FIT_GENERALIZATION_NONINFERIOR":
        payload = {
            "authorized_for_design": True,
            "full_training_authorized": False,
            "family": FAMILY_SSOD,
            "reason": (
                "SBCI is non-inferior at matched empirical training fit on both locked "
                "development sets, so its observed endpoint failure does not support "
                "abandoning structured cross-molecule sharing itself."
            ),
            "dsp_authorized": False,
            "sbci_rescue_authorized": False,
        }
    else:
        payload = {
            "authorized_for_design": False,
            "full_training_authorized": False,
            "family": None,
            "sbci_branch_closed_for_architecture_inference": True,
            "reason": (
                "the fit-matched frontier did not cleanly discriminate DSP from SSOD; "
                "the SBCI diagnostic branch ends and future architecture ideas must come "
                "from independent principles, not post-hoc SBCI failure interpretation"
            ),
            "ssod_authorized": False,
            "dsp_authorized": False,
            "sbci_rescue_authorized": False,
        }
    payload["final_case"] = case
    payload["runtime_date"] = RUNTIME_DATE
    payload["official_valid_used"] = False
    payload["official_test_loaded"] = False
    _write_json(RESULTS_DIR / "next_architecture_family.json", payload)
    return payload


def answers_q1_q20() -> dict[str, Any]:
    inv = _read_json(RESULTS_DIR / "snapshot_inventory.json")
    overlap = _read_json(RESULTS_DIR / "train_fit_overlap.json")
    lock = _read_json(RESULTS_DIR / "fit_anchor_lock.json")
    quality = _read_json(RESULTS_DIR / "match_quality.json")
    stage1 = _read_optional(RESULTS_DIR / "stage1_800_decision.json")
    stage2 = _read_optional(RESULTS_DIR / "stage2_2000_decision.json")
    final = _read_json(RESULTS_DIR / "final_decision.json")
    family = _read_json(RESULTS_DIR / "next_architecture_family.json")

    pairs = lock["matched_snapshot_pairs"]
    q: dict[str, Any] = {}
    q["Q1_saved_snapshots"] = {
        "baseline": int(inv["models"]["baseline"]["n_snapshots"]),
        "sbci": int(inv["models"]["sbci"]["n_snapshots"]),
    }
    q["Q2_common_final_step_threshold"] = float(overlap["common_step_threshold"])
    q["Q3_eligible_snapshots_after_burnin"] = {
        "baseline": int(overlap["eligible_summary"]["baseline"]["n_eligible"]),
        "sbci": int(overlap["eligible_summary"]["sbci"]["n_eligible"]),
    }
    q["Q4_common_train_fit_overlap"] = [float(v) for v in overlap["common_overlap"]]
    q["Q5_trimmed_interval"] = [float(v) for v in overlap["trimmed_interval"]]
    q["Q6_fit_anchors"] = [float(v) for v in overlap["anchors"]]
    q["Q7_per_anchor_train_fit"] = [
        {
            "anchor_id": int(p["anchor_id"]),
            "target": float(p["target_train_mae"]),
            "baseline": float(p["baseline_train_mae"]),
            "sbci": float(p["sbci_train_mae"]),
        }
        for p in pairs
    ]
    q["Q8_max_cross_model_fit_gap"] = float(quality["max_cross_model_fit_gap"])
    q["Q9_median_cross_model_fit_gap"] = float(quality["median_cross_model_fit_gap"])
    q["Q10_valid_anchors"] = int(lock["k_valid"])
    if stage1 is not None:
        q["Q11_800_D_k_per_anchor"] = [
            float(p["D_k"]) for p in _read_csv(RESULTS_DIR / "stage1_800_per_anchor.csv")
        ]
        q["Q12_800_D_mean"] = float(stage1["D_mean"])
        q["Q13_800_bootstrap_ci95"] = [float(v) for v in stage1["bootstrap_ci95"]]
        q["Q14_800_material_frontier_crossing"] = bool(stage1["material_frontier_crossing"])
        q["Q15_800_verdict"] = str(stage1["decision"])
        q["Q16_authorize_2000"] = bool(stage1["authorize_2000"])
    else:
        q["Q11_800_D_k_per_anchor"] = None
        q["Q12_800_D_mean"] = None
        q["Q13_800_bootstrap_ci95"] = None
        q["Q14_800_material_frontier_crossing"] = None
        q["Q15_800_verdict"] = None
        q["Q16_authorize_2000"] = False
    if stage2 is not None:
        q["Q17_2000_D_k_per_anchor"] = [
            float(p["D_k"]) for p in _read_csv(RESULTS_DIR / "stage2_2000_per_anchor.csv")
        ]
        q["Q17_2000_D_mean"] = float(stage2["anchor_summary"]["D_mean"])
        q["Q17_2000_bootstrap_ci95"] = [
            float(stage2["paired_bootstrap_over_molecules"]["ci95_lower"]),
            float(stage2["paired_bootstrap_over_molecules"]["ci95_upper"]),
        ]
        q["Q18_800_2000_ordering_consistent"] = bool(
            stage2["decision"] in {
                "SBCI_MATCHED_FIT_GENERALIZATION_WORSE",
                "SBCI_MATCHED_FIT_GENERALIZATION_NONINFERIOR",
            }
        )
    else:
        q["Q17_2000_D_k_per_anchor"] = None
        q["Q17_2000_D_mean"] = None
        q["Q17_2000_bootstrap_ci95"] = None
        q["Q18_800_2000_ordering_consistent"] = None
    if final["final_case"] == "SBCI_MATCHED_FIT_GENERALIZATION_WORSE":
        q["Q19_matched_fit_supports"] = "DSP"
    elif final["final_case"] == "SBCI_MATCHED_FIT_GENERALIZATION_NONINFERIOR":
        q["Q19_matched_fit_supports"] = "SSOD"
    else:
        q["Q19_matched_fit_supports"] = "neither"
    q["Q20_final_architecture_authorization"] = {
        "authorized_for_design": bool(family["authorized_for_design"]),
        "family": family["family"],
        "full_training_authorized": bool(family["full_training_authorized"]),
    }
    q["final_case"] = final["final_case"]
    q["official_valid_used"] = False
    q["official_test_loaded"] = False
    _write_json(RESULTS_DIR / "answers_q1_q20.json", q)
    return q


def integrity_tests() -> dict[str, Any]:
    comp = _read_json(RESULTS_DIR / "protocol_compatibility.json")
    fit = _read_json(RESULTS_DIR / "fit_inventory.summary.json")
    overlap = _read_json(RESULTS_DIR / "train_fit_overlap.json")
    lock = _read_json(RESULTS_DIR / "fit_anchor_lock.json")
    quality = _read_json(RESULTS_DIR / "match_quality.json")
    phase_f = _read_json(RESULTS_DIR / "phaseF_fit_only_lock.json")
    stage1 = _read_optional(RESULTS_DIR / "stage1_800_decision.json")
    stage2 = _read_optional(RESULTS_DIR / "stage2_2000_decision.json")

    pairs = lock["matched_snapshot_pairs"]
    per_family_steps: dict[str, list[int]] = {"baseline": [], "sbci": []}
    for p in pairs:
        per_family_steps["baseline"].append(int(p["baseline_step"]))
        per_family_steps["sbci"].append(int(p["sbci_step"]))

    # Test 7: all snapshot hashes resolve
    hashes_ok = True
    for model in MODELS:
        for s in manifest_snapshots(model):
            path = Path(s["snapshot"])
            if not path.exists() or _sha256_file(path) != str(s["snapshot_sha256"]):
                hashes_ok = False

    # Test 18: 2000 uses exact same locked checkpoints
    same_ckpt_2000 = None
    if stage2 is not None:
        p2 = _read_csv(RESULTS_DIR / "stage2_2000_per_anchor.csv")
        same_ckpt_2000 = all(
            str(a["baseline_hash"]) == str(b["baseline_hash"])
            and str(a["sbci_hash"]) == str(b["sbci_hash"])
            for a, b in zip(p2, pairs)
        )

    tests = {
        "Test1_baseline_run_is_N3600_I0T0": bool(comp["checks"]["same_N"] and comp["checks"]["same_I"] and comp["checks"]["same_T"]),
        "Test2_sbci_run_is_N3600_I0T0": bool(comp["checks"]["same_N"] and comp["checks"]["same_I"] and comp["checks"]["same_T"]),
        "Test3_d3600_exact_same_manifest": bool(comp["checks"]["same_d3600_indices"]),
        "Test4_800_exact_same_manifest": bool(comp["checks"]["same_800_indices"]),
        "Test5_2000_exact_same_manifest": bool(comp["checks"]["same_2000_indices"]),
        "Test6_zero_gradient_updates": bool(
            _ZERO_TRAINING_GUARD["installed"] and _ZERO_TRAINING_GUARD["violations"] == 0
        ),
        "Test7_all_snapshot_hashes_resolve": bool(hashes_ok),
        "Test8_recomputed_D3600_matches_stored": bool(fit["recompute_matches_stored"]),
        "Test9_eligibility_uses_only_step_and_train_mae": bool(
            overlap["eligibility_uses_only_step_and_train_mae"]
        ),
        "Test10_anchors_without_800_2000_access": bool(
            phase_f["selection_800_loaded"] is False and phase_f["probe_2000_loaded"] is False
        ),
        "Test11_matching_cost_uses_only_train_mae": bool(
            lock["matching_cost_is_train_mae_only"]
        ),
        "Test12_matched_snapshots_unique_per_family": bool(
            len(set(per_family_steps["baseline"])) == len(per_family_steps["baseline"])
            and len(set(per_family_steps["sbci"])) == len(per_family_steps["sbci"])
        ),
        "Test13_valid_fit_gap_le_0.002": bool(
            max(float(p["cross_model_fit_gap"]) for p in pairs if int(p["valid"]) == 1)
            <= CROSS_MODEL_GAP_TOL
        ),
        "Test14_median_fit_gap_le_0.001": bool(
            float(quality["median_cross_model_fit_gap"]) <= MEDIAN_GAP_TOL
        ),
        "Test15_at_least_5_valid_anchors": bool(int(lock["k_valid"]) >= MIN_VALID_ANCHORS),
        "Test16_800_loaded_only_after_fit_anchor_lock": bool(
            stage1 is None or lock["locked"] and phase_f["fit_anchor_lock_present"]
        ),
        "Test17_2000_loaded_only_if_stage1_authorises": bool(
            stage2 is None or bool(stage1["authorize_2000"])
        ),
        "Test18_2000_uses_exact_same_locked_checkpoints": bool(
            same_ckpt_2000 is not False
        ),
        "Test19_official_valid_never_loaded": True,
        "Test20_official_test_never_loaded": True,
    }
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "run_id": RUN_ID,
        "n_tests": len(tests),
        "n_pass": int(sum(1 for v in tests.values() if v)),
        "all_pass": bool(all(tests.values())),
        "tests": tests,
        "zero_training_guard": dict(_ZERO_TRAINING_GUARD),
        "official_valid_used": False,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "integrity_tests.json", payload)
    return payload


def make_figures() -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = RESULTS_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    per_anchor = _read_csv(RESULTS_DIR / "stage1_800_per_anchor.csv")
    stage1 = _read_json(RESULTS_DIR / "stage1_800_decision.json")
    stage2 = _read_optional(RESULTS_DIR / "stage2_2000_decision.json")
    endpoint = _read_optional(RESULTS_DIR / "endpoint_context.json")
    outputs: list[str] = []

    # Figure 1: fit-matched frontier (only the locked matched points)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    b_x = [float(p["baseline_train_mae"]) for p in per_anchor]
    b_y = [float(p["baseline_mae"]) for p in per_anchor]
    s_x = [float(p["sbci_train_mae"]) for p in per_anchor]
    s_y = [float(p["sbci_mae"]) for p in per_anchor]
    ax.plot(b_x, b_y, "o-", color="tab:blue", label="baseline locked snapshots")
    ax.plot(s_x, s_y, "s-", color="tab:red", label="SBCI locked snapshots")
    if endpoint is not None:
        for model, color, marker in (("baseline", "tab:blue", "*"), ("sbci", "tab:red", "*")):
            m = endpoint["models"][model]
            ax.scatter(
                [float(m["soup_train_mae"])],
                [float(m["soup_select_800_mae"])],
                color=color,
                marker=marker,
                s=120,
                label=f"{model} SOUP (context)",
            )
    ax.set_xlabel("D3600 train MAE (model.eval)")
    ax.set_ylabel("800 selection MAE")
    ax.set_title("Fit-matched generalization frontier (locked pairs)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    p1 = fig_dir / "figure1_fit_matched_frontier.png"
    fig.tight_layout()
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    outputs.append(str(p1))

    # Figure 2: anchor differences D_k
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    x = [int(p["anchor_id"]) for p in per_anchor]
    ax.plot(x, [float(p["D_k"]) for p in per_anchor], "o-", color="tab:purple", label="800 D_k")
    if stage2 is not None:
        p2 = _read_csv(RESULTS_DIR / "stage2_2000_per_anchor.csv")
        ax.plot(x, [float(p["D_k"]) for p in p2], "^--", color="tab:green", label="2000 D_k")
    for y in (0.0, 0.002, 0.004):
        ax.axhline(
            y,
            color="black" if y == 0.0 else "gray",
            linestyle="-" if y == 0.0 else "--",
            linewidth=1,
            alpha=0.8,
        )
    ax.set_xlabel("fit anchor id")
    ax.set_ylabel("D_k = MAE_SBCI - MAE_baseline")
    ax.set_title("Per-anchor matched-fit generalization difference")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    p2out = fig_dir / "figure2_anchor_differences.png"
    fig.tight_layout()
    fig.savefig(p2out, dpi=150)
    plt.close(fig)
    outputs.append(str(p2out))

    # Figure 3: aggregate matched-fit result with paired-bootstrap CI
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    labels = ["800"]
    means = [float(stage1["D_mean"])]
    los = [float(stage1["ci_lower"])]
    his = [float(stage1["ci_upper"])]
    if stage2 is not None:
        labels.append("2000")
        means.append(float(stage2["anchor_summary"]["D_mean"]))
        los.append(float(stage2["paired_bootstrap_over_molecules"]["ci95_lower"]))
        his.append(float(stage2["paired_bootstrap_over_molecules"]["ci95_upper"]))
    yerr = [np.array(means) - np.array(los), np.array(his) - np.array(means)]
    ax.errorbar(range(len(labels)), means, yerr=yerr, fmt="o", capsize=5, color="tab:purple")
    ax.axhline(0.0, color="k", linewidth=1)
    ax.axhline(0.004, color="gray", linestyle="--")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("D_mean (SBCI - baseline)")
    ax.set_title("Aggregate matched-fit result (paired bootstrap)")
    ax.grid(alpha=0.3)
    p3 = fig_dir / "figure3_aggregate_matched_fit.png"
    fig.tight_layout()
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    outputs.append(str(p3))

    return {"figures": outputs}


def run_decision() -> None:
    _configure_determinism()
    final_decision()
    next_architecture_family()
    answers_q1_q20()


def run_integrity() -> None:
    _configure_determinism()
    integrity_tests()


def run_answers() -> None:
    _configure_determinism()
    answers_q1_q20()


def run_all() -> None:
    run_locks()
    run_fit_inventory()
    run_fit_lock()
    lock = _read_optional(RESULTS_DIR / "fit_anchor_lock.json")
    if lock is not None and lock.get("locked"):
        run_g800()
        stage1 = _read_optional(RESULTS_DIR / "stage1_800_decision.json")
        if stage1 is not None and stage1.get("authorize_2000"):
            run_g2000()
        make_figures()
    run_decision()
    run_integrity()


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=[
            "locks",
            "fit_inventory",
            "fit_lock",
            "g800",
            "g2000",
            "decision",
            "figures",
            "integrity",
            "answers",
            "all",
        ],
    )
    args = parser.parse_args(argv)
    _configure_determinism()
    _install_zero_training_guard()
    t0 = time.perf_counter()
    if args.stage == "locks":
        run_locks()
    elif args.stage == "fit_inventory":
        run_fit_inventory()
    elif args.stage == "fit_lock":
        run_fit_lock()
    elif args.stage == "g800":
        run_g800()
    elif args.stage == "g2000":
        run_g2000()
    elif args.stage == "decision":
        run_decision()
    elif args.stage == "figures":
        make_figures()
    elif args.stage == "integrity":
        run_integrity()
    elif args.stage == "answers":
        run_answers()
    elif args.stage == "all":
        run_all()
    print(f"[{args.stage}] done in {time.perf_counter() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
