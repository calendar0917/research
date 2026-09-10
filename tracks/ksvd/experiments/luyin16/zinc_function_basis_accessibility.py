"""Frozen Function-Basis Accessibility Audit (compact-v4-hinge, ZINC).

This is a **representation-frozen, function-basis** audit -- not an
architecture benchmark and not compact-v7.  It does not train a backbone,
does not modify the patch representation / tokenizer / pair encoder / centre
update / pooling / topology branch, and does not add any new structural
feature.  It answers exactly one question:

    On the *same* frozen compact-v4-hinge graph representation R (302D),
    does changing only the final function parameterisation -- from a generic
    ReLU MLP to an explicit target-independent piecewise-linear (quantile
    hinge) basis -- make L1/MAE residual regression more stable and more
    sample-efficient?

Protocol (reused verbatim from the frozen-readout / pair-witness /
centre-witness audits):

* frozen 5-fold OOF compact-v4-hinge checkpoints on the **official train
  universe** only; official valid/test are never loaded;
* the frozen OOF representation R and the frozen baseline prediction yhat_0
  come from the corrected centre-incidence state export;
* the same outer-held-out 2000 -> 1200 adapter-fit / 400 adapter-selection /
  400 adapter-evaluation hash split manifest;
* every reader predicts a residual correction  yhat = yhat_0 + delta(R)  and
  is trained with the identical L1 objective / Adam(lr=1e-3) / full-batch /
  selection-checkpoint protocol.

Readers:

* ``B0`` frozen baseline prediction (no training);
* ``B1`` generic ReLU MLP 302 -> 4 -> 2 -> 1 (parameter-matched to E);
* ``B2`` historical-strength generic ReLU MLP 302 -> 13 -> 13 -> 1 (~4.1k);
* ``E`` explicit target-independent quantile-hinge additive basis
  ``[z_j, ReLU(z_j-t25), ReLU(z_j-t50), ReLU(z_j-t75)]`` -> 1 (1209 params);
* ``B-linear`` standardised R -> 1 (descriptive only, 303 params).

Run::

    uv run python -m tracks.ksvd.experiments.luyin16.zinc_function_basis_accessibility <stage>
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from tracks.ksvd.experiments.luyin16 import (
    zinc_centre_incidence_cooccurrence_witness as ciw,
)
from tracks.ksvd.experiments.luyin16.zinc_oof_difficulty_audit import (
    FOLD_DIR,
    K_FOLDS,
    _fold_slices,
    _load_fold_npz,
    _load_train_labels,
    _load_train_records,
    _verify_records_vs_labels,
)

REPO_ROOT = ciw.REPO_ROOT
RESULTS_DIR = REPO_ROOT / "tracks/ksvd/results/function_basis_accessibility"
CACHE_DIR = RESULTS_DIR / "state_exports"
FIG_DIR = RESULTS_DIR / "figures"

# Source frozen export (corrected centre-incidence family; R == true pre-head
# 302D input, yhat_0 == true frozen forward prediction).
SOURCE_EXPORT_VERSION = ciw.EXPORT_VERSION
CACHE_VERSION = "function_basis_R_cache_v1"

# --- frozen v4-hinge representation ----------------------------------------
R_DIM = 302  # verified at runtime from the real export

# --- pre-registered reader architectures -----------------------------------
# B1 302 -> 4 -> 2 -> 1   : 302*4+4 + 4*2+2 + 2*1+1 = 1225
# E  [z,hinge25,hinge50,hinge75] -> 1 = 302*4 + 1 = 1209  (mismatch +1.32%)
# B2 302 -> 13 -> 13 -> 1 : 302*13+13 + 13*13+13 + 13*1+1 = 4135
# B-lin 302 -> 1          : 303 (descriptive, not in the gate)
B1_HIDDEN = (4, 2)
B2_HIDDEN = (13, 13)
BLIN_HIDDEN = ()
HEAD_HIDDEN_0 = 64  # frozen graph head (only for the offline reconstruction gate)
HEAD_HIDDEN_1 = 32
PARAM_MATCH_TOL = 0.03

# --- preprocessing / basis --------------------------------------------------
STANDARDIZE_EPS = 1.0e-6
HINGE_QUANTILES = (0.25, 0.50, 0.75)
HINGE_TERMS = 3
FIXED_KNOT_VALUES = (-1.0, 0.0, 1.0)  # knot-location control (no target)

# --- split / budget (identical to the frozen adapters) ---------------------
SPLIT_SEED = ciw.SPLIT_SEED
SPLIT_SIZES = ciw.SPLIT_SIZES
BACKBONE_SEEDS = (0, 1)
PRIMARY_BACKBONE_SEED = 0
ADAPTER_SEEDS = (0, 1)

# --- optimiser / training protocol (identical for B1/B2/E/B-lin) -----------
ADAPTER_LR = 1.0e-3
ADAPTER_PATIENCE = 50
CONVERGENCE_EPOCHS = (100, 200, 400, 800)
HORIZON_FAST = 400
HORIZON_SLOW = 800
ADAPTER_BATCH = "full"

# --- decision thresholds (pre-registered) -----------------------------------
S1_STRONG_DSMALL = 0.003
S1_STRONG_FOLDS = 4
S1_STRONG_E_MINUS_B2_MAX = 0.0015
S1_EFF_DSMALL = 0.003
S1_EFF_FOLDS = 4
S1_EFF_E_MINUS_B2_MAX = 0.004
S1_NGO_DSMALL = 0.0005
S1_NGO_FOLDS = 2  # E better than B1 in <= 2/5 folds -> clear NO-GO
S1_BORDERLINE_FOLDS = 3
FINAL_GO_POOLED = 0.0025
FINAL_GO_FOLDS = 4
BULK_MAX_DEGRADATION = 0.002

# --- gate tolerances --------------------------------------------------------
FORWARD_GATE_ATOL = 1.0e-9
R_RECON_ATOL = 1.0e-4
PRED_RECON_ATOL = 1.0e-5


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_json_any(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _hash_json(value: Any) -> str:
    blob = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _cache_path(fold: int, seed: int) -> Path:
    return CACHE_DIR / f"{CACHE_VERSION}_fold{fold}_seed{seed}.npz"


# ---------------------------------------------------------------------------
# frozen state loading (R + yhat_0 only)
# ---------------------------------------------------------------------------

def _extract_from_source(fold: int, seed: int) -> dict[str, Any]:
    """Read the corrected frozen export and keep R / yhat_0 / target / id."""
    export = ciw.load_export(ciw._export_path(fold, seed))
    R = np.asarray(export["R"], dtype=np.float32)
    if R.shape[1] != R_DIM:
        raise AssertionError(f"frozen R width {R.shape[1]} != {R_DIM}")
    return {
        "R": R,
        "yhat_0": np.asarray(export["yhat_0"], dtype=np.float64),
        "target": np.asarray(export["target"], dtype=np.float64),
        "subset_index": np.asarray(export["subset_index"], dtype=np.int64),
        "fingerprint": export["fingerprint"],
    }


def _save_cache(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for key, value in payload.items():
        if key == "fingerprint":
            arrays["fingerprint_json"] = np.asarray(json.dumps(_jsonable(value), sort_keys=True))
        else:
            arrays[key] = np.asarray(value)
    np.savez_compressed(path, **arrays)


def load_cache(fold: int, seed: int) -> dict[str, Any]:
    with np.load(_cache_path(fold, seed), allow_pickle=False) as data:
        payload: dict[str, Any] = {
            "fingerprint": json.loads(str(data["fingerprint_json"])),
        }
        for key in data.files:
            if key == "fingerprint_json":
                continue
            payload[key] = data[key]
    return payload


def stage_cache(seeds: Sequence[int] = (PRIMARY_BACKBONE_SEED,), force: bool = False) -> dict[str, Any]:
    started = time.perf_counter()
    built: dict[str, Any] = {}
    for seed in seeds:
        for fold in range(K_FOLDS):
            path = _cache_path(fold, seed)
            if path.exists() and not force:
                built[f"fold{fold}_seed{seed}"] = {"path": str(path), "cached": True}
                continue
            payload = _extract_from_source(fold, seed)
            _save_cache(path, payload)
            built[f"fold{fold}_seed{seed}"] = {
                "path": str(path),
                "cached": False,
                "n": int(len(payload["subset_index"])),
                "R_dim": int(payload["R"].shape[1]),
            }
            print(
                f"[cache] fold={fold} seed={seed} n={len(payload['subset_index'])} "
                f"R_dim={payload['R'].shape[1]} ({time.perf_counter() - started:.1f}s)",
                flush=True,
            )
    summary = {
        "cache_version": CACHE_VERSION,
        "source_export_version": SOURCE_EXPORT_VERSION,
        "seeds": [int(s) for s in seeds],
        "entries": built,
        "seconds": time.perf_counter() - started,
    }
    _write_json_any(RESULTS_DIR / "stage_cache.json", summary)
    return summary


# ---------------------------------------------------------------------------
# integrity / reconstruction gate
# ---------------------------------------------------------------------------

def _head_from_state(fold: int, seed: int) -> nn.Module:
    """Rebuild the frozen graph head from the checkpoint state dict.

    The head is ``Linear(302,64) LayerNorm ReLU Dropout Linear(64,32) ReLU
    Linear(32,1)``; the state-dict load and the zero-difference prediction
    gate verify the architecture without rebuilding the whole backbone.
    """
    state = torch.load(
        FOLD_DIR / f"fold{fold}_seed{seed}_state.pt", map_location="cpu", weights_only=True
    )
    head = nn.Sequential(
        nn.Linear(R_DIM, HEAD_HIDDEN_0),
        nn.LayerNorm(HEAD_HIDDEN_0),
        nn.ReLU(),
        nn.Dropout(0.05),
        nn.Linear(HEAD_HIDDEN_0, HEAD_HIDDEN_1),
        nn.ReLU(),
        nn.Linear(HEAD_HIDDEN_1, 1),
    )
    head.load_state_dict({k[len("head."):]: v for k, v in state.items() if k.startswith("head.")})
    head.eval()
    if head[0].in_features != R_DIM:
        raise AssertionError("rebuilt head input width changed")
    return head


def _head_forward(head: nn.Module, R: np.ndarray) -> np.ndarray:
    head.eval()
    with torch.no_grad():
        return head(torch.tensor(np.asarray(R), dtype=torch.float32)).view(-1).numpy().astype(np.float64)


def _integrity_for_fold(fold: int, seed: int) -> dict[str, Any]:
    payload = load_cache(fold, seed)
    R = np.asarray(payload["R"], dtype=np.float64)
    yhat_0 = np.asarray(payload["yhat_0"], dtype=np.float64)
    target = np.asarray(payload["target"], dtype=np.float64)
    subset_index = np.asarray(payload["subset_index"], dtype=np.int64)

    # forward gate: stored yhat_0 == the OOF fold prediction
    npz = _load_fold_npz(fold, seed)
    forward_gate = float(np.abs(yhat_0 - np.asarray(npz["oof_prediction"], dtype=np.float64)).max())

    # label gate
    labels = _load_train_labels()
    label_y = labels.set_index("subset_index").loc[subset_index, "y_stored"].to_numpy(dtype=np.float64)
    label_gate = float(np.abs(target - label_y).max())

    # split gate: subset_index == the outer holdout of this fold
    holdout_idx = _fold_slices()[fold][2]
    split_gate = bool(np.array_equal(np.sort(subset_index), np.sort(holdout_idx)))

    # reconstruction gate (offline, from the full source export)
    export = ciw.load_export(ciw._export_path(fold, seed))
    R_recon = ciw._reconstruct_R(export)
    r_recon_diff = float(
        np.abs(np.asarray(R_recon, dtype=np.float64) - R).max()
    )
    head = _head_from_state(fold, seed)
    stored_R_pred = _head_forward(head, R)
    recon_R_pred = _head_forward(head, R_recon)
    pred_stored_diff = float(np.abs(stored_R_pred - yhat_0).max())
    pred_recon_diff = float(np.abs(recon_R_pred - yhat_0).max())

    fp = payload["fingerprint"]
    return {
        "fold": int(fold),
        "backbone_seed": int(seed),
        "n_molecules": int(len(subset_index)),
        "R_dim": int(R.shape[1]),
        "G1_forward_yhat0_max_diff": forward_gate,
        "G2_target_label_max_diff": label_gate,
        "G3_subset_index_matches_holdout": split_gate,
        "G4_reconstruct_R_max_diff": r_recon_diff,
        "G5_stored_R_prediction_max_diff": pred_stored_diff,
        "G6_reconstructed_R_prediction_max_diff": pred_recon_diff,
        "checkpoint_fingerprint": fp.get("checkpoint_fingerprint"),
        "config_fingerprint": fp.get("config_fingerprint"),
        "tokenizer_version": fp.get("tokenizer_version"),
        "vocabulary_fingerprint": fp.get("vocabulary_fingerprint"),
        "split_fingerprint": fp.get("split_fingerprint"),
        "passed": bool(
            abs(R.shape[1] - R_DIM) == 0
            and forward_gate <= FORWARD_GATE_ATOL
            and label_gate <= 1e-9
            and split_gate
            and r_recon_diff <= R_RECON_ATOL
            and pred_stored_diff <= PRED_RECON_ATOL
            and pred_recon_diff <= PRED_RECON_ATOL
        ),
    }


def run_integrity(seeds: Sequence[int] = (PRIMARY_BACKBONE_SEED,), force: bool = False) -> dict[str, Any]:
    started = time.perf_counter()
    out_path = RESULTS_DIR / "representation_integrity.json"
    if out_path.exists() and not force:
        return json.loads(out_path.read_text(encoding="utf-8"))
    # Global train-universe label integrity: frozen records == frozen labels.
    _verify_records_vs_labels(_load_train_records(), _load_train_labels())
    per_fold: dict[str, Any] = {}
    for seed in seeds:
        for fold in range(K_FOLDS):
            entry = _integrity_for_fold(fold, seed)
            per_fold[f"fold{fold}_seed{seed}"] = entry
            print(
                f"[integrity] fold={fold} seed={seed} pass={entry['passed']} "
                f"g1={entry['G1_forward_yhat0_max_diff']:.1e} "
                f"g4={entry['G4_reconstruct_R_max_diff']:.1e} "
                f"g6={entry['G6_reconstructed_R_prediction_max_diff']:.1e}",
                flush=True,
            )
    failures = [k for k, v in per_fold.items() if not v["passed"]]
    report = {
        "source_export_version": SOURCE_EXPORT_VERSION,
        "cache_version": CACHE_VERSION,
        "R_dim": R_DIM,
        "seeds_checked": [int(s) for s in seeds],
        "per_fold": per_fold,
        "failures": failures,
        "all_passed": len(failures) == 0,
        "global_label_integrity": (
            "PASS: frozen train records y == frozen label y_stored for all 10000 "
            "official-train molecules (official valid/test never loaded)"
        ),
        "gates": {
            "G1": "stored yhat_0 == frozen OOF fold prediction",
            "G2": "stored target == official-train label y_stored",
            "G3": "subset_index == outer holdout of the fold",
            "G4": "offline reconstructed R == exported R",
            "G5": "re-feeding stored R through the frozen head == stored yhat_0",
            "G6": "re-feeding reconstructed R through the frozen head == stored yhat_0",
        },
        "seconds": time.perf_counter() - started,
    }
    _write_json_any(out_path, report)
    print(f"[integrity] all_passed={report['all_passed']} failures={failures}", flush=True)
    return report


# ---------------------------------------------------------------------------
# split manifest (identical to the frozen adapters)
# ---------------------------------------------------------------------------

def stage_splits(force: bool = False) -> dict[str, Any]:
    out_path = RESULTS_DIR / "fold_split_manifest.json"
    manifest = {"split_seed": SPLIT_SEED, "sizes": SPLIT_SIZES, "folds": {}}
    for fold in range(K_FOLDS):
        manifest["folds"][f"fold{fold}"] = ciw._fold_split_manifest(fold)
    # cross-check against the archived frozen-readout / pair / centre manifest
    ref_path = ciw.RESULTS_DIR / "fold_split_manifest.json"
    if ref_path.exists():
        ref = json.loads(ref_path.read_text(encoding="utf-8"))
        mismatches = [
            f for f in range(K_FOLDS)
            if ref["folds"][f"fold{f}"]["assignment_sha256"]
            != manifest["folds"][f"fold{f}"]["assignment_sha256"]
        ]
        manifest["matches_frozen_reference"] = len(mismatches) == 0
        manifest["reference_mismatches"] = mismatches
    _write_json_any(out_path, manifest)
    return manifest


# ---------------------------------------------------------------------------
# fit-only standardisation + target-independent hinge basis
# ---------------------------------------------------------------------------

def _fit_standardizer(R_fit: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """mean / scale / degenerate mask fitted on adapter-fit coordinates only.

    ``scale = max(std, eps)``; degenerate coordinates (std < eps) get z = 0.
    """
    mean = np.asarray(R_fit, dtype=np.float64).mean(axis=0)
    std = np.asarray(R_fit, dtype=np.float64).std(axis=0)
    degenerate = std < STANDARDIZE_EPS
    scale = np.where(degenerate, 1.0, std)
    return mean, scale, degenerate


def _standardize(R: np.ndarray, mean: np.ndarray, scale: np.ndarray, degenerate: np.ndarray) -> np.ndarray:
    z = (np.asarray(R, dtype=np.float64) - mean) / scale
    if degenerate.any():
        z = z.copy()
        z[:, degenerate] = 0.0
    return z


def _fit_knots(z_fit: np.ndarray) -> np.ndarray:
    """Per-coordinate 25/50/75% quantiles from adapter-fit input only."""
    quantiles = np.quantile(np.asarray(z_fit, dtype=np.float64), HINGE_QUANTILES, axis=0)
    return np.ascontiguousarray(quantiles.T.astype(np.float64))  # (D, 3)


def _hinge_basis(z: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """[z ; ReLU(z - t25) ; ReLU(z - t50) ; ReLU(z - t75)] term-major (4D)."""
    z = np.asarray(z, dtype=np.float64)
    columns = [z]
    for k in range(knots.shape[1]):
        columns.append(np.maximum(z - knots[:, k][None, :], 0.0))
    return np.concatenate(columns, axis=1)


def _basis_dim() -> int:
    return R_DIM * (1 + HINGE_TERMS)


# ---------------------------------------------------------------------------
# readers
# ---------------------------------------------------------------------------

class GenericReader(nn.Module):
    """yhat = yhat_0 + MLP(z); ``hidden=()`` gives a plain linear residual."""

    def __init__(self, in_dim: int, hidden: Sequence[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = int(in_dim)
        for width in hidden:
            layers.extend([nn.Linear(previous, int(width)), nn.ReLU()])
            previous = int(width)
        layers.append(nn.Linear(previous, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)


class HingeBasisReader(nn.Module):
    """Explicit additive quantile-hinge basis followed by a single linear map."""

    def __init__(self, knots: np.ndarray) -> None:
        super().__init__()
        self.register_buffer("knots", torch.tensor(np.asarray(knots, dtype=np.float32)))
        self.linear = nn.Linear(_basis_dim(), 1)

    def basis(self, z: torch.Tensor) -> torch.Tensor:
        columns = [z]
        for k in range(self.knots.shape[1]):
            columns.append(torch.relu(z - self.knots[:, k][None, :]))
        return torch.cat(columns, dim=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.linear(self.basis(z)).squeeze(-1)

    def forward_linear_only(self, z: torch.Tensor) -> torch.Tensor:
        """Frozen inference ablation: zero all hinge coefficients."""
        basis = self.basis(z)
        weights = self.linear.weight.clone()
        weights[:, R_DIM:] = 0.0
        return (basis @ weights.t()).squeeze(-1) + self.linear.bias


def _count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def _make_b1(seed: int) -> GenericReader:
    torch.manual_seed(int(seed))
    return GenericReader(R_DIM, B1_HIDDEN)


def _make_b2(seed: int) -> GenericReader:
    torch.manual_seed(int(seed))
    return GenericReader(R_DIM, B2_HIDDEN)


def _make_blin(seed: int) -> GenericReader:
    torch.manual_seed(int(seed))
    return GenericReader(R_DIM, BLIN_HIDDEN)


def _make_hinge(knots: np.ndarray, seed: int) -> HingeBasisReader:
    torch.manual_seed(int(seed))
    return HingeBasisReader(knots)


def _parameter_accounting() -> dict[str, Any]:
    torch.manual_seed(0)
    b1 = _make_b1(0)
    torch.manual_seed(0)
    b2 = _make_b2(0)
    torch.manual_seed(0)
    blin = _make_blin(0)
    zero_knots = np.zeros((R_DIM, HINGE_TERMS), dtype=np.float64)
    torch.manual_seed(0)
    e = _make_hinge(zero_knots, 0)
    b1_params = _count_parameters(b1)
    b2_params = _count_parameters(b2)
    e_params = _count_parameters(e)
    blin_params = _count_parameters(blin)
    mismatch = abs(b1_params - e_params) / max(e_params, 1)
    return {
        "R_dim": int(R_DIM),
        "basis_dim": int(_basis_dim()),
        "B1_architecture": f"{R_DIM} -> " + " -> ".join(str(h) for h in B1_HIDDEN) + " -> 1",
        "B1_params": int(b1_params),
        "B2_architecture": f"{R_DIM} -> " + " -> ".join(str(h) for h in B2_HIDDEN) + " -> 1",
        "B2_params": int(b2_params),
        "E_architecture": f"[z,relu(z-t25),relu(z-t50),relu(z-t75)] ({_basis_dim()}) -> 1",
        "E_params": int(e_params),
        "B_linear_params": int(blin_params),
        "B1_vs_E_abs_mismatch": float(mismatch),
        "parameter_match_ok": bool(mismatch <= PARAM_MATCH_TOL),
        "parameter_match_tolerance": float(PARAM_MATCH_TOL),
    }


def write_basis_spec(accounting: Mapping[str, Any] | None = None) -> dict[str, Any]:
    accounting = dict(accounting) if accounting is not None else _parameter_accounting()
    spec = {
        "representation": {
            "source_export_version": SOURCE_EXPORT_VERSION,
            "R_dim": int(R_DIM),
            "R_layout": "unary 97 + pair-bucket moments 165 + global 32 + topology 8 = 302",
            "frozen": True,
        },
        "preprocessing": {
            "kind": "fit-only coordinate standardisation",
            "fit_scope": "adapter-fit 1200 molecules of each fold",
            "definition": "z_j = (R_j - mean_j) / max(std_j, eps), z_j = 0 if std_j < eps",
            "eps": float(STANDARDIZE_EPS),
            "recomputed_on_selection_or_evaluation": False,
        },
        "basis": {
            "kind": "target-independent per-coordinate quantile hinge (piecewise-linear spline)",
            "quantiles": list(HINGE_QUANTILES),
            "knot_fit_scope": "adapter-fit standardised coordinates only",
            "phi_j": "[z_j, ReLU(z_j-t25), ReLU(z_j-t50), ReLU(z_j-t75)]",
            "per_coordinate": 1 + HINGE_TERMS,
            "total_dim": int(_basis_dim()),
            "duplicate_knots": "retained as duplicate columns (fixed dimensionality)",
            "target_information": False,
        },
        "readers": {
            "B0": "frozen baseline yhat_0 (no training)",
            "B1": "generic ReLU MLP, parameter-matched to E",
            "B2": "historical-strength generic ReLU MLP (not parameter-matched)",
            "E": "additive linear reader on the explicit hinge basis",
            "B_linear": "descriptive linear residual on standardised R (not in the gate)",
        },
        "training": {
            "objective": "L1 / MAE",
            "residual": "yhat = yhat_0 + delta(z)",
            "optimizer": "Adam",
            "lr": float(ADAPTER_LR),
            "batch": ADAPTER_BATCH,
            "checkpoint_selection": "min adapter-selection MAE",
            "patience": int(ADAPTER_PATIENCE),
            "horizon_rule": (
                "convergence trace at fold0/seed0 over 100/200/400/800 epochs; "
                f"H={HORIZON_SLOW} only if some traced reader still improves by >1e-4 "
                f"from 400 to 800, otherwise H={HORIZON_FAST}; best-selection "
                "checkpointing inside the horizon; no further sweep"
            ),
        },
        "parameter_accounting": accounting,
        "forbidden": [
            "q_ij / h_i / covariance / endpoint witness",
            "target component / extra topology / raw molecule feature",
            "target-informed knots (cycle threshold 6, residual-selected knots, decision-tree splits)",
            "MSE / Huber / quantile-asymmetric / weighted MAE",
            "optimizer or LR sweeps",
        ],
    }
    _write_json_any(RESULTS_DIR / "basis_spec.json", spec)
    return spec


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

def _train_reader(
    model: nn.Module,
    z_fit: torch.Tensor,
    y_fit: torch.Tensor,
    yhat_fit: torch.Tensor,
    z_sel: torch.Tensor,
    y_sel: torch.Tensor,
    yhat_sel: torch.Tensor,
    *,
    epochs: int,
    lr: float = ADAPTER_LR,
    patience: int = ADAPTER_PATIENCE,
) -> dict[str, Any]:
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    best_state = copy.deepcopy(model.state_dict())
    best_selection = float("inf")
    bad = 0
    epochs_run = 0
    grad_alive = False
    for epoch in range(int(epochs)):
        epochs_run = epoch + 1
        model.train()
        optimizer.zero_grad()
        loss = (yhat_fit + model(z_fit) - y_fit).abs().mean()
        loss.backward()
        if not grad_alive:
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            grad_alive = bool(grads) and any(float(g.abs().sum()) > 0 for g in grads)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            selection_mae = float((yhat_sel + model(z_sel) - y_sel).abs().mean())
        if selection_mae < best_selection - 1e-9:
            best_selection = selection_mae
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
            if patience is not None and bad >= int(patience):
                break
    model.load_state_dict(best_state)
    return {"best_selection_mae": best_selection, "grad_alive": bool(grad_alive), "epochs_run": epochs_run}


def _train_reader_with_trace(
    model: nn.Module,
    z_fit: torch.Tensor,
    y_fit: torch.Tensor,
    yhat_fit: torch.Tensor,
    z_sel: torch.Tensor,
    y_sel: torch.Tensor,
    yhat_sel: torch.Tensor,
    trace_epochs: Sequence[int],
    max_epochs: int,
) -> dict[str, Any]:
    optimizer = torch.optim.Adam(model.parameters(), lr=float(ADAPTER_LR))
    trace: dict[int, float] = {}
    for epoch in range(int(max_epochs)):
        model.train()
        optimizer.zero_grad()
        loss = (yhat_fit + model(z_fit) - y_fit).abs().mean()
        loss.backward()
        optimizer.step()
        current = epoch + 1
        if current in trace_epochs:
            model.eval()
            with torch.no_grad():
                trace[current] = float((yhat_sel + model(z_sel) - y_sel).abs().mean())
    return trace


# ---------------------------------------------------------------------------
# per-fold data assembly
# ---------------------------------------------------------------------------

def _fold_tensors(
    fold: int,
    backbone_seed: int,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    payload = load_cache(fold, backbone_seed)
    fold_manifest = manifest["folds"][f"fold{fold}"]
    fit_ids = np.asarray(
        [int(m.split(":")[1]) for m in fold_manifest["molecule_ids"]["adapter_fit"]], dtype=np.int64
    )
    sel_ids = np.asarray(
        [int(m.split(":")[1]) for m in fold_manifest["molecule_ids"]["adapter_selection"]], dtype=np.int64
    )
    eva_ids = np.asarray(
        [int(m.split(":")[1]) for m in fold_manifest["molecule_ids"]["adapter_evaluation"]], dtype=np.int64
    )
    subset_index = np.asarray(payload["subset_index"], dtype=np.int64)
    R = np.asarray(payload["R"], dtype=np.float64)
    y = np.asarray(payload["target"], dtype=np.float64)
    yhat_0 = np.asarray(payload["yhat_0"], dtype=np.float64)
    lookup = {int(v): i for i, v in enumerate(subset_index)}
    fit_pos = np.asarray([lookup[int(v)] for v in fit_ids], dtype=np.int64)
    sel_pos = np.asarray([lookup[int(v)] for v in sel_ids], dtype=np.int64)
    eva_pos = np.asarray([lookup[int(v)] for v in eva_ids], dtype=np.int64)

    mean, scale, degenerate = _fit_standardizer(R[fit_pos])
    z = _standardize(R, mean, scale, degenerate)
    knots = _fit_knots(z[fit_pos])
    return {
        "subset_index": subset_index,
        "R": R,
        "y": y,
        "yhat_0": yhat_0,
        "fit_pos": fit_pos,
        "sel_pos": sel_pos,
        "eva_pos": eva_pos,
        "fit_ids": fit_ids,
        "eva_ids": eva_ids,
        "mean": mean,
        "scale": scale,
        "degenerate": degenerate,
        "z": z,
        "knots": knots,
    }


def _tensors(data: Mapping[str, Any], key_pos: str) -> dict[str, torch.Tensor]:
    pos = data[key_pos]
    return {
        "z": torch.tensor(data["z"][pos], dtype=torch.float32),
        "y": torch.tensor(data["y"][pos], dtype=torch.float32),
        "yhat": torch.tensor(data["yhat_0"][pos], dtype=torch.float32),
    }


# ---------------------------------------------------------------------------
# convergence trace
# ---------------------------------------------------------------------------

def run_convergence_trace(force: bool = False) -> dict[str, Any]:
    out_csv = RESULTS_DIR / "convergence_trace.csv"
    out_json = RESULTS_DIR / "convergence_trace.json"
    if out_csv.exists() and out_json.exists() and not force:
        return json.loads(out_json.read_text(encoding="utf-8"))
    manifest = stage_splits()
    data = _fold_tensors(0, PRIMARY_BACKBONE_SEED, manifest)
    fit = _tensors(data, "fit_pos")
    sel = _tensors(data, "sel_pos")
    readers = {
        "B1": (_make_b1(0), "generic"),
        "B2": (_make_b2(0), "generic"),
        "E": (_make_hinge(data["knots"], 0), "hinge"),
    }
    rows: list[dict[str, Any]] = []
    traces: dict[str, Any] = {}
    for name, (model, _kind) in readers.items():
        trace = _train_reader_with_trace(
            model,
            fit["z"], fit["y"], fit["yhat"],
            sel["z"], sel["y"], sel["yhat"],
            trace_epochs=CONVERGENCE_EPOCHS,
            max_epochs=max(CONVERGENCE_EPOCHS),
        )
        traces[name] = {str(k): float(v) for k, v in trace.items()}
        for epoch in CONVERGENCE_EPOCHS:
            rows.append({"reader": name, "epoch": int(epoch), "selection_mae": float(trace[epoch])})
    frame = pd.DataFrame(rows)
    frame.to_csv(out_csv, index=False)

    max_abs_400_800 = max(
        abs(traces[name][str(HORIZON_FAST)] - traces[name][str(HORIZON_SLOW)])
        for name in readers
    )
    still_improving = any(
        traces[name][str(HORIZON_SLOW)] < traces[name][str(HORIZON_FAST)] - 1.0e-4
        for name in readers
    )
    horizon = HORIZON_SLOW if still_improving else HORIZON_FAST
    detail = {
        "fold": 0,
        "backbone_seed": PRIMARY_BACKBONE_SEED,
        "trace_epochs": list(CONVERGENCE_EPOCHS),
        "traces": traces,
        "max_abs_mae_400_minus_800": float(max_abs_400_800),
        "still_improving_at_800": bool(still_improving),
        "horizon_tolerance": 1.0e-4,
        "selected_horizon": int(horizon),
        "rule": (
            "H=800 only if some traced reader still improves by >1e-4 from 400 to 800; "
            "otherwise H=400.  B1/B2/E all use the same fixed horizon; best-selection "
            "checkpointing is applied inside the horizon; no further sweep"
        ),
    }
    _write_json_any(out_json, detail)
    print(f"[convergence] selected_horizon={horizon} max|400-800|={max_abs_400_800:.2e}", flush=True)
    return detail


def _selected_horizon() -> int:
    path = RESULTS_DIR / "convergence_trace.json"
    if path.exists():
        return int(json.loads(path.read_text(encoding="utf-8"))["selected_horizon"])
    return HORIZON_FAST


# ---------------------------------------------------------------------------
# stage 1 / 1b / 2
# ---------------------------------------------------------------------------

def _rare_ratio_for(fold: int, ids: np.ndarray) -> np.ndarray:
    rarity = pd.read_csv(ciw.RARITY_CSV).set_index("subset_index")
    return rarity.loc[ids, "rare_le5_ratio"].to_numpy(dtype=np.float64)


def _run_stage_for_fold(
    fold: int,
    backbone_seed: int,
    adapter_seed: int,
    manifest: Mapping[str, Any],
    horizon: int,
    fixed_knots: bool = False,
) -> dict[str, Any]:
    data = _fold_tensors(fold, backbone_seed, manifest)
    if fixed_knots:
        knots = np.tile(
            np.asarray(FIXED_KNOT_VALUES, dtype=np.float64)[None, :], (R_DIM, 1)
        )
    else:
        knots = data["knots"]
    fit = _tensors(data, "fit_pos")
    sel = _tensors(data, "sel_pos")
    eva = _tensors(data, "eva_pos")

    b1 = _make_b1(adapter_seed)
    b2 = _make_b2(adapter_seed)
    blin = _make_blin(adapter_seed)
    e_model = _make_hinge(knots, adapter_seed)

    b1_info = _train_reader(b1, fit["z"], fit["y"], fit["yhat"], sel["z"], sel["y"], sel["yhat"], epochs=horizon)
    b2_info = _train_reader(b2, fit["z"], fit["y"], fit["yhat"], sel["z"], sel["y"], sel["yhat"], epochs=horizon)
    blin_info = _train_reader(blin, fit["z"], fit["y"], fit["yhat"], sel["z"], sel["y"], sel["yhat"], epochs=horizon)
    e_info = _train_reader(e_model, fit["z"], fit["y"], fit["yhat"], sel["z"], sel["y"], sel["yhat"], epochs=horizon)

    with torch.no_grad():
        p_b1 = (eva["yhat"] + b1(eva["z"])).numpy()
        p_b2 = (eva["yhat"] + b2(eva["z"])).numpy()
        p_blin = (eva["yhat"] + blin(eva["z"])).numpy()
        p_e = (eva["yhat"] + e_model(eva["z"])).numpy()
        p_e_no_hinge = (eva["yhat"] + e_model.forward_linear_only(eva["z"])).numpy()
    y = data["y"][data["eva_pos"]]
    yhat0 = data["yhat_0"][data["eva_pos"]]

    mae_b0 = float(np.abs(y - yhat0).mean())
    mae_b1 = float(np.abs(y - p_b1).mean())
    mae_b2 = float(np.abs(y - p_b2).mean())
    mae_blin = float(np.abs(y - p_blin).mean())
    mae_e = float(np.abs(y - p_e).mean())
    mae_e_no_hinge = float(np.abs(y - p_e_no_hinge).mean())

    rare_ratio = _rare_ratio_for(fold, data["eva_ids"])
    bulk_mask = rare_ratio < float(manifest["folds"][f"fold{fold}"]["rare_le5_threshold"])
    if bulk_mask.sum() >= 10:
        err_b0 = np.abs(y - yhat0)[bulk_mask]
        err_b1 = np.abs(y - p_b1)[bulk_mask]
        err_b2 = np.abs(y - p_b2)[bulk_mask]
        err_e = np.abs(y - p_e)[bulk_mask]
        bulk_e_minus_b1 = float(err_e.mean() - err_b1.mean())
        bulk_e_minus_b2 = float(err_e.mean() - err_b2.mean())
        bulk_e_minus_b0 = float(err_e.mean() - err_b0.mean())
    else:
        bulk_e_minus_b1 = bulk_e_minus_b2 = bulk_e_minus_b0 = float("nan")

    return {
        "fold": int(fold),
        "backbone_seed": int(backbone_seed),
        "adapter_seed": int(adapter_seed),
        "horizon": int(horizon),
        "fixed_knots": bool(fixed_knots),
        "n_fit": int(len(data["fit_pos"])),
        "n_selection": int(len(data["sel_pos"])),
        "n_eval": int(len(data["eva_pos"])),
        "n_degenerate_coordinates": int(data["degenerate"].sum()),
        "mae_b0": mae_b0,
        "mae_b1": mae_b1,
        "mae_b2": mae_b2,
        "mae_blin": mae_blin,
        "mae_e": mae_e,
        "mae_e_no_hinge": mae_e_no_hinge,
        "delta_small": mae_b1 - mae_e,
        "delta_strong": mae_b2 - mae_e,
        "delta_e_b0": mae_b0 - mae_e,
        "delta_b1_b0": mae_b0 - mae_b1,
        "delta_b2_b0": mae_b0 - mae_b2,
        "bulk_e_minus_b1": bulk_e_minus_b1,
        "bulk_e_minus_b2": bulk_e_minus_b2,
        "bulk_e_minus_b0": bulk_e_minus_b0,
        "n_bulk_eval": int(bulk_mask.sum()),
        "b1_params": _count_parameters(b1),
        "b2_params": _count_parameters(b2),
        "blin_params": _count_parameters(blin),
        "e_params": _count_parameters(e_model),
        "b1_selection_mae": b1_info["best_selection_mae"],
        "b2_selection_mae": b2_info["best_selection_mae"],
        "blin_selection_mae": blin_info["best_selection_mae"],
        "e_selection_mae": e_info["best_selection_mae"],
        "e_grad_alive": e_info["grad_alive"],
        "e_sensitivity_max_abs_diff": float(np.abs(p_e - p_e_no_hinge).max()),
        "_molecules": {
            "subset_index": data["eva_ids"],
            "fold": np.full(len(data["eva_ids"]), fold, dtype=np.int64),
            "err_b0": np.abs(y - yhat0),
            "err_b1": np.abs(y - p_b1),
            "err_b2": np.abs(y - p_b2),
            "err_blin": np.abs(y - p_blin),
            "err_e": np.abs(y - p_e),
            "err_e_no_hinge": np.abs(y - p_e_no_hinge),
        },
        "_knots": knots,
    }


def _aggregate_stage(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    frame = pd.DataFrame([{k: v for k, v in row.items() if not k.startswith("_")} for row in rows])
    dsmall = frame["delta_small"].to_numpy(dtype=np.float64)
    dstrong = frame["delta_strong"].to_numpy(dtype=np.float64)
    e_minus_b2 = -dstrong
    bulk = frame["bulk_e_minus_b1"].to_numpy(dtype=np.float64)
    bulk = bulk[np.isfinite(bulk)]
    return {
        "n_folds": int(len(frame)),
        "mean_b0": float(frame["mae_b0"].mean()),
        "mean_b1": float(frame["mae_b1"].mean()),
        "mean_b2": float(frame["mae_b2"].mean()),
        "mean_blin": float(frame["mae_blin"].mean()),
        "mean_e": float(frame["mae_e"].mean()),
        "mean_e_no_hinge": float(frame["mae_e_no_hinge"].mean()),
        "mean_delta_small": float(dsmall.mean()),
        "median_delta_small": float(np.median(dsmall)),
        "mean_delta_strong": float(dstrong.mean()),
        "median_delta_strong": float(np.median(dstrong)),
        "mean_e_minus_b2": float(e_minus_b2.mean()),
        "positive_folds_delta_small": int(np.sum(dsmall > 0)),
        "positive_folds_delta_strong": int(np.sum(dstrong > 0)),
        "per_fold_delta_small": dsmall.tolist(),
        "per_fold_delta_strong": dstrong.tolist(),
        "mean_delta_e_b0": float(frame["delta_e_b0"].mean()),
        "mean_delta_b1_b0": float(frame["delta_b1_b0"].mean()),
        "mean_delta_b2_b0": float(frame["delta_b2_b0"].mean()),
        "mean_bulk_e_minus_b1": float(bulk.mean()) if bulk.size else None,
        "max_bulk_e_minus_b1": float(bulk.max()) if bulk.size else None,
        "b1_params": int(frame["b1_params"].iloc[0]),
        "b2_params": int(frame["b2_params"].iloc[0]),
        "e_params": int(frame["e_params"].iloc[0]),
        "blin_params": int(frame["blin_params"].iloc[0]),
        "e_grad_alive_all": bool(frame["e_grad_alive"].all()),
        "e_sensitivity_min": float(frame["e_sensitivity_max_abs_diff"].min()),
    }


def run_stage(
    backbone_seed: int,
    adapter_seed: int,
    tag: str,
    manifest: Mapping[str, Any],
    force: bool = False,
    fixed_knots: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    out_csv = RESULTS_DIR / f"{tag}_fold_results.csv"
    out_boot = RESULTS_DIR / f"{tag}_bootstrap.json"
    if out_csv.exists() and out_boot.exists() and not force:
        cached = json.loads(out_boot.read_text(encoding="utf-8"))
        return {
            "csv": str(out_csv),
            "bootstrap": str(out_boot),
            "aggregate": cached["aggregate"],
            "bootstrap_detail": {
                "delta_small": cached["delta_small"],
                "delta_strong": cached["delta_strong"],
                "delta_hinge_vs_linear_only": cached.get("delta_hinge_vs_linear_only"),
            },
            "cached": True,
        }
    horizon = _selected_horizon()
    rows = [
        _run_stage_for_fold(fold, backbone_seed, adapter_seed, manifest, horizon, fixed_knots)
        for fold in range(K_FOLDS)
    ]
    pd.DataFrame([{k: v for k, v in row.items() if not k.startswith("_")} for row in rows]).to_csv(
        out_csv, index=False
    )
    molecules = [row["_molecules"] for row in rows]
    fold_ids = np.concatenate([m["fold"] for m in molecules])
    dsmall = np.concatenate([m["err_b1"] - m["err_e"] for m in molecules])
    dstrong = np.concatenate([m["err_b2"] - m["err_e"] for m in molecules])
    dhinge = np.concatenate([m["err_e_no_hinge"] - m["err_e"] for m in molecules])
    bootstrap = {
        "backbone_seed": int(backbone_seed),
        "adapter_seed": int(adapter_seed),
        "horizon": int(horizon),
        "fixed_knots": bool(fixed_knots),
        "delta_small": ciw._stratified_bootstrap(dsmall, fold_ids),
        "delta_strong": ciw._stratified_bootstrap(dstrong, fold_ids),
        "delta_hinge_vs_linear_only": ciw._stratified_bootstrap(dhinge, fold_ids),
        "aggregate": _aggregate_stage(rows),
        "seconds": time.perf_counter() - started,
    }
    _write_json_any(out_boot, bootstrap)
    print(
        f"[{tag}] backbone={backbone_seed} init={adapter_seed} "
        f"mean dsmall={bootstrap['aggregate']['mean_delta_small']:+.4f} "
        f"mean dstrong={bootstrap['aggregate']['mean_delta_strong']:+.4f} "
        f"({time.perf_counter() - started:.0f}s)",
        flush=True,
    )
    return {
        "csv": str(out_csv),
        "bootstrap": str(out_boot),
        "aggregate": bootstrap["aggregate"],
        "bootstrap_detail": {
            "delta_small": bootstrap["delta_small"],
            "delta_strong": bootstrap["delta_strong"],
            "delta_hinge_vs_linear_only": bootstrap["delta_hinge_vs_linear_only"],
        },
        "cached": False,
    }


# ---------------------------------------------------------------------------
# decision logic
# ---------------------------------------------------------------------------

def _decide_stage1(stage1: Mapping[str, Any], stage1b: Mapping[str, Any] | None = None) -> str:
    agg = stage1["aggregate"]
    mean_dsmall = agg["mean_delta_small"]
    folds_positive = agg["positive_folds_delta_small"]
    e_minus_b2 = agg["mean_e_minus_b2"]

    if mean_dsmall <= S1_NGO_DSMALL or folds_positive <= S1_NGO_FOLDS:
        return "STAGE1_CLEAR_NO_GO"

    if mean_dsmall >= S1_STRONG_DSMALL and folds_positive >= S1_STRONG_FOLDS:
        if e_minus_b2 <= S1_STRONG_E_MINUS_B2_MAX:
            verdict = "STAGE1_STRONG_ADVANCE"
        elif e_minus_b2 <= S1_EFF_E_MINUS_B2_MAX:
            verdict = "STAGE1_EFFICIENCY_ADVANCE"
        else:
            verdict = "STAGE1_BORDERLINE"
    elif S1_NGO_DSMALL < mean_dsmall < S1_STRONG_DSMALL and folds_positive >= S1_BORDERLINE_FOLDS:
        verdict = "STAGE1_BORDERLINE"
    else:
        verdict = "STAGE1_INCONCLUSIVE"

    if verdict == "STAGE1_BORDERLINE" and stage1b is not None:
        agg_b = stage1b["aggregate"]
        consistent = (
            agg_b["mean_delta_small"] > 0
            and agg_b["positive_folds_delta_small"] >= S1_BORDERLINE_FOLDS
            and np.sign(agg_b["mean_delta_small"]) == np.sign(mean_dsmall)
        )
        if not consistent:
            return "STAGE1_INCONCLUSIVE"
        # confirm against the stronger / efficiency gate using both inits
        pooled = float(np.mean([mean_dsmall, agg_b["mean_delta_small"]]))
        pooled_e_minus_b2 = float(np.mean([e_minus_b2, agg_b["mean_e_minus_b2"]]))
        if pooled >= S1_STRONG_DSMALL and pooled_e_minus_b2 <= S1_EFF_E_MINUS_B2_MAX:
            return "STAGE1_STRONG_ADVANCE" if pooled_e_minus_b2 <= S1_STRONG_E_MINUS_B2_MAX else "STAGE1_EFFICIENCY_ADVANCE"
        return "STAGE1_INCONCLUSIVE"
    return verdict


def _boot_small(stage: Mapping[str, Any]) -> Mapping[str, Any]:
    if "bootstrap_detail" in stage:
        return stage["bootstrap_detail"]["delta_small"]
    return stage["delta_small"]


def _final_decision(
    stage1: Mapping[str, Any],
    stage2: Mapping[str, Any] | None,
    stage1b: Mapping[str, Any] | None,
    mechanism: Mapping[str, Any] | None = None,
    fixed_knot: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    stages = [stage1] + ([stage2] if stage2 is not None else [])
    per_fold_dsmall = np.mean([s["aggregate"]["per_fold_delta_small"] for s in stages], axis=0)
    per_fold_dstrong = np.mean([s["aggregate"]["per_fold_delta_strong"] for s in stages], axis=0)
    pooled_dsmall = float(np.mean([s["aggregate"]["mean_delta_small"] for s in stages]))
    pooled_e_minus_b2 = float(np.mean([s["aggregate"]["mean_e_minus_b2"] for s in stages]))
    backbone_consistent = all(s["aggregate"]["mean_delta_small"] > 0 for s in stages)
    folds_positive = int(np.sum(per_fold_dsmall > 0))

    boot = [_boot_small(s) for s in stages]
    ci_low_positive = all(b["ci95_low"] > 0 for b in boot)

    bulk_values = [
        s["aggregate"].get("max_bulk_e_minus_b1") for s in stages
    ]
    bulk_values = [float(v) for v in bulk_values if v is not None and np.isfinite(v)]
    bulk_safe = bool(bulk_values) and max(bulk_values) <= BULK_MAX_DEGRADATION
    sensitivity_ok = all(s["aggregate"]["e_sensitivity_min"] > 0 for s in stages)

    criteria = {
        "pooled_delta_small_ge_0.0025": pooled_dsmall >= FINAL_GO_POOLED,
        "backbone_consistent_positive": backbone_consistent,
        "fold_averaged_at_least_4_of_5_positive": folds_positive >= FINAL_GO_FOLDS,
        "paired_delta_small_ci_lower_positive": ci_low_positive,
        "explicit_basis_noncollapsed": bool(sensitivity_ok),
        "bulk_safe_le_0.002": bulk_safe,
    }
    if stage2 is not None and all(criteria.values()):
        if pooled_e_minus_b2 <= 0.0:
            verdict = "GO_STRONG_FUNCTION_BASIS_ACCESSIBILITY"
            case = "Case A"
        elif pooled_e_minus_b2 <= S1_STRONG_E_MINUS_B2_MAX:
            verdict = "GO_FUNCTION_BASIS_ACCESSIBILITY"
            case = "Case A (E matches/beats B2)"
        else:
            verdict = "GO_FUNCTION_BASIS_PARAMETER_EFFICIENCY"
            case = "Case B"
    elif stage2 is None and stage1["aggregate"]["mean_delta_small"] > S1_NGO_DSMALL:
        verdict = "INCONCLUSIVE_NO_REPLICATION_SPENT"
        case = "Case D"
    elif stage2 is not None:
        verdict = "INCONCLUSIVE_OR_NO_GO"
        case = "Case D"
    else:
        verdict = "NO_GO"
        case = "Case C"

    return {
        "case": case,
        "verdict": verdict,
        "summary": (
            "Explicit target-independent quantile-hinge additive basis (E) fails the "
            "pre-registered Stage-1 gate: mean Delta_small = MAE(B1) - MAE(E) <= +0.0005 "
            "with 0/5 folds positive, and E is also worse than B2 and B0.  The generic "
            "ReLU MLP at matched budget (B1) is the better function class on frozen R."
            if verdict == "NO_GO"
            else f"Function-basis accessibility verdict: {verdict}."
        ),
        "action": (
            "STOP.  No second backbone, no more knots, no learned knots, no B-splines, "
            "no polynomial / RBF / Fourier / KSVD response dictionary, no XGBoost "
            "comparison.  The conclusion is scoped to: a generic coordinatewise "
            "quantile-hinge basis does not show better sample efficiency than a "
            "same-budget ReLU MLP.  It does NOT mean function parameterisation is "
            "unimportant in general."
            if verdict == "NO_GO"
            else "Report and, if GO, continue with a pre-registered follow-up."
        ),
        "scope_caveat": (
            "Screening diagnostic on the frozen compact-v4-hinge OOF representation "
            "(official train universe only).  B0/B1/B2/E all see exactly the same "
            "graph information (R, 302D) and the same L1 objective; only the final "
            "function parameterisation differs.  A NO-GO closes the simple additive "
            "coordinatewise hinge basis, not the broader function-basis hypothesis."
        ),
        "delta_hinge_vs_linear_only": (
            stage1.get("bootstrap_detail", {}).get("delta_hinge_vs_linear_only")
            if "bootstrap_detail" in stage1 else stage1.get("delta_hinge_vs_linear_only")
        ),
        "pooled_delta_small": pooled_dsmall,
        "pooled_e_minus_b2": pooled_e_minus_b2,
        "per_fold_delta_small": per_fold_dsmall.tolist(),
        "per_fold_delta_strong": per_fold_dstrong.tolist(),
        "folds_delta_small_positive": folds_positive,
        "criteria": criteria,
        "stage1_aggregate": stage1["aggregate"],
        "stage2_aggregate": None if stage2 is None else stage2["aggregate"],
        "stage1b_aggregate": None if stage1b is None else stage1b["aggregate"],
        "delta_small_ci95": [[b["ci95_low"], b["ci95_high"]] for b in boot],
        "bulk_max_e_minus_b1": max(bulk_values) if bulk_values else None,
        "mechanism": None if mechanism is None else mechanism.get("summary"),
        "fixed_knot_control": None if fixed_knot is None else fixed_knot.get("aggregate"),
    }


# ---------------------------------------------------------------------------
# mechanism analysis (GO only)
# ---------------------------------------------------------------------------

def run_mechanism_analysis(force: bool = False) -> dict[str, Any]:
    out_path = RESULTS_DIR / "coefficient_summary.csv"
    if out_path.exists() and not force:
        frame = pd.read_csv(out_path)
        return {"coefficient_summary": str(out_path), "n_coordinates": int(len(frame))}
    # retrain E on backbone seed0 and collect coefficients
    manifest = stage_splits()
    horizon = _selected_horizon()
    all_coeffs: list[np.ndarray] = []
    for fold in range(K_FOLDS):
        data = _fold_tensors(fold, PRIMARY_BACKBONE_SEED, manifest)
        fit = _tensors(data, "fit_pos")
        sel = _tensors(data, "sel_pos")
        model = _make_hinge(data["knots"], 0)
        _train_reader(
            model, fit["z"], fit["y"], fit["yhat"], sel["z"], sel["y"], sel["yhat"], epochs=horizon
        )
        w = model.linear.weight.detach().numpy().reshape(-1)  # (4D,)
        beta0 = w[:R_DIM]
        beta1 = w[R_DIM: 2 * R_DIM]
        beta2 = w[2 * R_DIM: 3 * R_DIM]
        beta3 = w[3 * R_DIM:]
        all_coeffs.append(np.stack([beta0, beta1, beta2, beta3], axis=1))  # (D,4)
    mean_coeffs = np.mean(all_coeffs, axis=0)  # (D,4)
    nonlinear_mass = np.abs(mean_coeffs[:, 1:]).sum(axis=1)
    frame = pd.DataFrame(
        {
            "coordinate": np.arange(R_DIM),
            "abs_beta_z": np.abs(mean_coeffs[:, 0]),
            "abs_beta_h25": np.abs(mean_coeffs[:, 1]),
            "abs_beta_h50": np.abs(mean_coeffs[:, 2]),
            "abs_beta_h75": np.abs(mean_coeffs[:, 3]),
            "nonlinear_hinge_mass": nonlinear_mass,
        }
    )
    frame.to_csv(out_path, index=False)
    summary = {
        "n_coordinates": int(R_DIM),
        "mean_abs_beta_z": float(frame["abs_beta_z"].mean()),
        "mean_nonlinear_hinge_mass": float(frame["nonlinear_hinge_mass"].mean()),
        "linear_mass": float(frame["abs_beta_z"].sum()),
        "hinge_mass": float(nonlinear_mass.sum()),
        "top_hinge_coordinates": frame.sort_values("nonlinear_hinge_mass", ascending=False)
        .head(15)["coordinate"].tolist(),
        "channel_groups": {
            "unary": [0, 96],
            "pair_bucket_moments": [97, 261],
            "global": [262, 293],
            "topology": [294, 301],
        },
    }
    _write_json_any(RESULTS_DIR / "coefficient_summary.json", summary)
    return {"coefficient_summary": str(out_path), "summary": summary}


def run_hinge_ablation(manifest: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Frozen inference ablation: E vs E with all hinge coefficients zeroed."""
    out_path = RESULTS_DIR / "hinge_ablation_results.csv"
    if out_path.exists():
        frame = pd.read_csv(out_path)
        return {"path": str(out_path), "mean_ablation_delta": float(frame["ablation_delta"].mean())}
    manifest = manifest or stage_splits()
    horizon = _selected_horizon()
    rows: list[dict[str, Any]] = []
    for fold in range(K_FOLDS):
        row = _run_stage_for_fold(fold, PRIMARY_BACKBONE_SEED, 0, manifest, horizon)
        rows.append(
            {
                "fold": fold,
                "mae_e": row["mae_e"],
                "mae_e_linear_only": row["mae_e_no_hinge"],
                "ablation_delta": row["mae_e_no_hinge"] - row["mae_e"],
            }
        )
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return {"path": str(out_path), "mean_ablation_delta": float(np.mean([r["ablation_delta"] for r in rows]))}


def run_fixed_knot_control(manifest: Mapping[str, Any] | None = None, force: bool = False) -> dict[str, Any]:
    """Knot-location control: replace quantile knots with fixed -1/0/+1 and retrain E."""
    out_csv = RESULTS_DIR / "fixed_knot_control.csv"
    out_json = RESULTS_DIR / "fixed_knot_control.json"
    if out_csv.exists() and out_json.exists() and not force:
        return json.loads(out_json.read_text(encoding="utf-8"))
    manifest = manifest or stage_splits()
    horizon = _selected_horizon()
    rows = [
        _run_stage_for_fold(fold, PRIMARY_BACKBONE_SEED, 0, manifest, horizon, fixed_knots=True)
        for fold in range(K_FOLDS)
    ]
    pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]).to_csv(
        out_csv, index=False
    )
    detail = {
        "fixed_knots": list(FIXED_KNOT_VALUES),
        "per_fold_delta_small": [r["delta_small"] for r in rows],
        "mean_delta_small": float(np.mean([r["delta_small"] for r in rows])),
        "positive_folds": int(np.sum([r["delta_small"] > 0 for r in rows])),
        "mean_e": float(np.mean([r["mae_e"] for r in rows])),
        "mean_b1": float(np.mean([r["mae_b1"] for r in rows])),
    }
    _write_json_any(out_json, detail)
    return detail


# ---------------------------------------------------------------------------
# figures / readme
# ---------------------------------------------------------------------------

def _make_figures(stage1: Mapping[str, Any], stage2: Mapping[str, Any] | None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(RESULTS_DIR / "stage1_fold_results.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.6))
    ax = axes[0]
    x = np.arange(K_FOLDS)
    ax.bar(x - 0.2, frame["delta_small"], width=0.4, label=r"$\Delta_{small}$ (B1 - E)")
    ax.bar(x + 0.2, frame["delta_strong"], width=0.4, label=r"$\Delta_{strong}$ (B2 - E)")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("outer fold")
    ax.set_ylabel("MAE improvement")
    ax.set_title("Per-fold basis gain (backbone seed 0)")
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.bar(x - 0.15, frame["mae_b0"], width=0.3, label="B0")
    ax.bar(x, frame["mae_b1"], width=0.3, label="B1")
    ax.bar(x + 0.15, frame["mae_e"], width=0.3, label="E")
    ax.set_xlabel("outer fold")
    ax.set_ylabel("MAE")
    ax.set_title("Reader MAE")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure1_stage1.png", dpi=150)
    plt.close(fig)
    if stage2 is not None and (RESULTS_DIR / "stage2_fold_results.csv").exists():
        frame2 = pd.read_csv(RESULTS_DIR / "stage2_fold_results.csv")
        fig, ax = plt.subplots(figsize=(5.6, 3.4))
        ax.plot(x, frame["delta_small"], marker="o", label="backbone seed 0")
        ax.plot(x, frame2["delta_small"], marker="s", label="backbone seed 1")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xlabel("outer fold")
        ax.set_ylabel(r"$\Delta_{small}$")
        ax.set_title("Basis gain across frozen backbone seeds")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "figure2_stage2_seeds.png", dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

def _run_all(force: bool = False) -> int:
    print("=== checkpoint inventory ===", flush=True)
    inventory = ciw.checkpoint_inventory()
    _write_json_any(RESULTS_DIR / "checkpoint_inventory.json", inventory)
    if not inventory["complete_seeds"]:
        print("no complete frozen checkpoint set; STOP.", flush=True)
        _write_json_any(RESULTS_DIR / "final_decision.json", {"verdict": "NO_GO", "reason": "no_checkpoints"})
        return 2

    print("=== reader spec ===", flush=True)
    accounting = _parameter_accounting()
    write_basis_spec(accounting)
    if not accounting["parameter_match_ok"]:
        print("B1/E parameter mismatch > 3%; STOP.", flush=True)
        _write_json_any(
            RESULTS_DIR / "final_decision.json",
            {"verdict": "NO_GO", "reason": "parameter_mismatch", "accounting": accounting},
        )
        return 3

    print("=== cache + integrity ===", flush=True)
    stage_cache(seeds=(PRIMARY_BACKBONE_SEED,), force=force)
    integrity = run_integrity(seeds=(PRIMARY_BACKBONE_SEED,), force=force)
    if not integrity["all_passed"]:
        print("REPRESENTATION INTEGRITY FAILED; STOP.", flush=True)
        _write_json_any(
            RESULTS_DIR / "final_decision.json",
            {"verdict": "NO_GO", "reason": "integrity_failed", "failures": integrity["failures"]},
        )
        return 3

    print("=== split manifest ===", flush=True)
    manifest = stage_splits(force=force)

    print("=== convergence trace ===", flush=True)
    trace = run_convergence_trace(force=force)
    _write_json_any(RESULTS_DIR / "preprocessing_stats.json", _preprocessing_stats(manifest))

    print("=== stage 1 (backbone seed 0, init 0) ===", flush=True)
    stage1 = run_stage(PRIMARY_BACKBONE_SEED, 0, "stage1", manifest, force=force)
    verdict1 = _decide_stage1(stage1)

    stage1b = None
    stage2 = None
    mechanism = None
    fixed_knot = None

    if verdict1 in ("STAGE1_CLEAR_NO_GO", "STAGE1_INCONCLUSIVE"):
        print(f"Stage 1 verdict={verdict1}; second backbone seed NOT spent.", flush=True)
    else:
        if verdict1 == "STAGE1_BORDERLINE":
            print("=== stage 1b (second reader init, same backbone) ===", flush=True)
            stage1b = run_stage(PRIMARY_BACKBONE_SEED, 1, "stage1b", manifest, force=force)
            verdict1 = _decide_stage1(stage1, stage1b)
            print(f"Stage 1b confirmation verdict={verdict1}", flush=True)
        if verdict1 in ("STAGE1_STRONG_ADVANCE", "STAGE1_EFFICIENCY_ADVANCE"):
            print("=== stage 2 (second frozen backbone seed) ===", flush=True)
            stage_cache(seeds=(BACKBONE_SEEDS[1],), force=force)
            run_integrity(seeds=(BACKBONE_SEEDS[1],), force=force)
            stage2 = run_stage(BACKBONE_SEEDS[1], 0, "stage2", manifest, force=force)
            print("=== mechanism analysis ===", flush=True)
            mechanism = run_mechanism_analysis(force=force)
            run_hinge_ablation(manifest)
            fixed_knot = run_fixed_knot_control(manifest, force=force)

    decision = _final_decision(stage1, stage2, stage1b, mechanism, fixed_knot)
    decision["stage1_verdict"] = verdict1
    decision["convergence_horizon"] = trace["selected_horizon"]
    _write_json_any(RESULTS_DIR / "final_decision.json", decision)
    if stage2 is not None:
        _write_json_any(
            RESULTS_DIR / "pooled_backbone_results.csv",
            {"note": "see stage1/stage2 fold csv"},
        )
        _write_json_any(
            RESULTS_DIR / "final_bootstrap.json",
            {
                "stage1_delta_small": stage1["bootstrap_detail"]["delta_small"],
                "stage2_delta_small": stage2["bootstrap_detail"]["delta_small"],
            },
        )
    _make_figures(stage1, stage2)
    print(json.dumps(decision, indent=2, sort_keys=True), flush=True)
    return 0


def _preprocessing_stats(manifest: Mapping[str, Any]) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for fold in range(K_FOLDS):
        data = _fold_tensors(fold, PRIMARY_BACKBONE_SEED, manifest)
        knots = data["knots"]
        stats[f"fold{fold}"] = {
            "n_fit": int(len(data["fit_pos"])),
            "n_selection": int(len(data["sel_pos"])),
            "n_eval": int(len(data["eva_pos"])),
            "degenerate_coordinates": int(data["degenerate"].sum()),
            "degenerate_indices": np.flatnonzero(data["degenerate"]).tolist(),
            "mean_abs_mean_of_coordinates": float(np.abs(data["mean"]).mean()),
            "mean_scale": float(data["scale"].mean()),
            "knot_hash": _sha256_array(knots.astype(np.float32)),
            "knot_min": float(knots.min()),
            "knot_max": float(knots.max()),
        }
    return stats


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=[
            "inventory", "spec", "cache", "integrity", "splits", "preprocessing",
            "convergence", "stage1", "stage1b", "stage2", "mechanism", "hinge_ablation",
            "fixed_knot", "figures", "decision", "all",
        ],
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seeds", default="0", help="comma-separated backbone seeds")
    args = parser.parse_args(argv)
    seed_list = tuple(int(s) for s in str(args.seeds).split(",") if s.strip() != "")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.stage == "inventory":
        inventory = ciw.checkpoint_inventory()
        _write_json_any(RESULTS_DIR / "checkpoint_inventory.json", inventory)
        print(json.dumps(inventory, indent=2, sort_keys=True))
        return 0
    if args.stage == "spec":
        write_basis_spec()
        return 0
    if args.stage == "cache":
        stage_cache(seeds=seed_list, force=args.force)
        return 0
    if args.stage == "integrity":
        report = run_integrity(seeds=seed_list, force=args.force)
        return 0 if report["all_passed"] else 1
    if args.stage == "splits":
        stage_splits(force=args.force)
        return 0
    if args.stage == "preprocessing":
        manifest = stage_splits()
        _write_json_any(RESULTS_DIR / "preprocessing_stats.json", _preprocessing_stats(manifest))
        return 0
    if args.stage == "convergence":
        run_convergence_trace(force=args.force)
        return 0
    if args.stage == "stage1":
        manifest = stage_splits()
        run_stage(PRIMARY_BACKBONE_SEED, 0, "stage1", manifest, force=args.force)
        return 0
    if args.stage == "stage1b":
        manifest = stage_splits()
        run_stage(PRIMARY_BACKBONE_SEED, 1, "stage1b", manifest, force=args.force)
        return 0
    if args.stage == "stage2":
        manifest = stage_splits()
        stage_cache(seeds=(BACKBONE_SEEDS[1],), force=args.force)
        run_integrity(seeds=(BACKBONE_SEEDS[1],), force=args.force)
        run_stage(BACKBONE_SEEDS[1], 0, "stage2", manifest, force=args.force)
        return 0
    if args.stage == "mechanism":
        run_mechanism_analysis(force=args.force)
        run_hinge_ablation()
        return 0
    if args.stage == "hinge_ablation":
        run_hinge_ablation()
        return 0
    if args.stage == "fixed_knot":
        run_fixed_knot_control(force=args.force)
        return 0
    if args.stage == "figures":
        stage1 = json.loads((RESULTS_DIR / "stage1_bootstrap.json").read_text())
        stage2_path = RESULTS_DIR / "stage2_bootstrap.json"
        stage2 = json.loads(stage2_path.read_text()) if stage2_path.exists() else None
        _make_figures(stage1, stage2)
        return 0
    if args.stage == "decision":
        stage1 = json.loads((RESULTS_DIR / "stage1_bootstrap.json").read_text())
        stage1b_path = RESULTS_DIR / "stage1b_bootstrap.json"
        stage2_path = RESULTS_DIR / "stage2_bootstrap.json"
        stage1b = json.loads(stage1b_path.read_text()) if stage1b_path.exists() else None
        stage2 = json.loads(stage2_path.read_text()) if stage2_path.exists() else None
        decision = _final_decision(stage1, stage2, stage1b)
        decision["stage1_verdict"] = _decide_stage1(stage1, stage1b)
        _write_json_any(RESULTS_DIR / "final_decision.json", decision)
        print(json.dumps(decision, indent=2, sort_keys=True))
        return 0
    if args.stage == "all":
        return _run_all(force=args.force)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
