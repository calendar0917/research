"""Graph-Head Function Family Audit (compact-v4, ZINC).

This is a **representation-frozen, downstream-function-family** audit -- not an
architecture benchmark.  It does not train a backbone, does not modify the
patch representation / tokenizer / pair encoder / centre update / pooling /
topology branch, and does not add any new structural feature.  It answers
exactly one question:

    On the frozen compact-v4 graph representation R (302D), is an explicit
    low-rank cross-coordinate interaction head (a rank-4 Factorization
    Machine) a more sample-efficient downstream regression function family
    than a *same-budget* generic ReLU MLP?

It follows directly from the frozen function-basis accessibility audit, whose
CLEAR NO-GO showed that a purely additive coordinatewise quantile-hinge reader
loses badly to a same-budget generic ReLU MLP -- consistent with the missing
ingredient being **coordinate interaction**, not a per-coordinate response
shape.  This audit changes only the downstream function family and keeps the
representation, objective and data fixed.

Protocol (differs from the earlier frozen audits on purpose):

* deterministic 5-fold OOF compact-v4 checkpoints on the **official train
  universe only**; official valid/test are never loaded;
* per fold the frozen backbone exposes a true nested structure
  (7200 backbone-fit / 800 backbone-selection / 2000 outer-heldout);
* each **direct** head is ``yhat = f(R)`` (no residual ``yhat_0``), fit on the
  7200, checkpoint-selected on the 800, and evaluated once on the untouched
  2000 -- so the head sees a training regime close to the real ZINC scale
  while the final comparison stays on molecules the backbone never trained on;
* fit-only coordinate standardisation (7200 only), identical L1/MAE objective,
  Adam(lr=1e-3), full-batch, deterministic seed, fixed horizon, no sweeps.

Heads:

* ``H0`` frozen original graph head prediction (jointly-trained reference);
* ``Hlinear`` direct ``302 -> 1`` L1 (descriptive baseline, not in the gate);
* ``H1`` direct generic ReLU MLP ``302 -> 5 -> 2 -> 1`` (parameter-matched to FM);
* ``H2`` direct generic ReLU MLP ``302 -> 13 -> 13 -> 1`` (strong reference);
* ``E`` direct rank-4 Factorization Machine (linear + low-rank pairwise
  interaction), the primary experimental head.

CatBoost-MAE is a pre-registered *secondary* tree-style reference; the package
is not installed, so it is recorded as unavailable and the audit continues.

Run::

    uv run python -m tracks.ksvd.experiments.luyin16.zinc_graph_head_function_family <stage>
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
from torch_geometric.loader import DataLoader

from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp
from tracks.ksvd.experiments.luyin16 import (
    zinc_centre_incidence_cooccurrence_witness as ciw,
)
from tracks.ksvd.experiments.luyin16.zinc_oof_difficulty_audit import (
    FOLD_DIR,
    K_FOLDS,
    _fold_slices,
    _frozen_config,
    _load_fold_npz,
    _load_train_labels,
    _load_train_records,
    _model_config_with_clamps,
    _verify_records_vs_labels,
)
from tracks.ksvd.experiments.luyin16.zinc_post_v4_residual_audit import _build_v4_model

REPO_ROOT = zpp.REPO_ROOT
RESULTS_DIR = REPO_ROOT / "tracks/ksvd/results/graph_head_function_family"
CACHE_DIR = RESULTS_DIR / "state_exports"
FIG_DIR = RESULTS_DIR / "figures"
CONFIG_PATH = REPO_ROOT / "tracks/ksvd/configs/luyin16/zinc_compact_v4_topology_hinge.yaml"

SOURCE_EXPORT_VERSION = ciw.EXPORT_VERSION  # frozen_state_export_v4_centre_incidence
CACHE_VERSION = "graph_head_R_cache_v1"

# --- frozen compact-v4 representation --------------------------------------
R_DIM = 302  # verified at runtime from the real backbone state dict
R_LAYOUT = {
    "unary": [0, 96],           # 97
    "pair_bucket_moments": [97, 261],  # 165
    "global": [262, 293],       # 32
    "topology": [294, 301],     # 8
}
PATCH_DIM = 48
PAIR_DIM = 16

# --- frozen graph head (reconstruction only) -------------------------------
HEAD_HIDDEN_0 = 64
HEAD_HIDDEN_1 = 32
HEAD_DROPOUT = 0.05

# --- pre-registered direct heads -------------------------------------------
H_LINEAR_HIDDEN = ()
H1_HIDDEN = (5, 2)       # 302 -> 5 -> 2 -> 1 = 1530 params
H2_HIDDEN = (13, 13)     # 302 -> 13 -> 13 -> 1 = 4135 params
FM_RANK = 4              # FIXED.  No rank sweep.
FM_INIT_STD = 1.0e-2
PARAM_MATCH_TOL = 0.03

# --- optimisation / training protocol (identical for every direct head) ----
ADAPTER_LR = 1.0e-3
ADAPTER_WEIGHT_DECAY = 0.0
ADAPTER_BATCH_SIZE = 512  # one unified deterministic mini-batch size for every head
HEAD_SEED = 0
CONVERGENCE_EPOCHS = (100, 200, 400, 800)
HORIZON_FAST = 400
HORIZON_SLOW = 800
HORIZON_TOL = 1.0e-4

# --- split / budget ---------------------------------------------------------
BACKBONE_SEEDS = (0, 1)
PRIMARY_BACKBONE_SEED = 0
ADAPTER_SEEDS = (0, 1)  # seed 1 only for Stage 1b (borderline)
N_FIT = 7200
N_SELECT = 800
N_EVAL = 2000

# --- decision thresholds (pre-registered) -----------------------------------
S1_STRONG_DSMALL = 0.003
S1_STRONG_FOLDS = 4
S1_STRONG_FM_MINUS_H2_MAX = 0.0015
S1_EFF_DSMALL = 0.003
S1_EFF_FOLDS = 4
S1_EFF_FM_MINUS_H2_MAX = 0.004
S1_NGO_DSMALL = 0.0005
S1_NGO_FOLDS = 2  # FM better than H1 in <= 2/5 folds -> clear NO-GO
S1_BORDERLINE_FOLDS = 3
FINAL_GO_POOLED = 0.0025
FINAL_GO_FOLDS = 4
MECHANISM_INTERACTION_MIN = 1.0e-4  # interaction must move predictions by more than this

# --- gate tolerances --------------------------------------------------------
FORWARD_GATE_ATOL = 1.0e-9
R_RECON_ATOL = 1.0e-4
HEAD_RECON_ATOL = 1.0e-5
STANDARDIZE_EPS = 1.0e-6

# --- CatBoost secondary reference (pre-registered preset; not swept) --------
CATBOOST_PRESET = {
    "loss_function": "MAE",
    "eval_metric": "MAE",
    "depth": 6,
    "learning_rate": 0.03,
    "iterations": 1000,
    "random_seed": 0,
    "use_best_model": True,
    "early_stopping_rounds": 100,
    "verbose": False,
}


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


def _sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _cache_path(fold: int, seed: int) -> Path:
    return CACHE_DIR / f"{CACHE_VERSION}_fold{fold}_seed{seed}.npz"


# ---------------------------------------------------------------------------
# Stage 0: full nested representation export (7200 / 800 / 2000)
# ---------------------------------------------------------------------------

def _capture_R(model: nn.Module, graphs: Sequence[Any], batch_size: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """Eval forward capturing the pre-head 302D representation R and yhat."""
    loader = DataLoader(list(graphs), batch_size=int(batch_size), shuffle=False)
    R_rows: list[np.ndarray] = []
    predictions: list[np.ndarray] = []

    def head_pre_hook(_module, args) -> None:
        R_rows.append(args[0].detach().cpu().numpy())

    handle = model.head[0].register_forward_pre_hook(head_pre_hook)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(torch.device("cpu"))
            predictions.append(model(batch).cpu().numpy())
    handle.remove()
    R = np.concatenate(R_rows, axis=0).astype(np.float32)
    yhat = np.concatenate(predictions).astype(np.float64)
    return R, yhat


def _build_full_export(fold: int, seed: int) -> dict[str, Any]:
    records = _load_train_records()
    labels = _load_train_labels()
    _verify_records_vs_labels(records, labels)
    config = _frozen_config()
    torch.set_num_threads(int(config.get("runtime", {}).get("torch_threads", 4)))
    inner_train_idx, inner_valid_idx, holdout_idx = _fold_slices()[fold]
    if not (len(inner_train_idx) == N_FIT and len(inner_valid_idx) == N_SELECT and len(holdout_idx) == N_EVAL):
        raise AssertionError(
            f"unexpected fold sizes: {len(inner_train_idx)}/{len(inner_valid_idx)}/{len(holdout_idx)}"
        )
    inner_train_records = [records[int(i)] for i in inner_train_idx]
    other_records = [records[int(i)] for i in inner_valid_idx] + [records[int(i)] for i in holdout_idx]
    encoded_fit, encoded_other, audit = zpp._phase_data(
        inner_train_records, other_records, config=config
    )
    config_mod = dict(config)
    config_mod["model"] = _model_config_with_clamps(config, audit)
    model = _build_v4_model(config_mod, audit)
    state = torch.load(
        FOLD_DIR / f"fold{fold}_seed{seed}_state.pt", map_location="cpu", weights_only=True
    )
    model.load_state_dict(state)
    if model.head[0].in_features != R_DIM:
        raise AssertionError(f"pre-head R width {model.head[0].in_features} != {R_DIM}")

    all_records = inner_train_records + other_records
    R, yhat_0 = _capture_R(model, encoded_fit + encoded_other)
    if R.shape != (len(all_records), R_DIM):
        raise AssertionError(f"captured R shape {R.shape} != ({len(all_records)}, {R_DIM})")
    target = np.asarray([float(record.y) for record in all_records], dtype=np.float64)
    subset_index = np.concatenate([inner_train_idx, inner_valid_idx, holdout_idx]).astype(np.int64)
    role = np.concatenate(
        [
            np.zeros(len(inner_train_idx), dtype=np.int64),
            np.ones(len(inner_valid_idx), dtype=np.int64),
            np.full(len(holdout_idx), 2, dtype=np.int64),
        ]
    )
    label_y = labels.set_index("subset_index").loc[subset_index, "y_stored"].to_numpy(dtype=np.float64)
    label_gate = float(np.abs(target - label_y).max())

    # forward gate: holdout predictions must equal the frozen OOF fold npz
    holdout_pred = yhat_0[len(inner_train_idx) + len(inner_valid_idx):]
    npz = _load_fold_npz(fold, seed)
    forward_gate = float(np.abs(holdout_pred - np.asarray(npz["oof_prediction"], dtype=np.float64)).max())

    # holdout R must equal the corrected centre-incidence export when present
    # (the centre-incidence audit did not spend backbone seed 1, so its export
    # only exists for seed 0; the forward gate still covers every seed).
    R_holdout = R[len(inner_train_idx) + len(inner_valid_idx):]
    ref_path = ciw._export_path(fold, seed)
    if ref_path.exists():
        exp = ciw.load_export(ref_path)
        R_ref_diff: float | None = float(
            np.abs(R_holdout.astype(np.float64) - np.asarray(exp["R"], dtype=np.float64)).max()
        )
    else:
        R_ref_diff = None

    head = _head_from_state(fold, seed)
    head_pred = _head_forward(head, R)
    head_recon = float(np.abs(head_pred - yhat_0).max())

    fingerprint = ciw._fold_fingerprint_inputs_v4(fold, seed, config)
    fingerprint["forward_gate_max_diff"] = forward_gate
    fingerprint["label_gate_max_diff"] = label_gate
    fingerprint["holdout_R_matches_centre_incidence_max_diff"] = R_ref_diff
    fingerprint["head_reconstruction_max_diff"] = head_recon
    fingerprint["n_fit"] = int(len(inner_train_idx))
    fingerprint["n_selection"] = int(len(inner_valid_idx))
    fingerprint["n_eval"] = int(len(holdout_idx))
    return {
        "subset_index": subset_index,
        "role": role,
        "target": target,
        "yhat_0": yhat_0,
        "R": R,
        "fingerprint": fingerprint,
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
        payload: dict[str, Any] = {"fingerprint": json.loads(str(data["fingerprint_json"]))}
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
            t0 = time.perf_counter()
            payload = _build_full_export(fold, seed)
            _save_cache(path, payload)
            built[f"fold{fold}_seed{seed}"] = {
                "path": str(path),
                "cached": False,
                "n": int(len(payload["subset_index"])),
                "R_dim": int(payload["R"].shape[1]),
                "seconds": time.perf_counter() - t0,
                "forward_gate_max_diff": float(payload["fingerprint"]["forward_gate_max_diff"]),
                "holdout_R_matches_centre_incidence_max_diff": (
                    None
                    if payload["fingerprint"]["holdout_R_matches_centre_incidence_max_diff"] is None
                    else float(payload["fingerprint"]["holdout_R_matches_centre_incidence_max_diff"])
                ),
            }
            rref = payload["fingerprint"]["holdout_R_matches_centre_incidence_max_diff"]
            rref_text = "n/a" if rref is None else f"{float(rref):.1e}"
            print(
                f"[cache] fold={fold} seed={seed} n={len(payload['subset_index'])} "
                f"gate_g={payload['fingerprint']['forward_gate_max_diff']:.1e} "
                f"gate_Rref={rref_text} "
                f"({time.perf_counter() - t0:.1f}s)",
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
    state = torch.load(
        FOLD_DIR / f"fold{fold}_seed{seed}_state.pt", map_location="cpu", weights_only=True
    )
    head = nn.Sequential(
        nn.Linear(R_DIM, HEAD_HIDDEN_0),
        nn.LayerNorm(HEAD_HIDDEN_0),
        nn.ReLU(),
        nn.Dropout(HEAD_DROPOUT),
        nn.Linear(HEAD_HIDDEN_0, HEAD_HIDDEN_1),
        nn.ReLU(),
        nn.Linear(HEAD_HIDDEN_1, 1),
    )
    head.load_state_dict({k[len("head."):]: v for k, v in state.items() if k.startswith("head.")})
    head.eval()
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
    role = np.asarray(payload["role"], dtype=np.int64)
    fp = payload["fingerprint"]

    labels = _load_train_labels()
    label_y = labels.set_index("subset_index").loc[subset_index, "y_stored"].to_numpy(dtype=np.float64)
    label_gate = float(np.abs(target - label_y).max())

    inner_train_idx, inner_valid_idx, holdout_idx = _fold_slices()[fold]
    for name, expected, tag in (
        ("fit", inner_train_idx, 0),
        ("selection", inner_valid_idx, 1),
        ("holdout", holdout_idx, 2),
    ):
        positions = np.flatnonzero(role == tag)
        if not np.array_equal(subset_index[positions], expected):
            raise AssertionError(f"fold {fold} {name} membership mismatch")

    head = _head_from_state(fold, seed)
    head_pred = _head_forward(head, R)
    head_recon = float(np.abs(head_pred - yhat_0).max())

    holdout_slice = slice(N_FIT + N_SELECT, N_FIT + N_SELECT + N_EVAL)
    npz = _load_fold_npz(fold, seed)
    forward_gate = float(
        np.abs(yhat_0[holdout_slice] - np.asarray(npz["oof_prediction"], dtype=np.float64)).max()
    )
    exp = ciw._export_path(fold, seed)
    if exp.exists():
        reference = ciw.load_export(exp)
        R_ref_diff: float | None = float(
            np.abs(R[holdout_slice] - np.asarray(reference["R"], dtype=np.float64)).max()
        )
    else:
        R_ref_diff = None
    return {
        "fold": int(fold),
        "backbone_seed": int(seed),
        "n_molecules": int(len(subset_index)),
        "n_fit": int((role == 0).sum()),
        "n_selection": int((role == 1).sum()),
        "n_eval": int((role == 2).sum()),
        "R_dim": int(R.shape[1]),
        "G1_target_label_max_diff": label_gate,
        "G2_split_role_membership": True,
        "G3_head_reconstruction_max_diff": head_recon,
        "G4_forward_holdout_max_diff": forward_gate,
        "G5_holdout_R_matches_centre_incidence_max_diff": R_ref_diff,
        "checkpoint_fingerprint": fp.get("checkpoint_fingerprint"),
        "config_fingerprint": fp.get("config_fingerprint"),
        "tokenizer_version": fp.get("tokenizer_version"),
        "vocabulary_fingerprint": fp.get("vocabulary_fingerprint"),
        "split_fingerprint": fp.get("split_fingerprint"),
        "passed": bool(
            R.shape[1] == R_DIM
            and label_gate <= 1e-9
            and head_recon <= HEAD_RECON_ATOL
            and forward_gate <= FORWARD_GATE_ATOL
            and (R_ref_diff is None or R_ref_diff <= R_RECON_ATOL)
        ),
    }


def run_integrity(seeds: Sequence[int] = (PRIMARY_BACKBONE_SEED,), force: bool = False) -> dict[str, Any]:
    started = time.perf_counter()
    out_path = RESULTS_DIR / "representation_integrity.json"
    existing: dict[str, Any] = {}
    if out_path.exists():
        existing = json.loads(out_path.read_text(encoding="utf-8"))
    per_fold: dict[str, Any] = dict(existing.get("per_fold", {})) if not force else {}
    checked = {int(s) for s in existing.get("seeds_checked", [])} if not force else set()
    _verify_records_vs_labels(_load_train_records(), _load_train_labels())
    for seed in seeds:
        for fold in range(K_FOLDS):
            entry = _integrity_for_fold(fold, seed)
            per_fold[f"fold{fold}_seed{seed}"] = entry
            checked.add(int(seed))
            print(
                f"[integrity] fold={fold} seed={seed} pass={entry['passed']} "
                f"g3={entry['G3_head_reconstruction_max_diff']:.1e} "
                f"g4={entry['G4_forward_holdout_max_diff']:.1e} "
                f"g5={entry['G5_holdout_R_matches_centre_incidence_max_diff']}",
                flush=True,
            )
    failures = [k for k, v in per_fold.items() if not v["passed"]]
    report = {
        "source_export_version": SOURCE_EXPORT_VERSION,
        "cache_version": CACHE_VERSION,
        "R_dim": R_DIM,
        "R_layout": R_LAYOUT,
        "seeds_checked": sorted(checked),
        "per_fold": per_fold,
        "failures": failures,
        "all_passed": len(failures) == 0,
        "global_label_integrity": (
            "PASS: frozen train records y == frozen label y_stored for all 10000 "
            "official-train molecules (official valid/test never loaded)"
        ),
        "gates": {
            "G1": "stored target == official-train label y_stored",
            "G2": "role membership == (7200 fit, 800 selection, 2000 holdout)",
            "G3": "re-feeding stored R through the frozen head == stored yhat_0",
            "G4": "holdout yhat_0 == frozen OOF fold prediction",
            "G5": "holdout R == corrected centre-incidence export R",
        },
        "seconds": time.perf_counter() - started,
    }
    _write_json_any(out_path, report)
    print(f"[integrity] all_passed={report['all_passed']} failures={failures}", flush=True)
    return report


# ---------------------------------------------------------------------------
# split manifest
# ---------------------------------------------------------------------------

def stage_splits(force: bool = False) -> dict[str, Any]:
    out_path = RESULTS_DIR / "split_manifest.json"
    if out_path.exists() and not force:
        return json.loads(out_path.read_text(encoding="utf-8"))
    manifest: dict[str, Any] = {
        "source": "existing 5-fold OOF nested structure (inner_splits.csv + outer_fold)",
        "sizes": {"head_fit": N_FIT, "head_selection": N_SELECT, "head_evaluation": N_EVAL},
        "folds": {},
    }
    for fold in range(K_FOLDS):
        inner_train_idx, inner_valid_idx, holdout_idx = _fold_slices()[fold]
        manifest["folds"][f"fold{fold}"] = {
            "n_fit": int(len(inner_train_idx)),
            "n_selection": int(len(inner_valid_idx)),
            "n_evaluation": int(len(holdout_idx)),
            "fit_sha256": _sha256_array(inner_train_idx),
            "selection_sha256": _sha256_array(inner_valid_idx),
            "evaluation_sha256": _sha256_array(holdout_idx),
            "fit_molecule_ids": [f"train:{int(i):04d}" for i in inner_train_idx[:8]],
            "selection_molecule_ids": [f"train:{int(i):04d}" for i in inner_valid_idx[:8]],
            "evaluation_molecule_ids": [f"train:{int(i):04d}" for i in holdout_idx[:8]],
        }
    _write_json_any(out_path, manifest)
    return manifest


# ---------------------------------------------------------------------------
# fit-only standardisation
# ---------------------------------------------------------------------------

def _fit_standardizer(R_fit: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


# ---------------------------------------------------------------------------
# direct heads
# ---------------------------------------------------------------------------

class GenericReader(nn.Module):
    """Direct ``R -> yhat`` head; ``hidden=()`` is a plain linear map."""

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


class FMReader(nn.Module):
    """Rank-``r`` Factorization Machine regression head (direct prediction).

        yhat = b + w^T z + sum_{i<j} <v_i, v_j> z_i z_j

    computed with the standard factorisation identity
    ``0.5 * sum_f [ (sum_i v_if z_i)^2 - sum_i v_if^2 z_i^2 ]`` so no
    ``d x d`` interaction matrix is ever materialised.  Rank is fixed at
    construction; no rank search exists in this module.
    """

    def __init__(self, in_dim: int, rank: int, init_std: float = FM_INIT_STD) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.rank = int(rank)
        self.bias = nn.Parameter(torch.zeros(1))
        self.linear = nn.Parameter(torch.zeros(self.in_dim))
        self.v = nn.Parameter(torch.empty(self.in_dim, self.rank))
        bound = 1.0 / float(self.in_dim) ** 0.5
        nn.init.uniform_(self.linear, -bound, bound)  # same scale as nn.Linear default
        nn.init.normal_(self.v, std=float(init_std))

    def linear_term(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.linear

    def interaction(self, z: torch.Tensor) -> torch.Tensor:
        xv = z @ self.v                                   # (N, r)
        sum_sq = (xv * xv).sum(dim=1)                     # (N,)
        sq_sum = ((z * z) @ (self.v * self.v)).sum(dim=1)  # (N,)
        return 0.5 * (sum_sq - sq_sum)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.bias + self.linear_term(z) + self.interaction(z)

    def forward_linear_only(self, z: torch.Tensor) -> torch.Tensor:
        """Frozen-inference interaction ablation: interaction term forced to 0."""
        return self.bias + self.linear_term(z)


def _explicit_pairwise_interaction(z: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Reference O(d^2) form of the FM interaction (tests only)."""
    A = v @ v.T
    d = z.shape[1]
    total = np.zeros(z.shape[0], dtype=np.float64)
    for i in range(d):
        for j in range(i + 1, d):
            total += A[i, j] * z[:, i] * z[:, j]
    return total


def _count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def _make_hlinear(seed: int = HEAD_SEED) -> GenericReader:
    torch.manual_seed(int(seed))
    return GenericReader(R_DIM, H_LINEAR_HIDDEN)


def _make_h1(seed: int = HEAD_SEED) -> GenericReader:
    torch.manual_seed(int(seed))
    return GenericReader(R_DIM, H1_HIDDEN)


def _make_h2(seed: int = HEAD_SEED) -> GenericReader:
    torch.manual_seed(int(seed))
    return GenericReader(R_DIM, H2_HIDDEN)


def _make_fm(seed: int = HEAD_SEED) -> FMReader:
    torch.manual_seed(int(seed))
    return FMReader(R_DIM, FM_RANK)


def _parameter_accounting() -> dict[str, Any]:
    hlin = _make_hlinear(0)
    h1 = _make_h1(0)
    h2 = _make_h2(0)
    fm = _make_fm(0)
    p_h1, p_fm = _count_parameters(h1), _count_parameters(fm)
    mismatch = abs(p_h1 - p_fm) / max(p_fm, 1)
    return {
        "R_dim": int(R_DIM),
        "H0_architecture": "frozen original graph head 302 -> 64 -> 32 -> 1 (jointly trained)",
        "Hlinear_architecture": f"{R_DIM} -> 1",
        "Hlinear_params": _count_parameters(hlin),
        "H1_architecture": f"{R_DIM} -> " + " -> ".join(str(h) for h in H1_HIDDEN) + " -> 1",
        "H1_params": int(p_h1),
        "H2_architecture": f"{R_DIM} -> " + " -> ".join(str(h) for h in H2_HIDDEN) + " -> 1",
        "H2_params": _count_parameters(h2),
        "FM_architecture": f"FM(rank={FM_RANK}): bias + linear({R_DIM}) + {R_DIM}x{FM_RANK} factors",
        "FM_rank": int(FM_RANK),
        "FM_params": int(p_fm),
        "FM_linear_params": int(R_DIM),
        "FM_interaction_params": int(R_DIM * FM_RANK),
        "FM_bias_params": 1,
        "H1_vs_FM_abs_mismatch": float(mismatch),
        "parameter_match_ok": bool(mismatch <= PARAM_MATCH_TOL),
        "parameter_match_tolerance": float(PARAM_MATCH_TOL),
    }


def write_head_spec(accounting: Mapping[str, Any] | None = None) -> dict[str, Any]:
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
            "fit_scope": "head-fit 7200 molecules of each fold",
            "definition": "z_j = (R_j - mean_j) / max(std_j, eps), z_j = 0 if std_j < eps",
            "eps": float(STANDARDIZE_EPS),
            "recomputed_on_selection_or_evaluation": False,
        },
        "objective": {
            "loss": "L1 / MAE",
            "direct": True,
            "residual_yhat_0_input": False,
            "optimizer": "Adam",
            "lr": float(ADAPTER_LR),
            "weight_decay": float(ADAPTER_WEIGHT_DECAY),
            "batch": f"minibatch {ADAPTER_BATCH_SIZE} (deterministic seeded order)",
            "horizon": f"{HORIZON_FAST} or {HORIZON_SLOW} chosen by the fold0/seed0 convergence trace",
        },
        "heads": {
            "H0": "frozen original graph head prediction (jointly-trained reference)",
            "Hlinear": "direct 302 -> 1 L1 (descriptive, not in the GO gate)",
            "H1": "direct generic ReLU MLP, parameter-matched to FM",
            "H2": "direct stronger generic ReLU MLP (not matched)",
            "FM": "direct rank-4 Factorization Machine (low-rank cross-coordinate interaction)",
        },
        "parameter_accounting": accounting,
        "catboost": {
            "available": _catboost_available(),
            "preset": CATBOOST_PRESET,
            "role": "secondary non-neural tree-style reference; never in the FM GO gate",
        },
        "forbidden": [
            "FM rank sweep (rank 8 / 16 / learned)",
            "DeepFM / Field-aware FM / CrossNet / polynomial network / bilinear head",
            "residual adapter yhat = yhat_0 + delta(R) as the primary experiment",
            "MSE / Huber / quantile / weighted MAE",
            "LR / optimizer / weight-decay sweep",
            "official valid / official test access",
        ],
    }
    _write_json_any(RESULTS_DIR / "head_spec.json", spec)
    return spec


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

def _iter_minibatches(
    z: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    epoch: int,
    seed: int = HEAD_SEED,
):
    """Deterministic seeded mini-batch order (same order for every head)."""
    n = int(z.shape[0])
    generator = torch.Generator().manual_seed(int(seed) * 1000003 + int(epoch))
    permutation = torch.randperm(n, generator=generator)
    for start in range(0, n, int(batch_size)):
        index = permutation[start: start + int(batch_size)]
        yield z[index], y[index]


def _train_head(
    model: nn.Module,
    z_fit: torch.Tensor,
    y_fit: torch.Tensor,
    z_sel: torch.Tensor,
    y_sel: torch.Tensor,
    *,
    epochs: int,
    lr: float = ADAPTER_LR,
    weight_decay: float = ADAPTER_WEIGHT_DECAY,
    batch_size: int = ADAPTER_BATCH_SIZE,
) -> dict[str, Any]:
    """Deterministic mini-batch L1/Adam training with best-selection checkpoint."""
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(lr), weight_decay=float(weight_decay)
    )
    model.eval()
    with torch.no_grad():
        initial_output = model(z_fit)
        initial_output_std = float(initial_output.std().item()) if initial_output.numel() > 1 else 0.0
    initial_grad_norm: float | None = None
    best_state = copy.deepcopy(model.state_dict())
    best_selection = float("inf")
    for epoch in range(int(epochs)):
        model.train()
        for batch_z, batch_y in _iter_minibatches(z_fit, y_fit, batch_size, epoch):
            optimizer.zero_grad()
            loss = (model(batch_z) - batch_y).abs().mean()
            loss.backward()
            if initial_grad_norm is None:
                total = 0.0
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        total += float(parameter.grad.detach().pow(2).sum().item())
                initial_grad_norm = float(total ** 0.5)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            selection_mae = float((model(z_sel) - y_sel).abs().mean().item())
        if selection_mae < best_selection - 1e-9:
            best_selection = selection_mae
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return {
        "best_selection_mae": best_selection,
        "initial_output_std": initial_output_std,
        "initial_grad_norm": initial_grad_norm,
    }


def _train_head_with_trace(
    model: nn.Module,
    z_fit: torch.Tensor,
    y_fit: torch.Tensor,
    z_sel: torch.Tensor,
    y_sel: torch.Tensor,
    trace_epochs: Sequence[int],
    max_epochs: int,
    batch_size: int = ADAPTER_BATCH_SIZE,
) -> dict[int, float]:
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(ADAPTER_LR), weight_decay=float(ADAPTER_WEIGHT_DECAY)
    )
    trace: dict[int, float] = {}
    for epoch in range(int(max_epochs)):
        model.train()
        for batch_z, batch_y in _iter_minibatches(z_fit, y_fit, batch_size, epoch):
            optimizer.zero_grad()
            loss = (model(batch_z) - batch_y).abs().mean()
            loss.backward()
            optimizer.step()
        current = epoch + 1
        if current in trace_epochs:
            model.eval()
            with torch.no_grad():
                trace[current] = float((model(z_sel) - y_sel).abs().mean().item())
    return trace


# ---------------------------------------------------------------------------
# per-fold data assembly
# ---------------------------------------------------------------------------

def _fold_tensors(fold: int, backbone_seed: int) -> dict[str, Any]:
    payload = load_cache(fold, backbone_seed)
    R = np.asarray(payload["R"], dtype=np.float64)
    y = np.asarray(payload["target"], dtype=np.float64)
    yhat_0 = np.asarray(payload["yhat_0"], dtype=np.float64)
    subset_index = np.asarray(payload["subset_index"], dtype=np.int64)
    role = np.asarray(payload["role"], dtype=np.int64)
    fit_pos = np.flatnonzero(role == 0)
    sel_pos = np.flatnonzero(role == 1)
    eva_pos = np.flatnonzero(role == 2)
    mean, scale, degenerate = _fit_standardizer(R[fit_pos])
    z = _standardize(R, mean, scale, degenerate)
    return {
        "subset_index": subset_index,
        "R": R,
        "y": y,
        "yhat_0": yhat_0,
        "role": role,
        "fit_pos": fit_pos,
        "sel_pos": sel_pos,
        "eva_pos": eva_pos,
        "mean": mean,
        "scale": scale,
        "degenerate": degenerate,
        "z": z,
    }


def _tensors(data: Mapping[str, Any], key_pos: str) -> dict[str, torch.Tensor]:
    pos = data[key_pos]
    return {
        "z": torch.tensor(data["z"][pos], dtype=torch.float32),
        "y": torch.tensor(data["y"][pos], dtype=torch.float32),
    }


# ---------------------------------------------------------------------------
# convergence trace
# ---------------------------------------------------------------------------

def run_convergence_trace(force: bool = False) -> dict[str, Any]:
    out_csv = RESULTS_DIR / "convergence_trace.csv"
    out_json = RESULTS_DIR / "convergence_trace.json"
    if out_csv.exists() and out_json.exists() and not force:
        return json.loads(out_json.read_text(encoding="utf-8"))
    data = _fold_tensors(0, PRIMARY_BACKBONE_SEED)
    fit = _tensors(data, "fit_pos")
    sel = _tensors(data, "sel_pos")
    readers: dict[str, nn.Module] = {
        "Hlinear": _make_hlinear(),
        "H1": _make_h1(),
        "H2": _make_h2(),
        "FM": _make_fm(),
    }
    rows: list[dict[str, Any]] = []
    traces: dict[str, Any] = {}
    for name, model in readers.items():
        trace = _train_head_with_trace(
            model,
            fit["z"], fit["y"], sel["z"], sel["y"],
            trace_epochs=CONVERGENCE_EPOCHS,
            max_epochs=max(CONVERGENCE_EPOCHS),
        )
        traces[name] = {str(k): float(v) for k, v in trace.items()}
        for epoch in CONVERGENCE_EPOCHS:
            rows.append({"reader": name, "epoch": int(epoch), "selection_mae": float(trace[epoch])})
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    abs_400_800 = {name: abs(traces[name][str(HORIZON_FAST)] - traces[name][str(HORIZON_SLOW)]) for name in readers}
    still_improving = any(
        traces[name][str(HORIZON_SLOW)] < traces[name][str(HORIZON_FAST)] - HORIZON_TOL
        for name in readers
    )
    horizon = HORIZON_SLOW if still_improving else HORIZON_FAST
    detail = {
        "fold": 0,
        "backbone_seed": PRIMARY_BACKBONE_SEED,
        "trace_epochs": list(CONVERGENCE_EPOCHS),
        "traces": traces,
        "abs_selection_mae_400_minus_800": {k: float(v) for k, v in abs_400_800.items()},
        "max_abs_400_800": float(max(abs_400_800.values())),
        "still_improving_at_800": bool(still_improving),
        "horizon_tolerance": float(HORIZON_TOL),
        "selected_horizon": int(horizon),
        "rule": (
            f"H={HORIZON_SLOW} if any traced head still improves by >{HORIZON_TOL:g} from "
            f"epoch {HORIZON_FAST} to {HORIZON_SLOW}; otherwise H={HORIZON_FAST}.  All "
            "heads share the fixed horizon with best-selection checkpointing; no sweep"
        ),
    }
    _write_json_any(out_json, detail)
    print(
        f"[convergence] selected_horizon={horizon} max|400-800|={detail['max_abs_400_800']:.2e}",
        flush=True,
    )
    return detail


def _selected_horizon() -> int:
    path = RESULTS_DIR / "convergence_trace.json"
    if path.exists():
        return int(json.loads(path.read_text(encoding="utf-8"))["selected_horizon"])
    return HORIZON_FAST


# ---------------------------------------------------------------------------
# stage runs
# ---------------------------------------------------------------------------

def _rare_ratio_for(fold: int, ids: np.ndarray) -> np.ndarray:
    rarity = pd.read_csv(ciw.RARITY_CSV).set_index("subset_index")
    return rarity.loc[ids, "rare_le5_ratio"].to_numpy(dtype=np.float64)


def _run_stage_for_fold(
    fold: int,
    backbone_seed: int,
    adapter_seed: int,
    horizon: int,
) -> dict[str, Any]:
    data = _fold_tensors(fold, backbone_seed)
    fit = _tensors(data, "fit_pos")
    sel = _tensors(data, "sel_pos")
    eva = _tensors(data, "eva_pos")

    hlin = _make_hlinear(adapter_seed)
    h1 = _make_h1(adapter_seed)
    h2 = _make_h2(adapter_seed)
    fm = _make_fm(adapter_seed)

    info_hlin = _train_head(hlin, fit["z"], fit["y"], sel["z"], sel["y"], epochs=horizon)
    info_h1 = _train_head(h1, fit["z"], fit["y"], sel["z"], sel["y"], epochs=horizon)
    info_h2 = _train_head(h2, fit["z"], fit["y"], sel["z"], sel["y"], epochs=horizon)
    info_fm = _train_head(fm, fit["z"], fit["y"], sel["z"], sel["y"], epochs=horizon)

    with torch.no_grad():
        p_hlin = hlin(eva["z"]).numpy()
        p_h1 = h1(eva["z"]).numpy()
        p_h2 = h2(eva["z"]).numpy()
        p_fm = fm(eva["z"]).numpy()
        p_fm_linear = fm.forward_linear_only(eva["z"]).numpy()
        fm_bias = float(fm.bias.item())
        fm_linear = fm.linear_term(eva["z"]).numpy()
        fm_interaction = fm.interaction(eva["z"]).numpy()
        fm_final_output_std = float(p_fm.std())

    y = data["y"][data["eva_pos"]]
    yhat0 = data["yhat_0"][data["eva_pos"]]
    mae_h0 = float(np.abs(y - yhat0).mean())
    mae_hlin = float(np.abs(y - p_hlin).mean())
    mae_h1 = float(np.abs(y - p_h1).mean())
    mae_h2 = float(np.abs(y - p_h2).mean())
    mae_fm = float(np.abs(y - p_fm).mean())
    mae_fm_linear_only = float(np.abs(y - p_fm_linear).mean())

    interaction_scale = float(np.std(fm_interaction))
    prediction_scale = float(np.std(p_fm))
    rare_ratio = _rare_ratio_for(fold, data["subset_index"][data["eva_pos"]])
    bulk_threshold = float(pd.Series(rare_ratio).quantile(0.80))
    bulk_mask = rare_ratio < bulk_threshold
    if bulk_mask.sum() >= 10:
        bulk_fm_minus_h1 = float(np.abs(y - p_fm)[bulk_mask].mean() - np.abs(y - p_h1)[bulk_mask].mean())
        bulk_fm_minus_h2 = float(np.abs(y - p_fm)[bulk_mask].mean() - np.abs(y - p_h2)[bulk_mask].mean())
    else:
        bulk_fm_minus_h1 = bulk_fm_minus_h2 = float("nan")

    return {
        "fold": int(fold),
        "backbone_seed": int(backbone_seed),
        "adapter_seed": int(adapter_seed),
        "horizon": int(horizon),
        "n_fit": int(len(data["fit_pos"])),
        "n_selection": int(len(data["sel_pos"])),
        "n_eval": int(len(data["eva_pos"])),
        "n_degenerate_coordinates": int(data["degenerate"].sum()),
        "mae_h0": mae_h0,
        "mae_hlin": mae_hlin,
        "mae_h1": mae_h1,
        "mae_h2": mae_h2,
        "mae_fm": mae_fm,
        "mae_fm_linear_only": mae_fm_linear_only,
        "delta_small": mae_h1 - mae_fm,
        "delta_strong": mae_h2 - mae_fm,
        "delta_orig": mae_h0 - mae_fm,
        "fm_minus_h2": mae_fm - mae_h2,
        "delta_h1_h0": mae_h0 - mae_h1,
        "delta_h2_h0": mae_h0 - mae_h2,
        "bulk_fm_minus_h1": bulk_fm_minus_h1,
        "bulk_fm_minus_h2": bulk_fm_minus_h2,
        "h1_params": _count_parameters(h1),
        "h2_params": _count_parameters(h2),
        "fm_params": _count_parameters(fm),
        "hlinear_params": _count_parameters(hlin),
        "h1_selection_mae": info_h1["best_selection_mae"],
        "h2_selection_mae": info_h2["best_selection_mae"],
        "fm_selection_mae": info_fm["best_selection_mae"],
        "hlinear_selection_mae": info_hlin["best_selection_mae"],
        "fm_initial_output_std": info_fm["initial_output_std"],
        "fm_initial_grad_norm": info_fm["initial_grad_norm"],
        "fm_final_output_std": fm_final_output_std,
        "hlin_initial_output_std": info_hlin["initial_output_std"],
        "h1_initial_output_std": info_h1["initial_output_std"],
        "h2_initial_output_std": info_h2["initial_output_std"],
        "hlin_initial_grad_norm": info_hlin["initial_grad_norm"],
        "h1_initial_grad_norm": info_h1["initial_grad_norm"],
        "h2_initial_grad_norm": info_h2["initial_grad_norm"],
        "hlin_final_output_std": float(p_hlin.std()),
        "h1_final_output_std": float(p_h1.std()),
        "h2_final_output_std": float(p_h2.std()),
        "fm_interaction_std": interaction_scale,
        "fm_prediction_std": prediction_scale,
        "fm_interaction_to_prediction_ratio": float(interaction_scale / max(prediction_scale, 1e-12)),
        "fm_mean_abs_interaction": float(np.abs(fm_interaction).mean()),
        "fm_interaction_sensitivity": float(np.abs(p_fm - p_fm_linear).max()),
        "fm_bias": fm_bias,
        "_molecules": {
            "subset_index": data["subset_index"][data["eva_pos"]],
            "fold": np.full(len(data["eva_pos"]), fold, dtype=np.int64),
            "err_h0": np.abs(y - yhat0),
            "err_hlin": np.abs(y - p_hlin),
            "err_h1": np.abs(y - p_h1),
            "err_h2": np.abs(y - p_h2),
            "err_fm": np.abs(y - p_fm),
            "err_fm_linear_only": np.abs(y - p_fm_linear),
        },
        "_fm_components": {
            "bias": np.full(len(y), fm_bias),
            "linear": fm_linear,
            "interaction": fm_interaction,
            "y": y,
        },
    }


def _rows_to_frame(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame([{k: v for k, v in row.items() if not k.startswith("_")} for row in rows])


def _aggregate_stage(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    frame = _rows_to_frame(rows)
    dsmall = frame["delta_small"].to_numpy(dtype=np.float64)
    dstrong = frame["delta_strong"].to_numpy(dtype=np.float64)
    dorig = frame["delta_orig"].to_numpy(dtype=np.float64)
    fm_minus_h2 = frame["fm_minus_h2"].to_numpy(dtype=np.float64)
    bulk = frame["bulk_fm_minus_h1"].to_numpy(dtype=np.float64)
    bulk = bulk[np.isfinite(bulk)]
    return {
        "n_folds": int(len(frame)),
        "mean_h0": float(frame["mae_h0"].mean()),
        "mean_hlin": float(frame["mae_hlin"].mean()),
        "mean_h1": float(frame["mae_h1"].mean()),
        "mean_h2": float(frame["mae_h2"].mean()),
        "mean_fm": float(frame["mae_fm"].mean()),
        "mean_fm_linear_only": float(frame["mae_fm_linear_only"].mean()),
        "mean_delta_small": float(dsmall.mean()),
        "median_delta_small": float(np.median(dsmall)),
        "mean_delta_strong": float(dstrong.mean()),
        "median_delta_strong": float(np.median(dstrong)),
        "mean_delta_orig": float(dorig.mean()),
        "mean_fm_minus_h2": float(fm_minus_h2.mean()),
        "positive_folds_delta_small": int(np.sum(dsmall > 0)),
        "positive_folds_delta_strong": int(np.sum(dstrong > 0)),
        "per_fold_delta_small": dsmall.tolist(),
        "per_fold_delta_strong": dstrong.tolist(),
        "per_fold_delta_orig": dorig.tolist(),
        "mean_delta_h1_h0": float(frame["delta_h1_h0"].mean()),
        "mean_delta_h2_h0": float(frame["delta_h2_h0"].mean()),
        "mean_bulk_fm_minus_h1": float(bulk.mean()) if bulk.size else None,
        "max_bulk_fm_minus_h1": float(bulk.max()) if bulk.size else None,
        "h1_params": int(frame["h1_params"].iloc[0]),
        "h2_params": int(frame["h2_params"].iloc[0]),
        "fm_params": int(frame["fm_params"].iloc[0]),
        "hlinear_params": int(frame["hlinear_params"].iloc[0]),
        "fm_interaction_sensitivity_min": float(frame["fm_interaction_sensitivity"].min()),
        "fm_initial_output_std": float(frame["fm_initial_output_std"].iloc[0]),
        "fm_final_output_std": float(frame["fm_final_output_std"].mean()),
        "h1_initial_output_std": float(frame["h1_initial_output_std"].iloc[0]),
        "h1_initial_grad_norm": float(frame["h1_initial_grad_norm"].iloc[0]),
        "h1_final_output_std": float(frame["h1_final_output_std"].mean()),
        "h2_initial_output_std": float(frame["h2_initial_output_std"].iloc[0]),
        "h2_initial_grad_norm": float(frame["h2_initial_grad_norm"].iloc[0]),
        "h2_final_output_std": float(frame["h2_final_output_std"].mean()),
    }


def run_stage(
    backbone_seed: int,
    adapter_seed: int,
    tag: str,
    force: bool = False,
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
                "delta_orig": cached["delta_orig"],
                "interaction_ablation": cached["interaction_ablation"],
            },
            "cached": True,
        }
    horizon = _selected_horizon()
    rows = [
        _run_stage_for_fold(fold, backbone_seed, adapter_seed, horizon)
        for fold in range(K_FOLDS)
    ]
    _rows_to_frame(rows).to_csv(out_csv, index=False)
    molecules = [row["_molecules"] for row in rows]
    fold_ids = np.concatenate([m["fold"] for m in molecules])
    dsmall = np.concatenate([m["err_h1"] - m["err_fm"] for m in molecules])
    dstrong = np.concatenate([m["err_h2"] - m["err_fm"] for m in molecules])
    dorig = np.concatenate([m["err_h0"] - m["err_fm"] for m in molecules])
    dablation = np.concatenate([m["err_fm_linear_only"] - m["err_fm"] for m in molecules])
    components = [row["_fm_components"] for row in rows]
    interaction = np.concatenate([c["interaction"] for c in components])
    linear = np.concatenate([c["linear"] for c in components])
    bootstrap = {
        "backbone_seed": int(backbone_seed),
        "adapter_seed": int(adapter_seed),
        "horizon": int(horizon),
        "delta_small": ciw._stratified_bootstrap(dsmall, fold_ids),
        "delta_strong": ciw._stratified_bootstrap(dstrong, fold_ids),
        "delta_orig": ciw._stratified_bootstrap(dorig, fold_ids),
        "interaction_ablation": ciw._stratified_bootstrap(dablation, fold_ids),
        "aggregate": _aggregate_stage(rows),
        "fm_interaction_contribution": {
            "std": float(np.std(interaction)),
            "mean_abs": float(np.abs(interaction).mean()),
            "linear_std": float(np.std(linear)),
            "interaction_to_linear_std_ratio": float(np.std(interaction) / max(np.std(linear), 1e-12)),
        },
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
            "delta_orig": bootstrap["delta_orig"],
            "interaction_ablation": bootstrap["interaction_ablation"],
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
    fm_minus_h2 = agg["mean_fm_minus_h2"]

    if mean_dsmall <= S1_NGO_DSMALL or folds_positive <= S1_NGO_FOLDS:
        base = "STAGE1_CLEAR_NO_GO"
    elif mean_dsmall >= S1_EFF_DSMALL and folds_positive >= S1_EFF_FOLDS:
        if fm_minus_h2 <= S1_STRONG_FM_MINUS_H2_MAX:
            base = "STAGE1_STRONG_ADVANCE"
        elif fm_minus_h2 <= S1_EFF_FM_MINUS_H2_MAX:
            base = "STAGE1_EFFICIENCY_ADVANCE"
        else:
            base = "STAGE1_BORDERLINE"
    elif S1_NGO_DSMALL < mean_dsmall < S1_EFF_DSMALL and folds_positive >= S1_BORDERLINE_FOLDS:
        base = "STAGE1_BORDERLINE"
    else:
        base = "STAGE1_INCONCLUSIVE"

    if base != "STAGE1_BORDERLINE" or stage1b is None:
        return base

    agg_b = stage1b["aggregate"]
    pooled_dsmall = float(np.mean([mean_dsmall, agg_b["mean_delta_small"]]))
    pooled_fm_minus_h2 = float(np.mean([fm_minus_h2, agg_b["mean_fm_minus_h2"]]))
    per_fold = np.mean(
        [agg["per_fold_delta_small"], agg_b["per_fold_delta_small"]], axis=0
    )
    pooled_positive = int(np.sum(per_fold > 0))
    consistent = (
        np.sign(agg_b["mean_delta_small"]) == np.sign(mean_dsmall)
        and agg_b["positive_folds_delta_small"] >= S1_BORDERLINE_FOLDS
    )
    if not consistent:
        return "STAGE1_INCONCLUSIVE"
    if pooled_dsmall >= S1_EFF_DSMALL and pooled_positive >= S1_EFF_FOLDS:
        if pooled_fm_minus_h2 <= S1_STRONG_FM_MINUS_H2_MAX:
            return "STAGE1_STRONG_ADVANCE"
        if pooled_fm_minus_h2 <= S1_EFF_FM_MINUS_H2_MAX:
            return "STAGE1_EFFICIENCY_ADVANCE"
    return "STAGE1_INCONCLUSIVE"


def _boot_small(stage: Mapping[str, Any]) -> Mapping[str, Any]:
    if "bootstrap_detail" in stage:
        return stage["bootstrap_detail"]["delta_small"]
    return stage["delta_small"]


def _final_decision(
    stage1: Mapping[str, Any],
    stage2: Mapping[str, Any] | None,
    stage1b: Mapping[str, Any] | None,
    mechanism: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    stages = [stage1] + ([stage1b] if stage1b is not None else []) + ([stage2] if stage2 is not None else [])
    per_fold_dsmall = np.mean([s["aggregate"]["per_fold_delta_small"] for s in stages], axis=0)
    pooled_dsmall = float(np.mean([s["aggregate"]["mean_delta_small"] for s in stages]))
    pooled_fm_minus_h2 = float(np.mean([s["aggregate"]["mean_fm_minus_h2"] for s in stages]))
    backbones = [stage1] + ([stage2] if stage2 is not None else [])
    backbone_consistent = all(s["aggregate"]["mean_delta_small"] > 0 for s in backbones)
    backbone_consistent_strong = all(s["aggregate"]["mean_fm_minus_h2"] <= 0.0 for s in backbones)
    folds_positive = int(np.sum(per_fold_dsmall > 0))

    boot = [_boot_small(s) for s in stages]
    ci_low_positive = all(b["ci95_low"] > 0 for b in boot)
    interaction_alive = all(s["aggregate"]["fm_interaction_sensitivity_min"] > MECHANISM_INTERACTION_MIN for s in stages)

    criteria = {
        "pooled_delta_small_ge_0.0025": pooled_dsmall >= FINAL_GO_POOLED,
        "backbone_consistent_positive": bool(backbone_consistent),
        "fold_averaged_at_least_4_of_5_positive": folds_positive >= FINAL_GO_FOLDS,
        "paired_delta_small_ci_lower_positive": bool(ci_low_positive),
        "fm_interaction_alive": bool(interaction_alive),
    }
    if stage2 is not None and all(criteria.values()):
        if pooled_fm_minus_h2 <= 0.0 and backbone_consistent_strong:
            verdict = "GO_STRONG_LOW_RANK_INTERACTION_HEAD"
            case = "Case A"
        else:
            verdict = "GO_LOW_RANK_INTERACTION_PARAMETER_EFFICIENCY"
            case = "Case B"
    elif stage2 is None and stage1["aggregate"]["mean_delta_small"] <= S1_NGO_DSMALL:
        verdict = "NO_GO"
        case = "Case C"
    else:
        verdict = "INCONCLUSIVE"
        case = "Case E"

    if verdict == "NO_GO":
        summary = (
            "Explicit rank-4 low-rank cross-coordinate interaction (FM) does not beat the "
            "same-budget generic ReLU MLP H1 on the frozen compact-v4 representation; the "
            "pre-registered Stage-1 CLEAR NO-GO clause fires."
        )
        action = (
            "STOP.  No rank 8/16/learned, no DeepFM / field-aware FM / CrossNet / "
            "polynomial network / bilinear head, no end-to-end FM replacement.  The "
            "conclusion is scoped to explicit low-rank pairwise interaction on this frozen R."
        )
    elif verdict.startswith("GO"):
        summary = (
            "Explicit rank-4 low-rank cross-coordinate interaction shows a stable "
            "parameter-efficiency (and possibly strong) advantage over the same-budget "
            "generic MLP on the frozen compact-v4 representation across two backbones."
        )
        action = (
            "Authorize a pre-registered end-to-end FM head replacement (seed0 then seed1); "
            "do not sweep rank."
        )
    else:
        summary = "Effect is unstable / small; do not expand the budget."
        action = "STOP at the pre-registered INCONCLUSIVE boundary."
    return {
        "case": case,
        "verdict": verdict,
        "summary": summary,
        "action": action,
        "scope_caveat": (
            "Frozen-representation screen on official TRAIN molecules only.  R was produced "
            "by a backbone jointly trained with the original MLP head, so it may be "
            "MLP-friendly; a positive frozen result is strong evidence, a negative frozen "
            "result closes the explicit low-rank interaction hypothesis on this R but does "
            "not prove end-to-end FM is useless."
        ),
        "pooled_delta_small": pooled_dsmall,
        "pooled_fm_minus_h2": pooled_fm_minus_h2,
        "per_fold_delta_small": per_fold_dsmall.tolist(),
        "folds_delta_small_positive": folds_positive,
        "criteria": criteria,
        "delta_small_ci95": [[b["ci95_low"], b["ci95_high"]] for b in boot],
        "stage1_aggregate": stage1["aggregate"],
        "stage1b_aggregate": None if stage1b is None else stage1b["aggregate"],
        "stage2_aggregate": None if stage2 is None else stage2["aggregate"],
        "mechanism": None if mechanism is None else mechanism,
    }


# ---------------------------------------------------------------------------
# catboost secondary reference
# ---------------------------------------------------------------------------

def _catboost_available() -> bool:
    try:
        import catboost  # noqa: F401

        return True
    except Exception:
        return False


def catboost_status() -> dict[str, Any]:
    available = _catboost_available()
    status = {
        "available": bool(available),
        "preset": CATBOOST_PRESET,
        "role": "secondary non-neural tree-style reference; never in the FM GO gate",
        "input_spec": "standardized z (fit-only, 7200) -- the exact same matrix as the neural heads",
        "extra_features": [],
        "note": (
            "catboost is not installed in this environment; recorded as unavailable and the "
            "audit continues.  No environment change is made for a secondary control."
            if not available
            else "catboost is available; the pre-registered MAE preset is used without search."
        ),
    }
    _write_json_any(RESULTS_DIR / "catboost_secondary_status.json", status)
    return status


def run_catboost_secondary(force: bool = False) -> dict[str, Any]:
    out_csv = RESULTS_DIR / "catboost_secondary_results.csv"
    status = catboost_status()
    if not status["available"]:
        return status
    if out_csv.exists() and not force:
        frame = pd.read_csv(out_csv)
        return {"available": True, "path": str(out_csv), "mean_mae": float(frame["mae_catboost"].mean())}
    from catboost import CatBoostRegressor

    rows: list[dict[str, Any]] = []
    for fold in range(K_FOLDS):
        data = _fold_tensors(fold, PRIMARY_BACKBONE_SEED)
        fit = _tensors(data, "fit_pos")
        sel = _tensors(data, "sel_pos")
        eva = _tensors(data, "eva_pos")
        model = CatBoostRegressor(**CATBOOST_PRESET)
        model.fit(
            fit["z"].numpy(),
            fit["y"].numpy(),
            eval_set=(sel["z"].numpy(), sel["y"].numpy()),
        )
        pred = model.predict(eva["z"].numpy())
        mae_catboost = float(np.abs(eva["y"].numpy() - pred).mean())
        rows.append({"fold": fold, "mae_catboost": mae_catboost, "best_iteration": int(model.get_best_iteration())})
    frame = pd.DataFrame(rows)
    frame.to_csv(out_csv, index=False)
    return {"available": True, "path": str(out_csv), "mean_mae": float(frame["mae_catboost"].mean())}


# ---------------------------------------------------------------------------
# mechanism analysis (advance only)
# ---------------------------------------------------------------------------

def run_mechanism_analysis(force: bool = False) -> dict[str, Any]:
    out_path = RESULTS_DIR / "fm_interaction_contributions.csv"
    ablation_path = RESULTS_DIR / "fm_interaction_ablation.csv"
    if out_path.exists() and not force:
        return json.loads((RESULTS_DIR / "fm_mechanism_summary.json").read_text(encoding="utf-8"))
    horizon = _selected_horizon()
    rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    for fold in range(K_FOLDS):
        row = _run_stage_for_fold(fold, PRIMARY_BACKBONE_SEED, 0, horizon)
        components = row["_fm_components"]
        interaction = components["interaction"]
        linear = components["linear"]
        bias = components["bias"]
        total = bias + linear + interaction
        for i in range(len(interaction)):
            rows.append(
                {
                    "fold": int(fold),
                    "molecule": i,
                    "bias": float(bias[i]),
                    "linear": float(linear[i]),
                    "interaction": float(interaction[i]),
                    "prediction": float(total[i]),
                }
            )
        ablation_rows.append(
            {
                "fold": int(fold),
                "mae_fm": row["mae_fm"],
                "mae_fm_linear_only": row["mae_fm_linear_only"],
                "ablation_delta": row["mae_fm_linear_only"] - row["mae_fm"],
            }
        )
    pd.DataFrame(rows).to_csv(out_path, index=False)
    pd.DataFrame(ablation_rows).to_csv(ablation_path, index=False)
    interaction = np.asarray([r["interaction"] for r in rows], dtype=np.float64)
    linear = np.asarray([r["linear"] for r in rows], dtype=np.float64)
    prediction = np.asarray([r["prediction"] for r in rows], dtype=np.float64)
    ablation_delta = np.asarray([r["ablation_delta"] for r in ablation_rows], dtype=np.float64)
    summary = {
        "n_evaluation_molecules": int(len(rows)),
        "std_linear_contribution": float(np.std(linear)),
        "std_interaction_contribution": float(np.std(interaction)),
        "mean_abs_interaction_contribution": float(np.abs(interaction).mean()),
        "interaction_to_prediction_scale": float(np.std(interaction) / max(np.std(prediction), 1e-12)),
        "interaction_to_linear_scale": float(np.std(interaction) / max(np.std(linear), 1e-12)),
        "mean_interaction_ablation_mae_increase": float(ablation_delta.mean()),
        "interaction_ablation_positive_folds": int(np.sum(ablation_delta > 0)),
        "ablation_per_fold_delta": ablation_delta.tolist(),
        "interaction_mechanism_alive": bool(
            float(np.std(interaction)) > MECHANISM_INTERACTION_MIN and float(ablation_delta.mean()) > 0
        ),
        "note": (
            "Contribution decomposition of the trained FM: yhat = bias + linear(z) + "
            "interaction(z).  The ablation forces V=0 (interaction=0) at inference without "
            "retraining; a positive ablation delta means the interaction term carries signal."
        ),
    }
    _write_json_any(RESULTS_DIR / "fm_mechanism_summary.json", summary)
    return summary


# ---------------------------------------------------------------------------
# end-to-end scaffold (authorized only after a frozen GO)
# ---------------------------------------------------------------------------

def _copy_shared_upstream(source: nn.Module, target: nn.Module, head_prefix: str = "head.") -> dict[str, Any]:
    """Tensor-by-tensor copy of every shared upstream parameter/buffer.

    The graph head (``head_prefix``) is deliberately left untouched.  Returns a
    hash audit proving every shared tensor is identical after the copy.
    """
    source_state = source.state_dict()
    target_state = target.state_dict()
    shared_keys = [k for k in source_state if not k.startswith(head_prefix)]
    missing = [k for k in shared_keys if k not in target_state]
    if missing:
        raise AssertionError(f"shared upstream keys missing in target: {missing}")
    with torch.no_grad():
        for key in shared_keys:
            target_state[key].copy_(source_state[key])
    audit: dict[str, str] = {}
    for key in shared_keys:
        audit[key] = hashlib.sha256(
            np.ascontiguousarray(target.state_dict()[key].detach().cpu().numpy()).tobytes()
        ).hexdigest()
    return {"n_shared_tensors": len(shared_keys), "shared_tensor_hashes": audit}


def e2e_authorized(decision: Mapping[str, Any]) -> bool:
    return str(decision.get("verdict", "")).startswith("GO")


# ---------------------------------------------------------------------------
# final decision + pooled outputs
# ---------------------------------------------------------------------------

def run_final_decision(force: bool = False) -> dict[str, Any]:
    stage1 = json.loads((RESULTS_DIR / "stage1_bootstrap.json").read_text(encoding="utf-8"))
    stage1b_path = RESULTS_DIR / "stage1b_init_bootstrap.json"
    stage2_path = RESULTS_DIR / "stage2_bootstrap.json"
    mechanism_path = RESULTS_DIR / "fm_mechanism_summary.json"
    stage1b = json.loads(stage1b_path.read_text(encoding="utf-8")) if stage1b_path.exists() else None
    stage2 = json.loads(stage2_path.read_text(encoding="utf-8")) if stage2_path.exists() else None
    mechanism = json.loads(mechanism_path.read_text(encoding="utf-8")) if mechanism_path.exists() else None

    accounting = _parameter_accounting()
    _write_json_any(RESULTS_DIR / "parameter_counts.json", accounting)

    decision = _final_decision(stage1, stage2, stage1b, mechanism)
    decision["stage1_verdict"] = _decide_stage1(stage1, stage1b)
    decision["convergence_horizon"] = _selected_horizon()
    decision["stages_present"] = {
        "stage1": True,
        "stage1b_init": stage1b is not None,
        "stage2": stage2 is not None,
        "mechanism": mechanism is not None,
    }
    decision["e2e_authorized"] = e2e_authorized(decision)
    _write_json_any(RESULTS_DIR / "final_frozen_decision.json", decision)

    pooled_rows: list[dict[str, Any]] = []
    for backbone_seed, stage in ((PRIMARY_BACKBONE_SEED, stage1), (BACKBONE_SEEDS[1], stage2)):
        if stage is None:
            continue
        frame = pd.read_csv(RESULTS_DIR / f"{'stage1' if backbone_seed == 0 else 'stage2'}_fold_results.csv")
        for _, row in frame.iterrows():
            pooled_rows.append(
                {
                    "backbone_seed": int(backbone_seed),
                    "fold": int(row["fold"]),
                    "mae_h0": float(row["mae_h0"]),
                    "mae_hlin": float(row["mae_hlin"]),
                    "mae_h1": float(row["mae_h1"]),
                    "mae_h2": float(row["mae_h2"]),
                    "mae_fm": float(row["mae_fm"]),
                    "delta_small": float(row["delta_small"]),
                    "delta_strong": float(row["delta_strong"]),
                }
            )
    pd.DataFrame(pooled_rows).to_csv(RESULTS_DIR / "pooled_backbone_results.csv", index=False)

    _write_json_any(
        RESULTS_DIR / "final_bootstrap.json",
        {
            "stage1_delta_small": stage1["bootstrap_detail"]["delta_small"]
            if "bootstrap_detail" in stage1
            else stage1["delta_small"],
            "stage2_delta_small": (
                stage2["bootstrap_detail"]["delta_small"] if stage2 is not None and "bootstrap_detail" in stage2 else None
            ),
            "pooled_delta_small": decision["pooled_delta_small"],
            "pooled_per_fold_delta_small": decision["per_fold_delta_small"],
        },
    )
    if not decision["e2e_authorized"]:
        _write_json_any(
            RESULTS_DIR / "e2e_status.json",
            {
                "authorized": False,
                "reason": (
                    "frozen final gate not met (INCONCLUSIVE); end-to-end FM replacement is "
                    "not authorized and no rank sweep / hybrid head is opened"
                ),
                "frozen_verdict": decision["verdict"],
            },
        )
    _make_figures()
    print(json.dumps(decision, indent=2, sort_keys=True), flush=True)
    return decision


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def _make_figures() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not (RESULTS_DIR / "stage1_fold_results.csv").exists():
        return
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(RESULTS_DIR / "stage1_fold_results.csv")
    x = np.arange(len(frame))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    ax = axes[0]
    ax.bar(x - 0.2, frame["delta_small"], width=0.4, label=r"$\Delta_{small}$ (H1 - FM)")
    ax.bar(x + 0.2, frame["delta_strong"], width=0.4, label=r"$\Delta_{strong}$ (H2 - FM)")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("outer fold")
    ax.set_ylabel("MAE improvement")
    ax.set_title("Per-fold low-rank interaction gain (backbone seed 0)")
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.bar(x - 0.25, frame["mae_h0"], width=0.25, label="H0")
    ax.bar(x, frame["mae_h1"], width=0.25, label="H1")
    ax.bar(x + 0.25, frame["mae_fm"], width=0.25, label="FM")
    ax.set_xlabel("outer fold")
    ax.set_ylabel("MAE")
    ax.set_title("Direct-head MAE (holdout 2000)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure1_stage1.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

def run_preprocessing_stats(force: bool = False) -> dict[str, Any]:
    out_path = RESULTS_DIR / "preprocessing_stats.json"
    if out_path.exists() and not force:
        return json.loads(out_path.read_text(encoding="utf-8"))
    stats: dict[str, Any] = {}
    for fold in range(K_FOLDS):
        data = _fold_tensors(fold, PRIMARY_BACKBONE_SEED)
        stats[f"fold{fold}"] = {
            "n_fit": int(len(data["fit_pos"])),
            "n_selection": int(len(data["sel_pos"])),
            "n_eval": int(len(data["eva_pos"])),
            "degenerate_coordinates": int(data["degenerate"].sum()),
            "degenerate_indices": np.flatnonzero(data["degenerate"]).tolist(),
            "mean_abs_mean_of_coordinates": float(np.abs(data["mean"]).mean()),
            "mean_scale": float(data["scale"].mean()),
            "fit_sha256": _sha256_array(np.asarray(data["fit_pos"], dtype=np.int64)),
            "standardizer_fit_scope": "head-fit 7200 molecules only",
        }
    _write_json_any(out_path, stats)
    return stats


def _run_all(force: bool = False) -> int:
    print("=== checkpoint inventory ===", flush=True)
    inventory = ciw.checkpoint_inventory()
    _write_json_any(RESULTS_DIR / "checkpoint_inventory.json", inventory)
    if not inventory["complete_seeds"]:
        _write_json_any(RESULTS_DIR / "final_frozen_decision.json", {"verdict": "NO_GO", "reason": "no_checkpoints"})
        return 2

    print("=== head spec / parameter accounting ===", flush=True)
    accounting = _parameter_accounting()
    _write_json_any(RESULTS_DIR / "parameter_counts.json", accounting)
    write_head_spec(accounting)
    catboost_status()
    if not accounting["parameter_match_ok"]:
        _write_json_any(
            RESULTS_DIR / "final_frozen_decision.json",
            {"verdict": "NO_GO", "reason": "parameter_mismatch", "accounting": accounting},
        )
        return 3

    print("=== cache + integrity ===", flush=True)
    stage_cache(seeds=(PRIMARY_BACKBONE_SEED,), force=force)
    integrity = run_integrity(seeds=(PRIMARY_BACKBONE_SEED,), force=force)
    if not integrity["all_passed"]:
        _write_json_any(
            RESULTS_DIR / "final_frozen_decision.json",
            {"verdict": "NO_GO", "reason": "integrity_failed", "failures": integrity["failures"]},
        )
        return 3
    stage_splits(force=force)

    print("=== convergence trace ===", flush=True)
    run_preprocessing_stats(force=force)
    run_convergence_trace(force=force)

    print("=== stage 1 (backbone seed 0, init 0) ===", flush=True)
    stage1 = run_stage(PRIMARY_BACKBONE_SEED, 0, "stage1", force=force)
    verdict1 = _decide_stage1(stage1)

    stage1b = None
    if verdict1 in ("STAGE1_CLEAR_NO_GO", "STAGE1_INCONCLUSIVE"):
        print(f"Stage 1 verdict={verdict1}; second backbone seed NOT spent.", flush=True)
    else:
        if verdict1 == "STAGE1_BORDERLINE":
            print("=== stage 1b (second head init, same backbone) ===", flush=True)
            stage1b = run_stage(PRIMARY_BACKBONE_SEED, 1, "stage1b_init", force=force)
            verdict1 = _decide_stage1(stage1, stage1b)
            print(f"Stage 1b confirmation verdict={verdict1}", flush=True)
        if verdict1 in ("STAGE1_STRONG_ADVANCE", "STAGE1_EFFICIENCY_ADVANCE"):
            print("=== stage 2 (second frozen backbone seed) ===", flush=True)
            stage_cache(seeds=(BACKBONE_SEEDS[1],), force=force)
            run_integrity(seeds=(BACKBONE_SEEDS[1],), force=force)
            run_stage(BACKBONE_SEEDS[1], 0, "stage2", force=force)
            print("=== mechanism analysis ===", flush=True)
            run_mechanism_analysis(force=force)

    decision = run_final_decision()
    decision["stage1_verdict"] = verdict1
    _write_json_any(RESULTS_DIR / "final_frozen_decision.json", decision)
    print(json.dumps(decision, indent=2, sort_keys=True), flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=[
            "inventory", "spec", "cache", "integrity", "splits", "preprocessing",
            "convergence", "stage1", "stage1b", "stage2", "mechanism", "catboost",
            "figures", "decision", "all",
        ],
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seeds", default="0")
    args = parser.parse_args(argv)
    seed_list = tuple(int(s) for s in str(args.seeds).split(",") if s.strip() != "")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.stage == "inventory":
        inventory = ciw.checkpoint_inventory()
        _write_json_any(RESULTS_DIR / "checkpoint_inventory.json", inventory)
        print(json.dumps(inventory, indent=2, sort_keys=True))
        return 0
    if args.stage == "spec":
        write_head_spec()
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
        run_preprocessing_stats(force=args.force)
        return 0
    if args.stage == "convergence":
        run_convergence_trace(force=args.force)
        return 0
    if args.stage == "stage1":
        run_stage(PRIMARY_BACKBONE_SEED, 0, "stage1", force=args.force)
        return 0
    if args.stage == "stage1b":
        run_stage(PRIMARY_BACKBONE_SEED, 1, "stage1b_init", force=args.force)
        return 0
    if args.stage == "stage2":
        stage_cache(seeds=(BACKBONE_SEEDS[1],), force=args.force)
        run_integrity(seeds=(BACKBONE_SEEDS[1],), force=args.force)
        run_stage(BACKBONE_SEEDS[1], 0, "stage2", force=args.force)
        return 0
    if args.stage == "mechanism":
        run_mechanism_analysis(force=args.force)
        return 0
    if args.stage == "catboost":
        run_catboost_secondary(force=args.force)
        return 0
    if args.stage == "figures":
        _make_figures()
        return 0
    if args.stage == "decision":
        run_final_decision(force=args.force)
        return 0
    if args.stage == "all":
        return _run_all(force=args.force)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
