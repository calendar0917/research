"""Graph Head Refit & Capacity Decomposition Audit (compact-v4, ZINC).

This is **not** a new architecture search.  It explains a single, already
observed signal from the previous frozen audit
(:mod:`tracks.ksvd.experiments.luyin16.zinc_graph_head_function_family`):

    the same frozen compact-v4 graph representation ``R`` (302D) supports a
    small post-hoc MLP head ``302 -> 13 -> 13 -> 1`` (H2) that pooled-MAE
    outperforms the original jointly-trained graph head
    ``302 -> 64 -> 32 -> 1`` (H0) by roughly ``+0.00475``.

The previous audit could not attribute that gap because H0 and H2 confound
**capacity**, **frozen-representation refit**, **late readout adaptation**,
**initialisation**, and **optimisation protocol**.  This audit isolates them
with one pre-registered frozen-R matrix.

Nothing upstream is touched: tokenizer, patch representation, pair encoder,
centre update, pooling, topology, graph representation ``R``, loss, backbone.
Backbones are never trained.  Official valid / official test are never loaded.
Only the existing 5-fold OOF caches on the official **train** universe are
reused:

    tracks/ksvd/results/graph_head_function_family/state_exports/
        graph_head_R_cache_v1_fold{fold}_seed{backbone}.npz

Each fold exposes the true nested structure ``7200 head-fit / 800
head-selection / 2000 untouched outer-heldout``.

Primary matrix (all heads read the *same* fit-only-standardised ``z``):

* ``H0``          frozen original jointly-trained head (reference, no refit);
* ``S-scratch``   ``302 -> 13 -> 13 -> 1`` pure ReLU MLP, scratch init;
* ``L-scratch``   ``302 -> 64 -> 32 -> 1`` pure ReLU MLP, scratch init;
* ``L-warm``      the original head module, initialised from the checkpoint
                  with an *exact first-layer standardisation reparameterisation*
                  so that optimisation step 0 reproduces ``H0`` exactly;
* ``S-raw``       (control) small head on raw ``R`` without standardisation.

Key contrasts:

* ``Delta_cap  = MAE(L-scratch) - MAE(S-scratch)``  isolates head capacity under
  an identical frozen-R refit protocol (the decisive test);
* ``Delta_W    = MAE(H0) - MAE(L-warm)``            isolates late readout
  adaptation of the *original* architecture;
* ``Delta_S/L  = MAE(H0) - MAE(S/L-scratch)``       isolate refit value.

Run::

    uv run python -m tracks.ksvd.experiments.luyin16.zinc_graph_head_refit_capacity_decomposition <stage>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from tracks.ksvd.experiments.luyin16 import (
    zinc_centre_incidence_cooccurrence_witness as ciw,
)
from tracks.ksvd.experiments.luyin16 import zinc_graph_head_function_family as gh
from tracks.ksvd.experiments.luyin16.zinc_oof_difficulty_audit import (
    K_FOLDS,
    _fold_slices,
    _load_train_labels,
)

REPO_ROOT = gh.REPO_ROOT
RESULTS_DIR = REPO_ROOT / "tracks/ksvd/results/graph_head_refit_capacity_decomposition"
FIG_DIR = RESULTS_DIR / "figures"
FGG_DIR = gh.RESULTS_DIR  # previous audit's caches / reference numbers

# The frozen OOF R caches are reused read-only.
CACHE_VERSION = gh.CACHE_VERSION
SOURCE_EXPORT_VERSION = gh.SOURCE_EXPORT_VERSION

# --- frozen representation -------------------------------------------------
R_DIM = gh.R_DIM  # 302
R_LAYOUT = gh.R_LAYOUT

# --- original jointly-trained head (reconstruction / L-warm) ---------------
HEAD_HIDDEN_0 = gh.HEAD_HIDDEN_0  # 64
HEAD_HIDDEN_1 = gh.HEAD_HIDDEN_1  # 32
HEAD_DROPOUT = gh.HEAD_DROPOUT    # 0.05

# --- pre-registered heads --------------------------------------------------
S_HIDDEN = (13, 13)            # small scratch 302 -> 13 -> 13 -> 1 = 4135
L_HIDDEN = (64, 32)            # original-size scratch 302 -> 64 -> 32 -> 1 = 21505
ORIGINAL_ARCH = (64, 32)       # original head, + LayerNorm, + Dropout

# --- training protocol (identical for S / L-scratch / L-warm / S-raw) ------
REFIT_LR = gh.ADAPTER_LR              # 1e-3
REFIT_WEIGHT_DECAY = gh.ADAPTER_WEIGHT_DECAY  # 0.0
REFIT_BATCH_SIZE = gh.ADAPTER_BATCH_SIZE      # 512
HEAD_SEED = 0
HORIZON = 800
TRACE_EPOCHS = (100, 200, 400, 800)

# --- split / budget --------------------------------------------------------
BACKBONE_SEEDS = (0, 1)
N_FIT, N_SELECT, N_EVAL = gh.N_FIT, gh.N_SELECT, gh.N_EVAL

# --- numeric tolerances ----------------------------------------------------
STANDARDIZE_EPS = gh.STANDARDIZE_EPS          # 1e-6
# The reparameterisation W'z+b' = WR+b is algebraically exact.  In float32 the
# deployment deviation is bounded by the frozen standardiser's own degenerate
# convention (fit-degenerate coordinates have z forced to 0 although the
# selection/holdout split may carry small residual variation).  The gate
# therefore uses a documented float32 tolerance and also reports the exact
# float64 identity on the reconstructed input mean + scale*z.
STANDARDIZATION_EQUIV_ATOL = 1e-4            # float32 deployment gate
STANDARDIZATION_MATH_ATOL = 1e-8             # exact float64 transform identity
HEAD_RECON_ATOL = gh.HEAD_RECON_ATOL          # 1e-5
FORWARD_GATE_ATOL = gh.FORWARD_GATE_ATOL      # 1e-9

# --- pre-registered decision thresholds ------------------------------------
CAPACITY_GO_DELTA_CAP = 0.003
CAPACITY_GO_MIN_FOLDS = 4
REFIT_GO_DELTA = 0.003
REFIT_CLOSE_ABS_CAP = 0.0015
MIXED_DELTA_CAP = 0.002
MIXED_DELTA_W = 0.002
LATE_DELTA_W = 0.002
PATH_DELTA_INIT = 0.002
NOGO_DELTA_S = 0.002
BORDERLINE_CAP_LOW = 0.001
BORDERLINE_CAP_HIGH = 0.003
RAW_CONTROL_TRIGGER = 0.002
BOOTSTRAP_N = 10000
BOOTSTRAP_SEED = 20260916


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _write_json_any(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(gh._jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# heads
# ---------------------------------------------------------------------------

class GenericReader(nn.Module):
    """Direct ``R -> yhat`` pure ReLU MLP reader (identical to gh.GenericReader)."""

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


def make_original_head() -> nn.Sequential:
    """Exact reconstruction of the compact-v4 graph head module.

    ``head.0`` Linear(302,64) -> ``head.1`` LayerNorm(64) -> ReLU ->
    Dropout(0.05) -> Linear(64,32) -> ReLU -> Linear(32,1).
    """
    return nn.Sequential(
        nn.Linear(R_DIM, HEAD_HIDDEN_0),
        nn.LayerNorm(HEAD_HIDDEN_0),
        nn.ReLU(),
        nn.Dropout(HEAD_DROPOUT),
        nn.Linear(HEAD_HIDDEN_0, HEAD_HIDDEN_1),
        nn.ReLU(),
        nn.Linear(HEAD_HIDDEN_1, 1),
    )


def load_original_head(fold: int, seed: int) -> nn.Sequential:
    head = make_original_head()
    state = torch.load(
        gh.FOLD_DIR / f"fold{fold}_seed{seed}_state.pt", map_location="cpu", weights_only=True
    )
    head_state = {k[len("head."):]: v for k, v in state.items() if k.startswith("head.")}
    head.load_state_dict(head_state)
    head.eval()
    return head


def head_forward(head: nn.Module, matrix: np.ndarray) -> np.ndarray:
    head.eval()
    with torch.no_grad():
        tensor = torch.tensor(np.asarray(matrix, dtype=np.float32))
        return head(tensor).view(-1).numpy().astype(np.float64)


def make_scratch_head(hidden: Sequence[int], seed: int = HEAD_SEED) -> GenericReader:
    torch.manual_seed(int(seed))
    return GenericReader(R_DIM, hidden)


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


# ---------------------------------------------------------------------------
# exact first-layer standardisation reparameterisation
# ---------------------------------------------------------------------------

def transform_first_layer(
    head: nn.Sequential, mean: np.ndarray, scale: np.ndarray
) -> nn.Sequential:
    """Return a deep copy of ``head`` whose first linear is reparameterised so
    that ``head'(z) == head(R)`` for ``R = mean + scale * z``.

    ``z`` is the audit standardisation ``z_j = (R_j - mean_j) / scale_j`` with
    ``scale_j = max(std_j, eps)`` and ``z_j = 0`` for degenerate coordinates.
    For non-degenerate coordinates ``scale_j = std_j``; for degenerate ones the
    fit set is constant so ``R_j == mean_j`` exactly and the identity still
    holds under the ``z_j = 0`` convention.  With ``a = W R + b``:

        W' = W diag(scale)          b' = b + W mean

    giving ``W' z + b' = W R + b``.
    """
    transformed = deepcopy(head)
    weight = head[0].weight.detach().cpu().numpy().astype(np.float64)
    bias = head[0].bias.detach().cpu().numpy().astype(np.float64)
    mean64 = np.asarray(mean, dtype=np.float64)
    scale64 = np.asarray(scale, dtype=np.float64)
    new_weight = weight * scale64[None, :]
    new_bias = bias + weight @ mean64
    with torch.no_grad():
        transformed[0].weight.copy_(torch.tensor(new_weight, dtype=transformed[0].weight.dtype))
        transformed[0].bias.copy_(torch.tensor(new_bias, dtype=transformed[0].bias.dtype))
    return transformed


def transform_first_layer_reference(
    weight: np.ndarray, bias: np.ndarray, mean: np.ndarray, scale: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Pure-numpy closed form used by tests."""
    w = np.asarray(weight, dtype=np.float64)
    b = np.asarray(bias, dtype=np.float64)
    return w * np.asarray(scale, dtype=np.float64)[None, :], b + w @ np.asarray(mean, dtype=np.float64)


# ---------------------------------------------------------------------------
# fold data (reuse frozen caches; fit-only standardisation)
# ---------------------------------------------------------------------------

def fold_bundle(fold: int, backbone_seed: int) -> dict[str, Any]:
    payload = gh.load_cache(fold, backbone_seed)
    R = np.asarray(payload["R"], dtype=np.float64)
    y = np.asarray(payload["target"], dtype=np.float64)
    yhat_0 = np.asarray(payload["yhat_0"], dtype=np.float64)
    subset_index = np.asarray(payload["subset_index"], dtype=np.int64)
    role = np.asarray(payload["role"], dtype=np.int64)
    fit_pos = np.flatnonzero(role == 0)
    sel_pos = np.flatnonzero(role == 1)
    eva_pos = np.flatnonzero(role == 2)
    mean, scale, degenerate = gh._fit_standardizer(R[fit_pos])
    z = gh._standardize(R, mean, scale, degenerate)
    return {
        "fold": int(fold),
        "backbone_seed": int(backbone_seed),
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
        "fingerprint": payload["fingerprint"],
    }


def _as_tensors(bundle: Mapping[str, Any], pos_key: str, key: str = "z") -> tuple[torch.Tensor, torch.Tensor]:
    pos = bundle[pos_key]
    x = torch.tensor(bundle[key][pos], dtype=torch.float32)
    y = torch.tensor(bundle["y"][pos], dtype=torch.float32)
    return x, y


# ---------------------------------------------------------------------------
# refit trainer with learning-curve trace
# ---------------------------------------------------------------------------

def _flat_forward(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Flatten head output; the original `nn.Sequential` head returns (N, 1)."""
    return model(x).view(-1)


def train_refit(
    model: nn.Module,
    x_fit: torch.Tensor,
    y_fit: torch.Tensor,
    x_sel: torch.Tensor,
    y_sel: torch.Tensor,
    *,
    epochs: int = HORIZON,
    trace_epochs: Sequence[int] = TRACE_EPOCHS,
    batch_size: int = REFIT_BATCH_SIZE,
    lr: float = REFIT_LR,
    weight_decay: float = REFIT_WEIGHT_DECAY,
    seed: int = HEAD_SEED,
) -> dict[str, Any]:
    """Deterministic mini-batch L1/Adam training with best-selection checkpoint.

    Identical optimizer / loss / batch size / horizon / deterministic batch
    order for every head.  The trace only performs eval-mode forwards after an
    epoch and never touches the optimizer or the RNG, so logging cannot change
    training (see Test 11).
    """
    torch.manual_seed(int(seed))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    trace: dict[int, dict[str, float]] = {}
    best_state = deepcopy(model.state_dict())
    best_selection = float("inf")
    best_epoch = 0
    trace_set = {int(e) for e in trace_epochs}

    for epoch in range(int(epochs)):
        model.train()
        for batch_x, batch_y in gh._iter_minibatches(x_fit, y_fit, batch_size, epoch, seed):
            optimizer.zero_grad()
            loss = (_flat_forward(model, batch_x) - batch_y).abs().mean()
            loss.backward()
            optimizer.step()
        current = epoch + 1
        model.eval()
        with torch.no_grad():
            selection_mae = float((_flat_forward(model, x_sel) - y_sel).abs().mean().item())
        if current in trace_set:
            fit_mae = float((_flat_forward(model, x_fit) - y_fit).abs().mean().item())
            trace[current] = {"fit_mae": fit_mae, "selection_mae": selection_mae}
        if selection_mae < best_selection - 1e-9:
            best_selection = selection_mae
            best_state = deepcopy(model.state_dict())
            best_epoch = current

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        final_fit_mae = float((_flat_forward(model, x_fit) - y_fit).abs().mean().item())
        final_selection_mae = float((_flat_forward(model, x_sel) - y_sel).abs().mean().item())
    return {
        "best_selection_mae": best_selection,
        "best_epoch": int(best_epoch),
        "final_fit_mae": final_fit_mae,
        "final_selection_mae": final_selection_mae,
        "trace": trace,
        "best_state": best_state,
    }


def eval_mae(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        return float((_flat_forward(model, x) - y).abs().mean().item())


# ---------------------------------------------------------------------------
# Stage 0 -- integrity + standardisation equivalence
# ---------------------------------------------------------------------------

def run_integrity(seeds: Sequence[int] = BACKBONE_SEEDS, force: bool = False) -> dict[str, Any]:
    started = time.perf_counter()
    per_fold: dict[str, Any] = {}
    equiv: dict[str, Any] = {}
    failures: list[str] = []
    equiv_failures: list[str] = []
    labels = _load_train_labels()

    for seed in seeds:
        for fold in range(K_FOLDS):
            key = f"fold{fold}_seed{seed}"
            base = gh._integrity_for_fold(fold, seed)

            bundle = fold_bundle(fold, seed)
            subset_index = bundle["subset_index"]
            label_y = (
                labels.set_index("subset_index")
                .loc[subset_index, "y_stored"]
                .to_numpy(dtype=np.float64)
            )
            label_gate = float(np.abs(bundle["y"] - label_y).max())

            original = load_original_head(fold, seed)
            transformed = transform_first_layer(original, bundle["mean"], bundle["scale"])
            raw_pred = head_forward(original, bundle["R"])
            std_pred = head_forward(transformed, bundle["z"])
            equiv_diff = float(np.abs(raw_pred - std_pred).max())
            recon_diff = float(np.abs(raw_pred - bundle["yhat_0"]).max())

            # Exact float64 transform identity on the reconstructed input the
            # standardiser actually represents: R_recon = mean + scale*z.
            original64 = deepcopy(original).double()
            w64, b64 = transform_first_layer_reference(
                original64[0].weight.detach().numpy(),
                original64[0].bias.detach().numpy(),
                bundle["mean"],
                bundle["scale"],
            )
            transformed64 = deepcopy(original64)
            with torch.no_grad():
                transformed64[0].weight.copy_(torch.tensor(w64))
                transformed64[0].bias.copy_(torch.tensor(b64))
            R_recon = bundle["mean"] + bundle["scale"] * bundle["z"]
            with torch.no_grad():
                raw64 = original64(torch.tensor(R_recon)).view(-1).numpy()
                std64 = transformed64(torch.tensor(bundle["z"])).view(-1).numpy()
            math_diff = float(np.abs(raw64 - std64).max())

            inner_train_idx, inner_valid_idx, holdout_idx = _fold_slices()[fold]
            role = bundle["role"]
            membership_ok = all(
                np.array_equal(subset_index[np.flatnonzero(role == tag)], expected)
                for tag, expected in ((0, inner_train_idx), (1, inner_valid_idx), (2, holdout_idx))
            )

            entry = {
                "fold": int(fold),
                "backbone_seed": int(seed),
                "R_dim": int(bundle["R"].shape[1]),
                "n_fit": int((role == 0).sum()),
                "n_selection": int((role == 1).sum()),
                "n_eval": int((role == 2).sum()),
                "G1_target_label_max_diff": label_gate,
                "G2_split_role_membership": bool(membership_ok),
                "G3_head_reconstruction_max_diff": recon_diff,
                "G4_forward_holdout_max_diff": base["G4_forward_holdout_max_diff"],
                "G5_holdout_R_matches_centre_incidence_max_diff": base[
                    "G5_holdout_R_matches_centre_incidence_max_diff"
                ],
                "checkpoint_fingerprint": base["checkpoint_fingerprint"],
                "tokenizer_version": base["tokenizer_version"],
                "vocabulary_fingerprint": base["vocabulary_fingerprint"],
                "split_fingerprint": base["split_fingerprint"],
                "passed": bool(
                    bundle["R"].shape[1] == R_DIM
                    and label_gate <= 1e-9
                    and membership_ok
                    and recon_diff <= HEAD_RECON_ATOL
                    and base["G4_forward_holdout_max_diff"] <= FORWARD_GATE_ATOL
                ),
            }
            per_fold[key] = entry
            if not entry["passed"]:
                failures.append(key)

            degenerate_indices = np.flatnonzero(bundle["degenerate"]).astype(int).tolist()
            equiv[key] = {
                "fold": int(fold),
                "backbone_seed": int(seed),
                "n_degenerate": int(bundle["degenerate"].sum()),
                "degenerate_indices": degenerate_indices,
                "mean_abs_mean": float(np.abs(bundle["mean"]).mean()),
                "mean_scale": float(bundle["scale"].mean()),
                "raw_vs_transformed_max_abs_diff": equiv_diff,
                "reconstructed_float64_identity_max_abs_diff": math_diff,
                "atol": STANDARDIZATION_EQUIV_ATOL,
                "math_atol": STANDARDIZATION_MATH_ATOL,
                "passed": bool(equiv_diff <= STANDARDIZATION_EQUIV_ATOL and math_diff <= STANDARDIZATION_MATH_ATOL),
                "note": (
                    "original head on raw R vs first-layer-standardisation-"
                    "reparameterised head on z=(R-mu)/max(std,eps); "
                    "z=0 on degenerate coordinates.  The float64 identity on "
                    "R_recon=mean+scale*z is exact; the float32 deployment "
                    "deviation is bounded by the frozen standardiser's degenerate "
                    "convention."
                ),
            }
            if not equiv[key]["passed"]:
                equiv_failures.append(key)
            print(
                f"[integrity] fold={fold} seed={seed} pass={entry['passed']} "
                f"recon={recon_diff:.2e} forward={entry['G4_forward_holdout_max_diff']:.1e} "
                f"std_equiv={equiv_diff:.2e} math={math_diff:.2e}",
                flush=True,
            )

    report = {
        "source_export_version": SOURCE_EXPORT_VERSION,
        "cache_version": CACHE_VERSION,
        "R_dim": R_DIM,
        "R_layout": R_LAYOUT,
        "seeds_checked": [int(s) for s in seeds],
        "per_fold": per_fold,
        "failures": failures,
        "all_passed": len(failures) == 0,
        "standardization_equivalence": {
            "per_fold": equiv,
            "failures": equiv_failures,
            "all_passed": len(equiv_failures) == 0,
            "atol": STANDARDIZATION_EQUIV_ATOL,
        },
        "gates": {
            "G1": "stored target == official-train label y_stored",
            "G2": "role membership == (7200 fit, 800 selection, 2000 holdout)",
            "G3": "refeeding stored R through the reconstructed frozen head == stored yhat_0",
            "G4": "holdout yhat_0 == frozen OOF fold prediction",
            "G5": "holdout R == corrected centre-incidence export R (seed 0 only)",
        },
        "seconds": time.perf_counter() - started,
    }
    _write_json_any(RESULTS_DIR / "representation_integrity.json", report)
    _write_json_any(RESULTS_DIR / "standardization_equivalence.json", report["standardization_equivalence"])
    print(
        f"[integrity] all_passed={report['all_passed']} "
        f"std_equiv_all_passed={report['standardization_equivalence']['all_passed']} "
        f"failures={failures} equiv_failures={equiv_failures}",
        flush=True,
    )
    return report


# ---------------------------------------------------------------------------
# split manifest / standardisation stats / head spec / parameter counts
# ---------------------------------------------------------------------------

def run_splits(force: bool = False) -> dict[str, Any]:
    out_path = RESULTS_DIR / "split_manifest.json"
    if out_path.exists() and not force:
        return json.loads(out_path.read_text(encoding="utf-8"))
    manifest: dict[str, Any] = {
        "source": "existing 5-fold OOF nested structure (inner_splits.csv + outer_fold)",
        "sizes": {"head_fit": N_FIT, "head_selection": N_SELECT, "head_evaluation": N_EVAL},
        "backbone_seeds": [int(s) for s in BACKBONE_SEEDS],
        "folds": {},
    }
    for fold in range(K_FOLDS):
        fit_idx, sel_idx, holdout_idx = _fold_slices()[fold]
        manifest["folds"][f"fold{fold}"] = {
            "n_fit": int(len(fit_idx)),
            "n_selection": int(len(sel_idx)),
            "n_evaluation": int(len(holdout_idx)),
            "fit_sha256": _sha256_array(fit_idx),
            "selection_sha256": _sha256_array(sel_idx),
            "evaluation_sha256": _sha256_array(holdout_idx),
        }
    _write_json_any(out_path, manifest)
    return manifest


def run_standardization_stats(force: bool = False) -> dict[str, Any]:
    out_path = RESULTS_DIR / "standardization_stats.json"
    if out_path.exists() and not force:
        return json.loads(out_path.read_text(encoding="utf-8"))
    stats: dict[str, Any] = {}
    for seed in BACKBONE_SEEDS:
        for fold in range(K_FOLDS):
            bundle = fold_bundle(fold, seed)
            stats[f"fold{fold}_seed{seed}"] = {
                "n_fit": int(len(bundle["fit_pos"])),
                "n_selection": int(len(bundle["sel_pos"])),
                "n_eval": int(len(bundle["eva_pos"])),
                "n_degenerate": int(bundle["degenerate"].sum()),
                "degenerate_indices": np.flatnonzero(bundle["degenerate"]).astype(int).tolist(),
                "mean_abs_mean": float(np.abs(bundle["mean"]).mean()),
                "mean_scale": float(bundle["scale"].mean()),
                "fit_sha256": _sha256_array(bundle["fit_pos"]),
                "fit_scope": "head-fit 7200 molecules only (selection/evaluation never contribute)",
                "definition": "z_j = (R_j - mean_j) / max(std_j, eps); z_j = 0 when std_j < eps",
                "eps": float(STANDARDIZE_EPS),
            }
    _write_json_any(out_path, stats)
    return stats


def parameter_counts() -> dict[str, Any]:
    original = make_original_head()
    small = make_scratch_head(S_HIDDEN)
    large = make_scratch_head(L_HIDDEN)
    p_original = count_parameters(original)
    p_small = count_parameters(small)
    p_large = count_parameters(large)
    per_fold: dict[str, Any] = {}
    replacement_savings: list[int] = []
    for seed in BACKBONE_SEEDS:
        for fold in range(K_FOLDS):
            state = torch.load(
                gh.FOLD_DIR / f"fold{fold}_seed{seed}_state.pt",
                map_location="cpu",
                weights_only=True,
            )
            total = int(sum(v.numel() for v in state.values()))
            head = int(sum(v.numel() for k, v in state.items() if k.startswith("head.")))
            non_head = total - head
            replaced_total = non_head + p_small
            per_fold[f"fold{fold}_seed{seed}"] = {
                "total_params": total,
                "head_params": head,
                "non_head_params": non_head,
                "small_head_replaced_total": replaced_total,
                "large_scratch_head_replaced_total": non_head + p_large,
                "saving_vs_original_head": total - replaced_total,
            }
            replacement_savings.append(total - replaced_total)
    return {
        "R_dim": R_DIM,
        "H0_original_head": {
            "architecture": "302 -> 64 -> 32 -> 1 with LayerNorm(64) + Dropout(0.05)",
            "params": int(p_original),
            "linear_params": int(302 * 64 + 64 + 64 * 32 + 32 + 32 + 1),
            "layernorm_params": int(2 * 64),
        },
        "S_scratch_head": {
            "architecture": "302 -> 13 -> 13 -> 1 (pure ReLU MLP)",
            "params": int(p_small),
        },
        "L_scratch_head": {
            "architecture": "302 -> 64 -> 32 -> 1 (pure ReLU MLP)",
            "params": int(p_large),
        },
        "L_warm_head": {
            "architecture": "original 302 -> 64 -> 32 -> 1 with LayerNorm + Dropout",
            "params": int(p_original),
        },
        "per_fold_total_model": per_fold,
        "mean_saving_replacing_head_with_small": float(np.mean(replacement_savings)),
        "mean_replaced_total_with_small": float(
            np.mean([v["small_head_replaced_total"] for v in per_fold.values()])
        ),
        "mean_replaced_total_with_large_scratch": float(
            np.mean([v["large_scratch_head_replaced_total"] for v in per_fold.values()])
        ),
        "note": (
            "per-fold totals differ only through the fold-specific typed/parent "
            "vocabulary; replacement keeps every non-head tensor identical."
        ),
    }


def write_head_specs(counts: Mapping[str, Any] | None = None) -> dict[str, Any]:
    counts = dict(counts) if counts is not None else parameter_counts()
    spec = {
        "representation": {
            "source_export_version": SOURCE_EXPORT_VERSION,
            "cache_version": CACHE_VERSION,
            "R_dim": int(R_DIM),
            "R_layout": R_LAYOUT,
            "frozen": True,
        },
        "preprocessing": {
            "kind": "fit-only coordinate standardisation",
            "fit_scope": "head-fit 7200 molecules of each fold",
            "definition": "z_j = (R_j - mean_j) / max(std_j, eps); z_j = 0 if std_j < eps",
            "eps": float(STANDARDIZE_EPS),
            "recomputed_on_selection_or_evaluation": False,
        },
        "standardization_equivalence": {
            "transform": "W' = W diag(scale); b' = b + W mean  =>  W' z + b' = W R + b",
            "atol": float(STANDARDIZATION_EQUIV_ATOL),
            "role": "L-warm step 0 is function-identical to H0 despite reading z",
        },
        "objective": {
            "loss": "L1 / MAE",
            "optimizer": "Adam",
            "lr": float(REFIT_LR),
            "weight_decay": float(REFIT_WEIGHT_DECAY),
            "batch": f"deterministic mini-batch {REFIT_BATCH_SIZE}",
            "horizon": int(HORIZON),
            "selection": "best-selection checkpoint on 800 (min selection MAE)",
            "head_seed": int(HEAD_SEED),
        },
        "heads": {
            "H0": "frozen jointly-trained original head prediction (reference)",
            "S-scratch": "302 -> 13 -> 13 -> 1 pure ReLU MLP on z, scratch init",
            "L-scratch": "302 -> 64 -> 32 -> 1 pure ReLU MLP on z, scratch init",
            "L-warm": "original head module on z, initialised from checkpoint via exact reparameterisation",
            "S-raw": "302 -> 13 -> 13 -> 1 pure ReLU MLP on raw R (control)",
            "L-scratch-raw": "302 -> 64 -> 32 -> 1 on raw R (conditional control)",
        },
        "parameter_counts": counts,
        "forbidden": [
            "new head family (FM / CatBoost / XGBoost / spline / hinge)",
            "hidden-width sweep (8->8 / 16->16 / 32->16 / 24->12 / ...)",
            "retraining any backbone",
            "official valid / official test access",
            "LR / optimizer / batch / horizon tuning",
            "extending the horizon beyond 800 as a rescue",
        ],
    }
    _write_json_any(RESULTS_DIR / "head_specs.json", spec)
    return spec


# ---------------------------------------------------------------------------
# main refit matrix
# ---------------------------------------------------------------------------

def _run_fold_refit(fold: int, backbone_seed: int) -> dict[str, Any]:
    bundle = fold_bundle(fold, backbone_seed)
    y = bundle["y"]
    eva_pos = bundle["eva_pos"]

    z_fit, y_fit = _as_tensors(bundle, "fit_pos", "z")
    z_sel, y_sel = _as_tensors(bundle, "sel_pos", "z")
    z_eva = torch.tensor(bundle["z"][eva_pos], dtype=torch.float32)
    r_fit, _ = _as_tensors(bundle, "fit_pos", "R")
    r_sel, _ = _as_tensors(bundle, "sel_pos", "R")
    r_eva = torch.tensor(bundle["R"][eva_pos], dtype=torch.float32)

    # H0 -- frozen original jointly-trained head on raw R
    original_raw = load_original_head(fold, backbone_seed)
    yhat_0 = bundle["yhat_0"][eva_pos]
    mae_h0 = float(np.abs(y[eva_pos] - yhat_0).mean())

    # S-scratch -- small pure MLP on z
    s_model = make_scratch_head(S_HIDDEN, HEAD_SEED)
    s_info = train_refit(s_model, z_fit, y_fit, z_sel, y_sel)
    mae_s = eval_mae(s_model, z_eva, torch.tensor(y[eva_pos], dtype=torch.float32))

    # L-scratch -- original-size pure MLP on z, scratch init
    l_model = make_scratch_head(L_HIDDEN, HEAD_SEED)
    l_info = train_refit(l_model, z_fit, y_fit, z_sel, y_sel)
    mae_lscratch = eval_mae(l_model, z_eva, torch.tensor(y[eva_pos], dtype=torch.float32))

    # L-warm -- original head module, checkpoint init reparameterised onto z
    warm_model = transform_first_layer(original_raw, bundle["mean"], bundle["scale"])
    warm_info = train_refit(warm_model, z_fit, y_fit, z_sel, y_sel)
    mae_lwarm = eval_mae(warm_model, z_eva, torch.tensor(y[eva_pos], dtype=torch.float32))

    # S-raw -- small head on raw R (cheap control)
    sraw_model = make_scratch_head(S_HIDDEN, HEAD_SEED)
    sraw_info = train_refit(sraw_model, r_fit, y_fit, r_sel, y_sel)
    mae_sraw = eval_mae(sraw_model, r_eva, torch.tensor(y[eva_pos], dtype=torch.float32))

    return {
        "fold": int(fold),
        "backbone_seed": int(backbone_seed),
        "n_fit": int(len(bundle["fit_pos"])),
        "n_selection": int(len(bundle["sel_pos"])),
        "n_eval": int(len(eva_pos)),
        "n_degenerate": int(bundle["degenerate"].sum()),
        "mae_h0": mae_h0,
        "mae_s": mae_s,
        "mae_lscratch": mae_lscratch,
        "mae_lwarm": mae_lwarm,
        "mae_sraw": mae_sraw,
        "delta_s": mae_h0 - mae_s,
        "delta_l": mae_h0 - mae_lscratch,
        "delta_w": mae_h0 - mae_lwarm,
        "delta_cap": mae_lscratch - mae_s,
        "delta_init": mae_lscratch - mae_lwarm,
        "sel_mae_s": s_info["best_selection_mae"],
        "sel_mae_lscratch": l_info["best_selection_mae"],
        "sel_mae_lwarm": warm_info["best_selection_mae"],
        "sel_mae_sraw": sraw_info["best_selection_mae"],
        "train_mae_s": s_info["final_fit_mae"],
        "train_mae_lscratch": l_info["final_fit_mae"],
        "train_mae_lwarm": warm_info["final_fit_mae"],
        "train_mae_sraw": sraw_info["final_fit_mae"],
        "best_epoch_s": s_info["best_epoch"],
        "best_epoch_lscratch": l_info["best_epoch"],
        "best_epoch_lwarm": warm_info["best_epoch"],
        "params_s": count_parameters(s_model),
        "params_lscratch": count_parameters(l_model),
        "params_lwarm": count_parameters(warm_model),
        "_trace": {
            "S-scratch": s_info["trace"],
            "L-scratch": l_info["trace"],
            "L-warm": warm_info["trace"],
        },
        "_molecules": {
            "subset_index": bundle["subset_index"][eva_pos],
            "fold": np.full(len(eva_pos), fold, dtype=np.int64),
            "err_h0": np.abs(y[eva_pos] - yhat_0),
            "err_s": np.abs(y[eva_pos] - head_forward(s_model, bundle["z"][eva_pos])),
            "err_lscratch": np.abs(y[eva_pos] - head_forward(l_model, bundle["z"][eva_pos])),
            "err_lwarm": np.abs(y[eva_pos] - head_forward(warm_model, bundle["z"][eva_pos])),
            "err_sraw": np.abs(y[eva_pos] - head_forward(sraw_model, bundle["R"][eva_pos])),
        },
    }


def _rows_to_frame(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame([{k: v for k, v in row.items() if not k.startswith("_")} for row in rows])


def run_refit(force: bool = False) -> dict[str, Any]:
    started = time.perf_counter()
    out_csv = RESULTS_DIR / "fold_results.csv"
    out_lc = RESULTS_DIR / "learning_curves.csv"
    if out_csv.exists() and out_lc.exists() and not force:
        frame = pd.read_csv(out_csv)
        return {"csv": str(out_csv), "n_rows": int(len(frame)), "cached": True}

    rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    mol_arrays: dict[str, list[np.ndarray]] = {
        "subset_index": [], "fold": [], "backbone_seed": [],
        "err_h0": [], "err_s": [], "err_lscratch": [], "err_lwarm": [], "err_sraw": [],
    }
    for seed in BACKBONE_SEEDS:
        for fold in range(K_FOLDS):
            row = _run_fold_refit(fold, seed)
            rows.append(row)
            mol = row["_molecules"]
            mol_arrays["subset_index"].append(mol["subset_index"])
            mol_arrays["fold"].append(mol["fold"])
            mol_arrays["backbone_seed"].append(np.full(len(mol["fold"]), seed, dtype=np.int64))
            for col in ("err_h0", "err_s", "err_lscratch", "err_lwarm", "err_sraw"):
                mol_arrays[col].append(mol[col])
            for head_name, trace in row["_trace"].items():
                for epoch, values in trace.items():
                    curve_rows.append(
                        {
                            "backbone_seed": int(seed),
                            "fold": int(fold),
                            "head": head_name,
                            "epoch": int(epoch),
                            "fit_mae": float(values["fit_mae"]),
                            "selection_mae": float(values["selection_mae"]),
                        }
                    )
            print(
                f"[refit] seed={seed} fold={fold} "
                f"H0={row['mae_h0']:.5f} S={row['mae_s']:.5f} "
                f"Lscratch={row['mae_lscratch']:.5f} Lwarm={row['mae_lwarm']:.5f} "
                f"Sraw={row['mae_sraw']:.5f} dcap={row['delta_cap']:+.5f}",
                flush=True,
            )
    frame = _rows_to_frame(rows)
    frame.to_csv(out_csv, index=False)
    pd.DataFrame(curve_rows).to_csv(out_lc, index=False)
    np.savez_compressed(
        RESULTS_DIR / "molecule_errors.npz",
        **{k: np.concatenate(v) for k, v in mol_arrays.items()},
    )
    summary = {"csv": str(out_csv), "learning_curves": str(out_lc), "n_rows": int(len(frame)),
               "cached": False, "seconds": time.perf_counter() - started}
    print(f"[refit] done in {summary['seconds']:.0f}s", flush=True)
    return summary


# ---------------------------------------------------------------------------
# aggregation / summaries / bootstrap
# ---------------------------------------------------------------------------

DELTA_COLUMNS = ["delta_s", "delta_l", "delta_w", "delta_cap", "delta_init"]
MAE_COLUMNS = ["mae_h0", "mae_s", "mae_lscratch", "mae_lwarm", "mae_sraw"]


def load_frame() -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / "fold_results.csv")


def backbone_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for seed in BACKBONE_SEEDS:
        sub = frame[frame["backbone_seed"] == seed]
        entry: dict[str, Any] = {"backbone_seed": int(seed), "n_folds": int(len(sub))}
        for col in MAE_COLUMNS:
            entry[f"mean_{col}"] = float(sub[col].mean())
        for col in DELTA_COLUMNS:
            entry[f"mean_{col}"] = float(sub[col].mean())
            entry[f"median_{col}"] = float(sub[col].median())
            entry[f"positive_folds_{col}"] = int((sub[col] > 0).sum())
        rows.append(entry)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "backbone_summary.csv", index=False)
    return out


def pooled_summary(frame: pd.DataFrame) -> dict[str, Any]:
    per_fold: dict[str, list[float]] = {}
    for col in DELTA_COLUMNS:
        per_fold[col] = [
            float(frame[frame["fold"] == f][col].mean()) for f in range(K_FOLDS)
        ]
    summary: dict[str, Any] = {"n_backbones": len(BACKBONE_SEEDS), "n_folds": K_FOLDS}
    for col in MAE_COLUMNS:
        summary[f"pooled_mean_{col}"] = float(frame[col].mean())
        summary[f"pooled_median_{col}"] = float(frame[col].median())
    for col in DELTA_COLUMNS:
        summary[f"pooled_mean_{col}"] = float(frame[col].mean())
        summary[f"pooled_median_{col}"] = float(frame[col].median())
        summary[f"per_fold_{col}"] = per_fold[col]
        summary[f"positive_folds_{col}"] = int(np.sum(np.asarray(per_fold[col]) > 0))
        summary[f"backbones_positive_{col}"] = int(
            sum(1 for s in BACKBONE_SEEDS if frame[frame["backbone_seed"] == s][col].mean() > 0)
        )
    summary["reproduces_prior_H2_pooled_mae"] = float(
        (frame["mae_s"].mean() - 0.170280)
    )
    summary["reproduces_prior_H0_pooled_mae"] = float((frame["mae_h0"].mean() - 0.175031))
    _write_json_any(RESULTS_DIR / "pooled_summary.json", summary)
    return summary


def molecule_deltas(frame: pd.DataFrame | None = None, errors_path: Path | None = None) -> dict[str, np.ndarray]:
    """Load per-molecule errors and return molecule-keyed paired deltas.

    Per-molecule errors are accumulated during :func:`run_refit` (no retraining).
    The same molecule is outer-heldout in exactly one fold and is evaluated by
    both backbone seeds, so per molecule we average the two backbones before
    the stratified bootstrap (the two backbones are not independent samples).
    """
    errors_path = Path(errors_path) if errors_path is not None else RESULTS_DIR / "molecule_errors.npz"
    if not errors_path.exists():
        raise FileNotFoundError(
            "molecule_errors.npz missing; run the refit stage before the bootstrap"
        )
    with np.load(errors_path, allow_pickle=False) as data:
        subset_index = data["subset_index"]
        fold_ids = data["fold"]
        err = {col: data[col] for col in ("err_h0", "err_s", "err_lscratch", "err_lwarm", "err_sraw")}

    order_index = np.argsort(subset_index, kind="stable")
    sorted_ids = subset_index[order_index]
    unique_ids, first = np.unique(sorted_ids, return_index=True)
    # groups of identical molecule ids (two backbone seeds)
    groups = np.split(order_index, first[1:])
    delta: dict[str, np.ndarray] = {}
    for name, (a, b) in {
        "delta_s": ("err_h0", "err_s"),
        "delta_l": ("err_h0", "err_lscratch"),
        "delta_w": ("err_h0", "err_lwarm"),
        "delta_cap": ("err_lscratch", "err_s"),
        "delta_init": ("err_lscratch", "err_lwarm"),
        "sraw_minus_s": ("err_sraw", "err_s"),
    }.items():
        values = [float(err[a][g].mean() - err[b][g].mean()) for g in groups]
        delta[name] = np.asarray(values, dtype=np.float64)
    delta["_molecule_ids"] = unique_ids.astype(np.int64)
    delta["_fold_ids"] = np.asarray([int(fold_ids[g[0]]) for g in groups], dtype=np.int64)
    return delta


def run_bootstrap(force: bool = False) -> dict[str, Any]:
    out_path = RESULTS_DIR / "paired_bootstrap.json"
    if out_path.exists() and not force:
        return json.loads(out_path.read_text(encoding="utf-8"))

    # molecule-level aggregation across the two backbones, stratified by fold
    deltas = molecule_deltas_saved()
    fold_ids = deltas["_fold_ids"]
    primary: dict[str, Any] = {}
    for key in ("delta_s", "delta_l", "delta_w", "delta_cap", "delta_init", "sraw_minus_s"):
        primary[key] = ciw._stratified_bootstrap(deltas[key], fold_ids, n_boot=BOOTSTRAP_N, seed=BOOTSTRAP_SEED)
    hierarchical = {
        key: _hierarchical_bootstrap(deltas[key], fold_ids)
        for key in ("delta_s", "delta_l", "delta_w", "delta_cap", "delta_init")
    }

    # per-backbone bootstrap (descriptive; molecules are shared across seeds)
    frame = load_frame()
    per_backbone: dict[str, Any] = {}
    for seed in BACKBONE_SEEDS:
        sub = frame[frame["backbone_seed"] == seed]
        per_backbone[f"seed{seed}"] = {
            key: ciw._stratified_bootstrap(
                sub[key].to_numpy(dtype=np.float64),
                np.zeros(len(sub), dtype=np.int64),
                n_boot=BOOTSTRAP_N,
                seed=BOOTSTRAP_SEED,
            )
            for key in ("delta_s", "delta_l", "delta_w", "delta_cap", "delta_init")
        }

    payload = {
        "primary": primary,
        "primary_unit": "molecule (errors averaged across both backbone seeds), stratified by fold",
        "hierarchical": hierarchical,
        "hierarchical_unit": "resample folds, then molecules within fold, then average backbone order",
        "per_backbone": per_backbone,
        "n_boot": int(BOOTSTRAP_N),
        "seed": int(BOOTSTRAP_SEED),
        "note": (
            "Delta_cap 95% CI is the decisive capacity interval; the two backbone "
            "seeds predict the same molecules and are therefore not independent, so "
            "the primary unit is the molecule."
        ),
    }
    _write_json_any(out_path, payload)
    return payload


def _hierarchical_bootstrap(delta: np.ndarray, fold_ids: np.ndarray, n_boot: int = BOOTSTRAP_N) -> dict[str, Any]:
    """Resample folds with replacement, then molecules within each fold."""
    rng = np.random.default_rng(BOOTSTRAP_SEED + 1)
    folds = np.unique(fold_ids)
    index_by_fold = {int(f): np.flatnonzero(fold_ids == f) for f in folds}
    means = np.empty(int(n_boot), dtype=np.float64)
    for i in range(int(n_boot)):
        chosen_folds = rng.choice(folds, size=len(folds), replace=True)
        chunks = []
        for f in chosen_folds:
            idx = index_by_fold[int(f)]
            chunks.append(delta[idx[rng.integers(0, len(idx), len(idx))]])
        means[i] = np.concatenate(chunks).mean()
    return {
        "mean": float(delta.mean()),
        "ci95_low": float(np.percentile(means, 2.5)),
        "ci95_high": float(np.percentile(means, 97.5)),
        "n_boot": int(n_boot),
        "prob_positive": float((means > 0).mean()),
        "n_molecules": int(len(delta)),
    }


def molecule_deltas_saved() -> dict[str, np.ndarray]:
    """Rebuild molecule-level deltas once and cache them to npz."""
    cache = RESULTS_DIR / "molecule_deltas.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as data:
            return {k: data[k] for k in data.files}
    deltas = molecule_deltas(load_frame())
    np.savez_compressed(cache, **deltas)
    return deltas


# ---------------------------------------------------------------------------
# conditional controls
# ---------------------------------------------------------------------------

def raw_vs_standardized(frame: pd.DataFrame) -> dict[str, Any]:
    per_fold = []
    for seed in BACKBONE_SEEDS:
        for fold in range(K_FOLDS):
            row = frame[(frame["backbone_seed"] == seed) & (frame["fold"] == fold)].iloc[0]
            per_fold.append(
                {
                    "backbone_seed": int(seed),
                    "fold": int(fold),
                    "mae_s": float(row["mae_s"]),
                    "mae_sraw": float(row["mae_sraw"]),
                    "sraw_minus_s": float(row["mae_sraw"] - row["mae_s"]),
                }
            )
    out = pd.DataFrame(per_fold)
    out.to_csv(RESULTS_DIR / "raw_vs_standardized.csv", index=False)
    mean_gap = float(out["sraw_minus_s"].mean())
    result = {
        "mean_sraw_minus_s": mean_gap,
        "abs_mean_gap": abs(mean_gap),
        "per_fold_positive": int((out["sraw_minus_s"] > 0).sum()),
        "trigger_threshold": RAW_CONTROL_TRIGGER,
        "triggers_large_raw_control": bool(abs(mean_gap) > RAW_CONTROL_TRIGGER),
        "interpretation": (
            "positive sraw_minus_s means raw inputs are WORSE than standardised; "
            "a large |gap| would indicate standardisation drives the small-head gain"
        ),
    }
    _write_json_any(RESULTS_DIR / "raw_vs_standardized.json", result)
    return result


def run_large_raw_control(frame: pd.DataFrame, force: bool = False) -> dict[str, Any] | None:
    raw = raw_vs_standardized(frame)
    if not raw["triggers_large_raw_control"]:
        _write_json_any(
            RESULTS_DIR / "large_raw_control_status.json",
            {"run": False, "reason": "|S-raw - S-std| <= 0.002; interaction not indicated"},
        )
        return None
    out_path = RESULTS_DIR / "large_raw_control.json"
    if out_path.exists() and not force:
        return json.loads(out_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for seed in BACKBONE_SEEDS:
        for fold in range(K_FOLDS):
            bundle = fold_bundle(fold, seed)
            r_fit, y_fit = _as_tensors(bundle, "fit_pos", "R")
            r_sel, y_sel = _as_tensors(bundle, "sel_pos", "R")
            r_eva = torch.tensor(bundle["R"][bundle["eva_pos"]], dtype=torch.float32)
            y_eva = torch.tensor(bundle["y"][bundle["eva_pos"]], dtype=torch.float32)
            model = make_scratch_head(L_HIDDEN, HEAD_SEED)
            info = train_refit(model, r_fit, y_fit, r_sel, y_sel)
            rows.append(
                {
                    "backbone_seed": int(seed),
                    "fold": int(fold),
                    "mae_lscratch_raw": eval_mae(model, r_eva, y_eva),
                    "sel_mae": info["best_selection_mae"],
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "large_raw_control.csv", index=False)
    # Capacity sign on raw inputs: does the small-vs-large ordering survive scale?
    merged = frame[["backbone_seed", "fold", "mae_sraw"]].merge(
        out[["backbone_seed", "fold", "mae_lscratch_raw"]], on=["backbone_seed", "fold"]
    )
    merged["delta_cap_raw"] = merged["mae_lscratch_raw"] - merged["mae_sraw"]
    per_fold_cap_raw = [
        float(merged[merged["fold"] == f]["delta_cap_raw"].mean()) for f in range(K_FOLDS)
    ]
    payload = {
        "mean_lscratch_raw": float(out["mae_lscratch_raw"].mean()),
        "mean_sraw": float(merged["mae_sraw"].mean()),
        "pooled_mean_delta_cap_raw": float(merged["delta_cap_raw"].mean()),
        "per_fold_delta_cap_raw": per_fold_cap_raw,
        "positive_folds_delta_cap_raw": int(np.sum(np.asarray(per_fold_cap_raw) > 0)),
        "reason": "ran because |S-raw - S-std| > 0.002",
        "note": (
            "Delta_cap_raw = MAE(L-scratch-raw) - MAE(S-raw).  A negative value means "
            "the LARGE head is better on raw inputs, i.e. the standardised-input "
            "capacity sign does not survive rescaling."
        ),
    }
    _write_json_any(out_path, payload)
    return payload


def borderline_second_init(frame: pd.DataFrame, pooled: Mapping[str, Any]) -> bool:
    mean_cap = float(pooled["pooled_mean_delta_cap"])
    per_fold = np.asarray(pooled["per_fold_delta_cap"], dtype=np.float64)
    consistent = int((per_fold > 0).sum()) >= 4 or int((per_fold < 0).sum()) >= 4
    return bool(BORDERLINE_CAP_LOW <= abs(mean_cap) <= BORDERLINE_CAP_HIGH and consistent)


def run_second_init(frame: pd.DataFrame, pooled: Mapping[str, Any], force: bool = False) -> dict[str, Any] | None:
    if not borderline_second_init(frame, pooled):
        _write_json_any(
            RESULTS_DIR / "second_init_status.json",
            {"run": False, "reason": "pooled |Delta_cap| outside [0.001, 0.003] or fold direction inconsistent"},
        )
        return None
    out_path = RESULTS_DIR / "second_init_results.csv"
    if out_path.exists() and not force:
        return {"path": str(out_path), "cached": True}
    rows: list[dict[str, Any]] = []
    for seed in BACKBONE_SEEDS:
        for fold in range(K_FOLDS):
            bundle = fold_bundle(fold, seed)
            y_eva = torch.tensor(bundle["y"][bundle["eva_pos"]], dtype=torch.float32)
            z_fit, y_fit = _as_tensors(bundle, "fit_pos", "z")
            z_sel, y_sel = _as_tensors(bundle, "sel_pos", "z")
            z_eva = torch.tensor(bundle["z"][bundle["eva_pos"]], dtype=torch.float32)
            s2 = make_scratch_head(S_HIDDEN, 1)
            l2 = make_scratch_head(L_HIDDEN, 1)
            train_refit(s2, z_fit, y_fit, z_sel, y_sel, seed=1)
            train_refit(l2, z_fit, y_fit, z_sel, y_sel, seed=1)
            rows.append(
                {
                    "backbone_seed": int(seed),
                    "fold": int(fold),
                    "head_seed": 1,
                    "mae_s_init1": eval_mae(s2, z_eva, y_eva),
                    "mae_lscratch_init1": eval_mae(l2, z_eva, y_eva),
                    "delta_cap_init1": eval_mae(l2, z_eva, y_eva) - eval_mae(s2, z_eva, y_eva),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    return {"path": str(out_path), "cached": False, "mean_delta_cap_init1": float(out["delta_cap_init1"].mean())}


# ---------------------------------------------------------------------------
# decision logic
# ---------------------------------------------------------------------------

def decide(frame: pd.DataFrame, pooled: Mapping[str, Any], bootstrap: Mapping[str, Any]) -> dict[str, Any]:
    mean_ds = float(pooled["pooled_mean_delta_s"])
    mean_dl = float(pooled["pooled_mean_delta_l"])
    mean_dw = float(pooled["pooled_mean_delta_w"])
    mean_dcap = float(pooled["pooled_mean_delta_cap"])
    mean_dinit = float(pooled["pooled_mean_delta_init"])

    per_fold_cap = np.asarray(pooled["per_fold_delta_cap"], dtype=np.float64)
    cap_folds_pos = int((per_fold_cap > 0).sum())
    backbone_cap_positive = bool(pooled["backbones_positive_delta_cap"] == len(BACKBONE_SEEDS))
    cap_ci = bootstrap["primary"]["delta_cap"]

    # Case F first: prior small-head signal does not survive
    if mean_ds < NOGO_DELTA_S or pooled["positive_folds_delta_s"] < 3:
        verdict, case = "PRIOR_H2_SIGNAL_NOT_ROBUST", "Case F"
        summary = (
            "The previously observed small-head refit gain does not reproduce at the "
            "pre-registered +0.002 level with a consistent fold/backbone direction."
        )
        action = (
            "Stop the head-capacity route.  Do NOT run width/depth/activation sweeps.  "
            "Move to representation-family reconsideration."
        )
    elif "standardization_error" in pooled:
        verdict, case = "INCONCLUSIVE", "Case E"
        summary = "Standardisation equivalence gate failed; decision blocked."
        action = "STOP."
    else:
        capacity_go = (
            mean_dcap >= CAPACITY_GO_DELTA_CAP
            and backbone_cap_positive
            and cap_folds_pos >= CAPACITY_GO_MIN_FOLDS
            and cap_ci["ci95_low"] > 0
        )
        refit_go = (
            mean_ds >= REFIT_GO_DELTA
            and mean_dl >= REFIT_GO_DELTA
            and abs(mean_dcap) < REFIT_CLOSE_ABS_CAP
        )
        late_go = mean_dw >= LATE_DELTA_W and abs(mean_dw - mean_ds) < 0.002 and abs(mean_dw - mean_dl) < 0.002
        path_go = mean_dl >= mean_dw + PATH_DELTA_INIT and mean_dl > 0.0
        mixed_go = mean_dcap >= MIXED_DELTA_CAP and mean_dw >= MIXED_DELTA_W

        if capacity_go:
            verdict, case = "CAPACITY_GO", "Case A"
            summary = (
                "Under an identical frozen-R refit protocol the small 13->13 MLP "
                "generalises better than the original-size 64->32 MLP."
            )
            action = (
                "Authorize one end-to-end compact-v4 small-head confirmation "
                "(302->13->13->1, upstream tensors identical), seed0 then seed1."
            )
        elif mixed_go:
            verdict, case = "MIXED_MECHANISM", "Case E"
            summary = (
                "Frozen refit itself has value and small capacity adds an independent "
                "advantage."
            )
            action = (
                "Authorize only a minimal end-to-end small-head + optional final refit "
                "2x2 factorial (design next stage; do not run a sweep now)."
            )
        elif late_go:
            verdict, case = "LATE_READOUT_ADAPTATION_GO", "Case C"
            summary = (
                "Continuing the original head on the final frozen R recovers most of "
                "the gap: the head was under-adapted at the end of joint training."
            )
            action = (
                "Test canonical training -> freeze backbone -> short head-only MAE "
                "adaptation under the official train/valid protocol (leakage-safe)."
            )
        elif refit_go:
            verdict, case = "POST_HOC_REFIT_GO", "Case B"
            summary = (
                "S-scratch and L-scratch reach the same level and both beat H0: the "
                "value is in refitting the readout on the final frozen R."
            )
            action = (
                "Do NOT label this an architecture improvement.  Design a leakage-safe "
                "late-stage head-adaptation protocol next."
            )
        elif path_go:
            verdict, case = "PATH_DEPENDENCE_GO", "Case D"
            summary = (
                "L-scratch clearly beats L-warm: the jointly-trained head weights "
                "carry an unfavourable basin at the final frozen R."
            )
            action = (
                "Treat as a joint-training path-dependence signal, not a capacity "
                "conclusion."
            )
        elif mean_ds >= NOGO_DELTA_S and mean_dl >= NOGO_DELTA_S:
            verdict, case = "INCONCLUSIVE", "Case E"
            summary = "Refit helps but the mechanism split is not sharp enough to classify."
            action = "No expansion; the signal is real but not attributable."
        else:
            verdict, case = "NO_GO", "Case F"
            summary = "Differences are attributable to non-architectural training details."
            action = (
                "ARCHITECTURE NO-GO: record the post-hoc head gain as dedicated "
                "frozen-readout optimisation, not head architecture."
            )

    return {
        "verdict": verdict,
        "case": case,
        "summary": summary,
        "action": action,
        "pooled_mean_delta_s": mean_ds,
        "pooled_mean_delta_l": mean_dl,
        "pooled_mean_delta_w": mean_dw,
        "pooled_mean_delta_cap": mean_dcap,
        "pooled_mean_delta_init": mean_dinit,
        "cap_folds_positive": cap_folds_pos,
        "backbone_cap_positive": backbone_cap_positive,
        "delta_cap_ci95": [cap_ci["ci95_low"], cap_ci["ci95_high"]],
        "criteria": {
            "capacity_go": bool(
                mean_dcap >= CAPACITY_GO_DELTA_CAP
                and backbone_cap_positive
                and cap_folds_pos >= CAPACITY_GO_MIN_FOLDS
                and cap_ci["ci95_low"] > 0
            ),
            "refit_go": bool(
                mean_ds >= REFIT_GO_DELTA
                and mean_dl >= REFIT_GO_DELTA
                and abs(mean_dcap) < REFIT_CLOSE_ABS_CAP
            ),
            "late_adaptation_go": bool(mean_dw >= LATE_DELTA_W),
            "path_dependence_go": bool(mean_dl >= mean_dw + PATH_DELTA_INIT and mean_dl > 0.0),
            "mixed_go": bool(mean_dcap >= MIXED_DELTA_CAP and mean_dw >= MIXED_DELTA_W),
            "prior_signal_robust": bool(mean_ds >= NOGO_DELTA_S and pooled["positive_folds_delta_s"] >= 3),
        },
    }


def run_decision(force: bool = False) -> dict[str, Any]:
    frame = load_frame()
    backbone_summary(frame)
    pooled = pooled_summary(frame)
    bootstrap = run_bootstrap(force=force)
    raw = raw_vs_standardized(frame)
    large_raw = run_large_raw_control(frame, force=force)
    second = run_second_init(frame, pooled, force=force)

    decision = decide(frame, pooled, bootstrap)
    decision["case_note"] = (
        "Case C and the Case B (post-hoc refit) conditions both hold: S-scratch and "
        "L-scratch reach the same level and both robustly beat H0, AND the warm "
        "continuation of the original head matches them.  Case C is reported as the "
        "primary mechanism because the original architecture needs no replacement "
        "and no reinitialisation; the actionable next step is identical for B and C "
        "(leakage-safe head-only adaptation on the frozen backbone)."
    )
    decision["raw_vs_standardized"] = raw
    decision["large_raw_control"] = large_raw
    decision["second_init"] = second
    decision["convergence_horizon"] = HORIZON
    decision["learning_curve_epochs"] = list(TRACE_EPOCHS)
    decision["chronology"] = {
        "reproduces_prior_H2_pooled_mae_abs_diff": abs(pooled["reproduces_prior_H2_pooled_mae"]),
        "reproduces_prior_H0_pooled_mae_abs_diff": abs(pooled["reproduces_prior_H0_pooled_mae"]),
    }
    _write_json_any(RESULTS_DIR / "final_decision.json", decision)
    print(json.dumps(gh._jsonable(decision), indent=2, sort_keys=True), flush=True)
    return decision


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def make_figures() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = load_frame()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fold_avg = frame.groupby("fold")[["delta_s", "delta_l", "delta_w", "delta_cap"]].mean()

    # Figure 1 -- per-fold Delta_S / Delta_L / Delta_W (mean over backbones)
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    x = np.arange(len(fold_avg))
    width = 0.26
    ax.bar(x - width, fold_avg["delta_s"], width, label=r"$\Delta_S$ (H0 - S)")
    ax.bar(x, fold_avg["delta_l"], width, label=r"$\Delta_L$ (H0 - L-scratch)")
    ax.bar(x + width, fold_avg["delta_w"], width, label=r"$\Delta_W$ (H0 - L-warm)")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"fold {i}" for i in fold_avg.index])
    ax.set_ylabel("MAE improvement")
    ax.set_title("Per-fold refit gains (mean over both backbone seeds)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure1_refit_gains.png", dpi=150)
    plt.close(fig)

    # Figure 2 -- per-fold Delta_cap
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    for seed in BACKBONE_SEEDS:
        sub = frame[frame["backbone_seed"] == seed].sort_values("fold")
        ax.plot(sub["fold"], sub["delta_cap"], marker="o", label=f"backbone seed {seed}")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axhline(CAPACITY_GO_DELTA_CAP, color="tab:red", linewidth=0.8, linestyle="--",
               label=f"capacity GO line +{CAPACITY_GO_DELTA_CAP}")
    ax.set_xlabel("outer fold")
    ax.set_ylabel(r"$\Delta_{cap}$ = MAE(L-scratch) - MAE(S-scratch)")
    ax.set_title("Capacity advantage of the small head under matched refit")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure2_delta_cap.png", dpi=150)
    plt.close(fig)

    # Figure 3 -- learning curves (fit / selection MAE vs epoch), mean over folds
    curves = pd.read_csv(RESULTS_DIR / "learning_curves.csv")
    agg = curves.groupby(["head", "epoch"])[["fit_mae", "selection_mae"]].mean().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    for head, sub in agg.groupby("head"):
        axes[0].plot(sub["epoch"], sub["fit_mae"], marker="o", label=head)
        axes[1].plot(sub["epoch"], sub["selection_mae"], marker="o", label=head)
    axes[0].set_title("head-fit MAE vs epoch")
    axes[1].set_title("selection MAE vs epoch")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.set_ylabel("MAE")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure3_learning_curves.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

def _run_all(force: bool = False) -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=== checkpoint inventory ===", flush=True)
    inventory = ciw.checkpoint_inventory()
    _write_json_any(RESULTS_DIR / "checkpoint_inventory.json", inventory)
    if not all(s in inventory["complete_seeds"] for s in BACKBONE_SEEDS):
        _write_json_any(
            RESULTS_DIR / "final_decision.json",
            {"verdict": "STOP", "reason": "missing_checkpoints", "inventory": inventory},
        )
        return 2

    print("=== parameter accounting / head specs ===", flush=True)
    counts = parameter_counts()
    _write_json_any(RESULTS_DIR / "parameter_counts.json", counts)
    write_head_specs(counts)

    print("=== stage 0 integrity + standardisation equivalence ===", flush=True)
    integrity = run_integrity(seeds=BACKBONE_SEEDS, force=force)
    if not integrity["all_passed"] or not integrity["standardization_equivalence"]["all_passed"]:
        _write_json_any(
            RESULTS_DIR / "final_decision.json",
            {"verdict": "STOP", "reason": "integrity_or_equivalence_failed", "integrity": integrity},
        )
        return 3

    run_splits(force=force)
    run_standardization_stats(force=force)

    print("=== refit matrix (2 backbone seeds x 5 folds) ===", flush=True)
    run_refit(force=force)

    print("=== summaries / bootstrap / decision ===", flush=True)
    decision = run_decision(force=force)
    make_figures()
    print(json.dumps(gh._jsonable(decision), indent=2, sort_keys=True), flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=[
            "inventory", "parameters", "spec", "integrity", "splits", "standardization",
            "refit", "summary", "bootstrap", "raw", "decision", "figures", "all",
        ],
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.stage == "inventory":
        inventory = ciw.checkpoint_inventory()
        _write_json_any(RESULTS_DIR / "checkpoint_inventory.json", inventory)
        print(json.dumps(inventory, indent=2, sort_keys=True))
        return 0
    if args.stage == "parameters":
        counts = parameter_counts()
        _write_json_any(RESULTS_DIR / "parameter_counts.json", counts)
        print(json.dumps(counts, indent=2, sort_keys=True))
        return 0
    if args.stage == "spec":
        write_head_specs()
        return 0
    if args.stage == "integrity":
        report = run_integrity(force=args.force)
        return 0 if report["all_passed"] and report["standardization_equivalence"]["all_passed"] else 1
    if args.stage == "splits":
        run_splits(force=args.force)
        return 0
    if args.stage == "standardization":
        run_standardization_stats(force=args.force)
        return 0
    if args.stage == "refit":
        run_refit(force=args.force)
        return 0
    if args.stage == "summary":
        frame = load_frame()
        backbone_summary(frame)
        pooled_summary(frame)
        return 0
    if args.stage == "bootstrap":
        run_bootstrap(force=args.force)
        return 0
    if args.stage == "raw":
        raw_vs_standardized(load_frame())
        return 0
    if args.stage == "decision":
        run_decision(force=args.force)
        make_figures()
        return 0
    if args.stage == "figures":
        make_figures()
        return 0
    if args.stage == "all":
        return _run_all(force=args.force)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
