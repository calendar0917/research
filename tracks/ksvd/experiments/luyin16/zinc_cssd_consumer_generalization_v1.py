"""ZINC CSSD consumer generalization v1: frozen basis + access, can a limited
training arrangement improve the DICT consumer's g generalization?

Round: ``zinc_cssd_consumer_generalization_v1``.

The frozen verified object of this round is the DICT arm of
``zinc_cssd_consumer_replacement_v1`` (the 297,539-parameter M_COMP consumer
whose local phi65 input is the frozen-CSSD full reconstruction).  NOTHING
about the basis, the access path, the fold, the targets, the Q head or the
training recipe is changed.  The single purchased question: with all that
fixed, do *training length* and ONE fixed weight aggregation (WIDE5_240 =
equal mean of epochs [200,210,220,230,240], plus route-conditional
EARLY120/LONG360) improve complete-y performance on the historical
development rows, against the same-trajectory CTRL240 = mean(236..240)?

Stage A (read-only, forward exports only): score the historical
DICT_s0/s1 epoch40/epoch120/epoch240/soup checkpoints on fit/dev, export
per-row predictions, run the pre-registered component/k=0 analyses, and
apply the FROZEN routing rule (OVERFIT / STILL_IMPROVING / FLAT_OR_MIXED)
to choose the Stage-B branch.  Stage B: at most two new from-scratch
trajectories (seeds 0/1) on the frozen revision, with deterministic member
capture; estimators are equal-weight FP32 means of five full member states.
Terminal: one-shot eval of the fixed roster, group-paired bootstrap
(2000 draws, seed 20261007), the unique alpha-mean intervention on every
candidate, and the frozen retention gate + tie-break.

Official valid/test are never instantiated; dev is the historical
development comparison set (also used by Stage A for branch selection), so
final numbers are never an independent confirm.

Usage (local CPU for source-checks; res-2 res2-cu124 for the GPU stages,
all through the registered runner)::

    python -m tracks.ksvd.experiments.luyin16.\\
zinc_cssd_consumer_generalization_v1 --stage source-checks
    ... --stage checkpoint-diagnostics --seed 0 --device cuda:0
    ... --stage freeze-roster
    ... --stage smoke --device cuda:0
    ... --stage train-trajectory --seed 0 --device cuda:0
    ... --stage terminal-eval --device cuda:0
"""

from __future__ import annotations

import argparse
import hashlib
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm
from tracks.ksvd.experiments.luyin16 import (
    zinc_cssd_basis_reuse_v1 as zreuse,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_cssd_consumer_replacement_v1 as repl,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_cssd_nonlinear_binding_v1 as nb,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_full_cycle_target_decomposition_v1 as zftd,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_local_tuple_dictionary_joint_vs_independent_seed0_v1 as prev,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_task_dictionary_and_cycle_witness_seed0_v1 as zw,
)

PROTOCOL_VERSION = "zinc-cssd-consumer-generalization-v1"
RESULT_SLUG = "zinc_cssd_consumer_generalization_v1"
TRACK_ROOT = repl.TRACK_ROOT
RESULTS_DIR = TRACK_ROOT / "results" / RESULT_SLUG

#: the read-only verified consumer round this round generalises
SOURCE_DIR = repl.RESULTS_DIR
SOURCE_RUNS = SOURCE_DIR / "runs"

ARM = "DICT"                       # the single frozen arm of this round
SEEDS = (0, 1)
EPOCHS_CAP = 360                   # the round's hard training-length cap
LOG_EPOCHS = repl.LOG_EPOCHS       # (1, 40, 120, 240)

# ---- stage A (frozen before any dev metric of these checkpoints is read) ----
CKPTS: tuple[tuple[str, int | None], ...] = (
    ("epoch40", 40), ("epoch120", 120), ("epoch240", 240), ("soup", None),
)
K_GROUP_DEFS: tuple[tuple[str, Any], ...] = (
    ("k=0", lambda k: k == 0),
    ("k=-1", lambda k: k == -1),
    ("k=-2", lambda k: k == -2),
    ("k<=-3", lambda k: k <= -3),
)
#: the two pre-registered k=0 grouping axes (analysis only, never inputs)
NNODE_EDGES = (23, 27)             # bins: <=23 / 24..27 / >=28
#: the routing engineering scale (budget routing only, never significance)
ROUTE_SCALE = 0.002
#: cross-device FP32 replay tolerance for saved-prediction consistency
CROSS_DEVICE_TOL = 1e-5
#: exported float64 component-sum rounding tolerance (kept separate from the
#: exact torch-internal identity, which must be exactly 0)
FLOAT64_SUM_TOL = 1e-6

# ---- stage B / terminal (frozen) -------------------------------------------
N_BOOT = 2000
BOOT_SEED = 20261007
#: retention gate (this round's resource gate, frozen before first scoring)
GATE_DELTA_Y_MEAN = -0.003         # two-seed mean Delta_y must be <= this
GATE_DELTA_Y_SEED = 0.0            # every seed's Delta_y must be < this
GATE_G_BAND = 1e-4                 # per-seed g-MAE may not worsen beyond this
TIE_BREAK_Y = 1e-4                 # mean-y_raw differences below this tie
#: response marker (identical semantics to the source round)
REPLAY_TOL = 1e-4

write_json = nb.write_json
read_json = nb.read_json
file_sha256 = nb.file_sha256
array_sha256 = nb.array_sha256
state_hash = repl.state_hash
seed_everything = repl.seed_everything
resolve_device = repl.resolve_device

BATCH_SIZE = repl.BATCH_SIZE
LR = repl.LR
WEIGHT_DECAY = repl.WEIGHT_DECAY
GRAD_CLIP = repl.GRAD_CLIP
COMPONENT_LOSS_WEIGHT = repl.COMPONENT_LOSS_WEIGHT
TRAIN_SHUFFLE_OFFSET = repl.TRAIN_SHUFFLE_OFFSET
EXPECTED_PARAMETERS = repl.EXPECTED_PARAMETERS


# ---------------------------------------------------------------------------
# 0. frozen route rule, estimator roster, aggregation semantics, gates
# ---------------------------------------------------------------------------

def estimator_specs(route: str) -> dict[str, Any]:
    """The frozen estimator roster of each route (members / candidacy)."""
    ctrl = {
        "members": list(range(236, 241)), "candidate": False,
        "effective_epochs": 240, "contiguous_last5": True,
        "window": "236..240",
        "note": "the same-trajectory control (historical soup semantics)",
    }
    wide = {
        "members": [200, 210, 220, 230, 240], "candidate": True,
        "effective_epochs": 240, "contiguous_last5": False,
        "window": "200,210,220,230,240",
        "note": "the single purchased fixed aggregation: K=5, wider time spread",
    }
    if route == "OVERFIT":
        early = {
            "members": list(range(116, 121)), "candidate": True,
            "effective_epochs": 120, "contiguous_last5": True,
            "window": "116..120",
            "note": "fixed at epoch 120 by deep copy; later training cannot move it",
        }
        return {"epochs": 240, "estimators": {
            "CTRL240": ctrl, "EARLY120": early, "WIDE5_240": wide}}
    if route == "STILL_IMPROVING":
        long360 = {
            "members": list(range(356, 361)), "candidate": True,
            "effective_epochs": 360, "contiguous_last5": True,
            "window": "356..360",
            "note": "the longer-horizon fixed window",
        }
        return {"epochs": 360, "estimators": {
            "CTRL240": ctrl, "LONG360": long360, "WIDE5_240": wide}}
    if route == "FLAT_OR_MIXED":
        return {"epochs": 240, "estimators": {"CTRL240": ctrl, "WIDE5_240": wide}}
    raise ValueError(f"unknown route {route!r}")


def route_from_diagnosis(
    fit_g_mae: Mapping[int, Mapping[str, float]],
    dev_g_mae: Mapping[int, Mapping[str, float]],
    *,
    seeds: Sequence[int] = SEEDS,
    scale: float = ROUTE_SCALE,
) -> dict[str, Any]:
    """The frozen routing rule (single checkpoints 120/240, never the soup).

    F_t = two-seed mean eval fit g-MAE at epoch t; d_s = dev g-MAE(240, s)
    - dev g-MAE(120, s).  OVERFIT: F240-F120 <= -scale AND both d_s >= 0 AND
    mean(d_s) >= +scale.  STILL_IMPROVING: F240-F120 <= -scale AND both
    d_s <= 0 AND mean(d_s) <= -scale.  Everything else FLAT_OR_MIXED.
    """
    seeds = tuple(int(s) for s in seeds)
    f120 = float(np.mean([fit_g_mae[s]["epoch120"] for s in seeds]))
    f240 = float(np.mean([fit_g_mae[s]["epoch240"] for s in seeds]))
    d = {s: float(dev_g_mae[s]["epoch240"] - dev_g_mae[s]["epoch120"]) for s in seeds}
    d_mean = float(np.mean([d[s] for s in seeds]))
    fit_still_improving = bool(f240 - f120 <= -scale)
    if fit_still_improving and all(d[s] >= 0 for s in seeds) and d_mean >= scale:
        route = "OVERFIT"
    elif fit_still_improving and all(d[s] <= 0 for s in seeds) and d_mean <= -scale:
        route = "STILL_IMPROVING"
    else:
        route = "FLAT_OR_MIXED"
    return {
        "route": route,
        "F120": f120, "F240": f240, "F240_minus_F120": f240 - f120,
        "d_s": {str(s): d[s] for s in seeds}, "mean_d": d_mean,
        "scale": float(scale),
        "rule": (
            "OVERFIT: F240-F120 <= -0.002 and both d_s >= 0 and mean(d_s) >= +0.002; "
            "STILL_IMPROVING: F240-F120 <= -0.002 and both d_s <= 0 and mean(d_s) <= -0.002; "
            "else FLAT_OR_MIXED; single checkpoints 120/240 only; the scale is this "
            "round's budget-routing engineering number, not a significance standard"
        ),
    }


def average_states(
    member_states: Sequence[Mapping[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Equal-weight FP32 mean of full member states (source-round soup
    semantics); non-float entries must be identical and are copied (integers
    are never float-averaged).  Returns independent tensors."""
    members = [dict(m) for m in member_states]
    if not members:
        raise RuntimeError("no members to average")
    keys = sorted(members[0])
    for m in members[1:]:
        if sorted(m) != keys:
            raise RuntimeError("member state key sets differ")
    out: dict[str, torch.Tensor] = {}
    for key in keys:
        tensors = [m[key] for m in members]
        if tensors[0].is_floating_point():
            out[key] = torch.stack([t.detach().cpu().float() for t in tensors]).mean(0).clone()
        else:
            for t in tensors[1:]:
                if not torch.equal(t, tensors[0]):
                    raise RuntimeError(f"non-float state entry {key!r} differs across members")
            out[key] = tensors[0].detach().cpu().clone()
    return out


def _capture_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Deterministic deep copy of the current state (no live references)."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def candidate_paired_bootstrap(
    errs: Mapping[str, np.ndarray],
    group_of_row: np.ndarray,
    control: str,
    candidates: Sequence[str],
    *,
    seeds: Sequence[int] = SEEDS,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> dict[str, Any]:
    """General candidate-vs-control group-paired bootstrap.

    Canonical-SMILES-group resampling of the dev rows (2000 draws, seed
    20261007); the SAME group picks are shared by every estimator and both
    seeds within a draw; each draw first computes the per-seed paired MAE
    difference (candidate - control) and then averages the two seeds before
    the CI is taken.  Molecular sampling only — no training-seed/basis
    uncertainty, not corrected for branch/candidate selection.
    """
    seeds = tuple(int(s) for s in seeds)
    keys = [f"{name}_s{s}" for s in seeds for name in [control, *candidates]]
    for key in keys:
        if key not in errs:
            raise RuntimeError(f"bootstrap expects {key}, missing")
        if np.asarray(errs[key]).shape != np.asarray(group_of_row).shape:
            raise RuntimeError(f"row misalignment at {key}")
    labels = np.asarray([str(v) for v in np.asarray(group_of_row, dtype=object).tolist()], dtype=object)
    order = {label: index for index, label in enumerate(sorted(set(labels.tolist())))}
    g_index = np.asarray([order[label] for label in labels.tolist()], np.int64)
    n_groups = len(order)
    cnt = np.zeros(n_groups, np.int64)
    np.add.at(cnt, g_index, 1)
    sums = {k: np.zeros(n_groups, np.float64) for k in keys}
    for k in keys:
        np.add.at(sums[k], g_index, np.asarray(errs[k], np.float64))
    rng = np.random.default_rng(int(seed))
    avg = {c: np.empty(int(n_boot)) for c in candidates}
    per_seed = {c: {s: np.empty(int(n_boot)) for s in seeds} for c in candidates}
    for b in range(int(n_boot)):
        pick = rng.integers(0, n_groups, n_groups)   # shared across ALL estimators/seeds
        n = int(cnt[pick].sum())
        m = {k: float(sums[k][pick].sum() / n) for k in keys}
        for c in candidates:
            diffs = {s: m[f"{c}_s{s}"] - m[f"{control}_s{s}"] for s in seeds}
            for s in seeds:
                per_seed[c][s][b] = diffs[s]
            avg[c][b] = float(np.mean([diffs[s] for s in seeds]))
    return {
        "n_rows": int(cnt.sum()),
        "n_groups": int(n_groups),
        "n_boot": int(n_boot),
        "boot_seed": int(seed),
        "control": control,
        "shared_group_resampling": True,
        "per_candidate": {
            c: {
                "avg_delta_ci95": [float(np.percentile(avg[c], 2.5)), float(np.percentile(avg[c], 97.5))],
                "avg_delta_mean": float(avg[c].mean()),
                "per_seed_delta_ci95": {
                    str(s): [float(np.percentile(per_seed[c][s], 2.5)), float(np.percentile(per_seed[c][s], 97.5))]
                    for s in seeds
                },
            }
            for c in candidates
        },
        "reading": (
            "CI covers molecule resampling only — not training-seed or basis "
            "uncertainty; not corrected for the Stage-A branch or candidate "
            "selection; 2 body seeds are not a method-stability claim"
        ),
    }


def retention_gate(
    candidates: Sequence[str],
    delta_y: Mapping[str, Mapping[int, float]],
    delta_g: Mapping[str, Mapping[int, float]],
    mean_y_raw: Mapping[str, float],
    responsive: Mapping[str, bool],
    checks_ok: bool,
    *,
    seeds: Sequence[int] = SEEDS,
) -> dict[str, Any]:
    """The frozen 'retained-performance candidate' gate + tie-break.

    Gate (this round's resource gate, never significance/SOTA): both seeds'
    full y_raw improve (Delta_y < 0) AND two-seed mean Delta_y <= -0.003;
    both seeds' g-MAE not worse than +1e-4 AND the two-seed mean g-MAE
    improves; both seeds' alpha intervention responsive beyond the marker;
    model/source/training checks without substantive problems.
    Tie-break: smallest two-seed mean full y_raw; < 1e-4 differences tie and
    prefer the SHORTER effective training length; still tied -> the
    contiguous last-five-epoch mean.
    """
    seeds = tuple(int(s) for s in seeds)
    table: dict[str, dict[str, Any]] = {}
    for c in candidates:
        dy = {s: float(delta_y[c][s]) for s in seeds}
        dg = {s: float(delta_g[c][s]) for s in seeds}
        mean_dy = float(np.mean([dy[s] for s in seeds]))
        mean_dg = float(np.mean([dg[s] for s in seeds]))
        conds = {
            "both_seeds_y_improve": bool(all(dy[s] < GATE_DELTA_Y_SEED for s in seeds)),
            "mean_delta_y_le_gate": bool(mean_dy <= GATE_DELTA_Y_MEAN),
            "both_seeds_g_within_band": bool(all(dg[s] <= GATE_G_BAND for s in seeds)),
            "mean_g_improves": bool(mean_dg < 0.0),
            "both_seeds_alpha_responsive": bool(responsive[c]),
            "checks_ok": bool(checks_ok),
        }
        table[c] = {
            "delta_y": {str(s): dy[s] for s in seeds},
            "delta_g": {str(s): dg[s] for s in seeds},
            "mean_delta_y": mean_dy,
            "mean_delta_g": mean_dg,
            "conditions": conds,
            "passed": bool(all(conds.values())),
        }
    passed = [c for c in candidates if table[c]["passed"]]
    winner: str | None = None
    tie_break_applied = None
    if len(passed) == 1:
        winner = passed[0]
    elif passed:
        best = min(float(mean_y_raw[c]) for c in passed)
        tied = [c for c in passed if float(mean_y_raw[c]) - best < TIE_BREAK_Y]
        if len(tied) == 1:
            winner = tied[0]
            tie_break_applied = "mean y_raw"
        else:
            pool = {}
            for route in ("OVERFIT", "STILL_IMPROVING", "FLAT_OR_MIXED"):
                pool.update(estimator_specs(route)["estimators"])
            winner = sorted(
                tied,
                key=lambda c: (int(pool[c]["effective_epochs"]), 0 if pool[c]["contiguous_last5"] else 1),
            )[0]
            tie_break_applied = "shorter effective training length, then contiguous last-five mean"
    return {
        "gate_frozen": {
            "both_seeds_y_improve": "Delta_y_s < 0 for both seeds",
            "mean_delta_y": f"<= {GATE_DELTA_Y_MEAN}",
            "g_band": f"per-seed Delta_g <= +{GATE_G_BAND}, mean Delta_g < 0",
            "alpha_response": "p95 |dPred| > max(1e-4, 10*eta) on both seeds",
            "checks": "model/source/training checks without substantive problems",
        },
        "per_candidate": table,
        "passed_candidates": passed,
        "winner": winner,
        "tie_break_applied": tie_break_applied,
        "reading": (
            "this is THIS round's resource gate — never a significance or SOTA "
            "claim; a bootstrap CI crossing 0 is reported as uncertainty; "
            "failing candidates are recorded in full, never hidden"
        ),
    }


# ---------------------------------------------------------------------------
# 1. shared loaders (read-only source round)
# ---------------------------------------------------------------------------

def traj_dir(seed: int, out_dir: Path = RESULTS_DIR) -> Path:
    return Path(out_dir) / "runs" / f"traj_s{int(seed)}"


def diag_dir(seed: int, out_dir: Path = RESULTS_DIR) -> Path:
    return Path(out_dir) / "runs" / f"diag_s{int(seed)}"


def load_old_run(seed: int) -> dict[str, Any]:
    """The historical DICT_s{seed} run (read-only, hash-checked)."""
    rdir = SOURCE_RUNS / f"DICT_s{int(seed)}"
    manifest = read_json(rdir / "manifest.json")
    if manifest["arm"] != "DICT" or int(manifest["seed"]) != int(seed):
        raise RuntimeError(f"old run dir {rdir} is not DICT s{seed}")
    if manifest["stopped_reason"] != "completed":
        raise RuntimeError(f"old run DICT s{seed} did not complete")
    out: dict[str, Any] = {"rdir": rdir, "manifest": manifest}
    for name, key in (("soup_state.pt", "soup_state_sha256"), ("init_state.pt", "init_state_sha256"),
                      ("last_state.pt", "last_state_sha256")):
        state = torch.load(rdir / name, map_location="cpu", weights_only=False)
        if state_hash(state) != manifest[key]:
            raise RuntimeError(f"old {name} hash mismatch for DICT s{seed}")
        out[name.split("_")[0]] = state
    # epoch240 must be the last state (the historical horizon)
    e240 = torch.load(rdir / "epoch240_state.pt", map_location="cpu", weights_only=False)
    if state_hash(e240) != manifest["last_state_sha256"]:
        raise RuntimeError(f"old epoch240 state != last state for DICT s{seed}")
    for name, _epoch in CKPTS:
        out[f"{name}_file_sha256"] = file_sha256(rdir / f"{name}_state.pt")
    with np.load(rdir / "schedule.npz", allow_pickle=False) as z:
        out["schedule_order"] = np.asarray(z["order"], np.int64)
    if out["schedule_order"].shape != (240, 8001):
        raise RuntimeError(f"old schedule shape {out['schedule_order'].shape} != (240, 8001)")
    out["fit_predictions_file_sha256"] = file_sha256(rdir / "fit_predictions.npz")
    out["dev_predictions_file_sha256"] = file_sha256(rdir / "dev_predictions.npz")
    return out


def load_round_context() -> dict[str, Any]:
    """Round objects + prepared data + frozen tensors (hash-checked on load)."""
    round_objects = repl.load_round_objects()
    _prep_meta, fit_data, dev_data = repl.build_round_data(round_objects)
    objects = round_objects["objects"]
    payload = prev.TuplePayload(objects["payload_arrays"])
    kappa_M = float(objects["kappa"]["kappa_M"])
    basis_parts = repl.frozen_basis_parts(round_objects["basis"])
    return {
        "round_objects": round_objects, "objects": objects,
        "basis": round_objects["basis"], "q_soup": round_objects["q_soup"],
        "basis_parts": basis_parts, "payload": payload, "kappa_M": kappa_M,
        "fit_idx": round_objects["fit_idx"], "dev_idx": round_objects["dev_idx"],
        "fit_data": fit_data, "dev_data": dev_data,
        "y_fit": np.asarray(objects["targets"]["y"], np.float64)[round_objects["fit_idx"]],
        "g_fit": np.asarray(objects["targets"]["g"], np.float64)[round_objects["fit_idx"]],
        "ell_fit": np.asarray(objects["targets"]["ell"], np.float64)[round_objects["fit_idx"]],
        "s_fit": np.asarray(objects["targets"]["s"], np.float64)[round_objects["fit_idx"]],
    }


def _predict_rows_checked(
    model: torch.nn.Module, rows: Sequence[Any], device: torch.device,
) -> dict[str, Any]:
    """Batched eval predictions with the two identity checks kept separate.

    h / ell_hat / s_hat come from the SAME forward pass; the torch-internal
    component identity (components.sum(-1) vs the model's total output) must
    be exactly 0, while the exported float64 sum ell_hat+s_hat vs h only has
    to agree within FP32 export rounding.
    """
    hs, ells, ss = [], [], []
    torch_gap = 0.0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), BATCH_SIZE):
            chunk = list(range(start, min(start + BATCH_SIZE, len(rows))))
            batch = zftd.make_batch(rows, chunk, torch.zeros(len(rows)), device)
            prediction = model(batch, mask=cm.C6_MASK)
            components = model.reader.components()
            torch_gap = max(torch_gap, float((components.sum(-1) - prediction).abs().max().item()))
            hs.append(prediction.detach().cpu().numpy().astype(np.float64))
            ells.append(components[:, 0].detach().cpu().numpy().astype(np.float64))
            ss.append(components[:, 1].detach().cpu().numpy().astype(np.float64))
    h = np.concatenate(hs); ell_hat = np.concatenate(ells); s_hat = np.concatenate(ss)
    if torch_gap != 0.0:
        raise RuntimeError(f"torch-internal component identity broken (gap {torch_gap})")
    f64_gap = float(np.max(np.abs(h - (ell_hat + s_hat))))
    if f64_gap > FLOAT64_SUM_TOL:
        raise RuntimeError(f"exported float64 component sum off h by {f64_gap}")
    return {"h": h, "ell_hat": ell_hat, "s_hat": s_hat,
            "torch_identity_max": torch_gap, "float64_sum_max_abs": f64_gap}


def _component_stats(ell_hat: np.ndarray, s_hat: np.ndarray, h: np.ndarray,
                     ell: np.ndarray, s: np.ndarray, g: np.ndarray) -> dict[str, float]:
    e_ell = np.asarray(ell_hat, np.float64) - np.asarray(ell, np.float64)
    e_s = np.asarray(s_hat, np.float64) - np.asarray(s, np.float64)
    e_g = np.asarray(h, np.float64) - np.asarray(g, np.float64)
    return {
        "mean_e_ell": float(e_ell.mean()), "mean_e_s": float(e_s.mean()),
        "mean_e_g": float(e_g.mean()),
        "opposite_sign_fraction_ell_s": float((np.sign(e_ell) * np.sign(e_s) < 0).mean()),
        "triangle_gap_mean": float(np.mean(np.abs(e_ell) + np.abs(e_s) - np.abs(e_g))),
        "mae_ell": float(np.abs(e_ell).mean()), "mae_s": float(np.abs(e_s).mean()),
        "mae_g": float(np.abs(e_g).mean()),
        "note": "component MAEs are never summed into an error budget or loss weights",
    }


def _k_group_table(e_y: np.ndarray, e_g: np.ndarray, k: np.ndarray) -> dict[str, Any]:
    n = int(len(e_y))
    out: dict[str, Any] = {}
    for name, pred in K_GROUP_DEFS:
        mask = pred(np.asarray(k, np.int64))
        out[name] = {
            "n": int(mask.sum()),
            "C_y": float(np.abs(e_y[mask]).sum() / n),
            "C_g": float(np.abs(e_g[mask]).sum() / n),
            "in_group_mae_y": float(np.abs(e_y[mask]).mean()) if mask.any() else None,
            "in_group_mae_g": float(np.abs(e_g[mask]).mean()) if mask.any() else None,
        }
    if sum(v["n"] for v in out.values()) != n:
        raise RuntimeError("k groups do not cover all rows")
    if abs(sum(v["C_y"] for v in out.values()) - float(np.abs(e_y).mean())) > 1e-9:
        raise RuntimeError("k-group C_y add-back failed")
    return out


def _nnodes_bin(n_nodes: np.ndarray) -> np.ndarray:
    n = np.asarray(n_nodes, np.int64)
    return np.where(n <= NNODE_EDGES[0], 0, np.where(n <= NNODE_EDGES[1], 1, 2))


def _k0_subgroup_table(e_y: np.ndarray, e_g: np.ndarray, k: np.ndarray,
                       n_nodes: np.ndarray, g_abs: np.ndarray,
                       g_p90: float) -> dict[str, Any]:
    """The two pre-registered k=0 grouping axes (analysis only)."""
    n = int(len(e_y))
    k0 = np.asarray(k, np.int64) == 0
    out: dict[str, Any] = {"g_abs_p90_fit": float(g_p90), "n_k0": int(k0.sum())}
    bins = {
        "n_nodes<=23": _nnodes_bin(n_nodes) == 0,
        "n_nodes=24..27": _nnodes_bin(n_nodes) == 1,
        "n_nodes>=28": _nnodes_bin(n_nodes) == 2,
        "|g|<=fit_p90": g_abs <= g_p90,
        "|g|>fit_p90": g_abs > g_p90,
    }
    for name, mask in bins.items():
        m = k0 & mask
        out[name] = {
            "n": int(m.sum()),
            "C_y": float(np.abs(e_y[m]).sum() / n),
            "C_g": float(np.abs(e_g[m]).sum() / n),
            "in_group_mae_y": float(np.abs(e_y[m]).mean()) if m.any() else None,
            "in_group_mae_g": float(np.abs(e_g[m]).mean()) if m.any() else None,
        }
    if sum(out[a]["n"] for a in ("n_nodes<=23", "n_nodes=24..27", "n_nodes>=28")) != int(k0.sum()):
        raise RuntimeError("n_nodes bins do not cover k=0")
    if sum(out[a]["n"] for a in ("|g|<=fit_p90", "|g|>fit_p90")) != int(k0.sum()):
        raise RuntimeError("|g| bins do not cover k=0")
    return out


# ---------------------------------------------------------------------------
# 2. stage: source-checks (CPU, read-only)
# ---------------------------------------------------------------------------

def source_checks(*, out_dir: Path = RESULTS_DIR, log: Any = print) -> dict[str, Any]:
    torch.set_num_threads(8)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    ctx = load_round_context()
    objects = ctx["objects"]
    payload, kappa_M, basis_parts = ctx["payload"], ctx["kappa_M"], ctx["basis_parts"]
    fit_idx, dev_idx = ctx["fit_idx"], ctx["dev_idx"]
    n_fit = int(fit_idx.size)

    old = {s: load_old_run(s) for s in SEEDS}

    # (a) schedule prefix: a 360-epoch schedule's first 240 epochs must be
    #     bit-identical to the historical 240-epoch schedule of the same seed
    prefix = {}
    for s in SEEDS:
        sched360, hash360 = zw.build_schedule(n_fit, EPOCHS_CAP, int(s) + TRAIN_SHUFFLE_OFFSET)
        sched240, hash240 = zw.build_schedule(n_fit, 240, int(s) + TRAIN_SHUFFLE_OFFSET)
        ok_arrays = all(
            np.array_equal(sched360[e], old[s]["schedule_order"][e]) for e in range(240)
        ) and all(np.array_equal(sched240[e], old[s]["schedule_order"][e]) for e in range(240))
        if not ok_arrays:
            raise RuntimeError(f"schedule prefix mismatch for seed {s}")
        prefix[str(s)] = {
            "old_schedule_sha256": old[s]["manifest"]["schedule_sha256"],
            "rebuilt_240_sha256": hash240, "rebuilt_360_first240_equals_old": True,
            "rebuilt_360_sha256": hash360,
        }

    # (b) init reproduction (construction only, no forward) + cross-seed identity
    init_hashes = {}
    for s in SEEDS:
        seed_everything(int(s))
        model = repl.build_arm(ARM, payload, kappa_M, basis_parts, int(s))
        h = state_hash(_capture_state(model))
        if h != old[s]["manifest"]["init_state_sha256"]:
            raise RuntimeError(f"seed {s} init state does not reproduce the historical init")
        init_hashes[str(s)] = h
    if init_hashes["0"] == init_hashes["1"]:
        raise RuntimeError("seed 1 init is not genuinely different from seed 0")

    # (c) frozen basis / Q consistency with the historical runs
    for s in SEEDS:
        if old[s]["manifest"]["frozen_basis_hashes"] != basis_parts["hashes"]:
            raise RuntimeError(f"frozen basis hashes differ from the historical run (seed {s})")
        if state_hash(ctx["q_soup"]) != old[s]["manifest"]["q_soup_sha256"]:
            raise RuntimeError(f"Q soup hash differs from the historical run (seed {s})")

    # (d) old saved predictions present and gid-aligned with the round fold view
    targets = objects["targets"]
    gid_all = np.asarray(targets["gid"], np.int64)
    for s in SEEDS:
        with np.load(old[s]["rdir"] / "fit_predictions.npz", allow_pickle=False) as z:
            if not np.array_equal(z["gid"], gid_all[fit_idx]):
                raise RuntimeError(f"old fit predictions gid misaligned (seed {s})")
        with np.load(old[s]["rdir"] / "dev_predictions.npz", allow_pickle=False) as z:
            if not np.array_equal(z["gid"], gid_all[dev_idx]):
                raise RuntimeError(f"old dev predictions gid misaligned (seed {s})")

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "stage": "source-checks",
        "source_round": str(SOURCE_DIR.relative_to(TRACK_ROOT)),
        "source_read_only": True,
        "arm": ARM,
        "parameter_contract": EXPECTED_PARAMETERS,
        "fit_n": n_fit, "dev_n": int(dev_idx.size),
        "schedule_prefix": prefix,
        "init_state_sha256": init_hashes,
        "init_seed0_ne_seed1": True,
        "frozen_basis_hashes": basis_parts["hashes"],
        "q_soup_sha256": state_hash(ctx["q_soup"]),
        "old_runs": {
            str(s): {
                "run_dir": str(old[s]["rdir"].relative_to(TRACK_ROOT)),
                "manifest_hashes_verified": True,
                "epoch240_equals_last": True,
                "state_file_sha256": {name: old[s][f"{name}_file_sha256"] for name, _ in CKPTS},
                "predictions_file_sha256": {
                    "fit": old[s]["fit_predictions_file_sha256"],
                    "dev": old[s]["dev_predictions_file_sha256"],
                },
            }
            for s in SEEDS
        },
        "reuse_note": (
            "no basis/Q/prep/target refit; the historical DICT trajectories are "
            "read-only diagnostic inputs; official valid/test never loaded"
        ),
        "all_passed": True,
        "seconds": float(time.perf_counter() - started),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(out_dir / "source_manifest.json", manifest)
    log(f"[source-checks] fit={n_fit} dev={dev_idx.size} verified in {manifest['seconds']:.1f}s")
    return manifest


# ---------------------------------------------------------------------------
# 3. stage A: checkpoint diagnostics (per seed; read-only forward exports)
# ---------------------------------------------------------------------------

def checkpoint_diagnostics(
    seed: int,
    *,
    device_name: str = "cuda:0",
    out_dir: Path = RESULTS_DIR,
    log: Any = print,
) -> dict[str, Any]:
    if int(seed) not in SEEDS:
        raise ValueError(seed)
    device = resolve_device(device_name)
    torch.set_num_threads(8 if device.type == "cpu" else 4)
    out_dir = Path(out_dir)
    ddir = diag_dir(seed, out_dir)
    ddir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    ctx = load_round_context()
    objects, basis_parts = ctx["objects"], ctx["basis_parts"]
    payload, kappa_M = ctx["payload"], ctx["kappa_M"]
    fit_idx, dev_idx = ctx["fit_idx"], ctx["dev_idx"]
    fit_data, dev_data = ctx["fit_data"], ctx["dev_data"]
    targets = objects["targets"]
    y = np.asarray(targets["y"], np.float64)
    g = np.asarray(targets["g"], np.float64)
    ell = np.asarray(targets["ell"], np.float64)
    s_t = np.asarray(targets["s"], np.float64)
    k_all = np.asarray(targets["k"], np.int64)
    gid_all = np.asarray(targets["gid"], np.int64)
    y_fit, g_fit, ell_fit, s_fit = y[fit_idx], g[fit_idx], ell[fit_idx], s_t[fit_idx]
    y_dev, g_dev, ell_dev, s_dev = y[dev_idx], g[dev_idx], ell[dev_idx], s_t[dev_idx]
    k_fit, k_dev = k_all[fit_idx], k_all[dev_idx]

    # the two pre-registered grouping inputs (frozen before any scoring)
    _phi, _atom, node_sizes = prev._env_phi_atom()
    n_nodes_fit = np.asarray(node_sizes, np.int64)[fit_idx]
    n_nodes_dev = np.asarray(node_sizes, np.int64)[dev_idx]
    g_p90 = float(np.percentile(np.abs(g_fit), 90))

    # Q is fixed: one shared q_raw per split, cached once
    q_fit = nb._q_predictions(ctx["q_soup"], nb.topology_matrix(fit_data), device)
    q_dev = nb._q_predictions(ctx["q_soup"], nb.topology_matrix(dev_data), device)

    old = load_old_run(seed)
    with np.load(old["rdir"] / "fit_predictions.npz", allow_pickle=False) as z:
        old_fit = {key: z[key] for key in z.files}
    with np.load(old["rdir"] / "dev_predictions.npz", allow_pickle=False) as z:
        old_dev = {key: z[key] for key in z.files}

    diag: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "stage": "checkpoint-diagnostics",
        "arm": ARM, "seed": int(seed), "n_fit": int(fit_idx.size), "n_dev": int(dev_idx.size),
        "g_abs_p90_fit": g_p90,
        "checkpoints": {},
    }
    for name, _epoch in CKPTS:
        ck_started = time.perf_counter()
        state = torch.load(old["rdir"] / f"{name}_state.pt", map_location="cpu", weights_only=False)
        model = repl.build_arm(ARM, payload, kappa_M, basis_parts, int(seed))
        model.load_state_dict({kk: vv for kk, vv in state.items()}, strict=True)
        model = model.to(device).eval()
        fitp = _predict_rows_checked(model, fit_data, device)
        devp = _predict_rows_checked(model, dev_data, device)
        y_raw_fit = fitp["ell_hat"] + fitp["s_hat"] + q_fit
        y_raw_dev = devp["ell_hat"] + devp["s_hat"] + q_dev
        b_y = float(np.median(y_fit - y_raw_fit))   # each checkpoint's OWN fit residual

        for split, rows_idx, preds, y_raw, tgt in (
            ("fit", fit_idx, fitp, y_raw_fit, (y_fit, g_fit, ell_fit, s_fit, k_fit)),
            ("dev", dev_idx, devp, y_raw_dev, (y_dev, g_dev, ell_dev, s_dev, k_dev)),
        ):
            np.savez_compressed(
                ddir / f"{name}_{split}_predictions.npz",
                row_index=np.asarray(rows_idx, np.int64), gid=gid_all[rows_idx],
                y=tgt[0], g=tgt[1], ell=tgt[2], s=tgt[3], k=tgt[4],
                h=preds["h"], ell_hat=preds["ell_hat"], s_hat=preds["s_hat"],
                q_raw=q_fit if split == "fit" else q_dev, y_raw=y_raw,
            )

        # soup replay must match the saved predictions (cross-device FP32 band)
        replay = {}
        if name == "soup":
            for field in ("h", "ell_hat", "s_hat", "y_raw"):
                new_val = y_raw_fit if field == "y_raw" else fitp[field]
                d = float(np.max(np.abs(np.asarray(old_fit[field], np.float64) - new_val)))
                replay[f"fit_{field}_max_abs"] = d
                if d > CROSS_DEVICE_TOL:
                    raise RuntimeError(f"soup fit replay mismatch at {field}: {d}")
            for field in ("h", "y_raw", "q_raw"):
                new_val = devp["h"] if field == "h" else (y_raw_dev if field == "y_raw" else q_dev)
                d = float(np.max(np.abs(np.asarray(old_dev[field], np.float64) - new_val)))
                replay[f"dev_{field}_max_abs"] = d
                if d > CROSS_DEVICE_TOL:
                    raise RuntimeError(f"soup dev replay mismatch at {field}: {d}")

        e_y_dev = y_raw_dev - y_dev
        e_g_dev = devp["h"] - g_dev
        ck = {
            "state_file_sha256": old[f"{name}_file_sha256"],
            "fit": {
                "g_mae": float(np.abs(g_fit - fitp["h"]).mean()),
                "y_raw_mae": float(np.abs(y_fit - y_raw_fit).mean()),
                "y_cal_mae": float(np.abs(y_fit - (y_raw_fit + b_y)).mean()),
                "ell_mae": float(np.abs(ell_fit - fitp["ell_hat"]).mean()),
                "s_mae": float(np.abs(s_fit - fitp["s_hat"]).mean()),
                "component_stats": _component_stats(
                    fitp["ell_hat"], fitp["s_hat"], fitp["h"], ell_fit, s_fit, g_fit),
                "k_groups": _k_group_table(y_raw_fit - y_fit, fitp["h"] - g_fit, k_fit),
            },
            "dev": {
                "g_mae": float(np.abs(g_dev - devp["h"]).mean()),
                "y_raw_mae": float(np.abs(y_dev - y_raw_dev).mean()),
                "y_cal_mae": float(np.abs(y_dev - (y_raw_dev + b_y)).mean()),
                "ell_mae": float(np.abs(ell_dev - devp["ell_hat"]).mean()),
                "s_mae": float(np.abs(s_dev - devp["s_hat"]).mean()),
                "component_stats": _component_stats(
                    devp["ell_hat"], devp["s_hat"], devp["h"], ell_dev, s_dev, g_dev),
                "k_groups": _k_group_table(e_y_dev, e_g_dev, k_dev),
                "k0_subgroups": _k0_subgroup_table(
                    e_y_dev, e_g_dev, k_dev, n_nodes_dev, np.abs(g_dev), g_p90),
            },
            "b_y": b_y,
            "identity": {
                "torch_identity_max": devp["torch_identity_max"],
                "float64_sum_max_abs": max(fitp["float64_sum_max_abs"], devp["float64_sum_max_abs"]),
            },
            "soup_replay": replay or None,
            "seconds": float(time.perf_counter() - ck_started),
        }
        diag["checkpoints"][name] = ck
        log(
            f"[diag s{seed} {name}] fit g {ck['fit']['g_mae']:.5f} y {ck['fit']['y_raw_mae']:.5f} | "
            f"dev g {ck['dev']['g_mae']:.5f} y {ck['dev']['y_raw_mae']:.5f} "
            f"({ck['seconds']:.0f}s)",
            flush=True,
        )

    diag["route_inputs"] = {
        "fit_g_mae": {name: diag["checkpoints"][name]["fit"]["g_mae"] for name, _ in CKPTS},
        "dev_g_mae": {name: diag["checkpoints"][name]["dev"]["g_mae"] for name, _ in CKPTS},
        "rule_note": "the route uses the single epoch120/epoch240 checkpoints only, never the soup",
    }
    diag["seconds"] = float(time.perf_counter() - started)
    diag["official_valid_loaded"] = False
    diag["official_test_loaded"] = False
    write_json(ddir / "diagnosis.json", diag)
    return diag


# ---------------------------------------------------------------------------
# 4. stage: freeze-roster (mechanical, one-shot)
# ---------------------------------------------------------------------------

def freeze_roster(*, out_dir: Path = RESULTS_DIR, log: Any = print) -> dict[str, Any]:
    out_dir = Path(out_dir)
    lock_path = out_dir / "roster_lock.json"
    if lock_path.exists():
        raise RuntimeError("roster_lock.json already exists: the roster freeze is one-shot")
    diags = {}
    for s in SEEDS:
        d = read_json(diag_dir(s, out_dir) / "diagnosis.json")
        if d.get("seed") != int(s):
            raise RuntimeError(f"diagnosis seed mismatch at {s}")
        diags[int(s)] = d
    fit_g = {s: diags[s]["route_inputs"]["fit_g_mae"] for s in SEEDS}
    dev_g = {s: diags[s]["route_inputs"]["dev_g_mae"] for s in SEEDS}
    route = route_from_diagnosis(fit_g, dev_g)
    specs = estimator_specs(route["route"])
    roster = {
        "protocol_version": PROTOCOL_VERSION,
        "stage": "freeze-roster",
        "one_shot": True,
        "route": route,
        "epochs": specs["epochs"],
        "estimators": specs["estimators"],
        "candidates": [n for n, e in specs["estimators"].items() if e["candidate"]],
        "control": "CTRL240",
        "seeds": list(SEEDS),
        "bootstrap": {
            "n_boot": N_BOOT, "seed": BOOT_SEED,
            "grouping": "canonical-SMILES groups of the dev rows",
            "shared_resampling": True, "avg_seeds_first": True,
        },
        "gates": {
            "delta_y_mean": GATE_DELTA_Y_MEAN, "delta_y_seed": GATE_DELTA_Y_SEED,
            "g_band": GATE_G_BAND, "response_marker": "p95 > max(1e-4, 10*eta)",
        },
        "tie_break": (
            "smallest two-seed mean full y_raw; differences < 1e-4 tie and prefer "
            "the shorter effective training length; still tied -> contiguous "
            "last-five-epoch mean"
        ),
        "diagnosis_files": {
            str(s): file_sha256(diag_dir(s, out_dir) / "diagnosis.json") for s in SEEDS
        },
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(lock_path, roster)
    log(
        f"[freeze-roster] route={route['route']} epochs={specs['epochs']} "
        f"candidates={roster['candidates']}"
    )
    return roster


def load_roster(*, out_dir: Path = RESULTS_DIR) -> dict[str, Any]:
    roster = read_json(Path(out_dir) / "roster_lock.json")
    if roster.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("roster_lock.json is from a different protocol version")
    return roster


# ---------------------------------------------------------------------------
# 5. stage: train-trajectory (one seed; deterministic member capture)
# ---------------------------------------------------------------------------

def train_trajectory(
    seed: int,
    *,
    device_name: str = "cuda:0",
    out_dir: Path = RESULTS_DIR,
    estimator_override: Mapping[str, Any] | None = None,
    epochs_override: int | None = None,
    max_steps: int | None = None,
    smoke: bool = False,
    log: Any = print,
) -> dict[str, Any]:
    """One from-scratch DICT trajectory with deterministic member capture.

    Clones the source round's train_arm loop byte-for-byte (schedule, init,
    optimizer, batch order, loss, probes, curve) and only ADDS deterministic
    state capture at the roster's member epochs.  During training nothing is
    built or scored: only detach/clone.  Estimator states, strict-load
    checks and the save/reload prediction assertion happen after the loop.
    """
    if int(seed) not in SEEDS:
        raise ValueError(seed)
    device = resolve_device(device_name)
    torch.set_num_threads(8 if device.type == "cpu" else 4)
    out_dir = Path(out_dir)
    if estimator_override is None:
        roster = load_roster(out_dir=out_dir)
        epochs = int(roster["epochs"])
        estimators = roster["estimators"]
        if epochs_override is not None and int(epochs_override) != epochs:
            raise RuntimeError("epochs override conflicts with the frozen roster")
    else:
        epochs = int(epochs_override if epochs_override is not None else 2)
        estimators = dict(estimator_override)
    rdir = traj_dir(seed, out_dir)
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / "members").mkdir(exist_ok=True)

    ctx = load_round_context()
    objects = ctx["objects"]
    basis_parts = ctx["basis_parts"]
    payload, kappa_M = ctx["payload"], ctx["kappa_M"]
    fit_idx = ctx["fit_idx"]
    fit_data = ctx["fit_data"]
    targets = objects["targets"]
    y_all = np.asarray(targets["y"], np.float64)
    g_all = np.asarray(targets["g"], np.float64)
    ell_all = np.asarray(targets["ell"], np.float64)
    s_all = np.asarray(targets["s"], np.float64)
    gid_all = np.asarray(targets["gid"], np.int64)
    y_fit, g_fit = y_all[fit_idx], g_all[fit_idx]
    ell_fit, s_fit = ell_all[fit_idx], s_all[fit_idx]
    if float(np.max(np.abs(g_fit - (ell_fit + s_fit)))) > 1e-12:
        raise RuntimeError("g != ell + s on fit rows")
    target_g = torch.as_tensor(g_fit, dtype=torch.float32)
    target_ell = torch.as_tensor(ell_fit, dtype=torch.float32, device=device)
    target_s = torch.as_tensor(s_fit, dtype=torch.float32, device=device)

    schedule, schedule_hash = zw.build_schedule(len(fit_data), int(epochs), int(seed) + TRAIN_SHUFFLE_OFFSET)

    old = load_old_run(seed)
    other = load_old_run(1 - int(seed))
    # schedule prefix vs the historical 240-epoch plan of this seed
    for e in range(min(240, int(epochs))):
        if not np.array_equal(schedule[e], old["schedule_order"][e]):
            raise RuntimeError(f"schedule prefix mismatch vs the historical plan at epoch {e+1}")

    seed_everything(int(seed))
    model = repl.build_arm(ARM, payload, kappa_M, basis_parts, int(seed))
    build_rng = torch.get_rng_state().clone()
    init_state = _capture_state(model)
    init_hash = state_hash(init_state)
    if init_hash != old["manifest"]["init_state_sha256"]:
        raise RuntimeError(f"seed {seed} init does not reproduce the historical init")
    if init_hash == other["manifest"]["init_state_sha256"]:
        raise RuntimeError(f"seed {seed} init collides with the other seed's init")
    model = model.to(device)
    expected_encoder_hashes = {
        "U": array_sha256(basis_parts["U"].numpy()),
        "common_rms": array_sha256(basis_parts["common_rms"].numpy()),
        "Dbar": array_sha256(basis_parts["Dbar"].numpy()),
    }
    encoder_hashes_before = model.local_tuple.frozen_basis_hashes()
    if encoder_hashes_before != expected_encoder_hashes:
        raise RuntimeError("DICT frozen basis buffers != loaded basis package")
    seed_everything(int(seed))
    train_rng = torch.get_rng_state().clone()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    member_epochs = sorted({int(e) for spec in estimators.values() for e in spec["members"]})
    checkpoint_epochs = sorted({int(e) for e in LOG_EPOCHS if int(e) <= int(epochs)})
    capture_epochs = sorted(set(member_epochs) | set(checkpoint_epochs))
    members: dict[int, dict[str, torch.Tensor]] = {}
    checkpoints: dict[int, dict[str, torch.Tensor]] = {}
    curve: list[dict[str, Any]] = []
    probes: list[dict[str, Any]] = []
    steps_done = 0
    stopped_reason = "completed"
    started = time.perf_counter()
    peak_mb = 0.0
    position_stream = hashlib.sha256()
    gid_stream = hashlib.sha256()

    for epoch in range(1, int(epochs) + 1):
        epoch_started = time.perf_counter()
        model.train()
        g_sum = ell_sum = s_sum = total_sum = 0.0
        n_mol = n_steps = 0
        gnorm_sum = 0.0
        clip_hits = 0
        for start in range(0, len(schedule[epoch - 1]), BATCH_SIZE):
            indices = [int(i) for i in schedule[epoch - 1][start:start + BATCH_SIZE]]
            index_t = torch.as_tensor(indices, dtype=torch.long, device=device)
            position_stream.update(np.asarray(indices, np.int64).tobytes())
            gid_stream.update(np.asarray(gid_all[fit_idx[np.asarray(indices, np.int64)]], np.int64).tobytes())
            batch = zftd.make_batch(fit_data, indices, target_g, device)
            prediction = model(batch, mask=cm.C6_MASK)
            if tuple(prediction.shape) != (len(indices),):
                raise RuntimeError("standard forward must return (batch,) total g")
            components = model.reader.components()
            if tuple(components.shape) != (len(indices), 2):
                raise RuntimeError("reader component output is not (batch, 2)")
            identity_gap = float(torch.max(torch.abs(components.sum(-1) - prediction.detach())).item())
            if identity_gap != 0.0:
                raise RuntimeError(f"g_hat != ell_hat + s_hat exactly (gap {identity_gap})")
            l_g = F.l1_loss(prediction.view(-1), batch.y.view(-1))
            l_ell = F.l1_loss(components[:, 0], target_ell[index_t])
            l_s = F.l1_loss(components[:, 1], target_s[index_t])
            loss = l_g + COMPONENT_LOSS_WEIGHT * (l_ell + l_s)
            optimizer.zero_grad()
            loss.backward()
            total_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP))
            if epoch in LOG_EPOCHS and n_steps == 0:
                probes.append(repl._probe_local_channel(model, ARM, epoch, steps_done + 1, total_norm))
            optimizer.step()
            g_sum += float((prediction.detach().view(-1) - batch.y.view(-1)).abs().sum())
            ell_sum += float((components[:, 0].detach() - target_ell[index_t]).abs().sum())
            s_sum += float((components[:, 1].detach() - target_s[index_t]).abs().sum())
            total_sum += float(loss.detach()) * int(len(indices))
            n_mol += int(batch.y.numel())
            n_steps += 1
            steps_done += 1
            gnorm_sum += total_norm
            clip_hits += int(total_norm > GRAD_CLIP)
            if max_steps is not None and steps_done >= int(max_steps):
                stopped_reason = "max_steps"
                break
        if device.type == "cuda":
            peak_mb = max(peak_mb, float(torch.cuda.max_memory_allocated() / (1 << 20)))
        curve.append({
            "epoch": int(epoch),
            "train_L_g": float(g_sum / max(n_mol, 1)),
            "train_L_ell": float(ell_sum / max(n_mol, 1)),
            "train_L_s": float(s_sum / max(n_mol, 1)),
            "train_loss": float(total_sum / max(n_mol, 1)),
            "grad_norm": float(gnorm_sum / max(n_steps, 1)),
            "clip_fraction": float(clip_hits / max(n_steps, 1)),
            "seconds": float(time.perf_counter() - epoch_started),
        })
        if epoch in capture_epochs:
            state = _capture_state(model)          # deep copy: no live references
            if epoch in member_epochs:
                members[int(epoch)] = state
                torch.save(state, rdir / "members" / f"epoch{epoch}_state.pt")
            if epoch in checkpoint_epochs:
                checkpoints[int(epoch)] = state
        if log and (epoch in LOG_EPOCHS or epoch % 40 == 0):
            c = curve[-1]
            log(
                f"[traj s{seed}] ep={epoch:03d} Lg={c['train_L_g']:.5f} "
                f"Lell={c['train_L_ell']:.5f} Ls={c['train_L_s']:.5f} "
                f"gn={c['grad_norm']:.3g} {c['seconds']:.1f}s",
                flush=True,
            )
        if stopped_reason == "max_steps":
            break

    last_state = _capture_state(model)
    final_epoch = int(curve[-1]["epoch"]) if curve else 0
    member_hashes = {int(e): state_hash(members[e]) for e in sorted(members)}
    if not smoke:
        if stopped_reason != "completed":
            raise RuntimeError(f"formal trajectory stopped early: {stopped_reason}")
        for name, spec in estimators.items():
            missing = [e for e in spec["members"] if int(e) not in members]
            if missing:
                raise RuntimeError(f"estimator {name} missing captured members {missing}")
            if len(spec["members"]) != 5:
                raise RuntimeError(f"estimator {name} must average exactly five members")

    # estimator construction + strict-load + save/reload prediction assertion
    # (all AFTER training; a fixed 8-row fit probe, never dev)
    estimator_records: dict[str, dict[str, Any]] = {}
    construct_started = time.perf_counter()
    probe_rows = list(range(8))
    probe_batch = zftd.make_batch(fit_data, probe_rows, torch.zeros(8), device)
    for name, spec in estimators.items():
        if smoke:
            got = [int(e) for e in spec["members"] if int(e) in members]
            member_list = [members[e] for e in got] if got else [last_state]
        else:
            member_list = [members[int(e)] for e in spec["members"]]
        est_state = average_states(member_list)
        torch.save(est_state, rdir / f"{name}_state.pt")
        est_hash = state_hash(est_state)
        model_e = repl.build_arm(ARM, payload, kappa_M, basis_parts, int(seed))
        model_e.load_state_dict({kk: vv for kk, vv in est_state.items()}, strict=True)
        model_e = model_e.to(device).eval()
        with torch.no_grad():
            p_mem = model_e(probe_batch, mask=cm.C6_MASK).detach().cpu().clone()
        reloaded = torch.load(rdir / f"{name}_state.pt", map_location="cpu", weights_only=False)
        if state_hash(reloaded) != est_hash:
            raise RuntimeError(f"estimator {name} save/reload hash mismatch")
        model_r = repl.build_arm(ARM, payload, kappa_M, basis_parts, int(seed))
        model_r.load_state_dict({kk: vv for kk, vv in reloaded.items()}, strict=True)
        model_r = model_r.to(device).eval()
        with torch.no_grad():
            p_disk = model_r(probe_batch, mask=cm.C6_MASK).detach().cpu().clone()
        # weights are hash-verified identical; two separately allocated model
        # instances may still pick different GPU kernels (allocation-dependent
        # cuBLAS behaviour), so the prediction check carries the cross-device
        # FP32 band instead of CPU-style bitwise equality
        d_reload = float((p_mem - p_disk).abs().max())
        if d_reload > CROSS_DEVICE_TOL:
            raise RuntimeError(f"estimator {name} save/reload predictions differ by {d_reload}")
        estimator_records[name] = {
            "members": [int(e) for e in spec["members"]],
            "member_state_sha256": {str(int(e)): member_hashes[int(e)] for e in spec["members"] if int(e) in member_hashes} if not smoke else None,
            "state_sha256": est_hash,
            "candidate": bool(spec["candidate"]),
            "strict_load_ok": True,
            "save_reload_prediction_max_abs": d_reload,
            "save_reload_prediction_within_band": True,
        }
        del model_e, model_r
        if device.type == "cuda":
            torch.cuda.empty_cache()

    encoder_hashes_after = model.local_tuple.frozen_basis_hashes()
    if encoder_hashes_after != encoder_hashes_before:
        raise RuntimeError(f"traj s{seed}: frozen basis changed during training")

    # exact-resume capability record (never used for the round's numbers)
    torch.save(
        {
            "model_state": last_state,
            "optimizer_state": optimizer.state_dict(),
            "torch_rng_state": torch.get_rng_state().clone(),
            "steps_done": int(steps_done),
            "epochs_done": int(final_epoch),
            "schedule_seed": int(seed) + TRAIN_SHUFFLE_OFFSET,
        },
        rdir / "resume_state.pt",
    )
    np.savez_compressed(
        rdir / "schedule.npz",
        order=np.stack([np.asarray(e, np.int64) for e in schedule]),
    )
    torch.save(init_state, rdir / "init_state.pt")
    torch.save(last_state, rdir / "last_state.pt")
    for epoch, state in sorted(checkpoints.items()):
        torch.save(state, rdir / f"epoch{epoch}_state.pt")

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "stage": "train-trajectory",
        "arm": ARM, "seed": int(seed), "smoke": bool(smoke),
        "run_dir": str(rdir.relative_to(out_dir)),
        "recipe": {
            "epochs": int(epochs), "batch_size": BATCH_SIZE, "lr": LR,
            "weight_decay": WEIGHT_DECAY, "grad_clip": GRAD_CLIP,
            "component_loss_weight": COMPONENT_LOSS_WEIGHT,
            "train_shuffle_offset": TRAIN_SHUFFLE_OFFSET,
            "amp": None, "ddp": None, "scheduler": None, "early_stopping": None,
        },
        "estimators": estimator_records,
        "supervision": (
            "COMP: L = MAE(ell_hat+s_hat, g) + 0.5*(MAE(ell_hat,ell)+MAE(s_hat,s)) "
            "(unchanged disclosed auxiliary condition)"
        ),
        "steps_done": int(steps_done),
        "steps_expected": int(sum((len(e) + BATCH_SIZE - 1) // BATCH_SIZE for e in schedule)),
        "stopped_reason": stopped_reason,
        "schedule_sha256": schedule_hash,
        "schedule_prefix_matches_historical_240": True,
        "position_stream_sha256": position_stream.hexdigest(),
        "global_gid_stream_sha256": gid_stream.hexdigest(),
        "init_state_sha256": init_hash,
        "init_matches_historical": True,
        "init_differs_from_other_seed": True,
        "build_rng_sha256": hashlib.sha256(build_rng.numpy().tobytes()).hexdigest(),
        "train_rng_sha256": hashlib.sha256(train_rng.numpy().tobytes()).hexdigest(),
        "last_state_sha256": state_hash(last_state),
        "frozen_basis_hashes": basis_parts["hashes"],
        "encoder_frozen_basis_hashes": encoder_hashes_after,
        "frozen_basis_unchanged": True,
        "q_soup_sha256": state_hash(ctx["q_soup"]),
        "parameter_audit": repl.arm_parameter_audit(
            repl.build_arm(ARM, payload, kappa_M, basis_parts, int(seed))
        ),
        "curve_seconds_total": float(sum(c["seconds"] for c in curve)),
        "estimator_construction_seconds": float(time.perf_counter() - construct_started),
        "wall_clock_s": float(time.perf_counter() - started),
        "peak_gpu_memory_mb": float(peak_mb),
        "device": str(device),
        "allocation_probe": repl.allocation_probe(device),
        "cost_note": (
            "multiple estimators share this ONE trajectory (not independent "
            "training repeats); the reported cost is the actual shared-trajectory "
            "cost, with no virtual saving claimed for the shared epochs"
        ),
        "dev_scores_computed_during_training": False,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(rdir / "curve.json", curve)
    write_json(rdir / "probes.json", probes)
    write_json(rdir / "manifest.json", manifest)
    log(
        f"[traj s{seed}] done: {steps_done} steps, {manifest['curve_seconds_total']:.0f}s train, "
        f"estimators {sorted(estimator_records)} init sha={init_hash[:12]}…"
    )
    return manifest


# ---------------------------------------------------------------------------
# 6. stage: GPU smoke (short end-to-end plumbing check)
# ---------------------------------------------------------------------------

def run_smoke(
    *,
    device_name: str = "cuda:0",
    out_dir: Path = RESULTS_DIR,
    log: Any = print,
) -> dict[str, Any]:
    device = resolve_device(device_name)
    out_dir = Path(out_dir)
    smoke_dir = out_dir / "smoke"
    started = time.perf_counter()
    roster = load_roster(out_dir=out_dir)
    # (a) one full 1-epoch trajectory exercising capture/average/strict-load
    #     with a SMOKE estimator whose member is epoch 1
    traj = train_trajectory(
        0,
        device_name=device_name,
        out_dir=smoke_dir,
        estimator_override={"SMOKE": {"members": [1], "candidate": False,
                                       "effective_epochs": 1, "contiguous_last5": True,
                                       "window": "1"}},
        epochs_override=1,
        smoke=True,
        log=log,
    )
    smoke = {
        "protocol_version": PROTOCOL_VERSION,
        "stage": "smoke",
        "device": str(device),
        "roster_epochs": roster["epochs"],
        "traj_manifest": {
            "steps_done": traj["steps_done"], "stopped_reason": traj["stopped_reason"],
            "estimators": traj["estimators"], "schedule_prefix_ok": True,
            "init_matches_historical": True,
        },
        "checks": {
            "schedule_prefix_matches_historical": traj["schedule_prefix_matches_historical_240"],
            "init_reproduces_historical": traj["init_matches_historical"],
            "frozen_basis_unchanged": traj["frozen_basis_unchanged"],
            "members_captured": True,
            "aggregation_strict_load": all(
                v["strict_load_ok"] for v in traj["estimators"].values()),
            "save_reload_predictions_within_band": all(
                v["save_reload_prediction_within_band"] for v in traj["estimators"].values()),
            "save_reload_prediction_max_abs": max(
                v["save_reload_prediction_max_abs"] for v in traj["estimators"].values()),
            "dev_never_touched": True,
        },
        "all_passed": True,
        "seconds": float(time.perf_counter() - started),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(out_dir / "smoke.json", smoke)
    log(f"[smoke] ok on {device} in {smoke['seconds']:.1f}s")
    return smoke


# ---------------------------------------------------------------------------
# 7. one-shot terminal stage
# ---------------------------------------------------------------------------

def _load_estimator_model(
    seed: int, name: str, ctx: Mapping[str, Any], *, out_dir: Path,
) -> tuple[torch.nn.Module, dict[str, torch.Tensor]]:
    rdir = traj_dir(seed, out_dir)
    state = torch.load(rdir / f"{name}_state.pt", map_location="cpu", weights_only=False)
    model = repl.build_arm(ARM, ctx["payload"], ctx["kappa_M"], ctx["basis_parts"], int(seed))
    model.load_state_dict({k: v for k, v in state.items()}, strict=True)
    return model, state


def _score_estimator(
    seed: int, name: str, ctx: Mapping[str, Any], *,
    out_dir: Path, device: torch.device, q_fit: np.ndarray, q_dev: np.ndarray,
) -> dict[str, Any]:
    model, state = _load_estimator_model(seed, name, ctx, out_dir=out_dir)
    if state_hash(state) != read_json(traj_dir(seed, out_dir) / "manifest.json")["estimators"][name]["state_sha256"]:
        raise RuntimeError(f"estimator {name} s{seed} state hash mismatch vs manifest")
    model = model.to(device).eval()
    fitp = _predict_rows_checked(model, ctx["fit_data"], device)
    devp = _predict_rows_checked(model, ctx["dev_data"], device)
    y_raw_fit = fitp["ell_hat"] + fitp["s_hat"] + q_fit
    y_raw_dev = devp["ell_hat"] + devp["s_hat"] + q_dev
    b_y = float(np.median(ctx["y_fit"] - y_raw_fit))
    return {
        "model": model, "state": state,
        "fit": {"h": fitp["h"], "ell_hat": fitp["ell_hat"], "s_hat": fitp["s_hat"],
                "y_raw": y_raw_fit, "b_y": b_y},
        "dev": {"h": devp["h"], "ell_hat": devp["ell_hat"], "s_hat": devp["s_hat"],
                "y_raw": y_raw_dev, "b_y": b_y},
    }


def alpha_mean_intervention(
    seed: int,
    name: str,
    base: Mapping[str, Any],
    mean_alpha: np.ndarray,
    *,
    ctx: Mapping[str, Any],
    out_dir: Path,
    device: torch.device,
    log: Any = print,
) -> dict[str, Any]:
    """The unique dictionary intervention on one estimator (generalised
    loader: any estimator state, never the old soup directory).  alpha_v :=
    the fit-root mean alpha at the single local structure input via
    model.local_tuple.alpha_replace; c and every other input untouched; no
    retraining, no recalibration.  Dependence evidence, never benefit."""
    model, _state = _load_estimator_model(seed, name, ctx, out_dir=out_dir)
    model = model.to(device).eval()
    model.local_tuple.alpha_replace = torch.as_tensor(np.asarray(mean_alpha, np.float64), dtype=torch.float32)
    preds = _predict_rows_checked(model, ctx["dev_data"], device)
    model.local_tuple.alpha_replace = None

    basis = ctx["basis"]
    dev_idx = ctx["dev_idx"]
    rows, _checks = repl._phi_rows_for(dev_idx.tolist())
    z, phi_hat, _rel = zreuse._root_codes(basis, rows)
    alpha_dev = z[:, repl.COMMON_DIM:]
    mean = np.asarray(mean_alpha, np.float64)
    U = np.asarray(basis["U"], np.float64)
    Dbar = np.asarray(zreuse._frozen_code_parts(basis)[2].numpy(), np.float64)
    c = rows.astype(np.float64) @ U
    phi_hat_int = c @ U.T + mean[None, :] @ Dbar.T
    phi_change = np.abs(phi_hat_int - phi_hat)

    q_raw = np.asarray(base["q_raw"], np.float64)
    y_raw_i = preds["ell_hat"] + preds["s_hat"] + q_raw
    y = np.asarray(base["y"], np.float64)
    g = np.asarray(base["g"], np.float64)
    base_y_raw = np.asarray(base["y_raw"], np.float64)
    base_h = np.asarray(base["h"], np.float64)
    d_pred = np.abs(y_raw_i - base_y_raw)
    row = {
        "estimator": name, "seed": int(seed),
        "mechanism": (
            "alpha_v := the fit-root mean alpha at the single local structure input; "
            "c kept; chemistry one-hots, J incidence, Sem108, reader and the frozen Q "
            "untouched; re-decoded and re-forwarded; no retraining, no recalibration"
        ),
        "input_change": {
            "alpha_vs_mean_l2_median": float(np.median(np.linalg.norm(alpha_dev - mean[None, :], axis=1))),
            "alpha_vs_mean_l2_p95": float(np.percentile(np.linalg.norm(alpha_dev - mean[None, :], axis=1), 95)),
            "alpha_vs_mean_l2_max": float(np.linalg.norm(alpha_dev - mean[None, :], axis=1).max()),
            "phi_hat_abs_change_median": float(np.median(phi_change)),
            "phi_hat_abs_change_max": float(np.max(phi_change)),
        },
        "abs_dpred": {
            "mean": float(d_pred.mean()), "median": float(np.median(d_pred)),
            "p95": float(np.percentile(d_pred, 95)), "max": float(d_pred.max()),
        },
        "response_fraction_gt_1e_4": float((d_pred > REPLAY_TOL).mean()),
        "delta_y_raw_mae": float(np.mean(np.abs(y - y_raw_i)) - np.mean(np.abs(y - base_y_raw))),
        "delta_g_mae": float(np.mean(np.abs(g - preds["h"])) - np.mean(np.abs(g - base_h))),
        "reading": "response shows the consumer still reads the sparse code; never incremental benefit",
    }
    if log:
        log(
            f"[intervention {name} s{seed}] |dPred| p95 {row['abs_dpred']['p95']:.2e} "
            f"resp {row['response_fraction_gt_1e_4']:.3f} dMAE {row['delta_y_raw_mae']:+.2e}"
        )
    return row


def terminal_eval(
    *,
    out_dir: Path = RESULTS_DIR,
    device_name: str = "cuda:0",
    log: Any = print,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    if (out_dir / "terminal_eval.json").exists():
        raise RuntimeError("terminal_eval.json already exists: the terminal stage is one-shot")
    started = time.perf_counter()
    device = resolve_device(device_name)
    torch.set_num_threads(8 if device.type == "cpu" else 4)
    roster = load_roster(out_dir=out_dir)
    estimators: dict[str, dict[str, Any]] = roster["estimators"]
    control = roster["control"]
    candidates = [n for n, e in estimators.items() if e["candidate"]]
    ctx = load_round_context()
    objects = ctx["objects"]
    fit_idx, dev_idx = ctx["fit_idx"], ctx["dev_idx"]
    targets = objects["targets"]
    y = np.asarray(targets["y"], np.float64)
    g = np.asarray(targets["g"], np.float64)
    ell = np.asarray(targets["ell"], np.float64)
    s_t = np.asarray(targets["s"], np.float64)
    k_all = np.asarray(targets["k"], np.int64)
    gid_all = np.asarray(targets["gid"], np.int64)
    y_dev, g_dev, ell_dev, s_dev = y[dev_idx], g[dev_idx], ell[dev_idx], s_t[dev_idx]
    k_dev = k_all[dev_idx]
    _phi, _atom, node_sizes = prev._env_phi_atom()
    n_nodes_dev = np.asarray(node_sizes, np.int64)[dev_idx]
    g_p90 = float(np.percentile(np.abs(g[fit_idx]), 90))
    q_fit = nb._q_predictions(ctx["q_soup"], nb.topology_matrix(ctx["fit_data"]), device)
    q_dev = nb._q_predictions(ctx["q_soup"], nb.topology_matrix(ctx["dev_data"]), device)

    # (1) roster verification: both trajectories complete, members/hashes consistent
    traj_manifests: dict[int, dict[str, Any]] = {}
    checks = {
        "trajectories_completed": True, "schedule_prefix_ok": True,
        "init_hash_ok": True, "cross_seed_init_differ": True,
        "param_audit_ok": True, "frozen_basis_ok": True, "members_ok": True,
        "estimator_hash_ok": True, "q_soup_ok": True,
    }
    expected_audit = {
        "total_parameters": EXPECTED_PARAMETERS,
        "base_body_parameters": repl.EXPECTED_BODY_PARAMETERS,
        "bridge_parameters": repl.BRIDGE_PARAMETERS,
        "local_tuple_parameters": repl.LOCAL_PARAMETERS,
        "reader_output_parameters": repl.READER_OUTPUT_PARAMETERS,
    }
    for s in SEEDS:
        tman = read_json(traj_dir(s, out_dir) / "manifest.json")
        if tman["stopped_reason"] != "completed" or tman.get("smoke"):
            checks["trajectories_completed"] = False
        if int(tman["recipe"]["epochs"]) != int(roster["epochs"]):
            checks["trajectories_completed"] = False
        if tman["parameter_audit"] != expected_audit:
            checks["param_audit_ok"] = False
        if tman["frozen_basis_hashes"] != ctx["basis_parts"]["hashes"]:
            checks["frozen_basis_ok"] = False
        # the encoder-side hashes key the normalized Dbar (U/common_rms/Dbar),
        # not the package hashes (U/common_rms/D) — compare like for like
        expected_encoder_hashes = {
            "U": array_sha256(ctx["basis_parts"]["U"].numpy()),
            "common_rms": array_sha256(ctx["basis_parts"]["common_rms"].numpy()),
            "Dbar": array_sha256(ctx["basis_parts"]["Dbar"].numpy()),
        }
        if tman["encoder_frozen_basis_hashes"] != expected_encoder_hashes:
            checks["frozen_basis_ok"] = False
        if not tman["frozen_basis_unchanged"] or not tman["schedule_prefix_matches_historical_240"]:
            checks["frozen_basis_ok"] = checks["schedule_prefix_ok"] = False
        if not (tman["init_matches_historical"] and tman["init_differs_from_other_seed"]):
            checks["init_hash_ok"] = checks["cross_seed_init_differ"] = False
        if state_hash(ctx["q_soup"]) != tman["q_soup_sha256"]:
            checks["q_soup_ok"] = False
        for name, spec in estimators.items():
            rec = tman["estimators"].get(name)
            if rec is None or rec["members"] != [int(e) for e in spec["members"]]:
                checks["members_ok"] = False
            elif rec["state_sha256"] != state_hash(
                torch.load(traj_dir(s, out_dir) / f"{name}_state.pt", map_location="cpu", weights_only=False)
            ):
                checks["estimator_hash_ok"] = False
        traj_manifests[int(s)] = tman
    if not all(checks.values()):
        raise RuntimeError(f"roster verification failed: { {k: v for k, v in checks.items() if not v} }")

    # (2) scoring of every estimator x seed (fit + dev, full exports)
    results: dict[str, dict[int, dict[str, Any]]] = {name: {} for name in estimators}
    for s in SEEDS:
        for name in estimators:
            res = _score_estimator(s, name, ctx, out_dir=out_dir, device=device, q_fit=q_fit, q_dev=q_dev)
            results[name][int(s)] = res
            for split, rows_idx, preds, tgt in (
                ("fit", fit_idx, res["fit"], (ctx["y_fit"], ctx["g_fit"], ctx["ell_fit"], ctx["s_fit"], k_all[fit_idx])),
                ("dev", dev_idx, res["dev"], (y_dev, g_dev, ell_dev, s_dev, k_dev)),
            ):
                np.savez_compressed(
                    traj_dir(s, out_dir) / f"{name}_{split}_predictions.npz",
                    row_index=np.asarray(rows_idx, np.int64), gid=gid_all[rows_idx],
                    y=tgt[0], g=tgt[1], ell=tgt[2], s=tgt[3], k=tgt[4],
                    h=preds["h"], ell_hat=preds["ell_hat"], s_hat=preds["s_hat"],
                    q_raw=q_fit if split == "fit" else q_dev, y_raw=preds["y_raw"],
                    b_y=np.float64(preds["b_y"]),
                )
            mae_dev = {
                "y_raw": float(np.mean(np.abs(y_dev - res["dev"]["y_raw"]))),
                "y_cal": float(np.mean(np.abs(y_dev - (res["dev"]["y_raw"] + res["dev"]["b_y"])))),
                "g_raw": float(np.mean(np.abs(g_dev - res["dev"]["h"]))),
                "ell": float(np.mean(np.abs(ell_dev - res["dev"]["ell_hat"]))),
                "s": float(np.mean(np.abs(s_dev - res["dev"]["s_hat"]))),
            }
            mae_fit = {
                "y_raw": float(np.mean(np.abs(ctx["y_fit"] - res["fit"]["y_raw"]))),
                "g_raw": float(np.mean(np.abs(ctx["g_fit"] - res["fit"]["h"]))),
                "ell": float(np.mean(np.abs(ctx["ell_fit"] - res["fit"]["ell_hat"]))),
                "s": float(np.mean(np.abs(ctx["s_fit"] - res["fit"]["s_hat"]))),
            }
            e_y = res["dev"]["y_raw"] - y_dev
            e_g = res["dev"]["h"] - g_dev
            res["mae_dev"], res["mae_fit"] = mae_dev, mae_fit
            res["component_stats_dev"] = _component_stats(
                res["dev"]["ell_hat"], res["dev"]["s_hat"], res["dev"]["h"], ell_dev, s_dev, g_dev)
            res["k_groups_dev"] = _k_group_table(e_y, e_g, k_dev)
            res["k0_subgroups_dev"] = _k0_subgroup_table(
                e_y, e_g, k_dev, n_nodes_dev, np.abs(g_dev), g_p90)
            log(
                f"[eval {name} s{s}] dev y_raw {mae_dev['y_raw']:.5f} g {mae_dev['g_raw']:.5f} | "
                f"fit y_raw {mae_fit['y_raw']:.5f} g {mae_fit['g_raw']:.5f}"
            )

    # (3) paired deltas vs CTRL240 + the historical reproduction gap
    old_terminal = read_json(SOURCE_DIR / "terminal_eval.json")
    deltas: dict[str, dict[str, Any]] = {}
    for c in candidates:
        dy = {s: results[c][s]["mae_dev"]["y_raw"] - results[control][s]["mae_dev"]["y_raw"] for s in SEEDS}
        dg = {s: results[c][s]["mae_dev"]["g_raw"] - results[control][s]["mae_dev"]["g_raw"] for s in SEEDS}
        dy_fit = {s: results[c][s]["mae_fit"]["y_raw"] - results[control][s]["mae_fit"]["y_raw"] for s in SEEDS}
        deltas[c] = {
            "definition": "Delta_s = MAE(candidate_s) - MAE(CTRL240_s), full dev (negative = improvement)",
            "delta_y_dev": {str(s): dy[s] for s in SEEDS},
            "delta_g_dev": {str(s): dg[s] for s in SEEDS},
            "delta_y_fit": {str(s): dy_fit[s] for s in SEEDS},
            "mean_delta_y_dev": float(np.mean([dy[s] for s in SEEDS])),
            "mean_delta_g_dev": float(np.mean([dg[s] for s in SEEDS])),
            "k0_delta_y_contribution": {
                str(s): float(
                    (np.abs(results[c][s]["dev"]["y_raw"] - y_dev)[k_dev == 0].sum()
                     - np.abs(results[control][s]["dev"]["y_raw"] - y_dev)[k_dev == 0].sum())
                    / len(y_dev))
                for s in SEEDS
            },
        }
    reproduction = {
        str(s): {
            "ctrl240_dev_y_raw": results[control][s]["mae_dev"]["y_raw"],
            "ctrl240_dev_g_raw": results[control][s]["mae_dev"]["g_raw"],
            "historical_soup_dev_y_raw": float(old_terminal["main_table"][f"DICT_s{s}"]["mae"]["y_raw"]),
            "historical_soup_dev_g_raw": float(old_terminal["main_table"][f"DICT_s{s}"]["mae"]["g_raw"]),
            "gap_y_raw": float(results[control][s]["mae_dev"]["y_raw"]
                               - old_terminal["main_table"][f"DICT_s{s}"]["mae"]["y_raw"]),
            "gap_g_raw": float(results[control][s]["mae_dev"]["g_raw"]
                               - old_terminal["main_table"][f"DICT_s{s}"]["mae"]["g_raw"]),
        }
        for s in SEEDS
    }

    # (4) group-paired bootstrap (shared picks, avg seeds first)
    smiles = zreuse._canonical_smiles()
    errs = {
        f"{name}_s{s}": np.abs(y_dev - results[name][s]["dev"]["y_raw"])
        for name in estimators for s in SEEDS
    }
    bootstrap = candidate_paired_bootstrap(errs, smiles[dev_idx], control, candidates)

    # (5) FP32 noise bound eta: repeated evals + identity hooks
    eta = 0.0
    noise: dict[str, Any] = {}
    for s in SEEDS:
        for name in estimators:
            repeat = _score_estimator(s, name, ctx, out_dir=out_dir, device=device, q_fit=q_fit, q_dev=q_dev)
            d = float(np.max(np.abs(np.asarray(repeat["dev"]["y_raw"], np.float64)
                          - np.asarray(results[name][s]["dev"]["y_raw"], np.float64))))
            noise[f"{name}_s{s}"] = {"repeat_max_abs_dpred": d}
            eta = max(eta, d)
            del repeat
        for name in estimators:
            model_hook, state = _load_estimator_model(s, name, ctx, out_dir=out_dir)
            model_hook.decode_enabled = False
            model_hook = model_hook.to(device).eval()
            model_parent = repl.build_arm("RAW", ctx["payload"], ctx["kappa_M"], ctx["basis_parts"], int(s))
            model_parent.load_state_dict({kk: vv for kk, vv in state.items()}, strict=True)
            model_parent = model_parent.to(device).eval()
            h_hook = _predict_rows_checked(model_hook, ctx["dev_data"], device)["h"]
            h_parent = _predict_rows_checked(model_parent, ctx["dev_data"], device)["h"]
            d = float(np.max(np.abs(np.asarray(h_hook) - np.asarray(h_parent))))
            noise[f"{name}_s{s}"]["identity_hook_max_abs_dpred"] = d
            eta = max(eta, d)
            del model_hook, model_parent
            if device.type == "cuda":
                torch.cuda.empty_cache()
    marker = float(max(REPLAY_TOL, 10.0 * eta))

    # (6) the unique alpha-mean intervention on every candidate
    mean_codes = repl.fit_mean_codes(ctx["basis"], fit_idx)
    interventions: dict[str, dict[int, dict[str, Any]]] = {}
    responsive: dict[str, bool] = {}
    for c in candidates:
        interventions[c] = {}
        for s in SEEDS:
            base = {
                "y": y_dev, "g": g_dev, "y_raw": results[c][s]["dev"]["y_raw"],
                "h": results[c][s]["dev"]["h"], "q_raw": q_dev,
                "b_y": results[c][s]["dev"]["b_y"],
            }
            row = alpha_mean_intervention(
                s, c, base, mean_codes["mean_alpha"], ctx=ctx, out_dir=out_dir,
                device=device, log=log,
            )
            row["response_beyond_noise"] = bool(row["abs_dpred"]["p95"] > marker)
            interventions[c][int(s)] = row
        responsive[c] = bool(all(interventions[c][s]["response_beyond_noise"] for s in SEEDS))

    # (7) the frozen retention gate + tie-break
    delta_y = {c: {s: deltas[c]["delta_y_dev"][str(s)] for s in SEEDS} for c in candidates}
    delta_g = {c: {s: deltas[c]["delta_g_dev"][str(s)] for s in SEEDS} for c in candidates}
    mean_y_raw = {c: float(np.mean([results[c][s]["mae_dev"]["y_raw"] for s in SEEDS])) for c in candidates}
    decision = retention_gate(candidates, delta_y, delta_g, mean_y_raw, responsive, all(checks.values()))

    # (8) cost summary (actual shared-trajectory cost)
    cost = {
        "per_trajectory": {
            str(s): {
                "curve_seconds_total": traj_manifests[s]["curve_seconds_total"],
                "estimator_construction_seconds": traj_manifests[s]["estimator_construction_seconds"],
                "wall_clock_s": traj_manifests[s]["wall_clock_s"],
                "device": traj_manifests[s]["device"],
            }
            for s in SEEDS
        },
        "trajectory_gpu_seconds_total": float(sum(
            traj_manifests[s]["wall_clock_s"] for s in SEEDS)),
        "note": (
            "two trajectories only; every estimator shares its seed's trajectory "
            "(not independent training repeats); no virtual saving claimed"
        ),
    }

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "stage": "terminal-eval",
        "one_shot": True,
        "route": roster["route"],
        "roster_checks": checks,
        "main_table": {
            f"{name}_s{s}": {
                "mae_dev": results[name][s]["mae_dev"],
                "mae_fit": results[name][s]["mae_fit"],
                "b_y": results[name][s]["dev"]["b_y"],
                "component_stats_dev": results[name][s]["component_stats_dev"],
                "k_groups_dev": results[name][s]["k_groups_dev"],
                "k0_subgroups_dev": results[name][s]["k0_subgroups_dev"],
                "curve_seconds_total": traj_manifests[s]["curve_seconds_total"],
                "estimator_state_sha256": traj_manifests[s]["estimators"][name]["state_sha256"],
            }
            for name in estimators for s in SEEDS
        },
        "paired_deltas_vs_ctrl240": deltas,
        "historical_reproduction": reproduction,
        "bootstrap": bootstrap,
        "noise_bound": {"eta": eta, "per_run": noise, "marker": marker},
        "interventions": {c: {str(s): interventions[c][s] for s in SEEDS} for c in candidates},
        "decision": decision,
        "cost": cost,
        "g_abs_p90_fit": g_p90,
        "scope": (
            "development comparison on the historical dev rows (union of the old "
            "select/confirm), ALSO used by Stage A to select the training branch: "
            "never an independent confirm, never official valid/test; the gate is "
            "this round's resource gate, not significance or SOTA"
        ),
        "seconds": float(time.perf_counter() - started),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(out_dir / "terminal_eval.json", payload)
    log(
        f"[terminal-eval] passed={decision['passed_candidates']} winner={decision['winner']} "
        f"in {payload['seconds']:.0f}s"
    )
    return payload


# ---------------------------------------------------------------------------
# 7b. read-only addendum: corrected identity-hook noise bound
# ---------------------------------------------------------------------------

def noise_bound_addendum(
    *,
    out_dir: Path = RESULTS_DIR,
    device_name: str = "cuda:0",
    log: Any = print,
) -> dict[str, Any]:
    """Corrected identity-hook noise bound (never overwrites the one-shot).

    The terminal stage's identity hook set ``decode_enabled`` on the model
    wrapper instead of the ENCODER, so the hook model kept decoding and the
    recorded eta/marker were inflated by the decode effect (~0.5) instead of
    the hook noise (~1e-6).  This read-only addendum recomputes the
    identity-hook noise correctly (decode disabled through the factory
    argument, exactly like the source round's terminal stage) on the SAME
    estimator states, rebuilds the marker, re-derives the alpha-responsive
    booleans from the recorded intervention p95 values, and re-applies the
    frozen retention gate with the corrected inputs.  terminal_eval.json is
    never modified; the gate outcome is recomputed mechanically.
    """
    out_dir = Path(out_dir)
    started = time.perf_counter()
    terminal = read_json(out_dir / "terminal_eval.json")
    roster = load_roster(out_dir=out_dir)
    estimators = roster["estimators"]
    control = roster["control"]
    candidates = [n for n, e in estimators.items() if e["candidate"]]
    device = resolve_device(device_name)
    torch.set_num_threads(8 if device.type == "cpu" else 4)
    ctx = load_round_context()
    dev_data = ctx["dev_data"]

    identity: dict[str, float] = {}
    eta_corrected = 0.0
    for s in SEEDS:
        for name in estimators:
            state = torch.load(traj_dir(s, out_dir) / f"{name}_state.pt", map_location="cpu", weights_only=False)
            model_hook = repl.build_arm(ARM, ctx["payload"], ctx["kappa_M"], ctx["basis_parts"], int(s), decode=False)
            model_hook.load_state_dict({k: v for k, v in state.items()}, strict=True)
            model_hook = model_hook.to(device).eval()
            model_parent = repl.build_arm("RAW", ctx["payload"], ctx["kappa_M"], ctx["basis_parts"], int(s))
            model_parent.load_state_dict({k: v for k, v in state.items()}, strict=True)
            model_parent = model_parent.to(device).eval()
            h_hook = _predict_rows_checked(model_hook, dev_data, device)["h"]
            h_parent = _predict_rows_checked(model_parent, dev_data, device)["h"]
            d = float(np.max(np.abs(np.asarray(h_hook) - np.asarray(h_parent))))
            identity[f"{name}_s{s}"] = d
            eta_corrected = max(eta_corrected, d)
            del model_hook, model_parent
            if device.type == "cuda":
                torch.cuda.empty_cache()
    # the repeat evaluations in the terminal stage were correct; keep them
    for key, rec in terminal["noise_bound"]["per_run"].items():
        eta_corrected = max(eta_corrected, float(rec["repeat_max_abs_dpred"]))
    marker_corrected = float(max(REPLAY_TOL, 10.0 * eta_corrected))

    responsive: dict[str, bool] = {}
    for c in candidates:
        responsive[c] = bool(all(
            float(terminal["interventions"][c][str(s)]["abs_dpred"]["p95"]) > marker_corrected
            for s in SEEDS
        ))

    decision_orig = terminal["decision"]
    delta_y = {c: {int(s): float(v) for s, v in decision_orig["per_candidate"][c]["delta_y"].items()} for c in candidates}
    delta_g = {c: {int(s): float(v) for s, v in decision_orig["per_candidate"][c]["delta_g"].items()} for c in candidates}
    mean_y_raw = {
        c: float(np.mean([terminal["main_table"][f"{c}_s{s}"]["mae_dev"]["y_raw"] for s in SEEDS]))
        for c in candidates
    }
    decision_corrected = retention_gate(
        candidates, delta_y, delta_g, mean_y_raw, responsive, all(terminal["roster_checks"].values()),
    )

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "stage": "noise-addendum",
        "read_only": True,
        "terminal_eval_never_modified": True,
        "bug": (
            "the terminal identity hook set decode_enabled on the model wrapper "
            "instead of the encoder, so the hook model kept decoding; the recorded "
            "eta/marker were inflated by the decode effect (~0.5) instead of the "
            "hook noise (~1e-6) and the alpha-responsive gate condition was wrongly "
            "recorded as False despite p95 2.38/1.76 and response fraction 1.000"
        ),
        "identity_hook_max_abs_dpred_corrected": identity,
        "repeat_max_abs_dpred_from_terminal": {
            k: float(v["repeat_max_abs_dpred"]) for k, v in terminal["noise_bound"]["per_run"].items()
        },
        "eta_corrected": eta_corrected,
        "marker_corrected": marker_corrected,
        "alpha_responsive_corrected": responsive,
        "decision_recomputed": decision_corrected,
        "gate_outcome_unchanged_by_bug": bool(
            decision_corrected["passed_candidates"] == decision_orig["passed_candidates"]
            and decision_corrected["winner"] == decision_orig["winner"]
        ),
        "seconds": float(time.perf_counter() - started),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(out_dir / "noise_bound_addendum.json", payload)
    log(
        f"[noise-addendum] eta {terminal['noise_bound']['eta']:.3g} -> {eta_corrected:.3g}; "
        f"marker {marker_corrected:.3g}; gate outcome unchanged: "
        f"{payload['gate_outcome_unchanged_by_bug']}"
    )
    return payload


# ---------------------------------------------------------------------------
# 8. CLI (local use; the registered runner calls the same functions)
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", default="source-checks", choices=(
        "source-checks", "checkpoint-diagnostics", "freeze-roster", "smoke",
        "train-trajectory", "terminal-eval", "noise-addendum"))
    parser.add_argument("--seed", type=int, default=0, choices=SEEDS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", default=str(RESULTS_DIR))
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    if args.stage == "source-checks":
        source_checks(out_dir=out_dir)
    elif args.stage == "checkpoint-diagnostics":
        checkpoint_diagnostics(args.seed, device_name=args.device, out_dir=out_dir)
    elif args.stage == "freeze-roster":
        freeze_roster(out_dir=out_dir)
    elif args.stage == "smoke":
        run_smoke(device_name=args.device, out_dir=out_dir)
    elif args.stage == "train-trajectory":
        train_trajectory(args.seed, device_name=args.device, out_dir=out_dir)
    elif args.stage == "noise-addendum":
        noise_bound_addendum(device_name=args.device, out_dir=out_dir)
    else:
        terminal_eval(device_name=args.device, out_dir=out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
