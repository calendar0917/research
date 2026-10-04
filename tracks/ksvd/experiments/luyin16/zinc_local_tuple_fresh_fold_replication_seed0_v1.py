"""ZINC local-tuple fresh-fold replication (seed 0): B_new vs M_J_new.

Round: ``zinc_local_tuple_fresh_fold_replication_seed0_v1``.

Question: does the already-observed M_J-vs-B chemical-component improvement
survive on a *pre-fixed*, non-overlapping-dev 8000/2000 fold of the same 10000
official-train rows?  Exactly two formal trajectories (seed 0):

* ``B``   — the original compressed M_g skeleton (Sem110, posterior MLP bridge,
  static relations, topology/reader), 267,611 parameters, no local-tuple path.
* ``M_J`` — the same skeleton plus the original M_J local tuple encoder
  (``f_M(x) = SiLU(x @ A_bar.T)``, 64x125 A_raw) and the original fusion
  first-layer injection through zero-init ``W_loc`` 342x64, 297,499 parameters.

The new fold is fixed exactly as specified (old fold seed 20261004, then
``np.random.default_rng(20261005).permutation(old_fit)[:2000]`` as new dev and
``setdiff1d(arange(10000), new_dev)`` as new fit).  Every fit-dependent object
is rebuilt on new fit only: the label constants (OLS cycle-free bulk + Nelder-
Mead snap), the body input standardizers, the 125-D tuple phi scaler and the
kappa sample.  Structural tuple arrays (root_base / pair_ptr / pair_t / pair_a
/ pair_wJ / pair_wI / root_atom, all 10000 raw graph indices) are reused from
the committed tuple index byte-for-byte; the old fit-only *scalers* are never
reused.

Official valid/test are never loaded, instantiated, predicted or scored.
The 2000 new-dev rows were trained on by the old models and the two new fits
overlap; this is a partition-stability replication, not an independent test.

Usage::

    uv run python -m tracks.ksvd.experiments.luyin16.\
zinc_local_tuple_fresh_fold_replication_seed0_v1 --phase-a
    ... --smoke --arm M --device cpu
    ... --train --arm B --device cuda --out <dir>
    ... --train --arm M --device cuda --out <dir>
    ... --analyze
    ... --budget
    ... --manifest
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm
from tracks.ksvd.experiments.luyin16 import zinc_chemistry_dictionary_vs_mlp_seed0_v1 as zcdm
from tracks.ksvd.experiments.luyin16 import zinc_full_cycle_target_decomposition_v1 as zftd
from tracks.ksvd.experiments.luyin16 import zinc_joint_dictionary_decision_v1 as zjd
from tracks.ksvd.experiments.luyin16 import (
    zinc_local_tuple_dictionary_joint_vs_independent_seed0_v1 as prev,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_local_tuple_dictionary_vs_mlp_seed0_v1 as mlpmod,
)
from tracks.ksvd.experiments.luyin16 import zinc_task_dictionary_and_cycle_witness_seed0_v1 as zw

PROTOCOL_VERSION = "zinc-local-tuple-fresh-fold-replication-seed0-v1"
RESULT_SLUG = "zinc_local_tuple_fresh_fold_replication_seed0_v1"
RESULTS_DIR = zjd.TRACK_ROOT / "results" / RESULT_SLUG

#: read-only historical sources
OLD_TUPLE_NPZ = prev.TUPLE_NPZ
OLD_TARGETS_NPZ = zcdm.TARGETS_NPZ
TRAIN_LABEL_CSV = zjd.CYCLE / "train_cycle_audit_label.csv"
GVEA_PROPS_NPZ = zjd.CYCLE / "cache" / "gvae_full_properties.npz"
HANDOFF_TRAIN = zftd.HANDOFF_TRAIN
PREP_BLOB = zjd.PREP_DIR / "fold_objects.npz"

# ---- frozen recipe (identical for both arms) ------------------------------
SEED = 0
EPOCHS = 240
LR = 1.0e-3
WEIGHT_DECAY = 1.0e-5
GRAD_CLIP = 5.0
BATCH_SIZE = 128
SOUP_EPOCHS = (236, 237, 238, 239, 240)
TRAIN_SHUFFLE_OFFSET = 101
LOG_EPOCHS = (1, 40, 120, 240)
N_FIT = 8000
N_DEV = 2000

#: fixed label-generation constant (never re-estimated)
MU_LOGP = 2.4570953396190123

#: fold definition / expected hashes
OLD_FOLD_SEED = 20261004
NEW_DEV_SEED = 20261005
OLD_FIT_SHA = "7bf1cfb856a79617baf83c4ba95290d08248b4633609fa24725a5c028c9096d9"
OLD_DEV_SHA = "a61c801073fbda237a46e4f8fe6c8e4f8f24e867e7a8b452f7aaffc86a9cceff"
NEW_FIT_SHA = "2a21cb8771f6e24cfb4a5cc50602cf0b4390db9c4367f8bb910052f9f0aebbcb"
NEW_DEV_SHA = "270ab4126b0f0413f1f6ff7ada2e9914bbaca91ae50cc65bbd8ca2673d8f9357"
FROZEN_SCHEDULE_SHA = "7b11a529584a8bdaa8e2f17a8c8413cdc6677714ae855339b5984e9fe6938e65"

#: statistics / gates (frozen before the two formal runs)
BOOT_SEED = 20261005
N_BOOT = 1000
PERF_DELTA = 0.003
KAPPA_SEED = 20261004
KAPPA_SAMPLE_MAX = 8192
REPLAY_TOL = 1.0e-5
GROUP_NAMES = ("k=0", "k=-1", "k=-2", "k<=-3")

jsonable = prev.jsonable
write_json = prev.write_json
file_sha256 = prev.file_sha256
state_hash = prev.state_hash
seed_everything = prev.seed_everything
array_sha256 = prev._array_sha256
group_masks = zw.group_masks


def resolve_device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _hash_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# 1. the fresh fold (fixed once, never selected)
# ---------------------------------------------------------------------------


def build_fresh_fold() -> dict[str, Any]:
    """Old fold (20261004) then new dev (20261005) inside old fit; exact spec."""
    old_perm = np.random.default_rng(OLD_FOLD_SEED).permutation(10000)
    old_fit = np.sort(old_perm[:8000]).astype(np.int64)
    old_dev = np.sort(old_perm[8000:]).astype(np.int64)
    new_dev = np.sort(
        np.random.default_rng(NEW_DEV_SEED).permutation(old_fit)[:2000]
    ).astype(np.int64)
    new_fit = np.setdiff1d(np.arange(10000, dtype=np.int64), new_dev)

    checks = {
        "old_fit_sha256": _hash_bytes(old_fit.tobytes()),
        "old_dev_sha256": _hash_bytes(old_dev.tobytes()),
        "new_fit_sha256": _hash_bytes(new_fit.tobytes()),
        "new_dev_sha256": _hash_bytes(new_dev.tobytes()),
        "sizes_ok": bool(old_fit.size == N_FIT and old_dev.size == N_DEV and new_fit.size == N_FIT and new_dev.size == N_DEV),
        "new_fit_dev_disjoint": bool(np.intersect1d(new_fit, new_dev).size == 0),
        "new_fit_dev_cover": bool(np.union1d(new_fit, new_dev).size == 10000),
        "new_dev_subset_old_fit": bool(np.isin(new_dev, old_fit).all()),
        "new_dev_not_in_old_dev": bool(np.intersect1d(new_dev, old_dev).size == 0),
        "fit_intersection_size": int(np.intersect1d(new_fit, old_fit).size),
        "old_fit_sha256_matches_frozen": _hash_bytes(old_fit.tobytes()) == OLD_FIT_SHA,
        "old_dev_sha256_matches_frozen": _hash_bytes(old_dev.tobytes()) == OLD_DEV_SHA,
        "new_fit_sha256_matches_frozen": _hash_bytes(new_fit.tobytes()) == NEW_FIT_SHA,
        "new_dev_sha256_matches_frozen": _hash_bytes(new_dev.tobytes()) == NEW_DEV_SHA,
    }
    if not all(
        checks[key]
        for key in (
            "sizes_ok",
            "new_fit_dev_disjoint",
            "new_fit_dev_cover",
            "new_dev_subset_old_fit",
            "new_dev_not_in_old_dev",
            "old_fit_sha256_matches_frozen",
            "old_dev_sha256_matches_frozen",
            "new_fit_sha256_matches_frozen",
            "new_dev_sha256_matches_frozen",
        )
    ):
        raise RuntimeError(f"fresh fold construction check failed: {checks}")
    if checks["fit_intersection_size"] != 6000:
        raise RuntimeError(f"expected |new_fit ∩ old_fit| = 6000, got {checks['fit_intersection_size']}")
    return {
        "old_fit": old_fit,
        "old_dev": old_dev,
        "fit_idx": new_fit,
        "dev_idx": new_dev,
        "definition": (
            "old_perm=rng(20261004).permutation(10000); old_fit=sort(old_perm[:8000]); "
            "old_dev=sort(old_perm[8000:]); new_dev=sort(rng(20261005).permutation(old_fit)[:2000]); "
            "new_fit=setdiff1d(arange(10000), new_dev)"
        ),
        "checks": checks,
    }


def canonical_group_cross_counts(fold: Mapping[str, Any]) -> dict[str, Any]:
    handoff = np.load(HANDOFF_TRAIN, allow_pickle=True)
    gid = np.asarray(handoff["canonical_group_id"], np.int64)
    fit_groups = set(gid[fold["fit_idx"]].tolist())
    dev_groups = set(gid[fold["dev_idx"]].tolist())
    shared = fit_groups & dev_groups
    return {
        "canonical_group_ids_available": True,
        "n_groups_total": int(np.unique(gid).size),
        "n_shared_fit_dev_groups": int(len(shared)),
        "n_fit_rows_in_shared_groups": int(np.isin(gid[fold["fit_idx"]], list(shared)).sum()),
        "n_dev_rows_in_shared_groups": int(np.isin(gid[fold["dev_idx"]], list(shared)).sum()),
    }


# ---------------------------------------------------------------------------
# 2. train-only raw label fields + new-fit-only target constants
# ---------------------------------------------------------------------------


def load_train_only_raw_rows() -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Per-molecule logP/SA/y_stored/cycle_score_gvae for the 10000 train rows.

    Sources (all train-only):
    * ``train_cycle_audit_label.csv``: positional train rows + selected GVAE
      smiles line + stored target.
    * ``gvae_full_properties.npz``: audit-computed raw GVAE logP/SA/cycle at
      that line (full ZINC pool; only train lines are read).
    * ``train.npz`` handoff: canonical y and canonical group id.

    The mixed ``formula_verification_per_molecule.csv`` (which also contains
    valid/test rows) is deliberately never opened.
    """
    import pandas as pd

    lab = pd.read_csv(TRAIN_LABEL_CSV)
    if len(lab) != 10000:
        raise RuntimeError(f"train label csv has {len(lab)} rows, expected 10000")
    if not np.array_equal(lab["subset_index"].to_numpy(np.int64), np.arange(10000)):
        raise RuntimeError("train label csv rows are not positional")
    if not (lab["split"].astype(str) == "train").all():
        raise RuntimeError("train label csv contains non-train rows")
    if not (lab["molecule_id"].astype(str) == [f"train:{i:04d}" for i in range(10000)]).all():
        raise RuntimeError("train label csv molecule_id mismatch")
    if not bool(lab["matched"].all()):
        raise RuntimeError("some train rows are unmatched in the audit")
    lines = lab["smi_line"].to_numpy(np.int64)
    if int((lines < 0).sum()):
        raise RuntimeError("some train rows have no smiles line")

    props = np.load(GVEA_PROPS_NPZ)
    logp = np.asarray(props["logP"], np.float64)[lines]
    sa = np.asarray(props["SA"], np.float64)[lines]
    cyc = np.asarray(props["cycle"], np.float64)[lines]
    # the audit's per-line cycle value must equal the order-verified column
    cycle_order = lab["cycle_score_gvae_order"].to_numpy(np.float64)
    if not np.array_equal(cyc, cycle_order):
        raise RuntimeError("gvae props cycle != audit order-verified cycle at selected line")
    if not (np.isfinite(logp).all() and np.isfinite(sa).all() and np.isfinite(cyc).all()):
        raise RuntimeError("non-finite raw label fields")

    handoff = np.load(HANDOFF_TRAIN, allow_pickle=True)
    ids = np.asarray(handoff["ids"]).astype(str)
    y = np.asarray(handoff["y"], np.float64)
    gid = np.asarray(handoff["canonical_group_id"], np.int64)
    if y.shape[0] != 10000 or gid.shape[0] != 10000:
        raise RuntimeError("handoff shape mismatch")
    if list(ids) != [str(i) for i in range(10000)]:
        raise RuntimeError("handoff ids are not positional")
    target = lab["target"].to_numpy(np.float64)
    y_max_abs_diff = float(np.max(np.abs(target - y)))
    if y_max_abs_diff > 1e-12:
        raise RuntimeError(f"label target vs handoff y max abs diff {y_max_abs_diff}")

    rows = np.empty(
        10000,
        dtype=[
            ("logP", np.float64),
            ("SA", np.float64),
            ("y_stored", np.float64),
            ("cycle_score_gvae", np.float64),
        ],
    )
    rows["logP"] = logp
    rows["SA"] = sa
    rows["y_stored"] = y
    rows["cycle_score_gvae"] = cyc

    checks = {
        "n_rows": 10000,
        "split_train_only": True,
        "y_is_train_only_handoff": True,
        "y_target_handoff_max_abs_diff": y_max_abs_diff,
        "cycle_matches_audit_order_column": True,
        "all_finite": True,
        "sources": {
            "train_label_csv": str(TRAIN_LABEL_CSV),
            "train_label_csv_sha256": file_sha256(TRAIN_LABEL_CSV),
            "gvae_full_properties_npz": str(GVEA_PROPS_NPZ),
            "gvae_full_properties_npz_sha256": file_sha256(GVEA_PROPS_NPZ),
            "handoff_train": str(HANDOFF_TRAIN),
            "handoff_train_sha256": file_sha256(HANDOFF_TRAIN),
        },
    }
    return rows, gid, checks


def refit_new_targets(
    rows: np.ndarray,
    fit_idx: np.ndarray,
    *,
    old_anchor: bool = True,
) -> dict[str, Any]:
    """Fit sigma/mu on new fit only, then apply to all 10000 train rows."""
    in_fit = np.zeros(10000, dtype=bool)
    in_fit[np.asarray(fit_idx, np.int64)] = True
    constants = zcdm._refit_constants(rows, in_fit, mu_logp=MU_LOGP)
    labels = zcdm._apply_constants(rows, constants, MU_LOGP)
    y = rows["y_stored"]
    c = labels["c"]
    g = y - c
    k = labels["k"]
    alpha = rows["logP"]
    beta = rows["SA"]
    gamma = rows["cycle_score_gvae"]
    cycle_free = gamma == 0.0
    checks: dict[str, Any] = {
        "fit_rows_only": True,
        "uses_dev_labels_in_fit": False,
        "n_fit": int(in_fit.sum()),
        "n_fit_cycle_free": int((cycle_free & in_fit).sum()),
        "constants_finite": bool(all(math.isfinite(float(constants[key])) for key in constants if isinstance(constants[key], float))),
        "sigma_cycle_nonzero": bool(float(constants["sigma_cycle"]) != 0.0),
        "y_identity_max_abs": float(np.max(np.abs(g - (y - c)))),
        "mu_logP_fixed": float(MU_LOGP),
        "k_fit_counts": {name: int(mask.sum()) for name, mask in group_masks(k[in_fit]).items()},
        "k_dev_counts": {name: int(mask.sum()) for name, mask in group_masks(k[~in_fit]).items()},
        "c_g0_value": float(c[k == 0][0]) if int((k == 0).sum()) else None,
        "c_g0_spread_max_abs": float(np.max(np.abs(c[k == 0] - np.median(c[k == 0])))) if int((k == 0).sum()) else None,
    }
    if abs(checks["y_identity_max_abs"]) > 1e-12:
        raise RuntimeError("y = g + c identity failed")
    if not checks["constants_finite"] or not checks["sigma_cycle_nonzero"]:
        raise RuntimeError("invalid refit constants")

    if old_anchor:
        # lineage check only: the reconstructed raw fields must reproduce the
        # historical ALL-train constants when fitted on the OLD fit mask; the
        # old targets are never used as supervision for this round.
        with np.load(OLD_TARGETS_NPZ, allow_pickle=False) as z:
            old_names = [str(s) for s in z["constant_names"].tolist()]
            old_values = {name: float(v) for name, v in zip(old_names, z["constants"].tolist())}
            old_k = np.asarray(z["k"], np.int64)
            old_c = np.asarray(z["c"], np.float64)
        old_fit = np.sort(np.random.default_rng(OLD_FOLD_SEED).permutation(10000)[:8000]).astype(np.int64)
        old_in_fit = np.zeros(10000, dtype=bool)
        old_in_fit[old_fit] = True
        old_refit = zcdm._refit_constants(rows, old_in_fit, mu_logp=MU_LOGP)
        old_labels = zcdm._apply_constants(rows, old_refit, MU_LOGP)
        checks["historical_anchor"] = {
            "old_targets_npz": str(OLD_TARGETS_NPZ),
            "old_targets_npz_sha256": file_sha256(OLD_TARGETS_NPZ),
            "refit_vs_saved_constant_max_abs": {
                key: float(abs(old_refit[key] - old_values[key]))
                for key in old_names
            },
            "k_diff_rows": int(np.sum(old_labels["k"] != old_k)),
            "c_max_abs_diff": float(np.max(np.abs(old_labels["c"] - old_c))),
            "note": "raw-field lineage check only; old g/c never used as supervision",
        }

    return {
        "constants": constants,
        "arrays": {"y": y, "c": c, "g": g, "k": k},
        "checks": checks,
    }


def dev_label_shuffle_independence(rows: np.ndarray, fit_idx: np.ndarray, dev_idx: np.ndarray) -> dict[str, Any]:
    """Permute new-dev raw label fields only; the fit constants must not move."""
    in_fit = np.zeros(10000, dtype=bool)
    in_fit[np.asarray(fit_idx, np.int64)] = True
    perm = np.random.default_rng(NEW_DEV_SEED + 1).permutation(int(dev_idx.size))
    shuffled = rows.copy()
    for field in ("logP", "SA", "y_stored", "cycle_score_gvae"):
        values = shuffled[field][dev_idx].copy()
        shuffled[field][dev_idx] = values[perm]
    base = zcdm._refit_constants(rows, in_fit, mu_logp=MU_LOGP)
    other = zcdm._refit_constants(shuffled, in_fit, mu_logp=MU_LOGP)
    diffs = {key: float(other[key]) - float(base[key]) for key in base}
    return {
        "dev_rows_permuted": int(dev_idx.size),
        "constant_diffs": diffs,
        "identical": bool(all(value == 0.0 for value in diffs.values())),
    }


# ---------------------------------------------------------------------------
# 3. fresh tuple payload (structure reused, scalers refit on new fit)
# ---------------------------------------------------------------------------


def build_fresh_payload(fold: Mapping[str, Any]) -> dict[str, Any]:
    old = {key: np.load(OLD_TUPLE_NPZ)[key] for key in (
        "root_base", "node_sizes", "pair_ptr", "pair_t", "pair_a",
        "pair_wJ", "pair_wI", "root_atom",
    )}
    phi_env, atom_env, node_sizes_env = prev._env_phi_atom()
    checks: dict[str, Any] = {
        "old_tuple_npz": str(OLD_TUPLE_NPZ),
        "old_tuple_npz_sha256": file_sha256(OLD_TUPLE_NPZ),
        "node_sizes_match_env": bool(np.array_equal(old["node_sizes"], node_sizes_env)),
        "root_atom_match_env": bool(np.array_equal(old["root_atom"], atom_env)),
        "pair_ptr_monotone": bool(np.all(np.diff(old["pair_ptr"]) >= 0)),
        "n_roots": int(old["root_base"][-1]),
        "n_pairs": int(old["pair_ptr"][-1]),
    }
    if not checks["node_sizes_match_env"] or not checks["root_atom_match_env"]:
        raise RuntimeError("reused tuple structure does not match the env cache")
    if not checks["pair_ptr_monotone"]:
        raise RuntimeError("pair_ptr is not monotone")

    fit_idx = np.asarray(fold["fit_idx"], np.int64)
    total_roots = int(node_sizes_env.sum())
    fit_mol = np.zeros(10000, dtype=bool)
    fit_mol[fit_idx] = True
    fit_root_mask = np.zeros(total_roots, dtype=bool)
    for index in np.nonzero(fit_mol)[0]:
        lo, hi = int(old["root_base"][index]), int(old["root_base"][index + 1])
        fit_root_mask[lo:hi] = True

    pair_root = np.repeat(np.arange(total_roots, dtype=np.int64), np.diff(old["pair_ptr"]))
    realized = old["pair_wJ"] > 0.0
    realized = realized & fit_root_mask[pair_root]
    fit_realized_roots = pair_root[realized]
    if fit_realized_roots.size == 0:
        raise RuntimeError("no realized fit incident tuples")
    phi_sample = phi_env[fit_realized_roots].astype(np.float64)
    phi_mean = phi_sample.mean(axis=0)
    phi_std = np.maximum(phi_sample.std(axis=0), prev.PHI_STD_FLOOR)
    z = (phi_env[fit_realized_roots].astype(np.float64) - phi_mean) / phi_std
    phi_scale = float(np.sqrt(np.mean(z * z)))
    if not math.isfinite(phi_scale) or phi_scale <= 1e-12:
        raise RuntimeError("phi L2-RMS scale is degenerate")

    fit_roots = np.nonzero(fit_root_mask)[0]
    rng = np.random.default_rng(int(KAPPA_SEED))
    take = min(int(KAPPA_SAMPLE_MAX), int(fit_roots.size))
    sample = np.sort(fit_roots[rng.permutation(int(fit_roots.size))[:take]])
    if sample.size > KAPPA_SAMPLE_MAX:
        raise RuntimeError("kappa sample larger than cap")
    if not fit_root_mask[sample].all():
        raise RuntimeError("kappa sample has off-fit roots")

    kappa, kappa_stats = prev._kappa_for_roots(
        prev.init_d_loc(),
        phi_env, atom_env, old["pair_ptr"], old["pair_t"], old["pair_a"],
        old["pair_wJ"], old["pair_wI"],
        phi_mean.astype(np.float32), phi_std.astype(np.float32), np.float32(phi_scale),
        sample,
    )
    if not math.isfinite(kappa) or kappa <= 0.0:
        raise RuntimeError(f"MECHANISM_INIT_BLOCKED: kappa={kappa}")

    old_npz = np.load(OLD_TUPLE_NPZ)
    old_phi_mean = old_npz["phi_mean"].astype(np.float64)
    old_phi_std = old_npz["phi_std"].astype(np.float64)
    old_phi_scale = float(old_npz["phi_scale"])
    checks["scaler_old_vs_new"] = {
        "phi_mean_max_abs_diff": float(np.max(np.abs(phi_mean - old_phi_mean))),
        "phi_std_max_abs_diff": float(np.max(np.abs(phi_std - old_phi_std))),
        "phi_scale_old": old_phi_scale,
        "phi_scale_new": phi_scale,
        "kappa_old": float(old_npz["kappa"]),
        "kappa_new": float(kappa),
    }
    checks["phi_scaler"] = {
        "fitted_on": "new fit rows, realized C>0 incident tuples",
        "n_samples": int(fit_realized_roots.size),
        "std_floor": float(prev.PHI_STD_FLOOR),
        "phi_scale_l2_rms": phi_scale,
        "phi_mean_sha256": array_sha256(phi_mean.astype(np.float32)),
        "phi_std_sha256": array_sha256(phi_std.astype(np.float32)),
    }
    checks["kappa"] = {
        "value": float(kappa),
        "definition": "1 / RMS(initial e_J, e_I) over up to 8192 uniformly sampled new-fit roots",
        "seed": int(KAPPA_SEED),
        "n_roots": int(sample.size),
        "sample_sha256": array_sha256(sample, np.int64),
        "sample_head": sample[:8].tolist(),
        "raw_rms": kappa_stats["rms"],
        "e_J_rms": kappa_stats["e_J_rms"],
        "e_I_rms": kappa_stats["e_I_rms"],
        "sample_in_new_fit": True,
    }
    arrays = dict(old)
    arrays.update(
        phi_mean=phi_mean.astype(np.float32),
        phi_std=phi_std.astype(np.float32),
        phi_scale=np.asarray(phi_scale, np.float32),
        kappa=np.asarray(kappa, np.float32),
        kappa_sample=sample.astype(np.int64),
    )
    return {"arrays": arrays, "checks": checks}


def compute_kappa_M(payload_arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    payload = prev.TuplePayload(payload_arrays)
    phi_env, atom_env, _node = prev._env_phi_atom()
    phi_t = torch.as_tensor(phi_env, dtype=torch.float32)
    atom_t = torch.as_tensor(atom_env, dtype=torch.long)
    root_ids = torch.as_tensor(payload.kappa_sample, dtype=torch.long)
    d_init = prev.init_d_loc()
    a_init = d_init.t().contiguous()
    with torch.no_grad():
        e_d = prev.compose_root_codes(
            d_init, phi_t, atom_t, payload.pair_ptr, payload.pair_t, payload.pair_a,
            payload.pair_wJ, root_ids, payload.phi_mean, payload.phi_std, payload.phi_scale,
        )
        e_m = mlpmod.compose_root_codes_m(
            a_init, phi_t, atom_t, payload.pair_ptr, payload.pair_t, payload.pair_a,
            payload.pair_wJ, root_ids, payload.phi_mean, payload.phi_std, payload.phi_scale,
        )
    kappa_D = float(payload.kappa)
    e_d_rms = float(e_d.pow(2).mean().sqrt())
    e_m_rms = float(e_m.pow(2).mean().sqrt())
    kappa_M = kappa_D * e_d_rms / e_m_rms if e_m_rms > 0 else float("nan")
    if not math.isfinite(kappa_M) or kappa_M <= 0.0 or kappa_M > 1e9:
        raise RuntimeError(f"MECHANISM_INIT_BLOCKED: kappa_M={kappa_M}")
    return {
        "definition": "kappa_M = RMS(kappa_D * e_D_init) / RMS(e_M_init) over the fresh fit-only kappa_sample",
        "kappa_D": kappa_D,
        "r_D": kappa_D * e_d_rms,
        "r_M": e_m_rms,
        "e_D_init_rms": e_d_rms,
        "e_M_init_rms": e_m_rms,
        "kappa_M": float(kappa_M),
        "n_roots": int(payload.kappa_sample.numel()),
        "sample_sha256": array_sha256(payload.kappa_sample.numpy(), np.int64),
        "labels_used": False,
        "dev_used": False,
    }


# ---------------------------------------------------------------------------
# 4. fresh body prep (new-fit-only standardizers)
# ---------------------------------------------------------------------------


def build_fresh_prep(train_data: Sequence[Any], fold: Mapping[str, Any]) -> dict[str, Any]:
    blob = np.load(PREP_BLOB, allow_pickle=False)
    allowed = (
        "patch_all_mean", "patch_all_scale", "ctx_all_mean", "ctx_all_scale",
        "anchor_all_mean", "anchor_all_scale", "topo_all_mean", "topo_all_scale",
    )
    recovery = {key: np.asarray(blob[key], np.float32) for key in allowed}
    meta = zw.apply_new_fit_prep(train_data, blob, fold["fit_idx"])
    meta = {key: value for key, value in meta.items()}
    meta["recovery_constants"] = {
        key: {"path": str(PREP_BLOB), "sha256": array_sha256(value)} for key, value in recovery.items()
    }
    meta["fitted_rows"] = "new fit only"
    return meta


# ---------------------------------------------------------------------------
# 5. loading the frozen fresh objects (explicit paths, hash checked)
# ---------------------------------------------------------------------------


def _artifact_paths(out_dir: Path) -> dict[str, Path]:
    return {
        "fold_npz": out_dir / "fresh_fold.npz",
        "targets_npz": out_dir / "fresh_targets.npz",
        "payload_npz": out_dir / "fresh_tuple_payload.npz",
        "prep_npz": out_dir / "fresh_prep.npz",
        "manifest": out_dir / "fresh_manifest.json",
    }


def load_fresh_objects(out_dir: Path = RESULTS_DIR) -> dict[str, Any]:
    paths = _artifact_paths(out_dir)
    manifest = json.loads(paths["manifest"].read_text())
    for name, path in paths.items():
        if name == "manifest":
            continue
        expected = manifest["artifacts"][name]["sha256"]
        actual = file_sha256(path)
        if actual != expected:
            raise RuntimeError(f"fresh artifact {name} sha256 {actual} != frozen {expected}")

    with np.load(paths["fold_npz"], allow_pickle=False) as z:
        fold = {"fit_idx": z["fit_idx"].astype(np.int64), "dev_idx": z["dev_idx"].astype(np.int64)}
    if _hash_bytes(fold["fit_idx"].tobytes()) != NEW_FIT_SHA or _hash_bytes(fold["dev_idx"].tobytes()) != NEW_DEV_SHA:
        raise RuntimeError("fresh fold hashes do not match the frozen values")

    with np.load(paths["targets_npz"], allow_pickle=False) as z:
        targets = {
            "y": z["y"].astype(np.float64),
            "c": z["c"].astype(np.float64),
            "g": z["g"].astype(np.float64),
            "k": z["k"].astype(np.int64),
            "gid": z["gid"].astype(np.int64),
        }
    with np.load(paths["payload_npz"], allow_pickle=False) as z:
        payload_arrays = {key: z[key] for key in z.files}
    with np.load(paths["prep_npz"], allow_pickle=False) as z:
        prep = {key: z[key] for key in z.files}
    kappa = json.loads((out_dir / "fresh_kappa.json").read_text())
    if abs(float(kappa["kappa_M"]) - float(manifest["kappa_M"]["kappa_M"])) > 0.0:
        raise RuntimeError("fresh_kappa.json kappa_M differs from the frozen manifest")
    return {
        "fold": fold,
        "targets": targets,
        "payload_arrays": payload_arrays,
        "prep": prep,
        "kappa": kappa,
        "manifest": manifest,
    }


def build_prepared_data(fresh: Mapping[str, Any], *, verify_prep: bool = True):
    train_data = zftd.load_train_only()
    if len(train_data) != 10000:
        raise RuntimeError("train-only cache length mismatch")
    fold = fresh["fold"]
    prep_meta = build_fresh_prep(train_data, fold)
    if verify_prep:
        for key, value in fresh["prep"].items():
            if key not in prep_meta:
                raise RuntimeError(f"fresh prep key missing from recomputation: {key}")
            if not np.array_equal(np.asarray(prep_meta[key], np.float32), np.asarray(value, np.float32)):
                raise RuntimeError(f"fresh prep mismatch at {key}")
    fit_data, dev_data = zftd.build_fit_dev(train_data, fold["fit_idx"], fold["dev_idx"])
    for position, index in enumerate(fold["fit_idx"].tolist()):
        fit_data[position].local_mol_id = torch.tensor([int(index)], dtype=torch.long)
    for position, index in enumerate(fold["dev_idx"].tolist()):
        dev_data[position].local_mol_id = torch.tensor([int(index)], dtype=torch.long)
    if len(fit_data) != N_FIT or len(dev_data) != N_DEV:
        raise RuntimeError("fit/dev rebuild length mismatch")
    return prep_meta, fit_data, dev_data


# ---------------------------------------------------------------------------
# 6. models (shared skeleton; only the local encoder differs)
# ---------------------------------------------------------------------------


def build_arm_b() -> torch.nn.Module:
    seed_everything(SEED)
    return zw.build_arm("M")


def build_arm_mj(payload: prev.TuplePayload, kappa_M: float) -> torch.nn.Module:
    seed_everything(SEED)
    return mlpmod.build_arm_mj(payload, kappa_M=float(kappa_M))


def init_identity_witness(payload: prev.TuplePayload, kappa_M: float) -> dict[str, Any]:
    """Same untrained skeleton for both arms; W_loc zero; A = D_init.T; RNG state equal."""
    model_b = build_arm_b()
    state_b = {key: value.detach().cpu().clone() for key, value in model_b.state_dict().items()}
    rng_after_b = torch.get_rng_state()
    model_m = build_arm_mj(payload, kappa_M)
    state_m = {key: value.detach().cpu().clone() for key, value in model_m.state_dict().items()}
    rng_after_m = torch.get_rng_state()

    shared_b = sorted(key for key in state_b if not key.startswith("local_tuple."))
    shared_m = sorted(key for key in state_m if not key.startswith("local_tuple."))
    if shared_b != shared_m:
        raise RuntimeError("shared state key sets differ")
    max_abs = 0.0
    for key in shared_b:
        max_abs = max(max_abs, float((state_b[key] - state_m[key]).abs().max()))
    if max_abs != 0.0:
        raise RuntimeError(f"shared initial tensors differ (max abs {max_abs})")

    a_init = state_m["local_tuple.A_raw"]
    a_expected = prev.init_d_loc().t().contiguous()
    w_loc = state_m["local_tuple.W_loc"]
    return {
        "shared_tensor_count": len(shared_b),
        "shared_tensor_max_abs_diff": max_abs,
        "shared_state_hash_b": state_hash({k: state_b[k] for k in shared_b}),
        "shared_state_hash_m": state_hash({k: state_m[k] for k in shared_m}),
        "A_equals_D_init_T": bool(torch.equal(a_init, a_expected)),
        "W_loc_exactly_zero": bool(float(w_loc.abs().sum()) == 0.0),
        "parameter_audit_b": int(sum(p.numel() for p in model_b.parameters())),
        "parameter_audit_m": mlpmod.parameter_audit_mj(model_m),
        "rng_state_equal_after_construction": bool(torch.equal(rng_after_b, rng_after_m)),
        "rng_after_b_sha256": _hash_bytes(rng_after_b.numpy().tobytes()),
        "rng_after_m_sha256": _hash_bytes(rng_after_m.numpy().tobytes()),
    }


# ---------------------------------------------------------------------------
# 7. phase A — build + freeze all fresh objects
# ---------------------------------------------------------------------------


def phase_a(*, out_dir: Path = RESULTS_DIR, log: Any = print) -> dict[str, Any]:
    torch.set_num_threads(8)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    fold = build_fresh_fold()
    group_cross = canonical_group_cross_counts(fold)
    rows, gid, label_checks = load_train_only_raw_rows()
    targets = refit_new_targets(rows, fold["fit_idx"], old_anchor=True)
    shuffle_check = dev_label_shuffle_independence(rows, fold["fit_idx"], fold["dev_idx"])
    if not shuffle_check["identical"]:
        raise RuntimeError(f"dev label shuffle changed the fit constants: {shuffle_check}")

    payload = build_fresh_payload(fold)
    kappa_m = compute_kappa_M(payload["arrays"])
    init_identity = init_identity_witness(prev.TuplePayload(payload["arrays"]), float(kappa_m["kappa_M"]))
    write_json(out_dir / "init_identity.json", init_identity)

    train_data = zftd.load_train_only()
    # capture raw (cache-recovered) values independently, then refit directly
    blob = np.load(PREP_BLOB, allow_pickle=False)
    raw_blocks: dict[str, tuple[np.ndarray, list[int]]] = {}
    for attr, all_mean_key, all_scale_key in (
        ("patch_cont", "patch_all_mean", "patch_all_scale"),
        ("global_context", "ctx_all_mean", "ctx_all_scale"),
        ("anchor", "anchor_all_mean", "anchor_all_scale"),
        ("topology_features", "topo_all_mean", "topo_all_scale"),
    ):
        values, counts = zjd._stack(train_data, attr)
        values = zjd.Std(blob[all_mean_key], blob[all_scale_key]).inverse(values)
        raw_blocks[attr] = (values, counts)
    prep_meta = build_fresh_prep(train_data, fold)
    fit_data, dev_data = zftd.build_fit_dev(train_data, fold["fit_idx"], fold["dev_idx"])
    direct: dict[str, Any] = {}
    for attr, key in (("patch_cont", "patch"), ("global_context", "ctx"), ("anchor", "anchor"), ("topology_features", "topo")):
        values, counts = raw_blocks[attr]
        mask = zjd._root_mask(counts, fold["fit_idx"])
        refit = zjd.Std.fit(values[mask], floor=zjd.CANON_STD_FLOOR)
        direct[key] = {
            "n_fit_root_rows": int(mask.sum()),
            "mean_max_abs_diff": float(np.max(np.abs(
                refit.mean.astype(np.float64) - np.asarray(prep_meta[f"{key}_fit_mean"], np.float64)
            ))),
            "scale_max_abs_diff": float(np.max(np.abs(
                refit.scale.astype(np.float64) - np.asarray(prep_meta[f"{key}_fit_scale"], np.float64)
            ))),
        }
    prep_check = {
        "n_fit": len(fit_data),
        "n_dev": len(dev_data),
        "fitted_on_new_fit_only": True,
        "direct_new_fit_recompute": direct,
    }

    np.savez_compressed(
        out_dir / "fresh_fold.npz",
        fit_idx=fold["fit_idx"],
        dev_idx=fold["dev_idx"],
    )
    np.savez_compressed(
        out_dir / "fresh_targets.npz",
        y=targets["arrays"]["y"],
        c=targets["arrays"]["c"],
        g=targets["arrays"]["g"],
        k=targets["arrays"]["k"],
        gid=gid,
        constants=np.asarray([targets["constants"][key] for key in
                              ("sigma_logP", "sigma_SA", "mu_SA", "sigma_cycle", "mu_cycle")], np.float64),
        constant_names=np.asarray(["sigma_logP", "sigma_SA", "mu_SA", "sigma_cycle", "mu_cycle"]),
    )
    np.savez_compressed(out_dir / "fresh_tuple_payload.npz", **payload["arrays"])
    np.savez_compressed(
        out_dir / "fresh_prep.npz",
        **{key: np.asarray(prep_meta[key], np.float32) for key in (
            "patch_fit_mean", "patch_fit_scale", "ctx_fit_mean", "ctx_fit_scale",
            "anchor_fit_mean", "anchor_fit_scale", "topo_fit_mean", "topo_fit_scale",
        )},
    )
    write_json(out_dir / "fresh_kappa.json", kappa_m)

    artifacts = {}
    for name, path in _artifact_paths(out_dir).items():
        if name == "manifest":
            continue
        artifacts[name] = {"path": str(path), "sha256": file_sha256(path), "bytes": int(path.stat().st_size)}
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "fold": fold["definition"],
        "fold_checks": fold["checks"],
        "group_cross": group_cross,
        "label_source_checks": label_checks,
        "target_checks": targets["checks"],
        "dev_label_shuffle_check": shuffle_check,
        "payload_checks": payload["checks"],
        "kappa_M": kappa_m,
        "init_identity": init_identity,
        "prep_check": prep_check,
        "artifacts": artifacts,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(out_dir / "fresh_manifest.json", manifest)
    input_manifest = {
        "sources": {
            "train_label_csv": {"path": str(TRAIN_LABEL_CSV), "sha256": file_sha256(TRAIN_LABEL_CSV)},
            "gvae_full_properties_npz": {"path": str(GVEA_PROPS_NPZ), "sha256": file_sha256(GVEA_PROPS_NPZ)},
            "handoff_train": {"path": str(HANDOFF_TRAIN), "sha256": file_sha256(HANDOFF_TRAIN)},
            "prep_blob": {"path": str(PREP_BLOB), "sha256": file_sha256(PREP_BLOB)},
            "old_tuple_npz": {"path": str(OLD_TUPLE_NPZ), "sha256": file_sha256(OLD_TUPLE_NPZ)},
            "old_targets_npz_lineage_only": {"path": str(OLD_TARGETS_NPZ), "sha256": file_sha256(OLD_TARGETS_NPZ)},
        },
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(out_dir / "input_manifest.json", input_manifest)
    write_json(out_dir / "fold_meta.json", {"definition": fold["definition"], "checks": fold["checks"], "group_cross": group_cross})
    write_json(out_dir / "targets_meta.json", {"constants": targets["constants"], "checks": targets["checks"], "dev_label_shuffle_check": shuffle_check})
    write_json(out_dir / "tuple_payload_meta.json", {"checks": payload["checks"]})
    write_json(out_dir / "prep_meta.json", {key: value for key, value in prep_meta.items() if not isinstance(value, np.ndarray)})

    result = {
        "seconds": float(time.perf_counter() - t0),
        "fold_checks": fold["checks"],
        "group_cross": group_cross,
        "label_checks": label_checks,
        "constants": targets["constants"],
        "target_checks": targets["checks"],
        "dev_label_shuffle_check": shuffle_check,
        "payload_checks": payload["checks"],
        "kappa_M": kappa_m,
        "artifact_hashes": {name: value["sha256"] for name, value in artifacts.items()},
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    log(f"[phase-a] fold/new targets/payload/prep built in {result['seconds']:.1f}s")
    log(f"[phase-a] new_fit={fold['checks']['new_fit_sha256'][:12]} new_dev={fold['checks']['new_dev_sha256'][:12]}")
    log(f"[phase-a] constants={targets['constants']}")
    log(f"[phase-a] kappa_D={kappa_m['kappa_D']:.6f} kappa_M={kappa_m['kappa_M']:.6f}")
    return result


# ---------------------------------------------------------------------------
# 8. training (recipe copied; data objects explicit)
# ---------------------------------------------------------------------------


def _model_for(arm: str, payload: prev.TuplePayload, kappa_M: float) -> torch.nn.Module:
    if arm == "B":
        return build_arm_b()
    if arm == "M":
        return build_arm_mj(payload, kappa_M)
    raise ValueError(arm)


def _probe(model: torch.nn.Module, arm: str, epoch: int, step: int, total_norm: float) -> dict[str, Any]:
    entry: dict[str, Any] = {"epoch": int(epoch), "step_in_epoch": int(step), "clip_total_norm": float(total_norm)}
    if arm == "M":
        w_loc = model.local_tuple.W_loc
        a_raw = model.local_tuple.A_raw
        entry["W_loc_grad_norm"] = float(w_loc.grad.norm()) if w_loc.grad is not None else None
        entry["A_grad_norm"] = float(a_raw.grad.norm()) if a_raw.grad is not None else None
        entry["W_loc_norm"] = float(w_loc.detach().norm())
        entry["A_norm"] = float(a_raw.detach().norm())
        entry["health"] = {key: value for key, value in model.local_tuple.last_stats.items()}
    else:
        for name, parameter in model.named_parameters():
            if name.startswith("local_dictionary_bridge.") and parameter.grad is not None:
                entry[name] = float(parameter.grad.norm())
    return entry


def train_arm(
    arm: str,
    *,
    device: torch.device,
    out_dir: Path = RESULTS_DIR,
    epochs: int = EPOCHS,
    max_steps: int | None = None,
    log: Any = print,
) -> dict[str, Any]:
    torch.set_num_threads(8 if device.type == "cpu" else 4)
    out_dir.mkdir(parents=True, exist_ok=True)
    fresh = load_fresh_objects(out_dir)
    fold, targets = fresh["fold"], fresh["targets"]
    payload = prev.TuplePayload(fresh["payload_arrays"])
    kappa_M = float(fresh["kappa"]["kappa_M"])
    g = np.asarray(targets["g"], np.float64)
    g_fit = g[fold["fit_idx"]]
    target_fit = torch.as_tensor(g_fit, dtype=torch.float32)

    prep_meta, fit_data, dev_data = build_prepared_data(fresh, verify_prep=True)
    schedule, schedule_hash = zw.build_schedule(len(fit_data), int(epochs), SEED + TRAIN_SHUFFLE_OFFSET)
    if int(epochs) == EPOCHS and schedule_hash != FROZEN_SCHEDULE_SHA:
        raise RuntimeError(f"schedule hash {schedule_hash} != frozen {FROZEN_SCHEDULE_SHA}")

    seed_everything(SEED)
    model = _model_for(arm, payload, kappa_M)
    build_rng = torch.get_rng_state()
    model = model.to(device)
    seed_everything(SEED)
    train_rng = torch.get_rng_state()
    audit = (
        mlpmod.parameter_audit_mj(model)
        if arm == "M"
        else {
            "total_parameters": int(sum(p.numel() for p in model.parameters())),
            "body_parameters": zw.body_parameter_count(model),
            "bridge_parameters": zw.bridge_parameter_count(model),
        }
    )
    init_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    soup_epochs = set(int(e) for e in SOUP_EPOCHS)
    soup: dict[int, dict[str, torch.Tensor]] = {}
    curve: list[dict[str, Any]] = []
    probe_log: list[dict[str, Any]] = []
    steps_done = 0
    gid_stream = hashlib.sha256()
    id_stream = hashlib.sha256()
    started = time.perf_counter()
    stopped_reason = "completed"

    for epoch in range(1, int(epochs) + 1):
        epoch_started = time.perf_counter()
        model.train()
        task_sum, n_mol, n_steps, gnorm_sum, clip_hits = 0.0, 0, 0, 0.0, 0
        for start in range(0, len(schedule[epoch - 1]), BATCH_SIZE):
            indices = schedule[epoch - 1][start : start + BATCH_SIZE]
            index_list = [int(i) for i in indices.tolist()]
            id_stream.update(np.asarray(index_list, np.int64).tobytes())
            gid_stream.update(np.asarray(targets["gid"][fold["fit_idx"][index_list]], np.int64).tobytes())
            batch = zftd.make_batch(fit_data, index_list, target_fit, device)
            prediction = model(batch, mask=cm.C6_MASK)
            loss = F.l1_loss(prediction.view(-1), batch.y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            total_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP))
            if epoch in LOG_EPOCHS and n_steps == 0:
                probe_log.append(_probe(model, arm, epoch, 1, total_norm))
            optimizer.step()
            task_sum += float((prediction.view(-1) - batch.y.view(-1)).abs().sum())
            n_mol += int(batch.y.numel())
            n_steps += 1
            steps_done += 1
            gnorm_sum += total_norm
            clip_hits += int(total_norm > GRAD_CLIP)
            if max_steps is not None and steps_done >= int(max_steps):
                stopped_reason = "max_steps"
                break
        curve.append(
            {
                "epoch": int(epoch),
                "train_task_mae": float(task_sum / max(n_mol, 1)),
                "grad_norm": float(gnorm_sum / max(n_steps, 1)),
                "clip_fraction": float(clip_hits / max(n_steps, 1)),
                "seconds": float(time.perf_counter() - epoch_started),
            }
        )
        if epoch in soup_epochs:
            soup[int(epoch)] = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if log and (epoch == 1 or epoch % 20 == 0 or epoch == int(epochs)):
            log(
                f"[{arm}] ep={epoch:03d} train={curve[-1]['train_task_mae']:.6f} "
                f"gnorm={curve[-1]['grad_norm']:.3g} clip={curve[-1]['clip_fraction']:.3f} "
                f"{curve[-1]['seconds']:.1f}s"
            )
        if max_steps is not None and steps_done >= int(max_steps):
            break

    if not soup and max_steps is not None:
        soup[int(epoch)] = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    members = sorted(soup)
    soup_state = {key: torch.stack([soup[e][key].float() for e in members]).mean(0) for key in soup[members[0]]}
    last_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    result: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "arm": arm,
        "supervision_target": "g = y - c (fresh-fold fit-only constants)",
        "seed": SEED,
        "epochs": int(epochs),
        "steps_done": int(steps_done),
        "steps_expected": int(epochs) * math.ceil(len(fit_data) / BATCH_SIZE),
        "stopped_reason": stopped_reason,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "grad_clip": GRAD_CLIP,
        "batch_size": BATCH_SIZE,
        "soup_epochs": members if max_steps is None else [],
        "n_fit": int(len(fit_data)),
        "n_dev": int(len(dev_data)),
        "kappa_M": kappa_M if arm == "M" else None,
        "fold": {
            "definition": fold.get("definition", "fresh fold npz"),
            "fit_idx_sha256": _hash_bytes(fold["fit_idx"].tobytes()),
            "dev_idx_sha256": _hash_bytes(fold["dev_idx"].tobytes()),
        },
        "target": {
            "source": str(out_dir / "fresh_targets.npz"),
            "g_sha256": array_sha256(g, np.float64),
            "definition": "g = y - c; constants fitted on new 8000 fit rows only",
        },
        "parameter_audit": audit,
        "init_state_hash": state_hash(init_state),
        "raw_soup_state_hash": state_hash(soup_state),
        "last_state_hash": state_hash(last_state),
        "build_rng_sha256": _hash_bytes(build_rng.numpy().tobytes()),
        "train_rng_sha256": _hash_bytes(train_rng.numpy().tobytes()),
        "schedule_sha256": schedule_hash,
        "position_stream_sha256": id_stream.hexdigest(),
        "global_gid_stream_sha256": gid_stream.hexdigest(),
        "curve": curve,
        "probe_log": probe_log,
        "wall_clock_s": float(time.perf_counter() - started),
        "device": str(device),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }

    if arm == "M":
        a_init = init_state["local_tuple.A_raw"]
        result.update(
            {
                "init_A_sha256": array_sha256(a_init.numpy()),
                "soup_A_sha256": array_sha256(soup_state["local_tuple.A_raw"].numpy()),
                "soup_W_loc_sha256": array_sha256(soup_state["local_tuple.W_loc"].numpy()),
                "soup_W_loc_norm": float(soup_state["local_tuple.W_loc"].norm()),
                "last_W_loc_norm": float(last_state["local_tuple.W_loc"].norm()),
                "A_relative_drift_soup": float(
                    (soup_state["local_tuple.A_raw"] - a_init).norm() / a_init.norm()
                ),
                "A_relative_drift_last": float(
                    (last_state["local_tuple.A_raw"] - a_init).norm() / a_init.norm()
                ),
            }
        )

    if max_steps is not None:
        result["smoke_last_state_hash"] = state_hash(last_state)
        return result

    torch.save(init_state, out_dir / f"{arm}_init_state.pt")
    torch.save(last_state, out_dir / f"{arm}_last_state.pt")
    torch.save(soup_state, out_dir / f"{arm}_raw_soup_state.pt")

    raw_predictions: dict[str, np.ndarray] = {}
    for state_name, state in (("init", init_state), ("last", last_state), ("raw_soup", soup_state)):
        replay = _model_for(arm, payload, kappa_M)
        replay.load_state_dict({key: value.to(torch.device("cpu")) for key, value in state.items()}, strict=True)
        replay = replay.to(device)
        raw_predictions[f"{state_name}_fit"] = zw.evaluate_state(replay, fit_data, g_fit, device)
        raw_predictions[f"{state_name}_dev"] = zw.evaluate_state(replay, dev_data, g[fold["dev_idx"]], device)
    np.savez_compressed(
        out_dir / f"{arm}_raw_predictions.npz",
        **{key: value.astype(np.float32) for key, value in raw_predictions.items()},
    )

    b_soup = float(np.median(g_fit - raw_predictions["raw_soup_fit"]))
    b_init = float(np.median(g_fit - raw_predictions["init_fit"]))
    b_last = float(np.median(g_fit - raw_predictions["last_fit"]))
    result["calibration_b"] = {"init": b_init, "last": b_last, "raw_soup": b_soup}

    replay = _model_for(arm, payload, kappa_M)
    replay.load_state_dict({key: value.to(torch.device("cpu")) for key, value in soup_state.items()}, strict=True)
    batch_data = fit_data[:128]
    gpu_pred = zw.evaluate_state(replay.to(device), batch_data, g_fit[:128], device)
    cpu_pred = zw.evaluate_state(
        replay.to(torch.device("cpu")), batch_data, g_fit[:128], torch.device("cpu")
    )
    max_diff = float(np.max(np.abs(gpu_pred - cpu_pred))) if device.type != "cpu" else 0.0
    result["replay_max_abs_diff"] = max_diff
    if device.type != "cpu" and max_diff > REPLAY_TOL:
        raise RuntimeError(f"CPU/GPU replay maxdiff {max_diff} > {REPLAY_TOL}")
    write_json(out_dir / f"{arm}_meta.json", result)
    write_json(out_dir / f"{arm}_curve.json", curve)
    log(f"[{arm}] done steps={steps_done} wall={result['wall_clock_s']:.1f}s b={b_soup:.6f} replay={max_diff:.2e}")
    return result


# ---------------------------------------------------------------------------
# 9. smoke (fit-only; states discarded)
# ---------------------------------------------------------------------------


def run_smoke(
    *,
    arm: str,
    device: torch.device,
    out_dir: Path = RESULTS_DIR,
    max_steps: int = 3,
    log: Any = print,
) -> dict[str, Any]:
    torch.set_num_threads(8)
    fresh = load_fresh_objects(out_dir)
    payload = prev.TuplePayload(fresh["payload_arrays"])
    kappa_M = float(fresh["kappa"]["kappa_M"])
    fold = fresh["fold"]
    prep_meta, fit_data, dev_data = build_prepared_data(fresh, verify_prep=True)
    g = np.asarray(fresh["targets"]["g"], np.float64)
    g_fit = g[fold["fit_idx"]]
    target_fit = torch.as_tensor(g_fit, dtype=torch.float32)
    schedule, schedule_hash = zw.build_schedule(len(fit_data), EPOCHS, SEED + TRAIN_SHUFFLE_OFFSET)
    checks: dict[str, Any] = {
        "arm": arm,
        "max_steps": int(max_steps),
        "schedule_hash": schedule_hash,
        "schedule_matches_frozen": schedule_hash == FROZEN_SCHEDULE_SHA,
    }

    init = init_identity_witness(payload, kappa_M) if arm == "M" else None
    if init is not None:
        checks["init_identity"] = init

    seed_everything(SEED)
    model = _model_for(arm, payload, kappa_M).to(device)
    seed_everything(SEED)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    steps = 0
    grad_probe: list[dict[str, Any]] = []
    first_health: dict[str, Any] = {}
    model.train()
    indices = [int(i) for i in schedule[0][:BATCH_SIZE].tolist()]
    for step in range(1, int(max_steps) + 1):
        batch = zftd.make_batch(fit_data, indices, target_fit, device)
        prediction = model(batch, mask=cm.C6_MASK)
        loss = F.l1_loss(prediction.view(-1), batch.y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        total_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP))
        grad_probe.append(_probe(model, arm, 1, step, total_norm))
        if step == 1 and arm == "M":
            first_health = dict(model.local_tuple.last_stats)
        optimizer.step()
        steps += 1
        if step == 2 and arm == "M":
            # W_loc is nonzero now (one Adam step); A must receive task gradient.
            batch2 = zftd.make_batch(fit_data, indices, target_fit, device)
            prediction2 = model(batch2, mask=cm.C6_MASK)
            loss2 = F.l1_loss(prediction2.view(-1), batch2.y.view(-1))
            optimizer.zero_grad()
            loss2.backward()
            grad_probe[-1] = _probe(model, arm, 1, step, float("nan"))
            grad_probe[-1]["W_loc_grad_norm_step_after"] = float(model.local_tuple.W_loc.grad.norm()) if model.local_tuple.W_loc.grad is not None else None
            grad_probe[-1]["A_grad_norm_step_after"] = float(model.local_tuple.A_raw.grad.norm()) if model.local_tuple.A_raw.grad is not None else None
            optimizer.zero_grad()
    checks["steps_done"] = steps
    checks["grad_probe"] = grad_probe
    checks["first_health"] = {key: value for key, value in first_health.items() if key != "atoms_used_mask"}

    # label permutation must not change the forward prediction
    model.eval()
    with torch.no_grad():
        indices128 = [int(i) for i in schedule[0][:128].tolist()]
        batch = zftd.make_batch(fit_data, indices128, target_fit, device)
        p1 = model(batch, mask=cm.C6_MASK).view(-1).clone()
        perm = torch.randperm(batch.y.shape[0])
        batch.y = batch.y[perm].clone()
        p2 = model(batch, mask=cm.C6_MASK).view(-1).clone()
    checks["label_permutation_forward_max_abs_diff"] = float((p1 - p2).abs().max())
    checks["label_permutation_forward_unchanged"] = bool(float((p1 - p2).abs().max()) == 0.0)

    # real save/reload path on the smoke state
    smoke_dir = out_dir / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    torch.save(state, smoke_dir / f"smoke_{arm}_state.pt")
    reloaded = _model_for(arm, payload, kappa_M)
    reloaded.load_state_dict(
        {key: value.to(torch.device("cpu")) for key, value in torch.load(smoke_dir / f"smoke_{arm}_state.pt", weights_only=True).items()},
        strict=True,
    )
    checks["save_reload_state_hash_equal"] = state_hash(state) == state_hash(
        {key: value.detach().cpu() for key, value in reloaded.state_dict().items()}
    )
    reloaded = reloaded.to(device)
    reloaded.eval()
    with torch.no_grad():
        batch = zftd.make_batch(fit_data, indices128, target_fit, device)
        p3 = reloaded(batch, mask=cm.C6_MASK).view(-1).clone()
    checks["save_reload_forward_max_abs_diff"] = float((p1 - p3).abs().max())
    for temp in smoke_dir.glob("*.pt"):
        temp.unlink()

    if arm == "M":
        w1 = grad_probe[0].get("W_loc_grad_norm")
        a1 = grad_probe[0].get("A_grad_norm")
        a2 = None
        for entry in grad_probe:
            if "A_grad_norm_step_after" in entry:
                a2 = entry["A_grad_norm_step_after"]
        checks["W_loc_task_grad_at_step1_nonzero"] = bool(w1 is not None and w1 > 0.0)
        checks["A_task_grad_at_step1_zero"] = bool(a1 is not None and a1 == 0.0)
        checks["A_task_grad_after_W_loc_step_nonzero"] = bool(a2 is not None and a2 > 0.0)
        checks["tuple_path_executes"] = bool(first_health.get("n_tuples", 0) > 0)
    checks["all_ok"] = bool(
        checks["schedule_matches_frozen"]
        and checks["steps_done"] == int(max_steps)
        and checks["label_permutation_forward_unchanged"]
        and checks["save_reload_state_hash_equal"]
        and checks["save_reload_forward_max_abs_diff"] == 0.0
        and (
            arm != "M"
            or (
                checks["W_loc_task_grad_at_step1_nonzero"]
                and checks["A_task_grad_at_step1_zero"]
                and checks["A_task_grad_after_W_loc_step_nonzero"]
                and checks["tuple_path_executes"]
            )
        )
    )
    write_json(out_dir / f"smoke_checks_{arm}.json", checks)
    if not checks["all_ok"]:
        raise RuntimeError(f"smoke checks failed: {checks}")
    return checks


# ---------------------------------------------------------------------------
# 10. per-graph operator / raw-graph / batch-offset checks (pre-training)
# ---------------------------------------------------------------------------


def operator_checks(*, out_dir: Path = RESULTS_DIR, device: torch.device | None = None, log: Any = print) -> dict[str, Any]:
    torch.set_num_threads(8)
    device = device or torch.device("cpu")
    fresh = load_fresh_objects(out_dir)
    fold = fresh["fold"]
    payload = prev.TuplePayload(fresh["payload_arrays"])
    kappa_M = float(fresh["kappa"]["kappa_M"])
    prep_meta, fit_data, dev_data = build_prepared_data(fresh, verify_prep=True)
    phi_env, atom_env, node_sizes = prev._env_phi_atom()
    root_base = payload.root_base.numpy()
    raw_graphs = prev.load_raw_train_graphs()
    chosen = [1, 3, 17, 401, 1999]
    target = torch.zeros(len(dev_data), dtype=torch.float32)

    model = build_arm_mj(payload, kappa_M)
    model.eval()
    checks: dict[str, Any] = {"chosen_dev_positions": chosen, "per_graph": []}
    with torch.no_grad():
        single_refs: list[np.ndarray] = []
        single_codes: list[np.ndarray] = []
        for position in chosen:
            data = dev_data[position]
            row = int(data.local_mol_id.item())
            lo, hi = int(root_base[row]), int(root_base[row + 1])
            raw_x, raw_edge_index, raw_edge_attr = raw_graphs[row]
            batch = zftd.make_batch(dev_data, [position], target, device)
            codes = model.local_tuple.root_codes(batch).double().cpu().numpy()
            root_ids = torch.arange(lo, hi, dtype=torch.long)
            ref = mlpmod.compose_root_codes_m(
                model.local_tuple.A_raw,
                torch.as_tensor(phi_env, dtype=torch.float32),
                torch.as_tensor(atom_env, dtype=torch.long),
                payload.pair_ptr,
                payload.pair_t,
                payload.pair_a,
                payload.pair_wJ,
                root_ids,
                payload.phi_mean,
                payload.phi_std,
                payload.phi_scale,
            ).double().numpy()
            single_refs.append(ref)
            single_codes.append(codes)
            checks["per_graph"].append(
                {
                    "dev_position": int(position),
                    "train_row": row,
                    "n_roots_batch": int(codes.shape[0]),
                    "n_roots_root_base": int(hi - lo),
                    "raw_graph_nodes_match_env": bool(int(raw_x.shape[0]) == int(hi - lo)),
                    "raw_graph_atoms_match_env": bool(np.array_equal(atom_env[lo:hi], raw_x)),
                    "raw_graph_edges": int(raw_edge_index.shape[1]),
                    "model_vs_reference_max_abs": float(np.max(np.abs(codes - ref))),
                }
            )
        # multi-graph concatenated batch must equal the concatenation of singles
        batch = zftd.make_batch(dev_data, chosen, target, device)
        batched = model.local_tuple.root_codes(batch).double().cpu().numpy()
        reference = np.concatenate(single_refs, axis=0)
        checks["batch_concat_max_abs_vs_singles"] = float(np.max(np.abs(batched - reference)))
        checks["batch_shape"] = [int(v) for v in batched.shape]
        checks["single_shapes"] = [list(map(int, arr.shape)) for arr in single_codes]
    checks["model_vs_reference_max_abs_overall"] = float(
        max(entry["model_vs_reference_max_abs"] for entry in checks["per_graph"])
    )
    checks["all_ok"] = bool(
        all(entry["raw_graph_nodes_match_env"] and entry["raw_graph_atoms_match_env"] for entry in checks["per_graph"])
        and checks["model_vs_reference_max_abs_overall"] <= 1e-5
        and checks["batch_concat_max_abs_vs_singles"] <= 1e-5
    )
    write_json(out_dir / "operator_path_checks.json", checks)
    log(f"[operator] all_ok={checks['all_ok']} model_vs_ref={checks['model_vs_reference_max_abs_overall']:.2e} "
        f"batch_vs_singles={checks['batch_concat_max_abs_vs_singles']:.2e}")
    if not checks["all_ok"]:
        raise RuntimeError(f"operator checks failed: {checks}")
    return checks


# ---------------------------------------------------------------------------
# 11. analysis (new dev only; two fitted arms)
# ---------------------------------------------------------------------------


def _mae(err: np.ndarray) -> float:
    return float(np.mean(np.abs(err))) if err.size else float("nan")


def _paired_bootstrap(
    err_b: Mapping[str, np.ndarray],
    err_m: Mapping[str, np.ndarray],
    k_dev: np.ndarray,
    *,
    seed: int = BOOT_SEED,
    n_boot: int = N_BOOT,
) -> dict[str, Any]:
    """1000 paired draws; one shared index set per endpoint per draw.

    ``err_b``/``err_m`` values for ``G0_*`` metrics are already G0-filtered;
    ``overall_*`` values are full-dev arrays.  The same index set is used for
    both arms, both raw/cal, within one endpoint per draw.
    """
    g0 = k_dev == 0
    n_all = int(k_dev.size)
    n_g0 = int(g0.sum())
    metrics = ("G0_cal", "overall_cal", "G0_raw", "overall_raw")
    samples = {metric: np.empty(int(n_boot), np.float64) for metric in metrics}
    rng = np.random.default_rng(int(seed))
    for draw in range(int(n_boot)):
        idx_all = rng.choice(n_all, size=n_all, replace=True)
        idx_g0 = rng.choice(n_g0, size=n_g0, replace=True)
        for metric in metrics:
            idx = idx_all if metric.startswith("overall") else idx_g0
            eb = np.abs(err_b[metric])[idx]
            em = np.abs(err_m[metric])[idx]
            samples[metric][draw] = float(eb.mean() - em.mean())
    out: dict[str, Any] = {"seed": int(seed), "n_boot": int(n_boot), "shared_indices_per_endpoint": True}
    for metric in metrics:
        point = _mae(err_b[metric]) - _mae(err_m[metric])
        out[metric] = {
            "point": point,
            "ci95": [float(np.percentile(samples[metric], 2.5)), float(np.percentile(samples[metric], 97.5))],
        }
    return out


def _bootstrap_witnesses(err_b: np.ndarray, err_m: np.ndarray, k_dev: np.ndarray) -> dict[str, Any]:
    g0 = k_dev == 0
    err_b = np.asarray(err_b, np.float64)
    err_m = np.asarray(err_m, np.float64)
    same_b = {"G0_cal": err_b[g0], "overall_cal": err_b, "G0_raw": err_b[g0], "overall_raw": err_b}
    same = _paired_bootstrap(same_b, same_b, k_dev, seed=BOOT_SEED, n_boot=100)
    forward_b = {"G0_cal": err_b[g0], "overall_cal": err_b, "G0_raw": err_b[g0], "overall_raw": err_b}
    forward_m = {"G0_cal": err_m[g0], "overall_cal": err_m, "G0_raw": err_m[g0], "overall_raw": err_m}
    forward = _paired_bootstrap(forward_b, forward_m, k_dev, seed=BOOT_SEED, n_boot=200)
    swapped = _paired_bootstrap(forward_m, forward_b, k_dev, seed=BOOT_SEED, n_boot=200)
    checks = {"same_predictions_zero": True}
    for metric in ("G0_cal", "overall_cal", "G0_raw", "overall_raw"):
        if not (same[metric]["point"] == 0.0 and same[metric]["ci95"] == [0.0, 0.0]):
            checks["same_predictions_zero"] = False
        checks[f"{metric}_swap_mirror_point"] = bool(
            abs(forward[metric]["point"] + swapped[metric]["point"]) < 1e-12
        )
        checks[f"{metric}_swap_mirror_ci"] = bool(
            abs(forward[metric]["ci95"][0] + swapped[metric]["ci95"][1]) < 1e-9
            and abs(forward[metric]["ci95"][1] + swapped[metric]["ci95"][0]) < 1e-9
        )
    checks["all_ok"] = all(checks.values())
    return {"same_predictions": same, "forward": forward, "swapped": swapped, "checks": checks}


def analyze(*, out_dir: Path = RESULTS_DIR, device: torch.device | None = None, log: Any = print) -> dict[str, Any]:
    torch.set_num_threads(8)
    device = device or torch.device("cpu")
    fresh = load_fresh_objects(out_dir)
    fold, targets = fresh["fold"], fresh["targets"]
    k = np.asarray(targets["k"], np.int64)
    dev_idx = fold["dev_idx"]
    k_dev = k[dev_idx]
    g = np.asarray(targets["g"], np.float64)
    g_dev = g[dev_idx]
    g_fit = g[fold["fit_idx"]]

    preds: dict[str, dict[str, np.ndarray]] = {}
    for arm in ("B", "M"):
        with np.load(out_dir / f"{arm}_raw_predictions.npz") as z:
            preds[arm] = {key: np.asarray(z[key], np.float64) for key in z.files}

    raw: dict[str, dict[str, np.ndarray]] = {}
    cal: dict[str, dict[str, np.ndarray]] = {}
    bias: dict[str, float] = {}
    for arm in ("B", "M"):
        bias[arm] = float(np.median(g_fit - preds[arm]["raw_soup_fit"]))
        raw[arm] = {
            "fit": preds[arm]["raw_soup_fit"],
            "dev": preds[arm]["raw_soup_dev"],
        }
        cal[arm] = {
            "fit": raw[arm]["fit"] + bias[arm],
            "dev": raw[arm]["dev"] + bias[arm],
        }

    def metrics_for(arm: str) -> dict[str, Any]:
        err_fit = g_fit - cal[arm]["fit"]
        err_dev = g_dev - cal[arm]["dev"]
        err_dev_raw = g_dev - raw[arm]["dev"]
        table = zw._metric_table(err_dev, k_dev)
        fit_table = zw._metric_table(err_fit, k[fold["fit_idx"]])
        return {
            "bias": bias[arm],
            "fit_overall_cal_mae": float(np.mean(np.abs(err_fit))),
            "dev_overall_raw_mae": float(np.mean(np.abs(err_dev_raw))),
            "dev_overall_cal_mae": float(np.mean(np.abs(err_dev))),
            "dev_G0_raw_mae": float(np.mean(np.abs(err_dev_raw[k_dev == 0]))),
            "dev_G0_cal_mae": float(np.mean(np.abs(err_dev[k_dev == 0]))),
            "dev_groups_cal": table,
            "fit_groups_cal": fit_table,
            "err_dev_cal": err_dev,
            "err_dev_raw": err_dev_raw,
        }

    arms: dict[str, Any] = {arm: metrics_for(arm) for arm in ("B", "M")}

    err_b = {
        "G0_cal": arms["B"]["err_dev_cal"][k_dev == 0],
        "overall_cal": arms["B"]["err_dev_cal"],
        "G0_raw": arms["B"]["err_dev_raw"][k_dev == 0],
        "overall_raw": arms["B"]["err_dev_raw"],
    }
    err_m = {
        "G0_cal": arms["M"]["err_dev_cal"][k_dev == 0],
        "overall_cal": arms["M"]["err_dev_cal"],
        "G0_raw": arms["M"]["err_dev_raw"][k_dev == 0],
        "overall_raw": arms["M"]["err_dev_raw"],
    }
    gains = _paired_bootstrap(err_b, err_m, k_dev, seed=BOOT_SEED, n_boot=N_BOOT)
    witnesses = _bootstrap_witnesses(arms["B"]["err_dev_cal"], arms["M"]["err_dev_cal"], k_dev)

    # pre-fixed sensitivity: drop the unique dev row with max mean calibrated error
    mean_err = (np.abs(arms["B"]["err_dev_cal"]) + np.abs(arms["M"]["err_dev_cal"])) / 2.0
    worst = int(np.argmax(mean_err))
    keep = np.ones(k_dev.size, dtype=bool)
    keep[worst] = False
    sens = {
        "dropped_dev_position": worst,
        "dropped_gid": int(targets["gid"][dev_idx[worst]]),
        "dropped_in_G0": bool(k_dev[worst] == 0),
        "G0_cal_gain": float(
            _mae(arms["B"]["err_dev_cal"][(k_dev == 0) & keep]) - _mae(arms["M"]["err_dev_cal"][(k_dev == 0) & keep])
        ),
        "overall_cal_gain": float(_mae(arms["B"]["err_dev_cal"][keep]) - _mae(arms["M"]["err_dev_cal"][keep])),
    }

    gate_conditions = {
        "G0_cal_gain_ge_0.003": bool(gains["G0_cal"]["point"] >= PERF_DELTA),
        "overall_cal_gain_ge_0.003": bool(gains["overall_cal"]["point"] >= PERF_DELTA),
        "G0_cal_ci_lower_gt_0": bool(gains["G0_cal"]["ci95"][0] > 0.0),
        "G0_raw_gain_gt_0": bool(gains["G0_raw"]["point"] > 0.0),
        "overall_raw_gain_gt_0": bool(gains["overall_raw"]["point"] > 0.0),
    }
    if all(gate_conditions.values()):
        classification = "PERFORMANCE_REPLICATED"
    elif gains["G0_cal"]["point"] > 0.0 and gains["overall_cal"]["point"] > 0.0:
        classification = "DIRECTIONAL_NOT_CONFIRMED"
    else:
        classification = "NOT_REPLICATED"

    # group contribution/gain identities
    identity = {}
    for arm in ("B", "M"):
        table = arms[arm]["dev_groups_cal"]
        contrib_sum = sum(table[name]["contribution"] for name in GROUP_NAMES if table[name]["n"] > 0)
        identity[f"{arm}_contribution_sum_minus_overall"] = float(contrib_sum - table["mae"])
    gain_sum = 0.0
    for name in GROUP_NAMES:
        gain_sum += arms["B"]["dev_groups_cal"][name]["contribution"] - arms["M"]["dev_groups_cal"][name]["contribution"]
    identity["group_gain_sum_minus_total"] = float(gain_sum - gains["overall_cal"]["point"])

    # mechanism + replay share one prepared data build
    fresh_payload = prev.TuplePayload(fresh["payload_arrays"])
    kappa_M = float(fresh["kappa"]["kappa_M"])
    prep_meta, fit_data, dev_data = build_prepared_data(fresh, verify_prep=True)

    # mechanism: load M soup and run a dev zero-ablation
    mech: dict[str, Any] = {"criteria": "HEALTHY unless the local path is dead or cannot execute"}
    m_meta = json.loads((out_dir / "M_meta.json").read_text())
    mech.update(
        {
            "A_relative_drift_soup": m_meta.get("A_relative_drift_soup"),
            "soup_W_loc_norm": m_meta.get("soup_W_loc_norm"),
            "probe_first_health": (m_meta.get("probe_log") or [{}])[0].get("health", {}),
        }
    )
    try:
        soup_state = torch.load(out_dir / "M_raw_soup_state.pt", weights_only=True, map_location="cpu")
        model = build_arm_mj(fresh_payload, kappa_M)
        model.load_state_dict({key: value.to(torch.device("cpu")) for key, value in soup_state.items()}, strict=True)
        model = model.to(device)
        base = zw.evaluate_state(model, dev_data, g_dev, device)
        model.local_tuple.ablate = True
        ablated = zw.evaluate_state(model, dev_data, g_dev, device)
        mech["zero_ablation_dev"] = {
            "mean_abs_delta_pred": float(np.mean(np.abs(base - ablated))),
            "cal_mae_full": float(np.mean(np.abs(g_dev - (base + bias["M"])))),
            "cal_mae_ablated": float(np.mean(np.abs(g_dev - (ablated + bias["M"])))),
        }
        mech["implementation_executes"] = True
    except Exception as exc:  # pragma: no cover
        mech["implementation_executes"] = False
        mech["error"] = repr(exc)

    health = mech.get("probe_first_health", {}) or {}
    dead_dims = int(health.get("root_code_dead_dims", 64)) if health else 64
    wloc_norm = float(mech.get("soup_W_loc_norm") or 0.0)
    a_drift = float(mech.get("A_relative_drift_soup") or 0.0)
    delta = float((mech.get("zero_ablation_dev") or {}).get("mean_abs_delta_pred", 0.0))
    if not mech.get("implementation_executes", False) or (a_drift == 0.0 and wloc_norm == 0.0):
        mechanism_class = "IMPLEMENTATION_FAILED"
    elif wloc_norm < 1e-3 or dead_dims >= 64 or delta < 1e-6:
        mechanism_class = "CHANNEL_COLLAPSED"
    else:
        mechanism_class = "HEALTHY"
    mech["classification"] = mechanism_class

    # replay: independently reload the stored soups and compare to stored predictions
    replay: dict[str, Any] = {}
    for arm in ("B", "M"):
        state = torch.load(out_dir / f"{arm}_raw_soup_state.pt", weights_only=True, map_location="cpu")
        model = build_arm_b() if arm == "B" else build_arm_mj(fresh_payload, kappa_M)
        model.load_state_dict({key: value.to(torch.device("cpu")) for key, value in state.items()}, strict=True)
        model = model.to(device)
        idx_fit = list(range(128))
        idx_dev = list(range(128))
        pred_fit = zw.evaluate_state(model, [fit_data[i] for i in idx_fit], g[fold["fit_idx"]][idx_fit], device)
        pred_dev = zw.evaluate_state(model, [dev_data[i] for i in idx_dev], g_dev[idx_dev], device)
        replay[arm] = {
            "fit_max_abs_vs_stored": float(np.max(np.abs(pred_fit - preds[arm]["raw_soup_fit"][idx_fit]))),
            "dev_max_abs_vs_stored": float(np.max(np.abs(pred_dev - preds[arm]["raw_soup_dev"][idx_dev]))),
        }
    replay["tol"] = REPLAY_TOL
    replay["all_ok"] = all(
        replay[arm][f"{split}_max_abs_vs_stored"] <= REPLAY_TOL for arm in ("B", "M") for split in ("fit", "dev")
    )

    main_rows = []
    for arm in ("B", "M"):
        table = arms[arm]
        main_rows.append(
            {
                "arm": arm,
                "fit_overall_cal": table["fit_overall_cal_mae"],
                "dev_overall_raw": table["dev_overall_raw_mae"],
                "dev_overall_cal": table["dev_overall_cal_mae"],
                "dev_G0_raw": table["dev_G0_raw_mae"],
                "dev_G0_cal": table["dev_G0_cal_mae"],
                "bias": table["bias"],
            }
        )
    group_rows = []
    for arm in ("B", "M"):
        for name in GROUP_NAMES:
            entry = arms[arm]["dev_groups_cal"][name]
            group_rows.append(
                {"arm": arm, "group": name, "n": entry["n"], "mae": entry["mae"], "contribution": entry["contribution"]}
            )

    per_graph = []
    for position, index in enumerate(dev_idx.tolist()):
        per_graph.append(
            {
                "dev_position": position,
                "train_row": int(index),
                "gid": int(targets["gid"][index]),
                "k": int(k_dev[position]),
                "g": float(g_dev[position]),
                "B_raw": float(raw["B"]["dev"][position]),
                "B_cal": float(cal["B"]["dev"][position]),
                "M_raw": float(raw["M"]["dev"][position]),
                "M_cal": float(cal["M"]["dev"][position]),
                "B_err_cal": float(arms["B"]["err_dev_cal"][position]),
                "M_err_cal": float(arms["M"]["err_dev_cal"][position]),
            }
        )
    per_graph_fit = []
    fit_idx = fold["fit_idx"]
    k_fit = k[fit_idx]
    for position, index in enumerate(fit_idx.tolist()):
        per_graph_fit.append(
            {
                "fit_position": position,
                "train_row": int(index),
                "gid": int(targets["gid"][index]),
                "k": int(k_fit[position]),
                "g": float(g_fit[position]),
                "B_raw": float(raw["B"]["fit"][position]),
                "B_cal": float(cal["B"]["fit"][position]),
                "M_raw": float(raw["M"]["fit"][position]),
                "M_cal": float(cal["M"]["fit"][position]),
            }
        )

    analysis = {
        "protocol_version": PROTOCOL_VERSION,
        "classification": classification,
        "mechanism_class": mechanism_class,
        "gate_conditions": gate_conditions,
        "gains_B_to_M": gains,
        "bias": bias,
        "arms": {arm: {key: value for key, value in arms[arm].items() if not key.startswith("err_")} for arm in ("B", "M")},
        "identity_checks": identity,
        "sensitivity_drop_worst": sens,
        "bootstrap_witnesses": witnesses,
        "mechanism": mech,
        "replay": replay,
        "n_dev": int(k_dev.size),
        "n_G0": int((k_dev == 0).sum()),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(out_dir / "analysis.json", analysis)
    write_json(out_dir / "gains.json", gains)
    write_json(out_dir / "gate.json", {"classification": classification, "mechanism_class": mechanism_class, "conditions": gate_conditions})
    write_json(out_dir / "bootstrap.json", {"gains": gains, "witnesses": witnesses})
    write_json(out_dir / "mechanism_health.json", mech)
    write_json(out_dir / "replay_checks.json", replay)
    import csv

    with open(out_dir / "main_table.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(main_rows[0]))
        writer.writeheader()
        writer.writerows(main_rows)
    with open(out_dir / "group_table.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(group_rows[0]))
        writer.writeheader()
        writer.writerows(group_rows)
    with open(out_dir / "per_graph_dev.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_graph[0]))
        writer.writeheader()
        writer.writerows(per_graph)
    with open(out_dir / "per_graph_fit.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_graph_fit[0]))
        writer.writeheader()
        writer.writerows(per_graph_fit)
    log(f"[analyze] classification={classification} mechanism={mechanism_class} replay_ok={replay['all_ok']}")
    log(f"[analyze] G0 cal gain={gains['G0_cal']['point']:+.6f} CI={gains['G0_cal']['ci95']}")
    log(f"[analyze] overall cal gain={gains['overall_cal']['point']:+.6f} CI={gains['overall_cal']['ci95']}")
    return analysis


# ---------------------------------------------------------------------------
# 11. manifest / budget
# ---------------------------------------------------------------------------


def make_manifest(*, out_dir: Path = RESULTS_DIR) -> dict[str, Any]:
    files = {}
    for path in sorted(out_dir.glob("*")):
        if path.is_file():
            files[path.name] = {
                "sha256": file_sha256(path),
                "bytes": int(path.stat().st_size),
            }
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "files": files,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(out_dir / "manifest.json", payload)
    return payload


def make_budget(
    *,
    out_dir: Path = RESULTS_DIR,
    round_start: str = "",
    round_end: str = "",
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "round_start": round_start,
        "round_end": round_end,
        "gpu_budget_limit_hours": 1.0,
    }
    b_meta = json.loads((out_dir / "B_meta.json").read_text())
    m_meta = json.loads((out_dir / "M_meta.json").read_text())
    payload["training_wall_seconds"] = {
        "B": b_meta["wall_clock_s"],
        "M": m_meta["wall_clock_s"],
    }
    payload["training_gpu_hours_upper_bound"] = (b_meta["wall_clock_s"] + m_meta["wall_clock_s"]) / 3600.0
    payload["steps"] = {"B": b_meta["steps_done"], "M": m_meta["steps_done"]}
    if extra:
        payload["extra"] = dict(extra)
    write_json(out_dir / "budget.json", payload)
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-a", action="store_true")
    parser.add_argument("--operator-checks", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--arm", choices=("B", "M"), default=None)
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--budget", action="store_true")
    parser.add_argument("--manifest", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args(argv)

    if args.phase_a:
        phase_a(out_dir=args.out)
    if args.operator_checks:
        operator_checks(out_dir=args.out, device=resolve_device(args.device))
    if args.smoke:
        if args.arm is None:
            raise SystemExit("--smoke requires --arm")
        run_smoke(arm=args.arm, device=resolve_device(args.device), out_dir=args.out, max_steps=args.max_steps or 3)
    if args.train:
        if args.arm is None:
            raise SystemExit("--train requires --arm")
        train_arm(
            args.arm,
            device=resolve_device(args.device),
            out_dir=args.out,
            epochs=args.epochs,
            max_steps=args.max_steps,
        )
    if args.analyze:
        analyze(out_dir=args.out, device=resolve_device(args.device))
    if args.budget:
        make_budget(out_dir=args.out)
    if args.manifest:
        make_manifest(out_dir=args.out)
    if not any((args.phase_a, args.operator_checks, args.smoke, args.train, args.analyze, args.budget, args.manifest)):
        parser.print_help()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
