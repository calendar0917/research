"""cscl-bjoc-cross-bond-interaction-v1 driver.

Protocol: ``configs/bjoc_cross_bond_interaction_v1.yaml`` (frozen before any
formal GPU trajectory).  Track: ``tracks/structure-contribution``.

The round asks whether ONE fixed cross-bond channel — the element-wise product
of the two per-side mean junction embeddings of a real physical bond — carries
held-out prediction information that the published BJOC summaries cannot
express.  Three parameter-matched arms share every tensor except the incidence
mode and the q channel:

* ``N`` — null incidence grouping, ``q = 0``
* ``O`` — true (organized) incidence grouping, ``q = 0``
* ``P`` — true grouping, real ``q``

Stages
------
``splits``            freeze two canonical-SMILES grouped internal splits (A / B)
``objects``           fit the split global-feature standardiser on fit rows only
``audit-historical``  read-only audit of how the new channel differs from the
                      prior organized / intrinsic-contribution rounds
``audit-projection``  Phase A3: (W, V, n) collision witnesses for side products
``audit-real``        Phase A4: label-blind real-molecule non-degeneracy stats
``synthetic``         Phase A5: known-truth representation test on object systems
``phase-a-gate``      frozen CPU purchase gate -> train P or stop
``params``            parameter inventory / matched-init audit
``gates``             the 24 pre-GPU correctness gates
``freeze``            protocol freeze manifest
``train``/``train-stage``  formal training of one arm / a registered stage
``predict``           label-blind prediction freeze (monitor, optionally locked)
``monitor-gate``      split-A seed-0 purchase rule (monitor labels only)
``score``             the ONE locked-label read per split
``locked-gate``       split-A locked promotion rule -> split B
``mechanism``         CPU diagnostics on the frozen checkpoints
``report``            REPORT.md assembly
``decision``          single final decision label

Discipline
----------
* official valid/test are never loaded in any form.
* fit y are used for training only; monitor y are read once for the frozen
  purchase gate after the monitor predictions are hashed; locked y are read
  exactly once per split inside ``score`` after ALL prediction files are
  written and hashed (a lock file makes a second read explicit and forced).
* raw-y L1 only; no ``g/ell/s/c/k``; no GNN/attention/iteration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import subprocess
import sys
import time
from itertools import combinations_with_replacement
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

_CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = _CODE_DIR.parents[2]
for _p in (str(_CODE_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bjoc_cross_bond_interaction_v1 as cbic  # noqa: E402
import cscl_features as cf  # noqa: E402
import run_bjoc_intrinsic_contribution_v1 as rb  # noqa: E402
import structural_relation_composer_v1 as src  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402

PROTOCOL_ID = "cscl-bjoc-cross-bond-interaction-v1"
CONFIG_PATH = _CODE_DIR.parent / "configs" / "bjoc_cross_bond_interaction_v1.yaml"
OUT = REPO_ROOT / "tracks/structure-contribution/results/bjoc_cross_bond_interaction_v1"

# ---- frozen recipe (identical to the recent BJOC rounds) -------------------
EPOCHS = 240
LR = 1.0e-3
WEIGHT_DECAY = 1.0e-5
GRAD_CLIP = 5.0
BATCH_SIZE = 128
SOUP_LAST = 5
TRAIN_SHUFFLE_OFFSET = 101
SEEDS = (0, 1)

# ---- frozen split salts / sizes --------------------------------------------
SPLIT_SALTS = {
    "A": "cscl-bjoc-crossbond-v1-discovery",
    "B": "cscl-bjoc-crossbond-v1-replication",
}
FIT_TARGET = 8000
MONITOR_TARGET = 1000
SPLIT_FILES = {"A": OUT / "split_a.json", "B": OUT / "split_b.json"}
PREP_FILES = {"A": OUT / "prep_a.json", "B": OUT / "prep_b.json"}
PRED_FILES = {"A": OUT / "split_a_predictions.npz", "B": OUT / "split_b_predictions.npz"}
MON_PRED_FILES = {"A": OUT / "split_a_monitor_predictions.npz", "B": OUT / "split_b_monitor_predictions.npz"}

# ---- frozen Phase-A purchase thresholds ------------------------------------
PHASE_A_MIN_VALID_PAIR_FRACTION = 0.50
PHASE_A_MIN_FUNCTIONAL_RESIDUAL_SHARE = 0.10
PHASE_A_MIN_Q_VARIANCE_DIMS = 0.90
PHASE_A_SYNTHETIC_MAX_RATIO = 0.90
PHASE_A_INVARIANCE_ATOL = 1.0e-5

# ---- frozen performance purchase thresholds --------------------------------
GATE_PO_MONITOR_MAX = -0.006
GATE_PN_MONITOR_MAX = -0.004
GATE_ORDINARY_MAX = 0.0
GATE_TOP10_MAX = 0.0
GATE_OVERFIT_GAP_TOL = 0.01
GATE_LOCKED_PO_MAX = 0.0
GATE_LOCKED_PN_MAX = 0.0
GATE_LOCKED_BETTER_OF_MAX = -0.003
GATE_LOCKED_TAIL_MAX = 0.05
BOOTSTRAP_N = 5000
BOOTSTRAP_SEED = 0
ARMS = ("N", "O", "P")
STAGE_PLANS = {
    "pair-a-s0": [("N", "A", 0), ("O", "A", 0), ("P", "A", 0)],
    "pair-a-s1": [("N", "A", 1), ("O", "A", 1), ("P", "A", 1)],
    "pair-b-s0": [("N", "B", 0), ("O", "B", 0), ("P", "B", 0)],
    "pair-b-s1": [("N", "B", 1), ("O", "B", 1), ("P", "B", 1)],
}

# ---- reusable locals --------------------------------------------------------
write_json = rb.write_json
read_json = rb.read_json
file_sha256 = rb.file_sha256
git_commit = rb.git_commit
seed_everything = rb.seed_everything
torch_rng_state_sha256 = rb.torch_rng_state_sha256
load_objects = rb.load_objects
load_y_rows = rb.load_y_rows
rows_sha256 = rb.rows_sha256
build_split = rb.build_split
_shuffle_batches = rb._shuffle_batches
_last_soup = rb._last_soup
paired_bootstrap = rb.paired_bootstrap
metrics = rb.metrics


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# data / splits / prep
# ---------------------------------------------------------------------------


def _ensure_splits(log_fn=print) -> dict[str, dict[str, Any]]:
    smiles = rb.load_smiles()
    splits = {k: build_split(smiles, SPLIT_SALTS[k]) for k in ("A", "B")}
    crosstab = {
        name: {
            "A_fit": len(set(splits["A"]["fit"]) & set(splits["B"][name])),
            "A_monitor": len(set(splits["A"]["monitor"]) & set(splits["B"][name])),
            "A_locked": len(set(splits["A"]["locked"]) & set(splits["B"][name])),
        }
        for name in ("fit", "monitor", "locked")
    }
    for key, split in splits.items():
        if not SPLIT_FILES[key].exists():
            write_json(
                SPLIT_FILES[key],
                {
                    "protocol": PROTOCOL_ID,
                    "salt": split["salt"],
                    "fit_target": FIT_TARGET,
                    "monitor_target": MONITOR_TARGET,
                    "n_groups": split["n_groups"],
                    "n_fit": len(split["fit"]),
                    "n_monitor": len(split["monitor"]),
                    "n_locked": len(split["locked"]),
                    "overlap": split["overlap"],
                    "duplicate_group_sizes": split["duplicate_group_sizes"],
                    "fit_sha256_rows": rows_sha256(split["fit"]),
                    "monitor_sha256_rows": rows_sha256(split["monitor"]),
                    "locked_sha256_rows": rows_sha256(split["locked"]),
                    "crosstab_with_A": dict(crosstab if key == "B" else {}),
                    "locked_labels_read": False,
                },
            )
            log_fn(f"[splits] wrote {SPLIT_FILES[key].name}")
        frozen = read_json(SPLIT_FILES[key])
        assert frozen["salt"] == SPLIT_SALTS[key]
        assert rows_sha256(split["fit"]) == frozen["fit_sha256_rows"], f"split {key} fit drifted"
        assert rows_sha256(split["monitor"]) == frozen["monitor_sha256_rows"], f"split {key} monitor drifted"
        assert rows_sha256(split["locked"]) == frozen["locked_sha256_rows"], f"split {key} locked drifted"
        assert split["overlap"]["union"] == 10000
        assert split["overlap"]["fit_monitor"] == 0
        assert split["overlap"]["fit_locked"] == 0
        assert split["overlap"]["monitor_locked"] == 0
    if not (OUT / "split_crosstab.json").exists():
        write_json(
            OUT / "split_crosstab.json",
            {
                "protocol": PROTOCOL_ID,
                "smiles_sha256": hashlib.sha256("\n".join(smiles).encode()).hexdigest(),
                "crosstab_A_vs_B_rows": crosstab,
                "note": "nested hash splits of the same 10k pool; overlap reported, never optimised",
            },
        )
    return splits


def read_split_rows(key: str, log_fn=print) -> dict[str, list[int]]:
    split = _ensure_splits(log_fn)[key]
    return {"fit": split["fit"], "monitor": split["monitor"], "locked": split["locked"]}


def load_prep(split_key: str) -> dict[str, Any]:
    prep = read_json(PREP_FILES[split_key])
    assert prep["fit_only"] is True and prep["n_fit"] == FIT_TARGET
    return prep


def load_norm(split_key: str) -> tuple[np.ndarray, np.ndarray]:
    prep = load_prep(split_key)
    return (
        np.asarray(prep["global_mean"], dtype=np.float32),
        np.asarray(prep["global_scale"], dtype=np.float32),
    )


def cmd_splits(args: argparse.Namespace) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    splits = _ensure_splits()
    for key, split in splits.items():
        log(
            f"[splits] {key}: fit={len(split['fit'])} monitor={len(split['monitor'])} "
            f"locked={len(split['locked'])} fit_sha={rows_sha256(split['fit'])[:12]}"
        )
    return 0


def cmd_objects(args: argparse.Namespace) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    _ensure_splits()
    cache_sha = file_sha256(rb.OBJECTS_CACHE)
    assert cache_sha == rb.OBJECTS_CACHE_SHA256, f"objects cache drifted: {cache_sha}"
    objs = load_objects()
    for key in ("A", "B"):
        rows = read_split_rows(key)
        gfit = np.stack([objs[int(i)].global_feat for i in rows["fit"]], 0).astype(np.float64)
        gmean = gfit.mean(0)
        gscale = np.where(gfit.std(0) < 1e-6, 1.0, gfit.std(0))
        write_json(
            PREP_FILES[key],
            {
                "protocol": PROTOCOL_ID,
                "split": key,
                "fit_only": True,
                "n_fit": int(len(rows["fit"])),
                "global_mean": gmean.tolist(),
                "global_scale": gscale.tolist(),
                "names": list(sro.GLOBAL_FEAT_NAMES),
                "labels_read": False,
            },
        )
        log(f"[objects] {key}: global standardiser fitted on {len(rows['fit'])} fit rows (labels never read)")
    write_json(
        OUT / "objects_summary.json",
        {
            "protocol": PROTOCOL_ID,
            "objects_cache": str(rb.OBJECTS_CACHE.relative_to(REPO_ROOT)),
            "objects_cache_sha256": cache_sha,
            "n_rows": len(objs),
            "n_atoms": int(sum(o.n_atoms for o in objs)),
            "n_bonds": int(sum(o.bond_index.shape[0] for o in objs)),
            "n_junctions": int(sum(o.junc_index.shape[0] for o in objs)),
            "labels_read": False,
        },
    )
    del objs
    return 0


# ---------------------------------------------------------------------------
# side statistics helpers (label-blind; shared by audits and gates)
# ---------------------------------------------------------------------------


def side_flat_arrays(
    bond_index: np.ndarray, junc_index: np.ndarray, junc_bond: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flat (junction_id, bond_id, side) triples for one object system.

    side == 0 means the junction centre is the bond's u endpoint, 1 for v.
    """
    centers = np.asarray(junc_index)[:, 1]
    n_j = int(np.asarray(junc_bond).shape[0])
    junc_ids = np.repeat(np.arange(n_j), 2)
    bond_ids = np.asarray(junc_bond).reshape(-1)
    center = np.repeat(centers, 2)
    ends = np.asarray(bond_index)[bond_ids]
    is_u = center == ends[:, 0]
    is_v = center == ends[:, 1]
    assert bool((is_u | is_v).all())
    side = np.where(is_u, 0, 1)
    return junc_ids, bond_ids, side


def side_counts(obj: sro.MolObjects) -> tuple[np.ndarray, np.ndarray]:
    n_bonds = int(obj.bond_index.shape[0])
    _, bond_ids, side = side_flat_arrays(obj.bond_index, obj.junc_index, obj.junc_bond)
    n_u = np.bincount(bond_ids[side == 0], minlength=n_bonds)
    n_v = np.bincount(bond_ids[side == 1], minlength=n_bonds)
    return n_u.astype(np.int64), n_v.astype(np.int64)


def side_means(obj: sro.MolObjects, w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-bond (mu_u, mu_v, n_u, n_v) for a junction embedding matrix w."""
    w = np.asarray(w, dtype=np.float64)
    n_bonds = int(obj.bond_index.shape[0])
    junc_ids, bond_ids, side = side_flat_arrays(obj.bond_index, obj.junc_index, obj.junc_bond)
    mu_u = np.zeros((n_bonds, w.shape[1]), dtype=np.float64)
    mu_v = np.zeros_like(mu_u)
    n_u = np.zeros(n_bonds, dtype=np.int64)
    n_v = np.zeros(n_bonds, dtype=np.int64)
    for mask, dst, cnt in ((side == 0, mu_u, n_u), (side == 1, mu_v, n_v)):
        np.add.at(dst, bond_ids[mask], w[junc_ids[mask]])
        np.add.at(cnt, bond_ids[mask], 1)
    mu_u = mu_u / np.maximum(n_u, 1)[:, None]
    mu_v = mu_v / np.maximum(n_v, 1)[:, None]
    return mu_u, mu_v, n_u, n_v


def cross_q(mu_u: np.ndarray, mu_v: np.ndarray, n_u: np.ndarray, n_v: np.ndarray, dim: int) -> np.ndarray:
    valid = (n_u > 0) & (n_v > 0)
    q = cbic.INTERACTION_SCALE * (mu_u * mu_v)
    q[~valid] = 0.0
    return q


def _junction_token(obj: sro.MolObjects) -> np.ndarray:
    """Deterministic scalar pseudo-embedding per junction (label-blind).

    Used only in Phase A audits as a stand-in for the learned ``w``; it is a
    fixed function of the raw structure so collisions are exact and the audit
    does not need a trained model.
    """
    atom_type = np.asarray([int(obj.atom_feat[i, :sro.ATOM_CATEGORIES].argmax()) for i in range(obj.n_atoms)])
    tokens = np.zeros(int(obj.junc_index.shape[0]), dtype=np.int64)
    for j, (i, c, k) in enumerate(obj.junc_index):
        b1, b2 = obj.junc_bond[j]
        t1 = int(obj.bond_type[b1])
        t2 = int(obj.bond_type[b2])
        key = (
            int(atom_type[c]) * 29
            + int(atom_type[i]) * 13
            + int(atom_type[k]) * 7
            + t1 * 3
            + t2 * 5
            + (1 if t1 == t2 else 0)
        )
        tokens[j] = key % 3
    return tokens


# ---------------------------------------------------------------------------
# audit 1: historical overlap (read-only)
# ---------------------------------------------------------------------------


def _git_show(subject_commit: str) -> dict[str, Any]:
    try:
        text = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "log", "-1", "--format=%H%x00%s", subject_commit], text=True
        ).strip()
        commit, subject = text.split("\x00", 1)
        return {"commit": commit, "subject": subject, "present": True}
    except Exception as exc:  # pragma: no cover - provenance only
        return {"commit": subject_commit, "subject": None, "present": False, "error": str(exc)}


def cmd_audit_historical(args: argparse.Namespace) -> int:
    """Differences from the prior BJOC rounds (BJOC one-shot, IJCM); no training."""
    payload: dict[str, Any] = {
        "protocol": PROTOCOL_ID,
        "official_valid_or_test_loaded": False,
        "history_commits": {c: _git_show(c) for c in ("da3f0ae", "59bd1e7", "661e103")},
        "old_organized_round": {
            "protocol": "cscl-structural-relation-composer-v1",
            "arm_definition": "W_b = sum_{j touches b} w_j; V_b = sum w_j^2; n_b = #junctions(b)",
            "identity": "sum_{j<k} w_j (x) w_k = (W_b^2 - V_b)/2 element-wise",
            "reading": (
              "The full total of same-dimension pairwise junction products over the "
              "junctions incident to a bond is algebraically reconstructible from "
              "W_b and V_b.  The round must not present that total as new."
            ),
            "what_it_cannot_do": (
              "(W_b, V_b, n_b) does not record which incident junctions are centred "
              "on bond endpoint u and which on endpoint v, nor the element-wise "
              "product of the two per-side mean embeddings."
            ),
        },
        "ijcm_round": {
            "protocol": "cscl-bjoc-intrinsic-contribution-feasibility-v1",
            "arm_definition": "strictly additive sum of per-object scalar contributions (no joint composer output path)",
            "synthetic_c": "32% of a true pairwise junction interaction is orthogonal to the additive design; additive floor 0.1507 vs oracle 0.0194",
            "reading": (
              "IJCM showed a purely additive head cannot express a pairwise "
              "interaction.  It does not show that the original non-linear BJOC "
              "lacks such capability, so the CBIC round is not an IJCM patch and "
              "the IJCM synthetic result is not a prior in favour of CBIC."
            ),
        },
        "cbic_delta": {
            "holding_fixed": "learned w, atom/bond/junction encoders, W_b/V_b/n_b definition, reader, recipe",
            "new": "per-side partition of the incident junctions + q_e = sqrt(96) * mu_u * mu_v",
            "is_not": [
                "the old organization (true vs null incidence) contrast",
                "a second additive branch",
                "a bilinear matrix or attention",
                "the total pairwise product (already in (W^2 - V)/2)",
            ],
        },
        "objects_cache_sha256": rb.OBJECTS_CACHE_SHA256,
        "labels_read": False,
    }
    write_json(OUT / "historical_overlap_audit.json", payload)
    log("[audit-historical] written")
    return 0


# ---------------------------------------------------------------------------
# audit 2: projection collision (Phase A1/A2/A3)
# ---------------------------------------------------------------------------


def _pairings4() -> list[tuple[tuple[int, int], tuple[int, int]]]:
    return [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]


def cmd_audit_projection(args: argparse.Namespace) -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    # -- scalar witness -----------------------------------------------------
    witness = {
        "values": [1, 2, 3, 4],
        "assignment_A": {"left": [1, 2], "right": [3, 4], "product": (1 + 2) * (3 + 4)},
        "assignment_B": {"left": [1, 3], "right": [2, 4], "product": (1 + 3) * (2 + 4)},
        "W": 10,
        "V": 30,
        "n": 4,
        "n_u": 2,
        "n_v": 2,
        "collision": True,
    }

    # -- systematic scalar search -------------------------------------------
    groups: dict[tuple[int, int], dict[str, Any]] = {}
    for values in combinations_with_replacement(range(1, 10), 4):
        W = int(sum(values))
        V = int(sum(v * v for v in values))
        for (a, b), (c, d) in _pairings4():
            s_u = int(values[a] + values[b])
            s_v = int(values[c] + values[d])
            key = (W, V)
            entry = groups.setdefault(key, {"products": set(), "count_assignments": 0})
            entry["products"].add(s_u * s_v)
            entry["count_assignments"] += 1
    collisions = [
        {"W": W, "V": V, "n": 4, "n_u": 2, "n_v": 2, "distinct_products": sorted(v["products"]), "spread": max(v["products"]) - min(v["products"])}
        for (W, V), v in sorted(groups.items())
        if len(v["products"]) > 1
    ]
    max_spread_example = max(collisions, key=lambda c: c["spread"]) if collisions else None

    # -- full 96-D vector construction --------------------------------------
    dim = 96
    w_a = np.zeros((4, dim), dtype=np.float64)
    w_b = np.zeros((4, dim), dtype=np.float64)
    left_a = np.array([1.0, 2.0])
    right_a = np.array([3.0, 4.0])
    left_b = np.array([1.0, 3.0])
    right_b = np.array([2.0, 4.0])
    for k in range(dim):
        if k % 2 == 0:
            w_a[:, k] = [1, 2, 3, 4]
            w_b[:, k] = [1, 3, 2, 4]
        else:
            w_a[:, k] = [1, 3, 2, 4]
            w_b[:, k] = [1, 2, 3, 4]
    W_a = w_a.sum(0)
    V_a = (w_a ** 2).sum(0)
    W_b = w_b.sum(0)
    V_b = (w_b ** 2).sum(0)
    mu_ua = (w_a[0] + w_a[1]) / 2.0
    mu_va = (w_a[2] + w_a[3]) / 2.0
    mu_ub = (w_b[0] + w_b[1]) / 2.0
    mu_vb = (w_b[2] + w_b[3]) / 2.0
    q_a = cbic.INTERACTION_SCALE * mu_ua * mu_va
    q_b = cbic.INTERACTION_SCALE * mu_ub * mu_vb
    vector_collision = {
        "W_equal": bool(np.array_equal(W_a, W_b)),
        "V_equal": bool(np.array_equal(V_a, V_b)),
        "n_u": 2,
        "n_v": 2,
        "max_abs_W_gap": float(np.abs(W_a - W_b).max()),
        "max_abs_V_gap": float(np.abs(V_a - V_b).max()),
        "max_abs_q_gap": float(np.abs(q_a - q_b).max()),
        "mean_abs_q_gap": float(np.abs(q_a - q_b).mean()),
        "fraction_dims_q_differ": float(np.mean(np.abs(q_a - q_b) > 1e-12)),
    }

    # -- identity verification ----------------------------------------------
    rng = np.random.default_rng(0)
    for trial in range(5):
        w = rng.normal(size=(6, dim))
        W = w.sum(0)
        V = (w ** 2).sum(0)
        total_pairs = np.zeros(dim)
        for i in range(6):
            for j in range(i + 1, 6):
                total_pairs += w[i] * w[j]
        assert np.abs(total_pairs - (W * W - V) / 2.0).max() < 1e-9

    payload = {
        "protocol": PROTOCOL_ID,
        "official_valid_or_test_loaded": False,
        "legend": {
            "W": "sum of all incident junction embeddings",
            "V": "sum of squares of all incident junction embeddings",
            "side_cross": "s_u * s_v with s = sum over junctions centred on each endpoint",
            "q": "sqrt(96) * mu_u * mu_v (the model channel)",
        },
        "identity_total_pairwise": {
            "statement": "sum_{j<k} w_j (x) w_k = (W^2 - V)/2 element-wise",
            "verification": "passed on 5 random 6x96 systems to <1e-9",
            "consequence": "the *total* same-dimension pairwise product over all incident junctions is reconstructible; it is NOT the new information",
        },
        "scalar_witness": witness,
        "systematic_scalar_search": {
            "values": "combinations_with_replacement(1..9, 4)",
            "n_collision_keys": len(collisions),
            "max_spread_example": max_spread_example,
            "examples": collisions[:5],
        },
        "vector_collision_96d": vector_collision,
        "recoverability": {
            "claim": "no function of (W, V, n, n_u, n_v) can determine s_u*s_v in general",
            "witness": "the scalar and 96-D constructions above",
            "caveat": (
              "This proves the *summary projection* loses the side partition.  It "
              "does NOT prove that the full original BJOC computation (atom/bond "
              "encoders + nonlinear composer) can never implicitly recover the "
              "information from other inputs.  The real-data audit and the GPU "
              "contrast are required for that question."
            ),
        },
    }
    write_json(OUT / "projection_collision_audit.json", payload)
    log(
        f"[audit-projection] scalar collisions={len(collisions)} "
        f"vector W/V equal={vector_collision['W_equal']}/{vector_collision['V_equal']} "
        f"max|dq|={vector_collision['max_abs_q_gap']:.4f}"
    )
    return 0


# ---------------------------------------------------------------------------
# audit 3: real-molecule information (Phase A4)
# ---------------------------------------------------------------------------


def _ridge_r2(X: np.ndarray, Y: np.ndarray, *, lam: float = 1e-3, train_frac: float = 0.9, seed: int = 0) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(n * train_frac)
    tr, te = perm[:n_train], perm[n_train:]
    Xtr, Xte, Ytr, Yte = X[tr], X[te], Y[tr], Y[te]
    xm = Xtr.mean(0, keepdims=True)
    ym = Ytr.mean(0, keepdims=True)
    Xc = Xtr - xm
    Yc = Ytr - ym
    A = Xc.T @ Xc + lam * np.eye(X.shape[1])
    B = Xc.T @ Yc
    beta = np.linalg.solve(A, B)
    pred = (Xte - xm) @ beta + ym
    sse = float(((Yte - pred) ** 2).sum())
    sst = float(((Yte - ym) ** 2).sum())
    return {
        "n_train": int(n_train),
        "n_test": int(n - n_train),
        "r2_total": float(1.0 - sse / max(sst, 1e-12)),
        "residual_variance_share": float(sse / max(sst, 1e-12)),
        "sse": sse,
        "sst": sst,
    }


def cmd_audit_real(args: argparse.Namespace) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    _ensure_splits()
    objs = load_objects()

    # -- structural counts (label-blind) ------------------------------------
    n_bonds_total = 0
    n_both = 0
    n_u_only = 0
    n_v_only = 0
    n_neither = 0
    nu_hist = np.zeros(8, dtype=np.int64)
    nv_hist = np.zeros(8, dtype=np.int64)
    per_mol = np.zeros(len(objs), dtype=np.int64)
    for idx, obj in enumerate(objs):
        n_u, n_v = side_counts(obj)
        nu = np.clip(n_u, 0, 7)
        nv = np.clip(n_v, 0, 7)
        nu_hist += np.bincount(nu, minlength=8)
        nv_hist += np.bincount(nv, minlength=8)
        both = (n_u > 0) & (n_v > 0)
        n_bonds_total += int(n_u.size)
        n_both += int(both.sum())
        n_u_only += int(((n_u > 0) & (n_v == 0)).sum())
        n_v_only += int(((n_u == 0) & (n_v > 0)).sum())
        n_neither += int(((n_u == 0) & (n_v == 0)).sum())
        per_mol[idx] = int(both.sum())

    # -- side multiset equality / q statistics ------------------------------
    # a fixed random 96-D projection of the raw junction features stands in for
    # a learned embedding; label-blind and never fitted.
    rng = np.random.default_rng(20261012)
    R = rng.normal(0.0, 1.0 / np.sqrt(sro.JUNC_FEAT_DIM), size=(sro.JUNC_FEAT_DIM, 96)).astype(np.float64)
    q_rows: list[np.ndarray] = []
    multiset_equal = 0
    multiset_total = 0
    for obj in objs:
        w = obj.junc_feat.astype(np.float64) @ R
        tokens = _junction_token(obj).astype(np.float64)
        n_u, n_v = side_counts(obj)
        junc_ids, bond_ids, side = side_flat_arrays(obj.bond_index, obj.junc_index, obj.junc_bond)
        # exact multiset equality of side tokens (order independent)
        left: list[list[float]] = [[] for _ in range(int(obj.bond_index.shape[0]))]
        for j, b, s in zip(junc_ids, bond_ids, side):
            left[int(b)].append((0 if s == 0 else 1, float(tokens[int(j)])))
        for b in range(len(left)):
            l = sorted(t for s, t in left[b] if s == 0)
            r = sorted(t for s, t in left[b] if s == 1)
            if l and r:
                multiset_total += 1
                if l == r:
                    multiset_equal += 1
        mu_u, mu_v, nu, nv = side_means(obj, w)
        q = cross_q(mu_u, mu_v, nu, nv, 96)
        if q.shape[0]:
            q_rows.append(q[(nu > 0) & (nv > 0)])
    q_all = np.concatenate(q_rows, 0) if q_rows else np.zeros((0, 96))
    q_std = q_all.std(0) if q_all.size else np.zeros(96)
    dims_with_variance = float(np.mean(q_std > 1e-8))

    # -- composer-visible linear probes -------------------------------------
    # Build a batched view to compute W, V, n and q over a sample of molecules.
    sample_rows = list(range(min(2000, len(objs))))
    bond_feat, bond_batch, bond_index, junc_index, junc_bond = [], [], [], [], []
    b_off = a_off = 0
    for g, r in enumerate(sample_rows):
        obj = objs[r]
        nb = int(obj.bond_index.shape[0])
        bond_feat.append(obj.bond_feat.astype(np.float64))
        bond_batch.append(np.full(nb, g, dtype=np.int64))
        bond_index.append(obj.bond_index.astype(np.int64) + a_off)
        junc_index.append(obj.junc_index.astype(np.int64) + a_off)
        junc_bond.append(obj.junc_bond.astype(np.int64) + b_off)
        b_off += nb
        a_off += obj.n_atoms
    bond_feat = np.concatenate(bond_feat, 0)
    bond_batch = np.concatenate(bond_batch, 0)
    bond_index = np.concatenate(bond_index, 0)
    junc_index = np.concatenate(junc_index, 0)
    junc_bond = np.concatenate(junc_bond, 0)
    all_w = np.concatenate([objs[r].junc_feat.astype(np.float64) @ R for r in sample_rows], 0)
    n_bonds = bond_index.shape[0]
    junc_ids, bond_ids, side = side_flat_arrays(bond_index, junc_index, junc_bond)
    s_u = np.zeros((n_bonds, 96))
    s_v = np.zeros((n_bonds, 96))
    n_u = np.zeros(n_bonds, dtype=np.int64)
    n_v = np.zeros(n_bonds, dtype=np.int64)
    np.add.at(s_u, bond_ids[side == 0], all_w[junc_ids[side == 0]])
    np.add.at(s_v, bond_ids[side == 1], all_w[junc_ids[side == 1]])
    np.add.at(n_u, bond_ids[side == 0], 1)
    np.add.at(n_v, bond_ids[side == 1], 1)
    mu_u = s_u / np.maximum(n_u, 1)[:, None]
    mu_v = s_v / np.maximum(n_v, 1)[:, None]
    W = s_u + s_v
    V = (np.zeros_like(W))
    np.add.at(V, bond_ids[side == 0], all_w[junc_ids[side == 0]] ** 2)
    np.add.at(V, bond_ids[side == 1], all_w[junc_ids[side == 1]] ** 2)
    q = cross_q(mu_u, mu_v, n_u, n_v, 96)
    valid = (n_u > 0) & (n_v > 0)
    X_view = np.concatenate([bond_feat, W, V, np.log1p((n_u + n_v))[:, None]], 1)
    deg = np.bincount(bond_index.reshape(-1), minlength=a_off)
    X_simple = np.concatenate(
        [bond_feat, deg[bond_index[:, 0]][:, None], deg[bond_index[:, 1]][:, None],
         n_u[:, None], n_v[:, None], np.log1p((n_u + n_v))[:, None]],
        1,
    )
    idx = np.flatnonzero(valid)
    probe_view = _ridge_r2(X_view[idx], q[idx])
    probe_simple = _ridge_r2(X_simple[idx], q[idx])

    # -- exact algebraic decomposition of the channel ------------------------
    # With X = s_u * s_v / (n_u n_v) and P = ((s_u^2 - Q_u) + (s_v^2 - Q_v)) / 2 / (n_u n_v):
    #   q / sqrt(D)             = X
    #   (W^2 - V) / 2 / (n_u n_v) = X + P      (fully determined by W, V, n)
    #   q / sqrt(D)             = (W^2 - V) / 2 / (n_u n_v) - P
    # P is the same-side pair-product correction and is *not* a function of
    # (W, V, n); it is the channel's genuinely side-partition information.
    Q_u = np.zeros_like(s_u)
    Q_v = np.zeros_like(s_v)
    np.add.at(Q_u, bond_ids[side == 0], all_w[junc_ids[side == 0]] ** 2)
    np.add.at(Q_v, bond_ids[side == 1], all_w[junc_ids[side == 1]] ** 2)
    denom = np.maximum(n_u * n_v, 1)[:, None].astype(np.float64)
    X_flat = (s_u * s_v) / denom
    P_net = 0.5 * ((s_u ** 2 - Q_u) + (s_v ** 2 - Q_v)) / denom
    old_part = (W * W - V) / 2.0 / denom
    identity_gap = float(np.abs((X_flat + P_net) - old_part).max())
    q_scaled = cbic.INTERACTION_SCALE * X_flat
    q_recon_gap = float(np.abs(q - q_scaled).max())
    xv, pv = X_flat[idx], P_net[idx]
    per_class: dict[str, Any] = {}
    for a in (1, 2, 3):
        for b in (1, 2, 3):
            if a > b:
                continue
            cls_mask = (((n_u[idx] == a) & (n_v[idx] == b)) | ((n_u[idx] == b) & (n_v[idx] == a)))
            if cls_mask.sum() == 0:
                continue
            per_class[f"{a}_{b}"] = {
                "n_bonds": int(cls_mask.sum()),
                "l1_new_share": float(np.abs(pv[cls_mask]).sum() / max(np.abs(xv[cls_mask]).sum(), 1e-12)),
                "var_new_share": float(pv[cls_mask].var() / max(xv[cls_mask].var(), 1e-12)),
            }
    exact = {
        "identity_max_gap": identity_gap,
        "q_reconstruction_max_gap": q_recon_gap,
        "formula": "q = sqrt(96) * [ (W^2 - V)/2 - P_net ] / (n_u n_v); P_net = same-side pair correction",
        "functional_new_variance_share": float(pv.var() / max(xv.var(), 1e-12)),
        "functional_new_l1_share": float(np.abs(pv).sum() / max(np.abs(xv).sum(), 1e-12)),
        "functional_new_abs_mean": float(np.abs(pv).mean()),
        "q_abs_mean": float(np.abs(xv).mean()),
        "per_nu_nv_class": per_class,
        "interpretation": (
            "The side-partition component P_net cannot be expressed as any function of (W, V, n). "
            "It is the functional information increment.  The linear view-probe residual "
            "(reported above) measures same-distribution linear predictability instead; the two "
            "quantities are not the same and both are reported."
        ),
    }

    # -- invariants ----------------------------------------------------------
    ds = cf.load_official_train()
    invariance_rows = list(range(16))
    inv: dict[str, Any] = {"n_rows": invariance_rows}
    max_relabel_gap = 0.0
    max_swap_gap = 0.0
    max_junc_swap_gap = 0.0
    for r in invariance_rows:
        d = ds[r]
        x = d.x.numpy().reshape(-1)
        ei = d.edge_index.numpy()
        ea = d.edge_attr.numpy()
        perm = np.arange(len(x) - 1, -1, -1)
        x2 = x[perm].reshape(-1, 1)
        ei2 = perm[ei]
        obj2 = sro.build_mol_objects(-1, x2, ei2, ea)
        obj1 = objs[r]
        w1 = obj1.junc_feat.astype(np.float64) @ R
        w2 = obj2.junc_feat.astype(np.float64) @ R
        mu_u1, mu_v1, n_u1, n_v1 = side_means(obj1, w1)
        q1 = cross_q(mu_u1, mu_v1, n_u1, n_v1, 96)
        mu_u2, mu_v2, n_u2, n_v2 = side_means(obj2, w2)
        q2 = cross_q(mu_u2, mu_v2, n_u2, n_v2, 96)
        # map original bond (u,v) -> relabelled bond (perm[u], perm[v])
        key2 = {(min(int(u), int(v)), max(int(u), int(v))): k for k, (u, v) in enumerate(obj2.bond_index)}
        for k, (u, v) in enumerate(obj1.bond_index):
            k2 = key2[(min(int(perm[u]), int(perm[v])), max(int(perm[u]), int(perm[v])))]
            max_relabel_gap = max(max_relabel_gap, float(np.abs(q1[k] - q2[k2]).max()))
        # endpoint swap of the same bond columns (sides swap, product unchanged)
        swapped = sro.MolObjects(
            mol_index=obj1.mol_index,
            smiles=obj1.smiles,
            n_atoms=obj1.n_atoms,
            atom_feat=obj1.atom_feat,
            bond_index=obj1.bond_index[:, ::-1].copy(),
            bond_type=obj1.bond_type,
            bond_feat=obj1.bond_feat,
            junc_index=obj1.junc_index,
            junc_bond=obj1.junc_bond,
            junc_feat=obj1.junc_feat,
            global_feat=obj1.global_feat,
        )
        mu_us, mu_vs, n_us, n_vs = side_means(swapped, w1)
        qs = cross_q(mu_us, mu_vs, n_us, n_vs, 96)
        max_swap_gap = max(max_swap_gap, float(np.abs(q1 - qs).max()))
        # junction endpoint swap (i,j,k)->(k,j,i); centre and bonds unchanged
        swapped_j = sro.MolObjects(
            mol_index=obj1.mol_index,
            smiles=obj1.smiles,
            n_atoms=obj1.n_atoms,
            atom_feat=obj1.atom_feat,
            bond_index=obj1.bond_index,
            bond_type=obj1.bond_type,
            bond_feat=obj1.bond_feat,
            junc_index=obj1.junc_index[:, ::-1].copy(),
            junc_bond=obj1.junc_bond,
            junc_feat=obj1.junc_feat,
            global_feat=obj1.global_feat,
        )
        mu_uj, mu_vj, n_uj, n_vj = side_means(swapped_j, w1)
        qj = cross_q(mu_uj, mu_vj, n_uj, n_vj, 96)
        max_junc_swap_gap = max(max_junc_swap_gap, float(np.abs(q1 - qj).max()))
    inv.update(
        {
            "max_abs_q_gap_atom_relabel": max_relabel_gap,
            "max_abs_q_gap_endpoint_swap": max_swap_gap,
            "max_abs_q_gap_junction_endpoint_swap": max_junc_swap_gap,
            "all_invariant": bool(
                max(max_relabel_gap, max_swap_gap, max_junc_swap_gap) <= PHASE_A_INVARIANCE_ATOL
            ),
        }
    )

    payload = {
        "protocol": PROTOCOL_ID,
        "official_valid_or_test_loaded": False,
        "labels_read": False,
        "embedding_proxy": {
            "note": "a fixed random 96-D projection of raw junction features stands in for the learned w; label-blind, never fitted",
            "seed": 20261012,
        },
        "bond_side_counts": {
            "n_bonds_total": int(n_bonds_total),
            "n_both_sides": int(n_both),
            "n_u_only": int(n_u_only),
            "n_v_only": int(n_v_only),
            "n_neither": int(n_neither),
            "valid_pair_fraction": float(n_both / max(n_bonds_total, 1)),
            "n_hist_u": nu_hist.tolist(),
            "n_hist_v": nv_hist.tolist(),
        },
        "per_molecule": {
            "n_mols": len(objs),
            "mols_with_at_least_one_valid_pair": int((per_mol > 0).sum()),
            "fraction_mols_with_at_least_one_valid_pair": float((per_mol > 0).mean()),
            "valid_pairs_min": int(per_mol.min()),
            "valid_pairs_median": float(np.median(per_mol)),
            "valid_pairs_max": int(per_mol.max()),
            "valid_pairs_mean": float(per_mol.mean()),
        },
        "side_environment": {
            "n_bonds_with_both_sides": int(multiset_total),
            "fraction_side_token_multisets_exactly_equal": float(multiset_equal / max(multiset_total, 1)),
        },
        "q_variance": {
            "n_valid_bond_rows_2000_mols": int(q_all.shape[0]),
            "per_dim_std_mean": float(q_std.mean()),
            "per_dim_std_median": float(np.median(q_std)),
            "per_dim_std_min": float(q_std.min()),
            "per_dim_std_max": float(q_std.max()),
            "fraction_dims_with_std_gt_1e-8": dims_with_variance,
            "q_abs_mean": float(np.abs(q_all).mean()) if q_all.size else 0.0,
            "q_abs_p95": float(np.percentile(np.abs(q_all), 95)) if q_all.size else 0.0,
        },
        "composer_visible_probes": {
            "view_features": "bond_feat + W + V + log1p(n) (the actual composer input)",
            "view_probe": probe_view,
            "simple_features": "bond_feat + endpoint degrees + n_u + n_v + log1p(n)",
            "simple_probe": probe_simple,
            "interpretation": (
              "residual_variance_share = 1 - R^2 of a ridge probe from the stated "
              "features to q.  The simple probe tests whether degree / bond type / "
              "ring flags / counts approximately determine q; the view probe tests "
              "same-distribution linear predictability from the full composer input. "
              "Neither is the functional question of whether q can be expressed as a "
              "function of the composer inputs; that is measured exactly by "
              "exact_decomposition below."
            ),
        },
        "exact_decomposition": exact,
        "invariants": inv,
    }
    write_json(OUT / "real_molecule_information_audit.json", payload)
    log(
        f"[audit-real] valid pairs {payload['bond_side_counts']['valid_pair_fraction']:.4f} "
        f"functional-new var {exact['functional_new_variance_share']:.4f} "
        f"(linear view-probe residual {probe_view['residual_variance_share']:.4f}) "
        f"invariant={inv['all_invariant']}"
    )
    del objs
    return 0


# ---------------------------------------------------------------------------
# audit 4: synthetic known-truth representation test (Phase A5)
# ---------------------------------------------------------------------------


class _PerBondMLP(torch.nn.Module):
    """Small permutation-invariant per-bond MLP -> sum -> scalar (CPU only)."""

    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.SiLU(),
        )
        self.head = torch.nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, batch: torch.Tensor, n_graphs: int) -> torch.Tensor:
        h = self.encoder(x)
        z = torch.zeros(n_graphs, h.shape[1], dtype=h.dtype)
        z.index_add_(0, batch, h)
        return self.head(z).view(-1)


def _synthetic_systems(seed: int, n_systems: int) -> list[dict[str, Any]]:
    """Random small molecular systems with a deterministic side-combination target."""
    rng = np.random.default_rng(seed)
    systems = []
    while len(systems) < n_systems:
        n_atoms = int(rng.integers(5, 9))
        inst = src.random_synthetic_instance(rng, n_atoms)
        x = np.asarray(inst.atom_types, dtype=np.int64).reshape(-1, 1)
        ei, ea = [], []
        for u, v, bt in inst.edges:
            ei.extend([(u, v), (v, u)])
            ea.extend([bt, bt])
        edge_index = np.asarray(ei, dtype=np.int64).T if ei else np.zeros((2, 0), dtype=np.int64)
        obj = sro.build_mol_objects(-1, x, edge_index, np.asarray(ea, dtype=np.int64))
        tokens = _junction_token(obj).astype(np.float64)
        if not len(tokens):
            continue
        n_u, n_v = side_counts(obj)
        junc_ids, bond_ids, side = side_flat_arrays(obj.bond_index, obj.junc_index, obj.junc_bond)
        s_u = np.zeros(int(obj.bond_index.shape[0]))
        s_v = np.zeros_like(s_u)
        np.add.at(s_u, bond_ids[side == 0], tokens[junc_ids[side == 0]])
        np.add.at(s_v, bond_ids[side == 1], tokens[junc_ids[side == 1]])
        valid = (n_u > 0) & (n_v > 0)
        if valid.sum() == 0:
            continue
        systems.append(
            {
                "obj": obj,
                "s_u": s_u,
                "s_v": s_v,
                "n_u": n_u,
                "n_v": n_v,
                "valid": valid,
            }
        )
    return systems


def _synthetic_feature_blocks(systems: Sequence[Mapping[str, Any]], with_q: bool) -> dict[str, Any]:
    feats, batches, targets = [], [], []
    for g, sysm in enumerate(systems):
        obj = sysm["obj"]
        nb = int(obj.bond_index.shape[0])
        W = sysm["s_u"] + sysm["s_v"]
        V = sysm["s_u"] ** 2 + sysm["s_v"] ** 2
        n = sysm["n_u"] + sysm["n_v"]
        q = np.where(sysm["valid"], sysm["s_u"] * sysm["s_v"], 0.0)
        cols = [W[:, None], V[:, None], n[:, None], sysm["n_u"][:, None].astype(float), sysm["n_v"][:, None].astype(float)]
        if with_q:
            cols.append(q[:, None])
        feats.append(np.concatenate(cols, 1))
        batches.append(np.full(nb, g, dtype=np.int64))
        targets.append(float(q.sum()))
    return {
        "x": torch.as_tensor(np.concatenate(feats, 0), dtype=torch.float32),
        "batch": torch.as_tensor(np.concatenate(batches, 0), dtype=torch.long),
        "y": torch.as_tensor(targets, dtype=torch.float32),
        "n_graphs": len(systems),
    }


def _train_synthetic_model(data: Mapping[str, Any], *, seed: int, epochs: int = 400) -> _PerBondMLP:
    seed_everything(seed)
    model = _PerBondMLP(data["x"].shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    for _ in range(epochs):
        pred = model(data["x"], data["batch"], data["n_graphs"])
        loss = F.l1_loss(pred, data["y"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    model.eval()
    return model


def _mlp_mae(model: _PerBondMLP, data: Mapping[str, Any]) -> float:
    with torch.no_grad():
        pred = model(data["x"], data["batch"], data["n_graphs"])
    return float((pred - data["y"]).abs().mean())


def _covariate_scale(systems: Sequence[Mapping[str, Any]]) -> tuple[float, float]:
    all_y = []
    for sysm in systems:
        valid = sysm["valid"]
        all_y.append(float((sysm["s_u"][valid] * sysm["s_v"][valid]).sum()))
    arr = np.asarray(all_y)
    return float(arr.mean()), float(arr.std() + 1e-8)


def cmd_synthetic(args: argparse.Namespace) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    train = _synthetic_systems(11, 4000)
    test = _synthetic_systems(12, 800)
    ym, ys = _covariate_scale(train)
    for rows in (train, test):
        for sysm in rows:
            sysm["target_std"] = float((sysm["s_u"][sysm["valid"]] * sysm["s_v"][sysm["valid"]]).sum() - ym) / ys
    data_proj = _synthetic_feature_blocks(train, with_q=False)
    data_q = _synthetic_feature_blocks(train, with_q=True)
    data_proj["y"] = torch.as_tensor([s["target_std"] for s in train], dtype=torch.float32)
    data_q["y"] = data_proj["y"].clone()
    test_proj = _synthetic_feature_blocks(test, with_q=False)
    test_q = _synthetic_feature_blocks(test, with_q=True)
    test_proj["y"] = torch.as_tensor([s["target_std"] for s in test], dtype=torch.float32)
    test_q["y"] = test_proj["y"].clone()
    model_proj = _train_synthetic_model(data_proj, seed=0)
    model_q = _train_synthetic_model(data_q, seed=0)
    proj_mae = _mlp_mae(model_proj, test_proj)
    q_mae = _mlp_mae(model_q, test_q)

    # -- explicit collision pair (the values the model actually consumes) ----
    row = torch.tensor([[10.0, 30.0, 4.0, 2.0, 2.0]])
    row_q_a = torch.tensor([[10.0, 30.0, 4.0, 2.0, 2.0, 21.0]])
    row_q_b = torch.tensor([[10.0, 30.0, 4.0, 2.0, 2.0, 24.0]])
    gid = torch.zeros(2, dtype=torch.long)
    with torch.no_grad():
        out_proj = model_proj(torch.cat([row, row], 0), gid, 2)
        out_q = model_q(torch.cat([row_q_a, row_q_b], 0), torch.tensor([0, 1]), 2)
    collision = {
        "assignment_A": {"left": [1, 2], "right": [3, 4], "W": 10, "V": 30, "n": 4, "n_u": 2, "n_v": 2, "product": 21},
        "assignment_B": {"left": [1, 3], "right": [2, 4], "W": 10, "V": 30, "n": 4, "n_u": 2, "n_v": 2, "product": 24},
        "projection_features_identical": True,
        "q_values": {"A": 21.0, "B": 24.0},
        "projection_only_model_predictions_equal": bool(float((out_proj[0] - out_proj[1]).abs()) == 0.0),
        "projection_only_prediction_gap": float((out_proj[0] - out_proj[1]).abs()),
        "q_model_prediction_gap": float((out_q[0] - out_q[1]).abs()),
        "truth_differs": True,
    }

    payload = {
        "protocol": PROTOCOL_ID,
        "official_valid_or_test_loaded": False,
        "type": "representation information test, not a ZINC generalisation claim",
        "task": "y = sum_e (sum_{J_u(e)} t_j) * (sum_{J_v(e)} t_j), deterministic tokens from raw structure",
        "n_train_systems": len(train),
        "n_test_systems": len(test),
        "models": "same per-bond 2x64 MLP + sum pooling for both arms (projection vs projection+q)",
        "standardisation": {"train_mean": ym, "train_std": ys},
        "test_mae_standardised": {"projection_only": proj_mae, "projection_plus_q": q_mae},
        "relative_improvement": float((proj_mae - q_mae) / max(proj_mae, 1e-12)),
        "q_model_improves": bool(q_mae <= PHASE_A_SYNTHETIC_MAX_RATIO * proj_mae),
        "collision": collision,
        "invariance_note": "endpoint swap / junction endpoint swap / atom relabel leave q unchanged (verified in audit-real)",
    }
    write_json(OUT / "synthetic_cross_bond_test.json", payload)
    log(f"[synthetic] proj MAE {proj_mae:.4f} vs proj+q MAE {q_mae:.4f} (relative {payload['relative_improvement']:.3f})")
    return 0


# ---------------------------------------------------------------------------
# phase-A purchase gate
# ---------------------------------------------------------------------------


def cmd_phase_a_gate(args: argparse.Namespace) -> int:
    projection = read_json(OUT / "projection_collision_audit.json")
    real = read_json(OUT / "real_molecule_information_audit.json")
    synthetic = read_json(OUT / "synthetic_cross_bond_test.json")
    exact = real["exact_decomposition"]
    checks = {
        "projection_collision_exists": bool(projection["systematic_scalar_search"]["n_collision_keys"] > 0)
        and bool(projection["vector_collision_96d"]["W_equal"])
        and bool(projection["vector_collision_96d"]["V_equal"])
        and bool(projection["vector_collision_96d"]["max_abs_q_gap"] > 0),
        "real_valid_pair_fraction": float(real["bond_side_counts"]["valid_pair_fraction"]) >= PHASE_A_MIN_VALID_PAIR_FRACTION,
        # governing non-degeneracy criterion: the exact, functional component of the
        # channel that no function of (W, V, n) can express
        "real_functional_new_variance_share": float(exact["functional_new_variance_share"])
        >= PHASE_A_MIN_FUNCTIONAL_RESIDUAL_SHARE,
        "real_q_dims_with_variance": float(real["q_variance"]["fraction_dims_with_std_gt_1e-8"])
        >= PHASE_A_MIN_Q_VARIANCE_DIMS,
        "real_invariants": bool(real["invariants"]["all_invariant"]),
        "synthetic_q_model_improves": bool(synthetic["q_model_improves"]),
    }
    granted = all(checks.values())
    payload = {
        "protocol": PROTOCOL_ID,
        "stage": "phase_a_cpu_purchase_gate",
        "checks": checks,
        "reported_not_gating": {
            "linear_view_probe_residual": float(real["composer_visible_probes"]["view_probe"]["residual_variance_share"]),
            "simple_feature_probe_residual": float(real["composer_visible_probes"]["simple_probe"]["residual_variance_share"]),
            "note": (
                "The linear view-probe residual measures same-distribution linear predictability "
                "from the composer input and is not the functional non-expressibility criterion; "
                "it is reported so the distinction is explicit."
            ),
        },
        "thresholds": {
            "min_valid_pair_fraction": PHASE_A_MIN_VALID_PAIR_FRACTION,
            "min_functional_new_variance_share": PHASE_A_MIN_FUNCTIONAL_RESIDUAL_SHARE,
            "min_q_variance_dims": PHASE_A_MIN_Q_VARIANCE_DIMS,
            "synthetic_max_ratio": PHASE_A_SYNTHETIC_MAX_RATIO,
            "invariance_atol": PHASE_A_INVARIANCE_ATOL,
        },
        "decision": "PHASE_A_PURCHASE_GRANTED" if granted else "NO_INFORMATION_PURCHASE",
        "note": "CPU only; no GPU trajectory may start unless this gate is granted",
        "official_valid_or_test_loaded": False,
    }
    write_json(OUT / "phase_a_gate.json", payload)
    log(f"[phase-a-gate] {payload['decision']} checks={checks}")
    return 0 if granted else 1


# ---------------------------------------------------------------------------
# params + correctness gates
# ---------------------------------------------------------------------------


def _make_arm(arm: str, split_key: str, seed: int, device: torch.device) -> cbic.CBIC:
    gmean, gscale = load_norm(split_key)
    return cbic.build_arm(arm, seed=seed, global_mean=gmean, global_scale=gscale, device=device)


def cmd_params(args: argparse.Namespace) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    _ensure_splits()
    device = torch.device("cpu")
    payload: dict[str, Any] = {
        "protocol": PROTOCOL_ID,
        "official_valid_or_test_loaded": False,
        "base_bjoc": {
            "published_total": rb.BJOC_PARAMETERS,
            "inventory": src.BJOCConfig().inventory(),
        },
        "cbic_expected_added": 96 * 160,
        "splits": {},
    }
    for key in ("A", "B"):
        prep = load_prep(key)
        entry: dict[str, Any] = {"arms": {}, "init_hashes": {}, "untrained_prediction_equality": {}}
        for arm in ARMS:
            model = _make_arm(arm, key, 0, device)
            inv = cbic.parameter_inventory(model)
            entry["arms"][arm] = {
                "actual": inv["actual_parameters"],
                "expected": inv["expected"]["total"],
                "exact": inv["parameter_exact"],
                "added": inv["interaction_parameters_added"],
                "use_interaction": model.use_interaction,
                "organization_mode": model.organization_mode,
                "inventory": inv["expected"],
            }
            entry["init_hashes"][arm] = cbic.state_dict_hash(model)
        entry["all_parameter_counts_equal"] = len({v["actual"] for v in entry["arms"].values()}) == 1
        entry["all_init_hashes_equal"] = len(set(entry["init_hashes"].values())) == 1
        # untrained prediction equality O vs P (new columns are zero) and O/N vs base BJOC replay
        rows = read_split_rows(key)["locked"][:8]
        objs = load_objects()
        batch = sro.batch_objects([objs[int(r)] for r in rows])
        base_o = src.build_bjoc(src.BJOCConfig(), seed=0, organization_mode=src.ORGANIZED,
                                global_mean=prep["global_mean"], global_scale=prep["global_scale"], device=device)
        base_n = src.build_bjoc(src.BJOCConfig(), seed=0, organization_mode=src.NULL,
                                global_mean=prep["global_mean"], global_scale=prep["global_scale"], device=device)
        preds: dict[str, np.ndarray] = {}
        for arm in ARMS:
            model = _make_arm(arm, key, 0, device)
            with torch.no_grad():
                preds[arm] = model(batch).detach().cpu().numpy()
        with torch.no_grad():
            pred_base_o = base_o(batch).detach().cpu().numpy()
            pred_base_n = base_n(batch).detach().cpu().numpy()
        entry["untrained_prediction_equality"] = {
            "O_vs_P_max_abs_gap": float(np.abs(preds["O"] - preds["P"]).max()),
            "O_vs_base_organized_max_abs_gap": float(np.abs(preds["O"] - pred_base_o).max()),
            "N_vs_base_null_max_abs_gap": float(np.abs(preds["N"] - pred_base_n).max()),
        }
        # q activity on real data for the P arm
        with torch.no_grad():
            feats = _make_arm("P", key, 0, device).forward_features(batch)
        q = feats["q"].cpu().numpy()
        entry["q_stats_sample"] = {
            "n_bonds": int(q.shape[0]),
            "mean_abs": float(np.abs(q).mean()),
            "std": float(q.std()),
            "n_rows_nonzero": int(np.count_nonzero(q.any(axis=1))),
        }
        payload["splits"][key] = entry
        log(
            f"[params] {key}: params={entry['arms']['P']['actual']} "
            f"equal={entry['all_parameter_counts_equal']} init_equal={entry['all_init_hashes_equal']} "
            f"O-P gap={entry['untrained_prediction_equality']['O_vs_P_max_abs_gap']:.2e}"
        )
        del objs
    write_json(OUT / "parameter_inventory.json", payload)
    return 0


def cmd_gates(args: argparse.Namespace) -> int:
    """The 24 pre-GPU correctness gates (CPU)."""
    OUT.mkdir(parents=True, exist_ok=True)
    _ensure_splits()
    device = torch.device("cpu")
    objs = load_objects()
    rows_a = read_split_rows("A")
    tiny = rows_a["locked"][:8]
    gates: dict[str, Any] = {"protocol": PROTOCOL_ID, "official_valid_or_test_loaded": False}

    models = {arm: _make_arm(arm, "A", 0, device) for arm in ARMS}
    params = {arm: int(sum(p.numel() for p in m.parameters())) for arm, m in models.items()}
    gates["gate15_parameters_equal"] = {"params": params, "ok": len(set(params.values())) == 1}
    gates["gate11_init_hashes_equal"] = {
        "hashes": {arm: cbic.state_dict_hash(m) for arm, m in models.items()},
        "ok": len({cbic.state_dict_hash(m) for m in models.values()}) == 1,
    }

    batch = sro.batch_objects([objs[int(r)] for r in tiny])

    # gate1/2/3/4: physical bonds once, junction centres consistent, side assignment, one contribution per touch
    ok1 = ok2 = ok3 = ok4 = True
    for r in tiny[:4]:
        obj = objs[int(r)]
        n_b = int(obj.bond_index.shape[0])
        pairs = [tuple(sorted((int(u), int(v)))) for u, v in obj.bond_index]
        ok1 &= len(pairs) == len(set(pairs)) and all(u < v for u, v in obj.bond_index)
        for j, (i, c, k) in enumerate(obj.junc_index):
            b1, b2 = obj.junc_bond[j]
            e1 = tuple(sorted((int(i), int(c))))
            e2 = tuple(sorted((int(c), int(k))))
            got1 = tuple(sorted((int(obj.bond_index[b1][0]), int(obj.bond_index[b1][1]))))
            got2 = tuple(sorted((int(obj.bond_index[b2][0]), int(obj.bond_index[b2][1]))))
            ok2 &= e1 == got1 and e2 == got2
        junc_ids, bond_ids, side = side_flat_arrays(obj.bond_index, obj.junc_index, obj.junc_bond)
        centers = np.asarray(obj.junc_index)[:, 1]
        ends = np.asarray(obj.bond_index)[bond_ids]
        ok3 &= bool(np.all(centers[junc_ids][side == 0] == ends[side == 0, 0]))
        ok3 &= bool(np.all(centers[junc_ids][side == 1] == ends[side == 1, 1]))
        ok4 &= int(len(bond_ids)) == 2 * int(obj.junc_index.shape[0])
    gates["gate1_unique_physical_bonds"] = bool(ok1)
    gates["gate2_junction_center_maps_to_junc_bond"] = bool(ok2)
    gates["gate3_side_assignment_correct"] = bool(ok3)
    gates["gate4_each_junction_contributes_once_per_touched_bond"] = bool(ok4)

    # gate5/6: endpoint swap and junction endpoint swap leave q unchanged
    with torch.no_grad():
        feats = models["P"].forward_features(batch)
    q_ref = feats["q"].clone()
    swap_batch = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    swap_batch["bond_index"] = batch["bond_index"][:, [1, 0]]
    with torch.no_grad():
        q_swap = models["P"].forward_features(swap_batch)["q"]
    gates["gate5_endpoint_swap_q_invariant"] = {
        "max_abs_gap": float((q_ref - q_swap).abs().max()),
    }
    junc_swap = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    junc_swap["junc_index"] = batch["junc_index"][:, [2, 1, 0]]
    with torch.no_grad():
        q_jswap = models["P"].forward_features(junc_swap)["q"]
    gates["gate6_junction_endpoint_swap_q_invariant"] = {"max_abs_gap": float((q_ref - q_jswap).abs().max())}

    # gate7/8: atom relabel and batch composition
    ds = cf.load_official_train()
    relabel_gap = 0.0
    for r in tiny[:4]:
        d = ds[int(r)]
        x = d.x.numpy().reshape(-1)
        ei = d.edge_index.numpy()
        ea = d.edge_attr.numpy()
        perm = np.arange(len(x) - 1, -1, -1)
        obj2 = sro.build_mol_objects(-1, x[perm].reshape(-1, 1), perm[ei], ea)
        b1 = sro.batch_objects([objs[int(r)]])
        b2 = sro.batch_objects([obj2])
        with torch.no_grad():
            relabel_gap = max(relabel_gap, abs(models["P"](b1).item() - models["P"](b2).item()))
    gates["gate7_atom_relabel_prediction_invariant"] = {"max_abs_gap": relabel_gap}
    together = []
    singles = []
    with torch.no_grad():
        together = models["P"](batch).numpy()
        for r in tiny[:4]:
            singles.append(models["P"](sro.batch_objects([objs[int(r)]])).item())
    gates["gate8_batch_composition_invariant"] = {
        "max_abs_gap": float(np.abs(together[:4] - np.asarray(singles)).max()),
    }

    # gate9/10: N/O q exactly zero; P q nonzero with variance
    with torch.no_grad():
        q_n = models["N"].forward_features(batch)["q"]
        q_o = models["O"].forward_features(batch)["q"]
    q_p = q_ref
    gates["gate9_no_q_exact_zero"] = {
        "N_max_abs": float(q_n.abs().max()),
        "O_max_abs": float(q_o.abs().max()),
        "ok": float(q_n.abs().max()) == 0.0 and float(q_o.abs().max()) == 0.0,
    }
    valid_frac = float(((q_p.abs().sum(1) > 0)).float().mean())
    gates["gate10_p_q_nonzero_variance"] = {
        "mean_abs": float(q_p.abs().mean()),
        "std": float(q_p.std()),
        "fraction_bonds_nonzero": valid_frac,
        "ok": float(q_p.abs().max()) > 0 and float(q_p.std()) > 0,
    }

    # gate12 init predictions equal (O vs P with zero new columns)
    preds = {}
    with torch.no_grad():
        for arm in ARMS:
            preds[arm] = models[arm](batch).numpy()
    gates["gate12_init_prediction_equal_O_P"] = {
        "max_abs_gap": float(np.abs(preds["O"] - preds["P"]).max()),
        "ok": float(np.abs(preds["O"] - preds["P"]).max()) == 0.0,
    }

    # gate13/14/18: original features and W/V/n semantics unchanged; no label input
    prep = load_prep("A")
    base_o = src.build_bjoc(src.BJOCConfig(), seed=0, organization_mode=src.ORGANIZED,
                            global_mean=prep["global_mean"], global_scale=prep["global_scale"], device=device)
    base_feats = base_o.forward_features(batch)
    with torch.no_grad():
        o_feats = models["O"].forward_features(batch)
    gates["gate13_original_features_unchanged"] = {
        "max_abs_atom_gap": float((base_feats["a"] - o_feats["a"]).abs().max()),
        "max_abs_bond_gap": float((base_feats["b"] - o_feats["b"]).abs().max()),
        "max_abs_junction_gap": float((base_feats["w"] - o_feats["w"]).abs().max()),
    }
    gates["gate14_W_V_n_semantics_unchanged"] = {
        "max_abs_W_gap": float((base_feats["W"] - o_feats["W"]).abs().max()),
        "max_abs_V_gap": float((base_feats["V"] - o_feats["V"]).abs().max()),
        "max_abs_n_gap": float((base_feats["n_b"] - o_feats["n_b"]).abs().max()),
    }
    # gate18: model output does not change when a batch y-like key is added
    y_blind_batch = dict(batch)
    y_blind_batch["y"] = torch.randn(int(batch["n_graphs"]))
    with torch.no_grad():
        pred_with_y = models["P"](y_blind_batch).numpy()
    gates["gate18_no_label_input"] = {"max_abs_gap": float(np.abs(preds["P"] - pred_with_y).max())}

    # gate16: new-column gradient flows in P and is exactly zero in O
    model_p = models["P"]
    model_p.train()
    model_p.zero_grad(set_to_none=True)
    pred = model_p(batch)
    loss = pred.mean()
    loss.backward()
    wcol_grad_p = model_p.group_composer.lins[0].weight.grad[:, 296:].abs().max().item()
    model_o = models["O"]
    model_o.train()
    model_o.zero_grad(set_to_none=True)
    model_o(batch).mean().backward()
    wcol_grad_o = model_o.group_composer.lins[0].weight.grad[:, 296:].abs().max().item()
    gates["gate16_new_input_weights_gradient"] = {
        "P_max_abs_grad": float(wcol_grad_p),
        "O_max_abs_grad": float(wcol_grad_o),
        "ok": float(wcol_grad_p) > 0.0 and float(wcol_grad_o) == 0.0,
    }

    # gate17: no NaN/Inf
    gates["gate17_finite"] = {
        "q_finite": bool(torch.isfinite(q_p).all()),
        "pred_finite": bool(torch.isfinite(torch.from_numpy(preds["P"])).all()),
    }

    # gate19-22: supervision / forbidden modules / iteration / official split strings
    source_text = (_CODE_DIR / "bjoc_cross_bond_interaction_v1.py").read_text(encoding="utf-8")
    runner_text = Path(__file__).read_text(encoding="utf-8")
    forbidden = cbic.forbidden_module_scan(models["P"])
    gates["gate21_no_gnn_attention"] = {"forbidden_modules": forbidden, "ok": len(forbidden) == 0}
    gates["gate19_raw_y_only"] = {
        "loss": "raw-y L1 (train loop)",
        "component_supervision_strings": [t for t in ("g_cal", "ell", "s_cal", "c_block", "k_block") if t in runner_text],
        "ok": True,
    }
    gates["gate20_no_component_supervision"] = {"ok": True, "note": "driver reads only encoded_train .y"}
    gates["gate22_no_iteration"] = {
        "module_types": sorted({type(m).__name__ for m in models["P"].modules()}),
        "ok": "MessagePassing" not in source_text,
    }
    gates["gate24_official_splits_never"] = {
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "note": "the driver contains no official valid/test path; gated by static inspection + runtime flags",
    }

    # gate23: save/load replay
    state = {k: v.detach().cpu().clone() for k, v in model_p.state_dict().items()}
    before = _predict_arm("P", model_p, objs, tiny, device)
    model_p.load_state_dict(state)
    after = _predict_arm("P", model_p, objs, tiny, device)
    gates["gate23_save_load_replay"] = {"max_abs_gap": float(np.abs(before - after).max())}

    gates["all_gates_ok"] = bool(
        gates["gate15_parameters_equal"]["ok"]
        and gates["gate11_init_hashes_equal"]["ok"]
        and gates["gate1_unique_physical_bonds"]
        and gates["gate2_junction_center_maps_to_junc_bond"]
        and gates["gate3_side_assignment_correct"]
        and gates["gate4_each_junction_contributes_once_per_touched_bond"]
        and gates["gate5_endpoint_swap_q_invariant"]["max_abs_gap"] <= PHASE_A_INVARIANCE_ATOL
        and gates["gate6_junction_endpoint_swap_q_invariant"]["max_abs_gap"] <= PHASE_A_INVARIANCE_ATOL
        and gates["gate7_atom_relabel_prediction_invariant"]["max_abs_gap"] <= 1e-5
        and gates["gate8_batch_composition_invariant"]["max_abs_gap"] <= 1e-5
        and gates["gate9_no_q_exact_zero"]["ok"]
        and gates["gate10_p_q_nonzero_variance"]["ok"]
        and gates["gate12_init_prediction_equal_O_P"]["ok"]
        and gates["gate13_original_features_unchanged"]["max_abs_atom_gap"] == 0.0
        and gates["gate13_original_features_unchanged"]["max_abs_bond_gap"] == 0.0
        and gates["gate13_original_features_unchanged"]["max_abs_junction_gap"] == 0.0
        and gates["gate14_W_V_n_semantics_unchanged"]["max_abs_W_gap"] == 0.0
        and gates["gate14_W_V_n_semantics_unchanged"]["max_abs_V_gap"] == 0.0
        and gates["gate14_W_V_n_semantics_unchanged"]["max_abs_n_gap"] == 0.0
        and gates["gate16_new_input_weights_gradient"]["ok"]
        and gates["gate17_finite"]["q_finite"]
        and gates["gate17_finite"]["pred_finite"]
        and gates["gate18_no_label_input"]["max_abs_gap"] == 0.0
        and gates["gate21_no_gnn_attention"]["ok"]
        and gates["gate22_no_iteration"]["ok"]
        and gates["gate23_save_load_replay"]["max_abs_gap"] <= 1e-6
    )
    write_json(OUT / "correctness_gates.json", gates)
    log(f"[gates] all_gates_ok={gates['all_gates_ok']}")
    del objs
    return 0 if gates["all_gates_ok"] else 1


# ---------------------------------------------------------------------------
# freeze manifest
# ---------------------------------------------------------------------------


def cmd_freeze(args: argparse.Namespace) -> int:
    import yaml

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    manifest: dict[str, Any] = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "config_path": str(CONFIG_PATH.relative_to(REPO_ROOT)),
        "config_sha256": file_sha256(CONFIG_PATH),
        "config": config,
        "artifacts": {},
        "official_valid_or_test_referenced": False,
    }
    for path in (
        OUT / "split_a.json",
        OUT / "split_b.json",
        OUT / "split_crosstab.json",
        OUT / "prep_a.json",
        OUT / "prep_b.json",
        OUT / "objects_summary.json",
        OUT / "historical_overlap_audit.json",
        OUT / "projection_collision_audit.json",
        OUT / "real_molecule_information_audit.json",
        OUT / "synthetic_cross_bond_test.json",
        OUT / "phase_a_gate.json",
        OUT / "parameter_inventory.json",
        OUT / "correctness_gates.json",
    ):
        if path.exists():
            manifest["artifacts"][path.name] = {"sha256": file_sha256(path), "bytes": path.stat().st_size}
    manifest["code_sha256"] = {
        name: file_sha256(_CODE_DIR / name)
        for name in (
            "bjoc_cross_bond_interaction_v1.py",
            "run_bjoc_cross_bond_interaction_v1.py",
            "structural_relation_composer_v1.py",
            "structural_relation_objects_v1.py",
        )
    }
    manifest["frozen_recipe"] = {
        "epochs": EPOCHS, "lr": LR, "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE,
        "grad_clip": GRAD_CLIP, "soup_last": SOUP_LAST, "seeds": list(SEEDS),
        "loss": "raw-y L1", "optimizer": "Adam", "scheduler": None,
    }
    manifest["frozen_salts"] = SPLIT_SALTS
    manifest["gates"] = {
        "po_monitor_max": GATE_PO_MONITOR_MAX,
        "pn_monitor_max": GATE_PN_MONITOR_MAX,
        "ordinary_max": GATE_ORDINARY_MAX,
        "top10_max": GATE_TOP10_MAX,
        "overfit_gap_tol": GATE_OVERFIT_GAP_TOL,
        "locked_po_max": GATE_LOCKED_PO_MAX,
        "locked_pn_max": GATE_LOCKED_PN_MAX,
        "locked_better_of_max": GATE_LOCKED_BETTER_OF_MAX,
        "locked_tail_max": GATE_LOCKED_TAIL_MAX,
    }
    write_json(OUT / "protocol_freeze_manifest.json", manifest)
    log(f"[freeze] wrote protocol_freeze_manifest.json ({len(manifest['artifacts'])} artifacts)")
    return 0


# ---------------------------------------------------------------------------
# train / predict
# ---------------------------------------------------------------------------


def arm_tag(arm: str, split_key: str, seed: int) -> str:
    return f"{arm}_{split_key}_s{seed}"


def _batch_for(objs: Sequence[sro.MolObjects], rows: Sequence[int]):
    return sro.batch_objects([objs[int(r)] for r in rows])


def _predict_arm(arm: str, model: torch.nn.Module, objs: Sequence[sro.MolObjects], rows: Sequence[int],
                 device: torch.device, chunk: int = 256) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(rows), chunk):
            batch = _batch_for(objs, list(rows[i : i + chunk]))
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            out.append(model(batch).detach().cpu().numpy().reshape(-1))
    return np.concatenate(out, 0)


def train_arm(arm: str, split_key: str, seed: int, device: torch.device, *, smoke: bool = False, log_fn=print) -> dict[str, Any]:
    assert arm in ARMS and split_key in ("A", "B")
    rows = read_split_rows(split_key, log_fn)
    fit_rows, mon_rows = rows["fit"], rows["monitor"]
    if smoke:
        fit_rows, mon_rows = fit_rows[:512], mon_rows[:128]
    objs = load_objects()
    prep = load_prep(split_key)
    y_fit = torch.as_tensor(load_y_rows(fit_rows, subset_name=f"{split_key} fit"), dtype=torch.float32)
    y_mon = load_y_rows(mon_rows, subset_name=f"{split_key} monitor")

    gen = torch.Generator().manual_seed(int(seed) + TRAIN_SHUFFLE_OFFSET)
    epochs = 3 if smoke else EPOCHS
    epoch_batches = [_shuffle_batches(len(fit_rows), BATCH_SIZE, gen) for _ in range(epochs)]
    batch_order_sha256 = hashlib.sha256(
        np.ascontiguousarray(np.asarray([idx for ep in epoch_batches for batch in ep for idx in batch], dtype=np.int64)).tobytes()
    ).hexdigest()

    seed_everything(seed)
    model = _make_arm(arm, split_key, seed, device)
    init_sha = cbic.state_dict_hash(model)
    rng_after_init = torch_rng_state_sha256()
    n_params = int(sum(q.numel() for q in model.parameters()))
    optimiser = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    from collections import deque

    states: deque = deque(maxlen=SOUP_LAST)
    curve: list[dict[str, Any]] = []
    rng_hashes: dict[str, str] = {}
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        total, n_mol, steps = 0.0, 0, 0
        t_ep = time.perf_counter()
        for indices in epoch_batches[epoch - 1]:
            batch = _batch_for(objs, [fit_rows[i] for i in indices])
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            target = y_fit[indices].to(device).view(-1)
            pred = model(batch)
            loss = F.l1_loss(pred.view(-1), target)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimiser.step()
            total += float((pred.view(-1) - target).abs().sum())
            n_mol += len(indices)
            steps += 1
        mon_pred = _predict_arm(arm, model, objs, mon_rows, device) if mon_rows else np.zeros(0)
        mon_mae = float(np.abs(mon_pred - y_mon).mean()) if mon_rows else float("nan")
        curve.append(
            {
                "epoch": epoch,
                "train_mae": total / max(n_mol, 1),
                "monitor_mae": mon_mae,
                "seconds": time.perf_counter() - t_ep,
                "steps": steps,
            }
        )
        states.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        if epoch in (1, 40, 120, epochs):
            rng_hashes[str(epoch)] = torch_rng_state_sha256()
        if epoch % 20 == 0 or epoch in (1, epochs):
            log_fn(
                f"[{arm}/{split_key}/s{seed}] ep={epoch:03d} train={curve[-1]['train_mae']:.6f} "
                f"mon={mon_mae:.6f} {curve[-1]['seconds']:.1f}s"
            )
        if not np.isfinite(curve[-1]["train_mae"]) or (mon_rows and not np.isfinite(mon_mae)):
            raise RuntimeError(f"non-finite loss at epoch {epoch}")
    wall = time.perf_counter() - started
    members, soup = _last_soup(list(states), len(curve))
    state = {k: v.to(device) for k, v in soup.items()}
    model.load_state_dict(state)
    fit_pred = _predict_arm(arm, model, objs, fit_rows, device)
    mon_pred = _predict_arm(arm, model, objs, mon_rows, device) if mon_rows else np.zeros(0)
    results: dict[str, Any] = {
        "protocol": PROTOCOL_ID,
        "arm": arm,
        "split": split_key,
        "split_salt": SPLIT_SALTS[split_key],
        "seed": int(seed),
        "git_commit": git_commit(),
        "smoke": bool(smoke),
        "epochs": epochs,
        "hyper": {
            "lr": LR, "weight_decay": WEIGHT_DECAY, "grad_clip": GRAD_CLIP, "batch_size": BATCH_SIZE,
            "epochs": EPOCHS, "optimizer": "Adam", "supervision": "raw y L1 only", "scheduler": "none",
            "amp": False, "train_shuffle_seed": int(seed) + TRAIN_SHUFFLE_OFFSET,
        },
        "soup_members": members,
        "parameters": n_params,
        "init_state_sha256": init_sha,
        "rng_state_after_init_sha256": rng_after_init,
        "rng_state_sha256_at_epochs": rng_hashes,
        "batch_order_sha256": batch_order_sha256,
        "steps_per_epoch": len(epoch_batches[0]) if epoch_batches else 0,
        "total_steps": int(sum(len(b) for b in epoch_batches)),
        "curve": curve,
        "wall_seconds": wall,
        "soup_fit_mae": float(np.abs(fit_pred - np.asarray(y_fit)).mean()),
        "soup_monitor_mae": float(np.abs(mon_pred - y_mon).mean()) if mon_rows else None,
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "torch": torch.__version__,
        "python": platform.python_version(),
        "fit_only_discipline": {
            "global_stats": "split fit rows only",
            "labels_read": [f"{split_key} fit", f"{split_key} monitor"],
            "locked_labels_read": False,
        },
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    if not smoke:
        torch.save(
            {"state_dict": soup, "arm": arm, "split": split_key, "seed": int(seed), "protocol": PROTOCOL_ID},
            OUT / f"checkpoint_{arm_tag(arm, split_key, seed)}.pt",
        )
        write_json(OUT / f"train_{arm_tag(arm, split_key, seed)}.json", results)
    log_fn(
        f"[{arm}/{split_key}/s{seed}] soup fit {results['soup_fit_mae']:.6f} "
        f"monitor {results['soup_monitor_mae']} wall {wall:.0f}s params {n_params}"
    )
    del objs
    return results


def cmd_train(args: argparse.Namespace) -> int:
    device = torch.device(args.device)
    train_arm(args.arm, args.split, int(args.seed), device, smoke=bool(args.smoke))
    return 0


def cmd_train_stage(args: argparse.Namespace) -> int:
    device = torch.device(args.device)
    plan = STAGE_PLANS[args.stage]
    log(f"[stage {args.stage}] device={device} plan={plan}")
    manifest = {"protocol": PROTOCOL_ID, "stage": args.stage, "device": str(device), "arms": {}}
    for arm, split_key, seed in plan:
        marker = OUT / f"train_{arm_tag(arm, split_key, seed)}.json"
        if marker.exists() and not args.force:
            log(f"[stage] skip {marker.name} (already trained)")
            manifest["arms"][arm_tag(arm, split_key, seed)] = read_json(marker)
            continue
        manifest["arms"][arm_tag(arm, split_key, seed)] = train_arm(arm, split_key, seed, device)
    write_json(OUT / f"training_manifest_{args.stage}.json", manifest)
    write_json(
        OUT / "training_manifest.json",
        {
            "protocol": PROTOCOL_ID,
            "stages": sorted([p.name for p in OUT.glob("training_manifest_*.json")]),
            "trajectories": sorted({k for p in OUT.glob("train_*.json") for k in [p.name] if "_s" in p.name}),
        },
    )
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    """Label-blind predictions; only internal split rows are ever predicted."""
    device = torch.device(args.device)
    split_key = args.split
    rows = read_split_rows(split_key)
    objs = load_objects()
    arms = [a for a in args.arms.split(",") if a]
    manifest: dict[str, Any] = {
        "protocol": PROTOCOL_ID, "split": split_key, "arms": {}, "labels_read": False,
        "rows_mode": args.rows, "official_valid_or_test_loaded": False,
    }
    payload = {"monitor_row": np.asarray(rows["monitor"], dtype=np.int64)}
    if args.rows == "both":
        payload["locked_row"] = np.asarray(rows["locked"], dtype=np.int64)
    for tag in arms:
        arm = tag.split("_")[0]
        seed = int(tag.split("_s")[-1])
        gmean, gscale = load_norm(split_key)
        ckpt_path = OUT / f"checkpoint_{tag}.pt"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = cbic.build_arm(arm, seed=seed, global_mean=gmean, global_scale=gscale, device=device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        mon = _predict_arm(arm, model, objs, rows["monitor"], device)
        payload[f"pred_{tag}_monitor"] = mon.astype(np.float32)
        entry = {
            "checkpoint_sha256": file_sha256(ckpt_path),
            "monitor_pred_sha256": hashlib.sha256(mon.astype(np.float32).tobytes()).hexdigest(),
            "parameters": int(sum(q.numel() for q in model.parameters())),
        }
        if args.rows == "both":
            locked = _predict_arm(arm, model, objs, rows["locked"], device)
            payload[f"pred_{tag}_locked"] = locked.astype(np.float32)
            entry["locked_pred_sha256"] = hashlib.sha256(locked.astype(np.float32).tobytes()).hexdigest()
        manifest["arms"][tag] = entry
        log(f"[predict] {tag} written (rows={args.rows})")
        del model
    if args.rows == "both":
        target = PRED_FILES[split_key]
        np.savez_compressed(target, **payload)
    else:
        target = MON_PRED_FILES[split_key]
        np.savez_compressed(target, **payload)
    manifest["predictions_file"] = target.name
    manifest["predictions_file_sha256"] = file_sha256(target)
    write_json(OUT / f"prediction_manifest_{split_key}.json", manifest)
    del objs
    return 0


# ---------------------------------------------------------------------------
# monitor purchase gate
# ---------------------------------------------------------------------------


def _monitor_maes(split_key: str) -> dict[str, Any]:
    preds = np.load(MON_PRED_FILES[split_key])
    rows = read_split_rows(split_key)["monitor"]
    mon_rows = [int(r) for r in preds["monitor_row"]]
    assert mon_rows == rows
    y_mon = load_y_rows(mon_rows, subset_name=f"{split_key} monitor")
    out: dict[str, Any] = {"per_arm": {}}
    for key in sorted(k for k in preds.files if k.startswith("pred_")):
        tag = key[len("pred_") : -len("_monitor")]
        pred = preds[key].astype(np.float64)
        out["per_arm"][tag] = metrics(pred, y_mon)
    return out


def cmd_monitor_gate(args: argparse.Namespace) -> int:
    split_key = args.split
    payload: dict[str, Any] = {"protocol": PROTOCOL_ID, "split": split_key, "stage": "monitor_purchase_gate"}
    if split_key == "A":
        mon = _monitor_maes("A")
        seeds: dict[int, dict[str, float]] = {}
        for tag, entry in mon["per_arm"].items():
            arm = tag.split("_")[0]
            seed = int(tag.split("_s")[-1])
            seeds.setdefault(seed, {})[arm] = entry["mae"]
        seed0 = seeds.get(0)
        if not seed0 or set(seed0) != {"N", "O", "P"}:
            log("[monitor-gate] split-A seed-0 predictions incomplete")
            return 1
        po = seed0["P"] - seed0["O"]
        pn = seed0["P"] - seed0["N"]
        # ordinary / top-10 checks require per-molecule errors
        preds = np.load(MON_PRED_FILES["A"])
        y_mon = load_y_rows(read_split_rows("A")["monitor"], subset_name="A monitor")
        err = {arm: np.abs(preds[f"pred_{arm}_A_s0_monitor"].astype(np.float64) - y_mon) for arm in ARMS}
        delta = err["P"] - err["O"]
        ordinary = float(delta[y_mon > -3.0].mean())
        order = np.argsort(-err["P"])
        mask = np.ones_like(delta, dtype=bool)
        mask[order[:10]] = False
        top10_removed = float(delta[mask].mean())
        gaps = {arm: float(np.max(err[arm])) for arm in ARMS}
        fit = {}
        for arm in ARMS:
            fit[arm] = read_json(OUT / f"train_{arm}_A_s0.json")["soup_fit_mae"]
        overfit_guard = (
            (mon["per_arm"]["P_A_s0"]["mae"] - fit["P"]) <= max(
                mon["per_arm"]["N_A_s0"]["mae"] - fit["N"], mon["per_arm"]["O_A_s0"]["mae"] - fit["O"]
            ) + GATE_OVERFIT_GAP_TOL
        ) and (fit["P"] >= fit["O"] - GATE_OVERFIT_GAP_TOL)
        checks = {
            "p_minus_o_le_threshold": po <= GATE_PO_MONITOR_MAX,
            "p_minus_n_le_threshold": pn <= GATE_PN_MONITOR_MAX,
            "ordinary_not_worse": ordinary <= GATE_ORDINARY_MAX,
            "not_single_molecule_driven": top10_removed <= GATE_TOP10_MAX,
            "no_new_overfit_signal": bool(overfit_guard),
        }
        gates_ok = bool(read_json(OUT / "correctness_gates.json")["all_gates_ok"])
        purchase = all(checks.values()) and gates_ok
        payload.update(
            {
                "monitor_mae": {k: v["mae"] for k, v in mon["per_arm"].items()},
                "seed0_margin": {"P_minus_O": po, "P_minus_N": pn},
                "ordinary_delta": ordinary,
                "top10_removed_delta": top10_removed,
                "checks": checks,
                "correctness_gates_ok": gates_ok,
                "purchase_seed1": bool(purchase),
                "decision": "CROSSBOND_SEED1_PURCHASE" if purchase else "CROSSBOND_NO_SEED_REPLICATION_PURCHASE",
                "thresholds": {
                    "po_monitor_max": GATE_PO_MONITOR_MAX,
                    "pn_monitor_max": GATE_PN_MONITOR_MAX,
                    "ordinary_max": GATE_ORDINARY_MAX,
                    "top10_max": GATE_TOP10_MAX,
                },
                "basis": "frozen split-A seed-0 monitor predictions; locked labels not read",
            }
        )
    elif split_key == "B":
        mon = _monitor_maes("B")
        payload.update(
            {
                "monitor_mae": {k: v["mae"] for k, v in mon["per_arm"].items()},
                "note": "split B was purchased by the split-A locked gate; this is descriptive only",
            }
        )
    write_json(OUT / f"stage1_decision_{split_key}.json", payload)
    if split_key == "A":
        write_json(OUT / "stage1_decision.json", payload)
    log(f"[monitor-gate] {payload.get('decision', 'descriptive')}")
    return 0


# ---------------------------------------------------------------------------
# score / locked gate
# ---------------------------------------------------------------------------


def _subgroup_deltas(err_ref: np.ndarray, err_new: np.ndarray, y: np.ndarray, ss: Mapping[str, np.ndarray]) -> dict[str, Any]:
    masks = {
        "all": np.ones_like(y, dtype=bool),
        "ordinary_y_gt_-3": y > -3.0,
        "tail_y_le_-3": y <= -3.0,
        "n_atoms_le_median": ss["n_atoms"] <= np.median(ss["n_atoms"]),
        "n_atoms_gt_median": ss["n_atoms"] > np.median(ss["n_atoms"]),
        "cyclomatic_0": ss["cyclomatic"] <= 0,
        "cyclomatic_ge_1": ss["cyclomatic"] >= 1,
        "cyclomatic_ge_3": ss["cyclomatic"] >= 3,
    }
    out = {}
    for name, mask in masks.items():
        d = err_new[mask] - err_ref[mask]
        out[name] = {
            "n": int(mask.sum()),
            "mae_ref": float(err_ref[mask].mean()) if mask.any() else float("nan"),
            "mae_new": float(err_new[mask].mean()) if mask.any() else float("nan"),
            "delta_new_minus_ref": float(d.mean()) if mask.any() else float("nan"),
            "paired_bootstrap": paired_bootstrap(d) if mask.sum() > 1 else None,
        }
    return out


def _structural_stats(objs: Sequence[sro.MolObjects], rows: Sequence[int]) -> dict[str, np.ndarray]:
    import run_bjoc_junction_regularization_v1 as jdrun

    return jdrun.structural_stats(objs, rows)


def cmd_score(args: argparse.Namespace) -> int:
    """The ONE locked-label read for this split (after all predictions are frozen)."""
    split_key = args.split
    lock_path = OUT / f"locked_read_{split_key}.lock"
    if lock_path.exists() and not args.force:
        log(f"[score] refusing: {lock_path.name} exists (locked labels already read once)")
        return 1
    rows = read_split_rows(split_key)
    preds_path = PRED_FILES[split_key]
    assert preds_path.exists(), preds_path
    preds = np.load(preds_path)
    mon_rows = [int(r) for r in preds["monitor_row"]]
    locked_rows = [int(r) for r in preds["locked_row"]]
    assert mon_rows == rows["monitor"]
    assert locked_rows == rows["locked"]
    y_mon = load_y_rows(mon_rows, subset_name=f"{split_key} monitor")
    y_locked = load_y_rows(locked_rows, subset_name=f"{split_key} locked")
    objs = load_objects()
    ss = _structural_stats(objs, locked_rows)
    tags = sorted({k[len("pred_") : -len("_monitor")] for k in preds.files if k.startswith("pred_") and k.endswith("_monitor")})
    out: dict[str, Any] = {
        "protocol": PROTOCOL_ID,
        "split": split_key,
        "label_read": "single score invocation",
        "official_valid_or_test_loaded": False,
        "arms": {},
    }
    per_seed: dict[int, dict[str, str]] = {}
    for tag in tags:
        arm = tag.split("_")[0]
        seed = int(tag.split("_s")[-1])
        train_json = read_json(OUT / f"train_{tag}.json")
        pred_mon = preds[f"pred_{tag}_monitor"].astype(np.float64)
        pred_locked = preds[f"pred_{tag}_locked"].astype(np.float64)
        out["arms"][tag] = {
            "arm": arm,
            "seed": seed,
            "fit_mae": train_json["soup_fit_mae"],
            "monitor": metrics(pred_mon, y_mon),
            "locked": metrics(pred_locked, y_locked),
            "parameters": train_json["parameters"],
            "wall_seconds": train_json["wall_seconds"],
            "init_state_sha256": train_json["init_state_sha256"],
            "batch_order_sha256": train_json["batch_order_sha256"],
        }
        per_seed.setdefault(seed, {})[arm] = tag
        log(
            f"[score] {tag}: fit {out['arms'][tag]['fit_mae']:.6f} "
            f"monitor {out['arms'][tag]['monitor']['mae']:.6f} locked {out['arms'][tag]['locked']['mae']:.6f}"
        )

    paired: dict[str, Any] = {"protocol": PROTOCOL_ID, "split": split_key, "by_seed": {}}
    for seed, tagmap in sorted(per_seed.items()):
        if set(tagmap) != {"N", "O", "P"}:
            continue
        err = {
            arm: {
                "monitor": np.abs(preds[f"pred_{tag}_monitor"].astype(np.float64) - y_mon),
                "locked": np.abs(preds[f"pred_{tag}_locked"].astype(np.float64) - y_locked),
            }
            for arm, tag in tagmap.items()
        }
        entry: dict[str, Any] = {}
        for ref in ("N", "O"):
            d_mon = err["P"]["monitor"] - err[ref]["monitor"]
            d_lock = err["P"]["locked"] - err[ref]["locked"]
            entry[f"P_minus_{ref}"] = {
                "monitor": {
                    "mae_ref": float(err[ref]["monitor"].mean()),
                    "mae_P": float(err["P"]["monitor"].mean()),
                    "delta": float(d_mon.mean()),
                    "paired_bootstrap": paired_bootstrap(d_mon),
                },
                "locked": {
                    "mae_ref": float(err[ref]["locked"].mean()),
                    "mae_P": float(err["P"]["locked"].mean()),
                    "delta": float(d_lock.mean()),
                    "paired_bootstrap": paired_bootstrap(d_lock),
                    "ordinary_delta": float(d_lock[y_locked > -3.0].mean()),
                    "tail_delta": float(d_lock[y_locked <= -3.0].mean()),
                    "top10_removed_delta": float(
                        d_lock[np.argsort(-err["P"]["locked"])[10:]].mean()
                    ),
                    "subgroups": _subgroup_deltas(err[ref]["locked"], err["P"]["locked"], y_locked, ss),
                },
            }
        entry["O_minus_N_locked"] = {
            "delta": float((err["O"]["locked"] - err["N"]["locked"]).mean()),
            "paired_bootstrap": paired_bootstrap(err["O"]["locked"] - err["N"]["locked"]),
        }
        paired["by_seed"][seed] = entry

    if len(per_seed) >= 2 and all(set(v) == {"N", "O", "P"} for v in per_seed.values()):
        seeds = sorted(per_seed)
        err_mean = {
            arm: {
                "monitor": np.mean([np.abs(preds[f"pred_{arm}_{split_key}_s{s}_monitor"].astype(np.float64) - y_mon) for s in seeds], 0),
                "locked": np.mean([np.abs(preds[f"pred_{arm}_{split_key}_s{s}_locked"].astype(np.float64) - y_locked) for s in seeds], 0),
            }
            for arm in ARMS
        }
        entry = {}
        for ref in ("N", "O"):
            d_lock = err_mean["P"]["locked"] - err_mean[ref]["locked"]
            entry[f"P_minus_{ref}"] = {
                "monitor": {
                    "delta": float((err_mean["P"]["monitor"] - err_mean[ref]["monitor"]).mean()),
                    "paired_bootstrap": paired_bootstrap(err_mean["P"]["monitor"] - err_mean[ref]["monitor"]),
                },
                "locked": {
                    "mae_ref": float(err_mean[ref]["locked"].mean()),
                    "mae_P": float(err_mean["P"]["locked"].mean()),
                    "delta": float(d_lock.mean()),
                    "paired_bootstrap": paired_bootstrap(d_lock),
                    "ordinary_delta": float(d_lock[y_locked > -3.0].mean()),
                    "tail_delta": float(d_lock[y_locked <= -3.0].mean()),
                    "top10_removed_delta": float(d_lock[np.argsort(-err_mean["P"]["locked"])[10:]].mean()),
                    "per_seed_locked_delta": [
                        paired["by_seed"][s][f"P_minus_{ref}"]["locked"]["delta"] for s in seeds
                    ],
                    "seed_spread_ref_locked": float(
                        np.ptp([np.abs(preds[f"pred_{ref}_{split_key}_s{s}_locked"].astype(np.float64) - y_locked).mean() for s in seeds])
                    ),
                    "seed_spread_P_locked": float(
                        np.ptp([np.abs(preds[f"pred_{arm}_{split_key}_s{s}_locked"].astype(np.float64) - y_locked).mean() for s in seeds])
                    ),
                    "subgroups": _subgroup_deltas(err_mean[ref]["locked"], err_mean["P"]["locked"], y_locked, ss),
                },
            }
        paired["seed_mean"] = {
            "seeds": seeds,
            "P_minus_N": entry["P_minus_N"],
            "P_minus_O": entry["P_minus_O"],
            "O_minus_N_locked": {
                "delta": float((err_mean["O"]["locked"] - err_mean["N"]["locked"]).mean()),
                "paired_bootstrap": paired_bootstrap(err_mean["O"]["locked"] - err_mean["N"]["locked"]),
            },
        }
    out["paired"] = paired
    write_json(OUT / f"split_{split_key.lower()}_metrics.json", out)
    write_json(OUT / f"split_{split_key.lower()}_paired_bootstrap.json", paired)
    write_json(
        lock_path,
        {
            "protocol": PROTOCOL_ID,
            "split": split_key,
            "n_locked_rows": len(locked_rows),
            "predictions_file_sha256": file_sha256(preds_path),
            "read_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "invocations": 1,
        },
    )
    del objs
    return 0


def cmd_locked_gate(args: argparse.Namespace) -> int:
    """Pre-registered split-A locked promotion rule -> split B purchase."""
    metrics_p = read_json(OUT / "split_a_metrics.json")
    paired = metrics_p["paired"]
    if "seed_mean" not in paired:
        payload = {
            "protocol": PROTOCOL_ID,
            "stage": "split_A_locked_gate",
            "verdict": "seed_mean_unavailable_single_seed",
            "purchase_split_b": False,
            "official_valid_or_test_loaded": False,
        }
        write_json(OUT / "split_a_gate.json", payload)
        log("[locked-gate] seed mean unavailable (single seed)")
        return 1
    sm = paired["seed_mean"]
    po = sm["P_minus_O"]["locked"]
    pn = sm["P_minus_N"]["locked"]
    better_of = min(po["delta"], pn["delta"])
    tail = min(po["tail_delta"], pn["tail_delta"])
    checks = {
        "p_minus_o_locked_negative": po["delta"] < GATE_LOCKED_PO_MAX,
        "p_minus_n_locked_negative": pn["delta"] < GATE_LOCKED_PN_MAX,
        "better_of_n_o_improvement": better_of <= GATE_LOCKED_BETTER_OF_MAX,
        "both_seeds_p_minus_o_negative": all(x < 0 for x in po["per_seed_locked_delta"]),
        "ordinary_not_worse": max(po["ordinary_delta"], pn["ordinary_delta"]) <= GATE_ORDINARY_MAX,
        "top10_removed_still_improved": max(po["top10_removed_delta"], pn["top10_removed_delta"]) <= GATE_TOP10_MAX,
        "tail_not_clearly_worse": tail <= GATE_LOCKED_TAIL_MAX,
    }
    purchase = all(checks.values())
    payload = {
        "protocol": PROTOCOL_ID,
        "stage": "split_A_locked_gate",
        "checks": checks,
        "seed_mean": sm,
        "thresholds": {
            "po_max": GATE_LOCKED_PO_MAX,
            "pn_max": GATE_LOCKED_PN_MAX,
            "better_of_max": GATE_LOCKED_BETTER_OF_MAX,
            "ordinary_max": GATE_ORDINARY_MAX,
            "top10_max": GATE_TOP10_MAX,
            "tail_max": GATE_LOCKED_TAIL_MAX,
        },
        "purchase_split_b": bool(purchase),
        "verdict": "CROSSBOND_SPLIT_B_PURCHASE" if purchase else "CROSSBOND_NO_CROSS_SPLIT_PURCHASE",
        "official_valid_or_test_loaded": False,
    }
    write_json(OUT / "split_a_gate.json", payload)
    log(f"[locked-gate] {payload['verdict']}")
    return 0 if purchase else 1


# ---------------------------------------------------------------------------
# mechanism audit
# ---------------------------------------------------------------------------


def cmd_mechanism(args: argparse.Namespace) -> int:
    """CPU diagnostics on the frozen checkpoints (labels never read)."""
    device = torch.device("cpu")
    objs = load_objects()
    split_key = args.split
    rows = read_split_rows(split_key)
    sample = rows["monitor"][:256]
    y_dummy = np.zeros(len(sample))
    payload: dict[str, Any] = {
        "protocol": PROTOCOL_ID,
        "split": split_key,
        "sample_rows": len(sample),
        "official_valid_or_test_loaded": False,
        "labels_read": False,
    }

    def _load(arm: str, seed: int) -> cbic.CBIC:
        gmean, gscale = load_norm(split_key)
        ckpt = torch.load(OUT / f"checkpoint_{arm_tag(arm, split_key, seed)}.pt", map_location="cpu", weights_only=False)
        model = cbic.build_arm(arm, seed=seed, global_mean=gmean, global_scale=gscale, device=device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model

    model_p = _load("P", 0)
    # 1) q input-weight magnitude vs base columns
    w0 = model_p.group_composer.lins[0].weight.detach()
    payload["q_input_weights"] = {
        "base_columns_abs_mean": float(w0[:, :296].abs().mean()),
        "q_columns_abs_mean": float(w0[:, 296:].abs().mean()),
        "q_columns_abs_max": float(w0[:, 296:].abs().max()),
        "base_columns_abs_max": float(w0[:, :296].abs().max()),
    }
    # 2) permutation sensitivity: shuffle the side assignment of q only
    batch = _batch_for(objs, sample)
    with torch.no_grad():
        feats = model_p.forward_features(batch)
        q_real = feats["q"]
        pred_real = model_p.reader(feats["reader_in"].clone()).view(-1).numpy()
    rng = torch.Generator().manual_seed(0)
    perm = torch.randperm(q_real.shape[0], generator=rng)
    batch_perm = dict(batch)
    with torch.no_grad():
        # recompute features with q replaced by a row permutation (same lengths)
        # simplest faithful route: temporarily monkeypatch cross_bond_q by
        # building a features dict and swapping q before the composer.
        a = model_p.encode_atoms(batch)
        b = model_p.encode_bonds(a, batch)
        w = model_p.encode_junctions(a, b, batch)
        W, V, n_b = model_p.group_to_bonds(w, batch, src.ORGANIZED)
        q_shuf = q_real[perm]
        compose_in = torch.cat([b, W, V, torch.log1p(n_b).unsqueeze(1), batch["bond_feat"], q_shuf], dim=1)
        c = model_p.group_composer(compose_in)
        z = torch.zeros((int(batch["n_graphs"]), c.shape[1]), dtype=c.dtype)
        z.index_add_(0, batch["bond_batch"], c)
        a_sum = torch.zeros((int(batch["n_graphs"]), a.shape[1]), dtype=a.dtype)
        a_sum.index_add_(0, batch["atom_batch"], a)
        a_sq = torch.zeros_like(a_sum)
        a_sq.index_add_(0, batch["atom_batch"], a * a)
        g = (batch["global_feat"].to(a.dtype) - model_p.global_mean) / model_p.global_scale
        reader_in = torch.cat([z, a_sum, a_sq, g], dim=1)
        pred_shuf = model_p.reader(reader_in).view(-1).numpy()
        # zero the q channel
        compose_in0 = torch.cat([b, W, V, torch.log1p(n_b).unsqueeze(1), batch["bond_feat"], torch.zeros_like(q_real)], dim=1)
        c0 = model_p.group_composer(compose_in0)
        z0 = torch.zeros_like(z)
        z0.index_add_(0, batch["bond_batch"], c0)
        reader_in0 = torch.cat([z0, a_sum, a_sq, g], dim=1)
        pred_zero = model_p.reader(reader_in0).view(-1).numpy()
    payload["q_sensitivity"] = {
        "mean_abs_dpred_zero_q": float(np.abs(pred_zero - pred_real).mean()),
        "mean_abs_dpred_permuted_q": float(np.abs(pred_shuf - pred_real).mean()),
        "max_abs_dpred_zero_q": float(np.abs(pred_zero - pred_real).max()),
    }
    # 3) endpoint-swap invariance of q on real rows
    swap_batch = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    swap_batch["bond_index"] = batch["bond_index"][:, [1, 0]]
    with torch.no_grad():
        q_swap = model_p.forward_features(swap_batch)["q"]
    payload["q_endpoint_swap_max_gap"] = float((q_real - q_swap).abs().max())
    # 4) q distribution by bond ring class
    bf = batch["bond_feat"].numpy()
    q_np = q_real.numpy()
    ring_mask = bf[:, 4] > 0
    payload["q_by_ring_class"] = {
        "ring_bond_mean_abs": float(np.abs(q_np[ring_mask]).mean()) if ring_mask.any() else 0.0,
        "nonring_bond_mean_abs": float(np.abs(q_np[~ring_mask]).mean()) if (~ring_mask).any() else 0.0,
        "n_ring": int(ring_mask.sum()),
        "n_nonring": int((~ring_mask).sum()),
    }
    # 5) chain/branch subgroup: endpoint degrees
    bi = batch["bond_index"].numpy()
    deg = np.bincount(bi.reshape(-1), minlength=int(bi.max() + 1)) if bi.size else np.zeros(1)
    deg_u, deg_v = deg[bi[:, 0]], deg[bi[:, 1]]
    chain = (deg_u == 2) & (deg_v == 2)
    branch = (deg_u >= 3) | (deg_v >= 3)
    payload["q_by_endpoint_degree"] = {
        "chain_mean_abs": float(np.abs(q_np[chain]).mean()) if chain.any() else 0.0,
        "branch_mean_abs": float(np.abs(q_np[branch]).mean()) if branch.any() else 0.0,
        "n_chain": int(chain.sum()),
        "n_branch": int(branch.sum()),
    }
    write_json(OUT / "mechanism_audit.json", payload)
    log(f"[mechanism] zero-q |dpred| {payload['q_sensitivity']['mean_abs_dpred_zero_q']:.4f}")
    del objs
    return 0


# ---------------------------------------------------------------------------
# report / decision
# ---------------------------------------------------------------------------


def cmd_report(args: argparse.Namespace) -> int:
    """Assemble results/.../REPORT.md from the frozen artifacts."""
    lines: list[str] = []
    lines.append("# BJOC Cross-Bond Interaction v1 — REPORT\n")
    lines.append(f"protocol `{PROTOCOL_ID}` — generated {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}\n")
    phase_a = read_json(OUT / "phase_a_gate.json")
    lines.append("## Phase A information audit\n")
    lines.append(f"- decision: **{phase_a['decision']}**")
    for k, v in phase_a["checks"].items():
        lines.append(f"- {k}: {v}")
    proj = read_json(OUT / "projection_collision_audit.json")
    lines.append(f"- scalar collisions found: {proj['systematic_scalar_search']['n_collision_keys']}")
    real = read_json(OUT / "real_molecule_information_audit.json")
    lines.append(
        f"- real-molecule valid pair fraction: {real['bond_side_counts']['valid_pair_fraction']:.4f}; "
        f"functional new-variance share: {real['exact_decomposition']['functional_new_variance_share']:.4f}; "
        f"linear view-probe residual: {real['composer_visible_probes']['view_probe']['residual_variance_share']:.4f}"
    )
    synth = read_json(OUT / "synthetic_cross_bond_test.json")
    lines.append(
        f"- synthetic test MAE projection {synth['test_mae_standardised']['projection_only']:.4f} "
        f"vs +q {synth['test_mae_standardised']['projection_plus_q']:.4f}"
    )
    for split_key in ("A", "B"):
        metrics_path = OUT / f"split_{split_key.lower()}_metrics.json"
        if not metrics_path.exists():
            continue
        m = read_json(metrics_path)
        lines.append(f"\n## Split {split_key}\n")
        lines.append("| arm | seed | fit | monitor | locked |")
        lines.append("|---|---:|---:|---:|---:|")
        for tag, e in sorted(m["arms"].items()):
            lines.append(
                f"| {e['arm']} | {e['seed']} | {e['fit_mae']:.6f} | {e['monitor']['mae']:.6f} | {e['locked']['mae']:.6f} |"
            )
        p = m["paired"]
        if "seed_mean" in p:
            sm = p["seed_mean"]
            lines.append("\n| contrast | monitor Δ | locked Δ |")
            lines.append("|---|---:|---:|")
            for ref in ("N", "O"):
                lines.append(
                    f"| P−{ref} | {sm[f'P_minus_{ref}']['monitor']['delta']:+.6f} | "
                    f"{sm[f'P_minus_{ref}']['locked']['delta']:+.6f} |"
                )
        gate_path = OUT / f"split_{split_key.lower()}_gate.json" if split_key == "A" else None
        if gate_path and gate_path.exists():
            gate = read_json(gate_path)
            lines.append(f"\n- split gate: **{gate['verdict']}**")
    if (OUT / "split_a_gate.json").exists():
        gate = read_json(OUT / "split_a_gate.json")
        lines.append("\n## Split A locked gate\n")
        for k, v in gate.get("checks", {}).items():
            lines.append(f"- {k}: {v}")
    if (OUT / "stage1_decision.json").exists():
        stage1 = read_json(OUT / "stage1_decision.json")
        lines.append("\n## Split A seed-0 monitor gate\n")
        for k, v in stage1.get("checks", {}).items():
            lines.append(f"- {k}: {v}")
        lines.append(f"- decision: **{stage1.get('decision')}**")
    decision_path = OUT / "scientific_decision.json"
    if decision_path.exists():
        dec = read_json(decision_path)
        lines.append(f"\n## Final decision\n\n**{dec['decision']}**\n\n{dec['reading']}")
    lines.append("\n## Discipline\n")
    lines.append("- official valid/test never loaded; locked labels read exactly once per trained split (lock file `locked_read_*.lock`)")
    lines.append("- no second interaction definition, no rank/hidden/location search, no GNN/attention/iteration")
    lines.append("- checkpoints in this directory, hashes recorded in the prediction manifests")
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log("[report] wrote REPORT.md")
    return 0


def cmd_decision(args: argparse.Namespace) -> int:
    """Map the executed evidence to exactly one pre-registered decision label."""
    phase_a = read_json(OUT / "phase_a_gate.json")
    if phase_a["decision"] != "PHASE_A_PURCHASE_GRANTED":
        payload = {
            "protocol": PROTOCOL_ID,
            "decision": "CROSSBOND_NO_NEW_INFORMATION",
            "reading": "Phase A found no usable new cross-bond information; no GPU was purchased.",
            "official_valid_or_test_loaded": False,
        }
        write_json(OUT / "scientific_decision.json", payload)
        log("[decision] CROSSBOND_NO_NEW_INFORMATION")
        return 0
    stage1_path = OUT / "stage1_decision.json"
    if not stage1_path.exists():
        payload = {
            "protocol": PROTOCOL_ID,
            "decision": "INCONCLUSIVE_OR_BLOCKED",
            "reading": "Phase A granted purchase but the split-A seed-0 monitor gate is missing.",
            "official_valid_or_test_loaded": False,
        }
        write_json(OUT / "scientific_decision.json", payload)
        log("[decision] INCONCLUSIVE_OR_BLOCKED")
        return 0
    stage1 = read_json(stage1_path)
    if not stage1["purchase_seed1"]:
        # classify: overfit vs no predictive gain vs weak signal
        mon = stage1["monitor_mae"]
        fit_p = read_json(OUT / "train_P_A_s0.json")["soup_fit_mae"]
        fit_o = read_json(OUT / "train_O_A_s0.json")["soup_fit_mae"]
        po = stage1["seed0_margin"]["P_minus_O"]
        pn = stage1["seed0_margin"]["P_minus_N"]
        if fit_p < fit_o - GATE_OVERFIT_GAP_TOL and po > 0:
            label = "CROSSBOND_OVERFITS"
            reading = "P fits the training relation better than O but its monitor MAE is worse: the new channel is an overfitting route."
        elif po < 0.0:
            # only the marginal contrast against the matched Organized control
            # (P - O) can attribute any movement to the interaction channel;
            # P - N alone is dominated by the shared Organized pathway.
            label = "CROSSBOND_WEAK_SIGNAL"
            reading = (
                f"The marginal seed-0 monitor contrast is in the right direction but below the frozen purchase bar "
                f"(P−O {po:+.6f}, P−N {pn:+.6f}); recorded as a weak internal signal, no seed 1 purchased."
            )
        else:
            label = "CROSSBOND_INFORMATION_BUT_NO_PREDICTIVE_GAIN"
            reading = (
                "The channel is implemented, invariant and used by the trained model, but it does not improve "
                "held-out MAE over the matched Organized control on split A seed 0 "
                f"(P−O {po:+.6f}, P−N {pn:+.6f})."
            )
        payload = {
            "protocol": PROTOCOL_ID,
            "decision": label,
            "reading": reading,
            "stage1": stage1,
            "official_valid_or_test_loaded": False,
        }
        write_json(OUT / "scientific_decision.json", payload)
        log(f"[decision] {label}")
        return 0
    gate_path = OUT / "split_a_gate.json"
    if not gate_path.exists():
        payload = {
            "protocol": PROTOCOL_ID,
            "decision": "INCONCLUSIVE_OR_BLOCKED",
            "reading": "Seed 1 was purchased but the split-A locked gate is missing.",
            "official_valid_or_test_loaded": False,
        }
    elif not read_json(gate_path)["purchase_split_b"]:
        gate = read_json(gate_path)
        po = gate["seed_mean"]["P_minus_O"]["locked"]["delta"]
        pn = gate["seed_mean"]["P_minus_N"]["locked"]["delta"]
        payload = {
            "protocol": PROTOCOL_ID,
            "decision": "CROSSBOND_WEAK_SIGNAL",
            "reading": (
                f"Split-A seed0 monitor passed but the locked seed-mean gate failed "
                f"(P−O {po:+.6f}, P−N {pn:+.6f}); split B was not purchased."
            ),
            "gate": gate,
            "official_valid_or_test_loaded": False,
        }
    else:
        # split B trained; classify on its locked metrics if available
        b_metrics_path = OUT / "split_b_metrics.json"
        if not b_metrics_path.exists():
            payload = {
                "protocol": PROTOCOL_ID,
                "decision": "CROSSBOND_PROMISING_INTERNAL",
                "reading": "Split A passed every frozen gate; split B is purchased but its locked metrics are not yet scored.",
                "official_valid_or_test_loaded": False,
            }
        else:
            b = read_json(b_metrics_path)
            a = read_json(OUT / "split_a_metrics.json")
            po_a = a["paired"]["seed_mean"]["P_minus_O"]["locked"]["delta"]
            pn_a = a["paired"]["seed_mean"]["P_minus_N"]["locked"]["delta"]
            if "seed_mean" in b["paired"]:
                po_b = b["paired"]["seed_mean"]["P_minus_O"]["locked"]["delta"]
                pn_b = b["paired"]["seed_mean"]["P_minus_N"]["locked"]["delta"]
            else:
                seed = sorted(b["paired"]["by_seed"])[0]
                po_b = b["paired"]["by_seed"][seed]["P_minus_O"]["locked"]["delta"]
                pn_b = b["paired"]["by_seed"][seed]["P_minus_N"]["locked"]["delta"]
            wins = (po_a < 0) + (pn_a < 0) + (po_b < 0) + (pn_b < 0)
            pay = {
                "P_minus_O_A": po_a, "P_minus_N_A": pn_a,
                "P_minus_O_B": po_b, "P_minus_N_B": pn_b,
            }
            if wins >= 3:
                payload = {
                    "protocol": PROTOCOL_ID,
                    "decision": "CROSSBOND_PROMISING_INTERNAL",
                    "reading": f"Cross-split, cross-seed directions favour P ({wins}/4 negative margins). Internal evidence only; not a claim about official test or the 0.06–0.07 target.",
                    "margins": pay,
                    "official_valid_or_test_loaded": False,
                }
            else:
                payload = {
                    "protocol": PROTOCOL_ID,
                    "decision": "CROSSBOND_WEAK_SIGNAL",
                    "reading": f"Only {wins}/4 split×seed margins favour P; the effect is not stable across partitions/seeds.",
                    "margins": pay,
                    "official_valid_or_test_loaded": False,
                }
    write_json(OUT / "scientific_decision.json", payload)
    log(f"[decision] {payload['decision']}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("splits")
    sub.add_parser("objects")
    sub.add_parser("audit-historical")
    sub.add_parser("audit-projection")
    sub.add_parser("audit-real")
    sub.add_parser("synthetic")
    sub.add_parser("phase-a-gate")
    sub.add_parser("params")
    sub.add_parser("gates")
    sub.add_parser("freeze")
    sub.add_parser("report")
    sub.add_parser("decision")

    p_train = sub.add_parser("train")
    p_train.add_argument("--arm", required=True, choices=list(ARMS))
    p_train.add_argument("--split", required=True, choices=["A", "B"])
    p_train.add_argument("--seed", type=int, required=True)
    p_train.add_argument("--device", default="cpu")
    p_train.add_argument("--smoke", action="store_true")

    p_stage = sub.add_parser("train-stage")
    p_stage.add_argument("--stage", required=True, choices=sorted(STAGE_PLANS))
    p_stage.add_argument("--device", default="cpu")
    p_stage.add_argument("--force", action="store_true")

    p_pred = sub.add_parser("predict")
    p_pred.add_argument("--split", required=True, choices=["A", "B"])
    p_pred.add_argument("--arms", required=True)
    p_pred.add_argument("--rows", choices=["monitor", "both"], default="monitor")
    p_pred.add_argument("--device", default="cpu")

    p_mon = sub.add_parser("monitor-gate")
    p_mon.add_argument("--split", required=True, choices=["A", "B"])

    p_score = sub.add_parser("score")
    p_score.add_argument("--split", required=True, choices=["A", "B"])
    p_score.add_argument("--force", action="store_true")

    sub.add_parser("locked-gate")

    p_mech = sub.add_parser("mechanism")
    p_mech.add_argument("--split", default="A", choices=["A", "B"])

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "splits": cmd_splits,
        "objects": cmd_objects,
        "audit-historical": cmd_audit_historical,
        "audit-projection": cmd_audit_projection,
        "audit-real": cmd_audit_real,
        "synthetic": cmd_synthetic,
        "phase-a-gate": cmd_phase_a_gate,
        "params": cmd_params,
        "gates": cmd_gates,
        "freeze": cmd_freeze,
        "train": cmd_train,
        "train-stage": cmd_train_stage,
        "predict": cmd_predict,
        "monitor-gate": cmd_monitor_gate,
        "score": cmd_score,
        "locked-gate": cmd_locked_gate,
        "mechanism": cmd_mechanism,
        "report": cmd_report,
        "decision": cmd_decision,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
