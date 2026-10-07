"""zinc_cssd_basis_reuse_v1 — frozen-CSSD basis reuse across a size domain gap.

Protocol ``zinc-cssd-basis-reuse-v1`` (track ksvd, study zinc-context-gap,
design basis commit 97b414bddaf29f3e2f2691ea9edd820fcca43356).  The round
fits the whole CSSD structural basis package (U / common_rms / D) on a SOURCE
domain (small molecules), freezes it, and asks whether a downstream consumer
learned on a TARGET domain (larger molecules) can use that frozen package,
vs the identical-method package refit on the target domain.  Both arms
share the identical Full446 / M_COMP+Q consumer skeleton, the identical
recipe, the identical shared Q head and identical fit-only calibration;
the ONLY difference is the frozen basis package that produces z.  This is
a reuse diagnostic — not a dictionary-vs-plain-encoding purchase, not SOTA.

The module reuses the CSSD-q1 scheme of
``e2e_dictenv_common_subspace_dictionary_v1`` (radius-2 induced root patches
phi65; K=32, s=8, tied-IHT-10, H1 lambda), the CSSD training math (loss =
L1(pred,y) + H1_LAMBDA*rec, Adam 1e-3/1e-5, clip 5, batch 128, locked shuffle
offsets) and the Full446 / M_COMP+Q consumer skeleton of
``zinc_cssd_nonlinear_binding_v1`` (CSSDBindingFull / arm A), bit-comparable
to the frozen design basis.

Domain split (single axis: n_nodes, the rooted-graph size; label-blind):

* Universe = the design-basis round's fit fold (8001 official-train rows);
  old select/confirm and official valid/test never enter this round.
* t = argmin_t |#{n_nodes<=t} - #{n_nodes>t}| over the universe, ties take
  the smaller t (t=23 -> source_pool 4126 / target_pool 3875).
* Groups are canonical SMILES (identical molecules never split).
* T_eval = prefix of target_pool groups ordered by
  sha256("cssd-reuse-v1-20261006|" + canonical_smiles) ascending, row count
  closest to 25% of target_pool (ties -> shorter prefix) = 969 rows; sealed
  until the one-shot terminal stage.
* N = min(|source_pool|, |T_fit_candidate|) = 2906; S_fit and T_fit each
  select whole groups in the same hash order to exactly N rows (the two
  CSSD refits fit the same number of molecules); remainders are UNUSED,
  never a second eval set (S_fit 2906 / T_fit 2906 / T_eval 969 /
  unused_source 1220 / unused_target 0, frozen by manifest).

Two arms (identical consumer, identical recipe, identical seed init):

* SOURCE — CSSD package fit on S_fit phi rows (K-SVD init + 320-epoch
  task-informed CSSD training, temporary task head discarded); frozen.  The
  consumer is trained from untrained init on T_fit using that frozen basis.
* TARGET — CSSD package fit on T_fit phi rows with the identical method;
  frozen.  The matched control.

Shared by both arms and both body seeds: the Q(topology25) head trained
once on T_fit only; targets/payload/kappa/prep refit on T_fit only; the
body recipe (240 epochs, batch 128, Adam 1e-3/wd 1e-5, clip 5, locked
schedule, last-5 soup, COMP loss g = ell + s, y_raw = ell_hat + s_hat +
Q_raw, single fit-median b_y).  Same-seed SOURCE/TARGET bodies start from
a bit-identical state per tensor (only the frozen dictionary buffers
 differ, asserted in checks); body seeds 0 and 1 are genuinely different
runs.  No select stage, no candidate selection; T_eval is opened once,
after every soup, bias, roster, coverage and intervention script is frozen.

Post-training diagnostics (frozen before the terminal run):

* Coverage — full relative reconstruction of T_eval phi65 under each
  package (phi_hat = U*(common*RMS) + Dbar*alpha), per-molecule root-mean
  then macro median/p95; atom usage counts/mass, N_eff, unused atoms,
  T_fit vs T_eval change.
* The unique dictionary intervention — every CSSD consumption entry's
  alpha_v replaced by the arm's T_fit mean root-code alpha (common c_v,
  original graphs, raw/root-MLP, Sem108 and Q inputs untouched; no
  retraining, no recalibration).  Dependence evidence, never incremental
  benefit.
* Frozen interpretation — per seed M_S/M_T, delta = M_S - M_T (positive =
  reuse worse), R = M_S/M_T; SMILES-group-paired bootstrap (2000 draws,
  fixed seed) per seed and on the two-seed averaged errors; engineering
  tolerance R <= 1.05; frozen conclusion branches A-F.

Usage::

    python -m tracks.ksvd.experiments.luyin16.zinc_cssd_basis_reuse_v1 \
        --write-smiles-table | --build-objects | --cssd-refit --arm SOURCE \
        | --cssd-refit --arm TARGET | --train-q | --checks | --smoke \
        | --train --arm SOURCE --seed 0 | --train --arm TARGET --seed 0 \
        | --train --arm SOURCE --seed 1 | --train --arm TARGET --seed 1 \
        | --terminal-eval
"""

from __future__ import annotations

import argparse
import gzip
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
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_h1_clarity_audit as audit
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p1 as p1
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p2_abs as p2
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_common_subspace_dictionary_v1 as cssd
from tracks.ksvd.experiments.luyin16 import sdb_v0 as sdb
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p2_abs as p2run
from tracks.ksvd.experiments.luyin16 import (
    zinc_local_dictionary_component_supervision_seed0_v1 as zldc,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_local_tuple_dictionary_joint_vs_independent_seed0_v1 as zlt,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_local_tuple_fresh_fold_replication_seed0_v1 as zfr,
)
from tracks.ksvd.experiments.luyin16 import zinc_cssd_nonlinear_binding_v1 as parent
from tracks.ksvd.experiments.luyin16 import zinc_full_cycle_target_decomposition_v1 as zftd
from tracks.ksvd.experiments.luyin16 import zinc_task_dictionary_and_cycle_witness_seed0_v1 as zw
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_v0 as v0

# ---------------------------------------------------------------------------
# frozen utilities re-exported from the design-basis round (bit-comparable)
# ---------------------------------------------------------------------------

seed_everything = parent.seed_everything
state_hash = parent.state_hash
array_sha256 = parent.array_sha256
file_sha256 = parent.file_sha256
write_json = parent.write_json
read_json = parent.read_json
resolve_device = parent.resolve_device
parameter_audit = parent.parameter_audit
topology_matrix = parent.topology_matrix
build_q_head = parent.build_q_head
q_forward = parent.q_forward
build_arm = parent.build_arm
_probe = parent._probe
_soup_predictions = parent._soup_predictions
_q_predictions = parent._q_predictions
_eval_batch_predict = parent._eval_batch_predict
_relabel_nodes = parent._relabel_nodes
_permuted_tuple_payload = parent._permuted_tuple_payload
graphs_with_relabelled_chemistry = parent.graphs_with_relabelled_chemistry
graphs_with_swapped_bond_endpoints = parent.graphs_with_swapped_bond_endpoints
graphs_with_permuted_occurrence_rows = parent.graphs_with_permuted_occurrence_rows
graphs_with_mutated_pair_relation = parent.graphs_with_mutated_pair_relation

# ---------------------------------------------------------------------------
# protocol constants
# ---------------------------------------------------------------------------

PROTOCOL_VERSION = "zinc-cssd-basis-reuse-v1"
RESULT_SLUG = "zinc_cssd_basis_reuse_v1"
TRACK_ROOT = zw.TRACK_ROOT
RESULTS_DIR = TRACK_ROOT / "results" / RESULT_SLUG
RUNS_DIR = RESULTS_DIR / "runs"
PARENT_RESULTS_DIR = parent.RESULTS_DIR

# frozen body recipe — identical to the design-basis round (asserted in checks)
SEED = parent.SEED
EPOCHS = parent.EPOCHS
LR = parent.LR
WEIGHT_DECAY = parent.WEIGHT_DECAY
GRAD_CLIP = parent.GRAD_CLIP
BATCH_SIZE = parent.BATCH_SIZE
SOUP_EPOCHS = parent.SOUP_EPOCHS
LOG_EPOCHS = parent.LOG_EPOCHS
TRAIN_SHUFFLE_OFFSET = parent.TRAIN_SHUFFLE_OFFSET
COMPONENT_LOSS_WEIGHT = parent.COMPONENT_LOSS_WEIGHT

#: the only consumer skeleton of this round (the parent's arm A Full446 path)
SKELETON_ARM = "A"
REUSE_ARMS = ("SOURCE", "TARGET")
#: body seeds of the frozen budget (2 arms x 2 seeds = 4 body runs)
REUSE_SEEDS = (0, 1)

# domain split (single axis: n_nodes within the parent fit fold; label-blind)
SPLIT_HASH_STRING = "cssd-reuse-v1-20261006"
EVAL_FRACTION = 0.25
MIN_FIT_ROWS = 1000
MIN_EVAL_ROWS = 300
#: frozen expected split (re-derived and asserted; label-blind by construction)
EXPECTED_SPLIT = {
    "threshold": 23,
    "universe": 8001,
    "s_pool": 4126,
    "t_pool": 3875,
    "t_eval": 969,
    "t_fit_candidate": 2906,
    "n_common": 2906,
    "s_fit": 2906,
    "t_fit": 2906,
    "s_unused": 1220,
    "t_unused": 0,
}
THRESHOLD_RULE = (
    "t = argmin_t |#{n_nodes<=t} - #{n_nodes>t}| over the parent fit fold "
    "(both pools nonempty); ties take the smaller t"
)
EVAL_PREFIX_RULE = (
    "T_eval = prefix of target_pool canonical-SMILES groups ordered by "
    "sha256('%s|' + canonical_smiles) ascending, row count closest to 25%% "
    "of target_pool; ties take the shorter prefix" % SPLIT_HASH_STRING
)
N_SELECTION_RULE = (
    "N = min(|source_pool|, |T_fit_candidate|); S_fit and T_fit each select "
    "whole groups in the same hash order to exactly N rows; if exact N is "
    "unreachable because of group sizes, the largest common achievable count "
    "<= N is used; remainders are UNUSED and never a second eval set"
)

# CSSD basis refit recipe — identical to the design-basis round
CSSD_Q = parent.CSSD_Q
CSSD_K_ATOMS = parent.CSSD_K_ATOMS
CSSD_SPARSITY = parent.CSSD_SPARSITY
CSSD_IHT_STEPS = parent.CSSD_IHT_STEPS
CSSD_DICT_SEED = parent.CSSD_DICT_SEED
CSSD_KSVD_EPOCHS = parent.CSSD_KSVD_EPOCHS
CSSD_TRAIN_SEED = parent.CSSD_TRAIN_SEED
CSSD_EPOCHS = parent.CSSD_EPOCHS
CSSD_SOUP_EPOCHS = parent.CSSD_SOUP_EPOCHS
H1_LAMBDA = parent.H1_LAMBDA
MONITOR_ROWS = 1024

# shared Q head recipe — identical to the design-basis round, T_fit only
Q_EPOCHS = parent.Q_EPOCHS
Q_HIDDEN = parent.Q_HIDDEN
Q_TOPOLOGY_IN = parent.Q_TOPOLOGY_IN
Q_SOUP_EPOCHS = parent.Q_SOUP_EPOCHS
Q_TRAIN_GEN_BASE = parent.Q_TRAIN_GEN_BASE
Q_PARAMETERS = parent.Q_PARAMETERS

# frozen interpretation (pre-registered before any training)
#: engineering tolerance: 5% relative loss of the SOURCE arm's y_raw MAE
REL_TOL = 1.05
N_BOOT = parent.N_BOOT
BOOT_SEED = parent.BOOT_SEED
#: coverage is severely failed if any code/reconstruction is non-finite or
#: the SOURCE T_eval macro median relative reconstruction error exceeds 1.0
COVERAGE_SEVERE_MEDIAN = 1.0
#: the sparse branch shows a clear response if at least half the T_eval rows
#: move beyond the numerical replay tolerance under the alpha-mean swap
SPARSE_RESPONSE_FRACTION_MIN = 0.5
#: numerical replay tolerance for the intervention response fraction
REPLAY_TOL = 1e-4
#: both arms are "weak" if their two-seed mean T_eval y_raw MAE both exceed 1.0
BOTH_WEAK_MAE = 1.0
#: seed disagreement flag if |R(seed0) - R(seed1)| exceeds this
SEED_DISAGREEMENT_R = 0.10

BASIS_FILES = {arm: f"cssd_basis_{arm}.npz" for arm in REUSE_ARMS}
PHI_DIM = int(p1.PHI_DIM)
COMMON_DIM = int(CSSD_Q)

#: committed canonical-SMILES table for the 10000 official-train rows
SMILES_TABLE = RESULTS_DIR / "train_canonical_smiles.npz"
SMILES_TABLE_PROVENANCE = RESULTS_DIR / "train_canonical_smiles_provenance.json"
#: the (untracked) long-cycle-audit sidecar the table is derived from once
SMI_SIDECARS = TRACK_ROOT / "results/zinc_long_cycle_audit/cache/smi_sidecars.pkl.gz"


# ---------------------------------------------------------------------------
# 1. domain split (n_nodes threshold inside the parent fit fold)
# ---------------------------------------------------------------------------


def _node_sizes() -> np.ndarray:
    """Per-molecule rooted-graph sizes from the train-only env cache."""
    _phi_env, _atom_env, node_sizes_env = zlt._env_phi_atom()
    sizes = np.asarray(node_sizes_env, np.int64)
    if sizes.shape != (10000,):
        raise RuntimeError(f"env cache node_sizes shape {sizes.shape}")
    return sizes


def _smiles_line() -> np.ndarray:
    """The train-label smi_line column (SMILES-duplicate group keys)."""
    import pandas as pd

    lab = pd.read_csv(zfr.TRAIN_LABEL_CSV)
    if len(lab) != 10000:
        raise RuntimeError(f"train label csv has {len(lab)} rows")
    if not np.array_equal(lab["subset_index"].to_numpy(np.int64), np.arange(10000)):
        raise RuntimeError("train label csv rows are not positional")
    smi_line = lab["smi_line"].to_numpy(np.int64)
    if int((smi_line < 0).sum()):
        raise RuntimeError("some train rows have no smiles line")
    return smi_line


# ---------------------------------------------------------------------------
# 1a. canonical SMILES (committed derived table; grouping never by label)
# ---------------------------------------------------------------------------


def write_canonical_smiles_table() -> dict[str, Any]:
    """One-shot local derivation of the committed canonical-SMILES table.

    Reads the (untracked) long-cycle-audit sidecar built from
    ``upstream/zinc250k.smi`` (RDKit canonicalisation inside the audit),
    extracts the canonical SMILES of the 10000 official-train rows by
    ``smi_line`` and stores them positionally.  The table is committed; every
    later stage (local or remote) reads only the committed file.
    """
    import pickle

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    smi_line = _smiles_line()
    with gzip.open(SMI_SIDECARS, "rb") as handle:
        side = pickle.load(handle)
    canon = side["canonical"]
    smiles = [str(canon[int(v)]) for v in smi_line.tolist()]
    if len(smiles) != 10000 or any(not s for s in smiles):
        raise RuntimeError("canonical smiles table derivation failed")
    table = np.asarray(smiles, dtype="<U256")
    np.savez_compressed(SMILES_TABLE, smiles=table)
    smiles_sha = hashlib.sha256("\n".join(smiles).encode("utf-8")).hexdigest()
    provenance = {
        "protocol_version": PROTOCOL_VERSION,
        "stage": "smiles-table",
        "n_rows": 10000,
        "positional": "row i = official-train subset_index i",
        "derived_from": {
            "sidecar": str(SMI_SIDECARS.relative_to(TRACK_ROOT)),
            "sidecar_sha256": file_sha256(SMI_SIDECARS),
            "smi_line_source": str(zfr.TRAIN_LABEL_CSV.relative_to(TRACK_ROOT)),
            "smi_line_source_sha256": file_sha256(zfr.TRAIN_LABEL_CSV),
            "canonicalisation": "RDKit canonical SMILES (zinc_long_cycle_audit sidecar cache)",
        },
        "smiles_sha256": smiles_sha,
        "note": "committed derived artifact; the raw zinc250k.smi is git-ignored",
    }
    write_json(SMILES_TABLE_PROVENANCE, provenance)
    print(f"[smiles-table] wrote {SMILES_TABLE.name} (sha {smiles_sha[:16]}…)")
    return provenance


def _canonical_smiles() -> np.ndarray:
    """Committed canonical SMILES per official-train position."""
    if not SMILES_TABLE.exists():
        raise RuntimeError(
            f"{SMILES_TABLE.name} missing: run --write-smiles-table once locally "
            "(it reads the untracked sidecar cache) and commit it"
        )
    with np.load(SMILES_TABLE, allow_pickle=False) as z:
        smiles = np.asarray(z["smiles"], dtype=object)
    if smiles.shape != (10000,):
        raise RuntimeError(f"canonical smiles table shape {smiles.shape}")
    return np.asarray([str(s) for s in smiles.tolist()], dtype=object)


def _group_hash(smiles: str) -> str:
    return hashlib.sha256((SPLIT_HASH_STRING + "|" + str(smiles)).encode("utf-8")).hexdigest()


def _ordered_groups(indices: np.ndarray, smiles: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Whole canonical-SMILES groups, ordered by (group hash, smiles)."""
    members: dict[str, list[int]] = {}
    for i in np.asarray(indices, np.int64).tolist():
        members.setdefault(str(smiles[i]), []).append(int(i))
    order = sorted(members, key=lambda s: (_group_hash(s), s))
    return [(s, np.sort(np.asarray(members[s], np.int64))) for s in order]


# ---------------------------------------------------------------------------
# 1b. domain split (n_nodes threshold + SMILES-group hash rules; label-blind)
# ---------------------------------------------------------------------------


def build_domain_split() -> dict[str, Any]:
    """S_fit / T_fit / T_eval by the frozen n_nodes + group-hash rules.

    The universe is the design-basis fit fold (canonical-SMILES groups
    already wholly inside it).  Only graph identities are used — never
    labels, errors or Q.
    """
    parent_fold = np.load(PARENT_RESULTS_DIR / "fold.npz")
    universe = np.sort(np.asarray(parent_fold["fit_idx"], np.int64))
    if universe.size < parent.N_FIT:
        raise RuntimeError(f"parent fit fold has only {universe.size} rows")
    node_sizes = _node_sizes()
    smiles = _canonical_smiles()

    # (1) integer threshold minimising |#small - #large| (ties -> smaller t)
    sizes_u = node_sizes[universe]
    threshold: int | None = None
    best_gap: int | None = None
    for t in range(int(sizes_u.min()), int(sizes_u.max())):
        n_small = int((sizes_u <= t).sum())
        if n_small == 0 or n_small == int(universe.size):
            continue
        gap = abs(n_small - int(universe.size - n_small))
        if best_gap is None or gap < best_gap:  # strict '<' keeps the smaller t on ties
            threshold, best_gap = int(t), int(gap)
    if threshold is None:
        raise RuntimeError("no legal threshold found")

    small_mask = node_sizes[universe] <= threshold
    s_pool = np.sort(universe[small_mask]).astype(np.int64)
    t_pool = np.sort(universe[~small_mask]).astype(np.int64)

    # identical canonical SMILES -> identical molecule -> identical n_nodes;
    # verified (same-graph-different-label rows, if any, stay one group)
    for groups in (_ordered_groups(s_pool, smiles), _ordered_groups(t_pool, smiles)):
        for _s, members in groups:
            if int(np.unique(node_sizes[members]).size) != 1:
                raise RuntimeError("a canonical-SMILES group straddles the size threshold")

    # (2) T_eval = prefix of target_pool groups (hash order) closest to 25%
    t_groups = _ordered_groups(t_pool, smiles)
    target_rows = EVAL_FRACTION * float(t_pool.size)
    best_prefix: int | None = None
    best_dist: float | None = None
    cumulative = 0
    for k, (_s, members) in enumerate(t_groups):
        cumulative += int(members.size)
        dist = abs(float(cumulative) - target_rows)
        if best_dist is None or dist < best_dist:  # ties keep the shorter prefix
            best_prefix, best_dist = k + 1, dist
    if best_prefix is None:
        raise RuntimeError("empty target pool")
    t_eval = np.sort(np.concatenate([m for _s, m in t_groups[:best_prefix]])).astype(np.int64)
    t_candidate = np.sort(np.concatenate([m for _s, m in t_groups[best_prefix:]])).astype(np.int64)

    # (3) N-balanced whole-group selection from source_pool and T_fit_candidate
    s_groups = _ordered_groups(s_pool, smiles)
    c_groups = _ordered_groups(t_candidate, smiles)
    n_target = int(min(s_pool.size, t_candidate.size))
    s_cum = {int(c) for c in np.cumsum([int(m.size) for _s, m in s_groups]).tolist()}
    c_cum = {int(c) for c in np.cumsum([int(m.size) for _s, m in c_groups]).tolist()}
    achievable = sorted(c for c in (s_cum & c_cum) if c <= n_target)
    if not achievable:
        raise RuntimeError("no common achievable group count <= N")
    n_common = int(max(achievable))  # exact N when both pools can hit it

    def _prefix_upto(groups: list[tuple[str, np.ndarray]], count: int) -> np.ndarray:
        total = 0
        take: list[np.ndarray] = []
        for _s, members in groups:
            if total >= count:
                break
            take.append(members)
            total += int(members.size)
        if total != count:
            raise RuntimeError(f"prefix selection reached {total} != {count}")
        return np.sort(np.concatenate(take)).astype(np.int64)

    s_fit = _prefix_upto(s_groups, n_common)
    t_fit = _prefix_upto(c_groups, n_common)
    s_unused = np.sort(np.setdiff1d(s_pool, s_fit)).astype(np.int64)
    t_unused = np.sort(np.setdiff1d(t_candidate, t_fit)).astype(np.int64)

    # (4) integrity checks (frozen; failures stop the round with a blocker)
    checks: dict[str, Any] = {
        "universe_is_parent_fit_fold": True,
        "smi_groups_never_split": True,
        "threshold_uses_only_graph_identity": True,
        "disjoint": bool(
            np.intersect1d(s_fit, t_fit).size == 0
            and np.intersect1d(s_fit, t_eval).size == 0
            and np.intersect1d(t_fit, t_eval).size == 0
            and np.intersect1d(s_unused, s_fit).size == 0
            and np.intersect1d(t_unused, t_fit).size == 0
        ),
        "covering": bool(
            np.union1d(s_fit, s_unused).size == s_pool.size
            and np.union1d(np.union1d(t_fit, t_unused), t_eval).size == t_pool.size
            and np.union1d(s_pool, t_pool).size == universe.size
        ),
        "equal_cssd_fit_molecule_counts": bool(s_fit.size == t_fit.size == n_common),
        "min_sizes_ok": bool(
            s_fit.size >= MIN_FIT_ROWS and t_fit.size >= MIN_FIT_ROWS and t_eval.size >= MIN_EVAL_ROWS
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"domain split integrity failed: {checks}")
    derived = {
        "threshold": threshold, "universe": universe.size,
        "s_pool": s_pool.size, "t_pool": t_pool.size,
        "t_eval": t_eval.size, "t_fit_candidate": t_candidate.size,
        "n_common": n_common, "s_fit": s_fit.size, "t_fit": t_fit.size,
        "s_unused": s_unused.size, "t_unused": t_unused.size,
    }
    for key, expected in EXPECTED_SPLIT.items():
        if int(derived[key]) != int(expected):
            raise RuntimeError(f"derived split {key}={derived[key]} != frozen expected {expected}")

    def _dist(idx: np.ndarray) -> dict[str, Any]:
        if idx.size == 0:
            return {"n": 0, "n_nodes_min": None, "n_nodes_median": None,
                    "n_nodes_max": None, "phi_rows": 0}
        sizes = node_sizes[idx]
        return {
            "n": int(idx.size),
            "n_nodes_min": int(sizes.min()), "n_nodes_median": float(np.median(sizes)),
            "n_nodes_max": int(sizes.max()), "phi_rows": int(sizes.sum()),
        }

    def _fingerprint(idx: np.ndarray) -> str:
        return hashlib.sha256(
            "\n".join(sorted(str(smiles[i]) for i in idx.tolist())).encode("utf-8")
        ).hexdigest()

    return {
        "threshold": int(threshold),
        "s_pool_idx": s_pool, "t_pool_idx": t_pool,
        "t_eval_idx": t_eval, "t_fit_idx": t_fit, "s_fit_idx": s_fit,
        "s_unused_idx": s_unused, "t_unused_idx": t_unused,
        "definition": {
            "universe": "the design-basis (zinc_cssd_nonlinear_binding_v1) fit fold; "
                        "old select/confirm and official valid/test never enter this round",
            "threshold_rule": THRESHOLD_RULE,
            "eval_prefix_rule": EVAL_PREFIX_RULE,
            "n_selection_rule": N_SELECTION_RULE,
            "hash_string": SPLIT_HASH_STRING,
            "eval_fraction": EVAL_FRACTION,
            "min_fit_rows": MIN_FIT_ROWS, "min_eval_rows": MIN_EVAL_ROWS,
            "label_blind": True,
        },
        "checks": {**checks, "threshold_gap": int(best_gap), "sizes": derived,
                  "equal_count_note": "the two CSSD refits fit the same number of molecules; "
                                      "source molecules are smaller so phi rows and runtime differ"},
        "distributions": {name: _dist(idx) for name, idx in (
            ("s_fit", s_fit), ("t_fit", t_fit), ("t_eval", t_eval),
            ("s_unused", s_unused), ("t_unused", t_unused),
        )},
        "smiles_fingerprints": {name: _fingerprint(idx) for name, idx in (
            ("s_fit", s_fit), ("t_fit", t_fit), ("t_eval", t_eval),
            ("s_unused", s_unused), ("t_unused", t_unused),
        )},
        "hashes": {
            f"{name}_sha256": array_sha256(idx, np.int64)
            for name, idx in (("s_pool", s_pool), ("t_pool", t_pool), ("t_eval", t_eval),
                              ("t_fit", t_fit), ("s_fit", s_fit),
                              ("s_unused", s_unused), ("t_unused", t_unused))
        },
    }


# ---------------------------------------------------------------------------
# 2. domain objects (targets / payload / kappa_M / prep, all T_fit-refit)
# ---------------------------------------------------------------------------


def _zldc_view(fold: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """The zldc fold view: T_fit is the fit set, T_eval the dev set."""
    return {
        "fit_idx": np.asarray(fold["t_fit_idx"], np.int64),
        "dev_idx": np.asarray(fold["t_eval_idx"], np.int64),
    }


def phase_build_objects(*, out_dir: Path = RESULTS_DIR, log: Any = print) -> dict[str, Any]:
    """Stage 0: domain fold + T_fit-refit targets/payload/kappa_M/prep."""
    torch.set_num_threads(8)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    fold = build_domain_split()
    np.savez_compressed(
        out_dir / "fold.npz",
        s_pool_idx=fold["s_pool_idx"], t_pool_idx=fold["t_pool_idx"],
        t_eval_idx=fold["t_eval_idx"], t_fit_idx=fold["t_fit_idx"],
        s_fit_idx=fold["s_fit_idx"],
        s_unused_idx=fold["s_unused_idx"], t_unused_idx=fold["t_unused_idx"],
    )
    write_json(out_dir / "fold_manifest.json", {
        "protocol_version": PROTOCOL_VERSION,
        "threshold": fold["threshold"],
        "definition": fold["definition"],
        "checks": fold["checks"],
        "distributions": fold["distributions"],
        "smiles_fingerprints": fold["smiles_fingerprints"],
        "hashes": fold["hashes"],
        "smiles_table_sha256": file_sha256(SMILES_TABLE),
    })

    view = _zldc_view(fold)
    targets = zldc.build_new_targets(view)
    t = targets["arrays"]
    np.savez_compressed(
        out_dir / "targets.npz",
        y=t["y"], c=t["c"], g=t["g"], k=t["k"], ell=t["ell"], s=t["s"],
        gid=targets["gid"],
        constants=np.asarray(
            [targets["constants"][name] for name in ("sigma_logP", "sigma_SA", "mu_SA", "sigma_cycle", "mu_cycle")],
            np.float64,
        ),
        constant_names=np.asarray(["sigma_logP", "sigma_SA", "mu_SA", "sigma_cycle", "mu_cycle"]),
        mu_logP=np.asarray(parent.MU_LOGP, np.float64),
    )
    payload = zldc.build_new_payload(view)
    np.savez_compressed(out_dir / "tuple_payload.npz", **payload["arrays"])
    kappa = zldc.compute_kappa_M_new(payload["arrays"])
    write_json(out_dir / "kappa_M.json", kappa)
    train_data = zftd.load_train_only()
    prep_meta = zldc.build_new_prep(train_data, view)
    np.savez_compressed(
        out_dir / "prep.npz",
        patch_fit_mean=prep_meta["patch_fit_mean"], patch_fit_scale=prep_meta["patch_fit_scale"],
        ctx_fit_mean=prep_meta["ctx_fit_mean"], ctx_fit_scale=prep_meta["ctx_fit_scale"],
        anchor_fit_mean=prep_meta["anchor_fit_mean"], anchor_fit_scale=prep_meta["anchor_fit_scale"],
        topo_fit_mean=prep_meta["topo_fit_mean"], topo_fit_scale=prep_meta["topo_fit_scale"],
    )
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "stage": "build-objects",
        "seconds": float(time.perf_counter() - started),
        "fit_set": "T_fit (target domain minus the sealed T_eval group prefix)",
        "sizes": fold["checks"]["sizes"],
        "distributions": fold["distributions"],
        "body_seeds": list(REUSE_SEEDS),
        "target_checks": targets["checks"],
        "payload_checks": payload["checks"],
        "kappa_M": kappa,
        "recipe": {
            "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR, "weight_decay": WEIGHT_DECAY,
            "grad_clip": GRAD_CLIP, "soup_epochs": list(SOUP_EPOCHS),
            "component_loss_weight": COMPONENT_LOSS_WEIGHT,
            "skeleton_arm": SKELETON_ARM,
        },
        "source_constants": {
            "p2run.LEARNING_RATE": float(p2run.LEARNING_RATE),
            "p2run.WEIGHT_DECAY": float(p2run.WEIGHT_DECAY),
            "p2run.GRAD_CLIP": float(p2run.GRAD_CLIP),
            "p2run.BATCH_SIZE": int(p2run.BATCH_SIZE),
            "cm.H1_LAMBDA": float(cm.H1_LAMBDA),
        },
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "artifacts": {
            name: {"sha256": file_sha256(out_dir / name)}
            for name in ("fold.npz", "targets.npz", "tuple_payload.npz", "kappa_M.json", "prep.npz")
        },
    }
    write_json(out_dir / "objects_manifest.json", manifest)
    log(f"[build-objects] done in {manifest['seconds']:.1f}s "
        f"(S_fit={fold['distributions']['s_fit']['n']} T_fit={fold['distributions']['t_fit']['n']} "
        f"T_eval={fold['distributions']['t_eval']['n']} unused={fold['distributions']['s_unused']['n']}+"
        f"{fold['distributions']['t_unused']['n']})")
    return manifest


def load_domain_objects(out_dir: Path = RESULTS_DIR) -> dict[str, Any]:
    """Fold + targets + payload + kappa + prep, hashes verified where frozen."""
    out_dir = Path(out_dir)
    fold_manifest = read_json(out_dir / "fold_manifest.json")
    with np.load(out_dir / "fold.npz", allow_pickle=False) as z:
        fold = {
            key: np.asarray(z[key], np.int64)
            for key in ("s_pool_idx", "t_pool_idx", "t_eval_idx", "t_fit_idx",
                        "s_fit_idx", "s_unused_idx", "t_unused_idx")
        }
    for name in ("s_pool", "t_pool", "t_eval", "t_fit", "s_fit", "s_unused", "t_unused"):
        if array_sha256(fold[f"{name}_idx"], np.int64) != fold_manifest["hashes"][f"{name}_sha256"]:
            raise RuntimeError(f"domain fold {name}_idx sha256 mismatch")
    with np.load(out_dir / "targets.npz", allow_pickle=False) as z:
        targets = {key: np.asarray(z[key]) for key in ("y", "c", "g", "k", "ell", "s", "gid")}
    kappa = read_json(out_dir / "kappa_M.json")
    with np.load(out_dir / "tuple_payload.npz", allow_pickle=False) as z:
        payload_arrays = {key: np.asarray(z[key]) for key in z.files}
    with np.load(out_dir / "prep.npz", allow_pickle=False) as z:
        prep = {key: np.asarray(z[key]) for key in z.files}
    return {
        "fold": fold,
        "fold_manifest": fold_manifest,
        "targets": targets,
        "payload_arrays": payload_arrays,
        "kappa": kappa,
        "prep": prep,
    }


def build_domain_data(objects: Mapping[str, Any]) -> dict[str, list[Any]]:
    """S_fit / T_fit / T_eval data lists (local_mol_id set; prep re-verified)."""
    train_data = zftd.load_train_only()
    if len(train_data) != 10000:
        raise RuntimeError("train-only cache length mismatch")
    view = _zldc_view(objects["fold"])
    prep_meta = zldc.build_new_prep(train_data, view)
    for key, value in objects["prep"].items():
        if key not in prep_meta or not np.array_equal(
            np.asarray(prep_meta[key], np.float32), np.asarray(value, np.float32)
        ):
            raise RuntimeError(f"stored prep mismatch at {key}")
    out: dict[str, list[Any]] = {}
    for name in ("s_fit", "t_fit", "t_eval"):
        idx = np.asarray(objects["fold"][f"{name}_idx"], np.int64)
        rows = [train_data[int(i)] for i in idx.tolist()]
        for position, index in enumerate(idx.tolist()):
            rows[position].local_mol_id = torch.tensor([int(index)], dtype=torch.long)
        out[name] = rows
    return out


def _phi_rows_for(indices: Sequence[int]) -> tuple[np.ndarray, dict[str, Any]]:
    """Per-node phi rows of the given global molecule indices (env cache)."""
    phi_env, _atom_env, node_sizes_env = zlt._env_phi_atom()
    idx = np.asarray(indices, np.int64)
    lo = np.concatenate([[0], np.cumsum(node_sizes_env)])
    rows = np.concatenate([phi_env[lo[i]:lo[i + 1]] for i in idx.tolist()], axis=0)
    checks = {
        "n_molecules": int(idx.size),
        "n_phi_rows": int(rows.shape[0]),
        "expected_rows": int(sum(int(node_sizes_env[i]) for i in idx.tolist())),
        "phi_dim": int(rows.shape[1]),
        "source": "results/e2e_dictenv_p1/cache/env_train.pt (train-only)",
    }
    if checks["n_phi_rows"] != checks["expected_rows"] or checks["phi_dim"] != PHI_DIM:
        raise RuntimeError(f"phi row checks failed: {checks}")
    return rows, checks


# ---------------------------------------------------------------------------
# 3. CSSD basis refit (one per domain; CPU-faithful to the design basis)
# ---------------------------------------------------------------------------


def cssd_refit_domain(
    arm: str,
    *,
    out_dir: Path = RESULTS_DIR,
    threads: int = 8,
    log: Any = print,
) -> dict[str, Any]:
    """Refit the frozen CSSD basis for one arm on its own domain rows.

    SOURCE -> K-SVD init + CSSD training on S_fit (small molecules).
    TARGET -> K-SVD init + CSSD training on T_fit (the target training set).
    Same CSSD recipe as the design-basis round (last-5 D soup, fit-only
    monitoring, CPU); the two refits fit the same number of molecules
    (asserted) and differ only in their input rows.  The task-informed
    training head is temporary: only U / common_rms / D enter the consumer.
    """
    if arm not in REUSE_ARMS:
        raise ValueError(arm)
    torch.set_num_threads(int(threads))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    objects = load_domain_objects(out_dir)
    fold_key = "s_fit_idx" if arm == "SOURCE" else "t_fit_idx"
    data_key = "s_fit" if arm == "SOURCE" else "t_fit"
    fit_idx = np.asarray(objects["fold"][fold_key], np.int64)
    other_idx = np.asarray(objects["fold"]["t_fit_idx" if arm == "SOURCE" else "s_fit_idx"], np.int64)
    if fit_idx.size != other_idx.size:
        raise RuntimeError("SOURCE/TARGET CSSD refits must fit the same number of molecules")
    started = time.perf_counter()

    data = build_domain_data(objects)
    fit_data = data[data_key]
    y_fit = np.asarray(objects["targets"]["y"], np.float64)[fit_idx]
    for position, row in enumerate(fit_data):
        if abs(float(getattr(row, "y", y_fit[position])) - float(y_fit[position])) > 1e-12:
            raise RuntimeError("fit data .y != y_stored on fit rows")

    phi_rows, phi_checks = _phi_rows_for(fit_idx)
    D_init, ksvd_meta = sdb.fit_ksvd(
        phi_rows, atoms=CSSD_K_ATOMS, s=CSSD_SPARSITY,
        epochs=CSSD_KSVD_EPOCHS, seed=CSSD_DICT_SEED,
    )
    subspace = cssd.build_common_subspace(phi_rows, q=CSSD_Q)

    monitor_rows = min(MONITOR_ROWS, len(fit_data))
    monitor_data = fit_data[:monitor_rows]
    eval_loader = p1.make_env_loader(
        monitor_data, int(p2run.BATCH_SIZE), False, int(CSSD_TRAIN_SEED) + int(p2run.EVAL_SHUFFLE_OFFSET)
    )

    seed_everything(CSSD_TRAIN_SEED)
    model = cssd.build_cssd_model(np.asarray(D_init, np.float32), CSSD_TRAIN_SEED, subspace)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(p2run.LEARNING_RATE), weight_decay=float(p2run.WEIGHT_DECAY))
    loader = p1.make_env_loader(
        fit_data, int(p2run.BATCH_SIZE), True, int(CSSD_TRAIN_SEED) + int(p2run.TRAIN_SHUFFLE_OFFSET)
    )
    mask = cm.C6_MASK
    lam = float(cm.H1_LAMBDA)
    from tracks.ksvd.code import tccd_v0 as v0

    device = torch.device("cpu")
    curve: list[dict[str, Any]] = []
    epoch_D: dict[int, torch.Tensor] = {}
    soup_epochs = set(int(e) for e in CSSD_SOUP_EPOCHS)
    for epoch in range(1, CSSD_EPOCHS + 1):
        model.train()
        task_sum = rec_sum = rec_term_sum = 0.0
        n_mol = n_nodes = n_batches = 0
        for batch in loader:
            batch = batch.to(device)
            prediction, aux = model(batch, mask=mask, return_aux=True)
            rec = model.reconstruction_loss(aux["phi"], aux["coord"])
            loss = F.l1_loss(prediction.view(-1), batch.y.view(-1)) + lam * rec
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(p2run.GRAD_CLIP))
            optimizer.step()
            task_sum += float((prediction.view(-1) - batch.y.view(-1)).abs().sum())
            n_mol += int(batch.y.numel())
            rec_term_sum += float(rec.detach())
            n_batches += 1
            phi = aux["phi"]
            phi_hat = model.reconstruct(phi, aux["coord"]).detach()
            rec_sum += float((((phi - phi_hat) ** 2).sum(dim=1) / ((phi**2).sum(dim=1) + v0.EPS)).sum())
            n_nodes += int(phi.shape[0])
        train_mae = float(task_sum / max(n_mol, 1))
        train_rec = float(rec_sum / max(n_nodes, 1))
        valid = audit._evaluate_model(model, eval_loader, device, mask)
        curve.append({
            "epoch": int(epoch),
            "train_mae": train_mae,
            "train_rec": train_rec,
            "train_rec_term": float(rec_term_sum / max(n_batches, 1)),
            "fit_monitor_mae": float(valid["mae"]),
            "d_norm": float(model.D.detach().norm()),
        })
        if epoch in soup_epochs:
            epoch_D[int(epoch)] = model.D.detach().cpu().clone()
        if log and (epoch == 1 or epoch % 20 == 0 or epoch == CSSD_EPOCHS):
            log(f"[cssd-refit {arm}] ep={epoch:03d} train={train_mae:.6f} rec={train_rec:.3e} "
                f"fit_monitor={float(valid['mae']):.6f} d_norm={curve[-1]['d_norm']:.4f}", flush=True)

    members = sorted(soup_epochs)
    if sorted(epoch_D) != members:
        raise RuntimeError("cssd soup epochs missing")
    D_frozen = torch.stack([epoch_D[e] for e in members]).mean(0)
    U = np.asarray(subspace.components, np.float32)
    common_rms = np.asarray(subspace.rms, np.float32)
    basis = {
        "U": U,
        "common_rms": common_rms,
        "D": D_frozen.numpy().astype(np.float32),
    }
    np.savez_compressed(out_dir / BASIS_FILES[arm], **basis)
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "stage": "cssd-refit",
        "arm": arm,
        "domain_rows": data_key,
        "seconds": float(time.perf_counter() - started),
        "device": "cpu",
        "threads": int(threads),
        "scheme": {
            "q": CSSD_Q, "k_atoms": CSSD_K_ATOMS, "sparsity": CSSD_SPARSITY,
            "iht_steps": CSSD_IHT_STEPS, "h1_lambda": H1_LAMBDA,
            "epochs": CSSD_EPOCHS, "soup": "mean of D over epochs 316..320 (never selected on holdout)",
            "ksvd": {"epochs": CSSD_KSVD_EPOCHS, "seed": CSSD_DICT_SEED, "meta": ksvd_meta},
            "train_seed": CSSD_TRAIN_SEED,
            "identical_to_design_basis_recipe": True,
            "task_head": "temporary task-informed training head; discarded (never enters the consumer)",
        },
        "phi_checks": phi_checks,
        "n_fit_molecules": int(fit_idx.size),
        "monitor_rows": int(monitor_rows),
        "monitor_note": "informational only; the soup is the last-5 average, never selected by the monitor",
        "U_sha256": array_sha256(U),
        "common_rms_sha256": array_sha256(common_rms),
        "D_init_sha256": array_sha256(np.asarray(D_init, np.float32)),
        "D_frozen_sha256": array_sha256(D_frozen.numpy()),
        "fit_monitor_mae_last": curve[-1]["fit_monitor_mae"],
        "curve_sha256": _hash_curve(curve),
        "fit_only": True,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(out_dir / f"cssd_refit_{arm}.json", manifest)
    write_json(out_dir / f"cssd_refit_{arm}_curve.json", curve)
    log(f"[cssd-refit {arm}] done in {manifest['seconds']:.1f}s "
        f"({phi_checks['n_phi_rows']} phi rows, D sha={manifest['D_frozen_sha256'][:16]}…)")
    return manifest


def _hash_curve(curve: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(json.dumps(list(curve)).encode("utf-8")).hexdigest()


def load_basis(arm: str, out_dir: Path = RESULTS_DIR) -> dict[str, np.ndarray]:
    if arm not in REUSE_ARMS:
        raise ValueError(arm)
    out_dir = Path(out_dir)
    manifest = read_json(out_dir / f"cssd_refit_{arm}.json")
    with np.load(out_dir / BASIS_FILES[arm], allow_pickle=False) as z:
        basis = {key: z[key] for key in ("U", "common_rms", "D")}
    for key, name in (("U_sha256", "U"), ("common_rms_sha256", "common_rms"), ("D_frozen_sha256", "D")):
        if array_sha256(basis[name]) != manifest[key]:
            raise RuntimeError(f"cssd basis {arm} {name} sha256 mismatch")
    return basis


# ---------------------------------------------------------------------------
# 4. shared Q head (topology25 -> c, T_fit only, one shared hash)
# ---------------------------------------------------------------------------


def train_q(*, out_dir: Path = RESULTS_DIR, log: Any = print) -> dict[str, Any]:
    """One shared Q(topology25->64->32->1) on the T_fit rows only."""
    torch.set_num_threads(8)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    objects = load_domain_objects(out_dir)
    data = build_domain_data(objects)
    fit_data = data["t_fit"]
    fit_idx = np.asarray(objects["fold"]["t_fit_idx"], np.int64)
    c = np.asarray(objects["targets"]["c"], np.float64)
    c_fit = c[fit_idx]
    T = topology_matrix(fit_data)
    if T.shape != (len(fit_data), Q_TOPOLOGY_IN):
        raise RuntimeError(f"T_fit topology matrix shape {T.shape}")
    bias_value = float(np.median(c_fit))

    head = build_q_head(SEED, bias_value)
    init_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
    optimizer = torch.optim.Adam(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    generator = torch.Generator().manual_seed(int(Q_TRAIN_GEN_BASE) + SEED)
    T_t = torch.as_tensor(T, dtype=torch.float32)
    c_t = torch.as_tensor(c_fit, dtype=torch.float32)
    n = int(T_t.shape[0])
    soup: dict[int, dict[str, torch.Tensor]] = {}
    soup_epochs = set(int(e) for e in Q_SOUP_EPOCHS)
    curve: list[dict[str, Any]] = []
    for epoch in range(1, Q_EPOCHS + 1):
        head.train()
        order = torch.randperm(n, generator=generator)
        abs_sum = 0.0
        grad_norm = 0.0
        for start in range(0, n, BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            prediction = q_forward(head, T_t[idx])
            loss = (prediction - c_t[idx]).abs().mean()
            optimizer.zero_grad()
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(head.parameters(), GRAD_CLIP))
            optimizer.step()
            abs_sum += float((prediction - c_t[idx]).abs().sum())
        if epoch in soup_epochs:
            soup[epoch] = {k: v.detach().clone() for k, v in head.state_dict().items()}
        curve.append({
            "epoch": int(epoch),
            "train_task_mae": float(abs_sum / n),
            "grad_norm": grad_norm,
        })
        if log and (epoch == 1 or epoch % 50 == 0 or epoch == Q_EPOCHS):
            log(f"[train-q] ep={epoch:03d} L1={curve[-1]['train_task_mae']:.6f} "
                f"gnorm={grad_norm:.3g}", flush=True)
    members = sorted(soup)
    soup_state = {key: torch.stack([soup[e][key].float() for e in members]).mean(0) for key in soup[members[0]]}
    last_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    torch.save(init_state, out_dir / "Q_init_state.pt")
    torch.save(last_state, out_dir / "Q_last_state.pt")
    torch.save(soup_state, out_dir / "Q_soup_state.pt")
    n_params = int(sum(p.numel() for p in head.parameters()))
    if n_params != Q_PARAMETERS:
        raise RuntimeError(f"Q parameter audit failed: {n_params}")
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "stage": "train-q",
        "seconds": float(time.perf_counter() - started),
        "device": "cpu",
        "fit_set": "T_fit only (shared by SOURCE and TARGET arms and both body seeds)",
        "n_fit": int(len(fit_data)),
        "q_parameters": n_params,
        "bias_value": bias_value,
        "epochs": Q_EPOCHS,
        "soup_epochs": list(Q_SOUP_EPOCHS),
        "train_gen_seed": int(Q_TRAIN_GEN_BASE) + SEED,
        "init_state_sha256": state_hash(init_state),
        "soup_state_sha256": state_hash(soup_state),
        "last_train_mae": curve[-1]["train_task_mae"],
        "fit_only": True,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(out_dir / "Q_meta.json", manifest)
    log(f"[train-q] done in {manifest['seconds']:.1f}s; soup sha={manifest['soup_state_sha256'][:16]}…")
    return manifest


def load_q_soup(out_dir: Path = RESULTS_DIR) -> dict[str, torch.Tensor]:
    out_dir = Path(out_dir)
    manifest = read_json(out_dir / "Q_meta.json")
    state = torch.load(out_dir / "Q_soup_state.pt", map_location="cpu", weights_only=False)
    if state_hash(state) != manifest["soup_state_sha256"]:
        raise RuntimeError("Q soup state hash mismatch")
    return state


# ---------------------------------------------------------------------------
# 5. body training (one arm, one seed; identical skeleton/recipe/seed init)
# ---------------------------------------------------------------------------


def run_dir(arm: str, seed: int, out_dir: Path = RESULTS_DIR) -> Path:
    if arm not in REUSE_ARMS:
        raise ValueError(arm)
    return Path(out_dir) / "runs" / f"{arm}_s{int(seed)}"


def train_reuse_arm(
    arm: str,
    *,
    seed: int = SEED,
    device_name: str = "cuda:0",
    out_dir: Path = RESULTS_DIR,
    epochs: int = EPOCHS,
    max_steps: int | None = None,
    log: Any = print,
) -> dict[str, Any]:
    """One reuse-arm body run: arm-A skeleton, the arm's frozen basis, T_fit.

    Identical to the design-basis ``train_arm`` in every tensor of the recipe
    (schedule, init stream, loss, soup, calibration); the only difference
    between the SOURCE and TARGET runs is the frozen basis package and the
    run directory.  T_eval is never touched here.
    """
    if arm not in REUSE_ARMS:
        raise ValueError(arm)
    device = resolve_device(device_name)
    torch.set_num_threads(8 if device.type == "cpu" else 4)
    out_dir = Path(out_dir)
    rdir = run_dir(arm, seed, out_dir)
    rdir.mkdir(parents=True, exist_ok=True)
    objects = load_domain_objects(out_dir)
    basis = load_basis(arm, out_dir)
    q_soup = load_q_soup(out_dir)
    payload = zlt.TuplePayload(objects["payload_arrays"])
    kappa_M = float(objects["kappa"]["kappa_M"])
    fit_idx = np.asarray(objects["fold"]["t_fit_idx"], np.int64)
    data = build_domain_data(objects)
    fit_data = data["t_fit"]
    g = np.asarray(objects["targets"]["g"], np.float64)
    ell = np.asarray(objects["targets"]["ell"], np.float64)
    s = np.asarray(objects["targets"]["s"], np.float64)
    y_all = np.asarray(objects["targets"]["y"], np.float64)
    gid_all = np.asarray(objects["targets"]["gid"], np.int64)
    g_fit, ell_fit, s_fit, y_fit = g[fit_idx], ell[fit_idx], s[fit_idx], y_all[fit_idx]
    if float(np.max(np.abs(g_fit - (ell_fit + s_fit)))) > 1e-12:
        raise RuntimeError("g != ell + s on fit rows")

    target_g = torch.as_tensor(g_fit, dtype=torch.float32)
    target_ell = torch.as_tensor(ell_fit, dtype=torch.float32, device=device)
    target_s = torch.as_tensor(s_fit, dtype=torch.float32, device=device)
    schedule, schedule_hash = zw.build_schedule(len(fit_data), int(epochs), int(seed) + TRAIN_SHUFFLE_OFFSET)

    seed_everything(int(seed))
    model = build_arm(SKELETON_ARM, payload, kappa_M, basis, int(seed))
    init_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model = model.to(device)
    frozen_reference = model.frozen_basis_hashes()
    expected_basis = {
        "U": array_sha256(basis["U"]),
        "common_rms": array_sha256(basis["common_rms"]),
        "D": array_sha256(basis["D"]),
    }
    if frozen_reference != expected_basis:
        raise RuntimeError(f"model basis != loaded frozen basis for arm {arm}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    soup_epochs = set(int(e) for e in SOUP_EPOCHS)
    soup: dict[int, dict[str, torch.Tensor]] = {}
    checkpoints: dict[int, dict[str, torch.Tensor]] = {}
    curve: list[dict[str, Any]] = []
    probes: list[dict[str, Any]] = []
    steps_done = 0
    stopped_reason = "completed"
    started = time.perf_counter()
    peak_mb = 0.0

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
                probes.append(_probe(model, SKELETON_ARM, epoch, steps_done + 1, total_norm, frozen_reference))
                probes[-1]["pred_shift"] = {
                    "l_g": float(l_g.detach()), "l_ell": float(l_ell.detach()), "l_s": float(l_s.detach()),
                }
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
        if epoch in LOG_EPOCHS:
            checkpoints[int(epoch)] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch in soup_epochs:
            soup[int(epoch)] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if log and (epoch in LOG_EPOCHS or epoch % 40 == 0):
            c = curve[-1]
            log(f"[train {arm} s{seed}] ep={epoch:03d} Lg={c['train_L_g']:.5f} "
                f"Lell={c['train_L_ell']:.5f} Ls={c['train_L_s']:.5f} "
                f"gn={c['grad_norm']:.3g} {c['seconds']:.1f}s", flush=True)
        if stopped_reason == "max_steps":
            break

    members = sorted(soup)
    if sorted(soup) != [int(e) for e in SOUP_EPOCHS]:
        raise RuntimeError(f"soup epochs missing: {members}")
    soup_state = {key: torch.stack([soup[e][key].float() for e in members]).mean(0) for key in soup[members[0]]}
    soup_hash = state_hash(soup_state)
    init_hash = state_hash(init_state)

    preds = _soup_predictions(model, soup_state, fit_data, device)
    h = preds["h"]
    T_topology = topology_matrix(fit_data)
    q_raw = _q_predictions(q_soup, T_topology, device)
    y_raw = preds["ell_hat"] + preds["s_hat"] + q_raw
    if float(np.max(np.abs(y_raw - (h + q_raw)))) > 1e-5:
        raise RuntimeError("y_raw != h + Q_raw")
    b_g = float(np.median(g_fit - h))
    b_y = float(np.median(y_fit - y_raw))
    g_cal_mae = float(np.mean(np.abs(g_fit - (h + b_g))))

    torch.save(init_state, rdir / "init_state.pt")
    torch.save(soup_state, rdir / "soup_state.pt")
    for epoch, state in checkpoints.items():
        torch.save(state, rdir / f"epoch{epoch}_state.pt")
    np.savez_compressed(
        rdir / "fit_predictions.npz",
        gid=gid_all[fit_idx], y=y_fit, g=g_fit, ell=ell_fit, s=s_fit,
        h=h, ell_hat=preds["ell_hat"], s_hat=preds["s_hat"], q_raw=q_raw, y_raw=y_raw,
    )
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "stage": "train",
        "arm": arm,
        "skeleton_arm": SKELETON_ARM,
        "basis_arm": arm,
        "seed": int(seed),
        "run_dir": str(rdir.relative_to(Path(out_dir))),
        "recipe": {
            "epochs": int(epochs), "batch_size": BATCH_SIZE, "lr": LR, "weight_decay": WEIGHT_DECAY,
            "grad_clip": GRAD_CLIP, "soup_epochs": list(SOUP_EPOCHS), "log_epochs": list(LOG_EPOCHS),
            "component_loss_weight": COMPONENT_LOSS_WEIGHT,
            "train_shuffle_offset": TRAIN_SHUFFLE_OFFSET,
        },
        "steps_done": int(steps_done),
        "stopped_reason": stopped_reason,
        "schedule_sha256": schedule_hash,
        "init_state_sha256": init_hash,
        "soup_state_sha256": soup_hash,
        "frozen_basis_hashes": frozen_reference,
        "q_soup_sha256": state_hash(q_soup),
        "parameter_audit": parameter_audit(model),
        "calibration": {"b_g": b_g, "b_y": b_y, "note": "single fit-median bias; never stacked"},
        "fit_diagnostics": {
            "fit_L_g_raw_mae": float(np.mean(np.abs(g_fit - h))),
            "fit_L_g_cal_mae": g_cal_mae,
            "fit_y_raw_mae": float(np.mean(np.abs(y_fit - y_raw))),
            "fit_y_cal_mae": float(np.mean(np.abs(y_fit - (y_raw + b_y)))),
            "fit_q_mae": float(np.mean(np.abs(objects["targets"]["c"][fit_idx] - q_raw))),
        },
        "curve_seconds_total": float(sum(c["seconds"] for c in curve)),
        "peak_gpu_memory_mb": float(peak_mb),
        "flops_note": "not measured; both arms share the identical skeleton so op counts match by construction",
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(rdir / "curve.json", curve)
    write_json(rdir / "probes.json", probes)
    write_json(rdir / "manifest.json", manifest)
    log(f"[train {arm} s{seed}] done: {steps_done} steps, {manifest['curve_seconds_total']:.0f}s "
        f"train, fit y_raw MAE {manifest['fit_diagnostics']['fit_y_raw_mae']:.5f} "
        f"b_y={b_y:.5f} soup sha={soup_hash[:12]}…")
    return manifest


def load_run(arm: str, seed: int, out_dir: Path = RESULTS_DIR) -> dict[str, Any]:
    rdir = run_dir(arm, seed, out_dir)
    manifest = read_json(rdir / "manifest.json")
    soup_state = torch.load(rdir / "soup_state.pt", map_location="cpu", weights_only=False)
    if state_hash(soup_state) != manifest["soup_state_sha256"]:
        raise RuntimeError(f"soup state hash mismatch for {arm} s{seed}")
    return {"manifest": manifest, "soup_state": soup_state, "run_dir": rdir}


# ---------------------------------------------------------------------------
# 6. coverage, the unique dictionary intervention, one-shot terminal evaluation
# ---------------------------------------------------------------------------


def _frozen_code_parts(basis: Mapping[str, np.ndarray]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """U / common_rms / normalized residual dictionary Dbar of a basis."""
    U = torch.as_tensor(np.asarray(basis["U"]), dtype=torch.float32)
    D = torch.as_tensor(np.asarray(basis["D"]), dtype=torch.float32)
    Dbar = v0.normalized_dictionary(D - U @ (U.t() @ D))
    return U, torch.as_tensor(np.asarray(basis["common_rms"]), dtype=torch.float32), Dbar


def _root_codes(basis: Mapping[str, np.ndarray], phi_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-root z (c~ | alpha), full phi_hat, per-root relative error.

    Pure frozen-basis function of the phi rows (no training state): the same
    code/reconstruct math the consumer uses (``cssd`` code + reconstruct,
    phi_hat = U*(common*RMS) + Dbar*alpha).
    """
    U, common_rms, Dbar = _frozen_code_parts(basis)
    phi = torch.as_tensor(np.asarray(phi_rows), dtype=torch.float32)
    c = phi @ U
    r = phi - c @ U.t()
    alpha = v0.tied_iht_codes(Dbar, r, s=CSSD_SPARSITY, steps=CSSD_IHT_STEPS)
    z = torch.cat([c / common_rms, alpha], dim=1)
    phi_hat = (z[:, : int(common_rms.shape[0])] * common_rms) @ U.t() + alpha @ Dbar.t()
    rel = (phi - phi_hat).norm(dim=1) / (phi.norm(dim=1) + v0.EPS)
    return (
        z.numpy().astype(np.float64),
        phi_hat.numpy().astype(np.float64),
        rel.numpy().astype(np.float64),
    )


def coverage_diagnostics(*, out_dir: Path = RESULTS_DIR, log: Any = print) -> dict[str, Any]:
    """Frozen coverage diagnostics of each basis package (T_fit vs T_eval).

    Full relative reconstruction of the phi65 patches (per molecule: mean
    over roots of ||phi - phi_hat|| / (||phi|| + eps); macro median/p95),
    atom usage counts/mass, N_eff, unused atoms and finiteness — on the
    T_fit rows and the sealed T_eval rows.  Health is not task gain; no
    success threshold is attached to N_eff.
    """
    out_dir = Path(out_dir)
    objects = load_domain_objects(out_dir)
    node_sizes = _node_sizes()
    arms: dict[str, Any] = {}
    for arm in REUSE_ARMS:
        basis = load_basis(arm, out_dir)
        arm_out: dict[str, Any] = {}
        for name in ("t_fit", "t_eval"):
            idx = np.asarray(objects["fold"][f"{name}_idx"], np.int64)
            phi_rows, checks = _phi_rows_for(idx)
            z, phi_hat, rel = _root_codes(basis, phi_rows)
            alpha = z[:, COMMON_DIM:]
            sizes = node_sizes[idx]
            bounds = np.concatenate([[0], np.cumsum(sizes)])
            mol_err = np.asarray([rel[bounds[i]:bounds[i + 1]].mean() for i in range(idx.size)])
            usage_count = (alpha != 0).sum(axis=0)
            usage_mass = np.abs(alpha).sum(axis=0)
            arm_out[name] = {
                "n_molecules": int(idx.size),
                "n_phi_rows": int(rel.size),
                "phi_rows_check": checks,
                "rel_err_macro_median": float(np.median(mol_err)),
                "rel_err_macro_p95": float(np.percentile(mol_err, 95)),
                "rel_err_row_median": float(np.median(rel)),
                "atom_usage_count": [int(v) for v in usage_count.tolist()],
                "atom_usage_mass": [float(v) for v in usage_mass.tolist()],
                "n_eff": float(usage_mass.sum() ** 2 / max(float((usage_mass ** 2).sum()), 1e-30)),
                "unused_atoms": int((usage_count == 0).sum()),
                "codes_finite": bool(
                    np.isfinite(z).all() and np.isfinite(phi_hat).all() and np.isfinite(rel).all()
                ),
            }
        fit, ev = arm_out["t_fit"], arm_out["t_eval"]
        arm_out["t_fit_to_t_eval_change"] = {
            "rel_err_macro_median": float(ev["rel_err_macro_median"] - fit["rel_err_macro_median"]),
            "unused_atoms": int(ev["unused_atoms"] - fit["unused_atoms"]),
            "n_eff": float(ev["n_eff"] - fit["n_eff"]),
        }
        arms[arm] = arm_out
    severe = {
        arm: bool(
            not arms[arm]["t_eval"]["codes_finite"]
            or arms[arm]["t_fit"]["codes_finite"] is False
            or (arm == "SOURCE" and arms[arm]["t_eval"]["rel_err_macro_median"] > COVERAGE_SEVERE_MEDIAN)
        )
        for arm in REUSE_ARMS
    }
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "stage": "coverage",
        "definition": (
            "full relative reconstruction phi_hat = U*(common*RMS) + Dbar*alpha; "
            "per molecule mean over roots of ||phi-phi_hat||/(||phi||+eps), macro median/p95; "
            "atom usage = nonzero alpha entries per atom (count / |alpha| mass); "
            "N_eff = mass^2/sum mass^2; no success threshold on N_eff"
        ),
        "severe_rule": (
            "severe if any code/reconstruction is non-finite, or the SOURCE T_eval "
            f"macro median relative reconstruction error exceeds {COVERAGE_SEVERE_MEDIAN}"
        ),
        "arms": arms,
        "severe": {**severe, "any": bool(any(severe.values()))},
        "note": (
            "residual-only reconstruction/energy metrics are not reported: the two "
            "arms have different U so the residual denominators differ and residual "
            "metrics cannot rank the packages directly; health is not task gain"
        ),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(out_dir / "coverage.json", payload)
    for arm in REUSE_ARMS:
        if log:
            log(f"[coverage {arm}] T_eval rel-err median "
                f"{arms[arm]['t_eval']['rel_err_macro_median']:.4f} p95 "
                f"{arms[arm]['t_eval']['rel_err_macro_p95']:.4f} unused_atoms "
                f"{arms[arm]['t_eval']['unused_atoms']} N_eff {arms[arm]['t_eval']['n_eff']:.2f}")
    return payload


@torch.no_grad()
def _predict_rows(model: Any, eval_rows: Sequence[Any], device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Batched A-path predictions: h, ell_hat, s_hat (identical mask)."""
    hs, ells, ss = [], [], []
    for start in range(0, len(eval_rows), BATCH_SIZE):
        batch = p1.env_collate(eval_rows[start:start + BATCH_SIZE]).to(device)
        prediction = model(batch, mask=cm.C6_MASK)
        components = model.reader.components()
        hs.append(prediction.detach().cpu().numpy().astype(np.float64))
        ells.append(components[:, 0].detach().cpu().numpy().astype(np.float64))
        ss.append(components[:, 1].detach().cpu().numpy().astype(np.float64))
    return np.concatenate(hs), np.concatenate(ells), np.concatenate(ss)


@torch.no_grad()
def evaluate_arm(
    arm: str,
    seed: int,
    *,
    out_dir: Path = RESULTS_DIR,
    device_name: str = "cuda:0",
    log: Any = print,
) -> dict[str, Any]:
    """T_eval predictions for one trained arm/seed (one-shot terminal stage)."""
    if arm not in REUSE_ARMS:
        raise ValueError(arm)
    device = resolve_device(device_name)
    torch.set_num_threads(8 if device.type == "cpu" else 4)
    out_dir = Path(out_dir)
    objects = load_domain_objects(out_dir)
    basis = load_basis(arm, out_dir)
    q_soup = load_q_soup(out_dir)
    payload = zlt.TuplePayload(objects["payload_arrays"])
    kappa_M = float(objects["kappa"]["kappa_M"])
    run = load_run(arm, seed, out_dir)
    soup_state = run["soup_state"]

    model = build_arm(SKELETON_ARM, payload, kappa_M, basis, int(seed))
    model.load_state_dict({k: v for k, v in soup_state.items()})
    model = model.to(device)
    model.eval()

    data = build_domain_data(objects)
    t_eval_data = data["t_eval"]
    eval_idx = np.asarray(objects["fold"]["t_eval_idx"], np.int64)
    h, ell_hat, s_hat = _predict_rows(model, t_eval_data, device)
    T_topology = topology_matrix(t_eval_data)
    q_raw = _q_predictions(q_soup, T_topology, device)
    y_raw = ell_hat + s_hat + q_raw
    b_y = float(run["manifest"]["calibration"]["b_y"])
    y_cal = y_raw + b_y

    targets = objects["targets"]
    y = np.asarray(targets["y"], np.float64)[eval_idx]
    g = np.asarray(targets["g"], np.float64)[eval_idx]
    ell = np.asarray(targets["ell"], np.float64)[eval_idx]
    s = np.asarray(targets["s"], np.float64)[eval_idx]
    gid = np.asarray(targets["gid"], np.int64)[eval_idx]
    if float(np.max(np.abs(y_raw - (h + q_raw)))) > 1e-5:
        raise RuntimeError("y_raw != h + Q_raw on eval")
    result = {
        "arm": arm, "seed": int(seed),
        "n": int(len(t_eval_data)),
        "gid": gid, "y": y, "g": g, "y_raw": y_raw, "y_cal": y_cal,
        "h": h, "ell_hat": ell_hat, "s_hat": s_hat, "q_raw": q_raw,
        "mae": {
            "y_raw": float(np.mean(np.abs(y - y_raw))),
            "y_cal": float(np.mean(np.abs(y - y_cal))),
            "g_raw": float(np.mean(np.abs(g - h))),
            "ell": float(np.mean(np.abs(ell - ell_hat))),
            "s": float(np.mean(np.abs(s - s_hat))),
        },
        "b_y": b_y,
        "soup_state_sha256": run["manifest"]["soup_state_sha256"],
    }
    np.savez_compressed(
        run["run_dir"] / "t_eval_predictions.npz",
        gid=gid, y=y, y_raw=y_raw, y_cal=y_cal, h=h, q_raw=q_raw,
    )
    if log:
        log(f"[t_eval {arm} s{seed}] y_raw MAE {result['mae']['y_raw']:.5f} "
            f"y_cal {result['mae']['y_cal']:.5f} g_raw {result['mae']['g_raw']:.5f}")
    return result


def _group_paired_bootstrap(
    e_source: np.ndarray,
    e_target: np.ndarray,
    group_of_row: np.ndarray,
    *,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> dict[str, Any]:
    """SMILES-group-paired bootstrap of the SOURCE - TARGET contrast.

    Whole canonical-SMILES groups of T_eval are resampled with replacement
    (2000 draws, fixed seed); identical molecules never split a draw.  The
    molecular bootstrap carries no basis or training-seed uncertainty.
    """
    e_source = np.asarray(e_source, np.float64)
    e_target = np.asarray(e_target, np.float64)
    if e_source.shape != e_target.shape:
        raise RuntimeError("paired bootstrap row misalignment")
    labels = np.asarray([str(v) for v in np.asarray(group_of_row, dtype=object).tolist()], dtype=object)
    order = {label: index for index, label in enumerate(sorted(set(labels.tolist())))}
    g_index = np.asarray([order[label] for label in labels.tolist()], np.int64)
    n_groups = len(order)
    sum_s = np.zeros(n_groups, np.float64)
    sum_t = np.zeros(n_groups, np.float64)
    cnt = np.zeros(n_groups, np.int64)
    np.add.at(sum_s, g_index, e_source)
    np.add.at(sum_t, g_index, e_target)
    np.add.at(cnt, g_index, 1)
    rng = np.random.default_rng(int(seed))
    diffs = np.empty(int(n_boot), np.float64)
    ratios = np.empty(int(n_boot), np.float64)
    for b in range(int(n_boot)):
        pick = rng.integers(0, n_groups, n_groups)
        n = int(cnt[pick].sum())
        m_s = float(sum_s[pick].sum() / n)
        m_t = float(sum_t[pick].sum() / n)
        diffs[b] = m_s - m_t
        ratios[b] = m_s / m_t if m_t > 0 else np.inf
    return {
        "n_rows": int(e_source.size),
        "n_groups": int(n_groups),
        "n_boot": int(n_boot),
        "boot_seed": int(seed),
        "source_mae": float(e_source.mean()),
        "target_mae": float(e_target.mean()),
        "mean_diff": float(e_source.mean() - e_target.mean()),
        "diff_ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
        "ratio": float(e_source.mean() / e_target.mean()) if e_target.mean() > 0 else float("inf"),
        "ratio_ci95": [float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))],
    }


def _t_fit_mean_alpha(arm: str, out_dir: Path = RESULTS_DIR) -> np.ndarray:
    """The arm's T_fit mean per-root alpha under its own frozen basis."""
    out_dir = Path(out_dir)
    objects = load_domain_objects(out_dir)
    basis = load_basis(arm, out_dir)
    fit_idx = np.asarray(objects["fold"]["t_fit_idx"], np.int64)
    phi_rows, _checks = _phi_rows_for(fit_idx)
    z, _phi_hat, _rel = _root_codes(basis, phi_rows)
    alpha = z[:, COMMON_DIM:]
    if not np.isfinite(alpha).all():
        raise RuntimeError(f"non-finite T_fit root codes for arm {arm}")
    return alpha.mean(axis=0)


@torch.no_grad()
def run_dict_interventions(
    arm: str,
    seed: int,
    base: Mapping[str, Any],
    mean_alpha: np.ndarray,
    *,
    out_dir: Path = RESULTS_DIR,
    device_name: str = "cuda:0",
    log: Any = print,
) -> dict[str, Any]:
    """The unique dictionary intervention for one trained arm/seed.

    Every CSSD consumption entry's alpha_v is replaced by the arm's T_fit
    mean root-code alpha.  The patch sits at ``model.code`` — the single
    point where phi is coded into z — so every downstream branch that
    consumes alpha is regenerated from the substituted alpha and nothing
    cached is left stale; common c_v, the original graphs, the raw/root-MLP
    inputs, the Sem108 interface and the shared Q head are untouched.  No
    retraining, no recalibration.  Dependence evidence, never incremental
    benefit; this is not a legal new molecule distribution and not a
    no-dictionary control.
    """
    device = resolve_device(device_name)
    torch.set_num_threads(8 if device.type == "cpu" else 4)
    out_dir = Path(out_dir)
    objects = load_domain_objects(out_dir)
    basis = load_basis(arm, out_dir)
    payload = zlt.TuplePayload(objects["payload_arrays"])
    kappa_M = float(objects["kappa"]["kappa_M"])
    run = load_run(arm, seed, out_dir)
    soup_state = run["soup_state"]

    model = build_arm(SKELETON_ARM, payload, kappa_M, basis, int(seed))
    model.load_state_dict({k: v for k, v in soup_state.items()})
    model = model.to(device)
    model.eval()
    data = build_domain_data(objects)
    t_eval_data = data["t_eval"]
    mean_alpha_t = torch.as_tensor(np.asarray(mean_alpha, np.float64), dtype=torch.float32)
    original_code = model.code

    def patched_code(phi: torch.Tensor) -> torch.Tensor:
        z = original_code(phi)
        out = z.clone()
        out[:, model.common_dim :] = mean_alpha_t.to(device=z.device, dtype=z.dtype).expand(
            int(z.shape[0]), -1
        )
        return out

    model.code = patched_code
    try:
        h_i, ell_hat_i, s_hat_i = _predict_rows(model, t_eval_data, device)
    finally:
        del model.code

    # original graphs -> identical topology -> the shared Q head and the
    # single fit-median b_y are reused unchanged
    y_raw_i = ell_hat_i + s_hat_i + np.asarray(base["q_raw"], np.float64)
    y_cal_i = y_raw_i + float(base["b_y"])
    y = np.asarray(base["y"], np.float64)
    g = np.asarray(base["g"], np.float64)
    base_y_raw = np.asarray(base["y_raw"], np.float64)
    base_h = np.asarray(base["h"], np.float64)
    d_pred = np.abs(y_raw_i - base_y_raw)
    row = {
        "arm": arm,
        "seed": int(seed),
        "mechanism": (
            "alpha_v := the arm's T_fit mean root-code alpha, patched at model.code "
            "(the single CSSD consumption entry); c_v, original graphs, raw/root-MLP, "
            "Sem108 and Q inputs untouched; no retraining, no recalibration"
        ),
        "base_y_raw_mae": float(np.mean(np.abs(y - base_y_raw))),
        "y_raw_mae": float(np.mean(np.abs(y - y_raw_i))),
        "y_cal_mae": float(np.mean(np.abs(y - y_cal_i))),
        "delta_y_raw_mae": float(np.mean(np.abs(y - y_raw_i)) - np.mean(np.abs(y - base_y_raw))),
        "delta_y_cal_mae": float(
            np.mean(np.abs(y - y_cal_i)) - np.mean(np.abs(y - (base_y_raw + float(base["b_y"]))))
        ),
        "delta_g_mae": float(np.mean(np.abs(g - h_i)) - np.mean(np.abs(g - base_h))),
        "g_note": "g change with Q unchanged: g reads only the A binding path, so q_raw is identical by construction",
        "abs_dpred": {
            "mean": float(d_pred.mean()),
            "median": float(np.median(d_pred)),
            "p95": float(np.percentile(d_pred, 95)),
            "max": float(d_pred.max()),
        },
        "response_fraction": float(np.mean(d_pred > REPLAY_TOL)),
        "replay_tol": REPLAY_TOL,
        "mean_alpha_sha256": array_sha256(np.asarray(mean_alpha, np.float64)),
    }
    np.savez_compressed(
        run["run_dir"] / "t_eval_predictions_alpha_mean.npz",
        gid=np.asarray(base["gid"], np.int64), y=y, y_raw=y_raw_i, y_cal=y_cal_i, h=h_i,
        q_raw=np.asarray(base["q_raw"], np.float64),
    )
    if log:
        log(f"[alpha-mean {arm} s{seed}] ΔMAE_y_raw {row['delta_y_raw_mae']:+.5f} "
            f"response {row['response_fraction']:.3f} |Δpred| max {row['abs_dpred']['max']:.5f}")
    return row


def frozen_decision_rules(
    per_seed: Mapping[int, Mapping[str, Any]],
    avg: Mapping[str, Any],
    coverage: Mapping[str, Any],
    interventions: Mapping[str, Mapping[str, Any]],
    *,
    seeds: Sequence[int] = REUSE_SEEDS,
) -> dict[str, Any]:
    """Frozen conclusion branches A-F (pre-registered; no post-hoc tuning).

    Per seed: M_S / M_T = SOURCE / TARGET full y_raw MAE on T_eval,
    delta = M_S - M_T (positive = reuse worse), R = M_S / M_T.
    Branch order F -> C -> A -> E -> D -> B.
    """
    seeds = tuple(int(s) for s in seeds)
    ratios = {int(seed): float(per_seed[int(seed)]["R"]) for seed in seeds}
    m_s_avg = float(avg["M_S"])
    m_t_avg = float(avg["M_T"])
    coverage_severe = bool(coverage["severe"]["any"])
    sparse_favorable = bool(all(
        interventions[f"{arm}_s{seed}"]["delta_y_raw_mae"] > 0.0
        and interventions[f"{arm}_s{seed}"]["response_fraction"] >= SPARSE_RESPONSE_FRACTION_MIN
        for arm in REUSE_ARMS for seed in seeds
    ))
    both_weak = bool(m_s_avg > BOTH_WEAK_MAE and m_t_avg > BOTH_WEAK_MAE)
    ratio_upper = float(avg["bootstrap"]["ratio_ci95"][1])
    seed_disagreement = bool(
        len(set(seeds)) == 2 and abs(ratios[seeds[0]] - ratios[seeds[1]]) > SEED_DISAGREEMENT_R
    )
    conditions = {
        "per_seed_R": {str(seed): ratios[seed] for seed in seeds},
        "avg_ratio_upper": ratio_upper,
        "rel_tol": REL_TOL,
        "coverage_severe": coverage_severe,
        "sparse_favorable": sparse_favorable,
        "sparse_favorable_rule": (
            f"all arm/seed runs: delta_y_raw_mae > 0 AND response_fraction >= {SPARSE_RESPONSE_FRACTION_MIN}"
        ),
        "both_weak": both_weak,
        "seed_disagreement": seed_disagreement,
    }
    if both_weak:
        branch = "F"
    elif all(r > REL_TOL for r in ratios.values()):
        branch = "C"
    elif (
        all(r <= REL_TOL for r in ratios.values())
        and ratio_upper <= REL_TOL
        and not coverage_severe
        and sparse_favorable
    ):
        branch = "A"
    elif all(r < 1.0 for r in ratios.values()):
        branch = "E"
    elif not sparse_favorable:
        branch = "D"
    else:
        branch = "B"
    reasoning = {
        "A": (
            "both seeds R <= 1.05, averaged-ratio bootstrap upper <= 1.05, no severe "
            "coverage failure, favorable sparse-branch dependence: continue with the "
            "frozen SOURCE package — limited evidence for this basis pair and this "
            "size-domain shift only; the saved target-basis refit cost is reported and "
            "the total training cost is not zero"
        ),
        "B": (
            "point estimates close but the ratio CI crosses 1.05 or the seeds disagree "
            "beyond the pre-registered bound: not equivalence — record uncertainty and "
            "stop; no seed-3 purchase"
        ),
        "C": (
            "SOURCE clearly worse on both seeds: use the coverage diagnostics to "
            "separate basis-coverage failure from consumer wiring/optimization — "
            "reconstruction also worse supports a coverage problem; reconstruction "
            "close with a large prediction gap leaves coverage insufficient as the "
            "sole explanation"
        ),
        "D": (
            "performance close but the sparse branch shows no favorable dependence: "
            "this round cannot support the sparse-basis claim from performance "
            "maintenance alone; keep the bypass / shared-common-direction competing "
            "explanations"
        ),
        "E": (
            "SOURCE stably better on both seeds: possibly source-domain structural "
            "prior / regularization or target-domain overfitting — recorded, still not "
            "a dictionary-specific advantage"
        ),
        "F": (
            "both arms weak on the target domain: report the common failure — relative "
            "closeness does not establish a strong model; this round is a reuse "
            "diagnostic, not a SOTA purchase"
        ),
    }
    recommendation = {
        "A": "continue-frozen-source-package",
        "B": "insufficient-pause",
        "C": "reuse-not-supported-stop",
        "D": "reuse-not-supported-stop",
        "E": "continue-frozen-source-package",
        "F": "insufficient-pause",
    }[branch]
    return {
        "branch": branch,
        "recommendation": recommendation,
        "conditions": conditions,
        "reasoning": reasoning[branch],
        "pre_registered": True,
    }


def terminal_eval(
    *,
    out_dir: Path = RESULTS_DIR,
    device_name: str = "cuda:0",
    seeds: Sequence[int] = REUSE_SEEDS,
    log: Any = print,
) -> dict[str, Any]:
    """One-shot T_eval: roster -> coverage -> intervention -> eval -> rules.

    Runs once, after every soup, fit-only bias, coverage and intervention
    script is frozen.  No result of one seed is used to alter another.  The
    output file's existence is the one-shot guard.
    """
    out_dir = Path(out_dir)
    if (out_dir / "terminal_eval.json").exists():
        raise RuntimeError(
            "terminal_eval.json already exists: T_eval is the one-shot terminal stage"
        )
    started = time.perf_counter()
    seeds = tuple(int(s) for s in seeds)
    if sorted(set(seeds)) != sorted(set(int(s) for s in REUSE_SEEDS)):
        raise RuntimeError(f"terminal eval expects the pre-registered body seeds {REUSE_SEEDS}")
    objects = load_domain_objects(out_dir)

    # (1) roster: every pre-registered run present, completed, biases frozen
    roster: dict[str, Any] = {}
    for seed in seeds:
        for arm in REUSE_ARMS:
            run = load_run(arm, seed, out_dir)
            manifest = run["manifest"]
            if manifest["stopped_reason"] != "completed":
                raise RuntimeError(
                    f"run {arm} s{seed} did not complete ({manifest['stopped_reason']})"
                )
            roster[f"{arm}_s{seed}"] = {
                "soup_state_sha256": manifest["soup_state_sha256"],
                "b_y": manifest["calibration"]["b_y"],
                "steps_done": manifest["steps_done"],
                "frozen_basis_hashes": manifest["frozen_basis_hashes"],
                "q_soup_sha256": manifest["q_soup_sha256"],
            }

    # (2) coverage of the frozen bases (no training state)
    coverage = coverage_diagnostics(out_dir=out_dir, log=log)

    # (3) base T_eval predictions for every pre-registered run
    results: dict[str, dict[str, Any]] = {}
    for seed in seeds:
        for arm in REUSE_ARMS:
            results[f"{arm}_s{seed}"] = evaluate_arm(
                arm, seed, out_dir=out_dir, device_name=device_name, log=log
            )

    # (4) the unique dictionary intervention on every trained soup
    mean_alpha = {arm: _t_fit_mean_alpha(arm, out_dir) for arm in REUSE_ARMS}
    interventions: dict[str, dict[str, Any]] = {}
    for seed in seeds:
        for arm in REUSE_ARMS:
            key = f"{arm}_s{seed}"
            interventions[key] = run_dict_interventions(
                arm, seed, results[key], mean_alpha[arm],
                out_dir=out_dir, device_name=device_name, log=log,
            )

    # (5) frozen interpretation
    t_eval_idx = np.asarray(objects["fold"]["t_eval_idx"], np.int64)
    group_of_row = _canonical_smiles()[t_eval_idx]
    y = np.asarray(objects["targets"]["y"], np.float64)[t_eval_idx]
    per_seed: dict[int, dict[str, Any]] = {}
    for seed in seeds:
        rs, rt = results[f"SOURCE_s{seed}"], results[f"TARGET_s{seed}"]
        e_s = np.abs(y - np.asarray(rs["y_raw"], np.float64))
        e_t = np.abs(y - np.asarray(rt["y_raw"], np.float64))
        m_s, m_t = float(e_s.mean()), float(e_t.mean())
        per_seed[int(seed)] = {
            "M_S": m_s, "M_T": m_t, "delta": m_s - m_t, "R": m_s / m_t,
            "mae": {"SOURCE": rs["mae"], "TARGET": rt["mae"]},
            "bootstrap": _group_paired_bootstrap(e_s, e_t, group_of_row),
        }
    e_s_avg = np.mean(
        [np.abs(y - np.asarray(results[f"SOURCE_s{seed}"]["y_raw"], np.float64)) for seed in seeds],
        axis=0,
    )
    e_t_avg = np.mean(
        [np.abs(y - np.asarray(results[f"TARGET_s{seed}"]["y_raw"], np.float64)) for seed in seeds],
        axis=0,
    )
    avg = {
        "M_S": float(e_s_avg.mean()),
        "M_T": float(e_t_avg.mean()),
        "R": float(e_s_avg.mean() / e_t_avg.mean()),
        "bootstrap": _group_paired_bootstrap(e_s_avg, e_t_avg, group_of_row),
        "note": (
            "two-seed averaged per-row errors; the molecular bootstrap carries no "
            "basis or training-seed uncertainty and 2 body seeds are not a claim of "
            "method stability"
        ),
    }
    decision = frozen_decision_rules(
        per_seed, avg, coverage, interventions, seeds=seeds
    )

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "stage": "terminal-eval",
        "seconds": float(time.perf_counter() - started),
        "split": "T_eval (sealed one-shot comparison set; never used in training)",
        "one_shot": True,
        "roster": roster,
        "coverage": coverage,
        "per_arm": {
            key: {
                "mae": value["mae"], "b_y": value["b_y"], "n": value["n"],
                "soup_state_sha256": value["soup_state_sha256"],
            }
            for key, value in results.items()
        },
        "interventions": interventions,
        "per_seed": {str(seed): per_seed[seed] for seed in seeds},
        "two_seed_average": avg,
        "decision": decision,
        "sign_convention": (
            "M_S/M_T = SOURCE/TARGET full y_raw MAE; delta = M_S - M_T, positive = reuse worse"
        ),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(out_dir / "terminal_eval.json", payload)
    log(f"[terminal-eval] branch {decision['branch']} ({decision['recommendation']}); "
        f"per-seed R: " + ", ".join(
            f"s{seed}={per_seed[seed]['R']:.4f}" for seed in seeds
        ) + f"; avg-ratio CI95 [{avg['bootstrap']['ratio_ci95'][0]:.4f}, "
        f"{avg['bootstrap']['ratio_ci95'][1]:.4f}]")
    return payload


# ---------------------------------------------------------------------------
# 8. pre-flight checks (CPU; frozen manifests verified before any GPU use)
# ---------------------------------------------------------------------------


def run_checks(*, n_graphs: int = 24, out_dir: Path = RESULTS_DIR, log: Any = print) -> dict[str, Any]:
    """Re-derive the split, verify manifests, contracts and invariants (CPU)."""
    torch.set_num_threads(8)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    device = torch.device("cpu")
    objects = load_domain_objects(out_dir)

    # (a) domain split re-derivation (frozen rule, not the stored arrays)
    split = build_domain_split()
    for key in ("s_pool_idx", "t_pool_idx", "t_eval_idx", "t_fit_idx",
                "s_fit_idx", "s_unused_idx", "t_unused_idx"):
        if not np.array_equal(np.asarray(split[key], np.int64), np.asarray(objects["fold"][key], np.int64)):
            raise RuntimeError(f"domain split re-derivation mismatch at {key}")
    split_ok = {
        "rederived_identical": True,
        "threshold": split["threshold"],
        "sizes": split["checks"]["sizes"],
        "expected_split_frozen": {k: int(v) for k, v in EXPECTED_SPLIT.items()},
        "hashes": split["hashes"],
    }

    # (b) bases present, shaped, distinct; hashes verified inside load_basis
    bases = {arm: load_basis(arm, out_dir) for arm in REUSE_ARMS}
    shape_ok: dict[str, Any] = {}
    for arm, basis in bases.items():
        if basis["D"].shape != (PHI_DIM, CSSD_K_ATOMS):
            raise RuntimeError(f"basis {arm} D shape {basis['D'].shape}")
        if basis["U"].shape[0] != PHI_DIM:
            raise RuntimeError(f"basis {arm} U shape {basis['U'].shape}")
        shape_ok[arm] = {
            "D": list(basis["D"].shape), "U": list(basis["U"].shape),
            "common_rms": list(basis["common_rms"].shape),
            "sha256": {k: array_sha256(basis[k]) for k in ("U", "common_rms", "D")},
        }
    if shape_ok["SOURCE"]["sha256"] == shape_ok["TARGET"]["sha256"]:
        raise RuntimeError("SOURCE and TARGET bases are identical (refit failed to differ)")

    # (c) bit-identical consumer init across arms: same seed, same payload,
    # different frozen basis; only dictionary buffers may differ
    payload = zlt.TuplePayload(objects["payload_arrays"])
    kappa_M = float(objects["kappa"]["kappa_M"])
    seed_everything(SEED)
    model_source = build_arm(SKELETON_ARM, payload, kappa_M, bases["SOURCE"], SEED)
    seed_everything(SEED)
    model_target = build_arm(SKELETON_ARM, payload, kappa_M, bases["TARGET"], SEED)
    sd_s, sd_t = model_source.state_dict(), model_target.state_dict()
    if sorted(sd_s) != sorted(sd_t):
        raise RuntimeError("arm state_dict key sets differ")
    differing = [k for k in sd_s if not torch.equal(sd_s[k], sd_t[k])]
    buffer_names = {name for name, _ in model_source.named_buffers()}
    if not set(differing) <= buffer_names:
        raise RuntimeError(f"non-buffer keys differ across arms: {differing}")
    init_ok = {
        "differing_state_keys": differing,
        "all_differing_keys_are_frozen_buffers": True,
        "n_compared_keys": len(sd_s),
        "source_frozen_basis_hashes": model_source.frozen_basis_hashes(),
        "target_frozen_basis_hashes": model_target.frozen_basis_hashes(),
    }
    for arm, model, basis in (("SOURCE", model_source, bases["SOURCE"]), ("TARGET", model_target, bases["TARGET"])):
        if model.frozen_basis_hashes() != {
            "U": array_sha256(basis["U"]), "common_rms": array_sha256(basis["common_rms"]), "D": array_sha256(basis["D"]),
        }:
            raise RuntimeError(f"arm {arm} frozen basis hashes != loaded basis")

    # (c2) seed 1 is a genuinely different body init; frozen buffers identical
    seed_everything(1)
    model_source1 = build_arm(SKELETON_ARM, payload, kappa_M, bases["SOURCE"], 1)
    sd_1 = model_source1.state_dict()
    differ_seed = [k for k in sd_s if not torch.equal(sd_s[k], sd_1[k])]
    frozen_keys = ("U", "common_rms", "D")
    frozen_identical = all(torch.equal(sd_s[k], sd_1[k]) for k in frozen_keys)
    _sched0, sched_hash0 = zw.build_schedule(len(objects["fold"]["t_fit_idx"]), EPOCHS, 0 + TRAIN_SHUFFLE_OFFSET)
    _sched1, sched_hash1 = zw.build_schedule(len(objects["fold"]["t_fit_idx"]), EPOCHS, 1 + TRAIN_SHUFFLE_OFFSET)
    seed_ok = {
        "differing_state_keys_seed0_vs_seed1": differ_seed,
        "n_differing_keys": len(differ_seed),
        "init_genuinely_differs": bool(len(differ_seed) > 0),
        "frozen_buffers_identical_across_seeds": bool(frozen_identical),
        "schedule_hash_seed0": sched_hash0,
        "schedule_hash_seed1": sched_hash1,
        "schedule_genuinely_differs": bool(sched_hash0 != sched_hash1),
    }
    if not (seed_ok["init_genuinely_differs"] and seed_ok["frozen_buffers_identical_across_seeds"]
            and seed_ok["schedule_genuinely_differs"]):
        raise RuntimeError(f"seed 1 is not a genuinely different run: {seed_ok}")

    # (d) two real training steps on CPU: finite loss, frozen basis unchanged,
    # gradients reach the A-slot modules; dictionary buffers not in the optimizer
    data = build_domain_data(objects)
    fit_data = data["t_fit"]
    fit_idx = np.asarray(objects["fold"]["t_fit_idx"], np.int64)
    g = np.asarray(objects["targets"]["g"], np.float64)[fit_idx]
    ell = np.asarray(objects["targets"]["ell"], np.float64)[fit_idx]
    s = np.asarray(objects["targets"]["s"], np.float64)[fit_idx]
    y = np.asarray(objects["targets"]["y"], np.float64)[fit_idx]
    for position, row in enumerate(fit_data):
        if abs(float(getattr(row, "y", y[position])) - float(y[position])) > 1e-12:
            raise RuntimeError("T_fit data .y != y_stored")
    target_g = torch.as_tensor(g, dtype=torch.float32)
    target_ell = torch.as_tensor(ell, dtype=torch.float32)
    target_s = torch.as_tensor(s, dtype=torch.float32)
    schedule, _ = zw.build_schedule(len(fit_data), 2, SEED + TRAIN_SHUFFLE_OFFSET)
    step_ok: dict[str, Any] = {}
    for arm, model in (("SOURCE", model_source), ("TARGET", model_target)):
        model = model.to(torch.device("cpu"))
        dict_param_names = [n for n, _ in model.named_parameters() if n in ("U", "common_rms", "D")]
        if dict_param_names:
            raise RuntimeError(f"dictionary buffers leaked into parameters: {dict_param_names}")
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        frozen_before = model.frozen_basis_hashes()
        losses = []
        for step in range(2):
            indices = [int(i) for i in schedule[0][step * BATCH_SIZE:(step + 1) * BATCH_SIZE]]
            index_t = torch.as_tensor(indices, dtype=torch.long)
            batch = zftd.make_batch(fit_data, indices, target_g, torch.device("cpu"))
            model.train()
            prediction = model(batch, mask=cm.C6_MASK)
            components = model.reader.components()
            loss = F.l1_loss(prediction.view(-1), batch.y.view(-1)) + COMPONENT_LOSS_WEIGHT * (
                F.l1_loss(components[:, 0], target_ell[index_t]) + F.l1_loss(components[:, 1], target_s[index_t])
            )
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError(f"arm {arm} non-finite loss at step {step}")
            optimizer.zero_grad()
            loss.backward()
            total_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP))
            if not math.isfinite(total_norm):
                raise RuntimeError(f"arm {arm} non-finite grad norm at step {step}")
            optimizer.step()
            losses.append(float(loss.detach()))
        slot_grad_norm = float(sum(
            p.grad.detach().pow(2).sum().item() for n, p in model.named_parameters()
            if n.startswith(("W_A_S", "W_A_C", "W_E_S", "W_E_C", "node_encoder.", "edge_encoder."))
        ) ** 0.5)
        if slot_grad_norm <= 0.0:
            raise RuntimeError(f"arm {arm}: gradients never reached the A-slot modules")
        if model.frozen_basis_hashes() != frozen_before:
            raise RuntimeError(f"arm {arm}: frozen basis changed during CPU checks")
        step_ok[arm] = {"losses": losses, "slot_grad_norm": slot_grad_norm,
                        "dictionary_buffers_not_parameters": True}

    # (e) recipe equality with the design-basis round
    recipe_ok = {
        name: bool(getattr(parent, name) == value)
        for name, value in (
            ("SEED", SEED), ("EPOCHS", EPOCHS), ("LR", LR), ("WEIGHT_DECAY", WEIGHT_DECAY),
            ("GRAD_CLIP", GRAD_CLIP), ("BATCH_SIZE", BATCH_SIZE), ("SOUP_EPOCHS", SOUP_EPOCHS),
            ("LOG_EPOCHS", LOG_EPOCHS), ("COMPONENT_LOSS_WEIGHT", COMPONENT_LOSS_WEIGHT),
            ("CSSD_K_ATOMS", CSSD_K_ATOMS), ("CSSD_SPARSITY", CSSD_SPARSITY),
            ("CSSD_EPOCHS", CSSD_EPOCHS), ("CSSD_SOUP_EPOCHS", CSSD_SOUP_EPOCHS),
            ("Q_EPOCHS", Q_EPOCHS), ("Q_SOUP_EPOCHS", Q_SOUP_EPOCHS), ("Q_PARAMETERS", Q_PARAMETERS),
        )
    }
    if not all(recipe_ok.values()):
        raise RuntimeError(f"recipe constants drifted from the design basis: {recipe_ok}")

    # (f) Q head parameter audit
    head = build_q_head(SEED, 0.0)
    n_q = int(sum(p.numel() for p in head.parameters()))
    if n_q != Q_PARAMETERS:
        raise RuntimeError(f"Q parameter audit failed: {n_q}")

    # (g) target identities: y = g + c, g = ell + s on every stored row
    targets = objects["targets"]
    y_all = np.asarray(targets["y"], np.float64)
    g_all = np.asarray(targets["g"], np.float64)
    ell_all = np.asarray(targets["ell"], np.float64)
    s_all = np.asarray(targets["s"], np.float64)
    c_all = np.asarray(targets["c"], np.float64)
    identity_ok = {
        "rows": int(y_all.size),
        "y_equals_g_plus_c_max_abs": float(np.max(np.abs(y_all - (g_all + c_all)))),
        "g_equals_ell_plus_s_max_abs": float(np.max(np.abs(g_all - (ell_all + s_all)))),
        "s_is_residual_note": "s = g - ell is the residual, not a pure SA label",
        "single_bias_note": "b_y = fit-median(y - y_raw), added exactly once at eval (per run manifest)",
    }
    if max(identity_ok["y_equals_g_plus_c_max_abs"], identity_ok["g_equals_ell_plus_s_max_abs"]) > 1e-12:
        raise RuntimeError(f"target identities failed: {identity_ok}")

    # (h) CSSD code invariances per arm (chemistry relabel, row chunking)
    graphs = list(fit_data[: min(n_graphs, len(fit_data))])
    batch = p1.env_collate(graphs).to(device)
    relabeled_batch = p1.env_collate(graphs_with_relabelled_chemistry(graphs, 20261012)).to(device)
    code_ok: dict[str, Any] = {}
    for arm in REUSE_ARMS:
        model = model_source if arm == "SOURCE" else model_target
        z0 = model.code(batch.dict_phi).detach().numpy()
        z1 = model.code(relabeled_batch.dict_phi).detach().numpy()
        z_chunked = np.concatenate(
            [model.code(batch.dict_phi[i : i + 7]).detach().numpy()
             for i in range(0, batch.dict_phi.shape[0], 7)],
            axis=0,
        )
        code_ok[arm] = {
            "z_chemistry_relabel_invariant": bool(np.array_equal(z0, z1)),
            "z_row_chunk_max_abs": float(np.max(np.abs(z0 - z_chunked))),
            "z_row_chunk_invariant": bool(np.max(np.abs(z0 - z_chunked)) <= 1e-5),
            "z_width": int(z0.shape[1]),
            "z_expected_width": int(COMMON_DIM + CSSD_K_ATOMS),
            "frozen_basis_hashes": model.frozen_basis_hashes(),
        }
        if not (code_ok[arm]["z_chemistry_relabel_invariant"] and code_ok[arm]["z_row_chunk_invariant"]):
            raise RuntimeError(f"CSSD code invariance failed for arm {arm}")
        if code_ok[arm]["z_width"] != code_ok[arm]["z_expected_width"]:
            raise RuntimeError(f"z width mismatch for arm {arm}")

    # (i) bond dedup on the checked fit graphs (each undirected bond once per root)
    bond_dedup_ok = True
    n_bond_rows = 0
    for graph in graphs:
        bu = graph.env_bond_u.cpu().numpy()
        bv = graph.env_bond_v.cpu().numpy()
        root = graph.env_bond_root.cpu().numpy()
        keys = set()
        for r, u, v in zip(root.tolist(), bu.tolist(), bv.tolist()):
            key = (int(r), min(int(u), int(v)), max(int(u), int(v)))
            if key in keys:
                bond_dedup_ok = False
            keys.add(key)
        n_bond_rows += int(bu.shape[0])
    if not bond_dedup_ok:
        raise RuntimeError("bond dedup failed")

    # (j) A-slot occurrence-row order invariance + endpoint swap (report) +
    #     static contract: pair_relation mutation leaves pre-pair slots intact
    model_iv = build_arm(SKELETON_ARM, payload, kappa_M, bases["SOURCE"], SEED).to(device)
    model_iv.eval()
    coord = model_iv.code(batch.dict_phi)
    node_slots, edge_slots = model_iv._arm_a_slots(coord, batch, cm.C6_MASK)
    shuffled_batch = p1.env_collate(graphs_with_permuted_occurrence_rows(graphs, 20261013)).to(device)
    node_shuf, edge_shuf = model_iv._arm_a_slots(coord, shuffled_batch, cm.C6_MASK)
    swapped_batch = p1.env_collate(graphs_with_swapped_bond_endpoints(graphs)).to(device)
    _node_sw, edge_sw = model_iv._arm_a_slots(coord, swapped_batch, cm.C6_MASK)
    batch_pair_mut = p1.env_collate(graphs_with_mutated_pair_relation(graphs)).to(device)
    node_pair_mut, edge_pair_mut = model_iv._arm_a_slots(coord, batch_pair_mut, cm.C6_MASK)
    slot_invariance = {
        "occurrence_row_order_node_max_abs": float(torch.max(torch.abs(node_slots - node_shuf)).item()),
        "occurrence_row_order_edge_max_abs": float(torch.max(torch.abs(edge_slots - edge_shuf)).item()),
        "endpoint_swap_edge_max_abs": float(torch.max(torch.abs(edge_slots - edge_sw)).item()),
        "pair_relation_node_slots_max_abs": float(torch.max(torch.abs(node_slots - node_pair_mut)).item()),
        "pair_relation_edge_slots_max_abs": float(torch.max(torch.abs(edge_slots - edge_pair_mut)).item()),
    }
    if max(slot_invariance["occurrence_row_order_node_max_abs"],
            slot_invariance["occurrence_row_order_edge_max_abs"],
            slot_invariance["pair_relation_node_slots_max_abs"],
            slot_invariance["pair_relation_edge_slots_max_abs"]) > 1e-4:
        raise RuntimeError(f"A-slot invariance/static contract failed: {slot_invariance}")

    # (k) prediction invariances: batch composition, reversed order, node
    #     relabelling (with the consistently reordered tuple payload), and
    #     same-node-count different-structure graphs using their own incidence
    g0 = fit_data[0]
    g1 = next(d for d in fit_data[1:n_graphs] if int(d.dict_phi.shape[0]) == int(g0.dict_phi.shape[0]))
    two = p1.env_collate([g0, g1]).to(device)
    p_two = _eval_batch_predict(model_iv, two, device)
    p_singles = np.concatenate(
        [_eval_batch_predict(model_iv, p1.env_collate([g]).to(device), device) for g in (g0, g1)]
    )
    reversed_two = p1.env_collate([g1, g0]).to(device)
    p_rev = _eval_batch_predict(model_iv, reversed_two, device)
    relabel_perm = torch.randperm(int(g0.dict_phi.shape[0]), generator=torch.Generator().manual_seed(20261015))
    g0r = _relabel_nodes(g0, relabel_perm.numpy())
    mol0 = int(torch.as_tensor(g0.local_mol_id).reshape(-1)[0].item())
    permuted_payload = zlt.TuplePayload(
        _permuted_tuple_payload(objects["payload_arrays"], mol0, relabel_perm.numpy())
    )
    original_payload = model_iv.local_tuple._payload
    original_cache = model_iv.local_tuple._device_cache
    model_iv.local_tuple._payload = permuted_payload
    model_iv.local_tuple._device_cache = {}
    try:
        p_g0r = _eval_batch_predict(model_iv, p1.env_collate([g0r]).to(device), device)
    finally:
        model_iv.local_tuple._payload = original_payload
        model_iv.local_tuple._device_cache = original_cache
    prediction_invariances = {
        "same_node_count_different_structure": {
            "node_count": int(g0.dict_phi.shape[0]),
            "structure_signature_differs": bool(not np.allclose(g0.dict_phi.numpy(), g1.dict_phi.numpy())),
            "batch_equals_individuals_max_abs": float(np.max(np.abs(p_two - p_singles))),
            "reversed_batch_max_abs": float(np.max(np.abs(p_rev - p_two[::-1]))),
        },
        "node_relabel_max_abs": float(np.max(np.abs(p_g0r - p_singles[:1]))),
        "prediction_range": [float(p_two.min()), float(p_two.max())],
        "prediction_has_variation": bool(np.std(p_two) > 0),
    }
    if max(prediction_invariances["same_node_count_different_structure"]["batch_equals_individuals_max_abs"],
            prediction_invariances["same_node_count_different_structure"]["reversed_batch_max_abs"],
            prediction_invariances["node_relabel_max_abs"]) > 1e-4:
        raise RuntimeError(f"prediction invariance failed: {prediction_invariances}")

    # (l) the temporary source-domain CSSD task head never entered the consumer
    audit_a = parameter_audit(model_iv)
    expected_total = int(parent._expected_arm_parameters(SKELETON_ARM))
    task_head_ok = {
        "consumer_total_parameters": int(audit_a["total_parameters"]),
        "design_basis_arm_A_total_parameters": expected_total,
        "cssd_task_head_discarded": True,
        "note": "the task-informed head lives only in the temporary CSSD training model; "
                "only U/common_rms/D buffers enter the consumer",
    }
    if int(audit_a["total_parameters"]) != expected_total:
        raise RuntimeError(f"consumer parameter count {audit_a['total_parameters']} != arm-A expected {expected_total}")

    checks = {
        "protocol_version": PROTOCOL_VERSION,
        "stage": "checks",
        "seconds": float(time.perf_counter() - started),
        "n_graphs": int(min(n_graphs, len(fit_data))),
        "domain_split": split_ok,
        "basis_shapes": shape_ok,
        "bit_identical_init": init_ok,
        "seed_parameterisation": seed_ok,
        "cpu_steps": step_ok,
        "recipe_equals_design_basis": recipe_ok,
        "q_parameters": n_q,
        "target_identities": identity_ok,
        "cssd_code": code_ok,
        "bond_dedup": {
            "n_graphs": int(min(n_graphs, len(fit_data))),
            "n_bond_rows": int(n_bond_rows),
            "each_undirected_bond_once_per_root": bool(bond_dedup_ok),
        },
        "a_slot_invariances": slot_invariance,
        "prediction_invariances": prediction_invariances,
        "task_head_not_in_consumer": task_head_ok,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(out_dir / "checks.json", checks)
    log(f"[checks] ok in {checks['seconds']:.1f}s "
        f"(split sizes {split_ok['sizes']}, differing init keys {init_ok['differing_state_keys']}, "
        f"endpoint-swap edge max abs {slot_invariance['endpoint_swap_edge_max_abs']:.2e} [report])")
    return checks


# ---------------------------------------------------------------------------
# 9. GPU smoke (a few real T_fit batches per arm; no silent CPU fallback)
# ---------------------------------------------------------------------------


def run_smoke(
    *,
    device_name: str = "cuda:0",
    n_batches: int = 4,
    out_dir: Path = RESULTS_DIR,
    log: Any = print,
) -> dict[str, Any]:
    device = resolve_device(device_name)
    torch.set_num_threads(4)
    started = time.perf_counter()
    objects = load_domain_objects(out_dir)
    bases = {arm: load_basis(arm, out_dir) for arm in REUSE_ARMS}
    q_soup = load_q_soup(out_dir)
    payload = zlt.TuplePayload(objects["payload_arrays"])
    kappa_M = float(objects["kappa"]["kappa_M"])
    data = build_domain_data(objects)
    fit_data = data["t_fit"]
    fit_idx = np.asarray(objects["fold"]["t_fit_idx"], np.int64)
    g = np.asarray(objects["targets"]["g"], np.float64)[fit_idx]
    ell = np.asarray(objects["targets"]["ell"], np.float64)[fit_idx]
    s = np.asarray(objects["targets"]["s"], np.float64)[fit_idx]
    target_g = torch.as_tensor(g, dtype=torch.float32, device=device)
    target_ell = torch.as_tensor(ell, dtype=torch.float32, device=device)
    target_s = torch.as_tensor(s, dtype=torch.float32, device=device)
    schedule, _ = zw.build_schedule(len(fit_data), EPOCHS, SEED + TRAIN_SHUFFLE_OFFSET)

    smoke: dict[str, Any] = {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "arms": {},
    }
    for arm in REUSE_ARMS:
        arm_started = time.perf_counter()
        seed_everything(SEED)
        model = build_arm(SKELETON_ARM, payload, kappa_M, bases[arm], SEED).to(device)
        off_device = [n for n, p in model.named_parameters() if p.device.type != device.type]
        off_device += [n for n, b in model.named_buffers() if b.device.type != device.type]
        if off_device:
            raise RuntimeError(f"arm {arm} tensors not on {device}: {off_device}")
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        frozen_before = model.frozen_basis_hashes()
        losses = []
        for step in range(int(n_batches)):
            indices = [int(i) for i in schedule[0][step * BATCH_SIZE:(step + 1) * BATCH_SIZE]]
            index_t = torch.as_tensor(indices, dtype=torch.long, device=device)
            batch = zftd.make_batch(fit_data, indices, target_g, device)
            model.train()
            prediction = model(batch, mask=cm.C6_MASK)
            components = model.reader.components()
            g_identity = float(torch.max(torch.abs(components.sum(-1) - prediction.detach())).item())
            loss = F.l1_loss(prediction.view(-1), batch.y.view(-1)) + COMPONENT_LOSS_WEIGHT * (
                F.l1_loss(components[:, 0], target_ell[index_t]) + F.l1_loss(components[:, 1], target_s[index_t])
            )
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError(f"arm {arm} non-finite loss at step {step}")
            optimizer.zero_grad()
            loss.backward()
            total_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP))
            if not math.isfinite(total_norm):
                raise RuntimeError(f"arm {arm} non-finite grad norm at step {step}")
            optimizer.step()
            losses.append(float(loss.detach()))
        frozen_after = model.frozen_basis_hashes()
        model.eval()
        with torch.no_grad():
            batch = zftd.make_batch(fit_data, [int(i) for i in schedule[0][:BATCH_SIZE]], target_g, device)
            coord = model.code(batch.dict_phi)
            node_slots, edge_slots = model._arm_a_slots(coord, batch, cm.C6_MASK)
            node_out = model.node_encoder(node_slots)
            edge_out = model.edge_encoder(edge_slots)
            slot_stats = {
                "node_slots_rms": float(node_slots.pow(2).mean().sqrt()),
                "edge_slots_rms": float(edge_slots.pow(2).mean().sqrt()),
                "node_out_std": float(node_out.std()),
                "edge_out_std": float(edge_out.std()),
            }
        slot_grad_norm = float(sum(
            p.grad.detach().pow(2).sum().item() for n, p in model.named_parameters()
            if n.startswith(("W_A_S", "W_A_C", "W_E_S", "W_E_C", "node_encoder.", "edge_encoder."))
        ) ** 0.5)
        if slot_grad_norm <= 0.0:
            raise RuntimeError(f"arm {arm}: gradients never reached the A-slot modules")
        if device.type == "cuda":
            peak_mb = float(torch.cuda.max_memory_allocated() / (1 << 20))
            torch.cuda.reset_peak_memory_stats()
        else:
            peak_mb = 0.0
        smoke["arms"][arm] = {
            "losses": losses,
            "component_identity_max_abs": g_identity,
            "slot_stats": slot_stats,
            "slot_grad_norm": slot_grad_norm,
            "frozen_basis_unchanged": bool(frozen_before == frozen_after),
            "peak_gpu_memory_mb": peak_mb,
            "seconds": float(time.perf_counter() - arm_started),
        }
        if not smoke["arms"][arm]["frozen_basis_unchanged"]:
            raise RuntimeError(f"arm {arm}: frozen basis changed during GPU smoke")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    smoke["q_soup_sha256"] = state_hash(q_soup)
    smoke["all_finite"] = True
    smoke["seconds"] = float(time.perf_counter() - started)
    smoke["official_valid_loaded"] = False
    smoke["official_test_loaded"] = False
    write_json(out_dir / "smoke.json", smoke)
    log(f"[smoke] both arms ok on {device} in {smoke['seconds']:.1f}s")
    return smoke


# ---------------------------------------------------------------------------
# 10. CLI (local use; the registered runner calls the same functions)
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", default=str(RESULTS_DIR))
    parser.add_argument("--write-smiles-table", action="store_true",
                        help="one-shot local derivation of the committed canonical-SMILES table")
    parser.add_argument("--build-objects", action="store_true")
    parser.add_argument("--cssd-refit", action="store_true")
    parser.add_argument("--train-q", action="store_true")
    parser.add_argument("--checks", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--terminal-eval", action="store_true")
    parser.add_argument("--arm", choices=REUSE_ARMS, default=None,
                        help="restrict --cssd-refit to one basis; required with --train")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)

    if args.write_smiles_table:
        write_canonical_smiles_table()
    if args.build_objects:
        phase_build_objects(out_dir=out_dir)
    if args.cssd_refit:
        arms = (args.arm,) if args.arm else REUSE_ARMS
        for arm in arms:
            cssd_refit_domain(arm, out_dir=out_dir)
    if args.train_q:
        train_q(out_dir=out_dir)
    if args.checks:
        run_checks(out_dir=out_dir)
    if args.smoke:
        run_smoke(device_name=args.device, out_dir=out_dir)
    if args.train:
        if args.arm is None:
            parser.error("--train requires --arm SOURCE|TARGET")
        if int(args.seed) not in REUSE_SEEDS:
            parser.error(f"--seed must be one of the pre-registered body seeds {REUSE_SEEDS}")
        train_reuse_arm(
            args.arm, seed=int(args.seed), device_name=args.device,
            out_dir=out_dir, max_steps=args.max_steps,
        )
    if args.terminal_eval:
        terminal_eval(out_dir=out_dir, device_name=args.device)
    if not any((
        args.write_smiles_table, args.build_objects, args.cssd_refit, args.train_q,
        args.checks, args.smoke, args.train, args.terminal_eval,
    )):
        parser.error("no stage requested (use --write-smiles-table / --build-objects / "
                     "--cssd-refit / --train-q / --checks / --smoke / --train / --terminal-eval)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
