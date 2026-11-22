"""Label-safe data layer for PSC: splits, category support, matched pairs.

Boundaries (see PROTOCOL.md section 4):

* structure members come from the label-free raw whitelist in ``psc_common``;
* Train labels come from the round-local ``labels_official_train.npz`` that was
  built from exactly two members of a frozen prediction file;
* Valid labels are *never* read here;
* every statistic (scaler, ``c_l``, category support, path vocabulary, pair
  graph / weights) is a function of the current fit rows only.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import psc_common as C
import psc_paths as PP

if str(C.AUDIT_DIR / "code") not in sys.path:
    sys.path.insert(0, str(C.AUDIT_DIR / "code"))
import audit_common as AU  # noqa: E402  (standalone module: no legacy loader is called)

#: TRC members that the matched-pair directory needs (all label-free)
TRC_PAIR_KEYS = (
    "pair_mol_a", "pair_mol_b", "pair_C", "pair_Sa", "pair_Sb",
    "pair_cut_bond_row", "pair_transform",
    "transform_word_a", "transform_word_b", "transform_bond_row",
    "parse_mol", "parse_S", "parse_C", "parse_cut_bond_row", "parse_kind",
    "parse_bond_index", "parse_port_S", "parse_cut_u", "parse_cut_v",
    "fragment_keys", "n_fragments",
)
#: ACRA members used to map an endpoint to its local identities
ACRA_PAIR_KEYS = ("ev_ptr", "ev_dir", "ev_bond", "ev_key_s", "ev_key_c", "n_molecules")


@dataclass
class DataBundle:
    raw: Mapping[str, Any]
    labels: np.ndarray            # NaN outside official Train
    row_id: np.ndarray
    label_meta: Mapping[str, Any]
    official_train: np.ndarray
    official_valid: np.ndarray
    hashes: dict[str, Any]

    def rows_to_rows(self, rows: Sequence[int]) -> np.ndarray:
        return np.asarray(rows, np.int64)


def load_bundle() -> DataBundle:
    raw = C.load_raw_structure()
    if not C.TRAIN_LABELS.is_file():
        # deterministic, label-safe extraction (two members of the frozen file)
        C.build_train_labels(raw)
    train = C.load_train_labels()
    n_mol = int(np.asarray(raw["node_ptr"]).size - 1)
    labels = np.full(n_mol, np.nan, np.float64)
    labels[train["row_id"]] = train["label"]
    official_train = np.sort(np.asarray(raw["train_idx"], np.int64))
    official_valid = np.sort(np.asarray(raw["valid_idx"], np.int64))
    if train["row_id"].size != official_train.size:
        raise ValueError("train label row count mismatch")
    if np.any(np.isfinite(labels[official_valid])):
        raise ValueError("Valid rows carry a finite label in the training object")
    C.assert_fit_rows(official_train, train["row_id"])
    hashes = {
        "raw_cache": {"path": str(C.RAW_CACHE.relative_to(C.REPO_ROOT)),
                      "sha256": str(raw["cache_sha256"])},
        "train_labels": train,
        "trc_cache": {"path": str(C.TRC_CACHE.relative_to(C.REPO_ROOT)),
                      "sha256": C.sha256_file(C.TRC_CACHE),
                      "members_used": list(TRC_PAIR_KEYS)},
        "acra_cache": {"path": str(C.ACRA_CACHE.relative_to(C.REPO_ROOT)),
                       "sha256": C.sha256_file(C.ACRA_CACHE),
                       "members_used": list(ACRA_PAIR_KEYS)},
    }
    return DataBundle(raw=raw, labels=labels, row_id=np.asarray(raw["row_ids"], np.int64),
                      label_meta=train, official_train=official_train,
                      official_valid=official_valid, hashes=hashes)


def labels_standardized(bundle: DataBundle, fit_rows: Sequence[int]) -> tuple[np.ndarray, dict[str, float]]:
    scaler = C.target_scaler(bundle.labels, fit_rows)
    y = (bundle.labels - scaler["mean"]) / scaler["std"]
    return y, scaler


# ---------------------------------------------------------------------------
# raw category support (fit rows only)
# ---------------------------------------------------------------------------


def category_support(raw: Mapping[str, Any], fit_rows: Sequence[int],
                     atom_dims: Sequence[int], bond_dims: Sequence[int]
                     ) -> tuple[list[np.ndarray], list[np.ndarray]]:
    node_ptr = np.asarray(raw["node_ptr"], np.int64)
    bond_ptr = np.asarray(raw["bond_ptr"], np.int64)
    atom_feat = np.asarray(raw["atom_feat"])
    bond_feat = np.asarray(raw["bond_feat"])
    atom_ok = [np.zeros(int(d), bool) for d in atom_dims]
    bond_ok = [np.zeros(int(d), bool) for d in bond_dims]
    for row in np.asarray(fit_rows, np.int64).tolist():
        a0, a1 = int(node_ptr[row]), int(node_ptr[row + 1])
        b0, b1 = int(bond_ptr[row]), int(bond_ptr[row + 1])
        for i in range(atom_feat.shape[1]):
            atom_ok[i][np.unique(atom_feat[a0:a1, i].astype(np.int64))] = True
        for i in range(bond_feat.shape[1]):
            bond_ok[i][np.unique(bond_feat[b0:b1, i].astype(np.int64))] = True
    return atom_ok, bond_ok


def unseen_category_fraction(raw: Mapping[str, Any], rows: Sequence[int],
                             atom_ok: Sequence[np.ndarray], bond_ok: Sequence[np.ndarray]
                             ) -> dict[str, float]:
    node_ptr = np.asarray(raw["node_ptr"], np.int64)
    bond_ptr = np.asarray(raw["bond_ptr"], np.int64)
    atom_feat = np.asarray(raw["atom_feat"])
    bond_feat = np.asarray(raw["bond_feat"])
    n_atoms = 0
    n_bonds = 0
    unk_atoms = 0
    unk_bonds = 0
    for row in np.asarray(rows, np.int64).tolist():
        a0, a1 = int(node_ptr[row]), int(node_ptr[row + 1])
        b0, b1 = int(bond_ptr[row]), int(bond_ptr[row + 1])
        block = atom_feat[a0:a1]
        unk = np.zeros(block.shape[0], bool)
        for i, mask in enumerate(atom_ok):
            unk |= ~mask[block[:, i].astype(np.int64)]
        n_atoms += int(block.shape[0])
        unk_atoms += int(unk.sum())
        bblock = bond_feat[b0:b1]
        bunk = np.zeros(bblock.shape[0], bool)
        for i, mask in enumerate(bond_ok):
            bunk |= ~mask[bblock[:, i].astype(np.int64)]
        n_bonds += int(bblock.shape[0])
        unk_bonds += int(bunk.sum())
    return {"atoms": float(unk_atoms) / max(1, n_atoms),
            "bonds": float(unk_bonds) / max(1, n_bonds),
            "n_atoms": float(n_atoms), "n_bonds": float(n_bonds)}


# ---------------------------------------------------------------------------
# matched-pair directory
# ---------------------------------------------------------------------------


class PairDirectory:
    """All official-Train matched pairs, mapped to exact structural identities.

    Thin wrapper around the audit's frozen pair table (same data structures,
    no legacy eager loader): ``build_pair_table`` is fed a hand-built
    ``Inputs`` whose ``trc`` / ``acra`` mappings contain only the whitelisted
    label-free members.
    """

    def __init__(self, bundle: DataBundle) -> None:
        trc: dict[str, np.ndarray] = {}
        with np.load(C.TRC_CACHE, allow_pickle=False) as z:
            for key in TRC_PAIR_KEYS:
                trc[key] = np.asarray(z[key])
        acra: dict[str, np.ndarray] = {}
        with np.load(C.ACRA_CACHE, allow_pickle=False) as z:
            for key in ACRA_PAIR_KEYS:
                acra[key] = np.asarray(z[key])
        n_mol = int(np.asarray(bundle.raw["node_ptr"]).size - 1)
        official_train = np.zeros(n_mol, bool)
        official_train[bundle.official_train] = True
        inputs = AU.Inputs(
            trc=trc, acra=acra, split={}, n_molecules=n_mol,
            official_train=official_train,
            train_sub=np.zeros(n_mol, bool), dev=np.zeros(n_mol, bool),
            labels=np.asarray(bundle.labels, np.float64),
            scaffold_of_mol=None, hashes={}, checks={})
        # build_pair_table only needs official_train; the scope columns are
        # discarded below because every official-Train row is "train_sub" here.
        inputs.train_sub = official_train.copy()
        self.table = AU.build_pair_table(inputs)
        self.trc = trc
        self.acra = acra
        self.n_fragments = int(trc["n_fragments"])
        self.mol_a = np.asarray(self.table.a, np.int64)
        self.mol_b = np.asarray(self.table.b, np.int64)
        self.word_a = np.asarray(self.table.Sa, np.int64)
        self.word_b = np.asarray(self.table.Sb, np.int64)
        self.bond_id = np.asarray(self.table.bond_row, np.int64)
        # T identity: the two rooted-fragment canonical keys in fixed
        # lexicographic order plus the full cut-bond attribute id.
        lo = np.minimum(self.word_a, self.word_b)
        hi = np.maximum(self.word_a, self.word_b)
        self.T_id = lo.astype(np.int64) * self.n_fragments + hi
        self.T_key = np.stack([lo, hi, self.bond_id], 1)
        self.checks = dict(self.table.checks)


def unique_pairs(mol_a: np.ndarray, mol_b: np.ndarray) -> np.ndarray:
    """Canonical ``a < b`` unique unordered molecule pairs (first occurrence)."""
    lo = np.minimum(mol_a, mol_b)
    hi = np.maximum(mol_a, mol_b)
    key = lo.astype(np.int64) * (int(max(mol_a.max(), mol_b.max())) + 1) + hi
    _, first = np.unique(key, return_index=True)
    first = np.sort(first)
    return first


@dataclass
class PairTargets:
    """Unique-pair supervision graph restricted to the current fit rows."""

    a: np.ndarray                 # row id, smaller
    b: np.ndarray                 # row id, larger
    cut_idx: np.ndarray           # index into the PairDirectory pair rows
    w: np.ndarray                 # 1 / (K * m_c)
    component: np.ndarray         # component id per unique pair
    n_components: int
    comp_edges: list[np.ndarray]  # unique-pair indices per component
    d: np.ndarray                 # per-molecule diagonal weight, length n_mol
    checks: dict[str, Any]


def build_pair_targets(dir_: PairDirectory, fit_rows: Sequence[int],
                       n_mol: int) -> PairTargets:
    fit = np.zeros(int(n_mol), bool)
    fit[np.asarray(fit_rows, np.int64)] = True
    keep = np.nonzero(fit[dir_.mol_a] & fit[dir_.mol_b])[0]
    first = unique_pairs(dir_.mol_a[keep], dir_.mol_b[keep])
    cut_idx = keep[first]
    a = np.minimum(dir_.mol_a[cut_idx], dir_.mol_b[cut_idx])
    b = np.maximum(dir_.mol_a[cut_idx], dir_.mol_b[cut_idx])
    # direction must follow row_id order: delta = y(row_id_small) -> y(row_id_large)
    comp_nodes = AU.components(int(n_mol), a, b)
    comp_of_pair = comp_nodes[a]
    # in-component edge index (0..m_c-1), deterministic by unique-pair order
    order = np.lexsort((b, a, comp_of_pair))
    comp_sorted = comp_of_pair[order]
    in_comp = np.empty(order.size, np.int64)
    start = 0
    for i in range(1, order.size + 1):
        if i == order.size or comp_sorted[i] != comp_sorted[i - 1]:
            m = i - start
            in_comp[order[start:i]] = np.arange(m)
            start = i
    uniq_comps, comp_sizes = np.unique(comp_of_pair, return_counts=True)
    K = int(uniq_comps.size)
    size_of_pair = comp_sizes[np.searchsorted(uniq_comps, comp_of_pair)]
    w = 1.0 / (float(K) * size_of_pair.astype(np.float64))
    comp_index = np.searchsorted(uniq_comps, comp_of_pair)
    comp_edges = [np.nonzero(comp_index == c)[0] for c in range(K)]
    d = np.zeros(int(n_mol), np.float64)
    np.add.at(d, a, w)
    np.add.at(d, b, w)
    checks = {
        "n_cut_rows_in_fit": int(keep.size),
        "n_unique_pairs": int(a.size),
        "n_duplicate_cut_rows": int(keep.size - a.size),
        "n_components": K,
        "max_component_edges": int(max((e.size for e in comp_edges), default=0)),
        "effective_components": float(1.0 / np.sum(np.square([w[e].sum() for e in comp_edges]))) if K else 0.0,
        "sum_w": float(w.sum()),
        "n_molecules_with_weight": int(np.count_nonzero(d)),
        "label_delta_sha256": C.array_sha256(np.stack([a, b], 1)),
    }
    return PairTargets(a=a, b=b, cut_idx=cut_idx, w=w, component=comp_index,
                       n_components=K, comp_edges=comp_edges, d=d, checks=checks)


def component_metrics(targets: PairTargets, n_mol: int) -> dict[str, Any]:
    """Component structure of the full pair graph over ``targets`` rows."""
    mask = np.zeros(int(n_mol), bool)
    mask[targets.a] = True
    mask[targets.b] = True
    wc = np.asarray([targets.w[e].sum() for e in targets.comp_edges])
    return {
        "n_unique_pairs": int(targets.a.size),
        "n_unique_molecules": int(mask.sum()),
        "n_components": int(targets.n_components),
        "max_component_mass": float(wc.max()) if wc.size else 0.0,
        "effective_components": float(1.0 / np.sum(wc ** 2)) if wc.size else 0.0,
    }


class PairSampler:
    """Unbiased minibatch sampler for ``L_pair`` / ``L_diag``.

    Uniform component, then uniform unique edge inside it, with replacement.
    Both PAIR and DIAG use the same sampler object and the same seed sequence.
    """

    def __init__(self, targets: PairTargets, seed: int) -> None:
        if targets.n_components == 0:
            raise ValueError("no matched-pair component in the fit rows")
        self.targets = targets
        self.rng = np.random.default_rng(int(seed))
        self.sizes = np.asarray([e.size for e in targets.comp_edges], np.int64)
        if int(self.sizes.sum()) != targets.a.size:
            raise AssertionError("component edge partition lost pairs")

    def sample(self, n_pairs: int) -> tuple[np.ndarray, np.ndarray]:
        comp = self.rng.integers(0, int(self.targets.n_components), size=int(n_pairs))
        edges = np.empty(int(n_pairs), np.int64)
        for i, c in enumerate(comp.tolist()):
            pool = self.targets.comp_edges[int(c)]
            edges[i] = int(pool[self.rng.integers(0, pool.size)])
        return self.targets.a[edges], self.targets.b[edges]


# ---------------------------------------------------------------------------
# vocabulary / counts (fit only)
# ---------------------------------------------------------------------------

@dataclass
class PathVocab:
    """Fit-only canonical ORDER path vocabulary and per-molecule counts."""

    ids: dict[bytes, int]
    c_l: np.ndarray
    fit_rows: np.ndarray
    n_by_length: np.ndarray
    entropy: dict[str, Any]


def fit_path_vocab(cache: PP.PathCache, fit_rows: Sequence[int], max_len: int
                   ) -> PathVocab:
    """Vocabulary from fit molecules only (unscaled occurrence counts)."""
    pos = cache.positions(fit_rows)
    ids: dict[int, int] = {}
    starts = cache.mol_ptr[pos]
    ends = cache.mol_ptr[pos + 1]
    for s, e in zip(starts.tolist(), ends.tolist()):
        for pid in cache.ent_path[s:e].tolist():
            ids.setdefault(int(pid), len(ids))
    counts = cache.path_counts(fit_rows).astype(np.float64)
    mean = counts.mean(0)
    c_l = 1.0 / np.maximum(1.0, mean)
    return PathVocab(ids=ids, c_l=c_l, fit_rows=np.asarray(fit_rows, np.int64),
                     n_by_length=counts.sum(0),
                     entropy={"n_canonical_keys": len(ids),
                              "mean_paths_by_len": mean.tolist()})