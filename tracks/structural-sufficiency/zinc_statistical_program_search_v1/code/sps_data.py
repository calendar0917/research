"""Static structural views for zinc-statistical-program-search-v1.

Only label-free structure is precomputed.  Two views are built from the frozen
architecture-discovery object cache (which is itself a deterministic function of
the official PyG ZINC atoms/bonds):

Q view  dense per-root record layout: for molecule i with n atoms and m physical
        bonds the records are the n*n (root, atom) pairs and the n*m (root, bond)
        pairs, in row-major order.  Only the per-record root/atom/bond index,
        the true graph distance id and the distance bucket are stored; the record
        *content* is computed from the shared encoders at every step (nothing
        learnable is cached).

F view  no extra cache: dense padded incidence is assembled per batch from the
        physical-bond list.

Grouping (fixed by the task):
  atom records  : d in {0,1,2} -> 0, {3,4} -> 1, {>=5 or unreachable} -> 2
  bond records  : min(d(r,u), d(r,v)) in the same three ranges -> 3,4,5
  distance id   : 0..8 exact, 9 = >=9, 10 = unreachable

The frozen cache stores ``dist_flat`` as uint8 (the original int64 unreachable
sentinel was truncated to 0 by that cast), so unreachable pairs are recovered as
``d == 0 and r != u``; the connectedness of every molecule is asserted at prep.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_AD = _HERE.parents[1] / "zinc_arch_discovery_v1" / "code"
for _p in (str(_HERE), str(_AD)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402

SPS = _HERE.parent
SPSCACHE = SPS / "cache"
ANALYSIS = SPS / "analysis"


def _bucket(d: np.ndarray) -> np.ndarray:
    return np.where(d <= 2, 0, np.where(d <= 4, 1, 2)).astype(np.uint8)


def _dist_id(d: np.ndarray) -> np.ndarray:
    return np.minimum(d, 9).astype(np.uint8)


def build_q_view(ad_cache: Mapping[str, Any]) -> dict:
    nmol = int(ad_cache["n_molecules"])
    atom_ptr = np.asarray(ad_cache["atom_ptr"])
    bond_ptr = np.asarray(ad_cache["bond_ptr"])
    dist_ptr = np.asarray(ad_cache["dist_ptr"])
    dist_flat = np.asarray(ad_cache["dist_flat"])
    bond_index = np.asarray(ad_cache["bond_index"], dtype=np.int64)

    atom_rec_ptr = np.zeros(nmol + 1, dtype=np.int64)
    bond_rec_ptr = np.zeros(nmol + 1, dtype=np.int64)
    ar_root, ar_atom, ar_dist, ar_grp = [], [], [], []
    br_root, br_bond, br_grp = [], [], []
    root_cnt = np.zeros((int(atom_ptr[-1]), 6), dtype=np.float32)
    n_disconnected = 0
    n_no_bond = 0
    max_n = 0

    for i in range(nmol):
        n = int(atom_ptr[i + 1] - atom_ptr[i])
        m = int(bond_ptr[i + 1] - bond_ptr[i])
        s, e = int(dist_ptr[i]), int(dist_ptr[i + 1])
        Dm = dist_flat[s:e].reshape(n, n).astype(np.int64)
        bad = (Dm == 0) & ~np.eye(n, dtype=bool)
        if bad.any():
            n_disconnected += 1
            Dm = np.where(bad, np.int64(1000), Dm)
        if m == 0:
            n_no_bond += 1
        max_n = max(max_n, n)
        b = bond_index[int(bond_ptr[i]):int(bond_ptr[i + 1])]
        bu, bv = b[:, 0], b[:, 1]

        root = np.repeat(np.arange(n, dtype=np.uint8), n)
        atom = np.tile(np.arange(n, dtype=np.uint8), n)
        dflat = Dm.reshape(-1)
        grp = _bucket(dflat)
        ar_root.append(root)
        ar_atom.append(atom)
        ar_dist.append(_dist_id(dflat))
        ar_grp.append(grp)
        cnt = np.zeros((n, 3), dtype=np.float32)
        gg = grp.reshape(n, n)
        for g in range(3):
            cnt[:, g] = (gg == g).sum(1)
        atom_rec_ptr[i + 1] = atom_rec_ptr[i] + n * n

        dmin = np.minimum(Dm[:, bu], Dm[:, bv]) if m else np.zeros((n, 0), dtype=np.int64)
        bgrp = _bucket(dmin.reshape(-1)) if m else np.zeros(0, dtype=np.uint8)
        br_root.append(np.repeat(np.arange(n, dtype=np.uint8), m))
        br_bond.append(np.tile(np.arange(m, dtype=np.uint8), n))
        br_grp.append(bgrp)
        bcnt = np.zeros((n, 3), dtype=np.float32)
        if m:
            bgg = bgrp.reshape(n, m)
            for g in range(3):
                bcnt[:, g] = (bgg == g).sum(1)
        bond_rec_ptr[i + 1] = bond_rec_ptr[i] + n * m
        a0 = int(atom_ptr[i])
        root_cnt[a0:a0 + n, :3] = cnt
        root_cnt[a0:a0 + n, 3:] = bcnt

    view = {
        "atom_rec_ptr": atom_rec_ptr,
        "bond_rec_ptr": bond_rec_ptr,
        "ar_root": np.concatenate(ar_root).astype(np.uint8),
        "ar_atom": np.concatenate(ar_atom).astype(np.uint8),
        "ar_dist": np.concatenate(ar_dist).astype(np.uint8),
        "ar_grp": np.concatenate(ar_grp).astype(np.uint8),
        "br_root": np.concatenate(br_root).astype(np.uint8),
        "br_bond": np.concatenate(br_bond).astype(np.uint8),
        "br_grp": np.concatenate(br_grp).astype(np.uint8),
        "root_grp_count": root_cnt,
        "meta": {
            "n_molecules": nmol,
            "n_atoms": int(atom_ptr[-1]),
            "n_bonds": int(bond_ptr[-1]),
            "atom_records": int(atom_rec_ptr[-1]),
            "bond_records": int(bond_rec_ptr[-1]),
            "disconnected_molecules": int(n_disconnected),
            "molecules_without_bonds": int(n_no_bond),
            "max_atoms": int(max_n),
            "feature_version": "sps-v1-q-records",
            "labels_used": False,
        },
    }
    return view


def build_all(force: bool = False) -> dict:
    SPSCACHE.mkdir(parents=True, exist_ok=True)
    out = {}
    for split in ("train", "valid"):
        ad = D.load_cache(split)
        path = SPSCACHE / f"sps_q_{split}.pt"
        if path.exists() and not force:
            out[split] = torch.load(path, map_location="cpu", weights_only=False)["meta"]
            continue
        view = build_q_view(ad)
        C.atomic_torch_save(view, path)
        out[split] = view["meta"]
    norm = D.load_norm()
    train_view = load_q_view("train")
    train_ad = D.load_cache("train")
    atom_ptr = np.asarray(train_ad["atom_ptr"])
    bond_ptr = np.asarray(train_ad["bond_ptr"])
    n_atoms = np.diff(atom_ptr).astype(np.float64)
    n_bonds = np.diff(bond_ptr).astype(np.float64)
    root_cnt = np.asarray(train_view["root_grp_count"], dtype=np.float64)
    fit = {
        "source": "official ZINC train 10k structures only; no valid/test labels",
        "global_mean": np.asarray(norm["global_mean"], dtype=np.float32).tolist(),
        "global_scale": np.asarray(norm["global_scale"], dtype=np.float32).tolist(),
        "fit_mean_atoms": float(n_atoms.mean()),
        "fit_mean_bonds": float(n_bonds.mean()),
        "fit_group_count": root_cnt.mean(0).astype(np.float32).tolist(),
        "root_group_count_mean": root_cnt.mean(0).astype(np.float32).tolist(),
        "q_view": out,
        "norm_source": norm.get("source"),
        "used_valid_labels": False,
    }
    C.write_json(ANALYSIS / "fit_stats.json", fit)
    return fit


def load_q_view(split: str) -> dict:
    path = SPSCACHE / f"sps_q_{split}.pt"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run sps_train.py prep")
    return torch.load(path, map_location="cpu", weights_only=False)


def load_fit() -> dict:
    return C.read_json(ANALYSIS / "fit_stats.json")


def _ranges(ptr: np.ndarray, rows: np.ndarray):
    ptr = np.asarray(ptr)
    rows = np.asarray(rows, dtype=np.int64)
    starts = ptr[rows].astype(np.int64)
    ends = ptr[rows + 1].astype(np.int64)
    counts = ends - starts
    idx = np.concatenate([np.arange(s, e, dtype=np.int64) for s, e in zip(starts, ends)]) \
        if len(rows) else np.zeros(0, dtype=np.int64)
    batch = np.repeat(np.arange(len(rows), dtype=np.int64), counts)
    local = np.concatenate([np.arange(c, dtype=np.int64) for c in counts]) \
        if len(rows) else np.zeros(0, dtype=np.int64)
    return idx, batch, local, counts


def _common(ad: Mapping[str, Any], rows: np.ndarray):
    atom_ptr = np.asarray(ad["atom_ptr"])
    bond_ptr = np.asarray(ad["bond_ptr"])
    atom_idx, atom_batch, atom_local, atom_counts = _ranges(atom_ptr, rows)
    bond_idx, bond_batch, bond_local, bond_counts = _ranges(bond_ptr, rows)
    atom_off = np.concatenate([[0], np.cumsum(atom_counts)[:-1]]).astype(np.int64)
    bond_off = np.concatenate([[0], np.cumsum(bond_counts)[:-1]]).astype(np.int64)
    return (atom_ptr, bond_ptr, atom_idx, atom_batch, atom_local, atom_counts,
            bond_idx, bond_batch, bond_local, bond_counts, atom_off, bond_off)


def q_collate(cache: Mapping[str, Any], rows: Sequence[int], candidate: str, *,
              valid: bool = False, walk_cols: np.ndarray | None = None) -> dict:
    """Batch one set of molecules for route Q (no labels are read here)."""
    view = cache["view"]
    ad = cache["ad"]
    rows = np.asarray(rows, dtype=np.int64)
    (atom_ptr, bond_ptr, atom_idx, atom_batch, atom_local, atom_counts,
     bond_idx, bond_batch, bond_local, bond_counts, atom_off, bond_off) = _common(ad, rows)
    B = len(rows)

    at_idx, at_batch, at_local, at_counts = _ranges(view["atom_rec_ptr"], rows)
    bt_idx, bt_batch, bt_local, bt_counts = _ranges(view["bond_rec_ptr"], rows)
    ar_base = np.concatenate([[0], np.cumsum(at_counts)[:-1]]).astype(np.int64)

    ar_root_local = np.asarray(view["ar_root"])[at_idx].astype(np.int64)
    ar_atom_local = np.asarray(view["ar_atom"])[at_idx].astype(np.int64)
    a_root = ar_root_local + atom_off[at_batch]
    a_atom = ar_atom_local + atom_off[at_batch]

    bi_local = np.asarray(ad["bond_index"])[bond_idx].astype(np.int64)  # per-molecule
    bi = bi_local + atom_off[bond_batch].reshape(-1, 1)                  # batch-local
    br_root_local = np.asarray(view["br_root"])[bt_idx].astype(np.int64)
    b_root = br_root_local + atom_off[bt_batch]
    # batch-local bond id (into the concatenated batch) and dataset-global id
    b_local = np.asarray(view["br_bond"])[bt_idx].astype(np.int64) + bond_off[bt_batch]
    b_global = np.asarray(view["br_bond"])[bt_idx].astype(np.int64) + bond_ptr[rows][bt_batch]
    # endpoint atom-record indices use the endpoints of the *record's* bond
    bi_rec = np.asarray(ad["bond_index"])[b_global].astype(np.int64)   # per-molecule
    n_local = np.diff(atom_ptr)[rows].astype(np.int64)
    base_rec = ar_base[bt_batch] + br_root_local * n_local[bt_batch]
    iu = base_rec + bi_rec[:, 0]
    iv = base_rec + bi_rec[:, 1]

    root_cnt = np.asarray(view["root_grp_count"])[atom_idx].astype(np.float32)

    batch = {
        "n_graphs": B,
        "atom_feat": torch.from_numpy(np.asarray(ad["atom_feat"])[atom_idx].astype(np.float32)),
        "bond_feat": torch.from_numpy(np.asarray(ad["bond_feat"])[bond_idx].astype(np.float32)),
        "bond_index": torch.from_numpy(bi),
        "atom_batch": torch.from_numpy(atom_batch),
        "atom_local": torch.from_numpy(atom_local),
        "bond_batch": torch.from_numpy(bond_batch),
        "bond_local": torch.from_numpy(bond_local),
        "global": torch.from_numpy(np.asarray(ad["global_feat"])[rows].astype(np.float32)),
        "atom_counts": torch.from_numpy(np.diff(atom_ptr)[rows].astype(np.float32)),
        "bond_counts": torch.from_numpy(np.diff(bond_ptr)[rows].astype(np.float32)),
        "atom_rec_atom": torch.from_numpy(a_atom),
        "atom_rec_dist": torch.from_numpy(np.asarray(view["ar_dist"])[at_idx].astype(np.int64)),
        "atom_rec_group": torch.from_numpy(np.asarray(view["ar_grp"])[at_idx].astype(np.int64)),
        "atom_rec_root": torch.from_numpy(a_root),
        "bond_rec_bond": torch.from_numpy(b_local),
        "bond_rec_group": torch.from_numpy(np.asarray(view["br_grp"])[bt_idx].astype(np.int64)),
        "bond_rec_root": torch.from_numpy(b_root),
        "bond_rec_iu": torch.from_numpy(iu),
        "bond_rec_iv": torch.from_numpy(iv),
        "root_cnt": torch.from_numpy(root_cnt),
    }
    return batch


def f_collate(cache: Mapping[str, Any], rows: Sequence[int], candidate: str, *,
              valid: bool = False, walk_cols: np.ndarray | None = None) -> dict:
    """Batch one set of molecules for route F (dense padded incidence)."""
    ad = cache["ad"]
    rows = np.asarray(rows, dtype=np.int64)
    (atom_ptr, bond_ptr, atom_idx, atom_batch, atom_local, atom_counts,
     bond_idx, bond_batch, bond_local, bond_counts, atom_off, bond_off) = _common(ad, rows)
    B = len(rows)
    Nmax = int(atom_counts.max())
    Emax = int(max(1, bond_counts.max()))
    bi_local = np.asarray(ad["bond_index"])[bond_idx].astype(np.int64)
    bi = bi_local + atom_off[bond_batch].reshape(-1, 1)

    edge_u = np.zeros((B, Emax), dtype=np.int64)
    edge_v = np.zeros((B, Emax), dtype=np.int64)
    edge_mask = np.zeros((B, Emax), dtype=bool)
    edge_u[bond_batch, bond_local] = bi_local[:, 0]
    edge_v[bond_batch, bond_local] = bi_local[:, 1]
    edge_mask[bond_batch, bond_local] = True

    return {
        "n_graphs": B,
        "Nmax": Nmax,
        "Emax": Emax,
        "atom_feat": torch.from_numpy(np.asarray(ad["atom_feat"])[atom_idx].astype(np.float32)),
        "bond_feat": torch.from_numpy(np.asarray(ad["bond_feat"])[bond_idx].astype(np.float32)),
        "bond_index": torch.from_numpy(bi),
        "atom_batch": torch.from_numpy(atom_batch),
        "atom_local": torch.from_numpy(atom_local),
        "bond_batch": torch.from_numpy(bond_batch),
        "bond_local": torch.from_numpy(bond_local),
        "global": torch.from_numpy(np.asarray(ad["global_feat"])[rows].astype(np.float32)),
        "atom_counts": torch.from_numpy(np.diff(atom_ptr)[rows].astype(np.float32)),
        "bond_counts": torch.from_numpy(np.diff(bond_ptr)[rows].astype(np.float32)),
        "edge_u": torch.from_numpy(edge_u),
        "edge_v": torch.from_numpy(edge_v),
        "edge_mask": torch.from_numpy(edge_mask),
        "edge_batch": torch.from_numpy(bond_batch),
        "edge_local": torch.from_numpy(bond_local),
    }


_VIEW_MEMO: dict[str, dict] = {}


def _view_for(split: str) -> dict:
    key = f"{split}:q"
    if key not in _VIEW_MEMO:
        _VIEW_MEMO[key] = load_q_view(split)
    return _VIEW_MEMO[key]


def collate_factory(candidate: str, split: str | None = None):
    """Return ONE collate closure that dispatches on the cache it is given.

    ``ad_train.train`` passes its own ``train_cache`` and ``valid_cache`` to the
    same callable, so the closure must never bind a fixed split: the split is
    taken from the cache mapping (``cache["split"]``).  Getting this wrong would
    evaluate valid labels against train features, which is exactly the failure
    mode the CPU checks guard against.
    """
    if candidate not in ("QF", "QA", "FC", "FX"):
        raise ValueError(candidate)

    def collate(c, rows, cand, valid=False, walk_cols=None):
        s = str(c.get("split", split))
        if s not in ("train", "valid"):
            raise ValueError(f"refusing split {s!r}: official test is never read in this round")
        if cand in ("QF", "QA"):
            return q_collate({"ad": c, "view": _view_for(s)}, rows, cand,
                             valid=valid, walk_cols=walk_cols)
        return f_collate({"ad": c}, rows, cand, valid=valid, walk_cols=walk_cols)

    return collate
