"""Data layer for the route-clarification round.

Read-only reuse of the previous round's verified data stack
(``cr_data.build_bundle``: frozen official caches, ``lipo_v1`` FeatureSpec,
Train-only normalisation, split-local -> official ``row_id`` mapping) plus one
new adapter: the *official OGB* categorical graph tensors needed by ``G0`` and a
single collate that produces both the LCRC arrays (for ``L``) and the GINE arrays
(for ``G0``/``Gx``) with the **same molecule/atom order**.

No label field is ever added to a structural cache; targets are attached per
batch from the frozen bundle exactly like the previous round.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import gr_common as C

_PATHS_READY = False


def prepare_imports() -> None:
    global _PATHS_READY
    if _PATHS_READY:
        return
    for path in (str(C.RC_CODE), str(C.SQL_CODE), str(C.PREP_CODE), str(C.DISSECT_CODE)):
        if path not in sys.path:
            sys.path.insert(0, path)
    _PATHS_READY = True


def modules() -> dict[str, Any]:
    prepare_imports()
    import cr_common  # noqa: F401
    import cr_data
    import cr_models
    import feature_spec
    import lcrc_data
    import sq_data
    import sq_lcrc

    return {
        "cr_common": cr_common,
        "cr_data": cr_data,
        "cr_models": cr_models,
        "feature_spec": feature_spec,
        "lcrc_data": lcrc_data,
        "sq_data": sq_data,
        "sq_lcrc": sq_lcrc,
    }


_BUNDLE_CACHE: dict[str, Any] = {}


def bundle(dataset: str, *, verify: bool = True) -> Any:
    if dataset in _BUNDLE_CACHE:
        return _BUNDLE_CACHE[dataset]
    mods = modules()
    b = mods["cr_data"].build_bundle(dataset, verify=verify)
    _BUNDLE_CACHE[dataset] = b
    return b


# ---------------------------------------------------------------------------
# official OGB categorical graph tensors
# ---------------------------------------------------------------------------


class GineCache:
    """Per-graph official graph tensors, concatenated on demand.

    Node order inside every graph is the frozen ``lipo_raw`` atom order, which is
    shared by the LCRC cache (verified in :meth:`audit_against_lcrc`).
    """

    def __init__(self, payload: Mapping[str, Any], global_row_ids: Mapping[str, np.ndarray]) -> None:
        self.payload = payload
        self.node_ptr = np.asarray(payload["node_ptr"], dtype=np.int64)
        self.bond_ptr = np.asarray(payload["bond_ptr"], dtype=np.int64)
        self.atom_feat = np.asarray(payload["atom_feat"], dtype=np.int64)
        self.degree = np.asarray(payload["degree"], dtype=np.int64)
        self.min_ring = np.asarray(payload["min_ring"], dtype=np.int64)
        self.bond_u = np.asarray(payload["bond_u"], dtype=np.int64)
        self.bond_v = np.asarray(payload["bond_v"], dtype=np.int64)
        self.bond_feat = np.asarray(payload["bond_feat"], dtype=np.int64)
        self.row_ids = {split: np.asarray(global_row_ids[split], dtype=np.int64) for split in C.SPLITS}
        self.n_graphs = {split: int(self.row_ids[split].size) for split in C.SPLITS}

    # -- per-graph views ---------------------------------------------------
    def graph_atoms(self, row: int) -> slice:
        return slice(int(self.node_ptr[row]), int(self.node_ptr[row + 1]))

    def graph_bonds(self, row: int) -> slice:
        return slice(int(self.bond_ptr[row]), int(self.bond_ptr[row + 1]))

    def n_atoms_of(self, row: int) -> int:
        return int(self.node_ptr[row + 1] - self.node_ptr[row])

    # -- categorical vocabulary -------------------------------------------
    def category_counts(self) -> dict[str, int]:
        counts = {f"atom_{j}": int(self.atom_feat[:, j].max()) + 1 for j in range(self.atom_feat.shape[1])}
        counts.update({f"bond_{j}": int(self.bond_feat[:, j].max()) + 1 for j in range(self.bond_feat.shape[1])})
        counts["degree"] = int(self.degree.max()) + 1
        counts["ring"] = int(self.min_ring.max()) + 1
        return counts

    # -- batch assembly ----------------------------------------------------
    def batch(self, split: str, rows: Sequence[int]):
        import torch

        rows = np.asarray(rows, dtype=np.int64)
        gids = self.row_ids[split][rows]
        x_parts, deg_parts, ring_parts, eidx_parts, ecat_parts = [], [], [], [], []
        counts: list[int] = []
        node_off = 0
        edge_off = 0
        for gid in gids.tolist():
            a0, a1 = int(self.node_ptr[gid]), int(self.node_ptr[gid + 1])
            b0, b1 = int(self.bond_ptr[gid]), int(self.bond_ptr[gid + 1])
            n = a1 - a0
            counts.append(n)
            x_parts.append(self.atom_feat[a0:a1])
            deg_parts.append(self.degree[a0:a1])
            ring_parts.append(self.min_ring[a0:a1])
            # the frozen OGB cache stores molecule-local bond endpoints
            bu = self.bond_u[b0:b1]
            bv = self.bond_v[b0:b1]
            bf = self.bond_feat[b0:b1]
            eidx_parts.append(
                np.concatenate(
                    [np.stack([bu + node_off, bv + node_off]), np.stack([bv + node_off, bu + node_off])], axis=1
                )
            )
            ecat_parts.append(np.concatenate([bf, bf], axis=0))
            node_off += n
            edge_off += 2 * (b1 - b0)
        x_cat = np.concatenate(x_parts, axis=0) if x_parts else np.zeros((0, 9), np.int64)
        deg = np.concatenate(deg_parts) if deg_parts else np.zeros(0, np.int64)
        ring = np.concatenate(ring_parts) if ring_parts else np.zeros(0, np.int64)
        edge_index = np.concatenate(eidx_parts, axis=1) if eidx_parts else np.zeros((2, 0), np.int64)
        edge_cat = np.concatenate(ecat_parts, axis=0) if ecat_parts else np.zeros((0, 3), np.int64)
        atom_batch = np.repeat(np.arange(len(counts), dtype=np.int64), np.asarray(counts, dtype=np.int64))
        return {
            "x_cat": torch.as_tensor(x_cat, dtype=torch.long),
            "deg": torch.as_tensor(deg, dtype=torch.long),
            "ring": torch.as_tensor(ring, dtype=torch.long),
            "edge_index": torch.as_tensor(edge_index, dtype=torch.long),
            "edge_cat": torch.as_tensor(edge_cat, dtype=torch.long),
            "gine_batch": torch.as_tensor(atom_batch, dtype=torch.long),
            "gine_n_atoms": torch.as_tensor(np.asarray(counts, dtype=np.int64), dtype=torch.long),
        }

    # -- atom-order audit --------------------------------------------------
    def audit_against_lcrc(self, b, split: str) -> dict[str, Any]:
        cache = b.caches[split]
        atom_ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)
        bond_ptr = np.asarray(cache["bond_ptr"], dtype=np.int64)
        gids = self.row_ids[split]
        mismatches = []
        for local, gid in enumerate(gids.tolist()):
            n_cache = int(atom_ptr[local + 1] - atom_ptr[local])
            if n_cache != self.n_atoms_of(gid):
                mismatches.append({"row": int(gid), "reason": "atom_count", "a": n_cache, "b": self.n_atoms_of(gid)})
                continue
            lo, hi = int(atom_ptr[local]), int(atom_ptr[local + 1])
            cache_scal = np.asarray(cache["atom_scalar"])[lo:hi, :]
            names = [f.name for f in b.spec.atom_scalar_fields]
            col = names.index("log1p_degree")
            raw_deg = np.expm1(cache_scal[:, col].astype(np.float64))
            a0, a1 = int(self.node_ptr[gid]), int(self.node_ptr[gid + 1])
            if not np.allclose(np.rint(raw_deg), self.degree[a0:a1].astype(np.float64)):
                mismatches.append({"row": int(gid), "reason": "degree_order"})
                continue
            cb = np.asarray(cache["bond_index"])[int(bond_ptr[local]) : int(bond_ptr[local + 1])]
            b0, b1 = int(self.bond_ptr[gid]), int(self.bond_ptr[gid + 1])
            pu = self.bond_u[b0:b1]
            pv = self.bond_v[b0:b1]
            if not (np.array_equal(cb[:, 0], pu) and np.array_equal(cb[:, 1], pv)):
                mismatches.append({"row": int(gid), "reason": "bond_order"})
        return {
            "split": split,
            "n_graphs": int(gids.size),
            "n_mismatch": len(mismatches),
            "mismatch_examples": mismatches[:5],
            "atom_order_shared": len(mismatches) == 0,
        }


_GINE_CACHE: dict[str, GineCache] = {}


def gine_cache(dataset: str) -> GineCache:
    if dataset not in _GINE_CACHE:
        b = bundle(dataset)
        _GINE_CACHE[dataset] = GineCache(b.payload, b.global_row_ids)
    return _GINE_CACHE[dataset]


# ---------------------------------------------------------------------------
# unified collate
# ---------------------------------------------------------------------------


def collate(b, split: str, rows: Sequence[int]) -> dict[str, Any]:
    """LCRC tensors + official categorical tensors for the *same* molecules."""
    cache = b.caches[split]
    out = b.collate(cache, rows)
    out.update(gine_cache(b.dataset).batch(split, rows))
    out["n_graphs"] = int(len(rows))
    return out


def eval_order(b, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(cache-local order, official row ids, raw targets) in ascending row_id."""
    order, row_ids = b.row_ids_sorted(split)
    return order, row_ids, b.targets_raw[split][order]


def size_feature(b, split: str) -> np.ndarray:
    return np.log1p(np.asarray(b.n_atoms[split], dtype=np.float64))


def size_scaler(b) -> tuple[float, float]:
    train = size_feature(b, "train")
    return float(train.mean()), float(train.std(ddof=0))


# ---------------------------------------------------------------------------
# oracle target bundle (same graphs / splits, different Train-only target)
# ---------------------------------------------------------------------------


class TargetOverride:
    """A thin bundle view whose raw targets are an oracle graph label.

    Structural caches, normalisation, row ids and splits are untouched; only the
    Train/Val/Test target columns and the Train mean/std change.
    """

    def __init__(self, base: Any, targets: Mapping[str, np.ndarray], label: str) -> None:
        self.base = base
        self.dataset = base.dataset
        self.lipo_raw = base.lipo_raw
        self.spec = base.spec
        self.norm = base.norm
        self.caches = base.caches
        self.global_row_ids = base.global_row_ids
        self.n_atoms = base.n_atoms
        self.smiles = base.smiles
        self.active_global = base.active_global
        self.contexts = base.contexts
        self.collate = base.collate
        self.extra = base.extra
        self.payload = base.payload
        self.targets_raw = {split: np.asarray(targets[split], dtype=np.float64) for split in C.SPLITS}
        train = self.targets_raw["train"]
        self.scaler = {
            "mean": float(train.mean()),
            "std": float(train.std(ddof=0)),
            "source": f"oracle:{label}:official-train",
            "n": int(train.size),
            "fit_split": "train",
            "ddof": 0,
            "valid_or_test_used": False,
        }
        self.y_std = {
            split: (self.targets_raw[split] - self.scaler["mean"]) / self.scaler["std"] for split in C.SPLITS
        }
        self.label = label
        self.build_seconds = float(getattr(base, "build_seconds", 0.0))


def oracle_bundle(dataset: str, property_name: str) -> Any:
    """Load the committed atom-contribution oracle table and override targets."""
    b = bundle(dataset)
    path = C.ANALYSIS_DIR / "oracle_contrib" / f"{dataset}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"missing committed oracle contributions: {path}")
    with np.load(path) as handle:
        payload = {key: handle[key] for key in handle.files}
    if property_name not in payload:
        raise KeyError(f"{property_name} not in {path}")
    graph = np.asarray(payload[property_name], dtype=np.float64)
    targets = {}
    for split in C.SPLITS:
        gids = b.global_row_ids[split]
        targets[split] = graph[gids]
    return TargetOverride(b, targets, property_name)
