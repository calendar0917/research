"""Data layer: the frozen official ``ogbg-mollipo`` split plus the routing cache.

Read-only reuse
---------------
* ``cr_data.build_bundle`` -> the verified frozen caches, the ``lipo_v1``
  FeatureSpec, the split-local -> official ``row_id`` map, Train-only feature
  normalisation, the Train-only target scaler of the readout-context round.
* ``lcrc_data.collate`` -> the base batch tensors of the frozen C1 pipeline.

New in this round
-----------------
1. the frozen topology-block routing cache (built once on CPU, SHA-256 pinned);
2. a routed collate that adds the block / interface / bucket tensors;
3. the two authorised target tables (real ``exp`` label; RDKit ``WC_LOGP``),
   each with its **own** Train-only scaler and its own checkpoint pool.  The two
   tasks never share a checkpoint, scaler or selection result.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import rr_common as C
import rr_routing as R

M = C.modules()


# ---------------------------------------------------------------------------
# bundle
# ---------------------------------------------------------------------------


@dataclass
class Bundle:
    base: Any
    routing: dict[str, dict[str, np.ndarray]]
    routing_stats: dict[str, Any]
    routing_fingerprints: dict[str, str]
    targets_raw: dict[str, dict[str, np.ndarray]]
    scalers: dict[str, dict[str, float]]
    y_std: dict[str, dict[str, np.ndarray]]
    extra: dict[str, Any] = field(default_factory=dict)

    # -- convenience -------------------------------------------------------
    @property
    def dataset(self) -> str:
        return self.base.dataset

    @property
    def spec(self):
        return self.base.spec

    @property
    def norm(self):
        return self.base.norm

    @property
    def caches(self):
        return self.base.caches

    @property
    def global_row_ids(self):
        return self.base.global_row_ids

    @property
    def split_sizes(self) -> dict[str, int]:
        return self.base.split_sizes

    # -- target helpers ----------------------------------------------------
    def targets(self, task: str, split: str) -> np.ndarray:
        return self.targets_raw[task][split]

    def target_std(self, task: str, split: str) -> np.ndarray:
        return self.y_std[task][split]

    def scaler(self, task: str) -> dict[str, float]:
        return self.scalers[task]

    def collate(self, split: str, rows: Sequence[int]) -> dict[str, Any]:
        return routed_collate(self.caches[split], self.routing[split], rows)


_BUNDLE_CACHE: dict[tuple[str, bool], Bundle] = {}


def verify_frozen_files() -> dict[str, Any]:
    """Byte-level SHA-256 check of the two frozen upstream caches."""
    mods = M
    paths = {
        "graph": Path(mods["sq_data"].cache_path(C.DATASET)),
        "lipo_raw": Path(mods["sq_data"].lipo_raw_path(C.DATASET)),
    }
    out = {"dataset": C.DATASET, "files": {}}
    ok = True
    for tag, path in paths.items():
        expected = C.EXPECTED_CACHE_SHA256[path.name]
        actual = C.sha256_file(path) if path.is_file() else None
        match = bool(actual == expected)
        ok = ok and match
        out["files"][path.name] = {
            "tag": tag,
            "path": str(path),
            "expected_sha256": expected,
            "actual_sha256": actual,
            "match": match,
        }
    out["all_match"] = ok
    if not ok:
        raise RuntimeError(f"frozen cache fingerprint mismatch: {out['files']}")
    return out


def load_routing(split: str, *, verify: bool = True) -> tuple[dict[str, np.ndarray], str]:
    path = C.routing_cache_path(split)
    if not path.is_file():
        raise FileNotFoundError(
            f"missing frozen routing cache {path}; run the prepare stage on CPU first"
        )
    digest = C.sha256_file(path)
    expected = C.expected_task_cache_sha256()
    if verify and expected:
        if expected.get(split) != digest:
            raise RuntimeError(
                f"{split}: routing cache sha256 {digest} != frozen {expected.get(split)}"
            )
    with np.load(path, allow_pickle=False) as mesh:
        arrays = {key: np.asarray(mesh[key]) for key in mesh.files}
    fingerprint = C.sha256_json(
        {key: C.sha256_array(np.ascontiguousarray(value)) for key, value in sorted(arrays.items())}
    )
    return arrays, fingerprint


def load_wc_logp_labels(base: Any) -> dict[str, np.ndarray]:
    """Verified RDKit WC_LOGP labels for the official original rows.

    The label file is the *computed-property* table of the oracle round; only
    ``source_kind == original`` rows with the fitted split are used, and the
    values are cross-checked against the per-atom contribution cache (which is
    the object used later by the oracle localisation analysis).
    """
    if not C.WC_LOG_PRED_CSV.is_file():
        raise FileNotFoundError(C.WC_LOG_PRED_CSV)
    table: dict[int, float] = {}
    with C.WC_LOG_PRED_CSV.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("source_kind") != "original":
                continue
            if str(row.get("task", "")).lower() != "wc_logp":
                continue
            table[int(row["row_id"])] = float(row["value"])
    if len(table) != C.EXPECTED_TOTAL_ROWS:
        raise RuntimeError(f"WC_LOGP label table has {len(table)} original rows, expected {C.EXPECTED_TOTAL_ROWS}")
    out = {}
    for split in C.SPLITS:
        ids = np.asarray(base.global_row_ids[split], dtype=np.int64)
        out[split] = np.asarray([table[int(i)] for i in ids], dtype=np.float64)
    # conservation identity against the independent per-atom contribution cache
    with np.load(C.ORACLE_CONTRIB_NPZ, allow_pickle=False) as mesh:
        graph_values = np.asarray(mesh["WC_LOGP"], dtype=np.float64)
        atom_values = np.asarray(mesh["WC_LOGP_atom"], dtype=np.float64)
        atom_ptr = np.asarray(mesh["atom_ptr"], dtype=np.int64)
    for split in C.SPLITS:
        ids = np.asarray(base.global_row_ids[split], dtype=np.int64)
        diff = np.abs(graph_values[ids] - out[split])
        if float(diff.max()) > 1e-9:
            raise RuntimeError(f"WC_LOGP label / contribution-cache mismatch on {split}: {diff.max():.3e}")
        for row, global_id in enumerate(ids.tolist()):
            a0, a1 = int(atom_ptr[global_id]), int(atom_ptr[global_id + 1])
            total = float(atom_values[a0:a1].sum())
            if abs(total - out[split][row]) > 1e-8:
                raise RuntimeError(
                    f"WC_LOGP per-atom contributions do not close on row {global_id}: {total} vs {out[split][row]}"
                )
    return out


def build_bundle(
    *, verify: bool = True, require_routing: bool = True, verify_routing_sha: bool = True
) -> Bundle:
    key = (C.DATASET, bool(verify), bool(require_routing), bool(verify_routing_sha))
    if key in _BUNDLE_CACHE:
        return _BUNDLE_CACHE[key]
    files = verify_frozen_files() if verify else {}
    base = M["cr_data"].build_bundle(C.DATASET, verify=verify)
    if base.split_sizes != C.EXPECTED_SPLIT_SIZES:
        raise RuntimeError(f"split sizes {base.split_sizes} != frozen {C.EXPECTED_SPLIT_SIZES}")
    lipo_targets = {split: np.asarray(base.targets_raw[split], dtype=np.float64) for split in C.SPLITS}
    wc_targets = load_wc_logp_labels(base)
    targets = {"lipo": lipo_targets, "wc_logp": wc_targets}
    scalers: dict[str, dict[str, float]] = {}
    y_std: dict[str, dict[str, np.ndarray]] = {}
    for task in C.TASKS:
        scaler = C.target_scaler(targets[task]["train"])
        scalers[task] = scaler
        y_std[task] = {split: (targets[task][split] - scaler["mean"]) / scaler["std"] for split in C.SPLITS}
    routing: dict[str, dict[str, np.ndarray]] = {}
    fingerprints: dict[str, str] = {}
    stats: dict[str, Any] = {}
    for split in C.SPLITS:
        if not require_routing and not C.routing_cache_path(split).is_file():
            routing[split] = {}
            fingerprints[split] = ""
            stats[split] = {}
            continue
        arrays, fingerprint = load_routing(split, verify=bool(verify and verify_routing_sha))
        base_cache = base.caches[split]
        if int(arrays["block_ptr"].size) - 1 != int(base_cache["n_molecules"]):
            raise RuntimeError(f"{split}: routing cache molecule count mismatch")
        if int(arrays["blk_bond"].size) != int(np.asarray(base_cache["bond_feat"]).shape[0]):
            raise RuntimeError(f"{split}: routing cache bond count mismatch")
        routing[split] = arrays
        fingerprints[split] = fingerprint
        stats[split] = R.cross_boundary_molecule_stats(arrays, base_cache)
    bundle = Bundle(
        base=base,
        routing=routing,
        routing_stats=stats,
        routing_fingerprints=fingerprints,
        targets_raw=targets,
        scalers=scalers,
        y_std=y_std,
        extra={
            "frozen_files": files,
            "spec_hash": base.extra["spec_hash"],
            "norm_fingerprint": base.extra["norm_fingerprint"],
            "graph_cache_sha256": str(base.payload.get("cache_sha256")),
            "lipo_raw_sha256": str(base.lipo_raw.get("cache_sha256")),
            "n_bond_types": int(_n_bond_types()),
            "wc_logp_label_sha256": C.sha256_file(C.WC_LOG_PRED_CSV),
            "oracle_contrib_sha256": C.sha256_file(C.ORACLE_CONTRIB_NPZ),
        },
    )
    _BUNDLE_CACHE[key] = bundle
    return bundle


def _n_bond_types() -> int:
    """Bond-type vocabulary size of the frozen spec (codes are 1..n)."""
    return int(M["feature_spec"].load_spec(C.PREP_DIR / "analysis/feature_spec_lipo.json").n_bond_types)


# ---------------------------------------------------------------------------
# routed collate
# ---------------------------------------------------------------------------


def _ranges(ptr: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flat indices of a molecule-level pointer table (one entry per molecule)."""
    starts = ptr[rows].astype(np.int64)
    ends = ptr[rows + 1].astype(np.int64)
    counts = ends - starts
    if rows.size == 0:
        return np.zeros(0, dtype=np.int64), counts
    idx = np.concatenate([np.arange(s, e, dtype=np.int64) for s, e in zip(starts, ends)])
    return idx, counts


def _atom_ranges(ptr: np.ndarray, atom_ptr: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flat indices of an atom-level pointer table for whole molecules."""
    starts = ptr[atom_ptr[rows]].astype(np.int64)
    ends = ptr[atom_ptr[rows + 1]].astype(np.int64)
    counts = ends - starts
    if rows.size == 0:
        return np.zeros(0, dtype=np.int64), counts
    idx = np.concatenate([np.arange(s, e, dtype=np.int64) for s, e in zip(starts, ends)])
    return idx, counts


def _offsets(counts: np.ndarray) -> np.ndarray:
    return np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64) if counts.size else np.zeros(0, np.int64)


def routed_collate(
    cache: Mapping[str, Any],
    routing: Mapping[str, np.ndarray],
    rows: Sequence[int],
) -> dict[str, Any]:
    """Base C1 batch plus block / interface / routing tensors."""
    import torch

    rows = np.asarray(rows, dtype=np.int64)
    base = M["lcrc_data"].collate(cache, rows)
    n_mol = int(rows.size)
    atom_ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)
    bond_ptr = np.asarray(cache["bond_ptr"], dtype=np.int64)
    belong_ptr = np.asarray(routing["belong_ptr"], dtype=np.int64)
    block_ptr = np.asarray(routing["block_ptr"], dtype=np.int64)
    iface_ptr = np.asarray(routing["iface_ptr"], dtype=np.int64)

    def _t(arr, dtype):
        return torch.from_numpy(np.asarray(arr, dtype=dtype))

    # ---- per-molecule offsets -------------------------------------------
    atom_counts = np.asarray(base["atom_counts"].detach().cpu().numpy(), dtype=np.int64)
    bond_counts = np.asarray(base["bond_counts"].detach().cpu().numpy(), dtype=np.int64)
    atom_off = _offsets(atom_counts)
    bond_off = _offsets(bond_counts)
    belong_counts = belong_ptr[atom_ptr[rows + 1]] - belong_ptr[atom_ptr[rows]]
    belong_off = _offsets(belong_counts)
    block_counts = (block_ptr[rows + 1] - block_ptr[rows]).astype(np.int64) if n_mol else np.zeros(0, dtype=np.int64)
    block_off = _offsets(block_counts)
    iface_counts = np.asarray([int(iface_ptr[r + 1] - iface_ptr[r]) for r in rows.tolist()], dtype=np.int64)
    iface_off = _offsets(iface_counts)

    # ---- belong (atom x block) ------------------------------------------
    b_idx, _ = _atom_ranges(belong_ptr, atom_ptr, rows)
    belong_mol = np.repeat(np.arange(n_mol, dtype=np.int64), belong_counts)
    belong_atom = np.asarray(routing["belong_atom"], dtype=np.int64)[b_idx] + atom_off[belong_mol]
    belong_block = np.asarray(routing["belong_block"], dtype=np.int64)[b_idx] + block_off[belong_mol]
    belong_weight = np.asarray(routing["belong_weight"], dtype=np.float32)[b_idx]

    # ---- bonds x owner ---------------------------------------------------
    bond_idx, _ = _ranges(bond_ptr, rows)
    bond_mol = np.repeat(np.arange(n_mol, dtype=np.int64), bond_counts)

    def _belong_to_batch(global_belong: np.ndarray, mol: np.ndarray) -> np.ndarray:
        base_index = belong_ptr[atom_ptr[rows][mol]]
        return global_belong - base_index + belong_off[mol]

    bond_bucket_u = _belong_to_batch(np.asarray(routing["bond_bucket_u"], dtype=np.int64)[bond_idx], bond_mol)
    bond_bucket_v = _belong_to_batch(np.asarray(routing["bond_bucket_v"], dtype=np.int64)[bond_idx], bond_mol)

    # ---- S1 occurrences --------------------------------------------------
    nbr_idx, _ = _atom_ranges(np.asarray(cache["nbr_ptr"], dtype=np.int64), atom_ptr, rows)
    nbr_mol = torch.as_tensor(base["atom_batch"])[base["nbr_atom"]].detach().cpu().numpy()
    nbr_bucket = _belong_to_batch(np.asarray(routing["nbr_bucket"], dtype=np.int64)[nbr_idx], nbr_mol)

    # ---- S2 occurrences --------------------------------------------------
    path_idx, _ = _atom_ranges(np.asarray(cache["path_ptr"], dtype=np.int64), atom_ptr, rows)
    path_mol = torch.as_tensor(base["path_graph"]).detach().cpu().numpy()
    path_bucket_g = np.asarray(routing["path_bucket"], dtype=np.int64)[path_idx]
    path_bucket = np.where(path_bucket_g >= 0, _belong_to_batch(np.maximum(path_bucket_g, 0), path_mol), -1)
    path_iface_g = np.asarray(routing["path_iface"], dtype=np.int64)[path_idx]
    iface_base = iface_ptr[rows][path_mol] if path_mol.size else np.zeros(0, dtype=np.int64)
    path_iface = np.where(path_iface_g >= 0, path_iface_g - iface_base + iface_off[path_mol], -1)

    # ---- blocks / interfaces --------------------------------------------
    blk_idx, _ = _ranges(block_ptr, rows)
    iface_idx, _ = _ranges(iface_ptr, rows)
    blk_n_bonds = np.asarray(routing["blk_n_bonds"], dtype=np.int64)[blk_idx]
    blk_n_atoms = np.asarray(routing["blk_n_atoms"], dtype=np.int64)[blk_idx]
    iface_a_g = np.asarray(routing["iface_a"], dtype=np.int64)[iface_idx] if iface_idx.size else np.zeros(0, np.int64)
    iface_b_g = np.asarray(routing["iface_b"], dtype=np.int64)[iface_idx] if iface_idx.size else np.zeros(0, np.int64)
    iface_mol = np.repeat(np.arange(n_mol, dtype=np.int64), iface_counts)
    if iface_idx.size:
        iface_a = iface_a_g - block_ptr[rows][iface_mol] + block_off[iface_mol]
        iface_b = iface_b_g - block_ptr[rows][iface_mol] + block_off[iface_mol]
    else:
        iface_a = np.zeros(0, dtype=np.int64)
        iface_b = np.zeros(0, dtype=np.int64)

    out = dict(base)
    out.update(
        {
            "belong_atom": _t(belong_atom, np.int64),
            "belong_block": _t(belong_block, np.int64),
            "belong_weight": _t(belong_weight, np.float32),
            "bond_bucket_u": _t(bond_bucket_u, np.int64),
            "bond_bucket_v": _t(bond_bucket_v, np.int64),
            "nbr_bucket": _t(nbr_bucket, np.int64),
            "path_bucket": _t(path_bucket, np.int64),
            "path_iface": _t(path_iface, np.int64),
            "n_blocks": int(block_counts.sum()),
            "n_interfaces": int(iface_counts.sum()),
            "blk_n_bonds": _t(blk_n_bonds, np.int64),
            "blk_n_atoms": _t(blk_n_atoms, np.int64),
            "block_graph": _t(np.repeat(np.arange(n_mol, dtype=np.int64), block_counts), np.int64),
            "iface_graph": _t(np.repeat(np.arange(n_mol, dtype=np.int64), iface_counts), np.int64),
            "iface_a": _t(iface_a, np.int64),
            "iface_b": _t(iface_b, np.int64),
            "n_blocks_per_graph": _t(block_counts, np.int64),
            "n_ifaces_per_graph": _t(iface_counts, np.int64),
            "block_off": _t(block_off, np.int64),
            "iface_off": _t(iface_off, np.int64),
            "belong_off": _t(belong_off, np.int64),
            "bond_off": _t(bond_off, np.int64),
        }
    )
    return out


# ---------------------------------------------------------------------------
# audit helpers
# ---------------------------------------------------------------------------


def target_audit(bundle: Bundle) -> dict[str, Any]:
    out: dict[str, Any] = {"dataset": bundle.dataset, "tasks": {}}
    for task in C.TASKS:
        entry = {
            "target": C.TASK_TARGET_NAME[task],
            "scaler": bundle.scalers[task],
            "scaler_source": "official Train split only",
            "valid_or_test_used_for_scaling": False,
            "splits": {},
        }
        for split in C.SPLITS:
            raw = bundle.targets_raw[task][split]
            entry["splits"][split] = {
                "n": int(raw.size),
                "mean": float(raw.mean()),
                "std": float(raw.std(ddof=0)),
                "min": float(raw.min()),
                "max": float(raw.max()),
                "sha256": C.sha256_array(raw),
            }
        out["tasks"][task] = entry
    return out


__all__ = [
    "Bundle",
    "build_bundle",
    "load_routing",
    "verify_frozen_files",
    "routed_collate",
    "load_wc_logp_labels",
    "target_audit",
]
