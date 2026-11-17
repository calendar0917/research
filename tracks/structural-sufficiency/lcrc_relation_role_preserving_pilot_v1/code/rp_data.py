"""Data layer: the frozen C1 caches, the frozen S2 routing cache, and the roles.

Read-only reuse
---------------
* ``rr_data.build_bundle`` (previous round) -> the verified frozen
  ``ogbg-mollipo`` caches, the Train-only feature normalisation, the official
  split row ids, the two frozen targets (official ``exp`` label and RDKit
  ``WC_LOGP``), the Train-only target scalers, and the frozen per-split topology
  routing cache (bridges / cycle blocks / bridge chains / isolated atoms).
* ``lcrc_data.collate`` -> the base C1 batch tensors, byte for byte the same
  collate the C1 baseline uses.

New in this round (and nowhere else)
------------------------------------
1. the **REAL** role of every S2 occurrence: ``path_cross`` of the frozen cache
   (0 = both endpoint bonds owned by the same block, 1 = cross-boundary);
2. the **CONTROL** role: a fixed, label-free, atom-order-free hash of the typed
   S2 record ``(path_cat, path_end63, path_closure)``; the mapping is frozen in
   ``cache/`` (one row per observed S2 type) and re-derived/verified on load;
3. a collate that adds the two role vectors and the batch-local block owners of
   every S2 occurrence (needed by the native block mask of the analysis).

Nothing here reads a property label or an oracle contribution to *build* a
role: the REAL role is a function of the bond graph, the CONTROL role is a
function of the S2 type code.  Targets are attached later by the training loop.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

import rp_common as C

M = C.modules()


# ---------------------------------------------------------------------------
# CONTROL role: label-free S2 type hash
# ---------------------------------------------------------------------------


def control_hash_role(path_cat: int, end63: int, closure: int) -> int:
    """Fixed hash of one typed S2 record -> {0, 1}.

    The hash input is exactly the frozen S2 *type* triple.  It contains no atom
    index, no bond id, no molecule row id, no property label and no oracle
    value, so it is invariant to atom / bond renumbering by construction.
    """
    key = f"{C.CONTROL_SALT}|{int(path_cat)}|{int(end63)}|{int(closure)}".encode()
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return int.from_bytes(digest, "big") & 1


def triple_key(path_cat, end63, closure) -> np.ndarray:
    """Compact integer lookup key of a typed S2 record (bijective on the codes)."""
    pc = np.asarray(path_cat, dtype=np.int64)
    e = np.asarray(end63, dtype=np.int64)
    k = np.asarray(closure, dtype=np.int64)
    return (pc * 80 + e) * 2 + k


def roles_from_triples(
    table: Mapping[str, np.ndarray], path_cat, end63, closure
) -> np.ndarray:
    """Map typed S2 records to their frozen CONTROL role (0/1)."""
    pc = np.asarray(path_cat, dtype=np.int64)
    e = np.asarray(end63, dtype=np.int64)
    k = np.asarray(closure, dtype=np.int64)
    keys = triple_key(pc, e, k)
    table_key = np.asarray(table["key"], dtype=np.int64)
    table_role = np.asarray(table["role"], dtype=np.uint8)
    if table_key.size == 0 or not bool((np.diff(table_key) > 0).all()):
        raise RuntimeError("CONTROL role table keys must be non-empty and strictly increasing")
    pos = np.searchsorted(table_key, keys)
    if bool((pos >= table_key.size).any()):
        missing = np.unique(keys[pos >= table_key.size])[:10]
        raise KeyError(f"CONTROL role table is missing S2 type keys {missing.tolist()}")
    if not np.array_equal(table_key[pos], keys):
        missing = np.unique(keys[table_key[pos] != keys])[:10]
        raise KeyError(f"CONTROL role table is missing S2 type keys {missing.tolist()}")
    return table_role[pos]


def collect_types(caches: Mapping[str, Mapping[str, Any]]) -> dict[str, np.ndarray]:
    """Every observed typed S2 record across the given split caches (sorted)."""
    rows = []
    for split in C.SPLITS:
        cache = caches[split]
        pc = np.asarray(cache["path_cat"], dtype=np.int64)
        e = np.asarray(cache["path_end63"], dtype=np.int64)
        k = np.asarray(cache["path_closure"], dtype=np.int64)
        rows.append(np.stack([pc, e, k], axis=1))
    allrows = np.concatenate(rows, axis=0)
    uniq = np.unique(allrows, axis=0)
    keys = triple_key(uniq[:, 0], uniq[:, 1], uniq[:, 2])
    order = np.argsort(keys)
    uniq = uniq[order]
    keys = keys[order]
    role = np.asarray(
        [control_hash_role(pc, e, k) for pc, e, k in uniq], dtype=np.uint8
    )
    return {
        "key": keys.astype(np.int64),
        "path_cat": uniq[:, 0].astype(np.int64),
        "path_end63": uniq[:, 1].astype(np.int64),
        "path_closure": uniq[:, 2].astype(np.int64),
        "role": role,
    }


def build_control_table(caches: Mapping[str, Mapping[str, Any]], *, force: bool = False) -> dict[str, Any]:
    """Freeze (or re-verify) the CONTROL type -> role table under ``cache/``."""
    C.ensure_dir(C.CACHE_DIR)
    table = collect_types(caches)
    # the table must be exactly the hash function, never a fitted assignment
    recomputed = np.asarray(
        [control_hash_role(a, b, c) for a, b, c in zip(table["path_cat"], table["path_end63"], table["path_closure"])],
        dtype=np.uint8,
    )
    if not np.array_equal(recomputed, table["role"]):
        raise RuntimeError("CONTROL role table does not match the frozen hash function")
    counts = {
        split: {
            "group0": int((roles_from_triples(table, caches[split]["path_cat"], caches[split]["path_end63"], caches[split]["path_closure"]) == 0).sum()),
            "group1": int((roles_from_triples(table, caches[split]["path_cat"], caches[split]["path_end63"], caches[split]["path_closure"]) == 1).sum()),
        }
        for split in C.SPLITS
    }
    for split, entry in counts.items():
        total = int(caches[split]["path_cat"].size)
        if entry["group0"] + entry["group1"] != total:
            raise RuntimeError(f"{split}: CONTROL groups do not partition the S2 occurrences")
        if entry["group0"] == 0 or entry["group1"] == 0:
            raise RuntimeError(f"{split}: CONTROL grouping is degenerate ({entry})")

    path = C.control_table_path()
    digest = C.sha256_file(path) if path.is_file() else None
    manifest_path = C.control_manifest_path()
    previous = C.read_json(manifest_path) if manifest_path.is_file() else {}
    reusable = (
        not force
        and digest is not None
        and previous.get("sha256") == digest
        and previous.get("table_sha256") == C.sha256_json({k: C.sha256_array(np.ascontiguousarray(v)) for k, v in table.items()})
    )
    if not reusable:
        C.atomic_write_npz(path, {k: np.ascontiguousarray(v) for k, v in table.items()})
        digest = C.sha256_file(path)
    loaded = load_control_table(verify=False)
    if C.sha256_json({k: C.sha256_array(np.ascontiguousarray(v)) for k, v in loaded.items()}) != C.sha256_json(
        {k: C.sha256_array(np.ascontiguousarray(v)) for k, v in table.items()}
    ):
        raise RuntimeError("CONTROL role table on disk does not match the rebuilt table")
    manifest = {
        "round_id": C.ROUND_ID,
        "generated_at": None,
        "generator": "rp_data.build_control_table",
        "salt": C.CONTROL_SALT,
        "hash": "blake2b(salt|path_cat|path_end63|path_closure, digest_size=8), low bit",
        "role_rule": "one class per observed typed S2 record; every occurrence inherits its type's class",
        "labels_used": False,
        "oracle_used": False,
        "atom_index_used": False,
        "row_id_used": False,
        "n_types": int(table["key"].size),
        "counts": counts,
        "sha256": digest,
        "table_sha256": C.sha256_json({k: C.sha256_array(np.ascontiguousarray(v)) for k, v in table.items()}),
    }
    C.atomic_write_json(manifest_path, manifest)
    return {"table": table, "manifest": manifest}


def load_control_table(*, verify: bool = True) -> dict[str, np.ndarray]:
    path = C.control_table_path()
    if not path.is_file():
        raise FileNotFoundError(f"missing CONTROL role table {path}; run the prepare stage first")
    with np.load(path, allow_pickle=False) as mesh:
        table = {key: np.asarray(mesh[key]) for key in mesh.files}
    if verify:
        expected = C.read_json(C.control_manifest_path())
        if C.sha256_file(path) != expected.get("sha256"):
            raise RuntimeError("CONTROL role table sha256 does not match the frozen manifest")
        recomputed = np.asarray(
            [control_hash_role(a, b, c) for a, b, c in zip(table["path_cat"], table["path_end63"], table["path_closure"])],
            dtype=np.uint8,
        )
        if not np.array_equal(recomputed, table["role"]):
            raise RuntimeError("CONTROL role table is not the frozen hash function")
    return table


# ---------------------------------------------------------------------------
# bundle
# ---------------------------------------------------------------------------


@dataclass
class Bundle:
    base: Any                                   # rr_data.Bundle of the previous round
    control: dict[str, np.ndarray]              # per split: uint8 role over all S2 occurrences
    control_table: dict[str, np.ndarray]
    control_manifest: dict[str, Any]
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def dataset(self) -> str:
        return self.base.dataset

    @property
    def cr(self):
        """The ``cr_data.Bundle`` underneath the previous round's wrapper."""
        return self.base.base

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
    def routing(self):
        return self.base.routing

    @property
    def routing_stats(self):
        return self.base.routing_stats

    @property
    def routing_fingerprints(self):
        return self.base.routing_fingerprints

    @property
    def global_row_ids(self):
        return self.base.global_row_ids

    @property
    def split_sizes(self) -> dict[str, int]:
        return self.base.split_sizes

    def targets(self, task: str, split: str) -> np.ndarray:
        return self.base.targets(task, split)

    def target_std(self, task: str, split: str) -> np.ndarray:
        return self.base.target_std(task, split)

    def scaler(self, task: str) -> dict[str, float]:
        return self.base.scaler(task)

    def roles(self, split: str, model: str) -> np.ndarray:
        key = C.MODEL_ROLE_KEY[str(model)]
        if key == "path_role_real":
            return np.asarray(self.routing[split]["path_cross"], dtype=np.uint8)
        if key == "path_role_control":
            return np.asarray(self.control[split], dtype=np.uint8)
        raise KeyError(key)

    def collate(self, split: str, rows: Sequence[int]) -> dict[str, Any]:
        return collate(self, split, rows)


_BUNDLE_CACHE: dict[bool, Bundle] = {}


def verify_reuse_files() -> dict[str, Any]:
    """Byte-level identity of the frozen routing caches reused from the previous round."""
    out = {"round_id": C.PRED_DIR.name, "files": {}, "all_match": True}
    for split in C.SPLITS:
        path = C.PRED_CACHE / f"ogbg_mollipo_{split}_topology-routing-cache-v1.npz"
        expected = C.EXPECTED_ROUTING_CACHE_SHA256[split]
        actual = C.sha256_file(path) if path.is_file() else None
        match = bool(actual == expected)
        out["files"][split] = {
            "path": str(path),
            "expected_sha256": expected,
            "actual_sha256": actual,
            "match": match,
        }
        out["all_match"] = out["all_match"] and match
    if not out["all_match"]:
        raise RuntimeError(f"frozen routing cache fingerprint mismatch: {out['files']}")
    return out


def build_bundle(*, verify: bool = True, require_routing: bool = True) -> Bundle:
    if bool(verify) in _BUNDLE_CACHE:
        return _BUNDLE_CACHE[bool(verify)]
    files = verify_reuse_files() if verify else {}
    base = M["rr_data"].build_bundle(verify=verify, require_routing=require_routing)
    if dict(base.split_sizes) != C.EXPECTED_SPLIT_SIZES:
        raise RuntimeError(f"split sizes {base.split_sizes} != frozen {C.EXPECTED_SPLIT_SIZES}")
    for split in C.SPLITS:
        expected = C.EXPECTED_ROUTING_FINGERPRINT[split]
        if verify and base.routing_fingerprints[split] != expected:
            raise RuntimeError(
                f"{split}: routing fingerprint {base.routing_fingerprints[split]} != frozen {expected}"
            )
        cache = base.caches[split]
        if np.asarray(base.routing[split]["path_cross"]).shape != np.asarray(cache["path_cat"]).shape:
            raise RuntimeError(f"{split}: routing role vector does not align with the S2 occurrences")
    table = load_control_table(verify=bool(verify))
    control = {
        split: roles_from_triples(
            table,
            base.caches[split]["path_cat"],
            base.caches[split]["path_end63"],
            base.caches[split]["path_closure"],
        )
        for split in C.SPLITS
    }
    manifest = C.read_json(C.control_manifest_path())
    bundle = Bundle(
        base=base,
        control=control,
        control_table=table,
        control_manifest=manifest,
        extra={
            "reuse_files": files,
            "spec_hash": base.extra["spec_hash"],
            "norm_fingerprint": base.extra["norm_fingerprint"],
            "graph_cache_sha256": base.extra["graph_cache_sha256"],
            "lipo_raw_sha256": base.extra["lipo_raw_sha256"],
            "n_bond_types": int(base.extra["n_bond_types"]),
            "wc_logp_label_sha256": base.extra.get("wc_logp_label_sha256"),
            "oracle_contrib_sha256": base.extra.get("oracle_contrib_sha256"),
            "control_table_sha256": manifest["sha256"],
        },
    )
    _BUNDLE_CACHE[bool(verify)] = bundle
    return bundle


# ---------------------------------------------------------------------------
# collate: base C1 batch + roles + batch-local block owners
# ---------------------------------------------------------------------------


def _offsets(counts: np.ndarray) -> np.ndarray:
    if counts.size == 0:
        return np.zeros(0, dtype=np.int64)
    return np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)


def path_index(cache: Mapping[str, Any], rows: np.ndarray, atom_off: np.ndarray, atom_batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flat S2 indices and molecule id of every occurrence of the given rows.

    Mirrors ``lcrc_data.collate`` exactly; the caller re-asserts the alignment
    against the batch tensors before using the roles.
    """
    atom_ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)
    path_ptr = np.asarray(cache["path_ptr"], dtype=np.int64)
    atom_counts = atom_ptr[rows + 1] - atom_ptr[rows]
    atom_idx = np.concatenate(
        [np.arange(int(a0), int(a1), dtype=np.int64) for a0, a1 in zip(atom_ptr[rows], atom_ptr[rows + 1])]
    ) if rows.size else np.zeros(0, dtype=np.int64)
    path_counts = path_ptr[atom_idx + 1] - path_ptr[atom_idx] if atom_idx.size else np.zeros(0, dtype=np.int64)
    path_idx = np.concatenate(
        [np.arange(int(path_ptr[a]), int(path_ptr[a]) + int(c), dtype=np.int64) for a, c in zip(atom_idx, path_counts)]
    ) if atom_idx.size else np.zeros(0, dtype=np.int64)
    path_graph = np.repeat(atom_batch, path_counts) if atom_idx.size else np.zeros(0, dtype=np.int64)
    if path_idx.size != int(path_counts.sum()):
        raise RuntimeError("path index construction lost occurrences")
    if int(atom_counts.sum()) != int(atom_batch.size):
        raise RuntimeError("atom index construction disagrees with the batch")
    del atom_off
    return path_idx, path_graph


def collate(bundle: Bundle, split: str, rows: Sequence[int]) -> dict[str, Any]:
    return collate_arrays(
        bundle.caches[split],
        bundle.routing[split],
        bundle.roles(split, "RR_REAL"),
        bundle.roles(split, "RR_CONTROL"),
        rows,
    )


def collate_arrays(
    cache: Mapping[str, Any],
    routing: Mapping[str, np.ndarray],
    role_real_all: np.ndarray,
    role_control_all: np.ndarray,
    rows: Sequence[int],
) -> dict[str, Any]:
    """Base C1 batch + roles + batch-local block owners (split-agnostic core)."""
    import torch

    rows = np.asarray(rows, dtype=np.int64)
    base = M["lcrc_data"].collate(cache, rows)
    n_mol = int(rows.size)
    atom_ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)
    block_ptr = np.asarray(routing["block_ptr"], dtype=np.int64)

    atom_counts = np.asarray(base["atom_counts"].detach().cpu().numpy(), dtype=np.int64)
    atom_off = _offsets(atom_counts)
    atom_batch = np.asarray(base["atom_batch"].detach().cpu().numpy(), dtype=np.int64)
    path_idx, path_graph = path_index(cache, rows, atom_off, atom_batch)

    # the role vectors must be aligned with exactly the batch's S2 occurrences
    pc = np.asarray(cache["path_cat"], dtype=np.int64)[path_idx]
    if not np.array_equal(pc, np.asarray(base["path_cat"].detach().cpu().numpy(), dtype=np.int64)):
        raise RuntimeError("path index does not align with the collated path_cat")
    pe = np.asarray(cache["path_end"], dtype=np.int64)[path_idx] + atom_off[path_graph]
    if not np.array_equal(pe, np.asarray(base["path_end"].detach().cpu().numpy(), dtype=np.int64)):
        raise RuntimeError("path index does not align with the collated path_end")
    pg = np.asarray(base["path_graph"].detach().cpu().numpy(), dtype=np.int64)
    if not np.array_equal(path_graph, pg):
        raise RuntimeError("path index does not align with the collated path_graph")

    role_real = np.asarray(role_real_all, dtype=np.uint8)[path_idx]
    role_control = np.asarray(role_control_all, dtype=np.uint8)[path_idx]

    # batch-local block owners of every S2 occurrence (for the native block mask).
    # ``path_owner1/2`` are block indices *within* the molecule, so they only need
    # the molecule's block offset.
    block_counts = (block_ptr[rows + 1] - block_ptr[rows]).astype(np.int64) if n_mol else np.zeros(0, np.int64)
    block_off = _offsets(block_counts)
    owner1 = np.asarray(routing["path_owner1"], dtype=np.int64)[path_idx]
    owner2 = np.asarray(routing["path_owner2"], dtype=np.int64)[path_idx]
    owner1 = owner1 + block_off[path_graph] if path_graph.size else owner1
    owner2 = owner2 + block_off[path_graph] if path_graph.size else owner2
    if path_graph.size:
        lo = block_off[path_graph]
        hi = lo + block_counts[path_graph]
        if bool(((owner1 < lo) | (owner1 >= hi) | (owner2 < lo) | (owner2 >= hi)).any()):
            raise RuntimeError("S2 block owner outside its molecule's block window")

    out = dict(base)
    out.update(
        {
            "path_role_real": torch.from_numpy(role_real.astype(np.int64)),
            "path_role_control": torch.from_numpy(role_control.astype(np.int64)),
            "path_owner1_local": torch.from_numpy(owner1.astype(np.int64)),
            "path_owner2_local": torch.from_numpy(owner2.astype(np.int64)),
            "block_off": torch.from_numpy(block_off.astype(np.int64)),
            "blocks_per_graph": torch.from_numpy(block_counts.astype(np.int64)),
            "n_blocks": int(block_counts.sum()),
            "cache_rows": torch.from_numpy(rows.astype(np.int64)),
        }
    )
    return out


# ---------------------------------------------------------------------------
# audits
# ---------------------------------------------------------------------------


def role_group_audit(bundle: Bundle) -> dict[str, Any]:
    """Group sizes of REAL and CONTROL, split by split, plus per-molecule coverage."""
    out: dict[str, Any] = {
        "dataset": bundle.dataset,
        "roles": {
            "RR_REAL": {
                "class_names": ["internal", "cross"],
                "rule": C.MODEL_ROLE_SOURCE["RR_REAL"],
                "source": "frozen routing cache path_cross (owner(e1) vs owner(e2))",
            },
            "RR_CONTROL": {
                "class_names": ["hash0", "hash1"],
                "rule": C.MODEL_ROLE_SOURCE["RR_CONTROL"],
                "salt": C.CONTROL_SALT,
                "n_types": int(bundle.control_table["key"].size),
            },
        },
        "splits": {},
    }
    all_ok = True
    for split in C.SPLITS:
        cache = bundle.caches[split]
        path_ptr = np.asarray(cache["path_ptr"], dtype=np.int64)
        atom_ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)
        n_mol = int(cache["n_molecules"])
        entry: dict[str, Any] = {}
        for model in C.MODELS:
            role = bundle.roles(split, model)
            total = int(role.size)
            counts = [int((role == k).sum()) for k in range(C.N_ROLE_CLASSES)]
            # per-molecule class coverage
            # ``path_ptr`` is atom-level, so a molecule's path window is
            # [path_ptr[atom_ptr[row]], path_ptr[atom_ptr[row + 1]]).
            per_mol_counts = []
            for row in range(n_mol):
                lo = int(path_ptr[int(atom_ptr[row])])
                hi = int(path_ptr[int(atom_ptr[row + 1])])
                local = role[lo:hi]
                per_mol_counts.append((int((local == 0).sum()), int((local == 1).sum())))
            arr = np.asarray(per_mol_counts, dtype=np.int64)
            n0 = arr[:, 0]
            n1 = arr[:, 1]
            entry[model] = {
                "field": C.MODEL_ROLE_KEY[model],
                "class_names": list(C.ROLE_NAMES[model]),
                "n_occurrences": total,
                "n_group0": counts[0],
                "n_group1": counts[1],
                "fraction_group0": (counts[0] / total) if total else 0.0,
                "fraction_group1": (counts[1] / total) if total else 0.0,
                "size_ratio_g1_over_g0": (counts[1] / counts[0]) if counts[0] else None,
                "n_molecules_with_empty_group0": int((n0 == 0).sum()),
                "n_molecules_with_empty_group1": int((n1 == 0).sum()),
                "n_molecules_both_groups": int(((n0 > 0) & (n1 > 0)).sum()),
                "n_molecules": n_mol,
                "per_molecule_group0": {
                    "mean": float(n0.mean()) if n_mol else 0.0,
                    "median": float(np.median(n0)) if n_mol else 0.0,
                    "max": int(n0.max()) if n_mol else 0,
                },
                "per_molecule_group1": {
                    "mean": float(n1.mean()) if n_mol else 0.0,
                    "median": float(np.median(n1)) if n_mol else 0.0,
                    "max": int(n1.max()) if n_mol else 0,
                },
            }
            ok = (
                counts[0] > 0
                and counts[1] > 0
                and int(sum(counts)) == total
                and int(np.asarray(role).max()) < C.N_ROLE_CLASSES
            )
            entry[model]["pass"] = bool(ok)
            all_ok = all_ok and ok
        entry["group_size_gap"] = {
            model: abs(entry[model]["fraction_group0"] - entry[model]["fraction_group1"])
            for model in C.MODELS
        }
        out["splits"][split] = entry
    out["all_pass"] = bool(all_ok)
    return out


def target_audit(bundle: Bundle) -> dict[str, Any]:
    out = {"dataset": bundle.dataset, "tasks": {}}
    for task in C.TASKS:
        entry = {
            "target": C.TASK_TARGET_NAME[task],
            "scaler": bundle.scaler(task),
            "scaler_source": "official Train split only",
            "valid_or_test_used_for_scaling": False,
            "splits": {},
        }
        for split in C.SPLITS:
            raw = np.asarray(bundle.targets(task, split), dtype=np.float64)
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
    "build_control_table",
    "load_control_table",
    "control_hash_role",
    "roles_from_triples",
    "role_group_audit",
    "target_audit",
    "collate",
    "collate_arrays",
    "path_index",
    "verify_reuse_files",
]
