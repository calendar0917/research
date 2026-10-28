"""Second-hop path supplement and batching for the pilot.

The frozen LCRC-v1 cache stores, per ordered non-backtracking two-hop path
``v -> w -> u``, only ``(path_cat, path_end63, path_end, path_closure)``; it does
not store the middle atom ``w`` nor the second bond id ``e2 = (w, u)``.  This
module re-derives them from the cache's *raw* adjacency (the same enumeration
order as ``lc_data.build_from_arrays``), verifies every frozen path field
element-wise, and writes a small supplement table.

No label is read here; the official test split is refused by construction.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch

import sh_common as S

INF_DIST = 1 << 20


# ---------------------------------------------------------------------------
# raw enumeration (identical ordering to lc_data.build_from_arrays)
# ---------------------------------------------------------------------------

def adjacency(n_atoms: int, bond_index: np.ndarray, bond_type: np.ndarray):
    """``adj[v] = sorted [(u, bond_type, bond_id)]`` in LCRC-v1 order."""
    adj: list[list[tuple[int, int, int]]] = [[] for _ in range(int(n_atoms))]
    for e, ((u, v), bt) in enumerate(zip(np.asarray(bond_index, dtype=np.int64),
                                         np.asarray(bond_type, dtype=np.int64))):
        adj[int(u)].append((int(v), int(bt), int(e)))
        adj[int(v)].append((int(u), int(bt), int(e)))
    for a in adj:
        a.sort()
    return adj


def molecule_paths(bond_index: np.ndarray, bond_type: np.ndarray, n_atoms: int,
                   atom_type: np.ndarray | None = None) -> dict:
    """All ordered non-backtracking two-hop paths of one molecule (local ids)."""
    adj = adjacency(n_atoms, bond_index, bond_type)
    atom_type = (np.zeros(int(n_atoms), dtype=np.int64) if atom_type is None
                 else np.asarray(atom_type, dtype=np.int64))
    root, mid, end, e1, e2, closure = [], [], [], [], [], []
    nbr_code, nbr_type = [], []
    ptr = [0]
    for v in range(int(n_atoms)):
        nbrs = {u for u, _t, _e in adj[v]}
        for (w, _bt1, f) in adj[v]:
            for (u, _bt2, s) in adj[w]:
                if u == v:
                    continue
                root.append(v)
                mid.append(w)
                end.append(u)
                e1.append(f)
                e2.append(s)
                closure.append(1 if u in nbrs else 0)
        nbr_code.extend(int(bt) for _u, bt, _e in adj[v])
        nbr_type.extend(int(atom_type[u]) for u, _bt, _e in adj[v])
        ptr.append(len(root))
    return {
        "root": np.asarray(root, dtype=np.int64),
        "mid": np.asarray(mid, dtype=np.int64),
        "end": np.asarray(end, dtype=np.int64),
        "e1": np.asarray(e1, dtype=np.int64),
        "e2": np.asarray(e2, dtype=np.int64),
        "closure": np.asarray(closure, dtype=np.uint8),
        "nbr_bt": np.asarray(nbr_code, dtype=np.uint8),
        "nbr_type": np.asarray(nbr_type, dtype=np.int64),
        "ptr": np.asarray(ptr, dtype=np.int64),
        "n_atoms": int(n_atoms),
    }


def enumerate_split(cache: Mapping[str, Any]) -> dict:
    """All paths of one split with global atom ids / global bond ids."""
    ap = np.asarray(cache["atom_ptr"], dtype=np.int64)
    bp = np.asarray(cache["bond_ptr"], dtype=np.int64)
    bi_all = np.asarray(cache["bond_index"], dtype=np.int64)
    bt_all = np.asarray(cache["bond_type"], dtype=np.int64)
    at_all = np.asarray(cache["atom_type"], dtype=np.int64)
    n_mols = int(cache["n_molecules"])
    keys = ("root", "mid", "end", "e1", "e2", "closure")
    parts: dict[str, list[np.ndarray]] = {k: [] for k in keys}
    nbr_parts: list[np.ndarray] = []
    nbr_type_parts: list[np.ndarray] = []
    nbr_ptr = [0]
    per_atom = np.zeros(int(ap[-1]), dtype=np.int64)
    for m in range(n_mols):
        a0, a1 = int(ap[m]), int(ap[m + 1])
        b0, b1 = int(bp[m]), int(bp[m + 1])
        p = molecule_paths(bi_all[b0:b1], bt_all[b0:b1], a1 - a0, at_all[a0:a1])
        parts["root"].append(p["root"] + a0)
        parts["mid"].append(p["mid"] + a0)
        parts["end"].append(p["end"] + a0)
        parts["e1"].append(p["e1"] + b0)
        parts["e2"].append(p["e2"] + b0)
        parts["closure"].append(p["closure"])
        nbr_parts.append(p["nbr_bt"])
        nbr_type_parts.append(p["nbr_type"])
        nbr_ptr.append(nbr_ptr[-1] + len(p["nbr_bt"]))
        per_atom[a0:a1] = np.diff(p["ptr"])
    out = {k: (np.concatenate(v) if v else np.zeros(0, dtype=np.int64))
           for k, v in parts.items()}
    out["nbr_bt"] = np.concatenate(nbr_parts) if nbr_parts else np.zeros(0, np.uint8)
    out["nbr_type"] = (np.concatenate(nbr_type_parts) if nbr_type_parts
                       else np.zeros(0, dtype=np.int64))
    out["nbr_ptr"] = np.asarray(nbr_ptr, dtype=np.int64)
    out["per_atom"] = per_atom
    return out


def bond_endpoints(cache: Mapping[str, Any]) -> np.ndarray:
    ap = np.asarray(cache["atom_ptr"], dtype=np.int64)
    bp = np.asarray(cache["bond_ptr"], dtype=np.int64)
    bi = np.asarray(cache["bond_index"], dtype=np.int64)
    n = int(cache["n_molecules"])
    parts = [bi[int(bp[m]):int(bp[m + 1])] + int(ap[m]) for m in range(n)]
    return np.concatenate(parts) if parts else np.zeros((0, 2), np.int64)


def _agree(a: np.ndarray, b: np.ndarray) -> dict:
    if a.shape != b.shape:
        return {"equal": False, "shape_a": list(a.shape), "shape_b": list(b.shape)}
    mismatch = int(np.sum(np.asarray(a) != np.asarray(b)))
    return {"equal": mismatch == 0, "n": int(a.size), "mismatch": mismatch}


def crosscheck(cache: Mapping[str, Any], paths: Mapping[str, Any]) -> dict:
    """Element-wise agreement between the fresh enumeration and the frozen cache."""
    ap = np.asarray(cache["atom_ptr"], dtype=np.int64)
    at = np.asarray(cache["atom_type"], dtype=np.int64)
    bt = np.asarray(cache["bond_type"], dtype=np.int64)
    root, mid, end, e1, e2 = (paths["root"], paths["mid"], paths["end"],
                              paths["e1"], paths["e2"])
    cat = (at[mid] * 9 + (bt[e1] - 1) * 3 + (bt[e2] - 1)).astype(np.uint8)
    end63 = (at[end] * 3 + (bt[e2] - 1)).astype(np.uint8)
    local_end = (end - ap[np.searchsorted(ap, end, side="right") - 1]).astype(np.int16)

    # nbr_code = neighbour atom type * 3 + (bond type - 1), in adjacency order
    nbr_code = (paths["nbr_type"] * 3 + (paths["nbr_bt"].astype(np.int64) - 1)).astype(np.uint8)

    ends = bond_endpoints(cache)
    be1, be2 = ends[e1], ends[e2]
    ok1 = (((be1[:, 0] == root) & (be1[:, 1] == mid)) |
           ((be1[:, 1] == root) & (be1[:, 0] == mid)))
    ok2 = (((be2[:, 0] == mid) & (be2[:, 1] == end)) |
           ((be2[:, 1] == mid) & (be2[:, 0] == end)))
    root_expected = np.repeat(np.arange(int(ap[-1]), dtype=np.int64), paths["per_atom"])

    checks = {
        "path_cat": _agree(cat, np.asarray(cache["path_cat"], dtype=np.uint8)),
        "path_end63": _agree(end63, np.asarray(cache["path_end63"], dtype=np.uint8)),
        "path_end": _agree(local_end, np.asarray(cache["path_end"], dtype=np.int16)),
        "path_closure": _agree(paths["closure"], np.asarray(cache["path_closure"], dtype=np.uint8)),
        "nbr_code": _agree(nbr_code, np.asarray(cache["nbr_code"], dtype=np.uint8)),
        "path_root_order": _agree(root, root_expected),
    }
    counts = {
        "per_atom_path_count_mismatch": int(
            np.sum(paths["per_atom"] != np.diff(np.asarray(cache["path_ptr"], dtype=np.int64)))),
        "nbr_ptr_total_mismatch": int(np.asarray(cache["nbr_ptr"])[-1] - len(paths["nbr_bt"])),
        "root_equals_end": int(np.sum(root == end)),
        "e1_equals_e2": int(np.sum(e1 == e2)),
        "bond_e1_incident_violations": int(np.sum(~ok1)),
        "bond_e2_incident_violations": int(np.sum(~ok2)),
        "e2_endpoint_is_path_end": int(np.sum(
            ~(((be2[:, 0] == end) | (be2[:, 1] == end))))),
    }
    passed = (all(c["equal"] for c in checks.values())
              and all(v == 0 for v in counts.values()))
    return {
        "n_paths": int(len(root)),
        "n_paths_from_cache": int(len(np.asarray(cache["path_cat"]))),
        "n_atoms": int(len(paths["per_atom"])),
        "codes": checks,
        "counts": counts,
        "passed": bool(passed),
    }


# ---------------------------------------------------------------------------
# supplement table (cache -> per-path middle atom + second bond id)
# ---------------------------------------------------------------------------

def supplement_from_cache(cache: Mapping[str, Any], split: str, *, write: bool = False) -> dict:
    """Enumerate + cross-check + (optionally) write the per-path supplement."""
    if split not in ("train", "valid"):
        raise ValueError(f"{split}: the pilot may only touch train/valid")
    if cache["meta"]["feature_version"] != S.L.FEATURE_VERSION:
        raise RuntimeError(f"{split}: unexpected LCRC feature version")
    paths = enumerate_split(cache)
    xc = crosscheck(cache, paths)
    if not xc["passed"]:
        raise RuntimeError(f"{split}: fresh path enumeration disagrees with the frozen "
                           f"cache: {xc}")
    sup = {
        "split": split,
        "feature_version": S.SUPPLEMENT_VERSION,
        "path_root": torch.from_numpy(paths["root"].astype(np.int32)),
        "path_mid": torch.from_numpy(paths["mid"].astype(np.int32)),
        "path_end": torch.from_numpy(paths["end"].astype(np.int32)),
        "path_e1": torch.from_numpy(paths["e1"].astype(np.int32)),
        "path_bond2": torch.from_numpy(paths["e2"].astype(np.int32)),
        "n_paths": int(len(paths["root"])),
        "n_atoms": int(len(paths["per_atom"])),
        "n_molecules": int(cache["n_molecules"]),
        "source_cache_sha256": S.L.C.sha256_file(S.D.cache_path(split)),
        "source_cache_content_sha256": S.D.content_fingerprint(cache),
        "crosscheck": xc,
        "official_test_loaded": False,
    }
    content = S.supplement_content_hash(sup)
    file_sha = None
    if write:
        path = S.supplement_path(split)
        S.L.ensure_dir(path.parent)
        S.L.atomic_save(dict(sup), path)
        file_sha = S.L.C.sha256_file(path)
    return {"supplement": sup, "content_sha256": content, "file_sha256": file_sha,
            "path": S.track_rel(S.supplement_path(split)), "crosscheck": xc}


def build_supplement(split: str, *, write: bool = True) -> dict:
    cache = S.D.load_cache(split)
    return supplement_from_cache(cache, split, write=write)


def load_supplement(split: str, *, verify_sha: str | None = None) -> dict:
    path = S.supplement_path(split)
    if not path.exists():
        build_supplement(split)
    sup = torch.load(path, map_location="cpu", weights_only=False)
    sha = S.supplement_content_hash(sup)
    if verify_sha is not None and sha != verify_sha:
        raise RuntimeError(f"{split}: supplement content sha {sha} != frozen {verify_sha}")
    return sup


# ---------------------------------------------------------------------------
# batching (frozen LCRC collate + second-hop fields)
# ---------------------------------------------------------------------------

def _ranges(ptr: np.ndarray, rows: np.ndarray):
    rows = np.asarray(rows, dtype=np.int64)
    starts = ptr[rows].astype(np.int64)
    ends = ptr[rows + 1].astype(np.int64)
    counts = ends - starts
    idx = np.concatenate([np.arange(s, e, dtype=np.int64) for s, e in zip(starts, ends)]) \
        if len(rows) else np.zeros(0, dtype=np.int64)
    batch = np.repeat(np.arange(len(rows), dtype=np.int64), counts) if len(rows) else np.zeros(0, np.int64)
    local = np.concatenate([np.arange(c, dtype=np.int64) for c in counts]) \
        if len(rows) else np.zeros(0, dtype=np.int64)
    return idx, batch, local, counts


def collate(cache: Mapping[str, Any], sup: Mapping[str, Any],
            rows: Sequence[int]) -> dict[str, torch.Tensor]:
    rows = np.asarray(rows, dtype=np.int64)
    B = len(rows)
    atom_ptr = np.asarray(cache["atom_ptr"])
    bond_ptr = np.asarray(cache["bond_ptr"])
    atom_idx, atom_batch, atom_local, atom_counts = _ranges(atom_ptr, rows)
    bond_idx, bond_batch, _, bond_counts = _ranges(bond_ptr, rows)
    atom_off = np.concatenate([[0], np.cumsum(atom_counts)[:-1]]).astype(np.int64) if B else np.zeros(0, np.int64)

    bi = np.asarray(cache["bond_index"])[bond_idx].astype(np.int64)
    bi = bi + atom_off[bond_batch].reshape(-1, 1)

    nbr_ptr = np.asarray(cache["nbr_ptr"])
    nbr_counts = nbr_ptr[atom_idx + 1] - nbr_ptr[atom_idx]
    atom_local_idx = np.arange(len(atom_idx), dtype=np.int64)
    nbr_idx = np.concatenate([np.arange(nbr_ptr[a], nbr_ptr[a] + c, dtype=np.int64)
                              for a, c in zip(atom_idx, nbr_counts)]) if len(atom_idx) else np.zeros(0, np.int64)
    nbr_atom = np.repeat(atom_local_idx, nbr_counts) if len(atom_idx) else np.zeros(0, np.int64)

    path_ptr = np.asarray(cache["path_ptr"])
    path_counts = path_ptr[atom_idx + 1] - path_ptr[atom_idx]
    path_idx = np.concatenate([np.arange(path_ptr[a], path_ptr[a] + c, dtype=np.int64)
                               for a, c in zip(atom_idx, path_counts)]) if len(atom_idx) else np.zeros(0, np.int64)
    path_graph = np.repeat(atom_batch, path_counts) if len(atom_idx) else np.zeros(0, np.int64)
    path_atom = np.repeat(atom_local_idx, path_counts) if len(atom_idx) else np.zeros(0, np.int64)
    path_end = np.asarray(cache["path_end"])[path_idx].astype(np.int64) \
        + atom_off[path_graph] if len(path_idx) else np.zeros(0, np.int64)

    # second hop: middle atom, second bond -> position inside this batch's bond array
    sh_mid = np.asarray(sup["path_mid"], dtype=np.int64)[path_idx] + atom_off[path_graph] \
        if len(path_idx) else np.zeros(0, np.int64)
    sh_bond2 = np.asarray(sup["path_bond2"], dtype=np.int64)[path_idx] \
        if len(path_idx) else np.zeros(0, np.int64)
    if len(path_idx):
        order = np.argsort(bond_idx, kind="stable")
        sorted_bonds = bond_idx[order]
        pos = np.searchsorted(sorted_bonds, sh_bond2)
        if not np.array_equal(sorted_bonds[pos], sh_bond2):
            raise RuntimeError("second-hop bond id not present in the batch bond array")
        sh_e2_pos = order[pos]
    else:
        sh_e2_pos = np.zeros(0, np.int64)

    pair_ptr = np.asarray(cache["pair_ptr"])
    pair_counts = pair_ptr[rows + 1] - pair_ptr[rows]
    pair_idx = np.concatenate([np.arange(pair_ptr[r], pair_ptr[r] + c, dtype=np.int64)
                               for r, c in zip(rows, pair_counts)]) if B else np.zeros(0, np.int64)
    pair_graph = np.repeat(np.arange(B, dtype=np.int64), pair_counts) if B else np.zeros(0, np.int64)
    pi = np.asarray(cache["pair_i"])[pair_idx].astype(np.int64) + atom_off[pair_graph] \
        if len(pair_idx) else np.zeros(0, np.int64)
    pj = np.asarray(cache["pair_j"])[pair_idx].astype(np.int64) + atom_off[pair_graph] \
        if len(pair_idx) else np.zeros(0, np.int64)

    return {
        "n_graphs": B,
        "atom_feat": torch.from_numpy(np.asarray(cache["atom_feat"])[atom_idx].astype(np.float32)),
        "atom_type": torch.from_numpy(np.asarray(cache["atom_type"])[atom_idx].astype(np.int64)),
        "atom_scalar": torch.from_numpy(np.asarray(cache["atom_scalar"])[atom_idx].astype(np.float32)),
        "atom_batch": torch.from_numpy(atom_batch),
        "atom_off": torch.from_numpy(atom_off),
        "bond_feat": torch.from_numpy(np.asarray(cache["bond_feat"])[bond_idx].astype(np.float32)),
        "bond_type": torch.from_numpy(np.asarray(cache["bond_type"])[bond_idx].astype(np.int64)),
        "bond_index": torch.from_numpy(bi.astype(np.int64)),
        "bond_batch": torch.from_numpy(bond_batch),
        "nbr_atom": torch.from_numpy(nbr_atom),
        "nbr_code": torch.from_numpy(np.asarray(cache["nbr_code"])[nbr_idx].astype(np.int64)),
        "path_atom": torch.from_numpy(path_atom),
        "path_count": torch.from_numpy(path_counts.astype(np.int64)),
        "path_cat": torch.from_numpy(np.asarray(cache["path_cat"])[path_idx].astype(np.int64)),
        "path_end63": torch.from_numpy(np.asarray(cache["path_end63"])[path_idx].astype(np.int64)),
        "path_end": torch.from_numpy(path_end),
        "path_closure": torch.from_numpy(np.asarray(cache["path_closure"])[path_idx].astype(np.int64)),
        "path_mid": torch.from_numpy(sh_mid),
        "path_e2_pos": torch.from_numpy(sh_e2_pos),
        "pair_i": torch.from_numpy(pi),
        "pair_j": torch.from_numpy(pj),
        "pair_cat": torch.from_numpy(np.asarray(cache["pair_cat"])[pair_idx].astype(np.int64)),
        "pair_graph": torch.from_numpy(pair_graph),
        "global": torch.from_numpy(np.asarray(cache["global"])[rows].astype(np.float32)),
        "atom_counts": torch.from_numpy(atom_counts.astype(np.int64)),
        "bond_counts": torch.from_numpy(bond_counts.astype(np.int64)),
        "pair_counts": torch.from_numpy(pair_counts.astype(np.int64)),
    }


__all__ = ["adjacency", "molecule_paths", "enumerate_split", "crosscheck",
           "bond_endpoints", "supplement_from_cache", "build_supplement",
           "load_supplement", "collate"]
