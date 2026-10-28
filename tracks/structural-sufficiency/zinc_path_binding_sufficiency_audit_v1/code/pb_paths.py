"""Raw-graph enumeration of ordered, non-backtracking two-hop paths with bond ids.

This is the *correctness* layer of the audit (PROTOCOL.md §4):

* every path carries root ``v``, middle ``w``, endpoint ``u``, first bond id
  ``e1=(v,w)`` and second bond id ``e2=(w,u)``;
* the enumeration order is identical to ``lc_data.build_from_arrays`` (adjacency
  sorted by ``(neighbour, bond type, bond id)``), so the multiset can be compared
  element-wise against the frozen LCRC cache, not only by histogram;
* the cache stores no bond ids, so ``e1``/``e2`` are provenance that exists only
  here.  Closure is decided on the raw adjacency (``u`` adjacent to ``v``),
  independently of the cache's stored flag.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

import pb_common as P
import lc_data as LD


def adjacency(n_atoms: int, bond_index: np.ndarray, bond_type: np.ndarray):
    """``adj[v] = sorted [(u, bond_type, bond_id)]`` -- same order as LCRC-v1."""
    adj: list[list[tuple[int, int, int]]] = [[] for _ in range(int(n_atoms))]
    for e, ((u, v), bt) in enumerate(zip(np.asarray(bond_index, dtype=np.int64),
                                         np.asarray(bond_type, dtype=np.int64))):
        adj[int(u)].append((int(v), int(bt), int(e)))
        adj[int(v)].append((int(u), int(bt), int(e)))
    for a in adj:
        a.sort()
    return adj


def molecule_paths(bond_index: np.ndarray, bond_type: np.ndarray, n_atoms: int) -> dict:
    """All ordered non-backtracking two-hop paths of one molecule (local ids)."""
    adj = adjacency(n_atoms, bond_index, bond_type)
    root, mid, end, e1, e2, closure = [], [], [], [], [], []
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
        ptr.append(len(root))
    return {
        "root": np.asarray(root, dtype=np.int32),
        "mid": np.asarray(mid, dtype=np.int32),
        "end": np.asarray(end, dtype=np.int32),
        "e1": np.asarray(e1, dtype=np.int32),
        "e2": np.asarray(e2, dtype=np.int32),
        "closure": np.asarray(closure, dtype=np.uint8),
        "ptr": np.asarray(ptr, dtype=np.int64),
        "n_atoms": int(n_atoms),
    }


def _molecule_offsets(cache: Mapping[str, Any]):
    ap = np.asarray(cache["atom_ptr"], dtype=np.int64)
    bpp = np.asarray(cache["bond_ptr"], dtype=np.int64)
    return ap, bpp


def split_paths(cache: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """All paths of a split, with global atom ids, global bond ids and closure."""
    ap, bp = _molecule_offsets(cache)
    bi_all = np.asarray(cache["bond_index"], dtype=np.int64)
    bt_all = np.asarray(cache["bond_type"], dtype=np.int64)
    n_mols = int(cache["n_molecules"])
    parts = {k: [] for k in ("root", "mid", "end", "e1", "e2", "closure")}
    per_atom = np.zeros(int(ap[-1]), dtype=np.int32)
    mol_ptr = [0]
    total_paths = 0
    for m in range(n_mols):
        a0, a1 = int(ap[m]), int(ap[m + 1])
        b0, b1 = int(bp[m]), int(bp[m + 1])
        p = molecule_paths(bi_all[b0:b1], bt_all[b0:b1], a1 - a0)
        parts["root"].append(p["root"].astype(np.int64) + a0)
        parts["mid"].append(p["mid"].astype(np.int64) + a0)
        parts["end"].append(p["end"].astype(np.int64) + a0)
        parts["e1"].append(p["e1"].astype(np.int64) + b0)
        parts["e2"].append(p["e2"].astype(np.int64) + b0)
        parts["closure"].append(p["closure"])
        per_atom[a0:a1] = np.diff(p["ptr"])
        total_paths += len(p["root"])
        mol_ptr.append(total_paths)
    out = {k: (np.concatenate(v) if v else np.zeros(0, dtype=np.int64))
           for k, v in parts.items()}
    out["root"] = out["root"].astype(np.int64)
    out["mid"] = out["mid"].astype(np.int64)
    out["end"] = out["end"].astype(np.int64)
    out["e1"] = out["e1"].astype(np.int64)
    out["e2"] = out["e2"].astype(np.int64)
    out["per_atom"] = per_atom
    out["mol_ptr"] = np.asarray(mol_ptr, dtype=np.int64)
    return out


def path_codes(cache: Mapping[str, Any], paths: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Recompute the frozen cache's S2 path codes from the raw enumeration."""
    at = np.asarray(cache["atom_type"], dtype=np.int64)
    bt = np.asarray(cache["bond_type"], dtype=np.int64)
    root, mid, end, e1, e2 = paths["root"], paths["mid"], paths["end"], paths["e1"], paths["e2"]
    cat = at[mid] * 9 + (bt[e1] - 1) * 3 + (bt[e2] - 1)
    end63 = at[end] * 3 + (bt[e2] - 1)
    return {"cat": cat.astype(np.uint8), "end63": end63.astype(np.uint8),
            "end": end.astype(np.int16), "closure": paths["closure"].astype(np.uint8)}


def nbr_codes(cache: Mapping[str, Any]) -> np.ndarray:
    """Recompute the S1 (neighbour type, bond type) codes in cache order."""
    at = np.asarray(cache["atom_type"], dtype=np.int64)
    bt = np.asarray(cache["bond_type"], dtype=np.int64)
    bi = np.asarray(cache["bond_index"], dtype=np.int64)
    bp = np.asarray(cache["bond_ptr"], dtype=np.int64)
    ap = np.asarray(cache["atom_ptr"], dtype=np.int64)
    out = []
    for m in range(int(cache["n_molecules"])):
        a0, a1 = int(ap[m]), int(ap[m + 1])
        b0, b1 = int(bp[m]), int(bp[m + 1])
        adj = adjacency(a1 - a0, bi[b0:b1], bt[b0:b1])
        for v in range(a1 - a0):
            out.append(np.asarray([at[a0 + u] * 3 + (t - 1) for u, t, _e in adj[v]], dtype=np.uint8))
    return np.concatenate(out) if out else np.zeros(0, dtype=np.uint8)


def bond_endpoints(cache: Mapping[str, Any]) -> np.ndarray:
    """Global bond -> (atom_a, atom_b) table."""
    ap, bp = _molecule_offsets(cache)
    bi = np.asarray(cache["bond_index"], dtype=np.int64)
    parts = [bi[int(bp[m]):int(bp[m + 1])] + int(ap[m]) for m in range(int(cache["n_molecules"]))]
    return np.concatenate(parts) if parts else np.zeros((0, 2), dtype=np.int64)


def crosscheck_split(cache: Mapping[str, Any], paths: Mapping[str, Any] | None = None) -> dict:
    """Element-wise agreement between the raw enumeration and the frozen cache."""
    if paths is None:
        paths = split_paths(cache)
    codes = path_codes(cache, paths)
    nbr = nbr_codes(cache)
    ref_path_ptr = np.asarray(cache["path_ptr"], dtype=np.int64)
    ref_nbr_ptr = np.asarray(cache["nbr_ptr"], dtype=np.int64)

    def agree(a, b):
        if a.shape != b.shape:
            return {"equal": False, "shape_a": list(a.shape), "shape_b": list(b.shape)}
        mismatch = int(np.sum(a != b))
        return {"equal": mismatch == 0, "n": int(a.size), "mismatch": mismatch,
                "mismatch_fraction": float(mismatch / max(int(a.size), 1))}

    ends = bond_endpoints(cache)
    # cache `path_end` is molecule-local; convert the global endpoint back
    ap = np.asarray(cache["atom_ptr"], dtype=np.int64)
    mol_of_path = np.repeat(np.arange(len(ap) - 1), np.diff(paths["mol_ptr"]))
    local_end = (paths["end"] - ap[mol_of_path]).astype(np.int16)
    be1 = ends[paths["e1"]]
    ok1 = ((be1[:, 0] == paths["root"]) & (be1[:, 1] == paths["mid"])) | \
          ((be1[:, 1] == paths["root"]) & (be1[:, 0] == paths["mid"]))
    be2 = ends[paths["e2"]]
    ok2 = ((be2[:, 0] == paths["mid"]) & (be2[:, 1] == paths["end"])) | \
          ((be2[:, 1] == paths["mid"]) & (be2[:, 0] == paths["end"]))

    checks = {
        "path_cat": agree(codes["cat"], np.asarray(cache["path_cat"], dtype=np.uint8)),
        "path_end63": agree(codes["end63"], np.asarray(cache["path_end63"], dtype=np.uint8)),
        "path_end": agree(local_end, np.asarray(cache["path_end"], dtype=np.int16)),
        "path_closure": agree(codes["closure"], np.asarray(cache["path_closure"], dtype=np.uint8)),
        "nbr_code": agree(nbr, np.asarray(cache["nbr_code"], dtype=np.uint8)),
    }
    counts = {
        "per_atom_path_count_mismatch": int(np.sum(paths["per_atom"] != np.diff(ref_path_ptr))),
        "nbr_ptr_total_mismatch": int(ref_nbr_ptr[-1] - len(nbr)),
        "root_equals_end": int(np.sum(paths["root"] == paths["end"])),
        "e1_equals_e2": int(np.sum(paths["e1"] == paths["e2"])),
        "bond_e1_incident_violations": int(np.sum(~ok1)),
        "bond_e2_incident_violations": int(np.sum(~ok2)),
    }
    passed = (all(c["equal"] for c in checks.values())
              and counts["per_atom_path_count_mismatch"] == 0
              and counts["nbr_ptr_total_mismatch"] == 0
              and counts["root_equals_end"] == 0
              and counts["e1_equals_e2"] == 0
              and counts["bond_e1_incident_violations"] == 0
              and counts["bond_e2_incident_violations"] == 0)
    return {
        "n_paths": int(len(paths["root"])),
        "n_paths_from_cache": int(len(np.asarray(cache["path_cat"]))),
        "n_atoms": int(len(paths["per_atom"])),
        "codes": checks,
        "counts": counts,
        "passed": bool(passed),
    }


# ---------------------------------------------------------------------------
# synthetic graphs for boundary / invariance / counterexample tests
# ---------------------------------------------------------------------------

def synthetic_cache(graphs) -> dict:
    """Build an LCRC-style cache from raw ``(atom_type, [(u, v, bt)])`` graphs."""
    mols = []
    for atom_type, bonds in graphs:
        atom_type = np.asarray(atom_type, dtype=np.int64)
        n = int(len(atom_type))
        bi = np.asarray([[u, v] for u, v, _ in bonds], dtype=np.int64).reshape(-1, 2)
        bt = np.asarray([t for _, _, t in bonds], dtype=np.int64)
        atom_feat = np.zeros((n, P.L.ATOM_FEAT_DIM), dtype=np.float32)
        atom_feat[np.arange(n), atom_type] = 1.0
        bond_feat = np.zeros((len(bonds), P.L.BOND_FEAT_DIM), dtype=np.float32)
        bond_feat[np.arange(len(bonds)), np.clip(bt, 0, 3)] = 1.0
        mols.append(LD.build_from_arrays(atom_feat, atom_type, bi, bt, bond_feat,
                                         np.zeros(P.L.GLOBAL_DIM, dtype=np.float32)))
    cache = LD._concat_molecules(mols)
    cache["split"] = "synthetic"
    cache["n_molecules"] = len(mols)
    cache["meta"] = {"feature_version": P.L.FEATURE_VERSION, "split": "synthetic"}
    return cache


__all__ = ["adjacency", "molecule_paths", "split_paths", "path_codes", "nbr_codes",
           "bond_endpoints", "crosscheck_split", "synthetic_cache"]
