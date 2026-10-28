"""Train-only radius-2 structural support (the existing key definition).

``k1(v) = (type(v), sorted (bond_type, type(u)) for u in N(v))``
``k2(v) = (type(v), sorted (bond_type, k1(u)) for u in N(v))``

i.e. exactly ``zinc_composition_transfer_v1/code/zct_keys.py`` (itself
``zinc_generalization_audit_v1::radius_signatures`` recursion depths 1/2).  Keys
are interned jointly over official-train and official-valid **structures**
(no label, no model, no split leak); support of a key is the number of **train
molecules** containing it.  The vocabulary sizes are cross-checked against the
committed ``keys_meta.json`` of that round.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

import pb_common as P
import pb_paths as PP


def _graphs(cache: Mapping[str, Any]):
    atom_ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)
    bond_ptr = np.asarray(cache["bond_ptr"], dtype=np.int64)
    at = np.asarray(cache["atom_type"], dtype=np.int64)
    bi = np.asarray(cache["bond_index"], dtype=np.int64)
    bt = np.asarray(cache["bond_type"], dtype=np.int64)
    for m in range(int(cache["n_molecules"])):
        a0, a1 = int(atom_ptr[m]), int(atom_ptr[m + 1])
        b0, b1 = int(bond_ptr[m]), int(bond_ptr[m + 1])
        adj = PP.adjacency(a1 - a0, bi[b0:b1], bt[b0:b1])
        yield int(a1 - a0), at[a0:a1], adj


def build_keys(caches: Mapping[str, Mapping[str, Any]]) -> dict:
    k1_ids: dict[tuple, int] = {}
    k2_ids: dict[tuple, int] = {}
    atom_k1: list[int] = []
    atom_k2: list[int] = []
    mol_k2: list[list[int]] = []
    slices: dict[str, dict] = {}
    for name in ("train", "valid"):
        cache = caches[name]
        a_start = len(atom_k1)
        m_start = len(mol_k2)
        for n, types, adj in _graphs(cache):
            k1_local = []
            for v in range(n):
                key = (int(types[v]), tuple(sorted((t, int(types[u])) for u, t, _e in adj[v])))
                kid = k1_ids.get(key)
                if kid is None:
                    kid = len(k1_ids)
                    k1_ids[key] = kid
                k1_local.append(kid)
            k2_local = []
            for v in range(n):
                key = (int(types[v]), tuple(sorted((t, k1_local[u]) for u, t, _e in adj[v])))
                kid = k2_ids.get(key)
                if kid is None:
                    kid = len(k2_ids)
                    k2_ids[key] = kid
                k2_local.append(kid)
            atom_k1.extend(k1_local)
            atom_k2.extend(k2_local)
            mol_k2.append(sorted(set(k2_local)))
        slices[name] = {"atom_start": int(a_start), "atom_end": int(len(atom_k1)),
                        "mol_start": int(m_start), "mol_end": int(len(mol_k2)),
                        "n_molecules": int(cache["n_molecules"])}
    return {"atom_key1": np.asarray(atom_k1, dtype=np.int32),
            "atom_key2": np.asarray(atom_k2, dtype=np.int32),
            "mol_k2": mol_k2, "n_key1": len(k1_ids), "n_key2": len(k2_ids),
            "slices": slices}


def support_tables(keys: Mapping[str, Any]) -> dict:
    """``sup2[key]`` = number of train molecules containing the radius-2 key."""
    n2 = int(keys["n_key2"])
    sup2 = np.zeros(n2, dtype=np.int32)
    tr = keys["slices"]["train"]
    for m in range(int(tr["mol_start"]), int(tr["mol_end"])):
        sup2[np.asarray(keys["mol_k2"][m], dtype=np.int64)] += 1
    return {"sup2": sup2}


def valid_atom_support(keys: Mapping[str, Any], sup2: np.ndarray) -> dict:
    va = keys["slices"]["valid"]
    a0, a1 = int(va["atom_start"]), int(va["atom_end"])
    atom_key2 = np.asarray(keys["atom_key2"][a0:a1], dtype=np.int64)
    atom_sup = sup2[atom_key2].astype(np.int64)
    unseen = atom_sup == 0
    low = (atom_sup >= 1) & (atom_sup <= P.SUP_LOW_MAX)
    seen = atom_sup > P.SUP_LOW_MAX
    group = np.where(unseen, 0, np.where(low, 1, 2)).astype(np.int64)
    # molecule level
    cache_valid = keys["slices"]["valid"]
    mol_k2 = keys["mol_k2"]
    mol_any_unseen = []
    mol_unseen_frac = []
    for m in range(int(cache_valid["mol_start"]), int(cache_valid["mol_end"])):
        ks = np.asarray(mol_k2[m], dtype=np.int64)
        s = sup2[ks]
        mol_any_unseen.append(bool((s == 0).any()))
        mol_unseen_frac.append(float((s == 0).mean()) if len(s) else 0.0)
    return {
        "atom_key2": atom_key2,
        "atom_support": atom_sup,
        "atom_group": group,
        "counts": {"unseen": int(unseen.sum()), "low": int(low.sum()), "seen": int(seen.sum()),
                   "n_atoms": int(len(atom_sup))},
        "fractions": {"unseen": float(unseen.mean()), "low": float(low.mean()),
                      "seen": float(seen.mean())},
        "molecule_any_unseen": np.asarray(mol_any_unseen, dtype=bool),
        "molecule_unseen_fraction": np.asarray(mol_unseen_frac, dtype=np.float32),
        "support_quantiles": {q: float(np.percentile(atom_sup, q))
                             for q in (1, 5, 10, 25, 50, 75, 90, 95, 99)},
    }


GROUP_NAMES = ("unseen", "low_support", "seen")


__all__ = ["build_keys", "support_tables", "valid_atom_support", "GROUP_NAMES"]
