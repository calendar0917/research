"""Rooted radius-1 / radius-2 content keys for the composition-transfer round.

The keys are the *same objects* as the old coverage audit
(``zinc_generalization_audit_v1/code/audit_coverage.py::radius_signatures``):

    k1(v) = (atom_type(v), sorted((bond_type, atom_type(u)) for u in N(v)))
    k2(v) = (atom_type(v), sorted((bond_type, k1(u)) for u in N(v)))

``k1`` is produced by the same recursive builder at depth 1 (never by string
truncation), and ``k2(v)`` determines ``k1(v)`` because each neighbour entry of
``k2`` carries that neighbour's ``k1``, which contains the atom type and the bond
back to ``v``.  That is asserted at build time, not assumed.

Keys are interned jointly over official-train and official-valid *structures* so
a valid key can be looked up in a train vocabulary.  No label, model prediction
or split membership enters a key.

Outputs (task ``cache/``, git-ignored):

    keys.npz        flat per-atom key ids (train atoms then valid atoms)
    keys_meta.json  vocabulary sizes, coarsening check, old-group reproduction
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_TASK = _HERE.parent
_SSUF = _TASK.parent
_V1 = _SSUF / "zinc_arch_discovery_v1" / "code"
for _p in (str(_HERE), str(_V1)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402

CACHE = _TASK / "cache"


def _molecule_graphs(cache: dict):
    ap = np.asarray(cache["atom_ptr"], dtype=np.int64)
    bp = np.asarray(cache["bond_ptr"], dtype=np.int64)
    at = np.asarray(cache["atom_type"], dtype=np.int64)
    bi = np.asarray(cache["bond_index"], dtype=np.int64)
    bt = np.asarray(cache["bond_type"], dtype=np.int64)
    for i in range(int(cache["n_molecules"])):
        a0, a1 = int(ap[i]), int(ap[i + 1])
        b0, b1 = int(bp[i]), int(bp[i + 1])
        cats = at[a0:a1]
        adj = [[] for _ in range(len(cats))]
        for (u, v), t in zip(bi[b0:b1], bt[b0:b1]):
            u, v, t = int(u), int(v), int(t)
            adj[u].append((v, t))
            adj[v].append((u, t))
        for a in adj:
            a.sort()
        yield len(cats), cats, adj


def build_joint(cache_by_split: dict[str, dict]) -> tuple[dict, dict]:
    k1_ids: dict[tuple, int] = {}
    k2_ids: dict[tuple, int] = {}
    k2_to_k1: dict[int, int] = {}
    atom_k1: list[int] = []
    atom_k2: list[int] = []
    mol_k1: list[int] = []
    mol_k1_off: list[int] = [0]
    mol_k2: list[int] = []
    mol_k2_off: list[int] = [0]
    sizes: list[int] = []
    split_slices: dict[str, dict] = {}
    for name, cache in cache_by_split.items():
        a_start = len(atom_k1)
        m_start = len(sizes)
        for n, cats, adj in _molecule_graphs(cache):
            k1_local = []
            for v in range(n):
                key = (int(cats[v]), tuple(sorted((t, int(cats[u])) for u, t in adj[v])))
                kid = k1_ids.get(key)
                if kid is None:
                    kid = len(k1_ids)
                    k1_ids[key] = kid
                k1_local.append(kid)
            k2_local = []
            for v in range(n):
                key = (int(cats[v]), tuple(sorted((t, k1_local[u]) for u, t in adj[v])))
                kid = k2_ids.get(key)
                if kid is None:
                    kid = len(k2_ids)
                    k2_ids[key] = kid
                prev = k2_to_k1.get(kid)
                if prev is None:
                    k2_to_k1[kid] = k1_local[v]
                elif prev != k1_local[v]:
                    raise RuntimeError(
                        f"k1 is not a function of k2: key {kid} -> k1 {prev} and {k1_local[v]}"
                    )
                k2_local.append(kid)
            atom_k1.extend(k1_local)
            atom_k2.extend(k2_local)
            mol_k1.extend(np.unique(np.asarray(k1_local, dtype=np.int64)).tolist())
            mol_k1_off.append(len(mol_k1))
            mol_k2.extend(np.unique(np.asarray(k2_local, dtype=np.int64)).tolist())
            mol_k2_off.append(len(mol_k2))
            sizes.append(n)
        split_slices[name] = {
            "atom_start": int(a_start),
            "atom_end": int(len(atom_k1)),
            "mol_start": int(m_start),
            "mol_end": int(len(sizes)),
            "n_molecules": int(cache["n_molecules"]),
        }
    out = {
        "atom_key1": np.asarray(atom_k1, dtype=np.int32),
        "atom_key2": np.asarray(atom_k2, dtype=np.int32),
        "mol_k1": np.asarray(mol_k1, dtype=np.int32),
        "mol_k1_off": np.asarray(mol_k1_off, dtype=np.int64),
        "mol_k2": np.asarray(mol_k2, dtype=np.int32),
        "mol_k2_off": np.asarray(mol_k2_off, dtype=np.int64),
        "n_atoms_per_mol": np.asarray(sizes, dtype=np.int64),
        "n_key1": np.asarray(len(k1_ids), dtype=np.int64),
        "n_key2": np.asarray(len(k2_ids), dtype=np.int64),
    }
    return out, split_slices


def build_all(verbose: bool = True) -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    train = D.load_cache("train")
    valid = D.load_cache("valid")
    if verbose:
        C.log("[keys] building radius-1/2 keys over train + valid structures")
    out, slices = build_joint({"train": train, "valid": valid})
    path = CACHE / "keys.npz"
    np.savez_compressed(path, **out)
    meta = {
        "n_key1": int(out["n_key1"]),
        "n_key2": int(out["n_key2"]),
        "slices": slices,
        "n_atom_occurrences_train": int(slices["train"]["atom_end"] - slices["train"]["atom_start"]),
        "n_atom_occurrences_valid": int(slices["valid"]["atom_end"] - slices["valid"]["atom_start"]),
        "k1_is_function_of_k2": True,
        "keys_npz": str(path),
        "keys_npz_sha256": C.sha256_file(path),
        "builder": "audit_coverage.radius_signatures recursion depth 1/2, joint interning",
        "source_cache_meta": {"train": train["meta"], "valid": valid["meta"]},
    }
    v = slices["valid"]
    off = np.asarray(out["mol_k2_off"])
    tr_lo = int(off[slices["train"]["mol_start"]])
    tr_hi = int(off[slices["train"]["mol_end"]])
    tr_k2 = np.unique(np.asarray(out["mol_k2"])[tr_lo:tr_hi])
    present = np.zeros(int(out["n_key2"]), dtype=bool)
    present[tr_k2] = True
    ptr = np.asarray(valid["atom_ptr"], dtype=np.int64)
    keys2 = np.asarray(out["atom_key2"])[v["atom_start"]:v["atom_end"]]
    frac = np.zeros(int(valid["n_molecules"]), dtype=np.float64)
    for i in range(int(valid["n_molecules"])):
        s, e = int(ptr[i]), int(ptr[i + 1])
        frac[i] = float((~present[keys2[s:e]]).mean())
    ref = _SSUF / "zinc_restricted_adaptation_v1" / "analysis" / "reference_preds_valid.csv"
    if ref.exists():
        with open(ref, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        ref_frac = np.asarray([float(r["r2_unseen_atom_frac"]) for r in rows], dtype=np.float64)
        meta["old_group_reproduction"] = {
            "reference": str(ref.relative_to(_SSUF)),
            "max_abs_diff_r2_unseen_atom_frac": float(np.max(np.abs(frac - ref_frac))),
            "n_valid": int(len(ref_frac)),
            "note": "unseen = k2 absent from the train molecule key set, as in the old audit",
        }
    C.write_json(CACHE / "keys_meta.json", meta)
    if verbose:
        C.log(f"[keys] k1={meta['n_key1']} k2={meta['n_key2']} -> {path}")
    return meta


def molecule_stats(cache: dict) -> dict:
    """Size and ring rank per molecule from the raw cached graph (no labels)."""
    ap = np.asarray(cache["atom_ptr"], dtype=np.int64)
    bp = np.asarray(cache["bond_ptr"], dtype=np.int64)
    n_atoms = np.diff(ap).astype(np.int64)
    n_bonds = np.diff(bp).astype(np.int64)
    return {"n_atoms": n_atoms, "n_bonds": n_bonds, "ring_rank": n_bonds - n_atoms + 1}


def load() -> dict:
    return dict(np.load(CACHE / "keys.npz"))


def main() -> int:
    meta = build_all()
    C.log(meta.get("old_group_reproduction", {}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
