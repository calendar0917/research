"""Stage 1 — structural support of the *constituents* RBO intends to share.

Question (CPU, minutes): for the radius-2-unseen atoms of ``Q_target``, do the
smaller structural pieces -- the ones whose parameters RBO would share -- occur
in ``D_minus`` (the 451-molecule supervised fit set with zero target-key
support)?

Pieces, from raw structure only, no labels:

  T  centre-bond-neighbour triple          (t_v, b(v,u), t_u)
  P  centre-bond-neighbour-bond-atom path  (t_v, b(v,u), t_u, b(u,w), t_w)
  B  neighbour branch multiset after deleting the centre bond
     sig(u|v) = sorted[(b(u,w), t_w) for w in N(u) minus {v}]
  B' the same branch with its incoming bond: (b(v,u), sig(u|v))
  C  two-branch combination at one centre: {B'_i, B'_j}, i<j

Also reported, for context only:
  * the same counts against the *full* official-train structure pool
    (upper bound: what the 451-row donor removal costs),
  * how many of the Q_target atoms are radius-2-unseen in D_minus at all.

No test split is read.  Q_target / D_minus row ids come from the frozen
``zinc_composition_transfer_v1/cache/support_sets.npz``.
"""

from __future__ import annotations

import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_TASK = _HERE.parent
_SSUF = _TASK.parent
_CT = _SSUF / "zinc_composition_transfer_v1"
_CACHE = _SSUF / "zinc_arch_discovery_v1" / "cache"
OUT = _TASK / "analysis"


def load_cache(split: str = "train") -> dict:
    return torch.load(_CACHE / f"ad_{split}.pt", map_location="cpu", weights_only=False)


def molecules(cache: dict, rows) -> list[dict]:
    ap = np.asarray(cache["atom_ptr"], dtype=np.int64)
    bp = np.asarray(cache["bond_ptr"], dtype=np.int64)
    at = np.asarray(cache["atom_type"], dtype=np.int64)
    bi = np.asarray(cache["bond_index"], dtype=np.int64)
    bt = np.asarray(cache["bond_type"], dtype=np.int64)
    out = []
    for i in rows:
        a0, a1 = int(ap[i]), int(ap[i + 1])
        b0, b1 = int(bp[i]), int(bp[i + 1])
        types = at[a0:a1].tolist()
        adj = [[] for _ in range(len(types))]
        for (u, v), t in zip(bi[b0:b1].tolist(), bt[b0:b1].tolist()):
            # bond_index is per-molecule local
            adj[u].append((v, int(t)))
            adj[v].append((u, int(t)))
        for a in adj:
            a.sort()
        out.append({"types": types, "adj": adj})
    return out


def branch_sig(mol: dict, u: int, v: int) -> tuple:
    """sorted[(bond type, atom type)] of u's incidences other than the back bond."""
    return tuple(sorted((t, mol["types"][w]) for w, t in mol["adj"][u] if w != v))


def iter_constituents(mol):
    """Yield T, P, B, B' signatures and per-centre C combinations for one molecule."""
    types, adj = mol["types"], mol["adj"]
    for v in range(len(types)):
        sigs = []
        for u, bvu in adj[v]:
            yield "T", (types[v], bvu, types[u])
            for w, buw in adj[u]:
                if w != v:
                    yield "P", (types[v], bvu, types[u], buw, types[w])
            b = branch_sig(mol, u, v)
            yield "B", b
            yield "Bp", (bvu, b)
            sigs.append((bvu, b))
        sigs.sort()
        for (b1, s1), (b2, s2) in itertools.combinations(sigs, 2):
            yield "C", ((b1, s1), (b2, s2))


def build_support(mols) -> dict[str, set]:
    sup: dict[str, set] = {"T": set(), "P": set(), "B": set(), "Bp": set(), "C": set()}
    for m in mols:
        for k, s in iter_constituents(m):
            sup[k].add(s)
    return sup


def atom_constituents(mol, v):
    """All constituents attached to atom v, grouped by kind."""
    types, adj = mol["types"], mol["adj"]
    T, P, Bp, C = [], [], [], []
    sigs = []
    for u, bvu in adj[v]:
        T.append((types[v], bvu, types[u]))
        for w, buw in adj[u]:
            if w != v:
                P.append((types[v], bvu, types[u], buw, types[w]))
        b = branch_sig(mol, u, v)
        Bp.append((bvu, b))
        sigs.append((bvu, b))
    sigs.sort()
    for s1, s2 in itertools.combinations(sigs, 2):
        C.append((s1, s2))
    return {"T": T, "P": P, "Bp": Bp, "C": C}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    supz = np.load(_CT / "cache" / "support_sets.npz")
    keyz = np.load(_CT / "cache" / "keys.npz")
    q_target = np.sort(supz["q_target"])
    d_minus = np.sort(supz["d_minus"])
    print(f"Q_target={len(q_target)} D_minus={len(d_minus)}")

    cache = load_cache("train")
    ap = np.asarray(cache["atom_ptr"], dtype=np.int64)
    k2 = np.asarray(keyz["atom_key2"], dtype=np.int64)  # train slice first
    k1 = np.asarray(keyz["atom_key1"], dtype=np.int64)

    def k2_set(rows):
        return {int(x) for r in rows for x in k2[ap[r]:ap[r + 1]]}

    def k1_set(rows):
        return {int(x) for r in rows for x in k1[ap[r]:ap[r + 1]]}

    k2_minus = k2_set(d_minus)
    k1_minus = k1_set(d_minus)

    q_mols = molecules(cache, q_target)
    d_mols = molecules(cache, d_minus)

    sup_minus = build_support(d_mols)
    sup_full = build_support(list(molecules(cache, range(int(cache["n_molecules"])))))

    # ---- which Q_target atoms are radius-2-unseen in D_minus --------------
    per_atom = []  # (mol_local_index, atom_index, is_unseen)
    for mi, r in enumerate(q_target):
        a0, a1 = int(ap[r]), int(ap[r + 1])
        for v in range(a1 - a0):
            per_atom.append((mi, v, int(k2[a0 + v]) not in k2_minus))
    per_atom = np.asarray(per_atom, dtype=np.int64)
    unseen = per_atom[per_atom[:, 2] == 1]
    print(f"Q_target atoms={len(per_atom)} r2-unseen w.r.t. D_minus={len(unseen)} "
          f"({len(unseen) / len(per_atom):.3f})")

    # radius-1 root signature support (the coarser object that is *shared* in
    # the existing P3 reader): how much of the unseen-ness is only at r2?
    n_k1_supported = sum(int(k1[int(ap[q_target[mi]]) + v]) in k1_minus
                         for mi, v, _ in unseen)
    n_k1_supported_full = sum(
        int(k1[int(ap[q_target[mi]]) + v]) in k1_set_cache
        for mi, v, _ in unseen) if False else None

    # ---- coverage of each constituent kind on the unseen atoms ------------
    rows = []
    for kind in ("T", "P", "Bp", "C"):
        n_atoms_all = 0
        n_atoms_full = 0
        n_items = 0
        n_items_full = 0
        n_atoms_with_any = 0
        n_atoms_with_any_full = 0
        for mi, v, _ in unseen:
            mol = q_mols[mi]
            items = atom_constituents(mol, v)[kind]
            n_atoms_all += 1
            if items:
                n_items += len(items)
                n_items_full += len(items)
                if all(s in sup_minus[kind] for s in items):
                    n_atoms_full += 1
                if any(s in sup_minus[kind] for s in items):
                    n_atoms_with_any += 1
                if all(s in sup_full[kind] for s in items):
                    n_atoms_with_any_full += 1
        rows.append({
            "kind": kind,
            "atoms": n_atoms_all,
            "items": n_items,
            "atoms_all_items_supported_D_minus": n_atoms_full,
            "frac_all_items_supported_D_minus": (n_atoms_full / max(n_atoms_all, 1)),
            "atoms_any_item_supported_D_minus": n_atoms_with_any,
            "frac_any_item_supported_D_minus": (n_atoms_with_any / max(n_atoms_all, 1)),
            "atoms_all_items_supported_full_train": n_atoms_with_any_full,
            "frac_all_items_supported_full_train": (n_atoms_with_any_full / max(n_atoms_all, 1)),
            "unique_sigs_D_minus": len(sup_minus[kind]),
            "unique_sigs_full_train": len(sup_full[kind]),
        })

    # ---- per-molecule: is every r2-unseen atom fully branch-supported? ----
    by_mol: dict[int, list[int]] = {}
    for mi, v, _ in unseen:
        by_mol.setdefault(int(mi), []).append(int(v))
    mol_complete = 0
    for mi, vs in by_mol.items():
        ok = True
        for v in vs:
            items = atom_constituents(q_mols[mi], v)["Bp"]
            if not all(s in sup_minus["Bp"] for s in items):
                ok = False
                break
        mol_complete += int(ok)

    # ---- instance-level support counts (how many D_minus occurrences) ----
    # count occurrences of every constituent kind in D_minus (not just presence)
    occ = {k: Counter() for k in ("T", "P", "Bp", "C")}
    for m in d_mols:
        for k, s in iter_constituents(m):
            if k in occ:
                occ[k][s] += 1
    inst = []
    for kind in ("T", "P", "Bp", "C"):
        c = occ[kind]
        vals = np.asarray(sorted(c.values()), dtype=np.int64)
        inst.append({
            "kind": kind,
            "support_occurrences": int(vals.sum()) if vals.size else 0,
            "unique": int(vals.size),
            "median_support": float(np.median(vals)) if vals.size else 0.0,
            "frac_unique_with_ge2": float(np.mean(vals >= 2)) if vals.size else 0.0,
        })

    # ---- concrete unsupported examples ------------------------------------
    examples = []
    n_show = 0
    for mi, v, _ in unseen:
        mol = q_mols[mi]
        for bvu, b in atom_constituents(mol, v)["Bp"]:
            if (bvu, b) not in sup_minus["Bp"] and n_show < 8:
                examples.append({
                    "q_target_row": int(q_target[mi]), "atom": int(v),
                    "centre_type": int(mol["types"][v]), "bond": int(bvu),
                    "branch": [[int(t), int(a)] for t, a in b],
                    "branch_key": [int(t) for t, _ in b],
                })
                n_show += 1
            if n_show >= 8:
                break
        if n_show >= 8:
            break

    res = {
        "definitions": {
            "T": "(t_v, b(v,u), t_u)",
            "P": "(t_v, b(v,u), t_u, b(u,w), t_w) over w in N(u)\\{v}",
            "Bp": "(b(v,u), sorted[(b(u,w), t_w) for w in N(u)\\{v}])",
            "C": "unordered pair of Bp at the same centre",
        },
        "counts": {
            "q_target_molecules": int(len(q_target)),
            "d_minus_molecules": int(len(d_minus)),
            "q_target_atoms": int(len(per_atom)),
            "q_target_atoms_r2_unseen_in_D_minus": int(len(unseen)),
            "frac_r2_unseen": float(len(unseen) / len(per_atom)),
            "q_target_molecules_with_unseen_atoms": int(len(by_mol)),
            "q_target_molecules_all_unseen_atoms_branch_supported": int(mol_complete),
            "r2_unseen_atoms_with_k1_supported_in_D_minus": int(n_k1_supported),
            "frac_r2_unseen_atoms_k1_supported": float(n_k1_supported / max(len(unseen), 1)),
        },
        "per_kind": rows,
        "instance_support_D_minus": inst,
        "unsupported_branch_examples": examples,
    }
    (OUT / "t0_support.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["counts"], indent=1))
    for r in rows:
        print(f"{r['kind']:>3} atoms={r['atoms']:5d} items={r['items']:6d} "
              f"all-supported(D-)={r['frac_all_items_supported_D_minus']:.3f} "
              f"any={r['frac_any_item_supported_D_minus']:.3f} "
              f"all-supported(full)={r['frac_all_items_supported_full_train']:.3f} "
              f"uniq D-={r['unique_sigs_D_minus']} full={r['unique_sigs_full_train']}")
    for r in inst:
        print("inst", r)
    print("examples:")
    for e in examples[:5]:
        print(" ", e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
