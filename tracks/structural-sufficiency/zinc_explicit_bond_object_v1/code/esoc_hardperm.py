"""HARD_PERM control for ESOC-v1.

Motivation
----------
The existing REAL vs NULL contrast changes **two** things at once:

1. the correspondence between a pair ``(i,j)`` and its bond type, and
2. the *input distribution*: REAL feeds a one-hot ``edge_ij`` per pair, NULL feeds
   the object-level 4-class frequency vector broadcast to every pair.

So a REAL gain cannot be attributed to correspondence alone.  HARD_PERM holds (2)
fixed — it feeds exactly the same one-hot inputs as REAL — and destroys only (1).

Design
------
For every bond-centred object ``P_b`` (one object per physical bond) we keep

* the node set, the atom types, the roles ``(dist(i,u), dist(i,v))`` — untouched,
* the skeleton: which pairs are bonds and which are non-bonds — untouched,
* the multiset of bond-type labels on the real bonds — preserved exactly,

and we **reassign** the labels to the real bonds by a deterministic rule keyed on
intrinsic, root-symmetric, renumbering-invariant edge descriptors.

Because the skeleton is fixed, the 4-class histogram, the number of edges
(``loc_nedges``), every node degree and every non-edge are *identical* to REAL, so
the NULL input vector (the 4-class frequency) is bitwise identical too.  The only
quantity that changes is *which* bond carries *which* label.

Consequences that must be reported, not hidden
----------------------------------------------
Reassigning labels while holding each node's degree fixed does **not** hold each
node's *bond-order sum* fixed, so a permuted object can be chemically impossible
(an atom that had one double bond may end up with two).  This is inherent to any
correspondence-breaking control: the control is a probe of whether correspondence
matters, not a chemically valid model input.  ``object_stats`` measures it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_SS = _HERE.parents[1]
for _p in (str(_HERE), str(_SS / "zinc_arch_discovery_v1" / "code")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import esoc_data as D  # noqa: E402

ANALYSIS = _HERE.parent / "analysis"
VARIANT = "hardperm"


# ---------------------------------------------------------------------------
# intrinsic descriptors (no atom ids, no labels)
# ---------------------------------------------------------------------------
def _node_key(t, ru, rv):
    """Root-symmetric node descriptor: type + the unordered pair of distances."""
    a, b = (ru, rv) if ru <= rv else (rv, ru)
    return (int(t), int(a), int(b))


def _edge_key(na, nb):
    """Root-symmetric, renumbering-invariant key for an undirected bond."""
    return tuple(sorted((na, nb)))


def molecule_objects(atom_type, atom_ptr, bond_ptr, bond_index, bond_type, i):
    """Rebuild one molecule's local objects in the same order as the cache."""
    a0, a1 = int(atom_ptr[i]), int(atom_ptr[i + 1])
    b0, b1 = int(bond_ptr[i]), int(bond_ptr[i + 1])
    types = np.asarray(atom_type[a0:a1], dtype=np.int64)
    n = int(a1 - a0)
    # bond_index is stored per molecule (already local atom indices).
    bi = np.asarray(bond_index[b0:b1], dtype=np.int64)
    bt = np.asarray(bond_type[b0:b1], dtype=np.int64)

    adj = [set() for _ in range(n)]
    bond_map: dict[tuple[int, int], int] = {}
    for (u, v), t in zip(bi.tolist(), bt.tolist()):
        adj[u].add(v)
        adj[v].add(u)
        bond_map[(u, v) if u < v else (v, u)] = int(t)

    dist = D._capped_distances([sorted(a) for a in adj], n, D.ROLE_CAP)

    objs = []
    for u, v in zip(bi[:, 0].tolist(), bi[:, 1].tolist()):
        nodes = sorted({u, v} | adj[u] | adj[v])
        m = len(nodes)
        lab = np.zeros((m, m), dtype=np.int64)
        for a in range(m):
            for b in range(a + 1, m):
                p, q = nodes[a], nodes[b]
                t = bond_map.get((p, q) if p < q else (q, p), 0)
                lab[a, b] = t
                lab[b, a] = t
        ru = np.array([min(int(dist[u, x]), D.ROLE_CAP) for x in nodes], dtype=np.int64)
        rv = np.array([min(int(dist[v, x]), D.ROLE_CAP) for x in nodes], dtype=np.int64)
        objs.append({
            "nodes": nodes,
            "type": np.asarray([types[x] for x in nodes], dtype=np.int64),
            "ru": ru,
            "rv": rv,
            "lab": lab,
        })
    return objs


# ---------------------------------------------------------------------------
# the permutation
# ---------------------------------------------------------------------------
def permute_labels(obj) -> tuple[np.ndarray, dict]:
    """Reassign bond-type labels among the real bonds of one object.

    Returns ``(new_lab, info)``.  ``info`` records whether anything changed and how
    many pair slots moved, so the caller can report the real change fraction
    instead of assuming the permutation is non-trivial.
    """
    lab = obj["lab"]
    m = lab.shape[0]
    new = lab.copy()

    real = [(a, b) for a in range(m) for b in range(a + 1, m) if lab[a, b] > 0]
    info = {"n_real_edges": len(real), "n_pair_slots": m * (m - 1) // 2}
    if len(real) < 2:
        info.update(changed=False, n_label_slots_changed=0, n_pair_slots_changed=0)
        return new, info

    nk = [_node_key(obj["type"][j], obj["ru"][j], obj["rv"][j]) for j in range(m)]

    # Order the real bonds by an intrinsic, root-symmetric key ONLY: (edge
    # descriptor, original label).  No positional tie-break is allowed -- position
    # inside the object depends on the atom numbering, which would break
    # renumbering invariance.  Ties are harmless: two edges with the same
    # descriptor *and* the same label are interchangeable in content, so any
    # order within a tie group yields the same (descriptor, label) multiset.
    order = sorted(range(len(real)),
                   key=lambda e: (_edge_key(nk[real[e][0]], nk[real[e][1]]),
                                  int(lab[real[e][0], real[e][1]])))
    labels = [int(lab[real[e][0], real[e][1]]) for e in order]

    # Rotate by half the cycle: deterministic, preserves the multiset exactly, and
    # avoids the identity whenever the labels are not all equal.
    shift = max(1, len(labels) // 2)
    rotated = labels[shift:] + labels[:shift]

    n_slot_changed = 0
    for pos, e in enumerate(order):
        a, b = real[e]
        v = int(rotated[pos])
        if v != int(lab[a, b]):
            n_slot_changed += 1
        new[a, b] = v
        new[b, a] = v

    diff = new != lab
    info.update(
        changed=bool(diff.any()),
        n_label_slots_changed=int(n_slot_changed),
        n_pair_slots_changed=int(np.triu(diff, 1).sum()),
    )
    return new, info


# ---------------------------------------------------------------------------
# whole-split transform
# ---------------------------------------------------------------------------
def build_variant(base: dict, rng_seed: int = 0) -> tuple[dict, dict]:
    """Return a copy of ``base`` with ``loc_et`` replaced by the HARD_PERM version."""
    atom_type = np.asarray(base["atom_type"])
    atom_ptr = np.asarray(base["atom_ptr"])
    bond_ptr = np.asarray(base["bond_ptr"])
    bond_index = np.asarray(base["bond_index"])
    bond_type = np.asarray(base["bond_type"])
    n_mol = int(base["n_molecules"])

    loc_et = np.array(base["loc_et"], copy=True)
    loc_nedges = np.asarray(base["loc_nedges"])

    per_obj = []
    n_changed = 0
    n_identity = 0
    n_edges_total = 0
    n_slots_changed = 0
    n_pairs_total = 0
    n_too_small = 0
    bond_order_moved = 0
    bond_order_total = 0

    for i in range(n_mol):
        objs = molecule_objects(atom_type, atom_ptr, bond_ptr, bond_index, bond_type, i)
        b0, b1 = int(bond_ptr[i]), int(bond_ptr[i + 1])
        k0 = int(base["bond_ptr"][i])
        if len(objs) != b1 - b0:
            raise RuntimeError(f"mol {i}: {len(objs)} objects != {b1 - b0} bonds")
        for j, obj in enumerate(objs):
            g = k0 + j
            new, info = permute_labels(obj)
            m = new.shape[0]
            # loc_et is stored at the full LMAX x LMAX slot width; the object's
            # real content is the leading m x m block (padding rows are masked).
            assert int((new > 0).sum()) // 2 == int(loc_nedges[g]), (
                f"obj {g}: edge count changed {int((new > 0).sum()) // 2} != {loc_nedges[g]}")
            if m > loc_et.shape[1]:
                raise RuntimeError(f"obj {g}: local size {m} exceeds slot width {loc_et.shape[1]}")
            loc_et[g, :m, :m] = new.astype(loc_et.dtype)

            n_edges_total += info["n_real_edges"]
            n_pairs_total += info["n_pair_slots"]
            n_slots_changed += info["n_label_slots_changed"]
            if info["changed"]:
                n_changed += 1
            else:
                n_identity += 1
                if info["n_real_edges"] < 2:
                    n_too_small += 1

            # bond-order (valence) plausibility, per atom inside the object
            old_sum = obj["lab"].sum(axis=1)
            new_sum = new.sum(axis=1)
            bond_order_moved += int((old_sum != new_sum).sum())
            bond_order_total += int(len(old_sum))
            per_obj.append(info)

    out = dict(base)
    out["loc_et"] = loc_et
    out["meta"] = dict(base.get("meta", {}))
    out["meta"].update({
        "variant": VARIANT,
        "variant_of": base.get("split"),
        "variant_rng_seed": int(rng_seed),
        "variant_reads_labels": False,
        "official_test_loaded": False,
        "hardperm": {
            "objects": int(n_mol and len(per_obj)),
            "objects_changed": int(n_changed),
            "objects_identity": int(n_identity),
            "objects_identity_fewer_than_2_edges": int(n_too_small),
            "real_bonds_total": int(n_edges_total),
            "pair_slots_total": int(n_pairs_total),
            "bond_label_slots_changed": int(n_slots_changed),
            "pair_slots_changed": int((loc_et != np.asarray(base["loc_et"])).sum()) // 2,
            "atom_slots_with_moved_bond_order_sum": int(bond_order_moved),
            "atom_slots_total": int(bond_order_total),
        },
    })
    stats = out["meta"]["hardperm"]
    stats["fraction_objects_changed"] = n_changed / max(len(per_obj), 1)
    stats["fraction_bond_labels_changed"] = n_slots_changed / max(n_edges_total, 1)
    stats["fraction_atom_slots_bond_order_moved"] = bond_order_moved / max(bond_order_total, 1)
    return out, stats


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------
def invariance_checks(base: dict, n_mol: int = 300, rng_seed: int = 0) -> dict:
    """Root-swap and renumbering invariance, measured on the *object content*.

    The local encoder aggregates with sums over nodes and pairs, so it is
    invariant to node order.  What must therefore hold is equality of the
    multiset of ``(role_i, role_j, label_ij)`` triples:

    * renumbering: relabel the molecule's atoms, rebuild the objects, apply
      HARD_PERM, and require the triple multiset to be unchanged;
    * root swap: rebuild with the bond direction flipped, apply HARD_PERM, swap the
      role digits, and require the triple multiset to be unchanged.
    """
    atom_type = np.asarray(base["atom_type"])
    atom_ptr = np.asarray(base["atom_ptr"])
    bond_ptr = np.asarray(base["bond_ptr"])
    bond_index = np.asarray(base["bond_index"])
    bond_type = np.asarray(base["bond_type"])
    rng = np.random.default_rng(rng_seed)
    pick = rng.choice(int(base["n_molecules"]), size=min(n_mol, int(base["n_molecules"])),
                      replace=False)

    def content(obj, lab):
        """Canonical content: the multiset of (sorted node keys, label) per pair.

        The keys are *sorted* exactly like ``_edge_key``, i.e. the comparison is
        root-symmetric.  This is the right notion of equivalence here because the
        encoder averages F(P,u,v) and F(P,v,u), so two objects related by a root
        swap are the same input to the model.  Comparing with unsorted roles
        instead would make every tie group in ``_edge_key`` look like a violation.
        """
        m = lab.shape[0]
        trip = []
        for a in range(m):
            for b in range(a + 1, m):
                ka = _node_key(obj["type"][a], obj["ru"][a], obj["rv"][a])
                kb = _node_key(obj["type"][b], obj["ru"][b], obj["rv"][b])
                trip.append(_edge_key(ka, kb) + (int(lab[a, b]),))
        return sorted(trip)

    ok_renum = 0
    ok_swap = 0
    n_checked = 0
    for i in pick.tolist():
        objs = molecule_objects(atom_type, atom_ptr, bond_ptr, bond_index, bond_type, i)
        a0, a1 = int(atom_ptr[i]), int(atom_ptr[i + 1])
        b0, b1 = int(bond_ptr[i]), int(bond_ptr[i + 1])
        types = atom_type[a0:a1]
        bi = bond_index[b0:b1]
        bt = bond_type[b0:b1]

        # ---- renumbering: relabel atoms, rebuild the molecule, apply HARD_PERM
        perm = rng.permutation(len(types))
        types2 = np.empty_like(types)
        types2[perm] = types
        bi2 = perm[bi]
        objs2 = molecule_objects(types2, np.array([0, len(types2)]),
                                 np.array([0, len(bi2)]), bi2, bt, 0)
        same = True
        for o1, o2 in zip(objs, objs2):
            n1, _ = permute_labels(o1)
            n2, _ = permute_labels(o2)
            same = same and content(o1, n1) == content(o2, n2)
        ok_renum += int(same)

        # ---- root swap: flip every bond direction, rebuild, apply HARD_PERM
        bi3 = bi[:, ::-1]
        objs3 = molecule_objects(types, np.array([0, len(types)]),
                                 np.array([0, len(bi3)]), bi3, bt, 0)
        same = True
        for o1, o3 in zip(objs, objs3):
            n1, _ = permute_labels(o1)
            n3, _ = permute_labels(o3)
            # swap the role digits of the flipped build, then compare content
            o3s = dict(o3)
            o3s["ru"], o3s["rv"] = o3["rv"].copy(), o3["ru"].copy()
            same = same and content(o1, n1) == content(o3s, n3)
        ok_swap += int(same)
        n_checked += 1

    return {
        "molecules_checked": int(n_checked),
        "renumbering_invariant": int(ok_renum),
        "renumbering_fraction": ok_renum / max(n_checked, 1),
        "root_swap_invariant": int(ok_swap),
        "root_swap_fraction": ok_swap / max(n_checked, 1),
    }


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["build", "check"])
    ap.add_argument("--split", default="train")
    ap.add_argument("--molecules", type=int, default=300)
    args = ap.parse_args(argv)

    if args.cmd == "build":
        base = D.load_cache(args.split)
        out, stats = build_variant(base)
        path = D.cache_path(args.split, VARIANT)
        C.atomic_torch_save(out, path)
        inv = invariance_checks(out, n_mol=args.molecules)
        info = {"path": str(path), "sha256": C.sha256_file(path),
                "stats": stats, "invariance": inv,
                "meta": out["meta"]["hardperm"]}
        ANALYSIS.mkdir(parents=True, exist_ok=True)
        C.write_json(ANALYSIS / f"hardperm_{args.split}.json", info)
        print(json.dumps({k: v for k, v in info.items() if k != "meta"}, indent=2))
    else:
        base = D.load_cache(args.split)
        out = D.load_cache(args.split, VARIANT)
        print(json.dumps(invariance_checks(out, n_mol=args.molecules), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
