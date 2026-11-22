"""Relational capacity (11.3): local multisets vs global bond relations.

Three probes, all structural:

A. real molecules (Train/Valid only) grouped by the exact multiset of their
   per-bond local signatures ``(S identity, C identity, cut-bond attributes)``.
   Within such a group every local input of ACRA is identical, so E_atom and
   E_bond are identical by construction; the question is whether the global
   bond-pair relation profile differs.

B. exhaustive search over **all connected unlabeled graphs on 4..7 vertices**
   (networkx graph atlas) with one common atom/bond attribute row.  This tests
   whether small graphs can even witness a relation-only difference.

C. a constructed witness found because (B) yields none: every edge of a cubic
   graph of girth >= 6 has the *same* edge-rooted radius-1/2 ball (the balls are
   trees), so two non-isomorphic cubic girth-6 graphs of equal order have
   identical local multisets.  Pappus (18 vertices) and a deterministic
   2-switch variant of it are run through the fitted ACRA_REL checkpoint.
"""

from __future__ import annotations

import argparse
import itertools
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

import acra_common as C
import acra_data as D
import acra_cache as K
import acra_model as M

PD = C.PD
RUN_REL = "ACRA_REL_S0"


def edge_signature(mol: PD.Molecule, adj, e: int, colors) -> tuple:
    """Direction-invariant local signature of bond ``e``."""
    sigs = []
    for direction in (0, 1):
        (root_s, atoms_s), (root_c, atoms_c) = D.event_balls(mol, adj, e, direction)
        ids = []
        for atoms, root in ((atoms_s, root_s), (atoms_c, root_c)):
            ids.append(D.local_identity(mol, atoms, root,
                                        D.induced_bonds(mol, atoms), colors))
        sigs.append((ids[0], ids[1]))
    bond = tuple(int(x) for x in mol.bond_feat[e])
    return (min(sigs), max(sigs), bond)


def molecule_signature(mol: PD.Molecule, adj, colors) -> tuple[tuple, ...]:
    return tuple(sorted(edge_signature(mol, adj, e, colors)
                        for e in range(int(mol.bond_u.size))))


def relation_profile(geo: Mapping[str, np.ndarray]) -> dict[str, Any]:
    cells = np.zeros(8, np.int64)
    for b, s in zip(geo["pair_bucket"].tolist(), geo["pair_shared_ring"].tolist()):
        cells[(int(b) - 1) * 2 + int(s)] += 1
    total = int(cells.sum())
    return {
        "n_bond_pairs": total,
        "bucket_shared_counts": [int(x) for x in cells],
        "shared_ring_fraction": float(cells[[1, 3, 5, 7]].sum() / max(1, total)),
    }


def _eta_squared(values: np.ndarray, labels: np.ndarray) -> float:
    """1 - within-group SS / total SS (NaN when the term has no variance)."""
    total = float(np.sum((values - values.mean()) ** 2))
    if total <= 1e-12:
        return float("nan")
    within = 0.0
    for label in np.unique(labels):
        part = values[labels == label]
        within += float(np.sum((part - part.mean()) ** 2))
    return 1.0 - within / total


def _group_ranges(values: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    worst = 0.0
    for label in np.unique(labels):
        part = values[labels == label]
        worst = max(worst, float(part.max() - part.min()))
    return {"max_within_group_range": worst}


def real_molecules(cfg, raw, pack, model, source, splits, log=print) -> dict[str, Any]:
    colors = PD.ColorSpace.from_raw(raw)
    rows = np.concatenate([splits["train"], splits["valid"]])
    signatures: dict[int, tuple] = {}
    profiles: dict[int, dict[str, Any]] = {}
    n_atoms: dict[int, int] = {}
    smiles: dict[int, str] = {}
    for row in rows.tolist():
        mol = D.raw_molecule(raw, row)
        adj = D.adjacency(mol)
        signatures[row] = molecule_signature(mol, adj, colors)
        profiles[row] = relation_profile(D.bond_geometry(mol, adj))
        n_atoms[row] = int(mol.n_atoms)
        smiles[row] = str(np.asarray(raw["smiles"])[row])
    groups: dict[tuple, list[int]] = defaultdict(list)
    for mol, sig in signatures.items():
        groups[(sig, int(n_atoms[mol]))].append(mol)
    groups = {k: sorted(v) for k, v in groups.items() if len(v) > 1}
    grouped = {m: gi for gi, members in enumerate(groups.values()) for m in members}
    ordered = sorted(grouped)
    idx = {m: i for i, m in enumerate(ordered)}
    pred = M.predict_rows(model, source, np.asarray(ordered, np.int64), "cpu")
    labels = np.asarray([grouped[m] for m in ordered])

    differing = []
    for gid, (key, members) in enumerate(groups.items()):
        if len({tuple(profiles[m]["bucket_shared_counts"]) for m in members}) > 1:
            differing.append({"group": gid, "members": members,
                              "n_bonds": len(signatures[members[0]]),
                              "n_atoms": n_atoms[members[0]]})
    train_set = set(splits["train"].tolist())
    rec: dict[str, Any] = {
        "n_molecules_considered": int(rows.size),
        "n_groups_with_identical_local_multiset": int(len(groups)),
        "n_grouped_molecules": int(len(ordered)),
        "largest_group_size": int(max((len(v) for v in groups.values()), default=0)),
        "n_groups_with_different_relation_profile": int(len(differing)),
        "within_group_term_range": {
            name: _group_ranges(np.asarray(pred[name]), labels)
            for name in ("e_atom", "e_bond", "e_rel", "yhat_std")
        },
        "eta_squared_group_explains": {
            name: _eta_squared(np.asarray(pred[name]), labels)
            for name in ("e_atom", "e_bond", "e_rel", "yhat_std")
        },
        "interpretation": "identical local multisets fix E_atom and E_bond exactly; "
                          "any within-group variation of E_rel/yhat can only come from "
                          "bond-pair relations. eta2 near 1 means the term is a function "
                          "of the local multiset; eta2 below 1 means it varies across "
                          "molecules with identical local inputs.",
        "largest_groups": [
            {
                "n_members": len(members),
                "n_atoms": int(n_atoms[members[0]]),
                "n_bonds": int(len(signatures[members[0]])),
                "relation_profiles": [profiles[m] for m in members],
                "members": [
                    {"row_id": int(pack["row_ids"][m]),
                     "split": "train" if m in train_set else "valid",
                     "smiles": smiles[m],
                     "e_atom_std": float(pred["e_atom"][idx[m]]),
                     "e_bond_std": float(pred["e_bond"][idx[m]]),
                     "e_rel_std": float(pred["e_rel"][idx[m]]),
                     "yhat_std": float(pred["yhat_std"][idx[m]])}
                    for m in members[:4]
                ],
            }
            for members in sorted(groups.values(), key=lambda v: (-len(v), v[0]))[:5]
        ],
    }
    log(f"  real: molecules={rows.size} groups={len(groups)} "
        f"largest={rec['largest_group_size']} differing={len(differing)}")
    return rec


def _to_molecule(g, atom_row, bond_row, row_id: int) -> PD.Molecule:
    n = int(g.number_of_nodes())
    edges = sorted(tuple(sorted(e)) for e in g.edges())
    return PD.Molecule(
        row_id=row_id, n_atoms=n,
        atom_feat=np.tile(atom_row.astype(np.uint8), (n, 1)),
        bond_u=np.asarray([e[0] for e in edges], np.int32),
        bond_v=np.asarray([e[1] for e in edges], np.int32),
        bond_feat=np.tile(bond_row.astype(np.uint8), (len(edges), 1)),
    )


def _attribute_rows(raw) -> tuple[np.ndarray, np.ndarray]:
    atom = np.unique(np.asarray(raw["atom_feat"]), axis=0, return_counts=True)
    bond = np.unique(np.asarray(raw["bond_feat"]), axis=0, return_counts=True)
    return atom[0][np.argmax(atom[1])], bond[0][np.argmax(bond[1])]


def _atlas_search(raw, colors, log=print) -> dict[str, Any]:
    import networkx as nx

    atom_row, bond_row = _attribute_rows(raw)
    graphs = []
    for g in nx.graph_atlas_g():
        n = g.number_of_nodes()
        if n < 4 or n > 7 or not nx.is_connected(g):
            continue
        mol = _to_molecule(g, atom_row, bond_row, len(graphs))
        adj = D.adjacency(mol)
        graphs.append({
            "graph_id": len(graphs),
            "graph6": nx.to_graph6_bytes(g, header=False).decode().strip(),
            "n_atoms": n,
            "n_bonds": int(g.number_of_edges()),
            "signature": molecule_signature(mol, adj, colors),
            "profile": relation_profile(D.bond_geometry(mol, adj)),
        })
    groups: dict[tuple, list[int]] = defaultdict(list)
    for rec in graphs:
        groups[(rec["signature"], rec["n_atoms"])].append(rec["graph_id"])
    groups = {k: v for k, v in groups.items() if len(v) > 1}
    differing = []
    for key, members in groups.items():
        if len({tuple(graphs[i]["profile"]["bucket_shared_counts"]) for i in members}) > 1:
            differing.append({"members": members, "n_profiles": len({
                tuple(graphs[i]["profile"]["bucket_shared_counts"]) for i in members})})
    differing.sort(key=lambda d: (-len(d["members"]), -d["n_profiles"], d["members"][0]))
    out = {
        "vocabulary": "all connected graphs on 4..7 vertices",
        "n_graphs_enumerated": len(graphs),
        "n_groups_with_identical_local_multiset": len(groups),
        "largest_group_size": int(max((len(v) for v in groups.values()), default=0)),
        "n_groups_with_different_relation_profile": len(differing),
        "attribute_assignment": {"atoms": atom_row.tolist(), "bonds": bond_row.tolist(),
                                 "note": "all atoms and bonds share one real attribute row"},
    }
    log(f"  atlas: graphs={len(graphs)} groups={len(groups)} differing={len(differing)}")
    return out


def _pappus_variant() -> Any:
    """Deterministic 2-switch of the Pappus graph preserving girth >= 6."""
    import networkx as nx

    base = nx.convert_node_labels_to_integers(nx.pappus_graph())
    edges = sorted(tuple(sorted(e)) for e in base.edges())
    for (a, b), (c, d) in itertools.combinations(edges, 2):
        if len({a, b, c, d}) < 4:
            continue
        for rw in (((a, c), (b, d)), ((a, d), (b, c))):
            if any(base.has_edge(*e) for e in rw):
                continue
            g = base.copy()
            g.remove_edge(a, b)
            g.remove_edge(c, d)
            g.add_edges_from(rw)
            if not nx.is_connected(g):
                continue
            if nx.girth(g) < 6:
                continue
            if nx.is_isomorphic(g, base):
                continue
            return g
    raise RuntimeError("no girth-preserving 2-switch of the Pappus graph found")


def _graph_record(g, name: str, mol, colors, model, row_index: int, source) -> dict[str, Any]:
    import networkx as nx

    adj = D.adjacency(mol)
    sig = molecule_signature(mol, adj, colors)
    pred = M.predict_rows(model, source, np.asarray([row_index], np.int64), "cpu")
    return {
        "name": name,
        "graph6": nx.to_graph6_bytes(nx.convert_node_labels_to_integers(g),
                                     header=False).decode().strip(),
        "n_atoms": int(mol.n_atoms),
        "n_bonds": int(mol.bond_u.size),
        "girth": int(nx.girth(g)),
        "signature_multiset_size": len(sig),
        "distinct_signatures": len(set(sig)),
        "signature_histogram": [int(x) for x in sorted(Counter(sig).values())],
        "relation_profile": relation_profile(D.bond_geometry(mol, adj)),
        "e_atom_std": float(pred["e_atom"][0]),
        "e_bond_std": float(pred["e_bond"][0]),
        "e_rel_std": float(pred["e_rel"][0]),
        "yhat_std": float(pred["yhat_std"][0]),
    }


def _witness(cfg, raw, ckpt, dims, colors, log=print) -> dict[str, Any]:
    import networkx as nx

    atom_row, bond_row = _attribute_rows(raw)
    pappus = nx.convert_node_labels_to_integers(nx.pappus_graph())
    variant = _pappus_variant()
    if nx.is_isomorphic(pappus, variant):
        raise RuntimeError("witness graphs are isomorphic")
    mols = [_to_molecule(pappus, atom_row, bond_row, 0),
            _to_molecule(variant, atom_row, bond_row, 1)]
    sigs = [molecule_signature(m, D.adjacency(m), colors) for m in mols]
    if sigs[0] != sigs[1]:
        raise RuntimeError("constructed witness does not share the local multiset")
    names = ["pappus", "pappus_2switch"]
    pack = K.pack_from_molecules(cfg, mols, names, colors, "synthetic_witness", [0, 1],
                                 log=lambda _m: None)
    model = M.ACRAModel(ckpt["config"], dims[0], dims[1], ckpt["arm"])
    model.load_pack(pack)
    model.load_state_dict(ckpt["state"])
    model.eval()
    source = D.BatchSource(pack, "cpu")
    graphs = [_graph_record(g, n, m, colors, model, i, source)
              for i, (g, n, m) in enumerate(zip((pappus, variant), names, mols))]
    for a, b in ((0, 1),):
        rec = graphs[b]
        rec["delta_vs_a"] = {
            "delta_e_atom_std": graphs[b]["e_atom_std"] - graphs[a]["e_atom_std"],
            "delta_e_bond_std": graphs[b]["e_bond_std"] - graphs[a]["e_bond_std"],
            "delta_e_rel_std": graphs[b]["e_rel_std"] - graphs[a]["e_rel_std"],
            "delta_yhat_std": graphs[b]["yhat_std"] - graphs[a]["yhat_std"],
        }
    out = {
        "construction": "cubic graphs of girth >= 6: after removing a cut bond every "
                        "radius-1/2 ball is a tree of the same shape, so every edge has "
                        "the same local signature.  Pappus and a deterministic 2-switch "
                        "variant are non-isomorphic, same order, identical local multiset.",
        "local_multiset_equal": True,
        "graphs": graphs,
        "pair_decomposition": [
            {"a": 0, "b": 1,
             "delta_e_atom_std": float(graphs[1]["e_atom_std"] - graphs[0]["e_atom_std"]),
             "delta_e_bond_std": float(graphs[1]["e_bond_std"] - graphs[0]["e_bond_std"]),
             "delta_e_rel_std": float(graphs[1]["e_rel_std"] - graphs[0]["e_rel_std"]),
             "delta_yhat_std": float(graphs[1]["yhat_std"] - graphs[0]["yhat_std"])},
        ],
        "shared_ring_profiles_differ": graphs[0]["relation_profile"]["bucket_shared_counts"]
        != graphs[1]["relation_profile"]["bucket_shared_counts"],
    }
    log(f"  witness: girths={[g['girth'] for g in graphs]} "
        f"delta_rel={out['pair_decomposition'][0]['delta_e_rel_std']:.4f} "
        f"delta_atom={out['pair_decomposition'][0]['delta_e_atom_std']:.2e} "
        f"delta_bond={out['pair_decomposition'][0]['delta_e_bond_std']:.2e}")
    return out


def run(cfg: Mapping[str, Any], log=print) -> dict[str, Any]:
    pack = K.ensure_pack(cfg, log=log)
    raw = D.load_raw(cfg)
    dims = C.official_feature_dims()
    splits = D.official_splits(raw)
    colors = PD.ColorSpace.from_raw(raw)
    model, ckpt = _load(RUN_REL, pack, dims)
    source = D.BatchSource(pack, "cpu")
    out: dict[str, Any] = {
        "protocol_id": cfg["protocol_id"],
        "generated_utc": C.now_utc(),
        "cache_sha256": pack.get("cache_sha256"),
        "model": {"run_id": RUN_REL, "best_epoch": int(ckpt["epoch"]),
                  "arm": ckpt["arm"], "params": ckpt.get("ledger")},
        "definitions": {
            "local_signature": "per-bond (S identity, C identity, cut-bond attributes), "
                               "direction invariant",
            "relation_profile": "counts of (bond separation bucket, common-cycle flag) "
                                "over bond pairs",
        },
        "real_molecules": real_molecules(cfg, raw, pack, model, source, splits, log=log),
        "synthetic_graphs": {
            "atlas": _atlas_search(raw, colors, log=log),
            "constructed_witness": _witness(cfg, raw, ckpt, dims, colors, log=log),
        },
        "test_labels_read": False,
    }
    C.write_json(C.analysis_dir() / "relational_capacity.json", out)
    log("wrote relational_capacity.json")
    return out


def _load(run_id, pack, dims):
    import acra_mechanism as ME

    return ME.load_model(run_id, pack, dims)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    _args = ap.parse_args(argv)
    run(C.load_config())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())