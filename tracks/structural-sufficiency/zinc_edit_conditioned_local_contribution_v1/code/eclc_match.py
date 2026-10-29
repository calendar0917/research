"""Strict, label-free matched-molecular-pair construction for ECLC-v1.

The construction uses *only* the official ZINC subset train-10k heavy-atom
graphs (atom types and bond types).  It never reads ``y``, valid or test.

A pair is accepted when two molecules share a connected body graph that is
isomorphic **with atom and bond labels, including the attachment port**, and
the molecules differ in exactly one pendant substituent region attached to
that port through a bridge.  The larger side of the bridge is the body; the
smaller side is the local edit region.

Two locality conditions are pre-registered:

* ``|sub_i| + |sub_j| <= min(|body_i|, |body_j|)``: the replaced region is
  no larger than the preserved body;
* ``|sub| <= 12``: the edit region is a local substituent, not a second
  molecule-sized fragment.

For every molecule pair we keep the match that minimises the replaced-region
size, then maximises the preserved-body size, then uses the lexicographically
smallest body key.  All ties and filters are deterministic.

RDKit is used as an independent reconstruction of the PyG graph (the GVAE atom
vocabulary carries element, charge and explicit-H count).  Any molecule whose
RDKit reconstruction fails, or any pair whose body/substituent mapping cannot
be verified by an explicit graph isomorphism, is rejected.  Canonical SMILES
keys are therefore backed by an isomorphism check, not trusted alone.
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import pickle
import time
from typing import Any, Iterable, Mapping, Sequence

import networkx as nx
import numpy as np

from eclc_common import ANALYSIS, L, ensure_dir, log, now, write_json

from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

BT = {
    1: Chem.BondType.SINGLE,
    2: Chem.BondType.DOUBLE,
    3: Chem.BondType.TRIPLE,
}

#: Frozen matching parameters (pre-registered).
MATCH_CONFIG = {
    "min_body_atoms": 5,
    "max_sub_atoms": 12,
    "max_region_to_body_ratio": "strictly_less_than_1",  # |sub_i|+|sub_j| < min(body sizes)
    "body_source": "larger side of a bridge",
    "edit_class": "unordered pair of rooted radius-1 typed signatures",
    "reject_rdkit_failure": True,
    "reject_identical_molecules": True,
    "reject_isomorphic_substituents": True,
    "verify_body_isomorphism": True,
}


class _AtomDict(dict):
    """The GVAE pickle was written with a class named ``Dictionary``."""


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__" and name == "Dictionary":
            return _AtomDict
        return super().find_class(module, name)


def load_atom_vocab() -> list[str]:
    path = L.REPO / "data/ZINC/raw/atom_dict.pickle"
    if not path.exists():
        raise FileNotFoundError(f"missing raw atom vocabulary: {path}")
    with open(path, "rb") as fh:
        d = _Unpickler(fh).load()
    words = d.__dict__["idx2word"]
    if len(words) != 28:
        # 28 is the raw GVAE vocabulary; the subset only uses 0..20.  A
        # different vocabulary means the local data is not the frozen one.
        raise RuntimeError(f"unexpected atom vocabulary size {len(words)}")
    return list(words)


def parse_atom_word(word: str) -> tuple[str, int, int]:
    parts = word.split()
    element = parts[0]
    charge = 0
    n_h = 0
    for part in parts[1:]:
        if part == "+":
            charge = 1
        elif part == "-":
            charge = -1
        elif part.startswith("H") and part[1:].isdigit():
            n_h = int(part[1:])
        else:
            raise ValueError(f"cannot parse atom word {word!r} part {part!r}")
    return element, charge, n_h


def build_rdkit_mol(x: np.ndarray, edge_index: np.ndarray, edge_attr: np.ndarray,
                    vocab: Sequence[str]) -> Chem.Mol:
    rw = Chem.RWMol()
    for t in np.asarray(x).reshape(-1).tolist():
        t = int(t)
        element, charge, n_h = parse_atom_word(vocab[t])
        atom = Chem.Atom(element)
        atom.SetFormalCharge(charge)
        atom.SetNumExplicitHs(n_h)
        atom.SetNoImplicit(True)
        atom.SetIntProp("eclc_type", t)
        rw.AddAtom(atom)
    ei = np.asarray(edge_index, dtype=np.int64)
    ea = np.asarray(edge_attr, dtype=np.int64).reshape(-1)
    for k in range(ei.shape[1]):
        u, v = int(ei[0, k]), int(ei[1, k])
        if u < v:
            bt = int(ea[k])
            if bt not in BT:
                raise ValueError(f"bond type {bt} outside 1..3")
            rw.AddBond(u, v, BT[bt])
    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    return mol


def whole_key(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def nx_graph(mol: Chem.Mol) -> nx.Graph:
    g = nx.Graph()
    for atom in mol.GetAtoms():
        g.add_node(atom.GetIdx(), label=int(atom.GetIntProp("eclc_type")))
    for bond in mol.GetBonds():
        g.add_edge(
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
            label=int(round(bond.GetBondTypeAsDouble())),
        )
    return g


def _copy_atom(src: Chem.Atom) -> Chem.Atom:
    dst = Chem.Atom(src.GetSymbol())
    dst.SetFormalCharge(src.GetFormalCharge())
    dst.SetNumExplicitHs(src.GetNumExplicitHs())
    dst.SetNoImplicit(True)
    dst.SetIsAromatic(src.GetIsAromatic())
    dst.SetIntProp("eclc_type", int(src.GetIntProp("eclc_type")))
    return dst


def fragment_mol(mol: Chem.Mol, atom_set: Iterable[int], port: int, cut_bt: int) -> Chem.Mol:
    """Build ``atom_set`` plus one dummy at ``port`` with bond ``cut_bt``.

    The dummy fills the valence of the removed bridge bond, so the fragment is
    a valid molecule whose atom labels are exactly the original atom types.
    """
    atoms = sorted(int(a) for a in atom_set)
    local = {a: i for i, a in enumerate(atoms)}
    rw = Chem.RWMol()
    for a in atoms:
        rw.AddAtom(_copy_atom(mol.GetAtomWithIdx(a)))
    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if u in local and v in local:
            rw.AddBond(local[u], local[v], bond.GetBondType())
    dummy = rw.AddAtom(Chem.Atom("*"))
    rw.GetAtomWithIdx(dummy).SetIntProp("eclc_type", -1)
    rw.AddBond(local[int(port)], dummy, BT[int(cut_bt)])
    frag = rw.GetMol()
    Chem.SanitizeMol(frag)
    return frag


def fragment_key(mol: Chem.Mol, atom_set: Iterable[int], port: int, cut_bt: int) -> str:
    return Chem.MolToSmiles(fragment_mol(mol, atom_set, port, cut_bt),
                            canonical=True, isomericSmiles=False)


def fragment_graph(mol: Chem.Mol, atom_set: Iterable[int], port: int,
                   cut_bt: int) -> nx.Graph:
    frag = fragment_mol(mol, atom_set, port, cut_bt)
    g = nx.Graph()
    for atom in frag.GetAtoms():
        g.add_node(atom.GetIdx(), label=int(atom.GetIntProp("eclc_type")))
    for bond in frag.GetBonds():
        g.add_edge(
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
            label=int(round(bond.GetBondTypeAsDouble())),
        )
    return g


def _node_match(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    return int(a["label"]) == int(b["label"])


def _edge_match(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    return int(a["label"]) == int(b["label"])


def graphs_isomorphic(g1: nx.Graph, g2: nx.Graph) -> bool:
    if g1.number_of_nodes() != g2.number_of_nodes():
        return False
    if g1.number_of_edges() != g2.number_of_edges():
        return False
    return bool(nx.is_isomorphic(g1, g2, node_match=_node_match, edge_match=_edge_match))


def atom_sig(atom: Chem.Atom) -> tuple[str, int, int]:
    return (atom.GetSymbol(), int(atom.GetFormalCharge()), int(atom.GetNumExplicitHs()))


def rooted_sig(mol: Chem.Mol, sub_atoms: Iterable[int], root: int,
               cut_bt: int, radius: int = 1) -> tuple:
    """Rooted, order-invariant signature of the substituent around the port.

    Radius 1 is the pre-registered edit class.  It contains the cut bond type,
    the root atom label and the multiset of (substituent-side neighbour label,
    bond type).  The body-side neighbours are excluded: the body is identical
    by construction, so the changed radius-1 environment is exactly this
    multiset.
    """
    atoms = frozenset(int(a) for a in sub_atoms)
    adj: dict[int, list[tuple[int, int]]] = collections.defaultdict(list)
    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if u in atoms and v in atoms:
            bt = int(round(bond.GetBondTypeAsDouble()))
            adj[u].append((v, bt))
            adj[v].append((u, bt))

    def rec(node: int, parent: int, depth: int):
        label = atom_sig(mol.GetAtomWithIdx(node))
        if depth == 0:
            return (label,)
        children = []
        for nb, bt in adj[node]:
            if nb == parent:
                continue
            children.append((bt, rec(nb, node, depth - 1)))
        children.sort(key=repr)
        return (label, tuple(children))

    return (int(cut_bt), rec(int(root), -1, int(radius)))


def sig_str(sig: tuple) -> str:
    cut_bt, root_sig = sig

    def fmt(s):
        label, children = s[0], s[1] if len(s) > 1 else ()
        el, ch, nh = label
        head = f"{el}{ch:+d}H{nh}"
        if not children:
            return head
        return head + "(" + ",".join(f"{bt}:{fmt(c)}" for bt, c in children) + ")"

    return f"cut{cut_bt}|{fmt(root_sig)}"


def edit_class(sig_i: tuple, sig_j: tuple) -> str:
    a, b = sig_str(sig_i), sig_str(sig_j)
    return a + "||" + b if a <= b else b + "||" + a


def _has_ring(mol: Chem.Mol, atoms: Iterable[int]) -> bool:
    s = set(int(a) for a in atoms)
    return any(b.IsInRing() and b.GetBeginAtomIdx() in s and b.GetEndAtomIdx() in s
               for b in mol.GetBonds())


def enumerate_bodies(mol: Chem.Mol, *, min_body: int, max_sub: int):
    """All bridge cuts, yielding the larger side as body and the smaller as sub."""
    g = nx_graph(mol)
    out = []
    for u, v in nx.bridges(g):
        cut_bt = int(g[u][v]["label"])
        g2 = g.copy()
        g2.remove_edge(u, v)
        comps = list(nx.connected_components(g2))
        if len(comps) != 2:
            raise RuntimeError("a bridge removal did not produce two components")
        cu = next(c for c in comps if u in c)
        cv = next(c for c in comps if v in c)
        for body, port, sub, sub_port in ((cu, u, cv, v), (cv, v, cu, u)):
            if len(body) < min_body or not (1 <= len(sub) <= max_sub):
                continue
            out.append({
                "body_key": fragment_key(mol, body, port, cut_bt),
                "sub_key": fragment_key(mol, sub, sub_port, cut_bt),
                "body_atoms": frozenset(int(a) for a in body),
                "sub_atoms": frozenset(int(a) for a in sub),
                "body_size": int(len(body)),
                "sub_size": int(len(sub)),
                "port": int(port),
                "sub_root": int(sub_port),
                "cut_bt": cut_bt,
                "sig": rooted_sig(mol, sub, sub_port, cut_bt, radius=1),
                "sub_has_ring": bool(_has_ring(mol, sub)),
                "body_has_ring": bool(_has_ring(mol, body)),
            })
    return out


def _canonical_body_graphs(mol: Chem.Mol, rec_i: Mapping[str, Any],
                           mol_j: Chem.Mol, rec_j: Mapping[str, Any]) -> bool:
    g1 = fragment_graph(mol, rec_i["body_atoms"], rec_i["port"], rec_i["cut_bt"])
    g2 = fragment_graph(mol_j, rec_j["body_atoms"], rec_j["port"], rec_j["cut_bt"])
    return graphs_isomorphic(g1, g2)


def build_pairs(*, min_body: int = 5, max_sub: int = 12,
                max_pairs: int | None = None, verbose: bool = True) -> dict[str, Any]:
    """Build the frozen strict edit-pair list from official train structures."""
    from torch_geometric.datasets import ZINC

    vocab = load_atom_vocab()
    t0 = time.perf_counter()
    ds = ZINC(root=str(L.C.ZINC_ROOT), subset=True, split="train")
    if len(ds) != 10000:
        raise RuntimeError(f"expected 10000 train molecules, got {len(ds)}")
    mols: list[Chem.Mol | None] = []
    rdkit_failures: list[dict] = []
    whole_keys: list[str] = []
    for i in range(len(ds)):
        d = ds[i]
        # Access only structure fields.  ``d.y`` is deliberately never read.
        try:
            m = build_rdkit_mol(d.x.reshape(-1).numpy(), d.edge_index.numpy(),
                                d.edge_attr.reshape(-1).numpy(), vocab)
            mols.append(m)
            whole_keys.append(whole_key(m))
        except Exception as exc:  # pragma: no cover - data-dependent guard
            mols.append(None)
            whole_keys.append("")
            rdkit_failures.append({"row": int(i), "error": str(exc)})
    if verbose:
        log(f"[match] built {len(mols)} RDKit mols in {time.perf_counter() - t0:.1f}s; "
            f"failures={len(rdkit_failures)}")

    body2entries: dict[str, list[dict]] = collections.defaultdict(list)
    n_fragments = 0
    for mi, m in enumerate(mols):
        if m is None:
            continue
        recs = enumerate_bodies(m, min_body=min_body, max_sub=max_sub)
        n_fragments += len(recs)
        for rec in recs:
            body2entries[rec["body_key"]].append({"molecule": mi, **rec})
    if verbose:
        log(f"[match] {n_fragments} body candidates, {len(body2entries)} distinct body keys")

    # Pair candidates.  Every candidate is checked for the locality ratio;
    # exact body isomorphism is verified after the representative is chosen.
    all_candidates: dict[tuple[int, int], list[dict[str, Any]]] = collections.defaultdict(list)
    n_raw = 0
    for body_key, entries in body2entries.items():
        if len(entries) < 2:
            continue
        for a, b in itertools.combinations(entries, 2):
            if a["molecule"] == b["molecule"]:
                continue
            if a["sub_key"] == b["sub_key"]:
                continue
            region = a["sub_size"] + b["sub_size"]
            # Pre-registered locality condition: the replaced region must fit
            # inside the smaller preserved body.
            if region >= min(a["body_size"], b["body_size"]):
                continue
            n_raw += 1
            i, j = sorted((int(a["molecule"]), int(b["molecule"])))
            if i == j:
                continue
            rec_i, rec_j = (a, b) if a["molecule"] == i else (b, a)
            all_candidates[(i, j)].append({
                "i": i,
                "j": j,
                "body_key": body_key,
                "region": int(region),
                "min_body": int(min(a["body_size"], b["body_size"])),
                "sub_i_key": rec_i["sub_key"],
                "sub_j_key": rec_j["sub_key"],
                "sub_i_size": int(rec_i["sub_size"]),
                "sub_j_size": int(rec_j["sub_size"]),
                "body_i_size": int(rec_i["body_size"]),
                "body_j_size": int(rec_j["body_size"]),
                "cut_bond": int(rec_i["cut_bt"]),
                "sig_i": rec_i["sig"],
                "sig_j": rec_j["sig"],
                "sub_i_has_ring": bool(rec_i["sub_has_ring"]),
                "sub_j_has_ring": bool(rec_j["sub_has_ring"]),
                "body_i_has_ring": bool(rec_i["body_has_ring"]),
                "body_j_has_ring": bool(rec_j["body_has_ring"]),
                "rec_i": rec_i,
                "rec_j": rec_j,
            })

    candidates: dict[tuple[int, int], dict[str, Any]] = {}
    for key, cands in all_candidates.items():
        # Prefer smaller replaced region, then larger preserved body, then
        # lexicographically smaller body key.  All three keys are deterministic.
        best = min(cands, key=lambda c: (c["region"], -c["min_body"], c["body_key"]))
        best = dict(best)
        best["n_alt_matches"] = int(len(cands) - 1)
        candidates[key] = best

    if verbose:
        log(f"[match] {n_raw} raw pair matches, {len(candidates)} unique molecule pairs before verification")

    accepted = []
    rejected = collections.Counter()
    for key in sorted(candidates):
        cand = candidates[key]
        i, j = cand["i"], cand["j"]
        mi, mj = mols[i], mols[j]
        if mi is None or mj is None:
            rejected["rdkit_missing"] += 1
            continue
        if whole_keys[i] == whole_keys[j]:
            rejected["identical_molecule"] += 1
            continue
        # The body graphs must be isomorphic including the dummy port.
        if not _canonical_body_graphs(mi, cand["rec_i"], mj, cand["rec_j"]):
            rejected["body_isomorphism_failed"] += 1
            continue
        # The substituents must not be isomorphic (otherwise the two
        # molecules would be the same graph at the labelled level).
        gsub_i = fragment_graph(mi, cand["rec_i"]["sub_atoms"],
                                cand["rec_i"]["sub_root"], cand["rec_i"]["cut_bt"])
        gsub_j = fragment_graph(mj, cand["rec_j"]["sub_atoms"],
                                cand["rec_j"]["sub_root"], cand["rec_j"]["cut_bt"])
        if graphs_isomorphic(gsub_i, gsub_j):
            rejected["isomorphic_substituents"] += 1
            continue
        # The body is the same labelled graph and the port is the unique dummy
        # neighbour, so the mapping is verified by the isomorphism above.
        cand["edit_class"] = edit_class(cand.pop("sig_i"), cand.pop("sig_j"))
        cand.pop("rec_i")
        cand.pop("rec_j")
        accepted.append(cand)
        if max_pairs is not None and len(accepted) >= int(max_pairs):
            break

    accepted.sort(key=lambda c: (c["i"], c["j"]))
    for k, c in enumerate(accepted):
        c["pair_id"] = int(k)
    if verbose:
        log(f"[match] accepted {len(accepted)} pairs; rejected={dict(rejected)}; "
            f"elapsed={time.perf_counter() - t0:.1f}s")

    return {
        "round": "zinc-edit-conditioned-local-contribution-v1",
        "created": now(),
        "labels_read": False,
        "official_test_loaded": False,
        "config": {**MATCH_CONFIG, "min_body_atoms": int(min_body),
                   "max_sub_atoms": int(max_sub)},
        "data": {
            "source": "official PyG ZINC subset=True train split",
            "n_molecules": int(len(ds)),
            "rdkit_failures": rdkit_failures,
            "n_duplicate_whole_keys": int(len(whole_keys) - len(set(whole_keys))),
        },
        "counts": {
            "n_body_candidates": int(n_fragments),
            "n_distinct_body_keys": int(len(body2entries)),
            "n_raw_pair_matches": int(n_raw),
            "n_unique_molecule_pairs_before_verification": int(len(candidates)),
            "n_rejected": {k: int(v) for k, v in sorted(rejected.items())},
            "n_accepted_pairs": int(len(accepted)),
            "n_independent_molecules": int(len({c["i"] for c in accepted} | {c["j"] for c in accepted})),
            "n_distinct_bodies": int(len({c["body_key"] for c in accepted})),
        },
        "pairs": accepted,
    }


def support_stats(pairs: Sequence[Mapping[str, Any]], n_train: int = 10000) -> dict[str, Any]:
    """Support, repetition and pair-graph statistics; no labels."""
    if not pairs:
        return {"n_pairs": 0}
    body_by_class: dict[str, set[str]] = collections.defaultdict(set)
    class_count: collections.Counter = collections.Counter()
    body_count: collections.Counter = collections.Counter()
    degree: collections.Counter = collections.Counter()
    for p in pairs:
        cls = str(p["edit_class"])
        body = str(p["body_key"])
        class_count[cls] += 1
        body_count[body] += 1
        body_by_class[cls].add(body)
        degree[int(p["i"])] += 1
        degree[int(p["j"])] += 1
    class_bodies = {k: len(v) for k, v in body_by_class.items()}
    degrees = np.asarray(sorted(degree.values()), dtype=np.float64)
    g = nx.Graph()
    for p in pairs:
        g.add_edge(int(p["i"]), int(p["j"]))
    comps = sorted((len(c) for c in nx.connected_components(g)), reverse=True)
    covered = len(degree)
    # Gini coefficient of pair degree over covered molecules.
    if len(degrees):
        d = np.sort(degrees)
        n = len(d)
        gini = float((2 * np.arange(1, n + 1) - n - 1).dot(d) / (n * d.sum())) if d.sum() else 0.0
    else:
        gini = 0.0
    ring_stats = {
        "both_substituents_acyclic": int(sum(
            1 for p in pairs if not p.get("sub_i_has_ring") and not p.get("sub_j_has_ring"))),
        "either_substituent_has_ring": int(sum(
            1 for p in pairs if p.get("sub_i_has_ring") or p.get("sub_j_has_ring"))),
        "both_bodies_have_ring": int(sum(
            1 for p in pairs if p.get("body_i_has_ring") and p.get("body_j_has_ring"))),
        "region_atoms": {
            "min": int(min(p["region"] for p in pairs)),
            "median": float(np.median([p["region"] for p in pairs])),
            "max": int(max(p["region"] for p in pairs)),
        },
        "min_body_atoms": {
            "min": int(min(p["min_body"] for p in pairs)),
            "median": float(np.median([p["min_body"] for p in pairs])),
            "max": int(max(p["min_body"] for p in pairs)),
        },
    }
    return {
        "n_pairs": int(len(pairs)),
        "n_independent_molecules": int(covered),
        "n_distinct_bodies": int(len(body_count)),
        "ring_composition": ring_stats,
        "n_edit_classes": int(len(class_count)),
        "n_edit_classes_ge3_bodies": int(sum(1 for k, v in class_bodies.items() if v >= 3)),
        "max_bodies_per_edit_class": int(max(class_bodies.values())),
        "edit_class_counts": dict(sorted(class_count.items(), key=lambda kv: (-kv[1], kv[0]))),
        "edit_class_body_counts": dict(sorted(class_bodies.items(), key=lambda kv: (-kv[1], kv[0]))),
        "body_pair_counts": dict(sorted(body_count.items(), key=lambda kv: (-kv[1], kv[0]))),
        "pair_graph": {
            "n_components": int(len(comps)),
            "largest_component_molecules": int(comps[0]) if comps else 0,
            "component_sizes_desc": [int(x) for x in comps[:20]],
        },
        "pair_degree": {
            "n_molecules_with_pairs": int(covered),
            "fraction_train_molecules_with_pairs": float(covered / n_train),
            "mean_degree_over_covered": float(degrees.mean()) if len(degrees) else 0.0,
            "max_degree": int(degrees.max()) if len(degrees) else 0,
            "median_degree": float(np.median(degrees)) if len(degrees) else 0.0,
            "p90_degree": float(np.percentile(degrees, 90)) if len(degrees) else 0.0,
            "p99_degree": float(np.percentile(degrees, 99)) if len(degrees) else 0.0,
            "gini_degree": gini,
            "total_pair_endpoints": int(2 * len(pairs)),
            "endpoint_to_train_ratio": float(2 * len(pairs) / n_train),
            "max_degree_to_mean_ratio": float(degrees.max() / degrees.mean()) if len(degrees) and degrees.mean() else 0.0,
        },
    }


def strict_substituent_class_stats(pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Descriptive: the full substituent-pair class (not the gate class)."""
    bodies: dict[str, set[str]] = collections.defaultdict(set)
    count: collections.Counter = collections.Counter()
    for p in pairs:
        cls = "||".join(sorted([str(p["sub_i_key"]), str(p["sub_j_key"])]))
        count[cls] += 1
        bodies[cls].add(str(p["body_key"]))
    return {
        "n_classes": int(len(count)),
        "n_classes_ge3_bodies": int(sum(1 for v in bodies.values() if len(v) >= 3)),
        "n_classes_ge3_pairs": int(sum(1 for v in count.values() if v >= 3)),
        "max_bodies_per_class": int(max(len(v) for v in bodies.values())) if bodies else 0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=str(ANALYSIS / "edit_pairs.json"))
    ap.add_argument("--min-body", type=int, default=MATCH_CONFIG["min_body_atoms"])
    ap.add_argument("--max-sub", type=int, default=MATCH_CONFIG["max_sub_atoms"])
    args = ap.parse_args(argv)
    ensure_dir(ANALYSIS)
    payload = build_pairs(min_body=args.min_body, max_sub=args.max_sub)
    stats = support_stats(payload["pairs"])
    strict = strict_substituent_class_stats(payload["pairs"])
    payload["support"] = stats
    payload["strict_substituent_classes"] = strict
    write_json(args.out, payload)
    print(json.dumps({"counts": payload["counts"], "support": {
        k: stats[k] for k in ("n_pairs", "n_independent_molecules", "n_distinct_bodies",
                              "n_edit_classes", "n_edit_classes_ge3_bodies",
                              "max_bodies_per_edit_class")},
        "strict": strict}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
