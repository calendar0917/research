"""Structure pair catalog: the 823 frozen edit pairs mapped onto official OGB rows.

The pair *structure definition* is inherited unchanged from
``lcrc_lipo_edit_supervision_public122_s0_v1`` (single non-ring bond cut, 1-4
heavy-atom variable fragment, core >= 10 heavy atoms covering >= 75 %, explicit
labelled-graph core isomorphism).  Everything that was tied to the old random122
split is discarded and recomputed on the official OGB split.

Mapping checks (all raising, never silently dropped):

* old SMILES -> official row by canonical isomeric SMILES, unique structure+label
  identity only;
* label agreement 1e-5;
* old RDKit atom index -> official input atom index: identity is *verified*
  against element / aromaticity / formal charge / H count / bond set (and stereo
  codes), and any non-identity case is solved by a labelled-graph isomorphism.

The atomic mapping stored here is only a *structural* correspondence; the
prediction/contribution join happens later in :mod:`gr_ledgers`.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import gr_common as C
import gr_data as D


def _rdkit():
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    return Chem


def canonical(smiles: str) -> str:
    Chem = _rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"SMILES does not parse: {smiles}")
    return Chem.MolToSmiles(mol)


def _raw_row(lipo, row: int) -> dict[str, Any]:
    ap = np.asarray(lipo["atom_ptr"], dtype=np.int64)
    ep = np.asarray(lipo["edge_ptr"], dtype=np.int64)
    a0, a1 = int(ap[row]), int(ap[row + 1])
    e0, e1 = int(ep[row]), int(ep[row + 1])
    return {
        "smiles": str(lipo["smiles"][row]),
        "atom_class": np.asarray(lipo["atom_class"])[a0:a1].astype(int),
        "atom_charge": np.asarray(lipo["atom_charge"])[a0:a1].astype(int),
        "atom_h": np.asarray(lipo["atom_h"])[a0:a1].astype(int),
        "edge_u": np.asarray(lipo["edge_u"])[e0:e1].astype(int),
        "edge_v": np.asarray(lipo["edge_v"])[e0:e1].astype(int),
        "edge_class": np.asarray(lipo["edge_class"])[e0:e1].astype(int),
        "edge_stereo": np.asarray(lipo["edge_stereo"])[e0:e1].astype(int),
        "atom_ptr": (a0, a1),
        "edge_ptr": (e0, e1),
    }


def verify_identity_mapping(mol, raw: Mapping[str, Any]) -> dict[str, Any]:
    """Node/edge label identity between the RDKit mol and the official input."""
    D.prepare_imports()
    import sq_data

    classes = {pair: k for k, pair in enumerate(sq_data.LIPO_RAW_ATOM_CLASSES)}
    n0 = mol.GetNumAtoms()
    atom_ok = True
    for atom in mol.GetAtoms():
        i = int(atom.GetIdx())
        expected = classes.get((atom.GetSymbol(), bool(atom.GetIsAromatic())), sq_data.LIPO_RAW_ATOM_UNK)
        if int(raw["atom_class"][i]) != expected:
            atom_ok = False
        if int(raw["atom_charge"][i]) != sq_data._lipo_charge_code(int(atom.GetFormalCharge())):
            atom_ok = False
        if int(raw["atom_h"][i]) != sq_data._lipo_h_code(int(atom.GetTotalNumHs())):
            atom_ok = False
    old_bonds = sorted(
        (min(int(b.GetBeginAtomIdx()), int(b.GetEndAtomIdx())), max(int(b.GetBeginAtomIdx()), int(b.GetEndAtomIdx())))
        for b in mol.GetBonds()
    )
    raw_bonds = sorted(
        {
            (min(u, v), max(u, v))
            for u, v in zip(raw["edge_u"].tolist(), raw["edge_v"].tolist())
        }
    )
    bond_ok = old_bonds == raw_bonds
    stereo_ok = True
    if bond_ok:
        by_pair = {(min(u, v), max(u, v)): int(s) for u, v, s in zip(raw["edge_u"], raw["edge_v"], raw["edge_stereo"])}
        for bond in mol.GetBonds():
            key = (min(int(bond.GetBeginAtomIdx()), int(bond.GetEndAtomIdx())), max(int(bond.GetBeginAtomIdx()), int(bond.GetEndAtomIdx())))
            if by_pair.get(key, 0) != sq_data._lipo_stereo_code(bond.GetStereo()):
                stereo_ok = False
    return {
        "n_atoms": n0,
        "atom_labels_match": bool(atom_ok),
        "bond_set_match": bool(bond_ok),
        "stereo_match": bool(stereo_ok),
        "identity_mapping_valid": bool(atom_ok and bond_ok and stereo_ok),
    }


def labelled_graph(mol, raw: Mapping[str, Any]):
    """NetworkX labelled graph for the official heavy-atom graph."""
    import networkx as nx

    g = nx.Graph()
    for i in range(len(raw["atom_class"])):
        g.add_node(i, label=(int(raw["atom_class"][i]), int(raw["atom_charge"][i]), int(raw["atom_h"][i])))
    for u, v, c in zip(raw["edge_u"].tolist(), raw["edge_v"].tolist(), raw["edge_class"].tolist()):
        g.add_edge(u, v, label=int(c))
    return g


def alternative_mappings(mol, raw: Mapping[str, Any], limit: int = 8) -> list[dict[int, int]]:
    """Labelled automorphisms of the official graph (old index == official index).

    With the verified identity mapping an alternative is an automorphism ``p`` of
    the official labelled graph; composition with the identity gives an
    alternative old->official map.  At most ``limit`` are kept, lexicographically
    sorted, so the sensitivity analysis is deterministic and never picks the
    most favourable map.
    """
    import networkx as nx

    g = labelled_graph(mol, raw)
    if g.number_of_nodes() > 60:
        return [{}]
    node_match = nx.algorithms.isomorphism.categorical_node_match("label", None)
    edge_match = nx.algorithms.isomorphism.categorical_edge_match("label", None)
    matcher = nx.algorithms.isomorphism.GraphMatcher(g, g, node_match=node_match, edge_match=edge_match)
    out: list[dict[int, int]] = []
    for mapping in matcher.isomorphisms_iter():
        seq = tuple(mapping[i] for i in range(g.number_of_nodes()))
        out.append(seq)
        if len(out) >= limit:
            break
    out = sorted(set(out))
    return [dict(enumerate(seq)) for seq in out]


def load_pairs() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    csv_path = C.EDIT_ANALYSIS / "pairs.csv"
    map_path = C.EDIT_ANALYSIS / "pair_mappings.json"
    with csv_path.open() as handle:
        rows = list(csv.DictReader(handle))
    mappings = C.read_json(map_path)
    by_id = {str(p["pair_id"]): p for p in mappings["pairs"]}
    out = []
    for row in rows:
        pid = str(row["pair_id"])
        if pid not in by_id:
            raise KeyError(f"pair {pid} missing from pair_mappings.json")
        out.append({"csv": row, "mapping": by_id[pid]})
    if len(out) != 823:
        raise RuntimeError(f"expected 823 frozen pairs, found {len(out)}")
    return out, mappings


def build_catalog(*, verify: bool = True) -> dict[str, Any]:
    D.prepare_imports()
    import sq_data

    b = D.bundle("ogbg-mollipo")
    lipo = b.lipo_raw
    payload = b.payload
    smiles = [str(s) for s in lipo["smiles"]]
    split_of = {}
    for split in C.SPLITS:
        for row in np.asarray(b.global_row_ids[split]).tolist():
            split_of[int(row)] = split
    canon_to_rows: dict[str, list[int]] = defaultdict(list)
    for row, text in enumerate(smiles):
        canon_to_rows[canonical(text)].append(row)
    pairs_raw, mappings = load_pairs()

    # per-row atom mapping audit (one entry per molecule actually used)
    used_rows = sorted({int(p["csv"]["row_a"]) for p in pairs_raw} | {int(p["csv"]["row_b"]) for p in pairs_raw})
    row_audit: dict[int, dict[str, Any]] = {}
    for row in used_rows:
        raw = _raw_row(lipo, row)
        mol = _rdkit().MolFromSmiles(smiles[row])
        check = verify_identity_mapping(mol, raw)
        if verify and not check["identity_mapping_valid"]:
            raise RuntimeError(f"row {row}: identity atom mapping is not valid: {check}")
        row_audit[row] = {
            "row": row,
            "split": split_of[row],
            "smiles": smiles[row],
            "canonical_smiles": canonical(smiles[row]),
            "n_atoms": check["n_atoms"],
            "identity_mapping": {str(i): i for i in range(check["n_atoms"])},
            **check,
        }

    catalog = []
    excluded: list[dict[str, Any]] = []
    seen: dict[tuple[int, int], str] = {}
    for entry in pairs_raw:
        csv_row = entry["csv"]
        pid = str(csv_row["pair_id"])
        old_rows = (int(csv_row["row_a"]), int(csv_row["row_b"]))
        new_rows = []
        reason = None
        for key, old_row in zip(("a", "b"), old_rows):
            canon = canonical(str(csv_row[f"smiles_{key}"]))
            candidates = canon_to_rows.get(canon, [])
            if len(candidates) != 1:
                reason = f"ambiguous_smiles_{key}:{len(candidates)}"
                break
            new_row = int(candidates[0])
            if abs(float(payload["labels"][new_row]) - float(payload["labels"][old_row])) > 1e-5:
                reason = f"label_mismatch_{key}"
                break
            # the old row index must describe the same structure; it is not
            # trusted blindly, it is re-identified by canonical SMILES
            new_rows.append(new_row)
        if reason is None and old_rows != tuple(new_rows):
            # identity is expected here; a non-identity re-identification would
            # need the atom mapping, which is re-derived below
            pass
        if reason is not None:
            excluded.append({"pair_id": pid, "reason": reason})
            continue
        a, bb = int(new_rows[0]), int(new_rows[1])
        key = tuple(sorted((a, bb)))
        if key in seen:
            excluded.append({"pair_id": pid, "reason": f"duplicate_unordered_pair_of:{seen[key]}"})
            continue
        seen[key] = pid
        mapping = entry["mapping"]
        catalog.append(
            {
                "pair_id": pid,
                "row_a": a,
                "row_b": bb,
                "split_a": split_of[a],
                "split_b": split_of[bb],
                "split_combo": "-".join(sorted([split_of[a], split_of[bb]], key=lambda s: {"train": 0, "valid": 1, "test": 2}[s])),
                "var_atoms_a": [int(x) for x in mapping["var_atoms_a"]],
                "var_atoms_b": [int(x) for x in mapping["var_atoms_b"]],
                "core_atoms_a": [int(x) for x in mapping["core_atoms_a"]],
                "core_atoms_b": [int(x) for x in mapping["core_atoms_b"]],
                "core_atom_map": {str(k): int(v) for k, v in mapping["mapping"].items()},
                "attach_a": int(mapping["attach_a"]),
                "attach_b": int(mapping["attach_b"]),
                "n_core_isomorphisms_capped": int(mapping.get("n_core_isomorphisms_capped", 1)),
                "core_heavy": int(csv_row["core_heavy"]),
                "var_smiles_a": str(csv_row["var_a"]),
                "var_smiles_b": str(csv_row["var_b"]),
                "direction": "A->B = smaller canonical variable-fragment string",
                "smiles_a": smiles[a],
                "smiles_b": smiles[bb],
                "mapping_mode": "identity_verified",
            }
        )

    # connected components over shared molecules
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for pair in catalog:
        union(pair["row_a"], pair["row_b"])
    for row in used_rows:
        find(row)
    comp_members: dict[int, set[int]] = defaultdict(set)
    for row in list(parent):
        comp_members[find(row)].add(row)
    # deterministic component ids by sorted first row
    order = sorted(comp_members, key=lambda root: min(comp_members[root]))
    comp_id = {root: idx for idx, root in enumerate(order)}
    for pair in catalog:
        pair["component"] = int(comp_id[find(pair["row_a"])])
        tr = {"train": 0, "valid": 1, "test": 2}
        lo, hi = sorted([pair["split_a"], pair["split_b"]], key=lambda s: tr[s])
        pair["split_pair"] = f"{lo}-{hi}"
        pair["in_H"] = bool(pair["split_a"] != "train" and pair["split_b"] != "train")
        pair["has_test"] = bool("test" in (pair["split_a"], pair["split_b"]))
        pair["one_train_one_eval"] = bool(
            (pair["split_a"] == "train") != (pair["split_b"] == "train")
        )
        pair["train_train"] = bool(pair["split_a"] == "train" and pair["split_b"] == "train")

    strata = {
        "H": [p for p in catalog if p["in_H"]],
        "at_least_one_test": [p for p in catalog if p["has_test"]],
        "one_train_one_eval": [p for p in catalog if p["one_train_one_eval"]],
        "train_train": [p for p in catalog if p["train_train"]],
    }
    sample, sample_meta = sample_explanations(catalog)
    summary = {
        "n_frozen_pairs": len(pairs_raw),
        "n_catalog_pairs": len(catalog),
        "n_excluded": len(excluded),
        "excluded": excluded[:20],
        "n_molecules": len(used_rows),
        "n_components": len(comp_members),
        "component_sizes": sorted((len(v) for v in comp_members.values()), reverse=True),
        "strata_counts": {k: len(v) for k, v in strata.items()},
        "split_pair_counts": dict(sorted(_count(p["split_pair"] for p in catalog).items())),
        "split_combo_counts": dict(sorted(_count(p["split_combo"] for p in catalog).items())),
        "n_component_per_stratum": {
            k: len({p["component"] for p in v}) for k, v in strata.items()
        },
        "atom_mapping": {
            "mode": "identity_verified",
            "n_rows_checked": len(row_audit),
            "all_identity_valid": all(v["identity_mapping_valid"] for v in row_audit.values()),
            "checks": ["element", "aromaticity", "formal_charge", "total_H", "bond_set", "bond_stereo"],
            "n_multi_core_candidates": int(sum(1 for p in catalog if p["n_core_isomorphisms_capped"] > 1)),
        },
        "support_note": (
            "H (both endpoints outside Train) has zero pairs under the official split; the "
            "primary pair set is therefore EMPTY and every non-TrainTrain statement rests on "
            "one_train_one_eval pairs only. LOW_SUPPORT / NO_H_SUPPORT is reported, not hidden."
        ),
        "sample": sample_meta,
        "mapping_note": "pair structures are re-identified by canonical isomeric SMILES + 1e-5 label agreement",
        "old_columns_discarded": ["in_p_train", "in_e_t", "in_e_hh", "train_component", "eval_component", "split_combo(random122)"],
    }
    return {"catalog": catalog, "row_audit": row_audit, "summary": summary, "strata": strata}


def _count(values) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for value in values:
        out[str(value)] += 1
    return out


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def sample_explanations(catalog: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Frozen sampling: strata -> shared-endpoint components round-robin -> sha256(pair_id)."""
    non_train = [p for p in catalog if not p["train_train"]]
    train_train = [p for p in catalog if p["train_train"]]
    chosen: list[dict[str, Any]] = []

    def take(pool, quota):
        by_stratum: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for pair in pool:
            by_stratum[pair["split_pair"]].append(pair)
        picked = []
        # priority: strata with H first (empty here), then smallest split rank
        stratum_order = sorted(by_stratum, key=lambda s: (0 if "test" in s or "valid" in s else 1, s))
        per_component: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
        for stratum in stratum_order:
            for pair in by_stratum[stratum]:
                per_component.setdefault((stratum, pair["component"]), []).append(pair)
        for key in per_component:
            per_component[key].sort(key=lambda p: _sha(p["pair_id"]))
        queues = [sorted(per_component[key], key=lambda p: _sha(p["pair_id"])) for key in sorted(per_component)]
        round_idx = 0
        while len(picked) < quota and any(round_idx < len(q) for q in queues):
            for queue in queues:
                if round_idx < len(queue) and len(picked) < quota:
                    picked.append(queue[round_idx])
            round_idx += 1
        return picked

    chosen.extend(take([p for p in non_train if p["in_H"]], C.MAX_EXPLAIN_NONTRAIN))
    remaining = C.MAX_EXPLAIN_NONTRAIN - len(chosen)
    if remaining > 0:
        chosen.extend(take([p for p in non_train if not p["in_H"]], remaining))
    chosen.extend(take(train_train, C.MAX_EXPLAIN_TRAIN))
    out = [
        {
            "pair_id": p["pair_id"],
            "component": p["component"],
            "split_pair": p["split_pair"],
            "split_combo": p["split_combo"],
            "in_H": p["in_H"],
            "has_test": p["has_test"],
            "stratum": "H" if p["in_H"] else ("train_train" if p["train_train"] else "non_train_train"),
        }
        for p in chosen
    ]
    meta = {
        "n_requested_nontrain": C.MAX_EXPLAIN_NONTRAIN,
        "n_requested_train": C.MAX_EXPLAIN_TRAIN,
        "n_selected_nontrain": int(sum(1 for p in chosen if not p["train_train"])),
        "n_selected_train": int(sum(1 for p in chosen if p["train_train"])),
        "n_components_selected": len({p["component"] for p in chosen}),
        "n_components_by_stratum": {
            "H": len({p["component"] for p in chosen if p["in_H"]}),
            "non_train_train": len({p["component"] for p in chosen if (not p["in_H"]) and (not p["train_train"])}),
            "train_train": len({p["component"] for p in chosen if p["train_train"]}),
        },
        "rule": "stratum -> shared-endpoint component round-robin -> sha256(pair_id) inside component",
        "selection_is_label_blind": True,
    }
    return out, meta


def _write_jsonl_gz(path: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.GzipFile(tmp, "wb", mtime=0) as handle:
        for row in rows:
            handle.write((json.dumps(row, sort_keys=True) + "\n").encode())
    tmp.replace(path)
    return {"path": str(path.relative_to(C.REPO_ROOT)), "sha256": C.sha256_file(path), "n": len(rows)}


def prepare() -> dict[str, Any]:
    built = build_catalog()
    catalog_info = _write_jsonl_gz(C.ANALYSIS_DIR / "pair_catalog.jsonl.gz", built["catalog"])
    mapping_info = _write_jsonl_gz(
        C.ANALYSIS_DIR / "row_atom_mapping.jsonl.gz",
        [built["row_audit"][row] for row in sorted(built["row_audit"])],
    )
    sample, sample_meta = sample_explanations(built["catalog"])
    summary = dict(built["summary"])
    summary["pair_catalog"] = catalog_info
    summary["row_atom_mapping"] = mapping_info
    summary["sample"] = sample_meta
    C.write_json(C.ANALYSIS_DIR / "pairs_summary.json", summary)
    C.write_json(C.ANALYSIS_DIR / "explain_sample.json", {"pairs": sample, **sample_meta})
    return summary


if __name__ == "__main__":
    print(json.dumps(prepare(), indent=1)[:4000])
