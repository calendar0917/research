"""P0 - build real molecule pairs by single non-ring bond cuts.

Structure-only, label-blind.  For every eligible molecule of the Lipophilicity
``random122`` **Train** rows we enumerate every non-ring single bond, cut it with
RDKit ``FragmentOnBonds`` (a dummy replaces the severed bond, so the attachment
atom keeps the valence it has in the parent), and keep the cut only when

    variable fragment: 1..4 heavy atoms
    core:              >= 10 heavy atoms and >= 75 % of every parent

Cores are bucketed with a Weisfeiler-Lehman refinement hash (candidate
generation only) and then partitioned by an **explicit labelled-graph
isomorphism with the attachment-point marker**.  Two molecules form a pair when
they share a core (up to that isomorphism) and their variable fragments are not
the same labelled fragment.

Nothing here reads a target value.  Targets are joined later, per partition, by
row id.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

import edit_common as C
import edit_graph as G

BOND_LABELS = {1.0: 1, 2.0: 2, 3.0: 3, 1.5: 4}


# ---------------------------------------------------------------------------
# molecule-level eligibility
# ---------------------------------------------------------------------------


def _bond_label(bond) -> int:
    from rdkit import Chem

    bt = bond.GetBondType()
    if bt == Chem.BondType.SINGLE:
        return 1
    if bt == Chem.BondType.DOUBLE:
        return 2
    if bt == Chem.BondType.TRIPLE:
        return 3
    if bt == Chem.BondType.AROMATIC:
        return 4
    return 5


def eligibility(smiles: str) -> dict[str, Any]:
    """Label-blind structural eligibility of one source SMILES."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"ok": False, "reason": "rdkit_parse_failed"}
    try:
        Chem.SanitizeMol(mol)
    except Exception as exc:  # pragma: no cover - defensive
        return {"ok": False, "reason": f"sanitize_failed:{type(exc).__name__}"}
    n_frags = len(Chem.GetMolFrags(mol))
    if n_frags != 1:
        return {"ok": False, "reason": "not_single_connected_component", "n_components": n_frags}
    if any(a.GetAtomicNum() == 1 for a in mol.GetAtoms()):
        return {"ok": False, "reason": "explicit_hydrogen_atoms"}
    canonical = Chem.MolToSmiles(mol)
    stereo = [t for t in ("@", "/", "\\") if t in canonical]
    if stereo:
        return {"ok": False, "reason": "stereochemistry", "stereo_tokens": stereo}
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    if len(heavy) != mol.GetNumAtoms():
        return {"ok": False, "reason": "non_heavy_atoms_present"}
    return {
        "ok": True,
        "reason": None,
        "canonical_smiles": canonical,
        "n_heavy": int(len(heavy)),
        "n_nonring_single_bonds": int(
            sum(1 for b in mol.GetBonds() if (not b.IsInRing()) and _bond_label(b) == 1)
        ),
    }


# ---------------------------------------------------------------------------
# cut -> (core graph, variable graph)
# ---------------------------------------------------------------------------


def _fragment_graph(frag, orig_ids: Sequence[int]) -> tuple[G.LGraph, int, list[int]]:
    """Labelled graph of a capped fragment.

    ``orig_ids[i]`` is the *source-molecule* atom index of fragment atom ``i``
    (``Chem.GetMolFrags(..., asMols=True)`` renumbers atoms, so the original
    indices must be carried explicitly).  Returns
    ``(graph, original index of the attachment atom, sorted original indices of
    the fragment's heavy atoms)``.
    """
    n = int(frag.GetNumAtoms())
    if len(orig_ids) != n:
        raise ValueError("orig_ids length mismatch")
    dummies = [i for i in range(n) if frag.GetAtomWithIdx(i).GetAtomicNum() == 0]
    if len(dummies) != 1:
        raise ValueError(f"fragment must carry exactly one dummy, got {len(dummies)}")
    d = dummies[0]
    neighbours = [int(a.GetIdx()) for a in frag.GetAtomWithIdx(d).GetNeighbors()]
    if len(neighbours) != 1:
        raise ValueError("dummy must have exactly one neighbour")
    attach_local = neighbours[0]
    local_index: dict[int, int] = {}
    labels: list[G.NodeLabel] = []
    heavy_orig: list[int] = []
    attach_orig: int | None = None
    for i in range(n):
        if i == d:
            continue
        atom = frag.GetAtomWithIdx(i)
        local_index[i] = len(labels)
        labels.append(
            (
                atom.GetSymbol(),
                bool(atom.GetIsAromatic()),
                int(atom.GetFormalCharge()),
                int(atom.GetTotalNumHs()),
                i == attach_local,
            )
        )
        heavy_orig.append(int(orig_ids[i]))
        if i == attach_local:
            attach_orig = int(orig_ids[i])
    if attach_orig is None:
        raise ValueError("attachment atom missing")
    edges = []
    for bond in frag.GetBonds():
        u, v = int(bond.GetBeginAtomIdx()), int(bond.GetEndAtomIdx())
        if u == d or v == d:
            continue
        a, b = local_index[u], local_index[v]
        if a > b:
            a, b = b, a
        edges.append((a, b, _bond_label(bond)))
    return G.LGraph(labels, edges), attach_orig, sorted(heavy_orig)


def _canonical_fragment_smiles(frag) -> str:
    """RDKit canonical SMILES of the capped fragment (dummy isotope normalised)."""
    from rdkit import Chem

    m = Chem.Mol(frag)
    for atom in m.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetIsotope(1)
            atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(m)


# ---------------------------------------------------------------------------
# cut enumeration for one molecule
# ---------------------------------------------------------------------------


def enumerate_cuts(smiles: str, row: int, max_var_heavy: int = 4) -> tuple[list[dict[str, Any]], dict[str, int]]:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"unparseable SMILES {smiles!r}")
    Chem.SanitizeMol(mol)
    n_orig = mol.GetNumAtoms()
    total_heavy = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    stats = defaultdict(int)
    out: list[dict[str, Any]] = []
    for bond in mol.GetBonds():
        if bond.IsInRing():
            continue
        if _bond_label(bond) != 1:
            continue
        stats["nonring_single_bonds"] += 1
        cut = Chem.FragmentOnBonds(mol, [int(bond.GetIdx())], addDummies=True)
        try:
            frag_mols = Chem.GetMolFrags(cut, asMols=True, sanitizeFrags=True)
            frag_ids = Chem.GetMolFrags(cut, asMols=False)
        except Exception:
            stats["fragmentation_failed"] += 1
            continue
        if len(frag_mols) != 2 or len(frag_ids) != 2:
            stats["not_two_fragments"] += 1
            continue
        info = []
        for frag, ids in zip(frag_mols, frag_ids):
            orig = [int(x) for x in ids]
            heavy = [
                orig[j]
                for j in range(int(frag.GetNumAtoms()))
                if frag.GetAtomWithIdx(j).GetAtomicNum() > 1
            ]
            info.append((frag, orig, heavy))
        info.sort(key=lambda t: len(t[2]))
        var_frag, var_orig, var_heavy = info[0]
        core_frag, core_orig, core_heavy = info[1]
        if not (1 <= len(var_heavy) <= int(max_var_heavy)):
            stats["var_too_large"] += 1
            continue
        if len(core_heavy) < 10:
            stats["core_too_small"] += 1
            continue
        if len(core_heavy) < 0.75 * total_heavy:
            stats["core_below_75pct"] += 1
            continue
        try:
            core_graph, core_attach_orig, core_atoms = _fragment_graph(core_frag, core_orig)
            var_graph, var_attach_orig, var_atoms = _fragment_graph(var_frag, var_orig)
        except Exception:
            stats["graph_build_failed"] += 1
            continue
        if sorted(core_atoms + var_atoms) != list(range(n_orig)):
            raise RuntimeError(f"row {row} bond {bond.GetIdx()}: core+variable atoms != molecule atoms")
        out.append(
            {
                "row": int(row),
                "bond_index": int(bond.GetIdx()),
                "core_graph": core_graph,
                "var_graph": var_graph,
                "core_smiles": _canonical_fragment_smiles(core_frag),
                "var_smiles": _canonical_fragment_smiles(var_frag),
                "core_atoms": [int(x) for x in core_atoms],
                "var_atoms": [int(x) for x in var_atoms],
                "core_attach_index": int(core_attach_orig),
                "var_attach_index": int(var_attach_orig),
                "core_heavy": int(len(core_atoms)),
                "var_heavy": int(len(var_atoms)),
                "n_heavy_total": int(total_heavy),
            }
        )
        stats["kept"] += 1
    return out, dict(stats)


# ---------------------------------------------------------------------------
# core classes
# ---------------------------------------------------------------------------


def build_core_classes(cuts: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """WL buckets -> exact isomorphism classes."""
    by_wl: dict[str, list[int]] = defaultdict(list)
    for idx, cut in enumerate(cuts):
        by_wl[str(cut["core_graph"].wl_hash())].append(idx)
    classes: list[dict[str, Any]] = []
    n_checks = 0
    n_wl_false = 0
    for wl, members in sorted(by_wl.items()):
        reps: list[dict[str, Any]] = []
        for idx in members:
            g = cuts[idx]["core_graph"]
            placed = False
            for rep in reps:
                maps, _ = g.find_isomorphisms(rep["graph"], cap=1)
                n_checks += 1
                if maps:
                    rep["members"].append(idx)
                    placed = True
                    break
            if not placed:
                reps.append({"graph": g, "wl": wl, "members": [idx]})
        for rep in reps:
            if len(rep["members"]) > 1:
                # the bucket already proves WL-positivity; count distinct canonical strings
                distinct = {str(cuts[i]["core_smiles"]) for i in rep["members"]}
                if len(distinct) > 1:
                    n_wl_false += 1
            classes.append(rep)
    for cid, rep in enumerate(classes):
        rep["class_index"] = int(cid)
        for i in rep["members"]:
            cuts[i]["core_class"] = int(cid)
    audit = {
        "n_wl_buckets": len(by_wl),
        "n_isomorphism_checks": int(n_checks),
        "n_core_classes": len(classes),
        "n_classes_with_multiple_canonical_strings": int(n_wl_false),
    }
    return classes, audit


# ---------------------------------------------------------------------------
# pair assembly
# ---------------------------------------------------------------------------


def _mapping_key(mapping: Mapping[int, int]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((int(a), int(b)) for a, b in mapping.items()))


def assemble_pairs(
    cuts: Sequence[Mapping[str, Any]], classes: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """One definition per molecule pair (max core, then canonical tie-breaks)."""
    best: dict[tuple[int, int], dict[str, Any]] = {}
    stats = defaultdict(int)
    for cls in classes:
        members = [cuts[i] for i in cls["members"]]
        by_row: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for cut in members:
            by_row[int(cut["row"])].append(cut)
        rows = sorted(by_row)
        for ii in range(len(rows)):
            for jj in range(ii + 1, len(rows)):
                ra, rb = rows[ii], rows[jj]
                for ca in by_row[ra]:
                    for cb in by_row[rb]:
                        if ca["var_smiles"] == cb["var_smiles"]:
                            stats["same_variable_fragment_skipped"] += 1
                            continue
                        stats["core_shared_candidates"] += 1
                        a, b = (ca, cb) if ca["var_smiles"] < cb["var_smiles"] else (cb, ca)
                        maps, _ = a["core_graph"].find_isomorphisms(b["core_graph"], cap=64)
                        if not maps:
                            raise RuntimeError("core class member is not isomorphic - internal bug")
                        chosen = min(maps, key=_mapping_key)
                        core_atoms_a = a["core_atoms"]
                        core_atoms_b = b["core_atoms"]
                        mapping = {int(core_atoms_a[i]): int(core_atoms_b[j]) for i, j in chosen.items()}
                        candidate = {
                            "row_a": int(a["row"]),
                            "row_b": int(b["row"]),
                            "core_class": int(a["core_class"]),
                            "core_smiles": str(a["core_smiles"]),
                            "var_a": str(a["var_smiles"]),
                            "var_b": str(b["var_smiles"]),
                            "var_heavy_a": int(a["var_heavy"]),
                            "var_heavy_b": int(b["var_heavy"]),
                            "core_heavy": int(a["core_heavy"]),
                            "core_atoms_a": [int(x) for x in core_atoms_a],
                            "core_atoms_b": [int(x) for x in core_atoms_b],
                            "var_atoms_a": [int(x) for x in a["var_atoms"]],
                            "var_atoms_b": [int(x) for x in b["var_atoms"]],
                            "attach_a": int(a["core_attach_index"]),
                            "attach_b": int(b["core_attach_index"]),
                            "mapping": {str(k): int(v) for k, v in sorted(mapping.items())},
                            "n_core_isomorphisms_capped": int(len(maps)),
                        }
                        key = (int(a["row"]), int(b["row"]))
                        prev = best.get(key)
                        if prev is None or _definition_order(candidate) < _definition_order(prev):
                            best[key] = candidate
                        else:
                            stats["superseded_definition"] += 1
    pairs = sorted(best.values(), key=lambda p: (p["row_a"], p["row_b"]))
    for i, p in enumerate(pairs):
        p["pair_id"] = f"p{p['row_a']:05d}_{p['row_b']:05d}"
        p["class_id"] = f"{p['var_a']}=>{p['var_b']}"
        p["same_var_heavy"] = bool(p["var_heavy_a"] == p["var_heavy_b"])
    return pairs, dict(stats)


def _definition_order(p: Mapping[str, Any]) -> tuple[Any, ...]:
    return (-int(p["core_heavy"]), str(p["core_smiles"]), str(p["var_a"]), str(p["var_b"]))


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def build(force: bool = False) -> dict[str, Any]:
    import dissect_source as S

    src = S.load_lipop(C.DATA_REPEAT, labels="none")
    rows = list(src["rows"]["train"])
    if C.row_sha256(rows) != C.TRAIN_ROWS_SHA256:
        raise SystemExit("random122 Train row sequence hash differs from the frozen value")
    if C.sha256_file(C.DC.LIPOP_CSV) != C.LIPOP_CSV_SHA256:
        raise SystemExit("Lipop.csv.gz sha256 differs from the frozen value")
    if C.sha256_file(C.DC.CHEMBENCH_LIPOP / C.SPLIT_FILE) != C.SPLIT_FILE_SHA256:
        raise SystemExit("split-file sha256 differs from the frozen value")
    if C.sha256_file(C.RAW_ARTIFACT) != C.RAW_ARTIFACT_SHA256:
        raise SystemExit("raw_attributes_v1.json.gz sha256 differs from the frozen value")
    if C.sha256_file(C.SPLIT_FREEZE_DIR / "split_freeze.json") == "":
        raise SystemExit("split freeze unreadable")

    C.log(f"[P0] eligibility scan over {len(rows)} Train rows (label-blind)")
    elig: dict[int, dict[str, Any]] = {}
    cuts: list[dict[str, Any]] = []
    agg = defaultdict(int)
    for k, row in enumerate(rows):
        info = eligibility(src["smiles"][row])
        elig[int(row)] = info
        if not info["ok"]:
            agg[f"excluded:{info['reason']}"] += 1
            continue
        agg["eligible_molecules"] += 1
        mol_cuts, stats = enumerate_cuts(src["smiles"][row], int(row))
        for key, value in stats.items():
            agg[f"cut:{key}"] += int(value)
        cuts.extend(mol_cuts)
        if (k + 1) % 500 == 0:
            C.log(f"[P0]   {k + 1}/{len(rows)} molecules, {len(cuts)} cuts, {dict(agg)}")
    C.log(f"[P0] {len(cuts)} cuts from {sum(1 for v in elig.values() if v['ok'])} eligible molecules")

    classes, class_audit = build_core_classes(cuts)
    C.log(f"[P0] {class_audit}")

    pairs, pair_stats = assemble_pairs(cuts, classes)
    C.log(f"[P0] {len(pairs)} unique molecule pairs; {pair_stats}")

    # per-class automorphism orbits (from the class representative)
    class_rows = []
    for cls in classes:
        orbits, n_auto, truncated = cls["graph"].orbits(cap=4096)
        n_orbits = len(set(orbits.values()))
        class_rows.append(
            {
                "core_class": int(cls["class_index"]),
                "wl_hash": str(cls["wl"]),
                "n_nodes": int(cls["graph"].n),
                "n_cuts": len(cls["members"]),
                "n_molecules": len({int(cuts[i]["row"]) for i in cls["members"]}),
                "n_automorphisms_capped": int(n_auto),
                "automorphism_truncated": bool(truncated),
                "mapping_unique": bool(n_auto == 1 and not truncated),
                "n_orbits": int(n_orbits),
                "canonical_string": str(cls["graph"].canonical_form()),
                "example_core_smiles": str(cuts[cls["members"][0]]["core_smiles"]),
            }
        )

    # ---- artifacts -------------------------------------------------------
    C.ANALYSIS.mkdir(parents=True, exist_ok=True)
    pairs_csv = C.ANALYSIS / "pairs.csv"
    fields = [
        "pair_id",
        "row_a",
        "row_b",
        "class_id",
        "core_class",
        "core_smiles",
        "core_heavy",
        "var_a",
        "var_b",
        "var_heavy_a",
        "var_heavy_b",
        "same_var_heavy",
        "smiles_a",
        "smiles_b",
    ]
    with open(pairs_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for p in pairs:
            w.writerow(
                {
                    "pair_id": p["pair_id"],
                    "row_a": p["row_a"],
                    "row_b": p["row_b"],
                    "class_id": p["class_id"],
                    "core_class": p["core_class"],
                    "core_smiles": p["core_smiles"],
                    "core_heavy": p["core_heavy"],
                    "var_a": p["var_a"],
                    "var_b": p["var_b"],
                    "var_heavy_a": p["var_heavy_a"],
                    "var_heavy_b": p["var_heavy_b"],
                    "same_var_heavy": int(p["same_var_heavy"]),
                    "smiles_a": src["smiles"][p["row_a"]],
                    "smiles_b": src["smiles"][p["row_b"]],
                }
            )
    C.write_json(
        C.ANALYSIS / "pair_mappings.json",
        {
            "schema": "lcrc-local-edit-pair-mappings-v1",
            "n_pairs": len(pairs),
            "note": (
                "core_atoms_* are original RDKit atom indices of the source molecule; mapping is a "
                "bijection core_atoms_a -> core_atoms_b from an explicit labelled-graph isomorphism; "
                "n_core_isomorphisms_capped counts the isomorphisms returned with cap 64, a value >1 "
                "means the per-atom correspondence is not unique (the summed core term is unaffected)"
            ),
            "pairs": [
                {
                    "pair_id": p["pair_id"],
                    "core_class": p["core_class"],
                    "core_atoms_a": p["core_atoms_a"],
                    "core_atoms_b": p["core_atoms_b"],
                    "var_atoms_a": p["var_atoms_a"],
                    "var_atoms_b": p["var_atoms_b"],
                    "attach_a": p["attach_a"],
                    "attach_b": p["attach_b"],
                    "mapping": p["mapping"],
                    "n_core_isomorphisms_capped": p["n_core_isomorphisms_capped"],
                }
                for p in pairs
            ],
        },
    )
    used = {int(p["core_class"]) for p in pairs}
    C.write_json(
        C.ANALYSIS / "core_classes.json",
        {
            "schema": "lcrc-local-edit-core-classes-v1",
            "n_core_classes_total": len(class_rows),
            "note": (
                "only classes that carry at least one pair are listed; n_core_classes_total counts "
                "every exact core class found by the cut enumeration"
            ),
            "counts": {
                "n_used": len(used),
                "n_classes_with_more_than_one_isomorphism": int(
                    sum(1 for c in class_rows if c["core_class"] in used and not c["mapping_unique"])
                ),
                "n_used_with_truncated_automorphism_search": int(
                    sum(1 for c in class_rows if c["core_class"] in used and c["automorphism_truncated"])
                ),
            },
            "classes": [c for c in class_rows if c["core_class"] in used],
        },
    )

    unique_rows = sorted({p["row_a"] for p in pairs} | {p["row_b"] for p in pairs})
    manifest = {
        "schema": "lcrc-local-edit-data-manifest-v1",
        "round": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "audit_scope": C.AUDIT_SCOPE,
        "label_blind": True,
        "labels_read": False,
        "rdkit": C.rdkit_version(),
        "sources": {
            "csv": {"path": C.track_rel(C.DC.LIPOP_CSV), "sha256": C.LIPOP_CSV_SHA256},
            "split_file": {"path": C.track_rel(C.DC.CHEMBENCH_LIPOP / C.SPLIT_FILE), "sha256": C.SPLIT_FILE_SHA256},
            "train_rows": {"n": len(rows), "sha256": C.TRAIN_ROWS_SHA256},
            "raw_artifact": {"path": C.track_rel(C.RAW_ARTIFACT), "sha256": C.RAW_ARTIFACT_SHA256},
            "split_freeze": {"path": C.track_rel(C.SPLIT_FREEZE_DIR / "split_freeze.json")},
        },
        "pair_rules": C.PAIR_RULES,
        "eligibility": {
            "counts": dict(sorted(agg.items())),
            "n_rows": len(rows),
            "n_eligible": sum(1 for v in elig.values() if v["ok"]),
            "n_excluded": sum(1 for v in elig.values() if not v["ok"]),
            "excluded_reasons": dict(
                sorted(
                    {
                        reason: sum(1 for v in elig.values() if v.get("reason") == reason)
                        for reason in {v.get("reason") for v in elig.values() if not v["ok"]}
                    }.items()
                )
            ),
        },
        "core_classes": class_audit,
        "pairs": {
            "n_pairs": len(pairs),
            "n_participating_molecules": len(unique_rows),
            "n_classes": len({p["class_id"] for p in pairs}),
            "n_core_classes_used": len({p["core_class"] for p in pairs}),
            **{k: int(v) for k, v in sorted(pair_stats.items())},
        },
        "artifacts": {
            "pairs_csv": {"path": C.track_rel(pairs_csv), "sha256": C.sha256_file(pairs_csv)},
        },
    }
    C.write_json(C.ANALYSIS / "data_manifest.json", manifest)
    C.log("[P0] wrote pairs.csv, pair_mappings.json, core_classes.json, data_manifest.json")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="P0: build real molecule pairs from single-bond cuts")
    ap.parse_args(argv)
    build()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
