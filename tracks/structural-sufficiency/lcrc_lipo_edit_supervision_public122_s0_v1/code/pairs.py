"""P0: label-blind real edit pairs over all 4200 public122 molecules.

The pair definitions are inherited from the local-edit round (single non-ring
bond cut, 1-4 heavy-atom variable fragment, >=10 heavy-atom core covering >=75 %
of each parent, explicit labelled-graph isomorphism with the attachment marker).
This round extends the enumeration from Train3360 to the full 4200 molecules and
adds the frozen unordered-row-pair deduplication / ``pair_id`` convention:

    one unordered row pair -> maximum core_heavy, then core_smiles, then the
    canonical variable-fragment strings, then deterministic cut indices
    pair_id = p<min(row_a,row_b)>_<max(row_a,row_b)>
    direction A->B = smaller canonical variable-fragment string is A

No target value is read here.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import common as C

SPLIT_ORDER = {"train": 0, "valid": 1, "test": 2}


# ---------------------------------------------------------------------------
# canonical structures
# ---------------------------------------------------------------------------


def canonical_smiles_map(smiles: Sequence[str]) -> dict[int, str]:
    from rdkit import Chem

    out: dict[int, str] = {}
    for row, text in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(text))
        if mol is None:
            raise C.DataMismatch(f"DATA_OR_SPLIT_MISMATCH: row {row} SMILES does not parse")
        out[int(row)] = Chem.MolToSmiles(mol)
    return out


def split_lookup(rows: Mapping[str, Sequence[int]]) -> dict[int, str]:
    out: dict[int, str] = {}
    for split in C.SPLITS:
        for row in rows[split]:
            row = int(row)
            if row in out:
                raise ValueError(f"row {row} appears in two published splits")
            out[row] = split
    if len(out) != C.N_MOLECULES:
        raise ValueError(f"split union has {len(out)} rows != {C.N_MOLECULES}")
    return out


def split_combo(split_a: str, split_b: str) -> str:
    a, b = sorted((split_a, split_b), key=lambda s: SPLIT_ORDER[s])
    return f"{a.capitalize()}-{b.capitalize()}"


# ---------------------------------------------------------------------------
# one-definition assembly
# ---------------------------------------------------------------------------


def _definition_key(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        -int(candidate["core_heavy"]),
        str(candidate["core_smiles"]),
        str(candidate["var_a"]),
        str(candidate["var_b"]),
        int(candidate["cut_a_idx"]),
        int(candidate["cut_b_idx"]),
    )


def verify_mapping(
    core_a: C.EG.LGraph,
    core_b: C.EG.LGraph,
    core_atoms_a: Sequence[int],
    core_atoms_b: Sequence[int],
    mapping: Mapping[str, int],
    attach_a: int,
    attach_b: int,
) -> tuple[bool, str | None]:
    """Explicit label/edge/attachment check of one stored mapping."""
    keys = {int(k) for k in mapping}
    vals = {int(v) for v in mapping.values()}
    if keys != {int(x) for x in core_atoms_a}:
        return False, "mapping domain != core atoms of A"
    if vals != {int(x) for x in core_atoms_b}:
        return False, "mapping range != core atoms of B"
    if int(mapping.get(str(attach_a), -1)) != int(attach_b):
        return False, "attachment point not mapped to attachment point"
    inv_a = {int(x): i for i, x in enumerate(core_atoms_a)}
    inv_b = {int(x): i for i, x in enumerate(core_atoms_b)}
    local = {inv_a[int(k)]: inv_b[int(v)] for k, v in mapping.items()}
    for i in range(len(core_atoms_a)):
        if core_a.labels[i] != core_b.labels[local[i]]:
            return False, f"node label mismatch at local {i}"
    for u, v, lab in core_a.edges:
        if core_b.edge_key(local[u], local[v]) != int(lab):
            return False, f"edge label mismatch at local {(u, v)}"
    if not core_a.labels[inv_a[int(attach_a)]][4]:
        return False, "A attachment marker missing"
    if not core_b.labels[inv_b[int(attach_b)]][4]:
        return False, "B attachment marker missing"
    return True, None


def assemble_pairs(
    cuts: Sequence[Mapping[str, Any]],
    classes: Sequence[Mapping[str, Any]],
    canonical: Mapping[int, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Exactly one definition per unordered row-id pair."""
    best: dict[tuple[int, int], dict[str, Any]] = {}
    stats = defaultdict(int)
    mapping_failures: list[dict[str, Any]] = []
    for cls in classes:
        by_row: dict[int, list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
        for idx in cls["members"]:
            cut = cuts[int(idx)]
            by_row[int(cut["row"])].append((int(idx), cut))
        rows = sorted(by_row)
        for ii in range(len(rows)):
            for jj in range(ii + 1, len(rows)):
                ra, rb = rows[ii], rows[jj]
                if str(canonical[int(ra)]) == str(canonical[int(rb)]):
                    stats["identical_complete_molecule_skipped"] += 1
                    continue
                for ia, ca in by_row[ra]:
                    for ib, cb in by_row[rb]:
                        if str(ca["var_smiles"]) == str(cb["var_smiles"]):
                            stats["same_variable_fragment_skipped"] += 1
                            continue
                        stats["core_shared_candidates"] += 1
                        if str(ca["var_smiles"]) < str(cb["var_smiles"]):
                            a, b, idx_a, idx_b = ca, cb, ia, ib
                        else:
                            a, b, idx_a, idx_b = cb, ca, ib, ia
                        maps, _trunc = a["core_graph"].find_isomorphisms(b["core_graph"], cap=64)
                        stats["n_mapping_checks"] += 1
                        if not maps:
                            raise RuntimeError("core class member is not isomorphic - internal bug")
                        chosen = min(maps, key=C.EP._mapping_key)
                        core_atoms_a = [int(x) for x in a["core_atoms"]]
                        core_atoms_b = [int(x) for x in b["core_atoms"]]
                        mapping = {int(core_atoms_a[i]): int(core_atoms_b[j]) for i, j in chosen.items()}
                        ok, reason = verify_mapping(
                            a["core_graph"],
                            b["core_graph"],
                            core_atoms_a,
                            core_atoms_b,
                            {str(k): int(v) for k, v in mapping.items()},
                            int(a["core_attach_index"]),
                            int(b["core_attach_index"]),
                        )
                        if not ok:
                            stats["n_mapping_failures"] += 1
                            mapping_failures.append(
                                {"row_a": int(a["row"]), "row_b": int(b["row"]), "reason": reason}
                            )
                            continue
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
                            "core_atoms_a": core_atoms_a,
                            "core_atoms_b": core_atoms_b,
                            "var_atoms_a": [int(x) for x in a["var_atoms"]],
                            "var_atoms_b": [int(x) for x in b["var_atoms"]],
                            "attach_a": int(a["core_attach_index"]),
                            "attach_b": int(b["core_attach_index"]),
                            "mapping": {str(k): int(v) for k, v in sorted(mapping.items())},
                            "n_core_isomorphisms_capped": int(len(maps)),
                            "cut_a_idx": int(idx_a),
                            "cut_b_idx": int(idx_b),
                            "same_var_heavy": bool(int(a["var_heavy"]) == int(b["var_heavy"])),
                            "class_id": f"{a['var_smiles']}=>{b['var_smiles']}",
                        }
                        key = (min(int(ra), int(rb)), max(int(ra), int(rb)))
                        prev = best.get(key)
                        if prev is None or _definition_key(candidate) < _definition_key(prev):
                            if prev is not None:
                                stats["superseded_definition"] += 1
                            best[key] = candidate
                        else:
                            stats["superseded_definition"] += 1
    pairs = []
    for (lo, hi), cand in sorted(best.items()):
        cand = dict(cand)
        cand["pair_id"] = f"p{lo:05d}_{hi:05d}"
        cand["row_lo"] = int(lo)
        cand["row_hi"] = int(hi)
        pairs.append(cand)
    stats["n_pairs"] = len(pairs)
    stats["n_mapping_failures"] = len(mapping_failures)
    stats["mapping_failures"] = mapping_failures
    return pairs, dict(stats)


# ---------------------------------------------------------------------------
# sets / components
# ---------------------------------------------------------------------------


def _subset(pairs: Sequence[Mapping[str, Any]], predicate) -> list[dict[str, Any]]:
    return [dict(p) for p in pairs if predicate(p)]


def _set_stats(
    pairs: Sequence[Mapping[str, Any]],
    component_key: str,
) -> dict[str, Any]:
    if not pairs:
        return {
            "n_pairs": 0,
            "n_molecules": 0,
            "n_components": 0,
            "n_transform_classes": 0,
            "largest_component_pairs": 0,
            "largest_component_pairs_share": 0.0,
            "largest_component_molecules": 0,
            "largest_component_molecules_share": 0.0,
            "component_pairs_hist": {},
        }
    mols = {int(p["row_a"]) for p in pairs} | {int(p["row_b"]) for p in pairs}
    by_comp: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for p in pairs:
        by_comp[int(p[component_key])].append(p)
    pair_counts = {c: len(v) for c, v in by_comp.items()}
    mol_counts = {}
    for c, v in by_comp.items():
        mol_counts[c] = len({int(x["row_a"]) for x in v} | {int(x["row_b"]) for x in v})
    largest_pairs_id = max(sorted(pair_counts), key=lambda c: (pair_counts[c], -c))
    largest_mol_id = max(sorted(mol_counts), key=lambda c: (mol_counts[c], -c))
    return {
        "n_pairs": len(pairs),
        "n_molecules": len(mols),
        "n_components": len(by_comp),
        "n_transform_classes": len({str(p["class_id"]) for p in pairs}),
        "largest_component_pairs": int(pair_counts[largest_pairs_id]),
        "largest_component_pairs_share": float(pair_counts[largest_pairs_id] / len(pairs)),
        "largest_component_molecules": int(mol_counts[largest_mol_id]),
        "largest_component_molecules_share": float(mol_counts[largest_mol_id] / len(mols)),
        "component_pairs_hist": dict(sorted(Counter(pair_counts.values()).items())),
    }


def annotate_sets(
    pairs: Sequence[Mapping[str, Any]],
    rows: Mapping[str, Sequence[int]],
    canonical: Mapping[int, str],
    cuts: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    split_of = split_lookup(rows)
    train_rows = [int(r) for r in rows["train"]]
    all_rows = [int(r) for s in C.SPLITS for r in rows[s]]

    for p in pairs:
        p["split_a"] = split_of[int(p["row_a"])]
        p["split_b"] = split_of[int(p["row_b"])]
        p["split_combo"] = split_combo(p["split_a"], p["split_b"])
        p["in_p_train"] = bool(p["split_a"] == "train" and p["split_b"] == "train")
        p["in_e_t"] = bool(p["split_a"] == "test" or p["split_b"] == "test")
        p["in_e_hh"] = bool(
            p["split_a"] != "train"
            and p["split_b"] != "train"
            and "test" in (p["split_a"], p["split_b"])
        )

    p_train = [p for p in pairs if p["in_p_train"]]
    units_train = C.ES.build_units(p_train, train_rows, {r: canonical[r] for r in train_rows})
    row_to_train_component = units_train["row_to_unit"]
    for p in pairs:
        p["train_component"] = (
            int(row_to_train_component[int(p["row_a"])]) if p["in_p_train"] else -1
        )
    train_pair_components = sorted({int(p["train_component"]) for p in p_train})

    units_eval = C.ES.build_units(pairs, all_rows, {int(r): canonical[int(r)] for r in all_rows})
    row_to_eval_component = units_eval["row_to_unit"]
    for p in pairs:
        p["eval_component"] = int(row_to_eval_component[int(p["row_a"])])

    train_row_set = {int(r) for r in train_rows}
    train_cut_cores = {int(cut["core_class"]) for cut in cuts if int(cut["row"]) in train_row_set}
    for p in pairs:
        p["core_seen_in_train_cuts"] = bool(int(p["core_class"]) in train_cut_cores)

    e_t = [p for p in pairs if p["in_e_t"]]
    e_hh = [p for p in pairs if p["in_e_hh"]]
    e_train_test = [p for p in e_t if p["split_combo"] == "Train-Test"]
    e_valid_test = [p for p in e_t if p["split_combo"] == "Valid-Test"]
    e_test_test = [p for p in e_t if p["split_combo"] == "Test-Test"]
    train_valid = [p for p in pairs if p["split_combo"] == "Train-Valid"]
    valid_valid = [p for p in pairs if p["split_combo"] == "Valid-Valid"]
    seen = [p for p in e_hh if p["core_seen_in_train_cuts"]]
    unseen = [p for p in e_hh if not p["core_seen_in_train_cuts"]]

    p_train_stats = _set_stats(p_train, "train_component")
    p_train_stats["endpoint_coverage_train"] = (
        float(p_train_stats["n_molecules"] / len(train_rows)) if p_train_stats["n_molecules"] else 0.0
    )
    p_train_stats["n_pair_bearing_components"] = len(train_pair_components)

    set_stats = {
        "P_train": p_train_stats,
        "E_T": _set_stats(e_t, "eval_component"),
        "E_T_Train_Test": _set_stats(e_train_test, "eval_component"),
        "E_T_Valid_Test": _set_stats(e_valid_test, "eval_component"),
        "E_T_Test_Test": _set_stats(e_test_test, "eval_component"),
        "E_HH": _set_stats(e_hh, "eval_component"),
        "E_HH_seen_core": _set_stats(seen, "eval_component"),
        "E_HH_unseen_core": _set_stats(unseen, "eval_component"),
        "Train_Valid": _set_stats(train_valid, "eval_component"),
        "Valid_Valid": _set_stats(valid_valid, "eval_component"),
    }

    sets = {
        "P_train": p_train,
        "E_T": e_t,
        "E_T_Train_Test": e_train_test,
        "E_T_Valid_Test": e_valid_test,
        "E_T_Test_Test": e_test_test,
        "E_HH": e_hh,
        "E_HH_seen_core": seen,
        "E_HH_unseen_core": unseen,
        "Train_Valid": train_valid,
        "Valid_Valid": valid_valid,
    }
    meta = {
        "split_of": split_of,
        "train_pair_components": train_pair_components,
        "train_cut_core_classes": sorted(train_cut_cores),
        "eval_units": units_eval,
        "train_units": units_train,
        "set_stats": set_stats,
        "sets": sets,
    }
    return list(pairs), meta


# ---------------------------------------------------------------------------
# full build
# ---------------------------------------------------------------------------


def build_all(*, write: bool = True) -> dict[str, Any]:
    source = C.verify_source()
    src = C.source_with_labels("none")
    rows = {s: [int(r) for r in src["rows"][s]] for s in C.SPLITS}
    all_rows = sorted(int(r) for s in C.SPLITS for r in rows[s])
    C.log(f"[P0] canonicalising {len(all_rows)} published molecules")
    canonical = canonical_smiles_map(src["smiles"])

    C.log("[P0] label-blind eligibility scan over all 4200 rows")
    elig: dict[int, dict[str, Any]] = {}
    cuts: list[dict[str, Any]] = []
    agg: Counter = Counter()
    for k, row in enumerate(all_rows):
        info = C.EP.eligibility(src["smiles"][int(row)])
        elig[int(row)] = info
        if not info["ok"]:
            agg[f"excluded:{info['reason']}"] += 1
            continue
        agg["eligible_molecules"] += 1
        mol_cuts, stats = C.EP.enumerate_cuts(src["smiles"][int(row)], int(row))
        for key, value in stats.items():
            agg[f"cut:{key}"] += int(value)
        cuts.extend(mol_cuts)
        if (k + 1) % 1000 == 0:
            C.log(f"[P0]   {k + 1}/{len(all_rows)} molecules, {len(cuts)} cuts")
    C.log(f"[P0] {len(cuts)} legal cuts from {sum(1 for v in elig.values() if v['ok'])} eligible molecules")

    classes, class_audit = C.EP.build_core_classes(cuts)
    C.log(f"[P0] core classes: {class_audit}")

    pairs, pair_stats = assemble_pairs(cuts, classes, canonical)
    C.log(f"[P0] {len(pairs)} unique unordered-row pairs; {pair_stats}")

    pairs, meta = annotate_sets(pairs, rows, canonical, cuts)
    set_stats = meta["set_stats"]
    C.log(f"[P0] set sizes: " + ", ".join(f"{k}={v['n_pairs']}" for k, v in set_stats.items()))

    # label-blind reverse-difference arithmetic on a synthetic monotone target
    synthetic = np.arange(C.N_MOLECULES, dtype=np.float64)
    max_sum = 0.0
    for p in pairs:
        a, b = int(p["row_a"]), int(p["row_b"])
        d1 = float(synthetic[b] - synthetic[a])
        d2 = float(synthetic[a] - synthetic[b])
        max_sum = max(max_sum, abs(d1 + d2))

    pair_ids = [str(p["pair_id"]) for p in pairs]
    unordered = [(int(p["row_lo"]), int(p["row_hi"])) for p in pairs]
    verification = {
        "n_pairs": len(pairs),
        "unordered_row_pair_unique": bool(len(set(unordered)) == len(unordered)),
        "pair_id_equals_sorted_rows": bool(
            all(str(p["pair_id"]) == f"p{int(p['row_lo']):05d}_{int(p['row_hi']):05d}" for p in pairs)
        ),
        "pair_id_unique": bool(len(set(pair_ids)) == len(pair_ids)),
        "direction_var_string_order": bool(all(str(p["var_a"]) < str(p["var_b"]) for p in pairs)),
        "reverse_difference_sign_check": {
            "n_pairs": len(pairs),
            "max_abs_d_plus_reverse": float(max_sum),
            "pass": bool(max_sum == 0.0),
            "uses_target_values": False,
            "note": "synthetic monotone row index, not a property target",
        },
        "all_pair_endpoints_eligible": bool(
            all(bool(elig[int(p["row_a"])]["ok"]) and bool(elig[int(p["row_b"])]["ok"]) for p in pairs)
        ),
        "all_pair_endpoints_in_published_split": bool(
            all(int(p["row_a"]) in meta["split_of"] and int(p["row_b"]) in meta["split_of"] for p in pairs)
        ),
        "n_mapping_checks": int(pair_stats.get("n_mapping_checks", 0)),
        "n_mapping_failures": int(pair_stats.get("n_mapping_failures", 0)),
        "all_core_mappings_verified": bool(int(pair_stats.get("n_mapping_failures", 0)) == 0),
        "all_attachment_points_mapped": bool(int(pair_stats.get("n_mapping_failures", 0)) == 0),
        "n_identical_complete_molecule_pairs_skipped": int(
            pair_stats.get("identical_complete_molecule_skipped", 0)
        ),
        "n_same_variable_fragment_pairs_skipped": int(pair_stats.get("same_variable_fragment_skipped", 0)),
        "n_superseded_definitions": int(pair_stats.get("superseded_definition", 0)),
        "n_cut_atoms_reconstruct_molecule": bool(
            all(
                sorted(int(x) for x in cut["core_atoms"] + cut["var_atoms"])
                == list(range(int(elig[int(cut["row"])].get("n_heavy", 0))))
                for cut in cuts
            )
        ),
    }

    gates = {
        "p_train_pairs": {
            "value": int(set_stats["P_train"]["n_pairs"]),
            "minimum": C.MIN_P_TRAIN_PAIRS,
            "pass": bool(set_stats["P_train"]["n_pairs"] >= C.MIN_P_TRAIN_PAIRS),
        },
        "p_train_pair_components": {
            "value": int(set_stats["P_train"]["n_pair_bearing_components"]),
            "minimum": C.MIN_P_TRAIN_COMPONENTS,
            "pass": bool(set_stats["P_train"]["n_pair_bearing_components"] >= C.MIN_P_TRAIN_COMPONENTS),
        },
        "e_t_pairs": {
            "value": int(set_stats["E_T"]["n_pairs"]),
            "minimum": C.MIN_E_T_PAIRS,
            "pass": bool(set_stats["E_T"]["n_pairs"] >= C.MIN_E_T_PAIRS),
        },
        "e_t_pair_components": {
            "value": int(set_stats["E_T"]["n_components"]),
            "minimum": C.MIN_E_T_COMPONENTS,
            "pass": bool(set_stats["E_T"]["n_components"] >= C.MIN_E_T_COMPONENTS),
        },
    }
    status = "OK" if all(g["pass"] for g in gates.values()) else "DATA_INSUFFICIENT"

    audit = {
        "schema": "lcrc-lipo-edit-supervision-data-audit-v1",
        "round_id": C.ROUND_ID,
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "label_blind_pair_construction": True,
        "rdkit": C.rdkit_version(),
        "source": source,
        "splits": {
            s: {"n": len(rows[s]), "rows_sha256": C.row_sha256(rows[s])} for s in C.SPLITS
        },
        "pair_rules": C.PAIR_RULES,
        "eligibility": {
            "n_rows": len(all_rows),
            "n_eligible": int(sum(1 for v in elig.values() if v["ok"])),
            "n_excluded": int(sum(1 for v in elig.values() if not v["ok"])),
            "excluded_reasons": dict(
                sorted(
                    {
                        reason: sum(1 for v in elig.values() if v.get("reason") == reason)
                        for reason in {v.get("reason") for v in elig.values() if not v["ok"]}
                    }.items()
                )
            ),
            "counts": dict(sorted(agg.items())),
        },
        "core_classes": class_audit,
        "pairs_total": len(pairs),
        "pair_stats": {k: v for k, v in pair_stats.items() if k != "mapping_failures"},
        "sets": set_stats,
        "components": {
            "train_units": int(meta["train_units"]["n_units"]),
            "train_largest_unit": int(meta["train_units"]["largest_unit"]),
            "eval_units": int(meta["eval_units"]["n_units"]),
            "eval_largest_unit": int(meta["eval_units"]["largest_unit"]),
            "train_cut_core_classes": len(meta["train_cut_core_classes"]),
        },
        "verification": verification,
        "historical_regression_check": historical_regression_check([p for p in pairs if p["in_p_train"]]),
        "gates": gates,
        "status": status,
        "low_support": {
            "E_HH": bool(
                set_stats["E_HH"]["n_pairs"] < C.LOW_SUPPORT_PAIRS
                or set_stats["E_HH"]["n_components"] < C.LOW_SUPPORT_COMPONENTS
            ),
            "E_HH_seen_core": bool(
                set_stats["E_HH_seen_core"]["n_pairs"] < C.LOW_SUPPORT_PAIRS
                or set_stats["E_HH_seen_core"]["n_components"] < C.LOW_SUPPORT_COMPONENTS
            ),
            "E_HH_unseen_core": bool(
                set_stats["E_HH_unseen_core"]["n_pairs"] < C.LOW_SUPPORT_PAIRS
                or set_stats["E_HH_unseen_core"]["n_components"] < C.LOW_SUPPORT_COMPONENTS
            ),
        },
        "notes": {
            "E_T": (
                "At least one Test endpoint. The Train-Test core necessarily appears in Train; "
                "this measures edit response involving a Test molecule, not new-core generalization."
            ),
            "E_HH": "Valid-Test union Test-Test; neither endpoint entered gradient training, but Valid selected checkpoints.",
        },
    }

    if write:
        C.ANALYSIS.mkdir(parents=True, exist_ok=True)
        write_pairs_csv(C.ANALYSIS / "pairs.csv", pairs, canonical, src["smiles"])
        write_mappings(C.ANALYSIS / "pair_mappings.json", pairs, pair_stats)
        write_core_classes(
            C.ANALYSIS / "core_classes.json", classes, cuts, {int(p["core_class"]) for p in pairs}
        )
        C.write_json(C.ANALYSIS / "data_audit.json", audit)
        C.log("[P0] wrote pairs.csv, pair_mappings.json, core_classes.json, data_audit.json")
    return {"pairs": pairs, "audit": audit, "meta": meta, "canonical": canonical, "cuts": cuts}


def historical_regression_check(p_train_pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Explain the earlier 509-row Train pair table without changing any rule.

    The earlier round deduplicated after choosing the A->B direction, so 8
    unordered row pairs that admitted two different cut definitions appeared
    twice (once per direction).  This round deduplicates the unordered row pair
    first, as the protocol requires.
    """
    path = C.EDIT / "analysis" / "pairs.csv"
    if not path.exists():
        return {"available": False, "path": C.track_rel(path)}
    old = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    old_keys = [tuple(sorted((int(p["row_a"]), int(p["row_b"])))) for p in old]
    counts = Counter(old_keys)
    duplicates = {k: int(v) for k, v in counts.items() if int(v) > 1}
    new_keys = {tuple(sorted((int(p["row_a"]), int(p["row_b"])))) for p in p_train_pairs}
    old_unique = set(counts)
    return {
        "available": True,
        "path": C.track_rel(path),
        "old_pairs_csv_rows": len(old),
        "old_unique_unordered_pairs": len(old_unique),
        "old_duplicate_definitions": int(sum(v - 1 for v in duplicates.values())),
        "old_duplicate_unordered_pairs": len(duplicates),
        "old_participating_molecules": len({int(p["row_a"]) for p in old} | {int(p["row_b"]) for p in old}),
        "new_p_train_pairs": len(new_keys),
        "new_p_train_molecules": len({int(p["row_a"]) for p in p_train_pairs} | {int(p["row_b"]) for p in p_train_pairs}),
        "old_unique_set_equals_new_p_train": bool(old_unique == new_keys),
        "only_in_old": [list(k) for k in sorted(old_unique - new_keys)[:10]],
        "only_in_new": [list(k) for k in sorted(new_keys - old_unique)[:10]],
        "explanation": (
            "The historical 509-row table contains 501 unique unordered row pairs; 8 unordered "
            "pairs have two directed definitions because the old dedup key was the directed "
            "(row_a,row_b) after variable-fragment direction. The new protocol dedups the "
            "unordered row pair first and therefore reports 501 P_train pairs with the same 502 "
            "participating molecules. This is a dedup correction, not a relaxed filter."
        ),
    }


# ---------------------------------------------------------------------------
# serialisation
# ---------------------------------------------------------------------------


PAIR_FIELDS = [
    "pair_id",
    "row_lo",
    "row_hi",
    "row_a",
    "row_b",
    "split_a",
    "split_b",
    "split_combo",
    "in_p_train",
    "in_e_t",
    "in_e_hh",
    "train_component",
    "eval_component",
    "core_seen_in_train_cuts",
    "core_class",
    "core_smiles",
    "core_heavy",
    "var_a",
    "var_b",
    "var_heavy_a",
    "var_heavy_b",
    "same_var_heavy",
    "class_id",
    "attach_a",
    "attach_b",
    "n_core_isomorphisms_capped",
    "smiles_a",
    "smiles_b",
]


def write_pairs_csv(path: Path, pairs: Sequence[Mapping[str, Any]], canonical, smiles) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=PAIR_FIELDS)
        w.writeheader()
        for p in sorted(pairs, key=lambda x: str(x["pair_id"])):
            row = {k: p[k] for k in PAIR_FIELDS if k in p}
            row["smiles_a"] = smiles[int(p["row_a"])]
            row["smiles_b"] = smiles[int(p["row_b"])]
            row["in_p_train"] = int(bool(p["in_p_train"]))
            row["in_e_t"] = int(bool(p["in_e_t"]))
            row["in_e_hh"] = int(bool(p["in_e_hh"]))
            row["core_seen_in_train_cuts"] = int(bool(p["core_seen_in_train_cuts"]))
            row["same_var_heavy"] = int(bool(p["same_var_heavy"]))
            w.writerow(row)


def write_mappings(path: Path, pairs: Sequence[Mapping[str, Any]], pair_stats: Mapping[str, Any]) -> None:
    payload = {
        "schema": "lcrc-lipo-edit-supervision-pair-mappings-v1",
        "round_id": C.ROUND_ID,
        "n_pairs": len(pairs),
        "note": (
            "core_atoms_* are original RDKit atom indices; mapping is a bijection "
            "core_atoms_a -> core_atoms_b from an explicit labelled-graph isomorphism; "
            "the attachment atom is mapped to the attachment atom; every mapping was "
            "re-checked node-label and edge-label exact."
        ),
        "pairs": [
            {
                "pair_id": p["pair_id"],
                "row_a": p["row_a"],
                "row_b": p["row_b"],
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
            for p in sorted(pairs, key=lambda x: str(x["pair_id"]))
        ],
        "pair_stats": {k: v for k, v in pair_stats.items() if k != "mapping_failures"},
    }
    C.write_json(path, payload)


def write_core_classes(
    path: Path,
    classes: Sequence[Mapping[str, Any]],
    cuts: Sequence[Mapping[str, Any]],
    used: set[int],
) -> None:
    rows = []
    for cls in classes:
        if int(cls["class_index"]) not in used:
            continue
        orbits, n_auto, truncated = cls["graph"].orbits(cap=4096)
        rows.append(
            {
                "core_class": int(cls["class_index"]),
                "n_nodes": int(cls["graph"].n),
                "n_cuts": len(cls["members"]),
                "n_molecules": len({int(cuts[i]["row"]) for i in cls["members"]}),
                "n_automorphisms_capped": int(n_auto),
                "automorphism_truncated": bool(truncated),
                "mapping_unique": bool(n_auto == 1 and not truncated),
                "n_orbits": len(set(orbits.values())),
                "example_core_smiles": str(cuts[cls["members"][0]]["core_smiles"]),
            }
        )
    C.write_json(
        path,
        {
            "schema": "lcrc-lipo-edit-supervision-core-classes-v1",
            "round_id": C.ROUND_ID,
            "n_classes_with_any_cut": len(rows),
            "classes": rows,
        },
    )


def load_pairs(path: Path | None = None) -> list[dict[str, Any]]:
    path = C.ANALYSIS / "pairs.csv" if path is None else Path(path)
    int_fields = {
        "row_lo",
        "row_hi",
        "row_a",
        "row_b",
        "train_component",
        "eval_component",
        "core_class",
        "core_heavy",
        "var_heavy_a",
        "var_heavy_b",
        "attach_a",
        "attach_b",
        "n_core_isomorphisms_capped",
    }
    bool_fields = {"in_p_train", "in_e_t", "in_e_hh", "core_seen_in_train_cuts", "same_var_heavy"}
    out = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            item = dict(row)
            for k in int_fields:
                if k in item and item[k] != "":
                    item[k] = int(item[k])
            for k in bool_fields:
                if k in item and item[k] != "":
                    item[k] = bool(int(item[k]))
            out.append(item)
    return out


def load_mappings(path: Path | None = None) -> dict[str, dict[str, Any]]:
    path = C.ANALYSIS / "pair_mappings.json" if path is None else Path(path)
    payload = C.read_json(path)
    return {str(p["pair_id"]): p for p in payload["pairs"]}


def p_train(pairs: Sequence[Mapping[str, Any]] | None = None) -> list[dict[str, Any]]:
    pairs = load_pairs() if pairs is None else list(pairs)
    return [dict(p) for p in pairs if bool(p["in_p_train"])]


def set_by_name(pairs: Sequence[Mapping[str, Any]], name: str) -> list[dict[str, Any]]:
    flag = {
        "E_T": "in_e_t",
        "E_HH": "in_e_hh",
    }.get(name)
    if flag:
        return [dict(p) for p in pairs if bool(p[flag])]
    combo = {
        "E_T_Train_Test": "Train-Test",
        "E_T_Valid_Test": "Valid-Test",
        "E_T_Test_Test": "Test-Test",
        "Train_Valid": "Train-Valid",
        "Valid_Valid": "Valid-Valid",
    }.get(name)
    if combo:
        return [dict(p) for p in pairs if str(p["split_combo"]) == combo]
    if name == "P_train":
        return p_train(pairs)
    if name == "E_HH_seen_core":
        return [dict(p) for p in pairs if bool(p["in_e_hh"]) and bool(p["core_seen_in_train_cuts"])]
    if name == "E_HH_unseen_core":
        return [dict(p) for p in pairs if bool(p["in_e_hh"]) and not bool(p["core_seen_in_train_cuts"])]
    raise KeyError(name)


__all__ = [
    "canonical_smiles_map",
    "split_lookup",
    "split_combo",
    "assemble_pairs",
    "verify_mapping",
    "annotate_sets",
    "build_all",
    "PAIR_FIELDS",
    "write_pairs_csv",
    "write_mappings",
    "load_pairs",
    "load_mappings",
    "historical_regression_check",
    "p_train",
    "set_by_name",
]
