"""P0 - indivisible units and the Fit/Select/Diagnose/Audit allocation (label-blind).

Unit = connected component of the bipartite graph

    molecule  --pair-->  molecule        (a valid single-bond-cut pair)
    molecule  --shared matched core-->  core isomorphism class
    molecule  --identical canonical molecule-->  molecule

so that no pair can ever straddle two partitions, and no two pairs sharing an
actual matched core can be assigned to different partitions.  This gives a
core/pair-component-disjoint internal evaluation; it is **not** Murcko
scaffold-disjoint, and the scaffold overlap is reported separately.

The allocation rule is frozen in :mod:`edit_common` (``SPLIT_RULE``) and uses
only structure: component sizes and a salted hash.  No target value is read.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

import edit_common as C


# ---------------------------------------------------------------------------
# union-find
# ---------------------------------------------------------------------------


class DSU:
    def __init__(self) -> None:
        self.parent: dict[int, int] = {}

    def find(self, x: int) -> int:
        parent = self.parent
        if x not in parent:
            parent[x] = x
            return x
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if ra < rb:
                self.parent[rb] = ra
            else:
                self.parent[ra] = rb


# ---------------------------------------------------------------------------
# units
# ---------------------------------------------------------------------------


def _canonical_smiles_map(rows: Sequence[int], smiles: Sequence[str]) -> dict[int, str]:
    from rdkit import Chem

    out: dict[int, str] = {}
    for row in rows:
        mol = Chem.MolFromSmiles(smiles[int(row)])
        if mol is None:
            raise SystemExit(f"row {row}: unparseable SMILES during component build")
        out[int(row)] = Chem.MolToSmiles(mol)
    return out


def build_units(pairs: Sequence[Mapping[str, Any]], rows: Sequence[int], canonical: Mapping[int, str]) -> dict[str, Any]:
    dsu = DSU()
    for row in rows:
        dsu.find(int(row))
    # identical complete molecules are one unit
    by_canon: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        by_canon[canonical[int(row)]].append(int(row))
    n_identical_merges = 0
    for canon, group in by_canon.items():
        first = group[0]
        for other in group[1:]:
            if dsu.find(first) != dsu.find(other):
                n_identical_merges += 1
            dsu.union(first, other)
    # pair edges + shared matched core
    core_offset = 10_000_000
    for pair in pairs:
        a, b = int(pair["row_a"]), int(pair["row_b"])
        dsu.union(a, b)
        core_node = core_offset + int(pair["core_class"])
        dsu.union(a, core_node)
    roots = {row: dsu.find(int(row)) for row in rows}
    by_root: dict[int, list[int]] = defaultdict(list)
    for row, root in roots.items():
        by_root[root].append(row)
    units = []
    for root, members in by_root.items():
        members = sorted(members)
        units.append({"unit_id": int(min(members)), "rows": members, "size": len(members)})
    units.sort(key=lambda u: (u["unit_id"]))
    row_to_unit = {}
    for unit in units:
        for row in unit["rows"]:
            row_to_unit[int(row)] = unit["unit_id"]
    return {
        "units": units,
        "row_to_unit": row_to_unit,
        "n_identical_canonical_merges": int(n_identical_merges),
        "n_units": len(units),
        "largest_unit": int(max(u["size"] for u in units)),
    }


# ---------------------------------------------------------------------------
# allocation
# ---------------------------------------------------------------------------


def allocate_split(
    units: Sequence[Mapping[str, Any]],
    pairs: Sequence[Mapping[str, Any]],
    row_to_unit: Mapping[int, int],
    n_rows: int,
) -> tuple[dict[int, str], dict[str, Any]]:
    pairs_by_unit: dict[int, list[str]] = defaultdict(list)
    for pair in pairs:
        pairs_by_unit[int(row_to_unit[int(pair["row_a"])])].append(str(pair["pair_id"]))
    targets = {p: float(C.SPLIT_FRACTIONS[p]) * float(n_rows) for p in C.PARTITIONS}
    filled = {p: 0.0 for p in C.PARTITIONS}
    order_key = lambda u: (-int(u["size"]), C.sha256_text(f"{C.SPLIT_SALT}|{int(u['unit_id'])}"))
    with_pairs = sorted([u for u in units if pairs_by_unit.get(int(u["unit_id"]))], key=order_key)
    without_pairs = sorted([u for u in units if not pairs_by_unit.get(int(u["unit_id"]))], key=order_key)
    assignment: dict[int, str] = {}
    trace = []

    def place(unit: Mapping[str, Any], phase: str) -> None:
        best = None
        for p in C.PARTITIONS:
            shortfall = (targets[p] - filled[p]) / targets[p]
            if best is None or shortfall > best[1] + 1e-12:
                best = (p, shortfall)
        part, shortfall = best
        assignment[int(unit["unit_id"])] = part
        filled[part] += float(unit["size"])
        trace.append(
            {
                "unit_id": int(unit["unit_id"]),
                "size": int(unit["size"]),
                "partition": part,
                "relative_shortfall_before": float(shortfall),
                "phase": phase,
                "n_pairs": len(pairs_by_unit.get(int(unit["unit_id"]), [])),
            }
        )

    for unit in with_pairs:
        place(unit, "with_pairs")
    for unit in without_pairs:
        place(unit, "filler")
    audit = {
        "targets": {p: float(targets[p]) for p in C.PARTITIONS},
        "filled": {p: float(filled[p]) for p in C.PARTITIONS},
        "n_units_with_pairs": len(with_pairs),
        "n_units_without_pairs": len(without_pairs),
        "rule": C.SPLIT_RULE,
        "salt": C.SPLIT_SALT,
        "fractions": dict(C.SPLIT_FRACTIONS),
        "n_trace_entries": len(trace),
        "trace_head": trace[:16],
        "trace_sha256": C.sha256_text(C.stable_json(trace)),
    }
    return assignment, audit


# ---------------------------------------------------------------------------
# partition statistics
# ---------------------------------------------------------------------------


def greedy_matching(pairs: Sequence[Mapping[str, Any]]) -> int:
    """Greedy count of pairs that share no molecule (upper bound on disjointness)."""
    used: set[int] = set()
    n = 0
    for pair in sorted(pairs, key=lambda p: str(p["pair_id"])):
        a, b = int(pair["row_a"]), int(pair["row_b"])
        if a in used or b in used:
            continue
        used.add(a)
        used.add(b)
        n += 1
    return n


def murcko(smiles: str) -> str:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        return ""


def partition_stats(
    name: str,
    rows: Sequence[int],
    pairs: Sequence[Mapping[str, Any]],
    canonical: Mapping[int, str],
    smiles: Sequence[str],
) -> dict[str, Any]:
    row_set = set(int(r) for r in rows)
    part_pairs = [p for p in pairs if int(p["row_a"]) in row_set]
    # every pair is intra-unit, therefore intra-partition by construction
    if any(int(p["row_b"]) not in row_set for p in part_pairs):
        raise SystemExit(f"partition {name}: a pair straddles the partition boundary")
    molecules = sorted({int(p["row_a"]) for p in part_pairs} | {int(p["row_b"]) for p in part_pairs})
    deg: Counter = Counter()
    for p in part_pairs:
        deg[int(p["row_a"])] += 1
        deg[int(p["row_b"])] += 1
    isolated_pairs = sum(1 for p in part_pairs if deg[int(p["row_a"])] == 1 and deg[int(p["row_b"])] == 1)
    classes = Counter(str(p["class_id"]) for p in part_pairs)
    return {
        "n_rows": len(row_set),
        "row_ids_sha256": C.row_sha256(sorted(row_set)),
        "n_pairs": len(part_pairs),
        "n_pair_molecules": len(molecules),
        "n_classes": len(classes),
        "class_counts": dict(sorted(classes.items())),
        "n_isolated_pairs": int(isolated_pairs),
        "n_greedy_disjoint_pairs": int(greedy_matching(part_pairs)),
        "same_var_heavy_pairs": int(sum(1 for p in part_pairs if bool(p["same_var_heavy"]))),
        "core_heavy_mean": float(np.mean([int(p["core_heavy"]) for p in part_pairs])) if part_pairs else 0.0,
        "var_heavy_hist": dict(sorted(Counter(int(p["var_heavy_a"]) for p in part_pairs).items())),
        "scaffolds": sorted({murcko(smiles[int(r)]) for r in row_set}),
        "pair_ids": [str(p["pair_id"]) for p in sorted(part_pairs, key=lambda p: str(p["pair_id"]))],
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="P0: build units and the frozen partition")
    ap.parse_args(argv)

    import dissect_source as S

    src = S.load_lipop(C.DATA_REPEAT, labels="none")
    rows = list(src["rows"]["train"])
    if C.row_sha256(rows) != C.TRAIN_ROWS_SHA256:
        raise SystemExit("random122 Train row sequence hash differs from the frozen value")

    pairs = list(csv.DictReader(open(C.ANALYSIS / "pairs.csv", encoding="utf-8")))
    for pair in pairs:
        pair["row_a"] = int(pair["row_a"])
        pair["row_b"] = int(pair["row_b"])
        pair["core_class"] = int(pair["core_class"])
        pair["core_heavy"] = int(pair["core_heavy"])
        pair["var_heavy_a"] = int(pair["var_heavy_a"])
        pair["same_var_heavy"] = pair["same_var_heavy"] == "1"

    C.log("[P0] canonicalising molecules for the unit graph (structure only)")
    canonical = _canonical_smiles_map(rows, src["smiles"])
    if len(set(canonical.values())) == len(rows):
        C.log("[P0] no duplicate complete molecules in the 3360 Train rows")

    units = build_units(pairs, rows, canonical)
    C.log(
        f"[P0] {units['n_units']} units, largest {units['largest_unit']}, "
        f"identical merges {units['n_identical_canonical_merges']}"
    )
    assignment, alloc_audit = allocate_split(units["units"], pairs, units["row_to_unit"], len(rows))

    part_rows: dict[str, list[int]] = {p: [] for p in C.PARTITIONS}
    for unit in units["units"]:
        part_rows[assignment[int(unit["unit_id"])]].extend(int(r) for r in unit["rows"])
    for p in C.PARTITIONS:
        part_rows[p] = sorted(part_rows[p])
    all_rows = sorted(r for p in C.PARTITIONS for r in part_rows[p])
    if all_rows != sorted(int(r) for r in rows):
        raise SystemExit("partition does not cover the 3360 Train rows exactly once")

    stats = {p: partition_stats(p, part_rows[p], pairs, canonical, src["smiles"]) for p in C.PARTITIONS}
    units_per_part = {p: sorted(u for u, q in assignment.items() if q == p) for p in C.PARTITIONS}
    n_pair_units = {
        p: len({int(units["row_to_unit"][int(pair["row_a"])]) for pair in pairs if str(pair["pair_id"]) in set(stats[p]["pair_ids"])})
        for p in C.PARTITIONS
    }
    for p in C.PARTITIONS:
        stats[p]["n_pair_units"] = int(n_pair_units[p])

    # cross-partition transformation-class coverage (seen / unseen)
    diag_classes = set(stats["diagnose"]["class_counts"])
    audit_classes = set(stats["audit"]["class_counts"])
    coverage = {
        "n_classes_total": len({str(p["class_id"]) for p in pairs}),
        "diagnose_only": sorted(diag_classes - audit_classes),
        "audit_only": sorted(audit_classes - diag_classes),
        "shared": sorted(diag_classes & audit_classes),
        "n_shared": len(diag_classes & audit_classes),
        "n_diagnose": len(diag_classes),
        "n_audit": len(audit_classes),
    }

    # Murcko scaffold overlap between partitions (reported, not the split unit)
    scaf = {p: set(stats[p]["scaffolds"]) for p in C.PARTITIONS}
    overlap = {}
    for i, p in enumerate(C.PARTITIONS):
        for q in C.PARTITIONS[i + 1 :]:
            inter = scaf[p] & scaf[q]
            union = scaf[p] | scaf[q]
            overlap[f"{p}|{q}"] = {
                "n_shared": len(inter),
                "n_scaffolds_a": len(scaf[p]),
                "n_scaffolds_b": len(scaf[q]),
                "jaccard": float(len(inter) / len(union)) if union else 0.0,
                "examples": sorted(inter)[:10],
            }

    gates = {
        "diagnose_pairs": stats["diagnose"]["n_pairs"] >= C.MIN_DIAGNOSE_PAIRS,
        "diagnose_pair_units": n_pair_units["diagnose"] >= C.MIN_DIAGNOSE_COMPONENTS,
        "diagnose_molecules": stats["diagnose"]["n_pair_molecules"] >= C.MIN_DIAGNOSE_MOLECULES,
        "audit_pairs": stats["audit"]["n_pairs"] >= C.MIN_AUDIT_PAIRS,
        "audit_pair_units": n_pair_units["audit"] >= C.MIN_AUDIT_COMPONENTS,
        "audit_molecules": stats["audit"]["n_pair_molecules"] >= C.MIN_AUDIT_MOLECULES,
    }
    ok = all(gates.values())

    payload = {
        "schema": "lcrc-local-edit-component-split-v1",
        "round": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "audit_scope": C.AUDIT_SCOPE,
        "label_blind": True,
        "labels_read": False,
        "split_rule": C.SPLIT_RULE,
        "salt": C.SPLIT_SALT,
        "fractions": dict(C.SPLIT_FRACTIONS),
        "n_rows": len(rows),
        "rows_sha256": C.row_sha256(sorted(int(r) for r in rows)),
        "units": {
            "n_units": units["n_units"],
            "largest_unit": units["largest_unit"],
            "n_identical_canonical_merges": units["n_identical_canonical_merges"],
            "size_histogram": dict(sorted(Counter(u["size"] for u in units["units"]).items())),
        },
        "allocation": alloc_audit,
        "partitions": {
            p: {
                k: v
                for k, v in stats[p].items()
                if k not in ("scaffolds", "class_counts", "pair_ids")
            }
            for p in C.PARTITIONS
        },
        "units_per_partition": units_per_part,
        "class_counts": {p: stats[p]["class_counts"] for p in C.PARTITIONS},
        "class_coverage": coverage,
        "scaffold_overlap": overlap,
        "reported_not_gated": {
            "n_disjoint_pairs_diagnose": stats["diagnose"]["n_greedy_disjoint_pairs"],
            "n_disjoint_pairs_audit": stats["audit"]["n_greedy_disjoint_pairs"],
            "n_isolated_pairs_diagnose": stats["diagnose"]["n_isolated_pairs"],
            "n_isolated_pairs_audit": stats["audit"]["n_isolated_pairs"],
            "note": "the protocol asks for these to be listed; only pairs/components/molecules are hard thresholds",
        },
        "thresholds": {
            "min_pairs": C.MIN_DIAGNOSE_PAIRS,
            "min_pair_components": C.MIN_DIAGNOSE_COMPONENTS,
            "min_molecules": C.MIN_DIAGNOSE_MOLECULES,
        },
        "gates": gates,
        "status": "OK" if ok else "DATA_INSUFFICIENT",
        "rows_by_partition": {p: part_rows[p] for p in C.PARTITIONS},
        "row_to_component": {str(k): int(v) for k, v in sorted(units["row_to_unit"].items())},
    }
    C.write_json(C.ANALYSIS / "component_split.json", payload)

    print()
    for p in C.PARTITIONS:
        s = stats[p]
        print(
            f"{p:9s} rows={s['n_rows']:5d} pairs={s['n_pairs']:4d} pair_molecules={s['n_pair_molecules']:4d} "
            f"classes={s['n_classes']:4d} isolated_pairs={s['n_isolated_pairs']:4d} "
            f"disjoint_pairs={s['n_greedy_disjoint_pairs']:4d} units={len(units_per_part[p])}"
        )
    print("class coverage:", {k: coverage[k] for k in ("n_classes_total", "n_diagnose", "n_audit", "n_shared")})
    print("scaffold overlap:", {k: round(v["jaccard"], 4) for k, v in overlap.items()})
    print("gates:", gates)
    print("STATUS:", payload["status"])
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
