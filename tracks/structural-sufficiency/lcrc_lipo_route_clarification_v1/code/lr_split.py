"""Build the four new split systems (S1, S2, SG) plus the adapted S0/FT payloads.

Everything here is label-blind: only structures, the frozen edit-pair table and a
fixed salt are used.  The S0 payload is a *read-only adaptation* of the frozen
``lcrc_local_edit_error_diagnosis_v1/analysis/component_split.json``; nothing in
that directory is modified.

Rules (frozen in ``PROTOCOL.md`` §4.2 / §5):

* indivisible unit = connected component of
  ``molecule --pair-- molecule``, ``molecule --shared matched core-- core``,
  ``molecule --identical canonical molecule-- molecule``;
* paired components first, ordered by ``(size desc, sha256(salt|stable_id) asc)``,
  then pair-free singletons fill the quota;
* every unit goes to the partition with the largest *relative* shortfall
  ``(target - filled)/target``, ties ``fit > select > diagnose > audit``;
* SG = the same allocation over groups that additionally merge every pair of
  molecules sharing a (chirality-free) Bemis-Murcko scaffold; empty scaffolds
  fall back to the canonical SMILES, so acyclic molecules are not one group.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

import lr_common as C
import lr_data as D

from lr_pairs import load_pairs  # noqa: F401  (re-exported for the prepare driver)


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
# frozen edit-pair table
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# structures
# ---------------------------------------------------------------------------


def molecule_structures(smiles: Sequence[str], rows: Sequence[int]) -> dict[int, dict[str, str]]:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    out: dict[int, dict[str, str]] = {}
    for row in rows:
        mol = Chem.MolFromSmiles(str(smiles[int(row)]))
        if mol is None:
            raise SystemExit(f"row {row}: unparseable SMILES")
        canon = Chem.MolToSmiles(mol, isomericSmiles=False)
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        except Exception:
            scaffold = ""
        out[int(row)] = {
            "canonical": canon,
            "scaffold": scaffold if scaffold else canon,
            "scaffold_empty": bool(not scaffold),
        }
    return out


def build_units(
    pairs: Sequence[Mapping[str, Any]],
    rows: Sequence[int],
    structures: Mapping[int, Mapping[str, str]],
    *,
    joint_scaffold: bool = False,
    scaffold_only: bool = False,
) -> dict[str, Any]:
    dsu = DSU()
    for row in rows:
        dsu.find(int(row))
    # identical complete molecules are one unit
    by_canon: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        by_canon[structures[int(row)]["canonical"]].append(int(row))
    for group in by_canon.values():
        for other in group[1:]:
            dsu.union(group[0], other)
    # pair edges + shared matched core (kept even in the scaffold-only fallback
    # for *identical canonical molecules*, but not for MMP edges)
    if not scaffold_only:
        core_offset = 10_000_000
        for pair in pairs:
            a, b = int(pair["row_a"]), int(pair["row_b"])
            dsu.union(a, b)
            dsu.union(a, core_offset + int(pair["core_class"]))
    if joint_scaffold or scaffold_only:
        by_scaffold: dict[str, list[int]] = defaultdict(list)
        for row in rows:
            by_scaffold[structures[int(row)]["scaffold"]].append(int(row))
        for group in by_scaffold.values():
            for other in group[1:]:
                dsu.union(group[0], other)
    roots = {int(r): dsu.find(int(r)) for r in rows}
    by_root: dict[int, list[int]] = defaultdict(list)
    for row, root in roots.items():
        by_root[root].append(int(row))
    units = []
    for root, members in by_root.items():
        members = sorted(members)
        units.append({"stable_id": C.stable_id(members), "rows": members, "size": len(members)})
    units.sort(key=lambda u: u["stable_id"])
    row_to_unit = {}
    for unit in units:
        for row in unit["rows"]:
            row_to_unit[int(row)] = unit["stable_id"]
    return {"units": units, "row_to_unit": row_to_unit, "n_units": len(units),
            "largest_unit": int(max(u["size"] for u in units))}


# ---------------------------------------------------------------------------
# allocation
# ---------------------------------------------------------------------------


def allocate(
    units: Sequence[Mapping[str, Any]],
    pairs: Sequence[Mapping[str, Any]],
    row_to_unit: Mapping[int, str],
    n_rows: int,
    salt: str,
    fractions: Mapping[str, float] = C.FRACTIONS,
) -> tuple[dict[str, str], dict[str, Any]]:
    pairs_by_unit: dict[str, list[str]] = defaultdict(list)
    for pair in pairs:
        key = row_to_unit.get(int(pair["row_a"]))
        if key is None:
            continue
        if row_to_unit.get(int(pair["row_b"])) != key:
            raise SystemExit(f"pair {pair['pair_id']} straddles a unit")
        pairs_by_unit[key].append(str(pair["pair_id"]))
    targets = {p: float(fractions[p]) * float(n_rows) for p in C.PARTITIONS}
    filled = {p: 0.0 for p in C.PARTITIONS}
    order_key = lambda u: (-int(u["size"]), C.sha256_text(f"{salt}|{u['stable_id']}"))
    with_pairs = sorted([u for u in units if pairs_by_unit.get(str(u["stable_id"]))], key=order_key)
    without = sorted([u for u in units if not pairs_by_unit.get(str(u["stable_id"]))], key=order_key)
    assignment: dict[str, str] = {}
    trace: list[dict[str, Any]] = []

    def place(unit: Mapping[str, Any], phase: str) -> None:
        best = None
        for p in C.PARTITIONS:
            shortfall = (targets[p] - filled[p]) / targets[p]
            if best is None or shortfall > best[1] + 1e-12:
                best = (p, shortfall)
        part, shortfall = best
        assignment[str(unit["stable_id"])] = part
        filled[part] += float(unit["size"])
        trace.append({"unit": str(unit["stable_id"]), "size": int(unit["size"]), "partition": part,
                      "relative_shortfall_before": float(shortfall), "phase": phase,
                      "n_pairs": len(pairs_by_unit.get(str(unit["stable_id"]), []))})

    for unit in with_pairs:
        place(unit, "with_pairs")
    for unit in without:
        place(unit, "filler")
    audit = {"salt": salt, "targets": {p: float(targets[p]) for p in C.PARTITIONS},
             "filled": {p: float(filled[p]) for p in C.PARTITIONS},
             "fractions": dict(fractions),
             "n_units_with_pairs": len(with_pairs), "n_units_without_pairs": len(without),
             "n_trace_entries": len(trace), "trace_sha256": C.sha256_text(C.stable_json(trace)),
             "trace_head": trace[:12]}
    return assignment, audit


def partition_statistics(
    rows: Sequence[int],
    pairs: Sequence[Mapping[str, Any]],
    row_to_unit: Mapping[int, str],
) -> dict[str, Any]:
    row_set = {int(r) for r in rows}
    part_pairs = [p for p in pairs if int(p["row_a"]) in row_set and int(p["row_b"]) in row_set]
    straddling = [p["pair_id"] for p in pairs
                  if (int(p["row_a"]) in row_set) != (int(p["row_b"]) in row_set)]
    molecules = sorted({int(p["row_a"]) for p in part_pairs} | {int(p["row_b"]) for p in part_pairs})
    comp = Counter(str(row_to_unit[int(r)]) for r in molecules)
    return {
        "n_rows": len(row_set),
        "row_ids_sha256": C.row_sha256(sorted(row_set)),
        "n_pairs": len(part_pairs),
        "n_pair_molecules": len(molecules),
        "n_pair_components": len(comp),
        "n_classes": len({str(p["class_id"]) for p in part_pairs}),
        "n_components": len({str(row_to_unit[int(r)]) for r in row_set}),
        "n_straddling_pairs": len(straddling),
        "straddling_pair_ids": sorted(straddling)[:20],
        "pair_ids": [str(p["pair_id"]) for p in sorted(part_pairs, key=lambda q: str(q["pair_id"]))],
    }


# ---------------------------------------------------------------------------
# payload writer
# ---------------------------------------------------------------------------


def _payload(
    split: str,
    *,
    salt: str,
    rule: str,
    assignment: Mapping[str, str],
    units: Sequence[Mapping[str, Any]],
    alloc_audit: Mapping[str, Any],
    stats: Mapping[str, Mapping[str, Any]],
    status: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    rows_by_partition = {p: [] for p in C.PARTITIONS}
    units_per_partition = {p: [] for p in C.PARTITIONS}
    for unit in units:
        part = assignment[str(unit["stable_id"])]
        rows_by_partition[part].extend(int(r) for r in unit["rows"])
        units_per_partition[part].append(sorted(int(r) for r in unit["rows"]))
    for p in C.PARTITIONS:
        rows_by_partition[p] = sorted(rows_by_partition[p])
        units_per_partition[p].sort()
    payload = {
        "schema": "lcrc-lipo-route-split-v1",
        "split": split,
        "round": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "label_blind": True,
        "labels_read": False,
        "status": status,
        "salt": salt,
        "rule": rule,
        "fractions": dict(C.FRACTIONS),
        "n_rows": len([r for p in C.PARTITIONS for r in rows_by_partition[p]]),
        "allocation": dict(alloc_audit),
        "partitions": {p: dict(stats[p]) for p in C.PARTITIONS},
        "rows_by_partition": rows_by_partition,
        "units_per_partition": units_per_partition,
        "row_to_unit": {},
    }
    # row -> stable unit id (string); the metric layer hashes it to a compact id
    for unit in units:
        for r in unit["rows"]:
            payload["row_to_unit"][str(int(r))] = str(unit["stable_id"])
    payload["row_to_component"] = {
        k: int(C.sha256_text(v)[:12], 16) % (2**31) for k, v in payload["row_to_unit"].items()
    }
    if extra:
        payload.update(dict(extra))
    return payload


def _support(payload: Mapping[str, Any]) -> dict[str, Any]:
    stats = payload["partitions"]
    gates = {}
    for part in ("diagnose", "audit"):
        gates[f"{part}_pairs"] = stats[part]["n_pairs"] >= C.MIN_EDIT_PAIRS
        gates[f"{part}_components"] = stats[part]["n_pair_components"] >= C.MIN_EDIT_COMPONENTS
    gates["fit_pair_components"] = stats["fit"]["n_pair_components"] >= C.MIN_FIT_COMPONENTS
    edit_ok = all(gates[f"{p}_{k}"] for p in ("diagnose", "audit") for k in ("pairs", "components"))
    return {"gates": gates, "edit_support": "OK" if edit_ok else "LOW_SUPPORT",
            "supervision_support": "OK" if gates["fit_pair_components"] else "LOW_SUPPORT"}


def cross_partition_checks(payload: Mapping[str, Any], structures: Mapping[int, Mapping[str, str]],
                           pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    units = {p: [set(int(r) for r in u) for u in payload["units_per_partition"][p]] for p in C.PARTITIONS}
    out: dict[str, Any] = {"component_intersections": {}, "canonical_intersections": {},
                           "pair_endpoints_in_other_partition": 0, "mmp_edges_across_partitions": []}
    parts = list(C.PARTITIONS)
    for i, p in enumerate(parts):
        for q in parts[i + 1:]:
            out["component_intersections"][f"{p}|{q}"] = 0  # components are disjoint by construction
            can_p = {structures[r]["canonical"] for r in payload["rows_by_partition"][p]}
            can_q = {structures[r]["canonical"] for r in payload["rows_by_partition"][q]}
            out["canonical_intersections"][f"{p}|{q}"] = len(can_p & can_q)
    rows_of = {int(r): p for p in C.PARTITIONS for r in payload["rows_by_partition"][p]}
    for pair in pairs:
        a, b = int(pair["row_a"]), int(pair["row_b"])
        if a in rows_of and b in rows_of and rows_of[a] != rows_of[b]:
            out["mmp_edges_across_partitions"].append(str(pair["pair_id"]))
            out["pair_endpoints_in_other_partition"] += 2
    return out


# ---------------------------------------------------------------------------
# drivers
# ---------------------------------------------------------------------------


def write_s0(pairs: Sequence[Mapping[str, Any]], rows: Sequence[int],
             structures: Mapping[int, Mapping[str, str]]) -> dict[str, Any]:
    src = C.read_json(C.S0_SPLIT_JSON)
    if src.get("status") != "OK":
        raise SystemExit("upstream S0 split status is not OK")
    rows_by_partition = {p: [int(r) for r in src["rows_by_partition"][p]] for p in C.PARTITIONS}
    row_to_component = {int(k): int(v) for k, v in src["row_to_component"].items()}
    by_unit: dict[int, list[int]] = {}
    for r, uid in row_to_component.items():
        by_unit.setdefault(int(uid), []).append(int(r))
    units_per_partition = {}
    for p in C.PARTITIONS:
        units = sorted([sorted(v) for k, v in by_unit.items() if k in set(src["units_per_partition"][p])])
        if sorted(int(r) for u in units for r in u) != rows_by_partition[p]:
            raise SystemExit(f"S0 {p}: units do not cover the partition rows")
        units_per_partition[p] = units
    row_to_unit = {}
    for p in C.PARTITIONS:
        for u in units_per_partition[p]:
            key = C.stable_id(u)
            for r in u:
                row_to_unit[str(int(r))] = key
    stats = {p: partition_statistics(rows_by_partition[p], pairs, {r: row_to_unit[str(r)] for r in rows_by_partition[p]})
             for p in C.PARTITIONS}
    payload = {
        "schema": "lcrc-lipo-route-split-v1", "split": "s0", "round": C.ROUND_ID,
        "created_utc": C.utcnow(), "label_blind": True, "labels_read": False,
        "status": "OK", "salt": src.get("salt"),
        "rule": "frozen upstream component split (lcrc-local-edit-error-diagnosis-v1) reused verbatim",
        "upstream_status": src.get("status"),
        "upstream_sha256": C.sha256_file(C.S0_SPLIT_JSON),
        "fractions": dict(C.FRACTIONS), "n_rows": sum(len(v) for v in rows_by_partition.values()),
        "partitions": {p: dict(stats[p]) for p in C.PARTITIONS},
        "rows_by_partition": rows_by_partition,
        "units_per_partition": units_per_partition,
        "row_to_unit": row_to_unit,
        "row_to_component": row_to_component,
        "allocation": src.get("allocation", {}),
    }
    payload.update(_support(payload))
    payload["cross_partition"] = cross_partition_checks(payload, structures, pairs)
    C.write_json(D.SPLIT_FILES["s0"], payload)
    return payload


def write_ft(pairs: Sequence[Mapping[str, Any]], structures: Mapping[int, Mapping[str, str]]) -> dict[str, Any]:
    src = D.source()
    fit = [int(r) for r in src["rows"]["train"]]
    select = [int(r) for r in src["rows"]["valid"]]
    payload = {
        "schema": "lcrc-lipo-route-split-v1", "split": "ft", "round": C.ROUND_ID,
        "created_utc": C.utcnow(), "label_blind": True, "labels_read": False, "status": "OK",
        "salt": None,
        "rule": "full-Train development stage: Fit = all 3360 published Train rows, Select = the published 420-row Validation",
        "fractions": None, "n_rows": len(fit),
        "rows_by_partition": {"fit": fit, "select": select, "diagnose": [], "audit": []},
        "units_per_partition": {"fit": [fit], "select": [select], "diagnose": [], "audit": []},
        "row_to_unit": {str(r): "ft-train" for r in fit} | {str(r): "ft-valid" for r in select},
        "row_to_component": {str(r): 0 for r in fit},
        "partitions": {"fit": partition_statistics(fit, pairs, {r: "ft-train" for r in fit}),
                       "select": partition_statistics(select, pairs, {r: "ft-valid" for r in select}),
                       "diagnose": {"n_rows": 0, "n_pairs": 0, "n_pair_molecules": 0, "n_pair_components": 0},
                       "audit": {"n_rows": 0, "n_pairs": 0, "n_pair_molecules": 0, "n_pair_components": 0}},
        "published_test_rows_used": False,
    }
    C.write_json(D.SPLIT_FILES["ft"], payload)
    return payload


def write_component_split(split: str, salt: str, pairs, rows, structures) -> dict[str, Any]:
    units = build_units(pairs, rows, structures)
    assignment, alloc = allocate(units["units"], pairs, units["row_to_unit"], len(rows), salt)
    rows_by_partition = {p: [] for p in C.PARTITIONS}
    for unit in units["units"]:
        rows_by_partition[assignment[unit["stable_id"]]].extend(unit["rows"])
    for p in C.PARTITIONS:
        rows_by_partition[p] = sorted(rows_by_partition[p])
    stats = {p: partition_statistics(rows_by_partition[p], pairs, units["row_to_unit"]) for p in C.PARTITIONS}
    rule = ("unit = connected component of molecule<->pair / molecule<->shared matched core / "
            "identical canonical molecule; paired units first ordered by (size desc, sha256(salt|stable id) asc), "
            "then pair-free singletons; largest relative shortfall, ties fit>select>diagnose>audit")
    payload = _payload(split, salt=salt, rule=rule, assignment=assignment, units=units["units"],
                       alloc_audit=alloc, stats=stats, status="OK",
                       extra={"n_units": units["n_units"], "largest_unit": units["largest_unit"]})
    payload.update(_support(payload))
    payload["cross_partition"] = cross_partition_checks(payload, structures, pairs)
    if payload["edit_support"] != "OK":
        payload["status"] = "LOW_SUPPORT"
    C.write_json(D.SPLIT_FILES[split], payload)
    return payload


def write_sg(pairs, rows, structures) -> dict[str, Any]:
    rule_joint = ("group = union-find of (molecule<->pair, molecule<->shared matched core, identical canonical "
                  "molecule) with (same chirality-free Bemis-Murcko scaffold); empty scaffold falls back to the "
                  "canonical SMILES; groups sorted by (size desc, sha256('lipo-route-v1-SG|stable id') asc), "
                  "largest relative shortfall, ties fit>select>diagnose>audit")
    units = build_units(pairs, rows, structures, joint_scaffold=True)
    assignment, alloc = allocate(units["units"], pairs, units["row_to_unit"], len(rows),
                                 C.SG_SALT_PREFIX, C.SG_FRACTIONS)
    rows_by_partition = {p: [] for p in C.PARTITIONS}
    for unit in units["units"]:
        rows_by_partition[assignment[unit["stable_id"]]].extend(unit["rows"])
    for p in C.PARTITIONS:
        rows_by_partition[p] = sorted(rows_by_partition[p])
    realized = {p: len(rows_by_partition[p]) for p in C.PARTITIONS}
    ok_joint = all(realized[p] >= C.SG_MIN_ROWS[p] for p in C.PARTITIONS)
    mode = "SG_JOINT_GROUPS"
    payload_base = {
        "n_units": units["n_units"], "largest_unit": units["largest_unit"],
        "joint_groups_realized": realized, "joint_groups_ok": ok_joint,
    }
    if not ok_joint:
        units = build_units(pairs, rows, structures, joint_scaffold=False, scaffold_only=True)
        assignment, alloc = allocate(units["units"], pairs, units["row_to_unit"], len(rows),
                                     C.SG_SALT_PREFIX, C.SG_FRACTIONS)
        rows_by_partition = {p: [] for p in C.PARTITIONS}
        for unit in units["units"]:
            rows_by_partition[assignment[unit["stable_id"]]].extend(unit["rows"])
        for p in C.PARTITIONS:
            rows_by_partition[p] = sorted(rows_by_partition[p])
        realized = {p: len(rows_by_partition[p]) for p in C.PARTITIONS}
        ok_scaffold = all(realized[p] >= C.SG_MIN_ROWS[p] for p in C.PARTITIONS)
        mode = "SG_SCAFFOLD_ONLY" if ok_scaffold else "BLOCKED_SPLIT_FEASIBILITY"
        payload_base = {
            "n_units": units["n_units"], "largest_unit": units["largest_unit"],
            "joint_groups_realized": payload_base["joint_groups_realized"], "joint_groups_ok": False,
            "scaffold_only_realized": realized, "scaffold_only_ok": ok_scaffold,
        }
    stats = {p: partition_statistics(rows_by_partition[p], pairs, units["row_to_unit"]) for p in C.PARTITIONS}
    rule = rule_joint if mode == "SG_JOINT_GROUPS" else (
        "scaffold-only fallback: identical canonical molecules + same scaffold; MMP edges across partitions are "
        "excluded from the edit evaluation; allocation identical")
    payload = _payload("sg", salt=C.SG_SALT_PREFIX, rule=rule, assignment=assignment, units=units["units"],
                       alloc_audit=alloc, stats=stats, status=mode,
                       extra={**payload_base, "sg_mode": mode,
                              "min_rows": dict(C.SG_MIN_ROWS)})
    payload.update(_support(payload))
    payload["cross_partition"] = cross_partition_checks(payload, structures, pairs)
    if mode == "BLOCKED_SPLIT_FEASIBILITY":
        payload["status"] = "BLOCKED_SPLIT_FEASIBILITY"
    C.write_json(D.SPLIT_FILES["sg"], payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="build the S0/S1/S2/SG/FT split payloads")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    if not args.force and all(D.SPLIT_FILES[s].exists() for s in ("s1", "s2", "sg", "s0", "ft")):
        C.log("[split] all payloads already exist; use --force to rebuild")
        return 0
    src = D.source()
    rows = [int(r) for r in src["rows"]["train"]]
    pairs = load_pairs()
    C.log("[split] canonicalising molecules and scaffolds (structure only)")
    structures = molecule_structures(src["smiles"], rows)
    results = {}
    results["s0"] = write_s0(pairs, rows, structures)
    for split in ("s1", "s2"):
        results[split] = write_component_split(split, C.SPLIT_SALTS[split], pairs, rows, structures)
    results["sg"] = write_sg(pairs, rows, structures)
    results["ft"] = write_ft(pairs, structures)
    print()
    for split, payload in results.items():
        print(f"{split}: status={payload['status']} mode={payload.get('sg_mode', '')}")
        for p in C.PARTITIONS:
            s = payload["partitions"][p]
            if not s.get("n_rows"):
                print(f"  {p:9s} rows=0")
                continue
            print(f"  {p:9s} rows={s['n_rows']:5d} pairs={s['n_pairs']:4d} pair_comp={s['n_pair_components']:4d} "
                  f"classes={s['n_classes']:4d} components={s['n_components']:4d}")
        if "cross_partition" in payload:
            print("  canonical overlaps:", payload["cross_partition"]["canonical_intersections"])
            print("  mmp edges across partitions:", len(payload["cross_partition"]["mmp_edges_across_partitions"]))
        print("  support:", {k: payload[k] for k in ("edit_support", "supervision_support") if k in payload})
    summary = {s: {"status": p["status"], "partitions": {q: p["partitions"][q] for q in C.PARTITIONS},
                   "sha256": C.sha256_file(D.SPLIT_FILES[s])} for s, p in results.items()}
    C.write_json(C.ANALYSIS / "split_summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
