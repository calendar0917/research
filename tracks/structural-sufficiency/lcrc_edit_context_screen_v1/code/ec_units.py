"""Label-blind structural support: units and context comparisons.

A *unit* is ``(class_id, core_class, component)`` - one directed structural
replacement, one canonical core, one independent matched-pair component.  A
*comparison* joins two units of the same ``class_id`` that live in different
components **and** different core classes.  Nothing in this module opens the
Lipophilicity targets: eligibility is decided from structure only.
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

import ec_common as C


def load_pairs() -> pd.DataFrame:
    df = pd.read_csv(C.PAIRS_CSV)
    missing = [c for c in C.REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"pairs.csv is missing columns {missing}")
    if df["pair_id"].duplicated().any():
        raise ValueError("pairs.csv has duplicate pair_id rows")
    for col in ("row_a", "row_b"):
        df[col] = df[col].astype(np.int64)
    df["class_id"] = df["class_id"].astype(str)
    df["core_class"] = df["core_class"].astype(np.int64)
    return df


def load_component_map() -> dict[int, str]:
    payload = C.read_json(C.COMPONENT_SPLIT)
    raw = payload["row_to_component"]
    out = {int(k): str(v) for k, v in raw.items()}
    if not out:
        raise ValueError("component split has no rows")
    return out


def load_partition_map() -> dict[str, str]:
    traces = pd.read_csv(C.TRACES_CSV, usecols=["pair_id", "partition"])
    mapping = dict(zip(traces["pair_id"].astype(str), traces["partition"].astype(str)))
    if not mapping:
        raise ValueError("pair_traces.csv has no partition rows")
    return mapping


def attach_structure(pairs: pd.DataFrame) -> pd.DataFrame:
    comp = load_component_map()
    part = load_partition_map()
    df = pairs.copy()
    missing_rows = [int(r) for r in df["row_a"].tolist() + df["row_b"].tolist() if int(r) not in comp]
    if missing_rows:
        raise KeyError(f"rows missing from the component split: {missing_rows[:5]}")
    df["component_a"] = df["row_a"].map(lambda r: comp[int(r)])
    df["component_b"] = df["row_b"].map(lambda r: comp[int(r)])
    if not (df["component_a"] == df["component_b"]).all():
        bad = df.loc[df["component_a"] != df["component_b"], "pair_id"].head(5).tolist()
        raise ValueError(f"pairs whose two molecules fall in different components: {bad}")
    df["component"] = df["component_a"]
    missing_pairs = [p for p in df["pair_id"].astype(str) if p not in part]
    if missing_pairs:
        raise KeyError(f"pairs missing a partition: {missing_pairs[:5]}")
    df["partition"] = df["pair_id"].astype(str).map(part)
    return df.drop(columns=["component_a", "component_b"])


def eligible_pairs(structured: pd.DataFrame) -> pd.DataFrame:
    df = structured[structured["partition"].isin(C.PARTITIONS)].copy()
    if df.empty:
        raise ValueError("no Diagnose/Audit pairs")
    # one pair per (family, core class, component) is the unit; keep every pair
    # so the unit mean is taken later (never drop repeated edits).
    return df.reset_index(drop=True)


def build_units(df: pd.DataFrame) -> list[dict[str, Any]]:
    units = []
    keys = ["class_id", "core_class", "component"]
    for (class_id, core_class, component), grp in df.groupby(keys, sort=True):
        units.append(
            {
                "unit_id": f"{class_id}|{int(core_class)}|{component}",
                "class_id": str(class_id),
                "core_class": int(core_class),
                "component": str(component),
                "pair_ids": [str(p) for p in grp["pair_id"].tolist()],
                "n_pairs": int(len(grp)),
                "partitions": sorted(set(grp["partition"].tolist())),
                "rows": sorted({int(r) for r in grp["row_a"].tolist() + grp["row_b"].tolist()}),
            }
        )
    units.sort(key=lambda u: (u["class_id"], u["core_class"], u["component"]))
    return units


def component_molecule_sets(df: pd.DataFrame) -> dict[str, set[int]]:
    out: dict[str, set[int]] = defaultdict(set)
    for row in df.itertuples(index=False):
        out[str(row.component)].update({int(row.row_a), int(row.row_b)})
    return {k: set(v) for k, v in out.items()}


def component_overlaps(df: pd.DataFrame) -> list[dict[str, Any]]:
    comps = component_molecule_sets(df)
    keys = sorted(comps)
    out = []
    for a, b in itertools.combinations(keys, 2):
        shared = comps[a] & comps[b]
        if shared:
            out.append({"component_a": a, "component_b": b, "n_shared_molecules": len(shared)})
    return out


def build_comparisons(units: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_family: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for u in units:
        by_family[str(u["class_id"])].append(u)
    out = []
    for class_id in sorted(by_family):
        fam = sorted(by_family[class_id], key=lambda u: (int(u["core_class"]), str(u["component"])))
        for p, q in itertools.combinations(fam, 2):
            if str(p["component"]) == str(q["component"]):
                continue
            if int(p["core_class"]) == int(q["core_class"]):
                continue
            first, second = (p, q) if (p["component"], p["core_class"]) <= (
                q["component"],
                q["core_class"],
            ) else (q, p)
            out.append(
                {
                    "comparison_id": f"{class_id}||{first['unit_id']}||{second['unit_id']}",
                    "class_id": class_id,
                    "core_class_p": int(first["core_class"]),
                    "core_class_q": int(second["core_class"]),
                    "component_p": str(first["component"]),
                    "component_q": str(second["component"]),
                    "unit_p": str(first["unit_id"]),
                    "unit_q": str(second["unit_id"]),
                    "pair_p": first["pair_ids"][0],
                    "pair_q": second["pair_ids"][0],
                }
            )
    out.sort(key=lambda c: (c["class_id"], c["component_p"], c["component_q"], c["core_class_p"], c["core_class_q"]))
    return out


def greedy_non_overlapping(comparisons: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Deterministic label-free greedy: a component is used at most once."""
    used: set[str] = set()
    kept: list[Mapping[str, Any]] = []
    for comp in comparisons:
        cp, cq = str(comp["component_p"]), str(comp["component_q"])
        if cp in used or cq in used:
            continue
        used.update({cp, cq})
        kept.append(comp)
    return {
        "n_kept": int(len(kept)),
        "n_components_used": int(len(used)),
        "components_used": sorted(used),
        "comparison_ids": [str(c["comparison_id"]) for c in kept],
        "rule": "sort by (class_id, component_p, component_q, core_class_p, core_class_q); accept a comparison only if neither component was used before",
    }


def support_gates(counts: Mapping[str, Any]) -> dict[str, Any]:
    """Preset structural-support gates (label-free)."""
    gates = {
        "min_families": {
            "required": int(C.SUPPORT_GATES["min_families"]),
            "observed": int(counts["n_families_with_comparisons"]),
        },
        "min_components": {
            "required": int(C.SUPPORT_GATES["min_components"]),
            "observed": int(counts["n_components_in_comparisons"]),
        },
        "min_comparisons": {
            "required": int(C.SUPPORT_GATES["min_comparisons"]),
            "observed": int(counts["n_comparisons"]),
        },
    }
    for g in gates.values():
        g["pass"] = bool(g["observed"] >= g["required"])
    return gates


def family_structure(units: Sequence[Mapping[str, Any]], comparisons: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    per_comp = {str(c["comparison_id"]): c for c in comparisons}
    rows = []
    by_family: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for u in units:
        by_family[str(u["class_id"])].append(u)
    comp_by_family: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for c in comparisons:
        comp_by_family[str(c["class_id"])].append(c)
    for class_id in sorted(by_family):
        fam = by_family[class_id]
        comps = comp_by_family.get(class_id, [])
        rows.append(
            {
                "class_id": class_id,
                "n_units": int(len(fam)),
                "n_pairs": int(sum(u["n_pairs"] for u in fam)),
                "n_components": int(len({str(u["component"]) for u in fam})),
                "n_cores": int(len({int(u["core_class"]) for u in fam})),
                "n_comparisons": int(len(comps)),
                "comparison_components": int(
                    len({str(c["component_p"]) for c in comps} | {str(c["component_q"]) for c in comps})
                ),
                "comparison_cores": int(
                    len({int(c["core_class_p"]) for c in comps} | {int(c["core_class_q"]) for c in comps})
                ),
                "partitions": sorted({p for u in fam for p in u["partitions"]}),
            }
        )
    assert len(per_comp) == len(comparisons)
    return pd.DataFrame(rows).sort_values(["n_comparisons", "class_id"], ascending=[False, True]).reset_index(drop=True)


def eligibility(pairs: pd.DataFrame) -> dict[str, Any]:
    structured = attach_structure(pairs)
    df = eligible_pairs(structured)
    units = build_units(df)
    comparisons = build_comparisons(units)
    overlaps = component_overlaps(df)
    if overlaps:
        raise ValueError(f"components share molecules; refusing comparisons: {overlaps[:3]}")
    greedy = greedy_non_overlapping(comparisons)
    fam = family_structure(units, comparisons)
    n_comp_in_cmp = len({c["component_p"] for c in comparisons} | {c["component_q"] for c in comparisons})
    n_core_in_cmp = len({c["core_class_p"] for c in comparisons} | {c["core_class_q"] for c in comparisons})
    counts = {
        "n_pairs_total": int(len(structured)),
        "n_pairs_diagnose_audit": int(len(df)),
        "n_families": int(df["class_id"].nunique()),
        "n_components": int(df["component"].nunique()),
        "n_cores": int(df["core_class"].nunique()),
        "n_units": int(len(units)),
        "n_units_with_multiple_pairs": int(sum(1 for u in units if u["n_pairs"] > 1)),
        "n_comparisons": int(len(comparisons)),
        "n_components_in_comparisons": int(n_comp_in_cmp),
        "n_cores_in_comparisons": int(n_core_in_cmp),
        "n_families_with_comparisons": int((fam["n_comparisons"] > 0).sum()),
        "n_greedy_nonoverlapping_comparisons": int(greedy["n_kept"]),
        "n_greedy_components_used": int(greedy["n_components_used"]),
    }
    gates = support_gates(counts)
    status = "OK" if all(g["pass"] for g in gates.values()) else "INSUFFICIENT_CONTEXT_SUPPORT"
    dist = fam["n_components"].value_counts().sort_index()
    return {
        "schema": "lcrc-edit-context-eligibility-v1",
        "round_id": C.ROUND_ID,
        "result_label": C.RESULT_LABEL,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "environment": C.environment(),
        "label_blind": True,
        "labels_read_in_this_stage": False,
        "inputs": C.input_manifest(),
        "rule": {
            "unit": "(class_id, core_class, component); repeated edits inside a unit are averaged before comparisons",
            "comparison": "two units with identical class_id, different component and different core_class; both from the historical Diagnose/Audit partitions",
            "excluded": "Fit/Select partitions never contribute evaluation samples",
            "molecule_disjointness": "checked globally: no two components share an original molecule",
        },
        "counts": counts,
        "gates": gates,
        "support_gates_preset": C.SUPPORT_GATES,
        "family_structure": fam.to_dict(orient="records"),
        "family_component_histogram": {str(int(k)): int(v) for k, v in dist.items()},
        "family_core_histogram": {
            str(int(k)): int(v) for k, v in fam["n_cores"].value_counts().sort_index().items()
        },
        "greedy_non_overlapping": {
            "n_kept": greedy["n_kept"],
            "n_components_used": greedy["n_components_used"],
            "rule": greedy["rule"],
            "comparison_ids": greedy["comparison_ids"],
        },
        "partition_counts": {k: int(v) for k, v in df["partition"].value_counts().sort_index().items()},
        "status": status,
    }


def units_frame(units: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "unit_id": u["unit_id"],
                "class_id": u["class_id"],
                "core_class": int(u["core_class"]),
                "component": u["component"],
                "n_pairs_in_unit": int(u["n_pairs"]),
                "pair_ids": ";".join(u["pair_ids"]),
                "partitions": ";".join(u["partitions"]),
            }
            for u in units
        ]
    )


__all__ = [
    "attach_structure",
    "build_comparisons",
    "build_units",
    "component_overlaps",
    "eligibility",
    "eligible_pairs",
    "family_structure",
    "greedy_non_overlapping",
    "support_gates",
    "load_component_map",
    "load_pairs",
    "load_partition_map",
    "units_frame",
]
