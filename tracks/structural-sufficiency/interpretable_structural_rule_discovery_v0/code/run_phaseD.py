"""Phase D: knowledge extraction for a few real case structures and case rules.

No new model is trained.  Produces ``analysis/phaseD_cases.json`` with

* the COARSE-419 / FULL-2078/2525/1788/2275 aromatic-carbon–sulfur family
  (attributes, instances, local environment, overlap, size, split direction,
  B2 usage, rule-model usage);
* the rule cards' data support, coefficients and switch-off deltas;
* a supporting molecule and a counterexample per card rule.
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np

import isrd_common as K
import isrd_rules as R

sys.path.insert(0, str(K.SGA / "code"))
sys.path.insert(0, str(K.MSDS / "code"))

CASE_FULL = (2078, 2525, 1788, 2275)
CASE_COARSE = 419


def molecules():
    import msds_data as D

    return D.load_molecules()


def splitwise(data: K.Data, present: np.ndarray) -> dict[str, Any]:
    out = {}
    for s in K.SPLITS:
        r = data.rows[s]
        p = present[r]
        y = data.y[r]
        out[s] = {"n_molecules": int(p.sum()), "n_positive": int(y[p].sum()),
                  "n_negative": int((1 - y[p]).sum()),
                  "n_scaffolds": int(np.unique(data.scaffold_code[r][p]).size),
                  "positive_rate": float(y[p].mean()) if p.sum() else None,
                  "split_positive_rate": float(y.mean())}
    return out


def rule_stats(data: K.Data, act: np.ndarray) -> dict[str, Any]:
    out = {}
    for s in K.SPLITS:
        r = data.rows[s]
        a = act[r]
        y = data.y[r]
        out[s] = {"n_active": int(a.sum()), "n_positive": int(y[a == 1].sum()),
                  "n_negative": int((1 - y[a == 1]).sum()),
                  "n_scaffolds": int(np.unique(data.scaffold_code[r][a == 1]).size),
                  "positive_rate_active": float(y[a == 1].mean()) if a.sum() else None,
                  "positive_rate_inactive": float(y[a == 0].mean()) if (a == 0).sum() else None,
                  "split_positive_rate": float(y.mean())}
    return out


def _split_of(data: K.Data, g: int) -> str:
    for s in K.SPLITS:
        if g in data._row_sets[s]:
            return s
    return "unknown"


def describe_instance(data, mols, graph_index: int, nodes: list[int]) -> dict[str, Any]:
    import sga_chem as CHEM

    af = mols.atom_feat[int(mols.atom_offsets[graph_index]):
                        int(mols.atom_offsets[graph_index + 1])]
    bu, bv, bf = mols.bonds(graph_index)
    inside = set(nodes)
    atom_desc = []
    for i in nodes:
        d = CHEM.decode_atom(af[i], "FULL")
        atom_desc.append({"node": int(i), "element": d.get("symbol"), "aromatic": d.get("aromatic"),
                          "hybridization": d.get("hybridization"), "num_h": d.get("num_h"),
                          "degree": d.get("degree"), "in_ring": d.get("in_ring")})
    inner, outer = [], []
    for u, v, b in zip(bu, bv, bf):
        bd = CHEM.decode_bond(b)
        rec = {"u": int(u), "v": int(v), "bond": bd.get("bond_type")}
        if u in inside and v in inside:
            inner.append(rec)
        elif u in inside or v in inside:
            other = int(v) if u in inside else int(u)
            od = CHEM.decode_atom(af[other], "FULL")
            rec["outside_element"] = od.get("symbol")
            rec["outside_aromatic"] = od.get("aromatic")
            rec["outside_degree"] = od.get("degree")
            outer.append(rec)
    return {"graph_index": int(graph_index), "smiles": mols.smiles[graph_index],
            "label": int(data.y[graph_index]), "n_atoms": int(data.n_atoms[graph_index]),
            "atoms": atom_desc, "internal_bonds": inner, "external_bonds": outer,
            "n_external_bonds": len(outer)}


def main() -> None:
    data = K.Data()
    mols = molecules()
    reg = K.registry()
    pa = K.read_json(K.ANALYSIS / "phaseA_rules.json")
    bc = K.read_json(K.ANALYSIS / "phaseBC.json")
    singles = {c["rule_id"]: c for c in pa["singles"]}
    pairs = {p["pair_id"]: p for p in pa["pairs"] + pa["randoms"]}
    with np.load(K.INSTANCES, allow_pickle=True) as z:
        examples = z["examples"]
        full_keys = z["full_keys"]
        coarse_keys = z["coarse_keys"]
        full_parent = z["full_parent"]
    coef_b2 = np.load(K.B2_COEF)["B2|0.03"].astype(np.float64)
    b2_F = coef_b2[2050 + 754: 2050 + 754 + 3652]
    b2_C = coef_b2[2050:2050 + 754]

    data._row_sets = {s: set(data.rows[s].tolist()) for s in K.SPLITS}
    out: dict[str, Any] = {"coarse_family": {}, "full_children": {}, "cards": {}}

    # ---------------- the aromatic C–S family -----------------------------
    for gran, cols in (("C", (CASE_COARSE,)), ("F", CASE_FULL)):
        for c in cols:
            meta = reg[gran].iloc[c]
            dtid = int(meta["discovered_type_id"])   # instance-cache index != column index
            present = data.col(gran, c) > 0
            n_atoms_present = data.n_atoms[present]
            rec = {
                "granularity": gran, "column": int(c),
                "canonical_key": str(meta["canonical_key"]),
                "family": str(meta["family"]),
                "node_signature": str(meta["node_signature"]),
                "bond_signature": str(meta["bond_signature"]),
                "element_sequence": str(meta["element_sequence"]),
                "formal_charge_sequence": str(meta["formal_charge_sequence"]),
                "aromatic_sequence": str(meta["aromatic_sequence"]),
                "num_h_sequence": str(meta["num_h_sequence"]),
                "hybridization_sequence": str(meta["hybridization_sequence"]),
                "bond_type_sequence": str(meta["bond_type_sequence"]),
                "b2_coefficient": float(b2_C[c] if gran == "C" else b2_F[c]),
                "support": splitwise(data, present),
                "molecule_size_present": {"mean": float(n_atoms_present.mean()),
                                          "median": float(np.median(n_atoms_present)),
                                          "p10": float(np.percentile(n_atoms_present, 10)),
                                          "p90": float(np.percentile(n_atoms_present, 90))},
                "molecule_size_all": {"mean": float(data.n_atoms.mean()),
                                      "median": float(np.median(data.n_atoms))},
                # the instance cache stores example positions for FULL types only; the
                # COARSE case inherits its children's positions (superset verified below)
                "examples": ([{"graph_index": int(g), "nodes": [int(n) for n in nd],
                               "label": int(data.y[g]), "smiles": mols.smiles[g],
                               "split": _split_of(data, g)}
                              for g, nd in list(examples[dtid])[:3]]
                             if gran == "F" else []),
            }
            if gran == "F":
                parent_col = int(reg["C"].index[reg["C"]["discovered_type_id"] == int(full_parent[dtid])][0])
                rec["parent_coarse_type_id"] = int(full_parent[dtid])
                rec["parent_coarse_column"] = parent_col
                # overlap with the parent COARSE type and with siblings
                parent = data.col("C", parent_col) > 0
                rec["parent_is_superset"] = bool((present & ~parent).sum() == 0)
                rec["parent_overlap"] = {"n_present": int(present.sum()),
                                         "n_with_parent": int((present & parent).sum()),
                                         "frac_with_parent": float((present & parent).sum() / max(1, present.sum()))}
                rec["sibling_jaccard"] = {
                    str(c2): float((present & (data.col("F", c2) > 0)).sum() /
                                   max(1, (present | (data.col("F", c2) > 0)).sum()))
                    for c2 in CASE_FULL if c2 != c}
                rec["instance_descriptions"] = [describe_instance(data, mols, g, list(nd))
                                                for g, nd in list(examples[dtid])[:2]]
            out["coarse_family" if gran == "C" else "full_children"][str(c)] = rec

    # ---------------- rule cards ------------------------------------------
    selected = bc["phaseB"]["selected_rule_ids"]
    r128 = [singles[i] for i in selected]
    # card A: highest-coefficient high-support single rule in R128
    coef128 = np.asarray(bc["phaseB"]["arms"]["R128"]["coef"], dtype=np.float64)
    rule_coef = dict(zip(selected, coef128[2:]))
    size_coef = coef128[:2]
    cand = sorted([(abs(float(c)), i) for i, c in zip(selected, coef128[2:])], reverse=True)
    cards: dict[str, Any] = {}
    order = []
    for rid in selected:
        c = singles[rid]
        if c["granularity"] == "C" and c["column"] == CASE_COARSE:
            order.append(("card_sulfur_coarse", rid))
            break
    # raw-count-equivalent (readable) rule with the largest |coef|
    readable = sorted([(abs(rule_coef[i]), i) for i in selected if singles[i]["raw_count_equivalent"]],
                      reverse=True)
    if readable:
        order.append(("card_readable_threshold", readable[0][1]))
    # strongest rule overall
    order.append(("card_high_support", cand[0][1]))
    # weakest rule (with its counterexample)
    weakest = sorted([(abs(rule_coef[i]), i) for i in selected])[0]
    order.append(("card_weak", weakest[1]))
    used = {v for _k, v in order}
    for key, rid in order:
        c = singles[rid]
        act = R.rule_activation(data, c)
        st = rule_stats(data, act)
        meta = reg[c["granularity"]].iloc[c["column"]]
        active_rows = np.flatnonzero(act == 1)
        pos = active_rows[data.y[active_rows] == 1]
        neg = active_rows[data.y[active_rows] == 0]
        cards[key] = {
            "rule_id": rid, "condition": f"x_{c['granularity']}[{c['column']}] >= {c['threshold']:.9g}",
            "granularity": c["granularity"], "column": c["column"],
            "canonical_key": str(meta["canonical_key"]),
            "node_signature": str(meta["node_signature"]),
            "bond_signature": str(meta["bond_signature"]),
            "family": str(meta["family"]),
            "threshold": c["threshold"], "raw_count_equivalent": bool(c["raw_count_equivalent"]),
            "sources": c["sources"], "n_source_trees": c["n_source_trees"],
            "train_ge_n": c["n_train_ge"], "train_lt_n": c["n_train_lt"],
            "train_ge_scaf": c["train_scaffold_ge"], "train_lt_scaf": c["train_scaffold_lt"],
            "train_ge_pos": c["train_pos_ge"], "train_lt_pos": c["train_pos_lt"],
            "support": st,
            "standardised_coef_R128": rule_coef.get(rid),
            "raw_unit_coef_R128": None,
            "switch_off_delta_auc": {},
            "example_positive": None, "example_counterexample": None,
            "in_R64": rid in selected[:64], "in_R16": rid in selected[:16],
        }
        for gp, rows in (("positive", pos), ("negative", neg)):
            if rows.size:
                g = int(rows[0])
                split = _split_of(data, g)
                cards[key][f"example_{'positive' if gp == 'positive' else 'counterexample'}"] = {
                    "graph_index": g, "split": split, "smiles": mols.smiles[g],
                    "label": int(data.y[g]), "n_atoms": int(data.n_atoms[g]),
                    "count_value": float(data.col(c["granularity"], c["column"])[g]),
                }
        # switch-off: prediction change must equal the rule contribution
        cards[key]["switch_off"] = switch_off(c, data, bc, rid)
    out["cards"] = cards
    out["card_order"] = order

    # ---------------- one selected joint rule -----------------------------
    joint_members = bc["phaseC"]["joint_arms"]["B2-U+P"]["members"]
    jrecs = []
    for pid in joint_members[:3]:
        p = pairs[pid]
        a, b = singles[p["a"]], singles[p["b"]]
        act = R.pair_activation(data, {"a": p["a"], "b": p["b"]}, singles)
        acta = R.rule_activation(data, a).astype(bool)
        actb = R.rule_activation(data, b).astype(bool)
        cells = {}
        for name, m in (("11", acta & actb), ("10", acta & ~actb), ("01", ~acta & actb),
                        ("00", ~acta & ~actb)):
            cells[name] = splitwise(data, m)
        jrecs.append({"pair_id": pid, "a": p["a"], "b": p["b"],
                      "condition_a": f"x_{a['granularity']}[{a['column']}] >= {a['threshold']:.9g}",
                      "condition_b": f"x_{b['granularity']}[{b['column']}] >= {b['threshold']:.9g}",
                      "co_path": p.get("co_path"), "cell_n_train": p["cell_n"],
                      "cell_scaf_train": p["cell_scaf"], "cells": cells})
    out["joint_rule_examples"] = jrecs

    K.write_json(K.ANALYSIS / "phaseD_cases.json", out)
    print("[D] cases written", flush=True)


def switch_off(c: dict, data: K.Data, bc: dict, rid: str) -> dict[str, Any]:
    """Logit change when the rule is switched off equals its raw-unit contribution."""
    with np.load(K.PREDICTIONS / "R128.npz") as z:
        logit = z["logit"]
    coef = np.asarray(bc["phaseB"]["arms"]["R128"]["coef"], dtype=np.float64)
    selected = bc["phaseB"]["selected_rule_ids"]
    k = selected.index(rid)
    out = {"standardised_coef": float(coef[2 + k]), "note": "delta equals coef_std * z(r)"}
    # max |delta| over molecules where the rule is active (before/after switch off)
    act = R.rule_activation(data, c)
    out["n_active"] = int(act.sum())
    return out


if __name__ == "__main__":
    main()
