"""Contrasts, metrics table and complexity census for ISRD-v0.

Reads the stored per-molecule logits of every arm of this round plus the frozen
reference models (B0, B2, X0, X-C, X-F, X-CF), recomputes every metric
independently, and runs the paired scaffold bootstrap (1 000 resamples, seed 0).

Outputs
    analysis/metrics.csv     every arm x split (ROC-AUC, AP, OGB cross-check)
    analysis/contrasts.json  paired bootstrap for the core contrasts
    analysis/complexity.json rule counts, structures, active-rules-per-molecule,
                             contribution distributions
    analysis/contributions.npz  per-molecule size / rule / joint logit terms
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

import isrd_common as K
import isrd_rules as R

REFERENCE = {
    "B0": (K.MSDS / "data" / "phaseB_predictions.npz", "B0|0.03|{}"),
    "B2": (K.MSDS / "data" / "phaseB_predictions.npz", "B2|0.03|{}"),
    "B3": (K.MSDS / "data" / "phaseB_predictions.npz", "B3|0.03|{}"),
    "X-C": (K.SGA / "analysis" / "predictions" / "X-C.npz", "{}"),
    "X-F": (K.SGA / "analysis" / "predictions" / "X-F.npz", "{}"),
    "X-CF": (K.SGA / "analysis" / "predictions" / "X-CF.npz", "{}"),
}


def load_reference(name: str, data: K.Data) -> np.ndarray:
    path, pattern = REFERENCE[name]
    logit = np.full(data.y.size, np.nan)
    with np.load(path) as z:
        if "train" in z.files:
            for k in K.SPLITS:
                logit[data.rows[k]] = z[pattern.format(k)].astype(np.float64)
        else:  # X0 has no stored Train logits
            for k in ("valid", "test"):
                logit[data.rows[k]] = z[pattern.format(k)].astype(np.float64)
    return logit


def main() -> None:
    data = K.Data()
    bc = K.read_json(K.ANALYSIS / "phaseBC.json")
    phaseA = K.read_json(K.ANALYSIS / "phaseA_rules.json")
    singles = {c["rule_id"]: c for c in phaseA["singles"]}
    arrays = np.load(K.ANALYSIS / "phaseBC_arrays.npz")

    logits: dict[str, np.ndarray] = {}
    for name in ("R0", "R16", "R64", "R128", "B2-L64", "B2-U64", "B2-U+S", "B2-U+P",
                 "B2-U+N", "R64+P"):
        with np.load(K.PREDICTIONS / f"{name.replace('+', '_')}.npz") as z:
            logits[name] = z["logit"].astype(np.float64)
    for ref in REFERENCE:
        logits[ref] = load_reference(ref, data)

    # ---------------- metrics.csv ----------------------------------------
    rows = []
    for name, logit in logits.items():
        for split in K.SPLITS:
            r = data.rows[split]
            if not np.all(np.isfinite(logit[r])):
                continue
            m = K.metrics(logit[r], data.y[r], ogb=True)
            rows.append({"arm": name, "split": split, **m,
                         "evidence_label": "DEVELOPMENT_TEST" if split == "test" else
                         ("TRAIN" if split == "train" else "VALID"),
                         "source": "this_round" if name in bc["phaseB"]["arms"] or
                         name in bc["phaseC"]["arms"] or name in bc["phaseC"]["joint_arms"]
                         else "historical_reference"})
    import pandas as pd

    df = pd.DataFrame(rows)
    df.to_csv(K.ANALYSIS / "metrics.csv", index=False)
    print(df.pivot_table(index="arm", columns="split", values="roc_auc").round(4).to_string(), flush=True)

    # ---------------- contrasts ------------------------------------------
    contrasts = [
        ("Q1_capacity", "R16", "R0"), ("Q1_capacity", "R64", "R16"),
        ("Q1_capacity", "R128", "R64"), ("Q1_capacity", "R128", "R0"),
        ("Q1_gap", "B2", "R128"), ("Q1_gap", "X-C", "R128"),
        ("Q1_gap", "X-F", "R128"), ("Q1_gap", "X-CF", "R128"),
        ("Q2_threshold", "B2-U64", "B2"), ("Q2_threshold", "B2-L64", "B2"),
        ("Q2_threshold", "B2-U64", "B2-L64"),
        ("Q3_joint", "B2-U+P", "B2-U64"), ("Q3_joint", "B2-U+P", "B2-U+S"),
        ("Q3_joint", "B2-U+P", "B2-U+N"), ("Q3_joint", "B2-U+S", "B2-U64"),
        ("Q3_joint", "B2-U+N", "B2-U64"),
        ("Q3_transparent", "R64+P", "R64"),
    ]
    out: dict[str, Any] = {"round_id": K.ROUND_ID, "seed": K.SEED, "n_boot": 1000,
                           "contrast_sign": "delta = arm_a - arm_b (positive = a better)",
                           "contrasts": []}
    for group, a, b in contrasts:
        if a not in logits or b not in logits:
            continue
        entry = {"group": group, "a": a, "b": b, "splits": {}}
        for split in K.SPLITS:
            r = data.rows[split]
            if not (np.all(np.isfinite(logits[a][r])) and np.all(np.isfinite(logits[b][r]))):
                continue
            ma = K.metrics(logits[a][r], data.y[r])
            mb = K.metrics(logits[b][r], data.y[r])
            entry["splits"][split] = {
                "auc_a": ma["roc_auc"], "auc_b": mb["roc_auc"],
                "ap_a": ma["average_precision"], "ap_b": mb["average_precision"],
                "delta_auc": ma["roc_auc"] - mb["roc_auc"],
                "delta_ap": ma["average_precision"] - mb["average_precision"],
                **K.scaffold_bootstrap(r, data.y, logits[a], logits[b], data.scaffold_code,
                                       n_boot=1000 if split != "train" else 200),
            }
        out["contrasts"].append(entry)
    K.write_json(K.ANALYSIS / "contrasts.json", out)

    # ---------------- complexity census ----------------------------------
    comp: dict[str, Any] = {"arms": {}, "joint_rule_conditions": {}}
    r64 = [singles[i] for i in arrays["r64_ids"].tolist()]
    u64_act = arrays["u64_act"]
    for cap in (16, 64, 128):
        name = f"R{cap}"
        ids = bc["phaseB"]["selected_rule_ids"][:cap]
        acts = np.column_stack([R.rule_activation(data, singles[i]) for i in ids])
        comp["arms"][name] = {
            "n_rules": len(ids), "n_distinct_structures": len({(singles[i]["granularity"],
                                                               singles[i]["column"]) for i in ids}),
            "n_coarse": sum(1 for i in ids if singles[i]["granularity"] == "C"),
            "n_full": sum(1 for i in ids if singles[i]["granularity"] == "F"),
            "conditions_per_rule": 1,
            "active_rules_per_molecule": {
                split: K.active_counts(acts[data.rows[split]]) for split in K.SPLITS},
            "n_raw_count_equivalent_rules": sum(1 for i in ids if singles[i]["raw_count_equivalent"]),
        }
    comp["joint_rule_conditions"] = {"n_conditions_per_rule": 2}
    for name in ("B2-U+P", "B2-U+N"):
        members = bc["phaseC"]["joint_arms"][name]["members"]
        act = np.column_stack([u64_act[:, list(range(64))].sum(axis=1),
                               np.zeros(data.y.size)]) if False else None
        comp["arms"][name] = {"n_rules": len(members), "conditions_per_rule": 2,
                              "structure_pairs": members}
    # contribution decomposition: size / single-rule / joint terms
    contribs = contribution_decomposition(data, bc, singles, arrays)
    K.write_json(K.ANALYSIS / "complexity.json", {**comp, "contributions": contribs["summary"]})
    np.savez_compressed(K.ANALYSIS / "contributions.npz", **contribs["arrays"])
    print("[contrasts] written", flush=True)


def contribution_decomposition(data: K.Data, bc: dict, singles: dict, arrays) -> dict:
    """Per-molecule logit split into size / single-rule / joint terms and intercept."""
    out_arrays: dict[str, np.ndarray] = {}
    summary: dict[str, Any] = {}
    size_z = arrays["size_z"]
    rule_z = arrays["rule_z"]
    for name in ("R0", "R16", "R64", "R128"):
        arm = bc["phaseB"]["arms"][name]
        coef = np.asarray(arm["coef"], dtype=np.float64)
        n_rules = coef.size - 2
        size_term = size_z @ coef[:2]
        rule_term = rule_z[:, :n_rules] @ coef[2:] if n_rules else np.zeros(data.y.size)
        out_arrays[f"{name}_size"] = size_term
        out_arrays[f"{name}_rules"] = rule_term
        out_arrays[f"{name}_intercept"] = np.full(data.y.size, arm["intercept"])
        summary[name] = {"size_term": dist(size_term, data), "rule_term": dist(rule_term, data)}
    # B2-augmented arms: correction-only terms
    off = None
    with np.load(K.PREDICTIONS / "B2-U64.npz") as z:
        pass
    with np.load(K.MSDS / "data" / "phaseB_predictions.npz") as z:
        b2 = np.zeros(data.y.size)
        for k in K.SPLITS:
            b2[data.rows[k]] = z[f"B2|0.03|{k}"].astype(np.float64)
    out_arrays["B2_logit"] = b2
    tr = data.rows["train"]
    def zs(M: np.ndarray) -> np.ndarray:
        mu, sd = M[tr].mean(0), M[tr].std(0)
        return (M - mu) / np.where(sd > 0, sd, 1.0)

    u64_z = zs(arrays["u64_act"])
    l64_z = zs(arrays["l64_vals"])
    design = {"B2-L64": l64_z, "B2-U64": u64_z}
    for name in ("B2-U+S", "B2-U+P", "B2-U+N"):
        members = bc["phaseC"]["joint_arms"][name]["members"]
        if name == "B2-U+S":
            extra = np.column_stack([R.rule_activation(data, singles[m]) for m in members])
        else:
            extra = np.column_stack([R.pair_activation(data, {"a": p["a"], "b": p["b"]}, singles,
                                                       None) for p in
                                     [next(q for q in _pairs(data) if q["pair_id"] == m)
                                      for m in members]])
        design[name] = np.column_stack([u64_z, zs(extra)])
    for name, X in design.items():
        arm = bc["phaseC"]["joint_arms"].get(name) or bc["phaseC"]["arms"][name]
        coef = np.asarray(arm["coef"], dtype=np.float64)
        term = X @ coef
        out_arrays[f"{name}_correction"] = term
        out_arrays[f"{name}_intercept"] = np.full(data.y.size, arm["intercept"])
        summary[name] = {"correction_term": dist(term, data),
                         "n_parameters": int(coef.size)}
    return {"arrays": out_arrays, "summary": summary}


_PAIR_CACHE: dict[str, Any] = {}


def _pairs(data: K.Data) -> list[dict]:
    if "pairs" not in _PAIR_CACHE:
        phaseA = K.read_json(K.ANALYSIS / "phaseA_rules.json")
        _PAIR_CACHE["pairs"] = phaseA["pairs"] + phaseA["randoms"]
    return _PAIR_CACHE["pairs"]


def dist(values: np.ndarray, data: K.Data) -> dict[str, Any]:
    tr = data.rows["train"]
    return {"mean": float(values.mean()), "std": float(values.std()),
            "p05": float(np.percentile(values, 5)), "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "train_mean": float(values[tr].mean()), "train_std": float(values[tr].std()),
            "abs_mean": float(np.abs(values).mean())}


if __name__ == "__main__":
    main()
