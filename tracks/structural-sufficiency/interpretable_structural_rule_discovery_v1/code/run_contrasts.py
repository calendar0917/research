"""Contrasts, metrics table, complexity census and contribution decomposition (ISRD-v1).

Reads the stored per-molecule logits of this round plus every frozen reference
(B0/B2/B3 from MSDS-v0, X-C/X-F/X-CF from SGA-v1, all v0 rule arms now marked
``SUPERSEDED_BY_RULE_MAPPING_FIX``), recomputes every metric independently and runs
the paired scaffold bootstrap (1 000 resamples, seed 0).

Outputs: metrics.csv, contrasts.json, complexity.json, contributions.npz,
         v0_v1_rule_identity.json
"""

from __future__ import annotations

from typing import Any

import numpy as np

import isrd1_common as K
import isrd1_rules as R

REFERENCE = {
    "B0": (K.B2_PRED, "B0|0.03|{}"),
    "B2": (K.B2_PRED, "B2|0.03|{}"),
    "B3": (K.B2_PRED, "B3|0.03|{}"),
    "X-C": (K.SGA / "analysis" / "predictions" / "X-C.npz", "{}"),
    "X-F": (K.SGA / "analysis" / "predictions" / "X-F.npz", "{}"),
    "X-CF": (K.SGA / "analysis" / "predictions" / "X-CF.npz", "{}"),
}
V0_ARMS = ("R0", "R16", "R64", "R128", "B2-L64", "B2-U64", "B2-U+S", "B2-U+P",
           "B2-U+N", "R64+P")
V1_ARMS = ("R0", "R16", "R64", "R128", "B2-L64", "B2-U64", "B2-U+S", "B2-U+P",
           "B2-U+N", "R64+P", "COUNT128", "COUNT-ALL", "XGB-ALL")


def load_reference(name: str, data: K.Data) -> np.ndarray:
    path, pattern = REFERENCE[name]
    logit = np.full(data.y.size, np.nan)
    with np.load(path) as z:
        for k in K.SPLITS:
            logit[data.rows[k]] = z[pattern.format(k)].astype(np.float64)
    return logit


def load_v0(name: str, data: K.Data) -> np.ndarray:
    path = K.V0 / "analysis" / "predictions" / f"{name.replace('+', '_')}.npz"
    logit = np.full(data.y.size, np.nan)
    with np.load(path) as z:
        logit[:] = z["logit"].astype(np.float64)
    return logit


def main() -> None:
    import pandas as pd

    data = K.Data()
    bc = K.read_json(K.ANALYSIS / "phaseBC.json")
    pc = K.read_json(K.ANALYSIS / "phaseC.json")
    pa = K.read_json(K.ANALYSIS / "phaseA_rules.json")
    singles = {c["rule_id"]: c for c in pa["singles"]}

    logits: dict[str, np.ndarray] = {}
    source: dict[str, str] = {}
    for name in V1_ARMS:
        path = K.PREDICTIONS / f"{name.replace('+', '_')}.npz"
        if not path.exists():
            continue
        with np.load(path) as z:
            logits[name] = z["logit"].astype(np.float64)
        source[name] = "this_round"
    for name in REFERENCE:
        logits[name] = load_reference(name, data)
        source[name] = "historical_reference"
    for name in V0_ARMS:
        if name in ("R0",):
            continue
        logits[f"v0:{name}"] = load_v0(name, data)
        source[f"v0:{name}"] = "SUPERSEDED_BY_RULE_MAPPING_FIX"
    for name in ("CS-BASE", "CS-FINE"):
        path = K.PREDICTIONS / f"{name}.npz"
        if path.exists():
            with np.load(path) as z:
                logits[name] = z["logit"].astype(np.float64)
            source[name] = "this_round"

    # ---------------- metrics.csv -----------------------------------------
    rows = []
    for name, logit in logits.items():
        for split in K.SPLITS:
            r = data.rows[split]
            keep = np.isfinite(logit[r])
            if keep.sum() == 0:
                continue
            rr = r[keep]
            m = K.metrics(logit[rr], data.y[rr], ogb=(split != "train"))
            rows.append({"arm": name, "split": split, **m,
                         "n_split_total": int(r.size),
                         "scope": "FULL_SPLIT" if keep.all() else "COARSE419_SUBSET",
                         "evidence_label": "DEVELOPMENT_TEST" if split == "test" else
                         ("TRAIN" if split == "train" else "VALID"),
                         "source": source[name]})
    df = pd.DataFrame(rows)
    df.to_csv(K.ANALYSIS / "metrics.csv", index=False)
    print(df.pivot_table(index="arm", columns="split", values="roc_auc").round(4).to_string(),
          flush=True)

    # ---------------- contrasts --------------------------------------------
    contrasts = [
        ("v1_capacity", "R16", "R0"), ("v1_capacity", "R64", "R16"),
        ("v1_capacity", "R128", "R64"), ("v1_capacity", "R128", "R0"),
        ("v1_vs_v0", "R128", "v0:R128"), ("v1_vs_v0", "R64", "v0:R64"),
        ("v1_vs_v0", "R16", "v0:R16"),
        ("v1_gap", "B2", "R128"), ("v1_gap", "X-CF", "R128"),
        ("v1_threshold", "B2-U64", "B2"), ("v1_threshold", "B2-L64", "B2"),
        ("v1_threshold", "B2-U64", "B2-L64"),
        ("v1_joint", "B2-U+P", "B2-U64"), ("v1_joint", "B2-U+P", "B2-U+S"),
        ("v1_joint", "B2-U+P", "B2-U+N"), ("v1_joint", "B2-U+S", "B2-U64"),
        ("v1_joint", "B2-U+N", "B2-U64"), ("v1_joint", "R64+P", "R64"),
        ("sufficiency", "COUNT128", "R128"),
        ("sufficiency", "COUNT-ALL", "COUNT128"),
        ("sufficiency", "XGB-ALL", "COUNT-ALL"),
        ("reference", "B2", "COUNT-ALL"), ("reference", "X-CF", "XGB-ALL"),
        ("sufficiency", "XGB-ALL", "R128"), ("reference", "B2", "XGB-ALL"),
        ("conditional", "CS-FINE", "CS-BASE"),
    ]
    out: dict[str, Any] = {"round_id": K.ROUND_ID, "seed": K.SEED, "n_boot": 1000,
                           "contrast_sign": "delta = arm_a - arm_b (positive = a better)",
                           "test_policy": "DEVELOPMENT_TEST (reported, never used for selection)",
                           "contrasts": []}
    for group, a, b in contrasts:
        if a not in logits or b not in logits:
            out["contrasts"].append({"group": group, "a": a, "b": b, "status": "MISSING_ARM"})
            continue
        entry = {"group": group, "a": a, "b": b, "splits": {}}
        for split in K.SPLITS:
            r = data.rows[split]
            keep = np.isfinite(logits[a][r]) & np.isfinite(logits[b][r])
            if keep.sum() == 0:
                continue
            r = r[keep]
            entry["scope"] = "FULL_SPLIT" if keep.all() else "COARSE419_SUBSET"
            ma = K.metrics(logits[a][r], data.y[r])
            mb = K.metrics(logits[b][r], data.y[r])
            boot = ({} if split == "train" else
                    K.scaffold_bootstrap(r, data.y, logits[a], logits[b],
                                         data.scaffold_code, n_boot=1000))
            entry["splits"][split] = {
                "auc_a": ma["roc_auc"], "auc_b": mb["roc_auc"],
                "ap_a": ma["average_precision"], "ap_b": mb["average_precision"],
                "delta_auc": ma["roc_auc"] - mb["roc_auc"],
                "delta_ap": ma["average_precision"] - mb["average_precision"],
                **{k: v for k, v in boot.items() if k != "n_boot"},
            }
        out["contrasts"].append(entry)
        if "test" in entry["splits"]:
            s = entry["splits"]["test"]
            print(f"[contrast] {a} - {b}: valid {entry['splits']['valid']['delta_auc']:+.4f} "
                  f"test {s['delta_auc']:+.4f} [{s['delta_auc_ci95'][0]:+.4f}, "
                  f"{s['delta_auc_ci95'][1]:+.4f}]", flush=True)
    K.write_json(K.ANALYSIS / "contrasts.json", out)

    # ---------------- v0 vs v1 rule identity --------------------------------
    v0 = K.read_json(K.V0 / "analysis" / "phaseA_rules.json")
    v0_keys = set()
    for c in v0["singles"]:
        v0_keys.add((c["granularity"], int(c["column"]), R.f32(c["threshold"]), "ge"))
        v0_keys.add((c["granularity"], int(c["column"]), R.f32(c["threshold"]), "lt"))
    v1_keys = {(c["granularity"], int(c["column"]), R.f32(c["threshold"]), c["direction"])
               for c in pa["singles"]}
    v1_r128 = [singles[i]["rule_id"] for i in bc["phaseB"]["selected_rule_ids"][:128]]
    v1_r128_keys = [(singles[i]["granularity"], int(singles[i]["column"]),
                     R.f32(singles[i]["threshold"]), singles[i]["direction"]) for i in v1_r128]
    v0_r128 = K.read_json(K.V0 / "analysis" / "phaseBC.json")["phaseB"]["selected_rule_ids"][:128]
    v0_sel = {c["rule_id"]: c for c in v0["singles"]}
    v0_r128_keys = [(v0_sel[i]["granularity"], int(v0_sel[i]["column"]),
                     R.f32(v0_sel[i]["threshold"]), "ge") for i in v0_r128]
    # what the v0 mapping would have called the same condition, decoded correctly
    def v0_mapped(gran: str, col: int) -> tuple[str, int]:
        if gran == "C" and col < K.N_C:
            # X-F global 2258+col was read as C[col] -> really F[col]
            return ("C_OR_F", col)
        return (gran, col)

    ident = {
        "n_v1_single_pool": len(v1_keys), "n_v0_single_pool": len(v0_keys),
        "n_common_condition_identity": len(v1_keys & v0_keys),
        "n_v1_only": len(v1_keys - v0_keys), "n_v0_only": len(v0_keys - v1_keys),
        "n_v1_R128_in_v0_pool": int(sum(k in v0_keys for k in v1_r128_keys)),
        "n_v0_R128_cond_identity_in_v1_pool": int(sum(k in v1_keys for k in v0_r128_keys)),
        "n_v0_R128": len(v0_r128_keys),
        "v0_R128_granularity_counts": {g: int(sum(1 for k in v0_r128_keys if k[0] == g))
                                       for g in ("C", "F")},
        "v1_R128_granularity_counts": {g: int(sum(1 for k in v1_r128_keys if k[0] == g))
                                       for g in ("C", "F")},
        "v0_R64_structures": K.read_json(K.V0 / "analysis" / "phaseBC.json")["phaseC"]["r64_structures"],
        "v1_R64_structures": bc["phaseC"]["r64_structures"],
        "note": "v0 read X-F's structural columns with the H+C+F layout; every v0 COARSE "
                "condition with column < 754 may therefore denote C[col] (X-C) or "
                "F[col] (X-F). Only exact (granularity, column, threshold, direction) "
                "identities are counted as common.",
    }
    K.write_json(K.ANALYSIS / "v0_v1_rule_identity.json", ident)

    # ---------------- complexity census ------------------------------------
    arrays = np.load(K.ANALYSIS / "phaseBC_arrays.npz")
    comp: dict[str, Any] = {"arms": {}, "data": {}}
    for cap in (16, 64, 128):
        ids = bc["phaseB"]["selected_rule_ids"][:cap]
        acts = R.rule_matrix(data, [singles[i] for i in ids])
        comp["arms"][f"R{cap}"] = {
            "n_rules": len(ids),
            "n_distinct_structures": len({(singles[i]["granularity"], singles[i]["column"])
                                          for i in ids}),
            "n_coarse": sum(1 for i in ids if singles[i]["granularity"] == "C"),
            "n_full": sum(1 for i in ids if singles[i]["granularity"] == "F"),
            "n_direction_lt": sum(1 for i in ids if singles[i]["direction"] == "lt"),
            "n_empirically_raw_count_equivalent": sum(
                1 for i in ids if singles[i]["reading"].get("empirically_equivalent_to_count_ge")),
            "active_rules_per_molecule": {s: K.active_counts(acts[data.rows[s]]) for s in K.SPLITS},
        }
    u64 = np.load(K.ANALYSIS / "phaseBC_arrays.npz")["u64_act"]
    comp["arms"]["B2-U64"] = {"n_rules": 64, "n_distinct_structures": 64,
                              "active_rules_per_molecule": {
                                  s: K.active_counts(u64[data.rows[s]]) for s in K.SPLITS}}
    comp["arms"]["COUNT-ALL"] = {
        "n_parameters": pc["count_all"]["n_parameters"],
        "train_valid_gap_auc": pc["count_all"]["train_valid_gap_auc"],
        "local": pc["count_all_local"]}
    comp["arms"]["XGB-ALL"] = {"n_rounds": pc["xgb_all"]["n_rounds"],
                               "n_features_used": pc["xgb_all"]["n_features_used"],
                               "structural_gain_share": pc["xgb_all"]["structural_gain_share"]}
    comp["data"]["n_C"] = K.N_C
    comp["data"]["n_F"] = K.N_F
    comp["data"]["count_all_width"] = pc["input_identity"]["count_all_width"]
    K.write_json(K.ANALYSIS / "complexity.json", comp)

    # ---------------- contribution decomposition ----------------------------
    size_z = arrays["size_z"]
    rule_z = arrays["rule_z"]
    contrib_arrays: dict[str, np.ndarray] = {}
    summary: dict[str, Any] = {}

    def dist(v: np.ndarray) -> dict[str, Any]:
        trm = data.rows["train"]
        return {"mean": float(v.mean()), "std": float(v.std()),
                "p05": float(np.percentile(v, 5)), "p50": float(np.percentile(v, 50)),
                "p95": float(np.percentile(v, 95)),
                "train_mean": float(v[trm].mean()), "abs_mean": float(np.abs(v).mean())}

    for name in ("R0", "R16", "R64", "R128"):
        arm = bc["phaseB"]["arms"][name]
        coef = np.asarray(arm["coef"], dtype=np.float64)
        n_rules = coef.size - 2
        size_term = size_z @ coef[:2]
        rule_term = rule_z[:, :n_rules] @ coef[2:] if n_rules else np.zeros(data.y.size)
        contrib_arrays[f"{name}_size"] = size_term
        contrib_arrays[f"{name}_rules"] = rule_term
        summary[name] = {"size_term": dist(size_term), "rule_term": dist(rule_term)}
    b2 = np.zeros(data.y.size)
    with np.load(K.B2_PRED) as z:
        for k in K.SPLITS:
            b2[data.rows[k]] = z[f"B2|0.03|{k}"].astype(np.float64)
    contrib_arrays["B2_logit"] = b2

    def zs(M: np.ndarray) -> np.ndarray:
        trm = data.rows["train"]
        mu, sd = M[trm].mean(0), M[trm].std(0)
        return (M - mu) / np.where(sd > 0, sd, 1.0)

    design = {"B2-L64": zs(arrays["l64_vals"]), "B2-U64": zs(arrays["u64_act"])}
    for name in ("B2-U+S", "B2-U+P", "B2-U+N"):
        members = bc["phaseC"]["joint_arms"][name]["members"]
        if name == "B2-U+S":
            extra = np.column_stack([R.rule_activation(data, singles[m]) for m in members])
        else:
            extra = np.column_stack([
                R.rule_activation(data, {"granularity": k[0], "column": k[1],
                                         "threshold": k[2], "direction": k[3]})
                for k in _member_keys(name, bc)]) * np.column_stack([
                R.rule_activation(data, {"granularity": k[0], "column": k[1],
                                         "threshold": k[2], "direction": k[3]})
                for k in _member_keys_b(name, bc)])
        design[name] = np.column_stack([zs(arrays["u64_act"]), zs(extra)])
    for name, Xd in design.items():
        arm = bc["phaseC"]["joint_arms"].get(name) or bc["phaseC"]["arms"][name]
        coef = np.asarray(arm["coef"], dtype=np.float64)
        term = Xd @ coef
        contrib_arrays[f"{name}_correction"] = term
        summary[name] = {"correction_term": dist(term), "n_parameters": int(coef.size)}
    np.savez_compressed(K.ANALYSIS / "contributions.npz", **contrib_arrays)
    K.write_json(K.ANALYSIS / "contribution_summary.json", summary)
    print("[contrasts] written", flush=True)


def _member_keys(name: str, bc: dict) -> list[tuple]:
    sel = bc["phaseC"]["joint"]["selected"]
    src = sel["real_pairs"] if name == "B2-U+P" else sel["random_pairs"]
    return [_parse_condition(r["condition_a"]) for r in src]


def _member_keys_b(name: str, bc: dict) -> list[tuple]:
    sel = bc["phaseC"]["joint"]["selected"]
    src = sel["real_pairs"] if name == "B2-U+P" else sel["random_pairs"]
    return [_parse_condition(r["condition_b"]) for r in src]


def _parse_condition(text: str) -> tuple:
    head, op, val = text.split()
    gran, col = head[0], int(head[2:-1])
    return (gran, col, R.f32(float(val)), "ge" if op == ">=" else "lt")


if __name__ == "__main__":
    main()
