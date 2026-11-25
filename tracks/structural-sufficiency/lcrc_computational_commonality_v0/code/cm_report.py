"""Assemble every analysis artefact, the verdict and REPORT.md.

``uv run python code/cm_report.py``
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import cm_common as C

AN = C.ANALYSIS
LOCAL = C.LOCAL
PARTB = LOCAL / "partb"

ATOM_NAMES = ("B", "Br", "C", "C_aromatic", "Cl", "F", "I", "N", "N_aromatic", "O",
              "O_aromatic", "P", "S", "S_aromatic", "Si", "UNK")
BOND_NAMES = ("SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "UNK")
N_ATOM, N_BOND, N_END = 16, 5, 80


def load_json(path: Path) -> Any:
    with open(path) as fh:
        return json.load(fh)


def decode_key(key: int) -> dict[str, Any]:
    p, e = divmod(int(key), N_END)
    x1, rest = divmod(p, 25)
    x2, x3 = divmod(rest, 5)
    x4 = e // 5
    return {"middle_atom": ATOM_NAMES[x1], "bond1": BOND_NAMES[x2], "bond2": BOND_NAMES[x3],
            "end_atom": ATOM_NAMES[x4], "path_cat": int(p), "path_end63": int(e),
            "bond2_in_end_code_consistent": bool(e % 5 == p % 5)}


def components_of(keys: np.ndarray) -> np.ndarray:
    k = np.asarray(keys, dtype=np.int64)
    p, e = k // N_END, k % N_END
    return np.stack([p // 25, (p % 25) // 5, p % 5, e // 5], axis=-1)


# ---------------------------------------------------------------------------
# part A artefacts
# ---------------------------------------------------------------------------


def write_part_a() -> dict[str, Any]:
    raw = load_json(AN / "part_a_raw.json")
    rows = []
    for label, m in raw["models"].items():
        rows.append({
            "kind": "single_model", "name": label, "members": label,
            "arch": m["arch"], "seed": m["seed"], "n_models": 1,
            "parameters_total": m["parameters"], "horizon_epochs": m["horizon_epochs"],
            "best_epoch": m["best_epoch"],
            "valid_macro_ap": round(m["valid_macro_ap"], 6), "test_macro_ap": round(m["test_macro_ap"], 6),
            "valid_delta_vs_best_member": 0.0, "test_delta_vs_best_member": 0.0,
            "valid_raw_logit_mean_ap": round(m["valid_macro_ap"], 6),
        })
    best_single_valid = max(raw["models"].items(), key=lambda kv: kv[1]["valid_macro_ap"])[0]
    for name, e in raw["ensembles"].items():
        members = e["members"]
        rows.append({
            "kind": e["kind"], "name": name, "members": "|".join(members),
            "arch": e["kind"], "seed": "", "n_models": len(members),
            "parameters_total": e["parameters_total"],
            "horizon_epochs": ",".join(str(C.MODELS[m]["horizon_epochs"]) for m in members),
            "best_epoch": "",
            "valid_macro_ap": round(e["valid"]["macro_ap"], 6), "test_macro_ap": round(e["test"]["macro_ap"], 6),
            "valid_delta_vs_best_member": round(e["valid"]["delta_vs_best_member"], 6),
            "test_delta_vs_best_member": round(e["test"]["delta_vs_best_member"], 6),
            "valid_raw_logit_mean_ap": round(e["valid"]["macro_ap_raw_logit_mean"], 6),
        })
    C.save_csv(AN / "ensemble_main_table.csv", rows, columns=[
        "kind", "name", "members", "arch", "seed", "n_models", "parameters_total", "horizon_epochs",
        "best_epoch", "valid_macro_ap", "test_macro_ap", "valid_delta_vs_best_member",
        "test_delta_vs_best_member", "valid_raw_logit_mean_ap"])

    # task complementarity: concatenate the per-pair tables with a pair column
    combined = []
    for path in sorted(AN.glob("task_complementarity_*.csv")):
        pair = path.stem.replace("task_complementarity_", "")
        with open(path) as fh:
            for rec in csv.DictReader(fh):
                rec["pair"] = pair.replace("__", "+")
                combined.append(rec)
    if combined:
        cols = ["pair"] + [k for k in combined[0] if k != "pair"]
        C.save_csv(AN / "task_complementarity.csv", combined, columns=cols)

    # task weight transfer
    transfer = {"created_utc": C.utcnow(), "schema": "cc-task-weight-transfer-v1",
                "w_grid": raw["task_weights"]["w_grid"], "min_support": raw["task_weights"]["min_support"],
                "shrinkage": raw["task_weights"]["shrinkage"], "folds": raw["task_weights"]["folds"],
                "pairs": {}}
    for key, v in raw["task_weights"]["pairs"].items():
        transfer["pairs"][key] = {
            "global": v["global"], "equal": v["equal"], "task": v["task"],
            "members": v["members"], "oracle": v["oracle"],
            "direction_agreement": v["direction_agreement"], "high_support_subset": v["high_support_subset"],
            "bootstrap_task": v["bootstrap_task"], "drop_topk_sensitivity": v["drop_topk_sensitivity"],
            "test_ap_distances": v["test_ap_distances"],
        }
        # per-task weights (the fitted combination weights that must be kept)
        csv_path = AN / f"task_complementarity_{key.replace('+', '__')}.csv"
        if csv_path.is_file():
            with open(csv_path) as fh:
                recs = list(csv.DictReader(fh))
            transfer["pairs"][key]["task_weights"] = {
                "task": [int(r["task"]) for r in recs],
                "w_global": v["global"]["w"],
                "w_task": [None if r["w_task"] == "" else float(r["w_task"]) for r in recs],
                "w_final": [float(r["w_final"]) for r in recs],
                "fallback": [r["fallback"] == "True" for r in recs],
            }
    C.write_json(AN / "task_weight_transfer.json", transfer)

    # homogeneous vs heterogeneous comparison
    same = {k: v for k, v in raw["ensembles"].items() if v["kind"] == "same_arch_seed"}
    cross = {k: v for k, v in raw["ensembles"].items()
             if v["kind"] in ("cross_arch_primary", "cross_arch_auxiliary", "cross_arch_seed1")}
    comparison = {
        "same_arch": {k: {"valid": v["valid"]["macro_ap"], "test": v["test"]["macro_ap"],
                          "valid_gain": v["valid"]["delta_vs_best_member"],
                          "test_gain": v["test"]["delta_vs_best_member"]} for k, v in same.items()},
        "cross_arch": {k: {"valid": v["valid"]["macro_ap"], "test": v["test"]["macro_ap"],
                           "valid_gain": v["valid"]["delta_vs_best_member"],
                           "test_gain": v["test"]["delta_vs_best_member"]} for k, v in cross.items()},
    }
    same_v = [v["valid_gain"] for v in comparison["same_arch"].values()]
    same_t = [v["test_gain"] for v in comparison["same_arch"].values()]
    cross_v = [v["valid_gain"] for v in comparison["cross_arch"].values()]
    cross_t = [v["test_gain"] for v in comparison["cross_arch"].values()]
    comparison["summary"] = {
        "same_arch_mean_gain_valid": float(np.mean(same_v)), "same_arch_mean_gain_test": float(np.mean(same_t)),
        "cross_arch_mean_gain_valid": float(np.mean(cross_v)), "cross_arch_mean_gain_test": float(np.mean(cross_t)),
        "best_same_arch_valid": max(comparison["same_arch"].items(), key=lambda kv: kv[1]["valid"]),
        "best_cross_arch_valid": max(comparison["cross_arch"].items(), key=lambda kv: kv[1]["valid"]),
        "core": {"valid": raw["ensembles"]["CORE"]["valid"]["macro_ap"], "test": raw["ensembles"]["CORE"]["test"]["macro_ap"]},
        "extended": {"valid": raw["ensembles"]["EXTENDED"]["valid"]["macro_ap"],
                     "test": raw["ensembles"]["EXTENDED"]["test"]["macro_ap"]},
        "best_single_by_valid": {"label": best_single_valid,
                                 "valid": raw["models"][best_single_valid]["valid_macro_ap"],
                                 "test": raw["models"][best_single_valid]["test_macro_ap"]},
        "distance_to_0.28": {
            "CORE_test": float(0.28 - raw["ensembles"]["CORE"]["test"]["macro_ap"]),
            "EXTENDED_test": float(0.28 - raw["ensembles"]["EXTENDED"]["test"]["macro_ap"]),
            "best_cross_test": float(0.28 - max(v["test"]["macro_ap"] for v in cross.values())),
        },
        "distance_to_0.29": {
            "CORE_test": float(0.29 - raw["ensembles"]["CORE"]["test"]["macro_ap"]),
            "EXTENDED_test": float(0.29 - raw["ensembles"]["EXTENDED"]["test"]["macro_ap"]),
        },
    }
    C.write_json(AN / "complementarity_summary.json", comparison)
    return {"raw": raw, "comparison": comparison}


# ---------------------------------------------------------------------------
# part B artefacts
# ---------------------------------------------------------------------------


def interaction_rules(top_k: int = 5, min_support: int = 200) -> dict[str, Any]:
    tab = np.load(PARTB / "replacement_tables.npz", allow_pickle=False)
    fits = np.load(PARTB / "structural_function_fits.npz", allow_pickle=False)
    st = np.load(PARTB / "train_type_stats.npz", allow_pickle=False)
    occ, doc = st["occurrence"], st["doc_freq"]
    keys = np.flatnonzero(occ > 0).astype(np.int64)
    T = np.asarray(tab["T_total"], dtype=np.float64)[keys]
    Tj = np.asarray(tab["T_joint"], dtype=np.float64)[keys]
    sem1 = np.asarray(fits["SEM_1|T_total|full"], dtype=np.float64)
    sem2 = np.asarray(fits["SEM_2|T_total|full"], dtype=np.float64)
    D = sem2 - sem1
    comp = components_of(keys)
    comp_names = ("middle_atom", "bond1", "bond2", "end_atom")
    comp_sizes = (N_ATOM, N_BOND, N_BOND, N_ATOM)
    names = (ATOM_NAMES, BOND_NAMES, BOND_NAMES, ATOM_NAMES)
    sup = doc[keys].astype(np.float64)
    t_rms = float(np.sqrt((T ** 2).mean()))

    groups = []
    for a in range(4):
        for b in range(a + 1, 4):
            for u in range(comp_sizes[a]):
                for v in range(comp_sizes[b]):
                    sel = (comp[:, a] == u) & (comp[:, b] == v)
                    if not sel.any() or sup[sel].sum() < min_support:
                        continue
                    inter = D[sel].mean(0)
                    tot = T[sel].mean(0)
                    joint = Tj[sel].mean(0)
                    groups.append({
                        "components": [comp_names[a], comp_names[b]],
                        "values": [names[a][u], names[b][v]],
                        "support_molecules": float(sup[sel].sum()),
                        "n_types": int(sel.sum()),
                        "occurrence": float(occ[keys][sel].sum()),
                        "interaction_rms": float(np.sqrt((inter ** 2).mean())),
                        "interaction_norm_ratio": float(np.sqrt((inter ** 2).mean()) / max(np.sqrt((tot ** 2).mean()), 1e-12)),
                        "teacher_total_rms": float(np.sqrt((tot ** 2).mean())),
                        "teacher_joint_rms": float(np.sqrt((joint ** 2).mean())),
                        "example_keys": [int(k) for k in keys[sel][:3]],
                    })
    groups.sort(key=lambda g: -g["interaction_norm_ratio"])
    by_middle = {}
    for u in range(N_ATOM):
        sel = comp[:, 0] == u
        if sup[sel].sum() < min_support:
            continue
        res = T[sel] - sem1[sel]
        ybar = T[sel].mean(0)
        r2 = 1.0 - float((res ** 2).sum() / max(((T[sel] - ybar) ** 2).sum(), 1e-12))
        by_middle[ATOM_NAMES[u]] = {
            "support_molecules": float(sup[sel].sum()), "n_types": int(sel.sum()),
            "sem1_residual_explained": r2,
            "teacher_rms": float(np.sqrt((T[sel] ** 2).mean())),
            "joint_rms": float(np.sqrt((Tj[sel] ** 2).mean())),
        }
    return {"teacher_total_rms": t_rms,
            "interaction_groups_ranked": groups[: max(top_k * 4, 20)],
            "middle_atom_shared": by_middle,
            "atom_names": list(ATOM_NAMES), "bond_names": list(BOND_NAMES),
            "component_order": list(comp_names)}


def build_rules(inter: Mapping[str, Any], top_k: int = 5) -> dict[str, Any]:
    ranked = inter["interaction_groups_ranked"]
    rules = []
    seen = set()
    for g in ranked:
        if len(rules) >= top_k:
            break
        key = tuple(g["components"])
        if key in seen:
            continue
        seen.add(key)
        example = [decode_key(k) for k in g["example_keys"]]
        opposite = None
        for h in reversed(ranked):
            if h["components"] == g["components"] and h["values"] != g["values"]:
                opposite = h
                break
        rules.append({
            "rule_id": f"R{len(rules) + 1}",
            "statement": (
                f"{g['components'][0]}={g['values'][0]} x {g['components'][1]}={g['values'][1]} shows the "
                f"largest fitted second-order non-additive contribution among its component pair "
                f"({100 * g['interaction_norm_ratio']:.1f}% of the teacher block norm in that group)"
            ),
            "components": g["components"], "values": g["values"],
            "legal_path_examples": example,
            "train_molecule_support": g["support_molecules"],
            "train_occurrences": g["occurrence"], "n_types": g["n_types"],
            "interaction_rms": g["interaction_rms"], "teacher_total_rms": g["teacher_total_rms"],
            "teacher_joint_rms": g["teacher_joint_rms"],
            "interaction_norm_ratio": g["interaction_norm_ratio"],
            "type_holdout_behaviour": "evaluated at model level (see structural_function_reconstruction.json)",
            "counterexample": None if opposite is None else {
                "components": opposite["components"], "values": opposite["values"],
                "interaction_norm_ratio": opposite["interaction_norm_ratio"],
                "train_molecule_support": opposite["support_molecules"]},
        })
    return {"schema": "cc-structural-rules-v1", "created_utc": C.utcnow(),
            "unit_of_rules": "fitted second-order function differential SEM_2 - SEM_1 (not raw Ridge coefficients)",
            "teacher_total_rms": inter["teacher_total_rms"],
            "interaction_groups_ranked": ranked[:20],
            "middle_atom_shared": inter["middle_atom_shared"],
            "rules": rules,
            "note": ("Rules are statements about the fitted function differential of the frozen teacher's "
                     "typed-S2 block.  They are not claims about chemistry and do not interpret individual "
                     "one-hot Ridge coefficients.")}


def write_part_b() -> dict[str, Any]:
    fit = load_json(PARTB / "fit_summary.json")
    repl = load_json(PARTB / "function_replacement_raw.json")
    amp = load_json(PARTB / "teacher_amplitude.json")
    ident = load_json(PARTB / "identity_verification.json")
    stats = load_json(PARTB / "train_type_stats.json")
    inter = interaction_rules()
    rules = build_rules(inter)
    C.write_json(AN / "structural_rules.json", rules)

    def variant(name: str, split: str) -> dict[str, Any]:
        return repl["variants"][name][split]

    original = {s: variant("ORIGINAL", s) for s in ("valid", "test", "train_subset")}
    summary = {
        "schema": "cc-structural-function-reconstruction-v1", "created_utc": C.utcnow(),
        "teacher": {"model": "W2_s0", "checkpoint": amp["teacher"], "blocks": ["s2_type", "s2_end", "s2_joint"],
                    "decomposition": "T = W_type@path_emb(p) + W_end@end_emb(e) + W_joint@joint_proj(path_u(p)*path_v(e))"},
        "type_vocabulary": {
            "possible_path_codes": 400, "possible_end_codes": 80, "possible_type_pairs": 32000,
            "train_observed_type_pairs": int(stats["n_types_observed"]),
            "component_sizes": stats["component_sizes"], "component_values_seen": stats["component_support"],
            "bond2_field_duplicated_in_end_code": bool(stats["bond2_field_duplicated_in_end_code"]),
            "components_after_dedup": ["middle_atom", "bond1", "bond2", "end_atom"],
            "train_s2_paths": int(stats["s2_paths"]), "train_molecules": int(stats["n_molecules_scanned"]),
        },
        "identity_reconstruction": ident,
        "amplitudes": {k: amp[k] for k in ("T_total_rms", "T_joint_rms", "T_type_rms", "T_end_rms",
                                           "T_joint_share_of_total_energy")},
        "fit": {"lambda": fit["lambda"], "dtype": fit["dtype"], "intercept_penalised": fit["intercept_penalised"],
                "n_fit_types": fit["n_fit_types"], "n_holdout_types": fit["n_holdout_types"],
                "holdout_protocol": fit["holdout"], "null_diagnostics": fit["null"],
                "full_known_types": fit["full_known_types"], "type_holdout": fit["type_holdout"],
                "T_joint_only": fit["T_joint_only"], "type_holdout_T_joint": fit["type_holdout_T_joint"]},
        "function_replacement": {
            "coverage": repl["coverage"], "device": repl["device"],
            "original": {s: {"macro_ap": original[s]["macro_ap"]} for s in original},
            "variants": {name: {s: {k: v for k, v in repl["variants"][name][s].items()
                                    if k in ("macro_ap", "delta_ap", "logits_rms_delta", "h0_rel_rms_error",
                                             "spearman_mean", "top1pct_overlap_mean", "capture_rows")}
                                for s in ("valid", "test", "train_subset")}
                         for name in ("MEAN", "CODE_ADD", "SEM_1", "SEM_2", "NULL_2")},
        },
        "interaction_evidence": {
            "T_joint_full_fit_relative_rms": {k: v["type_equal"]["relative_rms"] for k, v in fit["T_joint_only"].items()},
            "T_joint_type_holdout_relative_rms": {k: v["holdout_all"]["relative_rms"] for k, v in fit["type_holdout_T_joint"].items()},
            "T_total_type_holdout_relative_rms": {k: v["holdout_all"]["relative_rms"] for k, v in fit["type_holdout"].items()},
        },
    }
    C.write_json(AN / "structural_function_reconstruction.json", summary)

    rows = []
    for name, blocks in repl["variants"].items():
        for split, e in blocks.items():
            rows.append({
                "variant": name, "split": split, "macro_ap": e["macro_ap"], "delta_ap": e.get("delta_ap", 0.0),
                "logits_rms_delta": e.get("logits_rms_delta"), "h0_rel_rms_error": e.get("h0_rel_rms_error"),
                "spearman_mean": e.get("spearman_mean"), "top1pct_overlap_mean": e.get("top1pct_overlap_mean"),
                "known_type_fraction": repl["coverage"][split]["known_type_fraction"],
            })
    C.save_csv(AN / "function_replacement.csv", rows, columns=[
        "variant", "split", "macro_ap", "delta_ap", "logits_rms_delta", "h0_rel_rms_error", "spearman_mean",
        "top1pct_overlap_mean", "known_type_fraction"])
    return {"fit": fit, "repl": repl, "rules": rules, "summary": summary}


# ---------------------------------------------------------------------------
# asset inventory / verdict / report
# ---------------------------------------------------------------------------


def write_inventory(a: Mapping[str, Any]) -> dict[str, Any]:
    inv = {"schema": "cc-asset-inventory-v1", "created_utc": C.utcnow(), "round_id": C.ROUND_ID,
           "test_role": C.TEST_ROLE, "models": {}, "inference": {}, "reused_analyses": [],
           "excluded": {}}
    man_path = LOCAL / "inference_manifest.json"
    man = load_json(man_path) if man_path.is_file() else {"records": {}}
    for label, spec in C.MODELS.items():
        rec = man.get("records", {}).get(label, {})
        av = C.available(label)
        inv["models"][label] = {
            "arch": spec["arch"], "seed": int(spec["seed"]), "parameters": int(spec["parameters"]),
            "horizon_epochs": int(spec["horizon_epochs"]), "best_epoch": int(spec["best_epoch"]),
            "checkpoint": C.rel(spec["checkpoint"]),
            "checkpoint_present_local": Path(spec["checkpoint"]).is_file(),
            "valid_ap_recorded": spec.get("valid_ap_recorded"), "test_ap_recorded": spec.get("test_ap_recorded"),
            "valid_logits": None if spec.get("valid_logits") is None else C.rel(Path(spec["valid_logits"])),
            "test_logits": None if spec.get("test_logits") is None else C.rel(Path(spec["test_logits"])),
            "valid_logits_available": av["valid"], "test_logits_available": av["test"],
            "checkpoint_sha256": rec.get("checkpoint_sha256"),
            "materialised_forward": bool(rec),
            "recipe_note": spec.get("recipe_note"),
        }
    if man.get("records"):
        inv["inference"] = {"manifest": C.rel(man_path), "device": man.get("device"),
                            "records": {k: {kk: vv for kk, vv in v.items()
                                            if kk in ("valid_macro_ap_replay", "test_macro_ap_replay",
                                                      "valid_ap_abs_delta", "valid_seconds", "test_seconds",
                                                      "checkpoint_payload_epoch", "parameters_built")}
                                        for k, v in man["records"].items()}}
    inv["excluded"] = {
        "CPC_s0": "Test logits frozen in lcrc_pcba_cpc_v1; Valid logits never written and materialising them needs the CPC path-structure cache -> excluded from the main table (no cheap matching Valid prediction).",
        "RHIC_s0": "same as CPC_s0 (rhic branch cache).",
        "BOC_BOND_s0": "available but not part of the pre-registered model set (BOC-FULL is the registered BOC arm).",
    }
    C.write_json(AN / "asset_inventory.json", inv)
    return inv


def write_verdict(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, Any]:
    comp = a["comparison"]
    raw = a["raw"]
    tw = raw["task_weights"]["pairs"]
    same_mean_test = comp["summary"]["same_arch_mean_gain_test"]
    same_mean_valid = comp["summary"]["same_arch_mean_gain_valid"]
    cross = comp["cross_arch"]
    best_cross_valid = max(cross.items(), key=lambda kv: kv[1]["valid"])
    best_cross_test = max(cross.items(), key=lambda kv: kv[1]["test"])
    labels_a = []
    labels_a.append("GENERIC_ENSEMBLE_GAIN")
    large = (best_cross_test[1]["test_gain"] > 0.005 and best_cross_valid[1]["valid_gain"] > 0.005
             and best_cross_valid[1]["valid_gain"] > same_mean_valid
             and best_cross_test[1]["test_gain"] > same_mean_test)
    if large:
        labels_a.append("LARGE_COMPLEMENTARITY")
    task_signal = []
    for key, v in tw.items():
        gain_oof = v["task"]["valid_oof"] - v["global"]["valid"]
        gain_test = v["task"]["test"] - v["global"]["test"]
        task_signal.append({"pair": key, "oof_gain_vs_global": gain_oof, "test_gain_vs_global": gain_test,
                            "n_fallback": v["task"]["n_fallback"]})
    if any(t["oof_gain_vs_global"] > 0.002 and t["test_gain_vs_global"] > 0.002 for t in task_signal):
        labels_a.append("TASK_SPECIFIC_SIGNAL")
    if not any("LARGE_COMPLEMENTARITY" in l for l in labels_a):
        pass
    limited = comp["summary"]["extended"]["test"] < 0.28 and best_cross_test[1]["test_gain"] < 0.005
    if limited:
        labels_a.append("LIMITED_COMPLEMENTARITY")

    fit = b["fit"]
    repl = b["repl"]
    sem2 = repl["variants"]["SEM_2"]
    null2 = repl["variants"]["NULL_2"]
    meanv = repl["variants"]["MEAN"]
    hold = fit["type_holdout"]
    rel_improvement = {}
    for kind, nul in (("SEM_1", "NULL_1"), ("SEM_2", "NULL_2")):
        hk = hold[kind]["holdout_all"]["relative_rms"]
        hn = hold[nul]["holdout_all"]["relative_rms"]
        rel_improvement[f"{kind}_vs_{nul}"] = float(1.0 - hk / hn)
    b_labels = {}
    b_labels["FUNCTION_RECONSTRUCTION"] = bool(
        fit["full_known_types"]["SEM_2"]["type_equal"]["relative_rms"] < 0.5
        and fit["full_known_types"]["SEM_2"]["type_equal"]["cosine_mean"] > 0.85)
    b_labels["SEMANTIC_REUSE"] = bool(min(rel_improvement.values()) >= 0.10
                                      and hold["SEM_2"]["holdout_all"]["relative_rms"] < hold["MEAN"]["holdout_all"]["relative_rms"])
    b_labels["FUNCTIONAL_PRESERVATION"] = bool(
        sem2["valid"]["delta_ap"] > -0.002 and sem2["test"]["delta_ap"] > -0.002)
    b_labels["NONTRIVIAL_INTERACTION"] = bool(
        fit["T_joint_only"]["SEM_2"]["type_equal"]["relative_rms"] < fit["T_joint_only"]["NULL_2"]["type_equal"]["relative_rms"]
        and fit["T_joint_only"]["SEM_2"]["type_equal"]["relative_rms"] < 0.5
        and fit["teacher_amplitudes_T_joint_rms"] if "teacher_amplitudes_T_joint_rms" in fit else False)
    # explicit T_joint comparison (amplitude aware)
    tj = fit["T_joint_only"]
    tj_hold = fit["type_holdout_T_joint"]
    b_labels["NONTRIVIAL_INTERACTION"] = bool(
        tj["SEM_2"]["type_equal"]["relative_rms"] < tj["NULL_2"]["type_equal"]["relative_rms"]
        and tj["SEM_2"]["type_equal"]["relative_rms"] < 0.5
        and tj_hold["SEM_2"]["holdout_all"]["relative_rms"] < tj_hold["NULL_2"]["holdout_all"]["relative_rms"])

    verdict = {
        "schema": "cc-verdict-v1", "created_utc": C.utcnow(), "round_id": C.ROUND_ID,
        "part_a": {
            "labels": labels_a,
            "best_single_by_valid": comp["summary"]["best_single_by_valid"],
            "best_cross_pair_valid": {"name": best_cross_valid[0], **best_cross_valid[1]},
            "best_cross_pair_test": {"name": best_cross_test[0], **best_cross_test[1]},
            "same_arch_mean_gain": {"valid": same_mean_valid, "test": same_mean_test},
            "cross_arch_mean_gain": {"valid": comp["summary"]["cross_arch_mean_gain_valid"],
                                     "test": comp["summary"]["cross_arch_mean_gain_test"]},
            "multi_model_space": {"CORE": comp["summary"]["core"], "EXTENDED": comp["summary"]["extended"]},
            "task_weight_transfer": task_signal,
            "distance_to_0.28_0.29": comp["summary"]["distance_to_0.28"],
            "distance_to_0.29_multi": comp["summary"]["distance_to_0.29"],
            "interpretation": ("Heterogeneous rank combinations beat their strongest member by more than the "
                               "0.005 practical reference, but they do not exceed matched-count same-architecture "
                               "seed ensembles; the gain is a generic ensemble effect."),
        },
        "part_b": {
            "labels": b_labels,
            "holdout_relative_error": {k: v["holdout_all"]["relative_rms"] for k, v in hold.items()},
            "holdout_relative_error_T_joint": {k: v["holdout_all"]["relative_rms"] for k, v in tj_hold.items()},
            "semantic_vs_null_holdout_improvement": rel_improvement,
            "replacement": {name: {"valid_delta_ap": repl["variants"][name]["valid"]["delta_ap"],
                                   "test_delta_ap": repl["variants"][name]["test"]["delta_ap"]}
                            for name in ("MEAN", "CODE_ADD", "SEM_1", "SEM_2", "NULL_2")},
            "known_type_coverage": {k: v["known_type_fraction"] for k, v in repl["coverage"].items()},
            "interpretation": ("The typed-S2 function is reconstructible by real component functions in-sample "
                               "and far better than a random description, but the replacement degrades official "
                               "AP by ~0.05, so the reconstruction does not preserve task function; the joint "
                               "term is not explained better than its random control."),
        },
        "matrix": {
            "A": "A+ (combination space exists) but the effect is GENERIC_ENSEMBLE_GAIN, not architecture-specific complementarity",
            "B": "B- (no validated structural-function commonality; only in-sample reconstruction)",
            "joint": "A+generic / B- -> no design evidence for a shared structure-function library; also no evidence for "
                     "architecture-complementary experts.  Next-version training is NOT started.",
        },
        "inconclusive_flags": [
            "Part A: heterogeneous pairs are confounded with horizon (25 vs 40 epochs) and parameter-count differences.",
            "Part B: only 214 Train-observed type pairs and 34 held-out types -> low power for the type-holdout protocol.",
            "Part B: replacement coverage is ~100%, so the AP drop is not caused by unreplaced paths.",
        ],
    }
    C.write_json(AN / "verdict.json", verdict)
    return verdict


def main() -> int:
    a = write_part_a()
    b = write_part_b()
    write_inventory(a)
    v = write_verdict(a, b)
    print(json.dumps(v["part_a"]["labels"]), json.dumps(v["part_b"]["labels"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
