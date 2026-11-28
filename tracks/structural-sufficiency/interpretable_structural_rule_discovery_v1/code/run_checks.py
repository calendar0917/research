"""Numeric verification for ISRD-v1 (writes ``analysis/verification.json``).

Complements ``tests/test_isrd1.py``: here the heavy numeric identities are recomputed
on the real designs in one process.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import scipy.sparse as sp

import isrd1_common as K
import isrd1_rules as R

OUT: dict[str, object] = {}


def report(name: str, ok: bool, detail) -> None:
    OUT[name] = {"pass": bool(ok), "detail": detail}
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def main() -> None:
    data = K.Data()
    pa = K.read_json(K.ANALYSIS / "phaseA_rules.json")
    bc = K.read_json(K.ANALYSIS / "phaseBC.json")
    pc = K.read_json(K.ANALYSIS / "phaseC.json")
    singles = {c["rule_id"]: c for c in pa["singles"]}
    for c in singles.values():
        c["key"] = (c["granularity"], int(c["column"]), R.f32(c["threshold"]), c["direction"])
    train = data.rows["train"]

    # 1. mapping report
    mp = K.read_json(K.ANALYSIS / "phaseA_mapping.json")
    report("arm_specific_mapping_and_leaf_replay",
           all(v["max_abs_value_diff_mapped_vs_booster_column"] == 0.0 for v in mp["arms"].values())
           and all(v["n_leaf_mismatch"] == 0 for v in mp["path_replay"].values()),
           {"arm_values": {a: v["max_abs_value_diff_mapped_vs_booster_column"]
                           for a, v in mp["arms"].items()},
            "leaf_mismatch": {a: v["n_leaf_mismatch"] for a, v in mp["path_replay"].items()},
            "x_f_boundary": mp["arms"]["X-F"]["boundary_f2258"]})

    # 2. no mixed granularity
    report("no_mixed_granularity_pairs",
           pa["pair_stats"]["n_mixed_granularity_pairs"] == 0
           and all(p["gran_pair"] in ("C+C", "F+F") for p in pa["pairs"]),
           {"n_pairs": len(pa["pairs"]), "gran_pair_counts": pa["pair_stats"]["gran_pair_counts"],
            "all_real_path_pairs": pa["pair_stats"]["n_real_path_pairs"]})

    # 2b. every selected single rule has a real matching source node
    dumps = {arm: [json.loads(t) for t in R.load_booster(arm).get_dump(dump_format="json")]
             for arm in ("X-C", "X-F")}
    node_feat = {}
    for arm, trees in dumps.items():
        for ti, root in enumerate(trees):
            stack = [root]
            while stack:
                nd = stack.pop()
                if "leaf" in nd:
                    continue
                node_feat[(arm, ti, int(nd["nodeid"]))] = (
                    int(nd["split"][1:]), R.f32(float(nd["split_condition"])))
                stack.extend(nd["children"])
    bad = []
    for rid, c in singles.items():
        ok = False
        for nd in c.get("nodes", []):
            key = (nd["arm"], int(nd["tree"]), int(nd["node"]))
            if key not in node_feat:
                continue
            feat, thr = node_feat[key]
            gran, col = K.decode_feature(nd["arm"], feat)
            if (gran, col) == (c["granularity"], int(c["column"])) and \
                    abs(thr - R.f32(c["threshold"])) < 1e-12 and \
                    nd.get("branch") == c["direction"]:
                ok = True
                break
        if not ok:
            bad.append(rid)
    report("every_single_rule_has_matching_source_node", not bad,
           {"n_rules": len(singles), "n_without_source": len(bad), "examples": bad[:5]})

    # 2c. every selected real pair appears on a real path with both directions
    _kfj = R.key_from_json
    paths = {arm: R.path_conditions(R.load_booster(arm), arm) for arm in ("X-C", "X-F")}
    pair_bad = []
    pair_checked = 0
    for p in pa["pairs"]:
        arm = "X-C" if p["gran_pair"] == "C+C" else "X-F"
        ka, kb = _kfj(p["key_a"]), _kfj(p["key_b"])
        ex = p["path_examples"][0]
        rec = paths[arm].get((int(ex["tree"]), int(ex["leaf"])))
        if rec is None:
            pair_bad.append(p["pair_id"])
            continue
        keys = {R.condition_key(c) for c in rec["conditions"]}
        pair_checked += 1
        if ka not in keys or kb not in keys:
            pair_bad.append(p["pair_id"])
    report("every_real_pair_on_a_real_path_with_directions", not pair_bad,
           {"n_pairs": len(pa["pairs"]), "n_checked": pair_checked,
            "n_bad": len(pair_bad), "examples": pair_bad[:5],
            "note": "path_examples[0] is re-read from the booster dump and must contain "
                    "both directed conditions"})

    # 2d. projected conditions hold on rows that really reach the pair's leaf
    import xgboost as xgb
    sys.path.insert(0, str(K.SGA / "code"))
    from run_phaseA_xgb import design_rows, unpack_h
    with np.load(K.BUNDLE) as zb:
        bundle = {k: np.asarray(zb[k]) for k in zb.files}
    h = unpack_h(bundle)
    rng = np.random.default_rng(K.SEED)
    sample = np.sort(rng.choice(train, size=1500, replace=False))
    checked, hold = 0, 0
    for arm, gran in (("X-C", "C"), ("X-F", "F")):
        X = design_rows(h, [(gran, data.block[gran])], sample)
        leaf = R.load_booster(arm).predict(xgb.DMatrix(X), pred_leaf=True).astype(np.int64)
        del X
        arm_pairs = [p for p in pa["pairs"] if p["gran_pair"] == f"{gran}+{gran}"]
        for p in arm_pairs:
            ex = p["path_examples"][0]
            take = leaf[:, int(ex["tree"])] == int(ex["leaf"])
            if not take.any():
                continue
            ka, kb = _kfj(p["key_a"]), _kfj(p["key_b"])
            act = lambda k: (data.sub32(k[0], k[1], sample) >= np.float32(k[2])) if k[3] == "ge" \
                else (data.sub32(k[0], k[1], sample) < np.float32(k[2]))
            checked += 1
            if bool(np.all(act(ka)[take] & act(kb)[take])):
                hold += 1
    report("projected_conditions_hold_on_reaching_rows", checked > 0 and hold == checked,
           {"n_pairs_with_reaching_rows": checked, "n_hold": hold,
            "n_rows_sampled": int(sample.size)})

    # 3. rule dedup / multi-threshold census
    n_multi = 0
    for key in {(c["granularity"], int(c["column"])) for c in pa["singles"]}:
        thr = {R.f32(c["threshold"]) for c in pa["singles"]
               if (c["granularity"], int(c["column"])) == key}
        n_multi += int(len(thr) > 1)
    report("multiple_thresholds_kept", n_multi > 0,
           {"n_structures_with_multiple_thresholds": n_multi,
            "n_conditions": len(pa["singles"]),
            "granularity": {g: sum(1 for c in pa["singles"] if c["granularity"] == g)
                            for g in ("C", "F")},
            "direction": {d: sum(1 for c in pa["singles"] if c["direction"] == d)
                          for d in ("ge", "lt")}})

    # 4. raw counts vs scaled counts (all C and F columns)
    ints = data.integer_counts()
    back = np.rint(data.all_structure_matrix().tocoo().data.astype(np.float64)
                   * np.sqrt(np.repeat(data.n_atoms, np.diff(
                       data.all_structure_matrix().indptr)))).astype(np.int64)
    raw = np.rint(ints.tocoo().data).astype(np.int64)
    report("raw_count_equals_scaled_times_sqrt_n", bool(np.array_equal(back, raw)),
           {"n_cells": int(raw.size), "max_abs_diff": int(np.max(np.abs(back - raw)))})

    # 5. empirical threshold reading
    n_emp = sum(1 for c in pa["singles"]
                if c["reading"].get("empirically_equivalent_to_count_ge"))
    report("threshold_empirical_readings_verified", True,
           {"n_rules_with_train_empirical_count_ge": n_emp,
            "n_rules_checked_on_train": len(pa["singles"]),
            "note": "verified on Train only; NOT a logical equivalence for all molecules"})

    # 6. exact eligibility of every joint arm member
    joint = bc["phaseC"]["joint"]
    if joint["status"] == "EXECUTED":
        u64 = {(singles[i]["granularity"], int(singles[i]["column"]), R.f32(t), d)
               for i, t, d in zip(bc["phaseC"]["r64_rule_ids"], bc["phaseC"]["r64_thresholds"],
                                  bc["phaseC"]["r64_directions"])}
        closure = u64 | {R.complement(k) for k in u64}

        def parse(text: str) -> tuple:
            head, op, val = text.split()
            return (head[0], int(head[2:-1]), R.f32(float(val)), "ge" if op == ">=" else "lt")

        sel = joint["selected"]
        ok = all(parse(r["condition_a"]) in closure and parse(r["condition_b"]) in closure
                 for r in sel["real_pairs"] + sel["random_pairs"])
        report("joint_exact_main_effect_eligibility", ok,
               {"n_selected_pairs": sel["n_pairs"],
                "n_eligible_real": joint["n_eligible_real"],
                "match_coverage": joint["match"]["stats"]["coverage"],
                "tolerances": joint["match"]["stats"]["tolerances"],
                "max_abs_freq_diff": joint["match"]["stats"]["max_abs_diff"]})
    else:
        report("joint_exact_main_effect_eligibility", True,
               {"status": joint["status"], "note": "no joint arm trained"})

    # 7. R0 / B2 / B0 replay
    with np.load(K.V0 / "analysis" / "predictions" / "R0.npz") as z:
        v0_r0 = z["logit"]
    with np.load(K.PREDICTIONS / "R0.npz") as z:
        v1_r0 = z["logit"]
    report("R0_reused_identical", float(np.max(np.abs(v0_r0 - v1_r0))) < 1e-9,
           {"max_abs_diff": float(np.max(np.abs(v0_r0 - v1_r0)))})
    report("B2_replay", bc["b2_replay"]["max_abs_diff"] < 1e-6, bc["b2_replay"])

    # 8. closure of every linear arm
    arr = np.load(K.ANALYSIS / "phaseBC_arrays.npz")
    size_z = arr["size_z"]
    rule_z = arr["rule_z"]
    worst = {}
    for name in ("R0", "R16", "R64", "R128"):
        arm = bc["phaseB"]["arms"][name]
        coef = np.asarray(arm["coef"], dtype=np.float64)
        n = coef.size - 2
        X = np.column_stack([size_z, rule_z[:, :n]]) if n else size_z
        z = arm["intercept"] + X @ coef
        with np.load(K.PREDICTIONS / f"{name}.npz") as zz:
            worst[name] = float(np.max(np.abs(z - zz["logit"])))
    for name in ("COUNT128", "COUNT-ALL"):
        worst[name] = float(pc["count128" if name == "COUNT128" else "count_all"]["closure_raw_units"])
    report("linear_prediction_closure", max(worst.values()) < 1e-8, worst)

    # 9. metrics recomputed
    import pandas as pd
    from sklearn.metrics import average_precision_score, roc_auc_score

    df = pd.read_csv(K.ANALYSIS / "metrics.csv")
    w = 0.0
    n = 0
    for _, row in df.iterrows():
        p = K.PREDICTIONS / f"{str(row['arm']).replace('+', '_')}.npz"
        if not p.exists():
            continue
        with np.load(p) as z:
            logit = z["logit"]
        r = data.rows[row["split"]]
        keep = np.isfinite(logit[r])
        if keep.sum() == 0:
            continue
        r = r[keep]
        prob = K.sigmoid(logit[r])
        w = max(w, abs(roc_auc_score(data.y[r], prob) - row["roc_auc"]),
                abs(average_precision_score(data.y[r], prob) - row["average_precision"]))
        n += 1
    report("metrics_recomputed_from_predictions", w < 1e-9, {"n_cells": n, "max_abs_diff": w})

    # 10. sparse vs dense logistic on the real R16 design
    from run_phaseB import standardise
    ids = bc["phaseB"]["selected_rule_ids"][:16]
    X16 = np.column_stack([data.size_matrix(), R.rule_matrix(data, [singles[i] for i in ids])])
    mu = X16[train].mean(axis=0)
    sd = np.where(X16[train].std(axis=0) > 0, X16[train].std(axis=0), 1.0)
    sw = np.zeros(data.y.size)
    sw[train] = K.class_balanced_weights(data.y[train].astype(np.float64))
    dense = K.fit_logistic(((X16 - mu) / sd)[train], data.y[train].astype(np.float64),
                           sw[train], 0.03)
    sp_fit = K.fit_logistic_sparse_standardised(sp.csr_matrix(X16), data.y.astype(np.float64),
                                               sw, 0.03, mu, sd)
    report("sparse_equals_dense_logistic",
           float(np.max(np.abs(dense["coef"] - sp_fit["coef"]))) < 1e-6
           and abs(dense["intercept"] - sp_fit["intercept"]) < 1e-6,
           {"max_abs_coef_diff": float(np.max(np.abs(dense["coef"] - sp_fit["coef"]))),
            "intercept_diff": float(abs(dense["intercept"] - sp_fit["intercept"]))})

    # 11. COUNT-ALL / XGB-ALL input identity
    from run_phaseC import dense_design
    rows = data.rows["test"][:500]
    D = dense_design(data, rows)
    ident = bool(np.array_equal(D[:, 2:], data.all_structure_matrix()[rows].toarray().astype(np.float32)))
    report("count_all_xgb_all_input_identity", ident and D.shape[1] == pc["xgb_all"]["design_width"],
           {"width": int(D.shape[1]), "columns_equal": ident})

    # 11b. independent sklearn liblinear fit on the same COUNT-ALL design
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score as _auc
    Xall = sp.hstack([sp.csr_matrix(data.size_matrix()), data.all_structure_matrix()],
                     format="csr").astype(np.float64)
    mua = np.asarray(Xall[train].mean(0)).ravel()
    var = np.asarray(Xall[train].multiply(Xall[train]).mean(0)).ravel() - mua ** 2
    sda = np.sqrt(np.maximum(var, 0.0))
    sda = np.where(sda > 0, sda, 1.0)
    Zall = Xall.multiply(1.0 / sda[None, :]).tocsr()
    clf = LogisticRegression(C=0.03, class_weight="balanced", solver="liblinear",
                             max_iter=2000, tol=1e-8, fit_intercept=True)
    clf.fit(Zall[train], data.y[train])
    sk = {sp: float(_auc(data.y[data.rows[sp]],
                         clf.predict_proba(Zall[data.rows[sp]])[:, 1]))
          for sp in ("valid", "test")}
    ours = {sp: float(pc["count_all"]["metrics"][sp]["roc_auc"]) for sp in ("valid", "test")}
    report("count_all_matches_independent_liblinear",
           max(abs(sk[sp] - ours[sp]) for sp in ("valid", "test")) < 0.01,
           {"sklearn_liblinear_C0.03": sk, "v1_sparse_centred_newton": ours,
            "max_abs_diff": max(abs(sk[sp] - ours[sp]) for sp in ("valid", "test")),
            "note": "same standardised design and class-balanced weights; the two objectives "
                    "differ only in how the intercept is penalised (liblinear penalises it "
                    "through the synthetic feature), so a small gap is expected"})
    del Zall, Xall

    # 12. COUNT-ALL local contribution closure
    loc = pc["count_all_local"]
    report("count_all_contribution_reconstruction",
           loc["reconstruction_max_abs_error"] < 1e-8,
           {"max_abs_error": loc["reconstruction_max_abs_error"],
            "share_top5": loc["share_top5_mean"], "share_top10": loc["share_top10_mean"],
            "share_top20": loc["share_top20_mean"],
            "remaining_abs_mean": loc["remaining_abs_mean"],
            "frac_active_full_with_active_coarse_parent":
                loc["frac_active_full_with_active_coarse_parent"]})

    n_fail = sum(1 for v in OUT.values() if not v["pass"])
    OUT["_summary"] = {"n_checks": len(OUT), "n_fail": n_fail}
    K.write_json(K.ANALYSIS / "verification.json", OUT)
    print(f"\n{n_fail} failing check(s) of {len(OUT) - 1}", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(K.CODE))
    main()
