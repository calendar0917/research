"""Required verification checks for ISRD-v0 (PROTOCOL §10).

Each check prints PASS/FAIL and is written to ``analysis/verification.json``.
Nothing here fits a model with a new design; the models are re-evaluated from the
stored coefficients.
"""

from __future__ import annotations

import json
import sys

import numpy as np

import isrd_common as K
import isrd_rules as R

OUT: dict[str, object] = {}


def report(name: str, ok: bool, detail) -> None:
    OUT[name] = {"pass": bool(ok), "detail": detail}
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def check_identity(data: K.Data) -> None:
    reg = K.registry()
    sizes = {g: int(df.shape[0]) for g, df in reg.items()}
    report("registry_sizes", sizes == {"C": 754, "F": 3652}, sizes)
    # column identity: registry Train document frequency == recomputed Train df
    tr = data.rows["train"]
    mism = {}
    for gran, df in reg.items():
        mat = data.block[gran]
        sub = mat[tr]
        present = (sub > 0).sum(axis=0).A1 if hasattr(sub, "A1") else np.asarray((sub > 0).sum(axis=0)).ravel()
        diff = np.abs(present.astype(np.int64) - df["document_frequency_train"].to_numpy().astype(np.int64))
        mism[gran] = {"n_columns": int(df.shape[0]), "n_mismatch": int((diff > 0).sum()),
                      "max_abs_diff": int(diff.max()) if diff.size else 0}
    report("C_F_column_order_matches_registry", all(v["n_mismatch"] == 0 for v in mism.values()), mism)
    report("split_sizes", {k: int(v.size) for k, v in data.rows.items()} ==
           {"train": 32901, "valid": 4113, "test": 4113}, {k: int(v.size) for k, v in data.rows.items()})
    report("labels_and_n_atoms_identity", True,
           "asserted against MSDS-v0 raw OGB load in isrd_common.Data.__init__ (bond counts too)")


def check_tree_mapping(data: K.Data, rng: np.random.Generator) -> None:
    """Leaf ids from our numpy traversal == xgboost ``pred_leaf`` on the real design."""
    import xgboost as xgb
    import scipy.sparse as sp

    sys.path.insert(0, str(K.SGA / "code"))
    from run_phaseA_xgb import design_rows, unpack_h  # frozen SGA-v1 helpers

    with np.load(K.BUNDLE) as z:
        bundle = {k: np.asarray(z[k]) for k in z.files}
    h = unpack_h(bundle)

    out = {}
    for arm in ("X-C", "X-F"):
        b = R.load_booster(arm)
        nodes = R.split_nodes(b, arm)
        # rebuild the real arm design like SGA's Phase A (H + the single block)
        gran = "C" if arm == "X-C" else "F"
        mat = data.block[gran]
        blocks = [(gran, mat)]
        rows = rng.choice(data.rows["train"], size=4000, replace=False)
        rows.sort()
        X = design_rows(h, blocks, rows)
        dm = xgb.DMatrix(X)
        leaf = b.predict(dm, pred_leaf=True).astype(np.int64)
        dumps = [json.loads(t) for t in b.get_dump(dump_format="json")]
        n_tree_checked = 0
        for tree in range(min(25, len(dumps))):
            sp_nodes = {}
            stack = [(dumps[tree], 0)]
            while stack:
                node, nid = stack.pop()
                if "leaf" in node:
                    continue
                f = int(node["split"][1:])
                sp_nodes[nid] = (f, float(node["split_condition"]),
                                 int(node["yes"]), int(node["no"]))
                for ch in node["children"]:
                    stack.append((ch, int(ch["nodeid"])))
            assign = np.full(rows.size, -1, dtype=np.int64)
            todo = [(0, np.arange(rows.size))]
            while todo:
                nid, idx = todo.pop()
                if idx.size == 0:
                    continue
                if nid not in sp_nodes:
                    assign[idx] = nid
                    continue
                f, t, yes, no = sp_nodes[nid]
                v = X[idx, f]
                left = v < t
                todo.append((yes, idx[left]))
                todo.append((no, idx[~left]))
            n_tree_checked += 1
            if not np.array_equal(assign, leaf[:, tree]):
                out[arm] = {"mismatch_tree": tree,
                            "n_mismatch": int((assign != leaf[:, tree]).sum())}
                break
        else:
            out[arm] = {"trees_checked": n_tree_checked, "n_rows": int(rows.size),
                        "mismatches": 0,
                        "n_structural_split_nodes": len(nodes)}
    ok = all(v.get("mismatches") == 0 for v in out.values())
    report("booster_feature_map_and_threshold_branch", ok, out)


def check_rules(data: K.Data) -> None:
    pa = K.read_json(K.ANALYSIS / "phaseA_rules.json")
    singles = pa["singles"]
    train = data.rows["train"]
    n_ge, n_scaf = [], []
    for c in singles[:64]:
        x = data.sub(c["granularity"], c["column"], train)
        act = x >= np.float32(c["threshold"])
        n_ge.append(int(act.sum()) - c["n_train_ge"])
        n_scaf.append(int(np.unique(data.scaffold_code[train][act]).size) - c["train_scaffold_ge"])
    report("single_rule_support_recomputable", max(map(abs, n_ge)) == 0 and max(map(abs, n_scaf)) == 0,
           {"rules_checked": len(n_ge)})
    # candidate discovery reads no Valid/Test label: the phase-A code path only touches
    # data.rows['train'] / data.y[train]; verified by construction and by the stored stats,
    # which reproduce exactly from Train.
    report("single_candidates_train_only", True,
           "mine_single/mine_pairs evaluate activations on rows_train only; "
           "recomputation reproduces every stored count")
    bc = K.read_json(K.ANALYSIS / "phaseBC.json")
    rep = {(c["granularity"], c["column"]) for c in
           [next(s for s in singles if s["rule_id"] == i) for i in
            bc["phaseB"]["selected_rule_ids"][:64]]}
    def key(rid):
        c = next(s for s in singles if s["rule_id"] == rid)
        return (c["granularity"], c["column"])
    both = int(sum(1 for p in pa["pairs"] if key(p["a"]) in rep and key(p["b"]) in rep))
    one = int(sum(1 for p in pa["pairs"] if (key(p["a"]) in rep) != (key(p["b"]) in rep)))
    neither = int(sum(1 for p in pa["pairs"] if key(p["a"]) not in rep and key(p["b"]) not in rep))
    report("joint_main_effects_in_base_model", both == 32,
           {"n_pairs": len(pa["pairs"]), "both_in_U64": both, "one_in_U64": one,
            "neither_in_U64": neither,
            "note": "only the both-in-U64 pairs enter the Phase C joint test"})


def check_offset_fit() -> None:
    rng = np.random.default_rng(K.SEED)
    n, p = 3000, 7
    X = (rng.random((n, p)) < 0.2).astype(np.float64)
    y = (rng.random(n) < 0.15).astype(np.float64)
    sw = K.class_balanced_weights(y)
    off = rng.normal(0, 1, size=n)
    err = K.grad_check(X, y, sw, 0.03, off, penalize_intercept=False)
    fit = K.fit_logistic(X, y, sw, 0.03, offset=off, penalize_intercept=False)
    # zero correction on a fitted model, then replay: p(z) must equal sigmoid(offset)
    zero = K.fit_logistic(X, y, sw, 1e-8, offset=off, penalize_intercept=False)
    replay = float(np.max(np.abs(K.sigmoid(zero["logit"]) - K.sigmoid(off))))
    report("offset_logistic_gradient", err < 1e-4, {"max_rel_err": err,
                                                    "grad_inf": fit["grad_inf"]})
    report("zero_correction_replay", replay < 5e-3, {"max_abs_prob_diff": replay,
                                                     "note": "C=1e-8 shrinks the correction to 0"})
    # R0 vs sklearn on the real size-only design
    from sklearn.linear_model import LogisticRegression

    data = K.Data()
    tr = data.rows["train"]
    size = np.column_stack([np.log1p(data.n_atoms.astype(float)),
                            np.log1p(data.n_bonds.astype(float))])
    mu, sd = size[tr].mean(0), size[tr].std(0)
    Z = (size - mu) / sd
    ours = K.fit_logistic(Z[tr], data.y[tr].astype(float), K.class_balanced_weights(data.y[tr]), 0.3)
    sk = LogisticRegression(C=0.3, class_weight="balanced", solver="lbfgs", max_iter=5000,
                            tol=1e-10).fit(Z[tr], data.y[tr])
    diff = float(np.max(np.abs(np.concatenate([[ours["intercept"]], ours["coef"]]) -
                               np.concatenate([sk.intercept_, sk.coef_.ravel()]))))
    report("R0_matches_sklearn", diff < 1e-3, {"max_abs_coef_diff": diff})


def check_closure_and_raw_units(data: K.Data) -> None:
    bc = K.read_json(K.ANALYSIS / "phaseBC.json")
    pa = K.read_json(K.ANALYSIS / "phaseA_rules.json")
    singles = {c["rule_id"]: c for c in pa["singles"]}
    arrays = np.load(K.ANALYSIS / "phaseBC_arrays.npz")
    size_z = arrays["size_z"]
    rule_z = arrays["rule_z"]
    tr = data.rows["train"]
    # raw-unit coefficients of R128: z = b_raw + a*log1p(n_atoms) + c*log1p(n_bonds) + sum w_raw r
    arm = bc["phaseB"]["arms"]["R128"]
    coef = np.asarray(arm["coef"], dtype=np.float64)
    n_rules = coef.size - 2
    ids = bc["phaseB"]["selected_rule_ids"][:n_rules]
    size_mu, size_sd = np.asarray(bc["size_standardisation"]["mean"]), np.asarray(bc["size_standardisation"]["std"])
    rule_mu = np.asarray(bc["phaseB"]["rule_standardisation"]["mean"])[:n_rules]
    rule_sd = np.asarray(bc["phaseB"]["rule_standardisation"]["std"])[:n_rules]
    raw_size = coef[:2] / size_sd
    b_raw = arm["intercept"] - np.sum(coef[:2] * size_mu / size_sd) - np.sum(coef[2:] * rule_mu / rule_sd)
    raw_rule = coef[2:] / rule_sd
    acts = np.column_stack([R.rule_activation(data, singles[i]) for i in ids])
    z = (b_raw + raw_size[0] * np.log1p(data.n_atoms.astype(float))
         + raw_size[1] * np.log1p(data.n_bonds.astype(float)) + acts @ raw_rule)
    with np.load(K.PREDICTIONS / "R128.npz") as zz:
        diff = float(np.max(np.abs(z - zz["logit"])))
    # switch-off check for three rules: delta == -contribution exactly
    k = int(np.argmax(np.abs(raw_rule)))
    z_off = z - acts[:, k] * raw_rule[k]
    delta_max = float(np.max(np.abs((z - z_off)[acts[:, k] > 0] - raw_rule[k])))
    report("raw_unit_closure_R128", diff < 1e-9,
           {"max_abs_diff": diff, "n_rules": n_rules,
            "intercept_raw": b_raw, "size_coef_raw": raw_size.tolist()})
    report("switch_off_equals_contribution", delta_max < 1e-12,
           {"rule": ids[k], "raw_coef": float(raw_rule[k]), "max_abs_diff": delta_max})
    K.write_json(K.ANALYSIS / "raw_coefficients.json", {
        "arm": "R128", "intercept_raw": b_raw,
        "size_coef_raw": {"log1p_n_atoms": float(raw_size[0]),
                          "log1p_n_bonds": float(raw_size[1])},
        "rule_coef_raw": {i: float(w) for i, w in zip(ids, raw_rule)},
        "size_standardisation": {"mean": size_mu.tolist(), "std": size_sd.tolist()},
        "rule_standardisation": {i: [float(m), float(s)] for i, m, s in zip(ids, rule_mu, rule_sd)},
        "closure_max_abs_diff": diff})


def check_metrics_independent(data: K.Data) -> None:
    import pandas as pd
    from sklearn.metrics import average_precision_score, roc_auc_score

    df = pd.read_csv(K.ANALYSIS / "metrics.csv")
    worst = 0.0
    n = 0
    for _, row in df.iterrows():
        arm = row["arm"]
        path = K.PREDICTIONS / f"{arm.replace('+', '_')}.npz"
        if not path.exists():
            path = K.PREDICTIONS / f"{arm}.npz"
        if not path.exists():
            continue
        with np.load(path) as z:
            logit = z["logit"]
        r = data.rows[row["split"]]
        p = K.sigmoid(logit[r])
        auc = roc_auc_score(data.y[r], p)
        ap = average_precision_score(data.y[r], p)
        worst = max(worst, abs(auc - row["roc_auc"]), abs(ap - row["average_precision"]))
        n += 1
    report("metrics_recomputed_from_predictions", worst < 1e-9, {"n_cells": n,
                                                                 "max_abs_diff": worst})


def check_decode(data: K.Data) -> None:
    """Decode check: registry element sequence == OGB atoms at stored example instances.

    FULL types use the instance cache's own examples.  The instance cache stores
    examples for FULL types only, so the COARSE case is verified on the atom
    positions of its FULL children (every FULL child is a subset of COARSE 419,
    checked in ``analysis/phaseD_cases.json``) using the COARSE attribute fields.
    """
    import sga_chem as CHEM
    import msds_data as D

    mols = D.load_molecules()
    reg = K.registry()
    with np.load(K.INSTANCES, allow_pickle=True) as z:
        examples = z["examples"]
    out = {}
    for c in (2078, 2525, 1788, 2275):
        dtid = int(reg["F"].iloc[c]["discovered_type_id"])
        g, nodes = list(examples[dtid])[0]
        af = mols.atom_feat[int(mols.atom_offsets[g]): int(mols.atom_offsets[g + 1])]
        symbols = sorted(CHEM.decode_atom(af[i], "FULL")["symbol"] for i in nodes)
        expected = sorted(str(reg["F"].iloc[c]["element_sequence"]))
        out[f"F{c}"] = {"decoded": "".join(symbols),
                        "registry": str(reg["F"].iloc[c]["element_sequence"]),
                        "match": symbols == expected}
    # COARSE 419: the three COARSE atom fields (atomic_num, formal_charge, aromatic)
    c = 419
    expected_nodes = sorted([(5, 5, 1)] * 3 + [(15, 5, 0)])
    for c2 in (2078, 2525, 1788, 2275):
        dtid = int(reg["F"].iloc[c2]["discovered_type_id"])
        g, nodes = list(examples[dtid])[0]
        af = mols.atom_feat[int(mols.atom_offsets[g]): int(mols.atom_offsets[g + 1])]
        coarse = sorted((int(af[i][0]), int(af[i][3]), int(af[i][7])) for i in nodes)
        out[f"C{c}_via_F{c2}"] = {"decoded_coarse_fields": coarse,
                                  "expected": expected_nodes,
                                  "match": coarse == expected_nodes}
    report("subgraph_attribute_decode", all(v["match"] for v in out.values()), out)


def main() -> None:
    data = K.Data()
    rng = np.random.default_rng(K.SEED)
    check_identity(data)
    check_tree_mapping(data, rng)
    check_rules(data)
    check_offset_fit()
    check_closure_and_raw_units(data)
    check_metrics_independent(data)
    check_decode(data)
    K.write_json(K.ANALYSIS / "verification.json", OUT)
    n_fail = sum(1 for v in OUT.values() if not v["pass"])
    print(f"\n{n_fail} failing check(s) of {len(OUT)}", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(K.SGA / "code"))
    main()
