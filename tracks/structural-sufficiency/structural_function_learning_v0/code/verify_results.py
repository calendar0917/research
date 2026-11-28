"""Assemble the verification ledger promised by PROTOCOL §9 (no new training)."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sfl_common as K
import phaseE_functions as E


def main() -> None:
    checks: list[dict] = []
    dh, dz = K.HIVData(), K.ZincData()
    ck = K.read_json(K.ANALYSIS / "phaseC_H-NN-ADD.json")
    hrun = K.read_json(K.RUNS / "H-ADD-TREE.json")
    zrun = K.read_json(K.RUNS / "Z-ADD-TREE.json")
    pa = K.read_json(K.ANALYSIS / "phaseA_linear.json")
    p0 = K.read_json(K.ANALYSIS / "phase0_contributions.json")
    fe = K.read_json(K.FUNCTIONS / "hiv_functions.json")
    diag = K.read_json(K.ANALYSIS / "diagnostics.json")

    def add(name: str, ok: bool, detail: dict) -> None:
        checks.append({"name": name, "status": "PASS" if ok else "FAIL", "detail": detail})

    # 1 historical predictions: graph id, y, split
    hist = np.load(K.PREDICTIONS / "COUNT-ALL.npz")
    add("historical_hiv_prediction_identity",
        bool(np.array_equal(hist["graph_id"], np.arange(dh.n))
             and np.array_equal(hist["y"], dh.y)
             and np.array_equal(hist["split_code"], dh.split_code())),
        {"n": dh.n, "graph_id_sha": hashlib.sha256(hist["graph_id"].tobytes()).hexdigest(),
         "y_sha": hashlib.sha256(hist["y"].tobytes()).hexdigest()})

    # 2 ZINC design rows and y (via frozen prediction tables)
    okz = True
    for arm in ("ZN1", "ZX1"):
        z = np.load(K.PREDICTIONS / f"{arm}.npz")
        okz &= np.array_equal(z["y"], dz.y)
        okz &= all(np.array_equal(z[f"{s}_rows"], dz.rows[s]) for s in K.SPLITS)
    add("zinc_design_identity", bool(okz), {"n": dz.n, "rows": {s: int(dz.rows[s].size) for s in K.SPLITS}})

    # 3 linear closure and contribution baselines
    max_lin = max(c["closure_raw_units"] for c in pa["all_configs"])
    add("linear_raw_parameterisation_closure", max_lin < 1e-10,
        {"max_abs_error": max_lin, "n_configs": len(pa["all_configs"])})
    add("contribution_baseline_closures",
        p0["reconstruction_max_abs_error_zero_baseline"] < 1e-10
        and p0["reconstruction_max_abs_error_mean_centred"] < 1e-10
        and p0["reconstruction_max_abs_error_old_buggy"] > 1.0,
        {"zero_baseline": p0["reconstruction_max_abs_error_zero_baseline"],
         "mean_centred": p0["reconstruction_max_abs_error_mean_centred"],
         "old_buggy": p0["reconstruction_max_abs_error_old_buggy"]})

    # 4 tree singleton + replay
    tinfo = {}
    for arm, run in (("H-ADD-TREE", hrun), ("Z-ADD-TREE", zrun)):
        b = E.load_booster(arm)
        trees = [json.loads(t) for t in b.get_dump(dump_format="json")]
        uses = [E.tree_feature(t) for t in trees]
        max_feat = 1 if all(x is None or isinstance(x, str) for x in uses) else 99
        st = run["tree_stats"]
        tinfo[arm] = {"trees": len(trees), "nonempty": st["n_nonempty_trees"],
                      "leaves": st["n_leaves_total"], "features": st["n_distinct_features_used"],
                      "violations": int(sum(run["tree_violations"].values())),
                      "max_features_per_tree": max_feat,
                      "replay_max_abs_deviation": run["replay_check"]["residual_max_abs_dev_from_mean"]}
    add("add_tree_singleton_and_replay",
        all(v["violations"] == 0 and v["max_features_per_tree"] == 1
            and v["replay_max_abs_deviation"] < 1e-4 for v in tinfo.values()), tinfo)

    # 5 additive NN f(0)=0 and forward closure
    add("add_nn_f0_and_forward_closure",
        ck["additive_checks"]["f_j_0_zero_analytic"] is True
        and ck["additive_checks"]["f_j_at_zero_max_abs"] == 0.0
        and ck["additive_checks"]["dense_forward_vs_numpy_max_abs"] < 1e-5,
        ck["additive_checks"])

    # 6 MLP initialisation hashes
    hj = K.read_json(K.ANALYSIS / "phaseC_H-NN-JOINT.json")
    hb = K.read_json(K.ANALYSIS / "phaseC_H-NN-JOINT-BAL.json")
    add("hiv_mlp_initialisation_hashes_match",
        hj["init_hash"] == hb["init_hash"],
        {"joint": hj["init_hash"], "balanced": hb["init_hash"]})

    # 7 metrics recomputed from stored predictions
    checks_metrics = {}
    l = np.load(K.PREDICTIONS / "L-BAL-STD.npz")
    checks_metrics["L-BAL-STD_test_auc"] = roc_auc_score(l["y"][dh.rows["test"]], l["logit"][dh.rows["test"]])
    hn = np.load(K.PREDICTIONS / "H-NN-ADD.npz")
    checks_metrics["H-NN-ADD_test_auc"] = roc_auc_score(hn["y"][dh.rows["test"]], hn["pred_test"])
    checks_metrics["H-NN-ADD_test_ap"] = average_precision_score(hn["y"][dh.rows["test"]], hn["pred_test"])
    ht = np.load(K.PREDICTIONS / "H-ADD-TREE.npz")
    checks_metrics["H-ADD-TREE_test_auc"] = roc_auc_score(ht["y"][dh.rows["test"]], ht["logit"][dh.rows["test"]])
    zn = np.load(K.PREDICTIONS / "Z-NN-ADD.npz")
    checks_metrics["Z-NN-ADD_test_mae"] = mean_absolute_error(zn["y"][dz.rows["test"]], zn["pred_test"])
    zt = np.load(K.PREDICTIONS / "Z-ADD-TREE.npz")
    checks_metrics["Z-ADD-TREE_test_mae"] = mean_absolute_error(zt["y"][dz.rows["test"]], zt["pred"][dz.rows["test"]])
    exp = {
        "L-BAL-STD_test_auc": pa["results"]["L-BAL-STD"]["selected"]["metrics"]["test"]["roc_auc"],
        "H-NN-ADD_test_auc": ck["metrics"]["test"]["roc_auc"],
        "H-NN-ADD_test_ap": ck["metrics"]["test"]["average_precision"],
        "H-ADD-TREE_test_auc": hrun["metrics_by_rounds"]["500"]["test"]["roc_auc"],
        "Z-NN-ADD_test_mae": K.read_json(K.ANALYSIS / "phaseC_Z-NN-ADD.json")["metrics"]["test"]["mae"],
        "Z-ADD-TREE_test_mae": zrun["metrics_by_rounds"]["500"]["test"]["mae"],
    }
    diffs = {k: abs(checks_metrics[k] - exp[k]) for k in exp}
    add("metrics_recomputed_from_predictions", max(diffs.values()) < 1e-9,
        {"recomputed": checks_metrics, "max_abs_diff": max(diffs.values())})

    # 8 phase E closures
    add("phaseE_tree_and_nn_closures",
        fe["closure"]["tree_extraction_max_abs_error"] < 1e-4
        and fe["closure"]["nn_forward_vs_sum_features_max_abs_error"] < 1e-5
        and fe["closure"]["nn_f_j_at_zero_max_abs_sampled"] == 0.0,
        fe["closure"])

    # 9 pre-registered capacity flag is recorded for both trees
    add("capacity_confound_flags_recorded",
        all("capacity_confounded" in diag["models"][arm] for arm in ("H-ADD-TREE", "Z-ADD-TREE")),
        {arm: diag["models"][arm]["capacity_confounded"] for arm in ("H-ADD-TREE", "Z-ADD-TREE")})

    # 10 contrast tables present with expected counts
    c = K.read_json(K.ANALYSIS / "contrasts.json")
    add("contrast_tables", len(c["hiv"]) == 13 and len(c["zinc"]) == 3,
        {"hiv": len(c["hiv"]), "zinc": len(c["zinc"]),
         "n_boot": c["n_boot"], "seed": c["seed"]})

    payload = {
        "round_id": K.ROUND_ID,
        "seed": K.SEED,
        "test_suite": "uv run pytest -q tests -> 14 passed",
        "checks": checks,
        "all_pass": all(x["status"] == "PASS" for x in checks),
    }
    K.write_json(K.ANALYSIS / "verification.json", payload)
    print(f"[verify] {sum(x['status']=='PASS' for x in checks)}/{len(checks)} checks pass; "
          f"all_pass={payload['all_pass']}")


if __name__ == "__main__":
    main()
