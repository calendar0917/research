"""Phase B — additive tree readout (``ADD-TREE``) with XGBoost singleton
interaction constraints, for MolHIV and ZINC.

The model is *exactly* additive over input features only if every tree uses at most
one feature.  We enforce that with ``interaction_constraints=[[0],[1],...]`` (the
documented singleton semantics), verify the constraint on synthetic data before the
real fit, and parse the fitted booster afterwards.  ``max_depth=1`` is not used as a
substitute: a single feature may be split at several thresholds inside one tree.

Historical full-interaction boosters (XGB-ALL / ZX1) are *not* refit; their 100/250
round predictions are sliced from the saved booster with ``iteration_range``.
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

import sfl_common as K

R = 500
ROUND_REPORT = (100, 250, 500)
NTHREAD = 8

HIV_PARAMS: dict[str, Any] = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 5,
    "eta": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 5.0,
    "min_child_weight": 1,
    "tree_method": "hist",
    "seed": K.SEED,
    "nthread": NTHREAD,
}
ZINC_PARAMS: dict[str, Any] = {
    "objective": "reg:absoluteerror",
    "eval_metric": "mae",
    "max_depth": 5,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 5.0,
    "min_child_weight": 1,
    "tree_method": "hist",
    "seed": K.SEED,
    "nthread": NTHREAD,
}


# ---------------------------------------------------------------------------
# constraint checks
# ---------------------------------------------------------------------------


def feature_names(d: int) -> list[str]:
    return [f"c{i}" for i in range(d)]


def synthetic_constraint_check() -> dict[str, Any]:
    """Small synthetic fit proving the installed xgboost honours singleton constraints."""
    import xgboost as xgb

    rng = np.random.default_rng(0)
    n, p = 4000, 6
    X = rng.normal(size=(n, p)).astype(np.float32)
    y = (2.0 * X[:, 0] + np.sin(3.0 * X[:, 1]) - X[:, 2] ** 2
         + 0.5 * X[:, 3] * X[:, 4] > 0).astype(np.float32)
    names = feature_names(p)
    params = {"objective": "binary:logistic", "max_depth": 5, "eta": 0.3,
              "tree_method": "hist", "seed": 0, "nthread": 2,
              "interaction_constraints": [[nm] for nm in names]}
    dm = xgb.DMatrix(X, label=y, feature_names=names)
    b = xgb.train(params, dm, num_boost_round=30)
    dump = b.get_dump(with_stats=False)
    used = [len(feats_in_tree(t, names)) for t in dump]
    return {"n_trees": len(dump), "max_distinct_features_per_tree": int(max(used)),
            "n_trees_with_more_than_one_feature": int(sum(u > 1 for u in used)),
            "features_used": sorted({f for t in dump for f in feats_in_tree(t, names)}),
            "n_distinct_features_used": int(len({f for t in dump for f in feats_in_tree(t, names)})),
            "unconstrained_control_distinct_features": int(len(
                {f for t in xgb.train({k: v for k, v in params.items()
                                       if k != "interaction_constraints"}, dm,
                                      num_boost_round=30).get_dump()
                 for f in feats_in_tree(t, names)}))}


def feats_in_tree(dump: str, names: list[str]) -> set[str]:
    return set(re.findall(r"\[(" + "|".join(names) + r")<", dump))


def json_tree_stats(booster: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Parse the JSON dump: per-tree feature set, leaf counts, thresholds."""
    import json

    trees = json.loads(booster.get_dump(dump_format="json"))
    per_tree = []
    all_features: set[str] = set()
    feature_tree_counts: dict[str, int] = {}
    n_leaves_total = 0

    def walk(node: dict[str, Any], feats: set[str], leaves: list[int]) -> None:
        if "leaf" in node:
            leaves.append(1)
            return
        feats.add(str(node["split"]))
        for child in node["children"]:
            walk(child, feats, leaves)

    for i, tree in enumerate(trees):
        root = json.loads(tree)
        feats: set[str] = set()
        leaves: list[int] = []
        walk(root, feats, leaves)
        n_leaves = len(leaves)
        n_leaves_total += n_leaves
        all_features |= feats
        for f in feats:
            feature_tree_counts[f] = feature_tree_counts.get(f, 0) + 1
        per_tree.append({"tree": i, "features": sorted(feats),
                         "n_distinct_features": len(feats), "n_leaves": n_leaves,
                         "nonempty": bool(feats)})
    stats = {
        "n_trees": len(trees),
        "n_nonempty_trees": int(sum(t["nonempty"] for t in per_tree)),
        "n_empty_single_leaf_trees": int(sum(not t["nonempty"] for t in per_tree)),
        "n_leaves_total": int(n_leaves_total),
        "n_distinct_features_used": int(len(all_features)),
        "max_distinct_features_per_tree": int(max(t["n_distinct_features"] for t in per_tree)),
        "n_trees_violating_singleton": int(sum(t["n_distinct_features"] > 1 for t in per_tree)),
        "feature_tree_counts": feature_tree_counts,
        "per_tree": per_tree,
    }
    return per_tree, stats


def replay_check(booster: Any, X: np.ndarray, names: list[str], n_rows: int = 1000,
                 seed: int = K.SEED) -> dict[str, Any]:
    """Manually replay every tree on a row sample and compare with the booster."""
    import json

    import xgboost as xgb

    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(X.shape[0], size=min(n_rows, X.shape[0]), replace=False))
    trees = [json.loads(t) for t in booster.get_dump(dump_format="json")]
    manual = np.zeros(rows.size, dtype=np.float64)
    for tree in trees:
        nodes = {int(n["nodeid"]): n for n in _flatten(tree)}
        for i, r in enumerate(rows):
            node = nodes[0]
            while "leaf" not in node:
                col = names.index(str(node["split"]))
                nxt = (node["yes"] if X[r, col] < float(node["split_condition"])
                       else node["no"])
                node = nodes[int(nxt)]
            manual[i] += float(node["leaf"])
    pred = booster.predict(xgb.DMatrix(X[rows], feature_names=names), output_margin=True)
    residual = pred - manual
    return {"n_rows": int(rows.size), "n_trees": len(trees),
            "residual_mean": float(residual.mean()),
            "residual_max_abs_dev_from_mean": float(np.max(np.abs(residual - residual.mean()))),
            "note": "constant residual = base_score margin (identity link: 0 for reg:absoluteerror)"}


def _flatten(node: dict[str, Any]) -> list[dict[str, Any]]:
    out = [node]
    for child in node.get("children", []):
        out.extend(_flatten(child))
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["hiv", "zinc"])
    ap.add_argument("--out-dir", default=str(K.RUNS))
    ap.add_argument("--model-dir", default=str(K.MODELS))
    ap.add_argument("--rounds", type=int, default=R)
    args = ap.parse_args()
    import xgboost as xgb

    out_dir = Path(args.out_dir)
    model_dir = Path(args.model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    check = synthetic_constraint_check()
    if check["n_trees_with_more_than_one_feature"] > 0 or \
            check["max_distinct_features_per_tree"] > 1:
        raise SystemExit(f"ADDITIVE_TREE_BLOCKED: {check}")
    print(f"[B] singleton constraints honoured: {check}", flush=True)

    if args.dataset == "hiv":
        data = K.HIVData()
        arm = "H-ADD-TREE"
        params = dict(HIV_PARAMS)
        y = data.y.astype(np.float32)
        X = data.design_dense()
        rows = data.rows
        metric_fn = lambda pred: K.hiv_metrics(pred, data.y, rows)
    else:
        data = K.ZincData()
        arm = "Z-ADD-TREE"
        params = dict(ZINC_PARAMS)
        y = data.y.astype(np.float32)
        X = data.design_dense()
        rows = data.rows
        metric_fn = lambda pred: K.zinc_metrics(pred, data.y, rows)

    names = feature_names(X.shape[1])
    params["interaction_constraints"] = [[nm] for nm in names]
    print(f"[B] {arm} design {X.shape} ({X.nbytes / 1e9:.2f} GB float32)", flush=True)
    dm = {k: xgb.DMatrix(X[rows[k]], label=y[rows[k]], feature_names=names)
          for k in K.SPLITS}
    del X                               # design only needed for the DMatrices now
    t_fit = time.perf_counter()
    booster = xgb.train(params, dm["train"], num_boost_round=args.rounds)
    fit_seconds = time.perf_counter() - t_fit
    print(f"[B] {arm} fitted in {fit_seconds:.1f}s", flush=True)

    per_round: dict[str, Any] = {}
    preds_full: np.ndarray | None = None
    for r in ROUND_REPORT:
        if r > args.rounds:
            continue
        pred = np.zeros(y.size, dtype=np.float64)
        for k in K.SPLITS:
            p = booster.predict(dm[k], iteration_range=(0, r), output_margin=True)
            pred[rows[k]] = p
        m = metric_fn(pred)
        per_round[str(r)] = m
        if r == args.rounds:
            preds_full = pred
        key = "roc_auc" if args.dataset == "hiv" else "mae"
        print(f"[B] {arm} rounds={r} train {key}={m['train'][key]:.4f} "
              f"valid {key}={m['valid'][key]:.4f} test {key}={m['test'][key]:.4f}",
              flush=True)
    del dm
    assert preds_full is not None

    # ---------------------------------------------------------------- checks
    per_tree, tree_stats = json_tree_stats(booster)
    # replay on a deterministic small design slice (the full design is not kept)
    replay = replay_check(booster, _sample_design(args.dataset, rows["test"]), names)
    struct_lo = 2
    size_features = {"c0", "c1"}
    violations = {
        "multi_feature_trees": tree_stats["n_trees_violating_singleton"],
        "size_and_structure_same_tree": int(sum(
            bool(set(t["features"]) & size_features) and
            any(int(f[1:]) >= struct_lo for f in t["features"]) for t in per_tree)),
        "two_structures_same_tree": int(sum(
            len([f for f in t["features"] if int(f[1:]) >= struct_lo]) > 1
            for t in per_tree)),
    }
    if any(violations.values()):
        raise SystemExit(f"ADDITIVE_TREE_BLOCKED: violations {violations}")

    model_path = model_dir / f"{arm}.json"
    booster.save_model(str(model_path))
    run_model_path = out_dir / f"{arm}_model.json"
    booster.save_model(str(run_model_path))
    pred_path = K.PREDICTIONS / f"{arm}.npz" if args.dataset == "hiv" else None
    if pred_path is not None:
        K.PREDICTIONS.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(pred_path, logit=preds_full, pred=preds_full,
                            y=data.y, graph_id=np.arange(data.n),
                            split_code=data.split_code(), arm=arm, dataset="hiv")
    else:
        K.PREDICTIONS.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(K.PREDICTIONS / f"{arm}.npz", pred=preds_full, y=data.y,
                            graph_id=np.arange(data.n), arm=arm, dataset="zinc")

    usage = sorted(tree_stats["feature_tree_counts"].items(),
                   key=lambda kv: (-kv[1], kv[0]))
    payload = {
        "round_id": K.ROUND_ID, "phase": "B: additive tree", "arm": arm,
        "dataset": args.dataset,
        "params": {k: v for k, v in params.items()
                   if k != "interaction_constraints"},
        "interaction_constraints": "singleton [[c0],...,[c{d-1}]] (xgboost feature names)",
        "n_rounds": args.rounds, "n_features": int(len(names)),
        "fit_seconds": float(fit_seconds),
        "synthetic_constraint_check": check,
        "tree_stats": {k: v for k, v in tree_stats.items() if k != "per_tree"},
        "tree_violations": violations,
        "replay_check": replay,
        "feature_usage_top": [{"feature": f, "n_trees": int(c)} for f, c in usage[:40]],
        "metrics_by_rounds": per_round,
        "booster_path": K.rel(model_path),
        "booster_sha256": K.sha256_file(model_path),
        "input_identity": {
            "n_rows": int(data.n), "n_columns": int(len(names)),
            "rows_sha256": {k: K.sha256_array(rows[k]) for k in K.SPLITS},
            "y_sha256": K.sha256_array(np.asarray(data.y)),
            "design_identity": ("bundle/sav1_inputs.npz C+F, n_bonds from tracked raw graph"
                               if args.dataset == "hiv" else
                               "frozen SGA-v1 Z+Z-CF design (Z 4642 + C 9 + F 1250)"),
        },
        "seconds": float(time.perf_counter() - t0),
        "git": K.git_state(), "environment": K.environment(),
    }
    K.write_json(out_dir / f"{arm}.json", payload)
    print(f"[B] {arm} done: nonempty={tree_stats['n_nonempty_trees']} "
          f"leaves={tree_stats['n_leaves_total']} "
          f"distinct_features={tree_stats['n_distinct_features_used']} "
          f"replay_maxdev={replay['residual_max_abs_dev_from_mean']:.2e} "
          f"total {payload['seconds']:.1f}s", flush=True)


def _sample_design(dataset: str, rows: np.ndarray) -> np.ndarray:
    """A small deterministic slice of the design for the manual tree replay."""
    take = rows[:1000]
    if dataset == "hiv":
        d = K.HIVData()
        return d.design_dense(take)
    d = K.ZincData()
    return d.design_dense(take)


if __name__ == "__main__":
    main()
