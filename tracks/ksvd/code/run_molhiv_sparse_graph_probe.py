"""Selection-only convex probe over rich sparse dictionary graph descriptors."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

_TRACK = Path(__file__).resolve().parents[1]
if str(_TRACK) not in sys.path:
    sys.path.insert(0, str(_TRACK))

from code.data_molhiv import check_env, load_molhiv


def array_hash(a: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()


def rich_features(tokens: np.ndarray, dictionary: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    atom_count = int(tokens.shape[1])
    gram = dictionary.astype(np.float64).T @ dictionary.astype(np.float64)
    rows = []
    for graph_i in range(len(offsets) - 1):
        lo, hi = int(offsets[graph_i]), int(offsets[graph_i + 1])
        codes = tokens[lo:hi].astype(np.float64, copy=False)
        absolute = np.abs(codes)
        n_nodes = int(len(codes))
        if n_nodes:
            quantiles = np.quantile(absolute, [0.75, 0.90], axis=0)
            top_count = min(3, n_nodes)
            top_mean = np.partition(absolute, n_nodes - top_count, axis=0)[
                n_nodes - top_count:
            ].mean(axis=0)
            winner = np.bincount(
                np.argmax(absolute, axis=1), minlength=atom_count
            ).astype(np.float64) / n_nodes
            blocks = [
                absolute.mean(axis=0), absolute.max(axis=0), top_mean,
                absolute.std(axis=0), (absolute > 1e-10).mean(axis=0),
                quantiles[0], quantiles[1], np.square(codes).mean(axis=0),
                codes.mean(axis=0), winner,
            ]
            explained = np.einsum(
                "ni,ij,nj->n", codes, gram, codes, optimize=True
            )
            errors = np.sqrt(np.maximum(1.0 - explained, 0.0))
            error_features = np.asarray([
                errors.mean(), errors.std(),
                *np.quantile(errors, [0.50, 0.75, 0.90]),
                errors.max(), float(n_nodes), np.log1p(n_nodes),
            ])
        else:
            blocks = [np.zeros(atom_count) for _ in range(10)]
            error_features = np.zeros(8)
        rows.append(np.concatenate([*blocks, error_features]))
    return np.stack(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--token-cache", required=True)
    ap.add_argument("--family", choices=("ksvd", "pca", "random_patch"), required=True)
    ap.add_argument("--inner-split-cache", required=True)
    ap.add_argument("--inner-fold", type=int, required=True)
    ap.add_argument("--max-graphs", type=int, default=8000)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--c", type=float, default=0.1)
    ap.add_argument(
        "--c-selection", choices=("fixed", "scaffold_cv"), default="fixed",
        help="use fixed --c or select from --c-grid using only outer-fit scaffold groups",
    )
    ap.add_argument(
        "--c-grid", default="0.003,0.01,0.03,0.1,0.3,1.0,3.0",
        help="comma-separated positive C values for outer-fit-only scaffold CV",
    )
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if args.c <= 0:
        raise ValueError("--c must be positive")
    c_grid = [float(value) for value in args.c_grid.split(",") if value.strip()]
    if not c_grid or any(value <= 0 for value in c_grid):
        raise ValueError("--c-grid must contain positive values")

    t0 = time.time()
    repo = Path(__file__).resolve().parents[3]
    max_graphs = None if args.max_graphs <= 0 else args.max_graphs
    bundle = load_molhiv(
        root=repo / "data" / "ogb", max_graphs=max_graphs,
        seed=args.data_seed, with_features=False,
    )
    y = np.asarray(bundle.y, dtype=np.int64)
    official_train = np.asarray(bundle.split["train"], dtype=np.int64)
    official_valid = np.asarray(bundle.split["valid"], dtype=np.int64)
    with np.load(args.inner_split_cache, allow_pickle=False) as z:
        fit = np.asarray(z[f"fold_{args.inner_fold}_train_indices"], dtype=np.int64)
        valid = np.asarray(z[f"fold_{args.inner_fold}_valid_indices"], dtype=np.int64)
        scaffold_valid_groups = {
            int(name.split("_")[1]): np.asarray(z[name], dtype=np.int64)
            for name in z.files
            if name.startswith("fold_") and name.endswith("_valid_indices")
        }
        if "official_train_indices" in z.files and not np.array_equal(
            official_train, np.asarray(z["official_train_indices"], dtype=np.int64)
        ):
            raise ValueError("fold cache official train mismatch")
    if set(fit.tolist()) & set(valid.tolist()):
        raise ValueError("inner train/valid overlap")
    if set(fit.tolist()) | set(valid.tolist()) != set(official_train.tolist()):
        raise ValueError("inner fold does not partition official train")

    with np.load(args.token_cache, allow_pickle=False) as z:
        offsets = np.asarray(z["offsets"], dtype=np.int64)
        original_indices = np.asarray(z["original_indices"], dtype=np.int64)
        expected = np.asarray(bundle.meta["original_indices"], dtype=np.int64)
        if not np.array_equal(original_indices, expected):
            raise ValueError("token cache subset mismatch")
        tokens = np.asarray(z[f"tokens_{args.family}"], dtype=np.float32)
        dictionary = np.asarray(z[f"dictionary_{args.family}"], dtype=np.float64)
    features = rich_features(tokens, dictionary, offsets)
    mean = features[fit].mean(axis=0, keepdims=True)
    scale = np.maximum(features[fit].std(axis=0, keepdims=True), 1e-6)
    scaled = (features - mean) / scale

    selected_c = args.c
    c_selection_history = None
    if args.c_selection == "scaffold_cv":
        fold_ids = sorted(scaffold_valid_groups)
        fit_groups = [fold_id for fold_id in fold_ids if fold_id != args.inner_fold]
        if len(fit_groups) < 2:
            raise ValueError("scaffold_cv requires at least two outer-fit scaffold groups")
        c_selection_history = []
        for candidate_c in c_grid:
            split_aucs = []
            for heldout_group in fit_groups:
                cv_valid = scaffold_valid_groups[heldout_group]
                cv_train = np.setdiff1d(fit, cv_valid, assume_unique=False)
                if not set(cv_valid.tolist()) <= set(fit.tolist()):
                    raise ValueError("scaffold CV validation group is outside outer fit")
                cv_clf = LogisticRegression(
                    penalty="l2", C=candidate_c, class_weight="balanced",
                    solver="lbfgs", max_iter=2000, tol=1e-8, random_state=0,
                )
                cv_clf.fit(scaled[cv_train], y[cv_train])
                split_aucs.append(float(roc_auc_score(
                    y[cv_valid], cv_clf.decision_function(scaled[cv_valid])
                )))
            c_selection_history.append({
                "c": candidate_c, "split_aucs": split_aucs,
                "mean_auc": float(np.mean(split_aucs)),
            })
        selected_c = min(
            c_selection_history, key=lambda row: (-row["mean_auc"], row["c"])
        )["c"]

    clf = LogisticRegression(
        penalty="l2", C=selected_c, class_weight="balanced", solver="lbfgs",
        max_iter=2000, tol=1e-8, random_state=0,
    )
    clf.fit(scaled[fit], y[fit])
    valid_score = clf.decision_function(scaled[valid])
    auc = float(roc_auc_score(y[valid], valid_score))
    coef = clf.coef_.reshape(-1)
    result = {
        "protocol_id": "molhiv-rich-sparse-graph-probe-selection-only-v1",
        "family": args.family,
        "inner_fold": args.inner_fold,
        "inner_split_cache": args.inner_split_cache,
        "token_cache": args.token_cache,
        "feature_dim": int(features.shape[1]),
        "classifier": "balanced_l2_logistic_regression",
        "c": selected_c,
        "requested_c": args.c,
        "c_selection": args.c_selection,
        "c_grid": c_grid if args.c_selection == "scaffold_cv" else None,
        "c_selection_history": c_selection_history,
        "solver": "lbfgs",
        "max_iter": 2000,
        "tol": 1e-8,
        "inner_valid_auc": auc,
        "n_fit": int(len(fit)),
        "n_valid": int(len(valid)),
        "n_fit_pos": int(y[fit].sum()),
        "n_valid_pos": int(y[valid].sum()),
        "trainable_parameter_count": int(coef.size + 1),
        "coefficient_norm": float(np.linalg.norm(coef)),
        "coefficient_abs_max": float(np.abs(coef).max()),
        "intercept": float(clf.intercept_[0]),
        "iterations": int(clf.n_iter_[0]),
        "coefficient": coef.tolist(),
        "official_valid_auc": None,
        "official_valid_evaluations": 0,
        "official_test_auc": None,
        "official_test_evaluations": 0,
        "test_policy": "official valid and official test were not evaluated",
        "audit_fingerprints": {
            "official_train_sha256": array_hash(official_train),
            "official_valid_sha256": array_hash(official_valid),
            "inner_train_sha256": array_hash(fit),
            "inner_valid_sha256": array_hash(valid),
            "feature_mean_sha256": array_hash(mean.astype(np.float32)),
            "feature_scale_sha256": array_hash(scale.astype(np.float32)),
        },
        "elapsed_sec": time.time() - t0,
        "env": check_env(),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"family={args.family} fold={args.inner_fold} auc={auc:.6f}; wrote {output}")


if __name__ == "__main__":
    main()
