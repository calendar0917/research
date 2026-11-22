"""Task 3: two light statistical references, Train-sub -> Dev transfer."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

import audit_common as A
import audit_features as AF


def _shrunk_mean(values: np.ndarray, global_mean: float) -> float:
    n = int(values.size)
    if n == 0:
        return float(global_mean)
    return float((np.sum(values) + A.SHRINK_K * global_mean) / (n + A.SHRINK_K))


def _predict_a(pairs: A.PairTable, keys: np.ndarray, global_mean: float
               ) -> tuple[np.ndarray, np.ndarray, dict[Any, np.ndarray], int]:
    train = np.nonzero(pairs.scope == 0)[0]
    groups = A.group_positions(keys, train)
    n = pairs.a.size
    a_full = np.full(n, float(global_mean), np.float64)
    a_lomo = np.full(n, float(global_mean), np.float64)
    n_excluded = 0
    for j in range(n):
        key = A.hashable_key(keys[j])
        pos = groups.get(key, np.zeros(0, np.int64))
        if pos.size == 0:
            continue
        a_full[j] = _shrunk_mean(pairs.d[pos], global_mean)
        mols = {int(pairs.a[j]), int(pairs.b[j])}
        keep = [int(x) for x in pos.tolist()
                if int(pairs.a[x]) not in mols and int(pairs.b[x]) not in mols]
        if int(pairs.scope[j]) == 0:
            n_excluded += int(pos.size - len(keep))
        a_lomo[j] = _shrunk_mean(pairs.d[np.asarray(keep, np.int64)], global_mean) if keep else float(global_mean)
    return a_full, a_lomo, groups, n_excluded


def _fit_b(pairs: A.PairTable, F: np.ndarray, a_lomo: np.ndarray,
           compact_dim: int, lam: float) -> dict[str, Any]:
    train = np.nonzero(pairs.scope == 0)[0]
    X = F[train].copy()
    sd = X.std(0)
    keep = sd > 1e-8
    X = X[:, keep]
    mu = X.mean(0)
    sd = X.std(0)
    sd[sd == 0] = 1.0
    X = (X - mu) / sd
    r = pairs.d[train] - a_lomo[train]
    # component-balanced weights: each Train-sub pair component contributes equally
    comp = pairs.comp_train[train]
    counts = {int(c): int((comp == c).sum()) for c in np.unique(comp)}
    w = np.asarray([1.0 / counts[int(c)] for c in comp.tolist()], np.float64)
    w = w / w.mean()
    r_mean = float(np.average(r, weights=w))
    r_centered = r - r_mean
    beta = np.linalg.solve(X.T @ (X * w[:, None]) + float(lam) * np.eye(X.shape[1]),
                           X.T @ (w * r_centered))
    X_all = (F[:, keep] - mu) / sd
    correction = X_all @ beta
    train_pred = a_lomo[train] + correction[train]
    train_mae = float(np.mean(np.abs(train_pred - pairs.d[train])))
    train_rmse = float(np.sqrt(np.mean((train_pred - pairs.d[train]) ** 2)))
    resid = pairs.d[train] - a_lomo[train]
    sst = float(np.sum((resid - np.average(resid, weights=w)) ** 2))
    sse = float(np.sum(w * (resid - r_mean - correction[train]) ** 2))
    w_r2 = 1.0 - sse / sst if sst > 0 else float("nan")
    return {
        "beta": beta,
        "correction": correction,
        "keep_columns": keep,
        "mu": mu,
        "sd": sd,
        "r_mean": r_mean,
        "train_mae": train_mae,
        "train_rmse": train_rmse,
        "train_residual_weighted_r2": w_r2,
        "lambda": float(lam),
        "n_train_pairs": int(train.size),
        "n_raw_features": int(F.shape[1]),
        "n_used_features": int(keep.sum()),
        "component_weight_range": [float(w.min()), float(w.max())],
        "n_components": int(len(counts)),
        "beta_norm": float(np.linalg.norm(beta)),
    }


def _fit_b_compact(pairs: A.PairTable, F: np.ndarray, a_lomo: np.ndarray,
                   compact_dim: int, lam: float) -> dict[str, Any]:
    return _fit_b(pairs, F[:, :int(compact_dim)], a_lomo, compact_dim, lam)


def _subset_metrics(y: np.ndarray, predictors: Mapping[str, np.ndarray],
                    clusters: np.ndarray, mask: np.ndarray | None,
                    reps: int) -> dict[str, Any]:
    if mask is not None:
        mask = np.asarray(mask, bool)
        if not mask.any():
            return {"n_pairs": 0}
        y2 = y[mask]
        clusters2 = clusters[mask]
        preds2 = {k: v[mask] for k, v in predictors.items()}
    else:
        y2, clusters2, preds2 = y, clusters, dict(predictors)
    if np.unique(clusters2).size < 2:
        return {
            "n_pairs": int(y2.size),
            "n_clusters": int(np.unique(clusters2).size),
            "point": {k: A.delta_metrics(y2, v) for k, v in preds2.items()},
            "ci_reliable": False,
        }
    boot = A.cluster_bootstrap(y2, preds2, clusters2, reps=reps, seed=A.BOOTSTRAP_SEED)
    boot["ci_reliable"] = bool(boot["n_clusters"] >= 5)
    return boot


def _paired(y: np.ndarray, a: np.ndarray, b: np.ndarray, clusters: np.ndarray,
            mask: np.ndarray | None, reps: int) -> dict[str, Any]:
    if mask is not None and not np.asarray(mask, bool).any():
        return {"n_pairs": 0}
    return A.paired_error_bootstrap(y, a, b, clusters, reps=reps, seed=A.BOOTSTRAP_SEED, mask=mask)


def _calibration(y: np.ndarray, pred: np.ndarray, clusters: np.ndarray,
                 reps: int) -> dict[str, Any]:
    y = np.asarray(y, np.float64)
    pred = np.asarray(pred, np.float64)
    if y.size < 3 or np.std(pred) == 0:
        return {"n": int(y.size), "slope": None, "intercept": None, "r2": None}
    slope = float(np.cov(pred, y, ddof=0)[0, 1] / np.var(pred))
    intercept = float(y.mean() - slope * pred.mean())
    pred_line = intercept + slope * pred
    r2 = float(1.0 - np.sum((y - pred_line) ** 2) / np.sum((y - y.mean()) ** 2))
    uniq = np.unique(clusters)
    members = {int(c): np.nonzero(clusters == c)[0] for c in uniq}
    rng = np.random.default_rng(A.BOOTSTRAP_SEED)
    slopes = []
    for _ in range(int(reps)):
        picks = uniq[rng.integers(0, uniq.size, size=uniq.size)]
        idx = np.concatenate([members[int(c)] for c in picks])
        yy, pp = y[idx], pred[idx]
        if np.std(pp) == 0:
            continue
        slopes.append(float(np.cov(pp, yy, ddof=0)[0, 1] / np.var(pp)))
    return {
        "n": int(y.size),
        "slope": slope,
        "slope_ci": A.percentile_ci(np.asarray(slopes, np.float64)),
        "intercept": intercept,
        "r2": r2,
        "sd_ratio_pred_over_true": float(np.std(pred) / np.std(y)) if np.std(y) > 0 else None,
        "mean_true": float(y.mean()),
        "mean_pred": float(pred.mean()),
    }


def _support_bin(n: int) -> str:
    if n <= 0:
        return "unseen"
    if n == 1:
        return "1"
    if n == 2:
        return "2"
    if n <= 4:
        return "3-4"
    if n <= 9:
        return "5-9"
    return ">=10"


def _ctx_bin(n: int) -> str:
    if n <= 0:
        return "unseen"
    if n == 1:
        return "1"
    if n == 2:
        return "2"
    if n <= 4:
        return "3-4"
    return ">=5"


def _word_strings(inp: A.Inputs) -> np.ndarray:
    return np.asarray([str(x) for x in inp.trc["fragment_keys"]])


def _failure_rows(inp: A.Inputs, pairs: A.PairTable, pos: np.ndarray,
                  y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
                  keys: np.ndarray, groups: Mapping[Any, np.ndarray],
                  seen_ctx: set[int], n: int = 15) -> list[dict[str, Any]]:
    words = _word_strings(inp)
    err_a = np.abs(pred_a - y)
    order = np.argsort(-err_a)[:int(n)]
    rows = []
    for k in order.tolist():
        j = int(pos[k])
        key = A.hashable_key(keys[j])
        pos_train = groups.get(key, np.zeros(0, np.int64))
        n_ctx = len(set(int(x) for x in pairs.C[pos_train].tolist())) if pos_train.size else 0
        rows.append({
            "pair_row": int(pairs.idx[j]),
            "a": int(pairs.a[j]),
            "b": int(pairs.b[j]),
            "type_seen": bool(pos_train.size > 0),
            "n_train_type_pairs": int(pos_train.size),
            "n_train_type_contexts": int(n_ctx),
            "ctx_full_seen": bool(int(pairs.C[j]) in seen_ctx),
            "delta_true": float(pairs.d[j]),
            "pred_a": float(pred_a[k]),
            "pred_b": float(pred_b[k]),
            "err_a": float(err_a[k]),
            "err_b": float(abs(pred_b[k] - y[k])),
            "delta_err_b_minus_a": float(abs(pred_b[k] - y[k]) - err_a[k]),
            "sa_text": words[int(pairs.Sa[j])][:160],
            "sb_text": words[int(pairs.Sb[j])][:160],
        })
    return rows


def run(inp: A.Inputs, pairs: A.PairTable, features: AF.FeatureSet) -> dict[str, Any]:
    F = features.F
    train = np.nonzero(pairs.scope == 0)[0]
    cross = np.nonzero(pairs.scope == 1)[0]
    dd = np.nonzero(pairs.scope == 2)[0]
    combined = np.nonzero(pairs.scope != 0)[0]
    g = float(pairs.d[train].mean())

    def make(keys: np.ndarray, lam: float = A.RIDGE_LAMBDA,
             use_compact: bool = False) -> dict[str, Any]:
        a_full, a_lomo, groups, n_excluded = _predict_a(pairs, keys, g)
        fit = (_fit_b_compact if use_compact else _fit_b)(
            pairs, F, a_lomo, features.compact_dim, lam)
        b = a_lomo + fit["correction"]
        b_full_offset = a_full + fit["correction"]
        return {
            "keys": keys,
            "a_full": a_full,
            "a_lomo": a_lomo,
            "b": b,
            "b_full_offset": b_full_offset,
            "groups": groups,
            "n_lomo_excluded": int(n_excluded),
            "fit": fit,
        }

    exact = make(pairs.T)
    exact_compact = make(pairs.T, use_compact=True)
    exact_lam100 = make(pairs.T, lam=100.0)
    fit_self = _fit_b(pairs, F, exact["a_full"], features.compact_dim, A.RIDGE_LAMBDA)
    exact_self = {
        "b": exact["a_full"] + fit_self["correction"],
        "fit": fit_self,
    }
    coarse = make(pairs.coarse)

    # clusters for each evaluation scope
    cross_clusters = np.asarray([int(pairs.a[j]) if inp.dev[int(pairs.a[j])] else int(pairs.b[j])
                                 for j in cross.tolist()], np.int64)
    dd_clusters = np.asarray([int(pairs.comp_all[j]) for j in dd.tolist()], np.int64)
    combined_clusters = np.asarray([int(pairs.comp_all[j]) for j in combined.tolist()], np.int64)

    def scope_block(scope_idx: np.ndarray, clusters: np.ndarray,
                    model: Mapping[str, Any], cluster_unit: str) -> dict[str, Any]:
        y = pairs.d[scope_idx]
        preds = {
            "zero": np.zeros_like(y),
            "global": np.full_like(y, g),
            "A_full": model["a_full"][scope_idx],
            "A_lomo": model["a_lomo"][scope_idx],
            "B": model["b"][scope_idx],
        }
        boot = A.cluster_bootstrap(y, preds, clusters, reps=A.BOOTSTRAP_REPS,
                                   seed=A.BOOTSTRAP_SEED)
        boot["cluster_unit"] = cluster_unit
        return boot

    scopes = {
        "cross": scope_block(cross, cross_clusters, exact, "Dev molecule"),
        "dev_dev": scope_block(dd, dd_clusters, exact, "full Train pair component"),
        "combined": scope_block(combined, combined_clusters, exact, "full Train pair component"),
    }

    # seen/unseen and context subsets, evaluated with the same scope clusters
    tr_types = set(int(x) for x in pairs.T[train].tolist())
    tr_ctx = set(int(x) for x in pairs.C[train].tolist())
    tr_r1 = set(int(x) for x in pairs.r1e[train].tolist())
    tr_r2 = set(int(x) for x in pairs.r2e[train].tolist())
    ctx2_types = set()
    for key, pos in exact["groups"].items():
        if len(set(int(x) for x in pairs.C[pos].tolist())) >= 2:
            ctx2_types.add(key)

    def subset_mask(scope_idx: np.ndarray, name: str) -> np.ndarray:
        if name == "seen_type":
            return np.asarray([int(pairs.T[j]) in tr_types for j in scope_idx.tolist()], bool)
        if name == "unseen_type":
            return np.asarray([int(pairs.T[j]) not in tr_types for j in scope_idx.tolist()], bool)
        if name == "seen_type_ctx2":
            return np.asarray([A.hashable_key(pairs.T[j]) in ctx2_types for j in scope_idx.tolist()], bool)
        if name == "seen_ctx_full":
            return np.asarray([int(pairs.C[j]) in tr_ctx for j in scope_idx.tolist()], bool)
        if name == "unseen_ctx_full":
            return np.asarray([int(pairs.C[j]) not in tr_ctx for j in scope_idx.tolist()], bool)
        if name == "seen_r1":
            return np.asarray([int(pairs.r1e[j]) in tr_r1 for j in scope_idx.tolist()], bool)
        if name == "seen_r2":
            return np.asarray([int(pairs.r2e[j]) in tr_r2 for j in scope_idx.tolist()], bool)
        raise KeyError(name)

    subset_names = ("seen_type", "unseen_type", "seen_type_ctx2",
                    "seen_ctx_full", "unseen_ctx_full", "seen_r1", "seen_r2")
    subsets: dict[str, Any] = {}
    for scope_name, scope_idx, clusters in (
            ("cross", cross, cross_clusters),
            ("dev_dev", dd, dd_clusters),
            ("combined", combined, combined_clusters)):
        subsets[scope_name] = {}
        y = pairs.d[scope_idx]
        preds = {
            "global": np.full_like(y, g),
            "A_lomo": exact["a_lomo"][scope_idx],
            "B": exact["b"][scope_idx],
        }
        for name in subset_names:
            m = subset_mask(scope_idx, name)
            block = _subset_metrics(y, preds, clusters, m, A.BOOTSTRAP_REPS)
            block["paired_B_minus_A"] = _paired(y, preds["A_lomo"], preds["B"], clusters, m,
                                                A.BOOTSTRAP_REPS)
            subsets[scope_name][name] = block

    # support bins: type pair count and type full-context count in Train-sub
    support_bins: dict[str, Any] = {}
    for scope_name, scope_idx, clusters in (
            ("cross", cross, cross_clusters),
            ("dev_dev", dd, dd_clusters)):
        y = pairs.d[scope_idx]
        preds = {"global": np.full_like(y, g),
                 "A_lomo": exact["a_lomo"][scope_idx],
                 "B": exact["b"][scope_idx]}
        for kind, fn in (("type_pairs", lambda j: int(exact["groups"].get(A.hashable_key(pairs.T[j]), np.zeros(0)).size)),
                         ("type_contexts", lambda j: len(set(int(x) for x in pairs.C[exact["groups"].get(A.hashable_key(pairs.T[j]), np.zeros(0, np.int64))].tolist())))):
            bins: dict[str, Any] = {}
            labels = np.asarray([_support_bin(fn(j)) if kind == "type_pairs" else _ctx_bin(fn(j))
                                 for j in scope_idx.tolist()], dtype=object)
            for label in ("unseen", "1", "2", "3-4", "5-9", ">=10") if kind == "type_pairs" else ("unseen", "1", "2", "3-4", ">=5"):
                m = np.asarray([x == label for x in labels.tolist()], bool)
                if not m.any():
                    continue
                block = _subset_metrics(y, preds, clusters, m, A.BOOTSTRAP_REPS)
                block["paired_B_minus_A"] = _paired(y, preds["A_lomo"], preds["B"], clusters, m,
                                                    A.BOOTSTRAP_REPS)
                bins[label] = block
            support_bins[f"{scope_name}_{kind}"] = bins

    # calibration of A and B on the two held-out scopes
    calibration = {}
    for scope_name, scope_idx, clusters in (("cross", cross, cross_clusters),
                                            ("dev_dev", dd, dd_clusters)):
        calibration[scope_name] = {
            "A_lomo": _calibration(pairs.d[scope_idx], exact["a_lomo"][scope_idx], clusters,
                                   A.BOOTSTRAP_REPS),
            "B": _calibration(pairs.d[scope_idx], exact["b"][scope_idx], clusters,
                              A.BOOTSTRAP_REPS),
        }
        m = subset_mask(scope_idx, "seen_type")
        if m.sum() >= 3:
            calibration[scope_name]["A_lomo_seen_type"] = _calibration(
                pairs.d[scope_idx][m], exact["a_lomo"][scope_idx][m], clusters[m],
                A.BOOTSTRAP_REPS)
            calibration[scope_name]["B_seen_type"] = _calibration(
                pairs.d[scope_idx][m], exact["b"][scope_idx][m], clusters[m],
                A.BOOTSTRAP_REPS)

    # paired comparisons on the primary scopes and key subsets
    paired = {}
    for scope_name, scope_idx, clusters in (("cross", cross, cross_clusters),
                                            ("dev_dev", dd, dd_clusters),
                                            ("combined", combined, combined_clusters)):
        y = pairs.d[scope_idx]
        a = exact["a_lomo"][scope_idx]
        b = exact["b"][scope_idx]
        paired[scope_name] = {
            "overall": _paired(y, a, b, clusters, None, A.BOOTSTRAP_REPS),
        }
        for name in ("seen_type", "unseen_type", "seen_type_ctx2", "seen_ctx_full"):
            m = subset_mask(scope_idx, name)
            paired[scope_name][name] = _paired(y, a, b, clusters, m, A.BOOTSTRAP_REPS)

    paired_a_global = {}
    for scope_name, scope_idx, clusters in (("cross", cross, cross_clusters),
                                            ("dev_dev", dd, dd_clusters),
                                            ("combined", combined, combined_clusters)):
        y = pairs.d[scope_idx]
        gpred = np.full_like(y, g)
        a = exact["a_lomo"][scope_idx]
        paired_a_global[scope_name] = {
            "overall": _paired(y, gpred, a, clusters, None, A.BOOTSTRAP_REPS),
        }
        for name in ("seen_type", "unseen_type", "seen_type_ctx2", "seen_ctx_full"):
            m = subset_mask(scope_idx, name)
            paired_a_global[scope_name][name] = _paired(y, gpred, a, clusters, m,
                                                         A.BOOTSTRAP_REPS)

    # coarse-type sensitivity: same two references, coarser key
    coarse_block = {}
    for scope_name, scope_idx, clusters in (("cross", cross, cross_clusters),
                                            ("dev_dev", dd, dd_clusters)):
        y = pairs.d[scope_idx]
        preds = {"global": np.full_like(y, g),
                 "A_lomo": coarse["a_lomo"][scope_idx],
                 "B": coarse["b"][scope_idx]}
        block = _subset_metrics(y, preds, clusters, None, A.BOOTSTRAP_REPS)
        block["paired_B_minus_A"] = _paired(y, preds["A_lomo"], preds["B"], clusters, None,
                                            A.BOOTSTRAP_REPS)
        seen = np.asarray([A.hashable_key(pairs.coarse[j]) in coarse["groups"] for j in scope_idx.tolist()], bool)
        block["seen_type"] = _subset_metrics(y, preds, clusters, seen, A.BOOTSTRAP_REPS)
        block["unseen_type"] = _subset_metrics(y, preds, clusters, ~seen, A.BOOTSTRAP_REPS)
        coarse_block[scope_name] = block

    # post-hoc regularisation robustness (not used for model selection)
    lam100 = {}
    for scope_name, scope_idx, clusters in (("cross", cross, cross_clusters),
                                            ("dev_dev", dd, dd_clusters)):
        y = pairs.d[scope_idx]
        preds = {"A_lomo": exact["a_lomo"][scope_idx],
                 "B_lambda100": exact_lam100["b"][scope_idx]}
        block = _subset_metrics(y, preds, clusters, None, A.BOOTSTRAP_REPS)
        block["paired_B_minus_A"] = _paired(y, preds["A_lomo"], preds["B_lambda100"], clusters, None,
                                            A.BOOTSTRAP_REPS)
        lam100[scope_name] = block

    self_included = {}
    for scope_name, scope_idx, clusters in (("cross", cross, cross_clusters),
                                            ("dev_dev", dd, dd_clusters)):
        y = pairs.d[scope_idx]
        preds = {"A_lomo": exact["a_lomo"][scope_idx],
                 "B_self_included_offset": exact_self["b"][scope_idx]}
        block = _subset_metrics(y, preds, clusters, None, A.BOOTSTRAP_REPS)
        block["paired_B_minus_A"] = _paired(y, preds["A_lomo"], preds["B_self_included_offset"],
                                            clusters, None, A.BOOTSTRAP_REPS)
        self_included[scope_name] = block

    # compact-only B ablation
    compact = {}
    for scope_name, scope_idx, clusters in (("cross", cross, cross_clusters),
                                            ("dev_dev", dd, dd_clusters)):
        y = pairs.d[scope_idx]
        preds = {"A_lomo": exact_compact["a_lomo"][scope_idx],
                 "B_compact": exact_compact["b"][scope_idx]}
        block = _subset_metrics(y, preds, clusters, None, A.BOOTSTRAP_REPS)
        block["paired_B_minus_A"] = _paired(y, preds["A_lomo"], preds["B_compact"], clusters, None,
                                            A.BOOTSTRAP_REPS)
        compact[scope_name] = block

    # failure cases: worst A errors and worst B-A degradations on cross pairs
    all_cross_fail = _failure_rows(inp, pairs, cross, pairs.d[cross],
                                   exact["a_lomo"][cross], exact["b"][cross],
                                   exact["keys"], exact["groups"], tr_ctx, n=len(cross))

    def dedupe(rows: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
        seen: set[tuple[int, int]] = set()
        out: list[dict[str, Any]] = []
        for r in rows:
            key = (min(r["a"], r["b"]), max(r["a"], r["b"]))
            if key in seen:
                continue
            seen.add(key)
            out.append(r)
            if len(out) >= limit:
                break
        return out

    cross_fail_a = dedupe(sorted(all_cross_fail, key=lambda r: -r["err_a"]))
    cross_degrade = dedupe(sorted(
        [r for r in all_cross_fail if r["delta_err_b_minus_a"] > 0],
        key=lambda r: -r["delta_err_b_minus_a"]))
    seen_fail = [r for r in all_cross_fail if r["type_seen"]]
    seen_fail_a = dedupe(sorted(seen_fail, key=lambda r: -r["err_a"]))
    seen_degrade = dedupe(sorted(
        [r for r in seen_fail if r["delta_err_b_minus_a"] > 0],
        key=lambda r: -r["delta_err_b_minus_a"]))

    payload = {
        "meta": A.meta_block(inp, "structural-response-learnability-audit-v1/transfer", {
            "source_pairs": A.relative(A.TRC_CACHE),
            "train_scope": "Train-sub--Train-sub real matched pairs only",
            "eval_scopes": "Train-sub--Dev cross pairs (primary) and Dev--Dev pairs (secondary)",
            "no_dev_pair_in_training": True,
        }),
        "models": {
            "A_transformation_only": {
                "definition": "seen exact type: Train-sub empirical mean delta shrunk to the Train-sub global mean; unseen type: global mean",
                "shrink_k": A.SHRINK_K,
                "global_mean": g,
                "n_train_pairs": int(train.size),
                "n_train_types": int(len(exact["groups"])),
                "support_policy": "lomo: exclude Train-sub pairs sharing either molecule with the evaluation pair; full reported as a secondary anchor",
                "n_lomo_excluded_cross": int(sum(1 for j in cross.tolist()
                                                 if exact["a_lomo"][j] != exact["a_full"][j])),
                "n_lomo_excluded_dev_dev": int(sum(1 for j in dd.tolist()
                                                   if exact["a_lomo"][j] != exact["a_full"][j])),
                "lomo_note": ("Train-sub pairs contain only Train-sub molecules and Dev pairs contain "
                              "at least one Dev molecule, so no Train-sub pair can share a molecule "
                              "with an evaluation pair; A_lomo and A_full coincide on both eval scopes."),
            },
            "B_context_conditioned_linear": {
                "definition": "A + ridge context correction using fixed radius-1/radius-2 environment descriptors",
                "lambda": A.RIDGE_LAMBDA,
                "feature_blocks": "compact composition/scalars + rooted WL fingerprint (2 iterations, 256 bits per ball), radius-1 and radius-2",
                "n_raw_features": int(features.F.shape[1]),
                "n_used_features": int(exact["fit"]["n_used_features"]),
                "n_train_pairs": int(exact["fit"]["n_train_pairs"]),
                "n_components": int(exact["fit"]["n_components"]),
                "component_weight_range": exact["fit"]["component_weight_range"],
                "correction_centered": True,
                "in_sample": {
                    "A_full_self_included_mae": float(np.mean(np.abs(exact["a_full"][train] - pairs.d[train]))),
                    "A_lomo_train_mae": float(np.mean(np.abs(exact["a_lomo"][train] - pairs.d[train]))),
                    "B_train_mae": float(exact["fit"]["train_mae"]),
                    "B_train_rmse": float(exact["fit"]["train_rmse"]),
                    "B_residual_weighted_r2": float(exact["fit"]["train_residual_weighted_r2"]),
                    "beta_norm": float(exact["fit"]["beta_norm"]),
                },
            },
        },
        "scopes": scopes,
        "subsets": subsets,
        "support_bins": support_bins,
        "paired_B_minus_A": paired,
        "paired_A_minus_global": paired_a_global,
        "calibration": calibration,
        "coarse_type_sensitivity": coarse_block,
        "b_compact_ablation": compact,
        "posthoc_lambda_100": lam100,
        "posthoc_B_self_included_offset": self_included,
        "b_compact_in_sample": {
            "n_used_features": int(exact_compact["fit"]["n_used_features"]),
            "train_mae": float(exact_compact["fit"]["train_mae"]),
            "residual_weighted_r2": float(exact_compact["fit"]["train_residual_weighted_r2"]),
            "beta_norm": float(exact_compact["fit"]["beta_norm"]),
        },
        "failure_cases": {
            "cross_worst_A_error_all": cross_fail_a,
            "cross_worst_A_error_seen_type": seen_fail_a,
            "cross_worst_B_minus_A_degradation_all": cross_degrade,
            "cross_worst_B_minus_A_degradation_seen_type": seen_degrade,
            "definitions": {
                "err_a": "|A_lomo - delta_true|",
                "delta_err_b_minus_a": "|B - delta_true| - |A_lomo - delta_true|",
                "cross_scope": "pairs with one Train-sub endpoint and one Dev endpoint",
            },
        },
        "evaluation_coverage": {
            "n_dev_molecules": int(inp.dev.sum()),
            "n_dev_molecules_in_cross_pairs": int(len({int(m) for j in cross.tolist()
                                                       for m in (pairs.a[j], pairs.b[j])
                                                       if inp.dev[int(m)]})),
            "n_dev_molecules_in_dev_dev_pairs": int(len({int(m) for j in dd.tolist()
                                                         for m in (pairs.a[j], pairs.b[j])
                                                         if inp.dev[int(m)]})),
            "n_dev_molecules_in_any_eval_pair": int(len({int(m) for j in combined.tolist()
                                                         for m in (pairs.a[j], pairs.b[j])
                                                         if inp.dev[int(m)]})),
            "note": "transfer conclusions apply to the matched-pair evaluation subset, not all 504 Dev molecules",
        },
        "bootstrap": {
            "reps": A.BOOTSTRAP_REPS,
            "seed": A.BOOTSTRAP_SEED,
            "ci": "percentile 95%",
            "cross_cluster": "Dev molecule (129 clusters, 474 pairs)",
            "dev_dev_cluster": "full Train pair component (72 clusters, 611 pairs)",
            "combined_cluster": "full Train pair component",
        },
        "notes": [
            "No pair containing a Dev molecule is used to fit A or B.",
            "The LOMO support policy excludes any Train-sub pair sharing either molecule with the evaluation pair; for cross pairs this prevents the train endpoint's label from entering its own type mean.",
            "B is fitted on Train-sub residuals only; feature vocabulary, standardisation and ridge coefficients use no Dev label.",
            "The post-hoc lambda=100 block is a robustness check, not a model-selection step.",
            "Metrics are delta-level; they are not whole-molecule RMSE and are not comparable to the parent rounds' whole-molecule tables.",
        ],
    }
    return payload