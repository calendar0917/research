"""Task 2: real replacement-response variation and stability."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

import audit_common as A


def _eta2(values: np.ndarray, groups: np.ndarray) -> float:
    values = np.asarray(values, np.float64)
    groups = np.asarray(groups, np.int64)
    if values.size == 0:
        return float("nan")
    sst = float(np.sum((values - values.mean()) ** 2))
    if sst <= 1e-15:
        return 1.0
    sse = 0.0
    for g in np.unique(groups):
        v = values[groups == g]
        sse += float(np.sum((v - v.mean()) ** 2))
    return float(1.0 - sse / sst)


def _describe(values: np.ndarray) -> dict[str, Any]:
    v = np.asarray(values, np.float64)
    return {
        "n": int(v.size),
        "mean": float(v.mean()) if v.size else float("nan"),
        "sd": float(v.std()) if v.size else float("nan"),
        "min": float(v.min()) if v.size else float("nan"),
        "max": float(v.max()) if v.size else float("nan"),
        "q05": float(np.percentile(v, 5)) if v.size else float("nan"),
        "q25": float(np.percentile(v, 25)) if v.size else float("nan"),
        "q50": float(np.percentile(v, 50)) if v.size else float("nan"),
        "q75": float(np.percentile(v, 75)) if v.size else float("nan"),
        "q95": float(np.percentile(v, 95)) if v.size else float("nan"),
        "mean_abs": float(np.mean(np.abs(v))) if v.size else float("nan"),
    }


def _word_strings(inp: A.Inputs) -> np.ndarray:
    return np.asarray([str(x) for x in inp.trc["fragment_keys"]])


def _type_key_payload(key: Any) -> dict[str, Any]:
    if isinstance(key, tuple):
        return {"kind": "coarse", "r1_a": int(key[0]), "r1_b": int(key[1]), "bond_row": int(key[2])}
    return {"kind": "exact", "transform": int(key)}


def type_response_records(pairs: A.PairTable, inp: A.Inputs, keys: np.ndarray,
                          detail_contexts: bool) -> list[dict[str, Any]]:
    groups = A.group_positions(keys, np.nonzero(pairs.scope == 0)[0])
    words = _word_strings(inp)
    rows: list[dict[str, Any]] = []
    for key in groups:
        pos = groups[key]
        if len(set(int(x) for x in pairs.C[pos].tolist())) < 2:
            continue
        y = pairs.d[pos]
        sign = np.sign(y)
        pos_sign = int(np.sum(sign > 0))
        neg_sign = int(np.sum(sign < 0))
        zero_sign = int(np.sum(sign == 0))
        majority = max(pos_sign, neg_sign)
        rec: dict[str, Any] = {
            "type": _type_key_payload(key),
            "n_pairs": int(pos.size),
            "n_ctx_full": int(len(set(int(x) for x in pairs.C[pos].tolist()))),
            "n_ctx_r1": int(len(set(int(x) for x in pairs.r1e[pos].tolist()))),
            "n_ctx_r2": int(len(set(int(x) for x in pairs.r2e[pos].tolist()))),
            "n_scaffolds": int(len({inp.scaffold_of_mol[int(m)] for m in A.union_molecules(pairs.a, pairs.b, pos)})),
            "n_components": int(len(A.union_components(pairs.comp_train_nodes, pairs.a, pairs.b, pos))),
            "mean_delta": float(y.mean()),
            "sd_delta": float(y.std()),
            "min_delta": float(y.min()),
            "max_delta": float(y.max()),
            "range_delta": float(y.max() - y.min()),
            "mean_abs_deviation": float(np.mean(np.abs(y - y.mean()))),
            "sign_consistency": float(majority / max(1, pos.size)),
            "sign_counts": {"positive": pos_sign, "negative": neg_sign, "zero": zero_sign},
            "all_same_sign": bool(majority == pos.size and zero_sign == 0),
            "eta2_r1": _eta2(y, pairs.r1e[pos]),
            "eta2_r2": _eta2(y, pairs.r2e[pos]),
            "eta2_full": 1.0,
            "deltas": [float(x) for x in y.tolist()],
        }
        if isinstance(key, int):
            rec["type"]["sa_word"] = int(pairs.Sa[pos[0]])
            rec["type"]["sb_word"] = int(pairs.Sb[pos[0]])
            rec["type"]["bond_row"] = int(pairs.bond_row[pos[0]])
            rec["type"]["sa_text"] = words[int(pairs.Sa[pos[0]])][:160]
            rec["type"]["sb_text"] = words[int(pairs.Sb[pos[0]])][:160]
        if detail_contexts:
            ctx_rows = []
            for j in pos.tolist():
                ctx_rows.append({
                    "c": int(pairs.C[j]),
                    "r1": int(pairs.r1e[j]),
                    "r2": int(pairs.r2e[j]),
                    "delta": float(pairs.d[j]),
                    "a": int(pairs.a[j]),
                    "b": int(pairs.b[j]),
                    "scaffold_a": str(inp.scaffold_of_mol[int(pairs.a[j])])[:120],
                    "scaffold_b": str(inp.scaffold_of_mol[int(pairs.b[j])])[:120],
                    "component": int(pairs.comp_train[j]),
                })
            rec["contexts"] = ctx_rows
        rows.append(rec)
    rows.sort(key=lambda r: (-r["n_pairs"], r["type"].get("transform", -1)))
    return rows


def _support_class(n_ctx: int) -> str:
    if n_ctx <= 2:
        return "2"
    if n_ctx <= 4:
        return "3-4"
    if n_ctx <= 9:
        return "5-9"
    return ">=10"


def support_class_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for cls in ("2", "3-4", "5-9", ">=10"):
        selected = [r for r in rows if _support_class(int(r["n_ctx_full"])) == cls]
        if not selected:
            out[cls] = {"n_types": 0, "n_pairs": 0}
            continue
        sds = np.asarray([r["sd_delta"] for r in selected], np.float64)
        signs = np.asarray([r["sign_consistency"] for r in selected], np.float64)
        means = np.asarray([r["mean_delta"] for r in selected], np.float64)
        pairs = np.asarray([r["n_pairs"] for r in selected], np.float64)
        eta1 = np.asarray([r["eta2_r1"] for r in selected], np.float64)
        eta2 = np.asarray([r["eta2_r2"] for r in selected], np.float64)
        out[cls] = {
            "n_types": len(selected),
            "n_pairs": int(pairs.sum()),
            "mean_type_mean": float(means.mean()),
            "mean_abs_type_mean": float(np.abs(means).mean()),
            "median_within_type_sd": float(np.median(sds)),
            "mean_within_type_sd": float(sds.mean()),
            "median_eta2_r1": float(np.median(eta1)),
            "mean_eta2_r1": float(eta1.mean()),
            "median_eta2_r2": float(np.median(eta2)),
            "mean_eta2_r2": float(eta2.mean()),
            "frac_types_eta2_r1_gt_0.5": float(np.mean(eta1 > 0.5)),
            "frac_types_eta2_r2_gt_0.5": float(np.mean(eta2 > 0.5)),
            "pair_weighted_sign_consistency": float(np.sum(pairs * signs) / pairs.sum()),
            "types_with_any_reversal": int(sum(1 for r in selected if not r["all_same_sign"])),
            "frac_types_with_any_reversal": float(sum(1 for r in selected if not r["all_same_sign"]) / len(selected)),
            "types_all_same_sign": int(sum(1 for r in selected if r["all_same_sign"])),
            "frac_types_all_same_sign": float(sum(1 for r in selected if r["all_same_sign"]) / len(selected)),
            "max_n_pairs": int(max(r["n_pairs"] for r in selected)),
        }
    return out


def _design(columns: Sequence[np.ndarray], n: int) -> np.ndarray:
    mats = []
    for keys in columns:
        uniq, inv = np.unique(keys, axis=0, return_inverse=True)
        inv = np.asarray(inv).reshape(-1)
        mat = np.zeros((n, max(0, uniq.shape[0] - 1)), np.float64)
        for g in range(1, uniq.shape[0]):
            mat[:, g - 1] = (inv == g).astype(np.float64)
        mats.append(mat)
    return np.column_stack(mats) if mats else np.zeros((n, 0), np.float64)


def _ols_r2(X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    X1 = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    pred = X1 @ beta
    sse = float(np.sum((y - pred) ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else float("nan")
    rank = int(np.linalg.matrix_rank(X1))
    p = max(0, rank - 1)
    n = len(y)
    adj = 1.0 - (1.0 - r2) * (n - 1) / (n - p - 1) if n - p - 1 > 0 else float("nan")
    return {"r2": float(r2), "adj_r2": float(adj), "df": int(p), "n": int(n)}


def cross_context_differences(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for cls in ("2", "3-4", "5-9", ">=10"):
        selected = [r for r in rows if _support_class(int(r["n_ctx_full"])) == cls]
        if not selected:
            out[cls] = {"n_types": 0}
            continue
        diffs: list[float] = []
        for r in selected:
            v = np.asarray(r["deltas"], np.float64)
            if v.size < 2:
                continue
            iu = np.triu_indices(v.size, 1)
            diffs.extend(np.abs(v[iu[0]] - v[iu[1]]).tolist())
        diffs = np.asarray(diffs, np.float64)
        out[cls] = {
            "n_types": len(selected),
            "n_context_pairs": int(diffs.size),
            "mean_abs_diff": float(diffs.mean()) if diffs.size else None,
            "median_abs_diff": float(np.median(diffs)) if diffs.size else None,
            "q75_abs_diff": float(np.percentile(diffs, 75)) if diffs.size else None,
            "q90_abs_diff": float(np.percentile(diffs, 90)) if diffs.size else None,
            "max_abs_diff": float(diffs.max()) if diffs.size else None,
            "frac_abs_diff_gt_0.5": float(np.mean(diffs > 0.5)) if diffs.size else None,
            "frac_abs_diff_gt_1.0": float(np.mean(diffs > 1.0)) if diffs.size else None,
        }
    return out


def within_type_context_explanation(pairs: A.PairTable, keys: np.ndarray,
                                     rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Pooled within-type ANOVA: how much within-type variance do r1/r2 explain?

    This avoids the small-sample per-type eta2 values (which are binary at two
    contexts) by pooling the within-type sums of squares across types.
    """
    types = set()
    for r in rows:
        t = r["type"]
        types.add(int(t["transform"]) if t["kind"] == "exact" else
                  (int(t["r1_a"]), int(t["r1_b"]), int(t["bond_row"])))
    train = np.nonzero(pairs.scope == 0)[0]
    groups = A.group_positions(keys, train)
    pos = np.asarray([j for j in train.tolist() if A.hashable_key(keys[j]) in types], np.int64)
    y = pairs.d[pos]
    sst = 0.0
    for key, tpos in groups.items():
        if key not in types:
            continue
        v = pairs.d[tpos]
        sst += float(np.sum((v - v.mean()) ** 2))
    out: dict[str, Any] = {
        "n_pairs": int(pos.size),
        "n_types": int(len(types)),
        "within_type_ss": float(sst),
        "within_type_sd": float(np.sqrt(sst / max(1, pos.size))),
    }
    for name, ctx in (("r1", pairs.r1e), ("r2", pairs.r2e)):
        sse = 0.0
        n_groups = 0
        for key, tpos in groups.items():
            if key not in types:
                continue
            v = pairs.d[tpos]
            c = ctx[tpos]
            for g in np.unique(c):
                vv = v[c == g]
                sse += float(np.sum((vv - vv.mean()) ** 2))
                n_groups += 1
        frac = 1.0 - sse / sst if sst > 0 else float("nan")
        out[f"{name}_within_type_explained_fraction"] = float(frac)
        out[f"{name}_within_type_residual_fraction"] = float(1.0 - frac) if sst > 0 else float("nan")
        out[f"{name}_n_groups"] = int(n_groups)
    out["note"] = ("Pooled within-type ANOVA. The residual after r2 is not identified as "
                   "measurement noise; it can contain higher-order context or type-by-context interaction.")
    return out


def variance_decomposition(pairs: A.PairTable, rows_exact: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    multi_types = {int(r["type"]["transform"]) for r in rows_exact}
    pos = np.asarray([j for j in np.nonzero(pairs.scope == 0)[0].tolist()
                      if int(pairs.T[j]) in multi_types and int(pairs.C[j]) >= 0], np.int64)
    # The filter above uses T membership; keep pairs whose exact type has >=2 contexts.
    # rows_exact already contains exactly those types.
    pos = pos[np.asarray([int(pairs.T[j]) in multi_types for j in pos.tolist()], bool)]
    y = pairs.d[pos]
    n = len(y)
    X_type = _design([pairs.T[pos][:, None]], n)
    X_coarse = _design([pairs.coarse[pos]], n)
    X_r1 = _design([pairs.r1e[pos][:, None]], n)
    X_r2 = _design([pairs.r2e[pos][:, None]], n)
    X_full = _design([pairs.C[pos][:, None]], n)
    models = {
        "coarse_type": X_coarse,
        "exact_type": X_type,
        "r1_context": X_r1,
        "r2_context": X_r2,
        "exact_type+r1": np.column_stack([X_type, X_r1]),
        "exact_type+r2": np.column_stack([X_type, X_r2]),
        "exact_type+r1+r2": np.column_stack([X_type, X_r1, X_r2]),
        "exact_type+full_context": np.column_stack([X_type, X_full]),
    }
    out = {name: _ols_r2(X, y) for name, X in models.items()}
    base = out["exact_type"]["r2"]
    base_adj = out["exact_type"]["adj_r2"]
    for name in out:
        out[name]["delta_r2_over_exact_type"] = float(out[name]["r2"] - base)
        out[name]["delta_adj_r2_over_exact_type"] = float(out[name]["adj_r2"] - base_adj)
    # permutation null for the exact-type effect
    rng = np.random.default_rng(0)
    obs = out["exact_type"]["r2"]
    null = []
    for _ in range(1000):
        perm = rng.permutation(n)
        X_perm = _design([pairs.T[pos][perm][:, None]], n)
        null.append(_ols_r2(X_perm, y)["r2"])
    null = np.asarray(null, np.float64)
    p = float((np.sum(null >= obs) + 1) / (len(null) + 1))
    return {
        "subset": "Train-sub pairs whose exact replacement type has >= 2 full contexts",
        "n_pairs": int(n),
        "n_types": int(len(multi_types)),
        "models": out,
        "permutation_test_exact_type": {
            "reps": 1000,
            "observed_r2": float(obs),
            "null_mean": float(null.mean()),
            "null_sd": float(null.std()),
            "p_value": p,
            "note": "type labels permuted across pairs; this tests type-level variance, not within-type context stability",
        },
        "note": ("Unweighted in-sample OLS with an intercept. Incremental R2 is descriptive; "
                 "the transfer check is the out-of-split test."),
    }


def _loco_predictions(pairs: A.PairTable, keys: np.ndarray,
                      min_ctx: int, exclude_component: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    groups = A.group_positions(keys, np.nonzero(pairs.scope == 0)[0])
    g = float(pairs.d[pairs.scope == 0].mean())
    idx: list[int] = []
    pred: list[float] = []
    nsup: list[int] = []
    for key, pos in groups.items():
        if len(set(int(x) for x in pairs.C[pos].tolist())) < int(min_ctx):
            continue
        for j in pos.tolist():
            if exclude_component:
                c = int(pairs.comp_train[j])
                others = [int(x) for x in pos.tolist() if int(pairs.comp_train[x]) != c]
            else:
                others = [int(x) for x in pos.tolist() if int(x) != int(j)]
            n = len(others)
            if n == 0:
                p = g
            else:
                p = (float(np.sum(pairs.d[others])) + A.SHRINK_K * g) / (n + A.SHRINK_K)
            idx.append(int(j))
            pred.append(float(p))
            nsup.append(n)
    return np.asarray(idx, np.int64), np.asarray(pred, np.float64), np.asarray(nsup, np.int64)


def loco_summary(pairs: A.PairTable, keys: np.ndarray, min_ctx: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    g = float(pairs.d[pairs.scope == 0].mean())
    for name, exclude_comp in (("loco", False), ("loco_component", True)):
        idx, pred, nsup = _loco_predictions(pairs, keys, min_ctx, exclude_comp)
        y = pairs.d[idx]
        clusters = pairs.comp_train[idx]
        boot = A.cluster_bootstrap(y, {"pred": pred}, clusters, reps=A.BOOTSTRAP_REPS,
                                   seed=A.BOOTSTRAP_SEED)
        global_pred = np.full_like(y, g)
        gm = A.delta_metrics(y, global_pred)
        out[name] = {
            "n_pairs": int(len(y)),
            "n_types": int(len(set(A.hashable_key(keys[x]) for x in idx.tolist()))),
            "n_zero_support_predictions": int(np.sum(nsup == 0)),
            "mae": float(boot["point"]["pred"]["mae"]),
            "mae_ci": boot["bootstrap"]["pred"]["mae_ci"],
            "rmse": float(boot["point"]["pred"]["rmse"]),
            "rmse_ci": boot["bootstrap"]["pred"]["rmse_ci"],
            "corr": float(boot["point"]["pred"]["corr"]),
            "sign": float(boot["point"]["pred"]["sign"]),
            "bias": float(boot["point"]["pred"]["bias"]),
            "global_mae": float(gm["mae"]),
            "global_rmse": float(gm["rmse"]),
            "mae_gain_over_global": float(gm["mae"] - boot["point"]["pred"]["mae"]),
            "mean_true": float(y.mean()),
            "mean_pred": float(pred.mean()),
        }
    return out


def sign_stability(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for cls in ("2", "3-4", "5-9", ">=10"):
        selected = [r for r in rows if _support_class(int(r["n_ctx_full"])) == cls]
        if not selected:
            out[cls] = {"n_types": 0}
            continue
        signs = np.asarray([r["sign_consistency"] for r in selected], np.float64)
        pairs = np.asarray([r["n_pairs"] for r in selected], np.float64)
        out[cls] = {
            "n_types": len(selected),
            "n_pairs": int(pairs.sum()),
            "pair_weighted_sign_consistency": float(np.sum(pairs * signs) / pairs.sum()),
            "median_sign_consistency": float(np.median(signs)),
            "n_types_all_same_sign": int(sum(1 for r in selected if r["all_same_sign"])),
            "n_types_with_reversal": int(sum(1 for r in selected if not r["all_same_sign"])),
            "frac_types_with_reversal": float(sum(1 for r in selected if not r["all_same_sign"]) / len(selected)),
            "sign_consistency_histogram": dict(sorted(Counter(np.round(signs, 2).tolist()).items())),
        }
    return out


def naive_split_half_diagnostic(pairs: A.PairTable, keys: np.ndarray,
                                min_ctx: int = 4, reps: int = 500) -> dict[str, Any]:
    """A deliberately naive check, reported only to show that it is not diagnostic.

    Splitting contexts within a type and correlating the two half-means across
    types is dominated by between-type differences.  The permutation null keeps
    the type means and destroys within-type context structure; the observed
    correlation is not larger than this null.
    """
    import hashlib

    groups = A.group_positions(keys, np.nonzero(pairs.scope == 0)[0])
    xs, ys, ws, specs = [], [], [], []
    for key, pos in groups.items():
        ctxs = sorted(set(int(x) for x in pairs.C[pos].tolist()))
        if len(ctxs) < int(min_ctx):
            continue
        order = sorted(ctxs, key=lambda c: hashlib.sha256(f"audit-sh:{c}".encode()).hexdigest())
        half = len(order) // 2
        A_set = set(order[:half])
        B_set = set(order[half:])
        ja = [int(j) for j in pos.tolist() if int(pairs.C[j]) in A_set]
        jb = [int(j) for j in pos.tolist() if int(pairs.C[j]) in B_set]
        if ja and jb:
            xs.append(float(pairs.d[ja].mean()))
            ys.append(float(pairs.d[jb].mean()))
            ws.append(min(len(ja), len(jb)))
            specs.append((pos, ja, jb))
    if len(xs) < 3:
        return {"n_types": len(xs), "note": "too few types"}
    xs, ys, ws = np.asarray(xs), np.asarray(ys), np.asarray(ws)

    def corr(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
        ma, mb = np.average(a, weights=w), np.average(b, weights=w)
        denom = np.sqrt(np.sum(w * (a - ma) ** 2) * np.sum(w * (b - mb) ** 2))
        return float(np.sum(w * (a - ma) * (b - mb)) / denom) if denom > 0 else float("nan")

    obs = corr(xs, ys, ws)
    rng = np.random.default_rng(0)
    null = []
    for _ in range(int(reps)):
        xs_n, ys_n, ws_n = [], [], []
        for pos, ja, jb in specs:
            yy = pairs.d[pos]
            perm = rng.permutation(len(pos))
            na = len(ja)
            xs_n.append(float(yy[perm[:na]].mean()))
            ys_n.append(float(yy[perm[na:]].mean()))
            ws_n.append(min(na, len(pos) - na))
        null.append(corr(np.asarray(xs_n), np.asarray(ys_n), np.asarray(ws_n)))
    null = np.asarray(null, np.float64)
    return {
        "n_types": len(xs),
        "observed_weighted_r": obs,
        "null_mean": float(null.mean()),
        "null_sd": float(null.std()),
        "p_value_observed_ge_null": float((np.sum(null >= obs) + 1) / (len(null) + 1)),
        "note": ("This statistic is dominated by between-type differences; the permutation null "
                 "keeps type means and destroys within-type context structure. It is reported "
                 "only as a negative methodological check, not as evidence of stability."),
    }


def noise_floor(inp: A.Inputs, pairs: A.PairTable) -> dict[str, Any]:
    from rdkit import Chem

    smiles = np.asarray([str(x) for x in inp.acra["smiles"]])
    canonical = {}
    for row in np.nonzero(inp.official_train)[0].tolist():
        mol = Chem.MolFromSmiles(smiles[row])
        canonical[row] = Chem.MolToSmiles(mol) if mol is not None else smiles[row]
    by: dict[str, list[int]] = defaultdict(list)
    for row, smi in canonical.items():
        by[smi].append(int(row))
    duplicates = {k: v for k, v in by.items() if len(v) > 1}
    distinct_labels = {k: v for k, v in duplicates.items() if len(set(float(inp.labels[r]) for r in v)) > 1}
    sds = []
    for v in distinct_labels.values():
        vals = np.asarray([float(inp.labels[r]) for r in v], np.float64)
        sds.append(float(vals.std(ddof=1)))
    cell = Counter(zip(pairs.T.tolist(), pairs.C.tolist()))
    return {
        "estimated": False,
        "reason": ("No replicate structures and no repeated (exact type, full context) cell; "
                   "within-type across-context variation mixes effect modification and measurement noise."),
        "n_duplicate_canonical_structures_train": int(len(duplicates)),
        "n_rows_in_duplicate_structures": int(sum(len(v) for v in duplicates.values())),
        "n_duplicate_groups_with_distinct_labels": int(len(distinct_labels)),
        "within_structure_sd": {
            "n": len(sds),
            "mean": float(np.mean(sds)) if sds else None,
            "median": float(np.median(sds)) if sds else None,
            "max": float(max(sds)) if sds else None,
        },
        "max_pairs_per_exact_type_full_context": int(max(cell.values())) if cell else 0,
        "n_exact_type_full_context_cells": int(len(cell)),
    }


def failure_cases(rows_exact: Sequence[Mapping[str, Any]],
                  rows_coarse: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    exact_multi = [r for r in rows_exact if r["n_ctx_full"] >= 3]
    coarse_multi = [r for r in rows_coarse if r["n_ctx_full"] >= 5]

    def compact(r: Mapping[str, Any], with_contexts: bool = False) -> dict[str, Any]:
        out = {
            "type": r["type"],
            "n_pairs": r["n_pairs"],
            "n_ctx_full": r["n_ctx_full"],
            "n_ctx_r1": r["n_ctx_r1"],
            "n_ctx_r2": r["n_ctx_r2"],
            "n_scaffolds": r["n_scaffolds"],
            "n_components": r["n_components"],
            "mean_delta": r["mean_delta"],
            "sd_delta": r["sd_delta"],
            "sign_consistency": r["sign_consistency"],
            "deltas": r["deltas"],
        }
        if with_contexts and "contexts" in r:
            out["contexts"] = r["contexts"]
        return out

    worst_sign = sorted(
        exact_multi,
        key=lambda r: (-(1.0 - r["sign_consistency"]) * r["mean_abs_deviation"],
                       -r["n_pairs"]))[:10]
    worst_spread = sorted(exact_multi, key=lambda r: (-r["sd_delta"], -r["n_pairs"]))[:10]
    opposite_pairs = [r for r in exact_multi
                      if r["sign_counts"]["positive"] > 0 and r["sign_counts"]["negative"] > 0]
    opposite_pairs = sorted(opposite_pairs, key=lambda r: (-r["n_ctx_full"], r["sign_consistency"]))[:10]
    coarse_worst = sorted(
        coarse_multi,
        key=lambda r: (-(1.0 - r["sign_consistency"]) * r["mean_abs_deviation"],
                       -r["n_pairs"]))[:5]
    return {
        "exact_high_support_low_sign_consistency": [compact(r) for r in worst_sign],
        "exact_high_support_largest_within_type_sd": [compact(r) for r in worst_spread],
        "exact_high_support_with_opposite_sign_contexts": [compact(r, with_contexts=True) for r in opposite_pairs],
        "coarse_high_support_low_sign_consistency": [compact(r) for r in coarse_worst],
        "definitions": {
            "high_support_exact": "exact type with >= 3 full contexts in Train-sub",
            "high_support_coarse": "coarse type with >= 5 full contexts in Train-sub",
            "opposite_sign_contexts": "at least one positive and one negative delta in the type's full contexts",
        },
    }


def run(inp: A.Inputs, pairs: A.PairTable) -> dict[str, Any]:
    exact_rows = type_response_records(pairs, inp, pairs.T, detail_contexts=True)
    coarse_rows = type_response_records(pairs, inp, pairs.coarse, detail_contexts=False)

    overall = {}
    for scope, name in enumerate(A.SCOPE_NAMES):
        y = pairs.d[pairs.scope == scope]
        overall[name] = _describe(y)

    payload = {
        "meta": A.meta_block(inp, "structural-response-learnability-audit-v1/response", {
            "source_pairs": A.relative(A.TRC_CACHE),
            "delta_definition": "delta_y = y(G_b) - y(G_a), original lipophilicity units, pair direction fixed by the directory (word_a < word_b)",
        }),
        "definitions": {
            "type": "exact replacement type = (word_a, word_b, cut bond raw attributes)",
            "coarse_type": "(radius-1 identity of word_a, radius-1 identity of word_b, cut bond raw attributes)",
            "support_requirement": "exact type must have at least 2 distinct full contexts in Train-sub for a response record",
            "sign_consistency": "max(#positive, #negative) / n_pairs; direction is the fixed canonical word order, not a chemical direction",
            "eta2_r1": "1 - within-radius-1-group SS / total SS within the type",
            "eta2_r2": "1 - within-radius-2-group SS / total SS within the type",
            "eta2_full": "1 by construction: one real pair per (exact type, full context) cell",
        },
        "overall_delta": overall,
        "support_class_summary": {
            "exact": support_class_summary(exact_rows),
            "coarse": support_class_summary(coarse_rows),
        },
        "sign_stability": {
            "exact": sign_stability(exact_rows),
            "coarse": sign_stability(coarse_rows),
        },
        "cross_context_differences": {
            "exact": cross_context_differences(exact_rows),
            "coarse": cross_context_differences(coarse_rows),
        },
        "within_type_context_explanation": {
            "exact": within_type_context_explanation(pairs, pairs.T, exact_rows),
            "coarse": within_type_context_explanation(pairs, pairs.coarse, coarse_rows),
        },
        "within_train_transfer": {
            "exact": {
                "ctx2": loco_summary(pairs, pairs.T, 2),
                "ctx3": loco_summary(pairs, pairs.T, 3),
                "ctx5": loco_summary(pairs, pairs.T, 5),
            },
            "coarse": {
                "ctx2": loco_summary(pairs, pairs.coarse, 2),
                "ctx3": loco_summary(pairs, pairs.coarse, 3),
                "ctx5": loco_summary(pairs, pairs.coarse, 5),
            },
        },
        "variance_decomposition": variance_decomposition(pairs, exact_rows),
        "diagnostics": {
            "naive_split_half_exact": naive_split_half_diagnostic(pairs, pairs.T, min_ctx=4),
            "naive_split_half_coarse": naive_split_half_diagnostic(pairs, pairs.coarse, min_ctx=4),
        },
        "noise_floor": noise_floor(inp, pairs),
        "failure_cases": failure_cases(exact_rows, coarse_rows),
        "types": {
            "exact": exact_rows,
            "coarse": coarse_rows,
        },
        "notes": [
            "All deltas are label differences on real matched pairs; they are associational contrasts, not chemical causal effects.",
            "The full context is the whole environment of the cut, not a local approximation; within a type, each full context contributes exactly one pair.",
            "Many exact types have exactly one context; their mean delta is not a repeatable estimate and is kept only in the support table.",
            "The coarsening to radius-1 attachment identity increases support but mixes distinct exact fragment replacements; it is a sensitivity level, not the primary type.",
        ],
    }
    return payload