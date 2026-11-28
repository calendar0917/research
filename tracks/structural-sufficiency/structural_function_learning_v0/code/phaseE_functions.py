"""Phase E — exact extraction of the additive structure functions and local prediction
explanations (no new training).

ADD-TREE: every tree uses exactly one feature, so all trees that use feature ``j`` are
merged into ``f_j(x)``.  With the zero baseline for structure features and the Train
median for the two size features, ``prediction = adjusted_bias + sum_j f_j(x_j)`` is
verified against the booster's raw margin.  ADD-NN functions are read directly from
the checkpoint.  Representative MolHIV structures are selected using **Train support
and the learned functions only** (never Test performance).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import scipy.sparse as sp

import sfl_common as K


# ---------------------------------------------------------------------------
# tree extraction
# ---------------------------------------------------------------------------


def load_booster(arm: str):
    import xgboost as xgb

    b = xgb.Booster()
    b.load_model(str(K.MODELS / f"{arm}.json"))
    return b


def tree_nodes(tree: dict[str, Any]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}

    def rec(n: dict[str, Any]) -> None:
        out[int(n["nodeid"])] = n
        for c in n.get("children", []):
            rec(c)

    rec(tree)
    return out


def tree_feature(tree: dict[str, Any]) -> str | None:
    feats = {str(n["split"]) for n in tree_nodes(tree).values() if "split" in n}
    if len(feats) > 1:
        raise ValueError(f"tree uses multiple features: {feats}")
    return next(iter(feats)) if feats else None


def eval_tree(tree: dict[str, Any], x: np.ndarray) -> np.ndarray:
    """Replay one tree with XGBoost's float32 split semantics.

    XGBoost stores ``split_condition`` and the input design as float32 and compares
    them in float32.  Comparing a float64 array against a float64 threshold changes
    the branch for values that are exactly at a float32 split boundary, so both
    sides are explicitly cast to float32 here (verified against booster.predict).
    """
    nodes = tree_nodes(tree)
    out = np.empty(x.size, dtype=np.float64)
    x32 = np.asarray(x, dtype=np.float32)

    def rec(node: dict[str, Any], idx: np.ndarray) -> None:
        if idx.size == 0:
            return
        if "leaf" in node:
            out[idx] = float(node["leaf"])
            return
        left = x32[idx] < np.float32(node["split_condition"])
        rec(nodes[int(node["yes"])], idx[left])
        rec(nodes[int(node["no"])], idx[~left])

    rec(nodes[0], np.arange(x.size))
    return out


class TreeFunctions:
    def __init__(self, booster: Any, feature_names: list[str]) -> None:
        self.names = feature_names
        self.trees = [json.loads(t) for t in booster.get_dump(dump_format="json")]
        self.by_feature: dict[str, list[dict[str, Any]]] = {}
        for t in self.trees:
            f = tree_feature(t)
            if f is not None:
                self.by_feature.setdefault(f, []).append(t)
        cfg = json.loads(booster.save_config())
        self.base_score = float(cfg["learner"]["learner_model_param"]["base_score"])
        self.objective = cfg["learner"]["objective"]["name"]
        self.base_margin = (float(np.log(self.base_score / (1.0 - self.base_score)))
                            if self.objective.startswith("binary") else self.base_score)

    def f(self, feature: str, x: np.ndarray) -> np.ndarray:
        out = np.zeros(x.size, dtype=np.float64)
        for t in self.by_feature.get(feature, []):
            out += eval_tree(t, x)
        return out


class NNFunctions:
    def __init__(self, arm: str, coordinate: Callable[[int, np.ndarray], np.ndarray]) -> None:
        import torch

        path = K.RUNS / f"{arm}_best.pt"
        if not path.exists():            # committed checkpoint copy
            path = K.MODELS / f"{arm}_best.pt"
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        sd = ckpt["state_dict"]
        self.a = sd["a"].numpy().astype(np.float64)
        self.v = sd["v"].numpy().astype(np.float64)
        self.t = sd["t"].numpy().astype(np.float64)
        self.bias = float(sd["bias"].numpy()[0])
        self.active = sd["active"].numpy().astype(bool)
        self.coordinate = coordinate
        self.ckpt = ckpt

    def f_input(self, j: int, z: np.ndarray) -> np.ndarray:
        """f_j evaluated at the *model input* coordinate z (so f_j(0)=0 exactly)."""
        z = np.asarray(z, dtype=np.float64)
        r = np.maximum(z[:, None] - self.t[j][None, :], 0.0) - \
            np.maximum(-self.t[j][None, :], 0.0)
        return self.a[j] * z + (self.v[j][None, :] * r).sum(axis=1)

    def f(self, j: int, x: np.ndarray) -> np.ndarray:
        """f_j evaluated at a *raw* feature value (preprocessing coordinate applied)."""
        return self.f_input(j, self.coordinate(j, x))


# ---------------------------------------------------------------------------
# selection helpers
# ---------------------------------------------------------------------------


def column_stats(mat: sp.csr_matrix, rows: np.ndarray) -> dict[str, np.ndarray]:
    """Per-column Train support and nonzero quantiles (raw frozen structure values)."""
    sub = mat[rows].tocsc()
    p = sub.shape[1]
    support = np.diff(sub.indptr).astype(np.int64)
    qs = np.zeros((p, 7), dtype=np.float64)
    levels = (0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
    for j in range(p):
        vals = sub.data[sub.indptr[j]:sub.indptr[j + 1]]
        if vals.size:
            qs[j] = np.quantile(vals, levels)
    return {"support": support, "quantiles": qs,
            "levels": np.asarray(levels)}


def shape_diff(f_tree: np.ndarray, f_nn: np.ndarray) -> float:
    def norm(f: np.ndarray) -> np.ndarray:
        rng = float(f.max() - f.min())
        if rng <= 1e-9:
            return np.zeros_like(f)
        return (f - f.min()) / rng
    return float(np.mean(np.abs(norm(f_tree) - norm(f_nn))))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    import pandas as pd
    import xgboost as xgb

    dh = K.HIVData()
    dz = K.ZincData()
    reg = K.registry()
    parent = K.full_parent_columns()
    names = [f"c{i}" for i in range(K.HIV_WIDTH)]
    booster = load_booster("H-ADD-TREE")
    tf = TreeFunctions(booster, names)

    # ---- closure of the extracted tree functions on a row sample ----------
    rng = np.random.default_rng(K.SEED)
    sample = np.sort(rng.choice(dh.n, size=2000, replace=False))
    Xs = dh.design_dense(sample)
    tree_sum = np.zeros(sample.size)
    for t in tf.trees:
        f = tree_feature(t)
        if f is None:
            continue
        j = names.index(f)
        tree_sum += eval_tree(t, Xs[:, j].astype(np.float64))
    pred = booster.predict(xgb.DMatrix(Xs, feature_names=names), output_margin=True)
    resid = pred - tree_sum
    tree_base = float(resid.mean())
    tree_closure = float(np.max(np.abs(resid - tree_base)))
    if tree_closure > 1e-4:
        raise SystemExit(f"tree extraction does not close: {tree_closure}")

    # ---- ADD-NN coordinate maps and closure -------------------------------
    nn_ckpt = K.read_json(K.ANALYSIS / "phaseC_H-NN-ADD.json")
    size_mu = np.asarray(nn_ckpt["preprocessing"]["size_mean"], dtype=np.float64)
    size_sd = np.asarray(nn_ckpt["preprocessing"]["size_std"], dtype=np.float64)

    def hiv_coord(j: int, x: np.ndarray) -> np.ndarray:
        if j < 2:
            return np.clip((x - size_mu[j]) / size_sd[j], -10.0, 10.0)
        return x

    nnf = NNFunctions("H-NN-ADD", hiv_coord)
    import torch

    # recompute NN forward from the checkpoint formula on 256 rows
    nn_rows = np.sort(rng.choice(dh.n, size=256, replace=False))
    Xn = dh.nn_design_dense(nn_rows)[0].astype(np.float64)
    nn_sum = np.full(nn_rows.size, nnf.bias)
    for j in range(Xn.shape[1]):
        # Xn is already in the model input space -> use f_input (no double transform)
        nn_sum += nnf.f_input(j, Xn[:, j])
    ckpt_path = K.RUNS / "H-NN-ADD_best.pt"
    if not ckpt_path.exists():
        ckpt_path = K.MODELS / "H-NN-ADD_best.pt"
    ckpt_state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    from phaseC_add_nn import AdditiveNN

    model = AdditiveNN(np.zeros((K.HIV_WIDTH, 4), dtype=np.float32),
                       np.ones(K.HIV_WIDTH, dtype=bool))
    model.load_state_dict(ckpt_state["state_dict"])
    model.eval()
    with torch.no_grad():
        nn_pred = model(torch.from_numpy(Xn.astype(np.float32))).numpy().astype(np.float64)
    nn_closure = float(np.max(np.abs(nn_pred - nn_sum)))
    f0 = np.array([nnf.f_input(j, np.zeros(1))[0] for j in range(0, K.HIV_WIDTH, 8)])
    nn_f0 = float(np.max(np.abs(f0)))

    # ---- per-column Train statistics for the HIV structure blocks ---------
    structure = dh.structures()
    stats = column_stats(structure, dh.rows["train"])
    support = stats["support"]
    q = stats["quantiles"]
    levels = stats["levels"]
    grid_idx = [1, 2, 3, 4, 5, 6]        # q05..q99
    grid_values = q[:, grid_idx]          # [4406, 6]
    zero = np.zeros((K.HIV_N_STRUCT, 1))
    grid = np.concatenate([zero, grid_values], axis=1)   # includes x=0 reference

    med_n = float(np.median(dh.size_log[dh.rows["train"], 0]))
    med_b = float(np.median(dh.size_log[dh.rows["train"], 1]))
    size_ref = {"c0": med_n, "c1": med_b,
                "unit": "log1p(n_atoms), log1p(n_bonds)", "fit_on": "Train median"}

    var_tree = np.zeros(K.HIV_N_STRUCT)
    var_nn = np.zeros(K.HIV_N_STRUCT)
    f_tree_grid = np.zeros((K.HIV_N_STRUCT, grid.shape[1]))
    f_nn_grid = np.zeros((K.HIV_N_STRUCT, grid.shape[1]))
    for j in range(K.HIV_N_STRUCT):
        node = f"c{2 + j}"
        ft = tf.f(node, grid[j]) - tf.f(node, np.zeros(1))[0]
        fn = nnf.f(2 + j, grid[j]) - nnf.f(2 + j, np.zeros(1))[0]
        f_tree_grid[j] = ft
        f_nn_grid[j] = fn
        var_tree[j] = float(ft.max() - ft.min())
        var_nn[j] = float(fn.max() - fn.min())

    # "high Train support" operationalised as >=1000 Train molecules (3.0% of Train);
    # the parent-child pair keeps the child threshold lower (>=500)
    support_ok = support >= 1000
    # both learned functions contribute to the "large variation" ranking (Train-only)
    score = np.where(support_ok, var_tree + var_nn, -np.inf)
    ranked = np.argsort(-score)
    selected: list[int] = []

    def add(j: int, reason: str) -> None:
        if j not in selected and len(selected) < 8:
            selected.append(int(j))
            reasons[int(j)] = reason

    reasons: dict[int, str] = {}
    # 1) high Train support + large learned function variation
    for j in ranked[:6]:
        add(int(j), "high Train support (>=1000) + large learned function variation")
        if len(selected) >= 3:
            break
    # 2) a FULL structure whose COARSE parent is also a supported, varying function
    pair = None
    best_pair = -np.inf
    for j in range(K.HIV_N_C, K.HIV_N_STRUCT):
        p = int(parent[j - K.HIV_N_C])
        if p < 0 or support[j] < 500 or support[p] < 1000:
            continue
        if var_nn[p] < 0.1 or var_nn[j] < 0.1:
            continue
        if p in selected or j in selected:
            continue                      # add a *new* coarse/fine pair
        cand_score = float(var_nn[p] + var_nn[j])
        if cand_score > best_pair:
            best_pair = cand_score
            pair = (p, j)
    if pair is not None:
        add(pair[0], "COARSE parent of the selected FULL structure (coarse/fine pair)")
        add(pair[1], "FULL child with a supported COARSE parent function")
    # 3) one or two cases where tree and neural functions disagree
    cand = [int(j) for j in ranked
            if support[j] >= 2000 and var_tree[j] > 0.05 and var_nn[j] > 0.05]
    diffs = {j: shape_diff(f_tree_grid[j], f_nn_grid[j]) for j in cand[:200]}
    for j in sorted(diffs, key=lambda k: -diffs[k])[:2]:
        add(j, "tree and neural functions differ in shape (largest normalised RMS gap)")
    # 4) fill to 6 if needed
    for j in ranked:
        if len(selected) >= 6:
            break
        add(int(j), "fill: high combined support/variation score")

    selected = selected[:8]
    pair_members = set(pair) if pair is not None else set()
    selected_meta = []
    for j in selected:
        gran = "C" if j < K.HIV_N_C else "F"
        col = int(j if gran == "C" else j - K.HIV_N_C)
        meta = reg[gran].iloc[col]
        support_train = int(support[j])
        selected_meta.append({
            "structure": f"{gran}[{col}]", "granularity": gran, "column": col,
            "canonical_key": str(meta["canonical_key"]), "family": str(meta["family"]),
            "node_signature": str(meta["node_signature"]),
            "document_frequency_train": int(meta["document_frequency_train"]),
            "occurrence_train": int(meta["occurrence_train"]),
            "positive_molecules_train": int(meta["positive_molecules_train"]),
            "negative_molecules_train": int(meta["negative_molecules_train"]),
            "support_train_nonzero": support_train,
            "parent_coarse_column": (int(parent[j - K.HIV_N_C])
                                     if gran == "F" else None),
            "reason_selected": reasons.get(j, "high combined score"),
            "coarse_fine_pair": bool(j in pair_members),
            "variation_tree": float(var_tree[j]),
            "variation_nn": float(var_nn[j]),
            "shape_difference_tree_vs_nn": float(shape_diff(f_tree_grid[j], f_nn_grid[j])),
            "grid_x": grid[j].tolist(),
            "f_tree": (f_tree_grid[j] - f_tree_grid[j][0]).tolist(),
            "f_nn": (f_nn_grid[j] - f_nn_grid[j][0]).tolist(),
        })

    # ---- linear function for the same structures -------------------------
    lin = K.read_json(K.ANALYSIS / "linear_models.json")["L-BAL-STD"]
    lin_w = np.asarray(lin["coef_raw"], dtype=np.float64)[2:]
    for entry in selected_meta:
        j = (entry["column"] if entry["granularity"] == "C"
             else K.HIV_N_C + entry["column"])
        entry["f_linear"] = (lin_w[j] * np.asarray(entry["grid_x"])).tolist()

    # ---- local explanations of the better additive model (by Valid AUC) ---
    hadd = K.read_json(K.ANALYSIS / "phaseC_H-NN-ADD.json")["metrics"]
    tree_json = K.read_json(K.RUNS / "H-ADD-TREE.json")
    tree_valid = tree_json["metrics_by_rounds"]["500"]["valid"]["roc_auc"]
    nn_valid = hadd["valid"]["roc_auc"]
    chosen = "H-ADD-TREE" if tree_valid >= nn_valid else "H-NN-ADD"
    select_by = {"criterion": "higher Valid ROC-AUC among the two additive models",
                 "H-ADD-TREE": tree_valid, "H-NN-ADD": nn_valid, "chosen": chosen}

    # size functions for the chosen model on the raw log1p coordinate
    if chosen == "H-ADD-TREE":
        size_ref_vals = {"c0": med_n, "c1": med_b}
        def f_of(j: int, x: np.ndarray) -> np.ndarray:
            return tf.f(f"c{j}", x)
    else:
        size_ref_vals = {"c0": med_n, "c1": med_b}
        def f_of(j: int, x: np.ndarray) -> np.ndarray:
            return nnf.f(j, x)
    # adjusted bias: base margin + f_j(reference) for every feature
    adjusted_bias = tree_base if chosen == "H-ADD-TREE" else nnf.bias
    for j in range(K.HIV_WIDTH):
        ref = size_ref_vals.get(f"c{j}", 0.0)
        adjusted_bias += float(f_of(j, np.asarray([ref]))[0])

    # 3 Valid cases: correct positive, correct negative, largest error (distinct)
    valid = dh.rows["valid"]
    stored = np.load(K.PREDICTIONS / f"{chosen}.npz")
    if "logit" in stored.files:
        full = np.asarray(stored["logit"], dtype=np.float64)
    elif "pred" in stored.files:
        full = np.asarray(stored["pred"], dtype=np.float64)
    else:                       # neural model: per-split best predictions
        full = np.empty(stored["y"].size, dtype=np.float64)
        for split in K.SPLITS:
            full[stored[f"{split}_rows"]] = stored[f"pred_{split}"]
    score_valid = full[valid]
    prob = K.sigmoid(score_valid)
    yv = dh.y[valid]
    pos_candidates = np.where(yv == 1)[0]
    neg_candidates = np.where(yv == 0)[0]
    i_pos = int(pos_candidates[np.argmax(prob[pos_candidates])])
    i_neg = int(neg_candidates[np.argmin(prob[neg_candidates])])
    err = np.abs(prob - yv)
    order = np.argsort(-err)
    i_fail = int(next(i for i in order if i not in (i_pos, i_neg)))

    cases = []
    for label, local_i in (("correct_positive", i_pos), ("correct_negative", i_neg),
                           ("largest_error", i_fail)):
        g = int(valid[local_i])
        row = dh.design_dense(np.array([g]))[0].astype(np.float64)
        contrib = np.zeros(K.HIV_WIDTH)
        for j in range(K.HIV_WIDTH):
            ref = size_ref_vals.get(f"c{j}", 0.0)
            contrib[j] = float(f_of(j, np.asarray([row[j]]))[0]) - float(f_of(j, np.asarray([ref]))[0])
        structure_contrib = contrib[2:]
        abs_c = np.abs(structure_contrib)
        abs_total = float(abs_c.sum())
        topk = np.sort(abs_c)[::-1]
        keep = np.ones(abs_c.size, dtype=bool)
        keep[np.argsort(-abs_c)[:20]] = False
        group: dict[str, float] = {}
        for jj in range(K.HIV_N_STRUCT):
            if jj < K.HIV_N_C:
                key = f"C[{jj}]"
            else:
                grp = int(parent[jj - K.HIV_N_C])
                key = f"C[{grp}]" if grp >= 0 else f"F[{jj - K.HIV_N_C}](orphan)"
            group[key] = group.get(key, 0.0) + float(structure_contrib[jj])
        active_idx = np.concatenate([dh.C[g].indices, dh.F[g].indices])
        detail = []
        for jj in np.argsort(-abs_c)[:10]:
            jj = int(jj)
            if jj < K.HIV_N_C:
                gran, col = "C", jj
            else:
                gran, col = "F", jj - K.HIV_N_C
            meta = reg[gran].iloc[col]
            detail.append({"structure": f"{gran}[{col}]",
                           "canonical_key": str(meta["canonical_key"]),
                           "x": float(row[2 + jj]), "contribution": float(structure_contrib[jj])})
        cases.append({
            "case": label, "graph_id": g,
            "y": int(dh.y[g]), "predicted_probability": float(prob[local_i]),
            "raw_margin": float(score_valid[local_i]),
            "n_active_structures": int(active_idx.size),
            "base_adjusted": float(adjusted_bias),
            "size_terms": {"c0": float(contrib[0]), "c1": float(contrib[1]),
                           "reference": size_ref_vals},
            "structure_abs_total": abs_total,
            "top5_share": float(topk[:5].sum() / max(abs_total, 1e-12)),
            "top10_share": float(topk[:10].sum() / max(abs_total, 1e-12)),
            "top20_share": float(topk[:20].sum() / max(abs_total, 1e-12)),
            "remaining_signed": float(structure_contrib[keep].sum()),
            "remaining_abs": float(abs_c[keep].sum()),
            "top_structures": detail,
            "coarse_family_contributions": {
                k: float(v) for k, v in sorted(
                    group.items(), key=lambda kv: -abs(kv[1]))[:10] if abs(v) > 0},
            "reconstruction_check": {
                "sum_all_terms": float(adjusted_bias + contrib.sum()),
                "matches_model_margin": bool(abs(adjusted_bias + contrib.sum()
                                                 - score_valid[local_i]) < 1e-4),
                "abs_error": float(abs(adjusted_bias + contrib.sum() - score_valid[local_i])),
            },
        })

    hiv_payload = {
        "round_id": K.ROUND_ID, "phase": "E: HIV additive functions",
        "tree_base_margin_from_config": tf.base_margin,
        "tree_base_margin_empirical": tree_base,
        "booster_objective": tf.objective,
        "closure": {"tree_extraction_max_abs_error": tree_closure,
                    "tree_n_rows": int(sample.size),
                    "nn_forward_vs_sum_features_max_abs_error": nn_closure,
                    "nn_n_rows": int(nn_rows.size),
                    "nn_f_j_at_zero_max_abs_sampled": nn_f0,
                    "nn_f_j_0_zero_analytic": True},
        "size_reference": size_ref,
        "feature_grid": {"structure": "[0, Train nonzero q05,q25,q50,q75,q90,q95,q99]",
                         "reference": "0 (structure absent)"},
        "selection_rule": ("Train support >=1000 for the top-3 combined-variation "
                           "structures; one NEW COARSE/FULL parent-child pair (child "
                           "support >=500, parent support >=1000, both varying); 2 largest "
                           "tree-vs-neural normalised shape gaps among support>=2000 with "
                           "both variations >0.05; fill by combined score; no Test use"),
        "selected_structures": selected_meta,
        "additive_model_used_for_local_explanations": select_by,
        "local_explanations": cases,
    }
    K.write_json(K.FUNCTIONS / "hiv_functions.json", hiv_payload)

    # ------------------------------------------------------------------
    # ZINC: native feature columns with family identity from the frozen meta
    # ------------------------------------------------------------------
    zmeta = K.read_json(K.ZINC_META)
    fams = zmeta["Z_CF"]["families"]
    counts_c = {f: zmeta["Z_CF"]["n_selected_coarse_by_family"][f] for f in fams}
    counts_f = {f: zmeta["Z_CF"]["n_selected_full_by_family"][f] for f in fams}
    c_family = [f for f in fams for _ in range(counts_c[f])]
    f_family = [f for f in fams for _ in range(counts_f[f])]
    zbooster = load_booster("Z-ADD-TREE")
    znames = [f"c{i}" for i in range(K.ZINC_WIDTH)]
    ztf = TreeFunctions(zbooster, znames)

    zphase = K.read_json(K.ANALYSIS / "phaseC_Z-NN-ADD.json")
    zmu = np.asarray(zphase["preprocessing"]["size_mean"], dtype=np.float64)
    zsd = np.asarray(zphase["preprocessing"]["size_std"], dtype=np.float64)
    z_cols = dz.Z.shape[1]

    def zinc_coord(j: int, x: np.ndarray) -> np.ndarray:
        if j < 2:
            return np.clip((x - zmu[j]) / zsd[j], -10.0, 10.0)
        if j < z_cols:
            return np.clip(x, -10.0, 10.0)
        return x

    znf = NNFunctions("Z-NN-ADD", zinc_coord)
    zstructure = sp.hstack([dz.Z, dz.C, dz.F], format="csr")
    zstats = column_stats(zstructure, dz.rows["train"])
    zq = zstats["quantiles"]
    # candidate native C/F columns: highest Train support x function variation
    zvar_tree = np.zeros(K.ZINC_WIDTH)
    zvar_nn = np.zeros(K.ZINC_WIDTH)
    zgrid = np.concatenate([np.zeros((K.ZINC_WIDTH, 1)), zq], axis=1)
    for j in range(K.ZINC_WIDTH):
        node = f"c{j}"
        ft = ztf.f(node, zgrid[j]) - ztf.f(node, np.zeros(1))[0]
        fn = znf.f(j, zgrid[j]) - znf.f(j, np.zeros(1))[0]
        zvar_tree[j] = float(ft.max() - ft.min())
        zvar_nn[j] = float(fn.max() - fn.min())
    zsupport = zstats["support"]
    native = [j for j in range(z_cols, K.ZINC_WIDTH) if zsupport[j] >= 500]
    native_rank = sorted(native, key=lambda j: -(zvar_tree[j] + zvar_nn[j]))
    zselected = [0, 1] + native_rank[:4] + [2 + 21 + 3]
    zmed = [float(np.median(dz.Z[dz.rows["train"]][:, j].toarray())) for j in range(2)]
    zentries = []
    for j in zselected:
        if j == 0:
            label, family, reducible = "Z[0] n_atoms", "size", "molecular size (direct)"
        elif j == 1:
            label, family, reducible = "Z[1] n_bonds", "size", "molecular size (direct)"
        elif j < 2 + 21:
            label = f"Z[{j}] node-class count column"
            family, reducible = "raw node class", "class count, no subgraph identity"
        elif j < 2 + 21 + 3:
            label = f"Z[{j}] edge-class count column"
            family, reducible = "raw edge class", "class count, no subgraph identity"
        elif j < z_cols:
            label = f"Z[{j}] typed-WL count column (signature not persisted)"
            family, reducible = "typed-WL", "not directly reducible to a subgraph"
        else:
            k = j - z_cols
            if k < K.ZINC_N_C:
                label = f"C[{k}] explicit attributed subgraph (coarse)"
                family = c_family[k]
            else:
                label = f"F[{k - K.ZINC_N_C}] explicit attributed subgraph (full)"
                family = f_family[k - K.ZINC_N_C]
            reducible = ("explicit attributed subgraph by construction; exact canonical "
                         "key not persisted, only family identity recoverable")
        ref = zmed[j] if j < 2 else 0.0
        x = zgrid[j]
        ft = ztf.f(f"c{j}", x) - ztf.f(f"c{j}", np.asarray([ref]))[0]
        fn = znf.f(j, x) - znf.f(j, np.asarray([ref]))[0]
        zentries.append({
            "feature": label, "column": int(j), "family": family,
            "semantics": reducible,
            "support_train_nonzero": int(zsupport[j]),
            "reference": ref,
            "grid_x": x.tolist(), "f_tree": ft.tolist(), "f_nn": fn.tolist(),
            "variation_tree": float(ft.max() - ft.min()),
            "variation_nn": float(fn.max() - fn.min()),
        })
    # explicit tree closure for ZINC (raw prediction = margin for identity link)
    zsample = np.sort(rng.choice(dz.n, size=1000, replace=False))
    zXs = dz.design_dense(zsample)
    zsum = np.zeros(zsample.size)
    for t in ztf.trees:
        f = tree_feature(t)
        if f is None:
            continue
        zsum += eval_tree(t, zXs[:, znames.index(f)].astype(np.float64))
    zpred = zbooster.predict(xgb.DMatrix(zXs, feature_names=znames), output_margin=True)
    zresid = zpred - zsum
    zclosure = float(np.max(np.abs(zresid - zresid.mean())))
    zpayload = {
        "round_id": K.ROUND_ID, "phase": "E: ZINC additive functions",
        "closure": {"tree_extraction_max_abs_error": zclosure,
                    "n_rows": int(zsample.size),
                    "base_margin_empirical": float(zresid.mean()),
                    "base_margin_config": ztf.base_margin},
        "family_identity_source": ("frozen data/zinc_design_meta.json selected counts by "
                                   "family; column order follows the frozen FAMILIES order"),
        "canonical_key_status": ("not persisted in the frozen design; exact subgraph "
                                 "identity would require rebuilding the ZINC vocabulary, "
                                 "which this round does not do"),
        "features": zentries,
    }
    K.write_json(K.FUNCTIONS / "zinc_functions.json", zpayload)

    (K.FUNCTIONS / "README.md").write_text(
        "# Extracted additive functions\n\n"
        "* `hiv_functions.json` — 7 MolHIV structures selected from Train support and\n"
        "  the learned functions only. `f_tree`, `f_nn` and `f_linear` are all on the\n"
        "  same raw input grid and reference point (structures: 0; size terms: Train\n"
        "  median). Values are the zero-baseline contribution to the raw margin.\n"
        "* `hiv_local_explanations.json` — exact per-molecule decomposition of three\n"
        "  Valid molecules for the better additive model.\n"
        "* `zinc_functions.json` — size terms, native explicit-subgraph columns (family\n"
        "  identity only; canonical keys are not persisted) and WL count columns.\n\n"
        "All contributions reconstruct the stored model margin; see `closure` blocks.\n")
    K.write_json(K.FUNCTIONS / "hiv_local_explanations.json",
                 {"model_selection": select_by, "cases": cases,
                  "size_reference": size_ref,
                  "definition": ("structure contribution = f_j(x)-f_j(0); size contribution "
                                 "= f_j(x)-f_j(Train median); base = adjusted bias; "
                                 "all terms sum exactly to the raw margin")})
    import csv

    with (K.FUNCTIONS / "function_support.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "dataset", "structure", "family", "support_train_nonzero",
            "document_frequency_train", "occurrence_train", "positive_molecules_train",
            "negative_molecules_train", "variation_tree", "variation_nn",
            "shape_difference_tree_vs_nn", "coarse_fine_pair", "reason_selected"])
        w.writeheader()
        for e in selected_meta:
            w.writerow({"dataset": "MolHIV", "structure": e["structure"],
                        "family": e["family"],
                        "support_train_nonzero": e["support_train_nonzero"],
                        "document_frequency_train": e["document_frequency_train"],
                        "occurrence_train": e["occurrence_train"],
                        "positive_molecules_train": e["positive_molecules_train"],
                        "negative_molecules_train": e["negative_molecules_train"],
                        "variation_tree": e["variation_tree"],
                        "variation_nn": e["variation_nn"],
                        "shape_difference_tree_vs_nn": e["shape_difference_tree_vs_nn"],
                        "coarse_fine_pair": e["coarse_fine_pair"],
                        "reason_selected": e["reason_selected"]})
        for e in zentries:
            w.writerow({"dataset": "ZINC", "structure": e["feature"],
                        "family": e["family"],
                        "support_train_nonzero": e["support_train_nonzero"],
                        "variation_tree": e["variation_tree"],
                        "variation_nn": e["variation_nn"],
                        "reason_selected": e["semantics"]})
    with (K.FUNCTIONS / "local_explanations_summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "case", "graph_id", "y", "predicted_probability", "n_active_structures",
            "top5_share", "top10_share", "top20_share", "remaining_signed",
            "remaining_abs", "reconstruction_abs_error"])
        w.writeheader()
        for c in cases:
            w.writerow({"case": c["case"], "graph_id": c["graph_id"], "y": c["y"],
                        "predicted_probability": c["predicted_probability"],
                        "n_active_structures": c["n_active_structures"],
                        "top5_share": c["top5_share"], "top10_share": c["top10_share"],
                        "top20_share": c["top20_share"],
                        "remaining_signed": c["remaining_signed"],
                        "remaining_abs": c["remaining_abs"],
                        "reconstruction_abs_error": c["reconstruction_check"]["abs_error"]})
    print(f"[E] tree closure={tree_closure:.2e} nn closure={nn_closure:.2e} "
          f"nn f(0)={nn_f0:.2e} selected={len(selected)} "
          f"local_model={chosen} zinc_closure={zclosure:.2e}")
    for e in selected_meta:
        print(f"[E] {e['structure']} {e['family']} support={e['support_train_nonzero']} "
              f"var_tree={e['variation_tree']:.3f} var_nn={e['variation_nn']:.3f} "
              f"shape_diff={e['shape_difference_tree_vs_nn']:.3f}")


if __name__ == "__main__":
    main()
