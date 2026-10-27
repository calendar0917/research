"""Follow-up CPU-only decomposition for the ZINC error-mechanism audit.

The first script (``zema_audit.py``) reports individual-group and pairwise
tables.  This script adds the pieces needed to answer the four headline
questions without over-reading a reused valid split:

1. A MECE P x U x L partition (label penalty x any-unseen radius-2 environment
   x permutation-invariant long-ring proxy), with tail overlaid.
2. P x U inside molecule-size strata.
3. A constant-lambda cycle attribution per penalty severity, plus a row-172
   sensitivity check.
4. A random-order permutation probe on every valid penalty row.
5. The strict train-internal OOF conditional-information test, split by
   penalty status.  This is the key check: does any new ring scale predict the
   weak OOF model's difficulty *outside* the penalty rows?
6. A train-only 5-fold OOF check of whether the new ring scales add anything to
   ``global33`` for predicting the label, both overall and without penalty rows.

No model is trained.  Labels are used only as diagnostic targets / grouping
variables, never as model inputs.  Official test is never loaded.

Run:
    uv run python \
      tracks/structural-sufficiency/zinc_error_mechanism_audit_v1/code/zema_decomposition.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
SS = REPO / "tracks/structural-sufficiency"
OUT = HERE.parent / "analysis"
OUT.mkdir(parents=True, exist_ok=True)

CYCLE_VALID = REPO / "tracks/ksvd/results/zinc_long_cycle_audit/valid_cycle_audit_label.csv"
CYCLE_TRAIN = REPO / "tracks/ksvd/results/zinc_long_cycle_audit/train_cycle_audit_label.csv"
PER_MOL = SS / "zinc_generalization_audit_v1/analysis/per_molecule.csv"
DCCL_PRED = SS / "zinc_performance_search_v1/analysis/per_molecule_valid.csv"
BOSI_PRED = SS / "zinc_bosi_joint_moment_v1/runs/B_s0/valid_best.npz"
ZBO_R_PRED = SS / "zinc_block_ownership_v1/runs/R_s0/valid_best.npz"
ZBO_N_PRED = SS / "zinc_block_ownership_v1/runs/N_s0/valid_best.npz"
ZBO_S_PRED = SS / "zinc_block_ownership_v2/runs/S_s0/valid_best.npz"
AD_TRAIN = SS / "zinc_arch_discovery_v1/cache/ad_train.pt"
AD_VALID = SS / "zinc_arch_discovery_v1/cache/ad_valid.pt"
OOF = REPO / "tracks/ksvd/results/oof_difficulty_audit/oof_per_molecule.csv"

# From tracks/structure-contribution/code/structural_relation_objects_v1.py
GLOBAL33_NAMES = [
    "log1p_n_atoms",
    "log1p_n_bonds",
    "log1p_n_junctions",
    "cyclomatic_rank",
    "ring_atom_fraction",
    "ring_bond_fraction",
    "n_ring_systems",
    "diameter",
    "mean_distance",
    "frac_dist_ge3",
    "bonds_per_atom",
    "log1p_n_ring_atoms",
    "max_degree",
    "terminal_atom_fraction",
    "junction_per_bond",
    "frac_junction_centers_ring",
    "frac_deg2_junctions",
    "frac_deg3_junctions",
    "frac_deg4_junctions",
    "bond_type0_frac",
    "bond_type1_frac",
    "bond_type2_frac",
    "bond_type3_frac",
    "ring_size3_frac",
    "ring_size4_frac",
    "ring_size5_frac",
    "ring_size6_frac",
    "ring_size7_frac",
    "ring_size8plus_frac",
    "fused_ring_atom_fraction",
    "mean_pair_distance",
    "distinct_bond_types",
    "distinct_atom_types",
]

OOF_RARITY = [
    "oov_patch_ratio",
    "rare_le1_ratio",
    "rare_le2_ratio",
    "rare_le5_ratio",
    "rare_le10_ratio",
    "min_train_frequency",
    "mean_train_frequency",
    "median_train_frequency",
    "mean_log1p_train_frequency",
    "mean_inverse_of_1p_frequency",
    "num_unique_patch_tokens",
    "unique_patch_ratio",
]
OOF_STRUCTURE = [
    "num_nodes",
    "num_edges",
    "num_patches",
    "num_pairs",
    "diameter",
    "mean_shortest_path_distance",
    "max_shortest_path_distance",
    "average_degree",
    "max_degree",
    "num_branching_nodes",
    "branching_node_ratio",
    "cycle_rank",
    "num_components",
    "density",
    "fraction_nodes_in_cycles",
    "num_simple_cycles_bounded",
    "cycle_length_mean",
    "num_nodes_in_cycles",
    "frac_dist_gt_2",
]
OOF_RING_NEW = [
    "topo_longest_simple_cycle",
    "topo_mcb_max_length",
    "topo_mcb_mean_length",
    "topo_mcb_total_length",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def npz_pred(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=True) as z:
        row = np.asarray(z["row"], dtype=np.int64)
        pred = np.asarray(z["pred"], dtype=np.float64)
    order = np.argsort(row)
    assert np.array_equal(row[order], np.arange(1000)), path
    return pred[order]


def load_valid():
    cyc = pd.read_csv(CYCLE_VALID).sort_values("subset_index").set_index("subset_index")
    pm = pd.read_csv(PER_MOL).set_index("row_id")
    dcc = pd.read_csv(DCCL_PRED).set_index("row_id")
    assert np.array_equal(cyc.index.to_numpy(), np.arange(1000))
    assert np.array_equal(pm.index.to_numpy(), np.arange(1000))
    assert np.array_equal(dcc.index.to_numpy(), np.arange(1000))
    y = pm["y_raw"].to_numpy(dtype=np.float64)
    assert np.max(np.abs(y - cyc["target"].to_numpy(dtype=np.float64))) < 1e-6
    preds: dict[str, np.ndarray] = {}
    for s in range(4):
        preds[f"DCCL_s{s}"] = dcc[f"pred_B-R0-online-s{s}"].to_numpy(dtype=np.float64)
    preds["DCCL_mean"] = np.mean([preds[f"DCCL_s{s}"] for s in range(4)], axis=0)
    preds["BOSI_B_s0"] = npz_pred(BOSI_PRED)
    preds["ZBO_R_s0"] = npz_pred(ZBO_R_PRED)
    preds["ZBO_N_s0"] = npz_pred(ZBO_N_PRED)
    preds["ZBO_S_s0"] = npz_pred(ZBO_S_PRED)
    return cyc, pm, preds, y


def cell_metrics(pred: np.ndarray, y: np.ndarray, mask: np.ndarray, total_ae: float) -> dict[str, Any]:
    m = np.asarray(mask, dtype=bool)
    n = int(m.sum())
    if n == 0:
        return {"n": 0, "mae": None, "ae_share": 0.0, "signed": None, "mean_y": None}
    ae = np.abs(pred[m] - y[m])
    return {
        "n": n,
        "mae": float(ae.mean()),
        "ae_share": float(ae.sum() / total_ae),
        "signed": float((pred[m] - y[m]).mean()),
        "mean_y": float(y[m].mean()),
    }


def main() -> int:
    cyc, pm, preds, y = load_valid()
    n = len(y)
    penalty = cyc["label_excess"].to_numpy(dtype=np.float64) > 0
    severe = cyc["label_excess"].to_numpy(dtype=np.float64) >= 2
    unseen = pm["r2_unseen_atom_frac"].to_numpy(dtype=np.float64) > 0
    low4 = (
        (pm["r2_unseen_atom_frac"].to_numpy(dtype=np.float64) == 0)
        & (pm["r2_low4_atom_frac"].to_numpy(dtype=np.float64) > 0)
    )
    full = (
        (pm["r2_unseen_atom_frac"].to_numpy(dtype=np.float64) == 0)
        & (pm["r2_low4_atom_frac"].to_numpy(dtype=np.float64) == 0)
    )
    tail = y <= -3
    long7 = cyc["longest_simple_cycle_length_exact_all"].to_numpy(dtype=np.float64) >= 7
    size = pm["n_atoms"].to_numpy(dtype=int)

    # ------------------------------------------------------------------
    # 1. MECE P x U x L.  L is the exact longest simple cycle >= 7, the
    #    permutation-invariant long-ring proxy.  (The label penalty is a
    #    basis-order quantity; see the permutation probe below.)
    # ------------------------------------------------------------------
    mece_rows: list[dict[str, Any]] = []
    for model, pred in preds.items():
        total_ae = float(np.abs(pred - y).sum())
        for p in (False, True):
            for u in (False, True):
                for l in (False, True):
                    m = (penalty == p) & (unseen == u) & (long7 == l)
                    row = {
                        "model": model,
                        "penalty": int(p),
                        "unseen": int(u),
                        "long7": int(l),
                        **cell_metrics(pred, y, m, total_ae),
                        "n_tail": int((m & tail).sum()),
                        "ae_share_tail": float(np.abs(pred[m & tail] - y[m & tail]).sum() / total_ae),
                    }
                    mece_rows.append(row)
    mece = pd.DataFrame(mece_rows)
    mece.to_csv(OUT / "mece_penalty_unseen_long7.csv", index=False)

    # ------------------------------------------------------------------
    # 2. P x U inside size strata.  Two cut sets: the train cuts used by the
    #    DCC-L audit (21/25) and a coarser sensitivity (22/26).
    # ------------------------------------------------------------------
    size_rows: list[dict[str, Any]] = []
    for cuts in ([21, 25], [22, 26]):
        sb = np.digitize(size, cuts)
        for model, pred in preds.items():
            total_ae = float(np.abs(pred - y).sum())
            for b in range(3):
                for p in (False, True):
                    for u in (False, True):
                        m = (sb == b) & (penalty == p) & (unseen == u)
                        size_rows.append(
                            {
                                "cuts": "/".join(map(str, cuts)),
                                "size_bin": b,
                                "size_min": int(size[sb == b].min()) if (sb == b).any() else None,
                                "size_max": int(size[sb == b].max()) if (sb == b).any() else None,
                                "model": model,
                                "penalty": int(p),
                                "unseen": int(u),
                                **cell_metrics(pred, y, m, total_ae),
                            }
                        )
    pd.DataFrame(size_rows).to_csv(OUT / "penalty_x_unseen_x_size.csv", index=False)

    # ------------------------------------------------------------------
    # 3. Constant-lambda cycle attribution.  For a severity group,
    #       lambda = 1 + mean(residual) / mean(cycle_component)
    #    so the signed residual mass attributable to the uncaptured cycle
    #    term is (1-lambda) * sum(cycle_component), and the remaining AE mass
    #    is the non-cycle error on those rows.  This is an approximation, not
    #    a per-row oracle.
    # ------------------------------------------------------------------
    scal_rows: list[dict[str, Any]] = []
    comp = cyc["label_cycle_component"].to_numpy(dtype=np.float64)
    sev_groups = [
        ("excess=1", cyc["label_excess"].to_numpy(dtype=np.float64) == 1),
        ("excess=2", cyc["label_excess"].to_numpy(dtype=np.float64) == 2),
        ("excess>=4", cyc["label_excess"].to_numpy(dtype=np.float64) >= 4),
        ("all_penalty", penalty),
        ("all_penalty_no_172", penalty & (np.arange(n) != 172)),
        ("excess=1_no_172", (cyc["label_excess"].to_numpy(dtype=np.float64) == 1) & (np.arange(n) != 172)),
    ]
    for model, pred in preds.items():
        r = pred - y
        total_ae = float(np.abs(r).sum())
        for name, m in sev_groups:
            if not m.any():
                continue
            lam = 1.0 + float(r[m].mean()) / float(comp[m].mean())
            signed_cycle_mass = (1.0 - lam) * float(comp[m].sum())
            ae_mass = float(np.abs(r[m]).sum())
            scal_rows.append(
                {
                    "model": model,
                    "group": name,
                    "n": int(m.sum()),
                    "mean_cycle_component": float(comp[m].mean()),
                    "mean_residual": float(r[m].mean()),
                    "lambda": lam,
                    "signed_cycle_mass": signed_cycle_mass,
                    "ae_mass": ae_mass,
                    "ae_share_total": ae_mass / total_ae,
                    "cycle_share_of_group_ae": abs(signed_cycle_mass) / ae_mass if ae_mass else None,
                    "hard_ceiling_delta_mae": ae_mass / n,
                    "cycle_delta_mae": abs(signed_cycle_mass) / n,
                }
            )
    scal = pd.DataFrame(scal_rows)
    scal.to_csv(OUT / "cycle_attribution.csv", index=False)

    # ------------------------------------------------------------------
    # 4. Random-order probe on every valid penalty row.  adjacency comes from
    #    the frozen valid cache; the label itself is the generator-order value.
    # ------------------------------------------------------------------
    va = torch.load(AD_VALID, map_location="cpu", weights_only=False)
    atom_ptr = np.asarray(va["atom_ptr"], dtype=np.int64)
    bond_ptr = np.asarray(va["bond_ptr"], dtype=np.int64)
    bond_index = np.asarray(va["bond_index"], dtype=np.int64)
    rng = np.random.default_rng(20261027)
    order_rows: list[dict[str, Any]] = []
    for r in np.flatnonzero(penalty):
        a0, a1 = atom_ptr[r], atom_ptr[r + 1]
        b0, b1 = bond_ptr[r], bond_ptr[r + 1]
        edges = [tuple(map(int, bond_index[k])) for k in range(b0, b1)]
        n_atoms = int(a1 - a0)
        excess = []
        for _ in range(50):
            perm = rng.permutation(n_atoms)
            g = nx.Graph()
            g.add_nodes_from(range(n_atoms))
            g.add_edges_from((int(perm[u]), int(perm[v])) for u, v in edges)
            mx = max((len(c) for c in nx.cycle_basis(g)), default=0)
            excess.append(max(0, mx - 6))
        ex = np.asarray(excess, dtype=np.int64)
        lab = int(cyc["label_excess"].to_numpy(dtype=np.float64)[r])
        order_rows.append(
            {
                "row": int(r),
                "label_excess": lab,
                "n_atoms": n_atoms,
                "perm_min": int(ex.min()),
                "perm_max": int(ex.max()),
                "perm_mean": float(ex.mean()),
                "match_fraction": float(np.mean(ex == lab)),
                "n_unique_excess": int(len(np.unique(ex))),
                "unique_excess": "/".join(map(str, sorted(np.unique(ex).tolist()))),
                "label_is_modal": bool(lab == np.bincount(ex).argmax()),
            }
        )
    order = pd.DataFrame(order_rows)
    order.to_csv(OUT / "penalty_order_sensitivity.csv", index=False)
    order_summary = {
        "penalty_rows": int(len(order)),
        "order_sensitive_rows": int((order["n_unique_excess"] > 1).sum()),
        "order_sensitive_rows_list": order.loc[order["n_unique_excess"] > 1, "row"].tolist(),
        "mean_match_fraction": float(order["match_fraction"].mean()),
        "label_not_modal_rows": order.loc[~order["label_is_modal"], "row"].tolist(),
        "permutations_per_row": 50,
        "note": (
            "Random node orders are a diagnostic of how arbitrary the basis-order "
            "statistic is.  They are not the generator's RDKit canonical order; a "
            "model could in principle learn a canonicalisation, so this is a "
            "difficulty diagnostic, not an information-theoretic lower bound."
        ),
    }

    # ------------------------------------------------------------------
    # 5. Strict train-internal OOF conditional-information test.  The only
    #    strict OOF in the repo is compact-v4-hinge (MAE 0.175), not DCC-L.
    #    Split the test by penalty status: the all-rows gain is entirely from
    #    the penalty rows if the no-penalty delta is zero.
    # ------------------------------------------------------------------
    oof = pd.read_csv(OOF)
    ctr = pd.read_csv(CYCLE_TRAIN).sort_values("subset_index").reset_index(drop=True)
    oof = oof.merge(
        ctr[
            [
                "subset_index",
                "mean_maxbasis",
                "frac_ge7",
                "max_min_basis_cycle_length",
                "total_min_basis_cycle_length",
            ]
        ],
        on="subset_index",
        how="left",
        validate="one_to_one",
    )
    assert oof[["mean_maxbasis", "frac_ge7"]].notna().all().all()
    oof["mean_excess"] = np.maximum(0.0, oof["mean_maxbasis"] - 6.0)
    topo = [c for c in oof.columns if c.startswith("topo_")]
    topo_nonring = [
        c
        for c in topo
        if c not in OOF_RING_NEW + ["topo_longest", "topo_longest_squared", "topo_mcb_count"]
    ]
    oof_variants = {
        "base_nonring": OOF_RARITY + OOF_STRUCTURE + topo_nonring,
        "plus_new_ring": OOF_RARITY + OOF_STRUCTURE + topo_nonring + OOF_RING_NEW,
        "plus_mean_excess": OOF_RARITY + OOF_STRUCTURE + topo_nonring + ["mean_excess", "frac_ge7"],
        "plus_new_ring_mean_excess": OOF_RARITY
        + OOF_STRUCTURE
        + topo_nonring
        + OOF_RING_NEW
        + ["mean_excess", "frac_ge7"],
    }

    def cross_fit(df: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
        X = df[cols].to_numpy(dtype=np.float64)
        log_ae = np.log(df["mean_absolute_error"].to_numpy(dtype=np.float64) + 1e-3)
        folds = df["outer_fold"].to_numpy(dtype=np.int64)
        pred = np.full(len(df), np.nan)
        for f in range(5):
            tr = folds != f
            te = folds == f
            if tr.sum() < 10:
                continue
            scaler = StandardScaler().fit(X[tr])
            model = Ridge(alpha=1.0).fit(scaler.transform(X[tr]), log_ae[tr])
            pred[te] = model.predict(scaler.transform(X[te]))
        ae_pred = np.clip(np.exp(pred) - 1e-3, 0.0, None)
        return pred, ae_pred

    cond_rows: list[dict[str, Any]] = []
    for subset_name, mask in [
        ("all", np.ones(len(oof), dtype=bool)),
        ("no_penalty", oof["label_excess"].to_numpy(dtype=np.float64) == 0),
        ("penalty", oof["label_excess"].to_numpy(dtype=np.float64) > 0),
    ]:
        df = oof[mask].reset_index(drop=True)
        ae = df["mean_absolute_error"].to_numpy(dtype=np.float64)
        log_ae = np.log(ae + 1e-3)
        folds = df["outer_fold"].to_numpy(dtype=np.int64)
        preds_sub: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for vname, cols in oof_variants.items():
            p, pe = cross_fit(df, cols)
            preds_sub[vname] = (p, pe)
            ok = np.isfinite(p)
            cond_rows.append(
                {
                    "subset": subset_name,
                    "variant": vname,
                    "n": int(ok.sum()),
                    "n_features": len(cols),
                    "r2_log": float(r2_score(log_ae[ok], p[ok])),
                    "spearman_pred_ae": float(spearmanr(pe[ok], ae[ok]).statistic),
                    "mae_top_quintile": float(
                        ae[ok][pe[ok] >= np.quantile(pe[ok], 0.8)].mean()
                    ),
                }
            )
        base = preds_sub["base_nonring"][0]
        for vname in ("plus_new_ring", "plus_mean_excess", "plus_new_ring_mean_excess"):
            p = preds_sub[vname][0]
            for f in range(5):
                m = folds == f
                if not m.any():
                    continue
                cond_rows.append(
                    {
                        "subset": subset_name,
                        "variant": f"{vname}__fold{f}",
                        "n": int(m.sum()),
                        "n_features": None,
                        "delta_r2_log": float(r2_score(log_ae[m], p[m]) - r2_score(log_ae[m], base[m])),
                        "delta_spearman": float(
                            spearmanr(np.exp(p[m]) - 1e-3, ae[m]).statistic
                            - spearmanr(np.exp(base[m]) - 1e-3, ae[m]).statistic
                        ),
                    }
                )
    pd.DataFrame(cond_rows).to_csv(OUT / "oof_conditional.csv", index=False)

    # Individual correlations on the no-penalty subset, with the same partial
    # controls as the first script (size, rarity, cycle rank).
    from scipy.stats import rankdata

    def partial_spearman(x, yv, controls):
        def resid(v):
            r = rankdata(np.asarray(v, dtype=float), method="average")
            c = np.column_stack(
                [np.ones(len(r))] + [rankdata(np.asarray(z, dtype=float), method="average") for z in controls]
            )
            return r - c @ np.linalg.lstsq(c, r, rcond=None)[0]

        return float(spearmanr(resid(x), resid(yv)).statistic)

    np_mask = oof["label_excess"].to_numpy(dtype=np.float64) == 0
    no_pen = oof[np_mask]
    corr_rows = []
    for c in OOF_RING_NEW + ["mean_excess", "frac_ge7"]:
        corr_rows.append(
            {
                "feature": c,
                "subset": "no_penalty",
                "n": int(np_mask.sum()),
                "spearman_ae": float(spearmanr(no_pen[c], no_pen["mean_absolute_error"]).statistic),
                "partial_spearman_ae_size_rarity_cycle": partial_spearman(
                    no_pen[c].to_numpy(dtype=np.float64),
                    no_pen["mean_absolute_error"].to_numpy(dtype=np.float64),
                    [
                        no_pen["num_nodes"].to_numpy(dtype=np.float64),
                        no_pen["rare_le5_ratio"].to_numpy(dtype=np.float64),
                        no_pen["cycle_rank"].to_numpy(dtype=np.float64),
                    ],
                ),
                "per_fold_spearman_ae": [
                    float(
                        spearmanr(
                            no_pen.loc[no_pen["outer_fold"] == f, c],
                            no_pen.loc[no_pen["outer_fold"] == f, "mean_absolute_error"],
                        ).statistic
                    )
                    for f in range(5)
                ],
            }
        )
    pd.DataFrame(corr_rows).to_csv(OUT / "oof_ring_correlations_no_penalty.csv", index=False)

    # ------------------------------------------------------------------
    # 6. Train-only 5-fold OOF: does global33 + ring scale beat global33 for
    #    the label?  This is a diagnostic of graph information, not a model
    #    proposal.  Use the OOF fold assignment for consistency.
    # ------------------------------------------------------------------
    tr = torch.load(AD_TRAIN, map_location="cpu", weights_only=False)
    G = np.asarray(tr["global_feat"], dtype=np.float64)
    y_tr = np.asarray(tr["labels"], dtype=np.float64)
    assert G.shape == (10000, 33)
    assert np.max(np.abs(y_tr - ctr["target"].to_numpy(dtype=np.float64))) < 1e-6
    folds_tr = oof["outer_fold"].to_numpy(dtype=np.int64)
    assert np.array_equal(oof["subset_index"].to_numpy(dtype=np.int64), np.arange(10000))
    ring_cols = [
        "longest_simple_cycle_length_exact_all",
        "max_min_basis_cycle_length",
        "total_min_basis_cycle_length",
        "mean_maxbasis",
        "frac_ge7",
    ]
    R = ctr[ring_cols].to_numpy(dtype=np.float64)
    X_base = G
    X_ring = np.column_stack([G, R])
    y_pen = ctr["label_excess"].to_numpy(dtype=np.float64) > 0
    g33_rows: list[dict[str, Any]] = []
    preds_g: dict[str, np.ndarray] = {}
    for name, X in [("global33", X_base), ("global33+ring_scale", X_ring)]:
        pred = np.full(10000, np.nan)
        for f in range(5):
            trm = folds_tr != f
            tem = folds_tr == f
            model = HistGradientBoostingRegressor(
                max_iter=300,
                learning_rate=0.05,
                max_leaf_nodes=31,
                l2_regularization=1.0,
                random_state=0,
            )
            model.fit(X[trm], y_tr[trm])
            pred[tem] = model.predict(X[tem])
        preds_g[name] = pred
        for subset, m in [
            ("all", np.ones(10000, dtype=bool)),
            ("no_penalty", ~y_pen),
            ("penalty", y_pen),
        ]:
            g33_rows.append(
                {
                    "features": name,
                    "subset": subset,
                    "n": int(m.sum()),
                    "mae": float(mean_absolute_error(y_tr[m], pred[m])),
                    "r2": float(r2_score(y_tr[m], pred[m])),
                }
            )
    base_pred = preds_g["global33"]
    ring_pred = preds_g["global33+ring_scale"]
    for f in range(5):
        m = folds_tr == f
        g33_rows.append(
            {
                "features": "delta_ring_minus_global33",
                "subset": f"fold{f}",
                "n": int(m.sum()),
                "delta_mae": float(
                    mean_absolute_error(y_tr[m], ring_pred[m])
                    - mean_absolute_error(y_tr[m], base_pred[m])
                ),
                "delta_r2": float(r2_score(y_tr[m], ring_pred[m]) - r2_score(y_tr[m], base_pred[m])),
            }
        )
    pd.DataFrame(g33_rows).to_csv(OUT / "global33_target_oof.csv", index=False)

    # ------------------------------------------------------------------
    # Persist a compact JSON with provenance and the headline of each block.
    # ------------------------------------------------------------------
    payload = {
        "protocol": "zinc-error-mechanism-audit-v1::decomposition",
        "official_test_loaded": False,
        "official_valid_loaded": True,
        "valid_is_exposed_development_split": True,
        "inputs": {
            str(p): sha256(p)
            for p in [
                CYCLE_VALID,
                CYCLE_TRAIN,
                PER_MOL,
                DCCL_PRED,
                BOSI_PRED,
                ZBO_R_PRED,
                ZBO_N_PRED,
                ZBO_S_PRED,
                AD_TRAIN,
                AD_VALID,
                OOF,
            ]
        },
        "mece_headline_dccl_mean": mece[
            (mece["model"] == "DCCL_mean") & (mece["n"] > 0)
        ].to_dict(orient="records"),
        "cycle_attribution_headline": scal[scal["model"] == "DCCL_mean"].to_dict(
            orient="records"
        ),
        "order_sensitivity": order_summary,
        "oof_conditional_headline": [
            r
            for r in cond_rows
            if r.get("variant") in {"base_nonring", "plus_new_ring", "plus_mean_excess"}
            and "fold" not in str(r.get("variant"))
        ],
        "global33_target_oof": g33_rows,
    }
    (OUT / "decomposition.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Print a compact human digest.
    pd.set_option("display.width", 220)
    print("== MECE P x U x L (DCCL_mean) ==")
    print(
        mece[(mece["model"] == "DCCL_mean") & (mece["n"] > 0)][
            ["penalty", "unseen", "long7", "n", "mae", "ae_share", "signed", "n_tail"]
        ].to_string(index=False)
    )
    print("\n== cycle attribution (DCCL_mean) ==")
    print(
        scal[scal["model"] == "DCCL_mean"][
            [
                "group",
                "n",
                "mean_cycle_component",
                "mean_residual",
                "lambda",
                "ae_mass",
                "cycle_share_of_group_ae",
                "hard_ceiling_delta_mae",
                "cycle_delta_mae",
            ]
        ].to_string(index=False)
    )
    print("\n== penalty row order probe ==")
    print(json.dumps(order_summary, indent=1))
    print("\n== OOF conditional (all / no_penalty / penalty) ==")
    print(
        pd.DataFrame(cond_rows)
        .query("variant in ['base_nonring','plus_new_ring','plus_mean_excess']")
        .to_string(index=False)
    )
    print("\n== global33 target OOF ==")
    print(pd.DataFrame(g33_rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
