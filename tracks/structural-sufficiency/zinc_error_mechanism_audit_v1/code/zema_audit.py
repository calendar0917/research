"""CPU-only error-mechanism audit for the frozen ZINC valid predictions.

The audit answers four narrow questions:

1. How much of the strong-model MAE can be removed by an oracle that perfectly
   repairs every molecule carrying the benchmark's artificial cycle penalty?
2. Once those penalty rows are excluded, is the unseen radius-2 environment
   group still the main error concentration?
3. Does the existing ``global33`` already contain the per-atom minimum-ring
   information, and are exact longest-simple-cycle / minimum-cycle-basis scales
   new graph information?
4. Is there a strict train-only OOF prediction that licenses a ring-feature
   claim for a strong model?

No model is trained here.  The only fitting is a CPU diagnostic regressor that
checks whether graph-level ring scales are predictable from ``global33`` and a
Ridge meta-model that checks whether ring scales predict OOF difficulty.  Labels
are used only as diagnostic targets; they are never model inputs.  Official test
is never loaded.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
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

# frozen ``GLOBAL_FEAT_NAMES`` from tracks/structure-contribution/code/
# structural_relation_objects_v1.py
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


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rank_residual(values: np.ndarray, controls: Iterable[np.ndarray]) -> np.ndarray:
    r = rankdata(np.asarray(values, dtype=float), method="average")
    cols = [np.ones(len(r), dtype=float)]
    cols.extend(rankdata(np.asarray(c, dtype=float), method="average") for c in controls)
    c = np.column_stack(cols)
    return r - c @ np.linalg.lstsq(c, r, rcond=None)[0]


def partial_spearman(
    x: np.ndarray, y: np.ndarray, controls: Iterable[np.ndarray]
) -> float:
    xr = rank_residual(x, controls)
    yr = rank_residual(y, controls)
    return float(spearmanr(xr, yr).statistic)


def npz_pred(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=True) as z:
        row = np.asarray(z["row"], dtype=np.int64)
        pred = np.asarray(z["pred"], dtype=np.float64)
    order = np.argsort(row)
    assert np.array_equal(row[order], np.arange(1000)), path
    return pred[order]


def metric_row(pred: np.ndarray, y: np.ndarray, mask: np.ndarray, total_ae: float) -> dict[str, Any]:
    m = np.asarray(mask, dtype=bool)
    n = int(m.sum())
    if n == 0:
        return {
            "n": 0,
            "mae": None,
            "ae_share": 0.0,
            "signed": None,
            "median_ae": None,
            "p90_ae": None,
            "mean_y": None,
        }
    ae = np.abs(pred[m] - y[m])
    signed = pred[m] - y[m]
    return {
        "n": n,
        "mae": float(ae.mean()),
        "ae_share": float(ae.sum() / total_ae),
        "signed": float(signed.mean()),
        "median_ae": float(np.median(ae)),
        "p90_ae": float(np.percentile(ae, 90)),
        "mean_y": float(y[m].mean()),
    }


def add_metrics(
    rows: list[dict[str, Any]],
    table: str,
    model: str,
    group: str,
    pred: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    total_ae: float,
) -> None:
    m = metric_row(pred, y, mask, total_ae)
    rows.append({"table": table, "model": model, "group": group, **m})


def load_data():
    cyc = pd.read_csv(CYCLE_VALID).sort_values("subset_index").set_index("subset_index")
    pm = pd.read_csv(PER_MOL).set_index("row_id")
    dcc = pd.read_csv(DCCL_PRED).set_index("row_id")
    assert np.array_equal(cyc.index.to_numpy(), np.arange(1000))
    assert np.array_equal(pm.index.to_numpy(), np.arange(1000))
    assert np.array_equal(dcc.index.to_numpy(), np.arange(1000))
    y = pm["y_raw"].to_numpy(dtype=np.float64)
    assert np.max(np.abs(y - cyc["target"].to_numpy(dtype=np.float64))) < 1e-6
    assert np.max(np.abs(y - dcc["y_raw"].to_numpy(dtype=np.float64))) < 1e-6

    preds: dict[str, np.ndarray] = {}
    for s in range(4):
        preds[f"DCCL_s{s}"] = dcc[f"pred_B-R0-online-s{s}"].to_numpy(dtype=np.float64)
    preds["DCCL_mean"] = np.mean([preds[f"DCCL_s{s}"] for s in range(4)], axis=0)
    preds["BOSI_B_s0"] = npz_pred(BOSI_PRED)
    preds["ZBO_R_s0"] = npz_pred(ZBO_R_PRED)
    preds["ZBO_N_s0"] = npz_pred(ZBO_N_PRED)
    preds["ZBO_S_s0"] = npz_pred(ZBO_S_PRED)
    for name, p in preds.items():
        assert p.shape == (1000,), name
        assert np.isfinite(p).all(), name
    return cyc, pm, preds, y


def basic_model_table(preds: dict[str, np.ndarray], y: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, p in preds.items():
        ae = np.abs(p - y)
        out[name] = {
            "n": 1000,
            "mae": float(ae.mean()),
            "median_ae": float(np.median(ae)),
            "p90_ae": float(np.percentile(ae, 90)),
            "mean_signed": float((p - y).mean()),
        }
    # The historical DCC-L headline is the mean of per-seed MAEs, not the MAE of
    # the 4-seed prediction average.  Record both explicitly.
    per_seed = [np.mean(np.abs(preds[f"DCCL_s{s}"] - y)) for s in range(4)]
    out["DCCL_per_seed_mean"] = {
        "n": 1000,
        "mae": float(np.mean(per_seed)),
        "per_seed_mae": [float(x) for x in per_seed],
    }
    return out


def main() -> int:
    cyc, pm, preds, y = load_data()
    n = len(y)
    total_ae = {k: float(np.abs(p - y).sum()) for k, p in preds.items()}

    # ------------------------------------------------------------------
    # Masks.  These are fixed before inspecting any group metric.
    # ------------------------------------------------------------------
    excess = cyc["label_excess"].to_numpy(dtype=np.float64)
    penalty = excess > 0
    severe = excess >= 2
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
    longest = cyc["longest_simple_cycle_length_exact_all"].to_numpy(dtype=np.float64)
    long7 = longest >= 7
    long10 = longest >= 10
    rdkit_large = cyc["rdkit_rings_gt_6"].to_numpy(dtype=np.float64) > 0
    mcb_gt6 = cyc["max_min_basis_cycle_length"].to_numpy(dtype=np.float64) > 6
    meanmaxbasis_gt6 = cyc["mean_maxbasis"].to_numpy(dtype=np.float64) > 6

    masks = {
        "all": np.ones(n, dtype=bool),
        "label_penalty": penalty,
        "label_penalty_severe": severe,
        "no_label_penalty": ~penalty,
        "r2_unseen": unseen,
        "r2_low4_only": low4,
        "r2_full": full,
        "tail_y_le_minus3": tail,
        "exact_longest_ge7": long7,
        "exact_longest_ge10": long10,
        "rdkit_large_gt6": rdkit_large,
        "mcb_max_gt6": mcb_gt6,
        "mean_maxbasis_gt6": meanmaxbasis_gt6,
    }

    rows: list[dict[str, Any]] = []
    for model, p in preds.items():
        for group, mask in masks.items():
            add_metrics(rows, "individual", model, group, p, y, mask, total_ae[model])

    # ------------------------------------------------------------------
    # MECE cross-tabs.  Each cell is disjoint; reported separately from the
    # overlapping individual groups above.
    # ------------------------------------------------------------------
    cross_rows: list[dict[str, Any]] = []
    # penalty x unseen
    for model, p in preds.items():
        for a, aname in [(True, "penalty"), (False, "no_penalty")]:
            for b, bname in [(True, "unseen"), (False, "seen")]:
                mask = (penalty == a) & (unseen == b)
                add_metrics(cross_rows, "penalty_x_unseen", model, f"{aname}__{bname}", p, y, mask, total_ae[model])

    # penalty x support tier
    tier = np.where(unseen, "unseen", np.where(low4, "low4", "full"))
    for model, p in preds.items():
        for a, aname in [(True, "penalty"), (False, "no_penalty")]:
            for b in ["full", "low4", "unseen"]:
                mask = (penalty == a) & (tier == b)
                add_metrics(cross_rows, "penalty_x_support", model, f"{aname}__{b}", p, y, mask, total_ae[model])

    # penalty x tail
    for model, p in preds.items():
        for a, aname in [(True, "penalty"), (False, "no_penalty")]:
            for b, bname in [(True, "tail"), (False, "ordinary")]:
                mask = (penalty == a) & (tail == b)
                add_metrics(cross_rows, "penalty_x_tail", model, f"{aname}__{bname}", p, y, mask, total_ae[model])

    # long-cycle proxy x penalty, for the three common proxies
    for model, p in preds.items():
        for prox_name, prox in [
            ("exact_longest_ge7", long7),
            ("exact_longest_ge10", long10),
            ("rdkit_large_gt6", rdkit_large),
        ]:
            for a, aname in [(True, "penalty"), (False, "no_penalty")]:
                for b, bname in [(True, "proxy"), (False, "nonproxy")]:
                    mask = (penalty == a) & (prox == b)
                    add_metrics(
                        cross_rows,
                        f"{prox_name}_x_penalty",
                        model,
                        f"{aname}__{bname}",
                        p,
                        y,
                        mask,
                        total_ae[model],
                    )

    # No-penalty support tier by size tertile (train cuts 21/25, from the
    # DCC-L audit).  This is the size control for the unseen effect.
    size = pm["n_atoms"].to_numpy(dtype=int)
    size_bin = np.digitize(size, [21, 25])
    for model, p in preds.items():
        for b in range(3):
            for sname, smask in [("full", full), ("low4", low4), ("unseen", unseen)]:
                mask = (~penalty) & (size_bin == b) & smask
                add_metrics(
                    cross_rows,
                    "no_penalty_support_x_size_tertile",
                    model,
                    f"size_bin{b}__{sname}",
                    p,
                    y,
                    mask,
                    total_ae[model],
                )

    # ------------------------------------------------------------------
    # Oracle ceiling: zero all absolute error on a fixed row set.  This is an
    # upper bound on what "perfect repair" of that set could buy; it also
    # zeroes non-cycle error on those rows, so it is not an achievable model.
    # ------------------------------------------------------------------
    oracle_rows: list[dict[str, Any]] = []
    oracle_sets = {
        "label_penalty": penalty,
        "label_penalty_severe": severe,
        "rdkit_large_gt6": rdkit_large,
        "mcb_max_gt6": mcb_gt6,
        "mean_maxbasis_gt6": meanmaxbasis_gt6,
        "exact_longest_ge7": long7,
        "exact_longest_ge10": long10,
    }
    for model, p in preds.items():
        ae = np.abs(p - y)
        base_mae = float(ae.mean())
        for name, mask in oracle_sets.items():
            m = np.asarray(mask, dtype=bool)
            reduction = float(ae[m].sum() / n)
            oracle_rows.append(
                {
                    "model": model,
                    "group": name,
                    "n": int(m.sum()),
                    "base_mae": base_mae,
                    "new_mae_if_zero_ae": base_mae - reduction,
                    "delta_mae": reduction,
                    "ae_share": float(ae[m].sum() / ae.sum()) if ae.sum() else 0.0,
                    "note": "oracle: zero all AE on the fixed group; upper bound, not model-achievable",
                }
            )

    # ------------------------------------------------------------------
    # global33: does it already contain the ring scale?
    # ------------------------------------------------------------------
    import torch

    tr = torch.load(AD_TRAIN, map_location="cpu", weights_only=False)
    va = torch.load(AD_VALID, map_location="cpu", weights_only=False)
    G_train = np.asarray(tr["global_feat"], dtype=np.float64)
    G_valid = np.asarray(va["global_feat"], dtype=np.float64)
    y_train = np.asarray(tr["labels"], dtype=np.float64)
    y_valid = np.asarray(va["labels"], dtype=np.float64)
    ctr = pd.read_csv(CYCLE_TRAIN).sort_values("subset_index").reset_index(drop=True)
    assert G_train.shape == (10000, 33), G_train.shape
    assert np.max(np.abs(y_train - ctr["target"].to_numpy(dtype=np.float64))) < 1e-6
    assert np.max(np.abs(y_valid - y)) < 1e-6

    global_rows: list[dict[str, Any]] = []
    for target in [
        "longest_simple_cycle_length_exact_all",
        "max_min_basis_cycle_length",
        "total_min_basis_cycle_length",
        "mean_maxbasis",
        "frac_ge7",
    ]:
        yt = ctr[target].to_numpy(dtype=np.float64)
        yv = cyc[target].to_numpy(dtype=np.float64)
        m = HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=0,
        )
        m.fit(G_train, yt)
        pred = m.predict(G_valid)
        global_rows.append(
            {
                "target": target,
                "fit": "train-10k global33 -> valid-1000",
                "n_train": 10000,
                "n_valid": 1000,
                "r2_valid": float(r2_score(yv, pred)),
                "mae_valid": float(mean_absolute_error(yv, pred)),
                "spearman_valid": float(spearmanr(yv, pred).statistic),
                "target_min": float(yv.min()),
                "target_max": float(yv.max()),
                "pred_min": float(pred.min()),
                "pred_max": float(pred.max()),
            }
        )

    # The global33 ring block is a pooled per-atom minimum-ring histogram, not a
    # per-atom vector.  Record the exact indices so the report is checkable.
    global33_meta = {
        "names": GLOBAL33_NAMES,
        "ring_atom_fraction_index": GLOBAL33_NAMES.index("ring_atom_fraction"),
        "n_ring_systems_index": GLOBAL33_NAMES.index("n_ring_systems"),
        "cyclomatic_rank_index": GLOBAL33_NAMES.index("cyclomatic_rank"),
        "ring_size_histogram_slice": [
            GLOBAL33_NAMES.index("ring_size3_frac"),
            GLOBAL33_NAMES.index("ring_size8plus_frac") + 1,
        ],
        "fused_ring_atom_fraction_index": GLOBAL33_NAMES.index("fused_ring_atom_fraction"),
        "has_per_atom_min_ring_vector": False,
        "has_pooled_per_atom_min_ring_histogram": True,
        "has_exact_longest_simple_cycle": False,
        "has_minimum_cycle_basis_max": False,
        "has_minimum_cycle_basis_total": False,
        "source": "tracks/structure-contribution/code/structural_relation_objects_v1.py::_global_features",
        "note": (
            "ring_size3_frac..ring_size8plus_frac is a histogram of each atom's "
            "minimum ring size, normalised by the number of ring atoms; the "
            "per-atom one-hot lives in atom_feat, not in global33."
        ),
    }

    # Permutation check on a fixed sample.  Exact-longest invariance is already
    # recorded in the KSVD audit files; MCB is recomputed here from the stored
    # adjacency.  cycle_basis is included as the counter-example.
    import networkx as nx

    atom_ptr = np.asarray(va["atom_ptr"], dtype=np.int64)
    bond_ptr = np.asarray(va["bond_ptr"], dtype=np.int64)
    bond_index = np.asarray(va["bond_index"], dtype=np.int64)
    rng = np.random.default_rng(20261027)
    sample = list(range(0, 1000, 20))
    perm_rows: list[dict[str, Any]] = []
    mcb_orig_max: list[int] = []
    mcb_orig_total: list[int] = []
    for r in sample:
        a0, a1 = atom_ptr[r], atom_ptr[r + 1]
        b0, b1 = bond_ptr[r], bond_ptr[r + 1]
        edges = [tuple(map(int, bond_index[k])) for k in range(b0, b1)]
        n_atoms = int(a1 - a0)
        g0 = nx.Graph()
        g0.add_nodes_from(range(n_atoms))
        g0.add_edges_from(edges)
        cb0 = max((len(c) for c in nx.cycle_basis(g0)), default=0)
        mcb0 = nx.minimum_cycle_basis(g0)
        mcb_max0 = max((len(c) for c in mcb0), default=0)
        mcb_total0 = sum(len(c) for c in mcb0)
        mcb_orig_max.append(mcb_max0)
        mcb_orig_total.append(mcb_total0)
        cb_vals = {cb0}
        mcb_max_vals = {mcb_max0}
        mcb_total_vals = {mcb_total0}
        for _ in range(5):
            perm = rng.permutation(n_atoms)
            gp = nx.Graph()
            gp.add_nodes_from(range(n_atoms))
            gp.add_edges_from((int(perm[u]), int(perm[v])) for u, v in edges)
            cb_vals.add(max((len(c) for c in nx.cycle_basis(gp)), default=0))
            mp = nx.minimum_cycle_basis(gp)
            mcb_max_vals.add(max((len(c) for c in mp), default=0))
            mcb_total_vals.add(sum(len(c) for c in mp))
        perm_rows.append(
            {
                "row": r,
                "n_atoms": n_atoms,
                "cycle_basis_max_unique": len(cb_vals),
                "mcb_max_unique": len(mcb_max_vals),
                "mcb_total_unique": len(mcb_total_vals),
                "cycle_basis_values": sorted(cb_vals),
                "mcb_max_values": sorted(mcb_max_vals),
                "mcb_total_values": sorted(mcb_total_vals),
            }
        )
    # Check the stored MCB columns against our recomputation.
    assert np.array_equal(
        np.asarray(mcb_orig_max),
        cyc["max_min_basis_cycle_length"].to_numpy(dtype=int)[sample],
    )
    assert np.array_equal(
        np.asarray(mcb_orig_total),
        cyc["total_min_basis_cycle_length"].to_numpy(dtype=int)[sample],
    )
    perm_summary = {
        "sample_rows": len(sample),
        "permutations_per_row": 5,
        "cycle_basis_max_rows_with_multiple_values": int(
            sum(r["cycle_basis_max_unique"] > 1 for r in perm_rows)
        ),
        "mcb_max_rows_with_multiple_values": int(
            sum(r["mcb_max_unique"] > 1 for r in perm_rows)
        ),
        "mcb_total_rows_with_multiple_values": int(
            sum(r["mcb_total_unique"] > 1 for r in perm_rows)
        ),
        "mcb_max_agrees_with_stored_csv": True,
        "mcb_total_agrees_with_stored_csv": True,
        "exact_longest_permutation_reference": (
            "tracks/ksvd/results/zinc_long_cycle_audit/longest_cycle_permutation_check.csv"
        ),
        "cycle_basis_instability_reference": (
            "tracks/ksvd/results/zinc_long_cycle_audit/table_c_basis_instability.csv"
        ),
        "ksvd_check_counts": {
            "cycle_basis_max_tested": 655,
            "cycle_basis_max_unstable": 34,
            "exact_longest_tested": 5,
            "exact_longest_unstable": 0,
            "mcb_max_tested": 20,
            "mcb_max_unstable": 0,
        },
        "note": (
            "exact longest and minimum-cycle-basis scales are permutation-invariant; "
            "the stored-order nx.cycle_basis maximum is not."
        ),
    }

    # ------------------------------------------------------------------
    # Strict train-only OOF: does ring scale add conditional error information?
    # The only strict OOF in the repo is a compact-v4-hinge model (MAE 0.175),
    # not a strong DCC-L OOF.  It is used only as a diagnostic, not to claim
    # strong-model generalization.
    # ------------------------------------------------------------------
    oof = pd.read_csv(OOF)
    oof_mae = oof["mean_absolute_error"].to_numpy(dtype=np.float64)
    oof_signed = oof["mean_signed_residual"].to_numpy(dtype=np.float64)
    oof_log = np.log(oof_mae + 1e-3)
    folds = oof["outer_fold"].to_numpy(dtype=np.int64)
    rarity = [
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
    structure = [
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
    ring_new = [
        "topo_longest_simple_cycle",
        "topo_mcb_max_length",
        "topo_mcb_mean_length",
        "topo_mcb_total_length",
    ]
    topo = [c for c in oof.columns if c.startswith("topo_")]
    topo_nonring = [
        c
        for c in topo
        if c
        not in ring_new
        + ["topo_longest", "topo_longest_squared", "topo_mcb_count"]
    ]
    variants = {
        "rarity": rarity,
        "structure_global": rarity + structure,
        "plus_topo_nonring": rarity + structure + topo_nonring,
        "plus_new_ring": rarity + structure + topo_nonring + ring_new,
        "topo_all": rarity + structure + topo,
    }

    def cross_fit(cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
        X = oof[cols].to_numpy(dtype=np.float64)
        pred = np.full(len(oof), np.nan)
        for f in range(5):
            tr_mask = folds != f
            te_mask = folds == f
            scaler = StandardScaler().fit(X[tr_mask])
            model = Ridge(alpha=1.0)
            model.fit(scaler.transform(X[tr_mask]), oof_log[tr_mask])
            pred[te_mask] = model.predict(scaler.transform(X[te_mask]))
        pe = np.clip(np.exp(pred) - 1e-3, 0.0, None)
        return pred, pe

    oof_rows: list[dict[str, Any]] = []
    oof_preds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, cols in variants.items():
        pred, pe = cross_fit(cols)
        oof_preds[name] = (pred, pe)
        oof_rows.append(
            {
                "variant": name,
                "n_features": len(cols),
                "r2_log": float(r2_score(oof_log, pred)),
                "spearman_pred_ae": float(spearmanr(pe, oof_mae).statistic),
                "spearman_pred_log": float(spearmanr(pred, oof_log).statistic),
                "mae_top_quintile": float(oof_mae[pe >= np.quantile(pe, 0.8)].mean()),
            }
        )
    p0 = oof_preds["plus_topo_nonring"][0]
    p1 = oof_preds["plus_new_ring"][0]
    fold_deltas = []
    for f in range(5):
        m = folds == f
        fold_deltas.append(
            {
                "fold": f,
                "n": int(m.sum()),
                "delta_r2_log": float(r2_score(oof_log[m], p1[m]) - r2_score(oof_log[m], p0[m])),
                "delta_spearman": float(
                    spearmanr(np.exp(p1[m]) - 1e-3, oof_mae[m]).statistic
                    - spearmanr(np.exp(p0[m]) - 1e-3, oof_mae[m]).statistic
                ),
            }
        )
    ring_corr = []
    for c in ring_new:
        ring_corr.append(
            {
                "feature": c,
                "spearman_ae": float(spearmanr(oof[c], oof_mae).statistic),
                "spearman_signed": float(spearmanr(oof[c], oof_signed).statistic),
                "partial_spearman_ae_size_rarity_cycle": partial_spearman(
                    oof[c].to_numpy(dtype=np.float64),
                    oof_mae,
                    [
                        oof["num_nodes"].to_numpy(dtype=np.float64),
                        oof["rare_le5_ratio"].to_numpy(dtype=np.float64),
                        oof["cycle_rank"].to_numpy(dtype=np.float64),
                    ],
                ),
                "partial_spearman_signed_size_rarity_cycle": partial_spearman(
                    oof[c].to_numpy(dtype=np.float64),
                    oof_signed,
                    [
                        oof["num_nodes"].to_numpy(dtype=np.float64),
                        oof["rare_le5_ratio"].to_numpy(dtype=np.float64),
                        oof["cycle_rank"].to_numpy(dtype=np.float64),
                    ],
                ),
                "per_fold_spearman_ae": [
                    float(
                        spearmanr(
                            oof.loc[folds == f, c],
                            oof_mae[folds == f],
                        ).statistic
                    )
                    for f in range(5)
                ],
            }
        )
    oof_penalty = oof["label_excess"].to_numpy(dtype=np.float64) > 0
    oof_summary = {
        "source": str(OOF),
        "protocol": "compact-v4-hinge 5-fold OOF on official train 10k; two seeds; strict train-only",
        "model_mae": float(oof_mae.mean()),
        "model_note": (
            "Not a strong DCC-L OOF. The repo has no strict strong-model OOF; "
            "this diagnostic cannot license a strong-model generalization claim."
        ),
        "n": int(len(oof)),
        "label_excess_positive_n": int(oof_penalty.sum()),
        "mae_label_excess_positive": float(oof_mae[oof_penalty].mean()),
        "mae_label_excess_zero": float(oof_mae[~oof_penalty].mean()),
        "variants": oof_rows,
        "fold_deltas_plus_new_ring_vs_nonring": fold_deltas,
        "ring_feature_correlations": ring_corr,
    }

    # ------------------------------------------------------------------
    # Persist.
    # ------------------------------------------------------------------
    ind = pd.DataFrame(rows)
    cross = pd.DataFrame(cross_rows)
    oracle = pd.DataFrame(oracle_rows)
    glob = pd.DataFrame(global_rows)
    oof_df = pd.DataFrame(oof_rows)
    fold_df = pd.DataFrame(fold_deltas)
    corr_df = pd.DataFrame(ring_corr)
    perm_df = pd.DataFrame(perm_rows)

    ind.to_csv(OUT / "individual_groups.csv", index=False)
    cross.to_csv(OUT / "cross_tabs.csv", index=False)
    oracle.to_csv(OUT / "oracle_ceiling.csv", index=False)
    glob.to_csv(OUT / "global33_ring_information.csv", index=False)
    oof_df.to_csv(OUT / "oof_variants.csv", index=False)
    fold_df.to_csv(OUT / "oof_fold_deltas.csv", index=False)
    corr_df.to_csv(OUT / "oof_ring_correlations.csv", index=False)
    perm_df.to_csv(OUT / "permutation_sample.csv", index=False)

    payload = {
        "protocol": "zinc-error-mechanism-audit-v1",
        "status": "AUDIT_COMPLETE_CPU_ONLY",
        "official_test_loaded": False,
        "official_valid_loaded": True,
        "valid_is_exposed_development_split": True,
        "no_model_training": "Only CPU diagnostic regressors on structure/error targets; no neural model was trained.",
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
        "model_table": basic_model_table(preds, y),
        "counts": {
            "n_valid": n,
            "n_penalty": int(penalty.sum()),
            "n_penalty_severe": int(severe.sum()),
            "n_r2_unseen": int(unseen.sum()),
            "n_r2_low4_only": int(low4.sum()),
            "n_r2_full": int(full.sum()),
            "n_tail": int(tail.sum()),
            "n_exact_longest_ge7": int(long7.sum()),
            "n_exact_longest_ge10": int(long10.sum()),
            "n_rdkit_large": int(rdkit_large.sum()),
            "n_no_penalty_tail": int((~penalty & tail).sum()),
            "n_no_penalty_unseen": int((~penalty & unseen).sum()),
            "n_penalty_unseen": int((penalty & unseen).sum()),
            "n_penalty_seen": int((penalty & ~unseen).sum()),
        },
        "individual_groups": rows,
        "cross_tabs": cross_rows,
        "oracle_ceiling": oracle_rows,
        "global33": {
            "meta": global33_meta,
            "ring_scale_predictability": global_rows,
        },
        "permutation": {
            "summary": perm_summary,
            "rows": perm_rows,
        },
        "oof": oof_summary,
        "headline": {
            "perfect_penalty_repair_dccl_mean": {
                "base_mae": float(np.abs(preds["DCCL_mean"] - y).mean()),
                "delta_mae": float(np.abs(preds["DCCL_mean"] - y)[penalty].sum() / n),
                "new_mae": float(
                    np.abs(preds["DCCL_mean"] - y).mean()
                    - np.abs(preds["DCCL_mean"] - y)[penalty].sum() / n
                ),
                "ae_share": float(
                    np.abs(preds["DCCL_mean"] - y)[penalty].sum()
                    / np.abs(preds["DCCL_mean"] - y).sum()
                ),
            },
            "no_penalty_unseen_share_of_no_penalty_ae": float(
                np.abs(preds["DCCL_mean"] - y)[~penalty & unseen].sum()
                / np.abs(preds["DCCL_mean"] - y)[~penalty].sum()
            ),
            "no_penalty_unseen_share_of_total_ae": float(
                np.abs(preds["DCCL_mean"] - y)[~penalty & unseen].sum()
                / np.abs(preds["DCCL_mean"] - y).sum()
            ),
        },
    }
    (OUT / "audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Small human-readable digest.
    lines = [
        "# ZINC error-mechanism audit — machine digest",
        "",
        "Official test loaded: **false**.  CPU only; no neural model trained.",
        "",
        "Follow-up decomposition: `code/zema_decomposition.py` -> "
        "`analysis/decomposition.json`, `analysis/mece_penalty_unseen_long7.csv`, "
        "`analysis/cycle_attribution.csv`, `analysis/oof_conditional.csv`. "
        "Narrative report: `REPORT.md`.",
        "",
        "## Model MAE",
        "",
        "| model | MAE | median AE | P90 AE |",
        "|---|---:|---:|---:|",
    ]
    for name in [
        "DCCL_s0",
        "DCCL_mean",
        "BOSI_B_s0",
        "ZBO_R_s0",
        "ZBO_N_s0",
        "ZBO_S_s0",
    ]:
        m = basic_model_table(preds, y)[name]
        lines.append(
            f"| {name} | {m['mae']:.6f} | {m['median_ae']:.6f} | {m['p90_ae']:.6f} |"
        )
    lines += [
        "",
        "`DCCL_per_seed_mean` (historical headline) = "
        f"**{basic_model_table(preds, y)['DCCL_per_seed_mean']['mae']:.6f}**; "
        "`DCCL_mean` is the MAE of the 4-seed prediction average (diagnostic ensemble), not a single model.",
        "",
        "## Oracle ceiling (DCC-L 4-seed mean)",
        "",
        "| fixed group | n | AE share | ΔMAE if zero AE | new MAE |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in oracle_rows:
        if r["model"] == "DCCL_mean":
            lines.append(
                f"| {r['group']} | {r['n']} | {r['ae_share']:.3f} | "
                f"{r['delta_mae']:.6f} | {r['new_mae_if_zero_ae']:.6f} |"
            )
    lines += [
        "",
        "## Cross-tab (DCC-L 4-seed mean)",
        "",
        "| table | cell | n | MAE | AE share | signed |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in cross_rows:
        if r["model"] == "DCCL_mean" and r["table"] in {
            "penalty_x_unseen",
            "penalty_x_support",
            "penalty_x_tail",
        }:
            lines.append(
                f"| {r['table']} | {r['group']} | {r['n']} | {r['mae']:.4f} | "
                f"{r['ae_share']:.3f} | {r['signed']:+.4f} |"
            )
    lines += [
        "",
        "## global33 ring-scale predictability (train -> valid)",
        "",
        "| target | valid R² | valid MAE | valid Spearman |",
        "|---|---:|---:|---:|",
    ]
    for r in global_rows:
        lines.append(
            f"| {r['target']} | {r['r2_valid']:.3f} | {r['mae_valid']:.3f} | "
            f"{r['spearman_valid']:.3f} |"
        )
    lines += [
        "",
        "High valid R2 means these scales are largely reconstructable from the 33-D",
        "global descriptor (MCB total is essentially a restatement of cyclomatic rank",
        "and the ring-size histogram).  They are not new per-atom information; see",
        "`REPORT.md` section 4.",
        "",
        "## OOF ring-feature diagnostic (compact-v4-hinge, not DCC-L)",
        "",
        "| variant | features | R²(log AE) | Spearman(pred, AE) | top-quintile MAE |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in oof_rows:
        lines.append(
            f"| {r['variant']} | {r['n_features']} | {r['r2_log']:.4f} | "
            f"{r['spearman_pred_ae']:.4f} | {r['mae_top_quintile']:.4f} |"
        )
    lines += [
        "",
        "**Caveat:** the all-row OOF R2 gain is entirely penalty-row-driven.  On the",
        "9,628 no-penalty rows the same ring scales add `delta R2 = +0.00006` and",
        "`delta Spearman = -0.0008`; see `analysis/oof_conditional.csv` and",
        "`REPORT.md` section 5.  Do not quote the all-row gain as general ring",
        "information.",
        "",
        "Full JSON: `analysis/audit.json`; tables: `analysis/*.csv`.",
        "",
    ]
    (OUT / "DIGEST.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
