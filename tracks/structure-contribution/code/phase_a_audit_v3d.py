"""cscl-v3d Phase A: CPU-only gap / UNK / frozen-gamma / info-loss diagnostics.

No training.  Reads committed per-molecule predictions of F / O / S_v3 /
S_gauge / S_typecond and the trained v3c checkpoint (frozen), and produces:

1. same-molecule paired error diagnostics, especially molecules WITH vs WITHOUT
   unknown RINGCHAIN types (both groups evaluated for ALL arms -- the v3c
   finding "UNK molecules 0.206 vs 0.111" must NOT be read as a structured-only
   bottleneck without the F/O comparison);
2. confounder checks (UNK vs n_atoms / y left tail / cyclomatic / type
   frequency) on the identical dev rows;
3. frozen-model gamma diagnostics for the v3c checkpoint: per-molecule relation
   contribution sums, |gamma| vs relation size, bonded vs non-bonded unit pairs,
   dominant-bucket grouping, and the ZERO-GAMMA counterfactual prediction
   change (a computation-dependence diagnostic ONLY -- not a retrained ablation,
   no chemical-causal reading);
4. a quantitative inventory of the distance-bucket information the v3 S/O
   inputs lost (bucket dispersion of pair moments + per-molecule pair bucket
   distributions computed label-free from topology).

Outputs results/cscl_v3d/phase_a_s0.json.  Official valid/test never loaded.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

_CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = _CODE_DIR.parents[2]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import cscl_features as cf  # noqa: E402
import strong_backbone_attribution as sba  # noqa: E402
from run_cscl_v3 import (  # noqa: E402
    _index_by_row,
    collate,
    paired_bootstrap,
    prepare_all,
    write_json,
)
from run_cscl_v3b import subgroup_table_v3b  # noqa: E402
from run_cscl_v3c import TypeCondContribHead  # noqa: E402

PROTOCOL_ID = "cscl-v3d-distance-conditioned-contribution"
OUT_DIR = REPO_ROOT / "tracks/structure-contribution/results/cscl_v3d"
V3C_DIR = REPO_ROOT / "tracks/structure-contribution/results/cscl_v3c"

RECORDED_MAE = {"F": 0.11641, "O": 0.14129, "S_v3": 0.19551, "S_gauge": 0.17317, "S_typecond": 0.15209}


def load_arms(rows: np.ndarray) -> dict[str, np.ndarray]:
    arms: dict[str, np.ndarray] = {}
    sources = {
        "F": sba.DEV_PREDS_NPZ,
        "O": REPO_ROOT / "tracks/structure-contribution/results/cscl_v3/O_s0_dev_preds.npz",
        "S_v3": REPO_ROOT / "tracks/structure-contribution/results/cscl_v3/S_s0_dev_preds.npz",
        "S_gauge": REPO_ROOT / "tracks/structure-contribution/results/cscl_v3b/S_gauge_s0_dev_preds.npz",
        "S_typecond": V3C_DIR / "S_typecond_s0_dev_preds.npz",
    }
    for name, path in sources.items():
        z = np.load(path)
        assert np.array_equal(z["row"], rows), f"{name} row order mismatch"
        arms[name] = z["pred"].astype(np.float64)
    return arms


def group_mae_table(y: np.ndarray, arms: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {"n": int(mask.sum())}
    for a, p in arms.items():
        out[f"mae_{a}"] = float(np.abs(p[mask] - y[mask]).mean())
    return out


def paired_within_group(y: np.ndarray, arms: dict[str, np.ndarray], mask: np.ndarray, a: str, b: str) -> dict:
    return paired_bootstrap(np.abs(arms[a][mask] - y[mask]), np.abs(arms[b][mask] - y[mask]))


def molecule_bucket_distribution(rows: np.ndarray, dataset) -> dict[str, np.ndarray]:
    """Label-free per-molecule pair bucket counts from topological distances
    (Full-Y rule: bucket = min(max(d,1),5)-1).  Verified against the real
    ``pair_bucket`` of the env collate on sampled molecules."""
    import networkx as nx

    n = len(rows)
    counts = np.zeros((n, 5), dtype=np.int64)
    for i, row in enumerate(rows):
        data = dataset[int(row)]
        ei = data.edge_index.numpy()
        g = nx.Graph()
        g.add_nodes_from(range(int(data.x.shape[0])))
        for k in range(ei.shape[1]):
            u, v = int(ei[0, k]), int(ei[1, k])
            if u != v:
                g.add_edge(u, v)
        spl = dict(nx.all_pairs_shortest_path_length(g))
        for a in range(int(data.x.shape[0])):
            for b, d in spl[a].items():
                if b > a:
                    counts[i, min(max(int(d), 1), 5) - 1] += 1
    return {"counts": counts}


def main() -> int:
    parser = argparse.ArgumentParser(description=PROTOCOL_ID + " phase A")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(8)
    device = torch.device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    log = lambda *a, **k: print(*a, **k, flush=True)  # noqa: E731

    ns = prepare_all(log=log)
    dev_pos = _index_by_row(ns.features, ns.idx["dev"])
    rows = np.asarray(ns.idx["dev"], dtype=np.int64)
    y_dev = ns.y[rows]
    arms = load_arms(rows)

    mae = {a: float(np.abs(p - y_dev).mean()) for a, p in arms.items()}
    check = {a: abs(mae[a] - RECORDED_MAE[a]) <= 1e-4 for a in arms}
    log("[phaseA] recomputed MAE " + json.dumps({k: round(v, 5) for k, v in mae.items()}) + f" matches_recorded={check}")
    assert all(check.values()), "committed prediction files do not reproduce the recorded reference MAEs"

    out: dict[str, Any] = {
        "protocol": PROTOCOL_ID,
        "stage": "phase_a_cpu_diagnostics",
        "n_dev": len(rows),
        "recomputed_mae": mae,
        "recorded_mae_check": check,
        "paired_all": {
            f"{a}_minus_{b}": paired_bootstrap(np.abs(arms[a] - y_dev), np.abs(arms[b] - y_dev))
            for a, b in (("S_typecond", "O"), ("S_typecond", "F"), ("O", "F"), ("S_typecond", "S_gauge"))
        },
    }

    # ------------------------------------------------------------------
    # 1. UNK grouping on identical molecules, all arms
    # ------------------------------------------------------------------
    known_unit = np.concatenate([
        np.asarray([ns.vocab.is_known(s) for s in ns.features[fi].unit_sigs])
        for fi in dev_pos
    ])
    unit_mol = np.concatenate([np.full(ns.features[fi].n_units, i) for i, fi in enumerate(dev_pos)])
    has_unk = np.zeros(len(rows), dtype=bool)
    np.add.at(has_unk, unit_mol[~known_unit], 1)
    n_unk = np.zeros(len(rows), dtype=np.int64)
    np.add.at(n_unk, unit_mol[~known_unit], 1)
    unk_frac_units = np.zeros(len(rows))
    n_units_mol = np.asarray([ns.features[fi].n_units for fi in dev_pos], dtype=np.float64)
    unk_frac_units[has_unk] = n_unk[has_unk] / n_units_mol[has_unk]

    out["unk_groups"] = {
        "n_with_unk": int(has_unk.sum()),
        "n_without_unk": int((~has_unk).sum()),
        "mae_with_unk": group_mae_table(y_dev, arms, has_unk),
        "mae_without_unk": group_mae_table(y_dev, arms, ~has_unk),
        "paired_within_with_unk": {
            f"{a}_minus_{b}": paired_within_group(y_dev, arms, has_unk, a, b)
            for a, b in (("S_typecond", "O"), ("S_typecond", "F"), ("O", "F"))
        },
        "paired_within_without_unk": {
            f"{a}_minus_{b}": paired_within_group(y_dev, arms, ~has_unk, a, b)
            for a, b in (("S_typecond", "O"), ("S_typecond", "F"), ("O", "F"))
        },
        "by_n_unk": {
            f"n_unk_{k}": group_mae_table(y_dev, arms, n_unk == k)
            for k in sorted(set(n_unk.tolist()))
        },
        "by_unk_fraction": {
            "f_eq_0": group_mae_table(y_dev, arms, unk_frac_units == 0),
            "f_le_half": group_mae_table(y_dev, arms, (unk_frac_units > 0) & (unk_frac_units <= 0.5)),
            "f_gt_half": group_mae_table(y_dev, arms, unk_frac_units > 0.5),
        },
    }

    # ------------------------------------------------------------------
    # 2. confounder checks on identical rows
    # ------------------------------------------------------------------
    dataset = cf.load_official_train()
    import networkx as nx

    n_atoms = np.zeros(len(rows))
    cyc = np.zeros(len(rows))
    for i, row in enumerate(rows):
        data = dataset[int(row)]
        n_atoms[i] = data.x.shape[0]
        ei = data.edge_index.numpy()
        g = nx.Graph()
        g.add_nodes_from(range(int(data.x.shape[0])))
        for k in range(ei.shape[1]):
            u, v = int(ei[0, k]), int(ei[1, k])
            if u != v:
                g.add_edge(u, v)
        cyc[i] = max(ei.shape[1] // 2 - int(data.x.shape[0]) + nx.number_connected_components(g), 0)
    ytail = y_dev <= -3

    def _corr(a, b):
        return float(np.corrcoef(a.astype(np.float64), b.astype(np.float64))[0, 1])

    out["confounders"] = {
        "corr_has_unk_n_atoms": _corr(has_unk, n_atoms),
        "corr_has_unk_cyc": _corr(has_unk, cyc),
        "corr_has_unk_ytail": _corr(has_unk.astype(float), ytail.astype(float)),
        "corr_has_unk_abs_y": _corr(has_unk, np.abs(y_dev)),
        "mean_n_atoms_with_vs_without": [float(n_atoms[has_unk].mean()), float(n_atoms[~has_unk].mean())],
        "frac_ytail_with_vs_without": [float(ytail[has_unk].mean()), float(ytail[~has_unk].mean())],
        "mean_cyc_with_vs_without": [float(cyc[has_unk].mean()), float(cyc[~has_unk].mean())],
        "unk_x_n_atoms": {
            "atoms_lt_22": group_mae_table(y_dev, arms, has_unk & (n_atoms < 22)),
            "atoms_ge_22": group_mae_table(y_dev, arms, has_unk & (n_atoms >= 22)),
        },
        "unk_x_ytail": {
            "ytail": group_mae_table(y_dev, arms, has_unk & ytail),
            "no_ytail": group_mae_table(y_dev, arms, has_unk & ~ytail),
        },
        "no_unk_x_ytail": {
            "ytail": group_mae_table(y_dev, arms, ~has_unk & ytail),
            "no_ytail": group_mae_table(y_dev, arms, ~has_unk & ~ytail),
        },
    }

    # ------------------------------------------------------------------
    # 3. frozen-model gamma diagnostics (v3c checkpoint, no training)
    # ------------------------------------------------------------------
    model = TypeCondContribHead(ns.vocab.n_total, ns.tau_ctx, ns.tau_topo)
    model.load_state_dict(torch.load(V3C_DIR / "S_typecond_s0_best_state.pt", map_location="cpu", weights_only=True))
    model = model.to(device).eval()
    dev_batch = collate(ns.features, dev_pos, ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean).to(device)
    with torch.no_grad():
        o = model(dev_batch)
    pred = o["pred"].cpu().numpy().astype(np.float64)
    gamma = o["gamma"].cpu().numpy().astype(np.float64)
    rel_mol = dev_batch.rel_mol.cpu().numpy()
    # consistency: collate-based predictions equal the committed file
    gmax = float(np.abs(pred - arms["S_typecond"]).max())
    assert gmax <= 1e-4, f"v3c replay gap {gmax}"

    rel_bond_hist = np.concatenate([
        ns.features[fi].rel_bond_hist for fi in dev_pos if len(ns.features[fi].rel_lo)
    ], axis=0) if any(len(ns.features[fi].rel_lo) for fi in dev_pos) else np.zeros((0, 4))
    rel_count = np.concatenate([ns.features[fi].rel_count for fi in dev_pos if len(ns.features[fi].rel_lo)])
    rel_bucket_counts = np.concatenate([
        ns.features[fi].rel_bucket_counts for fi in dev_pos if len(ns.features[fi].rel_lo)
    ], axis=0)
    assert len(gamma) == len(rel_count)
    bonded = rel_bond_hist.sum(1) > 0
    dom_bucket = rel_bucket_counts.argmax(1)
    rel_sum_mol = np.zeros(len(rows))
    np.add.at(rel_sum_mol, rel_mol, gamma)

    absgamma = np.abs(gamma)
    out["gamma_frozen_v3c"] = {
        "note": (
            "frozen-model computation-dependence diagnostics.  The zero-gamma "
            "counterfactual measures what the CURRENT trained model stops "
            "varying when its gamma channel is cut; it is NOT the performance "
            "of a retrained no-gamma model and carries no chemical-causal reading."
        ),
        "collate_vs_committed_max_abs_pred_diff": gmax,
        "n_relations_dev": int(len(gamma)),
        "gamma_mean": float(gamma.mean()), "gamma_std": float(gamma.std()),
        "frac_abs_le_0.01": float((absgamma <= 0.01).mean()),
        "per_molecule_rel_sum_abs_mean": float(np.abs(rel_sum_mol).mean()),
        "corr_abs_gamma_rel_count": _corr(absgamma, rel_count),
        "spearman_abs_gamma_rel_count": _spearman(absgamma, rel_count),
        "bonded_vs_not": {
            "n_bonded": int(bonded.sum()), "n_not_bonded": int((~bonded).sum()),
            "abs_gamma_mean_bonded": float(absgamma[bonded].mean()) if bonded.any() else None,
            "abs_gamma_mean_not_bonded": float(absgamma[~bonded].mean()) if (~bonded).any() else None,
            "frac_abs_le_0.01_bonded": float((absgamma[bonded] <= 0.01).mean()) if bonded.any() else None,
            "frac_abs_le_0.01_not_bonded": float((absgamma[~bonded] <= 0.01).mean()) if (~bonded).any() else None,
        },
        "by_dominant_bucket": {
            f"bucket_{b}": {
                "n": int((dom_bucket == b).sum()),
                "abs_gamma_mean": float(absgamma[dom_bucket == b].mean()) if (dom_bucket == b).any() else None,
                "frac_abs_le_0.01": float((absgamma[dom_bucket == b] <= 0.01).mean()) if (dom_bucket == b).any() else None,
            }
            for b in range(5)
        },
        "zero_gamma_counterfactual": _zero_gamma(pred, o, rel_mol, y_dev),
    }

    # ------------------------------------------------------------------
    # 4. quantitative info-loss inventory (label-free)
    # ------------------------------------------------------------------
    bdist = molecule_bucket_distribution(rows, dataset)
    # verify the topology-derived buckets against the real collate on samples
    from run_prediction_feasibility_v2 import load_train_rows

    train_rows = load_train_rows(log=log)
    from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p1 as p1

    mism = 0
    for fi in dev_pos[:24]:
        batch = p1.env_collate([train_rows[ns.features[fi].row]])
        real = np.bincount(batch.pair_bucket.numpy(), minlength=5)
        got = bdist["counts"][list(dev_pos).index(fi)]
        mism += int(not np.array_equal(real, got))
    assert mism == 0, f"bucket distribution mismatch on {mism}/24 sampled molecules"
    out["bucket_rule_verification"] = {
        "rule": "min(max(topological distance,1),5)-1",
        "molecules_checked_vs_real_pair_bucket": 24,
        "mismatches": mism,
    }
    counts = bdist["counts"]
    total_pairs = counts.sum(1)
    frac = counts / np.maximum(total_pairs[:, None], 1)
    p = np.clip(frac, 1e-12, 1)
    ent = -(p * np.log(p)).sum(1) / np.log(5)
    # dispersion of per-bucket ERROR contribution: use F residual correlation
    f_resid = np.abs(arms["F"] - y_dev)
    o_resid = np.abs(arms["O"] - y_dev)
    out["bucket_inventory"] = {
        "per_bucket_pair_share_mean": [float(frac[:, b].mean()) for b in range(5)],
        "molecules_with_pairs_in_all_5_buckets": int(((counts > 0).all(1)).sum()),
        "mean_normalized_bucket_entropy": float(ent.mean()),
        "corr_frac_bonded_pairs_F_residual": _corr(frac[:, 0], f_resid),
        "corr_frac_bonded_pairs_O_residual": _corr(frac[:, 0], o_resid),
        "corr_bucket_entropy_F_residual": _corr(ent, f_resid),
        "corr_bucket_entropy_O_residual": _corr(ent, o_resid),
        "note": (
            "v3 S/O inputs keep only bucket-MARGINAL pair moments (unit_sum_pv / "
            "rel_sum_pv) plus per-relation bucket COUNTS; the per-bucket first/"
            "second moments of BOTH intra- and cross-unit pairs were computed by "
            "the extractor but not stored (results/cscl_v3b/phase_a_boundaries.json); "
            "v3d stores them (unit_bucket_* / rel_bucket_sum_*)"
        ),
    }

    out["subgroups_v3c_reference"] = subgroup_table_v3b(rows, y_dev, arms, ns, dataset)
    out["official_valid_loaded"] = False
    out["official_test_loaded"] = False
    write_json(args.out / "phase_a_s0.json", out)
    log("[phaseA] wrote " + str(args.out / "phase_a_s0.json"))
    return 0


def _spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def _zero_gamma(pred, out_dict, rel_mol, y_dev):
    """Zero-gamma counterfactual: same frozen model, gamma output forced to 0."""
    rel_sum = np.zeros(len(pred))
    gamma = out_dict["gamma"].cpu().numpy().astype(np.float64)
    np.add.at(rel_sum, rel_mol, gamma)
    pred0 = pred - rel_sum
    err = np.abs(pred - y_dev)
    err0 = np.abs(pred0 - y_dev)
    return {
        "max_abs_pred_change": float(np.abs(rel_sum).max()),
        "mean_abs_pred_change": float(np.abs(rel_sum).mean()),
        "mae_current": float(err.mean()),
        "mae_gamma_zeroed": float(err0.mean()),
        "mae_change": float(err0.mean() - err.mean()),
        "frac_molecules_abs_pred_change_le_0.01": float((np.abs(rel_sum) <= 0.01).mean()),
        "corr_abs_pred_change_abs_err": float(np.corrcoef(np.abs(rel_sum), err)[0, 1]),
    }


if __name__ == "__main__":
    raise SystemExit(main())
