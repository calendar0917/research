"""Same-molecule R2 vs EGO2 gap, paired scaffold bootstrap and D0 seed check."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr

import ib_common as C


def _molecule_meta(bundle: Any, rows: np.ndarray) -> dict[int, dict[str, Any]]:
    import gwic_data as GD

    out: dict[int, dict[str, Any]] = {}
    for row in rows.tolist():
        view = GD.molecule_view(bundle.raw, int(row))
        out[int(row)] = {
            "smiles": str(view["smiles"]),
            "n_atoms": int(view["n_atoms"]),
            "n_bonds": int(view["n_bonds"]),
        }
    return out


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()
                if k not in ("delta", "draws")}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(float(value)) else float(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _tails(rows: np.ndarray, y: np.ndarray, pa: np.ndarray, pb: np.ndarray,
           meta: dict[int, dict[str, Any]], top_n: int = 10) -> list[dict[str, Any]]:
    delta = np.abs(pa - y) - np.abs(pb - y)
    order = np.argsort(-delta, kind="mergesort")
    selected: list[int] = list(order[:top_n]) + list(order[-top_n:][::-1])
    out: list[dict[str, Any]] = []
    seen: set[int] = set()
    for i in selected:
        if int(i) in seen:
            continue
        seen.add(int(i))
        row = int(rows[i])
        out.append({
            "row_id": row,
            "smiles": meta[row]["smiles"],
            "n_atoms": meta[row]["n_atoms"],
            "label": float(y[i]),
            "pred_r2": float(pa[i]),
            "pred_ego2": float(pb[i]),
            "abs_error_r2": float(abs(pa[i] - y[i])),
            "abs_error_ego2": float(abs(pb[i] - y[i])),
            "delta_abs_error": float(delta[i]),
            "tail": "ego2_better" if delta[i] > 0 else ("r2_better" if delta[i] < 0 else "tie"),
        })
    return out


def _loo(delta: np.ndarray) -> dict[str, Any]:
    n = int(delta.size)
    order = np.argsort(-np.abs(delta), kind="mergesort")
    max_abs = int(order[0])
    max_pos = int(np.argmax(delta))
    max_neg = int(np.argmin(delta))
    gain = float(delta.mean())
    return {
        "gain_mae": gain,
        "max_abs_index": max_abs,
        "max_abs_delta": float(delta[max_abs]),
        "gain_after_max_abs": float(np.delete(delta, max_abs).mean()) if n > 1 else None,
        "max_positive_index": max_pos,
        "max_positive_delta": float(delta[max_pos]),
        "gain_after_max_positive": float(np.delete(delta, max_pos).mean()) if n > 1 else None,
        "max_negative_index": max_neg,
        "max_negative_delta": float(delta[max_neg]),
        "gain_after_max_negative": float(np.delete(delta, max_neg).mean()) if n > 1 else None,
    }


def main() -> None:
    C.ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    C.assert_inputs()
    bundles = {split: C.load_bundle(split) for split in C.SPLITS}
    scaffolds = {split: C.scaffold_map(split) for split in C.SPLITS}
    identities = {split: C.split_identity(bundles[split]) for split in C.SPLITS}

    summary: dict[str, Any] = {
        "schema": "r2-internal-binding-gap-audit-gap-v1",
        "created_utc": C.now_utc(),
        "bootstrap": {"n_boot": C.N_BOOT, "seed": C.BOOT_SEED,
                      "cluster": "Bemis-Murcko scaffold, whole-cluster resampling"},
        "identities": {split: identities[split] for split in C.SPLITS},
        "overlap": {
            "d0_dev_intersection_d1_dev": int(np.intersect1d(
                bundles["D0"].eval_rows, bundles["D1"].eval_rows).size),
            "note": "D0 and D1 are never pooled; overlap is recorded for transparency.",
        },
        "cells": {},
    }

    per_molecule: list[dict[str, Any]] = []
    tails: list[dict[str, Any]] = []
    bootstrap_draws: dict[str, np.ndarray] = {}
    metrics_rows: list[dict[str, Any]] = []

    for split, seed in C.CELLS:
        bundle = bundles[split]
        rows = np.asarray(bundle.eval_rows, np.int64)
        pred_r2 = C.load_predictions(C.RUNS[(split, seed)]["R2"], bundle)
        pred_ego2 = C.load_predictions(C.RUNS[(split, seed)]["EGO2"], bundle)
        if not np.array_equal(pred_r2["row_id"], pred_ego2["row_id"]):
            raise ValueError(f"{split}/S{seed}: R2 and EGO2 row order differs")
        if not np.allclose(pred_r2["label"], pred_ego2["label"], rtol=0.0, atol=1e-9):
            raise ValueError(f"{split}/S{seed}: R2 and EGO2 labels differ")
        y = pred_r2["y"]
        pa = pred_r2["pred"]
        pb = pred_ego2["pred"]
        clusters = C.clusters_for(rows, scaffolds[split])
        meta = _molecule_meta(bundle, rows)

        point = C.point_metrics(y, pa, pb, clusters=clusters)
        delta = point["delta"]
        boot = C.bootstrap_paired(y, pa, pb, clusters)
        draws = boot.pop("draws")
        for key, values in draws.items():
            bootstrap_draws[f"{split}_S{seed}_{key}"] = values
        loo = _loo(delta)
        tail_rows = _tails(rows, y, pa, pb, meta)
        tails.extend({"split": split, "seed": seed, **row} for row in tail_rows)
        cell_summary = {
            "split": split,
            "seed": int(seed),
            "run_r2": C.RUNS[(split, seed)]["R2"],
            "run_ego2": C.RUNS[(split, seed)]["EGO2"],
            "n_molecules": int(rows.size),
            "n_clusters": int(np.unique(clusters).size),
            "rmse_r2": point["rmse_r2"],
            "rmse_ego2": point["rmse_ego2"],
            "mae_r2": point["mae_r2"],
            "mae_ego2": point["mae_ego2"],
            "gain_rmse": point["gain_rmse"],
            "gain_mae": point["gain_mae"],
            "prop_ego2_better": point["prop_ego2_better"],
            "prop_r2_better": point["prop_r2_better"],
            "prop_tie": point["prop_tie"],
            "max_delta": point["max_delta"],
            "min_delta": point["min_delta"],
            "bootstrap": boot,
            "leave_one_out": loo,
            "dominance": C.dominance(delta),
            "top_tails": tail_rows,
        }
        summary["cells"][f"{split}_S{seed}"] = cell_summary
        metrics_rows.append({
            "split": split,
            "seed": int(seed),
            "n_molecules": int(rows.size),
            "n_clusters": int(np.unique(clusters).size),
            "rmse_r2": point["rmse_r2"],
            "rmse_ego2": point["rmse_ego2"],
            "rmse_gain_r2_minus_ego2": point["gain_rmse"],
            "rmse_gain_ci_lo": boot["gain_rmse_ci"]["lo"],
            "rmse_gain_ci_hi": boot["gain_rmse_ci"]["hi"],
            "mae_r2": point["mae_r2"],
            "mae_ego2": point["mae_ego2"],
            "mae_gain_r2_minus_ego2": point["gain_mae"],
            "mae_gain_ci_lo": boot["gain_mae_ci"]["lo"],
            "mae_gain_ci_hi": boot["gain_mae_ci"]["hi"],
            "prop_ego2_better": point["prop_ego2_better"],
            "prop_ego2_better_ci_lo": boot["prop_ego2_better_ci"]["lo"],
            "prop_ego2_better_ci_hi": boot["prop_ego2_better_ci"]["hi"],
            "gain_mae_after_max_abs": loo["gain_after_max_abs"],
        })
        for i, row in enumerate(rows.tolist()):
            per_molecule.append({
                "split": split,
                "seed": int(seed),
                "row_id": int(row),
                "smiles": meta[int(row)]["smiles"],
                "n_atoms": meta[int(row)]["n_atoms"],
                "label": float(y[i]),
                "pred_r2": float(pa[i]),
                "pred_ego2": float(pb[i]),
                "abs_error_r2": float(abs(pa[i] - y[i])),
                "abs_error_ego2": float(abs(pb[i] - y[i])),
                "delta_abs_error": float(delta[i]),
                "cluster": str(clusters[i]),
            })

    # D0 seed consistency.
    b0 = bundles["D0"]
    rows0 = np.asarray(b0.eval_rows, np.int64)
    r2_s0 = C.load_predictions(C.RUNS[("D0", 0)]["R2"], b0)
    e2_s0 = C.load_predictions(C.RUNS[("D0", 0)]["EGO2"], b0)
    r2_s1 = C.load_predictions(C.RUNS[("D0", 1)]["R2"], b0)
    e2_s1 = C.load_predictions(C.RUNS[("D0", 1)]["EGO2"], b0)
    for other in (e2_s0, r2_s1, e2_s1):
        if not np.array_equal(other["row_id"], rows0):
            raise ValueError("D0 seed comparison: row order differs")
    y0 = r2_s0["y"]
    d0 = np.abs(r2_s0["pred"] - y0) - np.abs(e2_s0["pred"] - y0)
    d1 = np.abs(r2_s1["pred"] - y0) - np.abs(e2_s1["pred"] - y0)
    sign0 = np.sign(d0).astype(int)
    sign1 = np.sign(d1).astype(int)
    both_pos = int(np.sum((sign0 > 0) & (sign1 > 0)))
    both_neg = int(np.sum((sign0 < 0) & (sign1 < 0)))
    mixed = int(np.sum((sign0 > 0) & (sign1 < 0)) + np.sum((sign0 < 0) & (sign1 > 0)))
    ties = int(np.sum((sign0 == 0) | (sign1 == 0)))
    non_tie = int(np.sum((sign0 != 0) & (sign1 != 0)))
    same_abs_01 = int(np.sum((np.abs(d0) >= 0.01) & (np.abs(d1) >= 0.01)
                             & (sign0 == sign1)))
    total_abs_01 = int(np.sum((np.abs(d0) >= 0.01) & (np.abs(d1) >= 0.01)))
    same_abs_05 = int(np.sum((np.abs(d0) >= 0.05) & (np.abs(d1) >= 0.05)
                             & (sign0 == sign1)))
    total_abs_05 = int(np.sum((np.abs(d0) >= 0.05) & (np.abs(d1) >= 0.05)))
    meta0 = _molecule_meta(b0, rows0)
    consistency_rows: list[dict[str, Any]] = []
    for i, row in enumerate(rows0.tolist()):
        consistency_rows.append({
            "row_id": int(row),
            "smiles": meta0[int(row)]["smiles"],
            "n_atoms": meta0[int(row)]["n_atoms"],
            "label": float(y0[i]),
            "delta_s0": float(d0[i]),
            "delta_s1": float(d1[i]),
            "sign_s0": int(sign0[i]),
            "sign_s1": int(sign1[i]),
            "same_sign": bool(sign0[i] == sign1[i] and sign0[i] != 0),
            "both_abs_ge_0_01": bool(abs(d0[i]) >= 0.01 and abs(d1[i]) >= 0.01),
            "both_abs_ge_0_05": bool(abs(d0[i]) >= 0.05 and abs(d1[i]) >= 0.05),
        })
    order_consistent = np.argsort(-np.minimum(d0, d1), kind="mergesort")[:10]
    top_consistent = [
        {
            "row_id": int(rows0[i]),
            "smiles": meta0[int(rows0[i])]["smiles"],
            "delta_s0": float(d0[i]),
            "delta_s1": float(d1[i]),
            "min_delta": float(min(d0[i], d1[i])),
        }
        for i in order_consistent
    ]
    summary["d0_seed_consistency"] = {
        "n_molecules": int(rows0.size),
        "both_positive": both_pos,
        "both_negative": both_neg,
        "mixed_sign": mixed,
        "ties": ties,
        "non_tie": non_tie,
        "sign_agreement_non_tie": (None if non_tie == 0
                                   else float((both_pos + both_neg) / non_tie)),
        "pearson_delta": float(pearsonr(d0, d1).statistic),
        "spearman_delta": float(spearmanr(d0, d1).statistic),
        "both_abs_ge_0_01": total_abs_01,
        "same_sign_abs_ge_0_01": same_abs_01,
        "sign_agreement_abs_ge_0_01": (None if total_abs_01 == 0
                                       else float(same_abs_01 / total_abs_01)),
        "both_abs_ge_0_05": total_abs_05,
        "same_sign_abs_ge_0_05": same_abs_05,
        "sign_agreement_abs_ge_0_05": (None if total_abs_05 == 0
                                       else float(same_abs_05 / total_abs_05)),
        "top_consistent_positive": top_consistent,
    }

    C.write_csv(C.ANALYSIS_DIR / "gap_metrics.csv", metrics_rows)
    C.write_csv(C.ANALYSIS_DIR / "gap_per_molecule.csv", per_molecule)
    C.write_csv(C.ANALYSIS_DIR / "gap_tails.csv", tails)
    C.write_csv(C.ANALYSIS_DIR / "gap_d0_seed_consistency.csv", consistency_rows)
    C.write_json(C.ANALYSIS_DIR / "gap_summary.json", _json_safe(summary))
    np.savez_compressed(C.ANALYSIS_DIR / "bootstrap_gap_draws.npz", **bootstrap_draws)

    print("same-molecule gap")
    for split, seed in C.CELLS:
        cell = summary["cells"][f"{split}_S{seed}"]
        print(f"  {split}/S{seed}: RMSE {cell['rmse_r2']:.6f} -> "
              f"{cell['rmse_ego2']:.6f} (gain {cell['gain_rmse']:+.6f}, "
              f"CI [{cell['bootstrap']['gain_rmse_ci']['lo']:+.6f}, "
              f"{cell['bootstrap']['gain_rmse_ci']['hi']:+.6f}]); "
              f"MAE gain {cell['gain_mae']:+.6f}; "
              f"EGO2 better {cell['prop_ego2_better']:.3f}")
    cons = summary["d0_seed_consistency"]
    print(f"  D0 seed sign agreement: {cons['sign_agreement_non_tie']:.3f} "
          f"({cons['both_positive']}+ / {cons['both_negative']}- / "
          f"{cons['mixed_sign']} mixed / {cons['ties']} ties)")
    print(f"  D0 seed Pearson delta: {cons['pearson_delta']:.3f}; "
          f"Spearman: {cons['spearman_delta']:.3f}")
    print(f"  D0/D1 Dev overlap: {summary['overlap']['d0_dev_intersection_d1_dev']}")


if __name__ == "__main__":
    main()