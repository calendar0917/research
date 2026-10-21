"""CPU audit A: are errors shared, are models complementary, are seeds noisy?

All statistics use the same fixed official-valid 1,000 rows.  Tail numbers
(top-k shares) are descriptive decompositions of the full-split MAE, never a
replacement score.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import audit_common as C  # noqa: E402

ANALYSIS = C.ANALYSIS


def metrics(pred: np.ndarray, y: np.ndarray) -> dict:
    e = pred - y
    ae = np.abs(e)
    s = float(ae.sum())
    order = np.sort(ae)[::-1]
    return {
        "n": int(len(y)),
        "mae": float(ae.mean()),
        "rmse": float(np.sqrt(np.mean(e**2))),
        "median_ae": float(np.median(ae)),
        "p90_ae": float(np.percentile(ae, 90)),
        "p95_ae": float(np.percentile(ae, 95)),
        "max_ae": float(ae.max()),
        "mean_signed_error": float(e.mean()),
        "positive_error_fraction": float(np.mean(e > 0)),
        "top1_abs_error_share": float(order[:1].sum() / s),
        "top10_abs_error_share": float(order[:10].sum() / s),
        "top50_abs_error_share": float(order[:50].sum() / s),
        "top100_abs_error_share": float(order[:100].sum() / s),
    }


def bootstrap_means(diff: np.ndarray, idx: np.ndarray) -> dict:
    means = diff[idx].mean(axis=1)
    return {
        "mean_delta_mae": float(diff.mean()),
        "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
    }


def main() -> int:
    runs, y, valid = C.load_all_predictions()
    names = list(runs)
    pred = {k: runs[k]["pred"] for k in names}
    ae = {k: np.abs(pred[k] - y) for k in names}
    err = {k: pred[k] - y for k in names}

    # ---- model manifest -------------------------------------------------
    manifest = {
        "protocol_id": "zinc-generalization-audit-v1",
        "split": "official PyG ZINC subset valid, n=1000; raw-y; not a fresh blind test",
        "official_test_loaded": False,
        "models": {},
    }
    for k in names:
        r = runs[k]
        info = r["info"]
        manifest["models"][k] = {
            "family": r["meta"]["family"],
            "seed": r["meta"]["seed"],
            "recipe": r["meta"]["recipe"],
            "ssl": r["meta"]["ssl"],
            "selection": r["meta"]["selection"],
            "parameters": r["meta"]["params"],
            "run_dir": str(r["run"].relative_to(C.REPO)),
            "prediction_path": str((r["run"] / "predictions" / "valid_best.npz").relative_to(C.REPO)),
            "prediction_sha256": C.sha256_file(r["run"] / "predictions" / "valid_best.npz"),
            "best_valid_recorded": float(info["fixed_objects"]["best"]["valid_mae"]),
            "best_epoch": int(info["fixed_objects"]["best"]["epoch"]),
            "train_mae_at_best_eval": float(info["fixed_objects"]["best"]["train_mae_eval_mode"]),
            "final_valid_recorded": float(info["fixed_objects"]["final"]["valid_mae"]),
            "final_train_mae_eval": float(info["fixed_objects"]["final"]["train_mae_eval_mode"]),
        }
    C.write_json(ANALYSIS / "model_manifest.json", manifest)

    # ---- per-model metrics ---------------------------------------------
    model_metrics = {k: metrics(pred[k], y) for k in names}
    C.write_json(ANALYSIS / "model_metrics.json", model_metrics)

    # ---- signed vs absolute error --------------------------------------
    error_relations = {}
    for k in names:
        error_relations[k] = {
            "pearson_signed_vs_abs": float(pearsonr(err[k], ae[k]).statistic),
            "spearman_signed_vs_abs": float(spearmanr(err[k], ae[k]).statistic),
            "mean_signed_error": float(err[k].mean()),
            "median_signed_error": float(np.median(err[k])),
            "sign_agreement_with_P3B": float(np.mean(np.sign(err[k]) == np.sign(err["P3B_s0"]))),
        }

    # ---- pairwise error correlation / direction / win rates ------------
    pairs = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            d = ae[a] - ae[b]
            pairs[f"{a}_minus_{b}"] = {
                "pearson_signed_errors": float(pearsonr(err[a], err[b]).statistic),
                "spearman_signed_errors": float(spearmanr(err[a], err[b]).statistic),
                "sign_agreement": float(np.mean(np.sign(err[a]) == np.sign(err[b]))),
                "prediction_pearson": float(pearsonr(pred[a], pred[b]).statistic),
                "fraction_a_better": float(np.mean(ae[a] < ae[b])),
                "fraction_b_better": float(np.mean(ae[b] < ae[a])),
                "mean_delta_ae": float(d.mean()),
                "median_delta_ae": float(np.median(d)),
            }
    C.write_json(ANALYSIS / "error_relations.json", {"per_model": error_relations, "pairs": pairs})

    # ---- fixed P3B worst-10 set ----------------------------------------
    worst10 = np.argsort(ae["P3B_s0"])[-10:][::-1]
    worst10_rows = []
    for k in names:
        worst10_rows.append(
            {
                "model": k,
                "mean_ae_on_P3B_worst10": float(ae[k][worst10].mean()),
                "median_ae_on_P3B_worst10": float(np.median(ae[k][worst10])),
                "max_ae_on_P3B_worst10": float(ae[k][worst10].max()),
                "fraction_better_than_P3B_on_set": float(np.mean(ae[k][worst10] < ae["P3B_s0"][worst10])),
            }
        )
    C.write_json(
        ANALYSIS / "fixed_worst10.json",
        {
            "selection": "top-10 absolute error of fixed primary reference P3B_s0, fixed before looking at other models",
            "row_ids": [int(i) for i in worst10],
            "y": [float(y[i]) for i in worst10],
            "P3B_abs_errors": [float(ae["P3B_s0"][i]) for i in worst10],
            "models": worst10_rows,
        },
    )

    # ---- pre-fixed equal-weight ensembles -------------------------------
    ensemble_defs = {
        "P3A_s0+s1": ["P3A_s0", "P3A_s1"],
        "M_s0+s1": ["M_s0", "M_s1"],
        "P3B_s0+J_s0": ["P3B_s0", "J_s0"],
    }
    ensembles = {}
    rng = np.random.default_rng(20261021)
    boot_idx = rng.integers(0, len(y), size=(10000, len(y)))
    for label, members in ensemble_defs.items():
        p = np.mean([pred[m] for m in members], axis=0)
        m = metrics(p, y)
        # Reference: best available single member on this split, and primary P3B.
        member_maes = {x: float(ae[x].mean()) for x in members}
        best_member = min(member_maes, key=member_maes.get)
        diff_best = ae[best_member] - np.abs(p - y)
        diff_p3b = ae["P3B_s0"] - np.abs(p - y)
        ensembles[label] = {
            "members": members,
            "equal_weight": True,
            "metrics": m,
            "member_maes": member_maes,
            "best_member": best_member,
            "vs_best_member": bootstrap_means(diff_best, boot_idx),
            "vs_P3B_s0": bootstrap_means(diff_p3b, boot_idx),
            "note": "diagnostic error cancellation only; not a <=500k single model and not a paper ensemble",
        }
    C.write_json(ANALYSIS / "ensembles.json", ensembles)

    # ---- two-seed consensus / disagreement ------------------------------
    seed_pairs = {"P3A": ("P3A_s0", "P3A_s1"), "M": ("M_s0", "M_s1")}
    consensus = {}
    for label, (a, b) in seed_pairs.items():
        avg = 0.5 * (pred[a] + pred[b])
        e = avg - y
        dis = np.abs(pred[a] - pred[b])
        mse = float(np.mean(e**2))
        var = float(np.mean((0.5 * (pred[a] - pred[b])) ** 2))
        consensus[label] = {
            "members": [a, b],
            "consensus_mae": float(np.abs(e).mean()),
            "consensus_median_ae": float(np.median(np.abs(e))),
            "mean_signed_residual": float(e.mean()),
            "prediction_disagreement_mean": float(dis.mean()),
            "prediction_disagreement_median": float(np.median(dis)),
            "prediction_disagreement_p90": float(np.percentile(dis, 90)),
            "corr_abs_consensus_error_vs_disagreement_pearson": float(pearsonr(np.abs(e), dis).statistic),
            "corr_abs_consensus_error_vs_disagreement_spearman": float(spearmanr(np.abs(e), dis).statistic),
            "mse_of_consensus": mse,
            "algebraic_bias2": mse - var,
            "algebraic_seed_var": var,
            "algebraic_seed_var_share": var / mse,
            "note": "Two fixed checkpoints only; algebraic two-model identity, not a population bias/variance decomposition.",
        }
    C.write_json(ANALYSIS / "seed_consensus.json", consensus)

    # ---- key paired bootstrap contrasts ---------------------------------
    key = [
        ("J_s0", "P3B_s0"),
        ("Q_s0", "P3B_s0"),
        ("J_s0", "Q_s0"),
        ("P3A_s1", "P3B_s0"),
        ("M_s0", "P3B_s0"),
        ("M_s1", "P3B_s0"),
    ]
    contrasts = {}
    for a, b in key:
        contrasts[f"{a}_minus_{b}"] = bootstrap_means(ae[a] - ae[b], boot_idx)
    C.write_json(
        ANALYSIS / "key_contrasts.json",
        {
            "resamples": 10000,
            "rng_seed": 20261021,
            "note": "Same resampled molecule IDs for every contrast. Descriptive fixed-checkpoint molecule uncertainty only.",
            "contrasts": contrasts,
        },
    )

    # ---- error rank table for all models (useful machine-readable) ------
    ranks = np.empty_like(np.stack([ae[k] for k in names], axis=0), dtype=np.int64)
    for j, k in enumerate(names):
        ranks[j] = np.argsort(np.argsort(ae[k]))
    C.write_json(
        ANALYSIS / "error_summary.json",
        {
            "model_metrics": model_metrics,
            "median_ae_across_models_mean": float(np.median(np.stack([ae[k] for k in names], axis=0), axis=0).mean()),
            "consensus_equal_weight_all7_mae": float(np.abs(np.mean([pred[k] for k in names], axis=0) - y).mean()),
            "notes": [
                "All metrics use all 1000 valid rows; top-k shares are only tail decompositions.",
                "P3B_s0 is the primary fixed reference; P3A and M rows are different recipes/initialisations.",
            ],
        },
    )
    C.write_json(ANALYSIS / "pairwise_error_ranks.json", {"models": names, "rank_matrix": ranks.tolist()})
    print(json.dumps({"ok": True, "models": names, "mae": {k: model_metrics[k]["mae"] for k in names}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
