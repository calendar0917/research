"""Analyze train-only scaffold-fold complementarity of Morgan and saved structural models.

No model is trained here and official valid/test are never read.  All arrays are
held-out predictions from the three precomputed scaffold folds inside official
train.  Blend weights are shared across folds; per-fold oracle weights are not
used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import expit
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score


def sha256(x: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()


def rank01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return rankdata(x, method="average") / len(x)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def row(indices: Any, labels: Any, values: Any, value_kind: str) -> dict[str, np.ndarray]:
    idx = np.asarray(indices, dtype=np.int64)
    y = np.asarray(labels, dtype=np.int64)
    raw = np.asarray(values, dtype=np.float64)
    if not (len(idx) == len(y) == len(raw)):
        raise ValueError("prediction length mismatch")
    if value_kind == "probability":
        prob = raw
    elif value_kind == "logit":
        prob = expit(raw)
    else:
        raise ValueError(value_kind)
    return {"indices": idx, "y": y, "prob": prob, "raw": raw}


def align(reference: dict[str, np.ndarray], other: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if np.array_equal(reference["indices"], other["indices"]):
        order = np.arange(len(other["indices"]))
    else:
        pos = {int(index): i for i, index in enumerate(other["indices"].tolist())}
        try:
            order = np.asarray([pos[int(index)] for index in reference["indices"]], dtype=np.int64)
        except KeyError as exc:
            raise ValueError(f"heldout index mismatch: {exc}") from exc
    out = {key: value[order] for key, value in other.items()}
    if not np.array_equal(reference["indices"], out["indices"]):
        raise ValueError("failed to align heldout indices")
    if not np.array_equal(reference["y"], out["y"]):
        raise ValueError("heldout label mismatch")
    return out


def auc(y: np.ndarray, score: np.ndarray) -> float:
    return float(roc_auc_score(y, score))


def summarize(values: list[float]) -> dict[str, Any]:
    a = np.asarray(values, dtype=np.float64)
    return {
        "fold_auc": a.tolist(),
        "mean_auc": float(a.mean()),
        "sample_std_auc": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        "min_fold_auc": float(a.min()),
        "max_fold_auc": float(a.max()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="tracks/ksvd/results/molhiv")
    ap.add_argument(
        "--output",
        default="tracks/ksvd/results/molhiv/internal_fingerprint_complementarity_20260729.json",
    )
    args = ap.parse_args()
    root = Path(args.results_dir)

    fp = load_json(root / "chemical_fingerprint_fulltrain_probe_20260729.json")
    folds: list[dict[str, dict[str, np.ndarray]]] = []
    for fold in range(3):
        controls = fp["folds"][fold]["controls"]
        r2 = controls["morgan_r2_logreg_c0.01"]
        reference = row(
            r2["heldout_indices"], r2["heldout_y"], r2["heldout_probabilities"], "probability"
        )
        r3 = controls["morgan_r3_logreg_c0.01"]
        models: dict[str, dict[str, np.ndarray]] = {
            "morgan_r2_c0.01": reference,
            "morgan_r3_c0.01": align(
                reference,
                row(r3["heldout_indices"], r3["heldout_y"], r3["heldout_probabilities"], "probability"),
            ),
        }

        stable = load_json(root / f"scaffold_stable_vocabulary_n41127_fold{fold}_seed0.json")
        for key in ("farthest", "scaffold_facility"):
            held = stable["results"][key]["heldout"]
            models[f"realproto_{key}"] = align(
                reference, row(held["graph_indices"], held["labels"], held["scores"], "logit")
            )

        gates = load_json(root / f"fixed_vocabulary_relation_gates_n41127_fold{fold}_seed0.json")
        held = gates["results"]["uniform__real"]["heldout"]
        models["realproto_broad_pair"] = align(
            reference, row(held["graph_indices"], held["labels"], held["base_scores"], "logit")
        )
        models["realproto_uniform_relation"] = align(
            reference, row(held["graph_indices"], held["labels"], held["scores"], "logit")
        )

        pair = load_json(root / f"label_free_pair_pca_n41127_fold{fold}_seed0.json")
        for key in ("covariance_pca__real", "correlation_pca__real"):
            held = pair["results"][key]["heldout"]
            models[f"pairpca_{key.removesuffix('__real')}"] = align(
                reference, row(held["graph_indices"], held["labels"], held["scores"], "logit")
            )

        # Explicit fixed Morgan-only ensemble, not fitted per fold.
        mean_prob = 0.5 * models["morgan_r2_c0.01"]["prob"] + 0.5 * models["morgan_r3_c0.01"]["prob"]
        models["morgan_r2_r3_prob_equal"] = {
            "indices": reference["indices"], "y": reference["y"], "prob": mean_prob, "raw": mean_prob
        }
        mean_rank = 0.5 * rank01(models["morgan_r2_c0.01"]["prob"]) + 0.5 * rank01(models["morgan_r3_c0.01"]["prob"])
        models["morgan_r2_r3_rank_equal"] = {
            "indices": reference["indices"], "y": reference["y"], "prob": mean_rank, "raw": mean_rank
        }
        folds.append(models)

    model_names = list(folds[0])
    if any(list(models) != model_names for models in folds[1:]):
        raise ValueError("model set mismatch across folds")

    singles: dict[str, Any] = {}
    for name in model_names:
        values = [auc(models[name]["y"], models[name]["prob"]) for models in folds]
        all_y = np.concatenate([models[name]["y"] for models in folds])
        all_p = np.concatenate([models[name]["prob"] for models in folds])
        singles[name] = {
            **summarize(values),
            "concatenated_oof_auc": auc(all_y, all_p),
            "fold_probability_sha256": [sha256(models[name]["prob"]) for models in folds],
        }

    correlations: dict[str, Any] = {}
    morgan_names = ["morgan_r2_c0.01", "morgan_r3_c0.01", "morgan_r2_r3_rank_equal"]
    structural_names = [name for name in model_names if not name.startswith("morgan_")]
    for morgan in morgan_names:
        for structural in structural_names:
            key = f"{morgan}__vs__{structural}"
            pearson, spearman = [], []
            for models in folds:
                a, b = models[morgan]["prob"], models[structural]["prob"]
                pearson.append(float(np.corrcoef(a, b)[0, 1]))
                spearman.append(float(spearmanr(a, b).statistic))
            correlations[key] = {
                "pearson_per_fold": pearson,
                "pearson_mean": float(np.mean(pearson)),
                "spearman_per_fold": spearman,
                "spearman_mean": float(np.mean(spearman)),
            }

    # Shared weights only.  Weight means Morgan contribution.
    weights = np.linspace(0.0, 1.0, 11)
    blends: list[dict[str, Any]] = []
    morgan_candidates = ["morgan_r2_c0.01", "morgan_r3_c0.01", "morgan_r2_r3_rank_equal"]
    for morgan in morgan_candidates:
        for structural in structural_names:
            for mode in ("probability", "rank"):
                for weight in weights:
                    fold_auc = []
                    for models in folds:
                        y = models[morgan]["y"]
                        a = models[morgan]["prob"]
                        b = models[structural]["prob"]
                        if mode == "rank":
                            a, b = rank01(a), rank01(b)
                        score = float(weight) * a + (1.0 - float(weight)) * b
                        fold_auc.append(auc(y, score))
                    base_m = singles[morgan]["fold_auc"]
                    base_s = singles[structural]["fold_auc"]
                    summary = summarize(fold_auc)
                    blends.append({
                        "morgan": morgan,
                        "structural": structural,
                        "mode": mode,
                        "morgan_weight": float(weight),
                        **summary,
                        "delta_vs_morgan_per_fold": (np.asarray(fold_auc) - np.asarray(base_m)).tolist(),
                        "delta_vs_structural_per_fold": (np.asarray(fold_auc) - np.asarray(base_s)).tolist(),
                        "mean_delta_vs_morgan": float(np.mean(np.asarray(fold_auc) - np.asarray(base_m))),
                        "mean_delta_vs_structural": float(np.mean(np.asarray(fold_auc) - np.asarray(base_s))),
                        "wins_vs_morgan": int(np.sum(np.asarray(fold_auc) > np.asarray(base_m))),
                        "wins_vs_structural": int(np.sum(np.asarray(fold_auc) > np.asarray(base_s))),
                    })

    nontrivial = [x for x in blends if 0.0 < x["morgan_weight"] < 1.0]
    stable_gain = [
        x for x in nontrivial
        if x["wins_vs_morgan"] == 3 and x["wins_vs_structural"] == 3
    ]
    ranking = sorted(
        nontrivial,
        key=lambda x: (x["mean_auc"], x["min_fold_auc"], -x["sample_std_auc"]),
        reverse=True,
    )
    stable_ranking = sorted(
        stable_gain,
        key=lambda x: (x["mean_auc"], x["min_fold_auc"], -x["sample_std_auc"]),
        reverse=True,
    )

    output: dict[str, Any] = {
        "protocol_id": "molhiv-officialtrain-internal-fingerprint-complementarity-v1",
        "date": "2026-07-29",
        "scope": "three held-out scaffold folds inside official train only",
        "official_valid_evaluations": 0,
        "official_test_evaluations": 0,
        "selection_rule": "one shared weight across all folds; no per-fold oracle weights",
        "single_models": singles,
        "correlations": correlations,
        "all_blends": blends,
        "top_nontrivial_blends": ranking[:30],
        "top_stable_3of3_blends": stable_ranking[:30],
        "heldout_index_sha256": [sha256(models[model_names[0]]["indices"]) for models in folds],
        "heldout_label_sha256": [sha256(models[model_names[0]]["y"]) for models in folds],
    }
    Path(args.output).write_text(json.dumps(output, indent=2))
    print(json.dumps({
        "output": args.output,
        "single_models": {name: singles[name]["mean_auc"] for name in model_names},
        "best_nontrivial": ranking[0] if ranking else None,
        "best_stable_3of3": stable_ranking[0] if stable_ranking else None,
        "n_stable_3of3": len(stable_gain),
    }, indent=2))


if __name__ == "__main__":
    main()
