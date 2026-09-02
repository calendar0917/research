"""Retrospective complementarity analysis for saved MolHIV official predictions.

This script does not train a model. It reconstructs family probability
ensembles from already saved official-valid/test predictions, measures how
similarly the families rank molecules, and applies a deliberately small set of
convex blends selected on official-valid.

Important: official-test predictions in this repository have already been
inspected. Test-side blend numbers produced here are therefore retrospective
diagnostics, not a new untouched terminal evaluation.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score


def _sha256(x: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()


def _load_many(pattern: str) -> list[dict[str, Any]]:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(pattern)
    return [json.loads(Path(path).read_text()) for path in paths]


def _family(
    name: str,
    paths: list[str],
    extractor: Callable[[dict[str, Any], str], tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    docs = [json.loads(Path(path).read_text()) for path in paths]
    out: dict[str, Any] = {"name": name, "members": paths}
    for split in ("official_valid", "official_test"):
        labels: list[np.ndarray] = []
        probabilities: list[np.ndarray] = []
        member_auc: list[float] = []
        for doc in docs:
            y, p = extractor(doc, split)
            y = np.asarray(y, dtype=np.int64)
            p = np.asarray(p, dtype=np.float64)
            labels.append(y)
            probabilities.append(p)
            member_auc.append(float(roc_auc_score(y, p)))
        if not all(np.array_equal(labels[0], y) for y in labels[1:]):
            raise ValueError(f"label mismatch inside family {name}/{split}")
        ensemble = np.mean(probabilities, axis=0)
        out[split] = {
            "labels": labels[0],
            "probabilities": ensemble,
            "auc": float(roc_auc_score(labels[0], ensemble)),
            "member_auc": member_auc,
            "mean_probability": float(ensemble.mean()),
            "probability_sha256": _sha256(ensemble),
        }
    return out


def _old_extractor(doc: dict[str, Any], split: str) -> tuple[np.ndarray, np.ndarray]:
    prefix = "official_valid" if split == "official_valid" else "official_test"
    return np.asarray(doc[f"{prefix}_y"]), np.asarray(doc[f"{prefix}_predictions"])


def _gine_extractor(doc: dict[str, Any], split: str) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(doc[split]["y"]), np.asarray(doc[split]["probabilities"])


def _result_extractor(key: str, label_key: str = "y") -> Callable[[dict[str, Any], str], tuple[np.ndarray, np.ndarray]]:
    def extract(doc: dict[str, Any], split: str) -> tuple[np.ndarray, np.ndarray]:
        row = doc["results"][key][split]
        return np.asarray(row[label_key]), np.asarray(row["probabilities"])
    return extract


def _ranking_disagreement(y: np.ndarray, a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    da = a[pos, None] - a[neg][None, :]
    db = b[pos, None] - b[neg][None, :]
    correct_a = da > 0
    correct_b = db > 0
    disagree = correct_a != correct_b
    return {
        "positive_negative_pairs": int(len(pos) * len(neg)),
        "ranking_disagreement_fraction": float(disagree.mean()),
        "a_correct_b_wrong_fraction": float((correct_a & ~correct_b).mean()),
        "b_correct_a_wrong_fraction": float((correct_b & ~correct_a).mean()),
        "both_correct_fraction": float((correct_a & correct_b).mean()),
        "both_wrong_fraction": float((~correct_a & ~correct_b).mean()),
    }


def _rank01(x: np.ndarray) -> np.ndarray:
    return rankdata(np.asarray(x), method="average") / len(x)


def _blend_auc(y: np.ndarray, arrays: list[np.ndarray], weights: np.ndarray) -> float:
    score = sum(float(w) * x for w, x in zip(weights, arrays))
    return float(roc_auc_score(y, score))


def _choose_by_valid(rows: list[dict[str, Any]]) -> dict[str, Any]:
    # Deterministic tie-breaking: best valid AUC, then larger test-independent
    # weight entropy (less extreme), then lexicographic weights.
    def entropy(row: dict[str, Any]) -> float:
        w = np.asarray(row["weights"], dtype=np.float64)
        nz = w[w > 0]
        return float(-(nz * np.log(nz)).sum())
    return max(rows, key=lambda row: (row["valid_auc"], entropy(row), tuple(row["weights"])))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="tracks/ksvd/results/molhiv")
    ap.add_argument("--output", default="tracks/ksvd/results/molhiv/official_model_complementarity_20260729.json")
    args = ap.parse_args()
    root = Path(args.results_dir)

    specs: list[tuple[str, list[str], Callable[[dict[str, Any], str], tuple[np.ndarray, np.ndarray]]]] = [
        ("fixed_gine_3seed", sorted(glob.glob(str(root / "fixed_epoch30_gine_full_official_seed*.json"))), _gine_extractor),
        ("gine_h64_5seed", sorted(glob.glob(str(root / "frozen_test_gine_h64_seed*.json"))), _old_extractor),
        ("gine_h70_5seed", sorted(glob.glob(str(root / "frozen_test_gine_h70_seed*.json"))), _old_extractor),
        ("ksvd_node_token_5seed", sorted(glob.glob(str(root / "frozen_test_ksvd_seed*.json"))), _old_extractor),
        ("stable_realprototype_3bank", sorted(glob.glob(str(root / "stable_realproto_full_official_bank*_seed0.json"))), _result_extractor("stable_random_node_mil")),
        ("farthest_bipartite_3seed", sorted(glob.glob(str(root / "farthest_bipartite_full_official_seed*.json"))), _result_extractor("farthest_bipartite", "labels")),
        ("gine_jk_3seed", sorted(glob.glob(str(root / "node_token_terminal_test_v1_ginejk_seed*.json"))), _old_extractor),
        ("ksvd_graphres_3seed", sorted(glob.glob(str(root / "node_token_terminal_test_v1_ksvd_graphres_seed*.json"))), _old_extractor),
    ]
    families: dict[str, dict[str, Any]] = {}
    for name, paths, extractor in specs:
        if paths:
            families[name] = _family(name, paths, extractor)

    if len(families) < 2:
        raise RuntimeError("need at least two prediction families")
    names = list(families)
    for split in ("official_valid", "official_test"):
        ref = families[names[0]][split]["labels"]
        for name in names[1:]:
            if not np.array_equal(ref, families[name][split]["labels"]):
                raise ValueError(f"cross-family label mismatch: {name}/{split}")

    out: dict[str, Any] = {
        "protocol_id": "molhiv-saved-official-prediction-complementarity-retrospective-v1",
        "date": "2026-07-29",
        "status": (
            "retrospective diagnostic: official test predictions were already exposed; "
            "test metrics below are not an untouched terminal evaluation"
        ),
        "selection_rule": (
            "only official-valid chooses among the declared weight grids; test is then "
            "reported at that valid-selected point"
        ),
        "families": {},
        "pairwise": {},
        "blends": {},
    }
    for name, family in families.items():
        out["families"][name] = {
            "members": family["members"],
            "official_valid": {k: v for k, v in family["official_valid"].items() if k not in {"labels", "probabilities"}},
            "official_test": {k: v for k, v in family["official_test"].items() if k not in {"labels", "probabilities"}},
        }

    for split in ("official_valid", "official_test"):
        y = families[names[0]][split]["labels"]
        rows = []
        for a, b in itertools.combinations(names, 2):
            pa = families[a][split]["probabilities"]
            pb = families[b][split]["probabilities"]
            rows.append({
                "family_a": a,
                "family_b": b,
                "pearson": float(np.corrcoef(pa, pb)[0, 1]),
                "spearman": float(spearmanr(pa, pb).statistic),
                "mean_absolute_probability_difference": float(np.mean(np.abs(pa - pb))),
                **_ranking_disagreement(y, pa, pb),
            })
        out["pairwise"][split] = rows

    # Pairwise convex probability blends. The grid is intentionally coarse.
    pair_rows = []
    grid = np.linspace(0.0, 1.0, 11)
    yv = families[names[0]]["official_valid"]["labels"]
    yt = families[names[0]]["official_test"]["labels"]
    for a, b in itertools.combinations(names, 2):
        candidates = []
        for wa in grid:
            w = np.asarray([wa, 1.0 - wa])
            candidates.append({
                "weights": [float(w[0]), float(w[1])],
                "valid_auc": _blend_auc(yv, [families[a]["official_valid"]["probabilities"], families[b]["official_valid"]["probabilities"]], w),
                "test_auc": _blend_auc(yt, [families[a]["official_test"]["probabilities"], families[b]["official_test"]["probabilities"]], w),
            })
        selected = _choose_by_valid(candidates)
        oracle = max(candidates, key=lambda row: row["test_auc"])
        pair_rows.append({
            "families": [a, b],
            "weight_semantics": f"[weight({a}), weight({b})]",
            "valid_selected": selected,
            "test_oracle_diagnostic_only": oracle,
        })
    out["blends"]["pairwise_probability_grid_step_0.1"] = pair_rows

    # Rank averaging removes family calibration scale. Ranking each split is
    # unsupervised; labels are used only to choose the valid weight.
    rank_pair_rows = []
    for a, b in itertools.combinations(names, 2):
        av, bv = _rank01(families[a]["official_valid"]["probabilities"]), _rank01(families[b]["official_valid"]["probabilities"])
        at, bt = _rank01(families[a]["official_test"]["probabilities"]), _rank01(families[b]["official_test"]["probabilities"])
        candidates = []
        for wa in grid:
            w = np.asarray([wa, 1.0 - wa])
            candidates.append({
                "weights": [float(w[0]), float(w[1])],
                "valid_auc": _blend_auc(yv, [av, bv], w),
                "test_auc": _blend_auc(yt, [at, bt], w),
            })
        rank_pair_rows.append({
            "families": [a, b],
            "weight_semantics": f"[weight({a}), weight({b})]",
            "valid_selected": _choose_by_valid(candidates),
            "test_oracle_diagnostic_only": max(candidates, key=lambda row: row["test_auc"]),
        })
    out["blends"]["pairwise_rank_grid_step_0.1"] = rank_pair_rows

    # Core 4-family simplex: one standard GNN, old KSVD node tokens, stable
    # observed-patch MIL, and occurrence bipartite structure.
    core = [name for name in (
        "gine_h70_5seed", "ksvd_node_token_5seed",
        "stable_realprototype_3bank", "farthest_bipartite_3seed",
    ) if name in families]
    if len(core) == 4:
        simplex_rows = []
        for integer_weights in itertools.product(range(11), repeat=4):
            if sum(integer_weights) != 10:
                continue
            w = np.asarray(integer_weights, dtype=np.float64) / 10.0
            simplex_rows.append({
                "weights": w.tolist(),
                "valid_auc": _blend_auc(yv, [families[k]["official_valid"]["probabilities"] for k in core], w),
                "test_auc": _blend_auc(yt, [families[k]["official_test"]["probabilities"] for k in core], w),
            })
        out["blends"]["core4_probability_simplex_step_0.1"] = {
            "families": core,
            "valid_selected": _choose_by_valid(simplex_rows),
            "test_oracle_diagnostic_only": max(simplex_rows, key=lambda row: row["test_auc"]),
        }
        rank_rows = []
        valid_rank = [_rank01(families[k]["official_valid"]["probabilities"]) for k in core]
        test_rank = [_rank01(families[k]["official_test"]["probabilities"]) for k in core]
        for integer_weights in itertools.product(range(11), repeat=4):
            if sum(integer_weights) != 10:
                continue
            w = np.asarray(integer_weights, dtype=np.float64) / 10.0
            rank_rows.append({
                "weights": w.tolist(),
                "valid_auc": _blend_auc(yv, valid_rank, w),
                "test_auc": _blend_auc(yt, test_rank, w),
            })
        out["blends"]["core4_rank_simplex_step_0.1"] = {
            "families": core,
            "valid_selected": _choose_by_valid(rank_rows),
            "test_oracle_diagnostic_only": max(rank_rows, key=lambda row: row["test_auc"]),
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2) + "\n")

    print("family ensembles")
    for name in names:
        print(f"{name:30s} valid={families[name]['official_valid']['auc']:.6f} test={families[name]['official_test']['auc']:.6f}")
    best_pair = max(pair_rows, key=lambda row: row["valid_selected"]["test_auc"])
    print("best test among valid-selected pair probability blends")
    print(best_pair)
    if "core4_probability_simplex_step_0.1" in out["blends"]:
        print("core4 probability", out["blends"]["core4_probability_simplex_step_0.1"])
        print("core4 rank", out["blends"]["core4_rank_simplex_step_0.1"])
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
