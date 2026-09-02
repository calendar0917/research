"""Frozen low-capacity Morgan candidates on MolHIV official valid.

The candidate set was fixed from three scaffold folds inside official train:
  * Morgan radius 2, C=0.01
  * Morgan radius 3, C=0.01
  * fixed equal probability/rank ensembles of radius 2 and 3
  * fixed 0.6 Morgan-rank + 0.4 structural-rank blends, using already-frozen
    compact/pair-PCA official-valid predictions.

Only official train is used for fitting.  Test molecules are neither parsed nor
fingerprinted by this runner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import sys
_TRACK = Path(__file__).resolve().parents[1]
if str(_TRACK) not in sys.path:
    sys.path.insert(0, str(_TRACK))
from code.data_molhiv import load_molhiv


def sha256(x: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()


def rank01(x: np.ndarray) -> np.ndarray:
    return rankdata(np.asarray(x, dtype=np.float64), method="average") / len(x)


def result(y_fit: np.ndarray, p_fit: np.ndarray, indices: np.ndarray, y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    return {
        "fit_auc": float(roc_auc_score(y_fit, p_fit)),
        "official_valid_auc": float(roc_auc_score(y, p)),
        "official_valid_indices": indices.astype(int).tolist(),
        "official_valid_y": y.astype(int).tolist(),
        "official_valid_probabilities": np.asarray(p, dtype=np.float64).tolist(),
        "official_valid_probability_sha256": sha256(np.asarray(p, dtype=np.float64)),
    }


def load_structural_valid(results_dir: Path, valid_indices: np.ndarray, valid_y: np.ndarray) -> dict[str, np.ndarray]:
    docs = [json.loads((results_dir / f"label_free_pair_pca_n41127_official_valid_seed{seed}.json").read_text()) for seed in range(3)]
    pair_prob = []
    compact_prob = []
    for doc in docs:
        held = doc["results"]["covariance_pca__real"]["heldout"]
        idx = np.asarray(held["graph_indices"], dtype=np.int64)
        y = np.asarray(held["labels"], dtype=np.int64)
        if not np.array_equal(idx, valid_indices) or not np.array_equal(y, valid_y):
            raise ValueError("structural official-valid prediction alignment mismatch")
        pair_prob.append(expit(np.asarray(held["scores"], dtype=np.float64)))
        compact_prob.append(expit(np.asarray(held["base_scores"], dtype=np.float64)))
    if max(float(np.max(np.abs(compact_prob[0] - x))) for x in compact_prob[1:]) > 0:
        raise ValueError("compact base differs across pair-PCA seeds")
    return {
        "compact_base": compact_prob[0],
        "pairpca_covariance_3seed": np.mean(pair_prob, axis=0),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=2048)
    ap.add_argument("--c", type=float, default=0.01)
    ap.add_argument("--results-dir", default="tracks/ksvd/results/molhiv")
    ap.add_argument("--output", default="tracks/ksvd/results/molhiv/chemical_fingerprint_official_valid_20260729.json")
    args = ap.parse_args()
    t0 = time.time()
    repo = Path(__file__).resolve().parents[3]
    results_dir = Path(args.results_dir)

    bundle = load_molhiv(repo / "data" / "ogb", max_graphs=None, seed=0, with_features=False)
    y_all = np.asarray(bundle.y, dtype=np.int64)
    original_indices = np.asarray(bundle.meta["original_indices"], dtype=np.int64)
    train = np.asarray(bundle.split["train"], dtype=np.int64)
    valid = np.asarray(bundle.split["valid"], dtype=np.int64)
    test = np.asarray(bundle.split["test"], dtype=np.int64)
    allowed = np.concatenate([train, valid])

    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    mapping = pd.read_csv(repo / "data" / "ogb" / "ogbg_molhiv" / "mapping" / "mol.csv.gz")
    smiles = mapping.iloc[original_indices]["smiles"].astype(str).tolist()
    mols: list[Any | None] = [None] * len(smiles)
    fallback = []
    for i in allowed.tolist():
        mol = Chem.MolFromSmiles(smiles[int(i)])
        if mol is None:
            mol = Chem.MolFromSmiles(smiles[int(i)], sanitize=False)
            if mol is not None:
                mol.UpdatePropertyCache(strict=False)
                Chem.GetSymmSSSR(mol)
                fallback.append(int(i))
        if mol is None:
            raise RuntimeError(f"RDKit parse failed local={i} original={original_indices[int(i)]}")
        mols[int(i)] = mol
    if any(mols[int(i)] is not None for i in test.tolist()):
        raise AssertionError("official test molecule was parsed")

    matrices: dict[int, np.ndarray] = {}
    for radius in (2, 3):
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=args.bits, includeChirality=True)
        matrix = np.zeros((len(smiles), args.bits), dtype=np.uint8)
        for i in allowed.tolist():
            matrix[int(i)] = generator.GetFingerprintAsNumPy(mols[int(i)]).astype(np.uint8)
        if np.any(matrix[test]):
            raise AssertionError("official test fingerprint rows must remain zero")
        matrices[radius] = matrix

    output: dict[str, Any] = {
        "protocol_id": "molhiv-chemical-fingerprint-official-valid-v1",
        "date": "2026-07-29",
        "candidate_source": "three official-train internal scaffold folds",
        "fit_split": "exact official train",
        "evaluation_split": "exact official valid",
        "official_valid_evaluations": 1,
        "official_test_evaluations": 0,
        "test_isolation": "test molecules not parsed; test fingerprint rows exactly zero",
        "config": vars(args),
        "n_train": int(len(train)), "n_train_positive": int(y_all[train].sum()),
        "n_valid": int(len(valid)), "n_valid_positive": int(y_all[valid].sum()),
        "n_test": int(len(test)),
        "train_indices_sha256": sha256(train), "valid_indices_sha256": sha256(valid), "test_indices_sha256": sha256(test),
        "fallback_unsanitized_local_indices": fallback,
        "fingerprint_train_hashes": {f"morgan_r{r}": sha256(x[train]) for r, x in matrices.items()},
        "fingerprint_valid_hashes": {f"morgan_r{r}": sha256(x[valid]) for r, x in matrices.items()},
        "candidates": {},
    }

    probabilities: dict[str, np.ndarray] = {}
    fit_probabilities: dict[str, np.ndarray] = {}
    for radius in (2, 3):
        model = LogisticRegression(
            C=args.c, class_weight="balanced", solver="liblinear", max_iter=3000, random_state=0
        )
        x = matrices[radius].astype(np.float32)
        model.fit(x[train], y_all[train])
        p_fit = model.predict_proba(x[train])[:, 1]
        p_valid = model.predict_proba(x[valid])[:, 1]
        name = f"morgan_r{radius}_c0.01"
        probabilities[name] = p_valid
        fit_probabilities[name] = p_fit
        row_out = result(y_all[train], p_fit, valid, y_all[valid], p_valid)
        row_out.update({
            "trainable_parameters": int(args.bits + 1),
            "coefficient_sha256": sha256(np.asarray(model.coef_, dtype=np.float64)),
            "intercept_sha256": sha256(np.asarray(model.intercept_, dtype=np.float64)),
            "n_iter": np.asarray(model.n_iter_, dtype=int).tolist(),
        })
        output["candidates"][name] = row_out

    prob_equal = 0.5 * probabilities["morgan_r2_c0.01"] + 0.5 * probabilities["morgan_r3_c0.01"]
    fit_prob_equal = 0.5 * fit_probabilities["morgan_r2_c0.01"] + 0.5 * fit_probabilities["morgan_r3_c0.01"]
    output["candidates"]["morgan_r2_r3_prob_equal"] = result(
        y_all[train], fit_prob_equal, valid, y_all[valid], prob_equal
    )
    rank_equal = 0.5 * rank01(probabilities["morgan_r2_c0.01"]) + 0.5 * rank01(probabilities["morgan_r3_c0.01"])
    fit_rank_equal = 0.5 * rank01(fit_probabilities["morgan_r2_c0.01"]) + 0.5 * rank01(fit_probabilities["morgan_r3_c0.01"])
    output["candidates"]["morgan_r2_r3_rank_equal"] = result(
        y_all[train], fit_rank_equal, valid, y_all[valid], rank_equal
    )

    structural = load_structural_valid(results_dir, valid, y_all[valid])
    output["structural_references"] = {}
    for name, p in structural.items():
        output["structural_references"][name] = {
            "official_valid_auc": float(roc_auc_score(y_all[valid], p)),
            "official_valid_probability_sha256": sha256(p),
        }
        blended = 0.6 * rank01(rank_equal) + 0.4 * rank01(p)
        candidate = f"rankblend_morgan06_{name}04"
        output["candidates"][candidate] = {
            "official_valid_auc": float(roc_auc_score(y_all[valid], blended)),
            "official_valid_indices": valid.astype(int).tolist(),
            "official_valid_y": y_all[valid].astype(int).tolist(),
            "official_valid_probabilities": blended.tolist(),
            "official_valid_probability_sha256": sha256(blended),
            "fixed_weights": {"morgan_r2_r3_rank_equal": 0.6, name: 0.4},
            "weight_source": "shared-weight three-fold official-train internal analysis",
        }

    ranked = sorted(
        ((name, row["official_valid_auc"]) for name, row in output["candidates"].items()),
        key=lambda x: x[1], reverse=True,
    )
    output["ranking"] = [{"rank": i + 1, "candidate": name, "official_valid_auc": value} for i, (name, value) in enumerate(ranked)]
    output["elapsed_sec"] = time.time() - t0
    Path(args.output).write_text(json.dumps(output, indent=2))
    print(json.dumps({"output": args.output, "ranking": output["ranking"], "elapsed_sec": output["elapsed_sec"]}, indent=2))


if __name__ == "__main__":
    main()
