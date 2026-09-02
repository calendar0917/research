"""Chemical-fragment fingerprint probe on MolHIV official-train scaffold folds.

This is a deliberately low-capacity diagnostic for a signal missing from the
current KSVD/GNN families. Morgan fingerprints explicitly enumerate hashed
local chemical environments from the original atom/bond graph. The script
also tests a label-matched prototype dictionary: positive and negative
training molecules are selected separately by farthest-point coverage under
Tanimoto similarity, then a small classifier reads similarities to them.

Only official-train internal scaffold folds are used. Official-valid/test
molecules are not fingerprinted or evaluated by this runner.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

import sys
_TRACK = Path(__file__).resolve().parents[1]
if str(_TRACK) not in sys.path:
    sys.path.insert(0, str(_TRACK))
from code.data_molhiv import load_molhiv


def _sha256(x: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()


def _tanimoto_to_one(x: np.ndarray, q: np.ndarray, x_sum: np.ndarray | None = None) -> np.ndarray:
    xf = np.asarray(x, dtype=np.float32)
    qf = np.asarray(q, dtype=np.float32)
    if x_sum is None:
        x_sum = xf.sum(axis=1)
    inter = xf @ qf
    union = x_sum + qf.sum() - inter
    return inter / np.maximum(union, 1.0)


def _tanimoto_matrix(x: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    xf = np.asarray(x, dtype=np.float32)
    pf = np.asarray(prototypes, dtype=np.float32)
    inter = xf @ pf.T
    union = xf.sum(axis=1, keepdims=True) + pf.sum(axis=1)[None, :] - inter
    return inter / np.maximum(union, 1.0)


def _select_farthest_tanimoto(x: np.ndarray, indices: np.ndarray, n_select: int) -> np.ndarray:
    """Deterministic max-min coverage within the provided training indices."""
    indices = np.asarray(indices, dtype=np.int64)
    if not len(indices):
        raise ValueError("empty prototype candidate set")
    n_select = min(int(n_select), len(indices))
    rows = np.asarray(x[indices], dtype=np.uint8)
    row_sum = rows.sum(axis=1).astype(np.float32)
    # Start from the chemically richest molecule; source-index tie-break is stable.
    first_local = int(np.flatnonzero(row_sum == row_sum.max())[0])
    chosen_local = [first_local]
    min_distance = 1.0 - _tanimoto_to_one(rows, rows[first_local], row_sum)
    for _ in range(1, n_select):
        min_distance[np.asarray(chosen_local, dtype=np.int64)] = -np.inf
        nxt = int(np.argmax(min_distance))
        chosen_local.append(nxt)
        min_distance = np.minimum(
            min_distance, 1.0 - _tanimoto_to_one(rows, rows[nxt], row_sum)
        )
    return indices[np.asarray(chosen_local, dtype=np.int64)]


def _summary_similarity(sim: np.ndarray) -> np.ndarray:
    sim = np.asarray(sim, dtype=np.float32)
    if sim.shape[1] == 0:
        return np.zeros((len(sim), 5), dtype=np.float32)
    sorted_sim = np.sort(sim, axis=1)
    blocks = [
        sorted_sim[:, -1],
        sorted_sim[:, -min(3, sim.shape[1]):].mean(axis=1),
        sorted_sim[:, -min(8, sim.shape[1]):].mean(axis=1),
        sim.mean(axis=1),
        sim.std(axis=1),
    ]
    return np.stack(blocks, axis=1).astype(np.float32)


def _descriptor_matrix(mols: list[Any]) -> tuple[np.ndarray, list[str]]:
    from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
    names = [
        "MolWt", "MolLogP", "TPSA", "HDonors", "HAcceptors",
        "RotatableBonds", "RingCount", "AromaticRings", "AliphaticRings",
        "FractionCSP3", "HeavyAtomCount", "NHOHCount", "NOCount", "LabuteASA",
    ]
    rows = []
    for mol in mols:
        rows.append([
            Descriptors.MolWt(mol), Crippen.MolLogP(mol), rdMolDescriptors.CalcTPSA(mol),
            Lipinski.NumHDonors(mol), Lipinski.NumHAcceptors(mol),
            Lipinski.NumRotatableBonds(mol), Lipinski.RingCount(mol),
            Lipinski.NumAromaticRings(mol), Lipinski.NumAliphaticRings(mol),
            rdMolDescriptors.CalcFractionCSP3(mol), Lipinski.HeavyAtomCount(mol),
            Lipinski.NHOHCount(mol), Lipinski.NOCount(mol), rdMolDescriptors.CalcLabuteASA(mol),
        ])
    return np.asarray(rows, dtype=np.float32), names


def _fit_eval(
    x: np.ndarray,
    y: np.ndarray,
    fit: np.ndarray,
    held: np.ndarray,
    c: float,
    standardize: bool,
) -> dict[str, Any]:
    if standardize:
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=c, class_weight="balanced", solver="liblinear", max_iter=3000, random_state=0),
        )
    else:
        model = LogisticRegression(
            C=c, class_weight="balanced", solver="liblinear", max_iter=3000, random_state=0
        )
    model.fit(x[fit], y[fit])
    fit_prob = model.predict_proba(x[fit])[:, 1]
    held_prob = model.predict_proba(x[held])[:, 1]
    n_features = int(x.shape[1])
    return {
        "trainable_parameters": n_features + 1,
        "fit_auc": float(roc_auc_score(y[fit], fit_prob)),
        "heldout_auc": float(roc_auc_score(y[held], held_prob)),
        "heldout_indices": held.tolist(),
        "heldout_y": y[held].astype(int).tolist(),
        "heldout_probabilities": held_prob.tolist(),
        "heldout_probability_sha256": _sha256(held_prob),
    }



def _fit_eval_tanimoto_svm(
    fingerprints: np.ndarray,
    y: np.ndarray,
    fit: np.ndarray,
    held: np.ndarray,
    c: float,
) -> dict[str, Any]:
    train_rows = np.asarray(fingerprints[fit], dtype=np.float32)
    held_rows = np.asarray(fingerprints[held], dtype=np.float32)
    train_kernel = _tanimoto_matrix(train_rows, train_rows)
    held_kernel = _tanimoto_matrix(held_rows, train_rows)
    model = SVC(
        C=c, kernel="precomputed", class_weight="balanced",
        probability=False, shrinking=True, cache_size=1024, random_state=0,
    )
    model.fit(train_kernel, y[fit])
    fit_scores = model.decision_function(train_kernel)
    held_scores = model.decision_function(held_kernel)
    return {
        "trainable_parameters": int(len(model.support_) + 1),
        "n_support_vectors": int(len(model.support_)),
        "fit_auc": float(roc_auc_score(y[fit], fit_scores)),
        "heldout_auc": float(roc_auc_score(y[held], held_scores)),
        "heldout_indices": held.tolist(),
        "heldout_y": y[held].astype(int).tolist(),
        "heldout_probabilities": (1.0 / (1.0 + np.exp(-np.clip(held_scores, -30.0, 30.0)))).tolist(),
        "heldout_scores": held_scores.tolist(),
        "heldout_score_sha256": _sha256(held_scores),
    }

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold-cache", default="tracks/ksvd/results/molhiv/molhiv_n8000_scaffold_folds3_seed20260726.npz")
    ap.add_argument("--max-graphs", type=int, default=8000)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--bits", type=int, default=2048)
    ap.add_argument("--positive-prototypes", type=int, default=32)
    ap.add_argument("--skip-svm", action="store_true")
    ap.add_argument("--skip-prototypes", action="store_true")
    ap.add_argument("--negative-prototypes", type=int, default=32)
    ap.add_argument("--output", default="tracks/ksvd/results/molhiv/chemical_fingerprint_probe_20260729.json")
    args = ap.parse_args()

    t0 = time.time()
    repo = Path(__file__).resolve().parents[3]
    max_graphs = None if args.max_graphs <= 0 else args.max_graphs
    bundle = load_molhiv(repo / "data" / "ogb", max_graphs=max_graphs, seed=args.data_seed, with_features=False)
    y = np.asarray(bundle.y, dtype=np.int64)
    original_indices = np.asarray(bundle.meta["original_indices"], dtype=np.int64)
    with np.load(args.fold_cache, allow_pickle=False) as z:
        if not np.array_equal(original_indices, np.asarray(z["original_indices"], dtype=np.int64)):
            raise ValueError("fold cache original index mismatch")
        folds = [
            (np.asarray(z[f"fold_{fold}_train_indices"], dtype=np.int64), np.asarray(z[f"fold_{fold}_valid_indices"], dtype=np.int64))
            for fold in range(3)
        ]
    allowed = np.asarray(bundle.split["train"], dtype=np.int64)
    forbidden = np.concatenate([bundle.split["valid"], bundle.split["test"]]).astype(np.int64)
    if any(set(fit.tolist()) | set(held.tolist()) != set(allowed.tolist()) for fit, held in folds):
        raise ValueError("internal folds must partition official train")

    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    mapping = pd.read_csv(repo / "data" / "ogb" / "ogbg_molhiv" / "mapping" / "mol.csv.gz")
    smiles = mapping.iloc[original_indices]["smiles"].astype(str).tolist()
    mols: list[Any | None] = [None] * len(smiles)
    fallback_unsanitized = []
    for i in allowed.tolist():
        smiles_i = smiles[int(i)]
        mol = Chem.MolFromSmiles(smiles_i)
        if mol is None:
            mol = Chem.MolFromSmiles(smiles_i, sanitize=False)
            if mol is not None:
                mol.UpdatePropertyCache(strict=False)
                Chem.GetSymmSSSR(mol)
                fallback_unsanitized.append(int(i))
        if mol is None:
            raise RuntimeError(f"RDKit failed to parse official-train molecule local={i} original={original_indices[int(i)]}")
        mols[int(i)] = mol

    fingerprints: dict[str, np.ndarray] = {}
    for radius in (2, 3):
        generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=args.bits, includeChirality=True
        )
        matrix = np.zeros((len(smiles), args.bits), dtype=np.uint8)
        for i in allowed.tolist():
            matrix[int(i)] = generator.GetFingerprintAsNumPy(mols[int(i)]).astype(np.uint8)
        fingerprints[f"morgan_r{radius}"] = matrix
    descriptor_allowed, descriptor_names = _descriptor_matrix([mols[int(i)] for i in allowed.tolist()])
    descriptors = np.zeros((len(smiles), descriptor_allowed.shape[1]), dtype=np.float32)
    descriptors[allowed] = descriptor_allowed
    for key in fingerprints:
        if np.any(fingerprints[key][forbidden]):
            raise AssertionError("official-valid/test fingerprints must remain zero")
    if np.any(descriptors[forbidden]):
        raise AssertionError("official-valid/test descriptor rows must remain zero")

    out: dict[str, Any] = {
        "protocol_id": "molhiv-chemical-fragment-fingerprint-scaffold-probe-v1",
        "date": "2026-07-29",
        "scope": "8000-graph development subset; supervised fitting/evaluation only on official-train internal scaffold folds",
        "official_valid_evaluations": 0,
        "official_test_evaluations": 0,
        "note": "only official-train molecules were parsed/fingerprinted; official-valid/test feature rows remain exactly zero",
        "config": vars(args),
        "descriptor_names": descriptor_names,
        "unsanitized_fallback_local_indices": fallback_unsanitized,
        "unsanitized_fallback_original_indices": original_indices[np.asarray(fallback_unsanitized, dtype=np.int64)].tolist(),
        "fingerprint_hashes_official_train": {key: _sha256(value[allowed]) for key, value in fingerprints.items()},
        "folds": [],
    }

    for fold, (fit, held) in enumerate(folds):
        fold_out: dict[str, Any] = {
            "fold": fold,
            "n_fit": int(len(fit)), "n_heldout": int(len(held)),
            "n_fit_positive": int(y[fit].sum()), "n_heldout_positive": int(y[held].sum()),
            "controls": {},
        }
        for radius in (2, 3):
            x = fingerprints[f"morgan_r{radius}"].astype(np.float32)
            for c in (0.01, 0.1, 1.0):
                name = f"morgan_r{radius}_logreg_c{c:g}"
                fold_out["controls"][name] = _fit_eval(x, y, fit, held, c, False)
            x_desc = np.concatenate([x, descriptors], axis=1)
            fold_out["controls"][f"morgan_r{radius}_descriptors_logreg_c0.1"] = _fit_eval(
                x_desc, y, fit, held, 0.1, True
            )
            if not args.skip_svm:
                for c in (0.1, 1.0, 10.0):
                    fold_out["controls"][f"morgan_r{radius}_tanimoto_svm_c{c:g}"] = _fit_eval_tanimoto_svm(
                        fingerprints[f"morgan_r{radius}"], y, fit, held, c
                    )

        if not args.skip_prototypes:
            base = fingerprints["morgan_r2"]
            pos_candidates = fit[y[fit] == 1]
            neg_candidates = fit[y[fit] == 0]
            pos_idx = _select_farthest_tanimoto(base, pos_candidates, args.positive_prototypes)
            neg_idx = _select_farthest_tanimoto(base, neg_candidates, args.negative_prototypes)
            proto_idx = np.concatenate([pos_idx, neg_idx])
            sim_pos = _tanimoto_matrix(base, base[pos_idx])
            sim_neg = _tanimoto_matrix(base, base[neg_idx])
            sim_identity = np.concatenate([sim_pos, sim_neg, descriptors], axis=1)
            sim_summary = np.concatenate([
                _summary_similarity(sim_pos), _summary_similarity(sim_neg), descriptors
            ], axis=1)
            fold_out["prototype_indices"] = {
                "positive_local_indices": pos_idx.tolist(),
                "negative_local_indices": neg_idx.tolist(),
                "positive_original_indices": original_indices[pos_idx].tolist(),
                "negative_original_indices": original_indices[neg_idx].tolist(),
                "prototype_fingerprint_sha256": _sha256(base[proto_idx]),
            }
            for c in (0.01, 0.1, 1.0):
                fold_out["controls"][f"labelmatched_similarity_identity_c{c:g}"] = _fit_eval(
                    sim_identity, y, fit, held, c, True
                )
                fold_out["controls"][f"labelmatched_similarity_summary_c{c:g}"] = _fit_eval(
                    sim_summary, y, fit, held, c, True
                )
        out["folds"].append(fold_out)
        best = max(fold_out["controls"].items(), key=lambda kv: kv[1]["heldout_auc"])
        print(f"fold {fold}: best {best[0]} heldout={best[1]['heldout_auc']:.6f}", flush=True)

    names = sorted(out["folds"][0]["controls"])
    out["aggregate"] = {}
    for name in names:
        values = [float(fold["controls"][name]["heldout_auc"]) for fold in out["folds"]]
        fit_values = [float(fold["controls"][name]["fit_auc"]) for fold in out["folds"]]
        out["aggregate"][name] = {
            "heldout_auc_by_fold": values,
            "heldout_auc_mean": float(np.mean(values)),
            "heldout_auc_std": float(np.std(values, ddof=1)),
            "fit_auc_mean": float(np.mean(fit_values)),
            "mean_fit_minus_heldout": float(np.mean(np.asarray(fit_values) - np.asarray(values))),
        }
    ranking = sorted(out["aggregate"].items(), key=lambda kv: kv[1]["heldout_auc_mean"], reverse=True)
    out["ranking"] = [name for name, _ in ranking]
    out["elapsed_sec"] = time.time() - t0
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2) + "\n")
    print("aggregate")
    for name, row in ranking:
        print(f"{name:48s} mean={row['heldout_auc_mean']:.6f} folds={row['heldout_auc_by_fold']}")
    print(f"wrote {output} elapsed={out['elapsed_sec']:.1f}s")


if __name__ == "__main__":
    main()
