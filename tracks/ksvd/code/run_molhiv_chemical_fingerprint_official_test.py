"""Execute the frozen Morgan official-test evaluation exactly once.

This runner verifies the freeze manifest hash, fits only official train, and
fingerprints only official train/test.  Official valid molecules are not
parsed or fingerprinted.  Because this repository has historical test
exposure, the result is retrospective and must not be used for further model
selection.
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

import sys
_TRACK = Path(__file__).resolve().parents[1]
if str(_TRACK) not in sys.path:
    sys.path.insert(0, str(_TRACK))
from code.data_molhiv import load_molhiv


def sha256(x: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", default="tracks/ksvd/results/molhiv/chemical_fingerprint_official_test_freeze_v1.json")
    ap.add_argument("--expected-freeze-sha256", required=True)
    ap.add_argument("--output", default="tracks/ksvd/results/molhiv/chemical_fingerprint_official_test_20260729.json")
    args = ap.parse_args()
    t0 = time.time()
    freeze_path = Path(args.freeze)
    actual_freeze_hash = hashlib.sha256(freeze_path.read_bytes()).hexdigest()
    if actual_freeze_hash != args.expected_freeze_sha256:
        raise ValueError(f"freeze hash mismatch: expected {args.expected_freeze_sha256}, got {actual_freeze_hash}")
    freeze = json.loads(freeze_path.read_text())
    if freeze["selected_candidate"]["candidate"] != "morgan_r2_c0.01":
        raise ValueError("unexpected frozen candidate")
    cfg = freeze["model"]

    repo = Path(__file__).resolve().parents[3]
    bundle = load_molhiv(repo / "data" / "ogb", max_graphs=None, seed=0, with_features=False)
    y = np.asarray(bundle.y, dtype=np.int64)
    original_indices = np.asarray(bundle.meta["original_indices"], dtype=np.int64)
    train = np.asarray(bundle.split["train"], dtype=np.int64)
    valid = np.asarray(bundle.split["valid"], dtype=np.int64)
    test = np.asarray(bundle.split["test"], dtype=np.int64)
    allowed = np.concatenate([train, test])

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
    if any(mols[int(i)] is not None for i in valid.tolist()):
        raise AssertionError("official valid molecule was parsed during frozen test run")

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=int(cfg["radius"]), fpSize=int(cfg["bits"]), includeChirality=bool(cfg["include_chirality"])
    )
    x = np.zeros((len(smiles), int(cfg["bits"])), dtype=np.uint8)
    for i in allowed.tolist():
        x[int(i)] = generator.GetFingerprintAsNumPy(mols[int(i)]).astype(np.uint8)
    if np.any(x[valid]):
        raise AssertionError("official valid fingerprint rows must remain zero")
    train_fp_hash = sha256(x[train])
    if train_fp_hash != cfg["expected_train_fingerprint_sha256"]:
        raise ValueError("official-train fingerprint does not reproduce frozen valid run")

    model = LogisticRegression(
        C=float(cfg["C"]), class_weight=cfg["class_weight"], solver=cfg["solver"],
        max_iter=int(cfg["max_iter"]), random_state=int(cfg["random_state"]),
    )
    xf = x.astype(np.float32)
    model.fit(xf[train], y[train])
    coefficient_hash = sha256(np.asarray(model.coef_, dtype=np.float64))
    intercept_hash = sha256(np.asarray(model.intercept_, dtype=np.float64))
    if coefficient_hash != cfg["expected_coefficient_sha256"] or intercept_hash != cfg["expected_intercept_sha256"]:
        raise ValueError("fitted classifier does not reproduce frozen official-valid model")
    if np.asarray(model.n_iter_, dtype=int).tolist() != cfg["expected_n_iter"]:
        raise ValueError("solver iteration count mismatch")

    p_fit = model.predict_proba(xf[train])[:, 1]
    p_test = model.predict_proba(xf[test])[:, 1]
    output = {
        "protocol_id": "molhiv-chemical-fingerprint-official-test-v1",
        "date": "2026-07-29",
        "status": "retrospective_controlled_frozen_evaluation_complete",
        "freeze_manifest": str(freeze_path),
        "freeze_manifest_sha256": actual_freeze_hash,
        "selected_candidate": freeze["selected_candidate"],
        "fit_split": "exact official train",
        "evaluation_split": "exact official test",
        "official_valid_evaluations_in_this_runner": 0,
        "official_test_evaluations_in_this_runner": 1,
        "historical_test_disclosure": freeze["disclosure"],
        "valid_isolation": "official valid molecules not parsed; valid fingerprint rows exactly zero",
        "n_train": int(len(train)), "n_train_positive": int(y[train].sum()),
        "n_test": int(len(test)), "n_test_positive": int(y[test].sum()),
        "fit_auc": float(roc_auc_score(y[train], p_fit)),
        "official_test_auc": float(roc_auc_score(y[test], p_test)),
        "test_indices": test.astype(int).tolist(),
        "test_y": y[test].astype(int).tolist(),
        "test_probabilities": p_test.tolist(),
        "test_probability_sha256": sha256(p_test),
        "train_fingerprint_sha256": train_fp_hash,
        "test_fingerprint_sha256": sha256(x[test]),
        "coefficient_sha256": coefficient_hash,
        "intercept_sha256": intercept_hash,
        "n_iter": np.asarray(model.n_iter_, dtype=int).tolist(),
        "fallback_unsanitized_local_indices": fallback,
        "elapsed_sec": time.time() - t0,
    }
    Path(args.output).write_text(json.dumps(output, indent=2))
    print(json.dumps({
        "output": args.output,
        "freeze_manifest_sha256": actual_freeze_hash,
        "fit_auc": output["fit_auc"],
        "official_valid_auc_frozen_selection": freeze["selected_candidate"]["official_valid_auc"],
        "official_test_auc": output["official_test_auc"],
        "elapsed_sec": output["elapsed_sec"],
    }, indent=2))


if __name__ == "__main__":
    main()
