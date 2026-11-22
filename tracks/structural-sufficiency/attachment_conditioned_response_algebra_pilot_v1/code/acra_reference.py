"""Light non-GNN reference: Morgan radius-2 count fingerprint + Ridge(alpha=10).

Fit on Train only, evaluated on Valid only.  Not part of ACRA, not used for any
architecture decision.
"""

from __future__ import annotations

import argparse
import time
from typing import Any, Mapping, Sequence

import numpy as np

import acra_common as C
import acra_data as D


def morgan_counts(smiles: Sequence[str], radius: int, n_bits: int, log=print):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=int(radius), fpSize=int(n_bits))
    out = np.zeros((len(smiles), int(n_bits)), np.float32)
    failed = 0
    for i, text in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(text))
        if mol is None:
            failed += 1
            continue
        fp = gen.GetCountFingerprint(mol)
        for idx, value in fp.GetNonzeroElements().items():
            out[i, int(idx)] = float(value)
    log(f"  morgan r={radius} bits={n_bits}: {len(smiles)} molecules, parse failures={failed}, "
        f"mean nnz={float((out > 0).sum(axis=1).mean()):.1f}")
    return out, failed


def run(cfg: Mapping[str, Any], log=print) -> dict[str, Any]:
    from sklearn.linear_model import Ridge

    ref = cfg["reference"]
    raw = D.load_raw(cfg)
    splits = D.official_splits(raw)
    labels = np.asarray(raw["labels"], np.float64)
    smiles = np.asarray(raw["smiles"])
    t0 = time.time()
    X, failed = morgan_counts(smiles, int(ref["morgan_radius"]), int(ref["morgan_bits"]), log)
    train, valid = splits["train"], splits["valid"]
    model = Ridge(alpha=float(ref["ridge_alpha"]))
    model.fit(X[train], labels[train])
    pred_train = model.predict(X[train])
    pred_valid = model.predict(X[valid])
    out = {
        "protocol_id": cfg["protocol_id"],
        "generated_utc": C.now_utc(),
        "features": {
            "type": "Morgan count fingerprint",
            "radius": int(ref["morgan_radius"]),
            "n_bits": int(ref["morgan_bits"]),
            "parse_failures": int(failed),
        },
        "regression": {"type": "Ridge", "alpha": float(ref["ridge_alpha"]),
                       "fit_on": "official Train only"},
        "n_train": int(train.size),
        "n_valid": int(valid.size),
        "train_rmse": C.rmse(pred_train, labels[train]),
        "train_mae": C.mae(pred_train, labels[train]),
        "valid_rmse": C.rmse(pred_valid, labels[valid]),
        "valid_mae": C.mae(pred_valid, labels[valid]),
        "wall_seconds": float(time.time() - t0),
        "test_labels_read": False,
    }
    path = C.analysis_dir() / "reference_morgan_ridge.json"
    C.write_json(path, out)
    np.savez(C.analysis_dir() / "reference_morgan_ridge_pred.npz",
             valid_row_id=np.asarray(raw["row_ids"])[valid],
             valid_label=labels[valid], valid_pred=pred_valid)
    log(f"  morgan+ridge valid rmse={out['valid_rmse']:.6f} mae={out['valid_mae']:.6f} "
        f"({out['wall_seconds']:.1f}s)")
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="")
    args = ap.parse_args(argv)

    def log(msg: str) -> None:
        print(msg, flush=True)
        if args.log:
            with open(args.log, "a", encoding="utf-8") as handle:
                handle.write(msg + "\n")

    run(C.load_config(), log=log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())