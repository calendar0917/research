"""Freeze the already-aligned P3B / L / DBT per-molecule predictions.

The official test split is already exposed in this repository.  This script only
copies old, frozen predictions into a small CSV so the new round can compute
paired descriptive contrasts without depending on git-ignored ``.npz`` files on
the compute host.  It does not train, select or modify any model.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
_EXP = _HERE.parent
_SSUF = _EXP.parent
_V1 = _SSUF / "zinc_arch_discovery_v1" / "code"
_DBT = _SSUF / "zinc_directed_bond_transformer_v1" / "code"
for _p in (str(_HERE), str(_V1), str(_DBT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402

ANALYSIS = _EXP / "analysis"
AUDIT = _SSUF / "zinc_generalization_audit_v1" / "analysis"
DCC = _SSUF / "zinc_dual_channel_composer_v1" / "analysis"
DBT_RUN = _SSUF / "zinc_directed_bond_transformer_v1" / "runs" / "dbt_s0" / "predictions"

EXPECTED = {
    "valid": {"P3B_s0": 0.07622375367576023, "L_s0": 0.078537152,
              "DBT_s0": 0.077768402},
    "test": {"P3B_s0": 0.069864841, "L_s0": 0.066266555, "DBT_s0": 0.069401494},
}


def _rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _npz(path: Path) -> dict[str, np.ndarray]:
    d = np.load(path)
    return {k: d[k] for k in d.files}


def _check_rows(rows: list[int], name: str, n: int = 1000) -> None:
    got = np.asarray(rows, dtype=np.int64)
    if got.shape != (n,) or not np.array_equal(got, np.arange(n)):
        raise RuntimeError(f"{name}: row ids are not 0..{n - 1}")


def _mae(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.abs(pred.astype(np.float64) - y.astype(np.float64)).mean())


def build_valid() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    p3b_rows = _rows(AUDIT / "per_molecule.csv")
    l_rows = _rows(DCC / "per_molecule_dual.csv")
    if len(p3b_rows) != 1000 or len(l_rows) != 1000:
        raise RuntimeError("valid reference tables are not 1000 rows")
    l_by_row = {int(r["row_id"]): r for r in l_rows}
    dbt = _npz(DBT_RUN / "valid_best.npz")
    _check_rows(dbt["row"].tolist(), "DBT valid")
    out = []
    for i, r in enumerate(p3b_rows):
        row = int(r["row_id"])
        if row != i:
            raise RuntimeError(f"audit valid row order broken at {i}: {row}")
        lr = l_by_row.get(row)
        if lr is None:
            raise RuntimeError(f"L valid missing row {row}")
        y = float(r["y_raw"])
        p3b = float(r["pred_P3B_s0"])
        l = float(lr["pred_L_s0"])
        d = float(dbt["pred"][i])
        unseen_frac = float(r["r2_unseen_atom_frac"])
        if abs(float(lr["y_raw"]) - y) > 1e-6:
            raise RuntimeError(f"valid y mismatch at row {row}")
        out.append({
            "row": row, "y_raw": y,
            "pred_P3B_s0": p3b, "pred_L_s0": l, "pred_DBT_s0": d,
            "r2_unseen_atom_frac": unseen_frac,
            "signature_unseen": int(unseen_frac > 0),
            "ordinary_y_gt_minus3": int(y > -3.0),
            "tail_y_le_minus3": int(y <= -3.0),
        })
    y = np.asarray([r["y_raw"] for r in out])
    meta = {"rows": len(out), "mae": {}, "expected": EXPECTED["valid"]}
    for name in ("P3B_s0", "L_s0", "DBT_s0"):
        pred = np.asarray([r[f"pred_{name}"] for r in out])
        meta["mae"][name] = _mae(pred, y)
        if abs(meta["mae"][name] - EXPECTED["valid"][name]) > 1e-6:
            raise RuntimeError(f"valid MAE for {name} does not reproduce: {meta['mae'][name]}")
    return out, meta


def build_test() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base_rows = _rows(DCC / "test_per_molecule.csv")
    if len(base_rows) != 1000:
        raise RuntimeError("test reference table is not 1000 rows")
    dbt = _npz(DBT_RUN / "test_best.npz")
    _check_rows(dbt["row"].tolist(), "DBT test")
    p3b_npz = _npz(DCC / "test_preds" / "P3B_s0_test.npz")
    l_npz = _npz(DCC / "test_preds" / "L_s0_test.npz")
    out = []
    for i, r in enumerate(base_rows):
        row = int(r["row_id"])
        if row != i:
            raise RuntimeError(f"test reference row order broken at {i}: {row}")
        y = float(r["y_raw"])
        if abs(float(p3b_npz["y"][i]) - y) > 1e-6 or abs(float(l_npz["y"][i]) - y) > 1e-6:
            raise RuntimeError(f"test y mismatch at row {row}")
        unseen_frac = float(r["r2_unseen_atom_frac"])
        out.append({
            "row": row, "y_raw": y,
            "pred_P3B_s0": float(r["pred_P3B_s0"]),
            "pred_L_s0": float(r["pred_L_s0"]),
            "pred_DBT_s0": float(dbt["pred"][i]),
            "r2_unseen_atom_frac": unseen_frac,
            "signature_unseen": int(r["signature_unseen"]),
            "ordinary_y_gt_minus3": int(r["ordinary_y_gt_minus3"]),
            "tail_y_le_minus3": int(r["tail_y_le_minus3"]),
        })
    y = np.asarray([r["y_raw"] for r in out])
    meta = {"rows": len(out), "mae": {}, "expected": EXPECTED["test"]}
    for name in ("P3B_s0", "L_s0", "DBT_s0"):
        pred = np.asarray([r[f"pred_{name}"] for r in out])
        meta["mae"][name] = _mae(pred, y)
        if abs(meta["mae"][name] - EXPECTED["test"][name]) > 1e-6:
            raise RuntimeError(f"test MAE for {name} does not reproduce: {meta['mae'][name]}")
    return out, meta


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    valid, vmeta = build_valid()
    test, tmeta = build_test()
    _write(ANALYSIS / "reference_preds_valid.csv", valid)
    _write(ANALYSIS / "reference_preds_test.csv", test)
    meta = {
        "purpose": "frozen old per-molecule predictions reused for descriptive contrasts",
        "official_test_historically_exposed": True,
        "new_model_selection_used": False,
        "sources": {
            "valid": {
                "P3B": str((AUDIT / "per_molecule.csv").relative_to(C.REPO)),
                "L": str((DCC / "per_molecule_dual.csv").relative_to(C.REPO)),
                "DBT": str((DBT_RUN / "valid_best.npz").relative_to(C.REPO)),
            },
            "test": {
                "groups_and_P3B_L": str((DCC / "test_per_molecule.csv").relative_to(C.REPO)),
                "DBT": str((DBT_RUN / "test_best.npz").relative_to(C.REPO)),
                "P3B_npz": str((DCC / "test_preds" / "P3B_s0_test.npz").relative_to(C.REPO)),
                "L_npz": str((DCC / "test_preds" / "L_s0_test.npz").relative_to(C.REPO)),
            },
        },
        "valid": vmeta,
        "test": tmeta,
        "row_alignment": "row ids are exactly 0..999 and every y column agrees",
        "group_definition": "signature_unseen = any atom with a radius-2 content key absent from train; tail = y <= -3; labels are analysis-only",
    }
    C.write_json(ANALYSIS / "reference_preds_meta.json", meta)
    C.log(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
