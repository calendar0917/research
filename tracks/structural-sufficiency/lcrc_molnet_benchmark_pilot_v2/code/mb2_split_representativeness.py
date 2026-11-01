"""Post-hoc diagnostic: is the Validation split a usable proxy for Test?

Both models (LCRC and the matched baseline) drop sharply from Validation to Test
on BBBP.  Before attributing that to either model, this module measures the split
itself: the class prior and a *single trivial descriptor* (molecular weight) per
split.  If molecular weight alone separates the Validation classes far better than
it separates the Test classes, the Validation split is not a small version of the
Test split, and no model-choice rule based on it can be trusted.

Computed after the one-shot Test read (labels are already public), so it changes
no protocol decision.  Writes ``analysis/split_representativeness.json``.

Usage
-----
    uv run python .../code/mb2_split_representativeness.py
"""

from __future__ import annotations

import argparse
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score

import mb2_common as C
import mb2_source as S


def mol_weight(smiles: str) -> float:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    return float(Descriptors.MolWt(mol)) if mol is not None else float("nan")


def _trivial_signal(task: str, y: np.ndarray, mw: np.ndarray) -> dict[str, Any]:
    """How well does MW alone rank the labels?  AUC for classification, |r| otherwise."""
    if task == "regression":
        ok = np.isfinite(mw) & np.isfinite(y.reshape(-1))
        r = float(np.corrcoef(mw[ok], y.reshape(-1)[ok])[0, 1]) if ok.sum() > 2 else float("nan")
        return {"kind": "pearson_r", "value": r, "abs_value": abs(r)}
    y2 = y.reshape(y.shape[0], -1)
    per_task = []
    for j in range(y2.shape[1]):
        mask = np.isfinite(y2[:, j]) & np.isfinite(mw)
        if mask.sum() > 2 and len(np.unique(y2[mask, j])) == 2:
            per_task.append(float(roc_auc_score(y2[mask, j], mw[mask])))
    value = float(np.mean(per_task)) if per_task else float("nan")
    return {
        "kind": "roc_auc_of_MW",
        "value": value,
        # a trivial descriptor is *informative* if it is far from 0.5 in either direction
        "abs_value": abs(value - 0.5) * 2.0,
        "n_tasks_scored": len(per_task),
    }


def audit(dataset: str) -> dict[str, Any]:
    data = S.load_dataset(dataset, labels="all")
    task = C.dataset_entry(dataset)["task"]
    mw = np.array([mol_weight(s) for s in data["smiles"]], dtype=np.float64)
    out: dict[str, Any] = {"dataset": dataset, "task": task, "splits": {}}
    for split in ("train", "valid", "test"):
        rows = np.asarray(data["rows"][split], dtype=np.int64)
        y = np.asarray(data["y"][split], dtype=np.float64)
        entry: dict[str, Any] = {
            "n": int(len(rows)),
            "n_labelled_cells": int(np.isfinite(y).sum()),
            "mean_molecular_weight": float(np.nanmean(mw[rows])),
            "median_molecular_weight": float(np.nanmedian(mw[rows])),
        }
        if task == "regression":
            entry["label_mean"] = float(np.nanmean(y))
            entry["label_std"] = float(np.nanstd(y))
        else:
            entry["positive_rate"] = float(np.nanmean(y))
        entry["trivial_descriptor_signal"] = _trivial_signal(task, y, mw[rows])
        out["splits"][split] = entry
    v = out["splits"]["valid"]["trivial_descriptor_signal"]["abs_value"]
    t = out["splits"]["test"]["trivial_descriptor_signal"]["abs_value"]
    out["trivial_signal_valid_minus_test"] = float(v - t)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validation-vs-Test representativeness diagnostic")
    parser.add_argument("--out", default=str(C.ANALYSIS / "split_representativeness.json"))
    args = parser.parse_args(argv)
    rows = [audit(d) for d in C.DATASET_ORDER]
    payload = {
        "schema": "lcrc-molnet-v2-split-representativeness-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "purpose": (
            "a Validation split is only a model-selection proxy if its label/size structure resembles the "
            "Test split; a single trivial descriptor (molecular weight) is used as the yardstick"
        ),
        "computed_after": "the one-shot Test read (no protocol decision depends on it)",
        "datasets": {r["dataset"]: r for r in rows},
    }
    C.write_json(args.out, payload)
    for r in rows:
        s = r["splits"]
        print(
            f"{r['dataset']:14s} pos/label tr/va/te = "
            f"{s['train'].get('positive_rate', s['train'].get('label_mean')):.3f}/"
            f"{s['valid'].get('positive_rate', s['valid'].get('label_mean')):.3f}/"
            f"{s['test'].get('positive_rate', s['test'].get('label_mean')):.3f}  "
            f"MW tr/va/te = {s['train']['mean_molecular_weight']:.0f}/"
            f"{s['valid']['mean_molecular_weight']:.0f}/{s['test']['mean_molecular_weight']:.0f}  "
            f"|signal| va={s['valid']['trivial_descriptor_signal']['abs_value']:.2f} "
            f"te={s['test']['trivial_descriptor_signal']['abs_value']:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
