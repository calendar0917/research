#!/usr/bin/env python
"""Assemble the round's deliverable files from the run outputs.

* ``curves/<row>__<target>__<split>.json``   per-epoch curves of every unit
* ``predictions/dev_molecule_predictions.csv`` tidy Dev-504 predictions
* ``predictions/esol_valid_predictions.csv``    phase-C Valid-113 predictions
* ``resource_ledger.json``                  every accounting entry, merged
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import sr_common as C

DELIVER_CURVES = C.ROUND_DIR / "curves"
DELIVER_PRED = C.ROUND_DIR / "predictions"


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})
    print(f"wrote {path.relative_to(C.ROUND_DIR)} ({len(rows)} rows)")


def curves() -> None:
    DELIVER_CURVES.mkdir(parents=True, exist_ok=True)
    n = 0
    for phase, base in (("B", C.RESULTS_DIR / "B"), ("P", C.RESULTS_DIR / "P")):
        if not base.is_dir():
            continue
        for path in sorted(base.glob("*/*/curves.json")):
            row, target = path.parent.parent.name, path.parent.name
            payload = C.read_json(path)
            C.write_json(DELIVER_CURVES / f"{phase}__{row}__{target}.json", payload)
            n += 1
    if (C.RESULTS_DIR / "A").is_dir():
        for path in sorted((C.RESULTS_DIR / "A").glob("*/curves.json")):
            payload = C.read_json(path)
            C.write_json(DELIVER_CURVES / f"A__{path.parent.name}.json", payload)
            n += 1
    for path in sorted((C.RESULTS_DIR / "C").glob("*/curves.json")) if (C.RESULTS_DIR / "C").is_dir() else []:
        payload = C.read_json(path)
        C.write_json(DELIVER_CURVES / f"C__{path.parent.name}__BASE.json", payload)
        n += 1
    logs = sorted(C.RESULTS_DIR.rglob("*.log"))
    for path in logs:
        text = path.read_text(encoding="utf-8", errors="replace")
        (DELIVER_CURVES.parent / "logs").mkdir(parents=True, exist_ok=True)
        (DELIVER_CURVES.parent / "logs" / path.name).write_text(text, encoding="utf-8")
    print(f"copied {n} curve files and {len(logs)} logs")


def dev_predictions() -> None:
    rows: list[dict] = []
    truth: dict[int, float] = {}
    for path in sorted((C.RESULTS_DIR / "B").glob("*/*/predictions.npz")):
        row, target = path.parent.parent.name, path.parent.name
        with np.load(path) as handle:
            ids = np.asarray(handle["dev_row_ids"], dtype=np.int64)
            y = np.asarray(handle["dev_y_true"], dtype=np.float64)
            best = np.asarray(handle["best_dev_y_pred"], dtype=np.float64)
            final = np.asarray(handle["final_dev_y_pred"], dtype=np.float64)
        for i, r in enumerate(ids.tolist()):
            truth.setdefault(int(r), float(y[i]))
        for ckpt, series in (("best", best), ("final", final)):
            if series is None:
                continue
            for i, r in enumerate(ids.tolist()):
                rows.append({"phase": "B", "row": row, "target": target, "checkpoint": ckpt,
                             "row_id": int(r), "y_true": float(y[i]),
                             "y_pred": float(series[i])})
    if rows:
        base = [r for r in rows if r["checkpoint"] == "best"]
        err = np.array([r["y_pred"] - r["y_true"] for r in base])
        print(f"dev predictions: {len(rows)} rows, best-checkpoint RMSE "
              f"{float(np.sqrt((err ** 2).mean())):.6f} (pooled over units)")
        _write_csv(DELIVER_PRED / "dev_molecule_predictions.csv", rows,
                   ["phase", "row", "target", "checkpoint", "row_id", "y_true", "y_pred"])

    esol: list[dict] = []
    for path in sorted((C.RESULTS_DIR / "C").glob("*/predictions.npz")):
        with np.load(path) as handle:
            ids = np.asarray(handle["valid_row_ids"], dtype=np.int64)
            y = np.asarray(handle["valid_y_true"], dtype=np.float64)
            best = np.asarray(handle["best_valid_y_pred"], dtype=np.float64)
            final = np.asarray(handle["final_valid_y_pred"], dtype=np.float64)
        for ckpt, series in (("best", best), ("final", final)):
            for i, r in enumerate(ids.tolist()):
                esol.append({"phase": "C", "row": path.parent.name, "target": "BASE",
                             "checkpoint": ckpt, "row_id": int(r),
                             "y_true": float(y[i]), "y_pred": float(series[i])})
    if esol:
        _write_csv(DELIVER_PRED / "esol_valid_predictions.csv", esol,
                   ["phase", "row", "target", "checkpoint", "row_id", "y_true", "y_pred"])


def edit_pairs() -> None:
    """Tidy per-pair deltas of the frozen 276-pair strict-edit set (best checkpoint)."""
    paths = sorted(C.PRED_DIR.glob("edit_response__*__best.npz"))
    if not paths:
        return
    base: dict[str, np.ndarray] = {}
    columns: list[str] = []
    for path in paths:
        row, target = path.name.split("__")[1], path.name.split("__")[2]
        with np.load(path, allow_pickle=True) as handle:
            keys = ("pair_id", "row_a", "row_b", "component_id", "cell", "transform_id",
                    "core_uid", "core_size", "cut_bond_type", "s_size_a", "s_size_b",
                    "support_other_component", "support_two_components", "delta_y",
                    "delta_hat")
            data = {k: np.asarray(handle[k]) for k in keys}
        sel = data["cell"] == "DD"
        if not base:
            base = {k: data[k][sel] for k in keys if k != "delta_hat"}
            base["n_pairs_all"] = np.asarray([data["pair_id"].size])
        col = f"{row}__{target}"
        base[col] = data["delta_hat"][sel]
        columns.append(col)
    rows = []
    for i in range(base["pair_id"].size):
        out = {k: (float(v[i]) if isinstance(v[i], (np.floating, float)) else
                   (int(v[i]) if isinstance(v[i], (np.integer, int)) else v[i]))
               for k, v in base.items() if k not in ("n_pairs_all",)}
        out.update({c: float(base[c][i]) for c in columns})
        rows.append(out)
    _write_csv(DELIVER_PRED / "edit_pair_predictions.csv", rows,
               ["pair_id", "row_a", "row_b", "component_id", "cell", "transform_id",
                "core_uid", "core_size", "cut_bond_type", "s_size_a", "s_size_b",
                "support_other_component", "support_two_components", "delta_y"] + columns)


def ledger() -> None:
    sources = [C.RESULTS_DIR / "resource_ledger_remote.json",
               C.RESULTS_DIR / "resource_ledger_A_remote.json",
               C.RESULTS_DIR / "resource_ledger.json"]
    entries: list[dict] = []
    seen = set()
    for path in sources:
        if not path.is_file():
            continue
        for e in C.read_json(path).get("entries", []):
            key = (e.get("phase"), e.get("unit"), e.get("finished_at"), e.get("seconds"))
            if key in seen:
                continue
            seen.add(key)
            entries.append(e)
    # runs killed before they could append: bounded by the rr job records
    entries.extend([
        {"phase": "B", "unit": "RESP_PRODUCT/BASE (cancelled)", "device": "cuda:0",
         "seconds": 420.0, "gpu_hours": 420.0 / 3600.0, "status": "cancelled_invalidated",
         "estimated": True, "finished_at": "2026-11-27T01:12:00Z",
         "note": "first-response run, killed after the RESP prediction-index bug was found"},
        {"phase": "B", "unit": "RESP_FUSION_MLP/BASE (cancelled)", "device": "cuda:0",
         "seconds": 400.0, "gpu_hours": 400.0 / 3600.0, "status": "cancelled_invalidated",
         "estimated": True, "finished_at": "2026-11-27T01:12:00Z",
         "note": "first-response run, killed after the RESP prediction-index bug was found"},
    ])
    for e in entries:
        e.setdefault("gpu_hours", 0.0)
    total = float(sum(float(e.get("gpu_hours") or 0.0) for e in entries))
    payload = {
        "round_id": C.ROUND_ID,
        "generated_at": C.utc_stamp(),
        "entries": sorted(entries, key=lambda e: (str(e.get("phase")), str(e.get("unit")),
                                                  str(e.get("finished_at")))),
        "n_entries": len(entries),
        "gpu_hours_total": total,
        "gpu_hours_budget_soft": 14.0,
        "gpu_hours_budget_hard": 16.0,
        "note": "every launched run counts, including the cancelled and invalidated ones",
    }
    C.write_json(C.ROUND_DIR / "resource_ledger.json", payload)
    print(f"ledger: {len(entries)} entries, {total:.3f} GPU-hours")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--what", default="all",
                        choices=["all", "curves", "predictions", "ledger"])
    args = parser.parse_args()
    if args.what in ("all", "curves"):
        curves()
    if args.what in ("all", "predictions"):
        dev_predictions()
        edit_pairs()
    if args.what in ("all", "ledger"):
        ledger()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
