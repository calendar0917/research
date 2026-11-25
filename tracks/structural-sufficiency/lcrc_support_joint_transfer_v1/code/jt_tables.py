"""Merge the two analysis products into the reported main tables."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

ROUND = Path(__file__).resolve().parents[1]
ANALYSIS = ROUND / "analysis"


def read(name: str) -> Any:
    return json.loads((ANALYSIS / name).read_text())


def write(name: str, obj: Any) -> None:
    (ANALYSIS / name).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def main() -> None:
    lipo = read("lipo_analysis.json")
    pcba = read("pcba_analysis.json")

    lipo_rows = []
    for r in lipo["main_table"]:
        lipo_rows.append(
            {
                "family": "lipo",
                "arm": r["arm"],
                "parameters": r["parameters"],
                "best_epoch": r["best_epoch"],
                "valid_metric": "RMSE",
                "test_metric": "RMSE",
                "valid": r["valid_rmse_best"],
                "test": r["test_rmse_best"],
                "valid_final": r["valid_rmse_final"],
                "test_final": r["test_rmse_final"],
                "valid_last20": r["valid_rmse_last20"],
                "test_last20": r["test_rmse_last20"],
            }
        )
    base_valid = next(r["valid"] for r in lipo_rows if r["arm"] == "C1")
    base_test = next(r["test"] for r in lipo_rows if r["arm"] == "C1")
    for r in lipo_rows:
        if r["arm"] != "C1":
            r["delta_valid_vs_C1"] = r["valid"] - base_valid
            r["delta_test_vs_C1"] = r["test"] - base_test
            r["relative_delta_test_vs_C1"] = (r["test"] - base_test) / base_test
    lipo_rows.append(
        {
            "family": "lipo",
            "arm": "GINE_reference",
            "parameters": None,
            "best_epoch": None,
            "valid_metric": "RMSE",
            "test_metric": "RMSE",
            "valid": 0.6091,
            "test": 0.6456,
            "valid_final": None,
            "test_final": None,
            "valid_last20": None,
            "test_last20": None,
            "note": "descriptive external reference, not retrained",
        }
    )

    pcba_rows = []
    for r in pcba["main_table"]:
        pcba_rows.append(
            {
                "family": "pcba",
                "arm": r["arm"],
                "parameters": r["parameters"],
                "best_epoch": r["best_epoch"],
                "valid_metric": "AP",
                "test_metric": "AP",
                "valid": r["valid_ap"],
                "test": r["test_ap"],
                "valid_final": r.get("valid_ap_final"),
                "test_final": r.get("test_ap_final"),
                "valid_last20": r.get("valid_ap_last20_prefix"),
                "test_last20": r.get("test_ap_last20"),
                "valid_at_horizon": r.get("valid_ap_at_horizon"),
                "delta_valid_vs_original": r.get("delta_valid_ap_vs_original"),
                "delta_test_vs_original": r.get("delta_test_ap_vs_original"),
            }
        )
    pcba_rows.append(
        {
            "family": "pcba",
            "arm": "DYNAMIC_reference",
            "parameters": 1128468,
            "best_epoch": None,
            "valid_metric": "AP",
            "test_metric": "AP",
            "valid": 0.275067,
            "test": 0.262226,
            "valid_final": None,
            "test_final": None,
            "valid_last20": None,
            "test_last20": None,
            "note": "descriptive reference, not retrained",
        }
    )

    merged = {
        "schema": "lcrc-support-joint-transfer-main-table-v1",
        "round_id": "lcrc-support-joint-transfer-v1",
        "lipo": lipo_rows,
        "pcba": pcba_rows,
        "pcba_interaction": pcba["interaction"],
        "pcba_macro_bootstrap": pcba["macro_bootstrap"],
        "lipo_paired": lipo["paired"],
    }
    write("main_table.json", merged)
    fields = [
        "family",
        "arm",
        "parameters",
        "best_epoch",
        "valid_metric",
        "test_metric",
        "valid",
        "test",
        "valid_final",
        "test_final",
        "valid_last20",
        "test_last20",
        "valid_at_horizon",
        "delta_valid_vs_C1",
        "delta_test_vs_C1",
        "relative_delta_test_vs_C1",
        "delta_valid_vs_original",
        "delta_test_vs_original",
        "note",
    ]
    with (ANALYSIS / "main_table.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in lipo_rows + pcba_rows:
            w.writerow(row)
    write("decomposition.json", {"lipo": lipo["decomposition"], "pcba": pcba["decomposition"]})
    write("intervention.json", lipo["intervention"])
    print(json.dumps({"lipo": len(lipo_rows), "pcba": len(pcba_rows)}, indent=2))


if __name__ == "__main__":
    main()
