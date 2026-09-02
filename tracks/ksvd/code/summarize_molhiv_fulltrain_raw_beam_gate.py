"""Validate the frozen full-train RAW Beam8 stage-A gate.

The runner writes model metrics but deliberately does not make a promotion
decision. Keeping this checker separate makes the locked comparison and its
isolation requirements inspectable after a long external training run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_VARIANTS = (
    "cin",
    "cin_beam8_bond",
    "cin_beam8_bond_shuffled",
    "cin_beam8_bond_bag",
    "cin_beam8_bond_no_patch",
)


def _metric(report: dict[str, Any], variant: str, key: str) -> float:
    try:
        return float(report["results"][variant]["heldout"][key])
    except KeyError as exc:
        raise ValueError(f"missing held-out {key} for {variant}") from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = json.loads(args.input.read_text(encoding="utf-8"))
    variants = tuple(report.get("config", {}).get("variants", ()))
    missing = sorted(set(REQUIRED_VARIANTS).difference(variants))
    if missing:
        raise ValueError(f"result is missing required variants: {missing}")
    if report.get("official_valid_evaluations") != 0:
        raise ValueError("official-valid evaluation must remain zero")
    if report.get("official_test_evaluations") != 0:
        raise ValueError("official-test evaluation must remain zero")

    auc = {name: _metric(report, name, "roc_auc") for name in REQUIRED_VARIANTS}
    ap = {
        name: _metric(report, name, "average_precision")
        for name in REQUIRED_VARIANTS
    }
    true_name = "cin_beam8_bond"
    delta_auc = {
        "true_minus_cin": auc[true_name] - auc["cin"],
        "true_minus_shuffled": auc[true_name] - auc["cin_beam8_bond_shuffled"],
        "true_minus_bag": auc[true_name] - auc["cin_beam8_bond_bag"],
        "true_minus_no_patch": auc[true_name] - auc["cin_beam8_bond_no_patch"],
    }
    delta_ap = {
        "true_minus_cin": ap[true_name] - ap["cin"],
        "true_minus_shuffled": ap[true_name] - ap["cin_beam8_bond_shuffled"],
        "true_minus_bag": ap[true_name] - ap["cin_beam8_bond_bag"],
        "true_minus_no_patch": ap[true_name] - ap["cin_beam8_bond_no_patch"],
    }
    fusion_norm = float(report["results"][true_name]["beam_bond_fusion_norm"])
    checks = {
        "true_minus_cin_at_least_0005": delta_auc["true_minus_cin"] >= 0.005,
        "true_minus_shuffled_at_least_0003": (
            delta_auc["true_minus_shuffled"] >= 0.003
        ),
        "true_minus_bag_at_least_0003": delta_auc["true_minus_bag"] >= 0.003,
        "ap_not_negative_against_all_primary_controls": any(
            delta_ap[key] >= 0.0
            for key in ("true_minus_cin", "true_minus_shuffled", "true_minus_bag")
        ),
        "true_branch_active": fusion_norm > 0.0,
        "official_splits_unread": True,
    }
    summary = {
        "protocol_id": "molhiv-fulltrain-raw-beam8-gate-summary-v1",
        "input": str(args.input),
        "variants": list(REQUIRED_VARIANTS),
        "heldout_roc_auc": auc,
        "heldout_average_precision": ap,
        "delta_roc_auc": delta_auc,
        "delta_average_precision": delta_ap,
        "true_branch_fusion_norm": fusion_norm,
        "checks": checks,
        "decision": (
            "ADVANCE_TO_FOLDS_0_2"
            if all(checks.values())
            else "STOP_RAW_BEAM8_TO_CLASSIFICATION_ROUTE"
        ),
    }
    rendered = json.dumps(summary, indent=2) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(args.output)


if __name__ == "__main__":
    main()
