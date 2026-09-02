"""Summarize split seeds 3/4 x model seeds 0/1/2 for frozen Beam8 BAG residual."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .run_beam8_nci1_chain_classification import ROOT, _atomic_json, _atomic_text


PROTOCOL = "tracks/ksvd/docs/KSVD_ATTRIBUTED_BEAM8_FROZEN_BAG_DUAL_AXIS_CONFIRMATION_PROTOCOL_20260814.md"
RESULT_DIR = ROOT / "tracks/ksvd/results/luyin14"
DEFAULT_INPUTS = (
    RESULT_DIR / "attributed_beam8_frozen_gine_residual_split_seed3_20260814.json",
    RESULT_DIR / "attributed_beam8_frozen_gine_residual_split_seed4_20260814.json",
    *(RESULT_DIR / f"attributed_beam8_frozen_bag_residual_split_seed{split}_model_seed{model}_20260814.json" for split in (3, 4) for model in (1, 2)),
)
DEFAULT_JSON = RESULT_DIR / "attributed_beam8_frozen_bag_dual_axis_20260814.json"
DEFAULT_REPORT = RESULT_DIR / "ATTRIBUTED_BEAM8_FROZEN_BAG_DUAL_AXIS_20260814.md"


def run(paths: list[Path]) -> dict[str, Any]:
    rows = []
    cells = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        config = payload.get("config") or {}
        split_seed = int(config.get("split_seed", -1))
        model_seed = int(config.get("model_seed", 0))
        if split_seed not in (3, 4) or model_seed not in (0, 1, 2):
            raise ValueError(f"unexpected config in {path}: {config}")
        cell = (split_seed, model_seed)
        if cell in cells:
            raise ValueError(f"duplicate cell {cell}")
        cells.add(cell)
        for fold in payload["folds"]:
            rows.append(
                {
                    "split_seed": split_seed,
                    "model_seed": model_seed,
                    "fold_index": int(fold["fold_index"]),
                    "gine": float(fold["scores"]["GINE_FROZEN"]["balanced_accuracy"]),
                    "bag": float(fold["scores"]["BAG_RESIDUAL"]["balanced_accuracy"]),
                }
            )
    expected_cells = {(split, model) for split in (3, 4) for model in (0, 1, 2)}
    if cells != expected_cells or len(rows) != 18:
        raise ValueError(f"expected six cells and 18 folds, got cells={sorted(cells)}, rows={len(rows)}")

    values = np.asarray([row["bag"] - row["gine"] for row in rows])
    split_means = {
        str(split): float(np.mean([value for value, row in zip(values, rows) if row["split_seed"] == split]))
        for split in (3, 4)
    }
    model_means = {
        str(model): float(np.mean([value for value, row in zip(values, rows) if row["model_seed"] == model]))
        for model in (0, 1, 2)
    }
    cell_means = {
        f"{split}:{model}": float(
            np.mean(
                [
                    value
                    for value, row in zip(values, rows)
                    if row["split_seed"] == split and row["model_seed"] == model
                ]
            )
        )
        for split in (3, 4)
        for model in (0, 1, 2)
    }
    checks = {
        "mean": float(values.mean()) >= 0.005,
        "fold_wins": int(np.sum(values > 1e-12)) >= 12,
        "both_splits": all(value > 0 for value in split_means.values()),
        "model_majority": int(np.sum(np.asarray(list(model_means.values())) > 0)) >= 2,
        "worst_model": min(model_means.values()) >= -0.005,
        "cell_majority": int(np.sum(np.asarray(list(cell_means.values())) > 0)) >= 4,
    }
    decision = (
        "FROZEN_BAG_ADVANCES_TO_FULL_DUAL_AXIS_MATRIX"
        if all(checks.values())
        else "FROZEN_BAG_NOT_STABLE_STOP_BEAM8_CLASSIFICATION_EXPANSION"
    )
    return {
        "protocol": PROTOCOL,
        "inputs": [str(path) for path in paths],
        "units": rows,
        "summary": {
            "gine_mean": float(np.mean([row["gine"] for row in rows])),
            "bag_mean": float(np.mean([row["bag"] for row in rows])),
            "delta_mean": float(values.mean()),
            "delta_std": float(values.std()),
            "wins": int(np.sum(values > 1e-12)),
            "ties": int(np.sum(np.abs(values) <= 1e-12)),
            "losses": int(np.sum(values < -1e-12)),
            "split_means": split_means,
            "model_means": model_means,
            "cell_means": cell_means,
            "checks": checks,
            "decision": decision,
        },
    }


def render(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Attributed Beam8 graph-conditioned frozen BAG dual-axis confirmation",
        "",
        f"> 协议：`{payload['protocol']}`  ",
        f"> 判定：`{summary['decision']}`",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| GINE mean | {summary['gine_mean']:.4f} |",
        f"| BAG mean | {summary['bag_mean']:.4f} |",
        f"| BAG−GINE | {summary['delta_mean']:+.4f} ± {summary['delta_std']:.4f} |",
        f"| W/T/L | {summary['wins']}/{summary['ties']}/{summary['losses']} |",
        "",
        "## Axis means",
        "",
        "| axis | deltas |",
        "|---|---:|",
        "| split3/4 | " + " / ".join(f"{summary['split_means'][str(seed)]:+.4f}" for seed in (3, 4)) + " |",
        "| model0/1/2 | " + " / ".join(f"{summary['model_means'][str(seed)]:+.4f}" for seed in (0, 1, 2)) + " |",
        "",
        "## Cell means",
        "",
        "| split:model | BAG−GINE |",
        "|---|---:|",
    ]
    for key, value in summary["cell_means"].items():
        lines.append(f"| {key} | {value:+.4f} |")
    lines.extend(["", "## Frozen checks", ""])
    for name, value in summary["checks"].items():
        lines.append(f"- {name}：`{value}`；")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- This route uses only graph-conditioned BAG fields and a frozen GINE backbone.",
            "- Passing only authorizes completion of the remaining split/model grid.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="*", type=Path, default=list(DEFAULT_INPUTS))
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    payload = run(args.inputs)
    _atomic_json(args.json, payload)
    _atomic_text(args.report, render(payload))
    print(args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
