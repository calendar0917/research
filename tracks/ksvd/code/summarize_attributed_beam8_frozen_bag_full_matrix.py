"""Summarize the full split0-4 x model0-2 frozen Beam8 BAG matrix."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .run_beam8_nci1_chain_classification import ROOT, _atomic_json, _atomic_text


PROTOCOL = "tracks/ksvd/docs/KSVD_ATTRIBUTED_BEAM8_FROZEN_BAG_FULL_MATRIX_PROTOCOL_20260814.md"
RESULT_DIR = ROOT / "tracks/ksvd/results/luyin14"
DEFAULT_JSON = RESULT_DIR / "attributed_beam8_frozen_bag_full_matrix_20260814.json"
DEFAULT_REPORT = RESULT_DIR / "ATTRIBUTED_BEAM8_FROZEN_BAG_FULL_MATRIX_20260814.md"


def _path(split_seed: int, model_seed: int) -> Path:
    if split_seed == 0:
        if model_seed == 0:
            return RESULT_DIR / "attributed_beam8_frozen_gine_residual_20260814.json"
        return RESULT_DIR / f"attributed_beam8_frozen_gine_residual_model_seed{model_seed}_20260814.json"
    if model_seed == 0:
        return RESULT_DIR / f"attributed_beam8_frozen_gine_residual_split_seed{split_seed}_20260814.json"
    return RESULT_DIR / f"attributed_beam8_frozen_bag_residual_split_seed{split_seed}_model_seed{model_seed}_20260814.json"


DEFAULT_CELLS = tuple(
    (split_seed, model_seed, _path(split_seed, model_seed))
    for split_seed in range(5)
    for model_seed in range(3)
)


def run() -> dict[str, Any]:
    rows = []
    inputs = []
    for split_seed, model_seed, path in DEFAULT_CELLS:
        payload = json.loads(path.read_text(encoding="utf-8"))
        config = payload.get("config") or {}
        observed_split = int(config.get("split_seed", split_seed))
        observed_model = int(config.get("model_seed", model_seed))
        if (observed_split, observed_model) != (split_seed, model_seed):
            raise ValueError(
                f"cell mismatch for {path}: expected {(split_seed, model_seed)}, got {(observed_split, observed_model)}"
            )
        inputs.append({"split_seed": split_seed, "model_seed": model_seed, "path": str(path)})
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
    if len(rows) != 45:
        raise ValueError(f"expected 45 fold units, got {len(rows)}")

    values = np.asarray([row["bag"] - row["gine"] for row in rows])
    gine_values = np.asarray([row["gine"] for row in rows])
    bag_values = np.asarray([row["bag"] for row in rows])
    split_means = {
        str(split): float(np.mean([value for value, row in zip(values, rows) if row["split_seed"] == split]))
        for split in range(5)
    }
    model_means = {
        str(model): float(np.mean([value for value, row in zip(values, rows) if row["model_seed"] == model]))
        for model in range(3)
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
        for split in range(5)
        for model in range(3)
    }
    checks = {
        "mean": float(values.mean()) >= 0.005,
        "fold_wins": int(np.sum(values > 1e-12)) >= 30,
        "split_majority": int(np.sum(np.asarray(list(split_means.values())) > 0)) >= 4,
        "model_majority": int(np.sum(np.asarray(list(model_means.values())) > 0)) >= 2,
        "worst_split": min(split_means.values()) >= -0.005,
        "worst_model": min(model_means.values()) >= -0.005,
        "cell_majority": int(np.sum(np.asarray(list(cell_means.values())) > 0)) >= 10,
    }
    decision = (
        "FROZEN_BAG_CONFIRMED_AS_STABLE_BEAM8_CLASSIFICATION_INTERFACE"
        if all(checks.values())
        else "FROZEN_BAG_FULL_MATRIX_FAILS_STOP_BEAM8_CLASSIFICATION"
    )
    strong_mask = gine_values >= 0.74
    bottom_indices = np.argsort(gine_values)[:5]
    return {
        "protocol": PROTOCOL,
        "inputs": inputs,
        "units": rows,
        "summary": {
            "gine_mean": float(gine_values.mean()),
            "gine_std": float(gine_values.std()),
            "bag_mean": float(bag_values.mean()),
            "bag_std": float(bag_values.std()),
            "delta_mean": float(values.mean()),
            "delta_std": float(values.std()),
            "delta_median": float(np.median(values)),
            "gine_delta_correlation": float(np.corrcoef(gine_values, values)[0, 1]),
            "strong_base": {
                "threshold": 0.74,
                "units": int(strong_mask.sum()),
                "delta_mean": float(values[strong_mask].mean()),
                "wins": int(np.sum(values[strong_mask] > 1e-12)),
                "losses": int(np.sum(values[strong_mask] < -1e-12)),
            },
            "bottom5_base": {
                "gine_mean": float(gine_values[bottom_indices].mean()),
                "bag_mean": float(bag_values[bottom_indices].mean()),
                "delta_mean": float(values[bottom_indices].mean()),
            },
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
        "# Attributed Beam8 graph-conditioned frozen BAG full matrix",
        "",
        f"> 协议：`{payload['protocol']}`  ",
        f"> 判定：`{summary['decision']}`",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| GINE mean | {summary['gine_mean']:.4f} |",
        f"| GINE std | {summary['gine_std']:.4f} |",
        f"| BAG mean | {summary['bag_mean']:.4f} |",
        f"| BAG std | {summary['bag_std']:.4f} |",
        f"| BAG−GINE | {summary['delta_mean']:+.4f} ± {summary['delta_std']:.4f} |",
        f"| median BAG−GINE | {summary['delta_median']:+.4f} |",
        f"| W/T/L | {summary['wins']}/{summary['ties']}/{summary['losses']} |",
        "",
        "## Split means",
        "",
        "| split seed | BAG−GINE |",
        "|---:|---:|",
    ]
    for seed in range(5):
        lines.append(f"| {seed} | {summary['split_means'][str(seed)]:+.4f} |")
    lines.extend(["", "## Model means", "", "| model seed | BAG−GINE |", "|---:|---:|"])
    for seed in range(3):
        lines.append(f"| {seed} | {summary['model_means'][str(seed)]:+.4f} |")
    lines.extend(["", "## Cell means", "", "| split:model | BAG−GINE |", "|---|---:|"])
    for key, value in summary["cell_means"].items():
        lines.append(f"| {key} | {value:+.4f} |")
    strong = summary["strong_base"]
    bottom = summary["bottom5_base"]
    lines.extend(
        [
            "",
            "## Post-hoc stability diagnosis",
            "",
            f"- corr(GINE, BAG−GINE)：`{summary['gine_delta_correlation']:+.4f}`；",
            f"- GINE≥{strong['threshold']:.2f}：{strong['units']} units，mean delta `{strong['delta_mean']:+.4f}`，W/L `{strong['wins']}/{strong['losses']}`；",
            f"- weakest 5 GINE units：GINE `{bottom['gine_mean']:.4f}` → BAG `{bottom['bag_mean']:.4f}`，delta `{bottom['delta_mean']:+.4f}`；",
        ]
    )
    lines.extend(["", "## Frozen checks", ""])
    for name, value in summary["checks"].items():
        lines.append(f"- {name}：`{value}`；")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- The stable interface is graph-conditioned BAG calibration on a frozen GINE.",
            "- The post-hoc diagnosis characterizes variance reduction and does not alter the frozen gate.",
            "- This result does not revive localized binding, atom gates, cross-attention or ordinary KSVD updates.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    payload = run()
    _atomic_json(args.json, payload)
    _atomic_text(args.report, render(payload))
    print(args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
