"""Summarize frozen-GINE Beam8 residual runs over split seeds 0/1/2."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .run_beam8_nci1_chain_classification import ROOT, _atomic_json, _atomic_text


PROTOCOL = "tracks/ksvd/docs/KSVD_ATTRIBUTED_BEAM8_FROZEN_RESIDUAL_MULTI_SPLIT_PROTOCOL_20260814.md"
RESULT_DIR = ROOT / "tracks/ksvd/results/luyin14"
DEFAULT_INPUTS = (
    RESULT_DIR / "attributed_beam8_frozen_gine_residual_20260814.json",
    RESULT_DIR / "attributed_beam8_frozen_gine_residual_split_seed1_20260814.json",
    RESULT_DIR / "attributed_beam8_frozen_gine_residual_split_seed2_20260814.json",
)
DEFAULT_JSON = RESULT_DIR / "attributed_beam8_frozen_residual_multi_split_20260814.json"
DEFAULT_REPORT = RESULT_DIR / "ATTRIBUTED_BEAM8_FROZEN_RESIDUAL_MULTI_SPLIT_20260814.md"
VARIANTS = ("GINE_FROZEN", "TRUE_RESIDUAL", "SHUFFLED_RESIDUAL", "BAG_RESIDUAL")


def _delta(rows: list[dict[str, Any]], left: str, right: str) -> dict[str, Any]:
    values = np.asarray(
        [row["scores"][left]["balanced_accuracy"] - row["scores"][right]["balanced_accuracy"] for row in rows]
    )
    split_means = {
        str(seed): float(np.mean([value for value, row in zip(values, rows) if row["split_seed"] == seed]))
        for seed in (0, 1, 2)
    }
    return {
        "mean": float(values.mean()),
        "wins": int(np.sum(values > 1e-12)),
        "ties": int(np.sum(np.abs(values) <= 1e-12)),
        "losses": int(np.sum(values < -1e-12)),
        "values": values.tolist(),
        "split_means": split_means,
    }


def run(paths: list[Path]) -> dict[str, Any]:
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    rows = []
    for fallback_seed, payload in enumerate(payloads):
        config = payload.get("config") or {}
        split_seed = int(config.get("split_seed", fallback_seed))
        model_seed = int(config.get("model_seed", 0))
        if model_seed != 0:
            raise ValueError(f"multi-split protocol requires model_seed=0, got {model_seed} in {paths[fallback_seed]}")
        for fold in payload["folds"]:
            rows.append(
                {
                    "split_seed": split_seed,
                    "model_seed": model_seed,
                    "fold_index": fold["fold_index"],
                    "scores": fold["scores"],
                }
            )
    observed = sorted({row["split_seed"] for row in rows})
    if observed != [0, 1, 2] or len(rows) != 9:
        raise ValueError(f"expected split seeds 0/1/2 and 9 units, got seeds={observed}, units={len(rows)}")
    variants = {
        variant: {
            "balanced_accuracy_mean": float(np.mean([row["scores"][variant]["balanced_accuracy"] for row in rows])),
            "balanced_accuracy_std": float(np.std([row["scores"][variant]["balanced_accuracy"] for row in rows])),
        }
        for variant in VARIANTS
    }
    paired = {
        "increment": _delta(rows, "TRUE_RESIDUAL", "GINE_FROZEN"),
        "binding": _delta(rows, "TRUE_RESIDUAL", "SHUFFLED_RESIDUAL"),
        "localization": _delta(rows, "TRUE_RESIDUAL", "BAG_RESIDUAL"),
    }
    checks = {
        "increment": paired["increment"]["mean"] >= 0.01 and paired["increment"]["wins"] >= 6,
        "binding": paired["binding"]["mean"] >= 0.005 and paired["binding"]["wins"] >= 6,
        "localization": paired["localization"]["mean"] >= 0.005 and paired["localization"]["wins"] >= 6,
        "split_majority": all(
            int(np.sum(np.asarray(list(row["split_means"].values())) > 0)) >= 2 for row in paired.values()
        ),
        "worst_split_increment": min(paired["increment"]["split_means"].values()) >= -0.005,
    }
    decision = (
        "FROZEN_RESIDUAL_STABLE_ACROSS_SPLITS"
        if all(checks.values())
        else "FROZEN_RESIDUAL_NOT_STABLE_ACROSS_SPLITS"
    )
    return {
        "protocol": PROTOCOL,
        "inputs": [str(path) for path in paths],
        "units": rows,
        "summary": {"variants": variants, "paired": paired, "checks": checks, "decision": decision},
    }


def render(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Attributed Beam8 frozen residual multi-split validation",
        "",
        f"> 协议：`{payload['protocol']}`  ",
        f"> 判定：`{summary['decision']}`",
        "",
        "| variant | balanced accuracy over 9 units |",
        "|---|---:|",
    ]
    for variant, row in summary["variants"].items():
        lines.append(f"| {variant} | {row['balanced_accuracy_mean']:.4f} ± {row['balanced_accuracy_std']:.4f} |")
    lines.extend(["", "## Paired attribution", "", "| comparison | mean | W/T/L | split0/1/2 |", "|---|---:|---:|---:|"])
    for name, row in summary["paired"].items():
        split_text = " / ".join(f"{row['split_means'][str(seed)]:+.4f}" for seed in (0, 1, 2))
        lines.append(f"| {name} | {row['mean']:+.4f} | {row['wins']}/{row['ties']}/{row['losses']} | {split_text} |")
    lines.extend(["", "## Frozen checks", ""])
    for name, value in summary["checks"].items():
        lines.append(f"- {name}：`{value}`；")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- model seed 固定为0；本报告只回答跨数据划分稳定性。",
            "- 每个 split/fold 的 dictionary 只在对应 outer-train 上拟合。",
            "- TRUE/SHUFFLED/BAG 在每个 split/fold 内共用完全相同的 frozen GINE。",
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
