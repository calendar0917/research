"""Summarize unseen split seeds 3/4 for the frozen Beam8 residual route."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .run_beam8_nci1_chain_classification import ROOT, _atomic_json, _atomic_text


PROTOCOL = "tracks/ksvd/docs/KSVD_ATTRIBUTED_BEAM8_FROZEN_UNSEEN_SPLIT_CONFIRMATION_PROTOCOL_20260814.md"
RESULT_DIR = ROOT / "tracks/ksvd/results/luyin14"
DEFAULT_RESIDUALS = (
    RESULT_DIR / "attributed_beam8_frozen_gine_residual_split_seed3_20260814.json",
    RESULT_DIR / "attributed_beam8_frozen_gine_residual_split_seed4_20260814.json",
)
DEFAULT_AUDIT = RESULT_DIR / "attributed_beam8_frozen_binding_shuffle_audit_unseen_splits34_20260814.json"
DEFAULT_JSON = RESULT_DIR / "attributed_beam8_frozen_unseen_split_confirmation_20260814.json"
DEFAULT_REPORT = RESULT_DIR / "ATTRIBUTED_BEAM8_FROZEN_UNSEEN_SPLIT_CONFIRMATION_20260814.md"
VARIANTS = ("GINE_FROZEN", "TRUE_RESIDUAL", "SHUFFLED_RESIDUAL", "BAG_RESIDUAL")


def _paired(rows: list[dict[str, Any]], left: str, right: str) -> dict[str, Any]:
    values = np.asarray(
        [row["scores"][left]["balanced_accuracy"] - row["scores"][right]["balanced_accuracy"] for row in rows]
    )
    split_means = {
        str(seed): float(np.mean([value for value, row in zip(values, rows) if row["split_seed"] == seed]))
        for seed in (3, 4)
    }
    return {
        "mean": float(values.mean()),
        "wins": int(np.sum(values > 1e-12)),
        "ties": int(np.sum(np.abs(values) <= 1e-12)),
        "losses": int(np.sum(values < -1e-12)),
        "values": values.tolist(),
        "split_means": split_means,
    }


def run(residual_paths: list[Path], audit_path: Path) -> dict[str, Any]:
    residuals = [json.loads(path.read_text(encoding="utf-8")) for path in residual_paths]
    rows = []
    for payload, path in zip(residuals, residual_paths):
        config = payload.get("config") or {}
        split_seed = int(config.get("split_seed", -1))
        model_seed = int(config.get("model_seed", -1))
        if split_seed not in (3, 4) or model_seed != 0:
            raise ValueError(f"unexpected config in {path}: {config}")
        for fold in payload["folds"]:
            rows.append(
                {
                    "split_seed": split_seed,
                    "fold_index": int(fold["fold_index"]),
                    "scores": fold["scores"],
                }
            )
    if len(rows) != 6 or sorted({row["split_seed"] for row in rows}) != [3, 4]:
        raise ValueError("expected six units over split seeds 3/4")

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("config", {}).get("split_seeds") != [3, 4] or audit.get("summary", {}).get("comparisons") != 48:
        raise ValueError("binding audit does not match split seeds 3/4 with 48 comparisons")

    variants = {
        variant: {
            "balanced_accuracy_mean": float(np.mean([row["scores"][variant]["balanced_accuracy"] for row in rows])),
            "balanced_accuracy_std": float(np.std([row["scores"][variant]["balanced_accuracy"] for row in rows])),
        }
        for variant in VARIANTS
    }
    paired = {
        "increment": _paired(rows, "TRUE_RESIDUAL", "GINE_FROZEN"),
        "localization": _paired(rows, "TRUE_RESIDUAL", "BAG_RESIDUAL"),
    }
    binding = audit["summary"]
    checks = {
        "classification": paired["increment"]["mean"] >= 0.01 and paired["increment"]["wins"] >= 4,
        "localization": paired["localization"]["mean"] >= 0.005 and paired["localization"]["wins"] >= 4,
        "classification_both_splits": all(value > 0 for value in paired["increment"]["split_means"].values()),
        "localization_both_splits": all(value > 0 for value in paired["localization"]["split_means"].values()),
        "binding_audit": binding["decision"] == "EXACT_BINDING_SUPPORTED_AGAINST_SHUFFLE_DISTRIBUTION",
        "parity": bool(binding["checks"]["repeat0_parity"]),
    }
    decision = (
        "UNSEEN_SPLITS_CONFIRM_FROZEN_BEAM8_ROUTE_ADVANCE_TO_ATOM_GATE"
        if all(checks.values())
        else "UNSEEN_SPLITS_DO_NOT_CONFIRM_FROZEN_BEAM8_ROUTE_STOP_EXPANSION"
    )
    return {
        "protocol": PROTOCOL,
        "residual_inputs": [str(path) for path in residual_paths],
        "binding_audit_input": str(audit_path),
        "units": rows,
        "summary": {
            "variants": variants,
            "paired": paired,
            "binding": binding,
            "checks": checks,
            "decision": decision,
        },
    }


def render(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Attributed Beam8 frozen residual unseen-split confirmation",
        "",
        f"> 协议：`{payload['protocol']}`  ",
        f"> 判定：`{summary['decision']}`",
        "",
        "## Classification variants",
        "",
        "| variant | balanced accuracy over 6 units |",
        "|---|---:|",
    ]
    for variant, row in summary["variants"].items():
        lines.append(f"| {variant} | {row['balanced_accuracy_mean']:.4f} ± {row['balanced_accuracy_std']:.4f} |")
    lines.extend(["", "## Paired classification", "", "| comparison | mean | W/T/L | split3/4 |", "|---|---:|---:|---:|"])
    for name, row in summary["paired"].items():
        split_text = " / ".join(f"{row['split_means'][str(seed)]:+.4f}" for seed in (3, 4))
        lines.append(f"| {name} | {row['mean']:+.4f} | {row['wins']}/{row['ties']}/{row['losses']} | {split_text} |")
    binding = summary["binding"]
    lines.extend(
        [
            "",
            "## Repeated binding",
            "",
            f"- TRUE−SHUFFLED mean：`{binding['mean_delta']:+.4f}`；",
            f"- W/T/L：`{binding['wins']}/{binding['ties']}/{binding['losses']}`；",
            "- split3/4：`"
            + " / ".join(f"{binding['split_means'][str(seed)]:+.4f}" for seed in (3, 4))
            + "`；",
            "",
            "## Frozen checks",
            "",
        ]
    )
    for name, value in summary["checks"].items():
        lines.append(f"- {name}：`{value}`；")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- split seeds 3/4 在协议冻结前未运行。",
            "- model seed 固定为0；本轮只确认新的数据划分。",
            "- 全部通过只授权下一步低容量 frozen-dictionary gate，不授权 cross-attention 或普通 KSVD updates。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--residuals", nargs=2, type=Path, default=list(DEFAULT_RESIDUALS))
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    payload = run(args.residuals, args.audit)
    _atomic_json(args.json, payload)
    _atomic_text(args.report, render(payload))
    print(args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
