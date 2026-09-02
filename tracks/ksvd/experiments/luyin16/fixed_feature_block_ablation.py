"""Attribute the fixed MolHIV result to topology, chemistry, and K-SVD blocks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from tracks.ksvd.experiments.luyin16.mentor_concept_replication import (
    REPO_ROOT,
    TOPOLOGY_FEATURE_NAMES,
    _write_json,
    _write_records,
    fit_xgboost_views,
)


DEFAULT_FEATURE_DIR = REPO_ROOT / "tracks/ksvd/results/luyin16/mentor_concept_v1_k32_official_valid"
DEFAULT_RESULT_DIR = REPO_ROOT / "tracks/ksvd/results/luyin16/mentor_concept_v1_feature_blocks"


def build_feature_block_views(
    composition: np.ndarray,
    t_initial: np.ndarray,
    t_final: np.ndarray,
    schema: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    topology_dim = len(TOPOLOGY_FEATURE_NAMES)
    atom_dim = sum(int(value) for value in schema["atom_dimensions"])
    bond_dim = sum(int(value) for value in schema["bond_dimensions"])
    expected = topology_dim + atom_dim + bond_dim
    if composition.ndim != 2 or composition.shape[1] != expected:
        raise ValueError(f"composition shape {composition.shape} does not match schema dimension {expected}")
    if t_initial.shape != t_final.shape or t_initial.shape[0] != composition.shape[0]:
        raise ValueError("T_init/T_final must be aligned with composition")

    topology = composition[:, :topology_dim]
    atom = composition[:, topology_dim : topology_dim + atom_dim]
    bond = composition[:, topology_dim + atom_dim :]
    chemistry = np.concatenate([atom, bond], axis=1)
    views = {
        "topology": topology,
        "atom_composition": atom,
        "bond_composition": bond,
        "chemistry_composition": chemistry,
        "s": composition,
        "topology_t_init": np.concatenate([topology, t_initial], axis=1),
        "topology_t_final": np.concatenate([topology, t_final], axis=1),
        "chemistry_t_init": np.concatenate([chemistry, t_initial], axis=1),
        "chemistry_t_final": np.concatenate([chemistry, t_final], axis=1),
        "s_t_init": np.concatenate([composition, t_initial], axis=1),
        "s_t_final": np.concatenate([composition, t_final], axis=1),
    }
    return views, {"topology": topology_dim, "atom": atom_dim, "bond": bond_dim}


def _render_summary(payload: Mapping[str, Any]) -> str:
    lines = [
        "# luyin16 fixed-feature block ablation",
        "",
        f"Source protocol: `{payload['source_protocol_id']}`",
        "",
        "All models use the frozen classifier settings and official validation only; test is not evaluated.",
        "",
        "| view | dim | train ROC-AUC | valid ROC-AUC |",
        "|---|---:|---:|---:|",
    ]
    for name, result in payload["classifier"]["per_view"].items():
        lines.append(
            f"| {name} | {result['dimension']} | "
            f"{result['train']['mean']:.6f} ± {result['train']['std_sample']:.6f} | "
            f"{result['valid']['mean']:.6f} ± {result['valid']['std_sample']:.6f} |"
        )
    lines.extend(
        [
            "",
            f"Best by validation only: `{payload['classifier']['best_by_valid_only']}`.",
            "",
            "Interpretation gates:",
            "",
            "- `topology` measures explicit structure without atom/bond categories.",
            "- `chemistry_composition` measures atom/bond categories without explicit topology statistics.",
            "- `*_t_final - *_t_init` attributes K-SVD updates under the same downstream model.",
            "- `chemistry_t_final - chemistry_composition` tests dictionary information beyond composition.",
            "",
        ]
    )
    return "\n".join(lines)


def run(feature_dir: Path, result_dir: Path) -> dict[str, Any]:
    source_summary = json.loads((feature_dir / "summary.json").read_text(encoding="utf-8"))
    schema = json.loads((feature_dir / "composition_schema.json").read_text(encoding="utf-8"))
    with np.load(feature_dir / "features_and_predictions.npz") as payload:
        labels = np.asarray(payload["labels"])
        split = {
            name: np.asarray(payload[f"{name}_indices"], dtype=np.int64)
            for name in ("train", "valid", "test")
        }
        views, dimensions = build_feature_block_views(
            np.asarray(payload["composition"], dtype=np.float32),
            np.asarray(payload["t_initial"], dtype=np.float32),
            np.asarray(payload["t_final"], dtype=np.float32),
            schema,
        )
    classifier_config = dict(source_summary["resolved_config"]["classifier"])
    records, fitted = fit_xgboost_views(
        views,
        labels,
        split,
        classifier_config,
        evaluate_test=False,
    )
    result = {
        "protocol_id": "mentor-concept-replication-v1-feature-block-ablation",
        "source_protocol_id": source_summary["protocol_id"],
        "source_feature_dir": str(feature_dir),
        "block_dimensions": dimensions,
        "classifier": fitted["summary"],
        "test_evaluated": False,
    }
    result_dir.mkdir(parents=True, exist_ok=True)
    _write_records(result_dir / "records.csv", records)
    _write_json(result_dir / "summary.json", result)
    (result_dir / "FEATURE_BLOCK_ABLATION.md").write_text(_render_summary(result), encoding="utf-8")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(args.feature_dir.expanduser().resolve(), args.result_dir.expanduser().resolve())
    print(
        json.dumps(
            {
                "protocol_id": result["protocol_id"],
                "source_protocol_id": result["source_protocol_id"],
                "best_by_valid_only": result["classifier"]["best_by_valid_only"],
                "test_evaluated": result["test_evaluated"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
