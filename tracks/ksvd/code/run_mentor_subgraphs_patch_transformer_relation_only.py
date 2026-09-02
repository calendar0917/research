"""Relation-only diagnostic for the mentor patch Transformer pilot.

All visible patch contents are zero.  Only the exact pair-relation tensor is
available.  This distinguishes useful content-position binding from a model
that predicts masked structure using the patch-relation graph alone.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data_mentor_subgraphs import (
    default_cache_path,
    default_source_path,
    density_stratum,
    load_bundle,
    select_pilot_indices,
)
from .mentor_grouped_splits import balanced_group_folds
from .run_beam8_coverage_operating_point_audit import GEOMETRIES
from .run_mentor_subgraphs_patch_transformer import (
    GEOMETRY,
    N_SPLITS,
    PILOT_SEED,
    COVER_SEED,
    SPLIT_SEED,
    VIEW,
    MAXIMUM_PATCHES,
    build_samples,
    fit_standardizer,
    prepare_examples,
    standardize_samples,
    train_and_evaluate,
)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_JSON = (
    ROOT
    / "tracks/ksvd/results/mentor_subgraphs/patch_transformer_relation_only_20260817.json"
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Relation-only patch Transformer diagnostic")
    parser.add_argument("--source", type=Path, default=default_source_path())
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--pilot-size", type=int, default=100)
    parser.add_argument("--pilot-seed", type=int, default=PILOT_SEED)
    parser.add_argument("--cover-seed", type=int, default=COVER_SEED)
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--model-seed", type=int, default=20260817)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    args = parser.parse_args()

    bundle = load_bundle(
        source_pkl=args.source,
        cache_path=args.cache or default_cache_path(args.source),
        force=False,
    )
    selection = select_pilot_indices(bundle, n_pilot=args.pilot_size, seed=args.pilot_seed)
    indices = selection.indices
    all_strata = density_stratum(bundle.avg_degrees)
    prepared, missing = prepare_examples(
        bundle,
        indices,
        all_strata,
        cover_seed=args.cover_seed,
        maximum_patches=MAXIMUM_PATCHES,
    )
    if missing:
        raise RuntimeError(f"missing covers: {missing[:3]}")
    folds = balanced_group_folds(
        indices,
        all_strata[indices],
        bundle.roots[indices],
        n_splits=N_SPLITS,
        seed=args.split_seed,
        view=VIEW,
    )
    patch_size, _overlap = GEOMETRIES[GEOMETRY]
    fold_rows = []
    for fold in folds:
        train_examples = [prepared[int(index)] for index in fold.train_indices]
        test_examples = [prepared[int(index)] for index in fold.test_indices]
        mean, scale = fit_standardizer(train_examples)
        train_samples = standardize_samples(
            build_samples(train_examples, branch="RELATION_ONLY", patch_size=patch_size),
            mean,
            scale,
        )
        test_samples = standardize_samples(
            build_samples(test_examples, branch="RELATION_ONLY", patch_size=patch_size),
            mean,
            scale,
        )
        result = train_and_evaluate(
            train_samples,
            test_samples,
            mean=mean,
            scale=scale,
            seed=args.model_seed + int(fold.fold_index) * 101,
            hidden=args.hidden,
            heads=args.heads,
            layers=args.layers,
            dropout=args.dropout,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        fold_rows.append({"fold_index": int(fold.fold_index), **result})
        print(
            f"fold={fold.fold_index} relation_only_rmse={result['graph_balanced_rmse']:.6f}",
            flush=True,
        )
    payload = {
        "diagnostic": "all visible patch contents zero; exact relations only",
        "selection": selection.summary(),
        "config": vars(args) | {"source": str(args.source), "cache": str(args.cache), "json": str(args.json)},
        "folds": fold_rows,
        "mean_rmse": sum(row["graph_balanced_rmse"] for row in fold_rows) / len(fold_rows),
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"mean_rmse": payload["mean_rmse"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
