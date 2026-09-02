"""Controlled readout ablation for the clean rooted-WL MolHIV object.

This experiment keeps the radius-2 rooted-WL roles and strict atom/bond
semantics fixed.  It compares the historical graph-level mean readout with a
permutation-invariant distribution readout over the *same* centre-level
marginal patch rows.  An optional 205-D global composition/topology block from
the frozen mentor-shaped proxy is included only as an information-budget
control; it is never used to build the local rows.

Official test is deliberately not loaded.  The output is a mechanism probe,
not a terminal model-selection run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from ksvd_research.data import load_molhiv
from tracks.ksvd.experiments.luyin16.role_attribute_binding_screen import (
    REPO_ROOT,
    _resolve,
)
from tracks.ksvd.experiments.luyin16.structural_role_fusion_screen import (
    patch_feature_rows,
)


DEFAULT_CONFIG = REPO_ROOT / "tracks/ksvd/configs/luyin16/structural_role_terminal_audit.yaml"
DEFAULT_RESULT_DIR = REPO_ROOT / "tracks/ksvd/results/luyin16/clean_patch_distribution_readout"
LOCAL_BLOCKS = ("node_role", "edge_role", "node_attribute", "edge_attribute")
STAT_NAMES = (
    "mean",
    "std",
    "min",
    "max",
    "q10",
    "q25",
    "q50",
    "q75",
    "q90",
    "abs_mean",
    "rms",
    "nonzero_fraction",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def distribution_readout(rows: np.ndarray) -> np.ndarray:
    """Compute the same 12 fixed summaries used by the mentor proxy."""
    values = np.asarray(rows, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"expected [n_centres, width], got {values.shape}")
    if values.shape[0] == 0:
        return np.zeros(len(STAT_NAMES) * values.shape[1], dtype=np.float32)
    quantiles = np.quantile(values, [0.10, 0.25, 0.50, 0.75, 0.90], axis=0)
    blocks = (
        values.mean(axis=0),
        values.std(axis=0),
        values.min(axis=0),
        values.max(axis=0),
        *quantiles,
        np.abs(values).mean(axis=0),
        np.sqrt(np.mean(values * values, axis=0)),
        (np.abs(values) > 1.0e-10).mean(axis=0),
    )
    result = np.concatenate(blocks).astype(np.float32, copy=False)
    if result.shape != (len(STAT_NAMES) * values.shape[1],):
        raise RuntimeError("distribution readout width changed")
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("distribution readout contains non-finite values")
    return result


def _local_marginal_rows(
    graph,
    node_features: np.ndarray,
    edge_features: Mapping[tuple[int, int], np.ndarray],
    representation: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    rows = patch_feature_rows(
        graph,
        node_features,
        edge_features,
        "rooted_wl",
        representation,
    )
    local = np.concatenate([rows[name] for name in LOCAL_BLOCKS], axis=1)
    # ``context`` is already the exact graph-level context used by graph_features.
    context = np.asarray(rows["context"], dtype=np.float32).reshape(-1)
    return local.astype(np.float32, copy=False), context


def build_distribution_features(
    bundle,
    indices: np.ndarray,
    representation: Mapping[str, Any],
    cache_path: Path,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Build/read [graph, 12*149] distribution features and graph context."""
    selected = np.asarray(indices, dtype=np.int64)
    signature_payload = {
        "schema": "rooted_wl_local_marginal_distribution",
        "representation": dict(representation),
        "indices": selected.tolist(),
        "encoder": _sha256(Path(__file__).resolve().parent / "structural_role_fusion_screen.py"),
    }
    signature = json.dumps(signature_payload, sort_keys=True, separators=(",", ":"))
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as archive:
            if str(np.asarray(archive["signature"]).reshape(-1)[0]) != signature:
                raise ValueError(f"distribution cache signature mismatch: {cache_path}")
            return (
                np.asarray(archive["distribution"], dtype=np.float32),
                np.asarray(archive["context"], dtype=np.float32),
                True,
            )
    if bundle.node_feats is None or bundle.edge_feats is None:
        raise ValueError("local distribution features require node and edge features")
    if selected.size == 0:
        raise ValueError("no graph indices selected")
    first_local, first_context = _local_marginal_rows(
        bundle.graphs[int(selected[0])],
        bundle.node_feats[int(selected[0])],
        bundle.edge_feats[int(selected[0])],
        representation,
    )
    distribution = np.zeros(
        (selected.size, len(STAT_NAMES) * first_local.shape[1]), dtype=np.float32
    )
    contexts = np.zeros((selected.size, first_context.shape[0]), dtype=np.float32)
    for position, raw_index in enumerate(selected):
        if position == 0:
            local, context = first_local, first_context
        else:
            local, context = _local_marginal_rows(
                bundle.graphs[int(raw_index)],
                bundle.node_feats[int(raw_index)],
                bundle.edge_feats[int(raw_index)],
                representation,
            )
        distribution[position] = distribution_readout(local)
        contexts[position] = context
        if position and position % 500 == 0:
            print(f"distribution feature graphs: {position}/{selected.size}", flush=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            signature=np.asarray([signature]),
            dataset_indices=selected,
            distribution=distribution,
            context=contexts,
        )
    temporary.replace(cache_path)
    return distribution, contexts, False


def _fit_score(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    params: Mapping[str, Any],
    seeds: Sequence[int],
) -> dict[str, Any]:
    rows = []
    for seed in seeds:
        current = dict(params)
        current["random_state"] = int(seed)
        current["scale_pos_weight"] = float(
            np.sum(y_train == 0) / max(1, np.sum(y_train == 1))
        )
        model = XGBClassifier(**current)
        model.fit(x_train, y_train)
        prediction = model.predict_proba(x_valid)[:, 1]
        rows.append({"seed": int(seed), "auc": float(roc_auc_score(y_valid, prediction))})
    values = [row["auc"] for row in rows]
    return {
        "rows": rows,
        "mean_auc": float(np.mean(values)),
        "std_auc": float(np.std(values, ddof=0)),
    }


def run(
    *,
    config_path: Path = DEFAULT_CONFIG,
    result_dir: Path = DEFAULT_RESULT_DIR,
    model_seeds: Sequence[int] = (0, 1, 2),
) -> dict[str, Any]:
    import yaml

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    representation = config["representation"]
    data_root = _resolve(config["data"]["root"])
    bundle = load_molhiv(root=data_root, with_features=True)
    old_path = _resolve(
        "tracks/ksvd/results/luyin16/mentor_r2_atom_chem_k64_s8_all_official_valid/feature_views_train_valid.npz"
    )
    terminal_path = _resolve(
        "tracks/ksvd/results/luyin16/structural_role_terminal_audit/dev_features.npz"
    )
    with np.load(terminal_path, allow_pickle=False) as archive:
        indices = np.asarray(archive["dataset_indices"], dtype=np.int64)
        y = np.asarray(archive["labels"], dtype=np.int64)
        mean_ta = np.asarray(archive["t_a"], dtype=np.float32)
    train_count = int(np.sum(np.isin(indices, bundle.split["train"])))
    # The cache is ordered by the same sorted train+valid indices as the
    # terminal feature cache; assert this explicitly before any comparison.
    expected = np.sort(np.concatenate([bundle.split["train"], bundle.split["valid"]]))
    if not np.array_equal(np.sort(indices), expected):
        raise RuntimeError("terminal cache does not contain exactly official train+valid graphs")
    if not np.all(np.isin(indices[:train_count], bundle.split["train"])) or not np.all(
        np.isin(indices[train_count:], bundle.split["valid"])
    ):
        raise RuntimeError("terminal cache is not train-then-valid ordered")
    distribution, context, _cache_hit = build_distribution_features(
        bundle,
        indices,
        representation,
        result_dir / "clean_distribution_features.npz",
    )
    if not np.allclose(
        distribution[:, : mean_ta.shape[1] - context.shape[1]],
        mean_ta[:, : mean_ta.shape[1] - context.shape[1]],
        atol=2.0e-6,
        rtol=0.0,
    ):
        raise RuntimeError("distribution mean block does not reproduce T+A")
    with np.load(old_path, allow_pickle=False) as archive:
        old_indices = np.asarray(archive["dataset_indices"], dtype=np.int64)
        global_s = np.asarray(archive["s"], dtype=np.float32)
    if not np.array_equal(old_indices, indices):
        raise RuntimeError("global S and clean feature caches are not aligned")

    manifest = json.loads(
        (_resolve("tracks/ksvd/results/luyin16/structural_role_terminal_audit/frozen_manifest.json")).read_text(
            encoding="utf-8"
        )
    )
    ta_params = dict(manifest["views"]["t_a"]["search"]["best_params"])
    # Static XGBoost fields are already valid; force a deterministic, explicit
    # local thread count for the probe.
    ta_params.update(
        {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
            "n_jobs": 8,
            "max_bin": 256,
        }
    )
    train = np.arange(train_count, dtype=np.int64)
    valid = np.arange(train_count, indices.size, dtype=np.int64)
    # Keep the original graph-level context in the mean baseline.  For the
    # distribution view, append context once (not 12 redundant copies).
    dist_with_context = np.concatenate([distribution, context], axis=1)
    views = {
        "ta_mean": mean_ta,
        "ta_distribution": dist_with_context,
        "s_plus_ta_mean": np.concatenate([global_s, mean_ta], axis=1),
        "s_plus_ta_distribution": np.concatenate([global_s, dist_with_context], axis=1),
    }
    scores = {}
    for name, matrix in views.items():
        scores[name] = _fit_score(
            matrix[train],
            y[train],
            matrix[valid],
            y[valid],
            ta_params,
            model_seeds,
        )
        print(
            f"{name}: dim={matrix.shape[1]} valid={scores[name]['mean_auc']:.6f}",
            flush=True,
        )
    result = {
        "protocol_id": "luyin16-clean-local-distribution-readout-probe-v1",
        "official_test_evaluated": False,
        "data": {
            "dataset": "ogbg-molhiv",
            "n_graphs": int(indices.size),
            "n_train": int(train_count),
            "n_valid": int(indices.size - train_count),
            "global_s_source": str(old_path),
        },
        "representation": dict(representation),
        "local_blocks": list(LOCAL_BLOCKS),
        "statistics": list(STAT_NAMES),
        "views": {
            name: {"dimension": int(matrix.shape[1]), **scores[name]}
            for name, matrix in views.items()
        },
        "params": ta_params,
        "model_seeds": [int(value) for value in model_seeds],
    }
    result_dir.mkdir(parents=True, exist_ok=True)
    _write_json(result_dir / "summary.json", result)
    lines = [
        "# Clean local patch distribution readout probe",
        "",
        f"Protocol: `{result['protocol_id']}`",
        "",
        "Official test was not encoded or evaluated.",
        "",
        "| view | dim | valid ROC-AUC |",
        "|---|---:|---:|",
    ]
    for name, row in result["views"].items():
        lines.append(f"| `{name}` | {row['dimension']} | {row['mean_auc']:.6f} ± {row['std_auc']:.6f} |")
    lines.extend(
        [
            "",
            "The distribution view changes only the graph-level readout of the same clean centre-level marginals.",
            "The global `S` block is a control for information budget, not part of the invariant local object.",
        ]
    )
    (result_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()
    run(config_path=args.config, result_dir=args.result_dir, model_seeds=args.seeds)


if __name__ == "__main__":
    main()
