#!/usr/bin/env python3
"""Evaluate no-handcrafted-ring semantic/structural crosses on ogbg-molhiv.

The fixed no-ring backbone is the historical 693-D ``composition +
recon_typed`` representation.  It is augmented with NCI1-style interactions
between compact atom semantics in each rooted patch and the absolute sparse
K-SVD code of that patch.  No enumerated/artificial ring-context cache is read.

The original OGB 9-D atom attributes are left intact.  Consequently the
standard ``aromatic`` and ``is_in_ring`` atom flags remain inside the compact
48-D atom semantics; only the separately engineered ring5/ring6/aromatic-ring/
multi-ring/ring-boundary context blocks are excluded.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path

import numpy as np


SEMANTIC_DIM = 48
EXPECTED_CAPACITY = 13
DEFAULT_ATOMS = 64
DEFAULT_SPARSITY = 8
BACKBONE_DIM = 69 + 624

BASE_MODE = "STA_no_ring_base"
CROSS_MODES = {
    "STA_no_ring_patch_mass": ("patch_mass",),
    "STA_no_ring_root_cond": ("root_cond",),
    "STA_no_ring_mass_root": ("patch_mass", "root_cond"),
}
EVAL_MODES = {BASE_MODE: (), **CROSS_MODES}

BEST_PARAMS = {
    "colsample_bytree": 0.771539336853686,
    "gamma": 3.56280140950248e-05,
    "learning_rate": 0.025211848097051265,
    "max_bin": 128,
    "max_depth": 8,
    "min_child_weight": 8,
    "n_estimators": 900,
    "reg_alpha": 0.038496992268316245,
    "reg_lambda": 0.42691729197459727,
    "subsample": 0.8985617661211255,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path("results/molhiv_historical_m13_seed0")
    parser.add_argument("--patch-dir", type=Path, default=root / "cache_M13_q999")
    parser.add_argument("--code-dir", type=Path, default=root / "ksvd_K64_s8")
    parser.add_argument(
        "--diagnostic-dir", type=Path, default=root / "diagnostic_K64_s8"
    )
    parser.add_argument(
        "--reference-summary",
        type=Path,
        default=Path(
            "results/molhiv_historical_seed0_features_ablation_6views_xgb_10seed/"
            "auc_objective_summary.json"
        ),
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path("results/molhiv_no_ring_semantic_cross_v1"),
    )
    parser.add_argument(
        "--cross-dir",
        type=Path,
        default=Path(
            "results/molhiv_no_ring_semantic_cross_v1/cross_cache_K64"
        ),
        help="Large resumable cross caches; defaults to the E-drive result directory.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument(
        "--expected-atoms",
        type=int,
        default=DEFAULT_ATOMS,
        help="Expected K-SVD dictionary size (and semantic-cross atom axis).",
    )
    parser.add_argument(
        "--expected-sparsity",
        type=int,
        default=DEFAULT_SPARSITY,
        help="Expected maximum number of active dictionary atoms per patch.",
    )
    parser.add_argument("--modes", nargs="+", choices=tuple(EVAL_MODES), default=None)
    parser.add_argument(
        "--include-base",
        action="store_true",
        help="Also evaluate the capacity-specific 693-D no-ring STA backbone.",
    )
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--features-only", action="store_true")
    parser.add_argument(
        "--smoke-graphs",
        type=int,
        default=0,
        help="Validate only the first N graph encodings without writing full caches.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def write_json(path: Path, value: dict) -> None:
    def convert(item):
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, dict):
            return {str(key): convert(local) for key, local in item.items()}
        if isinstance(item, (list, tuple)):
            return [convert(local) for local in item]
        return item

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(convert(value), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def stats(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std_sample_ddof1": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "std_population_ddof0": float(array.std(ddof=0)),
        "min": float(array.min()),
        "median": float(np.median(array)),
        "max": float(array.max()),
    }


def require_array(path: Path, shape: tuple[int, ...], dtype) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = np.load(path, mmap_mode="r")
    if value.shape != shape or value.dtype != np.dtype(dtype):
        raise ValueError(
            f"Unexpected array {path}: shape={value.shape}, dtype={value.dtype}; "
            f"expected {shape}/{np.dtype(dtype)}"
        )
    return value


def load_sources(args: argparse.Namespace) -> dict:
    patch_meta = read_json(args.patch_dir / "metadata.json")
    layout = patch_meta["layout"]
    n_graphs = int(patch_meta["n_graphs"])
    n_patches = int(patch_meta["total_patches"])
    capacity = int(layout["capacity"])
    feature_dim = int(layout["feature_dim"])
    if capacity != EXPECTED_CAPACITY:
        raise ValueError(f"Expected capacity {EXPECTED_CAPACITY}, got {capacity}")
    if int(layout["atom_semantic_dim"]) != SEMANTIC_DIM:
        raise ValueError(f"Expected {SEMANTIC_DIM}-D atom semantics")

    patches = require_array(
        args.patch_dir / "patches_uint8.npy", (n_patches, feature_dim), np.uint8
    )
    offsets = require_array(
        args.patch_dir / "graph_offsets.npy", (n_graphs + 1,), np.int64
    )
    dataset_indices = require_array(
        args.patch_dir / "selected_dataset_indices.npy", (n_graphs,), np.int64
    )
    patch_split = require_array(
        args.patch_dir / "graph_split.npy", (n_graphs,), np.uint8
    )
    if int(offsets[0]) != 0 or int(offsets[-1]) != n_patches:
        raise ValueError("Invalid graph offsets")
    if np.any(np.diff(offsets) <= 0):
        raise ValueError("Every graph must own at least one rooted patch")

    expected_atoms = int(args.expected_atoms)
    expected_sparsity = int(args.expected_sparsity)
    if expected_atoms <= 0 or not 0 < expected_sparsity <= expected_atoms:
        raise ValueError(
            f"Invalid K/sparsity: {expected_atoms}/{expected_sparsity}"
        )

    code_indices = require_array(
        args.code_dir / "code_indices.npy",
        (n_patches, expected_sparsity),
        np.int16,
    )
    code_values = require_array(
        args.code_dir / "code_values.npy",
        (n_patches, expected_sparsity),
        np.float32,
    )
    code_nnz = require_array(
        args.code_dir / "code_nnz.npy", (n_patches,), np.uint8
    )
    if int(np.max(code_nnz)) > expected_sparsity:
        raise ValueError("Sparse code nnz exceeds configured sparsity")

    diagnostic_path = args.diagnostic_dir / "diagnostic_metadata.npz"
    with np.load(diagnostic_path) as archive:
        diagnostic_indices = np.asarray(archive["dataset_indices"], dtype=np.int64)
        graph_split = np.asarray(archive["graph_split"], dtype=np.uint8)
        labels = np.asarray(archive["labels"], dtype=np.int64).reshape(-1)
        composition = np.asarray(archive["composition"], dtype=np.float32)
        n_atoms = int(archive["n_atoms"])
        sparsity = int(archive["sparsity"])
        diagnostic_capacity = int(archive["capacity"])
    recon = np.load(
        args.diagnostic_dir / "recon_decoded_typed_pool.npy", mmap_mode="r"
    )
    if composition.shape != (n_graphs, 69):
        raise ValueError(f"Unexpected composition shape {composition.shape}")
    if recon.shape != (n_graphs, 624) or recon.dtype != np.float32:
        raise ValueError(f"Unexpected recon_typed array {recon.shape}/{recon.dtype}")
    if labels.shape != (n_graphs,) or not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("Expected one binary label per graph")
    if n_atoms != expected_atoms or sparsity != expected_sparsity:
        raise ValueError(f"Unexpected K/sparsity: {n_atoms}/{sparsity}")
    if diagnostic_capacity != capacity:
        raise ValueError("Diagnostic and patch capacities differ")
    if not np.array_equal(dataset_indices, diagnostic_indices):
        raise ValueError("Patch and diagnostic dataset-index orders differ")
    if not np.array_equal(patch_split, graph_split):
        raise ValueError("Patch and diagnostic split flags differ")

    split_counts = {int(flag): int(np.sum(graph_split == flag)) for flag in (0, 1, 2)}
    if split_counts != {0: 32901, 1: 4113, 2: 4113}:
        raise ValueError(f"Unexpected official split counts: {split_counts}")
    if not np.all(np.isfinite(composition)) or not np.all(np.isfinite(recon)):
        raise FloatingPointError("Backbone features contain non-finite values")

    atom_start, atom_end = (int(value) for value in layout["blocks"]["atom"])
    mask_start, mask_end = (int(value) for value in layout["blocks"]["mask"])
    if atom_end - atom_start != capacity * SEMANTIC_DIM:
        raise ValueError("Unexpected atom block layout")
    if mask_end - mask_start != capacity:
        raise ValueError("Unexpected mask block layout")

    return {
        "patch_meta": patch_meta,
        "n_graphs": n_graphs,
        "n_patches": n_patches,
        "capacity": capacity,
        "n_atoms": expected_atoms,
        "sparsity": expected_sparsity,
        "patches": patches,
        "offsets": offsets,
        "code_indices": code_indices,
        "code_values": code_values,
        "code_nnz": code_nnz,
        "dataset_indices": np.asarray(dataset_indices),
        "graph_split": graph_split,
        "labels": labels,
        "composition": composition,
        "recon": recon,
        "atom_slice": slice(atom_start, atom_end),
        "mask_slice": slice(mask_start, mask_end),
    }


def graph_crosses(source: dict, graph_id: int) -> tuple[np.ndarray, np.ndarray]:
    start = int(source["offsets"][graph_id])
    end = int(source["offsets"][graph_id + 1])
    patch_rows = np.asarray(source["patches"][start:end])
    count = end - start
    capacity = int(source["capacity"])

    atom = patch_rows[:, source["atom_slice"]].reshape(
        count, capacity, SEMANTIC_DIM
    ).astype(np.float32)
    mask = patch_rows[:, source["mask_slice"]].astype(bool)
    valid_counts = mask.sum(axis=1)
    if np.any(valid_counts <= 0) or not np.all(mask[:, 0]):
        raise ValueError(f"Invalid atom mask in graph {graph_id}")
    patch_semantic = atom.sum(axis=1) / valid_counts[:, None]
    root_semantic = atom[:, 0, :]

    indices = np.asarray(source["code_indices"][start:end], dtype=np.int64)
    values = np.abs(np.asarray(source["code_values"][start:end], dtype=np.float32))
    nnz = np.asarray(source["code_nnz"][start:end], dtype=np.int64)
    sparsity = int(source["sparsity"])
    n_atoms = int(source["n_atoms"])
    active = np.arange(sparsity)[None, :] < nnz[:, None]
    if np.any(indices[active] < 0) or np.any(indices[active] >= n_atoms):
        raise ValueError(f"Invalid sparse atom index in graph {graph_id}")
    dense_codes = np.zeros((count, n_atoms), dtype=np.float32)
    rows = np.broadcast_to(np.arange(count)[:, None], indices.shape)
    dense_codes[rows[active], indices[active]] = values[active]

    raw_patch = patch_semantic.T @ dense_codes
    raw_root = root_semantic.T @ dense_codes
    patch_mass = (raw_patch / count).reshape(-1).astype(np.float32)
    root_cond = (
        raw_root / (root_semantic.sum(axis=0)[:, None] + 1e-8)
    ).reshape(-1).astype(np.float32)
    if not np.all(np.isfinite(patch_mass)) or not np.all(np.isfinite(root_cond)):
        raise FloatingPointError(f"Non-finite cross in graph {graph_id}")
    return patch_mass, root_cond


def prepare_cross(path: Path, shape: tuple[int, int], resume: bool) -> np.ndarray:
    if resume and path.exists():
        value = np.load(path, mmap_mode="r+")
        if value.shape != shape or value.dtype != np.float32:
            raise ValueError(f"Invalid resumable cache {path}: {value.shape}/{value.dtype}")
        return value
    return np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)


def encode_crosses(source: dict, args: argparse.Namespace) -> dict[str, np.ndarray]:
    cross_dim = SEMANTIC_DIM * int(source["n_atoms"])
    if args.smoke_graphs:
        limit = min(int(args.smoke_graphs), int(source["n_graphs"]))
        maxima = {"patch_mass": 0.0, "root_cond": 0.0}
        for graph_id in range(limit):
            values = graph_crosses(source, graph_id)
            for name, value in zip(maxima, values):
                if value.shape != (cross_dim,):
                    raise ValueError(f"Unexpected {name} shape {value.shape}")
                maxima[name] = max(maxima[name], float(np.max(np.abs(value))))
        print(f"smoke encoding passed for {limit} graphs; max_abs={maxima}", flush=True)
        return {}

    args.cross_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.cross_dir / "encoding_progress.json"
    next_graph = 0
    if args.resume and progress_path.exists():
        progress = read_json(progress_path)
        next_graph = int(progress.get("next_graph", 0))
        missing = [
            name
            for name in ("patch_mass", "root_cond")
            if not (args.cross_dir / f"{name}.npy").exists()
        ]
        if next_graph > 0 and missing:
            raise FileNotFoundError(
                f"Checkpoint is at graph {next_graph}, but cross caches are missing: {missing}"
            )
    arrays = {
        name: prepare_cross(
            args.cross_dir / f"{name}.npy",
            (int(source["n_graphs"]), cross_dim),
            args.resume,
        )
        for name in ("patch_mass", "root_cond")
    }
    if not 0 <= next_graph <= int(source["n_graphs"]):
        raise ValueError(f"Invalid encoding checkpoint: {next_graph}")

    for graph_id in range(next_graph, int(source["n_graphs"])):
        patch_mass, root_cond = graph_crosses(source, graph_id)
        arrays["patch_mass"][graph_id] = patch_mass
        arrays["root_cond"][graph_id] = root_cond
        completed = graph_id + 1
        if completed % args.checkpoint_every == 0 or completed == source["n_graphs"]:
            for value in arrays.values():
                value.flush()
            write_json(
                progress_path,
                {
                    "next_graph": completed,
                    "n_graphs": int(source["n_graphs"]),
                    "cross_dim": cross_dim,
                },
            )
            print(f"encoded {completed}/{source['n_graphs']} graphs", flush=True)
    return arrays


def split_slice(flags: np.ndarray, flag: int) -> slice:
    positions = np.flatnonzero(flags == flag)
    if len(positions) == 0 or not np.array_equal(
        positions, np.arange(positions[0], positions[-1] + 1)
    ):
        raise ValueError("Expected each official split to be a contiguous block")
    return slice(int(positions[0]), int(positions[-1]) + 1)


def make_split_matrix(
    source: dict, crosses: dict[str, np.ndarray], names: tuple[str, ...], local: slice
) -> np.ndarray:
    rows = local.stop - local.start
    total_dim = BACKBONE_DIM + len(names) * SEMANTIC_DIM * int(source["n_atoms"])
    matrix = np.empty((rows, total_dim), dtype=np.float32)
    cursor = 0
    for block in (source["composition"], source["recon"]):
        width = int(block.shape[1])
        matrix[:, cursor : cursor + width] = block[local]
        cursor += width
    for name in names:
        block = crosses[name]
        width = int(block.shape[1])
        matrix[:, cursor : cursor + width] = block[local]
        cursor += width
    if cursor != total_dim or not np.all(np.isfinite(matrix)):
        raise ValueError("Failed to construct a finite split feature matrix")
    return matrix


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict]) -> None:
    ordered = sorted(rows, key=lambda row: (row["mode"], int(row["seed"])))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("mode", "seed", "train_rocauc", "valid_rocauc", "test_rocauc"),
        )
        writer.writeheader()
        writer.writerows(ordered)


def evaluate(source: dict, crosses: dict[str, np.ndarray], args: argparse.Namespace) -> None:
    from ogb.graphproppred import Evaluator
    from xgboost import XGBClassifier

    import run_molhiv_shared_atom_classification_old_protocol as protocol

    seeds = list(dict.fromkeys(int(seed) for seed in args.seeds))
    if len(seeds) != len(args.seeds):
        raise ValueError("Duplicate seeds are not allowed")
    modes = args.modes or list(CROSS_MODES)
    if args.include_base and BASE_MODE not in modes:
        modes = [BASE_MODE, *modes]
    labels = source["labels"]
    train_slice = split_slice(source["graph_split"], 0)
    valid_slice = split_slice(source["graph_split"], 1)
    test_slice = split_slice(source["graph_split"], 2)
    y_train, y_valid, y_test = (
        labels[train_slice],
        labels[valid_slice],
        labels[test_slice],
    )
    evaluator = Evaluator(name="ogbg-molhiv")
    score = lambda y, p: protocol.official_rocauc(evaluator, y, p)
    fixed = protocol.common_xgb_params(
        class_weight=protocol.positive_class_weight(y_train), n_jobs=args.n_jobs
    )

    predictions_dir = args.result_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    result_csv = args.result_dir / "per_seed_results.csv"
    rows = read_rows(result_csv) if args.resume else []
    done = {(row["mode"], int(row["seed"])) for row in rows}

    reference = read_json(args.reference_summary)["per_cell"]["st::xgb_logistic"]
    summary = {
        "dataset": "ogbg-molhiv",
        "protocol": "fixed official scaffold train/valid/test; no retuning",
        "seeds": seeds,
        "baseline": {
            "name": "historical_no_ring_composition_plus_recon_typed",
            "feature_dim": BACKBONE_DIM,
            "source": str(args.reference_summary.resolve()),
            **reference,
        },
        "views": {},
        "selection_rule": "Cross variants are selected by validation mean only; test is reported transparently.",
    }

    for mode in modes:
        names = EVAL_MODES[mode]
        print(f"building split matrices for {mode}", flush=True)
        x_train = make_split_matrix(source, crosses, names, train_slice)
        x_valid = make_split_matrix(source, crosses, names, valid_slice)
        x_test = make_split_matrix(source, crosses, names, test_slice)
        valid_predictions, test_predictions = [], []
        for seed in seeds:
            prediction_path = predictions_dir / f"{mode}_seed{seed}.npz"
            if (mode, seed) in done and prediction_path.exists():
                with np.load(prediction_path) as archive:
                    valid_pred = np.asarray(archive["valid"], dtype=np.float32)
                    test_pred = np.asarray(archive["test"], dtype=np.float32)
            else:
                model = XGBClassifier(
                    **BEST_PARAMS, **fixed, random_state=seed
                )
                model.fit(x_train, y_train)
                train_pred = model.predict_proba(x_train)[:, 1]
                valid_pred = model.predict_proba(x_valid)[:, 1]
                test_pred = model.predict_proba(x_test)[:, 1]
                record = {
                    "mode": mode,
                    "seed": seed,
                    "train_rocauc": score(y_train, train_pred),
                    "valid_rocauc": score(y_valid, valid_pred),
                    "test_rocauc": score(y_test, test_pred),
                }
                rows = [
                    row
                    for row in rows
                    if not (row["mode"] == mode and int(row["seed"]) == seed)
                ]
                rows.append(record)
                write_rows(result_csv, rows)
                np.savez_compressed(
                    prediction_path,
                    valid=valid_pred.astype(np.float32),
                    test=test_pred.astype(np.float32),
                )
                done.add((mode, seed))
                print(
                    f"mode={mode} seed={seed:02d} valid={record['valid_rocauc']:.6f} "
                    f"test={record['test_rocauc']:.6f}",
                    flush=True,
                )
                del model, train_pred
            valid_predictions.append(valid_pred)
            test_predictions.append(test_pred)

        local_rows = [row for row in rows if row["mode"] == mode]
        valid_values = [float(row["valid_rocauc"]) for row in local_rows]
        test_values = [float(row["test_rocauc"]) for row in local_rows]
        summary["views"][mode] = {
            "blocks": list(names),
            "feature_dim": int(x_train.shape[1]),
            "valid": stats(valid_values),
            "test": stats(test_values),
            "mean_prediction_valid_rocauc": score(
                y_valid, np.mean(valid_predictions, axis=0)
            ),
            "mean_prediction_test_rocauc": score(
                y_test, np.mean(test_predictions, axis=0)
            ),
            "beats_80_49_test_mean": float(np.mean(test_values)) > float(reference["test"]["mean"]),
        }
        write_json(args.result_dir / "summary.json", summary)
        del x_train, x_valid, x_test, valid_predictions, test_predictions
        gc.collect()

    winner = max(summary["views"], key=lambda name: summary["views"][name]["valid"]["mean"])
    summary["best_by_validation"] = {
        "mode": winner,
        "valid_mean": summary["views"][winner]["valid"]["mean"],
        "test_mean": summary["views"][winner]["test"]["mean"],
        "beats_80_49_test_mean": summary["views"][winner]["beats_80_49_test_mean"],
    }
    write_json(args.result_dir / "summary.json", summary)


def main() -> None:
    args = parse_args()
    args.patch_dir = args.patch_dir.resolve()
    args.code_dir = args.code_dir.resolve()
    args.diagnostic_dir = args.diagnostic_dir.resolve()
    args.reference_summary = args.reference_summary.resolve()
    args.result_dir = args.result_dir.resolve()
    args.cross_dir = args.cross_dir.resolve()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    source = load_sources(args)
    manifest = {
        "dataset": "ogbg-molhiv",
        "official_split_counts": {"train": 32901, "valid": 4113, "test": 4113},
        "backbone": "composition(69) + recon_typed(624)",
        "backbone_dim": BACKBONE_DIM,
        "semantic_dim": SEMANTIC_DIM,
        "ksvd_atoms": int(args.expected_atoms),
        "ksvd_sparsity": int(args.expected_sparsity),
        "cross_dim_per_block": SEMANTIC_DIM * int(args.expected_atoms),
        "definitions": {
            "patch_mass": "mean_patch_semantic.T @ abs(sparse_code) / number_of_rooted_patches",
            "root_cond": "root_atom_semantic.T @ abs(sparse_code) / sum(root_atom_semantic)",
        },
        "explicit_ring_context_used": False,
        "excluded_engineered_contexts": [
            "ring_any", "ring5", "ring6", "aromatic_ring", "multi_ring", "ring_boundary"
        ],
        "raw_ogb_atom_aromatic_and_in_ring_flags_retained": True,
        "patch_dir": args.patch_dir,
        "code_dir": args.code_dir,
        "diagnostic_dir": args.diagnostic_dir,
        "cross_dir": args.cross_dir,
        "modes": (
            args.modes
            or ([BASE_MODE] if args.include_base else [])
            + list(CROSS_MODES)
        ),
        "seeds": args.seeds,
        "best_params": BEST_PARAMS,
    }
    write_json(args.result_dir / "manifest.json", manifest)
    crosses = encode_crosses(source, args)
    if args.smoke_graphs:
        return
    if not args.features_only:
        evaluate(source, crosses, args)


if __name__ == "__main__":
    main()
