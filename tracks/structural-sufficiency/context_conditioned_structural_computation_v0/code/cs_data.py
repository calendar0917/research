"""Data layer for CSC v0: the frozen small-task LCRC pipeline plus relations.

Everything about the graphs, the official scaffold split, the ``lipo_v1``
FeatureSpec, the Train-only feature/target normalisation and the collate layer is
*read-only reuse* of ``lcrc_readout_context_pilot_small_v1`` (which itself reuses
``tracks/structural-query-learning`` and ``lcrc_cross_task_preparation_v1``).
This module adds only:

1. byte-level re-verification of the four frozen caches and the split identity,
2. the relation grouping of real directed bonds (SINGLE..AROMATIC, OTHER), with
   its mapping re-checked against the raw ``edge_class`` artefact,
3. the shared per-epoch batch-order plan (one draw per epoch, every arm identical),
4. full-split evaluation in raw label units.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

import cs_common as C

C.setup_paths()

import cr_data as D  # noqa: E402  (read-only reuse of the small-task pipeline)
import cr_common as RC  # noqa: E402


# ---------------------------------------------------------------------------
# bundle
# ---------------------------------------------------------------------------


def load_bundle(dataset: str, *, verify: bool = True) -> Any:
    """Build the frozen bundle and re-check every identity this round relies on."""
    if dataset not in C.DATASETS:
        raise KeyError(dataset)
    bundle = D.build_bundle(dataset, verify=bool(verify))
    audit = verify_bundle(bundle)
    bundle.extra["identity_audit"] = audit
    return bundle


def verify_bundle(bundle: Any) -> dict[str, Any]:
    """Split sizes, cache bytes, split identity and relation mapping."""
    dataset = str(bundle.dataset)
    sizes = {split: int(bundle.caches[split]["n_molecules"]) for split in C.SPLITS}
    if sizes != C.EXPECTED_SPLIT_SIZES[dataset]:
        raise RuntimeError(f"{dataset}: split sizes {sizes} != {C.EXPECTED_SPLIT_SIZES[dataset]}")
    rows = {split: np.asarray(bundle.global_row_ids[split], dtype=np.int64) for split in C.SPLITS}
    all_rows = np.sort(np.concatenate([rows[s] for s in C.SPLITS]))
    if not np.array_equal(all_rows, np.arange(all_rows.size)):
        raise RuntimeError(f"{dataset}: official splits do not partition 0..N-1")
    if len({int(x) for x in all_rows.tolist()}) != all_rows.size:
        raise RuntimeError(f"{dataset}: duplicate official row id across splits")

    files = {}
    for name, expected in C.EXPECTED_CACHE_SHA256.items():
        if dataset.replace("-", "_") not in name:
            continue
        path = None
        for candidate in (C.SQL_DIR / "data" / name,):
            if candidate.is_file():
                path = candidate
                break
        if path is None:
            files[name] = {"present": False, "expected_sha256": expected}
            continue
        actual = RC.sha256_file(path)
        files[name] = {
            "present": True,
            "path": C.track_rel(path),
            "expected_sha256": expected,
            "actual_sha256": actual,
            "match": bool(actual == expected),
            "bytes": int(path.stat().st_size),
        }
        if actual != expected:
            raise RuntimeError(f"{name}: cache fingerprint {actual} != {expected}")

    rel = verify_relations(bundle)
    return {
        "dataset": dataset,
        "split_sizes": sizes,
        "split_row_id_sha256": {s: C.sha256_array(rows[s]) for s in C.SPLITS},
        "split_partition_sha256": C.sha256_array(np.concatenate([rows[s] for s in C.SPLITS])),
        "cache_files": files,
        "feature_spec_hash": bundle.extra.get("spec_hash"),
        "norm_fingerprint": bundle.extra.get("norm_fingerprint"),
        "target_scaler": {k: bundle.scaler[k] for k in ("mean", "std", "n", "ddof")},
        "relation_mapping": rel,
        "n_atoms": {s: int(bundle.n_atoms[s].sum()) for s in C.SPLITS},
    }


def verify_relations(bundle: Any) -> dict[str, Any]:
    """Check the relation vocabulary against the one-hot bond class and the raw artefact.

    ``lipo_v1`` encodes ``bond_class`` as the first five columns of ``bond_feat``
    (SINGLE, DOUBLE, TRIPLE, AROMATIC, UNK) and ``bond_type`` as the raw
    ``edge_class + 1``.  Both statements are assertions here, not assumptions.
    """
    spec = bundle.spec
    if tuple(spec.bond_type_names) != ("SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "UNK"):
        raise RuntimeError(f"unexpected bond vocabulary {spec.bond_type_names}")
    n_types = int(spec.n_bond_types)
    if n_types != 5 or C.N_RELATIONS != 5:
        raise RuntimeError("relation count does not match the bond vocabulary")

    counts = np.zeros(5, dtype=np.int64)
    n_bonds = 0
    raw_checked = 0
    raw_mismatch = 0
    endpoint_mismatch = 0
    onehot_ok = True
    argmax_ok = True
    raw = bundle.lipo_raw
    edge_ptr = np.asarray(raw["edge_ptr"], dtype=np.int64)
    edge_u = np.asarray(raw["edge_u"], dtype=np.int64)
    edge_v = np.asarray(raw["edge_v"], dtype=np.int64)
    edge_class = np.asarray(raw["edge_class"], dtype=np.int64)
    raw_class_counts = np.zeros(5, dtype=np.int64)
    for split in C.SPLITS:
        cache = bundle.caches[split]
        bf = np.asarray(cache["bond_feat"])
        bt = np.asarray(cache["bond_type"], dtype=np.int64)
        if bt.size:
            block = bf[:, :5]
            onehot_ok = onehot_ok and bool(np.all((block == 0.0) | (block == 1.0)))
            onehot_ok = onehot_ok and bool(np.all(block.sum(axis=1) == 1.0))
            argmax_ok = argmax_ok and bool(np.all(block.argmax(axis=1) + 1 == bt))
            counts += np.bincount(np.clip(bt - 1, 0, 4), minlength=5)
            n_bonds += int(bt.size)
        bond_ptr = np.asarray(cache["bond_ptr"], dtype=np.int64)
        bond_index = np.asarray(cache["bond_index"], dtype=np.int64)
        for i in range(int(cache["n_molecules"])):
            row = int(bundle.global_row_ids[split][i])
            r0, r1 = int(edge_ptr[row]), int(edge_ptr[row + 1])
            # the adapter deduplicates the directed records into sorted physical bonds
            pairs: dict[tuple[int, int], int] = {}
            for k in range(r0, r1):
                key = (min(int(edge_u[k]), int(edge_v[k])), max(int(edge_u[k]), int(edge_v[k])))
                cls = int(edge_class[k])
                if key in pairs and pairs[key] != cls:
                    raise RuntimeError(f"{bundle.dataset}: inconsistent raw directed records for {key}")
                pairs[key] = cls
            keys = sorted(pairs)
            c0, c1 = int(bond_ptr[i]), int(bond_ptr[i + 1])
            if len(keys) != c1 - c0:
                raise RuntimeError(
                    f"{bundle.dataset}: molecule {row} raw physical bonds {len(keys)} != cache {c1 - c0}"
                )
            raw_checked += len(keys)
            for pos, key in enumerate(keys):
                cls = pairs[key]
                expected = cls + 1 if cls <= 3 else 5
                raw_class_counts[int(expected) - 1] += 1
                if expected != int(bt[c0 + pos]):
                    raw_mismatch += 1
                if tuple(int(x) for x in bond_index[c0 + pos]) != key:
                    endpoint_mismatch += 1
    if not onehot_ok or not argmax_ok:
        raise RuntimeError("bond_class one-hot / bond_type mapping is not exact")
    if raw_mismatch or endpoint_mismatch:
        raise RuntimeError(
            f"{bundle.dataset}: raw artefact mismatch (bond_type {raw_mismatch}, endpoints {endpoint_mismatch} "
            f"of {raw_checked} bonds)"
        )
    return {
        "bond_type_names": list(spec.bond_type_names),
        "relation_names": list(C.RELATION_NAMES),
        "bond_type_to_relation": {str(k): int(v) for k, v in C.BOND_TYPE_TO_RELATION.items()},
        "bond_class_one_hot_exact": True,
        "bond_type_equals_argmax_plus_one": True,
        "raw_physical_bonds_checked": int(raw_checked),
        "raw_edge_class_mismatches": int(raw_mismatch),
        "raw_endpoint_mismatches": int(endpoint_mismatch),
        "raw_edge_class_counts_by_bond_type": {str(i + 1): int(raw_class_counts[i]) for i in range(5)},
        "bond_counts_by_bond_type": {str(i + 1): int(counts[i]) for i in range(5)},
        "bond_counts_by_relation": {name: int(counts[i]) for i, name in enumerate(C.RELATION_NAMES)},
        "n_bonds": int(n_bonds),
    }


# ---------------------------------------------------------------------------
# batch order
# ---------------------------------------------------------------------------


def batch_order_plan(dataset: str, epochs: int | None = None) -> dict[str, Any]:
    """One frozen permutation per epoch, shared by every arm of this dataset."""
    n_train = C.EXPECTED_SPLIT_SIZES[dataset]["train"]
    epochs = int(C.RECIPE["epochs"] if epochs is None else epochs)
    rng = np.random.default_rng(C.SEED)
    rows = np.arange(n_train, dtype=np.int64)
    plan = [rng.permutation(rows) for _ in range(epochs)]
    return {
        "generator": "np.random.default_rng(0), one permutation per epoch over 0..n_train-1",
        "n_train": int(n_train),
        "epochs": int(epochs),
        "batch_size": int(C.RECIPE["batch_size"]),
        "plan_sha256": C.sha256_array(np.concatenate(plan)),
        "per_epoch_sha256": [C.sha256_array(p) for p in plan],
        "plan": plan,
    }


# ---------------------------------------------------------------------------
# collate / batches
# ---------------------------------------------------------------------------


def make_batch(bundle: Any, split: str, rows: np.ndarray, device=None) -> dict[str, Any]:
    import torch

    batch = bundle.collate(bundle.caches[split], np.asarray(rows, dtype=np.int64))
    if device is not None:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    return batch


def relation_index(batch: Mapping[str, Any]) -> np.ndarray:
    """Directed-bond relation id (0..4) for the ``bond_index`` rows, from ``bond_type``."""
    bt = np.asarray(batch["bond_type"].detach().cpu().numpy() if hasattr(batch["bond_type"], "detach") else batch["bond_type"])
    return np.asarray([C.BOND_TYPE_TO_RELATION[int(t)] for t in bt.tolist()], dtype=np.int64)


def attach_target(batch: dict[str, Any], rows: np.ndarray, y_std: np.ndarray):
    import torch

    batch["y"] = torch.as_tensor(np.asarray(y_std, dtype=np.float32)[np.asarray(rows, dtype=np.int64)], dtype=torch.float32)
    return batch


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------


def split_order(bundle: Any, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cache-local order, official global row ids and raw targets in ascending row-id order."""
    order, row_ids = bundle.row_ids_sorted(split)
    ids = np.asarray(row_ids, dtype=np.int64)
    return np.asarray(order, dtype=np.int64), ids, np.asarray([bundle.targets_raw[split][int(o)] for o in order], dtype=np.float64)


def evaluate(bundle: Any, model, split: str, device, *, batch_size: int | None = None) -> dict[str, Any]:
    """Full split in eval mode; raw label units; rows in ascending official row id."""
    import torch

    order, row_ids, y_raw = split_order(bundle, split)
    batch_size = int(C.RECIPE["eval_batch_size"] if batch_size is None else batch_size)
    preds: list[np.ndarray] = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for start in range(0, order.size, batch_size):
            chunk = order[start:start + batch_size]
            batch = make_batch(bundle, split, chunk, device)
            pred = model(batch)
            preds.append(pred.detach().cpu().numpy().astype(np.float64).reshape(-1))
    if was_training:
        model.train()
    pred_std = np.concatenate(preds) if preds else np.zeros(0, dtype=np.float64)
    mean, std = float(bundle.scaler["mean"]), float(bundle.scaler["std"])
    pred_raw = pred_std * std + mean
    return {
        "split": split,
        "row_ids": row_ids,
        "y_raw": y_raw,
        "pred_std": pred_std,
        "pred_raw": pred_raw,
        "rmse_raw": C.rmse(pred_raw, y_raw),
        "mae_raw": C.mae(pred_raw, y_raw),
        "n_rows": int(row_ids.size),
    }


__all__ = [
    "D",
    "RC",
    "attach_target",
    "batch_order_plan",
    "evaluate",
    "load_bundle",
    "make_batch",
    "relation_index",
    "split_order",
    "verify_bundle",
    "verify_relations",
]
