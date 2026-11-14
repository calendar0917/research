"""Structure caches, train-only normalisation, batching, per-epoch evaluation.

The structure pipeline is the frozen ``lcrc_cross_task_preparation_v1`` one
(``lipo_v1`` :class:`FeatureSpec` -> ``MolecularGraph`` -> ``build_cache``); only
the row order is new: every cache row is one *official OGB row id*
(``mol.csv.gz`` order) and the cache's ``molecule_ids`` record that mapping.

Caches are label-free (``labels_present = False``); the target array is loaded
separately from ``data/labels_v1.npz`` and sliced by the official split rows.

Everything here is shared by all six arms, so every arm sees exactly the same
structures, normalisation, target scaling and epoch plan.
"""

from __future__ import annotations

import math
import os
import random
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import lp_common as C
import lp_source as S

import adapters as PREP_ADAPTERS
import lcrc_common as PREP_COMMON
import lcrc_data as PREP_DATA
from feature_spec import FeatureSpec

CACHE_VERSION = "lcrc-lipo-context-small-s0-v1"


# ---------------------------------------------------------------------------
# feature spec
# ---------------------------------------------------------------------------


def frozen_spec() -> FeatureSpec:
    """Load the frozen ``lipo_v1`` spec and prove the code reproduces its hash."""
    import mb_data as D

    spec = D.frozen_spec()
    if spec.hash() != C.FEATURE_SPEC_HASH:
        raise ValueError(f"lipo_v1 spec hash {spec.hash()} != frozen {C.FEATURE_SPEC_HASH}")
    if (spec.atom_feat_dim, spec.bond_feat_dim, spec.n_scalars, spec.global_dim) != (
        C.ATOM_FEAT_DIM,
        C.BOND_FEAT_DIM,
        C.ATOM_SCALAR_DIM,
        C.GLOBAL_DIM,
    ):
        raise ValueError("frozen lipo_v1 input widths changed")
    return spec


# ---------------------------------------------------------------------------
# caches
# ---------------------------------------------------------------------------


def cache_path(split: str) -> Path:
    return C.CACHE / f"mollipo_{split}_v1.pt"


def _atomic_save(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def build_split_cache(split: str, spec: FeatureSpec, *, save: bool = True) -> dict[str, Any]:
    if split not in C.SPLITS:
        raise ValueError(split)
    split_payload = S.load_split()
    rows = [int(r) for r in np.asarray(split_payload["rows"][split]).tolist()]
    raw_rows = S.load_raw_artifact(verify=True)
    graphs = []
    for row in rows:
        item = raw_rows[row]
        if int(item["mol_index"]) != row:
            raise ValueError(f"raw row {row} has mol_index {item['mol_index']}")
        graphs.append(PREP_ADAPTERS.lipo_row_to_graph(item, row, spec))
    cache = PREP_DATA.build_cache(graphs, spec, split=f"{C.DATASET}_{split}")
    cache["rows"] = np.asarray(rows, dtype=np.int64)
    cache["dataset"] = C.DATASET
    cache["split_name"] = split
    cache["cache_version"] = CACHE_VERSION
    cache["meta"]["labels_present"] = False
    cache["meta"]["row_ids"] = "official OGB mol.csv.gz row id (0-based)"
    PREP_DATA.validate_cache(cache, spec)
    if cache["molecule_ids"] != [str(r) for r in rows]:
        raise ValueError(f"{split}: cache molecule_ids do not match the official row ids")
    cache["meta"]["content_fingerprint"] = PREP_COMMON.content_fingerprint(cache)
    if save:
        _atomic_save(cache, cache_path(split))
    return cache


def load_or_build_split_cache(split: str, spec: FeatureSpec, *, rebuild: bool = False) -> dict[str, Any]:
    path = cache_path(split)
    if not rebuild and path.is_file():
        cache = torch.load(path, weights_only=False)
        if cache.get("cache_version") != CACHE_VERSION:
            raise ValueError(f"{path}: cache version {cache.get('cache_version')} is not this round")
        PREP_DATA.validate_cache(cache, spec)
        expected = [int(x) for x in np.asarray(S.load_split()["rows"][split]).tolist()]
        if [int(x) for x in np.asarray(cache["rows"]).tolist()] != expected:
            raise ValueError(f"{path}: rows drifted from the official split")
        return cache
    return build_split_cache(split, spec)


# ---------------------------------------------------------------------------
# normalisation / view
# ---------------------------------------------------------------------------


def target_normaliser(y_train: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y_train, dtype=np.float64)
    mean = float(y.mean())
    scale = float(y.std())
    if not (scale > 0):
        raise ValueError("degenerate train target")
    return {
        "mean": mean,
        "scale": scale,
        "source": "official Train rows only",
        "valid_or_test_used": False,
        "n": int(y.size),
    }


class View:
    """All shared, frozen inputs of one round (caches, norms, rows, targets)."""

    def __init__(self, *, spec, caches, train_norm, target_norm, rows, y):
        self.spec = spec
        self.caches = dict(caches)
        self.train_norm = dict(train_norm)
        self.target_norm = dict(target_norm)
        self.rows = {k: np.asarray(v, dtype=np.int64) for k, v in rows.items()}
        self.y = {k: np.asarray(v, dtype=np.float64).reshape(-1) for k, v in y.items()}

    # -- convenience ------------------------------------------------------
    def cache(self, split: str) -> Mapping[str, Any]:
        return self.caches[split]

    def y_scaled(self, split: str) -> torch.Tensor:
        y = self.y[split]
        return torch.as_tensor((y - self.target_norm["mean"]) / self.target_norm["scale"], dtype=torch.float32)

    def to_raw(self, standardised: np.ndarray) -> np.ndarray:
        return np.asarray(standardised, dtype=np.float64) * float(self.target_norm["scale"]) + float(
            self.target_norm["mean"]
        )

    def audit(self) -> dict[str, Any]:
        return {
            "dataset": C.DATASET,
            "split_method": C.OGB_SPLIT_METHOD,
            "sizes": {k: int(self.caches[k]["n_molecules"]) for k in C.SPLITS},
            "cache_fingerprints": {k: self.caches[k]["meta"]["content_fingerprint"] for k in C.SPLITS},
            "row_sha256": {k: C.row_sha256(self.rows[k].tolist()) for k in C.SPLITS},
            "train_norm_fingerprint": PREP_DATA.norm_fingerprint(self.train_norm),
            "train_norm_source": self.train_norm.get("source"),
            "target_norm": dict(self.target_norm),
            "feature_spec_hash": self.spec.hash(),
            "labels_in_structure_cache": False,
        }


def build_view(*, rebuild: bool = False) -> View:
    spec = frozen_spec()
    split_payload = S.load_split()
    labels = S.load_labels()
    caches = {k: load_or_build_split_cache(k, spec, rebuild=rebuild) for k in C.SPLITS}
    for k in C.SPLITS:
        expected = [int(x) for x in np.asarray(split_payload["rows"][k]).tolist()]
        got = [int(x) for x in np.asarray(caches[k]["rows"]).tolist()]
        if got != expected:
            raise ValueError(f"{k}: cache rows != official split rows")
    train_norm = PREP_DATA.fit_norm(caches["train"], spec, source=f"{C.DATASET}_official_train_structures_only")
    if train_norm.get("valid_or_test_used") is not False:
        raise ValueError("train normalisation must not use Valid/Test")
    y = {k: labels["y"][split_payload["rows"][k]] for k in C.SPLITS}
    target_norm = target_normaliser(y["train"])
    return View(
        spec=spec,
        caches=caches,
        train_norm=train_norm,
        target_norm=target_norm,
        rows=split_payload["rows"],
        y=y,
    )


# ---------------------------------------------------------------------------
# epoch plan (one shared, pre-generated row order for every C1-family arm)
# ---------------------------------------------------------------------------


def epoch_plan(n: int, epochs: int, seed: int, *, save: bool = True) -> dict[str, Any]:
    """The exact per-epoch ``randperm`` stream of the frozen training loop."""
    path = C.CACHE / f"epoch_plan_s{int(seed)}_n{int(n)}_e{int(epochs)}.npz"
    if path.is_file():
        with np.load(path, allow_pickle=False) as handle:
            plan = np.asarray(handle["plan"], dtype=np.int32)
        if plan.shape != (epochs, n):
            raise ValueError(f"{path}: plan shape {plan.shape} != {(epochs, n)}")
    else:
        generator = torch.Generator().manual_seed(int(seed) + int(C.SHUFFLE_OFFSET))
        rows = np.empty((epochs, n), dtype=np.int32)
        for epoch in range(epochs):
            rows[epoch] = torch.randperm(n, generator=generator).numpy().astype(np.int32)
        plan = rows
        if save:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, plan=plan)
    digest = C.sha256_bytes(np.ascontiguousarray(plan).tobytes())
    return {
        "plan": plan,
        "sha256": digest,
        "shape": [int(epochs), int(n)],
        "seed": int(seed),
        "shuffle_offset": int(C.SHUFFLE_OFFSET),
        "generator": "torch.Generator().manual_seed(seed + 917000); one randperm per epoch",
        "path": C.track_rel(path) if path.is_file() else None,
    }


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def regression_metrics(pred: np.ndarray, y: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    err = pred - y
    ss_res = float((err * err).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {
        "n": int(y.size),
        "rmse": float(math.sqrt(ss_res / max(y.size, 1))),
        "mae": float(np.abs(err).mean()),
        "bias": float(err.mean()),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
    }


def raw_metrics(standardised: np.ndarray, y: np.ndarray, target_norm: Mapping[str, Any]) -> dict[str, float]:
    pred = np.asarray(standardised, dtype=np.float64) * float(target_norm["scale"]) + float(target_norm["mean"])
    return regression_metrics(pred, y)


# ---------------------------------------------------------------------------
# batching / prediction with RNG isolation
# ---------------------------------------------------------------------------


def to_device(batch: Mapping[str, Any], device) -> dict[str, Any]:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def rng_snapshot(device) -> dict[str, Any]:
    snap = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": None,
    }
    if device.type == "cuda" and torch.cuda.is_available():
        snap["torch_cuda"] = torch.cuda.get_rng_state(device).clone()
    return snap


def rng_restore(snap: Mapping[str, Any], device) -> None:
    random.setstate(snap["python"])
    np.random.set_state(snap["numpy"])
    torch.set_rng_state(snap["torch_cpu"])
    if snap["torch_cuda"] is not None and device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.set_rng_state(snap["torch_cuda"], device)


def predict_cache(
    model,
    cache: Mapping[str, Any],
    device,
    *,
    rows: Sequence[int] | None = None,
    batch_size: int = C.EVAL_BATCH_SIZE,
    batch_fn=None,
    pred_fn=None,
) -> np.ndarray:
    """Full-split forward with an independent iterator (no dropout, no RNG use)."""
    n = int(cache["n_molecules"])
    order = list(range(n)) if rows is None else [int(r) for r in rows]
    out: list[np.ndarray] = []
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for start in range(0, len(order), batch_size):
                chunk = order[start : start + batch_size]
                if batch_fn is None:
                    batch = to_device(PREP_DATA.collate(cache, chunk), device)
                else:
                    batch = batch_fn(chunk)
                pred = model(batch) if pred_fn is None else pred_fn(batch)
                out.append(pred.detach().float().cpu().numpy().reshape(-1))
    finally:
        model.train(was_training)
    return np.concatenate(out).astype(np.float64)


def evaluate_all_splits(
    predict_split,
    view: View,
    device,
    *,
    keep_predictions: bool = False,
) -> tuple[dict[str, dict[str, float]], dict[str, np.ndarray]]:
    """One full Train/Valid/Test pass at the current weights.

    The evaluation takes an RNG snapshot before and restores it afterwards, so
    it can never consume the training row order or any other stochastic state;
    each ``predict_split`` is responsible for restoring the ``train()/eval()``
    flag of its own model.
    """
    snap = rng_snapshot(device)
    metrics: dict[str, dict[str, float]] = {}
    preds: dict[str, np.ndarray] = {}
    try:
        for split in C.SPLITS:
            standardised = predict_split(split)
            metrics[split] = raw_metrics(standardised, view.y[split], view.target_norm)
            if keep_predictions:
                preds[split] = standardised
    finally:
        rng_restore(snap, device)
    return metrics, preds


def state_hash(state: Mapping[str, torch.Tensor]) -> str:
    import hashlib

    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode("utf-8"))
        h.update(np.ascontiguousarray(value.detach().cpu().numpy(), dtype=np.float32).tobytes())
    return h.hexdigest()


__all__ = [name for name in dir() if not name.startswith("_")]
