"""Structure caches, split views and target handling for the seven-task round.

The graph/cache layer is the frozen ``lcrc_cross_task_preparation_v1`` one:

    SMILES -> frozen raw attributes -> MolecularGraph -> build_molecule -> concat_molecules

with the ``lipo_v1`` FeatureSpec unchanged.  v2 adds:

* one cache per (dataset, split) for all seven tasks, including ``ogbg-molhiv``;
* a *multi-task* target container: Tox21 keeps ``NaN`` for unassayed cells and
  exposes an observed-mask instead of imputing a zero;
* a train-only target normaliser for regression and a train-only per-endpoint
  positive rate record for diagnostics (never used as a loss weight).

Every cache is label-free: it holds structure tensors plus row ids only.  The
targets live in a separate in-memory object and, for the one-shot evaluation, in
``test_eval.json``.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import mb2_common as C  # noqa: I001  (adds the frozen prep code dirs to sys.path)
import mb2_raw_artifact as RAW
import mb2_rdkit as R
import mb2_source as S

import lcrc_common as PREP_COMMON
import lcrc_data as PREP_DATA
from feature_spec import FeatureSpec

STRUCTURE_CACHE_VERSION = "lcrc-molnet-v2"


# ---------------------------------------------------------------------------
# feature spec
# ---------------------------------------------------------------------------


def frozen_spec() -> FeatureSpec:
    """Load the frozen ``lipo_v1`` FeatureSpec and prove the code reproduces it."""
    from feature_spec import lipo_v1_spec, load_spec

    if not C.FROZEN_SPEC_JSON.exists():
        raise FileNotFoundError(f"frozen FeatureSpec {C.track_rel(C.FROZEN_SPEC_JSON)} is missing")
    payload = C.read_json(C.FROZEN_SPEC_JSON)
    spec = load_spec(C.FROZEN_SPEC_JSON)
    recorded = payload.get("spec_hash")
    if spec.hash() != recorded:
        raise ValueError(f"loaded lipo_v1 spec hash {spec.hash()} != recorded {recorded}")
    rebuilt = lipo_v1_spec(payload.get("source_provenance", {}))
    if rebuilt.hash() != recorded:
        raise ValueError(
            "rebuilding lipo_v1 from the imported code no longer reproduces the frozen spec: "
            f"{rebuilt.hash()} != {recorded}"
        )
    return spec


# ---------------------------------------------------------------------------
# cache construction
# ---------------------------------------------------------------------------


def cache_path(dataset: str, split: str) -> Path:
    return C.CACHE / f"{dataset}_{split}.pt"


def load_raw_rows(dataset: str, rows: Sequence[int]) -> list[dict[str, Any]]:
    """Frozen raw attributes for the requested global row ids (no RDKit needed)."""
    return RAW.load_rows(dataset, rows, verify=True)


def build_split_cache(
    dataset: str,
    split: str,
    rows: Sequence[int],
    smiles: Sequence[str],
    spec: FeatureSpec,
    *,
    save: bool = True,
) -> dict[str, Any]:
    """Build one split's label-free structure cache."""
    if split not in ("train", "valid", "test"):
        raise ValueError("split must be train/valid/test")
    rows = [int(r) for r in rows]
    t0 = time.perf_counter()
    raw_rows = load_raw_rows(dataset, rows)
    for raw, row in zip(raw_rows, rows):
        if str(raw["smiles"]) != str(smiles[row]):
            raise ValueError(
                f"{dataset}/{split}: artifact SMILES for row {row} differs from the published source"
            )
    graphs = [R.raw_row_to_graph(raw, int(raw["mol_index"]), spec) for raw in raw_rows]
    cache = PREP_DATA.build_cache(graphs, spec, split=f"{dataset}_{split}")
    cache["rows"] = np.asarray(rows, dtype=np.int64)
    cache["dataset"] = dataset
    cache["split_name"] = split
    cache["cache_version"] = STRUCTURE_CACHE_VERSION
    cache["build_seconds"] = float(time.perf_counter() - t0)
    cache["meta"]["labels_present"] = False
    cache["meta"]["row_ids"] = "global 0-based source row index"
    PREP_DATA.validate_cache(cache, spec)
    cache["meta"]["content_fingerprint"] = PREP_COMMON.content_fingerprint(cache)
    if cache["molecule_ids"] != [str(r) for r in rows]:
        raise ValueError(f"{dataset}/{split}: molecule_ids do not match the source row ids")
    if save:
        C.ensure_dir(C.CACHE)
        torch.save(cache, cache_path(dataset, split))
    return cache


def load_or_build_split_cache(
    dataset: str,
    split: str,
    rows: Sequence[int],
    smiles: Sequence[str],
    spec: FeatureSpec,
    *,
    rebuild: bool = False,
) -> dict[str, Any]:
    path = cache_path(dataset, split)
    if path.exists() and not rebuild:
        cache = torch.load(path, map_location="cpu", weights_only=False)
        if cache.get("cache_version") != STRUCTURE_CACHE_VERSION:
            raise ValueError(f"{path}: unexpected cache version {cache.get('cache_version')!r}")
        if list(np.asarray(cache["rows"]).tolist()) != [int(r) for r in rows]:
            raise ValueError(f"{path}: cached row ids differ from the frozen split")
        PREP_DATA.validate_cache(cache, spec)
        return cache
    return build_split_cache(dataset, split, rows, smiles, spec)


# ---------------------------------------------------------------------------
# targets
# ---------------------------------------------------------------------------


def target_normaliser(y: np.ndarray) -> dict[str, Any]:
    """Train-only target standardisation for regression tasks."""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if not np.isfinite(y).all():
        raise ValueError("regression target contains non-finite values")
    mean = float(y.mean())
    scale = float(y.std(ddof=0))
    if not np.isfinite(mean) or not np.isfinite(scale) or scale <= 0:
        raise ValueError("degenerate train target distribution")
    return {
        "mean": mean,
        "scale": scale,
        "n": int(y.size),
        "source": "train_split_only",
        "valid_or_test_used": False,
        "formula": "y_scaled = (y - mean)/scale; metrics reported after y = pred*scale + mean",
    }


class Targets:
    """Task-aware target container (1 task for regression/binary, 12 for Tox21)."""

    def __init__(self, task: str, y_train: np.ndarray, y_valid: np.ndarray, *, task_names: Sequence[str] | None = None):
        self.task = task
        self.y_train = np.asarray(y_train, dtype=np.float64)
        self.y_valid = np.asarray(y_valid, dtype=np.float64)
        if self.y_train.ndim != 2:
            raise ValueError("targets must be 2-D [n_rows, n_tasks]")
        self.n_tasks = int(self.y_train.shape[1])
        self.task_names = list(task_names) if task_names else [f"task{i}" for i in range(self.n_tasks)]
        if self.task == "multitask":
            if self.n_tasks != 12:
                raise ValueError("Tox21 must expose exactly 12 endpoints")
        elif self.n_tasks != 1:
            raise ValueError(f"{task} must expose exactly 1 output")
        self.train_mask = np.isfinite(self.y_train)
        self.valid_mask = np.isfinite(self.y_valid)

    # -- train-side ------------------------------------------------------
    def loss_reference(self, target_norm: Mapping[str, Any] | None) -> np.ndarray:
        if self.task == "regression":
            if target_norm is None:
                raise ValueError("regression requires a target normaliser")
            return (self.y_train - float(target_norm["mean"])) / float(target_norm["scale"])
        return self.y_train  # binary / multitask: raw 0-1, NaN kept for missing

    def audit(self, split: str) -> dict[str, Any]:
        y = self.y_train if split == "train" else self.y_valid
        mask = self.train_mask if split == "train" else self.valid_mask
        out: dict[str, Any] = {"n_rows": int(y.shape[0]), "n_tasks": self.n_tasks}
        if self.task == "multitask":
            per_task = {}
            for i, name in enumerate(self.task_names):
                obs = mask[:, i]
                pos = int(np.nansum(y[obs, i]))
                per_task[name] = {
                    "n_observed": int(obs.sum()),
                    "n_missing": int((~obs).sum()),
                    "n_positive": pos,
                    "n_negative": int(obs.sum()) - pos,
                    "positive_rate": float(pos / max(obs.sum(), 1)),
                    "both_classes_present": bool(pos > 0 and (obs.sum() - pos) > 0),
                }
            out["per_task"] = per_task
            out["n_observed_cells"] = int(mask.sum())
            out["n_missing_cells"] = int((~mask).sum())
            out["missing_fraction"] = float((~mask).mean())
            out["tasks_with_both_classes"] = int(sum(v["both_classes_present"] for v in per_task.values()))
            out["rows_without_any_observation"] = int((~mask.any(axis=1)).sum())
            out["missing_policy"] = "NaN kept as missing; never treated as negative"
        else:
            flat = y.reshape(-1)
            out.update(
                {
                    "min": float(np.nanmin(flat)),
                    "max": float(np.nanmax(flat)),
                    "mean": float(np.nanmean(flat)),
                    "std": float(np.nanstd(flat)),
                }
            )
            if self.task == "binary":
                values = np.unique(flat)
                out["class_counts"] = {str(int(v)): int((flat == v).sum()) for v in values}
                out["both_classes_present"] = bool(values.size == 2)
                out["positive_rate"] = float((flat == 1).sum() / max(flat.size, 1))
        return out


# ---------------------------------------------------------------------------
# views
# ---------------------------------------------------------------------------


class DatasetView:
    """Everything a training run needs: structure caches, targets, normalisers."""

    def __init__(
        self,
        *,
        dataset: str,
        spec: FeatureSpec,
        train_cache: Mapping[str, Any],
        valid_cache: Mapping[str, Any],
        targets: Targets,
        train_norm: Mapping[str, Any],
        target_norm: Mapping[str, Any] | None,
        rows: Mapping[str, Sequence[int]],
        label_columns: Sequence[str],
        units: str,
        source_meta: Mapping[str, Any],
    ) -> None:
        self.dataset = dataset
        self.spec = spec
        self.train_cache = train_cache
        self.valid_cache = valid_cache
        self.targets = targets
        self.task = targets.task
        self.train_norm = train_norm
        self.target_norm = target_norm
        self.rows = {k: list(v) for k, v in rows.items()}
        self.label_columns = list(label_columns)
        self.units = units
        self.source_meta = dict(source_meta)

    @property
    def task_names(self) -> list[str]:
        return self.targets.task_names

    def audit(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "task": self.task,
            "label_columns": self.label_columns,
            "units": self.units,
            "n_train": int(self.train_cache["n_molecules"]),
            "n_valid": int(self.valid_cache["n_molecules"]),
            "train_cache_fingerprint": self.train_cache["meta"]["content_fingerprint"],
            "valid_cache_fingerprint": self.valid_cache["meta"]["content_fingerprint"],
            "train_norm_fingerprint": PREP_DATA.norm_fingerprint(self.train_norm),
            "target_norm": None if self.target_norm is None else dict(self.target_norm),
            "rows": {k: {"n": len(v), "sha256": C.row_sha256(v)} for k, v in self.rows.items()},
            "targets": {"train": self.targets.audit("train"), "valid": self.targets.audit("valid")},
            "source": self.source_meta,
            "feature_spec": self.spec.name,
            "feature_spec_hash": self.spec.hash(),
            "labels_in_structure_cache": False,
        }


def build_view(dataset: str, *, spec: FeatureSpec | None = None, rebuild: bool = False) -> DatasetView:
    """Build/load the train+valid view.  The test split is never loaded here."""
    meta = C.dataset_entry(dataset)
    spec = spec or frozen_spec()
    source = S.load_dataset(dataset, labels="train_valid")
    rows = source["rows"]
    train_cache = load_or_build_split_cache(
        dataset, "train", rows["train"], source["smiles"], spec, rebuild=rebuild
    )
    valid_cache = load_or_build_split_cache(
        dataset, "valid", rows["valid"], source["smiles"], spec, rebuild=rebuild
    )
    targets = Targets(
        meta["task"],
        source["y"]["train"],
        source["y"]["valid"],
        task_names=C.TOX21_TASKS if meta["task"] == "multitask" else meta["label_columns"],
    )
    train_norm = PREP_DATA.fit_norm(train_cache, spec, source=f"{dataset}_train_structures_only")
    target_norm = None
    if meta["task"] == "regression":
        target_norm = target_normaliser(targets.y_train)
    if dataset == "ogbg-molhiv":
        source_meta = {
            "source_kind": "ogb",
            "csv_sha256": source["csv_sha256"],
            "split_file": "split/scaffold/{train,valid,test}.csv.gz",
            "split_file_sha256": source["files"]["split/scaffold/train.csv.gz"]["sha256"],
            "split_rows_sha256": [C.row_sha256(rows[k]) for k in ("train", "valid", "test")],
            "split_method": meta["split_method"],
            "split_repeat_index": None,
            "label_agreement": source["label_agreement"],
            "ogb_release": "v1 (2020-05-04)",
        }
    else:
        source_meta = {
            "source_kind": "chembench",
            "source_commit": source["source_commit"],
            "csv_sha256": source["csv_sha256"],
            "split_file": source["selected_repeat"]["file"],
            "split_file_sha256": source["selected_repeat"]["sha256"],
            "split_rows_sha256": source["selected_repeat"]["rows_sha256"],
            "split_method": meta["split_method"],
            "split_repeat_index": C.SPLIT_REPEAT_INDEX,
        }
    return DatasetView(
        dataset=dataset,
        spec=spec,
        train_cache=train_cache,
        valid_cache=valid_cache,
        targets=targets,
        train_norm=train_norm,
        target_norm=target_norm,
        rows={"train": rows["train"], "valid": rows["valid"], "test": rows["test"]},
        label_columns=meta["label_columns"],
        units=meta["units"],
        source_meta=source_meta,
    )


def build_test_structure_cache(dataset: str, *, spec: FeatureSpec | None = None, rebuild: bool = False) -> dict[str, Any]:
    """Build the label-free test structure cache (no target column is decoded)."""
    spec = spec or frozen_spec()
    source = S.load_dataset(dataset, labels="none")
    rows = source["rows"]["test"]
    return load_or_build_split_cache(dataset, "test", rows, source["smiles"], spec, rebuild=rebuild)


def load_test_cache(
    dataset: str, *, spec: FeatureSpec | None = None, cache: Mapping[str, Any] | None = None
) -> tuple[dict[str, Any], np.ndarray, list[int]]:
    """Load the test structure cache *and its targets*.  One-shot evaluation only."""
    spec = spec or frozen_spec()
    if cache is None:
        cache = build_test_structure_cache(dataset, spec=spec)
    source = S.load_dataset(dataset, labels="all")
    rows = source["rows"]["test"]
    if [int(x) for x in np.asarray(cache["rows"]).tolist()] != [int(x) for x in rows]:
        raise ValueError("test cache row ids differ from the frozen published split")
    return cache, source["y"]["test"], list(rows)


# ---------------------------------------------------------------------------
# batching / prediction helpers
# ---------------------------------------------------------------------------


def to_device(batch: Mapping[str, Any], device) -> dict[str, Any]:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def predict_matrix(model, cache: Mapping[str, Any], device, *, batch_size: int = 256) -> np.ndarray:
    """Raw model output for every molecule in cache: shape ``[n_molecules, n_outputs]``."""
    model.eval()
    n = int(cache["n_molecules"])
    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            rows = list(range(start, min(start + batch_size, n)))
            batch = to_device(PREP_DATA.collate(cache, rows), device)
            pred = model(batch)
            if pred.ndim == 1:
                pred = pred.reshape(-1, 1)
            out.append(pred.detach().float().cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float64)


def predict_scores(model, cache: Mapping[str, Any], device, *, batch_size: int = 256) -> np.ndarray:
    """Single-output convenience wrapper: 1-D raw output."""
    mat = predict_matrix(model, cache, device, batch_size=batch_size)
    if mat.shape[1] != 1:
        raise ValueError(f"expected a single output, got {mat.shape[1]}")
    return mat[:, 0]


__all__ = [
    "STRUCTURE_CACHE_VERSION",
    "frozen_spec",
    "cache_path",
    "load_raw_rows",
    "build_split_cache",
    "load_or_build_split_cache",
    "build_test_structure_cache",
    "target_normaliser",
    "Targets",
    "DatasetView",
    "build_view",
    "load_test_cache",
    "to_device",
    "predict_matrix",
    "predict_scores",
]
