"""Structure caches, split views and train-only normalisation for the pilot.

The pipeline is the frozen ``lcrc_cross_task_preparation_v1`` one:

    SMILES -> raw attributes -> MolecularGraph -> build_molecule -> concat_molecules

with the ``lipo_v1`` FeatureSpec unchanged.  The only additions here are
per-dataset cache files, an explicit row-id ledger, a train-only target
normaliser for regression, and a regression/classification aware view object.

Every cache is label-free: a cache contains structure tensors plus row ids, and
the target vector is stored in a *separate* file.  This is asserted in the P0
audit.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import mb_common as C  # noqa: I001  (adds the frozen prep code dir to sys.path)
import mb_raw_artifact as RAW
import mb_rdkit as R
import mb_source as S

import lcrc_common as PREP_COMMON
import lcrc_data as PREP_DATA
from feature_spec import FeatureSpec

STRUCTURE_CACHE_VERSION = "lcrc-molnet-pilot-v1"


# ---------------------------------------------------------------------------
# feature spec
# ---------------------------------------------------------------------------


def frozen_spec() -> FeatureSpec:
    """Load the frozen ``lipo_v1`` FeatureSpec and prove the code reproduces it.

    The spec is loaded from the committed upstream JSON (which is the artifact
    the Lipophilicity results were produced with).  Rebuilding the spec *from
    code* with the recorded source provenance must reproduce the same hash;
    otherwise the imported feature-spec code has drifted and the round stops.
    """
    from feature_spec import lipo_v1_spec, load_spec

    if not C.FROZEN_SPEC_JSON.exists():
        raise FileNotFoundError(
            f"frozen FeatureSpec {C.track_rel(C.FROZEN_SPEC_JSON)} is missing; "
            "the round requires the frozen lipo_v1 spec artifact"
        )
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


def label_path(dataset: str) -> Path:
    return C.CACHE / f"{dataset}_labels.npz"


_ARTIFACT_CACHE: dict[str, Any] | None = None


def raw_artifact() -> dict[str, Any]:
    """Load the committed label-free raw-attribute artifact once per process."""
    global _ARTIFACT_CACHE
    if _ARTIFACT_CACHE is None:
        _ARTIFACT_CACHE = RAW.load(verify=True)
    return _ARTIFACT_CACHE


def load_raw_rows(dataset: str, rows: Sequence[int]) -> list[dict[str, Any]]:
    """Frozen raw attributes for the requested global row ids (no RDKit needed)."""
    payload = raw_artifact()
    if dataset not in payload["rows"]:
        raise KeyError(f"{dataset} not present in the raw attribute artifact")
    all_rows = payload["rows"][dataset]
    out: list[dict[str, Any]] = []
    for row in rows:
        r = int(row)
        if r < 0 or r >= len(all_rows):
            raise IndexError(f"{dataset}: row {r} outside the raw artifact")
        item = all_rows[r]
        if int(item["mol_index"]) != r:
            raise ValueError(f"{dataset}: artifact row {r} has mol_index {item['mol_index']}")
        out.append(item)
    return out


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
        # the artifact must describe exactly the SMILES that the split points at
        if str(raw["smiles"]) != str(smiles[row]):
            raise ValueError(
                f"{dataset}/{split}: artifact SMILES for row {row} differs from the published CSV"
            )
    graphs = [R.raw_row_to_graph(raw, int(raw["mol_index"]), spec) for raw in raw_rows]
    cache = PREP_DATA.build_cache(graphs, spec, split=f"{dataset}_{split}")
    cache["rows"] = np.asarray(rows, dtype=np.int64)
    cache["dataset"] = dataset
    cache["split_name"] = split
    cache["cache_version"] = STRUCTURE_CACHE_VERSION
    cache["source_commit"] = S.CHEMBENCH_COMMIT
    cache["build_seconds"] = float(time.perf_counter() - t0)
    cache["meta"]["labels_present"] = False
    cache["meta"]["row_ids"] = "global 0-based ChemBench CSV row index"
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
        return cache
    return build_split_cache(dataset, split, rows, smiles, spec)


# ---------------------------------------------------------------------------
# normalisation
# ---------------------------------------------------------------------------


def target_normaliser(y: np.ndarray) -> dict[str, Any]:
    """Train-only target standardisation for regression tasks."""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
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
        "formula": "target_standardised = (y - mean) / scale; metrics reported after y = pred*scale + mean",
    }


# ---------------------------------------------------------------------------
# views
# ---------------------------------------------------------------------------


class DatasetView:
    """Everything a run needs: structure caches, targets, normalisers."""

    def __init__(
        self,
        *,
        dataset: str,
        spec: FeatureSpec,
        train_cache: Mapping[str, Any],
        valid_cache: Mapping[str, Any],
        y_train: np.ndarray,
        y_valid: np.ndarray,
        train_norm: Mapping[str, Any],
        target_norm: Mapping[str, Any] | None,
        rows: Mapping[str, Sequence[int]],
        task: str,
        label_column: str,
        units: str,
        source_meta: Mapping[str, Any],
    ) -> None:
        self.dataset = dataset
        self.spec = spec
        self.train_cache = train_cache
        self.valid_cache = valid_cache
        self.y_train = np.asarray(y_train, dtype=np.float64)
        self.y_valid = np.asarray(y_valid, dtype=np.float64)
        self.train_norm = train_norm
        self.target_norm = target_norm
        self.rows = {k: list(v) for k, v in rows.items()}
        self.task = task
        self.label_column = label_column
        self.units = units
        self.source_meta = dict(source_meta)

    def audit(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "task": self.task,
            "label_column": self.label_column,
            "units": self.units,
            "n_train": int(self.train_cache["n_molecules"]),
            "n_valid": int(self.valid_cache["n_molecules"]),
            "train_cache_fingerprint": self.train_cache["meta"]["content_fingerprint"],
            "valid_cache_fingerprint": self.valid_cache["meta"]["content_fingerprint"],
            "train_norm_fingerprint": PREP_DATA.norm_fingerprint(self.train_norm),
            "target_norm": None if self.target_norm is None else dict(self.target_norm),
            "rows": {k: {"n": len(v), "sha256": C.row_sha256(v)} for k, v in self.rows.items()},
            "source": self.source_meta,
            "feature_spec": self.spec.name,
            "feature_spec_hash": self.spec.hash(),
            "labels_in_structure_cache": False,
        }


def build_view(
    dataset: str,
    *,
    spec: FeatureSpec | None = None,
    rebuild: bool = False,
) -> DatasetView:
    """Build/load the train+valid view.  The test split is never loaded here."""
    meta = C.DATASETS[dataset]
    spec = spec or frozen_spec()
    source = S.load_dataset(dataset, labels="train_valid")
    rows = source["rows"]
    train_cache = load_or_build_split_cache(
        dataset, "train", rows["train"], source["smiles"], spec, rebuild=rebuild
    )
    valid_cache = load_or_build_split_cache(
        dataset, "valid", rows["valid"], source["smiles"], spec, rebuild=rebuild
    )
    y_train = source["y"]["train"]
    y_valid = source["y"]["valid"]

    train_norm = PREP_DATA.fit_norm(train_cache, spec, source=f"{dataset}_train_structures_only")
    target_norm = None
    if meta["task"] == "regression":
        target_norm = target_normaliser(y_train)
    return DatasetView(
        dataset=dataset,
        spec=spec,
        train_cache=train_cache,
        valid_cache=valid_cache,
        y_train=y_train,
        y_valid=y_valid,
        train_norm=train_norm,
        target_norm=target_norm,
        rows={"train": rows["train"], "valid": rows["valid"], "test": rows["test"]},
        task=meta["task"],
        label_column=meta["label_column"],
        units=meta["units"],
        source_meta={
            "source_commit": source["source_commit"],
            "csv_sha256": source["csv_sha256"],
            "split_file": source["selected_repeat"]["file"],
            "split_file_sha256": source["selected_repeat"]["sha256"],
            "split_rows_sha256": source["selected_repeat"]["rows_sha256"],
            "split_method": meta["split_method"],
            "split_repeat_index": C.SPLIT_REPEAT_INDEX,
        },
    )


def build_test_structure_cache(dataset: str, *, spec: FeatureSpec | None = None, rebuild: bool = False) -> dict[str, Any]:
    """Build the label-free test structure cache (no target column is decoded)."""
    spec = spec or frozen_spec()
    source = S.load_dataset(dataset, labels="none")
    rows = source["rows"]["test"]
    return load_or_build_split_cache(dataset, "test", rows, source["smiles"], spec, rebuild=rebuild)


def load_test_cache(dataset: str, *, spec: FeatureSpec | None = None) -> tuple[dict[str, Any], np.ndarray, list[int]]:
    """Load the test structure cache *and its targets*.  P2 evaluation only."""
    spec = spec or frozen_spec()
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


def predict_scores(model, cache: Mapping[str, Any], device, *, batch_size: int = 256) -> np.ndarray:
    """Raw model output (regression space or logit) for every molecule in cache."""
    model.eval()
    n = int(cache["n_molecules"])
    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            rows = list(range(start, min(start + batch_size, n)))
            batch = to_device(PREP_DATA.collate(cache, rows), device)
            pred = model(batch)
            out.append(pred.detach().float().cpu().numpy().reshape(-1))
    return np.concatenate(out).astype(np.float64)


__all__ = [
    "STRUCTURE_CACHE_VERSION",
    "frozen_spec",
    "cache_path",
    "label_path",
    "raw_artifact",
    "load_raw_rows",
    "build_split_cache",
    "load_or_build_split_cache",
    "build_test_structure_cache",
    "target_normaliser",
    "DatasetView",
    "build_view",
    "load_test_cache",
    "to_device",
    "predict_scores",
]
