"""Structure caches and target views for *arbitrary, frozen* row lists.

Sprint 02's ``mb2_data.build_view`` is hard-wired to the published repeat 0 of
each dataset.  This round needs three other row lists:

* ``Lipop.random123`` / ``Lipop.random124`` (P1),
* the three BBBP inner-CV folds (P2),

and it must keep the Sprint-02 invariants while doing so:

* the graph path is unchanged (committed raw artifact -> frozen adapter ->
  ``lrc_data.build_molecule``), so a row built here is byte-identical to the
  same row in the Sprint-02 cache;
* every cache carries the exact global row ids it was built from, and the SMILES
  of each row is re-checked against the published source before building;
* the standardiser is fitted on that split's **Train** rows only, once;
* no Test label is decoded outside the explicit one-shot evaluation entry point.

Usage (library only).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import ta_common as C  # noqa: I001

import mb2_data as D
import mb2_raw_artifact as RAW
import mb2_rdkit as RDK
import mb2_source as S

import lcrc_data as PREP_DATA
from feature_spec import FeatureSpec

CACHE_VERSION = "lcrc-taskaware-v1"


# ---------------------------------------------------------------------------
# published split files for the extra Lipophilicity repeats
# ---------------------------------------------------------------------------


def _read_split_pkl(path: Path) -> list[list[int]]:
    import pandas as pd

    series = pd.read_pickle(path)
    parts = [list(part) for part in series.to_list()]
    if len(parts) != 3:
        raise ValueError(f"{path.name}: expected 3 split parts, got {len(parts)}")
    return [[int(x) for x in part] for part in parts]


def lipo_repeat_split(suffix: int) -> dict[str, Any]:
    """Read one published Lipophilicity repeat and verify it against Sprint 02.

    Sprint 02 recorded ``sha256`` and per-part ``rows_sha256`` for all three
    repeats before any label was read.  The bytes are the vendored Sprint-01
    ChemBench files; nothing is regenerated.
    """
    if suffix not in (122, 123, 124):
        raise ValueError("suffix must be one of 122/123/124")
    path = C.V1 / f"analysis/source/chembench/Lipop/Lipop.random{suffix}.ind.pkl"
    manifest = C.read_json(C.V2 / "analysis/split_manifest_Lipophilicity.json")
    selector = [
        {
            "file": manifest["selected_split_file"],
            "sha256": manifest["selected_split_sha256"],
            "sizes": [manifest["sizes"][k] for k in ("train", "valid", "test")],
            "rows_sha256": [manifest["rows_sha256"][k] for k in ("train", "valid", "test")],
        }
    ] + list(manifest.get("other_repeats", []))
    recorded = None
    for item in selector:
        if str(item.get("file", "")).endswith(f"random{suffix}.ind.pkl"):
            recorded = item
            break
    if recorded is None:  # pragma: no cover - the manifest always lists all three
        raise KeyError(f"Sprint 02 manifest does not list random{suffix}")
    actual_sha = C.sha256_file(path)
    if actual_sha != recorded["sha256"]:
        raise ValueError(f"random{suffix}: sha256 {actual_sha} != Sprint-02 recorded {recorded['sha256']}")
    train, valid, test = _read_split_pkl(path)
    sizes = (len(train), len(valid), len(test))
    if list(sizes) != list(recorded["sizes"]):
        raise ValueError(f"random{suffix}: sizes {sizes} != recorded {recorded['sizes']}")
    recorded_rows = recorded["rows_sha256"]
    for name, part, expected in zip(("train", "valid", "test"), (train, valid, test), recorded_rows):
        if C.row_sha(part) != expected:
            raise ValueError(f"random{suffix}/{name}: row hash does not match Sprint 02")
    if set(train) & set(valid) or set(train) & set(test) or set(valid) & set(test):
        raise ValueError(f"random{suffix}: split partitions overlap")
    return {
        "suffix": int(suffix),
        "file": path.name,
        "sha256": actual_sha,
        "rows": {"train": train, "valid": valid, "test": test},
        "rows_sha256": {k: C.row_sha(v) for k, v in (("train", train), ("valid", valid), ("test", test))},
        "sizes": {k: len(v) for k, v in (("train", train), ("valid", valid), ("test", test))},
        "recorded_in": "lcrc_molnet_benchmark_pilot_v2/analysis/split_manifest_Lipophilicity.json",
    }


def chembench_targets(dataset: str, splits: Mapping[str, Sequence[int]], *, include_test: bool = False) -> dict[str, Any]:
    """Targets for explicit row lists, from the frozen source CSV."""
    meta = C.dataset_entry(dataset)
    if meta["source"] != "chembench":
        raise ValueError(f"{dataset} is not a ChemBench dataset")
    task = meta["chembench_task"]
    import pandas as pd

    csv_path = S.chembench_path(f"{task}/{task}.csv.gz")
    df = pd.read_csv(csv_path)
    missing = [c for c in meta["label_columns"] if c not in df.columns]
    if missing:
        raise ValueError(f"{dataset}: label columns {missing} not in the CSV")
    raw = df[list(meta["label_columns"])].to_numpy(dtype=np.float64)
    if meta["task"] != "multitask":
        raw = np.asarray(raw, dtype=np.float64)
        if not np.isfinite(raw).all():
            raise ValueError(f"{dataset}: non-finite labels in the source CSV")
    else:
        observed = np.isfinite(raw)
        if observed.any():
            bad = observed & ~np.isin(raw, (0.0, 1.0))
            if bad.any():
                raise ValueError("non-binary observed label in Tox21")
    out = {"train": raw[np.asarray(splits["train"], dtype=np.int64)]}
    out["valid"] = raw[np.asarray(splits["valid"], dtype=np.int64)]
    if include_test:
        out["test"] = raw[np.asarray(splits["test"], dtype=np.int64)]
    return {"y": out, "csv_sha256": C.sha256_file(csv_path), "csv_path": C.track_rel(csv_path), "n_rows": int(len(df))}


def ogb_targets(dataset: str, splits: Mapping[str, Sequence[int]], *, include_test: bool = False) -> dict[str, Any]:
    """Official OGB labels sliced by explicit row ids (never re-ordered)."""
    if dataset != "ogbg-molhiv":
        raise ValueError("only ogbg-molhiv uses the OGB path")
    import pandas as pd

    graph_label = pd.read_csv(S.ogb_path("raw/graph-label.csv.gz"), header=None).to_numpy().reshape(-1, 1)
    mapping = pd.read_csv(S.ogb_path("mapping/mol.csv.gz"))
    same = np.asarray(mapping["HIV_active"].to_numpy(dtype=np.float64).reshape(-1, 1) == graph_label)
    if not same.all():
        raise ValueError("mapping HIV_active disagrees with raw/graph-label.csv.gz")
    full = np.asarray(graph_label, dtype=np.float64)
    out = {
        "train": full[np.asarray(splits["train"], dtype=np.int64)],
        "valid": full[np.asarray(splits["valid"], dtype=np.int64)],
    }
    if include_test:
        out["test"] = full[np.asarray(splits["test"], dtype=np.int64)]
    return {
        "y": out,
        "label_agreement": {"rows_compared": int(len(mapping)), "mismatches": int((~same).sum()), "identical": bool(same.all())},
        "csv_sha256": C.sha256_file(S.ogb_path("mapping/mol.csv.gz")),
    }


# ---------------------------------------------------------------------------
# cache construction for arbitrary row lists
# ---------------------------------------------------------------------------


def cache_path(dataset: str, tag: str, rows: Sequence[int]) -> Path:
    digest = C.row_sha(rows)[:12]
    return C.CACHE / f"{dataset}.{tag}.{digest}.pt"


def build_or_load_cache(
    dataset: str,
    rows: Sequence[int],
    *,
    tag: str,
    split_label: str,
    spec: FeatureSpec | None = None,
    rebuild: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build (or load) one label-free structure cache for an explicit row list."""
    spec = spec or D.frozen_spec()
    rows = [int(r) for r in rows]
    if len(set(rows)) != len(rows):
        raise ValueError(f"{dataset}/{tag}: duplicate rows")
    path = cache_path(dataset, tag, rows)
    info: dict[str, Any] = {
        "dataset": dataset,
        "tag": tag,
        "split_label": split_label,
        "path": C.track_rel(path),
        "rows_sha256": C.row_sha(rows),
        "n_rows": len(rows),
        "cache_hit": False,
    }
    if path.exists() and not rebuild:
        cache = torch.load(path, map_location="cpu", weights_only=False)
        if list(np.asarray(cache["rows"]).tolist()) != rows:
            raise ValueError(f"{path}: cached rows differ from the request")
        if cache.get("cache_version") != CACHE_VERSION:
            raise ValueError(f"{path}: unexpected cache version {cache.get('cache_version')!r}")
        PREP_DATA.validate_cache(cache, spec)
        info.update({"sha256": C.sha256_file(path), "content_fingerprint": cache["meta"]["content_fingerprint"], "cache_hit": True})
        return cache, info

    published_smiles = S.all_smiles(dataset)
    raw_rows = RAW.load_rows(dataset, rows, verify=True)
    for raw, row in zip(raw_rows, rows):
        if str(raw["smiles"]) != str(published_smiles[row]):
            raise ValueError(f"{dataset}/{tag}: artifact SMILES for row {row} differs from the published source")
    graphs = [RDK.raw_row_to_graph(raw, int(raw["mol_index"]), spec) for raw in raw_rows]
    cache = PREP_DATA.build_cache(graphs, spec, split=f"{dataset}_{split_label}")
    cache["rows"] = np.asarray(rows, dtype=np.int64)
    cache["dataset"] = dataset
    cache["split_name"] = split_label
    cache["cache_version"] = CACHE_VERSION
    cache["meta"]["labels_present"] = False
    cache["meta"]["row_ids"] = "global 0-based source row index"
    PREP_DATA.validate_cache(cache, spec)
    cache["meta"]["content_fingerprint"] = C.content_fingerprint(cache)
    if cache["molecule_ids"] != [str(r) for r in rows]:
        raise ValueError(f"{dataset}/{tag}: molecule_ids do not match the source row ids")
    C.CACHE.mkdir(parents=True, exist_ok=True)
    torch.save(cache, path)
    info.update({"sha256": C.sha256_file(path), "content_fingerprint": cache["meta"]["content_fingerprint"]})
    return cache, info


@dataclass
class View:
    dataset: str
    spec: FeatureSpec
    task: str
    train_cache: Mapping[str, Any]
    valid_cache: Mapping[str, Any]
    targets: D.Targets
    train_norm: Mapping[str, Any]
    target_norm: Mapping[str, Any] | None
    rows: dict[str, list[int]]
    label_columns: list[str]
    units: str
    split_meta: dict[str, Any]
    cache_meta: dict[str, Any]

    @property
    def task_names(self) -> list[str]:
        return self.targets.task_names

    def audit(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "task": self.task,
            "split_meta": self.split_meta,
            "cache_meta": self.cache_meta,
            "rows": {k: {"n": len(v), "sha256": C.row_sha(v)} for k, v in self.rows.items()},
            "train_norm": {
                "source": self.train_norm.get("source"),
                "fit_split": self.train_norm.get("fit_split"),
                "fingerprint": PREP_DATA.norm_fingerprint(self.train_norm),
                "scalar_constant_indices": list(self.train_norm.get("scalar_constant_indices", [])),
                "global_constant_indices": list(self.train_norm.get("global_constant_indices", [])),
                "zero_variance_policy": self.train_norm.get("zero_variance_policy"),
            },
            "target_norm": None if self.target_norm is None else dict(self.target_norm),
            "targets": {"train": self.targets.audit("train"), "valid": self.targets.audit("valid")},
            "feature_spec_hash": self.spec.hash(),
        }


def build_view_rows(
    dataset: str,
    rows: Mapping[str, Sequence[int]],
    *,
    tag: str,
    split_meta: Mapping[str, Any],
    spec: FeatureSpec | None = None,
) -> View:
    """Train+valid view over explicit row lists (Test is never built here)."""
    spec = spec or D.frozen_spec()
    meta = C.dataset_entry(dataset)
    train_cache, train_info = build_or_load_cache(dataset, rows["train"], tag=f"{tag}_train", split_label=f"{tag}_train", spec=spec)
    valid_cache, valid_info = build_or_load_cache(dataset, rows["valid"], tag=f"{tag}_valid", split_label=f"{tag}_valid", spec=spec)
    if dataset == "ogbg-molhiv":
        labelled = ogb_targets(dataset, rows, include_test=False)
        source_meta = {"source_kind": "ogb", "label_agreement": labelled.get("label_agreement")}
    else:
        labelled = chembench_targets(dataset, rows, include_test=False)
        source_meta = {"source_kind": "chembench", "csv_sha256": labelled["csv_sha256"], "csv": labelled["csv_path"]}
    targets = D.Targets(
        meta["task"],
        labelled["y"]["train"],
        labelled["y"]["valid"],
        task_names=C.V2C.TOX21_TASKS if meta["task"] == "multitask" else meta["label_columns"],
    )
    target_norm = D.target_normaliser(targets.y_train) if meta["task"] == "regression" else None
    train_norm = PREP_DATA.fit_norm(train_cache, spec, source=f"{dataset}_{tag}_train_structures_only")
    return View(
        dataset=dataset,
        spec=spec,
        task=meta["task"],
        train_cache=train_cache,
        valid_cache=valid_cache,
        targets=targets,
        train_norm=train_norm,
        target_norm=target_norm,
        rows={"train": [int(r) for r in rows["train"]], "valid": [int(r) for r in rows["valid"]]},
        label_columns=list(meta["label_columns"]),
        units=meta["units"],
        split_meta={**dict(split_meta), "source": source_meta},
        cache_meta={"train": train_info, "valid": valid_info},
    )


def build_test_view(
    dataset: str,
    test_rows: Sequence[int],
    *,
    tag: str,
    split_meta: Mapping[str, Any] | None = None,
    spec: FeatureSpec | None = None,
) -> tuple[dict[str, Any], np.ndarray, list[int]]:
    """One-shot Test structure cache *and* targets.

    Only called from ``ta_eval.py`` after ``analysis/checkpoint_manifest.json``
    exists; the caller records the access in ``test_access_ledger.json``.
    """
    spec = spec or D.frozen_spec()
    cache, info = build_or_load_cache(dataset, test_rows, tag=f"{tag}_test", split_label=f"{tag}_test", spec=spec)
    if dataset == "ogbg-molhiv":
        labelled = ogb_targets(dataset, {"train": [], "valid": [], "test": list(test_rows)}, include_test=True)
        y = labelled["y"]["test"]
    else:
        labelled = chembench_targets(dataset, {"train": [], "valid": [], "test": list(test_rows)}, include_test=True)
        y = labelled["y"]["test"]
    return cache, np.asarray(y, dtype=np.float64), [int(r) for r in test_rows]
