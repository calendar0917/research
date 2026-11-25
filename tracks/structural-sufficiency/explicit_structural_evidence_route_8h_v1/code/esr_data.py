"""Dataset views: frozen splits, label-free caches, Train-only normalisation.

Four datasets, all read-only inherited where possible:

* ``LIPO`` -- official-scaffold ``ogbg-mollipo`` caches of
  ``lcrc_lipo_context_small_s0_v1`` (3360/420/420), labels ``labels_v1.npz``;
* ``ESOL`` -- official-scaffold ``ogbg-molesol`` view of
  ``lcrc_explicit_semantic_learning_8h_s0_v1`` (902/113/113, label-free raw
  artifact built once with RDKit);
* ``TOX21`` -- the frozen ChemBench Tox21 caches of
  ``lcrc_molnet_benchmark_pilot_v2`` (also used by
  ``lcrc_predictive_trace_benchmark_v1``), 12 endpoints, NaN preserved;
* ``PCBA`` -- official OGB scaffold split through ``sg_data.load_task`` (350343 /
  43793 / 43793); the frozen label artifact is NaN on every Test row, so PCBA
  Test cannot be scored and is never indexed.

Data isolation rules enforced in code:

* the structural normaliser is fitted on train structures only;
* the target normaliser (regression) is Train-only;
* the auxiliary global statistics ``log1p(n_atoms)`` / ``log1p(n_pairs)`` are
  standardised on Train rows only;
* training views only expose ``train``/``valid``; ``include_test=True`` is used
  exclusively by the final one-shot scoring script.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import esr_common as C
import esr_pairs as P
import lcrc_data as PREP_DATA
import mb2_data as MD
import mb2_metrics as MM
import pc_metrics as PCM
import se_data as SED
import lp_data as LPD
import sg_data as SGD


class View:
    """Everything one dataset run needs; no Test entry unless requested."""

    def __init__(
        self,
        *,
        dataset: str,
        task: str,
        spec: Any,
        caches: Mapping[str, Mapping[str, Any]],
        pairs: Mapping[str, Mapping[str, Any]],
        train_norm: Mapping[str, Any],
        target_norm: Mapping[str, Any] | None,
        rows: Mapping[str, np.ndarray],
        ids: Mapping[str, np.ndarray],
        y: Mapping[str, np.ndarray],
        task_names: Sequence[str],
        extra_norm: Mapping[str, Any],
        source: Mapping[str, Any],
        relation_mode: str = "real",
        max_dist: int = 2,
        global_cache: bool = False,
    ):
        self.dataset = str(dataset)
        self.task = str(task)
        self.spec = spec
        self.caches = dict(caches)
        self.pairs = dict(pairs)
        self.train_norm = dict(train_norm)
        self.target_norm = None if target_norm is None else dict(target_norm)
        self.rows = {k: np.asarray(v, dtype=np.int64) for k, v in rows.items()}
        self.ids = {k: np.asarray(v, dtype=np.int64) for k, v in ids.items()}
        self.y = {k: np.asarray(v, dtype=np.float64) for k, v in y.items()}
        self.task_names = list(task_names)
        self.n_tasks = len(self.task_names)
        self.extra_norm = dict(extra_norm)
        self.source = dict(source)
        self.relation_mode = str(relation_mode)
        self.max_dist = int(max_dist)
        self.global_cache = bool(global_cache)
        for split in self.caches:
            cache = self.caches[split]
            if self.global_cache:
                if int(self.rows[split].max(initial=-1)) >= int(cache["n_molecules"]):
                    raise RuntimeError(f"{self.dataset}/{split}: row id outside the global cache")
                gids = np.asarray(cache["gids"]).astype(np.int64)
                if not np.array_equal(gids, np.arange(int(cache["n_molecules"]), dtype=np.int64)):
                    raise RuntimeError(f"{self.dataset}: global cache rows are not the official row order")
                if self.rows[split].size != self.y[split].shape[0]:
                    raise RuntimeError(f"{self.dataset}/{split}: target rows differ from split rows")
            else:
                if self.rows[split].size != int(cache["n_molecules"]):
                    raise RuntimeError(f"{self.dataset}/{split}: rows {self.rows[split].size} != cache molecules")
                if not np.array_equal(self.rows[split], np.arange(int(cache["n_molecules"]), dtype=np.int64)):
                    raise RuntimeError(f"{self.dataset}/{split}: split-local cache must be addressed by position")
                if [int(x) for x in np.asarray(cache["rows"]).tolist()] != [int(x) for x in self.ids[split].tolist()]:
                    raise RuntimeError(f"{self.dataset}/{split}: cache rows differ from the frozen split")
            if cache["meta"]["feature_spec_hash"] != spec.hash():
                raise RuntimeError(f"{self.dataset}/{split}: feature spec hash drift")
            if self.pairs and split in self.pairs:
                ident = self.pairs[split]["meta"]["cache_identity"]
                if int(ident["n_molecules"]) != int(cache["n_molecules"]):
                    raise RuntimeError(f"{self.dataset}/{split}: pair cache for another cache")

    # -- batch construction ------------------------------------------------
    def collate(self, split: str, rows: Sequence[int]) -> dict[str, Any]:
        batch = P.collate_with_pairs(
            self.caches[split], self.pairs[split], rows, relation_mode=self.relation_mode
        )
        batch["global_ext"] = P.batch_global_ext(batch, self.extra_norm)
        return batch

    def to_device(self, batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
        return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    def loss_target(self, split: str) -> torch.Tensor:
        y = self.y[split]
        if self.task == "regression":
            assert self.target_norm is not None
            return torch.as_tensor(
                (y.reshape(-1) - float(self.target_norm["mean"])) / float(self.target_norm["scale"]),
                dtype=torch.float32,
            )
        return torch.as_tensor(y, dtype=torch.float32)

    # -- metrics -----------------------------------------------------------
    def metrics(self, split: str, logits: np.ndarray) -> dict[str, Any]:
        y = self.y[split]
        if self.task == "regression":
            return MM.task_metrics("regression", logits.reshape(-1), y.reshape(-1), self.target_norm)
        if self.dataset == C.PCBA:
            report = ap_report(y, logits)
            report["masked_bce"] = report.pop("valid_bce")
            return report
        if self.task == "multitask":
            return MM.task_metrics("multitask", logits, y, None, self.task_names)
        raise RuntimeError(f"unsupported task {self.task!r}")

    def primary(self, metrics: Mapping[str, Any]) -> float:
        if self.dataset == C.PCBA:
            return float(metrics["ap"])
        return MM.primary_value(self.task, metrics)

    def better(self, candidate: float, incumbent: float | None) -> bool:
        if self.dataset == C.PCBA:
            return incumbent is None or float(candidate) > float(incumbent)
        return MM.better(self.task, float(candidate), None if incumbent is None else float(incumbent))

    # -- audit -------------------------------------------------------------
    def audit(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "task": self.task,
            "n_tasks": int(self.n_tasks),
            "task_names": list(self.task_names),
            "splits": {k: int(v.size) for k, v in self.rows.items()},
            "cache_fingerprints": {
                k: str(self.caches[k]["meta"].get("content_fingerprint")) for k in self.caches
            },
            "row_sha256": {k: C.sha256_bytes(np.ascontiguousarray(v, dtype=np.int64).tobytes()) for k, v in self.ids.items()},
            "split_sizes": {k: int(v.size) for k, v in self.ids.items()},
            "train_norm": {k: self.train_norm[k] for k in ("source", "fit_split", "valid_or_test_used")},
            "train_norm_fingerprint": PREP_DATA.norm_fingerprint(self.train_norm),
            "target_norm": None if self.target_norm is None else dict(self.target_norm),
            "extra_norm": dict(self.extra_norm),
            "feature_spec_hash": self.spec.hash(),
            "relation_mode": self.relation_mode,
            "max_dist": int(self.max_dist),
            "pair_counts": {
                k: {
                    "n_pairs": int(self.pairs[k]["meta"]["n_pairs"]),
                    "per_molecule": self.pairs[k]["meta"]["n_pairs_per_molecule"],
                }
                for k in self.pairs
            },
            "labels_in_structure_cache": False,
            "source": dict(self.source),
        }


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------


def _attach_pairs(
    dataset: str,
    caches: Mapping[str, Mapping[str, Any]],
    *,
    max_dist: int,
    workers: int,
    split_keys: Mapping[str, str] | None = None,
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Any]]:
    out: dict[str, Mapping[str, Any]] = {}
    audit: dict[str, Any] = {}
    split_keys = dict(split_keys or {k: k for k in caches})
    for split, cache in caches.items():
        key = split_keys[split]
        path = P.pair_cache_dir(dataset, key, max_dist)
        ident = P.cache_identity(cache)
        if (path / "meta.json").is_file():
            payload = P.load_pair_cache(dataset, key, max_dist=max_dist)
            if payload["meta"]["cache_identity"] != ident:
                raise RuntimeError(f"{dataset}/{split}: pair cache identity differs from the structure cache")
        else:
            built = P.build_pair_cache(cache, dataset, key, max_dist=max_dist, workers=workers)
            payload = {"arrays": built["arrays"], "meta": built["meta"]}
        out[split] = payload
        audit[split] = {
            "n_pairs": int(payload["meta"]["n_pairs"]),
            "per_molecule": payload["meta"]["n_pairs_per_molecule"],
            "path": C.track_rel(path),
            "built_utc": payload["meta"].get("created_utc"),
        }
    return out, audit


def _extra_norm(cache: Mapping[str, Any], pair_payload: Mapping[str, Any]) -> dict[str, Any]:
    atom_ptr = np.asarray(cache["atom_ptr"]).astype(np.int64)
    n_atoms = np.diff(atom_ptr)
    ptr = np.asarray(pair_payload["arrays"]["ptr"]).astype(np.int64)
    n_pairs = np.diff(ptr)
    return P.extra_norm_from_counts(n_atoms, n_pairs)


def _subset(view: Mapping[str, Any], splits: Sequence[str]) -> dict[str, Any]:
    return {k: view[k] for k in splits}


def load_view(
    dataset: str,
    *,
    include_test: bool = False,
    max_dist: int = 2,
    workers: int = 8,
) -> View:
    if dataset not in C.DATASETS:
        raise KeyError(dataset)
    meta = C.DATASETS[dataset]
    splits = ["train", "valid"] + (["test"] if include_test else [])
    t0 = time.perf_counter()
    if dataset == C.LIPO:
        base = LPD.build_view()
        spec = base.spec
        caches = {k: base.caches[k] for k in splits}
        ids = {k: np.asarray(base.rows[k], dtype=np.int64) for k in splits}
        rows = {k: np.arange(int(base.caches[k]["n_molecules"]), dtype=np.int64) for k in splits}
        y = {k: np.asarray(base.y[k], dtype=np.float64).reshape(-1) for k in splits}
        train_norm = base.train_norm
        target_norm = base.target_norm
        task_names = ["lipo"]
        source = {"kind": "ogb_mollipo_official_scaffold", "audit": base.audit()}
    elif dataset == C.ESOL:
        base = SED.build_esol_view()
        spec = base.spec
        caches = {k: base.caches[k] for k in splits}
        ids = {k: np.asarray(base.rows[k], dtype=np.int64) for k in splits}
        rows = {k: np.arange(int(base.caches[k]["n_molecules"]), dtype=np.int64) for k in splits}
        y = {k: np.asarray(base.y[k], dtype=np.float64).reshape(-1) for k in splits}
        train_norm = base.train_norm
        target_norm = base.target_norm
        task_names = ["esol"]
        source = {"kind": "ogb_molesol_official_scaffold", "audit": base.audit()}
    elif dataset == C.TOX21:
        base = MD.build_view("Tox21")
        spec = base.spec
        caches = {"train": base.train_cache, "valid": base.valid_cache}
        ids = {"train": np.asarray(base.rows["train"]), "valid": np.asarray(base.rows["valid"])}
        rows = {
            "train": np.arange(int(base.train_cache["n_molecules"]), dtype=np.int64),
            "valid": np.arange(int(base.valid_cache["n_molecules"]), dtype=np.int64),
        }
        y = {"train": base.targets.y_train, "valid": base.targets.y_valid}
        if include_test:
            tcache, ty, trows = MD.load_test_cache("Tox21")
            caches["test"] = tcache
            ids["test"] = np.asarray(trows, dtype=np.int64)
            rows["test"] = np.arange(int(tcache["n_molecules"]), dtype=np.int64)
            y["test"] = np.asarray(ty, dtype=np.float64)
        train_norm = base.train_norm
        target_norm = None
        task_names = list(base.task_names)
        source = {"kind": "chembench_tox21_frozen_split", "audit": base.audit()}
    elif dataset == C.PCBA:
        if include_test:
            raise RuntimeError("the PCBA frozen label artifact is NaN on every Test row; Test cannot be scored")
        task = SGD.load_task()
        spec = task.spec
        caches = {"train": task.cache, "valid": task.cache}
        ids = {
            "train": np.asarray(task.train_rows, dtype=np.int64),
            "valid": np.asarray(task.valid_rows, dtype=np.int64),
        }
        # the PCBA cache is one global array in official row order: the split rows
        # *are* the cache addresses (and the split order is preserved)
        rows = {"train": ids["train"], "valid": ids["valid"]}
        y = {
            "train": np.asarray(task.labels[np.asarray(task.train_rows)], dtype=np.float64),
            "valid": np.asarray(task.labels[np.asarray(task.valid_rows)], dtype=np.float64),
        }
        train_norm = task.norm
        target_norm = None
        task_names = [f"task{i}" for i in range(int(meta["n_tasks"]))]
        source = {"kind": "ogb_molpcba_official_scaffold", "audit": task.data_audit}
    else:  # pragma: no cover
        raise KeyError(dataset)

    pairs: dict[str, Mapping[str, Any]] = {}
    global_cache = bool(dataset == C.PCBA)
    if global_cache:
        path = P.pair_cache_dir(dataset, "all", max_dist)
        ident = P.cache_identity(caches["train"])
        if (path / "meta.json").is_file():
            payload = P.load_pair_cache(dataset, "all", max_dist=max_dist)
            if payload["meta"]["cache_identity"] != ident:
                raise RuntimeError(f"{dataset}: full pair cache identity differs from the structure cache")
        else:
            built = P.build_pair_cache(caches["train"], dataset, "all", max_dist=max_dist, workers=workers)
            payload = {"arrays": built["arrays"], "meta": built["meta"]}
        pairs = {split: payload for split in splits}
    else:
        for split in splits:
            path = P.pair_cache_dir(dataset, split, max_dist)
            ident = P.cache_identity(caches[split])
            if (path / "meta.json").is_file():
                payload = P.load_pair_cache(dataset, split, max_dist=max_dist)
                if payload["meta"]["cache_identity"] != ident:
                    raise RuntimeError(f"{dataset}/{split}: pair cache identity differs from the structure cache")
            else:
                built = P.build_pair_cache(caches[split], dataset, split, max_dist=max_dist, workers=workers)
                payload = {"arrays": built["arrays"], "meta": built["meta"]}
            pairs[split] = payload
    extra_norm = _extra_norm(caches["train"], pairs["train"])
    view = View(
        dataset=dataset,
        task=meta["task"],
        spec=spec,
        caches=caches,
        pairs=pairs,
        train_norm=train_norm,
        target_norm=target_norm,
        rows=rows,
        ids=ids,
        y=y,
        task_names=task_names,
        extra_norm=extra_norm,
        source=source,
        max_dist=max_dist,
        global_cache=global_cache,
    )
    view.source["load_seconds"] = float(time.perf_counter() - t0)
    return view


def view_with_pairs(
    base_view: View, *, relation_mode: str = "real", max_dist: int = 2, workers: int = 8
) -> View:
    """Re-use a loaded view's caches but another pair payload / relation definition."""
    if max_dist == base_view.max_dist and relation_mode == base_view.relation_mode:
        return base_view
    splits = list(base_view.caches)
    split_keys = {s: ("all" if base_view.global_cache else s) for s in splits}
    pairs, _audit = _attach_pairs(
        base_view.dataset, base_view.caches, max_dist=max_dist, workers=workers, split_keys=split_keys
    )
    extra_norm = _extra_norm(base_view.caches["train"], pairs["train"])
    return View(
        dataset=base_view.dataset,
        task=base_view.task,
        spec=base_view.spec,
        caches=base_view.caches,
        pairs=pairs,
        train_norm=base_view.train_norm,
        target_norm=base_view.target_norm,
        rows=base_view.rows,
        ids=base_view.ids,
        y=base_view.y,
        task_names=base_view.task_names,
        extra_norm=extra_norm,
        source=base_view.source,
        relation_mode=relation_mode,
        max_dist=max_dist,
        global_cache=base_view.global_cache,
    )


# ---------------------------------------------------------------------------
# prediction
# ---------------------------------------------------------------------------


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    view: View,
    split: str,
    device: torch.device,
    *,
    rows: Sequence[int] | None = None,
    batch_size: int = 256,
) -> np.ndarray:
    model.eval()
    n = int(view.rows[split].size)
    order = list(range(n)) if rows is None else [int(r) for r in rows]
    cache_rows = view.rows[split]
    out: list[np.ndarray] = []
    for start in range(0, len(order), int(batch_size)):
        chunk = [int(cache_rows[i]) for i in order[start : start + int(batch_size)]]
        batch = view.to_device(view.collate(split, chunk), device)
        pred = model(batch)
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        out.append(pred.detach().float().cpu().numpy())
    if not out:
        return np.zeros((0, view.n_tasks), dtype=np.float64)
    result = np.concatenate(out, axis=0).astype(np.float64)
    if result.shape[0] != len(order):
        raise RuntimeError(f"prediction rows {result.shape[0]} != {len(order)}")
    return result


def evaluate(
    model: torch.nn.Module,
    view: View,
    split: str,
    device: torch.device,
    *,
    batch_size: int = 256,
) -> tuple[dict[str, Any], np.ndarray]:
    logits = predict(model, view, split, device, batch_size=batch_size)
    return view.metrics(split, logits), logits


def ap_report(y: np.ndarray, logits: np.ndarray) -> dict[str, Any]:
    probs = 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64)))
    report = PCM.validation_report(np.asarray(y, dtype=np.float64), probs.astype(np.float32))
    report["valid_bce"] = float(PCM.masked_bce_numpy(np.asarray(logits, dtype=np.float64), np.asarray(y, dtype=np.float64)))
    return report


__all__ = [name for name in dir() if not name.startswith("_")]
