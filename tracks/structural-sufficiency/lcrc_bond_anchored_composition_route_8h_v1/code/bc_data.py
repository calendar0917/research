"""Data layer: frozen split caches, the two tasks, the path3 cache and batching.

Everything is read-only reuse of the frozen data rounds:

* ``gr_data.bundle`` -> the verified official ``ogbg-mollipo`` graph cache, the
  Train-only feature normalisation, the split-local -> official ``row_id`` map and
  the Train-only target scaler of the experimental logD label;
* ``gr_data.oracle_bundle`` -> the same graphs / splits with the RDKit WC_LOGP
  target and *its own* Train-only scaler (never mixed with logD);
* ``as_data.collate_gated`` -> the frozen ``lcrc_data.collate`` batch plus the
  verified S1/S2 occurrence -> physical-bond bookkeeping, which is what makes the
  physical-bond evidence gate of section C4 possible at all.

This round adds one object: the per-split three-bond-path cache of
:mod:`bc_path3`, built deterministically from the frozen cache and stored with a
SHA-256 digest so every training unit can prove it consumed the same events.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import bc_common as C
import bc_path3 as P3

PATHS3_CACHE_VERSION = "paths3-v2"
PATHS3_MANIFEST = "paths3_manifest.json"


# ---------------------------------------------------------------------------
# bundles
# ---------------------------------------------------------------------------


def bundle(task: str, *, verify: bool = True):
    """The task bundle: official graphs + that task's Train-only target scaler."""
    G = C.modules()["gr_data"]
    if task == "lipo":
        return G.bundle(C.DATASET, verify=verify)
    if task == "wc_logp":
        return G.oracle_bundle(C.DATASET, C.TASK_ORACLE_PROPERTY[task])
    raise KeyError(task)


def split_sizes(b) -> dict[str, int]:
    return {split: int(np.asarray(b.caches[split]["n_molecules"])) for split in C.SPLITS}


def eval_order(b, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order, row_ids = b.row_ids_sorted(split)
    return order, row_ids, np.asarray(b.targets_raw[split], dtype=np.float64)[order]


def target_context(task: str, b) -> dict[str, Any]:
    scaler = dict(b.scaler)
    out = {
        "task": task,
        "target_name": C.TASK_TARGET_NAME[task],
        "scaler": scaler,
        "scaler_mean": float(scaler["mean"]),
        "scaler_std": float(scaler["std"]),
        "source": str(scaler.get("source", "")),
        "fit_split": str(scaler.get("fit_split", "train")),
        "valid_or_test_used": bool(scaler.get("valid_or_test_used", False)),
        "n_train": int(np.asarray(b.targets_raw["train"]).size),
        "per_split": {},
    }
    for split in C.SPLITS:
        raw = np.asarray(b.targets_raw[split], dtype=np.float64)
        out["per_split"][split] = {
            "n": int(raw.size),
            "mean": float(raw.mean()),
            "std": float(raw.std(ddof=0)),
            "min": float(raw.min()),
            "max": float(raw.max()),
        }
    return out


# ---------------------------------------------------------------------------
# path3 cache
# ---------------------------------------------------------------------------


def cache_paths(split: str) -> dict[str, Path]:
    base = C.CACHE_DIR / f"paths3_{split}_{PATHS3_CACHE_VERSION}"
    return {"npz": base.with_suffix(".npz"), "json": base.with_suffix(".json")}


def manifest_path() -> Path:
    return C.CACHE_DIR / PATHS3_MANIFEST


def _to_npz(p3: P3.Path3Split) -> dict[str, np.ndarray]:
    om = p3.occ_map
    return {
        **p3.arrays(),
        "nbr_bond": om.nbr_bond,
        "path_bond1": om.path_bond1,
        "path_bond2": om.path_bond2,
        "path_middle": om.path_middle,
        "atom_counts": om.atom_counts,
        "bond_counts": om.bond_counts,
        "n_s2_total": om.n_s2_total,
        "n_bond_types": np.asarray([om.n_bond_types], dtype=np.int64),
        "n_molecules": np.asarray([p3.n_molecules], dtype=np.int64),
    }


def _from_npz(path: Path, dataset: str, split: str, spec) -> P3.Path3Split:
    AD = C.modules()["as_data"]
    with np.load(path, allow_pickle=False) as mesh:
        arr = {key: np.asarray(mesh[key]) for key in mesh.files}
    occ_map = AD.SplitOcc(
        dataset=dataset,
        split=split,
        n_molecules=int(arr["n_molecules"][0]),
        nbr_bond=arr["nbr_bond"].astype(np.int64),
        path_bond1=arr["path_bond1"].astype(np.int64),
        path_bond2=arr["path_bond2"].astype(np.int64),
        path_middle=arr["path_middle"].astype(np.int64),
        bond_counts=arr["bond_counts"].astype(np.int64),
        atom_counts=arr["atom_counts"].astype(np.int64),
        n_s2_total=arr["n_s2_total"].astype(np.int64),
        n_bond_types=int(arr["n_bond_types"][0]),
        sha256="",
    )
    return P3.Path3Split(
        dataset=dataset,
        split=split,
        n_molecules=int(arr["n_molecules"][0]),
        ptr=arr["p3_ptr"].astype(np.int64),
        atoms=arr["p3_atoms"].astype(np.int64),
        occ=arr["p3_occ"].astype(np.int64),
        anchor=arr["p3_anchor"].astype(np.int64),
        closure4=arr["p3_closure4"].astype(np.int64),
        occ_root=arr["occ_root"].astype(np.int64),
        occ_end=arr["occ_end"].astype(np.int64),
        occ_map=occ_map,
        sha256="",
    )


def _digest(p3: P3.Path3Split) -> str:
    om = p3.occ_map
    return C.sha256_json(
        {
            "dataset": p3.dataset,
            "split": p3.split,
            "n_molecules": int(p3.n_molecules),
            "n_s2": C.sha256_array(om.n_s2_total),
            "nbr_bond": C.sha256_array(om.nbr_bond),
            "path_bond1": C.sha256_array(om.path_bond1),
            "path_bond2": C.sha256_array(om.path_bond2),
            "path_middle": C.sha256_array(om.path_middle),
            "p3_ptr": C.sha256_array(p3.ptr),
            "p3_atoms": C.sha256_array(p3.atoms),
            "p3_occ": C.sha256_array(p3.occ),
            "p3_anchor": C.sha256_array(p3.anchor),
            "p3_closure4": C.sha256_array(p3.closure4),
            "occ_root": C.sha256_array(p3.occ_root),
            "occ_end": C.sha256_array(p3.occ_end),
        }
    )


def build_paths3(
    b,
    *,
    rebuild: bool = False,
    write: bool = True,
    splits: Sequence[str] = C.SPLITS,
) -> dict[str, P3.Path3Split]:
    """Build (or verify-load) the three-bond-path cache for every split."""
    manifest = C.read_json(manifest_path()) if manifest_path().is_file() else {}
    out: dict[str, P3.Path3Split] = {}
    updated = dict(manifest)
    for split in splits:
        paths = cache_paths(split)
        cached_ok = False
        if not rebuild and paths["npz"].is_file():
            p3 = _from_npz(paths["npz"], C.DATASET, split, b.spec)
            digest = _digest(p3)
            expected = (manifest.get("splits", {}).get(split) or {}).get("sha256")
            if expected is None or digest == expected:
                p3.sha256 = digest
                cached_ok = True
        if not cached_ok:
            p3 = P3.build_split(b.caches[split], b.spec, C.DATASET, split)
            p3.sha256 = _digest(p3)
            if write:
                C.atomic_write_npz(paths["npz"], _to_npz(p3))
        out[split] = p3
        updated.setdefault("splits", {})[split] = {
            "sha256": p3.sha256,
            "n_molecules": int(p3.n_molecules),
            "n_path3_events": int(p3.n_events),
            "n_s2_occurrences": int(p3.occ_map.n_s2_total.sum()),
            "path": C.relpath(cache_paths(split)["npz"]),
            "verified_online": bool(p3.verification and p3.verification.get("all_paths_verified")),
        }
    updated["version"] = PATHS3_CACHE_VERSION
    updated["dataset"] = C.DATASET
    updated["n_bond_types"] = int(b.spec.n_bond_types)
    if write:
        C.atomic_write_json(manifest_path(), updated)
    return out


# ---------------------------------------------------------------------------
# batch
# ---------------------------------------------------------------------------


def _range_index(ptr: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Identical index algebra to ``lcrc_data.collate`` (verified in the gates)."""
    rows = np.asarray(rows, dtype=np.int64).reshape(-1)
    counts = ptr[rows + 1] - ptr[rows]
    if counts.size == 0 or int(counts.sum()) == 0:
        return np.zeros(0, dtype=np.int64), counts
    idx = np.concatenate(
        [np.arange(int(s), int(s) + int(c), dtype=np.int64) for s, c in zip(ptr[rows], counts)]
    )
    return idx, counts


@dataclass
class Batch:
    """A frozen gated batch plus the three-bond-path accounting of the same rows."""

    gb: Any                       # as_data.GatedBatch
    batch: dict[str, Any]         # gb.batch + path3 tensors (model input)
    p3: dict[str, Any]            # path3 accounting in batch coordinates
    rows: np.ndarray
    split: str
    n_graphs: int
    checks: dict[str, Any] = field(default_factory=dict)

    def to(self, device) -> "Batch":
        gb = self.gb.to(device)
        batch = {
            key: (value.to(device) if hasattr(value, "to") and hasattr(value, "dtype") else value)
            for key, value in gb.batch.items()
        }
        batch["n_graphs"] = self.n_graphs
        p3 = {key: value.to(device) for key, value in self.p3.items()}
        return Batch(gb=gb, batch=batch, p3=p3, rows=self.rows, split=self.split,
                     n_graphs=self.n_graphs, checks=self.checks)


def make_batch(b: Any, split: str, rows: Sequence[int], p3: P3.Path3Split) -> Batch:
    """Frozen collate + path3 bookkeeping in one batch."""
    AD = C.modules()["as_data"]
    torch = __import__("torch")
    rows = np.asarray(rows, dtype=np.int64).reshape(-1)
    cache = b.caches[split]
    gb = AD.collate_gated(cache, rows, p3.occ_map)
    batch = dict(gb.batch)

    atom_ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)
    path_ptr = np.asarray(cache["path_ptr"], dtype=np.int64)
    atom_idx, atom_counts = _range_index(atom_ptr, rows)
    path_idx, path_counts = _range_index(path_ptr, atom_idx)
    n_occ = int(path_idx.size)
    if n_occ != int(batch["path_cat"].numel()):
        raise RuntimeError(
            f"{split}: rebuilt occurrence index has {n_occ} rows but the frozen collate has "
            f"{int(batch['path_cat'].numel())}"
        )
    inverse = np.full(int(p3.occ_map.n_s2_total.sum()), -1, dtype=np.int64)
    inverse[path_idx] = np.arange(n_occ, dtype=np.int64)

    events = P3.collate_events(
        p3, rows, bond_jump=np.asarray(gb.bond_jump.cpu(), dtype=np.int64), gb_occ=gb
    )
    occ_split = events["p3_occ_split"]
    occ_batch = inverse[occ_split] if occ_split.size else np.zeros((0, 4), dtype=np.int64)
    if occ_batch.size and int(occ_batch.min()) < 0:
        raise RuntimeError(f"{split}: a composition event references an occurrence outside the batch")
    # ``gb.bond_jump`` is the *graph row index* of every collated bond row; the
    # per-graph bond offset is the cumulative bond count.
    bond_counts_np = np.asarray(gb.bond_counts, dtype=np.int64)
    bond_offsets = (
        np.concatenate([[0], np.cumsum(bond_counts_np)[:-1]]).astype(np.int64)
        if bond_counts_np.size
        else np.zeros(0, dtype=np.int64)
    )
    anchor_bond_batch = events["p3_anchor_local"] + bond_offsets[events["p3_graph"]]

    torch_p3 = {
        "p3_occ": torch.from_numpy(occ_batch.astype(np.int64)),
        "p3_graph": torch.from_numpy(events["p3_graph"].astype(np.int64)),
        "p3_counts": torch.from_numpy(events["p3_counts"].astype(np.int64)),
        "p3_anchor_bond": torch.from_numpy(anchor_bond_batch.astype(np.int64)),
        "p3_closure4": torch.from_numpy(events["p3_closure4"].astype(np.float32)),
        "p3_atoms": torch.from_numpy(events["p3_atoms"].astype(np.int64)),
        "p3_index": torch.from_numpy(events["p3_index"].astype(np.int64)),
    }
    batch["n_graphs"] = int(gb.n_graphs)
    batch["p3_occ"] = torch_p3["p3_occ"]
    batch["p3_graph"] = torch_p3["p3_graph"]
    batch["p3_counts"] = torch_p3["p3_counts"]
    batch["p3_anchor_bond"] = torch_p3["p3_anchor_bond"]
    batch["p3_closure4"] = torch_p3["p3_closure4"]

    # independent identity: the frozen S2 code of every occurrence must be
    # reproducible from the reconstructed (middle atom type, bond1 type, bond2 type)
    # in *batch* coordinates (molecule-local ids carry the graph offset).
    occ_map = p3.occ_map
    nb = int(occ_map.n_bond_types)
    if n_occ:
        graph = batch["path_graph"].numpy().astype(np.int64)
        atom_off = batch["atom_off"].numpy().astype(np.int64)
        batch_atom_type = np.asarray(batch["atom_type"], dtype=np.int64)
        batch_bond_type = np.asarray(batch["bond_type"], dtype=np.int64)
        mid_local = occ_map.path_middle[path_idx]
        p1_local = occ_map.path_bond1[path_idx]
        p2_local = occ_map.path_bond2[path_idx]
        mid_type = batch_atom_type[atom_off[graph] + mid_local]
        bt1 = batch_bond_type[bond_offsets[graph] + p1_local]
        bt2 = batch_bond_type[bond_offsets[graph] + p2_local]
        rebuilt_cat = mid_type * nb * nb + (bt1 - 1) * nb + (bt2 - 1)
        cat_ok = bool(np.array_equal(rebuilt_cat.astype(np.int64), np.asarray(batch["path_cat"], dtype=np.int64)))
        p1_ok = bool(np.array_equal(gb.path_bond1.numpy().astype(np.int64), p1_local))
        p2_ok = bool(np.array_equal(gb.path_bond2.numpy().astype(np.int64), p2_local))
    else:
        cat_ok, p1_ok, p2_ok = True, True, True
    checks = {
        "n_occurrences": n_occ,
        "path_code_reproduced_from_bonds": cat_ok,
        "path_bond1_alignment": p1_ok,
        "path_bond2_alignment": p2_ok,
        "occurrence_bookkeeping": gb.occurrence_check.get("n_failed", 0) == 0
        if isinstance(gb.occurrence_check, dict)
        else True,
    }
    if not (cat_ok and p1_ok and p2_ok):
        raise RuntimeError(f"{split}: S2 code is not reproducible from the reconstructed physical bonds")
    return Batch(gb=gb, batch=batch, p3=torch_p3, rows=rows, split=split, n_graphs=int(gb.n_graphs), checks=checks)


# ---------------------------------------------------------------------------
# prediction helpers
# ---------------------------------------------------------------------------


def predict_std(
    model,
    b: Any,
    p3: P3.Path3Split,
    split: str,
    *,
    device,
    batch_size: int = 256,
    branch: bool = True,
) -> dict[str, np.ndarray]:
    """Standardised predictions in ascending official ``row_id`` order."""
    import torch

    order, row_ids, y_raw = eval_order(b, split)
    preds: list[np.ndarray] = []
    bases: list[np.ndarray] = []
    deltas: list[np.ndarray] = []
    counts: list[np.ndarray] = []
    model.eval()
    composed = hasattr(model, "compose")
    with torch.no_grad():
        for start in range(0, order.size, int(batch_size)):
            chunk = order[start : start + int(batch_size)]
            bt = make_batch(b, split, chunk, p3).to(device)
            if composed:
                parts = model(bt.batch, bt.p3, branch=branch, return_parts=True)
                pred_std = parts["pred_std"]
                base_std = parts["pred_base_std"]
                delta_std = parts["delta_comp"]
            else:
                parts = model.parts(dict(bt.batch))
                pred_std = parts["pred_std"]
                base_std = parts["pred_std"]
                delta_std = parts["pred_std"] * 0.0
            preds.append(pred_std.detach().float().cpu().numpy().astype(np.float64))
            bases.append(base_std.detach().float().cpu().numpy().astype(np.float64))
            deltas.append(delta_std.detach().float().cpu().numpy().astype(np.float64))
            counts.append(bt.p3["p3_counts"].detach().cpu().numpy().astype(np.int64))
    return {
        "row_id": np.asarray(row_ids, dtype=np.int64),
        "order": np.asarray(order, dtype=np.int64),
        "y_raw": np.asarray(y_raw, dtype=np.float64),
        "pred_std": np.concatenate(preds) if preds else np.zeros(0),
        "base_std": np.concatenate(bases) if bases else np.zeros(0),
        "delta_std": np.concatenate(deltas) if deltas else np.zeros(0),
        "n_paths3": np.concatenate(counts) if counts else np.zeros(0, dtype=np.int64),
    }


def to_raw(values, scaler: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(values, dtype=np.float64) * float(scaler["std"]) + float(scaler["mean"])


__all__ = [
    "bundle",
    "build_paths3",
    "cache_paths",
    "eval_order",
    "make_batch",
    "manifest_path",
    "predict_std",
    "split_sizes",
    "target_context",
    "to_raw",
    "Batch",
]
