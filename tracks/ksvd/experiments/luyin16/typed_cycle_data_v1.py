"""Static typed-cycle ring cache and data attachment (ZINC, official train/valid).

Round ``zinc_e2e_dictenv_typed_cycle_v1`` (Workstream Z, ZINC).

The ring object is enumerated **once** per molecule (all chordless cycles of
length 3..10), expanded into all ``2L`` dihedral integer views and cached; no
epoch re-enumerates or samples rings.  The fields follow the pre-registration:

* ``ring_view_atom[V,10]``, ``ring_view_bond[V,10]``, ``ring_view_mask[V,10]``,
  ``ring_view_length[V]`` — no offset;
* ``ring_view_cycle[V]`` — local cycle id (cumulative cycle offset on batch);
* ``ring_cycle_anchor[C]`` — one ring node, routing only (node ptr on batch);
* ``ring_cycle_length[C]`` — no offset.

Official **test** is never loaded here.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_typed_cycle_v1 as tc
from tracks.ksvd.experiments.luyin16.zinc_long_range_proxy import _data_to_graph, _load_zinc

REFERENCE_DIR = (
    Path(__file__).resolve().parent / "typed_cycle_reference" / "v1_20261002"
)
if str(REFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(REFERENCE_DIR))

import typed_cycle_reference as _tcr  # noqa: E402

CACHE_VERSION = "typed_cycle_ring_cache_v1"
SPLIT_TO_RAW = {"train": "train", "valid": "val"}


def ring_fields(ref_graph: Any) -> dict[str, np.ndarray]:
    """Build the per-molecule integer ring fields (see module docstring)."""
    cycles = _tcr.chordless_cycles(ref_graph, max_length=tc.MAX_RING_LENGTH)
    atom_parts: list[np.ndarray] = []
    bond_parts: list[np.ndarray] = []
    mask_parts: list[np.ndarray] = []
    length_parts: list[np.ndarray] = []
    cycle_parts: list[np.ndarray] = []
    anchors: list[int] = []
    cycle_lengths: list[int] = []
    for cycle_id, cycle in enumerate(cycles):
        atom, bond, mask, length = tc.dihedral_view_components(ref_graph, cycle)
        atom_parts.append(atom)
        bond_parts.append(bond)
        mask_parts.append(mask.astype(np.float32))
        length_parts.append(length)
        cycle_parts.append(np.full((atom.shape[0],), cycle_id, dtype=np.int64))
        anchors.append(int(cycle[0]))
        cycle_lengths.append(int(len(cycle)))
    if atom_parts:
        atom = np.concatenate(atom_parts, axis=0)
        bond = np.concatenate(bond_parts, axis=0)
        mask = np.concatenate(mask_parts, axis=0)
        length = np.concatenate(length_parts, axis=0)
        cycle_ids = np.concatenate(cycle_parts, axis=0)
    else:
        atom = np.zeros((0, tc.MAX_RING_LENGTH), dtype=np.int64)
        bond = np.zeros((0, tc.MAX_RING_LENGTH), dtype=np.int64)
        mask = np.zeros((0, tc.MAX_RING_LENGTH), dtype=np.float32)
        length = np.zeros((0,), dtype=np.int64)
        cycle_ids = np.zeros((0,), dtype=np.int64)
    return {
        "ring_view_atom": atom,
        "ring_view_bond": bond,
        "ring_view_mask": mask,
        "ring_view_length": length,
        "ring_view_cycle": cycle_ids,
        "ring_cycle_anchor": np.asarray(anchors, dtype=np.int64),
        "ring_cycle_length": np.asarray(cycle_lengths, dtype=np.int64),
    }


def build_ring_payload(root: Path, split: str, subset: int | None = None) -> list[dict[str, np.ndarray]]:
    raw = list(_load_zinc(Path(root), SPLIT_TO_RAW[split]))
    if subset is not None:
        raw = raw[: int(subset)]
    payload: list[dict[str, np.ndarray]] = []
    for data in raw:
        graph, node_types, edge_types = _data_to_graph(data)
        ref_graph = _to_reference(graph, node_types, edge_types)
        payload.append(ring_fields(ref_graph))
    return payload


def _to_reference(graph: Any, node_types: np.ndarray, edge_types: dict) -> Any:
    from tracks.ksvd.experiments.luyin16.typed_cycle_probe_v1 import to_reference_graph

    return to_reference_graph(graph, node_types, edge_types)


def build_cache(root: Path, split: str, cache_dir: Path, *, force: bool = False) -> dict[str, Any]:
    from tracks.ksvd.experiments.luyin16.typed_cycle_probe_v1 import split_fingerprint

    cache_dir.mkdir(parents=True, exist_ok=True)
    split_fp = split_fingerprint(Path(root))["split_fingerprint"]
    key = hashlib.sha256(
        f"{CACHE_VERSION}|{split}|{split_fp}|{tc.RING_INPUT_DIM}".encode("ascii")
    ).hexdigest()[:16]
    path = cache_dir / f"ring_{split}_{key}.pt"
    if path.exists() and not force:
        return {"path": str(path), "reused": True, "key": key, "n_molecules": None}
    payload = build_ring_payload(root, split)
    torch.save(payload, path)
    meta = {
        "path": str(path),
        "reused": False,
        "key": key,
        "n_molecules": int(len(payload)),
        "total_cycles": int(sum(row["ring_cycle_anchor"].shape[0] for row in payload)),
        "total_views": int(sum(row["ring_view_atom"].shape[0] for row in payload)),
        "cache_version": CACHE_VERSION,
        "sha256": _sha256(path),
        "bytes": int(path.stat().st_size),
    }
    return meta


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def attach_rings(data_list: Sequence[Any], payload: Sequence[dict[str, np.ndarray]], subset: int | None = None) -> None:
    count = len(payload) if subset is None else min(int(subset), len(payload))
    if count > len(data_list):
        raise RuntimeError(f"ring payload {count} exceeds data list {len(data_list)}")
    for index in range(count):
        data = data_list[index]
        row = payload[index]
        data.ring_view_atom = torch.as_tensor(row["ring_view_atom"], dtype=torch.long)
        data.ring_view_bond = torch.as_tensor(row["ring_view_bond"], dtype=torch.long)
        data.ring_view_mask = torch.as_tensor(row["ring_view_mask"], dtype=torch.float32)
        data.ring_view_length = torch.as_tensor(row["ring_view_length"], dtype=torch.long)
        data.ring_view_cycle = torch.as_tensor(row["ring_view_cycle"], dtype=torch.long)
        data.ring_cycle_anchor = torch.as_tensor(row["ring_cycle_anchor"], dtype=torch.long)
        data.ring_cycle_length = torch.as_tensor(row["ring_cycle_length"], dtype=torch.long)


__all__ = [
    "CACHE_VERSION",
    "ring_fields",
    "build_ring_payload",
    "build_cache",
    "attach_rings",
]