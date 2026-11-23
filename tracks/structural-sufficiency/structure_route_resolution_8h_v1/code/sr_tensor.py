"""Batch assembly for the E/F arms: RICE-v0 collation + global format ids."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch

import rice_tensor as RT

SIZES = (2, 3, 4)


def _move(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    if device == "cpu":
        return batch
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def collate_ef(rows: Sequence[int], structs: Mapping[int, Any],
               raw: Mapping[str, Any] | None, device: Any = "cpu") -> dict[str, Any]:
    rows = [int(r) for r in rows]
    batch = RT.collate(rows, structs, raw, "cpu")
    if all(hasattr(structs[row], "f_atom_type") for row in rows):
        type_arrays: dict[int, list[np.ndarray]] = {1: [], 2: [], 3: [], 4: []}
        for row in rows:
            st = structs[row]
            type_arrays[1].append(np.asarray(st.f_atom_type, np.int64))
            for size in SIZES:
                type_arrays[size].append(np.asarray(st.f_type[size], np.int64))
        for size in (1, 2, 3, 4):
            cat = np.concatenate(type_arrays[size]) if type_arrays[size] \
                else np.zeros(0, np.int64)
            batch[f"type_{size}"] = torch.as_tensor(cat, dtype=torch.long)
    return _move(batch, device)