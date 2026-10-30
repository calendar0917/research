"""Dataset wrapper: frozen LCRC base batch + HC3 structure tensors.

The base batch is produced by the frozen ``lc_data.collate`` unchanged, so B and
H/F consume bit-identical atom/bond/chord inputs.  The HC3 tensors are appended
by :func:`hc_structure.collate` and are label-blind.

The wrapper asserts that no label-derived key is present in a forward batch.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import torch

import hc_common as C
import hc_structure as HS

BASE_KEYS = frozenset({
    "n_graphs", "atom_feat", "atom_type", "atom_scalar", "atom_batch", "atom_off",
    "bond_feat", "bond_type", "bond_index", "bond_batch", "nbr_atom", "nbr_code",
    "path_atom", "path_cat", "path_end63", "path_end", "path_closure",
    "pair_i", "pair_j", "pair_cat", "pair_graph", "global",
    "atom_counts", "bond_counts", "pair_counts",
})
HC_KEYS = frozenset({
    "hc_z_atom", "hc_z_bond", "hc_z_chord", "hc_path_atom", "hc_path_bonds",
    "hc_path_bond", "hc_path_chord", "hc_path_z", "hc_event_n", "hc_path_batch",
    "hc_z_batch",
})
FORWARD_KEYS = BASE_KEYS | HC_KEYS
FORBIDDEN = frozenset({"labels", "y", "logp", "sa", "penalty", "label", "target"})


def assert_forward_batch(batch: Mapping[str, object]) -> None:
    bad = sorted(k for k in batch if k not in FORWARD_KEYS)
    if bad:
        raise RuntimeError(f"unexpected forward keys: {bad}")
    leaked = sorted(k for k in batch if k.lower() in FORBIDDEN or "label" in k.lower())
    if leaked:
        raise RuntimeError(f"label-derived keys reached the forward batch: {leaked}")


class SplitData:
    """One split (train/valid) with base tensors, HC3 structure and labels."""

    def __init__(self, split: str, device="cpu", *, cache=None, struct=None):
        if split not in ("train", "valid"):
            raise PermissionError(f"split {split!r} is outside the LCRC-HC3 data boundary")
        self.split = split
        self.cache = C.load_cache(split) if cache is None else cache
        self.struct = HS.load_split(split, cache=self.cache) if struct is None else struct
        self.labels = np.asarray(self.cache["labels"], dtype=np.float64)
        self.n = int(self.cache["n_molecules"])
        self.rows = np.arange(self.n, dtype=np.int64)
        self.device = torch.device(device)

    def batch(self, rows: Sequence[int]) -> dict:
        rows = np.asarray(rows, dtype=np.int64)
        batch = C.LD.collate(self.cache, rows)
        batch.update(HS.collate(self.struct, batch, rows))
        assert_forward_batch(batch)
        return C.to_device(batch, self.device)

    def targets(self, rows: Sequence[int]) -> np.ndarray:
        return self.labels[np.asarray(rows, dtype=np.int64)]

    def target_tensor(self, rows: Sequence[int], device) -> torch.Tensor:
        return torch.as_tensor(self.targets(rows), dtype=torch.float32, device=device)


@torch.no_grad()
def predict(model, data: SplitData, rows: Sequence[int], *, batch_size: int = 256,
            device=None, return_aux: bool = False):
    model.eval()
    rows = np.asarray(rows, dtype=np.int64)
    dev = data.device if device is None else device
    preds, aux_parts = [], {}
    for start in range(0, len(rows), int(batch_size)):
        chunk = rows[start:start + int(batch_size)]
        batch = data.batch(chunk)
        if return_aux:
            pred, aux = model(batch, return_aux=True)
            for k, v in aux.items():
                if torch.is_tensor(v):
                    aux_parts.setdefault(k, []).append(v.detach().float().cpu())
        else:
            pred = model(batch)
        preds.append(pred.detach().float().cpu().numpy().reshape(-1))
    pred = np.concatenate(preds, 0) if preds else np.zeros(0, dtype=np.float32)
    if not return_aux:
        return pred
    aux = {k: (torch.cat(v, 0).numpy() if v else np.zeros(0)) for k, v in aux_parts.items()}
    return pred, aux


def mae(pred, y) -> float:
    return float(np.mean(np.abs(np.asarray(pred, dtype=np.float64)
                                - np.asarray(y, dtype=np.float64))))


__all__ = ["BASE_KEYS", "HC_KEYS", "FORWARD_KEYS", "FORBIDDEN",
           "assert_forward_batch", "SplitData", "predict", "mae"]
