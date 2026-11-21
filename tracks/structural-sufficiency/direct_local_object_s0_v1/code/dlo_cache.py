"""Batch views over the prepared object/frame cache.

The cache stores objects in molecule order and frames in object order.  A batch
may be any shuffled subset of molecules, so every mapping is built explicitly
(``obj_to_mol`` from batch positions, ``frame_to_obj`` from object positions).
No offset subtraction, no assumption that the batch is sorted.

The frame table is deduped by the complete raw input.  Within a forward pass we
dedup again over the active frame ids (``np.unique(..., return_inverse=True)``)
and share the encoder output; gradients accumulate through ``index_add``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import dlo_common as C


class ArmCache:
    def __init__(self, payload: dict[str, np.ndarray], stats: dict[str, Any], arm: str):
        if arm not in C.ARMS:
            raise ValueError(arm)
        self.arm = arm
        prefix = C.ARM_PREFIX[arm] + "_"
        self.mol_obj_ptr = np.ascontiguousarray(payload[prefix + "mol_obj_ptr"], np.int64)
        self.obj_root = np.ascontiguousarray(payload[prefix + "obj_root"], np.uint8)
        self.obj_atoms = np.ascontiguousarray(payload[prefix + "obj_atoms"], np.uint8)
        self.obj_size = np.ascontiguousarray(payload[prefix + "obj_size"], np.uint8)
        self.obj_bucket = np.ascontiguousarray(payload[prefix + "obj_bucket"], np.uint8)
        self.obj_core = np.ascontiguousarray(payload[prefix + "obj_core"], bool)
        self.obj_mol = np.ascontiguousarray(payload[prefix + "obj_mol"], np.int32)
        self.mol_n_bucket = np.ascontiguousarray(payload[prefix + "mol_n_bucket"], np.int32)
        self.obj_frame_ptr = np.ascontiguousarray(payload[prefix + "obj_frame_ptr"], np.int64)
        self.frame_input_id = np.ascontiguousarray(payload[prefix + "frame_input_id"], np.uint32)
        self.frame_weight = np.ascontiguousarray(payload[prefix + "frame_weight"], np.float32)
        self.frame_k = np.ascontiguousarray(payload[prefix + "frame_k"], np.uint8)
        self.frame_atom_attr = np.ascontiguousarray(payload[prefix + "frame_atom_attr"], np.uint16)
        self.frame_bond_attr = np.ascontiguousarray(payload[prefix + "frame_bond_attr"], np.uint16)
        self.frame_node_mask = np.ascontiguousarray(payload[prefix + "frame_node_mask"], np.uint8)
        self.frame_edge_mask = np.ascontiguousarray(payload[prefix + "frame_edge_mask"], np.uint8)
        self.frame_adj = np.ascontiguousarray(payload[prefix + "frame_adj"], np.uint8)
        self.atom_code = np.ascontiguousarray(payload["atom_code"], np.uint8)
        self.bond_code = np.ascontiguousarray(payload["bond_code"], np.uint8)
        self.row_ids = np.ascontiguousarray(payload["row_ids"], np.int64)
        self.split_id = np.ascontiguousarray(payload["split_id"], np.int8)
        self.labels = np.ascontiguousarray(payload["labels"], np.float32)
        self.n_atoms = np.ascontiguousarray(payload["n_atoms"], np.int64)
        self.n_bonds = np.ascontiguousarray(payload["n_bonds"], np.int64)
        self.scaler = {k: float(v) for k, v in stats["target_scaler"].items()}
        self.cache_sha256 = stats.get("cache_sha256")
        arm_stats = stats["arms"][arm]
        self.mu_bucket = np.asarray(arm_stats["mu_bucket"], np.float64)
        self.bucket_normalizer = np.asarray(arm_stats.get("bucket_normalizer", arm_stats["mu_bucket"]), np.float64)
        self.split_rows = {name: np.nonzero(self.split_id == s)[0] for s, name in C.SPLIT_NAME.items()}
        self.n_molecules = int(self.row_ids.size)
        self.stats = arm_stats

    # ------------------------------------------------------------------ batch
    @staticmethod
    def _expand(ptr: np.ndarray, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        starts = ptr[ids]
        counts = ptr[ids + 1] - starts
        total = int(counts.sum())
        if total == 0:
            return np.zeros(0, np.int64), counts
        offsets = np.repeat(np.cumsum(counts, dtype=np.int64) - counts, counts)
        out = np.repeat(starts, counts) + (np.arange(total, dtype=np.int64) - offsets)
        return out, counts

    def pack(self, mol_ids: np.ndarray) -> dict[str, np.ndarray]:
        mol_ids = np.asarray(mol_ids, np.int64)
        n_mol = int(mol_ids.size)
        obj_ids, counts_obj = self._expand(self.mol_obj_ptr, mol_ids)
        n_obj = int(obj_ids.size)
        frame_ids, counts_f = self._expand(self.obj_frame_ptr, obj_ids)
        n_frames = int(frame_ids.size)
        obj_to_mol = np.repeat(np.arange(n_mol, dtype=np.int64), counts_obj)
        frame_to_obj = np.repeat(np.arange(n_obj, dtype=np.int64), counts_f)
        ref_to_input = self.frame_input_id[frame_ids]
        active, inverse = np.unique(ref_to_input, return_inverse=True)
        active = np.ascontiguousarray(active, np.int64)
        inverse = np.ascontiguousarray(inverse, np.int64)
        return {
            "mol_ids": mol_ids,
            "n_molecules": np.int64(n_mol),
            "n_objects": np.int64(n_obj),
            "n_frames": np.int64(n_frames),
            "obj_to_mol": obj_to_mol,
            "obj_bucket": self.obj_bucket[obj_ids],
            "frame_to_obj": frame_to_obj,
            "frame_inverse": inverse,
            "frame_weight": self.frame_weight[frame_ids],
            "frame_input_id": active,
            "frame_k": self.frame_k[active],
            "frame_atom_attr": self.frame_atom_attr[active],
            "frame_bond_attr": self.frame_bond_attr[active],
            "frame_node_mask": self.frame_node_mask[active],
            "frame_edge_mask": self.frame_edge_mask[active],
            "frame_adj": self.frame_adj[active],
            "n_bucket": self.mol_n_bucket[mol_ids],
            "n_atoms": self.n_atoms[mol_ids],
            "n_bonds": self.n_bonds[mol_ids],
        }

    def target_std(self, mol_ids: np.ndarray) -> np.ndarray:
        y = self.labels[np.asarray(mol_ids, np.int64)].astype(np.float64)
        return ((y - self.scaler["mean"]) / self.scaler["std"]).astype(np.float32)


def load_payload(path: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as handle:
        return {k: np.ascontiguousarray(handle[k]) for k in handle.files}
