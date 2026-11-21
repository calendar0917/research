"""Batch packing for the frozen word-response structure.

The structure npz files are pure structure (no learned quantity, no labels for
the held-out split).  This module turns a list of molecule ids of one split into
a flat padded batch of index tensors; every downstream quantity is computed from
them so that gradients reach the per-atom / per-bond / per-pair / per-word
parameters.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

import wo_common as C
import wo_struct as S


class SplitData:
    def __init__(self, name: str, path, device=None) -> None:
        self.name = name
        payload = C.load_npz(path)
        self.arrays = {}
        self.pads = {}
        for key, value in payload.items():
            if key.startswith("pad/"):
                self.pads[key[4:]] = value
            else:
                self.arrays[key] = value
        a = self.arrays
        self.n_molecules = int(len(a["row_ids"]))
        self.n_atoms = int(len(a["atom_mol"]))
        self.graph_ptr_atom = a["graph_ptr_atom"].astype(np.int64)
        self.graph_ptr_boundary = a["graph_ptr_boundary"].astype(np.int64)
        self.graph_n_atoms = a["graph_n_atoms"].astype(np.int64)
        self.graph_n_boundary = a["graph_n_boundary"].astype(np.int64)
        # per-atom / per-word derived lookup tables (structure only)
        self.atom_row = a["atom_mol"] * 0 + 0
        self.atom_row = (np.arange(self.n_atoms, dtype=np.int64)
                         - self.graph_ptr_atom[a["atom_mol"].astype(np.int64)])
        self.word_atom_row = np.zeros((len(a["word_mol"]), C.MAX_WORD_ATOMS), dtype=np.int64)
        ptr = a["word_ptr_atom"]
        for w in range(len(a["word_mol"])):
            n = int(a["word_n_atoms"][w])
            atoms = a["word_atoms"][int(ptr[w]):int(ptr[w]) + n].astype(np.int64)
            self.word_atom_row[w, :n] = self.atom_row[atoms]
        self.word_boundary_start_local = (
            a["word_boundary_start"].astype(np.int64)
            - self.graph_ptr_boundary[a["word_mol"].astype(np.int64)])
        self.cross_slot_u_local = (
            a["cross_slot_u"].astype(np.int64)
            - self.graph_ptr_boundary[a["cross_mol"].astype(np.int64)])
        self.cross_slot_v_local = (
            a["cross_slot_v"].astype(np.int64)
            - self.graph_ptr_boundary[a["cross_mol"].astype(np.int64)])
        self.z = a["z_count"].astype(np.float32)
        self.labels = a["labels"].astype(np.float32)
        self.row_ids = a["row_ids"].astype(np.int64)
        self.z_std = None
        self.n0 = 1.0

    def set_normalizer(self, z_std: np.ndarray, n0: float) -> None:
        """Fit-only standardisation of the count vector and the frozen n0."""
        if z_std.shape[0] != self.z.shape[0]:
            raise ValueError("z standardisation row mismatch")
        self.z_std = np.asarray(z_std, dtype=np.float32)
        self.n0 = float(n0)

    # ------------------------------------------------------------------
    def pack(self, mol_ids: Sequence[int], device=None) -> dict[str, Any]:
        import torch

        a = self.arrays
        pads = self.pads
        mol_ids = np.asarray(mol_ids, dtype=np.int64)
        B = len(mol_ids)
        n_atoms = self.graph_n_atoms[mol_ids]
        n_pad = int(max(1, n_atoms.max()))
        n_bnd = self.graph_n_boundary[mol_ids]
        b_pad = int(max(1, n_bnd.max()))
        mmap = np.full(self.n_molecules, -1, dtype=np.int64)
        mmap[mol_ids] = np.arange(B)
        local_mol = lambda mol: mmap[mol]  # noqa: E731

        atom_pad = pads["atom_pad"][mol_ids][:, :n_pad]
        atom_mask = pads["atom_pad_mask"][mol_ids][:, :n_pad]
        atom_ids = atom_pad[atom_mask]
        Bn = atom_mask.shape[1]

        def gather_split(name: str):
            sel = pads[f"{name}_pad"][mol_ids]
            mask = pads[f"{name}_pad_mask"][mol_ids]
            return sel[mask]

        bond_ids = gather_split("bond")
        pair_ids = gather_split("pair")
        word_ids = gather_split("word")
        cross_ids = gather_split("cross")

        wmap = np.full(len(a["word_mol"]), -1, dtype=np.int64)
        wmap[word_ids] = np.arange(len(word_ids))
        atom_word_pad = np.full((B, Bn), -1, dtype=np.int64)
        atom_site_pad = np.zeros((B, Bn), dtype=np.int64)
        atom_orbit_pad = np.zeros((B, Bn), dtype=np.int64)
        atom_word_pad[atom_mask] = wmap[a["atom_word"][atom_ids]]
        atom_site_pad[atom_mask] = a["atom_site"][atom_ids]
        atom_orbit_pad[atom_mask] = a["word_site_orbit"][atom_ids]
        bond_mol = local_mol(a["bond_mol"][bond_ids])
        bond_u_row = self.atom_row[a["bond_u"][bond_ids]]
        bond_v_row = self.atom_row[a["bond_v"][bond_ids]]
        bond_u_atom = a["bond_u"][bond_ids]
        bond_v_atom = a["bond_v"][bond_ids]
        pair_mol = local_mol(a["pair_mol"][pair_ids])
        pair_u_row = self.atom_row[a["pair_u"][pair_ids]]
        pair_v_row = self.atom_row[a["pair_v"][pair_ids]]
        pair_u_atom = a["pair_u"][pair_ids]
        pair_v_atom = a["pair_v"][pair_ids]
        word_mol = local_mol(a["word_mol"][word_ids])
        cross_mol = local_mol(a["cross_mol"][cross_ids])

        out = {
            "mol_ids": mol_ids,
            "n_pad": n_pad,
            "b_pad": b_pad,
            "n_atoms": n_atoms,
            "n_boundary": n_bnd,
            "atom_cat": a["atom_feat"][atom_pad].astype(np.int64),
            "atom_mask": atom_mask,
            "bond_ids": bond_ids,
            "bond_mol": bond_mol,
            "bond_u_row": bond_u_row,
            "bond_v_row": bond_v_row,
            "bond_cat": a["bond_feat"][bond_ids].astype(np.int64),
            "bond_site_u": a["atom_site"][bond_u_atom].astype(np.int64),
            "bond_cross": a["bond_cross"][bond_ids].astype(bool),
            "bond_site_v": a["atom_site"][bond_v_atom].astype(np.int64),
            "bond_word_u": wmap[a["bond_word_u"][bond_ids]],
            "bond_word_v": wmap[a["bond_word_v"][bond_ids]],
            "pair_ids": pair_ids,
            "pair_mol": pair_mol,
            "pair_u_row": pair_u_row,
            "pair_v_row": pair_v_row,
            "pair_site_u": a["atom_site"][pair_u_atom].astype(np.int64),
            "pair_site_v": a["atom_site"][pair_v_atom].astype(np.int64),
            "pair_dist": a["pair_dist"][pair_ids].astype(np.int64),
            "pair_orbit": a["pair_orbit"][pair_ids].astype(np.int64),
            "pair_word": wmap[a["pair_word"][pair_ids]],
            "pair_n_atoms": a["word_n_atoms"][a["pair_word"][pair_ids]].astype(np.int64),
            "pair_m": a["word_m"][a["pair_word"][pair_ids]].astype(np.int64),
            "pair_rank": a["word_rank"][a["pair_word"][pair_ids]].astype(np.int64),
            "word_ids": word_ids,
            "word_mol": word_mol,
            "word_n_atoms": a["word_n_atoms"][word_ids].astype(np.int64),
            "word_m": a["word_m"][word_ids].astype(np.int64),
            "word_rank": a["word_rank"][word_ids].astype(np.int64),
            "word_s": a["word_s"][word_ids].astype(np.float32),
            "word_n_boundary": a["word_n_boundary"][word_ids].astype(np.int64),
            "word_atoms_row": self.word_atom_row[word_ids],
            "word_atom_mask": pads["word_atoms_mask"][word_ids],
            "word_bnd_start": self.word_boundary_start_local[word_ids],
            "word_ii_sites": a["word_ii_sites"][word_ids].astype(np.int64),
            "word_ii_mask": a["word_ii_mask"][word_ids],
            "word_bnd_sites": a["word_bnd_sites"][word_ids].astype(np.int64),
            "word_bnd_mask": a["word_bnd_mask"][word_ids],
            "word_site_bnd_local": a["word_site_bnd_local"][word_ids].astype(np.int64),
            "cross_ids": cross_ids,
            "cross_mol": cross_mol,
            "cross_u_row": self.atom_row[a["cross_u"][cross_ids]],
            "cross_v_row": self.atom_row[a["cross_v"][cross_ids]],
            "cross_local_u": a["cross_local_u"][cross_ids].astype(np.int64),
            "cross_local_v": a["cross_local_v"][cross_ids].astype(np.int64),
            "cross_word_u": wmap[a["cross_word_u"][cross_ids]],
            "cross_word_v": wmap[a["cross_word_v"][cross_ids]],
            "cross_slot_u": self.cross_slot_u_local[cross_ids],
            "cross_slot_v": self.cross_slot_v_local[cross_ids],
            "atom_word": wmap[a["atom_word"][atom_ids]],
            "atom_site": a["atom_site"][atom_ids].astype(np.int64),
            "atom_site_orbit": a["word_site_orbit"][atom_ids].astype(np.int64),
            "atom_word_pad": atom_word_pad,
            "atom_site_pad": atom_site_pad,
            "atom_orbit_pad": atom_orbit_pad,
            "z": self.z[mol_ids],
            "n0": float(self.n0),
            "z_std": self.z_std[mol_ids] if self.z_std is not None else self.z[mol_ids],
            "labels": self.labels[mol_ids],
            "rows": np.arange(B, dtype=np.int64),
        }
        if device is None:
            return out
        return {k: (torch.as_tensor(v, device=device) if isinstance(v, np.ndarray) else v)
                for k, v in out.items()}


def load_split(name: str, path) -> SplitData:
    return SplitData(name, path)
