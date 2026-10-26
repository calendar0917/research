"""Leak-free masked structural rebuild for BOSI (this round's only data path).

Frozen design (PROTOCOL.md §2.1):

* mask **raw attribute codes only** — atom type and physical-bond type — with
  dedicated tokens (atom ``27``, bond ``0``); the *topology* is never touched,
  so ``bond_index`` and therefore the whole T/Q/R/N relation cache, the
  biconnected blocks and the closure counts are exactly the frozen objects of
  BOSI v0;
* rebuild **every feature cell that depends on a masked attribute** from the
  masked raw graph, from scratch, never by patching the unmasked tensors:
    - ``atom_feat``: own type one-hot (0:28), incident bond-type histogram
      (42:46), radius-1 neighbour-type histogram (46:74), radius-2 atom-type
      histogram (74:102);
    - ``bond_feat``: own bond type one-hot (0:4);
    - ``global_feat``: bond-type fractions (19:23), ``distinct_bond_types``
      (31), ``distinct_atom_types`` (32);
  every other cell is a pure function of the topology and is copied verbatim
  (and gate G2/G8 proves it is type-invariant bit-for-bit);
* the masked view is verified bit-identical to a **full rebuild** with
  ``structural_relation_objects_v1.build_mol_objects`` on the masked raw graph
  (gate G3), so the cheap path is the production feature path, not a patch.

The hidden truth never enters the batch dict: it is returned separately and is
used only as a cross-entropy target.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import s_common as SC

ATOM_MASK_TOKEN = SC.ATOM_MASK_TOKEN
BOND_MASK_TOKEN = SC.BOND_MASK_TOKEN
ATOM_MASK_RATIO = SC.ATOM_MASK_RATIO
BOND_MASK_RATIO = SC.BOND_MASK_RATIO


# ---------------------------------------------------------------------------
# mask sampling — identical mechanism to bjoc_masked_pretraining_v1
# ---------------------------------------------------------------------------
def mask_seed_sequence(seed: int, mol_id: int, visit_index: int) -> np.random.SeedSequence:
    return np.random.SeedSequence([int(seed), int(mol_id), int(visit_index)])


def sample_mask(seed: int, mol_id: int, visit_index: int, n_atoms: int, n_bonds: int):
    """``max(1, round(ratio * n))`` objects of each kind, without replacement."""
    rng = np.random.default_rng(mask_seed_sequence(seed, mol_id, visit_index))
    n_atoms, n_bonds = int(n_atoms), int(n_bonds)
    if n_atoms > 0:
        k = max(1, int(round(ATOM_MASK_RATIO * n_atoms)))
        atom_ids = np.sort(rng.choice(n_atoms, size=k, replace=False)).astype(np.int64)
    else:
        atom_ids = np.zeros(0, dtype=np.int64)
    if n_bonds > 0:
        k = max(1, int(round(BOND_MASK_RATIO * n_bonds)))
        bond_ids = np.sort(rng.choice(n_bonds, size=k, replace=False)).astype(np.int64)
    else:
        bond_ids = np.zeros(0, dtype=np.int64)
    return atom_ids, bond_ids


# ---------------------------------------------------------------------------
# topology cache (type-independent by construction)
# ---------------------------------------------------------------------------
def build_topology(cache: Mapping[str, Any]) -> dict:
    """All type-*independent* feature cells + the radius-2 support."""
    atom_ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)
    bond_ptr = np.asarray(cache["bond_ptr"], dtype=np.int64)
    bond_index = np.asarray(cache["bond_index"], dtype=np.int64)
    atom_feat = np.asarray(cache["atom_feat"], dtype=np.float32)
    bond_feat = np.asarray(cache["bond_feat"], dtype=np.float32)
    global_feat = np.asarray(cache["global_feat"], dtype=np.float32)
    n_mol = len(atom_ptr) - 1

    d2_ptr = np.zeros(n_mol + 1, dtype=np.int64)
    d2_src: list[np.ndarray] = []
    d2_dst: list[np.ndarray] = []
    for m in range(n_mol):
        a0, a1 = int(atom_ptr[m]), int(atom_ptr[m + 1])
        n = a1 - a0
        b0, b1 = int(bond_ptr[m]), int(bond_ptr[m + 1])
        if n == 0:
            d2_ptr[m + 1] = d2_ptr[m]
            continue
        bi = bond_index[b0:b1]
        adj = np.zeros((n, n), dtype=np.int32)
        if len(bi):
            adj[bi[:, 0], bi[:, 1]] = 1
            adj[bi[:, 1], bi[:, 0]] = 1
        a2 = adj @ adj
        sel = (a2 > 0) & (adj == 0)
        np.fill_diagonal(sel, False)
        src, dst = np.nonzero(sel)
        d2_src.append(src.astype(np.int64))
        d2_dst.append(dst.astype(np.int64))
        d2_ptr[m + 1] = d2_ptr[m] + len(src)

    return {
        "n_mol": int(n_mol),
        "atom_ptr": atom_ptr,
        "bond_ptr": bond_ptr,
        "bond_index": bond_index,
        "atom_topo": atom_feat[:, SC.ATOM_TOPO_COLS].copy(),
        "bond_topo": bond_feat[:, SC.BOND_TOPO_COLS].copy(),
        "global_topo": global_feat[:, SC.GLOBAL_TOPO_COLS].copy(),
        "d2_ptr": d2_ptr,
        "d2_src": (np.concatenate(d2_src) if d2_src else np.zeros(0, np.int64)),
        "d2_dst": (np.concatenate(d2_dst) if d2_dst else np.zeros(0, np.int64)),
        "n_atoms": np.diff(atom_ptr).astype(np.int64),
        "n_bonds": np.diff(bond_ptr).astype(np.int64),
    }


def topology_hash(topo: Mapping[str, Any]) -> str:
    import hashlib

    h = hashlib.sha256()
    for key in ("atom_topo", "bond_topo", "global_topo", "d2_ptr", "d2_src", "d2_dst",
                "bond_index", "atom_ptr", "bond_ptr"):
        v = np.ascontiguousarray(topo[key])
        h.update(key.encode())
        h.update(str(v.dtype).encode())
        h.update(str(v.shape).encode())
        h.update(v.tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# masked feature rebuild (per molecule)
# ---------------------------------------------------------------------------
def molecule_features(topo: Mapping[str, Any], m: int, atom_types: np.ndarray,
                      bond_types: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """108-d atom features, 7-d bond features, 33-d global features.

    ``atom_types`` / ``bond_types`` are the **masked** raw codes.  Every
    type-dependent cell is recomputed from them; every other cell is copied
    from the topology cache.  No unmasked type array is read anywhere.
    """
    a0, a1 = int(topo["atom_ptr"][m]), int(topo["atom_ptr"][m + 1])
    b0, b1 = int(topo["bond_ptr"][m]), int(topo["bond_ptr"][m + 1])
    n, n_b = a1 - a0, b1 - b0
    atom_types = np.asarray(atom_types, dtype=np.int64).reshape(-1)
    bond_types = np.asarray(bond_types, dtype=np.int64).reshape(-1)
    if atom_types.shape[0] != n or bond_types.shape[0] != n_b:
        raise ValueError(f"molecule {m}: type arrays do not match the topology")

    af = np.zeros((n, 108), dtype=np.float32)
    af[:, SC.ATOM_TOPO_COLS] = topo["atom_topo"][a0:a1]
    if n:
        af[np.arange(n), atom_types] = 1.0
    bf = np.zeros((n_b, 7), dtype=np.float32)
    bf[:, SC.BOND_TOPO_COLS] = topo["bond_topo"][b0:b1]
    if n_b:
        bf[np.arange(n_b), bond_types] = 1.0

    if n:
        flat = af.reshape(-1)
        if n_b:
            bi = topo["bond_index"][b0:b1]
            u, v = bi[:, 0], bi[:, 1]
            # incident bond-type histogram (both endpoints)
            np.add.at(flat, np.concatenate([u * 108 + 42 + bond_types,
                                            v * 108 + 42 + bond_types]), 1.0)
            # radius-1 neighbour-type histogram (both endpoints, both directions)
            np.add.at(flat, np.concatenate([u * 108 + 46 + atom_types[v],
                                            v * 108 + 46 + atom_types[u]]), 1.0)
        p0, p1 = int(topo["d2_ptr"][m]), int(topo["d2_ptr"][m + 1])
        if p1 > p0:
            src = topo["d2_src"][p0:p1]
            dst = topo["d2_dst"][p0:p1]
            np.add.at(flat, src * 108 + 74 + atom_types[dst], 1.0)

    g = np.zeros(33, dtype=np.float64)
    g[SC.GLOBAL_TOPO_COLS] = topo["global_topo"][m]
    counts = np.bincount(bond_types, minlength=4).astype(np.float64)
    if n_b:
        counts /= float(n_b)
    g[SC.GLOBAL_BOND_FRAC] = counts
    g[SC.GLOBAL_DISTINCT_BOND] = float(len(set(int(t) for t in bond_types)))
    g[SC.GLOBAL_DISTINCT_ATOM] = float(len(set(int(t) for t in atom_types)))
    return af, bf, np.asarray(g, dtype=np.float32)


def masked_molecule(cache: Mapping[str, Any], topo: Mapping[str, Any], m: int, *,
                    seed: int, visit_index: int) -> dict:
    """One masked molecule: masked features + the hidden truth kept outside."""
    a0, a1 = int(topo["atom_ptr"][m]), int(topo["atom_ptr"][m + 1])
    b0, b1 = int(topo["bond_ptr"][m]), int(topo["bond_ptr"][m + 1])
    true_atoms = np.asarray(cache["atom_type"], dtype=np.int64)[a0:a1].copy()
    true_bonds = np.asarray(cache["bond_type"], dtype=np.int64)[b0:b1].copy()
    atom_ids, bond_ids = sample_mask(seed, m, visit_index, a1 - a0, b1 - b0)
    masked_atoms = true_atoms.copy()
    masked_atoms[atom_ids] = ATOM_MASK_TOKEN
    masked_bonds = true_bonds.copy()
    masked_bonds[bond_ids] = BOND_MASK_TOKEN
    af, bf, g = molecule_features(topo, m, masked_atoms, masked_bonds)
    return {
        "mol": int(m),
        "atom_ids": atom_ids,
        "bond_ids": bond_ids,
        "atom_targets": true_atoms[atom_ids],
        "bond_targets": true_bonds[bond_ids],
        "atom_feat": af,
        "bond_feat": bf,
        "global_feat": g,
        "masked_atom_types": masked_atoms,
        "masked_bond_types": masked_bonds,
        "n_atoms": int(a1 - a0),
        "n_bonds": int(b1 - b0),
    }


# ---------------------------------------------------------------------------
# batch assembly
# ---------------------------------------------------------------------------
def masked_batch(cache: Mapping[str, Any], rel: Mapping[str, Any], topo: Mapping[str, Any],
                 rows: Sequence[int], *, seed: int, visit_index: int) -> dict:
    """Assemble one pretext batch (features + mask targets).

    The returned dict has exactly the keys of ``bosi_data.collate`` (so the
    frozen model forward runs unchanged) plus ``masked_atom_ids`` /
    ``masked_bond_ids`` (batch-local row ids).  The cross-entropy targets are
    returned **outside** the batch dict.
    """
    rows = np.asarray(rows, dtype=np.int64)
    batch = SC.BD.collate(cache, rel, rows)

    atoms, bonds, globs, masked_bt = [], [], [], []
    masked_atom_rows, masked_bond_rows = [], []
    atom_targets, bond_targets = [], []
    mol_meta = []
    a_off = b_off = 0
    for r in rows:
        mm = masked_molecule(cache, topo, int(r), seed=seed, visit_index=visit_index)
        atoms.append(mm["atom_feat"])
        bonds.append(mm["bond_feat"])
        globs.append(mm["global_feat"])
        masked_bt.append(mm["masked_bond_types"])
        if len(mm["atom_ids"]):
            masked_atom_rows.append(mm["atom_ids"] + a_off)
            atom_targets.append(mm["atom_targets"])
        if len(mm["bond_ids"]):
            masked_bond_rows.append(mm["bond_ids"] + b_off)
            bond_targets.append(mm["bond_targets"])
        mol_meta.append({"row": int(r), "n_atoms": mm["n_atoms"], "n_bonds": mm["n_bonds"],
                         "atom_offset": a_off, "bond_offset": b_off,
                         "n_masked_atoms": int(len(mm["atom_ids"])),
                         "n_masked_bonds": int(len(mm["bond_ids"]))})
        a_off += mm["n_atoms"]
        b_off += mm["n_bonds"]

    if a_off != int(batch["atom_feat"].shape[0]) or b_off != int(batch["bond_feat"].shape[0]):
        raise RuntimeError("masked rebuild disagrees with the frozen collate offsets")

    batch["atom_feat"] = torch.from_numpy(np.concatenate(atoms, axis=0))
    batch["bond_feat"] = torch.from_numpy(np.concatenate(bonds, axis=0))
    batch["global"] = torch.from_numpy(np.stack(globs, axis=0))
    # the raw bond_type key is not consumed by the forward, but it must not carry
    # unmasked truth either
    batch["bond_type"] = torch.from_numpy(np.concatenate(masked_bt).astype(np.int64))

    batch["masked_atom_ids"] = (torch.from_numpy(np.concatenate(masked_atom_rows))
                                if masked_atom_rows else torch.zeros(0, dtype=torch.long))
    batch["masked_bond_ids"] = (torch.from_numpy(np.concatenate(masked_bond_rows))
                                if masked_bond_rows else torch.zeros(0, dtype=torch.long))
    return {
        "batch": batch,
        "atom_targets": (torch.from_numpy(np.concatenate(atom_targets))
                         if atom_targets else torch.zeros(0, dtype=torch.long)),
        "bond_targets": (torch.from_numpy(np.concatenate(bond_targets))
                         if bond_targets else torch.zeros(0, dtype=torch.long)),
        "molecules": mol_meta,
    }


def reference_masked_batch(cache: Mapping[str, Any], rel: Mapping[str, Any],
                           topo: Mapping[str, Any], rows: Sequence[int], *, seed: int,
                           visit_index: int) -> dict:
    """The *full-rebuild* reference used only by the audit.

    Builds a masked copy of the split cache (all three feature tensors replaced
    for the requested molecules by ``build_mol_objects`` on the masked raw
    graph) and hands it to the frozen ``bosi_data.collate``.  Also recomputes
    the relation arrays of the masked graph and asserts they are unchanged.
    """
    rows = np.asarray(rows, dtype=np.int64)
    cache_m = {k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v)
               for k, v in cache.items()}
    bi_all = np.asarray(cache["bond_index"], dtype=np.int64)
    bt_all = np.asarray(cache["bond_type"], dtype=np.int64)
    at_all = np.asarray(cache["atom_type"], dtype=np.int64)
    for r in rows:
        mm = masked_molecule(cache, topo, int(r), seed=seed,
                             visit_index=visit_index)
        a0 = int(np.asarray(cache["atom_ptr"])[r])
        b0 = int(np.asarray(cache["bond_ptr"])[r])
        n_a, n_b = mm["n_atoms"], mm["n_bonds"]
        cache_m["atom_feat"][a0:a0 + n_a] = mm["atom_feat"]
        cache_m["bond_feat"][b0:b0 + n_b] = mm["bond_feat"]
        cache_m["global_feat"][r] = mm["global_feat"]
        cache_m["atom_type"][a0:a0 + n_a] = mm["masked_atom_types"]
        cache_m["bond_type"][b0:b0 + n_b] = mm["masked_bond_types"]
        _assert_relations_unchanged(
            n_a, bi_all[b0:b0 + n_b], bt_all[b0:b0 + n_b],
            mm["masked_bond_types"])
    return SC.BD.collate(cache_m, rel, rows)


def full_rebuild_molecule(cache: Mapping[str, Any], topo: Mapping[str, Any], m: int, *,
                          seed: int, visit_index: int):
    """``build_mol_objects`` on the masked raw graph (audit reference)."""
    import structural_relation_objects_v1 as sro

    a0 = int(np.asarray(cache["atom_ptr"])[m])
    a1 = int(np.asarray(cache["atom_ptr"])[m + 1])
    b0 = int(np.asarray(cache["bond_ptr"])[m])
    b1 = int(np.asarray(cache["bond_ptr"])[m + 1])
    mm = masked_molecule(cache, topo, m, seed=seed, visit_index=visit_index)
    atom_types = mm["masked_atom_types"]
    bond_types = mm["masked_bond_types"]
    bi = np.asarray(cache["bond_index"], dtype=np.int64)[b0:b1]
    ei = np.stack([bi[:, 0], bi[:, 1]], axis=1)
    ei2 = np.concatenate([ei, ei[:, ::-1]], axis=0).T.copy()
    ea = np.concatenate([bond_types, bond_types], axis=0)
    return mm, sro.build_mol_objects(int(m), atom_types, ei2, ea)


# ---------------------------------------------------------------------------
# relation invariance helper
# ---------------------------------------------------------------------------
def relation_content_for(n_atoms: int, bond_index: np.ndarray, bond_type: np.ndarray) -> dict:
    return SC.BD.relation_arrays(int(n_atoms), np.asarray(bond_index, dtype=np.int64),
                                 np.asarray(bond_type, dtype=np.int64))


def _assert_relations_unchanged(n_atoms: int, bond_index: np.ndarray,
                                bond_type_true: np.ndarray,
                                bond_type_masked: np.ndarray) -> None:
    a = relation_content_for(n_atoms, bond_index, bond_type_true)
    b = relation_content_for(n_atoms, bond_index, bond_type_masked)
    for key in a:
        va, vb = a[key], b[key]
        if not np.array_equal(np.asarray(va), np.asarray(vb)):
            raise AssertionError(f"relation field {key} changed under type masking")


def relation_cache_under_masking(cache: Mapping[str, Any], topo: Mapping[str, Any],
                                 rows: Sequence[int], seed: int, visit_index: int) -> dict:
    """All relation fields of the masked graphs, rebuilt from scratch."""
    out = {"partner_rel": [], "closure": [], "half_deg": [], "partner_ptr": [],
           "n_tgt": 0}
    for r in np.asarray(rows, dtype=np.int64):
        mm = masked_molecule(cache, topo, int(r), seed=seed,
                             visit_index=visit_index)
        a0 = int(np.asarray(cache["atom_ptr"])[r])
        b0 = int(np.asarray(cache["bond_ptr"])[r])
        bi = np.asarray(cache["bond_index"], dtype=np.int64)[b0:b0 + mm["n_bonds"]]
        bt = np.asarray(cache["bond_type"], dtype=np.int64)[b0:b0 + mm["n_bonds"]]
        bm = mm["masked_bond_types"]
        a = relation_content_for(mm["n_atoms"], bi, bt)
        b = relation_content_for(mm["n_atoms"], bi, bm)
        for key in ("partner_rel", "closure", "half_deg", "partner_ptr"):
            if not np.array_equal(np.asarray(a[key]), np.asarray(b[key])):
                raise AssertionError(f"relation field {key} changed under type masking")
        out["partner_rel"].append(b["partner_rel"])
    return out
