"""Masked atom/bond structural pretraining for BJOC (cscl-bjoc-masked-pretraining-v1).

Protocol: ``configs/bjoc_masked_pretraining_v1.yaml``.

This module implements the *only* self-supervised task of the round:

* sample a deterministic mask over raw atom-type codes (15%) and raw physical
  bond-type codes (15%) from ``seed + molecule ID + visit index``;
* replace the masked raw codes by dedicated mask tokens (atom 27, bond 0),
  **before** any BJOC object feature is derived;
* rebuild every masked BJOC object field from the masked raw graph
  (``structural_relation_objects_v1.build_mol_objects``) so no type-dependent
  histogram / junction / global field can leak the hidden truth;
* predict the hidden raw types from the masked-graph encoder embeddings with
  two shared-MLP pretext heads (atom 28-way, bond 4-way), using static
  incidence gathers only (no message passing, no extra learned rounds);
* ``L_SSL = CE(masked atoms) + CE(masked physical bonds)`` with 1.0/1.0.

Discipline
----------
* ``y`` is never read (not even imported) in any function here.  Pretext
  targets are raw ``x`` / ``edge_attr`` values, saved outside the model input.
* Only the three published encoders are trained; downstream
  ``group_composer`` and ``reader`` are untouched by pretraining.
* The mask RNG is a :class:`numpy.random.SeedSequence` over
  ``(seed, molecule_id, visit_index)`` and is independent of the data-shuffle
  ``torch.Generator``.
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = _CODE_DIR.parents[2]
for _p in (str(_CODE_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cscl_units as cu  # noqa: E402
import structural_relation_composer_v1 as src  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402

PROTOCOL_ID = "cscl-bjoc-masked-pretraining-v1"

#: dedicated mask tokens inside the frozen feature vocabulary
ATOM_MASK_TOKEN = 27   # unused ZINC atom class inside ATOM_CATEGORIES = 28
BOND_MASK_TOKEN = 0    # unused raw edge_attr class inside BOND_CATEGORIES = 4

#: frozen mask ratios
ATOM_MASK_RATIO = 0.15
BOND_MASK_RATIO = 0.15

#: frozen pretraining budget
PRETRAIN_UPDATES = 7560
PRETRAIN_BATCH = 128
PRETRAIN_LR = 1.0e-3
PRETRAIN_WD = 1.0e-5
PRETRAIN_CLIP = 5.0
PRETRAIN_SHUFFLE_OFFSET = 202
ATOM_HEAD_HIDDEN = 160
BOND_HEAD_HIDDEN = 160


# ---------------------------------------------------------------------------
# mask sampling
# ---------------------------------------------------------------------------


def mask_seed_sequence(seed: int, mol_id: int, visit_index: int) -> np.random.SeedSequence:
    """Deterministic, independent mask entropy for (seed, molecule, visit)."""
    return np.random.SeedSequence([int(seed), int(mol_id), int(visit_index)])


def sample_mask(
    seed: int,
    mol_id: int,
    visit_index: int,
    n_atoms: int,
    n_bonds: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Frozen mask: ``max(1, round(0.15 n))`` objects of each kind.

    Atoms and physical bonds are drawn without replacement, independently.
    Each molecule with atoms (bonds) masks at least one atom (bond).
    """
    rng = np.random.default_rng(mask_seed_sequence(seed, mol_id, visit_index))
    n_atoms, n_bonds = int(n_atoms), int(n_bonds)
    if n_atoms > 0:
        k_a = max(1, int(round(ATOM_MASK_RATIO * n_atoms)))
        atom_ids = np.sort(rng.choice(n_atoms, size=k_a, replace=False)).astype(np.int64)
    else:
        atom_ids = np.zeros(0, dtype=np.int64)
    if n_bonds > 0:
        k_b = max(1, int(round(BOND_MASK_RATIO * n_bonds)))
        bond_ids = np.sort(rng.choice(n_bonds, size=k_b, replace=False)).astype(np.int64)
    else:
        bond_ids = np.zeros(0, dtype=np.int64)
    return atom_ids, bond_ids


# ---------------------------------------------------------------------------
# raw-graph masking (token replacement before feature construction)
# ---------------------------------------------------------------------------


def directed_bond_index(edge_index: np.ndarray) -> dict[tuple[int, int], list[int]]:
    """Map every physical bond ``(u, v)`` with u < v to its two column ids."""
    mapping: dict[tuple[int, int], list[int]] = {}
    ei = np.asarray(edge_index)
    for col in range(ei.shape[1]):
        a, b = int(ei[0, col]), int(ei[1, col])
        key = (a, b) if a < b else (b, a)
        mapping.setdefault(key, []).append(col)
    return mapping


def mask_raw_arrays(
    x: np.ndarray,
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
    bond_index: np.ndarray,
    atom_ids: np.ndarray,
    bond_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Replace masked raw codes by the dedicated mask tokens.

    ``bond_index`` is the *physical* bond table (``u < v``) of the original
    object.  Every physical bond has exactly two directed PyG records and both
    are replaced together.
    """
    x_m = np.asarray(x).copy().reshape(-1).astype(np.int64)
    x_m[np.asarray(atom_ids, dtype=np.int64)] = ATOM_MASK_TOKEN
    ea_m = np.asarray(edge_attr).copy().reshape(-1).astype(np.int64)
    if len(bond_ids):
        mapping = directed_bond_index(edge_index)
        for k in np.asarray(bond_ids, dtype=np.int64):
            u, v = int(bond_index[k][0]), int(bond_index[k][1])
            cols = mapping[(min(u, v), max(u, v))]
            for col in cols:
                ea_m[col] = BOND_MASK_TOKEN
    return x_m.reshape(np.asarray(x).shape), ea_m.reshape(np.asarray(edge_attr).shape)


@dataclass
class PretextSample:
    """A masked object plus the hidden truth kept *outside* the model input."""

    masked: sro.MolObjects
    atom_ids: np.ndarray        # row ids inside the mol object
    atom_types: np.ndarray      # hidden true raw atom-type codes
    bond_ids: np.ndarray        # physical bond ids inside the mol object
    bond_types: np.ndarray      # hidden true raw bond-type codes


def build_masked_sample(
    base: sro.MolObjects,
    x: np.ndarray,
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
    *,
    seed: int,
    visit_index: int,
) -> PretextSample:
    """Full-rebuild masked sample: replace raw codes, then rebuild all BJOC
    features from the masked raw graph (no partial / incremental update)."""
    bond_index = np.asarray(base.bond_index, dtype=np.int64)
    atom_ids, bond_ids = sample_mask(
        seed, base.mol_index, visit_index, base.n_atoms, bond_index.shape[0]
    )
    x_m, ea_m = mask_raw_arrays(x, edge_index, edge_attr, bond_index, atom_ids, bond_ids)
    masked = sro.build_mol_objects(base.mol_index, x_m, edge_index, ea_m, smiles="")
    atom_types = np.asarray([int(np.asarray(x).reshape(-1)[i]) for i in atom_ids], dtype=np.int64)
    bond_types = np.asarray([int(base.bond_type[i]) for i in bond_ids], dtype=np.int64)
    return PretextSample(
        masked=masked,
        atom_ids=atom_ids,
        atom_types=atom_types,
        bond_ids=bond_ids,
        bond_types=bond_types,
    )


# ---------------------------------------------------------------------------
# dependency map (which feature cells a hidden raw type can reach)
# ---------------------------------------------------------------------------


def atom_type_dependent_cells(base: sro.MolObjects, atom_id: int) -> dict[str, Any]:
    """Cells of the *unmasked* feature tensors whose value depends on the raw
    atom-type code of ``atom_id`` (used by the leakage audit)."""
    blocks = sro.atom_blocks()
    n = base.n_atoms
    adj = [[] for _ in range(n)]
    for u, v in np.asarray(base.bond_index):
        adj[int(u)].append(int(v))
        adj[int(v)].append(int(u))
    # distance exactly 2 neighbours (deterministic BFS)
    dist2: list[int] = []
    for s in adj[atom_id]:
        for t in adj[s]:
            if t != atom_id and t not in adj[atom_id] and t not in dist2:
                dist2.append(t)
    return {
        "atom_type_row": [int(atom_id)],
        "neighbour_hist_rows": sorted(adj[atom_id]),
        "radius2_hist_rows": sorted(dist2),
        "global_distinct_atom_types": True,
    }


def bond_type_dependent_cells(base: sro.MolObjects, bond_id: int) -> dict[str, Any]:
    """Cells whose value depends on the raw bond-type code of physical bond
    ``bond_id`` (unmasked tensors)."""
    u, v = (int(z) for z in base.bond_index[bond_id])
    touching_junc = [j for j, (b1, b2) in enumerate(np.asarray(base.junc_bond)) if bond_id in (int(b1), int(b2))]
    return {
        "bond_onehot_row": int(bond_id),
        "atom_bond_type_counts_rows": sorted({u, v}),
        "junction_type_sum_rows": sorted(touching_junc),
        "junction_type_outer_rows": sorted(touching_junc),
        "global_bond_type_fractions": True,
        "global_distinct_bond_types": True,
    }


def dependent_cell_mask(
    base: sro.MolObjects,
    atom_ids: Sequence[int],
    bond_ids: Sequence[int],
) -> dict[str, np.ndarray]:
    """Boolean masks over the four feature tensors marking every cell that may
    change when the listed raw types are hidden."""
    blocks = sro.atom_blocks()
    atom_mask = np.zeros_like(base.atom_feat, dtype=bool)
    bond_mask = np.zeros_like(base.bond_feat, dtype=bool)
    junc_mask = np.zeros_like(base.junc_feat, dtype=bool)
    global_mask = np.zeros_like(base.global_feat, dtype=bool)
    off_bond_counts = blocks["bond_type_counts"].start  # 4 slots
    off_sum = 1 + 1 + 1 + 1 + 1 + 1 + 6  # junc topology fields before type sum
    off_outer = off_sum + 4
    for a in atom_ids:
        dep = atom_type_dependent_cells(base, int(a))
        for r in dep["atom_type_row"]:
            atom_mask[r, blocks["atom_type"]] = True
        for r in dep["neighbour_hist_rows"]:
            atom_mask[r, blocks["neighbour_types"]] = True
        for r in dep["radius2_hist_rows"]:
            atom_mask[r, blocks["radius2_types"]] = True
        global_mask[32] = True  # distinct_atom_types
    for b in bond_ids:
        dep = bond_type_dependent_cells(base, int(b))
        atom_mask[dep["atom_bond_type_counts_rows"], off_bond_counts : off_bond_counts + 4] = True
        bond_mask[dep["bond_onehot_row"], :4] = True
        for r in dep["junction_type_sum_rows"]:
            junc_mask[r, off_sum : off_sum + 4] = True
        for r in dep["junction_type_outer_rows"]:
            junc_mask[r, off_outer : off_outer + 16] = True
        global_mask[19:23] = True  # bond_type fractions
        global_mask[31] = True     # distinct_bond_types
    return {"atom": atom_mask, "bond": bond_mask, "junc": junc_mask, "global": global_mask}


# ---------------------------------------------------------------------------
# pretext model
# ---------------------------------------------------------------------------


class PretextModel(nn.Module):
    """The published BJOC encoders + two discarded reconstruction heads.

    The backbone is held by reference so that ``backbone`` parameters are the
    exact ``BJOC`` modules; only the three encoders are ever used downstream.
    """

    def __init__(self, backbone: src.BJOC) -> None:
        super().__init__()
        self.backbone = backbone
        atom_out = int(backbone.config.atom_out)
        bond_out = int(backbone.config.bond_out)
        junc_out = int(backbone.config.junc_out)
        self.atom_head = src._MLP(4 * atom_out, ATOM_HEAD_HIDDEN, sro.ATOM_CATEGORIES, 1, 0.0)
        self.bond_head = src._MLP(4 * bond_out + junc_out, BOND_HEAD_HIDDEN, sro.BOND_CATEGORIES, 1, 0.0)

    # -- static incidence gathers (one-shot, no learned rounds) --------------
    @staticmethod
    def _incidence(batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        bi = batch["bond_index"]
        n_bonds = int(bi.shape[0])
        ji = batch["junc_index"]
        n_junc = int(ji.shape[0])
        dev = bi.device
        bond_range = torch.arange(n_bonds, device=dev, dtype=torch.long)
        junc_range = torch.arange(n_junc, device=dev, dtype=torch.long)
        return {
            "hole_atom": torch.cat([bi[:, 0], bi[:, 1]]),          # atom <- incident bond
            "hole_bond": torch.cat([bond_range, bond_range]),
            "junc_center_atom": ji[:, 1],                           # atom <- junction centre
            "junc_endpoint_atom": torch.cat([ji[:, 0], ji[:, 2]]),  # atom <- junction endpoint
            "junc_endpoint_row": torch.cat([junc_range, junc_range]),
            "jb1": batch["junc_bond"][:, 0],                        # bond <- touching junction
            "jb2": batch["junc_bond"][:, 1],
        }

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        masked_atom_ids: torch.Tensor,
        masked_bond_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        backbone = self.backbone
        a = backbone.encode_atoms(batch)
        b = backbone.encode_bonds(a, batch)
        w = backbone.encode_junctions(a, b, batch)
        inc = self._incidence(batch)
        n_atoms = int(a.shape[0])
        n_bonds = int(b.shape[0])
        n_junc = int(w.shape[0])

        sum_b = torch.zeros((n_atoms, b.shape[1]), device=a.device, dtype=a.dtype)
        sum_b = sum_b.index_add(0, inc["hole_atom"], b[inc["hole_bond"]])
        sum_wc = torch.zeros((n_atoms, w.shape[1]), device=a.device, dtype=a.dtype)
        sum_wc = sum_wc.index_add(0, inc["junc_center_atom"], w)
        sum_we = torch.zeros((n_atoms, w.shape[1]), device=a.device, dtype=a.dtype)
        sum_we = sum_we.index_add(0, inc["junc_endpoint_atom"], w[inc["junc_endpoint_row"]])
        cat_atom = torch.cat([a, sum_b, sum_wc, sum_we], dim=1)

        sum_wb = torch.zeros((n_bonds, w.shape[1]), device=a.device, dtype=a.dtype)
        sum_wb = sum_wb.index_add(0, inc["jb1"], w)
        sum_wb = sum_wb.index_add(0, inc["jb2"], w)
        bi = batch["bond_index"]
        au, av = a[bi[:, 0]], a[bi[:, 1]]
        cat_bond = torch.cat([b, au + av, (au - av).abs(), au * av, sum_wb], dim=1)

        atom_ctx = cat_atom[masked_atom_ids]
        bond_ctx = cat_bond[masked_bond_ids]
        return {
            "a": a,
            "b": b,
            "w": w,
            "atom_ctx": atom_ctx,
            "bond_ctx": bond_ctx,
            "atom_logits": self.atom_head(atom_ctx),
            "bond_logits": self.bond_head(bond_ctx),
            "sum_wb": sum_wb,
        }

    def loss(
        self,
        batch: Mapping[str, torch.Tensor],
        atom_targets: torch.Tensor,
        bond_targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        logits = self(batch, batch["masked_atom_ids"], batch["masked_bond_ids"])
        ce_atom = F.cross_entropy(logits["atom_logits"], atom_targets) if atom_targets.numel() else None
        ce_bond = F.cross_entropy(logits["bond_logits"], bond_targets) if bond_targets.numel() else None
        if ce_atom is None and ce_bond is None:
            raise ValueError("empty pretext batch")
        loss = torch.zeros((), device=logits["a"].device, dtype=logits["a"].dtype)
        out: dict[str, torch.Tensor] = {}
        if ce_atom is not None:
            loss = loss + ce_atom
            out["ce_atom"] = ce_atom
        if ce_bond is not None:
            loss = loss + ce_bond
            out["ce_bond"] = ce_bond
        out["loss"] = loss
        return loss, out


# ---------------------------------------------------------------------------
# batch assembly for the pretext task
# ---------------------------------------------------------------------------


def batch_pretext_samples(
    samples: Sequence[PretextSample],
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    """Assemble masked objects into a BJOC batch plus flat pretext labels.

    Returns ``(batch, atom_targets, bond_targets, atom_global_ids, bond_global_ids)``
    where the target tensors are deliberately **not** part of ``batch``.
    """
    batch = sro.batch_objects([s.masked for s in samples])
    a_off, b_off = 0, 0
    atom_rows, atom_targets = [], []
    bond_rows, bond_targets = [], []
    for s in samples:
        atom_rows.append(torch.as_tensor(s.atom_ids + a_off, dtype=torch.long))
        atom_targets.append(torch.as_tensor(s.atom_types, dtype=torch.long))
        bond_rows.append(torch.as_tensor(s.bond_ids + b_off, dtype=torch.long))
        bond_targets.append(torch.as_tensor(s.bond_types, dtype=torch.long))
        a_off += s.masked.n_atoms
        b_off += s.masked.bond_index.shape[0]
    batch["masked_atom_ids"] = torch.cat(atom_rows, 0) if atom_rows else torch.zeros(0, dtype=torch.long)
    batch["masked_bond_ids"] = torch.cat(bond_rows, 0) if bond_rows else torch.zeros(0, dtype=torch.long)
    atom_targets_t = torch.cat(atom_targets, 0) if atom_targets else torch.zeros(0, dtype=torch.long)
    bond_targets_t = torch.cat(bond_targets, 0) if bond_targets else torch.zeros(0, dtype=torch.long)
    return (
        batch,
        atom_targets_t,
        bond_targets_t,
        batch["masked_atom_ids"].numpy().copy(),
        batch["masked_bond_ids"].numpy().copy(),
    )


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def encoder_state(model: nn.Module) -> dict[str, dict[str, torch.Tensor]]:
    """Extract the three pretrained encoder parameter groups only."""
    return {
        "atom_encoder": {k: v.detach().cpu().clone() for k, v in model.atom_encoder.state_dict().items()},
        "bond_encoder": {k: v.detach().cpu().clone() for k, v in model.bond_encoder.state_dict().items()},
        "junction_encoder": {k: v.detach().cpu().clone() for k, v in model.junction_encoder.state_dict().items()},
    }


def load_encoder_state(model: src.BJOC, state: Mapping[str, Mapping[str, torch.Tensor]], device: torch.device | str = "cpu") -> None:
    model.atom_encoder.load_state_dict({k: v.to(device) for k, v in state["atom_encoder"].items()})
    model.bond_encoder.load_state_dict({k: v.to(device) for k, v in state["bond_encoder"].items()})
    model.junction_encoder.load_state_dict({k: v.to(device) for k, v in state["junction_encoder"].items()})


def group_state_hash(model: src.BJOC) -> str:
    """Hash of the *downstream* non-encoder groups (composer + reader)."""
    h = hashlib.sha256()
    for name in ("group_composer", "reader"):
        module = getattr(model, name)
        for k, v in sorted(module.state_dict().items()):
            h.update(f"{name}.{k}".encode())
            h.update(np.ascontiguousarray(v.detach().cpu().numpy(), dtype=np.float32).tobytes())
    return h.hexdigest()


def is_masked(atom_code: int) -> bool:
    return int(atom_code) == ATOM_MASK_TOKEN


def class_prior_stats(atom_codes: np.ndarray, bond_codes: np.ndarray) -> dict[str, Any]:
    """Entropy / majority-class baselines for the two reconstruction targets."""
    out: dict[str, Any] = {}
    for name, codes, n_classes in (
        ("atom", np.asarray(atom_codes, dtype=np.int64), sro.ATOM_CATEGORIES),
        ("bond", np.asarray(bond_codes, dtype=np.int64), sro.BOND_CATEGORIES),
    ):
        counts = np.bincount(codes, minlength=n_classes).astype(np.float64)
        p = counts / max(counts.sum(), 1.0)
        nz = p[p > 0]
        entropy = float(-(nz * np.log(nz)).sum())
        out[name] = {
            "n": int(counts.sum()),
            "counts": counts.astype(int).tolist(),
            "entropy_nats": entropy,
            "majority_class": int(np.argmax(counts)),
            "majority_fraction": float(p.max()),
            "marginal_ce_nats": entropy,
            "majority_accuracy": float(p.max()),
        }
    return out
