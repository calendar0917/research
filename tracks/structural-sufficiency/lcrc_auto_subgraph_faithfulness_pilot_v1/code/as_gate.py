"""Physical-bond continuous gating of the frozen LCRC-C1 (arm A) forward.

The intervention is named, always and everywhere,
``CONDITIONAL_S1_S2_EVIDENCE_MASK``.  It is a multiplicative mask ``m[e] in
[0, 1]`` on the *explicit bond-derived evidence* of each deduplicated physical
bond ``e``:

S1 evidence
    * the bond's contribution to both endpoints' ``b_sum`` (the ``b_mean``
      branch does not exist in C1 and is therefore not gated);
    * the ``nbr_emb`` term of the directed neighbour code that the same bond
      produces at each endpoint.

S2 evidence
    For every non-backtracking 2-path occurrence ``p = (root -> middle -> end)``
    with physical bonds ``e1, e2``, the weight ``w[p] = m[e1] * m[e2]`` multiplies
    the occurrence's *entire* contribution: type embedding, endpoint embedding,
    joint projection, closure embedding and the learned content gate applied to
    ``a[end]``.  An occurrence whose two bonds are both masked is removed exactly
    once, because it is one term of one sum.

Not gated at all: ``atom_feat``, ``atom_scalar``, ``global``, atom counts, atom
encodings, and every other structural proxy already baked into the frozen
inputs.  ``BondEncoder`` parameters and the physical bond list are unchanged.

The module owns its own copy of the C1 local aggregation so that the frozen
upstream modules stay read-only; all *parameters* (atom/bond encoders, S1/S2
embeddings, ``LocalMLP``, ``LayerNorm``, ``f``, ``rho`` and the norm buffers) are
used by reference from the loaded arm-A module.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch

import as_common as C
import as_data as D

C.install_import_paths()

import cr_models as M  # noqa: E402  (frozen read-only reuse: arms, widths)

CONDITION_NAME = C.CONDITION_NAME
WIDTH = M.WIDTH


# ---------------------------------------------------------------------------
# gated forward
# ---------------------------------------------------------------------------


def gated_parts(
    arm: torch.nn.Module,
    gb: D.GatedBatch,
    s1: torch.Tensor,
    s2: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Frozen arm-A forward with S1/S2 evidence gates ``s1`` and ``s2``.

    ``s1`` / ``s2`` are ``(n_graphs, max_bonds)`` gate matrices; ``s2=None`` means
    "reuse ``s1``" (the round's default single-matrix intervention).
    """
    if arm.training:
        raise RuntimeError("gated forward requires the frozen model in eval() mode")
    shared_gate = s2 is None or s2 is s1
    if s2 is None:
        s2 = s1
    trunk = arm.trunk
    local = trunk.local
    device = trunk.global_mean.device
    gb = gb.to(device)
    batch = gb.batch
    s1 = s1.to(device)
    s2 = s1 if shared_gate else s2.to(device)

    a = trunk.atom_encoder(batch["atom_feat"])
    idx = batch["bond_index"]
    if idx.numel():
        au, av = a[idx[:, 0]], a[idx[:, 1]]
        bond_input = torch.cat([au + av, (au - av).abs(), au * av, batch["bond_feat"]], dim=1)
    else:
        bond_input = torch.zeros(
            (0, 3 * a.shape[1] + batch["bond_feat"].shape[1]), dtype=a.dtype, device=a.device
        )
    b = trunk.bond_encoder(bond_input)

    bond_gate, nbr_gate = gb.s1_gate(s1)
    b_gated = b * bond_gate.unsqueeze(1) if b.numel() else b

    n_atoms = a.shape[0]
    if idx.numel():
        u, v = idx[:, 0], idx[:, 1]
        degree = a.new_zeros(n_atoms)
        degree = degree.index_add_(0, u, torch.ones_like(u, dtype=a.dtype))
        degree = degree.index_add_(0, v, torch.ones_like(v, dtype=a.dtype))
        b_sum = a.new_zeros((n_atoms, a.shape[1])).index_add_(0, u, b_gated).index_add_(0, v, b_gated)
    else:
        degree = a.new_zeros(n_atoms)
        b_sum = a.new_zeros((n_atoms, a.shape[1]))

    if batch["nbr_code"].numel():
        nbr_emb = local.nbr_emb(batch["nbr_code"]) * nbr_gate.unsqueeze(1)
        nbr = a.new_zeros((n_atoms, local.rank)).index_add_(0, batch["nbr_atom"], nbr_emb)
    else:
        nbr = a.new_zeros((n_atoms, local.rank))

    path_weight = gb.s2_weight(s2)
    if batch["path_cat"].numel():
        w = path_weight.unsqueeze(1)
        parts = [
            local.path_emb(batch["path_cat"]) * w,
            local.end_emb(batch["path_end63"]) * w,
            local.joint_proj(local.path_u(batch["path_cat"]) * local.path_v(batch["path_end63"])) * w,
            local.close_emb(batch["path_closure"]) * w,
        ]
        path_emb = torch.cat(parts, dim=1)
        path_agg = a.new_zeros((n_atoms, path_emb.shape[1])).index_add_(0, batch["path_atom"], path_emb)
        content = a[batch["path_end"]] * local.path_gate(batch["path_cat"]) * w
        content_agg = a.new_zeros((n_atoms, a.shape[1])).index_add_(0, batch["path_atom"], content)
    else:
        path_agg = a.new_zeros((n_atoms, 3 * local.rank + 8))
        content_agg = a.new_zeros((n_atoms, a.shape[1]))

    scal = (batch["atom_scalar"] - trunk.scalar_mean) / trunk.scalar_scale
    x = torch.cat([a, b_sum, nbr, path_agg, content_agg, scal[:, list(local.scalar_cols)]], dim=1)
    if x.shape[1] != len(local.local_cols):
        raise RuntimeError(f"local width {x.shape[1]} != frozen C1 column count {len(local.local_cols)}")
    h = trunk.drop(local.norm(local.mlp(x)))

    atom_batch = batch["atom_batch"]
    n_graphs = int(batch["n_graphs"])
    count = h.new_zeros(n_graphs).index_add_(0, atom_batch, torch.ones_like(atom_batch, dtype=h.dtype))
    denom = count.clamp(min=1.0).unsqueeze(1)
    mean_h = h.new_zeros((n_graphs, WIDTH)).index_add_(0, atom_batch, h) / denom
    mean_h2 = h.new_zeros((n_graphs, WIDTH)).index_add_(0, atom_batch, h * h) / denom
    contrib_atom = arm.f(h).view(-1)
    local_sum = contrib_atom.new_zeros(n_graphs).index_add_(0, atom_batch, contrib_atom)
    g = (batch["global"] - trunk.global_mean) / trunk.global_scale
    rho_in = torch.cat([g, mean_h, mean_h2, torch.log1p(count).unsqueeze(1)], dim=1)
    rho = arm.rho(rho_in).view(-1)
    return {
        "pred_std": local_sum + rho,
        "local_sum": local_sum,
        "rho": rho,
        "contrib_atom": contrib_atom,
        "h": h,
        "count": count,
        "mean_h": mean_h,
        "mean_h2": mean_h2,
        "rho_input": rho_in,
        "degree": degree,
        "term_names": ["local_sum", "rho"],
    }


# ---------------------------------------------------------------------------
# reference forward (unchanged frozen arm A)
# ---------------------------------------------------------------------------


def reference_parts(arm: torch.nn.Module, batch: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    if arm.training:
        raise RuntimeError("reference forward requires the frozen model in eval() mode")
    return arm.parts(batch)


def equivalence_check(
    arm: torch.nn.Module,
    gb: D.GatedBatch,
    *,
    sigma: float,
    mu: float,
) -> dict[str, Any]:
    """All-ones gates must reproduce the frozen arm A exactly (raw-unit tolerance)."""
    with torch.no_grad():
        ref = reference_parts(arm, gb.batch)
        ones = gb.gate_ones(device=ref["pred_std"].device, dtype=ref["pred_std"].dtype)
        gated = gated_parts(arm, gb, ones)
    ref_pred = ref["pred_std"].detach().cpu().numpy().astype(np.float64)
    got_pred = gated["pred_std"].detach().cpu().numpy().astype(np.float64)
    ref_raw = ref_pred * sigma + mu
    got_raw = got_pred * sigma + mu
    pred_err = np.abs(ref_raw - got_raw)
    l_err = np.abs(ref["local_sum"].detach().cpu().numpy().astype(np.float64) * sigma
                   - gated["local_sum"].detach().cpu().numpy().astype(np.float64) * sigma)
    p_err = np.abs(ref["rho"].detach().cpu().numpy().astype(np.float64) * sigma
                   - gated["rho"].detach().cpu().numpy().astype(np.float64) * sigma)
    tol = C.trace_tolerance(ref_raw)
    return {
        "condition": CONDITION_NAME,
        "n_graphs": int(ref_pred.size),
        "pred_raw_max_abs_err": float(pred_err.max()) if pred_err.size else 0.0,
        "pred_raw_tolerance_max": float(np.max(tol)) if np.size(tol) else 0.0,
        "local_sum_max_abs_err_raw": float(l_err.max()) if l_err.size else 0.0,
        "rho_max_abs_err_raw": float(p_err.max()) if p_err.size else 0.0,
        "passed": bool(np.all(pred_err <= tol)),
    }


# ---------------------------------------------------------------------------
# gates on one molecule
# ---------------------------------------------------------------------------


def bond_mask(n_bonds: int, bond_ids, value: float = 0.0) -> np.ndarray:
    """One molecule's gate vector with ``value`` on ``bond_ids`` and 1 elsewhere."""
    m = np.ones(int(n_bonds), dtype=np.float64)
    ids = np.asarray(list(bond_ids), dtype=np.int64).reshape(-1)
    if ids.size:
        if ids.min() < 0 or ids.max() >= int(n_bonds):
            raise IndexError(f"bond id outside 0..{int(n_bonds) - 1}")
        m[ids] = float(value)
    return m


def occ_counts(bonds, path_b1: np.ndarray, path_b2: np.ndarray, n_s1: np.ndarray | None = None) -> dict[str, int]:
    """How many S2 occurrences (and optionally S1 codes) a bond set touches."""
    ids = set(int(b) for b in np.asarray(list(bonds), dtype=np.int64).reshape(-1).tolist())
    if path_b1.size:
        hit = np.isin(path_b1, list(ids)) & np.isin(path_b2, list(ids))
        n_s2 = int(hit.sum())
        n_s2_either = int((np.isin(path_b1, list(ids)) | np.isin(path_b2, list(ids))).sum())
    else:
        n_s2 = 0
        n_s2_either = 0
    out = {"n_s2_occurrences": n_s2, "n_s2_occurrences_either_bond": n_s2_either}
    if n_s1 is not None:
        out["n_s1_codes"] = int(np.isin(n_s1, list(ids)).sum())
    return out


class SingleMolecule:
    """Everything the intervention stage needs for one cached molecule.

    Built once per molecule; holds the analysed graph, the verified occurrence
    map, the per-bond descriptors and the frozen split-cache slice.
    """

    def __init__(self, cache: Mapping[str, Any], row: int, spec, *, strict: bool = True) -> None:
        self.row = int(row)
        self.mol = D.slice_molecule(cache, self.row)
        self._graph, self.analysis = D.analysis_from_mol(self.mol)
        self.occ = D.build_occ_map(self.mol["atom_type"], self.analysis, int(spec.n_bond_types))
        self.verification = D.verify_occurrences(self.occ, self.mol, strict=strict)
        self.n_atoms = int(self.mol["n_atoms"])
        self.n_bonds = int(self.mol["n_bonds"])
        self.bond_u = np.asarray(self.mol["bond_index"], dtype=np.int64)[:, 0].copy()
        self.bond_v = np.asarray(self.mol["bond_index"], dtype=np.int64)[:, 1].copy()
        self.bond_type = np.asarray(self.mol["bond_type"], dtype=np.int64).copy()
        self.smiles = str(self.mol["smiles"])

    # -- structure ---------------------------------------------------------
    def bond_atoms(self, bond_ids) -> list[int]:
        ids = [int(b) for b in bond_ids]
        atoms = sorted({int(self.bond_u[b]) for b in ids} | {int(self.bond_v[b]) for b in ids})
        return atoms

    def adjacent_bonds(self, bond_ids) -> list[int]:
        ids = set(int(b) for b in bond_ids)
        atoms = set(self.bond_atoms(ids))
        out = []
        for e in range(self.n_bonds):
            if e in ids:
                continue
            if int(self.bond_u[e]) in atoms or int(self.bond_v[e]) in atoms:
                out.append(e)
        return out

    def shared_endpoint_pairs(self) -> list[tuple[int, int]]:
        pairs: list[tuple[int, int]] = []
        for atom in range(self.n_atoms):
            inc = np.where((self.bond_u == atom) | (self.bond_v == atom))[0].tolist()
            for i in range(len(inc)):
                for j in range(i + 1, len(inc)):
                    pairs.append((int(inc[i]), int(inc[j])))
        return pairs

    def atoms_connected(self, bond_ids) -> bool:
        ids = [int(b) for b in bond_ids]
        if not ids:
            return False
        parent: dict[int, int] = {}

        def find(x: int) -> int:
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for e in ids:
            ra, rb = find(int(self.bond_u[e])), find(int(self.bond_v[e]))
            if ra != rb:
                parent[ra] = rb
        roots = {find(int(self.bond_u[e])) for e in ids}
        return len(roots) == 1

    def s2_count(self, bond_ids) -> int:
        ids = [int(b) for b in bond_ids]
        if not self.occ.path_bond1.size:
            return 0
        hit = np.isin(self.occ.path_bond1, ids) & np.isin(self.occ.path_bond2, ids)
        return int(hit.sum())

    def s2_count_either(self, bond_ids) -> int:
        ids = [int(b) for b in bond_ids]
        if not self.occ.path_bond1.size:
            return 0
        return int((np.isin(self.occ.path_bond1, ids) | np.isin(self.occ.path_bond2, ids)).sum())

    def s1_count(self, bond_ids) -> int:
        ids = [int(b) for b in bond_ids]
        return int(np.isin(self.occ.nbr_bond, ids).sum())

    def bond_record(self) -> list[dict[str, Any]]:
        return [
            {
                "edge_id": int(e),
                "atom_u": int(self.bond_u[e]),
                "atom_v": int(self.bond_v[e]),
                "bond_type": int(self.bond_type[e]),
            }
            for e in range(self.n_bonds)
        ]


__all__ = [
    "CONDITION_NAME",
    "gated_parts",
    "reference_parts",
    "equivalence_check",
    "bond_mask",
    "occ_counts",
    "SingleMolecule",
]
