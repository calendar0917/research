"""Frozen LCRC-A model + the two task-specific output/loss adapters.

The model object is the upstream ``lcrc_cross_task_preparation_v1`` ``LCRC``
(``lipo_v1`` widths), imported unchanged.  The only additions are:

* ``DropoutAtLocal``: the frozen R3 recipe applies dropout ``p=0.10`` to the
  final atom-level local representation ``h`` (the output of ``model.local``),
  before the per-atom contribution ``f`` and the global ``rho``.  This is the
  same hook used by ``lcrc_lipo_mechanism_generalization_ssl_v2``.
* regression / classification adapters: identical forward graph, identical
  single scalar output.  Regression trains MSE on the standardised target and
  reports denormalised metrics; classification trains ``BCEWithLogitsLoss`` on
  the raw scalar and interprets it as a logit.  No extra layer, no second head.

Nothing else changes depth, width, S1/S2 modules or ``rho``.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn

import mb_common as C  # noqa: I001  (adds the frozen prep code dir to sys.path)

import lcrc_model as PREP_MODEL
from feature_spec import FeatureSpec

#: Unambiguous substrings: a module named like this would mean the frozen
#: architecture was changed.
FORBIDDEN_MODULE_SUBSTRINGS = (
    "attention",
    "message",
    "mpnn",
    "transformer",
    "relation",
    "junction",
    "conv",
)

#: Whole-name tokens (checked per dot-separated module part, so that the frozen
#: ``path_gate`` embedding is not confused with a graph-attention module).
FORBIDDEN_MODULE_TOKENS = ("gat", "gin", "gine", "gnn", "mpnn", "attn", "conv", "convs")


class DropoutAtLocal:
    """Apply dropout to the output of the one-shot local encoder."""

    def __init__(self, model: nn.Module, p: float):
        if not (0.0 <= p < 1.0):
            raise ValueError("dropout p must be in [0, 1)")
        self.module = nn.Dropout(float(p))
        model.add_module("_pilot_dropout_head", self.module)
        self.target = model.local
        self.handle = self.target.register_forward_hook(self._hook)
        self.calls = 0

    def _hook(self, module, inputs, output):
        self.calls += 1
        if not torch.is_tensor(output):
            raise TypeError("dropout hook expected a tensor output")
        return self.module(output)

    def remove(self) -> None:
        self.handle.remove()


def state_dict_fingerprint(state: Mapping[str, torch.Tensor]) -> str:
    import hashlib

    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode("utf-8"))
        h.update(np.ascontiguousarray(value.detach().cpu().numpy(), dtype=np.float32).tobytes())
    return h.hexdigest()


def build_lcrc(
    spec: FeatureSpec,
    norm: Mapping[str, Any],
    seed: int,
    device,
    *,
    dropout: float = float(C.RECIPE["dropout"]),
) -> tuple[PREP_MODEL.LCRC, dict[str, Any]]:
    """Frozen LCRC-A construction with the frozen dropout position."""
    torch.manual_seed(int(seed))
    model = PREP_MODEL.LCRC(spec, norm, PREP_MODEL.DEFAULT_CONFIG)
    hook = DropoutAtLocal(model, dropout)
    model.to(device)
    names = [n for n, _ in model.named_modules()]
    forbidden = [
        n
        for n in names
        if n
        and not n.startswith("_pilot_dropout_head")
        and (
            any(s in n.lower() for s in FORBIDDEN_MODULE_SUBSTRINGS)
            or any(tok in FORBIDDEN_MODULE_TOKENS for tok in n.lower().split("."))
        )
    ]
    if forbidden:
        raise RuntimeError(f"frozen LCRC-A unexpectedly contains modules matching forbidden names: {forbidden}")
    meta = {
        "schema": spec.name,
        "feature_spec_hash": spec.hash(),
        "seed": int(seed),
        "device": str(device),
        "config": dict(model.cfg),
        "parameters": PREP_MODEL.parameter_count(model),
        "parameters_breakdown": PREP_MODEL.module_breakdown(model),
        "init_state_hash": state_dict_fingerprint(model.state_dict()),
        "dropout": float(dropout),
        "dropout_position": C.RECIPE["dropout_position"],
        "dropout_hook": hook,
        "input_widths": {
            "atom_feat": spec.atom_feat_dim,
            "bond_feat": spec.bond_feat_dim,
            "global": spec.global_dim,
            "atom_scalars": spec.n_scalars,
            "n_nbr_codes": spec.n_nbr_codes,
            "n_path_codes": spec.n_path_codes,
            "local_mlp_input": 3 * int(model.cfg["atom_out"])
            + int(model.cfg["stat_rank"])
            + 3 * int(model.cfg["stat_rank"])
            + 8
            + int(model.cfg["atom_out"])
            + spec.n_scalars,
            "rho_input": spec.global_dim + 2 * int(model.cfg["width"]) + 1,
        },
        "has_relation_head": False,
        "has_attention": False,
        "iterative_hidden_message_passing": False,
        "readout": "sum_v f(h_v) + rho(global, mean_v h_v, mean_v h_v^2, log1p(n_atoms))",
    }
    return model, meta


def load_checkpoint_model(
    spec: FeatureSpec,
    norm: Mapping[str, Any],
    state: Mapping[str, torch.Tensor],
    device,
    *,
    dropout: float = float(C.RECIPE["dropout"]),
) -> PREP_MODEL.LCRC:
    model = PREP_MODEL.LCRC(spec, norm, PREP_MODEL.DEFAULT_CONFIG)
    DropoutAtLocal(model, dropout)
    model.load_state_dict({k: v.to(device) for k, v in state.items()}, strict=True)
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# structural audit helpers
# ---------------------------------------------------------------------------


def module_name_audit(model: nn.Module) -> dict[str, Any]:
    names = [n for n, _ in model.named_modules()]
    flagged = [
        n
        for n in names
        if n
        and (
            any(s in n.lower() for s in FORBIDDEN_MODULE_SUBSTRINGS)
            or any(tok in FORBIDDEN_MODULE_TOKENS for tok in n.lower().split("."))
        )
    ]
    return {
        "n_modules": len([n for n in names if n]),
        "module_names": [n for n in names if n],
        "forbidden_substrings": list(FORBIDDEN_MODULE_SUBSTRINGS),
        "forbidden_tokens": list(FORBIDDEN_MODULE_TOKENS),
        "flagged_modules": flagged,
        "passed": not flagged,
    }


def locality_audit(model, cache: Mapping[str, Any], spec: FeatureSpec, *, max_tries: int = 400) -> dict[str, Any]:
    """Empirical radius-2 locality check of the frozen construction.

    Pick an atom ``j`` and change only ``j``'s element one-hot / categorical atom
    type.  Because S1 is radius 1 and S2 is radius 2 in the frozen construction,
    ``h_i`` for every atom ``i`` with ``dist(i, j) > 2`` must be bit-identical,
    while ``h_j`` itself must change.  A message-passing model would propagate
    the change to every atom in the connected component.
    """
    import lcrc_data as PREP_DATA

    model.eval()
    device = next(model.parameters()).device
    n = int(cache["n_molecules"])
    atom_ptr = np.asarray(cache["atom_ptr"])
    bond_ptr = np.asarray(cache["bond_ptr"])
    for g in range(min(n, int(max_tries))):
        a0, a1 = int(atom_ptr[g]), int(atom_ptr[g + 1])
        b0, b1 = int(bond_ptr[g]), int(bond_ptr[g + 1])
        n_atoms = a1 - a0
        if n_atoms < 6:
            continue
        bonds = np.asarray(cache["bond_index"])[b0:b1]
        adj: list[list[int]] = [[] for _ in range(n_atoms)]
        for u, v in bonds.tolist():
            adj[int(u)].append(int(v))
            adj[int(v)].append(int(u))

        def bfs(source: int) -> np.ndarray:
            dist = np.full(n_atoms, -1, dtype=np.int64)
            dist[source] = 0
            queue = [source]
            while queue:
                cur = queue.pop(0)
                for nxt in adj[cur]:
                    if dist[nxt] < 0:
                        dist[nxt] = dist[cur] + 1
                        queue.append(nxt)
            return dist

        dist_from_root = bfs(0)
        far = np.flatnonzero(dist_from_root >= 3)
        if far.size == 0:
            continue
        j = int(far[0])
        dist_from_j = bfs(j)

        atom_type = np.asarray(cache["atom_type"])[a0:a1].copy()
        atom_feat = np.asarray(cache["atom_feat"])[a0:a1].copy()
        old = int(atom_type[j])
        new = 2 if old != 2 else 7  # C <-> N, both inside the frozen vocabulary
        atom_type[j] = new
        atom_feat[j, :16] = 0.0
        atom_feat[j, new] = 1.0

        single = _single_molecule_cache(cache, g)
        perturbed = {k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v) for k, v in single.items()}
        perturbed["atom_type"] = atom_type
        perturbed["atom_feat"] = atom_feat
        batch = _to_device(PREP_DATA.collate(single, [0]), device)
        batch_p = _to_device(PREP_DATA.collate(perturbed, [0]), device)
        with torch.no_grad():
            _y, aux = model(batch, return_aux=True)
            _y2, aux2 = model(batch_p, return_aux=True)
        h = aux["h"].detach().cpu().numpy()
        h2 = aux2["h"].detach().cpu().numpy()
        if h.shape != (n_atoms, h.shape[1]) or h2.shape != h.shape:
            raise RuntimeError("unexpected h shape in locality audit")
        delta = np.abs(h - h2).max(axis=1)
        far_mask = dist_from_j > 2
        near_mask = ~far_mask
        return {
            "molecule_index_in_cache": int(g),
            "n_atoms": int(n_atoms),
            "root_atom": 0,
            "perturbed_atom": j,
            "distance_root_to_perturbed": int(dist_from_root[j]),
            "atom_type_change": [old, new],
            "max_abs_h_delta_atom_permuted": float(delta[j]),
            "max_abs_h_delta_within_radius_2": float(delta[near_mask].max()),
            "max_abs_h_delta_beyond_radius_2": float(delta[far_mask].max()) if far_mask.any() else 0.0,
            "n_atoms_beyond_radius_2": int(far_mask.sum()),
            "perturbed_atom_responded": bool(delta[j] > 0.0),
            "beyond_radius_2_unchanged": bool(delta[far_mask].max() == 0.0) if far_mask.any() else False,
            "passed": bool(delta[j] > 0.0 and (delta[far_mask].max() == 0.0 if far_mask.any() else False)),
        }
    return {"passed": False, "error": "no molecule with a distance>=3 atom pair found"}


def _to_device(batch: Mapping[str, Any], device) -> dict[str, Any]:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _single_molecule_cache(cache: Mapping[str, Any], g: int) -> dict[str, Any]:
    """Extract molecule ``g`` as a standalone one-molecule cache."""
    atom_ptr = np.asarray(cache["atom_ptr"])
    bond_ptr = np.asarray(cache["bond_ptr"])
    nbr_ptr = np.asarray(cache["nbr_ptr"])
    path_ptr = np.asarray(cache["path_ptr"])
    a0, a1 = int(atom_ptr[g]), int(atom_ptr[g + 1])
    b0, b1 = int(bond_ptr[g]), int(bond_ptr[g + 1])
    nbr_s, nbr_e = int(nbr_ptr[a0]), int(nbr_ptr[a1])
    path_s, path_e = int(path_ptr[a0]), int(path_ptr[a1])
    path_end = np.asarray(cache["path_end"])[path_s:path_e]
    return {
        "atom_ptr": np.asarray([0, a1 - a0], dtype=np.int64),
        "bond_ptr": np.asarray([0, b1 - b0], dtype=np.int64),
        "nbr_ptr": np.asarray(nbr_ptr[a0 : a1 + 1] - nbr_s, dtype=np.int64),
        "path_ptr": np.asarray(path_ptr[a0 : a1 + 1] - path_s, dtype=np.int64),
        "atom_feat": np.array(np.asarray(cache["atom_feat"])[a0:a1], copy=True),
        "atom_type": np.array(np.asarray(cache["atom_type"])[a0:a1], copy=True),
        "atom_scalar": np.array(np.asarray(cache["atom_scalar"])[a0:a1], copy=True),
        "bond_feat": np.array(np.asarray(cache["bond_feat"])[b0:b1], copy=True),
        "bond_type": np.array(np.asarray(cache["bond_type"])[b0:b1], copy=True),
        "bond_index": np.array(np.asarray(cache["bond_index"])[b0:b1], copy=True),
        "nbr_code": np.array(np.asarray(cache["nbr_code"])[nbr_s:nbr_e], copy=True),
        "path_cat": np.array(np.asarray(cache["path_cat"])[path_s:path_e], copy=True),
        "path_end63": np.array(np.asarray(cache["path_end63"])[path_s:path_e], copy=True),
        "path_end": np.array(path_end, copy=True),
        "path_closure": np.array(np.asarray(cache["path_closure"])[path_s:path_e], copy=True),
        "global": np.array(np.asarray(cache["global"])[g : g + 1], copy=True),
        "n_components": np.asarray([int(np.asarray(cache["n_components"])[g])], dtype=np.int64),
        "n_junctions": np.asarray([int(np.asarray(cache["n_junctions"])[g])], dtype=np.int64),
        "n_ring_systems": np.asarray([int(np.asarray(cache["n_ring_systems"])[g])], dtype=np.int64),
        "molecule_ids": [str(cache["molecule_ids"][g])],
        "smiles": [str(cache["smiles"][g])],
        "n_molecules": 1,
        "split": "single",
        "rows": np.asarray([int(np.asarray(cache["rows"])[g])], dtype=np.int64),
        "meta": {"labels_present": False, "split": "single"},
    }


__all__ = [
    "DropoutAtLocal",
    "build_lcrc",
    "load_checkpoint_model",
    "state_dict_fingerprint",
    "module_name_audit",
    "locality_audit",
]
