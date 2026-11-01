"""Frozen LCRC-A model + the task-specific output adapters of Sprint 02.

The model object is the upstream ``lcrc_cross_task_preparation_v1`` ``LCRC``
(``lipo_v1`` widths), imported unchanged.  v2 adds exactly two things:

1. ``DropoutAtLocal`` -- the frozen R3 dropout position (final atom-level local
   representation ``h``), identical to Sprint 01.
2. ``LCRCMulti`` -- the **only** structural change authorised this round: the two
   final output layers ``f`` (per-atom contribution) and ``rho`` (global
   correction) widen from 1 to 12 outputs so that Tox21 can be trained with a
   shared encoder.  Everything upstream of them (atom encoder, bond encoder,
   ``LocalContent``/S1/S2, the single local encode) is byte-identical in
   structure and initialisation to the single-task model, which
   ``shared_encoder_state_hash`` proves at audit time.

Nothing else changes depth, width, S1/S2 modules, readout or ``rho`` inputs.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn

import mb2_common as C  # noqa: I001  (adds the frozen prep code dirs to sys.path)

import lcrc_model as PREP_MODEL
from feature_spec import FeatureSpec

#: A module whose name matches one of these would mean the frozen architecture
#: was changed.
FORBIDDEN_MODULE_SUBSTRINGS = (
    "attention",
    "message",
    "mpnn",
    "transformer",
    "relation",
    "junction",
    "conv",
)
#: Whole-name tokens (checked per dot-separated module part, so the frozen
#: ``path_gate`` embedding is not confused with a graph-attention module).
FORBIDDEN_MODULE_TOKENS = ("gat", "gin", "gine", "gnn", "mpnn", "attn", "conv", "convs")

#: Parameter count of the frozen single-task LCRC-A (Sprint 01, re-asserted here).
SINGLE_TASK_PARAMETERS = 247_106


class DropoutAtLocal:
    """Apply dropout to the output of the one-shot local encoder."""

    def __init__(self, model: nn.Module, p: float):
        if not (0.0 <= p < 1.0):
            raise ValueError("dropout p must be in [0, 1)")
        self.p = float(p)
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


class LCRCMulti(PREP_MODEL.LCRC):
    """LCRC-A with widened ``f`` / ``rho`` heads (Tox21, 12 endpoints).

    Only the two output layers differ; the encoder, the local content module and
    the buffers are untouched, and the state-dict keys stay identical
    (``f.lins.{0,1}.weight``, ``rho.lins.{0,1}.weight``).
    """

    def __init__(self, spec: FeatureSpec, norm: Mapping[str, Any], cfg: Mapping[str, Any], n_tasks: int):
        super().__init__(spec, norm, cfg)
        if int(n_tasks) < 2:
            raise ValueError("LCRCMulti needs n_tasks >= 2")
        self.n_tasks = int(n_tasks)
        width = int(self.cfg["width"])
        rho_in = int(spec.global_dim) + 2 * width + 1
        self.f = PREP_MODEL.MLP([width, width, self.n_tasks])
        self.rho = PREP_MODEL.MLP([rho_in, int(self.cfg["rho_hidden"]), self.n_tasks])

    def forward(self, batch: Mapping[str, torch.Tensor], *, return_aux: bool = False):
        n_graphs = int(batch["n_graphs"])
        _a, _b, h = self._encode(batch)
        atom_batch = batch["atom_batch"]
        width = h.shape[1]
        count = h.new_zeros(n_graphs).index_add_(
            0, atom_batch, torch.ones_like(atom_batch, dtype=h.dtype)
        )
        denom = count.clamp(min=1.0).unsqueeze(1)
        mean_h = h.new_zeros((n_graphs, width)).index_add_(0, atom_batch, h) / denom
        mean_h2 = h.new_zeros((n_graphs, width)).index_add_(0, atom_batch, h * h) / denom
        contrib_atom = self.f(h)  # [n_atoms, n_tasks]
        contrib = h.new_zeros((n_graphs, self.n_tasks)).index_add_(0, atom_batch, contrib_atom)
        g = (batch["global"] - self.global_mean) / self.global_scale
        rho_in = torch.cat([g, mean_h, mean_h2, torch.log1p(count).unsqueeze(1)], dim=1)
        rho = self.rho(rho_in)  # [n_graphs, n_tasks]
        y = contrib + rho
        if not return_aux:
            return y
        aux = {
            "contrib": contrib,
            "rho": rho,
            "contrib_atom": contrib_atom,
            "mean_h": mean_h,
            "mean_h2": mean_h2,
            "h": h,
            "count": count,
            "a": _a,
            "b": _b,
        }
        return y, aux


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------


def state_dict_fingerprint(state: Mapping[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode("utf-8"))
        h.update(np.ascontiguousarray(value.detach().cpu().numpy(), dtype=np.float32).tobytes())
    return h.hexdigest()


def _forbidden_modules(model: nn.Module) -> list[str]:
    names = [n for n, _ in model.named_modules()]
    return [
        n
        for n in names
        if n
        and not n.startswith("_pilot_dropout_head")
        and (
            any(s in n.lower() for s in FORBIDDEN_MODULE_SUBSTRINGS)
            or any(tok in n.lower().split(".") for tok in FORBIDDEN_MODULE_TOKENS)
        )
    ]


def build_lcrc(
    spec: FeatureSpec,
    norm: Mapping[str, Any],
    seed: int,
    device,
    *,
    dropout: float = float(C.RECIPE["dropout"]),
    n_tasks: int = 1,
) -> tuple[nn.Module, dict[str, Any]]:
    """Frozen LCRC-A construction (single task, or the 12-output Tox21 variant)."""
    torch.manual_seed(int(seed))
    if int(n_tasks) == 1:
        model: nn.Module = PREP_MODEL.LCRC(spec, norm, PREP_MODEL.DEFAULT_CONFIG)
    else:
        model = LCRCMulti(spec, norm, PREP_MODEL.DEFAULT_CONFIG, int(n_tasks))
    hook = DropoutAtLocal(model, dropout)
    model.to(device)
    forbidden = _forbidden_modules(model)
    if forbidden:
        raise RuntimeError(f"frozen LCRC-A unexpectedly contains modules matching forbidden names: {forbidden}")
    meta = {
        "name": "LCRC-A" if int(n_tasks) == 1 else "LCRC-A (12-output head: Tox21)",
        "schema": spec.name,
        "feature_spec_hash": spec.hash(),
        "seed": int(seed),
        "device": str(device),
        "n_outputs": int(n_tasks),
        "config": dict(model.cfg),
        "parameters": PREP_MODEL.parameter_count(model),
        "parameters_breakdown": PREP_MODEL.module_breakdown(model),
        "init_state_hash": state_dict_fingerprint(model.state_dict()),
        "shared_encoder_state_hash": shared_encoder_state_hash(model),
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
        "readout": (
            "sum_v f(h_v) + rho(global, mean_v h_v, mean_v h_v^2, log1p(n_atoms))"
            if int(n_tasks) == 1
            else "[sum_v f(h_v) + rho(...)] with 12 outputs per molecule (shared encoder)"
        ),
    }
    return model, meta


def load_checkpoint_model(
    spec: FeatureSpec,
    norm: Mapping[str, Any],
    state: Mapping[str, torch.Tensor],
    device,
    *,
    dropout: float = float(C.RECIPE["dropout"]),
    n_tasks: int = 1,
) -> nn.Module:
    torch.manual_seed(0)
    if int(n_tasks) == 1:
        model: nn.Module = PREP_MODEL.LCRC(spec, norm, PREP_MODEL.DEFAULT_CONFIG)
    else:
        model = LCRCMulti(spec, norm, PREP_MODEL.DEFAULT_CONFIG, int(n_tasks))
    DropoutAtLocal(model, dropout)
    model.load_state_dict({k: v.to(device) for k, v in state.items()}, strict=True)
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# structural audits
# ---------------------------------------------------------------------------


def encoder_parameters(model: nn.Module) -> dict[str, torch.Tensor]:
    """Every parameter that is *not* one of the two output heads."""
    return {
        k: v.detach().cpu()
        for k, v in model.state_dict().items()
        if not (k.startswith("f.") or k.startswith("rho."))
    }


def shared_encoder_state_hash(model: nn.Module) -> str:
    return state_dict_fingerprint(encoder_parameters(model))


def module_name_audit(model: nn.Module) -> dict[str, Any]:
    names = [n for n, _ in model.named_modules()]
    flagged = _forbidden_modules(model)
    return {
        "n_modules": len([n for n in names if n]),
        "module_names": [n for n in names if n],
        "forbidden_substrings": list(FORBIDDEN_MODULE_SUBSTRINGS),
        "forbidden_tokens": list(FORBIDDEN_MODULE_TOKENS),
        "flagged_modules": flagged,
        "passed": not flagged,
    }


def _single_molecule_cache(cache: Mapping[str, Any], g: int) -> dict[str, Any]:
    atom_ptr = np.asarray(cache["atom_ptr"])
    bond_ptr = np.asarray(cache["bond_ptr"])
    nbr_ptr = np.asarray(cache["nbr_ptr"])
    path_ptr = np.asarray(cache["path_ptr"])
    a0, a1 = int(atom_ptr[g]), int(atom_ptr[g + 1])
    b0, b1 = int(bond_ptr[g]), int(bond_ptr[g + 1])
    nbr_s, nbr_e = int(nbr_ptr[a0]), int(nbr_ptr[a1])
    path_s, path_e = int(path_ptr[a0]), int(path_ptr[a1])
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
        "path_end": np.array(np.asarray(cache["path_end"])[path_s:path_e], copy=True),
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


def locality_audit(model, cache: Mapping[str, Any], spec: FeatureSpec, *, max_tries: int = 400) -> dict[str, Any]:
    """Empirical radius-2 locality check of the frozen construction.

    Change only atom ``j``'s element one-hot / categorical atom type.  Because S1
    is radius 1 and S2 is radius 2, ``h_i`` for every atom with ``dist(i, j) > 2``
    must stay bit-identical while ``h_j`` itself changes.  A message-passing model
    would propagate the change across the connected component.
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


__all__ = [
    "DropoutAtLocal",
    "LCRCMulti",
    "SINGLE_TASK_PARAMETERS",
    "build_lcrc",
    "load_checkpoint_model",
    "state_dict_fingerprint",
    "shared_encoder_state_hash",
    "encoder_parameters",
    "module_name_audit",
    "locality_audit",
]
