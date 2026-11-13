"""Frozen LCRC-C1 plus the single added correction operator.

For one atom ``v`` with local representation ``h_v`` (the output of the frozen
local LayerNorm) and one fixed 16-d structural vector ``s_v``::

    psi(h, s) = W2 * SiLU(W_h h + W_s s + b1) + b2

``h`` is 64-d, ``s`` is 16-d, the hidden width is 32, therefore::

    W_h 64x32 = 2048, W_s 16x32 = 512, b1 32, W2 32x64 = 2048, b2 64
    total 4704  ->  LCRC-C1 (228,674) + 4,704 = 233,378

The base model is **not** re-implemented here.  The correction is inserted with
a forward hook on the frozen ``LocalContent`` module, exactly between the local
LayerNorm and the single frozen dropout call.  Both ``f(.)`` and ``rho(.)``
therefore read the same corrected-and-dropped ``h'`` and every other part of the
frozen forward executes unchanged.

Arms:

* ``LOCAL3``: ``h'_v = h_v + psi(h_v, s3_v)``
* ``DUP2``:   ``h'_v = h_v + psi(h_v, s2_v)``
* ``NULL3``:  ``h'_v = h_v + mean_{u in C(v)} psi(h_v, s3_u)``

``C(v)`` is the set of atoms in the same molecule with the same ``atom_type`` as
``v``, including ``v``.  The mean field uses only fixed structural vectors
``s3_u`` -- never another atom's learned ``h_u`` -- so it is evaluated exactly as
``W2 * mean_u SiLU(W_h h_v + W_s s3_u + b1) + b2``.  It is not the mean-vector
shortcut ``psi(h_v, mean_u s3_u)``.

With ``W2 = 0`` and ``b2 = 0`` the operator returns exactly zero, so the model is
bit-identical to the frozen C1 model with the same base weights.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import common as C

PAIR_CHUNK = 1 << 18  # bound on (root, group-member) pairs evaluated at once
MODES = ("local3", "dup2", "null3", "off")


# ---------------------------------------------------------------------------
# the 4704-parameter corrector
# ---------------------------------------------------------------------------


class S3Corrector(nn.Module):
    def __init__(self, h_dim: int = 64, s_dim: int = C.S3_DIM, hidden: int = C.CORRECTOR_HIDDEN):
        super().__init__()
        self.h_dim = int(h_dim)
        self.s_dim = int(s_dim)
        self.hidden = int(hidden)
        self.W_h = nn.Linear(self.h_dim, self.hidden, bias=False)
        self.W_s = nn.Linear(self.s_dim, self.hidden, bias=False)
        self.b1 = nn.Parameter(torch.zeros(self.hidden))
        self.W2 = nn.Linear(self.hidden, self.h_dim, bias=False)
        self.b2 = nn.Parameter(torch.zeros(self.h_dim))

    def activation(self, h: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """``SiLU(W_h h + W_s s + b1)``."""
        return F.silu(self.W_h(h) + self.W_s(s) + self.b1)

    def from_activation(self, act: torch.Tensor) -> torch.Tensor:
        return self.W2(act) + self.b2

    def forward(self, h: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        return self.from_activation(self.activation(h, s))


def parameter_count(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def build_corrector_state(seed: int = C.CORRECTOR_INIT_SEED) -> dict[str, torch.Tensor]:
    """One fixed corrector initialisation, shared tensor-by-tensor by all arms.

    ``W_h`` / ``W_s`` use the deterministic PyTorch default initialisation under
    a dedicated seed, ``b1`` starts at zero, and both output tensors are
    *strictly* zero so the initial operator is exactly the identity.
    """
    torch.manual_seed(int(seed))
    module = S3Corrector()
    state = {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
    state["W2.weight"].zero_()
    state["b2"].zero_()
    return state


def corrector_fingerprint(state: Mapping[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for key in sorted(state):
        h.update(key.encode("utf-8"))
        h.update(np.ascontiguousarray(np.asarray(state[key].detach().cpu()), dtype=np.float32).tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# exact NULL3 mean field
# ---------------------------------------------------------------------------


def mean_field_activation(
    A: torch.Tensor,
    B: torch.Tensor,
    group: torch.Tensor,
    *,
    pair_chunk: int = PAIR_CHUNK,
) -> torch.Tensor:
    """``mean_{u in C(v)} SiLU(A_v + B_u)`` for every atom ``v``.

    ``A = W_h h + b1`` and ``B = W_s s``.  ``group`` assigns each atom to one
    group (same molecule, same atom type); the molecule index is part of the key
    so groups never cross molecules.  Implemented exactly per group size: no
    sampling, no atom-id-dependent permutation, no cross-molecule mixing.
    """
    n = int(A.shape[0])
    if n == 0:
        return A.new_zeros((0, A.shape[1]))
    order = torch.argsort(group, stable=True)
    sorted_group = group[order]
    unique, counts = torch.unique_consecutive(sorted_group, return_counts=True)
    starts = torch.zeros(unique.numel(), dtype=torch.long, device=group.device)
    if unique.numel() > 1:
        starts[1:] = torch.cumsum(counts, dim=0)[:-1]
    size_per_position = counts.repeat_interleave(counts)
    start_per_position = starts.repeat_interleave(counts)

    out = A.new_zeros((n, A.shape[1]))
    for k in torch.unique(counts).tolist():
        k = int(k)
        sel = torch.nonzero(size_per_position == k, as_tuple=False).view(-1)
        n_groups = int(sel.numel() // k)
        group_starts = start_per_position[sel[::k]]
        offsets = torch.arange(k, dtype=torch.long, device=group.device)
        member_pos = group_starts.view(-1, 1) + offsets.view(1, -1)  # (G, k)
        members = order[member_pos]  # (G, k) atom indices
        step = max(1, int(pair_chunk) // max(k * k, 1))
        for lo in range(0, n_groups, step):
            hi = min(lo + step, n_groups)
            m = members[lo:hi]
            g = int(m.shape[0])
            v_idx = m.view(g, k, 1).expand(g, k, k).reshape(-1)
            u_idx = m.view(g, 1, k).expand(g, k, k).reshape(-1)
            z = F.silu(A[v_idx] + B[u_idx])  # (g*k*k, hidden)
            acc = out.new_zeros((n, A.shape[1]))
            acc.index_add_(0, v_idx, z)
            out = out + acc / float(k)
    return out


def null_field(corrector: S3Corrector, h: torch.Tensor, s: torch.Tensor, group: torch.Tensor) -> torch.Tensor:
    A = corrector.W_h(h) + corrector.b1
    B = corrector.W_s(s)
    return corrector.from_activation(mean_field_activation(A, B, group))


def null_field_naive(corrector: S3Corrector, h: torch.Tensor, s: torch.Tensor, group: torch.Tensor) -> torch.Tensor:
    """Reference double loop (test oracle only; never used in a formal run)."""
    n = int(h.shape[0])
    out = []
    for v in range(n):
        idx = torch.nonzero(group == group[v], as_tuple=False).view(-1).tolist()
        terms = [corrector(h[v : v + 1], s[u : u + 1])[0] for u in idx]
        out.append(torch.stack(terms, 0).mean(0))
    return torch.stack(out, 0)


# ---------------------------------------------------------------------------
# hook installation / model construction
# ---------------------------------------------------------------------------


def _group_key(batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return batch["atom_batch"] * C.N_ATOM_TYPES + batch["atom_type"]


def install(model: nn.Module, mode: str, *, capture: bool = True):
    """Insert the correction between the local LayerNorm and the frozen dropout."""
    if mode not in MODES:
        raise ValueError(mode)
    model.__dict__["s3_mode"] = str(mode)
    if getattr(model, "_s3_handle", None) is not None:
        model._s3_handle.remove()

    def _hook(module, inputs, output):
        batch = inputs[0]
        h, aux = output
        mode_now = model.__dict__.get("s3_mode", mode)
        corr = model.s3
        if mode_now == "off":
            delta = torch.zeros_like(h)
        else:
            s = batch["s3"] if mode_now in ("local3", "null3") else batch["s2"]
            if mode_now in ("local3", "dup2"):
                delta = corr(h, s)
            else:
                delta = null_field(corr, h, s, _group_key(batch))
        h_new = h + delta
        if capture:
            with torch.no_grad():
                model.s3_last_stats = {
                    "correction_norm_mean": float(delta.norm(dim=1).mean()) if delta.numel() else 0.0,
                    "correction_norm_max": float(delta.norm(dim=1).max()) if delta.numel() else 0.0,
                    "correction_abs_mean": float(delta.abs().mean()) if delta.numel() else 0.0,
                    "h_norm_mean": float(h.norm(dim=1).mean()) if h.numel() else 0.0,
                }
        if aux is not None:
            aux = dict(aux)
            aux["h_pre_correction"] = h
            aux["correction"] = delta
        return (h_new, aux)

    model._s3_handle = model.local.register_forward_hook(_hook)
    return model._s3_handle


def set_mode(model: nn.Module, mode: str) -> None:
    if mode not in MODES:
        raise ValueError(mode)
    model.__dict__["s3_mode"] = str(mode)


def base_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v for k, v in model.state_dict().items() if not k.startswith("s3.")}


def build_model(
    spec,
    norm: Mapping[str, Any],
    device,
    arm: str,
    *,
    base_state: Mapping[str, torch.Tensor] | None = None,
    corrector_state: Mapping[str, torch.Tensor] | None = None,
    base_is_variant_state: bool = False,
) -> tuple[nn.Module, dict[str, Any]]:
    """Build one arm from the frozen FULL seed-0 state (never from a trained A).

    ``base_state`` is the *FULL* seed-0 state and is remapped to C1 here. A
    checkpoint already stores the remapped C1 base, so reloading it passes
    ``base_is_variant_state=True`` and skips the second remap.
    """
    if base_state is None:
        full_state = C.DM.build_full_state(spec, norm, C.TRAIN_SEED)
        init_state = C.DM.remap_state(C.VARIANT, full_state)
    elif base_is_variant_state:
        full_state = None
        init_state = dict(base_state)
    else:
        full_state = dict(base_state)
        init_state = C.DM.remap_state(C.VARIANT, full_state)
    model = C.DM.DissectLCRC(spec, norm, C.VARIANT)
    model.load_state_dict(init_state, strict=True)

    corr_state = build_corrector_state() if corrector_state is None else dict(corrector_state)
    corrector = S3Corrector()
    corrector.load_state_dict({k: v.clone() for k, v in corr_state.items()}, strict=True)
    base_params = C.DM.parameter_count(model)
    model.s3 = corrector

    install(model, arm.lower())
    model.to(device)
    info = {
        "variant": C.VARIANT,
        "variant_name": C.VARIANT_NAME,
        "arm": arm,
        "base_parameters": int(base_params),
        "corrector_parameters": parameter_count(corrector),
        "parameters": parameter_count(model),
        "init_state_hash": C.DM.state_fingerprint(base_state_dict(model)),
        "full_init_state_hash": (C.DM.state_fingerprint(full_state) if full_state is not None else None),
        "corrector_state_hash": corrector_fingerprint(corrector.state_dict()),
        "base_shared_state_equal": (
            C.DM.shared_state_equal(C.VARIANT, full_state, init_state) if full_state is not None else None
        ),
        "reloaded_from_checkpoint": bool(base_is_variant_state),
    }
    if int(info["parameters"]) != C.EXPECTED_PARAMS:
        raise RuntimeError(f"{arm}: {info['parameters']} parameters != frozen {C.EXPECTED_PARAMS}")
    if int(info["base_parameters"]) != C.BASE_PARAMS:
        raise RuntimeError(f"{arm}: base parameters {info['base_parameters']} != {C.BASE_PARAMS}")
    if int(info["corrector_parameters"]) != C.CORRECTOR_PARAMS:
        raise RuntimeError(f"{arm}: corrector parameters {info['corrector_parameters']} != {C.CORRECTOR_PARAMS}")
    return model, info


def load_trained_model(
    spec,
    norm: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    device,
    *,
    mode: str | None = None,
) -> nn.Module:
    """Rebuild one arm from a saved checkpoint (base + corrector tensors)."""
    state = checkpoint["state_dict"]
    base_state = {k: v for k, v in state.items() if not k.startswith("s3.")}
    corr_state = {k[len("s3.") :]: v for k, v in state.items() if k.startswith("s3.")}
    arm = str(checkpoint["arm"])
    model, info = build_model(
        spec,
        norm,
        device,
        mode if mode is not None else arm.lower(),
        base_state=base_state,
        corrector_state=corr_state,
        base_is_variant_state=True,
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing} unexpected={unexpected}")
    model.eval()
    return model


__all__ = [
    "PAIR_CHUNK",
    "MODES",
    "S3Corrector",
    "parameter_count",
    "build_corrector_state",
    "corrector_fingerprint",
    "mean_field_activation",
    "null_field",
    "null_field_naive",
    "install",
    "set_mode",
    "base_state_dict",
    "build_model",
    "load_trained_model",
]
