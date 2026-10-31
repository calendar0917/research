"""Model zoo for the best-epoch / distance-fix round.

``BASE``, ``ROLEBLIND``, ``G2L-LOO`` and ``G2L-ALL`` are built by the frozen
historical builders (``dc_models.build_family``) and are therefore bitwise the
same models as in the previous rounds, including the shared BASE
initialisation for a given seed.

``DIST-TRUE-FIXED`` re-implements the B-family with the corrected aggregation:

    c34_i = (1/|D34_i|) sum_{j in D34_i} h_j          (root atom i)
    c5_i  = (1/|D5_i|)  sum_{j in D5_i}  h_j          (root atom i)
    u_i   = tanh(W_atom h_i)
    p34_i = has34_i * tanh(W_34 c34_i)
    p5_i  = has5_i  * tanh(W_5  c5_i)
    h_new_i = h_i + both_nonempty_i * W_out(u_i * (p34_i + p5_i))

``DIST-BLIND`` shares the module, the parameter names and the initialisation
with ``DIST-TRUE-FIXED`` but replaces each bucket's *content* by the
leave-one-out all-atom mean of the same molecule, keeping only bucket
existence.  It is the parameter- and routing-matched negative control.

The historical ``DistanceCorrection`` (``dc_models``) is neither imported nor
overwritten; the historical ``DIST-TRUE`` checkpoints and report stay as they
are.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

import be_common as C
import be_data as D
import dc_models as DCM  # read-only frozen BASE / ROLEBLIND / G2L builders
import lcrc_model  # noqa: E402
import x1_models as X1M  # noqa: E402

HISTORICAL_FAMILIES = ("base", "roleblind", "g2l_loo", "g2l_all")
DISTANCE_FAMILIES = ("dist_true_fixed", "dist_blind")


# ---------------------------------------------------------------------------
# corrected distance correction
# ---------------------------------------------------------------------------


def _leave_one_out_mean(h: torch.Tensor, atom_batch: torch.Tensor, n_graphs: int) -> torch.Tensor:
    width = h.shape[1]
    count = h.new_zeros(n_graphs).index_add_(
        0, atom_batch, torch.ones_like(atom_batch, dtype=h.dtype)
    )
    total = h.new_zeros((n_graphs, width)).index_add_(0, atom_batch, h)
    numerator = total.index_select(0, atom_batch) - h
    denom = (count.index_select(0, atom_batch) - 1.0).clamp(min=1.0)
    out = numerator / denom.unsqueeze(1)
    singleton = (count.index_select(0, atom_batch) <= 1).unsqueeze(1)
    return torch.where(singleton, torch.zeros_like(out), out)


class DistanceCorrectionFixed(nn.Module):
    """Per-root D34/D5 context with the aggregation bug fixed."""

    def __init__(self, width: int = C.WIDTH, dim: int = C.CORR_DIM):
        super().__init__()
        self.W_atom = nn.Linear(width, dim)
        self.W_34 = nn.Linear(width, dim)
        self.W_5 = nn.Linear(width, dim)
        self.W_out = nn.Linear(dim, width)
        with torch.no_grad():
            self.W_out.weight.normal_(mean=0.0, std=C.W_OUT_INIT_STD)
            self.W_out.bias.zero_()

    def forward(self, h, batch, n_graphs: int, mode: str):
        n_atoms = int(h.shape[0])
        d34_idx = batch["d34_idx"]
        d5_idx = batch["d5_idx"]
        c34_counts = batch["d34_counts"]
        c5_counts = batch["d5_counts"]
        has34 = c34_counts > 0
        has5 = c5_counts > 0

        m_except = _leave_one_out_mean(h, batch["atom_batch"], int(n_graphs))
        c34, c5 = D.bucket_means(h, d34_idx, d5_idx, c34_counts, c5_counts)
        if mode == "blind":
            c34 = torch.where(has34.unsqueeze(1), m_except, torch.zeros_like(c34))
            c5 = torch.where(has5.unsqueeze(1), m_except, torch.zeros_like(c5))
        elif mode != "true":
            raise ValueError(mode)

        u = torch.tanh(self.W_atom(h))
        p34 = torch.tanh(self.W_34(c34)) * has34.to(h.dtype).unsqueeze(1)
        p5 = torch.tanh(self.W_5(c5)) * has5.to(h.dtype).unsqueeze(1)
        change = self.W_out(u * (p34 + p5))
        both_empty = (~has34) & (~has5)
        change = torch.where(both_empty.unsqueeze(1), torch.zeros_like(change), change)
        ctx = {
            "c34": c34,
            "c5": c5,
            "m_except": m_except,
            "p34": p34,
            "p5": p5,
            "u": u,
            "has34": has34,
            "has5": has5,
            "both_empty": both_empty,
            "d34_counts": c34_counts,
            "d5_counts": c5_counts,
            "d34_root": D.bucket_roots(c34_counts, c5_counts, n_atoms)[0],
            "d5_root": D.bucket_roots(c34_counts, c5_counts, n_atoms)[1],
            "n_atoms": n_atoms,
        }
        return change, ctx


class LCRCDistanceFixed(lcrc_model.LCRC):
    """LCRC with the corrected D34/D5 correction on the ``f`` input."""

    def __init__(self, spec, norm, cfg=None, mode: str = "true"):
        super().__init__(spec, norm, cfg)
        self.dist = DistanceCorrectionFixed(C.WIDTH, C.CORR_DIM)
        self.mode = str(mode)

    def forward(self, batch: Mapping[str, torch.Tensor], *, return_aux: bool = False):
        a, b, h = self._encode(batch)
        change, ctx = self.dist(h, batch, int(batch["n_graphs"]), self.mode)
        h_new = h + change
        y, aux = DCM.standard_readout(self, batch, h, h_new)
        if not return_aux:
            return y
        aux.update({"a_encoded": a, "b_encoded": b, "change": change, **ctx})
        return y, aux


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------


def _load_shared_from_base(model: nn.Module, base: nn.Module) -> dict[str, Any]:
    own = model.state_dict()
    shared = {
        key: value.detach().clone()
        for key, value in base.state_dict().items()
        if key in own and tuple(own[key].shape) == tuple(value.shape)
    }
    model.load_state_dict({**own, **shared}, strict=True)
    audit = X1M.shared_init_audit(base, model)
    if not audit["all_shared_bitwise_identical"]:
        raise RuntimeError(f"shared initialisation mismatch: {audit['mismatched']}")
    return audit


def _attach(model, dropout: float):
    hook = X1M.attach_dropout(model, float(dropout))
    model.eval()
    return hook


def _build_distance(family: str, spec, norm, seed: int, device, dropout: float):
    base, base_meta = X1M.build_base(spec, norm, seed, "cpu", dropout)
    if family == "dist_true_fixed":
        torch.manual_seed(int(seed))
        model = LCRCDistanceFixed(spec, norm, lcrc_model.DEFAULT_CONFIG, mode="true")
        shared = _load_shared_from_base(model, base)
        twin_init = None
    elif family == "dist_blind":
        true_model, true_meta = _build_distance("dist_true_fixed", spec, norm, seed, "cpu", dropout)
        torch.manual_seed(int(seed))
        model = LCRCDistanceFixed(spec, norm, lcrc_model.DEFAULT_CONFIG, mode="blind")
        model.load_state_dict(true_model.state_dict(), strict=True)
        shared = true_meta["shared_init"]
        twin_init = true_meta["init_state_hash"]
    else:
        raise ValueError(family)
    hook = _attach(model, dropout)
    meta = {
        "family": family,
        "arm": family,
        "class": f"LCRCDistanceFixed(mode={model.mode})",
        "seed": int(seed),
        "dropout": float(dropout),
        "dropout_hook": hook,
        "parameters": parameter_count(model),
        "parameters_base": int(base_meta["parameters"]),
        "parameters_new": parameter_count(model) - int(base_meta["parameters"]),
        "parameter_breakdown": parameter_breakdown(model),
        "init_state_hash": C.state_dict_fingerprint(model.state_dict()),
        "base_init_state_hash": base_meta["init_state_hash"],
        "twin_init_state_hash": twin_init,
        "shared_init": shared,
        "definition": (
            "h_new_i = h_i + W_out(tanh(W_atom h_i) * (p34_i + p5_i)); "
            "p34_i = has34_i*tanh(W_34 c34_i), c34_i = mean_{j in D34_i} h_j (root i); "
            "p5_i = has5_i*tanh(W_5 c5_i), c5_i = mean_{j in D5_i} h_j (root i); "
            "empty bucket -> zero context and zero change"
        ),
    }
    model.to(device)
    return model, meta


def build_family(
    family: str,
    spec,
    norm,
    seed: int,
    device="cpu",
    dropout: float = C.TRAIN_RECIPE["dropout"],
):
    if family in HISTORICAL_FAMILIES:
        return DCM.build_family(family, spec, norm, seed, device, dropout)
    if family in DISTANCE_FAMILIES:
        return _build_distance(family, spec, norm, seed, device, dropout)
    raise ValueError(f"unknown family {family!r}")


# ---------------------------------------------------------------------------
# parameter / state helpers
# ---------------------------------------------------------------------------


def parameter_count(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def parameter_breakdown(module: nn.Module) -> dict[str, int]:
    out: dict[str, int] = {}
    for name, child in module.named_children():
        if name == "_v2_dropout_head":
            continue
        out[name] = parameter_count(child)
    out["total"] = parameter_count(module)
    return out


def new_parameter_prefix(family: str) -> str:
    return {
        "base": "",
        "roleblind": "context.",
        "g2l_loo": "context.",
        "g2l_all": "context.",
        "dist_true_fixed": "dist.",
        "dist_blind": "dist.",
    }[family]


def new_parameter_names(model, family: str) -> list[str]:
    prefix = new_parameter_prefix(family)
    if not prefix:
        return []
    return [name for name, _ in model.named_parameters() if name.startswith(prefix)]


def correction_output(module: nn.Module, family: str):
    if family == "base":
        return None
    if family in ("roleblind", "g2l_loo", "g2l_all"):
        return module.context.W_out
    if family in DISTANCE_FAMILIES:
        return module.dist.W_out
    raise KeyError(family)


# ---------------------------------------------------------------------------
# inference
# ---------------------------------------------------------------------------


def to_device(batch: Mapping[str, Any], device) -> dict[str, Any]:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def model_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


def predict(model, cache, device="cpu", batch_size: int = 256) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, int(cache["n_molecules"]), int(batch_size)):
            rows = list(range(start, min(start + int(batch_size), int(cache["n_molecules"]))))
            batch = to_device(D.collate(cache, rows), device)
            out.append(model(batch).detach().float().cpu().numpy().reshape(-1))
    return np.concatenate(out).astype(np.float64)


def predict_with_correction_disabled(model, cache, family: str, device="cpu", batch_size: int = 256) -> np.ndarray:
    out = correction_output(model, family)
    if out is None:
        return predict(model, cache, device=device, batch_size=batch_size)
    w = out.weight.detach().clone()
    b = out.bias.detach().clone()
    try:
        with torch.no_grad():
            out.weight.zero_()
            out.bias.zero_()
        return predict(model, cache, device=device, batch_size=batch_size)
    finally:
        with torch.no_grad():
            out.weight.copy_(w)
            out.bias.copy_(b)


def load_checkpoint(
    model,
    path,
    *,
    expected_sha256: str | None = None,
    expected_family: str | None = None,
    expected_seed: int | None = None,
):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if expected_sha256 is not None and C.sha256_file(path) != expected_sha256:
        raise PermissionError(f"checkpoint sha256 mismatch for {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if expected_family is not None and str(payload.get("family")) != str(expected_family):
        raise PermissionError(f"checkpoint family {payload.get('family')} != {expected_family}")
    if expected_seed is not None and int(payload.get("seed", -1)) != int(expected_seed):
        raise PermissionError(f"checkpoint seed {payload.get('seed')} != {expected_seed}")
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return payload


__all__ = [
    "DistanceCorrectionFixed",
    "LCRCDistanceFixed",
    "build_family",
    "predict",
    "predict_with_correction_disabled",
    "to_device",
    "model_device",
    "load_checkpoint",
    "parameter_count",
    "parameter_breakdown",
    "new_parameter_names",
    "new_parameter_prefix",
    "correction_output",
]
