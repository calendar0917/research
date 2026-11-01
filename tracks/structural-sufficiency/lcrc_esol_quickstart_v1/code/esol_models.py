"""ESOL model builders: unchanged LCRC-A, plus the single optional P2 variant.

``ESOL-BASE`` / ``ESOL-DROP20``
    ``lcrc_model.LCRC`` (the frozen ``lipo_v1``-sized LCRC-A: width 64, stat rank
    32, explicit S1 joint codes, explicit S2 typed two-hop paths, per-atom
    contribution readout, global ``rho``) with the frozen dropout-at-``h`` hook
    attached to ``model.local``.  No module, no input and no readout changes
    relative to the Lipophilicity BASE arm; only ``p`` differs between DROP20
    and BASE (0.20 vs 0.10).

``ESOL-PHYSCHEM6``
    Same model with exactly six extra inputs to the **global correction only**:

        rho_in = [g (42) | mean_h (64) | mean_h2 (64) | log1p(n) | phys (6)]

    The six descriptors are MolWt, MolLogP, TPSA, HBD, HBA, RotatableBonds,
    recomputed from SMILES with RDKit, standardised with Train-only statistics.
    They never enter ``a_v``, the S1 aggregate, the S2 path aggregate, the
    local MLP or ``f``.  Everything that is shared with BASE is initialised
    bit-identically from the BASE model of the same seed (verified), so a
    BASE/PHYSCHEM6 difference cannot be an initialisation artefact.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

import esol_common as C

import lcrc_data as PREP_DATA  # noqa: E402
import lcrc_model as PREP_MODEL  # noqa: E402

N_PHYSCHEM = len(C.PHYSCHEM_NAMES)


# ---------------------------------------------------------------------------
# dropout at the final atom-level local representation (frozen v2 semantics)
# ---------------------------------------------------------------------------


class DropoutAtLocal:
    """Dropout on the output of ``model.local``, before the readout.

    Semantics are those of the frozen
    ``lcrc_lipo_mechanism_generalization_ssl_v2/code/v2_models.py::DropoutAtLocal``
    arm ``"A"``: an ``nn.Dropout`` module registered on the model (so
    ``train()``/``eval()`` propagate) applied through a forward hook on
    ``model.local``.  ``p = 0`` is the exact identity.
    """

    def __init__(self, model: nn.Module, p: float):
        if p < 0 or p >= 1:
            raise ValueError("dropout p must be in [0, 1)")
        self.p = float(p)
        self.module = nn.Dropout(float(p))
        model.add_module("_dropout_at_local", self.module)
        self.target = model.local
        self.handle = self.target.register_forward_hook(self._hook)
        self.calls = 0

    def _hook(self, module, inputs, output):  # noqa: ARG002 - torch hook signature
        self.calls += 1
        if not torch.is_tensor(output):
            raise TypeError("dropout hook expected a tensor output")
        return self.module(output)

    def remove(self) -> None:
        self.handle.remove()


# ---------------------------------------------------------------------------
# ESOL-BASE
# ---------------------------------------------------------------------------


def build_base(
    spec,
    norm: Mapping[str, Any],
    seed: int,
    device: Any = "cpu",
    dropout: float = C.TRAIN_RECIPE["dropout"],
) -> tuple[PREP_MODEL.LCRC, dict[str, Any]]:
    torch.manual_seed(int(seed))
    model = PREP_MODEL.LCRC(spec, norm, PREP_MODEL.DEFAULT_CONFIG)
    hook = DropoutAtLocal(model, float(dropout))
    model.to(device)
    model.eval()
    return model, {
        "family": "base",
        "class": "lcrc_model.LCRC",
        "seed": int(seed),
        "dropout": float(dropout),
        "dropout_hook": hook,
        "parameters": parameter_count(model),
        "parameter_breakdown": parameter_breakdown(model),
        "init_state_hash": C.state_dict_fingerprint(model.state_dict()),
    }


# ---------------------------------------------------------------------------
# ESOL-PHYSCHEM6 (six descriptors, global rho branch only)
# ---------------------------------------------------------------------------


class LCRCPhyschem6(PREP_MODEL.LCRC):
    """LCRC-A whose global correction also reads six recomputed descriptors."""

    def __init__(
        self,
        spec,
        norm: Mapping[str, Any],
        cfg: Mapping[str, Any] | None = None,
        physchem_norm: Mapping[str, Any] | None = None,
    ):
        super().__init__(spec, norm, cfg)
        if physchem_norm is None:
            raise ValueError("LCRCPhyschem6 requires train-only physchem normalisation")
        if list(physchem_norm["names"]) != list(C.PHYSCHEM_NAMES):
            raise ValueError("physchem normalisation names drifted")
        width = int(self.cfg["width"])
        # The only structural change: the rho input is 6 columns wider.
        self.rho = PREP_MODEL.MLP([spec.global_dim + 2 * width + 1 + N_PHYSCHEM, int(self.cfg["rho_hidden"]), 1])
        self.register_buffer("phys_mean", torch.as_tensor(physchem_norm["mean"], dtype=torch.float32))
        self.register_buffer("phys_scale", torch.as_tensor(physchem_norm["scale"], dtype=torch.float32))

    def forward(self, batch: Mapping[str, torch.Tensor], *, return_aux: bool = False):
        n_graphs = int(batch["n_graphs"])
        a, b, h = self._encode(batch)
        atom_batch = batch["atom_batch"]
        width = h.shape[1]
        count = h.new_zeros(n_graphs).index_add_(0, atom_batch, torch.ones_like(atom_batch, dtype=h.dtype))
        denom = count.clamp(min=1.0).unsqueeze(1)
        mean_h = h.new_zeros((n_graphs, width)).index_add_(0, atom_batch, h) / denom
        mean_h2 = h.new_zeros((n_graphs, width)).index_add_(0, atom_batch, h * h) / denom
        contrib_atom = self.f(h).view(-1)
        contrib = h.new_zeros(n_graphs).index_add_(0, atom_batch, contrib_atom)
        g = (batch["global"] - self.global_mean) / self.global_scale
        phys = (batch["physchem"] - self.phys_mean) / self.phys_scale
        rho_in = torch.cat([g, mean_h, mean_h2, torch.log1p(count).unsqueeze(1), phys], dim=1)
        rho = self.rho(rho_in).view(-1)
        y = contrib + rho
        if not return_aux:
            return y
        return y, {
            "contrib": contrib,
            "rho": rho,
            "contrib_atom": contrib_atom,
            "rho_input": rho_in,
            "mean_h": mean_h,
            "mean_h2": mean_h2,
            "h": h,
            "count": count,
            "physchem_used_in": "rho_only",
        }


def build_physchem6(
    spec,
    norm: Mapping[str, Any],
    seed: int,
    device: Any = "cpu",
    dropout: float = C.TRAIN_RECIPE["dropout"],
    physchem_norm: Mapping[str, Any] | None = None,
) -> tuple[LCRCPhyschem6, dict[str, Any]]:
    base, base_meta = build_base(spec, norm, seed, "cpu", dropout)
    torch.manual_seed(int(seed))
    model = LCRCPhyschem6(spec, norm, PREP_MODEL.DEFAULT_CONFIG, physchem_norm=physchem_norm)
    own = model.state_dict()
    shared = {
        key: value.detach().clone()
        for key, value in base.state_dict().items()
        if key in own and tuple(own[key].shape) == tuple(value.shape)
    }
    model.load_state_dict({**own, **shared}, strict=True)
    audit = shared_init_audit(base, model)
    if not audit["all_shared_bitwise_identical"]:
        raise RuntimeError(f"PHYSCHEM6 shared initialisation mismatch: {audit['mismatched']}")
    hook = DropoutAtLocal(model, float(dropout))
    model.to(device)
    model.eval()
    return model, {
        "family": "physchem6",
        "class": "LCRCPhyschem6(lcrc_model.LCRC + 6 descriptors on rho)",
        "seed": int(seed),
        "dropout": float(dropout),
        "dropout_hook": hook,
        "parameters": parameter_count(model),
        "parameters_base": parameter_count(base),
        "parameters_new": parameter_count(model) - parameter_count(base),
        "parameter_breakdown": parameter_breakdown(model),
        "init_state_hash": C.state_dict_fingerprint(model.state_dict()),
        "base_init_state_hash": base_meta["init_state_hash"],
        "shared_init": {k: v for k, v in audit.items() if k != "shared_keys"},
        "definition": (
            "rho_in = [global | mean_v h_v | mean_v h_v^2 | log1p(n_atoms) | 6 standardised descriptors]; "
            "local S1/S2 representation and f are unchanged"
        ),
    }


def build(
    model_kind: str,
    spec,
    norm: Mapping[str, Any],
    seed: int,
    device: Any = "cpu",
    *,
    dropout: float | None = None,
    physchem_norm: Mapping[str, Any] | None = None,
):
    p = float(C.MODEL_DROPOUT[model_kind] if dropout is None else dropout)
    if model_kind in ("base", "drop20"):
        model, meta = build_base(spec, norm, seed, device, p)
        meta["family"] = model_kind
        return model, meta
    if model_kind == "physchem6":
        return build_physchem6(spec, norm, seed, device, p, physchem_norm=physchem_norm)
    raise ValueError(f"unknown ESOL model {model_kind!r}")


# ---------------------------------------------------------------------------
# parameter / init audits
# ---------------------------------------------------------------------------


def parameter_count(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def parameter_breakdown(module: nn.Module) -> dict[str, int]:
    out: dict[str, int] = {}
    for name, child in module.named_children():
        if name == "_dropout_at_local":
            continue
        out[name] = parameter_count(child)
    out["total"] = parameter_count(module)
    return out


def shared_init_audit(base: nn.Module, model: nn.Module) -> dict[str, Any]:
    base_state = base.state_dict()
    model_state = model.state_dict()
    shared_keys: list[str] = []
    mismatched: list[str] = []
    for key, value in base_state.items():
        if key in model_state and tuple(model_state[key].shape) == tuple(value.shape):
            shared_keys.append(key)
            if not torch.equal(value, model_state[key]):
                mismatched.append(key)
    return {
        "n_shared_entries": len(shared_keys),
        "all_shared_bitwise_identical": not mismatched,
        "mismatched": mismatched,
        "model_only_keys": sorted(set(model_state) - set(base_state)),
        "base_only_keys": sorted(set(base_state) - set(model_state)),
    }


def new_parameter_names(model: nn.Module) -> list[str]:
    """Parameters that do not exist in BASE (only the wider rho input layer)."""
    if not isinstance(model, LCRCPhyschem6):
        return []
    return [name for name, _ in model.named_parameters() if name.startswith("rho.lins.0.")]


# ---------------------------------------------------------------------------
# inference
# ---------------------------------------------------------------------------


def collate(cache: Mapping[str, Any], rows: Sequence[int], *, physchem: np.ndarray | None = None):
    batch = PREP_DATA.collate(cache, rows)
    if physchem is not None:
        idx = np.asarray(rows, dtype=np.int64)
        batch["physchem"] = torch.as_tensor(np.asarray(physchem, dtype=np.float32)[idx], dtype=torch.float32)
    return batch


def to_device(batch: Mapping[str, Any], device) -> dict[str, Any]:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def predict(
    model: nn.Module,
    cache: Mapping[str, Any],
    device: Any = "cpu",
    batch_size: int = 256,
    *,
    physchem: np.ndarray | None = None,
) -> np.ndarray:
    model.eval()
    out = []
    n = int(cache["n_molecules"])
    with torch.no_grad():
        for start in range(0, n, int(batch_size)):
            rows = list(range(start, min(start + int(batch_size), n)))
            batch = to_device(collate(cache, rows, physchem=physchem), device)
            out.append(model(batch).detach().float().cpu().numpy().reshape(-1))
    return np.concatenate(out).astype(np.float64)


def load_checkpoint(
    model,
    path,
    *,
    expected_sha256: str | None = None,
    expected_family: str | None = None,
    expected_seed: int | None = None,
) -> dict[str, Any]:
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
    "N_PHYSCHEM",
    "DropoutAtLocal",
    "LCRCPhyschem6",
    "build",
    "build_base",
    "build_physchem6",
    "parameter_count",
    "parameter_breakdown",
    "shared_init_audit",
    "new_parameter_names",
    "collate",
    "to_device",
    "predict",
    "load_checkpoint",
]
