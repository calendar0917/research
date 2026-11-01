"""ESOL arms for ``lcrc_esol_physchem_s2_v1``.

``ESOL-BASE``
    Unchanged LCRC-A, unchanged dropout hook.  Reused **read-only** from the
    previous ESOL round: this round never retrains it.

``ESOL-PHYSCHEM6`` (H1)
    ``rho_in = [g(42) | mean_h(64) | mean_h2(64) | log1p(n) | phys(6)]``.
    Three repairs relative to the v1 implementation:

    1. **alignment** — the six descriptors arrive from ``views["physchem_train"]
       / ["physchem_valid"]`` (cache-ordered).  ``collate`` refuses a matrix
       whose length differs from the cache, which makes the v1 failure mode
       (local row → CSV row) structurally impossible;
    2. **single standardisation** — ``collate`` forwards *raw* RDKit values and
       ``forward`` applies ``(raw - train_mean) / train_scale`` exactly once;
    3. **initialisation** — the widened first ``rho`` layer is initialised
       column by column from the freshly seeded BASE: the original 171 input
       columns and the bias are copied bitwise, the six new columns are zero.
       So at step 0 the arm is numerically identical to BASE.

``ESOL-S2BLIND-E`` (H2)
    The validated capacity-matched control of
    ``lcrc_lipo_mechanism_generalization_ssl_v2/code/v2_models.py`` ported
    unchanged (imported, not re-derived): the typed-S2 modules are removed, the
    200-wide S2 slot into the LocalMLP stays zero (dead), and a
    320→135→200 S1-context network is injected as a parameter-free fixed
    residual into the LocalMLP hidden state.  Effective capacity is BASE − 57
    parameters (0.023%).  It never reads ``path_cat``, ``path_end63``,
    ``path_end``, ``path_closure``, ``path_graph`` or ``path_atom``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

import esol2_common as C
import esol_common as V1C
import esol_models as V1M
import v2_models as V2M

import lcrc_data as PREP_DATA  # noqa: E402
import lcrc_model as PREP_MODEL  # noqa: E402

N_PHYSCHEM = C.N_PHYSCHEM
TYPED_S2_KEYS = tuple(V2M.TYPED_S2_KEYS)


# ---------------------------------------------------------------------------
# dropout at the final atom-level local representation (frozen v2 semantics)
# ---------------------------------------------------------------------------


class DropoutAtLocal:
    """Dropout on the output of ``model.local``, before the readout."""

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
# ESOL-BASE (read-only reuse of the frozen builder)
# ---------------------------------------------------------------------------


def build_base(
    spec,
    norm: Mapping[str, Any],
    seed: int,
    device: Any = "cpu",
    dropout: float = C.TRAIN_RECIPE["dropout"],
):
    """v1's frozen builder, reused verbatim (BASE is never retrained here)."""
    return V1M.build_base(spec, norm, seed, device, dropout)


# ---------------------------------------------------------------------------
# ESOL-PHYSCHEM6 (H1): six descriptors, global rho branch only
# ---------------------------------------------------------------------------


class LCRCPhyschem6(PREP_MODEL.LCRC):
    """LCRC-A whose global correction also reads six recomputed descriptors.

    ``batch["physchem"]`` must carry the **raw** RDKit values for the molecules
    in ``batch["atom_batch"]`` order; standardisation happens here, once.
    """

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
        if physchem_norm.get("valid_or_test_used") is not False:
            raise PermissionError("physchem normalisation is not certified Train-only")
        width = int(self.cfg["width"])
        # The only structural change: the rho input is N_PHYSCHEM columns wider.
        self.rho = PREP_MODEL.MLP(
            [spec.global_dim + 2 * width + 1 + N_PHYSCHEM, int(self.cfg["rho_hidden"]), 1]
        )
        self.register_buffer("phys_mean", torch.as_tensor(physchem_norm["mean"], dtype=torch.float32))
        self.register_buffer("phys_scale", torch.as_tensor(physchem_norm["scale"], dtype=torch.float32))

    def forward(self, batch: Mapping[str, torch.Tensor], *, return_aux: bool = False):
        n_graphs = int(batch["n_graphs"])
        # Structural path first: descriptors cannot reach a / b / h / f.
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
        phys_raw = batch["physchem"]
        phys = (phys_raw - self.phys_mean) / self.phys_scale  # the single standardisation
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
            "physchem_raw": phys_raw,
            "physchem_standardised": phys,
            "physchem_used_in": "rho_only",
        }


def build_physchem6(
    spec,
    norm: Mapping[str, Any],
    seed: int,
    device: Any = "cpu",
    dropout: float = C.TRAIN_RECIPE["dropout"],
    physchem_norm: Mapping[str, Any] | None = None,
):
    """BASE-identical initialisation except the six new rho input columns (= 0)."""
    base, base_meta = build_base(spec, norm, seed, "cpu", dropout)
    torch.manual_seed(int(seed))
    model = LCRCPhyschem6(spec, norm, PREP_MODEL.DEFAULT_CONFIG, physchem_norm=physchem_norm)
    own = model.state_dict()
    base_state = base.state_dict()
    merged = {key: value.detach().clone() for key, value in own.items()}
    for key, value in base_state.items():
        if key in merged and tuple(merged[key].shape) == tuple(value.shape):
            merged[key] = value.detach().clone()
    # The widened first rho layer: old columns + bias bitwise from BASE, new 0.
    w_key = "rho.lins.0.weight"
    w_new, w_base = merged[w_key], base_state[w_key]
    n_old = int(w_base.shape[1])
    if int(w_new.shape[1]) != n_old + N_PHYSCHEM:
        raise RuntimeError(f"rho input width {w_new.shape[1]} != {n_old} + {N_PHYSCHEM}")
    w_new = w_new.clone()
    w_new[:, :n_old] = w_base
    w_new[:, n_old:] = 0.0
    merged[w_key] = w_new
    model.load_state_dict(merged, strict=True)
    audit = shared_init_audit_physchem(base, model)
    if not audit["all_shared_bitwise_identical"]:
        raise RuntimeError(f"PHYSCHEM6 shared initialisation mismatch: {audit['mismatched']}")
    if not (audit["rho_old_columns_bitwise_identical"] and audit["rho_new_columns_zero"] and audit["rho_bias_bitwise_identical"]):
        raise RuntimeError("PHYSCHEM6 rho first-layer initialisation is not BASE-consistent")
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
        "shared_init": audit,
        "physchem_input": {
            "columns": list(C.PHYSCHEM_NAMES),
            "rho_input_width_base": int(n_old),
            "rho_input_width_physchem6": int(w_new.shape[1]),
            "new_columns_initialised_to_zero": True,
            "standardisation": "once, inside forward, Train-only buffers",
        },
        "definition": (
            "rho_in = [global | mean_v h_v | mean_v h_v^2 | log1p(n_atoms) | 6 standardised descriptors]; "
            "local S1/S2 representation and f are unchanged"
        ),
    }


def shared_init_audit_physchem(base: nn.Module, model: nn.Module) -> dict[str, Any]:
    base_state = base.state_dict()
    own = model.state_dict()
    shared_keys: list[str] = []
    mismatched: list[str] = []
    for key, value in base_state.items():
        if key == "rho.lins.0.weight":
            continue
        if key in own and tuple(own[key].shape) == tuple(value.shape):
            shared_keys.append(key)
            if not torch.equal(value, own[key]):
                mismatched.append(key)
    w_key = "rho.lins.0.weight"
    w_new, w_base = own[w_key], base_state[w_key]
    n_old = int(w_base.shape[1])
    return {
        "n_shared_entries": len(shared_keys),
        "all_shared_bitwise_identical": not mismatched,
        "mismatched": mismatched,
        "shared_keys_excluding_rho_first_layer": shared_keys,
        "rho_first_layer_widened": True,
        "rho_old_columns": int(n_old),
        "rho_new_columns": int(w_new.shape[1] - n_old),
        "rho_old_columns_bitwise_identical": bool(torch.equal(w_new[:, :n_old], w_base)),
        "rho_new_columns_zero": bool(torch.equal(w_new[:, n_old:], torch.zeros_like(w_new[:, n_old:]))),
        "rho_bias_bitwise_identical": bool(torch.equal(own["rho.lins.0.bias"], base_state["rho.lins.0.bias"])),
        "model_only_keys": sorted(set(own) - set(base_state)),
        "base_only_keys": sorted(set(base_state) - set(own)),
    }


# ---------------------------------------------------------------------------
# ESOL-S2BLIND-E (H2): capacity-matched typed-S2-blind control
# ---------------------------------------------------------------------------


def build_s2blind_e(
    spec,
    norm: Mapping[str, Any],
    seed: int,
    device: Any = "cpu",
    dropout: float = C.TRAIN_RECIPE["dropout"],
):
    """The historical E control (v2_models.build_e), dropout 0.10 attached here."""
    model, base, shared = V2M.build_e(spec, norm, int(seed), "cpu")
    audit = V2M.shared_init_audit(base, model)
    if not audit["all_shared_bitwise_identical"]:
        raise RuntimeError(f"E shared initialisation mismatch: {audit['mismatched']}")
    capacity = e_capacity_audit(model, base)
    if not capacity["passed"]:
        raise RuntimeError(f"E capacity audit failed: {capacity}")
    our_base, our_base_meta = build_base(spec, norm, int(seed), "cpu", dropout)
    base_state, our_state = base.state_dict(), our_base.state_dict()
    base_matches_our_builder = bool(
        set(base_state) == set(our_state) and all(torch.equal(base_state[k], our_state[k]) for k in base_state)
    )
    if not base_matches_our_builder:
        raise RuntimeError("E's reference BASE differs from this round's BASE initialisation")
    hook = DropoutAtLocal(model, float(dropout))
    model.to(device)
    model.eval()
    return model, {
        "family": "s2blind_e",
        "class": "v2_models.LCRCCapacityMatched (imported unchanged)",
        "reference_base_init_state_hash": our_base_meta["init_state_hash"],
        "reference_base_matches_our_builder": base_matches_our_builder,
        "seed": int(seed),
        "dropout": float(dropout),
        "dropout_hook": hook,
        "parameters": parameter_count(model),
        "parameters_effective": int(capacity["effective_parameters"]),
        "parameters_base": int(capacity["base_parameters"]),
        "parameters_new_nominal": parameter_count(model) - int(capacity["base_parameters"]),
        "parameter_breakdown": parameter_breakdown(model),
        "init_state_hash": C.state_dict_fingerprint(model.state_dict()),
        "shared_init": {k: v for k, v in audit.items() if k != "shared_keys"},
        "capacity": capacity,
        "definition": (
            "typed S2 path tensors removed; original 200-wide S2 LocalMLP slot stays zero (dead); "
            "320->135->200 S1-context replacement injected into the LocalMLP hidden via a "
            "parameter-free fixed residual; atom/bond encoders, f and rho unchanged"
        ),
        "shared_with_base": bool(audit["all_shared_bitwise_identical"]),
        "uses_typed_s2_inputs": False,
    }


def e_capacity_audit(model: nn.Module, base: nn.Module) -> dict[str, Any]:
    """Nominal vs effective parameters, computed from the model (not copied)."""
    local = model.local
    a_out = int(local.atom_out)
    rank = int(local.rank)
    replacement_out = int(local.replacement_out)
    lo = 3 * a_out + rank
    hi = lo + replacement_out
    weight = local.mlp.lins[0].weight
    dead = int(weight[:, lo:hi].numel())
    nominal = parameter_count(model)
    base_params = parameter_count(base)
    s2_modules = sum(
        parameter_count(getattr(base.local, name))
        for name in ("path_emb", "end_emb", "path_u", "path_v", "joint_proj", "close_emb", "path_gate")
    )
    replacement = parameter_count(local.replacement)
    effective = nominal - dead
    out = {
        "nominal_parameters": int(nominal),
        "base_parameters": int(base_params),
        "typed_s2_module_parameters_absent": int(s2_modules),
        "replacement_parameters": int(replacement),
        "dead_localmlp_s2_input_columns": int(dead),
        "dead_column_range": [int(lo), int(hi)],
        "s2_slot_width": int(replacement_out),
        "effective_parameters": int(effective),
        "delta_effective_vs_base": int(effective - base_params),
        "delta_fraction_vs_base": float((effective - base_params) / base_params),
        "nominal_identity_nominal_eq_base_minus_s2_plus_replacement": bool(
            nominal == base_params - s2_modules + replacement
        ),
        "effective_identity_effective_eq_nominal_minus_dead": bool(effective == nominal - dead),
        "effective_equals_base_minus_s2_plus_replacement_minus_dead": bool(
            effective == base_params - s2_modules + replacement - dead
        ),
        "expected_effective_parameters": int(C.E_EFFECTIVE_PARAMETERS),
        "expected_nominal_parameters": int(C.E_NOMINAL_PARAMETERS),
        "effective_matches_historical_control": bool(effective == C.E_EFFECTIVE_PARAMETERS),
        "nominal_matches_historical_control": bool(nominal == C.E_NOMINAL_PARAMETERS),
        "typed_s2_inputs_used": False,
        "iterative_hidden_message_passing": any(
            "message" in name.lower() or "attention" in name.lower() or "gine" in name.lower()
            for name, _ in model.named_modules()
        ),
        "typed_s2_modules_present": [
            name for name in ("path_emb", "end_emb", "path_u", "path_v", "joint_proj", "close_emb", "path_gate")
            if hasattr(local, name)
        ],
        "forward_reads_typed_s2_keys_statically": sorted(
            set(V2M.LocalContentCapacityMatched.forward.__code__.co_names) & set(TYPED_S2_KEYS)
        ),
    }
    out["dead_weight_max_abs"] = float(weight[:, lo:hi].detach().abs().max())
    out["passed"] = bool(
        out["effective_matches_historical_control"]
        and out["nominal_matches_historical_control"]
        and out["nominal_identity_nominal_eq_base_minus_s2_plus_replacement"]
        and out["effective_identity_effective_eq_nominal_minus_dead"]
        and not out["typed_s2_modules_present"]
        and not out["forward_reads_typed_s2_keys_statically"]
        and not out["iterative_hidden_message_passing"]
    )
    return out


# ---------------------------------------------------------------------------
# generic build / audit helpers
# ---------------------------------------------------------------------------


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
    if model_kind == "base":
        model, meta = build_base(spec, norm, seed, device, p)
        meta["family"] = "base"
        return model, meta
    if model_kind == "physchem6":
        return build_physchem6(spec, norm, seed, device, p, physchem_norm=physchem_norm)
    if model_kind == "s2blind_e":
        return build_s2blind_e(spec, norm, seed, device, p)
    raise ValueError(f"unknown ESOL model {model_kind!r}")


def parameter_count(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def parameter_breakdown(module: nn.Module) -> dict[str, int]:
    out: dict[str, int] = {}
    for name, child in module.named_children():
        if name == "_dropout_at_local":
            continue
        out[name] = parameter_count(child)
        if name == "local":
            for sub_name, sub_child in child.named_children():
                out[f"local.{sub_name}"] = parameter_count(sub_child)
    out["total"] = parameter_count(module)
    return out


# ---------------------------------------------------------------------------
# collation / inference
# ---------------------------------------------------------------------------


def collate(cache: Mapping[str, Any], rows: Sequence[int], *, physchem: np.ndarray | None = None):
    batch = PREP_DATA.collate(cache, rows)
    if physchem is not None:
        matrix = np.asarray(physchem)
        if matrix.ndim != 2 or matrix.shape[1] != N_PHYSCHEM:
            raise ValueError(f"physchem matrix must be 2-D with 6 columns, got {matrix.shape}")
        if int(matrix.shape[0]) != int(cache["n_molecules"]):
            raise ValueError(
                "physchem matrix is not aligned with this cache: "
                f"{matrix.shape[0]} rows vs {cache['n_molecules']} molecules "
                "(cache-local rows must not index a full CSV-row-order matrix)"
            )
        idx = np.asarray(rows, dtype=np.int64)
        batch["physchem"] = torch.as_tensor(matrix[idx], dtype=torch.float32)
    return batch


def to_device(batch: Mapping[str, Any], device):
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


# ---------------------------------------------------------------------------
# information-path probes (CPU only)
# ---------------------------------------------------------------------------


class RestrictedBatch(dict):
    """A batch whose *access* to typed-S2 tensors raises (v2 semantics)."""

    def __getitem__(self, key):
        if key in TYPED_S2_KEYS:
            raise KeyError(f"this arm must not read typed-S2 tensor {key!r}")
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key in TYPED_S2_KEYS:
            raise KeyError(f"this arm must not read typed-S2 tensor {key!r}")
        return super().get(key, default)


def restricted_batch(batch: Mapping[str, Any]) -> RestrictedBatch:
    return RestrictedBatch(dict(batch))


class AccessLog(dict):
    """Batch that records the order in which keys are read."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.order: list[str] = []

    def __getitem__(self, key):
        self.order.append(str(key))
        return super().__getitem__(key)


def perturb_typed_s2(
    batch: Mapping[str, Any],
    *,
    mode: str,
    n_path_codes: int = 400,
    n_nbr_codes: int = 80,
) -> dict[str, Any]:
    """Change only typed-S2 content, keeping atom / S1 / global / scalar tensors."""
    out = {key: value for key, value in batch.items()}
    if not torch.is_tensor(out.get("path_cat")) or not bool(out["path_cat"].numel()):
        raise ValueError("perturbation needs a batch with at least one typed S2 path")
    if mode == "code_shift":
        out["path_cat"] = (out["path_cat"] + 7) % int(n_path_codes)
        out["path_end63"] = (out["path_end63"] * 3 + 5) % int(n_nbr_codes)
    elif mode == "closure_flip":
        out["path_closure"] = 1 - out["path_closure"]
    elif mode == "path_end_shift":
        n_atoms = int(out["atom_feat"].shape[0])
        out["path_end"] = (out["path_end"] + 1) % n_atoms
    else:
        raise ValueError(f"unknown perturbation mode {mode!r} for keys {sorted(TYPED_S2_KEYS)}")
    return out


__all__ = [
    "N_PHYSCHEM",
    "TYPED_S2_KEYS",
    "DropoutAtLocal",
    "LCRCPhyschem6",
    "build",
    "build_base",
    "build_physchem6",
    "build_s2blind_e",
    "parameter_count",
    "parameter_breakdown",
    "shared_init_audit_physchem",
    "e_capacity_audit",
    "collate",
    "to_device",
    "predict",
    "load_checkpoint",
    "RestrictedBatch",
    "restricted_batch",
    "AccessLog",
    "perturb_typed_s2",
]
