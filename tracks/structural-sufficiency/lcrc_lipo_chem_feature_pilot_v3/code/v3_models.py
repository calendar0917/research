"""Model builders for the v3 chemistry pilot.

The only architectural change is the width of the global input of ``rho``:

    y = sum_v f(h_v) + rho([g(42) | chem(12 or 13)], mean h, mean h^2, log1p n)

The local encoders, S1, S2, ``f``, the readout and every atom51/bond13/scalar9
semantic are byte-identical to the frozen v2 A/C models.  The training loop is
v2's ``train_job``; this module only installs a builder dispatch for the three
v3 arm names.

Weight migration
----------------
For the same seed we build the original model first, then copy every
same-shaped tensor into the augmented model.  ``rho.lins.0.weight`` is widened
by copying the original first 42 global columns, zeroing the new chemistry
columns, and shifting the original ``mean_h`` / ``mean_h2`` / ``count``
columns to the right.  The new input columns therefore start with exactly zero
effect.  This makes the zero-chem forward numerically identical to the original
model at initialisation and keeps the comparison a feature-input comparison,
not a capacity/initialisation comparison.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

import v3_common as C
import v3_features as F

prep_model = C.C.prep_model


# ---------------------------------------------------------------------------
# weight migration
# ---------------------------------------------------------------------------


def rho_input_layout(base_dim: int, n_extra: int, width: int) -> dict[str, tuple[int, int]]:
    """Offsets into ``rho.lins[0]`` input for the frozen layout."""
    start = 0
    layout = {"global_base": (start, start + base_dim)}
    start += base_dim
    layout["global_extra"] = (start, start + n_extra)
    start += n_extra
    layout["mean_h"] = (start, start + width)
    start += width
    layout["mean_h2"] = (start, start + width)
    start += width
    layout["count"] = (start, start + 1)
    return layout


def migrate_rho_first_layer(
    base_weight: torch.Tensor,
    new_weight: torch.Tensor,
    *,
    base_dim: int,
    n_extra: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Widen the first rho layer with zero new input columns."""
    if base_weight.ndim != 2 or new_weight.ndim != 2:
        raise ValueError("rho first-layer weights must be 2-D")
    if base_weight.shape[0] != new_weight.shape[0]:
        raise ValueError("rho hidden width changed")
    if new_weight.shape[1] != base_weight.shape[1] + n_extra:
        raise ValueError("new rho input width is not base + n_extra")
    out = new_weight.clone()
    out[:, :base_dim] = base_weight[:, :base_dim]
    out[:, base_dim : base_dim + n_extra] = 0.0
    out[:, base_dim + n_extra :] = base_weight[:, base_dim:]
    audit = {
        "base_input_width": int(base_weight.shape[1]),
        "new_input_width": int(new_weight.shape[1]),
        "base_dim": int(base_dim),
        "n_extra": int(n_extra),
        "new_columns_initialised_to_zero": bool(torch.count_nonzero(out[:, base_dim : base_dim + n_extra]) == 0),
        "base_global_columns_copied_exactly": bool(torch.equal(out[:, :base_dim], base_weight[:, :base_dim])),
        "shifted_columns_copied_exactly": bool(torch.equal(out[:, base_dim + n_extra :], base_weight[:, base_dim:])),
        "layout": rho_input_layout(base_dim, n_extra, (base_weight.shape[1] - base_dim - 1) // 2),
    }
    if not all(
        audit[k]
        for k in (
            "new_columns_initialised_to_zero",
            "base_global_columns_copied_exactly",
            "shifted_columns_copied_exactly",
        )
    ):
        raise RuntimeError(f"rho weight migration failed: {audit}")
    return out, audit


def migrate_state(
    base_state: Mapping[str, torch.Tensor],
    own_state: Mapping[str, torch.Tensor],
    *,
    base_dim: int,
    n_extra: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    out = {k: v.detach().clone() for k, v in own_state.items()}
    shared = []
    for key, value in base_state.items():
        if key in out and tuple(out[key].shape) == tuple(value.shape):
            out[key] = value.detach().clone()
            shared.append(key)
    key = "rho.lins.0.weight"
    out[key], rho_audit = migrate_rho_first_layer(
        base_state[key].detach().clone(), own_state[key].detach().clone(), base_dim=base_dim, n_extra=n_extra
    )
    base_buffers = {}
    for name in ("global_mean", "global_scale"):
        if name in base_state and name in out:
            same_prefix = torch.equal(out[name][:base_dim], base_state[name])
            base_buffers[name] = {
                "base_width": int(base_state[name].numel()),
                "new_width": int(out[name].numel()),
                "first_base_dim_matches": bool(same_prefix),
            }
            if not same_prefix:
                raise RuntimeError(f"augmented norm buffer {name} changed the original {base_dim} entries")
    mismatched = [k for k in shared if not torch.equal(out[k], base_state[k])]
    audit = {
        "n_shared_tensors": len(shared),
        "shared_tensors": shared,
        "all_shared_bitwise_equal": not mismatched,
        "mismatched": mismatched,
        "rho_first_layer": rho_audit,
        "global_buffers": base_buffers,
        "new_only_keys": sorted(set(out) - set(base_state)),
        "base_only_keys": sorted(set(base_state) - set(out)),
    }
    if mismatched:
        raise RuntimeError(f"shared tensors are not bitwise identical: {mismatched}")
    return out, audit


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------


def _base_parts(spec, norm):
    base_spec = C.load_frozen_spec()
    base_dim = int(base_spec.global_dim)
    if int(spec.global_dim) <= base_dim:
        raise ValueError("augmented spec has no extra global dimensions")
    n_extra = int(spec.global_dim) - base_dim
    base_norm = F.slice_base_norm(norm, base_dim)
    return base_spec, base_norm, base_dim, n_extra


def build_chem_model(arm: str, spec, norm, seed: int, device="cpu", *, dropout: float = 0.0):
    if arm not in C.ARMS:
        raise ValueError(f"unknown v3 arm {arm!r}")
    base_arm, variant = C.ARM_VARIANT[arm]
    base_spec, base_norm, base_dim, n_extra = _base_parts(spec, norm)
    if n_extra != (12 if variant == "chem12" else 13):
        raise ValueError(f"arm {arm} has unexpected extra width {n_extra}")

    if base_arm == "A":
        base_model = C.M.build_a(base_spec, base_norm, int(seed))
    elif base_arm == "C":
        base_model = C.M.build_c(base_spec, base_norm, int(seed))
    else:
        raise ValueError(base_arm)
    base_state = {k: v.detach().clone() for k, v in base_model.state_dict().items()}

    torch.manual_seed(int(seed))
    if base_arm == "A":
        model = prep_model.LCRC(spec, norm, prep_model.DEFAULT_CONFIG)
    else:
        model = C.M.V1GINE(spec, norm, hidden=C.C.v1c.GINE_HIDDEN, layers=C.C.v1c.GINE_LAYERS)

    own_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    migrated, audit = migrate_state(base_state, own_state, base_dim=base_dim, n_extra=n_extra)
    model.load_state_dict(migrated, strict=True)
    model.to(device)
    hook = C.M.DropoutAtLocal(model, base_arm, float(dropout))
    meta = {
        "arm": arm,
        "base_arm": base_arm,
        "feature_variant": variant,
        "dropout": float(dropout),
        "dropout_hook": hook,
        "migration": audit,
        "parameters_augmented": C.M.parameter_count(model),
        "parameters_original": C.M.parameter_count(base_model),
        "parameter_delta": C.M.parameter_count(model) - C.M.parameter_count(base_model),
        "expected_parameter_delta": int(C.C.prep_model.DEFAULT_CONFIG["rho_hidden"] * n_extra)
        if base_arm == "A"
        else int(C.C.v1c.GINE_RHO_HIDDEN * n_extra),
        "base_state_hash": C.M.model_state_hash(base_model),
        "augmented_state_hash": C.M.model_state_hash(model),
        "shared_with_original": base_model,
    }
    if meta["parameter_delta"] != meta["expected_parameter_delta"]:
        raise RuntimeError(f"unexpected parameter delta {meta['parameter_delta']} != {meta['expected_parameter_delta']}")
    return model, meta


@contextmanager
def patched_model_builder():
    """Install the v3 arm dispatch for the duration of v2's training loop."""
    original = C.M.build_model

    def dispatch(arm, spec, norm, seed, device="cpu", *, dropout: float = 0.0):
        if arm in C.ARMS:
            return build_chem_model(arm, spec, norm, seed, device, dropout=dropout)
        return original(arm, spec, norm, seed, device, dropout=dropout)

    C.M.build_model = dispatch
    try:
        yield
    finally:
        C.M.build_model = original


# ---------------------------------------------------------------------------
# training job
# ---------------------------------------------------------------------------


@dataclass
class ChemTrainJob(C.T.TrainJob):
    v3_audit: dict[str, Any] | None = None

    def validate(self) -> None:
        if self.arm not in C.ARMS:
            raise ValueError(f"arm must be one of {C.ARMS}")
        if self.recipe != C.RECIPE:
            raise ValueError(f"v3 pilot only runs recipe {C.RECIPE}")
        if self.fold != C.FOLD:
            raise ValueError(f"v3 pilot only runs Fold {C.FOLD}")
        if self.seed != C.SEED:
            raise ValueError(f"v3 pilot only runs seed {C.SEED}")
        if self.variant != "scratch":
            raise ValueError("v3 pilot only runs scratch models (no SSL)")
        if self.device not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        if self.global_extra_train is None or self.global_extra_dev is None:
            raise ValueError("v3 chemistry arms require global_extra_train/dev")
        if self.spec_override is None or self.norm_override is None:
            raise ValueError("v3 chemistry arms require spec_override/norm_override")


def train_chem_job(job: ChemTrainJob) -> dict[str, Any]:
    """Run v2's training loop with only the model-builder dispatch changed."""
    job.validate()
    with patched_model_builder():
        payload = C.T.train_job(job)
    out_dir = Path(job.out_root) / job.key
    v2_protocol = payload.get("protocol_id")
    v2_config_hash = payload.get("config_hash")
    payload["protocol_id"] = C.PROTOCOL_ID
    payload["round_id"] = C.ROUND_ID
    payload["config_hash"] = C.config_hash(C.load_config())
    payload["v2_loop"] = {
        "protocol_id": v2_protocol,
        "config_hash": v2_config_hash,
        "function": "v2_training.train_job reused verbatim",
        "model_builder_dispatch": "v3_models.build_chem_model",
    }
    payload["v3"] = {
        "protocol_id": C.PROTOCOL_ID,
        "arm": job.arm,
        "base_arm": C.ARM_VARIANT[job.arm][0],
        "feature_variant": C.ARM_VARIANT[job.arm][1],
        "fold": int(job.fold),
        "seed": int(job.seed),
        "recipe": job.recipe,
        "audit": job.v3_audit,
        "global_extra": job.global_extra_audit,
    }
    # The original parameter count is not available after the loop; recompute it
    # from the frozen base model for an explicit capacity record.
    base_spec = C.load_frozen_spec()
    base_norm = F.slice_base_norm(job.norm_override, int(base_spec.global_dim))
    if C.ARM_VARIANT[job.arm][0] == "A":
        base_model = C.M.build_a(base_spec, base_norm, int(job.seed))
    else:
        base_model = C.M.build_c(base_spec, base_norm, int(job.seed))
    payload["v3"]["original_parameters"] = int(C.M.parameter_count(base_model))
    payload["v3"]["new_parameters"] = int(payload["model"]["parameters_nominal"]) - int(
        payload["v3"]["original_parameters"]
    )
    payload["v3"]["parameter_fraction_of_original"] = (
        payload["v3"]["new_parameters"] / payload["v3"]["original_parameters"]
    )
    payload["v3"]["local_encoder_state_unchanged"] = True
    payload["v3"]["s2_typed_semantics_unchanged"] = True
    payload["v3"]["s2_local_encoder_present"] = bool(C.ARM_VARIANT[job.arm][0] == "A")
    payload["v3"]["atom51_bond13_scalar9_unchanged"] = True
    payload["v3"]["iterative_message_passing"] = bool(C.ARM_VARIANT[job.arm][0] == "C")
    payload["v3"]["training_loop_changed"] = False
    C.write_json(out_dir / "train.json", payload)
    return payload


__all__ = [
    "rho_input_layout",
    "migrate_rho_first_layer",
    "migrate_state",
    "build_chem_model",
    "patched_model_builder",
    "ChemTrainJob",
    "train_chem_job",
]
