"""Model builders and training plumbing for the v4 local-chemistry pilot.

Both arms use the frozen LCRC-A graph and the frozen v2 training loop.  The
*only* change is the width of the atom encoder input: 51 -> 55.  The original
51 columns keep the historical seed-0 initialisation bit-for-bit; the four new
columns start at zero so that a zero AtomChem4 input reproduces the original
model exactly at initialisation.

``A_AtomChem4`` feeds the true per-atom features.  ``A_BroadcastChem4`` feeds
the per-molecule mean of the same features to every atom.  The two arms share
the feature definitions, the parameter count, the initialisation and the
training recipe; only the atom <-> feature correspondence differs.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

import v4_common as C

ATOM_WEIGHT_KEY = "atom_encoder.lins.0.weight"


# ---------------------------------------------------------------------------
# initialisation migration
# ---------------------------------------------------------------------------


def migrate_atom_input(
    base_state: Mapping[str, torch.Tensor],
    own_state: Mapping[str, torch.Tensor],
    *,
    base_dim: int = C.ATOM_DIM_BASE,
    n_extra: int = C.N_ATOM_EXTRA,
    atom_hidden: int = C.ATOM_HIDDEN,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Copy the base state; widen the atom input with zero new columns."""
    out = {k: v.detach().clone() for k, v in own_state.items()}
    shared = []
    for key, value in base_state.items():
        if key in out and tuple(out[key].shape) == tuple(value.shape):
            out[key] = value.detach().clone()
            shared.append(key)
    if ATOM_WEIGHT_KEY not in base_state:
        raise KeyError("base state lacks the atom encoder first layer")
    base_w = base_state[ATOM_WEIGHT_KEY].detach().clone()
    own_w = own_state[ATOM_WEIGHT_KEY].detach().clone()
    if tuple(base_w.shape) != (atom_hidden, base_dim):
        raise ValueError(f"base atom weight shape {tuple(base_w.shape)} != {(atom_hidden, base_dim)}")
    if tuple(own_w.shape) != (atom_hidden, base_dim + n_extra):
        raise ValueError(f"augmented atom weight shape {tuple(own_w.shape)} != {(atom_hidden, base_dim + n_extra)}")
    new_w = own_w.clone()
    new_w[:, :base_dim] = base_w
    new_w[:, base_dim:] = 0.0
    out[ATOM_WEIGHT_KEY] = new_w
    mismatched = [k for k in shared if not torch.equal(out[k], base_state[k])]
    audit = {
        "base_input_width": int(base_dim),
        "augmented_input_width": int(base_dim + n_extra),
        "atom_hidden": int(atom_hidden),
        "n_shared_tensors": len(shared),
        "shared_tensors": shared,
        "all_shared_bitwise_equal": not mismatched,
        "mismatched": mismatched,
        "base_51_columns_bitwise_equal": bool(torch.equal(new_w[:, :base_dim], base_w)),
        "new_columns_zero_count": int(torch.count_nonzero(new_w[:, base_dim:])),
        "new_columns_initialised_to_zero": bool(torch.count_nonzero(new_w[:, base_dim:]) == 0),
        "new_only_keys": sorted(set(out) - set(base_state)),
        "base_only_keys": sorted(set(base_state) - set(out)),
    }
    if mismatched or not audit["base_51_columns_bitwise_equal"] or not audit["new_columns_initialised_to_zero"]:
        raise RuntimeError(f"atom-input migration failed: {audit}")
    return out, audit


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------


def build_atomchem_model(arm: str, spec, norm, seed: int, device="cpu", *, dropout: float = 0.0):
    if arm not in C.ARMS:
        raise ValueError(f"unknown v4 arm {arm!r}")
    if int(spec.atom_feat_dim) != C.ATOM_DIM_AUGMENTED:
        raise ValueError(f"augmented spec atom width {spec.atom_feat_dim} != {C.ATOM_DIM_AUGMENTED}")
    if int(spec.global_dim) != C.GLOBAL_DIM_BASE:
        raise ValueError("global width must stay 42; v4 only widens the atom input")
    base_spec = C.load_frozen_spec()

    torch.manual_seed(int(seed))
    base_model = C.prep_model.LCRC(base_spec, norm, C.prep_model.DEFAULT_CONFIG)
    base_state = {k: v.detach().clone() for k, v in base_model.state_dict().items()}

    torch.manual_seed(int(seed))
    model = C.prep_model.LCRC(spec, norm, C.prep_model.DEFAULT_CONFIG)
    own_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    migrated, audit = migrate_atom_input(base_state, own_state)
    model.load_state_dict(migrated, strict=True)
    model.to(device)
    hook = C.M.DropoutAtLocal(model, "A", float(dropout))

    params_base = C.M.parameter_count(base_model)
    params_model = C.M.parameter_count(model)
    delta = params_model - params_base
    meta = {
        "arm": arm,
        "feature_variant": C.ARM_VARIANT[arm],
        "dropout": float(dropout),
        "dropout_hook": hook,
        "migration": audit,
        "parameters_original": int(params_base),
        "parameters_augmented": int(params_model),
        "parameter_delta": int(delta),
        "expected_parameter_delta": int(C.EXPECTED_PARAMETER_DELTA),
        "base_state_hash": C.M.model_state_hash(base_model),
        "augmented_state_hash": C.M.model_state_hash(model),
        "shared_with_original": base_model,
        "message_passing_changed": False,
        "s2_semantics_changed": False,
        "scalar9_global42_changed": False,
    }
    if delta != C.EXPECTED_PARAMETER_DELTA:
        raise RuntimeError(f"unexpected parameter delta {delta} != {C.EXPECTED_PARAMETER_DELTA}")
    return model, meta


@contextmanager
def patched_model_builder():
    original = C.M.build_model

    def dispatch(arm, spec, norm, seed, device="cpu", *, dropout: float = 0.0):
        if arm in C.ARMS:
            return build_atomchem_model(arm, spec, norm, seed, device, dropout=dropout)
        return original(arm, spec, norm, seed, device, dropout=dropout)

    C.M.build_model = dispatch
    try:
        yield
    finally:
        C.M.build_model = original


# ---------------------------------------------------------------------------
# atom-extra collate
# ---------------------------------------------------------------------------


def resolve_atom_extra(cache: Mapping[str, Any], extras: Mapping[str, Any]) -> tuple[np.ndarray, str]:
    """Return the atom-extra matrix in this cache's atom order.

    The train/dev caches are frozen v2 fold views; the smoke cache is a prefix
    of the train cache.  Any other cache is rejected.
    """
    ids = tuple(str(x) for x in cache["molecule_ids"])
    n_atoms = int(np.asarray(cache["atom_ptr"], dtype=np.int64)[-1])
    for split in ("train", "dev"):
        expected = tuple(str(x) for x in extras[f"{split}_molecule_ids"])
        arr = np.asarray(extras[split], dtype=np.float32)
        if ids == expected:
            if arr.shape[0] != n_atoms:
                raise ValueError(f"{split} extras atom count {arr.shape[0]} != cache atom count {n_atoms}")
            return arr, split
        if 0 < len(ids) < len(expected) and ids == expected[: len(ids)]:
            if n_atoms > arr.shape[0]:
                raise ValueError(f"{split} prefix extras are shorter than the cache")
            return arr[:n_atoms], f"{split}_prefix"
    raise ValueError(
        "unrecognised cache for the v4 atom extras: molecule ids match neither the fold train/dev caches "
        "nor a train prefix; refusing to guess the atom correspondence"
    )


#: captured before any monkey-patch so the base collate can never dispatch to itself
_BASE_COLLATE = C.D.collate


def collate_with_atom_extra(cache: Mapping[str, Any], rows, extras: Mapping[str, Any]):
    batch = _BASE_COLLATE(cache, rows)
    arr, tag = resolve_atom_extra(cache, extras)
    atom_ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)
    row_list = [int(r) for r in rows]
    atom_idx = np.concatenate([np.arange(int(atom_ptr[r]), int(atom_ptr[r + 1]), dtype=np.int64) for r in row_list])
    add = torch.from_numpy(np.ascontiguousarray(arr[atom_idx], dtype=np.float32))
    if int(add.shape[0]) != int(batch["atom_feat"].shape[0]):
        raise RuntimeError("atom-extra rows do not match the collated atom rows")
    cat = torch.cat([batch["atom_feat"], add], dim=1)
    if int(cat.shape[1]) != C.ATOM_DIM_AUGMENTED:
        raise RuntimeError(f"augmented atom matrix has width {cat.shape[1]}")
    batch["atom_feat"] = cat
    batch["_v4_atom_extra_tag"] = tag
    return batch


@contextmanager
def patched_collate(extras: Mapping[str, Any]):
    original = C.D.collate  # restore whatever was installed on entry (nesting safe)

    def dispatch(cache, rows):
        return collate_with_atom_extra(cache, rows, extras)

    C.D.collate = dispatch
    try:
        yield
    finally:
        C.D.collate = original


# ---------------------------------------------------------------------------
# training job
# ---------------------------------------------------------------------------


@dataclass
class LocalChemTrainJob(C.T.TrainJob):
    atom_extras: Mapping[str, Any] | None = None
    v4_audit: dict[str, Any] | None = None

    def validate(self) -> None:
        if self.arm not in C.ARMS:
            raise ValueError(f"arm must be one of {C.ARMS}")
        if self.recipe != C.RECIPE:
            raise ValueError(f"v4 pilot only runs recipe {C.RECIPE}")
        if self.fold != C.FOLD:
            raise ValueError(f"v4 pilot only runs Fold {C.FOLD}")
        if self.seed != C.SEED:
            raise ValueError(f"v4 pilot only runs seed {C.SEED}")
        if self.variant != "scratch":
            raise ValueError("v4 pilot only runs scratch models (no SSL)")
        if self.device not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        if self.atom_extras is None:
            raise ValueError("v4 arms require the frozen atom extras")
        if self.spec_override is None:
            raise ValueError("v4 arms require the augmented spec")
        if int(self.spec_override.atom_feat_dim) != C.ATOM_DIM_AUGMENTED:
            raise ValueError("spec atom width is not 55")
        if self.norm_override is not None:
            raise ValueError("v4 keeps the frozen fold-train normalisation; norm_override must be None")
        if self.global_extra_train is not None or self.global_extra_dev is not None:
            raise ValueError("v4 does not touch the global channel")


def train_local_chem_job(job: LocalChemTrainJob) -> dict[str, Any]:
    """Run v2's training loop with the v4 builder and atom-extra collate only."""
    job.validate()
    with patched_model_builder(), patched_collate(job.atom_extras):
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
        "model_builder_dispatch": "v4_models.build_atomchem_model",
        "collate_dispatch": "v4_models.collate_with_atom_extra (appends 4 atom columns)",
    }
    base_spec = C.load_frozen_spec()
    base_model = C.prep_model.LCRC(base_spec, C.load_frozen_norm(), C.prep_model.DEFAULT_CONFIG)
    base_params = int(C.M.parameter_count(base_model))
    payload["v4"] = {
        "protocol_id": C.PROTOCOL_ID,
        "arm": job.arm,
        "feature_variant": C.ARM_VARIANT[job.arm],
        "atom_feature_names": list(C.ATOMCHEM_NAMES),
        "fold": int(job.fold),
        "seed": int(job.seed),
        "recipe": job.recipe,
        "audit": job.v4_audit,
        "atom_input_width": int(job.spec_override.atom_feat_dim),
        "atom_input_width_base": C.ATOM_DIM_BASE,
        "original_parameters_reference": base_params,
        "parameters_nominal": int(payload["model"]["parameters_nominal"]),
        "new_parameters": int(payload["model"]["parameters_nominal"]) - base_params,
        "expected_new_parameters": int(C.EXPECTED_PARAMETER_DELTA),
        "message_passing_changed": False,
        "s2_semantics_changed": False,
        "bond13_scalar9_global42_changed": False,
        "iterative_message_passing": False,
        "training_loop_changed": False,
        "labels_used_for_features": False,
    }
    if payload["v4"]["new_parameters"] != C.EXPECTED_PARAMETER_DELTA:
        raise RuntimeError(
            f"unexpected trained parameter delta {payload['v4']['new_parameters']} != {C.EXPECTED_PARAMETER_DELTA}"
        )
    C.write_json(out_dir / "train.json", payload)
    return payload


__all__ = [
    "ATOM_WEIGHT_KEY",
    "migrate_atom_input",
    "build_atomchem_model",
    "patched_model_builder",
    "resolve_atom_extra",
    "collate_with_atom_extra",
    "patched_collate",
    "LocalChemTrainJob",
    "train_local_chem_job",
]
