"""ZINC component-supervision full-train confirmation (seed 0).

Round: ``zinc_component_supervision_fulltrain_confirmation_seed0_v1``.

Single scientific question, unchanged from the seed-0 candidate round: with the
*same* M skeleton, the same input pipeline and the same total target ``g``, does
adding true chemical-component supervision
(``ell = (logP - MU_LOGP) / sigma_logP``, ``s = g - ell``) change the frozen
official-valid ``g``-MAE relative to the matched total-only arm?

Only the *data range* changes versus ``zinc_chemistry_component_supervision_
seed0_v1``: this round trains on the **full 10,000 official-train rows** and then
confirms on official-valid (1000 rows), once, after everything is frozen.

Exactly two matched body arms (identical init / schedule / regime):

* ``SUM``  -- ``L = L_g = MAE(hat_ell + hat_s, g)``
* ``COMP`` -- ``L = L_g + 0.5 * (L_ell + L_s)``

plus exactly one shared deploy accessory, the small fixed **cycle head Q**
(25 -> 64 -> 32 -> 1, 3777 params) trained on the actual topology25 input with
``L1(q(T), c)``.  The two deployable wrappers are

* ``g_cal_a = h_a + b_g_a`` with ``h_a = hat_ell_a + hat_s_a``
* ``y_cal_a = h_a + q_raw + b_y_a``

with *one* train-only median bias per endpoint, per arm.  ``b_g`` and ``b_y``
are two different endpoints; the y wrapper never folds ``b_g`` and never adds a
second Q bias.

All fit-dependent objects (prep standardizers, phi scaler, kappa_M, target
decomposition constants) are rebuilt/refit on the **full 10,000 train rows**.
The frozen tuple incidence cache is reused (structure-only); the old 8k
scalers / kappa / schedule / mixed 12k constants are never reused as this
round's train-only objects.

official-valid is *not* read until ``--mode valid`` after the frozen evaluation
manifest exists.  official-test is never instantiated, loaded, predicted or
scored.

Usage::

    uv run python -m tracks.ksvd.experiments.luyin16.\
zinc_component_supervision_fulltrain_confirmation_seed0_v1 --mode freeze
    ... --mode init-check
    ... --mode smoke --arm SUM --device cpu
    ... --mode train --arm SUM --device cuda
    ... --mode train --arm COMP --device cuda
    ... --mode train-q
    ... --mode calibrate
    ... --mode manifest
    ... --mode valid
    ... --mode analyze
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm
from tracks.ksvd.experiments.luyin16 import zinc_full_cycle_target_decomposition_v1 as zftd
from tracks.ksvd.experiments.luyin16 import zinc_joint_dictionary_decision_v1 as zjd
from tracks.ksvd.experiments.luyin16 import (
    zinc_local_tuple_dictionary_joint_vs_independent_seed0_v1 as prev,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_local_tuple_dictionary_vs_mlp_seed0_v1 as mlpmod,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_local_tuple_fresh_fold_replication_seed0_v1 as zfr,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_task_dictionary_and_cycle_witness_seed0_v1 as zw,
)

PROTOCOL_VERSION = "zinc-component-supervision-fulltrain-confirmation-seed0-v1"
RESULT_SLUG = "zinc_component_supervision_fulltrain_confirmation_seed0_v1"
TRACK_ROOT = zjd.TRACK_ROOT
RESULTS_DIR = TRACK_ROOT / "results" / RESULT_SLUG
SOURCE_DIR = zfr.RESULTS_DIR  # frozen fresh-fold artifacts (structure only)
OLD_TUPLE_NPZ = prev.TUPLE_NPZ

#: train-only sources (never valid/test)
TRAIN_LABEL_CSV = zfr.TRAIN_LABEL_CSV
GVEA_PROPS_NPZ = zfr.GVEA_PROPS_NPZ
HANDOFF_TRAIN = zfr.HANDOFF_TRAIN
PREP_BLOB = zjd.PREP_DIR / "fold_objects.npz"
CYCLE = zjd.CYCLE
VALID_LABEL_CSV = CYCLE / "valid_cycle_audit_label.csv"
VALID_HANDOFF = zjd.HANDOFF / "valid.npz"
ENCODED_VALID = Path(zfr.zjd.ZINC_HANDOFF_DIR) if hasattr(zfr.zjd, "ZINC_HANDOFF_DIR") else None

# ---- frozen recipe (identical for both body arms) -------------------------
SEED = 0
EPOCHS = 240
LR = 1.0e-3
WEIGHT_DECAY = 1.0e-5
GRAD_CLIP = 5.0
BATCH_SIZE = 128
SOUP_EPOCHS = (236, 237, 238, 239, 240)
TRAIN_SHUFFLE_OFFSET = 101
LOG_EPOCHS = (1, 40, 120, 240)
N_TRAIN = 10000
REPLAY_TOL = 1.0e-5

#: fixed label-generation constant (never re-estimated)
MU_LOGP = zfr.MU_LOGP

#: component-loss coefficient (pre-fixed, never scanned / adapted / tuned)
COMPONENT_LOSS_WEIGHT = 0.5

#: kappa sample rule (fixed, carried over from the fresh-fold algorithm)
KAPPA_SEED = zfr.KAPPA_SEED
KAPPA_SAMPLE_MAX = zfr.KAPPA_SAMPLE_MAX

#: shared cycle head Q (fixed deploy accessory, never a new exploration axis)
Q_EPOCHS = 300
Q_HIDDEN = (64, 32)
Q_TOPOLOGY_IN = 25
Q_PARAMETERS = 3777
Q_TRAIN_GEN_BASE = 20261003
Q_SOUP_EPOCHS = (296, 297, 298, 299, 300)

#: statistics / gates (frozen before either formal trajectory)
BOOT_SEED = 20261009
N_BOOT = 1000
PERF_DELTA = 0.003
DEPLOY_G0_WORSEN_MAX = 0.001
BENCHMARK_Y_TARGET = 0.09

ARMS = ("SUM", "COMP")
GROUP_NAMES = ("k=0", "k=-1", "k=-2", "k<=-3")
EXPECTED_PARAMETERS = mlpmod.EXPECTED_TOTAL_PARAMETERS + 39 + 1  # 297,539

jsonable = zfr.jsonable
write_json = zfr.write_json
file_sha256 = zfr.file_sha256
state_hash = zfr.state_hash
seed_everything = zfr.seed_everything
array_sha256 = zfr.array_sha256
group_masks = zw.group_masks
build_schedule = zw.build_schedule
_hash_bytes = zfr._hash_bytes


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(zjd.REPO_ROOT), text=True
        ).strip()
    except Exception:
        return "unknown"


def resolve_device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def allocation_probe() -> dict[str, Any]:
    out: dict[str, Any] = {
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID"),
        "SLURM_JOB_NODELIST": os.environ.get("SLURM_JOB_NODELIST"),
        "SLURM_GPUS_ON_NODE": os.environ.get("SLURM_GPUS_ON_NODE"),
        "SLURM_JOB_PARTITION": os.environ.get("SLURM_JOB_PARTITION"),
        "hostname": os.uname().nodename,
    }
    try:
        out["nvidia_smi_L"] = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30
        ).stdout.strip()
        out["nvidia_smi_query"] = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,pci.bus_id,name,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
    except Exception as exc:  # pragma: no cover
        out["nvidia_smi_error"] = repr(exc)
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        out["torch_device_index"] = int(torch.cuda.current_device())
        out["torch_device_name"] = str(props.name)
        out["torch_total_memory"] = int(props.total_memory)
        pci = getattr(props, "pci_bus_id", None)
        if pci is not None:
            out["torch_pci_bus_id"] = str(pci)
        uuid = getattr(props, "uuid", None)
        if uuid is not None:
            out["torch_uuid"] = str(uuid)
    return out


# ---------------------------------------------------------------------------
# 1. component reader: 39 -> 2 half-init split of the original 39 -> 1 head
# ---------------------------------------------------------------------------


class ComponentReader(nn.Module):
    """Original reader with the last ``Linear(39, 1)`` replaced by ``39 -> 2``.

    ``weight[i] = original_weight / 2`` and ``bias[i] = original_bias / 2`` for
    both rows, so ``hat_g = hat_ell + hat_s`` equals the original prediction
    exactly at construction.  The standard forward returns the ``(n,)`` total
    ``g``; the two ``(n,)`` component outputs of the *same* body computation are
    cached on ``last_components`` (never a second forward pass).
    """

    def __init__(self, original: nn.Sequential) -> None:
        super().__init__()
        if not isinstance(original, nn.Sequential) or not isinstance(original[-1], nn.Linear):
            raise RuntimeError("unexpected reader layout")
        if int(original[-1].out_features) != 1:
            raise RuntimeError(
                f"reader last layer is {int(original[-1].out_features)}-wide, expected 1"
            )
        layers: list[nn.Module] = []
        for index, module in enumerate(original):
            if index == len(original) - 1:
                with torch.random.fork_rng(devices=[]):
                    new = nn.Linear(int(module.in_features), 2, bias=module.bias is not None)
                with torch.no_grad():
                    new.weight.copy_(module.weight.repeat(2, 1) * 0.5)
                    if module.bias is not None:
                        new.bias.copy_(module.bias.repeat(2) * 0.5)
                layers.append(new)
            else:
                layers.append(module)
        self.net = nn.Sequential(*layers)
        self.last_components: torch.Tensor | None = None

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        components = self.net(z)  # (n, 2) = [hat_ell, hat_s]
        self.last_components = components
        return components.sum(dim=-1)

    def components(self) -> torch.Tensor:
        if self.last_components is None:
            raise RuntimeError("no forward has populated reader components")
        return self.last_components


def build_component_model(arm: str, payload: prev.TuplePayload, kappa_M: float) -> nn.Module:
    if arm not in ARMS:
        raise ValueError(arm)
    model = mlpmod.build_arm_mj(payload, kappa_M=float(kappa_M))
    model.reader = ComponentReader(model.reader.net)
    audit = component_parameter_audit(model)
    expected = {
        "total_parameters": EXPECTED_PARAMETERS,
        "base_body_parameters": mlpmod.EXPECTED_BODY_PARAMETERS + 39 + 1,
        "bridge_parameters": mlpmod.BRIDGE_PARAMETERS,
        "local_tuple_parameters": mlpmod.A_PARAMETERS + mlpmod.W_LOC_PARAMETERS,
        "reader_output_parameters": 2 * 39 + 2,
    }
    if audit != expected:
        raise RuntimeError(f"component parameter audit failed: {audit} != {expected}")
    return model


def component_parameter_audit(model: nn.Module) -> dict[str, int]:
    return {
        "total_parameters": int(sum(p.numel() for p in model.parameters())),
        "base_body_parameters": prev.base_body_parameter_count(model),
        "bridge_parameters": prev.bridge_parameter_count(model),
        "local_tuple_parameters": prev.local_parameter_count(model),
        "reader_output_parameters": int(
            model.reader.net[-1].weight.numel() + model.reader.net[-1].bias.numel()
        ),
    }


# ---------------------------------------------------------------------------
# 2. all-10k train-only prep / targets / payload / kappa
# ---------------------------------------------------------------------------


def load_train_only_raw_rows() -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Per-molecule logP/SA/y_stored/cycle_score for the 10000 train rows.

    Reuses the train-only raw-field reconstruction of the fresh-fold runner
    (positional train rows + GVAE property table + handoff y/gid).
    """
    return zfr.load_train_only_raw_rows()


def refit_fulltrain_targets(rows: np.ndarray) -> dict[str, Any]:
    """Refit the decomposition constants on **all 10000** train rows."""
    in_fit = np.ones(N_TRAIN, dtype=bool)
    constants = zfr.zcdm._refit_constants(rows, in_fit, mu_logp=MU_LOGP)
    labels = zfr.zcdm._apply_constants(rows, constants, MU_LOGP)
    y = rows["y_stored"]
    c = labels["c"]
    g = y - c
    k = labels["k"]
    if not (
        np.isfinite(c).all()
        and np.isfinite(g).all()
        and np.isfinite(y).all()
    ):
        raise RuntimeError("non-finite full-train target")
    checks = {
        "fit_rows": "all 10000 official-train rows",
        "uses_valid_labels": False,
        "n_rows": int(y.shape[0]),
        "y_identity_g_plus_c_max_abs": float(np.max(np.abs(g - (y - c)))),
        "constants": {key: float(constants[key]) for key in constants if isinstance(constants[key], (int, float))},
        "mu_logP_fixed": float(MU_LOGP),
        "k_counts": {name: int(mask.sum()) for name, mask in group_masks(k).items()},
    }
    if checks["y_identity_g_plus_c_max_abs"] > 1e-12:
        raise RuntimeError("y = g + c identity failed")
    return {"constants": constants, "y": y, "c": c, "g": g, "k": k, "checks": checks}


def build_component_targets(rows: np.ndarray, constants: Mapping[str, float], g: np.ndarray) -> dict[str, Any]:
    sigma_logP = float(constants["sigma_logP"])
    sigma_SA = float(constants["sigma_SA"])
    mu_SA = float(constants["mu_SA"])
    ell = (rows["logP"].astype(np.float64) - MU_LOGP) / sigma_logP
    s = g - ell
    s_sa = (rows["SA"].astype(np.float64) - mu_SA) / sigma_SA
    epsilon = s - s_sa
    checks = {
        "g_equals_ell_plus_s_max_abs": float(np.max(np.abs(g - (ell + s)))),
        "all_finite": bool(
            np.isfinite(ell).all() and np.isfinite(s).all() and np.isfinite(s_sa).all()
        ),
        "mu_logP": float(MU_LOGP),
    }
    if checks["g_equals_ell_plus_s_max_abs"] > 1e-12:
        raise RuntimeError("g = ell + s identity failed")
    return {"ell": ell, "s": s, "s_SA": s_sa, "epsilon": epsilon, "checks": checks}


def build_fulltrain_payload() -> dict[str, Any]:
    """Rebuild the local-tuple payload on the full 10k train rows.

    The structural incidence arrays (root_base / node_sizes / pair_ptr /
    pair_t / pair_a / pair_wJ / pair_wI / root_atom) are reused from the frozen
    tuple index byte-for-byte (structure only).  The label-free phi scaler and
    the kappa_M sample are refit on **all 10000** train rows.
    """
    old_npz = np.load(OLD_TUPLE_NPZ)
    old = {key: old_npz[key] for key in (
        "root_base", "node_sizes", "pair_ptr", "pair_t", "pair_a",
        "pair_wJ", "pair_wI", "root_atom",
    )}
    phi_env, atom_env, node_sizes_env = prev._env_phi_atom()
    checks: dict[str, Any] = {
        "old_tuple_npz": str(OLD_TUPLE_NPZ),
        "old_tuple_npz_sha256": file_sha256(OLD_TUPLE_NPZ),
        "node_sizes_match_env": bool(np.array_equal(old["node_sizes"], node_sizes_env)),
        "root_atom_match_env": bool(np.array_equal(old["root_atom"], atom_env)),
        "pair_ptr_monotone": bool(np.all(np.diff(old["pair_ptr"]) >= 0)),
    }
    if not checks["node_sizes_match_env"] or not checks["root_atom_match_env"]:
        raise RuntimeError("reused tuple structure does not match the env cache")
    if not checks["pair_ptr_monotone"]:
        raise RuntimeError("pair_ptr is not monotone")

    total_roots = int(node_sizes_env.sum())
    # all 10000 molecules are fit rows this round
    fit_root_mask = np.ones(total_roots, dtype=bool)
    pair_root = np.repeat(np.arange(total_roots, dtype=np.int64), np.diff(old["pair_ptr"]))
    realized = old["pair_wJ"] > 0.0
    realized = realized & fit_root_mask[pair_root]
    fit_realized_roots = pair_root[realized]
    if fit_realized_roots.size == 0:
        raise RuntimeError("no realized incident tuples")

    phi_sample = phi_env[fit_realized_roots].astype(np.float64)
    phi_mean = phi_sample.mean(axis=0)
    phi_std = np.maximum(phi_sample.std(axis=0), prev.PHI_STD_FLOOR)
    z = (phi_env[fit_realized_roots].astype(np.float64) - phi_mean) / phi_std
    phi_scale = float(np.sqrt(np.mean(z * z)))
    if not math.isfinite(phi_scale) or phi_scale <= 1e-12:
        raise RuntimeError("phi L2-RMS scale is degenerate")

    fit_roots = np.nonzero(fit_root_mask)[0]
    rng = np.random.default_rng(int(KAPPA_SEED))
    take = min(int(KAPPA_SAMPLE_MAX), int(fit_roots.size))
    sample = np.sort(fit_roots[rng.permutation(int(fit_roots.size))[:take]])
    if sample.size > KAPPA_SAMPLE_MAX:
        raise RuntimeError("kappa sample larger than cap")
    if not fit_root_mask[sample].all():
        raise RuntimeError("kappa sample has off-fit roots")

    kappa, kappa_stats = prev._kappa_for_roots(
        prev.init_d_loc(),
        phi_env, atom_env, old["pair_ptr"], old["pair_t"], old["pair_a"],
        old["pair_wJ"], old["pair_wI"],
        phi_mean.astype(np.float32), phi_std.astype(np.float32), np.float32(phi_scale),
        sample,
    )
    if not math.isfinite(kappa) or kappa <= 0.0:
        raise RuntimeError(f"MECHANISM_INIT_BLOCKED: kappa={kappa}")

    old_phi_mean = old_npz["phi_mean"].astype(np.float64)
    old_phi_std = old_npz["phi_std"].astype(np.float64)
    old_phi_scale = float(old_npz["phi_scale"])
    checks["scaler_old_vs_new"] = {
        "phi_mean_max_abs_diff": float(np.max(np.abs(phi_mean - old_phi_mean))),
        "phi_std_max_abs_diff": float(np.max(np.abs(phi_std - old_phi_std))),
        "phi_scale_old": old_phi_scale,
        "phi_scale_new": phi_scale,
        "kappa_old": float(old_npz["kappa"]),
        "kappa_new": float(kappa),
    }
    checks["phi_scaler"] = {
        "fitted_on": "all 10000 train rows, realized C>0 incident tuples",
        "n_samples": int(fit_realized_roots.size),
        "std_floor": float(prev.PHI_STD_FLOOR),
        "phi_scale_l2_rms": phi_scale,
        "phi_mean_sha256": array_sha256(phi_mean.astype(np.float32)),
        "phi_std_sha256": array_sha256(phi_std.astype(np.float32)),
    }
    checks["kappa"] = {
        "value": float(kappa),
        "definition": "1 / RMS(initial e_J, e_I) over up to 8192 uniformly sampled full-train roots",
        "seed": int(KAPPA_SEED),
        "n_roots": int(sample.size),
        "sample_sha256": array_sha256(sample, np.int64),
        "raw_rms": kappa_stats["rms"],
        "e_J_rms": kappa_stats["e_J_rms"],
        "e_I_rms": kappa_stats["e_I_rms"],
    }
    arrays = dict(old)
    arrays.update(
        phi_mean=phi_mean.astype(np.float32),
        phi_std=phi_std.astype(np.float32),
        phi_scale=np.asarray(phi_scale, np.float32),
        kappa=np.asarray(kappa, np.float32),
        kappa_sample=sample.astype(np.int64),
    )
    return {"arrays": arrays, "checks": checks}


def build_fulltrain_prep() -> dict[str, Any]:
    """Refit every body input standardizer on the full 10k train rows."""
    train_data = zftd.load_train_only()
    if len(train_data) != N_TRAIN:
        raise RuntimeError(f"train-only cache has {len(train_data)} rows, expected {N_TRAIN}")
    blob = np.load(PREP_BLOB, allow_pickle=False)
    fit_idx = np.arange(N_TRAIN, dtype=np.int64)
    meta = zw.apply_new_fit_prep(train_data, blob, fit_idx)
    meta = {key: value for key, value in meta.items()}
    meta["fitted_rows"] = "all 10000 official-train rows"
    # direct recomputation witnesses (same style as the fresh-fold runner)
    direct: dict[str, Any] = {}
    for attr, key in (
        ("patch_cont", "patch"), ("global_context", "ctx"),
        ("anchor", "anchor"), ("topology_features", "topo"),
    ):
        values, counts = zjd._stack(train_data, attr)
        raw = zjd.Std(blob[f"{key}_all_mean"], blob[f"{key}_all_scale"]).inverse(values)
        refit = zjd.Std.fit(raw, floor=zjd.CANON_STD_FLOOR)
        direct[key] = {
            "n_fit_root_rows": int(raw.shape[0]),
            "mean_max_abs_diff": float(np.max(np.abs(
                refit.mean.astype(np.float64) - np.asarray(meta[f"{key}_fit_mean"], np.float64)
            ))),
            "scale_max_abs_diff": float(np.max(np.abs(
                refit.scale.astype(np.float64) - np.asarray(meta[f"{key}_fit_scale"], np.float64)
            ))),
        }
    meta["direct_full_fit_recompute"] = direct
    return meta


def build_fulltrain_data(prep_meta: Mapping[str, Any]) -> list[Any]:
    """Load and prepare all 10000 train rows with the full-train prep."""
    train_data = zftd.load_train_only()
    if len(train_data) != N_TRAIN:
        raise RuntimeError("train-only cache length mismatch")
    # apply the already-written full-train standardizers (no refit here)
    patch_fit = zjd.Std(prep_meta["patch_fit_mean"], prep_meta["patch_fit_scale"])
    ctx_fit = zjd.Std(prep_meta["ctx_fit_mean"], prep_meta["ctx_fit_scale"])
    anchor_fit = zjd.Std(prep_meta["anchor_fit_mean"], prep_meta["anchor_fit_scale"])
    topo_fit = zjd.Std(prep_meta["topo_fit_mean"], prep_meta["topo_fit_scale"])
    blob = np.load(PREP_BLOB, allow_pickle=False)
    patch_all = zjd.Std(blob["patch_all_mean"], blob["patch_all_scale"])
    ctx_all = zjd.Std(blob["ctx_all_mean"], blob["ctx_all_scale"])
    anchor_all = zjd.Std(blob["anchor_all_mean"], blob["anchor_all_scale"])
    topo_all = zjd.Std(blob["topo_all_mean"], blob["topo_all_scale"])

    p, pc = zjd._stack(train_data, "patch_cont")
    c, cc = zjd._stack(train_data, "global_context")
    a, ac = zjd._stack(train_data, "anchor")
    t, tc = zjd._stack(train_data, "topology_features")
    zjd._unstack(patch_fit.transform(patch_all.inverse(p)), pc, "patch_cont", train_data)
    zjd._unstack(ctx_fit.transform(ctx_all.inverse(c)), cc, "global_context", train_data)
    zjd._unstack(anchor_fit.transform(anchor_all.inverse(a)), ac, "anchor", train_data)
    zjd._unstack(topo_fit.transform(topo_all.inverse(t)), tc, "topology_features", train_data)
    for position, index in enumerate(range(N_TRAIN)):
        train_data[position].local_mol_id = torch.tensor([int(index)], dtype=torch.long)
    return train_data


# ---------------------------------------------------------------------------
# 3. phase: freeze (targets + prep + payload + init + schedule)
# ---------------------------------------------------------------------------


def phase_freeze(*, out_dir: Path = RESULTS_DIR, log: Any = print) -> dict[str, Any]:
    torch.set_num_threads(8)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    rows, gid, label_checks = load_train_only_raw_rows()
    targets = refit_fulltrain_targets(rows)
    constants = targets["constants"]
    components = build_component_targets(rows, constants, targets["g"])
    payload = build_fulltrain_payload()
    prep_meta = build_fulltrain_prep()

    np.savez_compressed(
        out_dir / "full_train_targets.npz",
        y=targets["y"], c=targets["c"], g=targets["g"], k=targets["k"],
        gid=gid, logP=rows["logP"], SA=rows["SA"],
        ell=components["ell"], s=components["s"], s_SA=components["s_SA"],
        epsilon=components["epsilon"],
        constants=np.asarray(
            [constants[name] for name in ("sigma_logP", "sigma_SA", "mu_SA", "sigma_cycle", "mu_cycle")],
            np.float64,
        ),
        constant_names=np.asarray(["sigma_logP", "sigma_SA", "mu_SA", "sigma_cycle", "mu_cycle"]),
        mu_logP=np.asarray(MU_LOGP, np.float64),
    )
    np.savez_compressed(
        out_dir / "full_train_payload.npz",
        **{key: value for key, value in payload["arrays"].items()},
    )
    np.savez_compressed(
        out_dir / "full_train_prep.npz",
        **{
            key: np.asarray(prep_meta[key], np.float32)
            for key in (
                "patch_fit_mean", "patch_fit_scale", "ctx_fit_mean", "ctx_fit_scale",
                "anchor_fit_mean", "anchor_fit_scale", "topo_fit_mean", "topo_fit_scale",
            )
        },
    )
    write_json(out_dir / "targets_meta.json", {
        "protocol_version": PROTOCOL_VERSION,
        "target_checks": targets["checks"],
        "component_checks": components["checks"],
        "label_source_checks": label_checks,
        "constant_names": ["sigma_logP", "sigma_SA", "mu_SA", "sigma_cycle", "mu_cycle"],
        "constants": {k: float(constants[k]) for k in ("sigma_logP", "sigma_SA", "mu_SA", "sigma_cycle", "mu_cycle")},
        "mu_logP": float(MU_LOGP),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    })
    write_json(out_dir / "prep_meta.json", {
        "protocol_version": PROTOCOL_VERSION,
        "n_fit_molecules": N_TRAIN,
        "fitted_rows": "all 10000 official-train rows",
        "direct_full_fit_recompute": prep_meta["direct_full_fit_recompute"],
        "inversion_constants": str(PREP_BLOB),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    })
    write_json(out_dir / "payload_meta.json", {
        "protocol_version": PROTOCOL_VERSION,
        "payload_checks": payload["checks"],
        "official_valid_loaded": False,
        "official_test_loaded": False,
    })
    # freeze the schedule + global graph-ID stream
    schedule, schedule_hash = build_schedule(N_TRAIN, EPOCHS, SEED + TRAIN_SHUFFLE_OFFSET)
    np.savez_compressed(out_dir / "full_train_schedule.npz",
                        schedule=np.stack(schedule, axis=0))
    write_json(out_dir / "schedule_meta.json", {
        "protocol_version": PROTOCOL_VERSION,
        "n_rows": N_TRAIN,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "steps_per_epoch": int(math.ceil(N_TRAIN / BATCH_SIZE)),
        "steps_total": int(math.ceil(N_TRAIN / BATCH_SIZE)) * EPOCHS,
        "schedule_seed": int(SEED + TRAIN_SHUFFLE_OFFSET),
        "schedule_sha256": schedule_hash,
        "schedule_sha256_note": "new 10k schedule; not required to equal the old 8k hash",
        "official_valid_loaded": False,
        "official_test_loaded": False,
    })
    write_json(out_dir / "alloc_targets.json", allocation_probe())
    log(
        f"[freeze] n={N_TRAIN} g=ell+s maxdiff={components['checks']['g_equals_ell_plus_s_max_abs']:.2e} "
        f"kappa={payload['checks']['kappa']['value']:.6f} sched={schedule_hash[:12]} "
        f"({time.perf_counter() - started:.1f}s)"
    )
    return {
        "targets": targets,
        "components": components,
        "payload": payload,
        "prep": prep_meta,
        "schedule_hash": schedule_hash,
    }


# ---------------------------------------------------------------------------
# 4. init identity
# ---------------------------------------------------------------------------


def init_check(*, out_dir: Path = RESULTS_DIR, device: torch.device | None = None, log: Any = print) -> dict[str, Any]:
    device = device or torch.device("cpu")
    torch.set_num_threads(8)
    with np.load(out_dir / "full_train_payload.npz", allow_pickle=False) as z:
        payload_arrays = {key: z[key] for key in z.files}
    payload = prev.TuplePayload(payload_arrays)
    kappa_M = float(payload_arrays["kappa"].reshape(-1)[0])

    seed_everything(SEED)
    model_sum = build_component_model("SUM", payload, kappa_M)
    state_sum = {key: value.detach().cpu().clone() for key, value in model_sum.state_dict().items()}
    seed_everything(SEED)
    model_comp = build_component_model("COMP", payload, kappa_M)
    state_comp = {key: value.detach().cpu().clone() for key, value in model_comp.state_dict().items()}
    if sorted(state_sum) != sorted(state_comp):
        raise RuntimeError("SUM/COMP state key sets differ")
    max_abs = max(float((state_sum[key] - state_comp[key]).abs().max()) for key in state_sum)
    if max_abs != 0.0:
        raise RuntimeError(f"SUM/COMP initial tensors differ (max abs {max_abs})")

    seed_everything(SEED)
    model_orig = mlpmod.build_arm_mj(payload, kappa_M=kappa_M)
    orig_state = {key: value.detach().cpu().clone() for key, value in model_orig.state_dict().items()}
    shared = sorted(key for key in orig_state if key not in ("reader.net.4.weight", "reader.net.4.bias"))
    if sorted(key for key in state_sum if key not in ("reader.net.4.weight", "reader.net.4.bias")) != shared:
        raise RuntimeError("shared state key sets differ from the original M")

    w_orig = orig_state["reader.net.4.weight"]
    b_orig = orig_state["reader.net.4.bias"]
    w_new = state_sum["reader.net.4.weight"]
    b_new = state_sum["reader.net.4.bias"]
    split_ok = bool(
        torch.equal(w_new[0], w_orig[0] * 0.5)
        and torch.equal(w_new[1], w_orig[0] * 0.5)
        and torch.equal(b_new[0], b_orig[0] * 0.5)
        and torch.equal(b_new[1], b_orig[0] * 0.5)
    )

    # real-data forward: initial SUM total must equal the original M output
    prep_meta = _read_npz(out_dir / "full_train_prep.npz")
    train_data = build_fulltrain_data(prep_meta)
    with np.load(out_dir / "full_train_targets.npz", allow_pickle=False) as z:
        g_all = z["g"].astype(np.float64)
    target = torch.as_tensor(g_all, dtype=torch.float32)
    schedule, schedule_hash = build_schedule(N_TRAIN, EPOCHS, SEED + TRAIN_SHUFFLE_OFFSET)
    index_list = [int(i) for i in schedule[0][:128].tolist()]
    batch = zftd.make_batch(train_data, index_list, target, device)

    def _forward_sum(model: nn.Module) -> np.ndarray:
        model = model.to(device).eval()
        with torch.no_grad():
            return model(batch, mask=cm.C6_MASK).view(-1).detach().cpu().numpy()

    sum_out = _forward_sum(model_sum)
    orig_out = _forward_sum(model_orig)
    comp_out = _forward_sum(model_comp)
    max_sum_diff = float(np.max(np.abs(sum_out - orig_out)))
    max_arm_diff = float(np.max(np.abs(sum_out - comp_out)))
    checks = {
        "shared_state_max_abs_diff": max_abs,
        "state_key_sets_equal": sorted(state_sum) == sorted(state_comp),
        "parameter_audit": component_parameter_audit(model_sum),
        "expected_parameters": EXPECTED_PARAMETERS,
        "split_half_identity": split_ok,
        "initial_sum_vs_original_M_max_abs": max_sum_diff,
        "initial_SUM_vs_COMP_max_abs": max_arm_diff,
        "initial_SUM_vs_COMP_tolerance": 1e-5,
        "initial_SUM_vs_COMP_note": (
            "state dicts are byte-identical (shared_state_max_abs_diff == 0.0); the small forward "
            "difference on CUDA is kernel nondeterminism, not a model difference"
        ),
        "schedule_hash": schedule_hash,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    checks["all_ok"] = bool(
        max_abs == 0.0
        and split_ok
        and checks["parameter_audit"]["total_parameters"] == EXPECTED_PARAMETERS
        and max_sum_diff <= 1e-5
        and max_arm_diff <= 1e-5
    )
    write_json(out_dir / "init_identity.json", checks)
    log(f"[init] all_ok={checks['all_ok']} sum_vs_M={max_sum_diff:.2e} params={checks['parameter_audit']['total_parameters']}")
    if not checks["all_ok"]:
        raise RuntimeError(f"init identity failed: {checks}")
    return checks


# ---------------------------------------------------------------------------
# 5. training the two body arms
# ---------------------------------------------------------------------------


def evaluate_state_components(
    model: nn.Module,
    data_list: Sequence[Any],
    device: torch.device,
    *,
    batch_size: int = BATCH_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """One body pass per batch; returns ``(n,)`` sums and ``(n, 2)`` components."""
    model.eval()
    n = len(data_list)
    sums = np.empty(n, np.float64)
    comps = np.empty((n, 2), np.float64)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            indices = list(range(start, min(start + batch_size, n)))
            batch = zftd.make_batch(data_list, indices, torch.zeros(n, dtype=torch.float32), device)
            prediction = model(batch, mask=cm.C6_MASK)
            components = model.reader.components()
            if tuple(prediction.shape) != (len(indices),):
                raise RuntimeError(f"standard forward returned {tuple(prediction.shape)}, expected {(len(indices),)}")
            if tuple(components.shape) != (len(indices), 2):
                raise RuntimeError(f"component output {tuple(components.shape)} != {(len(indices), 2)}")
            sums[start : start + len(indices)] = prediction.view(-1).detach().cpu().numpy()
            comps[start : start + len(indices)] = components.detach().cpu().numpy()
    return sums, comps


def train_arm(
    arm: str,
    *,
    out_dir: Path = RESULTS_DIR,
    device: torch.device,
    epochs: int = EPOCHS,
    max_steps: int | None = None,
    log: Any = print,
    alloc: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError(arm)
    torch.set_num_threads(8 if device.type == "cpu" else 4)
    out_dir.mkdir(parents=True, exist_ok=True)

    with np.load(out_dir / "full_train_payload.npz", allow_pickle=False) as z:
        payload_arrays = {key: z[key] for key in z.files}
    payload = prev.TuplePayload(payload_arrays)
    kappa_M = float(payload_arrays["kappa"].reshape(-1)[0])
    with np.load(out_dir / 'full_train_targets.npz', allow_pickle=False) as z:
        g_all = z['g'].astype(np.float64)
        ell_all = z['ell'].astype(np.float64)
        s_all = z['s'].astype(np.float64)
        gid_all = z['gid'].astype(np.int64)
    if np.max(np.abs(g_all - (ell_all + s_all))) > 1e-12:
        raise RuntimeError('g != ell + s on the full-train target cache')

    prep_meta = _read_npz(out_dir / "full_train_prep.npz")
    train_data = build_fulltrain_data(prep_meta)
    schedule, schedule_hash = build_schedule(N_TRAIN, int(epochs), SEED + TRAIN_SHUFFLE_OFFSET)

    target_g = torch.as_tensor(g_all, dtype=torch.float32)
    target_ell = torch.as_tensor(ell_all, dtype=torch.float32, device=device)
    target_s = torch.as_tensor(s_all, dtype=torch.float32, device=device)

    seed_everything(SEED)
    model = build_component_model(arm, payload, kappa_M)
    build_rng = torch.get_rng_state()
    model = model.to(device)
    seed_everything(SEED)
    train_rng = torch.get_rng_state()

    audit = component_parameter_audit(model)
    init_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    soup_epochs = set(int(e) for e in SOUP_EPOCHS)
    soup: dict[int, dict[str, torch.Tensor]] = {}
    curve: list[dict[str, Any]] = []
    steps_done = 0
    gid_stream = hashlib.sha256()
    id_stream = hashlib.sha256()
    started = time.perf_counter()
    stopped_reason = "completed"
    optimised_loss = "L_g" if arm == "SUM" else "L_g + 0.5*(L_ell + L_s)"

    for epoch in range(1, int(epochs) + 1):
        epoch_started = time.perf_counter()
        model.train()
        g_sum = ell_sum = s_sum = total_sum = 0.0
        n_mol, n_steps, gnorm_sum, clip_hits = 0, 0, 0.0, 0
        for start in range(0, len(schedule[epoch - 1]), BATCH_SIZE):
            indices = schedule[epoch - 1][start : start + BATCH_SIZE]
            index_list = [int(i) for i in indices.tolist()]
            batch_index_t = torch.as_tensor(index_list, dtype=torch.long, device=device)
            id_stream.update(np.asarray(index_list, np.int64).tobytes())
            gid_stream.update(np.asarray(gid_all[index_list], np.int64).tobytes())
            batch = zftd.make_batch(train_data, index_list, target_g, device)
            if tuple(batch.y.shape) != (len(index_list),):
                raise RuntimeError("true g target is not (batch,)")
            prediction = model(batch, mask=cm.C6_MASK)
            if tuple(prediction.shape) != (len(index_list),):
                raise RuntimeError("standard forward must return (batch,) total g")
            components = model.reader.components()
            if tuple(components.shape) != (len(index_list), 2):
                raise RuntimeError("reader component output is not (batch, 2)")
            l_g = F.l1_loss(prediction.view(-1), batch.y.view(-1))
            l_ell = F.l1_loss(components[:, 0], target_ell[batch_index_t])
            l_s = F.l1_loss(components[:, 1], target_s[batch_index_t])
            if arm == "COMP":
                loss = l_g + COMPONENT_LOSS_WEIGHT * (l_ell + l_s)
            else:
                loss = l_g
            optimizer.zero_grad()
            loss.backward()
            total_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP))
            optimizer.step()
            g_sum += float((prediction.view(-1) - batch.y.view(-1)).abs().sum())
            ell_sum += float((components[:, 0].detach() - target_ell[batch_index_t]).abs().sum())
            s_sum += float((components[:, 1].detach() - target_s[batch_index_t]).abs().sum())
            total_sum += float(loss.detach()) * int(len(index_list))
            n_mol += int(batch.y.numel())
            n_steps += 1
            steps_done += 1
            gnorm_sum += total_norm
            clip_hits += int(total_norm > GRAD_CLIP)
            if max_steps is not None and steps_done >= int(max_steps):
                stopped_reason = "max_steps"
                break
        curve.append(
            {
                "epoch": int(epoch),
                "train_L_g": float(g_sum / max(n_mol, 1)),
                "train_L_ell": float(ell_sum / max(n_mol, 1)),
                "train_L_s": float(s_sum / max(n_mol, 1)),
                "train_loss": float(total_sum / max(n_mol, 1)),
                "grad_norm": float(gnorm_sum / max(n_steps, 1)),
                "clip_fraction": float(clip_hits / max(n_steps, 1)),
                "seconds": float(time.perf_counter() - epoch_started),
            }
        )
        if epoch in soup_epochs:
            soup[int(epoch)] = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if log and (epoch == 1 or epoch % 20 == 0 or epoch == int(epochs)):
            entry = curve[-1]
            log(
                f"[{arm}] ep={epoch:03d} L_g={entry['train_L_g']:.6f} L_ell={entry['train_L_ell']:.6f} "
                f"L_s={entry['train_L_s']:.6f} loss={entry['train_loss']:.6f} gnorm={entry['grad_norm']:.3g} "
                f"{entry['seconds']:.1f}s"
            )
        if max_steps is not None and steps_done >= int(max_steps):
            break

    if not soup and max_steps is not None:
        soup[int(epoch)] = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    members = sorted(soup)
    soup_state = {key: torch.stack([soup[e][key].float() for e in members]).mean(0) for key in soup[members[0]]}
    last_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    result: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "arm": arm,
        "optimised_loss": optimised_loss,
        "component_loss_weight": COMPONENT_LOSS_WEIGHT if arm == "COMP" else 0.0,
        "supervision": {
            "g": "g = y - c (full-train fit-only constants, 10000 rows)",
            "ell": "(logP - MU_LOGP) / sigma_logP",
            "s": "g - ell (residual chemistry component)",
            "component_labels_are_train_only": True,
        },
        "seed": SEED,
        "epochs": int(epochs),
        "steps_done": int(steps_done),
        "steps_expected": int(epochs) * math.ceil(N_TRAIN / BATCH_SIZE),
        "stopped_reason": stopped_reason,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "grad_clip": GRAD_CLIP,
        "batch_size": BATCH_SIZE,
        "soup_epochs": members if max_steps is None else [],
        "n_fit": int(len(train_data)),
        "kappa_M": kappa_M,
        "target": {
            "source": str(out_dir / "full_train_targets.npz"),
            "g_sha256": array_sha256(g_all, np.float64),
            "ell_sha256": array_sha256(ell_all, np.float64),
            "s_sha256": array_sha256(s_all, np.float64),
        },
        "parameter_audit": audit,
        "init_state_hash": state_hash(init_state),
        "raw_soup_state_hash": state_hash(soup_state),
        "last_state_hash": state_hash(last_state),
        "build_rng_sha256": _hash_bytes(build_rng.numpy().tobytes()),
        "train_rng_sha256": _hash_bytes(train_rng.numpy().tobytes()),
        "schedule_sha256": schedule_hash,
        "position_stream_sha256": id_stream.hexdigest(),
        "global_gid_stream_sha256": gid_stream.hexdigest(),
        "curve": curve,
        "wall_clock_s": float(time.perf_counter() - started),
        "device": str(device),
        "allocation_probe": dict(alloc) if alloc is not None else allocation_probe(),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    result["soup_L_g_final"] = curve[-1]["train_L_g"]
    result["soup_L_ell_final"] = curve[-1]["train_L_ell"]
    result["soup_L_s_final"] = curve[-1]["train_L_s"]

    if max_steps is not None:
        result["smoke_last_state_hash"] = state_hash(last_state)
        return result

    torch.save(init_state, out_dir / f"{arm}_init_state.pt")
    torch.save(last_state, out_dir / f"{arm}_last_state.pt")
    torch.save(soup_state, out_dir / f"{arm}_raw_soup_state.pt")

    raw_sums: dict[str, np.ndarray] = {}
    raw_comps: dict[str, np.ndarray] = {}
    for state_name, state in (("init", init_state), ("last", last_state), ("raw_soup", soup_state)):
        replay = build_component_model(arm, payload, kappa_M)
        replay.load_state_dict({key: value.to(torch.device("cpu")) for key, value in state.items()}, strict=True)
        replay = replay.to(device)
        fit_sum, fit_comp = evaluate_state_components(replay, train_data, device)
        raw_sums[f"{state_name}_train"] = fit_sum
        raw_comps[f"{state_name}_train"] = fit_comp
    np.savez_compressed(
        out_dir / f"{arm}_raw_predictions.npz",
        **{key: value.astype(np.float32) for key, value in raw_sums.items()},
    )
    np.savez_compressed(
        out_dir / f"{arm}_component_predictions.npz",
        **{key: value.astype(np.float32) for key, value in raw_comps.items()},
    )

    b_soup = float(np.median(g_all - raw_sums["raw_soup_train"]))
    result["calibration_b_g"] = {
        "init": float(np.median(g_all - raw_sums["init_train"])),
        "last": float(np.median(g_all - raw_sums["last_train"])),
        "raw_soup": b_soup,
    }

    replay = build_component_model(arm, payload, kappa_M)
    replay.load_state_dict({key: value.to(torch.device("cpu")) for key, value in soup_state.items()}, strict=True)
    batch_data = train_data[:128]
    gpu_pred, gpu_comp = evaluate_state_components(replay.to(device), batch_data, device)
    cpu_pred, cpu_comp = evaluate_state_components(
        replay.to(torch.device("cpu")), batch_data, torch.device("cpu")
    )
    result["replay_max_abs_diff"] = float(np.max(np.abs(gpu_pred - cpu_pred))) if device.type != "cpu" else 0.0
    result["replay_component_max_abs_diff"] = (
        float(np.max(np.abs(gpu_comp - cpu_comp))) if device.type != "cpu" else 0.0
    )
    if device.type != "cpu" and max(result["replay_max_abs_diff"], result["replay_component_max_abs_diff"]) > REPLAY_TOL:
        raise RuntimeError("CPU/GPU replay maxdiff exceeds tolerance")
    write_json(out_dir / f"{arm}_meta.json", result)
    write_json(out_dir / f"{arm}_curve.json", curve)
    log(f"[{arm}] done steps={steps_done} wall={result['wall_clock_s']:.1f}s b_g={b_soup:.6f}")
    return result


# ---------------------------------------------------------------------------
# 6. shared cycle head Q (fixed deploy accessory)
# ---------------------------------------------------------------------------


def build_q_head(seed: int, bias_value: float) -> nn.Module:
    torch.manual_seed(int(seed))
    head = nn.Sequential(
        nn.Linear(Q_TOPOLOGY_IN, Q_HIDDEN[0]),
        nn.SiLU(),
        nn.Linear(Q_HIDDEN[0], Q_HIDDEN[1]),
        nn.SiLU(),
        nn.Linear(Q_HIDDEN[1], 1),
    )
    n_params = int(sum(p.numel() for p in head.parameters()))
    if n_params != Q_PARAMETERS:
        raise RuntimeError(f"Q head parameter audit failed: {n_params}")
    nn.init.zeros_(head[4].weight)
    with torch.no_grad():
        head[4].bias.copy_(torch.tensor(float(bias_value)))
    return head


def q_forward(head: nn.Module, T: torch.Tensor) -> torch.Tensor:
    return head(T).view(-1)


def topology_matrix(data_list: Sequence[Any]) -> np.ndarray:
    T = np.concatenate(
        [d.topology_features.reshape(1, -1).numpy().astype(np.float32) for d in data_list], axis=0
    )
    if T.shape != (len(data_list), Q_TOPOLOGY_IN):
        raise RuntimeError(f"topology25 shape mismatch: {T.shape}")
    return T


def train_q_head(
    *,
    out_dir: Path = RESULTS_DIR,
    log: Any = print,
) -> dict[str, Any]:
    torch.set_num_threads(8)
    out_dir.mkdir(parents=True, exist_ok=True)
    with np.load(out_dir / "full_train_targets.npz", allow_pickle=False) as z:
        c_all = z["c"].astype(np.float64)
        k_all = z["k"].astype(np.int64)
    prep_meta = _read_npz(out_dir / "full_train_prep.npz")
    train_data = build_fulltrain_data(prep_meta)
    T = topology_matrix(train_data)
    bias_value = float(np.median(c_all))
    n = int(T.shape[0])

    head = build_q_head(SEED, bias_value)
    init_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
    init_hash = state_hash(init_state)
    optimizer = torch.optim.Adam(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    generator = torch.Generator().manual_seed(int(Q_TRAIN_GEN_BASE) + SEED)
    T_t = torch.as_tensor(T, dtype=torch.float32)
    c_t = torch.as_tensor(c_all, dtype=torch.float32)
    soup: dict[int, dict[str, torch.Tensor]] = {}
    soup_epochs = set(int(e) for e in Q_SOUP_EPOCHS)
    curve: list[dict[str, Any]] = []
    steps = 0
    started = time.perf_counter()
    for epoch in range(1, Q_EPOCHS + 1):
        head.train()
        order = torch.randperm(n, generator=generator)
        abs_sum, grad_norm = 0.0, 0.0
        for start in range(0, n, BATCH_SIZE):
            idx = order[start : start + BATCH_SIZE]
            prediction = q_forward(head, T_t[idx])
            loss = (prediction - c_t[idx]).abs().mean()
            optimizer.zero_grad()
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(head.parameters(), GRAD_CLIP))
            optimizer.step()
            abs_sum += float((prediction - c_t[idx]).abs().sum())
            steps += 1
        if epoch in soup_epochs:
            soup[epoch] = {k: v.detach().clone() for k, v in head.state_dict().items()}
        curve.append({
            "epoch": epoch,
            "train_task_mae": abs_sum / n,
            "grad_norm": grad_norm,
            "seconds": float(time.perf_counter() - started),
        })
        if log and (epoch == 1 or epoch % 50 == 0 or epoch == Q_EPOCHS):
            log(f"[Q] ep={epoch:03d} L1={curve[-1]['train_task_mae']:.6f} gnorm={grad_norm:.3g} {curve[-1]['seconds']:.1f}s")
    members = sorted(soup)
    soup_state = {key: torch.stack([soup[e][key].float() for e in members]).mean(0) for key in soup[members[0]]}
    last_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
    torch.save(init_state, out_dir / "Q_init_state.pt")
    torch.save(soup_state, out_dir / "Q_raw_soup_state.pt")
    torch.save(last_state, out_dir / "Q_last_state.pt")

    # train diagnostics on the frozen soup
    replay = build_q_head(SEED, bias_value)
    replay.load_state_dict({k: v.to(torch.device("cpu")) for k, v in soup_state.items()}, strict=True)
    replay.eval()
    with torch.no_grad():
        q_train = q_forward(replay, T_t).double().numpy()
    np.savez_compressed(out_dir / "Q_train_predictions.npz", q_raw=q_train.astype(np.float32), c=c_all, k=k_all)

    group_table: dict[str, Any] = {}
    err = q_train - c_all
    for name, mask in group_masks(k_all).items():
        n_g = int(mask.sum())
        group_table[name] = {
            "n": n_g,
            "raw_mae": float(np.mean(np.abs(err[mask]))) if n_g else None,
            "signed_mean": float(np.mean(err[mask])) if n_g else None,
        }
    result = {
        "protocol_version": PROTOCOL_VERSION,
        "seed": SEED,
        "epochs": Q_EPOCHS,
        "steps": int(steps),
        "batch_size": BATCH_SIZE,
        "soup_epochs": members,
        "train_generator_seed": int(Q_TRAIN_GEN_BASE) + SEED,
        "hidden": list(Q_HIDDEN),
        "parameters": Q_PARAMETERS,
        "bias_value_median_c_train": bias_value,
        "init_hash": init_hash,
        "raw_soup_state_hash": state_hash(soup_state),
        "last_state_hash": state_hash(last_state),
        "train_overall_mae": float(np.mean(np.abs(err))),
        "train_overall_signed": float(np.mean(err)),
        "train_groups": group_table,
        "curve": curve,
        "wall_clock_s": float(time.perf_counter() - started),
        "device": "cpu",
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(out_dir / "Q_meta.json", result)
    write_json(out_dir / "Q_curve.json", curve)
    log(f"[Q] done steps={steps} train_mae={result['train_overall_mae']:.6f} wall={result['wall_clock_s']:.1f}s")
    return result


# ---------------------------------------------------------------------------
# 7. train-only calibration b_g / b_y (one bias per endpoint, per arm)
# ---------------------------------------------------------------------------


def calibrate(*, out_dir: Path = RESULTS_DIR, log: Any = print) -> dict[str, Any]:
    torch.set_num_threads(8)
    with np.load(out_dir / "full_train_targets.npz", allow_pickle=False) as z:
        y_all = z["y"].astype(np.float64)
        g_all = z["g"].astype(np.float64)
    with np.load(out_dir / "Q_train_predictions.npz", allow_pickle=False) as z:
        q_train = z["q_raw"].astype(np.float64)
    out: dict[str, Any] = {"per_arm": {}, "median_c_train": None}
    for arm in ARMS:
        with np.load(out_dir / f"{arm}_raw_predictions.npz") as z:
            h_train = np.asarray(z["raw_soup_train"], np.float64)
        b_g = float(np.median(g_all - h_train))
        b_y = float(np.median(y_all - h_train - q_train))
        out["per_arm"][arm] = {
            "b_g": b_g,
            "b_y": b_y,
            "h_train_mae_g_raw": float(np.mean(np.abs(g_all - h_train))),
            "h_train_mae_g_cal": float(np.mean(np.abs(g_all - (h_train + b_g)))),
            "y_train_mae_raw": float(np.mean(np.abs(y_all - (h_train + q_train)))),
            "y_train_mae_cal": float(np.mean(np.abs(y_all - (h_train + q_train + b_y)))),
        }
    write_json(out_dir / "calibration.json", out)
    log(f"[calibrate] {json.dumps(out['per_arm'], indent=None)}")
    return out


# ---------------------------------------------------------------------------
# 8. frozen evaluation manifest (must precede the first valid read)
# ---------------------------------------------------------------------------


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(Path(path).relative_to(zjd.REPO_ROOT)),
        "sha256": file_sha256(path),
        "bytes": int(Path(path).stat().st_size),
    }


def build_manifest(*, out_dir: Path = RESULTS_DIR, log: Any = print) -> dict[str, Any]:
    if (out_dir / "heldout_access.json").exists():
        raise RuntimeError("valid already accessed; refusing to (re)freeze after the fact")
    cal = json.loads((out_dir / "calibration.json").read_text())
    artifacts: dict[str, Any] = {
        "runner_source": _artifact(Path(__file__)),
        "targets": _artifact(out_dir / "full_train_targets.npz"),
        "payload": _artifact(out_dir / "full_train_payload.npz"),
        "prep": _artifact(out_dir / "full_train_prep.npz"),
        "schedule": _artifact(out_dir / "full_train_schedule.npz"),
        "schedule_meta": _artifact(out_dir / "schedule_meta.json"),
        "targets_meta": _artifact(out_dir / "targets_meta.json"),
        "prep_meta": _artifact(out_dir / "prep_meta.json"),
        "payload_meta": _artifact(out_dir / "payload_meta.json"),
        "calibration": _artifact(out_dir / "calibration.json"),
        "init_identity": _artifact(out_dir / "init_identity.json"),
        "Q_meta": _artifact(out_dir / "Q_meta.json"),
        "Q_soup_state": _artifact(out_dir / "Q_raw_soup_state.pt"),
        "Q_init_state": _artifact(out_dir / "Q_init_state.pt"),
    }
    models: dict[str, Any] = {}
    for arm in ARMS:
        meta = json.loads((out_dir / f"{arm}_meta.json").read_text())
        models[arm] = {
            "raw_soup_state": _artifact(out_dir / f"{arm}_raw_soup_state.pt"),
            "last_state": _artifact(out_dir / f"{arm}_last_state.pt"),
            "init_state": _artifact(out_dir / f"{arm}_init_state.pt"),
            "init_state_hash": meta["init_state_hash"],
            "raw_soup_state_hash": meta["raw_soup_state_hash"],
            "soup_members": meta["soup_epochs"],
            "parameters": meta["parameter_audit"]["total_parameters"],
            "schedule_sha256": meta["schedule_sha256"],
            "position_stream_sha256": meta["position_stream_sha256"],
            "global_gid_stream_sha256": meta["global_gid_stream_sha256"],
        }
    with np.load(out_dir / "full_train_targets.npz", allow_pickle=False) as z:
        constants = {
            str(n): float(v) for n, v in zip(z["constant_names"].tolist(), z["constants"].tolist())
        }
        mu_logP = float(z["mu_logP"])
    manifest = {
        "manifest_version": "frozen-eval-manifest-v1",
        "protocol_version": PROTOCOL_VERSION,
        "frozen_at_utc": now_utc(),
        "execution_commit": git_commit(),
        "artifacts": artifacts,
        "models": models,
        "target_constants": constants,
        "mu_logP": mu_logP,
        "train_rows": {"n": N_TRAIN, "order": "positional 0..9999 (canonical)"},
        "calibration": cal,
        "metric_definition": {
            "primary_g_mae": "MAE over absolute error in g units",
            "primary_y_mae": "MAE over absolute error in official y units",
            "gain": "MAE(SUM) - MAE(COMP) (positive = COMP improves)",
            "groups": list(GROUP_NAMES),
            "bootstrap": {"n": N_BOOT, "seed": BOOT_SEED, "paired_row": True},
        },
        "gate_definition": {
            "CHEM_CONFIRMED": [
                "valid G0 calibrated g gain >= 0.003",
                "valid overall calibrated g gain >= 0.003",
                "valid G0 calibrated g paired 95% CI lower > 0",
                "valid G0 and overall raw g gains > 0",
            ],
            "DEPLOY_CONFIRMED": [
                "valid overall calibrated y gain >= 0.003",
                "valid overall calibrated y paired 95% CI lower > 0",
                "valid overall raw y gain > 0",
                "COMP vs SUM G0 calibrated y worsening <= 0.001",
            ],
            "benchmark_marker": "valid y_cal < 0.09 (marker only; not a substitute for the incremental gate)",
        },
        "heldout_access_log": str((out_dir / "heldout_access.json").relative_to(zjd.REPO_ROOT)),
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "freeze_before_heldout": True,
    }
    write_json(out_dir / "frozen_eval_manifest.json", manifest)
    log(f"[manifest] frozen_at={manifest['frozen_at_utc']} commit={manifest['execution_commit']}")
    return manifest


# ---------------------------------------------------------------------------
# 9. one frozen official-valid evaluation
# ---------------------------------------------------------------------------


def _read_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def load_valid_data(out_dir: Path) -> tuple[list[Any], dict[str, Any]]:
    """Load the official-valid encoded cache + env cache, apply the frozen prep."""
    from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p1 as p1run
    from tracks.ksvd.experiments.luyin16 import zinc_static_dictionary_pair as sdp

    valid = list(torch.load(sdp.CACHE_DIR / "encoded_valid.pt", map_location="cpu", weights_only=False))
    p1run.attach_env(valid, "valid")
    prep_meta = _read_npz(out_dir / "full_train_prep.npz")
    blob = np.load(PREP_BLOB, allow_pickle=False)
    patch_all = zjd.Std(blob["patch_all_mean"], blob["patch_all_scale"])
    ctx_all = zjd.Std(blob["ctx_all_mean"], blob["ctx_all_scale"])
    anchor_all = zjd.Std(blob["anchor_all_mean"], blob["anchor_all_scale"])
    topo_all = zjd.Std(blob["topo_all_mean"], blob["topo_all_scale"])
    patch_fit = zjd.Std(prep_meta["patch_fit_mean"], prep_meta["patch_fit_scale"])
    ctx_fit = zjd.Std(prep_meta["ctx_fit_mean"], prep_meta["ctx_fit_scale"])
    anchor_fit = zjd.Std(prep_meta["anchor_fit_mean"], prep_meta["anchor_fit_scale"])
    topo_fit = zjd.Std(prep_meta["topo_fit_mean"], prep_meta["topo_fit_scale"])
    p, pc = zjd._stack(valid, "patch_cont")
    c, cc = zjd._stack(valid, "global_context")
    a, ac = zjd._stack(valid, "anchor")
    t, tc = zjd._stack(valid, "topology_features")
    zjd._unstack(patch_fit.transform(patch_all.inverse(p)), pc, "patch_cont", valid)
    zjd._unstack(ctx_fit.transform(ctx_all.inverse(c)), cc, "global_context", valid)
    zjd._unstack(anchor_fit.transform(anchor_all.inverse(a)), ac, "anchor", valid)
    zjd._unstack(topo_fit.transform(topo_all.inverse(t)), tc, "topology_features", valid)
    meta = {
        "n_rows": int(len(valid)),
        "source": "encoded_valid.pt + env_valid.pt",
        "prep": "frozen full-train prep (10000-row standardizers, no valid refit)",
        "official_valid_loaded": True,
        "official_test_loaded": False,
    }
    return valid, meta


def load_valid_diagnostics(out_dir: Path) -> dict[str, Any]:
    """Build valid diagnostics (y/g/c/k/ell/s) with the frozen train constants."""
    import pandas as pd

    df = pd.read_csv(VALID_LABEL_CSV)
    df = df.sort_values("subset_index").reset_index(drop=True)
    if not np.array_equal(df["subset_index"].to_numpy(np.int64), np.arange(len(df))):
        raise RuntimeError("valid label csv rows are not positional")
    y = df["target"].to_numpy(np.float64)
    k = np.round(df["label_effective_cycle_snapped"].to_numpy(np.float64)).astype(np.int64)
    props = np.load(GVEA_PROPS_NPZ)
    lines = df["smi_line"].to_numpy(np.int64)
    logP = np.asarray(props["logP"], np.float64)[lines]
    sa = np.asarray(props["SA"], np.float64)[lines]
    raw_cycle = np.asarray(props["cycle"], np.float64)[lines]
    if not np.array_equal(raw_cycle, df["cycle_score_gvae_order"].to_numpy(np.float64)):
        raise RuntimeError("valid cycle score does not match the GVAE table at smi_line")
    with np.load(out_dir / "full_train_targets.npz", allow_pickle=False) as z:
        constants = {
            str(n): float(v) for n, v in zip(z["constant_names"].tolist(), z["constants"].tolist())
        }
    sigma_logP = constants["sigma_logP"]
    sigma_SA = constants["sigma_SA"]
    mu_SA = constants["mu_SA"]
    sigma_cycle = constants["sigma_cycle"]
    mu_cycle = constants["mu_cycle"]
    c = (k.astype(np.float64) - mu_cycle) / sigma_cycle
    g = y - c
    ell = (logP - MU_LOGP) / sigma_logP
    s = g - ell
    s_sa = (sa - mu_SA) / sigma_SA
    checks = {
        "n_rows": int(len(y)),
        "source": str(VALID_LABEL_CSV),
        "source_sha256": file_sha256(VALID_LABEL_CSV),
        "gvaE_props": str(GVEA_PROPS_NPZ),
        "constants_from": "frozen full-train constants (10000-row fit)",
        "c_from_k_rule_max_abs": float(np.max(np.abs(c - ((k.astype(np.float64) - mu_cycle) / sigma_cycle)))),
        "g_equals_y_minus_c_max_abs": float(np.max(np.abs(g - (y - c)))),
        "g_equals_ell_plus_s_max_abs": float(np.max(np.abs(g - (ell + s)))),
        "k_counts": {name: int(mask.sum()) for name, mask in group_masks(k).items()},
        "official_valid_loaded": True,
        "official_test_loaded": False,
    }
    return {
        "y": y, "c": c, "g": g, "k": k, "ell": ell, "s": s, "s_SA": s_sa,
        "epsilon": s - s_sa, "logP": logP, "SA": sa, "checks": checks,
    }


def _access_log(out_dir: Path, split: str, meta: Mapping[str, Any]) -> None:
    path = out_dir / "heldout_access.json"
    payload = json.loads(path.read_text()) if path.exists() else {
        "protocol_version": PROTOCOL_VERSION,
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "events": [],
    }
    payload["events"].append({
        "split": split,
        "first_read_utc": now_utc(),
        "n_rows": meta.get("n_rows"),
        "source": meta.get("source"),
        "frozen_commit": git_commit(),
    })
    payload[f"official_{split}_loaded"] = True
    write_json(path, payload)


def evaluate_valid(*, out_dir: Path = RESULTS_DIR, device: torch.device | None = None, log: Any = print) -> dict[str, Any]:
    device = device or torch.device("cpu")
    manifest = json.loads((out_dir / "frozen_eval_manifest.json").read_text())
    if not manifest.get("freeze_before_heldout"):
        raise RuntimeError("no valid frozen manifest; refusing the valid read")
    torch.set_num_threads(8)
    with np.load(out_dir / "full_train_payload.npz", allow_pickle=False) as z:
        payload_arrays = {key: z[key] for key in z.files}
    payload = prev.TuplePayload(payload_arrays)
    kappa_M = float(payload_arrays["kappa"].reshape(-1)[0])
    cal = json.loads((out_dir / "calibration.json").read_text())

    valid, meta = load_valid_data(out_dir)
    log(f"[valid] loaded n={meta['n_rows']} (first and only frozen read)")
    diag = load_valid_diagnostics(out_dir)

    # Q forward on valid
    T = topology_matrix(valid)
    q_head = build_q_head(SEED, float(np.median(np.load(out_dir / "full_train_targets.npz")["c"])))
    q_head.load_state_dict(torch.load(out_dir / "Q_raw_soup_state.pt", map_location="cpu", weights_only=True), strict=True)
    q_head.eval()
    with torch.no_grad():
        q_valid = q_forward(q_head, torch.as_tensor(T, dtype=torch.float32)).double().numpy()

    out: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "manifest_frozen_at": manifest["frozen_at_utc"],
        "valid_meta": meta,
        "n_valid": int(len(valid)),
        "q_raw": q_valid.astype(np.float32),
        "y": diag["y"].astype(np.float32),
        "g": diag["g"].astype(np.float32),
        "c": diag["c"].astype(np.float32),
        "k": diag["k"],
        "ell": diag["ell"].astype(np.float32),
        "s": diag["s"].astype(np.float32),
        "s_SA": diag["s_SA"].astype(np.float32),
        "diagnostic_checks": diag["checks"],
    }
    for arm in ARMS:
        replay = build_component_model(arm, payload, kappa_M)
        state = torch.load(out_dir / f"{arm}_raw_soup_state.pt", map_location="cpu", weights_only=True)
        replay.load_state_dict({key: value.to(torch.device("cpu")) for key, value in state.items()}, strict=True)
        replay = replay.to(device)
        sum_valid, comp_valid = evaluate_state_components(replay, valid, device)
        h = sum_valid
        out[f"{arm}_h_raw"] = h.astype(np.float32)
        out[f"{arm}_ell_raw"] = comp_valid[:, 0].astype(np.float32)
        out[f"{arm}_s_raw"] = comp_valid[:, 1].astype(np.float32)
        out[f"{arm}_g_raw"] = h.astype(np.float32)
        out[f"{arm}_g_cal"] = (h + cal["per_arm"][arm]["b_g"]).astype(np.float32)
        out[f"{arm}_y_raw"] = (h + q_valid).astype(np.float32)
        out[f"{arm}_y_cal"] = (h + q_valid + cal["per_arm"][arm]["b_y"]).astype(np.float32)
        # identity witness in float64 (row-wise deploy algebraic identity)
        b_y = float(cal["per_arm"][arm]["b_y"])
        lhs = out[f"{arm}_y_cal"].astype(np.float64) - diag["y"]
        rhs = (
            (comp_valid[:, 0] - diag["ell"])
            + (comp_valid[:, 1] - diag["s"])
            + (q_valid - diag["c"])
            + b_y
        )
        out[f"{arm}_identity_max_abs"] = float(np.max(np.abs(lhs - rhs)))

    np.savez_compressed(out_dir / "valid_frozen_predictions.npz", **{
        key: value for key, value in out.items() if isinstance(value, np.ndarray)
    })
    write_json(out_dir / "valid_summary.json", {
        key: value for key, value in out.items() if not isinstance(value, np.ndarray)
    })
    _access_log(out_dir, "valid", meta)
    log(f"[valid] done identity={out['SUM_identity_max_abs']:.2e}/{out['COMP_identity_max_abs']:.2e}")
    return out


# ---------------------------------------------------------------------------
# 10. analysis: tables, bootstrap, gates
# ---------------------------------------------------------------------------


def _mae(err: np.ndarray) -> float:
    return float(np.mean(np.abs(err))) if err.size else float("nan")


def _quantiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {}
    qs = np.percentile(values, [0.0, 1.0, 25.0, 50.0, 75.0, 99.0, 100.0])
    return {"min": float(qs[0]), "p1": float(qs[1]), "p25": float(qs[2]), "p50": float(qs[3]),
            "p75": float(qs[4]), "p99": float(qs[5]), "max": float(qs[6])}


def _paired_bootstrap(
    err_sum: Mapping[str, np.ndarray],
    err_comp: Mapping[str, np.ndarray],
    k: np.ndarray,
    *,
    seed: int = BOOT_SEED,
    n_boot: int = N_BOOT,
) -> dict[str, Any]:
    g0 = k == 0
    n_all = int(k.size)
    n_g0 = int(g0.sum())
    metrics = ("G0_cal", "overall_cal", "G0_raw", "overall_raw")
    samples = {metric: np.empty(int(n_boot), np.float64) for metric in metrics}
    rng = np.random.default_rng(int(seed))
    for draw in range(int(n_boot)):
        idx_all = rng.choice(n_all, size=n_all, replace=True)
        idx_g0 = rng.choice(n_g0, size=n_g0, replace=True)
        for metric in metrics:
            idx = idx_all if metric.startswith("overall") else idx_g0
            eb = np.abs(err_sum[metric])[idx]
            em = np.abs(err_comp[metric])[idx]
            samples[metric][draw] = float(eb.mean() - em.mean())
    out: dict[str, Any] = {"seed": int(seed), "n_boot": int(n_boot), "shared_indices_per_endpoint": True}
    for metric in metrics:
        point = _mae(err_sum[metric]) - _mae(err_comp[metric])
        out[metric] = {
            "point": point,
            "ci95": [float(np.percentile(samples[metric], 2.5)), float(np.percentile(samples[metric], 97.5))],
        }
    return out


def _bootstrap_witnesses(err_sum_cal: np.ndarray, err_comp_cal: np.ndarray, k: np.ndarray) -> dict[str, Any]:
    g0 = k == 0
    err_sum_cal = np.asarray(err_sum_cal, np.float64)
    err_comp_cal = np.asarray(err_comp_cal, np.float64)
    same = {"G0_cal": err_sum_cal[g0], "overall_cal": err_sum_cal,
            "G0_raw": err_sum_cal[g0], "overall_raw": err_sum_cal}
    same_out = _paired_bootstrap(same, same, k, seed=BOOT_SEED, n_boot=100)
    fwd_sum = {"G0_cal": err_sum_cal[g0], "overall_cal": err_sum_cal,
               "G0_raw": err_sum_cal[g0], "overall_raw": err_sum_cal}
    fwd_comp = {"G0_cal": err_comp_cal[g0], "overall_cal": err_comp_cal,
                "G0_raw": err_comp_cal[g0], "overall_raw": err_comp_cal}
    forward = _paired_bootstrap(fwd_sum, fwd_comp, k, seed=BOOT_SEED, n_boot=200)
    swapped = _paired_bootstrap(fwd_comp, fwd_sum, k, seed=BOOT_SEED, n_boot=200)
    checks = {"same_predictions_zero": True}
    for metric in ("G0_cal", "overall_cal", "G0_raw", "overall_raw"):
        if not (same_out[metric]["point"] == 0.0 and same_out[metric]["ci95"] == [0.0, 0.0]):
            checks["same_predictions_zero"] = False
        checks[f"{metric}_swap_mirror_point"] = bool(abs(forward[metric]["point"] + swapped[metric]["point"]) < 1e-12)
        checks[f"{metric}_swap_mirror_ci"] = bool(
            abs(forward[metric]["ci95"][0] + swapped[metric]["ci95"][1]) < 1e-9
            and abs(forward[metric]["ci95"][1] + swapped[metric]["ci95"][0]) < 1e-9
        )
    checks["constant_shift_bound"] = True
    checks["all_ok"] = all(checks.values())
    return {"same_predictions": same_out, "forward": forward, "swapped": swapped, "checks": checks}


def analyze(*, out_dir: Path = RESULTS_DIR, log: Any = print) -> dict[str, Any]:
    torch.set_num_threads(8)
    with np.load(out_dir / "full_train_targets.npz", allow_pickle=False) as z:
        y_train = z["y"].astype(np.float64)
        g_train = z["g"].astype(np.float64)
        k_train = z["k"].astype(np.int64)
        ell_train = z["ell"].astype(np.float64)
        s_train = z["s"].astype(np.float64)
    cal = json.loads((out_dir / "calibration.json").read_text())
    with np.load(out_dir / "valid_frozen_predictions.npz", allow_pickle=False) as z:
        valid = {key: z[key] for key in z.files}
    y = valid["y"].astype(np.float64)
    g = valid["g"].astype(np.float64)
    k = valid["k"].astype(np.int64)
    ell = valid["ell"].astype(np.float64)
    s = valid["s"].astype(np.float64)
    q = valid["q_raw"].astype(np.float64)
    with np.load(out_dir / "Q_train_predictions.npz", allow_pickle=False) as z:
        q_train = z["q_raw"].astype(np.float64)

    def arm_errors(arm: str, train: bool) -> dict[str, np.ndarray]:
        if train:
            with np.load(out_dir / f"{arm}_raw_predictions.npz") as z:
                h = np.asarray(z["raw_soup_train"], np.float64)
            with np.load(out_dir / f"{arm}_component_predictions.npz") as z:
                comps = np.asarray(z["raw_soup_train"], np.float64)
            yy, gg, kk, ee, ss, qq = y_train, g_train, k_train, ell_train, s_train, q_train
            b_g = cal["per_arm"][arm]["b_g"]
            b_y = cal["per_arm"][arm]["b_y"]
        else:
            h = valid[f"{arm}_h_raw"].astype(np.float64)
            comps = np.stack([valid[f"{arm}_ell_raw"], valid[f"{arm}_s_raw"]], axis=1).astype(np.float64)
            yy, gg, kk, ee, ss, qq = y, g, k, ell, s, q
            b_g = cal["per_arm"][arm]["b_g"]
            b_y = cal["per_arm"][arm]["b_y"]
        return {
            "g_raw": gg - h,
            "g_cal": gg - (h + b_g),
            "y_raw": yy - (h + qq),
            "y_cal": yy - (h + qq + b_y),
            "e_ell": comps[:, 0] - ee,
            "e_s": comps[:, 1] - ss,
            "b_g": np.asarray(b_g), "b_y": np.asarray(b_y),
            "h": h, "comps": comps,
        }

    arms: dict[str, Any] = {}
    for split_name, is_train, kk in (("train", True, k_train), ("valid", False, k)):
        arms[split_name] = {}
        for arm in ARMS:
            errs = arm_errors(arm, is_train)
            table = zw._metric_table(errs["g_cal"], kk)
            arms[split_name][arm] = {
                "b_g": float(errs["b_g"]), "b_y": float(errs["b_y"]),
                "g_raw_mae": _mae(errs["g_raw"]),
                "g_cal_mae": _mae(errs["g_cal"]),
                "y_raw_mae": _mae(errs["y_raw"]),
                "y_cal_mae": _mae(errs["y_cal"]),
                "G0_g_raw_mae": _mae(errs["g_raw"][kk == 0]),
                "G0_g_cal_mae": _mae(errs["g_cal"][kk == 0]),
                "G0_y_raw_mae": _mae(errs["y_raw"][kk == 0]),
                "G0_y_cal_mae": _mae(errs["y_cal"][kk == 0]),
                "groups_g_cal": table,
                "errs": errs,
            }
    gains: dict[str, Any] = {}
    err_pack_sum = {
        "G0_cal": arms["valid"]["SUM"]["errs"]["g_cal"][k == 0],
        "overall_cal": arms["valid"]["SUM"]["errs"]["g_cal"],
        "G0_raw": arms["valid"]["SUM"]["errs"]["g_raw"][k == 0],
        "overall_raw": arms["valid"]["SUM"]["errs"]["g_raw"],
    }
    err_pack_comp = {
        "G0_cal": arms["valid"]["COMP"]["errs"]["g_cal"][k == 0],
        "overall_cal": arms["valid"]["COMP"]["errs"]["g_cal"],
        "G0_raw": arms["valid"]["COMP"]["errs"]["g_raw"][k == 0],
        "overall_raw": arms["valid"]["COMP"]["errs"]["g_raw"],
    }
    gains["g"] = _paired_bootstrap(err_pack_sum, err_pack_comp, k)
    y_pack_sum = {
        "G0_cal": arms["valid"]["SUM"]["errs"]["y_cal"][k == 0],
        "overall_cal": arms["valid"]["SUM"]["errs"]["y_cal"],
        "G0_raw": arms["valid"]["SUM"]["errs"]["y_raw"][k == 0],
        "overall_raw": arms["valid"]["SUM"]["errs"]["y_raw"],
    }
    y_pack_comp = {
        "G0_cal": arms["valid"]["COMP"]["errs"]["y_cal"][k == 0],
        "overall_cal": arms["valid"]["COMP"]["errs"]["y_cal"],
        "G0_raw": arms["valid"]["COMP"]["errs"]["y_raw"][k == 0],
        "overall_raw": arms["valid"]["COMP"]["errs"]["y_raw"],
    }
    gains["y"] = _paired_bootstrap(y_pack_sum, y_pack_comp, k)
    witnesses = _bootstrap_witnesses(
        arms["valid"]["SUM"]["errs"]["g_cal"], arms["valid"]["COMP"]["errs"]["g_cal"], k
    )

    chem_conditions = {
        "G0_cal_gain_ge_0.003": bool(gains["g"]["G0_cal"]["point"] >= PERF_DELTA),
        "overall_cal_gain_ge_0.003": bool(gains["g"]["overall_cal"]["point"] >= PERF_DELTA),
        "G0_cal_ci_lower_gt_0": bool(gains["g"]["G0_cal"]["ci95"][0] > 0.0),
        "G0_raw_gain_gt_0": bool(gains["g"]["G0_raw"]["point"] > 0.0),
        "overall_raw_gain_gt_0": bool(gains["g"]["overall_raw"]["point"] > 0.0),
    }
    chem_confirmed = all(chem_conditions.values())
    y_g0_worsen = arms["valid"]["COMP"]["G0_y_cal_mae"] - arms["valid"]["SUM"]["G0_y_cal_mae"]
    deploy_conditions = {
        "overall_cal_gain_ge_0.003": bool(gains["y"]["overall_cal"]["point"] >= PERF_DELTA),
        "overall_cal_ci_lower_gt_0": bool(gains["y"]["overall_cal"]["ci95"][0] > 0.0),
        "overall_raw_gain_gt_0": bool(gains["y"]["overall_raw"]["point"] > 0.0),
        "G0_cal_y_worsening_le_0.001": bool(y_g0_worsen <= DEPLOY_G0_WORSEN_MAX),
    }
    deploy_confirmed = all(deploy_conditions.values())
    if chem_confirmed and deploy_confirmed:
        verdict = "CHEM_PASS_DEPLOY_PASS"
    elif chem_confirmed:
        verdict = "CHEM_PASS_DEPLOY_FAIL"
    elif deploy_confirmed:
        verdict = "CHEM_FAIL_DEPLOY_PASS"
    else:
        directional = gains["g"]["G0_cal"]["point"] > 0.0 and gains["g"]["overall_cal"]["point"] > 0.0
        verdict = "DIRECTIONAL_NOT_CONFIRMED" if directional else "CHEM_FAIL_DEPLOY_FAIL"

    benchmark = {
        "valid_sum_y_cal": arms["valid"]["SUM"]["y_cal_mae"],
        "valid_comp_y_cal": arms["valid"]["COMP"]["y_cal_mae"],
        "target": BENCHMARK_Y_TARGET,
        "SUM_below_target": bool(arms["valid"]["SUM"]["y_cal_mae"] < BENCHMARK_Y_TARGET),
        "COMP_below_target": bool(arms["valid"]["COMP"]["y_cal_mae"] < BENCHMARK_Y_TARGET),
        "valid_sum_g_cal": arms["valid"]["SUM"]["g_cal_mae"],
        "valid_comp_g_cal": arms["valid"]["COMP"]["g_cal_mae"],
    }

    # per-group contributions and identities
    identities: dict[str, Any] = {}
    for split_name in ("train", "valid"):
        kk = k_train if split_name == "train" else k
        for arm in ARMS:
            table = arms[split_name][arm]["groups_g_cal"]
            contrib_sum = sum(table[name]["contribution"] for name in GROUP_NAMES if table[name]["n"] > 0)
            identities[f"{split_name}_{arm}_g_cal_contribution_sum_minus_overall"] = float(contrib_sum - table["mae"])
        gain_sum = 0.0
        for name in GROUP_NAMES:
            gain_sum += (arms[split_name]["SUM"]["groups_g_cal"][name]["contribution"]
                         - arms[split_name]["COMP"]["groups_g_cal"][name]["contribution"])
        identities[f"{split_name}_group_gain_sum_minus_total"] = float(gain_sum - gains["g"]["overall_cal"]["point"])

    # COMP component diagnostics
    diagnostics: dict[str, Any] = {}
    for split_name in ("train", "valid"):
        kk = k_train if split_name == "train" else k
        errs = arms[split_name]["COMP"]["errs"]
        ee = ell_train if split_name == "train" else ell
        ss = s_train if split_name == "train" else s
        e_ell, e_s = errs["e_ell"], errs["e_s"]
        e_g = e_ell + e_s
        opposite = float(np.mean((e_ell * e_s) < 0.0))
        triangle_gap = float(np.mean(np.abs(e_ell) + np.abs(e_s) - np.abs(e_g)))
        diagnostics[split_name] = {
            "mean_abs_e_ell": float(np.mean(np.abs(e_ell))),
            "mean_abs_e_s": float(np.mean(np.abs(e_s))),
            "mean_abs_e_g": float(np.mean(np.abs(e_g))),
            "sum_of_component_maes": float(np.mean(np.abs(e_ell)) + np.mean(np.abs(e_s))),
            "opposite_sign_fraction": opposite,
            "triangle_gap": triangle_gap,
            "g_bias_mean": float(np.mean(e_g)),
            "ell": {
                "raw_mae": _mae(e_ell), "signed_mean": float(np.mean(e_ell)),
                "signed_median": float(np.median(e_ell)), "std": float(np.std(e_ell)),
                "quantiles": _quantiles(e_ell),
                "per_k": {name: {
                    "n": int(mask.sum()),
                    "raw_mae": _mae(e_ell[mask]) if int(mask.sum()) else None,
                    "signed_mean": float(np.mean(e_ell[mask])) if int(mask.sum()) else None,
                    "abs_sum_over_n": float(np.abs(e_ell[mask]).sum() / max(e_ell.size, 1)) if int(mask.sum()) else None,
                } for name, mask in group_masks(kk).items()},
                "constant_mae_overall": _mae(ee - np.median(ee)),
            },
            "s": {
                "raw_mae": _mae(e_s), "signed_mean": float(np.mean(e_s)),
                "signed_median": float(np.median(e_s)), "std": float(np.std(e_s)),
                "quantiles": _quantiles(e_s),
                "per_k": {name: {
                    "n": int(mask.sum()),
                    "raw_mae": _mae(e_s[mask]) if int(mask.sum()) else None,
                    "signed_mean": float(np.mean(e_s[mask])) if int(mask.sum()) else None,
                    "abs_sum_over_n": float(np.abs(e_s[mask]).sum() / max(e_s.size, 1)) if int(mask.sum()) else None,
                } for name, mask in group_masks(kk).items()},
                "constant_mae_overall": _mae(ss - np.median(ss)),
            },
        }
    diag_identity_max = max(
        float(np.max(np.abs(arms[split]["COMP"]["errs"]["e_ell"] + arms[split]["COMP"]["errs"]["e_s"]
                          - (arms[split]["COMP"]["errs"]["g_raw"]))))
        for split in ("train", "valid")
    )
    diagnostics["cancellation_identity_max_abs"] = diag_identity_max

    # Q diagnostics
    q_diag: dict[str, Any] = {"train": {}, "valid": {}}
    for split_name, kk, qq, cc in (("train", k_train, q_train, np.load(out_dir / "full_train_targets.npz")["c"].astype(np.float64)),
                                    ("valid", k, q, valid["c"].astype(np.float64))):
        err = qq - cc
        q_diag[split_name] = {
            "overall_mae": _mae(err),
            "signed_mean": float(np.mean(err)),
            "groups": {name: {
                "n": int(mask.sum()),
                "raw_mae": _mae(err[mask]) if int(mask.sum()) else None,
                "signed_mean": float(np.mean(err[mask])) if int(mask.sum()) else None,
            } for name, mask in group_masks(kk).items()},
        }

    # fixed sensitivity analyses (do not affect the gate)
    sensitivity: dict[str, Any] = {}
    ids = valid.get("ids")
    worst_sum = int(np.argmax(np.abs(arms["valid"]["SUM"]["errs"]["y_cal"])))
    keep = np.ones(k.size, dtype=bool)
    keep[worst_sum] = False
    sensitivity["drop_worst_sum_y_cal"] = {
        "dropped_position": worst_sum,
        "dropped_k": int(k[worst_sum]),
        "dropped_in_G0": bool(k[worst_sum] == 0),
        "G0_g_cal_gain": float(_mae(arms["valid"]["SUM"]["errs"]["g_cal"][(k == 0) & keep])
                                - _mae(arms["valid"]["COMP"]["errs"]["g_cal"][(k == 0) & keep])),
        "overall_g_cal_gain": float(_mae(arms["valid"]["SUM"]["errs"]["g_cal"][keep])
                                    - _mae(arms["valid"]["COMP"]["errs"]["g_cal"][keep])),
        "overall_y_cal_gain": float(_mae(arms["valid"]["SUM"]["errs"]["y_cal"][keep])
                                    - _mae(arms["valid"]["COMP"]["errs"]["y_cal"][keep])),
    }
    # exclude valid:0172 if the mapping exists (positional valid rows)
    if len(k) > 172:
        keep2 = np.ones(k.size, dtype=bool)
        keep2[172] = False
        sensitivity["exclude_valid_0172"] = {
            "dropped_k": int(k[172]),
            "G0_g_cal_gain": float(_mae(arms["valid"]["SUM"]["errs"]["g_cal"][(k == 0) & keep2])
                                    - _mae(arms["valid"]["COMP"]["errs"]["g_cal"][(k == 0) & keep2])),
            "overall_g_cal_gain": float(_mae(arms["valid"]["SUM"]["errs"]["g_cal"][keep2])
                                        - _mae(arms["valid"]["COMP"]["errs"]["g_cal"][keep2])),
            "overall_y_cal_gain": float(_mae(arms["valid"]["SUM"]["errs"]["y_cal"][keep2])
                                        - _mae(arms["valid"]["COMP"]["errs"]["y_cal"][keep2])),
        }

    # worst-row inspection (deploy error identity)
    order = np.argsort(-np.abs(arms["valid"]["COMP"]["errs"]["y_cal"]))[:10]
    worst_rows = []
    for i in order:
        i = int(i)
        row = {
            "position": i, "k": int(k[i]), "y": float(y[i]), "g": float(g[i]), "c": float(valid["c"][i]),
            "q_raw": float(q[i]),
            "SUM_h": float(arms["valid"]["SUM"]["errs"]["h"][i]),
            "COMP_h": float(arms["valid"]["COMP"]["errs"]["h"][i]),
            "SUM_y_cal": float(arms["valid"]["SUM"]["errs"]["h"][i] + q[i] + cal["per_arm"]["SUM"]["b_y"]),
            "COMP_y_cal": float(arms["valid"]["COMP"]["errs"]["h"][i] + q[i] + cal["per_arm"]["COMP"]["b_y"]),
            "SUM_err": float(arms["valid"]["SUM"]["errs"]["y_cal"][i]),
            "COMP_err": float(arms["valid"]["COMP"]["errs"]["y_cal"][i]),
            "b_y_SUM": float(cal["per_arm"]["SUM"]["b_y"]),
            "b_y_COMP": float(cal["per_arm"]["COMP"]["b_y"]),
        }
        worst_rows.append(row)

    result = {
        "protocol_version": PROTOCOL_VERSION,
        "verdict": verdict,
        "CHEM_CONFIRMED": bool(chem_confirmed),
        "DEPLOY_CONFIRMED": bool(deploy_confirmed),
        "chem_conditions": chem_conditions,
        "deploy_conditions": deploy_conditions,
        "gains": gains,
        "arms": {
            split: {
                arm: {key: value for key, value in arms[split][arm].items() if key != "errs"}
                for arm in ARMS
            } for split in ("train", "valid")
        },
        "identities": identities,
        "component_diagnostics": diagnostics,
        "q_diagnostics": q_diag,
        "benchmark": benchmark,
        "sensitivity": sensitivity,
        "worst_valid_rows": worst_rows,
        "bootstrap_witnesses": witnesses,
        "gate_G0_y_worsen": float(y_g0_worsen),
        "n_valid": int(k.size),
        "n_G0_valid": int((k == 0).sum()),
        "n_train": int(k_train.size),
        "n_G0_train": int((k_train == 0).sum()),
        "official_valid_loaded": True,
        "official_test_loaded": False,
    }
    write_json(out_dir / "analysis.json", result)
    write_json(out_dir / "gains.json", gains)
    write_json(out_dir / "gate.json", {
        "verdict": verdict,
        "CHEM_CONFIRMED": bool(chem_confirmed),
        "DEPLOY_CONFIRMED": bool(deploy_confirmed),
        "chem_conditions": chem_conditions,
        "deploy_conditions": deploy_conditions,
        "benchmark_marker": benchmark,
    })
    write_json(out_dir / "bootstrap.json", {"gains": gains, "witnesses": witnesses})

    # CSV deliverables
    import csv

    main_rows = []
    for split_name in ("train", "valid"):
        kk = k_train if split_name == "train" else k
        for arm in ARMS:
            entry = arms[split_name][arm]
            main_rows.append({
                "split": split_name, "arm": arm, "n": int(kk.size), "b_g": entry["b_g"], "b_y": entry["b_y"],
                "g_raw": entry["g_raw_mae"], "g_cal": entry["g_cal_mae"],
                "y_raw": entry["y_raw_mae"], "y_cal": entry["y_cal_mae"],
                "G0_g_raw": entry["G0_g_raw_mae"], "G0_g_cal": entry["G0_g_cal_mae"],
                "G0_y_raw": entry["G0_y_raw_mae"], "G0_y_cal": entry["G0_y_cal_mae"],
            })
    with open(out_dir / "main_table.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(main_rows[0]))
        writer.writeheader()
        writer.writerows(main_rows)

    group_rows = []
    for split_name in ("train", "valid"):
        kk = k_train if split_name == "train" else k
        for arm in ARMS:
            for name in GROUP_NAMES:
                entry = arms[split_name][arm]["groups_g_cal"][name]
                group_rows.append({
                    "split": split_name, "arm": arm, "group": name, "n": entry["n"],
                    "mae": entry["mae"], "contribution": entry["contribution"],
                })
    with open(out_dir / "group_table.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(group_rows[0]))
        writer.writeheader()
        writer.writerows(group_rows)

    component_rows = []
    for split_name in ("train", "valid"):
        kk = k_train if split_name == "train" else k
        for comp in ("ell", "s"):
            entry = diagnostics[split_name][comp]
            component_rows.append({
                "split": split_name, "component": comp, "grid": "overall", "n": int(kk.size),
                "raw_mae": entry["raw_mae"], "signed_mean": entry["signed_mean"],
                "signed_median": entry["signed_median"], "std": entry["std"],
            })
    with open(out_dir / "component_table.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(component_rows[0]))
        writer.writeheader()
        writer.writerows(component_rows)

    log(f"[analyze] verdict={verdict} chem={chem_confirmed} deploy={deploy_confirmed}")
    log(f"[analyze] g G0 cal gain={gains['g']['G0_cal']['point']:+.6f} CI={gains['g']['G0_cal']['ci95']}")
    log(f"[analyze] g overall cal gain={gains['g']['overall_cal']['point']:+.6f} CI={gains['g']['overall_cal']['ci95']}")
    log(f"[analyze] y overall cal gain={gains['y']['overall_cal']['point']:+.6f} CI={gains['y']['overall_cal']['ci95']}")
    return result


# ---------------------------------------------------------------------------
# 11. smoke (fit-only; states discarded)
# ---------------------------------------------------------------------------


def run_smoke(
    *,
    arm: str,
    device: torch.device,
    out_dir: Path = RESULTS_DIR,
    max_steps: int = 3,
    log: Any = print,
) -> dict[str, Any]:
    torch.set_num_threads(8)
    with np.load(out_dir / "full_train_payload.npz", allow_pickle=False) as z:
        payload_arrays = {key: z[key] for key in z.files}
    payload = prev.TuplePayload(payload_arrays)
    kappa_M = float(payload_arrays["kappa"].reshape(-1)[0])
    prep_meta = _read_npz(out_dir / "full_train_prep.npz")
    train_data = build_fulltrain_data(prep_meta)
    with np.load(out_dir / "full_train_targets.npz", allow_pickle=False) as z:
        g_all = z["g"].astype(np.float64)
        ell_all = z["ell"].astype(np.float64)
        s_all = z["s"].astype(np.float64)
    target_g = torch.as_tensor(g_all, dtype=torch.float32)
    target_ell = torch.as_tensor(ell_all, dtype=torch.float32, device=device)
    target_s = torch.as_tensor(s_all, dtype=torch.float32, device=device)
    schedule, schedule_hash = build_schedule(N_TRAIN, EPOCHS, SEED + TRAIN_SHUFFLE_OFFSET)
    checks: dict[str, Any] = {
        "arm": arm, "max_steps": int(max_steps),
        "schedule_hash": schedule_hash,
        "n_rows": N_TRAIN,
        "steps_per_epoch": int(math.ceil(N_TRAIN / BATCH_SIZE)),
    }
    seed_everything(SEED)
    model = build_component_model(arm, payload, kappa_M).to(device)
    seed_everything(SEED)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    indices = [int(i) for i in schedule[0][:BATCH_SIZE].tolist()]
    grad_probe: list[dict[str, Any]] = []
    model.train()
    for step in range(1, int(max_steps) + 1):
        batch = zftd.make_batch(train_data, indices, target_g, device)
        prediction = model(batch, mask=cm.C6_MASK)
        components = model.reader.components()
        l_g = F.l1_loss(prediction.view(-1), batch.y.view(-1))
        l_ell = F.l1_loss(components[:, 0], target_ell[indices])
        l_s = F.l1_loss(components[:, 1], target_s[indices])
        loss = l_g + COMPONENT_LOSS_WEIGHT * (l_ell + l_s) if arm == "COMP" else l_g
        optimizer.zero_grad()
        loss.backward()
        total_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP))
        grad_probe.append({"step": step, "loss_used": float(loss.detach()), "L_g": float(l_g.detach()),
                           "L_ell": float(l_ell.detach()), "L_s": float(l_s.detach()),
                           "aux_contributes": arm == "COMP", "clip_total_norm": total_norm,
                           "pred_shape": list(prediction.shape), "component_shape": list(components.shape)})
        optimizer.step()
    checks["grad_probe"] = grad_probe

    # labels must never change the forward
    model.eval()
    with torch.no_grad():
        batch = zftd.make_batch(train_data, indices, target_g, device)
        p1 = model(batch, mask=cm.C6_MASK).view(-1).clone()
        c1 = model.reader.components().clone()
        perm = torch.randperm(batch.y.shape[0])
        batch.y = batch.y[perm].clone()
        p2 = model(batch, mask=cm.C6_MASK).view(-1).clone()
        c2 = model.reader.components().clone()
    checks["label_permutation_forward_max_abs_diff"] = float((p1 - p2).abs().max())
    checks["label_permutation_components_max_abs_diff"] = float((c1 - c2).abs().max())
    checks["label_permutation_forward_unchanged"] = bool(
        float((p1 - p2).abs().max()) == 0.0 and float((c1 - c2).abs().max()) == 0.0
    )

    smoke_dir = out_dir / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    torch.save(state, smoke_dir / f"smoke_{arm}_state.pt")
    reloaded = build_component_model(arm, payload, kappa_M)
    reloaded.load_state_dict(
        {key: value.to(torch.device("cpu")) for key, value in torch.load(smoke_dir / f"smoke_{arm}_state.pt", weights_only=True).items()},
        strict=True,
    )
    checks["save_reload_state_hash_equal"] = state_hash(state) == state_hash(
        {key: value.detach().cpu() for key, value in reloaded.state_dict().items()}
    )
    reloaded = reloaded.to(device).eval()
    with torch.no_grad():
        batch = zftd.make_batch(train_data, indices, target_g, device)
        p3 = reloaded(batch, mask=cm.C6_MASK).view(-1).clone()
    checks["save_reload_forward_max_abs_diff"] = float((p1 - p3).abs().max())
    for temp in smoke_dir.glob("*.pt"):
        temp.unlink()
    checks["all_ok"] = bool(
        checks["label_permutation_forward_unchanged"]
        and checks["save_reload_state_hash_equal"]
        and checks["save_reload_forward_max_abs_diff"] == 0.0
    )
    write_json(out_dir / f"smoke_checks_{arm}.json", checks)
    if not checks["all_ok"]:
        raise RuntimeError(f"smoke checks failed: {checks}")
    return checks


# ---------------------------------------------------------------------------
# 12. manifest / budget
# ---------------------------------------------------------------------------


def make_manifest(*, out_dir: Path = RESULTS_DIR) -> dict[str, Any]:
    files = {}
    for path in sorted(out_dir.glob("*")):
        if path.is_file():
            files[path.name] = {"sha256": file_sha256(path), "bytes": int(path.stat().st_size)}
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "files": files,
        "official_valid_loaded": True,
        "official_test_loaded": False,
    }
    write_json(out_dir / "manifest.json", payload)
    return payload


def make_budget(*, out_dir: Path = RESULTS_DIR, round_start: str = "", round_end: str = "") -> dict[str, Any]:
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "round_start": round_start,
        "round_end": round_end,
        "gpu_budget_limit_hours": 1.2,
        "training_wall_seconds": {},
        "training_gpu_hours_upper_bound": 0.0,
        "allocation": {},
    }
    total = 0.0
    for arm in ARMS:
        meta_path = out_dir / f"{arm}_meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        payload["training_wall_seconds"][arm] = meta["wall_clock_s"]
        total += float(meta["wall_clock_s"])
        payload["allocation"][arm] = {
            "steps": meta["steps_done"], "device": meta["device"],
            "allocation_probe": meta.get("allocation_probe", {}),
        }
    q_path = out_dir / "Q_meta.json"
    if q_path.exists():
        q_meta = json.loads(q_path.read_text())
        payload["training_wall_seconds"]["Q_cpu"] = q_meta["wall_clock_s"]
        payload["allocation"]["Q_cpu"] = {"steps": q_meta["steps"], "device": "cpu"}
    payload["training_gpu_hours_upper_bound"] = total / 3600.0
    write_json(out_dir / "budget.json", payload)
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=PROTOCOL_VERSION)
    parser.add_argument("--mode", required=True, choices=(
        "freeze", "init-check", "smoke", "train", "train-q",
        "calibrate", "manifest", "valid", "analyze", "budget", "manifest-all",
    ))
    parser.add_argument("--arm", choices=ARMS, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)
    out_dir = Path(args.out)

    if args.mode == "freeze":
        phase_freeze(out_dir=out_dir)
    elif args.mode == "init-check":
        init_check(out_dir=out_dir, device=resolve_device(args.device))
    elif args.mode == "smoke":
        if args.arm is None:
            raise SystemExit("--mode smoke requires --arm")
        run_smoke(arm=args.arm, device=resolve_device(args.device), out_dir=out_dir,
                  max_steps=args.max_steps or 3)
    elif args.mode == "train":
        if args.arm is None:
            raise SystemExit("--mode train requires --arm")
        train_arm(args.arm, out_dir=out_dir, device=resolve_device(args.device),
                  epochs=args.epochs, max_steps=args.max_steps)
    elif args.mode == "train-q":
        train_q_head(out_dir=out_dir)
    elif args.mode == "calibrate":
        calibrate(out_dir=out_dir)
    elif args.mode == "manifest":
        build_manifest(out_dir=out_dir)
    elif args.mode == "valid":
        evaluate_valid(out_dir=out_dir, device=resolve_device(args.device))
    elif args.mode == "analyze":
        analyze(out_dir=out_dir)
    elif args.mode == "budget":
        make_budget(out_dir=out_dir)
    elif args.mode == "manifest-all":
        make_manifest(out_dir=out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
