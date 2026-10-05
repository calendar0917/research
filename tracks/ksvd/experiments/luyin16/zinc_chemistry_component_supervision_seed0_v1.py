"""ZINC chemistry-component supervision (seed 0): matched SUM vs COMP.

Round: ``zinc_chemistry_component_supervision_seed0_v1``.

Single intervention, matched environment, identical fresh-fold M input /
skeleton / total objective ``g``.  The reader's last linear (39 -> 1) is
replaced by a 39 -> 2 output ``[hat_ell, hat_s]`` with both rows/bias equal to
half the original, so ``hat_g = hat_ell + hat_s`` reproduces the original
fresh-fold M prediction exactly at construction.  Exactly two formal arms:

* ``SUM``  -- ``L = L_g`` (the original total-MAE objective);
* ``COMP`` -- ``L = L_g + 0.5 * (L_ell + L_s)`` with
  ``ell = (logP - MU_LOGP) / sigma_logP`` and ``s = g - ell``.

Both arms start from the same untrained fresh-fold M init, share every state
key/value, and have identical parameter counts (297,539 = 297,499 + 39 + 1).
Only the loss geometry / identifiable-component supervision differs.  The
extra component labels are train-side supervision only and never enter the
forward.

Sources (read-only): the frozen fresh-fold artifacts under
``tracks/ksvd/results/zinc_local_tuple_fresh_fold_replication_seed0_v1/`` and
the train-only raw label reconstruction of the fresh-fold runner.

Official valid/test are never loaded, instantiated, predicted or scored.

Usage::

    uv run python -m tracks.ksvd.experiments.luyin16.\
zinc_chemistry_component_supervision_seed0_v1 --targets
    ... --init-check
    ... --smoke --arm SUM --device cpu
    ... --train --arm SUM --device cuda
    ... --train --arm COMP --device cuda
    ... --analyze
    ... --budget
    ... --manifest
"""

from __future__ import annotations

import argparse
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
from tracks.ksvd.experiments.luyin16 import (
    zinc_local_tuple_dictionary_joint_vs_independent_seed0_v1 as prev,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_local_tuple_dictionary_vs_mlp_seed0_v1 as mlpmod,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_local_tuple_fresh_fold_replication_seed0_v1 as zfr,
)
from tracks.ksvd.experiments.luyin16 import zinc_full_cycle_target_decomposition_v1 as zftd
from tracks.ksvd.experiments.luyin16 import zinc_joint_dictionary_decision_v1 as zjd
from tracks.ksvd.experiments.luyin16 import zinc_task_dictionary_and_cycle_witness_seed0_v1 as zw

PROTOCOL_VERSION = "zinc-chemistry-component-supervision-seed0-v1"
RESULT_SLUG = "zinc_chemistry_component_supervision_seed0_v1"
RESULTS_DIR = zjd.TRACK_ROOT / "results" / RESULT_SLUG
SOURCE_DIR = zfr.RESULTS_DIR  # frozen fresh-fold replication artifacts (read-only)

# ---- frozen recipe (identical for both arms) ------------------------------
SEED = zfr.SEED
EPOCHS = zfr.EPOCHS
LR = zfr.LR
WEIGHT_DECAY = zfr.WEIGHT_DECAY
GRAD_CLIP = zfr.GRAD_CLIP
BATCH_SIZE = zfr.BATCH_SIZE
SOUP_EPOCHS = zfr.SOUP_EPOCHS
TRAIN_SHUFFLE_OFFSET = zfr.TRAIN_SHUFFLE_OFFSET
LOG_EPOCHS = zfr.LOG_EPOCHS
N_FIT = zfr.N_FIT
N_DEV = zfr.N_DEV
REPLAY_TOL = zfr.REPLAY_TOL
MU_LOGP = zfr.MU_LOGP
FROZEN_SCHEDULE_SHA = zfr.FROZEN_SCHEDULE_SHA
NEW_FIT_SHA = zfr.NEW_FIT_SHA
NEW_DEV_SHA = zfr.NEW_DEV_SHA

ARMS = ("SUM", "COMP")
#: pre-fixed (never scanned / adapted / dev-tuned)
COMPONENT_LOSS_WEIGHT = 0.5

# ---- statistics / gates (frozen before the two formal runs) ---------------
BOOT_SEED = 20261008
N_BOOT = 1000
PERF_DELTA = 0.003
EQUIV_BAND = 0.003

# ---- pre-frozen component-diagnostic thresholds ---------------------------
SA_EPS_MAE_THRESHOLD = 1.0e-3
SA_EPS_MAX_THRESHOLD = 1.0e-2
FIT_ADEQUATE_RATIO = 0.20
DEGENERATE_CONST_MAE = 1.0e-6
GAP_RATIO_MARK = 1.5

EXPECTED_PARAMETERS = mlpmod.EXPECTED_TOTAL_PARAMETERS + 39 + 1  # 297,539
GROUP_NAMES = zfr.GROUP_NAMES

jsonable = zfr.jsonable
write_json = zfr.write_json
file_sha256 = zfr.file_sha256
state_hash = zfr.state_hash
seed_everything = zfr.seed_everything
array_sha256 = zfr.array_sha256
group_masks = zw.group_masks
resolve_device = zfr.resolve_device
_hash_bytes = zfr._hash_bytes


# ---------------------------------------------------------------------------
# 0. allocation provenance (physical GPU identity, not the local CUDA index)
# ---------------------------------------------------------------------------


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
    exactly.  The standard forward returns the ``(n,)`` total ``g``; the two
    ``(n,)`` component outputs of the *same* body computation are cached on
    ``last_components`` (never a second forward pass).
    """

    def __init__(self, original: nn.Sequential) -> None:
        super().__init__()
        if not isinstance(original, nn.Sequential) or not isinstance(original[-1], nn.Linear):
            raise RuntimeError("unexpected reader layout")
        if int(original[-1].out_features) != 1:
            raise RuntimeError(f"reader last layer is {int(original[-1].out_features)}-wide, expected 1")
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
        "reader_output_parameters": int(model.reader.net[-1].weight.numel() + model.reader.net[-1].bias.numel()),
    }


# ---------------------------------------------------------------------------
# 2. loading frozen fresh objects + building the component targets cache
# ---------------------------------------------------------------------------


def _artifact_paths(out_dir: Path) -> dict[str, Path]:
    return {
        "targets_npz": out_dir / "component_targets.npz",
        "labels_json": out_dir / "component_labels.json",
        "input_manifest": out_dir / "input_manifest.json",
        "init_identity": out_dir / "init_identity.json",
        "alloc_targets": out_dir / "alloc_targets.json",
    }


def load_source_objects() -> dict[str, Any]:
    """Load the frozen fresh-fold artifacts (read-only) with hash checks."""
    return zfr.load_fresh_objects(SOURCE_DIR)


def load_frozen_constants() -> tuple[dict[str, float], dict[str, np.ndarray]]:
    with np.load(SOURCE_DIR / "fresh_targets.npz", allow_pickle=False) as z:
        names = [str(s) for s in z["constant_names"].tolist()]
        values = z["constants"].astype(np.float64)
        constants = {name: float(value) for name, value in zip(names, values)}
        arrays = {
            "y": z["y"].astype(np.float64),
            "c": z["c"].astype(np.float64),
            "g": z["g"].astype(np.float64),
            "k": z["k"].astype(np.int64),
            "gid": z["gid"].astype(np.int64),
        }
    if list(constants) != ["sigma_logP", "sigma_SA", "mu_SA", "sigma_cycle", "mu_cycle"]:
        raise RuntimeError(f"unexpected frozen constant names: {list(constants)}")
    return constants, arrays


def build_component_targets(
    rows: np.ndarray,
    constants: Mapping[str, float],
    source_arrays: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """``ell``, ``s``, ``s_SA``, ``epsilon`` on the frozen fresh-fold targets."""
    g = np.asarray(source_arrays["g"], np.float64)
    y = np.asarray(source_arrays["y"], np.float64)
    c = np.asarray(source_arrays["c"], np.float64)
    k = np.asarray(source_arrays["k"], np.int64)
    sigma_logP = float(constants["sigma_logP"])
    sigma_SA = float(constants["sigma_SA"])
    mu_SA = float(constants["mu_SA"])
    sigma_cycle = float(constants["sigma_cycle"])
    mu_cycle = float(constants["mu_cycle"])
    ell = (rows["logP"].astype(np.float64) - MU_LOGP) / sigma_logP
    s = g - ell
    s_sa = (rows["SA"].astype(np.float64) - mu_SA) / sigma_SA
    epsilon = s - s_sa
    checks: dict[str, Any] = {
        "n_rows": int(g.shape[0]),
        "g_equals_y_minus_c_max_abs": float(np.max(np.abs(g - (y - c)))),
        "c_equals_snap_max_abs": float(np.max(np.abs(c - (k.astype(np.float64) - mu_cycle) / sigma_cycle))),
        "g_equals_ell_plus_s_max_abs": float(np.max(np.abs(g - (ell + s)))),
        "all_finite": bool(
            np.isfinite(ell).all() and np.isfinite(s).all() and np.isfinite(s_sa).all() and np.isfinite(epsilon).all()
        ),
        "mu_logP": float(MU_LOGP),
        "sigma_logP": sigma_logP,
        "sigma_SA": sigma_SA,
        "mu_SA": mu_SA,
        "sigma_cycle": sigma_cycle,
        "mu_cycle": mu_cycle,
        "source_g_sha256": array_sha256(g, np.float64),
        "identity_note": "g bytes are the frozen fresh-fold g; ell/s computed in float64",
    }
    if checks["g_equals_y_minus_c_max_abs"] > 1e-12:
        raise RuntimeError("y = g + c identity failed")
    if checks["g_equals_ell_plus_s_max_abs"] > 1e-12:
        raise RuntimeError("g = ell + s identity failed")
    if not checks["all_finite"]:
        raise RuntimeError("non-finite component labels")
    return {"ell": ell, "s": s, "s_SA": s_sa, "epsilon": epsilon, "checks": checks}


def _quantiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {}
    qs = np.percentile(values, [0.0, 1.0, 25.0, 50.0, 75.0, 99.0, 100.0])
    return {
        "min": float(qs[0]),
        "p1": float(qs[1]),
        "p25": float(qs[2]),
        "p50": float(qs[3]),
        "p75": float(qs[4]),
        "p99": float(qs[5]),
        "max": float(qs[6]),
    }


def component_label_diagnostics(
    labels: Mapping[str, np.ndarray],
    k: np.ndarray,
    fold: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Freeze label identity, epsilon stats and the SA-interpretation flag."""
    fit_idx = np.asarray(fold["fit_idx"], np.int64)
    dev_idx = np.asarray(fold["dev_idx"], np.int64)
    k_fit = k[fit_idx]
    eps = np.asarray(labels["epsilon"], np.float64)
    eps_fit = eps[fit_idx]
    eps_dev = eps[dev_idx]
    g0_fit = k_fit == 0

    def eps_summary(values: np.ndarray) -> dict[str, Any]:
        return {
            "n": int(values.size),
            "mae": float(np.mean(np.abs(values))) if values.size else None,
            "max_abs": float(np.max(np.abs(values))) if values.size else None,
            "quantiles": _quantiles(values),
        }

    per_k_fit = {name: eps_summary(eps_fit[group_masks(k_fit)[name]]) for name in GROUP_NAMES}
    per_k_dev = {
        name: eps_summary(eps_dev[group_masks(k[dev_idx])[name]]) for name in GROUP_NAMES
    }
    fit_eps = eps_summary(eps_fit)
    dev_eps = eps_summary(eps_dev)
    g0_eps = eps_summary(eps_fit[g0_fit])
    sa_interpretable = bool(
        g0_eps["mae"] is not None
        and g0_eps["mae"] <= SA_EPS_MAE_THRESHOLD
        and g0_eps["max_abs"] is not None
        and g0_eps["max_abs"] <= SA_EPS_MAX_THRESHOLD
    )
    return {
        "definition": "ell=(logP-MU_LOGP)/sigma_logP; s=g-ell; s_SA=(SA-mu_SA)/sigma_SA; epsilon=s-s_SA",
        "identity_checks": labels["checks"],
        "epsilon_fit": fit_eps,
        "epsilon_dev": dev_eps,
        "epsilon_G0_fit": g0_eps,
        "epsilon_fit_per_k": per_k_fit,
        "epsilon_dev_per_k": per_k_dev,
        "sa_interpretation": {
            "threshold_mae": SA_EPS_MAE_THRESHOLD,
            "threshold_max": SA_EPS_MAX_THRESHOLD,
            "G0_fit_epsilon_mae": g0_eps["mae"],
            "G0_fit_epsilon_max": g0_eps["max_abs"],
            "approx_SA_component_allowed": sa_interpretable,
            "rule": "s is named 'residual chemistry component'; call it 'approx SA' only if the G0-fit epsilon thresholds pass",
        },
        "component_scales": {
            "ell_fit_mean": float(np.mean(labels["ell"][fit_idx])),
            "ell_fit_std": float(np.std(labels["ell"][fit_idx])),
            "s_fit_mean": float(np.mean(labels["s"][fit_idx])),
            "s_fit_std": float(np.std(labels["s"][fit_idx])),
            "s_SA_fit_mean": float(np.mean(labels["s_SA"][fit_idx])),
            "s_SA_fit_std": float(np.std(labels["s_SA"][fit_idx])),
        },
    }


def phase_targets(*, out_dir: Path = RESULTS_DIR, log: Any = print) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    fresh = load_source_objects()
    fold = fresh["fold"]
    if _hash_bytes(fold["fit_idx"].tobytes()) != NEW_FIT_SHA or _hash_bytes(fold["dev_idx"].tobytes()) != NEW_DEV_SHA:
        raise RuntimeError("fresh fold hashes changed")
    constants, source_arrays = load_frozen_constants()
    rows, gid, label_checks = zfr.load_train_only_raw_rows()
    if not np.array_equal(gid, source_arrays["gid"]):
        raise RuntimeError("canonical group id stream differs from the frozen targets")
    labels = build_component_targets(rows, constants, source_arrays)
    diagnostics = component_label_diagnostics(labels, source_arrays["k"], fold)

    k = source_arrays["k"]
    np.savez_compressed(
        out_dir / "component_targets.npz",
        ell=labels["ell"],
        s=labels["s"],
        s_SA=labels["s_SA"],
        epsilon=labels["epsilon"],
        g=source_arrays["g"],
        y=source_arrays["y"],
        c=source_arrays["c"],
        k=k,
        gid=gid,
        constants=np.asarray([constants[name] for name in
                              ("sigma_logP", "sigma_SA", "mu_SA", "sigma_cycle", "mu_cycle")], np.float64),
        constant_names=np.asarray(["sigma_logP", "sigma_SA", "mu_SA", "sigma_cycle", "mu_cycle"]),
        mu_logP=np.asarray(MU_LOGP, np.float64),
    )
    write_json(out_dir / "component_labels.json", diagnostics)
    write_json(
        out_dir / "input_manifest.json",
        {
            "protocol_version": PROTOCOL_VERSION,
            "sources": {
                "fresh_fold_npz": {"path": str(SOURCE_DIR / "fresh_fold.npz"), "sha256": file_sha256(SOURCE_DIR / "fresh_fold.npz")},
                "fresh_targets_npz": {"path": str(SOURCE_DIR / "fresh_targets.npz"), "sha256": file_sha256(SOURCE_DIR / "fresh_targets.npz")},
                "fresh_tuple_payload_npz": {"path": str(SOURCE_DIR / "fresh_tuple_payload.npz"), "sha256": file_sha256(SOURCE_DIR / "fresh_tuple_payload.npz")},
                "fresh_prep_npz": {"path": str(SOURCE_DIR / "fresh_prep.npz"), "sha256": file_sha256(SOURCE_DIR / "fresh_prep.npz")},
                "fresh_manifest_json": {"path": str(SOURCE_DIR / "fresh_manifest.json"), "sha256": file_sha256(SOURCE_DIR / "fresh_manifest.json")},
                "train_label_csv": {"path": str(zfr.TRAIN_LABEL_CSV), "sha256": file_sha256(zfr.TRAIN_LABEL_CSV)},
                "gvae_full_properties_npz": {"path": str(zfr.GVEA_PROPS_NPZ), "sha256": file_sha256(zfr.GVEA_PROPS_NPZ)},
                "handoff_train": {"path": str(zfr.HANDOFF_TRAIN), "sha256": file_sha256(zfr.HANDOFF_TRAIN)},
            },
            "frozen_constants": constants,
            "mu_logP": MU_LOGP,
            "label_source_checks": label_checks,
            "official_valid_loaded": False,
            "official_test_loaded": False,
        },
    )
    write_json(out_dir / "alloc_targets.json", allocation_probe())
    log(
        f"[targets] g=ell+s maxdiff={labels['checks']['g_equals_ell_plus_s_max_abs']:.2e} "
        f"eps_G0_fit_mae={diagnostics['epsilon_G0_fit']['mae']:.4g} "
        f"approx_SA={diagnostics['sa_interpretation']['approx_SA_component_allowed']}"
    )
    return {"checks": labels["checks"], "diagnostics": diagnostics}


# ---------------------------------------------------------------------------
# 3. init identity: SUM/COMP identical, initial sum equals the original M
# ---------------------------------------------------------------------------


def init_check(*, out_dir: Path = RESULTS_DIR, device: torch.device | None = None, log: Any = print) -> dict[str, Any]:
    device = device or torch.device("cpu")
    torch.set_num_threads(8)
    fresh = load_source_objects()
    payload = prev.TuplePayload(fresh["payload_arrays"])
    kappa_M = float(fresh["kappa"]["kappa_M"])

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
    shared = sorted(key for key in orig_state if key != "reader.net.4.weight" and key != "reader.net.4.bias")
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

    # real data forward: initial SUM total must equal the original M output
    prep_meta, fit_data, dev_data = zfr.build_prepared_data(fresh, verify_prep=True)
    g = np.asarray(fresh["targets"]["g"], np.float64)
    g_fit = g[fresh["fold"]["fit_idx"]]
    target = torch.as_tensor(g_fit, dtype=torch.float32)
    schedule, schedule_hash = zw.build_schedule(len(fit_data), EPOCHS, SEED + TRAIN_SHUFFLE_OFFSET)
    index_list = [int(i) for i in schedule[0][:128].tolist()]
    batch = zftd.make_batch(fit_data, index_list, target, device)

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
        "schedule_hash": schedule_hash,
        "schedule_matches_frozen": schedule_hash == FROZEN_SCHEDULE_SHA,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    checks["all_ok"] = bool(
        max_abs == 0.0
        and split_ok
        and checks["parameter_audit"]["total_parameters"] == EXPECTED_PARAMETERS
        and max_sum_diff <= 1e-5
        and max_arm_diff == 0.0
        and checks["schedule_matches_frozen"]
    )
    write_json(out_dir / "init_identity.json", checks)
    log(f"[init] all_ok={checks['all_ok']} sum_vs_M={max_sum_diff:.2e} params={checks['parameter_audit']['total_parameters']}")
    if not checks["all_ok"]:
        raise RuntimeError(f"init identity failed: {checks}")
    return checks


# ---------------------------------------------------------------------------
# 4. training
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
    source_dir: Path = SOURCE_DIR,
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
    fresh = zfr.load_fresh_objects(source_dir)
    fold = fresh["fold"]
    payload = prev.TuplePayload(fresh["payload_arrays"])
    kappa_M = float(fresh["kappa"]["kappa_M"])
    g = np.asarray(fresh["targets"]["g"], np.float64)

    with np.load(out_dir / "component_targets.npz", allow_pickle=False) as z:
        ell_all = z["ell"].astype(np.float64)
        s_all = z["s"].astype(np.float64)
    fit_idx = fold["fit_idx"]
    g_fit = g[fit_idx]
    ell_fit = ell_all[fit_idx]
    s_fit = s_all[fit_idx]
    if not (np.array_equal(g_fit + 0.0, g_fit) and np.max(np.abs(g_fit - (ell_fit + s_fit))) <= 1e-12):
        raise RuntimeError("g != ell + s on fit rows")

    target_g = torch.as_tensor(g_fit, dtype=torch.float32)
    target_ell = torch.as_tensor(ell_fit, dtype=torch.float32, device=device)
    target_s = torch.as_tensor(s_fit, dtype=torch.float32, device=device)

    prep_meta, fit_data, dev_data = zfr.build_prepared_data(fresh, verify_prep=True)
    schedule, schedule_hash = zw.build_schedule(len(fit_data), int(epochs), SEED + TRAIN_SHUFFLE_OFFSET)
    if int(epochs) == EPOCHS and schedule_hash != FROZEN_SCHEDULE_SHA:
        raise RuntimeError(f"schedule hash {schedule_hash} != frozen {FROZEN_SCHEDULE_SHA}")

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
    probe_log: list[dict[str, Any]] = []
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
            gid_stream.update(np.asarray(fresh["targets"]["gid"][fit_idx[index_list]], np.int64).tobytes())
            batch = zftd.make_batch(fit_data, index_list, target_g, device)
            if tuple(batch.y.shape) != (len(index_list),):
                raise RuntimeError("true g target is not (batch,)")
            prediction = model(batch, mask=cm.C6_MASK)
            if tuple(prediction.shape) != (len(index_list),):
                raise RuntimeError("standard forward must return (batch,) total g")
            components = model.reader.components()
            if tuple(components.shape) != (len(index_list), 2):
                raise RuntimeError("reader component output is not (batch, 2)")
            # one body computation produced both the total and the two components
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
            if epoch in LOG_EPOCHS and n_steps == 0:
                probe_log.append(_probe(model, arm, epoch, 1, total_norm))
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
            "g": "g = y - c (fresh-fold fit-only constants)",
            "ell": "(logP - MU_LOGP) / sigma_logP",
            "s": "g - ell (residual chemistry component)",
            "component_labels_are_train_only": True,
        },
        "seed": SEED,
        "epochs": int(epochs),
        "steps_done": int(steps_done),
        "steps_expected": int(epochs) * math.ceil(len(fit_data) / BATCH_SIZE),
        "stopped_reason": stopped_reason,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "grad_clip": GRAD_CLIP,
        "batch_size": BATCH_SIZE,
        "soup_epochs": members if max_steps is None else [],
        "n_fit": int(len(fit_data)),
        "n_dev": int(len(dev_data)),
        "kappa_M": kappa_M,
        "fold": {
            "source_dir": str(source_dir),
            "fit_idx_sha256": _hash_bytes(fold["fit_idx"].tobytes()),
            "dev_idx_sha256": _hash_bytes(fold["dev_idx"].tobytes()),
        },
        "target": {
            "source": str(out_dir / "component_targets.npz"),
            "g_sha256": array_sha256(g, np.float64),
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
        "probe_log": probe_log,
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
        fit_sum, fit_comp = evaluate_state_components(replay, fit_data, device)
        dev_sum, dev_comp = evaluate_state_components(replay, dev_data, device)
        raw_sums[f"{state_name}_fit"] = fit_sum
        raw_sums[f"{state_name}_dev"] = dev_sum
        raw_comps[f"{state_name}_fit"] = fit_comp
        raw_comps[f"{state_name}_dev"] = dev_comp
    np.savez_compressed(
        out_dir / f"{arm}_raw_predictions.npz",
        **{key: value.astype(np.float32) for key, value in raw_sums.items()},
    )
    np.savez_compressed(
        out_dir / f"{arm}_component_predictions.npz",
        **{key: value.astype(np.float32) for key, value in raw_comps.items()},
    )

    b_soup = float(np.median(g_fit - raw_sums["raw_soup_fit"]))
    result["calibration_b"] = {
        "init": float(np.median(g_fit - raw_sums["init_fit"])),
        "last": float(np.median(g_fit - raw_sums["last_fit"])),
        "raw_soup": b_soup,
    }

    replay = build_component_model(arm, payload, kappa_M)
    replay.load_state_dict({key: value.to(torch.device("cpu")) for key, value in soup_state.items()}, strict=True)
    batch_data = fit_data[:128]
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
    log(f"[{arm}] done steps={steps_done} wall={result['wall_clock_s']:.1f}s b={b_soup:.6f}")
    return result


def _probe(model: nn.Module, arm: str, epoch: int, step: int, total_norm: float) -> dict[str, Any]:
    entry: dict[str, Any] = {"epoch": int(epoch), "step_in_epoch": int(step), "clip_total_norm": float(total_norm)}
    w_loc = model.local_tuple.W_loc
    a_raw = model.local_tuple.A_raw
    entry["W_loc_grad_norm"] = float(w_loc.grad.norm()) if w_loc.grad is not None else None
    entry["A_grad_norm"] = float(a_raw.grad.norm()) if a_raw.grad is not None else None
    entry["W_loc_norm"] = float(w_loc.detach().norm())
    entry["A_norm"] = float(a_raw.detach().norm())
    entry["health"] = {key: value for key, value in model.local_tuple.last_stats.items()}
    return entry


# ---------------------------------------------------------------------------
# 5. smoke (fit-only; states discarded)
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
    fresh = load_source_objects()
    payload = prev.TuplePayload(fresh["payload_arrays"])
    kappa_M = float(fresh["kappa"]["kappa_M"])
    fold = fresh["fold"]
    prep_meta, fit_data, dev_data = zfr.build_prepared_data(fresh, verify_prep=True)
    g = np.asarray(fresh["targets"]["g"], np.float64)
    with np.load(out_dir / "component_targets.npz", allow_pickle=False) as z:
        ell_all = z["ell"].astype(np.float64)
        s_all = z["s"].astype(np.float64)
    g_fit = g[fold["fit_idx"]]
    ell_fit = ell_all[fold["fit_idx"]]
    s_fit = s_all[fold["fit_idx"]]
    target_g = torch.as_tensor(g_fit, dtype=torch.float32)
    target_ell = torch.as_tensor(ell_fit, dtype=torch.float32, device=device)
    target_s = torch.as_tensor(s_fit, dtype=torch.float32, device=device)
    schedule, schedule_hash = zw.build_schedule(len(fit_data), EPOCHS, SEED + TRAIN_SHUFFLE_OFFSET)
    checks: dict[str, Any] = {
        "arm": arm,
        "max_steps": int(max_steps),
        "schedule_hash": schedule_hash,
        "schedule_matches_frozen": schedule_hash == FROZEN_SCHEDULE_SHA,
    }
    seed_everything(SEED)
    model = build_component_model(arm, payload, kappa_M).to(device)
    seed_everything(SEED)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    indices = [int(i) for i in schedule[0][:BATCH_SIZE].tolist()]
    grad_probe: list[dict[str, Any]] = []
    component_grads: list[dict[str, Any]] = []
    model.train()
    for step in range(1, int(max_steps) + 1):
        batch = zftd.make_batch(fit_data, indices, target_g, device)
        prediction = model(batch, mask=cm.C6_MASK)
        components = model.reader.components()
        l_g = F.l1_loss(prediction.view(-1), batch.y.view(-1))
        l_ell = F.l1_loss(components[:, 0], target_ell[indices])
        l_s = F.l1_loss(components[:, 1], target_s[indices])
        if arm == "COMP":
            loss = l_g + COMPONENT_LOSS_WEIGHT * (l_ell + l_s)
        else:
            loss = l_g
        optimizer.zero_grad()
        loss.backward()
        # the SUM loss must not send gradient into the component MAE objectives
        component_grads.append(
            {
                "loss_used": float(loss.detach()),
                "L_g": float(l_g.detach()),
                "L_ell": float(l_ell.detach()),
                "L_s": float(l_s.detach()),
                "aux_contributes": arm == "COMP",
            }
        )
        total_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP))
        grad_probe.append(_probe(model, arm, 1, step, total_norm))
        optimizer.step()
    checks["grad_probe"] = grad_probe
    checks["loss_probe"] = component_grads

    # labels must never change the forward
    model.eval()
    with torch.no_grad():
        batch = zftd.make_batch(fit_data, indices, target_g, device)
        p1 = model(batch, mask=cm.C6_MASK).view(-1).clone()
        c1 = model.reader.components().clone()
        perm = torch.randperm(batch.y.shape[0])
        batch.y = batch.y[perm].clone()
        p2 = model(batch, mask=cm.C6_MASK).view(-1).clone()
        c2 = model.reader.components().clone()
    checks["label_permutation_forward_max_abs_diff"] = float((p1 - p2).abs().max())
    checks["label_permutation_components_max_abs_diff"] = float((c1 - c2).abs().max())
    checks["label_permutation_forward_unchanged"] = bool(
        checks["label_permutation_forward_max_abs_diff"] == 0.0
        and checks["label_permutation_components_max_abs_diff"] == 0.0
    )

    # save/reload round trip
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
        batch = zftd.make_batch(fit_data, indices, target_g, device)
        p3 = reloaded(batch, mask=cm.C6_MASK).view(-1).clone()
    checks["save_reload_forward_max_abs_diff"] = float((p1 - p3).abs().max())
    for temp in smoke_dir.glob("*.pt"):
        temp.unlink()

    checks["shape_guard_ok"] = True
    checks["all_ok"] = bool(
        checks["schedule_matches_frozen"]
        and checks["label_permutation_forward_unchanged"]
        and checks["save_reload_state_hash_equal"]
        and checks["save_reload_forward_max_abs_diff"] == 0.0
    )
    write_json(out_dir / f"smoke_checks_{arm}.json", checks)
    if not checks["all_ok"]:
        raise RuntimeError(f"smoke checks failed: {checks}")
    return checks


# ---------------------------------------------------------------------------
# 6. analysis
# ---------------------------------------------------------------------------


def _mae(err: np.ndarray) -> float:
    return float(np.mean(np.abs(err))) if err.size else float("nan")


def _signed(values: np.ndarray) -> dict[str, Any]:
    if values.size == 0:
        return {"n": 0}
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
    }


def _component_stats(hat: np.ndarray, target: np.ndarray, k: np.ndarray) -> dict[str, Any]:
    err = hat - target
    out: dict[str, Any] = {
        "raw_mae": _mae(err),
        "signed": _signed(err),
        "quantiles": _quantiles(err),
    }
    table = {}
    for name, mask in group_masks(k).items():
        n = int(mask.sum())
        table[name] = {
            "n": n,
            "raw_mae": _mae(err[mask]) if n else None,
            "signed_mean": float(np.mean(err[mask])) if n else None,
            "abs_sum_over_n": float(np.abs(err[mask]).sum() / max(err.size, 1)) if n else None,
        }
    out["per_k"] = table
    return out


def _constant_mae(target: np.ndarray, median_value: float) -> float:
    return float(np.mean(np.abs(target - median_value))) if target.size else float("nan")


def analyze(*, out_dir: Path = RESULTS_DIR, device: torch.device | None = None, log: Any = print) -> dict[str, Any]:
    torch.set_num_threads(8)
    device = device or torch.device("cpu")
    fresh = load_source_objects()
    fold = fresh["fold"]
    fit_idx = fold["fit_idx"]
    dev_idx = fold["dev_idx"]
    with np.load(out_dir / "component_targets.npz", allow_pickle=False) as z:
        g = z["g"].astype(np.float64)
        k = z["k"].astype(np.int64)
        ell = z["ell"].astype(np.float64)
        s = z["s"].astype(np.float64)
        constants = {str(n): float(v) for n, v in zip(z["constant_names"].tolist(), z["constants"].tolist())}
    g_fit, g_dev = g[fit_idx], g[dev_idx]
    k_fit, k_dev = k[fit_idx], k[dev_idx]
    ell_fit, ell_dev = ell[fit_idx], ell[dev_idx]
    s_fit, s_dev = s[fit_idx], s[dev_idx]

    sums: dict[str, dict[str, np.ndarray]] = {}
    comps: dict[str, dict[str, np.ndarray]] = {}
    for arm in ARMS:
        with np.load(out_dir / f"{arm}_raw_predictions.npz") as z:
            sums[arm] = {key: np.asarray(z[key], np.float64) for key in z.files}
        with np.load(out_dir / f"{arm}_component_predictions.npz") as z:
            comps[arm] = {key: np.asarray(z[key], np.float64) for key in z.files}

    bias: dict[str, float] = {}
    cal: dict[str, dict[str, np.ndarray]] = {}
    raw: dict[str, dict[str, np.ndarray]] = {}
    for arm in ARMS:
        bias[arm] = float(np.median(g_fit - sums[arm]["raw_soup_fit"]))
        raw[arm] = {"fit": sums[arm]["raw_soup_fit"], "dev": sums[arm]["raw_soup_dev"]}
        cal[arm] = {"fit": raw[arm]["fit"] + bias[arm], "dev": raw[arm]["dev"] + bias[arm]}

    arms: dict[str, Any] = {}
    for arm in ARMS:
        err_fit = g_fit - cal[arm]["fit"]
        err_dev = g_dev - cal[arm]["dev"]
        err_dev_raw = g_dev - raw[arm]["dev"]
        arms[arm] = {
            "bias": bias[arm],
            "fit_overall_cal_mae": _mae(err_fit),
            "fit_overall_raw_mae": _mae(g_fit - raw[arm]["fit"]),
            "dev_overall_raw_mae": _mae(err_dev_raw),
            "dev_overall_cal_mae": _mae(err_dev),
            "dev_G0_raw_mae": _mae(err_dev_raw[k_dev == 0]),
            "dev_G0_cal_mae": _mae(err_dev[k_dev == 0]),
            "fit_groups_cal": zw._metric_table(err_fit, k_fit),
            "dev_groups_cal": zw._metric_table(err_dev, k_dev),
            "err_dev_cal": err_dev,
            "err_dev_raw": err_dev_raw,
        }
        arms[arm]["gap_overall_cal"] = arms[arm]["dev_overall_cal_mae"] - arms[arm]["fit_overall_cal_mae"]
        arms[arm]["gap_G0_cal"] = arms[arm]["dev_G0_cal_mae"] - _mae(err_fit[k_fit == 0])

    err_sum = {
        "G0_cal": arms["SUM"]["err_dev_cal"][k_dev == 0],
        "overall_cal": arms["SUM"]["err_dev_cal"],
        "G0_raw": arms["SUM"]["err_dev_raw"][k_dev == 0],
        "overall_raw": arms["SUM"]["err_dev_raw"],
    }
    err_comp = {
        "G0_cal": arms["COMP"]["err_dev_cal"][k_dev == 0],
        "overall_cal": arms["COMP"]["err_dev_cal"],
        "G0_raw": arms["COMP"]["err_dev_raw"][k_dev == 0],
        "overall_raw": arms["COMP"]["err_dev_raw"],
    }
    gains = zfr._paired_bootstrap(err_sum, err_comp, k_dev, seed=BOOT_SEED, n_boot=N_BOOT)
    witnesses = zfr._bootstrap_witnesses(arms["SUM"]["err_dev_cal"], arms["COMP"]["err_dev_cal"], k_dev)

    gate_conditions = {
        "G0_cal_gain_ge_0.003": bool(gains["G0_cal"]["point"] >= PERF_DELTA),
        "overall_cal_gain_ge_0.003": bool(gains["overall_cal"]["point"] >= PERF_DELTA),
        "G0_cal_ci_lower_gt_0": bool(gains["G0_cal"]["ci95"][0] > 0.0),
    }
    if all(gate_conditions.values()):
        classification = "COMPONENT_SUPERVISION_CANDIDATE"
    elif gains["G0_cal"]["point"] > 0.0 and gains["overall_cal"]["point"] > 0.0:
        classification = "DIRECTIONAL_NOT_CONFIRMED"
    else:
        classification = "NO_CANDIDATE"
    equivalence = bool(
        EQUIV_BAND >= abs(gains["G0_cal"]["ci95"][0]) and EQUIV_BAND >= abs(gains["G0_cal"]["ci95"][1])
        and EQUIV_BAND >= abs(gains["overall_cal"]["ci95"][0]) and EQUIV_BAND >= abs(gains["overall_cal"]["ci95"][1])
        and gains["G0_cal"]["ci95"][0] >= -EQUIV_BAND and gains["G0_cal"]["ci95"][1] <= EQUIV_BAND
        and gains["overall_cal"]["ci95"][0] >= -EQUIV_BAND and gains["overall_cal"]["ci95"][1] <= EQUIV_BAND
    )

    # contribution identities
    identity: dict[str, Any] = {}
    for arm in ARMS:
        table = arms[arm]["dev_groups_cal"]
        contrib_sum = sum(table[name]["contribution"] for name in GROUP_NAMES if table[name]["n"] > 0)
        identity[f"{arm}_contribution_sum_minus_overall"] = float(contrib_sum - table["mae"])
    gain_sum = 0.0
    for name in GROUP_NAMES:
        gain_sum += arms["SUM"]["dev_groups_cal"][name]["contribution"] - arms["COMP"]["dev_groups_cal"][name]["contribution"]
    identity["group_gain_sum_minus_total"] = float(gain_sum - gains["overall_cal"]["point"])

    # ---- COMP component diagnostics (no new fitting) ----------------------
    hat_ell = comps["COMP"]["raw_soup_fit"][:, 0]
    hat_s = comps["COMP"]["raw_soup_fit"][:, 1]
    hat_ell_dev = comps["COMP"]["raw_soup_dev"][:, 0]
    hat_s_dev = comps["COMP"]["raw_soup_dev"][:, 1]
    const_ell = float(np.median(ell_fit))
    const_s = float(np.median(s_fit))
    diag: dict[str, Any] = {
        "fit_constants": {"median_ell_fit": const_ell, "median_s_fit": const_s},
        "ell": {
            "fit_overall": _component_stats(hat_ell, ell_fit, k_fit),
            "fit_G0": _component_stats(hat_ell[k_fit == 0], ell_fit[k_fit == 0], np.zeros(int((k_fit == 0).sum()), np.int64)),
            "dev_overall": _component_stats(hat_ell_dev, ell_dev, k_dev),
            "dev_G0": _component_stats(hat_ell_dev[k_dev == 0], ell_dev[k_dev == 0], np.zeros(int((k_dev == 0).sum()), np.int64)),
            "constant_mae_fit": _constant_mae(ell_fit, const_ell),
            "constant_mae_dev": _constant_mae(ell_dev, const_ell),
            "constant_mae_G0_fit": _constant_mae(ell_fit[k_fit == 0], const_ell),
        },
        "s": {
            "fit_overall": _component_stats(hat_s, s_fit, k_fit),
            "fit_G0": _component_stats(hat_s[k_fit == 0], s_fit[k_fit == 0], np.zeros(int((k_fit == 0).sum()), np.int64)),
            "dev_overall": _component_stats(hat_s_dev, s_dev, k_dev),
            "dev_G0": _component_stats(hat_s_dev[k_dev == 0], s_dev[k_dev == 0], np.zeros(int((k_dev == 0).sum()), np.int64)),
            "constant_mae_fit": _constant_mae(s_fit, const_s),
            "constant_mae_dev": _constant_mae(s_dev, const_s),
            "constant_mae_G0_fit": _constant_mae(s_fit[k_fit == 0], const_s),
        },
    }
    for name in ("ell", "s"):
        entry = diag[name]
        entry["gap_overall_raw"] = entry["dev_overall"]["raw_mae"] - entry["fit_overall"]["raw_mae"]
        entry["gap_G0_raw"] = entry["dev_G0"]["raw_mae"] - entry["fit_G0"]["raw_mae"]
        entry["ratio_model_to_constant_fit"] = (
            entry["fit_overall"]["raw_mae"] / entry["constant_mae_fit"] if entry["constant_mae_fit"] else None
        )
        entry["ratio_model_to_constant_dev"] = (
            entry["dev_overall"]["raw_mae"] / entry["constant_mae_dev"] if entry["constant_mae_dev"] else None
        )
        const_g0 = entry["constant_mae_G0_fit"]
        model_g0 = entry["fit_G0"]["raw_mae"]
        if const_g0 is None or const_g0 <= DEGENERATE_CONST_MAE:
            entry["fit_adequacy"] = {"marker": "DEGENERATE_TARGET", "ratio": None}
        else:
            ratio = model_g0 / const_g0
            entry["fit_adequacy"] = {
                "marker": "FIT_ADEQUATE" if ratio <= FIT_ADEQUATE_RATIO else "FIT_NOT_ADEQUATE",
                "ratio": ratio,
                "threshold": FIT_ADEQUATE_RATIO,
            }

    # error cancellation on raw COMP predictions
    def cancellation(hat_e: np.ndarray, tgt_e: np.ndarray, hat_ss: np.ndarray, tgt_s: np.ndarray, selector: np.ndarray | None = None) -> dict[str, Any]:
        e_ell = (hat_e - tgt_e)
        e_s = (hat_ss - tgt_s)
        if selector is not None:
            e_ell, e_s = e_ell[selector], e_s[selector]
        e_g = e_ell + e_s
        abs_ell, abs_s, abs_g = np.abs(e_ell), np.abs(e_s), np.abs(e_g)
        opposite = float(np.mean((e_ell * e_s) < 0.0)) if e_ell.size else float("nan")
        return {
            "n": int(e_ell.size),
            "mean_abs_e_ell": float(np.mean(abs_ell)) if e_ell.size else None,
            "mean_abs_e_s": float(np.mean(abs_s)) if e_ell.size else None,
            "mean_abs_e_g": float(np.mean(abs_g)) if e_ell.size else None,
            "sum_of_component_maes": float(np.mean(abs_ell) + np.mean(abs_s)) if e_ell.size else None,
            "opposite_sign_fraction": opposite,
            "triangle_gap": float(np.mean(abs_ell + abs_s - abs_g)) if e_ell.size else None,
            "g_bias_mean": float(np.mean(e_g)) if e_ell.size else None,
        }

    g0_fit = k_fit == 0
    g0_dev = k_dev == 0
    cancellation_table = {
        "fit_G0": cancellation(hat_ell[g0_fit], ell_fit[g0_fit], hat_s[g0_fit], s_fit[g0_fit]),
        "fit_overall": cancellation(hat_ell, ell_fit, hat_s, s_fit),
        "dev_G0": cancellation(hat_ell_dev[g0_dev], ell_dev[g0_dev], hat_s_dev[g0_dev], s_dev[g0_dev]),
        "dev_overall": cancellation(hat_ell_dev, ell_dev, hat_s_dev, s_dev),
    }
    # the COMP raw g error identity must hold exactly
    e_ell_all = hat_ell - ell_fit
    e_s_all = hat_s - s_fit
    e_g_all = hat_ell + hat_s - (ell_fit + s_fit)
    cancellation_identity_max = float(np.max(np.abs(e_g_all - (e_ell_all + e_s_all))))
    diag["cancellation"] = cancellation_table
    diag["cancellation_identity_max_abs"] = cancellation_identity_max

    # diagnostic conclusion (no new fitting)
    ell_ok = diag["ell"]["fit_adequacy"]["marker"] == "FIT_ADEQUATE"
    s_ok = diag["s"]["fit_adequacy"]["marker"] == "FIT_ADEQUATE"
    ell_g0_dev = diag["ell"]["dev_G0"]["raw_mae"]
    s_g0_dev = diag["s"]["dev_G0"]["raw_mae"]
    ell_gap = diag["ell"]["gap_G0_raw"]
    s_gap = diag["s"]["gap_G0_raw"]
    if not ell_ok or not s_ok:
        conclusion = "FIT_INSUFFICIENT: at least one component is not yet fit-adequate; fit/optimisation is not separated from generalisation"
    elif ell_g0_dev is not None and s_g0_dev is not None and max(ell_g0_dev, s_g0_dev) >= GAP_RATIO_MARK * min(ell_g0_dev, s_g0_dev) and (
        (ell_g0_dev > s_g0_dev and ell_gap >= s_gap) or (s_g0_dev > ell_g0_dev and s_gap >= ell_gap)
    ):
        conclusion = "SINGLE_COMPONENT_TARGET: the worse G0-dev component is the next research object"
    elif ell_gap > 0.0 and s_gap > 0.0:
        conclusion = "BROAD_GENERALISATION_GAP: both components show a dev gap; no single property localised"
    else:
        conclusion = "INCONCLUSIVE"
    diag["conclusion"] = conclusion
    diag["fit_adequate"] = {"ell": ell_ok, "s": s_ok}
    if cancellation_table["dev_overall"]["triangle_gap"] is not None and cancellation_table["dev_overall"]["triangle_gap"] > 0.0:
        diag["cancellation_note"] = (
            "component MAEs do not add to the g budget; positive triangle_gap means favourable cancellation"
        )

    # sensitivity: drop the dev row with the largest SUM calibrated error
    worst = int(np.argmax(np.abs(arms["SUM"]["err_dev_cal"])))
    keep = np.ones(k_dev.size, dtype=bool)
    keep[worst] = False
    sensitivity = {
        "dropped_dev_position": worst,
        "dropped_in_G0": bool(k_dev[worst] == 0),
        "G0_cal_gain": float(
            _mae(arms["SUM"]["err_dev_cal"][(k_dev == 0) & keep]) - _mae(arms["COMP"]["err_dev_cal"][(k_dev == 0) & keep])
        ),
        "overall_cal_gain": float(_mae(arms["SUM"]["err_dev_cal"][keep]) - _mae(arms["COMP"]["err_dev_cal"][keep])),
    }

    # replay: independently reload stored soups and compare stored predictions
    replay: dict[str, Any] = {}
    prep_meta, fit_data, dev_data = zfr.build_prepared_data(fresh, verify_prep=True)
    for arm in ARMS:
        state = torch.load(out_dir / f"{arm}_raw_soup_state.pt", weights_only=True, map_location="cpu")
        model = build_component_model(arm, prev.TuplePayload(fresh["payload_arrays"]), float(fresh["kappa"]["kappa_M"]))
        model.load_state_dict({key: value.to(torch.device("cpu")) for key, value in state.items()}, strict=True)
        model = model.to(device)
        for split, data in (("fit", fit_data), ("dev", dev_data)):
            idx = list(range(128))
            sub = [data[i] for i in idx]
            pred_sum, pred_comp = evaluate_state_components(model, sub, device)
            stored_sum = sums[arm][f"raw_soup_{split}"][idx]
            stored_comp = comps[arm][f"raw_soup_{split}"][idx]
            replay[f"{arm}_{split}"] = {
                "sum_max_abs_vs_stored": float(np.max(np.abs(pred_sum - stored_sum))),
                "component_max_abs_vs_stored": float(np.max(np.abs(pred_comp - stored_comp))),
            }
    replay["tol"] = REPLAY_TOL
    replay["all_ok"] = all(
        replay[key]["sum_max_abs_vs_stored"] <= REPLAY_TOL and replay[key]["component_max_abs_vs_stored"] <= REPLAY_TOL
        for key in replay
        if key != "tol"
    )

    analysis = {
        "protocol_version": PROTOCOL_VERSION,
        "classification": classification,
        "gate_conditions": gate_conditions,
        "practical_equivalence": equivalence,
        "gains_COMP_minus_baseline_SUM": gains,
        "bias": bias,
        "arms": {arm: {key: value for key, value in arms[arm].items() if not key.startswith("err_")} for arm in ARMS},
        "identity_checks": identity,
        "component_diagnostics": diag,
        "sensitivity_drop_worst_sum": sensitivity,
        "bootstrap_witnesses": witnesses,
        "replay": replay,
        "n_dev": int(k_dev.size),
        "n_G0": int((k_dev == 0).sum()),
        "n_fit": int(k_fit.size),
        "n_G0_fit": int((k_fit == 0).sum()),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(out_dir / "analysis.json", analysis)
    write_json(out_dir / "gains.json", gains)
    write_json(out_dir / "gate.json", {"classification": classification, "conditions": gate_conditions,
                                       "practical_equivalence": equivalence})
    write_json(out_dir / "bootstrap.json", {"gains": gains, "witnesses": witnesses})

    # CSV tables
    import csv

    main_rows = []
    for arm in ARMS:
        table = arms[arm]
        main_rows.append(
            {
                "arm": arm,
                "fit_overall_raw": table["fit_overall_raw_mae"],
                "fit_overall_cal": table["fit_overall_cal_mae"],
                "fit_G0_cal": _mae((g_fit - cal[arm]["fit"])[k_fit == 0]),
                "dev_overall_raw": table["dev_overall_raw_mae"],
                "dev_overall_cal": table["dev_overall_cal_mae"],
                "dev_G0_raw": table["dev_G0_raw_mae"],
                "dev_G0_cal": table["dev_G0_cal_mae"],
                "bias": table["bias"],
                "gap_overall_cal": table["gap_overall_cal"],
                "gap_G0_cal": table["gap_G0_cal"],
            }
        )
    with open(out_dir / "main_table.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(main_rows[0]))
        writer.writeheader()
        writer.writerows(main_rows)

    group_rows = []
    for arm in ARMS:
        for name in GROUP_NAMES:
            entry = arms[arm]["dev_groups_cal"][name]
            group_rows.append(
                {"arm": arm, "split": "dev", "group": name, "n": entry["n"], "mae": entry["mae"], "contribution": entry["contribution"]}
            )
    with open(out_dir / "group_table.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(group_rows[0]))
        writer.writeheader()
        writer.writerows(group_rows)

    comp_rows = []
    for split in ("fit", "dev"):
        for comp in ("ell", "s"):
            for grid, key in (("overall", f"{split}_overall"), ("G0", f"{split}_G0")):
                entry = diag[comp][key]
                comp_rows.append(
                    {
                        "split": split,
                        "component": comp,
                        "grid": grid,
                        "n": entry["n"],
                        "raw_mae": entry["raw_mae"],
                        "signed_mean": entry["signed"]["mean"],
                        "signed_median": entry["signed"]["median"],
                        "std": entry["signed"]["std"],
                    }
                )
    with open(out_dir / "component_table.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comp_rows[0]))
        writer.writeheader()
        writer.writerows(comp_rows)

    log(f"[analyze] classification={classification} equiv={equivalence} replay_ok={replay['all_ok']}")
    log(f"[analyze] G0 cal gain={gains['G0_cal']['point']:+.6f} CI={gains['G0_cal']['ci95']}")
    log(f"[analyze] overall cal gain={gains['overall_cal']['point']:+.6f} CI={gains['overall_cal']['ci95']}")
    log(f"[analyze] component conclusion={conclusion}")
    return analysis


# ---------------------------------------------------------------------------
# 7. manifest / budget
# ---------------------------------------------------------------------------


def make_manifest(*, out_dir: Path = RESULTS_DIR) -> dict[str, Any]:
    files = {}
    for path in sorted(out_dir.glob("*")):
        if path.is_file():
            files[path.name] = {"sha256": file_sha256(path), "bytes": int(path.stat().st_size)}
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "files": files,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(out_dir / "manifest.json", payload)
    return payload


def make_budget(
    *,
    out_dir: Path = RESULTS_DIR,
    round_start: str = "",
    round_end: str = "",
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "round_start": round_start,
        "round_end": round_end,
        "gpu_budget_limit_hours": 0.8,
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
            "steps": meta["steps_done"],
            "device": meta["device"],
            "allocation_probe": meta.get("allocation_probe", {}),
        }
    payload["training_gpu_hours_upper_bound"] = total / 3600.0
    if extra:
        payload["extra"] = dict(extra)
    write_json(out_dir / "budget.json", payload)
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", action="store_true")
    parser.add_argument("--init-check", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--arm", choices=ARMS, default=None)
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--budget", action="store_true")
    parser.add_argument("--manifest", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    parser.add_argument("--source", type=Path, default=SOURCE_DIR)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args(argv)

    if args.targets:
        phase_targets(out_dir=args.out)
    if args.init_check:
        init_check(out_dir=args.out, device=resolve_device(args.device))
    if args.smoke:
        if args.arm is None:
            raise SystemExit("--smoke requires --arm")
        run_smoke(arm=args.arm, device=resolve_device(args.device), out_dir=args.out, max_steps=args.max_steps or 3)
    if args.train:
        if args.arm is None:
            raise SystemExit("--train requires --arm")
        train_arm(
            args.arm,
            source_dir=args.source,
            out_dir=args.out,
            device=resolve_device(args.device),
            epochs=args.epochs,
            max_steps=args.max_steps,
        )
    if args.analyze:
        analyze(out_dir=args.out, device=resolve_device(args.device))
    if args.budget:
        make_budget(out_dir=args.out)
    if args.manifest:
        make_manifest(out_dir=args.out)
    if not any((args.targets, args.init_check, args.smoke, args.train, args.analyze, args.budget, args.manifest)):
        parser.print_help()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
