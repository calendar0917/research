"""ZINC local tuple dictionary vs matched local MLP under chemical component
supervision (COMP loss), seed 0.

Round: ``zinc_local_dictionary_component_supervision_seed0_v1``.

The single question: at the *same* local 125-D tuple input, the same real J
incidence aggregation, the same M_g skeleton (Sem108+size2, C6 mask, sum/sq-sum,
static relations, topology25, 144->288->144 posterior MLP bridge) and the same
two-component supervision (``L = MAE(hat_g,g) + 0.5*(MAE(hat_ell,ell) +
MAE(hat_s,s))``, reader 39->2 half-init), does the shared sparse IHT local tuple
dictionary (``D_COMP``) beat the parameter-matched SiLU local MLP (``M_COMP``)
on the new internal dev chemical component g?

Two formal arms only, seed 0, 240 epochs, 15,120 steps each, fresh init (never
warm started from any trained soup).  New fold ``rng(20261006)`` 8000/2000; all
fitted statistics (body standardizers, tuple phi scaler, target constants,
kappa) are refitted on the new 8000 fit rows only.  Official valid/test are
never loaded; dev predictions/scores are computed only after both arms complete
and the evaluation roster is frozen.

``D_loc`` 125x64 (8,000) tied-IHT (10 steps, hard top-8, eta =
1/(1.05*sigma(Dbar)^2+1e-12), detached power-iteration sigma) vs ``A_raw``
64x125 (8,000) ``SiLU(x @ A_bar.T)``; ``A_init = D_init.T`` element-wise;
``W_loc`` 342x64 (21,888) zero init in both arms; ``kappa_M = r_D/r_M`` one-shot
label-free scale match on up to 8192 fit roots (seed 20261004).  Both arms
297,539 parameters.

Usage (local CPU for build/smoke; res-2 res2-cu124 for the two formal runs)::

    python -m tracks.ksvd.experiments.luyin16.\
zinc_local_dictionary_component_supervision_seed0_v1 --build-objects
    ... --init-check
    ... --smoke --device cuda
    ... --train --arm D_COMP --device cuda
    ... --train --arm M_COMP --device cuda
    ... --dev-eval --device cuda
    ... --mechanism-health
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

from tracks.ksvd.code import tccd_v0
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm
from tracks.ksvd.experiments.luyin16 import (
    zinc_chemistry_component_supervision_seed0_v1 as zcs,
)
from tracks.ksvd.experiments.luyin16 import zinc_chemistry_dictionary_vs_mlp_seed0_v1 as zcdm
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
from tracks.ksvd.experiments.luyin16 import zinc_task_dictionary_and_cycle_witness_seed0_v1 as zw

PROTOCOL_VERSION = "zinc-local-dictionary-component-supervision-seed0-v1"
RESULT_SLUG = "zinc_local_dictionary_component_supervision_seed0_v1"
RESULTS_DIR = zjd.TRACK_ROOT / "results" / RESULT_SLUG

# ---- frozen recipe (identical for both arms) ------------------------------
SEED = 0
EPOCHS = 240
LR = 1.0e-3
WEIGHT_DECAY = 1.0e-5
GRAD_CLIP = 5.0
BATCH_SIZE = 128
SOUP_EPOCHS = (236, 237, 238, 239, 240)
TRAIN_SHUFFLE_OFFSET = 101
LOG_EPOCHS = (1, 40, 120, 240)
N_FIT = 8000
N_DEV = 2000
STEPS_PER_EPOCH = 63          # ceil(8000/128) -> 15,120 steps at 240 epochs
STEPS_TOTAL = EPOCHS * STEPS_PER_EPOCH
REPLAY_TOL = 1.0e-5
TARGET_NOT_READ_TOL = 1.0e-5
MU_LOGP = 2.4570953396190123
COMPONENT_LOSS_WEIGHT = 0.5

# ---- the one new fold (fixed once, never re-drawn) -------------------------
FOLD_SEED = 20261006
NEW_FIT_SHA = "734d27f2818c5a1e13cd415ecb153eb0316036f72ffc0cf520c47c9538bd40c4"
NEW_DEV_SHA = "cb5f49dcb5e46b8f00314b3264b620e159219a273bfe0f058aa0fd85523309e4"

KAPPA_SEED = 20261004
KAPPA_SAMPLE_MAX = 8192

# ---- statistics / gates (frozen before the two formal runs) ---------------
BOOT_SEED = 20261006
N_BOOT = 1000
GATE_DELTA = 0.003
EQUIV_BAND = 0.003

ARMS = ("D_COMP", "M_COMP")
EXPECTED_PARAMETERS = 297_539
EXPECTED_BODY_PARAMETERS = 184_707      # base body 184,667 + reader 39->2 (+40)
BRIDGE_PARAMETERS = 82_944
LOCAL_PARAMETERS = 8_000 + 21_888
GROUP_NAMES = ("k=0", "k=-1", "k=-2", "k<=-3")

MAX_SMOKE_RUNS = 2
SMOKE_MAX_STEPS = 4


def _hash_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _array_sha256(value: np.ndarray, dtype: Any = np.float32) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(value, dtype)).tobytes()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value: np.ndarray, dtype: Any = np.float32) -> str:
    return _array_sha256(value, dtype)


def state_hash(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode())
        digest.update(np.ascontiguousarray(state[key].detach().cpu().numpy(), np.float32).tobytes())
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any] | Sequence[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(zw.jsonable(payload), indent=2, sort_keys=False) + "\n")


def seed_everything(seed: int) -> None:
    zw.seed_everything(int(seed))


def resolve_device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def group_masks(k: np.ndarray) -> dict[str, np.ndarray]:
    return zw.group_masks(k)


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
            capture_output=True, text=True, timeout=30,
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
# 1. the new fold (rng(20261006) permutation of all 10,000 train rows)
# ---------------------------------------------------------------------------


def build_fold_new() -> dict[str, Any]:
    """New fold fixed once: rng(20261006) permutation, first 8000 fit."""
    perm = np.random.default_rng(FOLD_SEED).permutation(10000)
    fit_idx = np.sort(perm[:N_FIT]).astype(np.int64)
    dev_idx = np.sort(perm[N_FIT:]).astype(np.int64)
    fit_sha = _hash_bytes(fit_idx.tobytes())
    dev_sha = _hash_bytes(dev_idx.tobytes())
    checks = {
        "sizes_ok": bool(fit_idx.size == N_FIT and dev_idx.size == N_DEV),
        "disjoint": bool(np.intersect1d(fit_idx, dev_idx).size == 0),
        "cover": bool(np.union1d(fit_idx, dev_idx).size == 10000),
        "fit_sha_matches_frozen": bool(fit_sha == NEW_FIT_SHA),
        "dev_sha_matches_frozen": bool(dev_sha == NEW_DEV_SHA),
    }
    if not all(checks.values()):
        raise RuntimeError(f"new fold construction check failed: {checks}")
    return {
        "fit_idx": fit_idx,
        "dev_idx": dev_idx,
        "definition": (
            "perm=np.random.default_rng(20261006).permutation(10000); "
            "fit_idx=np.sort(perm[:8000]); dev_idx=np.sort(perm[8000:])"
        ),
        "checks": checks,
        "fit_sha256": fit_sha,
        "dev_sha256": dev_sha,
    }


def load_fold() -> dict[str, Any]:
    with np.load(RESULTS_DIR / "new_fold.npz", allow_pickle=False) as z:
        fold = {"fit_idx": z["fit_idx"].astype(np.int64), "dev_idx": z["dev_idx"].astype(np.int64)}
    if _hash_bytes(fold["fit_idx"].tobytes()) != NEW_FIT_SHA or _hash_bytes(fold["dev_idx"].tobytes()) != NEW_DEV_SHA:
        raise RuntimeError("saved new fold hashes do not match the frozen values")
    return fold


# ---------------------------------------------------------------------------
# 2. phase A — build every fresh object for the new fold (fit-only stats)
# ---------------------------------------------------------------------------


def build_new_targets(fold: Mapping[str, Any]) -> dict[str, Any]:
    """Refit all target constants on the new 8000 fit rows; ell/s components."""
    rows, gid, label_checks = zfr.load_train_only_raw_rows()
    fit_idx = np.asarray(fold["fit_idx"], np.int64)
    dev_idx = np.asarray(fold["dev_idx"], np.int64)
    in_fit = np.zeros(10000, dtype=bool)
    in_fit[fit_idx] = True
    constants = zcdm._refit_constants(rows, in_fit, mu_logp=MU_LOGP)
    labels = zcdm._apply_constants(rows, constants, MU_LOGP)
    y = rows["y_stored"].astype(np.float64)
    c = labels["c"].astype(np.float64)
    k = labels["k"].astype(np.int64)
    g = y - c
    ell = (rows["logP"].astype(np.float64) - MU_LOGP) / float(constants["sigma_logP"])
    s = g - ell
    checks = {
        "n_rows": 10000,
        "n_fit": int(in_fit.sum()),
        "uses_dev_labels_in_fit": False,
        "y_identity_max_abs": float(np.max(np.abs(g - (y - c)))),
        "g_equals_ell_plus_s_max_abs": float(np.max(np.abs(g - (ell + s)))),
        "constants_finite": bool(all(math.isfinite(float(constants[key])) for key in constants)),
        "sigma_cycle_nonzero": bool(float(constants["sigma_cycle"]) != 0.0),
        "k_fit_counts": {name: int(mask.sum()) for name, mask in group_masks(k[fit_idx]).items()},
        "k_dev_counts": {name: int(mask.sum()) for name, mask in group_masks(k[dev_idx]).items()},
    }
    if checks["y_identity_max_abs"] > 1e-12 or checks["g_equals_ell_plus_s_max_abs"] > 1e-12:
        raise RuntimeError("target identity failed")
    return {
        "constants": constants, "gid": gid, "label_checks": label_checks,
        "arrays": {"y": y, "c": c, "g": g, "k": k, "ell": ell, "s": s},
        "checks": checks,
    }


def build_new_payload(fold: Mapping[str, Any]) -> dict[str, Any]:
    """Reuse the label-free tuple incidence structure; refit phi scaler +
    kappa_D on the new fit roots only (adapted from zfr.build_fresh_payload)."""
    with np.load(prev.TUPLE_NPZ) as old:
        arrays = {key: old[key] for key in (
            "root_base", "node_sizes", "pair_ptr", "pair_t", "pair_a",
            "pair_wJ", "pair_wI", "root_atom",
        )}
    phi_env, atom_env, node_sizes_env = prev._env_phi_atom()
    checks: dict[str, Any] = {
        "old_tuple_npz": str(prev.TUPLE_NPZ),
        "old_tuple_npz_sha256": file_sha256(prev.TUPLE_NPZ),
        "node_sizes_match_env": bool(np.array_equal(arrays["node_sizes"], node_sizes_env)),
        "root_atom_match_env": bool(np.array_equal(arrays["root_atom"], atom_env)),
        "pair_ptr_monotone": bool(np.all(np.diff(arrays["pair_ptr"]) >= 0)),
        "n_roots": int(arrays["root_base"][-1]),
        "n_pairs": int(arrays["pair_ptr"][-1]),
    }
    if not checks["node_sizes_match_env"] or not checks["root_atom_match_env"]:
        raise RuntimeError("reused tuple structure does not match the env cache")

    fit_idx = np.asarray(fold["fit_idx"], np.int64)
    total_roots = int(node_sizes_env.sum())
    fit_mol = np.zeros(10000, dtype=bool)
    fit_mol[fit_idx] = True
    fit_root_mask = np.zeros(total_roots, dtype=bool)
    for index in np.nonzero(fit_mol)[0]:
        lo, hi = int(arrays["root_base"][index]), int(arrays["root_base"][index + 1])
        fit_root_mask[lo:hi] = True

    pair_root = np.repeat(np.arange(total_roots, dtype=np.int64), np.diff(arrays["pair_ptr"]))
    realized = arrays["pair_wJ"] > 0.0
    realized = realized & fit_root_mask[pair_root]
    fit_realized_roots = pair_root[realized]
    if fit_realized_roots.size == 0:
        raise RuntimeError("no realized fit incident tuples")
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
    if not fit_root_mask[sample].all():
        raise RuntimeError("kappa sample has off-fit roots")

    kappa, kappa_stats = prev._kappa_for_roots(
        prev.init_d_loc(),
        phi_env, atom_env, arrays["pair_ptr"], arrays["pair_t"], arrays["pair_a"],
        arrays["pair_wJ"], arrays["pair_wI"],
        phi_mean.astype(np.float32), phi_std.astype(np.float32), np.float32(phi_scale),
        sample,
    )
    if not math.isfinite(kappa) or kappa <= 0.0:
        raise RuntimeError(f"MECHANISM_INIT_BLOCKED: kappa={kappa}")
    checks["phi_scaler"] = {
        "fitted_on": "new rng(20261006) fit rows, realized C>0 incident tuples",
        "n_samples": int(fit_realized_roots.size),
        "std_floor": float(prev.PHI_STD_FLOOR),
        "phi_scale_l2_rms": phi_scale,
        "phi_mean_sha256": array_sha256(phi_mean.astype(np.float32)),
        "phi_std_sha256": array_sha256(phi_std.astype(np.float32)),
    }
    checks["kappa_D"] = {
        "value": float(kappa),
        "definition": "1 / RMS(initial e_J, e_I) over up to 8192 sampled new-fit roots",
        "seed": int(KAPPA_SEED),
        "n_roots": int(sample.size),
        "sample_sha256": array_sha256(sample, np.int64),
        "raw_rms": kappa_stats["rms"],
        "e_J_rms": kappa_stats["e_J_rms"],
        "e_I_rms": kappa_stats["e_I_rms"],
    }
    arrays.update(
        phi_mean=phi_mean.astype(np.float32),
        phi_std=phi_std.astype(np.float32),
        phi_scale=np.asarray(phi_scale, np.float32),
        kappa=np.asarray(kappa, np.float32),
        kappa_sample=sample.astype(np.int64),
    )
    return {"arrays": arrays, "checks": checks}


def compute_kappa_M_new(payload_arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """kappa_M = r_D / r_M on the frozen fit-root sample (label-free)."""
    payload = prev.TuplePayload(payload_arrays)
    phi_env, atom_env, _node = prev._env_phi_atom()
    phi_t = torch.as_tensor(phi_env, dtype=torch.float32)
    atom_t = torch.as_tensor(atom_env, dtype=torch.long)
    root_ids = torch.as_tensor(payload.kappa_sample, dtype=torch.long)
    d_init = prev.init_d_loc()
    a_init = d_init.t().contiguous()
    with torch.no_grad():
        e_d = prev.compose_root_codes(
            d_init, phi_t, atom_t, payload.pair_ptr, payload.pair_t, payload.pair_a,
            payload.pair_wJ, root_ids, payload.phi_mean, payload.phi_std, payload.phi_scale,
        )
        e_m = mlpmod.compose_root_codes_m(
            a_init, phi_t, atom_t, payload.pair_ptr, payload.pair_t, payload.pair_a,
            payload.pair_wJ, root_ids, payload.phi_mean, payload.phi_std, payload.phi_scale,
        )
    kappa_D = float(payload.kappa)
    r_D = float((kappa_D * e_d).pow(2).mean().sqrt())
    r_M = float(e_m.pow(2).mean().sqrt())
    kappa_M = r_D / r_M if r_M > 0 else float("nan")
    if not math.isfinite(kappa_M) or kappa_M <= 0.0 or kappa_M > 1e9:
        raise RuntimeError(f"MECHANISM_INIT_BLOCKED: kappa_M={kappa_M}")
    return {
        "definition": "r_D = RMS(kappa_D * e_D_init); r_M = RMS(e_M_init); kappa_M = r_D/r_M on the frozen fit-root sample",
        "kappa_D": kappa_D,
        "r_D": r_D,
        "r_M": r_M,
        "e_D_init_rms": float(e_d.pow(2).mean().sqrt()),
        "e_M_init_rms": r_M,
        "kappa_M": float(kappa_M),
        "n_roots": int(payload.kappa_sample.numel()),
        "sample_sha256": array_sha256(payload.kappa_sample.numpy(), np.int64),
        "labels_used": False,
        "dev_used": False,
        "note": "r_D is not assumed to be 1; it is the RMS of the scaled initial dictionary codes",
    }


def build_new_prep(train_data: Sequence[Any], fold: Mapping[str, Any]) -> dict[str, Any]:
    meta = zw.apply_new_fit_prep(train_data, np.load(zfr.PREP_BLOB, allow_pickle=False), fold["fit_idx"])
    meta["fitted_rows"] = "new rng(20261006) fit only"
    return meta


def load_new_objects(out_dir: Path = RESULTS_DIR) -> dict[str, Any]:
    """Load the saved fresh objects with manifest hash checks."""
    manifest = json.loads((out_dir / "new_objects_manifest.json").read_text())
    paths = {
        "new_fold.npz": out_dir / "new_fold.npz",
        "new_targets.npz": out_dir / "new_targets.npz",
        "new_tuple_payload.npz": out_dir / "new_tuple_payload.npz",
        "new_prep.npz": out_dir / "new_prep.npz",
    }
    for name, path in paths.items():
        expected = manifest["artifacts"][name]["sha256"]
        actual = file_sha256(path)
        if actual != expected:
            raise RuntimeError(f"fresh artifact {name} sha256 {actual} != frozen {expected}")
    with np.load(paths["new_fold.npz"], allow_pickle=False) as z:
        fold = {"fit_idx": z["fit_idx"].astype(np.int64), "dev_idx": z["dev_idx"].astype(np.int64)}
    with np.load(paths["new_targets.npz"], allow_pickle=False) as z:
        targets = {
            key: z[key] for key in ("y", "c", "g", "k", "ell", "s", "gid")
        }
        constants = {
            str(n): float(v) for n, v in zip(z["constant_names"].tolist(), z["constants"].tolist())
        }
    with np.load(paths["new_tuple_payload.npz"], allow_pickle=False) as z:
        payload_arrays = {key: z[key] for key in z.files}
    with np.load(paths["new_prep.npz"], allow_pickle=False) as z:
        prep = {key: z[key] for key in z.files}
    kappa = json.loads((out_dir / "new_kappa.json").read_text())
    if abs(float(kappa["kappa_M"]) - float(manifest["kappa_M"]["kappa_M"])) > 0.0:
        raise RuntimeError("new_kappa.json kappa_M differs from the frozen manifest")
    return {
        "fold": fold, "targets": targets, "constants": constants,
        "payload_arrays": payload_arrays, "prep": prep, "kappa": kappa,
        "manifest": manifest,
    }


def build_prepared_data(new: Mapping[str, Any], *, verify_prep: bool = True):
    """Train-only loader + new-fit prep + fit/dev split (never official valid)."""
    train_data = zftd.load_train_only()
    if len(train_data) != 10000:
        raise RuntimeError("train-only cache length mismatch")
    fold = new["fold"]
    prep_meta = build_new_prep(train_data, fold)
    if verify_prep:
        for key, value in new["prep"].items():
            if key not in prep_meta or not np.array_equal(np.asarray(prep_meta[key], np.float32), np.asarray(value, np.float32)):
                raise RuntimeError(f"new prep mismatch at {key}")
    fit_data, dev_data = zftd.build_fit_dev(train_data, fold["fit_idx"], fold["dev_idx"])
    for position, index in enumerate(fold["fit_idx"].tolist()):
        fit_data[position].local_mol_id = torch.tensor([int(index)], dtype=torch.long)
    for position, index in enumerate(fold["dev_idx"].tolist()):
        dev_data[position].local_mol_id = torch.tensor([int(index)], dtype=torch.long)
    return prep_meta, fit_data, dev_data


def phase_build_objects(*, out_dir: Path = RESULTS_DIR, log: Any = print) -> dict[str, Any]:
    torch.set_num_threads(8)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    fold = build_fold_new()
    np.savez_compressed(out_dir / "new_fold.npz", fit_idx=fold["fit_idx"], dev_idx=fold["dev_idx"])
    targets = build_new_targets(fold)
    t = targets["arrays"]
    np.savez_compressed(
        out_dir / "new_targets.npz",
        y=t["y"], c=t["c"], g=t["g"], k=t["k"], ell=t["ell"], s=t["s"],
        gid=targets["gid"],
        constants=np.asarray([targets["constants"][name] for name in
                              ("sigma_logP", "sigma_SA", "mu_SA", "sigma_cycle", "mu_cycle")], np.float64),
        constant_names=np.asarray(["sigma_logP", "sigma_SA", "mu_SA", "sigma_cycle", "mu_cycle"]),
        mu_logP=np.asarray(MU_LOGP, np.float64),
    )
    payload = build_new_payload(fold)
    np.savez_compressed(out_dir / "new_tuple_payload.npz", **payload["arrays"])
    kappa_M = compute_kappa_M_new(payload["arrays"])
    write_json(out_dir / "new_kappa.json", kappa_M)
    train_data = zftd.load_train_only()
    prep_meta = build_new_prep(train_data, fold)
    np.savez_compressed(
        out_dir / "new_prep.npz",
        patch_fit_mean=prep_meta["patch_fit_mean"], patch_fit_scale=prep_meta["patch_fit_scale"],
        ctx_fit_mean=prep_meta["ctx_fit_mean"], ctx_fit_scale=prep_meta["ctx_fit_scale"],
        anchor_fit_mean=prep_meta["anchor_fit_mean"], anchor_fit_scale=prep_meta["anchor_fit_scale"],
        topo_fit_mean=prep_meta["topo_fit_mean"], topo_fit_scale=prep_meta["topo_fit_scale"],
    )
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "fold": {
            "definition": fold["definition"],
            "fit_idx_sha256": fold["fit_sha256"],
            "dev_idx_sha256": fold["dev_sha256"],
            "checks": fold["checks"],
        },
        "constants": targets["constants"],
        "mu_logP": MU_LOGP,
        "target_checks": targets["checks"],
        "label_source_checks": targets["label_checks"],
        "payload_checks": payload["checks"],
        "kappa_M": kappa_M,
        "prep": {
            "fitted_on": prep_meta["fitted_on"],
            "fit_root_rows": prep_meta["fit_root_rows"],
            "n_fit_molecules": prep_meta["n_fit_molecules"],
        },
        "artifacts": {
            name: {"path": str(path), "sha256": file_sha256(path)}
            for name, path in (
                ("new_fold.npz", out_dir / "new_fold.npz"),
                ("new_targets.npz", out_dir / "new_targets.npz"),
                ("new_tuple_payload.npz", out_dir / "new_tuple_payload.npz"),
                ("new_prep.npz", out_dir / "new_prep.npz"),
            )
        },
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "seconds": float(time.perf_counter() - t0),
    }
    write_json(out_dir / "new_objects_manifest.json", manifest)
    log(
        f"[build] fold fit/dev sha={fold['fit_sha256'][:12]}/{fold['dev_sha256'][:12]} "
        f"kappa_D={payload['checks']['kappa_D']['value']:.6f} kappa_M={kappa_M['kappa_M']:.6f} "
        f"({manifest['seconds']:.1f}s)"
    )
    return manifest


# ---------------------------------------------------------------------------
# 3. the two arms (identical skeleton + supervision; only the local encoder)
# ---------------------------------------------------------------------------


class LocalTupleEncoderD(prev.LocalTupleEncoder):
    """Dictionary encoder + the fit-root mean-code switch (mirrors M's switch).

    The IHT production path is untouched (``super().root_codes``); the only
    addition is ``mean_replace`` for the frozen intervention, applied after the
    production aggregation, exactly as in the matched M encoder.
    """

    def __init__(self, payload: prev.TuplePayload, weight_key: str) -> None:
        super().__init__(payload, weight_key)
        self.mean_replace: torch.Tensor | None = None

    def _root_neighbour_counts(self, data: Any) -> torch.Tensor:
        phi = data.dict_phi
        device = phi.device
        n_rows = int(phi.shape[0])
        mol = data.local_mol_id.to(torch.long).reshape(-1)
        batch = data.batch.to(torch.long)
        graph_counts = torch.bincount(batch, minlength=int(mol.numel()))
        ptr = torch.cat([graph_counts.new_zeros(1), graph_counts.cumsum(0)])
        rows = torch.arange(n_rows, device=device)
        local_pos = rows - ptr[batch]
        root_base = self._buf("root_base", device)
        global_root = root_base[mol][batch] + local_pos
        pair_ptr = self._buf("pair_ptr", device)
        return pair_ptr[global_root + 1] - pair_ptr[global_root]

    def root_codes(self, data: Any) -> torch.Tensor:
        counts = self._root_neighbour_counts(data)
        out = super().root_codes(data)
        if self.mean_replace is not None and not self.ablate:
            replacement = self.mean_replace.to(device=out.device, dtype=out.dtype)
            has_neighbours = counts > 0
            out = torch.where(has_neighbours.unsqueeze(1), replacement.unsqueeze(0), torch.zeros_like(out))
        return out


def build_arm_comp(arm: str, payload: prev.TuplePayload, kappa_M: float) -> nn.Module:
    """Fresh seed-0 arm: D keeps the tied-IHT dictionary encoder (plus the
    mean-code switch), M the matched SiLU encoder; both get the 39->2
    half-init ComponentReader (fork_rng, RNG-neutral)."""
    if arm not in ARMS:
        raise ValueError(arm)
    if arm == "D_COMP":
        model = prev.build_arm("J", payload)      # IHT dictionary encoder, kappa=kappa_D
        original = model.local_tuple
        encoder = LocalTupleEncoderD(payload, original.weight_key)
        with torch.no_grad():
            encoder.D_loc_raw.copy_(original.D_loc_raw)
            encoder.W_loc.copy_(original.W_loc)
        model.local_tuple = encoder
    else:
        model = mlpmod.build_arm_mj(payload, kappa_M=float(kappa_M))
    model.reader = zcs.ComponentReader(model.reader.net)   # isolated RNG (fork_rng)
    audit = arm_parameter_audit(model)
    expected = {
        "total_parameters": EXPECTED_PARAMETERS,
        "base_body_parameters": EXPECTED_BODY_PARAMETERS,
        "bridge_parameters": BRIDGE_PARAMETERS,
        "local_tuple_parameters": LOCAL_PARAMETERS,
        "reader_output_parameters": 2 * 39 + 2,
    }
    if audit != expected:
        raise RuntimeError(f"{arm} parameter audit failed: {audit} != {expected}")
    if not isinstance(model.local_dictionary_bridge, zw.MLPBridge):
        raise RuntimeError(f"{arm} posterior bridge is not the matched MLP bridge")
    return model


def arm_parameter_audit(model: nn.Module) -> dict[str, int]:
    return {
        "total_parameters": int(sum(p.numel() for p in model.parameters())),
        "base_body_parameters": prev.base_body_parameter_count(model),
        "bridge_parameters": prev.bridge_parameter_count(model),
        "local_tuple_parameters": prev.local_parameter_count(model),
        "reader_output_parameters": int(model.reader.net[-1].weight.numel() + model.reader.net[-1].bias.numel()),
    }


# ---------------------------------------------------------------------------
# 4. init identity checks (all pre-training, fit-only)
# ---------------------------------------------------------------------------


def init_check(*, out_dir: Path = RESULTS_DIR, device: torch.device | None = None, log: Any = print) -> dict[str, Any]:
    torch.set_num_threads(8)
    device = device or torch.device("cpu")
    new = load_new_objects(out_dir)
    fold = new["fold"]
    payload = prev.TuplePayload(new["payload_arrays"])
    kappa_M = float(new["kappa"]["kappa_M"])
    checks: dict[str, Any] = {}

    # (a) fold hashes + fit/dev separation already verified on load; restate
    checks["fold"] = {"fit_sha": NEW_FIT_SHA, "dev_sha": NEW_DEV_SHA, "separation": "verified on load"}

    # (b) dev label shuffle does not move the fit-fitted constants
    rows, _gid, _ = zfr.load_train_only_raw_rows()
    in_fit = np.zeros(10000, dtype=bool)
    in_fit[fold["fit_idx"]] = True
    perm = np.random.default_rng(FOLD_SEED + 1).permutation(int(fold["dev_idx"].size))
    shuffled = rows.copy()
    for field in ("logP", "SA", "y_stored", "cycle_score_gvae"):
        values = shuffled[field][fold["dev_idx"]].copy()
        shuffled[field][fold["dev_idx"]] = values[perm]
    base = zcdm._refit_constants(rows, in_fit, mu_logp=MU_LOGP)
    other = zcdm._refit_constants(shuffled, in_fit, mu_logp=MU_LOGP)
    diffs = {key: float(other[key]) - float(base[key]) for key in base}
    checks["dev_label_shuffle_constant_diffs"] = diffs
    checks["dev_label_shuffle_independent"] = bool(all(v == 0.0 for v in diffs.values()))

    # (c) build both arms; shared initial tensors identical; A = D_init.T;
    #     W_loc zero; reader half-split preserves the initial sum function
    seed_everything(SEED)
    model_d = build_arm_comp("D_COMP", payload, kappa_M).to(device)
    build_rng_d = torch.get_rng_state()
    seed_everything(SEED)
    model_m = build_arm_comp("M_COMP", payload, kappa_M).to(device)
    build_rng_m = torch.get_rng_state()
    state_d = {k: v.detach().cpu().clone() for k, v in model_d.state_dict().items()}
    state_m = {k: v.detach().cpu().clone() for k, v in model_m.state_dict().items()}
    shared = sorted(k for k in state_d if not k.startswith("local_tuple."))
    if shared != sorted(k for k in state_m if not k.startswith("local_tuple.")):
        raise RuntimeError("shared state key sets differ")
    max_abs = max(float((state_d[k] - state_m[k]).abs().max()) for k in shared)
    checks["shared_tensor_count"] = len(shared)
    checks["shared_tensor_max_abs_diff"] = max_abs
    checks["shared_tensors_identical_1e-5"] = bool(max_abs <= TARGET_NOT_READ_TOL)
    checks["A_equals_D_init_T"] = bool(torch.equal(
        state_m["local_tuple.A_raw"], state_d["local_tuple.D_loc_raw"].t().contiguous()))
    checks["W_loc_exactly_zero_both"] = bool(
        float(state_d["local_tuple.W_loc"].abs().sum()) == 0.0
        and float(state_m["local_tuple.W_loc"].abs().sum()) == 0.0)
    checks["D_uses_IHT_encoder"] = bool(
        isinstance(model_d.local_tuple, prev.LocalTupleEncoder)
        and model_d.local_tuple.D_loc_raw.shape == (prev.TUPLE_DIM, prev.K_TUPLE))
    checks["M_uses_SiLU_encoder"] = bool(
        isinstance(model_m.local_tuple, mlpmod.LocalTupleEncoderM)
        and model_m.local_tuple.A_raw.shape == (prev.K_TUPLE, prev.TUPLE_DIM))
    checks["bridge_is_MLP_both"] = bool(
        isinstance(model_d.local_dictionary_bridge, zw.MLPBridge)
        and isinstance(model_m.local_dictionary_bridge, zw.MLPBridge))
    checks["build_rng_equal"] = bool(torch.equal(build_rng_d, build_rng_m))
    checks["kappa_D"] = float(payload.kappa)
    checks["kappa_M"] = kappa_M

    # reader half-split: the initial sum function equals the original 39->1 head
    # (reference M_g skeleton built under an isolated RNG, same seed-0 stream)
    with torch.random.fork_rng(devices=[]):
        seed_everything(SEED)
        reference = zw.build_arm("M").to(device)
        reference.eval()
    last_w = state_d["reader.net." + str(len(model_d.reader.net) - 1) + ".weight"]
    last_b = state_d["reader.net." + str(len(model_d.reader.net) - 1) + ".bias"]
    checks["reader_last_layer_shape"] = list(last_w.shape)
    generator = torch.Generator().manual_seed(1234)
    reader_in = int(model_d.reader.net[0].in_features)
    z = torch.randn(256, reader_in, generator=generator, device=device)
    with torch.no_grad():
        split_sum = model_d.reader(z)
        original_pred = reference.reader(z)
    checks["reader_input_dim"] = reader_in
    checks["reader_split_sum_equals_original_head_max_abs"] = float((split_sum - original_pred).abs().max())

    # (d) forward label independence + shapes on a fit batch
    prep_meta, fit_data, dev_data = build_prepared_data(new)
    g = np.asarray(new["targets"]["g"], np.float64)
    g_fit = g[fold["fit_idx"]]
    target_g = torch.as_tensor(g_fit, dtype=torch.float32)
    batch = zftd.make_batch(fit_data, list(range(8)), target_g, device)
    model_d.eval()
    model_m.eval()
    with torch.no_grad():
        pred_d = model_d(batch, mask=cm.C6_MASK)
        comp_d = model_d.reader.components()
        pred_m = model_m(batch, mask=cm.C6_MASK)
        comp_m = model_m.reader.components()
    checks["forward_shapes"] = {
        "D_sum": list(pred_d.shape), "D_components": list(comp_d.shape),
        "M_sum": list(pred_m.shape), "M_components": list(comp_m.shape),
        "initial_sum_function_equal": bool(
            torch.equal(pred_d, pred_m) and torch.equal(comp_d, comp_m)),
    }
    # labels do not enter forward: permuted targets leave predictions unchanged
    batch2 = zftd.make_batch(fit_data, list(range(8)), torch.zeros(8), device)
    with torch.no_grad():
        pred_d2 = model_d(batch2, mask=cm.C6_MASK)
    checks["label_independence_max_abs"] = float((pred_d - pred_d2).abs().max())

    # (e) root codes vs the production reference operators (one whole fit molecule)
    first_fit = int(fold["fit_idx"][0])
    root_base = np.asarray(new["payload_arrays"]["root_base"], np.int64)
    global_roots = torch.tensor(
        list(range(int(root_base[first_fit]), int(root_base[first_fit + 1]))), dtype=torch.long)
    phi_env, atom_env, _ = prev._env_phi_atom()
    phi_t = torch.as_tensor(phi_env, dtype=torch.float32)
    atom_t = torch.as_tensor(atom_env, dtype=torch.long)
    with torch.no_grad():
        ref_d = prev.compose_root_codes(
            model_d.local_tuple.D_loc_raw.cpu(), phi_t, atom_t, payload.pair_ptr,
            payload.pair_t, payload.pair_a, payload.pair_wJ, global_roots,
            payload.phi_mean, payload.phi_std, payload.phi_scale)
        ref_m = mlpmod.compose_root_codes_m(
            model_m.local_tuple.A_raw.cpu(), phi_t, atom_t, payload.pair_ptr,
            payload.pair_t, payload.pair_a, payload.pair_wJ, global_roots,
            payload.phi_mean, payload.phi_std, payload.phi_scale)
    root_batch = zftd.make_batch(fit_data[:1], [0], target_g, device)
    with torch.no_grad():
        model_root_d = model_d.local_tuple.root_codes(root_batch)
        model_root_m = model_m.local_tuple.root_codes(root_batch)
    checks["root_codes_vs_reference_max_abs"] = {
        "D": float((model_root_d.cpu() * payload.kappa - ref_d * payload.kappa).abs().max()),
        "M": float((model_root_m.cpu() * kappa_M - ref_m * kappa_M).abs().max()),
    }
    # IHT sparsity witness: per-tuple codes have <= 8 nonzero entries of 64
    with torch.no_grad():
        n_take = min(5, int(root_batch.dict_phi.shape[0]))
        x = model_d.local_tuple.tuple_features(
            root_batch.dict_phi[:n_take], root_batch.dict_atom.to(torch.long)[:n_take],
            torch.zeros(n_take, dtype=torch.long, device=device),
            torch.zeros(n_take, dtype=torch.long, device=device))
        alpha = prev.tied_tuple_codes(model_d.local_tuple.D_loc_raw, x)
    checks["IHT_per_tuple_nnz_max"] = int((alpha != 0).sum(1).max())
    checks["IHT_per_tuple_nnz_le_8"] = bool(int((alpha != 0).sum(1).max()) <= prev.SPARSITY)
    # eta/spectral estimate is deterministic (frozen start vector, no RNG)
    checks["power_iter_sigma_deterministic"] = bool(
        float(tccd_v0.power_iter_sigma(F.normalize(model_d.local_tuple.D_loc_raw, dim=0))) ==
        float(tccd_v0.power_iter_sigma(F.normalize(model_d.local_tuple.D_loc_raw, dim=0))))

    # (f) schedule + eval loader do not consume the training RNG stream
    rng_before = torch.get_rng_state().clone()
    schedule, schedule_hash = zw.build_schedule(len(fit_data), EPOCHS, SEED + TRAIN_SHUFFLE_OFFSET)
    _ = zftd.make_batch(fit_data, list(range(4)), target_g, torch.device("cpu"))
    rng_after = torch.get_rng_state()
    checks["schedule_and_loader_rng_neutral"] = bool(torch.equal(rng_before, rng_after))
    checks["schedule_sha256"] = schedule_hash
    checks["steps_expected"] = STEPS_TOTAL
    checks["steps_per_epoch"] = math.ceil(len(fit_data) / BATCH_SIZE)
    if int(math.ceil(len(fit_data) / BATCH_SIZE)) * EPOCHS != STEPS_TOTAL:
        raise RuntimeError("step count does not match the frozen 15,120")

    checks["official_valid_loaded"] = False
    checks["official_test_loaded"] = False
    write_json(out_dir / "init_identity.json", checks)
    log(
        f"[init] shared_max_abs={max_abs:.2e} A=D.T={checks['A_equals_D_init_T']} "
        f"W_loc0={checks['W_loc_exactly_zero_both']} IHTnnz<=8={checks['IHT_per_tuple_nnz_le_8']} "
        f"rng_neutral={checks['schedule_and_loader_rng_neutral']}"
    )
    return checks


# ---------------------------------------------------------------------------
# 5. fit-only smoke (max 2 runs, <= 4 steps per arm, states discarded)
# ---------------------------------------------------------------------------


def run_smoke(*, device: torch.device, out_dir: Path = RESULTS_DIR, log: Any = print) -> dict[str, Any]:
    smoke_path = out_dir / "smoke_runs.json"
    runs = json.loads(smoke_path.read_text())["runs"] if smoke_path.exists() else []
    if len(runs) >= MAX_SMOKE_RUNS:
        raise RuntimeError(f"smoke budget exhausted ({len(runs)}/{MAX_SMOKE_RUNS})")
    new = load_new_objects(out_dir)
    payload = prev.TuplePayload(new["payload_arrays"])
    kappa_M = float(new["kappa"]["kappa_M"])
    prep_meta, fit_data, _dev_data = build_prepared_data(new)
    fold = new["fold"]
    g = np.asarray(new["targets"]["g"], np.float64)
    ell = np.asarray(new["targets"]["ell"], np.float64)
    s = np.asarray(new["targets"]["s"], np.float64)
    g_fit, ell_fit, s_fit = g[fold["fit_idx"]], ell[fold["fit_idx"]], s[fold["fit_idx"]]
    target_g = torch.as_tensor(g_fit, dtype=torch.float32)
    schedule, schedule_hash = zw.build_schedule(len(fit_data), EPOCHS, SEED + TRAIN_SHUFFLE_OFFSET)
    record: dict[str, Any] = {"device": str(device), "steps": SMOKE_MAX_STEPS, "schedule_sha256": schedule_hash}
    seed_everything(SEED)
    grad_log: dict[str, list[dict[str, float | None]]] = {"D_COMP": [], "M_COMP": []}
    for arm in ARMS:
        model = build_arm_comp(arm, payload, kappa_M).to(device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        target_ell = torch.as_tensor(ell_fit, dtype=torch.float32, device=device)
        target_s = torch.as_tensor(s_fit, dtype=torch.float32, device=device)
        for step in range(SMOKE_MAX_STEPS):
            indices = [int(i) for i in schedule[0][step * BATCH_SIZE:(step + 1) * BATCH_SIZE].tolist()]
            index_t = torch.as_tensor(indices, dtype=torch.long, device=device)
            batch = zftd.make_batch(fit_data, indices, target_g, device)
            prediction = model(batch, mask=cm.C6_MASK)
            components = model.reader.components()
            loss = (
                F.l1_loss(prediction.view(-1), batch.y.view(-1))
                + COMPONENT_LOSS_WEIGHT * (
                    F.l1_loss(components[:, 0], target_ell[index_t])
                    + F.l1_loss(components[:, 1], target_s[index_t])
                )
            )
            optimizer.zero_grad()
            loss.backward()
            total_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP))
            enc = model.local_tuple
            w_loc_grad = float(enc.W_loc.grad.norm())
            local_key = "D_loc_raw" if hasattr(enc, "D_loc_raw") else "A_raw"
            local_grad = float(getattr(enc, local_key).grad.norm())
            grad_log[arm].append({
                "step": step + 1, "clip_total_norm": total_norm,
                "W_loc_grad_norm": w_loc_grad, f"{local_key}_task_grad_norm": local_grad,
            })
            optimizer.step()
    # W_loc must receive task gradient at step 1; the dictionary/MLP tensor gets
    # zero task gradient at W_loc=0 (expected) and nonzero after W_loc moves
    for arm in ARMS:
        first, later = grad_log[arm][0], grad_log[arm][-1]
        key = [k for k in first if k.endswith("_task_grad_norm")][0]
        record[f"{arm}_W_loc_grad_step1"] = first["W_loc_grad_norm"]
        record[f"{arm}_local_task_grad_step1"] = first[key]
        record[f"{arm}_local_task_grad_step{SMOKE_MAX_STEPS}"] = later[key]
        if not (first["W_loc_grad_norm"] > 0.0):
            raise RuntimeError(f"{arm}: W_loc has no task gradient at step 1")
        if not (later[key] > 0.0):
            raise RuntimeError(f"{arm}: local tensor has no task gradient after W_loc moved")
    record["grad_log"] = grad_log
    record["states_discarded"] = True
    record["fit_only"] = True
    record["dev_scores_computed"] = False
    runs.append(record)
    write_json(smoke_path, {"runs": runs, "max_runs": MAX_SMOKE_RUNS})
    log(f"[smoke] run {len(runs)}/{MAX_SMOKE_RUNS} ok: W_loc step1 grads > 0, local grads live by step {SMOKE_MAX_STEPS}")
    return record


# ---------------------------------------------------------------------------
# 6. training (identical recipe; checkpoints at LOG_EPOCHS; fit-only eval)
# ---------------------------------------------------------------------------


def evaluate_state_components(
    model: nn.Module,
    data_list: Sequence[Any],
    device: torch.device,
    *,
    batch_size: int = BATCH_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    return zcs.evaluate_state_components(model, data_list, device, batch_size=batch_size)


def train_arm(
    arm: str,
    *,
    out_dir: Path = RESULTS_DIR,
    device: torch.device,
    epochs: int = EPOCHS,
    max_steps: int | None = None,
    log: Any = print,
) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError(arm)
    torch.set_num_threads(8 if device.type == "cpu" else 4)
    out_dir.mkdir(parents=True, exist_ok=True)
    new = load_new_objects(out_dir)
    fold = new["fold"]
    payload = prev.TuplePayload(new["payload_arrays"])
    kappa_M = float(new["kappa"]["kappa_M"])
    g = np.asarray(new["targets"]["g"], np.float64)
    ell = np.asarray(new["targets"]["ell"], np.float64)
    s = np.asarray(new["targets"]["s"], np.float64)
    fit_idx = fold["fit_idx"]
    g_fit, ell_fit, s_fit = g[fit_idx], ell[fit_idx], s[fit_idx]
    if float(np.max(np.abs(g_fit - (ell_fit + s_fit)))) > 1e-12:
        raise RuntimeError("g != ell + s on fit rows")

    target_g = torch.as_tensor(g_fit, dtype=torch.float32)
    target_ell = torch.as_tensor(ell_fit, dtype=torch.float32, device=device)
    target_s = torch.as_tensor(s_fit, dtype=torch.float32, device=device)

    prep_meta, fit_data, dev_data = build_prepared_data(new)
    schedule, schedule_hash = zw.build_schedule(len(fit_data), int(epochs), SEED + TRAIN_SHUFFLE_OFFSET)

    seed_everything(SEED)
    model = build_arm_comp(arm, payload, kappa_M)
    build_rng = torch.get_rng_state()
    model = model.to(device)
    seed_everything(SEED)
    train_rng = torch.get_rng_state()

    init_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    soup_epochs = set(int(e) for e in SOUP_EPOCHS)
    soup: dict[int, dict[str, torch.Tensor]] = {}
    checkpoints: dict[int, dict[str, torch.Tensor]] = {}
    curve: list[dict[str, Any]] = []
    probe_log: list[dict[str, Any]] = []
    steps_done = 0
    gid_stream = hashlib.sha256()
    id_stream = hashlib.sha256()
    started = time.perf_counter()
    stopped_reason = "completed"

    for epoch in range(1, int(epochs) + 1):
        epoch_started = time.perf_counter()
        model.train()
        g_sum = ell_sum = s_sum = total_sum = 0.0
        n_mol, n_steps, gnorm_sum, clip_hits = 0, 0, 0.0, 0
        for start in range(0, len(schedule[epoch - 1]), BATCH_SIZE):
            indices = schedule[epoch - 1][start:start + BATCH_SIZE]
            index_list = [int(i) for i in indices.tolist()]
            index_t = torch.as_tensor(index_list, dtype=torch.long, device=device)
            id_stream.update(np.asarray(index_list, np.int64).tobytes())
            gid_stream.update(np.asarray(new["targets"]["gid"][fit_idx[index_list]], np.int64).tobytes())
            batch = zftd.make_batch(fit_data, index_list, target_g, device)
            prediction = model(batch, mask=cm.C6_MASK)
            if tuple(prediction.shape) != (len(index_list),):
                raise RuntimeError("standard forward must return (batch,) total g")
            components = model.reader.components()
            if tuple(components.shape) != (len(index_list), 2):
                raise RuntimeError("reader component output is not (batch, 2)")
            l_g = F.l1_loss(prediction.view(-1), batch.y.view(-1))
            l_ell = F.l1_loss(components[:, 0], target_ell[index_t])
            l_s = F.l1_loss(components[:, 1], target_s[index_t])
            loss = l_g + COMPONENT_LOSS_WEIGHT * (l_ell + l_s)
            optimizer.zero_grad()
            loss.backward()
            total_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP))
            if epoch in LOG_EPOCHS and n_steps == 0:
                probe_log.append(_probe(model, arm, epoch, 1, total_norm))
            optimizer.step()
            g_sum += float((prediction.view(-1) - batch.y.view(-1)).abs().sum())
            ell_sum += float((components[:, 0].detach() - target_ell[index_t]).abs().sum())
            s_sum += float((components[:, 1].detach() - target_s[index_t]).abs().sum())
            total_sum += float(loss.detach()) * int(len(index_list))
            n_mol += int(batch.y.numel())
            n_steps += 1
            steps_done += 1
            gnorm_sum += total_norm
            clip_hits += int(total_norm > GRAD_CLIP)
            if max_steps is not None and steps_done >= int(max_steps):
                stopped_reason = "max_steps"
                break
        curve.append({
            "epoch": int(epoch),
            "train_L_g": float(g_sum / max(n_mol, 1)),
            "train_L_ell": float(ell_sum / max(n_mol, 1)),
            "train_L_s": float(s_sum / max(n_mol, 1)),
            "train_loss": float(total_sum / max(n_mol, 1)),
            "grad_norm": float(gnorm_sum / max(n_steps, 1)),
            "clip_fraction": float(clip_hits / max(n_steps, 1)),
            "seconds": float(time.perf_counter() - epoch_started),
        })
        if epoch in LOG_EPOCHS:
            checkpoints[int(epoch)] = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch in soup_epochs:
            soup[int(epoch)] = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if log and (epoch == 1 or epoch % 20 == 0 or epoch == int(epochs)):
            entry = curve[-1]
            log(
                f"[{arm}] ep={epoch:03d} L_g={entry['train_L_g']:.6f} L_ell={entry['train_L_ell']:.6f} "
                f"L_s={entry['train_L_s']:.6f} gnorm={entry['grad_norm']:.3g} {entry['seconds']:.1f}s"
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
        "optimised_loss": "L_g + 0.5*(L_ell + L_s)",
        "supervision": {
            "g": "g = y - c (new rng(20261006) fold fit-only constants)",
            "ell": "(logP - MU_LOGP) / sigma_logP",
            "s": "g - ell (residual chemistry component)",
        },
        "seed": SEED, "epochs": int(epochs), "steps_done": int(steps_done),
        "steps_expected": STEPS_TOTAL, "stopped_reason": stopped_reason,
        "lr": LR, "weight_decay": WEIGHT_DECAY, "grad_clip": GRAD_CLIP,
        "batch_size": BATCH_SIZE, "soup_epochs": members if max_steps is None else [],
        "n_fit": int(len(fit_data)), "n_dev": int(len(dev_data)),
        "kappa_D": float(payload.kappa), "kappa_M": kappa_M,
        "fold": {"fit_idx_sha256": NEW_FIT_SHA, "dev_idx_sha256": NEW_DEV_SHA},
        "schedule_sha256": schedule_hash,
        "position_stream_sha256": id_stream.hexdigest(),
        "global_gid_stream_sha256": gid_stream.hexdigest(),
        "init_state_hash": state_hash(init_state),
        "raw_soup_state_hash": state_hash(soup_state),
        "last_state_hash": state_hash(last_state),
        "build_rng_sha256": _hash_bytes(build_rng.numpy().tobytes()),
        "train_rng_sha256": _hash_bytes(train_rng.numpy().tobytes()),
        "curve_final": curve[-1] if curve else None,
        "wall_clock_s": float(time.perf_counter() - started),
        "device": str(device),
        "allocation_probe": allocation_probe(),
        "dev_scores_computed_during_training": False,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    if max_steps is not None:
        result["smoke_last_state_hash"] = state_hash(last_state)
        return result

    torch.save(init_state, out_dir / f"{arm}_init_state.pt")
    torch.save(last_state, out_dir / f"{arm}_last_state.pt")
    torch.save(soup_state, out_dir / f"{arm}_raw_soup_state.pt")
    for epoch, state in sorted(checkpoints.items()):
        torch.save(state, out_dir / f"{arm}_epoch{epoch}_state.pt")

    # fit-only predictions (init/last/soup); dev predictions deferred to the
    # frozen dev-eval phase (never during training)
    sums: dict[str, np.ndarray] = {}
    comps: dict[str, np.ndarray] = {}
    for state_name, state in (("init", init_state), ("last", last_state), ("raw_soup", soup_state)):
        replay = build_arm_comp(arm, payload, kappa_M)
        replay.load_state_dict({key: value.to(torch.device("cpu")) for key, value in state.items()}, strict=True)
        replay = replay.to(device)
        fit_sum, fit_comp = evaluate_state_components(replay, fit_data, device)
        sums[f"{state_name}_fit"] = fit_sum
        comps[f"{state_name}_fit"] = fit_comp
    np.savez_compressed(
        out_dir / f"{arm}_fit_predictions.npz",
        **{key: value.astype(np.float32) for key, value in sums.items()},
    )
    np.savez_compressed(
        out_dir / f"{arm}_fit_component_predictions.npz",
        **{key: value.astype(np.float32) for key, value in comps.items()},
    )
    b_g = float(np.median(g_fit - sums["raw_soup_fit"]))
    result["calibration_b_g_fit_only"] = {
        "init": float(np.median(g_fit - sums["init_fit"])),
        "last": float(np.median(g_fit - sums["last_fit"])),
        "raw_soup": b_g,
    }
    write_json(out_dir / f"{arm}_curve.json", curve)
    write_json(out_dir / f"{arm}_probe_log.json", probe_log)
    log(f"[{arm}] done steps={steps_done} wall={result['wall_clock_s']:.1f}s b_g={b_g:.6f}")
    write_json(out_dir / f"{arm}_meta.json", result)
    return result


def _probe(model: nn.Module, arm: str, epoch: int, step: int, total_norm: float) -> dict[str, Any]:
    entry: dict[str, Any] = {"epoch": int(epoch), "step_in_epoch": int(step), "clip_total_norm": float(total_norm)}
    enc = model.local_tuple
    w_loc = enc.W_loc
    entry["W_loc_grad_norm"] = float(w_loc.grad.norm()) if w_loc.grad is not None else None
    entry["W_loc_norm"] = float(w_loc.detach().norm())
    if hasattr(enc, "D_loc_raw"):
        entry["D_loc_grad_norm"] = float(enc.D_loc_raw.grad.norm()) if enc.D_loc_raw.grad is not None else None
        entry["D_loc_norm"] = float(enc.D_loc_raw.detach().norm())
    if hasattr(enc, "A_raw"):
        entry["A_grad_norm"] = float(enc.A_raw.grad.norm()) if enc.A_raw.grad is not None else None
        entry["A_norm"] = float(enc.A_raw.detach().norm())
    entry["health"] = {key: value for key, value in enc.last_stats.items() if not torch.is_tensor(value)}
    return entry


# ---------------------------------------------------------------------------
# 7. frozen dev evaluation (both arms complete; roster frozen first)
# ---------------------------------------------------------------------------


def _mae(err: np.ndarray) -> float:
    return float(np.mean(np.abs(err))) if err.size else float("nan")


def phase_freeze_eval_roster(*, out_dir: Path = RESULTS_DIR, log: Any = print) -> dict[str, Any]:
    """Freeze the evaluation roster (both arms' soup states + fit biases) and
    the dev input definition BEFORE any dev prediction is computed."""
    metas = {}
    for arm in ARMS:
        meta = json.loads((out_dir / f"{arm}_meta.json").read_text())
        if meta.get("stopped_reason") != "completed" or int(meta["steps_done"]) != STEPS_TOTAL:
            raise RuntimeError(f"{arm} is not a completed 15,120-step trajectory")
        metas[arm] = meta
    roster = {
        "protocol_version": PROTOCOL_VERSION,
        "arms": ARMS,
        "main_endpoint": "dev G0(k=0) calibrated g-MAE; overall cal is the common gate; raw reported in parallel",
        "bias": {arm: float(metas[arm]["calibration_b_g_fit_only"]["raw_soup"]) for arm in ARMS},
        "soup_state_hashes": {arm: metas[arm]["raw_soup_state_hash"] for arm in ARMS},
        "schedule_sha256": {arm: metas[arm]["schedule_sha256"] for arm in ARMS},
        "position_streams_equal": bool(
            metas["D_COMP"]["position_stream_sha256"] == metas["M_COMP"]["position_stream_sha256"]),
        "global_gid_streams_equal": bool(
            metas["D_COMP"]["global_gid_stream_sha256"] == metas["M_COMP"]["global_gid_stream_sha256"]),
        "bootstrap": {"seed": BOOT_SEED, "n_boot": N_BOOT, "paired_same_indices": True},
        "gates": {
            "dictionary_candidate": [
                "G0_cal_gain >= 0.003", "overall_cal_gain >= 0.003", "G0_cal_ci_lower > 0",
                "G0_raw_gain > 0", "overall_raw_gain > 0",
            ],
            "equivalence_band": [-EQUIV_BAND, EQUIV_BAND],
        },
        "dev_split": "rng(20261006) fold dev 2000 rows; labels frozen in new_targets.npz",
        "dev_scores_computed_before_freeze": False,
    }
    if not roster["position_streams_equal"] or not roster["global_gid_streams_equal"]:
        raise RuntimeError("the two arms did not follow the identical position/gid stream")
    write_json(out_dir / "eval_roster.json", roster)
    log("[freeze] roster frozen (both soups + fit biases; no dev prediction computed yet)")
    return roster


def phase_dev_eval(*, out_dir: Path = RESULTS_DIR, device: torch.device | None = None, log: Any = print) -> dict[str, Any]:
    torch.set_num_threads(8)
    device = device or torch.device("cpu")
    roster = json.loads((out_dir / "eval_roster.json").read_text())
    new = load_new_objects(out_dir)
    fold = new["fold"]
    fit_idx, dev_idx = fold["fit_idx"], fold["dev_idx"]
    t = new["targets"]
    g, k, ell, s = (np.asarray(t[key], np.float64 if key != "k" else np.int64) for key in ("g", "k", "ell", "s"))
    g_fit, g_dev = g[fit_idx], g[dev_idx]
    k_fit, k_dev = k[fit_idx], k[dev_idx]
    ell_fit, ell_dev = ell[fit_idx], ell[dev_idx]
    s_fit, s_dev = s[fit_idx], s[dev_idx]
    payload = prev.TuplePayload(new["payload_arrays"])
    kappa_M = float(new["kappa"]["kappa_M"])
    bias = {arm: float(roster["bias"][arm]) for arm in ARMS}

    _, fit_data, dev_data = build_prepared_data(new)
    sums: dict[str, dict[str, np.ndarray]] = {}
    comps: dict[str, dict[str, np.ndarray]] = {}
    for arm in ARMS:
        state = torch.load(out_dir / f"{arm}_raw_soup_state.pt", map_location="cpu", weights_only=True)
        if state_hash(state) != roster["soup_state_hashes"][arm]:
            raise RuntimeError(f"{arm} soup state hash mismatch")
        replay = build_arm_comp(arm, payload, kappa_M)
        replay.load_state_dict({key: value.to(torch.device("cpu")) for key, value in state.items()}, strict=True)
        replay = replay.to(device)
        fit_sum, fit_comp = evaluate_state_components(replay, fit_data, device)
        dev_sum, dev_comp = evaluate_state_components(replay, dev_data, device)
        sums[arm] = {"fit": fit_sum, "dev": dev_sum}
        comps[arm] = {"fit": fit_comp, "dev": dev_comp}
        # fit-prediction replay must match the training-time fit predictions
        with np.load(out_dir / f"{arm}_fit_predictions.npz") as z:
            fit_saved = np.asarray(z["raw_soup_fit"], np.float64)
        replay_max = float(np.max(np.abs(fit_sum - fit_saved)))
        if replay_max > REPLAY_TOL:
            raise RuntimeError(f"{arm} fit replay max abs {replay_max} > {REPLAY_TOL}")
        np.savez_compressed(
            out_dir / f"{arm}_dev_predictions.npz",
            **{key: value.astype(np.float32) for key, value in sums[arm].items()},
        )
        np.savez_compressed(
            out_dir / f"{arm}_dev_component_predictions.npz",
            **{key: value.astype(np.float32) for key, value in comps[arm].items()},
        )

    arms: dict[str, Any] = {}
    for arm in ARMS:
        raw = {"fit": sums[arm]["fit"], "dev": sums[arm]["dev"]}
        cal = {key: raw[key] + bias[arm] for key in raw}
        err_fit = g_fit - cal["fit"]
        err_dev = g_dev - cal["dev"]
        err_dev_raw = g_dev - raw["dev"]
        arms[arm] = {
            "bias": bias[arm],
            "fit_overall_cal_mae": _mae(err_fit),
            "fit_overall_raw_mae": _mae(g_fit - raw["fit"]),
            "dev_overall_raw_mae": _mae(err_dev_raw),
            "dev_overall_cal_mae": _mae(err_dev),
            "dev_G0_raw_mae": _mae(err_dev_raw[k_dev == 0]),
            "dev_G0_cal_mae": _mae(err_dev[k_dev == 0]),
            "fit_groups_cal": zw._metric_table(err_fit, k_fit),
            "dev_groups_cal": zw._metric_table(err_dev, k_dev),
            "dev_groups_raw": zw._metric_table(err_dev_raw, k_dev),
            "err_dev_cal": err_dev,
            "err_dev_raw": err_dev_raw,
        }

    # paired bootstrap: gain = MAE(M_COMP) - MAE(D_COMP); positive = dictionary improves
    err_m = {
        "G0_cal": arms["M_COMP"]["err_dev_cal"][k_dev == 0],
        "overall_cal": arms["M_COMP"]["err_dev_cal"],
        "G0_raw": arms["M_COMP"]["err_dev_raw"][k_dev == 0],
        "overall_raw": arms["M_COMP"]["err_dev_raw"],
    }
    err_d = {
        "G0_cal": arms["D_COMP"]["err_dev_cal"][k_dev == 0],
        "overall_cal": arms["D_COMP"]["err_dev_cal"],
        "G0_raw": arms["D_COMP"]["err_dev_raw"][k_dev == 0],
        "overall_raw": arms["D_COMP"]["err_dev_raw"],
    }
    gains = zfr._paired_bootstrap(err_m, err_d, k_dev, seed=BOOT_SEED, n_boot=N_BOOT)
    witnesses = zfr._bootstrap_witnesses(
        arms["M_COMP"]["err_dev_cal"], arms["D_COMP"]["err_dev_cal"], k_dev)

    gate = {
        "1_G0_cal_gain_ge_0.003": bool(gains["G0_cal"]["point"] >= GATE_DELTA),
        "2_overall_cal_gain_ge_0.003": bool(gains["overall_cal"]["point"] >= GATE_DELTA),
        "3_G0_cal_ci_lower_gt_0": bool(gains["G0_cal"]["ci95"][0] > 0.0),
        "4_G0_raw_gain_gt_0": bool(gains["G0_raw"]["point"] > 0.0),
        "5_overall_raw_gain_gt_0": bool(gains["overall_raw"]["point"] > 0.0),
    }
    m_gate = {
        "M_G0_cal_gain_ge_0.003": bool(-gains["G0_cal"]["point"] >= GATE_DELTA),
        "M_overall_cal_gain_ge_0.003": bool(-gains["overall_cal"]["point"] >= GATE_DELTA),
        "M_G0_cal_ci_lower_gt_0": bool(-gains["G0_cal"]["ci95"][1] > 0.0),
        "M_G0_raw_gain_gt_0": bool(-gains["G0_raw"]["point"] > 0.0),
        "M_overall_raw_gain_gt_0": bool(-gains["overall_raw"]["point"] > 0.0),
    }
    dictionary_pass = all(gate.values())
    mlp_pass = all(m_gate.values())
    equiv = bool(
        -EQUIV_BAND <= gains["G0_cal"]["ci95"][0] and gains["G0_cal"]["ci95"][1] <= EQUIV_BAND
        and -EQUIV_BAND <= gains["overall_cal"]["ci95"][0] and gains["overall_cal"]["ci95"][1] <= EQUIV_BAND
        and not dictionary_pass and not mlp_pass
    )
    mechanism_failed = False  # set by the mechanism-health phase; recorded here as pending
    if dictionary_pass and not mechanism_failed:
        classification = "DICTIONARY_CANDIDATE"
    elif mlp_pass:
        classification = "MLP_SUPPORTED_IN_THIS_INTERFACE"
    elif equiv:
        classification = "LOCAL_EQUIVALENCE"
    elif (gains["G0_cal"]["point"] > 0.0) == (gains["overall_cal"]["point"] > 0.0) and gains["G0_cal"]["point"] != 0.0:
        classification = "DIRECTIONAL_NOT_CONFIRMED"
    else:
        classification = "INCONCLUSIVE"

    # contribution identities + group table
    identity: dict[str, Any] = {}
    for arm in ARMS:
        table = arms[arm]["dev_groups_cal"]
        contrib = sum(table[name]["contribution"] for name in GROUP_NAMES if table[name]["n"] > 0)
        identity[f"{arm}_contribution_sum_minus_overall"] = float(contrib - table["mae"])
    gain_sum = sum(
        arms["M_COMP"]["dev_groups_cal"][name]["contribution"] - arms["D_COMP"]["dev_groups_cal"][name]["contribution"]
        for name in GROUP_NAMES
    )
    identity["group_gain_sum_minus_total"] = float(gain_sum - gains["overall_cal"]["point"])

    # sensitivity: drop M_COMP's dev max-|err| row once
    worst = int(np.argmax(np.abs(arms["M_COMP"]["err_dev_cal"])))
    keep = np.ones(k_dev.size, dtype=bool)
    keep[worst] = False
    sensitivity = {
        "dropped_dev_position": worst,
        "dropped_in_G0": bool(k_dev[worst] == 0),
        "G0_cal_gain": float(
            _mae(arms["M_COMP"]["err_dev_cal"][(k_dev == 0) & keep])
            - _mae(arms["D_COMP"]["err_dev_cal"][(k_dev == 0) & keep])),
        "overall_cal_gain": float(
            _mae(arms["M_COMP"]["err_dev_cal"][keep]) - _mae(arms["D_COMP"]["err_dev_cal"][keep])),
    }

    # component diagnostics + cancellation for both arms (raw, uncalibrated)
    def cancellation(hat_e, tgt_e, hat_ss, tgt_ss, selector=None):
        e_ell, e_s = hat_e - tgt_e, hat_ss - tgt_ss
        if selector is not None:
            e_ell, e_s = e_ell[selector], e_s[selector]
        abs_ell, abs_s, abs_g = np.abs(e_ell), np.abs(e_s), np.abs(e_ell + e_s)
        return {
            "n": int(e_ell.size),
            "mean_abs_e_ell": float(np.mean(abs_ell)),
            "mean_abs_e_s": float(np.mean(abs_s)),
            "mean_abs_e_g_raw": float(np.mean(abs_g)),
            "triangle_gap_raw": float(np.mean(abs_ell + abs_s - abs_g)),
            "opposite_sign_fraction": float(np.mean((e_ell * e_s) < 0.0)) if e_ell.size else None,
        }

    component_diag: dict[str, Any] = {}
    for arm in ARMS:
        hat_ell = comps[arm]["fit"][:, 0]
        hat_s = comps[arm]["fit"][:, 1]
        hat_ell_dev = comps[arm]["dev"][:, 0]
        hat_s_dev = comps[arm]["dev"][:, 1]
        g0_fit, g0_dev = k_fit == 0, k_dev == 0
        e_ell_dev = hat_ell_dev - ell_dev
        e_s_dev = hat_s_dev - s_dev
        component_diag[arm] = {
            "ell": {
                "fit_overall_raw_mae": _mae(hat_ell - ell_fit),
                "dev_overall_raw_mae": _mae(e_ell_dev),
                "dev_G0_raw_mae": _mae(e_ell_dev[g0_dev]),
            },
            "s": {
                "fit_overall_raw_mae": _mae(hat_s - s_fit),
                "dev_overall_raw_mae": _mae(e_s_dev),
                "dev_G0_raw_mae": _mae(e_s_dev[g0_dev]),
            },
            "cancellation": {
                "fit_overall": cancellation(hat_ell, ell_fit, hat_s, s_fit),
                "fit_G0": cancellation(hat_ell[g0_fit], ell_fit[g0_fit], hat_s[g0_fit], s_fit[g0_fit]),
                "dev_overall": cancellation(hat_ell_dev, ell_dev, hat_s_dev, s_dev),
                "dev_G0": cancellation(hat_ell_dev[g0_dev], ell_dev[g0_dev], hat_s_dev[g0_dev], s_dev[g0_dev]),
            },
            "cal_g_identity_max_abs": float(np.max(np.abs(
                (arms[arm]["err_dev_raw"]) - (e_ell_dev + e_s_dev)))),
        }
    # D - M component changes
    component_diag["D_minus_M"] = {
        "ell_dev_raw_mae_delta": float(component_diag["M_COMP"]["ell"]["dev_overall_raw_mae"]
                                       - component_diag["D_COMP"]["ell"]["dev_overall_raw_mae"]),
        "s_dev_raw_mae_delta": float(component_diag["M_COMP"]["s"]["dev_overall_raw_mae"]
                                     - component_diag["D_COMP"]["s"]["dev_overall_raw_mae"]),
        "ell_dev_G0_raw_mae_delta": float(component_diag["M_COMP"]["ell"]["dev_G0_raw_mae"]
                                          - component_diag["D_COMP"]["ell"]["dev_G0_raw_mae"]),
        "s_dev_G0_raw_mae_delta": float(component_diag["M_COMP"]["s"]["dev_G0_raw_mae"]
                                        - component_diag["D_COMP"]["s"]["dev_G0_raw_mae"]),
    }

    # frozen interventions (native fit bias; labels not used in the intervention)
    interventions = phase_interventions(
        out_dir=out_dir, new=new, payload=payload, kappa_M=kappa_M,
        sums=sums, comps=comps, bias=bias, device=device, g_dev=g_dev, k_dev=k_dev,
        fit_data=fit_data, dev_data=dev_data)

    result = {
        "protocol_version": PROTOCOL_VERSION,
        "roster": {"arms": list(ARMS), "bias": bias},
        "main_table": {
            arm: {
                "bias": bias[arm],
                "fit_overall_cal": arms[arm]["fit_overall_cal_mae"],
                "dev_overall_raw": arms[arm]["dev_overall_raw_mae"],
                "dev_overall_cal": arms[arm]["dev_overall_cal_mae"],
                "dev_G0_raw": arms[arm]["dev_G0_raw_mae"],
                "dev_G0_cal": arms[arm]["dev_G0_cal_mae"],
                "gap_overall_cal": arms[arm]["dev_overall_cal_mae"] - arms[arm]["fit_overall_cal_mae"],
            } for arm in ARMS
        },
        "dev_groups_cal": {arm: arms[arm]["dev_groups_cal"] for arm in ARMS},
        "dev_groups_raw": {arm: arms[arm]["dev_groups_raw"] for arm in ARMS},
        "k_dev_counts": {name: int(mask.sum()) for name, mask in group_masks(k_dev).items()},
        "gains": gains,
        "bootstrap_witnesses": witnesses,
        "gate": gate,
        "m_gate_swapped_roles": m_gate,
        "classification": classification,
        "contribution_identities": identity,
        "sensitivity_drop_M_worst_dev_row": sensitivity,
        "component_diagnostics": component_diag,
        "interventions": interventions,
        "internal_dev_note": (
            "internal dev g-MAE on the rng(20261006) fold; NOT official-valid y-MAE, "
            "not a deployment threshold, not SOTA-comparable"),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(out_dir / "dev_eval.json", result)
    log(
        f"[dev-eval] D G0_cal={arms['D_COMP']['dev_G0_cal_mae']:.6f} "
        f"M G0_cal={arms['M_COMP']['dev_G0_cal_mae']:.6f} "
        f"G0_cal_gain={gains['G0_cal']['point']:.6f} CI=[{gains['G0_cal']['ci95'][0]:.6f}, "
        f"{gains['G0_cal']['ci95'][1]:.6f}] class={classification}"
    )
    return result


def phase_interventions(
    *, out_dir: Path, new: Mapping[str, Any], payload: prev.TuplePayload, kappa_M: float,
    sums: Mapping[str, Mapping[str, np.ndarray]], comps: Mapping[str, Mapping[str, np.ndarray]],
    bias: Mapping[str, float], device: torch.device, g_dev: np.ndarray, k_dev: np.ndarray,
    fit_data: Sequence[Any], dev_data: Sequence[Any], log: Any = print,
) -> dict[str, Any]:
    """Three frozen operator checks per arm on dev, native fit bias, no re-fit.

    1. zero local injection; 2. root code replaced by that arm's fit-roots mean
    code (same kappa/W_loc); 3. J aggregation switched to the marginal-product I
    operator (existing incidence weights; no new features, no I-model training).
    """
    results: dict[str, Any] = {}
    for arm in ARMS:
        state = torch.load(out_dir / f"{arm}_raw_soup_state.pt", map_location="cpu", weights_only=True)
        model = build_arm_comp(arm, payload, kappa_M)
        model.load_state_dict({key: value.to(torch.device("cpu")) for key, value in state.items()}, strict=True)
        model = model.to(device)
        model.eval()
        baseline_sum, _ = evaluate_state_components(model, dev_data, device)
        # fit-roots mean code (frozen, label-free)
        fit_dummy = torch.zeros(len(fit_data), dtype=torch.float32)
        mean_code = None
        acc, count = None, 0
        with torch.no_grad():
            for start in range(0, len(fit_data), 512):
                indices = list(range(start, min(start + 512, len(fit_data))))
                batch = zftd.make_batch(fit_data, indices, fit_dummy, device)
                codes = model.local_tuple.root_codes(batch)
                if acc is None:
                    acc = codes.sum(0)
                else:
                    acc = acc + codes.sum(0)
                count += int(codes.shape[0])
        mean_code = (acc / max(count, 1)).detach()

        def run_variant(mode: str) -> dict[str, Any]:
            enc = model.local_tuple
            if mode == "zero_injection":
                enc.ablate = True
                enc.mean_replace = None
                if hasattr(enc, "switch_independent"):
                    enc.switch_independent = False
                if hasattr(enc, "weight_key"):
                    enc.weight_key = "joint"
            elif mode == "mean_root_code":
                enc.ablate = False
                enc.mean_replace = None
                if hasattr(enc, "switch_independent"):
                    enc.switch_independent = False
                if hasattr(enc, "weight_key"):
                    enc.weight_key = "joint"
                _apply_mean_replace(enc, mean_code, device)
            elif mode == "J_to_I":
                enc.ablate = False
                _clear_mean_replace(enc)
                if hasattr(enc, "switch_independent"):
                    enc.switch_independent = True
                if hasattr(enc, "weight_key"):
                    enc.weight_key = "independent"
            else:
                raise ValueError(mode)
            variant_sum, variant_comp = evaluate_state_components(model, dev_data, device)
            enc.ablate = False
            _clear_mean_replace(enc)
            if hasattr(enc, "switch_independent"):
                enc.switch_independent = False
            if hasattr(enc, "weight_key"):
                enc.weight_key = "joint"
            delta_pred = float(np.max(np.abs(variant_sum - baseline_sum)))
            return {
                "mean_abs_delta_pred": float(np.mean(np.abs(variant_sum - baseline_sum))),
                "max_abs_delta_pred": delta_pred,
                "g_raw_mae": _mae(g_dev - variant_sum),
                "g_cal_mae": _mae(g_dev - (variant_sum + float(bias[arm]))),
                "G0_cal_mae": _mae(g_dev[k_dev == 0] - (variant_sum[k_dev == 0] + float(bias[arm]))),
                "delta_ell_mean_output": float(np.mean(np.abs(
                    variant_comp[:, 0] - comps[arm]["dev"][:, 0]))),
                "delta_s_mean_output": float(np.mean(np.abs(
                    variant_comp[:, 1] - comps[arm]["dev"][:, 1]))),
            }

        baseline = {
            "g_raw_mae": _mae(g_dev - baseline_sum),
            "g_cal_mae": _mae(g_dev - (baseline_sum + float(bias[arm]))),
            "G0_cal_mae": _mae(g_dev[k_dev == 0] - (baseline_sum[k_dev == 0] + float(bias[arm]))),
        }
        results[arm] = {
            "baseline": baseline,
            "zero_local_injection": run_variant("zero_injection"),
            "mean_root_code": run_variant("mean_root_code"),
            "J_to_I_switch": run_variant("J_to_I"),
            "mean_code_source": "fit roots, label-free, per arm frozen soup",
            "note": "sensitivity-only: large zeroing losses include synergy destruction, not information share",
        }
        log(f"[interventions {arm}] zero/mean/J->I G0_cal="
            f"{results[arm]['zero_local_injection']['G0_cal_mae']:.6f}/"
            f"{results[arm]['mean_root_code']['G0_cal_mae']:.6f}/"
            f"{results[arm]['J_to_I_switch']['G0_cal_mae']:.6f}")
    return results


def _apply_mean_replace(enc: nn.Module, mean_code: torch.Tensor, device: torch.device) -> None:
    """Fit-root mean-code replacement, mirroring the M encoder's switch."""
    if hasattr(enc, "mean_replace"):
        enc.mean_replace = mean_code.detach().to(device=device, dtype=torch.float32)
        return
    raise RuntimeError("encoder has no mean_replace switch; resolve in pre-training smoke")


def _clear_mean_replace(enc: nn.Module) -> None:
    if getattr(enc, "mean_replace", None) is not None:
        enc.mean_replace = None


# ---------------------------------------------------------------------------
# 8. mechanism health (offline from the saved checkpoints; fit-only)
# ---------------------------------------------------------------------------


def phase_mechanism_health(*, out_dir: Path = RESULTS_DIR, device: torch.device | None = None, log: Any = print) -> dict[str, Any]:
    torch.set_num_threads(8)
    device = device or torch.device("cpu")
    new = load_new_objects(out_dir)
    payload = prev.TuplePayload(new["payload_arrays"])
    kappa_M = float(new["kappa"]["kappa_M"])
    _, fit_data, _dev_data = build_prepared_data(new)
    d_init = prev.init_d_loc()
    results: dict[str, Any] = {}
    n_probe = 512
    for arm in ARMS:
        arm_health: dict[str, Any] = {}
        for epoch in LOG_EPOCHS:
            path = out_dir / (f"{arm}_init_state.pt" if epoch == 1 else f"{arm}_epoch{epoch}_state.pt")
            state = torch.load(path, map_location="cpu", weights_only=True)
            model = build_arm_comp(arm, payload, kappa_M)
            model.load_state_dict(state, strict=True)
            model = model.to(device)
            model.eval()
            batch = zftd.make_batch(fit_data[:n_probe], list(range(n_probe)), torch.zeros(n_probe), device)
            with torch.no_grad():
                _ = model(batch, mask=cm.C6_MASK)
                enc = model.local_tuple
                stats = {key: value for key, value in enc.last_stats.items() if not torch.is_tensor(value)}
                codes = enc.root_codes(batch)
                std = codes.std(dim=0)
                if hasattr(enc, "D_loc_raw"):
                    drift = float((enc.D_loc_raw - d_init.to(device)).norm())
                    local_norm = float(enc.D_loc_raw.norm())
                else:
                    drift = float((enc.A_raw - d_init.t().to(device)).norm())
                    local_norm = float(enc.A_raw.norm())
                contribution = F.linear(codes * enc.kappa, enc.W_loc)
                injection_rms = float(contribution.pow(2).mean().sqrt())
            arm_health[f"epoch{epoch}"] = {
                "batch_stats": stats,
                "per_tuple_nnz": stats.get("alpha_per_tuple_nonzero"),
                "atoms_used_of_64": stats.get("atoms_used"),
                "atoms_used_fraction": stats.get("atoms_used_fraction"),
                "local_drift_from_init": drift,
                "local_norm": local_norm,
                "root_code_rms": float(codes.pow(2).mean().sqrt()),
                "root_code_effective_rank": int((std > 1e-8).sum()),
                "root_code_dead_dims": int((std < 1e-8).sum()),
                "root_code_all_zero": bool(float(codes.abs().sum()) == 0.0),
                "W_loc_norm": float(enc.W_loc.norm()),
                "injection_rms": injection_rms,
            }
        # gradient liveness from the probe log (fit-only, training-time)
        probes = json.loads((out_dir / f"{arm}_probe_log.json").read_text())
        arm_health["probe_task_grads"] = [
            {key: value for key, value in p.items() if "grad" in key or key in ("epoch",)}
            for p in probes
        ]
        results[arm] = arm_health
        log(f"[health {arm}] ep240 drift={arm_health['epoch240']['local_drift_from_init']:.3f} "
            f"atoms={arm_health['epoch240'].get('atoms_used_of_64')} "
            f"injRMS={arm_health['epoch240']['injection_rms']:.4g}")
    write_json(out_dir / "mechanism_health.json", results)
    return results


# ---------------------------------------------------------------------------
# 9. CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-objects", action="store_true")
    parser.add_argument("--init-check", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--freeze-eval-roster", action="store_true")
    parser.add_argument("--dev-eval", action="store_true")
    parser.add_argument("--mechanism-health", action="store_true")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=str(RESULTS_DIR))
    args = parser.parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    if args.build_objects:
        phase_build_objects(out_dir=out_dir)
    elif args.init_check:
        init_check(out_dir=out_dir, device=device)
    elif args.smoke:
        run_smoke(device=device, out_dir=out_dir)
    elif args.train:
        if not args.arm:
            raise SystemExit("--train requires --arm")
        train_arm(args.arm, out_dir=out_dir, device=device)
    elif args.freeze_eval_roster:
        phase_freeze_eval_roster(out_dir=out_dir)
    elif args.dev_eval:
        phase_dev_eval(out_dir=out_dir, device=device)
    elif args.mechanism_health:
        phase_mechanism_health(out_dir=out_dir, device=device)
    else:
        raise SystemExit("choose a phase")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
