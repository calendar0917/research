"""zinc_cssd_nonlinear_binding_v1 — pre-pooling joint structure–semantics binding on a frozen CSSD basis.

Protocol ``zinc-cssd-nonlinear-binding-v1`` (track ksvd, study zinc-context-gap,
design basis commit 50eaf5f7).  The round fixes the CSSD structural basis and
tests whether a **member-level nonlinear joint mapping** of each node's frozen
structure code ``z_v`` (33-D CSSD) with its atom-type one-hot ``q_v`` (28-D)
— and of each real induced bond's endpoint pair ``([z_v;q_v];[z_w;q_w];b_e)``
— beats a **separated additive binding** of the same marginal inputs, before
occurrence pooling into the fixed 446-D fusion layout of the canonical Full
skeleton.

Five arms (identical body skeleton, identical frozen basis, identical recipe):

* ``A``    — strength reference: the M_COMP + Q task path with the ORIGINAL
             product node/edge binding slots restored on the frozen CSSD code
             (``(z@W_A_S)*(q@W_A_C)`` / ``[cu+cv;|cu-cv|;cu*cv]@W_E_S`` with
             the bond one-hot ``@W_E_C``, slot encoders kept).
* ``C00``  — N_sep (MLP_S(z) 33-44-48 + MLP_C(q) 28-44-48) + E_sep
             (symmetric psi_S 66-53-32 + symmetric psi_C 60-53-32).
* ``C10``  — N_joint (MLP([z;q]) 61-64-48) + E_sep.
* ``C01``  — N_sep + E_joint (symmetric psi 126-64-32 over
             ``[x_v;x_w;b_e]``, ``x=[z;q]``).
* ``C11``  — N_joint + E_joint.

Shared, frozen across all arms and seeds (fit-only): CSSD basis
(U / common RMS / D — refit on the 8001-row fit split, 320 epochs, last-5-epoch
average), input prep, tuple payload + kappa_M, component constants, and one
Q(topology25) head (300 epochs, last-5 soup).  The posterior bridge stays the
matched MLP bridge; the reader is the ComponentReader (39->2); the root-tuple
local MLP (W_loc) injection is kept in every arm.

Body recipe (all arms): 240 epochs, batch 128, Adam 1e-3 / wd 1e-5, clip 5,
fixed seed-0 locked schedule, soup = mean of epochs 236..240, COMP loss
``L = L1(g) + 0.5*(L1(ell)+L1(s))``.  Full prediction
``y_raw = ell_hat + s_hat + Q_raw``; ``y_cal = y_raw + b_y`` with a single
fit-median bias.  select/confirm are never read during training; official
valid/test are never loaded.

Usage (see the runner ``zinc_cssd_nonlinear_binding_v1`` for the registered
entry)::

    python -m tracks.ksvd.experiments.luyin16.zinc_cssd_nonlinear_binding_v1 \
        --build-objects | --cssd-refit | --train-q | --checks | --smoke \
        --train --arm C10 [--seed 0] | --select-eval | --confirm-eval | \
        --interventions
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm
from tracks.ksvd.experiments.luyin16 import (
    e2e_dictenv_common_subspace_dictionary_v1 as cssd,
)
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_h1_clarity_audit as audit
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p1 as p1
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p2_abs as p2
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p2_abs as p2run
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_scale_v1 as sc
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_sem108_v1 as sem
from tracks.ksvd.experiments.luyin16 import sdb_v0 as sdb
from tracks.ksvd.experiments.luyin16 import (
    zinc_chemistry_component_supervision_seed0_v1 as zcs,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_component_supervision_fulltrain_confirmation_seed0_v1 as zffc,
)
from tracks.ksvd.experiments.luyin16 import zinc_full_cycle_target_decomposition_v1 as zftd
from tracks.ksvd.experiments.luyin16 import (
    zinc_local_dictionary_component_supervision_seed0_v1 as zldc,
)
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
    zinc_structure_semantic_factorial_seed0_v1 as zsf,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_task_dictionary_and_cycle_witness_seed0_v1 as zw,
)

PROTOCOL_VERSION = "zinc-cssd-nonlinear-binding-v1"
RESULT_SLUG = "zinc_cssd_nonlinear_binding_v1"
TRACK_ROOT = zw.TRACK_ROOT
RESULTS_DIR = TRACK_ROOT / "results" / RESULT_SLUG
RUNS_DIR = RESULTS_DIR / "runs"

# ---------------------------------------------------------------------------
# frozen recipe (identical for all five arms; actual source constants recorded
# in the stage-0 manifest — verified unchanged from the M_COMP lineage)
# ---------------------------------------------------------------------------

SEED = 0
EPOCHS = 240
LR = float(p2run.LEARNING_RATE)          # 1e-3
WEIGHT_DECAY = float(p2run.WEIGHT_DECAY)  # 1e-5
GRAD_CLIP = float(p2run.GRAD_CLIP)        # 5.0
BATCH_SIZE = int(p2run.BATCH_SIZE)        # 128
SOUP_EPOCHS = tuple(range(236, 241))      # fixed last-5 soup
LOG_EPOCHS = (1, 40, 120, 240)
TRAIN_SHUFFLE_OFFSET = int(zldc.TRAIN_SHUFFLE_OFFSET)
COMPONENT_LOSS_WEIGHT = 0.5
MU_LOGP = zfr.MU_LOGP
SCALE_SEED = 0
BODY_TRAIN_SEED_OFFSET = 0                # body seed s -> torch.manual_seed(s)

ARMS = ("A", "C00", "C10", "C01", "C11")
N_KIND = {"C00": "sep", "C10": "joint", "C01": "sep", "C11": "joint"}
E_KIND = {"C00": "sep", "C10": "sep", "C01": "joint", "C11": "joint"}

# fold: hash/seed 20261006 permutation of the 10000 official-train rows,
# 8000/1000/1000 with SMILES-duplicate groups kept inside one split
# (straddling groups move wholly into fit; actual sizes recorded).
FOLD_SEED = 20261006
N_FIT = 8000
N_SELECT = 1000
N_CONFIRM = 1000

# CSSD basis (frozen, fit-only refit of the existing q1 scheme)
CSSD_Q = 1
CSSD_K_ATOMS = int(cssd.K_ATOMS)          # 32
CSSD_SPARSITY = int(cssd.SPARSITY)        # 8
CSSD_IHT_STEPS = int(cssd.IHT_STEPS)      # 10
CSSD_DICT_SEED = int(sdb.DICT_SEED)       # 20260924
CSSD_KSVD_EPOCHS = 10
CSSD_TRAIN_SEED = 0
CSSD_EPOCHS = 320
CSSD_SOUP_EPOCHS = tuple(range(316, 321))  # last-5 average, never by select/confirm
H1_LAMBDA = float(cm.H1_LAMBDA)

# Q head (frozen after stage 0; shared by every arm and seed)
Q_EPOCHS = 300
Q_HIDDEN = (64, 32)
Q_TOPOLOGY_IN = 25
Q_SOUP_EPOCHS = (296, 297, 298, 299, 300)
Q_TRAIN_GEN_BASE = 20261003
Q_PARAMETERS = 3777

# branch sizes (pre-registered; asserted at construction)
N_JOINT_SIZES = (61, 64, 48)
N_SEP_SIZES_S = (33, 44, 48)
N_SEP_SIZES_C = (28, 44, 48)
E_JOINT_SIZES = (126, 64, 32)
E_SEP_SIZES_S = (66, 53, 32)
E_SEP_SIZES_C = (60, 53, 32)
N_JOINT_PARAMETERS = 7088
N_SEP_PARAMETERS = 7092
E_JOINT_PARAMETERS = 10208
E_SEP_PARAMETERS = 10240

#: private generator seeds for the N/E branch modules (arm-specific, seed-specific)
BRANCH_SEED_BASE = 20261007
BRANCH_SEED_STEP = {"A": 0, "C00": 1, "C10": 2, "C01": 3, "C11": 4}

# mechanism-intervention permutation seeds (post-training only)
INTERVENTION_SEED = 20261008

# paired bootstrap (descriptive molecule-resampling uncertainty only)
BOOT_SEED = 20261011
N_BOOT = 2000

# expected parameter audits (derived from the canonical Full inventory)
CANONICAL_FULL_PARAMETERS = 408651        # sc.FULL audit (CSSD-widened bindings)
FROZEN_DICTIONARY_PARAMETERS = int(cssd.PHI_DIM) * CSSD_K_ATOMS  # 2080 -> buffer
READER_DELTA_PARAMETERS = 40              # 39->2 last layer (+40 vs +0)
LOCAL_TUPLE_PARAMETERS = mlpmod.A_PARAMETERS + mlpmod.W_LOC_PARAMETERS  # 29888


def _mlp_count(fin: int, fh: int, fout: int) -> int:
    return (fin * fh + fh) + (fh * fout + fout)


def _original_slot_path_parameters() -> int:
    """W_A_S/W_A_C/W_E_S/W_E_C + node/edge slot encoders (canonical Full)."""
    d_a = int(p2.D_A)
    d_e = int(cm.H1_CONFIG.d_e)
    node_binding = (1 + cssd.K_ATOMS + p2.ATOM_CATEGORIES) * d_a  # 5856
    edge_binding = (3 * (1 + cssd.K_ATOMS) + p2.BOND_CATEGORIES) * d_e  # 4944
    node_encoder = _mlp_count(d_a, 64, int(p2.ENV_DIM))  # 15568
    edge_encoder = _mlp_count(d_e, 48, 32)  # 3920
    return node_binding + edge_binding + node_encoder + edge_encoder


ORIGINAL_SLOT_PARAMETERS = _original_slot_path_parameters()  # 30288


def _expected_arm_parameters(arm: str) -> int:
    base = (
        CANONICAL_FULL_PARAMETERS
        - FROZEN_DICTIONARY_PARAMETERS
        + READER_DELTA_PARAMETERS
        + LOCAL_TUPLE_PARAMETERS
    )
    if arm == "A":
        return base
    branch = {"C00": N_SEP_PARAMETERS + E_SEP_PARAMETERS,
              "C10": N_JOINT_PARAMETERS + E_SEP_PARAMETERS,
              "C01": N_SEP_PARAMETERS + E_JOINT_PARAMETERS,
              "C11": N_JOINT_PARAMETERS + E_JOINT_PARAMETERS}[arm]
    return base - ORIGINAL_SLOT_PARAMETERS + branch


# ---------------------------------------------------------------------------
# small utilities (same conventions as the parent rounds)
# ---------------------------------------------------------------------------


def _hash_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def array_sha256(array: np.ndarray, dtype: Any = np.float32) -> str:
    return _hash_bytes(np.ascontiguousarray(np.asarray(array, dtype)).tobytes())


def state_hash(state: Mapping[str, torch.Tensor]) -> str:
    hasher = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        hasher.update(key.encode("utf-8"))
        hasher.update(np.asarray(value.numpy(), np.float32).tobytes())
    return hasher.hexdigest()


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def seed_everything(seed: int) -> None:
    p2run._seed_everything(int(seed))


def resolve_device(name: str | None) -> torch.device:
    if name is None or name == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError(f"GPU requested ({name!r}) but CUDA is not available; no silent CPU fallback")
    return torch.device(name)


def allocation_probe(device: torch.device) -> dict[str, Any]:
    return zldc.allocation_probe(device)


def _linear_from_generator(fin: int, fout: int, generator: torch.Generator) -> nn.Linear:
    """``nn.Linear`` with the default uniform init drawn from a private generator."""
    lin = nn.Linear(int(fin), int(fout))
    bound = 1.0 / math.sqrt(int(fin))
    with torch.no_grad():
        lin.weight.uniform_(-bound, bound, generator=generator)
        lin.bias.uniform_(-bound, bound, generator=generator)
    return lin


def _mlp_from_generator(sizes: Sequence[int], generator: torch.Generator) -> nn.Sequential:
    layers: list[nn.Module] = []
    for fin, fout in zip(sizes[:-1], sizes[1:]):
        layers.append(_linear_from_generator(fin, fout, generator))
        layers.append(nn.SiLU())
    if not layers:
        raise ValueError("empty mlp")
    return nn.Sequential(*layers[:-1])  # affine-SiLU-affine (two affine layers)


def _branch_parameter_count(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


# ---------------------------------------------------------------------------
# 1. fold: 8000/1000/1000 from rng(20261006), SMILES groups kept unsplit
# ---------------------------------------------------------------------------


def build_fold() -> dict[str, Any]:
    """New three-way fold, fixed once.

    ``perm = np.random.default_rng(20261006).permutation(10000)``; fit =
    ``perm[:8000]``, select = ``perm[8000:9000]``, confirm = ``perm[9000:]``
    (each sorted), then every SMILES-duplicate group that straddles a boundary
    is moved wholly into fit.  Actual sizes and every moved row are recorded.
    """
    import pandas as pd

    lab = pd.read_csv(zfr.TRAIN_LABEL_CSV)
    if len(lab) != 10000:
        raise RuntimeError(f"train label csv has {len(lab)} rows")
    if not np.array_equal(lab["subset_index"].to_numpy(np.int64), np.arange(10000)):
        raise RuntimeError("train label csv rows are not positional")
    if not (lab["split"].astype(str) == "train").all():
        raise RuntimeError("train label csv contains non-train rows")
    smi_line = lab["smi_line"].to_numpy(np.int64)
    if int((smi_line < 0).sum()):
        raise RuntimeError("some train rows have no smiles line")

    values, counts = np.unique(smi_line, return_counts=True)
    dup_values = values[counts > 1]
    dup_groups = {int(v): np.where(smi_line == v)[0].astype(np.int64) for v in dup_values}

    perm = np.random.default_rng(FOLD_SEED).permutation(10000)
    fit_idx = np.sort(perm[:N_FIT]).astype(np.int64)
    select_idx = np.sort(perm[N_FIT:N_FIT + N_SELECT]).astype(np.int64)
    confirm_idx = np.sort(perm[N_FIT + N_SELECT:]).astype(np.int64)

    in_fit = np.zeros(10000, dtype=bool)
    in_fit[fit_idx] = True
    moved: list[dict[str, Any]] = []
    for value, members in dup_groups.items():
        flags = in_fit[members]
        if bool(flags.all()) or not bool(flags.any()):
            continue  # wholly inside fit or wholly inside one holdout split
        incoming = members[~flags]
        moved.append({
            "smi_line": int(value),
            "group_size": int(members.size),
            "rows_moved_into_fit": incoming.astype(int).tolist(),
            "source_split": sorted(
                {("select" if i in set(select_idx.tolist()) else "confirm") for i in incoming.tolist()}
            ),
        })
        fit_idx = np.union1d(fit_idx, incoming)
        select_idx = np.setdiff1d(select_idx, incoming)
        confirm_idx = np.setdiff1d(confirm_idx, incoming)
        in_fit[incoming] = True
    fit_idx = np.sort(fit_idx).astype(np.int64)
    select_idx = np.sort(select_idx).astype(np.int64)
    confirm_idx = np.sort(confirm_idx).astype(np.int64)

    checks = {
        "sizes": {"fit": int(fit_idx.size), "select": int(select_idx.size), "confirm": int(confirm_idx.size)},
        "disjoint": bool(
            np.intersect1d(fit_idx, select_idx).size == 0
            and np.intersect1d(fit_idx, confirm_idx).size == 0
            and np.intersect1d(select_idx, confirm_idx).size == 0
        ),
        "cover": bool(np.union1d(np.union1d(fit_idx, select_idx), confirm_idx).size == 10000),
        "n_duplicate_smi_groups": int(len(dup_groups)),
        "duplicate_group_sizes": sorted(int(m.size) for m in dup_groups.values()),
        "n_groups_moved": int(len(moved)),
        "smi_groups_unsplit": True,
    }
    if not (checks["disjoint"] and checks["cover"]):
        raise RuntimeError(f"fold construction check failed: {checks}")
    return {
        "fit_idx": fit_idx,
        "select_idx": select_idx,
        "confirm_idx": confirm_idx,
        "definition": (
            "perm=np.random.default_rng(20261006).permutation(10000); "
            "fit=sort(perm[:8000]); select=sort(perm[8000:9000]); confirm=sort(perm[9000:]); "
            "straddling SMILES-duplicate groups moved wholly into fit"
        ),
        "dup_groups": {str(k): v.tolist() for k, v in dup_groups.items()},
        "moved": moved,
        "checks": checks,
        "fit_sha256": array_sha256(fit_idx, np.int64),
        "select_sha256": array_sha256(select_idx, np.int64),
        "confirm_sha256": array_sha256(confirm_idx, np.int64),
    }


def load_fold(out_dir: Path = RESULTS_DIR) -> dict[str, Any]:
    with np.load(out_dir / "fold.npz", allow_pickle=False) as z:
        fold = {
            "fit_idx": z["fit_idx"].astype(np.int64),
            "select_idx": z["select_idx"].astype(np.int64),
            "confirm_idx": z["confirm_idx"].astype(np.int64),
        }
    manifest = read_json(out_dir / "fold_manifest.json")
    for key, column in (("fit_sha256", "fit_idx"), ("select_sha256", "select_idx"), ("confirm_sha256", "confirm_idx")):
        if array_sha256(fold[column], np.int64) != manifest[key]:
            raise RuntimeError(f"fold {column} sha256 mismatch")
    fold["manifest"] = manifest
    return fold


#: the zldc prep helpers take a two-key fold mapping; select stands in for dev
#: (only used for recorded diagnostics, never for fitting).
def _zldc_fold_view(fold: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return {"fit_idx": np.asarray(fold["fit_idx"], np.int64), "dev_idx": np.asarray(fold["select_idx"], np.int64)}


# ---------------------------------------------------------------------------
# 2. phase-A objects: targets / tuple payload / kappa_M / prep (fit-only)
# ---------------------------------------------------------------------------


def build_targets(fold: Mapping[str, Any]) -> dict[str, Any]:
    """Component constants refit on the new fit rows (zldc.build_new_targets)."""
    return zldc.build_new_targets(_zldc_fold_view(fold))


def build_payload(fold: Mapping[str, Any]) -> dict[str, Any]:
    """Tuple incidence reused; phi scaler + kappa_D refit on the new fit roots."""
    return zldc.build_new_payload(_zldc_fold_view(fold))


def compute_kappa_M(payload_arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    return zldc.compute_kappa_M_new(payload_arrays)


def build_prep(train_data: Sequence[Any], fold: Mapping[str, Any]) -> dict[str, Any]:
    return zldc.build_new_prep(train_data, _zldc_fold_view(fold))


def build_prepared_data(objects: Mapping[str, Any], *, verify_prep: bool = True):
    """Train-only cache + new-fit prep + three-way split (never official valid)."""
    train_data = zftd.load_train_only()
    if len(train_data) != 10000:
        raise RuntimeError("train-only cache length mismatch")
    fold = objects["fold"]
    prep_meta = build_prep(train_data, fold)
    if verify_prep:
        for key, value in objects["prep"].items():
            if key not in prep_meta or not np.array_equal(
                np.asarray(prep_meta[key], np.float32), np.asarray(value, np.float32)
            ):
                raise RuntimeError(f"prep mismatch at {key}")
    fit_idx = np.asarray(fold["fit_idx"], np.int64)
    select_idx = np.asarray(fold["select_idx"], np.int64)
    confirm_idx = np.asarray(fold["confirm_idx"], np.int64)
    fit_data = [train_data[int(i)] for i in fit_idx.tolist()]
    select_data = [train_data[int(i)] for i in select_idx.tolist()]
    confirm_data = [train_data[int(i)] for i in confirm_idx.tolist()]
    for position, index in enumerate(fit_idx.tolist()):
        fit_data[position].local_mol_id = torch.tensor([int(index)], dtype=torch.long)
    for position, index in enumerate(select_idx.tolist()):
        select_data[position].local_mol_id = torch.tensor([int(index)], dtype=torch.long)
    for position, index in enumerate(confirm_idx.tolist()):
        confirm_data[position].local_mol_id = torch.tensor([int(index)], dtype=torch.long)
    return prep_meta, fit_data, select_data, confirm_data


def phase_build_objects(*, out_dir: Path = RESULTS_DIR, log: Any = print) -> dict[str, Any]:
    """Stage-0 phase A: fold + targets + payload + kappa_M + prep, fit-only."""
    torch.set_num_threads(8)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    fold = build_fold()
    np.savez_compressed(
        out_dir / "fold.npz",
        fit_idx=fold["fit_idx"], select_idx=fold["select_idx"], confirm_idx=fold["confirm_idx"],
    )
    write_json(out_dir / "fold_manifest.json", {k: v for k, v in fold.items() if k != "dup_groups"} | {
        "dup_groups": fold["dup_groups"],
        "protocol_version": PROTOCOL_VERSION,
    })

    targets = build_targets(fold)
    t = targets["arrays"]
    np.savez_compressed(
        out_dir / "targets.npz",
        y=t["y"], c=t["c"], g=t["g"], k=t["k"], ell=t["ell"], s=t["s"],
        gid=targets["gid"],
        constants=np.asarray(
            [targets["constants"][name] for name in ("sigma_logP", "sigma_SA", "mu_SA", "sigma_cycle", "mu_cycle")],
            np.float64,
        ),
        constant_names=np.asarray(["sigma_logP", "sigma_SA", "mu_SA", "sigma_cycle", "mu_cycle"]),
        mu_logP=np.asarray(MU_LOGP, np.float64),
    )
    payload = build_payload(fold)
    np.savez_compressed(out_dir / "tuple_payload.npz", **payload["arrays"])
    kappa = compute_kappa_M(payload["arrays"])
    write_json(out_dir / "kappa_M.json", kappa)
    train_data = zftd.load_train_only()
    prep_meta = build_prep(train_data, fold)
    np.savez_compressed(
        out_dir / "prep.npz",
        patch_fit_mean=prep_meta["patch_fit_mean"], patch_fit_scale=prep_meta["patch_fit_scale"],
        ctx_fit_mean=prep_meta["ctx_fit_mean"], ctx_fit_scale=prep_meta["ctx_fit_scale"],
        anchor_fit_mean=prep_meta["anchor_fit_mean"], anchor_fit_scale=prep_meta["anchor_fit_scale"],
        topo_fit_mean=prep_meta["topo_fit_mean"], topo_fit_scale=prep_meta["topo_fit_scale"],
    )
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "stage": "build-objects",
        "seconds": float(time.perf_counter() - started),
        "fold": {
            "definition": fold["definition"],
            "fit_sha256": fold["fit_sha256"],
            "select_sha256": fold["select_sha256"],
            "confirm_sha256": fold["confirm_sha256"],
            "checks": fold["checks"],
            "moved": fold["moved"],
        },
        "constants": targets["constants"],
        "mu_logP": MU_LOGP,
        "target_checks": targets["checks"],
        "label_source_checks": targets["label_checks"],
        "payload_checks": payload["checks"],
        "kappa_M": kappa,
        "recipe": {
            "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR, "weight_decay": WEIGHT_DECAY,
            "grad_clip": GRAD_CLIP, "soup_epochs": list(SOUP_EPOCHS),
            "component_loss_weight": COMPONENT_LOSS_WEIGHT,
            "source_constants": {
                "p2run.LEARNING_RATE": float(p2run.LEARNING_RATE),
                "p2run.WEIGHT_DECAY": float(p2run.WEIGHT_DECAY),
                "p2run.GRAD_CLIP": float(p2run.GRAD_CLIP),
                "p2run.BATCH_SIZE": int(p2run.BATCH_SIZE),
                "p2run.SOUP_K": int(p2run.SOUP_K),
                "cm.H1_LAMBDA": float(cm.H1_LAMBDA),
            },
        },
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "artifacts": {
            name: {"sha256": file_sha256(out_dir / name)}
            for name in ("fold.npz", "targets.npz", "tuple_payload.npz", "kappa_M.json", "prep.npz")
        },
    }
    write_json(out_dir / "objects_manifest.json", manifest)
    log(f"[phase-build-objects] done in {manifest['seconds']:.1f}s "
        f"(fit={fold['checks']['sizes']['fit']} select={fold['checks']['sizes']['select']} "
        f"confirm={fold['checks']['sizes']['confirm']}, moved={fold['checks']['n_groups_moved']})")
    return manifest


def load_objects(out_dir: Path = RESULTS_DIR) -> dict[str, Any]:
    """Load the stage-0 objects with manifest hash checks."""
    manifest = read_json(out_dir / "objects_manifest.json")
    for name in ("fold.npz", "targets.npz", "tuple_payload.npz", "prep.npz"):
        expected = manifest["artifacts"][name]["sha256"]
        actual = file_sha256(out_dir / name)
        if actual != expected:
            raise RuntimeError(f"stage-0 artifact {name} sha256 {actual} != frozen {expected}")
    fold = load_fold(out_dir)
    with np.load(out_dir / "targets.npz", allow_pickle=False) as z:
        targets = {key: z[key] for key in ("y", "c", "g", "k", "ell", "s", "gid")}
        constants = {str(n): float(v) for n, v in zip(z["constant_names"].tolist(), z["constants"].tolist())}
    with np.load(out_dir / "tuple_payload.npz", allow_pickle=False) as z:
        payload_arrays = {key: z[key] for key in z.files}
    with np.load(out_dir / "prep.npz", allow_pickle=False) as z:
        prep = {key: z[key] for key in z.files}
    kappa = read_json(out_dir / "kappa_M.json")
    return {
        "fold": fold, "targets": targets, "constants": constants,
        "payload_arrays": payload_arrays, "prep": prep, "kappa": kappa,
        "manifest": manifest,
    }


# ---------------------------------------------------------------------------
# 3. CSSD basis refit (fit-only; CPU-faithful to train_cssd, last-5 soup)
# ---------------------------------------------------------------------------


def _fit_phi_rows(fold: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Per-node phi rows of the fit molecules (env cache, train-only)."""
    phi_env, _atom_env, node_sizes_env = prev._env_phi_atom()
    fit_idx = np.asarray(fold["fit_idx"], np.int64)
    lo = np.concatenate([[0], np.cumsum(node_sizes_env)])
    rows = np.concatenate([phi_env[lo[i]:lo[i + 1]] for i in fit_idx.tolist()], axis=0)
    checks = {
        "n_fit_molecules": int(fit_idx.size),
        "n_phi_rows": int(rows.shape[0]),
        "expected_rows": int(lo[fit_idx].astype(np.int64).sum()) if False else int(sum(int(node_sizes_env[i]) for i in fit_idx.tolist())),
        "phi_dim": int(rows.shape[1]),
        "source": "results/e2e_dictenv_p1/cache/env_train.pt (train-only)",
    }
    if checks["n_phi_rows"] != checks["expected_rows"]:
        raise RuntimeError("fit phi row count mismatch")
    return rows, checks


def cssd_refit(*, out_dir: Path = RESULTS_DIR, threads: int = 8, log: Any = print) -> dict[str, Any]:
    """Refit the CSSD q1 basis on the fit split only.

    Faithful to ``cssd.train_cssd`` semantics (loss ``L1(pred, y) +
    H1_LAMBDA * rec``, Adam 1e-3/1e-5, clip 5, batch 128, locked shuffle
    offsets, C6 mask) with exactly two pre-registered deviations, both
    mandated by the round contract:

    * the dictionary is initialised from a K-SVD fit on the **fit** phi rows
      (never the historical all-10k sdb32, never valid);
    * the soup is the **last-5-epoch average** of D (epochs 316..320), never
      a Top-5 selected on any holdout.  Monitoring runs on fit rows only.

    CPU-only by construction (the historical runner is CPU-only; refitting on
    CPU keeps the numerical/optimisation semantics bit-comparable without any
    GPU port, so no small-batch GPU reference comparison is needed).
    """
    torch.set_num_threads(int(threads))
    out_dir.mkdir(parents=True, exist_ok=True)
    objects = load_objects(out_dir)
    fold = objects["fold"]
    started = time.perf_counter()

    phi_rows, phi_checks = _fit_phi_rows(fold)
    D_init, ksvd_meta = sdb.fit_ksvd(
        phi_rows, atoms=CSSD_K_ATOMS, s=CSSD_SPARSITY,
        epochs=CSSD_KSVD_EPOCHS, seed=CSSD_DICT_SEED,
    )
    subspace = cssd.build_common_subspace(phi_rows, q=CSSD_Q)

    # fit data with the new-fit prep (the body's own view of the reader inputs)
    _prep_meta, fit_data, _select_data, _confirm_data = build_prepared_data(objects)
    y_fit = np.asarray(objects["targets"]["y"], np.float64)[fold["fit_idx"]]
    for position, row in enumerate(fit_data):
        if abs(float(getattr(row, "y", y_fit[position])) - float(y_fit[position])) > 1e-12:
            raise RuntimeError("fit data .y != y_stored on fit rows")
    # monitoring loader on a fixed deterministic fit subset (informational
    # only: the soup is the last-5 average, never selected by the monitor;
    # the full-fit monitor was too slow at ~8x the historical valid pass)
    MONITOR_ROWS = 1024
    monitor_data = fit_data[:MONITOR_ROWS]
    eval_loader = p1.make_env_loader(
        monitor_data, int(p2run.BATCH_SIZE), False, int(CSSD_TRAIN_SEED) + int(p2run.EVAL_SHUFFLE_OFFSET)
    )

    seed_everything(CSSD_TRAIN_SEED)
    model = cssd.build_cssd_model(np.asarray(D_init, np.float32), CSSD_TRAIN_SEED, subspace)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(p2run.LEARNING_RATE), weight_decay=float(p2run.WEIGHT_DECAY))
    loader = p1.make_env_loader(
        fit_data, int(p2run.BATCH_SIZE), True, int(CSSD_TRAIN_SEED) + int(p2run.TRAIN_SHUFFLE_OFFSET)
    )
    mask = cm.C6_MASK
    lam = float(cm.H1_LAMBDA)

    device = torch.device("cpu")
    curve: list[dict[str, Any]] = []
    epoch_D: dict[int, torch.Tensor] = {}
    soup_epochs = set(int(e) for e in CSSD_SOUP_EPOCHS)
    monitor_started = time.perf_counter()
    for epoch in range(1, CSSD_EPOCHS + 1):
        model.train()
        task_sum = rec_sum = rec_term_sum = 0.0
        n_mol = n_nodes = n_batches = 0
        for batch in loader:
            batch = batch.to(device)
            prediction, aux = model(batch, mask=mask, return_aux=True)
            rec = model.reconstruction_loss(aux["phi"], aux["coord"])
            loss = F.l1_loss(prediction.view(-1), batch.y.view(-1)) + lam * rec
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(p2run.GRAD_CLIP))
            optimizer.step()
            task_sum += float((prediction.view(-1) - batch.y.view(-1)).abs().sum())
            n_mol += int(batch.y.numel())
            rec_term_sum += float(rec.detach())
            n_batches += 1
            phi = aux["phi"]
            phi_hat = model.reconstruct(phi, aux["coord"]).detach()
            from tracks.ksvd.code import tccd_v0 as v0

            rec_sum += float((((phi - phi_hat) ** 2).sum(dim=1) / ((phi**2).sum(dim=1) + v0.EPS)).sum())
            n_nodes += int(phi.shape[0])
        train_mae = float(task_sum / max(n_mol, 1))
        train_rec = float(rec_sum / max(n_nodes, 1))
        valid = audit._evaluate_model(model, eval_loader, device, mask)
        curve.append({
            "epoch": int(epoch),
            "train_mae": train_mae,
            "train_rec": train_rec,
            "train_rec_term": float(rec_term_sum / max(n_batches, 1)),
            "fit_monitor_mae": float(valid["mae"]),
            "d_norm": float(model.D.detach().norm()),
        })
        if epoch in soup_epochs:
            epoch_D[int(epoch)] = model.D.detach().cpu().clone()
        if log and (epoch == 1 or epoch % 20 == 0 or epoch == CSSD_EPOCHS):
            log(f"[cssd-refit] ep={epoch:03d} train={train_mae:.6f} rec={train_rec:.3e} "
                f"fit_monitor={float(valid['mae']):.6f} d_norm={curve[-1]['d_norm']:.4f}", flush=True)

    members = sorted(soup_epochs)
    if sorted(epoch_D) != members:
        raise RuntimeError("cssd soup epochs missing")
    D_frozen = torch.stack([epoch_D[e] for e in members]).mean(0)
    U = np.asarray(subspace.components, np.float32)
    common_rms = np.asarray(subspace.rms, np.float32)
    basis = {
        "U": U,
        "common_rms": common_rms,
        "D": D_frozen.numpy().astype(np.float32),
    }
    np.savez_compressed(out_dir / "cssd_basis.npz", **basis)
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "stage": "cssd-refit",
        "seconds": float(time.perf_counter() - started),
        "device": "cpu",
        "threads": int(threads),
        "scheme": {
            "q": CSSD_Q, "k_atoms": CSSD_K_ATOMS, "sparsity": CSSD_SPARSITY,
            "iht_steps": CSSD_IHT_STEPS, "h1_lambda": H1_LAMBDA,
            "epochs": CSSD_EPOCHS, "soup": "mean of D over epochs 316..320 (never selected on holdout)",
            "ksvd": {"epochs": CSSD_KSVD_EPOCHS, "seed": CSSD_DICT_SEED, "meta": ksvd_meta},
            "train_seed": CSSD_TRAIN_SEED,
            "deviations_from_train_cssd": [
                "dictionary initialised from K-SVD on fit phi rows (not the all-10k sdb32)",
                "last-5-epoch D soup (not Top-5 by a holdout)",
                "monitoring on fit rows only (official valid never loaded)",
            ],
        },
        "phi_checks": phi_checks,
        "monitor_rows": 1024,
        "monitor_note": "informational only; the soup is the last-5 average, never selected by the monitor",
        "U_sha256": array_sha256(U),
        "common_rms_sha256": array_sha256(common_rms),
        "D_init_sha256": array_sha256(np.asarray(D_init, np.float32)),
        "D_frozen_sha256": array_sha256(D_frozen.numpy()),
        "fit_monitor_mae_last": curve[-1]["fit_monitor_mae"],
        "curve_sha256": _hash_bytes(json.dumps(curve).encode("utf-8")),
        "fit_only": True,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(out_dir / "cssd_refit.json", manifest)
    write_json(out_dir / "cssd_refit_curve.json", curve)
    log(f"[cssd-refit] done in {manifest['seconds']:.1f}s; D_frozen sha={manifest['D_frozen_sha256'][:16]}…")
    return manifest


def load_cssd_basis(out_dir: Path = RESULTS_DIR) -> dict[str, np.ndarray]:
    manifest = read_json(out_dir / "cssd_refit.json")
    with np.load(out_dir / "cssd_basis.npz", allow_pickle=False) as z:
        basis = {key: z[key] for key in ("U", "common_rms", "D")}
    for key, name in (("U_sha256", "U"), ("common_rms_sha256", "common_rms"), ("D_frozen_sha256", "D")):
        if array_sha256(basis[name]) != manifest[key]:
            raise RuntimeError(f"cssd basis {name} sha256 mismatch")
    return basis


# ---------------------------------------------------------------------------
# 4. shared Q head (fit-only, 300 epochs, last-5 soup)
# ---------------------------------------------------------------------------


def build_q_head(seed: int, bias_value: float) -> nn.Module:
    return zffc.build_q_head(int(seed), float(bias_value))


def q_forward(head: nn.Module, T: torch.Tensor) -> torch.Tensor:
    return zffc.q_forward(head, T)


def topology_matrix(data_list: Sequence[Any]) -> np.ndarray:
    return zffc.topology_matrix(data_list)


def train_q(*, out_dir: Path = RESULTS_DIR, log: Any = print) -> dict[str, Any]:
    """One shared Q(topology25->64->32->1) on the fit rows only."""
    torch.set_num_threads(8)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    objects = load_objects(out_dir)
    fold = objects["fold"]
    _prep_meta, fit_data, _select_data, _confirm_data = build_prepared_data(objects)
    c = np.asarray(objects["targets"]["c"], np.float64)
    c_fit = c[fold["fit_idx"]]
    T = topology_matrix(fit_data)
    if T.shape != (len(fit_data), Q_TOPOLOGY_IN):
        raise RuntimeError(f"fit topology matrix shape {T.shape}")
    bias_value = float(np.median(c_fit))

    head = build_q_head(SEED, bias_value)
    init_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
    optimizer = torch.optim.Adam(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    generator = torch.Generator().manual_seed(int(Q_TRAIN_GEN_BASE) + SEED)
    T_t = torch.as_tensor(T, dtype=torch.float32)
    c_t = torch.as_tensor(c_fit, dtype=torch.float32)
    n = int(T_t.shape[0])
    soup: dict[int, dict[str, torch.Tensor]] = {}
    soup_epochs = set(int(e) for e in Q_SOUP_EPOCHS)
    curve: list[dict[str, Any]] = []
    for epoch in range(1, Q_EPOCHS + 1):
        head.train()
        order = torch.randperm(n, generator=generator)
        abs_sum = 0.0
        grad_norm = 0.0
        for start in range(0, n, BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            prediction = q_forward(head, T_t[idx])
            loss = (prediction - c_t[idx]).abs().mean()
            optimizer.zero_grad()
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(head.parameters(), GRAD_CLIP))
            optimizer.step()
            abs_sum += float((prediction - c_t[idx]).abs().sum())
        if epoch in soup_epochs:
            soup[epoch] = {k: v.detach().clone() for k, v in head.state_dict().items()}
        curve.append({
            "epoch": int(epoch),
            "train_task_mae": float(abs_sum / n),
            "grad_norm": grad_norm,
        })
        if log and (epoch == 1 or epoch % 50 == 0 or epoch == Q_EPOCHS):
            log(f"[train-q] ep={epoch:03d} L1={curve[-1]['train_task_mae']:.6f} "
                f"gnorm={grad_norm:.3g}", flush=True)
    members = sorted(soup)
    soup_state = {key: torch.stack([soup[e][key].float() for e in members]).mean(0) for key in soup[members[0]]}
    last_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    torch.save(init_state, out_dir / "Q_init_state.pt")
    torch.save(last_state, out_dir / "Q_last_state.pt")
    torch.save(soup_state, out_dir / "Q_soup_state.pt")
    n_params = int(sum(p.numel() for p in head.parameters()))
    if n_params != Q_PARAMETERS:
        raise RuntimeError(f"Q parameter audit failed: {n_params}")
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "stage": "train-q",
        "seconds": float(time.perf_counter() - started),
        "device": "cpu",
        "n_fit": int(len(fit_data)),
        "q_parameters": n_params,
        "bias_value": bias_value,
        "epochs": Q_EPOCHS,
        "soup_epochs": list(Q_SOUP_EPOCHS),
        "train_gen_seed": int(Q_TRAIN_GEN_BASE) + SEED,
        "init_state_sha256": state_hash(init_state),
        "soup_state_sha256": state_hash(soup_state),
        "last_train_mae": curve[-1]["train_task_mae"],
        "fit_only": True,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(out_dir / "Q_meta.json", manifest)
    log(f"[train-q] done in {manifest['seconds']:.1f}s; soup sha={manifest['soup_state_sha256'][:16]}…")
    return manifest


def load_q_soup(out_dir: Path = RESULTS_DIR) -> dict[str, torch.Tensor]:
    manifest = read_json(out_dir / "Q_meta.json")
    state = torch.load(out_dir / "Q_soup_state.pt", map_location="cpu", weights_only=False)
    if state_hash(state) != manifest["soup_state_sha256"]:
        raise RuntimeError("Q soup state hash mismatch")
    return state


# ---------------------------------------------------------------------------
# 5. the N/E member-level binding branches (private generators only)
# ---------------------------------------------------------------------------

NODE_SLOT_WIDTH = 48   # N branch output = per-shell slot width (p2.ENV_DIM / 3)
EDGE_SLOT_WIDTH = 32   # E branch output = per-shellpair slot width (cm d_e)
N_SHELLS = int(p2.N_SHELLS)
SHELLPAIR_CLASSES = int(p2.SHELLPAIR_CLASSES)
ATOM_CATEGORIES = int(p2.ATOM_CATEGORIES)
BOND_CATEGORIES = int(p2.BOND_CATEGORIES)


class NodeBranch(nn.Module):
    """Member-level node binding, evaluated per node-occurrence row.

    ``joint``: ``N(v) = MLP([z_v; q_v])`` (61->64->48, affine-SiLU-affine).
    ``sep``:   ``N(v) = MLP_S(z_v) + MLP_C(q_v)`` (33-44-48 + 28-44-48).
    The pooled ``(root, shell)`` sum of ``sep`` cannot see the z/q pairing.
    """

    def __init__(self, kind: str, generator: torch.Generator) -> None:
        super().__init__()
        if kind == "joint":
            self.net = _mlp_from_generator(N_JOINT_SIZES, generator)
            expected = N_JOINT_PARAMETERS
        elif kind == "sep":
            self.net_s = _mlp_from_generator(N_SEP_SIZES_S, generator)
            self.net_c = _mlp_from_generator(N_SEP_SIZES_C, generator)
            expected = N_SEP_PARAMETERS
        else:
            raise ValueError(f"unknown node branch kind {kind!r}")
        self.kind = str(kind)
        count = _branch_parameter_count(self)
        if count != expected:
            raise RuntimeError(f"node branch parameter audit failed: {count} != {expected}")

    def forward(self, z: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        if self.kind == "joint":
            return self.net(torch.cat([z, q], dim=1))
        return self.net_s(z) + self.net_c(q)


class EdgeBranch(nn.Module):
    """Endpoint-paired edge binding, evaluated per bond-occurrence row.

    ``joint``: ``E(e) = 0.5*(psi([x_v;x_w;b_e]) + psi([x_w;x_v;b_e]))``,
    ``x = [z;q]`` (psi 126->64->32).
    ``sep``:   ``0.5*(psi_S([z_v;z_w]) + psi_S([z_w;z_v]))
               + 0.5*(psi_C([q_v;q_w;b_e]) + psi_C([q_w;q_v;b_e]))``
    (66-53-32 + 60-53-32).  Both are endpoint-swap symmetric; ``sep`` never
    joins a structure endpoint to a chemistry endpoint of the same bond.
    """

    def __init__(self, kind: str, generator: torch.Generator) -> None:
        super().__init__()
        if kind == "joint":
            self.net = _mlp_from_generator(E_JOINT_SIZES, generator)
            expected = E_JOINT_PARAMETERS
        elif kind == "sep":
            self.net_s = _mlp_from_generator(E_SEP_SIZES_S, generator)
            self.net_c = _mlp_from_generator(E_SEP_SIZES_C, generator)
            expected = E_SEP_PARAMETERS
        else:
            raise ValueError(f"unknown edge branch kind {kind!r}")
        self.kind = str(kind)
        count = _branch_parameter_count(self)
        if count != expected:
            raise RuntimeError(f"edge branch parameter audit failed: {count} != {expected}")

    def forward(
        self, z_v: torch.Tensor, z_w: torch.Tensor, q_v: torch.Tensor, q_w: torch.Tensor, b_e: torch.Tensor
    ) -> torch.Tensor:
        if self.kind == "joint":
            x_v = torch.cat([z_v, q_v], dim=1)
            x_w = torch.cat([z_w, q_w], dim=1)
            psi = self.net
            return 0.5 * (
                psi(torch.cat([x_v, x_w, b_e], dim=1)) + psi(torch.cat([x_w, x_v, b_e], dim=1))
            )
        psi_s = self.net_s
        psi_c = self.net_c
        return 0.5 * (
            psi_s(torch.cat([z_v, z_w], dim=1)) + psi_s(torch.cat([z_w, z_v], dim=1))
        ) + 0.5 * (
            psi_c(torch.cat([q_v, q_w, b_e], dim=1)) + psi_c(torch.cat([q_w, q_v, b_e], dim=1))
        )


def branch_seed_for(arm: str, seed: int) -> int:
    if arm not in BRANCH_SEED_STEP:
        raise ValueError(arm)
    return int(BRANCH_SEED_BASE) + 100 * int(BRANCH_SEED_STEP[arm]) + int(seed)


# ---------------------------------------------------------------------------
# 6. the five-arm model on the frozen CSSD basis
# ---------------------------------------------------------------------------


class CSSDBindingFull(zsf.FactorialFull):
    """Canonical Full skeleton + frozen CSSD + M_COMP/Q task path + arm slots.

    Construction consumes exactly the canonical Full global stream (the
    caller seeds it with the body seed); every post-hoc surgery consumes no
    global draws:

    * ``D`` -> frozen buffer (the shared CSSD dictionary; never trained);
    * the task-dictionary bridge -> the matched MLP bridge (same widths);
    * ``local_tuple`` -> the M root-tuple encoder (private FRAME_SEED frame,
      zero ``W_loc``; identical to the M_COMP lineage);
    * ``reader`` -> ComponentReader (39->2, private fork_rng, g = ell + s
      exactly);
    * arm ``A`` keeps the original product node/edge binding modules;
      the C arms delete them and attach ``node_branch`` / ``edge_branch``
      built from an isolated private generator.
    """

    def __init__(
        self,
        payload: prev.TuplePayload,
        kappa_M: float,
        arm: str,
        *,
        dictionary: np.ndarray,
        subspace: cssd.CommonSubspace,
        branch_seed: int,
    ) -> None:
        super().__init__(
            cm.H1_CONFIG,
            np.asarray(dictionary, np.float32),
            subspace=subspace,
            spec=sc.FULL,
            coding_mode="sparse",
            binding_mode="paired",
            scale_seed=SCALE_SEED,
        )
        if int(self.fusion[0].in_features) != int(sc.FULL.fusion_in):
            raise RuntimeError("unexpected fusion input width")
        # freeze the CSSD dictionary (buffer, never a Parameter)
        frozen_d = self.D.detach().clone()
        delattr(self, "D")
        self.register_buffer("D", frozen_d)
        # matched MLP posterior bridge (M_COMP lineage)
        bridge = self.local_dictionary_bridge
        zw_frames = (bridge.D_L.detach().clone(), bridge.V_L.detach().clone())
        self.local_dictionary_bridge = zw.MLPBridge(zw_frames[0], zw_frames[1])
        # root-tuple local MLP (M encoder, private frame generator)
        self.local_tuple = mlpmod.LocalTupleEncoderM(
            payload, prev.init_d_loc().t().contiguous()
        )
        self.local_tuple.kappa = float(kappa_M)
        # component reader (39 -> 2, private fork_rng)
        self.reader = zcs.ComponentReader(self.reader.net)
        self.arm = str(arm)
        #: optional row permutations for the mechanism interventions
        self.n_pairing_perm: torch.Tensor | None = None
        self.e_pairing_perm: torch.Tensor | None = None
        if arm == "A":
            for name in ("W_A_S", "W_A_C", "W_E_S", "W_E_C", "node_encoder", "edge_encoder"):
                if not hasattr(self, name):
                    raise RuntimeError(f"arm A requires the original binding module {name}")
        else:
            for name in ("W_A_S", "W_A_C", "W_E_S", "W_E_C", "node_encoder", "edge_encoder"):
                delattr(self, name)
            generator = torch.Generator().manual_seed(int(branch_seed))
            self.node_branch = NodeBranch(N_KIND[arm], generator)
            self.edge_branch = EdgeBranch(E_KIND[arm], generator)

    # -- arm slot construction -------------------------------------------------

    def _arm_a_slots(
        self, coord: torch.Tensor, data: Any, mask: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Original product binding path (frozen parent semantics, paired)."""
        n = int(coord.shape[0])
        q = F.one_hot(data.dict_atom, num_classes=ATOM_CATEGORIES).to(coord.dtype)
        occ_coord_node = None
        if bool(getattr(mask, "use_node_shuffle", False)):
            occ_coord_node = getattr(data, "env_occ_coord_node", None)
            if occ_coord_node is None:
                raise RuntimeError("node-shuffle intervention requires data.env_occ_coord_node")
        if occ_coord_node is None:
            occ_coord_node = data.env_occ_node
        occ_coord_node = occ_coord_node.to(coord.device)
        c = coord[occ_coord_node]
        qc = q[data.env_occ_node.to(coord.device)]
        u = self._node_binding(c, qc)
        if bool(getattr(mask, "node_binding_zero", False)):
            u = torch.zeros_like(u)
        flat = torch.zeros((n * N_SHELLS, int(p2.D_A)), device=u.device, dtype=u.dtype)
        flat.index_add_(
            0,
            data.env_occ_root.to(coord.device) * N_SHELLS + data.env_occ_shell.to(coord.device),
            u,
        )
        node_slots = flat.view(n, N_SHELLS, int(p2.D_A))
        bond_u = bond_v = None
        if bool(getattr(mask, "use_edge_shuffle", False)):
            bond_u = getattr(data, "env_bond_u_shuffled", None)
            bond_v = getattr(data, "env_bond_v_shuffled", None)
            if bond_u is None or bond_v is None:
                raise RuntimeError("edge-shuffle intervention requires shuffled endpoint fields")
        if bond_u is None:
            bond_u = data.env_bond_u
        if bond_v is None:
            bond_v = data.env_bond_v
        edge_slots = self._edge_env_slots(
            coord,
            data,
            bond_u.to(coord.device),
            bond_v.to(coord.device),
            edge_binding_zero=bool(getattr(mask, "edge_binding_zero", False)),
            mask=mask,
        )
        return node_slots, edge_slots

    def node_branch_slots(self, coord: torch.Tensor, data: Any) -> torch.Tensor:
        """C-arm node slots ``[n, 3, 48]``: member binding + occurrence pooling."""
        n = int(coord.shape[0])
        q = F.one_hot(data.dict_atom, num_classes=ATOM_CATEGORIES).to(coord.dtype)
        occ_node = data.env_occ_node.to(coord.device)
        z_occ = coord[occ_node]
        if self.n_pairing_perm is None:
            q_occ = q[occ_node]
        else:
            perm = self.n_pairing_perm.to(coord.device)
            q_occ = q[occ_node[perm]]
        rep = self.node_branch(z_occ, q_occ)
        flat = torch.zeros((n * N_SHELLS, NODE_SLOT_WIDTH), device=coord.device, dtype=coord.dtype)
        flat.index_add_(
            0,
            data.env_occ_root.to(coord.device) * N_SHELLS + data.env_occ_shell.to(coord.device),
            rep,
        )
        return flat.view(n, N_SHELLS, NODE_SLOT_WIDTH)

    def edge_branch_slots(self, coord: torch.Tensor, data: Any) -> torch.Tensor:
        """C-arm edge slots ``[n, 6, 32]``: endpoint binding + bond pooling."""
        n = int(coord.shape[0])
        q = F.one_hot(data.dict_atom, num_classes=ATOM_CATEGORIES).to(coord.dtype)
        bu = data.env_bond_u.to(coord.device)
        bv = data.env_bond_v.to(coord.device)
        z_v = coord[bu]
        z_w = coord[bv]
        if self.e_pairing_perm is None:
            q_v = q[bu]
            q_w = q[bv]
        else:
            sig = self.e_pairing_perm.to(coord.device)
            q_v = q[bu[sig]]
            q_w = q[bv[sig]]
        b_e = F.one_hot(data.env_bond_type.to(coord.device), num_classes=BOND_CATEGORIES).to(coord.dtype)
        rep = self.edge_branch(z_v, z_w, q_v, q_w, b_e)
        flat = torch.zeros((n * SHELLPAIR_CLASSES, EDGE_SLOT_WIDTH), device=coord.device, dtype=coord.dtype)
        flat.index_add_(
            0,
            data.env_bond_root.to(coord.device) * SHELLPAIR_CLASSES
            + data.env_bond_shellpair.to(coord.device),
            rep,
        )
        return flat.view(n, SHELLPAIR_CLASSES, EDGE_SLOT_WIDTH)

    # -- the single overridden environment path ---------------------------------

    def environments(self, coord: torch.Tensor, data: Any) -> torch.Tensor:
        """Unmasked path routed through the same single override (identity mask)."""
        return self.environments_masked(coord, data, sem.SEMMask())

    def environments_masked(
        self,
        coord: torch.Tensor,
        data: Any,
        mask: Any = None,
        fill: Mapping[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        mask = sem.SEMMask() if mask is None else mask
        if bool(getattr(mask, "residual_zero", False)):
            coord = coord.clone()
            coord[:, self.common_dim :] = 0.0
        n = int(coord.shape[0])
        interface = sem.SEM108Model.semantic_interface(self, coord, data, mask, fill)
        if self.arm == "A":
            node_slots, edge_slots = self._arm_a_slots(coord, data, mask)
            node_out = self.node_encoder(node_slots)
            edge_out = self.edge_encoder(edge_slots)
        else:
            node_out = self.node_branch_slots(coord, data)
            edge_out = self.edge_branch_slots(coord, data)
        fused = torch.cat(
            [interface, node_out.reshape(n, -1), edge_out.reshape(n, -1)], dim=1
        )
        preact = self.fusion[0](fused)
        if not bool(self.local_tuple.ablate):
            preact = preact + F.linear(self.local_tuple(data), self.local_tuple.W_loc)
        hidden = self.fusion[1](preact)
        return self.local_dictionary_bridge(self.fusion[2](hidden))

    # -- intervention hooks ------------------------------------------------------

    def set_interventions(self, *, n_pairing_perm: torch.Tensor | None = None, e_pairing_perm: torch.Tensor | None = None) -> None:
        self.n_pairing_perm = None if n_pairing_perm is None else torch.as_tensor(n_pairing_perm, dtype=torch.long)
        self.e_pairing_perm = None if e_pairing_perm is None else torch.as_tensor(e_pairing_perm, dtype=torch.long)

    def frozen_basis_hashes(self) -> dict[str, str]:
        return {
            "U": array_sha256(self.U.detach().cpu().numpy()),
            "common_rms": array_sha256(self.common_rms.detach().cpu().numpy()),
            "D": array_sha256(self.D.detach().cpu().numpy()),
        }


#: state keys shared by every arm (identical at construction for a given seed):
#: the canonical base modules built from the same global stream.  Arm A keeps
#: the original slot modules and the C arms attach branch modules, so both
#: arm-specific groups are excluded from the shared comparison.
_SHARED_EXCLUDE_PREFIXES = (
    "node_branch.",
    "edge_branch.",
    "W_A_S",
    "W_A_C",
    "W_E_S",
    "W_E_C",
    "node_encoder.",
    "edge_encoder.",
)


def shared_state(model: CSSDBindingFull) -> dict[str, torch.Tensor]:
    return {
        key: value
        for key, value in model.state_dict().items()
        if not any(key.startswith(prefix) for prefix in _SHARED_EXCLUDE_PREFIXES)
    }


def parameter_audit(model: CSSDBindingFull) -> dict[str, Any]:
    names = [name for name, _ in model.named_parameters()]
    branch_names = [n for n in names if n.startswith(("node_branch.", "edge_branch."))]
    slot_names = [
        n for n in names
        if n.startswith(("W_A_S", "W_A_C", "W_E_S", "W_E_C", "node_encoder.", "edge_encoder."))
    ]
    return {
        "arm": str(model.arm),
        "total_parameters": int(sum(p.numel() for p in model.parameters())),
        "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "branch_parameters": int(sum(p.numel() for n, p in model.named_parameters() if n.startswith(("node_branch.", "edge_branch.")))),
        "original_slot_parameters": int(sum(p.numel() for n, p in model.named_parameters() if n.startswith(("W_A_S", "W_A_C", "W_E_S", "W_E_C", "node_encoder.", "edge_encoder.")))),
        "n_branch_modules": len(branch_names),
        "n_original_slot_modules": len(slot_names),
        "frozen_buffers": sorted(
            k for k, v in model.named_buffers() if k in ("U", "common_rms", "D")
        ),
    }


def build_arm(
    arm: str,
    payload: prev.TuplePayload,
    kappa_M: float,
    basis: Mapping[str, np.ndarray],
    seed: int,
    *,
    branch_seed: int | None = None,
) -> CSSDBindingFull:
    """Fresh untrained arm (canonical Full global stream from ``seed``)."""
    if arm not in ARMS:
        raise ValueError(arm)
    if branch_seed is None:
        branch_seed = branch_seed_for(arm, seed)
    torch.manual_seed(int(seed))
    subspace = cssd.CommonSubspace(
        components=np.asarray(basis["U"], np.float32),
        rms=np.asarray(basis["common_rms"], np.float32),
        kind="q1",
    )
    model = CSSDBindingFull(
        payload,
        float(kappa_M),
        arm,
        dictionary=np.asarray(basis["D"], np.float32),
        subspace=subspace,
        branch_seed=int(branch_seed),
    )
    audit = parameter_audit(model)
    expected_total = _expected_arm_parameters(arm)
    if audit["total_parameters"] != expected_total or audit["trainable_parameters"] != expected_total:
        raise RuntimeError(
            f"parameter audit failed for arm {arm}: {audit} (expected total {expected_total})"
        )
    if arm == "A":
        if audit["original_slot_parameters"] != ORIGINAL_SLOT_PARAMETERS:
            raise RuntimeError(f"arm A slot parameter audit failed: {audit}")
    else:
        if audit["original_slot_parameters"] != 0:
            raise RuntimeError(f"C arm must not keep original slot modules: {audit}")
        expected_branch = {
            "C00": N_SEP_PARAMETERS + E_SEP_PARAMETERS,
            "C10": N_JOINT_PARAMETERS + E_SEP_PARAMETERS,
            "C01": N_SEP_PARAMETERS + E_JOINT_PARAMETERS,
            "C11": N_JOINT_PARAMETERS + E_JOINT_PARAMETERS,
        }[arm]
        if audit["branch_parameters"] != expected_branch:
            raise RuntimeError(f"branch parameter audit failed for {arm}: {audit}")
    return model


# ---------------------------------------------------------------------------
# 7. intervention permutation builders (post-training diagnostics only)
# ---------------------------------------------------------------------------


def build_group_perm(group_keys: np.ndarray, seed: int) -> np.ndarray:
    """Permutation of occurrence rows that only reorders within each group."""
    keys = np.asarray(group_keys, np.int64)
    rng = np.random.default_rng(int(seed))
    perm = np.arange(keys.shape[0], dtype=np.int64)
    for key in np.unique(keys):
        members = np.where(keys == key)[0]
        if members.size > 1:
            perm[members] = rng.permutation(members)
    return perm


def n_group_keys(data: Any) -> np.ndarray:
    return (
        data.env_occ_root.cpu().numpy().astype(np.int64) * N_SHELLS
        + data.env_occ_shell.cpu().numpy().astype(np.int64)
    )


def e_group_keys(data: Any) -> np.ndarray:
    return (
        data.env_bond_root.cpu().numpy().astype(np.int64) * SHELLPAIR_CLASSES * BOND_CATEGORIES
        + data.env_bond_shellpair.cpu().numpy().astype(np.int64) * BOND_CATEGORIES
        + data.env_bond_type.cpu().numpy().astype(np.int64)
    )


def intervention_stats(data: Any, perm: np.ndarray, kind: str) -> dict[str, Any]:
    """Actual pairing-change fraction of a permutation on one batch.

    A permuted row counts as changed only if the swapped-in member values
    really differ (identical atom types / chemistry tuples are not a change).
    """
    perm = np.asarray(perm, np.int64)
    if kind == "n":
        atom = data.dict_atom.cpu().numpy().astype(np.int64)
        occ = data.env_occ_node.cpu().numpy().astype(np.int64)
        changed = atom[occ[perm]] != atom[occ]
        rows = int(occ.shape[0])
    elif kind == "e":
        atom = data.dict_atom.cpu().numpy().astype(np.int64)
        bu = data.env_bond_u.cpu().numpy().astype(np.int64)
        bv = data.env_bond_v.cpu().numpy().astype(np.int64)
        changed = (atom[bu[perm]] != atom[bu]) | (atom[bv[perm]] != atom[bv])
        rows = int(bu.shape[0])
    else:
        raise ValueError(kind)
    return {
        "rows": rows,
        "rows_changed": int(changed.sum()),
        "changed_fraction": float(changed.sum() / max(rows, 1)),
    }


# ---------------------------------------------------------------------------
# 8. focused correctness checks (CPU, fast; only checks that can catch
#    substantive errors)
# ---------------------------------------------------------------------------


def _clone_graph(data: Any) -> Any:
    """Independent copy of one graph's env payload (never shares the store)."""
    if hasattr(data, "clone"):
        return data.clone()
    import copy

    return copy.copy(data)


def _relabel_nodes(data: Any, perm: np.ndarray) -> Any:
    """Copy of one graph with node rows permuted and every node-indexed
    structure remapped (occurrence roots and members, bond roots and
    endpoints, pair endpoints)."""
    out = _clone_graph(data)
    out.dict_phi = data.dict_phi[perm].clone()
    out.dict_atom = data.dict_atom[perm].clone()
    out.anchor = data.anchor[perm].clone()
    if getattr(data, "patch_cont", None) is not None and int(data.patch_cont.shape[0]) == int(data.dict_phi.shape[0]):
        out.patch_cont = data.patch_cont[perm].clone()
    mapping = torch.empty(int(perm.shape[0]), dtype=torch.long)
    mapping[torch.as_tensor(perm, dtype=torch.long)] = torch.arange(perm.shape[0])
    if getattr(data, "env_occ_node", None) is not None:
        out.env_occ_node = mapping[data.env_occ_node].clone()
        out.env_occ_root = mapping[data.env_occ_root].clone()
        out.env_bond_root = mapping[data.env_bond_root].clone()
        out.env_bond_u = mapping[data.env_bond_u].clone()
        out.env_bond_v = mapping[data.env_bond_v].clone()
        if getattr(data, "env_occ_coord_node", None) is not None:
            out.env_occ_coord_node = mapping[data.env_occ_coord_node].clone()
    if getattr(data, "pair_index", None) is not None:
        out.pair_index = torch.stack(
            [mapping[data.pair_index[0]], mapping[data.pair_index[1]]], dim=0
        ).clone()
    return out


def _permuted_tuple_payload(
    payload_arrays: Mapping[str, np.ndarray], mol_index: int, perm: np.ndarray
) -> dict[str, np.ndarray]:
    """Tuple-payload copy with one molecule's per-root incidence blocks reordered
    to match a node permutation (the incidence is positional by design)."""
    arrays = {k: np.asarray(v) for k, v in payload_arrays.items()}
    root_base = arrays["root_base"].astype(np.int64)
    pair_ptr = arrays["pair_ptr"].astype(np.int64)
    m = int(mol_index)
    lo, hi = int(root_base[m]), int(root_base[m + 1])
    perm = np.asarray(perm, np.int64)
    if perm.shape[0] != hi - lo or sorted(perm.tolist()) != list(range(hi - lo)):
        raise ValueError("perm must be a permutation of the molecule's node positions")
    # per-root pair-row blocks, reordered so new root position q gets the
    # block of old root position perm^{-1}(q)  (block of old root p -> new
    # position perm[p])
    blocks = [pair_ptr[lo + p : lo + p + 1] for p in range(hi - lo)]
    # new position q holds old node perm[q] (out.dict_phi = old.dict_phi[perm]),
    # so new root q gets the incidence block of old root perm[q]
    rows = (
        np.concatenate([blocks[int(p)] for p in perm.tolist()])
        if blocks else np.empty(0, np.int64)
    )
    sizes = [int(blocks[int(p)].size) for p in perm.tolist()]
    new_ptr = np.concatenate([[pair_ptr[lo]], pair_ptr[lo] + np.cumsum(sizes)]) if sizes else np.array([pair_ptr[lo], pair_ptr[lo]])
    out = dict(arrays)
    # rebuild the full pair-row list molecule-by-molecule with m replaced
    all_rows = []
    for mm in range(root_base.shape[0] - 1):
        rlo, rhi = int(root_base[mm]), int(root_base[mm + 1])
        if mm == m:
            all_rows.append(rows)
        else:
            all_rows.append(np.arange(pair_ptr[rlo], pair_ptr[rhi]))
    full_rows = np.concatenate(all_rows)
    for key in ("pair_t", "pair_a", "pair_wJ", "pair_wI"):
        if key in out:
            out[key] = arrays[key][full_rows]
    out["pair_ptr"] = pair_ptr.copy()
    out["pair_ptr"][lo : hi + 1] = new_ptr
    if "root_atom" in out:
        ra = arrays["root_atom"].copy()
        ra[lo:hi] = arrays["root_atom"][lo:hi][perm]
        out["root_atom"] = ra
    return out


def _eval_batch_predict(model: CSSDBindingFull, batch: Any, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        batch = batch.to(device)
        return model(batch, mask=cm.C6_MASK).detach().cpu().numpy().astype(np.float64)


def run_checks(*, n_graphs: int = 24, out_dir: Path = RESULTS_DIR, log: Any = print) -> dict[str, Any]:
    """Pre-training focused checks (contract items 1-5 of the round)."""
    torch.set_num_threads(8)
    started = time.perf_counter()
    device = torch.device("cpu")
    objects = load_objects(out_dir)
    basis = load_cssd_basis(out_dir)
    _load_q_soup = load_q_soup(out_dir)  # hash-checked load
    payload = prev.TuplePayload(objects["payload_arrays"])
    kappa_M = float(objects["kappa"]["kappa_M"])
    fold = objects["fold"]
    _prep, fit_data, select_data, confirm_data = build_prepared_data(objects)

    checks: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "stage": "checks",
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }

    # (1) fold / provenance --------------------------------------------------
    fchecks = objects["manifest"]["fold"]["checks"]
    checks["fold"] = {
        **fchecks,
        "fit_sha256": objects["manifest"]["fold"]["fit_sha256"],
        "select_sha256": objects["manifest"]["fold"]["select_sha256"],
        "confirm_sha256": objects["manifest"]["fold"]["confirm_sha256"],
        "target_refit_uses_dev_labels_in_fit": bool(
            objects["manifest"]["target_checks"]["uses_dev_labels_in_fit"] is False
        ),
    }
    if not (fchecks["disjoint"] and fchecks["cover"]):
        raise RuntimeError("fold disjoint/cover failed")

    # (2) build all arms; shared init identity + parameter audits ------------
    models: dict[str, CSSDBindingFull] = {}
    audits = {}
    for arm in ARMS:
        models[arm] = build_arm(arm, payload, kappa_M, basis, SEED)
        audits[arm] = parameter_audit(models[arm])
    shared_hashes = {arm: state_hash(shared_state(models[arm])) for arm in ARMS}
    checks["init_identity"] = {
        "expected_total_parameters": {arm: _expected_arm_parameters(arm) for arm in ARMS},
        "audits": audits,
        "shared_state_sha256": shared_hashes,
        "shared_identical_across_arms": bool(len(set(shared_hashes.values())) == 1),
        "c_arm_pairwise_parameter_spread": int(
            max(audits[a]["total_parameters"] for a in ("C00", "C10", "C01", "C11"))
            - min(audits[a]["total_parameters"] for a in ("C00", "C10", "C01", "C11"))
        ),
    }
    if not checks["init_identity"]["shared_identical_across_arms"]:
        raise RuntimeError("shared body init is not identical across arms")
    if checks["init_identity"]["c_arm_pairwise_parameter_spread"] > 100:
        raise RuntimeError("C-arm parameter spread exceeds the pre-registered tens-of-params budget")

    # seed-1 parameterisation is genuinely different
    m1 = build_arm("C10", payload, kappa_M, basis, 1)
    _sched0, sched_hash0 = zw.build_schedule(len(fit_data), EPOCHS, SEED + TRAIN_SHUFFLE_OFFSET)
    _sched1, sched_hash1 = zw.build_schedule(len(fit_data), EPOCHS, 1 + TRAIN_SHUFFLE_OFFSET)
    checks["seed_parameterisation"] = {
        "seed0_shared_sha256": shared_hashes["C10"],
        "seed1_shared_sha256": state_hash(shared_state(m1)),
        "seed1_branch_seed": branch_seed_for("C10", 1),
        "seed0_branch_seed": branch_seed_for("C10", 0),
        "schedule_hash_seed0": sched_hash0,
        "schedule_hash_seed1": sched_hash1,
        "init_differs": bool(state_hash(shared_state(m1)) != shared_hashes["C10"]),
        "schedule_differs": bool(sched_hash1 != sched_hash0),
        "branch_seed_differs": bool(branch_seed_for("C10", 1) != branch_seed_for("C10", 0)),
    }
    if not all(checks["seed_parameterisation"][k] for k in ("init_differs", "schedule_differs", "branch_seed_differs")):
        raise RuntimeError("seed 1 is not a genuinely different run")

    # (3) CSSD invariances on real fit batches --------------------------------
    graphs = list(fit_data[: min(n_graphs, len(fit_data))])
    batch = p1.env_collate(graphs).to(device)
    model = models["C10"]
    z0 = model.code(batch.dict_phi).detach().numpy()
    relabeled_batch = p1.env_collate(graphs_with_relabelled_chemistry(graphs, 20261012)).to(device)
    z1 = model.code(relabeled_batch.dict_phi).detach().numpy()
    # row-chunked coding == whole-batch coding (node-centric pure row function)
    z_chunked = np.concatenate(
        [model.code(batch.dict_phi[i : i + 7]).detach().numpy() for i in range(0, batch.dict_phi.shape[0], 7)],
        axis=0,
    )
    checks["cssd_code"] = {
        "z_chemistry_relabel_invariant": bool(np.array_equal(z0, z1)),
        "z_row_chunk_max_abs": float(np.max(np.abs(z0 - z_chunked))),
        "z_row_chunk_invariant": bool(np.max(np.abs(z0 - z_chunked)) <= 1e-5),
        "z_width": int(z0.shape[1]),
        "frozen_basis_hashes": models["C10"].frozen_basis_hashes(),
        "basis_hashes_at_load": {
            "U": array_sha256(basis["U"]), "common_rms": array_sha256(basis["common_rms"]), "D": array_sha256(basis["D"]),
        },
    }
    if not (checks["cssd_code"]["z_chemistry_relabel_invariant"] and checks["cssd_code"]["z_row_chunk_invariant"]):
        raise RuntimeError("CSSD code invariance failed")

    # (4) bond dedup / occurrence validity on the checked fit graphs ----------
    bond_dedup_ok = True
    n_bond_rows = 0
    for graph in fit_data[:n_graphs]:
        bu = graph.env_bond_u.cpu().numpy()
        bv = graph.env_bond_v.cpu().numpy()
        root = graph.env_bond_root.cpu().numpy()
        keys = set()
        for r, u, v in zip(root.tolist(), bu.tolist(), bv.tolist()):
            key = (int(r), min(int(u), int(v)), max(int(u), int(v)))
            if key in keys:
                bond_dedup_ok = False
            keys.add(key)
        n_bond_rows += int(bu.shape[0])
    checks["bond_dedup"] = {
        "n_graphs": int(min(n_graphs, len(fit_data))),
        "n_bond_rows": int(n_bond_rows),
        "each_undirected_bond_once_per_root": bool(bond_dedup_ok),
    }
    if not bond_dedup_ok:
        raise RuntimeError("bond dedup failed")

    # (5) branch operator reference loop + invariances ------------------------
    coord = model.code(batch.dict_phi)
    node_out = model.node_branch_slots(coord, batch)
    edge_out = model.edge_branch_slots(coord, batch)

    # independent per-occurrence python loop reference (graph by graph)
    loop_node = []
    loop_edge = []
    n_graphs_b = int(batch.env_occ_root.max().item()) + 1 if batch.env_occ_root.numel() else 0
    q_np = F.one_hot(batch.dict_atom, num_classes=ATOM_CATEGORIES).numpy()
    z_np = coord.detach().numpy()
    for g in range(n_graphs_b):
        sel_occ = (batch.env_occ_root.numpy() == g)
        occ_nodes = batch.env_occ_node.numpy()[sel_occ]
        occ_shells = batch.env_occ_shell.numpy()[sel_occ]
        rep = model.node_branch(
            torch.as_tensor(z_np[occ_nodes], dtype=torch.float32),
            torch.as_tensor(q_np[occ_nodes], dtype=torch.float32),
        ).detach().numpy()
        gnode = np.zeros((N_SHELLS, NODE_SLOT_WIDTH), dtype=np.float64)
        for row, shell in enumerate(occ_shells.tolist()):
            gnode[int(shell)] += rep[row]
        loop_node.append(gnode)
        sel_bond = (batch.env_bond_root.numpy() == g)
        bu = batch.env_bond_u.numpy()[sel_bond]
        bv = batch.env_bond_v.numpy()[sel_bond]
        btype = batch.env_bond_type.numpy()[sel_bond]
        sp = batch.env_bond_shellpair.numpy()[sel_bond]
        b_one = np.eye(BOND_CATEGORIES, dtype=np.float32)[btype]
        zv = z_np[bu]; z_w = z_np[bv]; qv = q_np[bu]; qw = q_np[bv]
        rep_e = model.edge_branch(
            torch.as_tensor(zv), torch.as_tensor(z_w), torch.as_tensor(qv), torch.as_tensor(qw), torch.as_tensor(b_one)
        ).detach().numpy()
        gedge = np.zeros((SHELLPAIR_CLASSES, EDGE_SLOT_WIDTH), dtype=np.float64)
        for row, sp_ in enumerate(sp.tolist()):
            gedge[int(sp_)] += rep_e[row]
        loop_edge.append(gedge)
    loop_node = np.stack(loop_node)
    loop_edge = np.stack(loop_edge)
    checks["branch_reference_loop"] = {
        "n_graphs": int(n_graphs_b),
        "node_slots_max_abs_diff": float(np.max(np.abs(node_out.detach().numpy() - loop_node))),
        "edge_slots_max_abs_diff": float(np.max(np.abs(edge_out.detach().numpy() - loop_edge))),
    }
    if max(checks["branch_reference_loop"]["node_slots_max_abs_diff"],
            checks["branch_reference_loop"]["edge_slots_max_abs_diff"]) > 1e-4:
        raise RuntimeError("branch reference loop mismatch")

    # endpoint swap symmetry (E) and group-preserving occurrence-row order
    swapped_batch = p1.env_collate(graphs_with_swapped_bond_endpoints(graphs)).to(device)
    edge_swapped = model.edge_branch_slots(coord, swapped_batch)
    shuffled_batch = p1.env_collate(graphs_with_permuted_occurrence_rows(graphs, 20261013)).to(device)
    node_shuf = model.node_branch_slots(coord, shuffled_batch)
    edge_shuf = model.edge_branch_slots(coord, shuffled_batch)
    checks["operator_invariances"] = {
        "e_endpoint_swap_symmetric": float(torch.max(torch.abs(edge_out - edge_swapped)).item()),
        "occurrence_row_order_node_max_abs": float(torch.max(torch.abs(node_out - node_shuf)).item()),
        "occurrence_row_order_edge_max_abs": float(torch.max(torch.abs(edge_out - edge_shuf)).item()),
    }
    if max(checks["operator_invariances"].values()) > 1e-4:
        raise RuntimeError("operator invariance failed")

    # node relabelling / batch composition invariances on full predictions
    g0 = fit_data[0]
    g1 = next(d for d in fit_data[1:n_graphs] if int(d.dict_phi.shape[0]) == int(g0.dict_phi.shape[0]))
    two = p1.env_collate([g0, g1]).to(device)
    p_two = _eval_batch_predict(model, two, device)
    p_singles = np.concatenate([_eval_batch_predict(model, p1.env_collate([g]).to(device), device) for g in (g0, g1)])
    relabel_perm = torch.randperm(int(g0.dict_phi.shape[0]), generator=torch.Generator().manual_seed(20261015))
    g0r = _relabel_nodes(g0, relabel_perm.numpy())
    # the tuple incidence is positional by historical design, so the relabelled
    # prediction swaps in the consistently block-reordered payload for that
    # one molecule (restored afterwards)
    mol0 = int(torch.as_tensor(g0.local_mol_id).reshape(-1)[0].item())
    permuted_payload = prev.TuplePayload(
        _permuted_tuple_payload(objects["payload_arrays"], mol0, relabel_perm.numpy())
    )
    original_payload = model.local_tuple._payload
    original_cache = model.local_tuple._device_cache
    model.local_tuple._payload = permuted_payload
    model.local_tuple._device_cache = {}
    try:
        p_g0r = _eval_batch_predict(model, p1.env_collate([g0r]).to(device), device)
    finally:
        model.local_tuple._payload = original_payload
        model.local_tuple._device_cache = original_cache
    reversed_two = p1.env_collate([g1, g0]).to(device)
    p_rev = _eval_batch_predict(model, reversed_two, device)
    checks["prediction_invariances"] = {
        "same_node_count_different_structure": {
            "node_count": int(g0.dict_phi.shape[0]),
            "structure_signature_differs": bool(
                not np.allclose(g0.dict_phi.numpy(), g1.dict_phi.numpy())
            ),
            "batch_equals_individuals_max_abs": float(np.max(np.abs(p_two - p_singles))),
            "reversed_batch_max_abs": float(np.max(np.abs(p_rev - p_two[::-1]))),
        },
        "node_relabel_max_abs": float(np.max(np.abs(p_g0r - p_singles[:1]))),
        "prediction_range": [float(p_two.min()), float(p_two.max())],
        "prediction_has_variation": bool(np.std(p_two) > 0),
    }
    if max(checks["prediction_invariances"]["same_node_count_different_structure"]["batch_equals_individuals_max_abs"],
            checks["prediction_invariances"]["same_node_count_different_structure"]["reversed_batch_max_abs"],
            checks["prediction_invariances"]["node_relabel_max_abs"]) > 1e-4:
        raise RuntimeError("prediction invariance failed")

    # (6) static contract: pair_relation mutation leaves pre-pair slots intact
    pair_before = model.node_branch_slots(coord, batch), model.edge_branch_slots(coord, batch)
    batch_pair_mut = p1.env_collate(graphs_with_mutated_pair_relation(graphs)).to(device)
    pair_after = model.node_branch_slots(coord, batch_pair_mut), model.edge_branch_slots(coord, batch_pair_mut)
    checks["static_contract"] = {
        "pair_relation_mutation_node_slots_max_abs": float(torch.max(torch.abs(pair_before[0] - pair_after[0])).item()),
        "pair_relation_mutation_edge_slots_max_abs": float(torch.max(torch.abs(pair_before[1] - pair_after[1])).item()),
        "branch_reads_only_pre_pair_inputs": True,
    }
    if max(checks["static_contract"]["pair_relation_mutation_node_slots_max_abs"],
            checks["static_contract"]["pair_relation_mutation_edge_slots_max_abs"]) > 0.0:
        raise RuntimeError("static contract violated: pre-pair slots depend on pair_relation")

    # (7) intervention semantics on a real batch: C00 invariant, joint changes
    n_perm = build_group_perm(n_group_keys(batch), INTERVENTION_SEED + 1)
    e_perm = build_group_perm(e_group_keys(batch), INTERVENTION_SEED + 2)
    m00, m10, m01 = models["C00"], models["C10"], models["C01"]
    base00_n = m00.node_branch_slots(coord, batch)
    base00_e = m00.edge_branch_slots(coord, batch)
    base10_n = m10.node_branch_slots(coord, batch)
    base01_e = m01.edge_branch_slots(coord, batch)
    m00.set_interventions(n_pairing_perm=n_perm)
    shuffled00_n = m00.node_branch_slots(coord, batch)
    m00.set_interventions()
    m00.set_interventions(e_pairing_perm=e_perm)
    shuffled00_e = m00.edge_branch_slots(coord, batch)
    m00.set_interventions()
    m10.set_interventions(n_pairing_perm=n_perm)
    shuffled10_n = m10.node_branch_slots(coord, batch)
    m10.set_interventions()
    m01.set_interventions(e_pairing_perm=e_perm)
    shuffled01_e = m01.edge_branch_slots(coord, batch)
    m01.set_interventions()

    def _relmax(a: torch.Tensor, b: torch.Tensor) -> float:
        denom = float(b.abs().max()) + 1e-12
        return float((a - b).abs().max() / denom)

    checks["intervention_semantics"] = {
        "n_shuffle": {
            "c00_node_slots_relmax": _relmax(shuffled00_n, base00_n),
            "c10_node_slots_relmax_vs_own_base": _relmax(shuffled10_n, base10_n),
            "pairing_changed_fraction": intervention_stats(batch, n_perm, "n")["changed_fraction"],
            "c00_invariant_tolerance": 1e-4,
        },
        "e_shuffle": {
            "c00_edge_slots_relmax": _relmax(shuffled00_e, base00_e),
            "c01_edge_slots_relmax_vs_own_base": _relmax(shuffled01_e, base01_e),
            "pairing_changed_fraction": intervention_stats(batch, e_perm, "e")["changed_fraction"],
            "c00_invariant_tolerance": 1e-4,
        },
        "c00_n_invariant": bool(_relmax(shuffled00_n, base00_n) < 1e-4),
        "c00_e_invariant": bool(_relmax(shuffled00_e, base00_e) < 1e-4),
        "c10_n_changes": bool(_relmax(shuffled10_n, base10_n) > 1e-6),
        "c01_e_changes": bool(_relmax(shuffled01_e, base01_e) > 1e-6),
    }
    if not (checks["intervention_semantics"]["c00_n_invariant"]
            and checks["intervention_semantics"]["c00_e_invariant"]
            and checks["intervention_semantics"]["c10_n_changes"]
            and checks["intervention_semantics"]["c01_e_changes"]):
        raise RuntimeError("intervention semantics failed (sep invariant / joint responsive)")

    # (8) frozen basis stays frozen under a training step
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    before = model.frozen_basis_hashes()
    model.train()
    prediction = model(batch, mask=cm.C6_MASK)
    loss = F.l1_loss(prediction.view(-1), batch.y.view(-1))
    optimizer.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    optimizer.step()
    after = model.frozen_basis_hashes()
    checks["frozen_basis_under_training"] = {
        "before": before, "after": after, "unchanged": bool(before == after),
        "d_requires_grad": bool(any(p.requires_grad for n, p in model.named_parameters() if n == "D")),
    }
    if not checks["frozen_basis_under_training"]["unchanged"]:
        raise RuntimeError("frozen CSSD basis changed during a training step")

    checks["all_passed"] = True
    checks["seconds"] = float(time.perf_counter() - started)
    checks["n_graphs"] = int(n_graphs)
    write_json(out_dir / "checks.json", checks)
    log(f"[checks] all passed in {checks['seconds']:.1f}s")
    return checks


def graphs_with_relabelled_chemistry(graphs: Sequence[Any], seed: int) -> list[Any]:
    """Per-graph copies with atom-type rows permuted (structure rows untouched)."""
    out = []
    for index, g in enumerate(graphs):
        c = _clone_graph(g)
        perm = np.random.default_rng(int(seed) + 31 * index).permutation(int(g.dict_atom.shape[0]))
        c.dict_atom = g.dict_atom[torch.as_tensor(perm, dtype=torch.long)].clone()
        out.append(c)
    return out


def graphs_with_swapped_bond_endpoints(graphs: Sequence[Any]) -> list[Any]:
    """Per-graph copies with every bond's endpoints exchanged (u <-> v)."""
    out = []
    for g in graphs:
        c = _clone_graph(g)
        c.env_bond_u = g.env_bond_v.clone()
        c.env_bond_v = g.env_bond_u.clone()
        out.append(c)
    return out


def graphs_with_permuted_occurrence_rows(graphs: Sequence[Any], seed: int) -> list[Any]:
    """Per-graph copies with occurrence rows permuted within each pooling group."""
    out = []
    for index, g in enumerate(graphs):
        c = _clone_graph(g)
        n_perm = build_group_perm(n_group_keys(g), int(seed) + 10007 * index)
        e_perm = build_group_perm(e_group_keys(g), int(seed) + 10009 * index)
        n_t = torch.as_tensor(n_perm, dtype=torch.long)
        e_t = torch.as_tensor(e_perm, dtype=torch.long)
        for name in ("env_occ_root", "env_occ_shell", "env_occ_node"):
            setattr(c, name, getattr(g, name)[n_t].clone())
        for name in ("env_bond_root", "env_bond_shellpair", "env_bond_type", "env_bond_u", "env_bond_v"):
            setattr(c, name, getattr(g, name)[e_t].clone())
        out.append(c)
    return out


def graphs_with_mutated_pair_relation(graphs: Sequence[Any]) -> list[Any]:
    """Per-graph copies with pair_relation replaced (pre-pair slots must not care)."""
    out = []
    for g in graphs:
        c = _clone_graph(g)
        c.pair_relation = torch.randn_like(g.pair_relation)
        out.append(c)
    return out


# ---------------------------------------------------------------------------
# 9. GPU smoke (a few real fit batches per arm; no silent CPU fallback)
# ---------------------------------------------------------------------------


def run_smoke(*, device_name: str = "cuda:0", n_batches: int = 4, out_dir: Path = RESULTS_DIR, log: Any = print) -> dict[str, Any]:
    device = resolve_device(device_name)
    torch.set_num_threads(4)
    started = time.perf_counter()
    objects = load_objects(out_dir)
    basis = load_cssd_basis(out_dir)
    q_soup = load_q_soup(out_dir)
    payload = prev.TuplePayload(objects["payload_arrays"])
    kappa_M = float(objects["kappa"]["kappa_M"])
    _prep, fit_data, _sel, _conf = build_prepared_data(objects)
    g = np.asarray(objects["targets"]["g"], np.float64)
    ell = np.asarray(objects["targets"]["ell"], np.float64)
    s = np.asarray(objects["targets"]["s"], np.float64)
    fit_idx = objects["fold"]["fit_idx"]
    target_g = torch.as_tensor(g[fit_idx], dtype=torch.float32, device=device)
    target_ell = torch.as_tensor(ell[fit_idx], dtype=torch.float32, device=device)
    target_s = torch.as_tensor(s[fit_idx], dtype=torch.float32, device=device)
    schedule, _schedule_hash = zw.build_schedule(len(fit_data), EPOCHS, SEED + TRAIN_SHUFFLE_OFFSET)

    smoke: dict[str, Any] = {"device": str(device), "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu", "arms": {}}
    for arm in ARMS:
        arm_started = time.perf_counter()
        seed_everything(SEED)
        model = build_arm(arm, payload, kappa_M, basis, SEED).to(device)
        # every parameter and the frozen buffers live on the GPU
        off_device = [n for n, p in model.named_parameters() if p.device.type != device.type]
        off_device += [n for n, b in model.named_buffers() if b.device.type != device.type]
        if off_device:
            raise RuntimeError(f"arm {arm} tensors not on {device}: {off_device}")
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        frozen_before = model.frozen_basis_hashes()
        losses = []
        for step in range(int(n_batches)):
            indices = [int(i) for i in schedule[0][step * BATCH_SIZE:(step + 1) * BATCH_SIZE]]
            index_t = torch.as_tensor(indices, dtype=torch.long, device=device)
            batch = zftd.make_batch(fit_data, indices, target_g, device)
            model.train()
            prediction = model(batch, mask=cm.C6_MASK)
            components = model.reader.components()
            g_identity = float(torch.max(torch.abs(components.sum(-1) - prediction.detach())).item())
            loss = F.l1_loss(prediction.view(-1), batch.y.view(-1)) + COMPONENT_LOSS_WEIGHT * (
                F.l1_loss(components[:, 0], target_ell[index_t]) + F.l1_loss(components[:, 1], target_s[index_t])
            )
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError(f"arm {arm} non-finite loss at step {step}")
            optimizer.zero_grad(); loss.backward()
            total_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP))
            if not math.isfinite(total_norm):
                raise RuntimeError(f"arm {arm} non-finite grad norm at step {step}")
            optimizer.step()
            losses.append(float(loss.detach()))
        frozen_after = model.frozen_basis_hashes()
        # new-slot variation + gradient reach (post steps, eval mode)
        model.eval()
        with torch.no_grad():
            batch = zftd.make_batch(fit_data, [int(i) for i in schedule[0][:BATCH_SIZE]], target_g, device)
            coord = model.code(batch.dict_phi)
            if arm == "A":
                node_slots, edge_slots = model._arm_a_slots(coord, batch, cm.C6_MASK)
                node_out = model.node_encoder(node_slots)
                edge_out = model.edge_encoder(edge_slots)
                slot_stats = {
                    "node_slots_rms": float(node_slots.pow(2).mean().sqrt()),
                    "edge_slots_rms": float(edge_slots.pow(2).mean().sqrt()),
                    "node_out_std": float(node_out.std()),
                    "edge_out_std": float(edge_out.std()),
                }
            else:
                node_out = model.node_branch_slots(coord, batch)
                edge_out = model.edge_branch_slots(coord, batch)
                slot_stats = {
                    "node_out_std": float(node_out.std()),
                    "edge_out_std": float(edge_out.std()),
                }
        grad_names = [n for n in ("node_branch.", "edge_branch.") if arm != "A"]
        if arm != "A":
            branch_grad_norm = float(sum(
                p.grad.detach().pow(2).sum().item() for n, p in model.named_parameters() if n.startswith(("node_branch.", "edge_branch."))
            ) ** 0.5)
            if branch_grad_norm <= 0.0:
                raise RuntimeError(f"arm {arm}: gradients never reached the N/E branch modules")
        else:
            branch_grad_norm = float(sum(
                p.grad.detach().pow(2).sum().item() for n, p in model.named_parameters() if n.startswith(("W_A_S", "W_A_C", "W_E_S", "W_E_C", "node_encoder.", "edge_encoder."))
            ) ** 0.5)
        if device.type == "cuda":
            peak_mb = float(torch.cuda.max_memory_allocated() / (1 << 20))
            torch.cuda.reset_peak_memory_stats()
        else:
            peak_mb = 0.0
        smoke["arms"][arm] = {
            "losses": losses,
            "component_identity_max_abs": g_identity,
            "slot_stats": slot_stats,
            "branch_or_slot_grad_norm": branch_grad_norm,
            "frozen_basis_unchanged": bool(frozen_before == frozen_after),
            "peak_gpu_memory_mb": peak_mb,
            "seconds": float(time.perf_counter() - arm_started),
        }
        if not smoke["arms"][arm]["frozen_basis_unchanged"]:
            raise RuntimeError(f"arm {arm}: frozen basis changed during GPU smoke")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    smoke["q_soup_sha256"] = state_hash(q_soup)
    smoke["all_finite"] = True
    smoke["seconds"] = float(time.perf_counter() - started)
    smoke["official_valid_loaded"] = False
    smoke["official_test_loaded"] = False
    write_json(out_dir / "smoke.json", smoke)
    log(f"[smoke] all arms ok on {device} in {smoke['seconds']:.1f}s")
    return smoke


# ---------------------------------------------------------------------------
# 10. body training (one arm, one seed) + fit-only calibration
# ---------------------------------------------------------------------------


def run_dir(arm: str, seed: int, out_dir: Path = RESULTS_DIR) -> Path:
    return Path(out_dir) / "runs" / f"{arm}_s{int(seed)}"


def _prefix_stats(model: nn.Module, prefixes: tuple[str, ...]) -> dict[str, float]:
    items = [(n, p) for n, p in model.named_parameters() if n.startswith(prefixes)]
    if not items:
        return {"present": 0.0}
    tensors = torch.cat([p.detach().flatten().cpu() for _, p in items])
    grads = [p.grad.detach().flatten().cpu() for _, p in items if p.grad is not None]
    out = {
        "present": float(len(items)),
        "norm": float(tensors.norm()),
        "zero_frac": float((tensors == 0).to(torch.float64).mean()),
        "nonfinite": int((~torch.isfinite(tensors)).sum()),
    }
    if grads:
        g = torch.cat(grads)
        out["grad_norm"] = float(g.norm())
        out["grad_nonfinite"] = int((~torch.isfinite(g)).sum())
    return out


def _probe(
    model: CSSDBindingFull, arm: str, epoch: int, step: int, total_norm: float, frozen_reference: dict[str, str]
) -> dict[str, Any]:
    """Small diagnostics at LOG_EPOCHS (first step of the epoch)."""
    stats: dict[str, Any] = {"arm": arm, "epoch": int(epoch), "step": int(step), "grad_norm": float(total_norm)}
    with torch.no_grad():
        if arm == "A":
            stats["original_slot"] = {
                name: _prefix_stats(model, (name,))
                for name in ("W_A_S", "W_A_C", "W_E_S", "W_E_C", "node_encoder.", "edge_encoder.")
            }
        else:
            stats["branch"] = {
                "node_branch": _prefix_stats(model, ("node_branch.",)),
                "edge_branch": _prefix_stats(model, ("edge_branch.",)),
            }
        stats["fusion"] = _prefix_stats(model, ("fusion.",))
        stats["bridge"] = _prefix_stats(model, ("local_dictionary_bridge.",))
        stats["local_tuple"] = _prefix_stats(model, ("local_tuple.",))
        stats["local_tuple_last_stats"] = dict(getattr(model.local_tuple, "last_stats", {}))
        nonfinite = [
            n for n, p in model.named_parameters()
            if not bool(torch.isfinite(p.detach()).all())
        ]
        stats["nonfinite_parameters"] = nonfinite
        basis = model.frozen_basis_hashes()
        stats["frozen_basis"] = basis
        stats["frozen_basis_matches"] = bool(basis == frozen_reference)
    if not stats["frozen_basis_matches"]:
        raise RuntimeError("frozen CSSD basis changed during training (implementation error)")
    if nonfinite:
        raise RuntimeError(f"non-finite parameters during training: {nonfinite}")
    return stats


@torch.no_grad()
def _soup_predictions(
    model: CSSDBindingFull,
    soup_state: Mapping[str, torch.Tensor],
    data_list: Sequence[Any],
    device: torch.device,
) -> dict[str, np.ndarray]:
    """h (total g), ell_hat, s_hat per molecule for a loaded soup state."""
    model.load_state_dict({k: v for k, v in soup_state.items()})
    model.eval()
    hs, ells, ss = [], [], []
    for start in range(0, len(data_list), BATCH_SIZE):
        batch = p1.env_collate(data_list[start:start + BATCH_SIZE]).to(device)
        prediction = model(batch, mask=cm.C6_MASK)
        components = model.reader.components()
        hs.append(prediction.detach().cpu().numpy().astype(np.float64))
        ells.append(components[:, 0].detach().cpu().numpy().astype(np.float64))
        ss.append(components[:, 1].detach().cpu().numpy().astype(np.float64))
    return {
        "h": np.concatenate(hs),
        "ell_hat": np.concatenate(ells),
        "s_hat": np.concatenate(ss),
    }


def _q_predictions(q_soup: Mapping[str, torch.Tensor], T: np.ndarray, device: torch.device) -> np.ndarray:
    head = build_q_head(SEED, 0.0)
    head.load_state_dict({k: v for k, v in q_soup.items()})
    head = head.to(device)
    head.eval()
    with torch.no_grad():
        return q_forward(head, torch.as_tensor(T, dtype=torch.float32, device=device)).cpu().numpy().astype(np.float64)


def train_arm(
    arm: str,
    *,
    seed: int = SEED,
    device_name: str = "cuda:0",
    out_dir: Path = RESULTS_DIR,
    epochs: int = EPOCHS,
    max_steps: int | None = None,
    log: Any = print,
) -> dict[str, Any]:
    """One full body run: fixed recipe, last-5 soup, fit-only calibration."""
    if arm not in ARMS:
        raise ValueError(arm)
    device = resolve_device(device_name)
    torch.set_num_threads(8 if device.type == "cpu" else 4)
    rdir = run_dir(arm, seed, out_dir)
    rdir.mkdir(parents=True, exist_ok=True)
    objects = load_objects(out_dir)
    basis = load_cssd_basis(out_dir)
    q_soup = load_q_soup(out_dir)
    payload = prev.TuplePayload(objects["payload_arrays"])
    kappa_M = float(objects["kappa"]["kappa_M"])
    fold = objects["fold"]
    fit_idx = np.asarray(fold["fit_idx"], np.int64)
    _prep, fit_data, _select, _confirm = build_prepared_data(objects)
    g = np.asarray(objects["targets"]["g"], np.float64)
    ell = np.asarray(objects["targets"]["ell"], np.float64)
    s = np.asarray(objects["targets"]["s"], np.float64)
    y_all = np.asarray(objects["targets"]["y"], np.float64)
    gid_all = np.asarray(objects["targets"]["gid"], np.int64)
    g_fit, ell_fit, s_fit, y_fit = g[fit_idx], ell[fit_idx], s[fit_idx], y_all[fit_idx]
    if float(np.max(np.abs(g_fit - (ell_fit + s_fit)))) > 1e-12:
        raise RuntimeError("g != ell + s on fit rows")

    target_g = torch.as_tensor(g_fit, dtype=torch.float32)
    target_ell = torch.as_tensor(ell_fit, dtype=torch.float32, device=device)
    target_s = torch.as_tensor(s_fit, dtype=torch.float32, device=device)
    schedule, schedule_hash = zw.build_schedule(len(fit_data), int(epochs), int(seed) + TRAIN_SHUFFLE_OFFSET)

    seed_everything(int(seed))
    model = build_arm(arm, payload, kappa_M, basis, int(seed))
    init_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model = model.to(device)
    frozen_reference = model.frozen_basis_hashes()
    expected_basis = {
        "U": array_sha256(basis["U"]),
        "common_rms": array_sha256(basis["common_rms"]),
        "D": array_sha256(basis["D"]),
    }
    if frozen_reference != expected_basis:
        raise RuntimeError("model basis != loaded frozen basis")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    soup_epochs = set(int(e) for e in SOUP_EPOCHS)
    soup: dict[int, dict[str, torch.Tensor]] = {}
    checkpoints: dict[int, dict[str, torch.Tensor]] = {}
    curve: list[dict[str, Any]] = []
    probes: list[dict[str, Any]] = []
    steps_done = 0
    stopped_reason = "completed"
    started = time.perf_counter()
    peak_mb = 0.0

    for epoch in range(1, int(epochs) + 1):
        epoch_started = time.perf_counter()
        model.train()
        g_sum = ell_sum = s_sum = total_sum = 0.0
        n_mol = n_steps = 0
        gnorm_sum = 0.0
        clip_hits = 0
        for start in range(0, len(schedule[epoch - 1]), BATCH_SIZE):
            indices = [int(i) for i in schedule[epoch - 1][start:start + BATCH_SIZE]]
            index_t = torch.as_tensor(indices, dtype=torch.long, device=device)
            batch = zftd.make_batch(fit_data, indices, target_g, device)
            prediction = model(batch, mask=cm.C6_MASK)
            if tuple(prediction.shape) != (len(indices),):
                raise RuntimeError("standard forward must return (batch,) total g")
            components = model.reader.components()
            if tuple(components.shape) != (len(indices), 2):
                raise RuntimeError("reader component output is not (batch, 2)")
            identity_gap = float(torch.max(torch.abs(components.sum(-1) - prediction.detach())).item())
            if identity_gap != 0.0:
                raise RuntimeError(f"g_hat != ell_hat + s_hat exactly (gap {identity_gap})")
            l_g = F.l1_loss(prediction.view(-1), batch.y.view(-1))
            l_ell = F.l1_loss(components[:, 0], target_ell[index_t])
            l_s = F.l1_loss(components[:, 1], target_s[index_t])
            loss = l_g + COMPONENT_LOSS_WEIGHT * (l_ell + l_s)
            optimizer.zero_grad()
            loss.backward()
            total_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP))
            if epoch in LOG_EPOCHS and n_steps == 0:
                probes.append(_probe(model, arm, epoch, steps_done + 1, total_norm, frozen_reference))
                probes[-1]["pred_shift"] = {
                    "l_g": float(l_g.detach()), "l_ell": float(l_ell.detach()), "l_s": float(l_s.detach()),
                }
            optimizer.step()
            g_sum += float((prediction.detach().view(-1) - batch.y.view(-1)).abs().sum())
            ell_sum += float((components[:, 0].detach() - target_ell[index_t]).abs().sum())
            s_sum += float((components[:, 1].detach() - target_s[index_t]).abs().sum())
            total_sum += float(loss.detach()) * int(len(indices))
            n_mol += int(batch.y.numel())
            n_steps += 1
            steps_done += 1
            gnorm_sum += total_norm
            clip_hits += int(total_norm > GRAD_CLIP)
            if max_steps is not None and steps_done >= int(max_steps):
                stopped_reason = "max_steps"
                break
        if device.type == "cuda":
            peak_mb = max(peak_mb, float(torch.cuda.max_memory_allocated() / (1 << 20)))
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
            checkpoints[int(epoch)] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch in soup_epochs:
            soup[int(epoch)] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if log and (epoch in LOG_EPOCHS or epoch % 40 == 0):
            c = curve[-1]
            log(f"[train {arm} s{seed}] ep={epoch:03d} Lg={c['train_L_g']:.5f} "
                f"Lell={c['train_L_ell']:.5f} Ls={c['train_L_s']:.5f} "
                f"gn={c['grad_norm']:.3g} {c['seconds']:.1f}s", flush=True)
        if stopped_reason == "max_steps":
            break

    members = sorted(soup)
    if sorted(soup) != [int(e) for e in SOUP_EPOCHS]:
        raise RuntimeError(f"soup epochs missing: {members}")
    soup_state = {key: torch.stack([soup[e][key].float() for e in members]).mean(0) for key in soup[members[0]]}
    soup_hash = state_hash(soup_state)
    init_hash = state_hash(init_state)

    # fit predictions from the soup (fit-only; select/confirm untouched)
    preds = _soup_predictions(model, soup_state, fit_data, device)
    h = preds["h"]
    T_fit = topology_matrix(fit_data)
    q_raw = _q_predictions(q_soup, T_fit, device)
    y_raw = preds["ell_hat"] + preds["s_hat"] + q_raw
    # the in-forward component sum rounds at float32, so the float64
    # per-component re-sum agrees only to float32 epsilon (~1e-7)
    if float(np.max(np.abs(y_raw - (h + q_raw)))) > 1e-5:
        raise RuntimeError("y_raw != h + Q_raw")
    b_g = float(np.median(g_fit - h))
    b_y = float(np.median(y_fit - y_raw))
    g_cal_mae = float(np.mean(np.abs(g_fit - (h + b_g))))

    torch.save(init_state, rdir / "init_state.pt")
    torch.save(soup_state, rdir / "soup_state.pt")
    for epoch, state in checkpoints.items():
        torch.save(state, rdir / f"epoch{epoch}_state.pt")
    np.savez_compressed(
        rdir / "fit_predictions.npz",
        gid=gid_all[fit_idx], y=y_fit, g=g_fit, ell=ell_fit, s=s_fit,
        h=h, ell_hat=preds["ell_hat"], s_hat=preds["s_hat"], q_raw=q_raw, y_raw=y_raw,
    )
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "stage": "train",
        "arm": arm,
        "seed": int(seed),
        "run_dir": str(rdir.relative_to(Path(out_dir))),
        "recipe": {
            "epochs": int(epochs), "batch_size": BATCH_SIZE, "lr": LR, "weight_decay": WEIGHT_DECAY,
            "grad_clip": GRAD_CLIP, "soup_epochs": list(SOUP_EPOCHS), "log_epochs": list(LOG_EPOCHS),
            "component_loss_weight": COMPONENT_LOSS_WEIGHT,
            "train_shuffle_offset": TRAIN_SHUFFLE_OFFSET,
        },
        "steps_done": int(steps_done),
        "stopped_reason": stopped_reason,
        "schedule_sha256": schedule_hash,
        "init_state_sha256": init_hash,
        "soup_state_sha256": soup_hash,
        "frozen_basis_hashes": frozen_reference,
        "q_soup_sha256": state_hash(q_soup),
        "parameter_audit": parameter_audit(model),
        "calibration": {"b_g": b_g, "b_y": b_y, "note": "single fit-median bias; never stacked"},
        "fit_diagnostics": {
            "fit_L_g_raw_mae": float(np.mean(np.abs(g_fit - h))),
            "fit_L_g_cal_mae": g_cal_mae,
            "fit_y_raw_mae": float(np.mean(np.abs(y_fit - y_raw))),
            "fit_y_cal_mae": float(np.mean(np.abs(y_fit - (y_raw + b_y)))),
            "fit_q_mae": float(np.mean(np.abs(objects["targets"]["c"][fit_idx] - q_raw))),
        },
        "curve_seconds_total": float(sum(c["seconds"] for c in curve)),
        "peak_gpu_memory_mb": float(peak_mb),
        "flops_note": "not measured; per-arm op counts differ by construction (arm A product binding vs C-arm MLPs)",
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(rdir / "curve.json", curve)
    write_json(rdir / "probes.json", probes)
    write_json(rdir / "manifest.json", manifest)
    log(f"[train {arm} s{seed}] done: {steps_done} steps, {manifest['curve_seconds_total']:.0f}s "
        f"train, fit y_raw MAE {manifest['fit_diagnostics']['fit_y_raw_mae']:.5f} "
        f"b_y={b_y:.5f} soup sha={soup_hash[:12]}…")
    return manifest


def load_run(arm: str, seed: int, out_dir: Path = RESULTS_DIR) -> dict[str, Any]:
    rdir = run_dir(arm, seed, out_dir)
    manifest = read_json(rdir / "manifest.json")
    state = torch.load(rdir / "soup_state.pt", map_location="cpu", weights_only=False)
    if state_hash(state) != manifest["soup_state_sha256"]:
        raise RuntimeError(f"run {arm} s{seed} soup state hash mismatch")
    return {"manifest": manifest, "soup_state": state, "run_dir": rdir}


# ---------------------------------------------------------------------------
# 11. endpoint evaluation (select / confirm) + contrasts + bootstrap
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_split(
    arm: str,
    seed: int,
    split: str,
    *,
    out_dir: Path = RESULTS_DIR,
    device_name: str = "cuda:0",
    interventions: Mapping[str, bool] | None = None,
    log: Any = print,
) -> dict[str, Any]:
    """Full y endpoint on one split for one trained run; paired predictions.

    ``interventions`` optionally enables the N/E pairing shuffles
    (post-training diagnostics only; seeds fixed per batch index).
    """
    if split not in ("select", "confirm"):
        raise ValueError(split)
    device = resolve_device(device_name)
    torch.set_num_threads(8 if device.type == "cpu" else 4)
    objects = load_objects(out_dir)
    basis = load_cssd_basis(out_dir)
    q_soup = load_q_soup(out_dir)
    payload = prev.TuplePayload(objects["payload_arrays"])
    kappa_M = float(objects["kappa"]["kappa_M"])
    fold = objects["fold"]
    _prep, _fit_data, select_data, confirm_data = build_prepared_data(objects)
    data_list = select_data if split == "select" else confirm_data
    split_idx = np.asarray(fold[f"{split}_idx"], np.int64)
    run = load_run(arm, seed, out_dir)
    soup_state = run["soup_state"]

    model = build_arm(arm, payload, kappa_M, basis, int(seed))
    model.load_state_dict({k: v for k, v in soup_state.items()})
    model = model.to(device)
    model.eval()

    arm_index = ARMS.index(arm)
    rows = len(data_list)
    n_changed: list[float] = []
    e_changed: list[float] = []
    hs, ells, ss = [], [], []
    for bstart, start in enumerate(range(0, rows, BATCH_SIZE)):
        batch_data = data_list[start:start + BATCH_SIZE]
        batch = p1.env_collate(batch_data).to(device)
        if interventions and interventions.get("n", False) and arm != "A":
            perm = build_group_perm(n_group_keys(batch), INTERVENTION_SEED + 1000 * arm_index + bstart)
            model.set_interventions(n_pairing_perm=torch.as_tensor(perm, dtype=torch.long))
            n_changed.append(intervention_stats(batch, perm, "n")["changed_fraction"])
        if interventions and interventions.get("e", False) and arm != "A":
            perm = build_group_perm(e_group_keys(batch), INTERVENTION_SEED + 2000 * arm_index + bstart)
            model.set_interventions(e_pairing_perm=torch.as_tensor(perm, dtype=torch.long))
            e_changed.append(intervention_stats(batch, perm, "e")["changed_fraction"])
        prediction = model(batch, mask=cm.C6_MASK)
        components = model.reader.components()
        hs.append(prediction.detach().cpu().numpy().astype(np.float64))
        ells.append(components[:, 0].detach().cpu().numpy().astype(np.float64))
        ss.append(components[:, 1].detach().cpu().numpy().astype(np.float64))
        model.set_interventions()
    h = np.concatenate(hs)
    ell_hat = np.concatenate(ells)
    s_hat = np.concatenate(ss)
    T = topology_matrix(data_list)
    q_raw = _q_predictions(q_soup, T, device)
    y_raw = ell_hat + s_hat + q_raw
    manifest = run["manifest"]
    b_y = float(manifest["calibration"]["b_y"])
    y_cal = y_raw + b_y

    targets = objects["targets"]
    y = np.asarray(targets["y"], np.float64)[split_idx]
    g = np.asarray(targets["g"], np.float64)[split_idx]
    ell = np.asarray(targets["ell"], np.float64)[split_idx]
    s = np.asarray(targets["s"], np.float64)[split_idx]
    gid = np.asarray(targets["gid"], np.int64)[split_idx]
    if float(np.max(np.abs(y_raw - (h + q_raw)))) > 1e-5:
        raise RuntimeError("y_raw != h + Q_raw on eval")
    result = {
        "arm": arm, "seed": int(seed), "split": split,
        "n": int(rows),
        "gid": gid, "y": y, "y_raw": y_raw, "y_cal": y_cal,
        "h": h, "ell_hat": ell_hat, "s_hat": s_hat, "q_raw": q_raw,
        "mae": {
            "y_raw": float(np.mean(np.abs(y - y_raw))),
            "y_cal": float(np.mean(np.abs(y - y_cal))),
            "g_raw": float(np.mean(np.abs(g - h))),
            "ell": float(np.mean(np.abs(ell - ell_hat))),
            "s": float(np.mean(np.abs(s - s_hat))),
        },
        "b_y": b_y,
        "interventions": dict(interventions) if interventions else None,
        "n_pairing_changed_fraction_mean": float(np.mean(n_changed)) if n_changed else None,
        "e_pairing_changed_fraction_mean": float(np.mean(e_changed)) if e_changed else None,
        "soup_state_sha256": manifest["soup_state_sha256"],
    }
    tag = "-".join(sorted((interventions or {}).keys())) or "none"
    np.savez_compressed(
        run["run_dir"] / f"{split}_predictions{'_' + tag if tag != 'none' else ''}.npz",
        gid=gid, y=y, y_raw=y_raw, y_cal=y_cal, h=h, q_raw=q_raw,
    )
    if log:
        log(f"[{split}-eval {arm} s{seed}{'' if not interventions else ' ' + tag}] "
            f"y_raw MAE {result['mae']['y_raw']:.5f} y_cal {result['mae']['y_cal']:.5f}")
    return result


def _contrast_stats(errors: Mapping[str, np.ndarray], n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict[str, Any]:
    """Paired contrasts + descriptive bootstrap CIs from per-row |y - y_hat|."""
    e00, e10, e01, e11 = (errors[k] for k in ("C00", "C10", "C01", "C11"))
    g_n = 0.5 * ((e00 - e10) + (e01 - e11))
    g_e = 0.5 * ((e00 - e01) + (e10 - e11))
    inter = (e10 - e00) + (e01 - e00) - (e11 - e00)
    rng = np.random.default_rng(int(seed))
    n = int(e00.shape[0])
    m00, m10, m01, m11 = (float(np.mean(v)) for v in (e00, e10, e01, e11))
    contrasts = {
        "M00": m00, "M10": m10, "M01": m01, "M11": m11,
        "G_N": float(np.mean(g_n)),
        "G_E": float(np.mean(g_e)),
        "I": float(np.mean(inter)),
    }
    if "A" in errors:
        contrasts["A_vs_C00"] = float(np.mean(errors["A"] - e00))
    cis: dict[str, Any] = {}
    for name, per_row in (("G_N", g_n), ("G_E", g_e), ("I", inter)):
        if "A" in errors and name == "I":
            pass
        boots = np.empty(int(n_boot), dtype=np.float64)
        for b in range(int(n_boot)):
            idx = rng.integers(0, n, size=n)
            boots[b] = float(np.mean(per_row[idx]))
        cis[name] = {
            "bootstrap_mean": float(np.mean(boots)),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        }
    if "A" in errors:
        per_row = errors["A"] - e00
        boots = np.empty(int(n_boot), dtype=np.float64)
        for b in range(int(n_boot)):
            idx = rng.integers(0, n, size=n)
            boots[b] = float(np.mean(per_row[idx]))
        cis["A_vs_C00"] = {
            "bootstrap_mean": float(np.mean(boots)),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        }
    return {
        "contrasts": contrasts,
        "bootstrap_ci95": cis,
        "n_boot": int(n_boot),
        "seed": int(seed),
        "note": "bootstrap describes molecule-resampling uncertainty only; it does not replace seed repetition",
    }


def _training_health(run: Mapping[str, Any]) -> dict[str, Any]:
    probes = read_json(Path(run["run_dir"]) / "probes.json")
    manifest = run["manifest"]
    arm = manifest["arm"]
    last = probes[-1]
    if arm == "A":
        slot = last["original_slot"]
        alive = all(
            slot[name]["zero_frac"] < 1.0 for name in ("W_A_S", "W_A_C", "W_E_S", "W_E_C")
        )
        grad_reached = any(slot[name].get("grad_norm", 0.0) > 0.0 for name in slot)
        slot_std = None
    else:
        branch = last["branch"]
        alive = branch["node_branch"]["zero_frac"] < 1.0 and branch["edge_branch"]["zero_frac"] < 1.0
        grad_reached = (
            branch["node_branch"].get("grad_norm", 0.0) > 0.0
            and branch["edge_branch"].get("grad_norm", 0.0) > 0.0
        )
        slot_std = None
    curve = read_json(Path(run["run_dir"]) / "curve.json")
    return {
        "arm": arm,
        "seed": manifest["seed"],
        "joint_or_slot_alive": bool(alive),
        "gradients_reached": bool(grad_reached),
        "final_train_L_g": float(curve[-1]["train_L_g"]),
        "nonfinite_parameters": bool(last["nonfinite_parameters"]),
        "healthy": bool(alive and grad_reached and not last["nonfinite_parameters"]),
    }


def frozen_decision_rules(
    health: Mapping[str, Mapping[str, Any]],
    mae_raw: Mapping[str, float],
) -> dict[str, Any]:
    """The pre-registered Stage-A decision skeleton (no arbitrary thresholds).

    Records the rule inputs and which branch of the frozen budget rules
    matches; the final scientific call is written in the round report.
    """
    joint_arms = ("C10", "C01", "C11")
    active = {a: bool(health[a]["healthy"]) for a in ARMS}
    beat_c00 = {a: bool(mae_raw[a] < mae_raw["C00"]) for a in joint_arms}
    c00_improves = bool(mae_raw["C00"] < mae_raw["A"])
    candidates = [a for a in joint_arms if active[a] and beat_c00[a]]
    c11_interaction = bool(
        active["C11"] and beat_c00["C11"] and mae_raw["C11"] < min(mae_raw["C10"], mae_raw["C01"])
    )
    if not any(active.values()):
        recommendation, seed1_arms = "BLOCKED: no arm trained healthily; fix before any seed-1 spend", []
    elif not any(beat_c00[a] and active[a] for a in joint_arms) and not c00_improves:
        recommendation, seed1_arms = (
            "STOP: all joint arms active with no raw-y gain over C00 and C00 does not improve on A; "
            "close the round as a negative result, no seed-1 rescue",
            [],
        )
    elif not any(beat_c00[a] and active[a] for a in joint_arms) and c00_improves:
        recommendation, seed1_arms = (
            "CAPACITY/INTERFACE ONLY: C00 improves on A but no joint arm beats C00; "
            "the signal is attributable to the shared local interface/capacity, not correspondence; "
            "re-check A + C00 only (seed 1, 2 runs)",
            ["A", "C00"],
        )
    elif len(candidates) == 1 and not c11_interaction:
        recommendation, seed1_arms = (
            f"ONE CANDIDATE: {candidates[0]} beats C00 on raw y; seed 1 runs A + C00 + {candidates[0]} (3 runs)",
            ["A", "C00", candidates[0]],
        )
    else:
        recommendation, seed1_arms = (
            "TWO CANDIDATES OR C11 INTERACTION: seed 1 re-runs the same five arms (5 runs)",
            list(ARMS),
        )
    total_seed0_runs = 5
    budget = {
        "max_total_body_runs": 10,
        "seed0_runs": total_seed0_runs,
        "planned_seed1_runs": len(seed1_arms),
        "within_budget": bool(total_seed0_runs + len(seed1_arms) <= 10),
    }
    return {
        "rule_inputs": {
            "arm_active": active,
            "joint_beats_c00_raw": beat_c00,
            "c00_improves_on_a_raw": c00_improves,
            "c11_beats_both_single_joint_raw": c11_interaction,
            "mae_raw": dict(mae_raw),
        },
        "recommendation": recommendation,
        "seed1_arms": seed1_arms,
        "budget": budget,
    }


def select_eval(
    *,
    seeds: Sequence[int] = (0,),
    device_name: str = "cuda:0",
    out_dir: Path = RESULTS_DIR,
    log: Any = print,
) -> dict[str, Any]:
    """Stage A: per-arm full-y raw/cal MAE on select + contrasts + rules."""
    device_name = str(device_name)
    results: dict[str, Any] = {}
    errors_raw: dict[str, dict[int, np.ndarray]] = {}
    errors_cal: dict[str, dict[int, np.ndarray]] = {}
    health: dict[str, Any] = {}
    for seed in seeds:
        for arm in ARMS:
            row = evaluate_split(arm, int(seed), "select", out_dir=out_dir, device_name=device_name, log=log)
            results[f"{arm}_s{seed}"] = row
            errors_raw.setdefault(arm, {})[int(seed)] = np.abs(row["y"] - row["y_raw"])
            errors_cal.setdefault(arm, {})[int(seed)] = np.abs(row["y"] - row["y_cal"])
            if int(seed) == 0:
                health[arm] = _training_health(load_run(arm, 0, out_dir))
    seed0_raw = {arm: float(np.mean(errors_raw[arm][0])) for arm in ARMS}
    seed0_cal = {arm: float(np.mean(errors_cal[arm][0])) for arm in ARMS}
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "stage": "select-eval",
        "primary_ranking": "full y_raw MAE on select (calibrated reported alongside)",
        "seeds": [int(s) for s in seeds],
        "per_arm": {k: {"mae": v["mae"], "b_y": v["b_y"], "soup_state_sha256": v["soup_state_sha256"]} for k, v in results.items()},
        "seed0_mae_raw": seed0_raw,
        "seed0_mae_cal": seed0_cal,
        "training_health": health,
        "contrasts_raw": _contrast_stats({a: errors_raw[a][0] for a in ARMS}),
        "contrasts_cal": _contrast_stats({a: errors_cal[a][0] for a in ARMS}),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    if int(0) in seeds:
        payload["frozen_decision"] = frozen_decision_rules(health, seed0_raw)
    write_json(out_dir / "select_eval.json", payload)
    log("[select-eval] raw MAE: " + " ".join(f"{a}={seed0_raw[a]:.5f}" for a in ARMS))
    if "frozen_decision" in payload:
        log(f"[select-eval] frozen rule recommendation: {payload['frozen_decision']['recommendation']}")
    return payload


def write_confirm_roster(
    arms: Sequence[str],
    seeds: Sequence[int],
    *,
    out_dir: Path = RESULTS_DIR,
) -> dict[str, Any]:
    if any(a not in ARMS for a in arms):
        raise ValueError(arms)
    roster = {
        "protocol_version": PROTOCOL_VERSION,
        "arms": [str(a) for a in arms],
        "seeds": [int(s) for s in seeds],
        "frozen_after": "select-eval decision; no further model change allowed",
    }
    write_json(out_dir / "confirm_roster.json", roster)
    return roster


def confirm_eval(
    *,
    device_name: str = "cuda:0",
    out_dir: Path = RESULTS_DIR,
    log: Any = print,
) -> dict[str, Any]:
    """Stage B: one-shot confirm evaluation of the frozen roster."""
    roster = read_json(out_dir / "confirm_roster.json")
    arms = [str(a) for a in roster["arms"]]
    seeds = [int(s) for s in roster["seeds"]]
    results: dict[str, Any] = {}
    errors_raw: dict[str, np.ndarray] = {}
    errors_cal: dict[str, np.ndarray] = {}
    for seed in seeds:
        for arm in arms:
            row = evaluate_split(arm, seed, "confirm", out_dir=out_dir, device_name=device_name, log=log)
            results[f"{arm}_s{seed}"] = {"mae": row["mae"], "b_y": row["b_y"]}
            errors_raw[f"{arm}_s{seed}"] = np.abs(row["y"] - row["y_raw"])
            errors_cal[f"{arm}_s{seed}"] = np.abs(row["y"] - row["y_cal"])
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "stage": "confirm-eval",
        "roster": roster,
        "per_arm": results,
        "pairwise_raw": {
            f"{a}_vs_{b}": float(np.mean(errors_raw[f"{a}_s{seed}"] - errors_raw[f"{b}_s{seed}"]))
            for seed in seeds for a in arms for b in arms if a < b
        },
        "seed_repeatability_raw": {
            arm: {
                "per_seed_mae": {int(s): float(np.mean(errors_raw[f"{arm}_s{s}"])) for s in seeds},
                "note": "independent seeds are the repetition evidence; 2 seeds are not a stability theorem",
            }
            for arm in arms
        },
        "one_shot": True,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    if all(a in errors_raw for a in ("C00", "C10", "C01", "C11")) and 0 in seeds:
        payload["contrasts_raw_seed0"] = _contrast_stats(
            {a: errors_raw[f"{a}_s0"] for a in ("A", "C00", "C10", "C01", "C11")}
        )
    write_json(out_dir / "confirm_eval.json", payload)
    log("[confirm-eval] done: " + " ".join(
        f"{k}={v['mae']['y_raw']:.5f}" for k, v in results.items()))
    return payload


def run_interventions(
    *,
    device_name: str = "cuda:0",
    out_dir: Path = RESULTS_DIR,
    seeds: Sequence[int] = (0,),
    log: Any = print,
) -> dict[str, Any]:
    """Post-training branch-mechanism interventions on the trained soups."""
    results: dict[str, Any] = {}
    for seed in seeds:
        for arm in ARMS:
            base = evaluate_split(arm, int(seed), "select", out_dir=out_dir, device_name=device_name, log=None)
            row = {"base_y_raw_mae": base["mae"]["y_raw"]}
            for kind in ("n", "e"):
                if arm == "A":
                    continue  # arm A keeps the original product binding; no C-arm branch to shuffle
                shuffled = evaluate_split(
                    arm, int(seed), "select", out_dir=out_dir, device_name=device_name,
                    interventions={kind: True}, log=log,
                )
                row[f"{kind}_shuffle"] = {
                    "y_raw_mae": shuffled["mae"]["y_raw"],
                    "delta": float(shuffled["mae"]["y_raw"] - base["mae"]["y_raw"]),
                    "pairing_changed_fraction_mean": shuffled[f"{kind}_pairing_changed_fraction_mean"],
                }
            results[f"{arm}_s{seed}"] = row
    c00 = results.get("C00_s0")
    invariance = None
    if c00 is not None:
        invariance = {
            "c00_n_shuffle_abs_delta": abs(c00["n_shuffle"]["delta"]) if "n_shuffle" in c00 else None,
            "c00_e_shuffle_abs_delta": abs(c00["e_shuffle"]["delta"]) if "e_shuffle" in c00 else None,
            "tolerance": 1e-4,
        }
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "stage": "interventions",
        "per_arm": results,
        "c00_invariance": invariance,
        "note": "shuffle deltas show dependence, not incremental benefit; single-element/same-type groups are not strong interventions",
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(out_dir / "interventions.json", payload)
    log("[interventions] done")
    return payload


# ---------------------------------------------------------------------------
# 12. CLI (local use; the registered runner calls the same functions)
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", default=str(RESULTS_DIR))
    parser.add_argument("--build-objects", action="store_true")
    parser.add_argument("--cssd-refit", action="store_true")
    parser.add_argument("--train-q", action="store_true")
    parser.add_argument("--checks", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--arm", default="C00", choices=list(ARMS))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--select-eval", action="store_true")
    parser.add_argument("--confirm-eval", action="store_true")
    parser.add_argument("--write-confirm-roster", default=None, help="comma list arm:seed entries")
    parser.add_argument("--interventions", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    if args.build_objects:
        phase_build_objects(out_dir=out_dir)
    if args.cssd_refit:
        cssd_refit(out_dir=out_dir, threads=int(args.threads))
    if args.train_q:
        train_q(out_dir=out_dir)
    if args.checks:
        run_checks(out_dir=out_dir)
    if args.smoke:
        run_smoke(device_name=args.device, out_dir=out_dir)
    if args.train:
        train_arm(
            args.arm, seed=int(args.seed), device_name=args.device,
            out_dir=out_dir, epochs=int(args.epochs),
            max_steps=None if args.max_steps is None else int(args.max_steps),
        )
    if args.select_eval:
        select_eval(device_name=args.device, out_dir=out_dir)
    if args.write_confirm_roster is not None:
        arms: list[str] = []
        seeds: set[int] = set()
        for entry in str(args.write_confirm_roster).split(","):
            arm, seed = entry.strip().split(":")
            arms.append(str(arm))
            seeds.add(int(seed))
        write_confirm_roster(sorted(set(arms)), sorted(seeds), out_dir=out_dir)
    if args.confirm_eval:
        confirm_eval(device_name=args.device, out_dir=out_dir)
    if args.interventions:
        run_interventions(device_name=args.device, out_dir=out_dir)
    if not any((
        args.build_objects, args.cssd_refit, args.train_q, args.checks, args.smoke,
        args.train, args.select_eval, args.confirm_eval, args.write_confirm_roster is not None,
        args.interventions,
    )):
        parser.print_help()
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
