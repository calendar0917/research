"""ZINC dictionary responsibility split: separate structure coding fused with
explicit local semantics (F construction) vs the joint 125-D tuple coding (J
construction), dictionary vs matched MLP, all under COMP supervision, seed 0.

Round: ``zinc_dictionary_fusion_clarity_overnight_seed0_v1``.

Motivation (luyin19): the joint tuple dictionary (structure+semantics coded
together in one 125-D tuple) is active but buys no advantage over a matched MLP.
The open question is the *information division of labour*: maybe structure and
semantics should be coded separately and combined explicitly, instead of being
fused inside one joint dictionary input.  This round compares, at exactly
matched parameters (297,539), matched body/bridge/reader, matched COMP
supervision and the identical real J incidence aggregation:

* ``J_D`` — source joint construction, 125-D tuple -> tied IHT top-8 dictionary
  (the source-round ``D_COMP`` arm, re-trained this round as the contemporaneous
  control);
* ``J_M`` — source joint construction, 125-D tuple -> SiLU projection (the
  source-round ``M_COMP`` arm);
* ``F_D`` — phi65 (pure topology) coded alone by a 65->64 dictionary via the
  source tied IHT (top-8), combined per tuple with an explicit semantic branch
  ``u = SiLU(q @ B_bar.T)`` over q = [onehot28(root atom); onehot28(neighbour
  atom); onehot4(bond)] (60-D), by elementwise product, then J-weight pooled;
* ``F_M`` — the same separate construction with a SiLU projection coder for the
  structure half (``z = SiLU(p @ A_S_bar.T)``), sharing the identical semantic
  branch initialisation.

All four arms: seed 0, Adam 1e-3, coupled WD 1e-5, clip 5, batch 128,
240 epochs (15,120 steps), soup 236-240, fresh init, fold A =
rng(20261006) 8000/2000 reused from the source round (all fit statistics
reused verbatim), COMP loss with fixed 0.5 component weight, reader 39->2.

Stage 2 (clarity contrasts, split A): ``F_D_I``/``F_M_I`` (marginal-I
incidence weights), ``F_D_RAND`` (structure dictionary frozen at the random
frame), ``F_D_REC`` (1000-step label-free reconstruction pre-training, then
frozen), ``F_D_REC_TASK`` (the same pre-trained dictionary, trainable).

Stage 3 (split B = rng(20261007) 8000/2000, all statistics refit on B): the
frozen-rule selected candidates vs the contemporaneous ``J_M``.

Stage 4 (only if Stage 3 passes the clear-signal gate): full-10k training and a
single frozen official-valid read.

Split B's fold indices are generated and frozen at Stage 0; its fitted
statistics are computed only after the Stage 3 purchase.  Official valid is
never loaded before Stage 4's frozen roster; official test is never loaded.

Usage::

    python -m tracks.ksvd.experiments.luyin16.\
zinc_dictionary_fusion_clarity_overnight_seed0_v1 --freeze-folds
    ... --compute-kappa-fusion            # split A (Stage 0)
    ... --init-check --device cuda
    ... --smoke --device cuda
    ... --rec-pretrain --split A --device cuda
    ... --train --arm J_D --device cuda   # (J_M, F_D, F_M, F_D_I, ...)
    ... --dev-eval --arms J_D,J_M,F_D,F_M --device cuda
    ... --mechanism-health --arms J_D,J_M,F_D,F_M --device cuda
    ... --build-B-objects                 # Stage 3 purchase
    ... --train --arm B_J_M --device cuda # (B_* arms)
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
    zinc_local_dictionary_component_supervision_seed0_v1 as src,
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
from tracks.ksvd.experiments.luyin16 import zinc_task_dictionary_and_cycle_witness_seed0_v1 as zw
from tracks.ksvd.experiments.luyin16 import (
    zinc_component_supervision_fulltrain_confirmation_seed0_v1 as zft,
)

PROTOCOL_VERSION = "zinc-dictionary-fusion-clarity-overnight-seed0-v1"
RESULT_SLUG = "zinc_dictionary_fusion_clarity_overnight_seed0_v1"
RESULTS_DIR = zjd.TRACK_ROOT / "results" / RESULT_SLUG

# ---- frozen recipe (identical for every formal arm; never changed) ---------
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
STEPS_PER_EPOCH = 63
STEPS_TOTAL = EPOCHS * STEPS_PER_EPOCH
FULL_STEPS_PER_EPOCH = 79            # ceil(10000/128) -> 18,960 steps at 240 ep
FULL_STEPS_TOTAL = EPOCHS * FULL_STEPS_PER_EPOCH
FULL_PREFIX = "T_"                   # Stage-4 full-10k arms (purchase only)
N_TRAIN = 10000
REPLAY_TOL = 1.0e-5
MU_LOGP = src.MU_LOGP
COMPONENT_LOSS_WEIGHT = 0.5

# ---- folds -----------------------------------------------------------------
FOLD_B_SEED = 20261007              # split B, frozen at Stage 0
KAPPA_SEED = src.KAPPA_SEED         # 20261004, same sampling recipe as the source
KAPPA_SAMPLE_MAX = src.KAPPA_SAMPLE_MAX

# ---- statistics / gates (frozen before any formal run) --------------------
BOOT_SEED = 20261010
N_BOOT = 2000
GATE_DELTA = 0.003                   # "clear performance signal" cal thresholds
CONFIRM_G0_CAL = 0.002              # Stage 3 purchase minimums vs J_M
CONFIRM_OVERALL_CAL = 0.001

STAGE1_ARMS = ("J_D", "J_M", "F_D", "F_M")
STAGE2_ARMS = ("F_D_I", "F_M_I", "F_D_RAND", "F_D_REC", "F_D_REC_TASK")

EXPECTED_PARAMETERS = src.EXPECTED_PARAMETERS          # 297,539
EXPECTED_BODY_PARAMETERS = src.EXPECTED_BODY_PARAMETERS  # 184,707 incl. reader
BRIDGE_PARAMETERS = src.BRIDGE_PARAMETERS              # 82,944
W_LOC_PARAMETERS = prev.W_LOC_OUT * prev.K_TUPLE       # 21,888
F_SEM_PARAMETERS = prev.K_TUPLE * (prev.TUPLE_DIM - prev.PHI_DIM)  # 64*60=3,840
F_STRUCT_PARAMETERS = prev.K_TUPLE * prev.PHI_DIM      # 64*65=4,160
LOCAL_PARAMETERS = src.LOCAL_PARAMETERS                # 8,000 + 21,888
GROUP_NAMES = ("k=0", "k=-1", "k=-2", "k<=-3")

MAX_SMOKE_RUNS = 2
SMOKE_MAX_STEPS = 4

# ---- REC pre-training (the single fixed recipe; Stage 2 only) -------------
REC_SAMPLER_SEED = 20261010
REC_STEPS = 1000
REC_BATCH = 256
REC_LR = 1.0e-3
REC_WD = 1.0e-5
REC_CLIP = 5.0


def _hash_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def array_sha256(value: np.ndarray, dtype: Any = np.float32) -> str:
    return src.array_sha256(value, dtype)


def file_sha256(path: Path) -> str:
    return src.file_sha256(path)


def state_hash(state: Mapping[str, torch.Tensor]) -> str:
    return src.state_hash(state)


def write_json(path: Path, payload: Mapping[str, Any] | Sequence[Any]) -> None:
    src.write_json(path, payload)


def seed_everything(seed: int) -> None:
    src.seed_everything(int(seed))


def resolve_device(name: str | None) -> torch.device:
    return src.resolve_device(name)


def group_masks(k: np.ndarray) -> dict[str, np.ndarray]:
    return zw.group_masks(k)


def allocation_probe() -> dict[str, Any]:
    return src.allocation_probe()


def _mae(err: np.ndarray) -> float:
    return float(np.mean(np.abs(err))) if err.size else float("nan")


# ---------------------------------------------------------------------------
# arm registry (single source of truth for what each arm is)
# ---------------------------------------------------------------------------


def arm_spec(arm: str) -> dict[str, Any]:
    """Decode an arm name into its frozen construction spec."""
    if not isinstance(arm, str) or not arm:
        raise ValueError(arm)
    split = "FULL" if arm.startswith(FULL_PREFIX) else ("B" if arm.startswith("B_") else "A")
    base = arm[2:] if split in ("B", "FULL") else arm
    spec: dict[str, Any] = {
        "arm": arm,
        "split": split,
        "base": base,
        "weight_key": "joint",
        "freeze_d_s": False,
        "d_s_source": "frame",
    }
    if base == "F_D_I":
        spec.update(base="F_D", weight_key="independent")
    elif base == "F_M_I":
        spec.update(base="F_M", weight_key="independent")
    elif base == "F_D_RAND":
        spec.update(base="F_D", freeze_d_s=True)
    elif base == "F_D_REC":
        spec.update(base="F_D", d_s_source="rec", freeze_d_s=True)
    elif base == "F_D_REC_TASK":
        spec.update(base="F_D", d_s_source="rec")
    elif base not in ("J_D", "J_M", "F_D", "F_M"):
        raise ValueError(f"unknown arm {arm!r}")
    return spec


def known_arms() -> list[str]:
    bases = list(STAGE1_ARMS) + list(STAGE2_ARMS)
    return bases + [f"B_{b}" for b in ("J_M", "F_D", "F_M", "F_D_I", "F_M_I", "F_D_RAND", "F_D_REC", "F_D_REC_TASK")] + [
        f"{FULL_PREFIX}{b}" for b in ("J_D", "J_M", "F_D", "F_M", "F_D_I", "F_M_I", "F_D_RAND")
    ]


def requires_rec(arm: str) -> bool:
    return arm_spec(arm)["d_s_source"] == "rec"


# ---------------------------------------------------------------------------
# fold B (rng(20261007); indices frozen at Stage 0; statistics only at Stage 3)
# ---------------------------------------------------------------------------


def build_fold_b() -> dict[str, Any]:
    perm = np.random.default_rng(FOLD_B_SEED).permutation(10000)
    fit_idx = np.sort(perm[:N_FIT]).astype(np.int64)
    dev_idx = np.sort(perm[N_FIT:]).astype(np.int64)
    fit_sha = _hash_bytes(fit_idx.tobytes())
    dev_sha = _hash_bytes(dev_idx.tobytes())
    checks = {
        "sizes_ok": bool(fit_idx.size == N_FIT and dev_idx.size == N_DEV),
        "disjoint": bool(np.intersect1d(fit_idx, dev_idx).size == 0),
        "cover": bool(np.union1d(fit_idx, dev_idx).size == 10000),
        "differs_from_fold_A": bool(fit_sha != src.NEW_FIT_SHA),
    }
    if not all(checks.values()):
        raise RuntimeError(f"fold B construction check failed: {checks}")
    return {
        "fit_idx": fit_idx,
        "dev_idx": dev_idx,
        "definition": (
            "perm=np.random.default_rng(20261007).permutation(10000); "
            "fit_idx=np.sort(perm[:8000]); dev_idx=np.sort(perm[8000:])"
        ),
        "checks": checks,
        "fit_sha256": fit_sha,
        "dev_sha256": dev_sha,
    }


def phase_freeze_folds(*, out_dir: Path = RESULTS_DIR, log: Any = print) -> dict[str, Any]:
    """Stage 0: freeze fold B indices (no label fitting, no statistics)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fold_b = build_fold_b()
    np.savez_compressed(out_dir / "B_fold.npz", fit_idx=fold_b["fit_idx"], dev_idx=fold_b["dev_idx"])
    # verify the reused split-A objects still load and match the frozen hashes
    new_a = src.load_new_objects(src.RESULTS_DIR)
    record = {
        "protocol_version": PROTOCOL_VERSION,
        "fold_A": {
            "definition": "rng(20261006) permutation, source round, reused verbatim",
            "fit_idx_sha256": src.NEW_FIT_SHA,
            "dev_idx_sha256": src.NEW_DEV_SHA,
            "verified_load": True,
            "manifest_constants": new_a["manifest"]["constants"],
        },
        "fold_B": {
            "definition": fold_b["definition"],
            "fit_idx_sha256": fold_b["fit_sha256"],
            "dev_idx_sha256": fold_b["dev_sha256"],
            "checks": fold_b["checks"],
            "statistics_fitted": False,
            "note": "split B statistics (prep/targets/scalers/kappa) are fitted only after the Stage 3 purchase",
        },
        "artifacts": {"B_fold.npz": {"sha256": file_sha256(out_dir / "B_fold.npz")}},
    }
    write_json(out_dir / "frozen_folds.json", record)
    log(
        f"[freeze-folds] B fit/dev sha={fold_b['fit_sha256'][:12]}/{fold_b['dev_sha256'][:12]} "
        f"(A reused, verified)"
    )
    return record


def load_fold_b(out_dir: Path = RESULTS_DIR) -> dict[str, Any]:
    record = json.loads((out_dir / "frozen_folds.json").read_text())
    with np.load(out_dir / "B_fold.npz", allow_pickle=False) as z:
        fold = {"fit_idx": z["fit_idx"].astype(np.int64), "dev_idx": z["dev_idx"].astype(np.int64)}
    expected = record["fold_B"]
    if _hash_bytes(fold["fit_idx"].tobytes()) != expected["fit_idx_sha256"]:
        raise RuntimeError("fold B fit index hash mismatch")
    if _hash_bytes(fold["dev_idx"].tobytes()) != expected["dev_idx_sha256"]:
        raise RuntimeError("fold B dev index hash mismatch")
    fold["definition"] = expected["definition"]
    return fold


# ---------------------------------------------------------------------------
# the F operators (explicit reference implementations, fit-free, label-free)
# ---------------------------------------------------------------------------


def structure_codes_ref(structure: str, s_raw: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """z(v): the structure code of the standardised 65-D phi coordinate.

    ``dict``: the *source-tied* IHT operator verbatim (``prev.tied_tuple_codes``
    with a 65-D input): 10 steps, hard top-8 of 64, eta =
    1/(1.05*sigma(Dbar)^2+1e-12) with detached power-iteration sigma.
    ``mlp``: ``SiLU(p @ A_S_bar.T)`` with row-L2-normalised A_S.
    """
    if structure == "dict":
        if tuple(s_raw.shape) != (prev.PHI_DIM, prev.K_TUPLE):
            raise ValueError(f"D_S_raw shape {tuple(s_raw.shape)} != {(prev.PHI_DIM, prev.K_TUPLE)}")
        return prev.tied_tuple_codes(s_raw, p)
    if structure == "mlp":
        if tuple(s_raw.shape) != (prev.K_TUPLE, prev.PHI_DIM):
            raise ValueError(f"A_S_raw shape {tuple(s_raw.shape)} != {(prev.K_TUPLE, prev.PHI_DIM)}")
        a_bar = F.normalize(s_raw, dim=1, eps=1e-12)
        return F.silu(p @ a_bar.t())
    raise ValueError(structure)


def semantic_codes_ref(b_raw: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """u(v,a,t) = SiLU(q @ B_bar.T); B_bar is row-L2 normalised [64,60]."""
    if tuple(b_raw.shape) != (prev.K_TUPLE, prev.TUPLE_DIM - prev.PHI_DIM):
        raise ValueError(f"B_raw shape {tuple(b_raw.shape)}")
    b_bar = F.normalize(b_raw, dim=1, eps=1e-12)
    return F.silu(q @ b_bar.t())


def standardised_phi(phi: torch.Tensor, phi_mean: torch.Tensor, phi_std: torch.Tensor, phi_scale: float) -> torch.Tensor:
    return (phi - phi_mean) / phi_std / float(phi_scale)


def compose_root_codes_f(
    structure: str,
    s_raw: torch.Tensor,
    b_raw: torch.Tensor,
    phi: torch.Tensor,
    atom: torch.Tensor,
    pair_ptr: torch.Tensor,
    pair_t: torch.Tensor,
    pair_a: torch.Tensor,
    pair_w: torch.Tensor,
    root_ids: torch.Tensor,
    phi_mean: torch.Tensor,
    phi_std: torch.Tensor,
    phi_scale: float,
) -> torch.Tensor:
    """``e_F(v) = sum_(a,t) w(v,a,t) * z(v) * u(v,a,t)`` for requested roots.

    Explicit reference operator (mirrors ``prev.compose_root_codes``): the
    structure code z(v) is computed once per root over the standardised phi65
    (this is algebraically identical to a per-tuple recomputation and keeps the
    real gradient path through z); the semantic code u is per tuple; they are
    combined by elementwise product *before* the J-weight incidence pooling.
    """
    start = pair_ptr[root_ids]
    end = pair_ptr[root_ids + 1]
    counts = end - start
    total = int(counts.sum().item())
    n_roots = int(root_ids.numel())
    if total == 0:
        return torch.zeros(n_roots, prev.K_TUPLE, device=phi.device, dtype=phi.dtype)
    offsets = counts.cumsum(0) - counts
    flat = torch.repeat_interleave(start, counts) + (
        torch.arange(total, device=phi.device) - torch.repeat_interleave(offsets, counts)
    )
    tuple_row = torch.repeat_interleave(torch.arange(n_roots, device=phi.device), counts)
    t = pair_t[flat]
    a_u = pair_a[flat]
    w = pair_w[flat]
    p = standardised_phi(phi[root_ids], phi_mean, phi_std, phi_scale)
    z = structure_codes_ref(structure, s_raw, p)
    atom_v = atom[root_ids][tuple_row]
    q = torch.cat(
        [
            F.one_hot(atom_v, prev.ATOM_CATEGORIES).to(phi.dtype),
            F.one_hot(a_u, prev.ATOM_CATEGORIES).to(phi.dtype),
            F.one_hot(t, prev.BOND_CATEGORIES).to(phi.dtype),
        ],
        dim=1,
    )
    u = semantic_codes_ref(b_raw, q)
    f = z[tuple_row] * u
    out = torch.zeros(n_roots, prev.K_TUPLE, device=phi.device, dtype=f.dtype)
    out.index_add_(0, tuple_row, f * w.unsqueeze(1))
    return out


def naive_root_codes_f(
    structure: str,
    s_raw: torch.Tensor,
    b_raw: torch.Tensor,
    payload: prev.TuplePayload,
    phi: torch.Tensor,
    atom: torch.Tensor,
    root_ids: Sequence[int],
) -> np.ndarray:
    """Independent per-root/per-pair Python reference (no gather operators)."""
    out = np.zeros((len(root_ids), prev.K_TUPLE), np.float64)
    b_bar = F.normalize(b_raw, dim=1, eps=1e-12)
    for row, root in enumerate(int(r) for r in root_ids):
        start = int(payload.pair_ptr[root].item())
        end = int(payload.pair_ptr[root + 1].item())
        if end <= start:
            continue
        p = standardised_phi(
            phi[root].view(1, -1), payload.phi_mean, payload.phi_std, payload.phi_scale
        )
        z = structure_codes_ref(structure, s_raw, p).double()[0]
        acc = torch.zeros(prev.K_TUPLE, dtype=torch.float64)
        for pair in range(start, end):
            a_u = torch.tensor([int(payload.pair_a[pair].item())], dtype=torch.long)
            t = torch.tensor([int(payload.pair_t[pair].item())], dtype=torch.long)
            atom_v = torch.tensor([int(atom[root].item())], dtype=torch.long)
            q = torch.cat(
                [
                    F.one_hot(atom_v, prev.ATOM_CATEGORIES).double(),
                    F.one_hot(a_u, prev.ATOM_CATEGORIES).double(),
                    F.one_hot(t, prev.BOND_CATEGORIES).double(),
                ],
                dim=1,
            )
            u = F.silu(q @ b_bar.t().double())[0]
            acc = acc + float(payload.pair_wJ[pair].item()) * (z * u)
        out[row] = acc.numpy()
    return out


# ---------------------------------------------------------------------------
# the F local encoder (production path inside the model)
# ---------------------------------------------------------------------------


class FusionEncoder(nn.Module):
    """Separate structure/semantic local encoder + zero-init W_loc injection.

    ``z(v)``: structure code over the standardised 65-D phi (pure topology) —
    either the source-tied IHT dictionary (``D_S_raw`` [65,64], top-8 of 64,
    col-L2 frame) or the SiLU projection (``A_S_raw`` [64,65], row-L2 frame).
    ``u(v,a,t) = SiLU(q @ B_bar.T)``: explicit semantic branch over
    q = [onehot28(root atom); onehot28(neighbour atom); onehot4(bond)] (60-D).
    Per tuple ``f = z(v) * u(v,a,t)`` (elementwise, 64-D, SiLU applied to the
    semantic branch *before* aggregation), then ``e(v) = sum_p w_p f_p`` over
    the real J (or marginal I) incidence weights.  ``W_loc`` [342,64] is the
    only injection into the fusion first-layer preactivation, exactly as in
    the J arms.  No message passing, no second layer, no bias/gain/norm.
    """

    def __init__(
        self,
        payload: prev.TuplePayload,
        *,
        structure: str,
        kappa: float,
        d_s_init: torch.Tensor,
        b_init: torch.Tensor,
        weight_key: str = "joint",
        freeze_d_s: bool = False,
    ) -> None:
        super().__init__()
        if structure not in ("dict", "mlp"):
            raise ValueError(structure)
        if weight_key not in ("joint", "independent"):
            raise ValueError(weight_key)
        self.structure = str(structure)
        self.weight_key = str(weight_key)
        self.kappa = float(kappa)
        if structure == "dict":
            if tuple(d_s_init.shape) != (prev.PHI_DIM, prev.K_TUPLE):
                raise ValueError(f"d_s_init {tuple(d_s_init.shape)} != [65,64]")
            self.D_S_raw = nn.Parameter(d_s_init.detach().clone().contiguous())
            if freeze_d_s:
                self.D_S_raw.requires_grad_(False)
            self._d_s_name = "D_S_raw"
        else:
            if tuple(d_s_init.shape) != (prev.K_TUPLE, prev.PHI_DIM):
                raise ValueError(f"d_s_init {tuple(d_s_init.shape)} != [64,65]")
            self.A_S_raw = nn.Parameter(d_s_init.detach().clone().contiguous())
            if freeze_d_s:
                self.A_S_raw.requires_grad_(False)
            self._d_s_name = "A_S_raw"
        if tuple(b_init.shape) != (prev.K_TUPLE, prev.TUPLE_DIM - prev.PHI_DIM):
            raise ValueError(f"b_init {tuple(b_init.shape)} != [64,60]")
        self.B_raw = nn.Parameter(b_init.detach().clone().contiguous())
        self.W_loc = nn.Parameter(torch.zeros(prev.W_LOC_OUT, prev.K_TUPLE, dtype=torch.float32))
        self.freeze_d_s = bool(freeze_d_s)
        self._payload = payload
        self._device_cache: dict[str, dict[str, torch.Tensor]] = {}
        self.ablate = False
        self.mean_replace: torch.Tensor | None = None
        self.last_stats: dict[str, Any] = {}

    # -- payload / feature helpers -----------------------------------------
    def _buf(self, name: str, device: torch.device) -> torch.Tensor:
        cache = self._device_cache.setdefault(str(device), {})
        if name not in cache:
            cache[name] = self._payload[name].to(device)
        return cache[name]

    def _s_raw(self) -> torch.Tensor:
        return getattr(self, self._d_s_name)

    def structure_codes(self, p: torch.Tensor) -> torch.Tensor:
        return structure_codes_ref(self.structure, self._s_raw(), p)

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
        phi = data.dict_phi
        atom = data.dict_atom.to(torch.long)
        device = phi.device
        n_rows = int(phi.shape[0])
        mol = data.local_mol_id.to(torch.long).reshape(-1)
        n_graphs = int(mol.numel())
        batch = data.batch.to(torch.long)
        graph_counts = torch.bincount(batch, minlength=n_graphs)
        ptr = torch.cat([graph_counts.new_zeros(1), graph_counts.cumsum(0)])
        rows = torch.arange(n_rows, device=device)
        local_pos = rows - ptr[batch]
        root_base = self._buf("root_base", device)
        global_root = root_base[mol][batch] + local_pos
        pair_ptr = self._buf("pair_ptr", device)
        start = pair_ptr[global_root]
        end = pair_ptr[global_root + 1]
        counts = end - start
        total = int(counts.sum().item())
        zeros = torch.zeros(n_rows, prev.K_TUPLE, device=device, dtype=phi.dtype)
        if total == 0:
            out = zeros
        else:
            offsets = counts.cumsum(0) - counts
            flat = torch.repeat_interleave(start, counts) + (
                torch.arange(total, device=device) - torch.repeat_interleave(offsets, counts)
            )
            tuple_row = torch.repeat_interleave(rows, counts)
            t = self._buf("pair_t", device)[flat]
            a_u = self._buf("pair_a", device)[flat]
            key = "pair_wI" if self.weight_key == "independent" else "pair_wJ"
            w = self._buf(key, device)[flat]
            p = standardised_phi(
                phi, self._buf("phi_mean", device), self._buf("phi_std", device), self._payload.phi_scale
            )
            z = self.structure_codes(p)                       # [n_roots, 64]
            q = torch.cat(
                [
                    F.one_hot(atom[tuple_row], prev.ATOM_CATEGORIES).to(phi.dtype),
                    F.one_hot(a_u, prev.ATOM_CATEGORIES).to(phi.dtype),
                    F.one_hot(t, prev.BOND_CATEGORIES).to(phi.dtype),
                ],
                dim=1,
            )                                                # [n_tuples, 60]
            u = semantic_codes_ref(self.B_raw, q)            # [n_tuples, 64]
            f = z[tuple_row] * u
            out = zeros
            out.index_add_(0, tuple_row, f * w.unsqueeze(1))
        with torch.no_grad():
            s_raw = self._s_raw()
            if total > 0:
                z_nonzero = z.detach() != 0
                z_nnz_mean = float(z_nonzero.float().sum(1).mean())
                atoms_used = z_nonzero.any(dim=0)
            else:
                z_nnz_mean = 0.0
                atoms_used = torch.zeros(prev.K_TUPLE, dtype=torch.bool, device=device)
            self.last_stats = {
                "n_roots": n_rows,
                "n_molecules": n_graphs,
                "n_tuples": total,
                "structure": self.structure,
                "structure_code_nnz": z_nnz_mean,
                "structure_atoms_used": int(atoms_used.sum()),
                "structure_atoms_used_fraction": float(atoms_used.float().mean()),
                "root_code_nonzero_fraction": float((out != 0).float().mean()),
                "root_code_dead_dims": int((out.std(dim=0) < 1e-8).sum()) if n_rows > 1 else 0,
                "root_code_rms": float(out.pow(2).mean().sqrt()),
                "s_raw_norm": float(s_raw.norm()),
                "B_raw_norm": float(self.B_raw.norm()),
                "W_loc_norm": float(self.W_loc.norm()),
                "weight_key": self.weight_key,
                "freeze_d_s": bool(self.freeze_d_s),
            }
        if self.mean_replace is not None and not self.ablate:
            has_neighbours = counts > 0
            replacement = self.mean_replace.to(device=out.device, dtype=out.dtype)
            out = torch.where(has_neighbours.unsqueeze(1), replacement.unsqueeze(0), torch.zeros_like(out))
        if self.ablate:
            return torch.zeros_like(out)
        return out

    def forward(self, data: Any) -> torch.Tensor:
        codes = self.root_codes(data)
        scaled = codes * self.kappa
        with torch.no_grad():
            contribution = F.linear(scaled, self.W_loc)
            self.last_stats = dict(self.last_stats)
            self.last_stats["e_rms_scaled"] = float(scaled.pow(2).mean().sqrt())
            self.last_stats["contribution_rms"] = float(contribution.pow(2).mean().sqrt())
        return scaled


# ---------------------------------------------------------------------------
# arm construction (J arms are the source-round constructors, verbatim)
# ---------------------------------------------------------------------------


def arm_parameter_audit(model: nn.Module) -> dict[str, int]:
    audit = src.arm_parameter_audit(model)
    enc = model.local_tuple
    if isinstance(enc, FusionEncoder):
        audit = dict(audit)
        audit["local_parts"] = {
            "structure": int(enc._s_raw().numel()),
            "semantic_B": int(enc.B_raw.numel()),
            "W_loc": int(enc.W_loc.numel()),
            "structure_trainable": int(enc._s_raw().numel()) if enc._s_raw().requires_grad else 0,
        }
    return audit


def f_arm_kappa(split: str, structure: str, objects: Mapping[str, Any]) -> float:
    fusion = objects["fusion_kappa"]
    return float(fusion["kappa_F_D" if structure == "dict" else "kappa_F_M"])


def load_rec_state(split: str, out_dir: Path = RESULTS_DIR) -> dict[str, Any]:
    path = out_dir / f"rec_pretrain_{split}.json"
    if not path.exists():
        raise FileNotFoundError(f"REC pre-training for split {split} missing: {path}")
    return json.loads(path.read_text())


def build_arm_model(arm: str, objects: Mapping[str, Any]) -> nn.Module:
    """Fresh seed-0 arm from the frozen construction specs (no trained state)."""
    spec = arm_spec(arm)
    payload = objects["payload"]
    if not isinstance(payload, prev.TuplePayload):
        payload = prev.TuplePayload(objects["payload_arrays"])
    if spec["base"] == "J_D":
        model = src.build_arm_comp("D_COMP", payload, float(objects["kappa"]["kappa_M"]))
    elif spec["base"] == "J_M":
        model = src.build_arm_comp("M_COMP", payload, float(objects["kappa"]["kappa_M"]))
    else:
        structure = "dict" if spec["base"] == "F_D" else "mlp"
        model = prev.build_arm("J", payload)     # exact source skeleton constructor
        d0 = prev.init_d_loc()                   # canonical untrained frame, seed 20261004
        if spec["d_s_source"] == "rec":
            rec = load_rec_state(spec["split"])
            d_s_init = torch.as_tensor(np.asarray(rec["D_S_last"], np.float32))
            expected_shape = (prev.PHI_DIM, prev.K_TUPLE) if structure == "dict" else (prev.K_TUPLE, prev.PHI_DIM)
            if tuple(d_s_init.shape) != expected_shape:
                raise RuntimeError(f"REC D_S shape {tuple(d_s_init.shape)} != {expected_shape}")
        else:
            d_s_init = d0[: prev.PHI_DIM, :].contiguous() if structure == "dict" else d0[: prev.PHI_DIM, :].t().contiguous()
        b_init = d0[prev.PHI_DIM:, :].t().contiguous()      # [64,60], shared by both F arms
        kappa = f_arm_kappa(spec["split"], structure, objects)
        if spec["d_s_source"] == "rec":
            rec = load_rec_state(spec["split"])
            kappa = float(rec["kappa_REC"]) if structure == "dict" else kappa
        encoder = FusionEncoder(
            payload,
            structure=structure,
            kappa=kappa,
            d_s_init=d_s_init,
            b_init=b_init,
            weight_key=spec["weight_key"],
            freeze_d_s=spec["freeze_d_s"],
        )
        model.local_tuple = encoder
        model.reader = zcs.ComponentReader(model.reader.net)   # isolated RNG (fork_rng)
    audit = arm_parameter_audit(model)
    expected_total = EXPECTED_PARAMETERS
    if audit["total_parameters"] != expected_total:
        raise RuntimeError(f"{arm} parameter audit failed: {audit} (total != {expected_total})")
    if audit["base_body_parameters"] != EXPECTED_BODY_PARAMETERS or audit["bridge_parameters"] != BRIDGE_PARAMETERS:
        raise RuntimeError(f"{arm} body/bridge audit failed: {audit}")
    if audit["local_tuple_parameters"] != LOCAL_PARAMETERS:
        raise RuntimeError(f"{arm} local parameter audit failed: {audit}")
    if not isinstance(model.local_dictionary_bridge, zw.MLPBridge):
        raise RuntimeError(f"{arm} posterior bridge is not the matched MLP bridge")
    return model


# ---------------------------------------------------------------------------
# split objects (A: source objects reused verbatim; B: refit at Stage 3 only)
# ---------------------------------------------------------------------------


def load_split_objects(split: str, out_dir: Path = RESULTS_DIR, *, require_fusion: bool = True) -> dict[str, Any]:
    """Load every frozen object for the requested split (hash-checked)."""
    if split == "A":
        new = src.load_new_objects(src.RESULTS_DIR)
        fusion_path = out_dir / "kappa_fusion_A.json"
        if fusion_path.exists():
            fusion = json.loads(fusion_path.read_text())
        elif require_fusion:
            raise FileNotFoundError("kappa_fusion_A.json missing; run --compute-kappa-fusion first")
        else:
            fusion = None
        return {
            "split": "A",
            "fold": new["fold"],
            "targets": new["targets"],
            "constants": new["constants"],
            "payload_arrays": new["payload_arrays"],
            "payload": prev.TuplePayload(new["payload_arrays"]),
            "prep": new["prep"],
            "kappa": new["kappa"],
            "fusion_kappa": fusion,
            "manifest": new["manifest"],
        }
    if split == "B":
        manifest = json.loads((out_dir / "B_objects_manifest.json").read_text())
        paths = {
            "B_fold.npz": out_dir / "B_fold.npz",
            "B_targets.npz": out_dir / "B_targets.npz",
            "B_tuple_payload.npz": out_dir / "B_tuple_payload.npz",
            "B_prep.npz": out_dir / "B_prep.npz",
        }
        for name, path in paths.items():
            expected = manifest["artifacts"][name]["sha256"]
            if file_sha256(path) != expected:
                raise RuntimeError(f"B artifact {name} sha256 mismatch")
        with np.load(paths["B_fold.npz"], allow_pickle=False) as z:
            fold = {"fit_idx": z["fit_idx"].astype(np.int64), "dev_idx": z["dev_idx"].astype(np.int64)}
        with np.load(paths["B_targets.npz"], allow_pickle=False) as z:
            targets = {key: z[key] for key in ("y", "c", "g", "k", "ell", "s", "gid")}
            constants = {
                str(n): float(v) for n, v in zip(z["constant_names"].tolist(), z["constants"].tolist())
            }
        with np.load(paths["B_tuple_payload.npz"], allow_pickle=False) as z:
            payload_arrays = {key: z[key] for key in z.files}
        with np.load(paths["B_prep.npz"], allow_pickle=False) as z:
            prep = {key: z[key] for key in z.files}
        kappa = json.loads((out_dir / "B_kappa.json").read_text())
        fusion_path = out_dir / f"kappa_fusion_B.json"
        if require_fusion and not fusion_path.exists():
            raise FileNotFoundError("kappa_fusion_B.json missing; run --compute-kappa-fusion --split B first")
        fusion = json.loads(fusion_path.read_text()) if fusion_path.exists() else None
        return {
            "split": "B",
            "fold": fold,
            "targets": targets,
            "constants": constants,
            "payload_arrays": payload_arrays,
            "payload": prev.TuplePayload(payload_arrays),
            "prep": prep,
            "kappa": kappa,
            "fusion_kappa": fusion,
            "manifest": manifest,
        }
    if split == "FULL":
        manifest = json.loads((out_dir / "FULL_objects_manifest.json").read_text())
        paths = {
            "FULL_fold.npz": out_dir / "FULL_fold.npz",
            "FULL_targets.npz": out_dir / "FULL_targets.npz",
            "FULL_tuple_payload.npz": out_dir / "FULL_tuple_payload.npz",
            "FULL_prep.npz": out_dir / "FULL_prep.npz",
        }
        for name, path in paths.items():
            if not path.exists():
                raise FileNotFoundError(f"FULL artifact {name} missing; run --build-FULL-objects first")
            expected = manifest["artifacts"][name]["sha256"]
            if file_sha256(path) != expected:
                raise RuntimeError(f"FULL artifact {name} sha256 mismatch")
        with np.load(paths["FULL_fold.npz"], allow_pickle=False) as z:
            fold = {"fit_idx": z["fit_idx"].astype(np.int64), "dev_idx": z["dev_idx"].astype(np.int64)}
        with np.load(paths["FULL_targets.npz"], allow_pickle=False) as z:
            targets = {key: z[key] for key in ("y", "c", "g", "k", "ell", "s", "gid")}
            constants = {
                str(n): float(v) for n, v in zip(z["constant_names"].tolist(), z["constants"].tolist())
            }
        with np.load(paths["FULL_tuple_payload.npz"], allow_pickle=False) as z:
            payload_arrays = {key: z[key] for key in z.files}
        with np.load(paths["FULL_prep.npz"], allow_pickle=False) as z:
            prep = {key: z[key] for key in z.files}
        kappa = json.loads((out_dir / "FULL_kappa.json").read_text())
        fusion_path = out_dir / f"kappa_fusion_FULL.json"
        if require_fusion and not fusion_path.exists():
            raise FileNotFoundError("kappa_fusion_FULL.json missing; run --compute-kappa-fusion --split FULL first")
        fusion = json.loads(fusion_path.read_text()) if fusion_path.exists() else None
        return {
            "split": "FULL",
            "fold": fold,
            "targets": targets,
            "constants": constants,
            "payload_arrays": payload_arrays,
            "payload": prev.TuplePayload(payload_arrays),
            "prep": prep,
            "kappa": kappa,
            "fusion_kappa": fusion,
            "manifest": manifest,
        }
    raise ValueError(split)


def build_prepared_data(objects: Mapping[str, Any]) -> tuple[dict[str, Any], list[Any], list[Any]]:
    """Train-only loader + split prep + fit/dev lists (never official valid)."""
    train_data = zftd.load_train_only()
    if len(train_data) != 10000:
        raise RuntimeError("train-only cache length mismatch")
    fold = objects["fold"]
    if objects["split"] == "A":
        prep_meta = src.build_new_prep(train_data, fold)
        for key, value in objects["prep"].items():
            if key not in prep_meta or not np.array_equal(np.asarray(prep_meta[key], np.float32), np.asarray(value, np.float32)):
                raise RuntimeError(f"split-A prep mismatch at {key}")
    elif objects["split"] == "FULL":
        prep_meta = zw.apply_new_fit_prep(
            train_data, np.load(zfr.PREP_BLOB, allow_pickle=False), np.arange(N_TRAIN, dtype=np.int64)
        )
        for key in ("patch_fit_mean", "patch_fit_scale", "ctx_fit_mean", "ctx_fit_scale",
                    "anchor_fit_mean", "anchor_fit_scale", "topo_fit_mean", "topo_fit_scale"):
            if not np.array_equal(np.asarray(prep_meta[key], np.float32), np.asarray(objects["prep"][key], np.float32)):
                raise RuntimeError(f"split-FULL prep mismatch at {key}")
    else:
        prep_meta = zw.apply_new_fit_prep(train_data, np.load(zfr.PREP_BLOB, allow_pickle=False), fold["fit_idx"])
        for key in ("patch_fit_mean", "patch_fit_scale", "ctx_fit_mean", "ctx_fit_scale",
                    "anchor_fit_mean", "anchor_fit_scale", "topo_fit_mean", "topo_fit_scale"):
            if not np.array_equal(np.asarray(prep_meta[key], np.float32), np.asarray(objects["prep"][key], np.float32)):
                raise RuntimeError(f"split-B prep mismatch at {key}")
    fit_data, dev_data = zftd.build_fit_dev(train_data, fold["fit_idx"], fold["dev_idx"])
    for position, index in enumerate(fold["fit_idx"].tolist()):
        fit_data[position].local_mol_id = torch.tensor([int(index)], dtype=torch.long)
    for position, index in enumerate(fold["dev_idx"].tolist()):
        dev_data[position].local_mol_id = torch.tensor([int(index)], dtype=torch.long)
    return prep_meta, fit_data, dev_data


# ---------------------------------------------------------------------------
# Stage 0: one initial scale match for the F arms (label-free, fit-only)
# ---------------------------------------------------------------------------


def _phi_atom_tensors() -> tuple[torch.Tensor, torch.Tensor]:
    phi_env, atom_env, _node = prev._env_phi_atom()
    return (
        torch.as_tensor(phi_env, dtype=torch.float32),
        torch.as_tensor(atom_env, dtype=torch.long),
    )


def compute_fusion_kappa(
    split: str,
    *,
    out_dir: Path = RESULTS_DIR,
    log: Any = print,
) -> dict[str, Any]:
    """R* = RMS(kappa_D * e_JD) on the frozen fit-root sample (fresh state);
    kappa_F = R* / RMS(e_F) per F arm (J weights, fresh init).  J_M keeps the
    source-verified kappa.  No labels, no dev, frozen afterwards."""
    torch.set_num_threads(8)
    objects = load_split_objects(split, require_fusion=False)
    payload = objects["payload"]
    phi_t, atom_t = _phi_atom_tensors()
    sample = payload.kappa_sample
    kappa_D = float(payload.kappa)
    d0 = prev.init_d_loc()
    with torch.no_grad():
        e_jd = prev.compose_root_codes(
            d0, phi_t, atom_t, payload.pair_ptr, payload.pair_t, payload.pair_a,
            payload.pair_wJ, sample, payload.phi_mean, payload.phi_std, payload.phi_scale,
        )
        r_star = float((kappa_D * e_jd).pow(2).mean().sqrt())
        d_s_dict = d0[: prev.PHI_DIM, :].contiguous()
        d_s_mlp = d0[: prev.PHI_DIM, :].t().contiguous()
        b0 = d0[prev.PHI_DIM:, :].t().contiguous()
        e_fd = compose_root_codes_f(
            "dict", d_s_dict, b0, phi_t, atom_t, payload.pair_ptr, payload.pair_t,
            payload.pair_a, payload.pair_wJ, sample, payload.phi_mean, payload.phi_std, payload.phi_scale,
        )
        e_fm = compose_root_codes_f(
            "mlp", d_s_mlp, b0, phi_t, atom_t, payload.pair_ptr, payload.pair_t,
            payload.pair_a, payload.pair_wJ, sample, payload.phi_mean, payload.phi_std, payload.phi_scale,
        )
        # J/I identifiability of the F interface at the fresh state (fit sample)
        e_fd_i = compose_root_codes_f(
            "dict", d_s_dict, b0, phi_t, atom_t, payload.pair_ptr, payload.pair_t,
            payload.pair_a, payload.pair_wI, sample, payload.phi_mean, payload.phi_std, payload.phi_scale,
        )
        e_fm_i = compose_root_codes_f(
            "mlp", d_s_mlp, b0, phi_t, atom_t, payload.pair_ptr, payload.pair_t,
            payload.pair_a, payload.pair_wI, sample, payload.phi_mean, payload.phi_std, payload.phi_scale,
        )
    rms_fd = float(e_fd.pow(2).mean().sqrt())
    rms_fm = float(e_fm.pow(2).mean().sqrt())
    kappa_f_d = r_star / rms_fd if rms_fd > 0 else float("nan")
    kappa_f_m = r_star / rms_fm if rms_fm > 0 else float("nan")
    for name, value in (("kappa_F_D", kappa_f_d), ("kappa_F_M", kappa_f_m)):
        if not math.isfinite(value) or value <= 0.0 or value > 1e9:
            raise RuntimeError(f"MECHANISM_INIT_BLOCKED: {name}={value}")
    record = {
        "definition": (
            "R* = RMS(kappa_D * e_JD_init) over the frozen <=8192 fit-root sample "
            "(J weights, fresh init); kappa_F = R*/RMS(e_F_init) per F arm; "
            "J_M keeps the source-verified kappa_M; frozen after this fit-only, "
            "label-free one-shot match"
        ),
        "split": split,
        "sample_sha256": array_sha256(sample.numpy(), np.int64),
        "n_roots": int(sample.numel()),
        "kappa_D": kappa_D,
        "kappa_M": float(objects["kappa"]["kappa_M"]),
        "r_star": r_star,
        "e_F_D_rms_unscaled": rms_fd,
        "e_F_M_rms_unscaled": rms_fm,
        "kappa_F_D": float(kappa_f_d),
        "kappa_F_M": float(kappa_f_m),
        "scaled_rms": {
            "kappa_D_e_JD": r_star,
            "kappa_F_D_e_FD": float((kappa_f_d * e_fd).pow(2).mean().sqrt()),
            "kappa_F_M_e_FM": float((kappa_f_m * e_fm).pow(2).mean().sqrt()),
        },
        "JI_identifiability_fresh": {
            "e_F_D_J_minus_I_rms": float((e_fd - e_fd_i).pow(2).mean().sqrt()),
            "e_F_M_J_minus_I_rms": float((e_fm - e_fm_i).pow(2).mean().sqrt()),
            "note": "if ~0 the I-weights contrast is unidentifiable in this interface (Stage 2 I arms must then be skipped)",
        },
        "labels_used": False,
        "dev_used": False,
    }
    write_json(out_dir / f"kappa_fusion_{split}.json", record)
    log(
        f"[kappa-fusion {split}] R*={r_star:.6f} kappa_F_D={kappa_f_d:.6f} "
        f"kappa_F_M={kappa_f_m:.6f} JI_diff={record['JI_identifiability_fresh']['e_F_D_J_minus_I_rms']:.3e}"
    )
    return record


# ---------------------------------------------------------------------------
# Stage 0 init checks (all pre-training, fit-only, no labels in forward)
# ---------------------------------------------------------------------------


def _ji_synthetic_checks() -> dict[str, Any]:
    """J vs marginal-I weights: synthetic distinguishability + boundaries."""
    # n = 3 root; grid [4 bond types x 28 atom types]
    count = np.zeros((3, prev.BOND_CATEGORIES, prev.ATOM_CATEGORIES), dtype=np.int64)
    # root 0: two bonds of different types to different atom kinds -> realized
    # support is strictly inside the marginal support
    count[0, 0, 5] = 1
    count[0, 1, 7] = 1
    # root 1: a single incident edge (n=1 boundary: J == I exactly)
    count[1, 2, 9] = 1
    # root 2: no incident edge (n=0 boundary: both zero)
    d, n_t, n_a, w_j, w_i = prev.root_weights(count)
    checks = {
        "n0_both_zero": bool(float(np.abs(w_j[2]).sum()) == 0.0 and float(np.abs(w_i[2]).sum()) == 0.0),
        "n1_joint_equals_independent": bool(np.array_equal(w_j[1], w_i[1])),
        "n1_marginal_identity": bool(
            np.allclose((w_i[1] * d[1]).sum(axis=1), n_t[1]) and np.allclose((w_i[1] * d[1]).sum(axis=0), n_a[1])
        ),
        "n2_supports_differ": bool(
            float(np.abs(w_j[0]).sum()) > 0.0
            and not np.array_equal(np.nonzero(w_i[0].reshape(-1))[0], np.nonzero(w_j[0].reshape(-1))[0])
        ),
        "n2_marginal_identity": bool(
            np.allclose((w_i[0] * d[0]).sum(axis=1), n_t[0]) and np.allclose((w_i[0] * d[0]).sum(axis=0), n_a[0])
        ),
    }
    return {"synthetic_checks": checks, "passed": all(checks.values())}


def _ji_fit_checks(payload: prev.TuplePayload) -> dict[str, Any]:
    """J vs I incidence weights over the fit realized support (label-free)."""
    wj = payload.pair_wJ.numpy().astype(np.float64)
    wi = payload.pair_wI.numpy().astype(np.float64)
    realized = wj > 0.0
    diff = np.abs(wj[realized] - wi[realized])
    return {
        "n_realized_tuples": int(realized.sum()),
        "realized_wJI_rms_diff": float(np.sqrt((diff ** 2).mean())) if diff.size else 0.0,
        "realized_fraction_differing": float((diff > 0.0).mean()) if diff.size else 0.0,
        "marginal_support_extra_mass_I": float(wi[~realized].sum()),
    }


def init_check(
    *,
    out_dir: Path = RESULTS_DIR,
    device: torch.device | None = None,
    log: Any = print,
) -> dict[str, Any]:
    torch.set_num_threads(8)
    device = device or torch.device("cpu")
    objects = load_split_objects("A", out_dir)
    fold = objects["fold"]
    payload = objects["payload"]
    checks: dict[str, Any] = {}

    # (a) folds
    checks["fold_A"] = {"fit_sha": src.NEW_FIT_SHA, "dev_sha": src.NEW_DEV_SHA, "verified_on_load": True}
    fold_b = load_fold_b(out_dir)
    checks["fold_B"] = {
        "definition": fold_b["definition"],
        "fit_sha256": _hash_bytes(fold_b["fit_idx"].tobytes()),
        "dev_sha256": _hash_bytes(fold_b["dev_idx"].tobytes()),
        "differs_from_A": bool(_hash_bytes(fold_b["fit_idx"].tobytes()) != src.NEW_FIT_SHA),
    }

    # (b) J/I weights: synthetic + fit + fresh-state F-interface identifiability
    checks["J_vs_I"] = _ji_synthetic_checks()
    checks["J_vs_I_fit"] = _ji_fit_checks(payload)
    fusion = objects["fusion_kappa"]
    checks["J_vs_I"]["fresh_state_F_interface"] = fusion["JI_identifiability_fresh"]
    checks["J_vs_I"]["identifiable"] = bool(
        fusion["JI_identifiability_fresh"]["e_F_D_J_minus_I_rms"] > 1e-6
    )

    # (c) build all four Stage-1 arms; shared tensors identical; inits correct
    seed_everything(SEED)
    models = {}
    rngs = {}
    for arm in STAGE1_ARMS:
        seed_everything(SEED)
        models[arm] = build_arm_model(arm, objects).to(device)
        rngs[arm] = torch.get_rng_state()
    checks["build_rng_equal"] = bool(all(torch.equal(rngs[a], rngs["J_D"]) for a in STAGE1_ARMS))
    state = {a: {k: v.detach().cpu().clone() for k, v in models[a].state_dict().items()} for a in STAGE1_ARMS}
    shared_keys = sorted(k for k in state["J_D"] if not k.startswith("local_tuple."))
    for arm in STAGE1_ARMS:
        if sorted(k for k in state[arm] if not k.startswith("local_tuple.")) != shared_keys:
            raise RuntimeError(f"{arm} shared state key set differs")
    max_shared = max(
        float((state["J_D"][k] - state[arm][k]).abs().max()) for arm in STAGE1_ARMS for k in shared_keys
    )
    checks["shared_tensors_identical_1e-5"] = bool(max_shared <= src.TARGET_NOT_READ_TOL)
    checks["shared_tensor_max_abs_diff"] = max_shared
    d0 = prev.init_d_loc()
    checks["J_D_D_equals_D0"] = bool(torch.equal(state["J_D"]["local_tuple.D_loc_raw"], d0))
    checks["J_M_A_equals_D0T"] = bool(torch.equal(state["J_M"]["local_tuple.A_raw"], d0.t().contiguous()))
    checks["F_D_DS_equals_D0_first65"] = bool(
        torch.equal(state["F_D"]["local_tuple.D_S_raw"], d0[: prev.PHI_DIM, :].contiguous()))
    checks["F_M_AS_equals_D0_first65T"] = bool(
        torch.equal(state["F_M"]["local_tuple.A_S_raw"], d0[: prev.PHI_DIM, :].t().contiguous()))
    checks["F_arms_share_B_init"] = bool(torch.equal(
        state["F_D"]["local_tuple.B_raw"], state["F_M"]["local_tuple.B_raw"]))
    checks["F_B_equals_D0_last60T"] = bool(torch.equal(
        state["F_D"]["local_tuple.B_raw"], d0[prev.PHI_DIM:, :].t().contiguous()))
    checks["W_loc_exactly_zero_all"] = bool(all(
        float(state[a]["local_tuple.W_loc"].abs().sum()) == 0.0 for a in STAGE1_ARMS))
    checks["parameter_audit"] = {a: arm_parameter_audit(models[a]) for a in STAGE1_ARMS}
    checks["bridge_is_MLP_all"] = bool(all(
        isinstance(models[a].local_dictionary_bridge, zw.MLPBridge) for a in STAGE1_ARMS))
    checks["kappa"] = {
        "kappa_D": float(payload.kappa),
        "kappa_M": float(objects["kappa"]["kappa_M"]),
        "kappa_F_D": float(fusion["kappa_F_D"]),
        "kappa_F_M": float(fusion["kappa_F_M"]),
        "r_star": float(fusion["r_star"]),
    }

    # (d) forward label independence + shapes + identical initial graph function
    prep_meta, fit_data, dev_data = build_prepared_data(objects)
    g = np.asarray(objects["targets"]["g"], np.float64)
    target_g = torch.as_tensor(g[fold["fit_idx"]], dtype=torch.float32)
    batch = zftd.make_batch(fit_data, list(range(8)), target_g, device)
    preds, comps = {}, {}
    for arm in STAGE1_ARMS:
        models[arm].eval()
        with torch.no_grad():
            preds[arm] = models[arm](batch, mask=cm.C6_MASK)
            comps[arm] = models[arm].reader.components()
    checks["forward_shapes"] = {
        a: {"sum": list(preds[a].shape), "components": list(comps[a].shape)} for a in STAGE1_ARMS
    }
    checks["initial_graph_function_equal_all_arms"] = bool(all(
        torch.equal(preds[a], preds["J_D"]) and torch.equal(comps[a], comps["J_D"]) for a in STAGE1_ARMS))
    batch2 = zftd.make_batch(fit_data, list(range(8)), torch.zeros(8), device)
    with torch.no_grad():
        pred2 = models["F_D"](batch2, mask=cm.C6_MASK)
    checks["label_independence_max_abs"] = float((preds["F_D"] - pred2).abs().max())

    # (e) production root codes vs the explicit reference operators (one fit molecule)
    first_fit = int(fold["fit_idx"][0])
    root_base = np.asarray(objects["payload_arrays"]["root_base"], np.int64)
    global_roots = torch.tensor(
        list(range(int(root_base[first_fit]), int(root_base[first_fit + 1]))), dtype=torch.long)
    phi_t, atom_t = _phi_atom_tensors()
    ref = {}
    with torch.no_grad():
        ref["J_D"] = prev.compose_root_codes(
            models["J_D"].local_tuple.D_loc_raw.cpu(), phi_t, atom_t, payload.pair_ptr,
            payload.pair_t, payload.pair_a, payload.pair_wJ, global_roots,
            payload.phi_mean, payload.phi_std, payload.phi_scale)
        ref["J_M"] = mlpmod.compose_root_codes_m(
            models["J_M"].local_tuple.A_raw.cpu(), phi_t, atom_t, payload.pair_ptr,
            payload.pair_t, payload.pair_a, payload.pair_wJ, global_roots,
            payload.phi_mean, payload.phi_std, payload.phi_scale)
        ref["F_D"] = compose_root_codes_f(
            "dict", models["F_D"].local_tuple.D_S_raw.cpu(), models["F_D"].local_tuple.B_raw.cpu(),
            phi_t, atom_t, payload.pair_ptr, payload.pair_t, payload.pair_a,
            payload.pair_wJ, global_roots, payload.phi_mean, payload.phi_std, payload.phi_scale)
        ref["F_M"] = compose_root_codes_f(
            "mlp", models["F_M"].local_tuple.A_S_raw.cpu(), models["F_M"].local_tuple.B_raw.cpu(),
            phi_t, atom_t, payload.pair_ptr, payload.pair_t, payload.pair_a,
            payload.pair_wJ, global_roots, payload.phi_mean, payload.phi_std, payload.phi_scale)
    root_batch = zftd.make_batch(fit_data[:1], [0], target_g, device)
    ref_vs_prod = {}
    for arm in STAGE1_ARMS:
        with torch.no_grad():
            prod = models[arm].local_tuple.root_codes(root_batch)
        kappa = float(models[arm].local_tuple.kappa)
        ref_vs_prod[arm] = float((prod.cpu() * kappa - ref[arm] * kappa).abs().max())
    checks["root_codes_production_vs_reference_max_abs"] = ref_vs_prod

    # (f) independent naive reference for the F arms (a few roots, Python loop)
    naive = {}
    with torch.no_grad():
        for arm, structure in (("F_D", "dict"), ("F_M", "mlp")):
            enc = models[arm].local_tuple
            s_raw = enc.D_S_raw.cpu() if structure == "dict" else enc.A_S_raw.cpu()
            naive[arm] = naive_root_codes_f(
                structure, s_raw, enc.B_raw.cpu(), payload, phi_t, atom_t,
                [int(r) for r in global_roots[:5].tolist()],
            )
            prod5 = enc.root_codes(root_batch)[:5].cpu().numpy() * float(enc.kappa)
            naive[f"{arm}_max_abs_diff"] = float(np.max(np.abs(prod5 - naive[arm] * float(enc.kappa))))
    checks["naive_vs_production_max_abs"] = {k: v for k, v in naive.items() if k.endswith("_max_abs_diff")}

    # (g) IHT structure code sparsity witness + eta determinism
    with torch.no_grad():
        p5 = standardised_phi(phi_t[global_roots[:5]], payload.phi_mean, payload.phi_std, payload.phi_scale)
        z5 = models["F_D"].local_tuple.structure_codes(p5)
    checks["F_D_structure_code_nnz_max"] = int((z5 != 0).sum(1).max().item())
    checks["F_D_structure_code_nnz_le_8"] = bool(int((z5 != 0).sum(1).max().item()) <= prev.SPARSITY)
    checks["power_iter_sigma_deterministic"] = bool(
        float(tccd_v0.power_iter_sigma(F.normalize(models["F_D"].local_tuple.D_S_raw, dim=0))) ==
        float(tccd_v0.power_iter_sigma(F.normalize(models["F_D"].local_tuple.D_S_raw, dim=0))))

    # (h) schedule + loader RNG neutrality; step count
    rng_before = torch.get_rng_state().clone()
    schedule, schedule_hash = zw.build_schedule(len(fit_data), EPOCHS, SEED + TRAIN_SHUFFLE_OFFSET)
    _ = zftd.make_batch(fit_data, list(range(4)), target_g, torch.device("cpu"))
    checks["schedule_and_loader_rng_neutral"] = bool(torch.equal(rng_before, torch.get_rng_state()))
    checks["schedule_sha256"] = schedule_hash
    checks["steps_expected"] = STEPS_TOTAL
    if int(math.ceil(len(fit_data) / BATCH_SIZE)) * EPOCHS != STEPS_TOTAL:
        raise RuntimeError("step count does not match the frozen 15,120")

    # (i) serialization round-trip (list/dict/NumPy/Tensor), state save/reload
    probe_payload = {"a": [1, 2.5], "b": {"c": np.zeros(3, np.float32), "d": torch.ones(2)}}
    tmp = out_dir / "_serialization_probe.json"
    write_json(tmp, probe_payload)
    loaded = json.loads(tmp.read_text())
    checks["serialization_round_trip_ok"] = bool(
        loaded["a"] == [1, 2.5] and np.array_equal(np.asarray(loaded["b"]["c"], np.float32), np.zeros(3, np.float32)))
    tmp.unlink()
    state_path = out_dir / "_probe_state.pt"
    torch.save({k: v for k, v in state["F_D"].items()}, state_path)
    reloaded = build_arm_model("F_D", objects)
    reloaded.load_state_dict(
        {k: v.to(torch.device("cpu")) for k, v in torch.load(state_path, map_location="cpu", weights_only=True).items()},
        strict=True,
    )
    state_path.unlink()
    reloaded.eval()
    with torch.no_grad():
        reload_pred = reloaded.cpu()(batch.cpu(), mask=cm.C6_MASK)
    checks["state_save_reload_round_trip_max_abs"] = float((reload_pred - preds["F_D"].cpu()).abs().max())

    checks["official_valid_loaded"] = False
    checks["official_test_loaded"] = False
    write_json(out_dir / "init_identity.json", checks)
    log(
        f"[init] shared={max_shared:.2e} JI_identifiable={checks['J_vs_I']['identifiable']} "
        f"prod_vs_ref={max(ref_vs_prod.values()):.2e} nnz<=8={checks['F_D_structure_code_nnz_le_8']} "
        f"rng_neutral={checks['schedule_and_loader_rng_neutral']}"
    )
    return checks


# ---------------------------------------------------------------------------
# fit-only engineering smoke (max 2 runs, <= 4 steps per arm, states discarded)
# ---------------------------------------------------------------------------


def run_smoke(*, device: torch.device, out_dir: Path = RESULTS_DIR, log: Any = print) -> dict[str, Any]:
    smoke_path = out_dir / "smoke_runs.json"
    runs = json.loads(smoke_path.read_text())["runs"] if smoke_path.exists() else []
    if len(runs) >= MAX_SMOKE_RUNS:
        raise RuntimeError(f"smoke budget exhausted ({len(runs)}/{MAX_SMOKE_RUNS})")
    objects = load_split_objects("A", out_dir)
    payload = objects["payload"]
    fold = objects["fold"]
    prep_meta, fit_data, _dev = build_prepared_data(objects)
    g = np.asarray(objects["targets"]["g"], np.float64)
    ell = np.asarray(objects["targets"]["ell"], np.float64)
    s = np.asarray(objects["targets"]["s"], np.float64)
    fit_idx = fold["fit_idx"]
    target_g = torch.as_tensor(g[fit_idx], dtype=torch.float32)
    target_ell = torch.as_tensor(ell[fit_idx], dtype=torch.float32, device=device)
    target_s = torch.as_tensor(s[fit_idx], dtype=torch.float32, device=device)
    schedule, schedule_hash = zw.build_schedule(len(fit_data), EPOCHS, SEED + TRAIN_SHUFFLE_OFFSET)
    record: dict[str, Any] = {
        "device": str(device), "steps": SMOKE_MAX_STEPS, "arms": list(STAGE1_ARMS),
        "schedule_sha256": schedule_hash, "fit_only": True, "dev_scores_computed": False,
    }
    grad_log: dict[str, list[dict[str, Any]]] = {arm: [] for arm in STAGE1_ARMS}
    for arm in STAGE1_ARMS:
        seed_everything(SEED)
        model = build_arm_model(arm, objects).to(device)
        model.train()
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=WEIGHT_DECAY
        )
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
            entry: dict[str, Any] = {
                "step": step + 1, "clip_total_norm": total_norm,
                "W_loc_grad_norm": float(model.local_tuple.W_loc.grad.norm()),
            }
            for key in ("D_loc_raw", "A_raw", "D_S_raw", "A_S_raw", "B_raw"):
                if hasattr(model.local_tuple, key):
                    entry[f"{key}_task_grad_norm"] = float(getattr(model.local_tuple, key).grad.norm())
            grad_log[arm].append(entry)
            optimizer.step()
        # exercise the full save / reload / per-row prediction / JSON exits
        tmp_state = out_dir / f"_smoke_{arm}_state.pt"
        torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, tmp_state)
        replay = build_arm_model(arm, objects)
        replay.load_state_dict(
            {k: v.to("cpu") for k, v in torch.load(tmp_state, map_location="cpu", weights_only=True).items()},
            strict=True,
        )
        replay = replay.to(device)
        replay.eval()
        sums, comps_arr = src.evaluate_state_components(replay, fit_data[:16], device, batch_size=16)
        record[f"{arm}_save_reload_per_row_max_abs_bias"] = float(
            np.median(g[fit_idx[:16]] - sums))
        tmp_state.unlink()
        _ = json.dumps(zw.jsonable({"curve": [{"loss": float(loss.detach())}]}))
    for arm in STAGE1_ARMS:
        first, later = grad_log[arm][0], grad_log[arm][-1]
        if not (first["W_loc_grad_norm"] > 0.0):
            raise RuntimeError(f"{arm}: W_loc has no task gradient at step 1")
        for key in (k for k in later if k.endswith("_task_grad_norm")):
            if not (later[key] > 0.0):
                raise RuntimeError(f"{arm}: {key} has no task gradient by step {SMOKE_MAX_STEPS}")
    record["grad_log"] = grad_log
    record["states_discarded"] = True
    runs.append(record)
    write_json(smoke_path, {"runs": runs, "max_runs": MAX_SMOKE_RUNS})
    log(
        f"[smoke] run {len(runs)}/{MAX_SMOKE_RUNS} ok: W_loc step1 > 0 all arms; "
        f"local task grads live by step {SMOKE_MAX_STEPS}; save/reload exits exercised"
    )
    return record


# ---------------------------------------------------------------------------
# Stage 2: the single fixed REC pre-training recipe (label-free, fit-only)
# ---------------------------------------------------------------------------


def rec_pretrain(
    split: str,
    *,
    device: torch.device | None = None,
    out_dir: Path = RESULTS_DIR,
    log: Any = print,
) -> dict[str, Any]:
    """1000-step label-free reconstruction pre-training of the F_D structure
    dictionary on the frozen fit-root sample; fixed last (no best selection).

    Only ``D_S_raw`` trains.  Loss: batch mean of
    ``||p - alpha @ Dbar.T||^2 / (||p||^2 + 1e-12)`` with the same 10-step
    top-8 source-tied IHT encoding of the standardised phi65.  Sampler:
    ``rng(20261010)``, 256 roots per step with replacement.  Non-finite stops
    REC (no rescue).  After training, ``kappa_REC = R*/RMS(e_REC_J)`` on the
    same fit-root sample with J weights (R* from the split's frozen kappa file).
    """
    torch.set_num_threads(8)
    device = device or torch.device("cpu")
    out_path = out_dir / f"rec_pretrain_{split}.json"
    if out_path.exists():
        raise RuntimeError(f"REC pre-training for split {split} already exists (single run only)")
    objects = load_split_objects(split, out_dir, require_fusion=False)
    payload = objects["payload"]
    sample = payload.kappa_sample
    phi_t, atom_t = _phi_atom_tensors()
    p_sample = standardised_phi(
        phi_t[sample], payload.phi_mean, payload.phi_std, payload.phi_scale
    )
    d_s = prev.init_d_loc()[: prev.PHI_DIM, :].contiguous().to(device)
    d_s = nn.Parameter(d_s)
    optimizer = torch.optim.Adam([d_s], lr=REC_LR, weight_decay=REC_WD)
    rng = np.random.default_rng(REC_SAMPLER_SEED)
    n_sample = int(sample.numel())
    curve: list[dict[str, Any]] = []
    started = time.perf_counter()
    stopped_reason = "completed"
    for step in range(1, REC_STEPS + 1):
        rows = torch.as_tensor(rng.choice(n_sample, size=REC_BATCH, replace=True), dtype=torch.long, device=device)
        p = p_sample.to(device)[rows]
        alpha = structure_codes_ref("dict", d_s, p)
        dbar = F.normalize(d_s, dim=0, eps=1e-12)
        rec_p = alpha @ dbar.t()
        loss = ((p - rec_p).pow(2).sum(dim=1) / (p.pow(2).sum(dim=1) + 1e-12)).mean()
        if not math.isfinite(float(loss)):
            stopped_reason = f"non_finite_at_step_{step}"
            break
        optimizer.zero_grad()
        loss.backward()
        _ = float(torch.nn.utils.clip_grad_norm_([d_s], REC_CLIP))
        optimizer.step()
        if step == 1 or step % 50 == 0 or step == REC_STEPS:
            with torch.no_grad():
                alpha_all = structure_codes_ref("dict", d_s.detach(), p_sample.to(device))
                dbar_all = F.normalize(d_s.detach(), dim=0, eps=1e-12)
                rel = float(
                    ((p_sample.to(device) - alpha_all @ dbar_all.t()).pow(2).sum(dim=1)
                     / (p_sample.to(device).pow(2).sum(dim=1) + 1e-12)).mean()
                )
            curve.append({"step": step, "batch_loss": float(loss), "sample_rel_recon": rel})
            log(f"[rec {split}] step={step} batch={float(loss):.6f} sample_rel={rel:.6f}")
    d_s_last = d_s.detach().cpu().clone()
    with torch.no_grad():
        dbar_last = F.normalize(d_s_last, dim=0, eps=1e-12)
        alpha_last = structure_codes_ref("dict", d_s_last, p_sample)
        rel_last = float(
            ((p_sample - alpha_last @ dbar_last.t()).pow(2).sum(dim=1)
             / (p_sample.pow(2).sum(dim=1) + 1e-12)).mean()
        )
        # kappa_REC on the same frozen sample, J weights, fresh B frame
        d0 = prev.init_d_loc()
        b0 = d0[prev.PHI_DIM:, :].t().contiguous()
        e_rec_j = compose_root_codes_f(
            "dict", d_s_last, b0, phi_t, atom_t, payload.pair_ptr, payload.pair_t,
            payload.pair_a, payload.pair_wJ, sample, payload.phi_mean, payload.phi_std, payload.phi_scale,
        )
    fusion = objects["fusion_kappa"]
    r_star = float(fusion["r_star"]) if fusion is not None else float("nan")
    rms_rec = float(e_rec_j.pow(2).mean().sqrt())
    kappa_rec = r_star / rms_rec if rms_rec > 0 else float("nan")
    if stopped_reason != "completed" or not math.isfinite(kappa_rec) or kappa_rec <= 0:
        record = {
            "split": split, "stopped_reason": stopped_reason, "kappa_REC": None,
            "verdict": "REC_BLOCKED_non_finite_or_degenerate",
        }
        write_json(out_path, record)
        log(f"[rec {split}] BLOCKED: {stopped_reason}")
        return record
    record = {
        "split": split,
        "recipe": {
            "steps": REC_STEPS, "batch": REC_BATCH, "lr": REC_LR, "weight_decay": REC_WD,
            "clip": REC_CLIP, "sampler_seed": REC_SAMPLER_SEED,
            "sampler": "rng(20261010) choice(n_sample, 256, replace=True) per step",
            "sample": "the split's frozen <=8192 fit-root kappa sample",
            "encoding": "source-tied IHT (10 steps, top-8), eta detached power-iteration",
            "loss": "batch mean ||p - alpha @ Dbar.T||^2 / (||p||^2 + 1e-12)",
            "selection": "fixed last, no best epoch, no extra supervision/value matrix",
        },
        "stopped_reason": stopped_reason,
        "sample_sha256": array_sha256(sample.numpy(), np.int64),
        "n_roots": int(sample.numel()),
        "D_S_init_sha256": state_hash({"D_S": prev.init_d_loc()[: prev.PHI_DIM, :].contiguous()}),
        "D_S_last": d_s_last.numpy().astype(np.float32).tolist(),
        "sample_rel_recon_init": curve[0]["sample_rel_recon"] if curve else None,
        "sample_rel_recon_last": rel_last,
        "batch_curve": curve,
        "r_star": r_star,
        "e_REC_J_rms": rms_rec,
        "kappa_REC": float(kappa_rec),
        "labels_used": False,
        "dev_used": False,
        "seconds": float(time.perf_counter() - started),
    }
    write_json(out_path, record)
    log(
        f"[rec {split}] done rel {curve[0]['sample_rel_recon'] if curve else float('nan'):.6f}"
        f" -> {rel_last:.6f} kappa_REC={kappa_rec:.6f} ({record['seconds']:.1f}s)"
    )
    return record


# ---------------------------------------------------------------------------
# training (identical recipe for every formal arm; checkpoints at LOG_EPOCHS)
# ---------------------------------------------------------------------------


def _probe(model: nn.Module, arm: str, epoch: int, step: int, total_norm: float) -> dict[str, Any]:
    entry: dict[str, Any] = {"epoch": int(epoch), "step_in_epoch": int(step), "clip_total_norm": float(total_norm)}
    enc = model.local_tuple
    entry["W_loc_grad_norm"] = float(enc.W_loc.grad.norm()) if enc.W_loc.grad is not None else None
    entry["W_loc_norm"] = float(enc.W_loc.detach().norm())
    for key in ("D_loc_raw", "A_raw", "D_S_raw", "A_S_raw", "B_raw"):
        if hasattr(enc, key):
            param = getattr(enc, key)
            entry[f"{key}_grad_norm"] = float(param.grad.norm()) if param.grad is not None else None
            entry[f"{key}_norm"] = float(param.detach().norm())
    entry["health"] = {key: value for key, value in enc.last_stats.items() if not torch.is_tensor(value)}
    return entry


def train_arm(
    arm: str,
    *,
    out_dir: Path = RESULTS_DIR,
    device: torch.device,
    epochs: int = EPOCHS,
    max_steps: int | None = None,
    log: Any = print,
) -> dict[str, Any]:
    spec = arm_spec(arm)
    split = spec["split"]
    torch.set_num_threads(8 if device.type == "cpu" else 4)
    out_dir.mkdir(parents=True, exist_ok=True)
    objects = load_split_objects(split, out_dir)
    if spec["d_s_source"] == "rec" and not (out_dir / f"rec_pretrain_{split}.json").exists():
        raise RuntimeError(f"{arm} requires the split-{split} REC pre-training first")
    fold = objects["fold"]
    payload = objects["payload"]
    g = np.asarray(objects["targets"]["g"], np.float64)
    ell = np.asarray(objects["targets"]["ell"], np.float64)
    s = np.asarray(objects["targets"]["s"], np.float64)
    fit_idx = fold["fit_idx"]
    g_fit, ell_fit, s_fit = g[fit_idx], ell[fit_idx], s[fit_idx]
    if float(np.max(np.abs(g_fit - (ell_fit + s_fit)))) > 1e-12:
        raise RuntimeError("g != ell + s on fit rows")

    target_g = torch.as_tensor(g_fit, dtype=torch.float32)
    target_ell = torch.as_tensor(ell_fit, dtype=torch.float32, device=device)
    target_s = torch.as_tensor(s_fit, dtype=torch.float32, device=device)

    prep_meta, fit_data, dev_data = build_prepared_data(objects)
    schedule, schedule_hash = zw.build_schedule(len(fit_data), int(epochs), SEED + TRAIN_SHUFFLE_OFFSET)
    steps_expected = int(math.ceil(len(fit_data) / BATCH_SIZE)) * int(epochs)
    if len(fit_data) == N_FIT and steps_expected != STEPS_TOTAL:
        raise RuntimeError("step count does not match the frozen 15,120 for an 8,000-fit arm")
    if len(fit_data) == N_TRAIN and steps_expected != FULL_STEPS_TOTAL:
        raise RuntimeError("step count does not match the frozen 18,960 for a full-10k arm")

    seed_everything(SEED)
    model = build_arm_model(arm, objects)
    build_rng = torch.get_rng_state()
    model = model.to(device)
    seed_everything(SEED)
    train_rng = torch.get_rng_state()

    init_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=WEIGHT_DECAY
    )
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
            gid_stream.update(np.asarray(objects["targets"]["gid"][fit_idx[index_list]], np.int64).tobytes())
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

    enc = model.local_tuple
    kappa_value = float(enc.kappa)
    result: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "arm": arm,
        "arm_spec": spec,
        "optimised_loss": "L_g + 0.5*(L_ell + L_s)",
        "supervision": {
            "g": "g = y - c (split fit-only constants)",
            "ell": "(logP - MU_LOGP) / sigma_logP",
            "s": "g - ell (residual chemistry component)",
        },
        "seed": SEED, "epochs": int(epochs), "steps_done": int(steps_done),
        "steps_expected": steps_expected, "stopped_reason": stopped_reason,
        "lr": LR, "weight_decay": WEIGHT_DECAY, "grad_clip": GRAD_CLIP,
        "batch_size": BATCH_SIZE, "soup_epochs": members if max_steps is None else [],
        "n_fit": int(len(fit_data)), "n_dev": int(len(dev_data)),
        "kappa": kappa_value,
        "fold": {"split": split, "fit_idx_sha256": _hash_bytes(np.asarray(fold["fit_idx"], np.int64).tobytes()),
                 "dev_idx_sha256": _hash_bytes(np.asarray(fold["dev_idx"], np.int64).tobytes())},
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

    sums: dict[str, np.ndarray] = {}
    comps: dict[str, np.ndarray] = {}
    for state_name, state in (("init", init_state), ("last", last_state), ("raw_soup", soup_state)):
        replay = build_arm_model(arm, objects)
        replay.load_state_dict({key: value.to(torch.device("cpu")) for key, value in state.items()}, strict=True)
        replay = replay.to(device)
        fit_sum, fit_comp = src.evaluate_state_components(replay, fit_data, device)
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


# ---------------------------------------------------------------------------
# frozen dev evaluation (roster frozen before any dev prediction)
# ---------------------------------------------------------------------------


def _expected_steps_for(arm: str) -> int:
    spec = arm_spec(arm)
    return STEPS_TOTAL if spec["split"] != "FULL" else FULL_STEPS_TOTAL


def phase_freeze_eval_roster(
    arms: Sequence[str],
    *,
    out_dir: Path = RESULTS_DIR,
    log: Any = print,
) -> dict[str, Any]:
    """Freeze the roster (soups + fit biases + streams) BEFORE any dev score."""
    arms = list(arms)
    splits = {arm_spec(a)["split"] for a in arms}
    if len(splits) != 1:
        raise RuntimeError(f"evaluation roster must be single-split, got {splits}")
    split = splits.pop()
    metas = {}
    for arm in arms:
        meta = json.loads((out_dir / f"{arm}_meta.json").read_text())
        if meta.get("stopped_reason") != "completed" or int(meta["steps_done"]) != int(meta["steps_expected"]):
            raise RuntimeError(f"{arm} is not a completed trajectory")
        metas[arm] = meta
    streams = {arm: (metas[arm]["position_stream_sha256"], metas[arm]["global_gid_stream_sha256"]) for arm in arms}
    roster = {
        "protocol_version": PROTOCOL_VERSION,
        "split": split,
        "arms": arms,
        "main_endpoint": "dev G0(k=0) calibrated g-MAE; overall cal is the common gate; raw reported in parallel",
        "bias": {arm: float(metas[arm]["calibration_b_g_fit_only"]["raw_soup"]) for arm in arms},
        "soup_state_hashes": {arm: metas[arm]["raw_soup_state_hash"] for arm in arms},
        "schedule_sha256": {arm: metas[arm]["schedule_sha256"] for arm in arms},
        "position_streams_equal": bool(len({v[0] for v in streams.values()}) == 1),
        "global_gid_streams_equal": bool(len({v[1] for v in streams.values()}) == 1),
        "bootstrap": {"seed": BOOT_SEED, "n_boot": N_BOOT, "shared_indices_all_arms": True},
        "dev_scores_computed_before_freeze": False,
    }
    if not roster["position_streams_equal"] or not roster["global_gid_streams_equal"]:
        raise RuntimeError("the arms did not follow the identical position/gid stream")
    write_json(out_dir / f"eval_roster_{split}_{'_'.join(arms)}.json", roster)
    log(f"[freeze] roster frozen for {arms} (no dev prediction computed yet)")
    return roster


def paired_bootstrap_multi(
    err_by_arm: Mapping[str, Mapping[str, np.ndarray]],
    k_dev: np.ndarray,
    *,
    gain_specs: Sequence[tuple[str, str, str]],
    seed: int = BOOT_SEED,
    n_boot: int = N_BOOT,
) -> dict[str, Any]:
    """Paired bootstrap with ONE shared per-graph resample index set per draw,
    shared across every arm and every gain.  ``err_by_arm[arm][metric]`` is the
    per-graph dev error array for that metric, already G0-filtered for the
    ``G0_*`` metrics.  ``gain = MAE(minuend) - MAE(subtrahend)`` per triple."""
    metrics = ("G0_cal", "overall_cal", "G0_raw", "overall_raw")
    for arm, err_map in err_by_arm.items():
        for metric in metrics:
            if metric not in err_map:
                raise KeyError(f"arm {arm} missing error array for {metric}")
    sizes = {metric: int(next(iter(err_by_arm.values()))[metric].shape[0]) for metric in metrics}
    for arm, err_map in err_by_arm.items():
        for metric in metrics:
            if int(err_map[metric].shape[0]) != sizes[metric]:
                raise RuntimeError(f"arm {arm} metric {metric} length mismatch")
    point: dict[str, dict[str, float]] = {}
    for name, a, b in gain_specs:
        point[name] = {
            metric: _mae(np.abs(err_by_arm[a][metric])) - _mae(np.abs(err_by_arm[b][metric]))
            for metric in metrics
        }
    samples: dict[str, dict[str, np.ndarray]] = {
        name: {metric: np.empty(int(n_boot), np.float64) for metric in metrics}
        for name, _a, _b in gain_specs
    }
    rng = np.random.default_rng(int(seed))
    per_draw_identity_gap: list[float] = []
    has_interaction = all(n in point for n in ("G_J", "G_F", "C_D", "C_M", "I"))
    for _draw in range(int(n_boot)):
        idx = {
            metric: rng.choice(sizes[metric], size=sizes[metric], replace=True)
            for metric in metrics
        }
        resampled: dict[str, dict[str, float]] = {}
        for arm, err_map in err_by_arm.items():
            resampled[arm] = {
                metric: float(np.abs(err_map[metric])[idx[metric]].mean())
                for metric in metrics
            }
        for name, a, b in gain_specs:
            for metric in metrics:
                samples[name][metric][_draw] = resampled[a][metric] - resampled[b][metric]
        if has_interaction:
            for metric in metrics:
                left = samples["G_F"][metric][_draw] - samples["G_J"][metric][_draw]
                right = samples["C_D"][metric][_draw] - samples["C_M"][metric][_draw]
                per_draw_identity_gap.append(abs(left - right))
    out: dict[str, Any] = {
        "seed": int(seed), "n_boot": int(n_boot),
        "definition": "gain = MAE(minuend) - MAE(subtrahend); positive = subtrahend better",
        "shared_indices_per_draw": True,
    }
    for name, _a, _b in gain_specs:
        out[name] = {
            metric: {
                "point": point[name][metric],
                "ci95": [
                    float(np.percentile(samples[name][metric], 2.5)),
                    float(np.percentile(samples[name][metric], 97.5)),
                ],
            }
            for metric in metrics
        }
    if per_draw_identity_gap:
        out["interaction_identity_per_draw_max_abs"] = float(max(per_draw_identity_gap))
    return out


def _set_weight_key(enc: nn.Module, key: str) -> None:
    if hasattr(enc, "weight_key"):
        enc.weight_key = key
    if hasattr(enc, "switch_independent"):
        enc.switch_independent = key == "independent"


def run_interventions(
    arm: str,
    *,
    objects: Mapping[str, Any],
    out_dir: Path,
    device: torch.device,
    fit_data: Sequence[Any],
    dev_data: Sequence[Any],
    comps_dev: np.ndarray,
    bias: float,
    g_dev: np.ndarray,
    k_dev: np.ndarray,
    log: Any = print,
) -> dict[str, Any]:
    """Three frozen operator checks on the arm's native soup (native fit bias)."""
    state = torch.load(out_dir / f"{arm}_raw_soup_state.pt", map_location="cpu", weights_only=True)
    model = build_arm_model(arm, objects)
    model.load_state_dict({key: value.to(torch.device("cpu")) for key, value in state.items()}, strict=True)
    model = model.to(device)
    model.eval()
    baseline_sum, _ = src.evaluate_state_components(model, dev_data, device)
    fit_dummy = torch.zeros(len(fit_data), dtype=torch.float32)
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
            _set_weight_key(enc, "joint")
        elif mode == "mean_root_code":
            enc.ablate = False
            enc.mean_replace = mean_code.detach().to(device=device, dtype=torch.float32)
            _set_weight_key(enc, "joint")
        elif mode == "J_to_I":
            enc.ablate = False
            enc.mean_replace = None
            _set_weight_key(enc, "independent")
        else:
            raise ValueError(mode)
        variant_sum, variant_comp = src.evaluate_state_components(model, dev_data, device)
        enc.ablate = False
        enc.mean_replace = None
        _set_weight_key(enc, "joint")
        return {
            "mean_abs_delta_pred": float(np.mean(np.abs(variant_sum - baseline_sum))),
            "g_raw_mae": _mae(g_dev - variant_sum),
            "g_cal_mae": _mae(g_dev - (variant_sum + float(bias))),
            "G0_cal_mae": _mae(g_dev[k_dev == 0] - (variant_sum[k_dev == 0] + float(bias))),
            "delta_ell_mean_output": float(np.mean(np.abs(variant_comp[:, 0] - comps_dev[:, 0]))),
            "delta_s_mean_output": float(np.mean(np.abs(variant_comp[:, 1] - comps_dev[:, 1]))),
        }

    results = {
        "baseline": {
            "g_raw_mae": _mae(g_dev - baseline_sum),
            "g_cal_mae": _mae(g_dev - (baseline_sum + float(bias))),
            "G0_cal_mae": _mae(g_dev[k_dev == 0] - (baseline_sum[k_dev == 0] + float(bias))),
        },
        "zero_local_injection": run_variant("zero_injection"),
        "mean_root_code": run_variant("mean_root_code"),
        "J_to_I_switch": run_variant("J_to_I"),
        "mean_code_source": "fit roots, label-free, per arm frozen soup",
        "note": "sensitivity-only: large zeroing losses include synergy destruction, not information share",
    }
    log(
        f"[interventions {arm}] zero/mean/J->I G0_cal="
        f"{results['zero_local_injection']['G0_cal_mae']:.6f}/"
        f"{results['mean_root_code']['G0_cal_mae']:.6f}/"
        f"{results['J_to_I_switch']['G0_cal_mae']:.6f}"
    )
    return results


def check_clear_signal(gain: Mapping[str, Any]) -> bool:
    """The frozen 'clear performance signal': G0 cal >= .003, overall cal >= .003,
    G0 cal CI lower > 0, both raw gains > 0."""
    return bool(
        gain["G0_cal"]["point"] >= GATE_DELTA
        and gain["overall_cal"]["point"] >= GATE_DELTA
        and gain["G0_cal"]["ci95"][0] > 0.0
        and gain["G0_raw"]["point"] > 0.0
        and gain["overall_raw"]["point"] > 0.0
    )


def phase_dev_eval(
    arms: Sequence[str],
    *,
    contrasts: Sequence[tuple[str, str, str]] | None = None,
    out_dir: Path = RESULTS_DIR,
    device: torch.device | None = None,
    log: Any = print,
) -> dict[str, Any]:
    """Frozen dev evaluation of completed arms (single split, roster first)."""
    torch.set_num_threads(8)
    device = device or torch.device("cpu")
    arms = list(arms)
    splits = {arm_spec(a)["split"] for a in arms}
    if len(splits) != 1:
        raise RuntimeError(f"dev-eval arms must share one split, got {splits}")
    split = splits.pop()
    roster = phase_freeze_eval_roster(arms, out_dir=out_dir, log=log)
    objects = load_split_objects(split, out_dir)
    fold = objects["fold"]
    fit_idx, dev_idx = fold["fit_idx"], fold["dev_idx"]
    t = objects["targets"]
    g, k, ell, s = (
        np.asarray(t[key], np.float64 if key != "k" else np.int64) for key in ("g", "k", "ell", "s")
    )
    g_fit, g_dev = g[fit_idx], g[dev_idx]
    k_fit, k_dev = k[fit_idx], k[dev_idx]
    ell_fit, ell_dev = ell[fit_idx], ell[dev_idx]
    s_fit, s_dev = s[fit_idx], s[dev_idx]
    payload = objects["payload"]
    bias = {arm: float(roster["bias"][arm]) for arm in arms}

    prep_meta, fit_data, dev_data = build_prepared_data(objects)
    sums: dict[str, dict[str, np.ndarray]] = {}
    comps: dict[str, dict[str, np.ndarray]] = {}
    for arm in arms:
        state = torch.load(out_dir / f"{arm}_raw_soup_state.pt", map_location="cpu", weights_only=True)
        if state_hash(state) != roster["soup_state_hashes"][arm]:
            raise RuntimeError(f"{arm} soup state hash mismatch")
        replay = build_arm_model(arm, objects)
        replay.load_state_dict({key: value.to(torch.device("cpu")) for key, value in state.items()}, strict=True)
        replay = replay.to(device)
        fit_sum, fit_comp = src.evaluate_state_components(replay, fit_data, device)
        dev_sum, dev_comp = src.evaluate_state_components(replay, dev_data, device)
        sums[arm] = {"fit": fit_sum, "dev": dev_sum}
        comps[arm] = {"fit": fit_comp, "dev": dev_comp}
        with np.load(out_dir / f"{arm}_fit_predictions.npz") as z:
            fit_saved = np.asarray(z["raw_soup_fit"], np.float64)
        if float(np.max(np.abs(fit_sum - fit_saved))) > REPLAY_TOL:
            raise RuntimeError(f"{arm} fit replay mismatch")
        np.savez_compressed(
            out_dir / f"{arm}_dev_predictions.npz",
            **{key: value.astype(np.float32) for key, value in sums[arm].items()},
        )
        np.savez_compressed(
            out_dir / f"{arm}_dev_component_predictions.npz",
            **{key: value.astype(np.float32) for key, value in comps[arm].items()},
        )

    g0_dev = k_dev == 0
    g0_fit = k_fit == 0
    arm_rows: dict[str, Any] = {}
    err_by_arm: dict[str, dict[str, np.ndarray]] = {}
    for arm in arms:
        raw = {"fit": sums[arm]["fit"], "dev": sums[arm]["dev"]}
        cal = {key: raw[key] + bias[arm] for key in raw}
        err_fit = g_fit - cal["fit"]
        err_dev_cal = g_dev - cal["dev"]
        err_dev_raw = g_dev - raw["dev"]
        err_by_arm[arm] = {
            "G0_cal": err_dev_cal[g0_dev],
            "overall_cal": err_dev_cal,
            "G0_raw": err_dev_raw[g0_dev],
            "overall_raw": err_dev_raw,
        }
        arm_rows[arm] = {
            "bias": bias[arm],
            "fit_overall_cal_mae": _mae(err_fit),
            "dev_overall_raw_mae": _mae(err_dev_raw),
            "dev_overall_cal_mae": _mae(err_dev_cal),
            "dev_G0_raw_mae": _mae(err_dev_raw[g0_dev]),
            "dev_G0_cal_mae": _mae(err_dev_cal[g0_dev]),
            "gap_overall_cal": _mae(err_dev_cal) - _mae(err_fit),
            "dev_groups_cal": zw._metric_table(err_dev_cal, k_dev),
            "dev_groups_raw": zw._metric_table(err_dev_raw, k_dev),
            "kappa": float(build_arm_model(arm, objects).local_tuple.kappa),
        }

    # contrasts (frozen defaults; explicit triples override)
    if contrasts is None:
        if set(arms) == set(STAGE1_ARMS):
            contrasts = [
                ("G_J", "J_M", "J_D"),
                ("G_F", "F_M", "F_D"),
                ("C_D", "J_D", "F_D"),
                ("C_M", "J_M", "F_M"),
            ]
        else:
            contrasts = [(f"{a}_vs_J_M", "J_M", a) for a in arms if a != "J_M"]
    contrasts = [tuple(c) for c in contrasts]
    for name, a, b in contrasts:
        if a not in err_by_arm or b not in err_by_arm:
            raise RuntimeError(f"contrast {name} references arms not in this evaluation: {a}, {b}")
    gains = paired_bootstrap_multi(err_by_arm, k_dev, gain_specs=contrasts)
    # derived interaction I = G_F - G_J = C_D - C_M (stage-1 roster only)
    if set(arms) == set(STAGE1_ARMS):
        derived_i: dict[str, Any] = {}
        for metric in ("G0_cal", "overall_cal", "G0_raw", "overall_raw"):
            gf, gj = gains["G_F"][metric], gains["G_J"][metric]
            cd, cm = gains["C_D"][metric], gains["C_M"][metric]
            derived_i[metric] = {
                "point": gf["point"] - gj["point"],
                "point_via_C": cd["point"] - cm["point"],
                "ci95": [
                    gf["ci95"][0] - gj["ci95"][1],
                    gf["ci95"][1] - gj["ci95"][0],
                ],
            }
        gains["I_derived"] = derived_i

    # contribution identities
    identity: dict[str, Any] = {}
    for arm in arms:
        table = arm_rows[arm]["dev_groups_cal"]
        contrib = sum(table[name]["contribution"] for name in GROUP_NAMES if table[name]["n"] > 0)
        identity[f"{arm}_contribution_sum_minus_overall"] = float(contrib - table["mae"])

    # component diagnostics (raw, never separately calibrated)
    def cancellation(hat_e, tgt_e, hat_ss, tgt_ss):
        e_ell, e_s = hat_e - tgt_e, hat_ss - tgt_ss
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
    for arm in arms:
        hat_ell, hat_s = comps[arm]["fit"][:, 0], comps[arm]["fit"][:, 1]
        hat_ell_dev, hat_s_dev = comps[arm]["dev"][:, 0], comps[arm]["dev"][:, 1]
        e_ell_dev, e_s_dev = hat_ell_dev - ell_dev, hat_s_dev - s_dev
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
                (g_dev - sums[arm]["dev"]) + (e_ell_dev + e_s_dev)))),
        }

    # sensitivity: drop the common reference's worst dev row once
    reference = "J_M" if "J_M" in arm_rows else arms[0]
    worst = int(np.argmax(np.abs(err_by_arm[reference]["overall_cal"])))
    keep = np.ones(k_dev.size, dtype=bool)
    keep[worst] = False
    keep_g0 = keep[g0_dev]
    sensitivity = {
        "reference": reference,
        "dropped_dev_position": worst,
        "dropped_in_G0": bool(k_dev[worst] == 0),
    }
    for name, a, b in contrasts:
        sensitivity[name] = {
            metric: _mae(np.abs(err_by_arm[a][metric])[keep_g0 if metric.startswith("G0") else keep])
            - _mae(np.abs(err_by_arm[b][metric])[keep_g0 if metric.startswith("G0") else keep])
            for metric in ("G0_cal", "overall_cal", "G0_raw", "overall_raw")
        }

    # interventions on each arm's native soup (native fit bias)
    interventions = {
        arm: run_interventions(
            arm, objects=objects, out_dir=out_dir, device=device,
            fit_data=fit_data, dev_data=dev_data, comps_dev=comps[arm]["dev"], bias=bias[arm],
            g_dev=g_dev, k_dev=k_dev, log=log,
        )
        for arm in arms
    }

    # candidate gates vs the contemporaneous J_M reference (exploratory)
    gate_checks: dict[str, Any] = {}
    if "J_M" in arm_rows:
        for arm in arms:
            if arm == "J_M":
                continue
            key = f"{arm}_vs_J_M"
            if key in gains:
                gate_checks[arm] = {
                    "clear_signal": check_clear_signal(gains[key]),
                    "confirm_purchase_minimum": bool(
                        gains[key]["G0_cal"]["point"] >= CONFIRM_G0_CAL
                        and gains[key]["overall_cal"]["point"] >= CONFIRM_OVERALL_CAL
                        and gains[key]["G0_raw"]["point"] > 0.0
                        and gains[key]["overall_raw"]["point"] > 0.0
                    ),
                }

    result = {
        "protocol_version": PROTOCOL_VERSION,
        "split": split,
        "roster": {"arms": arms, "bias": bias},
        "main_table": {arm: {
            key: value for key, value in arm_rows[arm].items()
            if key not in ("dev_groups_cal", "dev_groups_raw")
        } for arm in arms},
        "dev_groups_cal": {arm: arm_rows[arm]["dev_groups_cal"] for arm in arms},
        "dev_groups_raw": {arm: arm_rows[arm]["dev_groups_raw"] for arm in arms},
        "k_dev_counts": {name: int(mask.sum()) for name, mask in group_masks(k_dev).items()},
        "contrasts": {"definition": "gain = MAE(minuend) - MAE(subtrahend); positive = subtrahend better",
                      "specs": [list(c) for c in contrasts]},
        "gains": gains,
        "contribution_identities": identity,
        "sensitivity_drop_reference_worst_row": sensitivity,
        "component_diagnostics": component_diag,
        "interventions": interventions,
        "gate_checks_vs_J_M": gate_checks,
        "internal_dev_note": (
            "internal dev g-MAE on the round's fold; NOT official-valid y-MAE, "
            "not a deployment threshold, not SOTA-comparable"
        ),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    tag = f"{split}_{'_'.join(arms)}"
    write_json(out_dir / f"dev_eval_{tag}.json", result)
    main_line = " ".join(f"{a} G0cal={arm_rows[a]['dev_G0_cal_mae']:.6f}" for a in arms)
    log(f"[dev-eval {split}] {main_line}")
    for name, _a, _b in contrasts:
        entry = gains[name]["G0_cal"]
        log(
            f"[dev-eval] {name} G0_cal={entry['point']:.6f} "
            f"CI=[{entry['ci95'][0]:.6f}, {entry['ci95'][1]:.6f}]"
        )
    return result


# ---------------------------------------------------------------------------
# mechanism health (offline from saved checkpoints; fit-only, label-free)
# ---------------------------------------------------------------------------


def _rel_recon_structure(d_s: torch.Tensor, payload: prev.TuplePayload, device: torch.device) -> float | None:
    """Relative phi65 reconstruction of the *structure* code (dict arms only)."""
    phi_t, _atom_t = _phi_atom_tensors()
    sample = payload.kappa_sample
    with torch.no_grad():
        p = standardised_phi(phi_t[sample], payload.phi_mean, payload.phi_std, payload.phi_scale).to(device)
        dbar = F.normalize(d_s.to(device), dim=0, eps=1e-12)
        alpha = structure_codes_ref("dict", d_s.to(device), p)
        rel = float(
            ((p - alpha @ dbar.t()).pow(2).sum(dim=1) / (p.pow(2).sum(dim=1) + 1e-12)).mean()
        )
    return rel


def phase_mechanism_health(
    arms: Sequence[str],
    *,
    out_dir: Path = RESULTS_DIR,
    device: torch.device | None = None,
    log: Any = print,
) -> dict[str, Any]:
    torch.set_num_threads(8)
    device = device or torch.device("cpu")
    arms = list(arms)
    splits = {arm_spec(a)["split"] for a in arms}
    if len(splits) != 1:
        raise RuntimeError("health arms must share one split")
    objects = load_split_objects(splits.pop(), out_dir)
    payload = objects["payload"]
    _, fit_data, _dev = build_prepared_data(objects)
    phi_t, atom_t = _phi_atom_tensors()
    sample = payload.kappa_sample
    d0 = prev.init_d_loc()
    results: dict[str, Any] = {}
    n_probe = 512
    for arm in arms:
        spec = arm_spec(arm)
        init_state = torch.load(out_dir / f"{arm}_init_state.pt", map_location="cpu", weights_only=True)
        arm_health: dict[str, Any] = {"arm_spec": spec}
        for epoch in LOG_EPOCHS:
            path = out_dir / (f"{arm}_init_state.pt" if epoch == 1 else f"{arm}_epoch{epoch}_state.pt")
            state = torch.load(path, map_location="cpu", weights_only=True)
            model = build_arm_model(arm, objects)
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
                local: dict[str, Any] = {}
                if hasattr(enc, "D_loc_raw"):
                    d_now = enc.D_loc_raw.detach().cpu()
                    d_init = init_state["local_tuple.D_loc_raw"].cpu()
                    local = {
                        "raw_drift_from_init": float((d_now - d_init).norm()),
                        "raw_norm": float(d_now.norm()),
                        "init_norm": float(d_init.norm()),
                        "col_normalized_same_index_change": float(
                            (F.normalize(d_now, dim=0) - F.normalize(d_init, dim=0)).norm()
                        ),
                        # the joint 125x64 dictionary reconstructs 125-D tuples, not
                        # phi65: a different object, not computed here (the F-arm
                        # structure dictionaries report rel_recon_structure)
                        "rel_recon_structure": None,
                    }
                elif hasattr(enc, "A_raw"):
                    a_now = enc.A_raw.detach().cpu()
                    a_init = init_state["local_tuple.A_raw"].cpu()
                    local = {
                        "raw_drift_from_init": float((a_now - a_init).norm()),
                        "raw_norm": float(a_now.norm()),
                        "init_norm": float(a_init.norm()),
                    }
                else:
                    s_key = "D_S_raw" if hasattr(enc, "D_S_raw") else "A_S_raw"
                    s_now = getattr(enc, s_key).detach().cpu()
                    s_init = init_state[f"local_tuple.{s_key}"].cpu()
                    local = {
                        "raw_drift_from_init": float((s_now - s_init).norm()),
                        "raw_norm": float(s_now.norm()),
                        "init_norm": float(s_init.norm()),
                        "col_normalized_same_index_change": float(
                            (F.normalize(s_now, dim=0) - F.normalize(s_init, dim=0)).norm()
                        ) if s_key == "D_S_raw" else float(
                            (F.normalize(s_now, dim=1) - F.normalize(s_init, dim=1)).norm()
                        ),
                        "rel_recon_structure": (
                            _rel_recon_structure(s_now, payload, device) if s_key == "D_S_raw" else None
                        ),
                        "B_raw_drift_from_init": float(
                            (enc.B_raw.detach().cpu() - init_state["local_tuple.B_raw"].cpu()).norm()
                        ),
                        "B_row_normalized_change": float(
                            (F.normalize(enc.B_raw.detach().cpu(), dim=1)
                             - F.normalize(init_state["local_tuple.B_raw"].cpu(), dim=1)).norm()
                        ),
                    }
                contribution = F.linear(codes * enc.kappa, enc.W_loc)
                arm_health[f"epoch{epoch}"] = {
                    "batch_stats": stats,
                    "local_tensor": local,
                    "root_code_rms": float(codes.pow(2).mean().sqrt()),
                    "root_code_effective_rank": int((std > 1e-8).sum()),
                    "root_code_dead_dims": int((std < 1e-8).sum()),
                    "root_code_all_zero": bool(float(codes.abs().sum()) == 0.0),
                    "W_loc_norm": float(enc.W_loc.norm()),
                    "injection_rms": float(contribution.pow(2).mean().sqrt()),
                }
        probes = json.loads((out_dir / f"{arm}_probe_log.json").read_text())
        arm_health["probe_task_grads"] = [
            {key: value for key, value in p.items() if "grad" in key or key in ("epoch",)}
            for p in probes
        ]
        results[arm] = arm_health
        ep240 = arm_health["epoch240"]
        log(
            f"[health {arm}] drift={ep240['local_tensor'].get('raw_drift_from_init', float('nan')):.3f} "
            f"structnnz={ep240['batch_stats'].get('structure_code_nnz', float('nan'))} "
            f"atoms={ep240['batch_stats'].get('structure_atoms_used', '-')} "
            f"injRMS={ep240['injection_rms']:.4g}"
        )
    has_b = any(a.startswith("B_") for a in arms)
    write_json(out_dir / ("mechanism_health_B.json" if has_b else "mechanism_health.json"), results)
    return results


# ---------------------------------------------------------------------------
# Stage 3 purchase: refit every split-B statistic on the B fit rows
# ---------------------------------------------------------------------------


def phase_build_b_objects(*, out_dir: Path = RESULTS_DIR, log: Any = print) -> dict[str, Any]:
    """Build every split-B fitted statistic from the frozen fold B (Stage 3).

    Same fitting recipes as the source round's A objects, on the B fit rows
    only: target constants, phi scaler, kappa_D (same sampling seed), prep
    standardizers, kappa_M, then the F-arm kappa match.  No warm start from any
    A object; no dev labels enter any fit.
    """
    torch.set_num_threads(8)
    fold = load_fold_b(out_dir)
    t0 = time.perf_counter()
    targets = src.build_new_targets(fold)
    t = targets["arrays"]
    np.savez_compressed(
        out_dir / "B_targets.npz",
        y=t["y"], c=t["c"], g=t["g"], k=t["k"], ell=t["ell"], s=t["s"], gid=targets["gid"],
        constants=np.asarray([targets["constants"][name] for name in
                              ("sigma_logP", "sigma_SA", "mu_SA", "sigma_cycle", "mu_cycle")], np.float64),
        constant_names=np.asarray(["sigma_logP", "sigma_SA", "mu_SA", "sigma_cycle", "mu_cycle"]),
        mu_logP=np.asarray(MU_LOGP, np.float64),
    )
    payload = src.build_new_payload(fold)
    np.savez_compressed(out_dir / "B_tuple_payload.npz", **payload["arrays"])
    kappa_m = src.compute_kappa_M_new(payload["arrays"])
    kappa_record = dict(kappa_m)
    kappa_record["kappa_D"] = float(payload["checks"]["kappa_D"]["value"])
    kappa_record["definition"] = (
        "kappa_D via the source recipe on the B fit-root sample (seed 20261004, <=8192); "
        "kappa_M = r_D/r_M on the same B sample"
    )
    write_json(out_dir / "B_kappa.json", kappa_record)
    train_data = zftd.load_train_only()
    prep_meta = zw.apply_new_fit_prep(train_data, np.load(zfr.PREP_BLOB, allow_pickle=False), fold["fit_idx"])
    np.savez_compressed(
        out_dir / "B_prep.npz",
        patch_fit_mean=prep_meta["patch_fit_mean"], patch_fit_scale=prep_meta["patch_fit_scale"],
        ctx_fit_mean=prep_meta["ctx_fit_mean"], ctx_fit_scale=prep_meta["ctx_fit_scale"],
        anchor_fit_mean=prep_meta["anchor_fit_mean"], anchor_fit_scale=prep_meta["anchor_fit_scale"],
        topo_fit_mean=prep_meta["topo_fit_mean"], topo_fit_scale=prep_meta["topo_fit_scale"],
    )
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "fold": {"definition": fold["definition"],
                 "fit_idx_sha256": _hash_bytes(fold["fit_idx"].tobytes()),
                 "dev_idx_sha256": _hash_bytes(fold["dev_idx"].tobytes())},
        "constants": targets["constants"],
        "target_checks": targets["checks"],
        "payload_checks": payload["checks"],
        "kappa_M": kappa_m,
        "kappa_D": kappa_record["kappa_D"],
        "prep": {"fitted_on": "split B fit rows only",
                 "fit_root_rows": prep_meta["fit_root_rows"],
                 "n_fit_molecules": prep_meta["n_fit_molecules"]},
        "artifacts": {
            name: {"sha256": file_sha256(out_dir / name)}
            for name in ("B_fold.npz", "B_targets.npz", "B_tuple_payload.npz", "B_prep.npz")
        },
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "seconds": float(time.perf_counter() - t0),
    }
    write_json(out_dir / "B_objects_manifest.json", manifest)
    compute_fusion_kappa("B", out_dir=out_dir, log=log)
    log(
        f"[B-objects] built in {manifest['seconds']:.1f}s kappa_D_B={kappa_record['kappa_D']:.6f} "
        f"kappa_M_B={kappa_m['kappa_M']:.6f}"
    )
    return manifest


# ---------------------------------------------------------------------------
# Stage 4 (purchase only): full-10k objects, cycle-module reuse, one valid read
# ---------------------------------------------------------------------------

ZFT_RESULTS_DIR = zjd.TRACK_ROOT / "results" / "zinc_component_supervision_fulltrain_confirmation_seed0_v1"
CYCLE_RESULTS_DIR = zjd.TRACK_ROOT / "results" / "zinc_cycle_level_transfer_terminal_test_seed0_v1"
PROTO_RESULTS_DIR = zjd.TRACK_ROOT / "results" / "zinc_cycle_prototype_transfer_cpu_v1"
VALID_PROCESSED = zjd.REPO_ROOT / "data/ZINC/subset/processed/val.pt"
CYCLE_BUILD_SEED = 0
CYCLE_TOPOLOGY_IN = 25
CYCLE_HIDDEN = (64, 32)


def phase_build_full_objects(*, out_dir: Path = RESULTS_DIR, log: Any = print) -> dict[str, Any]:
    """Build the full-10k fitted objects with the frozen fulltrain recipes
    (target constants, tuple payload + phi scaler + kappa sample, prep
    standardizers, kappa_M) — every statistic refit on all 10,000 official-train
    rows; then the F-arm kappa match.  No valid/test object is touched."""
    torch.set_num_threads(8)
    t0 = time.perf_counter()
    rows, gid, label_checks = zft.load_train_only_raw_rows()
    targets = zft.refit_fulltrain_targets(rows)
    constants = targets["constants"]
    components = zft.build_component_targets(rows, constants, targets["g"])
    np.savez_compressed(
        out_dir / "FULL_targets.npz",
        y=targets["y"], c=targets["c"], g=targets["g"], k=targets["k"], gid=gid,
        ell=components["ell"], s=components["s"],
        constants=np.asarray(
            [constants[name] for name in ("sigma_logP", "sigma_SA", "mu_SA", "sigma_cycle", "mu_cycle")],
            np.float64,
        ),
        constant_names=np.asarray(["sigma_logP", "sigma_SA", "mu_SA", "sigma_cycle", "mu_cycle"]),
        mu_logP=np.asarray(MU_LOGP, np.float64),
    )
    payload = zft.build_fulltrain_payload()
    np.savez_compressed(out_dir / "FULL_tuple_payload.npz", **payload["arrays"])
    kappa_m = src.compute_kappa_M_new(payload["arrays"])
    kappa_record = dict(kappa_m)
    kappa_record["kappa_D"] = float(payload["checks"]["kappa"]["value"])
    kappa_record["definition"] = (
        "kappa_D via the source recipe on the full-10k fit-root sample (seed 20261004, <=8192); "
        "kappa_M = r_D/r_M on the same full-10k sample"
    )
    write_json(out_dir / "FULL_kappa.json", kappa_record)
    prep_meta = zft.build_fulltrain_prep()
    np.savez_compressed(
        out_dir / "FULL_prep.npz",
        patch_fit_mean=prep_meta["patch_fit_mean"], patch_fit_scale=prep_meta["patch_fit_scale"],
        ctx_fit_mean=prep_meta["ctx_fit_mean"], ctx_fit_scale=prep_meta["ctx_fit_scale"],
        anchor_fit_mean=prep_meta["anchor_fit_mean"], anchor_fit_scale=prep_meta["anchor_fit_scale"],
        topo_fit_mean=prep_meta["topo_fit_mean"], topo_fit_scale=prep_meta["topo_fit_scale"],
    )
    np.savez_compressed(
        out_dir / "FULL_fold.npz",
        fit_idx=np.arange(N_TRAIN, dtype=np.int64),
        dev_idx=np.zeros(0, dtype=np.int64),
    )
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "fold": {
            "definition": "full 10,000 official-train rows are fit; no internal dev",
            "fit_idx_sha256": _hash_bytes(np.arange(N_TRAIN, dtype=np.int64).tobytes()),
            "dev_idx_sha256": _hash_bytes(np.zeros(0, dtype=np.int64).tobytes()),
        },
        "constants": constants,
        "target_checks": targets["checks"],
        "component_checks": components["checks"],
        "label_checks": label_checks,
        "payload_checks": payload["checks"],
        "kappa": kappa_record,
        "prep": {"fitted_on": "all 10,000 official-train rows",
                 "fit_root_rows": prep_meta["fit_root_rows"]},
        "recipes": {
            "targets": "zft.refit_fulltrain_targets + zft.build_component_targets (frozen fulltrain recipes)",
            "payload": "zft.build_fulltrain_payload (structure reused byte-for-byte; scaler/kappa refit on 10k)",
            "prep": "zft.build_fulltrain_prep (all 10,000 fit rows)",
            "kappa_M": "src.compute_kappa_M_new on the full-10k payload",
        },
        "artifacts": {
            name: {"sha256": file_sha256(out_dir / name)}
            for name in ("FULL_fold.npz", "FULL_targets.npz", "FULL_tuple_payload.npz", "FULL_prep.npz")
        },
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "seconds": float(time.perf_counter() - t0),
    }
    write_json(out_dir / "FULL_objects_manifest.json", manifest)
    compute_fusion_kappa("FULL", out_dir=out_dir, log=log)
    log(
        f"[FULL-objects] built in {manifest['seconds']:.1f}s kappa_D={kappa_record['kappa_D']:.6f} "
        f"kappa_M={kappa_m['kappa_M']:.6f}"
    )
    return manifest


def load_raw_valid_graphs() -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Raw official-valid PyG graphs (positional; the single authorized read)."""
    loaded = torch.load(VALID_PROCESSED, map_location="cpu", weights_only=False)
    data, slices, _cls = loaded
    x = data["x"].reshape(-1).numpy().astype(np.int64, copy=False)
    edge_index = data["edge_index"].numpy().astype(np.int64, copy=False)
    edge_attr = data["edge_attr"].reshape(-1).numpy().astype(np.int64, copy=False)
    node_slices = slices["x"].numpy().astype(np.int64)
    edge_slices = slices["edge_index"].numpy().astype(np.int64)
    graphs: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for index in range(len(node_slices) - 1):
        lo, hi = int(node_slices[index]), int(node_slices[index + 1])
        elo, ehi = int(edge_slices[index]), int(edge_slices[index + 1])
        graphs.append((x[lo:hi].copy(), edge_index[:, elo:ehi].copy(), edge_attr[elo:ehi].copy()))
    if len(graphs) != 1000:
        raise RuntimeError(f"raw valid graph count {len(graphs)} != 1000")
    return graphs


def build_valid_structure() -> dict[str, np.ndarray]:
    """Incidence structure of the 1,000 official-valid graphs (label-free).

    The historical fulltrain valid read indexed the *train* payload's
    root_base/pair_ptr with valid local_mol_id (0..999), attaching unrelated
    train pair structures to valid roots (ERRATA; see valid_structure_check).
    This builder gives each valid root its own adjacency; the train-frozen
    phi scaler / kappa of the FULL payload are kept.
    """
    env = torch.load(
        zjd.TRACK_ROOT / "results/e2e_dictenv_p1/cache/env_valid.pt",
        map_location="cpu", weights_only=False,
    )
    phi = env["phi"].numpy()
    atom = env["atom"].numpy().astype(np.int64)
    node_sizes = env["node_sizes"].numpy().astype(np.int64)
    graphs = load_raw_valid_graphs()
    if int(node_sizes.sum()) != int(phi.shape[0]):
        raise RuntimeError("env_valid node-size mismatch")
    for index, (atom_types, _ei, _ea) in enumerate(graphs):
        if int(atom_types.shape[0]) != int(node_sizes[index]):
            raise RuntimeError(f"valid graph {index} node count mismatch")
        if not np.array_equal(
            atom[int(node_sizes[:index].sum()):int(node_sizes[: index + 1].sum())], atom_types
        ):
            raise RuntimeError(f"valid graph {index} root atom order mismatch")
    root_base = np.concatenate([[0], np.cumsum(node_sizes)]).astype(np.int64)
    total_roots = int(node_sizes.sum())
    pair_t: list[np.ndarray] = []
    pair_a: list[np.ndarray] = []
    pair_j: list[np.ndarray] = []
    pair_i: list[np.ndarray] = []
    counts: list[int] = []
    for index in range(1000):
        atom_types, edge_index, edge_attr = graphs[index]
        count, _stats = prev.molecule_incidence(atom_types, edge_index, edge_attr)
        _d, n_t, n_a, w_joint, w_ind = prev.root_weights(count)
        support = prev._support_union(count, n_t, n_a)
        base = int(root_base[index])
        for local in range(len(n_t)):
            rows = np.nonzero(support[local].reshape(-1))[0]
            if rows.size == 0:
                counts.append(0)
                continue
            pair_t.append((rows // prev.ATOM_CATEGORIES).astype(np.int64))
            pair_a.append((rows % prev.ATOM_CATEGORIES).astype(np.int64))
            pair_j.append(w_joint[local].reshape(-1)[rows].astype(np.float32))
            pair_i.append(w_ind[local].reshape(-1)[rows].astype(np.float32))
            counts.append(int(rows.size))
    return {
        "root_base": root_base,
        "pair_ptr": np.concatenate([[0], np.cumsum(np.asarray(counts, dtype=np.int64))]).astype(np.int64),
        "pair_t": np.concatenate(pair_t).astype(np.int64),
        "pair_a": np.concatenate(pair_a).astype(np.int64),
        "pair_wJ": np.concatenate(pair_j).astype(np.float32),
        "pair_wI": np.concatenate(pair_i).astype(np.float32),
        "root_atom": atom.astype(np.int64),
        "node_sizes": node_sizes,
    }


def load_valid_data_full_prep(objects: Mapping[str, Any]) -> tuple[list[Any], dict[str, Any]]:
    """Official-valid encoded cache + env cache, frozen FULL prep, own ids."""
    from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p1 as p1run
    from tracks.ksvd.experiments.luyin16 import zinc_static_dictionary_pair as sdp

    valid = list(torch.load(sdp.CACHE_DIR / "encoded_valid.pt", map_location="cpu", weights_only=False))
    p1run.attach_env(valid, "valid")
    blob = np.load(zfr.PREP_BLOB, allow_pickle=False)
    prep = objects["prep"]
    patch_all = zjd.Std(blob["patch_all_mean"], blob["patch_all_scale"])
    ctx_all = zjd.Std(blob["ctx_all_mean"], blob["ctx_all_scale"])
    anchor_all = zjd.Std(blob["anchor_all_mean"], blob["anchor_all_scale"])
    topo_all = zjd.Std(blob["topo_all_mean"], blob["topo_all_scale"])
    patch_fit = zjd.Std(prep["patch_fit_mean"], prep["patch_fit_scale"])
    ctx_fit = zjd.Std(prep["ctx_fit_mean"], prep["ctx_fit_scale"])
    anchor_fit = zjd.Std(prep["anchor_fit_mean"], prep["anchor_fit_scale"])
    topo_fit = zjd.Std(prep["topo_fit_mean"], prep["topo_fit_scale"])
    p, pc = zjd._stack(valid, "patch_cont")
    c, cc = zjd._stack(valid, "global_context")
    a, ac = zjd._stack(valid, "anchor")
    t, tc = zjd._stack(valid, "topology_features")
    zjd._unstack(patch_fit.transform(patch_all.inverse(p)), pc, "patch_cont", valid)
    zjd._unstack(ctx_fit.transform(ctx_all.inverse(c)), cc, "global_context", valid)
    zjd._unstack(anchor_fit.transform(anchor_all.inverse(a)), ac, "anchor", valid)
    zjd._unstack(topo_fit.transform(topo_all.inverse(t)), tc, "topology_features", valid)
    for index, data in enumerate(valid):
        data.local_mol_id = torch.tensor([int(index)], dtype=torch.long)
    meta = {
        "n_rows": int(len(valid)),
        "source": "encoded_valid.pt + env_valid.pt",
        "prep": "frozen FULL prep (10,000-row standardizers, no valid refit)",
        "structure": "valid graphs' own incidence (build_valid_structure)",
        "official_valid_loaded": True,
        "official_test_loaded": False,
    }
    return valid, meta


def _cycle_key_bytes(row: np.ndarray) -> bytes:
    row = np.ascontiguousarray(row, np.float32)
    if np.isnan(row).any() or np.isinf(row).any():
        raise RuntimeError("T25 key contains NaN/Inf")
    neg = np.signbit(row) & (row == 0.0)
    if neg.any():
        row = row.copy()
        row[neg] = 0.0
    return row.tobytes()


def _cycle_decode_levels(logits: np.ndarray, c_levels: np.ndarray) -> np.ndarray:
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)
    order = np.argsort(c_levels, kind="stable")
    sorted_probs = probs[:, order]
    cum = np.cumsum(sorted_probs, axis=1)
    hit = (cum >= 0.5).argmax(axis=1)
    return c_levels[order[hit]]


def load_cycle_deployment() -> dict[str, Any]:
    """The frozen deployed cycle module (candidate C) and its pieces.

    C rule per METHOD_CONTRACT \u00a76: consistent table hit -> class median c;
    conflict hit -> frozen Q(T25); unseen -> decode(D_full(T25)).
    """
    pkg = torch.load(PROTO_RESULTS_DIR / "H_model_package.pt", map_location="cpu", weights_only=False)
    key_map = {bytes.fromhex(kk): int(ss) for kk, ss in zip(pkg["key_order"], pkg["key_slot"])}
    with np.load(CYCLE_RESULTS_DIR / "T25_group_folds.npz", allow_pickle=False) as z:
        vocab = z["vocab"].astype(np.int64)
        c_levels = z["c_levels"].astype(np.float64)
    with np.load(ZFT_RESULTS_DIR / "full_train_targets.npz", allow_pickle=False) as z:
        q_bias = float(np.median(z["c"]))
    q_head = zft.build_q_head(CYCLE_BUILD_SEED, q_bias)
    q_head.load_state_dict(
        torch.load(ZFT_RESULTS_DIR / "Q_raw_soup_state.pt", map_location="cpu", weights_only=True),
        strict=True,
    )
    q_head.eval()
    torch.manual_seed(CYCLE_BUILD_SEED)
    d_full = nn.Sequential(
        nn.Linear(CYCLE_TOPOLOGY_IN, CYCLE_HIDDEN[0]), nn.SiLU(),
        nn.Linear(CYCLE_HIDDEN[0], CYCLE_HIDDEN[1]), nn.SiLU(),
        nn.Linear(CYCLE_HIDDEN[1], len(vocab)),
    )
    nn.init.zeros_(d_full[4].weight)
    with torch.no_grad():
        d_full[4].bias.zero_()
    d_full.load_state_dict(
        torch.load(CYCLE_RESULTS_DIR / "D_full_soup_state.pt", map_location="cpu", weights_only=True),
        strict=True,
    )
    d_full.eval()
    return {
        "key_map": key_map,
        "proto_val": np.asarray(pkg["proto_val"], np.float64),
        "consistent": np.asarray(pkg["consistent"], bool),
        "b_H": float(pkg["b_H"]),
        "vocab": vocab,
        "c_levels": c_levels,
        "q_head": q_head,
        "d_full": d_full,
    }


def cycle_q_c(dep: Mapping[str, Any], T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Frozen candidate-C c-channel on T25 rows (train-only rule, no labels)."""
    T = np.asarray(T, np.float32)
    with torch.no_grad():
        q = zft.q_forward(dep["q_head"], torch.as_tensor(T)).double().numpy()
    route = np.empty(len(T), dtype=object)
    unseen: list[int] = []
    for i in range(len(T)):
        slot = dep["key_map"].get(_cycle_key_bytes(T[i]))
        if slot is None:
            route[i] = "UNSEEN_FALLBACK"
            unseen.append(i)
        elif not dep["consistent"][slot]:
            route[i] = "TRAIN_CONFLICT_FALLBACK"
        else:
            route[i] = "CONSISTENT_HIT"
            q[i] = float(dep["proto_val"][slot])
    if unseen:
        idx = np.asarray(unseen, np.int64)
        with torch.no_grad():
            logits = dep["d_full"](torch.as_tensor(T[idx])).double().numpy()
        q[idx] = _cycle_decode_levels(logits, dep["c_levels"])
        # the deployed decoder fills unseen with the D_full decode (route kept)
    return q, route


def phase_valid_read(
    arms: Sequence[str],
    *,
    out_dir: Path = RESULTS_DIR,
    device: torch.device | None = None,
    log: Any = print,
) -> dict[str, Any]:
    """The single frozen official-valid read for purchased FULL arms.

    Order: verify the cycle deployment and constants identities first; then
    evaluate each frozen FULL soup on the valid graphs' own incidence
    structure; compose y = h_raw + q_C + b_y with per-arm full-train biases;
    paired bootstrap vs the matched reference.  If any cycle identity fails,
    only g metrics are reported (no y composition).
    """
    device = device or torch.device("cpu")
    torch.set_num_threads(8)
    manifest = json.loads((out_dir / "FULL_objects_manifest.json").read_text())
    objects = load_split_objects("FULL", out_dir)
    for arm in arms:
        spec = arm_spec(arm)
        if spec["split"] != "FULL":
            raise RuntimeError(f"--valid-read requires FULL (T_) arms, got {arm}")
        meta = json.loads((out_dir / f"{arm}_meta.json").read_text())
        if meta.get("stopped_reason") != "completed" or meta.get("steps_done") != FULL_STEPS_TOTAL:
            raise RuntimeError(f"{arm} is not a completed full-10k arm")

    # ---- identity 1: constants vs the frozen fulltrain round ----
    with np.load(ZFT_RESULTS_DIR / "full_train_targets.npz", allow_pickle=False) as z:
        zft_constants = {
            str(n): float(v) for n, v in zip(z["constant_names"].tolist(), z["constants"].tolist())
        }
    constants_identity = {
        name: float(objects["constants"][name]) - float(zft_constants[name])
        for name in ("sigma_logP", "sigma_SA", "mu_SA", "sigma_cycle", "mu_cycle")
    }
    constants_ok = all(abs(v) <= 1e-12 for v in constants_identity.values())

    # ---- valid data (the single authorized read) + own incidence structure ----
    valid, valid_meta = load_valid_data_full_prep(objects)
    log(f"[valid-read] n={valid_meta['n_rows']} (single frozen read)")
    vstruct = build_valid_structure()
    valid_arrays = dict(objects["payload_arrays"])
    for key in ("root_base", "pair_ptr", "pair_t", "pair_a", "pair_wJ", "pair_wI", "root_atom"):
        valid_arrays[key] = vstruct[key].astype(valid_arrays[key].dtype)
    valid_payload = prev.TuplePayload(valid_arrays)
    valid_objects = dict(objects)
    valid_objects["payload"] = valid_payload
    valid_objects["payload_arrays"] = valid_arrays

    # ---- valid labels/diagnostics (frozen train constants) ----
    diag = zft.load_valid_diagnostics(ZFT_RESULTS_DIR)
    y_v = np.asarray(diag["y"], np.float64)
    g_v = np.asarray(diag["g"], np.float64)
    k_v = np.asarray(diag["k"], np.int64)

    # ---- identity 2: cycle deployment reproduces the frozen valid C-channel ----
    dep = load_cycle_deployment()
    T_valid = zft.topology_matrix(valid)
    q_c_valid, route_valid = cycle_q_c(dep, T_valid)
    with np.load(CYCLE_RESULTS_DIR / "valid_row_predictions.npz", allow_pickle=True) as z:
        frozen_c_q = np.asarray(z["C_q"], np.float64)
    cycle_identity = {
        "q_C_valid_vs_frozen_max_abs": float(np.max(np.abs(q_c_valid - frozen_c_q))),
        "route_counts": {r: int((route_valid == r).sum()) for r in set(route_valid.tolist())},
        "constants_max_abs_delta": max(abs(v) for v in constants_identity.values()),
    }
    # ---- identity 3: T25 train keys reproduce the frozen cache ----
    with np.load(PROTO_RESULTS_DIR / "T25_cache.npz", allow_pickle=False) as z:
        T_train_frozen = np.asarray(z["T_train"], np.float32)
    prep_meta, fit_data, _dev = build_prepared_data(objects)
    T_train_mine = np.asarray(zft.topology_matrix(fit_data), np.float32)
    cycle_identity["T25_train_vs_cache_max_abs"] = float(np.max(np.abs(T_train_mine - T_train_frozen)))
    q_c_train, route_train = cycle_q_c(dep, T_train_frozen)
    cycle_identity["train_route_counts"] = {r: int((route_train == r).sum()) for r in set(route_train.tolist())}
    cycle_ok = (
        cycle_identity["q_C_valid_vs_frozen_max_abs"] <= REPLAY_TOL
        and cycle_identity["T25_train_vs_cache_max_abs"] <= REPLAY_TOL
        and constants_ok
        and int((route_train == "UNSEEN_FALLBACK").sum()) == 0
    )
    log(
        f"[valid-read] cycle identity ok={cycle_ok} "
        f"qC_valid_diff={cycle_identity['q_C_valid_vs_frozen_max_abs']:.2e} "
        f"T25_train_diff={cycle_identity['T25_train_vs_cache_max_abs']:.2e}"
    )

    out: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "arms": list(arms),
        "valid_meta": valid_meta,
        "constants_identity": constants_identity,
        "cycle_identity": cycle_identity,
        "cycle_verified": bool(cycle_ok),
        "n_valid": int(len(valid)),
        "y": y_v.astype(np.float32),
        "g": g_v.astype(np.float32),
        "k": k_v,
        "k_counts": {name: int(mask.sum()) for name, mask in group_masks(k_v).items()},
        "q_c_valid": q_c_valid.astype(np.float32),
        "route_valid_counts": cycle_identity["route_counts"],
    }

    # ---- per-arm valid evaluation on the valid structure ----
    per_arm: dict[str, dict[str, Any]] = {}
    for arm in arms:
        meta = json.loads((out_dir / f"{arm}_meta.json").read_text())
        replay = build_arm_model(arm, valid_objects)   # valid incidence structure
        state = torch.load(out_dir / f"{arm}_raw_soup_state.pt", map_location="cpu", weights_only=True)
        replay.load_state_dict({key: value.to("cpu") for key, value in state.items()}, strict=True)
        replay = replay.to(device)
        h_v, comp_v = src.evaluate_state_components(replay, valid, device)
        with np.load(out_dir / f"{arm}_fit_predictions.npz", allow_pickle=False) as z:
            h_train = np.asarray(z["raw_soup_fit"], np.float64)
        y_train = np.asarray(objects["targets"]["y"], np.float64)
        b_g = float(meta["calibration_b_g_fit_only"]["raw_soup"])
        entry: dict[str, Any] = {
            "h_raw": h_v.astype(np.float32),
            "ell_raw": comp_v[:, 0].astype(np.float32),
            "s_raw": comp_v[:, 1].astype(np.float32),
            "b_g": b_g,
            "g_raw": h_v.astype(np.float32),
            "g_cal": (h_v + b_g).astype(np.float32),
        }
        if cycle_ok:
            b_y = float(np.median(y_train - h_train - q_c_train))
            entry["b_y"] = b_y
            entry["y_raw"] = (h_v + q_c_valid).astype(np.float32)
            entry["y_cal"] = (h_v + q_c_valid + b_y).astype(np.float32)
        per_arm[arm] = entry
        log(
            f"[valid-read] {arm} g_cal={np.mean(np.abs(entry['g_cal'] - y_v)):.6f}"
            + (f" y_cal={np.mean(np.abs(entry['y_cal'] - y_v)):.6f}" if cycle_ok else " (g-only)")
        )
    out["per_arm"] = per_arm

    # ---- metrics + paired bootstrap vs the first arm as reference ----
    def _paired_gain(err_ref: np.ndarray, err_cand: np.ndarray) -> dict[str, Any]:
        gain = float(np.mean(err_ref) - np.mean(err_cand))
        rng = np.random.default_rng(int(BOOT_SEED))
        n = int(err_ref.shape[0])
        draws = np.empty(int(N_BOOT), np.float64)
        for d in range(int(N_BOOT)):
            idx = rng.choice(n, size=n, replace=True)
            draws[d] = float(err_ref[idx].mean() - err_cand[idx].mean())
        return {
            "gain": gain,
            "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
            "n_boot": int(N_BOOT), "seed": int(BOOT_SEED),
            "definition": "gain = MAE(reference) - MAE(candidate); positive = candidate better",
        }

    metrics: dict[str, Any] = {"reference": reference, "n_valid": int(len(y_v)),
                              "k_counts": out["k_counts"]}
    ref = per_arm[reference]
    for arm in arms[1:]:
        cand = per_arm[arm]
        gains: dict[str, Any] = {}
        for metric in ("g_cal", "g_raw", "y_cal", "y_raw"):
            if metric.startswith("y") and not cycle_ok:
                continue
            target = y_v if metric.startswith("y") else g_v
            err_ref = np.abs(np.asarray(ref[metric], np.float64) - target)
            err_cand = np.abs(np.asarray(cand[metric], np.float64) - target)
            gains[metric] = _paired_gain(err_ref, err_cand)
        gains["mae"] = {
            metric: float(np.mean(np.abs(np.asarray(per_arm[arm][metric], np.float64)
                                        - (y_v if metric.startswith("y") else g_v))))
            for metric in ("g_cal", "g_raw", "y_cal", "y_raw")
            if metric.startswith("g") or cycle_ok
        }
        metrics[arm] = gains
    metrics["mae"][reference] = {
        metric: float(np.mean(np.abs(np.asarray(per_arm[reference][metric], np.float64)
                                    - (y_v if metric.startswith("y") else g_v))))
        for metric in ("g_cal", "g_raw", "y_cal", "y_raw")
        if metric.startswith("g") or cycle_ok
    }
    out["metrics"] = metrics

    # ---- context: the frozen fulltrain COMP body (old deployment) on the
    # clean valid structure, with the same frozen cycle C channel ----
    if cycle_ok:
        comp_h_clean = np.load(out_dir / "valid_structure_check.npz")["h_valid_payload"].astype(np.float64) \
            if (out_dir / "valid_structure_check.npz").exists() else None
        if comp_h_clean is not None:
            cal = json.loads((ZFT_RESULTS_DIR / "calibration.json").read_text())
            b_y_comp = float(cal["per_arm"]["COMP"]["b_y"])
            old_y = comp_h_clean + q_c_valid + b_y_comp
            out["context_old_deployment_C"] = {
                "note": "frozen fulltrain COMP soup + frozen cycle C channel + frozen b_y, "
                        "re-scored on the valid graphs' own incidence structure (ERRATA fix)",
                "y_cal_mae_clean_structure": float(np.mean(np.abs(old_y - y_v))),
                "y_cal_mae_historical_train_structure": 0.11740618350630393,
            }

    np.savez_compressed(
        out_dir / "valid_read_predictions.npz",
        **{
            key: value for key, value in {
                **{k: v for k, v in out.items() if isinstance(v, np.ndarray)},
                **{f"{arm}_{key}": value for arm, entry in per_arm.items()
                  for key, value in entry.items() if isinstance(value, np.ndarray)},
            }.items()
        },
    )
    write_json(out_dir / "valid_read_summary.json", {
        key: value for key, value in out.items() if not isinstance(value, dict) or key in (
            "constants_identity", "cycle_identity", "metrics", "context_old_deployment_C"
        )
    })
    log(f"[valid-read] done cycle_verified={cycle_ok}")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-folds", action="store_true")
    parser.add_argument("--compute-kappa-fusion", action="store_true")
    parser.add_argument("--init-check", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--rec-pretrain", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--dev-eval", action="store_true")
    parser.add_argument("--mechanism-health", action="store_true")
    parser.add_argument("--build-B-objects", action="store_true")
    parser.add_argument("--build-FULL-objects", action="store_true")
    parser.add_argument("--valid-read", action="store_true")
    parser.add_argument("--arm")
    parser.add_argument("--arms", default=None, help="comma-separated arm list for eval/health phases")
    parser.add_argument("--contrasts", default=None,
                        help="comma-separated name:minuend:subtrahend triples for --dev-eval")
    parser.add_argument("--split", default="A", choices=("A", "B"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=str(RESULTS_DIR))
    args = parser.parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    if args.freeze_folds:
        phase_freeze_folds(out_dir=out_dir)
    elif args.compute_kappa_fusion:
        compute_fusion_kappa(args.split, out_dir=out_dir)
    elif args.init_check:
        init_check(out_dir=out_dir, device=device)
    elif args.smoke:
        run_smoke(device=device, out_dir=out_dir)
    elif args.rec_pretrain:
        rec_pretrain(args.split, device=device, out_dir=out_dir)
    elif args.train:
        if not args.arm:
            raise SystemExit("--train requires --arm")
        train_arm(args.arm, out_dir=out_dir, device=device)
    elif args.build_B_objects:
        phase_build_b_objects(out_dir=out_dir)
    elif args.build_FULL_objects:
        phase_build_full_objects(out_dir=out_dir)
    elif args.valid_read:
        if not args.arms:
            raise SystemExit("--valid-read requires --arms (FULL T_ arms; first arm = reference)")
        arms = [a.strip() for a in args.arms.split(",") if a.strip()]
        phase_valid_read(arms, out_dir=out_dir, device=device)
    elif args.dev_eval:
        if not args.arms:
            raise SystemExit("--dev-eval requires --arms")
        arms = [a.strip() for a in args.arms.split(",") if a.strip()]
        contrasts = None
        if args.contrasts:
            contrasts = []
            for chunk in args.contrasts.split(","):
                name, a, b = chunk.strip().split(":")
                contrasts.append((name, a, b))
        phase_dev_eval(arms, contrasts=contrasts, out_dir=out_dir, device=device)
    elif args.mechanism_health:
        if not args.arms:
            raise SystemExit("--mechanism-health requires --arms")
        arms = [a.strip() for a in args.arms.split(",") if a.strip()]
        phase_mechanism_health(arms, out_dir=out_dir, device=device)
    else:
        raise SystemExit("choose a phase")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
