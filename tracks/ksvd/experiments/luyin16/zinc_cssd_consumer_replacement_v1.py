"""ZINC CSSD consumer replacement v1: can the frozen CSSD basis supply the
local phi65 interface of the strong M_COMP consumer?

Round: ``zinc_cssd_consumer_replacement_v1``.

The single question: replacing the existing strong M_COMP consumer's local
phi65 input (``root_codes`` reads ``data.dict_phi``) with the *full
reconstruction* provided by the frozen CSSD basis package (fit on this
round's own fit fold, historical stage-0 of
``zinc_cssd_nonlinear_binding_v1``) — can the prediction performance on
complete y be held, while the sparse residual actually shows molecular
differences in the read-out?

* ``RAW``  — the reference: the original 297,539-parameter M_COMP consumer
  (SiLU local encoder on 125-D tuple features, real J incidence, zero-init
  ``W_loc`` 342x64 injection, ComponentReader 39->2, COMP supervision) with
  the original phi65 input.
* ``DICT`` — the identical consumer, identical trainable parameters, with
  the single difference that the phi entering the tuple-feature
  standardisation inside ``root_codes`` is the frozen-CSSD full
  reconstruction ``phi_hat = c @ U.T + alpha @ Dbar.T`` (common term kept;
  alpha = tied-IHT-10 top-8 on the residual; CSSD in the original phi
  coordinates, the historical phi_mean/phi_std/phi_scale applied *after*
  decoding, never twice).

RAW is the performance-carrying reference: this round does not ask DICT to
beat RAW, does not claim the dictionary must be used, and makes no
y-only/SOTA claim.  COMP supervision is retained as a disclosed auxiliary
condition.  Four formal runs only (RAW/DICT x body seeds 0/1) on the
historical 8001-row fit fold; dev = sorted union(select, confirm) = the
historical development rows (a *development comparison*, never a new
independent confirm).  Official valid/test are never loaded.

Frozen reused objects (read-only, from
``results/zinc_cssd_nonlinear_binding_v1``): fold, targets, tuple payload
(phi scaler/kappa), prep, CSSD basis (U/common_rms/D; its historical refit
included task-supervised training — recorded, not claimed as a purely
unsupervised basis), and the shared frozen Q(topology25) soup.
``y_raw = ell_hat + s_hat + Q_raw``; ``y_cal = y_raw + b_y`` with one
fit-median ``b_y``.

Usage (local CPU for source/checks/probe; res-2 res2-cu124 for the formal
runs, all through the registered runner)::

    python -m tracks.ksvd.experiments.luyin16.\\
zinc_cssd_consumer_replacement_v1 --stage source-objects
    ... --stage checks
    ... --stage probe
    ... --stage smoke --device cuda:0
    ... --stage train --arm RAW --seed 0 --device cuda:0
    ... --stage terminal-eval --device cuda:0
"""

from __future__ import annotations

import argparse
import hashlib
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_v0 as v0
from tracks.ksvd.experiments.luyin16 import (
    zinc_chemistry_component_supervision_seed0_v1 as zcs,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_cssd_basis_reuse_v1 as zreuse,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_cssd_nonlinear_binding_v1 as parent,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_full_cycle_target_decomposition_v1 as zftd,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_local_dictionary_component_supervision_seed0_v1 as zldc,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_local_tuple_dictionary_joint_vs_independent_seed0_v1 as prev,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_local_tuple_dictionary_vs_mlp_seed0_v1 as mlpmod,
)
from tracks.ksvd.experiments.luyin16 import zinc_task_dictionary_and_cycle_witness_seed0_v1 as zw
from tracks.ksvd.experiments.luyin16 import zinc_zero_binding_baseline_seed0_v1 as zbn

PROTOCOL_VERSION = "zinc-cssd-consumer-replacement-v1"
RESULT_SLUG = "zinc_cssd_consumer_replacement_v1"
TRACK_ROOT = zw.TRACK_ROOT
RESULTS_DIR = TRACK_ROOT / "results" / RESULT_SLUG

#: read-only source of every frozen object this round reuses
SOURCE_DIR = TRACK_ROOT / "results" / "zinc_cssd_nonlinear_binding_v1"
#: committed canonical-SMILES table (reuse round; group verification only)
SMILES_TABLE = TRACK_ROOT / "results" / "zinc_cssd_basis_reuse_v1" / "train_canonical_smiles.npz"
#: the historical strong M_COMP soup (fit-only read-only interface probe)
OLD_SOUP_DIR = zldc.RESULTS_DIR
OLD_SOUP_PATH = OLD_SOUP_DIR / "M_COMP_raw_soup_state.pt"

# ---- frozen recipe (identical for both arms / all four runs) ---------------
ARMS = ("RAW", "DICT")
SEEDS = (0, 1)
SEED = 0
EPOCHS = parent.EPOCHS                    # 240
LR = parent.LR                            # 1e-3
WEIGHT_DECAY = parent.WEIGHT_DECAY        # 1e-5 (coupled)
GRAD_CLIP = parent.GRAD_CLIP              # 5.0
BATCH_SIZE = parent.BATCH_SIZE            # 128
SOUP_EPOCHS = parent.SOUP_EPOCHS          # 236..240 fixed five-epoch soup
LOG_EPOCHS = parent.LOG_EPOCHS            # (1, 40, 120, 240)
TRAIN_SHUFFLE_OFFSET = parent.TRAIN_SHUFFLE_OFFSET  # 101
COMPONENT_LOSS_WEIGHT = parent.COMPONENT_LOSS_WEIGHT  # 0.5

# CSSD basis specification (frozen, from the source round)
CSSD_Q = parent.CSSD_Q                    # 1
CSSD_K_ATOMS = parent.CSSD_K_ATOMS        # 32
CSSD_SPARSITY = parent.CSSD_SPARSITY      # 8
CSSD_IHT_STEPS = parent.CSSD_IHT_STEPS    # 10
COMMON_DIM = int(CSSD_Q)

# parameter contract (the historical M_COMP consumer, byte-for-byte)
EXPECTED_PARAMETERS = zldc.EXPECTED_PARAMETERS            # 297,539
EXPECTED_BODY_PARAMETERS = zldc.EXPECTED_BODY_PARAMETERS  # 184,707
BRIDGE_PARAMETERS = zldc.BRIDGE_PARAMETERS                # 82,944
LOCAL_PARAMETERS = zldc.LOCAL_PARAMETERS                  # 29,888
READER_OUTPUT_PARAMETERS = 2 * 39 + 2                     # 80

# interpretation constants (frozen before any training)
BOOT_SEED = 20261012
N_BOOT = 2000
REPLAY_TOL = 1e-4
#: engineering tolerance for keeping DICT as a next-stage candidate
#: (a budget rule for THIS round, never a general equivalence threshold)
TOL_AVG = 0.003
TOL_SEED = 0.005
#: pre-training fit-only interface probe sample cap
PROBE_MOLECULES = 256

write_json = parent.write_json
read_json = parent.read_json
file_sha256 = parent.file_sha256
array_sha256 = parent.array_sha256
state_hash = parent.state_hash
seed_everything = parent.seed_everything
resolve_device = parent.resolve_device


def allocation_probe(device: torch.device | None = None) -> dict[str, Any]:
    """Current-process allocation probe (the no-arg zldc probe).

    The source module's wrapper forwards a device argument to a no-arg
    probe (a latent signature bug caught by this round's first train runs);
    this round calls the underlying no-arg probe directly.
    """
    return zldc.allocation_probe()

_residual_structure_inputs = (
    "The consumer keeps every structure input except the local phi65 patch: "
    "chemistry one-hot28(root)/one-hot28(neighbor)/onehot4(bond) inside the tuple "
    "features, the real J incidence weights pair_wJ, the Sem108 semantic interface "
    "+ size2 into fusion layer 1 (where W_loc injects the local channel), the static "
    "relations, the posterior MLP bridge, the reader inputs and the shared frozen "
    "Q(topology25). This round only claims that the one effective local structure "
    "interface (the phi65 block of the tuple features) is supplied by the basis."
)


# ---------------------------------------------------------------------------
# 1. frozen source objects (read-only) and the round fold view
# ---------------------------------------------------------------------------

def round_dev_idx(objects: Mapping[str, Any]) -> np.ndarray:
    """dev = sorted union(select, confirm): the historical development rows."""
    fold = objects["fold"]
    return np.sort(np.union1d(
        np.asarray(fold["select_idx"], np.int64),
        np.asarray(fold["confirm_idx"], np.int64),
    ))


def load_round_objects() -> dict[str, Any]:
    """Every frozen object this round reuses, hash-checked at load."""
    objects = parent.load_objects(SOURCE_DIR)
    fold = objects["fold"]
    fit_idx = np.asarray(fold["fit_idx"], np.int64)
    dev_idx = round_dev_idx(objects)
    if np.intersect1d(fit_idx, dev_idx).size != 0:
        raise RuntimeError("fit/dev overlap in the round fold view")
    if np.union1d(fit_idx, dev_idx).size != 10000:
        raise RuntimeError("fit + dev does not cover the 10000 train rows")
    basis = parent.load_cssd_basis(SOURCE_DIR)
    q_soup = parent.load_q_soup(SOURCE_DIR)
    return {
        "objects": objects, "basis": basis, "q_soup": q_soup,
        "fit_idx": fit_idx, "dev_idx": dev_idx,
    }


def build_round_data(round_objects: Mapping[str, Any]) -> tuple[dict[str, Any], list[Any], list[Any]]:
    """prep-verified fit data + the merged dev data (sorted by global index)."""
    objects = round_objects["objects"]
    prep_meta, fit_data, select_data, confirm_data = parent.build_prepared_data(objects, verify_prep=True)
    fit_idx = np.asarray(objects["fold"]["fit_idx"], np.int64)
    for position, index in enumerate(fit_idx.tolist()):
        if int(fit_data[position].local_mol_id.reshape(-1)[0].item()) != int(index):
            raise RuntimeError("fit data local_mol_id misaligned with fit_idx")
    by_index: dict[int, Any] = {}
    for row in list(select_data) + list(confirm_data):
        by_index[int(row.local_mol_id.reshape(-1)[0].item())] = row
    dev_idx = round_dev_idx(objects)
    if sorted(by_index) != dev_idx.tolist():
        raise RuntimeError("merged select+confirm rows != dev_idx")
    dev_data = [by_index[int(i)] for i in dev_idx.tolist()]
    return prep_meta, fit_data, dev_data


def frozen_basis_parts(basis: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """U / common_rms / normalized residual dictionary Dbar (frozen tensors)."""
    U, common_rms, Dbar = zreuse._frozen_code_parts(basis)
    return {
        "U": U.contiguous(),
        "common_rms": common_rms.contiguous(),
        "Dbar": Dbar.contiguous(),
        "hashes": {
            "U": array_sha256(basis["U"]),
            "common_rms": array_sha256(basis["common_rms"]),
            "D": array_sha256(basis["D"]),
        },
    }


def _phi_rows_for(indices: Sequence[int]) -> tuple[np.ndarray, dict[str, Any]]:
    """Per-root phi65 rows of the given molecules from the train-only env cache."""
    phi, _atom, node_sizes = prev._env_phi_atom()
    idx = np.asarray(indices, np.int64)
    bounds = np.concatenate([[0], np.cumsum(node_sizes)])
    rows = np.concatenate([phi[bounds[i]:bounds[i + 1]] for i in idx.tolist()], axis=0)
    checks = {
        "n_molecules": int(idx.size),
        "n_phi_rows": int(rows.shape[0]),
        "expected_rows": int(sum(int(node_sizes[i]) for i in idx.tolist())),
        "phi_dim": int(rows.shape[1]),
    }
    if checks["n_phi_rows"] != checks["expected_rows"]:
        raise RuntimeError("phi row count mismatch")
    return rows, checks


def fit_mean_codes(basis: Mapping[str, np.ndarray], fit_idx: np.ndarray) -> dict[str, np.ndarray]:
    """Fit-root mean alpha / mean phi / mean phi_hat under the frozen basis."""
    rows, checks = _phi_rows_for(fit_idx.tolist())
    z, phi_hat, rel = zreuse._root_codes(basis, rows)
    alpha = z[:, COMMON_DIM:]
    if not np.isfinite(z).all() or not np.isfinite(phi_hat).all():
        raise RuntimeError("non-finite frozen-basis codes on the fit rows")
    return {
        "mean_alpha": alpha.mean(axis=0).astype(np.float64),
        "mean_phi": rows.mean(axis=0).astype(np.float64),
        "mean_phi_hat": phi_hat.mean(axis=0).astype(np.float64),
        "checks": checks,
        "fit_rel_err_median": float(np.median(rel)),
        "fit_rel_err_p95": float(np.percentile(rel, 95)),
        "fit_alpha_nnz": float((alpha != 0).sum(axis=1).mean()),
    }


# ---------------------------------------------------------------------------
# 2. the DICT encoder: the single phi-input swap inside root_codes
# ---------------------------------------------------------------------------

class _DictPhiView:
    """Attribute view of a batch whose ``dict_phi`` rows are substituted.

    ``root_codes`` only reads ``dict_phi`` / ``dict_atom`` / ``local_mol_id``
    / ``batch`` from the data object; every other attribute delegates to the
    original batch.  The global ``data.dict_phi`` is never modified.
    """

    def __init__(self, data: Any, phi: torch.Tensor) -> None:
        self._data = data
        self._phi = phi

    def __getattr__(self, name: str) -> Any:
        if name == "dict_phi":
            return self._phi
        return getattr(self._data, name)


class LocalTupleEncoderMCSSD(mlpmod.LocalTupleEncoderM):
    """M encoder whose per-root phi input is the frozen CSSD reconstruction.

    ``cssd_input`` is applied once per ``root_codes`` call (per root row of
    the batch), exactly between reading ``data.dict_phi`` and the
    ``tuple_features`` standardisation.  Math identical to the reuse round's
    frozen operator::

        c     = phi @ U
        r     = phi - c @ U.T
        alpha = tied_iht_codes(Dbar, r, s=8, steps=10)
        phi_hat = c @ U.T + alpha @ Dbar.T      (common term kept)

    The basis tensors are non-persistent buffers (frozen, never trained, not
    in the state dict, not in the optimizer).  Hooks: ``alpha_replace`` (the
    frozen dictionary intervention), ``phi_replace`` (the read-only
    localisation diagnostic; takes precedence over decoding), and
    ``decode_enabled=False`` (the identity hook — the machinery engaged with
    a pass-through input).
    """

    def __init__(
        self,
        payload: prev.TuplePayload,
        a_init: torch.Tensor,
        basis_parts: Mapping[str, torch.Tensor],
        *,
        decode: bool = True,
    ) -> None:
        super().__init__(payload, a_init)
        self.register_buffer("cssd_U", basis_parts["U"].detach().clone().to(torch.float32), persistent=False)
        self.register_buffer(
            "cssd_common_rms", basis_parts["common_rms"].detach().clone().to(torch.float32), persistent=False
        )
        self.register_buffer("cssd_Dbar", basis_parts["Dbar"].detach().clone().to(torch.float32), persistent=False)
        self.decode_enabled = bool(decode)
        self.alpha_replace: torch.Tensor | None = None
        self.phi_replace: torch.Tensor | None = None
        self.cssd_stats: dict[str, Any] = {}

    # -- the frozen CSSD operator (identical to the reuse round) ------------
    def cssd_decode(self, phi: torch.Tensor) -> torch.Tensor:
        U = self.cssd_U.to(dtype=phi.dtype)
        Dbar = self.cssd_Dbar.to(dtype=phi.dtype)
        c = phi @ U
        if self.alpha_replace is not None:
            alpha = self.alpha_replace.to(device=phi.device, dtype=phi.dtype).expand(int(phi.shape[0]), -1)
        else:
            r = phi - c @ U.t()
            alpha = v0.tied_iht_codes(Dbar, r, s=CSSD_SPARSITY, steps=CSSD_IHT_STEPS)
        return c @ U.t() + alpha @ Dbar.t()

    def cssd_input(self, phi: torch.Tensor) -> torch.Tensor:
        if self.phi_replace is not None:
            return self.phi_replace.to(device=phi.device, dtype=phi.dtype).expand(int(phi.shape[0]), -1)
        if self.decode_enabled:
            return self.cssd_decode(phi)
        return phi

    def frozen_basis_hashes(self) -> dict[str, str]:
        return {
            "U": array_sha256(self.cssd_U.detach().cpu().numpy()),
            "common_rms": array_sha256(self.cssd_common_rms.detach().cpu().numpy()),
            "Dbar": array_sha256(self.cssd_Dbar.detach().cpu().numpy()),
        }

    def root_codes(self, data: Any) -> torch.Tensor:
        phi = data.dict_phi
        replaced = self.cssd_input(phi)
        if replaced is not phi:
            with torch.no_grad():
                rel = ((phi - replaced).norm(dim=1) / (phi.norm(dim=1) + v0.EPS))
                self.cssd_stats = {
                    "cssd_decode_enabled": bool(self.decode_enabled),
                    "cssd_alpha_replace_active": bool(self.alpha_replace is not None),
                    "cssd_phi_replace_active": bool(self.phi_replace is not None),
                    "cssd_input_rel_err_median": float(rel.median()),
                    "cssd_input_rel_err_p95": float(torch.quantile(rel, 0.95)),
                }
            out = super().root_codes(_DictPhiView(data, replaced))
            self.last_stats = dict(self.last_stats)
            self.last_stats.update(self.cssd_stats)
            return out
        return super().root_codes(data)


def arm_parameter_audit(model: nn.Module) -> dict[str, int]:
    return zcs.component_parameter_audit(model)


def build_arm(
    arm: str,
    payload: prev.TuplePayload,
    kappa_M: float,
    basis_parts: Mapping[str, torch.Tensor],
    seed: int,
    *,
    decode: bool = True,
) -> nn.Module:
    """Fresh untrained arm with the body seed *explicitly penetrated*.

    ``LocalTupleFullM``'s skeleton (``zbn.DeployFull.__init__``) internally
    calls ``torch.manual_seed(int(zbn.SEED))`` with the historical fixed 0,
    so an outer ``manual_seed(seed)`` cannot produce seed 1.  This factory
    substitutes the seed constant for the duration of construction only
    (scoped, restored in ``finally``; the historical module global is never
    permanently modified and parallel processes are unaffected), giving the
    exact same constructor and stream with ``seed`` in place of the fixed 0.
    ``A_raw`` stays the FRAME_SEED frame (cross-seed identical by design) and
    ``W_loc`` stays exactly zero; ``ComponentReader`` is fork_rng (RNG
    neutral).  The replacement encoder consumes no RNG draws, so the
    post-construction RNG state is the seed stream in both arms.
    """
    if arm not in ARMS:
        raise ValueError(arm)
    a_init = prev.init_d_loc().t().contiguous()
    saved_seed = zbn.SEED
    try:
        zbn.SEED = int(seed)
        model = mlpmod.LocalTupleFullM(payload, a_init)
    finally:
        zbn.SEED = saved_seed
    if arm == "DICT":
        model.local_tuple = LocalTupleEncoderMCSSD(payload, a_init, basis_parts, decode=decode)
    model.local_tuple.kappa = float(kappa_M)
    model.reader = zcs.ComponentReader(model.reader.net)
    audit = arm_parameter_audit(model)
    expected = {
        "total_parameters": EXPECTED_PARAMETERS,
        "base_body_parameters": EXPECTED_BODY_PARAMETERS,
        "bridge_parameters": BRIDGE_PARAMETERS,
        "local_tuple_parameters": LOCAL_PARAMETERS,
        "reader_output_parameters": READER_OUTPUT_PARAMETERS,
    }
    if audit != expected:
        raise RuntimeError(f"{arm} parameter audit failed: {audit} != {expected}")
    if not isinstance(model.local_dictionary_bridge, zw.MLPBridge):
        raise RuntimeError(f"{arm} posterior bridge is not the matched MLP bridge")
    if arm == "RAW" and not isinstance(model.local_tuple, mlpmod.LocalTupleEncoderM):
        raise RuntimeError("RAW arm must use the untouched historical M encoder")
    if arm == "DICT" and not isinstance(model.local_tuple, LocalTupleEncoderMCSSD):
        raise RuntimeError("DICT arm must use the CSSD-input M encoder")
    return model


def run_dir(arm: str, seed: int, out_dir: Path = RESULTS_DIR) -> Path:
    if arm not in ARMS:
        raise ValueError(arm)
    return Path(out_dir) / "runs" / f"{arm}_s{int(seed)}"


def load_run(arm: str, seed: int, out_dir: Path = RESULTS_DIR) -> dict[str, Any]:
    rdir = run_dir(arm, seed, out_dir)
    manifest = read_json(rdir / "manifest.json")
    soup_state = torch.load(rdir / "soup_state.pt", map_location="cpu", weights_only=False)
    if state_hash(soup_state) != manifest["soup_state_sha256"]:
        raise RuntimeError(f"soup state hash mismatch for {arm} s{seed}")
    return {"manifest": manifest, "soup_state": soup_state, "run_dir": rdir}


# ---------------------------------------------------------------------------
# 3. stage: source objects (manifest of everything reused, read-only)
# ---------------------------------------------------------------------------

def phase_source_manifest(*, out_dir: Path = RESULTS_DIR, log: Any = print) -> dict[str, Any]:
    torch.set_num_threads(8)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    round_objects = load_round_objects()
    objects = round_objects["objects"]
    fold = objects["fold"]
    fit_idx = round_objects["fit_idx"]
    dev_idx = round_objects["dev_idx"]
    basis = round_objects["basis"]

    # canonical-SMILES groups must not straddle fit / dev
    smiles = zreuse._canonical_smiles()
    in_fit = np.zeros(10000, dtype=bool)
    in_fit[fit_idx] = True
    in_dev = np.zeros(10000, dtype=bool)
    in_dev[dev_idx] = True
    straddling = 0
    for value in np.unique(smiles):
        members = np.where(smiles == value)[0]
        flags = in_fit[members]
        if flags.any() and not flags.all():
            straddling += 1
    if straddling != 0:
        raise RuntimeError(f"{straddling} canonical-SMILES groups straddle fit/dev")

    # prep fingerprint: recomputed and compared inside build_round_data
    prep_meta, _fit_data, _dev_data = build_round_data(round_objects)

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "stage": "source-objects",
        "source_dir": str(SOURCE_DIR.relative_to(TRACK_ROOT)),
        "source_read_only": True,
        "reused_artifacts": {
            name: file_sha256(SOURCE_DIR / name)
            for name in (
                "fold.npz", "fold_manifest.json", "targets.npz", "tuple_payload.npz",
                "kappa_M.json", "prep.npz", "objects_manifest.json",
                "cssd_basis.npz", "cssd_refit.json", "Q_soup_state.pt", "Q_meta.json",
            )
        },
        "fold_view": {
            "fit": {
                "n": int(fit_idx.size), "sha256": array_sha256(fit_idx, np.int64),
                "definition": "the source round's committed fit fold (straddling SMILES groups moved into fit)",
            },
            "dev": {
                "n": int(dev_idx.size), "sha256": array_sha256(dev_idx, np.int64),
                "definition": (
                    "sorted union(select_idx, confirm_idx) of the source fold — the "
                    "historical development rows; a DEVELOPMENT COMPARISON set, not a "
                    "new independent confirm"
                ),
            },
            "disjoint": True,
            "cover_10000": True,
            "smiles_groups_never_cross_fit_dev": True,
            "n_straddling_groups": int(straddling),
        },
        "prep_fingerprint_verified": True,
        "cssd_basis": {
            "U_sha256": array_sha256(basis["U"]),
            "common_rms_sha256": array_sha256(basis["common_rms"]),
            "D_sha256": array_sha256(basis["D"]),
            "spec": (
                "radius-2 untyped phi65, q=1, 32 atoms, top-8, 10-step tied IHT; "
                "basis package U/common_rms/D; consumer-side fully frozen"
            ),
            "fit_note": (
                "the historical basis refit included training-task supervision in its "
                "temporary head (recorded as-is; NOT claimed as a purely unsupervised basis)"
            ),
            "refit_domain": "this round's own fit fold (8001 rows, source round stage-0)",
        },
        "q_soup": {
            "state_sha256": state_hash(round_objects["q_soup"]),
            "n_fit": 8001,
            "note": "frozen shared Q(topology25->64->32->1); never retrained this round",
        },
        "kappa_M": float(objects["kappa"]["kappa_M"]),
        "kappa_D": float(prev.TuplePayload(objects["payload_arrays"]).kappa),
        "old_soup_probe_available": bool(OLD_SOUP_PATH.exists()),
        "residual_structure_inputs": _residual_structure_inputs,
        "no_re_split": True,
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "seconds": float(time.perf_counter() - started),
    }
    write_json(out_dir / "source_manifest.json", manifest)
    log(
        f"[source-objects] fit={fit_idx.size} dev={dev_idx.size} verified in "
        f"{manifest['seconds']:.1f}s (groups straddling fit/dev: 0)"
    )
    return manifest


# ---------------------------------------------------------------------------
# 4. stage: focused pre-training checks (CPU)
# ---------------------------------------------------------------------------

def _fit_sample(objects: Mapping[str, Any], n: int = PROBE_MOLECULES) -> np.ndarray:
    """First n fit molecules in ascending gid order (label-blind, fixed)."""
    fit_idx = np.asarray(objects["fold"]["fit_idx"], np.int64)
    gid = np.asarray(objects["targets"]["gid"], np.int64)
    order = np.argsort(gid[fit_idx], kind="stable")
    return fit_idx[order[: int(n)]]


def run_checks(*, out_dir: Path = RESULTS_DIR, log: Any = print) -> dict[str, Any]:
    torch.set_num_threads(8)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    device = torch.device("cpu")
    round_objects = load_round_objects()
    objects = round_objects["objects"]
    basis = round_objects["basis"]
    basis_parts = frozen_basis_parts(basis)
    payload = prev.TuplePayload(objects["payload_arrays"])
    kappa_M = float(objects["kappa"]["kappa_M"])
    _prep_meta, fit_data, _dev_data = build_round_data(round_objects)
    fit_idx = round_objects["fit_idx"]
    checks: dict[str, Any] = {}

    # (a) source provenance restated (verified on load)
    checks["source"] = {
        "reused_from": str(SOURCE_DIR.relative_to(TRACK_ROOT)),
        "hash_checks_passed": True,
        "fit_n": int(fit_idx.size),
        "dev_n": int(round_objects["dev_idx"].size),
    }

    # (b) the RAW factory reproduces the historical M_COMP bit-for-bit
    historical = zldc.build_arm_comp("M_COMP", payload, kappa_M)
    seed_everything(SEED)
    mine = build_arm("RAW", payload, kappa_M, basis_parts, SEED)
    sd_h = {k: v.detach().clone() for k, v in historical.state_dict().items()}
    sd_m = {k: v.detach().clone() for k, v in mine.state_dict().items()}
    if sorted(sd_h) != sorted(sd_m):
        raise RuntimeError("RAW state keys differ from the historical M_COMP")
    raw_matches_history = all(torch.equal(sd_h[k], sd_m[k]) for k in sd_h)
    checks["raw_equals_historical_m_comp_state"] = bool(raw_matches_history)
    if not raw_matches_history:
        raise RuntimeError("RAW factory did not reproduce the historical M_COMP init")

    # (c) identity hook: the CSSD machinery engaged with a pass-through input
    seed_everything(SEED)
    model_dict_nohook = build_arm("DICT", payload, kappa_M, basis_parts, SEED, decode=False)
    sd_n = {k: v.detach().clone() for k, v in model_dict_nohook.state_dict().items()}
    checks["identity_hook_state_equals_raw"] = bool(all(torch.equal(sd_m[k], sd_n[k]) for k in sd_m))
    g = np.asarray(objects["targets"]["g"], np.float64)
    target_g = torch.as_tensor(g[fit_idx], dtype=torch.float32)
    probe_indices = list(range(8))
    batch = zftd.make_batch(fit_data, probe_indices, target_g, device)
    mine.eval()
    model_dict_nohook.eval()
    with torch.no_grad():
        p_raw = mine(batch, mask=cm.C6_MASK)
        p_nohook = model_dict_nohook(batch, mask=cm.C6_MASK)
    checks["identity_hook_predictions_equal"] = bool(torch.equal(p_raw, p_nohook))
    if not checks["identity_hook_predictions_equal"]:
        raise RuntimeError("identity hook changed predictions")

    # (d) parameter contract + frozen basis not trainable / not in state
    audit = {arm: arm_parameter_audit(m) for arm, m in (("RAW", mine), ("DICT", model_dict_nohook))}
    checks["parameter_audit"] = audit
    checks["parameter_audit_ok"] = bool(all(
        a == {
            "total_parameters": EXPECTED_PARAMETERS,
            "base_body_parameters": EXPECTED_BODY_PARAMETERS,
            "bridge_parameters": BRIDGE_PARAMETERS,
            "local_tuple_parameters": LOCAL_PARAMETERS,
            "reader_output_parameters": READER_OUTPUT_PARAMETERS,
        } for a in audit.values()
    ))
    dict_buffers_in_state = [
        k for k in model_dict_nohook.state_dict() if k.startswith("local_tuple.cssd_")
    ]
    checks["cssd_buffers_not_in_state_dict"] = bool(not dict_buffers_in_state)
    trainable = [n for n, p in model_dict_nohook.named_parameters()]
    checks["cssd_buffers_not_trainable"] = bool(
        not any(n.startswith("local_tuple.cssd_") for n in trainable)
    )
    if dict_buffers_in_state or any(n.startswith("local_tuple.cssd_") for n in trainable):
        raise RuntimeError("CSSD basis leaked into the state dict / parameters")

    # (e) seed identity: same-seed arms identical, seed 1 genuinely different
    seed_everything(SEED)
    model_dict = build_arm("DICT", payload, kappa_M, basis_parts, SEED)
    sd_d = {k: v.detach().clone() for k, v in model_dict.state_dict().items()}
    if sorted(sd_m) != sorted(sd_d):
        raise RuntimeError("RAW/DICT state key sets differ")
    max_abs = max(float((sd_m[k] - sd_d[k]).abs().max()) for k in sd_m)
    checks["same_seed_raw_dict_identical"] = bool(max_abs == 0.0)
    checks["same_seed_max_abs_diff"] = max_abs
    seed_everything(1)
    model_raw1 = build_arm("RAW", payload, kappa_M, basis_parts, 1)
    sd_r1 = {k: v.detach().clone() for k, v in model_raw1.state_dict().items()}
    differ_seed = [k for k in sd_m if not torch.equal(sd_m[k], sd_r1[k])]
    checks["seed0_vs_seed1_differing_keys"] = differ_seed
    checks["seed0_vs_seed1_n_differing"] = len(differ_seed)
    checks["seed1_genuinely_differs"] = bool(len(differ_seed) > 0)
    checks["a_raw_frame_identical_across_seeds"] = bool(torch.equal(
        sd_m["local_tuple.A_raw"], sd_r1["local_tuple.A_raw"]
    ))
    checks["w_loc_zero_across_seeds"] = bool(
        float(sd_m["local_tuple.W_loc"].abs().sum()) == 0.0
        and float(sd_r1["local_tuple.W_loc"].abs().sum()) == 0.0
    )
    if not (checks["same_seed_raw_dict_identical"] and checks["seed1_genuinely_differs"]):
        raise RuntimeError(f"seed identity checks failed: {checks}")

    # (f) CSSD encode/decode == the reuse round's reference operator; the
    #     mean-alpha intervention really changes the input; c is untouched
    sample = _fit_sample(objects, 256)
    sample_rows, _rows_checks = _phi_rows_for(sample.tolist())
    encoder = model_dict.local_tuple
    encoder.eval()
    phi_t = torch.as_tensor(sample_rows, dtype=torch.float32)
    with torch.no_grad():
        phi_hat_model = encoder.cssd_decode(phi_t)
    z_ref, phi_hat_ref, _rel = zreuse._root_codes(basis, sample_rows)
    checks["cssd_vs_reference"] = {
        "n_rows": int(sample_rows.shape[0]),
        "max_abs_phi_hat_diff": float(np.abs(
            phi_hat_model.numpy().astype(np.float64) - phi_hat_ref).max()),
        "z_width": int(z_ref.shape[1]),
        "alpha_nnz_exact_8": bool(((z_ref[:, COMMON_DIM:] != 0).sum(axis=1) == CSSD_SPARSITY).all()),
    }
    if checks["cssd_vs_reference"]["max_abs_phi_hat_diff"] > 1e-5 or not checks["cssd_vs_reference"]["alpha_nnz_exact_8"]:
        raise RuntimeError(f"CSSD decode != reference operator: {checks['cssd_vs_reference']}")
    mean_codes = fit_mean_codes(basis, fit_idx)
    mean_alpha = torch.as_tensor(mean_codes["mean_alpha"], dtype=torch.float32)
    encoder.alpha_replace = mean_alpha
    with torch.no_grad():
        phi_hat_mean = encoder.cssd_decode(phi_t)
    encoder.alpha_replace = None
    dbar_ref = np.asarray(zreuse._frozen_code_parts(basis)[2].numpy(), np.float64)
    alpha_ref = z_ref[:, COMMON_DIM:].astype(np.float64)
    common_ref = phi_hat_ref - alpha_ref @ dbar_ref.T
    common_int = phi_hat_mean.numpy().astype(np.float64) - np.asarray(mean_codes["mean_alpha"], np.float64) @ dbar_ref.T
    checks["mean_alpha_intervention_input"] = {
        "phi_hat_max_abs_change": float(np.abs(
            phi_hat_mean.numpy().astype(np.float64) - phi_hat_ref).max()),
        "phi_hat_median_abs_change": float(np.median(np.abs(
            phi_hat_mean.numpy().astype(np.float64) - phi_hat_ref))),
        "common_term_max_abs_change": float(np.abs(common_int - common_ref).max()),
        "c_definition": "c = phi @ U recomputed from the SAME phi inside decode (kept by construction)",
        "mean_alpha_l2": float(mean_alpha.double().norm()),
    }
    if checks["mean_alpha_intervention_input"]["phi_hat_max_abs_change"] <= 0.0:
        raise RuntimeError("mean-alpha replacement did not change phi_hat")
    if checks["mean_alpha_intervention_input"]["common_term_max_abs_change"] > 1e-5:
        raise RuntimeError("mean-alpha replacement moved the common term")

    # (g) batch merge / order invariance (fit molecules only, both arms)
    invariance: dict[str, Any] = {}
    subset = fit_data[:24]
    for arm, model in (("RAW", mine), ("DICT", model_dict)):
        model.eval()
        order = list(range(len(subset)))
        with torch.no_grad():
            singles = []
            for i in order:
                b1 = zftd.make_batch(subset, [i], torch.zeros(len(subset)), device)
                singles.append(float(model(b1, mask=cm.C6_MASK).view(-1)[0]))
            shuffled = [order[(i * 7 + 3) % len(order)] for i in range(len(order))]
            bs = zftd.make_batch(subset, shuffled, torch.zeros(len(subset)), device)
            grouped = model(bs, mask=cm.C6_MASK).view(-1).numpy().astype(np.float64)
        d = np.abs(np.asarray(singles)[np.asarray(shuffled)] - grouped)
        invariance[arm] = {"max_abs": float(d.max()), "mean_abs": float(d.mean())}
    checks["batch_order_invariance"] = invariance
    if any(v["max_abs"] > 1e-5 for v in invariance.values()):
        raise RuntimeError(f"batch/order invariance failed: {invariance}")

    # (h) label independence
    batch_y0 = zftd.make_batch(fit_data, probe_indices, torch.zeros(8), device)
    with torch.no_grad():
        p_y0 = mine(batch_y0, mask=cm.C6_MASK)
    checks["label_independence_max_abs"] = float((p_raw - p_y0).abs().max())
    if checks["label_independence_max_abs"] != 0.0:
        raise RuntimeError("labels entered the forward path")

    # (i) molecule-ID / tuple-row correspondence vs the reference operators
    first_fit = int(fit_idx[0])
    root_base = np.asarray(objects["payload_arrays"]["root_base"], np.int64)
    global_roots = torch.tensor(
        list(range(int(root_base[first_fit]), int(root_base[first_fit + 1]))), dtype=torch.long
    )
    phi_env, atom_env, _sizes = prev._env_phi_atom()
    phi_t_env = torch.as_tensor(phi_env, dtype=torch.float32)
    atom_t_env = torch.as_tensor(atom_env, dtype=torch.long)
    single = zftd.make_batch(fit_data[:1], [0], target_g, device)
    corr: dict[str, Any] = {}
    with torch.no_grad():
        codes_raw = mine.local_tuple.root_codes(single)
        ref_raw = mlpmod.compose_root_codes_m(
            mine.local_tuple.A_raw.cpu(), phi_t_env, atom_t_env, payload.pair_ptr,
            payload.pair_t, payload.pair_a, payload.pair_wJ, global_roots,
            payload.phi_mean, payload.phi_std, payload.phi_scale,
        )
        corr["raw_root_codes_vs_reference_max_abs"] = float(
            (codes_raw.cpu() - ref_raw).abs().max())
        # DICT: decode the same global rows, then compose with phi_hat
        rows = phi_t_env[global_roots]
        phi_hat_rows = model_dict.local_tuple.cssd_decode(rows)
        ref_dict = mlpmod.compose_root_codes_m(
            model_dict.local_tuple.A_raw.cpu(), phi_hat_rows.cpu(), atom_t_env, payload.pair_ptr,
            payload.pair_t, payload.pair_a, payload.pair_wJ, global_roots,
            payload.phi_mean, payload.phi_std, payload.phi_scale,
        )
        codes_dict = model_dict.local_tuple.root_codes(single)
        corr["dict_root_codes_vs_decoded_reference_max_abs"] = float(
            (codes_dict.cpu() - ref_dict).abs().max())
    checks["root_codes_correspondence"] = corr
    if max(corr.values()) > 1e-5:
        raise RuntimeError(f"root codes do not match the reference operators: {corr}")

    # (j) gradient probes: W_loc and the local encoder receive task gradients
    #     (first-step zero upstream gradient through W_loc=0 is the known null)
    ell = np.asarray(objects["targets"]["ell"], np.float64)
    s_t = np.asarray(objects["targets"]["s"], np.float64)
    target_ell = torch.as_tensor(ell[fit_idx], dtype=torch.float32)
    target_s = torch.as_tensor(s_t[fit_idx], dtype=torch.float32)
    schedule, _schedule_hash = zw.build_schedule(len(fit_data), 3, SEED + TRAIN_SHUFFLE_OFFSET)
    grads: dict[str, Any] = {}
    for arm, model in (("RAW", mine), ("DICT", model_dict)):
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        frozen_before = (
            model_dict.local_tuple.frozen_basis_hashes() if arm == "DICT" else None
        )
        per_step = []
        for step in range(3):
            indices = [int(i) for i in schedule[0][step * BATCH_SIZE:(step + 1) * BATCH_SIZE]]
            index_t = torch.as_tensor(indices, dtype=torch.long)
            bstep = zftd.make_batch(fit_data, indices, target_g, device)
            model.train()
            prediction = model(bstep, mask=cm.C6_MASK)
            components = model.reader.components()
            loss = F.l1_loss(prediction.view(-1), bstep.y.view(-1)) + COMPONENT_LOSS_WEIGHT * (
                F.l1_loss(components[:, 0], target_ell[index_t])
                + F.l1_loss(components[:, 1], target_s[index_t])
            )
            optimizer.zero_grad()
            loss.backward()
            total_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP))
            optimizer.step()
            per_step.append({
                "step": step + 1,
                "loss": float(loss.detach()),
                "grad_norm": total_norm,
                "W_loc_grad_norm": float(model.local_tuple.W_loc.grad.norm()),
                "A_grad_norm": float(model.local_tuple.A_raw.grad.norm()),
                "W_loc_norm": float(model.local_tuple.W_loc.detach().norm()),
            })
        frozen_after = (
            model_dict.local_tuple.frozen_basis_hashes() if arm == "DICT" else None
        )
        grads[arm] = {
            "steps": per_step,
            "frozen_basis_unchanged": True if arm == "RAW" else bool(frozen_before == frozen_after),
            "note": "step-1 A_grad == 0 through the zero-init W_loc is the known null, not dead",
        }
        if per_step[0]["W_loc_grad_norm"] <= 0.0:
            raise RuntimeError(f"{arm}: W_loc received no task gradient at step 1")
        if per_step[-1]["A_grad_norm"] <= 0.0:
            raise RuntimeError(f"{arm}: the local encoder received no task gradient by step 3")
        if arm == "DICT" and frozen_before != frozen_after:
            raise RuntimeError("frozen basis changed during the gradient probe")
    checks["gradient_probes"] = grads

    checks["all_passed"] = True
    checks["seconds"] = float(time.perf_counter() - started)
    checks["official_valid_loaded"] = False
    checks["official_test_loaded"] = False
    write_json(out_dir / "checks.json", checks)
    log(f"[checks] all passed in {checks['seconds']:.1f}s")
    return checks


# ---------------------------------------------------------------------------
# 5. stage: fit-only interface probe (<=256 molecules, ascending gid)
# ---------------------------------------------------------------------------

def interface_probe(*, out_dir: Path = RESULTS_DIR, log: Any = print) -> dict[str, Any]:
    torch.set_num_threads(8)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    device = torch.device("cpu")
    round_objects = load_round_objects()
    objects = round_objects["objects"]
    basis = round_objects["basis"]
    basis_parts = frozen_basis_parts(basis)
    payload = prev.TuplePayload(objects["payload_arrays"])
    kappa_M = float(objects["kappa"]["kappa_M"])
    fit_idx = round_objects["fit_idx"]
    _prep_meta, fit_data, _dev_data = build_round_data(round_objects)

    sample = _fit_sample(objects, PROBE_MOLECULES)
    rows, rows_checks = _phi_rows_for(sample.tolist())
    mean_codes = fit_mean_codes(basis, fit_idx)
    z, phi_hat, rel = zreuse._root_codes(basis, rows)
    alpha = z[:, COMMON_DIM:]
    mean_alpha = np.asarray(mean_codes["mean_alpha"], np.float64)
    phi_hat_mean = (
        (torch.as_tensor(rows, dtype=torch.float32) @ torch.as_tensor(basis["U"], dtype=torch.float32))
        @ torch.as_tensor(basis["U"], dtype=torch.float32).t()
        + torch.as_tensor(mean_alpha, dtype=torch.float32)
        @ zreuse._frozen_code_parts(basis)[2].t()
    ).numpy().astype(np.float64)
    input_stats = {
        "n_molecules": int(sample.size),
        "phi_rows": rows_checks,
        "phi_hat_vs_phi_rel_err_median": float(np.median(rel)),
        "phi_hat_vs_phi_rel_err_p95": float(np.percentile(rel, 95)),
        "alpha_nnz_mean": float((alpha != 0).sum(axis=1).mean()),
        "alpha_vs_fit_mean_l2_median": float(np.median(
            np.linalg.norm(alpha - mean_alpha[None, :], axis=1))),
        "alpha_vs_fit_mean_l2_p95": float(np.percentile(
            np.linalg.norm(alpha - mean_alpha[None, :], axis=1), 95)),
        "phi_hat_meanalpha_vs_phi_hat_abs_median": float(np.median(np.abs(phi_hat_mean - phi_hat))),
        "phi_hat_meanalpha_vs_phi_hat_abs_max": float(np.abs(phi_hat_mean - phi_hat).max()),
    }

    # fresh-init neutrality: W_loc = 0 nulls the local channel exactly, so the
    # fresh RAW and DICT consumers predict identically at initialisation
    seed_everything(SEED)
    model_raw = build_arm("RAW", payload, kappa_M, basis_parts, SEED)
    seed_everything(SEED)
    model_dict = build_arm("DICT", payload, kappa_M, basis_parts, SEED)
    positions = [int(np.where(fit_idx == int(i))[0][0]) for i in sample.tolist()]
    sample_data = [fit_data[p] for p in positions]
    model_raw.eval()
    model_dict.eval()
    h_raw, _c_raw = zcs.evaluate_state_components(model_raw, sample_data, device)
    h_dict, _c_dict = zcs.evaluate_state_components(model_dict, sample_data, device)
    fresh = {
        "raw_dict_max_abs_pred_diff": float(np.abs(h_raw - h_dict).max()),
        "note": "expected exactly 0: the zero-init W_loc nulls the local channel at init",
    }
    if fresh["raw_dict_max_abs_pred_diff"] != 0.0:
        raise RuntimeError("fresh RAW/DICT predictions differ at init (W_loc must null the channel)")

    # read-only probe with the historical strong M_COMP soup (its own payload
    # and prep; diagnostic only, never a gate and never generalisation)
    old_soup: dict[str, Any] = {"available": bool(OLD_SOUP_PATH.exists())}
    if OLD_SOUP_PATH.exists():
        old = zldc.load_new_objects(OLD_SOUP_DIR)
        payload_old = prev.TuplePayload(old["payload_arrays"])
        kappa_old = float(old["kappa"]["kappa_M"])
        old_meta = read_json(OLD_SOUP_DIR / "M_COMP_meta.json")
        soup = torch.load(OLD_SOUP_PATH, map_location="cpu", weights_only=False)
        if state_hash(soup) != old_meta["raw_soup_state_hash"]:
            raise RuntimeError("old M_COMP soup hash mismatch")
        _pm, old_fit_data, _old_dev = zldc.build_prepared_data(old, verify_prep=True)
        old_fit_idx = np.asarray(old["fold"]["fit_idx"], np.int64)
        old_gid = np.asarray(old["targets"]["gid"], np.int64)
        old_sample = old_fit_idx[np.argsort(old_gid[old_fit_idx], kind="stable")][:PROBE_MOLECULES]
        old_positions = [int(np.where(old_fit_idx == int(i))[0][0]) for i in old_sample.tolist()]
        old_sample_data = [old_fit_data[p] for p in old_positions]
        old_g = np.asarray(old["targets"]["g"], np.float64)[old_sample]
        seed_everything(0)
        old_model = zldc.build_arm_comp("M_COMP", payload_old, kappa_old)
        old_model.load_state_dict({k: v for k, v in soup.items()}, strict=True)
        old_model.eval()
        h_base, _ = zcs.evaluate_state_components(old_model, old_sample_data, device)
        original = old_model.local_tuple
        encoder = LocalTupleEncoderMCSSD(
            payload_old, original.A_raw.detach().clone(), basis_parts
        )
        with torch.no_grad():
            encoder.W_loc.copy_(original.W_loc)
        encoder.kappa = float(original.kappa)
        old_model.local_tuple = encoder
        old_model.eval()
        h_dict_hat, _ = zcs.evaluate_state_components(old_model, old_sample_data, device)
        encoder.alpha_replace = torch.as_tensor(mean_alpha, dtype=torch.float32)
        h_dict_mean, _ = zcs.evaluate_state_components(old_model, old_sample_data, device)
        encoder.alpha_replace = None
        encoder.decode_enabled = False
        h_identity, _ = zcs.evaluate_state_components(old_model, old_sample_data, device)

        def stats(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
            d = np.abs(a - b)
            return {
                "mean": float(d.mean()), "median": float(np.median(d)),
                "p95": float(np.percentile(d, 95)), "max": float(d.max()),
                "fraction_gt_1e-4": float((d > REPLAY_TOL).mean()),
            }

        old_soup = {
            "available": True,
            "n_molecules": int(old_sample.size),
            "base_g_mae": float(np.abs(old_g - h_base).mean()),
            "dict_hat": {
                "abs_dpred": stats(h_dict_hat, h_base),
                "delta_g_mae": float(np.abs(old_g - h_dict_hat).mean() - np.abs(old_g - h_base).mean()),
            },
            "dict_mean_alpha": {
                "abs_dpred": stats(h_dict_mean, h_base),
                "delta_g_mae": float(np.abs(old_g - h_dict_mean).mean() - np.abs(old_g - h_base).mean()),
            },
            "identity_hook_abs_dpred_max": float(np.abs(h_identity - h_base).max()),
            "reading": (
                "interface diagnostic only: a weak response here never vetoes the "
                "from-scratch formal runs, and fit numbers are never generalisation"
            ),
        }

    probe = {
        "protocol_version": PROTOCOL_VERSION,
        "stage": "probe",
        "sample_rule": f"first {PROBE_MOLECULES} fit molecules in ascending gid order (label-blind)",
        "input_stats": input_stats,
        "fresh_init": fresh,
        "old_m_comp_soup": old_soup,
        "seconds": float(time.perf_counter() - started),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(out_dir / "interface_probe.json", probe)
    log(
        f"[probe] rel-err median {input_stats['phi_hat_vs_phi_rel_err_median']:.4f}; "
        f"old-soup |dPred| p95 "
        f"{(old_soup.get('dict_hat', {}).get('abs_dpred', {}).get('p95', float('nan'))):.2e}"
    )
    return probe


# ---------------------------------------------------------------------------
# 6. stage: GPU smoke (<= 4 batches per arm x both seeds)
# ---------------------------------------------------------------------------

def run_smoke(
    *,
    device_name: str = "cuda:0",
    n_batches: int = 4,
    out_dir: Path = RESULTS_DIR,
    log: Any = print,
) -> dict[str, Any]:
    device = resolve_device(device_name)
    torch.set_num_threads(8 if device.type == "cpu" else 4)
    started = time.perf_counter()
    round_objects = load_round_objects()
    objects = round_objects["objects"]
    basis_parts = frozen_basis_parts(round_objects["basis"])
    payload = prev.TuplePayload(objects["payload_arrays"])
    kappa_M = float(objects["kappa"]["kappa_M"])
    _prep_meta, fit_data, _dev_data = build_round_data(round_objects)
    fit_idx = round_objects["fit_idx"]
    g = np.asarray(objects["targets"]["g"], np.float64)[fit_idx]
    ell = np.asarray(objects["targets"]["ell"], np.float64)[fit_idx]
    s_t = np.asarray(objects["targets"]["s"], np.float64)[fit_idx]
    target_g = torch.as_tensor(g, dtype=torch.float32, device=device)
    target_ell = torch.as_tensor(ell, dtype=torch.float32, device=device)
    target_s = torch.as_tensor(s_t, dtype=torch.float32, device=device)

    smoke: dict[str, Any] = {"device": str(device), "n_batches": int(n_batches), "cells": {}}
    rng_hashes: dict[int, dict[str, str]] = {0: {}, 1: {}}
    schedule_hashes: dict[int, str] = {}
    for seed in SEEDS:
        schedule, schedule_hash = zw.build_schedule(len(fit_data), EPOCHS, int(seed) + TRAIN_SHUFFLE_OFFSET)
        schedule_hashes[int(seed)] = schedule_hash
        for arm in ARMS:
            cell_started = time.perf_counter()
            seed_everything(seed)
            model = build_arm(arm, payload, kappa_M, basis_parts, seed).to(device)
            off_device = [
                n for n, t in list(model.named_parameters()) + list(model.named_buffers())
                if t.device.type != device.type
            ]
            if off_device:
                raise RuntimeError(f"{arm} s{seed} tensors not on {device}: {off_device}")
            optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            frozen_before = model.local_tuple.frozen_basis_hashes() if arm == "DICT" else None
            losses = []
            rng_after: list[str] = []
            identity_gap_max = 0.0
            for step in range(int(n_batches)):
                indices = [int(i) for i in schedule[0][step * BATCH_SIZE:(step + 1) * BATCH_SIZE]]
                index_t = torch.as_tensor(indices, dtype=torch.long, device=device)
                batch = zftd.make_batch(fit_data, indices, target_g, device)
                model.train()
                prediction = model(batch, mask=cm.C6_MASK)
                components = model.reader.components()
                identity_gap_max = max(
                    identity_gap_max,
                    float(torch.max(torch.abs(components.sum(-1) - prediction.detach())).item()),
                )
                loss = F.l1_loss(prediction.view(-1), batch.y.view(-1)) + COMPONENT_LOSS_WEIGHT * (
                    F.l1_loss(components[:, 0], target_ell[index_t])
                    + F.l1_loss(components[:, 1], target_s[index_t])
                )
                if not bool(torch.isfinite(loss).item()):
                    raise RuntimeError(f"{arm} s{seed} non-finite loss at step {step}")
                optimizer.zero_grad()
                loss.backward()
                total_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP))
                if not math.isfinite(total_norm):
                    raise RuntimeError(f"{arm} s{seed} non-finite grad norm at step {step}")
                optimizer.step()
                losses.append(float(loss.detach()))
                rng_after.append(hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest())
            frozen_after = model.local_tuple.frozen_basis_hashes() if arm == "DICT" else None
            if arm == "DICT" and frozen_before != frozen_after:
                raise RuntimeError(f"{arm} s{seed}: frozen basis changed during smoke")
            rng_hashes[seed][arm] = rng_after[-1]
            smoke["cells"][f"{arm}_s{seed}"] = {
                "losses": losses,
                "component_identity_max_abs": identity_gap_max,
                "frozen_basis_unchanged": True if arm == "RAW" else bool(frozen_before == frozen_after),
                "seconds": float(time.perf_counter() - cell_started),
                "local_channel_stats": {
                    k: float(v) for k, v in model.local_tuple.last_stats.items()
                    if isinstance(v, (int, float, bool))
                },
            }
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    smoke["rng_plan"] = {
        str(seed): {
            "raw_equals_dict_after_steps": rng_hashes[seed]["RAW"] == rng_hashes[seed]["DICT"],
            "raw_rng_sha": rng_hashes[seed]["RAW"],
            "dict_rng_sha": rng_hashes[seed]["DICT"],
        }
        for seed in SEEDS
    }
    smoke["schedule_hashes"] = schedule_hashes
    if not all(v["raw_equals_dict_after_steps"] for v in smoke["rng_plan"].values()):
        raise RuntimeError("the deterministic CSSD encoding consumed training RNG")
    smoke["peak_gpu_memory_mb"] = (
        float(torch.cuda.max_memory_allocated() / (1 << 20)) if device.type == "cuda" else 0.0
    )
    smoke["all_finite"] = True
    smoke["seconds"] = float(time.perf_counter() - started)
    smoke["official_valid_loaded"] = False
    smoke["official_test_loaded"] = False
    write_json(out_dir / "smoke.json", smoke)
    log(f"[smoke] 4 cells ok on {device} in {smoke['seconds']:.1f}s")
    return smoke


# ---------------------------------------------------------------------------
# 7. stage: one formal body run (240 epochs, from-scratch init)
# ---------------------------------------------------------------------------

def _probe_local_channel(model: nn.Module, arm: str, epoch: int, step: int, total_norm: float) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "arm": arm, "epoch": int(epoch), "step_in_epoch": int(step),
        "clip_total_norm": float(total_norm),
    }
    enc = model.local_tuple
    entry["W_loc_grad_norm"] = float(enc.W_loc.grad.norm()) if enc.W_loc.grad is not None else None
    entry["W_loc_norm"] = float(enc.W_loc.detach().norm())
    entry["A_grad_norm"] = float(enc.A_raw.grad.norm()) if enc.A_raw.grad is not None else None
    entry["A_norm"] = float(enc.A_raw.detach().norm())
    entry["encoder_stats"] = {
        key: value for key, value in enc.last_stats.items()
        if isinstance(value, (int, float, bool))
    }
    return entry


def train_arm(
    arm: str,
    seed: int,
    *,
    device_name: str = "cuda:0",
    out_dir: Path = RESULTS_DIR,
    epochs: int = EPOCHS,
    max_steps: int | None = None,
    log: Any = print,
) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError(arm)
    if int(seed) not in SEEDS:
        raise ValueError(seed)
    device = resolve_device(device_name)
    torch.set_num_threads(8 if device.type == "cpu" else 4)
    out_dir = Path(out_dir)
    rdir = run_dir(arm, seed, out_dir)
    rdir.mkdir(parents=True, exist_ok=True)
    round_objects = load_round_objects()
    objects = round_objects["objects"]
    basis = round_objects["basis"]
    basis_parts = frozen_basis_parts(basis)
    q_soup = round_objects["q_soup"]
    payload = prev.TuplePayload(objects["payload_arrays"])
    kappa_M = float(objects["kappa"]["kappa_M"])
    fit_idx = round_objects["fit_idx"]
    _prep_meta, fit_data, _dev_data = build_round_data(round_objects)
    y_all = np.asarray(objects["targets"]["y"], np.float64)
    g_all = np.asarray(objects["targets"]["g"], np.float64)
    ell_all = np.asarray(objects["targets"]["ell"], np.float64)
    s_all = np.asarray(objects["targets"]["s"], np.float64)
    gid_all = np.asarray(objects["targets"]["gid"], np.int64)
    y_fit, g_fit = y_all[fit_idx], g_all[fit_idx]
    ell_fit, s_fit = ell_all[fit_idx], s_all[fit_idx]
    if float(np.max(np.abs(g_fit - (ell_fit + s_fit)))) > 1e-12:
        raise RuntimeError("g != ell + s on fit rows")
    target_g = torch.as_tensor(g_fit, dtype=torch.float32)
    target_ell = torch.as_tensor(ell_fit, dtype=torch.float32, device=device)
    target_s = torch.as_tensor(s_fit, dtype=torch.float32, device=device)
    schedule, schedule_hash = zw.build_schedule(len(fit_data), int(epochs), int(seed) + TRAIN_SHUFFLE_OFFSET)

    seed_everything(int(seed))
    model = build_arm(arm, payload, kappa_M, basis_parts, int(seed))
    build_rng = torch.get_rng_state().clone()
    init_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model = model.to(device)
    if arm == "DICT":
        expected_encoder_hashes = {
            "U": array_sha256(basis_parts["U"].numpy()),
            "common_rms": array_sha256(basis_parts["common_rms"].numpy()),
            "Dbar": array_sha256(basis_parts["Dbar"].numpy()),
        }
        encoder_hashes_before = model.local_tuple.frozen_basis_hashes()
        if encoder_hashes_before != expected_encoder_hashes:
            raise RuntimeError("DICT frozen basis buffers != loaded basis package")
    else:
        encoder_hashes_before = None
    seed_everything(int(seed))
    train_rng = torch.get_rng_state().clone()

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
    position_stream = hashlib.sha256()
    gid_stream = hashlib.sha256()

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
            position_stream.update(np.asarray(indices, np.int64).tobytes())
            gid_stream.update(np.asarray(gid_all[fit_idx[np.asarray(indices, np.int64)]], np.int64).tobytes())
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
                probes.append(_probe_local_channel(model, arm, epoch, steps_done + 1, total_norm))
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
            log(
                f"[train {arm} s{seed}] ep={epoch:03d} Lg={c['train_L_g']:.5f} "
                f"Lell={c['train_L_ell']:.5f} Ls={c['train_L_s']:.5f} "
                f"gn={c['grad_norm']:.3g} {c['seconds']:.1f}s",
                flush=True,
            )
        if stopped_reason == "max_steps":
            break

    members = sorted(soup)
    if not members:
        # short smoke runs (max_steps) never reach the soup epochs; take the
        # final state so the downstream manifest/soup path stays exercisable
        soup[int(epoch)] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        members = sorted(soup)
    if max_steps is None and members != [int(e) for e in SOUP_EPOCHS]:
        raise RuntimeError(f"soup epochs missing: {members}")
    soup_state = {key: torch.stack([soup[e][key].float() for e in members]).mean(0) for key in soup[members[0]]}
    soup_hash = state_hash(soup_state)
    init_hash = state_hash(init_state)
    last_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # the batch-order plan and training RNG, saved explicitly
    np.savez_compressed(
        rdir / "schedule.npz",
        order=np.stack([np.asarray(e, np.int64) for e in schedule]),
    )

    # fit-only diagnostics from the soup (never dev, never a selection)
    replay = build_arm(arm, payload, kappa_M, basis_parts, int(seed))
    replay.load_state_dict({k: v for k, v in soup_state.items()}, strict=True)
    replay = replay.to(device)
    replay.eval()
    h, comps = zcs.evaluate_state_components(replay, fit_data, device)
    ell_hat, s_hat = comps[:, 0], comps[:, 1]
    T_topology = parent.topology_matrix(fit_data)
    q_raw = parent._q_predictions(q_soup, T_topology, device)
    y_raw = ell_hat + s_hat + q_raw
    if float(np.max(np.abs(y_raw - (h + q_raw)))) > 1e-5:
        raise RuntimeError("y_raw != h + Q_raw on fit rows")
    b_y = float(np.median(y_fit - y_raw))
    encoder_hashes_after = replay.local_tuple.frozen_basis_hashes() if arm == "DICT" else None
    if arm == "DICT" and encoder_hashes_before != encoder_hashes_after:
        raise RuntimeError(f"{arm} s{seed}: frozen basis changed during training")

    torch.save(init_state, rdir / "init_state.pt")
    torch.save(last_state, rdir / "last_state.pt")
    torch.save(soup_state, rdir / "soup_state.pt")
    for epoch, state in sorted(checkpoints.items()):
        torch.save(state, rdir / f"epoch{epoch}_state.pt")
    np.savez_compressed(
        rdir / "fit_predictions.npz",
        gid=gid_all[fit_idx], y=y_fit, g=g_fit, ell=ell_fit, s=s_fit,
        h=h, ell_hat=ell_hat, s_hat=s_hat, q_raw=q_raw, y_raw=y_raw,
    )
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "stage": "train",
        "arm": arm,
        "seed": int(seed),
        "run_dir": str(rdir.relative_to(out_dir)),
        "recipe": {
            "epochs": int(epochs), "batch_size": BATCH_SIZE, "lr": LR,
            "weight_decay": WEIGHT_DECAY, "grad_clip": GRAD_CLIP,
            "soup_epochs": list(SOUP_EPOCHS), "log_epochs": list(LOG_EPOCHS),
            "component_loss_weight": COMPONENT_LOSS_WEIGHT,
            "train_shuffle_offset": TRAIN_SHUFFLE_OFFSET,
            "amp": None, "ddp": None, "scheduler": None, "early_stopping": None,
        },
        "supervision": "COMP: L = MAE(ell_hat+s_hat, g) + 0.5*(MAE(ell_hat,ell)+MAE(s_hat,s)) (disclosed auxiliary condition)",
        "steps_done": int(steps_done),
        "steps_expected": int(sum((len(e) + BATCH_SIZE - 1) // BATCH_SIZE for e in schedule)),
        "stopped_reason": stopped_reason,
        "schedule_sha256": schedule_hash,
        "position_stream_sha256": position_stream.hexdigest(),
        "global_gid_stream_sha256": gid_stream.hexdigest(),
        "init_state_sha256": init_hash,
        "soup_state_sha256": soup_hash,
        "last_state_sha256": state_hash(last_state),
        "build_rng_sha256": hashlib.sha256(build_rng.numpy().tobytes()).hexdigest(),
        "train_rng_sha256": hashlib.sha256(train_rng.numpy().tobytes()).hexdigest(),
        "frozen_basis_hashes": basis_parts["hashes"],
        "encoder_frozen_basis_hashes": encoder_hashes_after,
        "frozen_basis_unchanged": True if arm == "RAW" else bool(encoder_hashes_before == encoder_hashes_after),
        "q_soup_sha256": state_hash(q_soup),
        "parameter_audit": arm_parameter_audit(replay),
        "calibration": {
            "b_y": b_y,
            "note": "single fit-median b_y = median(y_fit - y_raw_fit); never stacked with any old bias",
        },
        "fit_diagnostics": {
            "fit_y_raw_mae": float(np.mean(np.abs(y_fit - y_raw))),
            "fit_y_cal_mae": float(np.mean(np.abs(y_fit - (y_raw + b_y)))),
            "fit_g_raw_mae": float(np.mean(np.abs(g_fit - h))),
            "fit_ell_mae": float(np.mean(np.abs(ell_fit - ell_hat))),
            "fit_s_mae": float(np.mean(np.abs(s_fit - s_hat))),
            "fit_q_mae": float(np.mean(np.abs(objects["targets"]["c"][fit_idx] - q_raw))),
        },
        "curve_seconds_total": float(sum(c["seconds"] for c in curve)),
        "wall_clock_s": float(time.perf_counter() - started),
        "peak_gpu_memory_mb": float(peak_mb),
        "device": str(device),
        "allocation_probe": allocation_probe(device),
        "dev_scores_computed_during_training": False,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(rdir / "curve.json", curve)
    write_json(rdir / "probes.json", probes)
    write_json(rdir / "manifest.json", manifest)
    log(
        f"[train {arm} s{seed}] done: {steps_done} steps, {manifest['curve_seconds_total']:.0f}s "
        f"train, fit y_raw MAE {manifest['fit_diagnostics']['fit_y_raw_mae']:.5f} "
        f"b_y={b_y:.5f} soup sha={soup_hash[:12]}…"
    )
    return manifest


# ---------------------------------------------------------------------------
# 8. one-shot terminal stage: dev scoring, interventions, decision rules
# ---------------------------------------------------------------------------

def _predict_rows(model: nn.Module, eval_rows: Sequence[Any], device: torch.device) -> dict[str, np.ndarray]:
    """Batched predictions: h, ell_hat, s_hat (identical mask, no labels)."""
    hs, ells, ss = [], [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(eval_rows), BATCH_SIZE):
            chunk = list(eval_rows[start:start + BATCH_SIZE])
            batch = zftd.make_batch(eval_rows, list(range(start, start + len(chunk))), torch.zeros(len(eval_rows)), device)
            prediction = model(batch, mask=cm.C6_MASK)
            components = model.reader.components()
            hs.append(prediction.detach().cpu().numpy().astype(np.float64))
            ells.append(components[:, 0].detach().cpu().numpy().astype(np.float64))
            ss.append(components[:, 1].detach().cpu().numpy().astype(np.float64))
    return {"h": np.concatenate(hs), "ell_hat": np.concatenate(ells), "s_hat": np.concatenate(ss)}


def _dpred_stats(new: np.ndarray, base: np.ndarray) -> dict[str, float]:
    d = np.abs(np.asarray(new, np.float64) - np.asarray(base, np.float64))
    return {
        "mean": float(d.mean()),
        "median": float(np.median(d)),
        "p95": float(np.percentile(d, 95)),
        "max": float(d.max()),
        "fraction_gt_1e-4": float((d > REPLAY_TOL).mean()),
    }


@torch.no_grad()
def evaluate_run(
    arm: str,
    seed: int,
    *,
    out_dir: Path = RESULTS_DIR,
    device_name: str = "cuda:0",
    round_objects: Mapping[str, Any] | None = None,
    data: tuple[list[Any], list[Any]] | None = None,
    log: Any = print,
) -> dict[str, Any]:
    device = resolve_device(device_name)
    torch.set_num_threads(8 if device.type == "cpu" else 4)
    out_dir = Path(out_dir)
    if round_objects is None:
        round_objects = load_round_objects()
    objects = round_objects["objects"]
    basis_parts = frozen_basis_parts(round_objects["basis"])
    payload = prev.TuplePayload(objects["payload_arrays"])
    kappa_M = float(objects["kappa"]["kappa_M"])
    run = load_run(arm, seed, out_dir)
    if data is None:
        _pm, _fit_data, dev_data = build_round_data(round_objects)
    else:
        dev_data = data[1]
    model = build_arm(arm, payload, kappa_M, basis_parts, int(seed))
    model.load_state_dict({k: v for k, v in run["soup_state"].items()}, strict=True)
    model = model.to(device)
    model.eval()
    preds = _predict_rows(model, dev_data, device)
    T_topology = parent.topology_matrix(dev_data)
    q_raw = parent._q_predictions(round_objects["q_soup"], T_topology, device)
    y_raw = preds["ell_hat"] + preds["s_hat"] + q_raw
    if float(np.max(np.abs(y_raw - (preds["h"] + q_raw)))) > 1e-5:
        raise RuntimeError("y_raw != h + Q_raw on dev")
    b_y = float(run["manifest"]["calibration"]["b_y"])
    y_cal = y_raw + b_y
    dev_idx = round_objects["dev_idx"]
    targets = objects["targets"]
    y = np.asarray(targets["y"], np.float64)[dev_idx]
    g = np.asarray(targets["g"], np.float64)[dev_idx]
    ell = np.asarray(targets["ell"], np.float64)[dev_idx]
    s_t = np.asarray(targets["s"], np.float64)[dev_idx]
    gid = np.asarray(targets["gid"], np.int64)[dev_idx]
    result = {
        "arm": arm, "seed": int(seed), "n": int(len(dev_data)),
        "gid": gid, "y": y, "g": g, "ell": ell, "s": s_t,
        "y_raw": y_raw, "y_cal": y_cal, "b_y": b_y,
        "h": preds["h"], "ell_hat": preds["ell_hat"], "s_hat": preds["s_hat"], "q_raw": q_raw,
        "mae": {
            "y_raw": float(np.mean(np.abs(y - y_raw))),
            "y_cal": float(np.mean(np.abs(y - y_cal))),
            "g_raw": float(np.mean(np.abs(g - preds["h"]))),
            "ell": float(np.mean(np.abs(ell - preds["ell_hat"]))),
            "s": float(np.mean(np.abs(s_t - preds["s_hat"]))),
        },
        "fit_diagnostics": run["manifest"]["fit_diagnostics"],
        "curve_seconds_total": run["manifest"]["curve_seconds_total"],
        "soup_state_sha256": run["manifest"]["soup_state_sha256"],
    }
    if log:
        log(
            f"[dev-eval {arm} s{seed}] y_raw MAE {result['mae']['y_raw']:.5f} "
            f"y_cal {result['mae']['y_cal']:.5f} g_raw {result['mae']['g_raw']:.5f}"
        )
    return result


def group_paired_bootstrap(
    errs: Mapping[str, np.ndarray],
    group_of_row: np.ndarray,
    *,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> dict[str, Any]:
    """SMILES-group-paired bootstrap of the DICT - RAW contrast.

    Whole canonical-SMILES groups of dev are resampled with replacement
    (2000 draws, seed 20261012); the SAME group resampling is shared by both
    arms and both seeds, and each draw first averages the two seeds' MAE
    differences before the CI is taken.  The molecular bootstrap carries no
    training-seed or basis uncertainty.
    """
    keys = ["RAW_s0", "DICT_s0", "RAW_s1", "DICT_s1"]
    for key in keys:
        if key not in errs:
            raise RuntimeError(f"bootstrap expects {keys}, missing {key}")
        if np.asarray(errs[key]).shape != np.asarray(group_of_row).shape:
            raise RuntimeError(f"row misalignment at {key}")
    labels = np.asarray([str(v) for v in np.asarray(group_of_row, dtype=object).tolist()], dtype=object)
    order = {label: index for index, label in enumerate(sorted(set(labels.tolist())))}
    g_index = np.asarray([order[label] for label in labels.tolist()], np.int64)
    n_groups = len(order)
    sums = {k: np.zeros(n_groups, np.float64) for k in keys}
    cnt = np.zeros(n_groups, np.int64)
    np.add.at(cnt, g_index, 1)
    for k in keys:
        np.add.at(sums[k], g_index, np.asarray(errs[k], np.float64))
    rng = np.random.default_rng(int(seed))
    avg = np.empty(int(n_boot))
    per_seed = {0: np.empty(int(n_boot)), 1: np.empty(int(n_boot))}
    for b in range(int(n_boot)):
        pick = rng.integers(0, n_groups, n_groups)
        n = int(cnt[pick].sum())
        m = {k: float(sums[k][pick].sum() / n) for k in keys}
        d0 = m["DICT_s0"] - m["RAW_s0"]
        d1 = m["DICT_s1"] - m["RAW_s1"]
        per_seed[0][b] = d0
        per_seed[1][b] = d1
        avg[b] = 0.5 * (d0 + d1)
    return {
        "n_rows": int(cnt.sum()),
        "n_groups": int(n_groups),
        "n_boot": int(n_boot),
        "boot_seed": int(seed),
        "shared_group_resampling": True,
        "avg_delta_ci95": [float(np.percentile(avg, 2.5)), float(np.percentile(avg, 97.5))],
        "per_seed_delta_ci95": {
            str(seed_): [float(np.percentile(per_seed[seed_], 2.5)), float(np.percentile(per_seed[seed_], 97.5))]
            for seed_ in (0, 1)
        },
        "reading": (
            "CI covers molecule resampling only — not training-seed or basis "
            "uncertainty; 2 body seeds are not a method-stability claim"
        ),
    }


def run_alpha_intervention(
    seed: int,
    base: Mapping[str, Any],
    mean_alpha: np.ndarray,
    *,
    out_dir: Path = RESULTS_DIR,
    device_name: str = "cuda:0",
    round_objects: Mapping[str, Any] | None = None,
    data: tuple[list[Any], list[Any]] | None = None,
    log: Any = print,
) -> dict[str, Any]:
    """The unique dictionary intervention: alpha_v := fit mean alpha, c kept.

    The patch sits at the single local structure input: every root's sparse
    residual code is replaced by the same fit-root mean alpha while the
    common coefficient c (and the chemistry one-hots, the J incidence, the
    Sem108 interface, the reader and the frozen Q) are untouched; phi_hat is
    re-decoded and re-forwarded; no retraining, no recalibration.
    Dependence evidence, never incremental benefit.
    """
    device = resolve_device(device_name)
    if round_objects is None:
        round_objects = load_round_objects()
    objects = round_objects["objects"]
    basis = round_objects["basis"]
    basis_parts = frozen_basis_parts(basis)
    payload = prev.TuplePayload(objects["payload_arrays"])
    kappa_M = float(objects["kappa"]["kappa_M"])
    if data is None:
        _pm, _fit_data, dev_data = build_round_data(round_objects)
    else:
        dev_data = data[1]
    run = load_run("DICT", seed, out_dir)
    model = build_arm("DICT", payload, kappa_M, basis_parts, int(seed))
    model.load_state_dict({k: v for k, v in run["soup_state"].items()}, strict=True)
    model = model.to(device)
    model.eval()
    model.local_tuple.alpha_replace = torch.as_tensor(np.asarray(mean_alpha, np.float64), dtype=torch.float32)
    preds = _predict_rows(model, dev_data, device)
    model.local_tuple.alpha_replace = None

    # input-level change actually applied (frozen math, outside the model)
    dev_idx = round_objects["dev_idx"]
    rows, _checks = _phi_rows_for(dev_idx.tolist())
    z, phi_hat, _rel = zreuse._root_codes(basis, rows)
    alpha_dev = z[:, COMMON_DIM:]
    mean = np.asarray(mean_alpha, np.float64)
    U = np.asarray(basis["U"], np.float64)
    Dbar = np.asarray(zreuse._frozen_code_parts(basis)[2].numpy(), np.float64)
    c = rows.astype(np.float64) @ U
    phi_hat_int = c @ U.T + mean[None, :] @ Dbar.T
    alpha_dev_from_mean = np.linalg.norm(alpha_dev - mean[None, :], axis=1)
    phi_change = np.abs(phi_hat_int - phi_hat)

    q_raw = np.asarray(base["q_raw"], np.float64)
    y_raw_i = preds["ell_hat"] + preds["s_hat"] + q_raw
    y_cal_i = y_raw_i + float(base["b_y"])
    y = np.asarray(base["y"], np.float64)
    g = np.asarray(base["g"], np.float64)
    base_y_raw = np.asarray(base["y_raw"], np.float64)
    base_h = np.asarray(base["h"], np.float64)
    d_pred = np.abs(y_raw_i - base_y_raw)
    row = {
        "arm": "DICT",
        "seed": int(seed),
        "mechanism": (
            "alpha_v := the fit-root mean alpha at the single local structure input; "
            "c_v kept; chemistry one-hots, J incidence, Sem108, reader and the frozen Q "
            "untouched; re-decoded and re-forwarded; no retraining, no recalibration"
        ),
        "input_change": {
            "alpha_vs_mean_l2_median": float(np.median(alpha_dev_from_mean)),
            "alpha_vs_mean_l2_p95": float(np.percentile(alpha_dev_from_mean, 95)),
            "alpha_vs_mean_l2_max": float(alpha_dev_from_mean.max()),
            "phi_hat_abs_change_median": float(np.median(phi_change)),
            "phi_hat_abs_change_max": float(phi_change.max()),
        },
        "abs_dpred": {
            "mean": float(d_pred.mean()),
            "median": float(np.median(d_pred)),
            "p95": float(np.percentile(d_pred, 95)),
            "max": float(d_pred.max()),
        },
        "response_fraction_gt_1e-4": float((d_pred > REPLAY_TOL).mean()),
        "base_y_raw_mae": float(np.mean(np.abs(y - base_y_raw))),
        "y_raw_mae": float(np.mean(np.abs(y - y_raw_i))),
        "y_cal_mae": float(np.mean(np.abs(y - y_cal_i))),
        "delta_y_raw_mae": float(np.mean(np.abs(y - y_raw_i)) - np.mean(np.abs(y - base_y_raw))),
        "delta_y_cal_mae": float(
            np.mean(np.abs(y - y_cal_i)) - np.mean(np.abs(y - (base_y_raw + float(base["b_y"]))))
        ),
        "delta_g_mae": float(np.mean(np.abs(g - preds["h"])) - np.mean(np.abs(g - base_h))),
        "g_note": "Q input unchanged by construction: q_raw reused bitwise from the base evaluation",
        "reading": "response shows the consumer reads the sparse code; never incremental benefit",
    }
    if log:
        log(
            f"[intervention DICT s{seed}] |dPred| p95 {row['abs_dpred']['p95']:.2e} "
            f"resp {row['response_fraction_gt_1e-4']:.3f} dMAE {row['delta_y_raw_mae']:+.2e}"
        )
    return row


def run_localization(
    arm: str,
    seed: int,
    base: Mapping[str, Any],
    mean_codes: Mapping[str, np.ndarray],
    *,
    out_dir: Path = RESULTS_DIR,
    device_name: str = "cuda:0",
    round_objects: Mapping[str, Any] | None = None,
    data: tuple[list[Any], list[Any]] | None = None,
    log: Any = print,
) -> dict[str, Any]:
    """Read-only localisation: replace the arm's whole local structure input.

    Pre-registered, conditional (only when the alpha intervention is
    unresponsive): at the SAME local interface, RAW's phi (DICT's phi_hat) is
    replaced by that arm's own fit-root mean structure input.  A sensitivity
    diagnostic only — never a new training arm, never a rescue purchase.
    """
    device = resolve_device(device_name)
    if round_objects is None:
        round_objects = load_round_objects()
    objects = round_objects["objects"]
    basis_parts = frozen_basis_parts(round_objects["basis"])
    payload = prev.TuplePayload(objects["payload_arrays"])
    kappa_M = float(objects["kappa"]["kappa_M"])
    if data is None:
        _pm, _fit_data, dev_data = build_round_data(round_objects)
    else:
        dev_data = data[1]
    run = load_run(arm, seed, out_dir)
    model = build_arm(arm, payload, kappa_M, basis_parts, int(seed))
    model.load_state_dict({k: v for k, v in run["soup_state"].items()}, strict=True)
    model = model.to(device)
    model.eval()
    a_raw = model.local_tuple.A_raw.detach().clone()
    w_loc = model.local_tuple.W_loc.detach().clone()
    kappa = float(model.local_tuple.kappa)
    mean_input = (
        np.asarray(mean_codes["mean_phi"], np.float64) if arm == "RAW"
        else np.asarray(mean_codes["mean_phi_hat"], np.float64)
    )
    encoder = LocalTupleEncoderMCSSD(payload, a_raw, basis_parts, decode=(arm == "DICT"))
    with torch.no_grad():
        encoder.W_loc.copy_(w_loc)
    encoder.kappa = kappa
    encoder.phi_replace = torch.as_tensor(mean_input, dtype=torch.float32)
    model.local_tuple = encoder
    preds = _predict_rows(model, dev_data, device)
    q_raw = np.asarray(base["q_raw"], np.float64)
    y_raw_i = preds["ell_hat"] + preds["s_hat"] + q_raw
    y = np.asarray(base["y"], np.float64)
    g = np.asarray(base["g"], np.float64)
    base_y_raw = np.asarray(base["y_raw"], np.float64)
    base_h = np.asarray(base["h"], np.float64)
    d_pred = np.abs(y_raw_i - base_y_raw)
    return {
        "arm": arm,
        "seed": int(seed),
        "mechanism": (
            f"the {arm} arm's whole local structure input (phi for RAW / phi_hat for DICT) "
            "replaced by that arm's own fit-root mean structure input at the same local "
            "interface; read-only sensitivity diagnostic, never a training arm"
        ),
        "abs_dpred": {
            "mean": float(d_pred.mean()),
            "median": float(np.median(d_pred)),
            "p95": float(np.percentile(d_pred, 95)),
            "max": float(d_pred.max()),
        },
        "response_fraction_gt_1e-4": float((d_pred > REPLAY_TOL).mean()),
        "delta_y_raw_mae": float(np.mean(np.abs(y - y_raw_i)) - np.mean(np.abs(y - base_y_raw))),
        "delta_g_mae": float(np.mean(np.abs(g - preds["h"])) - np.mean(np.abs(g - base_h))),
        "reading": (
            "responsive -> the local structure channel IS read (the common term carries it); "
            "unresponsive -> the whole local structure interface is bypassed by this consumer"
        ),
    }


def frozen_decision_rules(
    per_seed: Mapping[int, Mapping[str, float]],
    interventions: Mapping[str, Mapping[str, Any]],
    eta: float,
    bootstrap: Mapping[str, Any],
    localization: Mapping[str, Mapping[str, Any]] | None,
    *,
    seeds: Sequence[int] = SEEDS,
) -> dict[str, Any]:
    """Frozen interpretation branches A-D (pre-registered, no post-hoc tuning).

    delta_s = MAE(DICT_s) - MAE(RAW_s), positive = DICT worse.  Engineering
    tolerance for keeping DICT as a next-stage candidate: two-seed mean
    delta <= +0.003 AND each seed delta <= +0.005 (a budget rule for this
    round only, never a general equivalence threshold, and never proof of
    losslessness by itself).
    """
    seeds = tuple(int(s) for s in seeds)
    deltas = {int(seed): float(per_seed[int(seed)]["delta"]) for seed in seeds}
    mean_delta = float(np.mean([deltas[s] for s in seeds]))
    marker = float(max(REPLAY_TOL, 10.0 * float(eta)))
    within = bool(mean_delta <= TOL_AVG and all(deltas[s] <= TOL_SEED for s in seeds))
    clearly_worse = bool(mean_delta > TOL_AVG and max(deltas.values()) > TOL_SEED)
    flip = bool(len(seeds) == 2 and deltas[seeds[0]] * deltas[seeds[1]] < 0.0)
    responsive = bool(all(
        interventions[f"DICT_s{seed}"]["abs_dpred"]["p95"] > marker for seed in seeds
    ))
    conditions = {
        "deltas": {str(s): deltas[s] for s in seeds},
        "mean_delta": mean_delta,
        "tol_avg": TOL_AVG,
        "tol_seed": TOL_SEED,
        "within_tolerance": within,
        "clearly_worse": clearly_worse,
        "seed_direction_flip": flip,
        "response_marker": marker,
        "eta": float(eta),
        "alpha_responsive": responsive,
        "bootstrap_avg_delta_ci95": bootstrap["avg_delta_ci95"],
    }
    if clearly_worse:
        branch = "C"
    elif flip:
        branch = "D"
    elif within and responsive:
        branch = "A"
    elif within and not responsive:
        branch = "B"
    else:
        branch = "D"
    reasoning = {
        "A": (
            "raw performance within the engineering tolerance AND the sparse code shows "
            "real response: keep DICT as a next-stage candidate — this supports only "
            "'this effective local structure interface can be supplied by the basis', "
            "never 'all structural information flows through CSSD' and never a "
            "dictionary-beats-plain-encoding claim"
        ),
        "B": (
            "performance held but the alpha intervention is near-unresponsive: separate "
            "common-term carriage from whole-interface bypass using the localisation "
            "diagnostic; sparse-atom responsibility is NOT established — performance "
            "maintenance must not be packaged as successful reuse"
        ),
        "C": (
            "performance clearly worse: report the fit/dev gap, the reconstruction error "
            "and the consumer wiring; representation-approximation loss and training "
            "adaptation remain competing explanations — not a general verdict against "
            "dictionaries, and no new relational architecture is purchased"
        ),
        "D": (
            "seed directions flip or the evidence sits on the boundary: unconfirmed — no "
            "seed 3, no threshold change, no roster change"
        ),
    }[branch]
    recommendation = {
        "A": "keep-dict-candidate",
        "B": "sparse-responsibility-not-established",
        "C": "dict-worse-competing-explanations",
        "D": "unconfirmed-no-new-seeds",
    }[branch]
    notes: list[str] = []
    if branch == "A":
        ci = bootstrap["avg_delta_ci95"]
        if float(ci[1]) > TOL_AVG:
            notes.append(
                "the bootstrap CI95 crosses the +0.003 tolerance boundary: statistical "
                "performance carriage remains unconfirmed even though the point estimate "
                "is within budget"
            )
        notes.append(
            "satisfying the point-estimate rule or a CI crossing zero is never 'proven "
            "lossless'; calibration-only improvements never pass a raw-performance verdict"
        )
        notes.append("next research step is decided by the user, never auto-started")
    if branch == "B" and localization:
        loc_responsive = all(
            localization[f"{arm}_s{seed}"]["abs_dpred"]["p95"] > marker
            for arm in ARMS for seed in seeds
        )
        notes.append(
            "localisation diagnostic: the local structure channel IS read (the common "
            "term carries it)" if loc_responsive else
            "localisation diagnostic: the whole local structure interface is bypassed "
            "by this consumer (neither common nor sparse is read out)"
        )
    if branch == "C":
        notes.append(
            "fit/dev gap, phi_hat reconstruction error and consumer wiring are reported "
            "separately; no automatic architecture purchase"
        )
    return {
        "branch": branch,
        "recommendation": recommendation,
        "conditions": conditions,
        "reasoning": reasoning,
        "notes": notes,
        "pre_registered": True,
    }


def terminal_eval(
    *,
    out_dir: Path = RESULTS_DIR,
    device_name: str = "cuda:0",
    seeds: Sequence[int] = SEEDS,
    log: Any = print,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    if (out_dir / "terminal_eval.json").exists():
        raise RuntimeError(
            "terminal_eval.json already exists: the terminal stage is one-shot and must not be overwritten"
        )
    started = time.perf_counter()
    device = resolve_device(device_name)
    torch.set_num_threads(8 if device.type == "cpu" else 4)
    seeds = tuple(int(s) for s in seeds)
    if sorted(set(seeds)) != sorted(set(int(s) for s in SEEDS)):
        raise RuntimeError(f"terminal eval expects the pre-registered body seeds {SEEDS}")
    round_objects = load_round_objects()
    objects = round_objects["objects"]
    basis = round_objects["basis"]
    dev_idx = round_objects["dev_idx"]
    _pm, fit_data, dev_data = build_round_data(round_objects)
    data = (fit_data, dev_data)
    payload = prev.TuplePayload(objects["payload_arrays"])
    kappa_M = float(objects["kappa"]["kappa_M"])
    basis_parts = frozen_basis_parts(basis)

    # (1) roster: all four runs completed; same-seed arms share the plan
    roster: dict[str, Any] = {}
    for seed in seeds:
        for arm in ARMS:
            run = load_run(arm, seed, out_dir)
            manifest = run["manifest"]
            if manifest["stopped_reason"] != "completed":
                raise RuntimeError(f"run {arm} s{seed} did not complete")
            roster[f"{arm}_s{seed}"] = {
                "soup_state_sha256": manifest["soup_state_sha256"],
                "b_y": manifest["calibration"]["b_y"],
                "steps_done": manifest["steps_done"],
                "schedule_sha256": manifest["schedule_sha256"],
                "train_rng_sha256": manifest["train_rng_sha256"],
                "frozen_basis_hashes": manifest["frozen_basis_hashes"],
                "q_soup_sha256": manifest["q_soup_sha256"],
                "curve_seconds_total": manifest["curve_seconds_total"],
            }
    for seed in seeds:
        if roster[f"RAW_s{seed}"]["schedule_sha256"] != roster[f"DICT_s{seed}"]["schedule_sha256"]:
            raise RuntimeError(f"seed {seed} arms trained on different batch-order plans")
        if roster[f"RAW_s{seed}"]["train_rng_sha256"] != roster[f"DICT_s{seed}"]["train_rng_sha256"]:
            raise RuntimeError(f"seed {seed} arms trained on different RNG streams")

    # (2) dev scoring of all four runs (one-shot; predictions saved with gid)
    results: dict[str, dict[str, Any]] = {}
    for seed in seeds:
        for arm in ARMS:
            results[f"{arm}_s{seed}"] = evaluate_run(
                arm, seed, out_dir=out_dir, device_name=device_name,
                round_objects=round_objects, data=data, log=log,
            )
            np.savez_compressed(
                run_dir(arm, seed, out_dir) / "dev_predictions.npz",
                gid=results[f"{arm}_s{seed}"]["gid"],
                y=results[f"{arm}_s{seed}"]["y"],
                y_raw=results[f"{arm}_s{seed}"]["y_raw"],
                y_cal=results[f"{arm}_s{seed}"]["y_cal"],
                h=results[f"{arm}_s{seed}"]["h"],
                q_raw=results[f"{arm}_s{seed}"]["q_raw"],
            )

    # (3) paired deltas + shared-group bootstrap (2000 draws, seed 20261012)
    y = np.asarray(objects["targets"]["y"], np.float64)[dev_idx]
    errs = {
        f"{arm}_s{seed}": np.abs(y - np.asarray(results[f"{arm}_s{seed}"]["y_raw"], np.float64))
        for seed in seeds for arm in ARMS
    }
    per_seed: dict[int, dict[str, float]] = {}
    for seed in seeds:
        m_dict = float(errs[f"DICT_s{seed}"].mean())
        m_raw = float(errs[f"RAW_s{seed}"].mean())
        per_seed[int(seed)] = {
            "M_DICT": m_dict, "M_RAW": m_raw, "delta": m_dict - m_raw,
            "mae": {"DICT": results[f"DICT_s{seed}"]["mae"], "RAW": results[f"RAW_s{seed}"]["mae"]},
        }
    smiles = zreuse._canonical_smiles()
    bootstrap = group_paired_bootstrap(errs, smiles[dev_idx])
    avg_delta = float(np.mean([per_seed[s]["delta"] for s in seeds]))

    # (4) FP32 noise bound eta: repeated evaluation of every run (same soup,
    #     rebuilt model, second pass) + the identity hook on the trained DICT
    #     soups (the subclass machinery engaged with a pass-through input vs
    #     the same soup loaded through the parent-encoder path — both are the
    #     same pass-through forward, so this measures hook noise, never the
    #     decode effect)
    eta = 0.0
    noise: dict[str, Any] = {}
    for seed in seeds:
        for arm in ARMS:
            repeat = evaluate_run(
                arm, seed, out_dir=out_dir, device_name=device_name,
                round_objects=round_objects, data=data, log=None,
            )
            d = float(np.abs(np.asarray(repeat["y_raw"]) - np.asarray(results[f"{arm}_s{seed}"]["y_raw"])).max())
            noise[f"{arm}_s{seed}"] = {"repeat_max_abs_dpred": d}
            eta = max(eta, d)
    for seed in seeds:
        run = load_run("DICT", seed, out_dir)
        model_hook = build_arm("DICT", payload, kappa_M, basis_parts, int(seed), decode=False)
        model_hook.load_state_dict({k: v for k, v in run["soup_state"].items()}, strict=True)
        model_hook = model_hook.to(device).eval()
        model_parent = build_arm("RAW", payload, kappa_M, basis_parts, int(seed))
        model_parent.load_state_dict({k: v for k, v in run["soup_state"].items()}, strict=True)
        model_parent = model_parent.to(device).eval()
        h_hook = _predict_rows(model_hook, dev_data, device)["h"]
        h_parent = _predict_rows(model_parent, dev_data, device)["h"]
        d = float(np.abs(np.asarray(h_hook) - np.asarray(h_parent)).max())
        noise[f"DICT_s{seed}"]["identity_hook_max_abs_dpred"] = d
        eta = max(eta, d)

    # (5) the unique dictionary intervention on both DICT soups
    mean_codes = fit_mean_codes(basis, round_objects["fit_idx"])
    interventions: dict[str, dict[str, Any]] = {}
    for seed in seeds:
        interventions[f"DICT_s{seed}"] = run_alpha_intervention(
            seed, results[f"DICT_s{seed}"], mean_codes["mean_alpha"],
            out_dir=out_dir, device_name=device_name,
            round_objects=round_objects, data=data, log=log,
        )
        interventions[f"DICT_s{seed}"]["response_beyond_noise"] = bool(
            interventions[f"DICT_s{seed}"]["abs_dpred"]["p95"] > max(REPLAY_TOL, 10.0 * eta)
        )
    write_json(out_dir / "interventions.json", interventions)

    # (6) conditional read-only localisation (only if alpha is unresponsive)
    marker = float(max(REPLAY_TOL, 10.0 * eta))
    unresponsive = all(not interventions[f"DICT_s{seed}"]["response_beyond_noise"] for seed in seeds)
    localization: dict[str, dict[str, Any]] | None = None
    if unresponsive:
        log("[terminal-eval] alpha intervention unresponsive -> running the pre-registered localisation diagnostic")
        localization = {}
        for seed in seeds:
            for arm in ARMS:
                localization[f"{arm}_s{seed}"] = run_localization(
                    arm, seed, results[f"{arm}_s{seed}"], mean_codes,
                    out_dir=out_dir, device_name=device_name,
                    round_objects=round_objects, data=data, log=log,
                )
        write_json(out_dir / "localization.json", localization)

    # (7) frozen decision rules
    decision = frozen_decision_rules(per_seed, interventions, eta, bootstrap, localization, seeds=seeds)

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "stage": "terminal-eval",
        "one_shot": True,
        "roster": roster,
        "main_table": {
            f"{arm}_s{seed}": {
                "mae": results[f"{arm}_s{seed}"]["mae"],
                "b_y": results[f"{arm}_s{seed}"]["b_y"],
                "fit_diagnostics": results[f"{arm}_s{seed}"]["fit_diagnostics"],
                "curve_seconds_total": results[f"{arm}_s{seed}"]["curve_seconds_total"],
            }
            for seed in seeds for arm in ARMS
        },
        "paired": {
            "definition": "delta_s = MAE(DICT_s) - MAE(RAW_s) on full dev y_raw (positive = DICT worse)",
            "per_seed": {str(s): per_seed[s] for s in seeds},
            "avg_delta": avg_delta,
            "tolerance_rule": (
                f"two-seed mean delta <= +{TOL_AVG} AND each seed delta <= +{TOL_SEED} "
                "(engineering budget rule for keeping the candidate; never a general "
                "equivalence threshold; CI must be shown alongside)"
            ),
        },
        "bootstrap": bootstrap,
        "noise_bound": {"eta": eta, "per_run": noise, "marker": marker},
        "mean_codes": {
            "fit_rel_err_median": mean_codes["fit_rel_err_median"],
            "fit_rel_err_p95": mean_codes["fit_rel_err_p95"],
            "fit_alpha_nnz": mean_codes["fit_alpha_nnz"],
        },
        "interventions": interventions,
        "localization": localization,
        "decision": decision,
        "residual_structure_inputs": _residual_structure_inputs,
        "scope": (
            "development comparison on the historical dev rows (union of the old "
            "select/confirm), never a new independent confirm and never official "
            "valid/test; COMP supervision is a disclosed auxiliary condition; this "
            "round claims at most 'this effective local structure interface can be "
            "supplied by the basis'"
        ),
        "seconds": float(time.perf_counter() - started),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(out_dir / "terminal_eval.json", payload)
    log(
        f"[terminal-eval] branch {decision['branch']} ({decision['recommendation']}) "
        f"avg delta {avg_delta:+.5f} in {payload['seconds']:.0f}s"
    )
    return payload


# ---------------------------------------------------------------------------
# 9. CLI (local use; the registered runner calls the same functions)
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", default="source-objects",
                        choices=("source-objects", "checks", "probe", "smoke", "train", "terminal-eval"))
    parser.add_argument("--arm", default="RAW", choices=ARMS)
    parser.add_argument("--seed", type=int, default=SEED, choices=SEEDS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--out-dir", default=str(RESULTS_DIR))
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    if args.stage == "source-objects":
        phase_source_manifest(out_dir=out_dir)
    elif args.stage == "checks":
        run_checks(out_dir=out_dir)
    elif args.stage == "probe":
        interface_probe(out_dir=out_dir)
    elif args.stage == "smoke":
        run_smoke(device_name=args.device, out_dir=out_dir)
    elif args.stage == "train":
        train_arm(
            args.arm, args.seed, device_name=args.device, out_dir=out_dir,
            epochs=args.epochs, max_steps=args.max_steps,
        )
    else:
        terminal_eval(device_name=args.device, out_dir=out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
