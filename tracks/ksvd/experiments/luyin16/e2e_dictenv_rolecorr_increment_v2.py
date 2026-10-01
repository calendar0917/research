"""E2E-DictEnv-RoleCorr-Increment-v2 (core).

Round ``e2e_dictenv_rolecorr_increment_v2`` (study ``zinc-context-gap``).
Pre-registration: ``tracks/ksvd/notes/e2e_dictenv_rolecorr_increment_v2_preregistration.md``.

This round answers the *increment* question that RoleCorr-v1 could not: v1
replaced half of the structural dictionary (K32/s8 -> K16/s4) to make room for
16 correspondence coordinates, so its negative result mixed "adds
correspondence information" with "removes structural capacity".  Here the full
structural budget is kept and the third block is *appended*:

Route 1 (arms, all width 49, frozen dictionaries, trainable readout only):

* ``EXTRA-STRUCT``  A: ``[c~ ; alpha_base32 (s=8) ; alpha_extra16 (s=4)]``
  where ``alpha_extra`` is the frozen RoleCorr K16/s4 structural residual
  dictionary (the same object v1 used as its *only* structural coordinate);
* ``CORR-ADD``      B: ``[c~ ; alpha_base32 ; alpha_C16 (s=4)]`` with the
  frozen RoleCorr scaler + correspondence dictionary;
* ``CORR-PCA-ADD``  C: ``[c~ ; alpha_base32 ; PCA16(C_scaled)]`` — dense
  diagnostic control for the same correspondence object.

Primary comparison is B - A (does correspondence add anything *on top of* the
full structural budget); auxiliary comparison is B - C (is the increment
specific to sparse dictionary coding of ``C``).

Route 2 (conditional, only entered if route 1 does not fire its gate):

* ``JOINT-SPARSE``  D: ``[c~ ; IHT12(colnorm(D_joint48), joint_scaled)]`` where
  ``joint_scaled`` is the train-only scale-balanced concatenation of the
  structural residual (65), ``Sem108`` (108) and the RoleCorr correspondence
  object (536) = 709 dims; one frozen K48/s12 dictionary.
* ``JOINT-PCA``     E: ``[c~ ; PCA48(joint_scaled)]`` — the fixed dense control
  on the identical joint input.

Everything else (Sem108 interface, C6 mask, node/edge bindings, 48-D
environment, backend, reader, optimizer, 320-epoch horizon, Top-5 soup) is the
frozen Sem108 parent.  Dictionaries are frozen; only the static bindings, the
backend and the reader are trained.

``official_test_loaded = false`` everywhere; the official test split is never
instantiated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_a1 as a1
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_common_subspace_dictionary_v1 as cssd
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_rolecorr_v1 as rc
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_sem108_v1 as sem
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_v0 as v0
from tracks.ksvd.experiments.luyin16 import sdb_v0 as sdb

PROTOCOL_VERSION = "e2e_dictenv_rolecorr_increment_v2"

# ---------------------------------------------------------------------------
# frozen geometry
# ---------------------------------------------------------------------------

PHI_DIM = int(rc.PHI_DIM)                  # 65
SEM_DIM = int(getattr(sem, "SEM_DIM"))     # 108
CORR_DIM = int(rc.CORR_DIM)                # 536

COMMON_DIM = 1
BASE_ATOMS = int(rc.BASE_ATOMS)            # 32 (frozen SDB dictionary)
BASE_SPARSITY = int(rc.BASE_SPARSITY)      # 8
EXTRA_ATOMS = int(rc.K_ATOMS)              # 16 (frozen RoleCorr structural dict)
EXTRA_SPARSITY = int(rc.SPARSITY)          # 4
IHT_STEPS = int(rc.IHT_STEPS)              # 10

COORD_DIM = COMMON_DIM + BASE_ATOMS + EXTRA_ATOMS  # 49
BASE_SLICE = slice(COMMON_DIM, COMMON_DIM + BASE_ATOMS)          # 1..33
EXTRA_SLICE = slice(COMMON_DIM + BASE_ATOMS, COORD_DIM)          # 33..49

# route 2
JOINT_ATOMS = 48
JOINT_SPARSITY = 12
JOINT_STRUCT_DIM = PHI_DIM                 # 65
JOINT_SEM_DIM = SEM_DIM                    # 108
JOINT_CORR_DIM = CORR_DIM                  # 536
JOINT_DIM = JOINT_STRUCT_DIM + JOINT_SEM_DIM + JOINT_CORR_DIM    # 709
JOINT_BLOCKS: tuple[str, ...] = ("struct", "sem", "corr")
JOINT_BLOCK_SLICES: dict[str, slice] = {
    "struct": slice(0, JOINT_STRUCT_DIM),
    "sem": slice(JOINT_STRUCT_DIM, JOINT_STRUCT_DIM + JOINT_SEM_DIM),
    "corr": slice(JOINT_STRUCT_DIM + JOINT_SEM_DIM, JOINT_DIM),
}
JOINT_SLICE = slice(COMMON_DIM, COORD_DIM)                       # 1..49

DICT_EPOCHS = int(rc.DICT_EPOCHS)          # 10
DICT_SEED = int(rc.DICT_SEED)              # 20260924
SCALER_FLOOR = float(rc.SCALER_FLOOR)
SCALER_EPS = float(rc.SCALER_EPS)

#: arms
ARM_EXTRA_STRUCT = "EXTRA-STRUCT"
ARM_CORR_ADD = "CORR-ADD"
ARM_CORR_PCA_ADD = "CORR-PCA-ADD"
ARM_JOINT_SPARSE = "JOINT-SPARSE"
ARM_JOINT_PCA = "JOINT-PCA"
ROUTE1_ARMS = (ARM_EXTRA_STRUCT, ARM_CORR_ADD, ARM_CORR_PCA_ADD)
ROUTE2_ARMS = (ARM_JOINT_SPARSE, ARM_JOINT_PCA)
ALL_ARMS = ROUTE1_ARMS + ROUTE2_ARMS

BLOCK_MODE_SPARSE = "sparse"
BLOCK_MODE_PCA = "pca"
ARM_BLOCK_MODE = {
    ARM_EXTRA_STRUCT: BLOCK_MODE_SPARSE,
    ARM_CORR_ADD: BLOCK_MODE_SPARSE,
    ARM_CORR_PCA_ADD: BLOCK_MODE_PCA,
    ARM_JOINT_SPARSE: BLOCK_MODE_SPARSE,
    ARM_JOINT_PCA: BLOCK_MODE_PCA,
}
ARM_BLOCK_DIM = {
    ARM_EXTRA_STRUCT: PHI_DIM,
    ARM_CORR_ADD: CORR_DIM,
    ARM_CORR_PCA_ADD: CORR_DIM,
    ARM_JOINT_SPARSE: JOINT_DIM,
    ARM_JOINT_PCA: JOINT_DIM,
}
ARM_BLOCK_ATOMS = {
    ARM_EXTRA_STRUCT: EXTRA_ATOMS,
    ARM_CORR_ADD: EXTRA_ATOMS,
    ARM_CORR_PCA_ADD: EXTRA_ATOMS,
    ARM_JOINT_SPARSE: JOINT_ATOMS,
    ARM_JOINT_PCA: JOINT_ATOMS,
}
ARM_BLOCK_SPARSITY = {
    ARM_EXTRA_STRUCT: EXTRA_SPARSITY,
    ARM_CORR_ADD: EXTRA_SPARSITY,
    ARM_CORR_PCA_ADD: EXTRA_SPARSITY,
    ARM_JOINT_SPARSE: JOINT_SPARSITY,
    ARM_JOINT_PCA: JOINT_SPARSITY,
}
ROUTE_OF_ARM = {arm: 1 for arm in ROUTE1_ARMS}
ROUTE_OF_ARM.update({arm: 2 for arm in ROUTE2_ARMS})

#: mechanism thresholds (frozen)
GATE_MECHANISM_DIRECTIONAL = float(rc.GATE_MECHANISM_DIRECTIONAL)  # 0.003
GATE_MECHANISM_CLEAR = float(rc.GATE_MECHANISM_CLEAR)              # 0.010
#: primary screening gate: absolute soup-MAE improvement of B over A
SCREEN_ABS_GATE = 0.003
#: auxiliary sparse-vs-dense tolerance (B must not be worse than C by this)
SPARSE_DENSE_TOLERANCE = 0.0

VERDICTS = {
    "increment_no_gain": "INCREMENT_NO_MATERIAL_GAIN",
    "increment_gain_sparse": "INCREMENT_SUPPORTED_SPARSE_SPECIFIC",
    "increment_gain_dense": "INCREMENT_SUPPORTED_NOT_SPARSE_SPECIFIC",
    "increment_block_unused": "INCREMENT_COORDINATE_NOT_LOAD_BEARING",
    "route2_no_gain": "JOINT_ENCODING_NO_MATERIAL_GAIN",
    "route2_gain_sparse": "JOINT_ENCODING_SUPPORTED_SPARSE_SPECIFIC",
    "route2_gain_dense": "JOINT_ENCODING_SUPPORTED_NOT_SPARSE_SPECIFIC",
    "incomplete": "INCREMENT_ROUND_INCOMPLETE",
}

#: attribute-shuffle probe seeds for the object-level correspondence shuffle
SHUFFLE_SEEDS = tuple(rc.SHUFFLE_SEEDS)
#: coordinate row-shuffle probe seeds (distribution-preserving block shuffle)
COORD_SHUFFLE_SEEDS: tuple[int, ...] = (11, 22, 33, 44, 55)
CTRL_SHUFFLE_SEED = int(SHUFFLE_SEEDS[0])


def official_test_blocker(payload: Mapping[str, Any]) -> None:
    if payload.get("official_test_loaded") is not False:
        raise RuntimeError(
            "official-test blocker violated (official_test_loaded must be False)"
        )


def resolve_device(spec: Any) -> torch.device:
    """Resolve an explicit device spec (``cpu`` / ``cuda`` / ``cuda:0``).

    ``None`` and ``"cpu"`` keep the historical CPU regime.  CUDA is only
    accepted when it is actually available; a missing GPU is a blocker, never a
    silent CPU fallback.
    """
    if spec is None:
        return torch.device("cpu")
    device = torch.device(str(spec))
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "device spec requests CUDA but torch.cuda.is_available() is False"
            )
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"device {device} exceeds visible CUDA device count "
                f"{torch.cuda.device_count()}"
            )
    elif device.type != "cpu":
        raise RuntimeError(f"unsupported device type {device.type!r}")
    return device


def device_report(device: torch.device) -> dict[str, Any]:
    """Environment provenance for one resolved device."""
    report: dict[str, Any] = {
        "device": str(device),
        "device_type": device.type,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "official_test_loaded": False,
    }
    if device.type == "cuda":
        index = int(device.index if device.index is not None else 0)
        properties = torch.cuda.get_device_properties(index)
        report.update(
            {
                "gpu_name": str(properties.name),
                "gpu_index_visible": index,
                "gpu_total_memory_bytes": int(properties.total_memory),
                "gpu_uuid": str(getattr(properties, "uuid", "n/a")),
            }
        )
    return report


# ---------------------------------------------------------------------------
# route 2: train-only joint scale balancing
# ---------------------------------------------------------------------------


@dataclass
class JointScaler:
    """Three blocks (struct / sem / corr): per-coordinate RMS + block energy."""

    struct: a1.BlockScaler
    sem: a1.BlockScaler
    corr: a1.BlockScaler

    def to_json(self) -> dict[str, Any]:
        return {
            "blocks": {name: getattr(self, name).to_json() for name in JOINT_BLOCKS},
            "dim": int(JOINT_DIM),
            "block_slices": {k: [v.start, v.stop] for k, v in JOINT_BLOCK_SLICES.items()},
            "mean_subtracted": False,
            "fit_split": "official train",
            "official_test_loaded": False,
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "JointScaler":
        blocks: dict[str, a1.BlockScaler] = {}
        for name in JOINT_BLOCKS:
            entry = payload["blocks"][name]
            scale = np.asarray(entry["scale"], dtype=np.float64)
            mask = np.asarray(entry["mask"], dtype=np.float64)
            blocks[name] = a1.BlockScaler(
                name=str(name),
                scale=scale,
                mask=mask,
                weight=float(entry["weight"]),
                rms_raw=np.zeros(scale.shape[0], dtype=np.float64),
                block_energy=float(entry["block_energy"]),
                masked_coordinates=int(entry["masked_coordinates"]),
                dim=int(entry["dim"]),
            )
        return cls(struct=blocks["struct"], sem=blocks["sem"], corr=blocks["corr"])


def fit_joint_scaler(
    struct_block: np.ndarray, sem_block: np.ndarray, corr_block: np.ndarray
) -> JointScaler:
    """Fit the frozen three-block train-only scale balancer on the raw blocks."""
    struct_block = np.asarray(struct_block, dtype=np.float64)
    sem_block = np.asarray(sem_block, dtype=np.float64)
    corr_block = np.asarray(corr_block, dtype=np.float64)
    if struct_block.shape[1] != JOINT_STRUCT_DIM:
        raise RuntimeError("struct block width mismatch")
    if sem_block.shape[1] != JOINT_SEM_DIM:
        raise RuntimeError("sem block width mismatch")
    if corr_block.shape[1] != JOINT_CORR_DIM:
        raise RuntimeError("corr block width mismatch")
    return JointScaler(
        struct=a1.fit_block_scaler("struct", struct_block),
        sem=a1.fit_block_scaler("sem", sem_block),
        corr=a1.fit_block_scaler("corr", corr_block),
    )


def apply_joint_scaler(
    scaler: JointScaler,
    struct_block: np.ndarray,
    sem_block: np.ndarray,
    corr_block: np.ndarray,
) -> np.ndarray:
    """``[N, 709]`` balanced joint input: ``w_b * (block / scale) * mask_b``."""
    parts: list[np.ndarray] = []
    for name, block in (
        ("struct", struct_block),
        ("sem", sem_block),
        ("corr", corr_block),
    ):
        entry = getattr(scaler, name)
        block = np.asarray(block, dtype=np.float64)
        if block.shape[1] != int(entry.dim):
            raise RuntimeError(f"block {name} width {block.shape[1]} != {entry.dim}")
        parts.append(
            float(entry.weight) * (block / entry.scale[None, :]) * entry.mask[None, :]
        )
    out = np.concatenate(parts, axis=1)
    if out.shape[1] != JOINT_DIM:
        raise RuntimeError(f"balanced joint width {out.shape[1]} != {JOINT_DIM}")
    return out


def joint_block_report(
    struct_block: np.ndarray, sem_block: np.ndarray, corr_block: np.ndarray
) -> dict[str, Any]:
    """Raw per-block energy / mask accounting used to freeze the balance rule."""
    out: dict[str, Any] = {"n_rows": int(struct_block.shape[0]), "blocks": {}}
    for name, block in (
        ("struct", struct_block),
        ("sem", sem_block),
        ("corr", corr_block),
    ):
        block = np.asarray(block, dtype=np.float64)
        col_rms = np.sqrt(np.mean(block * block, axis=0))
        row_energy = np.sum(block * block, axis=1)
        out["blocks"][name] = {
            "dim": int(block.shape[1]),
            "mean_row_squared_norm": float(np.mean(row_energy)),
            "zero_rms_coordinates": int(np.sum(col_rms <= SCALER_FLOOR)),
            "raw_rms_min": float(col_rms.min()),
            "raw_rms_max": float(col_rms.max()),
        }
    out["official_test_loaded"] = False
    return out


def fit_joint_dictionary(
    joint_scaled_train: np.ndarray,
    *,
    atoms: int = JOINT_ATOMS,
    s: int = JOINT_SPARSITY,
    epochs: int = DICT_EPOCHS,
    seed: int = DICT_SEED,
    log: Any = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Unlabeled K-SVD of the balanced joint input (route 2, frozen budget)."""
    X = np.asarray(joint_scaled_train, dtype=np.float64)
    if X.shape[1] != JOINT_DIM:
        raise RuntimeError(f"joint width {X.shape[1]} != {JOINT_DIM}")
    D, info = sdb.fit_ksvd(
        X, atoms=int(atoms), s=int(s), epochs=int(epochs), seed=int(seed), log=log
    )
    return np.asarray(D, dtype=np.float32), info


def fit_joint_pca(X: np.ndarray, rank: int = JOINT_ATOMS) -> rc.PCA16:
    """Rank-``rank`` affine PCA of the balanced joint input (dense control)."""
    X = np.asarray(X, dtype=np.float64)
    if X.shape[1] != JOINT_DIM:
        raise RuntimeError(f"joint width {X.shape[1]} != {JOINT_DIM}")
    mean = X.mean(axis=0)
    centred = X - mean[None, :]
    gram = centred.T @ centred
    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1][: int(rank)]
    components = np.ascontiguousarray(vectors[:, order].T)
    for row in range(components.shape[0]):
        pivot = int(np.argmax(np.abs(components[row])))
        if components[row, pivot] < 0.0:
            components[row] = -components[row]
    return rc.PCA16(mean=mean, components=components)


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------


class IncrementModel(sem.SEM108Model):
    """``SEM108Model`` with a 49-wide coordinate and one appended block.

    Route 1 coordinate: ``[c~ ; alpha_base32 (s=8) ; block16]``.
    Route 2 coordinate: ``[c~ ; joint block48]`` (the base 32-coordinate is not
    produced; the frozen SDB dictionary stays as an inert diagnostic object).

    The binding extension is deterministic and shared by every arm: the
    historical ``W_A_S [33, 48]`` / ``W_E_S [99, 48]`` rows are copied into a
    ``[49, 48]`` / ``[147, 48]`` layout and the 16 new rows are zero-initialised.
    Only ``D`` and the block object differ between arms; every other parameter
    is copied bit-for-bit from a shared reference state.
    """

    def __init__(
        self,
        config: Any,
        dictionary: np.ndarray,
        *,
        subspace: cssd.CommonSubspace,
        arm: str = ARM_EXTRA_STRUCT,
        block_dictionary: np.ndarray | None = None,
        pca_mean: np.ndarray | None = None,
        pca_components: np.ndarray | None = None,
        freeze_dictionary: bool = True,
        iht_steps: int = IHT_STEPS,
    ) -> None:
        arm = str(arm)
        if arm not in ALL_ARMS:
            raise ValueError(f"unknown arm {arm!r}")
        dictionary = np.asarray(dictionary, dtype=np.float32)
        if dictionary.shape != (PHI_DIM, BASE_ATOMS):
            raise RuntimeError(
                f"base structural dictionary {dictionary.shape} != "
                f"({PHI_DIM}, {BASE_ATOMS})"
            )
        super().__init__(
            config,
            dictionary,
            subspace=subspace,
            node_binding="paired",
            edge_binding="paired",
            coding="sparse",
        )
        self.arm = arm
        self.route = int(ROUTE_OF_ARM[arm])
        self.block_mode = ARM_BLOCK_MODE[arm]
        self.block_dim = int(ARM_BLOCK_DIM[arm])
        self.block_atoms = int(ARM_BLOCK_ATOMS[arm])
        self.block_sparsity = int(ARM_BLOCK_SPARSITY[arm])
        self.iht_steps = int(iht_steps)
        self.D.requires_grad_(not bool(freeze_dictionary))
        self.dead_column_fallbacks += self._extend_binding_()
        self.inference_zero_base = False
        self.inference_zero_block = False
        self.inference_shuffle_block_seed: int | None = None
        self._pending_block: torch.Tensor | None = None
        self._pending_batch: torch.Tensor | None = None
        self.D_block = None
        self.block_pca_mean_np = None
        self.block_pca_components_np = None
        if self.block_mode == BLOCK_MODE_SPARSE:
            if block_dictionary is None:
                raise RuntimeError(f"arm {arm} requires a block dictionary")
            block_dictionary = np.asarray(block_dictionary, dtype=np.float32)
            if block_dictionary.shape != (self.block_dim, self.block_atoms):
                raise RuntimeError(
                    f"block dictionary {block_dictionary.shape} != "
                    f"({self.block_dim}, {self.block_atoms})"
                )
            self.D_block = nn.Parameter(
                torch.as_tensor(block_dictionary, dtype=torch.float32).clone(),
                requires_grad=not bool(freeze_dictionary),
            )
        else:
            if block_dictionary is not None:
                raise RuntimeError("pca block mode does not take a dictionary")
            if pca_mean is None or pca_components is None:
                raise RuntimeError("pca block mode requires the fitted mean/components")
            mean = np.asarray(pca_mean, dtype=np.float32)
            components = np.asarray(pca_components, dtype=np.float32)
            if mean.shape != (self.block_dim,):
                raise RuntimeError(f"pca mean {mean.shape} != ({self.block_dim},)")
            if components.shape != (self.block_atoms, self.block_dim):
                raise RuntimeError(
                    f"pca components {components.shape} != "
                    f"({self.block_atoms}, {self.block_dim})"
                )
            self.register_buffer("block_pca_mean", torch.as_tensor(mean))
            self.register_buffer("block_pca_components", torch.as_tensor(components))
            self.block_pca_mean_np = mean
            self.block_pca_components_np = components

    # -- deterministic binding extension --------------------------------------

    def _extend_binding_(self) -> int:
        """``[33,48] -> [49,48]`` and ``[99,48] -> [147,48]`` (zero new rows)."""
        old_rows = COMMON_DIM + BASE_ATOMS
        with torch.no_grad():
            old_a = self.W_A_S.detach().clone()
            if old_a.shape[0] != old_rows:
                raise RuntimeError(
                    f"expected the widened {old_rows}-row binding, got {old_a.shape[0]}"
                )
            new_a = torch.zeros(
                COORD_DIM, old_a.shape[1], dtype=old_a.dtype, device=old_a.device
            )
            new_a[:old_rows] = old_a
            self.W_A_S = nn.Parameter(new_a)
            old_e = self.W_E_S.detach().clone()
            d_e = old_e.shape[1]
            if old_e.shape[0] != 3 * old_rows:
                raise RuntimeError(
                    f"expected the widened {3 * old_rows}-row edge binding, "
                    f"got {old_e.shape[0]}"
                )
            new_e = torch.zeros(
                3 * COORD_DIM, d_e, dtype=old_e.dtype, device=old_e.device
            )
            for block in range(3):
                new_e[
                    block * COORD_DIM : block * COORD_DIM + old_rows
                ] = old_e[block * old_rows : (block + 1) * old_rows]
            self.W_E_S = nn.Parameter(new_e)
        return 0

    # -- frozen coordinate -----------------------------------------------------

    def residual_dictionary(self) -> torch.Tensor:
        return super().residual_dictionary()

    def base_codes(self, phi: torch.Tensor) -> torch.Tensor:
        """The frozen SDB K32/s8 coordinate (identical to RoleCorr ``TOPO``)."""
        U = self.U.to(dtype=phi.dtype)
        r = phi - (phi @ U) @ U.t()
        return v0.tied_iht_codes(
            self.residual_dictionary(), r, s=BASE_SPARSITY, steps=self.iht_steps
        )

    def block_input(self, phi: torch.Tensor, block: torch.Tensor | None) -> torch.Tensor:
        if self.route == 2:
            if block is None:
                raise RuntimeError("route 2 requires the balanced joint input")
            return block
        if self.arm == ARM_EXTRA_STRUCT:
            U = self.U.to(dtype=phi.dtype)
            return phi - (phi @ U) @ U.t()
        if block is None:
            raise RuntimeError(f"arm {self.arm} requires its block input")
        return block

    def block_codes(self, phi: torch.Tensor, block: torch.Tensor | None) -> torch.Tensor:
        x = self.block_input(phi, block)
        if self.block_mode == BLOCK_MODE_SPARSE:
            dictionary = v0.normalized_dictionary(self.D_block)
            if self.arm == ARM_EXTRA_STRUCT:
                U = self.U.to(dtype=x.dtype)
                D_perp = self.D_block - U @ (U.t() @ self.D_block)
                dictionary = v0.normalized_dictionary(D_perp)
            return v0.tied_iht_codes(
                dictionary, x, s=self.block_sparsity, steps=self.iht_steps
            )
        mean = self.block_pca_mean.to(dtype=x.dtype)
        components = self.block_pca_components.to(dtype=x.dtype)
        return (x - mean) @ components.t()

    def code(self, phi: torch.Tensor, block: torch.Tensor | None = None) -> torch.Tensor:
        U = self.U.to(dtype=phi.dtype)
        c = phi @ U
        scale = self.common_rms.to(dtype=phi.dtype)
        if block is None:
            block = self._pending_block
        if self.route == 2:
            joint = self.block_codes(phi, block)
            if self.inference_zero_block:
                joint = torch.zeros_like(joint)
            if self.inference_shuffle_block_seed is not None:
                joint = shuffle_block_rows(
                    joint, int(self.inference_shuffle_block_seed), self._pending_batch
                )
            return torch.cat([c / scale, joint], dim=1)
        alpha_base = self.base_codes(phi)
        if self.inference_zero_base:
            alpha_base = torch.zeros_like(alpha_base)
        alpha_block = self.block_codes(phi, block)
        if self.inference_zero_block:
            alpha_block = torch.zeros_like(alpha_block)
        if self.inference_shuffle_block_seed is not None:
            alpha_block = shuffle_block_rows(
                alpha_block, int(self.inference_shuffle_block_seed), self._pending_batch
            )
        return torch.cat([c / scale, alpha_base, alpha_block], dim=1)

    def forward(self, data: Any, **kwargs: Any):
        self._pending_block = self.extract_block(data)
        self._pending_batch = getattr(data, "batch", None)
        try:
            return super().forward(data, **kwargs)
        finally:
            self._pending_block = None
            self._pending_batch = None

    def extract_block(self, data: Any) -> torch.Tensor | None:
        if self.route == 2:
            return getattr(data, "joint_vec", None)
        if self.arm in (ARM_CORR_ADD, ARM_CORR_PCA_ADD):
            return getattr(data, "corr_vec", None)
        return None

    # -- diagnostic reconstruction (inert with frozen dictionaries) -----------

    def reconstruct(self, phi: torch.Tensor, coord: torch.Tensor) -> torch.Tensor:
        if self.route == 2:
            return torch.zeros_like(phi)
        return coord[:, BASE_SLICE] @ self.residual_dictionary().t()

    def reconstruction_loss(self, phi: torch.Tensor, coord: torch.Tensor) -> torch.Tensor:
        U = self.U.to(dtype=phi.dtype)
        r = phi - (phi @ U) @ U.t()
        phi_hat = self.reconstruct(phi, coord)
        numerator = ((r - phi_hat) ** 2).sum(dim=1)
        denominator = (phi**2).sum(dim=1) + v0.EPS
        return (numerator / denominator).mean()


def build_increment_model(
    *,
    arm: str,
    dictionary: np.ndarray,
    seed: int,
    subspace: cssd.CommonSubspace,
    block_dictionary: np.ndarray | None = None,
    pca_mean: np.ndarray | None = None,
    pca_components: np.ndarray | None = None,
    freeze_dictionary: bool = True,
    reference_state: Mapping[str, torch.Tensor] | None = None,
) -> IncrementModel:
    """Build an arm with the exact frozen ``build_sem108_model`` RNG stream.

    ``reference_state`` copies every same-shaped entry after construction (the
    shared readout / bindings initialisation), skipping the arm-specific
    ``D`` / ``D_block`` objects.
    """
    torch.manual_seed(int(seed))
    model = IncrementModel(
        cm.H1_CONFIG,
        dictionary,
        subspace=subspace,
        arm=str(arm),
        block_dictionary=block_dictionary,
        pca_mean=pca_mean,
        pca_components=pca_components,
        freeze_dictionary=bool(freeze_dictionary),
    )
    if reference_state is not None:
        with torch.no_grad():
            state = model.state_dict()
            copied = 0
            for key, value in reference_state.items():
                if key in ("D", "D_block"):
                    continue
                if key in state and tuple(state[key].shape) == tuple(value.shape):
                    state[key].copy_(value)
                    copied += 1
        model._shared_init_keys = copied  # type: ignore[attr-defined]
    return model


def coordinate_slices() -> dict[str, Any]:
    return {
        "common": [0, COMMON_DIM],
        "base": [int(BASE_SLICE.start), int(BASE_SLICE.stop)],
        "block": [int(EXTRA_SLICE.start), int(EXTRA_SLICE.stop)],
        "coord_dim": int(COORD_DIM),
        "route2_joint": [int(JOINT_SLICE.start), int(JOINT_SLICE.stop)],
    }


def coordinate_payload(model: IncrementModel) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "arm": str(model.arm),
        "route": int(model.route),
        "block_mode": str(model.block_mode),
        "block_dim": int(model.block_dim),
        "block_atoms": int(model.block_atoms),
        "block_sparsity": int(model.block_sparsity),
        "iht_steps": int(model.iht_steps),
        "coordinate": coordinate_slices(),
        "base_dictionary_shape": list(model.D.shape),
        "base_dictionary_trainable": bool(model.D.requires_grad),
        "block_dictionary_shape": (
            None if model.D_block is None else list(model.D_block.shape)
        ),
        "block_dictionary_trainable": (
            None if model.D_block is None else bool(model.D_block.requires_grad)
        ),
        "W_A_S": list(model.W_A_S.shape),
        "W_E_S": list(model.W_E_S.shape),
        "binding_extension_rows": int(EXTRA_ATOMS),
        "trainable_parameters": int(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        ),
        "total_parameters": int(sum(p.numel() for p in model.parameters())),
        "official_test_loaded": False,
    }
    official_test_blocker(payload)
    return payload


# ---------------------------------------------------------------------------
# coordinate diagnostics / usage
# ---------------------------------------------------------------------------


def zero_block_purity(
    model: IncrementModel, phi: torch.Tensor, block: torch.Tensor | None
) -> dict[str, Any]:
    """Verify that zeroing the appended block touches only its coordinate span.

    Route 1 zeroes ``EXTRA_SLICE`` (and keeps ``[c~ ; alpha_base]``); route 2
    zeroes the whole ``JOINT_SLICE`` (and keeps only ``c~``).
    """
    span = JOINT_SLICE if model.route == 2 else EXTRA_SLICE
    coord_full = model.code(phi, block)
    model.inference_zero_block = True
    try:
        coord_zero = model.code(phi, block)
    finally:
        model.inference_zero_block = False
    keep_full = torch.cat(
        [coord_full[:, : int(span.start)], coord_full[:, int(span.stop) :]], dim=1
    )
    keep_zero = torch.cat(
        [coord_zero[:, : int(span.start)], coord_zero[:, int(span.stop) :]], dim=1
    )
    return {
        "span": [int(span.start), int(span.stop)],
        "other_columns_bit_identical": bool(torch.equal(keep_full, keep_zero)),
        "block_columns_zero": bool(
            torch.equal(coord_zero[:, span], torch.zeros_like(coord_zero[:, span]))
        ),
        "max_abs_shift_on_zeroed_slice": float(
            (coord_full[:, span] - coord_zero[:, span]).abs().max()
        ),
        "official_test_loaded": False,
    }


def zero_base_purity(
    model: IncrementModel, phi: torch.Tensor, block: torch.Tensor | None
) -> dict[str, Any]:
    """Verify that zeroing the base structural block touches only ``BASE_SLICE``."""
    if model.route == 2:
        return {"applicable": False, "official_test_loaded": False}
    coord_full = model.code(phi, block)
    model.inference_zero_base = True
    try:
        coord_zero = model.code(phi, block)
    finally:
        model.inference_zero_base = False
    keep_full = torch.cat(
        [coord_full[:, : int(BASE_SLICE.start)], coord_full[:, int(BASE_SLICE.stop) :]],
        dim=1,
    )
    keep_zero = torch.cat(
        [coord_zero[:, : int(BASE_SLICE.start)], coord_zero[:, int(BASE_SLICE.stop) :]],
        dim=1,
    )
    return {
        "applicable": True,
        "base_columns_zero": bool(
            torch.equal(
                coord_zero[:, BASE_SLICE], torch.zeros_like(coord_zero[:, BASE_SLICE])
            )
        ),
        "other_slices_bit_identical": bool(torch.equal(keep_full, keep_zero)),
        "official_test_loaded": False,
    }


def shuffle_block_rows(
    codes: torch.Tensor, seed: int, batch: torch.Tensor | None = None
) -> torch.Tensor:
    """Distribution-preserving row shuffle of one block of codes.

    With ``batch`` given the shuffle is within each graph (so the molecule's
    block multiset is preserved); the block's marginal distribution over the
    split is preserved either way.  Pure function; returns a new tensor.
    """
    generator = torch.Generator().manual_seed(int(seed))
    n_rows = int(codes.shape[0])
    if batch is None:
        index = torch.randperm(n_rows, generator=generator)
    else:
        batch = batch.to(codes.device)
        n_graphs = int(batch.max().item()) + 1 if int(batch.numel()) else 0
        index = torch.empty(n_rows, dtype=torch.long)
        for graph in range(n_graphs):
            rows = (batch == graph).nonzero(as_tuple=False).view(-1)
            index[rows] = rows[torch.randperm(int(rows.numel()), generator=generator)]
        index = index.to(codes.device)
    return codes[index]


__all__ = [
    "PROTOCOL_VERSION",
    "PHI_DIM",
    "SEM_DIM",
    "CORR_DIM",
    "COMMON_DIM",
    "BASE_ATOMS",
    "BASE_SPARSITY",
    "EXTRA_ATOMS",
    "EXTRA_SPARSITY",
    "IHT_STEPS",
    "COORD_DIM",
    "BASE_SLICE",
    "EXTRA_SLICE",
    "JOINT_ATOMS",
    "JOINT_SPARSITY",
    "JOINT_DIM",
    "JOINT_BLOCKS",
    "JOINT_BLOCK_SLICES",
    "JOINT_SLICE",
    "DICT_EPOCHS",
    "DICT_SEED",
    "SCALER_FLOOR",
    "SCALER_EPS",
    "ARM_EXTRA_STRUCT",
    "ARM_CORR_ADD",
    "ARM_CORR_PCA_ADD",
    "ARM_JOINT_SPARSE",
    "ARM_JOINT_PCA",
    "ROUTE1_ARMS",
    "ROUTE2_ARMS",
    "ALL_ARMS",
    "BLOCK_MODE_SPARSE",
    "BLOCK_MODE_PCA",
    "ARM_BLOCK_MODE",
    "ARM_BLOCK_DIM",
    "ARM_BLOCK_ATOMS",
    "ARM_BLOCK_SPARSITY",
    "ROUTE_OF_ARM",
    "GATE_MECHANISM_DIRECTIONAL",
    "GATE_MECHANISM_CLEAR",
    "SCREEN_ABS_GATE",
    "SPARSE_DENSE_TOLERANCE",
    "VERDICTS",
    "SHUFFLE_SEEDS",
    "COORD_SHUFFLE_SEEDS",
    "CTRL_SHUFFLE_SEED",
    "official_test_blocker",
    "resolve_device",
    "device_report",
    "JointScaler",
    "fit_joint_scaler",
    "apply_joint_scaler",
    "joint_block_report",
    "fit_joint_dictionary",
    "fit_joint_pca",
    "IncrementModel",
    "build_increment_model",
    "coordinate_slices",
    "coordinate_payload",
    "zero_block_purity",
    "zero_base_purity",
    "shuffle_block_rows",
]
