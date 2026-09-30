"""E2E-DictEnv-RoleCorr-v1 — role<->attribute correspondence dictionary (core).

Round ``e2e_dictenv_rolecorr_v1`` (study ``zinc-context-gap``).
Pre-registration: ``tracks/ksvd/notes/e2e_dictenv_rolecorr_v1_preregistration.md``.

The round changes exactly one thing relative to the frozen ``CSSD-Sem108 + C6``
pipeline: the **dictionary coordinate**.

* Baseline ``TOPO`` (A): the existing pure-topology coordinate
  ``[ c~ ; IHT10(colnorm((I-UU^T) D_SDB), r) ]`` with the historical frozen
  ``D_SDB`` (K=32, s=8).
* Candidate ``CORR`` (B): a new object ``C in R^536`` — the within-group
  centred role x attribute cross statistic per node shell / edge shellpair —
  with a shared sparse dictionary ``D_C in R^{536x16}`` (s=4) learned on the
  official-train object, plus a 16-atom structural dictionary ``D_S`` (s=4)
  learned on the common-1 residual.  Coordinate
  ``[ c~ ; alpha_S (16) ; alpha_C (16) ]`` (width 33, identical to A).

Everything else (Sem108 interface, C6 mask, node/edge bindings, 48-D
environment, backend, reader, optimizer, 320-epoch horizon, Top-5 soup) is the
frozen Sem108 parent.  Dictionaries are frozen; only the readout is trained.

The module contains only definitions so every algebraic property demanded by
the pre-registration is unit-testable without loading ZINC.  The stage runner
is ``zinc_e2e_dictenv_rolecorr_v1.py``; focused CPU tests are
``tracks/ksvd/tests/test_e2e_dictenv_rolecorr_v1.py``.

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
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_sem108_v1 as sem
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_v0 as v0
from tracks.ksvd.experiments.luyin16 import fsar_r2_ar0 as r2
from tracks.ksvd.experiments.luyin16 import fsar_v2 as v2
from tracks.ksvd.experiments.luyin16 import sdb_v0 as sdb

PROTOCOL_VERSION = "e2e_dictenv_rolecorr_v1"

# ---------------------------------------------------------------------------
# frozen geometry of the correspondence object C
# ---------------------------------------------------------------------------

PHI_DIM = int(r2.PHI_DIM)                      # 65
NODE_BASIS_DIM = int(r2.NODE_BASIS_DIM)        # 11
EDGE_BASIS_DIM = int(r2.EDGE_BASIS_DIM)        # 15
ATOM_CATEGORIES = int(r2.ATOM_CATEGORIES)      # 28
BOND_CATEGORIES = int(r2.BOND_CATEGORIES)      # 4
PATCH_RADIUS = int(r2.PATCH_RADIUS)            # 2

#: node role columns kept after deleting the constant-in-group root/shell
#: indicators (columns 0..3) and node shells stored (root shell 0 is singleton).
NODE_ROLE_SLICE = slice(4, NODE_BASIS_DIM)     # 7 columns
NODE_SHELLS: tuple[int, ...] = (1, 2)
#: edge role columns kept after deleting the constant-in-group shellpair
#: one-hot (columns 0..5) and the edge shellpairs stored.
EDGE_ROLE_SLICE = slice(6, EDGE_BASIS_DIM)     # 9 columns
EDGE_SHELL_PAIRS: tuple[tuple[int, int], ...] = ((0, 1), (1, 1), (1, 2), (2, 2))
ALL_SHELL_PAIRS: tuple[tuple[int, int], ...] = tuple(
    tuple(int(v) for v in pair) for pair in v2.SHELL_PAIRS
)
N_SHELLS = int(v2.N_SHELLS)

NODE_ROLE_DIM = int(NODE_ROLE_SLICE.stop - NODE_ROLE_SLICE.start)   # 7
EDGE_ROLE_DIM = int(EDGE_ROLE_SLICE.stop - EDGE_ROLE_SLICE.start)   # 9
NODE_BLOCK_DIM = len(NODE_SHELLS) * NODE_ROLE_DIM * ATOM_CATEGORIES   # 392
EDGE_BLOCK_DIM = len(EDGE_SHELL_PAIRS) * EDGE_ROLE_DIM * BOND_CATEGORIES  # 144
CORR_DIM = NODE_BLOCK_DIM + EDGE_BLOCK_DIM                            # 536

CORR_BLOCKS: tuple[str, ...] = ("node", "edge")
CORR_BLOCK_SLICES: dict[str, slice] = {
    "node": slice(0, NODE_BLOCK_DIM),
    "edge": slice(NODE_BLOCK_DIM, CORR_DIM),
}

# ---------------------------------------------------------------------------
# frozen dictionary / coordinate configuration
# ---------------------------------------------------------------------------

K_ATOMS = 16
SPARSITY = 4
IHT_STEPS = int(v0.IHT_STEPS)          # 10
DICT_EPOCHS = 10
DICT_SEED = int(sdb.DICT_SEED)         # 20260924
SCALER_FLOOR = float(r2.SCALER_FLOOR)  # 1.0e-9
SCALER_EPS = float(r2.SCALER_EPS)      # 1.0e-12

COMMON_DIM = 1
STRUCT_DIM = K_ATOMS                   # 16
COORD_DIM = COMMON_DIM + STRUCT_DIM + K_ATOMS  # 33
STRUCT_SLICE = slice(COMMON_DIM, COMMON_DIM + STRUCT_DIM)          # 1..16
CORR_SLICE = slice(COMMON_DIM + STRUCT_DIM, COORD_DIM)             # 17..32

#: frozen baseline dictionary geometry (SDB-v0)
BASE_ATOMS = int(sdb.K_ATOMS)          # 32
BASE_SPARSITY = int(sdb.SPARSITY)      # 8

#: arms
ARM_TOPO = "TOPO"
ARM_CORR = "CORR"
ARM_CORR_SHUF = "CORR-SHUF"
ARM_CORR_PCA = "CORR-PCA"
ARMS = (ARM_TOPO, ARM_CORR, ARM_CORR_SHUF, ARM_CORR_PCA)

#: correspondence coding modes
CORR_MODE_SPARSE = "sparse"
CORR_MODE_PCA = "pca"
CORR_MODES = (CORR_MODE_SPARSE, CORR_MODE_PCA)

#: attribute-shuffle probe seeds (frozen)
SHUFFLE_SEEDS: tuple[int, ...] = (101, 202, 303, 404, 505)
#: the classification threshold for "different" in the construction audit
PERMUTE_DELTA_FLOOR = 1.0e-12
#: primary screening gate: relative soup-MAE improvement vs the frozen baseline
SCREEN_RELATIVE_GATE = 0.02
#: mechanism direction / clarity thresholds
GATE_MECHANISM_DIRECTIONAL = 0.003
GATE_MECHANISM_CLEAR = 0.010

VERDICTS = {
    "no_gain": "ROLE_CORR_NO_MATERIAL_GAIN",
    "proceed": "ROLE_CORR_SCREEN_PROCEED",
    "supported_sparse": "ROLE_CORR_SUPPORTED_SPARSE_DICT",
    "supported_not_sparse": "ROLE_CORR_SUPPORTED_NOT_SPARSE_SPECIFIC",
    "corr_not_used": "ROLE_CORR_COORDINATE_NOT_LOAD_BEARING",
    "incomplete": "ROLE_CORR_INCOMPLETE",
}


def official_test_blocker(payload: Mapping[str, Any]) -> None:
    if payload.get("official_test_loaded") is not False:
        raise RuntimeError(
            "official-test blocker violated (official_test_loaded must be False)"
        )


def cpu_only_guard(device: Any) -> None:
    resolved = torch.device(device)
    if resolved.type != "cpu":
        raise RuntimeError(f"RoleCorr round is CPU-only, got device={resolved}")


# ---------------------------------------------------------------------------
# 1. the correspondence object
# ---------------------------------------------------------------------------


def _group_index_from_one_hot(
    one_hot: np.ndarray, categories: int, name: str
) -> np.ndarray:
    """Recover the exact group id of each row from a one-hot block."""
    block = np.asarray(one_hot, dtype=np.float64)[:, : int(categories)]
    if block.size == 0:
        return np.zeros(0, dtype=np.int64)
    index = block.argmax(axis=1)
    exact = block[np.arange(block.shape[0]), index] == 1.0
    if not bool(exact.all()):
        raise RuntimeError(f"{name}: expected an exact one-hot group indicator")
    return index.astype(np.int64)


def _centered_cross(
    role: np.ndarray, attribute: np.ndarray, permute_seed: int | None = None
) -> np.ndarray:
    """``sum_i (role_i - mean(role))(attribute_i - mean(attribute))^T``.

    Returns zeros for a group with fewer than two members (empty or singleton).
    ``permute_seed`` permutes the attribute rows inside the group (keeping both
    multisets and the coarse marginals) — the frozen correspondence-shuffle.
    """
    role = np.asarray(role, dtype=np.float64)
    attribute = np.asarray(attribute, dtype=np.float64)
    n = int(role.shape[0])
    if n < 2:
        return np.zeros((role.shape[1], attribute.shape[1]), dtype=np.float64)
    if permute_seed is not None:
        rng = np.random.default_rng(int(permute_seed))
        attribute = attribute[rng.permutation(n)]
    role_centred = role - role.mean(axis=0, keepdims=True)
    attribute_centred = attribute - attribute.mean(axis=0, keepdims=True)
    return role_centred.T @ attribute_centred


@dataclass
class CorrBlocks:
    """One patch's raw correspondence object plus its group bookkeeping."""

    node: np.ndarray                 # [392]
    edge: np.ndarray                 # [144]
    node_group_sizes: dict[int, int] = field(default_factory=dict)
    edge_group_sizes: dict[tuple[int, int], int] = field(default_factory=dict)

    @property
    def vector(self) -> np.ndarray:
        return np.concatenate([self.node, self.edge])

    def group_size_vector(self) -> np.ndarray:
        """Frozen group-size order: node shells then edge shellpairs."""
        values = [int(self.node_group_sizes.get(s, 0)) for s in NODE_SHELLS]
        values += [int(self.edge_group_sizes.get(p, 0)) for p in EDGE_SHELL_PAIRS]
        return np.asarray(values, dtype=np.int64)


def corr_blocks(
    patch: Mapping[str, Any], permute_seed: int | None = None
) -> CorrBlocks:
    """Compute ``C`` blocks for one ``a1.patch_blocks`` payload.

    ``patch`` must expose ``node_basis [n, 11]``, ``edge_basis [m, 15]``,
    ``q [n, 28]`` and ``r [m, 4]`` — exactly the fields returned by
    ``e2e_dictenv_a1.patch_blocks``.  ``permute_seed`` applies the frozen
    within-group attribute permutation (per group independently).
    """
    node_basis = np.asarray(patch["node_basis"], dtype=np.float64)
    edge_basis = np.asarray(patch["edge_basis"], dtype=np.float64)
    q = np.asarray(patch["q"], dtype=np.float64)
    r = np.asarray(patch["r"], dtype=np.float64)
    if node_basis.shape[1] != NODE_BASIS_DIM or q.shape[1] != ATOM_CATEGORIES:
        raise RuntimeError("node basis / atom one-hot width mismatch")
    if edge_basis.shape[1] != EDGE_BASIS_DIM or r.shape[1] != BOND_CATEGORIES:
        raise RuntimeError("edge basis / bond one-hot width mismatch")

    node_role = node_basis[:, NODE_ROLE_SLICE]
    edge_role = edge_basis[:, EDGE_ROLE_SLICE]
    shells = _group_index_from_one_hot(
        node_basis[:, 1 : 1 + N_SHELLS], N_SHELLS, "node shell"
    )
    edge_pairs = _group_index_from_one_hot(
        edge_basis[:, : len(ALL_SHELL_PAIRS)], len(ALL_SHELL_PAIRS), "edge shellpair"
    )

    node_out = np.zeros(NODE_BLOCK_DIM, dtype=np.float64)
    node_group_sizes: dict[int, int] = {}
    for position, shell in enumerate(NODE_SHELLS):
        mask = shells == int(shell)
        count = int(mask.sum())
        node_group_sizes[int(shell)] = count
        if count < 2:
            continue
        seed = None if permute_seed is None else int(permute_seed) + 1000 + int(shell)
        block = _centered_cross(node_role[mask], q[mask], permute_seed=seed)
        low = position * NODE_ROLE_DIM * ATOM_CATEGORIES
        high = low + NODE_ROLE_DIM * ATOM_CATEGORIES
        node_out[low:high] = block.reshape(-1)

    edge_out = np.zeros(EDGE_BLOCK_DIM, dtype=np.float64)
    edge_group_sizes: dict[tuple[int, int], int] = {}
    for position, pair in enumerate(EDGE_SHELL_PAIRS):
        pair_id = ALL_SHELL_PAIRS.index(tuple(pair))
        mask = edge_pairs == pair_id
        count = int(mask.sum())
        edge_group_sizes[tuple(pair)] = count
        if count < 2:
            continue
        seed = (
            None
            if permute_seed is None
            else int(permute_seed) + 2000 + int(pair_id)
        )
        block = _centered_cross(edge_role[mask], r[mask], permute_seed=seed)
        low = position * EDGE_ROLE_DIM * BOND_CATEGORIES
        high = low + EDGE_ROLE_DIM * BOND_CATEGORIES
        edge_out[low:high] = block.reshape(-1)

    return CorrBlocks(
        node=node_out,
        edge=edge_out,
        node_group_sizes=node_group_sizes,
        edge_group_sizes=edge_group_sizes,
    )


def corr_vector(patch: Mapping[str, Any], permute_seed: int | None = None) -> np.ndarray:
    """The frozen 536-D raw object of one patch (float64)."""
    return corr_blocks(patch, permute_seed=permute_seed).vector


# ---------------------------------------------------------------------------
# 2. train-only block scaling (reuses the A1 primitive verbatim)
# ---------------------------------------------------------------------------


@dataclass
class CorrScaler:
    """Two block scalers (node 392 / edge 144) for the correspondence object."""

    node: a1.BlockScaler
    edge: a1.BlockScaler

    def to_json(self) -> dict[str, Any]:
        return {
            "blocks": {name: getattr(self, name).to_json() for name in CORR_BLOCKS},
            "dim": int(CORR_DIM),
            "mean_subtracted": False,
            "fit_split": "official train",
            "official_test_loaded": False,
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "CorrScaler":
        blocks: dict[str, a1.BlockScaler] = {}
        for name in CORR_BLOCKS:
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
        return cls(node=blocks["node"], edge=blocks["edge"])


def fit_corr_scaler(node_block: np.ndarray, edge_block: np.ndarray) -> CorrScaler:
    """Fit the frozen two-layer train-only scaler on the raw C blocks."""
    node_block = np.asarray(node_block, dtype=np.float64)
    edge_block = np.asarray(edge_block, dtype=np.float64)
    if node_block.shape[1] != NODE_BLOCK_DIM or edge_block.shape[1] != EDGE_BLOCK_DIM:
        raise RuntimeError("C block width mismatch while fitting the scaler")
    return CorrScaler(
        node=a1.fit_block_scaler("node", node_block),
        edge=a1.fit_block_scaler("edge", edge_block),
    )


def apply_corr_scaler(
    scaler: CorrScaler, node_block: np.ndarray, edge_block: np.ndarray
) -> np.ndarray:
    """``[N, 536]`` scaled object: ``w_b * (block / scale) * mask`` per block."""
    node_block = np.asarray(node_block, dtype=np.float64)
    edge_block = np.asarray(edge_block, dtype=np.float64)
    parts: list[np.ndarray] = []
    for name, block in (("node", node_block), ("edge", edge_block)):
        entry = getattr(scaler, name)
        if block.shape[1] != int(entry.dim):
            raise RuntimeError(f"block {name} width {block.shape[1]} != {entry.dim}")
        parts.append(
            float(entry.weight) * (block / entry.scale[None, :]) * entry.mask[None, :]
        )
    out = np.concatenate(parts, axis=1)
    if out.shape[1] != CORR_DIM:
        raise RuntimeError(f"scaled C width {out.shape[1]} != {CORR_DIM}")
    return out


def corr_block_report(node_block: np.ndarray, edge_block: np.ndarray) -> dict[str, Any]:
    """Non-zero-object and coordinate-usage accounting (train/valid)."""
    node_block = np.asarray(node_block, dtype=np.float64)
    edge_block = np.asarray(edge_block, dtype=np.float64)
    out: dict[str, Any] = {"n_patches": int(node_block.shape[0]), "blocks": {}}
    for name, block in (("node", node_block), ("edge", edge_block)):
        row_energy = np.sum(block * block, axis=1)
        col_rms = np.sqrt(np.mean(block * block, axis=0))
        out["blocks"][name] = {
            "dim": int(block.shape[1]),
            "nonzero_row_fraction": float(np.mean(row_energy > 0.0)),
            "nonzero_row_count": int(np.sum(row_energy > 0.0)),
            "nonzero_coordinate_fraction": float(np.mean(col_rms > 0.0)),
            "zero_rms_coordinates": int(np.sum(col_rms <= SCALER_FLOOR)),
            "raw_rms_min": float(col_rms.min()),
            "raw_rms_max": float(col_rms.max()),
            "mean_row_squared_norm": float(np.mean(row_energy)),
        }
    out["nonzero_object_fraction"] = float(
        np.mean(
            np.sum(node_block * node_block, axis=1)
            + np.sum(edge_block * edge_block, axis=1)
            > 0.0
        )
    )
    out["official_test_loaded"] = False
    return out


# ---------------------------------------------------------------------------
# 3. unlabeled dictionary fitting (existing K-SVD pipeline, frozen budget)
# ---------------------------------------------------------------------------


def fit_struct_dictionary(
    residual_train: np.ndarray,
    *,
    atoms: int = K_ATOMS,
    s: int = SPARSITY,
    epochs: int = DICT_EPOCHS,
    seed: int = DICT_SEED,
    log: Any = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Unlabeled K-SVD of the structural residual ``r = (I-UU^T) phi``."""
    X = np.asarray(residual_train, dtype=np.float64)
    D, info = sdb.fit_ksvd(X, atoms=int(atoms), s=int(s), epochs=int(epochs), seed=int(seed), log=log)
    return np.asarray(D, dtype=np.float32), info


def fit_corr_dictionary(
    corr_scaled_train: np.ndarray,
    *,
    atoms: int = K_ATOMS,
    s: int = SPARSITY,
    epochs: int = DICT_EPOCHS,
    seed: int = DICT_SEED,
    log: Any = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Unlabeled K-SVD of the scaled correspondence object (shared dictionary)."""
    X = np.asarray(corr_scaled_train, dtype=np.float64)
    if X.shape[1] != CORR_DIM:
        raise RuntimeError(f"C width {X.shape[1]} != {CORR_DIM}")
    D, info = sdb.fit_ksvd(X, atoms=int(atoms), s=int(s), epochs=int(epochs), seed=int(seed), log=log)
    return np.asarray(D, dtype=np.float32), info


# ---------------------------------------------------------------------------
# 4. PCA16 control (train-only affine PCA of the same real object)
# ---------------------------------------------------------------------------


@dataclass
class PCA16:
    """Affine rank-16 PCA of the scaled C object (train fit only)."""

    mean: np.ndarray       # [536]
    components: np.ndarray  # [16, 536]

    def codes(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean[None, :]) @ self.components.T

    def to_json(self) -> dict[str, Any]:
        return {
            "dim": int(self.mean.shape[0]),
            "rank": int(self.components.shape[0]),
            "mean": [float(v) for v in self.mean.tolist()],
            "components": [[float(v) for v in row] for row in self.components.tolist()],
            "fit_split": "official train",
            "official_test_loaded": False,
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "PCA16":
        return cls(
            mean=np.asarray(payload["mean"], dtype=np.float64),
            components=np.asarray(payload["components"], dtype=np.float64),
        )


def fit_pca16(X: np.ndarray, rank: int = K_ATOMS) -> PCA16:
    """Rank-``rank`` affine PCA of ``X`` via the 536x536 Gram eigendecomposition.

    The Gram route is algebraically identical to ``sdb.fit_pca_rank`` (the
    principal right singular vectors of the centred matrix) but avoids the
    ``[N, 536]`` full SVD.  Component signs are fixed deterministically
    (largest-|.| entry positive) so the artifact is reproducible.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.shape[1] != CORR_DIM:
        raise RuntimeError(f"C width {X.shape[1]} != {CORR_DIM}")
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
    return PCA16(mean=mean, components=components)


def pca16_report(pca: PCA16, X: np.ndarray) -> dict[str, Any]:
    codes = pca.codes(X)
    return {
        "rank": int(pca.components.shape[0]),
        "mean_norm": float(np.linalg.norm(pca.mean)),
        "code_rms": [float(np.sqrt(np.mean(codes[:, i] ** 2))) for i in range(codes.shape[1])],
        "code_absmax": [float(np.abs(codes[:, i]).max()) for i in range(codes.shape[1])],
        "official_test_loaded": False,
    }


# ---------------------------------------------------------------------------
# 5. the model: frozen Sem108 readout with a swappable coordinate
# ---------------------------------------------------------------------------


class RoleCorrModel(sem.SEM108Model):
    """``SEM108Model`` whose 33-D coordinate is built explicitly per arm.

    * ``TOPO``    : ``[ c~ ; IHT10(colnorm((I-UU^T) D), r) ]`` (K=32, s=8);
    * ``CORR``    : ``[ c~ ; IHT10(D_S_perp, r) (16) ; alpha_C (16) ]``;
    * ``CORR-PCA``: same but ``alpha_C = (C - mean) @ components^T``.

    The node/edge bindings keep the parent width (33 / 99), so the readout
    parameter shapes are identical across arms.  Dictionaries are frozen by
    default (``requires_grad=False``); the reconstruction term is a diagnostic
    only (the frozen dictionaries make it inert).
    """

    def __init__(
        self,
        config: Any,
        dictionary: np.ndarray,
        *,
        subspace: cssd.CommonSubspace,
        arm: str = ARM_CORR,
        dictionary_corr: np.ndarray | None = None,
        corr_mode: str = CORR_MODE_SPARSE,
        struct_sparsity: int | None = None,
        corr_sparsity: int = SPARSITY,
        iht_steps: int = IHT_STEPS,
        freeze_dictionary: bool = True,
        pca_mean: np.ndarray | None = None,
        pca_components: np.ndarray | None = None,
    ) -> None:
        arm = str(arm)
        if arm not in (ARM_TOPO, ARM_CORR, ARM_CORR_SHUF, ARM_CORR_PCA):
            raise ValueError(f"unknown arm {arm!r}")
        corr_mode = str(corr_mode)
        if corr_mode not in CORR_MODES:
            raise ValueError(f"unknown corr_mode {corr_mode!r}")
        if arm == ARM_CORR_PCA and corr_mode != CORR_MODE_PCA:
            raise ValueError("CORR-PCA requires corr_mode='pca'")
        placeholder = np.zeros((PHI_DIM, int(config.K)), dtype=np.float32)
        super().__init__(
            config,
            placeholder,
            subspace=subspace,
            node_binding="paired",
            edge_binding="paired",
            coding="sparse",
        )
        dictionary = np.asarray(dictionary, dtype=np.float32)
        expected_width = BASE_ATOMS if arm == ARM_TOPO else K_ATOMS
        if dictionary.shape != (PHI_DIM, expected_width):
            raise RuntimeError(
                f"arm {arm}: structural dictionary {dictionary.shape} != "
                f"({PHI_DIM}, {expected_width})"
            )
        self.D = nn.Parameter(
            torch.as_tensor(dictionary, dtype=torch.float32).clone(),
            requires_grad=not bool(freeze_dictionary),
        )
        # The parent repairs degenerate projected columns at initialization; the
        # placeholder above means the repair has to be re-applied to the real
        # dictionary so the operator matches ``build_sem108_model`` bit-for-bit.
        self.dead_column_fallbacks += self._ensure_live_projection_()
        self.arm = arm
        self.corr_mode = corr_mode
        self.struct_sparsity = int(
            (BASE_SPARSITY if arm == ARM_TOPO else SPARSITY)
            if struct_sparsity is None
            else struct_sparsity
        )
        self.corr_sparsity = int(corr_sparsity)
        self.iht_steps = int(iht_steps)
        self._pending_corr: torch.Tensor | None = None
        self.inference_zero_struct = False
        self.inference_zero_corr = False
        self.D_corr = None
        self.pca_mean = None
        self.pca_components = None
        if arm == ARM_TOPO:
            if dictionary_corr is not None:
                raise RuntimeError("TOPO has no correspondence dictionary")
            if int(self.struct_sparsity) != BASE_SPARSITY:
                raise RuntimeError("TOPO sparsity is frozen to the SDB value 8")
        else:
            if corr_mode == CORR_MODE_SPARSE:
                if dictionary_corr is None:
                    raise RuntimeError(f"arm {arm} requires a correspondence dictionary")
                dictionary_corr = np.asarray(dictionary_corr, dtype=np.float32)
                if dictionary_corr.shape != (CORR_DIM, K_ATOMS):
                    raise RuntimeError(
                        f"correspondence dictionary {dictionary_corr.shape} != "
                        f"({CORR_DIM}, {K_ATOMS})"
                    )
                self.D_corr = nn.Parameter(
                    torch.as_tensor(dictionary_corr, dtype=torch.float32).clone(),
                    requires_grad=not bool(freeze_dictionary),
                )
            elif dictionary_corr is not None:
                raise RuntimeError("pca corr_mode does not take a correspondence dictionary")
            if corr_mode == CORR_MODE_PCA:
                if pca_mean is None or pca_components is None:
                    raise RuntimeError("PCA corr_mode requires the fitted mean/components")
                self.register_buffer(
                    "corr_pca_mean",
                    torch.as_tensor(np.asarray(pca_mean, dtype=np.float32)),
                )
                self.register_buffer(
                    "corr_pca_components",
                    torch.as_tensor(np.asarray(pca_components, dtype=np.float32)),
                )
                self.pca_mean = np.asarray(pca_mean, dtype=np.float64)
                self.pca_components = np.asarray(pca_components, dtype=np.float64)

    # -- frozen coordinate -----------------------------------------------------

    def residual_dictionary(self) -> torch.Tensor:
        """``Dbar_perp = colnorm((I - UU^T) D)`` (CSSD hard constraint)."""
        return super().residual_dictionary()

    def structural_codes(self, phi: torch.Tensor) -> torch.Tensor:
        U = self.U.to(dtype=phi.dtype)
        c = phi @ U
        r = phi - c @ U.t()
        return v0.tied_iht_codes(
            self.residual_dictionary(), r, s=int(self.struct_sparsity), steps=int(self.iht_steps)
        )

    def corr_codes(self, corr: torch.Tensor) -> torch.Tensor:
        if self.corr_mode == CORR_MODE_SPARSE:
            return v0.tied_iht_codes(
                v0.normalized_dictionary(self.D_corr),
                corr,
                s=int(self.corr_sparsity),
                steps=int(self.iht_steps),
            )
        mean = self.corr_pca_mean.to(dtype=corr.dtype)
        components = self.corr_pca_components.to(dtype=corr.dtype)
        return (corr - mean) @ components.t()

    def code(self, phi: torch.Tensor, corr: torch.Tensor | None = None) -> torch.Tensor:
        U = self.U.to(dtype=phi.dtype)
        c = phi @ U
        alpha_s = self.structural_codes(phi)
        if self.inference_zero_struct:
            alpha_s = torch.zeros_like(alpha_s)
        scale = self.common_rms.to(dtype=phi.dtype)
        if self.arm == ARM_TOPO:
            return torch.cat([c / scale, alpha_s], dim=1)
        if corr is None:
            corr = self._pending_corr
        if corr is None:
            raise RuntimeError(
                "RoleCorrModel.corr_override/corr is not set for this forward"
            )
        alpha_c = self.corr_codes(corr)
        if self.inference_zero_corr:
            alpha_c = torch.zeros_like(alpha_c)
        return torch.cat([c / scale, alpha_s, alpha_c], dim=1)

    def forward(self, data: Any, **kwargs: Any):
        corr = getattr(data, "corr_vec", None)
        self._pending_corr = corr
        try:
            return super().forward(data, **kwargs)
        finally:
            self._pending_corr = None

    # -- diagnostic reconstruction (inert with frozen dictionaries) ------------

    def reconstruct(self, phi: torch.Tensor, coord: torch.Tensor) -> torch.Tensor:
        # TOPO keeps the parent's 32-atom coordinate; the candidate's structural
        # slice is the first 16 coordinates after the common one.
        alpha_s = (
            coord[:, self.common_dim :] if self.arm == ARM_TOPO else coord[:, STRUCT_SLICE]
        )
        return alpha_s @ self.residual_dictionary().t()

    def reconstruction_loss(self, phi: torch.Tensor, coord: torch.Tensor) -> torch.Tensor:
        U = self.U.to(dtype=phi.dtype)
        r = phi - (phi @ U) @ U.t()
        phi_hat = self.reconstruct(phi, coord)
        numerator = ((r - phi_hat) ** 2).sum(dim=1)
        denominator = (phi**2).sum(dim=1) + v0.EPS
        return (numerator / denominator).mean()


def build_rolecorr_model(
    *,
    arm: str,
    dictionary: np.ndarray,
    seed: int,
    subspace: cssd.CommonSubspace,
    dictionary_corr: np.ndarray | None = None,
    corr_mode: str = CORR_MODE_SPARSE,
    pca_mean: np.ndarray | None = None,
    pca_components: np.ndarray | None = None,
    freeze_dictionary: bool = True,
    reference_state: Mapping[str, torch.Tensor] | None = None,
) -> RoleCorrModel:
    """Build an arm with the exact frozen ``build_sem108_model`` RNG stream.

    ``reference_state`` (optional) copies every same-shaped entry after
    construction; the round uses it to make the readout initialisation of all
    arms bit-identical to the baseline arm.
    """
    torch.manual_seed(int(seed))
    model = RoleCorrModel(
        cm.H1_CONFIG,
        dictionary,
        subspace=subspace,
        arm=str(arm),
        dictionary_corr=dictionary_corr,
        corr_mode=str(corr_mode),
        freeze_dictionary=bool(freeze_dictionary),
        pca_mean=pca_mean,
        pca_components=pca_components,
    )
    if reference_state is not None:
        with torch.no_grad():
            state = model.state_dict()
            copied = 0
            for key, value in reference_state.items():
                if key in state and tuple(state[key].shape) == tuple(value.shape):
                    if key in ("D", "D_corr"):
                        continue
                    state[key].copy_(value)
                    copied += 1
        model._shared_init_keys = copied  # type: ignore[attr-defined]
    return model


def coordinate_slices() -> dict[str, Any]:
    return {
        "common": [0, COMMON_DIM],
        "struct": [int(STRUCT_SLICE.start), int(STRUCT_SLICE.stop)],
        "corr": [int(CORR_SLICE.start), int(CORR_SLICE.stop)],
        "coord_dim": int(COORD_DIM),
    }


def coordinate_payload(model: RoleCorrModel) -> dict[str, Any]:
    """Inference-only parameter/geometry accounting of a built model."""
    payload: dict[str, Any] = {
        "arm": str(model.arm),
        "coordinate": coordinate_slices(),
        "struct_dictionary_shape": list(model.D.shape),
        "struct_sparsity": int(model.struct_sparsity),
        "corr_sparsity": int(model.corr_sparsity),
        "iht_steps": int(model.iht_steps),
        "corr_mode": str(model.corr_mode),
        "corr_dictionary_shape": (
            None if model.D_corr is None else list(model.D_corr.shape)
        ),
        "dictionary_trainable": bool(model.D.requires_grad),
        "corr_dictionary_trainable": (
            None if model.D_corr is None else bool(model.D_corr.requires_grad)
        ),
        "trainable_parameters": int(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        ),
        "total_parameters": int(sum(p.numel() for p in model.parameters())),
        "W_A_S": list(model.W_A_S.shape),
        "W_E_S": list(model.W_E_S.shape),
        "official_test_loaded": False,
    }
    official_test_blocker(payload)
    return payload


# ---------------------------------------------------------------------------
# 6. coordinate diagnostics / usage
# ---------------------------------------------------------------------------


def code_usage(codes: np.ndarray, atoms: int | None = None) -> dict[str, Any]:
    """Active / effective atom usage of a sparse code matrix ``[N, K]``."""
    codes = np.asarray(codes, dtype=np.float64)
    frequencies = (np.abs(codes) > 0.0).mean(axis=0)
    effective = int(atoms or codes.shape[1])
    probability = frequencies / max(float(frequencies.sum()), 1e-12)
    nonzero = probability[probability > 0.0]
    entropy = float(-(nonzero * np.log(nonzero)).sum()) if nonzero.size else 0.0
    order = np.sort(frequencies)[::-1]
    return {
        "active_atoms": int((frequencies > 0.0).sum()),
        "atoms": int(effective),
        "effective_atoms": float(math.exp(entropy)),
        "max_activation_rate": float(frequencies.max()) if frequencies.size else 0.0,
        "top1_share": float(order[:1].sum()) if order.size else 0.0,
        "top4_share": float(order[:4].sum()) if order.size else 0.0,
        "exact_l0_mean": float((np.abs(codes) > 0.0).sum(axis=1).mean()),
        "frequencies": [float(v) for v in frequencies.tolist()],
        "official_test_loaded": False,
    }


def zero_corr_purity(
    model: RoleCorrModel, phi: torch.Tensor, corr: torch.Tensor
) -> dict[str, Any]:
    """Verify that zeroing ``alpha_C`` touches only the last 16 coordinates."""
    coord_full = model.code(phi, corr)
    model.inference_zero_corr = True
    try:
        coord_zero = model.code(phi, corr)
    finally:
        model.inference_zero_corr = False
    head_equal = bool(torch.equal(coord_full[:, :CORR_SLICE.start], coord_zero[:, :CORR_SLICE.start]))
    return {
        "head_columns_bit_identical": head_equal,
        "corr_columns_zero": bool(torch.equal(coord_zero[:, CORR_SLICE], torch.zeros_like(coord_zero[:, CORR_SLICE]))),
        "max_abs_shift_on_zeroed_slice": float(
            (coord_full[:, CORR_SLICE] - coord_zero[:, CORR_SLICE]).abs().max()
        ),
        "official_test_loaded": False,
    }


__all__ = [
    "PROTOCOL_VERSION",
    "PHI_DIM",
    "NODE_BASIS_DIM",
    "EDGE_BASIS_DIM",
    "ATOM_CATEGORIES",
    "BOND_CATEGORIES",
    "PATCH_RADIUS",
    "NODE_ROLE_SLICE",
    "NODE_SHELLS",
    "EDGE_ROLE_SLICE",
    "EDGE_SHELL_PAIRS",
    "ALL_SHELL_PAIRS",
    "N_SHELLS",
    "NODE_ROLE_DIM",
    "EDGE_ROLE_DIM",
    "NODE_BLOCK_DIM",
    "EDGE_BLOCK_DIM",
    "CORR_DIM",
    "CORR_BLOCKS",
    "CORR_BLOCK_SLICES",
    "K_ATOMS",
    "SPARSITY",
    "IHT_STEPS",
    "DICT_EPOCHS",
    "DICT_SEED",
    "SCALER_FLOOR",
    "SCALER_EPS",
    "COMMON_DIM",
    "STRUCT_DIM",
    "COORD_DIM",
    "STRUCT_SLICE",
    "CORR_SLICE",
    "BASE_ATOMS",
    "BASE_SPARSITY",
    "ARM_TOPO",
    "ARM_CORR",
    "ARM_CORR_SHUF",
    "ARM_CORR_PCA",
    "ARMS",
    "CORR_MODE_SPARSE",
    "CORR_MODE_PCA",
    "CORR_MODES",
    "SHUFFLE_SEEDS",
    "PERMUTE_DELTA_FLOOR",
    "SCREEN_RELATIVE_GATE",
    "GATE_MECHANISM_DIRECTIONAL",
    "GATE_MECHANISM_CLEAR",
    "VERDICTS",
    "official_test_blocker",
    "cpu_only_guard",
    "CorrBlocks",
    "corr_blocks",
    "corr_vector",
    "CorrScaler",
    "fit_corr_scaler",
    "apply_corr_scaler",
    "corr_block_report",
    "fit_struct_dictionary",
    "fit_corr_dictionary",
    "PCA16",
    "fit_pca16",
    "pca16_report",
    "RoleCorrModel",
    "build_rolecorr_model",
    "coordinate_slices",
    "coordinate_payload",
    "code_usage",
    "zero_corr_purity",
]
