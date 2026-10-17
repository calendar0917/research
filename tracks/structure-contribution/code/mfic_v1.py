"""MFIC-v1 — one-shot matrix-function structure composition on the frozen BJOC family.

Protocol ``cscl-bjoc-mfic-v1`` (``results/mfic_v1/PROTOCOL.md``).

What this module is
-------------------
The frozen BJOC model encodes atoms, physical bonds and junctions once, then the
Null composer builds one per-bond content vector ``c``.  The graph-level reader
consumes ``[sum_b c_b, sum_i a_i, sum_i a_i^2, global]`` (321 dims).

This round replaces the ``sum_i a_i^2`` block by a **single analytic,
label-blind, graph-level quadratic form** built from the physical heavy-atom
adjacency matrix ``A``::

    D_ii = sum_j A_ij
    S    = D^{-1/2} A D^{-1/2}          (isolated vertex -> zero row/column)
    R    = solve(I - alpha*S, I)        alpha = 0.65, frozen
    K    = R - I - alpha*S = sum_{k>=2} alpha^k S^k

    q_i  = Linear(96 -> 16)(a_i)        shared projection of the *final* atom
    C    = q^T K q / max(1, n_atoms)    only graph-level contraction
    reader_in = [sum_b c_b (96), sum_i a_i (96), upper_triangle(C) (136),
                 global (33)]           = 361 dims

``K`` is never written back into a per-atom or per-bond state.  It is not
learned, does not use ``y``, and is cacheable at preprocessing time.  The reader
is two hidden layers of width 256, SiLU, dropout 0, exactly as in P3.

Control arm ``MFIC-U`` receives the same dense matmul, same shapes, same
parameters and the same training budget, but with a content-null kernel built
from the *same* per-molecule diagonal mass and off-diagonal total mass::

    d = trace(K)/n
    o = (sum(K) - trace(K)) / (n*(n-1))       n >= 2
    K_U = d*I + o*(11^T - I)                  n = 1 -> K_U = d*I (= 0)

``K_U`` is a diagnostic content-null, **not** a strict graph-null: it does not
preserve the full spectrum, row sums or spatial distribution of ``K``.  A
``MFIC-R - MFIC-U`` difference therefore cannot be interpreted as a pure causal
effect of chemistry.

Nothing here uses a GNN, attention, Transformer, learned adjacency, or iterative
neighbourhood propagation.  The only matrix function is the fixed resolvent
above and it enters the computation once, at final graph-level readout.
"""

from __future__ import annotations

import hashlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_CODE_DIR = Path(__file__).resolve().parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import bjoc_masked_pretraining_v1 as mp  # noqa: E402
import bjoc_perfmech_long_v1 as bp  # noqa: E402
import structural_relation_composer_v1 as src  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402

PROTOCOL_ID = "cscl-bjoc-mfic-v1"
SPLIT_TAG = "V"
SPLIT_SALT = "cscl-bjoc-mfic-v1"

SEEDS = (0, 1)
ARMS = ("P3", "MFIC-R", "MFIC-U")
RECIPE_KEY = "R3"

ALPHA = 0.65
Q_DIM = 16
READER_HIDDEN = 256
READER_LAYERS = 2
READER_IN = 361
C_UPPER = Q_DIM * (Q_DIM + 1) // 2

P3_PARAMETERS = 413_505
P3_READER_PARAMETERS = 148_481
MFIC_PARAMETERS = 425_297
MFIC_READER_PARAMETERS = 158_721
Q_PROJECTION_PARAMETERS = 96 * Q_DIM + Q_DIM
SHARED_PARAMETERS = 265_024
PARAM_CAP = 500_000

MEANINGFUL_DELTA = 0.004
NULL_DELTA = 0.003
TAIL_THRESHOLD = -3.0
TOP_K_REMOVED = 10
BOOTSTRAP_N = 5_000
BOOTSTRAP_SEED = 0
RMSE_TOL = 0.005
TAIL_TOL = 0.005
PARTICIPATION_MEAN = 0.0025
PARTICIPATION_MAX = 0.02
GPU_BUDGET_CAP_HOURS = 3.0
SMOKE_STOP_HOURS = 2.7

CPU_FOLDS = 4
CPU_RIDGE_ALPHA = 1.0
CPU_PCA_COMPONENTS = Q_DIM
CPU_MIN_MEAN_IMPROVEMENT = 0.020
CPU_MIN_FOLDS_IMPROVED = 3
CPU_TRIM_K = 10

ZONES = ("monitor", "locked")
SHARED_MODULE_NAMES = ("atom_encoder", "bond_encoder", "junction_encoder", "group_composer")


class MFICError(RuntimeError):
    """Hard protocol violation (raises instead of silently continuing)."""


def require(condition: Any, message: str) -> None:
    if not condition:
        raise MFICError(message)


# ---------------------------------------------------------------------------
# analytic kernel
# ---------------------------------------------------------------------------


def physical_adjacency(obj: sro.MolObjects) -> np.ndarray:
    """Dense 0/1 adjacency of the unique physical heavy-atom bonds."""
    n = int(obj.n_atoms)
    A = np.zeros((n, n), dtype=np.float64)
    seen: set[tuple[int, int]] = set()
    for u, v in obj.bond_index:
        u, v = int(u), int(v)
        require(u != v, f"self loop in molecule {obj.mol_index}")
        key = (u, v) if u < v else (v, u)
        require(key not in seen, f"duplicate physical bond {key} in molecule {obj.mol_index}")
        seen.add(key)
        A[u, v] = 1.0
        A[v, u] = 1.0
    return A


def adjacency_from_edges(n: int, edges: Sequence[tuple[int, int]]) -> np.ndarray:
    A = np.zeros((int(n), int(n)), dtype=np.float64)
    seen: set[tuple[int, int]] = set()
    for u, v in edges:
        u, v = int(u), int(v)
        require(u != v, "self loop in synthetic graph")
        key = (u, v) if u < v else (v, u)
        require(key not in seen, f"duplicate synthetic edge {key}")
        seen.add(key)
        A[u, v] = 1.0
        A[v, u] = 1.0
    return A


def kernel_from_adjacency(A: np.ndarray, alpha: float = ALPHA) -> dict[str, np.ndarray | float]:
    """Stable resolvent kernel ``K = R - I - alpha*S``.

    Uses a linear solve (not an explicit inverse followed by powers).  All
    returned arrays are float64 and finite; symmetry is asserted.
    """
    A = np.asarray(A, dtype=np.float64)
    require(A.ndim == 2 and A.shape[0] == A.shape[1], "adjacency must be square")
    n = int(A.shape[0])
    require(np.allclose(A, A.T, atol=0.0), "adjacency is not symmetric")
    require(np.all((A == 0.0) | (A == 1.0)), "adjacency must be 0/1")
    deg = A.sum(1)
    inv_sqrt = np.zeros_like(deg)
    nz = deg > 0.0
    inv_sqrt[nz] = 1.0 / np.sqrt(deg[nz])
    S = (A * inv_sqrt[:, None]) * inv_sqrt[None, :]
    M = np.eye(n, dtype=np.float64) - float(alpha) * S
    min_eig = float(np.linalg.eigvalsh(M).min())
    require(min_eig > 0.0, f"I - alpha*S is not positive definite: min eig {min_eig}")
    R = np.linalg.solve(M, np.eye(n, dtype=np.float64))
    K = R - np.eye(n, dtype=np.float64) - float(alpha) * S
    require(np.all(np.isfinite(R)) and np.all(np.isfinite(K)), "non-finite R/K")
    require(np.allclose(R, R.T, atol=1e-10), "R is not symmetric")
    require(np.allclose(K, K.T, atol=1e-10), "K is not symmetric")
    return {"A": A, "S": S, "R": R, "K": K, "min_eig": min_eig}


def content_null_kernel(K: np.ndarray) -> np.ndarray:
    """Content-null kernel with the same diagonal and off-diagonal total mass."""
    K = np.asarray(K, dtype=np.float64)
    n = int(K.shape[0])
    require(n >= 1, "empty kernel")
    if n == 1:
        return K.copy()
    tr = float(np.trace(K))
    total = float(K.sum())
    d = tr / float(n)
    o = (total - tr) / float(n * (n - 1))
    K_U = d * np.eye(n, dtype=np.float64) + o * (np.ones((n, n), dtype=np.float64) - np.eye(n, dtype=np.float64))
    return K_U


def kernel_stats(K: np.ndarray) -> dict[str, float]:
    K = np.asarray(K, dtype=np.float64)
    n = int(K.shape[0])
    return {
        "n_atoms": n,
        "trace": float(np.trace(K)),
        "sum": float(K.sum()),
        "offdiag_sum": float(K.sum() - np.trace(K)),
        "fro_norm": float(np.linalg.norm(K, "fro")),
        "max_abs": float(np.abs(K).max()) if K.size else 0.0,
        "rank": int(np.linalg.matrix_rank(K, tol=1e-10)),
    }


def walk_count(A: np.ndarray, k: int) -> float:
    """Number of walks of exact length k (sum of entries of A^k)."""
    P = np.eye(int(A.shape[0]), dtype=np.float64)
    for _ in range(int(k)):
        P = P @ A
    return float(P.sum())


def truncated_series(S: np.ndarray, alpha: float = ALPHA, kmax: int = 40) -> np.ndarray:
    """``sum_{k>=2} alpha^k S^k`` by explicit powers (diagnostic only)."""
    n = int(S.shape[0])
    P = np.eye(n, dtype=np.float64)
    out = np.zeros((n, n), dtype=np.float64)
    for k in range(1, int(kmax) + 1):
        P = P @ S
        if k >= 2:
            out += (float(alpha) ** k) * P
    return out


# ---------------------------------------------------------------------------
# unsupervised PCA on raw atom descriptors (CPU probe only)
# ---------------------------------------------------------------------------


def pca_fit(X: np.ndarray, n_components: int = CPU_PCA_COMPONENTS) -> dict[str, Any]:
    """Deterministic unsupervised PCA via SVD (center only, no scaling).

    Component signs are fixed deterministically: the largest-|loading| entry of
    each component is made positive.  No labels enter this function.
    """
    X = np.asarray(X, dtype=np.float64)
    require(X.ndim == 2 and X.shape[0] >= 1, "PCA needs a non-empty 2-D matrix")
    n_comp = int(n_components)
    require(1 <= n_comp <= min(X.shape), f"invalid PCA component count {n_comp}")
    mean = X.mean(0)
    Xc = X - mean
    _u, sv, vt = np.linalg.svd(Xc, full_matrices=False)
    comp = vt[:n_comp].copy()
    for j in range(n_comp):
        idx = int(np.argmax(np.abs(comp[j])))
        if comp[j, idx] < 0:
            comp[j] *= -1.0
    total = float((sv ** 2).sum())
    explained = (sv[:n_comp] ** 2) / max(total, 1e-30)
    return {
        "mean": mean,
        "components": comp,
        "explained_variance_ratio": explained,
        "explained_variance_ratio_sum": float(explained.sum()),
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_components": n_comp,
    }


def pca_transform(X: np.ndarray, pca: Mapping[str, Any]) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    return (X - np.asarray(pca["mean"], dtype=np.float64)) @ np.asarray(pca["components"], dtype=np.float64).T


_UPPER_I, _UPPER_J = np.triu_indices(Q_DIM)


def upper_triangle(C: np.ndarray) -> np.ndarray:
    C = np.asarray(C, dtype=np.float64)
    require(C.shape == (Q_DIM, Q_DIM), f"C shape {C.shape} != {(Q_DIM, Q_DIM)}")
    return C[_UPPER_I, _UPPER_J]


def matrix_features_from_projection(Q: np.ndarray, K: np.ndarray) -> np.ndarray:
    """``upper(Q^T K Q / max(1,n))`` with ``Q`` already projected atoms."""
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    n = int(Q.shape[0])
    require(K.shape == (n, n), "Q/K size mismatch")
    if n == 0:
        return np.zeros(C_UPPER, dtype=np.float64)
    C = (Q.T @ K @ Q) / float(max(1, n))
    return upper_triangle(C)


def matrix_features_for_molecule(atom_feat: np.ndarray, K: np.ndarray, pca: Mapping[str, Any]) -> np.ndarray:
    Q = pca_transform(atom_feat, pca)
    return matrix_features_from_projection(Q, K)


def base_features(obj: sro.MolObjects) -> np.ndarray:
    """CPU-Base: raw atom-type counts, bond-type counts, global33, raw size.

    No target component, no model prediction, no label-derived statistic.
    """
    atom_counts = obj.atom_feat[:, : sro.ATOM_CATEGORIES].sum(0).astype(np.float64)
    bond_counts = np.bincount(np.asarray(obj.bond_type, dtype=np.int64), minlength=sro.BOND_CATEGORIES).astype(np.float64)
    size = np.asarray(
        [obj.n_atoms, obj.bond_index.shape[0], obj.junc_index.shape[0]],
        dtype=np.float64,
    )
    return np.concatenate([atom_counts, bond_counts, np.asarray(obj.global_feat, dtype=np.float64), size], 0)


BASE_DIM = sro.ATOM_CATEGORIES + sro.BOND_CATEGORIES + sro.GLOBAL_DIM + 3
MATRIX_DIM = BASE_DIM + C_UPPER


# ---------------------------------------------------------------------------
# grouped folds and Ridge probe
# ---------------------------------------------------------------------------


def grouped_folds(
    rows: Sequence[int],
    identity_of_row: Mapping[int | str, str],
    *,
    n_folds: int = CPU_FOLDS,
    salt: str = SPLIT_SALT,
) -> list[list[int]]:
    """Deterministic identity-grouped fold assignment, label-blind.

    Groups are ordered by ``sha256(salt|fold|identity)``; each whole group is
    greedily assigned to the currently smallest fold (ties: fewer groups, then
    fold index).  No ``y`` is read here.
    """
    rows = [int(r) for r in rows]
    groups: dict[str, list[int]] = {}
    for r in rows:
        key = str(identity_of_row.get(r, identity_of_row.get(str(r))))
        require(key != "None", f"missing identity for row {r}")
        groups.setdefault(key, []).append(r)
    order = sorted(groups, key=lambda g: hashlib.sha256(f"{salt}|fold|{g}".encode()).hexdigest())
    folds: list[list[int]] = [[] for _ in range(int(n_folds))]
    n_groups = [0 for _ in range(int(n_folds))]
    for key in order:
        members = sorted(groups[key])
        f = min(range(int(n_folds)), key=lambda i: (len(folds[i]), n_groups[i], i))
        folds[f].extend(members)
        n_groups[f] += 1
    for f in folds:
        f.sort()
    require(sorted(r for f in folds for r in f) == sorted(rows), "folds do not cover the rows exactly")
    for key, members in groups.items():
        where = {i for i, f in enumerate(folds) for r in f if r in set(members)}
        require(len(where) == 1, f"identity group crosses folds: {key}")
    return folds


def ridge_fit_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    *,
    alpha: float = CPU_RIDGE_ALPHA,
) -> dict[str, Any]:
    """Ridge with intercept, train-fold-only standardisation, fixed alpha=1."""
    Xtr = np.asarray(X_train, dtype=np.float64)
    Xev = np.asarray(X_eval, dtype=np.float64)
    ytr = np.asarray(y_train, dtype=np.float64).reshape(-1)
    require(Xtr.shape[0] == ytr.size, "X/y train length mismatch")
    require(Xtr.shape[1] == Xev.shape[1], "X train/eval width mismatch")
    mu = Xtr.mean(0)
    sd = Xtr.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    Ztr = (Xtr - mu) / sd
    Zev = (Xev - mu) / sd
    ybar = float(ytr.mean())
    gram = Ztr.T @ Ztr + float(alpha) * np.eye(Ztr.shape[1], dtype=np.float64)
    rhs = Ztr.T @ (ytr - ybar)
    w = np.linalg.solve(gram, rhs)
    pred = Zev @ w + ybar
    require(np.all(np.isfinite(pred)), "non-finite Ridge prediction")
    return {
        "pred": pred,
        "weights": w,
        "intercept": ybar,
        "feature_mean": mu,
        "feature_scale": sd,
        "n_zero_variance_features": int(np.sum(Xtr.std(0) < 1e-8)),
        "alpha": float(alpha),
    }


def mae(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.abs(np.asarray(y, dtype=np.float64) - np.asarray(pred, dtype=np.float64)).mean())


# ---------------------------------------------------------------------------
# model config / model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MFICConfig(src.BJOCConfig):
    q_dim: int = Q_DIM

    @property
    def reader_in(self) -> int:
        # [sum_composer 96, sum_atom 96, upper(C) 136, global 33] = 361.
        return self.compose_out + self.atom_out + C_UPPER + sro.GLOBAL_DIM

    def mfic_inventory(self) -> dict[str, int]:
        base = self.inventory()
        q = self.atom_out * self.q_dim + self.q_dim
        return {
            "atom_encoder": base["atom_encoder"],
            "bond_encoder": base["bond_encoder"],
            "junction_encoder": base["junction_encoder"],
            "group_composer": base["group_composer"],
            "q_projection": q,
            "reader": base["reader"],
            "total": base["total"] + q,
        }


class BJOCMFIC(src.BJOC):
    """BJOC with one analytic graph-level quadratic form replacing ``sum a_i^2``."""

    def __init__(
        self,
        config: MFICConfig,
        global_mean: np.ndarray,
        global_scale: np.ndarray,
        organization_mode: str = src.NULL,
        kernel_mode: str = "R",
    ):
        super().__init__(config, global_mean, global_scale, organization_mode)
        if kernel_mode not in ("R", "U"):
            raise ValueError(f"unknown kernel_mode {kernel_mode!r}")
        self.kernel_mode = kernel_mode
        self.q_proj = nn.Linear(config.atom_out, config.q_dim)

    def _quadratic_form(self, batch: Mapping[str, torch.Tensor], q: torch.Tensor) -> torch.Tensor:
        """One-shot ``q^T K q / max(1,n)`` per graph using padded dense K."""
        K_pad = batch["k_pad"]
        n_graphs = int(batch["n_graphs"])
        max_n = int(K_pad.shape[1])
        q_pad = torch.zeros((n_graphs, max_n, q.shape[1]), device=q.device, dtype=q.dtype)
        q_pad[batch["atom_batch"], batch["atom_pos"]] = q
        C = torch.bmm(q_pad.transpose(1, 2), torch.bmm(K_pad, q_pad))
        denom = batch["n_atoms_per_graph"].to(dtype=C.dtype).clamp(min=1.0).view(n_graphs, 1, 1)
        return C / denom

    def forward_features(
        self,
        batch: Mapping[str, torch.Tensor],
        organization_mode: str | None = None,
    ) -> dict[str, torch.Tensor]:
        mode = self.organization_mode if organization_mode is None else organization_mode
        a = self.encode_atoms(batch)
        b = self.encode_bonds(a, batch)
        w = self.encode_junctions(a, b, batch)
        W, V, n_b = self.group_to_bonds(w, batch, mode)
        compose_in = torch.cat(
            [b, W, V, torch.log1p(n_b).unsqueeze(1), batch["bond_feat"]],
            dim=1,
        )
        c = self.group_composer(compose_in)
        n_graphs = int(batch["n_graphs"])
        z = torch.zeros((n_graphs, c.shape[1]), device=c.device, dtype=c.dtype)
        z.index_add_(0, batch["bond_batch"], c)
        a_sum = torch.zeros((n_graphs, a.shape[1]), device=a.device, dtype=a.dtype)
        a_sum.index_add_(0, batch["atom_batch"], a)
        q = self.q_proj(a)
        C = self._quadratic_form(batch, q)
        tri_i = _UPPER_I_T.to(C.device)
        tri_j = _UPPER_J_T.to(C.device)
        c_upper = C[:, tri_i, tri_j]
        g = (batch["global_feat"].to(a.dtype) - self.global_mean) / self.global_scale
        reader_in = torch.cat([z, a_sum, c_upper, g], dim=1)
        return {
            "a": a,
            "b": b,
            "w": w,
            "W": W,
            "V": V,
            "n_b": n_b,
            "c": c,
            "z": z,
            "a_sum": a_sum,
            "q": q,
            "C": C,
            "c_upper": c_upper,
            "reader_in": reader_in,
        }


_UPPER_I_T = torch.as_tensor(_UPPER_I, dtype=torch.long)
_UPPER_J_T = torch.as_tensor(_UPPER_J, dtype=torch.long)


def build_p3(seed: int, norm: Mapping[str, Any], device: torch.device | str = "cpu") -> src.BJOC:
    return src.build_bjoc(
        src.BJOCConfig(),
        seed=int(seed),
        organization_mode=src.NULL,
        global_mean=np.asarray(norm["global_mean"], dtype=np.float32),
        global_scale=np.asarray(norm["global_scale"], dtype=np.float32),
        device=device,
    )


def build_mfic(
    seed: int,
    norm: Mapping[str, Any],
    device: torch.device | str = "cpu",
    kernel_mode: str = "R",
) -> BJOCMFIC:
    torch.manual_seed(int(seed))
    model = BJOCMFIC(
        MFICConfig(),
        np.asarray(norm["global_mean"], dtype=np.float32),
        np.asarray(norm["global_scale"], dtype=np.float32),
        organization_mode=src.NULL,
        kernel_mode=kernel_mode,
    )
    return model.to(device)


def copy_common_modules(source: src.BJOC, target: src.BJOC) -> None:
    for name in SHARED_MODULE_NAMES:
        getattr(target, name).load_state_dict(getattr(source, name).state_dict())


def build_three_arms(seed: int, norm: Mapping[str, Any], device: torch.device | str = "cpu") -> dict[str, nn.Module]:
    """P3 + MFIC-R + MFIC-U with bit-identical common modules and R/U init."""
    p3 = build_p3(seed, norm, device)
    r = build_mfic(seed, norm, device, kernel_mode="R")
    u = build_mfic(seed, norm, device, kernel_mode="U")
    copy_common_modules(p3, r)
    copy_common_modules(p3, u)
    return {"P3": p3, "MFIC-R": r, "MFIC-U": u}


# ---------------------------------------------------------------------------
# batching with padded K
# ---------------------------------------------------------------------------


def batch_objects_with_k(
    objects: Sequence[sro.MolObjects],
    kernels: Sequence[np.ndarray],
    *,
    dtype: torch.dtype = torch.float32,
) -> dict[str, Any]:
    require(len(objects) == len(kernels), "objects/kernels length mismatch")
    batch = sro.batch_objects(list(objects))
    n_graphs = len(objects)
    max_n = max((int(o.n_atoms) for o in objects), default=0)
    K_pad = torch.zeros((n_graphs, max_n, max_n), dtype=dtype)
    atom_pos = torch.zeros((sum(int(o.n_atoms) for o in objects),), dtype=torch.long)
    n_atoms_per_graph = torch.zeros((n_graphs,), dtype=torch.long)
    off = 0
    for g, (o, K) in enumerate(zip(objects, kernels)):
        n = int(o.n_atoms)
        K = np.asarray(K, dtype=np.float64)
        require(K.shape == (n, n), f"kernel shape {K.shape} != {(n, n)} for molecule {o.mol_index}")
        K_pad[g, :n, :n] = torch.as_tensor(K, dtype=dtype)
        atom_pos[off : off + n] = torch.arange(n, dtype=torch.long)
        n_atoms_per_graph[g] = n
        off += n
    batch["k_pad"] = K_pad
    batch["atom_pos"] = atom_pos
    batch["n_atoms_per_graph"] = n_atoms_per_graph
    return batch


def move_batch(batch: Mapping[str, Any], device: torch.device | str) -> dict[str, Any]:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


# ---------------------------------------------------------------------------
# parameter / hash audits
# ---------------------------------------------------------------------------


def model_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def module_parameter_counts(model: nn.Module) -> dict[str, int]:
    out = {name: int(sum(p.numel() for p in getattr(model, name).parameters())) for name in SHARED_MODULE_NAMES + ("reader",)}
    if hasattr(model, "q_proj"):
        out["q_projection"] = int(sum(p.numel() for p in model.q_proj.parameters()))
    out["total"] = model_parameters(model)
    return out


def state_group_sha256(state: Mapping[str, torch.Tensor]) -> str:
    return bp.state_group_sha256({k: v for k, v in state.items()})


def shared_hash(model: nn.Module) -> str:
    combined: dict[str, torch.Tensor] = {}
    for name in SHARED_MODULE_NAMES:
        for k, v in getattr(model, name).state_dict().items():
            combined[f"{name}:{k}"] = v
    return state_group_sha256(combined)


def encoder_hash(model: nn.Module) -> str:
    return bp.encoder_group_hash(model)


def full_hash(model: nn.Module) -> str:
    return state_group_sha256(model.state_dict())


def parameter_inventory() -> dict[str, Any]:
    torch.manual_seed(0)
    norm = {"global_mean": np.zeros(sro.GLOBAL_DIM, dtype=np.float32), "global_scale": np.ones(sro.GLOBAL_DIM, dtype=np.float32)}
    p3, r, u = build_three_arms(0, norm, "cpu").values()
    cfg = MFICConfig()
    expected = cfg.mfic_inventory()
    actual = {
        "p3": module_parameter_counts(p3),
        "mfic_r": module_parameter_counts(r),
        "mfic_u": module_parameter_counts(u),
    }
    return {
        "protocol": PROTOCOL_ID,
        "expected": expected,
        "expected_p3": src.BJOCConfig().inventory(),
        "actual": actual,
        "reader_in": {"p3": int(p3.config.reader_in), "mfic": int(r.config.reader_in)},
        "all_expected": bool(
            actual["p3"]["total"] == P3_PARAMETERS
            and actual["mfic_r"]["total"] == MFIC_PARAMETERS
            and actual["mfic_u"]["total"] == MFIC_PARAMETERS
            and actual["p3"]["reader"] == P3_READER_PARAMETERS
            and actual["mfic_r"]["reader"] == MFIC_READER_PARAMETERS
            and actual["mfic_r"]["q_projection"] == Q_PROJECTION_PARAMETERS
            and int(r.config.reader_in) == READER_IN
        ),
        "param_cap": PARAM_CAP,
    }


# ---------------------------------------------------------------------------
# CPU mathematical / numerical gates
# ---------------------------------------------------------------------------


def _synthetic_objects(
    n: int,
    edges: Sequence[tuple[int, int]],
    *,
    atom_types: Sequence[int] | None = None,
    mol_index: int = -1,
) -> sro.MolObjects:
    n = int(n)
    x = np.zeros((n, 1), dtype=np.int64) if atom_types is None else np.asarray(atom_types, dtype=np.int64).reshape(n, 1)
    rows = []
    for u, v in edges:
        rows.append((int(u), int(v), 1))
        rows.append((int(v), int(u), 1))
    edge_index = np.asarray([[u, v] for u, v, _ in rows], dtype=np.int64).T if rows else np.zeros((2, 0), dtype=np.int64)
    edge_attr = np.asarray([bt for _, _, bt in rows], dtype=np.int64)
    return sro.build_mol_objects(int(mol_index), x, edge_index, edge_attr)


def _kernel_for_graph(n: int, edges: Sequence[tuple[int, int]]) -> dict[str, Any]:
    A = adjacency_from_edges(n, edges)
    return kernel_from_adjacency(A)


def _relabel_kernel(K: np.ndarray, perm: Sequence[int]) -> np.ndarray:
    perm = np.asarray(perm, dtype=np.int64)
    return K[np.ix_(perm, perm)]


def _graph_family_checks() -> dict[str, Any]:
    cases: dict[str, Any] = {}
    families = {
        "single_atom": (1, []),
        "single_bond": (2, [(0, 1)]),
        "chain5": (5, [(0, 1), (1, 2), (2, 3), (3, 4)]),
        "ring6": (6, [(i, (i + 1) % 6) for i in range(6)]),
        "naphthalene_fused": (
            10,
            [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (4, 6), (6, 7), (7, 8), (8, 9), (9, 5)],
        ),
        "bicyclic_bridged": (
            7,
            [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (1, 6), (4, 6)],
        ),
        "chain3_plus_isolated": (4, [(0, 1), (1, 2)]),
    }
    rng = np.random.default_rng(0)
    for name, (n, edges) in families.items():
        out = _kernel_for_graph(n, edges)
        K, R, S, A = out["K"], out["R"], out["S"], out["A"]
        series = truncated_series(S, ALPHA, kmax=40)
        perm = rng.permutation(n)
        K_perm = _relabel_kernel(K, perm)
        # A' = P A P^T; K' computed independently from the relabelled A.
        A_perm = A[np.ix_(perm, perm)]
        K_from_permuted_A = kernel_from_adjacency(A_perm)["K"]
        Q = rng.normal(size=(n, Q_DIM))
        C = Q.T @ K @ Q / max(1, n)
        C_perm = Q[perm].T @ K_from_permuted_A @ Q[perm] / max(1, n)
        K_U = content_null_kernel(K)
        isolated_zero = True
        deg = A.sum(1)
        for i in np.where(deg == 0)[0]:
            isolated_zero = isolated_zero and bool(np.allclose(K[i], 0.0) and np.allclose(K[:, i], 0.0))
        cases[name] = {
            "n": int(n),
            "n_edges": int(A.sum() // 2),
            "symmetric_S": bool(np.allclose(S, S.T, atol=0.0)),
            "symmetric_R": bool(np.allclose(R, R.T, atol=1e-10)),
            "symmetric_K": bool(np.allclose(K, K.T, atol=1e-10)),
            "finite_R": bool(np.all(np.isfinite(R))),
            "finite_K": bool(np.all(np.isfinite(K))),
            "min_eig_I_minus_alpha_S": float(out["min_eig"]),
            "min_eig_positive": bool(out["min_eig"] > 0.0),
            "series_max_abs_error": float(np.abs(K - series).max()),
            "series_matches": bool(np.abs(K - series).max() < 1e-6),
            "relabel_kernel_max_abs": float(np.abs(K_from_permuted_A - K_perm).max()),
            "relabel_kernel_invariant": bool(np.abs(K_from_permuted_A - K_perm).max() < 1e-9),
            "relabel_quadratic_max_abs": float(np.abs(C - C_perm).max()),
            "relabel_quadratic_invariant": bool(np.abs(C - C_perm).max() < 1e-9),
            "isolated_vertices_zero": bool(isolated_zero),
            "K_U_finite": bool(np.all(np.isfinite(K_U))),
            "K_U_symmetric": bool(np.allclose(K_U, K_U.T, atol=0.0)),
            "K_vs_KU_max_abs": float(np.abs(K - K_U).max()),
            "K_vs_KU_differ": bool(np.abs(K - K_U).max() > 1e-8),
            "K_trace": float(np.trace(K)),
            "K_offdiag_sum": float(K.sum() - np.trace(K)),
            "K_U_trace": float(np.trace(K_U)),
            "K_U_offdiag_sum": float(K_U.sum() - np.trace(K_U)),
            "K_U_preserves_trace": bool(abs(np.trace(K_U) - np.trace(K)) <= 1e-10 * max(1.0, abs(np.trace(K)))),
            "K_U_preserves_offdiag_sum": bool(
                abs((K_U.sum() - np.trace(K_U)) - (K.sum() - np.trace(K)))
                <= 1e-10 * max(1.0, abs(K.sum() - np.trace(K)))
            ),
            "K_rank": int(np.linalg.matrix_rank(K, tol=1e-10)),
        }
    # Exact single-bond identity: K = alpha^2/(1-alpha^2) * I.
    a = ALPHA
    exact = a * a / (1.0 - a * a)
    cases["single_bond"]["exact_expected_diagonal"] = float(exact)
    cases["single_bond"]["exact_identity_max_abs"] = float(np.abs(cases["single_bond"]["K_trace"] / 2.0 - exact))
    return cases


def _witness_pair() -> dict[str, Any]:
    """Two connected non-isomorphic 3-regular graphs with identical walk counts."""
    n = 6
    bipartite = [(i, j) for i in range(3) for j in range(3, 6)]
    prism = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (0, 3), (1, 4), (2, 5)]
    kb = _kernel_for_graph(n, bipartite)
    kp = _kernel_for_graph(n, prism)
    walk_counts_b = [walk_count(kb["A"], k) for k in range(1, 9)]
    walk_counts_p = [walk_count(kp["A"], k) for k in range(1, 9)]
    rng = np.random.default_rng(1)
    Q = rng.normal(size=(n, Q_DIM))
    Cb = Q.T @ kb["K"] @ Q / n
    Cp = Q.T @ kp["K"] @ Q / n
    # A second pair: C6 versus two disjoint triangles (both 2-regular on 6 atoms).
    c6_edges = [(i, (i + 1) % 6) for i in range(6)]
    two_tri_edges = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)]
    kc6 = _kernel_for_graph(6, c6_edges)
    k2t = _kernel_for_graph(6, two_tri_edges)
    return {
        "connected_3_regular_pair": {
            "graph_1": "K_{3,3}",
            "graph_2": "triangular prism",
            "both_connected": True,
            "both_3_regular": True,
            "non_isomorphic": True,
            "walk_counts_1_to_8_graph_1": walk_counts_b,
            "walk_counts_1_to_8_graph_2": walk_counts_p,
            "walk_counts_identical": bool(np.allclose(walk_counts_b, walk_counts_p, atol=0.0)),
            "K_max_abs_difference": float(np.abs(kb["K"] - kp["K"]).max()),
            "K_distinguishes": bool(np.abs(kb["K"] - kp["K"]).max() > 1e-8),
            "quadratic_max_abs_difference": float(np.abs(Cb - Cp).max()),
            "quadratic_distinguishes": bool(np.abs(Cb - Cp).max() > 1e-8),
            "note": (
                "Both graphs are connected, 3-regular and have identical total walk counts "
                "n*3^k at EVERY length; only the distribution of walks (the full K matrix) differs."
            ),
        },
        "c6_vs_two_triangles": {
            "both_2_regular": True,
            "walk_counts_identical": bool(
                np.allclose(
                    [walk_count(kc6["A"], k) for k in range(1, 9)],
                    [walk_count(k2t["A"], k) for k in range(1, 9)],
                    atol=0.0,
                )
            ),
            "K_max_abs_difference": float(np.abs(kc6["K"] - k2t["K"]).max()),
            "K_distinguishes": bool(np.abs(kc6["K"] - k2t["K"]).max() > 1e-8),
            "note": "C6 is connected; 2C3 is disconnected. This pair is a secondary illustration only.",
        },
        "synthetic_witness_does_not_imply_zinc_gain": True,
    }


def run_math_gates(real_objects: Sequence[sro.MolObjects], *, max_real: int = 200) -> dict[str, Any]:
    """Full CPU math/numerical gate for K, R, K_U and the quadratic form."""
    require(len(real_objects) > 0, "math gates need real objects")
    fam = _graph_family_checks()
    witness = _witness_pair()

    sample = list(real_objects[: int(max_real)])
    real_checks: list[dict[str, Any]] = []
    for o in sample:
        A = physical_adjacency(o)
        out = kernel_from_adjacency(A)
        K, R, S = out["K"], out["R"], out["S"]
        series = truncated_series(S, ALPHA, kmax=40)
        K_U = content_null_kernel(K)
        rng = np.random.default_rng(int(o.mol_index) + 17)
        perm = rng.permutation(int(o.n_atoms))
        K_perm = _relabel_kernel(K, perm)
        A_perm = A[np.ix_(perm, perm)]
        K_from_perm = kernel_from_adjacency(A_perm)["K"]
        Q = rng.normal(size=(int(o.n_atoms), Q_DIM))
        C = Q.T @ K @ Q / max(1, int(o.n_atoms))
        C_perm = Q[perm].T @ K_from_perm @ Q[perm] / max(1, int(o.n_atoms))
        real_checks.append(
            {
                "row": int(o.mol_index),
                "n_atoms": int(o.n_atoms),
                "symmetric_S": bool(np.allclose(S, S.T, atol=0.0)),
                "symmetric_R": bool(np.allclose(R, R.T, atol=1e-10)),
                "symmetric_K": bool(np.allclose(K, K.T, atol=1e-10)),
                "finite_R": bool(np.all(np.isfinite(R))),
                "finite_K": bool(np.all(np.isfinite(K))),
                "min_eig_I_minus_alpha_S": float(out["min_eig"]),
                "min_eig_positive": bool(out["min_eig"] > 0.0),
                "series_max_abs_error": float(np.abs(K - series).max()),
                "series_matches": bool(np.abs(K - series).max() < 1e-6),
                "relabel_kernel_max_abs": float(np.abs(K_from_perm - K_perm).max()),
                "relabel_kernel_invariant": bool(np.abs(K_from_perm - K_perm).max() < 1e-9),
                "relabel_quadratic_max_abs": float(np.abs(C - C_perm).max()),
                "relabel_quadratic_invariant": bool(np.abs(C - C_perm).max() < 1e-9),
                "K_U_finite": bool(np.all(np.isfinite(K_U))),
                "K_U_symmetric": bool(np.allclose(K_U, K_U.T, atol=0.0)),
                "K_vs_KU_max_abs": float(np.abs(K - K_U).max()),
                "K_vs_KU_differ": bool(np.abs(K - K_U).max() > 1e-8),
                "sum_R": float(R.sum()),
                "sum_K": float(K.sum()),
                "trace_K": float(np.trace(K)),
                "K_rank": int(np.linalg.matrix_rank(K, tol=1e-10)),
                "K_fro_norm": float(np.linalg.norm(K, "fro")),
            }
        )
    sum_R = np.asarray([r["sum_R"] for r in real_checks], dtype=np.float64)
    sum_R_mean = float(sum_R.mean())
    sum_R_std = float(sum_R.std())
    sum_R_cv = float(sum_R_std / max(abs(sum_R_mean), 1e-30))
    # K is not a function of n alone: find two real objects with the same n_atoms.
    by_n: dict[int, list[sro.MolObjects]] = {}
    for o in sample:
        by_n.setdefault(int(o.n_atoms), []).append(o)
    same_n_pair = None
    for n, group in sorted(by_n.items()):
        if len(group) >= 2:
            K0 = kernel_from_adjacency(physical_adjacency(group[0]))["K"]
            K1 = kernel_from_adjacency(physical_adjacency(group[1]))["K"]
            if K0.shape == K1.shape:
                same_n_pair = {
                    "n_atoms": int(n),
                    "row_1": int(group[0].mol_index),
                    "row_2": int(group[1].mol_index),
                    "K_frobenius_difference": float(np.linalg.norm(K0 - K1, "fro")),
                    "K_differs": bool(np.linalg.norm(K0 - K1, "fro") > 1e-8),
                }
                break
    same_n_pair_source = "real_objects"
    if same_n_pair is None:
        # Fallback: the C6 / 2C3 pair has the same atom count and different K.
        c6 = _synthetic_objects(6, [(i, (i + 1) % 6) for i in range(6)], mol_index=990_001)
        two_tri = _synthetic_objects(
            6, [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)], mol_index=990_002
        )
        K0 = kernel_from_adjacency(physical_adjacency(c6))["K"]
        K1 = kernel_from_adjacency(physical_adjacency(two_tri))["K"]
        same_n_pair = {
            "n_atoms": 6,
            "row_1": int(c6.mol_index),
            "row_2": int(two_tri.mol_index),
            "K_frobenius_difference": float(np.linalg.norm(K0 - K1, "fro")),
            "K_differs": bool(np.linalg.norm(K0 - K1, "fro") > 1e-8),
        }
        same_n_pair_source = "synthetic_c6_vs_2c3_fallback"

    checks = {
        "all_family_checks_pass": bool(
            all(
                v["symmetric_S"]
                and v["symmetric_R"]
                and v["symmetric_K"]
                and v["finite_R"]
                and v["finite_K"]
                and v["min_eig_positive"]
                and v["series_matches"]
                and v["relabel_kernel_invariant"]
                and v["relabel_quadratic_invariant"]
                and v["isolated_vertices_zero"]
                and v["K_U_finite"]
                and v["K_U_symmetric"]
                and v["K_U_preserves_trace"]
                and v["K_U_preserves_offdiag_sum"]
                for v in fam.values()
            )
        ),
        "all_real_checks_pass": bool(
            all(
                v["symmetric_S"]
                and v["symmetric_R"]
                and v["symmetric_K"]
                and v["finite_R"]
                and v["finite_K"]
                and v["min_eig_positive"]
                and v["series_matches"]
                and v["relabel_kernel_invariant"]
                and v["relabel_quadratic_invariant"]
                and v["K_U_finite"]
                and v["K_U_symmetric"]
                for v in real_checks
            )
        ),
        "K_not_a_function_of_n_atoms_alone": bool(same_n_pair["K_differs"]),
        "K_U_differs_on_non_degenerate_graphs": bool(
            all(v["K_vs_KU_differ"] for name, v in fam.items() if name not in ("single_atom", "single_bond"))
        ),
        "single_atom_K_zero": bool(np.allclose(fam["single_atom"]["K_trace"], 0.0) and fam["single_atom"]["K_offdiag_sum"] == 0.0),
        "witness_connected_3_regular": bool(witness["connected_3_regular_pair"]["walk_counts_identical"] and witness["connected_3_regular_pair"]["K_distinguishes"] and witness["connected_3_regular_pair"]["quadratic_distinguishes"]),
        "sum_R_not_constant": bool(sum_R_cv > 1e-6),
        "sum_R_used_as_feature": False,
    }
    all_passed = bool(all(checks[k] for k in checks if k != "sum_R_used_as_feature"))
    return {
        "protocol": PROTOCOL_ID,
        "alpha": ALPHA,
        "q_dim": Q_DIM,
        "n_real_molecules": len(real_checks),
        "families": fam,
        "real_checks": real_checks,
        "same_n_atoms_pair": {**same_n_pair, "source": same_n_pair_source},
        "sum_R": {
            "n": int(sum_R.size),
            "mean": sum_R_mean,
            "std": sum_R_std,
            "cv": sum_R_cv,
            "min": float(sum_R.min()),
            "max": float(sum_R.max()),
            "n_distinct_rounded_8": int(np.unique(np.round(sum_R, 8)).size),
            "not_constant": bool(sum_R_cv > 1e-6),
            "used_as_feature": False,
        },
        "witness": witness,
        "checks": checks,
        "all_passed": all_passed,
        "labels_read": False,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }


# ---------------------------------------------------------------------------
# CPU Ridge purchase probe (Fit labels only; official train structures only)
# ---------------------------------------------------------------------------


def _feature_names() -> dict[str, list[str]]:
    base = [f"atom_type_{i}" for i in range(sro.ATOM_CATEGORIES)]
    base += [f"bond_type_{i}" for i in range(sro.BOND_CATEGORIES)]
    base += list(sro.GLOBAL_FEAT_NAMES)
    base += ["raw_n_atoms", "raw_n_bonds", "raw_n_junctions"]
    matrix = [f"C_{i}_{j}" for i, j in zip(_UPPER_I.tolist(), _UPPER_J.tolist())]
    return {"base": base, "matrix": matrix, "matrix_total": base + matrix}


def cpu_ridge_signal(
    *,
    rows: Sequence[int],
    objects: Sequence[sro.MolObjects],
    y: np.ndarray,
    identity_of_row: Mapping[int | str, str],
    kernels: Sequence[np.ndarray] | None = None,
    folds: Sequence[Sequence[int]] | None = None,
    n_folds: int = CPU_FOLDS,
    alpha: float = CPU_RIDGE_ALPHA,
    n_components: int = CPU_PCA_COMPONENTS,
) -> dict[str, Any]:
    """Four-fold grouped Ridge probe: CPU-Base vs CPU-Matrix.

    Discipline:
    * only ``Fit`` rows and their raw ``y`` are used;
    * PCA is fitted on the training fold's atoms only;
    * validation-fold atoms are only transformed;
    * both arms use the same train-fold-only standardisation rule and the same
      fixed Ridge ``alpha=1``; no hyper-parameter search;
    * no model prediction, official valid/test or Monitor/Locked label enters.
    """
    rows = [int(r) for r in rows]
    require(len(rows) == len(objects) == len(y), "rows/objects/y length mismatch")
    if folds is None:
        folds = grouped_folds(rows, identity_of_row, n_folds=n_folds, salt=SPLIT_SALT)
    folds = [sorted(int(r) for r in f) for f in folds]
    require(len(folds) == int(n_folds), "fold count mismatch")
    require(sorted(r for f in folds for r in f) == sorted(rows), "folds do not cover Fit rows")

    if kernels is None:
        kernels = [kernel_from_adjacency(physical_adjacency(o))["K"] for o in objects]
    require(len(kernels) == len(objects), "kernel count mismatch")

    base = np.stack([base_features(o) for o in objects], 0)
    atom_feats = [np.asarray(o.atom_feat, dtype=np.float64) for o in objects]
    offsets = np.cumsum([0] + [len(a) for a in atom_feats]).astype(np.int64)
    X_atoms = np.concatenate(atom_feats, 0)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    row_to_idx = {r: i for i, r in enumerate(rows)}
    all_idx = np.arange(len(rows), dtype=np.int64)

    records: list[dict[str, Any]] = []
    arrays: dict[str, list[np.ndarray]] = {
        "fold": [],
        "val_rows": [],
        "y_val": [],
        "pred_base": [],
        "pred_matrix": [],
        "err_base": [],
        "err_matrix": [],
    }
    for fold_id, val_rows in enumerate(folds):
        val_idx = np.asarray([row_to_idx[r] for r in val_rows], dtype=np.int64)
        train_idx = np.asarray(sorted(set(all_idx.tolist()) - set(val_idx.tolist())), dtype=np.int64)
        require(train_idx.size + val_idx.size == len(rows), "fold split size mismatch")
        # PCA is fit on TRAIN-fold atoms only; every validation atom is transform-only.
        atom_train_idx = np.concatenate([np.arange(offsets[i], offsets[i + 1], dtype=np.int64) for i in train_idx])
        pca = pca_fit(X_atoms[atom_train_idx], n_components=int(n_components))
        Q_all = pca_transform(X_atoms, pca)
        C_all = np.stack(
            [matrix_features_from_projection(Q_all[offsets[i] : offsets[i + 1]], kernels[i]) for i in range(len(objects))],
            0,
        )
        X_matrix = np.concatenate([base, C_all], 1)
        require(X_matrix.shape[1] == MATRIX_DIM, f"matrix feature width {X_matrix.shape[1]} != {MATRIX_DIM}")

        fit_base = ridge_fit_predict(base[train_idx], y[train_idx], base[val_idx], alpha=alpha)
        fit_matrix = ridge_fit_predict(X_matrix[train_idx], y[train_idx], X_matrix[val_idx], alpha=alpha)
        pred_base = np.asarray(fit_base["pred"], dtype=np.float64)
        pred_matrix = np.asarray(fit_matrix["pred"], dtype=np.float64)
        y_val = y[val_idx]
        err_base = np.abs(pred_base - y_val)
        err_matrix = np.abs(pred_matrix - y_val)
        delta = err_matrix - err_base
        mae_base = float(err_base.mean())
        mae_matrix = float(err_matrix.mean())
        delta_mean = float(delta.mean())
        # Frozen diagnostic: remove the 10 validation molecules with the largest BASE error.
        k_trim = int(min(CPU_TRIM_K, val_idx.size))
        worst = np.argsort(-err_base, kind="stable")[:k_trim]
        keep = np.ones(val_idx.size, dtype=bool)
        keep[worst] = False
        delta_trimmed = float(delta[keep].mean()) if keep.any() else float("nan")
        ordinary = y_val > TAIL_THRESHOLD
        delta_ordinary = float(delta[ordinary].mean()) if ordinary.any() else float("nan")
        tail = ~ordinary
        delta_tail = float(delta[tail].mean()) if tail.any() else float("nan")

        rec = {
            "fold": int(fold_id),
            "n_train": int(train_idx.size),
            "n_val": int(val_idx.size),
            "train_rows_sha256": hashlib.sha256(np.asarray([rows[i] for i in train_idx], dtype=np.int64).tobytes()).hexdigest(),
            "val_rows_sha256": hashlib.sha256(np.asarray(val_rows, dtype=np.int64).tobytes()).hexdigest(),
            "pca": {
                "n_train_atoms": int(atom_train_idx.size),
                "n_components": int(pca["n_components"]),
                "explained_variance_ratio": [float(v) for v in pca["explained_variance_ratio"]],
                "explained_variance_ratio_sum": float(pca["explained_variance_ratio_sum"]),
            },
            "mae_base": mae_base,
            "mae_matrix": mae_matrix,
            "delta_matrix_minus_base": delta_mean,
            "rmse_base": float(np.sqrt(np.mean((pred_base - y_val) ** 2))),
            "rmse_matrix": float(np.sqrt(np.mean((pred_matrix - y_val) ** 2))),
            "better_fraction_matrix": float(np.mean(err_matrix < err_base)),
            "trim_k": k_trim,
            "trimmed_delta": delta_trimmed,
            "ordinary_n": int(ordinary.sum()),
            "ordinary_delta": delta_ordinary,
            "tail_n": int(tail.sum()),
            "tail_delta": delta_tail,
            "base_zero_variance_features": int(fit_base["n_zero_variance_features"]),
            "matrix_zero_variance_features": int(fit_matrix["n_zero_variance_features"]),
            "matrix_C_nonzero_columns": int(np.sum(np.std(C_all[train_idx], axis=0) > 1e-12)),
            "base_feature_width": int(base.shape[1]),
            "matrix_feature_width": int(X_matrix.shape[1]),
        }
        records.append(rec)
        arrays["fold"].append(np.full(val_idx.size, fold_id, dtype=np.int64))
        arrays["val_rows"].append(np.asarray(val_rows, dtype=np.int64))
        arrays["y_val"].append(y_val)
        arrays["pred_base"].append(pred_base)
        arrays["pred_matrix"].append(pred_matrix)
        arrays["err_base"].append(err_base)
        arrays["err_matrix"].append(err_matrix)

    fold_deltas = [float(r["delta_matrix_minus_base"]) for r in records]
    fold_trimmed = [float(r["trimmed_delta"]) for r in records]
    fold_ordinary = [float(r["ordinary_delta"]) for r in records]
    conditions = {
        "C1_mean_mae_improvement_at_least_0.020": bool(float(np.mean(fold_deltas)) <= -CPU_MIN_MEAN_IMPROVEMENT),
        "C2_at_least_3_of_4_folds_negative": bool(sum(d < 0.0 for d in fold_deltas) >= CPU_MIN_FOLDS_IMPROVED),
        "C3_trimmed10_mean_direction_negative": bool(float(np.mean(fold_trimmed)) < 0.0),
        "C4_ordinary_mean_direction_negative": bool(float(np.mean(fold_ordinary)) < 0.0),
    }
    summary = {
        "n_fit_rows": int(len(rows)),
        "n_folds": int(n_folds),
        "fold_salt": SPLIT_SALT,
        "fold_rule": "identity-grouped, sha256(salt|fold|identity), greedy smallest fold",
        "n_identity_groups": int(len({str(identity_of_row.get(r, identity_of_row.get(str(r)))) for r in rows})),
        "pca_components": int(n_components),
        "ridge_alpha": float(alpha),
        "base_dim": int(base.shape[1]),
        "matrix_dim": int(MATRIX_DIM),
        "mean_mae_base": float(np.mean([r["mae_base"] for r in records])),
        "mean_mae_matrix": float(np.mean([r["mae_matrix"] for r in records])),
        "mean_delta_matrix_minus_base": float(np.mean(fold_deltas)),
        "mean_mae_improvement": float(np.mean([r["mae_base"] - r["mae_matrix"] for r in records])),
        "fold_deltas": fold_deltas,
        "folds_negative": int(sum(d < 0.0 for d in fold_deltas)),
        "fold_directions": ["improved" if d < 0.0 else "worse" for d in fold_deltas],
        "mean_trimmed10_delta": float(np.mean(fold_trimmed)),
        "trimmed10_deltas": fold_trimmed,
        "mean_ordinary_delta": float(np.mean(fold_ordinary)),
        "ordinary_deltas": fold_ordinary,
        "mean_tail_delta": float(np.mean([r["tail_delta"] for r in records])),
        "mean_better_fraction": float(np.mean([r["better_fraction_matrix"] for r in records])),
        "mean_pca_explained_ratio_sum": float(np.mean([r["pca"]["explained_variance_ratio_sum"] for r in records])),
        "feature_names": _feature_names(),
        "conditions": conditions,
        "labels_used": "Fit rows raw y only",
        "monitor_labels_read": False,
        "locked_labels_read": False,
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "model_predictions_used": False,
    }
    return {"summary": summary, "folds": records, "arrays": arrays}


# ---------------------------------------------------------------------------
# model-level CPU audit: params, gradients, consumption, invariances
# ---------------------------------------------------------------------------


def _tensor_hash(t: torch.Tensor) -> str:
    return hashlib.sha256(np.ascontiguousarray(t.detach().cpu().numpy(), dtype=np.float32).tobytes()).hexdigest()


def _permute_object(obj: sro.MolObjects, perm: np.ndarray) -> sro.MolObjects:
    """Rebuild one object after relabelling atom indices ``old -> perm[old]``."""
    n = int(obj.n_atoms)
    perm = np.asarray(perm, dtype=np.int64)
    require(sorted(perm.tolist()) == list(range(n)), "invalid permutation")
    atom_types = np.argmax(np.asarray(obj.atom_feat, dtype=np.float64)[:, : sro.ATOM_CATEGORIES], axis=1).astype(np.int64)
    x = atom_types[perm]  # new index i gets the atom that was at perm[i]
    # Equivalent: new_pos[old] = perm[old]; rebuild edges using new positions.
    new_pos = np.empty(n, dtype=np.int64)
    new_pos[perm] = np.arange(n, dtype=np.int64)
    edges = []
    for (u, v), bt in zip(obj.bond_index, obj.bond_type):
        a, b = int(new_pos[int(u)]), int(new_pos[int(v)])
        edges.append((a, b, int(bt)))
        edges.append((b, a, int(bt)))
    edge_index = np.asarray([[u, v] for u, v, _ in edges], dtype=np.int64).T if edges else np.zeros((2, 0), dtype=np.int64)
    edge_attr = np.asarray([bt for _, _, bt in edges], dtype=np.int64)
    return sro.build_mol_objects(int(obj.mol_index), x.reshape(n, 1), edge_index, edge_attr)


def _swap_edge_endpoints(obj: sro.MolObjects) -> sro.MolObjects:
    n = int(obj.n_atoms)
    atom_types = np.argmax(np.asarray(obj.atom_feat, dtype=np.float64)[:, : sro.ATOM_CATEGORIES], axis=1).astype(np.int64)
    edges = []
    for (u, v), bt in zip(obj.bond_index, obj.bond_type):
        edges.append((int(v), int(u), int(bt)))
        edges.append((int(u), int(v), int(bt)))
    edge_index = np.asarray([[u, v] for u, v, _ in edges], dtype=np.int64).T
    edge_attr = np.asarray([bt for _, _, bt in edges], dtype=np.int64)
    return sro.build_mol_objects(int(obj.mol_index), atom_types.reshape(n, 1), edge_index, edge_attr)


def run_model_audit(real_objects: Sequence[sro.MolObjects], *, seed: int = 0, n_objects: int = 32) -> dict[str, Any]:
    """CPU-only parameter / gradient / consumption / invariance audit."""
    require(len(real_objects) >= 2, "model audit needs at least two real objects")
    norm = {"global_mean": np.zeros(sro.GLOBAL_DIM, dtype=np.float32), "global_scale": np.ones(sro.GLOBAL_DIM, dtype=np.float32)}
    arms = build_three_arms(int(seed), norm, "cpu")
    p3, mr, mu = arms["P3"], arms["MFIC-R"], arms["MFIC-U"]
    inv = parameter_inventory()
    shared_equal = shared_hash(p3) == shared_hash(mr) == shared_hash(mu)
    encoder_equal = encoder_hash(p3) == encoder_hash(mr) == encoder_hash(mu)
    ru_full_equal = full_hash(mr) == full_hash(mu)

    objs = list(real_objects[: int(n_objects)])
    K_R = [kernel_from_adjacency(physical_adjacency(o))["K"] for o in objs]
    K_U = [content_null_kernel(K) for K in K_R]
    batch_R = batch_objects_with_k(objs, K_R)
    batch_U = batch_objects_with_k(objs, K_U)
    batch_0 = batch_objects_with_k(objs, [np.zeros_like(K) for K in K_R])

    mr.eval()
    mu.eval()
    with torch.no_grad():
        feat_R = mr.forward_features(batch_R)
        feat_RU = mr.forward_features(batch_U)
        feat_U = mu.forward_features(batch_U)
        feat_UR = mu.forward_features(batch_R)
        pred_R = mr(batch_R).view(-1)
        pred_RU = mr(batch_U).view(-1)
        pred_U = mu(batch_U).view(-1)
        pred_UR = mu(batch_R).view(-1)
        pred_0 = mr(batch_0).view(-1)
    hidden_names = ("a", "b", "w", "c", "z", "a_sum", "q")
    hidden_hashes = {
        name: {
            "K_R": _tensor_hash(feat_R[name]),
            "K_U": _tensor_hash(feat_RU[name]),
            "identical": bool(torch.equal(feat_R[name], feat_RU[name])),
        }
        for name in hidden_names
    }
    C_diff = float((feat_R["C"] - feat_RU["C"]).abs().max())
    pred_diff = float((pred_R - pred_RU).abs().max())
    pred_diff_mean = float((pred_R - pred_RU).abs().mean())
    pred_vs_zero_mean = float((pred_R - pred_0).abs().mean())
    pred_vs_zero_max = float((pred_R - pred_0).abs().max())
    # Explicit slices: [z 0:96, a_sum 96:192, C 192:328, global 328:361].
    # Explicit slices: [z 0:96, a_sum 96:192, C 192:328, global 328:361].
    reader_prefix_equal = bool(torch.equal(feat_R["reader_in"][:, :192], feat_RU["reader_in"][:, :192]))
    reader_global_equal = bool(torch.equal(feat_R["reader_in"][:, 328:], feat_RU["reader_in"][:, 328:]))
    reader_C_diff = float((feat_R["reader_in"][:, 192:328] - feat_RU["reader_in"][:, 192:328]).abs().max())

    # Gradient audit: two batches, ensure every parameter receives a finite non-zero gradient.
    for p in mr.parameters():
        p.grad = None
    losses = []
    for b in (batch_R, batch_U):
        mr.train()
        pred = mr(b).view(-1)
        loss = ((pred - 0.1) ** 2).mean()
        loss.backward()
        losses.append(float(loss.detach()))
    grad_stats = {}
    zero_params = []
    for name, p in mr.named_parameters():
        g = p.grad
        val = float(g.detach().abs().sum()) if g is not None else 0.0
        grad_stats[name] = val
        if g is None or not math.isfinite(val) or val == 0.0:
            zero_params.append(name)
    # Reader first-layer input-channel coverage (361 columns).
    w0 = mr.reader.lins[0].weight
    channel_grad = w0.grad.detach().abs().sum(0).cpu().numpy() if w0.grad is not None else np.zeros(w0.shape[1])
    c_channels = channel_grad[192:328]
    dead_c_channels = np.where(c_channels <= 0.0)[0].tolist()
    q_grad = {k: float(v.grad.detach().abs().sum()) for k, v in mr.q_proj.named_parameters() if v.grad is not None}

    # Invariance: relabelling, endpoint swap, batch order.
    synth = _synthetic_objects(
        6,
        [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)],
        atom_types=[0, 1, 2, 3, 4, 5],
        mol_index=999_001,
    )
    K_s = kernel_from_adjacency(physical_adjacency(synth))["K"]
    perm = np.asarray([3, 0, 5, 1, 4, 2], dtype=np.int64)
    synth_p = _permute_object(synth, perm)
    K_sp = kernel_from_adjacency(physical_adjacency(synth_p))["K"]
    K_expected = K_s[np.ix_(perm, perm)]
    b_s = batch_objects_with_k([synth], [K_s])
    b_sp = batch_objects_with_k([synth_p], [K_sp])
    with torch.no_grad():
        pred_s = mr(b_s).view(-1)
        pred_sp = mr(b_sp).view(-1)
    synth_swap = _swap_edge_endpoints(synth)
    K_sw = kernel_from_adjacency(physical_adjacency(synth_swap))["K"]
    b_sw = batch_objects_with_k([synth_swap], [K_sw])
    with torch.no_grad():
        pred_sw = mr(b_sw).view(-1)
    # Batch permutation.
    b_two = batch_objects_with_k([objs[0], objs[1]], [K_R[0], K_R[1]])
    b_two_rev = batch_objects_with_k([objs[1], objs[0]], [K_R[1], K_R[0]])
    with torch.no_grad():
        pred_two = mr(b_two).view(-1)
        pred_two_rev = mr(b_two_rev).view(-1)

    # Static forbidden-module scan (class names + exact call tokens in this file).
    tokens = ("MessagePassing", "GCNConv", "GINConv", "GATConv", "SAGEConv", "Transformer", "MultiheadAttention", "LSTM", "GRU", "RNN")
    module_hits = [f"{name}:{type(mod).__name__}" for name, mod in mr.named_modules() if any(tok in type(mod).__name__ for tok in tokens)]
    src_text = Path(__file__).read_text(encoding="utf-8")
    call_tokens = ("nn." + "MultiheadAttention", "Message" + "Passing(", "Transformer" + "Encoder(", "nn." + "Transformer", "GCN" + "Conv(", "GIN" + "Conv(", "GAT" + "Conv(", "SAGE" + "Conv(")
    source_hits = [tok for tok in call_tokens if tok in src_text]
    state_keys = list(mr.state_dict().keys())
    kernel_in_state = any("k_pad" in k.lower() or "kernel" in k.lower() for k in state_keys)
    checks = {
        "p3_parameters_exact": bool(model_parameters(p3) == P3_PARAMETERS),
        "mfic_parameters_exact": bool(model_parameters(mr) == MFIC_PARAMETERS and model_parameters(mu) == MFIC_PARAMETERS),
        "reader_parameters_exact": bool(
            module_parameter_counts(mr)["reader"] == MFIC_READER_PARAMETERS and module_parameter_counts(mr)["q_projection"] == Q_PROJECTION_PARAMETERS
        ),
        "shared_modules_bitwise_equal": bool(shared_equal and encoder_equal),
        "mfic_R_and_U_full_init_equal": bool(ru_full_equal),
        "reader_in_is_361": bool(int(mr.config.reader_in) == READER_IN and feat_R["reader_in"].shape[1] == READER_IN),
        "C_is_16x16": bool(feat_R["C"].shape[1:] == (Q_DIM, Q_DIM)),
        "K_only_changes_C_and_reader_C_block": bool(
            all(v["identical"] for v in hidden_hashes.values())
            and C_diff > 1e-8
            and reader_prefix_equal
            and reader_global_equal
            and reader_C_diff > 1e-8
        ),
        # CPU random-init audit: the effect must be numerically non-zero.  The
        # trained-model participation thresholds are checked in Stage B.
        "real_K_vs_KU_prediction_effect": bool(pred_diff_mean > 1e-8 and pred_diff > 1e-8),
        "zero_K_changes_prediction": bool(pred_vs_zero_mean > 1e-6 and pred_vs_zero_max > 1e-6),
        "q_projection_gradient_nonzero": bool(all(v > 0.0 for v in q_grad.values())),
        "all_parameters_have_nonzero_gradient": bool(len(zero_params) == 0),
        "reader_all_C_channels_have_gradient": bool(len(dead_c_channels) == 0),
        "relabel_kernel_invariant": bool(np.abs(K_sp - K_expected).max() < 1e-9),
        "relabel_prediction_invariant": bool(torch.allclose(pred_s, pred_sp, atol=1e-6, rtol=0.0)),
        "endpoint_swap_prediction_invariant": bool(torch.allclose(pred_s, pred_sw, atol=1e-6, rtol=0.0)),
        "batch_permutation_equivariant": bool(torch.allclose(pred_two, pred_two_rev.flip(0), atol=1e-6, rtol=0.0)),
        "K_not_in_model_state_dict": bool(not kernel_in_state),
        "no_forbidden_modules": bool(not module_hits and not source_hits),
    }
    all_passed = bool(all(checks.values()))
    return {
        "protocol": PROTOCOL_ID,
        "seed": int(seed),
        "n_objects": len(objs),
        "parameter_inventory": inv,
        "parameter_counts": {"P3": module_parameter_counts(p3), "MFIC-R": module_parameter_counts(mr), "MFIC-U": module_parameter_counts(mu)},
        "hashes": {
            "shared_p3": shared_hash(p3),
            "shared_mfic_r": shared_hash(mr),
            "shared_mfic_u": shared_hash(mu),
            "encoder_p3": encoder_hash(p3),
            "encoder_mfic_r": encoder_hash(mr),
            "encoder_mfic_u": encoder_hash(mu),
            "full_mfic_r": full_hash(mr),
            "full_mfic_u": full_hash(mu),
        },
        "hidden_state_hashes": hidden_hashes,
        "C_max_abs_diff_R_vs_U": C_diff,
        "prediction": {
            "mean_abs_diff_R_vs_RU": pred_diff_mean,
            "max_abs_diff_R_vs_RU": pred_diff,
            "mean_abs_diff_R_vs_zeroK": pred_vs_zero_mean,
            "max_abs_diff_R_vs_zeroK": pred_vs_zero_max,
            "mean_abs_diff_R_vs_U_model": float((pred_R - pred_U).abs().mean()),
            "mean_abs_diff_U_vs_UR": float((pred_U - pred_UR).abs().mean()),
            "reader_prefix_equal": reader_prefix_equal,
            "reader_global_equal": reader_global_equal,
            "reader_C_block_max_abs_diff": reader_C_diff,
            "trained_participation_thresholds": {
                "mean_abs_diff_min": PARTICIPATION_MEAN,
                "max_abs_diff_min": PARTICIPATION_MAX,
                "met_at_random_init": bool(pred_diff_mean >= PARTICIPATION_MEAN and pred_diff >= PARTICIPATION_MAX),
                "note": "Stage-B trained-model check; random-init CPU audit only requires non-zero consumption.",
            },
        },
        "gradients": {
            "losses": losses,
            "module_grad_abs_sum": grad_stats,
            "zero_grad_parameters": zero_params,
            "q_projection": q_grad,
            "reader_first_layer_C_channel_grad_abs_sum_min": float(c_channels.min()) if c_channels.size else 0.0,
            "reader_first_layer_C_dead_channels": dead_c_channels,
        },
        "invariance": {
            "relabel_kernel_max_abs": float(np.abs(K_sp - K_expected).max()),
            "relabel_prediction_max_abs": float((pred_s - pred_sp).abs().max()),
            "endpoint_swap_prediction_max_abs": float((pred_s - pred_sw).abs().max()),
            "batch_permutation_max_abs": float((pred_two - pred_two_rev.flip(0)).abs().max()),
        },
        "static_scan": {"module_hits": module_hits, "source_hits": source_hits, "kernel_in_state_dict": kernel_in_state},
        "checks": checks,
        "all_passed": all_passed,
        "labels_read": False,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
