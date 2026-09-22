#!/usr/bin/env python
"""TCCD-v6 POST gain decomposition: normalization vs marginal moments vs nonlinearity.

Frozen representation diagnostic.  This round does NOT change the graph
architecture, the frozen TCCD-v2 prototype vocabulary, the TCCD-v5 pair cache,
or the optimizer protocol.  It decomposes the TCCD-v5 POST mean pair
descriptor

    pbar_G = [A_G, B_G, Cprod_G, D_G]  in R^197

into a RECON block (coordinates proven to be exactly reconstructible from the
frozen ``h_base``) and a NOVEL block (coordinates not reconstructible), then
trains three matched frozen-screen readers:

* RECON-NL      : recon coordinates -> Linear(197,64) -> ReLU -> Linear(64,16)
* RECON-LINFACT : identical parameters, no ReLU in between
* NOVEL-NL      : novel coordinates -> Linear(197,64) -> ReLU -> Linear(64,16)

FULL-NL is the exact TCCD-v5 POST arm and is reused, with provenance verified by
re-evaluating the frozen TCCD-v5 soup/best checkpoints through this module's
reader.  Official ZINC test is never loaded.  Formal execution is GPU1-only.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from tracks.ksvd.code import tccd_v0 as T
from tracks.ksvd.code import tccd_v5 as V5
from tracks.ksvd.code.run_tccd_v0 import internal_split

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = REPO_ROOT / "tracks/ksvd/results/tccd_v6"
CACHE_DIR = V5.CACHE_DIR
PROTOCOL_VERSION = "tccd_v6_post_gain_decomposition_v1"

D_LOCAL = V5.D_LOCAL
K_PROTO = V5.K_PROTO
N_REL = V5.N_REL
PHI_HIDDEN = V5.PHI_HIDDEN
PHI_OUT = V5.PHI_OUT

PAIR_WIDTH = 3 * K_PROTO + N_REL  # 197
FROZEN_BASE_DIM = V5.frozen_base_dim()  # 10464

# ---------------------------------------------------------------------------
# descriptor block layout (explicit block names; never the assignment matrix C)
# ---------------------------------------------------------------------------
# p_ij = [c_i + c_j, |c_i - c_j|, c_i * c_j, r_ij]
BLOCK_A = slice(0, K_PROTO)          # normalized prototype mean
BLOCK_B = slice(K_PROTO, 2 * K_PROTO)  # absolute prototype dispersion
BLOCK_C = slice(2 * K_PROTO, 3 * K_PROTO)  # prototype product moment (prod_moment)
BLOCK_D = slice(3 * K_PROTO, 3 * K_PROTO + N_REL)  # relation marginals
BLOCK_NAMES = ("A", "B", "C_prod", "D")
BLOCK_DIMS = {
    "A": K_PROTO,
    "B": K_PROTO,
    "C_prod": K_PROTO,
    "D": N_REL,
}
RELATION_NAMES = ("R_int", "R_b0", "R_b1", "R_b2", "R_geo")

# Frozen relation diagonal conventions from ``tccd_v0.relation_matrices``:
# R_int has an explicitly zeroed diagonal, the three native bond operators are
# symmetric with zero diagonal, R_geo[i, i] = exp(0 / TAU) = 1.
RELATION_DIAGONAL_PER_OCCURRENCE = (0.0, 0.0, 0.0, 0.0, 1.0)

# ---------------------------------------------------------------------------
# frozen references (TCCD-v5 Stage-A seed 0, remote A100 GPU1)
# ---------------------------------------------------------------------------
BASE_BEST_MAE = 0.28791576623916626
BASE_SOUP_MAE = 0.2783639132976532
FULL_BEST_MAE = 0.2608269155025482
FULL_SOUP_MAE = 0.2522333264350891
G_FULL = BASE_SOUP_MAE - FULL_SOUP_MAE  # 0.026130587...
FULL_INIT_CHECKSUM = "9fbedb92ced44449ade422b152684e3706f34fb943c32b5bfd64979290e8487f"

V5_STAGE_A_JSON = V5.RESULTS_DIR / "stageA_seed0.json"
V5_POST_SOUP = V5.RESULTS_DIR / "stageA_post_seed0_soup.pt"
V5_POST_BEST = V5.RESULTS_DIR / "stageA_post_seed0_best.pt"
V2_BEST_CHECKPOINT = V5.V2_BEST_CHECKPOINT
V2_BEST_CHECKPOINT_SHA256 = V5.V2_BEST_CHECKPOINT_SHA256

REFERENCE_TOL = 1e-6
AUDIT_TOL = 1e-6

# ---------------------------------------------------------------------------
# preregistered decision thresholds
# ---------------------------------------------------------------------------
RECON_SUFFICIENT_GAIN = 0.015
RECON_SUFFICIENT_SLACK = 0.005
NOVEL_SUFFICIENT_GAIN = 0.015
NOVEL_SUFFICIENT_SLACK = 0.005
JOINT_MIN_GAIN = 0.005
JOINT_SLACK = 0.010
NONLIN_GAP_R1 = 0.005
NONLIN_GAP_R2 = 0.010
NONLIN_GAP_SEED1_MEAN = 0.0075

ARMS = ("full_nl", "recon_nl", "recon_linfact", "novel_nl")
ARM_NONLINEAR = {
    "full_nl": True,
    "recon_nl": True,
    "recon_linfact": False,
    "novel_nl": True,
}
ARM_INPUT_BLOCK = {
    "full_nl": "full",
    "recon_nl": "recon",
    "recon_linfact": "recon",
    "novel_nl": "novel",
}


def _torch():
    return T._torch()


# ===========================================================================
# small helpers
# ===========================================================================
def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def descriptor_block_slices() -> dict[str, tuple[int, int]]:
    return {
        "A": (BLOCK_A.start, BLOCK_A.stop),
        "B": (BLOCK_B.start, BLOCK_B.stop),
        "C_prod": (BLOCK_C.start, BLOCK_C.stop),
        "D": (BLOCK_D.start, BLOCK_D.stop),
    }


def branch_parameter_count(d_p: int = PAIR_WIDTH) -> int:
    return int(d_p * PHI_HIDDEN + PHI_HIDDEN + PHI_HIDDEN * PHI_OUT + PHI_OUT)


def head_parameter_count() -> int:
    return int((FROZEN_BASE_DIM + PHI_OUT) * 1 + 1)


def parameter_accounting() -> dict[str, int]:
    branch = branch_parameter_count()
    head = head_parameter_count()
    return {"branch": branch, "head": head, "total": branch + head}


# ===========================================================================
# masks (frozen before any target-dependent training)
# ===========================================================================
def recon_mask_from_relations(d_recon: Sequence[bool]) -> np.ndarray:
    """RECON coordinates: all of Block A plus exactly the proven D relations."""
    mask = np.zeros(PAIR_WIDTH, dtype=np.float32)
    mask[BLOCK_A] = 1.0
    for r, ok in enumerate(d_recon):
        if bool(ok):
            mask[BLOCK_D.start + r] = 1.0
    return mask


def novel_mask_from_relations(d_recon: Sequence[bool]) -> np.ndarray:
    """NOVEL coordinates: B, C_prod, and any D relation not proven reconstructible."""
    recon = recon_mask_from_relations(d_recon)
    novel = np.ones(PAIR_WIDTH, dtype=np.float32) - recon
    return novel


D_RECON_EXPECTED = (True, True, True, True, True)
RECON_MASK = recon_mask_from_relations(D_RECON_EXPECTED)
NOVEL_MASK = novel_mask_from_relations(D_RECON_EXPECTED)

ARM_MASKS = {
    "full_nl": np.ones(PAIR_WIDTH, dtype=np.float32),
    "recon_nl": RECON_MASK,
    "recon_linfact": RECON_MASK,
    "novel_nl": NOVEL_MASK,
}


# ===========================================================================
# label-free pair cache (y is never read by the audit)
# ===========================================================================
@dataclass
class LabelFreeCache:
    base: np.ndarray
    C: np.ndarray
    c_offsets: np.ndarray
    pair_i0: np.ndarray
    pair_i1: np.ndarray
    pair_rel: np.ndarray
    pair_offsets: np.ndarray
    graph_index: np.ndarray
    meta: dict[str, Any]
    row_of: dict[int, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.row_of = {int(g): int(r) for r, g in enumerate(self.graph_index)}

    @property
    def n_graphs(self) -> int:
        return int(len(self.pair_offsets) - 1)

    def rows_for(self, indices: Sequence[int]) -> np.ndarray:
        return np.asarray([self.row_of[int(g)] for g in indices], dtype=np.int64)

    def n_of_row(self, row: int) -> int:
        return int(self.c_offsets[row + 1] - self.c_offsets[row])

    def pair_count_of_row(self, row: int) -> int:
        return int(self.pair_offsets[row + 1] - self.pair_offsets[row])

    def pair_count(self, row: int) -> int:
        """Alias used by the frozen v5 ``make_pair_batch`` reference path."""
        return self.pair_count_of_row(row)


def load_label_free_cache(cache_dir: Path = CACHE_DIR, split: str = "all") -> LabelFreeCache:
    """Load the exact TCCD-v5 pair cache without touching ``pair_*_y.npy``."""
    paths = V5.pair_cache_paths(split)
    for key in ("base", "C", "c_offsets", "pair_i0", "pair_i1", "pair_rel", "pair_offsets", "graph_index", "meta"):
        if not Path(paths[key]).exists():
            raise FileNotFoundError(f"missing TCCD-v5 pair cache file: {paths[key]}")
    meta = json.loads(Path(paths["meta"]).read_text(encoding="utf-8"))
    if meta.get("v2_checkpoint_sha256") != V2_BEST_CHECKPOINT_SHA256:
        raise RuntimeError("pair cache was not built from the frozen TCCD-v2 checkpoint")
    if meta.get("official_test_loaded") is not False:
        raise RuntimeError("pair cache metadata does not certify official_test_loaded == false")
    return LabelFreeCache(
        base=np.load(Path(paths["base"])),
        C=np.load(Path(paths["C"])),
        c_offsets=np.load(Path(paths["c_offsets"])),
        pair_i0=np.load(Path(paths["pair_i0"])),
        pair_i1=np.load(Path(paths["pair_i1"])),
        pair_rel=np.load(Path(paths["pair_rel"])),
        pair_offsets=np.load(Path(paths["pair_offsets"])),
        graph_index=np.load(Path(paths["graph_index"])),
        meta=meta,
    )


def load_cache_labels(cache_dir: Path = CACHE_DIR, split: str = "all") -> np.ndarray:
    paths = V5.pair_cache_paths(split)
    return np.load(Path(paths["y"])).astype(np.float32)


# ===========================================================================
# descriptor blocks on the real cache
# ===========================================================================
def graph_pair_means(cache: LabelFreeCache, rows: Sequence[int] | None = None) -> dict[str, np.ndarray]:
    """Compute [A, B, C_prod, D] per graph from the frozen pair cache.

    This is the label-free descriptor used by every v6 arm; it does not read y.
    """
    row_list = list(range(cache.n_graphs)) if rows is None else [int(r) for r in rows]
    G = len(row_list)
    A = np.zeros((G, K_PROTO), dtype=np.float32)
    B = np.zeros((G, K_PROTO), dtype=np.float32)
    Cp = np.zeros((G, K_PROTO), dtype=np.float32)
    D = np.zeros((G, N_REL), dtype=np.float32)
    for out_row, row in enumerate(row_list):
        c0, c1 = int(cache.c_offsets[row]), int(cache.c_offsets[row + 1])
        p0, p1 = int(cache.pair_offsets[row]), int(cache.pair_offsets[row + 1])
        npair = p1 - p0
        if npair <= 0:
            continue
        C = cache.C[c0:c1]
        a = cache.pair_i0[p0:p1]
        b = cache.pair_i1[p0:p1]
        ci = C[a]
        cj = C[b]
        inv = np.float32(1.0 / npair)
        A[out_row] = (ci + cj).sum(axis=0) * inv
        B[out_row] = np.abs(ci - cj).sum(axis=0) * inv
        Cp[out_row] = (ci * cj).sum(axis=0) * inv
        D[out_row] = cache.pair_rel[p0:p1].sum(axis=0) * inv
    return {"A": A, "B": B, "C_prod": Cp, "D": D}


def full_descriptor_from_blocks(blocks: Mapping[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([blocks["A"], blocks["B"], blocks["C_prod"], blocks["D"]], axis=1).astype(np.float32)


def base_reconstruct_marginals(cache: LabelFreeCache) -> dict[str, np.ndarray]:
    """Exact float64 reconstruction of Block A and Block D from ``h_base``.

    A    : ``A_G = 2 * BAG_G / n_G`` with ``n_G = sum_k BAG_G[k]``.
    D_r  : ``sum_{i<j} R_r[i,j] / C(n,2)`` where
           ``sum_{k,l} (C^T R_r C)[k,l] = sum_{i,j} R_r[i,j]`` because every
           softmax assignment row sums to one, and the stored upper triangle
           (including diagonal) gives the full sum when ``R_r`` is symmetric.
    """
    base = cache.base.astype(np.float64)
    G = base.shape[0]
    bag = base[:, :K_PROTO]
    n = bag.sum(axis=1)
    if np.any(n <= 0):
        raise RuntimeError("degenerate graph with zero occurrence mass in frozen base")
    a_rec = (2.0 * bag / n[:, None]).astype(np.float32)

    tri = K_PROTO * (K_PROTO + 1) // 2
    blocks = base[:, K_PROTO:].reshape(G, N_REL, tri)
    iu0, iu1 = T.sym_indices(K_PROTO)
    diag_pos = np.where(iu0 == iu1)[0]
    s_up = blocks.sum(axis=2)
    diag_in_base = blocks[:, :, diag_pos].sum(axis=2)
    s_all = 2.0 * s_up - diag_in_base
    diag_r = n[:, None] * np.asarray(RELATION_DIAGONAL_PER_OCCURRENCE, dtype=np.float64)[None, :]
    sum_offdiag = (s_all - diag_r) / 2.0
    npairs = n * (n - 1.0) / 2.0
    d_rec = (sum_offdiag / npairs[:, None]).astype(np.float32)
    return {"A": a_rec, "D": d_rec, "n": n.astype(np.float32), "s_all": s_all, "diag_r": diag_r}


def label_free_audit(
    cache: LabelFreeCache,
    *,
    rows: Sequence[int] | None = None,
    tol: float = AUDIT_TOL,
) -> dict[str, Any]:
    """Label-free algebra audit of A / D reconstructibility (never reads y)."""
    blocks = graph_pair_means(cache, rows=rows)
    rec = base_reconstruct_marginals(cache)
    row_list = list(range(cache.n_graphs)) if rows is None else [int(r) for r in rows]

    if rows is None:
        a_rec = rec["A"]
        d_rec = rec["D"]
    else:
        a_rec = rec["A"][np.asarray(row_list, dtype=np.int64)]
        d_rec = rec["D"][np.asarray(row_list, dtype=np.int64)]

    a_diff = np.abs(blocks["A"].astype(np.float64) - a_rec.astype(np.float64))
    d_diff = np.abs(blocks["D"].astype(np.float64) - d_rec.astype(np.float64))
    a_max = float(a_diff.max()) if a_diff.size else 0.0
    d_per_relation = [float(d_diff[:, r].max()) if d_diff.size else 0.0 for r in range(N_REL)]
    d_recon = [bool(v < tol) for v in d_per_relation]

    # Reference identity check: sum_{k,l} M_r == sum_{i,j} R_r[i,j] (S_all).
    s_all = rec["s_all"]
    if rows is not None:
        s_all = s_all[np.asarray(row_list, dtype=np.int64)]
    d_diag_num = rec["diag_r"]
    if rows is not None:
        d_diag_num = d_diag_num[np.asarray(row_list, dtype=np.int64)]
    # Reconstruct S_all directly from the pair cache as an independent check:
    # S_all_check = 2*sum_{i<j} R_r + sum_i R_r[i,i] = 2*npairs*D + diag.
    d_pair = blocks["D"].astype(np.float64)
    n = rec["n"]
    if rows is not None:
        n = n[np.asarray(row_list, dtype=np.int64)]
    npairs = n * (n - 1.0) / 2.0
    s_all_check = 2.0 * npairs[:, None] * d_pair + d_diag_num
    s_all_max = float(np.abs(s_all - s_all_check).max()) if s_all.size else 0.0
    s_all_scale = float(np.abs(s_all).max()) if s_all.size else 1.0
    s_all_rel = s_all_max / max(s_all_scale, 1e-6)

    a_pass = bool(a_max < tol)
    d_mask = tuple(d_recon)
    return {
        "official_test_loaded": False,
        "target_not_read": True,
        "n_graphs": int(len(row_list)),
        "tolerance": float(tol),
        "A": {
            "formula": "A_G = 2 * BAG_G / n_G, n_G = sum_k BAG_G[k]",
            "max_abs_diff": a_max,
            "pass": a_pass,
        },
        "D": {
            "formula": "D_G[r] = (S_all[r] - diag_r) / (2 * C(n,2)); S_all = 2*S_up - diag_in_base",
            "relation_names": list(RELATION_NAMES),
            "diagonal_per_occurrence": list(RELATION_DIAGONAL_PER_OCCURRENCE),
            "max_abs_diff_per_relation": d_per_relation,
            "reconstructible_mask": list(d_recon),
            "status": "BASE_RECONSTRUCTIBLE" if all(d_recon) else "PARTIAL",
        },
        "S_all_identity_max_abs_diff": s_all_max,
        "S_all_identity_rel_diff": s_all_rel,
        "recon_mask_nonzero": int(recon_mask_from_relations(d_recon).sum()),
        "novel_mask_nonzero": int(novel_mask_from_relations(d_recon).sum()),
        "frozen_mask_matches": bool(
            np.array_equal(recon_mask_from_relations(d_recon), RECON_MASK)
            and np.array_equal(novel_mask_from_relations(d_recon), NOVEL_MASK)
        ),
        "all_pass": bool(a_pass and (a_max < tol) and (s_all_rel < 1e-4)),
    }


# ===========================================================================
# matched arm model: identical parameters, different input mask / activation
# ===========================================================================
def _branch_container():
    torch = _torch()
    nn = torch.nn

    class _Branch(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(PAIR_WIDTH, PHI_HIDDEN)
            self.fc2 = nn.Linear(PHI_HIDDEN, PHI_OUT)

        def forward(self, x):
            raise RuntimeError("ArmModel drives fc1/fc2 explicitly")

    return _Branch()


class ArmFactory:
    @staticmethod
    def build(arm: str, *, seed: int = 0):
        torch = _torch()
        nn = torch.nn
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}")
        torch.manual_seed(int(seed) + 33001)

        class _M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.branch = _branch_container()
                self.head = nn.Linear(FROZEN_BASE_DIM + PHI_OUT, 1)
                self.arm = arm
                self.nonlinear = bool(ARM_NONLINEAR[arm])
                self.input_block = ARM_INPUT_BLOCK[arm]
                self.input_mask = torch.as_tensor(ARM_MASKS[arm], dtype=torch.float32)
                self.graph_feature_kind = f"frozen_h_v2_plus_{arm}_pair_branch"

            def forward(self, batch: "ArmBatch"):
                torch = _torch()
                x = batch.pbar * self.input_mask.to(batch.pbar.device)
                h = self.branch.fc1(x)
                if self.nonlinear:
                    h = torch.relu(h)
                q = self.branch.fc2(h)
                q = q * batch.has_pairs.unsqueeze(-1).to(q.dtype)
                return self.head(torch.cat([batch.base, q], dim=1)).reshape(-1)

        return _M()


def arm_parameter_count(model) -> dict[str, int]:
    branch = int(sum(p.numel() for n, p in model.named_parameters() if n.startswith("branch.")))
    head = int(sum(p.numel() for n, p in model.named_parameters() if n.startswith("head.")))
    total = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    return {"branch": branch, "head": head, "total": total}


# ===========================================================================
# device-side batch (mirrors tccd_v5.make_pair_batch exactly, records-free)
# ===========================================================================
@dataclass
class ArmBatch:
    base: Any
    pbar: Any
    has_pairs: Any
    y: Any


def make_arm_batch(
    cache: LabelFreeCache,
    y: np.ndarray,
    indices: Sequence[int],
    device: str,
) -> ArmBatch:
    torch = _torch()
    rows = cache.rows_for(indices)
    ns = np.asarray([cache.n_of_row(int(r)) for r in rows], dtype=np.int64)
    B = len(indices)
    N = int(ns.max()) if B else 0
    P = int(max((cache.pair_count_of_row(int(r)) for r in rows), default=0))

    C_np = np.zeros((B, N, K_PROTO), dtype=np.float32)
    i0 = np.zeros((B, P), dtype=np.int64)
    i1 = np.zeros((B, P), dtype=np.int64)
    rel = np.zeros((B, P, N_REL), dtype=np.float32)
    valid = np.zeros((B, P), dtype=np.bool_)
    for bi, r in enumerate(rows):
        r = int(r)
        n = int(ns[bi])
        c0, c1 = int(cache.c_offsets[r]), int(cache.c_offsets[r + 1])
        C_np[bi, :n] = cache.C[c0:c1]
        p0, p1 = int(cache.pair_offsets[r]), int(cache.pair_offsets[r + 1])
        npair = p1 - p0
        if npair:
            i0[bi, :npair] = cache.pair_i0[p0:p1]
            i1[bi, :npair] = cache.pair_i1[p0:p1]
            rel[bi, :npair] = cache.pair_rel[p0:p1]
            valid[bi, :npair] = True

    base = torch.as_tensor(np.asarray(cache.base[rows], dtype=np.float32), device=device)
    C3 = torch.as_tensor(C_np, device=device)
    i0t = torch.as_tensor(i0, device=device)
    i1t = torch.as_tensor(i1, device=device)
    relt = torch.as_tensor(rel, device=device)
    validt = torch.as_tensor(valid, device=device)
    p = V5.pair_descriptor(C3, i0t, i1t, relt)
    vmask = validt.unsqueeze(-1).to(p.dtype)
    count = validt.sum(dim=1, keepdim=True).clamp_min(1).to(p.dtype)
    pbar = (p * vmask).sum(dim=1) / count
    return ArmBatch(
        base=base,
        pbar=pbar,
        has_pairs=validt.any(dim=1),
        y=torch.as_tensor(np.asarray(y[rows], dtype=np.float32), device=device),
    )


# ===========================================================================
# Stage A training (exact TCCD-v5 lightweight protocol, mask-aware)
# ===========================================================================
@dataclass
class ArmResult:
    best_valid: float
    best_epoch: int
    soup_valid: float | None
    soup_members: list[int]
    init_checksum: str
    state_best: Any = None
    state_soup: Any = None
    train_history: list[dict[str, Any]] = field(default_factory=list)
    wall_s: float = 0.0


def evaluate_arm(model, cache: LabelFreeCache, y: np.ndarray, indices: Sequence[int], device: str, *, batch: int = 64) -> float:
    torch = _torch()
    model.eval()
    errs = []
    with torch.no_grad():
        for start in range(0, len(indices), int(batch)):
            chunk = list(indices[start : start + int(batch)])
            b = make_arm_batch(cache, y, chunk, device)
            pred = model(b)
            errs.append((pred - b.y).abs().cpu().numpy())
    return float(np.concatenate(errs).mean())


def train_arm(
    cache: LabelFreeCache,
    y: np.ndarray,
    train_indices: Sequence[int],
    dev_indices: Sequence[int],
    device: str,
    arm: str,
    *,
    seed: int = 0,
    max_epochs: int = T.MAX_EPOCHS,
    patience: int = T.PATIENCE,
    batch: int = T.BATCH,
    lr: float = T.LR,
    wd: float = T.WD,
    clip: float = T.CLIP,
    log=print,
) -> ArmResult:
    torch = _torch()
    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed) + 91011)
    model = ArmFactory.build(arm, seed=seed).to(device)
    init_checksum = V5._state_checksum(model)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=wd)
    best = math.inf
    best_epoch = -1
    best_state = None
    top: list[tuple[float, int, Any]] = []
    history: list[dict[str, Any]] = []
    stale = 0
    started = time.time()
    train_indices = list(train_indices)
    for epoch in range(1, int(max_epochs) + 1):
        model.train()
        order = rng.permutation(len(train_indices))
        total = 0.0
        count = 0
        for start in range(0, len(order), int(batch)):
            sel = [train_indices[i] for i in order[start : start + int(batch)]]
            b = make_arm_batch(cache, y, sel, device)
            opt.zero_grad(set_to_none=True)
            pred = model(b)
            loss = (pred - b.y).abs().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], clip)
            opt.step()
            total += float(loss.detach()) * len(sel)
            count += len(sel)
        dev_mae = evaluate_arm(model, cache, y, dev_indices, device)
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        row = {"epoch": int(epoch), "train_loss": total / max(count, 1), "valid": dev_mae}
        history.append(row)
        if epoch == 1 or epoch % 20 == 0 or epoch == int(max_epochs):
            log(f"[stageA:{arm}] epoch={epoch:03d} train={row['train_loss']:.6f} valid={dev_mae:.6f} best={best:.6f}@{best_epoch}")
        if dev_mae < best - 1e-9:
            best = dev_mae
            best_epoch = epoch
            best_state = copy.deepcopy(state)
            stale = 0
        else:
            stale += 1
        if len(top) < T.TOP_K_SOUP or dev_mae < max(item[0] for item in top):
            top.append((dev_mae, epoch, state))
            top.sort(key=lambda item: (item[0], item[1]))
            top = top[: T.TOP_K_SOUP]
        if stale >= int(patience):
            log(f"[stageA:{arm}] early stop at epoch {epoch}")
            break

    soup = None
    soup_mae = None
    members: list[int] = []
    if top:
        members = [e for _, e, _ in top]
        keys = top[0][2].keys()
        soup = {k: sum(st[k].float() for _, _, st in top) / len(top) for k in keys}
        backup = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(soup)
        soup_mae = evaluate_arm(model, cache, y, dev_indices, device)
        model.load_state_dict(backup)
    if best_state is not None:
        model.load_state_dict(best_state)
    return ArmResult(
        best_valid=float(best),
        best_epoch=int(best_epoch),
        soup_valid=None if soup_mae is None else float(soup_mae),
        soup_members=members,
        init_checksum=init_checksum,
        state_best=best_state,
        state_soup=soup,
        train_history=history,
        wall_s=time.time() - started,
    )


def evaluate_saved_state(
    arm: str,
    state: Mapping[str, Any],
    cache: LabelFreeCache,
    y: np.ndarray,
    indices: Sequence[int],
    device: str,
    *,
    seed: int = 0,
    batch: int = 64,
) -> float:
    model = ArmFactory.build(arm, seed=seed).to(device)
    model.load_state_dict(state)
    return evaluate_arm(model, cache, y, indices, device, batch=batch)


# ===========================================================================
# decision logic
# ===========================================================================
def decomposition_decision(
    *,
    base_soup: float = BASE_SOUP_MAE,
    full_soup: float = FULL_SOUP_MAE,
    recon_nl_soup: float,
    recon_linfact_soup: float,
    novel_nl_soup: float,
    recon_linfact_gap_seed1: float | None = None,
) -> dict[str, Any]:
    g_full = float(base_soup) - float(full_soup)
    g_recon = float(base_soup) - float(recon_nl_soup)
    g_novel = float(base_soup) - float(novel_nl_soup)
    rho_recon = g_recon / g_full if g_full else float("nan")
    rho_novel = g_novel / g_full if g_full else float("nan")
    gap = float(recon_linfact_soup) - float(recon_nl_soup)

    recon_sufficient = (g_recon >= RECON_SUFFICIENT_GAIN) and (float(recon_nl_soup) <= float(full_soup) + RECON_SUFFICIENT_SLACK)
    novel_sufficient = (g_novel >= NOVEL_SUFFICIENT_GAIN) and (float(novel_nl_soup) <= float(full_soup) + NOVEL_SUFFICIENT_SLACK)
    joint = (
        g_recon >= JOINT_MIN_GAIN
        and g_novel >= JOINT_MIN_GAIN
        and float(recon_nl_soup) > float(full_soup) + JOINT_SLACK
        and float(novel_nl_soup) > float(full_soup) + JOINT_SLACK
    )
    neither = (g_recon < JOINT_MIN_GAIN) and (g_novel < JOINT_MIN_GAIN) and (g_full >= 0.02)

    needs_seed1 = False
    if recon_sufficient:
        if gap < NONLIN_GAP_R1:
            case = "R1"
        elif gap >= NONLIN_GAP_R2:
            case = "R2"
        elif recon_linfact_gap_seed1 is None:
            case = "R_AMBIGUOUS_NEEDS_SEED1"
            needs_seed1 = True
        else:
            mean_gap = 0.5 * (gap + float(recon_linfact_gap_seed1))
            case = "R2" if mean_gap >= NONLIN_GAP_SEED1_MEAN else "R1"
    elif novel_sufficient:
        case = "M"
    elif joint:
        case = "J"
    elif neither:
        case = "N"
    else:
        case = "INCONCLUSIVE_INTERMEDIATE"

    return {
        "case": case,
        "needs_seed1_recon": bool(needs_seed1),
        "case_R_recon_sufficient": bool(recon_sufficient),
        "case_M_novel_sufficient": bool(novel_sufficient),
        "case_J_joint": bool(joint),
        "case_N_strong_synergy": bool(neither),
        "g_full": g_full,
        "g_recon": g_recon,
        "g_novel": g_novel,
        "rho_recon": rho_recon,
        "rho_novel": rho_novel,
        "nonlinearity_gap": gap,
        "recon_linfact_gap_seed1": None if recon_linfact_gap_seed1 is None else float(recon_linfact_gap_seed1),
        "thresholds": {
            "recon_sufficient_gain": RECON_SUFFICIENT_GAIN,
            "recon_sufficient_slack": RECON_SUFFICIENT_SLACK,
            "novel_sufficient_gain": NOVEL_SUFFICIENT_GAIN,
            "novel_sufficient_slack": NOVEL_SUFFICIENT_SLACK,
            "joint_min_gain": JOINT_MIN_GAIN,
            "joint_slack": JOINT_SLACK,
            "nonlin_gap_r1": NONLIN_GAP_R1,
            "nonlin_gap_r2": NONLIN_GAP_R2,
            "nonlin_gap_seed1_mean": NONLIN_GAP_SEED1_MEAN,
        },
    }


# ===========================================================================
# Gate 0 -- data-free + real-cache correctness
# ===========================================================================
def _tiny_records() -> list[dict[str, Any]]:
    rng = np.random.default_rng(21)
    return [{"n": 5 + gi, "y": float(gi), "seed": int(rng.integers(0, 1 << 30))} for gi in range(3)]


def tiny_cache(seed: int = 0) -> tuple[LabelFreeCache, np.ndarray]:
    """Synthetic label-free cache in the exact ragged layout of the v5 cache."""
    records = _tiny_records()
    C_list, i0_list, i1_list, rel_list = [], [], [], []
    for rec in records:
        n = int(rec["n"])
        local = np.random.default_rng(int(rec["seed"])).standard_normal((n, K_PROTO)).astype(np.float32)
        C_list.append(local)
        a0, a1 = np.triu_indices(n, 1)
        i0_list.append(a0.astype(np.int64))
        i1_list.append(a1.astype(np.int64))
        rng = np.random.default_rng(500 + int(rec["seed"]))
        R = rng.standard_normal((N_REL, n, n)).astype(np.float32)
        for r in range(N_REL):
            R[r] = 0.5 * (R[r] + R[r].T)
        rel_list.append(np.ascontiguousarray(R[:, a0, a1].T, dtype=np.float32))
    C = np.concatenate(C_list)
    n_patches = np.asarray([len(c) for c in C_list])
    n_pairs = np.asarray([len(a) for a in i0_list])
    base = np.random.default_rng(31 + seed).standard_normal((len(records), FROZEN_BASE_DIM)).astype(np.float32)
    # keep the synthetic BAG slice non-negative so a deliberately inconsistent
    # base still exercises the audit failure path instead of a degenerate graph.
    base[:, :K_PROTO] = np.abs(base[:, :K_PROTO]) + 0.05
    y = np.asarray([rec["y"] for rec in records], dtype=np.float32)
    cache = LabelFreeCache(
        base=base,
        C=C,
        c_offsets=np.concatenate([[0], np.cumsum(n_patches)]).astype(np.int64),
        pair_i0=np.concatenate(i0_list),
        pair_i1=np.concatenate(i1_list),
        pair_rel=np.concatenate(rel_list),
        pair_offsets=np.concatenate([[0], np.cumsum(n_pairs)]).astype(np.int64),
        graph_index=np.arange(len(records), dtype=np.int64),
        meta={"protocol": PROTOCOL_VERSION, "official_test_loaded": False},
    )
    return cache, y


class _V5CacheView:
    """Shallow view exposing the v5 PairCache surface (plus y) for reference checks."""

    def __init__(self, cache: LabelFreeCache, y: np.ndarray) -> None:
        self._cache = cache
        self.y = np.asarray(y, dtype=np.float32)

    def __getattr__(self, name: str):
        return getattr(self._cache, name)


def _pair_means_via_v5(cache: LabelFreeCache, y: np.ndarray, indices: Sequence[int], device: str = "cpu"):
    """Reference pbar from the exact v5 tensor path on stub records."""
    records = [{"n": cache.n_of_row(r)} for r in range(cache.n_graphs)]
    batch = V5.make_pair_batch(_V5CacheView(cache, y), records, indices, device)
    p = V5.pair_descriptor(batch.C3, batch.i0, batch.i1, batch.rel)
    vmask = batch.valid.unsqueeze(-1).to(p.dtype)
    count = batch.valid.sum(dim=1, keepdim=True).clamp_min(1).to(p.dtype)
    return ((p * vmask).sum(dim=1) / count).detach().cpu().numpy()


def gate0_data_free_checks(device: str = "cpu") -> dict[str, Any]:
    torch = _torch()
    checks: dict[str, Any] = {"official_test_loaded": False, "target_not_read_for_audit": True}

    checks["pair_width"] = int(PAIR_WIDTH)
    checks["frozen_base_dim"] = int(FROZEN_BASE_DIM)
    checks["block_dims"] = dict(BLOCK_DIMS)
    checks["descriptor_dims_ok"] = bool(PAIR_WIDTH == 3 * K_PROTO + N_REL)

    # partition
    recon = RECON_MASK.astype(np.int64)
    novel = NOVEL_MASK.astype(np.int64)
    checks["recon_mask_nonzero"] = int(recon.sum())
    checks["novel_mask_nonzero"] = int(novel.sum())
    checks["mask_partition_exact"] = bool(np.array_equal(recon + novel, np.ones(PAIR_WIDTH, dtype=np.int64)))

    # parameter equality + initialization equality
    models = {arm: ArmFactory.build(arm, seed=0) for arm in ARMS}
    params = {arm: arm_parameter_count(m) for arm, m in models.items()}
    checks["parameter_counts"] = params
    checks["parameter_equality"] = bool(len({(p["branch"], p["head"], p["total"]) for p in params.values()}) == 1)
    checks["expected_parameter_count"] = parameter_accounting()
    checks["parameter_matches_registered"] = bool(
        all(p == parameter_accounting() for p in params.values())
    )
    init_checksums = {arm: V5._state_checksum(m) for arm, m in models.items()}
    checks["init_checksums"] = init_checksums
    checks["init_equality"] = bool(len(set(init_checksums.values())) == 1)
    checks["init_matches_v5_post"] = bool(init_checksums["full_nl"] == FULL_INIT_CHECKSUM)

    named = {arm: {n: p.detach().clone() for n, p in m.named_parameters()} for arm, m in models.items()}
    max_init_diff = 0.0
    for arm in ARMS:
        for name in named["full_nl"]:
            max_init_diff = max(max_init_diff, float((named[arm][name] - named["full_nl"][name]).abs().max()))
    checks["init_max_abs_diff"] = max_init_diff
    checks["init_bit_identical"] = bool(max_init_diff == 0.0)

    # shapes
    m = models["full_nl"]
    checks["shapes"] = {
        "fc1": [int(m.branch.fc1.in_features), int(m.branch.fc1.out_features)],
        "fc2": [int(m.branch.fc2.in_features), int(m.branch.fc2.out_features)],
        "head": [int(m.head.in_features), int(m.head.out_features)],
    }
    checks["shapes_ok"] = bool(
        checks["shapes"]["fc1"] == [197, 64]
        and checks["shapes"]["fc2"] == [64, 16]
        and checks["shapes"]["head"] == [FROZEN_BASE_DIM + PHI_OUT, 1]
    )

    # FULL == RECON + NOVEL per coordinate (on a synthetic descriptor)
    rng = np.random.default_rng(3)
    pbar = rng.standard_normal((4, PAIR_WIDTH)).astype(np.float32)
    full = pbar
    split = pbar * RECON_MASK[None, :] + pbar * NOVEL_MASK[None, :]
    checks["full_equals_recon_plus_novel_max_abs_diff"] = float(np.abs(full - split).max())
    checks["full_equals_recon_plus_novel"] = bool(float(np.abs(full - split).max()) < 1e-6)

    cache, y = tiny_cache()
    idx = list(range(cache.n_graphs))

    # our pbar vs exact v5 tensor path
    rows = cache.rows_for(idx)
    ours = np.concatenate([make_arm_batch(cache, y, [i], device).pbar.detach().cpu().numpy() for i in idx], axis=0)
    ref = _pair_means_via_v5(cache, y, idx, device)
    checks["v5_descriptor_equivalence_max_abs_diff"] = float(np.abs(ours - ref).max())
    checks["v5_descriptor_equivalence"] = bool(float(np.abs(ours - ref).max()) <= 1e-6)

    # batching invariance
    batched = make_arm_batch(cache, y, idx, device).pbar.detach().cpu().numpy()
    checks["batching_invariance_max_abs_diff"] = float(np.abs(batched - ours).max())
    checks["batching_invariant"] = bool(float(np.abs(batched - ours).max()) <= 1e-6)

    # pair-order invariance: rebuild the dynamic arrays in a permuted order
    C_list, i0_list, i1_list, rel_list, offs = [], [], [], [], [0]
    for row in range(cache.n_graphs):
        p0, p1 = int(cache.pair_offsets[row]), int(cache.pair_offsets[row + 1])
        local_order = np.random.default_rng(100 + row).permutation(p1 - p0)
        c0, c1 = int(cache.c_offsets[row]), int(cache.c_offsets[row + 1])
        C_list.append(cache.C[c0:c1])
        i0_list.append(cache.pair_i0[p0:p1][local_order])
        i1_list.append(cache.pair_i1[p0:p1][local_order])
        rel_list.append(cache.pair_rel[p0:p1][local_order])
        offs.append(offs[-1] + (p1 - p0))
    perm_cache = LabelFreeCache(
        base=cache.base,
        C=np.concatenate(C_list),
        c_offsets=cache.c_offsets.copy(),
        pair_i0=np.concatenate(i0_list),
        pair_i1=np.concatenate(i1_list),
        pair_rel=np.concatenate(rel_list),
        pair_offsets=np.asarray(offs, dtype=np.int64),
        graph_index=cache.graph_index.copy(),
        meta=cache.meta,
    )
    perm = np.concatenate([make_arm_batch(perm_cache, y, [i], device).pbar.detach().cpu().numpy() for i in idx], axis=0)
    checks["pair_order_invariance_max_abs_diff"] = float(np.abs(perm - ours).max())
    checks["pair_order_invariant"] = bool(float(np.abs(perm - ours).max()) <= 1e-6)

    # node-relabel invariance on one graph (joint C row and pair relabel)
    row = 0
    n = cache.n_of_row(row)
    perm_rows = np.random.default_rng(13).permutation(n)
    inv = np.empty(n, dtype=np.int64)
    inv[perm_rows] = np.arange(n)
    c0, c1 = int(cache.c_offsets[row]), int(cache.c_offsets[row + 1])
    p0, p1 = int(cache.pair_offsets[row]), int(cache.pair_offsets[row + 1])
    new_i0 = inv[cache.pair_i0[p0:p1]]
    new_i1 = inv[cache.pair_i1[p0:p1]]
    # relation values are scalar per pair: relabel only reorders/permutes pairs, so
    # rebuild them from a symmetric full operator to keep the check meaningful.
    R = np.random.default_rng(17).standard_normal((N_REL, n, n)).astype(np.float32)
    for r in range(N_REL):
        R[r] = 0.5 * (R[r] + R[r].T)
    a0, a1 = np.triu_indices(n, 1)
    rel_orig = np.ascontiguousarray(R[:, a0, a1].T, dtype=np.float32)
    rel_perm = np.ascontiguousarray(R[:, perm_rows][:, :, perm_rows][:, a0, a1].T, dtype=np.float32)
    rel_cache = LabelFreeCache(
        base=cache.base,
        C=cache.C[c0:c1][perm_rows],
        c_offsets=np.asarray([0, n], dtype=np.int64),
        pair_i0=a0.astype(np.int64),
        pair_i1=a1.astype(np.int64),
        pair_rel=rel_perm,
        pair_offsets=np.asarray([0, len(a0)], dtype=np.int64),
        graph_index=np.asarray([0], dtype=np.int64),
        meta=cache.meta,
    )
    orig_cache = LabelFreeCache(
        base=cache.base,
        C=cache.C[c0:c1],
        c_offsets=np.asarray([0, n], dtype=np.int64),
        pair_i0=a0.astype(np.int64),
        pair_i1=a1.astype(np.int64),
        pair_rel=rel_orig,
        pair_offsets=np.asarray([0, len(a0)], dtype=np.int64),
        graph_index=np.asarray([0], dtype=np.int64),
        meta=cache.meta,
    )
    pbar_o = make_arm_batch(orig_cache, y, [0], device).pbar
    pbar_r = make_arm_batch(rel_cache, y, [0], device).pbar
    checks["relabel_invariance_max_abs_diff"] = float((pbar_o - pbar_r).abs().max().detach().cpu())
    checks["relabel_invariant"] = bool(checks["relabel_invariance_max_abs_diff"] <= 1e-5)

    # arm prediction invariance under relabel with frozen weights
    m.eval()
    with torch.no_grad():
        pred_o = m(make_arm_batch(orig_cache, y, [0], device))
        pred_r = m(make_arm_batch(rel_cache, y, [0], device))
    checks["relabel_prediction_max_abs_diff"] = float((pred_o - pred_r).abs().max().detach().cpu())
    checks["relabel_prediction_invariant"] = bool(checks["relabel_prediction_max_abs_diff"] <= 1e-5)

    # empty-pair graph: branch output must be exactly zero
    empty = LabelFreeCache(
        base=np.zeros((1, FROZEN_BASE_DIM), dtype=np.float32),
        C=np.zeros((1, K_PROTO), dtype=np.float32),
        c_offsets=np.asarray([0, 1], dtype=np.int64),
        pair_i0=np.zeros(0, dtype=np.int64),
        pair_i1=np.zeros(0, dtype=np.int64),
        pair_rel=np.zeros((0, N_REL), dtype=np.float32),
        pair_offsets=np.asarray([0, 0], dtype=np.int64),
        graph_index=np.asarray([0], dtype=np.int64),
        meta=cache.meta,
    )
    with torch.no_grad():
        b = make_arm_batch(empty, np.asarray([0.0], dtype=np.float32), [0], device)
        x = b.pbar * m.input_mask.to(b.pbar.device)
        hh = m.branch.fc1(x)
        hh = torch.relu(hh)
        q = m.branch.fc2(hh) * b.has_pairs.unsqueeze(-1).to(torch.float32)
    checks["empty_pair_branch_zero"] = bool(float(q.abs().max().detach().cpu()) == 0.0)
    checks["empty_pair_pbar_zero"] = bool(float(b.pbar.abs().max().detach().cpu()) == 0.0)

    bool_keys = [k for k, v in checks.items() if isinstance(v, bool) and k != "official_test_loaded"]
    checks["data_free_all_pass"] = bool(all(checks[k] for k in bool_keys))
    return checks


def gate0_real_cache_checks(cache: LabelFreeCache, device: str = "cpu") -> dict[str, Any]:
    checks: dict[str, Any] = {"official_test_loaded": False}
    audit = label_free_audit(cache, tol=AUDIT_TOL)
    checks["audit"] = audit
    checks["A_reconstructible"] = bool(audit["A"]["pass"])
    checks["D_status"] = audit["D"]["status"]
    checks["D_reconstructible"] = bool(all(audit["D"]["reconstructible_mask"]))
    checks["frozen_mask_matches_audit"] = bool(audit["frozen_mask_matches"])
    checks["S_all_identity_max_abs_diff"] = audit["S_all_identity_max_abs_diff"]
    checks["S_all_identity_rel_diff"] = audit["S_all_identity_rel_diff"]
    checks["S_all_identity_ok"] = bool(audit["S_all_identity_rel_diff"] < 1e-4)

    # exact v5 descriptor equivalence on a small real-cache subset
    idx = list(range(min(24, cache.n_graphs)))
    ours = np.concatenate(
        [make_arm_batch(cache, np.zeros(cache.n_graphs, dtype=np.float32), [i], device).pbar.detach().cpu().numpy() for i in idx],
        axis=0,
    )
    ref = _pair_means_via_v5(cache, np.zeros(cache.n_graphs, dtype=np.float32), idx, device)
    checks["v5_descriptor_equivalence_max_abs_diff"] = float(np.abs(ours - ref).max())
    checks["v5_descriptor_equivalence"] = bool(float(np.abs(ours - ref).max()) <= 1e-6)

    # FULL = RECON + NOVEL on the real descriptor subset
    full = ours
    recon = ours * RECON_MASK[None, :]
    novel = ours * NOVEL_MASK[None, :]
    checks["full_split_max_abs_diff"] = float(np.abs(full - (recon + novel)).max())
    checks["full_split_ok"] = bool(checks["full_split_max_abs_diff"] < 1e-6)
    checks["real_all_pass"] = bool(
        checks["A_reconstructible"]
        and checks["D_reconstructible"]
        and checks["frozen_mask_matches_audit"]
        and checks["S_all_identity_ok"]
        and checks["v5_descriptor_equivalence"]
        and checks["full_split_ok"]
    )
    return checks
