#!/usr/bin/env python
"""TCCD-v7 explicit normalized prototype / relation moment representation.

Frozen closing round of the TCCD standalone-predictor route.  Nothing new is
computed structurally: the representation that TCCD-v5/v6 proved to be the true
source of the local gain (size normalization + normalized relation
contractions) is written explicitly, and the prototype vocabulary is then
allowed to co-adapt end-to-end behind a *linear* reader.

Representation (frozen in the preregistration):

    mu     = (1/n) sum_i c_i
    m2     = (1/n) sum_i c_i^{o2}
    v      = max(m2 - mu^{o2}, 0)
    M_r    = C^T R_r C
    s_r    = sum_ij R_r(i,j)
    Mhat_r = M_r / max(s_r, eps)          (0 exactly when s_r == 0)

    h_N    = [mu,   {vec_sym(Mhat_r)}, log(1+n), {log(1+s_r)}]          (10470)
    h_NM   = [mu, v,{vec_sym(Mhat_r)}, log(1+n), {log(1+s_r)}]          (10534)
    h_RAW  = [sum_i c_i, {vec_sym(C^T R_r C)}]                          (10464)

Reader: ``Linear(h_G, 1)`` only.  Official ZINC test is never loaded; formal
execution is GPU1-only.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import MethodType
from typing import Any, Mapping, Sequence

import numpy as np

from tracks.ksvd.code import tccd_v0 as T
from tracks.ksvd.code import tccd_v2 as V2
from tracks.ksvd.code import tccd_v5 as V5
from tracks.ksvd.code import tccd_v6 as V6

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = REPO_ROOT / "tracks/ksvd/results/tccd_v7"
CACHE_DIR = V5.CACHE_DIR
PROTOCOL_VERSION = "tccd_v7_final_normalized_moment_v1"

D_LOCAL = V2.D_LOCAL
K_PROTO = V2.K_PROTO
N_REL = V2.N_REL
EPS = V2.EPS
PROTO_INIT_SEED = V2.PROTO_INIT_SEED
TAU = T.TAU

TRI = K_PROTO * (K_PROTO + 1) // 2  # 2080
MEAN_DIM = K_PROTO
MOM_DIM = K_PROTO
REL_DIM = N_REL * TRI
SCALE_DIM = 1 + N_REL

RAW_DIM = V6.FROZEN_BASE_DIM  # 10464
NORM_DIM = MEAN_DIM + REL_DIM + SCALE_DIM  # 10470
NORM_MOM_DIM = MEAN_DIM + MOM_DIM + REL_DIM + SCALE_DIM  # 10534

RELATION_NAMES = V6.RELATION_NAMES
RELATION_DIAGONAL_PER_OCCURRENCE = V6.RELATION_DIAGONAL_PER_OCCURRENCE

V2_BEST_CHECKPOINT = V5.V2_BEST_CHECKPOINT
V2_BEST_CHECKPOINT_SHA256 = V5.V2_BEST_CHECKPOINT_SHA256
V5_STAGE_A_JSON = V5.RESULTS_DIR / "stageA_seed0.json"
V5_STAGE_A_SHA256 = "2ab232c74219529ef3ddbdf1c6175d664756eb0dc7c0a0a30325d62a94b8aefe"

# ---------------------------------------------------------------------------
# frozen references
# ---------------------------------------------------------------------------
RAW_BEST_MAE = 0.28791576623916626
RAW_SOUP_MAE = 0.2783639132976532
V2_INTERNAL_BEST_MAE = V5.TCCD_V2_BEST  # 0.2862437069416046
V2_INTERNAL_SOUP_MAE = V5.TCCD_V2_SOUP  # 0.26635152101516724
V2_OFFICIAL_VALID_BEST = V5.TCCD_V2_OFFICIAL_VALID_BEST  # 0.287337
V2_OFFICIAL_VALID_SOUP = V5.TCCD_V2_OFFICIAL_VALID_SOUP  # 0.261988
CANONICAL_GPU1_BASELINE = V5.CANONICAL_GPU1_BASELINE  # 0.119818

# ---------------------------------------------------------------------------
# preregistered thresholds (frozen)
# ---------------------------------------------------------------------------
STAGE_A_NM_ABS_MAX = 0.255
STAGE_A_DELTA_NORM_MIN = 0.015
STAGE_A_PROTOCOL_DRIFT_MAX = 0.005
DELTA_MOM_MATERIAL = 0.005

STAGE_B_STOP_GT = 0.23
STAGE_B_PARTIAL_GT = 0.20

OFFICIAL_COMPETITIVE_MAX = 0.15
OFFICIAL_INSUFFICIENT_GT = 0.20

ARMS_STAGE_A = ("raw", "raw_rescreen", "norm", "norm_mom")
ARM_WITH_MOMENTS = {"norm": False, "norm_mom": True}
ARM_DIM = {"raw": RAW_DIM, "raw_rescreen": RAW_DIM, "norm": NORM_DIM, "norm_mom": NORM_MOM_DIM}

VARIANCE_MATERIAL = "MATERIAL"
VARIANCE_NEGLIGIBLE = "NEGLIGIBLE"


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


def representation_dim(with_moments: bool) -> int:
    return int(NORM_MOM_DIM if with_moments else NORM_DIM)


def feature_layout(with_moments: bool) -> list[dict[str, Any]]:
    """Explicit, ordered feature layout (registered order)."""
    blocks: list[dict[str, Any]] = [{"name": "mu", "dim": MEAN_DIM, "definition": "(1/n) sum_i c_i"}]
    if with_moments:
        blocks.append(
            {"name": "v", "dim": MOM_DIM, "definition": "max((1/n) sum_i c_i^2 - mu^2, 0)"}
        )
    blocks.append(
        {
            "name": "Mhat_rel",
            "dim": REL_DIM,
            "definition": "vec_sym(C^T R_r C / max(s_r, eps)) for r=1..5 (upper triangle incl. diagonal)",
            "relations": list(RELATION_NAMES),
        }
    )
    blocks.append({"name": "log_size", "dim": 1, "definition": "log(1+n)"})
    blocks.append(
        {"name": "log_mass", "dim": N_REL, "definition": "log(1+s_r) for r=1..5", "relations": list(RELATION_NAMES)}
    )
    return blocks


def parameter_accounting(with_moments: bool = True) -> dict[str, int]:
    """End-to-end Stage-B parameter accounting (encoder + prototypes + temp + linear reader)."""
    layout = V2.frozen_layout()
    encoder = int(layout.feature_dim * D_LOCAL)
    prototypes = int(D_LOCAL * K_PROTO)
    temperature = 1
    reader = int(representation_dim(with_moments)) + 1
    return {
        "encoder": encoder,
        "prototypes": prototypes,
        "temperature": temperature,
        "reader": reader,
        "total": encoder + prototypes + temperature + reader,
    }


def reader_parameter_count(with_moments: bool) -> int:
    return int(representation_dim(with_moments)) + 1


# ===========================================================================
# Stage-A features from the frozen pair cache (no y is ever read)
# ===========================================================================
def relation_masses_from_cache(cache: V6.LabelFreeCache) -> np.ndarray:
    """s_r = sum_{ij} R_r(i,j) from the frozen upper-triangle pair cache."""
    offsets = np.asarray(cache.pair_offsets, dtype=np.int64)
    G = int(len(offsets) - 1)
    out = np.zeros((G, N_REL), dtype=np.float64)
    rel = cache.pair_rel.astype(np.float64)
    for g in range(G):
        a, b = int(offsets[g]), int(offsets[g + 1])
        if b > a:
            out[g] = rel[a:b].sum(axis=0)
    n = node_counts_from_cache(cache).astype(np.float64)
    diag = np.asarray(RELATION_DIAGONAL_PER_OCCURRENCE, dtype=np.float64)
    return 2.0 * out + n[:, None] * diag[None, :]


def node_counts_from_cache(cache: V6.LabelFreeCache) -> np.ndarray:
    return np.diff(np.asarray(cache.c_offsets, dtype=np.int64)).astype(np.float64)


def squared_assignment_sums(cache: V6.LabelFreeCache) -> np.ndarray:
    """sum_i c_i^{o2} per graph (float64), read from the frozen assignment cache."""
    offsets = np.asarray(cache.c_offsets, dtype=np.int64)
    G = int(len(offsets) - 1)
    out = np.zeros((G, K_PROTO), dtype=np.float64)
    C = cache.C.astype(np.float64)
    for g in range(G):
        a, b = int(offsets[g]), int(offsets[g + 1])
        if b > a:
            out[g] = (C[a:b] ** 2).sum(axis=0)
    return out


def stage_a_features(
    cache: V6.LabelFreeCache,
    *,
    with_moments: bool,
    rows: Sequence[int] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Frozen NORM / NORM+MOM feature matrix built only from base, C and pair R.

    Derives ``mu`` and ``M_r`` from the frozen TCCD-v2 ``h_base`` and ``s_r``
    from the frozen pair cache; never touches labels.
    """
    base = cache.base.astype(np.float64)
    G = int(base.shape[0])
    row_list = list(range(G)) if rows is None else [int(r) for r in rows]
    take = np.asarray(row_list, dtype=np.int64)

    n = node_counts_from_cache(cache)
    if np.any(n[take] <= 0):
        raise RuntimeError("degenerate graph with zero occurrences in the frozen base")
    mu = base[:, :K_PROTO][take] / n[take][:, None]

    tri = TRI
    M = base[:, K_PROTO:].reshape(G, N_REL, tri)[take]
    s = relation_masses_from_cache(cache)[take]
    Mhat = M / np.maximum(s, EPS)[:, :, None]

    parts: list[np.ndarray] = [mu]
    if with_moments:
        m2 = squared_assignment_sums(cache)[take] / n[take][:, None]
        v = np.maximum(m2 - mu**2, 0.0)
        parts.append(v)
    parts.append(Mhat.reshape(len(row_list), -1))
    parts.append(np.log1p(n[take])[:, None])
    parts.append(np.log1p(s))
    features = np.concatenate(parts, axis=1).astype(np.float32)
    diagnostics = {
        "n_graphs": int(len(row_list)),
        "dim": int(features.shape[1]),
        "with_moments": bool(with_moments),
        "mu_row_sum_max_abs_dev": float(np.abs(mu.sum(axis=1) - 1.0).max()),
        "relation_mass_min": float(s.min()),
        "relation_mass_zero_counts": (s == 0).sum(axis=0).astype(int).tolist(),
        "zero_mass_absolute_max": float(np.abs(Mhat[s == 0]).max()) if np.any(s == 0) else 0.0,
        "target_not_read": True,
    }
    return features, diagnostics


def raw_features(cache: V6.LabelFreeCache, rows: Sequence[int] | None = None) -> np.ndarray:
    """Exact TCCD-v2 raw representation (frozen ``h_base``)."""
    base = cache.base.astype(np.float32)
    if rows is None:
        return np.ascontiguousarray(base)
    return np.ascontiguousarray(base[np.asarray([int(r) for r in rows], dtype=np.int64)])


# ===========================================================================
# Stage-B / tensor features (the representation the end-to-end model emits)
# ===========================================================================
def nm_graph_features(C3, R_pad, valid, iu0, iu1, *, with_moments: bool = True):
    """[B,N,K] assignments + [B,m,N,N] relations + [B,N] mask -> [B, dim] h_NM."""
    torch = _torch()
    C = C3 * valid.unsqueeze(-1).to(C3.dtype)  # padded rows are exactly zero
    n = valid.sum(dim=1, keepdim=True).to(C.dtype)
    mu = C.sum(dim=1) / n
    s = R_pad.sum(dim=(2, 3))
    M = torch.einsum("bnk,brnm,bml->brkl", C, R_pad, C)
    Mhat = M / s.clamp_min(EPS)[:, :, None, None]
    parts = [mu]
    if with_moments:
        m2 = (C * C).sum(dim=1) / n
        v = (m2 - mu * mu).clamp_min(0.0)
        parts.append(v)
    parts.append(Mhat[:, :, iu0, iu1].reshape(C.shape[0], -1))
    parts.append(torch.log1p(n))
    parts.append(torch.log1p(s))
    return torch.cat(parts, dim=1)


def representation_from_assignments(C3, R_pad, valid, iu0, iu1, *, with_moments: bool):
    return nm_graph_features(C3, R_pad, valid, iu0, iu1, with_moments=with_moments)


# ===========================================================================
# Stage-B model: exact TCCD-v2 encoder/vocabulary + explicit NM representation
# ===========================================================================
class NMPrototypeModelFactory:
    """TCCD-v2 parameterization, TCCD-v2 regularizers, NM graph representation."""

    @staticmethod
    def build(
        F: int,
        d: int,
        K: int,
        n_rel: int,
        encoder_init: np.ndarray,
        *,
        with_moments: bool = True,
        proto_seed: int = PROTO_INIT_SEED,
        seed: int = 0,
    ):
        torch = _torch()
        nn = torch.nn
        model = V2.PrototypeModelFactory.build(
            F, d, K, n_rel, encoder_init, mode="rel", proto_seed=proto_seed, seed=seed
        )
        dim = representation_dim(with_moments)
        model.head = nn.Linear(dim, 1)
        model.with_moments = bool(with_moments)
        model.graph_feature_kind = "normalized_moment_representation_v1"

        def graph_features(self, C, batch, *, shuffle=False, indices=None, records=None, seed=0):
            C3 = C.reshape(batch["X_pad"].shape[0], batch["X_pad"].shape[1], self.K)

            def _apply_shuffle(C3, batch, *, shuffle, indices, records, seed):
                if not shuffle:
                    return C3
                if indices is None or records is None:
                    raise ValueError("shuffle requires indices and records")
                return V2.shuffle_assignments(C3, records, indices, seed, C.device)

            C3 = _apply_shuffle(C3, batch, shuffle=shuffle, indices=indices, records=records, seed=seed)
            return nm_graph_features(
                C3, batch["R_pad"], batch["valid"], batch["iu0"], batch["iu1"], with_moments=self.with_moments
            )

        def forward_padded(self, batch, *, shuffle=False, indices=None, records=None, seed=0):
            X_flat = batch["X_pad"].reshape(-1, self.F)
            C_flat = self.assign(X_flat)
            h = self.graph_features(
                C_flat, batch, shuffle=shuffle, indices=indices, records=records, seed=seed
            )
            pred = self.head(h).reshape(-1)
            return pred, C_flat, h

        model.graph_features = MethodType(graph_features, model)
        model.forward_padded = MethodType(forward_padded, model)
        return model


def build_nm_model(device: str, *, with_moments: bool = True, seed: int = 0):
    layout = V2.frozen_layout()
    init = V2.load_frozen_encoder_init()
    model = NMPrototypeModelFactory.build(
        layout.feature_dim,
        D_LOCAL,
        K_PROTO,
        N_REL,
        init,
        with_moments=with_moments,
        proto_seed=PROTO_INIT_SEED,
        seed=seed,
    ).to(device)
    return model


def train_stage_b(
    model,
    records_train,
    records_dev,
    train_indices,
    dev_indices,
    device: str,
    *,
    seed: int = 0,
    max_epochs: int = T.MAX_EPOCHS,
    patience: int = T.PATIENCE,
    batch: int = T.BATCH,
    log=print,
):
    """Exact TCCD-v2 training protocol on the NM representation."""
    return V2.train_model(
        model,
        records_train,
        records_dev,
        list(train_indices),
        list(dev_indices),
        device,
        seed=seed,
        max_epochs=max_epochs,
        patience=patience,
        batch=batch,
        log=log,
    )


def evaluate_model(model, records, indices, device: str, *, batch: int = 64) -> float:
    return V2.evaluate_mae(model, records, indices, device, batch=batch)


def _state_checksum(model) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype=np.float32).tobytes())
    return digest.hexdigest()


# ===========================================================================
# Stage-A reader training (exact TCCD-v5 frozen-reader protocol)
# ===========================================================================
@dataclass
class ReaderResult:
    best_valid: float
    best_epoch: int
    soup_valid: float | None
    soup_members: list[int]
    init_checksum: str
    feature_dim: int
    state_best: Any = None
    state_soup: Any = None
    train_history: list[dict[str, Any]] = field(default_factory=list)
    wall_s: float = 0.0


def train_linear_reader(
    X: np.ndarray,
    y: np.ndarray,
    train_indices: Sequence[int],
    dev_indices: Sequence[int],
    device: str,
    *,
    seed: int = 0,
    max_epochs: int = T.MAX_EPOCHS,
    patience: int = T.PATIENCE,
    batch: int = T.BATCH,
    lr: float = T.LR,
    wd: float = T.WD,
    clip: float = T.CLIP,
    log=print,
) -> ReaderResult:
    """Byte-identical protocol to ``tccd_v5.train_frozen_base_reader``."""
    torch = _torch()
    nn = torch.nn
    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed) + 91011)
    tr = np.asarray(train_indices, dtype=np.int64)
    dv = np.asarray(dev_indices, dtype=np.int64)
    xtr = torch.as_tensor(np.asarray(X[tr], dtype=np.float32), device=device)
    ytr = torch.as_tensor(np.asarray(y[tr], dtype=np.float32), device=device)
    xdv = torch.as_tensor(np.asarray(X[dv], dtype=np.float32), device=device)
    ydv = torch.as_tensor(np.asarray(y[dv], dtype=np.float32), device=device)
    model = nn.Linear(int(xtr.shape[1]), 1).to(device)
    init_checksum = _state_checksum(model)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best = math.inf
    best_epoch = -1
    best_state = None
    top: list[tuple[float, int, dict[str, Any]]] = []
    history: list[dict[str, Any]] = []
    stale = 0
    started = time.time()
    for epoch in range(1, int(max_epochs) + 1):
        model.train()
        order = rng.permutation(len(xtr))
        total = 0.0
        count = 0
        for start in range(0, len(order), int(batch)):
            sel = order[start : start + int(batch)]
            opt.zero_grad(set_to_none=True)
            loss = (model(xtr[sel]).reshape(-1) - ytr[sel]).abs().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            total += float(loss.detach()) * len(sel)
            count += len(sel)
        model.eval()
        with torch.no_grad():
            valid_mae = float((model(xdv).reshape(-1) - ydv).abs().mean())
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append({"epoch": int(epoch), "train_loss": total / max(count, 1), "valid": valid_mae})
        if epoch == 1 or epoch % 20 == 0 or epoch == int(max_epochs):
            log(f"[stageA] epoch={epoch:03d} train={history[-1]['train_loss']:.6f} valid={valid_mae:.6f} best={best:.6f}@{best_epoch}")
        if valid_mae < best - 1e-9:
            best = valid_mae
            best_epoch = epoch
            best_state = copy.deepcopy(state)
            stale = 0
        else:
            stale += 1
        if len(top) < T.TOP_K_SOUP or valid_mae < max(item[0] for item in top):
            top.append((valid_mae, epoch, state))
            top.sort(key=lambda item: (item[0], item[1]))
            top = top[: T.TOP_K_SOUP]
        if stale >= int(patience):
            log(f"[stageA] early stop at epoch {epoch}")
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
        model.eval()
        with torch.no_grad():
            soup_mae = float((model(xdv).reshape(-1) - ydv).abs().mean())
        model.load_state_dict(backup)
    return ReaderResult(
        best_valid=float(best),
        best_epoch=int(best_epoch),
        soup_valid=None if soup_mae is None else float(soup_mae),
        soup_members=members,
        init_checksum=init_checksum,
        feature_dim=int(xtr.shape[1]),
        state_best=best_state,
        state_soup=soup,
        train_history=history,
        wall_s=time.time() - started,
    )


# ===========================================================================
# decisions (frozen)
# ===========================================================================
def stage_a_decision(
    *,
    raw_soup: float = RAW_SOUP_MAE,
    raw_rescreen_soup: float,
    norm_soup: float,
    nm_soup: float,
    raw_best: float = RAW_BEST_MAE,
    raw_rescreen_best: float | None = None,
    norm_best: float | None = None,
    nm_best: float | None = None,
) -> dict[str, Any]:
    delta_norm = float(raw_soup) - float(nm_soup)
    delta_norm_rerun = float(raw_rescreen_soup) - float(nm_soup)
    delta_mom = float(norm_soup) - float(nm_soup)
    drift = abs(float(raw_rescreen_soup) - float(raw_soup))
    abs_ok = float(nm_soup) <= STAGE_A_NM_ABS_MAX
    gain_ok = delta_norm >= STAGE_A_DELTA_NORM_MIN
    rerun_ok = delta_norm_rerun >= STAGE_A_DELTA_NORM_MIN
    drift_ok = drift <= STAGE_A_PROTOCOL_DRIFT_MAX
    passed = bool(abs_ok and gain_ok and rerun_ok and drift_ok)
    if passed:
        verdict = "PASS"
    elif not drift_ok:
        verdict = "PROTOCOL_DRIFT"
    else:
        verdict = "FAIL"
    variance_band = VARIANCE_MATERIAL if delta_mom >= DELTA_MOM_MATERIAL else VARIANCE_NEGLIGIBLE
    out = {
        "verdict": verdict,
        "pass": passed,
        "primary_metric": "top5_soup",
        "raw_soup": float(raw_soup),
        "raw_rescreen_soup": float(raw_rescreen_soup),
        "norm_soup": float(norm_soup),
        "nm_soup": float(nm_soup),
        "raw_best": float(raw_best),
        "raw_rescreen_best": None if raw_rescreen_best is None else float(raw_rescreen_best),
        "norm_best": None if norm_best is None else float(norm_best),
        "nm_best": None if nm_best is None else float(nm_best),
        "delta_norm": delta_norm,
        "delta_norm_rerun": delta_norm_rerun,
        "delta_mom": delta_mom,
        "protocol_drift": drift,
        "checks": {
            "nm_soup_abs_ok": abs_ok,
            "delta_norm_ok": gain_ok,
            "delta_norm_rerun_ok": rerun_ok,
            "protocol_drift_ok": drift_ok,
        },
        "variance_diagnostic": variance_band,
        "variance_material": bool(variance_band == VARIANCE_MATERIAL),
        "thresholds": {
            "nm_soup_abs_max": STAGE_A_NM_ABS_MAX,
            "delta_norm_min": STAGE_A_DELTA_NORM_MIN,
            "protocol_drift_max": STAGE_A_PROTOCOL_DRIFT_MAX,
            "delta_mom_material": DELTA_MOM_MATERIAL,
        },
        "stage_b_authorized": passed,
        "official_valid_authorized": False,
    }
    return out


def stage_b_decision(nm_soup: float) -> dict[str, Any]:
    mae = float(nm_soup)
    if mae > STAGE_B_STOP_GT:
        case = "S"
    elif mae > STAGE_B_PARTIAL_GT:
        case = "P"
    else:
        case = "G"
    return {
        "case": case,
        "nm_soup": mae,
        "stop": bool(case == "S"),
        "official_valid_authorized": bool(case in ("P", "G")),
        "matched_reference": {
            "v2_internal_best": V2_INTERNAL_BEST_MAE,
            "v2_internal_soup": V2_INTERNAL_SOUP_MAE,
        },
        "delta_vs_v2_soup": float(V2_INTERNAL_SOUP_MAE) - mae,
        "thresholds": {
            "stop_gt": STAGE_B_STOP_GT,
            "partial_gt": STAGE_B_PARTIAL_GT,
        },
    }


def official_band(mae: float) -> str:
    mae = float(mae)
    if mae <= OFFICIAL_COMPETITIVE_MAX:
        return "COMPETITIVE_ISH"
    if mae <= OFFICIAL_INSUFFICIENT_GT:
        return "SUBSTANTIAL_BUT_INCOMPLETE"
    return "INSUFFICIENT"


# ===========================================================================
# Gate 0
# ===========================================================================
def synthetic_graph(n: int, seed: int, *, zero_relation: int | None = None, scale: float = 1.0):
    """Synthetic (C, R) with the real nonnegative TCCD relation conventions.

    ``R_int`` is a zero-diagonal incidence overlap ``B B^T``, the three native
    bond operators are symmetric 0/1 indicators with a zero diagonal, and
    ``R_geo`` is a positive distance-kernel matrix with a unit diagonal, so all
    relation masses ``s_r`` are nonnegative exactly as on real ZINC graphs.
    """
    rng = np.random.default_rng(int(seed))
    logits = (scale * rng.standard_normal((n, K_PROTO))).astype(np.float32)
    C = torch_softmax_np(logits)
    R = np.zeros((N_REL, n, n), dtype=np.float32)
    # R_int = B B^T with a removed diagonal
    B = (rng.random((n, max(2, n // 3 + 1))) < 0.3).astype(np.float32)
    R[0] = B @ B.T
    # three native bond operators: symmetric 0/1 indicators, zero diagonal
    for r in (1, 2, 3):
        A = np.triu((rng.random((n, n)) < 0.25).astype(np.float32), 1)
        R[r] = A + A.T
    # R_geo: positive kernel with an exact unit diagonal
    G = np.triu((rng.random((n, n)) < 0.5).astype(np.float32), 1)
    G = G + G.T
    R[4] = 0.3 + 0.4 * rng.random((n, n)).astype(np.float32)
    R[4] = 0.5 * (R[4] + R[4].T) * G + np.eye(n, dtype=np.float32)
    for r in range(N_REL):
        np.fill_diagonal(R[r], float(RELATION_DIAGONAL_PER_OCCURRENCE[r]))
    if zero_relation is not None:
        R[int(zero_relation)] = np.zeros((n, n), dtype=np.float32)
    return C, R


def torch_softmax_np(logits: np.ndarray) -> np.ndarray:
    x = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(x)
    return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)


def _tensor_batch(C_list: Sequence[np.ndarray], R_list: Sequence[np.ndarray], device: str):
    torch = _torch()
    B = len(C_list)
    N = max(c.shape[0] for c in C_list)
    C3 = np.zeros((B, N, K_PROTO), dtype=np.float32)
    R_pad = np.zeros((B, N_REL, N, N), dtype=np.float32)
    valid = np.zeros((B, N), dtype=np.bool_)
    for b, (C, R) in enumerate(zip(C_list, R_list)):
        n = C.shape[0]
        C3[b, :n] = C
        R_pad[b, :, :n, :n] = R
        valid[b, :n] = True
    iu0, iu1 = T.sym_indices(K_PROTO)
    return {
        "C3": torch.as_tensor(C3, device=device),
        "R_pad": torch.as_tensor(R_pad, device=device),
        "valid": torch.as_tensor(valid, device=device),
        "iu0": torch.as_tensor(iu0, dtype=torch.long, device=device),
        "iu1": torch.as_tensor(iu1, dtype=torch.long, device=device),
    }


def reference_features_np(C: np.ndarray, R: np.ndarray, *, with_moments: bool) -> np.ndarray:
    """Independent float64 numpy implementation of the registered representation."""
    Cn = np.asarray(C, dtype=np.float64)
    Rn = np.asarray(R, dtype=np.float64)
    n = Cn.shape[0]
    mu = Cn.sum(axis=0) / n
    s = Rn.sum(axis=(1, 2))
    iu0, iu1 = T.sym_indices(K_PROTO)
    parts: list[np.ndarray] = [mu]
    if with_moments:
        m2 = (Cn * Cn).sum(axis=0) / n
        parts.append(np.maximum(m2 - mu * mu, 0.0))
    blocks = []
    for r in range(N_REL):
        M = Cn.T @ Rn[r] @ Cn
        Mhat = M / max(float(s[r]), EPS)
        blocks.append(Mhat[iu0, iu1])
    parts.append(np.concatenate(blocks))
    parts.append(np.asarray([math.log1p(n)], dtype=np.float64))
    parts.append(np.log1p(s))
    return np.concatenate(parts)


def full_symmetric_from_upper(vec: np.ndarray) -> np.ndarray:
    """Rebuild a full symmetric matrix from the registered upper-triangle storage."""
    iu0, iu1 = T.sym_indices(K_PROTO)
    M = np.zeros((K_PROTO, K_PROTO), dtype=np.float64)
    M[iu0, iu1] = vec
    M[iu1, iu0] = vec
    np.fill_diagonal(M, np.asarray(vec, dtype=np.float64)[iu0 == iu1])
    return M


def tensor_inputs_from_cache(cache: V6.LabelFreeCache, rows: Sequence[int]):
    """Rebuild exact (C, R) tensors for cached graphs from the frozen cache."""
    C_list, R_list = [], []
    offs = np.asarray(cache.c_offsets, dtype=np.int64)
    poffs = np.asarray(cache.pair_offsets, dtype=np.int64)
    diag = np.asarray(RELATION_DIAGONAL_PER_OCCURRENCE, dtype=np.float32)
    for r in rows:
        r = int(r)
        a, b = int(offs[r]), int(offs[r + 1])
        C = cache.C[a:b].astype(np.float32)
        n = C.shape[0]
        R = np.zeros((N_REL, n, n), dtype=np.float32)
        p, q = int(poffs[r]), int(poffs[r + 1])
        if q > p:
            i0 = cache.pair_i0[p:q]
            i1 = cache.pair_i1[p:q]
            vals = cache.pair_rel[p:q].T  # [N_REL, npairs]
            R[:, i0, i1] = vals
            R[:, i1, i0] = vals
        for k in range(N_REL):
            np.fill_diagonal(R[k], diag[k])
        C_list.append(C)
        R_list.append(R)
    return C_list, R_list


def gate0_data_free_checks(device: str = "cpu") -> dict[str, Any]:
    torch = _torch()
    checks: dict[str, Any] = {"official_test_loaded": False, "target_not_read": True}

    checks["dims"] = {
        "raw": RAW_DIM,
        "norm": NORM_DIM,
        "norm_mom": NORM_MOM_DIM,
        "tri": TRI,
    }
    checks["dims_ok"] = bool(
        RAW_DIM == T.h_dim(K_PROTO, N_REL)
        and NORM_DIM == MEAN_DIM + REL_DIM + SCALE_DIM
        and NORM_MOM_DIM == MEAN_DIM + MOM_DIM + REL_DIM + SCALE_DIM
        and REL_DIM == N_REL * TRI
    )
    checks["feature_layout_norm"] = feature_layout(False)
    checks["feature_layout_norm_mom"] = feature_layout(True)
    checks["reader_parameters"] = {arm: reader_parameter_count(ARM_WITH_MOMENTS.get(arm, True)) for arm in ARMS_STAGE_A}
    checks["stage_b_parameter_accounting"] = parameter_accounting(True)

    # ---- A. permutation invariance (both variants) ----
    C, R = synthetic_graph(9, seed=7, scale=1.5)
    perm = np.asarray([4, 0, 8, 1, 7, 2, 6, 3, 5], dtype=np.int64)
    b0 = _tensor_batch([C], [R], device)
    b1 = _tensor_batch([C[perm]], [R[:, perm][:, :, perm]], device)
    perm_report: dict[str, float] = {}
    for with_moments in (False, True):
        h0 = nm_graph_features(b0["C3"], b0["R_pad"], b0["valid"], b0["iu0"], b0["iu1"], with_moments=with_moments)
        h1 = nm_graph_features(b1["C3"], b1["R_pad"], b1["valid"], b1["iu0"], b1["iu1"], with_moments=with_moments)
        scale = float(h0.abs().max())
        diff = float((h0 - h1).abs().max())
        perm_report["norm_mom" if with_moments else "norm"] = {"max_abs_diff": diff, "scale": scale, "rel": diff / max(scale, 1e-12)}
    checks["permutation_representation"] = perm_report
    checks["permutation_invariant"] = bool(all(v["rel"] <= 1e-5 for v in perm_report.values()))

    # component-wise invariance: mu, v, s, Mhat
    C3 = b0["C3"] * b0["valid"].unsqueeze(-1).to(b0["C3"].dtype)
    mu0 = C3.sum(dim=1) / b0["valid"].sum(dim=1, keepdim=True)
    m20 = (C3 * C3).sum(dim=1) / b0["valid"].sum(dim=1, keepdim=True)
    v0 = (m20 - mu0 * mu0).clamp_min(0.0)
    s0 = b0["R_pad"].sum(dim=(2, 3))
    M0 = torch.einsum("bnk,brnm,bml->brkl", C3, b0["R_pad"], C3) / s0.clamp_min(EPS)[:, :, None, None]
    C3p = b1["C3"] * b1["valid"].unsqueeze(-1).to(b1["C3"].dtype)
    mu1 = C3p.sum(dim=1) / b1["valid"].sum(dim=1, keepdim=True)
    m21 = (C3p * C3p).sum(dim=1) / b1["valid"].sum(dim=1, keepdim=True)
    v1 = (m21 - mu1 * mu1).clamp_min(0.0)
    s1 = b1["R_pad"].sum(dim=(2, 3))
    M1 = torch.einsum("bnk,brnm,bml->brkl", C3p, b1["R_pad"], C3p) / s1.clamp_min(EPS)[:, :, None, None]
    comp = {
        "mu": float((mu0 - mu1).abs().max()),
        "v": float((v0 - v1).abs().max()),
        "s": float((s0 - s1).abs().max()),
        "Mhat": float((M0 - M1).abs().max()),
    }
    checks["permutation_components"] = comp
    checks["permutation_components_invariant"] = bool(all(v <= 1e-5 for v in comp.values()))

    # ---- reference-implementation agreement (float64 numpy vs tensor path) ----
    ref_agreement: dict[str, Any] = {}
    for with_moments in (False, True):
        h = nm_graph_features(b0["C3"], b0["R_pad"], b0["valid"], b0["iu0"], b0["iu1"], with_moments=with_moments)
        ref = reference_features_np(C, R, with_moments=with_moments)
        h_np = h[0].detach().cpu().numpy().astype(np.float64)
        scale = max(float(np.abs(ref).max()), 1e-9)
        key = "norm_mom" if with_moments else "norm"
        ref_agreement[key] = {
            "max_abs_diff": float(np.abs(h_np - ref).max()),
            "rel": float(np.abs(h_np - ref).max() / scale),
            "dim": int(ref.size),
        }
    checks["reference_implementation"] = ref_agreement
    checks["reference_implementation_ok"] = bool(
        all(v["rel"] <= 1e-6 for v in ref_agreement.values())
    )

    # ---- B. batching invariance ----
    C_list, R_list = [], []
    for i in range(4):
        Ci, Ri = synthetic_graph(5 + 2 * i, seed=100 + i, scale=1.0)
        C_list.append(Ci)
        R_list.append(Ri)
    batched = _tensor_batch(C_list, R_list, device)
    h_batched = nm_graph_features(
        batched["C3"], batched["R_pad"], batched["valid"], batched["iu0"], batched["iu1"], with_moments=True
    )
    alone = []
    for Ci, Ri in zip(C_list, R_list):
        b = _tensor_batch([Ci], [Ri], device)
        alone.append(
            nm_graph_features(b["C3"], b["R_pad"], b["valid"], b["iu0"], b["iu1"], with_moments=True)[0]
        )
    h_alone = torch.stack(alone, dim=0)
    checks["batching_invariance_max_abs_diff"] = float((h_batched - h_alone).abs().max())
    checks["batching_invariant"] = bool(checks["batching_invariance_max_abs_diff"] <= 1e-5)

    # padded rows contribute exactly zero: pad one graph to a larger N
    Ci, Ri = C_list[0], R_list[0]
    n_i = Ci.shape[0]
    C_pad = np.zeros((1, n_i + 3, K_PROTO), dtype=np.float32)
    C_pad[0, :n_i] = Ci
    R_pad = np.zeros((1, N_REL, n_i + 3, n_i + 3), dtype=np.float32)
    R_pad[0, :, :n_i, :n_i] = Ri
    valid = np.zeros((1, n_i + 3), dtype=np.bool_)
    valid[0, :n_i] = True
    iu0, iu1 = T.sym_indices(K_PROTO)
    b_pad = {
        "C3": torch.as_tensor(C_pad, device=device),
        "R_pad": torch.as_tensor(R_pad, device=device),
        "valid": torch.as_tensor(valid, device=device),
        "iu0": torch.as_tensor(iu0, dtype=torch.long, device=device),
        "iu1": torch.as_tensor(iu1, dtype=torch.long, device=device),
    }
    h_pad = nm_graph_features(b_pad["C3"], b_pad["R_pad"], b_pad["valid"], b_pad["iu0"], b_pad["iu1"], with_moments=True)
    checks["padding_invariance_max_abs_diff"] = float((h_pad[0] - alone[0]).abs().max())
    checks["padding_invariant"] = bool(checks["padding_invariance_max_abs_diff"] <= 1e-5)

    # ---- C. algebra ----
    checks["mu_row_sum_max_abs_dev"] = float((mu0.sum(dim=1) - 1.0).abs().max())
    checks["mu_row_sum_ok"] = bool(checks["mu_row_sum_max_abs_dev"] <= 1e-6)
    m2_ref = (C.astype(np.float64) ** 2).sum(axis=0) / C.shape[0]
    mu_ref = C.astype(np.float64).sum(axis=0) / C.shape[0]
    v_ref = np.maximum(m2_ref - mu_ref**2, 0.0)
    h_nm = nm_graph_features(b0["C3"], b0["R_pad"], b0["valid"], b0["iu0"], b0["iu1"], with_moments=True)
    v_block = h_nm[0, MEAN_DIM : MEAN_DIM + MOM_DIM].detach().cpu().numpy().astype(np.float64)
    checks["variance_identity_max_abs_diff"] = float(np.abs(v_block - v_ref).max())
    checks["variance_identity_ok"] = bool(checks["variance_identity_max_abs_diff"] <= 1e-6)
    checks["variance_nonnegative"] = bool(float(v_block.min()) >= 0.0)
    # explicit clamp path on a planted negative pre-clamp variance
    planted = np.asarray([-1.5, 0.0, -1e-12], dtype=np.float64)
    checks["variance_clamp_path_ok"] = bool(float(np.maximum(planted, 0.0).min()) == 0.0)

    # ---- D. normalized composition mass (full symmetric rebuild) ----
    mass_report = {}
    for with_moments in (False, True):
        h = h_nm if with_moments else nm_graph_features(
            b0["C3"], b0["R_pad"], b0["valid"], b0["iu0"], b0["iu1"], with_moments=False
        )
        start = (MEAN_DIM + (MOM_DIM if with_moments else 0))
        vecs = h[0, start : start + REL_DIM].detach().cpu().numpy()
        s_np = s0[0].detach().cpu().numpy()
        totals = []
        for r in range(N_REL):
            M = full_symmetric_from_upper(vecs[r * TRI : (r + 1) * TRI])
            totals.append(float(M.sum()))
        mass_report["norm_mom" if with_moments else "norm"] = {
            "s_r": s_np.tolist(),
            "symmetric_total_per_relation": totals,
            "max_abs_dev_from_one": float(max(abs(t - 1.0) for t in totals)),
        }
    checks["normalized_mass"] = mass_report
    checks["normalized_mass_ok"] = bool(
        all(v["max_abs_dev_from_one"] <= 1e-3 for v in mass_report.values())
    )

    # ---- E. zero-mass relation ----
    Cz, Rz = synthetic_graph(8, seed=11, zero_relation=3, scale=1.0)
    bz = _tensor_batch([Cz], [Rz], device)
    hz = nm_graph_features(bz["C3"], bz["R_pad"], bz["valid"], bz["iu0"], bz["iu1"], with_moments=True)
    vecs = hz[0, MEAN_DIM + MOM_DIM : MEAN_DIM + MOM_DIM + REL_DIM].detach().cpu().numpy()
    zero_block = vecs[3 * TRI : 4 * TRI]
    checks["zero_mass_relation_abs_max"] = float(np.abs(zero_block).max())
    checks["zero_mass_relation_exact_zero"] = bool(float(np.abs(zero_block).max()) == 0.0)
    checks["zero_mass_representation_finite"] = bool(torch.isfinite(hz).all())

    # ---- F. no target leakage ----
    f_a = nm_graph_features(b0["C3"], b0["R_pad"], b0["valid"], b0["iu0"], b0["iu1"], with_moments=True)
    f_b = nm_graph_features(b0["C3"], b0["R_pad"], b0["valid"], b0["iu0"], b0["iu1"], with_moments=True)
    checks["target_free_max_abs_diff"] = float((f_a - f_b).abs().max())
    checks["target_free"] = bool(checks["target_free_max_abs_diff"] == 0.0)
    import inspect

    nm_params = list(inspect.signature(nm_graph_features).parameters)
    sa_params = list(inspect.signature(stage_a_features).parameters)
    checks["nm_feature_builder_signature"] = nm_params
    checks["stage_a_feature_builder_signature"] = sa_params
    bad = {"y", "target", "label", "labels", "targets"}
    checks["feature_builder_signature_target_free"] = bool(
        not (set(nm_params) & bad) and not (set(sa_params) & bad)
    )

    bool_keys = [
        k
        for k, v in checks.items()
        if isinstance(v, bool) and k not in {"official_test_loaded", "target_not_read"}
    ]
    checks["data_free_all_pass"] = bool(all(checks[k] for k in bool_keys))
    return checks


def gate0_real_cache_checks(cache: V6.LabelFreeCache, device: str = "cpu", *, n_subset: int = 24) -> dict[str, Any]:
    checks: dict[str, Any] = {"official_test_loaded": False}

    n = node_counts_from_cache(cache)
    s = relation_masses_from_cache(cache)
    checks["n_graphs"] = int(cache.n_graphs)
    checks["node_count_min"] = float(n.min())
    checks["node_count_max"] = float(n.max())
    checks["relation_mass_zero_counts"] = (s == 0).sum(axis=0).astype(int).tolist()
    checks["relation_mass_min_per_relation"] = s.min(axis=0).tolist()

    # numerator (from h_base) vs denominator (from the frozen pair cache)
    base = cache.base.astype(np.float64)
    M = base[:, K_PROTO:].reshape(int(cache.n_graphs), N_REL, TRI)
    iu0, iu1 = T.sym_indices(K_PROTO)
    diag_pos = np.where(iu0 == iu1)[0]
    s_up = M.sum(axis=2)
    diag_in_base = M[:, :, diag_pos].sum(axis=2)
    s_from_base = 2.0 * s_up - diag_in_base
    rel_dev = np.abs(s_from_base - s) / np.maximum(np.abs(s), 1.0)
    checks["relation_mass_from_base_vs_cache_max_rel_dev"] = float(rel_dev.max())
    checks["relation_mass_source_agreement"] = bool(float(rel_dev.max()) <= 1e-4)

    # zero-mass path on real graphs: Mhat must be exactly zero
    Mhat = M / np.maximum(s, EPS)[:, :, None]
    zero_mask = s == 0
    checks["real_zero_mass_entries"] = int(zero_mask.sum())
    checks["real_zero_mass_absolute_max"] = float(np.abs(Mhat[zero_mask]).max()) if zero_mask.any() else 0.0
    checks["real_zero_mass_exact"] = bool(
        (not zero_mask.any()) or float(np.abs(Mhat[zero_mask]).max()) == 0.0
    )

    # normalized mass on real graphs with s_r > 0 (full symmetric rebuild)
    totals = []
    for r in range(N_REL):
        sel = np.where(s[:, r] > 0)[0][:200]
        for g in sel:
            Mfull = full_symmetric_from_upper(Mhat[g, r])
            totals.append(float(Mfull.sum()))
    totals_arr = np.asarray(totals)
    checks["real_normalized_mass_count"] = int(totals_arr.size)
    checks["real_normalized_mass_max_abs_dev"] = float(np.abs(totals_arr - 1.0).max())
    checks["real_normalized_mass_ok"] = bool(float(np.abs(totals_arr - 1.0).max()) <= 1e-3)

    # mu row sums from the frozen base
    mu = base[:, :K_PROTO] / n[:, None]
    checks["real_mu_row_sum_max_abs_dev"] = float(np.abs(mu.sum(axis=1) - 1.0).max())
    checks["real_mu_row_sum_ok"] = bool(checks["real_mu_row_sum_max_abs_dev"] <= 1e-5)

    # equivalence: cache feature builder vs tensor feature builder on frozen assignments
    rows = list(range(min(int(n_subset), cache.n_graphs)))
    C_list, R_list = tensor_inputs_from_cache(cache, rows)
    b = _tensor_batch(C_list, R_list, device)
    for with_moments in (False, True):
        h = nm_graph_features(b["C3"], b["R_pad"], b["valid"], b["iu0"], b["iu1"], with_moments=with_moments)
        h_np = h.detach().cpu().numpy()
        feats, _ = stage_a_features(cache, with_moments=with_moments, rows=rows)
        diff = np.abs(h_np - feats)
        scale = max(float(np.abs(h_np).max()), 1e-6)
        checks[f"cache_vs_tensor_{'norm_mom' if with_moments else 'norm'}_max_abs_diff"] = float(diff.max())
        checks[f"cache_vs_tensor_{'norm_mom' if with_moments else 'norm'}_rel"] = float(diff.max() / scale)
        checks[f"cache_vs_tensor_{'norm_mom' if with_moments else 'norm'}_ok"] = bool(diff.max() <= 1e-4)

    # RAW feature path must be exactly the frozen base
    raw = raw_features(cache, rows)
    checks["raw_matches_base_exact"] = bool(np.array_equal(raw, cache.base[rows].astype(np.float32)))
    checks["raw_dim"] = int(raw.shape[1])

    real_bool = [k for k, v in checks.items() if isinstance(v, bool) and k != "official_test_loaded"]
    checks["real_all_pass"] = bool(all(checks[k] for k in real_bool))
    return checks
