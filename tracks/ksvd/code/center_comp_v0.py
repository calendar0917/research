#!/usr/bin/env python
"""CENTER-COMP-v0: center-preserving composition diagnostic.

Frozen TCCD-v2 local codes are reused.  The only new operator is an explicit
incidence-preserving composition:

    e_ij     = phi(p_ij)                (shared pair embedding)
    q_i      = sum_{j != i} e_ij        (center pair context)
    u_i      = rho([c_i, q_i])          (center nonlinear composition)
    h_center = sum_i u_i                (global center pool)
    h        = [h_base, h_center] -> Linear(10480,1)

Every unordered pair contributes e_ij to both endpoint centers.  The local
encoder W, prototypes P, and temperature are frozen and never instantiated as
trainable parameters.  The TCCD-v5 pair cache is reused byte-for-byte.

The evaluation-only CENTER-BIND-SHUFFLE intervention replaces q_i by
q_{pi(i)} with c_i fixed, destroying only the center<->context binding.

Official ZINC valid/test are never loaded.  Formal execution is GPU1-only.
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
from tracks.ksvd.code import tccd_v6 as V6

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = REPO_ROOT / "tracks/ksvd/results/center_comp_v0"
PROTOCOL_VERSION = "center_comp_v0_center_preserving_composition_v1"

D_LOCAL = V5.D_LOCAL
K_PROTO = V5.K_PROTO
N_REL = V5.N_REL
PHI_HIDDEN = V5.PHI_HIDDEN
PHI_OUT = V5.PHI_OUT
RHO_HIDDEN = 64
RHO_OUT = PHI_OUT

PAIR_WIDTH = 3 * K_PROTO + N_REL  # 197
RHO_IN = K_PROTO + PHI_OUT  # 80
FROZEN_BASE_DIM = V5.frozen_base_dim()  # 10464

PERM_SEED = 20260922
INIT_SEED_OFFSET = 33001

# ---------------------------------------------------------------------------
# frozen references (read from formal artifacts, never re-run)
# ---------------------------------------------------------------------------
V2_BEST_CHECKPOINT = V5.V2_BEST_CHECKPOINT
V2_BEST_CHECKPOINT_SHA256 = V5.V2_BEST_CHECKPOINT_SHA256
V5_STAGE_A_JSON = V5.RESULTS_DIR / "stageA_seed0.json"
V6_STAGE_A_JSON = V6.RESULTS_DIR / "stageA_seed0.json"

TCCD_V2_BASE_SOUP = 0.2783639132976532
V5_POST_SOUP = 0.2522333264350891
V5_PRE_SOUP = 0.2514181435108185
V5_PRE_SHUFFLE_SOUP = 0.25175556540489197
CANONICAL_GPU1_BASELINE = 0.119818

# ---------------------------------------------------------------------------
# preregistered thresholds
# ---------------------------------------------------------------------------
MATERIAL = 0.010
BIND_FAIL = 0.005
MATERIAL_CHANGE = 0.010
MATERIAL_CHANGE_STRONG = 0.050
MATERIAL_CHANGE_VERY_STRONG = 0.100


def _torch():
    return T._torch()


# ===========================================================================
# helpers
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


def phi_parameter_count(d_p: int = PAIR_WIDTH) -> int:
    return int(d_p * PHI_HIDDEN + PHI_HIDDEN + PHI_HIDDEN * PHI_OUT + PHI_OUT)


def rho_parameter_count(d_in: int = RHO_IN) -> int:
    return int(d_in * RHO_HIDDEN + RHO_HIDDEN + RHO_HIDDEN * RHO_OUT + RHO_OUT)


def head_parameter_count() -> int:
    return int(FROZEN_BASE_DIM + PHI_OUT + 1)


def parameter_accounting() -> dict[str, int]:
    phi = phi_parameter_count()
    rho = rho_parameter_count()
    head = head_parameter_count()
    return {"phi": phi, "rho": rho, "head": head, "total": phi + rho + head}


def frozen_reference_values() -> dict[str, Any]:
    """Read exact frozen references from the formal TCCD-v5/v6 artifacts."""
    if not V5_STAGE_A_JSON.exists():
        raise FileNotFoundError(f"missing TCCD-v5 Stage-A artifact: {V5_STAGE_A_JSON}")
    v5 = json.loads(V5_STAGE_A_JSON.read_text(encoding="utf-8"))
    if v5.get("official_test_loaded") is not False:
        raise RuntimeError("TCCD-v5 artifact does not certify official_test_loaded == false")
    res = v5["results"]
    base_soup = float(res["BASE"]["soup_valid_mae"])
    post_soup = float(res["POST"]["soup_valid_mae"])
    pre_soup = float(res["PRE"]["soup_valid_mae"])
    pre_shuffle_soup = float(res["PRE_SHUFFLE"]["soup_valid_mae"])
    return {
        "source": str(V5_STAGE_A_JSON.relative_to(REPO_ROOT)),
        "base_soup": base_soup,
        "post_soup": post_soup,
        "pre_soup": pre_soup,
        "pre_shuffle_soup": pre_shuffle_soup,
        "mae_prev": min(pre_soup, post_soup),
        "canonical_gpu1_baseline": CANONICAL_GPU1_BASELINE,
        "official_test_loaded": False,
    }


# ===========================================================================
# frozen local checkpoint (used only for Gate-0 C re-derivation)
# ===========================================================================
def build_frozen_v2_model(device: str):
    """Exact frozen TCCD-v2 model; SHA verified; all parameters frozen."""
    return V5.build_frozen_v2_model(device)


# ===========================================================================
# batch
# ===========================================================================
@dataclass
class CenterBatch:
    base: Any
    C3: Any
    i0: Any
    i1: Any
    rel: Any
    valid: Any
    occ_valid: Any
    center_perm: Any
    y: Any


def make_center_batch(
    cache: V6.LabelFreeCache,
    y: np.ndarray,
    indices: Sequence[int],
    device: str,
    *,
    shuffle: bool = False,
    perm_seed: int = PERM_SEED,
) -> CenterBatch:
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
    occ_valid = np.zeros((B, N), dtype=np.bool_)
    center_perm = np.zeros((B, N), dtype=np.int64)
    for bi, (r, gi) in enumerate(zip(rows, indices)):
        r = int(r)
        n = int(ns[bi])
        occ_valid[bi, :n] = True
        center_perm[bi, :n] = np.arange(n, dtype=np.int64)
        c0, c1 = int(cache.c_offsets[r]), int(cache.c_offsets[r + 1])
        C_np[bi, :n] = cache.C[c0:c1]
        if shuffle:
            perm = V5.fixed_permutation(n, int(gi), int(perm_seed))
            # q used at center i is q_{perm[i]}
            center_perm[bi, :n] = np.asarray(perm, dtype=np.int64)
        p0, p1 = int(cache.pair_offsets[r]), int(cache.pair_offsets[r + 1])
        npair = p1 - p0
        if npair:
            i0[bi, :npair] = cache.pair_i0[p0:p1]
            i1[bi, :npair] = cache.pair_i1[p0:p1]
            rel[bi, :npair] = cache.pair_rel[p0:p1]
            valid[bi, :npair] = True

    return CenterBatch(
        base=torch.as_tensor(np.asarray(cache.base[rows], dtype=np.float32), device=device),
        C3=torch.as_tensor(C_np, device=device),
        i0=torch.as_tensor(i0, device=device),
        i1=torch.as_tensor(i1, device=device),
        rel=torch.as_tensor(rel, device=device),
        valid=torch.as_tensor(valid, device=device),
        occ_valid=torch.as_tensor(occ_valid, device=device),
        center_perm=torch.as_tensor(center_perm, device=device),
        y=torch.as_tensor(np.asarray(y[rows], dtype=np.float32), device=device),
    )


# ===========================================================================
# center-preserving composition primitives
# ===========================================================================
def pair_descriptor(C3, i0, i1, rel):
    """Exact TCCD-v5 pair descriptor; reused unchanged."""
    return V5.pair_descriptor(C3, i0, i1, rel)


def center_pair_context(C3, i0, i1, rel, valid, phi) -> tuple[Any, Any]:
    """Compute e_ij and q_i = sum_{j != i} e_ij.

    Returns ``(e, q)`` with ``e`` masked to invalid pairs and ``q`` of shape
    ``[B, N, PHI_OUT]``.  Every valid unordered pair contributes exactly once
    to each of its two endpoint centers.
    """
    p = pair_descriptor(C3, i0, i1, rel)
    e = phi(p)
    e = e * valid.unsqueeze(-1).to(e.dtype)
    B, N = C3.shape[0], C3.shape[1]
    q = e.new_zeros((B, N, PHI_OUT))
    idx = i0.unsqueeze(-1).expand(-1, -1, PHI_OUT)
    q.scatter_add_(1, idx, e)
    idx = i1.unsqueeze(-1).expand(-1, -1, PHI_OUT)
    q.scatter_add_(1, idx, e)
    return e, q


def compose_center(C3, q, center_perm, occ_valid, rho):
    """u_i = rho([c_i, q_{perm(i)}]); h_center = sum_i u_i."""
    torch = _torch()
    q_sh = torch.gather(q, 1, center_perm.unsqueeze(-1).expand(-1, -1, PHI_OUT))
    u = rho(torch.cat([C3, q_sh], dim=-1))
    return (u * occ_valid.unsqueeze(-1).to(u.dtype)).sum(dim=1)


# ===========================================================================
# model
# ===========================================================================
class CenterCompFactory:
    @staticmethod
    def build(*, seed: int = 0):
        torch = _torch()
        nn = torch.nn
        torch.manual_seed(int(seed) + INIT_SEED_OFFSET)

        class _M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.phi = nn.Sequential(
                    nn.Linear(PAIR_WIDTH, PHI_HIDDEN),
                    nn.ReLU(),
                    nn.Linear(PHI_HIDDEN, PHI_OUT),
                )
                self.rho = nn.Sequential(
                    nn.Linear(RHO_IN, RHO_HIDDEN),
                    nn.ReLU(),
                    nn.Linear(RHO_HIDDEN, RHO_OUT),
                )
                self.head = nn.Linear(FROZEN_BASE_DIM + PHI_OUT, 1)
                self.graph_feature_kind = "frozen_h_v2_plus_center_preserving_composition"

            def forward(self, batch: CenterBatch):
                _, q = center_pair_context(batch.C3, batch.i0, batch.i1, batch.rel, batch.valid, self.phi)
                h_center = compose_center(batch.C3, q, batch.center_perm, batch.occ_valid, self.rho)
                h = _torch().cat([batch.base, h_center], dim=1)
                return self.head(h).reshape(-1)

        return _M()


def model_parameter_count(model) -> dict[str, int]:
    return {
        "phi": int(sum(p.numel() for n, p in model.named_parameters() if n.startswith("phi."))),
        "rho": int(sum(p.numel() for n, p in model.named_parameters() if n.startswith("rho."))),
        "head": int(sum(p.numel() for n, p in model.named_parameters() if n.startswith("head."))),
        "total": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
    }


# ===========================================================================
# Stage A training
# ===========================================================================
@dataclass
class CenterResult:
    best_valid: float
    best_epoch: int
    soup_valid: float | None
    soup_members: list[int]
    init_checksum: str
    state_best: Any = None
    state_soup: Any = None
    train_history: list[dict[str, Any]] = field(default_factory=list)
    wall_s: float = 0.0


def evaluate(
    model,
    cache: V6.LabelFreeCache,
    y: np.ndarray,
    indices: Sequence[int],
    device: str,
    *,
    batch: int = 64,
    shuffle: bool = False,
) -> float:
    torch = _torch()
    model.eval()
    errs = []
    with torch.no_grad():
        for start in range(0, len(indices), int(batch)):
            chunk = list(indices[start : start + int(batch)])
            b = make_center_batch(cache, y, chunk, device, shuffle=shuffle)
            pred = model(b)
            errs.append((pred - b.y).abs().cpu().numpy())
    return float(np.concatenate(errs).mean())


def predict(
    model,
    cache: V6.LabelFreeCache,
    y: np.ndarray,
    indices: Sequence[int],
    device: str,
    *,
    batch: int = 64,
    shuffle: bool = False,
) -> np.ndarray:
    torch = _torch()
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(indices), int(batch)):
            chunk = list(indices[start : start + int(batch)])
            b = make_center_batch(cache, y, chunk, device, shuffle=shuffle)
            out.append(model(b).detach().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def train_center_comp(
    cache: V6.LabelFreeCache,
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
) -> CenterResult:
    torch = _torch()
    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed) + 91011)
    model = CenterCompFactory.build(seed=seed).to(device)
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
            b = make_center_batch(cache, y, sel, device)
            opt.zero_grad(set_to_none=True)
            pred = model(b)
            loss = (pred - b.y).abs().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], clip)
            opt.step()
            total += float(loss.detach()) * len(sel)
            count += len(sel)
        dev_mae = evaluate(model, cache, y, dev_indices, device)
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        row = {"epoch": int(epoch), "train_loss": total / max(count, 1), "valid": dev_mae}
        history.append(row)
        if epoch == 1 or epoch % 20 == 0 or epoch == int(max_epochs):
            log(
                f"[center_comp] epoch={epoch:03d} train={row['train_loss']:.6f} "
                f"valid={dev_mae:.6f} best={best:.6f}@{best_epoch}"
            )
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
            log(f"[center_comp] early stop at epoch {epoch}")
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
        soup_mae = evaluate(model, cache, y, dev_indices, device)
        model.load_state_dict(backup)
    if best_state is not None:
        model.load_state_dict(best_state)
    return CenterResult(
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


def load_state_into(model, state, device: str):
    model.load_state_dict(state)
    return model


# ===========================================================================
# decision logic
# ===========================================================================
def center_comp_decision(
    *,
    g_center: float,
    g_bind: float,
    mae_prev: float = V5_PRE_SOUP,
    mae_center: float,
    mae_shuffle: float,
) -> dict[str, Any]:
    if g_center >= MATERIAL and g_bind >= MATERIAL:
        outcome = "A"
    elif g_center >= MATERIAL and g_bind < BIND_FAIL:
        outcome = "B"
    elif g_center >= MATERIAL and BIND_FAIL <= g_bind < MATERIAL:
        outcome = "B_AMBIGUOUS"
    elif g_center < MATERIAL and g_bind >= MATERIAL:
        outcome = "C"
    else:
        outcome = "D"
    return {
        "outcome": outcome,
        "mae_prev": float(mae_prev),
        "mae_center": float(mae_center),
        "mae_bind_shuffle": float(mae_shuffle),
        "g_center": float(g_center),
        "g_bind": float(g_bind),
        "gap_to_strong": float(mae_center - CANONICAL_GPU1_BASELINE),
        "thresholds": {
            "material": MATERIAL,
            "bind_fail": BIND_FAIL,
            "mae_prev": float(mae_prev),
            "canonical_gpu1_baseline": CANONICAL_GPU1_BASELINE,
        },
    }


# ===========================================================================
# Gate 0
# ===========================================================================
def _tiny_cache(seed: int = 0) -> tuple[V6.LabelFreeCache, np.ndarray]:
    cache, y = V6.tiny_cache(seed)
    return cache, y


def gate0_data_free_checks(device: str = "cpu") -> dict[str, Any]:
    torch = _torch()
    checks: dict[str, Any] = {"official_test_loaded": False, "official_valid_blocked": True}
    torch.manual_seed(7)

    checks["pair_width"] = int(PAIR_WIDTH)
    checks["rho_in"] = int(RHO_IN)
    checks["frozen_base_dim"] = int(FROZEN_BASE_DIM)
    checks["parameter_accounting"] = parameter_accounting()
    checks["parameter_accounting_expected"] = {"phi": 13712, "rho": 6224, "head": 10481, "total": 30417}
    checks["parameter_accounting_ok"] = bool(parameter_accounting() == checks["parameter_accounting_expected"])

    model = CenterCompFactory.build(seed=0)
    counts = model_parameter_count(model)
    checks["model_parameter_count"] = counts
    checks["model_parameter_count_ok"] = bool(counts == checks["parameter_accounting_expected"])
    checks["shapes"] = {
        "phi0": [int(model.phi[0].in_features), int(model.phi[0].out_features)],
        "phi2": [int(model.phi[2].in_features), int(model.phi[2].out_features)],
        "rho0": [int(model.rho[0].in_features), int(model.rho[0].out_features)],
        "rho2": [int(model.rho[2].in_features), int(model.rho[2].out_features)],
        "head": [int(model.head.in_features), int(model.head.out_features)],
    }
    checks["shapes_ok"] = bool(
        checks["shapes"]["phi0"] == [197, 64]
        and checks["shapes"]["phi2"] == [64, 16]
        and checks["shapes"]["rho0"] == [80, 64]
        and checks["shapes"]["rho2"] == [64, 16]
        and checks["shapes"]["head"] == [FROZEN_BASE_DIM + 16, 1]
    )

    # ---- pair contributes to both endpoint centers exactly once ----
    cache, y = _tiny_cache(seed=3)
    row = 0
    n = cache.n_of_row(row)
    c0, c1 = int(cache.c_offsets[row]), int(cache.c_offsets[row + 1])
    p0, p1 = int(cache.pair_offsets[row]), int(cache.pair_offsets[row + 1])
    C = torch.as_tensor(cache.C[c0:c1], dtype=torch.float32).unsqueeze(0)
    i0 = torch.as_tensor(cache.pair_i0[p0:p1]).unsqueeze(0)
    i1 = torch.as_tensor(cache.pair_i1[p0:p1]).unsqueeze(0)
    rel = torch.as_tensor(cache.pair_rel[p0:p1]).unsqueeze(0)
    valid = torch.ones((1, p1 - p0), dtype=torch.bool)
    phi = model.phi
    with torch.no_grad():
        e, q = center_pair_context(C, i0, i1, rel, valid, phi)
        manual = torch.zeros((1, n, PHI_OUT))
        for k in range(p1 - p0):
            manual[0, int(i0[0, k])] += e[0, k]
            manual[0, int(i1[0, k])] += e[0, k]
    checks["pair_to_centers_max_abs_diff"] = float((q - manual).abs().max())
    checks["pair_to_centers_exact"] = bool(float((q - manual).abs().max()) <= 1e-6)

    # ---- descriptor symmetry and pair-order invariance ----
    swap_i0, swap_i1 = i1, i0
    with torch.no_grad():
        e_swap, q_swap = center_pair_context(C, swap_i0, swap_i1, rel, valid, phi)
        _, q2 = center_pair_context(C, i0, i1, rel, valid, phi)
    checks["pair_swap_descriptor_max_abs_diff"] = float((e - e_swap).abs().max())
    checks["pair_swap_descriptor_invariant"] = bool(float((e - e_swap).abs().max()) <= 1e-6)

    order = np.random.default_rng(11).permutation(p1 - p0)
    with torch.no_grad():
        _, q_perm = center_pair_context(
            C,
            i0[:, torch.as_tensor(order)],
            i1[:, torch.as_tensor(order)],
            rel[:, torch.as_tensor(order)],
            valid,
            phi,
        )
    checks["pair_order_invariance_max_abs_diff"] = float((q - q_perm).abs().max())
    checks["pair_order_invariant"] = bool(float((q - q_perm).abs().max()) <= 1e-6)

    # ---- batching invariance ----
    cache_b, y_b = _tiny_cache(seed=3)
    idx = list(range(cache_b.n_graphs))
    with torch.no_grad():
        b_single = make_center_batch(cache_b, y_b, [0], device)
        pred_single = CenterCompFactory.build(seed=0).to(device)(b_single)
    # rebuild the model deterministically and compare single vs batched forward
    torch.manual_seed(7)
    model2 = CenterCompFactory.build(seed=0).to(device).eval()
    with torch.no_grad():
        pred_single2 = model2(make_center_batch(cache_b, y_b, [0], device))
        pred_batched = model2(make_center_batch(cache_b, y_b, idx, device))
    checks["batching_invariance_max_abs_diff"] = float((pred_single2[0] - pred_batched[0]).abs())
    checks["batching_invariant"] = bool(checks["batching_invariance_max_abs_diff"] <= 1e-5)
    checks["_bootstrap_pred_finite"] = bool(np.isfinite(pred_single.detach().cpu().numpy()).all())

    # ---- relabel invariance of q, h_center and prediction ----
    perm_rows = np.random.default_rng(13).permutation(n)
    inv = np.empty(n, dtype=np.int64)
    inv[perm_rows] = np.arange(n)
    new_i0 = inv[cache.pair_i0[p0:p1]]
    new_i1 = inv[cache.pair_i1[p0:p1]]
    C_rel = C[:, perm_rows]
    i0r = torch.as_tensor(new_i0).unsqueeze(0)
    i1r = torch.as_tensor(new_i1).unsqueeze(0)
    # keep this algebraic check on CPU so it runs identically for any requested device
    torch.manual_seed(7)
    model3 = CenterCompFactory.build(seed=0).eval()
    with torch.no_grad():
        _, q_rel = center_pair_context(C_rel, i0r, i1r, rel, valid, model3.phi)
        h_center_orig = compose_center(
            C,
            q,
            torch.arange(n).unsqueeze(0),
            torch.ones((1, n), dtype=torch.bool),
            model3.rho,
        )
        h_center_rel = compose_center(
            C_rel,
            q_rel,
            torch.arange(n).unsqueeze(0),
            torch.ones((1, n), dtype=torch.bool),
            model3.rho,
        )
    checks["relabel_h_center_max_abs_diff"] = float((h_center_orig - h_center_rel).abs().max())
    checks["relabel_h_center_invariant"] = bool(float((h_center_orig - h_center_rel).abs().max()) <= 1e-5)

    # ---- shuffle preserves {q}, {c}, h_base; changes the binding ----
    cache_sh, y_sh = _tiny_cache(seed=5)
    idx_sh = list(range(cache_sh.n_graphs))
    b_real = make_center_batch(cache_sh, y_sh, idx_sh, device, shuffle=False)
    b_shuf = make_center_batch(cache_sh, y_sh, idx_sh, device, shuffle=True)
    torch.manual_seed(7)
    m_sh = CenterCompFactory.build(seed=0).to(device).eval()
    with torch.no_grad():
        _, q_real = center_pair_context(b_real.C3, b_real.i0, b_real.i1, b_real.rel, b_real.valid, m_sh.phi)
        _, q_shuf = center_pair_context(b_shuf.C3, b_shuf.i0, b_shuf.i1, b_shuf.rel, b_shuf.valid, m_sh.phi)
        hc_real = compose_center(b_real.C3, q_real, b_real.center_perm, b_real.occ_valid, m_sh.rho)
        hc_shuf = compose_center(b_shuf.C3, q_shuf, b_shuf.center_perm, b_shuf.occ_valid, m_sh.rho)
    # c multiset and h_base unchanged
    checks["shuffle_c_multiset_max_abs_diff"] = float((b_real.C3 - b_shuf.C3).abs().max())
    checks["shuffle_c_multiset_preserved"] = bool(checks["shuffle_c_multiset_max_abs_diff"] == 0.0)
    checks["shuffle_h_base_max_abs_diff"] = float((b_real.base - b_shuf.base).abs().max())
    checks["shuffle_h_base_preserved"] = bool(checks["shuffle_h_base_max_abs_diff"] == 0.0)
    # q multiset per graph preserved
    qm = 0.0
    for bi in range(len(idx_sh)):
        ni = int(cache_sh.n_of_row(int(cache_sh.rows_for(idx_sh)[bi])))
        a = np.sort(q_real[bi, :ni].detach().cpu().numpy(), axis=0)
        b = np.sort(q_shuf[bi, :ni].detach().cpu().numpy(), axis=0)
        qm = max(qm, float(np.abs(a - b).max()))
    checks["shuffle_q_multiset_max_abs_diff"] = qm
    checks["shuffle_q_multiset_preserved"] = bool(qm <= 1e-6)
    checks["shuffle_changes_binding"] = bool(float((hc_real - hc_shuf).abs().max()) > 0.0)
    checks["shuffle_binding_max_abs_diff"] = float((hc_real - hc_shuf).abs().max())

    # ---- gradient scope ----
    torch.manual_seed(7)
    gmodel = CenterCompFactory.build(seed=0).to(device)
    gb = make_center_batch(cache, y, [0], device)
    (gmodel(gb) - gb.y).abs().mean().backward()
    grads = {
        "phi0_w": float(gmodel.phi[0].weight.grad.abs().sum()),
        "phi2_w": float(gmodel.phi[2].weight.grad.abs().sum()),
        "rho0_w": float(gmodel.rho[0].weight.grad.abs().sum()),
        "rho2_w": float(gmodel.rho[2].weight.grad.abs().sum()),
        "head_w": float(gmodel.head.weight.grad.abs().sum()),
    }
    checks["gradients"] = grads
    checks["gradient_reaches_phi"] = bool(grads["phi0_w"] > 0 and grads["phi2_w"] > 0)
    checks["gradient_reaches_rho"] = bool(grads["rho0_w"] > 0 and grads["rho2_w"] > 0)
    checks["gradient_reaches_reader"] = bool(grads["head_w"] > 0)
    frozen = build_frozen_v2_model(device)
    checks["frozen_local_params"] = int(sum(1 for _ in frozen.parameters()))
    checks["frozen_local_all_detached"] = bool(all(not p.requires_grad for p in frozen.parameters()))
    checks["frozen_local_no_grad"] = bool(all(p.grad is None for p in frozen.parameters()))
    trainable_names = sorted(n for n, _ in gmodel.named_parameters())
    checks["trainable_param_names"] = trainable_names
    checks["no_local_encoder_parameters"] = bool(
        all(n.startswith("phi.") or n.startswith("rho.") or n.startswith("head.") for n in trainable_names)
    )

    checks["official_test_blocked"] = True
    bool_keys = [
        k
        for k, v in checks.items()
        if isinstance(v, bool) and k not in {"official_test_loaded"}
    ]
    checks["data_free_all_pass"] = bool(all(checks[k] for k in bool_keys))
    return checks


def gate0_real_checks(cache: V6.LabelFreeCache, records: Sequence[Mapping[str, Any]], device: str = "cpu") -> dict[str, Any]:
    torch = _torch()
    checks: dict[str, Any] = {"official_test_loaded": False}

    # 1. frozen TCCD-v2 C re-derivation matches the reused cache
    idx = list(range(min(8, cache.n_graphs)))
    model = build_frozen_v2_model(device)
    base = V5.V2
    with torch.no_grad():
        b = base.make_batch(records, idx, device)
        _, C_flat, h_base = model.forward_padded(b)
    C_pred = C_flat.cpu().numpy().reshape(len(idx), b["X_pad"].shape[1], K_PROTO)
    diff = 0.0
    hb_diff = 0.0
    hb_scale = 1.0
    for bi, gi in enumerate(idx):
        row = int(cache.rows_for([gi])[0])
        n = int(cache.n_of_row(row))
        c0, c1 = int(cache.c_offsets[row]), int(cache.c_offsets[row + 1])
        diff = max(diff, float(np.abs(C_pred[bi, :n] - cache.C[c0:c1]).max()))
        hb_scale = max(hb_scale, float(np.abs(cache.base[row]).max()))
        hb_diff = max(hb_diff, float(np.abs(h_base.cpu().numpy()[bi] - cache.base[row]).max()))
    checks["frozen_C_recompute_max_abs_diff"] = float(diff)
    checks["frozen_C_matches_cache"] = bool(diff < 1e-6)
    # h_base is a large-magnitude global summary; cross-device float32 reduction
    # order is compared relatively (the cache was produced on the remote A100).
    checks["frozen_h_base_recompute_max_abs_diff"] = float(hb_diff)
    checks["frozen_h_base_recompute_rel_diff"] = float(hb_diff / hb_scale)
    checks["frozen_h_base_matches_cache"] = bool(hb_diff / hb_scale < 1e-5)

    # 2. v5 descriptor equivalence
    y = V6.load_cache_labels()
    with torch.no_grad():
        b = make_center_batch(cache, y, idx, device)
        ours = pair_descriptor(b.C3, b.i0, b.i1, b.rel)
    ref = V6._pair_means_via_v5(cache, y, idx, device)  # mean pbar reference
    # direct per-pair descriptor comparison against the v5 tensor path
    rec_stub = [{"n": cache.n_of_row(r)} for r in range(cache.n_graphs)]

    class _View:
        def __init__(self, c, yy):
            self._c = c
            self.y = yy

        def __getattr__(self, name):
            return getattr(self._c, name)

    v5b = V5.make_pair_batch(_View(cache, y), rec_stub, idx, device)
    v5p = V5.pair_descriptor(v5b.C3, v5b.i0, v5b.i1, v5b.rel)
    checks["v5_pair_descriptor_max_abs_diff"] = float((ours - v5p).abs().max())
    checks["v5_pair_descriptor_equivalence"] = bool(float((ours - v5p).abs().max()) <= 1e-6)
    checks["reference_pbar_finite"] = bool(np.isfinite(ref).all())

    checks["cache_meta_official_test_loaded"] = cache.meta.get("official_test_loaded")
    checks["cache_meta_official_test_false"] = bool(cache.meta.get("official_test_loaded") is False)
    checks["cache_v2_sha_ok"] = bool(cache.meta.get("v2_checkpoint_sha256") == V2_BEST_CHECKPOINT_SHA256)

    bool_keys = [
        k
        for k, v in checks.items()
        if isinstance(v, bool) and not k.endswith("official_test_loaded")
    ]
    checks["real_all_pass"] = bool(all(checks[k] for k in bool_keys))
    return checks


# ===========================================================================
# mechanism / stratification reporting (read-only)
# ===========================================================================
def prediction_change_report(real: np.ndarray, shuffled: np.ndarray) -> dict[str, Any]:
    delta = np.abs(np.asarray(real, dtype=np.float64) - np.asarray(shuffled, dtype=np.float64))
    return {
        "n": int(delta.size),
        "mean_abs_change": float(delta.mean()) if delta.size else 0.0,
        "median_abs_change": float(np.median(delta)) if delta.size else 0.0,
        "p90_abs_change": float(np.percentile(delta, 90)) if delta.size else 0.0,
        "max_abs_change": float(delta.max()) if delta.size else 0.0,
        "fraction_ge_0.01": float((delta >= MATERIAL_CHANGE).mean()) if delta.size else 0.0,
        "fraction_ge_0.05": float((delta >= MATERIAL_CHANGE_STRONG).mean()) if delta.size else 0.0,
        "fraction_ge_0.10": float((delta >= MATERIAL_CHANGE_VERY_STRONG).mean()) if delta.size else 0.0,
    }


def size_stratification(
    cache: V6.LabelFreeCache,
    dev_indices: Sequence[int],
    y: np.ndarray,
    real_pred: np.ndarray,
    shuffle_pred: np.ndarray,
    *,
    seed: int = 0,
) -> dict[str, Any]:
    dev = np.asarray(list(dev_indices), dtype=np.int64)
    rows = cache.rows_for(dev)
    ns = np.asarray([cache.n_of_row(int(r)) for r in rows], dtype=np.int64)
    ydev = np.asarray(y[rows], dtype=np.float64)
    real = np.asarray(real_pred, dtype=np.float64)
    shuf = np.asarray(shuffle_pred, dtype=np.float64)
    order = np.argsort(ns, kind="stable")
    thirds = np.array_split(order, 3)
    labels = ("small", "medium", "large")
    out: dict[str, Any] = {}
    for label, sel in zip(labels, thirds):
        if sel.size == 0:
            continue
        r = np.abs(real[sel] - ydev[sel]).mean()
        s = np.abs(shuf[sel] - ydev[sel]).mean()
        out[label] = {
            "n_graphs": int(sel.size),
            "n_min": int(ns[sel].min()),
            "n_max": int(ns[sel].max()),
            "n_mean": float(ns[sel].mean()),
            "real_mae": float(r),
            "bind_shuffle_mae": float(s),
            "g_bind": float(s - r),
        }
    return out
