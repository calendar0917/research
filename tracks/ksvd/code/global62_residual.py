#!/usr/bin/env python
"""TCCD-GLOBAL62-v0: frozen-TCCD + B-Full global-context residual screen.

Single variable: the exact 62-D B-Full ``global_all`` graph-level context.

    yhat_G = yhat_base,G + f(g_G),   g_G in R^62

``yhat_base`` is the exact frozen TCCD-v5 PRE Top-5 soup prediction, obtained
by a pure forward pass of the frozen soup state over the reused TCCD-v5 pair
cache.  No TCCD parameter is trained.  The only new trained object is one small
residual head whose hidden transformation is the exact B-Full ``global_encoder``
sequence (``_MLPBlock(62, 32, 32, dropout=0)``) followed by a scalar linear.

Official ZINC valid/test are never loaded.
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

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = REPO_ROOT / "tracks/ksvd/results/global62_residual"
PROTOCOL_VERSION = "tccd_global62_v0_frozen_global_residual_v1"

# ---------------------------------------------------------------------------
# frozen references (read from formal artifacts; never re-run)
# ---------------------------------------------------------------------------
V5_STAGE_A_JSON = V5.RESULTS_DIR / "stageA_seed0.json"
V5_PRE_SOUP_STATE = V5.RESULTS_DIR / "stageA_pre_seed0_soup.pt"
V2_BEST_CHECKPOINT = V5.V2_BEST_CHECKPOINT
V2_BEST_CHECKPOINT_SHA256 = V5.V2_BEST_CHECKPOINT_SHA256

MAE_BASE_REFERENCE = 0.2514181435108185     # TCCD-v5 PRE Top-5 soup, internal dev
MAE_POST_REFERENCE = 0.2522333264350891     # TCCD-v5 POST Top-5 soup
MAE_BASE_V2_SOUP = 0.2783639132976532       # TCCD-v2 BASE Top-5 soup
B_FULL_SCALE_REFERENCE = 0.11981802638241788  # canonical B-Full seed-0 soup (scale only)

REPRODUCTION_TOL = 1e-6

# ---------------------------------------------------------------------------
# global context definition (exact B-Full provenance)
# ---------------------------------------------------------------------------
GLOBAL_WIDTH = 62
GLOBAL_FEATURE_FUNCTION = (
    "tracks.ksvd.experiments.luyin16.zinc_long_range_proxy.global_feature_views(...)[\"global_all\"]"
)
GLOBAL_STRUCTURE_SHORT = 15
GLOBAL_STRUCTURE_LONG = 15
GLOBAL_ATOM_BINS = 28
GLOBAL_BOND_BINS = 4
GLOBAL_ATTRIBUTES = GLOBAL_ATOM_BINS + GLOBAL_BOND_BINS  # 32

# exact B-Full global encoder: _MLPBlock(GLOBAL_WIDTH, max(patch_hidden//2,32), 32, dropout)
GLOBAL_ENCODER_HIDDEN = 32
GLOBAL_ENCODER_OUT = 32
GLOBAL_ENCODER_PROVENANCE = (
    "tracks.ksvd.experiments.luyin16.zinc_ksvd_patch_path_pooling._MLPBlock"
    "(width=62, hidden=32, output=32, dropout=0.0); B-Full global_encoder uses"
    " GLOBAL_WIDTH=62 and global_encoder_hidden=max(patch_hidden//2,32)=32 for patch_hidden=64"
)

# ---------------------------------------------------------------------------
# frozen training protocol
# ---------------------------------------------------------------------------
SEED = 0
LR = 1.0e-3
WD = 1.0e-5
BATCH = 128
MAX_EPOCHS = 240
PATIENCE = 40
TOP_K_SOUP = 5
SHUFFLE_SEED = 20260923

# ---------------------------------------------------------------------------
# preregistered thresholds
# ---------------------------------------------------------------------------
CASE_A_GLOBAL = 0.030
CASE_A_BIND = 0.020
CASE_B_GLOBAL = 0.015
CASE_B_BIND = 0.010
CASE_C_GLOBAL = 0.005
BIND_PRECEDENCE = 0.005


def _torch():
    return T._torch()


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


# ===========================================================================
# frozen reference values
# ===========================================================================
def frozen_reference_values() -> dict[str, Any]:
    if not V5_STAGE_A_JSON.exists():
        raise FileNotFoundError(f"missing TCCD-v5 Stage-A artifact: {V5_STAGE_A_JSON}")
    v5 = json.loads(V5_STAGE_A_JSON.read_text(encoding="utf-8"))
    if v5.get("official_test_loaded") is not False:
        raise RuntimeError("TCCD-v5 artifact does not certify official_test_loaded == false")
    res = v5["results"]
    return {
        "source": str(V5_STAGE_A_JSON.relative_to(REPO_ROOT)),
        "source_commit": v5.get("commit"),
        "pre_soup": float(res["PRE"]["soup_valid_mae"]),
        "post_soup": float(res["POST"]["soup_valid_mae"]),
        "base_v2_soup": float(res["BASE"]["soup_valid_mae"]),
        "b_full_scale_reference": B_FULL_SCALE_REFERENCE,
        "official_test_loaded": False,
    }


# ===========================================================================
# frozen base prediction (pure forward pass; no training)
# ===========================================================================
def load_pair_cache():
    paths = V5.pair_cache_paths("all")
    cache = V5._load_pair_cache(paths)
    if cache.meta.get("official_test_loaded") is not False:
        raise RuntimeError("reused v5 pair cache does not certify official_test_loaded == false")
    if cache.meta.get("v2_checkpoint_sha256") != V2_BEST_CHECKPOINT_SHA256:
        raise RuntimeError("reused v5 pair cache was not built from the frozen TCCD-v2 checkpoint")
    return cache


def frozen_base_predictions(device: str = "cpu") -> dict[str, Any]:
    """Forward the exact frozen TCCD-v5 PRE soup over the reused pair cache."""
    if not V5_PRE_SOUP_STATE.exists():
        raise FileNotFoundError(f"missing frozen v5 PRE soup state: {V5_PRE_SOUP_STATE}")
    soup_sha = _sha256_file(V5_PRE_SOUP_STATE)

    cache = load_pair_cache()
    ns = np.diff(cache.c_offsets)
    records = [{"n": int(x)} for x in ns]

    torch = _torch()
    state = torch.load(V5_PRE_SOUP_STATE, map_location=device, weights_only=True)
    model = V5.StageAModelFactory.build("pre", seed=SEED).to(device)
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    n_graphs = len(cache.graph_index)
    preds = np.zeros(n_graphs, dtype=np.float32)
    with torch.no_grad():
        for start in range(0, n_graphs, 64):
            chunk = list(range(start, min(start + 64, n_graphs)))
            b = V5.make_pair_batch(cache, records, chunk, device)
            preds[chunk] = model(b).detach().cpu().numpy().astype(np.float32, copy=False)

    return {
        "pred": preds,
        "cache": cache,
        "records": records,
        "soup_sha256": soup_sha,
        "official_test_loaded": False,
    }


# ===========================================================================
# global context extraction (exact B-Full definition)
# ===========================================================================
def load_global62_raw(data_root: Path | None = None) -> dict[str, Any]:
    """Return the raw 62-D B-Full global_all matrix aligned with TCCD records."""
    from tracks.ksvd.experiments.luyin16.zinc_long_range_proxy import (
        ATOM_BINS,
        BOND_BINS,
        _load_zinc,
        global_feature_views,
    )

    if ATOM_BINS != GLOBAL_ATOM_BINS or BOND_BINS != GLOBAL_BOND_BINS:
        raise RuntimeError(f"B-Full histogram widths changed: ATOM_BINS={ATOM_BINS} BOND_BINS={BOND_BINS}")
    root = Path(data_root) if data_root is not None else (REPO_ROOT / "data/ZINC")
    dataset = _load_zinc(root, "train")
    views = global_feature_views(dataset)
    global_all = np.asarray(views["global_all"], dtype=np.float32)
    if global_all.shape[1] != GLOBAL_WIDTH:
        raise RuntimeError(f"global_all width {global_all.shape[1]} != {GLOBAL_WIDTH}")
    # y alignment cross-check against the dataset used to build TCCD records
    dataset_y = np.asarray([float(dataset[i].y.reshape(-1)[0]) for i in range(len(dataset))], dtype=np.float64)
    return {
        "global_all": global_all,
        "dataset_y": dataset_y,
        "n_graphs": int(global_all.shape[0]),
        "short": np.asarray(views["global_structure_short"], dtype=np.float32),
        "long": np.asarray(views["global_structure_long"], dtype=np.float32),
        "attributes": np.asarray(views["global_attributes"], dtype=np.float32),
        "atom_bins": int(ATOM_BINS),
        "bond_bins": int(BOND_BINS),
        "finite_rate": float(np.isfinite(global_all).mean()),
    }


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray
    constant_columns: list[int]

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((np.asarray(x, dtype=np.float32) - self.mean) / self.std).astype(np.float32)


def fit_train_only_standardizer(raw: np.ndarray, train_idx: Sequence[int]) -> Standardizer:
    train = np.asarray(raw, dtype=np.float64)[np.asarray(list(train_idx), dtype=np.int64)]
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    constant = [int(i) for i, value in enumerate(std) if not np.isfinite(value) or value <= 0.0]
    std = std.copy()
    std[constant] = 1.0
    return Standardizer(mean=mean.astype(np.float32), std=std.astype(np.float32), constant_columns=constant)


# ===========================================================================
# residual head (exact B-Full global_encoder hidden transformation)
# ===========================================================================
def global62_head_class():
    """62 -> Linear(62,32) -> LayerNorm(32) -> ReLU -> Linear(32,32) -> ReLU -> Linear(32,1)."""
    torch = _torch()
    nn = torch.nn

    class Global62Head(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(GLOBAL_WIDTH, GLOBAL_ENCODER_HIDDEN),
                nn.LayerNorm(GLOBAL_ENCODER_HIDDEN),
                nn.ReLU(),
                nn.Linear(GLOBAL_ENCODER_HIDDEN, GLOBAL_ENCODER_OUT),
                nn.ReLU(),
            )
            self.residual = nn.Linear(GLOBAL_ENCODER_OUT, 1)

        def forward(self, x):
            return self.residual(self.encoder(x)).reshape(-1)

    return Global62Head()


def parameter_accounting() -> dict[str, int]:
    encoder = GLOBAL_WIDTH * GLOBAL_ENCODER_HIDDEN + GLOBAL_ENCODER_HIDDEN
    layernorm = 2 * GLOBAL_ENCODER_HIDDEN
    second = GLOBAL_ENCODER_HIDDEN * GLOBAL_ENCODER_OUT + GLOBAL_ENCODER_OUT
    residual = GLOBAL_ENCODER_OUT + 1
    return {
        "encoder_linear1": int(encoder),
        "encoder_layernorm": int(layernorm),
        "encoder_linear2": int(second),
        "residual_linear": int(residual),
        "total": int(encoder + layernorm + second + residual),
    }


# ===========================================================================
# training
# ===========================================================================
@dataclass
class HeadResult:
    best_valid: float
    best_epoch: int
    soup_valid: float | None
    soup_members: list[int]
    init_checksum: str
    state_best: Any = None
    state_soup: Any = None
    train_history: list[dict[str, Any]] = field(default_factory=list)
    wall_s: float = 0.0


def _state_checksum(model) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype=np.float32).tobytes())
    return digest.hexdigest()


def _evaluate_head(model, X, base, y) -> float:
    torch = _torch()
    model.eval()
    with torch.no_grad():
        xt = torch.as_tensor(np.asarray(X, dtype=np.float32))
        bt = torch.as_tensor(np.asarray(base, dtype=np.float32))
        yt = torch.as_tensor(np.asarray(y, dtype=np.float32))
        pred = bt + model(xt)
        mae = float((pred - yt).abs().mean())
    return mae


def train_residual_head(
    X_train: np.ndarray,
    r_train: np.ndarray,
    base_train: np.ndarray,
    y_train: np.ndarray,
    X_dev: np.ndarray,
    base_dev: np.ndarray,
    y_dev: np.ndarray,
    *,
    seed: int = SEED,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    batch: int = BATCH,
    lr: float = LR,
    wd: float = WD,
    log=print,
) -> HeadResult:
    """Train f on r = y - yhat_base, minimising L1 of the final prediction."""
    torch = _torch()
    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed) + 91011)
    model = global62_head_class()
    init_checksum = _state_checksum(model)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=wd)

    Xtr = torch.as_tensor(np.asarray(X_train, dtype=np.float32))
    btr = torch.as_tensor(np.asarray(base_train, dtype=np.float32))
    ytr = torch.as_tensor(np.asarray(y_train, dtype=np.float32))

    best = math.inf
    best_epoch = -1
    best_state = None
    top: list[tuple[float, int, Any]] = []
    history: list[dict[str, Any]] = []
    stale = 0
    started = time.time()
    n = Xtr.shape[0]
    for epoch in range(1, int(max_epochs) + 1):
        model.train()
        order = rng.permutation(n)
        total = 0.0
        count = 0
        for start in range(0, n, int(batch)):
            sel = torch.as_tensor(order[start : start + int(batch)], dtype=torch.long)
            opt.zero_grad(set_to_none=True)
            pred = btr[sel] + model(Xtr[sel])
            loss = (pred - ytr[sel]).abs().mean()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * int(sel.numel())
            count += int(sel.numel())
        dev_mae = _evaluate_head(model, X_dev, base_dev, y_dev)
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append({"epoch": int(epoch), "train_loss": total / max(count, 1), "valid": dev_mae})
        if epoch == 1 or epoch % 20 == 0 or epoch == int(max_epochs):
            log(f"[global62] epoch={epoch:03d} train={total / max(count, 1):.6f} valid={dev_mae:.6f} best={best:.6f}@{best_epoch}")
        if dev_mae < best - 1e-9:
            best = dev_mae
            best_epoch = epoch
            best_state = copy.deepcopy(state)
            stale = 0
        else:
            stale += 1
        if len(top) < TOP_K_SOUP or dev_mae < max(item[0] for item in top):
            top.append((dev_mae, epoch, state))
            top.sort(key=lambda item: (item[0], item[1]))
            top = top[:TOP_K_SOUP]
        if stale >= int(patience):
            log(f"[global62] early stop at epoch {epoch}")
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
        soup_mae = _evaluate_head(model, X_dev, base_dev, y_dev)
        model.load_state_dict(backup)
    if best_state is not None:
        model.load_state_dict(best_state)
    return HeadResult(
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


def head_predictions(state, X, base) -> np.ndarray:
    torch = _torch()
    model = global62_head_class()
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        xt = torch.as_tensor(np.asarray(X, dtype=np.float32))
        bt = torch.as_tensor(np.asarray(base, dtype=np.float32))
        return (bt + model(xt)).detach().cpu().numpy()


# ===========================================================================
# shuffle / decision / secondary analyses
# ===========================================================================
def derangement(n: int, seed: int = SHUFFLE_SEED) -> np.ndarray:
    """Deterministic fixed-point-free permutation of ``n`` items."""
    n = int(n)
    if n < 2:
        raise ValueError("derangement requires n >= 2")
    rng = np.random.default_rng(int(seed))
    for _ in range(10000):
        perm = rng.permutation(n)
        if np.all(perm != np.arange(n)):
            return np.asarray(perm, dtype=np.int64)
    # deterministic fallback: cyclic shift (no fixed points for n >= 2)
    return np.roll(np.arange(n), 1)


def preregistered_verdict(g_total: float, g_global: float, g_bind: float) -> dict[str, Any]:
    if g_bind < BIND_PRECEDENCE:
        core = "DOWNGRADED_SHALLOW_BINDING"
    elif g_global >= CASE_A_GLOBAL and g_bind >= CASE_A_BIND:
        core = "A"
    elif g_global >= CASE_B_GLOBAL and g_bind >= CASE_B_BIND:
        core = "B"
    elif CASE_C_GLOBAL <= g_global < CASE_B_GLOBAL:
        core = "C"
    elif g_global < CASE_C_GLOBAL:
        core = "D"
    else:
        core = "D"
    return {
        "case": core,
        "g_total": float(g_total),
        "g_global": float(g_global),
        "g_bind": float(g_bind),
        "bind_precedence": BIND_PRECEDENCE,
        "thresholds": {
            "case_a_global": CASE_A_GLOBAL,
            "case_a_bind": CASE_A_BIND,
            "case_b_global": CASE_B_GLOBAL,
            "case_b_bind": CASE_B_BIND,
            "case_c_global": CASE_C_GLOBAL,
        },
    }


def residual_correlations(raw: np.ndarray, residual: np.ndarray) -> dict[str, list[float]]:
    from scipy.stats import pearsonr, spearmanr

    raw = np.asarray(raw, dtype=np.float64)
    residual = np.asarray(residual, dtype=np.float64)
    pearson: list[float] = []
    spearman: list[float] = []
    for column in range(raw.shape[1]):
        x = raw[:, column]
        if np.all(x == x[0]) or np.all(residual == residual[0]):
            pearson.append(0.0)
            spearman.append(0.0)
            continue
        pearson.append(float(pearsonr(x, residual)[0]))
        spearman.append(float(spearmanr(x, residual)[0]))
    return {"pearson": pearson, "spearman": spearman}


def size_stratification(
    atom_counts: np.ndarray,
    y_dev: np.ndarray,
    base_dev: np.ndarray,
    real_dev: np.ndarray,
    shuffle_dev: np.ndarray,
) -> dict[str, Any]:
    counts = np.asarray(atom_counts, dtype=np.int64)
    y = np.asarray(y_dev, dtype=np.float64)
    base = np.asarray(base_dev, dtype=np.float64)
    real = np.asarray(real_dev, dtype=np.float64)
    shuf = np.asarray(shuffle_dev, dtype=np.float64)
    order = np.argsort(counts, kind="stable")
    thirds = np.array_split(order, 3)
    labels = ("small", "medium", "large")
    out: dict[str, Any] = {}
    for label, sel in zip(labels, thirds):
        if sel.size == 0:
            continue
        base_mae = float(np.abs(base[sel] - y[sel]).mean())
        real_mae = float(np.abs(real[sel] - y[sel]).mean())
        shuf_mae = float(np.abs(shuf[sel] - y[sel]).mean())
        out[label] = {
            "n_graphs": int(sel.size),
            "n_min": int(counts[sel].min()),
            "n_max": int(counts[sel].max()),
            "n_mean": float(counts[sel].mean()),
            "base_mae": base_mae,
            "real_mae": real_mae,
            "shuffle_mae": shuf_mae,
            "improvement_real": float(base_mae - real_mae),
            "g_bind": float(shuf_mae - real_mae),
        }
    return out
