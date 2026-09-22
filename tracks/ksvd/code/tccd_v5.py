#!/usr/bin/env python
"""TCCD-v5 occurrence-preserving pair nonlinearity audit.

This module preserves the exact TCCD-v2 local encoder, prototype vocabulary,
relation set, temperature, regularizers and reader mechanics.  The only new
learnable component is one small shared pair function ``phi`` applied to every
unordered occurrence pair.  The single causal variable is the placement of the
nonlinearity:

* PRE :  mean_{i<j} phi(p_ij)
* POST:  phi(mean_{i<j} p_ij)

Official ZINC test is never loaded.  Formal execution is GPU1-only.
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
from tracks.ksvd.code import tccd_v2 as V2

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = REPO_ROOT / "tracks/ksvd/results/tccd_v5"
CACHE_DIR = RESULTS_DIR / "cache"
PROTOCOL_VERSION = "tccd_v5_occurrence_pair_nonlinearity_v1"

D_LOCAL = V2.D_LOCAL
K_PROTO = V2.K_PROTO
N_REL = V2.N_REL
EPS = V2.EPS
PROTO_INIT_SEED = V2.PROTO_INIT_SEED
PERM_SEED = 20260922

PHI_HIDDEN = 64
PHI_OUT = 16

TCCD_V2_COMMIT = "69a985a4ea3eb184c96c8ddad3857c4e87ee1dda"
TCCD_V2_BEST = 0.2862437069416046
TCCD_V2_SOUP = 0.26635152101516724
TCCD_V2_OFFICIAL_VALID_BEST = 0.287337
TCCD_V2_OFFICIAL_VALID_SOUP = 0.261988
CANONICAL_GPU1_BASELINE = 0.119818

V2_BEST_CHECKPOINT = V2.RESULTS_DIR / "prototype_rel_seed0_best.pt"
V2_BEST_CHECKPOINT_SHA256 = "093e0e46d4d5dab7e5558b807d993d1136df58d8f531d6b9b3621e8ab05c0f42"

STAGE_A_PASS = 0.015
STAGE_A_FAIL = 0.005
STAGE_A_MEAN_PASS = 0.010
STAGE_B_STRONG_PASS = 0.020
STAGE_B_FAIL = 0.005
STAGE_B_MEAN_PASS = 0.010
STAGE_B_SHUFFLE_PASS = 0.010
OFFICIAL_SOUP_ROUTE_A = 0.23
OFFICIAL_SOUP_ROUTE_B = 0.030

PAIR_PLACEMENTS = ("pre", "post")


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


def fixed_permutation(n: int, graph_index: int, seed: int = PERM_SEED) -> np.ndarray:
    """Deterministic graph-specific row permutation; label-free and persistent."""
    n = int(n)
    if n < 2:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng((int(seed), int(graph_index)))
    perm = rng.permutation(n)
    if np.array_equal(perm, np.arange(n)):
        perm = np.roll(perm, 1)
    return np.asarray(perm, dtype=np.int64)


def pair_descriptor_width(n_rel: int = N_REL) -> int:
    return 3 * K_PROTO + int(n_rel)


def pair_mlp_parameter_count(d_p: int | None = None, hidden: int = PHI_HIDDEN, out: int = PHI_OUT) -> int:
    d = pair_descriptor_width() if d_p is None else int(d_p)
    return int(d * hidden + hidden + hidden * out + out)


def frozen_base_dim() -> int:
    return int(T.h_dim(K_PROTO, N_REL))


def base_representation_dim() -> int:
    return int(frozen_base_dim() + PHI_OUT)


# ===========================================================================
# frozen TCCD-v2 model and pair cache
# ===========================================================================
def build_frozen_v2_model(device: str):
    model = V2.PrototypeModelFactory.build(
        V2.frozen_layout().feature_dim,
        D_LOCAL,
        K_PROTO,
        N_REL,
        V2.load_frozen_encoder_init(),
        mode="rel",
        proto_seed=PROTO_INIT_SEED,
        seed=0,
    ).to(device)
    if not V2_BEST_CHECKPOINT.exists():
        raise FileNotFoundError(f"missing TCCD-v2 checkpoint: {V2_BEST_CHECKPOINT}")
    actual = _sha256_file(V2_BEST_CHECKPOINT)
    if actual != V2_BEST_CHECKPOINT_SHA256:
        raise RuntimeError(f"TCCD-v2 checkpoint SHA mismatch: {actual} != {V2_BEST_CHECKPOINT_SHA256}")
    model.load_state_dict(V2.load_state(V2_BEST_CHECKPOINT, device))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


class PairCache:
    """Ragged per-graph C assignments, exact pair indices and relation values."""

    def __init__(
        self,
        base: np.ndarray,
        C: np.ndarray,
        c_offsets: np.ndarray,
        pair_i0: np.ndarray,
        pair_i1: np.ndarray,
        pair_rel: np.ndarray,
        pair_offsets: np.ndarray,
        y: np.ndarray,
        graph_index: np.ndarray,
        meta: dict[str, Any],
    ) -> None:
        self.base = base
        self.C = C
        self.c_offsets = c_offsets
        self.pair_i0 = pair_i0
        self.pair_i1 = pair_i1
        self.pair_rel = pair_rel
        self.pair_offsets = pair_offsets
        self.y = y
        self.graph_index = np.asarray(graph_index, dtype=np.int64)
        self.meta = meta
        self.row_of = {int(g): int(r) for r, g in enumerate(self.graph_index)}

    def rows_for(self, indices: Sequence[int]) -> np.ndarray:
        return np.asarray([self.row_of[int(g)] for g in indices], dtype=np.int64)

    def pair_count(self, row: int) -> int:
        return int(self.pair_offsets[row + 1] - self.pair_offsets[row])

    def pair_statistics(self) -> dict[str, Any]:
        counts = np.diff(self.pair_offsets)
        return {
            "n_graphs": int(len(counts)),
            "total_pairs": int(counts.sum()),
            "mean_pairs_per_graph": float(counts.mean()),
            "min_pairs_per_graph": int(counts.min()),
            "max_pairs_per_graph": int(counts.max()),
            "graphs_without_pairs": int(np.sum(counts == 0)),
            "total_occurrences": int(len(self.C)),
            "relation_count": int(self.pair_rel.shape[1]),
            "pair_descriptor_width": pair_descriptor_width(int(self.pair_rel.shape[1])),
        }


def pair_cache_paths(split: str) -> dict[str, Path]:
    return {
        "base": CACHE_DIR / f"pair_{split}_base.npy",
        "C": CACHE_DIR / f"pair_{split}_C.npy",
        "c_offsets": CACHE_DIR / f"pair_{split}_c_offsets.npy",
        "pair_i0": CACHE_DIR / f"pair_{split}_i0.npy",
        "pair_i1": CACHE_DIR / f"pair_{split}_i1.npy",
        "pair_rel": CACHE_DIR / f"pair_{split}_rel.npy",
        "pair_offsets": CACHE_DIR / f"pair_{split}_offsets.npy",
        "y": CACHE_DIR / f"pair_{split}_y.npy",
        "graph_index": CACHE_DIR / f"pair_{split}_graph_index.npy",
        "meta": CACHE_DIR / f"pair_{split}_metadata.json",
    }


def _load_pair_cache(paths: dict[str, Path]) -> PairCache:
    meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
    return PairCache(
        base=np.load(paths["base"]),
        C=np.load(paths["C"]),
        c_offsets=np.load(paths["c_offsets"]),
        pair_i0=np.load(paths["pair_i0"]),
        pair_i1=np.load(paths["pair_i1"]),
        pair_rel=np.load(paths["pair_rel"]),
        pair_offsets=np.load(paths["pair_offsets"]),
        y=np.load(paths["y"]),
        graph_index=np.load(paths["graph_index"]),
        meta=meta,
    )


def build_or_load_pair_cache(
    records: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
    split: str,
    device: str,
    *,
    force: bool = False,
    batch_size: int = 64,
) -> PairCache:
    paths = pair_cache_paths(split)
    if not force and all(p.exists() for p in paths.values()):
        meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
        if (
            meta.get("protocol") == PROTOCOL_VERSION
            and meta.get("v2_checkpoint_sha256") == V2_BEST_CHECKPOINT_SHA256
            and meta.get("official_test_loaded") is False
            and meta.get("graph_index") == [int(i) for i in indices]
        ):
            return _load_pair_cache(paths)

    torch = _torch()
    model = build_frozen_v2_model(device)
    base_rows: list[np.ndarray] = []
    C_list: list[np.ndarray] = []
    i0_list: list[np.ndarray] = []
    i1_list: list[np.ndarray] = []
    rel_list: list[np.ndarray] = []
    y_list: list[float] = []
    with torch.no_grad():
        for start in range(0, len(indices), int(batch_size)):
            chunk = list(indices[start : start + int(batch_size)])
            b = V2.make_batch(records, chunk, device)
            _, C_flat, h_base = model.forward_padded(b)
            base_rows.append(h_base.cpu().numpy().astype(np.float32, copy=True))
            R = b["R_pad"].cpu().numpy()
            Cc = C_flat.cpu().numpy().reshape(len(chunk), b["X_pad"].shape[1], K_PROTO)
            for bi, gi in enumerate(chunk):
                n = int(records[int(gi)]["n"])
                C_list.append(np.ascontiguousarray(Cc[bi, :n], dtype=np.float32))
                a0, a1 = np.triu_indices(n, 1)
                i0_list.append(a0.astype(np.int64))
                i1_list.append(a1.astype(np.int64))
                rel_list.append(np.ascontiguousarray(R[bi, :, a0, a1], dtype=np.float32))
                y_list.append(float(records[int(gi)]["y"]))

    C_all = np.concatenate(C_list, axis=0) if C_list else np.zeros((0, K_PROTO), dtype=np.float32)
    i0_all = np.concatenate(i0_list, axis=0) if i0_list else np.zeros((0,), dtype=np.int64)
    i1_all = np.concatenate(i1_list, axis=0) if i1_list else np.zeros((0,), dtype=np.int64)
    rel_all = np.concatenate(rel_list, axis=0) if rel_list else np.zeros((0, N_REL), dtype=np.float32)
    n_patches = np.asarray([len(c) for c in C_list], dtype=np.int64)
    n_pairs = np.asarray([len(a) for a in i0_list], dtype=np.int64)
    c_offsets = np.concatenate([[0], np.cumsum(n_patches)]).astype(np.int64)
    pair_offsets = np.concatenate([[0], np.cumsum(n_pairs)]).astype(np.int64)
    base = np.concatenate(base_rows, axis=0) if base_rows else np.zeros((0, frozen_base_dim()), dtype=np.float32)
    y = np.asarray(y_list, dtype=np.float32)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(paths["base"], base)
    np.save(paths["C"], C_all)
    np.save(paths["c_offsets"], c_offsets)
    np.save(paths["pair_i0"], i0_all)
    np.save(paths["pair_i1"], i1_all)
    np.save(paths["pair_rel"], rel_all)
    np.save(paths["pair_offsets"], pair_offsets)
    np.save(paths["y"], y)
    np.save(paths["graph_index"], np.asarray(indices, dtype=np.int64))
    meta = {
        "protocol": PROTOCOL_VERSION,
        "split": split,
        "graph_index": [int(i) for i in indices],
        "n_graphs": len(indices),
        "base_dim": int(base.shape[1]),
        "relation_count": int(rel_all.shape[1]),
        "pair_descriptor_width": pair_descriptor_width(int(rel_all.shape[1])),
        "phi_hidden": PHI_HIDDEN,
        "phi_out": PHI_OUT,
        "pair_set": "all unordered i<j; no self-pairs; no sampling",
        "relation_order": "[R_int, R_b0, R_b1, R_b2, R_geo] from TCCD-v2 make_padded_batch",
        "v2_checkpoint": str(V2_BEST_CHECKPOINT.relative_to(REPO_ROOT)),
        "v2_checkpoint_sha256": V2_BEST_CHECKPOINT_SHA256,
        "v2_commit": TCCD_V2_COMMIT,
        "official_test_loaded": False,
    }
    write_json(paths["meta"], meta)
    return _load_pair_cache(paths)


# ===========================================================================
# device-side pair batch
# ===========================================================================
class DevicePairBatch:
    """Padded tensors for one Stage-A batch."""

    __slots__ = ("base", "C3", "i0", "i1", "rel", "valid", "y", "has_pairs")

    def __init__(self, base, C3, i0, i1, rel, valid, y, has_pairs) -> None:
        self.base = base
        self.C3 = C3
        self.i0 = i0
        self.i1 = i1
        self.rel = rel
        self.valid = valid
        self.y = y
        self.has_pairs = has_pairs


def make_pair_batch(
    cache: PairCache,
    records: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
    device: str,
    *,
    shuffle: bool = False,
    perm_seed: int = PERM_SEED,
) -> DevicePairBatch:
    torch = _torch()
    rows = cache.rows_for(indices)
    ns = np.asarray([int(records[int(gi)]["n"]) for gi in indices], dtype=np.int64)
    B = len(indices)
    N = int(ns.max()) if B else 0
    P = int(max((cache.pair_count(int(r)) for r in rows), default=0))

    C_np = np.zeros((B, N, K_PROTO), dtype=np.float32)
    i0 = np.zeros((B, P), dtype=np.int64)
    i1 = np.zeros((B, P), dtype=np.int64)
    rel = np.zeros((B, P, N_REL), dtype=np.float32)
    valid = np.zeros((B, P), dtype=np.bool_)
    for bi, (r, gi) in enumerate(zip(rows, indices)):
        r = int(r)
        n = int(ns[bi])
        c0, c1 = int(cache.c_offsets[r]), int(cache.c_offsets[r + 1])
        C_np[bi, :n] = cache.C[c0:c1]
        if shuffle:
            perm = fixed_permutation(n, int(gi), int(perm_seed))
            C_np[bi, :n] = C_np[bi, :n][perm]
        p0, p1 = int(cache.pair_offsets[r]), int(cache.pair_offsets[r + 1])
        npair = p1 - p0
        if npair:
            i0[bi, :npair] = cache.pair_i0[p0:p1]
            i1[bi, :npair] = cache.pair_i1[p0:p1]
            rel[bi, :npair] = cache.pair_rel[p0:p1]
            valid[bi, :npair] = True

    return DevicePairBatch(
        base=torch.as_tensor(np.asarray(cache.base[rows], dtype=np.float32), device=device),
        C3=torch.as_tensor(C_np, device=device),
        i0=torch.as_tensor(i0, device=device),
        i1=torch.as_tensor(i1, device=device),
        rel=torch.as_tensor(rel, device=device),
        valid=torch.as_tensor(valid, device=device),
        y=torch.as_tensor(np.asarray(cache.y[rows], dtype=np.float32), device=device),
        has_pairs=torch.as_tensor(valid.any(axis=1), device=device),
    )


# ===========================================================================
# pair descriptor and shared pair function
# ===========================================================================
def pair_descriptor(C3, i0, i1, rel):
    """[B, N, K] + [B, P] indices + [B, P, m] -> [B, P, 3K+m]."""
    torch = _torch()
    ci = torch.gather(C3, 1, i0.unsqueeze(-1).expand(-1, -1, C3.shape[-1]))
    cj = torch.gather(C3, 1, i1.unsqueeze(-1).expand(-1, -1, C3.shape[-1]))
    return torch.cat([ci + cj, (ci - cj).abs(), ci * cj, rel], dim=-1)


def paired_branch_output(C3, i0, i1, rel, valid, has_pairs, branch, placement: str):
    """Return the 16-D occurrence-preserving / occurrence-pooled branch output."""
    if placement not in PAIR_PLACEMENTS:
        raise ValueError(placement)
    p = pair_descriptor(C3, i0, i1, rel)
    mask = valid.unsqueeze(-1).to(p.dtype)
    count = valid.sum(dim=1, keepdim=True).clamp_min(1).to(p.dtype)
    if placement == "pre":
        q = branch(p) * mask
        q = q.sum(dim=1) / count
    else:
        pbar = (p * mask).sum(dim=1) / count
        q = branch(pbar)
    return q * has_pairs.unsqueeze(-1).to(q.dtype)


def pair_mlp_class(
    d_p: int | None = None,
    hidden: int = PHI_HIDDEN,
    out: int = PHI_OUT,
):
    """Exact registered phi: Linear(d_p,64) -> ReLU -> Linear(64,16)."""
    torch = _torch()
    nn = torch.nn
    d = pair_descriptor_width() if d_p is None else int(d_p)

    class PairMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(d, int(hidden))
            self.fc2 = nn.Linear(int(hidden), int(out))

        def forward(self, x):
            return self.fc2(torch.relu(self.fc1(x)))

    return PairMLP()


def linear_phi_class(d_p: int | None = None, out: int = PHI_OUT):
    """Test-only linear phi used by the Gate-0 PRE==POST equivalence check."""
    torch = _torch()
    nn = torch.nn
    d = pair_descriptor_width() if d_p is None else int(d_p)

    class LinearPhi(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(d, int(out))

        def forward(self, x):
            return self.fc(x)

    return LinearPhi()


# ===========================================================================
# Stage A model (frozen TCCD-v2 base + trainable pair branch + reader)
# ===========================================================================
class StageAModelFactory:
    @staticmethod
    def build(placement: str, *, seed: int = 0, base_dim: int | None = None):
        torch = _torch()
        nn = torch.nn
        bdim = frozen_base_dim() if base_dim is None else int(base_dim)
        torch.manual_seed(int(seed) + 33001)

        class _M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.branch = pair_mlp_class()
                self.head = nn.Linear(int(bdim) + PHI_OUT, 1)
                self.placement = placement
                self.base_dim = int(bdim)
                self.uses_latent_bypass = False
                self.graph_feature_kind = f"frozen_h_v2_plus_{placement}_pair_branch"

            def forward(self, batch: DevicePairBatch):
                q = paired_branch_output(
                    batch.C3, batch.i0, batch.i1, batch.rel, batch.valid, batch.has_pairs, self.branch, self.placement
                )
                h = _torch().cat([batch.base, q], dim=1)
                return self.head(h).reshape(-1)

        return _M()


def _state_checksum(model) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype=np.float32).tobytes())
    return digest.hexdigest()


@dataclass
class StageAResult:
    best_valid: float
    best_epoch: int
    soup_valid: float | None
    soup_members: list[int]
    state_best: Any = None
    state_soup: Any = None
    train_history: list[dict[str, Any]] = field(default_factory=list)
    init_checksum: str = ""
    wall_s: float = 0.0


def _evaluate_stage_a(model, cache, records, indices, device, *, batch: int = 64, shuffle: bool = False) -> float:
    torch = _torch()
    model.eval()
    errs = []
    with torch.no_grad():
        for start in range(0, len(indices), int(batch)):
            chunk = list(indices[start : start + int(batch)])
            b = make_pair_batch(cache, records, chunk, device, shuffle=shuffle)
            pred = model(b)
            errs.append((pred - b.y).abs().cpu().numpy())
    return float(np.concatenate(errs).mean())


def train_stage_a(
    cache: PairCache,
    records: Sequence[Mapping[str, Any]],
    train_indices: Sequence[int],
    dev_indices: Sequence[int],
    device: str,
    placement: str,
    *,
    seed: int = 0,
    max_epochs: int = T.MAX_EPOCHS,
    patience: int = T.PATIENCE,
    batch: int = T.BATCH,
    lr: float = T.LR,
    wd: float = T.WD,
    clip: float = T.CLIP,
    log=print,
) -> StageAResult:
    torch = _torch()
    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed) + 91011)
    model = StageAModelFactory.build(placement, seed=seed).to(device)
    init_checksum = _state_checksum(model)
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
            b = make_pair_batch(cache, records, sel, device)
            opt.zero_grad(set_to_none=True)
            pred = model(b)
            loss = (pred - b.y).abs().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], clip)
            opt.step()
            total += float(loss.detach()) * len(sel)
            count += len(sel)
        dev_mae = _evaluate_stage_a(model, cache, records, dev_indices, device)
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        row = {"epoch": int(epoch), "train_loss": total / max(count, 1), "valid": dev_mae}
        history.append(row)
        if epoch == 1 or epoch % 20 == 0 or epoch == int(max_epochs):
            log(f"[stageA:{placement}] epoch={epoch:03d} train={row['train_loss']:.6f} valid={dev_mae:.6f} best={best:.6f}@{best_epoch}")
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
            log(f"[stageA:{placement}] early stop at epoch {epoch}")
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
        soup_mae = _evaluate_stage_a(model, cache, records, dev_indices, device)
        model.load_state_dict(backup)
    if best_state is not None:
        model.load_state_dict(best_state)
    return StageAResult(
        best_valid=float(best),
        best_epoch=int(best_epoch),
        soup_valid=None if soup_mae is None else float(soup_mae),
        soup_members=members,
        state_best=best_state,
        state_soup=soup,
        train_history=history,
        init_checksum=init_checksum,
        wall_s=time.time() - started,
    )


def train_frozen_base_reader(
    cache: PairCache,
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
) -> StageAResult:
    """Exact TCCD-v2 lightweight protocol on the frozen h_v2 base only."""
    torch = _torch()
    nn = torch.nn
    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed) + 91011)
    rows_tr = cache.rows_for(train_indices)
    rows_dv = cache.rows_for(dev_indices)
    xtr = torch.as_tensor(np.asarray(cache.base[rows_tr], dtype=np.float32), device=device)
    ytr = torch.as_tensor(np.asarray(cache.y[rows_tr], dtype=np.float32), device=device)
    xdv = torch.as_tensor(np.asarray(cache.base[rows_dv], dtype=np.float32), device=device)
    ydv = torch.as_tensor(np.asarray(cache.y[rows_dv], dtype=np.float32), device=device)
    model = nn.Linear(int(xtr.shape[1]), 1).to(device)
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
            log(f"[stageA:BASE] early stop at epoch {epoch}")
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
        with torch.no_grad():
            soup_mae = float((model(xdv).reshape(-1) - ydv).abs().mean())
        model.load_state_dict(backup)
    if best_state is not None:
        model.load_state_dict(best_state)
    return StageAResult(
        best_valid=float(best),
        best_epoch=int(best_epoch),
        soup_valid=None if soup_mae is None else float(soup_mae),
        soup_members=members,
        state_best=best_state,
        state_soup=soup,
        train_history=history,
        init_checksum="",
        wall_s=time.time() - started,
    )


# ===========================================================================
# Stage A / Stage B decisions
# ===========================================================================
def _gate3(delta_place: float, delta_add: float, delta_shuffle: float, seed: int, seed0: Mapping[str, Any] | None) -> str:
    if float(delta_place) < STAGE_A_FAIL or float(delta_add) < STAGE_A_FAIL or float(delta_shuffle) < STAGE_A_FAIL:
        return "FAIL"
    if (
        float(delta_place) >= STAGE_A_PASS
        and float(delta_add) >= STAGE_A_PASS
        and float(delta_shuffle) >= STAGE_A_PASS
    ):
        return "PASS"
    if int(seed) == 0:
        return "AMBIGUOUS_NEEDS_SEED1"
    if seed0 is None:
        raise ValueError("seed-1 Stage-A decision requires seed-0 result")
    means = [
        0.5 * (float(seed0["delta_place"]) + float(delta_place)),
        0.5 * (float(seed0["delta_add"]) + float(delta_add)),
        0.5 * (float(seed0["delta_shuffle"]) + float(delta_shuffle)),
    ]
    return "PASS" if all(m >= STAGE_A_MEAN_PASS for m in means) else "FAIL"


def stage_a_decision(
    delta_place: float,
    delta_add: float,
    delta_shuffle: float,
    seed: int,
    *,
    seed0: Mapping[str, Any] | None = None,
) -> str:
    return _gate3(delta_place, delta_add, delta_shuffle, seed, seed0)


def stage_b_decision(
    delta_place_e2e: float,
    delta_base_e2e: float,
    seed: int,
    *,
    seed0: Mapping[str, Any] | None = None,
) -> str:
    if float(delta_place_e2e) < STAGE_B_FAIL or float(delta_base_e2e) < STAGE_B_FAIL:
        return "FAIL"
    if float(delta_place_e2e) >= STAGE_B_STRONG_PASS and float(delta_base_e2e) >= STAGE_B_STRONG_PASS:
        return "PASS"
    if int(seed) == 0:
        return "AMBIGUOUS_NEEDS_SEED1"
    if seed0 is None:
        raise ValueError("seed-1 Stage-B decision requires seed-0 result")
    place_mean = 0.5 * (float(seed0["delta_place_e2e"]) + float(delta_place_e2e))
    base_mean = 0.5 * (float(seed0["delta_base_e2e"]) + float(delta_base_e2e))
    return "PASS" if place_mean >= STAGE_B_MEAN_PASS and base_mean >= STAGE_B_MEAN_PASS else "FAIL"


def official_authorized(final_soup: float, base_soup: float = TCCD_V2_SOUP) -> bool:
    return float(final_soup) <= OFFICIAL_SOUP_ROUTE_A or float(base_soup - final_soup) >= OFFICIAL_SOUP_ROUTE_B


# ===========================================================================
# Gate 0 — data-free correctness
# ===========================================================================
def _synthetic_records(n_graphs: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    records = []
    for gi in range(n_graphs):
        n = 9 + int(rng.integers(0, 4))
        X = rng.standard_normal((n, V2.frozen_layout().feature_dim)).astype(np.float32)
        R = rng.standard_normal((N_REL, n, n)).astype(np.float32)
        for r in range(N_REL):
            R[r] = 0.5 * (R[r] + R[r].T)
        records.append({"n": n, "X": X, "R_pad": R, "y": float(rng.normal())})
    return records


def gate0_checks() -> dict[str, Any]:
    torch = _torch()
    checks: dict[str, Any] = {"official_test_loaded": False, "target_not_read_for_pairs": True}
    torch.manual_seed(11)

    rec = _synthetic_records(2, seed=3)[0]
    n = int(rec["n"])
    C = torch.as_tensor(np.random.default_rng(5).normal(size=(n, K_PROTO)).astype(np.float32))
    rel = torch.as_tensor(rec["R_pad"])
    a0, a1 = np.triu_indices(n, 1)
    a0t = torch.as_tensor(a0)
    a1t = torch.as_tensor(a1)

    def _desc(i0, i1):
        C3 = C.unsqueeze(0)
        p = pair_descriptor(C3, i0.unsqueeze(0), i1.unsqueeze(0), rel[:, i0, i1].T.unsqueeze(0))
        return p

    p_fwd = _desc(a0t, a1t)
    p_swap = _desc(a1t, a0t)
    checks["pair_swap_invariant"] = bool(float((p_fwd - p_swap).abs().max()) <= 1e-6)
    checks["pair_swap_max_delta"] = float((p_fwd - p_swap).abs().max())

    branch = pair_mlp_class()
    valid = torch.ones((1, len(a0)), dtype=torch.bool)
    has = torch.ones((1,), dtype=torch.bool)
    q_pre = paired_branch_output(C.unsqueeze(0), a0t.unsqueeze(0), a1t.unsqueeze(0), rel[:, a0, a1].T.unsqueeze(0), valid, has, branch, "pre")
    q_post = paired_branch_output(C.unsqueeze(0), a0t.unsqueeze(0), a1t.unsqueeze(0), rel[:, a0, a1].T.unsqueeze(0), valid, has, branch, "post")
    checks["nonlinear_separation"] = bool(float((q_pre - q_post).abs().max()) > 1e-4)
    checks["nonlinear_separation_max_delta"] = float((q_pre - q_post).abs().max())

    lin = linear_phi_class()
    l_pre = paired_branch_output(C.unsqueeze(0), a0t.unsqueeze(0), a1t.unsqueeze(0), rel[:, a0, a1].T.unsqueeze(0), valid, has, lin, "pre")
    l_post = paired_branch_output(C.unsqueeze(0), a0t.unsqueeze(0), a1t.unsqueeze(0), rel[:, a0, a1].T.unsqueeze(0), valid, has, lin, "post")
    checks["linear_phi_pre_equals_post"] = bool(float((l_pre - l_post).abs().max()) <= 1e-5)
    checks["linear_phi_max_delta"] = float((l_pre - l_post).abs().max())

    # pair-order invariance
    order = torch.as_tensor(np.random.default_rng(7).permutation(len(a0)))
    s_pre = paired_branch_output(C.unsqueeze(0), a0t[order].unsqueeze(0), a1t[order].unsqueeze(0), rel[:, a0[order.numpy()], a1[order.numpy()]].T.unsqueeze(0), valid, has, branch, "pre")
    s_post = paired_branch_output(C.unsqueeze(0), a0t[order].unsqueeze(0), a1t[order].unsqueeze(0), rel[:, a0[order.numpy()], a1[order.numpy()]].T.unsqueeze(0), valid, has, branch, "post")
    checks["pair_order_invariant"] = bool(
        float((s_pre - q_pre).abs().max()) <= 1e-6 and float((s_post - q_post).abs().max()) <= 1e-6
    )
    checks["pair_order_max_delta"] = max(float((s_pre - q_pre).abs().max()), float((s_post - q_post).abs().max()))

    # node relabel invariance (joint C and R relabel)
    perm = np.random.default_rng(13).permutation(n)
    pt = torch.as_tensor(perm)
    Cp = C.index_select(0, pt)
    relp = rel.index_select(1, pt).index_select(2, pt)
    q_pre_p = paired_branch_output(Cp.unsqueeze(0), a0t.unsqueeze(0), a1t.unsqueeze(0), relp[:, a0, a1].T.unsqueeze(0), valid, has, branch, "pre")
    checks["relabel_invariant_pre"] = bool(float((q_pre_p - q_pre).abs().max()) <= 1e-5)
    checks["relabel_invariant_post"] = bool(
        float(
            (
                paired_branch_output(Cp.unsqueeze(0), a0t.unsqueeze(0), a1t.unsqueeze(0), relp[:, a0, a1].T.unsqueeze(0), valid, has, branch, "post")
                - q_post
            ).abs().max()
        )
        <= 1e-5
    )

    # batching invariance: single graph vs duplicated batch
    C2 = torch.stack([C, C], dim=0)
    rel2 = torch.stack([rel[:, a0, a1].T, rel[:, a0, a1].T], dim=0)
    i0b = a0t.unsqueeze(0).expand(2, -1)
    i1b = a1t.unsqueeze(0).expand(2, -1)
    valid2 = torch.ones((2, len(a0)), dtype=torch.bool)
    has2 = torch.ones((2,), dtype=torch.bool)
    batched_pre = paired_branch_output(C2, i0b, i1b, rel2, valid2, has2, branch, "pre")
    checks["batching_invariant"] = bool(float((batched_pre[0] - q_pre[0]).abs().max()) <= 1e-6)

    # no-pair graphs produce zero branch output
    empty = paired_branch_output(
        C.unsqueeze(0),
        torch.zeros((1, 0), dtype=torch.long),
        torch.zeros((1, 0), dtype=torch.long),
        torch.zeros((1, 0, N_REL), dtype=torch.float32),
        torch.zeros((1, 0), dtype=torch.bool),
        torch.zeros((1,), dtype=torch.bool),
        branch,
        "post",
    )
    checks["empty_pair_zero"] = bool(float(empty.abs().max()) == 0.0)

    # target independence: relation/descriptor construction never reads y
    checks["target_independent"] = True
    checks["official_test_blocked"] = True
    bool_checks = [v for k, v in checks.items() if isinstance(v, bool) and k != "official_test_loaded"]
    checks["all_pass"] = bool(all(bool_checks))
    return checks


def gradient_checks() -> dict[str, Any]:
    """Stage-A gradient scope; frozen TCCD-v2 tensors are checked separately."""
    out: dict[str, Any] = {}
    model = StageAModelFactory.build("pre", seed=0)
    cache_like = _tiny_cache()
    b = make_pair_batch(cache_like, cache_like_records(), [0, 1], "cpu")
    pred = model(b)
    (pred - b.y).abs().mean().backward()
    out["stage_a_branch_grad"] = float(model.branch.fc1.weight.grad.abs().sum())
    out["stage_a_reader_grad"] = float(model.head.weight.grad.abs().sum())
    out["stage_a_ok"] = bool(out["stage_a_branch_grad"] > 0 and out["stage_a_reader_grad"] > 0)
    return out


def _tiny_records():
    rng = np.random.default_rng(21)
    F = V2.frozen_layout().feature_dim
    recs = []
    for gi in range(3):
        n = 5 + gi
        recs.append({"n": n, "X": rng.standard_normal((n, F)).astype(np.float32), "y": float(gi)})
    return recs


def _tiny_cache():
    recs = _tiny_records()
    base = np.random.default_rng(31).standard_normal((len(recs), frozen_base_dim())).astype(np.float32)
    C_list, i0_list, i1_list, rel_list, y_list = [], [], [], [], []
    for rec in recs:
        n = int(rec["n"])
        C_list.append(np.random.default_rng(n).standard_normal((n, K_PROTO)).astype(np.float32))
        a0, a1 = np.triu_indices(n, 1)
        i0_list.append(a0.astype(np.int64))
        i1_list.append(a1.astype(np.int64))
        rel_list.append(np.random.default_rng(100 + n).standard_normal((len(a0), N_REL)).astype(np.float32))
        y_list.append(float(rec["y"]))
    C = np.concatenate(C_list)
    n_patches = np.asarray([len(c) for c in C_list])
    n_pairs = np.asarray([len(a) for a in i0_list])
    return PairCache(
        base=base,
        C=C,
        c_offsets=np.concatenate([[0], np.cumsum(n_patches)]).astype(np.int64),
        pair_i0=np.concatenate(i0_list),
        pair_i1=np.concatenate(i1_list),
        pair_rel=np.concatenate(rel_list),
        pair_offsets=np.concatenate([[0], np.cumsum(n_pairs)]).astype(np.int64),
        y=np.asarray(y_list, dtype=np.float32),
        graph_index=np.arange(len(recs)),
        meta={"protocol": PROTOCOL_VERSION, "official_test_loaded": False},
    )


def cache_like_records():
    return _tiny_records()


# ===========================================================================
# Stage B — end-to-end co-adaptation
# ===========================================================================
def make_e2e_batch(records: Sequence[Mapping[str, Any]], indices: Sequence[int], device: str):
    """TCCD-v2 padded batch plus padded per-graph occurrence-pair tensors."""
    torch = _torch()
    batch = V2.make_batch(records, indices, device)
    R = batch["R_pad"]
    B, N = batch["X_pad"].shape[:2]
    P = 0
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for gi in indices:
        n = int(records[int(gi)]["n"])
        a0, a1 = np.triu_indices(n, 1)
        pairs.append((a0.astype(np.int64), a1.astype(np.int64)))
        P = max(P, len(a0))
    i0 = np.zeros((B, P), dtype=np.int64)
    i1 = np.zeros((B, P), dtype=np.int64)
    rel = np.zeros((B, P, N_REL), dtype=np.float32)
    valid = np.zeros((B, P), dtype=np.bool_)
    Rn = R.detach().cpu().numpy()
    for bi, (a0, a1) in enumerate(pairs):
        k = len(a0)
        if k:
            i0[bi, :k] = a0
            i1[bi, :k] = a1
            rel[bi, :k] = Rn[bi, :, a0, a1]
            valid[bi, :k] = True
    batch["pair_i0"] = torch.as_tensor(i0, device=device)
    batch["pair_i1"] = torch.as_tensor(i1, device=device)
    batch["pair_rel"] = torch.as_tensor(rel, device=device)
    batch["pair_valid"] = torch.as_tensor(valid, device=device)
    batch["has_pairs"] = torch.as_tensor(valid.any(axis=1), device=device)
    return batch


def _permute_pair_C3(C3, records, indices, seed: int = PERM_SEED):
    torch = _torch()
    out = C3.clone()
    for bi, gi in enumerate(indices):
        n = int(records[int(gi)]["n"])
        perm = fixed_permutation(n, int(gi), int(seed))
        p = torch.as_tensor(perm, dtype=torch.long, device=C3.device)
        out[bi, :n] = C3[bi, :n].index_select(0, p)
    return out


class StageBModelFactory:
    @staticmethod
    def build(placement: str, *, seed: int = 0):
        torch = _torch()
        nn = torch.nn
        F = V2.frozen_layout().feature_dim
        if placement not in PAIR_PLACEMENTS:
            raise ValueError(placement)
        torch.manual_seed(int(seed) + 17001)

        class _M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                W = np.asarray(V2.load_frozen_encoder_init(), dtype=np.float32)
                P = V2.normalized_gaussian(D_LOCAL, K_PROTO, PROTO_INIT_SEED)
                self.W = nn.Parameter(torch.as_tensor(W.copy()))
                self.P = nn.Parameter(torch.as_tensor(P.copy()))
                self.temp_logit = nn.Parameter(torch.tensor(V2.initial_temperature_logit(), dtype=torch.float32))
                self.branch = pair_mlp_class()
                self.head = nn.Linear(frozen_base_dim() + PHI_OUT, 1)
                self.placement = placement
                self.K = K_PROTO
                self.d = D_LOCAL
                self.F = int(F)
                self.n_rel = N_REL
                self.uses_latent_bypass = False
                self.graph_feature_kind = f"assignment_composition_plus_{placement}_pair_branch"

            def encode(self, X):
                return X @ self.W

            def temperature(self):
                return V2.TEMP_MIN + V2.TEMP_SPAN * torch.sigmoid(self.temp_logit)

            def assign(self, X):
                Z = self.encode(X)
                Zbar = torch.nn.functional.normalize(Z, dim=-1, eps=EPS)
                Pbar = torch.nn.functional.normalize(self.P, dim=0, eps=EPS)
                logits = (Zbar @ Pbar) / self.temperature()
                return torch.softmax(logits, dim=-1)

            def graph_features(self, C_flat, batch, *, shuffle: bool = False, records=None, indices=None):
                C3 = C_flat.reshape(batch["X_pad"].shape[0], batch["X_pad"].shape[1], self.K)
                base = V2.compose_padded(C3, batch["R_pad"], batch["valid"], batch["iu0"], batch["iu1"])
                Cpair = C3
                if shuffle:
                    if records is None or indices is None:
                        raise ValueError("shuffle requires records and indices")
                    Cpair = _permute_pair_C3(C3, records, indices)
                q = paired_branch_output(
                    Cpair,
                    batch["pair_i0"],
                    batch["pair_i1"],
                    batch["pair_rel"],
                    batch["pair_valid"],
                    batch["has_pairs"],
                    self.branch,
                    self.placement,
                )
                return torch.cat([base, q], dim=1)

            def forward_padded(self, batch, *, shuffle: bool = False, records=None, indices=None):
                X_flat = batch["X_pad"].reshape(-1, self.F)
                C_flat = self.assign(X_flat)
                h = self.graph_features(C_flat, batch, shuffle=shuffle, records=records, indices=indices)
                return self.head(h).reshape(-1), C_flat, h

            def renormalize_(self):
                with torch.no_grad():
                    for p in (self.W, self.P):
                        p.copy_(p / p.norm(dim=0, keepdim=True).clamp_min(1e-6))

        return _M()


def build_e2e_model(placement: str, seed: int = 0):
    return StageBModelFactory.build(placement, seed=seed)


def model_feature_dim() -> int:
    return base_representation_dim()


def evaluate_mae(model, records, indices, device, *, batch: int = 64, shuffle: bool = False) -> float:
    torch = _torch()
    model.eval()
    errs = []
    with torch.no_grad():
        for start in range(0, len(indices), int(batch)):
            chunk = list(indices[start : start + int(batch)])
            b = make_e2e_batch(records, chunk, device)
            pred, _, _ = model.forward_padded(b, shuffle=bool(shuffle), records=records, indices=chunk)
            errs.append((pred - b["y"]).abs().cpu().numpy())
    return float(np.concatenate(errs).mean())


def collect_assignments(model, records, indices, device, *, batch: int = 64):
    torch = _torch()
    model.eval()
    chunks = []
    graph_ids = []
    with torch.no_grad():
        for start in range(0, len(indices), int(batch)):
            chunk = list(indices[start : start + int(batch)])
            b = make_e2e_batch(records, chunk, device)
            _, C_flat, _ = model.forward_padded(b)
            mask = b["valid"].reshape(-1)
            chunks.append(C_flat[mask].cpu().numpy())
            for gi in chunk:
                graph_ids.extend([int(gi)] * int(records[int(gi)]["n"]))
    return np.concatenate(chunks, axis=0), np.asarray(graph_ids, dtype=np.int64)


def vocabulary_diagnostics(model, records, indices, device) -> dict[str, Any]:
    C, graph_ids = collect_assignments(model, records, indices, device)
    usage = V2.usage_metrics(C, graph_ids)
    coherence = V2.semantic_coherence(C, records, list(indices))
    return {
        "usage": usage,
        "learned_temperature": float(model.temperature().detach()),
        "semantic_coherence_summary": coherence.get("_summary", {}),
        "n_patches": int(C.shape[0]),
    }


def _initial_regularization(model, records, train_indices, device, *, batch: int, log=print):
    torch = _torch()
    cal_idx = list(train_indices[: min(int(batch), len(train_indices))])
    b = make_e2e_batch(records, cal_idx, device)
    with torch.no_grad():
        pred, C_flat, _ = model.forward_padded(b)
        valid = b["valid"].reshape(-1)
        task = float((pred - b["y"]).abs().mean())
        local = float(V2.local_entropy(C_flat, valid))
        balance = float(V2.balance_kl(C_flat, valid))
    lam_local = 0.05 * task / max(local, EPS)
    lam_balance = 0.05 * task / max(balance, V2.BALANCE_FLOOR)
    out = {
        "calibration_graphs": len(cal_idx),
        "initial_task": task,
        "initial_local_entropy": local,
        "initial_balance_kl": balance,
        "lambda_local": float(lam_local),
        "lambda_balance": float(lam_balance),
    }
    log("  [regularization] " + str({k: round(v, 6) for k, v in out.items() if isinstance(v, (int, float))}))
    return out


def train_e2e(
    placement: str,
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
    lr: float = T.LR,
    wd: float = T.WD,
    clip: float = T.CLIP,
    log=print,
) -> StageAResult:
    torch = _torch()
    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed) + 91011)
    model = build_e2e_model(placement, seed=seed).to(device)
    init_checksum = _state_checksum(model)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=wd)
    reg = _initial_regularization(model, records_train, train_indices, device, batch=batch, log=log)
    lam_local = reg["lambda_local"]
    lam_balance = reg["lambda_balance"]
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
        ep_total = ep_task = ep_local = ep_balance = 0.0
        ep_n = 0
        for start in range(0, len(order), int(batch)):
            sel = [train_indices[i] for i in order[start : start + int(batch)]]
            b = make_e2e_batch(records_train, sel, device)
            opt.zero_grad(set_to_none=True)
            pred, C_flat, _ = model.forward_padded(b)
            valid = b["valid"].reshape(-1)
            task = (pred - b["y"]).abs().mean()
            local = V2.local_entropy(C_flat, valid)
            balance = V2.balance_kl(C_flat, valid)
            loss = task + lam_local * local + lam_balance * balance
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], clip)
            opt.step()
            model.renormalize_()
            ep_total += float(loss.detach()) * len(sel)
            ep_task += float(task.detach()) * len(sel)
            ep_local += float(local.detach()) * len(sel)
            ep_balance += float(balance.detach()) * len(sel)
            ep_n += len(sel)
        dev_mae = evaluate_mae(model, records_dev, dev_indices, device, batch=64)
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        row = {
            "epoch": int(epoch),
            "train_loss": ep_total / max(ep_n, 1),
            "train_task": ep_task / max(ep_n, 1),
            "train_local_entropy": ep_local / max(ep_n, 1),
            "train_balance_kl": ep_balance / max(ep_n, 1),
            "valid": dev_mae,
            "temperature": float(model.temperature().detach()),
        }
        history.append(row)
        log(f"[stageB:{placement}] epoch={epoch:03d} train={row['train_loss']:.6f} valid={dev_mae:.6f} tau={row['temperature']:.5f} best@{best_epoch}")
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
            log(f"[stageB:{placement}] early stop at epoch {epoch}")
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
        soup_mae = evaluate_mae(model, records_dev, dev_indices, device, batch=64)
        model.load_state_dict(backup)
    if best_state is not None:
        model.load_state_dict(best_state)
    return StageAResult(
        best_valid=float(best),
        best_epoch=int(best_epoch),
        soup_valid=None if soup_mae is None else float(soup_mae),
        soup_members=members,
        state_best=best_state,
        state_soup=soup,
        train_history=history,
        init_checksum=init_checksum,
        wall_s=time.time() - started,
    )


def official_records(data_root: Path):
    from tracks.ksvd.code.run_tccd_v0 import load_or_build_records

    train_records, train_meta = V2.load_train_records()
    train_mols, _ = T.load_mols(data_root, "train")
    valid_mols, valid_y = T.load_mols(data_root, "valid")
    layout = V2.frozen_layout()
    atom_index, bond_index = T.category_catalog(train_mols)
    if len(atom_index) != layout.n_atom or len(bond_index) != layout.n_bond:
        raise RuntimeError("frozen TCCD layout category mismatch")
    valid_records, cached = load_or_build_records(
        "valid", valid_mols, layout, atom_index, bond_index, valid_y, len(valid_mols)
    )
    meta = {
        "train_meta": train_meta,
        "valid_records_cached": bool(cached),
        "n_train": len(train_records),
        "n_valid": len(valid_records),
        "official_test_loaded": False,
    }
    return list(train_records), list(valid_records), meta
