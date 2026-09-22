#!/usr/bin/env python
"""TCCD-v4 higher-order assembly audit.

This module preserves the exact TCCD-v2 local encoder, prototype vocabulary,
existing relation set, optimizer, regularizers and reader mechanics.  It adds
only the parameter-free normalized two-hop walk relation R2=S^2, together with
a fixed graph-specific assignment permutation used by the matched topology
mismatch control.

Official ZINC test is never loaded.
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
from tracks.ksvd.code.run_tccd_v0 import load_or_build_records

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = REPO_ROOT / "tracks/ksvd/results/tccd_v4"
CACHE_DIR = RESULTS_DIR / "cache"
PROTOCOL_VERSION = "tccd_v4_higher_order_assembly_v1"

D_LOCAL = V2.D_LOCAL
K_PROTO = V2.K_PROTO
N_REL = V2.N_REL
TEMP_MIN = V2.TEMP_MIN
TEMP_SPAN = V2.TEMP_SPAN
TEMP_INIT = V2.TEMP_INIT
EPS = V2.EPS
PROTO_INIT_SEED = V2.PROTO_INIT_SEED
PERM_SEED = 20260922

TCCD_V2_COMMIT = "69a985a4ea3eb184c96c8ddad3857c4e87ee1dda"
TCCD_V2_BEST = 0.2862437069416046
TCCD_V2_SOUP = 0.26635152101516724
TCCD_V2_OFFICIAL_VALID_BEST = 0.287337
TCCD_V2_OFFICIAL_VALID_SOUP = 0.261988
CANONICAL_GPU1_BASELINE = 0.119818

V2_BEST_CHECKPOINT = V2.RESULTS_DIR / "prototype_rel_seed0_best.pt"
V2_BEST_CHECKPOINT_SHA256 = "093e0e46d4d5dab7e5558b807d993d1136df58d8f531d6b9b3621e8ab05c0f42"

STAGE_A_TOPO_PASS = 0.015
STAGE_A_FAIL = 0.005
STAGE_A_MEAN_PASS = 0.010
STAGE_B_STRONG_PASS = 0.020
STAGE_B_FAIL = 0.005
STAGE_B_MEAN_PASS = 0.010
STAGE_C_PASS = 0.010
OFFICIAL_SOUP_ROUTE_A = 0.24
OFFICIAL_SOUP_ROUTE_B = 0.030

READER_MAX_EPOCHS = T.MAX_EPOCHS
READER_PATIENCE = T.PATIENCE
READER_BATCH = T.BATCH
READER_LR = T.LR
READER_WD = T.WD
READER_CLIP = T.CLIP
READER_TOP_K = T.TOP_K_SOUP


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


def initial_temperature_logit() -> float:
    q = (TEMP_INIT - TEMP_MIN) / TEMP_SPAN
    return float(math.log(q / (1.0 - q)))


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


def native_untyped_adjacency(rec: Mapping[str, Any]) -> np.ndarray:
    """Native atom adjacency from bond incidence, with no self-loop."""
    rb = np.asarray(rec["Rb"], dtype=np.float32)
    if rb.ndim != 3:
        raise ValueError(f"expected Rb [n_bond,n,n], got {rb.shape}")
    A = (np.sum(rb, axis=0) > 0).astype(np.float32)
    np.fill_diagonal(A, 0.0)
    return A


def normalized_adjacency(rec: Mapping[str, Any]) -> np.ndarray:
    """S=D^{-1/2}AD^{-1/2}, using a numerically safe zero inverse degree."""
    A = native_untyped_adjacency(rec)
    degree = A.sum(axis=1)
    inv_sqrt = np.zeros_like(degree, dtype=np.float32)
    nonzero = degree > 0
    inv_sqrt[nonzero] = 1.0 / np.sqrt(degree[nonzero])
    return inv_sqrt[:, None] * A * inv_sqrt[None, :]


def walk_operator(rec: Mapping[str, Any], order: int = 2) -> np.ndarray:
    """Return the full normalized walk operator S^order, including diagonal."""
    if int(order) < 1:
        raise ValueError("walk order must be positive")
    S = normalized_adjacency(rec)
    out = S.copy()
    for _ in range(1, int(order)):
        out = out @ S
    return np.asarray(out, dtype=np.float32)


def walk_moment_torch(C, R, valid, iu0, iu1):
    """Batched vec_sym(C^T R C) using the exact TCCD-v2 convention."""
    torch = _torch()
    C = C * valid.unsqueeze(-1).to(C.dtype)
    M = torch.einsum("bnk,bnm,bml->bkl", C, R, C)
    return M[:, iu0, iu1]


def _apply_fixed_permutation(C3, records, indices, seed: int, device):
    torch = _torch()
    out = C3.clone()
    for bi, gi in enumerate(indices):
        n = int(records[int(gi)]["n"])
        perm = fixed_permutation(n, int(gi), int(seed))
        p = torch.as_tensor(perm, dtype=torch.long, device=device)
        out[bi, :n] = C3[bi, :n].index_select(0, p)
    return out


def make_batch(
    records: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
    device: str,
    *,
    max_order: int = 2,
):
    """Reuse TCCD-v2 padded batch and append fixed R2/R3 tensors."""
    torch = _torch()
    batch = V2.make_batch(records, indices, device)
    B, N = batch["X_pad"].shape[:2]
    if int(max_order) >= 2:
        r2 = np.zeros((B, N, N), dtype=np.float32)
    else:
        r2 = None
    if int(max_order) >= 3:
        r3 = np.zeros((B, N, N), dtype=np.float32)
    else:
        r3 = None
    for bi, gi in enumerate(indices):
        rec = records[int(gi)]
        n = int(rec["n"])
        if r2 is not None:
            r2[bi, :n, :n] = walk_operator(rec, 2)
        if r3 is not None:
            r3[bi, :n, :n] = walk_operator(rec, 3)
    if r2 is not None:
        batch["R2_pad"] = torch.as_tensor(r2, dtype=torch.float32, device=device)
    if r3 is not None:
        batch["R3_pad"] = torch.as_tensor(r3, dtype=torch.float32, device=device)
    return batch


def _base_features(C3, batch):
    return V2.compose_padded(C3, batch["R_pad"], batch["valid"], batch["iu0"], batch["iu1"])


def assembly_features(
    C_flat,
    batch,
    *,
    records: Sequence[Mapping[str, Any]] | None = None,
    indices: Sequence[int] | None = None,
    mismatch: bool = False,
    perm_seed: int = PERM_SEED,
    max_order: int = 2,
):
    """Return [h_v2, M2] or [h_v2, M2, M3]."""
    if records is None or indices is None:
        if mismatch:
            raise ValueError("mismatch assembly requires records and indices")
    C3 = C_flat.reshape(batch["X_pad"].shape[0], batch["X_pad"].shape[1], K_PROTO)
    base = _base_features(C3, batch)
    parts = [base]
    if int(max_order) >= 2:
        C2 = _apply_fixed_permutation(C3, records, indices, perm_seed, C_flat.device) if mismatch else C3
        parts.append(walk_moment_torch(C2, batch["R2_pad"], batch["valid"], batch["iu0"], batch["iu1"]))
    if int(max_order) >= 3:
        # Stage C uses real assignments for both higher-order blocks.
        parts.append(walk_moment_torch(C3, batch["R3_pad"], batch["valid"], batch["iu0"], batch["iu1"]))
    return _torch().cat(parts, dim=1)


def model_feature_dim(max_order: int) -> int:
    tri = K_PROTO * (K_PROTO + 1) // 2
    return int(T.h_dim(K_PROTO, N_REL) + max(0, int(max_order) - 1) * tri)


class PrototypeAssemblyModelFactory:
    @staticmethod
    def build(
        F: int,
        d: int,
        K: int,
        n_rel: int,
        encoder_init: np.ndarray,
        *,
        max_order: int = 2,
        proto_seed: int = PROTO_INIT_SEED,
        seed: int = 0,
    ):
        torch = _torch()
        nn = torch.nn
        H = int(T.h_dim(K, n_rel) + max(0, int(max_order) - 1) * (K * (K + 1) // 2))
        torch.manual_seed(int(seed) + 17001)

        class _M(nn.Module):
            def __init__(self):
                super().__init__()
                W = np.asarray(encoder_init, dtype=np.float32)
                if W.shape != (F, d):
                    raise ValueError(f"encoder init shape {W.shape} != {(F, d)}")
                P = V2.normalized_gaussian(d, K, int(proto_seed))
                self.W = nn.Parameter(torch.as_tensor(W.copy()))
                self.P = nn.Parameter(torch.as_tensor(P.copy()))
                self.temp_logit = nn.Parameter(torch.tensor(initial_temperature_logit(), dtype=torch.float32))
                self.head = nn.Linear(H, 1)
                self.K = int(K)
                self.d = int(d)
                self.F = int(F)
                self.n_rel = int(n_rel)
                self.max_order = int(max_order)
                self.graph_feature_kind = "assignment_composition_plus_twohop" if max_order == 2 else "assignment_composition_plus_twohop_threehop"
                self.uses_latent_bypass = False
                self.proto_seed = int(proto_seed)

            def encode(self, X):
                return X @ self.W

            def temperature(self):
                return TEMP_MIN + TEMP_SPAN * torch.sigmoid(self.temp_logit)

            def assign(self, X):
                Z = self.encode(X)
                Zbar = torch.nn.functional.normalize(Z, dim=-1, eps=EPS)
                Pbar = torch.nn.functional.normalize(self.P, dim=0, eps=EPS)
                logits = (Zbar @ Pbar) / self.temperature()
                return torch.softmax(logits, dim=-1)

            def graph_features(self, C, batch, *, mismatch=False, records=None, indices=None, perm_seed=PERM_SEED):
                return assembly_features(
                    C,
                    batch,
                    records=records,
                    indices=indices,
                    mismatch=bool(mismatch),
                    perm_seed=int(perm_seed),
                    max_order=self.max_order,
                )

            def forward_padded(self, batch, *, mismatch=False, records=None, indices=None, perm_seed=PERM_SEED):
                X_flat = batch["X_pad"].reshape(-1, self.F)
                C_flat = self.assign(X_flat)
                h = self.graph_features(C_flat, batch, mismatch=mismatch, records=records, indices=indices, perm_seed=perm_seed)
                pred = self.head(h).reshape(-1)
                return pred, C_flat, h

            def renormalize_(self):
                with torch.no_grad():
                    for p in (self.W, self.P):
                        p.copy_(p / p.norm(dim=0, keepdim=True).clamp_min(1e-6))

        return _M()


def build_model(max_order: int = 2, seed: int = 0):
    return PrototypeAssemblyModelFactory.build(
        V2.frozen_layout().feature_dim,
        D_LOCAL,
        K_PROTO,
        N_REL,
        V2.load_frozen_encoder_init(),
        max_order=int(max_order),
        proto_seed=PROTO_INIT_SEED,
        seed=seed,
    )


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


def collect_frozen_features(
    records: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
    device: str,
    *,
    batch_size: int = 64,
):
    """Extract h_v2, real M2 and fixed-mismatch M2 for one index set."""
    torch = _torch()
    model = build_frozen_v2_model(device)
    base_rows: list[np.ndarray] = []
    real_rows: list[np.ndarray] = []
    mis_rows: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(indices), int(batch_size)):
            chunk = list(indices[start : start + int(batch_size)])
            b = make_batch(records, chunk, device, max_order=2)
            _, C_flat, h1 = model.forward_padded(b)
            C3 = C_flat.reshape(b["X_pad"].shape[0], b["X_pad"].shape[1], K_PROTO)
            real2 = walk_moment_torch(C3, b["R2_pad"], b["valid"], b["iu0"], b["iu1"])
            Cpi = _apply_fixed_permutation(C3, records, chunk, PERM_SEED, device)
            mis2 = walk_moment_torch(Cpi, b["R2_pad"], b["valid"], b["iu0"], b["iu1"])
            base_rows.append(h1.cpu().numpy().astype(np.float32, copy=True))
            real_rows.append(real2.cpu().numpy().astype(np.float32, copy=True))
            mis_rows.append(mis2.cpu().numpy().astype(np.float32, copy=True))
    return (
        np.concatenate(base_rows, axis=0),
        np.concatenate(real_rows, axis=0),
        np.concatenate(mis_rows, axis=0),
    )


def frozen_cache_paths(split: str) -> dict[str, Path]:
    return {
        "base": CACHE_DIR / f"frozen_{split}_base.npy",
        "real2": CACHE_DIR / f"frozen_{split}_real2.npy",
        "mis2": CACHE_DIR / f"frozen_{split}_mis2.npy",
        "y": CACHE_DIR / f"frozen_{split}_y.npy",
        "meta": CACHE_DIR / f"frozen_{split}_metadata.json",
    }


def build_or_load_frozen_cache(records, indices, split: str, device: str, *, force: bool = False):
    paths = frozen_cache_paths(split)
    if not force and all(path.exists() for path in paths.values()):
        meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
        if (
            meta.get("protocol") == PROTOCOL_VERSION
            and meta.get("v2_checkpoint_sha256") == V2_BEST_CHECKPOINT_SHA256
            and meta.get("perm_seed") == PERM_SEED
            and meta.get("official_test_loaded") is False
        ):
            return {
                "base": np.load(paths["base"], mmap_mode="r"),
                "real2": np.load(paths["real2"], mmap_mode="r"),
                "mis2": np.load(paths["mis2"], mmap_mode="r"),
                "y": np.load(paths["y"], mmap_mode="r"),
                "meta": meta,
            }
    base, real2, mis2 = collect_frozen_features(records, indices, device)
    y = np.asarray([float(records[int(i)]["y"]) for i in indices], dtype=np.float32)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(paths["base"], base)
    np.save(paths["real2"], real2)
    np.save(paths["mis2"], mis2)
    np.save(paths["y"], y)
    meta = {
        "protocol": PROTOCOL_VERSION,
        "split": split,
        "graph_indices": [int(i) for i in indices],
        "n_graphs": len(indices),
        "base_dim": int(base.shape[1]),
        "twohop_dim": int(real2.shape[1]),
        "v2_checkpoint": str(V2_BEST_CHECKPOINT.relative_to(REPO_ROOT)),
        "v2_checkpoint_sha256": V2_BEST_CHECKPOINT_SHA256,
        "v2_commit": TCCD_V2_COMMIT,
        "perm_seed": PERM_SEED,
        "perm_rule": "default_rng((PERM_SEED, graph_index)).permutation(n); identity for n<2",
        "S_definition": "D^-1/2 A D^-1/2; native untyped atom adjacency; no self-loop; zero inverse degree",
        "R2_definition": "S @ S, full matrix including diagonal",
        "official_test_loaded": False,
    }
    write_json(paths["meta"], meta)
    return {"base": base, "real2": real2, "mis2": mis2, "y": y, "meta": meta}


@dataclass
class ReaderResult:
    best_valid: float
    best_epoch: int
    soup_valid: float | None
    soup_members: list[int]
    state_best: Any = None
    state_soup: Any = None
    train_history: list[dict[str, Any]] = field(default_factory=list)
    wall_s: float = 0.0
    peak_mem_mb: float | None = None


def train_frozen_reader(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    device: str,
    *,
    seed: int = 0,
    max_epochs: int = READER_MAX_EPOCHS,
    patience: int = READER_PATIENCE,
    batch: int = READER_BATCH,
    lr: float = READER_LR,
    wd: float = READER_WD,
    clip: float = READER_CLIP,
    log=print,
) -> ReaderResult:
    torch = _torch()
    nn = torch.nn
    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed) + 91011)
    model = nn.Linear(int(X_train.shape[1]), 1).to(device)
    xtr = torch.as_tensor(np.asarray(X_train, dtype=np.float32), device=device)
    ytr = torch.as_tensor(np.asarray(y_train, dtype=np.float32), device=device)
    xdv = torch.as_tensor(np.asarray(X_dev, dtype=np.float32), device=device)
    ydv = torch.as_tensor(np.asarray(y_dev, dtype=np.float32), device=device)
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
            xb = xtr[sel]
            yb = ytr[sel]
            opt.zero_grad(set_to_none=True)
            pred = model(xb).reshape(-1)
            loss = (pred - yb).abs().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            total += float(loss.detach()) * len(sel)
            count += len(sel)
        model.eval()
        with torch.no_grad():
            valid_mae = float((model(xdv).reshape(-1) - ydv).abs().mean())
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        row = {"epoch": int(epoch), "train_loss": total / max(count, 1), "valid": valid_mae}
        history.append(row)
        if valid_mae < best - 1e-9:
            best = valid_mae
            best_epoch = epoch
            best_state = copy.deepcopy(state)
            stale = 0
        else:
            stale += 1
        if len(top) < READER_TOP_K or valid_mae < max(item[0] for item in top):
            top.append((valid_mae, epoch, state))
            top.sort(key=lambda item: (item[0], item[1]))
            top = top[:READER_TOP_K]
        if epoch == 1 or epoch % 20 == 0 or epoch == int(max_epochs):
            log(f"[frozen-reader] epoch={epoch:03d} train={row['train_loss']:.6f} valid={valid_mae:.6f} best={best:.6f}@{best_epoch}")
        if stale >= int(patience):
            break
    soup_state = None
    soup_valid = None
    members: list[int] = []
    if top:
        members = [epoch for _, epoch, _ in top]
        keys = top[0][2].keys()
        soup_state = {key: sum(state[key].float() for _, _, state in top) / len(top) for key in keys}
        backup = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(soup_state)
        with torch.no_grad():
            soup_valid = float((model(xdv).reshape(-1) - ydv).abs().mean())
        model.load_state_dict(backup)
    if best_state is not None:
        model.load_state_dict(best_state)
    return ReaderResult(
        best_valid=float(best),
        best_epoch=int(best_epoch),
        soup_valid=None if soup_valid is None else float(soup_valid),
        soup_members=members,
        state_best=best_state,
        state_soup=soup_state,
        train_history=history,
        wall_s=time.time() - started,
    )


def train_model(
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
    lr: float = T.LR,
    wd: float = T.WD,
    clip: float = T.CLIP,
    log=print,
):
    """TCCD-v2 training loop with only the registered assembly feature append."""
    torch = _torch()
    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed) + 91011)
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
    t0 = time.time()
    for epoch in range(1, int(max_epochs) + 1):
        model.train()
        order = rng.permutation(len(train_indices))
        ep_total = ep_task = ep_local = ep_balance = 0.0
        ep_n = 0
        for start in range(0, len(order), int(batch)):
            sel = [train_indices[i] for i in order[start : start + int(batch)]]
            b = make_batch(records_train, sel, device, max_order=model.max_order)
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
            count = len(sel)
            ep_total += float(loss.detach()) * count
            ep_task += float(task.detach()) * count
            ep_local += float(local.detach()) * count
            ep_balance += float(balance.detach()) * count
            ep_n += count
        dev_mae = evaluate_mae(model, records_dev, dev_indices, device, batch=64)
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
        log(f"  epoch={epoch:03d} train={row['train_loss']:.6f} valid={dev_mae:.6f} tau={row['temperature']:.5f} best@{best_epoch}")
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if dev_mae < best - 1e-9:
            best = dev_mae
            best_epoch = epoch
            best_state = copy.deepcopy(state)
            stale = 0
        else:
            stale += 1
        if len(top) < T.TOP_K_SOUP or dev_mae < max(t[0] for t in top):
            top.append((dev_mae, epoch, state))
            top.sort(key=lambda t: (t[0], t[1]))
            top = top[:T.TOP_K_SOUP]
        if stale >= int(patience):
            log(f"  early stop at epoch {epoch}")
            break
    soup = None
    soup_mae = None
    members: list[int] = []
    if top:
        members = [e for _, e, _ in top]
        keys = top[0][2].keys()
        soup = {k: sum(st[k].float() for _, _, st in top) / len(top) for k in keys}
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(soup)
        soup_mae = evaluate_mae(model, records_dev, dev_indices, device, batch=64)
        model.load_state_dict(backup)
    if best_state is not None:
        model.load_state_dict(best_state)
    return V2.PrototypeTrainResult(
        best_valid=float(best),
        best_epoch=int(best_epoch),
        soup_valid=None if soup_mae is None else float(soup_mae),
        soup_members=members,
        train_history=history,
        state_best=best_state,
        state_soup=soup,
        regularization=reg,
        wall_s=time.time() - t0,
    )


def _initial_regularization(model, records, train_indices, device, *, batch: int, log=print):
    torch = _torch()
    cal_idx = list(train_indices[: min(int(batch), len(train_indices))])
    b = make_batch(records, cal_idx, device, max_order=model.max_order)
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
        "initial_local_contribution": float(lam_local * local),
        "initial_balance_contribution": float(lam_balance * balance),
        "balance_floor": V2.BALANCE_FLOOR,
    }
    log("  [regularization] " + str({k: round(v, 6) for k, v in out.items() if isinstance(v, (int, float))}))
    return out


def evaluate_mae(model, records, indices, device, *, batch: int = 64, mismatch: bool = False, perm_seed: int = PERM_SEED) -> float:
    torch = _torch()
    model.eval()
    errs = []
    with torch.no_grad():
        for start in range(0, len(indices), int(batch)):
            chunk = list(indices[start : start + int(batch)])
            b = make_batch(records, chunk, device, max_order=model.max_order)
            pred, _, _ = model.forward_padded(
                b,
                mismatch=bool(mismatch),
                records=records,
                indices=chunk,
                perm_seed=int(perm_seed),
            )
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
            b = make_batch(records, chunk, device, max_order=model.max_order)
            _, C_flat, _ = model.forward_padded(b)
            valid = b["valid"].reshape(-1)
            chunks.append(C_flat[valid].cpu().numpy())
            for gi in chunk:
                graph_ids.extend([int(gi)] * int(records[gi]["n"]))
    return np.concatenate(chunks, axis=0), np.asarray(graph_ids, dtype=np.int64)


def vocabulary_diagnostics(model, records, indices, device) -> dict[str, Any]:
    C, graph_ids = collect_assignments(model, records, indices, device)
    usage = V2.usage_metrics(C, graph_ids)
    coherence = V2.semantic_coherence(C, records, list(indices))
    return {
        "usage": usage,
        "learned_temperature": float(model.temperature().detach()),
        "semantic_coherence_summary": coherence.get("_summary", {}),
        "semantic_coherence": coherence,
        "n_patches": int(C.shape[0]),
    }


def gate0_checks() -> dict[str, Any]:
    """Data-free correctness checks for S, S^2 and mismatch semantics."""
    torch = _torch()
    checks: dict[str, Any] = {"official_test_loaded": False, "target_not_read_for_r2": True}

    def rec_from_edges(n: int, edges: Sequence[tuple[int, int]], y: float = 0.0):
        rb = np.zeros((1, n, n), dtype=np.float32)
        for a, b in edges:
            rb[0, a, b] = 1.0
            rb[0, b, a] = 1.0
        return {"Rb": rb, "n": n, "y": y}

    path = rec_from_edges(6, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)])
    two_tri = rec_from_edges(6, [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)])
    C = torch.as_tensor(np.random.default_rng(5).normal(size=(6, K_PROTO)).astype(np.float32))
    valid = torch.ones((1, 6), dtype=torch.bool)
    iu0_np, iu1_np = T.sym_indices(K_PROTO)
    iu0 = torch.as_tensor(iu0_np, dtype=torch.long)
    iu1 = torch.as_tensor(iu1_np, dtype=torch.long)

    S = normalized_adjacency(path)
    R2 = walk_operator(path, 2)
    checks["no_self_loop_in_A"] = bool(np.all(np.diag(native_untyped_adjacency(path)) == 0.0))
    checks["safe_zero_degree"] = bool(np.allclose(normalized_adjacency(rec_from_edges(2, [])), 0.0))
    checks["exact_s2"] = bool(np.max(np.abs(R2 - (S @ S))) <= 1e-7)

    perm = np.asarray([2, 5, 1, 4, 0, 3], dtype=np.int64)
    p = torch.as_tensor(perm)
    Rp = torch.as_tensor(R2).index_select(0, p).index_select(1, p)
    h = walk_moment_torch(C.unsqueeze(0), torch.as_tensor(R2).unsqueeze(0), valid, iu0, iu1)
    hp = walk_moment_torch(C.index_select(0, p).unsqueeze(0), Rp.unsqueeze(0), valid, iu0, iu1)
    checks["permutation_invariant"] = float((h - hp).abs().max()) <= 1e-5
    checks["permutation_max_delta"] = float((h - hp).abs().max())

    Cpad = C.unsqueeze(0)
    single = walk_moment_torch(Cpad, torch.as_tensor(R2).unsqueeze(0), valid, iu0, iu1)
    batch_R = torch.stack([torch.as_tensor(R2), torch.as_tensor(R2)], dim=0)
    batch_C = torch.stack([C, C], dim=0)
    batch_valid = torch.ones((2, 6), dtype=torch.bool)
    batched = walk_moment_torch(batch_C, batch_R, batch_valid, iu0, iu1)
    checks["batching_invariant"] = float((single - batched[:1]).abs().max()) <= 1e-6
    checks["batching_max_delta"] = float((single - batched[:1]).abs().max())

    R2_b = walk_operator(two_tri, 2)
    h_b = walk_moment_torch(C.unsqueeze(0), torch.as_tensor(R2_b).unsqueeze(0), valid, iu0, iu1)
    checks["twohop_sensitivity"] = float((h - h_b).abs().max()) > 1e-5
    checks["twohop_sensitivity_max_delta"] = float((h - h_b).abs().max())

    Cpi = C.index_select(0, torch.as_tensor(fixed_permutation(6, 17)))
    h_mis = walk_moment_torch(Cpi.unsqueeze(0), torch.as_tensor(R2).unsqueeze(0), valid, iu0, iu1)
    checks["mismatch_effective"] = float((h - h_mis).abs().max()) > 1e-5
    checks["mismatch_max_delta"] = float((h - h_mis).abs().max())

    path_y = dict(path)
    path_y["y"] = 123.0
    checks["target_independent"] = bool(np.array_equal(walk_operator(path), walk_operator(path_y)))
    checks["official_test_blocked"] = True
    bool_checks = [value for key, value in checks.items() if isinstance(value, bool) and key != "official_test_loaded"]
    checks["all_pass"] = bool(all(bool_checks))
    return checks


def stage_a_decision(delta_topo: float, delta_add: float, seed: int, seed0: Mapping[str, Any] | None = None) -> str:
    if float(delta_topo) < STAGE_A_FAIL or float(delta_add) < STAGE_A_FAIL:
        return "FAIL"
    if float(delta_topo) >= STAGE_A_TOPO_PASS and float(delta_add) >= STAGE_A_TOPO_PASS:
        return "PASS"
    if int(seed) == 0:
        return "AMBIGUOUS_NEEDS_SEED1"
    if seed0 is None:
        raise ValueError("seed-1 Stage-A decision requires seed-0 result")
    topo_mean = 0.5 * (float(seed0["delta_topo"]) + float(delta_topo))
    add_mean = 0.5 * (float(seed0["delta_add"]) + float(delta_add))
    return "PASS" if topo_mean >= STAGE_A_MEAN_PASS and add_mean >= STAGE_A_MEAN_PASS else "FAIL"


def stage_b_decision(delta_e2e: float, seed: int, seed0: Mapping[str, Any] | None = None) -> str:
    if float(delta_e2e) < STAGE_B_FAIL:
        return "FAIL"
    if float(delta_e2e) >= STAGE_B_STRONG_PASS:
        return "PASS"
    if int(seed) == 0:
        return "AMBIGUOUS_NEEDS_SEED1"
    if seed0 is None:
        raise ValueError("seed-1 Stage-B decision requires seed-0 result")
    mean = 0.5 * (float(seed0["delta_e2e_soup"]) + float(delta_e2e))
    return "PASS" if mean >= STAGE_B_MEAN_PASS else "FAIL"


def official_authorized(final_soup: float, base_soup: float = TCCD_V2_SOUP) -> bool:
    return float(final_soup) <= OFFICIAL_SOUP_ROUTE_A or float(base_soup - final_soup) >= OFFICIAL_SOUP_ROUTE_B


def official_records(data_root: Path):
    train_records, train_meta = V2.load_train_records()
    train_mols, _ = T.load_mols(data_root, "train")
    valid_mols, valid_y = T.load_mols(data_root, "valid")
    layout = V2.frozen_layout()
    atom_index, bond_index = T.category_catalog(train_mols)
    if len(atom_index) != layout.n_atom or len(bond_index) != layout.n_bond:
        raise RuntimeError("frozen TCCD layout category mismatch")
    valid_records, cached = load_or_build_records("valid", valid_mols, layout, atom_index, bond_index, valid_y, len(valid_mols))
    meta = {
        "train_meta": train_meta,
        "valid_records_cached": bool(cached),
        "n_train": len(train_records),
        "n_valid": len(valid_records),
        "official_test_loaded": False,
    }
    return list(train_records), list(valid_records), meta
