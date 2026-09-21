#!/usr/bin/env python
"""TCCD-v2 shared library: task-learned local prototype vocabulary.

This round replaces reconstruction/sparse-coding dictionaries with a shared
soft prototype vocabulary after the matched TCCD-v1 714 -> 64 local encoder.
No K-SVD, OMP, IHT, or reconstruction loss is used here.
"""

from __future__ import annotations

import hashlib
import math
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from tracks.ksvd.code import tccd_v0 as T
from tracks.ksvd.code import tccd_v1 as V

REPO_ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_VERSION = "tccd_v2"
RESULTS_DIR = REPO_ROOT / "tracks/ksvd/results/tccd_v2"
CACHE_DIR = RESULTS_DIR / "cache"

D_LOCAL = 64
K_PROTO = 64
N_REL = V.N_REL
TEMP_MIN = 0.05
TEMP_SPAN = 0.95
TEMP_INIT = 0.20
EPS = 1e-8
PROTO_INIT_SEED = T.DICT_SEED
DEAD_THRESHOLD = 1e-6
BALANCE_FLOOR = 1.0 / K_PROTO

DENSE_REFERENCE_BEST = 0.4039327800273895
DENSE_REFERENCE_SOUP = 0.3874090611934662
CANONICAL_GPU1_BASELINE = 0.119818

GATE_COMP_MIN = 0.010
GATE_PROTO_PASS = 0.010
GATE_PROTO_FAIL = 0.030
GATE_PROTO_AMBIG_MEAN_PASS = 0.015
ABSOLUTE_GATE = 0.30


def _torch():
    return T._torch()


def initial_temperature_logit() -> float:
    q = (TEMP_INIT - TEMP_MIN) / TEMP_SPAN
    return float(math.log(q / (1.0 - q)))


def normalized_gaussian(d: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    P = rng.standard_normal((int(d), int(k))).astype(np.float32)
    P /= np.maximum(np.linalg.norm(P, axis=0, keepdims=True), 1e-12)
    return P


def prototype_fingerprint(P: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(P, dtype=np.float32)).tobytes()).hexdigest()


def load_train_records(path: Path | None = None):
    return V.load_train_records(path)


def frozen_layout() -> T.PatchLayout:
    return V.frozen_layout()


def load_frozen_encoder_init() -> np.ndarray:
    D, _ = V.load_frozen_dictionary()
    D = np.asarray(D, dtype=np.float32)
    if D.shape != (D.shape[0], D_LOCAL):
        raise ValueError(f"expected 64-wide TCCD-v1 dense init, got {D.shape}")
    return D


def relation_operator_list(rec: Mapping[str, Any]) -> list[np.ndarray]:
    return V.rel_operator_list(rec)


def make_batch(records: Sequence[Mapping[str, Any]], indices: Sequence[int], device: str):
    return V.make_padded_batch(records, indices, device)


def compose_padded(C, R_pad, valid, iu0, iu1):
    return V.compose_padded(C, R_pad, valid, iu0, iu1)


def bag_padded(C, valid):
    torch = _torch()
    C = C * valid.unsqueeze(-1).to(C.dtype)
    return C.sum(dim=1)


def shuffle_assignments(C, records, indices, seed: int, device: str):
    torch = _torch()
    B, N, _ = C.shape
    out = C.clone()
    for bi, gi in enumerate(indices):
        n = int(records[gi]["n"])
        perm = V.shuffle_perm(n, int(seed), int(gi))
        p = torch.as_tensor(perm, dtype=torch.long, device=device)
        out[bi, :n] = C[bi, :n].index_select(0, p)
    return out


def local_entropy(C_flat, valid_flat):
    torch = _torch()
    c = C_flat[valid_flat]
    return -(c * torch.log(c + EPS)).sum(dim=1).mean()


def balance_kl(C_flat, valid_flat):
    torch = _torch()
    c = C_flat[valid_flat].mean(dim=0)
    return (c * torch.log(c * K_PROTO + EPS)).sum()


def usage_metrics(C_flat: np.ndarray, molecule_ids: np.ndarray | None = None) -> dict[str, Any]:
    C = np.asarray(C_flat, dtype=np.float64)
    cbar = C.mean(axis=0)
    arg = C.argmax(axis=1)
    order = np.argsort(cbar)[::-1]
    local_h = -(C * np.log(C + EPS)).sum(axis=1)
    global_h = float(-(cbar * np.log(cbar + EPS)).sum())
    metrics: dict[str, Any] = {
        "mean_usage": cbar.tolist(),
        "active_prototypes": int(np.sum(cbar >= DEAD_THRESHOLD)),
        "dead_prototypes": int(np.sum(cbar < DEAD_THRESHOLD)),
        "effective_prototype_count": float(1.0 / np.sum(cbar * cbar + EPS)),
        "top1_usage_mass": float(cbar[order[0]]),
        "top8_usage_mass": float(cbar[order[:8]].sum()),
        "mean_local_assignment_entropy": float(local_h.mean()),
        "global_usage_entropy": global_h,
        "global_usage_entropy_normalized": float(global_h / math.log(K_PROTO)),
        "top_prototypes": order[:8].astype(int).tolist(),
        "top1_counts": np.bincount(arg, minlength=K_PROTO).astype(int).tolist(),
    }
    if molecule_ids is not None:
        mids = np.asarray(molecule_ids, dtype=np.int64)
        coverage = []
        for k in range(K_PROTO):
            coverage.append(int(np.unique(mids[arg == k]).size) if np.any(arg == k) else 0)
        metrics["molecule_coverage"] = coverage
    return metrics


def semantic_coherence(C_flat: np.ndarray, records, dev_indices, top_n: int = 50) -> dict[str, Any]:
    keys_flat: list[bytes] = []
    for gi in dev_indices:
        keys_flat.extend(records[int(gi)]["keys"])
    keys_int: dict[bytes, int] = {}
    key_ids = np.empty(len(keys_flat), dtype=np.int64)
    for i, key in enumerate(keys_flat):
        if key not in keys_int:
            keys_int[key] = len(keys_int)
        key_ids[i] = keys_int[key]
    out: dict[str, Any] = {}
    C = np.asarray(C_flat, dtype=np.float64)
    for k in range(K_PROTO):
        take = min(int(top_n), C.shape[0])
        idx = np.argsort(C[:, k])[::-1][:take]
        modal = float(np.bincount(key_ids[idx]).max() / max(take, 1))
        rng = np.random.default_rng(k + 991)
        ridx = rng.choice(C.shape[0], size=take, replace=False)
        base = float(np.bincount(key_ids[ridx]).max() / max(take, 1))
        out[str(k)] = {
            "top50_key_concentration": modal,
            "random_baseline": base,
            "top_patch_indices": idx.astype(int).tolist(),
        }
    vals = np.asarray([out[str(k)]["top50_key_concentration"] for k in range(K_PROTO)])
    bases = np.asarray([out[str(k)]["random_baseline"] for k in range(K_PROTO)])
    out["_summary"] = {
        "mean_concentration": float(vals.mean()),
        "mean_baseline": float(bases.mean()),
        "active_mean_concentration": float(vals[vals > 0].mean()) if np.any(vals > 0) else 0.0,
    }
    return out


@dataclass
class PrototypeTrainResult:
    best_valid: float
    best_epoch: int
    soup_valid: float | None
    soup_members: list[int]
    train_history: list[dict[str, Any]] = field(default_factory=list)
    state_best: Any = None
    state_soup: Any = None
    regularization: dict[str, Any] = field(default_factory=dict)
    wall_s: float = 0.0
    peak_mem_mb: float | None = None


class PrototypeModelFactory:
    @staticmethod
    def build(
        F: int,
        d: int,
        K: int,
        n_rel: int,
        encoder_init: np.ndarray,
        *,
        mode: str = "rel",
        proto_seed: int = PROTO_INIT_SEED,
        seed: int = 0,
    ):
        torch = _torch()
        nn = torch.nn
        if mode not in {"bag", "rel"}:
            raise ValueError(mode)
        H = int(K if mode == "bag" else T.h_dim(K, n_rel))
        torch.manual_seed(int(seed) + 17001)

        class _M(nn.Module):
            def __init__(self):
                super().__init__()
                W = np.asarray(encoder_init, dtype=np.float32)
                if W.shape != (F, d):
                    raise ValueError(f"encoder init shape {W.shape} != {(F, d)}")
                P = normalized_gaussian(d, K, int(proto_seed))
                self.W = nn.Parameter(torch.as_tensor(W.copy()))
                self.P = nn.Parameter(torch.as_tensor(P.copy()))
                self.temp_logit = nn.Parameter(torch.tensor(initial_temperature_logit(), dtype=torch.float32))
                self.head = nn.Linear(H, 1)
                self.mode = mode
                self.K = int(K)
                self.d = int(d)
                self.F = int(F)
                self.n_rel = int(n_rel)
                self.graph_feature_kind = "bag_assignments_only" if mode == "bag" else "assignment_composition_only"
                self.uses_latent_bypass = False
                self.proto_seed = int(proto_seed)

            def encode(self, X):
                return X @ self.W

            def temperature(self):
                return TEMP_MIN + TEMP_SPAN * torch.sigmoid(self.temp_logit)

            def assign(self, X):
                torch = _torch()
                Z = self.encode(X)
                Zbar = torch.nn.functional.normalize(Z, dim=-1, eps=EPS)
                Pbar = torch.nn.functional.normalize(self.P, dim=0, eps=EPS)
                logits = (Zbar @ Pbar) / self.temperature()
                return torch.softmax(logits, dim=-1)

            def graph_features(self, C, batch, *, shuffle=False, indices=None, records=None, seed=0):
                C3 = C.reshape(batch["X_pad"].shape[0], batch["X_pad"].shape[1], self.K)
                if shuffle:
                    if indices is None or records is None:
                        raise ValueError("shuffle requires indices and records")
                    C3 = shuffle_assignments(C3, records, indices, seed, C.device)
                if self.mode == "bag":
                    return bag_padded(C3, batch["valid"])
                return compose_padded(C3, batch["R_pad"], batch["valid"], batch["iu0"], batch["iu1"])

            def forward_padded(self, batch, *, shuffle=False, indices=None, records=None, seed=0):
                X_flat = batch["X_pad"].reshape(-1, self.F)
                C_flat = self.assign(X_flat)
                h = self.graph_features(C_flat, batch, shuffle=shuffle, indices=indices, records=records, seed=seed)
                pred = self.head(h).reshape(-1)
                return pred, C_flat, h

            def renormalize_(self):
                with _torch().no_grad():
                    for p in (self.W, self.P):
                        p.copy_(p / p.norm(dim=0, keepdim=True).clamp_min(1e-6))

        return _M()


def evaluate_mae(model, records, indices, device, *, batch: int = 64, shuffle: bool = False, seed: int = 0) -> float:
    torch = _torch()
    model.eval()
    errs = []
    with torch.no_grad():
        for start in range(0, len(indices), batch):
            chunk = list(indices[start : start + batch])
            b = make_batch(records, chunk, device)
            pred, _, _ = model.forward_padded(b, shuffle=shuffle, indices=chunk, records=records, seed=seed)
            errs.append((pred - b["y"]).abs().cpu().numpy())
    return float(np.concatenate(errs).mean())


def collect_assignments(model, records, indices, device, *, batch: int = 64):
    torch = _torch()
    model.eval()
    chunks = []
    mol_ids = []
    with torch.no_grad():
        for start in range(0, len(indices), batch):
            chunk = list(indices[start : start + batch])
            b = make_batch(records, chunk, device)
            _, C_flat, _ = model.forward_padded(b)
            mask = b["valid"].reshape(-1)
            chunks.append(C_flat[mask].cpu().numpy())
            for gi in chunk:
                mol_ids.extend([int(gi)] * int(records[gi]["n"]))
    return np.concatenate(chunks, axis=0), np.asarray(mol_ids, dtype=np.int64)


def _initial_regularization(model, records, train_indices, device, *, batch: int, log=print):
    torch = _torch()
    cal_idx = list(train_indices[: min(512, len(train_indices))])
    b = make_batch(records, cal_idx, device)
    with torch.no_grad():
        pred, C_flat, _ = model.forward_padded(b)
        valid = b["valid"].reshape(-1)
        task = float((pred - b["y"]).abs().mean())
        local = float(local_entropy(C_flat, valid))
        balance = float(balance_kl(C_flat, valid))
    lam_local = 0.05 * task / max(local, EPS)
    lam_balance = 0.05 * task / max(balance, BALANCE_FLOOR)
    out = {
        "calibration_graphs": len(cal_idx),
        "initial_task": task,
        "initial_local_entropy": local,
        "initial_balance_kl": balance,
        "lambda_local": float(lam_local),
        "lambda_balance": float(lam_balance),
        "initial_local_contribution": float(lam_local * local),
        "initial_balance_contribution": float(lam_balance * balance),
        "balance_floor": BALANCE_FLOOR,
    }
    log("  [regularization] " + str({k: round(v, 6) for k, v in out.items() if isinstance(v, (int, float))}))
    return out


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
        for start in range(0, len(order), batch):
            sel = [train_indices[i] for i in order[start : start + batch]]
            b = make_batch(records_train, sel, device)
            opt.zero_grad(set_to_none=True)
            pred, C_flat, _ = model.forward_padded(b)
            valid = b["valid"].reshape(-1)
            task = (pred - b["y"]).abs().mean()
            local = local_entropy(C_flat, valid)
            balance = balance_kl(C_flat, valid)
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
            "epoch": epoch,
            "train_loss": ep_total / max(ep_n, 1),
            "train_task": ep_task / max(ep_n, 1),
            "train_local_entropy": ep_local / max(ep_n, 1),
            "train_balance_kl": ep_balance / max(ep_n, 1),
            "valid": dev_mae,
            "temperature": float(model.temperature().detach()),
        }
        history.append(row)
        log(f"  epoch={epoch:03d} train={row['train_loss']:.6f} valid={dev_mae:.6f} tau={row['temperature']:.5f} best@{best_epoch}")
        if dev_mae < best - 1e-9:
            best = dev_mae
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if len(top) < T.TOP_K_SOUP or dev_mae < max(t[0] for t in top):
            top.append((dev_mae, epoch, {k: v.detach().clone() for k, v in model.state_dict().items()}))
            top.sort(key=lambda t: t[0])
            top = top[: T.TOP_K_SOUP]
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
    return PrototypeTrainResult(
        best_valid=float(best), best_epoch=int(best_epoch), soup_valid=None if soup_mae is None else float(soup_mae),
        soup_members=members, train_history=history, state_best=best_state, state_soup=soup,
        regularization=reg, wall_s=time.time() - t0,
    )


def gate_a_verdict(delta_comp: float, delta_proto: float, seed: int, *, seed0_delta_proto: float | None = None) -> str:
    if float(delta_comp) < GATE_COMP_MIN:
        return "FAIL_COMPOSITION"
    if int(seed) == 0:
        if float(delta_proto) <= GATE_PROTO_PASS:
            return "PASS"
        if float(delta_proto) > GATE_PROTO_FAIL:
            return "FAIL_PROTO_BOTTLENECK"
        return "AMBIGUOUS_NEEDS_SEED1"
    if seed0_delta_proto is None:
        raise ValueError("seed-1 verdict requires seed-0 prototype gap")
    mean = 0.5 * (float(seed0_delta_proto) + float(delta_proto))
    return "PASS" if mean <= GATE_PROTO_AMBIG_MEAN_PASS else "FAIL_PROTO_BOTTLENECK"


def save_state(path: Path, state: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch = _torch()
    torch.save(state, path)


def load_state(path: Path, device: str):
    torch = _torch()
    return torch.load(path, map_location=device, weights_only=False)
